"""Deterministic, leakage-free object-set fusion for frozen MoME experts.

Stage019-S4 operates on decoded detections.  Same-class boxes are aligned with
a deterministic two-metre Hungarian match.  GACE-Lite calibrates score logits;
CP-AFR selects bounded centre, log-size, yaw and velocity attributes from the
anchor plus the three full-context experts.  No GT, corruption identity or
fault mask is accepted by the inference interface.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


ROUTE_ORDER = ("fused", "lidar", "camera")
FUSION_MODES = ("score_only", "attribute_only", "stacked")
FEATURE_VERSION = "stage019_s4_deployment_visible_object_features_v1"
CHECKPOINT_STATE_DICT_KEY = "object_set_attribute_fusion_state_dict"
OBJECT_EVIDENCE_KEYS = (
    "object_distance",
    "box_point_count",
    "neighborhood_point_count",
    "visible_camera_views",
)
FEATURE_NAMES = (
    "anchor_score_logit",
    "ego_center_distance_m",
    "log1p_points_inside_anchor",
    "log1p_points_within_2m",
    "camera_visible_view_fraction",
) + tuple(
    f"{role}_{name}"
    for role in ROUTE_ORDER
    for name in (
        "matched",
        "score_logit_delta",
        "dx_m",
        "dy_m",
        "dz_m",
        "dlog_w",
        "dlog_l",
        "dlog_h",
        "dyaw_rad",
        "dvx_mps",
        "dvy_mps",
        "bev_center_distance_m",
    )
)
_REQUIRED_PREDICTION_KEYS = ("boxes_3d", "scores_3d", "labels_3d")
_FORBIDDEN_INFERENCE_TOKENS = (
    "gt",
    "fault",
    "mask",
    "corruption",
    "condition",
)
_INTEGER_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


def _box_tensor(prediction: Mapping[str, object]) -> torch.Tensor:
    boxes = prediction["boxes_3d"]
    tensor = getattr(boxes, "tensor", boxes)
    if not torch.is_tensor(tensor):
        raise TypeError("boxes_3d must be a tensor or expose .tensor")
    return tensor


def _new_boxes_like(boxes: object, tensor: torch.Tensor) -> object:
    if torch.is_tensor(boxes):
        return tensor
    new_box = getattr(boxes, "new_box", None)
    if new_box is None:
        raise TypeError("boxes_3d container must expose new_box")
    return new_box(tensor)


def _validate_prediction(
    prediction: Mapping[str, object],
    label: str,
    max_detections: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(prediction, Mapping):
        raise TypeError(f"{label} must be a mapping")
    leaked = [
        str(key)
        for key in prediction
        if any(
            token in str(key).lower() for token in _FORBIDDEN_INFERENCE_TOKENS
        )
    ]
    if leaked:
        raise ValueError(f"{label} contains forbidden inference keys: {leaked}")
    missing = [key for key in _REQUIRED_PREDICTION_KEYS if key not in prediction]
    if missing:
        raise ValueError(f"{label} lacks {missing}")
    boxes = _box_tensor(prediction)
    scores = prediction["scores_3d"]
    labels = prediction["labels_3d"]
    if not torch.is_tensor(scores) or not torch.is_tensor(labels):
        raise TypeError(f"{label} scores and labels must be tensors")
    if boxes.ndim != 2 or boxes.shape[1] != 9:
        raise ValueError(f"{label} boxes must have shape [M,9]")
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError(f"{label} scores and labels must have shape [M]")
    if not (boxes.shape[0] == scores.numel() == labels.numel()):
        raise ValueError(f"{label} field lengths differ")
    if boxes.shape[0] > int(max_detections):
        raise ValueError(f"{label} exceeds max_detections")
    if labels.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{label} labels must use an integer dtype")
    if boxes.device != scores.device or boxes.device != labels.device:
        raise ValueError(f"{label} fields must share one device")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError(f"{label} contains non-finite values")
    if torch.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"{label} scores must lie in [0,1]")
    if torch.any(boxes[:, 3:6] <= 0.0):
        raise ValueError(f"{label} box dimensions must be positive")
    if torch.any(labels < 0):
        raise ValueError(f"{label} labels must be non-negative")
    return boxes, scores, labels.long()


def deterministic_class_hungarian_matches(
    anchor_boxes: torch.Tensor,
    anchor_labels: torch.Tensor,
    source_boxes: torch.Tensor,
    source_labels: torch.Tensor,
    route_index: int,
    max_distance: float = 2.0,
) -> torch.Tensor:
    """Return one source index per anchor, or ``-1`` when no valid match.

    The primary cost is BEV centre distance.  Exact ties are resolved by the
    fixed route order, then source index, then anchor index.  The route term is
    constant within one solve but records the cross-route ordering contract.
    """

    if anchor_boxes.ndim != 2 or anchor_boxes.shape[1] != 9:
        raise ValueError("anchor_boxes must have shape [M,9]")
    if source_boxes.ndim != 2 or source_boxes.shape[1] != 9:
        raise ValueError("source_boxes must have shape [K,9]")
    if anchor_labels.shape != (anchor_boxes.shape[0],):
        raise ValueError("anchor_labels must have shape [M]")
    if source_labels.shape != (source_boxes.shape[0],):
        raise ValueError("source_labels must have shape [K]")
    if int(route_index) not in (0, 1, 2):
        raise ValueError("route_index must be 0, 1, or 2")
    if not math.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("max_distance must be finite and positive")

    output = torch.full(
        (anchor_boxes.shape[0],),
        -1,
        device=anchor_labels.device,
        dtype=torch.long,
    )
    if anchor_boxes.shape[0] == 0 or source_boxes.shape[0] == 0:
        return output
    distance = torch.linalg.vector_norm(
        anchor_boxes[:, None, :2].detach().to(device="cpu", dtype=torch.float64)
        - source_boxes[None, :, :2].detach().to(
            device="cpu", dtype=torch.float64
        ),
        dim=2,
    ).numpy()
    valid = (
        anchor_labels[:, None].detach().cpu().numpy()
        == source_labels[None, :].detach().cpu().numpy()
    ) & (distance <= float(max_distance))
    source_tie = np.arange(source_boxes.shape[0], dtype=np.float64)[None, :] * 1e-10
    anchor_tie = np.arange(anchor_boxes.shape[0], dtype=np.float64)[:, None] * 1e-12
    route_tie = float(route_index) * 1e-11
    cost = np.where(
        valid,
        distance + route_tie + source_tie + anchor_tie,
        1e9,
    )
    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows.tolist(), columns.tolist()):
        if valid[row, column]:
            output[row] = int(column)
    return output


def _wrapped_angle_delta(source: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(source - anchor), torch.cos(source - anchor))


def _bounded_vector(delta: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(
        delta.new_tensor(maximum_norm) / norms.clamp_min(1e-12), max=1.0
    )
    return delta * scale


def _validate_object_evidence(
    evidence: Mapping[str, object],
    anchor_boxes: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if not isinstance(evidence, Mapping) or set(evidence) != set(OBJECT_EVIDENCE_KEYS):
        raise ValueError(
            "object_evidence must contain only object_distance, box_point_count, "
            "neighborhood_point_count and visible_camera_views"
        )
    output = {}
    count = anchor_boxes.shape[0]
    for key in OBJECT_EVIDENCE_KEYS:
        value = evidence[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(device=anchor_boxes.device, dtype=anchor_boxes.dtype)
        if value.shape != (count,) or not torch.isfinite(value).all():
            raise ValueError(f"object_evidence.{key} must be finite [M]")
        if torch.any(value < 0.0):
            raise ValueError(f"object_evidence.{key} must be non-negative")
        output[key] = value
    expected_distance = torch.linalg.vector_norm(anchor_boxes[:, :2], dim=1)
    if not torch.allclose(
        output["object_distance"], expected_distance, rtol=1e-5, atol=1e-4
    ):
        raise ValueError("object_distance differs from anchor BEV centre distance")
    if torch.any(output["visible_camera_views"] > 6.0):
        raise ValueError("visible_camera_views must be in [0,6]")
    return output


def build_feature_matrix(
    anchor: Mapping[str, object],
    experts: Mapping[str, Mapping[str, object]],
    object_evidence: Mapping[str, object],
    *,
    max_match_distance: float = 2.0,
    max_detections: int = 300,
) -> Dict[str, object]:
    """Build the canonical 41-D deployment-visible feature matrix.

    Missing expert matches contribute an all-zero 12-column block.  Returned
    aligned boxes/scores always use anchor order and can be consumed directly
    by CP-AFR.  This function is shared by cache generation, training and
    inference to prevent feature-schema drift.
    """

    anchor_boxes, anchor_scores, anchor_labels = _validate_prediction(
        anchor, "anchor", max_detections
    )
    if tuple(experts.keys()) != ROUTE_ORDER:
        raise ValueError("experts must preserve fused/lidar/camera insertion order")
    expert_tensors = {
        role: _validate_prediction(experts[role], f"expert.{role}", max_detections)
        for role in ROUTE_ORDER
    }
    for role, (boxes, scores, labels) in expert_tensors.items():
        if (
            boxes.device != anchor_boxes.device
            or boxes.dtype != anchor_boxes.dtype
            or scores.device != anchor_scores.device
            or scores.dtype != anchor_scores.dtype
            or labels.device != anchor_labels.device
        ):
            raise ValueError(f"expert.{role} dtype/device differs from anchor")
    evidence = _validate_object_evidence(object_evidence, anchor_boxes)

    count = anchor_boxes.shape[0]
    eps = anchor_scores.new_tensor(1e-6)
    anchor_clipped = anchor_scores.clamp(eps, 1.0 - eps)
    anchor_logit = torch.log(anchor_clipped) - torch.log1p(-anchor_clipped)
    columns = [
        anchor_logit,
        evidence["object_distance"],
        torch.log1p(evidence["box_point_count"]),
        torch.log1p(evidence["neighborhood_point_count"]),
        evidence["visible_camera_views"] / 6.0,
    ]
    aligned_boxes = anchor_boxes[:, None, :].expand(-1, 3, -1).clone()
    aligned_scores = anchor_scores[:, None].expand(-1, 3).clone()
    match_mask = torch.zeros((count, 3), device=anchor_boxes.device, dtype=torch.bool)
    source_indices = torch.full(
        (count, 3), -1, device=anchor_boxes.device, dtype=torch.long
    )
    for route_index, role in enumerate(ROUTE_ORDER):
        source_boxes, source_scores, source_labels = expert_tensors[role]
        matched = deterministic_class_hungarian_matches(
            anchor_boxes,
            anchor_labels,
            source_boxes,
            source_labels,
            route_index,
            max_distance=max_match_distance,
        )
        source_indices[:, route_index] = matched
        mask = matched >= 0
        match_mask[:, route_index] = mask
        if torch.any(mask):
            selected = matched[mask]
            aligned_boxes[mask, route_index] = source_boxes.index_select(0, selected)
            aligned_scores[mask, route_index] = source_scores.index_select(0, selected)
        delta = aligned_boxes[:, route_index] - anchor_boxes
        dim_delta = (
            aligned_boxes[:, route_index, 3:6].log()
            - anchor_boxes[:, 3:6].log()
        )
        yaw_delta = _wrapped_angle_delta(
            aligned_boxes[:, route_index, 6], anchor_boxes[:, 6]
        )
        source_clipped = aligned_scores[:, route_index].clamp(eps, 1.0 - eps)
        score_logit = torch.log(source_clipped) - torch.log1p(-source_clipped)
        block = (
            mask.to(dtype=anchor_boxes.dtype),
            score_logit - anchor_logit,
            delta[:, 0],
            delta[:, 1],
            delta[:, 2],
            dim_delta[:, 0],
            dim_delta[:, 1],
            dim_delta[:, 2],
            yaw_delta,
            delta[:, 7],
            delta[:, 8],
            torch.linalg.vector_norm(delta[:, :2], dim=1),
        )
        columns.extend(
            torch.where(mask, value, torch.zeros_like(value)) for value in block
        )
    feature_matrix = torch.stack(columns, dim=1)
    if feature_matrix.shape != (count, len(FEATURE_NAMES)):
        raise RuntimeError("Stage019-S4 feature matrix width drifted")
    if not torch.isfinite(feature_matrix).all():
        raise ValueError("Stage019-S4 feature matrix contains non-finite values")
    return {
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "feature_matrix": feature_matrix,
        "anchor_boxes": anchor_boxes,
        "anchor_scores": anchor_scores,
        "anchor_labels": anchor_labels,
        "aligned_expert_boxes": aligned_boxes,
        "aligned_expert_scores": aligned_scores,
        "expert_match_mask": match_mask,
        "expert_source_indices": source_indices,
        "expert_tensors": expert_tensors,
    }


class ObjectSetAttributeFusion(nn.Module):
    """GACE-Lite and CP-AFR over one decoded frame in anchor order."""

    feature_dim = len(FEATURE_NAMES)

    def __init__(
        self,
        hidden_dim: int = 64,
        max_match_distance: float = 2.0,
        max_detections: int = 300,
        max_rescue_boxes: int = 20,
        max_center_delta: float = 1.0,
        max_dim_log_delta: float = 0.20,
        max_yaw_delta: float = math.pi / 9.0,
        max_velocity_delta: float = 2.0,
        max_score_logit_residual: float = 1.0,
        max_abs_log_temperature: float = math.log(2.0),
        rescue_enabled: bool = False,
    ) -> None:
        super().__init__()
        if int(hidden_dim) != 64:
            raise ValueError("Stage019-S4 fixes hidden_dim at 64")
        if max_match_distance <= 0.0:
            raise ValueError("max_match_distance must be positive")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")
        if max_rescue_boxes < 0 or max_rescue_boxes > 20:
            raise ValueError("max_rescue_boxes must be in [0,20]")
        expected_limits = (0.20, math.pi / 9.0)
        if not math.isclose(max_dim_log_delta, expected_limits[0]):
            raise ValueError("Stage019-S4 fixes max_dim_log_delta at 0.20")
        if not math.isclose(max_yaw_delta, expected_limits[1]):
            raise ValueError("Stage019-S4 fixes max_yaw_delta at pi/9")

        self.max_match_distance = float(max_match_distance)
        self.max_detections = int(max_detections)
        self.max_rescue_boxes = int(max_rescue_boxes)
        self.max_center_delta = float(max_center_delta)
        self.max_dim_log_delta = float(max_dim_log_delta)
        self.max_yaw_delta = float(max_yaw_delta)
        self.max_velocity_delta = float(max_velocity_delta)
        self.max_score_logit_residual = float(max_score_logit_residual)
        self.max_abs_log_temperature = float(max_abs_log_temperature)
        self.rescue_enabled = bool(rescue_enabled)

        self.gace_lite = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # 4 attributes x (anchor + three experts), plus one accept logit.
        self.cp_afr = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 17),
        )
        self.log_temperature = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.gace_lite[-1].weight)
        nn.init.zeros_(self.gace_lite[-1].bias)
        nn.init.zeros_(self.cp_afr[-1].weight)
        nn.init.zeros_(self.cp_afr[-1].bias)
        if self.trainable_parameter_count() > 50000:
            raise RuntimeError("ObjectSetAttributeFusion exceeds 50k parameters")

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def _identity_audit(
        status: str,
        mode: str,
        hard_bypass: bool = False,
        reason: Optional[str] = None,
    ) -> Dict[str, object]:
        output = {
            "status": status,
            "mode": mode,
            "hard_bypass": bool(hard_bypass),
            "route_order": ROUTE_ORDER,
            "matched_count": 0,
            "accepted_count": 0,
            "rescued_count": 0,
        }
        if reason is not None:
            output["reason"] = reason
        return output

    def _apply_score_head(
        self,
        feature_matrix: torch.Tensor,
        anchor_scores: torch.Tensor,
        has_match: torch.Tensor,
    ) -> torch.Tensor:
        raw_residual = self.gace_lite(feature_matrix).squeeze(-1)
        residual = self.max_score_logit_residual * torch.tanh(raw_residual)
        temperature = torch.exp(
            self.log_temperature.clamp(
                -self.max_abs_log_temperature,
                self.max_abs_log_temperature,
            )
        )
        eps = anchor_scores.new_tensor(1e-6)
        clipped = anchor_scores.clamp(eps, 1.0 - eps)
        anchor_logit = torch.log(clipped) - torch.log1p(-clipped)
        calibrated_logit = anchor_logit / temperature + residual
        # Difference form is exactly identity at zero residual and T=1.
        candidate = anchor_scores + (
            torch.sigmoid(calibrated_logit) - torch.sigmoid(anchor_logit)
        )
        return torch.where(has_match, candidate.clamp(0.0, 1.0), anchor_scores)

    def _apply_attribute_head(
        self,
        feature_matrix: torch.Tensor,
        anchor_boxes: torch.Tensor,
        aligned_boxes: torch.Tensor,
        match_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cp_output = self.cp_afr(feature_matrix)
        attribute_logits = cp_output[:, :16].reshape(-1, 4, 4)
        accept_logit = cp_output[:, 16]
        source_valid = torch.cat(
            (
                torch.ones(
                    (match_mask.shape[0], 1),
                    device=match_mask.device,
                    dtype=torch.bool,
                ),
                match_mask,
            ),
            dim=1,
        )
        masked_logits = attribute_logits.masked_fill(
            ~source_valid[:, None, :], -torch.inf
        )
        weights = torch.softmax(masked_logits, dim=-1)
        has_match = match_mask.any(dim=1)
        hard_accept = (accept_logit > 0.0) & has_match
        if self.training:
            # Forward is a strict 0/1 accept decision; backward follows a
            # centred smooth surrogate.  At zero initialization this is exact
            # identity while the accept head still receives gradient.
            soft_accept = 2.0 * torch.sigmoid(accept_logit) - 1.0
            accept = (
                hard_accept.to(feature_matrix.dtype)
                + soft_accept
                - soft_accept.detach()
            )
        else:
            accept = hard_accept.to(feature_matrix.dtype)

        sources = torch.cat((anchor_boxes[:, None, :], aligned_boxes), dim=1)
        output = anchor_boxes.clone()

        centre_target = (
            weights[:, 0, :, None] * sources[:, :, :3]
        ).sum(dim=1)
        centre_delta = _bounded_vector(
            centre_target - anchor_boxes[:, :3], self.max_center_delta
        )
        output[:, :3] = anchor_boxes[:, :3] + accept[:, None] * centre_delta

        source_log_dims = sources[:, :, 3:6].log()
        dim_target = (
            weights[:, 1, :, None] * source_log_dims
        ).sum(dim=1)
        anchor_log_dims = anchor_boxes[:, 3:6].log()
        dim_delta = (dim_target - anchor_log_dims).clamp(
            -self.max_dim_log_delta, self.max_dim_log_delta
        )
        changed_dims = torch.exp(anchor_log_dims + accept[:, None] * dim_delta)
        output[:, 3:6] = (
            anchor_boxes[:, 3:6]
            + (changed_dims - torch.exp(anchor_log_dims))
        )

        source_yaw = sources[:, :, 6]
        yaw_sine = (weights[:, 2] * torch.sin(source_yaw)).sum(dim=1)
        yaw_cosine = (weights[:, 2] * torch.cos(source_yaw)).sum(dim=1)
        yaw_target = torch.atan2(yaw_sine, yaw_cosine)
        yaw_delta = _wrapped_angle_delta(yaw_target, anchor_boxes[:, 6]).clamp(
            -self.max_yaw_delta, self.max_yaw_delta
        )
        output[:, 6] = anchor_boxes[:, 6] + accept * yaw_delta

        velocity_target = (
            weights[:, 3, :, None] * sources[:, :, 7:9]
        ).sum(dim=1)
        velocity_delta = _bounded_vector(
            velocity_target - anchor_boxes[:, 7:9], self.max_velocity_delta
        )
        output[:, 7:9] = (
            anchor_boxes[:, 7:9] + accept[:, None] * velocity_delta
        )
        return output, hard_accept

    def forward(
        self,
        anchor: Mapping[str, object],
        experts: Mapping[str, Mapping[str, object]],
        object_evidence: Mapping[str, object],
        *,
        mode: str,
        hard_bypass: bool = False,
        enable_rescue: Optional[bool] = None,
    ) -> Tuple[Mapping[str, object], Dict[str, object]]:
        """Fuse a decoded frame with an explicit G1/G2/stacked mode."""

        if mode not in FUSION_MODES:
            return anchor, self._identity_audit(
                "fail_closed_invalid_mode", str(mode), reason="unsupported mode"
            )
        if bool(hard_bypass):
            return anchor, self._identity_audit(
                "hard_bypass_identity", mode, hard_bypass=True
            )
        rescue = self.rescue_enabled if enable_rescue is None else bool(enable_rescue)
        if rescue:
            return anchor, self._identity_audit(
                "scientifically_disabled_until_union_gate_and_trained_rescue_checkpoint",
                mode,
                reason=(
                    "no preregistered trained unmatched-object rescue head exists"
                ),
            )
        try:
            built = build_feature_matrix(
                anchor,
                experts,
                object_evidence,
                max_match_distance=self.max_match_distance,
                max_detections=self.max_detections,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            return anchor, self._identity_audit(
                "fail_closed_invalid_input", mode, reason=str(error)
            )

        feature_matrix = built["feature_matrix"]
        match_mask = built["expert_match_mask"]
        matched_count = int(match_mask.sum().detach().cpu().item())
        if matched_count == 0:
            return anchor, self._identity_audit("no_match_identity", mode)

        anchor_boxes = built["anchor_boxes"]
        anchor_scores = built["anchor_scores"]
        anchor_labels = built["anchor_labels"]
        output_boxes = anchor_boxes
        output_scores = anchor_scores
        if mode in ("score_only", "stacked"):
            output_scores = self._apply_score_head(
                feature_matrix, anchor_scores, match_mask.any(dim=1)
            )
        if mode in ("attribute_only", "stacked"):
            output_boxes, _ = self._apply_attribute_head(
                feature_matrix,
                anchor_boxes,
                built["aligned_expert_boxes"],
                match_mask,
            )

        output_labels = anchor_labels
        rescued_count = 0
        if (
            output_boxes.shape[0] > self.max_detections
            or rescued_count > self.max_rescue_boxes
            or not torch.isfinite(output_boxes).all()
            or not torch.isfinite(output_scores).all()
            or torch.any((output_scores < 0.0) | (output_scores > 1.0))
        ):
            return anchor, self._identity_audit(
                "fail_closed_nonfinite_output", mode
            )

        common_count = anchor_boxes.shape[0]
        modified = torch.zeros(
            (common_count,), device=anchor_boxes.device, dtype=torch.bool
        )
        if mode in ("score_only", "stacked"):
            modified |= output_scores[:common_count] != anchor_scores
        if mode in ("attribute_only", "stacked"):
            modified |= torch.any(
                output_boxes[:common_count] != anchor_boxes, dim=1
            )
        accepted_count = int(modified.sum().detach().cpu().item())

        output = dict(anchor)
        output["boxes_3d"] = _new_boxes_like(anchor["boxes_3d"], output_boxes)
        output["scores_3d"] = output_scores
        output["labels_3d"] = output_labels.to(dtype=anchor["labels_3d"].dtype)
        audit = {
            "status": "fused" if accepted_count or rescued_count else "identity",
            "mode": mode,
            "hard_bypass": False,
            "route_order": ROUTE_ORDER,
            "feature_version": FEATURE_VERSION,
            "matched_count": matched_count,
            "accepted_count": accepted_count,
            "rescued_count": rescued_count,
            "parameter_count": self.trainable_parameter_count(),
        }
        return output, audit


__all__ = [
    "CHECKPOINT_STATE_DICT_KEY",
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "FUSION_MODES",
    "OBJECT_EVIDENCE_KEYS",
    "ROUTE_ORDER",
    "ObjectSetAttributeFusion",
    "build_feature_matrix",
    "deterministic_class_hungarian_matches",
]
