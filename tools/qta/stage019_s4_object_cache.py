"""Deterministic object matching and leakage-separated S4 cache records."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .stage019_s4_contracts import (
        EXPERT_ROLES,
        MATCH_RADIUS_METERS,
        MAX_OUTPUT_BOXES,
        SOFT_LABEL_THRESHOLDS_METERS,
    )
except ImportError:
    from stage019_s4_contracts import (
        EXPERT_ROLES,
        MATCH_RADIUS_METERS,
        MAX_OUTPUT_BOXES,
        SOFT_LABEL_THRESHOLDS_METERS,
    )


BOX_DIM = 9
ROLE_COUNT = 3
FEATURE_VERSION = "stage019_s4_deployment_visible_object_features_v1"
FEATURE_NAMES = (
    "anchor_score_logit",
    "ego_center_distance_m",
    "log1p_points_inside_anchor",
    "log1p_points_within_2m",
    "camera_visible_view_fraction",
) + tuple(
    f"{role}_{name}"
    for role in EXPERT_ROLES
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

_FORMAL_FUSION_MODULE = None


def _formal_fusion_module():
    """Load the repo-native fusion module without importing plugin registries."""

    global _FORMAL_FUSION_MODULE
    if _FORMAL_FUSION_MODULE is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "projects"
            / "mmdet3d_plugin"
            / "models"
            / "utils"
            / "object_set_attribute_fusion.py"
        )
        spec = importlib.util.spec_from_file_location(
            "stage019_s4_formal_object_set_attribute_fusion", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load formal S4 fusion module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if tuple(module.FEATURE_NAMES) != FEATURE_NAMES:
            raise RuntimeError("tools/formal Stage019-S4 feature schemas differ")
        _FORMAL_FUSION_MODULE = module
    return _FORMAL_FUSION_MODULE


def _as_boxes(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != BOX_DIM:
        raise ValueError(f"{name} must have shape [M,9]")
    if array.shape[0] > MAX_OUTPUT_BOXES:
        raise ValueError(f"{name} exceeds the 300-box output cap")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if array.size and np.any(array[:, 3:6] <= 0.0):
        raise ValueError(f"{name} contains non-positive box dimensions")
    return array


def _as_scores(value: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite [M] array")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must lie in [0,1]")
    return array


def _as_labels(value: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64).reshape(-1)
    if array.shape != (count,) or np.any(array < 0):
        raise ValueError(f"{name} must be a non-negative [M] array")
    return array


def normalize_object_set(value: Mapping[str, Any], name: str) -> dict[str, np.ndarray]:
    boxes = _as_boxes(value["boxes"], f"{name}.boxes")
    return {
        "boxes": boxes,
        "scores": _as_scores(value["scores"], len(boxes), f"{name}.scores"),
        "labels": _as_labels(value["labels"], len(boxes), f"{name}.labels"),
    }


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def deterministic_hungarian_matches(
    anchor_boxes: np.ndarray,
    anchor_labels: np.ndarray,
    source_boxes: np.ndarray,
    source_labels: np.ndarray,
    route_index: int,
    max_distance: float = MATCH_RADIUS_METERS,
) -> np.ndarray:
    """Return source index per anchor, or -1, with deterministic tie breaks."""

    from scipy.optimize import linear_sum_assignment

    anchors = _as_boxes(anchor_boxes, "anchor_boxes")
    sources = _as_boxes(source_boxes, "source_boxes")
    anchor_labels = _as_labels(anchor_labels, len(anchors), "anchor_labels")
    source_labels = _as_labels(source_labels, len(sources), "source_labels")
    if route_index < 0 or route_index >= ROLE_COUNT:
        raise ValueError("route_index must be 0, 1, or 2")
    if not math.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("max_distance must be finite and positive")
    output = np.full((len(anchors),), -1, dtype=np.int64)
    if not len(anchors) or not len(sources):
        return output
    distance = np.linalg.norm(
        anchors[:, None, :2].astype(np.float64)
        - sources[None, :, :2].astype(np.float64),
        axis=2,
    )
    valid = (anchor_labels[:, None] == source_labels[None, :]) & (
        distance <= float(max_distance)
    )
    # Invalid entries are never accepted after assignment.  Epsilons are much
    # smaller than decoded coordinate precision; their sole role is stable ties.
    source_tie = np.arange(len(sources), dtype=np.float64)[None, :] * 1e-10
    anchor_tie = np.arange(len(anchors), dtype=np.float64)[:, None] * 1e-12
    route_tie = float(route_index) * 1e-11
    cost = np.where(valid, distance + route_tie + source_tie + anchor_tie, 1e9)
    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows.tolist(), columns.tolist()):
        if valid[row, column]:
            output[row] = column
    return output


def match_all_experts(
    original: Mapping[str, Any], experts: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    anchor = normalize_object_set(original, "original")
    if tuple(experts) != EXPERT_ROLES:
        raise ValueError("experts must preserve exact fused/lidar/camera insertion order")
    normalized = {
        role: normalize_object_set(experts[role], f"expert.{role}")
        for role in EXPERT_ROLES
    }
    matches = {
        role: deterministic_hungarian_matches(
            anchor["boxes"],
            anchor["labels"],
            normalized[role]["boxes"],
            normalized[role]["labels"],
            route_index,
        )
        for route_index, role in enumerate(EXPERT_ROLES)
    }
    return matches, normalized


def _safe_logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def points_inside_boxes(points_xyz: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    boxes = _as_boxes(boxes, "boxes")
    if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points[:, :3]).all():
        raise ValueError("points_xyz must be finite [P,>=3]")
    counts = np.zeros((len(boxes),), dtype=np.int32)
    for index, box in enumerate(boxes):
        relative = points[:, :3] - box[None, :3]
        cosine = math.cos(float(box[6]))
        sine = math.sin(float(box[6]))
        local_x = cosine * relative[:, 0] + sine * relative[:, 1]
        local_y = -sine * relative[:, 0] + cosine * relative[:, 1]
        inside = (
            (np.abs(local_x) <= box[3] * 0.5)
            & (np.abs(local_y) <= box[4] * 0.5)
            & (np.abs(relative[:, 2]) <= box[5] * 0.5)
        )
        counts[index] = int(np.count_nonzero(inside))
    return counts


def points_within_radius(
    points_xyz: np.ndarray, boxes: np.ndarray, radius: float = MATCH_RADIUS_METERS
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    boxes = _as_boxes(boxes, "boxes")
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points[:, :2]).all():
        raise ValueError("points_xyz must be finite [P,>=2]")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    return np.asarray(
        [
            np.count_nonzero(
                np.linalg.norm(points[:, :2] - box[None, :2], axis=1) <= radius
            )
            for box in boxes
        ],
        dtype=np.int32,
    )


def camera_visible_view_count(
    boxes: np.ndarray,
    lidar2img: Sequence[np.ndarray],
    image_shapes: Sequence[Sequence[int]],
) -> np.ndarray:
    boxes = _as_boxes(boxes, "boxes")
    matrices = [np.asarray(value, dtype=np.float64) for value in lidar2img]
    if len(matrices) != 6 or len(image_shapes) != 6:
        raise ValueError("camera visibility requires exactly six views")
    centers = np.concatenate(
        [boxes[:, :3].astype(np.float64), np.ones((len(boxes), 1))], axis=1
    )
    visible = np.zeros((len(boxes),), dtype=np.int8)
    for matrix, shape in zip(matrices, image_shapes):
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("lidar2img must contain finite 4x4 matrices")
        if len(shape) < 2:
            raise ValueError("image shape lacks height/width")
        projected = centers @ matrix.T
        depth = projected[:, 2]
        safe_depth = np.where(np.abs(depth) > 1e-8, depth, 1.0)
        x = projected[:, 0] / safe_depth
        y = projected[:, 1] / safe_depth
        height, width = int(shape[0]), int(shape[1])
        visible += (
            (depth > 1e-6)
            & (x >= 0.0)
            & (x < width)
            & (y >= 0.0)
            & (y < height)
        ).astype(np.int8)
    return visible


def candidate_soft_quality(
    boxes: np.ndarray,
    labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = _as_boxes(boxes, "boxes")
    labels = _as_labels(labels, len(boxes), "labels")
    gt_boxes = _as_boxes(gt_boxes, "gt_boxes")
    gt_labels = _as_labels(gt_labels, len(gt_boxes), "gt_labels")
    distances = np.full((len(boxes),), np.inf, dtype=np.float32)
    target_gt = np.full((len(boxes),), -1, dtype=np.int64)
    for index, (box, label) in enumerate(zip(boxes, labels)):
        candidates = np.flatnonzero(gt_labels == label)
        if not len(candidates):
            continue
        current = np.linalg.norm(gt_boxes[candidates, :2] - box[None, :2], axis=1)
        order = np.lexsort((candidates, current))
        chosen = int(candidates[order[0]])
        distances[index] = float(current[order[0]])
        target_gt[index] = chosen
    quality = np.mean(
        distances[:, None]
        <= np.asarray(SOFT_LABEL_THRESHOLDS_METERS, dtype=np.float32)[None, :],
        axis=1,
    ).astype(np.float32)
    return quality, target_gt


def one_to_one_soft_quality(
    boxes: np.ndarray,
    labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic maximum-cardinality matches at 0.5/1/2/4 metres."""

    from scipy.optimize import linear_sum_assignment

    boxes = _as_boxes(boxes, "boxes")
    labels = _as_labels(labels, len(boxes), "labels")
    gt_boxes = _as_boxes(gt_boxes, "gt_boxes")
    gt_labels = _as_labels(gt_labels, len(gt_boxes), "gt_labels")
    thresholds = SOFT_LABEL_THRESHOLDS_METERS
    hits = np.zeros((len(boxes), len(thresholds)), dtype=np.bool_)
    widest_target = np.full((len(boxes),), -1, dtype=np.int64)
    for threshold_index, threshold in enumerate(thresholds):
        for label in sorted(set(labels.tolist()) & set(gt_labels.tolist())):
            box_indices = np.flatnonzero(labels == label)
            gt_indices = np.flatnonzero(gt_labels == label)
            distance = np.linalg.norm(
                boxes[box_indices, None, :2].astype(np.float64)
                - gt_boxes[None, gt_indices, :2].astype(np.float64),
                axis=2,
            )
            valid = distance <= threshold
            cost = np.where(
                valid,
                distance
                + box_indices[:, None].astype(np.float64) * 1e-10
                + gt_indices[None, :].astype(np.float64) * 1e-12,
                1e6,
            )
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows.tolist(), columns.tolist()):
                if valid[row, column]:
                    box_index = int(box_indices[row])
                    hits[box_index, threshold_index] = True
                    if threshold_index == len(thresholds) - 1:
                        widest_target[box_index] = int(gt_indices[column])
    return hits.mean(axis=1).astype(np.float32), widest_target


def _attribute_source_targets(
    anchor: dict[str, np.ndarray],
    normalized_experts: Mapping[str, dict[str, np.ndarray]],
    matches: Mapping[str, np.ndarray],
    gt_boxes: np.ndarray,
    target_gt: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(anchor["boxes"])
    center = np.zeros((count,), dtype=np.int8)
    size = np.zeros((count,), dtype=np.int8)
    yaw = np.zeros((count,), dtype=np.int8)
    velocity = np.zeros((count,), dtype=np.int8)
    accept = np.zeros((count,), dtype=np.float32)
    for index in range(count):
        gt_index = int(target_gt[index])
        if gt_index < 0:
            continue
        sources = [anchor["boxes"][index]]
        source_ids = [0]
        for route_index, role in enumerate(EXPERT_ROLES):
            source_index = int(matches[role][index])
            if source_index >= 0:
                sources.append(normalized_experts[role]["boxes"][source_index])
                source_ids.append(route_index + 1)
        candidates = np.asarray(sources, dtype=np.float32)
        gt = gt_boxes[gt_index]
        losses = {
            "center": np.linalg.norm(candidates[:, :3] - gt[None, :3], axis=1),
            "size": np.abs(np.log(candidates[:, 3:6]) - np.log(gt[None, 3:6])).sum(axis=1),
            "yaw": np.abs(wrap_angle(candidates[:, 6] - gt[6])),
            "velocity": np.linalg.norm(candidates[:, 7:9] - gt[None, 7:9], axis=1),
        }
        selected = {}
        for name, values in losses.items():
            order = np.lexsort((np.asarray(source_ids), values))
            selected[name] = int(source_ids[int(order[0])])
        center[index] = selected["center"]
        size[index] = selected["size"]
        yaw[index] = selected["yaw"]
        velocity[index] = selected["velocity"]
        accept[index] = float(any(value != 0 for value in selected.values()))
    return {
        "center_source": center,
        "size_source": size,
        "yaw_source": yaw,
        "velocity_source": velocity,
        "accept_target": accept,
    }


def build_leakage_separated_cache_record(
    original: Mapping[str, Any],
    experts: Mapping[str, Mapping[str, Any]],
    points_xyz: np.ndarray,
    lidar2img: Sequence[np.ndarray],
    image_shapes: Sequence[Sequence[int]],
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
) -> dict[str, Any]:
    """Build one frame record with GT physically outside inference features."""

    anchor = normalize_object_set(original, "original")
    if tuple(experts) != EXPERT_ROLES:
        raise ValueError("experts must preserve exact fused/lidar/camera insertion order")
    normalized = {
        role: normalize_object_set(experts[role], f"expert.{role}")
        for role in EXPERT_ROLES
    }
    gt_boxes = _as_boxes(gt_boxes, "gt_boxes")
    gt_labels = _as_labels(gt_labels, len(gt_boxes), "gt_labels")
    inside = points_inside_boxes(points_xyz, anchor["boxes"])
    neighbors = points_within_radius(points_xyz, anchor["boxes"])
    visible = camera_visible_view_count(anchor["boxes"], lidar2img, image_shapes)
    import torch

    formal = _formal_fusion_module()
    formal_anchor = {
        "boxes_3d": torch.from_numpy(anchor["boxes"]),
        "scores_3d": torch.from_numpy(anchor["scores"]),
        "labels_3d": torch.from_numpy(anchor["labels"]),
    }
    formal_experts = {
        role: {
            "boxes_3d": torch.from_numpy(normalized[role]["boxes"]),
            "scores_3d": torch.from_numpy(normalized[role]["scores"]),
            "labels_3d": torch.from_numpy(normalized[role]["labels"]),
        }
        for role in EXPERT_ROLES
    }
    evidence = {
        "object_distance": torch.from_numpy(
            np.linalg.norm(anchor["boxes"][:, :2], axis=1).astype(np.float32)
        ),
        "box_point_count": torch.from_numpy(inside.astype(np.float32)),
        "neighborhood_point_count": torch.from_numpy(neighbors.astype(np.float32)),
        "visible_camera_views": torch.from_numpy(visible.astype(np.float32)),
    }
    built = formal.build_feature_matrix(formal_anchor, formal_experts, evidence)
    feature_matrix = built["feature_matrix"].detach().cpu().numpy().astype(np.float32)
    aligned_boxes = built["aligned_expert_boxes"].detach().cpu().numpy().astype(np.float32)
    aligned_scores = built["aligned_expert_scores"].detach().cpu().numpy().astype(np.float32)
    match_mask = built["expert_match_mask"].detach().cpu().numpy().astype(np.bool_)
    source_indices = built["expert_source_indices"].detach().cpu().numpy().astype(np.int64)
    matches = {
        role: source_indices[:, route_index]
        for route_index, role in enumerate(EXPERT_ROLES)
    }
    expert_quality = np.zeros((len(anchor["boxes"]), ROLE_COUNT), dtype=np.float32)
    for route_index, role in enumerate(EXPERT_ROLES):
        role_source_indices = matches[role]
        mask = role_source_indices >= 0
        if np.any(mask):
            quality, _ = one_to_one_soft_quality(
                aligned_boxes[mask, route_index],
                anchor["labels"][mask],
                gt_boxes,
                gt_labels,
            )
            expert_quality[mask, route_index] = quality
    if source_indices.shape != (len(anchor["boxes"]), ROLE_COUNT):
        raise AssertionError("S4 expert source-index matrix shape drifted")
    if feature_matrix.shape != (len(anchor["boxes"]), len(FEATURE_NAMES)):
        raise AssertionError("S4 feature matrix width drifted")
    if not np.isfinite(feature_matrix).all():
        raise ValueError("S4 inference feature matrix contains non-finite values")
    anchor_quality, target_gt = one_to_one_soft_quality(
        anchor["boxes"], anchor["labels"], gt_boxes, gt_labels
    )
    attribute_targets = _attribute_source_targets(
        anchor, normalized, matches, gt_boxes, target_gt
    )
    target_gt_boxes = np.zeros_like(anchor["boxes"], dtype=np.float32)
    # CP-AFR supervises matched anchor objects only.  A distant same-class GT
    # is not a valid attribute target; unmatched opportunities belong solely
    # to the separately gated union/rescue branch.
    target_valid = (target_gt >= 0) & (anchor_quality > 0.0)
    if np.any(target_valid):
        target_gt_boxes[target_valid] = gt_boxes[target_gt[target_valid]]
    return {
        "schema": "visfuse3d_stage019_s4_frame_object_cache_v1",
        "inference_features": {
            "feature_version": FEATURE_VERSION,
            "feature_names": FEATURE_NAMES,
            "feature_matrix": feature_matrix,
            "anchor_boxes": anchor["boxes"],
            "anchor_scores": anchor["scores"],
            "anchor_labels": anchor["labels"],
            "aligned_expert_boxes": aligned_boxes,
            "aligned_expert_scores": aligned_scores,
            "expert_match_mask": match_mask,
            "expert_source_indices": source_indices,
            "object_evidence": {
                "object_distance": np.linalg.norm(
                    anchor["boxes"][:, :2], axis=1
                ).astype(np.float32),
                "box_point_count": inside,
                "neighborhood_point_count": neighbors,
                "visible_camera_views": visible,
            },
        },
        "offline_targets": {
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "anchor_soft_quality": anchor_quality,
            "expert_soft_quality": expert_quality,
            "target_gt_index": target_gt,
            "target_gt_boxes": target_gt_boxes,
            "target_gt_valid": target_valid,
            **attribute_targets,
        },
    }


def assert_no_leakage(record: Mapping[str, Any]) -> None:
    if set(record) != {"schema", "inference_features", "offline_targets"}:
        raise ValueError("frame cache top-level schema drifted")
    inference = record["inference_features"]
    forbidden = (
        "gt",
        "condition",
        "fault_id",
        "mask_id",
        "corruption_parameter",
        "sidecar",
    )
    hits = [key for key in inference if any(marker in key.lower() for marker in forbidden)]
    if hits:
        raise ValueError(f"inference feature namespace leaks offline metadata: {hits}")
    matrix = np.asarray(inference["feature_matrix"])
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("inference feature matrix schema drifted")
    if not np.isfinite(matrix).all():
        raise ValueError("inference feature matrix contains non-finite values")
