"""Deterministic training20 adapters for the seven Stage019-S2 conditions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES


CONDITIONS = (
    "clean",
    "beam_reduction_4",
    "lidar_zero",
    "limited_fov_original_code_60",
    "lidar_object_failure",
    "camera_zero",
    "camera_mud_mask",
)
S3_ACTIONABLE_CONDITIONS = (
    "clean",
    "beam_reduction_4",
    "limited_fov_original_code_60",
    "lidar_object_failure",
    "camera_mud_mask",
)
S3_HARD_BYPASS_CONDITIONS = (
    "lidar_zero",
    "camera_zero",
)
OBJECT_FLAG_SALT = "mome-stage019-s2-training20-object-failure-v1"
MUD_MASK_SALT = "mome-stage019-s2-training20-mud-mask-v1"


def training_object_failure_flag(sample_token: str) -> bool:
    digest = hashlib.sha256(
        f"{OBJECT_FLAG_SALT}|{sample_token}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") < (1 << 63)


def training_mud_mask_id(sample_token: str, view_index: int) -> int:
    if view_index < 0 or view_index >= 6:
        raise ValueError("view_index must lie in 0..5")
    digest = hashlib.sha256(
        f"{MUD_MASK_SALT}|{sample_token}|{view_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % 16 + 1


def _sample_token(results) -> str:
    for key in ("sample_idx", "token"):
        value = results.get(key)
        if value is not None:
            return str(value)
    raise KeyError("training condition adapter lacks sample token")


def _image_list(results):
    images = results.get("img")
    if not isinstance(images, (list, tuple)) or len(images) != 6:
        raise ValueError("training condition adapter requires six images")
    return list(images)


def _points_in_boxes(points_xyz: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    try:
        from mmdet3d.core.bbox import box_np_ops
    except ImportError:  # pragma: no cover - newer mmdet3d layout.
        from mmdet3d.structures.ops import box_np_ops
    return box_np_ops.points_in_rbbox(points_xyz, boxes)


@PIPELINES.register_module()
class Stage019S2TrainingConditionAdapter:
    """Apply a locked S2 condition to train tokens without validation metadata.

    Validation-only nuScenes-R object flags and per-camera mask ids cannot be
    joined to train tokens.  The training calibration analogue therefore uses
    pre-registered SHA256 assignments while retaining the exact corruption
    operation: whole-object point removal and the frozen 16-mask mud formula.
    """

    def __init__(self, condition, mask_root, audit_path,
                 beam_overlay_root=None):
        if condition not in CONDITIONS:
            raise ValueError(f"unsupported Stage019-S2 condition: {condition}")
        self.condition = str(condition)
        self.mask_root = Path(mask_root)
        self.audit_path = Path(audit_path)
        self.beam_overlay_root = (
            Path(beam_overlay_root).resolve()
            if beam_overlay_root is not None else None
        )
        if self.condition == "camera_mud_mask" and not self.mask_root.is_dir():
            raise FileNotFoundError(self.mask_root)
        if self.condition == "beam_reduction_4" and self.beam_overlay_root is None:
            raise ValueError("beam condition requires its artifact-local overlay root")

    def _audit(self, payload) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def _apply_limited_fov(self, points):
        from nuscenes_r_mome_conditions import limited_fov_original_code_mask

        tensor = points.tensor
        mask = limited_fov_original_code_mask(
            tensor[:, :3].detach().cpu().numpy(),
            minimum_degrees=-60.0,
            maximum_degrees=60.0,
        )
        keep = torch.from_numpy(mask).to(device=tensor.device, dtype=torch.bool)
        return points[keep], int((~keep).sum().item())

    def _apply_object_failure(self, results, token):
        points = results["points"]
        boxes = results.get("gt_bboxes_3d")
        flag = training_object_failure_flag(token)
        if not flag or boxes is None or len(boxes) == 0:
            return points, flag, 0, 0
        point_xyz = points.tensor[:, :3].detach().cpu().numpy()
        box_values = boxes.tensor[:, :7].detach().cpu().numpy()
        membership = _points_in_boxes(point_xyz, box_values)
        remove = torch.from_numpy(membership.any(axis=1)).to(
            device=points.tensor.device, dtype=torch.bool
        )
        return points[~remove], flag, int(remove.sum().item()), int(len(boxes))

    def _apply_mud(self, results, token):
        from nuscenes_r_core4_adapter_v2 import mud_mask_blend

        images = _image_list(results)
        output = []
        mask_ids = []
        changed = []
        for view_index, image in enumerate(images):
            mask_id = training_mud_mask_id(token, view_index)
            mask = mmcv.imread(
                str(self.mask_root / f"mask_{mask_id}.jpg"), "unchanged"
            )
            if mask is None:
                raise RuntimeError(f"failed to read mud mask {mask_id}")
            original = np.asarray(image)
            blended = mud_mask_blend(original, mask)
            output.append(blended)
            mask_ids.append(mask_id)
            changed.append(int(np.count_nonzero(blended != original)))
        if any(value <= 0 for value in changed):
            raise RuntimeError("training mud mask produced an unchanged view")
        results["img"] = output
        return mask_ids, changed

    def __call__(self, results):
        token = _sample_token(results)
        points = results.get("points")
        if points is None or not hasattr(points, "tensor"):
            raise TypeError("training condition adapter requires loaded points")
        before_points = int(points.tensor.shape[0])
        details = {}
        removed = 0
        if self.condition == "beam_reduction_4":
            source = Path(str(results.get("pts_filename", ""))).resolve()
            try:
                source.relative_to(self.beam_overlay_root)
            except ValueError as error:
                raise RuntimeError(
                    "beam annotation did not bind to the artifact-local overlay"
                ) from error
            details["overlay_verified"] = True
        elif self.condition == "lidar_zero":
            results["points"].tensor = results["points"].tensor * 0.0
            details["zeroed_point_count"] = before_points
        elif self.condition == "limited_fov_original_code_60":
            results["points"], removed = self._apply_limited_fov(points)
        elif self.condition == "lidar_object_failure":
            output, flag, removed, box_count = self._apply_object_failure(
                results, token
            )
            results["points"] = output
            details.update(
                {
                    "object_failure": flag,
                    "gt_box_count": box_count,
                    "assignment_salt": OBJECT_FLAG_SALT,
                }
            )
        elif self.condition == "camera_zero":
            images = _image_list(results)
            results["img"] = [np.zeros_like(np.asarray(image)) for image in images]
            details["zeroed_views"] = 6
        elif self.condition == "camera_mud_mask":
            mask_ids, changed = self._apply_mud(results, token)
            details.update(
                {
                    "mask_ids": mask_ids,
                    "changed_pixels_per_view": changed,
                    "assignment_salt": MUD_MASK_SALT,
                }
            )

        after_points = int(results["points"].tensor.shape[0])
        details["removed_points"] = removed or max(0, before_points - after_points)
        results["qta_corruption"] = {
            "protocol": "mome_stage019_s2_training20_conditions_v1",
            "condition": self.condition,
            "details": details,
        }
        self._audit(
            {
                "schema": "visfuse3d_stage019_s2_training_condition_audit_v1",
                "token": token,
                "condition": self.condition,
                "before_point_count": before_points,
                "after_point_count": after_points,
                "details": details,
            }
        )
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(condition={self.condition!r}, "
            f"mask_root={str(self.mask_root)!r})"
        )
