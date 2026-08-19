"""GT-aware, bounded Stage019-S4 G0 object-set reachability oracles."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

try:
    from .stage019_s4_contracts import EXPERT_ROLES, MAX_OUTPUT_BOXES, MAX_RESCUE_BOXES
    from .stage019_s4_object_cache import (
        candidate_soft_quality,
        one_to_one_soft_quality,
        wrap_angle,
    )
except ImportError:
    from stage019_s4_contracts import EXPERT_ROLES, MAX_OUTPUT_BOXES, MAX_RESCUE_BOXES
    from stage019_s4_object_cache import (
        candidate_soft_quality,
        one_to_one_soft_quality,
        wrap_angle,
    )


ORACLE_CONSTRAINTS = {
    "same_class_match_radius_m": 2.0,
    "attribute_center_max_m": 1.0,
    "attribute_log_dimension_max": 0.20,
    "attribute_yaw_max_degrees": 20.0,
    "attribute_velocity_max_mps": 2.0,
    "union_max_added_boxes_per_frame": 20,
    "union_final_max_boxes_per_frame": 300,
    "union_same_class_dedup_radius_m": 2.0,
}


def _bounded_vector(delta: np.ndarray, maximum_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    return delta if norm <= maximum_norm else delta * (maximum_norm / max(norm, 1e-12))


def _oracle_scores(quality: np.ndarray, original_scores: np.ndarray) -> np.ndarray:
    # GT quality is the primary ranking key; the original score and source
    # index provide deterministic ties without changing the quality strata.
    count = len(quality)
    source_tie = (count - np.arange(count, dtype=np.float64)) * 1e-10
    value = quality.astype(np.float64) + original_scores.astype(np.float64) * 1e-3 + source_tie
    return np.clip(value / 1.0010001, 0.0, 1.0).astype(np.float32)


def _select_attribute_candidate(
    anchor_box: np.ndarray,
    candidate_boxes: list[np.ndarray],
    source_ids: list[int],
    gt_box: np.ndarray,
    component: str,
) -> int:
    values = np.asarray(candidate_boxes, dtype=np.float32)
    if component == "center":
        loss = np.linalg.norm(values[:, :3] - gt_box[None, :3], axis=1)
    elif component == "size":
        loss = np.abs(np.log(values[:, 3:6]) - np.log(gt_box[None, 3:6])).sum(axis=1)
    elif component == "yaw":
        loss = np.abs(wrap_angle(values[:, 6] - gt_box[6]))
    elif component == "velocity":
        loss = np.linalg.norm(values[:, 7:9] - gt_box[None, 7:9], axis=1)
    else:
        raise ValueError(component)
    order = np.lexsort((np.asarray(source_ids, dtype=np.int64), loss))
    return int(order[0])


def build_g0_oracles(
    original: Mapping[str, np.ndarray],
    experts: Mapping[str, Mapping[str, np.ndarray]],
    cache_record: Mapping[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """Return score/attribute/union predictions under exact formal caps."""

    inference = cache_record["inference_features"]
    targets = cache_record["offline_targets"]
    anchor_boxes = np.asarray(original["boxes"], dtype=np.float32)
    anchor_scores = np.asarray(original["scores"], dtype=np.float32)
    anchor_labels = np.asarray(original["labels"], dtype=np.int64)
    if not (
        np.array_equal(anchor_boxes, inference["anchor_boxes"])
        and np.array_equal(anchor_scores, inference["anchor_scores"])
        and np.array_equal(anchor_labels, inference["anchor_labels"])
    ):
        raise ValueError("G0 cache anchor differs from same-run original prediction")
    source_indices = np.asarray(inference["expert_source_indices"], dtype=np.int64)
    gt_boxes = np.asarray(targets["gt_boxes"], dtype=np.float32)
    gt_labels = np.asarray(targets["gt_labels"], dtype=np.int64)
    score_quality, target_gt = one_to_one_soft_quality(
        anchor_boxes, anchor_labels, gt_boxes, gt_labels
    )
    has_expert_match = np.asarray(
        inference["expert_match_mask"], dtype=np.bool_
    ).any(axis=1)
    reachable_scores = _oracle_scores(score_quality, anchor_scores)
    score = {
        "boxes": anchor_boxes.copy(),
        "scores": np.where(
            has_expert_match, reachable_scores, anchor_scores
        ).astype(np.float32),
        "labels": anchor_labels.copy(),
    }

    attribute_boxes = anchor_boxes.copy()
    modified = 0
    for anchor_index in range(len(anchor_boxes)):
        gt_index = int(target_gt[anchor_index])
        if gt_index < 0:
            continue
        candidates = [anchor_boxes[anchor_index]]
        source_ids = [0]
        for route_index, role in enumerate(EXPERT_ROLES):
            source_index = int(source_indices[anchor_index, route_index])
            if source_index >= 0:
                candidates.append(np.asarray(experts[role]["boxes"])[source_index])
                source_ids.append(route_index + 1)
        gt_box = gt_boxes[gt_index]
        selected = {
            component: _select_attribute_candidate(
                anchor_boxes[anchor_index], candidates, source_ids, gt_box, component
            )
            for component in ("center", "size", "yaw", "velocity")
        }
        output = anchor_boxes[anchor_index].copy()
        center_delta = (
            np.asarray(candidates[selected["center"]])[:3] - anchor_boxes[anchor_index, :3]
        )
        output[:3] += _bounded_vector(center_delta, 1.0)
        size_delta = np.log(np.asarray(candidates[selected["size"]])[3:6]) - np.log(
            anchor_boxes[anchor_index, 3:6]
        )
        output[3:6] = anchor_boxes[anchor_index, 3:6] * np.exp(
            np.clip(size_delta, -0.20, 0.20)
        )
        yaw_delta = float(
            wrap_angle(
                np.asarray([candidates[selected["yaw"]][6] - anchor_boxes[anchor_index, 6]])
            )[0]
        )
        output[6] = float(
            wrap_angle(
                np.asarray(
                    [
                        anchor_boxes[anchor_index, 6]
                        + np.clip(yaw_delta, -math.pi / 9.0, math.pi / 9.0)
                    ]
                )
            )[0]
        )
        velocity_delta = (
            np.asarray(candidates[selected["velocity"]])[7:9]
            - anchor_boxes[anchor_index, 7:9]
        )
        output[7:9] += _bounded_vector(velocity_delta, 2.0)
        if not np.array_equal(output, anchor_boxes[anchor_index]):
            modified += 1
        attribute_boxes[anchor_index] = output
    attribute = {
        "boxes": attribute_boxes,
        "scores": score["scores"].copy(),
        "labels": anchor_labels.copy(),
    }

    union_boxes = [value.copy() for value in attribute_boxes]
    union_scores = score["scores"].tolist()
    union_labels = anchor_labels.tolist()
    candidates = []
    claimed_gt = set(int(value) for value in target_gt.tolist() if int(value) >= 0)
    for route_index, role in enumerate(EXPERT_ROLES):
        role_boxes = np.asarray(experts[role]["boxes"], dtype=np.float32)
        role_scores = np.asarray(experts[role]["scores"], dtype=np.float32)
        role_labels = np.asarray(experts[role]["labels"], dtype=np.int64)
        matched_source = set(
            int(value) for value in source_indices[:, route_index].tolist() if int(value) >= 0
        )
        quality, gt_index = candidate_soft_quality(
            role_boxes, role_labels, gt_boxes, gt_labels
        )
        for source_index in range(len(role_boxes)):
            if (
                source_index in matched_source
                or quality[source_index] <= 0.0
                or int(gt_index[source_index]) in claimed_gt
            ):
                continue
            candidates.append(
                (
                    -float(quality[source_index]),
                    route_index,
                    source_index,
                    int(gt_index[source_index]),
                    role_boxes[source_index],
                    float(role_scores[source_index]),
                    int(role_labels[source_index]),
                )
            )
    candidates.sort(key=lambda value: value[:4])
    rescued = 0
    rescued_gt = set()
    for neg_quality, route_index, source_index, gt_index, box, raw_score, label in candidates:
        if rescued >= MAX_RESCUE_BOXES or len(union_boxes) >= MAX_OUTPUT_BOXES:
            break
        duplicate = any(
            existing_label == label
            and np.linalg.norm(np.asarray(existing_box)[:2] - box[:2]) <= 2.0
            for existing_box, existing_label in zip(union_boxes, union_labels)
        )
        if duplicate or gt_index in rescued_gt:
            continue
        union_boxes.append(box.copy())
        union_scores.append(float(min(1.0, max(0.0, -neg_quality))))
        union_labels.append(label)
        rescued += 1
        if gt_index >= 0:
            rescued_gt.add(gt_index)
    union = {
        "boxes": np.asarray(union_boxes, dtype=np.float32).reshape(-1, 9),
        "scores": np.asarray(union_scores, dtype=np.float32),
        "labels": np.asarray(union_labels, dtype=np.int64),
    }
    if len(union["boxes"]) > MAX_OUTPUT_BOXES or rescued > MAX_RESCUE_BOXES:
        raise AssertionError("G0-union exceeded preregistered object caps")
    return {
        "g0_score": score,
        "g0_attribute": attribute,
        "g0_union": union,
    }, {
        "attribute_modified_object_count": modified,
        "union_added_box_count": rescued,
        "union_added_true_positive_count": len(rescued_gt),
    }
