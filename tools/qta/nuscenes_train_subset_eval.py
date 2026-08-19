"""nuScenes detection metrics on an explicitly enumerated train-scene subset.

This uses the official detection_cvpr_2019 metric implementation, but the
result is deliberately labelled a train-subset probe rather than an official
validation-set score.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt,
    load_prediction,
)
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import DetectionEval

try:
    from .common import atomic_write_json
except ImportError:
    from common import atomic_write_json


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def subset_boxes(source: EvalBoxes, sample_tokens: Iterable[str]) -> EvalBoxes:
    output = EvalBoxes()
    for token in sample_tokens:
        output.add_boxes(str(token), source[str(token)])
    return output


def load_subset_ground_truth(
    nusc: NuScenes,
    sample_tokens: list[str],
    eval_set: str,
    eval_version: str = "detection_cvpr_2019",
) -> tuple[EvalBoxes, object]:
    if eval_set not in ("train", "val"):
        raise ValueError("subset evaluation supports only train or val")
    config = config_factory(eval_version)
    full_gt = load_gt(nusc, eval_set, DetectionBox, verbose=False)
    missing = set(sample_tokens) - set(full_gt.sample_tokens)
    if missing:
        raise ValueError(f"train subset has unknown sample tokens: {sorted(missing)[:5]}")
    gt_boxes = subset_boxes(full_gt, sample_tokens)
    gt_boxes = add_center_dist(nusc, gt_boxes)
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, config.class_range, verbose=False)
    return gt_boxes, config


def load_train_subset_ground_truth(
    nusc: NuScenes,
    sample_tokens: list[str],
    eval_version: str = "detection_cvpr_2019",
) -> tuple[EvalBoxes, object]:
    return load_subset_ground_truth(nusc, sample_tokens, "train", eval_version)


def evaluate_subset_prediction(
    nusc: NuScenes,
    result_path: Path,
    sample_tokens: list[str],
    gt_boxes: EvalBoxes,
    config,
    output_path: Path,
    eval_set: str,
) -> dict[str, object]:
    if eval_set not in ("train", "val"):
        raise ValueError("subset evaluation supports only train or val")
    pred_boxes, meta = load_prediction(
        str(result_path), config.max_boxes_per_sample, DetectionBox, verbose=False
    )
    if set(pred_boxes.sample_tokens) != set(sample_tokens):
        raise ValueError("prediction tokens do not exactly match the train subset")
    pred_boxes = add_center_dist(nusc, pred_boxes)
    pred_boxes = filter_eval_boxes(
        nusc, pred_boxes, config.class_range, verbose=False
    )
    evaluator = object.__new__(DetectionEval)
    evaluator.nusc = nusc
    evaluator.result_path = str(result_path)
    evaluator.eval_set = f"{eval_set}_subset"
    evaluator.output_dir = str(output_path.parent)
    evaluator.verbose = False
    evaluator.cfg = config
    evaluator.pred_boxes = pred_boxes
    evaluator.gt_boxes = gt_boxes
    evaluator.meta = meta
    evaluator.sample_tokens = list(sample_tokens)
    metrics, _ = DetectionEval.evaluate(evaluator)
    summary = metrics.serialize()
    summary.update(
        {
            "meta": meta,
            "source_split": eval_set,
            "subset_sample_count": len(sample_tokens),
            "protocol": f"official_detection_cvpr_2019_algorithm_{eval_set}_subset_v1",
            "official_validation_score": bool(
                eval_set == "val" and len(sample_tokens) == 6019
            ),
            "claim_boundary": (
                "stage019_s4_full_validation_result"
                if eval_set == "val" and len(sample_tokens) == 6019
                else (
                    "stage019_s4_validation_pilot_subset_go_no_go_only"
                    if eval_set == "val"
                    else "stage019_s4_train_subset_probe_only"
                )
            ),
        }
    )
    atomic_write_json(output_path, _json_safe(summary))
    return summary


def evaluate_train_subset_prediction(
    nusc: NuScenes,
    result_path: Path,
    sample_tokens: list[str],
    gt_boxes: EvalBoxes,
    config,
    output_path: Path,
) -> dict[str, object]:
    return evaluate_subset_prediction(
        nusc,
        result_path,
        sample_tokens,
        gt_boxes,
        config,
        output_path,
        "train",
    )
