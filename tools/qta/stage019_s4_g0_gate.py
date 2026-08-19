"""Aggregate the preregistered Stage019-S4 G0 reachability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import is_relative_to
    from .stage019_s4_contracts import (
        ACTIONABLE_CONDITIONS,
        FAULT_CONDITIONS,
        G0_FOLD_SIZES,
        G0_FRAME_COUNT,
        G0_SCENE_COUNT,
        PROTOCOL_PROFILE,
        evaluate_g0_gate,
        sha256_file,
    )
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import is_relative_to
    from stage019_s4_contracts import (
        ACTIONABLE_CONDITIONS,
        FAULT_CONDITIONS,
        G0_FOLD_SIZES,
        G0_FRAME_COUNT,
        G0_SCENE_COUNT,
        PROTOCOL_PROFILE,
        evaluate_g0_gate,
        sha256_file,
    )


ROLES = ("original_mome", "g0_score", "g0_attribute", "g0_union")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _metric_delta(left: dict, right: dict) -> dict[str, float]:
    return {key: float(left[key]) - float(right[key]) for key in ("mAP", "NDS")}


def _load_conditions(paths: list[Path]) -> dict[str, dict]:
    if len(paths) != len(ACTIONABLE_CONDITIONS):
        raise ValueError("G0 gate requires exactly five condition manifests")
    output = {}
    common_identity = None
    expected_constraints = {
        "same_class_match_radius_m": 2.0,
        "attribute_center_max_m": 1.0,
        "attribute_log_dimension_max": 0.20,
        "attribute_yaw_max_degrees": 20.0,
        "attribute_velocity_max_mps": 2.0,
        "union_max_added_boxes_per_frame": 20,
        "union_final_max_boxes_per_frame": 300,
        "union_same_class_dedup_radius_m": 2.0,
    }
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = str(payload.get("condition"))
        metrics = payload.get("metrics", {})
        folds = payload.get("fold_metrics", [])
        if (
            payload.get("schema") != "visfuse3d_stage019_s4_condition_merge_manifest_v1"
            or payload.get("status") != "complete_g0_train_subset_evaluation"
            or payload.get("protocol_profile") != PROTOCOL_PROFILE
            or payload.get("phase") != "g0"
            or payload.get("source_split") != "train"
            or condition not in ACTIONABLE_CONDITIONS
            or condition in output
            or int(payload.get("scene_count", -1)) != G0_SCENE_COUNT
            or int(payload.get("frame_count", -1)) != G0_FRAME_COUNT
            or int(payload.get("unique_token_count", -1)) != G0_FRAME_COUNT
            or int(payload.get("engineering_failure_count", -1)) != 0
            or set(metrics) != set(ROLES)
            or len(folds) != 3
            or tuple(int(value.get("scene_count", -1)) for value in folds)
            != G0_FOLD_SIZES
            or payload.get("oracle_constraints") != expected_constraints
        ):
            raise ValueError(f"invalid G0 condition manifest: {path}")
        for role in ROLES:
            for metric in ("mAP", "NDS"):
                value = float(metrics[role][metric])
                if not -1.0 <= value <= 1.0:
                    raise ValueError(f"invalid G0 metric {condition}/{role}/{metric}")
        if any(set(fold.get("metrics", {})) != set(ROLES) for fold in folds):
            raise ValueError("G0 fold role schema drifted")
        identity = payload.get("locked_worker_identity", {})
        current_common = {
            "scene_tokens": tuple(payload.get("scene_tokens", ())),
            "frame_token_order_sha256": payload.get("frame_token_order_sha256"),
            "fold_scene_tokens": tuple(
                tuple(fold.get("scene_tokens", ())) for fold in folds
            ),
            "config_sha256": identity.get("config_sha256"),
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "scene_map_sha256": identity.get("scene_map_sha256"),
            "scene_list_sha256": identity.get("scene_list_sha256"),
            "source_bundle_sha256": identity.get("source_bundle_sha256"),
        }
        if common_identity is None:
            common_identity = current_common
        elif current_common != common_identity:
            raise ValueError("G0 conditions mix scene/fold/frame/source identities")
        output[condition] = payload
    if set(output) != set(ACTIONABLE_CONDITIONS):
        raise ValueError("G0 condition set is incomplete")
    return output


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("G0 gate output must stay inside artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("G0 gate output cannot enter source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable G0 gate output exists: {output_dir}")
    conditions = _load_conditions(args.condition_manifest)
    score_deltas = {
        condition: _metric_delta(
            payload["metrics"]["g0_score"], payload["metrics"]["original_mome"]
        )
        for condition, payload in conditions.items()
        if condition in FAULT_CONDITIONS
    }
    attribute_deltas = {
        condition: _metric_delta(
            payload["metrics"]["g0_attribute"], payload["metrics"]["g0_score"]
        )
        for condition, payload in conditions.items()
        if condition in FAULT_CONDITIONS
    }
    union_deltas = {
        condition: _metric_delta(
            payload["metrics"]["g0_union"], payload["metrics"]["g0_attribute"]
        )
        for condition, payload in conditions.items()
        if condition in FAULT_CONDITIONS
    }
    score_fold_macro = []
    attribute_fold_macro = []
    for fold_index in range(3):
        score_rows = [
            _metric_delta(
                conditions[condition]["fold_metrics"][fold_index]["metrics"]["g0_score"],
                conditions[condition]["fold_metrics"][fold_index]["metrics"]["original_mome"],
            )
            for condition in FAULT_CONDITIONS
        ]
        attribute_rows = [
            _metric_delta(
                conditions[condition]["fold_metrics"][fold_index]["metrics"]["g0_attribute"],
                conditions[condition]["fold_metrics"][fold_index]["metrics"]["g0_score"],
            )
            for condition in FAULT_CONDITIONS
        ]
        score_fold_macro.append(
            {
                metric: sum(row[metric] for row in score_rows) / len(score_rows)
                for metric in ("mAP", "NDS")
            }
        )
        attribute_fold_macro.append(
            {
                metric: sum(row[metric] for row in attribute_rows) / len(attribute_rows)
                for metric in ("mAP", "NDS")
            }
        )
    opportunities = [
        condition
        for condition in FAULT_CONDITIONS
        if int(conditions[condition].get("union_added_true_positive_count", 0)) > 0
    ]
    decision = evaluate_g0_gate(
        score_deltas,
        attribute_deltas,
        union_deltas,
        score_fold_macro,
        attribute_fold_macro,
        opportunities,
    )
    manifest = {
        **decision,
        "schema": "visfuse3d_stage019_s4_g0_gate_manifest_v1",
        "protocol_profile": PROTOCOL_PROFILE,
        "score_deltas": score_deltas,
        "attribute_minus_score_deltas": attribute_deltas,
        "union_minus_attribute_deltas": union_deltas,
        "score_fold_fault_macro_deltas": score_fold_macro,
        "attribute_fold_fault_macro_deltas": attribute_fold_macro,
        "condition_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.condition_manifest
        ],
        "claim_boundary": (
            "gt_aware_train20_reachability_and_training_authorization_only_"
            "not_validation_result_or_deployable_method_success"
        ),
    }
    output_dir.mkdir(parents=True)
    atomic_write_json(output_dir / "stage019_s4_g0_gate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if bool(manifest["training_authorized"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
