"""Hash-verify, merge and evaluate paired Stage019-S4 scene shards."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import _import_plugin, is_relative_to
    from .nuscenes_train_subset_eval import (
        evaluate_subset_prediction,
        load_subset_ground_truth,
        subset_boxes,
    )
    from .stage019_s4_contracts import (
        EXPERT_ROLES,
        G0_FOLD_SIZES,
        PROTOCOL_PROFILE,
        scene_serialization_sha256,
        sha256_file,
        split_g0_folds,
        validate_pilot_identity,
        validate_merge_coverage,
    )
    from .stage019_s4_oracles import ORACLE_CONSTRAINTS
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import _import_plugin, is_relative_to
    from nuscenes_train_subset_eval import (
        evaluate_subset_prediction,
        load_subset_ground_truth,
        subset_boxes,
    )
    from stage019_s4_contracts import (
        EXPERT_ROLES,
        G0_FOLD_SIZES,
        PROTOCOL_PROFILE,
        scene_serialization_sha256,
        sha256_file,
        split_g0_folds,
        validate_pilot_identity,
        validate_merge_coverage,
    )
    from stage019_s4_oracles import ORACLE_CONSTRAINTS


G0_EVAL_ROLES = ("original_mome", "g0_score", "g0_attribute", "g0_union")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--phase", choices=("g0", "fit", "calibration", "pilot", "fullval", "smoke"), required=True
    )
    parser.add_argument("--worker-manifest", type=Path, action="append", required=True)
    parser.add_argument("--expected-scene-count", type=int, required=True)
    parser.add_argument("--expected-frame-count", type=int, required=True)
    parser.add_argument("--eval-set", choices=("train", "val"), required=True)
    parser.add_argument("--role", action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _locked_identity(value: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "scene_offset",
        "scene_count",
        "requested_scenes",
        "worker_id",
        "gpu_identity",
    }
    return {key: item for key, item in value.items() if key not in ignored}


def _metric_summary(raw: dict[str, Any]) -> dict[str, Any]:
    if "mean_ap" not in raw or "nd_score" not in raw:
        raise ValueError("nuScenes subset evaluation lacks mean_ap/nd_score")
    return {
        "mAP": float(raw["mean_ap"]),
        "NDS": float(raw["nd_score"]),
        "mATE": float(raw.get("tp_errors", {}).get("trans_err", float("nan"))),
        "mASE": float(raw.get("tp_errors", {}).get("scale_err", float("nan"))),
        "mAOE": float(raw.get("tp_errors", {}).get("orient_err", float("nan"))),
        "mAVE": float(raw.get("tp_errors", {}).get("vel_err", float("nan"))),
        "mAAE": float(raw.get("tp_errors", {}).get("attr_err", float("nan"))),
        "mean_dist_aps": raw.get("mean_dist_aps", {}),
    }


def _format_prediction_json(dataset, predictions: dict[str, dict], tokens: list[str], role: str, root: Path) -> Path:
    token_to_info = {str(info["token"]): info for info in dataset.data_infos}
    missing = [token for token in tokens if token not in token_to_info]
    if missing:
        raise ValueError(f"evaluation annotation lacks merged tokens: {missing[:3]}")
    dataset.data_infos = [token_to_info[token] for token in tokens]
    rows = [{"pts_bbox": predictions[token][role]} for token in tokens]
    result_files, temporary = dataset.format_results(rows, jsonfile_prefix=str(root))
    if temporary is not None:
        # A deterministic explicit prefix should never use a TemporaryDirectory.
        temporary.cleanup()
        raise RuntimeError("S4 formatting unexpectedly used a temporary result directory")
    path = Path(result_files["pts_bbox"] if isinstance(result_files, dict) else result_files)
    if not path.is_file():
        raise RuntimeError("S4 formatting did not produce results_nusc.json")
    return path


def _filter_result_json(source: Path, tokens: list[str], destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    if not set(tokens) <= set(results):
        raise ValueError("formatted prediction lacks fold tokens")
    atomic_write_json(
        destination,
        {"meta": payload.get("meta", {}), "results": {token: results[token] for token in tokens}},
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("S4 merge output must stay inside artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("S4 merge output cannot enter source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable S4 merge output exists: {output_dir}")
    if len(args.worker_manifest) != 2:
        raise ValueError("S4 merge requires exactly two worker manifests")
    raw_manifests = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.worker_manifest
    ]
    manifests = sorted(
        raw_manifests, key=lambda value: int(value["input_identity"]["scene_offset"])
    )
    validate_merge_coverage(
        manifests, args.expected_scene_count, args.expected_frame_count
    )
    locked = None
    predictions: dict[str, dict] = {}
    scene_to_tokens: dict[str, list[str]] = {}
    scene_order = []
    oracle_counts = {
        "attribute_modified_object_count": 0,
        "union_added_box_count": 0,
        "union_added_true_positive_count": 0,
    }
    changed_object_count = 0
    accepted_object_count = 0
    rescued_object_count = 0
    eval_identity = None
    roles = None
    for manifest in manifests:
        identity = manifest["input_identity"]
        schema = manifest.get("schema")
        if (
            schema
            not in (
                "visfuse3d_stage019_s4_extract_worker_manifest_v1",
                "visfuse3d_stage019_s4_eval_worker_manifest_v1",
            )
            or identity.get("protocol_profile") != PROTOCOL_PROFILE
            or manifest.get("phase") != args.phase
            or manifest.get("condition") != args.condition
        ):
            raise ValueError("S4 worker phase/condition/protocol identity mismatch")
        if args.phase == "g0" and schema != "visfuse3d_stage019_s4_extract_worker_manifest_v1":
            raise ValueError("G0 merge accepts extraction workers only")
        if args.phase in ("calibration", "pilot", "fullval") and schema != (
            "visfuse3d_stage019_s4_eval_worker_manifest_v1"
        ):
            raise ValueError("calibration/pilot/fullval merge accepts eval workers only")
        if schema == "visfuse3d_stage019_s4_eval_worker_manifest_v1":
            current_eval_identity = {
                "mode": manifest.get("mode"),
                "fusion_strength": float(manifest.get("fusion_strength", -1.0)),
                "head_checkpoint": manifest.get("head_checkpoint"),
                "source_hashes": manifest.get("source_hashes"),
            }
            if eval_identity is None:
                eval_identity = current_eval_identity
            elif eval_identity != current_eval_identity:
                raise ValueError("S4 eval shards have different mode/strength/checkpoint/source")
        changed_object_count += int(manifest.get("changed_object_count", 0))
        accepted_object_count += int(manifest.get("accepted_object_count", 0))
        rescued_object_count += int(manifest.get("rescued_object_count", 0))
        current_locked = _locked_identity(identity)
        if locked is None:
            locked = current_locked
        elif current_locked != locked:
            raise ValueError("S4 shards have different source/input identities")
        current_roles = tuple(manifest.get("roles", ()))
        if roles is None:
            roles = current_roles
        elif roles != current_roles:
            raise ValueError("S4 shards expose different role schemas")
        for scene in manifest.get("scenes", []):
            scene_token = str(scene["scene_token"])
            scene_order.append(scene_token)
            frame_tokens = [str(value) for value in scene["frame_tokens"]]
            scene_to_tokens[scene_token] = frame_tokens
            prediction_path = Path(scene["prediction_path"])
            diagnostic_path = Path(scene["diagnostic_path"])
            audit_path = Path(scene["corruption_audit_path"])
            for path, expected in (
                (prediction_path, scene["prediction_sha256"]),
                (diagnostic_path, scene["diagnostic_sha256"]),
                (audit_path, scene["corruption_audit_sha256"]),
            ):
                if not path.is_file() or sha256_file(path) != expected:
                    raise ValueError("S4 scene evidence hash mismatch")
            if audit_path.stat().st_size <= 0:
                raise ValueError("S4 scene corruption audit is empty")
            with prediction_path.open("rb") as stream:
                rows = pickle.load(stream)  # noqa: S301 - local immutable evidence.
            if list(rows) != frame_tokens or any(tuple(value) != roles for value in rows.values()):
                raise ValueError("S4 prediction token/role order drifted")
            if set(rows) & set(predictions):
                raise ValueError("S4 shards contain duplicate prediction tokens")
            predictions.update(rows)
            if schema == "visfuse3d_stage019_s4_eval_worker_manifest_v1":
                diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
                observed_tokens = [
                    str(value["frame_token"]) for value in diagnostic.get("frames", [])
                ]
                if observed_tokens != frame_tokens:
                    raise ValueError("S4 eval diagnostic token order drifted")
                continue
            with np.load(diagnostic_path, allow_pickle=False) as arrays:
                observed_tokens = [
                    value.decode("ascii") if isinstance(value, bytes) else str(value)
                    for value in arrays["frame_token"].tolist()
                ]
                if observed_tokens != frame_tokens:
                    raise ValueError("S4 diagnostic token order drifted")
                expected_calls = 1 if bool(manifest["hard_bypass"]) else 4
                if not np.all(arrays["decoder_call_count"] == expected_calls):
                    raise ValueError("S4 diagnostic decoder-call count drifted")
                if args.phase == "g0":
                    if arrays["feature_matrix"].shape[1] != 41:
                        raise ValueError("S4 diagnostic feature width drifted")
                    if arrays["expert_source_indices"].shape[1] != 3:
                        raise ValueError("S4 diagnostic source-index shape drifted")
                    oracle_counts["attribute_modified_object_count"] += int(
                        arrays["g0_attribute_modified_object_count"].sum()
                    )
                    oracle_counts["union_added_box_count"] += int(
                        arrays["g0_union_added_box_count"].sum()
                    )
                    oracle_counts["union_added_true_positive_count"] += int(
                        arrays["g0_union_added_true_positive_count"].sum()
                    )
    if len(scene_order) != args.expected_scene_count or len(predictions) != args.expected_frame_count:
        raise ValueError("S4 merge coverage changed after evidence loading")
    if args.phase == "pilot":
        validate_pilot_identity(scene_order, len(predictions))
    if (
        locked is None
        or sha256_file(args.config) != locked.get("config_sha256")
        or sha256_file(args.ann_file) != locked.get("annotation_sha256")
    ):
        raise ValueError("S4 merge config/annotation differs from worker lock")
    selected_roles = tuple(args.role or (G0_EVAL_ROLES if args.phase == "g0" else roles or ()))
    if not selected_roles or not set(selected_roles) <= set(roles or ()):
        raise ValueError("requested S4 evaluation role is absent from worker predictions")
    if args.phase == "g0" and selected_roles != G0_EVAL_ROLES:
        raise ValueError("G0 merge must evaluate exact original/score/attribute/union roles")

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from nuscenes import NuScenes

    cfg = Config.fromfile(str(args.config))
    _import_plugin(cfg, args.config)
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.data_root = str(args.data_root.resolve()).replace("\\", "/") + "/"
    dataset_cfg.ann_file = str(args.ann_file.resolve()).replace("\\", "/")
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    all_tokens = [token for scene in scene_order for token in scene_to_tokens[scene]]
    if len(all_tokens) != args.expected_frame_count or len(set(all_tokens)) != len(all_tokens):
        raise ValueError("S4 merged scene frame order is incomplete or duplicated")
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(args.data_root.resolve()), verbose=False)
    gt_boxes, eval_config = load_subset_ground_truth(nusc, all_tokens, args.eval_set)
    output_dir.mkdir(parents=True)
    metrics = {}
    prediction_outputs = {}
    role_json_paths = {}
    original_data_infos = list(dataset.data_infos)
    for role in selected_roles:
        dataset.data_infos = list(original_data_infos)
        result_path = _format_prediction_json(
            dataset, predictions, all_tokens, role, output_dir / "predictions" / role
        )
        role_json_paths[role] = result_path
        raw = evaluate_subset_prediction(
            nusc,
            result_path,
            all_tokens,
            gt_boxes,
            eval_config,
            output_dir / "metrics" / role / "metrics_summary.json",
            args.eval_set,
        )
        metrics[role] = _metric_summary(raw)
        prediction_outputs[role] = {
            "path": str(result_path.resolve()),
            "sha256": sha256_file(result_path),
        }
    hard_bypass_identity = None
    if args.condition in ("lidar_zero", "camera_zero"):
        if set(selected_roles) != {"original_mome", "final"}:
            raise ValueError("complete-zero merge must evaluate original and final only")
        original_record = prediction_outputs["original_mome"]
        final_record = prediction_outputs["final"]
        if original_record["sha256"] != final_record["sha256"]:
            raise ValueError("complete-zero original/final canonical prediction SHA differs")
        hard_bypass_identity = {
            "same_run_original_reference": True,
            "original_prediction": original_record,
            "final_prediction": final_record,
        }
    fold_metrics = []
    if args.phase == "g0":
        folds = split_g0_folds(scene_order)
        for fold_index, fold_scenes in enumerate(folds):
            fold_tokens = [token for scene in fold_scenes for token in scene_to_tokens[scene]]
            fold_gt, fold_config = subset_boxes(gt_boxes, fold_tokens), eval_config
            current_metrics = {}
            for role in G0_EVAL_ROLES:
                fold_json = output_dir / "folds" / f"fold_{fold_index}" / role / "results_nusc.json"
                _filter_result_json(role_json_paths[role], fold_tokens, fold_json)
                raw = evaluate_subset_prediction(
                    nusc,
                    fold_json,
                    fold_tokens,
                    fold_gt,
                    fold_config,
                    output_dir / "folds" / f"fold_{fold_index}" / role / "metrics_summary.json",
                    args.eval_set,
                )
                current_metrics[role] = _metric_summary(raw)
            fold_metrics.append(
                {
                    "fold_index": fold_index,
                    "scene_tokens": list(fold_scenes),
                    "scene_count": len(fold_scenes),
                    "frame_count": len(fold_tokens),
                    "metrics": current_metrics,
                }
            )
        if tuple(value["scene_count"] for value in fold_metrics) != G0_FOLD_SIZES:
            raise AssertionError("G0 fold coverage drifted")
    manifest = {
        "schema": "visfuse3d_stage019_s4_condition_merge_manifest_v1",
        "status": (
            "complete_g0_train_subset_evaluation"
            if args.phase == "g0"
            else "complete_condition_evaluation"
        ),
        "protocol_profile": PROTOCOL_PROFILE,
        "phase": args.phase,
        "condition": args.condition,
        "source_split": args.eval_set,
        "scene_count": len(scene_order),
        "frame_count": len(predictions),
        "unique_token_count": len(set(predictions)),
        "scene_tokens": scene_order,
        "frame_token_order_sha256": scene_serialization_sha256(all_tokens),
        "roles": list(selected_roles),
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "prediction_outputs": prediction_outputs,
        "hard_bypass_identity": hard_bypass_identity,
        "changed_object_count": changed_object_count,
        "accepted_object_count": accepted_object_count,
        "rescued_object_count": rescued_object_count,
        "oracle_constraints": ORACLE_CONSTRAINTS if args.phase == "g0" else None,
        **oracle_counts,
        "engineering_failure_count": 0,
        "worker_manifests": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.worker_manifest
        ],
        "locked_worker_identity": locked,
        "locked_eval_identity": eval_identity,
        "claim_boundary": (
            "gt_aware_train_subset_reachability_only_not_official_validation"
            if args.phase == "g0"
            else "phase_specific_condition_evaluation"
        ),
    }
    atomic_write_json(output_dir / "stage019_s4_condition_merge_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
