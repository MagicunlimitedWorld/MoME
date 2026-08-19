"""Merge paired Stage019-S2 shards and optionally run official nuScenes eval."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import _import_plugin, is_relative_to, sha256_file
    from .full_query_oracle_worker import (
        CONDITIONS,
        EXPECTED_QUERY_COUNT,
        EXPECTED_TOKEN_SET_SHA256,
    )
    from .merge_full_query_oracle_shards import (
        _evidence_tree,
        _json_compatible,
        _metric,
    )
    from .stage019_s2_oracle_worker import S2A_ROLES, S2B_ROLES
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import _import_plugin, is_relative_to, sha256_file
    from full_query_oracle_worker import (
        CONDITIONS,
        EXPECTED_QUERY_COUNT,
        EXPECTED_TOKEN_SET_SHA256,
    )
    from merge_full_query_oracle_shards import (
        _evidence_tree,
        _json_compatible,
        _metric,
    )
    from stage019_s2_oracle_worker import S2A_ROLES, S2B_ROLES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("s2a", "s2b"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--scene-map", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--worker-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, required=True)
    parser.add_argument("--expected-frame-count", type=int, required=True)
    parser.add_argument("--official-evaluation", action="store_true")
    return parser.parse_args()


def _locked_identity(identity: dict) -> dict:
    keys = (
        "stage_id",
        "experiment_id",
        "run_id",
        "protocol_profile",
        "mode",
        "condition",
        "seed",
        "config_sha256",
        "checkpoint_sha256",
        "annotation_file_sha256",
        "scene_map_sha256",
        "sidecar_sha256",
        "margins_sha256",
        "margins_status",
        "locked_margins",
        "source_bundle_sha256",
        "trainable_parameter_count",
        "sample_access",
        "filter_empty_gt",
        "point_shuffle",
        "random_geometry",
    )
    return {key: identity.get(key) for key in keys}


def _decode_ascii(array: np.ndarray) -> list[str]:
    return [
        value.decode("ascii") if isinstance(value, bytes) else str(value)
        for value in array.reshape(-1).tolist()
    ]


def _validate_s2a_diagnostics(payload, frame_tokens: list[str]) -> dict:
    count = len(frame_tokens)
    expected = {
        "frame_token": (count,),
        "input_sha256": (count,),
        "query_identity": (count, 4, EXPECTED_QUERY_COUNT),
        "original_routes": (count, EXPECTED_QUERY_COUNT),
        "keep_query_losses": (count, EXPECTED_QUERY_COUNT),
        "full_context_query_losses": (count, EXPECTED_QUERY_COUNT, 3),
        "keep_anchored_actions": (count, EXPECTED_QUERY_COUNT),
        "keep_anchored_switch_mask": (count, EXPECTED_QUERY_COUNT),
        "keep_anchored_lower_bounds": (count, EXPECTED_QUERY_COUNT, 3),
        "reconstruction_query_losses": (count, EXPECTED_QUERY_COUNT),
        "portfolio_actions": (count, EXPECTED_QUERY_COUNT),
        "portfolio_switch_mask": (count, EXPECTED_QUERY_COUNT),
        "portfolio_lower_bounds": (count, EXPECTED_QUERY_COUNT, 3),
        "fused_actions": (count, EXPECTED_QUERY_COUNT),
        "fused_route_agreement": (count, EXPECTED_QUERY_COUNT),
    }
    for role in S2A_ROLES:
        expected[f"rematched_{role}"] = (count,)
    if set(payload.files) != set(expected):
        missing = sorted(set(expected) - set(payload.files))
        extra = sorted(set(payload.files) - set(expected))
        raise ValueError(f"S2-A diagnostic schema mismatch: missing={missing}, extra={extra}")
    for key, shape in expected.items():
        if payload[key].shape != shape:
            raise ValueError(f"S2-A diagnostic shape mismatch: {key}")
    if _decode_ascii(payload["frame_token"]) != frame_tokens:
        raise ValueError("S2-A diagnostics lost frame order")
    identity = payload["query_identity"].astype(np.int64)
    canonical = np.arange(EXPECTED_QUERY_COUNT, dtype=np.int64)
    if not np.all(np.sort(identity, axis=2) == canonical[None, None, :]):
        raise ValueError("S2-A query identity is not four complete permutations")
    finite_keys = (
        "keep_query_losses",
        "full_context_query_losses",
        "keep_anchored_lower_bounds",
        "reconstruction_query_losses",
        "portfolio_lower_bounds",
    ) + tuple(f"rematched_{role}" for role in S2A_ROLES)
    if any(not np.isfinite(payload[key]).all() for key in finite_keys):
        raise ValueError("S2-A formal diagnostic contains non-finite loss")
    return {
        "keep_switch_count": int(payload["keep_anchored_switch_mask"].sum()),
        "portfolio_switch_count": int(payload["portfolio_switch_mask"].sum()),
        "fused_action_agreement_count": int(payload["fused_route_agreement"].sum()),
        "query_count": count * EXPECTED_QUERY_COUNT,
    }


def _validate_s2b_diagnostics(payload, frame_tokens: list[str]) -> dict:
    count = len(frame_tokens)
    expected = {
        "frame_token": (count,),
        "input_sha256": (count,),
        "base_routes": (count, EXPECTED_QUERY_COUNT),
        "final_routes": (count, EXPECTED_QUERY_COUNT),
        "base_rematched_group_loss": (count,),
        "final_rematched_group_loss": (count,),
    }
    if set(payload.files) != set(expected):
        raise ValueError("S2-B diagnostic schema mismatch")
    for key, shape in expected.items():
        if payload[key].shape != shape:
            raise ValueError(f"S2-B diagnostic shape mismatch: {key}")
    if _decode_ascii(payload["frame_token"]) != frame_tokens:
        raise ValueError("S2-B diagnostics lost frame order")
    for key in ("base_rematched_group_loss", "final_rematched_group_loss"):
        if not np.isfinite(payload[key]).all():
            raise ValueError("S2-B formal diagnostic contains non-finite frame loss")
    return {
        "route_change_count": int(
            np.count_nonzero(payload["base_routes"] != payload["final_routes"])
        ),
        "improved_frame_count": int(
            np.count_nonzero(
                payload["final_rematched_group_loss"]
                < payload["base_rematched_group_loss"]
            )
        ),
    }


def _validate_trace(path: Path, frame_tokens: list[str], epsilon_frame: float) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if [str(item.get("frame_token")) for item in frames] != frame_tokens:
        raise ValueError("S2-B trace frame order mismatch")
    accepted = 0
    for frame in frames:
        candidate_keys = [
            (int(item["query_id"]), int(item["destination_expert"]))
            for item in frame.get("candidate_order", [])
        ]
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError("S2-B trace contains duplicate candidates")
        ordered = sorted(
            frame.get("candidate_order", []),
            key=lambda item: (
                -float(item["quality"]),
                int(item["query_id"]),
                int(item["destination_expert"]),
            ),
        )
        if ordered != frame.get("candidate_order", []):
            raise ValueError("S2-B candidate ordering is unstable")
        if len(candidate_keys) > int(frame.get("K", -1)):
            raise ValueError("S2-B trace exceeds locked K")
        for step, row in enumerate(frame.get("trace", [])):
            if int(row["step"]) != step:
                raise ValueError("S2-B trace steps are not contiguous")
            if bool(row["accepted"]):
                gain = row.get("actual_frame_gain")
                if gain is None or not np.isfinite(float(gain)) or not float(gain) > epsilon_frame:
                    raise ValueError("S2-B accepted transition violates locked margin")
                accepted += 1
        if int(frame.get("accepted_count", -1)) != sum(
            int(row["accepted"]) for row in frame.get("trace", [])
        ):
            raise ValueError("S2-B frame accepted count drifted")
    return {"accepted_count": accepted}


def main() -> int:
    args = parse_args()
    if len(args.worker_manifest) != 2:
        raise ValueError("exactly two scene-shard worker manifests are required")
    if args.expected_scene_count <= 0 or args.expected_frame_count <= 0:
        raise ValueError("expected coverage counts must be positive")
    if args.official_evaluation and (
        args.expected_scene_count != 150 or args.expected_frame_count != 6019
    ):
        raise ValueError("official evaluation requires merged 150 scenes/6,019 frames")
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("S2 merge output must stay inside artifact-root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("S2 merge output cannot enter the source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable S2 merge exists: {output_dir}")
    scene_map = json.loads(args.scene_map.read_text(encoding="utf-8"))
    if (
        scene_map.get("token_set_sha256") != EXPECTED_TOKEN_SET_SHA256
        or int(scene_map.get("scene_count", -1)) != 150
        or int(scene_map.get("frame_count", -1)) != 6019
    ):
        raise ValueError("full validation scene map identity mismatch")
    roles = S2A_ROLES if args.mode == "s2a" else S2B_ROLES
    predictions = {}
    observed_scenes = set()
    worker_records = []
    locked = None
    diagnostic_counts = Counter()
    accepted_count = 0
    for manifest_path in args.worker_manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest.get("input_identity", {})
        if (
            manifest.get("status") != "complete"
            or identity.get("mode") != args.mode
            or identity.get("condition") != args.condition
            or tuple(manifest.get("roles", [])) != tuple(roles)
            or int(manifest.get("engineering_failure_count", -1)) != 0
            or int(identity.get("trainable_parameter_count", -1)) != 0
        ):
            raise ValueError(f"invalid S2 worker manifest: {manifest_path}")
        current_locked = _locked_identity(identity)
        if locked is None:
            locked = current_locked
        elif current_locked != locked:
            raise ValueError("S2 shards have different locked source/calibration identity")
        worker_records.append(
            {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)}
        )
        for scene in manifest.get("scenes", []):
            scene_token = str(scene["scene_token"])
            if scene_token in observed_scenes:
                raise ValueError("S2 shards overlap a scene")
            observed_scenes.add(scene_token)
            prediction_path = Path(scene["prediction_path"])
            diagnostic_path = Path(scene["diagnostic_path"])
            audit_path = Path(scene["corruption_audit_path"])
            for path, expected_hash in (
                (prediction_path, scene["prediction_sha256"]),
                (diagnostic_path, scene["diagnostic_sha256"]),
                (audit_path, scene["corruption_audit_sha256"]),
            ):
                if not path.is_file() or sha256_file(path) != expected_hash:
                    raise ValueError("S2 scene evidence hash mismatch")
            with prediction_path.open("rb") as stream:
                rows = pickle.load(stream)  # noqa: S301 - local run evidence.
            frame_tokens = [str(token) for token in scene["frame_tokens"]]
            if list(rows) != frame_tokens or any(tuple(row) != tuple(roles) for row in rows.values()):
                raise ValueError("S2 scene prediction roles/token order drifted")
            overlap = set(rows) & set(predictions)
            if overlap:
                raise ValueError(f"duplicate S2 prediction token: {sorted(overlap)[:3]}")
            predictions.update(rows)
            with np.load(diagnostic_path, allow_pickle=False) as payload:
                observed = (
                    _validate_s2a_diagnostics(payload, frame_tokens)
                    if args.mode == "s2a"
                    else _validate_s2b_diagnostics(payload, frame_tokens)
                )
                diagnostic_counts.update(observed)
            if args.mode == "s2b":
                trace_path = Path(scene["trace_path"])
                if not trace_path.is_file() or sha256_file(trace_path) != scene["trace_sha256"]:
                    raise ValueError("S2-B scene trace hash mismatch")
                accepted_count += _validate_trace(
                    trace_path,
                    frame_tokens,
                    float(identity["locked_margins"]["epsilon_frame"]),
                )["accepted_count"]
    if len(observed_scenes) != args.expected_scene_count:
        raise ValueError("merged S2 scene coverage does not match the locked count")
    if len(predictions) != args.expected_frame_count:
        raise ValueError("merged S2 frame coverage does not match the locked count")
    expected_tokens = set(scene_map["sample_to_scene"])
    if args.expected_frame_count == 6019 and set(predictions) != expected_tokens:
        raise ValueError("formal S2 merge does not exactly cover 6,019 unique tokens")
    output_dir.mkdir(parents=True)
    metrics = {}
    if args.official_evaluation:
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from mmcv import Config
        from mmdet3d.datasets import build_dataset

        cfg = Config.fromfile(str(args.config))
        _import_plugin(cfg, args.config)
        dataset_cfg = copy.deepcopy(cfg.data.test)
        dataset_cfg.data_root = str(args.data_root.resolve()).replace("\\", "/") + "/"
        dataset_cfg.ann_file = str(args.ann_file.resolve()).replace("\\", "/")
        dataset_cfg.test_mode = True
        dataset = build_dataset(dataset_cfg)
        ordered_tokens = [str(info["token"]) for info in dataset.data_infos]
        if len(ordered_tokens) != 6019 or set(ordered_tokens) != expected_tokens:
            raise ValueError("official evaluation dataset token identity mismatch")
        for role in roles:
            role_results = [{"pts_bbox": predictions[token][role]} for token in ordered_tokens]
            role_dir = output_dir / role
            raw = dataset.evaluate(
                role_results,
                jsonfile_prefix=str(role_dir),
                result_names=["pts_bbox"],
            )
            safe_raw, non_finite = _json_compatible(raw)
            metrics[role] = {
                "mAP": _metric(raw, "/mAP"),
                "NDS": _metric(raw, "/NDS"),
                "raw_metrics": safe_raw,
                "raw_metrics_non_finite_value_count": non_finite,
                "raw_metrics_non_finite_encoding": "json_null",
                "evidence_files": _evidence_tree(role_dir),
            }
    delta = {}
    if metrics and args.mode == "s2a":
        delta = {
            "keep_anchored_minus_original_mome": {
                key: metrics[S2A_ROLES[1]][key] - metrics[S2A_ROLES[0]][key]
                for key in ("mAP", "NDS")
            },
            "reconstruction_minus_original_mome": {
                key: metrics[S2A_ROLES[2]][key] - metrics[S2A_ROLES[0]][key]
                for key in ("mAP", "NDS")
            },
            "portfolio_minus_reconstruction": {
                key: metrics[S2A_ROLES[3]][key] - metrics[S2A_ROLES[2]][key]
                for key in ("mAP", "NDS")
            },
        }
    elif metrics:
        delta = {
            "greedy_final_minus_same_run_base": {
                key: metrics[S2B_ROLES[1]][key] - metrics[S2B_ROLES[0]][key]
                for key in ("mAP", "NDS")
            }
        }
    prediction_outputs = {}
    for role, payload in metrics.items():
        candidates = [
            record
            for record in payload["evidence_files"]
            if Path(record["path"]).name == "results_nusc.json"
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"official evaluation did not expose one canonical prediction file for {role}"
            )
        prediction_outputs[role] = candidates[0]
    manifest = {
        "schema": "visfuse3d_stage019_s2_condition_merge_v1",
        "status": "complete_official_evaluation" if metrics else "complete_diagnostic_merge",
        "mode": args.mode,
        "condition": args.condition,
        "scene_count": len(observed_scenes),
        "frame_count": len(predictions),
        "unique_token_count": len(set(predictions)),
        "roles": list(roles),
        "engineering_failure_count": 0,
        "diagnostic_counts": dict(diagnostic_counts),
        "accepted_count": accepted_count,
        "protocol_profile": (locked or {}).get("protocol_profile"),
        "metrics": metrics,
        "delta": delta,
        "prediction_outputs": prediction_outputs,
        "inputs": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "annotation": str(args.ann_file.resolve()),
            "annotation_sha256": sha256_file(args.ann_file),
            "scene_map": str(args.scene_map.resolve()),
            "scene_map_sha256": sha256_file(args.scene_map),
            "worker_manifests": worker_records,
            "locked_worker_identity": locked,
        },
        "claim_boundary": (
            "gt_informed_context_preserving_output_oracle_under_locked_matching_and_margins_not_metric_upper_bound"
            if args.mode == "s2a"
            else "locked_gt_guided_greedy_reachable_result_not_global_optimum"
        ),
    }
    atomic_write_json(output_dir / "condition_evaluation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
