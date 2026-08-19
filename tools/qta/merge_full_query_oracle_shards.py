"""Merge two Stage019-S1 scene shards and run official nuScenes evaluation."""

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
    from .extract_route_loss_cache import _import_plugin, sha256_file
    from .full_query_oracle_worker import (
        CONDITIONS,
        EXPECTED_QUERY_COUNT,
        EXPECTED_TOKEN_SET_SHA256,
        PROTOCOL,
    )
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import _import_plugin, sha256_file
    from full_query_oracle_worker import (
        CONDITIONS,
        EXPECTED_QUERY_COUNT,
        EXPECTED_TOKEN_SET_SHA256,
        PROTOCOL,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--scene-map", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--worker-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _metric(metrics: dict, suffix: str) -> float:
    matches = [float(value) for key, value in metrics.items() if str(key).endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"official evaluator did not return one {suffix} metric")
    return matches[0]


def _evidence_tree(root: Path) -> list[dict]:
    return [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _json_compatible(value):
    """Encode official undefined metrics as JSON null without changing scores."""

    if isinstance(value, dict):
        converted = {}
        non_finite_count = 0
        for key, item in value.items():
            safe_item, item_count = _json_compatible(item)
            converted[str(key)] = safe_item
            non_finite_count += item_count
        return converted, non_finite_count
    if isinstance(value, (list, tuple)):
        converted = []
        non_finite_count = 0
        for item in value:
            safe_item, item_count = _json_compatible(item)
            converted.append(safe_item)
            non_finite_count += item_count
        return converted, non_finite_count
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None, 1
    return value, 0


def main() -> int:
    args = parse_args()
    if len(args.worker_manifest) != 2:
        raise ValueError("exactly two 75-scene worker manifests are required")
    scene_map = json.loads(args.scene_map.read_text(encoding="utf-8"))
    if (
        scene_map.get("token_set_sha256") != EXPECTED_TOKEN_SET_SHA256
        or int(scene_map.get("scene_count", -1)) != 150
        or int(scene_map.get("frame_count", -1)) != 6019
    ):
        raise ValueError("fullval scene map identity mismatch")
    expected_scenes = set(scene_map["scene_tokens"])
    expected_tokens = set(scene_map["sample_to_scene"])
    predictions: dict[str, dict] = {}
    observed_scenes = set()
    worker_records = []
    switch_count = 0
    route_allocations = Counter()
    rematched = Counter()
    diagnostic_frames = 0
    observed_offsets = set()
    observed_cvd = set()
    locked_identity = None
    locked_identity_keys = (
        "protocol",
        "experiment_id",
        "run_id",
        "condition",
        "seed",
        "config_sha256",
        "checkpoint_sha256",
        "annotation_file_sha256",
        "scene_map_sha256",
        "sidecar_sha256",
        "adapter_source_sha256",
        "condition_source_sha256",
        "detector_source_sha256",
        "med_source_sha256",
        "router_source_sha256",
        "worker_source_sha256",
        "source_bundle_sha256",
        "filter_empty_gt",
        "sample_access",
        "point_shuffle",
        "random_geometry",
        "trainable_parameter_count",
    )
    for manifest_path in args.worker_manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest.get("input_identity", {})
        if (
            manifest.get("status") != "complete"
            or identity.get("protocol") != PROTOCOL
            or identity.get("condition") != args.condition
            or int(identity.get("scene_count", -1)) != 75
            or int(manifest.get("query_decision_count", -1))
            != int(manifest.get("frame_count", -1)) * EXPECTED_QUERY_COUNT
        ):
            raise ValueError(f"invalid fullval Oracle worker manifest: {manifest_path}")
        offset = int(identity.get("scene_offset", -1))
        cvd = str(identity.get("cuda_visible_devices"))
        requested_scenes = list(identity.get("requested_scenes", []))
        if (
            offset not in (0, 75)
            or requested_scenes != list(scene_map["scene_tokens"])[offset : offset + 75]
            or int(identity.get("trainable_parameter_count", -1)) != 0
            or identity.get("annotation_file_sha256") != sha256_file(args.ann_file)
            or identity.get("scene_map_sha256") != sha256_file(args.scene_map)
        ):
            raise ValueError(f"worker input identity mismatch: {manifest_path}")
        expected_cvd = "0" if offset == 0 else "1"
        if cvd != expected_cvd:
            raise ValueError(f"worker CVD/scene shard mapping mismatch: {manifest_path}")
        observed_offsets.add(offset)
        observed_cvd.add(cvd)
        current_locked_identity = {
            key: identity.get(key) for key in locked_identity_keys
        }
        if locked_identity is None:
            locked_identity = current_locked_identity
        elif current_locked_identity != locked_identity:
            raise ValueError("two Oracle workers have different locked input/source identity")
        worker_records.append(
            {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)}
        )
        for scene in manifest["scenes"]:
            scene_token = str(scene["scene_token"])
            if scene_token in observed_scenes:
                raise ValueError("worker manifests overlap a validation scene")
            observed_scenes.add(scene_token)
            prediction_path = Path(scene["prediction_path"])
            diagnostic_path = Path(scene["diagnostic_path"])
            if (
                sha256_file(prediction_path) != scene["prediction_sha256"]
                or sha256_file(diagnostic_path) != scene["diagnostic_sha256"]
                or sha256_file(Path(scene["corruption_audit_path"]))
                != scene["corruption_audit_sha256"]
            ):
                raise ValueError("scene evidence hash mismatch")
            with prediction_path.open("rb") as stream:
                rows = pickle.load(stream)  # noqa: S301 - run-local detector output.
            expected_frame_tokens = [str(value) for value in scene["frame_tokens"]]
            if set(rows) != set(expected_frame_tokens) or any(
                set(value) != {"base", "oracle"} for value in rows.values()
            ):
                raise ValueError("scene prediction/frame-token identity mismatch")
            overlap = set(rows) & set(predictions)
            if overlap:
                raise ValueError(f"duplicate prediction tokens: {sorted(overlap)[:3]}")
            predictions.update(rows)
            with np.load(diagnostic_path, allow_pickle=False) as payload:
                diagnostic_tokens = [
                    value.decode("ascii")
                    for value in payload["frame_token"].reshape(-1).tolist()
                ]
                frame_count = len(expected_frame_tokens)
                if diagnostic_tokens != expected_frame_tokens:
                    raise ValueError("scene diagnostics frame order mismatch")
                if payload["input_sha256"].shape != (frame_count,):
                    raise ValueError("scene diagnostics input hashes lost frame alignment")
                if payload["route_query_losses"].shape != (
                    frame_count,
                    EXPECTED_QUERY_COUNT,
                    3,
                ):
                    raise ValueError("scene diagnostics lost the 900x3 route-loss matrix")
                if payload["base_routes"].shape != (frame_count, EXPECTED_QUERY_COUNT):
                    raise ValueError("scene diagnostics lost the 900-query route vector")
                for key in ("oracle_routes", "query_gains", "switch_mask"):
                    if payload[key].shape != (frame_count, EXPECTED_QUERY_COUNT):
                        raise ValueError(f"scene diagnostics lost {key} alignment")
                diagnostic_frames += int(payload["base_routes"].shape[0])
                switch_count += int(payload["switch_mask"].sum())
                route_allocations.update(
                    int(value) for value in payload["oracle_routes"].reshape(-1).tolist()
                )
                base_loss = payload["base_rematched_group_loss"]
                oracle_loss = payload["oracle_rematched_group_loss"]
                rematched["improved"] += int(np.count_nonzero(oracle_loss < base_loss))
                rematched["equal"] += int(np.count_nonzero(oracle_loss == base_loss))
                rematched["worse"] += int(np.count_nonzero(oracle_loss > base_loss))
    if observed_offsets != {0, 75} or observed_cvd != {"0", "1"}:
        raise ValueError("two Oracle workers do not cover the locked CVD/offset pair")
    if observed_scenes != expected_scenes:
        raise ValueError("two Oracle workers do not exactly cover all 150 validation scenes")
    if set(predictions) != expected_tokens or len(predictions) != 6019:
        raise ValueError("two Oracle workers do not exactly cover all 6019 validation frames")
    if diagnostic_frames != 6019:
        raise ValueError("diagnostic frame coverage is not exactly 6019")

    source_root = Path(__file__).resolve().parents[2]
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
    if set(ordered_tokens) != expected_tokens or len(ordered_tokens) != 6019:
        raise ValueError("evaluation dataset token identity mismatch")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    role_metrics = {}
    for role in ("base", "oracle"):
        role_results = [
            {"pts_bbox": predictions[token][role]} for token in ordered_tokens
        ]
        role_dir = output_dir / role
        metrics = dataset.evaluate(
            role_results,
            jsonfile_prefix=str(role_dir),
            result_names=["pts_bbox"],
        )
        json_metrics, non_finite_count = _json_compatible(metrics)
        role_metrics[role] = {
            "mAP": _metric(metrics, "/mAP"),
            "NDS": _metric(metrics, "/NDS"),
            "raw_metrics": json_metrics,
            "raw_metrics_non_finite_value_count": non_finite_count,
            "raw_metrics_non_finite_encoding": "json_null",
            "evidence_files": _evidence_tree(role_dir),
        }
    manifest = {
        "schema": "visfuse3d_stage019_s1_condition_evaluation_v1",
        "status": "complete",
        "protocol": PROTOCOL,
        "condition": args.condition,
        "scene_count": 150,
        "frame_count": 6019,
        "query_decision_count": 6019 * EXPECTED_QUERY_COUNT,
        "route_loss_value_count": 6019 * EXPECTED_QUERY_COUNT * 3,
        "switch_count": switch_count,
        "switch_rate": switch_count / (6019 * EXPECTED_QUERY_COUNT),
        "oracle_route_allocations": {
            str(key): value for key, value in sorted(route_allocations.items())
        },
        "rematched_frame_counts": dict(rematched),
        "metrics": role_metrics,
        "delta": {
            "mAP": role_metrics["oracle"]["mAP"] - role_metrics["base"]["mAP"],
            "NDS": role_metrics["oracle"]["NDS"] - role_metrics["base"]["NDS"],
        },
        "inputs": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "annotation_file": str(args.ann_file.resolve()),
            "annotation_file_sha256": sha256_file(args.ann_file),
            "scene_map": str(args.scene_map.resolve()),
            "scene_map_sha256": sha256_file(args.scene_map),
            "workers": worker_records,
            "locked_worker_identity": locked_identity,
        },
        "claim_boundary": "official_nuscenes_gt_informed_empirical_upper_bound_not_deployable_method",
    }
    atomic_write_json(output_dir / "condition_evaluation_manifest.json", manifest)
    print(json.dumps({"status": "complete", "condition": args.condition, **manifest["delta"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
