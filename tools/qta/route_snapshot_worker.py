"""Capture complete frozen-MoME router snapshots for numerical-stability audit.

This helper is deliberately separate from the active strict-audit worker.  It
never trains, never evaluates nuScenes-R, and never decides whether a route
drift is scientifically acceptable.  It records all 900 original query ids,
the three original MoME router logits, default routes, reference points, and
input/corruption hashes so a later, pre-registered comparison can classify a
whole frame-condition as exact, numerical-boundary, or engineering failure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from .common import (
        BASELINE_ID,
        BASELINE_NAME,
        LOCAL_CONDITIONS,
        SEED,
        atomic_write_json,
        stable_digest,
    )
    from .extract_route_loss_cache import (
        EXPECTED_MOME_CHECKPOINT_SHA256,
        EXPECTED_SCENES,
        PARTITION_KEYS,
        _force_condition,
        _force_exact_frame_sampling,
        _force_identity_view,
        _import_plugin,
        _prepare_exact_sample,
        _sample_seed,
        _scene_indices,
        is_relative_to,
        sha256_file,
    )
except ImportError:
    from common import (
        BASELINE_ID,
        BASELINE_NAME,
        LOCAL_CONDITIONS,
        SEED,
        atomic_write_json,
        stable_digest,
    )
    from extract_route_loss_cache import (
        EXPECTED_MOME_CHECKPOINT_SHA256,
        EXPECTED_SCENES,
        PARTITION_KEYS,
        _force_condition,
        _force_exact_frame_sampling,
        _force_identity_view,
        _import_plugin,
        _prepare_exact_sample,
        _sample_seed,
        _scene_indices,
        is_relative_to,
        sha256_file,
    )


PROTOCOL = "mome_qta_complete_route_snapshot_v1"
EXPECTED_QUERY_COUNT = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--sample-scene-map", type=Path, required=True)
    parser.add_argument("--partition", choices=tuple(PARTITION_KEYS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--scored-cache", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scene-count", type=int)
    parser.add_argument("--scene-token")
    parser.add_argument("--frame-token")
    parser.add_argument("--max-frames-per-scene", type=int)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=LOCAL_CONDITIONS,
        default=list(LOCAL_CONDITIONS),
    )
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--repeat-id", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def select_snapshot_scenes(
    all_scenes: list[str],
    seed: int,
    scene_count: int | None,
    scene_token: str | None,
) -> list[str]:
    """Select scenes without consulting route gains or strict labels."""

    unique = sorted(set(str(value) for value in all_scenes))
    if len(unique) != len(all_scenes):
        raise ValueError("route-snapshot partition contains duplicate scene tokens")
    if scene_token is not None:
        if scene_token not in unique:
            raise ValueError("requested route-snapshot scene is outside the partition")
        if scene_count not in (None, 1):
            raise ValueError("scene-count must be one when scene-token is provided")
        return [scene_token]
    if scene_count is None or scene_count <= 0:
        raise ValueError("positive scene-count is required without scene-token")
    if scene_count > len(unique):
        raise ValueError("route-snapshot scene-count exceeds the partition")
    ranked = sorted(unique, key=lambda value: (stable_digest(value, seed), value))
    return ranked[:scene_count]


def _update_digest(digest: "hashlib._Hash", value) -> None:
    """Hash nested tensor-like values with explicit type, dtype, and shape tags."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        digest.update(b"array\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes())
        return
    if isinstance(value, np.generic):
        _update_digest(digest, value.item())
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: str(item)):
            _update_digest(digest, str(key))
            _update_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(("list" if isinstance(value, list) else "tuple").encode("ascii"))
        digest.update(b"\0")
        digest.update(str(len(value)).encode("ascii"))
        digest.update(b"\0")
        for item in value:
            _update_digest(digest, item)
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0")
        digest.update(value)
        return
    if isinstance(value, (str, bool, int, float)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
        return
    raise TypeError(f"unsupported route-snapshot hash value: {type(value)!r}")


def recursive_sha256(value) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest().upper()


def _metadata_subset(meta: dict, keys: tuple[str, ...]) -> dict:
    return {key: meta[key] for key in keys if key in meta}


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=str(path.parent)
    )
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_route_reference(path: Path | None):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as payload:
        required = {"frame_token", "condition", "query_index", "base_routes"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"scored cache lacks route-reference arrays: {sorted(missing)}")
        return {
            key: payload[key]
            for key in ("frame_token", "condition", "query_index", "base_routes")
        }


def _as_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values.reshape(-1).tolist()
        ]
    ).reshape(values.shape)


def retained_route_comparison(
    route_reference,
    frame_token: str,
    condition: str,
    routes: np.ndarray,
    scores: np.ndarray,
) -> dict:
    if route_reference is None:
        return {"available": False}
    frame = _as_strings(route_reference["frame_token"])
    conditions = _as_strings(route_reference["condition"])
    selected = (frame == frame_token) & (conditions == condition)
    query_ids = route_reference["query_index"][selected].astype(np.int64, copy=False)
    expected = route_reference["base_routes"][selected].astype(np.int64, copy=False)
    if query_ids.size != np.unique(query_ids).size:
        raise ValueError("scored cache repeats query ids within a frame-condition")
    if np.any((query_ids < 0) | (query_ids >= routes.shape[0])):
        raise ValueError("scored cache query id is outside the complete route vector")
    actual = routes[query_ids].astype(np.int64, copy=False)
    drift = actual != expected
    details = []
    for position in np.flatnonzero(drift).tolist():
        query_index = int(query_ids[position])
        order = np.argsort(-scores[query_index], kind="stable")
        details.append(
            {
                "query_index": query_index,
                "cached_route": int(expected[position]),
                "snapshot_route": int(actual[position]),
                "snapshot_top2_routes": [int(order[0]), int(order[1])],
                "snapshot_top2_margin": float(
                    scores[query_index, order[0]] - scores[query_index, order[1]]
                ),
                "snapshot_scores": [float(value) for value in scores[query_index]],
            }
        )
    return {
        "available": True,
        "retained_query_count": int(query_ids.size),
        "route_drift_count": int(drift.sum()),
        "route_drift_rate": float(drift.mean()) if drift.size else 0.0,
        "route_drift": details,
    }


def _normalize_router_state(result, selected_cls):
    features = result["router_features"]
    scores = selected_cls(features).detach().float().cpu().numpy()
    routes = result["base_routes"].detach().cpu().numpy()
    references = result["reference_points"].detach().float().cpu().numpy()
    if scores.ndim != 3 or scores.shape[0] != 1 or scores.shape[2] != 3:
        raise RuntimeError(f"unexpected MoME router score shape: {scores.shape}")
    if routes.shape != scores.shape[:2]:
        raise RuntimeError("MoME routes and router scores do not align")
    if routes.shape[1] != EXPECTED_QUERY_COUNT:
        raise RuntimeError(f"MoME route snapshot expected {EXPECTED_QUERY_COUNT} queries")
    if references.ndim == 2:
        references = references[None, ...]
    if references.shape[:2] != routes.shape:
        raise RuntimeError("MoME reference points and routes do not align")
    expected_routes = scores.argmax(axis=-1)
    if not np.array_equal(routes, expected_routes):
        raise RuntimeError("reported MoME routes differ from selected-cls argmax")
    return (
        routes[0].astype(np.int8, copy=False),
        scores[0].astype(np.float32, copy=False),
        references[0].astype(np.float32, copy=False),
    )


def main() -> int:
    args = parse_args()
    if args.seed != SEED:
        raise ValueError(f"route stability audit is locked to seed {SEED}")
    if args.max_frames_per_scene is not None and args.max_frames_per_scene <= 0:
        raise ValueError("max-frames-per-scene must be positive")
    source_root = Path(__file__).resolve().parents[2]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("route snapshot output must remain inside artifact-root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("route snapshots cannot be written into the source checkout")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != EXPECTED_MOME_CHECKPOINT_SHA256:
        raise ValueError("frozen MoME checkpoint SHA256 mismatch")

    split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    partition_scenes = [str(value) for value in split[PARTITION_KEYS[args.partition]]]
    if len(partition_scenes) != EXPECTED_SCENES[args.partition]:
        raise ValueError("route-snapshot partition scene count mismatch")
    requested = select_snapshot_scenes(
        partition_scenes, args.seed, args.scene_count, args.scene_token
    )
    sidecar = json.loads(args.sample_scene_map.read_text(encoding="utf-8"))
    if str(sidecar.get("annotation_file_sha256", "")).upper() != sha256_file(args.ann_file):
        raise ValueError("route-snapshot sidecar annotation hash mismatch")
    sample_to_scene = {str(key): str(value) for key, value in sidecar["sample_to_scene"].items()}
    route_reference = _load_route_reference(args.scored_cache)

    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg = Config.fromfile(str(args.config))
    _import_plugin(cfg, args.config)
    train_cfg = copy.deepcopy(cfg.data.train)
    if train_cfg.get("type") == "CBGSDataset":
        train_cfg = train_cfg.dataset
    train_cfg.data_root = str(args.data_root.resolve()).replace("\\", "/") + "/"
    train_cfg.ann_file = str(args.ann_file.resolve()).replace("\\", "/")
    train_cfg = _force_exact_frame_sampling(train_cfg)
    train_cfg = _force_identity_view(train_cfg)
    datasets = {
        condition: build_dataset(_force_condition(train_cfg, condition))
        for condition in args.conditions
    }
    reference_dataset = datasets[args.conditions[0]]
    grouped = _scene_indices(reference_dataset, set(requested), sample_to_scene)
    reference_tokens = [str(info["token"]) for info in reference_dataset.data_infos]
    for condition, dataset in datasets.items():
        if [str(info["token"]) for info in dataset.data_infos] != reference_tokens:
            raise RuntimeError(f"route-snapshot dataset order drifted for {condition}")

    cfg.model.pretrained = None
    set_random_seed(args.seed, deterministic=True)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", reference_dataset.CLASSES)
    model = model.cuda(args.gpu_id)
    model.eval()

    input_identity = {
        "protocol": PROTOCOL,
        "canonical_model_id": BASELINE_ID,
        "display_name": BASELINE_NAME,
        "partition": args.partition,
        "requested_scenes": requested,
        "conditions": list(args.conditions),
        "frame_token_filter": args.frame_token,
        "max_frames_per_scene": args.max_frames_per_scene,
        "seed": args.seed,
        "worker_id": args.worker_id,
        "repeat_id": args.repeat_id,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "local_cuda_id": args.gpu_id,
        "cuda_device_name": torch.cuda.get_device_name(args.gpu_id),
        "worker_source_sha256": sha256_file(Path(__file__).resolve()),
        "common_source_sha256": sha256_file(
            Path(__file__).with_name("common.py")
        ),
        "exact_sampling_source_sha256": sha256_file(
            Path(__file__).with_name("extract_route_loss_cache.py")
        ),
        "detector_source_sha256": sha256_file(
            source_root
            / "projects/mmdet3d_plugin/models/detectors/mome.py"
        ),
        "multi_expert_source_sha256": sha256_file(
            source_root
            / "projects/mmdet3d_plugin/models/utils/multi_expert.py"
        ),
        "med_source_sha256": sha256_file(
            source_root
            / "projects/mmdet3d_plugin/models/dense_heads/med.py"
        ),
        "config_sha256": sha256_file(args.config),
        "checkpoint_sha256": checkpoint_hash,
        "ann_file_sha256": sha256_file(args.ann_file),
        "scene_split_sha256": sha256_file(args.scene_split),
        "sample_scene_map_sha256": sha256_file(args.sample_scene_map),
        "scored_cache_sha256": (
            sha256_file(args.scored_cache) if args.scored_cache is not None else None
        ),
        "selection_rule": "explicit_scene_or_seeded_stable_digest_without_gain_labels",
    }
    progress_path = output_dir / "route_snapshot_progress.json"
    completed_scenes = {}
    records = []
    if args.resume and progress_path.exists():
        prior = json.loads(progress_path.read_text(encoding="utf-8"))
        prior_identity = dict(prior["input_identity"])
        prior_identity.pop("pid", None)
        current_identity = dict(input_identity)
        current_identity.pop("pid", None)
        if prior_identity != current_identity:
            raise ValueError("route-snapshot resume identity mismatch")
        for scene in prior.get("scenes", []):
            valid = all(
                Path(item["snapshot_path"]).is_file()
                and sha256_file(Path(item["snapshot_path"])) == item["snapshot_sha256"]
                for item in scene["frame_conditions"]
            )
            if not valid:
                raise ValueError("route-snapshot resume shard hash mismatch")
            completed_scenes[scene["scene_token"]] = scene
            records.extend(scene["frame_conditions"])
    progress = {
        "protocol": PROTOCOL,
        "status": "running_complete_route_snapshot",
        "input_identity": input_identity,
        "scenes": list(completed_scenes.values()),
        "completed_scene_count": len(completed_scenes),
        "frame_condition_count": len(records),
    }
    atomic_write_json(progress_path, progress)

    geometry_keys = (
        "lidar2img",
        "img_aug_matrix",
        "lidar_aug_matrix",
        "pcd_rotation",
        "pcd_scale_factor",
        "pcd_trans",
        "pcd_horizontal_flip",
        "pcd_vertical_flip",
        "img_shape",
        "pad_shape",
        "scale_factor",
        "flip",
    )
    corruption_keys = (
        "qta_corruption",
        "qta_route_available",
        "qta_hard_bypass",
    )
    query_ids = np.arange(EXPECTED_QUERY_COUNT, dtype=np.int16)
    found_target_frame = args.frame_token is None
    with torch.no_grad():
        for scene_token in requested:
            if scene_token in completed_scenes:
                continue
            scene_records = []
            scene_dataset_indices = grouped[scene_token]
            if args.max_frames_per_scene is not None:
                scene_dataset_indices = scene_dataset_indices[: args.max_frames_per_scene]
            for dataset_index in scene_dataset_indices:
                sample_seed = _sample_seed(scene_token, dataset_index, args.seed)
                for condition in args.conditions:
                    set_random_seed(sample_seed, deterministic=True)
                    sample = _prepare_exact_sample(
                        datasets[condition], dataset_index
                    )
                    batch = scatter(collate([sample], samples_per_gpu=1), [args.gpu_id])[0]
                    meta = batch["img_metas"][0]
                    frame_token = str(meta["sample_idx"])
                    observed_scene = sample_to_scene.get(frame_token)
                    if observed_scene != scene_token:
                        raise RuntimeError(
                            "route snapshot frame-to-scene identity drift: "
                            f"{frame_token} -> {observed_scene}, expected {scene_token}"
                        )
                    if args.frame_token is not None and frame_token != args.frame_token:
                        continue
                    found_target_frame = True
                    result = model.forward_qta_probe(
                        points=batch["points"],
                        img_metas=batch["img_metas"],
                        img=batch["img"],
                        gt_bboxes_3d=batch["gt_bboxes_3d"],
                        gt_labels_3d=batch["gt_labels_3d"],
                        return_query_losses=False,
                    )
                    routes, scores, references = _normalize_router_state(
                        result, model.pts_bbox_head.transformer.selected_cls
                    )
                    input_tensor_sha256 = recursive_sha256(
                        {"points": batch["points"], "img": batch["img"]}
                    )
                    corruption_sha256 = recursive_sha256(
                        {
                            "condition": condition,
                            "metadata": _metadata_subset(meta, corruption_keys),
                        }
                    )
                    geometry_sha256 = recursive_sha256(
                        _metadata_subset(meta, geometry_keys)
                    )
                    query_ids_sha256 = recursive_sha256(query_ids)
                    reference_points_sha256 = recursive_sha256(references)
                    frame_condition_identity_sha256 = recursive_sha256(
                        {
                            "scene_token": scene_token,
                            "frame_token": frame_token,
                            "condition": condition,
                            "sample_seed": sample_seed,
                            "input_tensor_sha256": input_tensor_sha256,
                            "corruption_sha256": corruption_sha256,
                            "geometry_sha256": geometry_sha256,
                            "query_ids_sha256": query_ids_sha256,
                            "reference_points_sha256": reference_points_sha256,
                        }
                    )
                    digest = hashlib.sha256(
                        f"{scene_token}|{frame_token}|{condition}".encode("utf-8")
                    ).hexdigest()[:20]
                    snapshot_path = output_dir / "snapshots" / f"frame_condition_{digest}.npz"
                    _atomic_savez(
                        snapshot_path,
                        query_ids=query_ids,
                        base_routes=routes,
                        router_scores=scores,
                        reference_points=references,
                    )
                    route_comparison = retained_route_comparison(
                        route_reference, frame_token, condition, routes, scores
                    )
                    record = {
                        "scene_token": scene_token,
                        "frame_token": frame_token,
                        "condition": condition,
                        "sample_seed": sample_seed,
                        "query_count": int(routes.size),
                        "snapshot_path": str(snapshot_path.resolve()),
                        "snapshot_sha256": sha256_file(snapshot_path),
                        "input_tensor_sha256": input_tensor_sha256,
                        "corruption_sha256": corruption_sha256,
                        "geometry_sha256": geometry_sha256,
                        "query_ids_sha256": query_ids_sha256,
                        "reference_points_sha256": reference_points_sha256,
                        "frame_condition_identity_sha256": frame_condition_identity_sha256,
                        "retained_route_comparison": route_comparison,
                    }
                    scene_records.append(record)
                    records.append(record)
                    progress.update(
                        {
                            "current_scene_token": scene_token,
                            "current_frame_token": frame_token,
                            "current_condition": condition,
                            "frame_condition_count": len(records),
                        }
                    )
                    atomic_write_json(progress_path, progress)
            scene_record = {
                "scene_token": scene_token,
                "frame_conditions": scene_records,
            }
            completed_scenes[scene_token] = scene_record
            progress["scenes"] = list(completed_scenes.values())
            progress["completed_scene_count"] = len(completed_scenes)
            atomic_write_json(progress_path, progress)

    if not found_target_frame:
        raise ValueError("requested route-snapshot frame is outside the selected scene")
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete_route_snapshot_not_boundary_classification_not_gate_result",
        "canonical_model_id": BASELINE_ID,
        "display_name": BASELINE_NAME,
        "input_identity": input_identity,
        "scenes": list(completed_scenes.values()),
        "frame_condition_count": len(records),
        "query_count_per_frame_condition": EXPECTED_QUERY_COUNT,
        "claim_boundary": (
            "router_snapshot_only_no_numerical_boundary_decision_no_scientific_metric"
        ),
    }
    manifest_path = output_dir / "route_snapshot_manifest.json"
    atomic_write_json(manifest_path, manifest)
    progress["status"] = "complete_route_snapshot"
    progress["manifest"] = str(manifest_path.resolve())
    atomic_write_json(progress_path, progress)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "scenes": len(completed_scenes),
                "frame_conditions": len(records),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
