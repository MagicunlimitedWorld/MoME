"""Extract frozen MoME route-loss shards for an authorized Stage019 run.

This detector-runtime tool is intentionally not invoked by the repository
smoke tests.  It reads only the selected nuScenes train scenes, writes one NPZ
shard per scene to the caller's artifact root, and performs no optimization or
official evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from .common import (
        BASELINE_ID,
        BASELINE_NAME,
        CANDIDATE_ID,
        CANDIDATE_NAME,
        CONDITIONS,
        SEED,
        atomic_write_json,
    )
except ImportError:
    from common import (
        BASELINE_ID,
        BASELINE_NAME,
        CANDIDATE_ID,
        CANDIDATE_NAME,
        CONDITIONS,
        SEED,
        atomic_write_json,
    )


EXPECTED_MOME_CHECKPOINT_SHA256 = (
    "72B863D9EA9CA63C4A2B6A5968620A7247A918E52C8BC876080DB3896881F5A2"
)
PARTITION_KEYS = {
    "stage019a_training": "stage019a_training_scenes",
    "stage019a_test": "stage019a_test_scenes",
    "proper_training": "proper_training_scenes",
    "calibration": "calibration_scenes",
}
EXPECTED_SCENES = {
    "stage019a_training": 80,
    "stage019a_test": 20,
    "proper_training": 560,
    "calibration": 140,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest().upper()


def _sample_seed(scene_token: str, dataset_index: int, seed: int) -> int:
    """Return the locked per-frame seed shared by extraction and strict replay."""

    return int.from_bytes(
        hashlib.sha256(
            f"{scene_token}|{dataset_index}|{seed}".encode("utf-8")
        ).digest()[:4],
        "little",
    )


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--scene-count", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--worker-id", default="worker0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--identity-view", action="store_true")
    parser.add_argument("--negative-queries-per-frame", type=int, default=64)
    return parser.parse_args()


def _import_plugin(cfg, config_path: Path) -> None:
    if not getattr(cfg, "plugin", False):
        return
    source_root = Path(__file__).resolve().parents[2]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    plugin_dir = getattr(cfg, "plugin_dir", str(config_path.parent))
    module_parts = os.path.dirname(plugin_dir).replace("\\", "/").split("/")
    module_path = ".".join(part for part in module_parts if part)
    importlib.import_module(module_path)


def _scene_indices(
    dataset,
    requested_scenes: set[str],
    sample_to_scene: dict[str, str],
) -> dict[str, list[int]]:
    grouped = defaultdict(list)
    for index, info in enumerate(dataset.data_infos):
        sample_token = str(info.get("token", ""))
        sidecar_scene = sample_to_scene.get(sample_token)
        if sidecar_scene is None:
            raise ValueError(
                f"sample-scene sidecar lacks dataset sample token: {sample_token}"
            )
        # The sidecar is generated from the NuScenes scene/sample tables and is
        # hash-bound to the annotation input.  Some legacy train-info rows carry
        # a stale scene_token, so using that field first can silently mix scenes
        # and partitions even though the frame token remains valid.
        scene_token = str(sidecar_scene)
        if scene_token in requested_scenes:
            grouped[scene_token].append(index)
    missing = requested_scenes - set(grouped)
    if missing:
        raise ValueError(f"dataset lacks requested train scenes: {sorted(missing)[:5]}")
    return dict(grouped)


def _force_exact_frame_sampling(dataset_cfg):
    """Keep empty-GT frames and forbid the training dataset's random fallback."""

    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg["filter_empty_gt"] = False
    return dataset_cfg


def _prepare_exact_sample(dataset, dataset_index: int):
    sample = dataset.prepare_train_data(dataset_index)
    if sample is None:
        raise RuntimeError(
            f"exact Stage019 frame preparation returned None at index {dataset_index}"
        )
    return sample


def _force_condition(train_cfg, condition: str):
    condition_cfg = copy.deepcopy(train_cfg)
    matches = [
        item
        for item in condition_cfg.pipeline
        if item.get("type") == "QtaLocalCorruption3D"
    ]
    if len(matches) != 1:
        raise ValueError("QTA pipeline must contain exactly one local corruption")
    matches[0]["forced_condition"] = condition
    return condition_cfg


def _force_identity_view(train_cfg):
    """Remove stochastic geometry while retaining GT for audit/calibration."""

    identity_cfg = copy.deepcopy(train_cfg)
    for item in identity_cfg.pipeline:
        if item.get("type") == "GlobalRotScaleTransAll":
            item["rot_range"] = [0.0, 0.0]
            item["scale_ratio_range"] = [1.0, 1.0]
            item["translation_std"] = [0.0, 0.0, 0.0]
        elif item.get("type") == "CustomRandomFlip3D":
            item["flip_ratio_bev_horizontal"] = 0.0
            item["flip_ratio_bev_vertical"] = 0.0
        elif item.get("type") == "ResizeCropFlipImage":
            item["training"] = False
    return identity_cfg


def _to_numpy(tensor, dtype=None):
    array = tensor.detach().cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def _repeat_string(value: str, rows: int) -> np.ndarray:
    return np.full(rows, value.encode("ascii"), dtype=f"S{max(1, len(value))}")


def _retained_query_indices(result, condition: str, sample_seed: int, negative_count: int):
    if negative_count < 0:
        raise ValueError("negative-queries-per-frame cannot be negative")
    positive = _to_numpy(result["positive_mask"][0], bool)
    affected = _to_numpy(result["affected_query"][0], bool)
    mandatory = positive.copy()
    if condition in CONDITIONS[3:]:
        mandatory |= affected
    candidates = np.flatnonzero(~mandatory)
    if candidates.size and negative_count:
        condition_seed = int.from_bytes(
            hashlib.sha256(f"{sample_seed}|{condition}|query-sample".encode("utf-8")).digest()[:4],
            "little",
        )
        rng = np.random.RandomState(condition_seed)
        selected = rng.choice(
            candidates, size=min(negative_count, candidates.size), replace=False
        )
        mandatory[selected] = True
    retained = np.flatnonzero(mandatory)
    if retained.size == 0:
        raise RuntimeError("QTA query sampling retained no queries")
    return retained


def _save_scene_shard(path: Path, arrays: dict[str, list[np.ndarray]]) -> int:
    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=str(path.parent)
    )
    os.close(handle)
    try:
        np.savez_compressed(temp_name, **merged)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return int(merged["base_routes"].shape[0])


def main() -> int:
    args = parse_args()
    if args.seed != SEED:
        raise ValueError(f"Stage019 first seed is locked to {SEED}")
    source_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("output-dir must be inside the declared artifact-root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("route-loss shards cannot be written into the source checkout")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != EXPECTED_MOME_CHECKPOINT_SHA256:
        raise ValueError(
            f"frozen MoME checkpoint SHA256 mismatch: {checkpoint_hash}"
        )

    split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    requested = list(split[PARTITION_KEYS[args.partition]])
    if len(requested) != EXPECTED_SCENES[args.partition]:
        raise ValueError(
            f"{args.partition} must contain {EXPECTED_SCENES[args.partition]} scenes"
        )
    if args.scene_offset < 0:
        raise ValueError("scene-offset cannot be negative")
    if args.scene_count is not None and args.scene_count <= 0:
        raise ValueError("scene-count must be positive")
    if args.max_scenes is not None and args.scene_count is not None:
        raise ValueError("use only one of --scene-count and --max-scenes")
    requested = requested[args.scene_offset :]
    selected_count = args.scene_count
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise ValueError("max-scenes must be positive")
        selected_count = args.max_scenes
    if selected_count is not None:
        requested = requested[:selected_count]
    if not requested:
        raise ValueError("worker scene slice is empty")

    sample_scene_payload = json.loads(
        args.sample_scene_map.read_text(encoding="utf-8")
    )
    if sample_scene_payload.get("protocol") != "mome_qta_sample_scene_sidecar_v1":
        raise ValueError("unsupported sample-scene sidecar protocol")
    if (
        int(sample_scene_payload.get("sample_count", -1)) != 28130
        or int(sample_scene_payload.get("scene_count", -1)) != 700
    ):
        raise ValueError("sample-scene sidecar is not the full nuScenes train split")
    if str(sample_scene_payload.get("annotation_file_sha256", "")).upper() != sha256_file(
        args.ann_file
    ):
        raise ValueError("sample-scene sidecar annotation hash mismatch")
    sample_to_scene = {
        str(key): str(value)
        for key, value in sample_scene_payload["sample_to_scene"].items()
    }

    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg = Config.fromfile(str(args.config))
    _import_plugin(cfg, args.config)
    set_random_seed(args.seed, deterministic=True)
    train_cfg = copy.deepcopy(cfg.data.train)
    if train_cfg.get("type") == "CBGSDataset":
        train_cfg = train_cfg.dataset
    train_cfg.data_root = str(args.data_root.resolve()).replace("\\", "/") + "/"
    train_cfg.ann_file = str(args.ann_file.resolve()).replace("\\", "/")
    train_cfg = _force_exact_frame_sampling(train_cfg)
    if args.identity_view:
        train_cfg = _force_identity_view(train_cfg)
    ann_file = str(train_cfg.ann_file).lower().replace("\\", "/")
    if "train" not in ann_file or "val" in Path(ann_file).name.lower():
        raise ValueError(f"QTA extraction requires a train annotation file: {ann_file}")
    pipeline_types = [item["type"] for item in train_cfg.pipeline]
    if "QtaLocalCorruption3D" not in pipeline_types or "ObjectPaste" in pipeline_types:
        raise ValueError("QTA train pipeline/corruption contract mismatch")
    datasets = {
        condition: build_dataset(_force_condition(train_cfg, condition))
        for condition in CONDITIONS
    }
    dataset = datasets[CONDITIONS[0]]
    grouped_indices = _scene_indices(dataset, set(requested), sample_to_scene)
    reference_tokens = [str(info.get("token", "")) for info in dataset.data_infos]
    for condition, condition_dataset in datasets.items():
        tokens = [str(info.get("token", "")) for info in condition_dataset.data_infos]
        if tokens != reference_tokens:
            raise RuntimeError(f"dataset ordering drifted for condition {condition}")

    cfg.model.pretrained = None
    model = build_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model = model.cuda(args.gpu_id)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "route_loss_progress.json"
    input_identity = {
        "partition": args.partition,
        "seed": args.seed,
        "worker_id": str(args.worker_id),
        "scene_offset": args.scene_offset,
        "scene_count": len(requested),
        "requested_scenes": requested,
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": sha256_file(args.config),
        "data_root": str(args.data_root.resolve()),
        "ann_file": str(args.ann_file.resolve()),
        "ann_file_sha256": sha256_file(args.ann_file),
        "identity_view": bool(args.identity_view),
        "negative_queries_per_frame": args.negative_queries_per_frame,
        "scene_split_sha256": sha256_file(args.scene_split),
        "sample_scene_map_sha256": sha256_file(args.sample_scene_map),
    }
    completed = {}
    if args.resume and progress_path.exists():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        if previous.get("input_identity") != input_identity:
            raise ValueError("resume progress input identity mismatch")
        completed = {
            str(record["scene_token"]): record
            for record in previous.get("shards", [])
        }
    progress = {
        "status": "running_route_loss_extraction",
        "protocol": "mome_qta_route_loss_progress_v1",
        "input_identity": input_identity,
        "shards": list(completed.values()),
    }
    atomic_write_json(progress_path, progress)
    condition_counts = Counter()
    condition_frame_counts = Counter()
    total_rows = 0
    processed_frames = 0
    shard_records = []
    with torch.no_grad():
        for scene_token in requested:
            prior = completed.get(scene_token)
            if prior is not None:
                prior_path = Path(prior["path"])
                if prior_path.exists() and sha256_file(prior_path) == str(prior["sha256"]).upper():
                    shard_records.append(prior)
                    total_rows += int(prior["rows"])
                    prior_counts = prior.get("condition_counts", {})
                    condition_counts.update(
                        {key: int(value) for key, value in prior_counts.items()}
                    )
                    prior_frames = prior.get("condition_frame_counts", {})
                    condition_frame_counts.update(
                        {key: int(value) for key, value in prior_frames.items()}
                    )
                    continue
                raise ValueError(
                    f"resume shard missing or hash-mismatched for scene {scene_token}"
                )
            scene_arrays = defaultdict(list)
            scene_condition_counts = Counter()
            scene_dataset_indices = grouped_indices[scene_token]
            for frame_position, dataset_index in enumerate(scene_dataset_indices, start=1):
                sample_seed = _sample_seed(scene_token, dataset_index, args.seed)
                for condition in CONDITIONS:
                    # The six conditions see the same deterministic augmentation.
                    set_random_seed(sample_seed, deterministic=True)
                    sample = _prepare_exact_sample(
                        datasets[condition], dataset_index
                    )
                    batch = collate([sample], samples_per_gpu=1)
                    # MMCV 1.6 on the frozen MoME stack indexes its stream
                    # table with integer CUDA ids (not ``torch.device``).
                    batch = scatter(batch, [args.gpu_id])[0]
                    result = model(
                        return_loss=True,
                        return_route_bundle=True,
                        **batch,
                    )
                    meta = batch["img_metas"][0]
                    observed_condition = str(meta["qta_corruption"]["condition"])
                    if observed_condition != condition:
                        raise RuntimeError(
                            f"forced QTA condition drift: {observed_condition} != {condition}"
                        )
                    retained = _retained_query_indices(
                        result,
                        condition,
                        sample_seed,
                        args.negative_queries_per_frame,
                    )
                    rows = int(retained.size)
                    condition_counts[condition] += rows
                    condition_frame_counts[condition] += 1
                    scene_condition_counts[condition] += rows
                    frame_token = str(meta["sample_idx"])
                    observed_scene = sample_to_scene.get(frame_token)
                    if observed_scene != scene_token:
                        raise RuntimeError(
                            "frame-to-scene identity drift after dataset loading: "
                            f"{frame_token} -> {observed_scene}, expected {scene_token}"
                        )
                    scene_arrays["router_features"].append(
                        _to_numpy(result["router_features"][0], np.float16)[retained]
                    )
                    scene_arrays["reference_points"].append(
                        _to_numpy(result["reference_points"][0], np.float32)[retained]
                    )
                    scene_arrays["base_routes"].append(
                        _to_numpy(result["base_routes"][0], np.int8)[retained]
                    )
                    scene_arrays["query_index"].append(
                        retained.astype(np.int16, copy=False)
                    )
                    scene_arrays["base_query_losses"].append(
                        _to_numpy(result["base_query_losses"][0], np.float32)[retained]
                    )
                    scene_arrays["route_query_losses"].append(
                        _to_numpy(result["route_query_losses"][0], np.float32)[retained]
                    )
                    scene_arrays["valid_mask"].append(
                        _to_numpy(result["valid_mask"][0], bool)[retained]
                    )
                    scene_arrays["positive_mask"].append(
                        _to_numpy(result["positive_mask"][0], bool)[retained]
                    )
                    scene_arrays["affected_query"].append(
                        _to_numpy(result["affected_query"][0], bool)[retained]
                    )
                    scene_arrays["lidar_point_count"].append(
                        _to_numpy(result["lidar_point_count"][0], np.float32)[retained]
                    )
                    scene_arrays["valid_camera_views"].append(
                        _to_numpy(result["valid_camera_views"][0], np.float32)[retained]
                    )
                    scene_arrays["row_weight"].append(
                        np.full(rows, 1.0 / rows, dtype=np.float32)
                    )
                    scene_arrays["scene_token"].append(_repeat_string(scene_token, rows))
                    scene_arrays["frame_token"].append(_repeat_string(frame_token, rows))
                    scene_arrays["source_split"].append(_repeat_string("train", rows))
                    scene_arrays["partition"].append(_repeat_string(args.partition, rows))
                    scene_arrays["condition"].append(_repeat_string(condition, rows))
                    scene_arrays["canonical_model_id"].append(
                        _repeat_string(CANDIDATE_ID, rows)
                    )
                    scene_arrays["display_name"].append(
                        _repeat_string(CANDIDATE_NAME, rows)
                    )
                    scene_arrays["frozen_baseline_canonical_model_id"].append(
                        _repeat_string(BASELINE_ID, rows)
                    )
                    scene_arrays["frozen_baseline_display_name"].append(
                        _repeat_string(BASELINE_NAME, rows)
                    )
                processed_frames += 1
                if frame_position == 1 or frame_position % 5 == 0:
                    progress.update(
                        {
                            "current_scene_token": scene_token,
                            "current_scene_frame": frame_position,
                            "current_scene_frames_total": len(scene_dataset_indices),
                            "processed_frames_in_worker": processed_frames,
                        }
                    )
                    atomic_write_json(progress_path, progress)
            shard_name = hashlib.sha256(scene_token.encode("utf-8")).hexdigest()[:16]
            shard_path = output_dir / "shards" / f"scene_{shard_name}.npz"
            scene_rows = _save_scene_shard(shard_path, scene_arrays)
            total_rows += scene_rows
            record = {
                    "scene_token": scene_token,
                    "rows": scene_rows,
                    "path": str(shard_path.resolve()),
                    "sha256": sha256_file(shard_path),
                    "condition_counts": dict(sorted(scene_condition_counts.items())),
                    "condition_frame_counts": {
                        key: len(grouped_indices[scene_token]) for key in CONDITIONS
                    },
                }
            shard_records.append(record)
            progress["shards"] = shard_records
            progress["completed_scene_count"] = len(shard_records)
            progress["query_rows"] = total_rows
            atomic_write_json(progress_path, progress)

    if set(condition_counts) != set(CONDITIONS):
        raise RuntimeError(f"not all six QTA conditions were observed: {condition_counts}")
    if len(set(condition_frame_counts.values())) != 1:
        raise RuntimeError(
            f"QTA condition frame counts are not exactly balanced: {condition_frame_counts}"
        )
    manifest = {
        "canonical_model_id": CANDIDATE_ID,
        "display_name": CANDIDATE_NAME,
        "frozen_baseline_canonical_model_id": BASELINE_ID,
        "frozen_baseline_display_name": BASELINE_NAME,
        "status": "route_loss_extraction_complete_not_trained_not_evaluated",
        "protocol": "mome_qta_route_loss_extraction_v1",
        "partition": args.partition,
        "worker_id": str(args.worker_id),
        "scene_offset": args.scene_offset,
        "seed": args.seed,
        "scene_count": len(requested),
        "query_rows": total_rows,
        "condition_counts": dict(sorted(condition_counts.items())),
        "condition_frame_counts": dict(sorted(condition_frame_counts.items())),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_hash,
        },
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "data_root": str(args.data_root.resolve()),
        "ann_file": str(args.ann_file.resolve()),
        "ann_file_sha256": sha256_file(args.ann_file),
        "identity_view": bool(args.identity_view),
        "negative_queries_per_frame": args.negative_queries_per_frame,
        "query_sampling": "all_positive_all_local_affected_plus_deterministic_background",
        "scene_split": str(args.scene_split.resolve()),
        "scene_split_sha256": sha256_file(args.scene_split),
        "sample_scene_map": str(args.sample_scene_map.resolve()),
        "sample_scene_map_sha256": sha256_file(args.sample_scene_map),
        "requested_scene_sha256": hashlib.sha256(
            "\n".join(requested).encode("utf-8")
        ).hexdigest(),
        "shards": shard_records,
        "data_boundary": "nuScenes train scenes only",
        "formal_nuscenes_r_used": False,
        "claim_boundary": "offline_loss_cache_only_no_training_or_metric_claim",
    }
    atomic_write_json(output_dir / "route_loss_manifest.json", manifest)
    progress["status"] = "route_loss_extraction_complete"
    progress["completed_scene_count"] = len(shard_records)
    progress["query_rows"] = total_rows
    progress["manifest"] = str((output_dir / "route_loss_manifest.json").resolve())
    atomic_write_json(progress_path, progress)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "scene_count": manifest["scene_count"],
                "query_rows": manifest["query_rows"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
