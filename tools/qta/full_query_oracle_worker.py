"""Run one immutable Stage019-S1 condition/scene shard on frozen MoME."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import (
        EXPECTED_MOME_CHECKPOINT_SHA256,
        _force_exact_frame_sampling,
        _force_identity_view,
        _import_plugin,
        _prepare_exact_sample,
        is_relative_to,
        sha256_file,
    )
    from .route_snapshot_worker import recursive_sha256
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import (
        EXPECTED_MOME_CHECKPOINT_SHA256,
        _force_exact_frame_sampling,
        _force_identity_view,
        _import_plugin,
        _prepare_exact_sample,
        is_relative_to,
        sha256_file,
    )
    from route_snapshot_worker import recursive_sha256


PROTOCOL = "mome_stage019_s1_gt_informed_full_query_oracle_v1"
EXPERIMENT_ID = "mome_stage019_s1_full_query_oracle_upper_bound"
RUN_ID = "2026-08-17-mome-stage019-s1-full-query-oracle-upper-bound-v1"
EXPECTED_QUERY_COUNT = 900
EXPECTED_TOKEN_SET_SHA256 = (
    "C409767F05E5A066717BEAD05E502AB7083BE92007746C49EA5841BDB3FE0FD4"
)
CONDITIONS = (
    "clean",
    "beam_reduction_4",
    "lidar_zero",
    "limited_fov_original_code_60",
    "lidar_object_failure",
    "camera_zero",
    "camera_mud_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--scene-map", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--adapter-scripts", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--scene-offset", type=int, required=True)
    parser.add_argument("--scene-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-frames-total", type=int)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--verify-default-parity", action="store_true")
    return parser.parse_args()


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
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


def _cpu_bbox(result: dict) -> dict:
    boxes = result["boxes_3d"]
    if hasattr(boxes, "to"):
        boxes = boxes.to("cpu")
    return {
        "boxes_3d": boxes,
        "scores_3d": result["scores_3d"].detach().cpu(),
        "labels_3d": result["labels_3d"].detach().cpu(),
    }


def _bbox_parity(left: dict, right: dict) -> dict:
    import torch

    left_boxes = left["boxes_3d"].tensor.detach().cpu()
    right_boxes = right["boxes_3d"].tensor.detach().cpu()
    left_scores = left["scores_3d"].detach().cpu()
    right_scores = right["scores_3d"].detach().cpu()
    left_labels = left["labels_3d"].detach().cpu()
    right_labels = right["labels_3d"].detach().cpu()
    shapes_equal = (
        left_boxes.shape == right_boxes.shape
        and left_scores.shape == right_scores.shape
        and left_labels.shape == right_labels.shape
    )
    boxes_close = shapes_equal and torch.allclose(
        left_boxes, right_boxes, rtol=1e-5, atol=1e-6, equal_nan=True
    )
    scores_close = shapes_equal and torch.allclose(
        left_scores, right_scores, rtol=1e-5, atol=1e-6, equal_nan=True
    )
    labels_equal = shapes_equal and torch.equal(left_labels, right_labels)
    return {
        "passed": bool(shapes_equal and boxes_close and scores_close and labels_equal),
        "exact": bool(
            shapes_equal
            and torch.equal(left_boxes, right_boxes)
            and torch.equal(left_scores, right_scores)
            and labels_equal
        ),
        "box_max_abs": (
            float((left_boxes - right_boxes).abs().max().item())
            if shapes_equal and left_boxes.numel()
            else 0.0
        ),
        "score_max_abs": (
            float((left_scores - right_scores).abs().max().item())
            if shapes_equal and left_scores.numel()
            else 0.0
        ),
        "left_box_count": int(left_boxes.shape[0]),
        "right_box_count": int(right_boxes.shape[0]),
        "label_mismatch_count": (
            int(torch.count_nonzero(left_labels != right_labels).item())
            if shapes_equal
            else None
        ),
    }


def _tensor_tree_parity(left: object, right: object) -> dict:
    import torch

    def leaves(value: object, prefix: str = "root") -> dict[str, torch.Tensor]:
        if torch.is_tensor(value):
            return {prefix: value.detach().cpu()}
        if isinstance(value, dict):
            output = {}
            for key in sorted(value):
                output.update(leaves(value[key], f"{prefix}.{key}"))
            return output
        if isinstance(value, (list, tuple)):
            output = {}
            for index, item in enumerate(value):
                output.update(leaves(item, f"{prefix}[{index}]"))
            return output
        return {}

    left_leaves = leaves(left)
    right_leaves = leaves(right)
    paths_equal = set(left_leaves) == set(right_leaves)
    paths = sorted(set(left_leaves) & set(right_leaves))
    shape_mismatches = 0
    nonfinite_pattern_mismatches = 0
    exact_tensors = 0
    strict_close_tensors = 0
    engineering_close_tensors = 0
    element_count = 0
    exact_element_count = 0
    max_abs = 0.0
    tensor_differences = []
    for path in paths:
        left_tensor = left_leaves[path]
        right_tensor = right_leaves[path]
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            shape_mismatches += 1
            continue
        element_count += left_tensor.numel()
        exact = torch.equal(left_tensor, right_tensor)
        exact_tensors += int(exact)
        exact_element_count += int(torch.count_nonzero(left_tensor == right_tensor).item())
        if left_tensor.is_floating_point():
            finite_left = torch.isfinite(left_tensor)
            finite_right = torch.isfinite(right_tensor)
            nonfinite_pattern_mismatches += int(
                torch.count_nonzero(finite_left != finite_right).item()
            )
            finite = finite_left & finite_right
            if finite.any():
                max_abs = max(
                    max_abs,
                    float((left_tensor[finite] - right_tensor[finite]).abs().max().item()),
                )
            strict_close = torch.allclose(
                left_tensor, right_tensor, rtol=1e-5, atol=1e-6, equal_nan=True
            )
            engineering_close = torch.allclose(
                left_tensor, right_tensor, rtol=1e-3, atol=1e-4, equal_nan=True
            )
        else:
            strict_close = exact
            engineering_close = exact
        tensor_differences.append(
            {
                "path": path,
                "shape": list(left_tensor.shape),
                "dtype": str(left_tensor.dtype),
                "exact": bool(exact),
                "strict_close": bool(strict_close),
                "engineering_close": bool(engineering_close),
                "max_abs": (
                    float(
                        (left_tensor[finite] - right_tensor[finite])
                        .abs()
                        .max()
                        .item()
                    )
                    if left_tensor.is_floating_point() and finite.any()
                    else 0.0
                ),
            }
        )
        strict_close_tensors += int(strict_close)
        engineering_close_tensors += int(engineering_close)
    tensor_count = len(paths)
    return {
        "raw_paths_equal": paths_equal,
        "raw_tensor_count": tensor_count,
        "raw_shape_mismatch_count": shape_mismatches,
        "raw_nonfinite_pattern_mismatch_count": nonfinite_pattern_mismatches,
        "raw_exact_tensor_count": exact_tensors,
        "raw_strict_close_tensor_count": strict_close_tensors,
        "raw_engineering_close_tensor_count": engineering_close_tensors,
        "raw_element_count": element_count,
        "raw_exact_element_count": exact_element_count,
        "raw_max_abs": max_abs,
        "raw_tensor_differences": tensor_differences,
        "raw_exact": bool(
            paths_equal and not shape_mismatches and exact_tensors == tensor_count
        ),
        "raw_strict_close": bool(
            paths_equal
            and not shape_mismatches
            and not nonfinite_pattern_mismatches
            and strict_close_tensors == tensor_count
        ),
        "raw_engineering_close": bool(
            paths_equal
            and not shape_mismatches
            and not nonfinite_pattern_mismatches
            and engineering_close_tensors == tensor_count
        ),
    }


def _load_token_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokens = payload.get("tokens") if isinstance(payload, dict) else payload
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("token-file must contain a non-empty tokens list")
    return {str(token) for token in tokens}


def _build_dataset_cfg(cfg, args: argparse.Namespace, audit_path: Path):
    dataset_cfg = copy.deepcopy(cfg.data.train)
    if dataset_cfg.get("type") == "CBGSDataset":
        dataset_cfg = dataset_cfg.dataset
    dataset_cfg.data_root = str(args.data_root.resolve()).replace("\\", "/") + "/"
    dataset_cfg.ann_file = str(args.ann_file.resolve()).replace("\\", "/")
    dataset_cfg.test_mode = False
    dataset_cfg = _force_exact_frame_sampling(dataset_cfg)
    dataset_cfg = _force_identity_view(dataset_cfg)
    pipeline = []
    inserted = False
    for item in dataset_cfg.pipeline:
        item = copy.deepcopy(item)
        type_name = item.get("type")
        if type_name in {
            "QtaLocalCorruption3D",
            "PointShuffle",
            "PointsRangeFilter",
            "ModalMask3D",
            "GlobalRotScaleTransAll",
            "CustomRandomFlip3D",
        }:
            continue
        pipeline.append(item)
        if type_name == "LoadAnnotations3D":
            pipeline.append(
                {
                    "type": "Core4CorruptionAdapterV2",
                    "condition": args.condition,
                    "modality": "camera_lidar",
                    "sidecar_path": str(args.sidecar.resolve()),
                    "mask_root": str(args.mask_root.resolve()),
                    "audit_path": str(audit_path.resolve()),
                }
            )
            inserted = True
    if not inserted:
        raise ValueError("evaluation-with-GT pipeline lacks LoadAnnotations3D")
    dataset_cfg.pipeline = pipeline
    return dataset_cfg


def _scene_dataset_indices(dataset, scene_map: dict, scenes: list[str]) -> dict[str, list[int]]:
    token_to_index = {str(info["token"]): index for index, info in enumerate(dataset.data_infos)}
    if len(token_to_index) != 6019:
        raise ValueError("condition annotation must contain 6019 unique tokens")
    output = {}
    for scene in scenes:
        tokens = [str(value) for value in scene_map["scene_to_tokens"][scene]]
        missing = [token for token in tokens if token not in token_to_index]
        if missing:
            raise ValueError(f"condition annotation lacks scene frames: {missing[:3]}")
        output[scene] = [token_to_index[token] for token in tokens]
    return output


def main() -> int:
    args = parse_args()
    source_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("output-dir must stay inside artifact-root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("Oracle artifacts cannot be written into the source checkout")
    if sha256_file(args.checkpoint) != EXPECTED_MOME_CHECKPOINT_SHA256:
        raise ValueError("frozen MoME checkpoint SHA256 mismatch")
    scene_map = json.loads(args.scene_map.read_text(encoding="utf-8"))
    if (
        scene_map.get("status") != "passed"
        or int(scene_map.get("scene_count", -1)) != 150
        or int(scene_map.get("frame_count", -1)) != 6019
        or scene_map.get("token_set_sha256") != EXPECTED_TOKEN_SET_SHA256
    ):
        raise ValueError("fullval scene-map identity mismatch")
    if args.scene_offset < 0 or args.scene_count <= 0:
        raise ValueError("scene slice must be positive")
    scenes = list(scene_map["scene_tokens"])[
        args.scene_offset : args.scene_offset + args.scene_count
    ]
    if len(scenes) != args.scene_count:
        raise ValueError("scene slice exceeds the official validation split")
    token_filter = _load_token_filter(args.token_file)
    if args.max_frames_total is not None and args.max_frames_total <= 0:
        raise ValueError("max-frames-total must be positive")

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    adapter_scripts = args.adapter_scripts.resolve()
    if str(adapter_scripts) not in sys.path:
        sys.path.insert(0, str(adapter_scripts))

    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from nuscenes_r_core4_adapter_v2 import register_pipeline_registries

    cfg = Config.fromfile(str(args.config))
    _import_plugin(cfg, args.config)
    register_pipeline_registries()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"output-dir already exists; use a new attempt or --resume: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "corruption_audit.jsonl"
    dataset_cfg = _build_dataset_cfg(cfg, args, audit_path)
    dataset = build_dataset(dataset_cfg)
    grouped = _scene_dataset_indices(dataset, scene_map, scenes)
    cfg.model.pretrained = None
    set_random_seed(args.seed, deterministic=True)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model.requires_grad_(False)
    model = model.cuda(args.gpu_id)
    model.eval()
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameter_count != 0:
        raise RuntimeError("Stage019-S1 requires every MoME parameter to be frozen")

    source_files = {
        "config": args.config.resolve(),
        "base_config": source_root / "projects/configs/mome/mome.py",
        "detector": source_root / "projects/mmdet3d_plugin/models/detectors/mome.py",
        "med": source_root / "projects/mmdet3d_plugin/models/dense_heads/med.py",
        "multi_expert": source_root
        / "projects/mmdet3d_plugin/models/utils/multi_expert.py",
        "router": source_root / "projects/mmdet3d_plugin/models/utils/qta_router.py",
        "worker": Path(__file__).resolve(),
        "worker_common": Path(__file__).resolve().parent / "common.py",
        "sampling_helpers": Path(__file__).resolve().parent
        / "extract_route_loss_cache.py",
        "snapshot_hash_helpers": Path(__file__).resolve().parent
        / "route_snapshot_worker.py",
        "corruption_adapter": adapter_scripts / "nuscenes_r_core4_adapter_v2.py",
        "condition_adapter": adapter_scripts / "nuscenes_r_mome_conditions.py",
    }
    source_hashes = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in sorted(source_files.items())
    }
    source_bundle_sha256 = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()

    input_identity = {
        "protocol": PROTOCOL,
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "condition": args.condition,
        "scene_offset": args.scene_offset,
        "scene_count": args.scene_count,
        "requested_scenes": scenes,
        "token_file": str(args.token_file.resolve()) if args.token_file else None,
        "token_file_sha256": sha256_file(args.token_file) if args.token_file else None,
        "max_frames_total": args.max_frames_total,
        "seed": args.seed,
        "worker_id": args.worker_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "local_cuda_id": args.gpu_id,
        "cuda_device_name": torch.cuda.get_device_name(args.gpu_id),
        "config_sha256": sha256_file(args.config),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "annotation_file_sha256": sha256_file(args.ann_file),
        "scene_map_sha256": sha256_file(args.scene_map),
        "sidecar_sha256": sha256_file(args.sidecar),
        "adapter_source_sha256": sha256_file(adapter_scripts / "nuscenes_r_core4_adapter_v2.py"),
        "condition_source_sha256": sha256_file(adapter_scripts / "nuscenes_r_mome_conditions.py"),
        "detector_source_sha256": sha256_file(
            source_root / "projects/mmdet3d_plugin/models/detectors/mome.py"
        ),
        "med_source_sha256": sha256_file(
            source_root / "projects/mmdet3d_plugin/models/dense_heads/med.py"
        ),
        "router_source_sha256": sha256_file(
            source_root / "projects/mmdet3d_plugin/models/utils/qta_router.py"
        ),
        "worker_source_sha256": sha256_file(Path(__file__).resolve()),
        "source_hashes": source_hashes,
        "source_bundle_sha256": source_bundle_sha256,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "filter_empty_gt": False,
        "sample_access": "direct_prepare_train_data_no_getitem_fallback",
        "point_shuffle": False,
        "random_geometry": False,
        "verify_default_parity": bool(args.verify_default_parity),
    }
    progress_path = output_dir / "oracle_progress.json"
    progress = {
        "schema": "visfuse3d_stage019_s1_oracle_progress_v1",
        "status": "running",
        "input_identity": input_identity,
        "completed_scenes": [],
        "completed_frame_count": 0,
    }
    completed_records = []
    if args.resume and progress_path.is_file():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        if previous.get("input_identity") != input_identity:
            raise ValueError("Oracle resume input identity mismatch")
        for record in previous.get("completed_scenes", []):
            prediction_path = Path(record["prediction_path"])
            diagnostic_path = Path(record["diagnostic_path"])
            audit = Path(record["corruption_audit_path"])
            if (
                not prediction_path.is_file()
                or sha256_file(prediction_path) != record["prediction_sha256"]
                or not diagnostic_path.is_file()
                or sha256_file(diagnostic_path) != record["diagnostic_sha256"]
                or not audit.is_file()
                or sha256_file(audit) != record["corruption_audit_sha256"]
            ):
                raise ValueError("Oracle resume scene evidence hash mismatch")
            completed_records.append(record)
        progress["completed_scenes"] = completed_records
        progress["completed_frame_count"] = sum(
            int(record["frame_count"]) for record in completed_records
        )
    atomic_write_json(progress_path, progress)

    scene_records = list(completed_records)
    completed_by_scene = {record["scene_token"]: record for record in scene_records}
    frame_total = sum(int(record["frame_count"]) for record in scene_records)
    query_total = frame_total * EXPECTED_QUERY_COUNT
    switch_total = 0
    route_allocations = Counter()
    parity_exact = sum(
        int(item["exact"])
        for record in scene_records
        for item in record.get("default_parity", [])
    )
    parity_close = sum(
        int(item["passed"])
        for record in scene_records
        for item in record.get("default_parity", [])
    )
    for record in scene_records:
        with np.load(record["diagnostic_path"], allow_pickle=False) as payload:
            switch_total += int(payload["switch_mask"].sum())
            route_allocations.update(
                int(value) for value in payload["oracle_routes"].reshape(-1).tolist()
            )
    with torch.no_grad():
        for scene in scenes:
            if scene in completed_by_scene:
                continue
            prediction_rows = {}
            arrays = {
                "frame_token": [],
                "input_sha256": [],
                "base_routes": [],
                "oracle_routes": [],
                "route_query_losses": [],
                "query_gains": [],
                "switch_mask": [],
                "base_rematched_group_loss": [],
                "oracle_rematched_group_loss": [],
            }
            scene_parity = []
            scene_key = hashlib.sha256(scene.encode("utf-8")).hexdigest()[:16]
            prediction_path = output_dir / "scenes" / f"scene_{scene_key}_predictions.pkl"
            diagnostic_path = output_dir / "scenes" / f"scene_{scene_key}_diagnostics.npz"
            scene_audit_path = output_dir / "scenes" / f"scene_{scene_key}_corruption_audit.jsonl"
            interrupted_at = time.time_ns()
            for partial_path in (prediction_path, diagnostic_path, scene_audit_path):
                if partial_path.exists():
                    interrupted = partial_path.with_name(
                        f"{partial_path.stem}.interrupted_{interrupted_at}{partial_path.suffix}"
                    )
                    os.replace(partial_path, interrupted)
            os.environ["VISFUSE3D_CORE4_AUDIT_PATH"] = str(scene_audit_path.resolve())
            for dataset_index in grouped[scene]:
                frame_token = str(dataset.data_infos[dataset_index]["token"])
                if token_filter is not None and frame_token not in token_filter:
                    continue
                if args.max_frames_total is not None and frame_total >= args.max_frames_total:
                    break
                set_random_seed(args.seed, deterministic=True)
                sample = _prepare_exact_sample(dataset, dataset_index)
                batch = scatter(collate([sample], samples_per_gpu=1), [args.gpu_id])[0]
                observed_token = str(batch["img_metas"][0]["sample_idx"])
                if observed_token != frame_token:
                    raise RuntimeError("prepared frame token drifted from annotation order")
                input_hash = recursive_sha256(
                    {
                        "points": batch["points"][0],
                        "img": batch["img"],
                        "sample_idx": observed_token,
                        "lidar2img": batch["img_metas"][0]["lidar2img"],
                        "img_shape": batch["img_metas"][0]["img_shape"],
                    }
                )
                result = model.forward_qta_full_query_oracle(
                    points=batch["points"],
                    img_metas=batch["img_metas"],
                    img=batch["img"],
                    gt_bboxes_3d=batch["gt_bboxes_3d"],
                    gt_labels_3d=batch["gt_labels_3d"],
                    return_predictions=True,
                    return_raw_base=args.verify_default_parity,
                    repeat_base_on_shared_features=args.verify_default_parity,
                )
                losses = result["route_query_losses"]
                if losses.shape != (1, EXPECTED_QUERY_COUNT, 3):
                    raise RuntimeError(f"unexpected Oracle loss shape: {tuple(losses.shape)}")
                base_bbox = _cpu_bbox(result["base_bbox_results"][0])
                oracle_bbox = _cpu_bbox(result["oracle_bbox_results"][0])
                base_routes = result["base_routes"][0].detach().cpu().numpy().astype(np.int8)
                oracle_routes = result["oracle_routes"][0].detach().cpu().numpy().astype(np.int8)
                switch_mask = result["switch_mask"][0].detach().cpu().numpy().astype(bool)
                if args.verify_default_parity:
                    repeated = model.forward_qta_probe(
                        points=batch["points"],
                        img_metas=batch["img_metas"],
                        img=batch["img"],
                        gt_bboxes_3d=batch["gt_bboxes_3d"],
                        gt_labels_3d=batch["gt_labels_3d"],
                        return_query_losses=False,
                    )
                    repeated_bbox_list = model.pts_bbox_head.get_bboxes(
                        repeated["preds_dicts"], batch["img_metas"], rescale=False
                    )
                    repeated_bbox = repeated_bbox_list[0]
                    default = {
                        "boxes_3d": repeated_bbox[0],
                        "scores_3d": repeated_bbox[1],
                        "labels_3d": repeated_bbox[2],
                    }
                    parity = _bbox_parity(base_bbox, default)
                    raw_parity = _tensor_tree_parity(
                        result["base_raw_predictions"], repeated["preds_dicts"]
                    )
                    shared_feature_parity = {
                        f"shared_feature_{key}": value
                        for key, value in _tensor_tree_parity(
                            result["base_raw_predictions"],
                            result["shared_feature_base_repeat_predictions"],
                        ).items()
                    }
                    repeated_routes = repeated["base_routes"][0].detach().cpu().numpy()
                    route_drift_count = int(np.count_nonzero(repeated_routes != base_routes))
                    scene_parity.append(
                        {
                            "frame_token": frame_token,
                            "input_sha256": input_hash,
                            "repeat_route_drift_count": route_drift_count,
                            "repeat_route_drift_rate": route_drift_count / EXPECTED_QUERY_COUNT,
                            "classification": (
                                "shared_feature_canonical_head_exact"
                                if shared_feature_parity["shared_feature_raw_exact"]
                                and route_drift_count == 0
                                else "shared_feature_canonical_head_engineering_close"
                                if shared_feature_parity[
                                    "shared_feature_raw_engineering_close"
                                ]
                                and route_drift_count == 0
                                else "recorded_same_input_numerical_repeat_drift"
                            ),
                            **shared_feature_parity,
                            **raw_parity,
                            **parity,
                        }
                    )
                    parity_exact += int(parity["exact"])
                    parity_close += int(parity["passed"])
                prediction_rows[frame_token] = {"base": base_bbox, "oracle": oracle_bbox}
                arrays["frame_token"].append(frame_token)
                arrays["input_sha256"].append(input_hash)
                arrays["base_routes"].append(base_routes)
                arrays["oracle_routes"].append(oracle_routes)
                arrays["route_query_losses"].append(
                    losses[0].detach().float().cpu().numpy().astype(np.float32)
                )
                arrays["query_gains"].append(
                    result["query_gains"][0].detach().float().cpu().numpy().astype(np.float32)
                )
                arrays["switch_mask"].append(switch_mask)
                arrays["base_rematched_group_loss"].append(
                    float(result["base_rematched_group_losses"][0].item())
                )
                arrays["oracle_rematched_group_loss"].append(
                    float(result["oracle_rematched_group_losses"][0].item())
                )
                frame_total += 1
                query_total += EXPECTED_QUERY_COUNT
                switch_total += int(switch_mask.sum())
                route_allocations.update(int(value) for value in oracle_routes.tolist())
                progress.update(
                    {
                        "current_scene_token": scene,
                        "current_frame_token": frame_token,
                        "completed_frame_count": frame_total,
                        "query_decision_count": query_total,
                        "switch_count": switch_total,
                    }
                )
                atomic_write_json(progress_path, progress)
            if prediction_rows:
                _atomic_pickle(prediction_path, prediction_rows)
                encoded_frames = np.asarray(arrays["frame_token"], dtype="S32")
                encoded_hashes = np.asarray(arrays["input_sha256"], dtype="S64")
                _atomic_savez(
                    diagnostic_path,
                    frame_token=encoded_frames,
                    input_sha256=encoded_hashes,
                    base_routes=np.stack(arrays["base_routes"]),
                    oracle_routes=np.stack(arrays["oracle_routes"]),
                    route_query_losses=np.stack(arrays["route_query_losses"]),
                    query_gains=np.stack(arrays["query_gains"]),
                    switch_mask=np.stack(arrays["switch_mask"]),
                    base_rematched_group_loss=np.asarray(
                        arrays["base_rematched_group_loss"], dtype=np.float32
                    ),
                    oracle_rematched_group_loss=np.asarray(
                        arrays["oracle_rematched_group_loss"], dtype=np.float32
                    ),
                )
                record = {
                    "scene_token": scene,
                    "frame_count": len(prediction_rows),
                    "prediction_path": str(prediction_path.resolve()),
                    "prediction_sha256": sha256_file(prediction_path),
                    "diagnostic_path": str(diagnostic_path.resolve()),
                    "diagnostic_sha256": sha256_file(diagnostic_path),
                    "corruption_audit_path": str(scene_audit_path.resolve()),
                    "corruption_audit_sha256": sha256_file(scene_audit_path),
                    "frame_tokens": list(prediction_rows),
                    "default_parity": scene_parity,
                }
                scene_records.append(record)
                progress["completed_scenes"] = scene_records
                atomic_write_json(progress_path, progress)
            if args.max_frames_total is not None and frame_total >= args.max_frames_total:
                break

    if frame_total <= 0:
        raise RuntimeError("worker selected no validation frames")
    if token_filter is not None and frame_total != len(token_filter):
        missing = token_filter - {
            token for record in scene_records for token in record["frame_tokens"]
        }
        raise RuntimeError(f"token filter was not completely covered: {sorted(missing)[:3]}")
    manifest = {
        "schema": "visfuse3d_stage019_s1_oracle_worker_manifest_v1",
        "status": "complete",
        "claim_boundary": "gt_informed_empirical_upper_bound_shard_not_official_metric_result",
        "input_identity": input_identity,
        "frame_count": frame_total,
        "query_decision_count": query_total,
        "route_loss_value_count": query_total * 3,
        "switch_count": switch_total,
        "switch_rate": switch_total / query_total,
        "oracle_route_allocations": {str(key): value for key, value in sorted(route_allocations.items())},
        "default_parity_checked_frames": sum(
            len(record.get("default_parity", [])) for record in scene_records
        ),
        "default_parity_close_frames": parity_close,
        "default_parity_exact_frames": parity_exact,
        "default_raw_parity_checked_frames": sum(
            len(record.get("default_parity", [])) for record in scene_records
        ),
        "default_raw_parity_exact_frames": sum(
            int(item.get("raw_exact", False))
            for record in scene_records
            for item in record.get("default_parity", [])
        ),
        "default_raw_parity_strict_close_frames": sum(
            int(item.get("raw_strict_close", False))
            for record in scene_records
            for item in record.get("default_parity", [])
        ),
        "default_raw_parity_engineering_close_frames": sum(
            int(item.get("raw_engineering_close", False))
            for record in scene_records
            for item in record.get("default_parity", [])
        ),
        "default_shared_feature_parity_exact_frames": sum(
            int(item.get("shared_feature_raw_exact", False))
            for record in scene_records
            for item in record.get("default_parity", [])
        ),
        "default_shared_feature_parity_engineering_close_frames": sum(
            int(item.get("shared_feature_raw_engineering_close", False))
            for record in scene_records
            for item in record.get("default_parity", [])
        ),
        "scenes": scene_records,
        "corruption_audits": [
            {
                "scene_token": record["scene_token"],
                "path": record["corruption_audit_path"],
                "sha256": record["corruption_audit_sha256"],
            }
            for record in scene_records
        ],
    }
    manifest_path = output_dir / "oracle_worker_manifest.json"
    atomic_write_json(manifest_path, manifest)
    progress.update(
        {
            "status": "complete",
            "completed_frame_count": frame_total,
            "query_decision_count": query_total,
            "switch_count": switch_total,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
        }
    )
    atomic_write_json(progress_path, progress)
    print(json.dumps({key: manifest[key] for key in ("status", "frame_count", "query_decision_count", "switch_count", "switch_rate")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
