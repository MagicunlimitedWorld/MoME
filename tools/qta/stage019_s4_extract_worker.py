"""Extract immutable original/three-expert object sets for Stage019-S4."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

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
    )
    from .full_query_oracle_worker import _atomic_pickle, _cpu_bbox
    from .route_snapshot_worker import recursive_sha256
    from .stage019_s4_contracts import (
        ACTIONABLE_CONDITIONS,
        ALL_CONDITIONS,
        EXPERT_ROLES,
        HARD_BYPASS_CONDITIONS,
        PROTOCOL_PROFILE,
        RUN_ID,
        STAGE_ID,
        TRAINING20_INPUTS_MANIFEST_SHA256,
        VALIDATION_BEAM4_ANNOTATION_SHA256,
        sha256_file,
    )
    from .stage019_s4_object_cache import (
        assert_no_leakage,
        build_leakage_separated_cache_record,
    )
    from .stage019_s4_oracles import build_g0_oracles
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import (
        EXPECTED_MOME_CHECKPOINT_SHA256,
        _force_exact_frame_sampling,
        _force_identity_view,
        _import_plugin,
        _prepare_exact_sample,
        is_relative_to,
    )
    from full_query_oracle_worker import _atomic_pickle, _cpu_bbox
    from route_snapshot_worker import recursive_sha256
    from stage019_s4_contracts import (
        ACTIONABLE_CONDITIONS,
        ALL_CONDITIONS,
        EXPERT_ROLES,
        HARD_BYPASS_CONDITIONS,
        PROTOCOL_PROFILE,
        RUN_ID,
        STAGE_ID,
        TRAINING20_INPUTS_MANIFEST_SHA256,
        VALIDATION_BEAM4_ANNOTATION_SHA256,
        sha256_file,
    )
    from stage019_s4_object_cache import (
        assert_no_leakage,
        build_leakage_separated_cache_record,
    )
    from stage019_s4_oracles import build_g0_oracles


PHASES = ("g0", "fit", "calibration", "pilot", "fullval", "smoke")
TRAIN_PHASES = ("g0", "fit", "calibration")
EXPECTED_QUERY_COUNT = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--scene-map", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--partition")
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--beam-overlay-root", type=Path)
    parser.add_argument("--adapter-scripts", type=Path, required=True)
    parser.add_argument("--condition", choices=ALL_CONDITIONS, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--scene-offset", type=int, required=True)
    parser.add_argument("--scene-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--expected-cvd", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-gpu-pci", required=True)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--max-frames-total", type=int)
    return parser.parse_args()


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


def _points_xyz(points: Any) -> np.ndarray:
    """Normalize the post-collate point representation without changing values."""
    tensor = getattr(points, "tensor", points)
    if not all(hasattr(tensor, name) for name in ("ndim", "shape", "detach")):
        raise TypeError("S4 point input must be a tensor or expose .tensor")
    if int(tensor.ndim) != 2 or int(tensor.shape[1]) < 3:
        raise ValueError("S4 point input must have shape [N, >=3]")
    return tensor[:, :3].detach().cpu().numpy()


def _gpu_identity(args: argparse.Namespace) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.expected_cvd:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES mismatch: expected {args.expected_cvd}, got {visible}"
        )
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    pci = args.expected_gpu_pci.lower().replace("00000000:", "")
    matched = [
        row
        for row in rows
        if args.expected_gpu_uuid.lower() in row.lower()
        and pci in row.lower().replace("00000000:", "")
    ]
    if len(matched) != 1 or not matched[0].lstrip().startswith(f"{visible},"):
        raise RuntimeError("S4 worker GPU UUID/PCI/CVD identity mismatch")
    return {
        "cuda_visible_devices": visible,
        "local_cuda_id": args.gpu_id,
        "expected_uuid": args.expected_gpu_uuid,
        "expected_pci": args.expected_gpu_pci,
        "nvidia_smi_row": matched[0],
    }


def _build_train_dataset_cfg(cfg, args: argparse.Namespace, audit_path: Path):
    if args.sidecar is not None:
        raise ValueError("train-only extraction must not receive a validation sidecar")
    if args.condition == "beam_reduction_4" and args.beam_overlay_root is None:
        raise ValueError("train beam4 extraction requires --beam-overlay-root")
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
        if item.get("type") in {
            "QtaLocalCorruption3D",
            "PointShuffle",
            "PointsRangeFilter",
            "ModalMask3D",
            "GlobalRotScaleTransAll",
            "CustomRandomFlip3D",
        }:
            continue
        pipeline.append(item)
        if item.get("type") == "LoadAnnotations3D":
            pipeline.append(
                {
                    "type": "Stage019S4TrainingConditionAdapter",
                    "condition": args.condition,
                    "mask_root": str(args.mask_root.resolve()),
                    "audit_path": str(audit_path.resolve()),
                    "beam_overlay_root": (
                        str(args.beam_overlay_root.resolve())
                        if args.beam_overlay_root is not None
                        else None
                    ),
                }
            )
            inserted = True
    if not inserted:
        raise ValueError("S4 train pipeline lacks LoadAnnotations3D")
    dataset_cfg.pipeline = pipeline
    return dataset_cfg


def _build_validation_dataset_cfg(cfg, args: argparse.Namespace, audit_path: Path):
    if args.sidecar is None or not args.sidecar.is_file():
        raise ValueError("validation extraction requires the locked corruption sidecar")
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
        if item.get("type") in {
            "QtaLocalCorruption3D",
            "PointShuffle",
            "PointsRangeFilter",
            "ModalMask3D",
            "GlobalRotScaleTransAll",
            "CustomRandomFlip3D",
        }:
            continue
        pipeline.append(item)
        if item.get("type") == "LoadAnnotations3D":
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
        raise ValueError("S4 validation pipeline lacks LoadAnnotations3D")
    dataset_cfg.pipeline = pipeline
    return dataset_cfg


def _load_scene_contract(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    scene_map = json.loads(args.scene_map.read_text(encoding="utf-8"))
    sample_to_scene = {
        str(key): str(value)
        for key, value in scene_map.get("sample_to_scene", {}).items()
    }
    if not sample_to_scene and "scene_to_tokens" in scene_map:
        sample_to_scene = {
            str(token): str(scene)
            for scene, tokens in scene_map["scene_to_tokens"].items()
            for token in tokens
        }
    list_payload = (
        json.loads(args.scene_list.read_text(encoding="utf-8"))
        if args.scene_list is not None
        else scene_map
    )
    if args.partition is not None:
        list_payload = list_payload.get("partitions", {}).get(args.partition)
        if not isinstance(list_payload, dict):
            raise ValueError("scene-list lacks requested S4 partition")
    scenes = list_payload.get("scene_tokens") if isinstance(list_payload, dict) else list_payload
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scene contract lacks a non-empty ordered scene_tokens list")
    scenes = [str(value) for value in scenes]
    if len(scenes) != len(set(scenes)):
        raise ValueError("scene contract contains duplicate scenes")
    return scenes, sample_to_scene


def _group_indices(dataset, sample_to_scene: Mapping[str, str], scenes: list[str]):
    output = {scene: [] for scene in scenes}
    for index, info in enumerate(dataset.data_infos):
        token = str(info["token"])
        scene = sample_to_scene.get(token)
        if scene in output:
            output[scene].append(index)
    missing = [scene for scene, indices in output.items() if not indices]
    if missing:
        raise ValueError(f"condition annotation lacks selected scenes: {missing[:3]}")
    return output


def _bbox_arrays(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    boxes = result["boxes_3d"].tensor.detach().cpu().numpy().astype(np.float32)
    scores = result["scores_3d"].detach().cpu().numpy().astype(np.float32)
    labels = result["labels_3d"].detach().cpu().numpy().astype(np.int64)
    return {"boxes": boxes, "scores": scores, "labels": labels}


def _bbox_from_arrays_like(template: Mapping[str, Any], arrays: Mapping[str, np.ndarray]):
    import torch

    template_boxes = template["boxes_3d"]
    device = template["scores_3d"].device
    box_tensor = torch.as_tensor(arrays["boxes"], device=device, dtype=template_boxes.tensor.dtype)
    return {
        "boxes_3d": template_boxes.new_box(box_tensor),
        "scores_3d": torch.as_tensor(
            arrays["scores"], device=device, dtype=template["scores_3d"].dtype
        ),
        "labels_3d": torch.as_tensor(
            arrays["labels"], device=device, dtype=template["labels_3d"].dtype
        ),
    }


def _validate_query_ids(bundle: Mapping[str, Any], hard_bypass: bool) -> None:
    import torch

    canonical = torch.arange(EXPECTED_QUERY_COUNT, dtype=torch.long)
    original = bundle["query_ids"]["original"].detach().cpu().long()
    if original.shape != (1, EXPECTED_QUERY_COUNT) or not torch.equal(
        torch.sort(original[0]).values, canonical
    ):
        raise RuntimeError("S4 original query ids are not one complete 900 permutation")
    experts = bundle["query_ids"].get("experts", {})
    if hard_bypass:
        if experts:
            raise RuntimeError("hard bypass unexpectedly exposed expert query ids")
        return
    if tuple(experts) != EXPERT_ROLES:
        raise RuntimeError("S4 expert query id role order drifted")
    for role in EXPERT_ROLES:
        value = experts[role].detach().cpu().long()
        if value.shape != (1, EXPECTED_QUERY_COUNT) or not torch.equal(
            torch.sort(value[0]).values, canonical
        ):
            raise RuntimeError(f"S4 {role} query ids are not a complete permutation")


def _flatten_records(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    offsets = [0]
    gt_offsets = [0]
    arrays: dict[str, list[np.ndarray]] = {}
    for record in records:
        inference = record["inference_features"]
        targets = record["offline_targets"]
        count = len(inference["anchor_scores"])
        offsets.append(offsets[-1] + count)
        gt_offsets.append(gt_offsets[-1] + len(targets["gt_labels"]))
        evidence = inference["object_evidence"]
        values = {
            "feature_matrix": inference["feature_matrix"],
            "anchor_boxes": inference["anchor_boxes"],
            "anchor_scores": inference["anchor_scores"],
            "anchor_labels": inference["anchor_labels"],
            "aligned_expert_boxes": inference["aligned_expert_boxes"],
            "aligned_expert_scores": inference["aligned_expert_scores"],
            "expert_match_mask": inference["expert_match_mask"],
            "expert_source_indices": inference["expert_source_indices"],
            "object_distance": evidence["object_distance"],
            "box_point_count": evidence["box_point_count"],
            "neighborhood_point_count": evidence["neighborhood_point_count"],
            "visible_camera_views": evidence["visible_camera_views"],
            "gt_boxes": targets["gt_boxes"],
            "gt_labels": targets["gt_labels"],
            "anchor_soft_quality": targets["anchor_soft_quality"],
            "expert_soft_quality": targets["expert_soft_quality"],
            "target_gt_index": targets["target_gt_index"],
            "target_gt_boxes": targets["target_gt_boxes"],
            "target_gt_valid": targets["target_gt_valid"],
            "center_source": targets["center_source"],
            "size_source": targets["size_source"],
            "yaw_source": targets["yaw_source"],
            "velocity_source": targets["velocity_source"],
            "accept_target": targets["accept_target"],
        }
        for key, value in values.items():
            arrays.setdefault(key, []).append(np.asarray(value))
    output = {
        key: np.concatenate(values, axis=0)
        for key, values in arrays.items()
    }
    output["frame_object_offsets"] = np.asarray(offsets, dtype=np.int64)
    output["frame_gt_offsets"] = np.asarray(gt_offsets, dtype=np.int64)
    return output


def _source_bundle(files: Mapping[str, Path]) -> tuple[dict[str, Any], str]:
    records = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in sorted(files.items())
    }
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    return records, digest


def _tree_sha256(root: Path) -> str:
    lines = []
    for path in sorted(
        (value for value in root.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        lines.append(
            "{}|{}|{}".format(
                path.relative_to(root).as_posix(), path.stat().st_size, sha256_file(path)
            )
        )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest().upper()


def _validate_phase_annotation(args: argparse.Namespace) -> None:
    if (
        args.phase not in TRAIN_PHASES
        and args.condition == "beam_reduction_4"
        and sha256_file(args.ann_file) != VALIDATION_BEAM4_ANNOTATION_SHA256
    ):
        raise ValueError(
            "validation beam4 extraction requires the locked pre-derived annotation"
        )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("S4 extract output must stay inside artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("S4 extract output cannot enter the source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable S4 extract worker exists: {output_dir}")
    if sha256_file(args.checkpoint) != EXPECTED_MOME_CHECKPOINT_SHA256:
        raise ValueError("frozen MoME checkpoint SHA256 mismatch")
    if args.scene_offset < 0 or args.scene_count <= 0:
        raise ValueError("scene slice must be positive")
    if args.phase in ("g0", "fit") and args.condition in HARD_BYPASS_CONDITIONS:
        raise ValueError("complete-zero conditions cannot enter G0 or fit extraction")
    _validate_phase_annotation(args)
    if args.phase == "g0":
        if (
            args.scene_list is None
            or sha256_file(args.scene_list) != TRAINING20_INPUTS_MANIFEST_SHA256
        ):
            raise ValueError("G0 requires the locked training20 input manifest")
        training20_identity = json.loads(args.scene_list.read_text(encoding="utf-8"))
        expected_annotation_sha = (
            training20_identity["beam4_annotation_sha256"]
            if args.condition == "beam_reduction_4"
            else training20_identity["clean_annotation_sha256"]
        )
        if (
            int(training20_identity.get("scene_count", -1)) != 20
            or int(training20_identity.get("frame_count", -1)) != 803
            or sha256_file(args.ann_file) != expected_annotation_sha
        ):
            raise ValueError("G0 annotation/training20 coverage identity mismatch")
        if args.condition == "beam_reduction_4":
            expected_root = Path(training20_identity["beam4_overlay_root"]).resolve()
            if (
                args.beam_overlay_root is None
                or args.beam_overlay_root.resolve() != expected_root
                or _tree_sha256(expected_root)
                != training20_identity["beam4_overlay_tree_sha256"]
            ):
                raise ValueError("G0 beam4 overlay root/tree identity mismatch")
    ordered_scenes, sample_to_scene = _load_scene_contract(args)
    scenes = ordered_scenes[args.scene_offset : args.scene_offset + args.scene_count]
    if len(scenes) != args.scene_count:
        raise ValueError("scene slice exceeds the ordered scene contract")
    token_filter = None
    if args.token_file is not None:
        payload = json.loads(args.token_file.read_text(encoding="utf-8"))
        tokens = payload.get("tokens") if isinstance(payload, dict) else payload
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("token-file must contain a non-empty token list")
        token_filter = {str(value) for value in tokens}
    gpu_identity = _gpu_identity(args)

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

    cfg = Config.fromfile(str(args.config))
    _import_plugin(cfg, args.config)
    if args.phase in TRAIN_PHASES:
        import stage019_s4_training_conditions  # noqa: F401
    else:
        from nuscenes_r_core4_adapter_v2 import register_pipeline_registries

        register_pipeline_registries()
    output_dir.mkdir(parents=True)
    initial_audit = output_dir / "initial_corruption_audit.jsonl"
    dataset_cfg = (
        _build_train_dataset_cfg(cfg, args, initial_audit)
        if args.phase in TRAIN_PHASES
        else _build_validation_dataset_cfg(cfg, args, initial_audit)
    )
    dataset = build_dataset(dataset_cfg)
    grouped = _group_indices(dataset, sample_to_scene, scenes)
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
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 0:
        raise RuntimeError("S4 extraction requires fully frozen MoME")
    source_files = {
        "config": args.config.resolve(),
        "detector": source_root / "projects/mmdet3d_plugin/models/detectors/mome.py",
        "med": source_root / "projects/mmdet3d_plugin/models/dense_heads/med.py",
        "fusion": source_root / "projects/mmdet3d_plugin/models/utils/object_set_attribute_fusion.py",
        "worker": Path(__file__).resolve(),
        "cache": Path(__file__).with_name("stage019_s4_object_cache.py"),
        "oracles": Path(__file__).with_name("stage019_s4_oracles.py"),
        "contracts": Path(__file__).with_name("stage019_s4_contracts.py"),
        "s4_training_conditions": Path(__file__).with_name(
            "stage019_s4_training_conditions.py"
        ),
        "s2_training_condition_base": Path(__file__).with_name(
            "stage019_s2_training_conditions.py"
        ),
        "condition_adapter": adapter_scripts / "nuscenes_r_mome_conditions.py",
        "corruption_adapter": adapter_scripts / "nuscenes_r_core4_adapter_v2.py",
    }
    source_hashes, source_bundle_sha256 = _source_bundle(source_files)
    identity = {
        "stage_id": STAGE_ID,
        "protocol_profile": PROTOCOL_PROFILE,
        "run_id": RUN_ID,
        "phase": args.phase,
        "condition": args.condition,
        "scene_offset": args.scene_offset,
        "scene_count": args.scene_count,
        "requested_scenes": scenes,
        "seed": args.seed,
        "worker_id": args.worker_id,
        "gpu_identity": gpu_identity,
        "config_sha256": sha256_file(args.config),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "annotation_sha256": sha256_file(args.ann_file),
        "scene_map_sha256": sha256_file(args.scene_map),
        "scene_list_sha256": sha256_file(args.scene_list) if args.scene_list else None,
        "sidecar_sha256": sha256_file(args.sidecar) if args.sidecar else None,
        "source_hashes": source_hashes,
        "source_bundle_sha256": source_bundle_sha256,
        "frozen_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": trainable,
        "sample_access": "direct_prepare_train_data_no_getitem_fallback",
        "filter_empty_gt": False,
        "point_shuffle": False,
        "random_geometry": False,
    }
    frame_total = 0
    run_started = time.perf_counter()
    frame_latencies = []
    torch.cuda.reset_peak_memory_stats(args.gpu_id)
    scene_records = []
    all_frame_tokens = []
    observed_bypass_values = []
    with torch.no_grad():
        for scene in scenes:
            key = hashlib.sha256(scene.encode("utf-8")).hexdigest()[:16]
            prediction_path = output_dir / "scenes" / f"scene_{key}_predictions.pkl"
            diagnostic_path = output_dir / "scenes" / f"scene_{key}_diagnostics.npz"
            audit_path = output_dir / "scenes" / f"scene_{key}_corruption_audit.jsonl"
            if any(path.exists() for path in (prediction_path, diagnostic_path, audit_path)):
                raise FileExistsError("partial S4 scene evidence requires a new attempt")
            os.environ["VISFUSE3D_CORE4_AUDIT_PATH"] = str(audit_path.resolve())
            os.environ["VISFUSE3D_STAGE019_S4_AUDIT_PATH"] = str(
                audit_path.resolve()
            )
            predictions = {}
            cache_records = []
            frame_tokens = []
            input_hashes = []
            decoder_calls = []
            oracle_attribute_modified = []
            oracle_union_added = []
            oracle_union_true_positive = []
            for dataset_index in grouped[scene]:
                token = str(dataset.data_infos[dataset_index]["token"])
                if token_filter is not None and token not in token_filter:
                    continue
                if args.max_frames_total is not None and frame_total >= args.max_frames_total:
                    break
                set_random_seed(args.seed, deterministic=True)
                sample = _prepare_exact_sample(dataset, dataset_index)
                batch = scatter(collate([sample], samples_per_gpu=1), [args.gpu_id])[0]
                if str(batch["img_metas"][0]["sample_idx"]) != token:
                    raise RuntimeError("S4 prepared frame token drifted")
                input_hash = recursive_sha256(
                    {
                        "points": batch["points"][0],
                        "img": batch["img"],
                        "sample_idx": token,
                        "lidar2img": batch["img_metas"][0]["lidar2img"],
                        "img_shape": batch["img_metas"][0]["img_shape"],
                    }
                )
                try:
                    torch.cuda.synchronize(args.gpu_id)
                    frame_started = time.perf_counter()
                    bundle = model._forward_full_context_expert_bundle(
                        points=batch["points"],
                        img_metas=batch["img_metas"],
                        img=batch["img"],
                        decode=True,
                    )
                    torch.cuda.synchronize(args.gpu_id)
                    frame_latencies.append(time.perf_counter() - frame_started)
                except Exception as exc:
                    atomic_write_json(
                        output_dir / "engineering_failure_manifest.json",
                        {
                            "schema": "visfuse3d_stage019_s4_engineering_failure_v1",
                            "status": "engineering_failure",
                            "condition": args.condition,
                            "phase": args.phase,
                            "scene_token": scene,
                            "frame_token": token,
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                            "formal_merge_allowed": False,
                        },
                    )
                    raise
                hard_bypass = bool(bundle["hard_bypass"])
                observed_bypass_values.append(hard_bypass)
                expected_bypass = args.condition in HARD_BYPASS_CONDITIONS
                if hard_bypass != expected_bypass:
                    raise RuntimeError("S4 hard-bypass classification drifted")
                expected_calls = 1 if hard_bypass else 4
                if int(bundle["decoder_call_count"]) != expected_calls:
                    raise RuntimeError("S4 decoder call count violates protocol")
                if tuple(bundle.get("route_order", ())) != EXPERT_ROLES:
                    raise RuntimeError("S4 expert route order drifted")
                _validate_query_ids(bundle, hard_bypass)
                original_bbox = bundle["original_bbox"][0]
                prediction_roles = {"original_mome": _cpu_bbox(original_bbox)}
                if hard_bypass:
                    if bundle["expert_bboxes"]:
                        raise RuntimeError("complete-zero bypass decoded expert boxes")
                    # Same object reference is intentional evidence that final is
                    # the same-run original result, not a copied/re-serialized box.
                    prediction_roles["final"] = prediction_roles["original_mome"]
                else:
                    if tuple(bundle["expert_bboxes"]) != EXPERT_ROLES:
                        raise RuntimeError("S4 expert bbox role order drifted")
                    raw_original = _bbox_arrays(original_bbox)
                    raw_experts = {
                        role: _bbox_arrays(bundle["expert_bboxes"][role][0])
                        for role in EXPERT_ROLES
                    }
                    points_xyz = _points_xyz(batch["points"][0])
                    gt_boxes = batch["gt_bboxes_3d"][0].tensor.detach().cpu().numpy()
                    gt_labels = batch["gt_labels_3d"][0].detach().cpu().numpy()
                    record = build_leakage_separated_cache_record(
                        raw_original,
                        raw_experts,
                        points_xyz,
                        batch["img_metas"][0]["lidar2img"],
                        batch["img_metas"][0]["img_shape"],
                        gt_boxes,
                        gt_labels,
                    )
                    assert_no_leakage(record)
                    cache_records.append(record)
                    prediction_roles.update(
                        {
                            role: _cpu_bbox(bundle["expert_bboxes"][role][0])
                            for role in EXPERT_ROLES
                        }
                    )
                    if args.phase == "g0":
                        oracle_predictions, oracle_counts = build_g0_oracles(
                            raw_original, raw_experts, record
                        )
                        prediction_roles.update(
                            {
                                role: _cpu_bbox(
                                    _bbox_from_arrays_like(
                                        original_bbox, oracle_predictions[role]
                                    )
                                )
                                for role in (
                                    "g0_score",
                                    "g0_attribute",
                                    "g0_union",
                                )
                            }
                        )
                        oracle_attribute_modified.append(
                            oracle_counts["attribute_modified_object_count"]
                        )
                        oracle_union_added.append(
                            oracle_counts["union_added_box_count"]
                        )
                        oracle_union_true_positive.append(
                            oracle_counts["union_added_true_positive_count"]
                        )
                predictions[token] = prediction_roles
                frame_tokens.append(token)
                all_frame_tokens.append(token)
                input_hashes.append(input_hash)
                decoder_calls.append(expected_calls)
                frame_total += 1
            if not frame_tokens:
                if args.max_frames_total is not None and frame_total >= args.max_frames_total:
                    break
                raise RuntimeError("selected S4 scene produced zero frames")
            _atomic_pickle(prediction_path, predictions)
            flattened = _flatten_records(cache_records) if cache_records else {}
            _atomic_savez(
                diagnostic_path,
                frame_token=np.asarray(frame_tokens, dtype="S32"),
                input_sha256=np.asarray(input_hashes, dtype="S64"),
                decoder_call_count=np.asarray(decoder_calls, dtype=np.int8),
                g0_attribute_modified_object_count=np.asarray(
                    oracle_attribute_modified, dtype=np.int32
                ),
                g0_union_added_box_count=np.asarray(oracle_union_added, dtype=np.int16),
                g0_union_added_true_positive_count=np.asarray(
                    oracle_union_true_positive, dtype=np.int16
                ),
                **flattened,
            )
            if not audit_path.is_file() or audit_path.stat().st_size <= 0:
                raise RuntimeError("S4 scene corruption audit is missing or empty")
            audit_rows = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            audited_tokens = [
                str(row.get("token", row.get("sample_token", "")))
                for row in audit_rows
            ]
            if audited_tokens != frame_tokens or any(
                str(row.get("condition")) != args.condition for row in audit_rows
            ):
                raise RuntimeError("S4 scene corruption audit token/condition coverage drifted")
            scene_records.append(
                {
                    "scene_token": scene,
                    "frame_count": len(frame_tokens),
                    "frame_tokens": frame_tokens,
                    "prediction_path": str(prediction_path.resolve()),
                    "prediction_sha256": sha256_file(prediction_path),
                    "diagnostic_path": str(diagnostic_path.resolve()),
                    "diagnostic_sha256": sha256_file(diagnostic_path),
                    "corruption_audit_path": str(audit_path.resolve()),
                    "corruption_audit_sha256": sha256_file(audit_path),
                }
            )
            if args.max_frames_total is not None and frame_total >= args.max_frames_total:
                break
    if not observed_bypass_values or len(set(observed_bypass_values)) != 1:
        raise RuntimeError("S4 worker hard-bypass observation is empty or inconsistent")
    observed_hard_bypass = observed_bypass_values[0]
    wall_elapsed = time.perf_counter() - run_started
    manifest = {
        "schema": "visfuse3d_stage019_s4_extract_worker_manifest_v1",
        "status": "complete",
        "input_identity": identity,
        "phase": args.phase,
        "condition": args.condition,
        "scene_tokens": [record["scene_token"] for record in scene_records],
        "frame_tokens": all_frame_tokens,
        "scene_count": len(scene_records),
        "frame_count": len(all_frame_tokens),
        "unique_token_count": len(set(all_frame_tokens)),
        "roles": (
            ["original_mome", "final"]
            if args.condition in HARD_BYPASS_CONDITIONS
            else [
                "original_mome",
                *EXPERT_ROLES,
                *(
                    ("g0_score", "g0_attribute", "g0_union")
                    if args.phase == "g0"
                    else ()
                ),
            ]
        ),
        "hard_bypass": observed_hard_bypass,
        "observed_hard_bypass": observed_hard_bypass,
        "condition_expected_hard_bypass": args.condition in HARD_BYPASS_CONDITIONS,
        "identity_only_not_training": bool(
            args.phase == "calibration"
            and args.condition in HARD_BYPASS_CONDITIONS
        ),
        "training_cache_eligible": bool(
            args.phase in TRAIN_PHASES
            and args.condition not in HARD_BYPASS_CONDITIONS
        ),
        "decoder_call_count_per_frame": (
            1 if observed_hard_bypass else 4
        ),
        "inference_features_exclude_gt_fault_mask_corruption_parameters": True,
        "offline_targets_contain_gt_only_for_training_and_oracle": True,
        "engineering_failure_count": 0,
        "resource_metrics": {
            "wall_elapsed_seconds": wall_elapsed,
            "frame_per_second": len(all_frame_tokens) / max(wall_elapsed, 1e-12),
            "per_frame_seconds_mean": float(np.mean(frame_latencies)),
            "per_frame_seconds_p50": float(np.percentile(frame_latencies, 50)),
            "per_frame_seconds_p95": float(np.percentile(frame_latencies, 95)),
            "cuda_max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(args.gpu_id)
            ),
            "cuda_max_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(args.gpu_id)
            ),
        },
        "scenes": scene_records,
    }
    atomic_write_json(output_dir / "stage019_s4_extract_worker_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
