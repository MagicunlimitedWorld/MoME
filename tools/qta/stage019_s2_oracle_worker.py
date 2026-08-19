"""Run one immutable Stage019-S2 validation/pilot scene shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import (
        EXPECTED_MOME_CHECKPOINT_SHA256,
        _import_plugin,
        _prepare_exact_sample,
        is_relative_to,
        sha256_file,
    )
    from .full_query_oracle_worker import (
        CONDITIONS,
        EXPECTED_QUERY_COUNT,
        EXPECTED_TOKEN_SET_SHA256,
        _atomic_pickle,
        _atomic_savez,
        _build_dataset_cfg,
        _cpu_bbox,
        _load_token_filter,
        _scene_dataset_indices,
    )
    from .route_snapshot_worker import recursive_sha256
    from .stage019_s2_training_conditions import S3_HARD_BYPASS_CONDITIONS
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import (
        EXPECTED_MOME_CHECKPOINT_SHA256,
        _import_plugin,
        _prepare_exact_sample,
        is_relative_to,
        sha256_file,
    )
    from full_query_oracle_worker import (
        CONDITIONS,
        EXPECTED_QUERY_COUNT,
        EXPECTED_TOKEN_SET_SHA256,
        _atomic_pickle,
        _atomic_savez,
        _build_dataset_cfg,
        _cpu_bbox,
        _load_token_filter,
        _scene_dataset_indices,
    )
    from route_snapshot_worker import recursive_sha256
    from stage019_s2_training_conditions import S3_HARD_BYPASS_CONDITIONS


STAGE_ID = "Stage019-S2"
EXPERIMENT_ID = (
    "mome_stage019_s2_context_preserving_output_oracle_and_greedy_joint_gt_oracle"
)
RUN_ID = (
    "2026-08-17-mome-stage019-s2-context-preserving-output-oracle-and-"
    "greedy-joint-gt-oracle-v1"
)
LEGACY_PROTOCOL_PROFILE = "stage019_s2_legacy_v1"
S3_PROTOCOL_PROFILE = "stage019_s3_actionable_hard_bypass_v1"
S2A_ROLES = (
    "original_mome",
    "keep_anchored_output_oracle",
    "pure_original_route_full_context_reconstruction",
    "pure_three_expert_portfolio_oracle",
)
S2B_ROLES = (
    "original_mome_greedy_run",
    "greedy_joint_routing_gt_oracle_result",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mode", choices=("s2a", "s2b"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--scene-map", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--adapter-scripts", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--margins", type=Path, required=True)
    parser.add_argument("--scene-offset", type=int, required=True)
    parser.add_argument("--scene-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--expected-cvd", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-gpu-pci", required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--stage-id", default=STAGE_ID)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--protocol-profile",
        choices=(LEGACY_PROTOCOL_PROFILE, S3_PROTOCOL_PROFILE),
        default=LEGACY_PROTOCOL_PROFILE,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-frames-total", type=int)
    parser.add_argument("--token-file", type=Path)
    return parser.parse_args()


def _gpu_identity(args: argparse.Namespace) -> dict:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd != args.expected_cvd:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES mismatch: expected {args.expected_cvd}, got {cvd}"
        )
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    matched = [
        row
        for row in rows
        if args.expected_gpu_uuid.lower() in row.lower()
        and args.expected_gpu_pci.lower().replace("00000000:", "")
        in row.lower().replace("00000000:", "")
    ]
    if len(matched) != 1:
        raise RuntimeError("expected S2 GPU UUID/PCI identity mismatch")
    if not matched[0].lstrip().startswith(f"{args.expected_cvd},"):
        raise RuntimeError("CVD physical index does not match locked GPU UUID/PCI")
    return {
        "cuda_visible_devices": cvd,
        "local_cuda_id": args.gpu_id,
        "expected_uuid": args.expected_gpu_uuid,
        "expected_pci": args.expected_gpu_pci,
        "nvidia_smi_row": matched[0],
    }


def _load_margins(args: argparse.Namespace) -> tuple[dict, dict]:
    payload = json.loads(args.margins.read_text(encoding="utf-8"))
    allowed = {
        "passed",
        "s2b_stopped_zero_positive_quality",
        "candidate_mass_not_coverable_under_K16",
    }
    if payload.get("status") not in allowed:
        raise ValueError("calibration margins are incomplete")
    condition = args.condition
    try:
        keep = payload["epsilon_num_query"][condition]["original_mome"]
        reconstruction = payload["epsilon_num_query"][condition]["reconstruction"]
    except KeyError as exc:
        raise ValueError(f"calibration margins lack {condition}: {exc}") from exc
    if any(len(value) != 3 for value in (keep, reconstruction)):
        raise ValueError("query margins must contain three destinations")
    proxy = [0.0, 0.0, 0.0]
    frame = 0.0
    if args.mode == "s2b":
        try:
            proxy = payload["proxy_overestimate"][condition]
            frame = payload["epsilon_num_frame"][condition]
        except KeyError as exc:
            raise ValueError(
                f"S2-B calibration margins lack actionable condition {condition}: {exc}"
            ) from exc
        if len(proxy) != 3:
            raise ValueError("proxy margins must contain three destinations")
    numeric = np.asarray(keep + reconstruction + proxy + [frame], dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric < 0):
        raise ValueError("calibration margins must be finite and non-negative")
    budget = payload.get("candidate_budget", {})
    if args.mode == "s2b":
        if payload.get("status") != "passed" or budget.get("status") != "coverage_reached":
            raise ValueError("S2-B is stopped by the locked calibration result")
        if int(budget.get("K", -1)) not in (4, 8, 16):
            raise ValueError("S2-B candidate K is not locked to 4, 8, or 16")
    return payload, {
        "epsilon_query_keep": keep,
        "epsilon_query_reconstruction": reconstruction,
        "proxy_overestimate": proxy,
        "epsilon_frame": float(frame),
        "K": int(budget.get("K") or 0),
    }


def _source_bundle(source_files: dict[str, Path]) -> tuple[dict, str]:
    records = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in sorted(source_files.items())
    }
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    return records, digest


def _tensor_float(value) -> float | None:
    if value is None:
        return None
    return float(value.detach().cpu().item())


def _greedy_trace(result: dict) -> list[dict]:
    rows = []
    for item in result["trace"]:
        rows.append(
            {
                "step": int(item["step"]),
                "query_id": int(item["query_id"]),
                "destination_expert": int(item["destination_expert"]),
                "quality": float(item["quality"]),
                "current_route_hash": item["current_route_hash"],
                "candidate_route_hash": item["candidate_route_hash"],
                "proxy_full_context_gain": _tensor_float(
                    item["proxy_full_context_gain"]
                ),
                "actual_query_gain": _tensor_float(item["actual_query_gain"]),
                "actual_frame_gain": _tensor_float(item["actual_frame_gain"]),
                "accepted": bool(item["accepted"]),
                "reason": item["reason"],
            }
        )
    return rows


def _frame_input_hash(batch: dict, token: str) -> str:
    return recursive_sha256(
        {
            "points": batch["points"][0],
            "img": batch["img"],
            "sample_idx": token,
            "lidar2img": batch["img_metas"][0]["lidar2img"],
            "img_shape": batch["img_metas"][0]["img_shape"],
        }
    )


def main() -> int:
    args = parse_args()
    if (
        args.protocol_profile == S3_PROTOCOL_PROFILE
        and args.mode == "s2b"
        and args.condition in S3_HARD_BYPASS_CONDITIONS
    ):
        raise ValueError(
            "Stage019-S3 complete-zero conditions require exact hard bypass, not S3-B search"
        )
    source_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("S2 worker output must stay inside artifact-root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("S2 artifacts cannot enter the source checkout")
    if sha256_file(args.checkpoint) != EXPECTED_MOME_CHECKPOINT_SHA256:
        raise ValueError("frozen MoME checkpoint SHA256 mismatch")
    if args.scene_offset < 0 or args.scene_count <= 0:
        raise ValueError("scene slice must be positive")
    if args.max_frames_total is not None and args.max_frames_total <= 0:
        raise ValueError("max-frames-total must be positive")
    scene_map = json.loads(args.scene_map.read_text(encoding="utf-8"))
    if (
        scene_map.get("status") != "passed"
        or int(scene_map.get("scene_count", -1)) != 150
        or int(scene_map.get("frame_count", -1)) != 6019
        or scene_map.get("token_set_sha256") != EXPECTED_TOKEN_SET_SHA256
    ):
        raise ValueError("full validation scene-map identity mismatch")
    scenes = list(scene_map["scene_tokens"])[
        args.scene_offset : args.scene_offset + args.scene_count
    ]
    if len(scenes) != args.scene_count:
        raise ValueError("scene slice exceeds validation split")
    token_filter = _load_token_filter(args.token_file)
    margin_payload, margins = _load_margins(args)
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
    from nuscenes_r_core4_adapter_v2 import register_pipeline_registries

    cfg = Config.fromfile(str(args.config))
    _import_plugin(cfg, args.config)
    register_pipeline_registries()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"immutable S2 worker exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cfg = _build_dataset_cfg(
        cfg, args, output_dir / "initial_corruption_audit.jsonl"
    )
    dataset = build_dataset(dataset_cfg)
    grouped = _scene_dataset_indices(dataset, scene_map, scenes)
    cfg.model.pretrained = None
    set_random_seed(args.seed, deterministic=True)
    model = build_model(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model.requires_grad_(False)
    model = model.cuda(args.gpu_id)
    model.eval()
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable != 0:
        raise RuntimeError("Stage019-S2 requires every model parameter frozen")

    source_files = {
        "config": args.config.resolve(),
        "detector": source_root / "projects/mmdet3d_plugin/models/detectors/mome.py",
        "med": source_root / "projects/mmdet3d_plugin/models/dense_heads/med.py",
        "router": source_root / "projects/mmdet3d_plugin/models/utils/qta_router.py",
        "worker": Path(__file__).resolve(),
        "condition_adapter": adapter_scripts / "nuscenes_r_mome_conditions.py",
        "corruption_adapter": adapter_scripts / "nuscenes_r_core4_adapter_v2.py",
    }
    source_hashes, source_bundle_sha256 = _source_bundle(source_files)
    input_identity = {
        "stage_id": args.stage_id,
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "protocol_profile": args.protocol_profile,
        "mode": args.mode,
        "condition": args.condition,
        "scene_offset": args.scene_offset,
        "scene_count": args.scene_count,
        "requested_scenes": scenes,
        "token_file": str(args.token_file.resolve()) if args.token_file else None,
        "token_file_sha256": sha256_file(args.token_file) if args.token_file else None,
        "seed": args.seed,
        "worker_id": args.worker_id,
        "gpu_identity": gpu_identity,
        "config_sha256": sha256_file(args.config),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "annotation_file_sha256": sha256_file(args.ann_file),
        "scene_map_sha256": sha256_file(args.scene_map),
        "sidecar_sha256": sha256_file(args.sidecar),
        "margins_sha256": sha256_file(args.margins),
        "margins_status": margin_payload["status"],
        "locked_margins": margins,
        "source_hashes": source_hashes,
        "source_bundle_sha256": source_bundle_sha256,
        "trainable_parameter_count": trainable,
        "sample_access": "direct_prepare_train_data_no_getitem_fallback",
        "filter_empty_gt": False,
        "point_shuffle": False,
        "random_geometry": False,
    }
    progress_path = output_dir / "s2_oracle_progress.json"
    progress = {
        "schema": "visfuse3d_stage019_s2_oracle_progress_v1",
        "status": "running",
        "input_identity": input_identity,
        "completed_scenes": [],
        "completed_frame_count": 0,
    }
    completed = []
    if args.resume and progress_path.is_file():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        if previous.get("input_identity") != input_identity:
            raise ValueError("S2 resume input/source identity mismatch")
        for record in previous.get("completed_scenes", []):
            for path_key, hash_key in (
                ("prediction_path", "prediction_sha256"),
                ("diagnostic_path", "diagnostic_sha256"),
                ("trace_path", "trace_sha256"),
                ("corruption_audit_path", "corruption_audit_sha256"),
            ):
                if path_key not in record:
                    continue
                path = Path(record[path_key])
                if not path.is_file() or sha256_file(path) != record[hash_key]:
                    raise ValueError("S2 resume scene evidence hash mismatch")
            completed.append(record)
        progress["completed_scenes"] = completed
        progress["completed_frame_count"] = sum(
            int(record["frame_count"]) for record in completed
        )
    atomic_write_json(progress_path, progress)

    completed_scene_tokens = {record["scene_token"] for record in completed}
    frame_total = int(progress["completed_frame_count"])
    role_switches = Counter()
    accepted_total = sum(int(record.get("accepted_count", 0)) for record in completed)
    with torch.no_grad():
        for scene in scenes:
            if scene in completed_scene_tokens:
                continue
            scene_key = hashlib.sha256(scene.encode("utf-8")).hexdigest()[:16]
            prediction_path = output_dir / "scenes" / f"scene_{scene_key}_predictions.pkl"
            diagnostic_path = output_dir / "scenes" / f"scene_{scene_key}_diagnostics.npz"
            trace_path = output_dir / "scenes" / f"scene_{scene_key}_traces.json"
            audit_path = output_dir / "scenes" / f"scene_{scene_key}_corruption_audit.jsonl"
            if any(path.exists() for path in (prediction_path, diagnostic_path, trace_path, audit_path)):
                raise FileExistsError("partial scene evidence requires a new immutable attempt")
            os.environ["VISFUSE3D_CORE4_AUDIT_PATH"] = str(audit_path.resolve())
            predictions = {}
            arrays: dict[str, list] = {"frame_token": [], "input_sha256": []}
            trace_frames = []
            scene_accepted = 0
            for dataset_index in grouped[scene]:
                token = str(dataset.data_infos[dataset_index]["token"])
                if token_filter is not None and token not in token_filter:
                    continue
                if args.max_frames_total is not None and frame_total >= args.max_frames_total:
                    break
                set_random_seed(args.seed, deterministic=True)
                sample = _prepare_exact_sample(dataset, dataset_index)
                batch = scatter(collate([sample], samples_per_gpu=1), [args.gpu_id])[0]
                observed = str(batch["img_metas"][0]["sample_idx"])
                if observed != token:
                    raise RuntimeError("prepared frame token drifted")
                input_sha256 = _frame_input_hash(batch, token)
                try:
                    if args.mode == "s2a":
                        result = model.forward_qta_context_preserving_output_oracle(
                            points=batch["points"],
                            img_metas=batch["img_metas"],
                            img=batch["img"],
                            gt_bboxes_3d=batch["gt_bboxes_3d"],
                            gt_labels_3d=batch["gt_labels_3d"],
                            epsilon_query_keep=margins["epsilon_query_keep"],
                            epsilon_query_reconstruction=margins[
                                "epsilon_query_reconstruction"
                            ],
                            return_predictions=True,
                        )
                    else:
                        result = model.forward_qta_greedy_joint_oracle(
                            points=batch["points"],
                            img_metas=batch["img_metas"],
                            img=batch["img"],
                            gt_bboxes_3d=batch["gt_bboxes_3d"],
                            gt_labels_3d=batch["gt_labels_3d"],
                            epsilon_query=margins["epsilon_query_reconstruction"],
                            proxy_overestimate=margins["proxy_overestimate"],
                            epsilon_frame=margins["epsilon_frame"],
                            candidate_k=margins["K"],
                            return_predictions=True,
                        )
                except Exception as exc:
                    atomic_write_json(
                        output_dir / "engineering_failure_manifest.json",
                        {
                            "schema": "visfuse3d_stage019_s2_engineering_failure_v1",
                            "status": "engineering_failure",
                            "condition": args.condition,
                            "mode": args.mode,
                            "scene_token": scene,
                            "frame_token": token,
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                            "formal_merge_allowed": False,
                        },
                    )
                    raise
                arrays["frame_token"].append(token)
                arrays["input_sha256"].append(input_sha256)
                if args.mode == "s2a":
                    if int(result["decoder_call_count"]) != 4:
                        raise RuntimeError("S2-A decoder call count is not exactly four")
                    if not result["all_keep_tensor_exact"] or not result["all_keep_bbox_exact"]:
                        raise RuntimeError("S2-A all-KEEP identity audit failed")
                    predictions[token] = {
                        role: _cpu_bbox(result["bbox_results"][role][0])
                        for role in S2A_ROLES
                    }
                    original_identity = result["query_identity"]["original_mome"]
                    route_identity = result["query_identity"]["full_context_experts"]
                    arrays.setdefault("query_identity", []).append(
                        torch.stack([original_identity[0]] + [value[0] for value in route_identity])
                        .detach().cpu().numpy().astype(np.int16)
                    )
                    tensor_keys = {
                        "original_routes": result["original_routes"][0],
                        "keep_query_losses": result["keep_query_losses"]["total"][0],
                        "full_context_query_losses": result["full_context_query_losses"][0],
                        "keep_anchored_actions": result["keep_anchored_actions"][0],
                        "keep_anchored_switch_mask": result["keep_anchored_switch_mask"][0],
                        "keep_anchored_lower_bounds": result["keep_anchored_lower_bounds"][0],
                        "reconstruction_query_losses": result["reconstruction_query_losses"]["total"][0],
                        "portfolio_actions": result["portfolio_actions"][0],
                        "portfolio_switch_mask": result["portfolio_switch_mask"][0],
                        "portfolio_lower_bounds": result["portfolio_lower_bounds"][0],
                        "fused_actions": result["fused_hungarian_sensitivity"]["actions"][0],
                        "fused_route_agreement": result["fused_hungarian_sensitivity"]["route_agreement"][0],
                    }
                    for key, value in tensor_keys.items():
                        arrays.setdefault(key, []).append(value.detach().cpu().numpy())
                    for role in S2A_ROLES:
                        arrays.setdefault(f"rematched_{role}", []).append(
                            float(result["rematched_group_losses"][role][0].item())
                        )
                    role_switches["keep_anchored_output_oracle"] += int(
                        result["keep_anchored_switch_mask"].sum().item()
                    )
                    role_switches["pure_three_expert_portfolio_oracle"] += int(
                        result["portfolio_switch_mask"].sum().item()
                    )
                else:
                    if int(result["decoder_call_count"]) > 4 + margins["K"]:
                        raise RuntimeError("S2-B decoder call count exceeded 4+K")
                    predictions[token] = {
                        S2B_ROLES[0]: _cpu_bbox(
                            result["original_mome_greedy_run_bbox_results"][0]
                        ),
                        S2B_ROLES[1]: _cpu_bbox(
                            result["greedy_joint_routing_gt_oracle_result_bbox_results"][0]
                        ),
                    }
                    for key in ("base_routes", "final_routes"):
                        arrays.setdefault(key, []).append(
                            result[key][0].detach().cpu().numpy().astype(np.int8)
                        )
                    arrays.setdefault("base_rematched_group_loss", []).append(
                        float(result["base_rematched_group_loss"][0].item())
                    )
                    arrays.setdefault("final_rematched_group_loss", []).append(
                        float(result["final_rematched_group_loss"][0].item())
                    )
                    trace = _greedy_trace(result)
                    accepted = int(result["accepted_count"])
                    scene_accepted += accepted
                    accepted_total += accepted
                    trace_frames.append(
                        {
                            "frame_token": token,
                            "input_sha256": input_sha256,
                            "K": margins["K"],
                            "candidate_order": [
                                {
                                    "query_id": int(value[0]),
                                    "destination_expert": int(value[1]),
                                    "quality": float(value[2]),
                                }
                                for value in result["candidate_order"]
                            ],
                            "base_route_hash": result["base_route_hash"],
                            "final_route_hash": result["final_route_hash"],
                            "accepted_count": accepted,
                            "trace": trace,
                        }
                    )
                frame_total += 1
                progress.update(
                    {
                        "current_scene_token": scene,
                        "current_frame_token": token,
                        "completed_frame_count": frame_total,
                        "accepted_count": accepted_total,
                    }
                )
                atomic_write_json(progress_path, progress)
            if predictions:
                _atomic_pickle(prediction_path, predictions)
                encoded = {
                    "frame_token": np.asarray(arrays.pop("frame_token"), dtype="S32"),
                    "input_sha256": np.asarray(arrays.pop("input_sha256"), dtype="S64"),
                }
                for key, values in arrays.items():
                    encoded[key] = np.stack(values) if isinstance(values[0], np.ndarray) else np.asarray(values, dtype=np.float32)
                _atomic_savez(diagnostic_path, **encoded)
                if args.mode == "s2b":
                    atomic_write_json(
                        trace_path,
                        {
                            "schema": "visfuse3d_stage019_s2_greedy_scene_trace_v1",
                            "condition": args.condition,
                            "scene_token": scene,
                            "frames": trace_frames,
                        },
                    )
                record = {
                    "scene_token": scene,
                    "frame_count": len(predictions),
                    "frame_tokens": list(predictions),
                    "prediction_path": str(prediction_path.resolve()),
                    "prediction_sha256": sha256_file(prediction_path),
                    "diagnostic_path": str(diagnostic_path.resolve()),
                    "diagnostic_sha256": sha256_file(diagnostic_path),
                    "corruption_audit_path": str(audit_path.resolve()),
                    "corruption_audit_sha256": sha256_file(audit_path),
                    "accepted_count": scene_accepted,
                }
                if args.mode == "s2b":
                    record.update(
                        {
                            "trace_path": str(trace_path.resolve()),
                            "trace_sha256": sha256_file(trace_path),
                        }
                    )
                completed.append(record)
                progress["completed_scenes"] = completed
                atomic_write_json(progress_path, progress)
            if args.max_frames_total is not None and frame_total >= args.max_frames_total:
                break

    observed_tokens = {
        token for record in completed for token in record.get("frame_tokens", [])
    }
    if token_filter is not None and observed_tokens != token_filter:
        missing = sorted(token_filter - observed_tokens)
        extra = sorted(observed_tokens - token_filter)
        raise RuntimeError(f"token filter coverage mismatch: missing={missing[:3]}, extra={extra[:3]}")
    expected_full = token_filter is None and args.max_frames_total is None
    expected_frames = sum(len(scene_map["scene_to_tokens"][scene]) for scene in scenes)
    if expected_full and frame_total != expected_frames:
        raise RuntimeError("worker scene shard frame coverage is incomplete")
    manifest = {
        "schema": "visfuse3d_stage019_s2_oracle_worker_manifest_v1",
        "status": "complete" if args.max_frames_total is None else "smoke_complete",
        "claim_boundary": (
            "s2a_gt_informed_context_preserving_output_oracle_shard_not_official_result"
            if args.mode == "s2a"
            else "locked_gt_greedy_reachable_result_shard_not_global_upper_bound"
        ),
        "input_identity": input_identity,
        "roles": list(S2A_ROLES if args.mode == "s2a" else S2B_ROLES),
        "frame_count": frame_total,
        "query_count_per_frame": EXPECTED_QUERY_COUNT,
        "accepted_count": accepted_total,
        "switch_counts": dict(role_switches),
        "engineering_failure_count": 0,
        "scenes": completed,
    }
    manifest_path = output_dir / "s2_oracle_worker_manifest.json"
    atomic_write_json(manifest_path, manifest)
    progress.update(
        {
            "status": manifest["status"],
            "current_scene_token": None,
            "current_frame_token": None,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
        }
    )
    atomic_write_json(progress_path, progress)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
