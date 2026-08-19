"""Run one immutable Stage019-S2 training20 calibration repeat."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
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
    from .stage019_s2_training_conditions import CONDITIONS
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
    from stage019_s2_training_conditions import CONDITIONS


EXPECTED_CLEAN_ANNOTATION_SOURCE_SHA256 = (
    "9214A60ADF903647DF027FE2F49CFBBB3CDF14DC4CE2C0B099337F7E92AB34A6"
)
EXPECTED_FRAMES = 803
EXPECTED_QUERIES = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training20-manifest", type=Path, required=True)
    parser.add_argument("--adapter-scripts", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=list(CONDITIONS))
    parser.add_argument("--repeat-id", required=True)
    parser.add_argument("--canonical-trace-root", type=Path)
    parser.add_argument("--locked-margins", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--expected-cvd", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-gpu-pci", required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-frames-total", type=int)
    parser.add_argument("--max-frames-per-condition", type=int)
    parser.add_argument("--frame-token")
    return parser.parse_args()


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


def _gpu_identity(args: argparse.Namespace) -> dict:
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
    matched = [
        row for row in rows
        if args.expected_gpu_uuid.lower() in row.lower()
        and args.expected_gpu_pci.lower().replace("00000000:", "")
        in row.lower().replace("00000000:", "")
    ]
    if len(matched) != 1:
        raise RuntimeError("expected calibration GPU UUID/PCI identity mismatch")
    if not matched[0].lstrip().startswith(f"{args.expected_cvd},"):
        raise RuntimeError(
            "calibration CVD physical index does not match locked GPU UUID/PCI"
        )
    return {
        "cuda_visible_devices": visible,
        "local_cuda_id": args.gpu_id,
        "expected_uuid": args.expected_gpu_uuid,
        "expected_pci": args.expected_gpu_pci,
        "nvidia_smi_row": matched[0],
    }


def _build_dataset_cfg(cfg, args, annotation: Path, condition: str,
                       audit_path: Path, beam_overlay_root: Path):
    dataset_cfg = copy.deepcopy(cfg.data.train)
    if dataset_cfg.get("type") == "CBGSDataset":
        dataset_cfg = dataset_cfg.dataset
    dataset_cfg.data_root = str(args.data_root.resolve()).replace("\\", "/") + "/"
    dataset_cfg.ann_file = str(annotation.resolve()).replace("\\", "/")
    dataset_cfg.test_mode = False
    dataset_cfg = _force_exact_frame_sampling(dataset_cfg)
    dataset_cfg = _force_identity_view(dataset_cfg)
    pipeline = []
    inserted = False
    forbidden = {
        "QtaLocalCorruption3D",
        "PointShuffle",
        "PointsRangeFilter",
    }
    for item in dataset_cfg.pipeline:
        item = copy.deepcopy(item)
        if item.get("type") in forbidden:
            continue
        pipeline.append(item)
        if item.get("type") == "LoadAnnotations3D":
            pipeline.append(
                {
                    "type": "Stage019S2TrainingConditionAdapter",
                    "condition": condition,
                    "mask_root": str(args.mask_root.resolve()),
                    "audit_path": str(audit_path.resolve()),
                    "beam_overlay_root": str(beam_overlay_root.resolve()),
                }
            )
            inserted = True
    if not inserted:
        raise RuntimeError("training20 pipeline lacks LoadAnnotations3D")
    dataset_cfg.pipeline = pipeline
    return dataset_cfg


def _tensor_float(value):
    return None if value is None else float(value.detach().cpu().item())


def _json_trace(result) -> list[dict]:
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
                "would_accept": bool(item.get("would_accept", False)),
                "accepted": bool(item["accepted"]),
                "trajectory_advanced": bool(
                    item.get("trajectory_advanced", item["accepted"])
                ),
                "reason": item["reason"],
            }
        )
    return rows


def _replay_payload(
    path: Path,
) -> tuple[list, list[bool], list[dict], np.ndarray, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    array_path = path.with_suffix(".npz")
    if not array_path.is_file():
        raise FileNotFoundError(array_path)
    with np.load(array_path, allow_pickle=False) as arrays:
        if "base_routes" not in arrays:
            raise RuntimeError(f"canonical replay lacks base_routes: {array_path}")
        base_routes = np.asarray(arrays["base_routes"]).copy()
    if base_routes.shape != (EXPECTED_QUERIES,):
        raise RuntimeError(
            "canonical replay base_routes must have shape "
            f"({EXPECTED_QUERIES},), got {base_routes.shape}"
        )
    if not np.issubdtype(base_routes.dtype, np.integer):
        raise RuntimeError("canonical replay base_routes must use an integer dtype")
    if np.any((base_routes < 0) | (base_routes > 2)):
        raise RuntimeError("canonical replay base_routes contain an invalid route")
    candidates = [
        (int(item["query_id"]), int(item["destination_expert"]))
        for item in payload["trace"]
    ]
    acceptance = [bool(item["accepted"]) for item in payload["trace"]]
    return (
        candidates,
        acceptance,
        payload["trace"],
        base_routes,
        str(payload["input_sha256"]),
    )


def main() -> int:
    args = parse_args()
    if (
        args.max_frames_per_condition is not None
        and args.max_frames_per_condition <= 0
    ):
        raise ValueError("max-frames-per-condition must be positive")
    source_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("calibration output must stay in the S2 artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("calibration output cannot enter the source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable calibration repeat exists: {output_dir}")
    if sha256_file(args.checkpoint) != EXPECTED_MOME_CHECKPOINT_SHA256:
        raise ValueError("frozen MoME checkpoint SHA256 mismatch")
    input_manifest = json.loads(
        args.training20_manifest.read_text(encoding="utf-8")
    )
    if input_manifest.get("status") != "passed" or int(
        input_manifest.get("frame_count", -1)
    ) != EXPECTED_FRAMES:
        raise ValueError("training20 input manifest is incomplete")
    clean_annotation = Path(input_manifest["clean_annotation"])
    beam_annotation = Path(input_manifest["beam4_annotation"])
    beam_overlay_root = Path(input_manifest["beam4_overlay_root"])
    locked_margins = None
    if args.locked_margins is not None:
        if args.canonical_trace_root is not None:
            raise ValueError("locked greedy audit cannot also be a calibration replay")
        locked_margins = json.loads(
            args.locked_margins.read_text(encoding="utf-8")
        )
        budget = locked_margins.get("candidate_budget", {})
        if (
            locked_margins.get("status") != "passed"
            or budget.get("status") != "coverage_reached"
            or int(budget.get("K", -1)) not in (4, 8, 16)
        ):
            raise ValueError("locked greedy audit requires passed margins and K")

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
    import stage019_s2_training_conditions  # noqa: F401

    gpu_identity = _gpu_identity(args)
    output_dir.mkdir(parents=True)
    cfg = Config.fromfile(str(args.config))
    _import_plugin(cfg, args.config)
    datasets = {}
    for condition in args.conditions:
        annotation = beam_annotation if condition == "beam_reduction_4" else clean_annotation
        dataset_cfg = _build_dataset_cfg(
            cfg,
            args,
            annotation,
            condition,
            output_dir / condition / "corruption_audit.jsonl",
            beam_overlay_root,
        )
        dataset = build_dataset(dataset_cfg)
        tokens = [str(info["token"]) for info in dataset.data_infos]
        if len(tokens) != EXPECTED_FRAMES or len(set(tokens)) != EXPECTED_FRAMES:
            raise RuntimeError(f"{condition} dataset is not 803 unique frames")
        datasets[condition] = dataset

    cfg.model.pretrained = None
    set_random_seed(args.seed, deterministic=True)
    model = build_model(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get(
        "CLASSES", next(iter(datasets.values())).CLASSES
    )
    model.requires_grad_(False)
    model = model.cuda(args.gpu_id)
    model.eval()
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable != 0:
        raise RuntimeError("calibration requires all model parameters frozen")

    progress_path = output_dir / "calibration_progress.json"
    progress = {
        "schema": "visfuse3d_stage019_s2_calibration_progress_v1",
        "status": "running",
        "repeat_id": args.repeat_id,
        "conditions": list(args.conditions),
        "completed_frame_conditions": 0,
        "gpu_identity": gpu_identity,
    }
    atomic_write_json(progress_path, progress)
    frame_condition_count = 0
    accepted_count = 0
    condition_records = []
    with torch.no_grad():
        for condition in args.conditions:
            dataset = datasets[condition]
            condition_dir = output_dir / condition
            completed = 0
            for dataset_index, info in enumerate(dataset.data_infos):
                if (
                    args.frame_token is not None
                    and str(info["token"]) != args.frame_token
                ):
                    continue
                if (
                    args.max_frames_per_condition is not None
                    and completed >= args.max_frames_per_condition
                ):
                    break
                if (
                    args.max_frames_total is not None
                    and frame_condition_count >= args.max_frames_total
                ):
                    break
                set_random_seed(args.seed, deterministic=True)
                sample = _prepare_exact_sample(dataset, dataset_index)
                batch = scatter(
                    collate([sample], samples_per_gpu=1), [args.gpu_id]
                )[0]
                frame_token = str(batch["img_metas"][0]["sample_idx"])
                input_sha256 = recursive_sha256(
                    {
                        "points": batch["points"][0],
                        "img": batch["img"],
                        "sample_idx": frame_token,
                        "lidar2img": batch["img_metas"][0]["lidar2img"],
                        "img_shape": batch["img_metas"][0]["img_shape"],
                    }
                )
                locked_candidates = None
                replay_acceptance = None
                canonical_trace = None
                locked_base_routes = None
                if args.canonical_trace_root is not None:
                    canonical_path = (
                        args.canonical_trace_root
                        / condition
                        / "frames"
                        / f"{frame_token}.json"
                    )
                    if not canonical_path.is_file():
                        raise FileNotFoundError(canonical_path)
                    (
                        locked_candidates,
                        replay_acceptance,
                        canonical_trace,
                        locked_base_routes,
                        canonical_input_sha256,
                    ) = _replay_payload(canonical_path)
                    if input_sha256 != canonical_input_sha256:
                        raise RuntimeError(
                            "calibration replay input hash drifted from canonical"
                        )
                result = model.forward_qta_greedy_joint_oracle(
                    points=batch["points"],
                    img_metas=batch["img_metas"],
                    img=batch["img"],
                    gt_bboxes_3d=batch["gt_bboxes_3d"],
                    gt_labels_3d=batch["gt_labels_3d"],
                    epsilon_query=(
                        locked_margins["epsilon_num_query"][condition][
                            "reconstruction"
                        ]
                        if locked_margins is not None
                        else 0.0
                    ),
                    proxy_overestimate=(
                        locked_margins["proxy_overestimate"][condition]
                        if locked_margins is not None
                        else 0.0
                    ),
                    epsilon_frame=(
                        locked_margins["epsilon_num_frame"][condition]
                        if locked_margins is not None
                        else 0.0
                    ),
                    candidate_k=(
                        int(locked_margins["candidate_budget"]["K"])
                        if locked_margins is not None
                        else 16
                    ),
                    locked_candidates=locked_candidates,
                    replay_acceptance=replay_acceptance,
                    locked_base_routes=(
                        torch.from_numpy(locked_base_routes)
                        if locked_base_routes is not None
                        else None
                    ),
                    return_predictions=False,
                )
                trace = _json_trace(result)
                accepted_count += int(result["accepted_count"])
                if canonical_trace is not None:
                    for observed, expected in zip(trace, canonical_trace):
                        for key in ("current_route_hash", "candidate_route_hash"):
                            if observed[key] != expected[key]:
                                raise RuntimeError(
                                    "calibration replay route hash drifted at "
                                    f"step={observed['step']}: {key}; "
                                    f"observed={observed[key]}, expected={expected[key]}"
                                )
                frame_dir = condition_dir / "frames"
                trace_payload = {
                    "schema": "visfuse3d_stage019_s2_calibration_trace_v1",
                    "repeat_id": args.repeat_id,
                    "condition": condition,
                    "frame_token": frame_token,
                    "input_sha256": input_sha256,
                    "candidate_order": [
                        {
                            "query_id": int(item[0]),
                            "destination_expert": int(item[1]),
                            "quality": float(item[2]),
                        }
                        for item in result["candidate_order"]
                    ],
                    "base_route_hash": result["base_route_hash"],
                    "final_route_hash": result["final_route_hash"],
                    "trace": trace,
                    "decoder_call_count": int(result["decoder_call_count"]),
                }
                atomic_write_json(frame_dir / f"{frame_token}.json", trace_payload)
                _atomic_savez(
                    frame_dir / f"{frame_token}.npz",
                    base_fixed_query_losses=result["base_fixed_query_losses"][0]
                    .detach().cpu().numpy().astype(np.float32),
                    route_fixed_query_losses=result["route_fixed_query_losses"][0]
                    .detach().cpu().numpy().astype(np.float32),
                    reconstruction_fixed_query_losses=result[
                        "reconstruction_fixed_query_losses"
                    ][0].detach().cpu().numpy().astype(np.float32),
                    full_context_gains=result["full_context_gains"][0]
                    .detach().cpu().numpy().astype(np.float32),
                    base_routes=result["base_routes"][0]
                    .detach().cpu().numpy().astype(np.int8),
                    candidate_quality=result["candidate_quality"][0]
                    .detach().cpu().numpy().astype(np.float32),
                )
                completed += 1
                frame_condition_count += 1
                progress["completed_frame_conditions"] = frame_condition_count
                progress["accepted_count"] = accepted_count
                progress["active_condition"] = condition
                progress["active_frame_token"] = frame_token
                atomic_write_json(progress_path, progress)
            if args.frame_token is not None and completed != 1:
                raise RuntimeError(
                    f"requested frame token was not completed for {condition}: "
                    f"{args.frame_token}"
                )
            condition_records.append(
                {"condition": condition, "completed_frames": completed}
            )

    expected_total = len(args.conditions) * EXPECTED_FRAMES
    smoke_limited = (
        args.max_frames_total is not None
        or args.max_frames_per_condition is not None
        or args.frame_token is not None
    )
    if not smoke_limited and frame_condition_count != expected_total:
        raise RuntimeError("calibration repeat did not cover every frame-condition")
    manifest = {
        "schema": "visfuse3d_stage019_s2_calibration_repeat_manifest_v1",
        "status": (
            "passed_locked_greedy_audit"
            if locked_margins is not None and not smoke_limited
            else "smoke_passed_locked_greedy_audit"
            if locked_margins is not None
            else "passed"
            if not smoke_limited
            else "smoke_passed"
        ),
        "repeat_id": args.repeat_id,
        "canonical_replay": args.canonical_trace_root is not None,
        "locked_greedy_audit": locked_margins is not None,
        "conditions": list(args.conditions),
        "frame_condition_count": frame_condition_count,
        "expected_full_frame_condition_count": expected_total,
        "max_frames_total": args.max_frames_total,
        "max_frames_per_condition": args.max_frames_per_condition,
        "frame_token": args.frame_token,
        "candidate_limit": (
            int(locked_margins["candidate_budget"]["K"])
            if locked_margins is not None
            else 16
        ),
        "accepted_count": accepted_count,
        "gpu_identity": gpu_identity,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
        "training20_manifest_sha256": sha256_file(args.training20_manifest),
        "locked_margins_sha256": (
            sha256_file(args.locked_margins) if args.locked_margins else None
        ),
        "worker_sha256": sha256_file(Path(__file__).resolve()),
        "detector_sha256": sha256_file(
            source_root / "projects/mmdet3d_plugin/models/detectors/mome.py"
        ),
        "med_sha256": sha256_file(
            source_root / "projects/mmdet3d_plugin/models/dense_heads/med.py"
        ),
        "router_sha256": sha256_file(
            source_root / "projects/mmdet3d_plugin/models/utils/qta_router.py"
        ),
        "trainable_parameter_count": trainable,
        "condition_records": condition_records,
    }
    atomic_write_json(output_dir / "calibration_repeat_manifest.json", manifest)
    progress["status"] = manifest["status"]
    progress["active_condition"] = None
    progress["active_frame_token"] = None
    atomic_write_json(progress_path, progress)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
