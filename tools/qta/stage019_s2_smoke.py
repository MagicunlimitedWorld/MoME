"""One-frame fail-closed smoke for the two Stage019-S2 model interfaces."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

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
    from .full_query_oracle_worker import (
        _build_dataset_cfg,
        _scene_dataset_indices,
    )
    from .route_snapshot_worker import recursive_sha256
    from .stage019_s2_oracle_worker import (
        _gpu_identity as _locked_gpu_identity,
        _load_margins as _load_locked_margins,
    )
    from .stage019_s2_training_conditions import (
        S3_ACTIONABLE_CONDITIONS,
        S3_HARD_BYPASS_CONDITIONS,
    )
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
    from full_query_oracle_worker import _build_dataset_cfg, _scene_dataset_indices
    from route_snapshot_worker import recursive_sha256
    from stage019_s2_oracle_worker import (
        _gpu_identity as _locked_gpu_identity,
        _load_margins as _load_locked_margins,
    )
    from stage019_s2_training_conditions import (
        S3_ACTIONABLE_CONDITIONS,
        S3_HARD_BYPASS_CONDITIONS,
    )


SUPPORTED_RUN_IDS = {
    (
        "2026-08-17-mome-stage019-s2-context-preserving-output-oracle-and-"
        "greedy-joint-gt-oracle-v1"
    ),
    (
        "2026-08-18-mome-stage019-s2-r3-context-preserving-output-oracle-and-"
        "greedy-joint-gt-oracle-v1"
    ),
    "2026-08-19-mome-stage019-s3-actionable-hard-bypass-v1",
}
LEGACY_PROTOCOL_PROFILE = "stage019_s2_legacy_v1"
S3_PROTOCOL_PROFILE = "stage019_s3_actionable_hard_bypass_v1"
S3_RUN_ID = "2026-08-19-mome-stage019-s3-actionable-hard-bypass-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--scene-map", type=Path, required=True)
    parser.add_argument("--sample-scene-map", type=Path)
    parser.add_argument("--scene-list-manifest", type=Path)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--adapter-scripts", type=Path, required=True)
    parser.add_argument("--condition", default="clean")
    parser.add_argument(
        "--protocol-profile",
        choices=(LEGACY_PROTOCOL_PROFILE, S3_PROTOCOL_PROFILE),
        default=LEGACY_PROTOCOL_PROFILE,
    )
    parser.add_argument("--margins", type=Path)
    parser.add_argument("--training-clean", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--expected-cvd")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--expected-gpu-pci")
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def _gpu_identity(args: argparse.Namespace) -> dict:
    expected = (
        args.expected_cvd,
        args.expected_gpu_uuid,
        args.expected_gpu_pci,
    )
    if any(expected):
        if not all(expected):
            raise ValueError("locked GPU identity requires CVD, UUID, and PCI")
        return _locked_gpu_identity(args)
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    rows = subprocess.check_output(command, text=True).strip().splitlines()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {"cuda_visible_devices": visible, "nvidia_smi": rows}


def _complete_query_permutations(result, torch) -> int:
    identities = [result["query_identity"]["original_mome"][0]] + [
        value[0] for value in result["query_identity"]["full_context_experts"]
    ]
    expected = torch.arange(900, device=identities[0].device)
    for identity in identities:
        if identity.shape != (900,) or not torch.equal(
            torch.sort(identity.long()).values, expected
        ):
            raise RuntimeError("S3 smoke lost a complete 900-query permutation")
    return len(identities)


def _prediction_sha256(result: dict) -> str:
    prediction = result["bbox_results"]["original_mome"][0]
    boxes = prediction["boxes_3d"]
    return recursive_sha256(
        {
            "boxes_3d": boxes.tensor.detach().cpu(),
            "scores_3d": prediction["scores_3d"].detach().cpu(),
            "labels_3d": prediction["labels_3d"].detach().cpu(),
        }
    )


def _validate_s3_hard_bypass_input(
    args: argparse.Namespace, batch: dict, torch
) -> dict:
    meta = batch["img_metas"][0]
    if meta.get("qta_hard_bypass") is not True:
        raise RuntimeError("complete-zero smoke input did not set qta_hard_bypass")
    expected_routes = {
        "lidar_zero": [False, False, True],
        "camera_zero": [False, True, False],
    }[args.condition]
    if list(meta.get("qta_route_available", ())) != expected_routes:
        raise RuntimeError("complete-zero route availability drifted")
    if args.condition == "lidar_zero":
        exact_zero = int(torch.count_nonzero(batch["points"][0]).item()) == 0
    else:
        exact_zero = int(torch.count_nonzero(batch["img"]).item()) == 0
    if not exact_zero:
        raise RuntimeError("complete-zero smoke input is not exactly zero")
    return {
        "qta_hard_bypass": True,
        "qta_route_available": expected_routes,
        "complete_modality_exact_zero": True,
    }


def _build_training_clean_dataset_cfg(cfg, args: argparse.Namespace):
    if args.condition != 'clean':
        raise ValueError('--training-clean only supports the clean condition')
    dataset_cfg = copy.deepcopy(cfg.data.train)
    if dataset_cfg.get('type') == 'CBGSDataset':
        dataset_cfg = dataset_cfg.dataset
    dataset_cfg.data_root = str(args.data_root.resolve()).replace('\\', '/') + '/'
    dataset_cfg.ann_file = str(args.ann_file.resolve()).replace('\\', '/')
    dataset_cfg.test_mode = False
    dataset_cfg = _force_exact_frame_sampling(dataset_cfg)
    dataset_cfg = _force_identity_view(dataset_cfg)
    forbidden = {
        'QtaLocalCorruption3D',
        'Core4CorruptionAdapterV2',
        'PointShuffle',
        'PointsRangeFilter',
        'ModalMask3D',
        'GlobalRotScaleTransAll',
        'CustomRandomFlip3D',
    }
    dataset_cfg.pipeline = [
        copy.deepcopy(item)
        for item in dataset_cfg.pipeline
        if item.get('type') not in forbidden
    ]
    return dataset_cfg


def main() -> int:
    args = parse_args()
    source_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    if artifact_root.name not in SUPPORTED_RUN_IDS:
        raise ValueError("artifact root does not match a locked Stage019-S2 run")
    if (artifact_root.name == S3_RUN_ID) != (
        args.protocol_profile == S3_PROTOCOL_PROFILE
    ):
        raise ValueError("smoke protocol profile does not match artifact run id")
    if args.protocol_profile == S3_PROTOCOL_PROFILE and args.condition not in (
        *S3_ACTIONABLE_CONDITIONS,
        *S3_HARD_BYPASS_CONDITIONS,
    ):
        raise ValueError("condition is outside the locked S3 condition set")
    if args.protocol_profile == S3_PROTOCOL_PROFILE and args.margins is None:
        raise ValueError("S3 smoke requires locked margins")
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("output-dir must stay inside the S2 artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("S2 artifacts cannot be written in the source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable smoke output already exists: {output_dir}")
    if sha256_file(args.checkpoint) != EXPECTED_MOME_CHECKPOINT_SHA256:
        raise ValueError("frozen MoME checkpoint SHA256 mismatch")

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

    output_dir.mkdir(parents=True)
    cfg = Config.fromfile(str(args.config))
    _import_plugin(cfg, args.config)
    register_pipeline_registries()
    dataset_cfg = (
        _build_training_clean_dataset_cfg(cfg, args)
        if args.training_clean
        else _build_dataset_cfg(cfg, args, output_dir / "corruption_audit.jsonl")
    )
    dataset = build_dataset(dataset_cfg)
    scene_map = json.loads(args.scene_map.read_text(encoding="utf-8"))
    if args.scene_list_manifest is not None:
        if args.sample_scene_map is None:
            raise ValueError(
                '--sample-scene-map is required with --scene-list-manifest'
            )
        selection = json.loads(
            args.scene_list_manifest.read_text(encoding="utf-8")
        )
        requested = selection.get('input_identity', {}).get('requested_scenes')
        if not isinstance(requested, list) or not requested:
            raise ValueError('scene-list manifest lacks requested_scenes')
        scene = str(requested[0])
        sample_map = json.loads(
            args.sample_scene_map.read_text(encoding="utf-8")
        )
        sample_to_scene = {
            str(token): str(value)
            for token, value in sample_map['sample_to_scene'].items()
        }
        dataset_index = next(
            index
            for index, info in enumerate(dataset.data_infos)
            if sample_to_scene.get(str(info['token'])) == scene
        )
    else:
        if args.scene_offset < 0 or args.scene_offset >= len(scene_map["scene_tokens"]):
            raise ValueError("scene-offset is outside the locked scene map")
        scene = str(scene_map["scene_tokens"][args.scene_offset])
        grouped = _scene_dataset_indices(dataset, scene_map, [scene])
        dataset_index = grouped[scene][0]

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
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable != 0:
        raise RuntimeError("S2 smoke requires every parameter to remain frozen")

    sample = _prepare_exact_sample(dataset, dataset_index)
    batch = scatter(collate([sample], samples_per_gpu=1), [args.gpu_id])[0]
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
    hard_bypass_input = None
    is_s3_hard_bypass = (
        args.protocol_profile == S3_PROTOCOL_PROFILE
        and args.condition in S3_HARD_BYPASS_CONDITIONS
    )
    if is_s3_hard_bypass:
        hard_bypass_input = _validate_s3_hard_bypass_input(args, batch, torch)

    if args.margins is not None:
        args.mode = "s2a"
        _, s2a_margins = _load_locked_margins(args)
    else:
        s2a_margins = {
            "epsilon_query_keep": 0.0,
            "epsilon_query_reconstruction": 0.0,
        }

    original_extract = model.extract_feat
    extract_calls = {"s2a": 0, "s2b": 0}
    active = {"name": "s2a"}

    def counted_extract(*values, **kwargs):
        extract_calls[active["name"]] += 1
        return original_extract(*values, **kwargs)

    model.extract_feat = counted_extract
    with torch.no_grad():
        s2a = model.forward_qta_context_preserving_output_oracle(
            points=batch["points"],
            img_metas=batch["img_metas"],
            img=batch["img"],
            gt_bboxes_3d=batch["gt_bboxes_3d"],
            gt_labels_3d=batch["gt_labels_3d"],
            epsilon_query_keep=s2a_margins["epsilon_query_keep"],
            epsilon_query_reconstruction=s2a_margins[
                "epsilon_query_reconstruction"
            ],
            return_predictions=True,
        )
        s2b = None
        if not is_s3_hard_bypass:
            active["name"] = "s2b"
            if args.margins is not None:
                args.mode = "s2b"
                _, s2b_margins = _load_locked_margins(args)
            else:
                s2b_margins = {
                    "epsilon_query_reconstruction": 0.0,
                    "proxy_overestimate": 0.0,
                    "epsilon_frame": 0.0,
                    "K": 4,
                }
            s2b = model.forward_qta_greedy_joint_oracle(
                points=batch["points"],
                img_metas=batch["img_metas"],
                img=batch["img"],
                gt_bboxes_3d=batch["gt_bboxes_3d"],
                gt_labels_3d=batch["gt_labels_3d"],
                epsilon_query=s2b_margins["epsilon_query_reconstruction"],
                proxy_overestimate=s2b_margins["proxy_overestimate"],
                epsilon_frame=s2b_margins["epsilon_frame"],
                candidate_k=s2b_margins["K"],
                return_predictions=True,
            )
    expected_extract_calls = {
        "s2a": 1,
        "s2b": 0 if is_s3_hard_bypass else 1,
    }
    if extract_calls != expected_extract_calls:
        raise RuntimeError(f"backbone call invariant failed: {extract_calls}")
    if int(s2a["decoder_call_count"]) != 4:
        raise RuntimeError("S2-A must use exactly four decoder executions")
    if not s2a["all_keep_tensor_exact"] or not s2a["all_keep_bbox_exact"]:
        raise RuntimeError("all-KEEP identity audit failed")
    query_permutation_count = _complete_query_permutations(s2a, torch)
    if s2b is not None:
        if int(s2b["decoder_call_count"]) > int(s2b["decoder_call_limit"]):
            raise RuntimeError("S2-B exceeded 4+K decoder executions")
        final_query_indices = s2b["final_query_indices"][0].long()
        if final_query_indices.shape != (900,) or not torch.equal(
            torch.sort(final_query_indices).values,
            torch.arange(900, device=final_query_indices.device),
        ):
            raise RuntimeError("S2-B lost its complete 900-query permutation")

    source_files = {
        "detector": source_root
        / "projects/mmdet3d_plugin/models/detectors/mome.py",
        "med": source_root / "projects/mmdet3d_plugin/models/dense_heads/med.py",
        "router": source_root
        / "projects/mmdet3d_plugin/models/utils/qta_router.py",
        "smoke": Path(__file__).resolve(),
    }
    manifest = {
        "schema": (
            "visfuse3d_stage019_s3_smoke_manifest_v1"
            if args.protocol_profile == S3_PROTOCOL_PROFILE
            else "visfuse3d_stage019_s2_smoke_manifest_v1"
        ),
        "status": "passed",
        "run_id": artifact_root.name,
        "protocol_profile": args.protocol_profile,
        "condition": args.condition,
        "training_clean": bool(args.training_clean),
        "scene_token": scene,
        "frame_token": frame_token,
        "input_sha256": input_sha256,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "annotation_sha256": sha256_file(args.ann_file),
        "scene_map_sha256": sha256_file(args.scene_map),
        "sample_scene_map_sha256": (
            sha256_file(args.sample_scene_map) if args.sample_scene_map else None
        ),
        "scene_list_manifest_sha256": (
            sha256_file(args.scene_list_manifest)
            if args.scene_list_manifest else None
        ),
        "gpu_identity": _gpu_identity(args),
        "trainable_parameter_count": trainable,
        "source_hashes": {
            name: sha256_file(path) for name, path in source_files.items()
        },
        "s2a": {
            "backbone_calls": extract_calls["s2a"],
            "decoder_calls": int(s2a["decoder_call_count"]),
            "keep_switch_count": int(
                s2a["keep_anchored_switch_mask"].sum().item()
            ),
            "portfolio_switch_count": int(
                s2a["portfolio_switch_mask"].sum().item()
            ),
            "all_keep_tensor_exact": bool(s2a["all_keep_tensor_exact"]),
            "all_keep_bbox_exact": bool(s2a["all_keep_bbox_exact"]),
            "output_names": sorted(s2a["bbox_results"]),
            "query_count": 900,
            "complete_query_permutation_count": query_permutation_count,
        },
    }
    if s2b is not None:
        manifest["s2b"] = {
            "applicability": "actionable_search",
            "K": int(s2b["K"]),
            "backbone_calls": extract_calls["s2b"],
            "decoder_calls": int(s2b["decoder_call_count"]),
            "decoder_call_limit": int(s2b["decoder_call_limit"]),
            "candidate_count": len(s2b["candidate_order"]),
            "accepted_count": int(s2b["accepted_count"]),
            "base_route_hash": s2b["base_route_hash"],
            "final_route_hash": s2b["final_route_hash"],
            "query_count": 900,
            "complete_query_permutation": True,
        }
    else:
        prediction_sha256 = _prediction_sha256(s2a)
        manifest["s2b"] = {
            "applicability": "hard_bypass_not_applicable",
            "search_executed": False,
            "input_identity": hard_bypass_input,
            "base_prediction_sha256": prediction_sha256,
            "final_prediction_sha256": prediction_sha256,
            "prediction_sha256_identical": True,
        }
    atomic_write_json(output_dir / "s2_smoke_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
