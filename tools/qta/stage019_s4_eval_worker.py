"""Apply a locked S4 small-head checkpoint to immutable expert extraction."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import is_relative_to
    from .full_query_oracle_worker import _atomic_pickle, _cpu_bbox
    from .stage019_s4_contracts import EXPERT_ROLES, HARD_BYPASS_CONDITIONS, PROTOCOL_PROFILE, sha256_file
    from .stage019_s4_training_common import load_formal_fusion_module, load_fusion_checkpoint
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import is_relative_to
    from full_query_oracle_worker import _atomic_pickle, _cpu_bbox
    from stage019_s4_contracts import EXPERT_ROLES, HARD_BYPASS_CONDITIONS, PROTOCOL_PROFILE, sha256_file
    from stage019_s4_training_common import load_formal_fusion_module, load_fusion_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-worker-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("score_only", "attribute_only", "stacked"), required=True)
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--fusion-strength", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _to_device_bbox(value: dict, device: torch.device) -> dict:
    boxes = value["boxes_3d"].to(device)
    return {
        "boxes_3d": boxes,
        "scores_3d": value["scores_3d"].to(device),
        "labels_3d": value["labels_3d"].to(device),
    }


def _interpolate_scores(anchor: dict, candidate: dict, strength: float) -> dict:
    output = dict(candidate)
    output["scores_3d"] = anchor["scores_3d"] + float(strength) * (
        candidate["scores_3d"] - anchor["scores_3d"]
    )
    return output


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("S4 eval output must stay inside artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("S4 eval output cannot enter source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable S4 eval worker exists: {output_dir}")
    if float(args.fusion_strength) not in (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("S4 fusion strength must be in the preregistered five-value grid")
    if args.mode == "attribute_only" and float(args.fusion_strength) != 1.0:
        raise ValueError("attribute-only evaluation does not use the G1 strength grid")
    extract = json.loads(args.extract_worker_manifest.read_text(encoding="utf-8"))
    identity = extract.get("input_identity", {})
    condition = str(extract.get("condition"))
    if (
        extract.get("schema") != "visfuse3d_stage019_s4_extract_worker_manifest_v1"
        or extract.get("status") != "complete"
        or identity.get("protocol_profile") != PROTOCOL_PROFILE
        or int(extract.get("engineering_failure_count", -1)) != 0
    ):
        raise ValueError("S4 eval input extraction is incomplete")
    hard_bypass = bool(extract.get("observed_hard_bypass"))
    if hard_bypass != bool(extract.get("condition_expected_hard_bypass")):
        raise ValueError("S4 extract observed/expected hard-bypass audit differs")
    if hard_bypass and args.head_checkpoint is not None:
        raise ValueError("complete-zero eval must not load or run a fusion head")
    if not hard_bypass and args.head_checkpoint is None:
        raise ValueError("actionable S4 eval requires a locked head checkpoint")
    device = torch.device(args.device)
    formal = load_formal_fusion_module()
    module = None
    checkpoint_payload = None
    if not hard_bypass:
        module = formal.ObjectSetAttributeFusion().to(device)
        checkpoint_payload = load_fusion_checkpoint(args.head_checkpoint, module, formal)
        expected = {
            "attribute_only": (
                "visfuse3d_stage019_s4_cp_afr_checkpoint_v1",
                "attribute_only",
            ),
            "stacked": (
                "visfuse3d_stage019_s4_cp_afr_checkpoint_v1",
                "stacked",
            ),
        }.get(args.mode)
        if args.mode == "score_only":
            score_identities = {
                (
                    "visfuse3d_stage019_s4_gace_checkpoint_v1",
                    "gace_lite",
                ),
                (
                    "visfuse3d_stage019_s4_temperature_checkpoint_v1",
                    "temperature_scaling_baseline",
                ),
            }
            method_identity_ok = (
                checkpoint_payload.get("schema"), checkpoint_payload.get("method")
            ) in score_identities
            if (
                checkpoint_payload.get("method") == "temperature_scaling_baseline"
                and extract.get("phase") != "calibration"
            ):
                method_identity_ok = False
        else:
            method_identity_ok = (
                checkpoint_payload.get("schema"), checkpoint_payload.get("method")
            ) == expected
        if (
            not method_identity_ok
            or checkpoint_payload.get("protocol_profile") != PROTOCOL_PROFILE
            or checkpoint_payload.get("feature_version") != formal.FEATURE_VERSION
            or tuple(checkpoint_payload.get("feature_names", ()))
            != tuple(formal.FEATURE_NAMES)
            or int(checkpoint_payload.get("seed", -1)) not in (20260710, 20260711)
            or (
                extract.get("phase") in ("pilot", "fullval")
                and int(checkpoint_payload.get("seed", -1)) != 20260710
            )
        ):
            raise ValueError("S4 checkpoint method/schema/feature/seed identity mismatch")
        module.eval()
        if module.trainable_parameter_count() > 50_000:
            raise RuntimeError("S4 eval head exceeds 50k parameters")
    output_dir.mkdir(parents=True)
    scene_records = []
    all_tokens = []
    total_changed = 0
    total_accepted = 0
    module_seconds = 0.0
    module_object_count = 0
    for scene in extract["scenes"]:
        source_prediction = Path(scene["prediction_path"])
        source_diagnostic = Path(scene["diagnostic_path"])
        if (
            sha256_file(source_prediction) != scene["prediction_sha256"]
            or sha256_file(source_diagnostic) != scene["diagnostic_sha256"]
        ):
            raise ValueError("S4 eval source scene evidence hash mismatch")
        with source_prediction.open("rb") as stream:
            raw_predictions = pickle.load(stream)  # noqa: S301 - locked local evidence.
        with np.load(source_diagnostic, allow_pickle=False) as arrays:
            offsets = (
                np.asarray(arrays["frame_object_offsets"], dtype=np.int64)
                if not hard_bypass
                else np.zeros((len(raw_predictions) + 1,), dtype=np.int64)
            )
            output_predictions = {}
            frame_audits = []
            for frame_index, (token, roles) in enumerate(raw_predictions.items()):
                original_cpu = roles["original_mome"]
                if hard_bypass:
                    final_cpu = original_cpu
                    score_cpu = original_cpu
                    audit = {
                        "status": "hard_bypass_identity",
                        "mode": args.mode,
                        "hard_bypass": True,
                        "accepted_count": 0,
                        "changed_count": 0,
                    }
                else:
                    start, stop = int(offsets[frame_index]), int(offsets[frame_index + 1])
                    original = _to_device_bbox(original_cpu, device)
                    experts = {
                        role: _to_device_bbox(roles[role], device) for role in EXPERT_ROLES
                    }
                    evidence = {
                        "object_distance": torch.as_tensor(
                            arrays["object_distance"][start:stop], device=device
                        ),
                        "box_point_count": torch.as_tensor(
                            arrays["box_point_count"][start:stop], device=device
                        ),
                        "neighborhood_point_count": torch.as_tensor(
                            arrays["neighborhood_point_count"][start:stop], device=device
                        ),
                        "visible_camera_views": torch.as_tensor(
                            arrays["visible_camera_views"][start:stop], device=device
                        ),
                    }
                    with torch.no_grad():
                        module_started = time.perf_counter()
                        if args.mode == "stacked":
                            full, audit = module(
                                original,
                                experts,
                                evidence,
                                mode="stacked",
                                hard_bypass=False,
                                enable_rescue=False,
                            )
                            score_full, _ = module(
                                original,
                                experts,
                                evidence,
                                mode="score_only",
                                hard_bypass=False,
                                enable_rescue=False,
                            )
                            score = _interpolate_scores(
                                original, score_full, args.fusion_strength
                            )
                            final = dict(full)
                            final["scores_3d"] = score["scores_3d"]
                        else:
                            full, audit = module(
                                original,
                                experts,
                                evidence,
                                mode=args.mode,
                                hard_bypass=False,
                                enable_rescue=False,
                            )
                            final = (
                                _interpolate_scores(original, full, args.fusion_strength)
                                if args.mode == "score_only"
                                else full
                            )
                            score = (
                                final if args.mode == "score_only" else original
                            )
                        module_seconds += time.perf_counter() - module_started
                        module_object_count += int(original["scores_3d"].numel())
                    changed = int(
                        torch.count_nonzero(
                            (final["scores_3d"] != original["scores_3d"])
                            | torch.any(
                                final["boxes_3d"].tensor != original["boxes_3d"].tensor,
                                dim=1,
                            )
                        ).item()
                    )
                    accepted = changed
                    total_changed += changed
                    total_accepted += accepted
                    audit = dict(audit)
                    audit.update(
                        {
                            "changed_count": changed,
                            "accepted_count": accepted,
                            "fusion_strength": float(args.fusion_strength),
                            "rescue_enabled": False,
                        }
                    )
                    final_cpu = _cpu_bbox(final)
                    score_cpu = _cpu_bbox(score)
                output_predictions[token] = {
                    "original_mome": original_cpu,
                    **(
                        {"score_baseline": score_cpu}
                        if args.mode in ("attribute_only", "stacked")
                        else {}
                    ),
                    "final": final_cpu,
                }
                all_tokens.append(token)
                frame_audits.append({"frame_token": token, **audit})
        key = Path(scene["prediction_path"]).stem.replace("_predictions", "")
        prediction_path = output_dir / "scenes" / f"{key}_predictions.pkl"
        diagnostic_path = output_dir / "scenes" / f"{key}_diagnostics.json"
        audit_path = output_dir / "scenes" / f"{key}_audit.jsonl"
        _atomic_pickle(prediction_path, output_predictions)
        atomic_write_json(
            diagnostic_path,
            {
                "schema": "visfuse3d_stage019_s4_eval_scene_diagnostic_v1",
                "scene_token": scene["scene_token"],
                "condition": condition,
                "mode": args.mode,
                "fusion_strength": float(args.fusion_strength),
                "frames": frame_audits,
            },
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in frame_audits:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        scene_records.append(
            {
                "scene_token": scene["scene_token"],
                "frame_count": len(output_predictions),
                "frame_tokens": list(output_predictions),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path),
                "diagnostic_path": str(diagnostic_path.resolve()),
                "diagnostic_sha256": sha256_file(diagnostic_path),
                "corruption_audit_path": str(audit_path.resolve()),
                "corruption_audit_sha256": sha256_file(audit_path),
            }
        )
    manifest = {
        "schema": "visfuse3d_stage019_s4_eval_worker_manifest_v1",
        "status": "complete",
        "protocol_profile": PROTOCOL_PROFILE,
        "input_identity": identity,
        "phase": extract["phase"],
        "condition": condition,
        "mode": args.mode,
        "fusion_strength": float(args.fusion_strength),
        "scene_tokens": [scene["scene_token"] for scene in scene_records],
        "frame_tokens": all_tokens,
        "scene_count": len(scene_records),
        "frame_count": len(all_tokens),
        "unique_token_count": len(set(all_tokens)),
        "roles": list(next(iter(output_predictions.values()))),
        "hard_bypass": hard_bypass,
        "decoder_call_count_per_frame": 1 if hard_bypass else 4,
        "changed_object_count": total_changed,
        "accepted_object_count": total_accepted,
        "rescued_object_count": 0,
        "engineering_failure_count": 0,
        "resource_metrics": {
            "module_elapsed_seconds": module_seconds,
            "objects_per_second": (
                module_object_count / max(module_seconds, 1e-12)
                if module_object_count
                else None
            ),
            "object_count": module_object_count,
        },
        "head_checkpoint": (
            None
            if args.head_checkpoint is None
            else {
                "path": str(args.head_checkpoint.resolve()),
                "sha256": sha256_file(args.head_checkpoint),
                "schema": checkpoint_payload.get("schema"),
            }
        ),
        "source_hashes": {
            "eval_worker": sha256_file(Path(__file__).resolve()),
            "formal_fusion_module": sha256_file(
                Path(__file__).resolve().parents[2]
                / "projects"
                / "mmdet3d_plugin"
                / "models"
                / "utils"
                / "object_set_attribute_fusion.py"
            ),
        },
        "source_extract_worker": {
            "path": str(args.extract_worker_manifest.resolve()),
            "sha256": sha256_file(args.extract_worker_manifest),
        },
        "scenes": scene_records,
    }
    atomic_write_json(output_dir / "stage019_s4_eval_worker_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
