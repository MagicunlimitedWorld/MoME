"""Train the formal bounded CP-AFR attribute head for Stage019-S4."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import EXPECTED_MOME_CHECKPOINT_SHA256, is_relative_to
    from .stage019_s4_contracts import PROTOCOL_PROFILE, sha256_file
    from .stage019_s4_training_common import (
        atomic_torch_save,
        atomic_write_training_trace,
        balanced_epoch_indices,
        gather_batch,
        iter_balanced_batches,
        load_cache_arrays,
        load_formal_fusion_module,
        load_fusion_checkpoint,
        mean_condition_loss,
        resolve_deterministic_training_device,
        resolve_training_cache_authority,
        set_seed,
        state_dict_sha256,
        write_optimizer_provenance,
    )
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import EXPECTED_MOME_CHECKPOINT_SHA256, is_relative_to
    from stage019_s4_contracts import PROTOCOL_PROFILE, sha256_file
    from stage019_s4_training_common import (
        atomic_torch_save,
        atomic_write_training_trace,
        balanced_epoch_indices,
        gather_batch,
        iter_balanced_batches,
        load_cache_arrays,
        load_formal_fusion_module,
        load_fusion_checkpoint,
        mean_condition_loss,
        resolve_deterministic_training_device,
        resolve_training_cache_authority,
        set_seed,
        state_dict_sha256,
        write_optimizer_provenance,
    )


REQUIRED_KEYS = (
    "feature_matrix",
    "anchor_boxes",
    "aligned_expert_boxes",
    "expert_match_mask",
    "target_gt_boxes",
    "target_gt_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-cache", action="append", required=True)
    parser.add_argument("--calibration-cache", action="append", required=True)
    parser.add_argument("--g0-gate", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--gace-checkpoint", type=Path)
    parser.add_argument("--mode", choices=("attribute_only", "stacked"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _bounded_vector(delta: torch.Tensor, maximum: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    scale = torch.clamp(delta.new_tensor(maximum) / norm.clamp_min(1e-12), max=1.0)
    return delta * scale


def _loss(module, payload: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    features = payload["feature_matrix"].float()
    anchor = payload["anchor_boxes"].float()
    aligned = payload["aligned_expert_boxes"].float()
    match_mask = payload["expert_match_mask"].bool()
    target = payload["target_gt_boxes"].float()
    valid = payload["target_gt_valid"].bool() & match_mask.any(dim=1)
    if not torch.any(valid):
        zero = module.cp_afr(features).sum() * 0.0
        return zero, {key: 0.0 for key in ("center", "size", "yaw", "velocity", "gate")}
    output = module.cp_afr(features)
    attribute_logits = output[:, :16].reshape(-1, 4, 4)
    accept_probability = torch.sigmoid(output[:, 16])
    source_valid = torch.cat(
        (
            torch.ones((len(features), 1), device=features.device, dtype=torch.bool),
            match_mask,
        ),
        dim=1,
    )
    weights = torch.softmax(
        attribute_logits.masked_fill(~source_valid[:, None, :], -torch.inf), dim=2
    )
    sources = torch.cat((anchor[:, None, :], aligned), dim=1)
    center_target = (weights[:, 0, :, None] * sources[:, :, :3]).sum(dim=1)
    center_delta = _bounded_vector(center_target - anchor[:, :3], 1.0)
    predicted_center = anchor[:, :3] + accept_probability[:, None] * center_delta

    source_log_size = sources[:, :, 3:6].log()
    size_target = (weights[:, 1, :, None] * source_log_size).sum(dim=1)
    anchor_log_size = anchor[:, 3:6].log()
    size_delta = (size_target - anchor_log_size).clamp(-0.20, 0.20)
    predicted_log_size = anchor_log_size + accept_probability[:, None] * size_delta

    source_yaw = sources[:, :, 6]
    yaw_target = torch.atan2(
        (weights[:, 2] * torch.sin(source_yaw)).sum(dim=1),
        (weights[:, 2] * torch.cos(source_yaw)).sum(dim=1),
    )
    yaw_delta = torch.atan2(
        torch.sin(yaw_target - anchor[:, 6]), torch.cos(yaw_target - anchor[:, 6])
    ).clamp(-math.pi / 9.0, math.pi / 9.0)
    predicted_yaw = anchor[:, 6] + accept_probability * yaw_delta

    velocity_target = (weights[:, 3, :, None] * sources[:, :, 7:9]).sum(dim=1)
    velocity_delta = _bounded_vector(velocity_target - anchor[:, 7:9], 2.0)
    predicted_velocity = anchor[:, 7:9] + accept_probability[:, None] * velocity_delta

    center = functional.smooth_l1_loss(predicted_center[valid], target[valid, :3])
    size = functional.smooth_l1_loss(predicted_log_size[valid], target[valid, 3:6].log())
    yaw = torch.mean(1.0 - torch.cos(predicted_yaw[valid] - target[valid, 6]))
    velocity = functional.smooth_l1_loss(
        predicted_velocity[valid], target[valid, 7:9]
    )
    gate = accept_probability.mean()
    total = center + 0.5 * size + 0.2 * yaw + 0.2 * velocity + 0.1 * gate
    terms = {
        "center": float(center.detach().cpu().item()),
        "size": float(size.detach().cpu().item()),
        "yaw": float(yaw.detach().cpu().item()),
        "velocity": float(velocity.detach().cpu().item()),
        "gate": float(gate.detach().cpu().item()),
    }
    return total, terms


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("CP-AFR output must stay inside artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("CP-AFR output cannot enter source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable CP-AFR output exists: {output_dir}")
    device = resolve_deterministic_training_device(args.device)
    if sha256_file(args.frozen_checkpoint) != EXPECTED_MOME_CHECKPOINT_SHA256:
        raise ValueError("CP-AFR frozen MoME checkpoint SHA256 mismatch")
    if (args.mode == "stacked") != (args.gace_checkpoint is not None):
        raise ValueError("stacked mode requires exactly one GACE checkpoint")
    gate = json.loads(args.g0_gate.read_text(encoding="utf-8"))
    if (
        gate.get("schema") != "visfuse3d_stage019_s4_g0_gate_manifest_v1"
        or gate.get("protocol_profile") != PROTOCOL_PROFILE
        or not bool(gate.get("training_authorized"))
        or not bool(gate.get("g2_authorized"))
    ):
        raise ValueError("CP-AFR training is not authorized by the G0-attribute gate")
    fit_paths, calibration_paths, cache_authorities = resolve_training_cache_authority(
        args.fit_cache, args.calibration_cache, args.inputs_manifest
    )
    fit = load_cache_arrays(fit_paths, REQUIRED_KEYS)
    calibration = load_cache_arrays(calibration_paths, REQUIRED_KEYS)
    set_seed(args.seed)
    formal = load_formal_fusion_module()
    module = formal.ObjectSetAttributeFusion().to(device)
    parent_gace = None
    if args.gace_checkpoint is not None:
        parent_payload = load_fusion_checkpoint(args.gace_checkpoint, module, formal)
        if (
            parent_payload.get("schema")
            != "visfuse3d_stage019_s4_gace_checkpoint_v1"
            or parent_payload.get("method") != "gace_lite"
            or parent_payload.get("protocol_profile") != PROTOCOL_PROFILE
            or parent_payload.get("feature_version") != formal.FEATURE_VERSION
            or tuple(parent_payload.get("feature_names", ()))
            != tuple(formal.FEATURE_NAMES)
            or int(parent_payload.get("seed", -1)) != int(args.seed)
        ):
            raise ValueError("stacked CP-AFR parent GACE identity mismatch")
        parent_gace = {
            "path": str(args.gace_checkpoint.resolve()),
            "sha256": sha256_file(args.gace_checkpoint),
            "state_dict_sha256": parent_payload.get("state_dict_sha256"),
        }
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for parameter in module.cp_afr.parameters():
        parameter.requires_grad_(True)
    trainable_names = [
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=5e-4,
        weight_decay=1e-4,
    )
    output_dir.mkdir(parents=True)
    authority = {"g0_gate": args.g0_gate, **cache_authorities}
    if args.gace_checkpoint is not None:
        authority["gace_checkpoint"] = args.gace_checkpoint
    write_optimizer_provenance(
        output_dir,
        method=args.mode,
        seed=args.seed,
        fit_paths=fit_paths,
        calibration_paths=calibration_paths,
        model=module,
        optimizer=optimizer,
        loss_contract={
            "center_smooth_l1": 1.0,
            "height_dimension_log_smooth_l1": 0.5,
            "yaw_cosine": 0.2,
            "velocity_smooth_l1": 0.2,
            "accept_gate_l1_sparsity": 0.1,
            "residual_caps": {
                "center_m": 1.0,
                "dimension_log": 0.20,
                "yaw_degrees": 20.0,
                "velocity_mps": 2.0,
            },
        },
        frozen_checkpoint=args.frozen_checkpoint,
        trainable_names=trainable_names,
        batch_size=2048,
        trainer_path=Path(__file__),
        config_path=args.config,
        authority_manifests=authority,
    )
    rng = np.random.RandomState(args.seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0
    history = []
    optimizer_steps = 0
    for epoch in range(100):
        module.train()
        indices = balanced_epoch_indices(fit, rng)
        losses = []
        term_totals = {key: [] for key in ("center", "size", "yaw", "velocity", "gate")}
        for rows in iter_balanced_batches(fit, indices, 2048, rng):
            batch = gather_batch(fit, rows, REQUIRED_KEYS, device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = _loss(module, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("CP-AFR training produced non-finite loss")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach().cpu().item()))
            for key, value in terms.items():
                term_totals[key].append(value)
        module.eval()
        calibration_loss = mean_condition_loss(
            calibration, lambda payload: _loss(module, payload)[0], device
        )
        history.append(
            {
                "epoch": epoch,
                "fit_loss": float(np.mean(losses)),
                "calibration_loss": calibration_loss,
                "optimizer_steps": optimizer_steps,
                **{
                    f"fit_{key}_loss": float(np.mean(values))
                    for key, values in term_totals.items()
                },
            }
        )
        if calibration_loss < best_loss - 1e-8:
            best_loss = calibration_loss
            best_epoch = epoch
            best_state = copy.deepcopy(module.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    if best_state is None or optimizer_steps <= 0:
        raise RuntimeError("CP-AFR training performed no valid optimizer step")
    module.load_state_dict(best_state, strict=True)
    trace_path = output_dir / "training_trace.csv"
    atomic_write_training_trace(trace_path, history)
    checkpoint_path = output_dir / "stage019_s4_cp_afr_checkpoint.pth"
    atomic_torch_save(
        checkpoint_path,
        {
            formal.CHECKPOINT_STATE_DICT_KEY: module.state_dict(),
            "schema": "visfuse3d_stage019_s4_cp_afr_checkpoint_v1",
            "protocol_profile": PROTOCOL_PROFILE,
            "method": args.mode,
            "seed": int(args.seed),
            "feature_version": formal.FEATURE_VERSION,
            "feature_names": formal.FEATURE_NAMES,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            ),
            "state_dict_sha256": state_dict_sha256(module.state_dict()),
            "parent_gace_checkpoint": parent_gace,
        },
    )
    manifest = {
        "schema": "visfuse3d_stage019_s4_cp_afr_training_manifest_v1",
        "status": "complete_training_awaiting_calibration_metric_gate",
        "protocol_profile": PROTOCOL_PROFILE,
        "mode": args.mode,
        "seed": int(args.seed),
        "seed_role": "primary" if int(args.seed) == 20260710 else "confirmation",
        "best_epoch": best_epoch,
        "best_calibration_loss": best_loss,
        "optimizer_step_count": optimizer_steps,
        "epochs_ran": len(history),
        "history": history,
        "training_trace": {
            "path": str(trace_path.resolve()),
            "sha256": sha256_file(trace_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "state_dict_sha256": state_dict_sha256(module.state_dict()),
            "top_level_state_dict_key": formal.CHECKPOINT_STATE_DICT_KEY,
        },
        "parent_gace_checkpoint": parent_gace,
        "claim_boundary": "training_completion_only_not_calibration_or_validation_success",
    }
    atomic_write_json(output_dir / "stage019_s4_cp_afr_training_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
