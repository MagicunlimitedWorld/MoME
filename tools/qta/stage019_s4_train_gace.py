"""Train the formal GACE-Lite score head on condition-balanced train objects."""

from __future__ import annotations

import argparse
import copy
import json
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
        mean_condition_loss,
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
        mean_condition_loss,
        resolve_training_cache_authority,
        set_seed,
        state_dict_sha256,
        write_optimizer_provenance,
    )


REQUIRED_KEYS = (
    "feature_matrix",
    "anchor_scores",
    "expert_match_mask",
    "anchor_soft_quality",
    "expert_soft_quality",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-cache", action="append", required=True)
    parser.add_argument("--calibration-cache", action="append", required=True)
    parser.add_argument("--g0-gate", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _loss(module, payload: dict[str, torch.Tensor]) -> torch.Tensor:
    features = payload["feature_matrix"].float()
    anchor_scores = payload["anchor_scores"].float()
    match_mask = payload["expert_match_mask"].bool()
    # GACE changes only the anchor score.  Its label must therefore describe
    # that fixed anchor candidate; borrowing a geometrically better expert's
    # label would promote an unchanged anchor false positive.
    target = payload["anchor_soft_quality"].float()
    has_match = match_mask.any(dim=1)
    prediction = module._apply_score_head(features, anchor_scores, has_match)
    if torch.any(has_match):
        prediction = prediction[has_match]
        target = target[has_match]
    return functional.binary_cross_entropy(
        prediction.clamp(1e-6, 1.0 - 1e-6), target
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("GACE output must stay inside artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("GACE output cannot enter source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable GACE output exists: {output_dir}")
    if sha256_file(args.frozen_checkpoint) != EXPECTED_MOME_CHECKPOINT_SHA256:
        raise ValueError("GACE frozen MoME checkpoint SHA256 mismatch")
    gate = json.loads(args.g0_gate.read_text(encoding="utf-8"))
    if (
        gate.get("schema") != "visfuse3d_stage019_s4_g0_gate_manifest_v1"
        or gate.get("protocol_profile") != PROTOCOL_PROFILE
        or not bool(gate.get("training_authorized"))
        or not bool(gate.get("g1_authorized"))
    ):
        raise ValueError("GACE training is not authorized by the locked G0-score gate")
    fit_paths, calibration_paths, cache_authorities = resolve_training_cache_authority(
        args.fit_cache, args.calibration_cache, args.inputs_manifest
    )
    fit = load_cache_arrays(fit_paths, REQUIRED_KEYS)
    calibration = load_cache_arrays(calibration_paths, REQUIRED_KEYS)
    device = torch.device(args.device)
    formal = load_formal_fusion_module()
    output_dir.mkdir(parents=True)

    # Required temperature-only baseline: identical data/seed/optimizer recipe,
    # zero GACE MLP frozen, and only log_temperature trainable.
    set_seed(args.seed)
    temperature_module = formal.ObjectSetAttributeFusion().to(device)
    for parameter in temperature_module.parameters():
        parameter.requires_grad_(False)
    temperature_module.log_temperature.requires_grad_(True)
    temperature_optimizer = torch.optim.AdamW(
        [temperature_module.log_temperature], lr=1e-3, weight_decay=1e-4
    )
    write_optimizer_provenance(
        output_dir,
        method="temperature_scaling_baseline",
        seed=args.seed,
        fit_paths=fit_paths,
        calibration_paths=calibration_paths,
        model=temperature_module,
        optimizer=temperature_optimizer,
        loss_contract={
            "type": "binary_cross_entropy",
            "target": (
                "fixed_anchor_candidate_fraction_passing_nuscenes_"
                "center_thresholds_0p5_1_2_4m"
            ),
            "gace_mlp": "frozen_exact_zero",
        },
        frozen_checkpoint=args.frozen_checkpoint,
        trainable_names=["log_temperature"],
        batch_size=4096,
        trainer_path=Path(__file__),
        config_path=args.config,
        authority_manifests={"g0_gate": args.g0_gate, **cache_authorities},
        manifest_name="temperature_optimizer_provenance_manifest.json",
    )
    temperature_rng = np.random.RandomState(args.seed)
    temperature_best_loss = float("inf")
    temperature_best_epoch = -1
    temperature_best_state = None
    temperature_patience = 0
    temperature_history = []
    temperature_steps = 0
    for epoch in range(50):
        temperature_module.train()
        indices = balanced_epoch_indices(fit, temperature_rng)
        epoch_losses = []
        for rows in iter_balanced_batches(fit, indices, 4096, temperature_rng):
            batch = gather_batch(fit, rows, REQUIRED_KEYS, device)
            temperature_optimizer.zero_grad(set_to_none=True)
            loss = _loss(temperature_module, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("temperature baseline produced non-finite loss")
            loss.backward()
            temperature_optimizer.step()
            temperature_steps += 1
            epoch_losses.append(float(loss.detach().cpu().item()))
        temperature_module.eval()
        calibration_loss = mean_condition_loss(
            calibration,
            lambda payload: _loss(temperature_module, payload),
            device,
        )
        temperature_history.append(
            {
                "epoch": epoch,
                "fit_loss": float(np.mean(epoch_losses)),
                "calibration_loss": calibration_loss,
                "optimizer_steps": temperature_steps,
            }
        )
        if calibration_loss < temperature_best_loss - 1e-8:
            temperature_best_loss = calibration_loss
            temperature_best_epoch = epoch
            temperature_best_state = copy.deepcopy(temperature_module.state_dict())
            temperature_patience = 0
        else:
            temperature_patience += 1
            if temperature_patience >= 5:
                break
    if temperature_best_state is None or temperature_steps <= 0:
        raise RuntimeError("temperature baseline performed no optimizer step")
    temperature_module.load_state_dict(temperature_best_state, strict=True)
    temperature_trace_path = output_dir / "temperature_training_trace.csv"
    atomic_write_training_trace(temperature_trace_path, temperature_history)
    temperature_checkpoint_path = output_dir / "stage019_s4_temperature_checkpoint.pth"
    atomic_torch_save(
        temperature_checkpoint_path,
        {
            formal.CHECKPOINT_STATE_DICT_KEY: temperature_module.state_dict(),
            "schema": "visfuse3d_stage019_s4_temperature_checkpoint_v1",
            "protocol_profile": PROTOCOL_PROFILE,
            "method": "temperature_scaling_baseline",
            "seed": int(args.seed),
            "feature_version": formal.FEATURE_VERSION,
            "feature_names": formal.FEATURE_NAMES,
            "trainable_parameter_names": ["log_temperature"],
            "trainable_parameter_count": 1,
            "state_dict_sha256": state_dict_sha256(temperature_module.state_dict()),
        },
    )

    # Reset RNG so GACE is independent of baseline optimizer consumption.
    set_seed(args.seed)
    module = formal.ObjectSetAttributeFusion().to(device)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for parameter in module.gace_lite.parameters():
        parameter.requires_grad_(True)
    module.log_temperature.requires_grad_(True)
    trainable_names = [
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )
    write_optimizer_provenance(
        output_dir,
        method="gace_lite",
        seed=args.seed,
        fit_paths=fit_paths,
        calibration_paths=calibration_paths,
        model=module,
        optimizer=optimizer,
        loss_contract={
            "type": "binary_cross_entropy",
            "target": (
                "fixed_anchor_candidate_fraction_passing_nuscenes_"
                "center_thresholds_0p5_1_2_4m"
            ),
        },
        frozen_checkpoint=args.frozen_checkpoint,
        trainable_names=trainable_names,
        batch_size=4096,
        trainer_path=Path(__file__),
        config_path=args.config,
        authority_manifests={"g0_gate": args.g0_gate, **cache_authorities},
    )
    rng = np.random.RandomState(args.seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0
    history = []
    optimizer_steps = 0
    for epoch in range(50):
        module.train()
        indices = balanced_epoch_indices(fit, rng)
        losses = []
        for rows in iter_balanced_batches(fit, indices, 4096, rng):
            batch = gather_batch(fit, rows, REQUIRED_KEYS, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(module, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("GACE training produced non-finite loss")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach().cpu().item()))
        module.eval()
        calibration_loss = mean_condition_loss(
            calibration, lambda payload: _loss(module, payload), device
        )
        history.append(
            {
                "epoch": epoch,
                "fit_loss": float(np.mean(losses)),
                "calibration_loss": calibration_loss,
                "optimizer_steps": optimizer_steps,
            }
        )
        if calibration_loss < best_loss - 1e-8:
            best_loss = calibration_loss
            best_epoch = epoch
            best_state = copy.deepcopy(module.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is None or optimizer_steps <= 0:
        raise RuntimeError("GACE training performed no valid optimizer step")
    module.load_state_dict(best_state, strict=True)
    trace_path = output_dir / "training_trace.csv"
    atomic_write_training_trace(trace_path, history)
    checkpoint_path = output_dir / "stage019_s4_gace_checkpoint.pth"
    atomic_torch_save(
        checkpoint_path,
        {
            formal.CHECKPOINT_STATE_DICT_KEY: module.state_dict(),
            "schema": "visfuse3d_stage019_s4_gace_checkpoint_v1",
            "protocol_profile": PROTOCOL_PROFILE,
            "method": "gace_lite",
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
        },
    )
    manifest = {
        "schema": "visfuse3d_stage019_s4_gace_training_manifest_v1",
        "status": "complete_training_awaiting_calibration_metric_grid",
        "protocol_profile": PROTOCOL_PROFILE,
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
        "temperature_scaling_baseline": {
            "status": "complete_training_awaiting_separate_calibration_metric_grid",
            "best_epoch": temperature_best_epoch,
            "best_calibration_loss": temperature_best_loss,
            "optimizer_step_count": temperature_steps,
            "checkpoint": {
                "path": str(temperature_checkpoint_path.resolve()),
                "sha256": sha256_file(temperature_checkpoint_path),
                "state_dict_sha256": state_dict_sha256(
                    temperature_module.state_dict()
                ),
            },
            "training_trace": {
                "path": str(temperature_trace_path.resolve()),
                "sha256": sha256_file(temperature_trace_path),
            },
            "promotion_role": "required_baseline_only_cannot_replace_gace_candidate",
        },
        "g0_gate": {"path": str(args.g0_gate.resolve()), "sha256": sha256_file(args.g0_gate)},
        "claim_boundary": "training_completion_only_not_calibration_or_validation_success",
    }
    atomic_write_json(output_dir / "stage019_s4_gace_training_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
