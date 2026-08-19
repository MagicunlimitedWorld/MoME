"""Shared, source-locked training utilities for the two S4 small heads."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import tempfile
import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

try:
    from .common import atomic_write_json
    from .stage019_s4_contracts import (
        ACTIONABLE_CONDITIONS,
        CONFIRMATION_SEED,
        MAX_TRAINABLE_PARAMETERS,
        PRIMARY_SEED,
        PROTOCOL_PROFILE,
        sha256_file,
    )
except ImportError:
    from common import atomic_write_json
    from stage019_s4_contracts import (
        ACTIONABLE_CONDITIONS,
        CONFIRMATION_SEED,
        MAX_TRAINABLE_PARAMETERS,
        PRIMARY_SEED,
        PROTOCOL_PROFILE,
        sha256_file,
    )


def formal_module_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "projects"
        / "mmdet3d_plugin"
        / "models"
        / "utils"
        / "object_set_attribute_fusion.py"
    )


def load_formal_fusion_module():
    path = formal_module_path()
    spec = importlib.util.spec_from_file_location("stage019_s4_training_formal_fusion", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load formal S4 fusion module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    if int(seed) not in (PRIMARY_SEED, CONFIRMATION_SEED):
        raise ValueError("S4 training seed must be primary 20260710 or confirmation 20260711")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_condition_paths(values: Iterable[str]) -> dict[str, list[Path]]:
    output = {condition: [] for condition in ACTIONABLE_CONDITIONS}
    for value in values:
        if "=" not in value:
            raise ValueError("cache argument must use CONDITION=PATH")
        condition, raw_path = value.split("=", 1)
        if condition not in output or not raw_path:
            raise ValueError(f"unknown/empty S4 cache condition: {condition}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        output[condition].append(path)
    missing = [condition for condition, paths in output.items() if not paths]
    if missing:
        raise ValueError(f"S4 caches lack conditions: {missing}")
    return output


def _token_set_sha256(tokens: Iterable[str]) -> str:
    values = sorted(set(str(value) for value in tokens))
    payload = "".join(f"{value}\n" for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def resolve_training_cache_authority(
    fit_values: Iterable[str],
    calibration_values: Iterable[str],
    inputs_manifest_path: Path,
) -> tuple[
    dict[str, list[Path]],
    dict[str, list[Path]],
    dict[str, Path],
]:
    """Resolve only hash-locked extract manifests into diagnostic cache paths."""

    inputs = json.loads(inputs_manifest_path.read_text(encoding="utf-8"))
    if (
        inputs.get("schema") != "visfuse3d_stage019_s4_inputs_manifest_v1"
        or inputs.get("status") != "passed"
        or inputs.get("protocol_profile") != PROTOCOL_PROFILE
    ):
        raise ValueError("S4 training inputs manifest is incomplete")
    expected_fit = inputs["partitions"]["fit80"]
    expected_calibration = inputs["partitions"]["calibration20"]

    def resolve(
        values: Iterable[str], allowed_phases: set[str], expected_partition: Mapping[str, Any]
    ) -> tuple[dict[str, list[Path]], dict[str, Path], dict[str, set[str]], dict[str, set[str]]]:
        expected_scenes = set(str(value) for value in expected_partition["scene_tokens"])
        expected_frame_count = int(expected_partition["frame_count"])
        expected_frame_sha256 = str(
            expected_partition["frame_token_set_sha256"]
        ).upper()
        if expected_frame_count <= 0 or len(expected_frame_sha256) != 64:
            raise ValueError("S4 input partition frame identity is incomplete")
        manifest_paths = parse_condition_paths(values)
        caches = {condition: [] for condition in ACTIONABLE_CONDITIONS}
        authorities = {}
        scenes_by_condition = {condition: set() for condition in ACTIONABLE_CONDITIONS}
        frames_by_condition = {condition: set() for condition in ACTIONABLE_CONDITIONS}
        for condition, paths in manifest_paths.items():
            for index, path in enumerate(paths):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if (
                    manifest.get("schema")
                    != "visfuse3d_stage019_s4_extract_worker_manifest_v1"
                    or manifest.get("status") != "complete"
                    or manifest.get("condition") != condition
                    or manifest.get("phase") not in allowed_phases
                    or manifest.get("input_identity", {}).get("protocol_profile")
                    != PROTOCOL_PROFILE
                    or int(manifest.get("engineering_failure_count", -1)) != 0
                    or not bool(manifest.get("training_cache_eligible"))
                ):
                    raise ValueError(f"invalid S4 training cache authority: {path}")
                flat_scene_tokens = [str(value) for value in manifest["scene_tokens"]]
                flat_frame_tokens = [str(value) for value in manifest["frame_tokens"]]
                scene_records = manifest.get("scenes", [])
                record_scene_tokens = [
                    str(scene.get("scene_token")) for scene in scene_records
                ]
                record_frame_tokens = [
                    str(token)
                    for scene in scene_records
                    for token in scene.get("frame_tokens", [])
                ]
                if (
                    flat_scene_tokens != record_scene_tokens
                    or flat_frame_tokens != record_frame_tokens
                    or len(flat_scene_tokens) != len(set(flat_scene_tokens))
                    or len(flat_frame_tokens) != len(set(flat_frame_tokens))
                    or int(manifest.get("scene_count", -1)) != len(flat_scene_tokens)
                    or int(manifest.get("frame_count", -1)) != len(flat_frame_tokens)
                    or int(manifest.get("unique_token_count", -1))
                    != len(flat_frame_tokens)
                ):
                    raise ValueError("S4 training worker manifest coverage is internally inconsistent")
                current_scenes = set(flat_scene_tokens)
                current_frames = set(flat_frame_tokens)
                if scenes_by_condition[condition] & current_scenes:
                    raise ValueError("S4 training worker manifests overlap scenes")
                if frames_by_condition[condition] & current_frames:
                    raise ValueError("S4 training worker manifests overlap frames")
                scenes_by_condition[condition].update(current_scenes)
                frames_by_condition[condition].update(current_frames)
                for scene in manifest["scenes"]:
                    diagnostic = Path(scene["diagnostic_path"])
                    if (
                        not diagnostic.is_file()
                        or sha256_file(diagnostic) != scene["diagnostic_sha256"]
                    ):
                        raise ValueError("S4 training diagnostic hash changed")
                    caches[condition].append(diagnostic)
                authorities[f"{condition}_{index}_{manifest['phase']}"] = path
        for condition in ACTIONABLE_CONDITIONS:
            if scenes_by_condition[condition] != expected_scenes:
                raise ValueError(f"S4 {condition} scene coverage differs from locked partition")
        reference_frames = frames_by_condition[ACTIONABLE_CONDITIONS[0]]
        if any(frames_by_condition[condition] != reference_frames for condition in ACTIONABLE_CONDITIONS):
            raise ValueError("S4 conditions do not cover the same frame-token set")
        if (
            len(reference_frames) != expected_frame_count
            or _token_set_sha256(reference_frames) != expected_frame_sha256
        ):
            raise ValueError("S4 frame coverage differs from the locked input partition")
        return caches, authorities, scenes_by_condition, frames_by_condition

    fit, fit_authority, fit_scenes, fit_frames = resolve(
        fit_values, {"g0", "fit"}, expected_fit
    )
    calibration, calibration_authority, calibration_scenes, calibration_frames = resolve(
        calibration_values, {"calibration"}, expected_calibration
    )
    if any(
        fit_scenes[condition] & calibration_scenes[condition]
        or fit_frames[condition] & calibration_frames[condition]
        for condition in ACTIONABLE_CONDITIONS
    ):
        raise ValueError("S4 fit/calibration cache authorities overlap")
    authorities = {"inputs_manifest": inputs_manifest_path.resolve()}
    authorities.update({f"fit_{key}": value for key, value in fit_authority.items()})
    authorities.update(
        {f"calibration_{key}": value for key, value in calibration_authority.items()}
    )
    return fit, calibration, authorities


def load_cache_arrays(
    paths: Mapping[str, list[Path]], required: tuple[str, ...]
) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for condition, condition_paths in paths.items():
        collected = {key: [] for key in required}
        for path in condition_paths:
            with np.load(path, allow_pickle=False) as payload:
                missing = set(required) - set(payload.files)
                if missing:
                    raise ValueError(f"S4 cache {path} lacks {sorted(missing)}")
                if payload["feature_matrix"].ndim != 2 or payload["feature_matrix"].shape[1] != 41:
                    raise ValueError(f"S4 cache {path} feature width drifted")
                for key in required:
                    collected[key].append(np.asarray(payload[key]))
        arrays = {
            key: np.concatenate(values, axis=0) for key, values in collected.items()
        }
        count = arrays["feature_matrix"].shape[0]
        if count <= 0:
            raise ValueError(f"S4 cache condition {condition} has zero objects")
        for key, value in arrays.items():
            if value.shape[0] != count:
                raise ValueError(f"S4 cache {condition}/{key} row count drifted")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"S4 cache {condition}/{key} contains non-finite values")
        output[condition] = arrays
    return output


def balanced_epoch_indices(
    arrays: Mapping[str, Mapping[str, np.ndarray]], rng: np.random.RandomState
) -> dict[str, np.ndarray]:
    minimum = min(value["feature_matrix"].shape[0] for value in arrays.values())
    if minimum <= 0:
        raise ValueError("condition-balanced S4 epoch has zero rows")
    return {
        condition: rng.permutation(value["feature_matrix"].shape[0])[:minimum]
        for condition, value in arrays.items()
    }


def iter_balanced_batches(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    indices: Mapping[str, np.ndarray],
    batch_size: int,
    rng: np.random.RandomState,
):
    combined = np.concatenate(
        [
            np.stack(
                (
                    np.full(len(indices[condition]), condition_index, dtype=np.int64),
                    indices[condition],
                ),
                axis=1,
            )
            for condition_index, condition in enumerate(ACTIONABLE_CONDITIONS)
        ],
        axis=0,
    )
    combined = combined[rng.permutation(len(combined))]
    for offset in range(0, len(combined), int(batch_size)):
        rows = combined[offset : offset + int(batch_size)]
        yield [
            (
                ACTIONABLE_CONDITIONS[int(condition_index)],
                int(row_index),
            )
            for condition_index, row_index in rows
        ]


def gather_batch(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    rows: list[tuple[str, int]],
    keys: tuple[str, ...],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output = {}
    for key in keys:
        values = np.stack([arrays[condition][key][index] for condition, index in rows])
        output[key] = torch.as_tensor(values, device=device)
    return output


def mean_condition_loss(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    loss_function,
    device: torch.device,
) -> float:
    values = []
    with torch.no_grad():
        for condition in ACTIONABLE_CONDITIONS:
            payload = {
                key: torch.as_tensor(value, device=device)
                for key, value in arrays[condition].items()
            }
            values.append(float(loss_function(payload).detach().cpu().item()))
    return float(np.mean(values))


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest().upper()


def validate_trainable_parameter_count(count: int) -> int:
    value = int(count)
    if value <= 0:
        raise ValueError("S4 trainable parameter count must be positive")
    if value > MAX_TRAINABLE_PARAMETERS:
        raise RuntimeError("S4 trainable parameter count exceeds 50k")
    return value


def write_optimizer_provenance(
    output_dir: Path,
    *,
    method: str,
    seed: int,
    fit_paths: Mapping[str, list[Path]],
    calibration_paths: Mapping[str, list[Path]],
    model,
    optimizer: torch.optim.Optimizer,
    loss_contract: Mapping[str, Any],
    frozen_checkpoint: Path,
    trainable_names: list[str],
    batch_size: int,
    trainer_path: Path,
    config_path: Path,
    authority_manifests: Mapping[str, Path],
    manifest_name: str = "optimizer_provenance_manifest.json",
) -> dict[str, Any]:
    trainable_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name in set(trainable_names) and parameter.requires_grad
    )
    total_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_count != total_trainable or total_trainable <= 0:
        raise RuntimeError("S4 trainable parameter-name closure drifted")
    validate_trainable_parameter_count(total_trainable)
    manifest = {
        "schema": "visfuse3d_stage019_s4_optimizer_provenance_v1",
        "status": "locked_before_first_optimizer_step",
        "protocol_profile": PROTOCOL_PROFILE,
        "method": method,
        "seed": int(seed),
        "rng": {
            "python": int(seed),
            "numpy": int(seed),
            "torch": int(seed),
            "deterministic_algorithms": True,
        },
        "data": {
            "fit": {
                condition: [
                    {"path": str(path), "sha256": sha256_file(path)} for path in paths
                ]
                for condition, paths in fit_paths.items()
            },
            "calibration": {
                condition: [
                    {"path": str(path), "sha256": sha256_file(path)} for path in paths
                ]
                for condition, paths in calibration_paths.items()
            },
        },
        "source": {
            "formal_fusion_module": {
                "path": str(formal_module_path()),
                "sha256": sha256_file(formal_module_path()),
            },
            "training_common": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "trainer": {
                "path": str(trainer_path.resolve()),
                "sha256": sha256_file(trainer_path),
            },
            "contracts": {
                "path": str(Path(__file__).with_name("stage019_s4_contracts.py").resolve()),
                "sha256": sha256_file(
                    Path(__file__).with_name("stage019_s4_contracts.py")
                ),
            },
            "config": {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
            },
        },
        "authority_manifests": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in authority_manifests.items()
        },
        "frozen_mome_checkpoint": {
            "path": str(frozen_checkpoint.resolve()),
            "sha256": sha256_file(frozen_checkpoint),
        },
        "model_initial_state_sha256": state_dict_sha256(model.state_dict()),
        "optimizer": {
            "class": type(optimizer).__name__,
            "defaults": {
                key: value
                for key, value in optimizer.defaults.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
            "batch_size": int(batch_size),
        },
        "loss": dict(loss_contract),
        "hardware": {
            "device": str(next(model.parameters()).device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": total_trainable,
    }
    atomic_write_json(output_dir / manifest_name, manifest)
    return manifest


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(handle)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_training_trace(path: Path, history: list[Mapping[str, Any]]) -> None:
    if not history:
        raise ValueError("training trace cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_fusion_checkpoint(path: Path, model, formal) -> dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu")
    if formal.CHECKPOINT_STATE_DICT_KEY not in payload:
        raise ValueError("S4 checkpoint lacks formal top-level state_dict key")
    model.load_state_dict(payload[formal.CHECKPOINT_STATE_DICT_KEY], strict=True)
    return payload
