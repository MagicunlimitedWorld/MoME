"""Shared contracts for MoME-QTA-AQR offline artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


SEED = 20260710
BASELINE_ID = "mome"
BASELINE_NAME = "MoME"
CANDIDATE_ID = "mome_qta_aqr_camera_lidar"
CANDIDATE_NAME = "MoME-QTA-AQR (Camera+LiDAR)"
HEALTH_CONTROL_ID = "mome_local_health_aqr_camera_lidar"
HEALTH_CONTROL_NAME = "MoME Local-Health AQR (Camera+LiDAR)"
RULE_CONTROL_ID = "mome_local_evidence_rule_camera_lidar"
RULE_CONTROL_NAME = "MoME Local-Evidence Rule (Camera+LiDAR)"
ROUTE_NAMES = ("Fused", "LiDAR", "Camera")
CONDITIONS = (
    "clean",
    "lidar_zero",
    "camera_zero",
    "lidar_sector_missing",
    "lidar_object_points_missing",
    "camera_local_occlusion",
)
LOCAL_CONDITIONS = CONDITIONS[3:]
FORBIDDEN_EVALUATION_MARKERS = (
    "nuscenes-r",
    "mud_mask",
    "limited_fov",
    "object_failure",
)


def set_reproducible_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_digest(value: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{value}|{seed}".encode("utf-8")).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def require_arrays(
    payload: dict[str, np.ndarray], required: Iterable[str]
) -> None:
    missing = sorted(set(required) - set(payload))
    if missing:
        raise ValueError(f"QTA cache is missing arrays: {', '.join(missing)}")


def _string_values(values: np.ndarray) -> list[str]:
    normalized: list[str] = []
    for value in values.reshape(-1).tolist():
        if isinstance(value, bytes):
            normalized.append(value.decode("utf-8"))
        else:
            normalized.append(str(value))
    return normalized


def validate_train_only_cache(payload: dict[str, np.ndarray]) -> None:
    """Fail closed on validation/corruption-benchmark label leakage."""

    require_arrays(payload, ("scene_token", "source_split", "condition"))
    source_splits = set(_string_values(payload["source_split"]))
    if source_splits != {"train"}:
        raise ValueError(f"QTA labels must be train-only; found {sorted(source_splits)}")
    conditions = set(_string_values(payload["condition"]))
    unexpected = conditions - set(CONDITIONS)
    if unexpected:
        raise ValueError(f"Unknown QTA conditions: {sorted(unexpected)}")
    if "source_path" in payload:
        joined = "|".join(_string_values(payload["source_path"])).lower().replace("\\", "/")
        hits = [item for item in FORBIDDEN_EVALUATION_MARKERS if item in joined]
        if hits:
            raise ValueError(
                "QTA cache references forbidden formal evaluation inputs: "
                + ", ".join(hits)
            )


def identity_payload(canonical_model_id: str) -> dict[str, str]:
    names = {
        BASELINE_ID: BASELINE_NAME,
        CANDIDATE_ID: CANDIDATE_NAME,
        HEALTH_CONTROL_ID: HEALTH_CONTROL_NAME,
        RULE_CONTROL_ID: RULE_CONTROL_NAME,
    }
    if canonical_model_id not in names:
        raise ValueError(f"Unknown QTA canonical_model_id: {canonical_model_id}")
    return {
        "canonical_model_id": canonical_model_id,
        "display_name": names[canonical_model_id],
        "frozen_baseline_canonical_model_id": BASELINE_ID,
        "frozen_baseline_display_name": BASELINE_NAME,
    }


def atomic_replace(source: str | Path, destination: str | Path) -> None:
    """Replace a file with bounded retries for transient Windows read locks."""

    attempts = 20
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.01 * (2**attempt), 0.25))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        atomic_replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()
