from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
QTA_ROOT = ROOT / "tools" / "qta"
MODULE_PATH = QTA_ROOT / "evaluate_stage019_s2b_fullval_gate.py"
CONDITIONS = (
    "clean",
    "beam_reduction_4",
    "lidar_zero",
    "limited_fov_original_code_60",
    "lidar_object_failure",
    "camera_zero",
    "camera_mud_mask",
)
ACTIONABLE = (
    "clean",
    "beam_reduction_4",
    "limited_fov_original_code_60",
    "lidar_object_failure",
    "camera_mud_mask",
)
BYPASS = ("lidar_zero", "camera_zero")


def _load_gate() -> ModuleType:
    common = ModuleType("common")
    common.atomic_write_json = lambda path, payload: None
    extract = ModuleType("extract_route_loss_cache")
    extract.is_relative_to = lambda path, root: True
    extract.sha256_file = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
    full = ModuleType("full_query_oracle_worker")
    full.CONDITIONS = CONDITIONS
    training = ModuleType("stage019_s2_training_conditions")
    training.S3_ACTIONABLE_CONDITIONS = ACTIONABLE
    training.S3_HARD_BYPASS_CONDITIONS = BYPASS
    stubs = {
        "common": common,
        "extract_route_loss_cache": extract,
        "full_query_oracle_worker": full,
        "stage019_s2_training_conditions": training,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("stage019_s3_gate", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_s3_hard_bypass_requires_same_prediction_path_and_sha(tmp_path: Path) -> None:
    gate = _load_gate()
    prediction = tmp_path / "results_nusc.json"
    prediction.write_text('{"results": {}}\n', encoding="utf-8")
    digest = hashlib.sha256(prediction.read_bytes()).hexdigest().upper()
    paths = []
    for condition in BYPASS:
        payload = {
            "status": "complete_hard_bypass_identity",
            "protocol_profile": gate.S3_PROTOCOL_PROFILE,
            "mode": "s3b_hard_bypass",
            "applicability": "hard_bypass_not_applicable",
            "condition": condition,
            "scene_count": 150,
            "frame_count": 6019,
            "unique_token_count": 6019,
            "engineering_failure_count": 0,
            "prediction_sha256_identical": True,
            "base_prediction": {"path": str(prediction), "sha256": digest},
            "final_prediction": {"path": str(prediction), "sha256": digest},
        }
        path = tmp_path / f"{condition}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    assert set(gate._load_hard_bypass(paths)) == set(BYPASS)

    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["final_prediction"]["sha256"] = "0" * 64
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hard-bypass"):
        gate._load_hard_bypass(paths)


def test_s3_pilot_loader_accepts_only_the_five_actionable_conditions(
    tmp_path: Path,
) -> None:
    gate = _load_gate()
    paths = []
    for condition in ACTIONABLE:
        path = tmp_path / f"{condition}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "complete_diagnostic_merge",
                    "mode": "s2b",
                    "condition": condition,
                    "engineering_failure_count": 0,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    assert set(
        gate._load_by_condition(paths, "s2b", False, ACTIONABLE)
    ) == set(ACTIONABLE)
    with pytest.raises(ValueError, match="every expected condition"):
        gate._load_by_condition(paths[:-1], "s2b", False, ACTIONABLE)
