from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
QTA_ROOT = ROOT / "tools" / "qta"
MODULE_PATH = QTA_ROOT / "aggregate_stage019_s2_calibration.py"
SPEC = importlib.util.spec_from_file_location(
    "stage019_s2_calibration_profiles", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
aggregate = importlib.util.module_from_spec(SPEC)
conditions_stub = ModuleType("stage019_s2_training_conditions")
conditions_stub.CONDITIONS = (
    "clean",
    "beam_reduction_4",
    "lidar_zero",
    "limited_fov_original_code_60",
    "lidar_object_failure",
    "camera_zero",
    "camera_mud_mask",
)
previous_conditions_module = sys.modules.get("stage019_s2_training_conditions")
sys.modules["stage019_s2_training_conditions"] = conditions_stub
sys.path.insert(0, str(QTA_ROOT))
try:
    SPEC.loader.exec_module(aggregate)
finally:
    sys.path.pop(0)
    if previous_conditions_module is None:
        sys.modules.pop("stage019_s2_training_conditions", None)
    else:
        sys.modules["stage019_s2_training_conditions"] = previous_conditions_module


def test_three_repeat_profile_requires_the_exact_locked_ids() -> None:
    repeat_ids = ["gpu1_repeat0", "gpu0_repeat0", "gpu0_repeat1"]
    aggregate.validate_repeat_identity(
        "three_repeat_v1", repeat_ids, "gpu0_repeat0"
    )
    for invalid in (
        ["gpu0_repeat0", "gpu0_repeat1"],
        ["gpu0_repeat0", "gpu0_repeat1", "gpu1_repeat1"],
        ["gpu0_repeat0", "gpu0_repeat1", "gpu0_repeat1"],
        ["gpu0_repeat0", "gpu0_repeat1", "gpu1_repeat0", "gpu1_repeat1"],
    ):
        with pytest.raises(ValueError):
            aggregate.validate_repeat_identity(
                "three_repeat_v1", invalid, "gpu0_repeat0"
            )


def test_profiles_lock_one_canonical_and_preserve_four_repeat_compatibility() -> None:
    aggregate.validate_repeat_identity(
        "four_repeat_v1",
        ["gpu0_repeat0", "gpu1_repeat0", "gpu0_repeat1", "gpu1_repeat1"],
        "gpu0_repeat0",
    )
    with pytest.raises(ValueError, match="canonical"):
        aggregate.validate_repeat_identity(
            "three_repeat_v1",
            ["gpu0_repeat0", "gpu0_repeat1", "gpu1_repeat0"],
            "gpu1_repeat0",
        )


def test_three_observations_use_max_minus_min_envelope() -> None:
    observations = np.asarray(
        [[1.0, -2.0], [4.0, -1.0], [2.5, -3.5]], dtype=np.float64
    )
    envelope = aggregate.observed_numerical_envelope(observations, axis=0)
    assert np.array_equal(envelope, np.asarray([3.0, 2.5]))
    assert aggregate.claim_boundary("three_repeat_v1").startswith(
        "observed_three_repeat_numerical_envelope"
    )
    assert aggregate.claim_boundary("four_repeat_v1").startswith(
        "observed_four_repeat_numerical_envelope"
    )
