from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'tools' / 'qta' / 'stage019_s2_training_conditions.py'
SPEC = importlib.util.spec_from_file_location('s2_training_conditions', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_training_condition_assignments_are_deterministic_and_bounded() -> None:
    token = 'd02204fcc72841109ce526cbd70f3c82'
    assert module.training_object_failure_flag(token) == (
        module.training_object_failure_flag(token)
    )
    observed = [module.training_mud_mask_id(token, view) for view in range(6)]
    assert observed == [
        module.training_mud_mask_id(token, view) for view in range(6)
    ]
    assert all(1 <= value <= 16 for value in observed)
    assert len(module.CONDITIONS) == 7
    assert module.S3_ACTIONABLE_CONDITIONS == (
        'clean',
        'beam_reduction_4',
        'limited_fov_original_code_60',
        'lidar_object_failure',
        'camera_mud_mask',
    )
    assert module.S3_HARD_BYPASS_CONDITIONS == ('lidar_zero', 'camera_zero')
