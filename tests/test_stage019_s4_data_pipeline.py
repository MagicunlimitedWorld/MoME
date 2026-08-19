from __future__ import annotations

import json
import hashlib
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
QTA = ROOT / "tools" / "qta"
sys.path.insert(0, str(QTA))
try:
    import stage019_s4_contracts as contracts
    import stage019_s4_g0_gate as g0_gate
    import stage019_s4_extract_worker as extract_worker
    import stage019_s4_object_cache as cache
    import stage019_s4_oracles as oracles
    import stage019_s4_training_common as training
finally:
    sys.path.pop(0)


def _boxes(centers, yaw=0.0):
    values = []
    for x, y in centers:
        values.append([x, y, 10.0, 2.0, 4.0, 2.0, yaw, 0.0, 0.0])
    return np.asarray(values, dtype=np.float32).reshape(-1, 9)


def _object_set(centers, scores=None, labels=None):
    count = len(centers)
    return {
        "boxes": _boxes(centers),
        "scores": np.asarray(scores or [0.5] * count, dtype=np.float32),
        "labels": np.asarray(labels or [0] * count, dtype=np.int64),
    }


def _camera_contract():
    matrices = [np.eye(4, dtype=np.float32) for _ in range(6)]
    shapes = [(100, 100, 3) for _ in range(6)]
    return matrices, shapes


def test_split_fold_and_pilot_identities_are_exact() -> None:
    fit = [f"fit{i:02d}" for i in range(80)]
    calibration = [f"cal{i:02d}" for i in range(20)]
    result = contracts.validate_train_scene_split(fit, calibration, fit[:20])
    assert result["training20_is_fit_subset"]
    assert tuple(map(len, contracts.split_g0_folds(fit[:20]))) == (7, 7, 6)
    contracts.validate_pilot_identity(contracts.PILOT_SCENES, 401)
    changed = list(contracts.PILOT_SCENES)
    changed[-1] = "0" * 32
    with pytest.raises(ValueError, match="pilot"):
        contracts.validate_pilot_identity(changed, 401)
    with pytest.raises(ValueError, match="overlap"):
        contracts.validate_train_scene_split(fit, fit[:20], fit[:20])


def test_validation_beam4_rejects_clean_or_unlocked_annotation(
    tmp_path: Path,
) -> None:
    unlocked = tmp_path / "clean.pkl"
    unlocked.write_bytes(b"not the locked beam4 validation annotation")
    with pytest.raises(ValueError, match="pre-derived annotation"):
        extract_worker._validate_phase_annotation(
            Namespace(
                phase="smoke",
                condition="beam_reduction_4",
                ann_file=unlocked,
            )
        )
    extract_worker._validate_phase_annotation(
        Namespace(phase="smoke", condition="clean", ann_file=unlocked)
    )


def test_extract_worker_accepts_collated_tensor_and_basepoints_wrapper() -> None:
    points = torch.tensor([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.7]])
    expected = points[:, :3].numpy()
    np.testing.assert_array_equal(extract_worker._points_xyz(points), expected)
    np.testing.assert_array_equal(
        extract_worker._points_xyz(Namespace(tensor=points)), expected
    )
    with pytest.raises(ValueError, match="shape"):
        extract_worker._points_xyz(torch.ones(2, 2))


def test_one_to_one_quality_does_not_label_duplicate_or_far_anchor() -> None:
    boxes = _boxes([(0.0, 0.0), (0.1, 0.0), (70.0, 0.0)])
    labels = np.zeros(3, dtype=np.int64)
    quality, target = cache.one_to_one_soft_quality(
        boxes, labels, _boxes([(0.0, 0.0)]), np.zeros(1, dtype=np.int64)
    )
    assert np.count_nonzero(quality) == 1
    assert target.tolist().count(0) == 1
    assert quality[2] == 0.0 and target[2] == -1


def test_cache_uses_formal_41d_builder_and_separates_gt() -> None:
    original = _object_set([(0.0, 0.0), (10.0, 0.0)], [0.6, 0.4])
    experts = {
        "fused": _object_set([(0.2, 0.0), (10.1, 0.0)], [0.7, 0.5]),
        "lidar": _object_set([(0.3, 0.0)], [0.8]),
        "camera": _object_set([(10.2, 0.0)], [0.9]),
    }
    matrices, shapes = _camera_contract()
    record = cache.build_leakage_separated_cache_record(
        original,
        experts,
        np.asarray([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0]], dtype=np.float32),
        matrices,
        shapes,
        _boxes([(0.0, 0.0), (10.0, 0.0)]),
        np.zeros(2, dtype=np.int64),
    )
    cache.assert_no_leakage(record)
    inference = record["inference_features"]
    assert inference["feature_matrix"].shape == (2, 41)
    assert inference["expert_source_indices"].shape == (2, 3)
    formal = cache._formal_fusion_module()
    anchor = {
        "boxes_3d": torch.from_numpy(original["boxes"]),
        "scores_3d": torch.from_numpy(original["scores"]),
        "labels_3d": torch.from_numpy(original["labels"]),
    }
    expert_tensors = {
        role: {
            "boxes_3d": torch.from_numpy(value["boxes"]),
            "scores_3d": torch.from_numpy(value["scores"]),
            "labels_3d": torch.from_numpy(value["labels"]),
        }
        for role, value in experts.items()
    }
    evidence = {
        key: torch.from_numpy(np.asarray(value).astype(np.float32))
        for key, value in inference["object_evidence"].items()
    }
    expected = formal.build_feature_matrix(anchor, expert_tensors, evidence)
    assert tuple(formal.FEATURE_NAMES) == tuple(cache.FEATURE_NAMES)
    assert torch.equal(
        expected["feature_matrix"], torch.from_numpy(inference["feature_matrix"])
    )
    assert set(record) == {"schema", "inference_features", "offline_targets"}


def test_g0_score_keeps_no_match_and_attribute_residuals_are_bounded() -> None:
    original = _object_set([(0.0, 0.0)], [0.2])
    experts = {
        "fused": _object_set([(1.5, 0.0)], [0.9]),
        "lidar": _object_set([(50.0, 0.0)], [0.9]),
        "camera": _object_set([(60.0, 0.0)], [0.9]),
    }
    experts["fused"]["boxes"][0, 3:6] = original["boxes"][0, 3:6] * np.exp(0.8)
    experts["fused"]["boxes"][0, 6] = np.pi
    experts["fused"]["boxes"][0, 7:9] = 10.0
    matrices, shapes = _camera_contract()
    record = cache.build_leakage_separated_cache_record(
        original,
        experts,
        np.asarray([[0.0, 0.0, 10.0]], dtype=np.float32),
        matrices,
        shapes,
        experts["fused"]["boxes"].copy(),
        np.zeros(1, dtype=np.int64),
    )
    outputs, _ = oracles.build_g0_oracles(original, experts, record)
    attribute = outputs["g0_attribute"]["boxes"][0]
    assert np.linalg.norm(attribute[:3] - original["boxes"][0, :3]) <= 1.000001
    assert np.max(
        np.abs(np.log(attribute[3:6]) - np.log(original["boxes"][0, 3:6]))
    ) <= 0.200001
    assert abs(cache.wrap_angle(np.asarray([attribute[6] - original["boxes"][0, 6]]))[0]) <= np.pi / 9 + 1e-6
    assert np.linalg.norm(attribute[7:9] - original["boxes"][0, 7:9]) <= 2.000001

    no_match_experts = {
        role: _object_set([(50.0 + index, 0.0)], [0.9])
        for index, role in enumerate(contracts.EXPERT_ROLES)
    }
    no_match_record = cache.build_leakage_separated_cache_record(
        original,
        no_match_experts,
        np.asarray([[0.0, 0.0, 10.0]], dtype=np.float32),
        matrices,
        shapes,
        original["boxes"].copy(),
        np.zeros(1, dtype=np.int64),
    )
    no_match_output, _ = oracles.build_g0_oracles(
        original, no_match_experts, no_match_record
    )
    assert np.array_equal(no_match_output["g0_score"]["scores"], original["scores"])


def test_parameter_boundary_and_checkpoint_strict_load() -> None:
    assert training.validate_trainable_parameter_count(50_000) == 50_000
    with pytest.raises(RuntimeError, match="50k"):
        training.validate_trainable_parameter_count(50_001)
    formal = training.load_formal_fusion_module()
    module = formal.ObjectSetAttributeFusion()
    assert module.trainable_parameter_count() <= 50_000
    clone = formal.ObjectSetAttributeFusion()
    clone.load_state_dict(module.state_dict(), strict=True)
    assert formal.CHECKPOINT_STATE_DICT_KEY == "object_set_attribute_fusion_state_dict"


def test_training_authority_rejects_partial_frames_even_when_conditions_agree(
    tmp_path: Path,
) -> None:
    fit_scenes = [f"fit{i:02d}" for i in range(80)]
    calibration_scenes = [f"cal{i:02d}" for i in range(20)]
    expected_fit_frames = [
        f"{scene}_frame{frame_index}"
        for scene in fit_scenes
        for frame_index in range(2)
    ]

    def token_set_sha256(values):
        serialized = "".join(f"{value}\n" for value in sorted(set(values)))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest().upper()

    inputs = {
        "schema": "visfuse3d_stage019_s4_inputs_manifest_v1",
        "status": "passed",
        "protocol_profile": contracts.PROTOCOL_PROFILE,
        "partitions": {
            "fit80": {
                "scene_tokens": fit_scenes,
                "frame_count": len(expected_fit_frames),
                "frame_token_set_sha256": token_set_sha256(expected_fit_frames),
            },
            "calibration20": {
                "scene_tokens": calibration_scenes,
                "frame_count": 20,
                "frame_token_set_sha256": token_set_sha256(
                    [f"{scene}_frame0" for scene in calibration_scenes]
                ),
            },
        },
    }
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
    partial_frames = [f"{scene}_frame0" for scene in fit_scenes]
    fit_values = []
    for condition in contracts.ACTIONABLE_CONDITIONS:
        diagnostic = tmp_path / f"{condition}.npz"
        diagnostic.write_bytes(b"locked diagnostic")
        scenes = [
            {
                "scene_token": scene,
                "frame_tokens": [frame],
                "diagnostic_path": str(diagnostic),
                "diagnostic_sha256": contracts.sha256_file(diagnostic),
            }
            for scene, frame in zip(fit_scenes, partial_frames)
        ]
        manifest = {
            "schema": "visfuse3d_stage019_s4_extract_worker_manifest_v1",
            "status": "complete",
            "condition": condition,
            "phase": "fit",
            "input_identity": {"protocol_profile": contracts.PROTOCOL_PROFILE},
            "engineering_failure_count": 0,
            "training_cache_eligible": True,
            "scene_tokens": fit_scenes,
            "frame_tokens": partial_frames,
            "scene_count": len(fit_scenes),
            "frame_count": len(partial_frames),
            "unique_token_count": len(partial_frames),
            "scenes": scenes,
        }
        manifest_path = tmp_path / f"{condition}.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        fit_values.append(f"{condition}={manifest_path}")
    with pytest.raises(ValueError, match="locked input partition"):
        training.resolve_training_cache_authority(fit_values, [], inputs_path)


def test_g0_gate_writes_outer_manifest_schema(tmp_path: Path, monkeypatch) -> None:
    scenes = [f"scene{i:02d}" for i in range(20)]
    roles = g0_gate.ROLES
    manifests = []
    for condition in contracts.ACTIONABLE_CONDITIONS:
        metrics = {
            "original_mome": {"mAP": 0.5, "NDS": 0.5},
            "g0_score": {"mAP": 0.506, "NDS": 0.501},
            "g0_attribute": {"mAP": 0.512, "NDS": 0.504},
            "g0_union": {"mAP": 0.514, "NDS": 0.505},
        }
        folds = [
            {"scene_count": size, "scene_tokens": list(fold), "metrics": metrics}
            for size, fold in zip(contracts.G0_FOLD_SIZES, contracts.split_g0_folds(scenes))
        ]
        payload = {
            "schema": "visfuse3d_stage019_s4_condition_merge_manifest_v1",
            "status": "complete_g0_train_subset_evaluation",
            "protocol_profile": contracts.PROTOCOL_PROFILE,
            "phase": "g0",
            "source_split": "train",
            "condition": condition,
            "scene_count": 20,
            "frame_count": 803,
            "unique_token_count": 803,
            "engineering_failure_count": 0,
            "scene_tokens": scenes,
            "frame_token_order_sha256": "A" * 64,
            "metrics": metrics,
            "fold_metrics": folds,
            "oracle_constraints": oracles.ORACLE_CONSTRAINTS,
            "union_added_true_positive_count": 0,
            "locked_worker_identity": {
                "config_sha256": "B" * 64,
                "checkpoint_sha256": "C" * 64,
                "scene_map_sha256": "D" * 64,
                "scene_list_sha256": contracts.TRAINING20_INPUTS_MANIFEST_SHA256,
                "source_bundle_sha256": "E" * 64,
            },
        }
        path = tmp_path / f"{condition}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        manifests.append(path)
    artifact = tmp_path / "artifact"
    output = artifact / "gate"
    argv = ["gate"]
    for path in manifests:
        argv += ["--condition-manifest", str(path)]
    argv += ["--output-dir", str(output), "--artifact-root", str(artifact)]
    monkeypatch.setattr(sys, "argv", argv)
    assert g0_gate.main() == 0
    result = json.loads((output / "stage019_s4_g0_gate_manifest.json").read_text())
    assert result["schema"] == "visfuse3d_stage019_s4_g0_gate_manifest_v1"
    assert result["training_authorized"] and result["g1_authorized"]

    first_payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    union_metrics = first_payload["metrics"].pop("g0_union")
    manifests[0].write_text(
        json.dumps(first_payload, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid G0 condition manifest"):
        g0_gate._load_conditions(manifests)

    first_payload["metrics"]["g0_union"] = union_metrics
    first_payload["metrics"]["unexpected_role"] = union_metrics
    manifests[0].write_text(
        json.dumps(first_payload, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid G0 condition manifest"):
        g0_gate._load_conditions(manifests)


def test_validation_gate_requires_both_changed_and_accepted_and_real_zero_sha(
    tmp_path: Path,
) -> None:
    prediction = tmp_path / "results_nusc.json"
    prediction.write_text('{"results": {}}\n', encoding="utf-8")
    digest = contracts.sha256_file(prediction)
    zero = {
        condition: {
            "same_run_original_reference": True,
            "original_prediction": {"path": str(prediction), "sha256": digest},
            "final_prediction": {"path": str(prediction), "sha256": digest},
        }
        for condition in contracts.HARD_BYPASS_CONDITIONS
    }
    deltas = {
        "clean": {"mAP": 0.0, "NDS": 0.0},
        **{
            condition: {"mAP": 0.01, "NDS": 0.01}
            for condition in contracts.FAULT_CONDITIONS
        },
    }
    passed = contracts.evaluate_validation_gate(
        deltas, 1, 1, zero, "gace_lite"
    )
    assert passed["passed"]
    rejected = contracts.evaluate_validation_gate(
        deltas, 1, 0, zero, "gace_lite"
    )
    assert not rejected["passed"]
    prediction.write_text("changed\n", encoding="utf-8")
    tampered = contracts.evaluate_validation_gate(
        deltas, 1, 1, zero, "gace_lite"
    )
    assert not tampered["passed"]


def test_provenance_call_precedes_every_optimizer_step_in_trainers() -> None:
    for name in ("stage019_s4_train_gace.py", "stage019_s4_train_cp_afr.py"):
        source = (QTA / name).read_text(encoding="utf-8")
        first_provenance = source.index("write_optimizer_provenance(")
        first_step = source.index("optimizer.step()")
        assert first_provenance < first_step
