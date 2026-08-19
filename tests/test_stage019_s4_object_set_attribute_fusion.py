from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from mmcv import Config

from projects.mmdet3d_plugin.models.utils.object_set_attribute_fusion import (
    CHECKPOINT_STATE_DICT_KEY,
    FEATURE_NAMES,
    FEATURE_VERSION,
    ObjectSetAttributeFusion,
    build_feature_matrix,
    deterministic_class_hungarian_matches,
)


ROOT = Path(__file__).resolve().parents[1]


def _prediction(boxes, scores, labels):
    return {
        "boxes_3d": torch.tensor(boxes, dtype=torch.float32),
        "scores_3d": torch.tensor(scores, dtype=torch.float32),
        "labels_3d": torch.tensor(labels, dtype=torch.long),
    }


def _frame():
    anchor = _prediction(
        [
            [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, math.pi - 0.05, 0.0, 0.0],
            [10.0, 0.0, 0.0, 1.5, 3.0, 1.4, 0.2, 1.0, 0.0],
        ],
        [0.8, 0.6],
        [0, 1],
    )
    experts = {
        "fused": _prediction(
            [
                [0.4, 0.0, 0.1, 2.2, 3.7, 1.6, -math.pi + 0.05, 0.5, 0.0],
                [10.3, 0.1, 0.0, 1.7, 2.8, 1.5, 0.3, 1.3, 0.2],
            ],
            [0.9, 0.7],
            [0, 1],
        ),
        "lidar": _prediction(
            [
                [-0.2, 0.1, -0.1, 1.8, 4.2, 1.4, math.pi - 0.1, -0.2, 0.0],
                [30.0, 0.0, 0.0, 1.5, 3.0, 1.4, 0.2, 1.0, 0.0],
            ],
            [0.75, 0.95],
            [0, 1],
        ),
        "camera": _prediction(
            [[0.1, -0.2, 0.0, 2.1, 4.1, 1.5, -math.pi + 0.02, 0.1, 0.1]],
            [0.85],
            [0],
        ),
    }
    evidence = {
        "object_distance": torch.tensor([0.0, 10.0]),
        "box_point_count": torch.tensor([5.0, 2.0]),
        "neighborhood_point_count": torch.tensor([12.0, 4.0]),
        "visible_camera_views": torch.tensor([6.0, 3.0]),
    }
    return anchor, experts, evidence


def test_hungarian_uses_distance_then_stable_source_index_and_two_metre_cap():
    anchors = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    sources = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [7.0001, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    observed = deterministic_class_hungarian_matches(
        anchors,
        torch.tensor([0, 0, 1]),
        sources,
        torch.tensor([0, 0, 1]),
        route_index=0,
    )
    assert torch.equal(observed, torch.tensor([0, 1, -1]))


def test_feature_builder_has_exact_41d_schema_and_zero_missing_blocks():
    anchor, experts, evidence = _frame()
    built = build_feature_matrix(anchor, experts, evidence)
    features = built["feature_matrix"]
    assert FEATURE_VERSION == "stage019_s4_deployment_visible_object_features_v1"
    assert len(FEATURE_NAMES) == 41
    assert features.shape == (2, 41)
    assert tuple(built["expert_match_mask"][0].tolist()) == (True, True, True)
    assert tuple(built["expert_match_mask"][1].tolist()) == (True, False, False)
    # LiDAR and Camera blocks of anchor object 1 are exactly absent, including
    # their explicit `matched` columns.
    assert torch.count_nonzero(features[1, 17:]).item() == 0
    assert features[0, 1].item() == pytest.approx(0.0)
    assert features[1, 1].item() == pytest.approx(10.0)
    assert features[0, 2].item() == pytest.approx(math.log1p(5.0))
    assert features[1, 4].item() == pytest.approx(0.5)


def test_zero_initialization_and_fail_closed_paths_are_exact_identity():
    anchor, experts, evidence = _frame()
    module = ObjectSetAttributeFusion().eval()
    for mode in ("score_only", "attribute_only", "stacked"):
        output, audit = module(anchor, experts, evidence, mode=mode)
        assert torch.equal(output["boxes_3d"], anchor["boxes_3d"])
        assert torch.equal(output["scores_3d"], anchor["scores_3d"])
        assert torch.equal(output["labels_3d"], anchor["labels_3d"])
        assert audit["accepted_count"] == 0

    bypass, bypass_audit = module(
        anchor, {}, {}, mode="stacked", hard_bypass=True
    )
    assert bypass is anchor
    assert bypass_audit["status"] == "hard_bypass_identity"

    invalid_experts = dict(experts)
    invalid_experts["camera"] = dict(experts["camera"])
    invalid_experts["camera"]["boxes_3d"] = experts["camera"]["boxes_3d"].clone()
    invalid_experts["camera"]["boxes_3d"][0, 0] = float("nan")
    failed, failed_audit = module(
        anchor, invalid_experts, evidence, mode="stacked"
    )
    assert failed is anchor
    assert failed_audit["status"] == "fail_closed_invalid_input"

    leaky_anchor = dict(anchor)
    leaky_anchor["fault_mask"] = torch.ones(2, dtype=torch.bool)
    leaked, leaked_audit = module(
        leaky_anchor, experts, evidence, mode="stacked"
    )
    assert leaked is leaky_anchor
    assert leaked_audit["status"] == "fail_closed_invalid_input"
    assert "forbidden inference keys" in leaked_audit["reason"]


def test_modes_isolate_score_and_geometry_and_count_actual_modifications():
    anchor, experts, evidence = _frame()
    module = ObjectSetAttributeFusion().eval()
    with torch.no_grad():
        module.gace_lite[-1].bias.fill_(0.25)
        module.cp_afr[-1].bias[16] = 1.0

    score, score_audit = module(
        anchor, experts, evidence, mode="score_only"
    )
    assert torch.equal(score["boxes_3d"], anchor["boxes_3d"])
    assert not torch.equal(score["scores_3d"], anchor["scores_3d"])
    assert score_audit["accepted_count"] == 2

    attributes, attribute_audit = module(
        anchor, experts, evidence, mode="attribute_only"
    )
    assert not torch.equal(attributes["boxes_3d"], anchor["boxes_3d"])
    assert torch.equal(attributes["scores_3d"], anchor["scores_3d"])
    assert torch.equal(attributes["labels_3d"], anchor["labels_3d"])
    assert attribute_audit["accepted_count"] == 2

    stacked, stacked_audit = module(
        anchor, experts, evidence, mode="stacked"
    )
    assert not torch.equal(stacked["boxes_3d"], anchor["boxes_3d"])
    assert not torch.equal(stacked["scores_3d"], anchor["scores_3d"])
    assert stacked_audit["accepted_count"] == 2


def test_attribute_bounds_and_yaw_wrap_across_pi():
    anchor, experts, evidence = _frame()
    module = ObjectSetAttributeFusion().eval()
    with torch.no_grad():
        module.cp_afr[-1].bias[16] = 1.0
    output, audit = module(
        anchor, experts, evidence, mode="attribute_only"
    )
    centre_shift = torch.linalg.vector_norm(
        output["boxes_3d"][:, :3] - anchor["boxes_3d"][:, :3], dim=1
    )
    assert torch.all(centre_shift <= 1.0 + 1e-6)
    dim_log_shift = (
        output["boxes_3d"][:, 3:6].log() - anchor["boxes_3d"][:, 3:6].log()
    ).abs()
    assert torch.all(dim_log_shift <= 0.20 + 1e-6)
    yaw_shift = torch.atan2(
        torch.sin(output["boxes_3d"][:, 6] - anchor["boxes_3d"][:, 6]),
        torch.cos(output["boxes_3d"][:, 6] - anchor["boxes_3d"][:, 6]),
    )
    assert torch.all(yaw_shift.abs() <= math.pi / 9.0 + 1e-6)
    # The first experts lie just across +/-pi; circular fusion must take the
    # short arc, not a roughly 2*pi regression jump.
    assert 0.0 < yaw_shift[0].item() < 0.2
    velocity_shift = torch.linalg.vector_norm(
        output["boxes_3d"][:, 7:9] - anchor["boxes_3d"][:, 7:9], dim=1
    )
    assert torch.all(velocity_shift <= 2.0 + 1e-6)
    assert audit["accepted_count"] == 2


def test_trainable_heads_have_gradients_and_checkpoint_is_directly_loadable():
    anchor, experts, evidence = _frame()
    module = ObjectSetAttributeFusion().train()
    with torch.no_grad():
        module.cp_afr[-1].bias[16] = 0.1
    output, _ = module(anchor, experts, evidence, mode="stacked")
    loss = output["boxes_3d"].sum() + output["scores_3d"].sum()
    loss.backward()
    assert module.trainable_parameter_count() == 6547
    assert module.trainable_parameter_count() < 50000
    assert torch.count_nonzero(module.gace_lite[-1].weight.grad).item() > 0
    assert torch.count_nonzero(module.cp_afr[-1].weight.grad).item() > 0
    assert module.log_temperature.grad is not None
    payload = {CHECKPOINT_STATE_DICT_KEY: module.state_dict()}
    reloaded = ObjectSetAttributeFusion()
    reloaded.load_state_dict(payload[CHECKPOINT_STATE_DICT_KEY], strict=True)


def test_parameter_contract_allows_exactly_fifty_thousand():
    class BoundaryFusion(ObjectSetAttributeFusion):
        def trainable_parameter_count(self):
            return 50000

    class OverLimitFusion(ObjectSetAttributeFusion):
        def trainable_parameter_count(self):
            return 50001

    BoundaryFusion()
    with pytest.raises(RuntimeError, match="exceeds 50k"):
        OverLimitFusion()


def test_rescue_is_scientifically_disabled_until_a_trained_head_exists():
    anchor, experts, evidence = _frame()
    module = ObjectSetAttributeFusion(rescue_enabled=False).eval()
    output, audit = module(
        anchor,
        experts,
        evidence,
        mode="stacked",
        enable_rescue=True,
    )
    assert output is anchor
    assert audit["rescued_count"] == 0
    assert audit["status"] == (
        "scientifically_disabled_until_union_gate_and_trained_rescue_checkpoint"
    )
    with pytest.raises(ValueError, match=r"\[0,20\]"):
        ObjectSetAttributeFusion(max_rescue_boxes=21)


def test_s4_config_loads_without_mutating_default_model_inference():
    config_path = (
        ROOT
        / "projects"
        / "configs"
        / "mome"
        / "mome_stage019_s4_object_set_attribute_fusion.py"
    )
    config = Config.fromfile(str(config_path))
    declaration = config.object_set_attribute_fusion
    assert config.stage019_s4_protocol_profile == (
        "stage019_s4_object_set_attribute_fusion_v1"
    )
    assert declaration.enabled is False
    assert declaration.default_inference_mode is None
    assert declaration.feature_dim == 41
    assert declaration.feature_version == FEATURE_VERSION
    assert declaration.checkpoint_state_dict_key == CHECKPOINT_STATE_DICT_KEY
    assert declaration.constructor.hidden_dim == 64
    assert declaration.constructor.max_dim_log_delta == pytest.approx(0.20)
    assert declaration.constructor.max_yaw_delta == pytest.approx(math.pi / 9.0)
    assert declaration.constructor.max_rescue_boxes == 20
    # The declaration remains top-level; inherited MoME/simple_test config is
    # not silently replaced with an untrained S4 path.
    assert "object_set_attribute_fusion" not in config.model
