from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from projects.mmdet3d_plugin.models.dense_heads.med import MultiExpertDecoding
from projects.mmdet3d_plugin.models.detectors.mome import MoME


FIELDS = ("cls_logits", "center", "height", "dim", "rot", "vel")
WIDTHS = {
    "cls_logits": 10,
    "center": 2,
    "height": 1,
    "dim": 3,
    "rot": 2,
    "vel": 2,
}


class _FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_query = 900
        self.transformer = SimpleNamespace(qta_router=None)
        self.calls = []

    def forward(
        self,
        pts_feats,
        img_feats,
        img_metas,
        route_override=None,
        return_router_state=False,
    ):
        assert return_router_state is True
        self.calls.append(route_override)
        route_code = -1 if route_override is None else int(route_override)
        shift = 7 if route_override is None else route_code
        query_ids = torch.roll(torch.arange(900), shifts=shift).unsqueeze(0)
        task = {
            "output_query_indices": query_ids,
            "base_routes": torch.arange(900).remainder(3).unsqueeze(0),
            "final_routes": torch.full((1, 900), max(route_code, 0)),
            "router_features": torch.full((1, 900, 256), float(route_code)),
            "reference_points": torch.zeros((1, 900, 3)),
            "route_code": route_code,
        }
        for field in FIELDS:
            task[field] = torch.full(
                (1, 1, 900, WIDTHS[field]), float(route_code)
            )
        return ([task],)

    def validate_s2_prediction_bundle(
        self, predictions, label, expected_query_count=None
    ):
        return MultiExpertDecoding.validate_s2_prediction_bundle(
            self, predictions, label, expected_query_count
        )

    @staticmethod
    def get_bboxes(predictions, img_metas, rescale=False):
        route_code = predictions[0][0]["route_code"]
        boxes = torch.tensor(
            [[float(route_code), 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
        )
        scores = torch.tensor([0.5 + 0.1 * float(route_code)])
        labels = torch.tensor([max(int(route_code), 0)], dtype=torch.long)
        return [(boxes, scores, labels)]


def _model():
    model = object.__new__(MoME)
    nn.Module.__init__(model)
    model.pts_bbox_head = _FakeHead()
    model.local_evidence_rule = None
    extraction_calls = []

    def extract_feat(points, img=None, img_metas=None):
        extraction_calls.append((points, img, img_metas))
        return [torch.ones(1)], [torch.ones(1)]

    model.extract_feat = extract_feat
    model.eval()
    return model, extraction_calls


def _inputs(zero_lidar=False, zero_camera=False):
    points = [
        torch.zeros((3, 5)) if zero_lidar else torch.ones((3, 5))
    ]
    img = (
        torch.zeros((1, 6, 3, 2, 2))
        if zero_camera
        else torch.ones((1, 6, 3, 2, 2))
    )
    return points, [{}], img


def test_no_gt_full_context_bundle_runs_one_backbone_and_route_order_012():
    model, extraction_calls = _model()
    points, img_metas, img = _inputs()
    output = model._forward_full_context_expert_bundle(
        points, img_metas, img=img
    )
    assert len(extraction_calls) == 1
    assert model.pts_bbox_head.calls == [None, 0, 1, 2]
    assert output["decoder_call_count"] == 4
    assert output["hard_bypass"] is False
    assert output["route_order"] == ("fused", "lidar", "camera")
    assert tuple(output["expert_bboxes"]) == ("fused", "lidar", "camera")
    assert tuple(output["query_ids"]["experts"]) == (
        "fused",
        "lidar",
        "camera",
    )
    assert output["query_ids"]["original"].shape == (1, 900)
    assert all(
        value.shape == (1, 900)
        for value in output["query_ids"]["experts"].values()
    )
    assert output["base_routes"].shape == (1, 900)
    assert output["router_features"].shape == (1, 900, 256)
    assert output["reference_points"].shape == (1, 900, 3)
    assert output["original_bbox"][0]["boxes_3d"].shape == (1, 9)
    assert output["expert_bboxes"]["camera"][0]["labels_3d"].item() == 2


def test_complete_zero_is_detected_from_inputs_and_runs_original_only():
    for zero_lidar, zero_camera in ((True, False), (False, True), (True, True)):
        model, extraction_calls = _model()
        points, img_metas, img = _inputs(zero_lidar, zero_camera)
        output = model._forward_full_context_expert_bundle(
            points, img_metas, img=img
        )
        assert len(extraction_calls) == 1
        assert model.pts_bbox_head.calls == [None]
        assert output["hard_bypass"] is True
        assert output["decoder_call_count"] == 1
        assert output["expert_bboxes"] == {}
        assert output["expert_raw_predictions"] == {}
        assert output["query_ids"]["experts"] == {}
        assert output["original_bbox"][0]["scores_3d"].item() == pytest.approx(0.4)


def test_legacy_oracle_switch_keeps_four_decoders_even_for_zero_input():
    model, _ = _model()
    points, img_metas, img = _inputs(zero_lidar=True)
    output = model._forward_full_context_expert_bundle(
        points,
        img_metas,
        img=img,
        decode=False,
        complete_zero_bypass=False,
    )
    assert model.pts_bbox_head.calls == [None, 0, 1, 2]
    assert output["hard_bypass"] is False
    assert output["decoder_call_count"] == 4
    assert output["original_bbox"] is None
    assert output["expert_bboxes"] == {}


def test_no_gt_bundle_signature_has_no_scientific_shortcuts():
    parameters = inspect.signature(
        MoME._forward_full_context_expert_bundle
    ).parameters
    forbidden = ("gt", "fault", "mask", "corruption", "condition")
    assert not any(
        token in name.lower() for name in parameters for token in forbidden
    )
