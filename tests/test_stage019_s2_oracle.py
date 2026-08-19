from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ROUTER_PATH = (
    ROOT
    / "projects"
    / "mmdet3d_plugin"
    / "models"
    / "utils"
    / "qta_router.py"
)
SPEC = importlib.util.spec_from_file_location("stage019_s2_qta", ROUTER_PATH)
assert SPEC is not None and SPEC.loader is not None
qta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qta)


def test_complete_query_permutation_accepts_only_exact_identity_set() -> None:
    observed = torch.tensor([[2, 0, 3, 1]])
    assert torch.equal(
        qta.validate_complete_query_permutation(observed, 4), observed.long()
    )
    for invalid in (
        torch.tensor([[0, 1, 1, 3]]),
        torch.tensor([[0, 1, 2, 4]]),
        torch.tensor([[0, 1, 2]]),
    ):
        with pytest.raises(RuntimeError):
            qta.validate_complete_query_permutation(invalid, 4)


def test_reorder_query_tensor_restores_original_query_order() -> None:
    values = torch.tensor([[[20.0], [0.0], [30.0], [10.0]]])
    query_ids = torch.tensor([[2, 0, 3, 1]])
    restored = qta.reorder_query_tensor_to_original(values, query_ids)
    assert torch.equal(
        restored, torch.tensor([[[0.0], [10.0], [20.0], [30.0]]])
    )


def test_robust_action_selection_uses_margin_and_locked_ties() -> None:
    anchor = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    route_losses = torch.tensor(
        [[
            [4.0, 3.0, 4.5],  # LiDAR has largest lower-bound gain.
            [3.0, 3.0, 4.0],  # tie; original Camera is not tied -> Fused.
            [3.0, 3.0, 4.0],  # tie; original LiDAR is tied -> LiDAR.
            [4.95, 4.95, 4.95],  # finite gain below epsilon -> KEEP.
        ]]
    )
    original = torch.tensor([[0, 2, 1, 0]])
    actions, switched, lower, best = qta.select_robust_oracle_actions(
        anchor,
        route_losses,
        epsilon_query=0.1,
        original_routes=original,
    )
    assert torch.equal(actions, torch.tensor([[1, 0, 1, -1]]))
    assert torch.equal(switched, torch.tensor([[True, True, True, False]]))
    assert lower.shape == (1, 4, 3)
    assert torch.isneginf(best[0, 3])

    invalid = route_losses.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        qta.select_robust_oracle_actions(
            anchor,
            invalid,
            epsilon_query=0.1,
            original_routes=original,
        )


def test_candidate_quality_and_order_are_unique_and_stable() -> None:
    gains = torch.tensor(
        [[
            [0.0, 1.0, 1.0],
            [0.5, 0.7, 0.7],
        ]]
    )
    quality = qta.robust_candidate_quality(
        gains,
        epsilon_query=0.1,
        proxy_overestimate=0.2,
        current_routes=torch.tensor([[0, 1]]),
    )
    ranked = qta.rank_greedy_candidates(quality)
    assert ranked == [
        (0, 1, pytest.approx(0.7)),
        (0, 2, pytest.approx(0.7)),
        (1, 2, pytest.approx(0.4)),
        (1, 0, pytest.approx(0.2)),
    ]


def test_candidate_budget_zero_reached_and_k16_stop() -> None:
    zero = qta.select_candidate_budget(torch.zeros((2, 4, 3)))
    assert zero["status"] == "zero_positive_quality"
    assert zero["K"] == 0

    concentrated = torch.zeros((2, 20, 3))
    concentrated[:, :2, 0] = 1.0
    reached = qta.select_candidate_budget(concentrated)
    assert reached["status"] == "coverage_reached"
    assert reached["K"] == 4

    diffuse = torch.ones((1, 100, 3))
    stopped = qta.select_candidate_budget(diffuse)
    assert stopped["status"] == "candidate_mass_not_coverable_under_K16"
    assert stopped["K"] is None


def test_locked_route_state_requires_complete_integer_routes() -> None:
    reference = torch.zeros((1, 4), dtype=torch.long)
    locked = torch.tensor([2, 1, 0, 2], dtype=torch.int8)
    normalized = qta.normalize_locked_route_state(locked, reference)
    assert torch.equal(normalized, locked.long().unsqueeze(0))

    with pytest.raises(ValueError, match="integer dtype"):
        qta.normalize_locked_route_state(locked.float(), reference)
    with pytest.raises(ValueError, match="shape"):
        qta.normalize_locked_route_state(torch.tensor([0, 1, 2]), reference)
    with pytest.raises(ValueError, match="invalid route|values"):
        qta.normalize_locked_route_state(torch.tensor([0, 1, 2, 3]), reference)


def test_route_assignment_masks_do_not_infer_presence_from_hidden_values() -> None:
    routes = torch.tensor([[0, 1, 2, 1]], dtype=torch.long)
    masks = qta.route_assignment_masks(routes, num_layers=2)
    assert len(masks) == 3
    assert all(mask.shape == (2, 1, 4) for mask in masks)
    assert torch.equal(
        torch.stack(masks, dim=0).sum(dim=0),
        torch.ones((2, 1, 4), dtype=torch.long),
    )
    assert torch.equal(
        qta.output_query_indices_from_masks(masks, num_queries=4),
        torch.tensor([[0, 1, 3, 2]]),
    )
