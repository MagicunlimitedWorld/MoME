"""Query-task-advantage routing for MoME.

The module intentionally contains only one trainable layer.  It predicts the
task-loss advantage of executing each MoME route relative to the route chosen
by the frozen upstream AQR.  Route selection is conservative: the upstream
route is kept unless a different route clears its calibrated threshold.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ROUTE_FUSED = 0
ROUTE_LIDAR = 1
ROUTE_CAMERA = 2
NUM_ROUTES = 3


def validate_complete_query_permutation(
    query_indices: torch.Tensor,
    num_queries: int,
    label: str = "output_query_indices",
) -> torch.Tensor:
    """Fail closed unless every batch row is exactly a permutation of 0..N-1."""

    if not torch.is_tensor(query_indices):
        raise TypeError(f"{label} must be a torch.Tensor")
    if query_indices.ndim != 2 or query_indices.shape[1] != int(num_queries):
        raise RuntimeError(
            f"{label} must have shape [B,{int(num_queries)}], got "
            f"{tuple(query_indices.shape)}"
        )
    if query_indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise RuntimeError(f"{label} must use an integer dtype")
    expected = torch.arange(
        int(num_queries), device=query_indices.device, dtype=query_indices.dtype
    ).unsqueeze(0).expand(query_indices.shape[0], -1)
    observed = torch.sort(query_indices, dim=1).values
    if not torch.equal(observed, expected):
        minimum = int(query_indices.min().item()) if query_indices.numel() else None
        maximum = int(query_indices.max().item()) if query_indices.numel() else None
        unique_counts = [
            int(torch.unique(row).numel()) for row in query_indices
        ]
        raise RuntimeError(
            f"{label} is not a complete unique 0..{int(num_queries) - 1} "
            f"permutation (min={minimum}, max={maximum}, "
            f"unique_counts={unique_counts})"
        )
    return query_indices.long()


def reorder_query_tensor_to_original(
    values: torch.Tensor,
    query_indices: torch.Tensor,
) -> torch.Tensor:
    """Reorder a ``[B,N,...]`` tensor from output order to original query ids."""

    if values.ndim < 2:
        raise ValueError("values must have shape [B,N,...]")
    indices = validate_complete_query_permutation(
        query_indices, values.shape[1], label="query_indices"
    )
    if values.shape[:2] != indices.shape:
        raise ValueError("values and query_indices must share [B,N]")
    output = torch.empty_like(values)
    for batch_index in range(values.shape[0]):
        output[batch_index].index_copy_(
            0, indices[batch_index], values[batch_index]
        )
    return output


def _broadcast_like(value: object, reference: torch.Tensor, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    try:
        return torch.broadcast_to(tensor, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"{label} must broadcast to {tuple(reference.shape)}"
        ) from error


def select_robust_oracle_actions(
    anchor_query_losses: torch.Tensor,
    route_query_losses: torch.Tensor,
    epsilon_query: object,
    original_routes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select finite positive lower-bound actions with the locked tie rule.

    The returned action is ``-1`` for KEEP and otherwise the destination route.
    Exact lower-bound ties prefer the original MoME route when it is eligible;
    remaining ties use the natural Fused=0, LiDAR=1, Camera=2 order.
    """

    if anchor_query_losses.ndim != 2:
        raise ValueError("anchor_query_losses must have shape [B,N]")
    if route_query_losses.shape != (*anchor_query_losses.shape, NUM_ROUTES):
        raise ValueError("route_query_losses must have shape [B,N,3]")
    if original_routes.shape != anchor_query_losses.shape:
        raise ValueError("original_routes must share anchor [B,N] shape")
    if torch.any((original_routes < 0) | (original_routes >= NUM_ROUTES)):
        raise ValueError("original_routes values must be in {0, 1, 2}")

    epsilon = _broadcast_like(epsilon_query, route_query_losses, "epsilon_query")
    if not torch.isfinite(anchor_query_losses).all():
        raise RuntimeError("anchor_query_losses contains non-finite values")
    if not torch.isfinite(route_query_losses).all():
        raise RuntimeError("route_query_losses contains non-finite values")
    if not torch.isfinite(epsilon).all() or torch.any(epsilon < 0):
        raise ValueError("epsilon_query must be finite and non-negative")
    gains = anchor_query_losses.unsqueeze(-1) - route_query_losses
    lower_bounds = gains - epsilon
    eligible = (
        torch.isfinite(anchor_query_losses).unsqueeze(-1)
        & torch.isfinite(route_query_losses)
        & torch.isfinite(epsilon)
        & torch.isfinite(lower_bounds)
        & (lower_bounds > 0)
    )
    masked = lower_bounds.masked_fill(~eligible, float("-inf"))
    best_lower_bounds, best_routes = masked.max(dim=-1)
    has_action = torch.isfinite(best_lower_bounds)

    original_lower = lower_bounds.gather(
        -1, original_routes.unsqueeze(-1)
    ).squeeze(-1)
    original_eligible = eligible.gather(
        -1, original_routes.unsqueeze(-1)
    ).squeeze(-1)
    prefer_original = (
        has_action
        & original_eligible
        & (original_lower == best_lower_bounds)
    )
    best_routes = torch.where(prefer_original, original_routes, best_routes)
    actions = torch.where(
        has_action, best_routes, torch.full_like(best_routes, -1)
    )
    return actions, has_action, lower_bounds, best_lower_bounds


def robust_candidate_quality(
    full_context_gains: torch.Tensor,
    epsilon_query: object,
    proxy_overestimate: object,
    current_routes: torch.Tensor,
) -> torch.Tensor:
    """Compute locked S2-B candidate quality ``max(0, g-epsilon-o)``."""

    if full_context_gains.ndim != 3 or full_context_gains.shape[-1] != NUM_ROUTES:
        raise ValueError("full_context_gains must have shape [B,N,3]")
    if current_routes.shape != full_context_gains.shape[:2]:
        raise ValueError("current_routes must have shape [B,N]")
    epsilon = _broadcast_like(epsilon_query, full_context_gains, "epsilon_query")
    overestimate = _broadcast_like(
        proxy_overestimate, full_context_gains, "proxy_overestimate"
    )
    if torch.any((current_routes < 0) | (current_routes >= NUM_ROUTES)):
        raise ValueError("current_routes values must be in {0, 1, 2}")
    if not torch.isfinite(epsilon).all() or torch.any(epsilon < 0):
        raise ValueError("epsilon_query must be finite and non-negative")
    if not torch.isfinite(overestimate).all() or torch.any(overestimate < 0):
        raise ValueError("proxy_overestimate must be finite and non-negative")
    raw = full_context_gains - epsilon - overestimate
    quality = torch.where(
        torch.isfinite(raw) & (raw > 0), raw, torch.zeros_like(raw)
    )
    quality = quality.scatter(
        -1, current_routes.unsqueeze(-1), torch.zeros_like(current_routes.unsqueeze(-1), dtype=quality.dtype)
    )
    return quality


def rank_greedy_candidates(
    quality: torch.Tensor,
) -> list[tuple[int, int, float]]:
    """Return unique positive candidates ordered by ``(-w, query, expert)``."""

    if quality.ndim == 3:
        if quality.shape[0] != 1:
            raise ValueError("candidate ranking requires batch size one")
        quality = quality[0]
    if quality.ndim != 2 or quality.shape[1] != NUM_ROUTES:
        raise ValueError("quality must have shape [N,3] or [1,N,3]")
    records = []
    for query_id, expert_id in torch.nonzero(
        torch.isfinite(quality) & (quality > 0), as_tuple=False
    ).detach().cpu().tolist():
        records.append(
            (int(query_id), int(expert_id), float(quality[query_id, expert_id].item()))
        )
    records.sort(key=lambda item: (-item[2], item[0], item[1]))
    return records


def select_candidate_budget(
    frame_quality: torch.Tensor,
    candidate_ks: Sequence[int] = (4, 8, 16),
    required_coverage: float = 0.95,
) -> dict[str, object]:
    """Lock K from pooled per-frame top-K robust quality mass."""

    if frame_quality.ndim != 3 or frame_quality.shape[-1] != NUM_ROUTES:
        raise ValueError("frame_quality must have shape [F,N,3]")
    ks = tuple(int(value) for value in candidate_ks)
    if not ks or any(value <= 0 for value in ks) or tuple(sorted(ks)) != ks:
        raise ValueError("candidate_ks must be positive and increasing")
    if not 0 < float(required_coverage) <= 1:
        raise ValueError("required_coverage must lie in (0,1]")
    finite_positive = torch.where(
        torch.isfinite(frame_quality) & (frame_quality > 0),
        frame_quality,
        torch.zeros_like(frame_quality),
    )
    flattened = finite_positive.reshape(finite_positive.shape[0], -1)
    total = float(flattened.sum().item())
    coverage: dict[int, float] = {}
    if total == 0.0:
        return {
            "status": "zero_positive_quality",
            "K": 0,
            "total_positive_quality": 0.0,
            "coverage": coverage,
        }
    for value in ks:
        top_count = min(value, flattened.shape[1])
        captured = torch.topk(
            flattened, k=top_count, dim=1, largest=True, sorted=False
        ).values.sum()
        coverage[value] = float(captured.item()) / total
        if coverage[value] >= float(required_coverage):
            return {
                "status": "coverage_reached",
                "K": value,
                "total_positive_quality": total,
                "coverage": coverage,
            }
    return {
        "status": "candidate_mass_not_coverable_under_K16",
        "K": None,
        "total_positive_quality": total,
        "coverage": coverage,
    }


def select_full_query_oracle_routes(
    base_routes: torch.Tensor,
    route_query_losses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the unique strictly-better GT route for every query.

    This is an evaluation-only Oracle. ``route_query_losses[b, n, e]`` must
    compare the same original query against the same fixed Hungarian target
    for all three frozen experts. Non-finite rows, ties, non-positive gains,
    and rows whose winner is already the upstream route retain ``base_routes``.
    """

    if base_routes.ndim != 2:
        raise ValueError("base_routes must have shape [B,N]")
    if route_query_losses.shape != (*base_routes.shape, NUM_ROUTES):
        raise ValueError("route_query_losses must have shape [B,N,3]")
    if torch.any((base_routes < 0) | (base_routes >= NUM_ROUTES)):
        raise ValueError("base_routes values must be in {0, 1, 2}")

    finite = torch.isfinite(route_query_losses)
    masked = route_query_losses.masked_fill(~finite, float("inf"))
    best_losses, best_routes = masked.min(dim=-1)
    base_losses = route_query_losses.gather(
        dim=-1, index=base_routes.unsqueeze(-1)
    ).squeeze(-1)
    gains = base_losses - best_losses
    winner_count = ((masked == best_losses.unsqueeze(-1)) & finite).sum(dim=-1)
    switch_mask = (
        finite.all(dim=-1)
        & torch.isfinite(base_losses)
        & torch.isfinite(best_losses)
        & torch.isfinite(gains)
        & (winner_count == 1)
        & (best_routes != base_routes)
        & (gains > 0)
    )
    oracle_routes = torch.where(switch_mask, best_routes, base_routes)
    return oracle_routes, switch_mask, gains, best_routes


def _as_threshold_tensor(
    thresholds: Iterable[float], reference: torch.Tensor
) -> torch.Tensor:
    value = reference.new_tensor(tuple(float(item) for item in thresholds))
    if value.numel() != NUM_ROUTES:
        raise ValueError("QTA requires exactly three destination thresholds")
    return value


def normalize_route_override(
    route_override: object,
    base_routes: torch.Tensor,
) -> torch.Tensor:
    """Normalize an audit-only route override to ``[batch, query]``."""

    if isinstance(route_override, int):
        routes = torch.full_like(base_routes, route_override)
    elif torch.is_tensor(route_override):
        routes = route_override.to(device=base_routes.device, dtype=torch.long)
        if routes.ndim == 1 and base_routes.shape[0] == 1:
            routes = routes.unsqueeze(0)
        if routes.shape != base_routes.shape:
            raise ValueError(
                "route_override must be an integer or have the same [B,N] "
                "shape as base_routes"
            )
    else:
        raise TypeError("route_override must be an integer or torch.Tensor")
    if torch.any((routes < 0) | (routes >= NUM_ROUTES)):
        raise ValueError("route_override values must be in {0, 1, 2}")
    return routes


def normalize_locked_route_state(
    locked_routes: torch.Tensor,
    reference_routes: torch.Tensor,
    label: str = "locked_base_routes",
) -> torch.Tensor:
    """Validate a complete replay route state without integer broadcast."""

    if not torch.is_tensor(locked_routes):
        raise TypeError(f"{label} must be a torch.Tensor")
    if locked_routes.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError(f"{label} must use an integer dtype")
    routes = locked_routes.to(device=reference_routes.device, dtype=torch.long)
    if routes.ndim == 1 and reference_routes.shape[0] == 1:
        routes = routes.unsqueeze(0)
    if routes.shape != reference_routes.shape:
        raise ValueError(
            f"{label} must have shape {tuple(reference_routes.shape)}, got "
            f"{tuple(routes.shape)}"
        )
    if torch.any((routes < 0) | (routes >= NUM_ROUTES)):
        raise ValueError(f"{label} values must be in {{0, 1, 2}}")
    return routes


def output_query_indices_from_masks(
    route_masks: list[torch.Tensor],
    num_queries: int,
) -> torch.Tensor:
    """Map grouped eval outputs back to their original query ids.

    Frozen MED concatenates Fused, LiDAR, and Camera output blocks, then keeps
    the non-zero entries described by ``zero_idx``.  QTA diagnostics run with
    batch size one and need this exact mapping without changing inference.
    """

    if not route_masks:
        raise ValueError("route_masks cannot be empty")
    combined = torch.cat(route_masks, dim=2)
    if combined.ndim != 3 or combined.shape[1] != 1:
        raise ValueError("QTA query-id diagnostics require batch size one")
    if any(mask.shape[2] != num_queries for mask in route_masks):
        raise ValueError("each route mask must contain the original query count")
    repeated = torch.arange(num_queries, device=combined.device).repeat(
        len(route_masks)
    )
    selected = combined[-1, 0].to(dtype=torch.bool)
    if selected.numel() != repeated.numel():
        raise ValueError("route mask concatenation does not match query ids")
    return repeated[selected].unsqueeze(0)


def route_assignment_masks(
    final_routes: torch.Tensor,
    num_layers: int,
    num_routes: int = NUM_ROUTES,
) -> list[torch.Tensor]:
    """Build exact diagnostic presence masks from the executed route state.

    A decoder hidden vector can legitimately sum to zero, so tensor values are
    not a reliable indication that a routed query was executed.  These masks
    are used only by explicit route-state diagnostics; normal MoME inference
    keeps its original path unchanged.
    """

    if not torch.is_tensor(final_routes) or final_routes.ndim != 2:
        raise ValueError("final_routes must be a [B,N] tensor")
    if final_routes.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("final_routes must use an integer dtype")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be positive")
    if int(num_routes) <= 0:
        raise ValueError("num_routes must be positive")
    routes = final_routes.long()
    if torch.any((routes < 0) | (routes >= int(num_routes))):
        raise ValueError("final_routes contain an invalid route")
    return [
        (routes == route_id)
        .unsqueeze(0)
        .expand(int(num_layers), -1, -1)
        .clone()
        for route_id in range(int(num_routes))
    ]


def select_conservative_routes(
    base_routes: torch.Tensor,
    advantage_scores: torch.Tensor,
    thresholds: Iterable[float],
    max_overrides_per_frame: Optional[int] = None,
    route_available: Optional[torch.Tensor] = None,
    hard_bypass: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select calibrated route overrides while retaining a literal KEEP.

    ``advantage_scores[b, n, e]`` estimates
    ``loss(base_routes[b,n]) - loss(do(route=e))``.  The base route is never
    treated as an override.  ``hard_bypass[b]`` preserves all upstream routes
    for complete-modality failures, and the optional frame cap limits
    non-additive multi-query interventions.
    """

    if base_routes.ndim != 2:
        raise ValueError("base_routes must have shape [B,N]")
    if advantage_scores.shape != (*base_routes.shape, NUM_ROUTES):
        raise ValueError("advantage_scores must have shape [B,N,3]")
    if max_overrides_per_frame is not None and max_overrides_per_frame < 0:
        raise ValueError("max_overrides_per_frame cannot be negative")

    thresholds_t = _as_threshold_tensor(thresholds, advantage_scores)
    eligible = advantage_scores > thresholds_t.view(1, 1, NUM_ROUTES)
    eligible.scatter_(2, base_routes.unsqueeze(-1), False)

    if route_available is not None:
        available = route_available.to(device=base_routes.device, dtype=torch.bool)
        if available.ndim == 2 and available.shape == (base_routes.shape[0], NUM_ROUTES):
            available = available.unsqueeze(1).expand(-1, base_routes.shape[1], -1)
        if available.shape != eligible.shape:
            raise ValueError("route_available must have shape [B,3] or [B,N,3]")
        eligible &= available

    masked_scores = advantage_scores.masked_fill(~eligible, float("-inf"))
    best_scores, best_routes = masked_scores.max(dim=-1)
    override_mask = torch.isfinite(best_scores)

    if hard_bypass is not None:
        bypass = hard_bypass.to(device=base_routes.device, dtype=torch.bool)
        if bypass.shape != (base_routes.shape[0],):
            raise ValueError("hard_bypass must have shape [B]")
        override_mask &= ~bypass.unsqueeze(1)

    if max_overrides_per_frame is not None:
        capped = torch.zeros_like(override_mask)
        if max_overrides_per_frame > 0:
            for batch_index in range(base_routes.shape[0]):
                candidate_indices = torch.nonzero(
                    override_mask[batch_index], as_tuple=False
                ).flatten()
                if candidate_indices.numel() <= max_overrides_per_frame:
                    capped[batch_index, candidate_indices] = True
                    continue
                candidate_scores = best_scores[batch_index, candidate_indices]
                selected = torch.topk(
                    candidate_scores,
                    k=max_overrides_per_frame,
                    largest=True,
                    sorted=False,
                ).indices
                capped[batch_index, candidate_indices[selected]] = True
        override_mask = capped

    final_routes = torch.where(override_mask, best_routes, base_routes)
    return final_routes, override_mask


def masked_advantage_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Huber regression over valid route-level loss-difference labels."""

    if predicted.shape != target.shape or predicted.shape != valid_mask.shape:
        raise ValueError("predicted, target and valid_mask must share shape [B,N,3]")
    element_loss = F.smooth_l1_loss(predicted, target, reduction="none")
    weights = valid_mask.to(dtype=element_loss.dtype)
    if sample_weight is not None:
        if sample_weight.shape not in {predicted.shape, predicted.shape[:-1]}:
            raise ValueError("sample_weight must have shape [B,N] or [B,N,3]")
        if sample_weight.ndim == predicted.ndim - 1:
            sample_weight = sample_weight.unsqueeze(-1)
        weights = weights * sample_weight.to(device=predicted.device, dtype=predicted.dtype)
    denominator = weights.sum().clamp_min(1.0)
    return (element_loss * weights).sum() / denominator


class QueryTaskAdvantageRouter(nn.Module):
    """The 256-to-3, 771-parameter MoME-QTA-AQR correction head."""

    def __init__(
        self,
        input_dim: int = 256,
        thresholds: Iterable[float] = (float("inf"),) * NUM_ROUTES,
        max_overrides_per_frame: Optional[int] = 0,
        enabled: bool = False,
        checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        if input_dim != 256:
            raise ValueError("MoME-QTA-AQR freezes the 256-dimensional AQR representation")
        self.advantage_head = nn.Linear(input_dim, NUM_ROUTES)
        self.thresholds = tuple(float(item) for item in thresholds)
        if len(self.thresholds) != NUM_ROUTES:
            raise ValueError("thresholds must contain fused, LiDAR and camera values")
        self.max_overrides_per_frame = max_overrides_per_frame
        self.enabled = bool(enabled)
        self.checkpoint = checkpoint
        if self.checkpoint is not None:
            self.load_advantage_checkpoint()

    def load_advantage_checkpoint(self) -> None:
        """Load/reload the 771-parameter head after parent initialization."""

        if self.checkpoint is None:
            return
        payload = torch.load(self.checkpoint, map_location="cpu")
        if int(payload.get("trainable_parameter_count", 771)) != 771:
            raise ValueError("QTA checkpoint parameter-count contract mismatch")
        state_dict = payload.get("state_dict", payload)
        if any(key.startswith("advantage_head.") for key in state_dict):
            self.load_state_dict(state_dict, strict=True)
        else:
            self.advantage_head.load_state_dict(state_dict, strict=True)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        router_features: torch.Tensor,
        base_routes: torch.Tensor,
        route_available: Optional[torch.Tensor] = None,
        hard_bypass: Optional[torch.Tensor] = None,
        thresholds: Optional[Iterable[float]] = None,
        max_overrides_per_frame: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        if router_features.shape[:-1] != base_routes.shape:
            raise ValueError("router_features must have shape [B,N,256]")
        advantage_scores = self.advantage_head(router_features)
        if not self.enabled:
            final_routes = base_routes.clone()
            override_mask = torch.zeros_like(base_routes, dtype=torch.bool)
        else:
            final_routes, override_mask = select_conservative_routes(
                base_routes=base_routes,
                advantage_scores=advantage_scores,
                thresholds=self.thresholds if thresholds is None else thresholds,
                max_overrides_per_frame=(
                    self.max_overrides_per_frame
                    if max_overrides_per_frame is None
                    else max_overrides_per_frame
                ),
                route_available=route_available,
                hard_bypass=hard_bypass,
            )
        return {
            "advantage_scores": advantage_scores,
            "final_routes": final_routes,
            "override_mask": override_mask,
        }
