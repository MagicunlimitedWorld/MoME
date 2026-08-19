from __future__ import annotations

import torch

from projects.mmdet3d_plugin.models.dense_heads.med import MultiExpertDecoding


FIELDS = ('cls_logits', 'center', 'height', 'dim', 'rot', 'vel')


def _bundle(query_ids, offset):
    task = {'output_query_indices': torch.tensor([query_ids], dtype=torch.long)}
    for field_index, field in enumerate(FIELDS):
        width = field_index + 1
        value = torch.zeros((2, 1, 4, width), dtype=torch.float32)
        for output_position, original_id in enumerate(query_ids):
            value[0, 0, output_position] = offset + 100 + original_id
            value[1, 0, output_position] = offset + original_id
        task[field] = value
    return ([task],)


def _head():
    head = object.__new__(MultiExpertDecoding)
    object.__setattr__(head, 'num_query', 4)
    return head


def test_final_six_fields_copy_by_original_identity_only() -> None:
    head = _head()
    anchor = _bundle([2, 0, 3, 1], 0)
    sources = [
        _bundle([3, 2, 1, 0], 1000),
        _bundle([1, 3, 0, 2], 2000),
        _bundle([0, 1, 2, 3], 3000),
    ]
    # Original ids 0 and 2 keep anchor; id 1 <- LiDAR; id 3 <- Camera.
    routes = torch.tensor([[-1, 1, -1, 2]])
    composed = head.compose_final_detection_head_outputs(
        anchor, sources, routes, label='unit'
    )
    anchor_positions = {query_id: pos for pos, query_id in enumerate([2, 0, 3, 1])}
    for field in FIELDS:
        observed = composed[0][0][field]
        original = anchor[0][0][field]
        # Earlier decoder layer is untouched.
        assert torch.equal(observed[0], original[0])
        assert torch.equal(
            observed[-1, 0, anchor_positions[0]],
            original[-1, 0, anchor_positions[0]],
        )
        assert torch.equal(
            observed[-1, 0, anchor_positions[2]],
            original[-1, 0, anchor_positions[2]],
        )
        assert torch.all(
            observed[-1, 0, anchor_positions[1]] == 2001
        )
        assert torch.all(
            observed[-1, 0, anchor_positions[3]] == 3003
        )


def test_composition_rejects_duplicate_query_identity_and_nonfinite_source() -> None:
    head = _head()
    anchor = _bundle([0, 1, 2, 3], 0)
    duplicate = _bundle([0, 1, 1, 3], 1000)
    valid = _bundle([0, 1, 2, 3], 2000)
    routes = torch.full((1, 4), -1, dtype=torch.long)
    try:
        head.compose_final_detection_head_outputs(
            anchor, [duplicate, valid, valid], routes, label='duplicate'
        )
    except RuntimeError as error:
        assert 'complete unique' in str(error)
    else:
        raise AssertionError('duplicate query ids must fail closed')

    nonfinite = _bundle([0, 1, 2, 3], 1000)
    nonfinite[0][0]['center'][-1, 0, 0, 0] = float('nan')
    try:
        head.compose_final_detection_head_outputs(
            anchor, [nonfinite, valid, valid], routes, label='nonfinite'
        )
    except RuntimeError as error:
        assert 'non-finite' in str(error)
    else:
        raise AssertionError('nonfinite source field must fail closed')


def test_exact_route_diagnostics_replace_stale_899_query_identity() -> None:
    routes = torch.arange(900, dtype=torch.long).remainder(3).unsqueeze(0)
    ca_dict = {
        'final_routes': routes,
        'output_query_indices': torch.arange(899).unsqueeze(0),
    }
    outs_dec = [
        torch.zeros((6, 1, 900, 128), dtype=torch.float32)
        for _ in range(3)
    ]

    masks, query_indices = MultiExpertDecoding._exact_route_assignment_diagnostics(
        ca_dict, outs_dec
    )

    assert all(mask.shape == (6, 1, 900) for mask in masks)
    assert query_indices.shape == (1, 900)
    assert torch.equal(torch.sort(query_indices, dim=1).values, torch.arange(900).unsqueeze(0))
