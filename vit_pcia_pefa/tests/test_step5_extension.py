from __future__ import annotations

import math

import torch

import vit_pcia_pefa.src.step5_extension as extension


def _rotational_port_fields(_model, root, *, requested_names, **_kwargs):
    outputs = []
    flat = root.reshape(-1)
    for name in requested_names:
        value = torch.zeros_like(root)
        if name == "head_0":
            value = value.reshape(-1)
            value[0] = -0.5 * flat[1]
            value[1] = 0.5 * flat[0]
            value = value.reshape_as(root)
        outputs.append(value)
    return torch.stack(outputs), tuple(requested_names)


def test_local_curl_and_closed_loop_match(monkeypatch):
    monkeypatch.setattr(extension, "port_fields", _rotational_port_fields)
    base = torch.zeros((1, 3, 2, 2), dtype=torch.float64)
    first = torch.zeros_like(base); first.reshape(-1)[0] = 1
    second = torch.zeros_like(base); second.reshape(-1)[1] = 1
    curl = extension.local_curl_scalars(
        None, base, block_index=0, class_id=0, component_names=("head_0",),
        first=first, second=second,
    )
    assert math.isclose(curl["head_0"], 1.0, rel_tol=0, abs_tol=1e-12)
    epsilon = 0.4
    corners = (base, base + epsilon * first, base + epsilon * first + epsilon * second,
               base + epsilon * second, base)
    total = torch.zeros(1, dtype=torch.float64)
    nodes = [0.21132486540518713, 0.7886751345948129]
    weights = [0.5, 0.5]
    for start, end in zip(corners[:-1], corners[1:]):
        total += extension.line_integral(
            None, start, end, block_index=0, class_id=0,
            component_names=("head_0",), nodes=nodes, weights=weights,
        ).cpu()
    assert math.isclose(float(total[0]), epsilon**2, rel_tol=0, abs_tol=1e-12)


def test_orthonormal_directions_are_reproducible():
    value = torch.zeros((1, 3, 4, 4))
    first, second = extension.orthonormal_directions(value, 123)
    again_first, again_second = extension.orthonormal_directions(value, 123)
    assert torch.equal(first, again_first)
    assert torch.equal(second, again_second)
    assert abs(float(first.norm()) - 1) < 1e-6
    assert abs(float(second.norm()) - 1) < 1e-6
    assert abs(float((first * second).sum())) < 1e-6


def test_spatial_partition_sums_exactly():
    value = torch.arange(2 * 1 * 3 * 28 * 28, dtype=torch.float64).reshape(2, 1, 3, 28, 28)
    for grid in (4, 7):
        grouped = extension._spatial_sums(value, grid)
        assert grouped.shape == (2, grid, grid)
        assert torch.equal(grouped.sum((1, 2)), value.sum((1, 2, 3, 4)))


def test_source_correlations_closed_schema():
    result = extension.source_correlations(
        {"pefa_R": [1.0, 2.0, 3.0], "random": [3.0, 1.0, 2.0]},
        {"effect": [2.0, 4.0, 6.0]},
    )
    assert math.isclose(result["pefa_R"]["effect"], 1.0)
    assert set(result) == {"pefa_R", "random"}
