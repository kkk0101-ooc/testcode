from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from vit_pcia_pefa.src.pilot import (
    ablate_projection_head, closure_summary, formation_fields,
    integrated_hessians_groups, integrated_hessians_scores, omega_directional_analysis,
    omega_complete_cut_audit,
    path_integral_ports, port_categories, quadrant_masks, representation_summary, spearman,
    stable_random_scores,
)
from vit_pcia_pefa.src.vit_adapter import port_fields


@pytest.fixture
def tiny():
    torch.manual_seed(20260921)
    model = VisionTransformer(
        img_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3,
        mlp_ratio=2, num_classes=1000, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ).double().eval()
    for block in model.blocks:
        block.attn.fused_attn = False
    image = torch.randn(1, 3, 8, 8, dtype=torch.float64)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    return model, image, (nodes + 1) / 2, weights / 2


def test_quadrants_are_an_exact_disjoint_partition():
    value = torch.randn(1, 3, 8, 8)
    masks = quadrant_masks(value)
    assert len(masks) == 4
    assert torch.equal(sum((value * mask for mask in masks), torch.zeros_like(value)), value)
    assert all(int(mask.sum()) == 3 * 4 * 4 for mask in masks)


def test_piecewise_path_closes_and_preserves_registered_ports(tiny):
    model, image, nodes, weights = tiny
    baseline = torch.zeros_like(image)
    masks = quadrant_masks(image)
    waypoint = baseline + image * (masks[0] | masks[3])
    fields = formation_fields(
        model, image=image, baseline=baseline, waypoints=(waypoint,),
        block_index=0, class_id=17, nodes=nodes, weights=weights,
    )
    assert fields["A"].shape == fields["T"].shape == fields["R"].shape
    assert torch.allclose(fields["T"], fields["A"] + fields["R"], atol=1e-15, rtol=1e-15)
    closure = closure_summary(model, 0, fields, "QKV_WITH_BYPASS_FULL")
    assert closure["pass"]
    assert set(closure["checks"]) == {
        "port_D_to_full_gradient", "pcia_to_ig", "pefa_to_endpoint",
        "T_equals_A_plus_R", "endpoint_minus_ig_equals_R",
    }


def test_path_segment_uses_end_minus_start_not_absolute_endpoint(tiny):
    model, image, nodes, weights = tiny
    midpoint = image * 0.25
    first, names = path_integral_ports(
        model, start=torch.zeros_like(image), end=midpoint, block_index=1,
        class_id=19, nodes=nodes, weights=weights,
    )
    second, second_names = path_integral_ports(
        model, start=midpoint, end=image, block_index=1, class_id=19,
        nodes=nodes, weights=weights,
    )
    assert names == second_names
    assert torch.isfinite(first + second).all()


def test_representation_has_frozen_categories_and_finite_masses(tiny):
    model, image, nodes, weights = tiny
    fields = formation_fields(
        model, image=image, baseline=torch.zeros_like(image), waypoints=(),
        block_index=0, class_id=23, nodes=nodes, weights=weights,
    )
    categories = port_categories(fields["names"])
    assert set(categories) == {
        "residual", "attention_branch", "query", "key", "value",
        "head", "projection", "mlp",
    }
    summary = representation_summary(fields["R"], fields["names"])
    assert summary["overall"]["finite"]
    assert 0 <= summary["routing_concentration_hhi"] <= 1


class BilinearFourGroup(nn.Module):
    def forward(self, value):
        masks = quadrant_masks(value)
        first = (value * masks[0]).flatten(1).sum(1)
        second = (value * masks[1]).flatten(1).sum(1)
        score = first * second
        return torch.stack((score, torch.zeros_like(score)), dim=1)


def test_group_integrated_hessians_matches_bilinear_recursive_ig():
    model = BilinearFourGroup().double()
    image = torch.zeros(1, 3, 4, 4, dtype=torch.float64)
    masks = quadrant_masks(image)
    image[masks[0]] = 1 / int(masks[0].sum())
    image[masks[1]] = 2 / int(masks[1].sum())
    nodes, weights = np.polynomial.legendre.leggauss(4)
    matrix = integrated_hessians_groups(
        model, image=image, baseline=torch.zeros_like(image), class_id=0,
        nodes=(nodes + 1) / 2, weights=weights / 2, masks=masks,
    )
    expected = torch.zeros((4, 4), dtype=torch.float64)
    expected[:2, :2] = 0.5
    assert torch.allclose(matrix, expected, atol=1e-12, rtol=1e-12)
    assert integrated_hessians_scores(matrix) == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_head_ablation_is_scoped_and_restores_forward(tiny):
    model, image, _nodes, _weights = tiny
    clean = model(image).detach()
    with ablate_projection_head(model, 0, 1):
        changed = model(image).detach()
    restored = model(image).detach()
    assert not torch.equal(clean, changed)
    assert torch.equal(clean, restored)


def test_rank_statistics_and_random_baseline_are_deterministic():
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1, 1], [4, 3, 2, 1]) is None
    assert stable_random_scores("task", 4) == stable_random_scores("task", 4)
    assert stable_random_scores("task", 4) != stable_random_scores("other", 4)


def test_requested_port_subset_matches_full_and_streamed_omega_is_finite(tiny):
    model, image, _nodes, _weights = tiny
    full, names = port_fields(model, image.requires_grad_(True), block_index=0, class_id=29)
    requested = ("identity", "query_head_1", "mlp_fc2")
    subset, subset_names = port_fields(
        model, image.detach().requires_grad_(True), block_index=0, class_id=29,
        requested_names=requested,
    )
    expected_names = tuple(name for name in names if name in requested)
    expected = full[[names.index(name) for name in expected_names]]
    assert subset_names == expected_names
    assert torch.allclose(subset, expected, atol=1e-12, rtol=1e-12)
    omega = omega_directional_analysis(
        model, image, block_index=0, class_id=29, component_names=names[:-1],
        cut_component_names=("identity", "head_0", "head_1", "head_2"), seed=20260921,
    )
    assert omega["representation"]["overall"]["finite"]
    assert set(omega["scores_l2"]) == set(names[:-1])
    assert omega["complete_cut_zero_sum"]["pass"]


def test_float64_coarse_complete_cut_omega_audit(tiny):
    model, image, _nodes, _weights = tiny
    declared = ("identity", "head_0", "head_1", "head_2")
    audit = omega_complete_cut_audit(
        model, image, block_index=0, class_id=31, seed=20260921,
        declared_cut_component_names=declared,
    )
    assert audit["dtype"] == "float64"
    assert audit["evaluation_component_ids"] == ["identity", "attention"]
    assert audit["pass"]
    with pytest.raises(ValueError, match="OMEGA_AUDIT_REQUIRES_FLOAT64"):
        omega_complete_cut_audit(
            model.float(), image.float(), block_index=0, class_id=31, seed=20260921,
            declared_cut_component_names=declared,
        )
