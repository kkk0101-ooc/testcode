from __future__ import annotations

import numpy as np
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from vit_pcia_pefa.src.step5_1_localization import (
    cut_formation_fields, endpoint_and_logit, head_component_names,
)
from vit_pcia_pefa.src.step5_2_prediction import (
    attention_head_level_norms, bootstrap_mean_interval, capture_attention_context, clean_prediction_scores, convergence_check,
    intervention_outcome, intervention_parity, natural_head_linearization, prediction_metrics,
)
from vit_pcia_pefa.src.vit_adapter import parity, port_fields


@pytest.fixture
def tiny():
    torch.manual_seed(5201)
    model = VisionTransformer(
        img_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3, mlp_ratio=2,
        num_classes=1000, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
    ).double().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.blocks:
        block.attn.fused_attn = False
    image = torch.randn(1, 3, 8, 8, dtype=torch.float64)
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(24)
    return model, image, (raw_nodes + 1) / 2, raw_weights / 2


def test_natural_head_tangent_preserves_clean_value_and_jacobian(tiny):
    model, image, _nodes, _weights = tiny
    clean_input, clean_heads = capture_attention_context(model, image, 1)
    clean_endpoint, clean_logit, clean_prediction = endpoint_and_logit(model, image, 17)
    requested = head_component_names(model, 1)
    clean_fields, names = port_fields(
        model, image.requires_grad_(True), block_index=1, class_id=17, requested_names=requested,
    )
    with natural_head_linearization(model, 1, 1, clean_input, strength=0.0):
        changed_input, changed_heads = capture_attention_context(model, image, 1)
        changed_endpoint, changed_logit, changed_prediction = endpoint_and_logit(model, image, 17)
        changed_fields, changed_names = port_fields(
            model, image.requires_grad_(True), block_index=1, class_id=17, requested_names=requested,
        )
    assert names == changed_names and torch.equal(clean_input, changed_input)
    assert parity(clean_heads, changed_heads, atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]
    assert clean_logit == pytest.approx(changed_logit, abs=1e-12) and clean_prediction == changed_prediction
    assert parity(changed_endpoint, clean_endpoint, atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]
    assert parity(changed_fields, clean_fields, atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]


def test_natural_head_linearization_changes_path_but_keeps_cut_valid(tiny):
    model, image, nodes, weights = tiny
    clean_input, clean_activation = capture_attention_context(model, image, 1)
    clean_endpoint, clean_logit, clean_prediction = endpoint_and_logit(model, image, 17)
    clean = cut_formation_fields(model, image, block_index=1, class_id=17, nodes=nodes, weights=weights)
    with natural_head_linearization(model, 1, 1, clean_input, strength=0.0):
        changed_endpoint, changed_logit, changed_prediction = endpoint_and_logit(model, image, 17)
        changed = cut_formation_fields(model, image, block_index=1, class_id=17, nodes=nodes, weights=weights)
    assert intervention_parity(clean, changed, clean_endpoint=clean_endpoint, modified_endpoint=changed_endpoint,
        clean_logit=clean_logit, modified_logit=changed_logit, clean_prediction=clean_prediction,
        modified_prediction=changed_prediction, atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]
    assert parity(changed["R"][:-1].sum(0), changed["R"][-1], atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]
    outcome = intervention_outcome(clean, changed, head_index=1, epsilon=1e-12)
    assert outcome["Y"] > 0 and torch.isfinite(outcome["delta_G"]).all()
    activation_norms, gradient_norms = attention_head_level_norms(model, image, 1, 17)
    scores = clean_prediction_scores(clean, activation_norms=activation_norms,
                                      gradient_norms=gradient_norms, random_namespace="test")
    assert set(scores) == {"pefa_R", "pcia_A", "endpoint_Q", "gradient_norm", "activation_norm", "random"}
    assert all(len(values) == 3 for values in scores.values())


def test_metrics_ties_and_convergence_are_explicit(tiny):
    model, image, nodes, weights = tiny
    clean = cut_formation_fields(model, image, block_index=1, class_id=17, nodes=nodes, weights=weights)
    assert convergence_check(clean, clean, atol=1e-12, rtol=1e-12, near_zero=1e-15)["pass"]
    metrics = prediction_metrics([3.0, 3.0, 1.0], [2.0, 1.0, 4.0], 1)
    assert metrics["predicted_top1"] == 0 and metrics["actual_top1"] == 2
    assert not metrics["top1_hit"] and metrics["spearman"] is not None
    interval = bootstrap_mean_interval([1.0, 2.0, 3.0], resamples=1000, confidence=0.95, seed=52)
    assert interval["mean"] == 2.0 and interval["lower"] <= 2.0 <= interval["upper"]


def test_global_map_convergence_rule_retains_elementwise_diagnostic():
    names = ("identity", "head_0", "__full__")
    fine_tensor = torch.tensor([[[[0.0, 1.0, -1.0]]]], dtype=torch.float64)
    coarse_tensor = fine_tensor + torch.tensor([[[[1e-3, 0.0, 0.0]]]], dtype=torch.float64)
    fine = {"names": names, "R": torch.stack((fine_tensor, fine_tensor, fine_tensor)),
            "A": torch.stack((fine_tensor, fine_tensor, fine_tensor))}
    coarse = {"names": names, "R": torch.stack((coarse_tensor, coarse_tensor, coarse_tensor)),
              "A": torch.stack((coarse_tensor, coarse_tensor, coarse_tensor))}
    strict = convergence_check(coarse, fine, atol=1e-4, rtol=1e-2, near_zero=1e-6)
    global_l2 = convergence_check(coarse, fine, atol=1e-4, rtol=1e-2, near_zero=1e-6,
                                  rule="global_relative_l2_or_absolute_near_zero")
    assert not strict["pass"]
    assert global_l2["pass"]
    assert not global_l2["checks"]["G"]["elementwise_pass"]
    assert global_l2["checks"]["G"]["gate_pass"]
