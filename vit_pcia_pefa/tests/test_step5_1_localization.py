from __future__ import annotations

import numpy as np
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from vit_pcia_pefa.src.step5_1_localization import (
    audit_head_port_semantics, capture_projection_input, clean_parity, cut_formation_fields, difference_direction,
    endpoint_and_logit, head_component_names, localization_scores, projection_head_wrapper,
    rank_metrics,
)
from vit_pcia_pefa.src.vit_adapter import parity, port_fields


@pytest.fixture
def tiny():
    torch.manual_seed(5101)
    model = VisionTransformer(
        img_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3, mlp_ratio=2,
        num_classes=1000, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
    ).double().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.blocks:
        block.attn.fused_attn = False
    image = torch.randn(1, 3, 8, 8, dtype=torch.float64)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    return model, image, (nodes + 1) / 2, weights / 2


def test_bounded_wrapper_preserves_clean_value_and_jacobian(tiny):
    model, image, _nodes, _weights = tiny
    activation = capture_projection_input(model, image, 1)
    clean, clean_logit, clean_prediction = endpoint_and_logit(model, image, 17)
    clean_fields, names = port_fields(
        model, image.requires_grad_(True), block_index=1, class_id=17,
        requested_names=head_component_names(model, 1),
    )
    with projection_head_wrapper(model, 1, 1, activation, strength=1.0):
        modified_activation = capture_projection_input(model, image, 1)
        modified, modified_logit, modified_prediction = endpoint_and_logit(model, image, 17)
        modified_fields, modified_names = port_fields(
            model, image.requires_grad_(True), block_index=1, class_id=17,
            requested_names=head_component_names(model, 1),
        )
    assert names == modified_names
    assert torch.equal(activation, modified_activation)
    assert clean_logit == modified_logit and clean_prediction == modified_prediction
    assert parity(modified, clean, atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]
    assert parity(modified_fields, clean_fields, atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]


def test_actual_consumer_port_cut_closes_with_wrapper_away_from_clean(tiny):
    model, image, _nodes, _weights = tiny
    activation = capture_projection_input(model, image, 1)
    root = (0.4 * image).requires_grad_(True)
    with projection_head_wrapper(model, 1, 2, activation, strength=0.75):
        fields, names = port_fields(
            model, root, block_index=1, class_id=23,
            requested_names=head_component_names(model, 1),
        )
    assert names == ("identity", "head_0", "head_1", "head_2", "__full__")
    assert parity(fields[:-1].sum(0), fields[-1], atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]


def test_cpu_float64_fixture_localizes_bounded_quadratic(tiny):
    model, image, nodes, weights = tiny
    activation = capture_projection_input(model, image, 1)
    clean_endpoint, clean_logit, clean_prediction = endpoint_and_logit(model, image, 17)
    clean = cut_formation_fields(
        model, image, block_index=1, class_id=17, nodes=nodes, weights=weights,
    )
    with projection_head_wrapper(model, 1, 1, activation, strength=1.0):
        modified_activation = capture_projection_input(model, image, 1)
        modified_endpoint, modified_logit, modified_prediction = endpoint_and_logit(model, image, 17)
        modified = cut_formation_fields(
            model, image, block_index=1, class_id=17, nodes=nodes, weights=weights,
        )
    assert clean_parity(
        clean, modified, clean_endpoint=clean_endpoint, modified_endpoint=modified_endpoint,
        clean_logit=clean_logit, modified_logit=modified_logit,
        clean_prediction=clean_prediction, modified_prediction=modified_prediction,
        atol=1e-9, rtol=1e-7, near_zero=1e-12,
    )["pass"]
    methods = localization_scores(
        clean, modified, clean_activation=activation, modified_activation=modified_activation,
        random_namespace="fixture",
    )
    assert rank_metrics(methods["pefa_R_delta"], 1, 1)["top1"]
    direction = difference_direction(clean, modified)
    assert direction["pass"] and direction["cosine"] > 0.999999


def test_zero_and_linear_tangent_controls_have_no_false_signal(tiny):
    model, image, nodes, weights = tiny
    activation = capture_projection_input(model, image, 0)
    clean = cut_formation_fields(
        model, image, block_index=0, class_id=31, nodes=nodes, weights=weights,
    )
    for mode, strength in (("bounded_quadratic", 0.0), ("linear_tangent", 1.0)):
        with projection_head_wrapper(model, 0, 0, activation, strength=strength, mode=mode):
            modified_activation = capture_projection_input(model, image, 0)
            modified = cut_formation_fields(
                model, image, block_index=0, class_id=31, nodes=nodes, weights=weights,
            )
        methods = localization_scores(
            clean, modified, clean_activation=activation, modified_activation=modified_activation,
            random_namespace=mode,
        )
        for key in ("pefa_R_delta", "pcia_delta", "endpoint_delta",
                    "gradient_delta", "activation_delta"):
            assert max(methods[key]) == 0.0


def test_rank_metrics_use_conservative_ties():
    assert rank_metrics([1.0, 4.0, 2.0], 1, 1)["top1"]
    tied = rank_metrics([3.0, 3.0, 1.0], 0, 1)
    assert tied["rank"] == 2 and not tied["top1"]


def test_held_out_head_slice_rotation_is_detected_and_localized():
    def port(slice_spec):
        return {"consumer_operator_id": "blocks.5.attn.proj", "input_slot": 0,
                "slice_spec": slice_spec}
    valid = {"cut_id": "HEADS_WITH_BYPASS_FULL", "block_index": 5,
             "port_ids": [port("ALL"), port("HEAD[0]"), port("HEAD[1]"), port("HEAD[2]")]}
    valid["port_ids"][0]["consumer_operator_id"] = "blocks.5.add1"
    assert audit_head_port_semantics(valid)["pass"]
    mutated = {**valid, "port_ids": [dict(row) for row in valid["port_ids"]]}
    slices = [row["slice_spec"] for row in mutated["port_ids"][1:]]
    for row, rotated in zip(mutated["port_ids"][1:], slices[1:] + slices[:1]): row["slice_spec"] = rotated
    result = audit_head_port_semantics(mutated)
    assert not result["pass"] and result["block_index"] == 5
    assert result["cut_id"] == "HEADS_WITH_BYPASS_FULL" and result["category"] == "head"
    assert {row["code"] for row in result["errors"]} == {"FAIL_PORT_ID_SEMANTIC_MISMATCH"}
