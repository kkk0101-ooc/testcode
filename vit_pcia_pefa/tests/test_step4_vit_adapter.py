from __future__ import annotations

import numpy as np
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from vit_pcia_pefa.src.vit_adapter import (
    VitAdapterError, parity, port_fields, port_layout, q_fields,
)


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
    return model, image


def _indexes(names):
    return {name: index for index, name in enumerate(names)}


def test_all_registered_cuts_close_on_unaggregated_coordinates(tiny):
    model, image = tiny
    fields, names = port_fields(model, image.requires_grad_(True), block_index=0, class_id=17)
    indexes = _indexes(names)
    layout = port_layout(3)
    assert tuple(names[:-1]) == layout.component_ids
    for keys in layout.cuts.values():
        reconstructed = fields[[indexes[key] for key in keys]].sum(0)
        assert parity(reconstructed, fields[-1], atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]


def test_q_jvp_equals_direct_plus_curvature_jvp(tiny):
    model, image = tiny
    root = (0.37 * image).requires_grad_(True)
    direction = image * torch.tensor(
        np.indices(image.shape[-2:]).sum(0) % 2 == 0, dtype=image.dtype,
    ).reshape(1, 1, 8, 8)
    fields, jvp_d = torch.autograd.functional.jvp(
        lambda value: port_fields(model, value, block_index=1, class_id=23)[0],
        root, direction, strict=True,
    )
    _q, jvp_q = torch.autograd.functional.jvp(
        lambda value: q_fields(model, value, block_index=1, class_id=23)[0],
        root, direction, strict=True,
    )
    direct = direction.unsqueeze(0) * fields
    curvature = root.unsqueeze(0) * jvp_d
    assert parity(jvp_q, direct + curvature, atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]


def test_jvp_and_vjp_show_full_cut_omega_zero(tiny):
    model, image = tiny
    layout = port_layout(3)
    indexes = _indexes((*layout.component_ids, "__full__"))
    direction = torch.randn_like(image)
    for keys in layout.cuts.values():
        selected = [indexes[key] for key in keys]
        def cut_field(root):
            fields, _ = port_fields(model, root, block_index=0, class_id=31)
            return fields[selected].sum(0)
        root = image.detach().requires_grad_(True)
        _out, jvp = torch.autograd.functional.jvp(cut_field, root, direction, strict=True)
        _out, vjp = torch.autograd.functional.vjp(cut_field, root, v=direction, strict=True)
        assert parity(vjp, jvp, atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]


def test_applicable_negative_controls_fail(tiny):
    model, image = tiny
    fields, names = port_fields(model, image.requires_grad_(True), block_index=1, class_id=47)
    indexes = _indexes(names)
    full = fields[-1]
    wrong = {
        "omit_identity": fields[indexes["attention"]],
        "duplicate_identity": fields[[indexes["identity"], indexes["attention"], indexes["identity"]]].sum(0),
        "omit_head": fields[[indexes["identity"], indexes["head_0"], indexes["head_1"]]].sum(0),
        "swap_input_slot": fields[indexes["identity"]] - fields[indexes["attention"]],
        "duplicate_serial_mlp": fields[[indexes["carry"], indexes["mlp_fc1"], indexes["mlp_gelu"]]].sum(0),
    }
    for observed in wrong.values():
        assert not parity(observed, full, atol=1e-9, rtol=1e-7, near_zero=1e-12)["pass"]


def test_fused_attention_is_rejected(tiny):
    model, image = tiny
    model.blocks[0].attn.fused_attn = True
    with pytest.raises(VitAdapterError, match="STRUCTURAL_BACKEND_REQUIRES_UNFUSED_ATTENTION"):
        port_fields(model, image.requires_grad_(True), block_index=0, class_id=0)


def test_port_identity_contains_required_fields():
    identity = {
        "model_id": "tiny", "graph_revision": "test", "block_index": 0,
        "producer_tensor_id": "blocks.0.input#0",
        "consumer_operator_id": "blocks.0.add1", "input_slot": 0,
        "invocation_index": 0, "slice_spec": "ALL",
    }
    assert set(identity) == {
        "model_id", "graph_revision", "block_index", "producer_tensor_id",
        "consumer_operator_id", "input_slot", "invocation_index", "slice_spec",
    }
