import math

import torch

from vitnns.extractor import NativeViTExtractor, recompute_attention, split_fused_qkv
from vitnns.modeling import load_canonical_model


def test_qkv_marker_order():
    projected = torch.tensor([[[1.0, 2.0, 11.0, 12.0, 21.0, 22.0]]])
    q, k, v = split_fused_qkv(projected, num_heads=1, head_dim=2)
    assert torch.equal(q, torch.tensor([[[[1.0, 2.0]]]]))
    assert torch.equal(k, torch.tensor([[[[11.0, 12.0]]]]))
    assert torch.equal(v, torch.tensor([[[[21.0, 22.0]]]]))


def test_scale_uses_head_dimension(state_dict_path):
    model = load_canonical_model(state_dict_path, unfused_attention=True)
    scale = model.blocks[0].attn.scale
    assert scale == 64 ** -0.5 == 0.125
    assert scale != 768 ** -0.5


@torch.inference_mode()
def test_real_attention_recomputation(state_dict_path):
    torch.manual_seed(7)
    model = load_canonical_model(state_dict_path, unfused_attention=True)
    result = NativeViTExtractor(model).extract(torch.randn(1, 3, 224, 224))
    assert len(result.layers) == 12
    for layer in result.layers:
        assert layer.q.shape == layer.k.shape == layer.v.shape == (1, 12, 197, 64)
        assert layer.attention.shape == (1, 12, 197, 197)
        assert layer.attention_max_abs_error < 1e-6
        assert torch.all(layer.attention >= 0)
        torch.testing.assert_close(
            layer.attention.sum(dim=-1), torch.ones_like(layer.attention.sum(dim=-1)),
            rtol=0, atol=1e-6,
        )

