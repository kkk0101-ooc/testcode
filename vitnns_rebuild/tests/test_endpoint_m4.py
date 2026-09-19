import math

import torch
from timm.models.vision_transformer import VisionTransformer

from vitnns.endpoint_m4 import scan_endpoint_maps, attention_jvp_diagnostics


def tiny_model():
    torch.manual_seed(190919)
    model = VisionTransformer(
        img_size=224, patch_size=224, embed_dim=12, depth=12, num_heads=3,
        mlp_ratio=2, num_classes=1000,
    ).float().eval()
    for block in model.blocks:
        block.attn.fused_attn = False
    return model


def test_m4_attention_residual_closure_detach_and_na_contracts():
    torch.set_num_threads(2)
    model = tiny_model()
    image = torch.randn(1, 3, 224, 224)
    result = scan_endpoint_maps(model, image, class_id=17, chunk_size=3, detach_validation=True)
    assert len(result["attention"]) == 36
    assert len(result["residual"]) == 24
    assert len(result["attention_detach_validation"]) == 12
    assert len(result["block_validation"]) == 12
    for row in result["attention"]:
        assert row["qkv_native_closure"]["pass"]
        assert row["frozen_attention_identity"]["pass"]
        assert row["routing_share"] is None or 0 <= row["routing_share"] <= 1
        assert 0 <= row["mean_query_saturation"] <= 1
    for row in result["residual"]:
        assert row["closure"]["relative_l2"] < 2e-4
        assert row["negative_control"]["passed"]
        if row["identity_share"] is not None:
            assert math.isclose(row["identity_share"] + row["branch_share"], 1.0)
    for row in result["block_validation"]:
        assert row["qk_routing_closure"]["pass"]
        assert row["routing_value_attention_closure"]["pass"]
    assert all(row["attention_detach_value_gradient"]["pass"]
               for row in result["attention_detach_validation"])


def test_m4_eight_direction_explicit_jvp():
    torch.set_num_threads(2)
    model = tiny_model()
    image = torch.randn(1, 3, 224, 224)
    rows = attention_jvp_diagnostics(model, image, task_id="tiny", directions=8)
    assert len(rows) == 36
    assert all(len(row["direction_records"]) == 8 for row in rows)
    assert max(row["explicit_jvp_relative_error"] for row in rows) < 2e-4
