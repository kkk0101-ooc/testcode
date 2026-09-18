from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from vitnns.endpoint_e3_closure import (
    EndpointE3Error, scan_structural_closure, validate_run_spec,
)


REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "config/nns_vit_endpoint/endpoint_e3_closure_run_spec.json"


def test_run_spec_is_closed_and_rejects_scope_drift():
    value = json.loads(SPEC.read_text())
    validate_run_spec(value)
    drift = deepcopy(value)
    drift["method_scope"]["activation_replacement_allowed"] = True
    with pytest.raises(EndpointE3Error, match="METHOD_SCOPE"):
        validate_run_spec(drift)
    drift = deepcopy(value)
    drift["numeric"]["rtol"] = 1e-3
    with pytest.raises(EndpointE3Error, match="NUMERIC_SCOPE"):
        validate_run_spec(drift)


def test_all_block_closure_and_negative_controls_on_tiny_model():
    torch.manual_seed(1909)
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=12, depth=12, num_heads=3,
        mlp_ratio=2, num_classes=1000,
    ).float().eval()
    for block in model.blocks:
        block.attn.fused_attn = False
    image = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    result = scan_structural_closure(
        model, image, ground_truth_class_id=17, retain_raw=True,
    )
    assert len(result.blocks) == 12
    assert result.raw_blocks is not None and len(result.raw_blocks) == 12
    assert result.raw_component_orders is not None
    for row, raw, order in zip(result.blocks, result.raw_blocks, result.raw_component_orders):
        assert len(row["cuts"]) == 8 and all(item["pass"] for item in row["cuts"])
        assert len(row["negative_controls"]) == 3
        assert all(item["passed"] for item in row["negative_controls"])
        assert raw.shape == (len(order), 3, 224, 224)
        assert torch.from_numpy(raw).isfinite().all()


def test_requires_frozen_production_shape_dtype_blocks_and_tolerance():
    model = VisionTransformer(
        img_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3,
        mlp_ratio=2, num_classes=1000,
    ).float().eval()
    image = torch.randn(1, 3, 8, 8)
    with pytest.raises(EndpointE3Error, match="INPUT_SHAPE_OR_DTYPE"):
        scan_structural_closure(model, image, ground_truth_class_id=0)
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=12, depth=12, num_heads=3,
        mlp_ratio=2, num_classes=1000,
    ).float().eval()
    image = torch.randn(1, 3, 224, 224)
    with pytest.raises(EndpointE3Error, match="FLOAT32_TOLERANCE_DRIFT"):
        scan_structural_closure(model, image, ground_truth_class_id=0, rtol=1e-3)
