from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from vitnns.endpoint_e5_spatial import (
    EndpointE5Error, candidate_mask_digests, normalized_auc, random_region_order,
    ranked_regions, region_scores, scan_e5_task, validate_run_spec,
)

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "config/nns_vit_endpoint/endpoint_e5_spatial_run_spec.json"


def tiny_model() -> VisionTransformer:
    torch.manual_seed(519)
    model = VisionTransformer(img_size=224, patch_size=224, embed_dim=12, depth=12,
                              num_heads=6, mlp_ratio=2, num_classes=1000).float().eval()
    for block in model.blocks:
        block.attn.fused_attn = False
    return model


def test_run_spec_is_closed_and_rejects_scope_drift():
    value = json.loads(SPEC.read_text()); validate_run_spec(value)
    drift = deepcopy(value); drift["method_scope"]["holm_allowed"] = True
    with pytest.raises(EndpointE5Error, match="METHOD_SCOPE"):
        validate_run_spec(drift)
    drift = deepcopy(value); drift["spatial_intervention"]["budget_counts"] = [0, 1, 3, 4]
    with pytest.raises(EndpointE5Error, match="SPATIAL_CONTRACT"):
        validate_run_spec(drift)


def test_region_mass_ties_masks_auc_and_random_are_exact():
    value = torch.zeros(1, 3, 224, 224); value[:, :, :56, :56] = -2; value[:, :, 56:112, :56] = 1
    scores = region_scores(value)
    assert scores[0] == 3 * 56 * 56 * 2 and scores[4] == 3 * 56 * 56
    assert ranked_regions(scores)[:3] == [0, 4, 1]
    assert len(candidate_mask_digests()) == len(set(candidate_mask_digests())) == 16
    assert normalized_auc([0, 1, 2, 4]) == pytest.approx(2.0)
    first = random_region_order(17, "deit_s16", "blur_rgb", 3, 7)
    assert first == random_region_order(17, "deit_s16", "blur_rgb", 3, 7)
    assert sorted(first[1]) == list(range(16))


def test_tiny_scan_has_complete_curves_and_no_inference():
    model = tiny_model(); image = torch.randn(1, 3, 224, 224)
    result = scan_e5_task(model, image, {"normalized_zero": torch.zeros_like(image),
        "blur_rgb": image * 0.8}, class_id=9, task_seed=1234, model_id="deit_s16",
        perturbed_batch_size=128, isolation_batch_size=64)
    assert result["block"] == 5 and result["heads"] == [0, 3, 5]
    assert result["counts"] == {"candidate_region_count": 16, "reference_count": 2,
        "selected_head_count": 3, "curve_count": 144, "random_curve_count": 120,
        "candidate_taylor_forward_evaluations": 32, "finite_union_full_forward_curve_points": 432,
        "finite_union_isolated_head_evaluations": 432}
    for reference in result["references"].values():
        assert len(reference["curves"]) == 72
        assert set(reference["image_method_scores"]) == {"endpoint_component", "total_gxi", "raw_gradient", "regional_taylor", "random_regions"}
        assert len(reference["paired_endpoint_minus_baseline"]) == 4
        assert all(len(row["points"]) == 4 for row in reference["curves"])
