from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from vitnns.endpoint_e4_intervention import (
    EndpointE4Error, capture_registered_edges, component_rows, component_tensor,
    directional_edges, edge_gradients, intervention_scores, scan_e4_task, spearman,
    validate_run_spec,
)


REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "config/nns_vit_endpoint/endpoint_e4_intervention_run_spec.json"


def tiny_model() -> VisionTransformer:
    torch.manual_seed(418)
    model = VisionTransformer(
        img_size=224, patch_size=224, embed_dim=12, depth=12, num_heads=3,
        mlp_ratio=2, num_classes=1000,
    ).float().eval()
    for block in model.blocks:
        block.attn.fused_attn = False
    return model


def test_run_spec_is_closed_and_rejects_scope_drift():
    value = json.loads(SPEC.read_text())
    validate_run_spec(value)
    drift = deepcopy(value); drift["method_scope"]["path_ig_allowed"] = True
    with pytest.raises(EndpointE4Error, match="METHOD_SCOPE"):
        validate_run_spec(drift)
    drift = deepcopy(value); drift["local_contract"]["eta"][-1] = 0.0001
    with pytest.raises(EndpointE4Error, match="LOCAL_CONTRACT"):
        validate_run_spec(drift)


def test_component_registry_has_all_frozen_families():
    main = component_rows(3)
    full = component_rows(3, include_secondary=True)
    assert len(main) == 84 and len(full) == 120
    assert len({row["component_id"] for row in full}) == 120
    assert {row["family"] for row in main} == {"heads", "routing_value", "residual_branches"}


def test_capture_gradient_direction_and_noop_replacement():
    model = tiny_model()
    image = torch.randn(1, 3, 224, 224)
    capture = capture_registered_edges(model, image, class_id=7, enable_grad=True)
    gradients = edge_gradients(capture)
    directions = directional_edges(model, image, class_id=7)
    row = next(row for row in component_rows(3) if row["component_id"] == "b00.head_01")
    clean = component_tensor(capture.edges[0], row).detach()
    direction = component_tensor(directions[0], row)
    gradient = component_tensor(gradients[0], row)
    predicted = (gradient.double() * direction.double()).sum()
    assert torch.isfinite(predicted)
    clean_score, values, _ = intervention_scores(
        model, image, class_id=7, candidates=[row], replacements={row["component_id"]: clean},
    )
    assert values[0] == pytest.approx(clean_score, abs=2e-5, rel=2e-5)


def test_complete_tiny_scan_has_finite_utility_metrics():
    model = tiny_model()
    image = torch.randn(1, 3, 224, 224)
    result = scan_e4_task(
        model, image, {"normalized_zero": torch.zeros_like(image), "blur_rgb": image * 0.8},
        class_id=9, task_seed=1234, include_local=False, include_secondary=False, chunk_size=16,
    )
    assert result["main_component_count"] == 84
    assert result["all_component_count"] == 84
    assert result["local_contract"] == {"performed": False, "components": [], "all_pass": None}
    for reference in result["finite_utility"].values():
        assert len(reference["components"]) == 84
        assert set(reference["per_image_metrics"]) == {"heads", "routing_value", "residual_branches"}
        assert all(row["reason"] in {None, "NA_CONSTANT_VECTOR"} for family in reference["per_image_metrics"].values() for key, row in family.items() if key.endswith("effect"))


def test_average_tie_spearman_and_constant_na():
    assert spearman(list(range(8)), list(range(8)))["value"] == pytest.approx(1.0)
    assert spearman([1.0] * 8, list(range(8)))["reason"] == "NA_CONSTANT_VECTOR"
