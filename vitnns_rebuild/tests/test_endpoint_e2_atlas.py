from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from vitnns.endpoint_e2_atlas import (
    AUTHORIZATION, CLAIM_DECISION, METHOD_ID, display_rgb_signed_sum,
    registered_targets, scan_registered_targets, validate_run_spec,
)

ROOT = Path(__file__).resolve().parents[1]


class TinyPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 4, kernel_size=16, stride=16)

    def forward(self, value):
        return self.proj(value).flatten(2).transpose(1, 2)


class TinyVisionModel(nn.Module):
    num_prefix_tokens = 0

    def __init__(self):
        super().__init__()
        self.patch_embed = TinyPatch()
        self.head = nn.Linear(4, 1000)

    def forward(self, value):
        return self.head(self.patch_embed(value).mean(1))


def test_run_spec_is_closed_to_endpoint_e2():
    value = json.loads((ROOT / "config/nns_vit_endpoint/endpoint_e2_atlas_run_spec.json").read_text())
    validate_run_spec(value)
    assert value["authorization_token"] == AUTHORIZATION
    assert value["claim_decision"] == CLAIM_DECISION
    assert value["method_scope"]["allowed"] == [METHOD_ID]
    assert not any(value["method_scope"][key] for key in (
        "path_ig_allowed", "input_fidelity_allowed", "input_stability_allowed",
        "finite_intervention_allowed", "utility_claim_allowed",
    ))


def test_run_spec_rejects_method_scope_mutation():
    value = json.loads((ROOT / "config/nns_vit_endpoint/endpoint_e2_atlas_run_spec.json").read_text())
    value["method_scope"]["path_ig_allowed"] = True
    with pytest.raises(ValueError, match="METHOD_SCOPE"):
        validate_run_spec(value)


def test_registered_scan_preserves_raw_maps_and_summaries():
    torch.manual_seed(9)
    model = TinyVisionModel().eval()
    rows = [
        {"target_id": "patch.sum.c0", "module": "patch_embed",
         "kind": "patch_channel_sum", "location": "output",
         "channel": 0, "token": -1, "head": -1},
        {"target_id": "class.gt", "module": "", "kind": "ground_truth_logit",
         "location": "output", "channel": -1, "token": -1, "head": -1},
        {"target_id": "class.clean_top1", "module": "",
         "kind": "frozen_clean_top1_logit", "location": "output",
         "channel": -1, "token": -1, "head": -1},
    ]
    targets = registered_targets(rows, ground_truth_class_id=17)
    image = torch.linspace(-1, 1, 3 * 224 * 224, dtype=torch.float32).reshape(1, 3, 224, 224)
    scan = scan_registered_targets(model, image, targets, retain_full_atlas=True)
    assert scan.forward_count == 1 and len(scan.summaries) == 3
    assert scan.class_maps.shape == (2, 3, 224, 224)
    assert scan.atlas_maps is not None and scan.atlas_maps.shape == (3, 3, 224, 224)
    assert scan.summaries[1]["resolved_class_id"] == 17
    assert scan.summaries[2]["resolved_class_id"] == scan.clean_top1_class_id
    for row in scan.summaries:
        assert row["finite"] and np.isfinite(row["absolute_mass"]) and row["absolute_mass"] >= 0


def test_display_is_linear_symmetric_and_zero_safe():
    zero = np.zeros((3, 224, 224), dtype=np.float32)
    image, scale, near_zero = display_rgb_signed_sum(zero)
    assert image.shape == (224, 224, 3) and image.dtype == np.uint8
    assert scale == 0 and near_zero and np.all(image == 255)
    value = zero.copy(); value[0, 0, 0] = 2; value[2, 0, 1] = -2
    image, scale, near_zero = display_rgb_signed_sum(value)
    assert scale == 2 and not near_zero
    assert tuple(image[0, 0]) == (255, 0, 0) and tuple(image[0, 1]) == (0, 0, 255)


def test_execution_amendment_cancels_replay_without_scope_drift():
    value = json.loads((ROOT / "config/nns_vit_endpoint/endpoint_e2_atlas_execution_amendment_r1.json").read_text())
    assert value["stage"] == "ENDPOINT-E2"
    assert value["preserved_formal_evidence"] == {
        "formal_queue_status": "800_OF_800_COMPLETED", "formal_worker_count": 4,
        "image_model_tasks": 800, "target_evaluations": 336800,
        "formal_failures": 0, "formal_retries": 0,
    }
    assert value["execution_change"]["full_single_gpu_replay_required"] is False
    assert value["execution_change"]["partial_replay_use_in_pass_decision"] is False
    assert value["execution_change"]["partial_replay_use_as_scientific_evidence"] is False
    assert value["unchanged"]["method"] == METHOD_ID
    assert not value["unchanged"]["path_ig_allowed"]
    assert not value["unchanged"]["finite_intervention_allowed"]
