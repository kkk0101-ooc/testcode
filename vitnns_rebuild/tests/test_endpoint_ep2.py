from __future__ import annotations

import pytest
import torch
from torch import nn
import json
from pathlib import Path

from vitnns.endpoint_ep2 import compare_pair, integrate_logit, integrate_logits, select_model_M

ROOT = Path(__file__).resolve().parents[1]


def test_ep2_run_spec_is_closed_to_calibration_scope():
    value = json.loads((ROOT / "config/nns_vit_endpoint/ep2_run_spec.json").read_text())
    assert value["stage"] == "EP2" and value["authorization_text"] == "继续"
    assert value["cohort"] == {
        "split": "calibration", "count": 64, "models_share_rows": True,
        "evaluation_access": False, "discovery_access": False,
        "replacement_allowed": False,
    }
    assert value["targets"] == ["ground_truth_logit", "frozen_clean_top1_logit"]
    assert value["ig"]["candidate_nodes"] == [32, 64, 128, 256, 512]
    assert value["ig"]["check_nodes"] == [64, 128, 256, 512, 1024]
    assert value["runtime"]["gpu_hours_cap"] == 16
    assert value["failure"]["no_candidate_pass"] == "EP2_NUMERICAL_CALIBRATION_FAILED"


class SmoothClassifier(nn.Module):
    def forward(self, value):
        score = torch.sin(value).flatten(1).sum(1)
        return torch.stack((score, -score), dim=1)


def test_gauss_legendre_ig_and_pair_metric():
    image = torch.linspace(-0.7, 0.9, 12, dtype=torch.float32).reshape(1, 3, 2, 2)
    model = SmoothClassifier()
    lower = integrate_logit(model, image, 0, steps=32, alpha_chunk=8)
    upper = integrate_logit(model, image, 0, steps=64, alpha_chunk=8)
    row = compare_pair(lower, upper)
    assert lower["finite"] and upper["finite"] and row["pair_pass"]
    assert row["map_relative_l2"] <= 0.02


def test_near_zero_completeness_branch():
    zero = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    result = integrate_logit(SmoothClassifier(), zero, 0, steps=32, alpha_chunk=8)
    row = compare_pair(result, result)
    assert row["completeness_branch"] == "absolute_near_zero" and row["pair_pass"]


def test_two_target_integration_matches_independent_calls():
    image = torch.linspace(-0.4, 0.8, 12, dtype=torch.float32).reshape(1, 3, 2, 2)
    model = SmoothClassifier()
    joint = integrate_logits(model, image, [0, 1], steps=32, alpha_chunk=8)
    for class_id in (0, 1):
        separate = integrate_logit(model, image, class_id, steps=32, alpha_chunk=8)
        torch.testing.assert_close(joint[class_id]["map"], separate["map"], atol=0, rtol=0)


def test_model_selection_is_smallest_jointly_passing_candidate():
    rows = []
    for target in ("ground_truth_logit", "frozen_clean_top1_logit"):
        for candidate in (32, 64, 128, 256, 512):
            for index in range(64):
                rows.append({
                    "target_kind": target, "M": candidate, "finite": True,
                    "pair_pass": candidate >= 64 and index < 61,
                })
    selected = select_model_M(rows)
    assert selected["status"] == "PASS" and selected["selected_M"] == 64


def test_model_selection_fails_on_missing_or_no_candidate():
    with pytest.raises(ValueError, match="CALIBRATION_COVERAGE"):
        select_model_M([])
    rows = [
        {"target_kind": target, "M": candidate, "finite": True, "pair_pass": False}
        for target in ("ground_truth_logit", "frozen_clean_top1_logit")
        for candidate in (32, 64, 128, 256, 512) for _ in range(64)
    ]
    assert select_model_M(rows)["status"] == "EP2_NUMERICAL_CALIBRATION_FAILED"
