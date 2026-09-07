from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from vitnns.display import DisplayBoundaryError, DisplayTransform, metric_ready
from vitnns.stage_f_renderer import (
    EXPECTED_D10_RECIPE_SHA256, EXPECTED_E4_RECIPE_SHA256,
    StageFRendererError, assert_f0_source_boundary, load_contracts,
    load_fixture, validate_font, verify_preconditions,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests/fixtures/stage_f/golden_maps.json"


def test_display_only_smoothed_or_resized_map_cannot_reenter_metric():
    displayed = DisplayTransform(gaussian_sigma=0.5, resize=(8, 8))(metric_ready(np.ones((4, 4), dtype=np.float32)))
    with pytest.raises(DisplayBoundaryError, match="cannot relabel display"):
        metric_ready(displayed)


def test_f0_has_no_model_torch_timm_cuda_or_jpeg_surface():
    result = assert_f0_source_boundary(REPO)
    assert result == {"status": "PASS", "model_access": False, "torch_access": False, "timm_access": False, "cuda_access": False, "jpeg_access": False}
    with pytest.raises(StageFRendererError, match="versioned JSON"):
        load_fixture(REPO / "assets/data/ILSVRC2012_val/example.JPEG")


def test_formal_or_reference_derived_fixture_is_rejected():
    source = json.loads(FIXTURE.read_text()); source["formal_scientific_figure"] = True
    path = FIXTURE.parent / "_formal_attack.json"; path.write_text(json.dumps(source))
    try:
        with pytest.raises(StageFRendererError): load_fixture(path)
    finally: path.unlink()


def test_unshared_limit_and_missing_font_attacks_are_rejected(monkeypatch: pytest.MonkeyPatch):
    source = json.loads(FIXTURE.read_text())
    source["panels"][1]["data"]["matched"] = [[1, 2], [3, 4], [5, 6]]
    path = FIXTURE.parent / "_shape_attack.json"; path.write_text(json.dumps(source))
    try:
        with pytest.raises(StageFRendererError, match="share one shape"): load_fixture(path)
    finally: path.unlink()
    config = load_contracts(REPO)["config"]
    monkeypatch.setitem(config["style"]["font"], "regular_file", "fonts/ttf/ABSENT.ttf")
    with pytest.raises(StageFRendererError, match="missing"): validate_font(config)


def test_upstream_hash_and_git_divergence_attacks_fail_closed(monkeypatch: pytest.MonkeyPatch):
    import vitnns.stage_f_renderer as renderer
    real_sha = renderer.sha256_file
    monkeypatch.setattr(renderer, "sha256_file", lambda path: "0" * 64 if path.name == "e4.json" else real_sha(path))
    with pytest.raises(StageFRendererError, match="completion file hash"):
        verify_preconditions(REPO, require_clean=False, require_remote=False)


def test_frozen_scientific_states_and_authorization_are_exact():
    config = load_contracts(REPO)["config"]
    assert config["scientific_state"]["stage_d"] == {"cost":"PASS_COST","decision":"GO_FIDELITY_SCOPE","fidelity":"PASS_FIDELITY","median_delta_fid":2.3081996221714247,"n":500,"paired_image_bootstrap_ci95":[2.1110806126093067,2.445235411755993],"positive":500,"validity":"PASS"}
    assert config["scientific_state"]["c10"]["n"] == 15
    assert config["scientific_state"]["cohort_pooling_allowed"] is False
    assert config["scientific_state"]["stage_e"]["scientific_validity"] == "MIXED"
    assert config["scientific_state"]["stage_e"]["e3_failed_selection_orders"] == [117, 352]
    assert config["authorization"] == {"current":"F0_ONLY","on_pass":"F1_ONLY","f1_started":False,"f2_started":False,"f3_started":False,"f4_started":False}
    assert config["reference_declaration"] == {"style_reference_only":True,"scientific_input":False,"content_or_graphics_copied":False,"reference_values_entered_results":False}


def test_recipe_hash_constants_are_frozen():
    assert EXPECTED_D10_RECIPE_SHA256 == "aaa1858e5a2b0e820d4ea5cb3072a41e3cb6f4f507daefa319dc9ab66a1f0ad1"
    assert EXPECTED_E4_RECIPE_SHA256 == "57306bc69e399d1ea2353d908ac0d8b22403d17ec29318a3eadf36b3b8d98a54"
