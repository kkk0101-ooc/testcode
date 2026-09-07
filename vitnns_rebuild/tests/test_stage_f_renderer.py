from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from jsonschema import Draft202012Validator

from vitnns.stage_d_provenance import canonical_sha256
from vitnns.stage_f_renderer import (
    StageFRendererError, load_contracts, load_fixture, parse_upstream_recipes,
    render_fixture, validate_cross_format_consistency, validate_font,
    validate_render_report,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/stage_f"
COMMIT = "cff6b72748c9fad70fd251337371d41bae983f6f"


def _render(tmp_path: Path, name: str):
    output = tmp_path / name
    report = render_fixture(FIXTURES / f"{name}.json", output, renderer_commit=COMMIT)
    return output / f"f0_{name}.report.json", report


def test_contract_schemas_accept_frozen_documents_and_reject_bad_examples():
    contracts = load_contracts(REPO)
    assert set(contracts) == {"config", "run_spec", "catalog", "captions"}
    config = copy.deepcopy(contracts["config"]); config["style"]["minimum_font_pt"] = 6.9
    config["canonical_digest"] = canonical_sha256({k: v for k, v in config.items() if k != "canonical_digest"})
    schema = json.loads((REPO / "config/stage_f_f0.schema.json").read_text())
    assert list(Draft202012Validator(schema).iter_errors(config))
    completion_schema = json.loads((REPO / "config/stage_f_completion_receipt.schema.json").read_text())
    assert list(Draft202012Validator(completion_schema).iter_errors({"status": "PASS", "next_authorization": "F2_ONLY"}))


def test_renderer_rejects_invalid_shapes_and_panel_order(tmp_path: Path):
    fixture = json.loads((FIXTURES / "golden_statistics.json").read_text())
    fixture["panels"][0]["data"]["series"][0]["y"] = [1, 2]
    path = FIXTURES / "_invalid_shape.json"; path.write_text(json.dumps(fixture))
    try:
        with pytest.raises(StageFRendererError, match="shape mismatch"):
            load_fixture(path)
    finally:
        path.unlink()
    fixture = json.loads((FIXTURES / "golden_maps.json").read_text()); fixture["caption_keys"].reverse()
    path = FIXTURES / "_invalid_order.json"; path.write_text(json.dumps(fixture))
    try:
        with pytest.raises(StageFRendererError, match="caption-key order"):
            load_fixture(path)
    finally:
        path.unlink()


def test_method_colors_are_fixed_and_non_color_encodings_differ():
    config = load_contracts(REPO)["config"]["style"]["methods"]
    assert config["nns"] == {"color": "#0072B2", "linestyle": "-", "marker": "o"}
    assert config["matched"] == {"color": "#D55E00", "linestyle": "--", "marker": "s"}


def test_font_file_hash_and_silent_fallback_detection(monkeypatch: pytest.MonkeyPatch):
    config = load_contracts(REPO)["config"]
    fonts = validate_font(config); assert all(item["family"] == "STIXGeneral" for item in fonts.values())
    monkeypatch.setitem(config["style"]["font"], "regular_sha256", "0" * 64)
    with pytest.raises(StageFRendererError, match="font file hash drifted"):
        validate_font(config)


@pytest.mark.parametrize("name,width,height", [("golden_statistics",3.5,7.2),("golden_maps",7.16,4.2),("golden_qualitative",7.16,4.8)])
def test_layout_sizes_and_cross_format_content(tmp_path: Path, name: str, width: float, height: float):
    report_path, report = _render(tmp_path, name)
    assert report["size_inches"] == [width, height]
    assert report["outputs"]["png"]["pixel_size"] == [round(width * 600), round(height * 600)]
    assert validate_cross_format_consistency(report_path)["status"] == "PASS"


def test_png_svg_report_and_pdf_are_deterministic(tmp_path: Path):
    first_path, first = _render(tmp_path / "first", "golden_maps")
    second_path, second = _render(tmp_path / "second", "golden_maps")
    for extension in ("png", "svg", "pdf"):
        assert first["outputs"][extension]["sha256"] == second["outputs"][extension]["sha256"]
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["outputs"]["pdf"]["metadata_normalized"] is True
    assert first["outputs"]["pdf"]["internal_object_digest"] == second["outputs"]["pdf"]["internal_object_digest"]


def test_shared_symmetric_limits_difference_scale_and_overlay_contract(tmp_path: Path):
    _, report = _render(tmp_path, "golden_maps")
    pair = next(panel for panel in report["panels"] if panel["kind"] == "signed_attribution_pair")
    assert pair["shared_colorbar"] is True and pair["nns_clim"] == pair["matched_clim"]
    assert pair["nns_clim"][0] == -pair["nns_clim"][1]
    difference = next(panel for panel in report["panels"] if panel["kind"] == "difference_map")
    assert difference["color_limit"][0] == -difference["color_limit"][1]
    overlay = next(panel for panel in report["panels"] if panel["kind"] == "rgb_overlay")
    assert overlay["alpha"] == 0.55 and overlay["resize_order"] == 1
    assert overlay["normalization"] == "symmetric_about_zero"


def test_tampered_source_or_render_is_rejected(tmp_path: Path):
    report_path, report = _render(tmp_path, "golden_statistics")
    png = report_path.parent / report["outputs"]["png"]["path"]
    png.write_bytes(png.read_bytes() + b"tamper")
    with pytest.raises(StageFRendererError, match="tampered"):
        validate_render_report(report_path)


def test_recipe_files_are_parsed_without_rendering_or_new_statistics():
    result = parse_upstream_recipes(REPO)
    assert result["status"] == "PASS_PARSE_ONLY"
    assert result["renders_created"] == 0 and result["new_statistics_computed"] is False
    assert result["d10"]["figure_ids"] == ["A", "B", "C"]


def test_golden_visual_sanity(tmp_path: Path):
    for name in ("golden_statistics", "golden_maps", "golden_qualitative"):
        report_path, report = _render(tmp_path, name)
        with Image.open(report_path.parent / report["outputs"]["png"]["path"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        assert rgb.std() > 8.0
        nonwhite = np.mean(np.any(rgb < 248, axis=2))
        assert 0.02 < nonwhite < 0.9
