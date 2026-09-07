from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from vitnns.stage_d_provenance import canonical_json_bytes, canonical_sha256
from vitnns.stage_f_f2 import (
    F2_REPORT_PATH, F2_ROOT_PATH, MODULES, LAYOUTS, StageFF2Error,
    _load_config, build_plot_data, build_render_spec, load_source_bundle,
    validate_completion, validate_f2_render_report, validate_outputs,
)
from vitnns.stage_f_renderer import render_panel_figure

REPO = Path(__file__).resolve().parents[1]


def _render(module: str, layout: str, output: Path) -> dict:
    source = load_source_bundle(REPO); plots = build_plot_data(source); config = _load_config(REPO)
    return render_panel_figure(
        build_render_spec(module, plots[module], layout), output, repo_root=REPO,
        renderer_commit="7a516571eabf364b7bb5d46ceb8aa74b450b3ae3",
        source_input_digest=source["source_input_digest"],
        plot_data_digest=plots[module]["plot_data_digest"],
        stage_config_digest=config["canonical_digest"], stage_id="F2",
        report_document_type="stage_f_f2_render_report",
        report_flags={"p_value_computed": False, "winner_selection_performed": False, "new_go_no_go_issued": False, "new_endpoint_computed": False, "final_combined_figures_rendered": False, "failed_rows_117_352_retained": True, "counts_accompany_valid_only_summaries": True, "e4_scientific_validity": "MIXED"},
    )


def test_f2_png_svg_pdf_are_deterministic_and_cross_format_bound(tmp_path: Path):
    first = _render("F2-B", "double_column", tmp_path / "first")
    second = _render("F2-B", "double_column", tmp_path / "second")
    for extension in ("png", "pdf", "svg"):
        assert first["outputs"][extension]["sha256"] == second["outputs"][extension]["sha256"]
    assert first["outputs"]["pdf"]["font_embedded"] is True
    assert first["outputs"]["pdf"]["internal_object_digest"] == second["outputs"]["pdf"]["internal_object_digest"]
    assert len(set(first["format_content_digests"].values())) == 1
    assert first["size_inches"] == [7.16, 4.2]
    assert first["dpi"] == 600
    assert validate_f2_render_report(tmp_path / "first" / "f2_b_double_column.report.json", REPO)["status"] == "PASS"


def test_f2_single_column_exact_canvas(tmp_path: Path):
    report = _render("F2-A", "single_column", tmp_path / "render")
    assert report["size_inches"] == [3.5, 7.2]
    assert report["outputs"]["png"]["pixel_size"] == [2100, 4320]


def test_f2_render_report_tamper_is_rejected(tmp_path: Path):
    report = _render("F2-D", "double_column", tmp_path / "render")
    report["winner_selection_performed"] = True
    report["canonical_digest"] = canonical_sha256({key: value for key, value in report.items() if key != "canonical_digest"})
    path = tmp_path / "render" / "f2_d_double_column.report.json"
    path.write_bytes(canonical_json_bytes(report) + b"\n")
    with pytest.raises((StageFF2Error, Exception)):
        validate_f2_render_report(path, REPO)


def test_formal_outputs_have_8_reports_24_exports_8_review_png_and_image_analysis_when_present():
    root = REPO / F2_ROOT_PATH
    if not root.exists():
        assert not (REPO / F2_REPORT_PATH).exists()
        return
    assert validate_outputs(REPO)["status"] == "PASS"
    reports = list((root / "renders").glob("**/*.report.json"))
    exports = [path for path in (root / "renders").glob("**/*") if path.suffix in {".png", ".pdf", ".svg"}]
    review = list((REPO / F2_REPORT_PATH / "figures_png").glob("*.png"))
    assert len(reports) == 8 and len(exports) == 24 and len(review) == 8
    analysis = (REPO / F2_REPORT_PATH / "figure_analysis.md").read_text(encoding="utf-8")
    assert all(f"Image #{index}" in analysis for index in range(1, 9))


def test_terminal_completion_is_atomic_and_authorizes_only_f3_when_present():
    completion = REPO / F2_ROOT_PATH / "completion/f2.json"
    if not completion.exists():
        pending = REPO / F2_ROOT_PATH / "completion/f2.pending.json"
        assert pending.exists() or not (REPO / F2_ROOT_PATH).exists()
        return
    assert not (REPO / F2_ROOT_PATH / "completion/f2.pending.json").exists()
    result = validate_completion(REPO)
    assert result["next_authorization"] == "F3_ONLY"
    assert result["f3_started"] is False and result["f4_started"] is False
