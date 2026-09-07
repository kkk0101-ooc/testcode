from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vitnns.stage_d_provenance import canonical_sha256
from vitnns.stage_f_f1 import (
    F1_RENDER_REPORT_SCHEMA_PATH,
    F1_ROOT_PATH,
    StageFF1Error,
    assert_no_future_outputs,
    build_plot_data,
    build_render_spec,
    load_source_bundle,
    validate_f1_render_report,
)
from vitnns.stage_f_renderer import render_panel_figure

REPO = Path(__file__).resolve().parents[1]


def test_f1_formal_render_reports_are_stable_and_cross_format_bound(tmp_path: Path):
    source = load_source_bundle(REPO)
    plot_data = build_plot_data(source)
    config = json.loads((REPO / "config/stage_f_f1.json").read_text())
    reports = []
    for module_id in ("F1-A", "F1-B", "F1-C"):
        for layout in ("single_column", "double_column"):
            spec = build_render_spec(module_id, plot_data[module_id], layout)
            report = render_panel_figure(
                spec,
                tmp_path / module_id.lower(),
                repo_root=REPO,
                renderer_commit="0" * 40,
                source_input_digest=source["source_input_digest"],
                plot_data_digest=plot_data[module_id]["plot_data_digest"],
                stage_config_digest=config["canonical_digest"],
            )
            report_path = tmp_path / module_id.lower() / f"{report['figure_id']}.report.json"
            assert validate_f1_render_report(report_path, REPO)["status"] == "PASS"
            assert len(set(report["format_content_digests"].values())) == 1
            assert report["outputs"]["png"]["pixel_size"][0] == (
                2100 if layout == "single_column" else 4296
            )
            for suffix in ("png", "pdf", "svg"):
                assert (report_path.parent / report["outputs"][suffix]["path"]).is_file()
            reports.append(report)
    assert len(reports) == 6


def test_f1_png_and_svg_byte_stability(tmp_path: Path):
    source = load_source_bundle(REPO)
    plot_data = build_plot_data(source)["F1-B"]
    config = json.loads((REPO / "config/stage_f_f1.json").read_text())
    kwargs = dict(
        repo_root=REPO,
        renderer_commit="1" * 40,
        source_input_digest=source["source_input_digest"],
        plot_data_digest=plot_data["plot_data_digest"],
        stage_config_digest=config["canonical_digest"],
    )
    first = render_panel_figure(build_render_spec("F1-B", plot_data, "single_column"), tmp_path / "first", **kwargs)
    second = render_panel_figure(build_render_spec("F1-B", plot_data, "single_column"), tmp_path / "second", **kwargs)
    for suffix in ("png", "svg", "pdf"):
        assert first["outputs"][suffix]["sha256"] == second["outputs"][suffix]["sha256"]
    assert first["outputs"]["pdf"]["internal_object_digest"] == second["outputs"]["pdf"]["internal_object_digest"]
    with Image.open(tmp_path / "first" / first["outputs"]["png"]["path"]) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    assert pixels.std() > 5.0


def test_f1_render_report_tamper_is_rejected(tmp_path: Path):
    source = load_source_bundle(REPO)
    plot_data = build_plot_data(source)["F1-A"]
    config = json.loads((REPO / "config/stage_f_f1.json").read_text())
    report = render_panel_figure(
        build_render_spec("F1-A", plot_data, "single_column"),
        tmp_path,
        repo_root=REPO,
        renderer_commit="2" * 40,
        source_input_digest=source["source_input_digest"],
        plot_data_digest=plot_data["plot_data_digest"],
        stage_config_digest=config["canonical_digest"],
    )
    report_path = tmp_path / f"{report['figure_id']}.report.json"
    png = tmp_path / report["outputs"]["png"]["path"]
    png.write_bytes(png.read_bytes() + b"tamper")
    with pytest.raises(StageFF1Error, match="tampered"):
        validate_f1_render_report(report_path, REPO)


def test_validate_outputs_rejects_final_figure_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "formal"
    monkeypatch.setattr("vitnns.stage_f_f1.F1_ROOT_PATH", str(root))
    (root / "renders").mkdir(parents=True)
    (root / "figure_2_combined.png").write_bytes(b"not allowed")
    with pytest.raises(StageFF1Error, match="final Figure 2"):
        assert_no_future_outputs(REPO)


def test_f1_completion_schema_rejects_f3_or_f4_authorization():
    schema = json.loads((REPO / F1_RENDER_REPORT_SCHEMA_PATH).read_text())
    assert schema["properties"]["final_combined_figure_2_rendered"]["const"] is False
    completion_schema = json.loads((REPO / "config/stage_f_f1_completion.schema.json").read_text())
    bad = {
        "schema_version": 1,
        "document_type": "stage_f_f1_terminal_completion",
        "stage_id": "F1",
        "status": "PASS",
        "next_authorization": "F4_ONLY",
    }
    from jsonschema import Draft202012Validator

    assert list(Draft202012Validator(completion_schema).iter_errors(bad))
