from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns.stage_d_provenance import canonical_sha256
from vitnns.stage_f_f1 import (
    ATTACKS,
    F1_COMPLETION_SCHEMA_PATH,
    F1_CONFIG_PATH,
    F1_CONFIG_SCHEMA_PATH,
    F1_RENDER_REPORT_SCHEMA_PATH,
    StageFF1Error,
    assert_f1_source_boundary,
    build_false_pass_audit,
    build_plot_data,
    build_render_spec,
    load_source_bundle,
    validate_runtime_isolation,
    verify_startup_gates,
)

REPO = Path(__file__).resolve().parents[1]


def _schema(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def test_f1_config_and_schema_accept_frozen_document_and_reject_bad_example():
    document = json.loads((REPO / F1_CONFIG_PATH).read_text())
    schema = _schema(F1_CONFIG_SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    assert not list(Draft202012Validator(schema).iter_errors(document))
    assert document["canonical_digest"] == canonical_sha256(
        {key: value for key, value in document.items() if key != "canonical_digest"}
    )
    bad = copy.deepcopy(document)
    bad["authorization"]["on_pass"] = "F3_ONLY"
    bad["canonical_digest"] = canonical_sha256(
        {key: value for key, value in bad.items() if key != "canonical_digest"}
    )
    assert list(Draft202012Validator(schema).iter_errors(bad))


def test_f1_startup_gate_accepts_clean_successor_and_rejects_f0_tamper(monkeypatch: pytest.MonkeyPatch):
    assert verify_startup_gates(REPO, require_clean=False, require_remote=False)["status"] == "PASS"
    import vitnns.stage_f_f1 as f1

    real_sha = f1.sha256_file
    monkeypatch.setattr(
        f1, "sha256_file",
        lambda path: "0" * 64 if path.name == "f0.json" else real_sha(path),
    )
    with pytest.raises(StageFF1Error, match="F0 completion file hash"):
        verify_startup_gates(REPO, require_clean=False, require_remote=False)


def test_f1_startup_gate_rejects_d10_ref_report_recipe_manifest_and_paper_data_tamper(monkeypatch: pytest.MonkeyPatch):
    import vitnns.stage_f_f1 as f1

    real_git = f1._git
    monkeypatch.setattr(
        f1, "_git",
        lambda repo, *args: "0" * 40
        if args[:2] == ("rev-parse", "refs/archive/stage-d-fidelity-v1-d10-completion^{commit}")
        else real_git(repo, *args),
    )
    with pytest.raises(StageFF1Error, match="D10 archive ref"):
        verify_startup_gates(REPO, require_clean=False, require_remote=False)
    monkeypatch.setattr(f1, "_git", real_git)
    real_sha = f1.sha256_file
    monkeypatch.setattr(
        f1, "sha256_file",
        lambda path: "0" * 64 if path.name == "figure_recipes.json" else real_sha(path),
    )
    with pytest.raises(StageFF1Error, match="figure recipes hash"):
        verify_startup_gates(REPO, require_clean=False, require_remote=False)
    monkeypatch.setattr(
        f1, "sha256_file",
        lambda path: "0" * 64 if path.name == "per_sample_delta.csv" else real_sha(path),
    )
    with pytest.raises(StageFF1Error, match="per_sample_delta.csv hash"):
        load_source_bundle(REPO)


def test_csv_json_parity_and_analysis_paper_copy_are_bound():
    source = load_source_bundle(REPO)
    assert source["status"] == "PASS"
    assert source["tables"]["cohort_comparison"]["row_count"] == 2
    assert source["tables"]["per_sample_delta"]["row_count"] == 500
    assert source["tables"]["curve_summary"]["row_count"] == 84
    assert all(item["byte_identical"] for item in source["analysis_paper_parity"])


def test_f1_a_cohort_order_n_median_ci_and_no_pooling():
    plot = build_plot_data(load_source_bundle(REPO))["F1-A"]
    points = plot["plot_payload"]["points"]
    assert [item["cohort"] for item in points] == ["c10", "stage_d"]
    assert [item["n"] for item in points] == [15, 500]
    assert plot["plot_payload"]["pooled_n"] is None
    assert plot["c10_stage_d_pooled"] is False
    assert points[1]["estimate"] == 2.3081996221714247
    assert points[1]["lower"] == 2.1110806126093067
    assert points[1]["upper"] == 2.445235411755993
    assert points[0]["overall_decision"] == "NO_GO_NOVELTY"
    assert points[0]["spatial_status"] == "FAIL_SPATIAL"
    assert points[0]["workflow_status"] == "BLOCKED"


def test_f1_b_uses_full_500_rows_ecdf_endpoint_and_keeps_misclassified():
    plot = build_plot_data(load_source_bundle(REPO))["F1-B"]
    payload = plot["plot_payload"]
    ecdf = payload["ecdf"]
    assert payload["rows_used"] == payload["ecdf_denominator"] == 500
    assert ecdf[-1]["ecdf"] == 1.0
    assert all(left["delta_fid"] <= right["delta_fid"] for left, right in zip(ecdf, ecdf[1:]))
    assert payload["misclassified_rows_retained"] > 0
    assert any(row["misclassified"] is True for row in ecdf)
    assert plot["misclassified_rows_filtered"] is False


def test_f1_c_direction_method_fraction_median_and_band_exactness():
    plot = build_plot_data(load_source_bundle(REPO))["F1-C"]
    payload = plot["plot_payload"]
    assert payload["direction_order"] == ["insertion", "deletion"]
    assert payload["method_order"] == ["nns", "matched"]
    assert payload["band"]["simultaneous"] is False
    assert payload["band"]["enters_primary_decision"] is False
    assert payload["within_image_reduction"] == "mean_of_five_tie_repeats"
    assert payload["across_image_statistic"] == "pointwise_median"
    for curve in payload["curves"]:
        assert curve["n"] == 500
        assert curve["fraction"] == [index / 20 for index in range(21)]
        assert all(low <= mid <= high for low, mid, high in zip(
            curve["pointwise_ci_lower"], curve["median"], curve["pointwise_ci_upper"]
        ))


def test_f1_render_specs_fix_styles_sizes_and_caption_order():
    plot_data = build_plot_data(load_source_bundle(REPO))
    single = build_render_spec("F1-A", plot_data["F1-A"], "single_column")
    double = build_render_spec("F1-C", plot_data["F1-C"], "double_column")
    assert single["layout_profile"] == "single_column"
    assert double["layout_profile"] == "double_column"
    assert [panel["caption_key"] for panel in double["panels"]] == double["caption_keys"]
    assert double["panels"][0]["panel_id"] == "f1_c_insertion"
    assert double["panels"][1]["panel_id"] == "f1_c_deletion"
    series = double["panels"][0]["data"]["series"]
    assert [item["method"] for item in series] == ["nns", "matched"]


def test_no_recompute_flags_and_no_forbidden_runtime_surface(monkeypatch: pytest.MonkeyPatch):
    plot_data = build_plot_data(load_source_bundle(REPO))
    assert all(item["bootstrap_ci_recomputed"] is False for item in plot_data.values())
    assert all(item["new_endpoint_computed"] is False for item in plot_data.values())
    assert assert_f1_source_boundary(REPO)["status"] == "PASS"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert validate_runtime_isolation()["status"] == "PASS"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(StageFF1Error, match="CUDA_VISIBLE_DEVICES"):
        validate_runtime_isolation()


def test_false_pass_audit_and_completion_schema_are_exact():
    audit = build_false_pass_audit(REPO)
    assert audit["status"] == "PASS"
    assert len(audit["attacks"]) == len(set(item["attack"] for item in audit["attacks"])) == 18
    assert set(item["attack"] for item in audit["attacks"]) == set(ATTACKS)
    schema = _schema(F1_COMPLETION_SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    bad = {"schema_version": 1, "document_type": "stage_f_f1_terminal_completion", "stage_id": "F1", "status": "PASS", "next_authorization": "F3_ONLY"}
    assert list(Draft202012Validator(schema).iter_errors(bad))
    render_schema = _schema(F1_RENDER_REPORT_SCHEMA_PATH)
    Draft202012Validator.check_schema(render_schema)
