"""Focused synthetic golden tests for Stage D D6 aggregation and export.

These fixtures issue no runtime authority and never open images, models, CUDA,
or a formal root.  The mathematical tests construct an in-process sealed D5
receipt; one separate minimum integration test exercises the D5 snapshot
validator with the pre-existing synthetic fixture root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from vitnns.stage_d_analysis import (
    StageDAnalysisContractError, _bootstrap, _decision, _median,
    aggregate_stage_d_analysis, closed_analysis_summary,
)
import vitnns.stage_d_analysis as analysis_module
from vitnns.stage_d_analysis_input import (
    StageDValidatedAnalysisInput, StageDValidatedAnalysisUnit,
    _ISSUER, validate_stage_d_analysis_input,
)
from vitnns.stage_d_finalize import StageDRootValidation, finalize_stage_d_root
from vitnns.stage_d_paper import (
    StageDPaperBlockedError, StageDPaperContractError,
    build_stage_d_analysis_package, build_stage_d_paper_package,
    validate_stage_d_analysis_package, validate_stage_d_paper_package,
)
from vitnns.stage_d_plan import MATCHED_METHOD_ID, METHOD_IDS, NNS_METHOD_ID


REPO_ROOT = Path(__file__).resolve().parents[1]
pytest_plugins = ("test_stage_d_finalize",)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _curve(auc: float, direction: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    x = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    # This quadratic response has the requested trapezoidal AUC and frozen
    # directional endpoints for every synthetic tie repeat.
    curvature = 6.0 * (auc - 0.5)
    if direction == "insertion":
        y = x + curvature * x * (1.0 - x)
        y[0], y[-1] = 0.0, 1.0
    else:
        y = 1.0 - x + curvature * x * (1.0 - x)
        y[0], y[-1] = 1.0, 0.0
    # The test mathematics consumes/recomputes AUC.  Make its record exact.
    observed = float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5))
    records = [
        {"direction": direction, "replicate": repeat,
         "actual_fractions": x.tolist(), "responses": y.tolist(), "auc": observed}
        for repeat in range(5)
    ]
    return records, np.tile(y, (5, 1))


def _l0(method: str, sample_id: str, delta: float, *, misclassified: bool) -> tuple[dict[str, Any], np.ndarray]:
    # nns: insertion +delta, deletion -delta; matched is neutral.
    if method == NNS_METHOD_ID:
        insertion, deletion = 0.5 + delta, 0.5 - delta
    else:
        insertion = deletion = 0.5
    insertion_records, insertion_curves = _curve(insertion, "insertion")
    deletion_records, deletion_curves = _curve(deletion, "deletion")
    return ({
        "fidelity_curves": insertion_records + deletion_records,
        "prediction": {"clean_top1_correct": not misclassified},
    }, np.stack((insertion_curves, deletion_curves)))


def _receipt(
    deltas: np.ndarray, *, validity: str = "PASS", invalid_order: int | None = None,
) -> tuple[StageDValidatedAnalysisInput, Any]:
    assert deltas.shape == (500,)
    rows: list[StageDValidatedAnalysisUnit] = []
    reconciliation_rows: list[dict[str, Any]] = []
    for order, delta in enumerate(deltas):
        sample_id = f"synthetic-{order:03d}"
        for method_index, method in enumerate(METHOD_IDS):
            failed = validity == "FAIL" and order == invalid_order and method_index == 0
            l0, curves = _l0(method, sample_id, float(delta), misclassified=order == 7)
            rows.append(StageDValidatedAnalysisUnit(
                selection_order=order, sample_id=sample_id, group_digest=_sha(f"group-{order}"),
                pair_digest=_sha(f"pair-{order}"), unit_digest=_sha(f"unit-{order}-{method_index}"),
                method_index=method_index, method_id=method, audit=False,
                terminal_state="scientific_failed" if failed else "succeeded",
                failure_code="non_finite" if failed else None,
                scientific_errors=("non_finite",) if failed else (),
                na_reason="non_finite" if failed else None, resources={
                    "elapsed_seconds": 0.0, "forward_calls": 0, "backward_calls": 0,
                    "l0_l1_payload_bytes": 0, "l2_payload_bytes": 0,
                    "control_and_provenance_bytes": 0, "cuda_peak_reserved_bytes": 0,
                }, numeric_payload_status=("WITHHELD_DUE_TO_SCIENTIFIC_INVALIDITY" if validity == "FAIL" else "VALIDATED"),
                l0=None if validity == "FAIL" else l0, curves=None if validity == "FAIL" else curves,
                l0_sha256=None, curves_sha256=None, _issuer=_ISSUER,
            ))
            reconciliation_rows.append({"selection_order": order, "method_id": method})
    root = StageDRootValidation(
        package_complete=True, scientific_validity=validity, workflow_status="PASS",
        plan_digest="a" * 64, final_status_sha256="b" * 64,
        root_inventory={"file_count": 1, "total_bytes": 1, "canonical_digest": "c" * 64},
        evidence_report={}, reconciliation={
            "expected_image_count": 500, "expected_unit_count": 1000,
            "image_count": 500, "unit_count": 1000,
            "rows": reconciliation_rows, "resources": {
                "elapsed_seconds": 0.0, "forward_calls": 0, "backward_calls": 0,
                "l0_l1_payload_bytes": 0, "l2_payload_bytes": 0,
                "control_and_provenance_bytes": 0, "cuda_peak_reserved_bytes": 0,
            },
        }, source_inventory={}, payload_manifest={}, whole_root_ordered_inventory=(),
    )
    # The numerical unit tests still require an issued receipt rather than a
    # serialized external aggregate.  The integration test below uses the
    # actual D5 issuer and source snapshot.
    receipt = StageDValidatedAnalysisInput(
        root_validation=root, unit_rows=tuple(rows), _issuer=_ISSUER,
    )
    return receipt, SimpleNamespace(canonical_digest="a" * 64)


def _aggregate(deltas: np.ndarray, *, cheap_bootstrap: bool = False, **kwargs: Any) -> dict[str, Any]:
    receipt, plan = _receipt(deltas, **kwargs)

    def lightweight(values, material, **_kwargs):
        center = analysis_module._median(np.asarray(values, dtype=np.float64))
        return ({
            "confidence_interval": [center, center],
            "draw_shape": [10000, 500],
        }, np.zeros(10_000))

    with patch.object(analysis_module, "validate_stage_d_plan", lambda value: value):
        if cheap_bootstrap:
            with patch.object(analysis_module, "_bootstrap", lightweight):
                return aggregate_stage_d_analysis(receipt, plan)
        return aggregate_stage_d_analysis(receipt, plan)


@pytest.fixture(scope="module")
def valid_result() -> dict[str, Any]:
    """One complete 10,000-draw A/B/C synthetic analysis for all export tests."""

    return _aggregate(np.full(500, 0.03))


def test_frozen_even_median_and_primary_seed_golden() -> None:
    values = np.arange(500, dtype=np.float64)
    assert _median(values) == 249.5
    metadata, distribution = _bootstrap(values, "stage-d-bootstrap|20260821|fine_4x4_cell_delta_fid_median")
    assert metadata["seed"] == "18136624947342364318"
    assert metadata["draw_shape"] == [10000, 500]
    assert distribution.shape == (10000,)


@pytest.mark.parametrize(
    ("positive", "median", "lower", "upper", "expected"),
    [
        (399, 0.03, 0.01, 0.03, "NO_GO_FIDELITY"),
        (400, 0.02, 0.0, 0.02, "INCONCLUSIVE_FIDELITY"),
        (400, 0.02, 0.001, 0.02, "GO_FIDELITY_SCOPE"),
        (500, 0.03, 0.01, 0.019, "NO_GO_FIDELITY"),
        (500, 0.03, 0.01, 0.03, "NO_GO_COST"),
    ],
)
def test_truth_table_equality_boundaries(
    positive: int, median: float, lower: float, upper: float, expected: str,
) -> None:
    summary = {"validity_status": "PASS", "positive_count": positive, "median": median,
               "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper}
    cost = {"status": "FAIL_COST" if expected == "NO_GO_COST" else "PASS_COST"}
    status = _decision(summary, cost)
    assert status[2] == expected


def test_zero_delta_is_not_positive_and_misclassification_is_retained() -> None:
    result = _aggregate(np.zeros(500, dtype=np.float64), cheap_bootstrap=True)
    assert result["summary"]["positive_count"] == 0
    assert result["overall_decision"] == "NO_GO_FIDELITY"
    assert result["per_sample"][7]["misclassified"] is True
    assert len(result["per_sample"]) == 500


def test_validity_failure_suppresses_effects_without_complete_case() -> None:
    result = _aggregate(np.full(500, 0.03), validity="FAIL", invalid_order=11)
    assert result["overall_decision"] == "NO_GO_VALIDITY"
    assert result["bootstrap"] is None
    assert not result["curves"]
    assert result["resources_and_qc"]["effect_outputs_status"] == "NOT_GENERATED_DUE_TO_VALIDITY"
    assert len(result["per_sample"]) == 500
    assert result["per_sample"][11]["delta_fid"] is None


@pytest.mark.parametrize(
    ("positive_count", "expected"),
    [(399, "NO_GO_FIDELITY"), (400, "GO_FIDELITY_SCOPE")],
)
def test_full_coordinate_aggregation_respects_399_400_positive_boundary(
    positive_count: int, expected: str,
) -> None:
    deltas = np.zeros(500, dtype=np.float64)
    deltas[:positive_count] = 0.03
    result = _aggregate(deltas, cheap_bootstrap=True)
    assert result["summary"]["positive_count"] == positive_count
    assert result["overall_decision"] == expected


def test_tie_repeat_population_sd_is_retained_but_not_an_image_cluster() -> None:
    result = _aggregate(np.full(500, 0.03, dtype=np.float64), cheap_bootstrap=True)
    row = result["per_sample"][0]
    assert row["nns_insertion_auc_repeat_sd_ddof_0"] == 0.0
    assert row["nns_deletion_auc_repeat_sd_ddof_0"] == 0.0
    assert tuple(result["bootstrap"]["draw_shape"]) == (10000, 500)


def test_closed_schema_rejects_unknown_analysis_field(
    tmp_path: Path, valid_result: dict[str, Any],
) -> None:
    forged = valid_result.to_dict()
    forged["unexpected"] = True
    with pytest.raises(StageDPaperContractError):
        build_stage_d_analysis_package(forged, tmp_path / "forged", repo_root=REPO_ROOT)


def test_package_is_deterministic_closed_and_no_overwrite(
    tmp_path: Path, valid_result: dict[str, Any],
) -> None:
    result = valid_result
    first = tmp_path / "first"
    second = tmp_path / "second"
    summary_one = build_stage_d_analysis_package(result, first, repo_root=REPO_ROOT)
    summary_two = build_stage_d_analysis_package(result, second, repo_root=REPO_ROOT)
    assert summary_one["analysis_canonical_digest"] == summary_two["analysis_canonical_digest"]
    assert validate_stage_d_analysis_package(first, repo_root=REPO_ROOT)["complete"] is True
    assert {path.relative_to(first).as_posix() for path in first.rglob("*") if path.is_file()} == {
        "analysis.json", "analysis_report.json", "decision.json", "figure_recipes.json",
        "provenance.json", "payload_manifest.json", "status.json",
        "paper_data/cohort_comparison.csv", "paper_data/cohort_comparison.json",
        "paper_data/curve_summary.csv", "paper_data/curve_summary.json",
        "paper_data/main_fidelity_table.csv", "paper_data/main_fidelity_table.json",
        "paper_data/per_sample_delta.csv", "paper_data/per_sample_delta.json",
        "paper_data/resource_and_qc.csv", "paper_data/resource_and_qc.json",
    }
    assert (first / "analysis.json").read_bytes() == (second / "analysis.json").read_bytes()
    assert {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*") if path.is_file()
    } == {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*") if path.is_file()
    }
    with pytest.raises(StageDPaperBlockedError):
        build_stage_d_analysis_package(result, first, repo_root=REPO_ROOT)
    with pytest.raises(StageDPaperContractError, match="D6-issued"):
        build_stage_d_analysis_package(result.to_dict(), tmp_path / "external-summary", repo_root=REPO_ROOT)


def test_package_detects_source_tamper_and_preserves_figure_input_contract(
    tmp_path: Path, valid_result: dict[str, Any],
) -> None:
    result = valid_result
    root = tmp_path / "analysis"
    build_stage_d_analysis_package(result, root, repo_root=REPO_ROOT)
    recipes = json.loads((root / "figure_recipes.json").read_text())
    assert [item["figure_id"] for item in recipes["figures"]] == ["A", "B", "C"]
    figure_c = recipes["figures"][2]
    assert figure_c["data"] == "paper_data/curve_summary.csv"
    assert figure_c["band"]["namespace"] == "stage-d-figure-bootstrap|20260821|"
    assert figure_c["band"]["simultaneous"] is False
    assert figure_c["band"]["enters_primary_decision"] is False
    assert figure_c["x_axis"] == {"label": "Intervention fraction", "scale": "linear"}
    assert figure_c["colors"][NNS_METHOD_ID] == "#1f77b4"
    assert figure_c["colors"][MATCHED_METHOD_ID] == "#ff7f0e"
    metadata = result["resources_and_qc"]["descriptive_curve_bootstrap"]
    assert metadata["namespace"] == "stage-d-figure-bootstrap|20260821|"
    assert len(metadata["coordinates"]) == 84
    first = metadata["coordinates"][0]["metadata"]
    assert first["resamples"] == 10000
    assert tuple(first["draw_shape"]) == (10000, 500)
    raw = (root / "analysis.json").read_bytes()
    (root / "analysis.json").write_bytes(raw.replace(b"synthetic-000", b"synthetic-999", 1))
    with pytest.raises(StageDPaperContractError):
        validate_stage_d_analysis_package(root, repo_root=REPO_ROOT)


def test_result_is_immutable_and_cohort_table_retains_c10_blockers(
    tmp_path: Path, valid_result: dict[str, Any],
) -> None:
    with pytest.raises(TypeError):
        valid_result["summary"]["median"] = 99.0
    root = tmp_path / "analysis"
    build_stage_d_analysis_package(valid_result, root, repo_root=REPO_ROOT)
    cohort = json.loads((root / "paper_data/cohort_comparison.json").read_text())
    c10 = cohort["rows"][0]
    stage_d = cohort["rows"][1]
    assert c10["n"] == 15 and stage_d["n"] == 500
    assert c10["pooled"] is False and stage_d["pooled"] is False
    assert c10["overall_decision"] == "NO_GO_NOVELTY"
    assert c10["workflow_status"] == "BLOCKED"
    assert c10["spatial_status"] == "FAIL_SPATIAL"
    assert c10["holm_status"] == "HOLM_BLOCKED_UNDERSPECIFIED"


def test_build_and_validate_paper_package_are_closed(
    tmp_path: Path, valid_result: dict[str, Any],
) -> None:
    result = valid_result
    analysis = tmp_path / "analysis"
    paper = tmp_path / "paper"
    build_stage_d_analysis_package(result, analysis, repo_root=REPO_ROOT)
    build_stage_d_paper_package(analysis, paper, repo_root=REPO_ROOT)
    assert validate_stage_d_paper_package(paper, repo_root=REPO_ROOT)["complete"] is True
    assert closed_analysis_summary(result)["coordinate_count"] == 500


def test_minimum_full_sealed_d5_root_receipt_is_closed_and_invalid_not_dropped(
    valid_root: Path, controls: tuple[Any, Any, Any],
) -> None:
    """One full 500-coordinate fixture proves the receipt is D5-derived."""

    plan, estimate, preflight = controls
    finalize_stage_d_root(valid_root, plan, estimate, preflight)
    receipt = validate_stage_d_analysis_input(valid_root, plan, estimate, preflight)
    assert receipt.package_complete is True
    assert len(receipt.unit_rows) == 1000
    assert {row.numeric_payload_status for row in receipt.unit_rows} == {
        "ANALYSIS_INCOMPLETE_SYNTHETIC_L0"
    }
    # This test is a synthetic fixture result, not a formal decision.  It
    # verifies the no-drop handling of a D5-valid but analysis-incomplete root.
    result = aggregate_stage_d_analysis(receipt, plan)
    assert result["overall_decision"] == "NO_GO_VALIDITY"
    assert len(result["per_sample"]) == 500
    # D6 cannot reuse a receipt after the D5-sealed source tree has drifted.
    l0 = next(valid_root.glob("units/*/attempts/attempt-000001/L0/result.json"))
    l0.write_bytes(l0.read_bytes() + b" ")
    with pytest.raises(Exception):
        validate_stage_d_analysis_input(valid_root, plan, estimate, preflight)
