"""Focused D10 formal decision report tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import vitnns.stage_d_analysis as analysis_module
import vitnns.stage_d_d10_completion as d10
from vitnns.stage_d_analysis import aggregate_stage_d_analysis
from vitnns.stage_d_paper import (
    build_stage_d_analysis_package,
    build_stage_d_paper_package,
)

from test_stage_d_analysis import REPO_ROOT, _aggregate, _receipt


def _cheap_bootstrap(values, material, **kwargs):
    return ({
        "confidence_interval": [0.03, 0.03],
        "draw_shape": [10000, 500],
        "status": "PERFORMED",
        "metric_id": kwargs.get(
            "metric_id", "fine_4x4_cell_delta_fid_median",
        ),
        "seed_material": material,
        "seed": "18136624947342364318",
        "seed_hash_extraction": "sha256_first_8_bytes_big_endian_unsigned",
        "prng": "numpy.random.Generator(numpy.random.PCG64(seed))",
        "resamples": 10000,
        "draw_indices_dtype_and_order": "little_endian_int64_c_order",
        "distribution_dtype_and_order": "little_endian_float64_c_order",
        "statistic": "paired_median",
        "even_median_rule": "arithmetic_mean_of_two_middle_order_statistics",
        "quantile_probabilities": [0.025, 0.975],
        "quantile_method": "linear",
        "draw_indices_sha256": "1" * 64,
        "distribution_sha256": "2" * 64,
    }, np.zeros(10_000))


def _d9_revalidation() -> dict:
    return {
        "schema_version": 1,
        "document_type": "stage_d_finalization_control",
        "operation": "validate",
        "status": "PASS",
        "package_complete": True,
        "scientific_validity": "PASS",
        "analysis_authorized": True,
        "cost_status": "PASS_COST",
        "plan_digest": "5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443",
        "final_status_sha256": (
            "9c2a7f6bbe89f393b17d39128c908b764c5481cb2ab640eea557ea87032e9f4c"
        ),
        "root_inventory": {
            "file_count": 7022,
            "total_bytes": 137250469,
            "canonical_digest": d10.D9_SEALED_ROOT_INVENTORY_DIGEST,
        },
    }


def _git_state() -> dict:
    return {
        "branch": "stage-d-d3-pass",
        "local_head_at_start": "f305adc5df16fdbbcb2d1f673892f9cef9d8ace6",
        "tracking_ref": "origin/stage-d-d3-pass",
        "tracking_sha_at_start": "f305adc5df16fdbbcb2d1f673892f9cef9d8ace6",
        "remote_ref": "refs/heads/stage-d-d3-pass",
        "remote_sha_at_start": "f305adc5df16fdbbcb2d1f673892f9cef9d8ace6",
        "d9_completion_ref": d10.D9_COMPLETION_REF,
        "d9_completion_ref_sha": "f305adc5df16fdbbcb2d1f673892f9cef9d8ace6",
        "d8_completion_ref": d10.D8_COMPLETION_REF,
        "d8_completion_ref_sha": d10.D8_COMPLETION_COMMIT,
        "d9_producer_ref": d10.D9_PRODUCER_REF,
        "d9_producer_ref_sha": d10.D9_PRODUCER_COMMIT,
        "worktree_index_clean_at_start": True,
        "status_porcelain_v2_at_start": [
            "# branch.oid f305adc5df16fdbbcb2d1f673892f9cef9d8ace6",
            "# branch.head stage-d-d3-pass",
            "# branch.upstream origin/stage-d-d3-pass",
            "# branch.ab +0 -0",
        ],
        "lfs_status_clean_at_start": True,
    }


def test_d10_report_binds_formal_packages_and_preserves_c10(tmp_path: Path) -> None:
    receipt, plan = _receipt(np.full(500, 0.03), formal=True)
    with patch.object(analysis_module, "validate_stage_d_plan", lambda value: value):
        with patch.object(analysis_module, "_bootstrap", _cheap_bootstrap):
            result = aggregate_stage_d_analysis(receipt, plan)

    analysis_root = tmp_path / "analysis"
    paper_root = tmp_path / "paper"
    build_stage_d_analysis_package(result, analysis_root, repo_root=REPO_ROOT)
    build_stage_d_paper_package(analysis_root, paper_root, repo_root=REPO_ROOT)

    report = d10.build_d10_formal_decision_report(
        repo_root=REPO_ROOT,
        analysis_root=analysis_root,
        paper_root=paper_root,
        d9_revalidation=_d9_revalidation(),
        git_state=_git_state(),
    )

    assert report["status"] == "PASS"
    assert report["statistics"]["positive_count"] == 500
    assert report["truth_table"]["overall_decision"] == "GO_FIDELITY_SCOPE"
    assert report["c10_status_preserved"] == d10.C10_STATUS_PRESERVED
    assert report["scope_boundaries"]["jpeg_read"] is False
    assert report["paper_artifacts"]["qualitative_overlays_generated"] is False
    assert report["canonical_digest"] == d10._semantic_digest(report)


def test_d10_report_rejects_synthetic_package(
    tmp_path: Path,
) -> None:
    with patch.object(analysis_module, "_bootstrap", _cheap_bootstrap):
        result = _aggregate(np.full(500, 0.03), cheap_bootstrap=False)
    analysis_root = tmp_path / "analysis"
    paper_root = tmp_path / "paper"
    build_stage_d_analysis_package(result, analysis_root, repo_root=REPO_ROOT)
    build_stage_d_paper_package(analysis_root, paper_root, repo_root=REPO_ROOT)

    with pytest.raises(d10.StageDD10CompletionError):
        d10.build_d10_formal_decision_report(
            repo_root=REPO_ROOT,
            analysis_root=analysis_root,
            paper_root=paper_root,
            d9_revalidation=_d9_revalidation(),
            git_state=_git_state(),
        )
