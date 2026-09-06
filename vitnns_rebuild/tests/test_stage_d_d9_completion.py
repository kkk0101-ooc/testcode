"""Focused D9 formal validation report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import vitnns.stage_d_d9_completion as d9


def _sha(index: int) -> str:
    return f"{index:x}" * 64


def _validation(*, formal: bool = True) -> dict:
    cost = {
        "evaluated": True,
        "actual_cost_includes_all_attempts_and_audit_repeats": True,
        "projection_source": (
            "D9_RECOMPUTED_FROM_VALIDATED_ATTEMPT_RESOURCES_AND_ROOT_INVENTORY"
        ),
        "resource_totals": {
            "elapsed_seconds": 10.0,
            "forward_calls": 20,
            "backward_calls": 10,
            "l0_l1_payload_bytes": 30,
            "l2_payload_bytes": 4,
            "control_and_provenance_bytes": 5,
            "cuda_peak_reserved_bytes": 6,
        },
        "limits": {
            "stage_d_total_retained_regular_file_bytes_max": 68719476736,
            "stage_d_total_h20_gpu_hours_max": 2000.0,
            "stage_d_h20_gpu_hours_per_image_group_max": 4.0,
        },
        "total_h20_gpu_hours": 10.0 / 3600.0,
        "h20_gpu_hours_per_image_group_max": 0.01,
        "total_retained_regular_file_bytes": 1000,
        "source_file_count": 10,
        "gate_failures": [],
        "status": "PASS_COST",
    }
    evidence = {
        "package_complete": True,
        "workflow_status": "PASS",
        "scientific_validity": "PASS",
        "analysis_authorized": formal,
        "authorization": {"formal": formal, "analysis_authorization": "D9_AUTHORIZED_D10_FULL_ANALYSIS"},
        "analysis_performed": False,
        "bootstrap_performed": False,
        "decision_performed": False,
        "scientific_failures": [],
        "actual_cost_audit": cost,
    }
    return {
        "package_complete": True,
        "workflow_status": "PASS",
        "scientific_validity": "PASS",
        "analysis_authorized": formal,
        "plan_digest": _sha(1),
        "final_status_sha256": _sha(2),
        "root_inventory": {
            "file_count": 11,
            "total_bytes": 2000,
            "canonical_digest": _sha(3),
        },
        "actual_cost_audit": cost,
        "evidence_report": evidence,
        "reconciliation": {
            "canonical_digest": _sha(4),
            "image_count": 500,
            "unit_count": 1000,
            "expected_counts": {"execution_units": 1000},
            "observed_counts": {"execution_units": 1000},
            "state_counts": {"succeeded": 1000},
            "na_rows": [],
            "failure_rows": [{}] * 1000,
            "no_rows_dropped": True,
            "zero_imputation_performed": False,
            "checkpoint_failure_table_reconciled": True,
            "checkpoint_resources_reconciled": True,
            "retention": {"observed": {}, "expected": {}},
        },
        "source_inventory": {"canonical_digest": _sha(5), "file_count": 10, "total_bytes": 1000},
        "payload_manifest": {"canonical_digest": _sha(6)},
    }


def test_d9_report_binds_formal_validation_and_d8_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / d9.FORMAL_ROOT
    (root / "final").mkdir(parents=True)
    for name in d9.FINAL_FILES:
        document = {"schema_version": 1, "document_type": name}
        document["canonical_digest"] = d9._semantic_digest(document)
        (root / "final" / name).write_bytes(d9.canonical_file_bytes(document))
    (tmp_path / d9.BLOCKED_ROOT).mkdir(parents=True)
    campaign = tmp_path / d9.CAMPAIGN_COMPLETION_PATH
    campaign.parent.mkdir(parents=True)
    campaign.write_bytes(b"campaign\n")
    monkeypatch.setattr(d9, "validate_d8_campaign_completion", lambda **_kwargs: {
        "canonical_digest": "50fa9276fbd3dcc9684dc0482b1be639cfd59aa111bbfb30ce0aec5b676333c1",
        "final_checkpoint": {
            "generation": 3016,
            "canonical_digest": "3c0951e07ef1ed10446188c0ebd56e141a26218962fd147a7ce2f13a5d426fc9",
        },
    })

    report = d9.build_d9_formal_validation_report(
        _validation(), repo_root=tmp_path, git_state={"branch": "stage-d-d3-pass"},
    )

    assert report["status"] == "PASS"
    assert report["analysis_authorized"] is True
    assert report["validation"]["d10_allowed_path"] == "D10_FULL_ANALYSIS_AND_DECISION"
    assert report["state_separation"] == {
        "package_completeness": "PASS",
        "workflow_integrity": "PASS",
        "scientific_validity": "PASS",
        "cost_status": "PASS_COST",
    }
    assert report["scope_boundaries"]["d10_aggregate_statistics_decision_run"] is False
    assert report["canonical_digest"] == d9._semantic_digest(report)


def test_d9_report_rejects_nonformal_validation(tmp_path: Path) -> None:
    with pytest.raises(d9.StageDD9CompletionError):
        d9.build_d9_formal_validation_report(
            _validation(formal=False),
            repo_root=tmp_path,
            git_state={"branch": "stage-d-d3-pass"},
        )
