"""Focused D8 campaign completion sealing checks."""

from __future__ import annotations

import json
from pathlib import Path

from vitnns.stage_d_d8_completion import (
    CAMPAIGN_COMPLETION_PATH,
    build_d8_campaign_completion,
    validate_d8_campaign_completion,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_d8_campaign_completion_is_source_recomputable() -> None:
    path = REPO_ROOT / CAMPAIGN_COMPLETION_PATH
    if not path.exists():
        # The focused test remains useful before the sealing artifact is written.
        record = build_d8_campaign_completion(repo_root=REPO_ROOT)
    else:
        record = validate_d8_campaign_completion(repo_root=REPO_ROOT)

    assert record["status"] == "PASS"
    assert record["parent_stage"] == "D8"
    assert record["next_task"] == "D9"
    assert record["reconciliation"]["all_shards_terminal"] is True
    assert record["reconciliation"]["all_receipts_verified"] is True
    assert record["campaign_totals"] == {
        "image_groups": 500,
        "units": 1000,
        "map_work_items": 1010,
        "fidelity_curves": 10000,
        "curve_points": 210000,
        "logical_work_items": 211010,
        "succeeded": 1000,
        "scientific_failed": 0,
        "retry_pending": 0,
        "blocked": 0,
    }


def test_d8_campaign_completion_keeps_analysis_boundary_closed() -> None:
    record = (
        validate_d8_campaign_completion(repo_root=REPO_ROOT)
        if (REPO_ROOT / CAMPAIGN_COMPLETION_PATH).exists()
        else build_d8_campaign_completion(repo_root=REPO_ROOT)
    )
    assert record["control_flags"] == {
        "scientific_failures_present": False,
        "prior_shards_rerun": False,
        "retry_attempts_created": False,
        "effect_summary_accessed": False,
        "finalize_run": False,
        "aggregate_run": False,
        "d9_started": False,
        "d10_started": False,
        "paper_data_or_figures_generated": False,
    }
    assert record["evaluation_boundaries"]["scientific_validity"] == "NOT_EVALUATED"
    assert record["evaluation_boundaries"]["analysis_authorization"] == "NOT_EVALUATED"
    assert record["evaluation_boundaries"]["completed_campaign_cost_gate"] == "NOT_EVALUATED"
    assert record["final_checkpoint"]["generation"] == 3016
    assert record["final_checkpoint"]["canonical_digest"] == (
        "3c0951e07ef1ed10446188c0ebd56e141a26218962fd147a7ce2f13a5d426fc9"
    )
    assert record["receipt_summary_field_deviations"] == [{
        "shard": "S04",
        "field": "counts.audit_image_groups",
        "stored": 0,
        "source_recomputed": 1,
        "disposition": (
            "historical summary-field deviation; source manifests and "
            "map/logical work-item totals verify audit evidence retained"
        ),
    }]
