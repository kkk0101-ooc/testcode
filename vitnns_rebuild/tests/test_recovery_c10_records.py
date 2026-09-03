from __future__ import annotations

import json
from pathlib import Path

from vitnns.stage_c_provenance import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    value = json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))
    assert value["canonical_digest"] == canonical_sha256({
        key: item for key, item in value.items() if key != "canonical_digest"
    })
    return value


def test_completion_separates_build_decision_acceptance_and_authorization() -> None:
    value = _load("recovery_c10_completion_20260903.json")
    assert value["build_complete"] is True
    assert value["integrity_valid"] is True
    assert value["status"] == "BLOCKED"
    assert value["scientific_results"]["primary_decision"] == "NO_GO_NOVELTY"
    assert value["scientific_results"]["effect_fidelity"] == "PASS_EFFECT"
    assert value["scientific_results"]["overall_spatial"] == "FAIL_SPATIAL"
    assert value["scientific_results"]["pass_effect_does_not_override_spatial_failure"] is True
    assert value["stage_d_handoff"]["execution_authorized"] is False
    assert value["stage_d_handoff"]["analysis_authorized"] is False
    assert value["stage_d_handoff"]["reserve_images_accessed"] is False


def test_validation_inventory_and_stage_d_handoff_are_canonical() -> None:
    validation = _load("recovery_c10_validation_20260903.json")
    inventory = _load("recovery_c10_whole_root_inventory_20260903.json")
    handoff = _load("recovery_c10_stage_d_handoff_20260903.json")
    assert validation["c9_source_gate"]["succeeded_units"] == 320
    assert inventory["regular_file_count"] == 12
    assert len(inventory["records"]) == 12
    assert handoff["future_stage_d_scope"]["run_stability"] is False
    assert handoff["future_stage_d_scope"]["cohort_n"] == 500
    assert handoff["authorization"]["stage_d_execution_authorized"] is False
    assert not any(handoff["access_attestation"].values())
