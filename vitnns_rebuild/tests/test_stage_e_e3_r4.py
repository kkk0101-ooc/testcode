"""Outcome-aware tests for the prospective E3-R4 typed-terminal contract."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_e_e3_r4 as r4

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_typed_terminal_acceptance() -> None:
    config = r4._validate_config(REPO_ROOT)
    assert config["canonical_digest"] == r4.EXPECTED_CONFIG_DIGEST
    assert config["scope"]["missing_complement_units"] == 1835
    assert config["scope"]["terminal_statuses"] == ["VALID", "FAILED", "NA"]
    assert config["scope"]["continue_after_scientific_failure_or_na"] is True
    assert config["acceptance"]["workflow_status"] == "PASS"
    assert config["acceptance"]["scientific_validity"] == "MIXED"
    assert config["acceptance"]["cost_policy"] == "MEASURE_AND_DISCLOSE_NOT_A_COMPLETION_GATE"
    assert tuple(config["numerical_contract"]["schedule"]) == r4.r1.COMPLETE_SCHEDULE
    assert config["numerical_contract"]["atol"] == 1e-4
    assert config["numerical_contract"]["rtol"] == 0.01
    Draft202012Validator.check_schema(json.loads((REPO_ROOT / r4.COMPLETION_SCHEMA_PATH).read_text()))


def test_upstream_and_r3_empty_pilot_are_bound() -> None:
    result = r4.validate_upstream(REPO_ROOT)
    assert result["e3_r2"]["status"] == "FAIL"
    assert result["e3_r2"]["valid_terminal_attempts"] == 1164
    assert result["e3_r2"]["failed_terminal_attempts"] == 1
    assert result["e3_r2"]["missing_attempts"] == 1835
    assert result["e3_r3"]["status"] == "PENDING_INTERRUPTED_WITHOUT_TERMINAL_PILOT_PAYLOAD"
    pilot_root = REPO_ROOT / r4.r3.PILOT_ROOT
    assert not pilot_root.exists() or not any(path.is_file() for path in pilot_root.rglob("*"))


def test_partition_inherits_failure_and_runs_only_1835_missing() -> None:
    inherited, eligible = r4._partition(REPO_ROOT)
    assert Counter(value[0] for value in inherited.values()) == Counter({
        "E3_V1_INHERITED": 516, "E3_R1_INHERITED": 12,
        "E3_R2_INHERITED_VALID": 636, "E3_R2_INHERITED_FAILED": 1,
    })
    assert len(eligible) == 1835
    failed = [item[1]["result"] for item in inherited.values() if item[0] == "E3_R2_INHERITED_FAILED"]
    assert len(failed) == 1
    assert (failed[0]["selection_order"], failed[0]["block"], failed[0]["head"], failed[0]["method"]) == (117, 11, 2, r4.e3.NNS_METHOD)
    assert failed[0]["unit_digest"] == "508feeb360fa42e11fc45ab8c138a8bb58d09b48144dd2280160a7e32db4e346"
    assert not any(unit["canonical_digest"] == failed[0]["unit_digest"] for unit in eligible)
    assert sum(1 for unit in eligible if unit["selection_order"] == 117) == 1


def test_shards_are_disjoint_and_cover_exact_missing_complement() -> None:
    _, eligible = r4._partition(REPO_ROOT)
    by_shard = {index: {unit["canonical_digest"] for unit in eligible if unit["selection_order"] % 8 == index} for index in range(8)}
    assert sum(len(value) for value in by_shard.values()) == 1835
    assert not any(by_shard[left] & by_shard[right] for left in range(8) for right in range(left + 1, 8))


@pytest.mark.parametrize("shard_id,shard_count", [(-1, 8), (8, 8), (0, 7), (True, 8)])
def test_invalid_shard_identity_fails_closed(shard_id: int, shard_count: int) -> None:
    with pytest.raises(r4.StageEE3R4Error):
        r4.run_shard(REPO_ROOT, shard_id=shard_id, shard_count=shard_count, device="cpu")


def test_typed_counts_never_erase_failed_or_na() -> None:
    counts = r4._typed_counts([{"validity_status": "VALID"}, {"validity_status": "FAILED"}, {"validity_status": "NA"}])
    assert counts == {"expected": 3, "valid": 1, "failed": 1, "na": 1}
    with pytest.raises(r4.StageEE3R4Error):
        r4._typed_counts([{"validity_status": "DROPPED"}])


def test_completion_requires_mixed_validity_and_disclosure() -> None:
    schema = json.loads((REPO_ROOT / r4.COMPLETION_SCHEMA_PATH).read_text())
    assert schema["properties"]["scientific_validity"]["const"] == "MIXED"
    assert schema["properties"]["next_authorization"]["const"] == "E4_ONLY_WITH_MIXED_VALIDITY_DISCLOSURE"
    assert schema["properties"]["e4_disclosure_required"]["const"] is True
    source = (REPO_ROOT / "src/vitnns/stage_e_e3_r4.py").read_text()
    assert "workflow_pass_relabelled_as_all_unit_scientific_pass" in source
    assert "invalid_row_included_in_numeric_summary" in source
    assert '"scientific_validity": "MIXED"' in source


def test_no_e4_execution_surface() -> None:
    public = {name for name in dir(r4) if name.startswith(("run_", "start_", "execute_"))}
    assert public == {"run_shard"}
