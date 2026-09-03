"""Independent checks for the terminal recovery audit and handoff."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from vitnns.stage_d_recovery import STAGES, semantic_digest, validate_stage_bundle


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "config/stage_d_recovery_summary.schema.json"
AUDIT = ROOT / "reports/stage_d_recovery_v2_20260904_r3/false_pass_audit.json"
HANDOFF = ROOT / "reports/stage_d_recovery_v2_20260904_r3/d6_to_future_d7_handoff.json"


def test_audit_and_handoff_schema_digests_and_closed_boundaries() -> None:
    schema = json.loads(SCHEMA.read_text())
    audit = json.loads(AUDIT.read_text())
    handoff = json.loads(HANDOFF.read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(audit)
    Draft202012Validator(schema).validate(handoff)
    assert audit["canonical_digest"] == semantic_digest(audit)
    assert handoff["canonical_digest"] == semantic_digest(handoff)
    assert set(audit["authorization"].values()) == {False}
    assert set(audit["access_attestation"].values()) == {False}
    assert set(handoff["authorization"].values()) == {False}
    assert set(handoff["access_attestation"].values()) == {False}
    assert handoff["not_started"] == ["D7", "D8", "D9", "D10"]


def test_audit_inventory_is_exhaustive_and_uniquely_classified() -> None:
    audit = json.loads(AUDIT.read_text())
    records = audit["historical_records"] + audit["current_records"]
    assert len(records) == audit["summary"]["records_audited"] == 29
    assert len({record["record_id"] for record in records}) == 29
    observed = Counter(record["classification"] for record in records)
    assert observed == Counter({
        "CURRENT_VALID_PASS": 7,
        "HISTORICAL_PASS_STALE_BINDING": 3,
        "SYNTHETIC_OR_IMPLEMENTATION_ONLY_PASS": 4,
        "INVALID_OR_TAMPERED": 15,
    })
    assert audit["integrity_findings"]["forgery_found"] is False
    assert audit["integrity_findings"]["tamper_found"] is False
    assert audit["integrity_findings"]["wrong_schema_binding_records_found"] == 7
    assert audit["integrity_findings"]["incomplete_source_inventory_records_found"] == 7
    assert [record["stage"] for record in audit["current_records"]] == list(STAGES)


def test_r2_stage_chain_refs_blobs_sources_and_schema_uris() -> None:
    for stage in STAGES:
        result = validate_stage_bundle(stage, repo_root=ROOT)
        assert result["status"] == "PASS"
        assert result["reserve_accessed"] is False
        assert result["downstream_authorized"] is False
        directory = ROOT / "reports/stage_d_recovery_v2_20260904_r3" / stage.lower()
        for name in ("completion.json", "validation.json", "status.json"):
            document = json.loads((directory / name).read_text())
            assert (directory / document["$schema"]).resolve() == SCHEMA.with_name(
                "stage_d_recovery_record.schema.json"
            ).resolve()
