from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns import recovery_c10_launch as recovery
from vitnns.stage_c_provenance import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_recovery_c10_schema_is_closed_and_scope_is_narrow() -> None:
    schema = json.loads(recovery.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    scope = schema["properties"]["scope"]["properties"]
    assert scope["c10_aggregation_authorized"]["const"] is True
    assert scope["holm_authorized"]["const"] is False
    assert scope["stage_d_execution_authorized"]["const"] is False
    assert scope["stage_d_analysis_authorized"]["const"] is False
    assert scope["stage_d_reserve_access_authorized"]["const"] is False


def test_recovery_c10_upstream_rejects_historical_c9_alias() -> None:
    value = {
        "$schema": recovery.DEFAULT_SCHEMA_PATH.name,
        "schema_version": 1, "document_type": recovery.DOCUMENT_TYPE,
        "record_id": recovery.RECORD_ID, "status": recovery.STATUS,
        "amendment": {}, "producer_git": {},
        "launch_record_commit": "SELF",
        "upstream": {
            "c8_salvage": recovery._c8_binding(),
            "c9": recovery._c9_binding(),
        },
        "model": {
            "identity": "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718",
            "state_dict_sha256": "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d",
            "canonical_tensor_digest": "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd",
            "historical_artifact_recovered": False,
        },
        "contracts": {}, "outputs": {},
        "scope": {
            "c10_aggregation_authorized": True,
            "source_recompute_required": True,
            "bootstrap_authorized": True, "decision_authorized": True,
            "holm_authorized": False,
            "holm_status_required": "HOLM_BLOCKED_UNDERSPECIFIED",
            "stage_d_execution_authorized": False,
            "stage_d_analysis_authorized": False,
            "stage_d_reserve_access_authorized": False,
        },
    }
    value["canonical_digest"] = canonical_sha256(value)
    value["upstream"]["c9"]["producer_commit"] = (
        "d8190b130d00d3b19fe7a586103a3a9cdb55e43d"
    )
    value["canonical_digest"] = canonical_sha256({
        key: item for key, item in value.items() if key != "canonical_digest"
    })
    with pytest.raises(recovery.RecoveryC10LaunchError, match="upstream"):
        recovery._validate_document(value)


def test_recovery_c10_bindings_name_new_c9_and_recovery_model() -> None:
    c9 = recovery._c9_binding()
    assert c9["producer_commit"] == "0d88fbb8b71c604d3dab94db19725e01b7519ae5"
    assert c9["authority_execution_commit"] == "c66e5224e87924253f10a9f5fead006fcc5b5ac7"
    assert c9["whole_root_inventory_canonical_digest"] == (
        "6ab8ee024ce7d5bfa3c6a408ec118a9eeac7960d046d7001d8019fb8ab6650c8"
    )
    assert "d8190" not in json.dumps(c9)
    assert "2e718" not in json.dumps(c9)
