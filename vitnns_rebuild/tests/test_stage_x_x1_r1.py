from __future__ import annotations

import copy
from pathlib import Path

import pytest

from vitnns import stage_x_x1_r1 as x1

REPO = Path(__file__).resolve().parents[1]


def test_artifact_envelopes_are_closed_and_self_digesting():
    schema = x1.strict_read(REPO / x1.SCHEMA_PATH)
    for filename, keys in x1.PAYLOAD_KEYS.items():
        value = x1.envelope("test", "PASS", {key: None for key in keys})
        x1.validate_envelope(value, filename)
        assert set(value) == {"$schema", "schema_version", "document_type", "stage_id", "status", "payload", "canonical_digest"}
        assert not list(x1.Draft202012Validator(schema).iter_errors(value))


def test_closed_schema_and_digest_tamper_are_rejected():
    value = x1.envelope("test", "PASS", {key: None for key in x1.PAYLOAD_KEYS["status.json"]})
    changed = copy.deepcopy(value); changed["unexpected"] = True
    with pytest.raises(x1.StageXX1R1Error):
        x1.validate_envelope(changed, "status.json")
    changed = copy.deepcopy(value); changed["payload"]["x2_started"] = True
    with pytest.raises(x1.StageXX1R1Error):
        x1.validate_envelope(changed, "status.json")


def test_atomic_terminal_refuses_existing_terminal_or_pending(tmp_path):
    value = x1.envelope("test", "PASS", {key: None for key in x1.PAYLOAD_KEYS["x1_r1.json"]})
    path = tmp_path / "completion/x1_r1.json"
    x1.publish_terminal(path, value)
    assert path.is_file() and not path.with_name("x1_r1.pending.json").exists()
    with pytest.raises(x1.StageXX1R1Error):
        x1.publish_terminal(path, value)
    path.unlink(); path.with_name("x1_r1.pending.json").write_text("pending")
    with pytest.raises(x1.StageXX1R1Error):
        x1.publish_terminal(path, value)


def test_environment_binding_is_source_recomputed():
    environment = x1._validate_environment(REPO)
    assert environment["vendored_sources"]["file_count"] == 46
    assert environment["vendored_sources"]["retained_bytes"] == 708444
    assert environment["zennit"]["version"] == "0.5.1"
