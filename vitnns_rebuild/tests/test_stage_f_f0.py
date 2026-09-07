from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns.stage_d_provenance import canonical_sha256
from vitnns.stage_f_f0 import (
    ATTACKS, F0_ROOT_PATH, StageFF0Error, _inventory, _signed,
    validate_runtime_isolation, verify_upstream,
)
from vitnns.stage_f_renderer import (
    COMPLETION_SCHEMA_PATH, FALSE_PASS_SCHEMA_PATH, VALIDATION_SCHEMA_PATH,
)

REPO = Path(__file__).resolve().parents[1]


def test_f0_upstream_gate_preserves_e4_d10_and_mixed_validity():
    result = verify_upstream(REPO)
    assert result["status"] == "PASS"
    assert result["e4"] == {"status":"PASS","package_completeness":"PASS","workflow_integrity":"PASS","scientific_validity":"MIXED","next_authorization":"F0_ONLY"}
    assert result["recipes"]["new_statistics_computed"] is False


def test_false_pass_attack_catalog_is_exact_and_complete():
    assert len(ATTACKS) == len(set(ATTACKS)) == 12
    assert "terminal_pass_before_validator" in ATTACKS
    assert "f0_completion_authorized_f2_f3_or_f4" in ATTACKS


def test_formal_runtime_isolation_requires_hidden_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert validate_runtime_isolation()["status"] == "PASS"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(StageFF0Error, match="CUDA_VISIBLE_DEVICES"):
        validate_runtime_isolation()


def test_signing_replaces_preexisting_canonical_digest():
    document = _signed({"value": 1, "canonical_digest": "stale"})
    assert document == {"value": 1, "canonical_digest": "48208f9428d64634bd8e28ff345bf0eab60d53c18fa2fbdb0b9bc1e84df2b5f6"}


def test_whole_root_document_digest_excludes_files_only_helper_digest(tmp_path: Path):
    (tmp_path / "payload.txt").write_text("synthetic")
    rebuilt = _inventory(tmp_path)
    unsigned = {"schema_version":1,"document_type":"stage_f_f0_whole_root_inventory","root":F0_ROOT_PATH,"scope":"all regular F0 files except self-referential inventory status and completion receipts","files":rebuilt["files"],"file_count":rebuilt["file_count"],"total_bytes":rebuilt["total_bytes"]}
    document = _signed({**unsigned, "canonical_digest": rebuilt["canonical_digest"]})
    assert document["canonical_digest"] == canonical_sha256(unsigned)


def test_terminal_schemas_fail_closed():
    schemas = {name: json.loads((REPO / path).read_text()) for name, path in {"completion":COMPLETION_SCHEMA_PATH,"validation":VALIDATION_SCHEMA_PATH,"audit":FALSE_PASS_SCHEMA_PATH}.items()}
    for schema in schemas.values(): Draft202012Validator.check_schema(schema)
    fake_completion = {"schema_version":1,"document_type":"stage_f_f0_terminal_completion","stage_id":"F0","status":"PASS","next_authorization":"F2_ONLY"}
    assert list(Draft202012Validator(schemas["completion"]).iter_errors(fake_completion))
    fake_validation = {"schema_version":1,"document_type":"stage_f_f0_validation_report","status":"PASS","formal_scientific_figures_rendered":1}
    assert list(Draft202012Validator(schemas["validation"]).iter_errors(fake_validation))
    fake_audit = {"schema_version":1,"document_type":"stage_f_f0_false_pass_audit","status":"PASS","attacks":[]}
    assert list(Draft202012Validator(schemas["audit"]).iter_errors(fake_audit))


def test_upstream_ref_or_inventory_drift_is_rejected(monkeypatch: pytest.MonkeyPatch):
    import vitnns.stage_f_f0 as f0
    real = f0._git
    monkeypatch.setattr(f0, "_git", lambda repo, *args: "0" * 40 if args[:2] == ("rev-parse", "refs/archive/stage-e-e4-completion^{commit}") else real(repo, *args))
    with pytest.raises(StageFF0Error, match="archive ref drifted"):
        verify_upstream(REPO)
