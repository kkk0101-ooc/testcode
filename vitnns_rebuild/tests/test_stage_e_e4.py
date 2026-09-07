"""Stage E E4 mixed-validity source validation and sealing tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_e_e4 as e4

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_e4_config_and_e3_custody_are_frozen() -> None:
    config = e4._validate_config(REPO_ROOT)
    assert config["canonical_digest"] == e4.EXPECTED_CONFIG_DIGEST
    assert config["scope"]["new_scientific_variant"] is False
    assert config["scope"]["new_model_or_image_compute"] is False
    assert config["scope"]["failure_disclosure_required"] is True
    completion = e4._validate_e3_custody(REPO_ROOT)
    assert completion["status"] == "PASS"
    assert completion["scientific_validity"] == "MIXED"
    assert completion["union"]["terminal_counts"] == {"expected": 3000, "valid": 2998, "failed": 2, "na": 0}


def test_e4_package_is_source_recomputable_and_complete() -> None:
    package = e4._build_package(REPO_ROOT)
    assert {key: value["row_count"] for key, value in package["tables"].items()} == {"patch_scale_sensitivity": 1500, "upsampling_sensitivity": 2000, "layer_head_descriptive": 2000, "prediction_strata": 60, "resource_and_qc": 7000}
    assert package["failure_identity"] == [
        (117, 11, 2, e4.e3.NNS_METHOD, "ig_conservation_failed_after_e3_r1_schedule"),
        (352, 11, 2, e4.e3.NNS_METHOD, "ig_conservation_failed_typed_terminal"),
    ]
    assert all(row.get("insertion_auc_mean") is None and row.get("deletion_auc_mean") is None for row in package["failures"])
    assert all("delta_fid" not in row for row in package["failures"])


def test_e4_tables_conform_to_frozen_output_schema() -> None:
    schema = json.loads((REPO_ROOT / "config/stage_e_output_table.schema.json").read_text())
    validator = Draft202012Validator(schema)
    for table in e4._build_package(REPO_ROOT)["tables"].values():
        assert not list(validator.iter_errors(table))


def test_e4_preconditions_preserve_mixed_validity() -> None:
    result = e4.verify_preconditions(REPO_ROOT, full=False)
    assert result["status"] == "PASS"
    assert result["upstream"]["e3"] == {"status": "PASS", "scientific_validity": "MIXED"}
    assert result["e4_started"] is False


def test_e4_fails_if_a_typed_failure_is_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    original = e4._table_from_source
    def tampered(repo: Path, relative: str, table_id: str):
        table, record = original(repo, relative, table_id)
        if table_id == "layer_head_descriptive": table["rows"] = [row for row in table["rows"] if row["validity_status"] != "FAILED"]; table["row_count"] = len(table["rows"])
        return table, record
    monkeypatch.setattr(e4, "_table_from_source", tampered)
    with pytest.raises(e4.StageEE4Error, match="failure disclosure"):
        e4._build_package(REPO_ROOT)


def test_e4_has_no_scientific_execution_surface() -> None:
    public = {name for name in dir(e4) if name.startswith(("run_", "start_", "execute_"))}
    assert public == set()
    source = (REPO_ROOT / "src/vitnns/stage_e_e4.py").read_text()
    assert "torch" not in source
    assert "load_model" not in source
    assert '"renders_png": False' in source
    assert 'predecessor_record["path"]' in source
    assert 'predecessor_record["sha256"]' in source
    assert "DELETE_ONLY_LISTED_E4_DERIVED_FILES_THEN_SOURCE_RECOMPUTE" in source
