from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns.stage_f_f3 import (
    ATTACKS, FAILURE_CODES, F3_COMPLETION_SCHEMA_PATH,
    F3_FALSE_PASS_SCHEMA_PATH, StageFF3Error, _load_config, _load_run_spec,
    assert_f3_source_boundary, build_false_pass_audit,
    build_selection_manifests, validate_runtime_isolation,
    verify_startup_gates,
)

REPO = Path(__file__).resolve().parents[1]


def _read(path: str) -> dict:
    return json.loads((REPO / path).read_bytes())


def test_f3_config_run_spec_and_completion_schema_are_frozen():
    config = _load_config(REPO); run_spec = _load_run_spec(REPO)
    assert config["authorization"] == {"current": "F3_ONLY", "on_pass": "F4_ONLY", "f4_started": False}
    assert config["cross_layer"]["coordinates"] == [[0, 9], [4, 2], [6, 10], [11, 2]]
    assert config["cross_layer"]["expected_secondary_terminals"] == 3000
    assert config["cross_layer"]["expected_secondary_valid_maps"] == 2998
    assert run_spec["authorization"]["next_authorization_on_pass"] == "F4_ONLY"
    schema = _read(F3_COMPLETION_SCHEMA_PATH)
    assert schema["properties"]["next_authorization"]["const"] == "F4_ONLY"
    assert schema["properties"]["f4_started"]["const"] is False


def test_selection_is_exactly_d10_d1_based_and_keeps_duplicate_hits():
    selection, mapping = build_selection_manifests(REPO)
    assert selection["selection_access"] == {"jpeg_paths_statted_or_bytes_read": False, "map_paths_statted_or_bytes_read": False, "e3_map_outcomes_inspected": False}
    assert selection["median_effect_sample_id"] == "ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG"
    assert selection["median_effect_selection_order"] == 44
    observed = [(item["rule"], item["selection_order"]) for item in mapping["rules"]]
    assert observed == [("best_effect", 372), ("median_effect", 44), ("baseline_strong_nns_stronger", 372), ("weak_or_failure", 281), ("misclassified_or_difficult", 387)]
    assert mapping["duplicates"] == [{"sample_id": "ILSVRC2012_val/ILSVRC2012_val_00033768.JPEG", "rules": ["best_effect", "baseline_strong_nns_stronger"]}]
    assert len(mapping["render_order_unique_sample_ids"]) == 4


def test_fixed_audit_ids_and_order_match_protocol():
    selection, _ = build_selection_manifests(REPO)
    assert [item["sample_id"] for item in selection["fixed_audit_examples"]] == [
        "ILSVRC2012_val/ILSVRC2012_val_00004877.JPEG",
        "ILSVRC2012_val/ILSVRC2012_val_00028787.JPEG",
        "ILSVRC2012_val/ILSVRC2012_val_00024160.JPEG",
        "ILSVRC2012_val/ILSVRC2012_val_00016533.JPEG",
        "ILSVRC2012_val/ILSVRC2012_val_00014223.JPEG",
    ]
    assert all(item["label"] == "fixed audit example" for item in selection["fixed_audit_examples"])


def test_startup_gate_accepts_authorized_successor_without_dirty_requirement():
    gate = verify_startup_gates(REPO, require_clean=False, require_remote=False)
    assert gate["status"] == "PASS"
    assert gate["f2"]["archive_commit"] == "89dee5ad25b9cf3baa3a2ffdb2261ec8343a2ef6"
    assert gate["scientific_state"]["stage_e"]["scientific_validity"] == "MIXED"
    assert gate["d8_d9_d10_e3_e4_payload_arrays_or_jpegs_read"] is False


def test_f2_archive_or_completion_tamper_is_rejected(monkeypatch: pytest.MonkeyPatch):
    import vitnns.stage_f_f3 as module
    monkeypatch.setattr(module, "F2_COMMIT", "0" * 40)
    with pytest.raises(StageFF3Error, match="F2 archive"):
        module.verify_startup_gates(REPO, require_clean=False, require_remote=False)


def test_protocol_hash_tamper_is_rejected(monkeypatch: pytest.MonkeyPatch):
    import vitnns.stage_f_f3 as module
    monkeypatch.setattr(module, "EXPECTED_CROSS_PROTOCOL_SHA256", "0" * 64)
    with pytest.raises(StageFF3Error, match="cross_layer_protocol"):
        module.verify_startup_gates(REPO, require_clean=False, require_remote=False)


def test_runtime_and_source_guards_reject_forbidden_surfaces(monkeypatch: pytest.MonkeyPatch):
    assert validate_runtime_isolation()["status"] == "PASS"
    assert assert_f3_source_boundary(REPO)["status"] == "PASS"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(StageFF3Error, match="CUDA_VISIBLE_DEVICES"):
        validate_runtime_isolation()


def test_false_pass_audit_has_all_attacks_and_schema():
    audit = build_false_pass_audit(REPO)
    assert [item["attack"] for item in audit["attacks"]] == list(ATTACKS)
    assert all(item["result"] == "REJECTED" for item in audit["attacks"])
    Draft202012Validator(_read(F3_FALSE_PASS_SCHEMA_PATH)).validate(audit)
    assert FAILURE_CODES == {117: "ig_conservation_failed_after_e3_r1_schedule", 352: "ig_conservation_failed_typed_terminal"}
