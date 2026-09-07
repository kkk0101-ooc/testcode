"""CPU-only final sealing wrapper for the F3C2 v1 pilot package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .constants import REPO_ROOT
from . import stage_f_f3c2 as f3c2
from . import stage_f_f3c2_validate as numeric_validator

POSTRUN_VALIDATION_PATH = f3c2.REPORT_PATH + "/postrun_numeric_validation.json"
POSTRUN_TEST_PATH = f3c2.REPORT_PATH + "/postrun_test_report.json"
SEAL_SOURCE_FILES = (
    "src/vitnns/stage_f_f3c2_validate.py",
    "src/vitnns/stage_f_f3c2_seal.py",
    "tests/test_stage_f_f3c2_validate.py",
)


def _validate_postrun(repo: Path) -> dict[str, Any]:
    numerical = numeric_validator.validate_all(repo)
    report = f3c2._read(repo / POSTRUN_VALIDATION_PATH)
    f3c2._validate_signed(report, "F3C2 post-run numerical validation")
    if report.get("status") != "PASS" or report.get("validated_units") != 25 or report.get("validated_samples") != 5 or report.get("empirical_values_reported_or_displayed") is not False or report.get("scientific_comparison_or_inference_performed") is not False:
        raise f3c2.StageFF3C2Error("post-run numerical validation report failed")
    if numerical != {"status": "PASS", "validated_units": 25, "samples": 5}:
        raise f3c2.StageFF3C2Error("post-run numerical validation drifted")
    validator_record = report.get("validator_file", {})
    if validator_record != f3c2._file_record(repo, validator_record.get("path", "")):
        raise f3c2.StageFF3C2Error("post-run validator source binding drifted")
    return report


def record_postrun_tests(repo_root: Path | str, *, passed: int, failed: int, deselected: int, command: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if passed <= 0 or failed != 0:
        raise f3c2.StageFF3C2Error("post-run sealing tests are not passing")
    source = [f3c2._file_record(repo, path) for path in SEAL_SOURCE_FILES]
    report = f3c2._signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_postrun_test_report",
        "status": "PASS", "passed": passed, "failed": failed, "deselected": deselected,
        "command": command, "source_files": source,
        "source_tree_digest": f3c2.canonical_sha256({"files": source}),
        "runner_source_remained_bound_to_predecode_seal": True,
        "pilot_recomputed": False, "model_loaded": False, "jpeg_decoded": False,
    })
    f3c2._write_new(repo / POSTRUN_TEST_PATH, report)
    return report


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    base = f3c2.validate_outputs(repo)
    postrun = _validate_postrun(repo)
    tests = f3c2._read(repo / POSTRUN_TEST_PATH)
    f3c2._validate_signed(tests, "F3C2 post-run test report")
    source = [f3c2._file_record(repo, path) for path in SEAL_SOURCE_FILES]
    if tests.get("status") != "PASS" or tests.get("failed") != 0 or tests.get("source_files") != source or tests.get("source_tree_digest") != f3c2.canonical_sha256({"files": source}):
        raise f3c2.StageFF3C2Error("post-run sealing test/source binding failed")
    return f3c2._signed({
        **{key: value for key, value in base.items() if key != "canonical_digest"},
        "postrun_numeric_validation": f3c2._record(repo, POSTRUN_VALIDATION_PATH),
        "postrun_test_report": f3c2._record(repo, POSTRUN_TEST_PATH),
        "postrun_validated_units": postrun["validated_units"],
        "combined_tests_passed": tests["passed"], "combined_tests_failed": 0,
        "runner_source_remained_bound_to_predecode_seal": True,
    })


def finalize(repo_root: Path | str = REPO_ROOT, *, completed_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validation = validate_outputs(repo)
    f3c2._write_new(repo / f3c2.VALIDATION_PATH, validation)
    payload = f3c2._signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_payload_manifest", "status": "PASS",
        "records": [f3c2._record(repo, relative) for relative in (f3c2.STARTUP_PATH, f3c2.PILOT_MANIFEST_PATH, f3c2.UNIT_PLAN_PATH, f3c2.MODEL_BINDING_PATH, f3c2.SOURCE_BINDING_PATH, f3c2.RESOURCE_SPEC_PATH, f3c2.BLIND_SPEC_PATH, f3c2.ACCESS_POLICY_PATH, f3c2.PLAN_SEAL_PATH, f3c2.TERMINAL_UNION_PATH, f3c2.ATTEMPT_JOURNAL_PATH, f3c2.RESOURCE_LEDGER_PATH, f3c2.BUDGET_PATH, f3c2.ACCESS_AUDIT_PATH, f3c2.BLIND_REPORT_PATH, f3c2.TEST_PATH, POSTRUN_TEST_PATH, f3c2.FALSE_PASS_PATH, POSTRUN_VALIDATION_PATH, f3c2.VALIDATION_PATH)],
        "typed_terminal_count": 25, "numeric_payload_present_only_for_valid_terminals": True,
        "display_or_render_payload_present": False, "runner_source_bound_before_decode": True,
        "postrun_validator_source_bound_after_compute_without_recompute": True,
    })
    f3c2._write_new(repo / f3c2.PAYLOAD_PATH, payload)
    status = f3c2._signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_status", "status": "PASS",
        "terminal": True, "immutable": True, "implementation_validation": "PASS",
        "workflow_validation": "PASS", "package_validation": "PASS",
        "pilot_samples": "5/5", "method_terminals": "25/25 VALID",
        "nns_endpoint_parity": "5/5 PASS", "nns_conservation": "5/5 PASS",
        "full_model_ig_conservation": "5/5 PASS", "resource_budget_projection": "PASS",
        "pilot_sample_access": True, "pilot_selection_orders": list(f3c2.PILOT_ORDERS),
        "formal_sample_access": False, "order44_recomputed": False, "maps_rendered": False,
        "maps_manually_inspected": False, "statistical_inference_performed": False,
        "scientific_comparison_performed": False, "winner_or_go_no_go_produced": False,
        "f3c0_f3c1_f3h_and_other_sealed_evidence_modified": False,
        "f3c3_started": False, "f3c4_started": False, "f4_started": False,
        "next_authorization": "F3C3_FORMAL_ONLY", "git": {"commit": "SELF", "ref": f3c2.ARCHIVE_REF},
    })
    f3c2._write_new(repo / f3c2.STATUS_PATH, status)
    inventory_payload = f3c2._inventory(repo / f3c2.ROOT_PATH, exclude=("whole_root_inventory.json", "completion/f3c2.json", "completion/f3c2.pending.json"))
    inventory = f3c2._signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_whole_root_inventory",
        "status": "PASS", **inventory_payload,
    })
    f3c2._write_new(repo / f3c2.INVENTORY_PATH, inventory)
    completion = f3c2._signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_terminal_completion",
        "status": "PASS", "completed_at": completed_at,
        "implementation_validation": "PASS", "workflow_validation": "PASS", "package_validation": "PASS",
        "pilot_samples": "5/5", "method_terminals": {"VALID": 25, "FAILED": 0, "NA": 0, "incomplete": 0},
        "nns_endpoint_parity": "5/5 PASS", "nns_conservation": "5/5 PASS",
        "full_model_ig_conservation": "5/5 PASS", "resource_budget_projection": "PASS",
        "startup_gate": f3c2._record(repo, f3c2.STARTUP_PATH), "pilot_manifest": f3c2._record(repo, f3c2.PILOT_MANIFEST_PATH),
        "unit_execution_plan": f3c2._record(repo, f3c2.UNIT_PLAN_PATH), "model_preprocessing_binding": f3c2._record(repo, f3c2.MODEL_BINDING_PATH),
        "source_config_binding": f3c2._record(repo, f3c2.SOURCE_BINDING_PATH), "resource_accounting_specification": f3c2._record(repo, f3c2.RESOURCE_SPEC_PATH),
        "blind_continuation_specification": f3c2._record(repo, f3c2.BLIND_SPEC_PATH), "access_policy": f3c2._record(repo, f3c2.ACCESS_POLICY_PATH),
        "plan_seal": f3c2._record(repo, f3c2.PLAN_SEAL_PATH), "terminal_union": f3c2._record(repo, f3c2.TERMINAL_UNION_PATH),
        "attempt_journal": f3c2._record(repo, f3c2.ATTEMPT_JOURNAL_PATH), "resource_ledger": f3c2._record(repo, f3c2.RESOURCE_LEDGER_PATH),
        "budget_projection": f3c2._record(repo, f3c2.BUDGET_PATH), "access_audit": f3c2._record(repo, f3c2.ACCESS_AUDIT_PATH),
        "blind_continuation_report": f3c2._record(repo, f3c2.BLIND_REPORT_PATH), "postrun_numeric_validation": f3c2._record(repo, POSTRUN_VALIDATION_PATH),
        "payload_manifest": f3c2._record(repo, f3c2.PAYLOAD_PATH), "whole_root_inventory": f3c2._record(repo, f3c2.INVENTORY_PATH),
        "validation_report": f3c2._record(repo, f3c2.VALIDATION_PATH), "false_pass_audit": f3c2._record(repo, f3c2.FALSE_PASS_PATH),
        "test_report": f3c2._record(repo, f3c2.TEST_PATH), "postrun_test_report": f3c2._record(repo, POSTRUN_TEST_PATH),
        "status_record": f3c2._record(repo, f3c2.STATUS_PATH), "pilot_sample_access": True,
        "pilot_selection_orders": list(f3c2.PILOT_ORDERS), "formal_sample_access": False, "order44_recomputed": False,
        "maps_rendered": False, "maps_manually_inspected": False, "statistical_inference_performed": False,
        "scientific_comparison_performed": False, "winner_or_go_no_go_produced": False,
        "f3c0_f3c1_f3h_and_other_sealed_evidence_modified": False, "f3c3_started": False,
        "f3c4_started": False, "f4_started": False, "next_authorization": "F3C3_FORMAL_ONLY",
        "git": {"commit": "SELF", "ref": f3c2.ARCHIVE_REF},
    })
    temporary = (repo / f3c2.PENDING_PATH).with_name(".f3c2-terminal.tmp")
    f3c2._write_new(temporary, completion)
    os.replace(temporary, repo / f3c2.PENDING_PATH)
    os.replace(repo / f3c2.PENDING_PATH, repo / f3c2.COMPLETION_PATH)
    descriptor = os.open((repo / f3c2.COMPLETION_PATH).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return completion


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    result = f3c2.validate_completion(repo_root)
    repo = Path(repo_root).resolve()
    completion = f3c2._read(repo / f3c2.COMPLETION_PATH)
    for field in ("postrun_numeric_validation", "postrun_test_report"):
        if completion[field] != f3c2._record(repo, completion[field]["path"]):
            raise f3c2.StageFF3C2Error(f"F3C2 post-run binding drifted: {field}")
    _validate_postrun(repo)
    return result


__all__ = ["finalize", "record_postrun_tests", "validate_completion", "validate_outputs"]
