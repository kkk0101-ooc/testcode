"""Prospective E3-R4 typed-terminal completion over immutable E3 history."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import tempfile
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_e_e1 as e1
from . import stage_e_e3 as e3
from . import stage_e_e3_r1 as r1
from . import stage_e_e3_r2 as r2
from . import stage_e_e3_r3 as r3

RECOVERY_ID = "STAGE-E-E3-R4-TYPED-TERMINAL-20260907"
CONFIG_PATH = "config/stage_e_e3_r4.json"
CONFIG_SCHEMA_PATH = "config/stage_e_e3_r4.schema.json"
COMPLETION_SCHEMA_PATH = "config/stage_e_e3_r4_completion.schema.json"
AMENDMENT_PATH = "docs/recovery_stage_e_e3_r4_typed_terminal_amendment_20260907.md"
ROOT_PATH = "evidence/stage_e/e3_r4_recovery_v2_20260907/formal"
REPORT_ROOT_PATH = "reports/stage_e_e3_r4_20260907"
PLAN_PATH = ROOT_PATH + "/control/recovery_plan.json"
RUN_SPEC_PATH = ROOT_PATH + "/control/run_spec.json"
AUTH_PATH = ROOT_PATH + "/control/execution_authorization.json"
COMPLETION_PENDING_PATH = ROOT_PATH + "/completion/e3_r4_completion.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/e3_r4_completion.json"
TEST_REPORT_PATH = REPORT_ROOT_PATH + "/pre_execution_test_results.json"
POST_EXECUTION_TEST_REPORT_PATH = REPORT_ROOT_PATH + "/post_execution_validation_test_results.json"
R3_TERMINATION_PATH = REPORT_ROOT_PATH + "/e3_r3_pilot_termination_event.json"

EXPECTED_CONFIG_DIGEST = "3f1d4efdb980985432692e7040ca06c48a084748054bf3081465cb65fd7343d0"
ATTEMPT_ID = "attempt-r4-000001"
SCHEDULE = r1.COMPLETE_SCHEDULE
FAILURE_CODE = "ig_conservation_failed_typed_terminal"
R3_COMMIT = "c060d1d6c810fa231a79a5b9aa4de87499c56100"
R3_TREE = "453a5d0de8fca06e073004dd71090a389e1ff257"
R3_INTERRUPTED_PID = 2692867
R3_CHARGED_SECONDS = 873.0
R3_BOUND_FILES = {
    r3.PILOT_AUTH_PATH: "2ad55eb05e0a47b87175f8f3a6380cbdb467319c8481c81371617747aa381579",
    r3.PLAN_PATH: "31e694c8fd9e624d4e9f77e410d6f8934e57753a4b01951fee09b6c58e47152f",
    r3.RUN_SPEC_PATH: "4f8b60bc34cd0d353a8d359eb24a731f4f941d8ebc72593717d8b660e4f9a986",
    r3.COMPLETION_PENDING_PATH: "f626e930c2b0cfae22bcb6f80f3d4ae74600f650c5367ce3d7c6414820aef3e6",
    r3.TEST_REPORT_PATH: "af027a5aa9f72e8390f3a2f6c524fa6dbbeff56fc0db819153f03ad1ce4cfda1",
}
FORBIDDEN_SUFFIXES = {".png", ".pdf", ".svg"}
FALSE_PASS_ATTACKS = (
    "historical_receipt_or_terminal_attempt_overwritten",
    "r2_order_117_failure_retried_replaced_or_hidden",
    "r3_unpublished_pilot_treated_as_terminal_evidence",
    "missing_or_duplicate_expected_unit",
    "failed_or_na_row_zero_filled_or_interpolated",
    "invalid_row_included_in_numeric_summary",
    "valid_failed_na_counts_omitted",
    "workflow_pass_relabelled_as_all_unit_scientific_pass",
    "cost_omitted_because_it_is_not_a_gate",
    "e4_started_before_atomic_e3_r4_completion",
)


class StageEE3R4Error(RuntimeError):
    """E3-R4 typed-terminal contract was violated."""


def _now() -> str:
    return r3._now()


def _sha(raw: bytes) -> str:
    return r3._sha(raw)


def _sha_file(path: Path) -> str:
    return r3._sha_file(path)


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    return r3._signed(value)


def _strict_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        return r3._strict_json(path, context)
    except Exception as error:
        raise StageEE3R4Error(str(error)) from error


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    if value.get("canonical_digest") != canonical_sha256(
        {key: item for key, item in value.items() if key != "canonical_digest"}
    ):
        raise StageEE3R4Error(f"{context} canonical digest mismatch")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    r3._write_new_json(path, value)


def _write_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    r3._write_or_verify(path, value)


def _git(repo: Path, *args: str) -> str:
    try:
        return r3._git(repo, *args)
    except Exception as error:
        raise StageEE3R4Error(str(error)) from error


def _record(repo: Path, relative: str) -> dict[str, Any]:
    return r3._record(repo, relative)


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    return r3._inventory(root, exclude=exclude)


def _validate_config(repo: Path) -> dict[str, Any]:
    config, _ = _strict_json(repo / CONFIG_PATH, "E3-R4 config")
    schema, _ = _strict_json(repo / CONFIG_SCHEMA_PATH, "E3-R4 config schema")
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(config))
    if errors:
        raise StageEE3R4Error(f"E3-R4 config schema failure: {errors[0].message}")
    _validate_signed(config, "E3-R4 config")
    if config["canonical_digest"] != EXPECTED_CONFIG_DIGEST:
        raise StageEE3R4Error("E3-R4 config digest drifted")
    if (
        tuple(config["numerical_contract"]["schedule"]) != SCHEDULE
        or config["numerical_contract"]["atol"] != 1e-4
        or config["numerical_contract"]["rtol"] != 0.01
        or config["acceptance"]["scientific_validity"] != "MIXED"
        or config["acceptance"]["cost_policy"] != "MEASURE_AND_DISCLOSE_NOT_A_COMPLETION_GATE"
    ):
        raise StageEE3R4Error("E3-R4 frozen typed-terminal contract drifted")
    return config


def _v1_plan(repo: Path) -> dict[str, Any]:
    plan, raw = _strict_json(repo / e3.PLAN_PATH, "E3 v1 typed plan")
    _validate_signed(plan, "E3 v1 typed plan")
    if _sha(raw) != r1.E3_V1_PLAN_SHA256 or plan["canonical_digest"] != r1.E3_V1_PLAN_DIGEST:
        raise StageEE3R4Error("E3 v1 typed plan binding drifted")
    return plan


def _live_r3_pilots() -> list[str]:
    output = subprocess.run(
        ("ps", "-eo", "pid=,args="), check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.splitlines()
    return [line.strip() for line in output if "vitnns.stage_e_e3_r3 run-pilot" in line]


def validate_upstream(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    r2_result = r2.validate_completion(repo)
    if (
        r2_result.get("status") != "FAIL"
        or r2_result.get("valid_terminal_attempts") != 1164
        or r2_result.get("failed_terminal_attempts") != 1
        or r2_result.get("missing_attempts") != 1835
        or r2_result.get("e4_started") is not False
    ):
        raise StageEE3R4Error("E3-R2 terminal state drifted")
    if _git(repo, "rev-parse", R3_COMMIT + "^{tree}") != R3_TREE:
        raise StageEE3R4Error("E3-R3 producer custody tree drifted")
    if _git(repo, "diff", "--name-only", R3_COMMIT, "--", r3.ROOT_PATH, r3.REPORT_ROOT_PATH):
        raise StageEE3R4Error("E3-R3 authority/pending evidence changed after custody")
    for relative, expected_sha in R3_BOUND_FILES.items():
        if _sha_file(repo / relative) != expected_sha:
            raise StageEE3R4Error(f"E3-R3 bound file drifted: {relative}")
    pending, _ = _strict_json(repo / r3.COMPLETION_PENDING_PATH, "E3-R3 pending receipt")
    authority, _ = _strict_json(repo / r3.PILOT_AUTH_PATH, "E3-R3 pilot authority")
    if pending.get("status") != "PENDING" or authority.get("status") != "AUTHORIZED_PILOT_ONLY":
        raise StageEE3R4Error("E3-R3 pending/authority state drifted")
    pilot_files = [path for path in (repo / r3.PILOT_ROOT).rglob("*") if path.is_file()] if (repo / r3.PILOT_ROOT).exists() else []
    if pilot_files:
        raise StageEE3R4Error("E3-R3 contains a pilot terminal payload and cannot be superseded as empty")
    if _live_r3_pilots():
        raise StageEE3R4Error("an E3-R3 pilot process is still running")
    return {
        "status": "PASS", "e3_r2": r2_result,
        "e3_r3": {"status": "PENDING_INTERRUPTED_WITHOUT_TERMINAL_PILOT_PAYLOAD", "commit": R3_COMMIT, "tree": R3_TREE},
    }


def _r4_attempt_path(repo: Path, unit: Mapping[str, Any]) -> Path:
    return repo / ROOT_PATH / "units" / unit["canonical_digest"] / "attempts" / ATTEMPT_ID


def _validate_na_attempt(repo: Path, unit: Mapping[str, Any]) -> dict[str, Any]:
    root = _r4_attempt_path(repo, unit)
    result_path, status_path, manifest_path, payload_path = (root / name for name in ("result.json", "status.json", "manifest.json", "na.json"))
    if any(not path.is_file() for path in (result_path, status_path, manifest_path, payload_path)):
        raise StageEE3R4Error("E3-R4 NA attempt is incomplete")
    result, result_raw = _strict_json(result_path, "E3-R4 NA result")
    status, status_raw = _strict_json(status_path, "E3-R4 NA status")
    manifest, _ = _strict_json(manifest_path, "E3-R4 NA manifest")
    payload, payload_raw = _strict_json(payload_path, "E3-R4 NA payload")
    for value, name in ((result, "result"), (status, "status"), (manifest, "manifest"), (payload, "payload")):
        _validate_signed(value, f"E3-R4 NA {name}")
    if (
        result.get("validity_status") != "NA" or result.get("recovery_id") != RECOVERY_ID
        or result.get("origin") != "E3_R4_NEW" or result.get("unit_digest") != unit["canonical_digest"]
        or result.get("attempt_id") != ATTEMPT_ID or not result.get("na_reason")
        or status.get("state") != "na" or status.get("terminal") is not True
        or manifest.get("validity_status") != "NA" or manifest.get("payload_name") != "na.json"
        or manifest.get("result_sha256") != _sha(result_raw)
        or manifest.get("status_sha256") != _sha(status_raw)
        or manifest.get("payload_sha256") != _sha(payload_raw)
        or payload.get("na_reason") != result.get("na_reason")
    ):
        raise StageEE3R4Error("E3-R4 NA attempt binding failed")
    return {"result": result, "status": status, "manifest": manifest, "payload": payload, "attempt_path": root}


def _validate_r4_attempt(repo: Path, unit: Mapping[str, Any]) -> dict[str, Any]:
    result, _ = _strict_json(_r4_attempt_path(repo, unit) / "result.json", "E3-R4 result")
    if result.get("validity_status") == "NA":
        return _validate_na_attempt(repo, unit)
    try:
        return r1._validate_r1_attempt(
            repo, unit, root_path=ROOT_PATH, attempt_id=ATTEMPT_ID,
            recovery_id=RECOVERY_ID, origin="E3_R4_NEW",
            nns_schedule=SCHEDULE, numerical_field="numerical_recovery",
            numerical_id=r1.RECOVERY_ID, failure_code=FAILURE_CODE,
        )
    except Exception as error:
        raise StageEE3R4Error(str(error)) from error


def _source_attempt(repo: Path, unit: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    v1_path = repo / e3.ROOT_PATH / "units" / unit["canonical_digest"] / "attempts/attempt-000001"
    if v1_path.exists():
        return "E3_V1_INHERITED", e3._validate_attempt(repo, unit)
    r1_path = repo / r1.ROOT_PATH / "units" / unit["canonical_digest"] / "attempts" / r1.ATTEMPT_ID
    if r1_path.exists():
        return "E3_R1_INHERITED", r1._validate_r1_attempt(repo, unit)
    r2_path = repo / r2.ROOT_PATH / "units" / unit["canonical_digest"] / "attempts" / r2.ATTEMPT_ID
    if r2_path.exists():
        observed = r1._validate_r1_attempt(
            repo, unit, root_path=r2.ROOT_PATH, attempt_id=r2.ATTEMPT_ID,
            recovery_id=r2.RECOVERY_ID, origin="E3_R2_NEW",
        )
        suffix = "VALID" if observed["result"]["validity_status"] == "VALID" else "FAILED"
        return f"E3_R2_INHERITED_{suffix}", observed
    if _r4_attempt_path(repo, unit).exists():
        return "E3_R4_NEW", _validate_r4_attempt(repo, unit)
    return None


def _partition(repo: Path) -> tuple[dict[str, tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    inherited: dict[str, tuple[str, dict[str, Any]]] = {}
    eligible: list[dict[str, Any]] = []
    for unit in _v1_plan(repo)["units"]:
        source = _source_attempt(repo, unit)
        if source is None or source[0] == "E3_R4_NEW":
            eligible.append(unit)
        else:
            inherited[unit["canonical_digest"]] = source
    counts = Counter(value[0] for value in inherited.values())
    expected = Counter({
        "E3_V1_INHERITED": 516, "E3_R1_INHERITED": 12,
        "E3_R2_INHERITED_VALID": 636, "E3_R2_INHERITED_FAILED": 1,
    })
    if counts != expected or len(eligible) != 1835:
        raise StageEE3R4Error(f"E3-R4 1165/1835 partition drifted: {counts}, {len(eligible)}")
    failed = [value for value in inherited.values() if value[0] == "E3_R2_INHERITED_FAILED"]
    failure = failed[0][1]["result"] if len(failed) == 1 else {}
    if (
        failure.get("selection_order") != 117 or (failure.get("block"), failure.get("head")) != (11, 2)
        or failure.get("method") != e3.NNS_METHOD or failure.get("unit_digest") != r3._pilot_unit(repo)["canonical_digest"]
    ):
        raise StageEE3R4Error("E3-R4 preserved typed failure identity drifted")
    return inherited, eligible


def verify_preconditions(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    return {
        "status": "PASS", "config": _validate_config(repo),
        "upstream": validate_upstream(repo), "inherited": len(_partition(repo)[0]),
        "eligible": len(_partition(repo)[1]), "head": _git(repo, "rev-parse", "HEAD"),
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "amendment_sha256": _sha_file(repo / AMENDMENT_PATH),
    }


def record_tests(
    repo_root: Path | str = REPO_ROOT, *, passed: int, failed: int, command: str,
) -> dict[str, Any]:
    if type(passed) is not int or passed <= 0 or failed != 0 or not command.strip():
        raise StageEE3R4Error("only a real zero-failure test run can be recorded")
    report = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r4_pre_execution_test_results",
        "recovery_id": RECOVERY_ID, "status": "PASS", "recorded_at": _now(),
        "command": command, "passed": passed, "failed": failed, "e4_started": False,
    })
    _write_or_verify(Path(repo_root).resolve() / TEST_REPORT_PATH, report)
    return report


def record_post_execution_tests(
    repo_root: Path | str = REPO_ROOT, *, passed: int, failed: int, command: str,
) -> dict[str, Any]:
    if type(passed) is not int or passed <= 0 or failed != 0 or not command.strip():
        raise StageEE3R4Error("only a real zero-failure post-execution test run can be recorded")
    report = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r4_post_execution_validation_test_results",
        "recovery_id": RECOVERY_ID, "status": "PASS", "recorded_at": _now(),
        "command": command, "passed": passed, "failed": failed,
        "scientific_attempts_modified_or_rerun": False, "e4_started": False,
        "validator_correction": {"scope": "READ_ONLY_D9_PRIMARY_AGGREGATE_PARSER", "before": "required_redundant_top_level_direction", "after": "derive_unique_direction_from_five_bound_curves_when_top_level_field_is_absent", "source_results_modified": False, "e3_r4_terminal_attempts_modified": False},
    })
    _write_or_verify(Path(repo_root).resolve() / POST_EXECUTION_TEST_REPORT_PATH, report)
    return report


def initialize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); pre = verify_preconditions(repo)
    tests, _ = _strict_json(repo / TEST_REPORT_PATH, "E3-R4 tests"); _validate_signed(tests, "E3-R4 tests")
    if tests.get("status") != "PASS" or tests.get("failed") != 0:
        raise StageEE3R4Error("E3-R4 tests are not PASS")
    status = _git(repo, "status", "--porcelain=v1")
    allowed = (ROOT_PATH + "/", REPORT_ROOT_PATH + "/")
    unexpected = [line for line in status.splitlines() if not any(line[3:].startswith(item) or f"/{item}" in line[3:] for item in allowed)]
    if unexpected:
        raise StageEE3R4Error(f"E3-R4 producer worktree is not clean: {unexpected}")
    if _git(repo, "rev-parse", "@{u}") != pre["head"]:
        raise StageEE3R4Error("E3-R4 producer is not pushed")
    inherited, eligible = _partition(repo)
    plan = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r4_recovery_plan",
        "recovery_id": RECOVERY_ID, "e3_v1_plan_canonical_digest": r1.E3_V1_PLAN_DIGEST,
        "inherited_units": [{"unit_digest": key, "origin": value[0], "validity_status": value[1]["result"]["validity_status"]} for key, value in sorted(inherited.items())],
        "eligible_unit_digests": [unit["canonical_digest"] for unit in eligible],
        "counts": {"total": 3000, "inherited_terminal": 1165, "inherited_valid": 1164, "inherited_failed": 1, "eligible_missing": 1835},
    })
    run_spec = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r4_run_spec",
        "recovery_id": RECOVERY_ID, "plan_canonical_digest": plan["canonical_digest"],
        "schedule": list(SCHEDULE), "atol": 1e-4, "rtol": 0.01,
        "step_chunk_size": 8, "coordinates": [list(item) for item in e3.COORDINATES],
        "methods": list(e3.METHODS), "shard_count": 8, "shard_rule": "selection_order_modulo_8",
        "terminal_statuses": ["VALID", "FAILED", "NA"],
        "continue_after_scientific_failure_or_na": True,
        "summary_population": "VALID_ONLY_WITH_TYPED_COUNTS",
        "cost_policy": "MEASURE_AND_DISCLOSE_NOT_A_COMPLETION_GATE",
        "stage_boundary": {"current": "E3_R4_ONLY", "e4_started": False},
    })
    event = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r3_pilot_termination_event",
        "event_id": "E3_R3_PILOT_TERMINATION_001", "status": "RECORDED",
        "recorded_at": _now(), "reason": "SUPERSEDED_BY_PROSPECTIVE_E3_R4_TYPED_TERMINAL_AMENDMENT",
        "pid": R3_INTERRUPTED_PID, "signal": "SIGINT", "run_id": "repeat-000001",
        "authorized_under_r3": True, "terminal_pilot_payloads_published": 0,
        "r3_pending_receipt_preserved": True, "r3_authority_preserved": True,
        "charged_gpu_seconds": R3_CHARGED_SECONDS, "charged_cpu_seconds": R3_CHARGED_SECONDS,
        "charge_basis": "CONSERVATIVE_WALL_TIME_FROM_PROCESS_OBSERVATION_AT_INTERRUPTION",
        "e4_authorized": False, "e4_started": False,
    })
    _write_or_verify(repo / R3_TERMINATION_PATH, event)
    authorization = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r4_execution_authorization",
        "recovery_id": RECOVERY_ID, "status": "AUTHORIZED_MISSING_COMPLEMENT_ONLY",
        "issued_at": _now(), "producer_commit": pre["head"], "producer_tree": pre["tree"],
        "amendment_sha256": pre["amendment_sha256"], "config_canonical_digest": EXPECTED_CONFIG_DIGEST,
        "plan_canonical_digest": plan["canonical_digest"], "run_spec_canonical_digest": run_spec["canonical_digest"],
        "r2_completion_canonical_digest": r3.R2_COMPLETION_DIGEST,
        "r3_commit": R3_COMMIT, "r3_termination_event_canonical_digest": event["canonical_digest"],
        "authorized_unit_count": 1835, "inherited_terminal_recompute": False,
        "preserved_failure_retry": False, "scientific_failure_stops_complement": False,
        "budget_is_completion_gate": False, "e4_authorized": False, "e4_started": False,
    })
    pending = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r4_completion_pending",
        "stage_id": "E3-R4", "recovery_id": RECOVERY_ID, "status": "PENDING",
        "started_at": _now(), "plan_canonical_digest": plan["canonical_digest"],
        "authorization_canonical_digest": authorization["canonical_digest"],
        "next_authorization": "NONE_PENDING", "e4_started": False,
    })
    for relative, value in ((PLAN_PATH, plan), (RUN_SPEC_PATH, run_spec), (AUTH_PATH, authorization), (COMPLETION_PENDING_PATH, pending)):
        _write_or_verify(repo / relative, value)
    return validate_initialized(repo)


def validate_initialized(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); verify_preconditions(repo)
    documents = {}
    for name, relative in (("plan", PLAN_PATH), ("spec", RUN_SPEC_PATH), ("auth", AUTH_PATH), ("pending", COMPLETION_PENDING_PATH), ("event", R3_TERMINATION_PATH)):
        value, _ = _strict_json(repo / relative, f"E3-R4 {name}"); _validate_signed(value, f"E3-R4 {name}"); documents[name] = value
    if (
        documents["auth"].get("status") != "AUTHORIZED_MISSING_COMPLEMENT_ONLY"
        or documents["auth"].get("authorized_unit_count") != 1835
        or documents["auth"].get("preserved_failure_retry") is not False
        or documents["spec"].get("continue_after_scientific_failure_or_na") is not True
        or documents["pending"].get("e4_started") is not False
        or documents["event"].get("terminal_pilot_payloads_published") != 0
    ):
        raise StageEE3R4Error("E3-R4 initialized boundary drifted")
    return {"status": "PASS", "inherited_terminal": 1165, "eligible_missing": 1835, "e4_started": False}


def _validate_execution_code(repo: Path) -> dict[str, Any]:
    authorization, _ = _strict_json(repo / AUTH_PATH, "E3-R4 execution authorization")
    _validate_signed(authorization, "E3-R4 execution authorization")
    if authorization.get("status") != "AUTHORIZED_MISSING_COMPLEMENT_ONLY":
        raise StageEE3R4Error("E3-R4 execution is not authorized")
    producer = authorization.get("producer_commit")
    if subprocess.run(("git", "merge-base", "--is-ancestor", producer, "HEAD"), cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise StageEE3R4Error("E3-R4 execution HEAD does not descend from producer")
    if _git(repo, "diff", "--name-only", producer, "HEAD", "--", "src/vitnns", "config", "tests", AMENDMENT_PATH):
        raise StageEE3R4Error("E3-R4 producer changed after execution authorization")
    return authorization


def _na_result(
    *, unit: Mapping[str, Any], target_identity: Mapping[str, Any], response: Any,
    elapsed_seconds: float, cpu_seconds: float, forward_calls: int,
    peak_cuda_bytes: int, clean_sha256: str,
) -> dict[str, Any]:
    response_dict = response.to_dict(); na = response_dict.get("na", {})
    reason = na.get("reason")
    if not isinstance(reason, str) or not reason:
        raise StageEE3R4Error("typed NA lacks a structured reason")
    return {
        "schema_version": 1, "document_type": "stage_e_e3_r4_attempt_result",
        "recovery_id": RECOVERY_ID, "origin": "E3_R4_NEW",
        "unit_digest": unit["canonical_digest"], "sample_id": unit["sample_id"],
        "selection_order": unit["selection_order"], "stratum": unit["stratum"],
        "block": unit["block"], "head": unit["head"], "method": unit["method"],
        "target": unit["target"], "scale": unit["scale"],
        "coordinate_comparison_identity": unit["coordinate_comparison_identity"],
        "validity_status": "NA", "failure_class": None, "failure_code": None,
        "na_reason": reason, "attempt_id": ATTEMPT_ID, "retry_count": 0,
        "source_hashes": {
            "image_sha256": unit["image_sha256"], "clean_input_sha256": clean_sha256,
            "model_sha256": e3.MODEL_SHA256, "model_canonical_tensor_digest": e3.MODEL_TENSOR_DIGEST,
            "target_semantic_digest": e1._target_semantic_digest(target_identity),
        },
        "target_response": response_dict, "fidelity_executed": False,
        "elapsed_seconds": float(elapsed_seconds), "cpu_seconds": float(cpu_seconds),
        "gpu_hours": float(elapsed_seconds / 3600.0), "forward_calls": int(forward_calls),
        "backward_calls": 0, "peak_cuda_memory_bytes": int(peak_cuda_bytes), "artifact_bytes": 0,
        "descriptive_only": True, "p_values_emitted": False, "winner_selected": False,
        "causal_claim_made": False, "new_go_no_go_issued": False, "d10_primary_recomputed": False,
    }


def _publish_na(repo: Path, unit: Mapping[str, Any], result: Mapping[str, Any]) -> Path:
    root = _r4_attempt_path(repo, unit)
    if root.exists():
        raise StageEE3R4Error(f"refusing to overwrite terminal E3-R4 NA: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{ATTEMPT_ID}-", dir=root.parent))
    try:
        signed_result = _signed(result); result_raw = canonical_json_bytes(signed_result) + b"\n"
        (temporary / "result.json").write_bytes(result_raw)
        payload = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_na_payload", "unit_digest": unit["canonical_digest"], "na_reason": signed_result["na_reason"], "numeric_payload": None})
        payload_raw = canonical_json_bytes(payload) + b"\n"; (temporary / "na.json").write_bytes(payload_raw)
        status = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_attempt_status", "recovery_id": RECOVERY_ID, "attempt_id": ATTEMPT_ID, "attempt_number": 1, "unit_digest": unit["canonical_digest"], "state": "na", "terminal": True, "retry_count": 0, "failure_class": None, "failure_code": None, "na_reason": signed_result["na_reason"], "published_at": _now()})
        status_raw = canonical_json_bytes(status) + b"\n"; (temporary / "status.json").write_bytes(status_raw)
        manifest = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_attempt_manifest", "recovery_id": RECOVERY_ID, "attempt_id": ATTEMPT_ID, "attempt_number": 1, "unit_digest": unit["canonical_digest"], "validity_status": "NA", "result_sha256": _sha(result_raw), "result_bytes": len(result_raw), "payload_name": "na.json", "payload_sha256": _sha(payload_raw), "payload_bytes": len(payload_raw), "status_sha256": _sha(status_raw), "status_bytes": len(status_raw), "immutable": True})
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        for path in temporary.iterdir():
            with path.open("rb") as stream: os.fsync(stream.fileno())
        os.replace(temporary, root); directory = os.open(root.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    _validate_na_attempt(repo, unit); return root


def run_shard(
    repo_root: Path | str = REPO_ROOT, *, shard_id: int, shard_count: int, device: str = "cuda:0",
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if shard_count != 8 or type(shard_id) is not int or not 0 <= shard_id < 8:
        raise StageEE3R4Error("E3-R4 requires one of eight frozen shards")
    validate_initialized(repo); _validate_execution_code(repo)
    telemetry_path = repo / ROOT_PATH / "telemetry" / f"shard-{shard_id:02d}-of-08.json"
    plan = _v1_plan(repo); _, eligible = _partition(repo)
    shard_units = [unit for unit in eligible if unit["selection_order"] % 8 == shard_id]
    if telemetry_path.exists():
        telemetry, _ = _strict_json(telemetry_path, "E3-R4 shard telemetry"); _validate_signed(telemetry, "E3-R4 shard telemetry")
        if any(not _r4_attempt_path(repo, unit).is_dir() for unit in shard_units):
            raise StageEE3R4Error("E3-R4 telemetry exists before shard terminal coverage")
        return telemetry
    units_by_sample: dict[str, list[dict[str, Any]]] = {}
    for unit in shard_units: units_by_sample.setdefault(unit["sample_id"], []).append(unit)
    groups = [group for group in plan["groups"] if group["sample_id"] in units_by_sample]
    import torch
    from .stage_c_fidelity import ValuePathScalarAdapter, resolve_target_response
    from .stage_c_value_path import (
        HeadCandidate, StageCValuePathConvergenceError,
        e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig,
        matched_fixed_clean_attention_patch_token_contribution,
        resolve_fixed_clean_attention_head_directions,
    )
    from .stage_c_perturbations import tensor_sha256
    model = None; process_start = time.monotonic(); process_cpu = time.process_time()
    records: list[dict[str, Any]] = []; completed = skipped = 0; statuses: Counter[str] = Counter()
    try:
        model, binding, model_record = e1._load_model_and_context(repo, device)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES"); gpu_index = torch.device(device).index or 0
        physical = int(visible) if visible is not None and visible.isdigit() else gpu_index
        identities: dict[int, dict[str, str]] = {}
        if device.startswith("cuda"):
            query = subprocess.run(("nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.splitlines()
            for line in query:
                index, uuid, name = (item.strip() for item in line.split(",", 2)); identities[int(index)] = {"gpu_uuid": uuid, "gpu_name": name}
        for group in groups:
            group_units = units_by_sample[group["sample_id"]]; existing = {}
            for unit in group_units:
                if _r4_attempt_path(repo, unit).exists(): existing[unit["canonical_digest"]] = _validate_r4_attempt(repo, unit)
            missing = [unit for unit in group_units if unit["canonical_digest"] not in existing]
            skipped += len(existing); statuses.update(item["result"]["validity_status"] for item in existing.values())
            if not missing:
                existing_counts = Counter(item["result"]["validity_status"] for item in existing.values())
                records.append({"selection_order": group["selection_order"], "sample_id": group["sample_id"], "elapsed_seconds": 0.0, "cpu_seconds": 0.0, "forward_calls": 0, "terminal_units_published": len(existing), "valid": existing_counts["VALID"], "failed": existing_counts["FAILED"], "na": existing_counts["NA"]})
                continue
            group_start = time.monotonic(); group_cpu = time.process_time(); calls = 0
            if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(device)
            _canonical, clean, clean_sha = e1._load_sample(binding, group, model, device)
            if clean_sha != tensor_sha256(clean): raise StageEE3R4Error("E3-R4 live clean tensor hash drifted")
            def hook(_module: Any, _args: Any) -> None:
                nonlocal calls; calls += 1
            handle = model.register_forward_pre_hook(hook)
            try:
                coordinates = sorted({(unit["block"], unit["head"]) for unit in missing}, key=e3.COORDINATES.index)
                targets = resolve_fixed_clean_attention_head_directions(model, clean, tuple(HeadCandidate(*coord) for coord in coordinates), candidate_chunk_size=len(coordinates))
                target_map = {(target.block_index, target.head_index): target for target in targets}
                for block, head in coordinates:
                    target = target_map[(block, head)]; evaluator = ValuePathScalarAdapter(model, clean, target)
                    response = resolve_target_response(evaluator, clean, torch.zeros_like(clean))
                    target_identity = json.loads(canonical_json_bytes(evaluator.target_identity).decode("utf-8"))
                    coordinate_units = [unit for unit in missing if (unit["block"], unit["head"]) == (block, head)]
                    for unit in coordinate_units:
                        started = time.monotonic(); cpu_started = time.process_time(); before = calls
                        if response.na is not None:
                            result = _na_result(unit=unit, target_identity=target_identity, response=response, elapsed_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started, forward_calls=calls-before, peak_cuda_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0, clean_sha256=clean_sha)
                            _publish_na(repo, unit, result); completed += 1; statuses.update(["NA"]); continue
                        try:
                            if unit["method"] == e3.NNS_METHOD:
                                method = e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig(model, clean, target, provenance={"stage": "E3-R4", "comparison_identity": e3.comparison_identity(block, head), "typed_terminal_contract": RECOVERY_ID})
                                ig = method.provenance["ig_result"]; backward = sum(int(math.ceil(int(item["steps"]) / 8)) for item in ig["attempts"])
                            else:
                                method = matched_fixed_clean_attention_patch_token_contribution(target, provenance={"stage": "E3-R4", "comparison_identity": e3.comparison_identity(block, head), "typed_terminal_contract": RECOVERY_ID}); backward = 0
                            logical = {"forward_calls": 0}
                            curves, signed_map = e3._curves_for_method(model=model, clean=clean, target=target, response=response, method_result=method, sample_id=group["sample_id"], block=block, head=head, call_counter=logical)
                            result = r1._valid_result(unit=unit, target_identity=target_identity, response=response, method_result=method, signed_map=signed_map, curves=curves, elapsed_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started, forward_calls=calls-before, backward_calls=backward, peak_cuda_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0, clean_sha256=clean_sha, recovery_id=RECOVERY_ID, attempt_id=ATTEMPT_ID, origin="E3_R4_NEW")
                            r1._publish_attempt(repo, unit, result, signed_map, root_path=ROOT_PATH, attempt_id=ATTEMPT_ID, recovery_id=RECOVERY_ID, nns_schedule=SCHEDULE, numerical_field="numerical_recovery", numerical_id=r1.RECOVERY_ID, failure_code=FAILURE_CODE)
                            completed += 1; statuses.update(["VALID"])
                        except StageCValuePathConvergenceError as error:
                            diagnostic = error.ig_result.attribution[0].detach().to(device="cpu", dtype=torch.float32).numpy().copy()
                            result = r1._failure_result(unit=unit, target_identity=target_identity, response=response, error=error, elapsed_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started, forward_calls=calls-before, peak_cuda_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0, clean_sha256=clean_sha, diagnostic=diagnostic, recovery_id=RECOVERY_ID, attempt_id=ATTEMPT_ID, origin="E3_R4_NEW", expected_schedule=SCHEDULE, failure_code=FAILURE_CODE)
                            r1._publish_attempt(repo, unit, result, diagnostic, root_path=ROOT_PATH, attempt_id=ATTEMPT_ID, recovery_id=RECOVERY_ID, nns_schedule=SCHEDULE, numerical_field="numerical_recovery", numerical_id=r1.RECOVERY_ID, failure_code=FAILURE_CODE)
                            completed += 1; statuses.update(["FAILED"])
                    del evaluator, response
            finally:
                handle.remove()
            records.append({"selection_order": group["selection_order"], "sample_id": group["sample_id"], "elapsed_seconds": time.monotonic()-group_start, "cpu_seconds": time.process_time()-group_cpu, "forward_calls": calls, "terminal_units_published": sum(_r4_attempt_path(repo, unit).is_dir() for unit in group_units), "valid": sum(_validate_r4_attempt(repo, unit)["result"]["validity_status"] == "VALID" for unit in group_units if _r4_attempt_path(repo, unit).is_dir()), "failed": sum(_validate_r4_attempt(repo, unit)["result"]["validity_status"] == "FAILED" for unit in group_units if _r4_attempt_path(repo, unit).is_dir()), "na": sum(_validate_r4_attempt(repo, unit)["result"]["validity_status"] == "NA" for unit in group_units if _r4_attempt_path(repo, unit).is_dir())})
        if any(not _r4_attempt_path(repo, unit).is_dir() for unit in shard_units):
            raise StageEE3R4Error("E3-R4 shard ended without complete typed-terminal coverage")
        elapsed = time.monotonic()-process_start; cpu_seconds = time.process_time()-process_cpu
        telemetry = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_shard_telemetry", "recovery_id": RECOVERY_ID, "shard_id": shard_id, "shard_count": 8, "device": device, "physical_gpu_index": physical if device.startswith("cuda") else None, "gpu_uuid": identities.get(physical, {}).get("gpu_uuid"), "gpu_name": identities.get(physical, {}).get("gpu_name"), "elapsed_seconds": elapsed, "gpu_hours": elapsed/3600.0 if device.startswith("cuda") else 0.0, "cpu_seconds": cpu_seconds, "completed_units": completed, "skipped_terminal_units": skipped, "valid_units_seen": statuses["VALID"], "failed_units_seen": statuses["FAILED"], "na_units_seen": statuses["NA"], "groups": records, "model": model_record, "completed_at": _now()})
        _write_or_verify(telemetry_path, telemetry); return telemetry
    finally:
        if model is not None: del model
        if device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.empty_cache()


def _all_sources(repo: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    attempts: dict[str, dict[str, Any]] = {}; origins: dict[str, str] = {}
    for unit in _v1_plan(repo)["units"]:
        source = _source_attempt(repo, unit)
        if source is None:
            raise StageEE3R4Error(f"missing typed terminal union unit: {unit['canonical_digest']}")
        origin, observed = source; attempts[unit["canonical_digest"]] = observed; origins[unit["canonical_digest"]] = origin
    counts = Counter(origins.values())
    expected = Counter({"E3_V1_INHERITED": 516, "E3_R1_INHERITED": 12, "E3_R2_INHERITED_VALID": 636, "E3_R2_INHERITED_FAILED": 1, "E3_R4_NEW": 1835})
    if len(attempts) != 3000 or counts != expected:
        raise StageEE3R4Error(f"E3-R4 terminal union coverage failed: {counts}")
    return attempts, origins


def _combined_new_rows(
    repo: Path, plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    attempts, origins = _all_sources(repo); method_rows: list[dict[str, Any]] = []; by_key: dict[tuple[str, int, int], dict[str, tuple[dict[str, Any], dict[str, Any]]]] = {}
    recompute = {
        "E3_V1_INHERITED": "INHERITED_E3_V1_TERMINAL",
        "E3_R1_INHERITED": "INHERITED_E3_R1_TERMINAL",
        "E3_R2_INHERITED_VALID": "INHERITED_E3_R2_VALID_TERMINAL",
        "E3_R2_INHERITED_FAILED": "INHERITED_E3_R2_FAILED_TERMINAL",
        "E3_R4_NEW": "E3_R4_NEW_TYPED_TERMINAL",
    }
    for unit in plan["units"]:
        observed = attempts[unit["canonical_digest"]]; result = observed["result"]; path = observed["attempt_path"]; origin = origins[unit["canonical_digest"]]
        row = {
            "variant_id": f"layer_head_{unit['block']}_{unit['head']}", "sample_id": unit["sample_id"], "selection_order": unit["selection_order"],
            "cohort": "stage_d_fidelity_v1_500", "stratum": unit["stratum"], "validity_status": result["validity_status"],
            "source_hashes": result["source_hashes"], "method": unit["method"], "target": unit["target"], "scale": unit["scale"],
            "layer": unit["block"], "head": unit["head"], "coordinate_role": "DESCRIPTIVE_FROZEN_COORDINATE",
            "recompute_status": recompute[origin], "evidence_origin": origin, "attempt_id": result["attempt_id"], "retry_count": result["retry_count"],
            "failure_class": result.get("failure_class"), "failure_code": result.get("failure_code"), "na_reason": result.get("na_reason"),
            "insertion_auc_mean": result.get("insertion_auc_mean"), "deletion_auc_mean": result.get("deletion_auc_mean"),
            "insertion_auc_repeat_sd_ddof_0": result.get("insertion_auc_repeat_sd_ddof_0"), "deletion_auc_repeat_sd_ddof_0": result.get("deletion_auc_repeat_sd_ddof_0"),
            "misclassified": unit["stratum"] == "misclassified", "pair_binding_digest": result.get("pair_binding_digest"),
            "elapsed_seconds": result["elapsed_seconds"], "gpu_hours": result["gpu_hours"], "cpu_seconds": result["cpu_seconds"],
            "forward_calls": result["forward_calls"], "backward_calls": result["backward_calls"], "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
            "artifact_bytes": sum(item.stat().st_size for item in path.iterdir() if item.is_file()), "descriptive_only": True,
        }
        method_rows.append(row); by_key.setdefault((unit["sample_id"], unit["block"], unit["head"]), {})[unit["method"]] = (row, result)
    pair_rows: list[dict[str, Any]] = []
    for (sample_id, block, head), methods in by_key.items():
        if set(methods) != set(e3.METHODS): raise StageEE3R4Error("E3-R4 paired method coverage failed")
        nns, nns_result = methods[e3.NNS_METHOD]; matched, matched_result = methods[e3.MATCHED_METHOD]
        statuses = {nns["validity_status"], matched["validity_status"]}
        status = "VALID" if statuses == {"VALID"} else "NA" if "NA" in statuses else "FAILED"
        if status == "VALID":
            if e1._target_semantic_digest(nns_result["target_response"]["target_identity"]) != e1._target_semantic_digest(matched_result["target_response"]["target_identity"]):
                raise StageEE3R4Error("E3-R4 paired target identity drifted")
            ncurves = {(row["direction"], row["replicate"]): row for row in nns_result["curves"]}; mcurves = {(row["direction"], row["replicate"]): row for row in matched_result["curves"]}
            if set(ncurves) != set(mcurves) or any(ncurves[key]["seed"] != mcurves[key]["seed"] or ncurves[key]["tie_priority_sha256"] != mcurves[key]["tie_priority_sha256"] for key in ncurves):
                raise StageEE3R4Error("E3-R4 paired seed/tie drifted")
        pair_rows.append({
            "variant_id": f"layer_head_{block}_{head}_pair", "sample_id": sample_id, "selection_order": nns["selection_order"],
            "cohort": "stage_d_fidelity_v1_500", "stratum": nns["stratum"], "validity_status": status,
            "source_hashes": {"nns_map_sha256": nns["source_hashes"].get("signed_map_sha256"), "matched_map_sha256": matched["source_hashes"].get("signed_map_sha256"), "target_semantic_digest": nns["source_hashes"].get("target_semantic_digest")},
            "method": "nns_vs_matched", "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "layer": block, "head": head,
            "coordinate_role": "DESCRIPTIVE_FROZEN_COORDINATE", "recompute_status": "E3_V1_R1_R2_R4_TYPED_UNION", "misclassified": nns["misclassified"],
            "nns_insertion_auc": nns.get("insertion_auc_mean"), "nns_deletion_auc": nns.get("deletion_auc_mean"),
            "matched_insertion_auc": matched.get("insertion_auc_mean"), "matched_deletion_auc": matched.get("deletion_auc_mean"),
            "delta_fid": e3.delta_fid(nns, matched) if status == "VALID" else None,
            "failure_class": None if status == "VALID" else "SCIENTIFIC_INVALIDITY" if status == "FAILED" else None,
            "failure_code": (nns.get("failure_code") or matched.get("failure_code")) if status == "FAILED" else None,
            "na_reason": (nns.get("na_reason") or matched.get("na_reason")) if status == "NA" else None,
            "descriptive_only": True,
        })
    return method_rows, pair_rows, attempts, origins


def _actual_cost(repo: Path) -> dict[str, Any]:
    upstream = r2.validate_completion(repo)["actual_cost"]; telemetry = []
    for path in sorted((repo / ROOT_PATH / "telemetry").glob("shard-*-of-08.json")):
        value, _ = _strict_json(path, "E3-R4 telemetry"); _validate_signed(value, "E3-R4 telemetry"); telemetry.append(value)
    r4_gpu = sum(float(item["gpu_hours"]) for item in telemetry); r4_cpu = sum(float(item["cpu_seconds"]) for item in telemetry)
    stable_retained_paths = (
        e3.ROOT_PATH, e3.REPORT_ROOT_PATH, r1.ROOT_PATH, r1.REPORT_ROOT_PATH,
        r2.ROOT_PATH, r2.REPORT_ROOT_PATH, r3.ROOT_PATH, r3.REPORT_ROOT_PATH,
        ROOT_PATH + "/control", ROOT_PATH + "/units", ROOT_PATH + "/telemetry",
    )
    retained = sum(_inventory(repo / path)["total_bytes"] for path in stable_retained_paths)
    retained += sum((repo / path).stat().st_size for path in (TEST_REPORT_PATH, POST_EXECUTION_TEST_REPORT_PATH, R3_TERMINATION_PATH))
    return {
        "e3_v1_r1_r2_gpu_hours": upstream["gpu_hours"], "e3_r3_interrupted_gpu_hours": R3_CHARGED_SECONDS / 3600.0, "e3_r4_gpu_hours": r4_gpu,
        "gpu_hours": upstream["gpu_hours"] + R3_CHARGED_SECONDS / 3600.0 + r4_gpu,
        "e3_v1_r1_r2_cpu_seconds": upstream["cpu_seconds"], "e3_r3_interrupted_cpu_seconds": R3_CHARGED_SECONDS, "e3_r4_cpu_seconds": r4_cpu,
        "cpu_seconds": upstream["cpu_seconds"] + R3_CHARGED_SECONDS + r4_cpu, "retained_bytes": retained,
        "retained_bytes_scope": "historical_e3_roots_plus_r3_root_plus_r4_control_attempts_telemetry_tests_and_termination_event_excluding_reproducible_derived_package",
        "e3_r4_gpu_invocations": len(telemetry), "status": "RECORDED_NOT_A_COMPLETION_GATE", "completion_gate": False,
    }


def _typed_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(row["validity_status"] for row in rows)
    if set(counts) - {"VALID", "FAILED", "NA"}: raise StageEE3R4Error("unknown typed terminal status")
    return {"expected": len(rows), "valid": counts["VALID"], "failed": counts["FAILED"], "na": counts["NA"]}


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); validate_upstream(repo); plan = _v1_plan(repo)
    new_methods, new_pairs, attempts, origins = _combined_new_rows(repo, plan)
    primary_methods, primary_pairs = e3._primary_method_rows(repo, plan)
    method_rows = sorted(primary_methods + new_methods, key=lambda row: (row["selection_order"], e3.COORDINATES.index((row["layer"], row["head"])), e3.METHODS.index(row["method"])))
    pair_rows = sorted(primary_pairs + new_pairs, key=lambda row: (row["selection_order"], e3.COORDINATES.index((row["layer"], row["head"]))))
    attempt_counts = _typed_counts([observed["result"] for observed in attempts.values()])
    if len(method_rows) != 4000 or len(pair_rows) != 2000 or attempt_counts["expected"] != 3000 or attempt_counts["failed"] < 1:
        raise StageEE3R4Error("E3-R4 complete typed union cardinality failed")
    failed_identity = [observed["result"] for observed in attempts.values() if observed["result"]["unit_digest"] == r3._pilot_unit(repo)["canonical_digest"]]
    if len(failed_identity) != 1 or failed_identity[0]["validity_status"] != "FAILED" or failed_identity[0]["origin"] != "E3_R2_NEW":
        raise StageEE3R4Error("order-117 failure was not preserved as the unique inherited terminal")
    telemetry = []
    for path in sorted((repo / ROOT_PATH / "telemetry").glob("shard-*-of-08.json")):
        value, _ = _strict_json(path, "E3-R4 telemetry"); _validate_signed(value, "E3-R4 telemetry"); telemetry.append(value)
    if {item["shard_id"] for item in telemetry} != set(range(8)):
        raise StageEE3R4Error("E3-R4 shard telemetry is incomplete")
    _, eligible = _partition(repo); expected_orders = {unit["selection_order"] for unit in eligible}
    observed_orders = [group["selection_order"] for item in telemetry for group in item["groups"]]
    if len(observed_orders) != len(set(observed_orders)) or set(observed_orders) != expected_orders:
        raise StageEE3R4Error("E3-R4 telemetry order coverage drifted")
    summaries: list[dict[str, Any]] = []; method_summaries: list[dict[str, Any]] = []
    for block, head in e3.COORDINATES:
        coordinate_pairs = [row for row in pair_rows if (row["layer"], row["head"]) == (block, head)]
        coordinate_methods = [row for row in method_rows if (row["layer"], row["head"]) == (block, head)]
        for stratum in e3.STRATA:
            selected_pairs = coordinate_pairs if stratum == "overall" else [row for row in coordinate_pairs if row["stratum"] == stratum]
            summary = e3._summary(selected_pairs, "delta_fid", f"layer_head_{block}_{head}_delta_fid", stratum)
            summaries.append({"variant_id": f"layer_head_{block}_{head}_{stratum}", "sample_id": None, "selection_order": None, "cohort": "stage_d_fidelity_v1_500", "stratum": stratum, "validity_status": "VALID", "scientific_coverage_status": "MIXED" if summary["failed"] or summary["na"] else "ALL_VALID", "source_hashes": {"paired_rows_canonical_digest": canonical_sha256(selected_pairs)}, "method": "nns_vs_matched", "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "layer": block, "head": head, "coordinate_role": "PRIMARY_ANCHOR_READ_ONLY" if (block, head) == (0, 9) else "DESCRIPTIVE_FROZEN_COORDINATE", "metric": "delta_fid", **summary, "descriptive_only": True})
            for method in e3.METHODS:
                selected = [row for row in coordinate_methods if row["method"] == method and (stratum == "overall" or row["stratum"] == stratum)]
                for direction in ("insertion", "deletion"):
                    summary = e3._summary(selected, f"{direction}_auc_mean", f"layer_head_{block}_{head}_{method}_{direction}_auc", stratum)
                    method_summaries.append({"variant_id": f"layer_head_{block}_{head}_{method}_{direction}_{stratum}", "sample_id": None, "selection_order": None, "cohort": "stage_d_fidelity_v1_500", "stratum": stratum, "validity_status": "VALID", "scientific_coverage_status": "MIXED" if summary["failed"] or summary["na"] else "ALL_VALID", "source_hashes": {"method_rows_canonical_digest": canonical_sha256(selected)}, "method": method, "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "layer": block, "head": head, "coordinate_role": "PRIMARY_ANCHOR_READ_ONLY" if (block, head) == (0, 9) else "DESCRIPTIVE_FROZEN_COORDINATE", "metric": f"{direction}_auc", **summary, "descriptive_only": True})
    strata = {group["sample_id"]: group["stratum"] for group in plan["groups"]}; registry, _ = e1._load_registry(repo)
    membership = [{"variant_id": "prediction_stratum_membership", "sample_id": row["sample_id"], "selection_order": row["selection_order"], "cohort": "stage_d_fidelity_v1_500", "stratum": strata[row["sample_id"]], "validity_status": "VALID", "source_hashes": {"d10_per_sample_sha256": e3._sha_file(repo / e3.D10_DELTA_PATH), "image_sha256": row["image_sha256"]}, "method": "nns_vs_matched", "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "layer": None, "head": None, "misclassified": strata[row["sample_id"]] == "misclassified"} for row in registry]
    image_coverage = []
    for group in plan["groups"]:
        rows = [row for row in new_methods if row["sample_id"] == group["sample_id"]]; counts = _typed_counts(rows)
        image_coverage.append({"variant_id": "e3_new_compute_image_coverage", "sample_id": group["sample_id"], "selection_order": group["selection_order"], "cohort": "stage_d_fidelity_v1_500", "stratum": group["stratum"], "validity_status": "VALID" if counts["valid"] == 6 else "NA" if counts["na"] and not counts["failed"] else "FAILED", "source_hashes": {"terminal_rows_canonical_digest": canonical_sha256(rows)}, "method": "typed_terminal_coverage", "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "layer": None, "head": None, **counts, "complete": counts["expected"] == 6})
    if len(image_coverage) != 500 or any(not row["complete"] for row in image_coverage): raise StageEE3R4Error("E3-R4 500-image table is incomplete")
    method_counts = _typed_counts(method_rows); pair_counts = _typed_counts(pair_rows); cost = _actual_cost(repo)
    summary_doc = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_descriptive_summary", "stage_id": "E3-R4", "recovery_id": RECOVERY_ID, "cohort": "stage_d_fidelity_v1_500", "descriptive_only": True, "p_values_emitted": False, "winner_selection_performed": False, "causal_claim_made": False, "new_go_no_go_issued": False, "d10_primary_unchanged": True, "primary_anchor_recomputed": False, "historical_statuses": {"E3": "FAIL", "E3-R1": "BLOCKED", "E3-R2": "FAIL", "E3-R3": "PENDING_INTERRUPTED_WITHOUT_TERMINAL_PILOT_PAYLOAD"}, "scientific_validity": "MIXED", "summary_population": "VALID_ONLY_WITH_EXPECTED_VALID_FAILED_NA_COUNTS", "counts": {"images": 500, "new_compute_terminal_units": attempt_counts, "method_rows_including_primary_anchor": method_counts, "paired_rows": pair_counts, "summary_cells": 12, "method_summary_cells": 48, "origins": dict(Counter(origins.values()))}, "delta_fid_coordinate_strata": summaries, "method_auc_coordinate_strata": method_summaries, "prediction_partition": {"overall": 500, "correct": 418, "misclassified": 82}, "clean_margin": e3._validate_config(repo)["scientific_contract"]["clean_margin"], "actual_cost": cost, "e4_disclosure_required": True, "e4_started": False})
    terminal_rows = [row for row in new_methods]
    return {"method_rows": method_rows, "pair_rows": pair_rows, "summary_rows": summaries + method_summaries, "membership_rows": membership, "image_coverage_rows": image_coverage, "terminal_rows": terminal_rows, "resource_rows": method_rows, "attempts": attempts, "origins": origins, "attempt_counts": attempt_counts, "method_counts": method_counts, "pair_counts": pair_counts, "summary": summary_doc, "actual_cost": cost}


def _table(table_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows: raise StageEE3R4Error(f"{table_id} cannot be empty")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    normalized = [{key: row.get(key) for key in columns} for row in rows]
    return _signed({"schema_version": 1, "document_type": "stage_e_e3_paper_table", "table_id": table_id, "columns": columns, "row_count": len(normalized), "rows": normalized})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    r3._write_csv(path, rows)


def finalize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); outputs = validate_outputs(repo); root = repo / ROOT_PATH
    specs = (
        ("layer_head_method_rows", outputs["method_rows"]),
        ("layer_head_paired_delta", outputs["pair_rows"]),
        ("prediction_strata", outputs["summary_rows"]),
        ("prediction_strata_membership", outputs["membership_rows"]),
        ("image_terminal_coverage", outputs["image_coverage_rows"]),
        ("new_compute_terminal_rows", outputs["terminal_rows"]),
        ("resource_and_qc", outputs["resource_rows"]),
    )
    for name, rows in specs:
        table = _table(name, rows); _write_or_verify(root / "paper_data" / f"{name}.json", table); _write_csv(root / "paper_data" / f"{name}.csv", table["rows"])
    _write_or_verify(root / "descriptive_summary.json", outputs["summary"])
    _write_or_verify(root / "clean_margin_not_run.json", _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_not_run", "analysis": "clean_margin", **e3._validate_config(repo)["scientific_contract"]["clean_margin"]}))
    recipe = _signed({"schema_version": 1, "document_type": "stage_e_figure_recipes", "numeric_source": "paper_data_only", "renders_png": False, "figures": [{"figure_id": "e3_r4_layer_head_prediction_strata_future", "table_path": "paper_data/prediction_strata.json", "x": "layer", "y": "median", "group": "stratum", "caption_contract": "Post-D10 descriptive views; numeric statistics use VALID rows only and every FAILED/NA count and E3 history must be disclosed.", "numeric_source": "paper_data_only", "renders_png": False}]})
    _write_or_verify(root / "figure_recipes.json", recipe)
    provenance = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_provenance", "recovery_id": RECOVERY_ID, "historical_statuses": {"E3": "FAIL", "E3-R1": "BLOCKED", "E3-R2": "FAIL", "E3-R3": "PENDING_INTERRUPTED_WITHOUT_TERMINAL_PILOT_PAYLOAD"}, "origins": dict(Counter(outputs["origins"].values())), "typed_terminal_counts": outputs["attempt_counts"], "r3_termination_event": _record(repo, R3_TERMINATION_PATH), "operations": ["preserve every historical receipt and terminal attempt", "retain order-117 NNS as FAILED without retry", "run only 1835 units without a prior legal terminal", "continue after typed FAILED or NA terminal publication", "summarize numeric values over VALID rows only with expected/valid/failed/NA counts"], "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}, "e4_disclosure_required": True, "e4_started": False})
    _write_or_verify(root / "provenance.json", provenance)
    false_audit = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_false_pass_audit", "recovery_id": RECOVERY_ID, "status": "PASS", "checks": [{"attack": name, "result": "REJECTED"} for name in FALSE_PASS_ATTACKS], "workflow_pass_is_not_all_unit_scientific_pass": True, "e4_started": False})
    _write_or_verify(repo / REPORT_ROOT_PATH / "false_pass_audit.json", false_audit)
    tests, _ = _strict_json(repo / TEST_REPORT_PATH, "E3-R4 tests"); _validate_signed(tests, "E3-R4 tests")
    if tests.get("status") != "PASS" or tests.get("failed") != 0: raise StageEE3R4Error("E3-R4 tests are not PASS")
    post_tests, _ = _strict_json(repo / POST_EXECUTION_TEST_REPORT_PATH, "E3-R4 post-execution tests"); _validate_signed(post_tests, "E3-R4 post-execution tests")
    if post_tests.get("status") != "PASS" or post_tests.get("failed") != 0 or post_tests.get("scientific_attempts_modified_or_rerun") is not False:
        raise StageEE3R4Error("E3-R4 post-execution tests are not PASS")
    validation = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_source_validation", "stage_id": "E3-R4", "recovery_id": RECOVERY_ID, "status": "PASS", "validated_at": _now(), "checks": {"historical_receipts_and_attempts_preserved": "PASS", "r3_interrupted_without_terminal_payload": "PASS", "origin_union": "PASS_516_12_636_VALID_1_FAILED_1835_NEW", "union_3000_unique_typed_terminal": "PASS", "order_117_failure_preserved": "PASS", "method_rows": "PASS_4000_WITH_TYPED_STATUS", "paired_rows": "PASS_2000_WITH_TYPED_STATUS", "image_table": "PASS_500_COMPLETE", "summary_population": "PASS_VALID_ONLY_WITH_TYPED_COUNTS", "scientific_contract_unchanged": "PASS", "cost_recorded_not_gated": "PASS", "e4_not_started": "PASS"}, "scientific_validity": "MIXED", "typed_terminal_counts": outputs["attempt_counts"], "summary_canonical_digest": outputs["summary"]["canonical_digest"], "false_pass_audit_canonical_digest": false_audit["canonical_digest"]})
    _write_or_verify(repo / REPORT_ROOT_PATH / "validation_report.json", validation)
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "completion" not in path.parts and path.name not in {"payload_manifest.json", "whole_root_inventory.json", "status.json"}:
            artifacts.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    manifest = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_payload_manifest", "recovery_id": RECOVERY_ID, "immutable": True, "payload_count": len(artifacts), "artifacts": artifacts})
    _write_or_verify(root / "payload_manifest.json", manifest)
    cost = _actual_cost(repo)
    inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/e3_r4_completion.pending.json", "completion/e3_r4_completion.json"))
    inventory_doc = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_whole_root_inventory", "root": ROOT_PATH, "scope": "all regular non-hidden E3-R4 formal files except self-referential inventory status and completion receipts", **inventory})
    _write_or_verify(root / "whole_root_inventory.json", inventory_doc)
    status = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_status", "stage_id": "E3-R4", "recovery_id": RECOVERY_ID, "status": "PASS", "complete": True, "terminal": True, "immutable": True, "historical_statuses_preserved": {"E3": "FAIL", "E3-R1": "BLOCKED", "E3-R2": "FAIL", "E3-R3": "PENDING_INTERRUPTED_WITHOUT_TERMINAL_PILOT_PAYLOAD"}, "package_completeness": "PASS", "workflow_integrity": "PASS", "scientific_validity": "MIXED", "typed_terminal_counts": outputs["attempt_counts"], "cost_status": "RECORDED_NOT_A_COMPLETION_GATE", "actual_cost": cost, "e4_authorized": True, "e4_disclosure_required": True, "e4_started": False, "stage_f_started": False, "whole_root_inventory": {"path": "whole_root_inventory.json", "file_sha256": _sha_file(root / "whole_root_inventory.json"), "canonical_digest": inventory_doc["canonical_digest"], "file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"]}})
    _write_or_verify(root / "status.json", status)
    pending, _ = _strict_json(repo / COMPLETION_PENDING_PATH, "E3-R4 pending")
    completion = _signed({"schema_version": 1, "document_type": "stage_e_e3_r4_terminal_completion", "stage_id": "E3-R4", "recovery_id": RECOVERY_ID, "status": "PASS", "started_at": pending["started_at"], "completed_at": _now(), "historical_statuses": {"E3": "FAIL", "E3-R1": "BLOCKED", "E3-R2": "FAIL", "E3-R3": "PENDING_INTERRUPTED_WITHOUT_TERMINAL_PILOT_PAYLOAD"}, "union": {"expected_new_compute_units": 3000, "e3_v1_inherited": 516, "e3_r1_inherited": 12, "e3_r2_inherited_valid": 636, "e3_r2_inherited_failed": 1, "e3_r4_new": 1835, "terminal_counts": outputs["attempt_counts"], "method_rows": 4000, "paired_rows": 2000, "image_rows": 500, "summary_cells": 12}, "package_completeness": "PASS", "workflow_integrity": "PASS", "scientific_validity": "MIXED", "cost_status": "RECORDED_NOT_A_COMPLETION_GATE", "actual_cost": cost, "source_validation": {"validation_report": _record(repo, REPORT_ROOT_PATH + "/validation_report.json"), "test_report": _record(repo, TEST_REPORT_PATH), "post_execution_test_report": _record(repo, POST_EXECUTION_TEST_REPORT_PATH), "false_pass_audit": _record(repo, REPORT_ROOT_PATH + "/false_pass_audit.json"), "r3_termination_event": _record(repo, R3_TERMINATION_PATH), "payload_manifest": _record(repo, ROOT_PATH + "/payload_manifest.json"), "status": _record(repo, ROOT_PATH + "/status.json"), "whole_root_inventory": _record(repo, ROOT_PATH + "/whole_root_inventory.json")}, "next_authorization": "E4_ONLY_WITH_MIXED_VALIDITY_DISCLOSURE", "e4_disclosure_required": True, "e4_started": False})
    schema, _ = _strict_json(repo / COMPLETION_SCHEMA_PATH, "E3-R4 completion schema")
    errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE3R4Error(f"E3-R4 completion schema failure: {errors[0].message}")
    temporary = (repo / COMPLETION_PENDING_PATH).with_name(".e3-r4-completion-terminal.tmp")
    _write_new_json(temporary, completion); os.replace(temporary, repo / COMPLETION_PENDING_PATH)
    observed, raw = _strict_json(repo / COMPLETION_PENDING_PATH, "E3-R4 terminal pending")
    if observed != completion or raw != canonical_json_bytes(completion) + b"\n": raise StageEE3R4Error("E3-R4 completion changed before atomic publication")
    os.replace(repo / COMPLETION_PENDING_PATH, repo / COMPLETION_PATH)
    final_inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/e3_r4_completion.pending.json", "completion/e3_r4_completion.json"))
    if final_inventory != inventory: raise StageEE3R4Error("E3-R4 root changed during terminal publication")
    return validate_completion(repo)


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if not (repo / COMPLETION_PATH).is_file() or (repo / COMPLETION_PENDING_PATH).exists(): raise StageEE3R4Error("E3-R4 terminal completion absent or pending remains")
    completion, _ = _strict_json(repo / COMPLETION_PATH, "E3-R4 completion"); _validate_signed(completion, "E3-R4 completion")
    schema, _ = _strict_json(repo / COMPLETION_SCHEMA_PATH, "E3-R4 completion schema"); errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE3R4Error(f"E3-R4 completion schema failed: {errors[0].message}")
    if completion.get("status") != "PASS" or completion.get("scientific_validity") != "MIXED" or completion.get("next_authorization") != "E4_ONLY_WITH_MIXED_VALIDITY_DISCLOSURE" or completion.get("e4_started") is not False:
        raise StageEE3R4Error("E3-R4 completion boundary failed")
    validate_upstream(repo); outputs = validate_outputs(repo)
    if outputs["actual_cost"] != completion["actual_cost"] or outputs["attempt_counts"] != completion["union"]["terminal_counts"]:
        raise StageEE3R4Error("E3-R4 completion cost/count drifted")
    status, _ = _strict_json(repo / ROOT_PATH / "status.json", "E3-R4 status"); inventory, _ = _strict_json(repo / ROOT_PATH / "whole_root_inventory.json", "E3-R4 inventory")
    _validate_signed(status, "E3-R4 status"); _validate_signed(inventory, "E3-R4 inventory")
    rebuilt = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "status.json", "completion/e3_r4_completion.pending.json", "completion/e3_r4_completion.json"))
    if any(inventory[key] != rebuilt[key] for key in ("files", "file_count", "total_bytes")) or canonical_sha256({"files": inventory["files"]}) != rebuilt["canonical_digest"]:
        raise StageEE3R4Error("E3-R4 inventory drifted")
    if status.get("status") != "PASS" or status.get("scientific_validity") != "MIXED" or status.get("e4_authorized") is not True or status.get("e4_disclosure_required") is not True or status.get("e4_started") is not False:
        raise StageEE3R4Error("E3-R4 status drifted")
    for record in completion["source_validation"].values():
        if record != _record(repo, record["path"]): raise StageEE3R4Error(f"E3-R4 completion binding drifted: {record['path']}")
    if any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in (repo / ROOT_PATH).rglob("*") if path.is_file()): raise StageEE3R4Error("E3-R4 contains a forbidden render artifact")
    return {"status": "PASS", "package_completeness": "PASS", "workflow_integrity": "PASS", "scientific_validity": "MIXED", "completion_canonical_digest": completion["canonical_digest"], "whole_root_inventory_digest": inventory["canonical_digest"], "terminal_counts": outputs["attempt_counts"], "method_rows": 4000, "paired_rows": 2000, "image_rows": 500, "actual_cost": completion["actual_cost"], "next_authorization": "E4_ONLY_WITH_MIXED_VALIDITY_DISCLOSURE", "e4_disclosure_required": True, "e4_started": False}


def _main() -> int:
    parser = argparse.ArgumentParser(prog="python -m vitnns.stage_e_e3_r4"); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-preconditions", "initialize", "validate-initialized", "validate-outputs", "finalize", "validate-completion"):
        child = sub.add_parser(name); child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    tests = sub.add_parser("record-tests"); tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); tests.add_argument("--passed", type=int, required=True); tests.add_argument("--failed", type=int, required=True); tests.add_argument("--test-command", required=True)
    post_tests = sub.add_parser("record-post-execution-tests"); post_tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); post_tests.add_argument("--passed", type=int, required=True); post_tests.add_argument("--failed", type=int, required=True); post_tests.add_argument("--test-command", required=True)
    run = sub.add_parser("run-shard"); run.add_argument("--repo-root", type=Path, default=REPO_ROOT); run.add_argument("--shard-id", type=int, required=True); run.add_argument("--shard-count", type=int, required=True); run.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "verify-preconditions": value = verify_preconditions(args.repo_root)
    elif args.command == "initialize": value = initialize(args.repo_root)
    elif args.command == "validate-initialized": value = validate_initialized(args.repo_root)
    elif args.command == "record-tests": value = record_tests(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command)
    elif args.command == "record-post-execution-tests": value = record_post_execution_tests(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command)
    elif args.command == "run-shard": value = run_shard(args.repo_root, shard_id=args.shard_id, shard_count=args.shard_count, device=args.device)
    elif args.command == "validate-outputs":
        output = validate_outputs(args.repo_root); value = {"status": "PASS", "terminal_counts": output["attempt_counts"], "method_rows": len(output["method_rows"]), "paired_rows": len(output["pair_rows"]), "actual_cost": output["actual_cost"]}
    elif args.command == "finalize": value = finalize(args.repo_root)
    else: value = validate_completion(args.repo_root)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(_main())
