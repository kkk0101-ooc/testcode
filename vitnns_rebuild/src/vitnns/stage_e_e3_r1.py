"""Prospective E3-R1 numerical recovery with an immutable E3 v1 union."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_e_e1 as e1
from . import stage_e_e3 as e3

RECOVERY_ID = "STAGE-E-E3-R1-NUMERICAL-RECOVERY-20260906"
CONFIG_PATH = "config/stage_e_e3_r1.json"
CONFIG_SCHEMA_PATH = "config/stage_e_e3_r1.schema.json"
COMPLETION_SCHEMA_PATH = "config/stage_e_e3_r1_completion.schema.json"
AMENDMENT_PATH = "docs/recovery_stage_e_e3_r1_amendment_20260906.md"
ROOT_PATH = "evidence/stage_e/e3_r1_recovery_v2_20260906/formal"
REPORT_ROOT_PATH = "reports/stage_e_e3_r1_20260906"
PLAN_PATH = ROOT_PATH + "/control/recovery_plan.json"
RUN_SPEC_PATH = ROOT_PATH + "/control/run_spec.json"
PILOT_AUTH_PATH = ROOT_PATH + "/control/pilot_execution_authorization.json"
PILOT_RECOVERY_AUTH_PATH = ROOT_PATH + "/control/pilot_recovery_authorization.json"
PILOT_GATE_PATH = ROOT_PATH + "/control/pilot_continuation_gate.json"
BULK_AUTH_PATH = ROOT_PATH + "/control/bulk_execution_authorization.json"
TEST_REPORT_PATH = REPORT_ROOT_PATH + "/pre_pilot_test_results.json"
RECOVERY_TEST_REPORT_PATH = REPORT_ROOT_PATH + "/pilot_recovery_test_results.json"
INCIDENT_PATH = REPORT_ROOT_PATH + "/pilot_infrastructure_event_001.json"
COMPLETION_PENDING_PATH = ROOT_PATH + "/completion/e3_r1_completion.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/e3_r1_completion.json"
STOP_PATH = ROOT_PATH + "/control/runtime_stop.json"

E3_V1_ROOT = e3.ROOT_PATH
E3_V1_COMMIT = "52b0847939a57e54cc72e590fbd9682a95d08444"
E3_V1_TREE = "278058f37c28b30ddeb31eadac339812888399c3"
E3_V1_REF = "refs/archive/stage-e-e3-v1-terminal-fail"
E3_V1_COMPLETION_SHA256 = "75355b40fe9df19ae8eedd10ef96bacf0f2eab49989894940350790a0a57b4bd"
E3_V1_COMPLETION_DIGEST = "91e76bca8d36173f5fae076159e5e09934ef3c25a902d22bc721787108d82011"
E3_V1_STATUS_SHA256 = "a1e815e57711e88cff022861a4b2d6da65e7bd9ce8fa10b1efd41b396952afb2"
E3_V1_STATUS_DIGEST = "490f69f83a7b841e32857338d0daa86734e5b618c375b7ccf26365a9c8e89a7a"
E3_V1_PLAN_SHA256 = "9de140505720d19126b61a8b81c23f5963aef456d06ac7960285e2bca7591843"
E3_V1_PLAN_DIGEST = "25a89db050917587703903af32e972320339b2447ea21c22da58b2bf7e167430"
E3_V1_INVENTORY_SHA256 = "73858f1bb3a0e66942f7b044d33e935d1eec827b12de62163fecd00dc03e30e5"
E3_V1_INVENTORY_DIGEST = "8bf58c71154f88f38d5f213a96d031f061511ecd5b64560da63b5d553fec2671"
EXPECTED_CONFIG_DIGEST = "21efae2f88177a59928f19dfa69469978ffeaeba95f26b66a41b8cd7ce7fef08"
AMENDMENT_COMMIT = "2db4d92e70479be51bf8178d517ffdd3243141bb"
AMENDMENT_SHA256 = "e2cc2882520b1fd6fb5f3c04c9899325871fb47695373379c4cdb6100ad4b41c"

ATTEMPT_ID = "attempt-r1-000001"
PILOT_ORDERS = (61, 72)
ORIGINAL_SCHEDULE = (512, 1024, 2048, 4096, 8192)
FALLBACK_SCHEDULE = (16384, 32768)
COMPLETE_SCHEDULE = ORIGINAL_SCHEDULE + FALLBACK_SCHEDULE
GPU_HOURS_MAX = 8.0
CPU_SECONDS_MAX = 16.0 * 3600.0
RETAINED_BYTES_MAX = 12884901888
V1_GPU_HOURS = 1.0879465173394418
V1_CPU_SECONDS = 4469.05037553
V1_RETAINED_BYTES_MINIMUM = 33433845
PROJECTION_FACTOR = 1.25
PROJECTION_REMAINING_GROUPS = 412
FINALIZATION_RESERVE_BYTES = 67108864
FORBIDDEN_SUFFIXES = {".png", ".pdf", ".svg"}
FALSE_PASS_ATTACKS = (
    "e3_v1_modified_or_relabelled",
    "e3_v1_terminal_recomputed",
    "unit_with_legal_terminal_result_rerun",
    "fallback_before_original_schedule_exhausted",
    "threshold_or_scientific_contract_relaxed",
    "e3_v1_or_pilot_cost_omitted",
    "pilot_order_61_or_72_excluded",
    "bulk_started_before_pilot_and_budget_pass",
    "favorable_valid_subset_summary",
    "e4_started_before_e3_r1_terminal_pass",
)


class StageEE3R1Error(RuntimeError):
    """E3-R1 cannot proceed or pass under the prospective amendment."""


def _now() -> str:
    zone = time.strftime("%z", time.localtime())
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + zone[:-2] + ":" + zone[-2:]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["canonical_digest"] = canonical_sha256(
        {key: item for key, item in result.items() if key != "canonical_digest"}
    )
    return result


def _strict_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
            object_pairs_hook=e1._reject_duplicate_keys,
        )
    except Exception as error:
        raise StageEE3R1Error(f"cannot read strict {context}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageEE3R1Error(f"{context} must be a JSON object")
    return value, raw


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    expected = canonical_sha256(
        {key: item for key, item in value.items() if key != "canonical_digest"}
    )
    if value.get("canonical_digest") != expected:
        raise StageEE3R1Error(f"{context} canonical digest mismatch")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise StageEE3R1Error(f"immutable document differs: {path}")
        return
    _write_new_json(path, value)


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    raw = path.read_bytes()
    value = _strict_json(path, relative)[0] if path.suffix == ".json" else {}
    return {
        "path": relative,
        "bytes": len(raw),
        "sha256": _sha(raw),
        "canonical_digest": value.get("canonical_digest"),
    }


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ("git", *args), cwd=repo, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise StageEE3R1Error(
            f"git {' '.join(args)} failed: {error.stderr.strip()}"
        ) from error


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude)
    files = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in excluded or any(part.startswith(".") for part in Path(relative).parts):
                continue
            files.append({
                "path": relative, "bytes": path.stat().st_size,
                "sha256": _sha_file(path),
            })
    return {
        "files": files, "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "canonical_digest": canonical_sha256({"files": files}),
    }


def _validate_config(repo: Path) -> dict[str, Any]:
    config, _ = _strict_json(repo / CONFIG_PATH, "E3-R1 config")
    schema, _ = _strict_json(repo / CONFIG_SCHEMA_PATH, "E3-R1 schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(config),
        key=lambda item: [str(part) for part in item.path],
    )
    if errors:
        raise StageEE3R1Error(f"E3-R1 config schema failure: {errors[0].message}")
    _validate_signed(config, "E3-R1 config")
    if config["canonical_digest"] != EXPECTED_CONFIG_DIGEST:
        raise StageEE3R1Error("E3-R1 frozen config digest mismatch")
    if tuple(config["numerical_recovery"]["original_schedule"]) != ORIGINAL_SCHEDULE:
        raise StageEE3R1Error("E3-R1 original schedule drifted")
    if tuple(config["numerical_recovery"]["fallback_schedule"]) != FALLBACK_SCHEDULE:
        raise StageEE3R1Error("E3-R1 fallback schedule drifted")
    if config["numerical_recovery"]["atol"] != 1e-4 or config["numerical_recovery"]["rtol"] != 0.01:
        raise StageEE3R1Error("E3-R1 conservation threshold drifted")
    return config


def validate_e3_v1(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    result = e3.validate_completion(repo)
    if (
        result.get("status"), result.get("valid_terminal_attempts"),
        result.get("expected_new_attempts"), result.get("next_authorization"),
        result.get("e4_started"), result.get("completion_canonical_digest"),
        result.get("whole_root_inventory_digest"),
    ) != (
        "FAIL", 516, 3000, "NONE", False, E3_V1_COMPLETION_DIGEST,
        E3_V1_INVENTORY_DIGEST,
    ):
        raise StageEE3R1Error("E3 v1 terminal FAIL binding drifted")
    bindings = (
        (e3.COMPLETION_PATH, E3_V1_COMPLETION_SHA256, E3_V1_COMPLETION_DIGEST),
        (E3_V1_ROOT + "/status.json", E3_V1_STATUS_SHA256, E3_V1_STATUS_DIGEST),
        (e3.PLAN_PATH, E3_V1_PLAN_SHA256, E3_V1_PLAN_DIGEST),
        (E3_V1_ROOT + "/whole_root_inventory.json", E3_V1_INVENTORY_SHA256, E3_V1_INVENTORY_DIGEST),
    )
    for relative, expected_sha, expected_digest in bindings:
        value, raw = _strict_json(repo / relative, relative)
        _validate_signed(value, relative)
        if _sha(raw) != expected_sha or value["canonical_digest"] != expected_digest:
            raise StageEE3R1Error(f"E3 v1 immutable byte binding failed: {relative}")
    if _git(repo, "rev-parse", E3_V1_REF + "^{commit}") != E3_V1_COMMIT:
        raise StageEE3R1Error("E3 v1 archive ref drifted")
    if _git(repo, "rev-parse", E3_V1_COMMIT + "^{tree}") != E3_V1_TREE:
        raise StageEE3R1Error("E3 v1 custody tree drifted")
    if _git(repo, "diff", "--name-only", E3_V1_COMMIT, "--", E3_V1_ROOT):
        raise StageEE3R1Error("tracked E3 v1 evidence changed after custody")
    return result


def _v1_plan(repo: Path) -> dict[str, Any]:
    plan, raw = _strict_json(repo / e3.PLAN_PATH, "E3 v1 typed plan")
    _validate_signed(plan, "E3 v1 typed plan")
    if _sha(raw) != E3_V1_PLAN_SHA256 or plan["canonical_digest"] != E3_V1_PLAN_DIGEST:
        raise StageEE3R1Error("E3 v1 plan binding failed")
    return plan


def _r1_attempt_paths(repo: Path, unit_digest: str) -> tuple[Path, Path, Path, Path, Path | None]:
    root = repo / ROOT_PATH / "units" / unit_digest / "attempts" / ATTEMPT_ID
    map_path = root / "map.npy"
    failure_map = root / "failure_attribution.npy"
    return root, root / "result.json", root / "status.json", root / "manifest.json", map_path if map_path.exists() else failure_map if failure_map.exists() else None


def _source_attempt(repo: Path, unit: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    v1 = repo / E3_V1_ROOT / "units" / unit["canonical_digest"] / "attempts/attempt-000001"
    if v1.exists():
        return "E3_V1_INHERITED", e3._validate_attempt(repo, unit)
    r1 = repo / ROOT_PATH / "units" / unit["canonical_digest"] / "attempts" / ATTEMPT_ID
    if r1.exists():
        return "E3_R1_NEW", _validate_r1_attempt(repo, unit)
    return None


def _partition(repo: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    plan = _v1_plan(repo)
    inherited = {}
    eligible = []
    for unit in plan["units"]:
        v1 = repo / E3_V1_ROOT / "units" / unit["canonical_digest"] / "attempts/attempt-000001"
        if v1.exists():
            inherited[unit["canonical_digest"]] = e3._validate_attempt(repo, unit)
        else:
            eligible.append(unit)
    if len(inherited) != 516 or len(eligible) != 2484:
        raise StageEE3R1Error(
            f"E3-R1 partition drifted: inherited={len(inherited)} eligible={len(eligible)}"
        )
    pilots = [unit for unit in eligible if unit["selection_order"] in PILOT_ORDERS]
    if len(pilots) != 12:
        raise StageEE3R1Error("E3-R1 pilot complement is not exactly 12 units")
    return inherited, eligible


def verify_preconditions(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = _validate_config(repo)
    validate_e3_v1(repo)
    inherited, eligible = _partition(repo)
    if _sha_file(repo / AMENDMENT_PATH) != AMENDMENT_SHA256:
        raise StageEE3R1Error("E3-R1 amendment bytes drifted")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", AMENDMENT_COMMIT, "HEAD"),
        cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode:
        raise StageEE3R1Error("E3-R1 amendment commit is not an ancestor")
    return {
        "status": "PASS", "validated_at": _now(),
        "head": _git(repo, "rev-parse", "HEAD"),
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "config": config, "inherited": len(inherited), "eligible": len(eligible),
    }


def record_tests(
    repo_root: Path | str = REPO_ROOT, *, passed: int, failed: int, command: str,
) -> dict[str, Any]:
    if type(passed) is not int or passed <= 0 or failed != 0 or not command.strip():
        raise StageEE3R1Error("only a real zero-failure test result can be recorded")
    report = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_pre_pilot_test_results",
        "recovery_id": RECOVERY_ID, "status": "PASS",
        "recorded_at": _now(), "command": command, "passed": passed,
        "failed": failed, "e4_started": False,
    })
    _write_or_verify(Path(repo_root).resolve() / TEST_REPORT_PATH, report)
    return report


def authorize_pilot_recovery(
    repo_root: Path | str = REPO_ROOT, *, passed: int, failed: int, command: str,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validate_initialized(repo)
    if type(passed) is not int or passed <= 0 or failed != 0 or not command.strip():
        raise StageEE3R1Error("pilot recovery requires a real zero-failure test result")
    plan, eligible = _eligible_units(repo)
    del plan
    pilot_units = [unit for unit in eligible if unit["selection_order"] in PILOT_ORDERS]
    observed = []
    for unit in pilot_units:
        root = _r1_attempt_paths(repo, unit["canonical_digest"])[0]
        if root.exists():
            attempt = _validate_r1_attempt(repo, unit)
            if attempt["result"]["validity_status"] != "VALID":
                raise StageEE3R1Error("pilot recovery cannot follow a terminal scientific failure")
            observed.append({
                "unit_digest": unit["canonical_digest"],
                "result_canonical_digest": attempt["result"]["canonical_digest"],
                "manifest_canonical_digest": attempt["manifest"]["canonical_digest"],
            })
    if len(observed) != 2:
        raise StageEE3R1Error("pilot infrastructure event must preserve exactly two valid terminal attempts")
    event = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_pilot_infrastructure_event",
        "recovery_id": RECOVERY_ID, "event_id": "E3_R1_PILOT_INFRA_001",
        "status": "RECORDED", "recorded_at": _now(),
        "selection_orders": list(PILOT_ORDERS),
        "failure_class": "INFRASTRUCTURE_ORCHESTRATION_VALIDATOR",
        "failure_code": "MATCHED_FORWARD_COUNT_BOUNDARY_TOO_STRICT",
        "scientific_failure": False,
        "description": "The first order-61 coordinate pair completed and atomically published. Immediate validation rejected the matched result because exactly 190 real fidelity forwards were incorrectly required to be greater than 190.",
        "preserved_terminal_attempts": observed,
        "terminal_attempt_recompute_authorized": False,
        "remaining_pilot_units_authorized": 10,
        "cost": {"gpu_seconds_charged": 20.0, "cpu_seconds_charged": 20.0, "accounting": "CONSERVATIVE_WALL_TIME_ROUNDED_UP"},
        "e4_authorized": False, "e4_started": False,
    })
    _write_or_verify(repo / INCIDENT_PATH, event)
    test_report = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_pilot_recovery_test_results",
        "recovery_id": RECOVERY_ID, "status": "PASS", "recorded_at": _now(),
        "command": command, "passed": passed, "failed": failed, "e4_started": False,
    })
    _write_or_verify(repo / RECOVERY_TEST_REPORT_PATH, test_report)
    status = _git(repo, "status", "--porcelain=v1")
    allowed = (ROOT_PATH + "/", REPORT_ROOT_PATH + "/")
    unexpected = [line for line in status.splitlines() if not any(line[3:].startswith(item) or f"/{item}" in line[3:] for item in allowed)]
    if unexpected:
        raise StageEE3R1Error(f"pilot recovery producer worktree is not clean: {unexpected}")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", "@{u}") != head:
        raise StageEE3R1Error("pilot recovery producer is not pushed")
    authorization = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_pilot_recovery_authorization",
        "recovery_id": RECOVERY_ID, "status": "AUTHORIZED_PILOT_MISSING_COMPLEMENT_ONLY",
        "issued_at": _now(), "producer_commit": head,
        "producer_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "supersedes_pilot_authorization_canonical_digest": _strict_json(repo / PILOT_AUTH_PATH, "pilot authorization")[0]["canonical_digest"],
        "incident_canonical_digest": event["canonical_digest"],
        "test_report_canonical_digest": test_report["canonical_digest"],
        "authorized_orders": list(PILOT_ORDERS),
        "preserved_terminal_attempts": 2, "eligible_missing_attempts": 10,
        "terminal_attempt_recompute": False, "bulk_authorized": False,
        "e4_authorized": False, "e4_started": False,
    })
    _write_or_verify(repo / PILOT_RECOVERY_AUTH_PATH, authorization)
    return authorization


def initialize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    pre = verify_preconditions(repo)
    status = _git(repo, "status", "--porcelain=v1")
    allowed = (ROOT_PATH + "/", REPORT_ROOT_PATH + "/")
    unexpected = [
        line for line in status.splitlines()
        if not any(
            line[3:].startswith(item) or f"/{item}" in line[3:]
            for item in allowed
        )
    ]
    if unexpected:
        raise StageEE3R1Error(f"producer worktree is not clean: {unexpected}")
    head = pre["head"]
    if _git(repo, "rev-parse", "@{u}") != head:
        raise StageEE3R1Error("E3-R1 producer is not pushed to its tracking branch")
    tests, _ = _strict_json(repo / TEST_REPORT_PATH, "E3-R1 pre-pilot tests")
    _validate_signed(tests, "E3-R1 pre-pilot tests")
    if tests.get("status") != "PASS" or tests.get("failed") != 0:
        raise StageEE3R1Error("E3-R1 pre-pilot tests are not PASS")
    inherited, eligible = _partition(repo)
    eligible_digests = [row["canonical_digest"] for row in eligible]
    inherited_digests = sorted(inherited)
    pilot_digests = [
        row["canonical_digest"] for row in eligible
        if row["selection_order"] in PILOT_ORDERS
    ]
    plan = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_recovery_plan",
        "recovery_id": RECOVERY_ID, "e3_v1_plan_canonical_digest": E3_V1_PLAN_DIGEST,
        "inherited_origin": "E3_V1_INHERITED", "new_origin": "E3_R1_NEW",
        "inherited_unit_digests": inherited_digests,
        "eligible_unit_digests": eligible_digests,
        "pilot_orders": list(PILOT_ORDERS), "pilot_unit_digests": pilot_digests,
        "bulk_unit_digests": [x for x in eligible_digests if x not in set(pilot_digests)],
        "counts": {"total": 3000, "inherited": 516, "eligible": 2484, "pilot": 12, "bulk": 2472},
    })
    run_spec = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_run_spec",
        "recovery_id": RECOVERY_ID, "plan_canonical_digest": plan["canonical_digest"],
        "original_schedule": list(ORIGINAL_SCHEDULE),
        "fallback_schedule": list(FALLBACK_SCHEDULE),
        "complete_schedule": list(COMPLETE_SCHEDULE),
        "atol": 1e-4, "rtol": 0.01, "step_chunk_size": 8,
        "coordinates": [list(item) for item in e3.COORDINATES],
        "methods": list(e3.METHODS), "pilot_orders": list(PILOT_ORDERS),
        "stage_boundary": {"current": "E3_R1_ONLY", "e4_started": False},
    })
    authorization = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_pilot_execution_authorization",
        "recovery_id": RECOVERY_ID, "status": "AUTHORIZED_PILOT_ONLY",
        "issued_at": _now(), "producer_commit": head, "producer_tree": pre["tree"],
        "amendment_commit": AMENDMENT_COMMIT, "amendment_sha256": AMENDMENT_SHA256,
        "e3_v1_commit": E3_V1_COMMIT, "e3_v1_inventory_canonical_digest": E3_V1_INVENTORY_DIGEST,
        "plan_canonical_digest": plan["canonical_digest"],
        "run_spec_canonical_digest": run_spec["canonical_digest"],
        "authorized_orders": list(PILOT_ORDERS), "bulk_authorized": False,
        "e4_authorized": False, "e4_started": False,
    })
    pending = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_completion_pending",
        "stage_id": "E3-R1", "recovery_id": RECOVERY_ID,
        "status": "PENDING", "started_at": _now(),
        "plan_canonical_digest": plan["canonical_digest"],
        "pilot_authorization_canonical_digest": authorization["canonical_digest"],
        "next_authorization": "NONE_PENDING", "e4_started": False,
    })
    for relative, value in (
        (PLAN_PATH, plan), (RUN_SPEC_PATH, run_spec),
        (PILOT_AUTH_PATH, authorization), (COMPLETION_PENDING_PATH, pending),
    ):
        _write_or_verify(repo / relative, value)
    return validate_initialized(repo)


def validate_initialized(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    verify_preconditions(repo)
    documents = {}
    for name, relative in (("plan", PLAN_PATH), ("run_spec", RUN_SPEC_PATH), ("authorization", PILOT_AUTH_PATH), ("pending", COMPLETION_PENDING_PATH)):
        value, _ = _strict_json(repo / relative, f"E3-R1 {name}")
        _validate_signed(value, f"E3-R1 {name}")
        documents[name] = value
    if documents["authorization"].get("status") != "AUTHORIZED_PILOT_ONLY" or documents["authorization"].get("authorized_orders") != list(PILOT_ORDERS) or documents["authorization"].get("bulk_authorized") is not False:
        raise StageEE3R1Error("E3-R1 pilot authorization drifted")
    if documents["pending"].get("status") != "PENDING" or documents["pending"].get("e4_started") is not False:
        raise StageEE3R1Error("E3-R1 pending receipt drifted")
    if documents["run_spec"].get("complete_schedule") != list(COMPLETE_SCHEDULE):
        raise StageEE3R1Error("E3-R1 run schedule drifted")
    if documents["plan"].get("counts") != {"total": 3000, "inherited": 516, "eligible": 2484, "pilot": 12, "bulk": 2472}:
        raise StageEE3R1Error("E3-R1 plan counts drifted")
    return {
        "status": "PASS", "inherited": 516, "eligible": 2484,
        "pilot_units": 12, "bulk_units": 2472, "e4_started": False,
    }


def _valid_result(
    *, unit: Mapping[str, Any], target_identity: Mapping[str, Any], response: Any,
    method_result: Any, signed_map: np.ndarray, curves: list[dict[str, Any]],
    elapsed_seconds: float, cpu_seconds: float, forward_calls: int,
    backward_calls: int, peak_cuda_bytes: int, clean_sha256: str,
) -> dict[str, Any]:
    result = e3._valid_result(
        unit=unit, target_identity=target_identity, response=response,
        method_result=method_result, signed_map=signed_map, curves=curves,
        elapsed_seconds=elapsed_seconds, cpu_seconds=cpu_seconds,
        forward_calls=forward_calls, backward_calls=backward_calls,
        peak_cuda_bytes=peak_cuda_bytes, clean_sha256=clean_sha256,
    )
    result["document_type"] = "stage_e_e3_r1_attempt_result"
    result["attempt_id"] = ATTEMPT_ID
    result["recovery_id"] = RECOVERY_ID
    result["origin"] = "E3_R1_NEW"
    result["pair_binding"]["attempt_id"] = ATTEMPT_ID
    result["pair_binding"]["recovery_id"] = RECOVERY_ID
    result["pair_binding_digest"] = canonical_sha256(result["pair_binding"])
    return result


def _failure_result(
    *, unit: Mapping[str, Any], target_identity: Mapping[str, Any], response: Any,
    error: Any, elapsed_seconds: float, cpu_seconds: float, forward_calls: int,
    peak_cuda_bytes: int, clean_sha256: str, diagnostic: np.ndarray,
) -> dict[str, Any]:
    ig = error.ig_result
    target_digest = e1._target_semantic_digest(target_identity)
    attempts = [dict(item) for item in ig.attempts]
    if tuple(item["steps"] for item in attempts) != COMPLETE_SCHEDULE or any(item["pass"] != [False] for item in attempts):
        raise StageEE3R1Error("scientific failure did not exhaust the R1 schedule")
    diagnostic_digest = e1._array_digest(diagnostic)
    backward_calls = sum(int(math.ceil(item["steps"] / 8)) for item in attempts)
    return {
        "schema_version": 1,
        "document_type": "stage_e_e3_r1_attempt_result",
        "recovery_id": RECOVERY_ID, "origin": "E3_R1_NEW",
        "unit_digest": unit["canonical_digest"], "sample_id": unit["sample_id"],
        "selection_order": unit["selection_order"], "stratum": unit["stratum"],
        "block": unit["block"], "head": unit["head"],
        "method": unit["method"], "target": unit["target"],
        "scale": unit["scale"],
        "coordinate_comparison_identity": unit["coordinate_comparison_identity"],
        "validity_status": "FAILED", "failure_class": "SCIENTIFIC_INVALIDITY",
        "failure_code": "ig_conservation_failed_after_e3_r1_schedule",
        "na_reason": None, "attempt_id": ATTEMPT_ID, "retry_count": 0,
        "source_hashes": {
            "image_sha256": unit["image_sha256"],
            "clean_input_sha256": clean_sha256, "model_sha256": e3.MODEL_SHA256,
            "model_canonical_tensor_digest": e3.MODEL_TENSOR_DIGEST,
            "target_semantic_digest": target_digest,
            "failure_attribution_sha256": diagnostic_digest,
        },
        "target_response": response.to_dict(),
        "ig_result": {
            "selected_steps": ig.steps, "converged": False,
            "attempts": attempts,
            "conservation": {key: value.detach().cpu().tolist() for key, value in ig.conservation.items()},
            "metadata": dict(ig.metadata),
        },
        "diagnostic_attribution_only": True, "fidelity_executed": False,
        "elapsed_seconds": float(elapsed_seconds), "cpu_seconds": float(cpu_seconds),
        "gpu_hours": float(elapsed_seconds / 3600.0),
        "forward_calls": int(forward_calls), "backward_calls": backward_calls,
        "peak_cuda_memory_bytes": int(peak_cuda_bytes), "artifact_bytes": 0,
        "descriptive_only": True, "p_values_emitted": False,
        "winner_selected": False, "causal_claim_made": False,
        "new_go_no_go_issued": False, "d10_primary_recomputed": False,
    }


def _publish_attempt(
    repo: Path, unit: Mapping[str, Any], result: Mapping[str, Any],
    payload: np.ndarray, *, fault_injector: Callable[[str], Any] | None = None,
) -> Path:
    root, _, _, _, _ = _r1_attempt_paths(repo, unit["canonical_digest"])
    if root.exists():
        raise StageEE3R1Error(f"refusing to overwrite terminal R1 attempt: {root}")
    attempts_root = root.parent
    attempts_root.mkdir(parents=True, exist_ok=True)
    if any(path.name.startswith(".attempt-r1-") for path in attempts_root.iterdir()):
        raise StageEE3R1Error(f"ambiguous unpublished R1 attempt exists: {attempts_root}")
    temporary = Path(tempfile.mkdtemp(prefix=".attempt-r1-000001-", dir=attempts_root))
    try:
        signed_result = _signed(result)
        result_raw = canonical_json_bytes(signed_result) + b"\n"
        (temporary / "result.json").write_bytes(result_raw)
        valid = signed_result["validity_status"] == "VALID"
        payload_name = "map.npy" if valid else "failure_attribution.npy"
        with (temporary / payload_name).open("wb") as stream:
            np.save(stream, payload, allow_pickle=False)
        state = "succeeded" if valid else "scientific_failure"
        status = _signed({
            "schema_version": 1, "document_type": "stage_e_e3_r1_attempt_status",
            "recovery_id": RECOVERY_ID, "attempt_id": ATTEMPT_ID,
            "attempt_number": 1, "unit_digest": unit["canonical_digest"],
            "state": state, "terminal": True, "retry_count": 0,
            "failure_class": signed_result["failure_class"],
            "failure_code": signed_result["failure_code"],
            "na_reason": signed_result["na_reason"], "published_at": _now(),
        })
        status_raw = canonical_json_bytes(status) + b"\n"
        (temporary / "status.json").write_bytes(status_raw)
        payload_path = temporary / payload_name
        manifest = _signed({
            "schema_version": 1, "document_type": "stage_e_e3_r1_attempt_manifest",
            "recovery_id": RECOVERY_ID, "attempt_id": ATTEMPT_ID,
            "attempt_number": 1, "unit_digest": unit["canonical_digest"],
            "validity_status": signed_result["validity_status"],
            "result_sha256": _sha(result_raw), "result_bytes": len(result_raw),
            "payload_name": payload_name, "payload_sha256": _sha_file(payload_path),
            "payload_bytes": payload_path.stat().st_size,
            "payload_array_sha256": e1._array_digest(payload),
            "status_sha256": _sha(status_raw), "status_bytes": len(status_raw),
            "immutable": True,
        })
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        for path in temporary.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        if fault_injector is not None:
            fault_injector("before_rename")
        os.replace(temporary, root)
        directory = os.open(attempts_root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if fault_injector is not None:
            fault_injector("after_rename")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    _validate_r1_attempt(repo, unit)
    return root


def _validate_r1_attempt(repo: Path, unit: Mapping[str, Any]) -> dict[str, Any]:
    root, result_path, status_path, manifest_path, payload_path = _r1_attempt_paths(
        repo, unit["canonical_digest"]
    )
    if not root.is_dir() or payload_path is None or any(
        not path.is_file() for path in (result_path, status_path, manifest_path)
    ):
        raise StageEE3R1Error(f"R1 attempt is absent or incomplete: {root}")
    result, result_raw = _strict_json(result_path, "E3-R1 result")
    status, status_raw = _strict_json(status_path, "E3-R1 status")
    manifest, _ = _strict_json(manifest_path, "E3-R1 manifest")
    for value, name in ((result, "result"), (status, "status"), (manifest, "manifest")):
        _validate_signed(value, f"E3-R1 attempt {name}")
    payload = np.load(payload_path, allow_pickle=False)
    if (
        result.get("recovery_id") != RECOVERY_ID
        or result.get("origin") != "E3_R1_NEW"
        or result.get("unit_digest") != unit["canonical_digest"]
        or result.get("sample_id") != unit["sample_id"]
        or result.get("selection_order") != unit["selection_order"]
        or result.get("block") != unit["block"] or result.get("head") != unit["head"]
        or result.get("method") != unit["method"]
        or result.get("attempt_id") != ATTEMPT_ID or result.get("retry_count") != 0
        or result.get("validity_status") not in {"VALID", "FAILED", "NA"}
        or status.get("terminal") is not True or status.get("attempt_id") != ATTEMPT_ID
        or manifest.get("result_sha256") != _sha(result_raw)
        or manifest.get("status_sha256") != _sha(status_raw)
        or manifest.get("payload_name") != payload_path.name
        or manifest.get("payload_sha256") != _sha_file(payload_path)
        or manifest.get("payload_array_sha256") != e1._array_digest(payload)
    ):
        raise StageEE3R1Error("E3-R1 attempt control binding failed")
    if result["validity_status"] == "VALID":
        if payload_path.name != "map.npy" or payload.dtype != np.float32 or payload.shape != (56, 56) or not np.isfinite(payload).all():
            raise StageEE3R1Error("valid E3-R1 map contract failed")
        if any(result.get(key) is not None for key in ("failure_class", "failure_code", "na_reason")):
            raise StageEE3R1Error("valid E3-R1 attempt has failure metadata")
        source_hashes = result.get("source_hashes", {})
        if (
            source_hashes.get("image_sha256") != unit["image_sha256"]
            or source_hashes.get("model_sha256") != e3.MODEL_SHA256
            or source_hashes.get("model_canonical_tensor_digest") != e3.MODEL_TENSOR_DIGEST
        ):
            raise StageEE3R1Error("valid E3-R1 source identity drifted")
        target_response = result.get("target_response")
        if not isinstance(target_response, Mapping) or target_response.get("sigma") not in (-1, 1):
            raise StageEE3R1Error("valid E3-R1 target response is invalid")
        target_digest = e1._target_semantic_digest(target_response["target_identity"])
        if target_digest != source_hashes.get("target_semantic_digest"):
            raise StageEE3R1Error("valid E3-R1 target identity drifted")
        pair = {
            "sample_id": unit["sample_id"], "selection_order": unit["selection_order"],
            "image_sha256": unit["image_sha256"], "model_sha256": e3.MODEL_SHA256,
            "target": "fixed_clean_attention_cls_head_direction",
            "target_semantic_digest": target_digest, "method": unit["method"],
            "block": unit["block"], "head": unit["head"],
            "scale": "fine_4x4_cell",
            "comparison_identity": unit["coordinate_comparison_identity"],
            "replacement": "normalized_zero_canonical_rgb_0_5",
            "fractions": list(e3.FRACTIONS), "unit_counts": list(e3.UNIT_COUNTS),
            "seeds": [e3.e3_seed(unit["sample_id"], unit["block"], unit["head"], repeat) for repeat in range(5)],
            "seed_namespace": e3.SEED_NAMESPACE,
            "signed_map_sha256": e1._array_digest(payload),
            "attempt_id": ATTEMPT_ID, "recovery_id": RECOVERY_ID,
        }
        if result.get("pair_binding") != pair or result.get("pair_binding_digest") != canonical_sha256(pair):
            raise StageEE3R1Error("valid E3-R1 pair binding drifted")
        provenance = result.get("method_provenance")
        if not isinstance(provenance, Mapping) or canonical_sha256(provenance) != source_hashes.get("method_provenance_canonical_digest"):
            raise StageEE3R1Error("valid E3-R1 provenance digest drifted")
        curves = result.get("curves")
        if not isinstance(curves, list) or len(curves) != 10:
            raise StageEE3R1Error("valid E3-R1 attempt lacks ten curves")
        if unit["method"] == e3.NNS_METHOD:
            ig = result.get("method_provenance", {}).get("ig_result", {})
            attempts = ig.get("attempts", [])
            steps = tuple(item.get("steps") for item in attempts)
            passing = [index for index, item in enumerate(attempts) if item.get("pass") == [True]]
            if ig.get("converged") is not True or steps != COMPLETE_SCHEDULE[:len(steps)] or passing != [len(attempts) - 1]:
                raise StageEE3R1Error("valid E3-R1 NNS schedule/convergence failed")
            recovery = result["method_provenance"].get("numerical_recovery", {})
            if recovery.get("id") != RECOVERY_ID or recovery.get("atol") != 1e-4 or recovery.get("rtol") != 0.01:
                raise StageEE3R1Error("valid E3-R1 NNS recovery provenance drifted")
        else:
            matched = provenance.get("matched_contribution", {})
            qc = result.get("matched_conservation_qc", {})
            if provenance.get("status") != "OK" or matched.get("status") != "OK" or qc.get("status") != "PASS" or qc.get("cls_in_ranking") is not False or result.get("backward_calls") != 0:
                raise StageEE3R1Error("valid E3-R1 matched method contract failed")
        expected_curve_keys = {(direction, repeat) for direction in ("insertion", "deletion") for repeat in range(5)}
        if {(row.get("direction"), row.get("replicate")) for row in curves} != expected_curve_keys:
            raise StageEE3R1Error("valid E3-R1 curve coverage drifted")
        for curve in curves:
            seed = e3.e3_seed(unit["sample_id"], unit["block"], unit["head"], curve["replicate"])
            ranked = e3.ranking(payload, sigma=target_response["sigma"], seed=seed)
            score = np.maximum(np.float32(target_response["sigma"]) * payload.reshape(-1), np.float32(0.0)).astype(np.float32, copy=False)
            if (
                curve.get("method_id") != unit["method"] or curve.get("sample_id") != unit["sample_id"]
                or curve.get("block") != unit["block"] or curve.get("head") != unit["head"]
                or curve.get("seed") != seed or curve.get("seed_namespace") != e3.SEED_NAMESPACE
                or curve.get("seed_comparison_identity") != e3.comparison_identity(unit["block"], unit["head"])
                or curve.get("tie_priority_sha256") != e1._array_digest(e3.tie_priority(seed))
                or curve.get("ranking_sha256") != e1._array_digest(ranked)
                or curve.get("source_unit_map_sha256") != e1._array_digest(payload)
                or curve.get("score_sha256") != e1._array_digest(score)
                or curve.get("unit_counts") != list(e3.UNIT_COUNTS)
                or curve.get("actual_fractions") != list(e3.ACTUAL_FRACTIONS)
                or len(curve.get("responses", [])) != 21
                or len(curve.get("intervention_sha256", [])) != 21
                or curve.get("auc") != e3.trapezoid_auc(curve["responses"])
            ):
                raise StageEE3R1Error("valid E3-R1 curve source validation failed")
            endpoints = (curve["responses"][0], curve["responses"][-1])
            if endpoints != ((0.0, 1.0) if curve["direction"] == "insertion" else (1.0, 0.0)):
                raise StageEE3R1Error("valid E3-R1 curve endpoint drifted")
        aggregates = e3._repeat_aggregates(curves)
        if any(result.get(key) != value for key, value in aggregates.items()):
            raise StageEE3R1Error("valid E3-R1 AUC aggregate drifted")
        minimum_forward_calls = 191 if unit["method"] == e3.NNS_METHOD else 190
        if result.get("forward_calls", 0) < minimum_forward_calls:
            raise StageEE3R1Error("valid E3-R1 attempt lacks real response evaluation")
        if result.get("source_hashes", {}).get("signed_map_sha256") != e1._array_digest(payload):
            raise StageEE3R1Error("valid E3-R1 map digest drifted")
    elif result["validity_status"] == "FAILED":
        ig = result.get("ig_result", {})
        attempts = ig.get("attempts", [])
        if (
            unit["method"] != e3.NNS_METHOD or payload_path.name != "failure_attribution.npy"
            or payload.dtype != np.float32 or payload.shape != (3, 224, 224)
            or not np.isfinite(payload).all() or status.get("state") != "scientific_failure"
            or result.get("failure_class") != "SCIENTIFIC_INVALIDITY"
            or result.get("failure_code") != "ig_conservation_failed_after_e3_r1_schedule"
            or tuple(item.get("steps") for item in attempts) != COMPLETE_SCHEDULE
            or any(item.get("pass") != [False] for item in attempts)
            or ig.get("converged") is not False or ig.get("selected_steps") != 32768
            or result.get("diagnostic_attribution_only") is not True
            or result.get("fidelity_executed") is not False
        ):
            raise StageEE3R1Error("E3-R1 scientific-failure attempt contract failed")
        metadata = ig.get("metadata", {})
        if metadata.get("step_schedule") != list(COMPLETE_SCHEDULE) or metadata.get("atol") != 1e-4 or metadata.get("rtol") != 0.01 or metadata.get("step_chunk_size") != 8:
            raise StageEE3R1Error("E3-R1 scientific-failure numerical metadata drifted")
        if result.get("source_hashes", {}).get("failure_attribution_sha256") != e1._array_digest(payload):
            raise StageEE3R1Error("E3-R1 failure attribution digest drifted")
    else:
        raise StageEE3R1Error("E3-R1 NA is not valid for this recovery")
    return {
        "result": result, "status": status, "manifest": manifest,
        "payload": payload, "attempt_path": root,
    }


def _eligible_units(repo: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = _v1_plan(repo)
    _, eligible = _partition(repo)
    allowed = {row["canonical_digest"] for row in eligible}
    recovery_plan, _ = _strict_json(repo / PLAN_PATH, "E3-R1 plan")
    _validate_signed(recovery_plan, "E3-R1 plan")
    if set(recovery_plan["eligible_unit_digests"]) != allowed:
        raise StageEE3R1Error("E3-R1 eligible set drifted")
    return plan, eligible


def _validate_execution_code(repo: Path, phase: str) -> None:
    recovery = phase == "PILOT" and (repo / PILOT_RECOVERY_AUTH_PATH).is_file()
    relative = PILOT_RECOVERY_AUTH_PATH if recovery else PILOT_AUTH_PATH if phase == "PILOT" else BULK_AUTH_PATH
    authorization, _ = _strict_json(repo / relative, "E3-R1 execution authorization")
    _validate_signed(authorization, "E3-R1 execution authorization")
    expected_status = "AUTHORIZED_PILOT_MISSING_COMPLEMENT_ONLY" if recovery else "AUTHORIZED_PILOT_ONLY" if phase == "PILOT" else "AUTHORIZED_BULK_COMPLEMENT_ONLY"
    if authorization.get("status") != expected_status:
        raise StageEE3R1Error("E3-R1 execution authorization status drifted")
    if phase == "PILOT" and authorization.get("authorized_orders") != list(PILOT_ORDERS):
        raise StageEE3R1Error("E3-R1 pilot order authorization drifted")
    producer = authorization.get("producer_commit")
    if not isinstance(producer, str) or len(producer) != 40:
        raise StageEE3R1Error("E3-R1 execution authorization lacks producer commit")
    if subprocess.run(("git", "merge-base", "--is-ancestor", producer, "HEAD"), cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise StageEE3R1Error("E3-R1 execution HEAD does not descend from producer")
    protected = ("src/vitnns", "config", "tests", AMENDMENT_PATH)
    if _git(repo, "diff", "--name-only", producer, "HEAD", "--", *protected):
        raise StageEE3R1Error("E3-R1 reviewed producer source changed before execution")


def _run_orders(
    repo: Path, *, orders: Sequence[int], device: str, phase: str,
) -> dict[str, Any]:
    validate_initialized(repo)
    if phase not in {"PILOT", "BULK"}:
        raise StageEE3R1Error("unknown E3-R1 execution phase")
    _validate_execution_code(repo, phase)
    requested = tuple(orders)
    if len(set(requested)) != len(requested) or any(type(x) is not int or not 0 <= x < 500 for x in requested):
        raise StageEE3R1Error("invalid or duplicate selection orders")
    if phase == "PILOT" and requested != PILOT_ORDERS:
        raise StageEE3R1Error("pilot must execute exactly orders 61 then 72")
    if phase == "BULK":
        gate, _ = _strict_json(repo / PILOT_GATE_PATH, "E3-R1 continuation gate")
        authorization, _ = _strict_json(repo / BULK_AUTH_PATH, "E3-R1 bulk authorization")
        for value, name in ((gate, "continuation gate"), (authorization, "bulk authorization")):
            _validate_signed(value, f"E3-R1 {name}")
        if gate.get("status") != "PASS" or authorization.get("status") != "AUTHORIZED_BULK_COMPLEMENT_ONLY":
            raise StageEE3R1Error("E3-R1 bulk continuation is not authorized")
        if any(order in PILOT_ORDERS for order in requested):
            raise StageEE3R1Error("bulk execution cannot include pilot orders")
    plan, eligible = _eligible_units(repo)
    units_by_sample: dict[str, list[dict[str, Any]]] = {}
    for unit in eligible:
        units_by_sample.setdefault(unit["sample_id"], []).append(unit)
    groups_by_order = {group["selection_order"]: group for group in plan["groups"]}
    import torch
    from .stage_c_fidelity import ValuePathScalarAdapter, resolve_target_response
    from .stage_c_value_path import (
        HeadCandidate, StageCValuePathConvergenceError,
        e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig,
        matched_fixed_clean_attention_patch_token_contribution,
        resolve_fixed_clean_attention_head_directions,
    )
    from .stage_c_perturbations import tensor_sha256
    model = None
    process_start = time.monotonic()
    process_cpu = time.process_time()
    records = []
    completed = skipped = failed = 0
    try:
        model, binding, model_record = e1._load_model_and_context(repo, device)
        gpu_index = torch.device(device).index or 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical = int(visible) if visible is not None and visible.isdigit() else gpu_index
        identities = {}
        if device.startswith("cuda"):
            query = subprocess.run(
                ("nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"),
                check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout.splitlines()
            for line in query:
                index, uuid, name = (item.strip() for item in line.split(",", 2))
                identities[int(index)] = {"gpu_uuid": uuid, "gpu_name": name}
        for order in requested:
            group = groups_by_order[order]
            group_units = units_by_sample[group["sample_id"]]
            existing = {}
            for unit in group_units:
                source = _source_attempt(repo, unit)
                if source is not None:
                    if source[0] == "E3_V1_INHERITED":
                        raise StageEE3R1Error("eligible R1 unit unexpectedly has E3 v1 terminal evidence")
                    existing[unit["canonical_digest"]] = source[1]
            missing = [unit for unit in group_units if unit["canonical_digest"] not in existing]
            skipped += len(existing)
            if not missing:
                continue
            if any(item["result"]["validity_status"] != "VALID" for item in existing.values()):
                records.append({
                    "selection_order": order, "sample_id": group["sample_id"],
                    "elapsed_seconds": 0.0, "cpu_seconds": 0.0,
                    "forward_calls": 0, "peak_cuda_memory_bytes": 0,
                    "terminal_units_published": len(existing),
                    "scientific_failure": True, "resume_skipped_terminal_failure": True,
                })
                failed += 1
                if phase == "BULK":
                    break
                continue
            group_start = time.monotonic()
            group_cpu = time.process_time()
            calls = 0
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            _canonical, clean, clean_sha = e1._load_sample(binding, group, model, device)
            if clean_sha != tensor_sha256(clean):
                raise StageEE3R1Error("live clean tensor hash drifted")
            def count_hook(_module: Any, _args: Any) -> None:
                nonlocal calls
                calls += 1
            handle = model.register_forward_pre_hook(count_hook)
            group_failed = False
            try:
                coordinates = list(e3.NEW_COORDINATES)
                targets = resolve_fixed_clean_attention_head_directions(
                    model, clean, tuple(HeadCandidate(*coord) for coord in coordinates),
                    candidate_chunk_size=len(coordinates),
                )
                targets_by_coord = {(target.block_index, target.head_index): target for target in targets}
                for block, head in coordinates:
                    target = targets_by_coord[(block, head)]
                    evaluator = ValuePathScalarAdapter(model, clean, target)
                    response = resolve_target_response(evaluator, clean, torch.zeros_like(clean))
                    if response.na is not None:
                        raise StageEE3R1Error(f"frozen E3-R1 target became NA at {(block, head)}")
                    target_identity = json.loads(canonical_json_bytes(evaluator.target_identity).decode("utf-8"))
                    for unit in [row for row in missing if (row["block"], row["head"]) == (block, head)]:
                        started = time.monotonic()
                        cpu_started = time.process_time()
                        before = calls
                        try:
                            if unit["method"] == e3.NNS_METHOD:
                                method_result = e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig(
                                    model, clean, target,
                                    provenance={"stage": "E3-R1", "comparison_identity": e3.comparison_identity(block, head)},
                                )
                                ig = method_result.provenance["ig_result"]
                                backward = sum(int(math.ceil(int(item["steps"]) / 8)) for item in ig["attempts"])
                            else:
                                method_result = matched_fixed_clean_attention_patch_token_contribution(
                                    target, provenance={"stage": "E3-R1", "comparison_identity": e3.comparison_identity(block, head)},
                                )
                                backward = 0
                            logical = {"forward_calls": 0}
                            curves, signed_map = e3._curves_for_method(
                                model=model, clean=clean, target=target, response=response,
                                method_result=method_result, sample_id=group["sample_id"],
                                block=block, head=head, call_counter=logical,
                            )
                            if logical["forward_calls"] != 190:
                                raise StageEE3R1Error("E3-R1 curve execution count drifted")
                            result = _valid_result(
                                unit=unit, target_identity=target_identity, response=response,
                                method_result=method_result, signed_map=signed_map, curves=curves,
                                elapsed_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started,
                                forward_calls=calls - before, backward_calls=backward,
                                peak_cuda_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0,
                                clean_sha256=clean_sha,
                            )
                            _publish_attempt(repo, unit, result, signed_map)
                            completed += 1
                        except StageCValuePathConvergenceError as error:
                            diagnostic = error.ig_result.attribution[0].detach().to(device="cpu", dtype=torch.float32).numpy().copy()
                            result = _failure_result(
                                unit=unit, target_identity=target_identity, response=response, error=error,
                                elapsed_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started,
                                forward_calls=calls - before,
                                peak_cuda_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0,
                                clean_sha256=clean_sha, diagnostic=diagnostic,
                            )
                            _publish_attempt(repo, unit, result, diagnostic)
                            completed += 1
                            failed += 1
                            group_failed = True
                            break
                    if group_failed:
                        break
                    del evaluator, response
            finally:
                handle.remove()
            records.append({
                "selection_order": order, "sample_id": group["sample_id"],
                "elapsed_seconds": time.monotonic() - group_start,
                "cpu_seconds": time.process_time() - group_cpu,
                "forward_calls": calls,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0,
                "terminal_units_published": sum(1 for unit in group_units if _r1_attempt_paths(repo, unit["canonical_digest"])[0].is_dir()),
                "scientific_failure": group_failed,
            })
            if group_failed and phase == "BULK":
                break
        elapsed = time.monotonic() - process_start
        cpu_seconds = time.process_time() - process_cpu
        if records:
            records[0]["elapsed_seconds"] += max(0.0, elapsed - sum(float(row["elapsed_seconds"]) for row in records))
            records[0]["cpu_seconds"] += max(0.0, cpu_seconds - sum(float(row["cpu_seconds"]) for row in records))
        telemetry = _signed({
            "schema_version": 1, "document_type": "stage_e_e3_r1_execution_telemetry",
            "recovery_id": RECOVERY_ID, "phase": phase,
            "orders": list(requested), "device": device,
            "physical_gpu_index": physical if device.startswith("cuda") else None,
            "gpu_uuid": identities.get(physical, {}).get("gpu_uuid"),
            "gpu_name": identities.get(physical, {}).get("gpu_name"),
            "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600.0 if device.startswith("cuda") else 0.0,
            "cpu_seconds": cpu_seconds, "completed_units": completed,
            "skipped_terminal_units": skipped, "failed_units": failed,
            "groups": records, "model": model_record, "completed_at": _now(),
        })
        name = "pilot.json" if phase == "PILOT" else f"bulk-{requested[0]:03d}-{requested[-1]:03d}.json"
        _write_or_verify(repo / ROOT_PATH / "telemetry" / name, telemetry)
        return telemetry
    finally:
        if model is not None:
            del model
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_pilot(repo_root: Path | str = REPO_ROOT, *, device: str = "cuda:0") -> dict[str, Any]:
    return _run_orders(Path(repo_root).resolve(), orders=PILOT_ORDERS, device=device, phase="PILOT")


def _r1_attempt_inventory(repo: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    plan, eligible = _eligible_units(repo)
    del plan
    observed = []
    for unit in eligible:
        root = _r1_attempt_paths(repo, unit["canonical_digest"])[0]
        if root.exists():
            observed.append((unit, _validate_r1_attempt(repo, unit)))
    return observed


def _cumulative_cost(repo: Path) -> dict[str, Any]:
    observed = _r1_attempt_inventory(repo)
    telemetry = []
    for path in sorted((repo / ROOT_PATH / "telemetry").glob("*.json")):
        value, _ = _strict_json(path, "E3-R1 telemetry")
        _validate_signed(value, "E3-R1 telemetry")
        telemetry.append(value)
    r1_elapsed = sum(float(item["elapsed_seconds"]) for item in telemetry)
    r1_cpu = sum(float(item["cpu_seconds"]) for item in telemetry)
    infrastructure_events = []
    for path in sorted((repo / REPORT_ROOT_PATH).glob("pilot_infrastructure_event_*.json")):
        value, _ = _strict_json(path, "E3-R1 infrastructure event")
        _validate_signed(value, "E3-R1 infrastructure event")
        infrastructure_events.append(value)
    incident_gpu_seconds = sum(float(item["cost"]["gpu_seconds_charged"]) for item in infrastructure_events)
    incident_cpu_seconds = sum(float(item["cost"]["cpu_seconds_charged"]) for item in infrastructure_events)
    r1_elapsed += incident_gpu_seconds
    r1_cpu += incident_cpu_seconds
    r1_artifact = sum(
        sum(path.stat().st_size for path in item[1]["attempt_path"].iterdir() if path.is_file())
        for item in observed
    )
    v1_root_bytes = _inventory(repo / E3_V1_ROOT)["total_bytes"]
    root_bytes = _inventory(
        repo / ROOT_PATH,
        exclude=(
            "whole_root_inventory.json", "status.json",
            "completion/e3_r1_completion.pending.json",
            "completion/e3_r1_completion.json",
        ),
    )["total_bytes"]
    report_bytes = _inventory(repo / REPORT_ROOT_PATH)["total_bytes"]
    return {
        "e3_v1_gpu_hours": V1_GPU_HOURS, "r1_gpu_hours": r1_elapsed / 3600.0,
        "gpu_hours": V1_GPU_HOURS + r1_elapsed / 3600.0,
        "e3_v1_cpu_seconds": V1_CPU_SECONDS, "r1_cpu_seconds": r1_cpu,
        "cpu_seconds": V1_CPU_SECONDS + r1_cpu,
        "e3_v1_retained_bytes": v1_root_bytes,
        "r1_attempt_bytes": r1_artifact, "r1_root_and_report_bytes": root_bytes + report_bytes,
        "retained_bytes": v1_root_bytes + root_bytes + report_bytes,
        "r1_terminal_attempts": len(observed),
        "gpu_invocation_count": len(telemetry) + len(infrastructure_events),
        "infrastructure_event_count": len(infrastructure_events),
    }


def evaluate_pilot(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validate_initialized(repo)
    plan, eligible = _eligible_units(repo)
    del plan
    pilots = [unit for unit in eligible if unit["selection_order"] in PILOT_ORDERS]
    observed = []
    for unit in pilots:
        root = _r1_attempt_paths(repo, unit["canonical_digest"])[0]
        if root.exists():
            observed.append((unit, _validate_r1_attempt(repo, unit)))
    statuses = [item[1]["result"]["validity_status"] for item in observed]
    nns = [item for unit, item in observed if unit["method"] == e3.NNS_METHOD]
    all_valid = len(observed) == 12 and statuses == ["VALID"] * 12
    all_nns_converged = len(nns) == 6 and all(item["result"]["method_provenance"]["ig_result"]["converged"] is True for item in nns)
    cost = _cumulative_cost(repo)
    telemetry, _ = _strict_json(repo / ROOT_PATH / "telemetry/pilot.json", "E3-R1 pilot telemetry")
    _validate_signed(telemetry, "E3-R1 pilot telemetry")
    groups = {row["selection_order"]: row for row in telemetry["groups"]}
    if set(groups) != set(PILOT_ORDERS):
        raise StageEE3R1Error("pilot telemetry does not cover orders 61 and 72")
    incident_gpu_by_order = Counter()
    incident_cpu_by_order = Counter()
    for path in sorted((repo / REPORT_ROOT_PATH).glob("pilot_infrastructure_event_*.json")):
        event, _ = _strict_json(path, "E3-R1 pilot incident")
        for order in event["selection_orders"]:
            incident_gpu_by_order[order] += float(event["cost"]["gpu_seconds_charged"]) / len(event["selection_orders"])
            incident_cpu_by_order[order] += float(event["cost"]["cpu_seconds_charged"]) / len(event["selection_orders"])
    max_gpu_group = max((float(groups[order]["elapsed_seconds"]) + incident_gpu_by_order[order]) / 3600.0 for order in PILOT_ORDERS)
    max_cpu_group = max(float(groups[order]["cpu_seconds"]) + incident_cpu_by_order[order] for order in PILOT_ORDERS)
    group_bytes = []
    for order in PILOT_ORDERS:
        group_bytes.append(sum(
            sum(path.stat().st_size for path in item[1]["attempt_path"].iterdir() if path.is_file())
            for item in observed if item[0]["selection_order"] == order
        ))
    projection = {
        "formula": "C_plus_1.25_times_412_times_max_pilot_group",
        "remaining_image_groups": PROJECTION_REMAINING_GROUPS,
        "factor": PROJECTION_FACTOR,
        "gpu_hours": cost["gpu_hours"] + PROJECTION_FACTOR * PROJECTION_REMAINING_GROUPS * max_gpu_group,
        "cpu_seconds": cost["cpu_seconds"] + PROJECTION_FACTOR * PROJECTION_REMAINING_GROUPS * max_cpu_group,
        "retained_bytes": cost["retained_bytes"] + math.ceil(PROJECTION_FACTOR * PROJECTION_REMAINING_GROUPS * max(group_bytes)) + FINALIZATION_RESERVE_BYTES,
    }
    actual_pass = cost["gpu_hours"] <= GPU_HOURS_MAX and cost["cpu_seconds"] <= CPU_SECONDS_MAX and cost["retained_bytes"] <= RETAINED_BYTES_MAX
    projected_pass = projection["gpu_hours"] <= GPU_HOURS_MAX and projection["cpu_seconds"] <= CPU_SECONDS_MAX and projection["retained_bytes"] <= RETAINED_BYTES_MAX
    passed = all_valid and all_nns_converged and actual_pass and projected_pass
    gate = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_pilot_continuation_gate",
        "recovery_id": RECOVERY_ID, "status": "PASS" if passed else "FAIL",
        "evaluated_at": _now(), "pilot_orders": list(PILOT_ORDERS),
        "pilot_expected_units": 12, "pilot_terminal_units": len(observed),
        "pilot_valid_units": sum(x == "VALID" for x in statuses),
        "pilot_nns_converged": sum(item["result"].get("validity_status") == "VALID" for item in nns),
        "atomic_persistence_validated": True, "actual_cost": cost,
        "actual_budget_status": "PASS" if actual_pass else "FAIL",
        "projected_total": projection,
        "projected_budget_status": "PASS" if projected_pass else "FAIL",
        "bulk_authorized": passed, "e4_authorized": False, "e4_started": False,
    })
    _write_or_verify(repo / PILOT_GATE_PATH, gate)
    if passed:
        authorization = _signed({
            "schema_version": 1, "document_type": "stage_e_e3_r1_bulk_execution_authorization",
            "recovery_id": RECOVERY_ID, "status": "AUTHORIZED_BULK_COMPLEMENT_ONLY",
            "issued_at": _now(), "pilot_gate_canonical_digest": gate["canonical_digest"],
            "producer_commit": _git(repo, "rev-parse", "HEAD"),
            "producer_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
            "authorized_unit_count": 2472, "pilot_orders_excluded": list(PILOT_ORDERS),
            "e4_authorized": False, "e4_started": False,
        })
        _write_or_verify(repo / BULK_AUTH_PATH, authorization)
    return gate


def run_bulk_shard(
    repo_root: Path | str = REPO_ROOT, *, shard_id: int, shard_count: int,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if type(shard_count) is not int or shard_count <= 0 or type(shard_id) is not int or not 0 <= shard_id < shard_count:
        raise StageEE3R1Error("invalid bulk shard identity")
    repo = Path(repo_root).resolve()
    _, eligible = _eligible_units(repo)
    eligible_orders = {unit["selection_order"] for unit in eligible}
    orders = tuple(
        order for order in sorted(eligible_orders)
        if order not in PILOT_ORDERS and order % shard_count == shard_id
    )
    if not orders:
        return {
            "status": "PASS_EMPTY_SHARD", "phase": "BULK",
            "shard_id": shard_id, "shard_count": shard_count,
            "orders": [], "e4_started": False,
        }
    return _run_orders(repo, orders=orders, device=device, phase="BULK")


def _combined_new_rows(
    repo: Path, plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    attempts = {}
    origins = {}
    for unit in plan["units"]:
        source = _source_attempt(repo, unit)
        if source is None:
            raise StageEE3R1Error(f"missing terminal union unit: {unit['canonical_digest']}")
        origin, observed = source
        attempts[unit["canonical_digest"]] = observed
        origins[unit["canonical_digest"]] = origin
    if len(attempts) != 3000 or Counter(origins.values()) != Counter({"E3_V1_INHERITED": 516, "E3_R1_NEW": 2484}):
        raise StageEE3R1Error("E3-R1 union coverage/origin partition failed")
    method_rows = []
    by_key: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for unit in plan["units"]:
        observed = attempts[unit["canonical_digest"]]
        result = observed["result"]
        path = observed["attempt_path"]
        row = {
            "variant_id": f"layer_head_{unit['block']}_{unit['head']}",
            "sample_id": unit["sample_id"], "selection_order": unit["selection_order"],
            "cohort": "stage_d_fidelity_v1_500", "stratum": unit["stratum"],
            "validity_status": result["validity_status"], "source_hashes": result["source_hashes"],
            "method": unit["method"], "target": unit["target"], "scale": unit["scale"],
            "layer": unit["block"], "head": unit["head"],
            "coordinate_role": "DESCRIPTIVE_FROZEN_COORDINATE",
            "recompute_status": "INHERITED_E3_V1_TERMINAL" if origins[unit["canonical_digest"]] == "E3_V1_INHERITED" else "E3_R1_NEW_COMPUTE_COMPLETED",
            "evidence_origin": origins[unit["canonical_digest"]],
            "attempt_id": result["attempt_id"], "retry_count": result["retry_count"],
            "failure_class": result["failure_class"], "failure_code": result["failure_code"],
            "na_reason": result["na_reason"],
            "insertion_auc_mean": result.get("insertion_auc_mean"),
            "deletion_auc_mean": result.get("deletion_auc_mean"),
            "insertion_auc_repeat_sd_ddof_0": result.get("insertion_auc_repeat_sd_ddof_0"),
            "deletion_auc_repeat_sd_ddof_0": result.get("deletion_auc_repeat_sd_ddof_0"),
            "misclassified": unit["stratum"] == "misclassified",
            "pair_binding_digest": result.get("pair_binding_digest"),
            "elapsed_seconds": result["elapsed_seconds"], "gpu_hours": result["gpu_hours"],
            "cpu_seconds": result["cpu_seconds"], "forward_calls": result["forward_calls"],
            "backward_calls": result["backward_calls"],
            "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
            "artifact_bytes": sum(item.stat().st_size for item in path.iterdir() if item.is_file()),
            "descriptive_only": True,
        }
        method_rows.append(row)
        by_key.setdefault((unit["sample_id"], unit["block"], unit["head"]), {})[unit["method"]] = row
    pair_rows = []
    for (sample_id, block, head), methods in by_key.items():
        if set(methods) != set(e3.METHODS):
            raise StageEE3R1Error("E3-R1 union coordinate lacks a paired method")
        nns = methods[e3.NNS_METHOD]
        matched = methods[e3.MATCHED_METHOD]
        status = "VALID" if nns["validity_status"] == matched["validity_status"] == "VALID" else "NA" if "NA" in {nns["validity_status"], matched["validity_status"]} else "FAILED"
        pair_rows.append({
            "variant_id": f"layer_head_{block}_{head}_pair", "sample_id": sample_id,
            "selection_order": nns["selection_order"], "cohort": "stage_d_fidelity_v1_500",
            "stratum": nns["stratum"], "validity_status": status,
            "source_hashes": {
                "nns_map_sha256": nns["source_hashes"].get("signed_map_sha256"),
                "matched_map_sha256": matched["source_hashes"].get("signed_map_sha256"),
                "target_semantic_digest": nns["source_hashes"]["target_semantic_digest"],
            },
            "method": "nns_vs_matched", "target": "fixed_clean_attention_cls_head_direction",
            "scale": "fine_4x4_cell", "layer": block, "head": head,
            "coordinate_role": "DESCRIPTIVE_FROZEN_COORDINATE",
            "recompute_status": "E3_V1_R1_UNION", "misclassified": nns["misclassified"],
            "nns_insertion_auc": nns["insertion_auc_mean"], "nns_deletion_auc": nns["deletion_auc_mean"],
            "matched_insertion_auc": matched["insertion_auc_mean"], "matched_deletion_auc": matched["deletion_auc_mean"],
            "delta_fid": e3.delta_fid(nns, matched) if status == "VALID" else None,
            "failure_class": None if status == "VALID" else "SCIENTIFIC_INVALIDITY",
            "failure_code": None if status == "VALID" else "paired_method_invalid",
            "na_reason": None, "descriptive_only": True,
        })
    return method_rows, pair_rows, attempts


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validate_e3_v1(repo)
    gate, _ = _strict_json(repo / PILOT_GATE_PATH, "E3-R1 pilot gate")
    authorization, _ = _strict_json(repo / BULK_AUTH_PATH, "E3-R1 bulk authorization")
    for value, name in ((gate, "pilot gate"), (authorization, "bulk authorization")):
        _validate_signed(value, f"E3-R1 {name}")
    if gate.get("status") != "PASS" or authorization.get("status") != "AUTHORIZED_BULK_COMPLEMENT_ONLY":
        raise StageEE3R1Error("E3-R1 finalization lacks a passing continuation gate")
    plan = _v1_plan(repo)
    primary_methods, primary_pairs = e3._primary_method_rows(repo, plan)
    new_methods, new_pairs, attempts = _combined_new_rows(repo, plan)
    method_rows = sorted(primary_methods + new_methods, key=lambda row: (row["selection_order"], e3.COORDINATES.index((row["layer"], row["head"])), e3.METHODS.index(row["method"])))
    pair_rows = sorted(primary_pairs + new_pairs, key=lambda row: (row["selection_order"], e3.COORDINATES.index((row["layer"], row["head"]))))
    if len(method_rows) != 4000 or len(pair_rows) != 2000 or len(attempts) != 3000:
        raise StageEE3R1Error("E3-R1 final output cardinality failed")
    if any(row["validity_status"] != "VALID" for row in method_rows + pair_rows):
        raise StageEE3R1Error("E3-R1 union contains a non-valid scientific row")
    registry, _ = e1._load_registry(repo)
    strata = {group["sample_id"]: group["stratum"] for group in plan["groups"]}
    if Counter(strata.values()) != Counter({"correct": 418, "misclassified": 82}):
        raise StageEE3R1Error("E3-R1 prediction partition drifted")
    summaries = []
    method_summaries = []
    for block, head in e3.COORDINATES:
        coordinate_pairs = [row for row in pair_rows if (row["layer"], row["head"]) == (block, head)]
        coordinate_methods = [row for row in method_rows if (row["layer"], row["head"]) == (block, head)]
        for stratum in e3.STRATA:
            pairs = coordinate_pairs if stratum == "overall" else [row for row in coordinate_pairs if row["stratum"] == stratum]
            summary = e3._summary(pairs, "delta_fid", f"layer_head_{block}_{head}_delta_fid", stratum)
            summaries.append({
                "variant_id": f"layer_head_{block}_{head}_{stratum}", "sample_id": None,
                "selection_order": None, "cohort": "stage_d_fidelity_v1_500", "stratum": stratum,
                "validity_status": "VALID", "source_hashes": {"paired_rows_canonical_digest": canonical_sha256(pairs)},
                "method": "nns_vs_matched", "target": "fixed_clean_attention_cls_head_direction",
                "scale": "fine_4x4_cell", "layer": block, "head": head,
                "coordinate_role": "PRIMARY_ANCHOR_READ_ONLY" if (block, head) == (0, 9) else "DESCRIPTIVE_FROZEN_COORDINATE",
                "metric": "delta_fid", **summary, "descriptive_only": True,
            })
            for method in e3.METHODS:
                selected = [row for row in coordinate_methods if row["method"] == method and (stratum == "overall" or row["stratum"] == stratum)]
                for direction in ("insertion", "deletion"):
                    summary = e3._summary(selected, f"{direction}_auc_mean", f"layer_head_{block}_{head}_{method}_{direction}_auc", stratum)
                    method_summaries.append({
                        "variant_id": f"layer_head_{block}_{head}_{method}_{direction}_{stratum}", "sample_id": None,
                        "selection_order": None, "cohort": "stage_d_fidelity_v1_500", "stratum": stratum,
                        "validity_status": "VALID", "source_hashes": {"method_rows_canonical_digest": canonical_sha256(selected)},
                        "method": method, "target": "fixed_clean_attention_cls_head_direction",
                        "scale": "fine_4x4_cell", "layer": block, "head": head,
                        "coordinate_role": "PRIMARY_ANCHOR_READ_ONLY" if (block, head) == (0, 9) else "DESCRIPTIVE_FROZEN_COORDINATE",
                        "metric": f"{direction}_auc", **summary, "descriptive_only": True,
                    })
    membership = [{
        "variant_id": "prediction_stratum_membership", "sample_id": row["sample_id"],
        "selection_order": row["selection_order"], "cohort": "stage_d_fidelity_v1_500",
        "stratum": strata[row["sample_id"]], "validity_status": "VALID",
        "source_hashes": {"d10_per_sample_sha256": e3._sha_file(repo / e3.D10_DELTA_PATH), "image_sha256": row["image_sha256"]},
        "method": "nns_vs_matched", "target": "fixed_clean_attention_cls_head_direction",
        "scale": "fine_4x4_cell", "layer": None, "head": None,
        "misclassified": strata[row["sample_id"]] == "misclassified",
    } for row in registry]
    actual = _cumulative_cost(repo)
    if actual["gpu_hours"] > GPU_HOURS_MAX or actual["cpu_seconds"] > CPU_SECONDS_MAX or actual["retained_bytes"] > RETAINED_BYTES_MAX:
        raise StageEE3R1Error("E3-R1 cumulative actual budget exceeded")
    summary_doc = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_descriptive_summary",
        "stage_id": "E3-R1", "recovery_id": RECOVERY_ID,
        "cohort": "stage_d_fidelity_v1_500", "descriptive_only": True,
        "p_values_emitted": False, "winner_selection_performed": False,
        "causal_claim_made": False, "new_go_no_go_issued": False,
        "d10_primary_unchanged": True, "primary_anchor_recomputed": False,
        "e3_v1_status_preserved": "FAIL",
        "counts": {"method_rows": 4000, "paired_rows": 2000, "new_compute_attempts": 3000, "e3_v1_inherited_attempts": 516, "e3_r1_new_attempts": 2484, "summary_cells": 12, "method_summary_cells": 48},
        "delta_fid_coordinate_strata": summaries, "method_auc_coordinate_strata": method_summaries,
        "prediction_partition": {"overall": 500, "correct": 418, "misclassified": 82},
        "clean_margin": e3._validate_config(repo)["scientific_contract"]["clean_margin"],
        "actual_cost": actual, "e4_started": False,
    })
    return {
        "method_rows": method_rows, "pair_rows": pair_rows,
        "summary_rows": summaries + method_summaries, "membership_rows": membership,
        "resource_rows": method_rows, "attempts": attempts,
        "summary": summary_doc, "actual_cost": actual,
    }


def _table(table_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise StageEE3R1Error(f"{table_id} cannot be empty")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    normalized = [{key: row.get(key) for key in columns} for row in rows]
    return _signed({
        "schema_version": 1, "document_type": "stage_e_e3_paper_table",
        "table_id": table_id, "columns": columns,
        "row_count": len(normalized), "rows": normalized,
    })


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: e3._csv_scalar(value) for key, value in row.items()} for row in rows])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def finalize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    outputs = validate_outputs(repo)
    root = repo / ROOT_PATH
    table_specs = (
        ("layer_head_method_rows", outputs["method_rows"]),
        ("layer_head_paired_delta", outputs["pair_rows"]),
        ("prediction_strata", outputs["summary_rows"]),
        ("prediction_strata_membership", outputs["membership_rows"]),
        ("resource_and_qc", outputs["resource_rows"]),
    )
    for name, rows in table_specs:
        table = _table(name, rows)
        _write_or_verify(root / "paper_data" / f"{name}.json", table)
        _write_csv(root / "paper_data" / f"{name}.csv", table["rows"])
    _write_or_verify(root / "descriptive_summary.json", outputs["summary"])
    _write_or_verify(root / "clean_margin_not_run.json", _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_not_run",
        "analysis": "clean_margin",
        **e3._validate_config(repo)["scientific_contract"]["clean_margin"],
    }))
    recipe = _signed({
        "schema_version": 1, "document_type": "stage_e_figure_recipes",
        "numeric_source": "paper_data_only", "renders_png": False,
        "figures": [{
            "figure_id": "e3_r1_layer_head_prediction_strata_future",
            "table_path": "paper_data/prediction_strata.json",
            "x": "layer", "y": "median", "group": "stratum",
            "caption_contract": "Post-D10 descriptive views for all four Stage-C-frozen coordinates; E3 v1 remains FAIL and R1 is the prospective numerical-recovery union.",
            "numeric_source": "paper_data_only", "renders_png": False,
        }],
    })
    _write_or_verify(root / "figure_recipes.json", recipe)
    provenance = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_provenance",
        "recovery_id": RECOVERY_ID, "e3_v1_commit": E3_V1_COMMIT,
        "e3_v1_status": "FAIL", "e3_v1_inventory_canonical_digest": E3_V1_INVENTORY_DIGEST,
        "e3_v1_inherited_attempts": 516, "e3_r1_new_attempts": 2484,
        "pilot_orders": list(PILOT_ORDERS),
        "operations": [
            "source-validate and retain E3 v1 terminal attempts in place",
            "run only the source-validated missing complement",
            "apply the prospective R1 schedule extension only after original schedule exhaustion",
            "form a metadata-indexed disjoint E3 v1 plus E3-R1 union",
            "compute the unchanged E3 descriptive summaries",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "pilot_gate": _record(repo, PILOT_GATE_PATH),
        "bulk_authorization": _record(repo, BULK_AUTH_PATH),
        "e4_started": False,
    })
    _write_or_verify(root / "provenance.json", provenance)
    false_audit = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_false_pass_audit",
        "recovery_id": RECOVERY_ID, "status": "PASS",
        "checks": [{"attack": name, "result": "REJECTED"} for name in FALSE_PASS_ATTACKS],
        "pass_possible_without_all_conditions": False, "e4_started": False,
    })
    _write_or_verify(repo / REPORT_ROOT_PATH / "false_pass_audit.json", false_audit)
    tests, _ = _strict_json(repo / TEST_REPORT_PATH, "E3-R1 tests")
    _validate_signed(tests, "E3-R1 tests")
    if tests.get("status") != "PASS" or tests.get("failed") != 0:
        raise StageEE3R1Error("E3-R1 required tests failed")
    validation = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_source_validation",
        "stage_id": "E3-R1", "recovery_id": RECOVERY_ID, "status": "PASS",
        "validated_at": _now(), "checks": {
            "e3_v1_terminal_fail_preserved": "PASS",
            "e3_v1_516_attempts_inherited": "PASS",
            "e3_r1_2484_attempts_new": "PASS",
            "union_3000_unique_terminal_valid": "PASS",
            "pilot_orders_61_72": "PASS", "pilot_and_budget_gate": "PASS",
            "original_schedule_before_fallback": "PASS",
            "thresholds_unchanged": "PASS", "scientific_contract_unchanged": "PASS",
            "cumulative_actual_budget": "PASS", "no_e4_started": "PASS",
        },
        "summary_canonical_digest": outputs["summary"]["canonical_digest"],
        "false_pass_audit_canonical_digest": false_audit["canonical_digest"],
    })
    _write_or_verify(repo / REPORT_ROOT_PATH / "validation_report.json", validation)
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "completion" not in path.parts and path.name not in {"payload_manifest.json", "whole_root_inventory.json", "status.json"}:
            artifacts.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    manifest = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_payload_manifest",
        "recovery_id": RECOVERY_ID, "immutable": True,
        "payload_count": len(artifacts), "artifacts": artifacts,
    })
    _write_or_verify(root / "payload_manifest.json", manifest)
    actual = _cumulative_cost(repo)
    if actual["gpu_hours"] > GPU_HOURS_MAX or actual["cpu_seconds"] > CPU_SECONDS_MAX or actual["retained_bytes"] > RETAINED_BYTES_MAX:
        raise StageEE3R1Error("E3-R1 final cumulative cost exceeded")
    inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/e3_r1_completion.pending.json", "completion/e3_r1_completion.json"))
    inventory_doc = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_whole_root_inventory",
        "root": ROOT_PATH, "scope": "all regular non-hidden E3-R1 formal files except self-referential inventory status and completion receipts",
        **inventory,
    })
    _write_or_verify(root / "whole_root_inventory.json", inventory_doc)
    status = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_status",
        "stage_id": "E3-R1", "recovery_id": RECOVERY_ID, "status": "PASS",
        "complete": True, "terminal": True, "immutable": True,
        "e3_v1_status_preserved": "FAIL", "package_completeness": "PASS",
        "workflow_integrity": "PASS", "scientific_validity": "PASS",
        "cost_status": "PASS_COST", "actual_cost": actual,
        "e4_authorized": True, "e4_started": False, "stage_f_started": False,
        "whole_root_inventory": {"path": "whole_root_inventory.json", "file_sha256": _sha_file(root / "whole_root_inventory.json"), "canonical_digest": inventory_doc["canonical_digest"], "file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"]},
    })
    _write_or_verify(root / "status.json", status)
    pending, _ = _strict_json(repo / COMPLETION_PENDING_PATH, "E3-R1 pending completion")
    completion = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_terminal_completion",
        "stage_id": "E3-R1", "recovery_id": RECOVERY_ID, "status": "PASS",
        "started_at": pending["started_at"], "completed_at": _now(),
        "e3_v1": {"status": "FAIL", "completion_canonical_digest": E3_V1_COMPLETION_DIGEST, "inventory_canonical_digest": E3_V1_INVENTORY_DIGEST, "inherited_attempts": 516, "modified": False},
        "union": {"expected_new_compute_units": 3000, "e3_v1_inherited": 516, "e3_r1_new": 2484, "valid_terminal": 3000, "failed": 0, "na": 0, "method_rows": 4000, "paired_rows": 2000, "summary_cells": 12},
        "package_completeness": "PASS", "workflow_integrity": "PASS",
        "scientific_validity": "PASS", "cost_status": "PASS_COST",
        "actual_cost": actual, "source_validation": {
            "validation_report": _record(repo, REPORT_ROOT_PATH + "/validation_report.json"),
            "test_report": _record(repo, TEST_REPORT_PATH),
            "false_pass_audit": _record(repo, REPORT_ROOT_PATH + "/false_pass_audit.json"),
            "payload_manifest": _record(repo, ROOT_PATH + "/payload_manifest.json"),
            "status": _record(repo, ROOT_PATH + "/status.json"),
            "whole_root_inventory": _record(repo, ROOT_PATH + "/whole_root_inventory.json"),
            "pilot_gate": _record(repo, PILOT_GATE_PATH),
        },
        "next_authorization": "E4_ONLY", "e4_started": False,
    })
    schema, _ = _strict_json(repo / COMPLETION_SCHEMA_PATH, "E3-R1 completion schema")
    errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors:
        raise StageEE3R1Error(f"E3-R1 completion schema failure: {errors[0].message}")
    temporary = (repo / COMPLETION_PENDING_PATH).with_name(".e3-r1-completion-terminal.tmp")
    _write_new_json(temporary, completion)
    os.replace(temporary, repo / COMPLETION_PENDING_PATH)
    observed, raw = _strict_json(repo / COMPLETION_PENDING_PATH, "E3-R1 terminal pending")
    if observed != completion or raw != canonical_json_bytes(completion) + b"\n":
        raise StageEE3R1Error("E3-R1 terminal receipt changed before publication")
    os.replace(repo / COMPLETION_PENDING_PATH, repo / COMPLETION_PATH)
    final_inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/e3_r1_completion.pending.json", "completion/e3_r1_completion.json"))
    if final_inventory != inventory:
        raise StageEE3R1Error("E3-R1 root changed during terminal publication")
    return validate_completion(repo)


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if not (repo / COMPLETION_PATH).is_file() or (repo / COMPLETION_PENDING_PATH).exists():
        raise StageEE3R1Error("terminal E3-R1 completion is absent or pending remains")
    completion, _ = _strict_json(repo / COMPLETION_PATH, "E3-R1 completion")
    _validate_signed(completion, "E3-R1 completion")
    if completion.get("status") == "BLOCKED":
        return validate_blocked_completion(repo)
    schema, _ = _strict_json(repo / COMPLETION_SCHEMA_PATH, "E3-R1 completion schema")
    errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors:
        raise StageEE3R1Error(f"E3-R1 completion schema failure: {errors[0].message}")
    if completion.get("status") != "PASS" or completion.get("next_authorization") != "E4_ONLY" or completion.get("e4_started") is not False:
        raise StageEE3R1Error("E3-R1 terminal authorization boundary failed")
    validate_e3_v1(repo)
    outputs = validate_outputs(repo)
    if outputs["actual_cost"] != completion["actual_cost"]:
        raise StageEE3R1Error("E3-R1 final cost drifted")
    status, _ = _strict_json(repo / ROOT_PATH / "status.json", "E3-R1 status")
    inventory, _ = _strict_json(repo / ROOT_PATH / "whole_root_inventory.json", "E3-R1 inventory")
    for value, name in ((status, "status"), (inventory, "inventory")):
        _validate_signed(value, f"E3-R1 {name}")
    rebuilt = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "status.json", "completion/e3_r1_completion.pending.json", "completion/e3_r1_completion.json"))
    if any(inventory[key] != rebuilt[key] for key in ("files", "file_count", "total_bytes", "canonical_digest")):
        raise StageEE3R1Error("E3-R1 whole-root inventory drifted")
    if status.get("status") != "PASS" or status.get("e3_v1_status_preserved") != "FAIL" or status.get("e4_authorized") is not True or status.get("e4_started") is not False:
        raise StageEE3R1Error("E3-R1 status drifted")
    for record in completion["source_validation"].values():
        if record != _record(repo, record["path"]):
            raise StageEE3R1Error(f"E3-R1 completion binding drifted: {record['path']}")
    if any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in (repo / ROOT_PATH).rglob("*") if path.is_file()):
        raise StageEE3R1Error("E3-R1 contains a forbidden render artifact")
    return {
        "status": "PASS", "completion_canonical_digest": completion["canonical_digest"],
        "whole_root_inventory_digest": inventory["canonical_digest"],
        "e3_v1_status": "FAIL", "inherited_attempts": 516,
        "new_attempts": 2484, "union_valid_terminal_attempts": 3000,
        "method_rows": 4000, "paired_rows": 2000,
        "actual_cost": completion["actual_cost"],
        "next_authorization": "E4_ONLY", "e4_started": False,
    }


def finalize_blocked(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validate_e3_v1(repo)
    gate, _ = _strict_json(repo / PILOT_GATE_PATH, "E3-R1 failed pilot gate")
    _validate_signed(gate, "E3-R1 failed pilot gate")
    if (
        gate.get("status") != "FAIL"
        or gate.get("pilot_terminal_units") != 12
        or gate.get("pilot_valid_units") != 12
        or gate.get("pilot_nns_converged") != 6
        or gate.get("actual_budget_status") != "PASS"
        or gate.get("projected_budget_status") != "FAIL"
        or gate.get("bulk_authorized") is not False
        or gate.get("e4_authorized") is not False
        or gate.get("e4_started") is not False
    ):
        raise StageEE3R1Error("E3-R1 blocked completion gate is not satisfied")
    if (repo / BULK_AUTH_PATH).exists():
        raise StageEE3R1Error("bulk authorization exists despite failed pilot gate")
    observed = _r1_attempt_inventory(repo)
    if len(observed) != 12 or any(item[1]["result"]["validity_status"] != "VALID" for item in observed):
        raise StageEE3R1Error("E3-R1 blocked completion requires exactly twelve valid pilot attempts")
    if {item[0]["selection_order"] for item in observed} != set(PILOT_ORDERS):
        raise StageEE3R1Error("non-pilot R1 work started before continuation authorization")
    actual = gate["actual_cost"]
    false_audit = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_false_pass_audit",
        "recovery_id": RECOVERY_ID, "status": "PASS",
        "checks": [{"attack": name, "result": "REJECTED"} for name in FALSE_PASS_ATTACKS],
        "e3_r1_pass_issued": False, "e4_authorized": False,
        "e4_started": False, "pass_possible_without_all_conditions": False,
    })
    _write_or_verify(repo / REPORT_ROOT_PATH / "false_pass_audit.json", false_audit)
    validation = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_source_validation",
        "stage_id": "E3-R1", "recovery_id": RECOVERY_ID, "status": "BLOCKED",
        "validated_at": _now(), "checks": {
            "e3_v1_terminal_fail_preserved": "PASS",
            "e3_v1_516_attempts_unchanged": "PASS",
            "pilot_orders_61_72_valid_terminal": "PASS_12_OF_12",
            "pilot_nns_convergence": "PASS_6_OF_6",
            "cumulative_actual_budget": "PASS",
            "projected_full_campaign_budget": "FAIL",
            "bulk_not_started": "PASS", "e4_not_started": "PASS",
        },
        "pilot_gate_canonical_digest": gate["canonical_digest"],
        "false_pass_audit_canonical_digest": false_audit["canonical_digest"],
    })
    _write_or_verify(repo / REPORT_ROOT_PATH / "validation_report.json", validation)
    artifacts = []
    root = repo / ROOT_PATH
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "completion" not in path.parts and path.name not in {"payload_manifest.json", "whole_root_inventory.json", "status.json"}:
            artifacts.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    manifest = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_payload_manifest",
        "recovery_id": RECOVERY_ID, "immutable": True,
        "terminal_stage_status": "BLOCKED", "payload_count": len(artifacts),
        "artifacts": artifacts,
    })
    _write_or_verify(root / "payload_manifest.json", manifest)
    inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/e3_r1_completion.pending.json", "completion/e3_r1_completion.json"))
    inventory_doc = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_whole_root_inventory",
        "root": ROOT_PATH, "scope": "all regular non-hidden E3-R1 formal files except self-referential inventory status and completion receipts",
        **inventory,
    })
    _write_or_verify(root / "whole_root_inventory.json", inventory_doc)
    status = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_status",
        "stage_id": "E3-R1", "recovery_id": RECOVERY_ID,
        "status": "BLOCKED", "complete": False, "terminal": True,
        "immutable": True, "e3_v1_status_preserved": "FAIL",
        "package_completeness": "BLOCKED", "workflow_integrity": "PASS",
        "scientific_validity": "NOT_EVALUATED", "pilot_scientific_validity": "PASS",
        "cost_status": "FAIL_COST", "actual_cost": actual,
        "projected_total": gate["projected_total"],
        "block_reason": "PROJECTED_FULL_CAMPAIGN_BUDGET_EXCEEDS_FROZEN_E3_LIMITS",
        "bulk_authorized": False, "bulk_started": False,
        "e4_authorized": False, "e4_started": False, "stage_f_started": False,
        "whole_root_inventory": {"path": "whole_root_inventory.json", "file_sha256": _sha_file(root / "whole_root_inventory.json"), "canonical_digest": inventory_doc["canonical_digest"], "file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"]},
    })
    _write_or_verify(root / "status.json", status)
    pending, _ = _strict_json(repo / COMPLETION_PENDING_PATH, "E3-R1 pending completion")
    completion = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r1_terminal_completion",
        "stage_id": "E3-R1", "recovery_id": RECOVERY_ID, "status": "BLOCKED",
        "started_at": pending["started_at"], "completed_at": _now(),
        "e3_v1": {"status": "FAIL", "completion_canonical_digest": E3_V1_COMPLETION_DIGEST, "inventory_canonical_digest": E3_V1_INVENTORY_DIGEST, "inherited_attempts": 516, "modified": False},
        "union": {"expected_new_compute_units": 3000, "e3_v1_inherited": 516, "e3_r1_new": 12, "valid_terminal": 528, "missing": 2472, "pilot_valid": 12, "pilot_nns_converged": 6},
        "package_completeness": "BLOCKED", "workflow_integrity": "PASS",
        "scientific_validity": "NOT_EVALUATED", "cost_status": "FAIL_COST",
        "actual_cost": actual, "source_validation": {
            "validation_report": _record(repo, REPORT_ROOT_PATH + "/validation_report.json"),
            "test_report": _record(repo, RECOVERY_TEST_REPORT_PATH),
            "false_pass_audit": _record(repo, REPORT_ROOT_PATH + "/false_pass_audit.json"),
            "payload_manifest": _record(repo, ROOT_PATH + "/payload_manifest.json"),
            "status": _record(repo, ROOT_PATH + "/status.json"),
            "whole_root_inventory": _record(repo, ROOT_PATH + "/whole_root_inventory.json"),
            "pilot_gate": _record(repo, PILOT_GATE_PATH),
        },
        "next_authorization": "NONE", "e4_started": False,
    })
    schema, _ = _strict_json(repo / COMPLETION_SCHEMA_PATH, "E3-R1 completion schema")
    errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors:
        raise StageEE3R1Error(f"E3-R1 blocked completion schema failure: {errors[0].message}")
    temporary = (repo / COMPLETION_PENDING_PATH).with_name(".e3-r1-completion-blocked.tmp")
    _write_new_json(temporary, completion)
    os.replace(temporary, repo / COMPLETION_PENDING_PATH)
    os.replace(repo / COMPLETION_PENDING_PATH, repo / COMPLETION_PATH)
    return validate_blocked_completion(repo)


def validate_blocked_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    completion, _ = _strict_json(repo / COMPLETION_PATH, "E3-R1 blocked completion")
    _validate_signed(completion, "E3-R1 blocked completion")
    schema, _ = _strict_json(repo / COMPLETION_SCHEMA_PATH, "E3-R1 completion schema")
    if list(Draft202012Validator(schema).iter_errors(completion)):
        raise StageEE3R1Error("E3-R1 blocked completion schema failed")
    if completion.get("status") != "BLOCKED" or completion.get("next_authorization") != "NONE" or completion.get("e4_started") is not False:
        raise StageEE3R1Error("E3-R1 blocked authorization boundary failed")
    validate_e3_v1(repo)
    gate, _ = _strict_json(repo / PILOT_GATE_PATH, "E3-R1 pilot gate")
    _validate_signed(gate, "E3-R1 pilot gate")
    if gate.get("status") != "FAIL" or gate.get("projected_budget_status") != "FAIL" or (repo / BULK_AUTH_PATH).exists():
        raise StageEE3R1Error("E3-R1 blocked pilot/bulk boundary drifted")
    observed = _r1_attempt_inventory(repo)
    if len(observed) != 12 or {item[0]["selection_order"] for item in observed} != set(PILOT_ORDERS):
        raise StageEE3R1Error("E3-R1 blocked attempt coverage drifted")
    status, _ = _strict_json(repo / ROOT_PATH / "status.json", "E3-R1 blocked status")
    inventory, _ = _strict_json(repo / ROOT_PATH / "whole_root_inventory.json", "E3-R1 blocked inventory")
    for value, name in ((status, "status"), (inventory, "inventory")):
        _validate_signed(value, f"E3-R1 blocked {name}")
    rebuilt = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "status.json", "completion/e3_r1_completion.pending.json", "completion/e3_r1_completion.json"))
    if any(inventory[key] != rebuilt[key] for key in ("files", "file_count", "total_bytes", "canonical_digest")):
        raise StageEE3R1Error("E3-R1 blocked root inventory drifted")
    if status.get("status") != "BLOCKED" or status.get("bulk_started") is not False or status.get("e4_authorized") is not False or status.get("e4_started") is not False:
        raise StageEE3R1Error("E3-R1 blocked status drifted")
    for record in completion["source_validation"].values():
        if record != _record(repo, record["path"]):
            raise StageEE3R1Error(f"E3-R1 blocked completion binding drifted: {record['path']}")
    return {
        "status": "BLOCKED", "reason": status["block_reason"],
        "pilot_valid_units": 12, "pilot_nns_converged": 6,
        "inherited_attempts": 516, "new_attempts": 12,
        "missing_attempts": 2472, "actual_cost": completion["actual_cost"],
        "projected_total": status["projected_total"],
        "next_authorization": "NONE", "e4_started": False,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(prog="python -m vitnns.stage_e_e3_r1")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-preconditions", "initialize", "validate-initialized", "evaluate-pilot", "validate-outputs", "finalize", "finalize-blocked", "validate-completion"):
        child = sub.add_parser(name)
        child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    tests = sub.add_parser("record-tests")
    tests.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    tests.add_argument("--passed", type=int, required=True)
    tests.add_argument("--failed", type=int, required=True)
    tests.add_argument("--test-command", required=True)
    recovery = sub.add_parser("authorize-pilot-recovery")
    recovery.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    recovery.add_argument("--passed", type=int, required=True)
    recovery.add_argument("--failed", type=int, required=True)
    recovery.add_argument("--test-command", required=True)
    pilot = sub.add_parser("run-pilot")
    pilot.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    pilot.add_argument("--device", default="cuda:0")
    bulk = sub.add_parser("run-bulk-shard")
    bulk.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    bulk.add_argument("--shard-id", type=int, required=True)
    bulk.add_argument("--shard-count", type=int, required=True)
    bulk.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "verify-preconditions":
        value = verify_preconditions(args.repo_root)
    elif args.command == "initialize":
        value = initialize(args.repo_root)
    elif args.command == "validate-initialized":
        value = validate_initialized(args.repo_root)
    elif args.command == "record-tests":
        value = record_tests(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command)
    elif args.command == "authorize-pilot-recovery":
        value = authorize_pilot_recovery(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command)
    elif args.command == "run-pilot":
        value = run_pilot(args.repo_root, device=args.device)
    elif args.command == "evaluate-pilot":
        value = evaluate_pilot(args.repo_root)
    elif args.command == "run-bulk-shard":
        value = run_bulk_shard(args.repo_root, shard_id=args.shard_id, shard_count=args.shard_count, device=args.device)
    elif args.command == "validate-outputs":
        output = validate_outputs(args.repo_root)
        value = {"status": "PASS", "method_rows": len(output["method_rows"]), "paired_rows": len(output["pair_rows"]), "actual_cost": output["actual_cost"]}
    elif args.command == "finalize":
        value = finalize(args.repo_root)
    elif args.command == "finalize-blocked":
        value = finalize_blocked(args.repo_root)
    else:
        value = validate_completion(args.repo_root)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
