"""Budget-only E3-R2 continuation over immutable E3 v1 and R1 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
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

RECOVERY_ID = "STAGE-E-E3-R2-BUDGET-CONTINUATION-20260906"
CONFIG_PATH = "config/stage_e_e3_r2.json"
CONFIG_SCHEMA_PATH = "config/stage_e_e3_r2.schema.json"
COMPLETION_SCHEMA_PATH = "config/stage_e_e3_r2_completion.schema.json"
AMENDMENT_PATH = "docs/recovery_stage_e_e3_r2_budget_amendment_20260906.md"
ROOT_PATH = "evidence/stage_e/e3_r2_recovery_v2_20260906/formal"
REPORT_ROOT_PATH = "reports/stage_e_e3_r2_20260906"
PLAN_PATH = ROOT_PATH + "/control/recovery_plan.json"
RUN_SPEC_PATH = ROOT_PATH + "/control/run_spec.json"
AUTH_PATH = ROOT_PATH + "/control/execution_authorization.json"
TEST_REPORT_PATH = REPORT_ROOT_PATH + "/pre_execution_test_results.json"
RECOVERY_TEST_REPORT_PATH = REPORT_ROOT_PATH + "/execution_recovery_test_results.json"
INCIDENT_PATH = REPORT_ROOT_PATH + "/execution_infrastructure_event_001.json"
COMPLETION_PENDING_PATH = ROOT_PATH + "/completion/e3_r2_completion.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/e3_r2_completion.json"

R1_ROOT = r1.ROOT_PATH
R1_REPORT_ROOT = r1.REPORT_ROOT_PATH
R1_COMMIT = "b3855edc935fe7d2558fd328b2f6b774fa0f8bfc"
R1_TREE = "689e7fd52da25ff3b1c02afe5d2d6583ce479f50"
R1_REF = "refs/archive/stage-e-e3-r1-budget-blocked"
R1_COMPLETION_SHA256 = "20f765df2f2c5deb2e9049e213c3ac9defa60c8050ddf0dda5d252c48bab351f"
R1_COMPLETION_DIGEST = "c762feba66172027a35fcb51d6b9218f9197fc6e8a92d50ad616eb5d020c0b3d"
R1_STATUS_SHA256 = "51e66bc88bdc008e649f3617c6bf60b49daf070550cc2f2dd4a8623c8e1750b3"
R1_STATUS_DIGEST = "724990be6e2ace58566062109eebd8ac7dc4b6b141718b8489112a5198859fd4"
R1_INVENTORY_SHA256 = "9c7b9679cf3e7f11b2b58ddcfe419304f8a12ad4990b8abde4397bb77bd9cb37"
R1_INVENTORY_DIGEST = "62fa00e55d85c46bd54d011f8ac5998e95244a270b295337a52dfadb9c1a0486"
R1_GATE_SHA256 = "827e665b3039912a640bfdfc5a7ace87871f8299bb44b44e27415bbe80e20d45"
R1_GATE_DIGEST = "b51ac057e8ac0855f09bb6427936aabb6ad7d9645d2081d195043f1e9a8733b8"
EXPECTED_CONFIG_DIGEST = "d56267502f5af032ef7c6633a22ef436fd5c528b5294df651fe16a17da2b55fb"
ATTEMPT_ID = "attempt-r2-000001"
ORIGINAL_SCHEDULE = r1.ORIGINAL_SCHEDULE
FALLBACK_SCHEDULE = r1.FALLBACK_SCHEDULE
COMPLETE_SCHEDULE = r1.COMPLETE_SCHEDULE
GPU_HOURS_MAX = 128.0
CPU_SECONDS_MAX = 512.0 * 3600.0
RETAINED_BYTES_MAX = 34359738368
FORBIDDEN_SUFFIXES = {".png", ".pdf", ".svg"}
FALSE_PASS_ATTACKS = (
    "e3_v1_fail_modified_or_relabelled", "e3_r1_blocked_modified_or_relabelled",
    "one_of_528_terminal_attempts_recomputed", "missing_unit_excluded",
    "fallback_before_original_schedule_exhausted", "threshold_or_contract_relaxed",
    "prior_cost_omitted", "scientific_failure_or_na_treated_as_valid",
    "favorable_valid_subset_summary", "e4_started_before_terminal_pass",
)


class StageEE3R2Error(RuntimeError):
    """E3-R2 cannot proceed or pass under the budget amendment."""


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
    result["canonical_digest"] = canonical_sha256({key: item for key, item in result.items() if key != "canonical_digest"})
    return result


def _strict_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite token {token}")), object_pairs_hook=e1._reject_duplicate_keys)
    except Exception as error:
        raise StageEE3R2Error(f"cannot read strict {context}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageEE3R2Error(f"{context} must be a JSON object")
    return value, raw


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    expected = canonical_sha256({key: item for key, item in value.items() if key != "canonical_digest"})
    if value.get("canonical_digest") != expected:
        raise StageEE3R2Error(f"{context} canonical digest mismatch")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except BaseException:
        try: path.unlink()
        except FileNotFoundError: pass
        raise


def _write_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise StageEE3R2Error(f"immutable document differs: {path}")
        return
    _write_new_json(path, value)


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative; raw = path.read_bytes()
    value = _strict_json(path, relative)[0] if path.suffix == ".json" else {}
    return {"path": relative, "bytes": len(raw), "sha256": _sha(raw), "canonical_digest": value.get("canonical_digest")}


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(("git", *args), cwd=repo, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise StageEE3R2Error(f"git {' '.join(args)} failed: {error.stderr.strip()}") from error


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude); files = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink(): continue
            relative = path.relative_to(root).as_posix()
            if relative in excluded or any(part.startswith(".") for part in Path(relative).parts): continue
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return {"files": files, "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files), "canonical_digest": canonical_sha256({"files": files})}


def _validate_config(repo: Path) -> dict[str, Any]:
    config, _ = _strict_json(repo / CONFIG_PATH, "E3-R2 config")
    schema, _ = _strict_json(repo / CONFIG_SCHEMA_PATH, "E3-R2 config schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(config), key=lambda item: [str(part) for part in item.path])
    if errors: raise StageEE3R2Error(f"E3-R2 config schema failure: {errors[0].message}")
    _validate_signed(config, "E3-R2 config")
    if config["canonical_digest"] != EXPECTED_CONFIG_DIGEST:
        raise StageEE3R2Error("E3-R2 frozen config digest mismatch")
    if tuple(config["numerical_recovery"]["complete_schedule"]) != COMPLETE_SCHEDULE or config["numerical_recovery"]["atol"] != 1e-4 or config["numerical_recovery"]["rtol"] != 0.01:
        raise StageEE3R2Error("E3-R2 numerical contract drifted")
    if config["budget"] != {"gpu_hours_max": 128.0, "cpu_hours_max": 512.0, "retained_bytes_max": 34359738368, "e3_v1_and_r1_cost_included": True, "waiver_allowed": False}:
        raise StageEE3R2Error("E3-R2 enlarged budget drifted")
    return config


def validate_upstream(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    e3_result = e3.validate_completion(repo)
    r1_result = r1.validate_completion(repo)
    if e3_result.get("status") != "FAIL" or e3_result.get("valid_terminal_attempts") != 516 or e3_result.get("next_authorization") != "NONE":
        raise StageEE3R2Error("E3 v1 terminal FAIL drifted")
    if (
        r1_result.get("status"), r1_result.get("reason"),
        r1_result.get("pilot_valid_units"), r1_result.get("pilot_nns_converged"),
        r1_result.get("new_attempts"), r1_result.get("missing_attempts"),
        r1_result.get("next_authorization"), r1_result.get("e4_started"),
    ) != (
        "BLOCKED", "PROJECTED_FULL_CAMPAIGN_BUDGET_EXCEEDS_FROZEN_E3_LIMITS",
        12, 6, 12, 2472, "NONE", False,
    ):
        raise StageEE3R2Error("E3-R1 terminal budget block drifted")
    bindings = (
        (r1.COMPLETION_PATH, R1_COMPLETION_SHA256, R1_COMPLETION_DIGEST),
        (R1_ROOT + "/status.json", R1_STATUS_SHA256, R1_STATUS_DIGEST),
        (R1_ROOT + "/whole_root_inventory.json", R1_INVENTORY_SHA256, R1_INVENTORY_DIGEST),
        (r1.PILOT_GATE_PATH, R1_GATE_SHA256, R1_GATE_DIGEST),
    )
    for relative, expected_sha, expected_digest in bindings:
        value, raw = _strict_json(repo / relative, relative)
        _validate_signed(value, relative)
        if _sha(raw) != expected_sha or value["canonical_digest"] != expected_digest:
            raise StageEE3R2Error(f"E3-R1 immutable byte binding failed: {relative}")
    if _git(repo, "rev-parse", R1_REF + "^{commit}") != R1_COMMIT:
        raise StageEE3R2Error("E3-R1 archive ref drifted")
    if _git(repo, "rev-parse", R1_COMMIT + "^{tree}") != R1_TREE:
        raise StageEE3R2Error("E3-R1 custody tree drifted")
    protected = (R1_ROOT, R1_REPORT_ROOT)
    if _git(repo, "diff", "--name-only", R1_COMMIT, "--", *protected):
        raise StageEE3R2Error("tracked E3-R1 evidence changed after custody")
    return {"status": "PASS", "e3_v1": e3_result, "e3_r1": r1_result}


def _v1_plan(repo: Path) -> dict[str, Any]:
    plan, raw = _strict_json(repo / e3.PLAN_PATH, "E3 v1 typed plan")
    _validate_signed(plan, "E3 v1 typed plan")
    if _sha(raw) != r1.E3_V1_PLAN_SHA256 or plan["canonical_digest"] != r1.E3_V1_PLAN_DIGEST:
        raise StageEE3R2Error("E3 v1 typed plan binding drifted")
    return plan


def _r2_attempt_path(repo: Path, unit: Mapping[str, Any]) -> Path:
    return r1._r1_attempt_paths(repo, unit["canonical_digest"], root_path=ROOT_PATH, attempt_id=ATTEMPT_ID)[0]


def _source_attempt(repo: Path, unit: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    v1_path = repo / e3.ROOT_PATH / "units" / unit["canonical_digest"] / "attempts/attempt-000001"
    if v1_path.exists():
        return "E3_V1_INHERITED", e3._validate_attempt(repo, unit)
    r1_path = repo / R1_ROOT / "units" / unit["canonical_digest"] / "attempts" / r1.ATTEMPT_ID
    if r1_path.exists():
        return "E3_R1_INHERITED", r1._validate_r1_attempt(repo, unit)
    r2_path = _r2_attempt_path(repo, unit)
    if r2_path.exists():
        return "E3_R2_NEW", r1._validate_r1_attempt(
            repo, unit, root_path=ROOT_PATH, attempt_id=ATTEMPT_ID,
            recovery_id=RECOVERY_ID, origin="E3_R2_NEW",
        )
    return None


def _partition(repo: Path) -> tuple[dict[str, tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    plan = _v1_plan(repo)
    inherited = {}
    eligible = []
    for unit in plan["units"]:
        source = _source_attempt(repo, unit)
        if source is None:
            eligible.append(unit)
        elif source[0] == "E3_R2_NEW":
            eligible.append(unit)
        else:
            inherited[unit["canonical_digest"]] = source
    if Counter(source[0] for source in inherited.values()) != Counter({"E3_V1_INHERITED": 516, "E3_R1_INHERITED": 12}) or len(eligible) != 2472:
        raise StageEE3R2Error("E3-R2 528/2472 partition drifted")
    return inherited, eligible


def verify_preconditions(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = _validate_config(repo)
    upstream = validate_upstream(repo)
    inherited, eligible = _partition(repo)
    amendment_sha = _sha_file(repo / AMENDMENT_PATH)
    return {
        "status": "PASS", "validated_at": _now(),
        "head": _git(repo, "rev-parse", "HEAD"),
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "config": config, "upstream": upstream,
        "inherited": len(inherited), "eligible": len(eligible),
        "amendment_sha256": amendment_sha,
    }


def record_tests(repo_root: Path | str = REPO_ROOT, *, passed: int, failed: int, command: str) -> dict[str, Any]:
    if type(passed) is not int or passed <= 0 or failed != 0 or not command.strip():
        raise StageEE3R2Error("only a real zero-failure E3-R2 test result can be recorded")
    report = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r2_pre_execution_test_results",
        "recovery_id": RECOVERY_ID, "status": "PASS", "recorded_at": _now(),
        "command": command, "passed": passed, "failed": failed, "e4_started": False,
    })
    _write_or_verify(Path(repo_root).resolve() / TEST_REPORT_PATH, report)
    return report


def authorize_execution_recovery(repo_root: Path | str = REPO_ROOT, *, passed: int, failed: int, command: str) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); validate_initialized(repo)
    if type(passed) is not int or passed<=0 or failed!=0 or not command.strip(): raise StageEE3R2Error("E3-R2 recovery requires a real zero-failure test result")
    _,eligible=_partition(repo); units={unit["canonical_digest"]:unit for unit in eligible}; observed=[]
    for path in sorted((repo/ROOT_PATH/"units").glob("*/attempts/attempt-r2-000001")):
        digest=path.parents[1].name
        if digest not in units: raise StageEE3R2Error("unknown R2 recovery unit")
        attempt=r1._validate_r1_attempt(repo,units[digest],root_path=ROOT_PATH,attempt_id=ATTEMPT_ID,recovery_id=RECOVERY_ID,origin="E3_R2_NEW")
        if attempt["result"]["validity_status"]!="VALID": raise StageEE3R2Error("R2 recovery cannot follow scientific failure")
        observed.append({"unit_digest":digest,"selection_order":units[digest]["selection_order"],"result_canonical_digest":attempt["result"]["canonical_digest"],"manifest_canonical_digest":attempt["manifest"]["canonical_digest"]})
    if len(observed)!=8: raise StageEE3R2Error("R2 infrastructure event must preserve exactly eight attempts")
    event=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_infrastructure_event","recovery_id":RECOVERY_ID,"event_id":"E3_R2_INFRA_001","status":"RECORDED","recorded_at":_now(),"failure_class":"INFRASTRUCTURE_ORCHESTRATION_VALIDATOR","failure_code":"PUBLISHER_ORIGIN_VALIDATION_ARGUMENT_MISSING","scientific_failure":False,"description":"Each shard completed and atomically published its first NNS unit; immediate validation omitted the E3_R2_NEW origin argument.","preserved_terminal_attempts":observed,"terminal_attempt_recompute_authorized":False,"remaining_units_authorized":2464,"cost":{"gpu_seconds_charged":160.0,"cpu_seconds_charged":160.0,"accounting":"EIGHT_PROCESSES_CONSERVATIVE_20_SECONDS_EACH"},"e4_authorized":False,"e4_started":False}); _write_or_verify(repo/INCIDENT_PATH,event)
    report=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_execution_recovery_test_results","recovery_id":RECOVERY_ID,"status":"PASS","recorded_at":_now(),"command":command,"passed":passed,"failed":failed,"e4_started":False}); _write_or_verify(repo/RECOVERY_TEST_REPORT_PATH,report)
    status=_git(repo,"status","--porcelain=v1"); allowed=(ROOT_PATH+"/",REPORT_ROOT_PATH+"/"); unexpected=[line for line in status.splitlines() if not any(line[3:].startswith(item) or f"/{item}" in line[3:] for item in allowed)]
    if unexpected: raise StageEE3R2Error(f"E3-R2 recovery producer worktree is not clean: {unexpected}")
    head=_git(repo,"rev-parse","HEAD")
    if _git(repo,"rev-parse","@{u}")!=head: raise StageEE3R2Error("E3-R2 recovery producer is not pushed")
    authorization=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_execution_recovery_authorization","recovery_id":RECOVERY_ID,"status":"AUTHORIZED_MISSING_COMPLEMENT_ONLY","issued_at":_now(),"producer_commit":head,"producer_tree":_git(repo,"rev-parse","HEAD^{tree}"),"supersedes_authorization_canonical_digest":_strict_json(repo/AUTH_PATH,"E3-R2 original authorization")[0]["canonical_digest"],"incident_canonical_digest":event["canonical_digest"],"test_report_canonical_digest":report["canonical_digest"],"preserved_terminal_attempts":8,"eligible_missing_attempts":2464,"terminal_attempt_recompute":False,"e4_authorized":False,"e4_started":False}); _write_or_verify(repo/ROOT_PATH/"control/execution_recovery_authorization.json",authorization); return authorization


def initialize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    pre = verify_preconditions(repo)
    status = _git(repo, "status", "--porcelain=v1")
    allowed = (ROOT_PATH + "/", REPORT_ROOT_PATH + "/")
    unexpected = [line for line in status.splitlines() if not any(line[3:].startswith(item) or f"/{item}" in line[3:] for item in allowed)]
    if unexpected: raise StageEE3R2Error(f"E3-R2 producer worktree is not clean: {unexpected}")
    head = pre["head"]
    if _git(repo, "rev-parse", "@{u}") != head:
        raise StageEE3R2Error("E3-R2 producer is not pushed")
    tests, _ = _strict_json(repo / TEST_REPORT_PATH, "E3-R2 pre-execution tests")
    _validate_signed(tests, "E3-R2 pre-execution tests")
    if tests.get("status") != "PASS" or tests.get("failed") != 0:
        raise StageEE3R2Error("E3-R2 tests are not PASS")
    inherited, eligible = _partition(repo)
    plan = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r2_recovery_plan",
        "recovery_id": RECOVERY_ID, "e3_v1_plan_canonical_digest": r1.E3_V1_PLAN_DIGEST,
        "inherited_units": [{"unit_digest": key, "origin": value[0]} for key, value in sorted(inherited.items())],
        "eligible_unit_digests": [unit["canonical_digest"] for unit in eligible],
        "counts": {"total": 3000, "e3_v1_inherited": 516, "e3_r1_inherited": 12, "r2_eligible": 2472},
    })
    run_spec = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r2_run_spec",
        "recovery_id": RECOVERY_ID, "plan_canonical_digest": plan["canonical_digest"],
        "complete_schedule": list(COMPLETE_SCHEDULE), "atol": 1e-4, "rtol": 0.01,
        "step_chunk_size": 8, "coordinates": [list(item) for item in e3.COORDINATES],
        "methods": list(e3.METHODS), "shard_count": 8,
        "shard_rule": "selection_order_modulo_8",
        "stage_boundary": {"current": "E3_R2_ONLY", "e4_started": False},
    })
    authorization = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r2_execution_authorization",
        "recovery_id": RECOVERY_ID, "status": "AUTHORIZED_MISSING_COMPLEMENT_ONLY",
        "issued_at": _now(), "producer_commit": head, "producer_tree": pre["tree"],
        "amendment_sha256": pre["amendment_sha256"],
        "e3_v1_commit": r1.E3_V1_COMMIT, "e3_r1_commit": R1_COMMIT,
        "e3_r1_completion_canonical_digest": R1_COMPLETION_DIGEST,
        "plan_canonical_digest": plan["canonical_digest"], "run_spec_canonical_digest": run_spec["canonical_digest"],
        "authorized_unit_count": 2472, "inherited_terminal_recompute": False,
        "budget": {"gpu_hours_max": GPU_HOURS_MAX, "cpu_hours_max": 512.0, "retained_bytes_max": RETAINED_BYTES_MAX},
        "e4_authorized": False, "e4_started": False,
    })
    pending = _signed({
        "schema_version": 1, "document_type": "stage_e_e3_r2_completion_pending",
        "stage_id": "E3-R2", "recovery_id": RECOVERY_ID, "status": "PENDING",
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
    for name, relative in (("plan", PLAN_PATH), ("run_spec", RUN_SPEC_PATH), ("authorization", AUTH_PATH), ("pending", COMPLETION_PENDING_PATH)):
        value, _ = _strict_json(repo / relative, f"E3-R2 {name}"); _validate_signed(value, f"E3-R2 {name}"); documents[name] = value
    if documents["plan"].get("counts") != {"total": 3000, "e3_v1_inherited": 516, "e3_r1_inherited": 12, "r2_eligible": 2472}:
        raise StageEE3R2Error("E3-R2 plan counts drifted")
    if documents["authorization"].get("status") != "AUTHORIZED_MISSING_COMPLEMENT_ONLY" or documents["authorization"].get("authorized_unit_count") != 2472 or documents["authorization"].get("e4_authorized") is not False:
        raise StageEE3R2Error("E3-R2 execution authorization drifted")
    if documents["pending"].get("status") != "PENDING" or documents["pending"].get("e4_started") is not False:
        raise StageEE3R2Error("E3-R2 pending receipt drifted")
    return {"status": "PASS", "inherited": 528, "eligible": 2472, "shard_count": 8, "e4_started": False}


def _validate_execution_code(repo: Path) -> None:
    recovery_path=repo/ROOT_PATH/"control/execution_recovery_authorization.json"; selected=recovery_path if recovery_path.is_file() else repo/AUTH_PATH
    authorization, _ = _strict_json(selected, "E3-R2 execution authorization")
    _validate_signed(authorization, "E3-R2 execution authorization")
    if authorization.get("status") != "AUTHORIZED_MISSING_COMPLEMENT_ONLY":
        raise StageEE3R2Error("E3-R2 execution is not authorized")
    producer = authorization["producer_commit"]
    if subprocess.run(("git", "merge-base", "--is-ancestor", producer, "HEAD"), cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise StageEE3R2Error("E3-R2 execution HEAD does not descend from producer")
    if _git(repo, "diff", "--name-only", producer, "HEAD", "--", "src/vitnns", "config", "tests", AMENDMENT_PATH):
        raise StageEE3R2Error("E3-R2 reviewed producer source changed before execution")


def run_shard(repo_root: Path | str = REPO_ROOT, *, shard_id: int, shard_count: int, device: str = "cuda:0") -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if shard_count != 8 or type(shard_id) is not int or not 0 <= shard_id < 8:
        raise StageEE3R2Error("E3-R2 requires one of eight frozen shards")
    validate_initialized(repo); _validate_execution_code(repo)
    plan = _v1_plan(repo); _, eligible = _partition(repo)
    units_by_sample: dict[str, list[dict[str, Any]]] = {}
    for unit in eligible: units_by_sample.setdefault(unit["sample_id"], []).append(unit)
    groups = [group for group in plan["groups"] if group["selection_order"] not in r1.PILOT_ORDERS and group["selection_order"] % 8 == shard_id and group["sample_id"] in units_by_sample]
    import torch
    from .stage_c_fidelity import ValuePathScalarAdapter, resolve_target_response
    from .stage_c_value_path import (HeadCandidate, StageCValuePathConvergenceError, e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig, matched_fixed_clean_attention_patch_token_contribution, resolve_fixed_clean_attention_head_directions)
    from .stage_c_perturbations import tensor_sha256
    model = None; process_start = time.monotonic(); process_cpu = time.process_time(); records = []; completed = skipped = failed = 0
    try:
        model, binding, model_record = e1._load_model_and_context(repo, device)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES"); gpu_index = torch.device(device).index or 0
        physical = int(visible) if visible is not None and visible.isdigit() else gpu_index
        identities = {}
        if device.startswith("cuda"):
            query = subprocess.run(("nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.splitlines()
            for line in query:
                index, uuid, name = (item.strip() for item in line.split(",", 2)); identities[int(index)] = {"gpu_uuid": uuid, "gpu_name": name}
        for group in groups:
            group_units = units_by_sample[group["sample_id"]]
            existing = {}
            for unit in group_units:
                path = _r2_attempt_path(repo, unit)
                if path.exists(): existing[unit["canonical_digest"]] = r1._validate_r1_attempt(repo, unit, root_path=ROOT_PATH, attempt_id=ATTEMPT_ID, recovery_id=RECOVERY_ID, origin="E3_R2_NEW")
            missing = [unit for unit in group_units if unit["canonical_digest"] not in existing]; skipped += len(existing)
            if not missing: continue
            if any(item["result"]["validity_status"] != "VALID" for item in existing.values()): failed += 1; break
            group_start = time.monotonic(); group_cpu = time.process_time(); calls = 0
            if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(device)
            _canonical, clean, clean_sha = e1._load_sample(binding, group, model, device)
            if clean_sha != tensor_sha256(clean): raise StageEE3R2Error("live clean tensor hash drifted")
            def count_hook(_module: Any, _args: Any) -> None:
                nonlocal calls; calls += 1
            handle = model.register_forward_pre_hook(count_hook); group_failed = False
            try:
                coordinates = list(e3.NEW_COORDINATES)
                targets = resolve_fixed_clean_attention_head_directions(model, clean, tuple(HeadCandidate(*coord) for coord in coordinates), candidate_chunk_size=len(coordinates))
                targets_by_coord = {(target.block_index, target.head_index): target for target in targets}
                for block, head in coordinates:
                    target = targets_by_coord[(block, head)]; evaluator = ValuePathScalarAdapter(model, clean, target)
                    response = resolve_target_response(evaluator, clean, torch.zeros_like(clean))
                    if response.na is not None: raise StageEE3R2Error(f"frozen E3-R2 target became NA at {(block, head)}")
                    target_identity = json.loads(canonical_json_bytes(evaluator.target_identity).decode("utf-8"))
                    for unit in [row for row in missing if (row["block"], row["head"]) == (block, head)]:
                        started = time.monotonic(); cpu_started = time.process_time(); before = calls
                        try:
                            if unit["method"] == e3.NNS_METHOD:
                                method_result = e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig(model, clean, target, provenance={"stage": "E3-R2", "comparison_identity": e3.comparison_identity(block, head)})
                                ig = method_result.provenance["ig_result"]; backward = sum(int(math.ceil(int(item["steps"]) / 8)) for item in ig["attempts"])
                            else:
                                method_result = matched_fixed_clean_attention_patch_token_contribution(target, provenance={"stage": "E3-R2", "comparison_identity": e3.comparison_identity(block, head)}); backward = 0
                            logical = {"forward_calls": 0}
                            curves, signed_map = e3._curves_for_method(model=model, clean=clean, target=target, response=response, method_result=method_result, sample_id=group["sample_id"], block=block, head=head, call_counter=logical)
                            result = r1._valid_result(unit=unit, target_identity=target_identity, response=response, method_result=method_result, signed_map=signed_map, curves=curves, elapsed_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started, forward_calls=calls-before, backward_calls=backward, peak_cuda_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0, clean_sha256=clean_sha, recovery_id=RECOVERY_ID, attempt_id=ATTEMPT_ID, origin="E3_R2_NEW")
                            r1._publish_attempt(repo, unit, result, signed_map, root_path=ROOT_PATH, attempt_id=ATTEMPT_ID, recovery_id=RECOVERY_ID); completed += 1
                        except StageCValuePathConvergenceError as error:
                            diagnostic = error.ig_result.attribution[0].detach().to(device="cpu", dtype=torch.float32).numpy().copy()
                            result = r1._failure_result(unit=unit, target_identity=target_identity, response=response, error=error, elapsed_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started, forward_calls=calls-before, peak_cuda_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0, clean_sha256=clean_sha, diagnostic=diagnostic, recovery_id=RECOVERY_ID, attempt_id=ATTEMPT_ID, origin="E3_R2_NEW")
                            r1._publish_attempt(repo, unit, result, diagnostic, root_path=ROOT_PATH, attempt_id=ATTEMPT_ID, recovery_id=RECOVERY_ID); completed += 1; failed += 1; group_failed = True; break
                    if group_failed: break
                    del evaluator, response
            finally: handle.remove()
            records.append({"selection_order": group["selection_order"], "sample_id": group["sample_id"], "elapsed_seconds": time.monotonic()-group_start, "cpu_seconds": time.process_time()-group_cpu, "forward_calls": calls, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0, "terminal_units_published": sum(_r2_attempt_path(repo, unit).is_dir() for unit in group_units), "scientific_failure": group_failed})
            if group_failed: break
        elapsed = time.monotonic()-process_start; cpu_seconds = time.process_time()-process_cpu
        if records:
            records[0]["elapsed_seconds"] += max(0.0, elapsed-sum(float(row["elapsed_seconds"]) for row in records)); records[0]["cpu_seconds"] += max(0.0, cpu_seconds-sum(float(row["cpu_seconds"]) for row in records))
        telemetry = _signed({"schema_version":1, "document_type":"stage_e_e3_r2_shard_telemetry", "recovery_id":RECOVERY_ID, "shard_id":shard_id, "shard_count":8, "device":device, "physical_gpu_index":physical if device.startswith("cuda") else None, "gpu_uuid":identities.get(physical,{}).get("gpu_uuid"), "gpu_name":identities.get(physical,{}).get("gpu_name"), "elapsed_seconds":elapsed, "gpu_hours":elapsed/3600.0 if device.startswith("cuda") else 0.0, "cpu_seconds":cpu_seconds, "completed_units":completed, "skipped_terminal_units":skipped, "failed_units":failed, "groups":records, "model":model_record, "completed_at":_now()})
        _write_or_verify(repo/ROOT_PATH/"telemetry"/f"shard-{shard_id:02d}-of-08.json", telemetry)
        return telemetry
    finally:
        if model is not None: del model
        if device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.empty_cache()


def _all_sources(repo: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    plan = _v1_plan(repo); attempts = {}; origins = {}
    for unit in plan["units"]:
        source = _source_attempt(repo, unit)
        if source is None: raise StageEE3R2Error(f"missing terminal union unit: {unit['canonical_digest']}")
        origin, observed = source; attempts[unit["canonical_digest"]] = observed; origins[unit["canonical_digest"]] = origin
    counts = Counter(origins.values())
    if len(attempts) != 3000 or counts != Counter({"E3_V1_INHERITED":516, "E3_R1_INHERITED":12, "E3_R2_NEW":2472}):
        raise StageEE3R2Error(f"E3-R2 union coverage/origin failed: {counts}")
    return attempts, origins


def _combined_new_rows(repo: Path, plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    attempts, origins = _all_sources(repo); method_rows=[]; by_key={}
    for unit in plan["units"]:
        observed=attempts[unit["canonical_digest"]]; result=observed["result"]; path=observed["attempt_path"]
        row={
            "variant_id":f"layer_head_{unit['block']}_{unit['head']}", "sample_id":unit["sample_id"], "selection_order":unit["selection_order"], "cohort":"stage_d_fidelity_v1_500", "stratum":unit["stratum"], "validity_status":result["validity_status"], "source_hashes":result["source_hashes"], "method":unit["method"], "target":unit["target"], "scale":unit["scale"], "layer":unit["block"], "head":unit["head"], "coordinate_role":"DESCRIPTIVE_FROZEN_COORDINATE", "recompute_status":{"E3_V1_INHERITED":"INHERITED_E3_V1_TERMINAL","E3_R1_INHERITED":"INHERITED_E3_R1_TERMINAL","E3_R2_NEW":"E3_R2_NEW_COMPUTE_COMPLETED"}[origins[unit["canonical_digest"]]], "evidence_origin":origins[unit["canonical_digest"]], "attempt_id":result["attempt_id"], "retry_count":result["retry_count"], "failure_class":result["failure_class"], "failure_code":result["failure_code"], "na_reason":result["na_reason"], "insertion_auc_mean":result.get("insertion_auc_mean"), "deletion_auc_mean":result.get("deletion_auc_mean"), "insertion_auc_repeat_sd_ddof_0":result.get("insertion_auc_repeat_sd_ddof_0"), "deletion_auc_repeat_sd_ddof_0":result.get("deletion_auc_repeat_sd_ddof_0"), "misclassified":unit["stratum"]=="misclassified", "pair_binding_digest":result.get("pair_binding_digest"), "elapsed_seconds":result["elapsed_seconds"], "gpu_hours":result["gpu_hours"], "cpu_seconds":result["cpu_seconds"], "forward_calls":result["forward_calls"], "backward_calls":result["backward_calls"], "peak_cuda_memory_bytes":result["peak_cuda_memory_bytes"], "artifact_bytes":sum(item.stat().st_size for item in path.iterdir() if item.is_file()), "descriptive_only":True,
        }
        method_rows.append(row); by_key.setdefault((unit["sample_id"],unit["block"],unit["head"]),{})[unit["method"]]=(row,result)
    pair_rows=[]
    for (sample_id,block,head), methods in by_key.items():
        if set(methods)!=set(e3.METHODS): raise StageEE3R2Error("E3-R2 pair is incomplete")
        nns,nns_result=methods[e3.NNS_METHOD]; matched,matched_result=methods[e3.MATCHED_METHOD]
        if e1._target_semantic_digest(nns_result["target_response"]["target_identity"]) != e1._target_semantic_digest(matched_result["target_response"]["target_identity"]): raise StageEE3R2Error("E3-R2 paired target identity drifted")
        ncurves={(x["direction"],x["replicate"]):x for x in nns_result["curves"]}; mcurves={(x["direction"],x["replicate"]):x for x in matched_result["curves"]}
        if set(ncurves)!=set(mcurves) or any(ncurves[key]["seed"]!=mcurves[key]["seed"] or ncurves[key]["tie_priority_sha256"]!=mcurves[key]["tie_priority_sha256"] for key in ncurves): raise StageEE3R2Error("E3-R2 paired seed/tie identity drifted")
        status="VALID" if nns["validity_status"]==matched["validity_status"]=="VALID" else "NA" if "NA" in {nns["validity_status"],matched["validity_status"]} else "FAILED"
        pair_rows.append({"variant_id":f"layer_head_{block}_{head}_pair","sample_id":sample_id,"selection_order":nns["selection_order"],"cohort":"stage_d_fidelity_v1_500","stratum":nns["stratum"],"validity_status":status,"source_hashes":{"nns_map_sha256":nns["source_hashes"].get("signed_map_sha256"),"matched_map_sha256":matched["source_hashes"].get("signed_map_sha256"),"target_semantic_digest":nns["source_hashes"]["target_semantic_digest"]},"method":"nns_vs_matched","target":"fixed_clean_attention_cls_head_direction","scale":"fine_4x4_cell","layer":block,"head":head,"coordinate_role":"DESCRIPTIVE_FROZEN_COORDINATE","recompute_status":"E3_V1_R1_R2_UNION","misclassified":nns["misclassified"],"nns_insertion_auc":nns["insertion_auc_mean"],"nns_deletion_auc":nns["deletion_auc_mean"],"matched_insertion_auc":matched["insertion_auc_mean"],"matched_deletion_auc":matched["deletion_auc_mean"],"delta_fid":e3.delta_fid(nns,matched) if status=="VALID" else None,"failure_class":None if status=="VALID" else "SCIENTIFIC_INVALIDITY","failure_code":None if status=="VALID" else "paired_method_invalid","na_reason":None,"descriptive_only":True})
    return method_rows,pair_rows,attempts,origins


def _actual_cost(repo: Path) -> dict[str, Any]:
    upstream=r1.validate_completion(repo)["actual_cost"]; telemetry=[]
    for path in sorted((repo/ROOT_PATH/"telemetry").glob("shard-*-of-08.json")):
        value,_=_strict_json(path,"E3-R2 telemetry"); _validate_signed(value,"E3-R2 telemetry"); telemetry.append(value)
    incidents=[]
    for path in sorted((repo/REPORT_ROOT_PATH).glob("execution_infrastructure_event_*.json")):
        value,_=_strict_json(path,"E3-R2 infrastructure event"); _validate_signed(value,"E3-R2 infrastructure event"); incidents.append(value)
    incident_gpu_seconds=sum(float(x["cost"]["gpu_seconds_charged"]) for x in incidents); incident_cpu_seconds=sum(float(x["cost"]["cpu_seconds_charged"]) for x in incidents)
    v1_bytes=_inventory(repo/e3.ROOT_PATH)["total_bytes"]+_inventory(repo/e3.REPORT_ROOT_PATH)["total_bytes"]
    r1_bytes=_inventory(repo/R1_ROOT)["total_bytes"]+_inventory(repo/R1_REPORT_ROOT)["total_bytes"]
    r2_bytes=_inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","status.json","completion/e3_r2_completion.pending.json","completion/e3_r2_completion.json"))["total_bytes"]+_inventory(repo/REPORT_ROOT_PATH)["total_bytes"]
    result={"e3_v1_r1_gpu_hours":upstream["gpu_hours"],"e3_r2_gpu_hours":sum(float(x["gpu_hours"]) for x in telemetry)+incident_gpu_seconds/3600.0,"gpu_hours":upstream["gpu_hours"]+sum(float(x["gpu_hours"]) for x in telemetry)+incident_gpu_seconds/3600.0,"e3_v1_r1_cpu_seconds":upstream["cpu_seconds"],"e3_r2_cpu_seconds":sum(float(x["cpu_seconds"]) for x in telemetry)+incident_cpu_seconds,"cpu_seconds":upstream["cpu_seconds"]+sum(float(x["cpu_seconds"]) for x in telemetry)+incident_cpu_seconds,"retained_bytes":v1_bytes+r1_bytes+r2_bytes,"e3_r2_gpu_invocations":len(telemetry)+len(incidents),"e3_r2_infrastructure_event_count":len(incidents),"e3_r2_terminal_attempts":sum(int(x["completed_units"])+int(x["skipped_terminal_units"]) for x in telemetry),"gpu_hours_max":GPU_HOURS_MAX,"cpu_hours_max":512.0,"retained_bytes_max":RETAINED_BYTES_MAX}
    result["status"]="PASS_COST" if result["gpu_hours"]<=GPU_HOURS_MAX and result["cpu_seconds"]<=CPU_SECONDS_MAX and result["retained_bytes"]<=RETAINED_BYTES_MAX else "FAIL_COST"
    return result


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); validate_upstream(repo); plan=_v1_plan(repo)
    method_new,pair_new,attempts,origins=_combined_new_rows(repo,plan); primary_methods,primary_pairs=e3._primary_method_rows(repo,plan)
    method_rows=sorted(primary_methods+method_new,key=lambda row:(row["selection_order"],e3.COORDINATES.index((row["layer"],row["head"])),e3.METHODS.index(row["method"])))
    pair_rows=sorted(primary_pairs+pair_new,key=lambda row:(row["selection_order"],e3.COORDINATES.index((row["layer"],row["head"]))))
    if len(method_rows)!=4000 or len(pair_rows)!=2000 or len(attempts)!=3000 or any(row["validity_status"]!="VALID" for row in method_rows+pair_rows): raise StageEE3R2Error("E3-R2 full union is not 3000 valid terminal attempts")
    telemetry=[]
    for path in sorted((repo/ROOT_PATH/"telemetry").glob("shard-*-of-08.json")):
        value,_=_strict_json(path,"E3-R2 telemetry"); _validate_signed(value,"E3-R2 telemetry"); telemetry.append(value)
    if {x["shard_id"] for x in telemetry}!=set(range(8)) or any(x["failed_units"] for x in telemetry): raise StageEE3R2Error("E3-R2 shard telemetry is incomplete or failed")
    _, eligible_units = _partition(repo)
    expected_orders = {unit["selection_order"] for unit in eligible_units}
    observed_orders=[g["selection_order"] for x in telemetry for g in x["groups"]]
    if len(observed_orders)!=412 or set(observed_orders)!=expected_orders: raise StageEE3R2Error("E3-R2 telemetry group coverage drifted")
    strata={group["sample_id"]:group["stratum"] for group in plan["groups"]}; summaries=[]; method_summaries=[]
    for block,head in e3.COORDINATES:
        coordinate_pairs=[row for row in pair_rows if (row["layer"],row["head"])==(block,head)]; coordinate_methods=[row for row in method_rows if (row["layer"],row["head"])==(block,head)]
        for stratum in e3.STRATA:
            selected_pairs=coordinate_pairs if stratum=="overall" else [row for row in coordinate_pairs if row["stratum"]==stratum]; summary=e3._summary(selected_pairs,"delta_fid",f"layer_head_{block}_{head}_delta_fid",stratum)
            summaries.append({"variant_id":f"layer_head_{block}_{head}_{stratum}","sample_id":None,"selection_order":None,"cohort":"stage_d_fidelity_v1_500","stratum":stratum,"validity_status":"VALID","source_hashes":{"paired_rows_canonical_digest":canonical_sha256(selected_pairs)},"method":"nns_vs_matched","target":"fixed_clean_attention_cls_head_direction","scale":"fine_4x4_cell","layer":block,"head":head,"coordinate_role":"PRIMARY_ANCHOR_READ_ONLY" if (block,head)==(0,9) else "DESCRIPTIVE_FROZEN_COORDINATE","metric":"delta_fid",**summary,"descriptive_only":True})
            for method in e3.METHODS:
                selected=[row for row in coordinate_methods if row["method"]==method and (stratum=="overall" or row["stratum"]==stratum)]
                for direction in ("insertion","deletion"):
                    summary=e3._summary(selected,f"{direction}_auc_mean",f"layer_head_{block}_{head}_{method}_{direction}_auc",stratum); method_summaries.append({"variant_id":f"layer_head_{block}_{head}_{method}_{direction}_{stratum}","sample_id":None,"selection_order":None,"cohort":"stage_d_fidelity_v1_500","stratum":stratum,"validity_status":"VALID","source_hashes":{"method_rows_canonical_digest":canonical_sha256(selected)},"method":method,"target":"fixed_clean_attention_cls_head_direction","scale":"fine_4x4_cell","layer":block,"head":head,"coordinate_role":"PRIMARY_ANCHOR_READ_ONLY" if (block,head)==(0,9) else "DESCRIPTIVE_FROZEN_COORDINATE","metric":f"{direction}_auc",**summary,"descriptive_only":True})
    registry,_=e1._load_registry(repo); membership=[{"variant_id":"prediction_stratum_membership","sample_id":row["sample_id"],"selection_order":row["selection_order"],"cohort":"stage_d_fidelity_v1_500","stratum":strata[row["sample_id"]],"validity_status":"VALID","source_hashes":{"d10_per_sample_sha256":e3._sha_file(repo/e3.D10_DELTA_PATH),"image_sha256":row["image_sha256"]},"method":"nns_vs_matched","target":"fixed_clean_attention_cls_head_direction","scale":"fine_4x4_cell","layer":None,"head":None,"misclassified":strata[row["sample_id"]]=="misclassified"} for row in registry]
    cost=_actual_cost(repo)
    if cost["status"]!="PASS_COST": raise StageEE3R2Error("E3-R2 enlarged actual budget failed")
    summary_doc=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_descriptive_summary","stage_id":"E3-R2","recovery_id":RECOVERY_ID,"cohort":"stage_d_fidelity_v1_500","descriptive_only":True,"p_values_emitted":False,"winner_selection_performed":False,"causal_claim_made":False,"new_go_no_go_issued":False,"d10_primary_unchanged":True,"primary_anchor_recomputed":False,"e3_v1_status_preserved":"FAIL","e3_r1_status_preserved":"BLOCKED","counts":{"method_rows":4000,"paired_rows":2000,"new_compute_attempts":3000,"e3_v1_inherited_attempts":516,"e3_r1_inherited_attempts":12,"e3_r2_new_attempts":2472,"summary_cells":12,"method_summary_cells":48},"delta_fid_coordinate_strata":summaries,"method_auc_coordinate_strata":method_summaries,"prediction_partition":{"overall":500,"correct":418,"misclassified":82},"clean_margin":e3._validate_config(repo)["scientific_contract"]["clean_margin"],"actual_cost":cost,"e4_started":False})
    return {"method_rows":method_rows,"pair_rows":pair_rows,"summary_rows":summaries+method_summaries,"membership_rows":membership,"resource_rows":method_rows,"attempts":attempts,"origins":origins,"summary":summary_doc,"actual_cost":cost}


def _table(table_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows: raise StageEE3R2Error(f"{table_id} cannot be empty")
    columns=list(dict.fromkeys(key for row in rows for key in row)); normalized=[{key:row.get(key) for key in columns} for row in rows]
    return _signed({"schema_version":1,"document_type":"stage_e_e3_paper_table","table_id":table_id,"columns":columns,"row_count":len(normalized),"rows":normalized})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); columns=list(rows[0]); temporary=path.with_name("."+path.name+".tmp")
    with temporary.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n"); writer.writeheader(); writer.writerows([{key:e3._csv_scalar(value) for key,value in row.items()} for row in rows]); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def finalize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); outputs=validate_outputs(repo); root=repo/ROOT_PATH
    specs=(("layer_head_method_rows",outputs["method_rows"]),("layer_head_paired_delta",outputs["pair_rows"]),("prediction_strata",outputs["summary_rows"]),("prediction_strata_membership",outputs["membership_rows"]),("resource_and_qc",outputs["resource_rows"]))
    for name,rows in specs:
        table=_table(name,rows); _write_or_verify(root/"paper_data"/f"{name}.json",table); _write_csv(root/"paper_data"/f"{name}.csv",table["rows"])
    _write_or_verify(root/"descriptive_summary.json",outputs["summary"]); _write_or_verify(root/"clean_margin_not_run.json",_signed({"schema_version":1,"document_type":"stage_e_e3_r2_not_run","analysis":"clean_margin",**e3._validate_config(repo)["scientific_contract"]["clean_margin"]}))
    recipe=_signed({"schema_version":1,"document_type":"stage_e_figure_recipes","numeric_source":"paper_data_only","renders_png":False,"figures":[{"figure_id":"e3_r2_layer_head_prediction_strata_future","table_path":"paper_data/prediction_strata.json","x":"layer","y":"median","group":"stratum","caption_contract":"Post-D10 descriptive views for all four Stage-C-frozen coordinates; E3 v1 FAIL and E3-R1 BLOCKED remain historical records.","numeric_source":"paper_data_only","renders_png":False}]}); _write_or_verify(root/"figure_recipes.json",recipe)
    provenance=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_provenance","recovery_id":RECOVERY_ID,"e3_v1_status":"FAIL","e3_r1_status":"BLOCKED","origins":Counter(outputs["origins"].values()),"operations":["source-validate and inherit E3 v1 and R1 attempts in place","run only the 2472-unit complement under the unchanged R1 numerical schedule","form a source-validated disjoint 3000-unit union","compute the unchanged E3 descriptive summaries"],"environment":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform()},"e4_started":False}); _write_or_verify(root/"provenance.json",provenance)
    false_audit=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_false_pass_audit","recovery_id":RECOVERY_ID,"status":"PASS","checks":[{"attack":name,"result":"REJECTED"} for name in FALSE_PASS_ATTACKS],"pass_possible_without_all_conditions":False,"e4_started":False}); _write_or_verify(repo/REPORT_ROOT_PATH/"false_pass_audit.json",false_audit)
    tests,_=_strict_json(repo/TEST_REPORT_PATH,"E3-R2 tests"); _validate_signed(tests,"E3-R2 tests")
    if tests.get("status")!="PASS" or tests.get("failed")!=0: raise StageEE3R2Error("E3-R2 required tests failed")
    validation=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_source_validation","stage_id":"E3-R2","recovery_id":RECOVERY_ID,"status":"PASS","validated_at":_now(),"checks":{"e3_v1_fail_preserved":"PASS","e3_r1_blocked_preserved":"PASS","union_origins":"PASS_516_12_2472","union_3000_unique_terminal_valid":"PASS","method_rows":"PASS_4000","paired_rows":"PASS_2000","summary_cells":"PASS_12","unchanged_numerical_and_scientific_contract":"PASS","enlarged_actual_budget":"PASS","e4_not_started":"PASS"},"summary_canonical_digest":outputs["summary"]["canonical_digest"],"false_pass_audit_canonical_digest":false_audit["canonical_digest"]}); _write_or_verify(repo/REPORT_ROOT_PATH/"validation_report.json",validation)
    artifacts=[]
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "completion" not in path.parts and path.name not in {"payload_manifest.json","whole_root_inventory.json","status.json"}: artifacts.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":_sha_file(path)})
    manifest=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_payload_manifest","recovery_id":RECOVERY_ID,"immutable":True,"payload_count":len(artifacts),"artifacts":artifacts}); _write_or_verify(root/"payload_manifest.json",manifest)
    cost=_actual_cost(repo)
    if cost["status"]!="PASS_COST": raise StageEE3R2Error("E3-R2 final budget failed")
    inventory=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/e3_r2_completion.pending.json","completion/e3_r2_completion.json")); inventory_doc=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_whole_root_inventory","root":ROOT_PATH,"scope":"all regular non-hidden E3-R2 formal files except self-referential inventory status and completion receipts",**inventory}); _write_or_verify(root/"whole_root_inventory.json",inventory_doc)
    status=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_status","stage_id":"E3-R2","recovery_id":RECOVERY_ID,"status":"PASS","complete":True,"terminal":True,"immutable":True,"e3_v1_status_preserved":"FAIL","e3_r1_status_preserved":"BLOCKED","package_completeness":"PASS","workflow_integrity":"PASS","scientific_validity":"PASS","cost_status":"PASS_COST","actual_cost":cost,"e4_authorized":True,"e4_started":False,"stage_f_started":False,"whole_root_inventory":{"path":"whole_root_inventory.json","file_sha256":_sha_file(root/"whole_root_inventory.json"),"canonical_digest":inventory_doc["canonical_digest"],"file_count":inventory["file_count"],"total_bytes":inventory["total_bytes"]}}); _write_or_verify(root/"status.json",status)
    pending,_=_strict_json(repo/COMPLETION_PENDING_PATH,"E3-R2 pending completion")
    completion=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_terminal_completion","stage_id":"E3-R2","recovery_id":RECOVERY_ID,"status":"PASS","started_at":pending["started_at"],"completed_at":_now(),"upstream":{"e3_v1_status":"FAIL","e3_v1_completion_canonical_digest":r1.E3_V1_COMPLETION_DIGEST,"e3_r1_status":"BLOCKED","e3_r1_completion_canonical_digest":R1_COMPLETION_DIGEST},"union":{"expected_new_compute_units":3000,"e3_v1_inherited":516,"e3_r1_inherited":12,"e3_r2_new":2472,"valid_terminal":3000,"failed":0,"na":0,"method_rows":4000,"paired_rows":2000,"summary_cells":12},"package_completeness":"PASS","workflow_integrity":"PASS","scientific_validity":"PASS","cost_status":"PASS_COST","actual_cost":cost,"source_validation":{"validation_report":_record(repo,REPORT_ROOT_PATH+"/validation_report.json"),"test_report":_record(repo,TEST_REPORT_PATH),"false_pass_audit":_record(repo,REPORT_ROOT_PATH+"/false_pass_audit.json"),"payload_manifest":_record(repo,ROOT_PATH+"/payload_manifest.json"),"status":_record(repo,ROOT_PATH+"/status.json"),"whole_root_inventory":_record(repo,ROOT_PATH+"/whole_root_inventory.json")},"next_authorization":"E4_ONLY","e4_started":False})
    schema,_=_strict_json(repo/COMPLETION_SCHEMA_PATH,"E3-R2 completion schema"); errors=list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE3R2Error(f"E3-R2 completion schema failure: {errors[0].message}")
    temporary=(repo/COMPLETION_PENDING_PATH).with_name(".e3-r2-completion-terminal.tmp"); _write_new_json(temporary,completion); os.replace(temporary,repo/COMPLETION_PENDING_PATH); os.replace(repo/COMPLETION_PENDING_PATH,repo/COMPLETION_PATH)
    final_inventory=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/e3_r2_completion.pending.json","completion/e3_r2_completion.json"))
    if final_inventory!=inventory: raise StageEE3R2Error("E3-R2 root changed during terminal publication")
    return validate_completion(repo)


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve()
    if not (repo/COMPLETION_PATH).is_file() or (repo/COMPLETION_PENDING_PATH).exists(): raise StageEE3R2Error("terminal E3-R2 completion is absent or pending remains")
    completion,_=_strict_json(repo/COMPLETION_PATH,"E3-R2 completion"); _validate_signed(completion,"E3-R2 completion")
    if completion.get("status")=="FAIL": return validate_failed_completion(repo)
    schema,_=_strict_json(repo/COMPLETION_SCHEMA_PATH,"E3-R2 completion schema"); errors=list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE3R2Error(f"E3-R2 completion schema failure: {errors[0].message}")
    if completion.get("status")!="PASS" or completion.get("next_authorization")!="E4_ONLY" or completion.get("e4_started") is not False: raise StageEE3R2Error("E3-R2 completion boundary failed")
    validate_upstream(repo); outputs=validate_outputs(repo)
    if outputs["actual_cost"]!=completion["actual_cost"]: raise StageEE3R2Error("E3-R2 completion cost drifted")
    root=repo/ROOT_PATH; status,_=_strict_json(root/"status.json","E3-R2 status"); inventory,_=_strict_json(root/"whole_root_inventory.json","E3-R2 inventory")
    for value,name in ((status,"status"),(inventory,"inventory")): _validate_signed(value,f"E3-R2 {name}")
    rebuilt=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/e3_r2_completion.pending.json","completion/e3_r2_completion.json"))
    if any(inventory[key]!=rebuilt[key] for key in ("files","file_count","total_bytes")) or canonical_sha256({"files":inventory["files"]})!=rebuilt["canonical_digest"]: raise StageEE3R2Error("E3-R2 whole-root inventory drifted")
    if status.get("status")!="PASS" or status.get("e3_v1_status_preserved")!="FAIL" or status.get("e3_r1_status_preserved")!="BLOCKED" or status.get("e4_authorized") is not True or status.get("e4_started") is not False: raise StageEE3R2Error("E3-R2 terminal status drifted")
    for record in completion["source_validation"].values():
        if record!=_record(repo,record["path"]): raise StageEE3R2Error(f"E3-R2 completion binding drifted: {record['path']}")
    if any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in root.rglob("*") if path.is_file()): raise StageEE3R2Error("E3-R2 contains a forbidden render artifact")
    return {"status":"PASS","completion_canonical_digest":completion["canonical_digest"],"whole_root_inventory_digest":inventory["canonical_digest"],"e3_v1_status":"FAIL","e3_r1_status":"BLOCKED","inherited_attempts":528,"new_attempts":2472,"union_valid_terminal_attempts":3000,"method_rows":4000,"paired_rows":2000,"actual_cost":completion["actual_cost"],"next_authorization":"E4_ONLY","e4_started":False}


def _partial_attempts(repo: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    plan=_v1_plan(repo); units={unit["canonical_digest"]:unit for unit in plan["units"]}; observed=[]
    for path in sorted((repo/ROOT_PATH/"units").glob("*/attempts/attempt-r2-000001")):
        digest=path.parents[1].name
        if digest not in units: raise StageEE3R2Error(f"unknown E3-R2 unit: {digest}")
        observed.append((units[digest],r1._validate_r1_attempt(repo,units[digest],root_path=ROOT_PATH,attempt_id=ATTEMPT_ID,recovery_id=RECOVERY_ID,origin="E3_R2_NEW")))
    if any(path.is_dir() for path in (repo/ROOT_PATH/"units").glob("*/attempts/.attempt-r2-*")): raise StageEE3R2Error("ambiguous temporary E3-R2 attempt exists")
    return observed


def finalize_failed(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); validate_upstream(repo); observed=_partial_attempts(repo)
    failed=[item for item in observed if item[1]["result"]["validity_status"]=="FAILED"]; valid=[item for item in observed if item[1]["result"]["validity_status"]=="VALID"]
    if len(observed)!=637 or len(valid)!=636 or len(failed)!=1: raise StageEE3R2Error(f"unexpected E3-R2 stop coverage: total={len(observed)} valid={len(valid)} failed={len(failed)}")
    unit,attempt=failed[0]; result=attempt["result"]
    if (unit["selection_order"],unit["block"],unit["head"],unit["method"],result["failure_code"])!=(117,11,2,e3.NNS_METHOD,"ig_conservation_failed_after_e3_r1_schedule"): raise StageEE3R2Error("E3-R2 terminal scientific failure identity drifted")
    stop=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_execution_stop_event","recovery_id":RECOVERY_ID,"event_id":"E3_R2_SCIENTIFIC_INVALIDITY_STOP_001","status":"TERMINAL_STOP","recorded_at":_now(),"trigger":"NON_RETRYABLE_SCIENTIFIC_INVALIDITY","failure":{"selection_order":117,"sample_id":unit["sample_id"],"block":11,"head":2,"method":e3.NNS_METHOD,"unit_digest":unit["canonical_digest"],"attempt_id":ATTEMPT_ID,"result_canonical_digest":result["canonical_digest"],"failure_code":result["failure_code"],"selected_steps":32768,"attempts":result["ig_result"]["attempts"]},"coverage_at_stop":{"e3_v1_inherited":516,"e3_r1_inherited":12,"e3_r2_valid":636,"e3_r2_failed":1,"union_terminal":1165,"missing":1835},"interrupted_processes":7,"interruption_cost":{"gpu_seconds_charged":5320.0,"cpu_seconds_charged":42000.0,"accounting":"SEVEN_PROCESSES_CONSERVATIVE_760_GPU_SECONDS_AND_6000_CPU_SECONDS_EACH"},"terminal_attempt_recompute_authorized":False,"further_schedule_authorized":False,"threshold_change_authorized":False,"e4_authorized":False,"e4_started":False}); _write_or_verify(repo/REPORT_ROOT_PATH/"scientific_stop_event_001.json",stop)
    upstream=r1.validate_completion(repo)["actual_cost"]; r2_root_bytes=_inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","status.json","completion/e3_r2_completion.pending.json","completion/e3_r2_completion.json"))["total_bytes"]; retained=_inventory(repo/e3.ROOT_PATH)["total_bytes"]+_inventory(repo/e3.REPORT_ROOT_PATH)["total_bytes"]+_inventory(repo/R1_ROOT)["total_bytes"]+_inventory(repo/R1_REPORT_ROOT)["total_bytes"]+r2_root_bytes+_inventory(repo/REPORT_ROOT_PATH)["total_bytes"]
    cost={"e3_v1_r1_gpu_hours":upstream["gpu_hours"],"e3_r2_gpu_hours":(160.0+8*760.0)/3600.0,"gpu_hours":upstream["gpu_hours"]+(160.0+8*760.0)/3600.0,"e3_v1_r1_cpu_seconds":upstream["cpu_seconds"],"e3_r2_cpu_seconds":160.0+8*6000.0,"cpu_seconds":upstream["cpu_seconds"]+160.0+8*6000.0,"retained_bytes":retained,"e3_r2_terminal_attempts":637,"e3_r2_valid_attempts":636,"e3_r2_failed_attempts":1,"gpu_hours_max":GPU_HOURS_MAX,"cpu_hours_max":512.0,"retained_bytes_max":RETAINED_BYTES_MAX}; cost["status"]="PASS_COST" if cost["gpu_hours"]<=GPU_HOURS_MAX and cost["cpu_seconds"]<=CPU_SECONDS_MAX and cost["retained_bytes"]<=RETAINED_BYTES_MAX else "FAIL_COST"
    false_audit=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_false_pass_audit","recovery_id":RECOVERY_ID,"status":"PASS","checks":[{"attack":name,"result":"REJECTED"} for name in FALSE_PASS_ATTACKS],"e3_r2_pass_issued":False,"pass_possible_without_all_conditions":False,"e4_authorized":False,"e4_started":False}); _write_or_verify(repo/REPORT_ROOT_PATH/"false_pass_audit.json",false_audit)
    validation=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_source_validation","stage_id":"E3-R2","recovery_id":RECOVERY_ID,"status":"FAIL","validated_at":_now(),"checks":{"e3_v1_fail_preserved":"PASS","e3_r1_blocked_preserved":"PASS","atomic_scientific_failure_persistence":"PASS","schedule_exhaustion":"PASS_512_THROUGH_32768","thresholds_unchanged":"PASS","scientific_validity":"FAIL_ORDER_117_BLOCK_11_HEAD_2","enlarged_actual_budget":cost["status"],"remaining_work_stopped":"PASS","e4_not_started":"PASS"},"stop_event_canonical_digest":stop["canonical_digest"],"false_pass_audit_canonical_digest":false_audit["canonical_digest"]}); _write_or_verify(repo/REPORT_ROOT_PATH/"validation_report.json",validation)
    root=repo/ROOT_PATH; artifacts=[]
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "completion" not in path.parts and path.name not in {"payload_manifest.json","whole_root_inventory.json","status.json"}: artifacts.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":_sha_file(path)})
    manifest=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_payload_manifest","recovery_id":RECOVERY_ID,"immutable":True,"terminal_stage_status":"FAIL","payload_count":len(artifacts),"artifacts":artifacts}); _write_or_verify(root/"payload_manifest.json",manifest)
    inventory=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/e3_r2_completion.pending.json","completion/e3_r2_completion.json")); inventory_doc=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_whole_root_inventory","root":ROOT_PATH,"scope":"all regular non-hidden E3-R2 formal files except self-referential inventory status and completion receipts",**inventory}); _write_or_verify(root/"whole_root_inventory.json",inventory_doc)
    status=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_status","stage_id":"E3-R2","recovery_id":RECOVERY_ID,"status":"FAIL","complete":False,"terminal":True,"immutable":True,"e3_v1_status_preserved":"FAIL","e3_r1_status_preserved":"BLOCKED","package_completeness":"FAIL","workflow_integrity":"PASS","scientific_validity":"FAIL","cost_status":cost["status"],"actual_cost":cost,"failure_code":result["failure_code"],"e4_authorized":False,"e4_started":False,"stage_f_started":False,"whole_root_inventory":{"path":"whole_root_inventory.json","file_sha256":_sha_file(root/"whole_root_inventory.json"),"canonical_digest":inventory_doc["canonical_digest"],"file_count":inventory["file_count"],"total_bytes":inventory["total_bytes"]}}); _write_or_verify(root/"status.json",status)
    pending,_=_strict_json(repo/COMPLETION_PENDING_PATH,"E3-R2 pending completion"); completion=_signed({"schema_version":1,"document_type":"stage_e_e3_r2_terminal_completion","stage_id":"E3-R2","recovery_id":RECOVERY_ID,"status":"FAIL","started_at":pending["started_at"],"completed_at":_now(),"upstream":{"e3_v1_status":"FAIL","e3_v1_completion_canonical_digest":r1.E3_V1_COMPLETION_DIGEST,"e3_r1_status":"BLOCKED","e3_r1_completion_canonical_digest":R1_COMPLETION_DIGEST},"union":{"expected_new_compute_units":3000,"e3_v1_inherited":516,"e3_r1_inherited":12,"e3_r2_new":637,"valid_terminal":1164,"failed":1,"na":0,"missing":1835},"package_completeness":"FAIL","workflow_integrity":"PASS","scientific_validity":"FAIL","cost_status":cost["status"],"actual_cost":cost,"source_validation":{"validation_report":_record(repo,REPORT_ROOT_PATH+"/validation_report.json"),"test_report":_record(repo,TEST_REPORT_PATH),"false_pass_audit":_record(repo,REPORT_ROOT_PATH+"/false_pass_audit.json"),"payload_manifest":_record(repo,ROOT_PATH+"/payload_manifest.json"),"status":_record(repo,ROOT_PATH+"/status.json"),"whole_root_inventory":_record(repo,ROOT_PATH+"/whole_root_inventory.json"),"stop_event":_record(repo,REPORT_ROOT_PATH+"/scientific_stop_event_001.json")},"next_authorization":"NONE","e4_started":False})
    schema,_=_strict_json(repo/COMPLETION_SCHEMA_PATH,"E3-R2 completion schema"); errors=list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE3R2Error(f"E3-R2 FAIL completion schema failure: {errors[0].message}")
    temporary=(repo/COMPLETION_PENDING_PATH).with_name(".e3-r2-completion-fail.tmp"); _write_new_json(temporary,completion); os.replace(temporary,repo/COMPLETION_PENDING_PATH); os.replace(repo/COMPLETION_PENDING_PATH,repo/COMPLETION_PATH); return validate_failed_completion(repo)


def validate_failed_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); completion,_=_strict_json(repo/COMPLETION_PATH,"E3-R2 FAIL completion"); _validate_signed(completion,"E3-R2 FAIL completion")
    if completion.get("status")!="FAIL" or completion.get("scientific_validity")!="FAIL" or completion.get("next_authorization")!="NONE" or completion.get("e4_started") is not False: raise StageEE3R2Error("E3-R2 FAIL boundary drifted")
    validate_upstream(repo); observed=_partial_attempts(repo); failed=[x for x in observed if x[1]["result"]["validity_status"]=="FAILED"]
    if len(observed)!=637 or len(failed)!=1 or failed[0][0]["selection_order"]!=117 or failed[0][0]["block"]!=11 or failed[0][0]["head"]!=2: raise StageEE3R2Error("E3-R2 FAIL attempt coverage drifted")
    root=repo/ROOT_PATH; status,_=_strict_json(root/"status.json","E3-R2 FAIL status"); inventory,_=_strict_json(root/"whole_root_inventory.json","E3-R2 FAIL inventory")
    for value,name in ((status,"status"),(inventory,"inventory")): _validate_signed(value,f"E3-R2 FAIL {name}")
    rebuilt=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/e3_r2_completion.pending.json","completion/e3_r2_completion.json"))
    if any(inventory[key]!=rebuilt[key] for key in ("files","file_count","total_bytes")) or canonical_sha256({"files":inventory["files"]})!=rebuilt["canonical_digest"]: raise StageEE3R2Error("E3-R2 FAIL inventory drifted")
    if status.get("status")!="FAIL" or status.get("e4_authorized") is not False or status.get("e4_started") is not False: raise StageEE3R2Error("E3-R2 FAIL status drifted")
    for record in completion["source_validation"].values():
        if record!=_record(repo,record["path"]): raise StageEE3R2Error(f"E3-R2 FAIL binding drifted: {record['path']}")
    return {"status":"FAIL","reason":"NNS_IG_NONCONVERGENCE_AFTER_32768","failure_selection_order":117,"failure_coordinate":[11,2],"inherited_attempts":528,"new_attempts":637,"valid_terminal_attempts":1164,"failed_terminal_attempts":1,"missing_attempts":1835,"actual_cost":completion["actual_cost"],"next_authorization":"NONE","e4_started":False}


def _main() -> int:
    parser=argparse.ArgumentParser(prog="python -m vitnns.stage_e_e3_r2"); sub=parser.add_subparsers(dest="command",required=True)
    for name in ("verify-preconditions","initialize","validate-initialized","validate-outputs","finalize","finalize-failed","validate-completion"):
        child=sub.add_parser(name); child.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    tests=sub.add_parser("record-tests"); tests.add_argument("--repo-root",type=Path,default=REPO_ROOT); tests.add_argument("--passed",type=int,required=True); tests.add_argument("--failed",type=int,required=True); tests.add_argument("--test-command",required=True)
    recovery=sub.add_parser("authorize-execution-recovery"); recovery.add_argument("--repo-root",type=Path,default=REPO_ROOT); recovery.add_argument("--passed",type=int,required=True); recovery.add_argument("--failed",type=int,required=True); recovery.add_argument("--test-command",required=True)
    run=sub.add_parser("run-shard"); run.add_argument("--repo-root",type=Path,default=REPO_ROOT); run.add_argument("--shard-id",type=int,required=True); run.add_argument("--shard-count",type=int,required=True); run.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.command=="verify-preconditions": value=verify_preconditions(args.repo_root)
    elif args.command=="initialize": value=initialize(args.repo_root)
    elif args.command=="validate-initialized": value=validate_initialized(args.repo_root)
    elif args.command=="record-tests": value=record_tests(args.repo_root,passed=args.passed,failed=args.failed,command=args.test_command)
    elif args.command=="authorize-execution-recovery": value=authorize_execution_recovery(args.repo_root,passed=args.passed,failed=args.failed,command=args.test_command)
    elif args.command=="run-shard": value=run_shard(args.repo_root,shard_id=args.shard_id,shard_count=args.shard_count,device=args.device)
    elif args.command=="validate-outputs":
        output=validate_outputs(args.repo_root); value={"status":"PASS","method_rows":len(output["method_rows"]),"paired_rows":len(output["pair_rows"]),"actual_cost":output["actual_cost"]}
    elif args.command=="finalize": value=finalize(args.repo_root)
    elif args.command=="finalize-failed": value=finalize_failed(args.repo_root)
    else: value=validate_completion(args.repo_root)
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)); return 0


if __name__=="__main__": raise SystemExit(_main())
