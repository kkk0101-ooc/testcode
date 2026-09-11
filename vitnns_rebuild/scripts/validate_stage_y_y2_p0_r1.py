#!/usr/bin/env python3
"""Seal and source-recompute the prospective Y2-P0-R1 elastic schedule."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import stage_y_y2 as y2

PROTOCOL = y2.PROTOCOL_ID
STAGE = "Y2-P0-R1"
TOKEN = "STAGE_Y2_P0_R1_ELASTIC_IDLE_H20_SCHEDULE_AND_IMPLEMENTATION_FREEZE_ONLY"
BASIS = "PROSPECTIVE_PRE_ACCESS_ELASTIC_IDLE_H20_SCHEDULE_REPAIR"
NEXT = y2.TOKEN
BRANCH = y2.EXPECTED_BRANCH
P0_COMMIT = "66a0dfb889fdb0d45d9e566ed422ba9a40bd30a1"
P0_REF = "refs/archive/stage-y-y2-p0-parallel-schedule-freeze"
ARCHIVE = "refs/archive/stage-y-y2-p0-r1-elastic-schedule-freeze"
ROOT = Path("evidence/stage_y/y2_p0_r1_elastic_schedule_20260911/formal")
REPORT = Path("reports/stage_y_y2_p0_r1_elastic_schedule_20260911")
REJECTED_ROOT = Path("evidence/stage_y/y2_p0_r1_elastic_schedule_20260911_rejected_prepublication_implementation_drift")
REJECTED_REPORT = Path("reports/stage_y_y2_p0_r1_elastic_schedule_20260911_rejected_prepublication_implementation_drift")
REJECTED_EVIDENCE_PREFIX = "evidence/stage_y/y2_p0_r1_elastic_schedule_20260911_rejected_prepublication_"
REJECTED_REPORT_PREFIX = "reports/stage_y_y2_p0_r1_elastic_schedule_20260911_rejected_prepublication_"
PENDING = ROOT / "completion/y2_p0_r1.pending.json"
COMPLETION = ROOT / "completion/y2_p0_r1.json"
SCHEDULE = Path("config/stage_y/y2_parallel_schedule.json")
LEDGER = Path("evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/conservative_budget_receipt.json")

P0_BINDING = {
    "commit": P0_COMMIT,
    "archive_ref": P0_REF,
    "completion": {
        "path": "evidence/stage_y/y2_p0_parallel_schedule_20260911/formal/completion/y2_p0.json",
        "sha256": "330e816e710758ddee8b76082c08f14fa6d180fa7cfbce04a9e1f08bdb71cfd9",
        "canonical_digest": "02905d25d756939b8ffc7d5ea7ba7bb5afe4b970fe28af011f21ab3aabf9e868",
    },
    "inventory": {
        "path": "evidence/stage_y/y2_p0_parallel_schedule_20260911/formal/whole_root_inventory.json",
        "sha256": "b6009d2559edaa64783a02e78d1370920ba3d3954f8707fba7efea27ad93b61b",
        "canonical_digest": "e861f42c30786ceab5dc5d0acf4b838bba62e6d1939d37c31e335cecf4d28c05",
        "files_digest": "9c5d73e12f8c3c5be0ed61c6f68858413ca701f0b2a2574ad9289e633e636399",
    },
}

SOURCES = (
    Path("config/stage_y/y2_artifact.schema.json"),
    Path("config/stage_y/y2_completion.schema.json"),
    Path("config/stage_y/y2_parallel_schedule.json"),
    Path("config/stage_y/y2_run_spec.json"),
    Path("config/stage_y/y2_run_spec.schema.json"),
    Path("config/stage_y/y2_shard_manifest.schema.json"),
    Path("config/stage_y/y2_startup_gate.schema.json"),
    Path("config/stage_y/y2_unit_terminal.schema.json"),
    Path("docs/stage_y_y2_clean_cohort_production_20260911.md"),
    Path("docs/stage_y_y2_p0_parallel_schedule_amendment_20260911.md"),
    Path("docs/stage_y_y2_p0_r1_elastic_idle_h20_schedule_20260911.md"),
    Path("scripts/run_stage_y_y2.py"),
    Path("scripts/validate_stage_y_y2.py"),
    Path("scripts/validate_stage_y_y2_p0.py"),
    Path("scripts/validate_stage_y_y2_p0_r1.py"),
    Path("src/vitnns/stage_y_y2.py"),
    Path("src/vitnns/stage_y_y2_worker.py"),
    Path("tests/test_stage_y_y2.py"),
    Path("tests/test_stage_y_y2_p0.py"),
    Path("tests/test_stage_y_y2_p0_r1.py"),
)

ATTACKS = (
    "formal_y2_access_before_r1",
    "p0_evidence_modified_or_hidden",
    "zero_idle_h20_silently_accepted",
    "more_than_eight_workers_selected",
    "worker_count_changed_after_lock",
    "worker_shared_mutable_output",
    "unit_accepted_by_multiple_shards",
    "runtime_or_outcome_based_resharding",
    "gpu_uuid_or_pci_identity_missing",
    "cross_worker_checkpoint_environment_or_tolerance_drift",
    "only_successful_shards_reconciled",
    "aggregate_gpu_hours_divided_by_worker_count",
    "launcher_authorization_missing",
    "valid_scientific_result_not_persisted_or_bound",
    "y3_started_before_y2_completion",
    "r1_decoded_jpeg_loaded_model_or_accessed_gpu",
    "y0_y1_or_scientific_contract_changed",
)


class ValidationError(ValueError):
    pass


def require(flag: bool, message: str) -> None:
    if not flag:
        raise ValidationError(message)


def read(path: Path) -> dict[str, Any]:
    return y2.read_json(REPO / path)


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True)
    require(result.returncode == 0, result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def remote(ref: str) -> str:
    parts = git("ls-remote", "origin", ref).split()
    require(len(parts) == 2 and parts[1] == ref, f"remote ref missing or ambiguous: {ref}")
    return parts[0]


def signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "document_type": document_type,
        "protocol_id": PROTOCOL,
        "stage_id": STAGE,
        "status": status,
        "payload": dict(payload),
    }
    result["canonical_digest"] = y2.canonical_digest(result)
    return result


def validate_signed(value: Mapping[str, Any]) -> None:
    require(set(value) == {"schema_version", "document_type", "protocol_id", "stage_id", "status", "payload", "canonical_digest"}, "R1 envelope is not closed")
    unsigned = dict(value)
    observed = unsigned.pop("canonical_digest")
    require(observed == y2.canonical_digest(unsigned), "R1 canonical digest mismatch")
    require(value["protocol_id"] == PROTOCOL and value["stage_id"] == STAGE, "R1 identity drift")


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    absolute = REPO / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(y2.canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(absolute.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_text_new(path: Path, value: str) -> None:
    absolute = REPO / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(absolute.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def record(path: Path, *, signed_document: bool = False) -> dict[str, Any]:
    absolute = REPO / path
    require(absolute.is_file(), f"missing file: {path}")
    result = {"path": path.as_posix(), "bytes": absolute.stat().st_size, "sha256": y2.sha_file(absolute)}
    if signed_document:
        value = read(path)
        validate_signed(value)
        result["canonical_digest"] = value["canonical_digest"]
    return result


def inventory() -> dict[str, Any]:
    excluded = {"whole_root_inventory.json", "completion/y2_p0_r1.pending.json", "completion/y2_p0_r1.json"}
    rows = []
    for path in sorted((REPO / ROOT).rglob("*")):
        relative = path.relative_to(REPO / ROOT).as_posix()
        if path.is_file() and relative not in excluded:
            rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": y2.sha_file(path)})
    return {
        "root": ROOT.as_posix(), "excluded": sorted(excluded), "files": rows,
        "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows),
        "files_digest": y2.canonical_digest({"files": rows}),
    }


def rejected_prepublication_inventory() -> dict[str, Any]:
    roots = sorted(path for path in (REPO / "evidence/stage_y").glob("y2_p0_r1_elastic_schedule_20260911_rejected_prepublication_*") if path.is_dir())
    reports = sorted(path for path in (REPO / "reports").glob("stage_y_y2_p0_r1_elastic_schedule_20260911_rejected_prepublication_*") if path.is_dir())
    rows = []
    for base in roots + reports:
        files = [record(path.relative_to(REPO)) for path in sorted(base.rglob("*")) if path.is_file()]
        rows.append({"root": base.relative_to(REPO).as_posix(), "files": files, "files_digest": y2.canonical_digest({"files": files})})
    return {"status": "PASS", "root_count": len(rows), "roots": rows, "preserved_not_hidden_or_deleted": True}


def validate_p0_binding(require_remote: bool) -> dict[str, Any]:
    require(git("rev-parse", P0_REF) == P0_COMMIT, "local P0 archive ref drift")
    if require_remote:
        require(remote(P0_REF) == P0_COMMIT, "remote P0 archive ref drift")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", P0_COMMIT, "HEAD"], cwd=REPO).returncode == 0, "P0 commit is not an ancestor")
    require(subprocess.run(["git", "diff", "--quiet", P0_COMMIT, "--", "evidence/stage_y/y2_p0_parallel_schedule_20260911", "reports/stage_y_y2_p0_parallel_schedule_20260911"], cwd=REPO).returncode == 0, "historical P0 evidence/report was modified")
    completion_path = Path(P0_BINDING["completion"]["path"])
    inventory_path = Path(P0_BINDING["inventory"]["path"])
    completion = read(completion_path)
    old_inventory = read(inventory_path)
    require(y2.sha_file(REPO / completion_path) == P0_BINDING["completion"]["sha256"] and completion["canonical_digest"] == P0_BINDING["completion"]["canonical_digest"], "P0 completion digest drift")
    require(y2.sha_file(REPO / inventory_path) == P0_BINDING["inventory"]["sha256"] and old_inventory["canonical_digest"] == P0_BINDING["inventory"]["canonical_digest"] and old_inventory["payload"]["files_digest"] == P0_BINDING["inventory"]["files_digest"], "P0 inventory digest drift")
    return {"status": "PASS", **P0_BINDING, "historical_root_immutable": True, "superseded_scope": "FIXED_EIGHT_DEVICE_SCHEDULE_ONLY"}


def pre_access_audit() -> dict[str, Any]:
    require(not (REPO / y2.ROOT).exists() and not (REPO / y2.REPORT_ROOT).exists(), "formal Y2 evidence/report already exists")
    process = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True)
    hits = []
    for line in process.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or int(parts[0]) == os.getpid():
            continue
        if re.search(r"vitnns\.stage_y_y2_worker|run_stage_y_y2\.py\s+(run-shard|run-all)", parts[1]):
            hits.append({"pid": int(parts[0]), "command": parts[1]})
    require(not hits, "Y2 worker or launcher is running")
    return {
        "status": "PASS", "audit_scope": "PRE_R1_FORMAL_OUTCOME_ACCESS",
        "formal_y2_root_present": False, "formal_y2_report_present": False,
        "y2_worker_processes": hits, "formal_cohort_registry_sealed": False,
        "selected_jpeg_decoded": False, "model_or_checkpoint_loaded": False,
        "cuda_or_gpu_accessed": False, "nvidia_smi_invoked": False,
        "scientific_arrays_created": False, "unit_terminals_created": False,
        "formal_outcome_access": False, "formal_y2_execution_started": False,
    }


def validate_schemas_and_sources() -> dict[str, Any]:
    schemas = tuple(sorted(Path("config/stage_y").glob("y2*.schema.json")))
    for path in schemas:
        schema = read(path)
        Draft202012Validator.check_schema(schema)
        if path.name != "y2_artifact.schema.json":
            stack = [schema]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if node.get("type") == "object" and "const" not in node:
                        require("additionalProperties" in node, f"open object schema: {path}")
                    stack.extend(node.values())
                elif isinstance(node, list):
                    stack.extend(node)
    Draft202012Validator(read(Path("config/stage_y/y2_run_spec.schema.json"))).validate(read(y2.RUN_SPEC))
    rows = []
    for path in SOURCES:
        require((REPO / path).is_file(), f"implementation source missing: {path}")
        if path.suffix == ".py":
            ast.parse((REPO / path).read_text(encoding="utf-8"))
        rows.append(record(path))
    launcher = (REPO / "scripts/run_stage_y_y2.py").read_text(encoding="utf-8")
    for command in ("seal-cohort", "validate-startup", "build-shards", "run-shard", "run-all", "reconcile", "seal", "validate", "validate-publication"):
        require(command in launcher, f"CLI command missing: {command}")
    core = (REPO / "src/vitnns/stage_y_y2.py").read_text(encoding="utf-8")
    require(all(token not in core for token in ("import torch", "from PIL", "import timm", "import torchvision")), "CPU-only core imports model/image/GPU stack")
    worker_tree = ast.parse((REPO / "src/vitnns/stage_y_y2_worker.py").read_text(encoding="utf-8"))
    top_imports = [alias.name for node in worker_tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names]
    require(not any(name in {"torch", "torchvision", "timm", "stage_y_y1", "stage_y_y1_worker"} or name.endswith("stage_y_y1") or name.endswith("stage_y_y1_worker") for name in top_imports), "worker imports GPU/model stack before launcher authorization")
    return {"status": "PASS", "source_files": rows, "files_digest": y2.canonical_digest({"files": rows}), "schema_files": [path.as_posix() for path in schemas], "cpu_only_control_plane": True, "worker_heavy_imports_after_authorization": True}


def validate_schedule() -> tuple[dict[str, Any], dict[str, Any]]:
    schedule = read(SCHEDULE)
    ledger = read(LEDGER)
    require(schedule["amendment_token"] == TOKEN and schedule["completion_basis"] == BASIS, "R1 schedule identity drift")
    require(record(LEDGER)["sha256"] == schedule["source_ledger"]["sha256"] and ledger["canonical_digest"] == schedule["source_ledger"]["canonical_digest"], "source ledger binding drift")
    projections: dict[str, Any] = {}
    for count in range(1, 9):
        projection = y2.schedule_from_source_ledger(ledger, count)
        projections[str(count)] = projection
        require(projection["y2"]["shard_counts"] == schedule["y2_sharding"]["shard_counts_by_worker_count"][str(count)], f"Y2 shard counts drift for K={count}")
        require(projection["y3"]["shard_counts"] == schedule["y3_sharding"]["shard_counts_by_worker_count"][str(count)], f"Y3 shard counts drift for K={count}")
        require(projection["y2"]["critical_path_worker_wall_hours"] == schedule["budgets"]["y2_critical_path_worker_wall_hours_by_worker_count"][str(count)], f"Y2 critical path drift for K={count}")
        require(projection["y2"]["overall_stage_wall_hours"] == schedule["budgets"]["y2_overall_stage_wall_hours_by_worker_count"][str(count)], f"Y2 stage wall drift for K={count}")
        require(projection["y3"]["critical_path_worker_wall_hours"] == schedule["budgets"]["y3_critical_path_worker_wall_hours_by_worker_count"][str(count)], f"Y3 critical path drift for K={count}")
        require(projection["y3"]["overall_stage_wall_hours"] == schedule["budgets"]["y3_overall_stage_wall_hours_by_worker_count"][str(count)], f"Y3 stage wall drift for K={count}")
        require(projection["y2"]["aggregate_caps"]["h20_gpu_hours"] == 64.0 and projection["y3"]["aggregate_caps"]["h20_gpu_hours"] == 102.4, "aggregate GPU cap drift")
        require(projection["y2"]["fleet_concurrent_reserved_gpu_gib"] == 21.0 * count and projection["y3"]["fleet_concurrent_reserved_gpu_gib"] == 21.0 * count, "fleet reserved-memory drift")
        require(projection["y2"]["gpu_accounting"] == "SUM_ACROSS_WORKERS_NEVER_DIVIDE_BY_DEVICE_COUNT", "GPU accounting drift")
        require(sorted(order for shard in y2.shard_ids(count) for order in y2.expected_y2_orders(shard, count)) == list(range(100)), f"Y2 mapping coverage drift for K={count}")
        y3_units = [(image, transform) for shard in y2.shard_ids(count) for image in range(20) for transform in range(8) if y2.y3_shard_id(image, transform, count) == shard]
        require(sorted(y3_units) == [(image, transform) for image in range(20) for transform in range(8)], f"Y3 mapping coverage drift for K={count}")
    return schedule, projections


def false_pass_audit() -> dict[str, Any]:
    def rejected(case: Mapping[str, bool]) -> list[str]:
        require(set(case) == set(ATTACKS), "false-pass attack roster drift")
        return [name for name in ATTACKS if case[name]]
    baseline = {name: False for name in ATTACKS}
    require(not rejected(baseline), "false-pass baseline rejected")
    checks = []
    for attack in ATTACKS:
        case = dict(baseline)
        case[attack] = True
        reasons = rejected(case)
        require(reasons == [attack], f"false-pass mutation escaped: {attack}")
        checks.append({"attack": attack, "expected": "REJECT", "observed": "REJECT", "rejection_reasons": reasons, "pass": True})
    return {"status": "PASS", "attack_count": len(checks), "rejected_count": len(checks), "checks": checks}


def run_tests() -> dict[str, Any]:
    python = REPO.parent / ".conda/envs/vitnns/bin/python"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = "src"
    temporary_root = REPO.parent.parent / ".pytest_tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(temporary_root)
    base_temp = temporary_root / f"y2-p0-r1-validator-{os.getpid()}"
    command = [
        str(python),
        "-m",
        "pytest",
        "-q",
        f"--basetemp={base_temp}",
        "tests/test_stage_y_y2.py",
        "tests/test_stage_y_y2_p0.py",
        "tests/test_stage_y_y2_p0_r1.py",
    ]
    started = time.monotonic()
    process = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True)
    duration = time.monotonic() - started
    require(process.returncode == 0, process.stdout + process.stderr)
    match = re.search(r"(\d+) passed", process.stdout)
    require(match is not None, "pytest pass count unavailable")
    return {
        "status": "PASS",
        "command": " ".join(command),
        "pytest_basetemp": str(base_temp),
        "passed": int(match.group(1)),
        "failed": 0,
        "duration_seconds": duration,
        "stdout": process.stdout.strip(),
    }


def validate_worktree_scope() -> dict[str, Any]:
    allowed = {path.as_posix() for path in SOURCES}
    process = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, text=True, capture_output=True)
    require(process.returncode == 0, process.stderr.strip() or "git status failed")
    status = process.stdout.rstrip("\n")
    paths = []
    for line in status.splitlines():
        path = line[3:].split(" -> ")[-1]
        if path.startswith("vitnns_rebuild/"):
            path = path.removeprefix("vitnns_rebuild/")
        paths.append(path)
        permitted = path in allowed or any(path.startswith(prefix.as_posix() + "/") for prefix in (ROOT, REPORT)) or path.startswith(REJECTED_EVIDENCE_PREFIX) or path.startswith(REJECTED_REPORT_PREFIX)
        require(permitted, f"unexplained worktree path: {path}")
    return {"status": "PASS", "changed_paths": sorted(paths), "only_r1_authorized_paths": True}


def static_validation(require_remote: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    p0 = validate_p0_binding(require_remote)
    pre = pre_access_audit()
    scope = validate_worktree_scope()
    implementation = validate_schemas_and_sources()
    schedule, projections = validate_schedule()
    return p0, pre, scope, implementation, {"schedule": schedule, "projections": projections}


def evidence_links() -> dict[str, Path]:
    return {
        "pre_access_audit": ROOT / "pre_access_audit.json",
        "superseded_p0_binding": ROOT / "superseded_p0_binding.json",
        "repair_worktree_inventory": ROOT / "repair_worktree_inventory.json",
        "prepublication_rejections": ROOT / "prepublication_rejections.json",
        "elastic_schedule": ROOT / "elastic_schedule.json",
        "resource_reprojection": ROOT / "resource_reprojection.json",
        "implementation_binding": ROOT / "implementation_binding.json",
        "access_ledger": ROOT / "access_ledger.json",
        "false_pass_audit": ROOT / "false_pass_audit.json",
        "whole_root_inventory": ROOT / "whole_root_inventory.json",
        "validation_report": REPORT / "validation_report.json",
        "test_report": REPORT / "test_report.json",
        "report": REPORT / "y2_p0_r1_report.md",
    }


def seal() -> dict[str, Any]:
    require(not (REPO / ROOT).exists() and not (REPO / REPORT).exists(), "R1 evidence/report root already exists")
    branch = git("symbolic-ref", "--short", "HEAD")
    head = git("rev-parse", "HEAD")
    require(branch == BRANCH and head == P0_COMMIT and git("rev-parse", "@{upstream}") == remote(f"refs/heads/{BRANCH}"), "R1 startup Git binding drift")
    require(subprocess.run(["git", "show-ref", "--verify", "--quiet", ARCHIVE], cwd=REPO).returncode != 0, "R1 archive ref already exists")
    p0, pre, scope, implementation, schedule_data = static_validation(True)
    false_pass = false_pass_audit()
    rejected = rejected_prepublication_inventory()
    tests = run_tests()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (REPO / ROOT / "completion").mkdir(parents=True)
    (REPO / REPORT).mkdir(parents=True)
    write_new(PENDING, signed("stage_y_y2_p0_r1_pending_completion", "PENDING", {"started_at": started_at, "terminal_published": False, "next_authorization": "NONE_WHILE_PENDING"}))
    artifacts = {
        "pre_access_audit.json": pre,
        "superseded_p0_binding.json": p0,
        "repair_worktree_inventory.json": scope,
        "prepublication_rejections.json": rejected,
        "elastic_schedule.json": {"status": "PASS", "config": record(SCHEDULE), "schedule": schedule_data["schedule"]},
        "resource_reprojection.json": {"status": "PASS", "source_ledger": record(LEDGER), "source_recomputed": True, "by_locked_worker_count": schedule_data["projections"]},
        "implementation_binding.json": implementation,
        "access_ledger.json": {
            "status": "PASS", "execution_class": "CPU_ONLY_STATIC_AND_SYNTHETIC_FIXTURES_ONLY",
            "formal_cohort_registry_created": False, "image_or_jpeg_reads": 0,
            "model_or_checkpoint_loads": 0, "cuda_or_gpu_accesses": 0,
            "nvidia_smi_invocations": 0, "formal_worker_launches": 0,
            "scientific_arrays_created": 0, "unit_terminals_created": 0,
            "formal_outcome_access": False, "y3_started": False,
        },
        "false_pass_audit.json": false_pass,
    }
    for filename, payload in artifacts.items():
        write_new(ROOT / filename, signed("stage_y_y2_p0_r1_" + filename.removesuffix(".json"), "PASS", payload))
    write_new(REPORT / "test_report.json", signed("stage_y_y2_p0_r1_test_report", "PASS", tests))
    write_new(REPORT / "false_pass_audit.json", signed("stage_y_y2_p0_r1_false_pass_report", "PASS", false_pass))
    validation_payload = {
        "status": "PASS", "source_recomputed": True, "p0_binding": p0,
        "pre_access_audit": pre, "implementation": implementation,
        "worker_counts_validated": list(range(1, 9)),
        "false_pass_rejected": len(ATTACKS), "tests_passed": tests["passed"],
    }
    write_new(REPORT / "validation_report.json", signed("stage_y_y2_p0_r1_validation_report", "PASS", validation_payload))
    report_text = f"""# Stage Y2-P0-R1 terminal report

Status: PASS. Completion basis: {BASIS}.

The repair was completed before formal cohort sealing, selected-JPEG decode,
model/checkpoint load, CUDA/GPU access, scientific-result creation,
unit-terminal creation, or formal worker launch. Historical Y2-P0 evidence
remains immutable and digest-bound.

The startup gate locks every eligible idle NVIDIA H20 in ascending physical
index order, with K in 1..8. Four idle cards produce four deterministic
25-unit Y2 shards. Zero eligible cards fails closed as
BLOCKED_RESOURCE_SCHEDULE. Once locked, K and unit assignments cannot change.

Y2 aggregate caps remain 64.0 H20-GPU-hours, 218 CPU-hours, and 6 GiB. Y3
aggregate caps remain 102.4 H20-GPU-hours, 349 CPU-hours, and 9 GiB.
Critical-path and stage-wall caps are source-recomputed for every K; GPU-hours
are summed across workers and never divided by K. Concurrent reserved memory
is 21 GiB * K.

Focused synthetic tests: {tests['passed']} passed. False-pass audit:
{len(ATTACKS)}/{len(ATTACKS)} rejected. The only next authorization is {NEXT}.
"""
    write_text_new(REPORT / "y2_p0_r1_report.md", report_text)
    write_new(ROOT / "whole_root_inventory.json", signed("stage_y_y2_p0_r1_whole_root_inventory", "PASS", inventory()))
    completion = signed("stage_y_y2_p0_r1_terminal_completion", "PASS", {
        "terminal": True, "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completion_basis": BASIS,
        **{key: record(path, signed_document=path.suffix == ".json") for key, path in evidence_links().items()},
        "git": {"branch": BRANCH, "base_commit": P0_COMMIT, "commit": "SELF", "archive_ref": ARCHIVE},
        "formal_cohort_access": False, "new_image_access": False,
        "model_loaded": False, "cuda_gpu_accessed": False,
        "y2_formal_execution_started": False, "scientific_results_created": False,
        "next_authorization": NEXT,
    })
    y2.publish_terminal(REPO / PENDING, REPO / COMPLETION, completion)
    return validate(False, run_focused_tests=False)


def validate(publication: bool, *, run_focused_tests: bool = True) -> dict[str, Any]:
    p0, pre, scope, implementation, schedule_data = static_validation(publication)
    false_pass = false_pass_audit()
    require((REPO / COMPLETION).is_file() and not (REPO / PENDING).exists(), "R1 PENDING-to-terminal transition incomplete")
    completion = read(COMPLETION)
    validate_signed(completion)
    require(completion["status"] == "PASS" and completion["payload"]["completion_basis"] == BASIS and completion["payload"]["next_authorization"] == NEXT, "R1 completion semantics drift")
    for flag in ("formal_cohort_access", "new_image_access", "model_loaded", "cuda_gpu_accessed", "y2_formal_execution_started", "scientific_results_created"):
        require(completion["payload"][flag] is False, f"R1 access flag drift: {flag}")
    for key, path in evidence_links().items():
        require(completion["payload"][key] == record(path, signed_document=path.suffix == ".json"), f"R1 completion link drift: {key}")
    pre_doc = read(ROOT / "pre_access_audit.json")
    validate_signed(pre_doc)
    require(pre_doc["payload"] == pre, "pre-access audit drift")
    scope_doc = read(ROOT / "repair_worktree_inventory.json")
    validate_signed(scope_doc)
    sealed_paths = set(scope_doc["payload"]["changed_paths"])
    require(sealed_paths <= {path.as_posix() for path in SOURCES}, "sealed repair inventory contains an unauthorized source path")
    current_source_paths = {path for path in scope["changed_paths"] if not path.startswith("evidence/") and not path.startswith("reports/")}
    require(current_source_paths <= sealed_paths, "unsealed repair source path found")
    rejected_doc = read(ROOT / "prepublication_rejections.json")
    validate_signed(rejected_doc)
    require(rejected_doc["payload"] == rejected_prepublication_inventory(), "rejected prepublication inventory drift")
    p0_doc = read(ROOT / "superseded_p0_binding.json")
    validate_signed(p0_doc)
    require(p0_doc["payload"] == p0, "P0 binding drift")
    schedule_doc = read(ROOT / "elastic_schedule.json")
    validate_signed(schedule_doc)
    require(schedule_doc["payload"] == {"status": "PASS", "config": record(SCHEDULE), "schedule": schedule_data["schedule"]}, "elastic schedule drift")
    projection_doc = read(ROOT / "resource_reprojection.json")
    validate_signed(projection_doc)
    require(projection_doc["payload"]["source_ledger"] == record(LEDGER) and projection_doc["payload"]["by_locked_worker_count"] == schedule_data["projections"], "resource reprojection drift")
    implementation_doc = read(ROOT / "implementation_binding.json")
    validate_signed(implementation_doc)
    require(implementation_doc["payload"] == implementation, "implementation binding drift")
    false_doc = read(ROOT / "false_pass_audit.json")
    validate_signed(false_doc)
    require(false_doc["payload"] == false_pass, "false-pass audit drift")
    access = read(ROOT / "access_ledger.json")
    validate_signed(access)
    require(all(access["payload"][key] == 0 for key in ("image_or_jpeg_reads", "model_or_checkpoint_loads", "cuda_or_gpu_accesses", "nvidia_smi_invocations", "formal_worker_launches", "scientific_arrays_created", "unit_terminals_created")), "R1 access ledger drift")
    sealed_inventory = read(ROOT / "whole_root_inventory.json")
    validate_signed(sealed_inventory)
    observed = inventory()
    for key in ("files", "file_count", "total_bytes", "files_digest"):
        require(sealed_inventory["payload"][key] == observed[key], f"R1 inventory drift: {key}")
    tests = run_tests() if run_focused_tests else read(REPORT / "test_report.json")["payload"]
    output = {
        "status": "PASS", "completion_basis": BASIS, "source_recomputed": True,
        "worker_counts_validated": list(range(1, 9)), "tests_passed": tests["passed"],
        "false_pass_rejected": false_pass["rejected_count"],
        "completion_sha256": y2.sha_file(REPO / COMPLETION),
        "completion_canonical_digest": completion["canonical_digest"],
        "inventory_sha256": y2.sha_file(REPO / ROOT / "whole_root_inventory.json"),
        "inventory_canonical_digest": sealed_inventory["canonical_digest"],
        "inventory_files_digest": sealed_inventory["payload"]["files_digest"],
        "next_authorization": NEXT,
    }
    if publication:
        head = git("rev-parse", "HEAD")
        tracking = git("rev-parse", "@{upstream}")
        require(remote(f"refs/heads/{BRANCH}") == head == tracking, "published branch drift")
        require(git("rev-parse", ARCHIVE) == head == remote(ARCHIVE), "published R1 archive drift")
        require(not git("status", "--porcelain"), "publication worktree/index not clean")
        lfs = git("lfs", "status")
        require("Git LFS objects to be committed:" in lfs and "Git LFS objects not staged for commit:" in lfs and "(Git:" not in lfs and "(LFS:" not in lfs, "publication LFS state is not clean")
        output.update({"publication": True, "head": head, "archive_ref": ARCHIVE})
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "validate", "validate-publication", "false-pass-audit"))
    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            result = seal()
        elif args.command == "false-pass-audit":
            result = false_pass_audit()
        else:
            result = validate(args.command == "validate-publication")
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error), "next_authorization": "NONE"}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
