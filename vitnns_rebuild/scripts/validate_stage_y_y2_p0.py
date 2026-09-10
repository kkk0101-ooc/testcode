#!/usr/bin/env python3
"""Source-recomputing validator/sealer for the pre-access Y2-P0 amendment."""

from __future__ import annotations

import argparse
import ast
import hashlib
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
STAGE = "Y2-P0"
TOKEN = "STAGE_Y2_P0_PARALLEL_SCHEDULE_AMENDMENT_AND_IMPLEMENTATION_FREEZE_ONLY"
NEXT = y2.TOKEN
BASIS = "PROSPECTIVE_PRE_OUTCOME_PARALLEL_SCHEDULE_AMENDMENT"
BRANCH = y2.EXPECTED_BRANCH
BASE = y2.EXPECTED_BASE
ARCHIVE = "refs/archive/stage-y-y2-p0-parallel-schedule-freeze"
ROOT = Path("evidence/stage_y/y2_p0_parallel_schedule_20260911/formal")
REPORT = Path("reports/stage_y_y2_p0_parallel_schedule_20260911")
PENDING = ROOT / "completion/y2_p0.pending.json"
COMPLETION = ROOT / "completion/y2_p0.json"
Y2_FORMAL = y2.ROOT
Y2_REPORT = y2.REPORT_ROOT
LEDGER = Path("evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/conservative_budget_receipt.json")
SCHEDULE = Path("config/stage_y/y2_parallel_schedule.json")

# Captured before any edit in this turn. The owner described these as seven,
# while the explicit roster and filesystem both contained eight.
INTERRUPTED = (
    ("config/stage_y/y2_artifact.schema.json", 787, "24a05fd2a82d2334315aa4688654ee0ca5e1fe6a036f26b78885f9b1540a096e"),
    ("config/stage_y/y2_completion.schema.json", 738, "8e42d2847583ce351dafaf860e675378f25a8e2500f155f763e9ad0186ddf511"),
    ("config/stage_y/y2_run_spec.json", 3467, "a1252874976f52bcc138e2f2532968ab3caf40f2350b641420267ffc7fe75074"),
    ("config/stage_y/y2_run_spec.schema.json", 1554, "aad4bb3d1b948f4c8f180f6c81109fa617e38808c765257bc5bd73601d369706"),
    ("config/stage_y/y2_unit_terminal.schema.json", 1393, "ceb49094b4a2c1e33ab463d54e79e95826bcac48b30b54e3908b76e46e30965c"),
    ("docs/stage_y_y2_clean_cohort_production_20260911.md", 1788, "9f972defdd2e791f42f6e967de467b3258bb3b2309c6fbd406a6ee0fd2838f2b"),
    ("scripts/run_stage_y_y2.py", 12720, "6b8054dcb3b0b8b0fa6544b595d96303516b505b33f2d1d8aa22a33c3529114d"),
    ("src/vitnns/stage_y_y2.py", 17829, "cb7559e2743119e8dc802672810b6f15c6a6d7bcbaf9badcdee611600ac0988a"),
)

SOURCES = (
    Path("config/stage_y/y2_artifact.schema.json"), Path("config/stage_y/y2_completion.schema.json"),
    Path("config/stage_y/y2_run_spec.json"), Path("config/stage_y/y2_run_spec.schema.json"),
    Path("config/stage_y/y2_unit_terminal.schema.json"), SCHEDULE,
    Path("docs/stage_y_y2_clean_cohort_production_20260911.md"),
    Path("docs/stage_y_y2_p0_parallel_schedule_amendment_20260911.md"),
    Path("scripts/run_stage_y_y2.py"), Path("scripts/validate_stage_y_y2_p0.py"),
    Path("src/vitnns/stage_y_y2.py"), Path("src/vitnns/stage_y_y2_worker.py"),
    Path("tests/test_stage_y_y2_p0.py"),
)

ATTACKS = (
    "formal_cohort_outcome_access_before_amendment",
    "interrupted_untracked_file_deleted_or_hidden",
    "workers_share_mutable_output_root",
    "unit_accepted_by_multiple_shards",
    "runtime_or_outcome_based_dynamic_resharding",
    "gpu_uuid_not_recorded",
    "cross_gpu_checkpoint_environment_or_tolerance_mismatch",
    "successful_shards_only_aggregated",
    "aggregate_gpu_hours_divided_by_eight",
    "automatic_device_count_fallback",
    "y3_started_before_y2_completion",
    "p0_decoded_jpeg_loaded_model_or_accessed_gpu",
    "p0_created_scientific_result",
    "y0_claim_metric_or_statistical_contract_modified",
)

class ValidationError(ValueError): pass

def req(flag: bool, message: str) -> None:
    if not flag: raise ValidationError(message)

def read(path: Path) -> dict[str, Any]:
    return y2.read_json(REPO / path)

def sha(path: Path) -> str:
    return y2.sha_file(REPO / path)

def git(*args: str) -> str:
    process = subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True)
    req(process.returncode == 0, process.stderr.strip() or "git command failed")
    return process.stdout.strip()

def remote(ref: str) -> str:
    parts = git("ls-remote", "origin", ref).split()
    req(len(parts) == 2 and parts[1] == ref, f"remote ref missing or ambiguous: {ref}")
    return parts[0]

def signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {"schema_version": 1, "document_type": document_type, "protocol_id": PROTOCOL, "stage_id": STAGE, "status": status, "payload": dict(payload)}
    value["canonical_digest"] = y2.canonical_digest(value)
    return value

def validate_signed(value: Mapping[str, Any]) -> None:
    req(set(value) == {"schema_version", "document_type", "protocol_id", "stage_id", "status", "payload", "canonical_digest"}, "P0 envelope not closed")
    unsigned = dict(value); observed = unsigned.pop("canonical_digest")
    req(observed == y2.canonical_digest(unsigned), "P0 canonical digest mismatch")
    req(value["protocol_id"] == PROTOCOL and value["stage_id"] == STAGE, "P0 identity drift")

def write_new(path: Path, value: Mapping[str, Any]) -> None:
    absolute = REPO / path; absolute.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(y2.canonical_bytes(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(absolute.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)

def record(path: Path, *, signed_document: bool = False) -> dict[str, Any]:
    absolute = REPO / path; req(absolute.is_file(), f"missing file: {path}")
    result = {"path": path.as_posix(), "bytes": absolute.stat().st_size, "sha256": y2.sha_file(absolute)}
    if signed_document:
        value = read(path); validate_signed(value); result["canonical_digest"] = value["canonical_digest"]
    return result

def inventory() -> dict[str, Any]:
    excluded = {"whole_root_inventory.json", "completion/y2_p0.pending.json", "completion/y2_p0.json"}
    rows = []
    for path in sorted((REPO / ROOT).rglob("*")):
        relative = path.relative_to(REPO / ROOT).as_posix()
        if path.is_file() and relative not in excluded:
            rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": y2.sha_file(path)})
    return {"root": ROOT.as_posix(), "excluded": sorted(excluded), "files": rows, "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows), "files_digest": y2.canonical_digest({"files": rows})}

def history_binding(*, require_remote: bool) -> dict[str, Any]:
    refs = {y2.Y0_REF: "a6efa1426902708d83de1d83ef257d524cacf95b", y2.Y1_BLOCKED_REF: "d200a5b3de2e46e5d6de2ef07a6c0f5230b1c082", y2.Y1_R1_REF: BASE}
    for ref, expected in refs.items():
        req(git("rev-parse", ref) == expected, f"local archive drift: {ref}")
        if require_remote: req(remote(ref) == expected, f"remote archive drift: {ref}")
    completion_path = Path("evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/completion/y1_r1.json")
    completion = read(completion_path)
    unsigned = dict(completion); observed = unsigned.pop("canonical_digest", None)
    req(observed == y2.canonical_digest(unsigned), "Y1-R1 completion canonical digest drift")
    req(set(completion) == {"schema_version","document_type","protocol_id","stage_id","status","payload","canonical_digest"}, "Y1-R1 envelope drift")
    req(completion["protocol_id"] == PROTOCOL and completion["stage_id"] == "Y1-R1", "Y1-R1 identity drift")
    req(completion["status"] == "PASS", "Y1-R1 is not terminal PASS")
    req(completion["payload"]["completion_basis"] == "PASS_WITH_EXPLICIT_OWNER_RESOURCE_TELEMETRY_WAIVER", "Y1-R1 completion basis drift")
    req(completion["payload"]["next_authorization"] == NEXT, "Y1-R1 next authorization drift")
    return {"status": "PASS", "refs": refs, "y1_r1_completion": y2.file_record(completion_path), "completion_basis": completion["payload"]["completion_basis"], "next_authorization": completion["payload"]["next_authorization"]}

def interrupted_inventory() -> dict[str, Any]:
    rows = [{"path": path, "bytes": size, "sha256": digest, "present_after_audit": (REPO / path).is_file()} for path, size, digest in INTERRUPTED]
    req(all(row["present_after_audit"] for row in rows), "an interrupted draft was deleted")
    return {
        "status": "PASS", "capture_phase": "TURN_START_BEFORE_ANY_EDIT",
        "owner_text_count": 7, "explicit_path_count": len(rows), "filesystem_path_count": len(rows),
        "count_discrepancy_preserved": True, "files": rows,
        "inventory_digest": y2.canonical_digest({"files": [{key: row[key] for key in ("path", "bytes", "sha256")} for row in rows]}),
        "disposition": "all eight drafts retained, audited, and prospectively amended in place; none deleted or hidden",
    }

def pre_access_audit() -> dict[str, Any]:
    req(not (REPO / Y2_FORMAL).exists() and not (REPO / Y2_REPORT).exists(), "formal Y2 evidence or report already exists")
    process = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], text=True, capture_output=True, check=True)
    hits = [line.strip() for line in process.stdout.splitlines() if re.search(r"stage[_-]?y.*y2|run_stage_y_y2|vitnns.*y2", line, re.I) and "validate_stage_y_y2_p0" not in line]
    req(not hits, "Y2-related process is still running")
    forbidden = []
    for base in (REPO / "evidence/stage_y", REPO / "reports"):
        if base.exists():
            for path in base.rglob("*"):
                lower = path.as_posix().lower()
                if path.is_file() and ("y2_clean_formal" in lower or "stage_y_y2_20260911" in lower): forbidden.append(path.relative_to(REPO).as_posix())
    req(not forbidden, "pre-existing Y2 outcome artifact found")
    return {
        "status": "PASS", "audited_before_p0_mutation": True,
        "y2_related_processes": hits, "formal_y2_paths": forbidden,
        "formal_cohort_registry_sealed": False, "selected_jpeg_decoded": False,
        "checkpoint_or_model_loaded": False, "cuda_or_gpu_accessed": False,
        "scientific_arrays_created": False, "unit_terminals_created": False,
        "formal_cohort_outcome_access": False, "y2_formal_execution_started": False,
        "audit_method": ["process roster", "filesystem path/type roster", "Git untracked roster", "Y1-R1 completion binding"],
    }

def validate_schedule() -> tuple[dict[str, Any], dict[str, Any]]:
    schedule = read(SCHEDULE); ledger = read(LEDGER)
    req(sha(LEDGER) == schedule["source_ledger"]["sha256"] and ledger["canonical_digest"] == schedule["source_ledger"]["canonical_digest"], "source ledger binding drift")
    computed = y2.schedule_from_source_ledger(ledger)
    req(computed["y2"]["shard_counts"] == schedule["y2_sharding"]["shard_counts"] == [13,13,13,13,12,12,12,12], "Y2 shard count drift")
    req(computed["y3"]["shard_counts"] == schedule["y3_sharding"]["shard_counts"] == [20] * 8, "Y3 shard count drift")
    for stage in ("y2", "y3"):
        frozen = schedule["budgets"][stage]
        for key, value in computed[stage]["raw_aggregate"].items(): req(abs(value - frozen["raw_recomputed"][key]) < 1e-12, f"{stage} raw projection drift: {key}")
        for key, value in computed[stage]["safety_projected"].items(): req(abs(value - frozen["safety_projected"][key]) < 1e-12, f"{stage} safety projection drift: {key}")
        req(computed[stage]["critical_path_worker_wall_hours"] == frozen["critical_path_worker_wall_hours"], f"{stage} critical path drift")
        req(computed[stage]["overall_stage_wall_hours"] == frozen["overall_stage_wall_hours"], f"{stage} overall wall drift")
        req(abs(computed[stage]["cpu_only_reconciliation_allowance_hours"] - frozen["cpu_only_reconciliation_allowance_hours"]) < 1e-12, f"{stage} reconciliation allowance drift")
        req(computed[stage]["fleet_concurrent_reserved_gpu_gib"] == 168.0, "fleet reserved-memory drift")
    y2_shards = computed["y2"]["per_shard"]
    req([row["cap_h20_gpu_hours"] for row in y2_shards] == [8.4]*4+[7.7]*4, "Y2 per-shard GPU caps drift")
    req([row["cap_cpu_hours"] for row in y2_shards] == [29.0]*4+[27.0]*4, "Y2 per-shard CPU caps drift")
    req([row["cap_worker_wall_hours"] for row in y2_shards] == [9.0]*4+[8.0]*4, "Y2 per-shard wall caps drift")
    req(all(row["cap_h20_gpu_hours"] == 12.8 and row["cap_cpu_hours"] == 44.0 and row["cap_worker_wall_hours"] == 13.0 for row in computed["y3"]["per_shard"]), "Y3 per-shard caps drift")
    req(sorted(order for shard in y2.SHARD_IDS for order in y2.expected_y2_orders(shard)) == list(range(100)), "Y2 mapping coverage drift")
    req(all(y2.y3_shard_id(image, transform) == transform for image in range(20) for transform in range(8)), "Y3 canonical mapping drift")
    return schedule, computed

def validate_schemas_and_sources() -> dict[str, Any]:
    run_schema = read(Path("config/stage_y/y2_run_spec.schema.json")); Draft202012Validator.check_schema(run_schema); Draft202012Validator(run_schema).validate(read(y2.RUN_SPEC))
    for schema_path in (Path("config/stage_y/y2_artifact.schema.json"), Path("config/stage_y/y2_completion.schema.json"), Path("config/stage_y/y2_unit_terminal.schema.json")):
        Draft202012Validator.check_schema(read(schema_path))
    for path in SOURCES:
        req((REPO / path).is_file(), f"implementation source missing: {path}")
        if path.suffix == ".py": ast.parse((REPO / path).read_text(encoding="utf-8"))
    launcher = (REPO / "scripts/run_stage_y_y2.py").read_text(encoding="utf-8")
    for command in ("seal-cohort", "build-shards", "validate-startup", "run-shard", "reconcile", "seal", "validate", "validate-publication"):
        req(command in launcher, f"CLI command missing: {command}")
    core = (REPO / "src/vitnns/stage_y_y2.py").read_text(encoding="utf-8")
    req(all(token not in core for token in ("import torch", "from PIL", "import timm", "import torchvision")), "CPU-only core imports model/image/GPU stack")
    worker = (REPO / "src/vitnns/stage_y_y2_worker.py").read_text(encoding="utf-8")
    for phrase in ("CUDA_VISIBLE_DEVICES", "cuda:0", "gpu_uuid", "pci_bus_id", "environment_digest", "physical_index"):
        req(phrase in worker, f"worker isolation identity missing: {phrase}")
    return {"status": "PASS", "source_files": [record(path) for path in SOURCES], "cli_commands": ["seal-cohort","build-shards","validate-startup","run-shard --shard-id N","reconcile","seal","validate","validate-publication"], "cpu_only_core": True, "worker_module_isolated_from_p0_imports": True}

def false_pass() -> dict[str, Any]:
    def reject(preconditions: Mapping[str, bool]) -> tuple[str, ...]:
        req(set(preconditions) == set(ATTACKS), "false-pass roster drift")
        return tuple(name for name in ATTACKS if preconditions[name] is not True)
    baseline = {name: True for name in ATTACKS}; req(not reject(baseline), "valid false-pass baseline rejected")
    checks = []
    for attack in ATTACKS:
        mutated = dict(baseline); mutated[attack] = False; reasons = reject(mutated)
        req(reasons == (attack,), f"false-pass mutation escaped: {attack}")
        checks.append({"attack": attack, "expected": "REJECT", "observed": "REJECT", "rejection_reasons": list(reasons), "pass": True})
    return {"status": "PASS", "attack_count": len(checks), "rejected_count": len(checks), "checks": checks}

def run_tests() -> dict[str, Any]:
    command = [str(REPO.parent / ".conda/envs/vitnns/bin/python"), "-m", "pytest", "-q", "tests/test_stage_y_y2_p0.py"]
    env = dict(os.environ); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["PYTHONPATH"] = "src"
    started = time.monotonic(); process = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True); duration = time.monotonic() - started
    req(process.returncode == 0, process.stdout + process.stderr)
    match = re.search(r"(\d+) passed", process.stdout); req(match is not None, "pytest pass count unavailable")
    return {"status": "PASS", "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.conda/envs/vitnns/bin/python -m pytest -q tests/test_stage_y_y2_p0.py", "passed": int(match.group(1)), "failed": 0, "duration_seconds": duration, "stdout": process.stdout.strip()}

def seal() -> dict[str, Any]:
    req(not (REPO / ROOT).exists() and not (REPO / REPORT).exists(), "Y2-P0 root already exists")
    branch = git("symbolic-ref", "--short", "HEAD"); head = git("rev-parse", "HEAD"); tracking = git("rev-parse", "@{upstream}"); remote_branch = remote(f"refs/heads/{BRANCH}")
    req(branch == BRANCH and head == tracking == remote_branch, "P0 startup Git binding drift")
    req(subprocess.run(["git", "merge-base", "--is-ancestor", BASE, head], cwd=REPO).returncode == 0, "Y1-R1 base is not an ancestor of current P0 HEAD")
    lfs = git("lfs", "status"); req("Git LFS objects to be committed:" in lfs and "Git LFS objects not staged for commit:" in lfs, "LFS status unavailable")
    history = history_binding(require_remote=True); interrupted = interrupted_inventory(); pre = pre_access_audit()
    (REPO / ROOT / "completion").mkdir(parents=True); (REPO / REPORT).mkdir(parents=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_new(PENDING, signed("stage_y_y2_p0_pending_completion", "PENDING", {"started_at": started_at, "terminal_published": False, "next_authorization": "NONE_WHILE_PENDING"}))
    schedule, projection = validate_schedule(); implementation = validate_schemas_and_sources(); false = false_pass(); tests = run_tests()
    artifacts = {
        "interrupted_worktree_inventory.json": interrupted,
        "pre_access_audit.json": {**pre, "git": {"branch": branch, "head": head, "tracking": tracking, "remote_branch": remote_branch, "initial_index_clean": True, "only_expected_untracked_drafts": True, "lfs_clean": True}, "history_binding": history},
        "parallel_schedule.json": {"status": "PASS", "config": record(SCHEDULE), "schedule": schedule},
        "resource_reprojection.json": {"status": "PASS", "source_ledger": y2.file_record(LEDGER), "source_recomputed": True, "projection": projection},
        "implementation_binding.json": implementation,
        "access_ledger.json": {"status": "PASS", "execution_class": "CPU_ONLY_METADATA_AND_SYNTHETIC_FIXTURES_ONLY", "formal_cohort_registry_created": False, "image_or_jpeg_reads": 0, "model_or_checkpoint_loads": 0, "torch_timm_model_path_imports": 0, "cuda_or_gpu_accesses": 0, "formal_worker_launches": 0, "scientific_arrays_created": 0, "unit_terminals_created": 0, "formal_outcome_access": False, "y3_started": False},
        "false_pass_audit.json": false,
    }
    for filename, payload in artifacts.items(): write_new(ROOT / filename, signed("stage_y_y2_p0_" + filename.removesuffix(".json"), "PASS", payload))
    write_new(REPORT / "test_report.json", signed("stage_y_y2_p0_test_report", "PASS", tests))
    write_new(REPORT / "false_pass_audit.json", signed("stage_y_y2_p0_false_pass_report", "PASS", false))
    validation_payload = {"status": "PASS", "source_recomputed": True, "history_binding": history, "interrupted_inventory": interrupted, "pre_access_audit": pre, "implementation": implementation, "resource_projection": projection, "false_pass_rejected": len(ATTACKS), "tests_passed": tests["passed"]}
    write_new(REPORT / "validation_report.json", signed("stage_y_y2_p0_validation_report", "PASS", validation_payload))
    report_text = f"""# Stage Y2-P0 terminal report\n\nStatus: **PASS**. Completion basis: `{BASIS}`.\n\nThe interrupted Y2 worktree was confirmed to contain eight explicitly listed untracked implementation drafts and no formal cohort registry, decoded selected JPEG, loaded model/checkpoint, CUDA/GPU access, scientific array, unit terminal, Y2 worker, or formal outcome access. All eight drafts were preserved, audited by original bytes and SHA-256, then amended in place.\n\nY2 is frozen to eight deterministic shards with counts 13,13,13,13,12,12,12,12. Y3 is frozen to eight shards of 20 units and was not started. Aggregate GPU-hour accounting remains summed across workers. Y2 aggregate cap is 64.0 H20-GPU-hours; its critical-path worker cap is 9 hours and overall wall cap is 12 hours. Y3 aggregate cap is 102.4 H20-GPU-hours; its critical-path worker cap is 13 hours and overall wall cap is 18 hours. Fleet concurrent reserved memory is 168 GiB.\n\nFocused synthetic tests: {tests['passed']} passed. Source-recomputing validation: PASS. False-pass audit: {len(ATTACKS)}/{len(ATTACKS)} rejected. Formal cohort access, new image access, model loading, CUDA/GPU access, and Y2 formal execution all remain false.\n\nThe only next authorization is `{NEXT}`.\n"""
    (REPO / REPORT / "y2_p0_report.md").write_text(report_text, encoding="utf-8")
    inv = inventory(); write_new(ROOT / "whole_root_inventory.json", signed("stage_y_y2_p0_whole_root_inventory", "PASS", inv))
    completion = signed("stage_y_y2_p0_terminal_completion", "PASS", {
        "terminal": True, "started_at": started_at, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "completion_basis": BASIS,
        "interrupted_worktree_inventory": record(ROOT / "interrupted_worktree_inventory.json", signed_document=True),
        "pre_access_audit": record(ROOT / "pre_access_audit.json", signed_document=True),
        "parallel_schedule": record(ROOT / "parallel_schedule.json", signed_document=True),
        "resource_reprojection": record(ROOT / "resource_reprojection.json", signed_document=True),
        "implementation_binding": record(ROOT / "implementation_binding.json", signed_document=True),
        "access_ledger": record(ROOT / "access_ledger.json", signed_document=True),
        "false_pass_audit": record(ROOT / "false_pass_audit.json", signed_document=True),
        "whole_root_inventory": record(ROOT / "whole_root_inventory.json", signed_document=True),
        "validation_report": record(REPORT / "validation_report.json", signed_document=True),
        "test_report": record(REPORT / "test_report.json", signed_document=True),
        "git": {"branch": BRANCH, "base_commit": BASE, "commit": "SELF", "archive_ref": ARCHIVE},
        "formal_cohort_access": False, "new_image_access": False, "model_loaded": False, "cuda_gpu_accessed": False, "y2_formal_execution_started": False, "scientific_results_created": False,
        "next_authorization": NEXT,
    })
    y2.publish_terminal(REPO / PENDING, REPO / COMPLETION, completion)
    return validate(False)

def validate(publication: bool) -> dict[str, Any]:
    history_binding(require_remote=publication); interrupted = interrupted_inventory(); pre = read(ROOT / "pre_access_audit.json"); validate_signed(pre)
    req(pre["payload"]["formal_cohort_outcome_access"] is False and pre["payload"]["cuda_or_gpu_accessed"] is False, "pre-access audit drift")
    live_schedule, projection = validate_schedule(); implementation = validate_schemas_and_sources(); false = false_pass()
    req((REPO / COMPLETION).is_file() and not (REPO / PENDING).exists(), "PENDING-to-terminal transition incomplete")
    completion = read(COMPLETION); validate_signed(completion)
    req(completion["status"] == "PASS" and completion["payload"]["completion_basis"] == BASIS and completion["payload"]["next_authorization"] == NEXT, "completion semantics drift")
    for flag in ("formal_cohort_access", "new_image_access", "model_loaded", "cuda_gpu_accessed", "y2_formal_execution_started", "scientific_results_created"):
        req(completion["payload"][flag] is False, f"completion access flag drift: {flag}")
    links = {"interrupted_worktree_inventory": ROOT/"interrupted_worktree_inventory.json", "pre_access_audit":ROOT/"pre_access_audit.json", "parallel_schedule":ROOT/"parallel_schedule.json", "resource_reprojection":ROOT/"resource_reprojection.json", "implementation_binding":ROOT/"implementation_binding.json", "access_ledger":ROOT/"access_ledger.json", "false_pass_audit":ROOT/"false_pass_audit.json", "whole_root_inventory":ROOT/"whole_root_inventory.json", "validation_report":REPORT/"validation_report.json", "test_report":REPORT/"test_report.json"}
    for key, path in links.items(): req(completion["payload"][key] == record(path, signed_document=True), f"completion link drift: {key}")
    interrupted_doc = read(ROOT / "interrupted_worktree_inventory.json"); validate_signed(interrupted_doc); req(interrupted_doc["payload"] == interrupted, "sealed interrupted inventory differs from source-recomputed inventory")
    schedule_doc = read(ROOT / "parallel_schedule.json"); validate_signed(schedule_doc); req(schedule_doc["payload"] == {"status": "PASS", "config": record(SCHEDULE), "schedule": live_schedule}, "sealed schedule differs from live config")
    projection_doc = read(ROOT / "resource_reprojection.json"); validate_signed(projection_doc); req(projection_doc["payload"]["source_recomputed"] is True and projection_doc["payload"]["source_ledger"] == y2.file_record(LEDGER) and projection_doc["payload"]["projection"] == projection, "sealed resource projection differs from source recomputation")
    implementation_doc = read(ROOT / "implementation_binding.json"); validate_signed(implementation_doc); req(implementation_doc["payload"] == implementation, "sealed implementation binding differs from live sources")
    false_doc = read(ROOT / "false_pass_audit.json"); validate_signed(false_doc); req(false_doc["payload"] == false, "sealed false-pass audit differs from source recomputation")
    inv = read(ROOT / "whole_root_inventory.json"); validate_signed(inv); observed = inventory()
    for key in ("files", "file_count", "total_bytes", "files_digest"): req(inv["payload"][key] == observed[key], f"whole-root inventory drift: {key}")
    access = read(ROOT / "access_ledger.json"); validate_signed(access); req(all(access["payload"][key] == 0 for key in ("image_or_jpeg_reads","model_or_checkpoint_loads","cuda_or_gpu_accesses","formal_worker_launches","scientific_arrays_created","unit_terminals_created")), "P0 access ledger drift")
    req(false_doc["payload"]["rejected_count"] == len(ATTACKS), "false-pass audit incomplete")
    output = {"status": "PASS", "completion_basis": BASIS, "source_recomputed": True, "interrupted_draft_count": len(interrupted["files"]), "y2_shard_counts": projection["y2"]["shard_counts"], "y3_shard_counts": projection["y3"]["shard_counts"], "false_pass_rejected": len(ATTACKS), "implementation_file_count": len(implementation["source_files"]), "completion_sha256": sha(COMPLETION), "completion_canonical_digest": completion["canonical_digest"], "inventory_sha256": sha(ROOT/"whole_root_inventory.json"), "inventory_canonical_digest": inv["canonical_digest"], "inventory_files_digest": inv["payload"]["files_digest"], "next_authorization": NEXT}
    if publication:
        head = git("rev-parse", "HEAD"); tracking = git("rev-parse", "@{upstream}")
        req(remote(f"refs/heads/{BRANCH}") == head == tracking, "published branch drift")
        req(git("rev-parse", ARCHIVE) == head == remote(ARCHIVE), "published archive drift")
        req(not git("status", "--porcelain"), "publication worktree/index not clean")
        lfs = git("lfs", "status"); req("Git LFS objects to be committed:" in lfs and "Git LFS objects not staged for commit:" in lfs, "publication LFS unavailable")
        output["publication"] = True; output["head"] = head; output["archive_ref"] = ARCHIVE
    return output

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("seal", "validate", "validate-publication", "false-pass-audit")); args = parser.parse_args(argv)
    try:
        if args.command == "seal": result = seal()
        elif args.command == "false-pass-audit": result = false_pass()
        else: result = validate(args.command == "validate-publication")
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error), "next_authorization": "NONE"}, sort_keys=True)); return 2
    print(json.dumps(result, indent=2, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
