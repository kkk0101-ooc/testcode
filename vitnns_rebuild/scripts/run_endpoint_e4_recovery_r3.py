#!/usr/bin/env python3
"""Run E4 R3 as a validated 768-result rebind plus 32-task local rerun."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GIT_ROOT = REPO.parent
PYTHON = GIT_ROOT / ".conda/envs/vitnns/bin/python"
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_e4_intervention_recovery_r3.json"
SOURCE = REPO / "reports/nns_vit_endpoint_e4_intervention_r2_20260918"
RUNTIME = GIT_ROOT / ".runtime/nns_vit_endpoint_e4_intervention_r3"
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import run_endpoint_e4_intervention as e4_runner  # noqa: E402
from run_endpoint_e4_intervention import (  # noqa: E402
    aggregate_descriptive, gpu_snapshot, launch, lock_gpus, remote,
)
from validate_endpoint_e4_recovery_r3 import validate_result  # noqa: E402
from vitnns.endpoint_e2_atlas import digest, file_sha256, linear_percentile, write_json_new  # noqa: E402
from vitnns.endpoint_e4_intervention import MAIN_FAMILIES, validate_recovery_r3_run_spec  # noqa: E402
from vitnns.endpoint_ep1_queue import audit_coverage, create_queue  # noqa: E402
e4_runner.RUNTIME = RUNTIME

PASS_STATUS = "ENDPOINT_E4_R3_PASS_EXECUTION_CONDITION_SCALED_JVP_AND_FINITE_REPLACEMENT"
ALLOWED_UNTRACKED = {
    "config/nns_vit_cut_intervention/p0_c.json",
    "manifests/nns_vit_cut_intervention/p0_c_operation_registry.json",
    "scripts/validate_nns_vit_cut_intervention_p0_c.py",
    "src/vitnns/nns_vit_cut_intervention_p0_c.py",
    "src/vitnns/nns_vit_cut_intervention_p0_c_worker.py",
    "tests/test_nns_vit_cut_intervention_p0_c.py",
}
SOURCE_FILES = [
    "config/nns_vit_endpoint/endpoint_e4_intervention_recovery_r3.json",
    "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE4恢复R3_20260918.md",
    "src/vitnns/endpoint_e4_intervention.py",
    "scripts/run_endpoint_e4_intervention_worker.py",
    "scripts/run_endpoint_e4_recovery_r3.py",
    "scripts/validate_endpoint_e4_recovery_r3.py",
    "scripts/validate_endpoint_e4_recovery_r2_failure.py",
    "scripts/diagnose_endpoint_e4_r2_jvp.py",
    "scripts/validate_endpoint_unified_plan.py",
    "tests/test_endpoint_e4_intervention.py",
]
BINDINGS = {
    "cohort_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json",
    "endpoint_e3_r2_terminal_sha256": "reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json",
    "endpoint_e4_r2_artifact_manifest_sha256": "reports/nns_vit_endpoint_e4_intervention_r2_20260918/artifact_manifest.json",
    "endpoint_e4_r2_coverage_sha256": "reports/nns_vit_endpoint_e4_intervention_r2_20260918/formal_coverage_audit.json",
    "endpoint_e4_r2_diagnostic_sha256": "reports/nns_vit_endpoint_e4_r2_jvp_diagnostic_20260918/diagnostic.json",
    "endpoint_e4_r2_terminal_sha256": "reports/nns_vit_endpoint_e4_intervention_r2_20260918/terminal_receipt.json",
    "endpoint_e4_r2_validation_sha256": "reports/nns_vit_endpoint_e4_intervention_r2_20260918/validation_report.json",
    "intervention_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/intervention_registry.json",
    "materialization_receipt_sha256": "reports/nns_vit_endpoint_e2_atlas_r1_20260918/materialization_receipt.json",
    "model_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
    "statistical_analysis_plan_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json",
    "target_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json",
    "unified_plan_sha256": "docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def source_attempt_resources() -> dict[str, float]:
    with sqlite3.connect(SOURCE / "formal_queue.sqlite3") as connection:
        row = connection.execute(
            "SELECT MIN(claimed_ns),MAX(completed_ns),SUM(completed_ns-claimed_ns) "
            "FROM attempts WHERE completed_ns IS NOT NULL"
        ).fetchone()
    return {
        "wall_seconds": (int(row[1]) - int(row[0])) / 1e9,
        "attempt_gpu_hours": int(row[2]) / 1e9 / 3600,
    }


def validate_source_and_select(spec: dict) -> tuple[list[dict], list[dict], dict]:
    completed = subprocess.run(
        [str(PYTHON), "scripts/validate_endpoint_e4_recovery_r2_failure.py", str(SOURCE)],
        cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), text=True, capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError("R2_SOURCE_VALIDATION_FAILED:" + completed.stderr)
    source_validation = json.loads(completed.stdout)
    diagnostic = load(REPO / "reports/nns_vit_endpoint_e4_r2_jvp_diagnostic_20260918/diagnostic.json")
    if not (diagnostic["status"] == "DIAGNOSTIC_ONLY_NOT_STAGE_RESULT" and
            diagnostic["task_id"] == "vit_b16__009" and
            diagnostic["source_r2_terminal_sha256"] == file_sha256(SOURCE / "terminal_receipt.json") and
            diagnostic["r2_failed_component_count"] == 1 and
            diagnostic["condition_scaled_failed_component_count"] == 0):
        raise RuntimeError("R3_DIAGNOSTIC_BINDING")
    diagnostic_script = "scripts/diagnose_endpoint_e4_r2_jvp.py"
    diagnostic_blob = subprocess.check_output(
        ["git", "show", f"{diagnostic['diagnostic_commit']}:vitnns_rebuild/{diagnostic_script}"], cwd=REPO,
    )
    if hashlib.sha256(diagnostic_blob).hexdigest() != diagnostic["diagnostic_script_sha256"]:
        raise RuntimeError("R3_DIAGNOSTIC_IMPLEMENTATION_BINDING")
    source_manifest = load(SOURCE / "task_manifest.json")
    payloads = {row["task_id"]: row["payload"] for row in source_manifest["tasks"]}
    material = load(REPO / spec["outputs"]["source_materialization"])
    frozen = {
        (model, row["split_order"]): row
        for model in ("vit_b16", "deit_s16") for row in material["records"]
    }
    with sqlite3.connect(SOURCE / "formal_queue.sqlite3") as connection:
        rows = connection.execute(
            "SELECT task_id,state,result_path,result_digest FROM tasks ORDER BY task_id"
        ).fetchall()
    inherited = []; rerun = []
    for task_id, state, result_path, result_digest in rows:
        payload = dict(payloads[task_id]); order = int(payload["split_order"]); model = payload["model_id"]
        if order < 16:
            payload.update({"stage": spec["stage"], "local_contract_mode": "condition_scaled_jvp_r3"})
            rerun.append({"task_id": task_id, "payload": payload})
            continue
        if state != "completed" or not result_path:
            raise RuntimeError("R3_INHERITED_SOURCE_INCOMPLETE:" + task_id)
        source_path = SOURCE / "formal_results" / result_path
        document = load(source_path); signed = dict(document); observed = signed.pop("canonical_digest")
        if digest(signed) != observed or observed != result_digest:
            raise RuntimeError("R3_INHERITED_RESULT_DIGEST:" + task_id)
        result = document["result"]
        if result["model_id"] != model or result["split_order"] != order or result["include_local"]:
            raise RuntimeError("R3_INHERITED_RESULT_SCOPE:" + task_id)
        validate_result(result, frozen[(model, order)], new=False)
        inherited.append({
            "task_id": task_id, "origin": "inherited_from_r2_source_revalidated",
            "source_result_path": source_path.relative_to(REPO).as_posix(),
            "source_result_digest": observed, "source_attempt": document["attempt"],
        })
    inherited.sort(key=lambda row: row["task_id"]); rerun.sort(key=lambda row: row["task_id"])
    if len(inherited) != 768 or len(rerun) != 32:
        raise RuntimeError("R3_PARTITION")
    return inherited, rerun, source_validation


def new_results(queue: Path, root: Path) -> tuple[dict[str, dict], list[dict]]:
    values = {}; index = []
    with sqlite3.connect(queue) as connection:
        rows = connection.execute(
            "SELECT task_id,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id"
        ).fetchall()
    for task_id, state, attempt, relative, result_digest in rows:
        if state != "completed" or not relative:
            raise RuntimeError("R3_NEW_RESULT_INCOMPLETE:" + task_id)
        path = root / "formal_results" / relative; document = load(path)
        signed = dict(document); observed = signed.pop("canonical_digest")
        if digest(signed) != observed or observed != result_digest:
            raise RuntimeError("R3_NEW_RESULT_DIGEST:" + task_id)
        values[task_id] = document["result"]
        index.append({
            "task_id": task_id, "origin": "newly_run_in_r3",
            "source_result_path": path.relative_to(REPO).as_posix(),
            "source_result_digest": observed, "source_attempt": int(attempt),
        })
    return values, index


def all_results(index: list[dict]) -> dict[str, dict]:
    return {row["task_id"]: load(REPO / row["source_result_path"])["result"] for row in index}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--max-gpus", type=int, default=8); args = parser.parse_args()
    spec = load(SPEC_PATH); validate_recovery_r3_run_spec(spec)
    for key, relative in BINDINGS.items():
        if file_sha256(REPO / relative) != spec["predecessors"][key]:
            raise RuntimeError("R3_PREDECESSOR_BINDING:" + key)
    r2 = load(SOURCE / "terminal_receipt.json")
    if r2["status"] != spec["recovery"]["source_status"]:
        raise RuntimeError("R3_SOURCE_AUTHORITY")
    head = git("rev-parse", "HEAD"); branch = git("symbolic-ref", "--short", "HEAD")
    upstream = git("rev-parse", "@{upstream}"); remote_head = remote("refs/heads/" + branch)
    if branch != "stage-d-d3-pass" or not head == upstream == remote_head:
        raise RuntimeError("R3_PRODUCER_NOT_PUBLISHED")
    for relative in SOURCE_FILES:
        if subprocess.run(["git", "cat-file", "-e", f"{head}:vitnns_rebuild/{relative}"], cwd=REPO).returncode:
            raise RuntimeError("R3_SOURCE_NOT_PUBLISHED:" + relative)
    predecessor_validation = []
    for command in (
        [str(PYTHON), "scripts/validate_endpoint_unified_plan.py"],
        [str(PYTHON), "scripts/validate_endpoint_primary_track_amendment.py"],
        [str(PYTHON), "scripts/validate_endpoint_ep1.py"],
        [str(PYTHON), "scripts/validate_endpoint_e2_atlas.py"],
        [str(PYTHON), "scripts/validate_endpoint_e3_closure.py", "reports/nns_vit_endpoint_e3_closure_r1_20260918"],
        [str(PYTHON), "scripts/validate_endpoint_e4_recovery_r2_failure.py"],
    ):
        completed = subprocess.run(command, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError("R3_PREDECESSOR_VALIDATION_FAILED:" + completed.stderr)
        predecessor_validation.extend(completed.stdout.splitlines())
    changed = []
    prefix = REPO.relative_to(GIT_ROOT).as_posix() + "/"
    for line in git("status", "--porcelain", "--untracked-files=all").splitlines():
        path = line[3:].split(" -> ")[-1]; changed.append(path[len(prefix):] if path.startswith(prefix) else path)
    if set(changed) - ALLOWED_UNTRACKED:
        raise RuntimeError("R3_UNEXPLAINED_WORKTREE:" + str(sorted(set(changed) - ALLOWED_UNTRACKED)))
    root = REPO / spec["outputs"]["formal_root"]
    if root.exists():
        raise RuntimeError("R3_ROOT_EXISTS")
    if shutil.disk_usage(REPO).free < 50 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_DISK")
    inherited, rerun, source_validation = validate_source_and_select(spec)
    root.mkdir(parents=True); RUNTIME.mkdir(parents=True, exist_ok=True); stage_started = time.monotonic()
    write_json_new(root / "source_revalidation.json", {
        "status": "PASS_R2_SOURCE_REVALIDATED_FOR_R3",
        "source_terminal_sha256": file_sha256(SOURCE / "terminal_receipt.json"),
        "source_validation": source_validation, "inherited_count": len(inherited),
        "rerun_count": len(rerun), "inherited_index_digest": digest(inherited),
    })
    queue = root / "formal_queue.sqlite3"; queue_identity = create_queue(queue, rerun)
    write_json_new(root / "task_manifest.json", {
        "stage": spec["stage"], "tasks": rerun, "task_count": 32,
        "full_union_task_count": 800, "inherited_task_count": 768,
        "local_task_count": 32, "references": ["normalized_zero", "blur_rgb"],
        "queue_identity": queue_identity, "manifest_digest": digest(rerun),
        "seed_rule": "SHA256(namespace|task_id) first8 bytes",
    })
    snapshot = gpu_snapshot(); eligible = [row for row in snapshot if row["eligible"]]
    candidates = [row for row in eligible if not row["existing_compute_processes"]] + [row for row in eligible if row["existing_compute_processes"]]
    selected, locks = lock_gpus(candidates, head, root, max(1, min(args.max_gpus, 8)))
    if not selected:
        raise RuntimeError("BLOCKED_INFRASTRUCTURE_NO_LOCKABLE_H20")
    write_json_new(root / "startup_snapshot.json", {
        "hostname": __import__("socket").gethostname(), "producer_commit": head,
        "tracking_commit": upstream, "remote_commit": remote_head, "gpu_snapshot": snapshot,
        "selected_gpus": selected, "free_disk_bytes": shutil.disk_usage(REPO).free,
        "predecessor_validation": predecessor_validation,
        "CUBLAS_WORKSPACE_CONFIG_before_workers": ":4096:8",
        "formal_image_reads": {"evaluation": 32, "calibration": 0, "discovery": 0},
        "existing_pids_recorded": True, "existing_processes_not_terminated": True,
    })
    try:
        exits, formal_seconds = launch(queue, root / "formal_results", selected, root)
    finally:
        for stream in locks:
            fcntl.flock(stream, fcntl.LOCK_UN); stream.close()
    coverage = audit_coverage(queue, root / "formal_results"); write_json_new(root / "formal_coverage_audit.json", coverage)
    if any(row["return_code"] for row in exits) or coverage["status"] != "PASS":
        raise RuntimeError("R3_WORKER_OR_COVERAGE_FAILED")
    new, new_index = new_results(queue, root)
    combined = sorted(inherited + new_index, key=lambda row: row["task_id"])
    if len(combined) != 800 or len({row["task_id"] for row in combined}) != 800:
        raise RuntimeError("R3_COMBINED_UNION")
    write_json_new(root / "combined_result_index.json", {
        "stage": spec["stage"], "records": combined, "record_count": 800,
        "inherited_count": 768, "new_count": 32, "records_digest": digest(combined),
    })
    formal = all_results(combined)
    local_rows = [item for result in new.values() for item in result["local_contract"]["components"]]
    if len(local_rows) != 6144 or not all(row["pass"] for row in local_rows):
        raise RuntimeError("R3_LOCAL_DIRECTION_CONTRACT")
    main_count = sum(sum(c["family"] in MAIN_FAMILIES for c in result["finite_utility"][ref]["components"]) for result in formal.values() for ref in ("normalized_zero", "blur_rgb"))
    secondary_count = sum(sum(c["family"] not in MAIN_FAMILIES for c in result["finite_utility"][ref]["components"]) for result in formal.values() for ref in ("normalized_zero", "blur_rgb"))
    zero_count = sum(len(result["zeroing_secondary"]["components"]) for result in formal.values())
    if (main_count, secondary_count, zero_count) != (249600, 9216, 16896):
        raise RuntimeError("R3_GLOBAL_COUNTS")
    source_resources = source_attempt_resources(); new_gpu_hours = formal_seconds * len(selected) / 3600
    combined_gpu_hours = source_resources["attempt_gpu_hours"] + new_gpu_hours
    combined_wall_seconds = source_resources["wall_seconds"] + formal_seconds
    retained = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    report = {
        "status": PASS_STATUS, "stage": spec["stage"],
        "claim_scope": "EXECUTION_PASS_NOT_FINAL_C4A_CLAIM_DECISION",
        "producer_commit": head, "evaluation_images": 400, "models": spec["models"],
        "image_model_tasks": 800, "inherited_task_count": 768, "new_task_count": 32,
        "r1_failure_preserved": True, "r2_failure_preserved": True,
        "r1_results_inherited": False, "r2_local_results_inherited": False,
        "local_contract_method": "condition_scaled_jvp_r3",
        "local_component_evaluations": len(local_rows), "local_eta_evaluations": 0,
        "maximum_local_directional_prediction_absolute_error": max(row["directional_prediction_absolute_error"] for row in local_rows),
        "maximum_local_error_over_absolute_dot_mass": max(row["directional_prediction_absolute_error"] / max(row["absolute_dot_mass"], 1e-30) for row in local_rows),
        "maximum_noop_replacement_abs": max(result["local_contract"]["noop_max_abs"] for result in new.values()),
        "main_component_reference_evaluations": main_count,
        "secondary_component_reference_evaluations": secondary_count,
        "zeroing_secondary_evaluations": zero_count,
        "descriptive_per_image_spearman_summary": aggregate_descriptive(formal),
        "formal_coverage": coverage, "combined_index_digest": digest(combined),
        "materialization_records_digest": load(REPO / spec["outputs"]["source_materialization"])["records_digest"],
        "latency_seconds_new_tasks": {
            "exclusive_scan_p50": linear_percentile([row["scan_seconds"] for row in new.values()], .5),
            "exclusive_scan_p95": linear_percentile([row["scan_seconds"] for row in new.values()], .95),
            "exclusive_scan_max": max(row["scan_seconds"] for row in new.values()),
        },
        "peak_reserved_gpu_bytes_max_new": max(row["peak_reserved_gpu_bytes"] for row in new.values()),
        "peak_cpu_rss_bytes_max_new": max(row["peak_cpu_rss_bytes"] for row in new.values()),
        "retained_bytes_before_report": retained, "new_formal_wall_seconds": formal_seconds,
        "new_gpu_hours_upper_observed": new_gpu_hours, "source_r2_resources": source_resources,
        "combined_gpu_hours_observed": combined_gpu_hours,
        "combined_wall_seconds_observed": combined_wall_seconds, "formal_workers": len(selected),
        "worker_exits": exits,
        "implementation_bindings": [{"path": path, "sha256": file_sha256(REPO / path), "commit": head} for path in SOURCE_FILES],
        "deferred": {
            "c4a_bootstrap": "ENDPOINT-E6", "c4a_holm": "ENDPOINT-E6",
            "c4a_claim_decisions": "ENDPOINT-E6",
            "c4b_spatial_intervention": "ENDPOINT-E5_REQUIRES_EXPLICIT_USER_REQUEST",
            "path_ig_input_fidelity_stability": "NOT_RUN_NOT_ALLOWED",
        },
        "resource_caps": {"gpu_hours": 136, "wall_hours": 132, "incremental_retained_gib": 56},
        "next_authorization": "ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    if combined_gpu_hours > 136 or combined_wall_seconds > 132 * 3600 or retained > 56 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP")
    write_json_new(root / "validation_report.json", report)
    files = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}]
    manifest = {"files": files, "canonical_digest": digest(files)}; write_json_new(root / "artifact_manifest.json", manifest)
    terminal = {
        "status": PASS_STATUS, "stage": spec["stage"],
        "claim_scope": "EXECUTION_PASS_NOT_FINAL_C4A_CLAIM_DECISION",
        "producer_commit": head, "evaluation_image_count": 400, "image_model_task_count": 800,
        "inherited_task_count": 768, "new_task_count": 32,
        "main_component_reference_evaluation_count": main_count,
        "secondary_component_reference_evaluation_count": secondary_count,
        "local_component_evaluation_count": len(local_rows), "local_eta_evaluation_count": 0,
        "zeroing_secondary_evaluation_count": zero_count, "validation_digest": digest(report),
        "manifest_digest": manifest["canonical_digest"],
        "next_authorization": "ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    write_json_new(root / "terminal_receipt.json", terminal); print(json.dumps(terminal, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
