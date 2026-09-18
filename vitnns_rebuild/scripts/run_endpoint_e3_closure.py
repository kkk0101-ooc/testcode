#!/usr/bin/env python3
"""Orchestrate the published ENDPOINT-E3 structural-closure cohort."""
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
import socket
import sqlite3
import subprocess
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GIT_ROOT = REPO.parent
PYTHON = GIT_ROOT / ".conda/envs/vitnns/bin/python"
EP0 = REPO / "manifests/nns_vit_endpoint/ep0_r1_20260918"
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_e3_closure_run_spec.json"
RUNTIME = GIT_ROOT / ".runtime/nns_vit_endpoint_e3_closure"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e2_atlas import digest, file_sha256, linear_percentile, write_json_new  # noqa: E402
from vitnns.endpoint_e3_closure import CLAIM_DECISION, PASS_STATUS, validate_run_spec  # noqa: E402
from vitnns.endpoint_ep1_queue import audit_coverage, create_queue  # noqa: E402

ALLOWED_UNTRACKED = {
    "config/nns_vit_cut_intervention/p0_c.json",
    "manifests/nns_vit_cut_intervention/p0_c_operation_registry.json",
    "scripts/validate_nns_vit_cut_intervention_p0_c.py",
    "src/vitnns/nns_vit_cut_intervention_p0_c.py",
    "src/vitnns/nns_vit_cut_intervention_p0_c_worker.py",
    "tests/test_nns_vit_cut_intervention_p0_c.py",
}
SOURCE_FILES = [
    "config/nns_vit_endpoint/endpoint_e3_closure_run_spec.json",
    "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE3执行合同_20260918.md",
    "src/vitnns/endpoint_e3_closure.py",
    "scripts/run_endpoint_e3_closure_worker.py",
    "scripts/run_endpoint_e3_closure.py",
    "scripts/validate_endpoint_e3_closure.py",
    "tests/test_endpoint_e3_closure.py",
]
BINDINGS = {
    "cohort_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json",
    "endpoint_e2_source_binding_sha256": "reports/nns_vit_endpoint_e2_atlas_r4_20260918/formal_source_binding.json",
    "endpoint_e2_terminal_sha256": "reports/nns_vit_endpoint_e2_atlas_r4_20260918/terminal_receipt.json",
    "endpoint_e2_validation_sha256": "reports/nns_vit_endpoint_e2_atlas_r4_20260918/validation_report.json",
    "implementation_contract_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/implementation_contract.json",
    "materialization_receipt_sha256": "reports/nns_vit_endpoint_e2_atlas_r1_20260918/materialization_receipt.json",
    "model_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
    "target_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json",
    "unified_plan_sha256": "docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def remote(ref: str) -> str:
    fields = subprocess.check_output(["git", "ls-remote", "origin", ref], cwd=REPO, text=True).split()
    return fields[0] if len(fields) == 2 else ""


def startup() -> tuple[dict, str, list[str]]:
    spec = load(SPEC_PATH)
    validate_run_spec(spec)
    for key, relative in BINDINGS.items():
        if file_sha256(REPO / relative) != spec["predecessors"][key]:
            raise RuntimeError("PREDECESSOR_BINDING:" + key)
    validators = []
    for command in (
        [str(PYTHON), "scripts/validate_endpoint_unified_plan.py"],
        [str(PYTHON), "scripts/validate_endpoint_primary_track_amendment.py"],
        [str(PYTHON), "scripts/validate_endpoint_ep1.py"],
        [str(PYTHON), "scripts/validate_endpoint_e2_atlas.py"],
    ):
        completed = subprocess.run(
            command, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            text=True, capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError("PREDECESSOR_VALIDATION_FAILED:" + completed.stderr)
        validators.extend(completed.stdout.splitlines())
    head = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--short", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    remote_head = remote("refs/heads/" + branch)
    if branch != "stage-d-d3-pass" or not head == upstream == remote_head:
        raise RuntimeError("E3_PRODUCER_NOT_PUBLISHED")
    for relative in SOURCE_FILES:
        if subprocess.run(
            ["git", "cat-file", "-e", f"{head}:vitnns_rebuild/{relative}"], cwd=REPO,
        ).returncode:
            raise RuntimeError("SOURCE_NOT_COMMITTED:" + relative)
    status = git("status", "--porcelain", "--untracked-files=all")
    changed = []
    prefix = REPO.relative_to(GIT_ROOT).as_posix() + "/"
    for line in status.splitlines():
        path = line[3:].split(" -> ")[-1]
        if path.startswith(prefix):
            path = path[len(prefix):]
        changed.append(path)
    if set(changed) - ALLOWED_UNTRACKED:
        raise RuntimeError("UNEXPLAINED_WORKTREE:" + str(sorted(set(changed) - ALLOWED_UNTRACKED)))
    return spec, head, validators


def gpu_snapshot() -> list[dict]:
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.free,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ], text=True)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits",
    ], text=True)
    active: dict[str, list[dict]] = {}
    for line in processes.splitlines():
        if line.strip():
            gpu_uuid, pid, memory = [item.strip() for item in line.split(",")]
            active.setdefault(gpu_uuid, []).append({"pid": int(pid), "used_memory_mib": int(memory)})
    rows = []
    for line in query.splitlines():
        index, gpu_uuid, name, pci, total, free, utilization, driver = [item.strip() for item in line.split(",")]
        rows.append({
            "physical_index": int(index), "gpu_uuid": gpu_uuid, "name": name,
            "pci_bus_id": pci, "memory_total_mib": int(total), "memory_free_mib": int(free),
            "utilization_percent": int(utilization), "driver_version": driver,
            "existing_compute_processes": active.get(gpu_uuid, []),
            "eligible": name == "NVIDIA H20" and int(free) >= 32768 and int(utilization) <= 10,
        })
    return rows


def lock_gpus(candidates: list[dict], head: str, root: Path, maximum: int) -> tuple[list[dict], list[Any]]:
    lock_root = RUNTIME / "gpu_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    locked, streams = [], []
    for gpu in candidates:
        if len(locked) >= maximum:
            break
        stream = (lock_root / f"{gpu['gpu_uuid']}.lock").open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            continue
        stream.seek(0); stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "head": head, "root": str(root)}))
        stream.flush(); os.fsync(stream.fileno())
        locked.append(gpu); streams.append(stream)
    return locked, streams


def launch(queue: Path, results: Path, artifacts: Path, selected: list[dict], root: Path) -> tuple[list[dict], float]:
    processes, streams = [], []
    started = time.monotonic()
    for index, gpu in enumerate(selected):
        worker_id = f"formal-worker-{index:02d}"
        log_path = root / "logs" / f"{worker_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("xb")
        streams.append(stream)
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(gpu["physical_index"]),
            CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(REPO / "src"),
            PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
        )
        command = [
            str(PYTHON), "scripts/run_endpoint_e3_closure_worker.py",
            "--queue", str(queue), "--results", str(results), "--artifacts", str(artifacts),
            "--worker-id", worker_id, "--physical-index", str(gpu["physical_index"]),
            "--gpu-uuid", gpu["gpu_uuid"],
        ]
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((worker_id, gpu, process))
    exits = []
    for (worker_id, gpu, process), stream in zip(processes, streams):
        code = process.wait()
        stream.flush(); os.fsync(stream.fileno()); stream.close()
        exits.append({"worker_id": worker_id, "gpu_uuid": gpu["gpu_uuid"], "return_code": code})
    return exits, time.monotonic() - started


def queue_results(queue: Path, results: Path) -> dict[str, dict]:
    output = {}
    with sqlite3.connect(queue) as connection:
        rows = connection.execute("SELECT task_id,result_path FROM tasks ORDER BY task_id").fetchall()
    for task_id, relative in rows:
        output[task_id] = load(results / relative)["result"]
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-gpus", type=int, default=8)
    args = parser.parse_args()
    spec, head, predecessor_validation = startup()
    root = REPO / spec["outputs"]["formal_root"]
    if root.exists():
        raise RuntimeError("E3_ROOT_EXISTS")
    if shutil.disk_usage(REPO).free < 50 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_DISK")
    root.mkdir(parents=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    stage_started = time.monotonic()
    material = load(REPO / spec["outputs"]["source_materialization"])
    if material["status"] != "PASS_EVALUATION_MATERIALIZATION" or material["authorized_member_count"] != 400:
        raise RuntimeError("MATERIALIZATION_AUTHORITY")
    if material["records_digest"] != digest(material["records"]):
        raise RuntimeError("MATERIALIZATION_DIGEST")
    models = {row["model_id"]: row for row in load(EP0 / "model_registry.json")["models"]}
    ep1 = load(REPO / "reports/nns_vit_endpoint_ep1_r11_20260918/validation_report.json")
    tasks = []
    for model_id in spec["models"]:
        model = models[model_id]
        for row in material["records"]:
            tasks.append({
                "task_id": f"{model_id}__{row['split_order']:03d}",
                "payload": {
                    "model_id": model_id, "timm_name": model["timm_name"],
                    "embedding_dim": model["embedding_dim"],
                    "checkpoint": model["checkpoint"]["path"],
                    "checkpoint_bytes": model["checkpoint"]["bytes"],
                    "checkpoint_sha256": model["checkpoint"]["sha256"],
                    "tensor_digest": ep1["tensor_digests"][model_id],
                    "mean": model["preprocessing"]["mean"], "std": model["preprocessing"]["std"],
                    "sample_id": row["sample_id"], "split_order": row["split_order"],
                    "class_id": row["class_id"], "wnid": row["wnid"],
                    "cache_path": row["cache_path"], "image_bytes": row["bytes"],
                    "image_sha256": row["sha256"],
                    "target_registry": "manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json",
                    "target_registry_sha256": spec["predecessors"]["target_registry_sha256"],
                    "retain_raw": row["split_order"] in spec["retention"]["raw_component_maps_evaluation_orders"],
                },
            })
    tasks.sort(key=lambda row: row["task_id"])
    if len(tasks) != 800:
        raise RuntimeError("TASK_COUNT")
    queue = root / "formal_queue.sqlite3"
    queue_identity = create_queue(queue, tasks)
    write_json_new(root / "task_manifest.json", {
        "stage": "ENDPOINT-E3", "tasks": tasks, "task_count": 800,
        "evaluation_images": 400, "model_count": 2, "block_count_per_task": 12,
        "cut_count_per_block": 8, "negative_control_count_per_block": 3,
        "positive_closure_evaluations": 800 * 12 * 8,
        "negative_control_evaluations": 800 * 12 * 3,
        "queue_identity": queue_identity, "manifest_digest": digest(tasks),
        "seed_rule": "SHA256(namespace|task_id) first8 bytes",
    })
    snapshot = gpu_snapshot()
    eligible = [row for row in snapshot if row["eligible"]]
    candidates = [row for row in eligible if not row["existing_compute_processes"]] + [
        row for row in eligible if row["existing_compute_processes"]
    ]
    requested = max(1, min(args.max_gpus, spec["runtime"]["max_gpus"]))
    selected, locks = lock_gpus(candidates, head, root, requested)
    if not selected:
        raise RuntimeError("BLOCKED_INFRASTRUCTURE_NO_LOCKABLE_H20")
    write_json_new(root / "startup_snapshot.json", {
        "hostname": socket.gethostname(), "producer_commit": head,
        "tracking_commit": git("rev-parse", "@{upstream}"),
        "remote_commit": remote("refs/heads/stage-d-d3-pass"),
        "predecessor_validation": predecessor_validation, "gpu_snapshot": snapshot,
        "selected_gpus": selected, "free_disk_bytes": shutil.disk_usage(REPO).free,
        "CUBLAS_WORKSPACE_CONFIG_before_workers": ":4096:8",
        "formal_image_reads": {"evaluation": 400, "calibration": 0, "discovery": 0},
        "existing_pids_recorded": True, "existing_processes_not_terminated": True,
    })
    try:
        exits, formal_seconds = launch(queue, root / "formal_results", root / "raw", selected, root)
    finally:
        for stream in locks:
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()
    coverage = audit_coverage(queue, root / "formal_results")
    write_json_new(root / "formal_coverage_audit.json", coverage)
    if any(row["return_code"] for row in exits) or coverage["status"] != "PASS":
        raise RuntimeError("FORMAL_WORKER_OR_COVERAGE_FAILED")
    formal = queue_results(queue, root / "formal_results")
    cut_rows = [cut for result in formal.values() for block in result["blocks"] for cut in block["cuts"]]
    negative_rows = [row for result in formal.values() for block in result["blocks"] for row in block["negative_controls"]]
    if not all(row["pass"] for row in cut_rows) or not all(row["passed"] for row in negative_rows):
        raise RuntimeError("CLOSURE_OR_NEGATIVE_CONTROL")
    if not all(math.isfinite(float(row[key])) for row in cut_rows for key in ("max_abs", "relative_l2", "reference_norm")):
        raise RuntimeError("NONFINITE_CLOSURE_METRIC")
    retained_before_report = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    gpu_hours = formal_seconds * len(selected) / 3600
    report = {
        "status": PASS_STATUS, "stage": "ENDPOINT-E3", "claim_decision": CLAIM_DECISION,
        "claim_scope": "CORRECTNESS_ONLY_NOT_UTILITY", "producer_commit": head,
        "evaluation_images": 400, "models": spec["models"], "image_model_tasks": 800,
        "blocks_per_task": 12, "positive_closure_evaluations": len(cut_rows),
        "negative_control_evaluations": len(negative_rows),
        "all_positive_closures_pass": True, "all_negative_controls_rejected": True,
        "maximum_closure_max_abs": max(row["max_abs"] for row in cut_rows),
        "maximum_closure_relative_l2": max(row["relative_l2"] for row in cut_rows),
        "closure_branch_counts": {
            branch: sum(row["branch"] == branch for row in cut_rows)
            for branch in ("absolute_near_zero", "relative_nonzero")
        },
        "cut_counts": {cut_id: sum(row["cut_id"] == cut_id for row in cut_rows) for cut_id in (
            "heads_branch", "heads_total", "qkv_branch", "routing_value_branch",
            "qkv_total", "routing_value_total", "residual1", "residual2",
        )},
        "raw_component_map_tasks": sum(bool(row["artifacts"]) for row in formal.values()),
        "formal_coverage": coverage, "task_manifest_digest": queue_identity["manifest_digest"],
        "materialization_records_digest": material["records_digest"],
        "latency_seconds": {
            "exclusive_scan_p50": linear_percentile([row["scan_seconds"] for row in formal.values()], 0.5),
            "exclusive_scan_p95": linear_percentile([row["scan_seconds"] for row in formal.values()], 0.95),
            "exclusive_scan_max": max(row["scan_seconds"] for row in formal.values()),
            "end_to_end_task_p50": linear_percentile([row["task_seconds"] for row in formal.values()], 0.5),
            "end_to_end_task_p95": linear_percentile([row["task_seconds"] for row in formal.values()], 0.95),
            "end_to_end_task_max": max(row["task_seconds"] for row in formal.values()),
        },
        "peak_allocated_gpu_bytes_max": max(row["peak_allocated_gpu_bytes"] for row in formal.values()),
        "peak_reserved_gpu_bytes_max": max(row["peak_reserved_gpu_bytes"] for row in formal.values()),
        "peak_cpu_rss_bytes_max": max(row["peak_cpu_rss_bytes"] for row in formal.values()),
        "retained_bytes_before_report": retained_before_report,
        "formal_wall_seconds": formal_seconds, "total_wall_seconds": time.monotonic() - stage_started,
        "gpu_hours_upper_observed": gpu_hours, "formal_workers": len(selected),
        "worker_exits": exits,
        "model_tensor_digests": {
            model_id: next(row for row in formal.values() if row["model_id"] == model_id)["model_binding"]["tensor_digest"]
            for model_id in spec["models"]
        },
        "implementation_bindings": [{
            "path": path, "sha256": file_sha256(REPO / path), "commit": head,
        } for path in SOURCE_FILES],
        "prohibited_methods": {
            "path_ig": "NOT_RUN_NOT_ALLOWED", "input_fidelity_stability": "NOT_RUN_NOT_ALLOWED",
            "activation_replacement": "NOT_RUN_NOT_ALLOWED", "finite_intervention": "NOT_RUN_NOT_ALLOWED",
            "utility_claim": "NOT_RUN_NOT_ALLOWED",
        },
        "resource_caps": {"gpu_hours": 64, "wall_hours": 72, "incremental_retained_gib": 128},
        "next_authorization": "ENDPOINT_E4_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    if gpu_hours > 64 or formal_seconds > 72 * 3600 or retained_before_report > 128 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP")
    write_json_new(root / "validation_report.json", report)
    files = [{
        "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path),
    } for path in sorted(root.rglob("*")) if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}]
    manifest = {"files": files, "canonical_digest": digest(files)}
    write_json_new(root / "artifact_manifest.json", manifest)
    terminal = {
        "status": PASS_STATUS, "stage": "ENDPOINT-E3", "claim_decision": CLAIM_DECISION,
        "claim_scope": "CORRECTNESS_ONLY_NOT_UTILITY", "producer_commit": head,
        "evaluation_image_count": 400, "image_model_task_count": 800,
        "block_evaluation_count": 9600, "positive_closure_evaluation_count": len(cut_rows),
        "negative_control_evaluation_count": len(negative_rows),
        "validation_digest": digest(report), "manifest_digest": manifest["canonical_digest"],
        "next_authorization": "ENDPOINT_E4_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    write_json_new(root / "terminal_receipt.json", terminal)
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
