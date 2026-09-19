#!/usr/bin/env python3
"""Freeze and execute Endpoint M4 on the published 800-task cohort."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
GIT_ROOT = REPO.parent
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_m4.json"
M1_ROOT = REPO / "reports/nns_vit_endpoint_m1_r1_20260918"
M3_ROOT = REPO / "reports/nns_vit_endpoint_m3_r1_20260919"
RUNTIME = GIT_ROOT / ".runtime/nns_vit_endpoint_m4"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e2_atlas import file_sha256  # noqa: E402
from vitnns.endpoint_ep1_queue import audit_coverage, create_queue, digest  # noqa: E402

SOURCE_FILES = [
    "config/nns_vit_endpoint/endpoint_m4.json",
    "docs/endpoint_m4_execution_20260919.md",
    "src/vitnns/endpoint_m4.py",
    "src/vitnns/endpoint_m4_statistics.py",
    "src/vitnns/endpoint_m3_protocol.py",
    "src/vitnns/endpoint_ep1_queue.py",
    "scripts/run_endpoint_m4.py",
    "scripts/run_endpoint_m4_worker.py",
    "scripts/run_endpoint_m4_analysis.py",
    "scripts/run_endpoint_e4_intervention_worker.py",
    "tests/test_endpoint_m4.py",
    "tests/test_endpoint_m4_statistics.py",
]
ALLOWED_UNTRACKED = {
    "config/nns_vit_cut_intervention/p0_c.json",
    "manifests/nns_vit_cut_intervention/p0_c_operation_registry.json",
    "scripts/validate_nns_vit_cut_intervention_p0_c.py",
    "src/vitnns/nns_vit_cut_intervention_p0_c.py",
    "src/vitnns/nns_vit_cut_intervention_p0_c_worker.py",
    "tests/test_nns_vit_cut_intervention_p0_c.py",
}


def load(path: Path):
    return json.loads(path.read_text())


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def remote(ref: str) -> str:
    fields = subprocess.check_output(["git", "ls-remote", "origin", ref], cwd=REPO, text=True).split()
    return fields[0] if len(fields) == 2 else ""


def changed_paths() -> set[str]:
    prefix = REPO.relative_to(GIT_ROOT).as_posix() + "/"
    rows = set()
    for line in git("status", "--porcelain", "--untracked-files=all").splitlines():
        path = line[3:].split(" -> ")[-1]
        rows.add(path[len(prefix):] if path.startswith(prefix) else path)
    return rows


def validate_sources(spec: dict) -> None:
    for relative, expected in spec["source_bindings"].items():
        if file_sha256(REPO / relative) != expected:
            raise ValueError("SOURCE_BINDING_DRIFT:" + relative)
    terminal = load(M3_ROOT / "terminal_receipt.json")
    if terminal["status"] != "ENDPOINT_M3_PASS_EXECUTION_AND_AUDIT" or terminal["next_authorization"] != "STOP_AFTER_M3":
        raise ValueError("M3_TERMINAL")
    completed = subprocess.run(
        [sys.executable, "scripts/seal_endpoint_m3.py", "--verify"], cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1"),
        text=True, capture_output=True,
    )
    if completed.returncode:
        raise ValueError("M3_REVALIDATION:" + completed.stderr)


def prepare() -> None:
    spec = load(SPEC_PATH); validate_sources(spec)
    root = REPO / spec["output_root"]
    if root.exists():
        raise ValueError("FRESH_ROOT_REQUIRED")
    source = load(M1_ROOT / "task_manifest.json")["tasks"]
    if len(source) != 800:
        raise ValueError("SOURCE_TASK_COUNT")
    tasks = []
    expected_fields = {
        "model_id", "timm_name", "embedding_dim", "checkpoint",
        "checkpoint_bytes", "checkpoint_sha256", "tensor_digest",
        "mean", "std", "sample_id", "split_order", "class_id",
        "wnid", "cache_path", "image_bytes", "image_sha256",
    }
    for task in source:
        old = task["payload"]
        payload = {key: old[key] for key in expected_fields}
        payload.update({
            "stage": "ENDPOINT-M4", "vjp_chunk_size": spec["numeric"]["vjp_chunk_size"],
            "jvp_directions": spec["attention"]["jvp_directions_per_image"],
            "detach_validation": old["split_order"] in spec["attention"]["attention_detach_validation_orders"],
            "e3_path": old["e3_path"], "e3_sha256": old["e3_sha256"],
        })
        e3 = REPO / payload["e3_path"]
        if file_sha256(e3) != payload["e3_sha256"]:
            raise ValueError("E3_TASK_SOURCE_DRIFT:" + task["task_id"])
        tasks.append({"task_id": task["task_id"], "payload": payload})
    tasks.sort(key=lambda row: row["task_id"])
    if len({row["task_id"] for row in tasks}) != 800:
        raise ValueError("TASK_ID_DUPLICATE")
    by_model = {}
    for row in tasks:
        by_model.setdefault(row["payload"]["model_id"], set()).add(row["payload"]["sample_id"])
    if set(by_model) != set(spec["models"]) or len(set.intersection(*by_model.values())) != 400:
        raise ValueError("SHARED_COHORT")
    root.mkdir(parents=True)
    write_new(root / "task_manifest.json", {"tasks": tasks, "task_count": 800, "digest": digest(tasks)})
    queue_identity = create_queue(root / "queue.sqlite3", tasks)
    write_new(root / "freeze.json", {
        "stage": "ENDPOINT-M4", "time_ns": time.time_ns(),
        "sources": {relative: file_sha256(REPO / relative) for relative in SOURCE_FILES},
        "task_manifest_sha256": file_sha256(root / "task_manifest.json"),
        "task_manifest_digest": queue_identity["manifest_digest"],
        "predecessor_head": git("rev-parse", "HEAD"),
    })
    write_new(root / "predecessor_validation.json", {
        "status": "PASS", "m3_terminal_status": "ENDPOINT_M3_PASS_EXECUTION_AND_AUDIT",
        "m3_verifier": "PASS_462_FILES_24_REQUIREMENTS",
        "source_bindings": spec["source_bindings"],
    })
    print(json.dumps({"status": "M4_PREPARED", "tasks": 800, "root": str(root)}))


def gpu_snapshot(spec: dict) -> tuple[list[dict], str]:
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    process_text = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    processes = {}
    for line in process_text.splitlines():
        if line.strip():
            uuid, pid, memory = (value.strip() for value in line.split(","))
            processes.setdefault(uuid, []).append({"pid": int(pid), "used_memory_mib": int(memory)})
    rows = []
    for line in query.splitlines():
        index, uuid, name, free, total, utilization = (value.strip() for value in line.split(","))
        row = {"index": int(index), "uuid": uuid, "name": name, "free_mib": int(free),
               "total_mib": int(total), "utilization_percent": int(utilization),
               "existing_compute_processes": processes.get(uuid, [])}
        row["eligible"] = name == "NVIDIA H20" and row["free_mib"] >= spec["runtime"]["min_free_mib"] and row["utilization_percent"] <= spec["runtime"]["max_utilization_percent"]
        rows.append(row)
    return rows, process_text


def run() -> None:
    spec = load(SPEC_PATH); validate_sources(spec); root = REPO / spec["output_root"]
    freeze = load(root / "freeze.json")
    for relative, expected in freeze["sources"].items():
        if file_sha256(REPO / relative) != expected:
            raise ValueError("FROZEN_SOURCE_DRIFT:" + relative)
    if file_sha256(root / "task_manifest.json") != freeze["task_manifest_sha256"]:
        raise ValueError("TASK_MANIFEST_DRIFT")
    head = git("rev-parse", "HEAD"); branch = git("symbolic-ref", "--short", "HEAD")
    upstream = git("rev-parse", "@{upstream}"); remote_head = remote("refs/heads/" + branch)
    if head != upstream or head != remote_head:
        raise ValueError("PUBLISHED_PRODUCER_REQUIRED")
    if subprocess.run(["git", "merge-base", "--is-ancestor", freeze["predecessor_head"], head], cwd=REPO).returncode:
        raise ValueError("FREEZE_COMMIT_ANCESTRY")
    for relative, expected in freeze["sources"].items():
        blob = subprocess.check_output(["git", "show", f"{head}:vitnns_rebuild/{relative}"], cwd=REPO)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError("PUBLISHED_SOURCE_DRIFT:" + relative)
    unexplained = changed_paths() - ALLOWED_UNTRACKED
    if unexplained:
        raise ValueError("UNEXPLAINED_WORKTREE:" + str(sorted(unexplained)))
    snapshot, process_text = gpu_snapshot(spec); RUNTIME.mkdir(parents=True, exist_ok=True)
    locks, selected = [], []
    for gpu in (row for row in snapshot if row["eligible"]):
        stream = (RUNTIME / (gpu["uuid"] + ".lock")).open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close(); continue
        locks.append(stream); selected.append(gpu)
        if len(selected) == spec["runtime"]["gpus_max"]:
            break
    if not selected:
        raise ValueError("NO_ELIGIBLE_LOCKABLE_H20")
    write_new(root / "startup.json", {
        "producer_commit": head, "tracking_commit": upstream, "remote_commit": remote_head,
        "freeze_sha256": file_sha256(root / "freeze.json"),
        "selected_gpus": selected, "gpu_snapshot": snapshot, "existing_pids_raw": process_text,
        "sys_executable_realpath": str(Path(sys.executable).resolve()),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "formal_image_reads": 400,
    })
    processes, streams = [], []; started = time.monotonic()
    try:
        for number, gpu in enumerate(selected):
            worker = f"gpu{gpu['index']}"; stream = (root / f"{worker}.log").open("xb")
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu["index"]),
                               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2",
                               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "scripts/run_endpoint_m4_worker.py", "--queue", str(root / "queue.sqlite3"),
                       "--results", str(root / "results"), "--worker-id", worker,
                       "--physical-index", str(gpu["index"]), "--gpu-uuid", gpu["uuid"]]
            processes.append(subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT))
            streams.append(stream)
        exits = [process.wait() for process in processes]
    finally:
        for stream in streams:
            stream.close()
        for lock in locks:
            fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    coverage = audit_coverage(root / "queue.sqlite3", root / "results")
    write_new(root / "coverage_audit.json", coverage)
    wall = time.monotonic() - started
    with sqlite3.connect(root / "queue.sqlite3") as connection:
        states = dict(connection.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state"))
    passed = all(code == 0 for code in exits) and coverage["status"] == "PASS" and states == {"completed": 800}
    write_new(root / "acquisition_receipt.json", {
        "status": "PASS" if passed else "FAIL", "worker_exits": exits, "states": states,
        "worker_count": len(selected), "wall_seconds": wall,
    })
    print(json.dumps({"status": "M4_ACQUISITION_PASS" if passed else "M4_ACQUISITION_FAIL",
                      "states": states, "workers": len(selected), "wall_seconds": wall}))
    if not passed:
        raise ValueError("M4_ACQUISITION_FAILED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose exactly one of --prepare or --run")
    prepare() if args.prepare else run()
