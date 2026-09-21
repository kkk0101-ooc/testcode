#!/usr/bin/env python3
"""Qualify idle H20s and launch the frozen Step-5.1 A1 queue."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PYTHON = REPO / ".conda/envs/vitnns/bin/python3.10"
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_1_queue import audit_coverage, create_queue  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def verify(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if hashlib.sha256(canonical(value)).hexdigest() != claimed: raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    body = dict(value); body["canonical_digest"] = hashlib.sha256(canonical(body)).hexdigest()
    payload = json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def gpu_snapshot() -> list[dict[str, Any]]:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,name",
                                         "--format=csv,noheader,nounits"], text=True)
    by_uuid: dict[str, list[dict[str, Any]]] = {}
    for line in processes.splitlines():
        if line.strip():
            uuid, pid, memory, name = [item.strip() for item in line.split(",", 3)]
            by_uuid.setdefault(uuid, []).append({"pid": int(pid), "used_memory_mib": int(memory), "name": name})
    rows = []
    for line in output.splitlines():
        index, uuid, name, total, free, utilization = [item.strip() for item in line.split(",", 5)]
        rows.append({"index": int(index), "uuid": uuid, "name": name, "memory_total_mib": int(total),
                     "memory_free_mib": int(free), "utilization_percent": int(utilization),
                     "existing_processes": by_uuid.get(uuid, [])})
    return rows


def main() -> int:
    if len(sys.argv) != 2: raise SystemExit("usage: launch_step5_1_a1.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve(); protocol = verify(run / "protocol.json")
    manifest = verify(run / "task_manifest.json"); inherited = verify(run / "a0_inheritance_receipt.json")
    decision = verify(run / "a0_decision.json")
    if inherited["status"] not in {"PASS_A0_INHERITANCE", "PASS_A0_C1_INHERITANCE"} or not decision["a1_authorized_by_gate"]:
        raise RuntimeError("A0_GATE_NOT_OPEN")
    if any((run / name).exists() for name in ("a1_startup_snapshot.json", "a1_launch_terminal.json")):
        raise RuntimeError("A1_LAUNCH_ALREADY_EXISTS")
    queue = run / "a1_queue.sqlite3"; results = run / "a1_results"; logs = run / "a1_logs"
    results.mkdir(); logs.mkdir(); queue_receipt = create_queue(queue, manifest["a1_tasks"])
    rows = gpu_snapshot(); cap = protocol["resources"]
    eligible = [row for row in rows if row["name"].startswith("NVIDIA H20")
                and row["memory_free_mib"] >= cap["minimum_free_gpu_mib"]
                and row["utilization_percent"] <= cap["maximum_gpu_utilization_percent"]]
    eligible.sort(key=lambda row: (-row["memory_free_mib"], row["index"]))
    selected = eligible[:cap["maximum_gpus"]]
    if not selected: raise RuntimeError("PAUSED_RESOURCE_CAP:NO_ELIGIBLE_H20")
    write_new(run / "a1_startup_snapshot.json", {
        "schema_version": 1, "run_id": protocol["run_id"], "status": "PASS_A1_STARTUP_GATE",
        "created_unix_ns": time.time_ns(), "all_gpus": rows, "selected_gpus": selected,
        "selected_gpu_count": len(selected), "queue": queue_receipt, "dynamic_global_queue": True,
        "static_modulo_sharding": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "existing_pids_recorded": True, "other_processes_terminated": False,
    })
    started = time.monotonic(); processes = []; streams = []
    for ordinal, gpu in enumerate(selected):
        stream = (logs / f"worker-{ordinal:02d}.log").open("xb"); streams.append(stream)
        environment = dict(os.environ); environment.update({"CUDA_VISIBLE_DEVICES": str(gpu["index"]),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": ".:vitnns_rebuild/src",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        command = [str(PYTHON), str(ROOT / "run_step5_1_a1_worker.py"), "--root", str(run),
                   "--physical-index", str(gpu["index"]), "--gpu-uuid", gpu["uuid"],
                   "--worker-id", f"step5-1-a1-worker-{ordinal:02d}"]
        processes.append((ordinal, subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)))
    exits = []
    try:
        for ordinal, process in processes: exits.append({"worker_id": f"step5-1-a1-worker-{ordinal:02d}", "exit_code": process.wait()})
    finally:
        for stream in streams: stream.flush(); os.fsync(stream.fileno()); stream.close()
    coverage = audit_coverage(queue, results)
    status = "PASS_A1_TASK_EXECUTION" if coverage["status"] == "PASS" and all(row["exit_code"] == 0 for row in exits) else "FAIL_A1_TASK_EXECUTION"
    write_new(run / "a1_launch_terminal.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": status, "completed_unix_ns": time.time_ns(), "elapsed_seconds": time.monotonic() - started,
        "selected_gpus": selected, "worker_exits": exits, "coverage": coverage, "step6_started": False})
    print(json.dumps({"status": status, "coverage": coverage}, sort_keys=True))
    return 0 if status.startswith("PASS") else 3


if __name__ == "__main__": raise SystemExit(main())
