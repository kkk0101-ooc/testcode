#!/usr/bin/env python3
"""Qualify H20s and launch the two frozen A0 development tasks."""
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


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if hashlib.sha256(canonical(value)).hexdigest() != claimed: raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    body = dict(value); body["canonical_digest"] = hashlib.sha256(canonical(body)).hexdigest()
    payload = json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


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
        rows.append({"index": int(index), "uuid": uuid, "name": name,
                     "memory_total_mib": int(total), "memory_free_mib": int(free),
                     "utilization_percent": int(utilization), "existing_processes": by_uuid.get(uuid, [])})
    return rows


def main() -> int:
    if len(sys.argv) != 2: raise SystemExit("usage: launch_step5_1_a0.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve(); protocol = verify_signed(run / "protocol.json")
    fixture = verify_signed(run / "a0_fixture_result.json")
    if fixture["status"] != "PASS_A0_FIXTURE": raise RuntimeError("A0_FIXTURE_GATE")
    if (run / "a0_startup_snapshot.json").exists() or (run / "a0_launch_terminal.json").exists():
        raise RuntimeError("A0_LAUNCH_ALREADY_EXISTS")
    rows = gpu_snapshot(); cap = protocol["resources"]
    eligible = [row for row in rows if row["name"].startswith("NVIDIA H20")
                and row["memory_free_mib"] >= cap["minimum_free_gpu_mib"]
                and row["utilization_percent"] <= cap["maximum_gpu_utilization_percent"]]
    eligible.sort(key=lambda row: (-row["memory_free_mib"], row["index"]))
    selected = eligible[:2]
    if len(selected) < 2: raise RuntimeError("PAUSED_RESOURCE_CAP:NEED_TWO_A0_H20")
    (run / "a0").mkdir(); (run / "logs").mkdir(exist_ok=True)
    write_new(run / "a0_startup_snapshot.json", {
        "schema_version": 1, "run_id": protocol["run_id"], "status": "PASS_A0_STARTUP_GATE",
        "created_unix_ns": time.time_ns(), "all_gpus": rows, "selected_gpus": selected,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "existing_pids_recorded": True,
        "other_processes_terminated": False, "fixture_digest": fixture["canonical_digest"],
    })
    started = time.monotonic(); processes = []; streams = []
    for model_id, gpu in zip(("deit_s16", "vit_b16"), selected):
        stream = (run / "logs" / f"a0-{model_id}.log").open("xb"); streams.append(stream)
        environment = dict(os.environ); environment.update({
            "CUDA_VISIBLE_DEVICES": str(gpu["index"]), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONPATH": ".:vitnns_rebuild/src", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        })
        command = [str(PYTHON), str(ROOT / "run_step5_1_a0_worker.py"), "--root", str(run),
                   "--model-id", model_id, "--physical-index", str(gpu["index"]), "--gpu-uuid", gpu["uuid"]]
        processes.append((model_id, subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)))
    exits = []
    try:
        for model_id, process in processes:
            exits.append({"model_id": model_id, "exit_code": process.wait()})
    finally:
        for stream in streams: stream.flush(); os.fsync(stream.fileno()); stream.close()
    status = "PASS_A0_TASK_EXECUTION" if all(row["exit_code"] == 0 for row in exits) else "FAIL_A0_TASK_EXECUTION"
    write_new(run / "a0_launch_terminal.json", {
        "schema_version": 1, "run_id": protocol["run_id"], "status": status,
        "completed_unix_ns": time.time_ns(), "elapsed_seconds": time.monotonic() - started,
        "selected_gpus": selected, "worker_exits": exits, "step6_started": False,
    })
    print(json.dumps({"status": status, "exits": exits}, sort_keys=True))
    return 0 if status.startswith("PASS") else 3


if __name__ == "__main__": raise SystemExit(main())
