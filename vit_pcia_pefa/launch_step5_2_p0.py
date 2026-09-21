#!/usr/bin/env python3
"""Run the Step-5.2 P0 startup gate and real-model workers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_2_io import verify, write_new  # noqa: E402

PYTHON = REPO / ".conda/envs/vitnns/bin/python3.10"


def gpu_snapshot():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,name",
        "--format=csv,noheader,nounits"], text=True)
    by_uuid = {}
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
    if len(sys.argv) != 2: raise SystemExit("usage: launch_step5_2_p0.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve(); protocol = verify(run / "protocol.json"); verify(run / "freeze_receipt.json")
    if any((run / name).exists() for name in ("p0_startup_snapshot.json", "p0_launch_terminal.json")):
        raise RuntimeError("P0_LAUNCH_ALREADY_EXISTS")
    rows = gpu_snapshot(); cap = protocol["resources"]
    eligible = [row for row in rows if row["name"].startswith("NVIDIA H20")
        and row["memory_free_mib"] >= cap["minimum_free_gpu_mib"]
        and row["utilization_percent"] <= cap["maximum_gpu_utilization_percent"]]
    eligible.sort(key=lambda row: (-row["memory_free_mib"], row["index"]))
    required = len(protocol["models"]); selected = eligible[:required]
    if len(selected) < required: raise RuntimeError(f"PAUSED_RESOURCE_CAP:NEED_{required}_ELIGIBLE_H20")
    write_new(run / "p0_startup_snapshot.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "PASS_P0_STARTUP_GATE", "created_unix_ns": time.time_ns(), "all_gpus": rows,
        "selected_gpus": selected, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "existing_pids_recorded": True,
        "other_processes_terminated": False})
    (run / "p0").mkdir(); (run / "p0_logs").mkdir(); processes = []; streams = []
    for model_id, gpu in zip(protocol["models"], selected):
        stream = (run / "p0_logs" / f"{model_id}.log").open("xb"); streams.append(stream)
        environment = dict(os.environ); environment.update({"CUDA_VISIBLE_DEVICES": str(gpu["index"]),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": ".:vitnns_rebuild/src",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        command = [str(PYTHON), str(ROOT / "run_step5_2_p0_worker.py"), "--root", str(run),
            "--model-id", model_id, "--physical-index", str(gpu["index"]), "--gpu-uuid", gpu["uuid"]]
        processes.append((model_id, subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)))
    exits = []
    try:
        for model_id, process in processes: exits.append({"model_id": model_id, "exit_code": process.wait()})
    finally:
        for stream in streams: stream.flush(); os.fsync(stream.fileno()); stream.close()
    passed = all(row["exit_code"] == 0 for row in exits)
    result = write_new(run / "p0_launch_terminal.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "PASS_P0_TASK_EXECUTION" if passed else "INVALID_P0_TASK_EXECUTION",
        "worker_exits": exits, "completed_unix_ns": time.time_ns(), "evaluation_access": False, "step6_started": False})
    print(json.dumps({"status": result["status"], "exits": exits}, sort_keys=True)); return 0 if passed else 3


if __name__ == "__main__": raise SystemExit(main())
