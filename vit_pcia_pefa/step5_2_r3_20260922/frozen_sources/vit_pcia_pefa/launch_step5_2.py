#!/usr/bin/env python3
"""Launch one frozen Step-5.2 phase on a dynamic global H20 queue."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_2_io import verify, write_new  # noqa: E402
from vit_pcia_pefa.src.step5_2_queue import audit_coverage, create_queue  # noqa: E402

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
            "memory_free_mib": int(free), "utilization_percent": int(utilization), "existing_processes": by_uuid.get(uuid, [])})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("prediction", "intervention")); parser.add_argument("root", type=Path)
    args = parser.parse_args(); run = args.root.resolve(); protocol = verify(run / "protocol.json"); manifest = verify(run / "task_manifest.json")
    if args.phase == "prediction":
        decision_path = run / "p0_decision.json"
        rebind_path = run / "p0_rebind_receipt.json"
        decision = verify(decision_path if decision_path.exists() else rebind_path)
        if not decision["prediction_authorized_by_gate"]: raise RuntimeError("P0_GATE_CLOSED")
        tasks = manifest["prediction_tasks"]; worker = ROOT / "run_step5_2_prediction_worker.py"
    else:
        receipt = verify(run / "prediction_freeze_receipt.json")
        if receipt["status"] != "PASS_PREDICTIONS_FROZEN_BEFORE_INTERVENTION": raise RuntimeError("PREDICTION_FREEZE_GATE_CLOSED")
        tasks = manifest["intervention_tasks"]; worker = ROOT / "run_step5_2_intervention_worker.py"
    startup_path = run / f"{args.phase}_startup_snapshot.json"; terminal_path = run / f"{args.phase}_launch_terminal.json"
    if startup_path.exists() or terminal_path.exists(): raise RuntimeError("PHASE_LAUNCH_ALREADY_EXISTS")
    queue = run / f"{args.phase}_queue.sqlite3"; results = run / f"{args.phase}_results"; logs = run / f"{args.phase}_logs"
    results.mkdir(); logs.mkdir(); queue_receipt = create_queue(queue, tasks, args.phase)
    rows = gpu_snapshot(); cap = protocol["resources"]
    eligible = [row for row in rows if row["name"].startswith("NVIDIA H20") and row["memory_free_mib"] >= cap["minimum_free_gpu_mib"]
        and row["utilization_percent"] <= cap["maximum_gpu_utilization_percent"]]
    eligible.sort(key=lambda row: (-row["memory_free_mib"], row["index"])); selected = eligible[:cap["maximum_gpus"]]
    if not selected: raise RuntimeError("PAUSED_RESOURCE_CAP:NO_ELIGIBLE_H20")
    write_new(startup_path, {"schema_version": 1, "run_id": protocol["run_id"],
        "status": f"PASS_{args.phase.upper()}_STARTUP_GATE", "created_unix_ns": time.time_ns(), "all_gpus": rows,
        "selected_gpus": selected, "queue": queue_receipt, "dynamic_global_queue": True, "static_modulo_sharding": False,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "existing_pids_recorded": True, "other_processes_terminated": False})
    started = time.monotonic(); processes = []; streams = []
    for ordinal, gpu in enumerate(selected):
        stream = (logs / f"worker-{ordinal:02d}.log").open("xb"); streams.append(stream)
        environment = dict(os.environ); environment.update({"CUDA_VISIBLE_DEVICES": str(gpu["index"]),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": ".:vitnns_rebuild/src", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        command = [str(PYTHON), str(worker), "--root", str(run), "--physical-index", str(gpu["index"]),
            "--gpu-uuid", gpu["uuid"], "--worker-id", f"step5-2-{args.phase}-worker-{ordinal:02d}"]
        processes.append((ordinal, subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)))
    exits = []
    try:
        for ordinal, process in processes: exits.append({"worker_id": f"step5-2-{args.phase}-worker-{ordinal:02d}", "exit_code": process.wait()})
    finally:
        for stream in streams: stream.flush(); os.fsync(stream.fileno()); stream.close()
    coverage = audit_coverage(queue, results, args.phase)
    passed = coverage["status"] == "PASS" and all(row["exit_code"] == 0 for row in exits)
    terminal = write_new(terminal_path, {"schema_version": 1, "run_id": protocol["run_id"],
        "status": f"PASS_{args.phase.upper()}_TASK_EXECUTION" if passed else f"FAIL_{args.phase.upper()}_TASK_EXECUTION",
        "completed_unix_ns": time.time_ns(), "elapsed_seconds": time.monotonic() - started, "selected_gpus": selected,
        "worker_exits": exits, "coverage": coverage, "evaluation_access": False, "step6_started": False})
    print(json.dumps({"status": terminal["status"], "coverage": coverage}, sort_keys=True)); return 0 if passed else 3


if __name__ == "__main__": raise SystemExit(main())
