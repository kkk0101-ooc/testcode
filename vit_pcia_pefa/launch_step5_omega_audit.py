#!/usr/bin/env python3
"""Launch the frozen independent float64 Omega audit queue."""
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
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.launch_step5 import gpu_snapshot, signed, verify_signed, write_new  # noqa: E402
from vit_pcia_pefa.src.step5_queue import audit_coverage, create_queue  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: launch_step5_omega_audit.py PILOT_ROOT")
    pilot = Path(sys.argv[1]).resolve()
    protocol = verify_signed(pilot / "protocol.json")
    manifest = verify_signed(pilot / "task_manifest.json")
    if protocol.get("omega_zero_sum_gate_source") != "independent_float64_complete_cut":
        raise RuntimeError("OMEGA_AUDIT_NOT_FROZEN")
    if any((pilot / name).exists() for name in ("omega_audit_startup_snapshot.json", "omega_audit_terminal.json")):
        raise RuntimeError("OMEGA_AUDIT_ALREADY_EXISTS")
    queue = pilot / "omega_audit_queue.sqlite3"
    results = pilot / "omega_audit_results"
    logs = pilot / "omega_audit_logs"
    results.mkdir(exist_ok=True); logs.mkdir(exist_ok=True)
    queue_receipt = create_queue(queue, manifest["tasks"])
    rows = gpu_snapshot(); cap = protocol["resources"]
    eligible = [row for row in rows if row["name"].startswith("NVIDIA H20")
                and row["memory_free_mib"] >= cap["minimum_free_gpu_mib"]
                and row["utilization_percent"] <= cap["maximum_gpu_utilization_percent"]]
    eligible.sort(key=lambda row: (-row["memory_free_mib"], row["index"]))
    selected = eligible[:cap["maximum_gpus"]]
    if not selected:
        raise RuntimeError("PAUSED_RESOURCE_CAP:NO_ELIGIBLE_H20")
    started_ns = time.time_ns(); started = time.monotonic()
    write_new(pilot / "omega_audit_startup_snapshot.json", signed({
        "schema_version": 1, "run_id": protocol["run_id"], "status": "PASS_OMEGA_AUDIT_STARTUP_GATE",
        "created_unix_ns": started_ns, "all_gpus": rows, "selected_gpus": selected,
        "selected_gpu_count": len(selected), "queue": queue_receipt,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "dtype": "float64",
        "dynamic_global_queue": True, "existing_pids_recorded": True,
        "other_processes_terminated": False,
    }))
    processes = []; streams = []
    for ordinal, gpu in enumerate(selected):
        stream = (logs / f"worker-{ordinal:02d}.log").open("xb"); streams.append(stream)
        environment = dict(os.environ)
        environment.update({"CUDA_VISIBLE_DEVICES": str(gpu["index"]),
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "PYTHONPATH": ".:vitnns_rebuild/src",
                            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        command = [str(REPO / ".conda/envs/vitnns/bin/python"),
                   str(ROOT / "run_step5_omega_audit_worker.py"),
                   "--root", str(pilot), "--physical-index", str(gpu["index"]),
                   "--gpu-uuid", gpu["uuid"], "--worker-id", f"omega-audit-worker-{ordinal:02d}"]
        processes.append(subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT))
    exits = []
    try:
        for ordinal, process in enumerate(processes):
            exits.append({"worker_id": f"omega-audit-worker-{ordinal:02d}", "exit_code": process.wait()})
    finally:
        for stream in streams:
            stream.flush(); os.fsync(stream.fileno()); stream.close()
    coverage = audit_coverage(queue, results)
    status = "PASS_OMEGA_AUDIT_EXECUTION" if coverage["status"] == "PASS" and all(row["exit_code"] == 0 for row in exits) else "FAIL_OMEGA_AUDIT_EXECUTION"
    terminal = signed({
        "schema_version": 1, "run_id": protocol["run_id"], "status": status,
        "started_unix_ns": started_ns, "completed_unix_ns": time.time_ns(),
        "elapsed_seconds": time.monotonic() - started, "selected_gpu_count": len(selected),
        "selected_gpus": selected, "worker_exits": exits, "coverage": coverage,
        "step6_started": False,
    })
    write_new(pilot / "omega_audit_terminal.json", terminal)
    print(json.dumps({"status": status, "elapsed_seconds": terminal["elapsed_seconds"],
                      "selected_gpu_count": len(selected), "coverage": coverage}, sort_keys=True))
    return 0 if status == "PASS_OMEGA_AUDIT_EXECUTION" else 3


if __name__ == "__main__":
    raise SystemExit(main())
