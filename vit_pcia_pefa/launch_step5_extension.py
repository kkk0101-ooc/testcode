#!/usr/bin/env python3
"""Qualify H20 devices and launch a frozen Step-5 extension queue."""
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
from vit_pcia_pefa.src.step5_extension_queue import audit_coverage, create_queue  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value); value["canonical_digest"] = digest(value); return value


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if digest(value) != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def write_new(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def gpu_snapshot() -> list[dict[str, Any]]:
    fields = "index,uuid,name,memory.total,memory.free,utilization.gpu,temperature.gpu"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True,
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,name",
         "--format=csv,noheader,nounits"], text=True,
    )
    by_uuid: dict[str, list[dict[str, Any]]] = {}
    for line in processes.splitlines():
        if line.strip():
            uuid, pid, memory, name = [part.strip() for part in line.split(",", 3)]
            by_uuid.setdefault(uuid, []).append({"pid": int(pid), "used_memory_mib": int(memory),
                                                 "name": name})
    rows = []
    for line in output.splitlines():
        index, uuid, name, total, free, utilization, temperature = [part.strip() for part in line.split(",", 6)]
        rows.append({"index": int(index), "uuid": uuid, "name": name,
                     "memory_total_mib": int(total), "memory_free_mib": int(free),
                     "utilization_percent": int(utilization), "temperature_c": int(temperature),
                     "existing_processes": by_uuid.get(uuid, [])})
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: launch_step5_extension.py EXTENSION_ROOT")
    root = Path(sys.argv[1]).resolve()
    protocol = verify_signed(root / "protocol.json")
    manifest = verify_signed(root / "task_manifest.json")
    freeze = verify_signed(root / "freeze_receipt.json")
    honest_recovery = (freeze["extension_outcome_access_before_freeze"]
                       and freeze.get("recovery_trigger_outcome_independent") is True
                       and protocol.get("prior_outcome_known") is True
                       and protocol.get("outcome_access_during_execution") is False)
    if ((freeze["extension_outcome_access_before_freeze"] and not honest_recovery)
            or freeze["step6_started"]):
        raise RuntimeError("OUTCOME_BLINDNESS_OR_STAGE_BOUNDARY")
    if any((root / name).exists() for name in ("queue.sqlite3", "startup_snapshot.json",
                                                "launch_terminal.json")):
        raise RuntimeError("LAUNCH_ALREADY_EXISTS")
    queue = root / "queue.sqlite3"; results = root / "results"; logs = root / "logs"
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
    snapshot = signed({
        "schema_version": 1, "run_id": protocol["run_id"], "phase": protocol["phase"],
        "status": "PASS_STARTUP_GATE", "created_unix_ns": started_ns, "all_gpus": rows,
        "eligible_gpu_count": len(eligible), "selected_gpus": selected,
        "selected_gpu_count": len(selected), "queue": queue_receipt,
        "dynamic_global_queue": True, "static_modulo_sharding": False,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "existing_pids_recorded": True,
        "other_processes_terminated": False, "step6_started": False,
    })
    write_new(root / "startup_snapshot.json", snapshot)
    processes, streams = [], []
    for ordinal, gpu in enumerate(selected):
        stream = (logs / f"worker-{ordinal:02d}.log").open("xb"); streams.append(stream)
        environment = dict(os.environ)
        environment.update({"CUDA_VISIBLE_DEVICES": str(gpu["index"]),
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "PYTHONPATH": ".:vitnns_rebuild/src",
                            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        command = [str(REPO / ".conda/envs/vitnns/bin/python"),
                   str(ROOT / "run_step5_extension_worker.py"), "--root", str(root),
                   "--physical-index", str(gpu["index"]), "--gpu-uuid", gpu["uuid"],
                   "--worker-id", f"step5-extension-worker-{ordinal:02d}"]
        processes.append(subprocess.Popen(command, cwd=REPO, env=environment,
                                          stdout=stream, stderr=subprocess.STDOUT))
    exits = []
    try:
        for ordinal, process in enumerate(processes):
            exits.append({"worker_id": f"step5-extension-worker-{ordinal:02d}",
                          "exit_code": process.wait()})
    finally:
        for stream in streams:
            stream.flush(); os.fsync(stream.fileno()); stream.close()
    coverage = audit_coverage(queue, results)
    status = ("PASS_EXTENSION_EXECUTION" if coverage["status"] == "PASS"
              and all(row["exit_code"] == 0 for row in exits) else "FAIL_EXTENSION_EXECUTION")
    terminal = signed({
        "schema_version": 1, "run_id": protocol["run_id"], "phase": protocol["phase"],
        "status": status, "started_unix_ns": started_ns, "completed_unix_ns": time.time_ns(),
        "elapsed_seconds": time.monotonic() - started, "selected_gpu_count": len(selected),
        "selected_gpus": selected, "worker_exits": exits, "coverage": coverage,
        "step6_started": False,
    })
    write_new(root / "launch_terminal.json", terminal)
    print(json.dumps({"status": status, "elapsed_seconds": terminal["elapsed_seconds"],
                      "selected_gpu_count": len(selected), "coverage": coverage}, sort_keys=True))
    return 0 if status == "PASS_EXTENSION_EXECUTION" else 3


if __name__ == "__main__":
    raise SystemExit(main())
