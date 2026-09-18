#!/usr/bin/env python3
"""Execute Endpoint Scanner EP1 engineering regression and GPU parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import fcntl
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO.parent / ".conda/envs/vitnns/bin/python"
EP0 = REPO / "manifests/nns_vit_endpoint/ep0_r1_20260918"
EP0_COMMIT = "6237ef87a1af0205973b76bee172defe33a0a6da"
REPORT_ROOT = REPO / "reports/nns_vit_endpoint_ep1_r11_20260918"
RUNTIME = REPO.parent / ".runtime/nns_vit_endpoint_ep1"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_ep1_queue import audit_coverage, canonical, create_queue  # noqa: E402


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def gpu_snapshot() -> list[dict]:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.free,utilization.gpu,driver_version",
         "--format=csv,noheader,nounits"], text=True,
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True,
    )
    active: dict[str, list[dict]] = {}
    for line in processes.splitlines():
        if not line.strip():
            continue
        uuid, pid, memory = [item.strip() for item in line.split(",")]
        active.setdefault(uuid, []).append({"pid": int(pid), "used_memory_mib": int(memory)})
    rows = []
    for line in query.splitlines():
        index, uuid, name, pci, total, free, utilization, driver = [item.strip() for item in line.split(",")]
        eligible = name == "NVIDIA H20" and int(free) >= 32768 and int(utilization) <= 10
        rows.append({
            "physical_index": int(index), "gpu_uuid": uuid, "name": name, "pci_bus_id": pci,
            "memory_total_mib": int(total), "memory_free_mib": int(free),
            "utilization_percent": int(utilization), "driver_version": driver,
            "existing_compute_processes": active.get(uuid, []), "eligible": eligible,
        })
    return rows


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def launch_workers(queue: Path, result_root: Path, selected: list[dict], log_root: Path) -> tuple[list[dict], float]:
    processes, logs = [], []
    started = time.monotonic()
    for index, gpu in enumerate(selected):
        worker = f"worker-{index:02d}"
        log_path = log_root / f"{worker}.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("xb"); logs.append(stream)
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(gpu["physical_index"]),
            CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(REPO / "src"),
            PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
        )
        command = [str(PYTHON), "scripts/run_endpoint_ep1_worker.py", "--queue", str(queue),
                   "--result-root", str(result_root), "--worker-id", worker,
                   "--physical-index", str(gpu["physical_index"]), "--gpu-uuid", gpu["gpu_uuid"]]
        processes.append((worker, gpu, subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)))
    exits = []
    for (worker, gpu, process), stream in zip(processes, logs):
        code = process.wait(); stream.flush(); os.fsync(stream.fileno()); stream.close()
        exits.append({"worker_id": worker, "gpu_uuid": gpu["gpu_uuid"], "return_code": code})
    return exits, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORT_ROOT)
    parser.add_argument("--max-gpus", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("EP1 root already exists; use a fresh versioned root")
    output.mkdir(parents=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    head = git("rev-parse", "HEAD")
    protocol = load_json(EP0 / "protocol.json")
    receipt = load_json(REPO / "reports/nns_vit_endpoint_ep0_r1_20260918/terminal_receipt.json")
    if protocol["status"] != "EP0_PROTOCOL_FROZEN" or receipt["status"] != "EP0_PROTOCOL_FROZEN":
        raise RuntimeError("EP0_NOT_FROZEN")
    if not subprocess.run(["git", "merge-base", "--is-ancestor", EP0_COMMIT, head], cwd=REPO).returncode == 0:
        raise RuntimeError("EP0_COMMIT_NOT_ANCESTOR")
    with tempfile.TemporaryDirectory(prefix="endpoint-ep0-verify-", dir=str(RUNTIME)) as temporary:
        checkout = Path(temporary) / "checkout"
        subprocess.run(["git", "worktree", "add", "--detach", str(checkout), EP0_COMMIT], cwd=REPO, check=True, capture_output=True)
        try:
            for relative in (
                "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth",
                "assets/models/stage_x_x2/deit_small_patch16_224.fb_in1k.pth",
            ):
                source, destination = REPO / relative, checkout / "vitnns_rebuild" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
                destination.symlink_to(source)
            local_images = checkout / "vitnns_rebuild/assets/data/ILSVRC2012_val"
            if local_images.is_dir():
                shutil.rmtree(local_images)
            elif local_images.is_symlink():
                local_images.unlink()
            local_images.parent.mkdir(parents=True, exist_ok=True)
            local_images.symlink_to(REPO / "assets/data/ILSVRC2012_val", target_is_directory=True)
            validation = subprocess.run(
                [str(PYTHON), "vitnns_rebuild/scripts/validate_endpoint_ep0.py"], cwd=checkout,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), text=True, capture_output=True,
            )
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(checkout)], cwd=REPO, check=True, capture_output=True)
    if validation.returncode:
        raise RuntimeError("EP0_REVALIDATION_FAILED: " + validation.stderr)
    tests = subprocess.run(
        [str(PYTHON), "-m", "pytest", "-q", "tests/test_endpoint_nns.py",
         "tests/test_endpoint_vit.py", "tests/test_endpoint_ep1_queue.py",
         "--basetemp=/tmp/nns-vit-endpoint-ep1-pytest"], cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                 PYTHONPATH=str(REPO / "src"), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
        text=True, capture_output=True,
    )
    write_json_new(output / "cpu_regression.json", {
        "return_code": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr,
    })
    if tests.returncode:
        raise RuntimeError("CPU_REGRESSION_FAILED")

    snapshot = gpu_snapshot()
    eligible = [row for row in snapshot if row["eligible"]]
    completely_idle = [row for row in eligible if not row["existing_compute_processes"]]
    shared = [row for row in eligible if row["existing_compute_processes"]]
    selected = (completely_idle + shared)[:max(1, min(8, args.max_gpus))]
    if not selected:
        raise RuntimeError("BLOCKED_INFRASTRUCTURE_NO_ELIGIBLE_H20")
    registry = load_json(EP0 / "model_registry.json")
    models = []
    for model in registry["models"]:
        models.append({
            "model_id": model["model_id"], "timm_name": model["timm_name"],
            "checkpoint": model["checkpoint"]["path"], "checkpoint_sha256": model["checkpoint"]["sha256"],
            "checkpoint_bytes": model["checkpoint"]["bytes"], "depth": model["depth"],
            "embedding_dim": model["embedding_dim"],
        })
    tasks = []
    for repeat in range(4):
        for model in models:
            blocks = [0, 3, 6] if repeat == 0 else [1, 4, 7] if repeat == 1 else [2, 5, 8] if repeat == 2 else [9, 10, 11]
            decomposition_blocks = blocks
            tasks.append({"task_id": f"{model['model_id']}__synthetic_{repeat}", "payload": {**model, "repeat": repeat, "blocks": blocks, "decomposition_blocks": decomposition_blocks}})
    queues = {"single": output / "single_queue.sqlite3", "multi": output / "multi_queue.sqlite3"}
    queue_identities = {name: create_queue(path, tasks) for name, path in queues.items()}
    write_json_new(output / "task_manifest.json", {
        "tasks": tasks, "queue_identities": queue_identities, "task_count": len(tasks),
        "manifest_digest": digest(tasks), "seed_rule": "SHA256(namespace|task_id) first8 bytes",
        "outcome_data_access": False, "inputs": "deterministic synthetic normalized-coordinate tensors",
    })
    write_json_new(output / "startup_snapshot.json", {
        "hostname": socket.gethostname(), "snapshot": snapshot, "selected": selected,
        "free_disk_bytes": shutil.disk_usage(REPO).free, "minimum_free_disk_bytes": 50 * 1024 ** 3,
        "git_head": head, "ep0_validation_stdout": validation.stdout.splitlines(),
        "CUBLAS_WORKSPACE_CONFIG_before_workers": ":4096:8",
    })
    if shutil.disk_usage(REPO).free < 50 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_DISK")

    lock_root = RUNTIME / "gpu_locks"; lock_root.mkdir(parents=True, exist_ok=True)
    locks = []
    locked_selected = []
    for gpu in selected:
        stream = (lock_root / f"{gpu['gpu_uuid']}.lock").open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close(); continue
        stream.seek(0); stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "gpu_uuid": gpu["gpu_uuid"], "output": str(output)})); stream.flush(); os.fsync(stream.fileno())
        locks.append(stream); locked_selected.append(gpu)
    selected = locked_selected
    if not selected:
        raise RuntimeError("BLOCKED_INFRASTRUCTURE_NO_LOCKABLE_H20")
    try:
        single_exits, single_seconds = launch_workers(queues["single"], output / "single_results", selected[:1], output / "logs/single")
        multi_exits, multi_seconds = launch_workers(queues["multi"], output / "multi_results", selected, output / "logs/multi")
    finally:
        for stream in locks:
            fcntl.flock(stream, fcntl.LOCK_UN); stream.close()
    exits = {"single": single_exits, "multi": multi_exits}
    coverages = {
        "single": audit_coverage(queues["single"], output / "single_results"),
        "multi": audit_coverage(queues["multi"], output / "multi_results"),
    }
    write_json_new(output / "coverage_audit.json", coverages)
    result_sets = {name: {load_json(path)["task_id"]: load_json(path)["result"] for path in sorted((output / f"{name}_results").rglob("*.json"))} for name in queues}
    results = list(result_sets["multi"].values())
    if any(row["return_code"] for group in exits.values() for row in group) or any(row["status"] != "PASS" for row in coverages.values()):
        raise RuntimeError("GPU_WORKER_OR_COVERAGE_FAILED")
    equivalent = {task: result_sets["single"][task]["numeric_digest"] == result_sets["multi"][task]["numeric_digest"] for task in sorted(result_sets["single"])}
    if set(result_sets["single"]) != set(result_sets["multi"]) or not all(equivalent.values()):
        raise RuntimeError("SINGLE_MULTI_RESULT_MISMATCH")
    single_results = list(result_sets["single"].values())
    latencies = sorted(row["scan_seconds"] for row in results)
    single_latencies = sorted(row["scan_seconds"] for row in single_results)
    p95 = latencies[-1]
    # With four deterministic engineering tasks, report empirical max as the conservative P95 proxy.
    summary = {
        "status": "PASS_EP1_IMPLEMENTATION_AND_REGRESSION", "scientific_results": "NOT_RUN",
        "production_outcome_access": False, "ep2_started": False, "ep0_commit": EP0_COMMIT,
        "executed_head": head, "cpu_tests": tests.stdout.strip(), "gpu_workers": len(selected),
        "worker_exits": exits, "task_coverage": coverages, "single_multi_numeric_equivalence": equivalent,
        "model_ids": sorted({row["model_id"] for row in results}),
        "tensor_digests": {row["model_id"]: row["model_binding"]["tensor_digest"] for row in results},
        "target_counts": {row["model_id"]: row["target_count"] for row in results},
        "all_block_coverage": sorted(set(block for row in results for block in row["blocks"])) == list(range(12)),
        "fused_unfused_max_abs": max(row["fused_unfused_max_abs"] for row in results if row["fused_unfused_max_abs"] is not None),
        "hook_max_abs": max(max(row["hook_parity_max_abs"], row["formal_hook_parity_max_abs"]) for row in results),
        "closure_max_abs": max(row["closure_max_abs"] for row in results),
        "closure_relative_l2": max(row["closure_relative_l2"] for row in results),
        "cost_axes_exercised": {
            "K": sorted({1} | {row["target_k"] for row in results}),
            "M": sorted({item["M"] for row in results for item in row["ig_benchmarks"]}),
            "C": sorted({value for row in results for value in row["component_counts"]}),
        },
        "scan_seconds": {"values": latencies, "p50": (latencies[1] + latencies[2]) / 2, "p95_conservative_max": p95, "max": latencies[-1]},
        "peak_reserved_gpu_bytes_max": max(row["peak_reserved_gpu_bytes"] for row in results),
        "worker_wall_seconds": {"single": single_seconds, "multi": multi_seconds},
        "throughput": {
            "single_tasks_per_second": len(tasks) / single_seconds,
            "multi_tasks_per_second": len(tasks) / multi_seconds,
            "single_task_latency_p50": (single_latencies[3] + single_latencies[4]) / 2,
            "single_task_latency_p95_conservative_max": single_latencies[-1],
            "single_task_latency_max": single_latencies[-1],
            "single_peak_reserved_gpu_bytes_max": max(row["peak_reserved_gpu_bytes"] for row in single_results),
        },
        "gpu_hours_upper_observed": (single_seconds + multi_seconds * len(selected)) / 3600,
        "retained_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "implementation_bindings": [{"path": relative, "sha256": sha256(REPO / relative)} for relative in [
            "src/vitnns/endpoint_nns.py", "src/vitnns/endpoint_vit.py",
            "src/vitnns/endpoint_ep1_queue.py", "scripts/run_endpoint_ep1.py",
            "scripts/run_endpoint_ep1_worker.py", "scripts/validate_endpoint_ep1.py",
            "scripts/validate_endpoint_vit_scan.py",
            "tests/test_endpoint_nns.py", "tests/test_endpoint_vit.py",
            "tests/test_endpoint_ep1_queue.py",
        ]],
        "ep1_checklist": [
            "paper_formula_CNN_GxI", "CNN_relu_off_unique_winner",
            "energy_weight_target_same_tensor", "all_block_each_target_kind",
            "residual_wrong_total_gradient_false_pass", "nonzero_head_omission",
            "nonzero_routing_omission", "QKV_branch_vs_total_scope",
            "head_projection_bias_downstream", "forward_no_hook_noop_hook_parity",
            "finite_difference_JVP", "local_direction_correctness",
            "float64_float32_absolute_relative", "single_multi_worker_equivalence",
            "expired_attempt_reject", "crash_exact_coverage",
            "immutable_results", "P95_single_gpu_and_multi_gpu_throughput",
        ],
        "next_authorization": "EP2_REQUIRES_EXPLICIT_USER_REQUEST_AND_PUBLISHED_EP1_PASS",
    }
    write_json_new(output / "validation_report.json", summary)
    files = [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(output.rglob("*")) if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}]
    manifest = {"files": files, "canonical_digest": digest(files)}
    write_json_new(output / "artifact_manifest.json", manifest)
    terminal = {
        "status": summary["status"], "scientific_results": "NOT_RUN",
        "production_outcome_access": False, "ep2_started": False,
        "validation_digest": digest(summary), "manifest_digest": manifest["canonical_digest"],
        "next_authorization": summary["next_authorization"], "elapsed_seconds": time.monotonic() - started,
    }
    write_json_new(output / "terminal_receipt.json", terminal)
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
