#!/usr/bin/env python3
"""Prepare and launch the frozen revision-v3 Step-2 production queue."""
import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_ep1_queue import create_queue, digest
from vitnns.endpoint_revision_v3_step2 import independently_sampled_channels, independently_sampled_tokens

SPEC = REPO / "config/nns_vit_endpoint/revision_v3_step2.json"
ROOT = REPO / "reports/nns_vit_endpoint_revision_v3_step2_20260921"
SOURCE_FILES = (
    "config/nns_vit_endpoint/revision_v3_step2.json",
    "docs/Endpoint修改意见v3第二步实验协议与结果_20260921.md",
    "src/vitnns/endpoint_revision_v3_step2.py",
    "scripts/run_endpoint_revision_v3_step2.py",
    "scripts/run_endpoint_revision_v3_step2_worker.py",
    "scripts/analyze_endpoint_revision_v3_step2.py",
    "scripts/validate_endpoint_revision_v3_step2.py",
    "tests/test_endpoint_revision_v3_step2.py",
)


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def prepare() -> None:
    spec = load(SPEC)
    if ROOT.exists():
        raise ValueError("FRESH_ROOT_REQUIRED")
    for relative, expected in spec["source_bindings"].items():
        if sha(REPO / relative) != expected:
            raise ValueError("SOURCE_BINDING_DRIFT:" + relative)
    m1_terminal = load(REPO / "reports/nns_vit_endpoint_m1_r1_20260918/terminal_receipt.json")
    m3_terminal = load(REPO / "reports/nns_vit_endpoint_m3_r1_20260919/terminal_receipt.json")
    if m1_terminal["status"] != "ENDPOINT_M1_PASS_EXECUTION_AND_AUDIT":
        raise ValueError("M1_NOT_PASS")
    if m3_terminal["status"] != "ENDPOINT_M3_PASS_EXECUTION_AND_AUDIT":
        raise ValueError("M3_NOT_PASS")

    m1_tasks = load(REPO / "reports/nns_vit_endpoint_m1_r1_20260918/task_manifest.json")["tasks"]
    primary = []
    for source in m1_tasks:
        payload = dict(source["payload"])
        task_id = "primary__" + source["task_id"]
        payload.update({
            "task_id": task_id, "kind": "primary",
            "alphas": spec["multi_amplitude"]["alphas"],
            "blocks": spec["qkv_population"]["blocks"],
        })
        primary.append({"task_id": task_id, "payload": payload})

    namespaces = [
        f"{spec['independent_sampling']['group_namespace_prefix']}|{index}" for index in range(1, 6)
    ]
    widths = {"vit_b16": 768, "deit_s16": 384}
    group_registry = {}
    for model_id in spec["models"]:
        group_registry[model_id] = []
        for index, namespace in enumerate(namespaces, 1):
            group_registry[model_id].append({
                "group_id": f"sample_{index}", "namespace": namespace,
                "tokens": list(independently_sampled_tokens(namespace)),
                "channels": {
                    str(block): list(independently_sampled_channels(
                        namespace, model_id, block, widths[model_id],
                        spec["independent_sampling"]["channels_per_block"],
                    )) for block in spec["independent_sampling"]["blocks"]
                },
            })

    m3_tasks = load(REPO / "reports/nns_vit_endpoint_m3_r1_20260919/task_manifest.json")["tasks"]
    sampling = []
    for source in m3_tasks:
        payload = dict(source["payload"])
        task_id = "sampling__" + source["task_id"]
        payload.update({
            "task_id": task_id, "kind": "sampling",
            "groups": group_registry[payload["model_id"]],
        })
        sampling.append({"task_id": task_id, "payload": payload})

    tasks = sorted(primary + sampling, key=lambda row: row["task_id"])
    if len(primary) != 800 or len(sampling) != 200 or len(tasks) != 1000:
        raise ValueError("TASK_SCOPE")
    ROOT.mkdir(parents=True)
    write_new(ROOT / "sampling_registry.json", {
        "groups": group_registry, "group_count": 5,
        "digest": digest(group_registry),
    })
    write_new(ROOT / "task_manifest.json", {
        "tasks": tasks, "primary_tasks": 800, "sampling_tasks": 200,
        "sampling_group_evaluations": 1000, "digest": digest(tasks),
    })
    source_hashes = {path: sha(REPO / path) for path in SOURCE_FILES}
    write_new(ROOT / "freeze.json", {
        "stage": spec["stage"], "status": "FROZEN", "time_ns": time.time_ns(),
        "producer_commit": git("rev-parse", "HEAD"), "sources": source_hashes,
        "source_bindings": spec["source_bindings"],
        "task_manifest_sha256": sha(ROOT / "task_manifest.json"),
        "sampling_registry_sha256": sha(ROOT / "sampling_registry.json"),
    })
    create_queue(ROOT / "queue.sqlite3", tasks)
    print(json.dumps({"status": "STEP2_PREPARED", "tasks": len(tasks), "groups": 5}))


def gpu_candidates(spec: dict) -> tuple[list[dict], str, str]:
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    pids = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in query.splitlines():
        index, uuid, name, free, utilization = [part.strip() for part in line.split(",")]
        if (name == "NVIDIA H20" and int(free) >= spec["runtime"]["min_free_mib"] and
                int(utilization) <= spec["runtime"]["max_utilization_percent"]):
            rows.append({"index": index, "uuid": uuid, "name": name,
                         "free_mib": int(free), "utilization": int(utilization)})
    return rows, query, pids


def run() -> None:
    spec = load(SPEC); freeze = load(ROOT / "freeze.json")
    for relative, expected in freeze["sources"].items():
        if sha(REPO / relative) != expected:
            raise ValueError("FROZEN_SOURCE_DRIFT:" + relative)
    if sha(ROOT / "task_manifest.json") != freeze["task_manifest_sha256"]:
        raise ValueError("TASK_MANIFEST_DRIFT")
    head = git("rev-parse", "HEAD"); upstream = git("rev-parse", "@{upstream}")
    branch = git("symbolic-ref", "--short", "HEAD")
    remote = git("ls-remote", "origin", "refs/heads/" + branch).split()[0]
    if head != upstream or head != remote or freeze["producer_commit"] != head:
        raise ValueError("PUBLISHED_PRODUCER_REQUIRED")
    prefix = "vitnns_rebuild/"
    for relative, expected in freeze["sources"].items():
        data = subprocess.check_output(["git", "show", head + ":" + prefix + relative], cwd=REPO)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("PUBLISHED_SOURCE_DRIFT:" + relative)

    candidates, snapshot, pids = gpu_candidates(spec)
    locks = []; selected = []
    lock_root = REPO.parent / ".runtime/endpoint_shared_gpu_locks"; lock_root.mkdir(parents=True, exist_ok=True)
    for row in candidates:
        stream = (lock_root / (row["uuid"] + ".lock")).open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close(); continue
        locks.append(stream); selected.append(row)
        if len(selected) == spec["runtime"]["gpus_max"]:
            break
    if not selected:
        raise ValueError("NO_ELIGIBLE_GPU")
    write_new(ROOT / "startup.json", {
        "producer_commit": head, "tracking_commit": upstream, "remote_commit": remote,
        "selected": selected, "gpu_snapshot": snapshot, "existing_pids": pids,
        "sys_executable_realpath": str(Path(sys.executable).resolve()),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    processes = []; started = time.monotonic()
    try:
        for row in selected:
            worker_id = "gpu" + row["index"]
            log = (ROOT / (worker_id + ".log")).open("xb")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=row["index"], CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1")
            command = [
                sys.executable, "scripts/run_endpoint_revision_v3_step2_worker.py",
                "--root", str(ROOT), "--worker-id", worker_id,
                "--physical-index", row["index"], "--gpu-uuid", row["uuid"],
            ]
            processes.append((subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        exits = [process.wait() for process, _log in processes]
    finally:
        for _process, log in processes:
            log.close()
        for stream in locks:
            fcntl.flock(stream, fcntl.LOCK_UN); stream.close()
    with sqlite3.connect(ROOT / "queue.sqlite3") as connection:
        states = dict(connection.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state"))
    passed = all(code == 0 for code in exits) and states == {"completed": 1000}
    write_new(ROOT / "acquisition_receipt.json", {
        "status": "PASS" if passed else "FAIL", "states": states, "exits": exits,
        "wall_seconds": time.monotonic() - started, "gpu_count": len(selected),
    })
    print(json.dumps(load(ROOT / "acquisition_receipt.json")))
    if not passed:
        raise ValueError("ACQUISITION_FAILED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose exactly one of --prepare or --run")
    prepare() if args.prepare else run()
