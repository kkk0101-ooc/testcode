#!/usr/bin/env python3
"""Prospective R1 recovery: inherit 403 valid tasks and execute the remainder."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from run_endpoint_revision_v3_step2 import SOURCE_FILES as BASE_SOURCE_FILES, gpu_candidates, load, sha, write_new
from vitnns.endpoint_ep1_queue import _write_new, create_queue, digest

PARENT = REPO / "reports/nns_vit_endpoint_revision_v3_step2_20260921"
ROOT = REPO / "reports/nns_vit_endpoint_revision_v3_step2_r1_20260921"
SPEC = REPO / "config/nns_vit_endpoint/revision_v3_step2.json"
PARENT_BINDINGS = {
    "acquisition_receipt.json": "dfc7a512fae912c8c5c9850a43f8213bb152885fd7b60cd632f6281a5436f385",
    "failure_primary__vit_b16__000.json": "747a901bf93037e38c85cdcb6bc6dfa7994e0eaebe8bde202dcdc8556ef4ebd1",
    "task_manifest.json": "0fce654435af35320ffe8f8b4390db857fcfd7223c74282044565bbfeab6acf0",
    "freeze.json": "acfb52d619fbad83151da29349eccba2dbde56a5cbe22b6cb5fcac7fc8815ccb",
}
EXPECTED_PARENT_STATES = {"completed": 403, "failed": 1, "pending": 596}
EXPECTED_INHERITED = 403
SOURCE_FILES = (*BASE_SOURCE_FILES, "docs/Endpoint修改意见v3第二步R1恢复修正_20260921.md", "scripts/run_endpoint_revision_v3_step2_r1.py")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def prepare() -> None:
    if ROOT.exists():
        raise ValueError("FRESH_ROOT_REQUIRED")
    for name, expected in PARENT_BINDINGS.items():
        if sha(PARENT / name) != expected:
            raise ValueError("PARENT_BINDING_DRIFT:" + name)
    parent_receipt = load(PARENT / "acquisition_receipt.json")
    if parent_receipt["status"] != "FAIL" or parent_receipt["states"] != EXPECTED_PARENT_STATES:
        raise ValueError("PARENT_FAILURE_STATE")
    tasks = load(PARENT / "task_manifest.json")["tasks"]
    ROOT.mkdir(parents=True); create_queue(ROOT / "queue.sqlite3", tasks)
    shutil.copy2(PARENT / "sampling_registry.json", ROOT / "sampling_registry.json")
    write_new(ROOT / "task_manifest.json", load(PARENT / "task_manifest.json"))

    inherited = []
    with sqlite3.connect(PARENT / "queue.sqlite3") as parent_connection:
        completed = parent_connection.execute(
            "SELECT task_id,result_path,result_digest FROM tasks WHERE state='completed' ORDER BY task_id"
        ).fetchall()
    for task_id, result_path, old_digest in completed:
        document = load(PARENT / "results" / result_path)
        result = document["result"]
        old_prefix = PARENT.relative_to(REPO).as_posix() + "/"
        new_prefix = ROOT.relative_to(REPO).as_posix() + "/"
        for key in ("amplitude_path", "qkv_path", "sampling_path"):
            if key in result:
                source = REPO / result[key]
                destination = REPO / result[key].replace(old_prefix, new_prefix, 1)
                destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
                result[key] = destination.relative_to(REPO).as_posix()
        if result["kind"] == "primary":
            task = next(row["payload"] for row in tasks if row["task_id"] == task_id)
            amplitude = __import__("numpy").load(REPO / result["amplitude_path"])
            e4 = load(REPO / task["e4_path"])["result"]
            source_effects = {
                (reference, int(row["block"]), int(row["head"])): float(row["effect"])
                for reference, value in e4["finite_utility"].items()
                for row in value["components"] if row["operation"] == "head"
            }
            alpha_one = __import__("numpy").isclose(amplitude["alpha"], 1.0, atol=0, rtol=0)
            ratios = []
            for reference, block, head, effect in zip(
                    amplitude["reference_id"][alpha_one], amplitude["block"][alpha_one],
                    amplitude["head"][alpha_one], amplitude["finite_effect"][alpha_one]):
                expected = source_effects[(str(reference), int(block), int(head))]
                ratios.append(abs(float(effect) - expected) / (1e-4 + 1e-4 * abs(expected)))
            result["amplitude_validation"]["alpha_one_source_max_tolerance_ratio"] = max(ratios)
            result["amplitude_validation"]["alpha_one_source_tolerance"] = "1e-4 + 1e-4*abs(sealed_effect)"
            if max(ratios) > 1.0:
                raise ValueError("INHERITED_ALPHA_ONE_SOURCE_PARITY")
        rebound = dict(document); rebound.pop("canonical_digest", None); rebound["result"] = result
        rebound["canonical_digest"] = digest(rebound)
        destination = ROOT / "results" / result_path
        _write_new(destination, rebound)
        with sqlite3.connect(ROOT / "queue.sqlite3") as connection:
            connection.execute(
                "UPDATE tasks SET state='completed',attempt=1,fencing_token=1,worker_id='inherited-r1',"
                "device_uuid='parent',result_path=?,result_digest=? WHERE task_id=?",
                (result_path, rebound["canonical_digest"], task_id),
            )
            connection.execute(
                "INSERT INTO attempts(task_id,attempt,fencing_token,worker_id,device_uuid,claimed_ns,lease_until_ns,status,completed_ns) VALUES(?,?,?,?,?,?,?,?,?)",
                (task_id, 1, 1, "inherited-r1", "parent", 0, 0, "COMPLETED", 0),
            )
            connection.commit()
        inherited.append({
            "task_id": task_id, "parent_result_digest": old_digest,
            "rebound_result_digest": rebound["canonical_digest"],
            "payload_files": [
                {"path": result[key], "sha256": result[key.replace("path", "sha256")]}
                for key in ("amplitude_path", "qkv_path", "sampling_path") if key in result
            ],
        })
    if len(inherited) != EXPECTED_INHERITED:
        raise ValueError("INHERITED_COUNT")
    write_new(ROOT / "inheritance.json", {
        "status": "PASS", "parent_root": PARENT.relative_to(REPO).as_posix(),
        "parent_bindings": PARENT_BINDINGS, "inherited_count": len(inherited),
        "records": inherited, "records_digest": digest(inherited),
    })
    source_hashes = {path: sha(REPO / path) for path in SOURCE_FILES}
    write_new(ROOT / "freeze.json", {
        "stage": "ENDPOINT-REVISION-V3-STEP2-R1", "status": "FROZEN", "time_ns": time.time_ns(),
        "producer_commit": git("rev-parse", "HEAD"), "sources": source_hashes,
        "parent_bindings": PARENT_BINDINGS, "task_manifest_sha256": sha(ROOT / "task_manifest.json"),
        "sampling_registry_sha256": sha(ROOT / "sampling_registry.json"),
        "inheritance_sha256": sha(ROOT / "inheritance.json"),
    })
    print(json.dumps({"status": "STEP2_R1_PREPARED", "inherited": len(inherited), "remaining": len(tasks) - len(inherited)}))


def run() -> None:
    spec = load(SPEC); freeze = load(ROOT / "freeze.json")
    for relative, expected in freeze["sources"].items():
        if sha(REPO / relative) != expected:
            raise ValueError("FROZEN_SOURCE_DRIFT:" + relative)
    head = git("rev-parse", "HEAD"); upstream = git("rev-parse", "@{upstream}"); branch = git("symbolic-ref", "--short", "HEAD")
    remote = git("ls-remote", "origin", "refs/heads/" + branch).split()[0]
    if head != upstream or head != remote or freeze["producer_commit"] != head:
        raise ValueError("PUBLISHED_PRODUCER_REQUIRED")
    for relative, expected in freeze["sources"].items():
        data = subprocess.check_output(["git", "show", head + ":vitnns_rebuild/" + relative], cwd=REPO)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("PUBLISHED_SOURCE_DRIFT:" + relative)
    candidates, snapshot, pids = gpu_candidates(spec); locks = []; selected = []
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
        "sys_executable_realpath": str(Path(sys.executable).resolve()), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    processes = []; started = time.monotonic()
    try:
        for row in selected:
            worker_id = "gpu" + row["index"]; log = (ROOT / (worker_id + ".log")).open("xb")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=row["index"], CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "scripts/run_endpoint_revision_v3_step2_worker.py", "--root", str(ROOT), "--worker-id", worker_id, "--physical-index", row["index"], "--gpu-uuid", row["uuid"]]
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
    write_new(ROOT / "acquisition_receipt.json", {"status": "PASS" if passed else "FAIL", "states": states, "exits": exits, "wall_seconds": time.monotonic() - started, "gpu_count": len(selected), "inherited_tasks": EXPECTED_INHERITED, "newly_executed_tasks": 1000 - EXPECTED_INHERITED})
    print(json.dumps(load(ROOT / "acquisition_receipt.json")))
    if not passed:
        raise ValueError("ACQUISITION_FAILED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose exactly one of --prepare or --run")
    prepare() if args.prepare else run()
