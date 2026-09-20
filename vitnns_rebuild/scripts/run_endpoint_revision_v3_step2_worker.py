#!/usr/bin/env python3
"""Deterministic GPU worker for revision v3 Step 2 experiments."""
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import hashlib
import json
import resource
import sqlite3
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from run_endpoint_e4_intervention_worker import load_inputs, load_model, worker_identity
from vitnns.endpoint_ep1_queue import claim_task, complete_task, heartbeat
from vitnns.endpoint_revision_v3_step2 import (
    scan_independent_sampling_groups, scan_multi_amplitude, scan_qkv_population,
)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def abandon(queue: Path, claim: dict, error: dict) -> None:
    with sqlite3.connect(queue, isolation_level=None, timeout=30) as connection:
        connection.execute("PRAGMA journal_mode=WAL"); connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE attempts SET status='FAILED',completed_ns=? WHERE task_id=? AND attempt=?",
            (time.time_ns(), claim["task_id"], claim["attempt"]),
        )
        connection.execute(
            "UPDATE tasks SET state='failed',fencing_token=NULL,worker_id=NULL,device_uuid=NULL,"
            "lease_until_ns=NULL,heartbeat_ns=NULL WHERE task_id=? AND attempt=? AND fencing_token=?",
            (claim["task_id"], claim["attempt"], claim["fencing_token"]),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            (f"failure:{claim['task_id']}:{claim['attempt']}", json.dumps(error, sort_keys=True)),
        )
        connection.commit()


def primary_task(model, image, references, task: dict, root: Path, progress) -> dict:
    root = root.resolve()
    amplitude, amplitude_validation = scan_multi_amplitude(
        model, image, references, task["class_id"], task["alphas"], progress=progress,
    )
    qkv = scan_qkv_population(model, image, task["class_id"], task["blocks"], progress=progress)
    destination = root / "payloads" / task["task_id"]
    destination.mkdir(parents=True, exist_ok=False)
    amplitude_path = destination / "multi_amplitude.npz"
    qkv_path = destination / "qkv_population.npz"
    np.savez_compressed(amplitude_path, **amplitude); np.savez_compressed(qkv_path, **qkv)

    e4_path = REPO / task["e4_path"]
    if sha(e4_path) != task["e4_sha256"]:
        raise ValueError("E4_SOURCE_SHA")
    e4 = load(e4_path)["result"]
    source = {
        (reference, int(row["block"]), int(row["head"])): float(row["effect"])
        for reference, value in e4["finite_utility"].items()
        for row in value["components"] if row["operation"] == "head"
    }
    alpha_one = np.isclose(amplitude["alpha"], 1.0, atol=0, rtol=0)
    parity = [
        abs(float(effect) - source[(str(reference), int(block), int(head))])
        for reference, block, head, effect in zip(
            amplitude["reference_id"][alpha_one], amplitude["block"][alpha_one],
            amplitude["head"][alpha_one], amplitude["finite_effect"][alpha_one],
        )
    ]
    amplitude_validation["alpha_one_source_max_abs"] = max(parity)
    parity_ratios = []
    for reference, block, head, effect in zip(
            amplitude["reference_id"][alpha_one], amplitude["block"][alpha_one],
            amplitude["head"][alpha_one], amplitude["finite_effect"][alpha_one]):
        expected = source[(str(reference), int(block), int(head))]
        parity_ratios.append(abs(float(effect) - expected) / (1e-4 + 1e-4 * abs(expected)))
    amplitude_validation["alpha_one_source_max_tolerance_ratio"] = max(parity_ratios)
    amplitude_validation["alpha_one_source_tolerance"] = "1e-4 + 1e-4*abs(sealed_effect)"
    if amplitude_validation["alpha_one_source_max_tolerance_ratio"] > 1.0:
        raise ValueError("ALPHA_ONE_SOURCE_PARITY")

    e3_path = REPO / task["e3_path"]
    if sha(e3_path) != task["e3_sha256"]:
        raise ValueError("E3_SOURCE_SHA")
    e3 = load(e3_path)["result"]
    source_qkv = {
        (int(block["block"]), path): block["components"][name]
        for block in e3["blocks"] for path, name in (("Q", "query"), ("K", "key"), ("V", "value"))
    }
    qkv_errors = []
    for block, path, absolute, signed in zip(qkv["block"], qkv["path"], qkv["absolute_mass"], qkv["signed_sum"]):
        expected = source_qkv[(int(block), str(path))]
        qkv_errors.extend((abs(float(absolute) - expected["absolute_mass"]), abs(float(signed) - expected["signed_sum"])))
    if max(qkv_errors) > 2e-3:
        raise ValueError("QKV_SOURCE_PARITY")
    return {
        "amplitude_path": amplitude_path.relative_to(REPO).as_posix(),
        "amplitude_sha256": sha(amplitude_path), "amplitude_validation": amplitude_validation,
        "qkv_path": qkv_path.relative_to(REPO).as_posix(), "qkv_sha256": sha(qkv_path),
        "qkv_record_count": len(qkv["block"]), "qkv_source_max_abs": max(qkv_errors),
    }


def sampling_task(model, image, task: dict, root: Path, progress) -> dict:
    root = root.resolve()
    sampling = scan_independent_sampling_groups(
        model, image, task["groups"], chunk_size=64, progress=progress,
    )
    destination = root / "payloads" / task["task_id"]
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "independent_sampling.npz"
    np.savez_compressed(path, **sampling)
    expected = len(task["groups"]) * 12 * 16 * 32
    if len(sampling["block"]) != expected:
        raise ValueError("SAMPLING_COVERAGE")
    return {
        "sampling_path": path.relative_to(REPO).as_posix(), "sampling_sha256": sha(path),
        "sampling_record_count": len(sampling["block"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-index", type=int, required=True); parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0"); identity = worker_identity(device, args); models = {}
    while not (args.root / "STOP_FAILURE").exists():
        claim = claim_task(
            args.root / "queue.sqlite3", worker_id=args.worker_id, device_uuid=args.gpu_uuid,
            lease_seconds=600, max_attempts=2,
        )
        if claim is None:
            return
        task = claim["payload"]; started = time.monotonic(); last = started
        try:
            if task["model_id"] not in models:
                models[task["model_id"]] = load_model(task, device)
            model, binding = models[task["model_id"]]
            image, references, _ = load_inputs(task, device)

            def progress():
                nonlocal last
                if time.monotonic() - last > 30:
                    heartbeat(args.root / "queue.sqlite3", claim, lease_seconds=600); last = time.monotonic()

            torch.cuda.reset_peak_memory_stats()
            if task["kind"] == "primary":
                result = primary_task(model, image, references, task, args.root, progress)
            elif task["kind"] == "sampling":
                result = sampling_task(model, image, task, args.root, progress)
            else:
                raise ValueError("UNKNOWN_TASK_KIND")
            result.update({
                "stage": "ENDPOINT-REVISION-V3-STEP2", "kind": task["kind"],
                "model_id": task["model_id"], "sample_id": task["sample_id"],
                "split_order": task["split_order"], "image_sha256": task["image_sha256"],
                "model_binding": binding, "worker_identity": identity,
                "seconds": time.monotonic() - started,
                "peak_gpu_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            })
            complete_task(args.root / "queue.sqlite3", args.root / "results", claim, result)
            print(json.dumps({"task": claim["task_id"], "seconds": result["seconds"]}), flush=True)
        except Exception as error:
            detail = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            abandon(args.root / "queue.sqlite3", claim, detail)
            (args.root / f"failure_{claim['task_id']}.json").write_text(json.dumps(detail, indent=2) + "\n")
            (args.root / "STOP_FAILURE").touch(); raise


if __name__ == "__main__":
    main()
