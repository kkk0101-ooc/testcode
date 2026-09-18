#!/usr/bin/env python3
"""One deterministic ENDPOINT-E5 GPU worker."""
from __future__ import annotations

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import json
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from run_endpoint_e4_intervention_worker import (  # noqa: E402
    abandon_task, load_inputs, load_model, worker_identity,
)
from vitnns.endpoint_e2_atlas import file_sha256  # noqa: E402
from vitnns.endpoint_e5_spatial import scan_e5_task  # noqa: E402
from vitnns.endpoint_ep1_queue import claim_task, complete_task, heartbeat  # noqa: E402


def write_raw_maps(path: Path, arrays: dict[str, np.ndarray]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    records = []
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(arrays):
            raise RuntimeError("RAW_MAP_ARCHIVE_KEYS")
        for name in sorted(loaded.files):
            value = loaded[name]
            if value.shape != (1, 3, 224, 224) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise RuntimeError("RAW_MAP_ARRAY_CONTRACT")
            records.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                            "array_sha256": __import__("hashlib").sha256(value.tobytes()).hexdigest()})
    return {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size,
            "sha256": file_sha256(path), "arrays": records}


def execute(model, task: dict, claim: dict, queue: Path, result_root: Path, device: torch.device) -> dict:
    task_started = time.perf_counter()
    if shutil.disk_usage(REPO).free < 50 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_DISK")
    image, references, original_size = load_inputs(task, device)
    reference_identity = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
        "tensor_sha256": __import__("hashlib").sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
        for name, value in references.items()}
    last_heartbeat = time.monotonic()

    def progress() -> None:
        nonlocal last_heartbeat
        if time.monotonic() - last_heartbeat >= 25:
            heartbeat(queue, claim, lease_seconds=180); last_heartbeat = time.monotonic()

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
    scan = scan_e5_task(model, image, references, class_id=task["class_id"], task_seed=claim["seed"],
        model_id=task["model_id"], perturbed_batch_size=task["perturbed_batch_size"],
        isolation_batch_size=task["isolation_batch_size"], retain_raw_maps=task["retain_raw_maps"],
        progress=progress)
    torch.cuda.synchronize(); scan_seconds = time.perf_counter() - started
    raw_arrays = scan.pop("raw_maps")
    raw_record = None
    if raw_arrays is not None:
        raw_record = write_raw_maps(
            result_root / "raw_maps" / "attempts" / claim["task_id"] / f"attempt-{claim['attempt']:02d}.npz",
            raw_arrays,
        )
    peak_allocated = int(torch.cuda.max_memory_allocated()); peak_reserved = int(torch.cuda.max_memory_reserved())
    rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    task_seconds = time.perf_counter() - task_started
    if task_seconds > 1800:
        raise RuntimeError("TASK_TIMEOUT")
    if peak_reserved > 24 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_GPU_MEMORY")
    if rss_bytes > 32 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_CPU_RSS")
    return {"stage": "ENDPOINT-E5", "claim_scope": "SPATIAL_PER_IMAGE_INPUTS_NO_GLOBAL_INFERENCE_OR_CLAIM_DECISION",
        "model_id": task["model_id"], "sample_id": task["sample_id"], "split_order": task["split_order"],
        "class_id": task["class_id"], "wnid": task["wnid"], "image_sha256": task["image_sha256"],
        "image_bytes": task["image_bytes"], "original_size_wh": original_size,
        "input_shape": list(image.shape), "dtype": str(image.dtype), "reference_identity": reference_identity,
        **scan, "raw_maps_record": raw_record, "scan_seconds": scan_seconds, "task_seconds": task_seconds,
        "peak_allocated_gpu_bytes": peak_allocated, "peak_reserved_gpu_bytes": peak_reserved,
        "peak_cpu_rss_bytes": rss_bytes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True); parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--worker-id", required=True); parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True); args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0"); identity = worker_identity(device, args)
    models = {}; completed = 0
    while True:
        claim = claim_task(args.queue, worker_id=args.worker_id, device_uuid=args.gpu_uuid,
                           lease_seconds=180, max_attempts=2)
        if claim is None:
            break
        task = claim["payload"]
        try:
            if task["model_id"] not in models:
                models[task["model_id"]] = load_model(task, device)
            model, binding = models[task["model_id"]]
            result = execute(model, task, claim, args.queue, args.results, device)
            result.update({"model_binding": binding, "worker_identity": identity})
            complete_task(args.queue, args.results, claim, result); completed += 1
        except Exception as error:
            failure = {"error_type": type(error).__name__, "message": str(error)}
            abandon_task(args.queue, claim, failure)
            print(json.dumps({"status": "TASK_FAILED", "task_id": claim["task_id"],
                              "attempt": claim["attempt"], **failure}, sort_keys=True), flush=True)
    print(json.dumps({"status": "WORKER_EXIT", "worker_id": args.worker_id, "completed": completed}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
