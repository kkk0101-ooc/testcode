#!/usr/bin/env python3
"""One deterministic GPU worker for the frozen Endpoint M4 queue."""
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import json
import math
import resource
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from run_endpoint_e4_intervention_worker import (  # noqa: E402
    abandon_task, load_inputs, load_model, worker_identity,
)
from vitnns.endpoint_e2_atlas import file_sha256  # noqa: E402
from vitnns.endpoint_ep1_queue import claim_task, complete_task, digest, heartbeat  # noqa: E402
from vitnns.endpoint_m4 import scan_m4_task  # noqa: E402
import torch  # noqa: E402


def load_envelope(path: Path, expected_sha: str) -> dict:
    if file_sha256(path) != expected_sha:
        raise ValueError("E3_SOURCE_SHA")
    value = json.loads(path.read_text())
    unsigned = dict(value); expected = unsigned.pop("canonical_digest")
    if digest(unsigned) != expected:
        raise ValueError("E3_SOURCE_CANONICAL")
    return value["result"]


def crosscheck_e3(scan: dict, source: dict) -> dict:
    if scan["full_endpoint_sha256"] != source["blocks"][0]["reconstructions"]["L_F"]["array_sha256"]:
        # Different VJP batching can change float32 bytes, so numeric comparison below is authoritative.
        byte_identity = False
    else:
        byte_identity = True
    maximum_ratio = 0.0
    checked = 0
    for block, row in enumerate(scan["block_validation"]):
        reference = source["blocks"][block]["components"]
        for name in ("query", "key", "routing", "value", "attention", "identity", "carry", "mlp"):
            for field in ("signed_sum", "absolute_mass", "l2_norm"):
                observed = float(row["components"][name][field])
                expected = float(reference[name][field])
                ratio = abs(observed - expected) / (2e-5 + 2e-4 * abs(expected))
                maximum_ratio = max(maximum_ratio, ratio); checked += 1
                if not math.isfinite(ratio) or ratio > 1:
                    raise ValueError(f"E3_NUMERIC_CROSSCHECK:{block}:{name}:{field}:{ratio}")
    return {"status": "PASS", "scalar_checks": checked,
            "maximum_tolerance_ratio": maximum_ratio, "full_map_byte_identity": byte_identity}


def execute(model, task: dict, claim: dict, queue: Path, device) -> dict:
    started = time.perf_counter()
    if shutil.disk_usage(REPO).free < 50 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_DISK")
    image, _references, original_size = load_inputs(task, device)
    source = load_envelope(REPO / task["e3_path"], task["e3_sha256"])
    last_heartbeat = time.monotonic()

    def progress():
        nonlocal last_heartbeat
        if time.monotonic() - last_heartbeat >= 25:
            heartbeat(queue, claim, lease_seconds=1800); last_heartbeat = time.monotonic()

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    scan = scan_m4_task(
        model, image, class_id=task["class_id"], task_id=claim["task_id"],
        detach_validation=bool(task["detach_validation"]),
        chunk_size=int(task["vjp_chunk_size"]), progress=progress,
    )
    e3_crosscheck = crosscheck_e3(scan, source)
    torch.cuda.synchronize(); seconds = time.perf_counter() - started
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if seconds > 1800:
        raise RuntimeError("TASK_TIMEOUT")
    return {
        "stage": "ENDPOINT-M4", "claim_scope": "LOCAL_MECHANISM_DESCRIPTIVE_NOT_CAUSAL_IMPORTANCE",
        "model_id": task["model_id"], "sample_id": task["sample_id"],
        "split_order": task["split_order"], "class_id": task["class_id"],
        "wnid": task["wnid"], "image_sha256": task["image_sha256"],
        "image_bytes": task["image_bytes"], "original_size_wh": original_size,
        "input_shape": list(image.shape), "dtype": str(image.dtype),
        "e3_crosscheck": e3_crosscheck, **scan, "task_seconds": seconds,
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
        "peak_cpu_rss_bytes": rss,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    identity = worker_identity(device, args); models = {}; completed = 0
    while True:
        claim = claim_task(args.queue, worker_id=args.worker_id, device_uuid=args.gpu_uuid,
                           lease_seconds=1800, max_attempts=2)
        if claim is None:
            break
        task = claim["payload"]
        try:
            if task["model_id"] not in models:
                models[task["model_id"]] = load_model(task, device)
            model, binding = models[task["model_id"]]
            result = execute(model, task, claim, args.queue, device)
            result.update(model_binding=binding, worker_identity=identity)
            complete_task(args.queue, args.results, claim, result); completed += 1
            print(json.dumps({"task": claim["task_id"], "seconds": result["task_seconds"]}), flush=True)
        except Exception as error:
            failure = {"error_type": type(error).__name__, "message": str(error)}
            abandon_task(args.queue, claim, failure)
            print(json.dumps({"status": "TASK_FAILED", "task": claim["task_id"], **failure}), flush=True)
    print(json.dumps({"status": "WORKER_EXIT", "worker_id": args.worker_id, "completed": completed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
