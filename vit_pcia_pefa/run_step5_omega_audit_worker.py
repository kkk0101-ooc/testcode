#!/usr/bin/env python3
"""Run the frozen float64 complete-cut Omega audit queue."""
from __future__ import annotations
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]

from vit_pcia_pefa.run_step4_attempt1_worker import load_image, load_model  # noqa: E402
from vit_pcia_pefa.run_step5_worker import abandon, json_load, verify_binding  # noqa: E402
from vit_pcia_pefa.src.pilot import omega_complete_cut_audit  # noqa: E402
from vit_pcia_pefa.src.step5_queue import claim_task, complete_task, heartbeat  # noqa: E402
from vit_pcia_pefa.src.vit_adapter import port_layout  # noqa: E402


def execute(model, spec, task, claim, queue: Path, protocol: dict, device: torch.device) -> dict:
    started = time.monotonic()
    last_heartbeat = started

    def progress() -> None:
        nonlocal last_heartbeat
        now = time.monotonic()
        if now - started > protocol["resources"]["task_timeout_seconds"]:
            raise RuntimeError("PAUSED_RESOURCE_CAP:TASK_TIMEOUT")
        if now - last_heartbeat >= protocol["resources"]["heartbeat_seconds"]:
            heartbeat(queue, claim, lease_seconds=protocol["resources"]["lease_seconds"])
            last_heartbeat = now

    image = load_image({
        "cache_path": task["image_path"], "image_bytes": task["image_bytes"],
        "image_sha256": task["image_sha256"],
    }, spec, device)[0].double()
    model = model.double()
    blocks = {}
    for block in protocol["blocks"]:
        declared = tuple(port_layout(model.blocks[block].attn.num_heads).cuts[protocol["primary_cut"]])
        audit = omega_complete_cut_audit(
            model, image, block_index=block, class_id=int(task["class_id"]),
            seed=int(claim["seed"]) + int(block), declared_cut_component_names=declared,
        )
        if not audit["pass"]:
            raise RuntimeError("FAIL_QUADRATURE_CLOSURE:OMEGA_FLOAT64_COMPLETE_CUT")
        blocks[str(block)] = audit
        progress()
    torch.cuda.synchronize()
    return {
        "status": "PASS_OMEGA_FLOAT64_AUDIT",
        "task_id": claim["task_id"],
        "model_id": task["model_id"],
        "sample_id": task["sample_id"],
        "split": task["split"],
        "split_order": int(task["split_order"]),
        "class_id": int(task["class_id"]),
        "image_sha256": task["image_sha256"],
        "task_seed": int(claim["seed"]),
        "seed_namespace": protocol["seed_namespace"],
        "blocks": blocks,
        "model_dtype": "float64",
        "scientific_omega_scores_recomputed": False,
        "tolerance_changed": False,
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
        "step6_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    verify_binding(args.root / "source_binding.json")
    protocol = json_load(args.root / "protocol.json")
    queue = args.root / "omega_audit_queue.sqlite3"
    results = args.root / "omega_audit_results"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    probe = torch.arange(16, device=device, dtype=torch.float64).reshape(4, 4).requires_grad_()
    (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize()
    registry = json_load(REPO / protocol["model_registry_path"])
    loaded = {}
    while True:
        claim = claim_task(
            queue, worker_id=args.worker_id, device_uuid=args.gpu_uuid,
            lease_seconds=protocol["resources"]["lease_seconds"],
            max_attempts=protocol["resources"]["max_attempts"],
        )
        if claim is None:
            break
        try:
            task = claim["payload"]
            model_id = task["model_id"]
            if model_id not in loaded:
                spec = next(row for row in registry["models"] if row["model_id"] == model_id)
                loaded[model_id] = (load_model(spec, device)[0], spec)
            model, spec = loaded[model_id]
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            result = execute(model, spec, task, claim, queue, protocol, device)
            if result["resources"]["peak_reserved_gpu_bytes"] > protocol["resources"]["gpu_memory_cap_bytes"]:
                raise RuntimeError("PAUSED_RESOURCE_CAP:GPU_MEMORY")
            complete_task(queue, results, claim, result)
            print(json.dumps({"status": result["status"], "task_id": claim["task_id"],
                              "seconds": result["resources"]["wall_seconds"]}), flush=True)
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc(), "time_ns": time.time_ns()}
            abandon(queue, claim, failure)
            print(json.dumps({"status": "OMEGA_AUDIT_FAILED", "task_id": claim["task_id"],
                              "attempt": claim["attempt"], **failure}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
