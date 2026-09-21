#!/usr/bin/env python3
"""Execute the frozen Step-5 mechanism-extension dynamic queue."""
from __future__ import annotations
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import json
import platform
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import torchvision
import timm
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]

from vit_pcia_pefa.run_step4_attempt1_worker import load_image, load_model  # noqa: E402
from vit_pcia_pefa.run_step5_worker import abandon, json_load, verify_binding  # noqa: E402
from vit_pcia_pefa.src.pilot import stable_random_scores  # noqa: E402
from vit_pcia_pefa.src.step5_extension import (  # noqa: E402
    boundary_directed_point, local_loop_analysis, source_correlations,
    source_resolved_pefa, spatial_baselines_and_effects,
)
from vit_pcia_pefa.src.step5_extension_queue import (  # noqa: E402
    claim_task, complete_task, heartbeat,
)
from vitnns.endpoint_ep2 import gauss_legendre  # noqa: E402


def worker_identity(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "sys_executable_realpath": str(Path(sys.executable).resolve()),
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "timm": timm.__version__,
        "numpy": np.__version__, "Pillow": Image.__version__, "cuda": torch.version.cuda,
        "gpu_uuid": claim["device_uuid"],
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32": False, "autocast": False, "startup_matmul_backward": "PASS",
        "driver_version": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True,
        ).splitlines()[0].strip(),
    }


def execute(model, spec, task: dict[str, Any], claim: dict[str, Any], queue: Path,
            protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    started = time.monotonic(); last_heartbeat = started

    def progress() -> None:
        nonlocal last_heartbeat
        now = time.monotonic()
        if now - started > protocol["resources"]["task_timeout_seconds"]:
            raise RuntimeError("PAUSED_RESOURCE_CAP:TASK_TIMEOUT")
        if now - last_heartbeat >= protocol["resources"]["heartbeat_seconds"]:
            heartbeat(queue, claim, lease_seconds=protocol["resources"]["lease_seconds"])
            last_heartbeat = now

    task_seed = int(claim["seed"])
    random.seed(task_seed); np.random.seed(task_seed % (2**32))
    torch.manual_seed(task_seed); torch.cuda.manual_seed_all(task_seed)
    image, original_size = load_image({
        "cache_path": task["image_path"], "image_bytes": task["image_bytes"],
        "image_sha256": task["image_sha256"],
    }, spec, device)
    class_id = int(task["class_id"]); block = int(protocol["block"])
    heads = tuple(f"head_{index}" for index in range(model.blocks[block].attn.num_heads))
    baseline = torch.zeros_like(image)
    result: dict[str, Any] = {
        "status": "PASS_EXTENSION_TASK", "phase": task["phase"],
        "task_id": claim["task_id"], "model_id": task["model_id"],
        "sample_id": task["sample_id"], "split": task["split"],
        "split_order": int(task["split_order"]), "class_id": class_id,
        "image_sha256": task["image_sha256"], "original_size_wh": original_size,
        "task_seed": task_seed, "seed_namespace": protocol["seed_namespace"],
        "step6_started": False,
    }
    if task["phase"] == "E1_E2":
        loop_nodes_t, loop_weights_t = gauss_legendre(
            protocol["loop"]["edge_nodes"], device, torch.float32,
        )
        nodes = [float(item) for item in loop_nodes_t]
        weights = [float(item) for item in loop_weights_t]
        scales = protocol["loop"]["epsilon_l2"]
        clean = local_loop_analysis(
            model, image, block_index=block, class_id=class_id, head_names=heads,
            seed=task_seed, scales=scales, nodes=nodes, weights=weights, progress=progress,
        )
        mean = torch.tensor(spec["preprocessing"]["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(spec["preprocessing"]["std"], device=device).view(1, 3, 1, 1)
        boundary, boundary_receipt = boundary_directed_point(
            model, image, class_id=class_id, mean=mean, std=std,
            candidates_rgb=protocol["nonlinearity"]["boundary_epsilons_rgb"],
        )
        boundary_loop = local_loop_analysis(
            model, boundary, block_index=block, class_id=class_id, head_names=heads,
            seed=task_seed, scales=scales, nodes=nodes, weights=weights, progress=progress,
        )
        result.update({"clean_loop": clean, "boundary_probe": boundary_receipt,
                       "boundary_loop": boundary_loop})
    elif task["phase"] == "E3A":
        source_nodes_t, source_weights_t = gauss_legendre(
            protocol["source_resolution"]["nodes"], device, torch.float32,
        )
        nodes = [float(item) for item in source_nodes_t]
        weights = [float(item) for item in source_weights_t]
        grids = {}
        for grid in protocol["source_resolution"]["grids"]:
            source = source_resolved_pefa(
                model, image, block_index=block, class_id=class_id, head_names=heads,
                grid=int(grid), nodes=nodes, weights=weights, seed=task_seed, progress=progress,
            )
            baselines = spatial_baselines_and_effects(
                model, image, baseline=baseline, class_id=class_id, grid=int(grid),
            )
            methods = dict(source["methods"])
            methods.update({"endpoint": baselines["endpoint"],
                            "input_norm": baselines["input_norm"]})
            effects = {"endpoint_change_l1": baselines["endpoint_change_l1"],
                       "target_logit_abs_change": baselines["target_logit_abs_change"]}
            source["methods"] = methods; source["effects"] = effects
            source["correlations"] = source_correlations(methods, effects)
            grids[str(grid)] = source
            progress()
        result["source_resolution"] = grids
    else:
        raise RuntimeError("UNKNOWN_EXTENSION_PHASE")

    torch.cuda.synchronize()
    result["worker_identity"] = worker_identity(claim)
    result["resources"] = {
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
        "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    binding = verify_binding(args.root / "source_binding.json")
    protocol = json_load(args.root / "protocol.json")
    queue = args.root / "queue.sqlite3"; results = args.root / "results"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).requires_grad_()
    (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize()
    registry = json_load(REPO / protocol["model_registry_path"]); loaded = {}
    while True:
        claim = claim_task(
            queue, worker_id=args.worker_id, device_uuid=args.gpu_uuid,
            lease_seconds=protocol["resources"]["lease_seconds"],
            max_attempts=protocol["resources"]["max_attempts"],
        )
        if claim is None:
            break
        try:
            task = claim["payload"]; model_id = task["model_id"]
            if model_id not in loaded:
                spec = next(row for row in registry["models"] if row["model_id"] == model_id)
                loaded[model_id] = (load_model(spec, device)[0], spec)
            model, spec = loaded[model_id]
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            result = execute(model, spec, task, claim, queue, protocol, device)
            if result["resources"]["peak_reserved_gpu_bytes"] > protocol["resources"]["gpu_memory_cap_bytes"]:
                raise RuntimeError("PAUSED_RESOURCE_CAP:GPU_MEMORY")
            if result["resources"]["peak_cpu_rss_bytes"] > protocol["resources"]["cpu_rss_cap_bytes"]:
                raise RuntimeError("PAUSED_RESOURCE_CAP:CPU_RSS")
            complete_task(queue, results, claim, result)
            print(json.dumps({"status": result["status"], "phase": task["phase"],
                              "task_id": claim["task_id"],
                              "seconds": result["resources"]["wall_seconds"]}), flush=True)
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc(), "time_ns": time.time_ns()}
            abandon(queue, claim, failure)
            print(json.dumps({"status": "EXTENSION_TASK_FAILED",
                              "task_id": claim["task_id"],
                              "attempt": claim["attempt"], **failure}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
