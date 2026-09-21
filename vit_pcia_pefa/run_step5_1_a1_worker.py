#!/usr/bin/env python3
"""Execute frozen Step-5.1 A1 tasks from the dynamic global queue."""
from __future__ import annotations

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import random
import resource
import sqlite3
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from run_step4_worker import load_image, load_model  # noqa: E402
from vit_pcia_pefa.src.pilot import closure_summary  # noqa: E402
from vit_pcia_pefa.src.step5_1_localization import (  # noqa: E402
    capture_projection_input, clean_parity, cut_formation_fields, difference_direction,
    endpoint_and_logit, localization_scores, projection_head_wrapper, rank_metrics,
)
from vit_pcia_pefa.src.step5_1_queue import claim_task, complete_task, heartbeat  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if hashlib.sha256(canonical(value)).hexdigest() != claimed: raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def verify_binding(binding: dict[str, Any]) -> None:
    for row in (*binding["files"], *binding["checkpoints"], *binding["images"]):
        path = REPO / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError("FAIL_SOURCE_BINDING:" + row["path"])


def rng_state(device: torch.device) -> dict[str, Any]:
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"bit_generator": state[0], "keys": state[1].tolist(), "position": int(state[2]),
                       "has_gauss": int(state[3]), "cached_gaussian": float(state[4])},
            "torch_cpu_base64": base64.b64encode(torch.get_rng_state().cpu().numpy().tobytes()).decode(),
            "torch_cuda_base64": base64.b64encode(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).decode()}


def abandon(queue: Path, claim: dict[str, Any], error: dict[str, Any]) -> None:
    now = time.time_ns()
    with sqlite3.connect(queue, timeout=30) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = connection.execute(
            "SELECT state,attempt,fencing_token,worker_id,device_uuid FROM tasks WHERE task_id=?",
            (claim["task_id"],),
        ).fetchone()
        valid = current is not None and current[0] == "claimed" and tuple(current[1:]) == (
            claim["attempt"], claim["fencing_token"], claim["worker_id"], claim["device_uuid"])
        if not valid: raise RuntimeError("STALE_FENCING_TOKEN")
        terminal = claim["attempt"] >= 2
        connection.execute("UPDATE attempts SET status=?,completed_ns=? WHERE task_id=? AND attempt=?",
                           ("FAILED" if terminal else "RETRYABLE", now, claim["task_id"], claim["attempt"]))
        connection.execute("UPDATE tasks SET state=?,fencing_token=NULL,worker_id=NULL,device_uuid=NULL,"
                           "lease_until_ns=NULL,heartbeat_ns=NULL WHERE task_id=?",
                           ("failed" if terminal else "pending", claim["task_id"]))
        connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                           (f"failure:{claim['task_id']}:{claim['attempt']}", json.dumps(error, sort_keys=True)))
        connection.commit()


def execute(model, spec: dict[str, Any], task: dict[str, Any], claim: dict[str, Any],
            queue: Path, protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    started = time.monotonic(); last_heartbeat = started; task_seed = int(claim["seed"])
    random.seed(task_seed); np.random.seed(task_seed % 2**32); torch.manual_seed(task_seed); torch.cuda.manual_seed_all(task_seed)
    before = rng_state(device)

    def progress() -> None:
        nonlocal last_heartbeat
        now = time.monotonic()
        if now - started > protocol["resources"]["task_timeout_seconds"]: raise RuntimeError("PAUSED_RESOURCE_CAP:TASK_TIMEOUT")
        if now - last_heartbeat >= 25:
            heartbeat(queue, claim, lease_seconds=7200); last_heartbeat = now

    image, original_size = load_image({"cache_path": task["image_path"], "image_bytes": task["image_bytes"],
                                      "image_sha256": task["image_sha256"]}, spec, device)
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(protocol["integration"]["real_nodes"])
    nodes, weights = (raw_nodes + 1) / 2, raw_weights / 2
    block, class_id = int(task["block_index"]), int(task["class_id"])
    activation = capture_projection_input(model, image, block)
    clean_endpoint, clean_logit, clean_prediction = endpoint_and_logit(model, image, class_id)
    clean = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=nodes, weights=weights, progress=progress)
    clean_closure = closure_summary(model, block, clean, protocol["cut_id"])
    conditions = []
    for role, head in (("primary", int(task["true_head"])), ("decoy", int(task["decoy_head"]))):
        for strength in protocol["wrapper"]["a1_strengths"]:
            with projection_head_wrapper(model, block, head, activation, strength=float(strength)):
                modified_activation = capture_projection_input(model, image, block)
                modified_endpoint, modified_logit, modified_prediction = endpoint_and_logit(model, image, class_id)
                modified = cut_formation_fields(model, image, block_index=block, class_id=class_id,
                                                nodes=nodes, weights=weights, progress=progress)
            parity_row = clean_parity(clean, modified, clean_endpoint=clean_endpoint, modified_endpoint=modified_endpoint,
                clean_logit=clean_logit, modified_logit=modified_logit, clean_prediction=clean_prediction,
                modified_prediction=modified_prediction,
                **{key: protocol["numeric"]["real"][key] for key in ("atol", "rtol", "near_zero")})
            closure = closure_summary(model, block, modified, protocol["cut_id"])
            methods = localization_scores(clean, modified, clean_activation=activation,
                                          modified_activation=modified_activation,
                                          random_namespace=f"{protocol['run_id']}|{claim['task_id']}|{role}|{strength}")
            conditions.append({"role": role, "modified_head": head, "strength": strength,
                "clean_parity": parity_row, "closure": closure, "difference_direction": difference_direction(clean, modified),
                "methods": methods, "rank_metrics": {name: rank_metrics(scores, head, int(task["top_k"]))
                                                            for name, scores in methods.items()}})
            progress()
    controls = []
    for mode, strength in (("bounded_quadratic", 0.0), ("linear_tangent", 1.0)):
        head = int(task["true_head"])
        with projection_head_wrapper(model, block, head, activation, strength=strength, mode=mode):
            modified_activation = capture_projection_input(model, image, block)
            modified_endpoint, modified_logit, modified_prediction = endpoint_and_logit(model, image, class_id)
            modified = cut_formation_fields(model, image, block_index=block, class_id=class_id,
                                            nodes=nodes, weights=weights, progress=progress)
        methods = localization_scores(clean, modified, clean_activation=activation,
                                      modified_activation=modified_activation, random_namespace=f"{task_seed}|{mode}")
        controls.append({"mode": mode, "strength": strength,
            "clean_parity": clean_parity(clean, modified, clean_endpoint=clean_endpoint, modified_endpoint=modified_endpoint,
                clean_logit=clean_logit, modified_logit=modified_logit, clean_prediction=clean_prediction,
                modified_prediction=modified_prediction,
                **{key: protocol["numeric"]["real"][key] for key in ("atol", "rtol", "near_zero")}),
            "closure": closure_summary(model, block, modified, protocol["cut_id"]),
            "maximum_change_score": max(max(methods[key]) for key in
                ("pefa_R_delta", "pcia_delta", "endpoint_delta", "gradient_delta", "activation_delta"))})
    if not clean_closure["pass"] or not all(row["clean_parity"]["pass"] and row["closure"]["pass"]
                                                and row["difference_direction"]["pass"] for row in conditions):
        raise RuntimeError("INVALID_INTERVENTION_CONSTRUCTION_OR_CLOSURE")
    if not all(row["clean_parity"]["pass"] and row["closure"]["pass"]
               and row["maximum_change_score"] <= protocol["numeric"]["real"]["atol"] for row in controls):
        raise RuntimeError("INVALID_NEGATIVE_CONTROL")
    elapsed = time.monotonic() - started
    return {"status": "PASS_A1_TASK", "model_id": task["model_id"], "sample_id": task["sample_id"],
            "split": task["split"], "split_order": task["split_order"], "class_id": class_id,
            "true_head": task["true_head"], "decoy_head": task["decoy_head"], "top_k": task["top_k"],
            "original_size_wh": original_size, "clean_closure": clean_closure, "conditions": conditions,
            "controls": controls, "worker_identity": {"sys_executable_realpath": str(Path(sys.executable).resolve()),
                "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu_uuid": claim["device_uuid"], "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "tf32": False, "autocast": False,
                "driver_version": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                    "--format=csv,noheader"], text=True).splitlines()[0].strip()},
            "reproducibility": {"task_seed": task_seed, "rng_state_before": before, "rng_state_after": rng_state(device),
                "checkpoint_sha256": spec["checkpoint"]["sha256"], "input_sha256": task["image_sha256"]},
            "resources": {"wall_seconds": elapsed, "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)}}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--physical-index", type=int, required=True); parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--worker-id", required=True); args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index): raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    run = args.root.resolve(); binding = verify_signed(run / "source_binding.json"); verify_binding(binding)
    protocol = verify_signed(run / "protocol.json"); queue = run / "a1_queue.sqlite3"; results = run / "a1_results"
    registry = json.loads((REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json").read_text())
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0"); probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).requires_grad_()
    (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize(); loaded = {}
    while True:
        claim = claim_task(queue, worker_id=args.worker_id, device_uuid=args.gpu_uuid, lease_seconds=7200, max_attempts=2)
        if claim is None: break
        try:
            model_id = claim["payload"]["model_id"]
            if model_id not in loaded:
                spec = next(row for row in registry["models"] if row["model_id"] == model_id)
                loaded[model_id] = (load_model(spec, device)[0], spec)
            model, spec = loaded[model_id]; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            result = execute(model, spec, claim["payload"], claim, queue, protocol, device)
            if result["resources"]["peak_reserved_gpu_bytes"] > protocol["resources"]["gpu_memory_cap_bytes"]:
                raise RuntimeError("PAUSED_RESOURCE_CAP:GPU_MEMORY")
            if result["resources"]["peak_cpu_rss_bytes"] > protocol["resources"]["cpu_rss_cap_bytes"]:
                raise RuntimeError("PAUSED_RESOURCE_CAP:CPU_RSS")
            complete_task(queue, results, claim, result)
            print(json.dumps({"status": "PASS_A1_TASK", "task_id": claim["task_id"],
                              "seconds": result["resources"]["wall_seconds"]}), flush=True)
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc(), "time_ns": time.time_ns()}
            abandon(queue, claim, failure)
            print(json.dumps({"status": "TASK_FAILED", "task_id": claim["task_id"], **failure}), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
