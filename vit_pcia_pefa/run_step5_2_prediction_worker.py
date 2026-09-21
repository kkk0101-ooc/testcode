#!/usr/bin/env python3
"""Compute and store Step-5.2 predictions from the clean model only."""
from __future__ import annotations

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import json
from pathlib import Path
import platform
import random
import resource
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from run_step4_worker import load_image, load_model  # noqa: E402
from vit_pcia_pefa.src.pilot import closure_summary  # noqa: E402
from vit_pcia_pefa.src.step5_1_localization import cut_formation_fields, endpoint_and_logit  # noqa: E402
from vit_pcia_pefa.src.step5_2_io import abandon, rng_state, verify, verify_binding  # noqa: E402
from vit_pcia_pefa.src.step5_2_prediction import (  # noqa: E402
    attention_head_level_norms, capture_attention_context, clean_prediction_scores, convergence_check, signed_head_maps,
)
from vit_pcia_pefa.src.step5_2_queue import claim_task, complete_task, heartbeat  # noqa: E402


def quadrature(count: int):
    nodes, weights = np.polynomial.legendre.leggauss(count); return (nodes + 1) / 2, weights / 2


def execute(model, spec, task, claim, queue, protocol, device):
    started = time.monotonic(); last_heartbeat = started; seed = int(claim["seed"])
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    before = rng_state(torch, device)
    def progress():
        nonlocal last_heartbeat
        now = time.monotonic()
        if now - started > protocol["resources"]["task_timeout_seconds"]: raise RuntimeError("PAUSED_RESOURCE_CAP:TASK_TIMEOUT")
        if now - last_heartbeat >= 25:
            heartbeat(queue, claim, lease_seconds=protocol["resources"]["task_timeout_seconds"]); last_heartbeat = now
    image, original_size = load_image({"cache_path": task["image_path"], "image_bytes": task["image_bytes"],
        "image_sha256": task["image_sha256"]}, spec, device)
    coarse_nodes, coarse_weights = quadrature(protocol["integration"]["real_coarse_nodes"])
    fine_nodes, fine_weights = quadrature(protocol["integration"]["real_fine_nodes"])
    block, class_id = int(task["block_index"]), int(task["class_id"])
    _, clean_activation = capture_attention_context(model, image, block)
    clean_endpoint, clean_logit, clean_prediction = endpoint_and_logit(model, image, class_id)
    coarse = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=coarse_nodes, weights=coarse_weights, progress=progress)
    clean = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=fine_nodes, weights=fine_weights, progress=progress)
    numeric = protocol["numeric"]["real"]
    convergence = convergence_check(coarse, clean, atol=numeric["convergence_atol"],
        rtol=numeric["convergence_rtol"], near_zero=numeric["near_zero"],
        rule=numeric.get("convergence_rule", "strict_elementwise_and_global_relative"))
    closure = closure_summary(model, block, clean, protocol["cut_id"])
    if not convergence["pass"] or not closure["pass"]: raise RuntimeError("INVALID_NUMERICAL_OR_INTERVENTION:PREDICTION")
    activation_norms, gradient_norms = attention_head_level_norms(model, image, block, class_id)
    task_id = claim["task_id"]
    scores = clean_prediction_scores(clean, activation_norms=activation_norms, gradient_norms=gradient_norms,
        random_namespace=f"{protocol['run_id']}|{task_id}")
    maps = signed_head_maps(clean)
    artifact_rel = Path("prediction_arrays") / task_id / f"attempt-{int(claim['attempt']):02d}.npz"
    artifact = queue.parent / artifact_rel
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("xb") as stream:
        np.savez_compressed(stream, R=maps["R"], A=maps["A"], Q=maps["T"],
            full_G=clean["R"][-1].cpu().numpy(), full_IG=clean["A"][-1].cpu().numpy(),
            full_D=clean["D"][-1].cpu().numpy(), endpoint=clean_endpoint.cpu().numpy())
        stream.flush(); os.fsync(stream.fileno())
    from vit_pcia_pefa.src.step5_2_io import sha256
    elapsed = time.monotonic() - started
    return {"status": "PASS_PREDICTION_TASK", "model_id": task["model_id"], "sample_id": task["sample_id"],
        "split": task["split"], "split_order": task["split_order"], "class_id": class_id,
        "head_count": task["head_count"], "top_k": task["top_k"], "original_size_wh": original_size,
        "scores": scores, "clean_logit": clean_logit, "clean_prediction": clean_prediction,
        "clean_closure": closure, "quadrature_convergence": convergence,
        "prediction_array": {"path": artifact_rel.as_posix(), "bytes": artifact.stat().st_size, "sha256": sha256(artifact)},
        "worker_identity": {"sys_executable_realpath": str(Path(sys.executable).resolve()), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu_uuid": claim["device_uuid"],
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "tf32": False, "autocast": False},
        "reproducibility": {"task_seed": seed, "rng_state_before": before, "rng_state_after": rng_state(torch, device),
            "checkpoint_sha256": spec["checkpoint"]["sha256"], "input_sha256": task["image_sha256"]},
        "resources": {"wall_seconds": elapsed, "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)}}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--physical-index", type=int, required=True); parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--worker-id", required=True); args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index): raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    run = args.root.resolve(); binding = verify(run / "source_binding.json"); verify_binding(binding, REPO)
    protocol = verify(run / "protocol.json")
    decision_path = run / "p0_decision.json"
    decision = verify(decision_path if decision_path.exists() else run / "p0_rebind_receipt.json")
    if not decision["prediction_authorized_by_gate"]: raise RuntimeError("P0_GATE_CLOSED")
    queue = run / "prediction_queue.sqlite3"; results = run / "prediction_results"
    registry = json.loads((REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json").read_text())
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; device = torch.device("cuda:0")
    probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).requires_grad_(); (probe @ probe.T).square().sum().backward()
    torch.cuda.synchronize(); loaded = {}
    while True:
        claim = claim_task(queue, worker_id=args.worker_id, device_uuid=args.gpu_uuid,
            lease_seconds=protocol["resources"]["task_timeout_seconds"], max_attempts=protocol["resources"]["max_attempts"])
        if claim is None: break
        try:
            model_id = claim["payload"]["model_id"]
            if model_id not in loaded:
                spec = next(row for row in registry["models"] if row["model_id"] == model_id); loaded[model_id] = (load_model(spec, device)[0], spec)
            model, spec = loaded[model_id]; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            result = execute(model, spec, claim["payload"], claim, queue, protocol, device)
            if result["resources"]["peak_reserved_gpu_bytes"] > protocol["resources"]["gpu_memory_cap_bytes"]: raise RuntimeError("PAUSED_RESOURCE_CAP:GPU_MEMORY")
            if result["resources"]["peak_cpu_rss_bytes"] > protocol["resources"]["cpu_rss_cap_bytes"]: raise RuntimeError("PAUSED_RESOURCE_CAP:CPU_RSS")
            complete_task(queue, results, claim, result); print(json.dumps({"status": result["status"], "task_id": claim["task_id"], "seconds": result["resources"]["wall_seconds"]}), flush=True)
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(), "time_ns": time.time_ns()}
            abandon(queue, claim, failure, protocol["resources"]["max_attempts"]); print(json.dumps({"status": "TASK_FAILED", "task_id": claim["task_id"], **failure}), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
