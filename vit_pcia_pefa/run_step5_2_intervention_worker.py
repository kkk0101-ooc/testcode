#!/usr/bin/env python3
"""Run natural-head interventions after Step-5.2 predictions are sealed."""
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
from vit_pcia_pefa.src.step5_2_io import abandon, rng_state, sha256, verify, verify_binding  # noqa: E402
from vit_pcia_pefa.src.step5_2_prediction import (  # noqa: E402
    capture_attention_context, intervention_outcome, intervention_parity, natural_head_linearization,
)
from vit_pcia_pefa.src.step5_2_queue import claim_task, complete_task, heartbeat  # noqa: E402


def quadrature(count: int):
    nodes, weights = np.polynomial.legendre.leggauss(count); return (nodes + 1) / 2, weights / 2


def execute(model, spec, task, claim, queue, protocol, prediction_freeze, device):
    started = time.monotonic(); last_heartbeat = started; seed = int(claim["seed"])
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    before = rng_state(torch, device)
    def progress():
        nonlocal last_heartbeat
        now = time.monotonic()
        if now - started > protocol["resources"]["task_timeout_seconds"]: raise RuntimeError("PAUSED_RESOURCE_CAP:TASK_TIMEOUT")
        if now - last_heartbeat >= 25:
            heartbeat(queue, claim, lease_seconds=protocol["resources"]["task_timeout_seconds"]); last_heartbeat = now
    prediction = next(row for row in prediction_freeze["predictions"] if row["task_id"] == claim["task_id"])
    if sha256(queue.parent / prediction["result_path"]) != prediction["result_sha256"]:
        raise RuntimeError("PREDICTION_RESULT_BINDING")
    if sha256(queue.parent / prediction["array_path"]) != prediction["array_sha256"]:
        raise RuntimeError("PREDICTION_ARRAY_BINDING")
    image, original_size = load_image({"cache_path": task["image_path"], "image_bytes": task["image_bytes"],
        "image_sha256": task["image_sha256"]}, spec, device)
    nodes, weights = quadrature(protocol["integration"]["real_fine_nodes"])
    block, class_id = int(task["block_index"]), int(task["class_id"])
    clean_input, _ = capture_attention_context(model, image, block)
    clean_endpoint, clean_logit, clean_prediction = endpoint_and_logit(model, image, class_id)
    clean = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=nodes, weights=weights, progress=progress)
    clean_closure = closure_summary(model, block, clean, protocol["cut_id"]); numeric = protocol["numeric"]["real"]
    if not clean_closure["pass"]: raise RuntimeError("FAIL_COMPLETE_CUT_CLOSURE:CLEAN")
    arrays = np.load(queue.parent / prediction["array_path"], allow_pickle=False)
    head_indexes = [index for index, name in enumerate(clean["names"]) if name.startswith("head_")]
    bindings = {"R": clean["R"][head_indexes], "A": clean["A"][head_indexes],
                "Q": clean["T"][head_indexes], "full_G": clean["R"][-1],
                "full_IG": clean["A"][-1], "full_D": clean["D"][-1], "endpoint": clean_endpoint}
    for key, current in bindings.items():
        frozen = torch.from_numpy(arrays[key]).to(device=current.device, dtype=current.dtype)
        if not torch.allclose(current, frozen, atol=numeric["atol"], rtol=numeric["rtol"]):
            raise RuntimeError("PREDICTION_CLEAN_RECOMPUTATION_DRIFT:" + key)
    prediction_document = json.loads((queue.parent / prediction["result_path"]).read_text())
    prediction_result = prediction_document["result"]
    if (abs(float(prediction_result["clean_logit"]) - clean_logit) > numeric["atol"] + numeric["rtol"] * abs(clean_logit)
            or int(prediction_result["clean_prediction"]) != clean_prediction):
        raise RuntimeError("PREDICTION_CLEAN_OUTPUT_DRIFT")
    rows = []
    for head in range(int(task["head_count"])):
        for strength in (0.5, 0.0):
            with natural_head_linearization(model, block, head, clean_input, strength=strength):
                modified_endpoint, modified_logit, modified_prediction = endpoint_and_logit(model, image, class_id)
                changed = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=nodes, weights=weights, progress=progress)
            parity = intervention_parity(clean, changed, clean_endpoint=clean_endpoint, modified_endpoint=modified_endpoint,
                clean_logit=clean_logit, modified_logit=modified_logit, clean_prediction=clean_prediction,
                modified_prediction=modified_prediction, **{key: numeric[key] for key in ("atol", "rtol", "near_zero")})
            closure = closure_summary(model, block, changed, protocol["cut_id"])
            outcome = intervention_outcome(clean, changed, head_index=head, epsilon=protocol["numeric"]["outcome_epsilon"]); outcome.pop("delta_G")
            if not parity["pass"] or not closure["pass"]: raise RuntimeError(f"INVALID_NUMERICAL_OR_INTERVENTION:HEAD_{head}:LAMBDA_{strength}")
            rows.append({"head_index": head, "lambda": strength, "clean_parity": parity, "closure": closure, "outcome": outcome})
            progress()
    dose = {str(head): rows[2 * head]["outcome"]["Y"] <= rows[2 * head + 1]["outcome"]["Y"] + protocol["numeric"]["dose_absolute_tolerance"]
            for head in range(int(task["head_count"]))}
    elapsed = time.monotonic() - started
    return {"status": "PASS_INTERVENTION_TASK", "model_id": task["model_id"], "sample_id": task["sample_id"],
        "split": task["split"], "split_order": task["split_order"], "class_id": class_id,
        "head_count": task["head_count"], "top_k": task["top_k"], "original_size_wh": original_size,
        "prediction_freeze_digest": prediction_freeze["canonical_digest"], "prediction_result_digest": prediction["result_digest"],
        "clean_closure": clean_closure, "rows": rows, "dose_order_by_head": dose,
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
    protocol = verify(run / "protocol.json"); prediction_freeze = verify(run / "prediction_freeze_receipt.json")
    if prediction_freeze["status"] != "PASS_PREDICTIONS_FROZEN_BEFORE_INTERVENTION": raise RuntimeError("PREDICTION_FREEZE_GATE_CLOSED")
    queue = run / "intervention_queue.sqlite3"; results = run / "intervention_results"
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
            result = execute(model, spec, claim["payload"], claim, queue, protocol, prediction_freeze, device)
            if result["resources"]["peak_reserved_gpu_bytes"] > protocol["resources"]["gpu_memory_cap_bytes"]: raise RuntimeError("PAUSED_RESOURCE_CAP:GPU_MEMORY")
            if result["resources"]["peak_cpu_rss_bytes"] > protocol["resources"]["cpu_rss_cap_bytes"]: raise RuntimeError("PAUSED_RESOURCE_CAP:CPU_RSS")
            complete_task(queue, results, claim, result); print(json.dumps({"status": result["status"], "task_id": claim["task_id"], "seconds": result["resources"]["wall_seconds"]}), flush=True)
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(), "time_ns": time.time_ns()}
            abandon(queue, claim, failure, protocol["resources"]["max_attempts"]); print(json.dumps({"status": "TASK_FAILED", "task_id": claim["task_id"], **failure}), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
