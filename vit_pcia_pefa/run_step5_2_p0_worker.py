#!/usr/bin/env python3
"""Run one frozen real-model Step-5.2 P0 task."""
from __future__ import annotations

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from run_step4_worker import load_image, load_model  # noqa: E402
from vit_pcia_pefa.src.pilot import closure_summary  # noqa: E402
from vit_pcia_pefa.src.step5_1_localization import cut_formation_fields, endpoint_and_logit  # noqa: E402
from vit_pcia_pefa.src.step5_2_io import verify, verify_binding, write_new  # noqa: E402
from vit_pcia_pefa.src.step5_2_prediction import (  # noqa: E402
    capture_attention_context, convergence_check, intervention_outcome, intervention_parity, natural_head_linearization,
)


def quadrature(count: int):
    nodes, weights = np.polynomial.legendre.leggauss(count); return (nodes + 1) / 2, weights / 2


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-id", required=True); parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True); args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index): raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    started = time.perf_counter(); run = args.root.resolve(); protocol = verify(run / "protocol.json")
    manifest = verify(run / "task_manifest.json"); binding = verify(run / "source_binding.json"); verify_binding(binding, REPO)
    task = next(row for row in manifest["p0_tasks"] if row["model_id"] == args.model_id)
    registry = json.loads((REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json").read_text())
    spec = next(row for row in registry["models"] if row["model_id"] == args.model_id)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0"); probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).requires_grad_()
    (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    model, model_binding = load_model(spec, device)
    image, original_size = load_image({"cache_path": task["image_path"], "image_bytes": task["image_bytes"],
        "image_sha256": task["image_sha256"]}, spec, device)
    coarse_nodes, coarse_weights = quadrature(protocol["integration"]["real_coarse_nodes"])
    fine_nodes, fine_weights = quadrature(protocol["integration"]["real_fine_nodes"])
    block, class_id, head = int(task["block_index"]), int(task["class_id"]), int(task["head_index"])
    clean_input, _ = capture_attention_context(model, image, block)
    clean_endpoint, clean_logit, clean_prediction = endpoint_and_logit(model, image, class_id)
    coarse = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=coarse_nodes, weights=coarse_weights)
    clean = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=fine_nodes, weights=fine_weights)
    numeric = protocol["numeric"]["real"]
    convergence = convergence_check(coarse, clean, atol=numeric["convergence_atol"],
        rtol=numeric["convergence_rtol"], near_zero=numeric["near_zero"],
        rule=numeric.get("convergence_rule", "strict_elementwise_and_global_relative"))
    rows = []
    for strength in (0.5, 0.0):
        with natural_head_linearization(model, block, head, clean_input, strength=strength):
            changed_endpoint, changed_logit, changed_prediction = endpoint_and_logit(model, image, class_id)
            changed = cut_formation_fields(model, image, block_index=block, class_id=class_id,
                                           nodes=fine_nodes, weights=fine_weights)
        outcome = intervention_outcome(clean, changed, head_index=head, epsilon=protocol["numeric"]["outcome_epsilon"]); outcome.pop("delta_G")
        rows.append({"lambda": strength, "clean_parity": intervention_parity(clean, changed,
            clean_endpoint=clean_endpoint, modified_endpoint=changed_endpoint, clean_logit=clean_logit,
            modified_logit=changed_logit, clean_prediction=clean_prediction, modified_prediction=changed_prediction,
            **{key: numeric[key] for key in ("atol", "rtol", "near_zero")}),
            "closure": closure_summary(model, block, changed, protocol["cut_id"]), "outcome": outcome})
    dose = rows[0]["outcome"]["Y"] <= rows[1]["outcome"]["Y"] + protocol["numeric"]["dose_absolute_tolerance"]
    clean_closure = closure_summary(model, block, clean, protocol["cut_id"])
    passed = convergence["pass"] and clean_closure["pass"] and dose and rows[1]["outcome"]["Y"] > numeric["atol"]
    passed = passed and all(row["clean_parity"]["pass"] and row["closure"]["pass"] for row in rows)
    torch.cuda.synchronize()
    result = write_new(run / "p0" / f"{args.model_id}.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "task_id": task["task_id"], "status": "PASS_P0_REAL_MODEL" if passed else "INVALID_P0_REAL_MODEL",
        "model_id": args.model_id, "sample_id": task["sample_id"], "class_id": class_id, "head_index": head,
        "original_size_wh": original_size, "quadrature_convergence": convergence, "clean_closure": clean_closure,
        "rows": rows, "dose_order": dose, "model_binding": model_binding, "scientific_claim": False,
        "worker_identity": {"sys_executable_realpath": str(Path(sys.executable).resolve()), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu_uuid": args.gpu_uuid,
            "physical_gpu_index": args.physical_index, "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "tf32": False, "autocast": False,
            "startup_matmul_backward": "PASS"},
        "resources": {"wall_seconds": time.perf_counter() - started,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)},
        "completed_unix_ns": time.time_ns()})
    print(json.dumps({"status": result["status"], "model_id": args.model_id,
                      "Y": [row["outcome"]["Y"] for row in rows]}, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__": raise SystemExit(main())
