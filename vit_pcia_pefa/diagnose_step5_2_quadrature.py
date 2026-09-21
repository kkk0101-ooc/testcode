#!/usr/bin/env python3
"""Outcome-blind P0 discovery-only quadrature diagnosis after r1 failure."""
from __future__ import annotations

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from run_step4_worker import load_image, load_model  # noqa: E402
from vit_pcia_pefa.src.step5_1_localization import cut_formation_fields  # noqa: E402
from vit_pcia_pefa.src.step5_2_io import verify, verify_binding, write_new  # noqa: E402
from vit_pcia_pefa.src.step5_2_prediction import convergence_check  # noqa: E402


def quadrature(count: int):
    nodes, weights = np.polynomial.legendre.leggauss(count); return (nodes + 1) / 2, weights / 2


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--model-id", required=True)
    parser.add_argument("--physical-index", type=int, required=True); parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--nodes", type=int, nargs="+", default=[512, 1024]); args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index): raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    if sorted(args.nodes) != args.nodes or len(args.nodes) < 2: raise RuntimeError("DIAGNOSIS_NODE_SEQUENCE")
    run = args.root.resolve(); protocol = verify(run / "protocol.json"); failure = verify(run / "failure_receipt.json")
    binding = verify(run / "source_binding.json"); verify_binding(binding, REPO)
    if failure["failure_code"] != "FAIL_QUADRATURE_CONVERGENCE": raise RuntimeError("DIAGNOSIS_SCOPE")
    task = next(row for row in verify(run / "task_manifest.json")["p0_tasks"] if row["model_id"] == args.model_id)
    registry = json.loads((REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json").read_text())
    spec = next(row for row in registry["models"] if row["model_id"] == args.model_id)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; device = torch.device("cuda:0")
    probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).requires_grad_(); (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize()
    started = time.perf_counter(); model, model_binding = load_model(spec, device)
    image, _ = load_image({"cache_path": task["image_path"], "image_bytes": task["image_bytes"],
        "image_sha256": task["image_sha256"]}, spec, device)
    fields = []
    for count in args.nodes:
        nodes, weights = quadrature(count); fields.append(cut_formation_fields(model, image, block_index=task["block_index"],
            class_id=task["class_id"], nodes=nodes, weights=weights))
    comparisons = []
    numeric = protocol["numeric"]["real"]
    for left, right, left_count, right_count in zip(fields, fields[1:], args.nodes, args.nodes[1:]):
        comparisons.append({"coarse_nodes": left_count, "fine_nodes": right_count,
            "frozen_r1_tolerance": convergence_check(left, right, atol=numeric["convergence_atol"],
                rtol=numeric["convergence_rtol"], near_zero=numeric["near_zero"])})
    torch.cuda.synchronize()
    node_suffix = "_".join(str(value) for value in args.nodes)
    output = ROOT / f"step5_2_r1_quadrature_diagnosis_{args.model_id}_{node_suffix}.json"
    result = write_new(output, {"schema_version": 1, "status": "P0_DISCOVERY_ONLY_DIAGNOSIS",
        "source_run_id": protocol["run_id"], "source_failure_digest": failure["canonical_digest"],
        "model_id": args.model_id, "sample_id": task["sample_id"], "split": "discovery",
        "evaluation_access": False, "calibration_outcomes_read": False, "nodes": args.nodes, "comparisons": comparisons,
        "model_binding": model_binding, "resources": {"wall_seconds": time.perf_counter() - started,
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)}, "completed_unix_ns": time.time_ns()})
    print(json.dumps({"status": result["status"], "model_id": args.model_id,
        "comparisons": [{"nodes": [row["coarse_nodes"], row["fine_nodes"]], "pass": row["frozen_r1_tolerance"]["pass"],
                         "relative_l2": {key: value["relative_l2"] for key, value in row["frozen_r1_tolerance"]["checks"].items()}} for row in comparisons]}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
