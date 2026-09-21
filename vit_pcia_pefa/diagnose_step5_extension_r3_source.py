#!/usr/bin/env python3
"""Diagnose E3A source-integral convergence without evaluating its rank claim."""
from __future__ import annotations
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.run_step4_attempt1_worker import load_image, load_model  # noqa: E402
from vit_pcia_pefa.src.step5_extension import source_resolved_pefa  # noqa: E402
from vit_pcia_pefa.src.step5_extension_queue import task_seed  # noqa: E402
from vitnns.endpoint_ep2 import gauss_legendre  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=("deit_s16", "vit_b16"), required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    output = ROOT / f"step5_extension_r3_source_diagnosis_{args.model_id}.json"
    if output.exists():
        raise RuntimeError("DIAGNOSIS_EXISTS")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0"); started = time.monotonic()
    probe = torch.arange(16, device=device).reshape(4, 4).float().requires_grad_()
    (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize()
    extension_root = ROOT / "step5_extension_r3_20260921"
    manifest = json.loads((extension_root / "task_manifest.json").read_text())
    task_id = f"e3a__{args.model_id}__00"
    task = next(row["payload"] for row in manifest["tasks"] if row["task_id"] == task_id)
    registry = json.loads((REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json").read_text())
    spec = next(row for row in registry["models"] if row["model_id"] == args.model_id)
    model = load_model(spec, device)[0]
    image = load_image({"cache_path": task["image_path"], "image_bytes": task["image_bytes"],
                        "image_sha256": task["image_sha256"]}, spec, device)[0]
    heads = tuple(f"head_{index}" for index in range(model.blocks[5].attn.num_heads))
    rows = []
    for count in (16, 64, 128, 256, 512):
        nodes_t, weights_t = gauss_legendre(count, device, torch.float32)
        node_started = time.monotonic()
        result = source_resolved_pefa(
            model, image, block_index=5, class_id=int(task["class_id"]), head_names=heads,
            grid=4, nodes=[float(item) for item in nodes_t], weights=[float(item) for item in weights_t],
            seed=task_seed(task_id),
        )
        rows.append({"nodes": count, "terminal_T_convergence": result["terminal_T_convergence"],
                     "source_sum_closure": result["source_sum_closure"],
                     "wall_seconds": time.monotonic() - node_started})
    torch.cuda.synchronize(); elapsed = time.monotonic() - started
    receipt = {
        "schema_version": 1, "run_id": f"step5_extension_r3_source_diagnosis_{args.model_id}_20260921",
        "status": "DIAGNOSIS_COMPLETE", "source_root": "step5_extension_r3_20260921",
        "source_terminal_status": "NOT_SUPPORTED", "task_id": task_id,
        "selection_reason": "fixed split_order 0 for both registered models",
        "scientific_rank_claim_evaluated": False, "threshold_changed": False,
        "rows": rows, "resources": {"wall_seconds": elapsed, "h20_hours": elapsed / 3600,
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "physical_index": args.physical_index, "gpu_uuid": args.gpu_uuid},
        "step6_started": False,
    }
    receipt["canonical_digest"] = hashlib.sha256(canonical(receipt)).hexdigest()
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": receipt["status"], "model_id": args.model_id,
                      "rows": rows, "resources": receipt["resources"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
