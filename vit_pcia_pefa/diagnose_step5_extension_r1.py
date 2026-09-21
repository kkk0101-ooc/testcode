#!/usr/bin/env python3
"""Diagnose the r1 closed-loop quadrature invalidity on its worst task."""
from __future__ import annotations
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import hashlib
import json
import math
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
from vit_pcia_pefa.src.step5_extension import boundary_directed_point, orthonormal_directions  # noqa: E402
from vit_pcia_pefa.src.step5_extension_queue import task_seed  # noqa: E402
from vitnns.endpoint_ep2 import gauss_legendre  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def line(model, start, end, class_id: int, nodes: torch.Tensor, weights: torch.Tensor) -> float:
    delta = end - start; total = 0.0
    for node, weight in zip(nodes, weights):
        root = (start + float(node) * delta).detach().requires_grad_(True)
        score = model(root)[:, class_id].sum()
        gradient = torch.autograd.grad(score, root)[0]
        total += float((gradient.double() * delta.double()).sum()) * float(weight)
    return total


def diagnose(model, base, class_id: int, seed: int, device) -> list[dict[str, Any]]:
    first, second = orthonormal_directions(base, seed); epsilon = 1.0
    corners = (base, base + epsilon * first, base + epsilon * first + epsilon * second,
               base + epsilon * second, base)
    with torch.no_grad():
        scores = [float(model(point)[:, class_id].sum()) for point in corners]
    rows = []
    for count in (8, 16, 32, 64):
        nodes, weights = gauss_legendre(count, device, torch.float32)
        edges = [line(model, start, end, class_id, nodes, weights)
                 for start, end in zip(corners[:-1], corners[1:])]
        rows.append({"nodes_per_edge": count, "edge_integrals": edges,
                     "closed_loop_circulation": math.fsum(edges),
                     "endpoint_score_telescoping": math.fsum(
                         scores[index + 1] - scores[index] for index in range(4))})
    return rows


def main() -> int:
    output = ROOT / "step5_extension_r1_quadrature_diagnosis.json"
    if output.exists():
        raise RuntimeError("DIAGNOSIS_EXISTS")
    physical = int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))
    if physical < 0:
        raise RuntimeError("CUDA_VISIBLE_DEVICES")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0"); started = time.monotonic()
    probe = torch.arange(16, device=device).reshape(4, 4).float().requires_grad_()
    (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize()
    root = ROOT / "step5_extension_r1_20260921"
    manifest = json.loads((root / "task_manifest.json").read_text())
    task_id = "e1_e2__vit_b16__07"
    task = next(row["payload"] for row in manifest["tasks"] if row["task_id"] == task_id)
    registry = json.loads((REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json").read_text())
    spec = next(row for row in registry["models"] if row["model_id"] == task["model_id"])
    model = load_model(spec, device)[0]
    image = load_image({"cache_path": task["image_path"], "image_bytes": task["image_bytes"],
                        "image_sha256": task["image_sha256"]}, spec, device)[0]
    seed = task_seed(task_id); class_id = int(task["class_id"])
    mean = torch.tensor(spec["preprocessing"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(spec["preprocessing"]["std"], device=device).view(1, 3, 1, 1)
    boundary, boundary_receipt = boundary_directed_point(
        model, image, class_id=class_id, mean=mean, std=std,
        candidates_rgb=(1/255, 2/255, 4/255, 8/255),
    )
    clean = diagnose(model, image, class_id, seed, device)
    boundary_rows = diagnose(model, boundary, class_id, seed, device)
    torch.cuda.synchronize(); elapsed = time.monotonic() - started
    receipt = {
        "schema_version": 1, "run_id": "step5_extension_r1_quadrature_diagnosis_20260921",
        "status": "DIAGNOSIS_COMPLETE", "source_root": "step5_extension_r1_20260921",
        "source_terminal_status": "INVALID", "task_id": task_id, "task_seed": seed,
        "selection_reason": "largest observed r1 full-gradient closed-loop residual",
        "scientific_claim_evaluated": False, "threshold_changed": False,
        "clean": clean, "boundary": boundary_rows, "boundary_probe": boundary_receipt,
        "resources": {"wall_seconds": elapsed, "h20_hours": elapsed / 3600,
                      "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
                      "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                      "physical_index": physical},
        "step6_started": False,
    }
    receipt["canonical_digest"] = hashlib.sha256(canonical(receipt)).hexdigest()
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
