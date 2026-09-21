#!/usr/bin/env python3
"""Outcome-independent numerical diagnosis of the r7 Omega closure failures."""
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

import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]

from vit_pcia_pefa.run_step4_attempt1_worker import load_image, load_model  # noqa: E402
from vit_pcia_pefa.src.pilot import omega_directional_analysis  # noqa: E402
from vit_pcia_pefa.src.vit_adapter import parity, port_fields, port_layout  # noqa: E402


def direct_curl(model, image, *, block_index: int, class_id: int, names: tuple[str, ...],
                direction: torch.Tensor) -> tuple[torch.Tensor, dict]:
    root = image.detach().requires_grad_(True)

    def field(value: torch.Tensor) -> torch.Tensor:
        fields, current_names = port_fields(
            model, value, block_index=block_index, class_id=class_id,
            second_order=True, requested_names=names,
        )
        if current_names != names:
            raise RuntimeError("DIRECT_PORT_ORDER_DRIFT")
        return fields.sum(0)

    _, jvp = torch.autograd.functional.jvp(
        field, root, direction, create_graph=False, strict=True,
    )
    _, vjp = torch.autograd.functional.vjp(
        field, root, v=direction, create_graph=False, strict=True,
    )
    curl = (vjp - jvp).detach().double()
    return curl, parity(curl, torch.zeros_like(curl))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--coarse-cut-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).requires_grad_()
    (probe @ probe.T).square().sum().backward()
    torch.cuda.synchronize()

    pilot = ROOT / "step5_pilot_r7_20260921"
    protocol = json.loads((pilot / "protocol.json").read_text())
    manifest = json.loads((pilot / "task_manifest.json").read_text())
    task_row = next(row for row in manifest["tasks"] if row["task_id"] == args.task_id)
    task = task_row["payload"]
    registry = json.loads((REPO / protocol["model_registry_path"]).read_text())
    spec = next(row for row in registry["models"] if row["model_id"] == task["model_id"])
    model = load_model(spec, device)[0]
    image = load_image({
        "cache_path": task["image_path"], "image_bytes": task["image_bytes"],
        "image_sha256": task["image_sha256"],
    }, spec, device)[0]
    if args.float64:
        model = model.double()
        image = image.double()
    seed_text = f"{protocol['seed_namespace']}|{args.task_id}".encode()
    import hashlib
    task_seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "big")
    direction_seed = task_seed + args.block
    generator = torch.Generator(device=device)
    generator.manual_seed(direction_seed)
    direction = torch.randn(image.shape, device=device, dtype=image.dtype, generator=generator)
    direction = direction / direction.norm().clamp_min(1e-30)
    layout = port_layout(model.blocks[args.block].attn.num_heads)
    cut_names = tuple(layout.cuts[protocol["primary_cut"]])
    component_names = tuple(layout.component_ids)

    started = time.monotonic()
    streamed = None
    if not args.coarse_cut_only:
        streamed = omega_directional_analysis(
            model, image, block_index=args.block, class_id=int(task["class_id"]),
            component_names=component_names, cut_component_names=cut_names, seed=direction_seed,
        )
    direct_names = ("identity", "attention") if args.coarse_cut_only else cut_names
    direct_cut_field, direct_cut = direct_curl(
        model, image, block_index=args.block, class_id=int(task["class_id"]),
        names=direct_names, direction=direction,
    )
    full_field, full = direct_curl(
        model, image, block_index=args.block, class_id=int(task["class_id"]),
        names=("__full__",), direction=direction,
    )
    cut_vs_full = parity(direct_cut_field, full_field)
    torch.cuda.synchronize()
    result = {
        "status": "DIAGNOSIS_COMPLETE",
        "task_id": args.task_id,
        "model_id": task["model_id"],
        "sample_id": task["sample_id"],
        "class_id": int(task["class_id"]),
        "block": args.block,
        "task_seed": task_seed,
        "direction_seed": direction_seed,
        "diagnostic_dtype": str(image.dtype),
        "cut_component_ids": list(cut_names),
        "direct_cut_evaluation_component_ids": list(direct_names),
        "streamed_component_sum": None if streamed is None else streamed["complete_cut_zero_sum"],
        "direct_cut_field": direct_cut,
        "full_gradient_field": full,
        "direct_cut_curl_vs_full_gradient_curl": cut_vs_full,
        "elapsed_seconds": time.monotonic() - started,
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
        "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "tolerance_changed": False,
        "step6_started": False,
    }
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
