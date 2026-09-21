#!/usr/bin/env python3
"""Compare source-group JVP contraction with the unstable reverse contraction."""
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
from vit_pcia_pefa.src.vit_adapter import port_fields  # noqa: E402
from vitnns.endpoint_ep2 import gauss_legendre  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def directions(image: torch.Tensor, grid: int) -> list[torch.Tensor]:
    height, width = image.shape[-2:]; rows = []
    for row in range(grid):
        for column in range(grid):
            value = torch.zeros_like(image)
            value[:, :, row * height // grid:(row + 1) * height // grid,
                  column * width // grid:(column + 1) * width // grid] = image[:, :, row * height // grid:(row + 1) * height // grid,
                                                                                     column * width // grid:(column + 1) * width // grid]
            rows.append(value)
    if not torch.equal(sum(rows, torch.zeros_like(image)), image):
        raise RuntimeError("DIRECTION_PARTITION")
    return rows


def run(model, image, class_id: int, nodes_count: int, grid: int) -> dict[str, Any]:
    head_names = tuple(f"head_{index}" for index in range(model.blocks[5].attn.num_heads))
    source_directions = directions(image, grid)
    nodes, weights = gauss_legendre(nodes_count, image.device, torch.float32)
    integrated = torch.zeros((len(head_names), grid * grid), dtype=torch.float64, device=image.device)
    for node, weight in zip(nodes, weights):
        root = (image * float(node)).detach().requires_grad_(True)
        for group, direction in enumerate(source_directions):
            def field(value):
                values, names = port_fields(
                    model, value, block_index=5, class_id=class_id, second_order=True,
                    requested_names=head_names,
                )
                if names != head_names:
                    raise RuntimeError("PORT_ORDER")
                return values
            current, derivative = torch.autograd.functional.jvp(
                field, root, direction, create_graph=False, strict=True,
            )
            t_direction = direction.double().unsqueeze(0) * current.detach().double()
            t_direction += root.detach().double().unsqueeze(0) * derivative.detach().double()
            integrated[:, group].add_(t_direction.flatten(1).sum(1), alpha=float(weight))
    fields, names = port_fields(
        model, image.detach().requires_grad_(True), block_index=5, class_id=class_id,
        second_order=False, requested_names=head_names,
    )
    if names != head_names:
        raise RuntimeError("TERMINAL_PORT_ORDER")
    terminal = (fields.detach().double() * image.double().unsqueeze(0)).flatten(1).sum(1)
    source_sum = integrated.sum(1); error = source_sum - terminal
    return {"nodes": nodes_count, "grid": grid,
            "terminal_T_convergence": {"max_abs": float(error.abs().max()),
                "relative_l2": float(error.norm() / terminal.norm().clamp_min(1e-30))}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=("deit_s16", "vit_b16"), required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    output = ROOT / f"step5_extension_r3_jvp_diagnosis_{args.model_id}.json"
    if output.exists():
        raise RuntimeError("DIAGNOSIS_EXISTS")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0"); started = time.monotonic()
    probe = torch.arange(16, device=device).reshape(4, 4).float().requires_grad_()
    (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize()
    root = ROOT / "step5_extension_r3_20260921"
    manifest = json.loads((root / "task_manifest.json").read_text())
    task_id = f"e3a__{args.model_id}__00"
    task = next(row["payload"] for row in manifest["tasks"] if row["task_id"] == task_id)
    registry = json.loads((REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json").read_text())
    spec = next(row for row in registry["models"] if row["model_id"] == args.model_id)
    model = load_model(spec, device)[0]
    image = load_image({"cache_path": task["image_path"], "image_bytes": task["image_bytes"],
                        "image_sha256": task["image_sha256"]}, spec, device)[0]
    rows = []
    for nodes in (16, 64):
        node_started = time.monotonic(); row = run(model, image, int(task["class_id"]), nodes, 4)
        row["wall_seconds"] = time.monotonic() - node_started; rows.append(row)
    torch.cuda.synchronize(); elapsed = time.monotonic() - started
    receipt = {"schema_version": 1, "run_id": f"step5_extension_r3_jvp_diagnosis_{args.model_id}_20260921",
        "status": "DIAGNOSIS_COMPLETE", "source_root": "step5_extension_r3_20260921",
        "task_id": task_id, "formulation": "source-direction JVP of Q via product rule",
        "scientific_rank_claim_evaluated": False, "threshold_changed": False, "rows": rows,
        "resources": {"wall_seconds": elapsed, "h20_hours": elapsed / 3600,
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "physical_index": args.physical_index, "gpu_uuid": args.gpu_uuid},
        "step6_started": False}
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
