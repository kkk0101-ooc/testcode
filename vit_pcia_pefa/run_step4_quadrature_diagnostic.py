#!/usr/bin/env python3
"""Frozen 256/512-node full-model PEFA closure diagnostic."""
from __future__ import annotations

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import timm
import torch
from PIL import Image
from timm.layers import set_fused_attn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vitnns_rebuild/src"))
from vit_pcia_pefa.src.vit_adapter import parity, tensor_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=("vit_b16", "deit_s16"), required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    if args.output.exists():
        raise RuntimeError("OUTPUT_EXISTS")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    registry = json.loads((REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json").read_text())
    spec = next(row for row in registry["models"] if row["model_id"] == args.model_id)
    set_fused_attn(False)
    model = timm.create_model(spec["timm_name"], pretrained=False).eval()
    loaded = torch.load(REPO / "vitnns_rebuild" / spec["checkpoint"]["path"], map_location="cpu", weights_only=True)
    model.load_state_dict(loaded.get("model", loaded), strict=True)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for block in model.blocks: block.attn.fused_attn = False
    model.cuda()
    receipt = json.loads((ROOT / "step4_runtime/discovery_receipt.json").read_text())
    with Image.open(REPO / receipt["cache_path"]) as source:
        rgb = source.convert("RGB")
        value = TF.pil_to_tensor(TF.center_crop(TF.resize(
            rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True,
        ), [224, 224])).float().div(255).unsqueeze(0)
    mean = torch.tensor(spec["preprocessing"]["mean"]).reshape(1, 3, 1, 1)
    std = torch.tensor(spec["preprocessing"]["std"]).reshape(1, 3, 1, 1)
    image = ((value - mean) / std).cuda()
    class_id = int(receipt["class_id"])

    def full_field(root):
        score = model(root)[:, class_id].sum()
        return torch.autograd.grad(score, root, create_graph=True)[0]

    def q_field(root):
        return root * full_field(root)

    endpoint = q_field(image.detach().requires_grad_(True)).detach().double()
    rows = []
    integrated = {}
    for count in (256, 512):
        nodes, weights = np.polynomial.legendre.leggauss(count)
        nodes = (nodes + 1.0) / 2.0; weights = weights / 2.0
        accumulation = torch.zeros_like(endpoint)
        maximum_integrand = 0.0
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
        for node, weight in zip(nodes, weights):
            root = (image * float(node)).detach().requires_grad_(True)
            _primal, derivative = torch.autograd.functional.jvp(
                q_field, root, image, create_graph=False, strict=True,
            )
            accumulation.add_(derivative.detach().double() * float(weight))
            maximum_integrand = max(maximum_integrand, float(derivative.abs().max()))
        torch.cuda.synchronize()
        integrated[count] = accumulation
        rows.append({
            "node_count": count, "closure": parity(accumulation, endpoint),
            "integral_summary": tensor_summary(accumulation),
            "endpoint_summary": tensor_summary(endpoint),
            "maximum_integrand_abs": maximum_integrand,
            "wall_seconds": time.perf_counter() - started,
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
        })
    convergence = parity(integrated[256], integrated[512])
    result = {
        "schema_version": 1, "stage": "ENDPOINT_PCIA_PEFA_STEP4",
        "diagnostic": "independent_full_model_Q_directional_derivative",
        "model_id": args.model_id, "sample_id": receipt["sample_id"],
        "class_id": class_id, "rows": rows, "convergence_256_to_512": convergence,
        "status": "PASS" if rows[-1]["closure"]["pass"] else "FAIL_QUADRATURE_CLOSURE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"model_id": args.model_id, "status": result["status"],
                      "closure_512": rows[-1]["closure"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
