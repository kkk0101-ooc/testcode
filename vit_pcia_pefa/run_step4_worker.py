#!/usr/bin/env python3
"""Run one frozen Step-4 model validation and cost measurement on one H20."""
from __future__ import annotations

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import hashlib
import io
import json
import math
import platform
import resource
from pathlib import Path
import sys
import time

import numpy as np
import timm
import torch
import torchvision
from PIL import Image
from timm.layers import set_fused_attn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "vitnns_rebuild/src"))
sys.path.insert(0, str(REPO))
from vit_pcia_pefa.src.vit_adapter import (  # noqa: E402
    first_order_all_blocks, negative_controls, patch_quadrant_directions,
    port_fields, stream_path_integrals, summarize_validation, tensor_summary,
)
from vitnns.endpoint_ep2 import gauss_legendre, integrate_logit  # noqa: E402
from vitnns.recovery_v2 import canonical_state_dict_tensor_digest  # noqa: E402

EP0 = REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(spec: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    path = REPO / "vitnns_rebuild" / spec["checkpoint"]["path"]
    if path.stat().st_size != spec["checkpoint"]["bytes"] or sha256(path) != spec["checkpoint"]["sha256"]:
        raise RuntimeError("CHECKPOINT_IDENTITY")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    state = loaded["model"] if isinstance(loaded, dict) and isinstance(loaded.get("model"), dict) else loaded
    set_fused_attn(False)
    model = timm.create_model(spec["timm_name"], pretrained=False).eval()
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("CHECKPOINT_STRICT_LOAD")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.blocks:
        block.attn.fused_attn = False
    model.to(device)
    return model, {
        "checkpoint_sha256": spec["checkpoint"]["sha256"],
        "tensor_digest": canonical_state_dict_tensor_digest(state),
        "state_key_count": len(state),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_image(receipt: dict, spec: dict, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    path = REPO / receipt["cache_path"]
    payload = path.read_bytes()
    if len(payload) != receipt["image_bytes"] or hashlib.sha256(payload).hexdigest() != receipt["image_sha256"]:
        raise RuntimeError("IMAGE_IDENTITY")
    with Image.open(io.BytesIO(payload)) as source:
        if source.format != "JPEG":
            raise RuntimeError("IMAGE_FORMAT")
        rgb = source.convert("RGB")
        original_size = list(rgb.size)
        resized = TF.resize(rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True)
        cropped = TF.center_crop(resized, [224, 224])
        value = TF.pil_to_tensor(cropped).float().div(255).unsqueeze(0)
    mean = torch.tensor(spec["preprocessing"]["mean"]).reshape(1, 3, 1, 1)
    std = torch.tensor(spec["preprocessing"]["std"]).reshape(1, 3, 1, 1)
    return ((value - mean) / std).to(device), original_size


def timed(device: torch.device, callback):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started = time.perf_counter(); result = callback(); torch.cuda.synchronize()
    return result, {
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=("vit_b16", "deit_s16"), required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    if args.output.exists():
        raise RuntimeError("OUTPUT_EXISTS")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).requires_grad_()
    (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize()
    identity = {
        "sys_executable_realpath": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__, "timm_version": timm.__version__,
        "numpy_version": np.__version__, "pillow_version": Image.__version__,
        "cuda_version": torch.version.cuda, "gpu_uuid": args.gpu_uuid,
        "physical_gpu_index": args.physical_index, "visible_name": torch.cuda.get_device_name(0),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32": False, "autocast": False, "startup_matmul_backward": "PASS",
    }
    registry = json.loads((EP0 / "model_registry.json").read_text())
    spec = next(row for row in registry["models"] if row["model_id"] == args.model_id)
    receipt = json.loads((ROOT / "step4_runtime/discovery_receipt.json").read_text())
    model, model_binding = load_model(spec, device)
    image, original_size = load_image(receipt, spec, device)
    class_id = int(receipt["class_id"])
    nodes512, weights512 = gauss_legendre(512, device, torch.float32)
    directions = patch_quadrant_directions(image)

    def endpoint():
        root = image.detach().requires_grad_(True)
        score = model(root)[:, class_id].sum()
        return root * torch.autograd.grad(score, root)[0]
    endpoint_map, endpoint_cost = timed(device, endpoint)
    endpoint_cost.update({"logical_model_forwards": 1, "logical_reverse_products": 1,
                          "integration_nodes": 0, "summary": tensor_summary(endpoint_map)})

    ig_result, ig_cost = timed(
        device, lambda: integrate_logit(model, image, class_id, steps=512, alpha_chunk=8),
    )
    ig_map = ig_result.pop("map")
    ig_cost.update({"logical_model_forwards": 514, "logical_reverse_products": 512,
                    "integration_nodes": 512, "result": ig_result,
                    "summary": tensor_summary(ig_map)})

    def pcia():
        accumulated = None
        for node, weight in zip(nodes512, weights512):
            root = (image * float(node)).detach().requires_grad_(True)
            fields, names = port_fields(model, root, block_index=0, class_id=class_id, second_order=False)
            term = (image.unsqueeze(0) * fields).detach().double() * float(weight)
            accumulated = term if accumulated is None else accumulated.add(term)
        return accumulated, names
    (pcia_value, pcia_names), pcia_cost = timed(device, pcia)
    batches_per_field = 10 + 4 * math.ceil(int(spec["heads"]) / 3)
    pcia_cost.update({
        "logical_model_forwards": 512, "logical_base_vjp_calls": 512 * batches_per_field,
        "integration_nodes": 512, "port_count_including_full": len(pcia_names),
        "summary": tensor_summary(pcia_value),
    })

    integration, pefa_cost = timed(
        device, lambda: stream_path_integrals(
            model, image, block_index=0, class_id=class_id,
            nodes=nodes512.tolist(), weights=weights512.tolist(), directions=directions,
            collect_group_diagnostics=True,
        ),
    )
    pefa_cost.update({
        "logical_model_forwards": int(integration["port_field_evaluations"]),
        "logical_base_vjp_calls": int(integration["port_field_evaluations"]) * batches_per_field,
        "logical_jvp_calls": int(integration["port_field_evaluations"]),
        "integration_nodes": 512, "patch_group_count": 4,
    })
    validation = summarize_validation(
        model, image, model_id=args.model_id, block_index=0, class_id=class_id,
        integration=integration, omega_directions=directions[:2],
    )
    all_blocks = first_order_all_blocks(model, image, class_id=class_id)
    controls = negative_controls(
        model, image, block_index=0, class_id=class_id, integration=integration,
    )
    retained_summaries = {key: tensor_summary(integration[key]) for key in ("A", "R", "T", "IG")}
    result = {
        "schema_version": 1, "stage": "ENDPOINT_PCIA_PEFA_STEP4",
        "status": "PASS" if validation["pass"] and all(row["pass"] for row in all_blocks) and all(row["pass"] for row in controls) else "FAIL",
        "model_id": args.model_id, "sample_id": receipt["sample_id"],
        "split": "discovery", "split_order": 0, "class_id": class_id,
        "image_sha256": receipt["image_sha256"], "original_size_wh": original_size,
        "input_shape": list(image.shape), "model_binding": model_binding,
        "worker_identity": identity, "primary_block_validation": validation,
        "all_blocks_first_order_validation": all_blocks, "negative_controls": controls,
        "retained_tensor_summaries": retained_summaries,
        "costs": {"endpoint": endpoint_cost, "ig": ig_cost, "pcia": pcia_cost, "pefa": pefa_cost},
        "process_peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "raw_hessian_materialized": False, "raw_port_by_input_by_input_materialized": False,
        "raw_result_arrays_retained": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "model_id": args.model_id,
                      "pefa_seconds": pefa_cost["wall_seconds"],
                      "peak_reserved": pefa_cost["peak_reserved_gpu_bytes"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
