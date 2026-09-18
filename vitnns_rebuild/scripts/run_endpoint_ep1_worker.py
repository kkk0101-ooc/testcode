#!/usr/bin/env python3
"""Run one Endpoint Scanner EP1 queue worker on a qualified GPU."""
from __future__ import annotations

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be set before Torch import")

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision
from timm.layers import set_fused_attn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_ep1_queue import claim_task, complete_task  # noqa: E402
from vitnns.attribution import integrated_gradients  # noqa: E402
from vitnns.endpoint_vit import (  # noqa: E402
    EndpointTarget, capture_unfused_attention_edges, decompose_endpoint_vit, scan_endpoint_vit,
)
from vitnns.recovery_v2 import canonical_state_dict_tensor_digest  # noqa: E402


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_model(spec: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    path = REPO / spec["checkpoint"]
    if path.stat().st_size != spec["checkpoint_bytes"] or sha256(path) != spec["checkpoint_sha256"]:
        raise ValueError("CHECKPOINT_BYTE_IDENTITY")
    with path.open("rb") as stream:
        loaded = torch.load(stream, map_location="cpu", weights_only=True)
    state = loaded["model"] if "model" in loaded and isinstance(loaded["model"], dict) else loaded
    set_fused_attn(False)
    model = timm.create_model(spec["timm_name"], pretrained=False).eval()
    outcome = model.load_state_dict(state, strict=True)
    if tuple(outcome.missing_keys) or tuple(outcome.unexpected_keys):
        raise ValueError("CHECKPOINT_STRICT_LOAD")
    for block in model.blocks:
        block.attn.fused_attn = False
    if len(model.blocks) != spec["depth"] or model.embed_dim != spec["embedding_dim"]:
        raise ValueError("MODEL_ARCHITECTURE")
    return model.to(device), {
        "tensor_digest": canonical_state_dict_tensor_digest(state),
        "key_count": len(state), "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def rel_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).norm() / right.norm().clamp_min(1e-30))


def tensor_digest(values: list[torch.Tensor]) -> str:
    value = hashlib.sha256()
    for tensor in values:
        array = tensor.detach().cpu().contiguous().numpy()
        value.update(str(array.dtype).encode())
        value.update(str(array.shape).encode())
        value.update(array.tobytes(order="C"))
    return value.hexdigest()


def model_task(model: torch.nn.Module, spec: dict, seed: int, device: torch.device) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = torch.randn((1, 3, 224, 224), generator=generator, dtype=torch.float32).to(device)
    no_hook_start = time.perf_counter()
    with torch.no_grad():
        baseline = model(image)
    torch.cuda.synchronize()
    no_hook_seconds = time.perf_counter() - no_hook_start

    calls = []
    handle = model.blocks[0].register_forward_hook(lambda *_: calls.append(1))
    try:
        with torch.no_grad():
            noop = model(image)
    finally:
        handle.remove()
    if calls != [1]:
        raise ValueError("NOOP_HOOK_COUNT")

    targets = []
    blocks = spec["blocks"]
    for block in blocks:
        targets.extend((
            EndpointTarget(f"b{block}_unit", f"blocks.{block}", kind="token_channel", token=1, channel=0),
            EndpointTarget(f"b{block}_sum", f"blocks.{block}", kind="patch_channel_sum", channel=0),
            EndpointTarget(f"b{block}_energy", f"blocks.{block}", kind="patch_channel_energy", channel=0),
            EndpointTarget(f"b{block}_head", f"blocks.{block}.attn.proj", kind="head_channel", location="input", head=0, token=0, channel=0),
        ))
    targets.extend((
        EndpointTarget("patch_embedding", "patch_embed", kind="patch_channel_sum", channel=0),
        EndpointTarget("final_cls", "norm", kind="token_channel", token=0, channel=0),
        EndpointTarget("ground_truth_logit", "", kind="ground_truth_logit", class_id=seed % 1000),
        EndpointTarget("frozen_clean_top1", "", kind="frozen_clean_top1_logit"),
    ))
    torch.cuda.reset_peak_memory_stats()
    single_start = time.perf_counter()
    single = scan_endpoint_vit(model, image, [targets[-2]])
    torch.cuda.synchronize()
    single_target_seconds = time.perf_counter() - single_start
    start = time.perf_counter()
    scan = scan_endpoint_vit(model, image, targets)
    torch.cuda.synchronize()
    scan_seconds = time.perf_counter() - start
    if scan.forward_count != 1 or set(scan.units) != {target.name for target in targets}:
        raise ValueError("ALL_BLOCK_TARGET_COVERAGE")
    for target in targets:
        unit = scan.units[target.name]
        if not bool(torch.isfinite(unit.attribution).all()):
            raise ValueError("NONFINITE_TARGET")
        if target.kind == "patch_channel_energy":
            expected = 0.5 * unit.token_values[:, model.num_prefix_tokens:].square().sum(1)
            # ``unit.value`` is reduced on the GPU graph while token_values are
            # detached to CPU; the target uses the same captured activation, but
            # cross-device float32 reduction order is only tolerance-equivalent.
            torch.testing.assert_close(unit.value, expected, atol=1e-5, rtol=1e-4)

    decomposition_start = time.perf_counter()
    decomposition_blocks = spec.get("decomposition_blocks", blocks)
    decompositions = [decompose_endpoint_vit(
        model, image, block=block,
        target=EndpointTarget("ground_truth_logit", "", kind="ground_truth_logit", class_id=seed % 1000),
        atol=1e-5, rtol=1e-4,
    ) for block in decomposition_blocks]
    torch.cuda.synchronize()
    decomposition_seconds = time.perf_counter() - decomposition_start
    ig_benchmarks = []
    if spec["repeat"] == 0:
        class_id = seed % 1000
        for nodes in (4, 8):
            ig_start = time.perf_counter()
            ig = integrated_gradients(
                lambda value, class_id=class_id: model(value)[:, class_id],
                image, torch.zeros_like(image), steps=nodes, step_chunk_size=4,
                atol=1e9, rtol=1e9, parameters=model,
            )
            torch.cuda.synchronize()
            ig_benchmarks.append({
                "M": nodes, "seconds": time.perf_counter() - ig_start,
                "finite": bool(torch.isfinite(ig.attribution).all()),
            })
    formal_edges = {}
    with capture_unfused_attention_edges(model.blocks[blocks[0]].attn, formal_edges):
        with torch.no_grad():
            formal_hook_logits = model(image)

    fused = None
    fused_supported = True
    try:
        for block in model.blocks:
            block.attn.fused_attn = True
        with torch.no_grad():
            fused = model(image)
    except (RuntimeError, NotImplementedError) as error:
        fused_supported = False
        fused_error = f"{type(error).__name__}: {error}"
    finally:
        for block in model.blocks:
            block.attn.fused_attn = False
    fused_max_abs = None if fused is None else float((fused - baseline).abs().max())
    if fused is not None:
        torch.testing.assert_close(fused, baseline, atol=1e-4, rtol=1e-4)

    hook_max_abs = float((noop - baseline).abs().max())
    formal_max_abs = float((formal_hook_logits - baseline).abs().max())
    if hook_max_abs != 0.0 or formal_max_abs != 0.0:
        raise ValueError("HOOK_FORWARD_PARITY")
    return {
        "model_id": spec["model_id"], "seed": seed, "target_count": len(targets),
        "blocks": blocks, "decomposition_blocks": decomposition_blocks, "target_k": len(targets),
        "integration_m": 0, "component_counts": sorted({len(row.components) for row in decompositions}),
        "no_hook_seconds": no_hook_seconds, "single_target_seconds": single_target_seconds,
        "scan_seconds": scan_seconds, "ig_benchmarks": ig_benchmarks,
        "decomposition_seconds": decomposition_seconds,
        "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(),
        "hook_parity_max_abs": hook_max_abs, "formal_hook_parity_max_abs": formal_max_abs,
        "fused_supported": fused_supported,
        "fused_unfused_max_abs": fused_max_abs,
        "fused_unfused_parity_pass": fused_supported,
        "fused_error": None if fused_supported else fused_error,
        "fused_structural_support": False, "unfused_structural_support": True,
        "input_dependent_attention_bias": False,
        "captured_edges": sorted(set().union(*(row.captured_edges for row in decompositions))),
        "closure_max_abs": max(max(row.closure_max_abs.values()) for row in decompositions),
        "closure_relative_l2": max(max(row.closure_relative_l2.values()) for row in decompositions),
        "numeric_digest": tensor_digest(
            [scan.logits, single.logits] + [scan.units[name].attribution for name in sorted(scan.units)]
            + [value for row in decompositions for key, value in sorted(row.components.items())]
        ),
    }


def worker_identity(device: torch.device, physical_index: int, gpu_uuid: str) -> dict:
    probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).requires_grad_()
    value = (probe @ probe.T).square().sum(); value.backward(); torch.cuda.synchronize()
    return {
        "sys_executable_realpath": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "timm_version": timm.__version__, "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__, "cuda_version": torch.version.cuda,
        "physical_gpu_index": physical_index, "gpu_uuid": gpu_uuid,
        "visible_device_name": torch.cuda.get_device_name(0),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "startup_matmul_backward": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("worker CUDA_VISIBLE_DEVICES drift")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    identity = worker_identity(device, args.physical_index, args.gpu_uuid)
    models: dict[str, tuple[torch.nn.Module, dict]] = {}
    completed = 0
    while True:
        claim = claim_task(args.queue, worker_id=args.worker_id, device_uuid=args.gpu_uuid)
        if claim is None:
            break
        spec = claim["payload"]
        if spec["model_id"] not in models:
            models[spec["model_id"]] = load_model(spec, device)
        model, binding = models[spec["model_id"]]
        try:
            result = model_task(model, spec, claim["seed"], device)
            result.update({"model_binding": binding, "worker_identity": identity})
            complete_task(args.queue, args.result_root, claim, result)
            completed += 1
        except Exception as error:
            print(json.dumps({
                "status": "TASK_FAILED", "task_id": claim["task_id"],
                "attempt": claim["attempt"], "error_type": type(error).__name__,
                "message": str(error),
            }), flush=True)
            raise
    print(json.dumps({"status": "PASS", "worker_id": args.worker_id, "completed": completed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
