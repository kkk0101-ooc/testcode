#!/usr/bin/env python3
"""One deterministic ENDPOINT-E4 GPU worker."""
from __future__ import annotations

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import hashlib
import io
import json
import platform
import resource
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision
from PIL import Image
from timm.layers import set_fused_attn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e2_atlas import canonical, file_sha256  # noqa: E402
from vitnns.endpoint_e4_intervention import scan_e4_task  # noqa: E402
from vitnns.endpoint_ep1_queue import claim_task, complete_task, heartbeat  # noqa: E402
from vitnns.recovery_v2 import canonical_state_dict_tensor_digest  # noqa: E402


def load_model(spec: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    path = REPO / spec["checkpoint"]
    if path.stat().st_size != spec["checkpoint_bytes"] or file_sha256(path) != spec["checkpoint_sha256"]:
        raise ValueError("CHECKPOINT_BYTE_IDENTITY")
    with path.open("rb") as stream:
        loaded = torch.load(stream, map_location="cpu", weights_only=True)
    state = loaded["model"] if isinstance(loaded, dict) and isinstance(loaded.get("model"), dict) else loaded
    tensor_digest = canonical_state_dict_tensor_digest(state)
    if tensor_digest != spec["tensor_digest"]:
        raise ValueError("CHECKPOINT_TENSOR_IDENTITY")
    set_fused_attn(False)
    model = timm.create_model(spec["timm_name"], pretrained=False).eval()
    outcome = model.load_state_dict(state, strict=True)
    if outcome.missing_keys or outcome.unexpected_keys:
        raise ValueError("CHECKPOINT_STRICT_LOAD")
    for block in model.blocks:
        block.attn.fused_attn = False
    if len(model.blocks) != 12 or model.embed_dim != spec["embedding_dim"]:
        raise ValueError("MODEL_ARCHITECTURE")
    model.to(device)
    return model, {
        "checkpoint_sha256": spec["checkpoint_sha256"], "tensor_digest": tensor_digest,
        "key_count": len(state), "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def load_inputs(task: dict, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[int]]:
    path = REPO / task["cache_path"]
    data = path.read_bytes()
    if len(data) != task["image_bytes"] or hashlib.sha256(data).hexdigest() != task["image_sha256"]:
        raise ValueError("IMAGE_BYTE_IDENTITY")
    with Image.open(io.BytesIO(data)) as source:
        if source.format != "JPEG":
            raise ValueError("IMAGE_FORMAT")
        rgb = source.convert("RGB"); original_size = list(rgb.size)
        resized = TF.resize(rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True)
        cropped = TF.center_crop(resized, [224, 224])
        raw = TF.pil_to_tensor(cropped).float().div(255).unsqueeze(0)
    mean = torch.tensor(task["mean"], dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor(task["std"], dtype=torch.float32).reshape(1, 3, 1, 1)
    clean = (raw - mean) / std
    blur_raw = TF.gaussian_blur(raw, kernel_size=[31, 31], sigma=[10.0, 10.0])
    references = {
        "normalized_zero": torch.zeros_like(clean).to(device),
        "blur_rgb": ((blur_raw - mean) / std).to(device),
    }
    return clean.to(device), references, original_size


def abandon_task(path: Path, claim: dict, error: dict) -> None:
    with sqlite3.connect(path, isolation_level=None, timeout=30) as connection:
        connection.execute("PRAGMA journal_mode=WAL"); connection.execute("BEGIN IMMEDIATE")
        current = connection.execute(
            "SELECT state,attempt,fencing_token,worker_id,device_uuid FROM tasks WHERE task_id=?",
            (claim["task_id"],),
        ).fetchone()
        valid = current is not None and current[0] == "claimed" and (
            int(current[1]), int(current[2]), current[3], current[4]
        ) == (claim["attempt"], claim["fencing_token"], claim["worker_id"], claim["device_uuid"])
        if not valid:
            connection.rollback(); raise RuntimeError("STALE_FENCING_TOKEN")
        terminal = claim["attempt"] >= 2
        connection.execute(
            "UPDATE attempts SET status=?,completed_ns=? WHERE task_id=? AND attempt=?",
            ("FAILED" if terminal else "RETRYABLE", time.time_ns(), claim["task_id"], claim["attempt"]),
        )
        connection.execute(
            "UPDATE tasks SET state=?,fencing_token=NULL,worker_id=NULL,device_uuid=NULL,"
            "lease_until_ns=NULL,heartbeat_ns=NULL WHERE task_id=?",
            ("failed" if terminal else "pending", claim["task_id"]),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            (f"failure:{claim['task_id']}:{claim['attempt']}", canonical(error).decode()),
        )
        connection.commit()


def worker_identity(device: torch.device, args: argparse.Namespace) -> dict:
    probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).requires_grad_()
    (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize()
    return {
        "sys_executable_realpath": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__, "timm_version": timm.__version__,
        "numpy_version": np.__version__, "pillow_version": Image.__version__,
        "cuda_version": torch.version.cuda, "gpu_uuid": args.gpu_uuid,
        "physical_gpu_index": args.physical_index, "visible_device_name": torch.cuda.get_device_name(0),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "autocast": False,
        "startup_matmul_backward": "PASS",
    }


def execute(model, task: dict, claim: dict, queue: Path, device: torch.device) -> dict:
    task_started = time.perf_counter()
    if shutil.disk_usage(REPO).free < 50 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_DISK")
    image, references, original_size = load_inputs(task, device)
    reference_identity = {name: {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "tensor_sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
    } for name, value in references.items()}
    last_heartbeat = time.monotonic()

    def progress() -> None:
        nonlocal last_heartbeat
        if time.monotonic() - last_heartbeat >= 25:
            heartbeat(queue, claim, lease_seconds=180); last_heartbeat = time.monotonic()

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
    scan = scan_e4_task(
        model, image, references, class_id=task["class_id"], task_seed=claim["seed"],
        include_local=bool(task["include_local"]),
        include_secondary=bool(task["include_secondary"]), chunk_size=task["intervention_batch_size"],
        progress=progress,
    )
    torch.cuda.synchronize(); scan_seconds = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    task_seconds = time.perf_counter() - task_started
    if task_seconds > 1800:
        raise RuntimeError("TASK_TIMEOUT")
    if peak_reserved > 24 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_GPU_MEMORY")
    if rss_bytes > 32 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_CPU_RSS")
    return {
        "stage": "ENDPOINT-E4", "claim_scope": "LOCAL_CORRECTNESS_AND_FINITE_REPLACEMENT_NO_FINAL_CLAIM_DECISION",
        "model_id": task["model_id"], "sample_id": task["sample_id"],
        "split_order": task["split_order"], "class_id": task["class_id"], "wnid": task["wnid"],
        "image_sha256": task["image_sha256"], "image_bytes": task["image_bytes"],
        "original_size_wh": original_size, "input_shape": list(image.shape), "dtype": str(image.dtype),
        "reference_identity": reference_identity, **scan, "scan_seconds": scan_seconds,
        "task_seconds": task_seconds, "peak_allocated_gpu_bytes": peak_allocated,
        "peak_reserved_gpu_bytes": peak_reserved, "peak_cpu_rss_bytes": rss_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True); parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--worker-id", required=True); parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True); args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0"); identity = worker_identity(device, args)
    models = {}; completed = 0
    while True:
        claim = claim_task(args.queue, worker_id=args.worker_id, device_uuid=args.gpu_uuid, lease_seconds=180, max_attempts=2)
        if claim is None:
            break
        task = claim["payload"]
        try:
            if task["model_id"] not in models:
                models[task["model_id"]] = load_model(task, device)
            model, binding = models[task["model_id"]]
            result = execute(model, task, claim, args.queue, device)
            result.update({"model_binding": binding, "worker_identity": identity})
            complete_task(args.queue, args.results, claim, result); completed += 1
        except Exception as error:
            failure = {"error_type": type(error).__name__, "message": str(error)}
            abandon_task(args.queue, claim, failure)
            print(json.dumps({"status": "TASK_FAILED", "task_id": claim["task_id"], "attempt": claim["attempt"], **failure}, sort_keys=True), flush=True)
    print(json.dumps({"status": "WORKER_EXIT", "worker_id": args.worker_id, "completed": completed}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
