#!/usr/bin/env python3
"""One deterministic GPU worker for ENDPOINT-E2 atlas production/replay."""
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
from vitnns.endpoint_e2_atlas import (  # noqa: E402
    METHOD_ID, canonical, digest, file_sha256, registered_targets, scan_registered_targets,
    write_npy_new,
)
from vitnns.endpoint_ep1_queue import claim_task, complete_task, heartbeat  # noqa: E402
from vitnns.recovery_v2 import canonical_state_dict_tensor_digest  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text())


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
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "tensor_digest": tensor_digest, "key_count": len(state),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_image(task: dict, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    path = REPO / task["cache_path"]
    data = path.read_bytes()
    if len(data) != task["image_bytes"] or hashlib.sha256(data).hexdigest() != task["image_sha256"]:
        raise ValueError("IMAGE_BYTE_IDENTITY")
    with Image.open(io.BytesIO(data)) as source:
        if source.format != "JPEG":
            raise ValueError("IMAGE_FORMAT")
        rgb = source.convert("RGB")
        original_size = list(rgb.size)
        resized = TF.resize(rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True)
        cropped = TF.center_crop(resized, [224, 224])
        value = TF.pil_to_tensor(cropped).float().div(255).unsqueeze(0)
    mean = torch.tensor(task["mean"], dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor(task["std"], dtype=torch.float32).reshape(1, 3, 1, 1)
    return ((value - mean) / std).to(device), original_size


def abandon_task(path: Path, claim: dict, error: dict) -> None:
    with sqlite3.connect(path, isolation_level=None, timeout=30) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("BEGIN IMMEDIATE")
        current = connection.execute(
            "SELECT state,attempt,fencing_token,worker_id,device_uuid FROM tasks WHERE task_id=?",
            (claim["task_id"],),
        ).fetchone()
        valid = current is not None and current[0] == "claimed" and (
            int(current[1]), int(current[2]), current[3], current[4]
        ) == (claim["attempt"], claim["fencing_token"], claim["worker_id"], claim["device_uuid"])
        if not valid:
            connection.rollback()
            raise RuntimeError("STALE_FENCING_TOKEN")
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
    (probe @ probe.T).square().sum().backward()
    torch.cuda.synchronize()
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


def execute(
    model: torch.nn.Module, task: dict, claim: dict, queue: Path, artifact_root: Path,
    mode: str, device: torch.device,
) -> dict:
    task_started = time.perf_counter()
    if shutil.disk_usage(REPO).free < 50 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_DISK")
    target_path = REPO / task["target_registry"]
    if file_sha256(target_path) != task["target_registry_sha256"]:
        raise ValueError("TARGET_REGISTRY_IDENTITY")
    registry = load_json(target_path)
    rows = registry["targets_by_model"][task["model_id"]]
    if len(rows) != task["target_count"] or digest(rows) != task["target_rows_digest"]:
        raise ValueError("TARGET_REGISTRY_COVERAGE")
    targets = registered_targets(rows, ground_truth_class_id=task["class_id"])
    image, original_size = load_image(task, device)
    last_heartbeat = time.monotonic()

    def progress() -> None:
        nonlocal last_heartbeat
        if time.monotonic() - last_heartbeat >= 25:
            heartbeat(queue, claim, lease_seconds=180)
            last_heartbeat = time.monotonic()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    scan = scan_registered_targets(
        model, image, targets, retain_full_atlas=bool(task["retain_full_atlas"] and mode == "formal"),
        near_zero_max_abs=1e-12, progress=progress,
    )
    torch.cuda.synchronize()
    scan_seconds = time.perf_counter() - started
    artifacts = []
    if mode == "formal":
        class_path = (artifact_root / "class_maps" / task["model_id"] /
                      f"{task['split_order']:03d}" / f"attempt-{claim['attempt']:02d}.npy")
        info = write_npy_new(class_path, scan.class_maps)
        info["path"] = class_path.relative_to(REPO).as_posix()
        info.update({"kind": "raw_signed_rgb_class_maps", "shape": list(scan.class_maps.shape), "dtype": "float32"})
        artifacts.append(info)
        if scan.atlas_maps is not None:
            atlas_path = (artifact_root / "full_atlas" / task["model_id"] /
                          f"{task['split_order']:03d}" / f"attempt-{claim['attempt']:02d}.npy")
            info = write_npy_new(atlas_path, scan.atlas_maps)
            info["path"] = atlas_path.relative_to(REPO).as_posix()
            info.update({"kind": "full_raw_atlas", "shape": list(scan.atlas_maps.shape), "dtype": "float32"})
            artifacts.append(info)
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
        "stage": "ENDPOINT-E2", "mode": mode, "method_id": METHOD_ID,
        "model_id": task["model_id"], "sample_id": task["sample_id"],
        "split_order": task["split_order"], "class_id": task["class_id"],
        "wnid": task["wnid"], "image_sha256": task["image_sha256"],
        "image_bytes": task["image_bytes"], "original_size_wh": original_size,
        "input_shape": list(image.shape), "dtype": str(image.dtype),
        "target_registry_sha256": task["target_registry_sha256"],
        "target_rows_digest": task["target_rows_digest"],
        "target_count": len(scan.summaries), "target_ids_digest": digest([row["target_id"] for row in scan.summaries]),
        "ground_truth_class_id": task["class_id"],
        "clean_top1_class_id": scan.clean_top1_class_id, "logits_sha256": scan.logits_sha256,
        "forward_count": scan.forward_count, "summaries": scan.summaries,
        "summaries_digest": digest(scan.summaries), "numeric_digest": scan.numeric_digest,
        "near_zero_count": sum(row["near_zero"] for row in scan.summaries),
        "finite_target_count": sum(row["finite"] for row in scan.summaries),
        "scan_seconds": scan_seconds, "peak_allocated_gpu_bytes": peak_allocated,
        "task_seconds": task_seconds,
        "peak_reserved_gpu_bytes": peak_reserved, "peak_cpu_rss_bytes": rss_bytes,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--mode", choices=("formal", "replay"), required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    identity = worker_identity(device, args)
    models = {}
    completed = 0
    while True:
        claim = claim_task(
            args.queue, worker_id=args.worker_id, device_uuid=args.gpu_uuid,
            lease_seconds=180, max_attempts=2,
        )
        if claim is None:
            break
        task = claim["payload"]
        try:
            if task["model_id"] not in models:
                models[task["model_id"]] = load_model(task, device)
            model, binding = models[task["model_id"]]
            result = execute(model, task, claim, args.queue, args.artifacts, args.mode, device)
            result.update({"model_binding": binding, "worker_identity": identity})
            complete_task(args.queue, args.results, claim, result)
            completed += 1
        except Exception as error:
            failure = {"error_type": type(error).__name__, "message": str(error)}
            abandon_task(args.queue, claim, failure)
            print(json.dumps({
                "status": "TASK_FAILED", "task_id": claim["task_id"],
                "attempt": claim["attempt"], **failure,
            }, sort_keys=True), flush=True)
    print(json.dumps({
        "status": "WORKER_EXIT", "worker_id": args.worker_id,
        "mode": args.mode, "completed": completed,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
