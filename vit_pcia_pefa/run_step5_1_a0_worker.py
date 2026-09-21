#!/usr/bin/env python3
"""Run one frozen real-model A0 development task on one H20."""
from __future__ import annotations

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from run_step4_worker import load_image, load_model  # noqa: E402
from vit_pcia_pefa.src.pilot import closure_summary  # noqa: E402
from vit_pcia_pefa.src.step5_1_localization import (  # noqa: E402
    capture_projection_input, clean_parity, cut_formation_fields, difference_direction,
    endpoint_and_logit, localization_scores, projection_head_wrapper, rank_metrics,
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if hashlib.sha256(canonical(value)).hexdigest() != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def verify_binding(binding: dict[str, Any]) -> None:
    for row in (*binding["files"], *binding["checkpoints"], *binding["images"]):
        path = REPO / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError("FAIL_SOURCE_BINDING:" + row["path"])


def write_new(path: Path, value: dict[str, Any]) -> None:
    body = dict(value); body["canonical_digest"] = hashlib.sha256(canonical(body)).hexdigest()
    payload = json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    started = time.perf_counter(); run = args.root.resolve()
    output = run / "a0" / f"{args.model_id}.json"
    protocol = verify_signed(run / "protocol.json")
    manifest = verify_signed(run / "task_manifest.json")
    binding = verify_signed(run / "source_binding.json"); verify_binding(binding)
    task = next(row for row in manifest["a0_tasks"] if row["model_id"] == args.model_id)
    registry = json.loads((REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json").read_text())
    spec = next(row for row in registry["models"] if row["model_id"] == args.model_id)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).requires_grad_()
    (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model, model_binding = load_model(spec, device)
    image, original_size = load_image(
        {"cache_path": task["image_path"], "image_bytes": task["image_bytes"],
         "image_sha256": task["image_sha256"]}, spec, device,
    )
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(protocol["integration"]["real_nodes"])
    nodes, weights = (raw_nodes + 1) / 2, raw_weights / 2
    block, class_id, true_head = int(task["block_index"]), int(task["class_id"]), int(task["true_head"])
    activation = capture_projection_input(model, image, block)
    clean_endpoint, clean_logit, clean_prediction = endpoint_and_logit(model, image, class_id)
    clean = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=nodes, weights=weights)
    rows = []
    for strength in protocol["wrapper"]["a0_strengths"]:
        with projection_head_wrapper(model, block, true_head, activation, strength=float(strength)):
            changed_activation = capture_projection_input(model, image, block)
            changed_endpoint, changed_logit, changed_prediction = endpoint_and_logit(model, image, class_id)
            changed = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=nodes, weights=weights)
        parity_row = clean_parity(
            clean, changed, clean_endpoint=clean_endpoint, modified_endpoint=changed_endpoint,
            clean_logit=clean_logit, modified_logit=changed_logit, clean_prediction=clean_prediction,
            modified_prediction=changed_prediction,
            **{k: protocol["numeric"]["real"][k] for k in ("atol", "rtol", "near_zero")},
        )
        methods = localization_scores(
            clean, changed, clean_activation=activation, modified_activation=changed_activation,
            random_namespace=f"{protocol['run_id']}|a0|{args.model_id}|{strength}",
        )
        rows.append({
            "strength": strength, "clean_parity": parity_row,
            "clean_closure": closure_summary(model, block, clean, protocol["cut_id"]),
            "modified_closure": closure_summary(model, block, changed, protocol["cut_id"]),
            "pefa_R_rank": rank_metrics(methods["pefa_R_delta"], true_head, 1),
            "true_head_pefa_R_delta": methods["pefa_R_delta"][true_head],
            "difference_direction": difference_direction(clean, changed),
        })
    controls = []
    for mode, strength in (("bounded_quadratic", 0.0), ("linear_tangent", 1.0)):
        with projection_head_wrapper(model, block, true_head, activation, strength=strength, mode=mode):
            changed_activation = capture_projection_input(model, image, block)
            changed = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=nodes, weights=weights)
        methods = localization_scores(clean, changed, clean_activation=activation, modified_activation=changed_activation,
                                      random_namespace=f"{protocol['run_id']}|a0|{args.model_id}|{mode}")
        controls.append({
            "mode": mode, "strength": strength,
            "maximum_change_score": max(max(methods[key]) for key in
                ("pefa_R_delta", "pcia_delta", "endpoint_delta", "gradient_delta", "activation_delta")),
            "closure": closure_summary(model, block, changed, protocol["cut_id"]),
        })
    signals = [row["true_head_pefa_R_delta"] for row in rows]
    tolerance = protocol["numeric"]["real"]["atol"]
    passed = (
        all(row["clean_parity"]["pass"] and row["clean_closure"]["pass"]
            and row["modified_closure"]["pass"] and row["difference_direction"]["pass"] for row in rows)
        and signals[0] > tolerance and all(a <= b + tolerance for a, b in zip(signals, signals[1:]))
        and all(row["maximum_change_score"] <= tolerance and row["closure"]["pass"] for row in controls)
    )
    torch.cuda.synchronize()
    result = {
        "schema_version": 1, "run_id": protocol["run_id"], "task_id": task["task_id"],
        "status": "PASS_A0_REAL_MODEL" if passed else "FAIL_A0_REAL_MODEL",
        "model_id": args.model_id, "sample_id": task["sample_id"],
        "class_id": class_id, "true_head": true_head, "original_size_wh": original_size,
        "rows": rows, "controls": controls,
        "dose_nondecreasing": all(a <= b + tolerance for a, b in zip(signals, signals[1:])),
        "model_binding": model_binding, "scientific_claim": False,
        "worker_identity": {
            "sys_executable_realpath": str(Path(sys.executable).resolve()), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu_uuid": args.gpu_uuid,
            "physical_gpu_index": args.physical_index, "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32": False, "autocast": False, "startup_matmul_backward": "PASS",
        },
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
        "completed_unix_ns": time.time_ns(),
    }
    write_new(output, result)
    print(json.dumps({"status": result["status"], "model_id": args.model_id, "signals": signals}, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "INVALID_A0_WORKER", "type": type(error).__name__,
                          "message": str(error), "traceback": traceback.format_exc()}), flush=True)
        raise
