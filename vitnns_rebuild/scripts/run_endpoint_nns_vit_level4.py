#!/usr/bin/env python3
"""Twenty-image, eight-GPU exploratory comparison of endpoint and path maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_level4_r1_20img_20260917"
MODEL = REPO / "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"
IMAGE_ROOT = REPO / "assets/data/ILSVRC2012_val"
SCRIPT = Path(__file__).resolve()
GPU_COUNT = 8
IMAGE_COUNT = 20
SCALES = (4, 16)
FRACTIONS = tuple(index / 20 for index in range(21))
PATH_SCHEDULE = (256, 512, 1024, 2048)
PATH_CHUNK = 4
FORWARD_CHUNK = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    payload = dict(value)
    payload["canonical_digest"] = canonical(value)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def gpu_snapshot() -> tuple[list[dict], list[dict]]:
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,compute_mode",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in query.splitlines():
        index, uuid, name, total, free, utilization, mode = [item.strip() for item in line.split(",")]
        rows.append({
            "physical_index": int(index), "uuid": uuid, "name": name,
            "memory_total_mib": int(total), "memory_free_mib": int(free),
            "utilization_percent": int(utilization), "compute_mode": mode,
        })
    process_output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    processes = []
    for line in process_output.splitlines():
        if not line.strip():
            continue
        uuid, pid, memory = [item.strip() for item in line.split(",")]
        processes.append({"gpu_uuid": uuid, "pid": int(pid), "used_memory_mib": int(memory)})
    return rows, processes


def prepare() -> None:
    if ROOT.exists():
        raise RuntimeError(f"level-4 root already exists: {ROOT}")
    ROOT.mkdir(parents=True)
    for folder in ("claims", "results", "failures", "workers"):
        (ROOT / folder).mkdir()
    candidates = sorted(IMAGE_ROOT.glob("ILSVRC2012_val_*.JPEG"))
    if len(candidates) < IMAGE_COUNT:
        raise RuntimeError("fewer than 20 local ImageNet validation images")
    ranked = sorted(candidates, key=lambda path: hashlib.sha256(path.name.encode()).hexdigest())
    selected = ranked[:IMAGE_COUNT]
    schedule = [{
        "schedule_index": index,
        "sample_id": path.name,
        "path": path.relative_to(REPO).as_posix(),
        "jpeg_sha256": sha256(path),
        "selection_key": hashlib.sha256(path.name.encode()).hexdigest(),
    } for index, path in enumerate(selected)]
    gpus, processes = gpu_snapshot()
    if len(gpus) != GPU_COUNT or [row["physical_index"] for row in gpus] != list(range(GPU_COUNT)):
        raise RuntimeError("exactly physical GPUs 0..7 are required")
    if any(row["name"] != "NVIDIA H20" or row["memory_free_mib"] < 40000 for row in gpus):
        raise RuntimeError("all eight H20 GPUs must have at least 40000 MiB free")
    protocol = {
        "status": "FROZEN_BEFORE_IMAGE_DECODE",
        "scope": "20-image post-hoc exploratory method-value pilot",
        "selection": "lowest SHA256(sample filename) among local ILSVRC2012 validation JPEGs",
        "image_count": IMAGE_COUNT,
        "methods": ["endpoint_gradient_times_input", "path_integrated_ig", "ordinary_gradient"],
        "target": "per-image clean predicted raw class logit, frozen for perturbations and flip",
        "references": ["normalized_zero_rgb_0.5", "normalized_minus_one_rgb_0"],
        "stability": "horizontal flip, inverse flip maps before comparison",
        "fidelity": {
            "scales": list(SCALES), "fractions": list(FRACTIONS),
            "directions": ["insertion", "deletion"],
            "ranking": "descending signed RGB-summed cell mass",
            "reference": "normalized_zero_rgb_0.5",
            "score": "0.5*(normalized insertion AUC-normalized deletion AUC)",
        },
        "path_schedule": list(PATH_SCHEDULE), "path_chunk": PATH_CHUNK,
        "go_to_50_gate": {
            "all_20_valid": True,
            "endpoint_identity_max_abs_lte": 1e-6,
            "endpoint_fine_median_fidelity_gt": 0.0,
            "endpoint_minus_path_fine_median_gte": -0.05,
            "endpoint_reference_pearson_median_gte": 0.5,
            "endpoint_flip_spearman_median_gte": 0.5,
            "endpoint_to_path_runtime_ratio_median_lte": 0.25,
        },
        "five_hundred_image_rerun_authorized": False,
        "gpu_execution": {
            "worker_count": GPU_COUNT, "physical_indices": list(range(GPU_COUNT)),
            "dynamic_atomic_claims": True, "static_sharding": False,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "gpu_snapshot": gpus,
            "preexisting_compute_processes": processes,
        },
        "model_sha256": sha256(MODEL),
        "script_sha256": sha256(SCRIPT),
    }
    write_json_new(ROOT / "protocol.json", protocol)
    write_json_new(ROOT / "schedule.json", {"records": schedule, "records_digest": canonical(schedule)})


def correlations(left, right) -> dict[str, float]:
    import numpy as np
    from scipy.stats import rankdata
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    ac, bc = a - a.mean(), b - b.mean()
    pearson_den = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    ar, br = rankdata(a), rankdata(b)
    arc, brc = ar - ar.mean(), br - br.mean()
    rank_den = float(np.linalg.norm(arc) * np.linalg.norm(brc))
    nonzero = (a != 0) | (b != 0)
    k = max(1, int(math.ceil(0.1 * a.size)))
    atop = set(np.argpartition(np.abs(a), -k)[-k:].tolist())
    btop = set(np.argpartition(np.abs(b), -k)[-k:].tolist())
    return {
        "pearson": float(np.dot(ac, bc) / pearson_den) if pearson_den else 1.0,
        "spearman": float(np.dot(arc, brc) / rank_den) if rank_den else 1.0,
        "sign_agreement": float(np.mean(np.sign(a[nonzero]) == np.sign(b[nonzero]))) if nonzero.any() else 1.0,
        "top10_abs_jaccard": len(atop & btop) / len(atop | btop),
        "l1_relative_to_right": float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30)),
    }


def aggregate_cells(spatial, unit: int):
    import numpy as np
    return np.asarray(spatial).reshape(224 // unit, unit, 224 // unit, unit).sum((1, 3))


def fidelity(model, image, baseline, class_id: int, spatial, unit: int) -> dict:
    import numpy as np
    import torch
    scores = aggregate_cells(spatial, unit).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    with torch.no_grad():
        clean_score = float(model(image)[0, class_id].item())
        baseline_score = float(model(baseline)[0, class_id].item())
    delta = clean_score - baseline_score
    if abs(delta) < 1e-6:
        return {"status": "INVALID_SMALL_TARGET_DELTA", "delta": delta}
    curves = {}
    for direction in ("insertion", "deletion"):
        batch = []
        for fraction in FRACTIONS:
            count = min(len(order), int(round(fraction * len(order))))
            value = baseline.clone() if direction == "insertion" else image.clone()
            for flat_index in order[:count]:
                row, column = divmod(int(flat_index), 224 // unit)
                row_slice = slice(row * unit, (row + 1) * unit)
                col_slice = slice(column * unit, (column + 1) * unit)
                source = image if direction == "insertion" else baseline
                value[:, :, row_slice, col_slice] = source[:, :, row_slice, col_slice]
            batch.append(value)
        raw = []
        with torch.no_grad():
            joined = torch.cat(batch, 0)
            for begin in range(0, joined.shape[0], FORWARD_CHUNK):
                raw.extend(model(joined[begin:begin + FORWARD_CHUNK])[:, class_id].detach().cpu().tolist())
        curves[direction] = [(score - baseline_score) / delta for score in raw]
    insertion_auc = float(np.trapz(curves["insertion"], FRACTIONS))
    deletion_auc = float(np.trapz(curves["deletion"], FRACTIONS))
    return {
        "status": "VALID", "delta": delta, "clean_score": clean_score,
        "baseline_score": baseline_score, "insertion_auc": insertion_auc,
        "deletion_auc": deletion_auc,
        "combined_fidelity": 0.5 * (insertion_auc - deletion_auc),
        "insertion_curve": curves["insertion"], "deletion_curve": curves["deletion"],
    }


def worker(root: Path, worker_id: str) -> None:
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(REPO / "src"))
    from vitnns.attribution import integrated_gradients
    from vitnns.endpoint_nns import endpoint_learning_image
    from vitnns.modeling import canonical_transform, load_canonical_model

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG missing before Torch import")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    model = load_canonical_model(
        MODEL, device=device, unfused_attention=True,
        expected_state_dict_sha256="489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d",
        expected_model_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        expected_model_identity="F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES",
        expected_canonical_tensor_digest="150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd",
    ).eval()
    model.requires_grad_(False)
    transform = canonical_transform(model)
    schedule = read_json(root / "schedule.json")["records"]
    for row in schedule:
        index = row["schedule_index"]
        result_path = root / "results" / f"{index:03d}.json"
        failure_path = root / "failures" / f"{index:03d}.json"
        if result_path.exists() or failure_path.exists():
            continue
        claim_path = root / "claims" / f"{index:03d}.claim"
        try:
            descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        os.write(descriptor, worker_id.encode())
        os.close(descriptor)
        started = time.monotonic()
        try:
            image_path = REPO / row["path"]
            if sha256(image_path) != row["jpeg_sha256"]:
                raise RuntimeError("JPEG digest changed")
            with Image.open(image_path) as source:
                image = transform(source.convert("RGB")).unsqueeze(0).to(device)
            zero = torch.zeros_like(image)
            black = torch.full_like(image, -1.0)
            with torch.no_grad():
                logits = model(image)
            class_id = int(logits.argmax(1).item())
            callback = lambda value: model(value)[:, class_id]
            methods = {}
            arrays = {}
            runtime = {}
            for reference_name, reference in (("zero", zero), ("black", black)):
                begin = time.monotonic()
                endpoint = endpoint_learning_image(callback, image, reference=reference, parameters=model)
                runtime[f"endpoint_{reference_name}"] = time.monotonic() - begin
                begin = time.monotonic()
                path_result = integrated_gradients(
                    callback, image, reference, steps=PATH_SCHEDULE,
                    step_chunk_size=PATH_CHUNK, atol=1e-4, rtol=0.01, parameters=model,
                    metadata={"scope": "endpoint_nns_vit_level4_20img"},
                )
                runtime[f"path_{reference_name}"] = time.monotonic() - begin
                if not path_result.converged:
                    raise RuntimeError(f"path IG did not converge for {reference_name}")
                endpoint_array = endpoint.attribution.detach().cpu().numpy()
                path_array = path_result.attribution.detach().cpu().numpy()
                arrays[f"endpoint_{reference_name}"] = endpoint_array
                arrays[f"path_{reference_name}"] = path_array
                methods[reference_name] = {
                    "endpoint_sum": endpoint.attribution_sum.detach().cpu().tolist(),
                    "endpoint_response_reconstruction_error": endpoint.response_reconstruction_error.detach().cpu().tolist(),
                    "path_steps": path_result.steps,
                    "path_conservation_abs": path_result.conservation["abs"].detach().cpu().tolist(),
                    "endpoint_vs_path": correlations(endpoint_array, path_array),
                }
            ordinary_gradient = endpoint_learning_image(callback, image, reference=zero, parameters=model).gradient.detach().cpu().numpy()
            endpoint_identity = float(np.max(np.abs(arrays["endpoint_zero"] - image.detach().cpu().numpy() * ordinary_gradient)))
            methods["ordinary_gradient"] = {
                "endpoint_identity_max_abs": endpoint_identity,
                "gradient_vs_endpoint_spatial": correlations(
                    ordinary_gradient.sum(1), arrays["endpoint_zero"].sum(1)
                ),
            }
            methods["reference_comparison"] = {
                "endpoint_zero_vs_black": correlations(arrays["endpoint_zero"], arrays["endpoint_black"]),
                "path_zero_vs_black": correlations(arrays["path_zero"], arrays["path_black"]),
            }

            flipped = torch.flip(image, dims=(-1,))
            flip_callback = lambda value: model(value)[:, class_id]
            begin = time.monotonic()
            flip_endpoint = endpoint_learning_image(flip_callback, flipped, reference=zero, parameters=model)
            runtime["endpoint_flip"] = time.monotonic() - begin
            begin = time.monotonic()
            flip_path = integrated_gradients(
                flip_callback, flipped, zero, steps=PATH_SCHEDULE, step_chunk_size=PATH_CHUNK,
                atol=1e-4, rtol=0.01, parameters=model,
                metadata={"scope": "endpoint_nns_vit_level4_flip"},
            )
            runtime["path_flip"] = time.monotonic() - begin
            if not flip_path.converged:
                raise RuntimeError("flip path IG did not converge")
            flip_endpoint_array = torch.flip(flip_endpoint.attribution, dims=(-1,)).detach().cpu().numpy()
            flip_path_array = torch.flip(flip_path.attribution, dims=(-1,)).detach().cpu().numpy()
            arrays["endpoint_flip_inverse"] = flip_endpoint_array
            arrays["path_flip_inverse"] = flip_path_array
            methods["flip_stability"] = {
                "endpoint": correlations(arrays["endpoint_zero"], flip_endpoint_array),
                "path": correlations(arrays["path_zero"], flip_path_array),
            }

            fidelity_results = {}
            for method_name in ("endpoint_zero", "path_zero"):
                spatial = arrays[method_name][0].sum(0)
                fidelity_results[method_name] = {
                    str(unit): fidelity(model, image, zero, class_id, spatial, unit)
                    for unit in SCALES
                }
            array_path = root / "results" / f"{index:03d}.npz"
            temporary = root / "results" / f".{index:03d}.{worker_id}.tmp"
            with temporary.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, array_path)
            result = {
                "status": "VALID", "schedule_index": index, "sample_id": row["sample_id"],
                "jpeg_sha256": row["jpeg_sha256"], "worker_id": worker_id,
                "physical_gpu_index": int(os.environ["PHYSICAL_GPU_INDEX"]),
                "gpu_name": torch.cuda.get_device_name(0), "class_id": class_id,
                "clean_logit": float(logits[0, class_id].item()),
                "methods": methods, "fidelity": fidelity_results, "runtime_seconds": runtime,
                "total_wall_seconds": time.monotonic() - started,
                "arrays": {"path": array_path.relative_to(root).as_posix(), "sha256": sha256(array_path)},
            }
            write_json_new(result_path, result)
            print(json.dumps({"worker": worker_id, "index": index, "status": "VALID"}), flush=True)
        except Exception as error:
            write_json_new(failure_path, {
                "status": "FAILED", "schedule_index": index, "sample_id": row["sample_id"],
                "worker_id": worker_id, "error_type": type(error).__name__, "message": str(error),
                "wall_seconds": time.monotonic() - started,
            })
            print(json.dumps({"worker": worker_id, "index": index, "status": "FAILED", "error": str(error)}), flush=True)


def median(values: list[float]) -> float:
    import numpy as np
    return float(np.median(np.asarray(values, dtype=np.float64)))


def finalize(root: Path, worker_exits: list[dict]) -> None:
    schedule = read_json(root / "schedule.json")["records"]
    results = []
    failures = []
    for row in schedule:
        result = root / "results" / f"{row['schedule_index']:03d}.json"
        failure = root / "failures" / f"{row['schedule_index']:03d}.json"
        if result.exists():
            results.append(read_json(result))
        elif failure.exists():
            failures.append(read_json(failure))
        else:
            failures.append({"schedule_index": row["schedule_index"], "status": "MISSING"})
    metrics: dict[str, Any] = {"valid_count": len(results), "failure_count": len(failures)}
    if results:
        metrics.update({
            "endpoint_identity_max_abs": max(r["methods"]["ordinary_gradient"]["endpoint_identity_max_abs"] for r in results),
            "endpoint_path_zero_pearson_median": median([r["methods"]["zero"]["endpoint_vs_path"]["pearson"] for r in results]),
            "endpoint_path_zero_sign_median": median([r["methods"]["zero"]["endpoint_vs_path"]["sign_agreement"] for r in results]),
            "endpoint_reference_pearson_median": median([r["methods"]["reference_comparison"]["endpoint_zero_vs_black"]["pearson"] for r in results]),
            "path_reference_pearson_median": median([r["methods"]["reference_comparison"]["path_zero_vs_black"]["pearson"] for r in results]),
            "endpoint_flip_spearman_median": median([r["methods"]["flip_stability"]["endpoint"]["spearman"] for r in results]),
            "path_flip_spearman_median": median([r["methods"]["flip_stability"]["path"]["spearman"] for r in results]),
            "endpoint_to_path_runtime_ratio_median": median([
                r["runtime_seconds"]["endpoint_zero"] / r["runtime_seconds"]["path_zero"] for r in results
            ]),
        })
        for unit in SCALES:
            endpoint_values = [r["fidelity"]["endpoint_zero"][str(unit)]["combined_fidelity"] for r in results if r["fidelity"]["endpoint_zero"][str(unit)]["status"] == "VALID"]
            path_values = [r["fidelity"]["path_zero"][str(unit)]["combined_fidelity"] for r in results if r["fidelity"]["path_zero"][str(unit)]["status"] == "VALID"]
            paired = [
                r["fidelity"]["endpoint_zero"][str(unit)]["combined_fidelity"] - r["fidelity"]["path_zero"][str(unit)]["combined_fidelity"]
                for r in results if r["fidelity"]["endpoint_zero"][str(unit)]["status"] == r["fidelity"]["path_zero"][str(unit)]["status"] == "VALID"
            ]
            metrics[f"fidelity_{unit}"] = {
                "valid_pairs": len(paired), "endpoint_median": median(endpoint_values) if endpoint_values else None,
                "path_median": median(path_values) if path_values else None,
                "endpoint_minus_path_median": median(paired) if paired else None,
            }
    protocol = read_json(root / "protocol.json")
    gate = protocol["go_to_50_gate"]
    checks = {
        "all_20_valid": len(results) == IMAGE_COUNT and not failures,
        "endpoint_identity": metrics.get("endpoint_identity_max_abs", float("inf")) <= gate["endpoint_identity_max_abs_lte"],
        "fine_fidelity_positive": (metrics.get("fidelity_4") or {}).get("endpoint_median", -float("inf")) > gate["endpoint_fine_median_fidelity_gt"],
        "fine_noninferiority_heuristic": (metrics.get("fidelity_4") or {}).get("endpoint_minus_path_median", -float("inf")) >= gate["endpoint_minus_path_fine_median_gte"],
        "reference_stability": metrics.get("endpoint_reference_pearson_median", -float("inf")) >= gate["endpoint_reference_pearson_median_gte"],
        "flip_stability": metrics.get("endpoint_flip_spearman_median", -float("inf")) >= gate["endpoint_flip_spearman_median_gte"],
        "runtime": metrics.get("endpoint_to_path_runtime_ratio_median", float("inf")) <= gate["endpoint_to_path_runtime_ratio_median_lte"],
    }
    decision = "GO_TO_PROSPECTIVE_50_IMAGE_VALIDATION" if all(checks.values()) else "NO_GO_CURRENT_ENDPOINT_PROTOCOL"
    write_json_new(root / "summary.json", {
        "status": "PASS_EXPLORATORY_ANALYSIS" if len(results) == IMAGE_COUNT else "FAIL_INCOMPLETE",
        "metrics": metrics, "gate_checks": checks, "decision": decision,
        "five_hundred_image_rerun_authorized": False, "failures": failures,
        "worker_exits": worker_exits,
    })
    files = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(root.rglob("*")) if path.is_file()]
    write_json_new(root / "whole_root_inventory.json", {"files": files, "files_digest": canonical(files)})
    print(json.dumps({"status": "FINALIZED", "decision": decision, "valid": len(results), "failures": len(failures)}), flush=True)


def launch() -> None:
    prepare()
    processes = []
    for physical_index in range(GPU_COUNT):
        worker_id = f"worker-{physical_index:02d}"
        log_path = ROOT / "workers" / f"{worker_id}.log"
        log_handle = log_path.open("xb")
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(physical_index),
            "PHYSICAL_GPU_INDEX": str(physical_index),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
        })
        command = [sys.executable, str(SCRIPT), "worker", "--root", str(ROOT), "--worker-id", worker_id]
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((worker_id, process, log_handle))
    exits = []
    for worker_id, process, log_handle in processes:
        code = process.wait()
        log_handle.close()
        exits.append({"worker_id": worker_id, "exit_code": code})
    finalize(ROOT, exits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "worker", "finalize"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--worker-id", default="")
    args = parser.parse_args()
    if args.mode == "launch":
        launch()
    elif args.mode == "worker":
        if not args.worker_id:
            parser.error("worker requires --worker-id")
        worker(args.root, args.worker_id)
    else:
        finalize(args.root, [])


if __name__ == "__main__":
    main()
