#!/usr/bin/env python3
"""Orchestrate the published ENDPOINT-E2 registered-target atlas."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO.parent / ".conda/envs/vitnns/bin/python"
EP0 = REPO / "manifests/nns_vit_endpoint/ep0_r1_20260918"
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_e2_atlas_run_spec.json"
RUNTIME = REPO.parent / ".runtime/nns_vit_endpoint_e2_atlas"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e2_atlas import (  # noqa: E402
    PASS_STATUS, canonical, digest, display_rgb_signed_sum, file_sha256,
    linear_percentile, validate_run_spec, write_json_new,
)
from vitnns.endpoint_ep1_queue import audit_coverage, create_queue  # noqa: E402

ALLOWED_UNTRACKED = {
    "config/nns_vit_cut_intervention/p0_c.json",
    "manifests/nns_vit_cut_intervention/p0_c_operation_registry.json",
    "scripts/validate_nns_vit_cut_intervention_p0_c.py",
    "src/vitnns/nns_vit_cut_intervention_p0_c.py",
    "src/vitnns/nns_vit_cut_intervention_p0_c_worker.py",
    "tests/test_nns_vit_cut_intervention_p0_c.py",
}
SOURCE_FILES = [
    "config/nns_vit_endpoint/endpoint_e2_atlas_run_spec.json",
    "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE2执行合同_20260918.md",
    "src/vitnns/endpoint_e2_atlas.py",
    "scripts/build_endpoint_e2_materialization.py",
    "scripts/run_endpoint_e2_atlas_worker.py",
    "scripts/run_endpoint_e2_atlas.py",
    "scripts/validate_endpoint_e2_atlas.py",
    "tests/test_endpoint_e2_atlas.py",
]
BINDINGS = {
    "target_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json",
    "cohort_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json",
    "model_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
    "evaluation_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/evaluation_registry.json",
    "endpoint_e1_terminal_sha256": "reports/nns_vit_endpoint_ep1_r11_20260918/terminal_receipt.json",
    "scope_amendment_terminal_sha256": "reports/nns_vit_endpoint_scope_amendment_r1_20260918/terminal_receipt.json",
    "unified_plan_sha256": "docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def remote(ref: str) -> str:
    fields = subprocess.check_output(["git", "ls-remote", "origin", ref], cwd=REPO, text=True).split()
    return fields[0] if len(fields) == 2 else ""


def startup() -> tuple[dict, str, list[str]]:
    spec = load(SPEC_PATH)
    validate_run_spec(spec)
    for key, relative in BINDINGS.items():
        if file_sha256(REPO / relative) != spec["predecessors"][key]:
            raise RuntimeError("PREDECESSOR_BINDING:" + key)
    validators = []
    for command in (
        [str(PYTHON), "scripts/validate_endpoint_unified_plan.py"],
        [str(PYTHON), "scripts/validate_endpoint_primary_track_amendment.py"],
        [str(PYTHON), "scripts/validate_endpoint_ep1.py"],
    ):
        completed = subprocess.run(
            command, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            text=True, capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError("PREDECESSOR_VALIDATION_FAILED:" + completed.stderr)
        validators.extend(completed.stdout.splitlines())
    head = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--short", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    remote_head = remote("refs/heads/" + branch)
    if branch != "stage-d-d3-pass" or not head == upstream == remote_head:
        raise RuntimeError("E2_PRODUCER_NOT_PUBLISHED")
    for relative in SOURCE_FILES:
        if subprocess.run(
            ["git", "cat-file", "-e", f"{head}:vitnns_rebuild/{relative}"], cwd=REPO,
        ).returncode:
            raise RuntimeError("SOURCE_NOT_COMMITTED:" + relative)
    status = git("status", "--porcelain", "--untracked-files=all")
    changed = []
    repo_prefix = REPO.relative_to(REPO.parent).as_posix() + "/"
    for line in status.splitlines():
        path = line[3:].split(" -> ")[-1]
        if path.startswith(repo_prefix):
            path = path[len(repo_prefix):]
        changed.append(path)
    if set(changed) - ALLOWED_UNTRACKED:
        raise RuntimeError("UNEXPLAINED_WORKTREE:" + str(sorted(set(changed) - ALLOWED_UNTRACKED)))
    return spec, head, validators


def gpu_snapshot() -> list[dict]:
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.free,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ], text=True)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits",
    ], text=True)
    active: dict[str, list[dict]] = {}
    for line in processes.splitlines():
        if line.strip():
            gpu_uuid, pid, memory = [item.strip() for item in line.split(",")]
            active.setdefault(gpu_uuid, []).append({"pid": int(pid), "used_memory_mib": int(memory)})
    rows = []
    for line in query.splitlines():
        index, gpu_uuid, name, pci, total, free, utilization, driver = [item.strip() for item in line.split(",")]
        rows.append({
            "physical_index": int(index), "gpu_uuid": gpu_uuid, "name": name,
            "pci_bus_id": pci, "memory_total_mib": int(total), "memory_free_mib": int(free),
            "utilization_percent": int(utilization), "driver_version": driver,
            "existing_compute_processes": active.get(gpu_uuid, []),
            "eligible": name == "NVIDIA H20" and int(free) >= 32768 and int(utilization) <= 10,
        })
    return rows


def lock_gpus(selected: list[dict], head: str, root: Path) -> tuple[list[dict], list[Any]]:
    lock_root = RUNTIME / "gpu_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    locked, streams = [], []
    for gpu in selected:
        stream = (lock_root / f"{gpu['gpu_uuid']}.lock").open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            continue
        stream.seek(0); stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "head": head, "root": str(root)}))
        stream.flush(); os.fsync(stream.fileno())
        locked.append(gpu); streams.append(stream)
    return locked, streams


def launch(
    queue: Path, results: Path, artifacts: Path, selected: list[dict], root: Path, mode: str,
) -> tuple[list[dict], float]:
    processes, streams = [], []
    started = time.monotonic()
    for index, gpu in enumerate(selected):
        worker_id = f"{mode}-worker-{index:02d}"
        log_path = root / "logs" / mode / f"{worker_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("xb")
        streams.append(stream)
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(gpu["physical_index"]),
            CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(REPO / "src"),
            PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
        )
        command = [
            str(PYTHON), "scripts/run_endpoint_e2_atlas_worker.py", "--queue", str(queue),
            "--results", str(results), "--artifacts", str(artifacts), "--mode", mode,
            "--worker-id", worker_id, "--physical-index", str(gpu["physical_index"]),
            "--gpu-uuid", gpu["gpu_uuid"],
        ]
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((worker_id, gpu, process))
    exits = []
    for (worker_id, gpu, process), stream in zip(processes, streams):
        code = process.wait()
        stream.flush(); os.fsync(stream.fileno()); stream.close()
        exits.append({"worker_id": worker_id, "gpu_uuid": gpu["gpu_uuid"], "return_code": code})
    return exits, time.monotonic() - started


def queue_results(queue: Path, results: Path) -> dict[str, dict]:
    output = {}
    with sqlite3.connect(queue) as connection:
        rows = connection.execute("SELECT task_id,result_path FROM tasks ORDER BY task_id").fetchall()
    for task_id, relative in rows:
        value = load(results / relative)
        output[task_id] = value["result"]
    return output


def render_galleries(root: Path, cache: Path, formal: dict[str, dict]) -> dict:
    rendered = []
    for model_id in ("vit_b16", "deit_s16"):
        panels = []
        scales = []
        for order in range(8):
            row = formal[f"{model_id}__{order:03d}"]
            image_path = cache / row["sample_id"]
            with Image.open(image_path) as source:
                clean = source.convert("RGB")
                width, height = clean.size
                if width <= height:
                    resized = (248, round(height * 248 / width))
                else:
                    resized = (round(width * 248 / height), 248)
                clean = clean.resize(resized, resample=Image.Resampling.BICUBIC)
                left = (clean.width - 224) // 2; top = (clean.height - 224) // 2
                clean = clean.crop((left, top, left + 224, top + 224))
            class_artifact = next(item for item in row["artifacts"] if item["kind"] == "raw_signed_rgb_class_maps")
            maps = np.load(REPO / class_artifact["path"], allow_pickle=False)
            gt, gt_scale, gt_zero = display_rgb_signed_sum(maps[0])
            top, top_scale, top_zero = display_rgb_signed_sum(maps[1])
            panels.append((clean, Image.fromarray(gt), Image.fromarray(top), row))
            scales.append({
                "split_order": order, "sample_id": row["sample_id"],
                "ground_truth_scale": gt_scale, "ground_truth_zero": gt_zero,
                "clean_top1_scale": top_scale, "clean_top1_zero": top_zero,
            })
        sheet = Image.new("RGB", (3 * 224, 8 * 252), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, (clean, gt, top, record) in enumerate(panels):
            y = row_index * 252
            sheet.paste(clean, (0, y + 28)); sheet.paste(gt, (224, y + 28)); sheet.paste(top, (448, y + 28))
            draw.text((4, y + 5), f"{row_index:03d} {record['sample_id']} | input | GT Endpoint | clean-top1 Endpoint", fill="black")
        path = root / "gallery" / f"{model_id}_first8.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            sheet.save(stream, format="PNG", optimize=False)
            stream.flush(); os.fsync(stream.fileno())
        rendered.append({
            "model_id": model_id, "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size, "sha256": file_sha256(path),
            "selection_orders": list(range(8)), "display_scales": scales,
            "display_only_not_statistics": True, "smoothing": False, "clipping": False,
        })
    return {"status": "PASS", "galleries": rendered}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-gpus", type=int, default=8)
    args = parser.parse_args()
    spec, head, predecessor_validation = startup()
    root = REPO / spec["outputs"]["formal_root"]
    cache = REPO / spec["outputs"]["evaluation_cache_root"]
    if root.exists() or cache.exists():
        raise RuntimeError("E2_ROOT_OR_CACHE_EXISTS")
    if shutil.disk_usage(REPO).free < 50 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_DISK")
    root.mkdir(parents=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    stage_started = time.monotonic()
    materialization = subprocess.run([
        str(PYTHON), "scripts/build_endpoint_e2_materialization.py",
        "--cache-root", str(cache), "--receipt", str(root / "materialization_receipt.json"),
    ], cwd=REPO, text=True, capture_output=True)
    if materialization.returncode:
        raise RuntimeError("MATERIALIZATION_FAILED:" + materialization.stderr)
    material = load(root / "materialization_receipt.json")
    models = {row["model_id"]: row for row in load(EP0 / "model_registry.json")["models"]}
    target_registry = load(EP0 / "target_registry.json")
    ep1 = load(REPO / "reports/nns_vit_endpoint_ep1_r11_20260918/validation_report.json")
    tasks = []
    for model_id in spec["models"]:
        model = models[model_id]
        target_rows = target_registry["targets_by_model"][model_id]
        for row in material["records"]:
            tasks.append({
                "task_id": f"{model_id}__{row['split_order']:03d}",
                "payload": {
                    "model_id": model_id, "timm_name": model["timm_name"],
                    "embedding_dim": model["embedding_dim"], "checkpoint": model["checkpoint"]["path"],
                    "checkpoint_bytes": model["checkpoint"]["bytes"],
                    "checkpoint_sha256": model["checkpoint"]["sha256"],
                    "tensor_digest": ep1["tensor_digests"][model_id],
                    "mean": model["preprocessing"]["mean"], "std": model["preprocessing"]["std"],
                    "sample_id": row["sample_id"], "split_order": row["split_order"],
                    "class_id": row["class_id"], "wnid": row["wnid"],
                    "cache_path": row["cache_path"], "image_bytes": row["bytes"],
                    "image_sha256": row["sha256"], "target_registry": spec["targets"]["registry"],
                    "target_registry_sha256": spec["targets"]["registry_sha256"],
                    "target_count": spec["targets"]["count_by_model"][model_id],
                    "target_rows_digest": digest(target_rows),
                    "retain_full_atlas": row["split_order"] in spec["retention"]["full_raw_atlas_evaluation_orders"],
                },
            })
    tasks.sort(key=lambda row: row["task_id"])
    if len(tasks) != 800:
        raise RuntimeError("TASK_COUNT")
    formal_queue = root / "formal_queue.sqlite3"
    replay_queue = root / "single_replay" / "queue.sqlite3"
    identities = {
        "formal": create_queue(formal_queue, tasks),
        "single_replay": create_queue(replay_queue, tasks),
    }
    write_json_new(root / "task_manifest.json", {
        "stage": "ENDPOINT-E2", "tasks": tasks, "task_count": 800,
        "evaluation_images": 400, "model_count": 2, "methods": ["canonical_endpoint_x_times_gradient"],
        "target_count_by_model": spec["targets"]["count_by_model"],
        "target_evaluations": 400 * (457 + 385), "queue_identities": identities,
        "manifest_digest": digest(tasks), "seed_rule": "SHA256(namespace|task_id) first8 bytes",
    })
    snapshot = gpu_snapshot()
    eligible = [row for row in snapshot if row["eligible"]]
    idle = [row for row in eligible if not row["existing_compute_processes"]]
    shared = [row for row in eligible if row["existing_compute_processes"]]
    requested = max(1, min(args.max_gpus, spec["runtime"]["max_gpus"]))
    selected, locks = lock_gpus((idle + shared)[:requested], head, root)
    if not selected:
        raise RuntimeError("BLOCKED_INFRASTRUCTURE_NO_LOCKABLE_H20")
    write_json_new(root / "startup_snapshot.json", {
        "hostname": socket.gethostname(), "producer_commit": head,
        "tracking_commit": git("rev-parse", "@{upstream}"),
        "remote_commit": remote("refs/heads/stage-d-d3-pass"),
        "predecessor_validation": predecessor_validation, "gpu_snapshot": snapshot,
        "selected_gpus": selected, "free_disk_bytes": shutil.disk_usage(REPO).free,
        "CUBLAS_WORKSPACE_CONFIG_before_workers": ":4096:8",
        "formal_image_reads": {"evaluation": 400, "calibration": 0, "discovery": 0},
        "existing_pids_recorded": True,
    })
    try:
        formal_exits, formal_seconds = launch(
            formal_queue, root / "formal_results", root / "raw", selected, root, "formal",
        )
        formal_coverage = audit_coverage(formal_queue, root / "formal_results")
        write_json_new(root / "formal_coverage_audit.json", formal_coverage)
        if any(row["return_code"] for row in formal_exits) or formal_coverage["status"] != "PASS":
            raise RuntimeError("FORMAL_WORKER_OR_COVERAGE_FAILED")
        elapsed_gpu_hours = formal_seconds * len(selected) / 3600
        if elapsed_gpu_hours + 1800 / 3600 > 64 or time.monotonic() - stage_started + 1800 > 72 * 3600:
            raise RuntimeError("PAUSED_RESOURCE_CAP_BEFORE_REPLAY")
        replay_exits, replay_seconds = launch(
            replay_queue, root / "single_replay/results", root / "single_replay/no_artifacts",
            selected[:1], root, "replay",
        )
    finally:
        for stream in locks:
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()
    replay_coverage = audit_coverage(replay_queue, root / "single_replay/results")
    write_json_new(root / "single_replay/coverage_audit.json", replay_coverage)
    if any(row["return_code"] for row in replay_exits) or replay_coverage["status"] != "PASS":
        raise RuntimeError("REPLAY_WORKER_OR_COVERAGE_FAILED")
    formal = queue_results(formal_queue, root / "formal_results")
    replay = queue_results(replay_queue, root / "single_replay/results")
    if set(formal) != set(replay):
        raise RuntimeError("SINGLE_MULTI_TASK_SET")
    equivalence = {task_id: formal[task_id]["numeric_digest"] == replay[task_id]["numeric_digest"] for task_id in formal}
    if not all(equivalence.values()):
        raise RuntimeError("SINGLE_MULTI_NUMERIC_MISMATCH")
    gallery = render_galleries(root, cache, formal)
    write_json_new(root / "gallery_manifest.json", gallery)
    near_zero = []
    failures = []
    for task_id, result in sorted(formal.items()):
        for target in result["summaries"]:
            if target["near_zero"]:
                near_zero.append({
                    "task_id": task_id, "model_id": result["model_id"],
                    "sample_id": result["sample_id"], "split_order": result["split_order"],
                    "target_id": target["target_id"], "max_abs": target["max_abs"],
                })
    write_json_new(root / "near_zero_index.json", {
        "threshold_max_abs": 1e-12, "count": len(near_zero), "rows": near_zero,
    })
    write_json_new(root / "failure_index.json", {"count": len(failures), "rows": failures})
    latencies = [row["scan_seconds"] for row in formal.values()]
    task_latencies = [row["task_seconds"] for row in formal.values()]
    replay_latencies = [row["scan_seconds"] for row in replay.values()]
    replay_task_latencies = [row["task_seconds"] for row in replay.values()]
    total_targets = sum(row["target_count"] for row in formal.values())
    raw_bytes = sum(item["bytes"] for row in formal.values() for item in row["artifacts"])
    model_costs = {}
    for model_id in spec["models"]:
        values = [row["scan_seconds"] for row in formal.values() if row["model_id"] == model_id]
        model_costs[model_id] = {
            "images": len(values), "registered_targets": spec["targets"]["count_by_model"][model_id],
            "median_seconds_per_complete_atlas": linear_percentile(values, 0.5),
            "p95_seconds_per_complete_atlas": linear_percentile(values, 0.95),
            "max_seconds_per_complete_atlas": max(values),
        }
    gpu_hours = formal_seconds * len(selected) / 3600 + replay_seconds / 3600
    retained_before_report = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    report = {
        "status": PASS_STATUS, "stage": "ENDPOINT-E2",
        "claim_decision": "C2_REGISTERED_SCAN_COMPLETION_AND_RESOURCES",
        "scientific_utility_claims": "NOT_RUN_NOT_ALLOWED",
        "path_ig": "NOT_RUN_NOT_ALLOWED", "input_fidelity_stability": "NOT_RUN_NOT_ALLOWED",
        "finite_intervention": "NOT_RUN_NOT_ALLOWED", "producer_commit": head,
        "materialization_records_digest": material["records_digest"],
        "task_manifest_digest": identities["formal"]["manifest_digest"],
        "formal_coverage": formal_coverage, "single_replay_coverage": replay_coverage,
        "single_multi_same_task_manifest": identities["formal"] == identities["single_replay"],
        "single_multi_exact_numeric_digest_count": sum(equivalence.values()),
        "single_multi_mismatch_task_ids": [key for key, value in equivalence.items() if not value],
        "evaluation_images": 400, "models": spec["models"], "image_model_tasks": len(formal),
        "target_count_by_model": spec["targets"]["count_by_model"],
        "target_evaluations": total_targets,
        "finite_target_evaluations": sum(row["finite_target_count"] for row in formal.values()),
        "near_zero_target_evaluations": len(near_zero),
        "raw_class_map_task_count": sum(any(item["kind"] == "raw_signed_rgb_class_maps" for item in row["artifacts"]) for row in formal.values()),
        "full_raw_atlas_task_count": sum(any(item["kind"] == "full_raw_atlas" for item in row["artifacts"]) for row in formal.values()),
        "gallery": gallery, "latency_seconds": {
            "exclusive_scan_p50": linear_percentile(latencies, 0.5),
            "exclusive_scan_p95": linear_percentile(latencies, 0.95),
            "exclusive_scan_max": max(latencies),
            "end_to_end_task_p50": linear_percentile(task_latencies, 0.5),
            "end_to_end_task_p95": linear_percentile(task_latencies, 0.95),
            "end_to_end_task_max": max(task_latencies),
            "single_replay_exclusive_scan_p50": linear_percentile(replay_latencies, 0.5),
            "single_replay_exclusive_scan_p95": linear_percentile(replay_latencies, 0.95),
            "single_replay_exclusive_scan_max": max(replay_latencies),
            "single_replay_end_to_end_task_p50": linear_percentile(replay_task_latencies, 0.5),
            "single_replay_end_to_end_task_p95": linear_percentile(replay_task_latencies, 0.95),
            "single_replay_end_to_end_task_max": max(replay_task_latencies),
        },
        "throughput": {
            "formal_targets_per_second": total_targets / formal_seconds,
            "formal_images_per_second": len(formal) / formal_seconds,
            "exclusive_scan_targets_per_gpu_second": total_targets / sum(latencies),
            "exclusive_scan_images_per_gpu_second": len(formal) / sum(latencies),
            "single_replay_targets_per_second": total_targets / replay_seconds,
            "single_replay_images_per_second": len(replay) / replay_seconds,
        },
        "complete_atlas_amortized_cost": model_costs,
        "peak_allocated_gpu_bytes_max": max(row["peak_allocated_gpu_bytes"] for row in formal.values()),
        "peak_reserved_gpu_bytes_max": max(row["peak_reserved_gpu_bytes"] for row in formal.values()),
        "peak_cpu_rss_bytes_max": max(row["peak_cpu_rss_bytes"] for row in formal.values()),
        "raw_retained_bytes": raw_bytes, "stored_raw_bytes_per_target_evaluation": raw_bytes / total_targets,
        "retained_bytes_before_report": retained_before_report, "formal_wall_seconds": formal_seconds,
        "single_replay_wall_seconds": replay_seconds, "total_wall_seconds": time.monotonic() - stage_started,
        "gpu_hours_upper_observed": gpu_hours, "formal_workers": len(selected), "replay_workers": 1,
        "worker_exits": {"formal": formal_exits, "single_replay": replay_exits},
        "model_tensor_digests": {
            model_id: next(row for row in formal.values() if row["model_id"] == model_id)["model_binding"]["tensor_digest"]
            for model_id in spec["models"]
        },
        "implementation_bindings": [{"path": path, "sha256": file_sha256(REPO / path)} for path in SOURCE_FILES],
        "resource_caps": {"gpu_hours": 64, "wall_hours": 72, "incremental_retained_gib": 128},
        "next_authorization": "ENDPOINT_E3_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    write_json_new(root / "validation_report.json", report)
    files = [{
        "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path),
    } for path in sorted(root.rglob("*")) if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}]
    manifest = {"files": files, "canonical_digest": digest(files)}
    write_json_new(root / "artifact_manifest.json", manifest)
    terminal = {
        "status": PASS_STATUS, "stage": "ENDPOINT-E2",
        "claim_decision": "C2_REGISTERED_SCAN_COMPLETION_AND_RESOURCES",
        "producer_commit": head, "evaluation_image_count": 400, "image_model_task_count": 800,
        "target_evaluation_count": total_targets, "validation_digest": digest(report),
        "manifest_digest": manifest["canonical_digest"],
        "next_authorization": "ENDPOINT_E3_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    write_json_new(root / "terminal_receipt.json", terminal)
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
