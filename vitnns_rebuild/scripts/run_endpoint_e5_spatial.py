#!/usr/bin/env python3
"""Orchestrate the published ENDPOINT-E5 spatial-intervention cohort."""
from __future__ import annotations

import argparse
import fcntl
import json
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

REPO = Path(__file__).resolve().parents[1]
GIT_ROOT = REPO.parent
PYTHON = GIT_ROOT / ".conda/envs/vitnns/bin/python"
EP0 = REPO / "manifests/nns_vit_endpoint/ep0_r1_20260918"
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_e5_spatial_run_spec.json"
ROOT_NAME = "reports/nns_vit_endpoint_e5_spatial_r1_20260918"
RUNTIME = GIT_ROOT / ".runtime/nns_vit_endpoint_e5_spatial_r1"
PASS_STATUS = "ENDPOINT_E5_PASS_SPATIAL_INTERVENTION_COMPLETE_NO_CLAIM_DECISION"
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from run_endpoint_e4_intervention import ALLOWED_UNTRACKED, gpu_snapshot, lock_gpus, remote  # noqa: E402
from vitnns.endpoint_e2_atlas import digest, file_sha256, linear_percentile, write_json_new  # noqa: E402
from vitnns.endpoint_e5_spatial import BASELINES, validate_run_spec  # noqa: E402
from vitnns.endpoint_ep1_queue import audit_coverage, create_queue  # noqa: E402

SOURCE_FILES = [
    "config/nns_vit_endpoint/endpoint_e5_spatial_run_spec.json",
    "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE5执行合同_20260918.md",
    "src/vitnns/endpoint_e5_spatial.py", "scripts/run_endpoint_e5_spatial_worker.py",
    "scripts/run_endpoint_e5_spatial.py", "scripts/validate_endpoint_e5_spatial.py",
    "scripts/validate_endpoint_unified_plan.py", "tests/test_endpoint_e5_spatial.py",
]
BINDINGS = {
    "cohort_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json",
    "endpoint_e4_r3_terminal_sha256": "reports/nns_vit_endpoint_e4_intervention_r3_20260918/terminal_receipt.json",
    "endpoint_e4_r3_validation_sha256": "reports/nns_vit_endpoint_e4_intervention_r3_20260918/validation_report.json",
    "intervention_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/intervention_registry.json",
    "materialization_receipt_sha256": "reports/nns_vit_endpoint_e2_atlas_r1_20260918/materialization_receipt.json",
    "model_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
    "statistical_analysis_plan_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json",
    "target_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json",
    "unified_plan_sha256": "docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md",
}
E5_FILES = set(SOURCE_FILES)


def load(path: Path):
    return json.loads(path.read_text())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def startup() -> tuple[dict, str, list[str]]:
    spec = load(SPEC_PATH); validate_run_spec(spec)
    for key, relative in BINDINGS.items():
        if file_sha256(REPO / relative) != spec["predecessors"][key]:
            raise RuntimeError("PREDECESSOR_BINDING:" + key)
    e4 = load(REPO / BINDINGS["endpoint_e4_r3_terminal_sha256"])
    if e4["status"] != "ENDPOINT_E4_R3_PASS_EXECUTION_CONDITION_SCALED_JVP_AND_FINITE_REPLACEMENT" or e4["next_authorization"] != "ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST":
        raise RuntimeError("E4_R3_AUTHORITY")
    validators = []
    for command in (
        [str(PYTHON), "scripts/validate_endpoint_unified_plan.py"],
        [str(PYTHON), "scripts/validate_endpoint_e4_recovery_r3.py"],
    ):
        completed = subprocess.run(command, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError("PREDECESSOR_VALIDATION_FAILED:" + completed.stderr)
        validators.extend(completed.stdout.splitlines())
    head = git("rev-parse", "HEAD"); branch = git("symbolic-ref", "--short", "HEAD")
    if branch != "stage-d-d3-pass" or not head == git("rev-parse", "@{upstream}") == remote("refs/heads/" + branch):
        raise RuntimeError("E5_PRODUCER_NOT_PUBLISHED")
    for relative in SOURCE_FILES:
        if subprocess.run(["git", "cat-file", "-e", f"{head}:vitnns_rebuild/{relative}"], cwd=REPO).returncode:
            raise RuntimeError("SOURCE_NOT_COMMITTED:" + relative)
    changed = []
    prefix = REPO.relative_to(GIT_ROOT).as_posix() + "/"
    for line in git("status", "--porcelain", "--untracked-files=all").splitlines():
        path = line[3:].split(" -> ")[-1]
        if path.startswith(prefix): path = path[len(prefix):]
        changed.append(path)
    if set(changed) - ALLOWED_UNTRACKED:
        raise RuntimeError("UNEXPLAINED_WORKTREE:" + str(sorted(set(changed) - ALLOWED_UNTRACKED)))
    return spec, head, validators


def launch(queue: Path, results: Path, selected: list[dict], root: Path) -> tuple[list[dict], float]:
    processes, streams = [], []; started = time.monotonic()
    for index, gpu in enumerate(selected):
        worker_id = f"formal-worker-{index:02d}"; log_path = root / "logs" / f"{worker_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True); stream = log_path.open("xb"); streams.append(stream)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu["physical_index"]),
            CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        command = [str(PYTHON), "scripts/run_endpoint_e5_spatial_worker.py", "--queue", str(queue),
            "--results", str(results), "--worker-id", worker_id, "--physical-index", str(gpu["physical_index"]),
            "--gpu-uuid", gpu["gpu_uuid"]]
        processes.append((worker_id, gpu, subprocess.Popen(command, cwd=REPO, env=environment,
                                                            stdout=stream, stderr=subprocess.STDOUT)))
    exits = []
    for (worker_id, gpu, process), stream in zip(processes, streams):
        code = process.wait(); stream.flush(); os.fsync(stream.fileno()); stream.close()
        exits.append({"worker_id": worker_id, "gpu_uuid": gpu["gpu_uuid"], "return_code": code})
    return exits, time.monotonic() - started


def queue_results(queue: Path, results: Path) -> dict[str, dict]:
    output = {}
    with sqlite3.connect(queue) as connection:
        rows = connection.execute("SELECT task_id,result_path FROM tasks ORDER BY task_id").fetchall()
    for task_id, relative in rows:
        output[task_id] = load(results / relative)["result"]
    return output


def descriptive(formal: dict[str, dict]) -> list[dict]:
    import numpy as np
    rows = []
    for model_id in ("vit_b16", "deit_s16"):
        model_rows = [row for row in formal.values() if row["model_id"] == model_id]
        for reference in ("normalized_zero", "blur_rgb"):
            values = {method: [row["references"][reference]["image_method_scores"][method] for row in model_rows]
                      for method in ("endpoint_component", *BASELINES)}
            paired = {baseline: [row["references"][reference]["paired_endpoint_minus_baseline"][baseline]
                                 for row in model_rows] for baseline in BASELINES}
            rows.append({"model_id": model_id, "reference_id": reference, "n": len(model_rows),
                         "median_image_method_scores": {key: float(np.median(value)) for key, value in values.items()},
                         "median_paired_endpoint_minus_baseline": {key: float(np.median(value)) for key, value in paired.items()},
                         "inference_performed": False})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--max-gpus", type=int, default=8); args = parser.parse_args()
    spec, head, predecessor_validation = startup(); root = REPO / spec["outputs"]["formal_root"]
    if root.exists(): raise RuntimeError("E5_ROOT_EXISTS")
    if shutil.disk_usage(REPO).free < 50 * 1024 ** 3: raise RuntimeError("PAUSED_RESOURCE_CAP_DISK")
    root.mkdir(parents=True); RUNTIME.mkdir(parents=True, exist_ok=True); stage_started = time.monotonic()
    material = load(REPO / spec["outputs"]["source_materialization"])
    if material["status"] != "PASS_EVALUATION_MATERIALIZATION" or material["authorized_member_count"] != 400 or material["records_digest"] != digest(material["records"]):
        raise RuntimeError("MATERIALIZATION_AUTHORITY")
    models = {row["model_id"]: row for row in load(EP0 / "model_registry.json")["models"]}
    ep1 = load(REPO / "reports/nns_vit_endpoint_ep1_r11_20260918/validation_report.json")
    tasks = []
    for model_id in spec["models"]:
        model = models[model_id]
        for row in material["records"]:
            tasks.append({"task_id": f"{model_id}__{row['split_order']:03d}", "payload": {
                "model_id": model_id, "timm_name": model["timm_name"], "embedding_dim": model["embedding_dim"],
                "checkpoint": model["checkpoint"]["path"], "checkpoint_bytes": model["checkpoint"]["bytes"],
                "checkpoint_sha256": model["checkpoint"]["sha256"], "tensor_digest": ep1["tensor_digests"][model_id],
                "mean": model["preprocessing"]["mean"], "std": model["preprocessing"]["std"],
                "sample_id": row["sample_id"], "split_order": row["split_order"], "class_id": row["class_id"],
                "wnid": row["wnid"], "cache_path": row["cache_path"], "image_bytes": row["bytes"],
                "image_sha256": row["sha256"], "retain_raw_maps": row["split_order"] < 16,
                "perturbed_batch_size": 16, "isolation_batch_size": 8, "stage": "ENDPOINT-E5"}})
    tasks.sort(key=lambda row: row["task_id"])
    if len(tasks) != 800: raise RuntimeError("TASK_COUNT")
    queue = root / "formal_queue.sqlite3"; queue_identity = create_queue(queue, tasks)
    write_json_new(root / "task_manifest.json", {"stage": "ENDPOINT-E5", "tasks": tasks, "task_count": 800,
        "evaluation_images": 400, "model_count": 2, "raw_map_task_count": 32, "references": ["normalized_zero", "blur_rgb"],
        "queue_identity": queue_identity, "manifest_digest": digest(tasks),
        "seed_rule": "SHA256(NNS_VIT_ENDPOINT_QUEUE_V1|task_id) first8 bytes big endian"})
    snapshot = gpu_snapshot(); eligible = [row for row in snapshot if row["eligible"]]
    candidates = [row for row in eligible if not row["existing_compute_processes"]] + [row for row in eligible if row["existing_compute_processes"]]
    selected, locks = lock_gpus(candidates, head, root, max(1, min(args.max_gpus, spec["runtime"]["max_gpus"])))
    if not selected: raise RuntimeError("BLOCKED_INFRASTRUCTURE_NO_LOCKABLE_H20")
    write_json_new(root / "startup_snapshot.json", {"hostname": socket.gethostname(), "producer_commit": head,
        "tracking_commit": git("rev-parse", "@{upstream}"), "remote_commit": remote("refs/heads/" + git("symbolic-ref", "--short", "HEAD")),
        "predecessor_validation": predecessor_validation, "gpu_snapshot": snapshot, "selected_gpus": selected,
        "free_disk_bytes": shutil.disk_usage(REPO).free, "CUBLAS_WORKSPACE_CONFIG_before_workers": ":4096:8",
        "formal_image_reads": {"evaluation": 400, "calibration": 0, "discovery": 0},
        "existing_pids_recorded": True, "existing_processes_not_terminated": True})
    try:
        exits, formal_seconds = launch(queue, root / "formal_results", selected, root)
    finally:
        for stream in locks: fcntl.flock(stream, fcntl.LOCK_UN); stream.close()
    coverage = audit_coverage(queue, root / "formal_results"); write_json_new(root / "formal_coverage_audit.json", coverage)
    if any(row["return_code"] for row in exits) or coverage["status"] != "PASS":
        raise RuntimeError("FORMAL_WORKER_OR_COVERAGE_FAILED")
    formal = queue_results(queue, root / "formal_results")
    retained = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    gpu_hours = formal_seconds * len(selected) / 3600
    e4 = load(REPO / "reports/nns_vit_endpoint_e4_intervention_r3_20260918/validation_report.json")
    combined_gpu = e4["combined_gpu_hours_observed"] + gpu_hours
    combined_wall = e4["combined_wall_seconds_observed"] + formal_seconds
    report = {"status": PASS_STATUS, "stage": "ENDPOINT-E5",
        "claim_scope": "EXECUTION_PASS_PER_IMAGE_C4B_INPUTS_NO_BOOTSTRAP_HOLM_OR_CLAIM_DECISION",
        "producer_commit": head, "evaluation_images": 400, "models": spec["models"], "image_model_tasks": 800,
        "reference_count": 2, "curves": sum(row["counts"]["curve_count"] for row in formal.values()),
        "random_curves": sum(row["counts"]["random_curve_count"] for row in formal.values()),
        "candidate_taylor_forward_evaluations": sum(row["counts"]["candidate_taylor_forward_evaluations"] for row in formal.values()),
        "finite_union_full_forward_curve_points": sum(row["counts"]["finite_union_full_forward_curve_points"] for row in formal.values()),
        "unique_multi_region_full_forward_inputs": sum(reference["unique_multi_region_full_forward_input_count"] for row in formal.values() for reference in row["references"].values()),
        "finite_union_isolated_head_evaluations": sum(row["counts"]["finite_union_isolated_head_evaluations"] for row in formal.values()),
        "raw_map_task_count": sum(row["raw_maps_record"] is not None for row in formal.values()),
        "descriptive_per_image_score_summary": descriptive(formal), "formal_coverage": coverage,
        "task_manifest_digest": queue_identity["manifest_digest"], "materialization_records_digest": material["records_digest"],
        "latency_seconds": {"exclusive_scan_p50": linear_percentile([row["scan_seconds"] for row in formal.values()], .5),
            "exclusive_scan_p95": linear_percentile([row["scan_seconds"] for row in formal.values()], .95),
            "exclusive_scan_max": max(row["scan_seconds"] for row in formal.values()),
            "end_to_end_task_p50": linear_percentile([row["task_seconds"] for row in formal.values()], .5),
            "end_to_end_task_p95": linear_percentile([row["task_seconds"] for row in formal.values()], .95),
            "end_to_end_task_max": max(row["task_seconds"] for row in formal.values())},
        "peak_allocated_gpu_bytes_max": max(row["peak_allocated_gpu_bytes"] for row in formal.values()),
        "peak_reserved_gpu_bytes_max": max(row["peak_reserved_gpu_bytes"] for row in formal.values()),
        "peak_cpu_rss_bytes_max": max(row["peak_cpu_rss_bytes"] for row in formal.values()),
        "retained_bytes_before_report": retained, "formal_wall_seconds": formal_seconds,
        "total_wall_seconds": time.monotonic() - stage_started, "gpu_hours_upper_observed": gpu_hours,
        "combined_e4_e5_gpu_hours_observed": combined_gpu, "combined_e4_e5_wall_seconds_observed": combined_wall,
        "formal_workers": len(selected), "worker_exits": exits,
        "implementation_bindings": [{"path": path, "sha256": file_sha256(REPO / path), "commit": head} for path in SOURCE_FILES],
        "deferred": {"bootstrap": "ENDPOINT-E6", "holm_28_tests": "ENDPOINT-E6",
                     "c4b_claim_decisions": "ENDPOINT-E6", "path_ig_input_fidelity_stability": "NOT_RUN_NOT_ALLOWED"},
        "resource_caps": {"combined_e4_e5_gpu_hours": 136, "combined_e4_e5_wall_hours": 132,
                          "combined_e4_e5_incremental_retained_gib": 56},
        "next_authorization": "ENDPOINT_E6_REQUIRES_EXPLICIT_USER_REQUEST"}
    if combined_gpu > 136 or combined_wall > 132 * 3600 or retained + e4["retained_bytes_before_report"] > 56 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP")
    write_json_new(root / "validation_report.json", report)
    files = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
             for path in sorted(root.rglob("*")) if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}]
    manifest = {"files": files, "canonical_digest": digest(files)}; write_json_new(root / "artifact_manifest.json", manifest)
    terminal = {"status": PASS_STATUS, "stage": "ENDPOINT-E5", "claim_scope": report["claim_scope"],
        "producer_commit": head, "evaluation_image_count": 400, "image_model_task_count": 800,
        "curve_count": report["curves"], "random_curve_count": report["random_curves"],
        "validation_digest": digest(report), "manifest_digest": manifest["canonical_digest"],
        "next_authorization": "ENDPOINT_E6_REQUIRES_EXPLICIT_USER_REQUEST"}
    write_json_new(root / "terminal_receipt.json", terminal); print(json.dumps(terminal, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
