#!/usr/bin/env python3
"""Orchestrate the published ENDPOINT-E4 activation-replacement cohort."""
from __future__ import annotations

import argparse
import fcntl
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

REPO = Path(__file__).resolve().parents[1]
GIT_ROOT = REPO.parent
PYTHON = GIT_ROOT / ".conda/envs/vitnns/bin/python"
EP0 = REPO / "manifests/nns_vit_endpoint/ep0_r1_20260918"
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_e4_intervention_run_spec.json"
RUNTIME = GIT_ROOT / ".runtime/nns_vit_endpoint_e4_intervention"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e2_atlas import digest, file_sha256, linear_percentile, write_json_new  # noqa: E402
from vitnns.endpoint_e4_intervention import MAIN_FAMILIES, validate_run_spec  # noqa: E402
from vitnns.endpoint_ep1_queue import audit_coverage, create_queue  # noqa: E402

PASS_STATUS = "ENDPOINT_E4_PASS_EXECUTION_LOCAL_CONTRACT_AND_FINITE_REPLACEMENT"
ALLOWED_UNTRACKED = {
    "config/nns_vit_cut_intervention/p0_c.json",
    "manifests/nns_vit_cut_intervention/p0_c_operation_registry.json",
    "scripts/validate_nns_vit_cut_intervention_p0_c.py",
    "src/vitnns/nns_vit_cut_intervention_p0_c.py",
    "src/vitnns/nns_vit_cut_intervention_p0_c_worker.py",
    "tests/test_nns_vit_cut_intervention_p0_c.py",
}
SOURCE_FILES = [
    "config/nns_vit_endpoint/endpoint_e4_intervention_run_spec.json",
    "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE4执行合同_20260918.md",
    "src/vitnns/endpoint_e4_intervention.py",
    "scripts/run_endpoint_e4_intervention_worker.py",
    "scripts/run_endpoint_e4_intervention.py",
    "scripts/validate_endpoint_e4_intervention.py",
    "tests/test_endpoint_e4_intervention.py",
]
BINDINGS = {
    "cohort_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json",
    "endpoint_e3_r1_artifact_manifest_sha256": "reports/nns_vit_endpoint_e3_closure_r1_20260918/artifact_manifest.json",
    "endpoint_e3_r2_terminal_sha256": "reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json",
    "endpoint_e3_r2_validation_sha256": "reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json",
    "intervention_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/intervention_registry.json",
    "materialization_receipt_sha256": "reports/nns_vit_endpoint_e2_atlas_r1_20260918/materialization_receipt.json",
    "model_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
    "statistical_analysis_plan_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json",
    "target_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json",
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
    spec = load(SPEC_PATH); validate_run_spec(spec)
    for key, relative in BINDINGS.items():
        if file_sha256(REPO / relative) != spec["predecessors"][key]:
            raise RuntimeError("PREDECESSOR_BINDING:" + key)
    e3 = load(REPO / "reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json")
    if e3["status"] != "ENDPOINT_E3_PASS_CORRECTNESS_ONLY_NEXT_ENDPOINT_E4_REQUIRES_EXPLICIT_REQUEST" or e3["next_authorization"] != "ENDPOINT_E4_REQUIRES_EXPLICIT_USER_REQUEST":
        raise RuntimeError("E3_AUTHORITY")
    validators = []
    for command in (
        [str(PYTHON), "scripts/validate_endpoint_unified_plan.py"],
        [str(PYTHON), "scripts/validate_endpoint_primary_track_amendment.py"],
        [str(PYTHON), "scripts/validate_endpoint_ep1.py"],
        [str(PYTHON), "scripts/validate_endpoint_e2_atlas.py"],
        [str(PYTHON), "scripts/validate_endpoint_e3_closure.py",
         "reports/nns_vit_endpoint_e3_closure_r1_20260918"],
    ):
        completed = subprocess.run(
            command, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            text=True, capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError("PREDECESSOR_VALIDATION_FAILED:" + completed.stderr)
        validators.extend(completed.stdout.splitlines())
    head = git("rev-parse", "HEAD"); branch = git("symbolic-ref", "--short", "HEAD")
    upstream = git("rev-parse", "@{upstream}"); remote_head = remote("refs/heads/" + branch)
    if branch != "stage-d-d3-pass" or not head == upstream == remote_head:
        raise RuntimeError("E4_PRODUCER_NOT_PUBLISHED")
    for relative in SOURCE_FILES:
        if subprocess.run(["git", "cat-file", "-e", f"{head}:vitnns_rebuild/{relative}"], cwd=REPO).returncode:
            raise RuntimeError("SOURCE_NOT_COMMITTED:" + relative)
    status = git("status", "--porcelain", "--untracked-files=all")
    changed = []
    prefix = REPO.relative_to(GIT_ROOT).as_posix() + "/"
    for line in status.splitlines():
        path = line[3:].split(" -> ")[-1]
        if path.startswith(prefix):
            path = path[len(prefix):]
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
            "physical_index": int(index), "gpu_uuid": gpu_uuid, "name": name, "pci_bus_id": pci,
            "memory_total_mib": int(total), "memory_free_mib": int(free),
            "utilization_percent": int(utilization), "driver_version": driver,
            "existing_compute_processes": active.get(gpu_uuid, []),
            "eligible": name == "NVIDIA H20" and int(free) >= 32768 and int(utilization) <= 10,
        })
    return rows


def lock_gpus(candidates: list[dict], head: str, root: Path, maximum: int) -> tuple[list[dict], list[Any]]:
    lock_root = RUNTIME / "gpu_locks"; lock_root.mkdir(parents=True, exist_ok=True)
    locked, streams = [], []
    for gpu in candidates:
        if len(locked) >= maximum:
            break
        stream = (lock_root / f"{gpu['gpu_uuid']}.lock").open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close(); continue
        stream.seek(0); stream.truncate(); stream.write(json.dumps({"pid": os.getpid(), "head": head, "root": str(root)}))
        stream.flush(); os.fsync(stream.fileno()); locked.append(gpu); streams.append(stream)
    return locked, streams


def launch(queue: Path, results: Path, selected: list[dict], root: Path) -> tuple[list[dict], float]:
    processes, streams = [], []; started = time.monotonic()
    for index, gpu in enumerate(selected):
        worker_id = f"formal-worker-{index:02d}"; log_path = root / "logs" / f"{worker_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True); stream = log_path.open("xb"); streams.append(stream)
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(gpu["physical_index"]), CUBLAS_WORKSPACE_CONFIG=":4096:8",
            PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
        )
        command = [
            str(PYTHON), "scripts/run_endpoint_e4_intervention_worker.py", "--queue", str(queue),
            "--results", str(results), "--worker-id", worker_id,
            "--physical-index", str(gpu["physical_index"]), "--gpu-uuid", gpu["gpu_uuid"],
        ]
        processes.append((worker_id, gpu, subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)))
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


def aggregate_descriptive(formal: dict[str, dict]) -> list[dict]:
    output = []
    for model_id in ("vit_b16", "deit_s16"):
        model_rows = [row for row in formal.values() if row["model_id"] == model_id]
        for reference in ("normalized_zero", "blur_rgb"):
            for family in MAIN_FAMILIES:
                metrics = [row["finite_utility"][reference]["per_image_metrics"][family] for row in model_rows]
                item = {"model_id": model_id, "reference_id": reference, "family": family, "n_total": len(metrics)}
                for source, target in (
                    ("m_vs_absolute_effect", "endpoint_mass"),
                    ("a_vs_effect", "endpoint_signed"),
                    ("p_ref_vs_effect", "p_ref_signed"),
                    ("abs_p_ref_vs_absolute_effect", "p_ref_absolute"),
                    ("output_norm_vs_absolute_effect", "output_norm"),
                    ("gradient_norm_vs_absolute_effect", "gradient_norm"),
                    ("abs_gradient_activation_vs_absolute_effect", "gradient_activation"),
                ):
                    values = [metric[source]["value"] for metric in metrics if metric[source]["value"] is not None]
                    item[target] = {
                        "n_valid": len(values), "n_na": len(metrics) - len(values),
                        "median_spearman": None if not values else float(__import__("numpy").median(values)),
                        "minimum": None if not values else min(values), "maximum": None if not values else max(values),
                    }
                random_values = [metric["random_20"]["mean_spearman"] for metric in metrics if metric["random_20"]["mean_spearman"] is not None]
                item["random_20"] = {"n_valid": len(random_values), "median_spearman": None if not random_values else float(__import__("numpy").median(random_values))}
                output.append(item)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--max-gpus", type=int, default=8); args = parser.parse_args()
    spec, head, predecessor_validation = startup(); root = REPO / spec["outputs"]["formal_root"]
    if root.exists():
        raise RuntimeError("E4_ROOT_EXISTS")
    if shutil.disk_usage(REPO).free < 50 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP_DISK")
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
                "image_sha256": row["sha256"], "include_local": row["split_order"] < 16,
                "include_secondary": row["split_order"] < 64, "intervention_batch_size": 8,
            }})
    tasks.sort(key=lambda row: row["task_id"])
    if len(tasks) != 800:
        raise RuntimeError("TASK_COUNT")
    queue = root / "formal_queue.sqlite3"; queue_identity = create_queue(queue, tasks)
    write_json_new(root / "task_manifest.json", {
        "stage": "ENDPOINT-E4", "tasks": tasks, "task_count": 800, "evaluation_images": 400,
        "model_count": 2, "local_task_count": 32, "secondary_task_count": 128,
        "references": ["normalized_zero", "blur_rgb"], "queue_identity": queue_identity,
        "manifest_digest": digest(tasks), "seed_rule": "SHA256(namespace|task_id) first8 bytes",
    })
    snapshot = gpu_snapshot(); eligible = [row for row in snapshot if row["eligible"]]
    candidates = [row for row in eligible if not row["existing_compute_processes"]] + [row for row in eligible if row["existing_compute_processes"]]
    requested = max(1, min(args.max_gpus, spec["runtime"]["max_gpus"]))
    selected, locks = lock_gpus(candidates, head, root, requested)
    if not selected:
        raise RuntimeError("BLOCKED_INFRASTRUCTURE_NO_LOCKABLE_H20")
    write_json_new(root / "startup_snapshot.json", {
        "hostname": socket.gethostname(), "producer_commit": head, "tracking_commit": git("rev-parse", "@{upstream}"),
        "remote_commit": remote("refs/heads/" + git("symbolic-ref", "--short", "HEAD")),
        "predecessor_validation": predecessor_validation, "gpu_snapshot": snapshot, "selected_gpus": selected,
        "free_disk_bytes": shutil.disk_usage(REPO).free, "CUBLAS_WORKSPACE_CONFIG_before_workers": ":4096:8",
        "formal_image_reads": {"evaluation": 400, "calibration": 0, "discovery": 0},
        "existing_pids_recorded": True, "existing_processes_not_terminated": True,
    })
    try:
        exits, formal_seconds = launch(queue, root / "formal_results", selected, root)
    finally:
        for stream in locks:
            fcntl.flock(stream, fcntl.LOCK_UN); stream.close()
    coverage = audit_coverage(queue, root / "formal_results"); write_json_new(root / "formal_coverage_audit.json", coverage)
    if any(row["return_code"] for row in exits) or coverage["status"] != "PASS":
        raise RuntimeError("FORMAL_WORKER_OR_COVERAGE_FAILED")
    formal = queue_results(queue, root / "formal_results")
    local_rows = [item for result in formal.values() for item in result["local_contract"]["components"]]
    if len(local_rows) != 6144 or not all(row["pass"] for row in local_rows):
        raise RuntimeError("LOCAL_DIRECTION_CONTRACT")
    main_count = sum(
        sum(component["family"] in MAIN_FAMILIES for component in result["finite_utility"][reference]["components"])
        for result in formal.values() for reference in ("normalized_zero", "blur_rgb")
    )
    secondary_count = sum(
        sum(component["family"] not in MAIN_FAMILIES for component in result["finite_utility"][reference]["components"])
        for result in formal.values() for reference in ("normalized_zero", "blur_rgb")
    )
    zeroing_count = sum(len(result["zeroing_secondary"]["components"]) for result in formal.values())
    retained = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    gpu_hours = formal_seconds * len(selected) / 3600
    report = {
        "status": PASS_STATUS, "stage": "ENDPOINT-E4",
        "claim_scope": "EXECUTION_PASS_NOT_FINAL_C4A_CLAIM_DECISION", "producer_commit": head,
        "evaluation_images": 400, "models": spec["models"], "image_model_tasks": 800,
        "reference_count": 2, "main_component_reference_evaluations": main_count,
        "secondary_component_reference_evaluations": secondary_count,
        "zeroing_secondary_evaluations": zeroing_count, "local_component_evaluations": len(local_rows),
        "local_eta_evaluations": len(local_rows) * 4, "all_local_contracts_pass": True,
        "maximum_local_minimum_absolute_error": max(row["minimum_absolute_error"] for row in local_rows),
        "maximum_local_directional_prediction_absolute_error": max(row["directional_prediction_absolute_error"] for row in local_rows),
        "maximum_noop_replacement_abs": max(result["local_contract"].get("noop_max_abs", 0.0) for result in formal.values()),
        "descriptive_per_image_spearman_summary": aggregate_descriptive(formal),
        "formal_coverage": coverage, "task_manifest_digest": queue_identity["manifest_digest"],
        "materialization_records_digest": material["records_digest"],
        "latency_seconds": {
            "exclusive_scan_p50": linear_percentile([row["scan_seconds"] for row in formal.values()], 0.5),
            "exclusive_scan_p95": linear_percentile([row["scan_seconds"] for row in formal.values()], 0.95),
            "exclusive_scan_max": max(row["scan_seconds"] for row in formal.values()),
            "end_to_end_task_p50": linear_percentile([row["task_seconds"] for row in formal.values()], 0.5),
            "end_to_end_task_p95": linear_percentile([row["task_seconds"] for row in formal.values()], 0.95),
            "end_to_end_task_max": max(row["task_seconds"] for row in formal.values()),
        },
        "peak_allocated_gpu_bytes_max": max(row["peak_allocated_gpu_bytes"] for row in formal.values()),
        "peak_reserved_gpu_bytes_max": max(row["peak_reserved_gpu_bytes"] for row in formal.values()),
        "peak_cpu_rss_bytes_max": max(row["peak_cpu_rss_bytes"] for row in formal.values()),
        "retained_bytes_before_report": retained, "formal_wall_seconds": formal_seconds,
        "total_wall_seconds": time.monotonic() - stage_started, "gpu_hours_upper_observed": gpu_hours,
        "formal_workers": len(selected), "worker_exits": exits,
        "implementation_bindings": [{"path": path, "sha256": file_sha256(REPO / path), "commit": head} for path in SOURCE_FILES],
        "deferred": {
            "c4a_bootstrap": "ENDPOINT-E6", "c4a_holm": "ENDPOINT-E6", "c4a_claim_decisions": "ENDPOINT-E6",
            "c4b_spatial_intervention": "ENDPOINT-E5_REQUIRES_EXPLICIT_USER_REQUEST",
            "path_ig_input_fidelity_stability": "NOT_RUN_NOT_ALLOWED",
        },
        "resource_caps": {"gpu_hours": 136, "wall_hours": 132, "incremental_retained_gib": 56},
        "next_authorization": "ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    if gpu_hours > 136 or formal_seconds > 132 * 3600 or retained > 56 * 1024 ** 3:
        raise RuntimeError("PAUSED_RESOURCE_CAP")
    write_json_new(root / "validation_report.json", report)
    files = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}]
    manifest = {"files": files, "canonical_digest": digest(files)}; write_json_new(root / "artifact_manifest.json", manifest)
    terminal = {
        "status": PASS_STATUS, "stage": "ENDPOINT-E4", "claim_scope": "EXECUTION_PASS_NOT_FINAL_C4A_CLAIM_DECISION",
        "producer_commit": head, "evaluation_image_count": 400, "image_model_task_count": 800,
        "main_component_reference_evaluation_count": main_count, "secondary_component_reference_evaluation_count": secondary_count,
        "local_component_evaluation_count": len(local_rows), "local_eta_evaluation_count": len(local_rows) * 4,
        "zeroing_secondary_evaluation_count": zeroing_count, "validation_digest": digest(report),
        "manifest_digest": manifest["canonical_digest"], "next_authorization": "ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    write_json_new(root / "terminal_receipt.json", terminal); print(json.dumps(terminal, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
