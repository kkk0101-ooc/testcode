#!/usr/bin/env python3
"""Finalize ENDPOINT-E2 from the completed formal queue after replay cancellation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import subprocess

import numpy as np
from PIL import Image, ImageDraw

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO / "reports/nns_vit_endpoint_e2_atlas_r1_20260918"
ROOT = REPO / "reports/nns_vit_endpoint_e2_atlas_r4_20260918"
CACHE = REPO / "assets/data/nns_vit_endpoint_e2_evaluation_20260918"
AMENDMENT = REPO / "config/nns_vit_endpoint/endpoint_e2_atlas_execution_amendment_r1.json"
RECOVERY = REPO / "config/nns_vit_endpoint/endpoint_e2_atlas_handoff_recovery_r4.json"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e2_atlas import (  # noqa: E402
    PASS_STATUS, digest, display_rgb_signed_sum, file_sha256, linear_percentile, write_json_new,
)


def load(path: Path):
    return json.loads(path.read_text())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def remote(ref: str) -> str:
    fields = subprocess.check_output(["git", "ls-remote", "origin", ref], cwd=REPO, text=True).split()
    return fields[0] if len(fields) == 2 else ""


def git_blob_sha(commit: str, path: str) -> str:
    value = subprocess.check_output(["git", "show", f"{commit}:vitnns_rebuild/{path}"], cwd=REPO)
    return hashlib.sha256(value).hexdigest()


def queue_results(queue: Path, results: Path) -> dict[str, dict]:
    output = {}
    with sqlite3.connect(queue) as connection:
        rows = connection.execute(
            "SELECT task_id,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id"
        ).fetchall()
        attempts = connection.execute("SELECT status,count(*) FROM attempts GROUP BY status").fetchall()
    if len(rows) != 800 or any(row[1] != "completed" for row in rows) or attempts != [("COMPLETED", 800)]:
        raise RuntimeError("FORMAL_QUEUE_NOT_EXACTLY_COMPLETE")
    for task_id, _state, attempt, relative, expected_digest in rows:
        value = load(results / relative)
        unsigned = dict(value); observed = unsigned.pop("canonical_digest")
        if observed != expected_digest or digest(unsigned) != observed or value["attempt"] != attempt:
            raise RuntimeError("FORMAL_RESULT_DIGEST:" + task_id)
        output[task_id] = value["result"]
    return output


def replay_disposition() -> dict:
    queue = SOURCE_ROOT / "single_replay/queue.sqlite3"
    with sqlite3.connect(queue) as connection:
        states = dict(connection.execute("SELECT state,count(*) FROM tasks GROUP BY state").fetchall())
        attempts = dict(connection.execute("SELECT status,count(*) FROM attempts GROUP BY status").fetchall())
        completed = connection.execute(
            "SELECT task_id FROM tasks WHERE state='completed' ORDER BY task_id"
        ).fetchall()
        claimed = connection.execute(
            "SELECT task_id,attempt,worker_id,device_uuid FROM tasks WHERE state='claimed' ORDER BY task_id"
        ).fetchall()
    return {
        "status": "USER_CANCELED_ENGINEERING_REPLAY_NOT_E2_EVIDENCE",
        "user_instruction": "不要单卡重放了，没意义啊，多卡跑完就行了，你设计规划的有问题",
        "queue_states_at_cancellation": states, "attempt_states_at_cancellation": attempts,
        "completed_task_count": len(completed), "completed_task_ids_digest": digest([row[0] for row in completed]),
        "claimed_at_cancellation": [{
            "task_id": row[0], "attempt": row[1], "worker_id": row[2], "device_uuid": row[3],
        } for row in claimed],
        "included_in_endpoint_e2_pass": False, "included_in_scientific_evidence": False,
        "completion_required": False, "preserved_without_deletion_or_rewrite": True,
    }


def render_galleries(formal: dict[str, dict]) -> dict:
    rendered = []
    for model_id in ("vit_b16", "deit_s16"):
        panels, scales = [], []
        for order in range(8):
            row = formal[f"{model_id}__{order:03d}"]
            with Image.open(CACHE / row["sample_id"]) as source:
                clean = source.convert("RGB")
                width, height = clean.size
                resized = (248, round(height * 248 / width)) if width <= height else (round(width * 248 / height), 248)
                clean = clean.resize(resized, resample=Image.Resampling.BICUBIC)
                left = (clean.width - 224) // 2; top = (clean.height - 224) // 2
                clean = clean.crop((left, top, left + 224, top + 224))
            artifact = next(item for item in row["artifacts"] if item["kind"] == "raw_signed_rgb_class_maps")
            maps = np.load(REPO / artifact["path"], allow_pickle=False)
            gt, gt_scale, gt_zero = display_rgb_signed_sum(maps[0])
            top_map, top_scale, top_zero = display_rgb_signed_sum(maps[1])
            panels.append((clean, Image.fromarray(gt), Image.fromarray(top_map), row))
            scales.append({
                "split_order": order, "sample_id": row["sample_id"],
                "ground_truth_scale": gt_scale, "ground_truth_zero": gt_zero,
                "clean_top1_scale": top_scale, "clean_top1_zero": top_zero,
            })
        sheet = Image.new("RGB", (3 * 224, 8 * 252), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, (clean, gt, top_map, record) in enumerate(panels):
            y = row_index * 252
            sheet.paste(clean, (0, y + 28)); sheet.paste(gt, (224, y + 28)); sheet.paste(top_map, (448, y + 28))
            draw.text((4, y + 5), f"{row_index:03d} {record['sample_id']} | input | GT Endpoint | clean-top1 Endpoint", fill="black")
        path = ROOT / "gallery" / f"{model_id}_first8.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            sheet.save(stream, format="PNG", optimize=False); stream.flush(); os.fsync(stream.fileno())
        rendered.append({
            "model_id": model_id, "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size, "sha256": file_sha256(path),
            "selection_orders": list(range(8)), "display_scales": scales,
            "display_only_not_statistics": True, "smoothing": False, "clipping": False,
        })
    return {"status": "PASS", "galleries": rendered}


def main() -> int:
    if ROOT.exists():
        raise RuntimeError("FINAL_ROOT_EXISTS")
    ROOT.mkdir(parents=True)
    amendment = load(AMENDMENT)
    recovery = load(RECOVERY)
    if file_sha256(REPO / amendment["base_contract"]["path"]) != amendment["base_contract"]["sha256"]:
        raise RuntimeError("BASE_CONTRACT_BINDING")
    amendment_commit = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--short", "HEAD")
    if branch != "stage-d-d3-pass" or not amendment_commit == git("rev-parse", "@{upstream}") == remote("refs/heads/" + branch):
        raise RuntimeError("AMENDMENT_NOT_PUBLISHED")
    formal = queue_results(SOURCE_ROOT / "formal_queue.sqlite3", SOURCE_ROOT / "formal_results")
    canceled = replay_disposition()
    write_json_new(ROOT / "replay_cancellation_receipt.json", canceled)
    gallery = render_galleries(formal)
    write_json_new(ROOT / "gallery_manifest.json", gallery)
    near_zero = []
    for task_id, result in sorted(formal.items()):
        for target in result["summaries"]:
            if target["near_zero"]:
                near_zero.append({
                    "task_id": task_id, "model_id": result["model_id"],
                    "sample_id": result["sample_id"], "split_order": result["split_order"],
                    "target_id": target["target_id"], "max_abs": target["max_abs"],
                })
    write_json_new(ROOT / "near_zero_index.json", {"threshold_max_abs": 1e-12, "count": len(near_zero), "rows": near_zero})
    write_json_new(ROOT / "failure_index.json", {"count": 0, "rows": []})
    latencies = [row["scan_seconds"] for row in formal.values()]
    task_latencies = [row["task_seconds"] for row in formal.values()]
    total_targets = sum(row["target_count"] for row in formal.values())
    raw_bytes = sum(item["bytes"] for row in formal.values() for item in row["artifacts"])
    with sqlite3.connect(SOURCE_ROOT / "formal_queue.sqlite3") as connection:
        bounds = connection.execute("SELECT min(claimed_ns),max(completed_ns) FROM attempts").fetchone()
    formal_wall_seconds = (bounds[1] - bounds[0]) / 1e9
    model_costs = {}
    target_counts = {"vit_b16": 457, "deit_s16": 385}
    for model_id in ("vit_b16", "deit_s16"):
        values = [row["scan_seconds"] for row in formal.values() if row["model_id"] == model_id]
        model_costs[model_id] = {
            "images": 400, "registered_targets": target_counts[model_id],
            "median_seconds_per_complete_atlas": linear_percentile(values, 0.5),
            "p95_seconds_per_complete_atlas": linear_percentile(values, 0.95),
            "max_seconds_per_complete_atlas": max(values),
        }
    retained_before_report = sum(path.stat().st_size for path in SOURCE_ROOT.rglob("*") if path.is_file())
    source_files = []
    source_prefixes = ("formal_results/", "raw/", "logs/formal/")
    source_names = {"materialization_receipt.json", "task_manifest.json", "startup_snapshot.json",
                    "formal_coverage_audit.json", "formal_queue.sqlite3"}
    for path in sorted(SOURCE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        if relative in source_names or relative.startswith(source_prefixes):
            source_files.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    source_binding = {
        "formal_source_root": SOURCE_ROOT.relative_to(REPO).as_posix(),
        "preliminary_finalization_status": "R1_R2_R3_INVALID_HANDOFFS_PRESERVED_NOT_AUTHORITY",
        "preliminary_terminals": [
            "reports/nns_vit_endpoint_e2_atlas_r1_20260918/terminal_receipt.json",
            "reports/nns_vit_endpoint_e2_atlas_r2_20260918/terminal_receipt.json"
            ,"reports/nns_vit_endpoint_e2_atlas_r3_20260918/terminal_receipt.json"
        ],
        "canceled_replay_root": "single_replay",
        "formal_source_files": source_files, "formal_source_files_digest": digest(source_files),
    }
    write_json_new(ROOT / "formal_source_binding.json", source_binding)
    production_paths = [
        "config/nns_vit_endpoint/endpoint_e2_atlas_run_spec.json",
        "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE2执行合同_20260918.md",
        "src/vitnns/endpoint_e2_atlas.py", "scripts/build_endpoint_e2_materialization.py",
        "scripts/run_endpoint_e2_atlas_worker.py", "scripts/run_endpoint_e2_atlas.py",
        "tests/test_endpoint_e2_atlas.py",
    ]
    amendment_paths = [
        "config/nns_vit_endpoint/endpoint_e2_atlas_execution_amendment_r1.json",
        "config/nns_vit_endpoint/endpoint_e2_atlas_handoff_recovery_r2.json",
        "config/nns_vit_endpoint/endpoint_e2_atlas_handoff_recovery_r3.json",
        "config/nns_vit_endpoint/endpoint_e2_atlas_handoff_recovery_r4.json",
        "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE2执行修订R1_20260918.md",
        "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE2封存恢复R2_20260918.md",
        "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE2封存恢复R3_20260918.md",
        "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE2封存恢复R4_20260918.md",
        "scripts/finalize_endpoint_e2_atlas_r1.py", "scripts/validate_endpoint_e2_atlas.py",
    ]
    report = {
        "status": PASS_STATUS, "stage": "ENDPOINT-E2",
        "claim_decision": "C2_REGISTERED_SCAN_COMPLETION_AND_RESOURCES",
        "execution_amendment": {
            "path": AMENDMENT.relative_to(REPO).as_posix(), "sha256": file_sha256(AMENDMENT),
            "timing": amendment["amendment_timing"],
        },
        "handoff_recovery": {
            "path": RECOVERY.relative_to(REPO).as_posix(), "sha256": file_sha256(RECOVERY),
            "status": recovery["status"],
        },
        "formal_source_binding": source_binding,
        "scientific_utility_claims": "NOT_RUN_NOT_ALLOWED", "path_ig": "NOT_RUN_NOT_ALLOWED",
        "input_fidelity_stability": "NOT_RUN_NOT_ALLOWED", "finite_intervention": "NOT_RUN_NOT_ALLOWED",
        "producer_commit": amendment["base_contract"]["producer_commit"],
        "amendment_commit": amendment_commit,
        "formal_coverage": load(SOURCE_ROOT / "formal_coverage_audit.json"),
        "canceled_replay": canceled, "evaluation_images": 400,
        "models": ["vit_b16", "deit_s16"], "image_model_tasks": 800,
        "target_count_by_model": target_counts, "target_evaluations": total_targets,
        "finite_target_evaluations": sum(row["finite_target_count"] for row in formal.values()),
        "near_zero_target_evaluations": len(near_zero),
        "raw_class_map_task_count": sum(any(item["kind"] == "raw_signed_rgb_class_maps" for item in row["artifacts"]) for row in formal.values()),
        "full_raw_atlas_task_count": sum(any(item["kind"] == "full_raw_atlas" for item in row["artifacts"]) for row in formal.values()),
        "gallery": gallery,
        "latency_seconds": {
            "exclusive_scan_p50": linear_percentile(latencies, 0.5), "exclusive_scan_p95": linear_percentile(latencies, 0.95),
            "exclusive_scan_max": max(latencies), "end_to_end_task_p50": linear_percentile(task_latencies, 0.5),
            "end_to_end_task_p95": linear_percentile(task_latencies, 0.95), "end_to_end_task_max": max(task_latencies),
        },
        "throughput": {
            "formal_targets_per_second": total_targets / formal_wall_seconds,
            "formal_images_per_second": 800 / formal_wall_seconds,
            "exclusive_scan_targets_per_gpu_second": total_targets / sum(latencies),
            "exclusive_scan_images_per_gpu_second": 800 / sum(latencies),
        },
        "complete_atlas_amortized_cost": model_costs,
        "peak_allocated_gpu_bytes_max": max(row["peak_allocated_gpu_bytes"] for row in formal.values()),
        "peak_reserved_gpu_bytes_max": max(row["peak_reserved_gpu_bytes"] for row in formal.values()),
        "peak_cpu_rss_bytes_max": max(row["peak_cpu_rss_bytes"] for row in formal.values()),
        "raw_retained_bytes": raw_bytes, "stored_raw_bytes_per_target_evaluation": raw_bytes / total_targets,
        "retained_bytes_before_report": retained_before_report, "formal_wall_seconds": formal_wall_seconds,
        "gpu_hours_upper_observed": formal_wall_seconds * 4 / 3600, "formal_workers": 4,
        "model_tensor_digests": {
            model_id: next(row for row in formal.values() if row["model_id"] == model_id)["model_binding"]["tensor_digest"]
            for model_id in ("vit_b16", "deit_s16")
        },
        "implementation_bindings": (
            [{"path": path, "sha256": git_blob_sha(amendment["base_contract"]["producer_commit"], path), "commit": amendment["base_contract"]["producer_commit"], "role": "formal_production"} for path in production_paths]
            + [{"path": path, "sha256": git_blob_sha(amendment_commit, path), "commit": amendment_commit, "role": "post_formal_user_amendment"} for path in amendment_paths]
        ),
        "resource_caps": {"gpu_hours": 64, "wall_hours": 72, "incremental_retained_gib": 128},
        "next_authorization": "ENDPOINT_E3_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    write_json_new(ROOT / "validation_report.json", report)
    files = [{
        "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path),
    } for path in sorted(ROOT.rglob("*")) if path.is_file() and path.name not in {"artifact_manifest.json", "terminal_receipt.json"}]
    manifest = {"files": files, "canonical_digest": digest(files)}
    write_json_new(ROOT / "artifact_manifest.json", manifest)
    terminal = {
        "status": PASS_STATUS, "stage": "ENDPOINT-E2",
        "claim_decision": "C2_REGISTERED_SCAN_COMPLETION_AND_RESOURCES",
        "producer_commit": report["producer_commit"], "execution_amendment_sha256": file_sha256(AMENDMENT),
        "amendment_commit": amendment_commit,
        "evaluation_image_count": 400, "image_model_task_count": 800,
        "target_evaluation_count": total_targets, "canceled_replay_included": False,
        "validation_digest": digest(report), "manifest_digest": manifest["canonical_digest"],
        "next_authorization": "ENDPOINT_E3_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    write_json_new(ROOT / "terminal_receipt.json", terminal)
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
