#!/usr/bin/env python3
"""CPU-only independent validation of the sealed ENDPOINT-E2 atlas."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_e2_atlas_r1_20260918"
EP0 = REPO / "manifests/nns_vit_endpoint/ep0_r1_20260918"
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_e2_atlas_run_spec.json"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e2_atlas import (  # noqa: E402
    PASS_STATUS, array_sha256, digest, file_sha256, validate_run_spec,
)


def require(condition, code: str) -> None:
    if not condition:
        raise ValueError(code)


def load(path: Path):
    return json.loads(path.read_text())


def git_blob_sha(commit: str, path: str) -> str:
    value = subprocess.check_output(["git", "show", f"{commit}:vitnns_rebuild/{path}"], cwd=REPO)
    return hashlib.sha256(value).hexdigest()


def load_queue(root: Path, queue: Path, results: Path, expected_tasks: int) -> dict[str, dict]:
    with sqlite3.connect(queue) as connection:
        rows = connection.execute(
            "SELECT task_id,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id"
        ).fetchall()
        attempts = connection.execute(
            "SELECT task_id,attempt,status FROM attempts ORDER BY task_id,attempt"
        ).fetchall()
    require(len(rows) == expected_tasks and all(row[1] == "completed" for row in rows), "QUEUE_COVERAGE")
    require(len([row for row in attempts if row[2] == "COMPLETED"]) == expected_tasks, "QUEUE_ACCEPTED_ATTEMPTS")
    output = {}
    for task_id, _state, attempt, relative, expected_digest in rows:
        value = load(results / relative)
        unsigned = dict(value)
        observed = unsigned.pop("canonical_digest")
        require(observed == expected_digest == digest(unsigned), "QUEUE_RESULT_DIGEST")
        require(value["task_id"] == task_id and value["attempt"] == attempt, "QUEUE_RESULT_IDENTITY")
        output[task_id] = value["result"]
    return output


def validate(root: Path) -> dict:
    root = root.resolve()
    spec = load(SPEC_PATH)
    validate_run_spec(spec)
    report = load(root / "validation_report.json")
    terminal = load(root / "terminal_receipt.json")
    manifest = load(root / "artifact_manifest.json")
    material = load(root / "materialization_receipt.json")
    tasks = load(root / "task_manifest.json")
    formal_coverage = load(root / "formal_coverage_audit.json")
    replay_coverage = load(root / "single_replay/coverage_audit.json")
    require(report["status"] == terminal["status"] == PASS_STATUS, "STATUS")
    require(report["stage"] == terminal["stage"] == "ENDPOINT-E2", "STAGE")
    require(report["claim_decision"] == terminal["claim_decision"] == "C2_REGISTERED_SCAN_COMPLETION_AND_RESOURCES", "CLAIM_SCOPE")
    require(report["scientific_utility_claims"] == "NOT_RUN_NOT_ALLOWED", "UTILITY_SCOPE")
    require(report["path_ig"] == report["input_fidelity_stability"] == report["finite_intervention"] == "NOT_RUN_NOT_ALLOWED", "METHOD_SCOPE")
    require(terminal["validation_digest"] == digest(report), "VALIDATION_DIGEST")
    require(terminal["manifest_digest"] == manifest["canonical_digest"] == digest(manifest["files"]), "MANIFEST_DIGEST")
    paths = [row["path"] for row in manifest["files"]]
    require(len(paths) == len(set(paths)), "MANIFEST_DUPLICATE_PATH")
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        require(path.is_relative_to(root) and path.is_file(), "MANIFEST_PATH")
        require(path.stat().st_size == row["bytes"] and file_sha256(path) == row["sha256"], "ARTIFACT_HASH:" + row["path"])
    require(subprocess.run(
        ["git", "merge-base", "--is-ancestor", report["producer_commit"], "HEAD"], cwd=REPO,
    ).returncode == 0, "PRODUCER_ANCESTRY")
    for row in report["implementation_bindings"]:
        require(git_blob_sha(report["producer_commit"], row["path"]) == row["sha256"], "IMPLEMENTATION_BINDING:" + row["path"])
    cohort = load(EP0 / "cohort_registry.json")
    expected = sorted(
        (row for row in cohort["rows"] if row["split"] == "evaluation"),
        key=lambda row: row["split_order"],
    )
    require(len(expected) == material["authorized_member_count"] == material["archive_member_reads"] == 400, "MATERIALIZATION_COUNT")
    require(material["calibration_member_reads"] == material["discovery_member_reads"] == material["replacement_count"] == 0, "OUT_OF_SCOPE_ACCESS")
    require(material["records_digest"] == digest(material["records"]), "MATERIALIZATION_DIGEST")
    for frozen, actual in zip(expected, material["records"]):
        require((
            actual["split_order"], actual["sample_id"], actual["class_id"], actual["wnid"], actual["source_object_id"]
        ) == (
            frozen["split_order"], frozen["sample_id"], frozen["class_id"], frozen["wnid"], frozen["source_object_id"]
        ), "COHORT_DRIFT")
    archive = REPO / material["archive"]["path"]
    require(archive.stat().st_size == material["archive"]["bytes"] and file_sha256(archive) == material["archive"]["sha256"], "ARCHIVE_IDENTITY")
    by_name = {row["sample_id"]: row for row in material["records"]}
    seen = set()
    with tarfile.open(archive, "r:") as stream:
        for member in stream:
            name = Path(member.name).name
            if name not in by_name:
                continue
            require(name not in seen and member.isfile(), "ARCHIVE_MEMBER_DUPLICATE_OR_TYPE")
            handle = stream.extractfile(member)
            require(handle is not None, "ARCHIVE_MEMBER_READ")
            payload = handle.read()
            record = by_name[name]
            require(len(payload) == record["bytes"] and hashlib.sha256(payload).hexdigest() == record["sha256"], "ARCHIVE_MEMBER_HASH")
            seen.add(name)
    require(seen == set(by_name), "ARCHIVE_MEMBER_COVERAGE")
    require(tasks["task_count"] == 800 and tasks["evaluation_images"] == 400 and tasks["model_count"] == 2, "TASK_SCOPE")
    require(tasks["target_count_by_model"] == {"vit_b16": 457, "deit_s16": 385}, "TARGET_COUNTS")
    require(tasks["target_evaluations"] == terminal["target_evaluation_count"] == 336800, "TARGET_EVALUATIONS")
    require(tasks["manifest_digest"] == digest(tasks["tasks"]), "TASK_DIGEST")
    require(tasks["queue_identities"]["formal"] == tasks["queue_identities"]["single_replay"], "SAME_TASK_MANIFEST")
    for coverage in (formal_coverage, replay_coverage):
        require(coverage["status"] == "PASS" and coverage["task_count"] == coverage["completed_count"] == 800, "COVERAGE")
        require(not coverage["missing_task_ids"] and not coverage["invalid"] and not coverage["duplicate_task_ids"], "COVERAGE_DETAILS")
    formal = load_queue(root, root / "formal_queue.sqlite3", root / "formal_results", 800)
    replay = load_queue(root, root / "single_replay/queue.sqlite3", root / "single_replay/results", 800)
    require(set(formal) == set(replay), "SINGLE_MULTI_TASK_SET")
    target_registry = load(EP0 / "target_registry.json")
    target_counts = {model_id: len(rows) for model_id, rows in target_registry["targets_by_model"].items()}
    require(target_counts == {"vit_b16": 457, "deit_s16": 385}, "REGISTRY_TARGET_COUNTS")
    target_evaluations = 0
    class_maps = 0
    full_atlases = 0
    observed_near_zero = []
    for task_id, result in formal.items():
        replay_result = replay[task_id]
        require(result["numeric_digest"] == replay_result["numeric_digest"], "SINGLE_MULTI_NUMERIC")
        require(result["summaries_digest"] == digest(result["summaries"]), "SUMMARY_DIGEST")
        target_rows = target_registry["targets_by_model"][result["model_id"]]
        require(result["target_count"] == len(result["summaries"]) == len(target_rows), "TARGET_COVERAGE")
        require([row["target_id"] for row in result["summaries"]] == [row["target_id"] for row in target_rows], "TARGET_ORDER")
        require(result["finite_target_count"] == result["target_count"] and all(row["finite"] for row in result["summaries"]), "FINITE_TARGETS")
        require(all(math.isfinite(float(row[key])) for row in result["summaries"] for key in ("value", "signed_sum", "absolute_mass", "max_abs")), "FINITE_SUMMARIES")
        require(result["method_id"] == "canonical_endpoint_x_times_gradient" and result["forward_count"] == 1, "METHOD_IDENTITY")
        require(result["input_shape"] == [1, 3, 224, 224] and result["dtype"] == "torch.float32", "INPUT_IDENTITY")
        require(result["image_sha256"] == replay_result["image_sha256"] and result["logits_sha256"] == replay_result["logits_sha256"], "REPLAY_SOURCE")
        artifacts = {row["kind"]: row for row in result["artifacts"]}
        require(set(artifacts) >= {"raw_signed_rgb_class_maps"}, "CLASS_MAP_MISSING")
        class_maps += 1
        class_path = REPO / artifacts["raw_signed_rgb_class_maps"]["path"]
        class_value = np.load(class_path, allow_pickle=False)
        require(class_value.shape == (2, 3, 224, 224) and class_value.dtype == np.float32 and np.isfinite(class_value).all(), "CLASS_MAP_CONTRACT")
        class_rows = [row for row in result["summaries"] if row["kind"] in {"ground_truth_logit", "frozen_clean_top1_logit"}]
        require(len(class_rows) == 2 and all(array_sha256(class_value[index]) == row["map_sha256"] for index, row in enumerate(class_rows)), "CLASS_MAP_NUMERIC_BINDING")
        require(class_path.stat().st_size == artifacts["raw_signed_rgb_class_maps"]["bytes"] and file_sha256(class_path) == artifacts["raw_signed_rgb_class_maps"]["sha256"], "CLASS_MAP_HASH")
        if result["split_order"] < 16:
            require("full_raw_atlas" in artifacts, "FULL_ATLAS_MISSING")
            atlas_path = REPO / artifacts["full_raw_atlas"]["path"]
            atlas_value = np.load(atlas_path, mmap_mode="r", allow_pickle=False)
            require(atlas_value.shape == (result["target_count"], 3, 224, 224) and atlas_value.dtype == np.float32, "FULL_ATLAS_CONTRACT")
            require(all(np.isfinite(atlas_value[index]).all() and array_sha256(atlas_value[index]) == summary["map_sha256"] for index, summary in enumerate(result["summaries"])), "FULL_ATLAS_NUMERIC_BINDING")
            require(atlas_path.stat().st_size == artifacts["full_raw_atlas"]["bytes"] and file_sha256(atlas_path) == artifacts["full_raw_atlas"]["sha256"], "FULL_ATLAS_HASH")
            full_atlases += 1
        else:
            require("full_raw_atlas" not in artifacts, "FULL_ATLAS_SCOPE")
        for summary in result["summaries"]:
            if summary["near_zero"]:
                observed_near_zero.append((task_id, summary["target_id"], summary["max_abs"]))
        target_evaluations += result["target_count"]
        identity = result["worker_identity"]
        require(identity["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and identity["startup_matmul_backward"] == "PASS", "CUDA_STARTUP")
        require(identity["deterministic_algorithms"] and not identity["cuda_matmul_allow_tf32"] and not identity["cudnn_allow_tf32"] and not identity["autocast"], "DETERMINISM")
    require(target_evaluations == report["target_evaluations"] == report["finite_target_evaluations"] == 336800, "TOTAL_TARGET_COVERAGE")
    require(class_maps == report["raw_class_map_task_count"] == 800, "CLASS_MAP_COVERAGE")
    require(full_atlases == report["full_raw_atlas_task_count"] == 32, "ATLAS_COVERAGE")
    near_zero = load(root / "near_zero_index.json")
    require(near_zero["threshold_max_abs"] == 1e-12 and near_zero["count"] == len(observed_near_zero), "NEAR_ZERO_INDEX")
    require(load(root / "failure_index.json") == {"count": 0, "rows": []}, "FAILURE_INDEX")
    gallery = load(root / "gallery_manifest.json")
    require(gallery == report["gallery"] and gallery["status"] == "PASS" and len(gallery["galleries"]) == 2, "GALLERY")
    for row in gallery["galleries"]:
        path = root / row["path"]
        require(row["selection_orders"] == list(range(8)) and row["display_only_not_statistics"] and not row["smoothing"] and not row["clipping"], "GALLERY_SCOPE")
        require(path.is_file() and path.stat().st_size == row["bytes"] and file_sha256(path) == row["sha256"], "GALLERY_HASH")
    require(report["single_multi_exact_numeric_digest_count"] == 800 and not report["single_multi_mismatch_task_ids"], "SINGLE_MULTI_REPORT")
    require(report["formal_workers"] >= 1 and report["replay_workers"] == 1, "WORKERS")
    require(report["gpu_hours_upper_observed"] <= 64 and report["total_wall_seconds"] <= 72 * 3600, "TIME_CAP")
    require(report["peak_reserved_gpu_bytes_max"] <= 24 * 1024 ** 3 and report["peak_cpu_rss_bytes_max"] <= 32 * 1024 ** 3, "MEMORY_CAP")
    require(report["retained_bytes_before_report"] <= 128 * 1024 ** 3 and report["raw_retained_bytes"] > 0, "STORAGE_CAP")
    require(all(value > 0 and math.isfinite(value) for value in report["latency_seconds"].values()), "LATENCY")
    require(all(value > 0 and math.isfinite(value) for value in report["throughput"].values()), "THROUGHPUT")
    require(terminal["next_authorization"] == report["next_authorization"] == "ENDPOINT_E3_REQUIRES_EXPLICIT_USER_REQUEST", "NEXT_AUTHORIZATION")
    return {
        "status": PASS_STATUS, "producer_commit": report["producer_commit"],
        "evaluation_images": 400, "image_model_tasks": 800,
        "target_evaluations": 336800, "full_raw_atlas_tasks": 32,
        "raw_class_map_tasks": 800, "single_multi_exact": 800,
        "gpu_hours": report["gpu_hours_upper_observed"],
        "next_authorization": report["next_authorization"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    print(json.dumps(validate(parser.parse_args().root), sort_keys=True))
