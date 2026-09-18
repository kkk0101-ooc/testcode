#!/usr/bin/env python3
"""Recompute the sealed Endpoint Scanner EP1 engineering evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "reports/nns_vit_endpoint_ep1_r11_20260918"
EP0_COMMIT = "6237ef87a1af0205973b76bee172defe33a0a6da"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def validate(root: Path) -> dict:
    root = root.resolve()
    report, manifest, terminal = (read(root / name) for name in ("validation_report.json", "artifact_manifest.json", "terminal_receipt.json"))
    require(report["status"] == terminal["status"] == "PASS_EP1_IMPLEMENTATION_AND_REGRESSION", "EP1_STATUS")
    require(report["scientific_results"] == terminal["scientific_results"] == "NOT_RUN", "SCIENTIFIC_SCOPE")
    require(not report["production_outcome_access"] and not terminal["production_outcome_access"], "OUTCOME_ACCESS")
    require(not report["ep2_started"] and not terminal["ep2_started"], "STAGE_SCOPE")
    require(report["ep0_commit"] == EP0_COMMIT, "EP0_BINDING")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", EP0_COMMIT, "HEAD"], cwd=REPO).returncode == 0, "EP0_ANCESTRY")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", EP0_COMMIT, report["executed_head"]], cwd=REPO).returncode == 0, "EXECUTION_HEAD_ANCESTRY")
    require(terminal["validation_digest"] == digest(report), "VALIDATION_BINDING")
    require(manifest["canonical_digest"] == terminal["manifest_digest"] == digest(manifest["files"]), "MANIFEST_BINDING")
    paths = [row["path"] for row in manifest["files"]]
    require(len(paths) == len(set(paths)), "DUPLICATE_MANIFEST_PATH")
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        require(path.is_relative_to(root) and path.is_file(), "MANIFEST_PATH")
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"], "ARTIFACT_HASH:" + row["path"])
    for row in report["implementation_bindings"]:
        require(file_sha(REPO / row["path"]) == row["sha256"], "IMPLEMENTATION_DRIFT:" + row["path"])
    task_manifest = read(root / "task_manifest.json")
    require(task_manifest["task_count"] == 8 and not task_manifest["outcome_data_access"], "TASK_MANIFEST_SCOPE")
    require(task_manifest["manifest_digest"] == digest(task_manifest["tasks"]), "TASK_MANIFEST_DIGEST")
    result_sets = {}
    for mode in ("single", "multi"):
        with sqlite3.connect(root / f"{mode}_queue.sqlite3") as connection:
            rows = connection.execute("SELECT task_id,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id").fetchall()
            attempts = connection.execute("SELECT task_id,attempt,status FROM attempts ORDER BY task_id,attempt").fetchall()
        require(len(rows) == 8 and all(row[1] == "completed" for row in rows), "QUEUE_COVERAGE:" + mode)
        require(len([row for row in attempts if row[2] == "COMPLETED"]) == 8, "QUEUE_DUPLICATE_ACCEPTED_ATTEMPT:" + mode)
        result_sets[mode] = {}
        for task_id, _state, attempt, relative, expected_digest in rows:
            value = read(root / f"{mode}_results" / relative)
            observed = dict(value); signed_digest = observed.pop("canonical_digest")
            require(signed_digest == expected_digest == digest(observed), "RESULT_DIGEST")
            require(value["task_id"] == task_id and value["attempt"] == attempt, "RESULT_IDENTITY")
            result_sets[mode][task_id] = value["result"]
    require(set(result_sets["single"]) == set(result_sets["multi"]), "SINGLE_MULTI_TASK_SET")
    require(all(result_sets["single"][key]["numeric_digest"] == result_sets["multi"][key]["numeric_digest"] for key in result_sets["single"]), "SINGLE_MULTI_NUMERIC")
    results = list(result_sets["multi"].values())
    required_models = {"vit_b16", "deit_s16"}
    require({row["model_id"] for row in results} == required_models, "MODEL_COVERAGE")
    require(sorted(set(block for row in results for block in row["blocks"])) == list(range(12)), "ALL_BLOCK_TARGET_COVERAGE")
    require(all(row["target_count"] == 16 for row in results), "TARGET_KIND_COVERAGE")
    require(all(row["unfused_structural_support"] and not row["fused_structural_support"] for row in results), "BACKEND_SCOPE")
    require(all(row["fused_supported"] and row["fused_unfused_max_abs"] <= 2e-5 for row in results), "FUSED_UNFUSED_PARITY")
    require(all(row["hook_parity_max_abs"] == row["formal_hook_parity_max_abs"] == 0 for row in results), "HOOK_PARITY")
    require(report["cost_axes_exercised"]["K"] == [1, 16] and report["cost_axes_exercised"]["M"] == [4, 8] and len(report["cost_axes_exercised"]["C"]) >= 1, "COST_AXES")
    require(all(item["finite"] for row in results for item in row["ig_benchmarks"]), "IG_BENCHMARK_NONFINITE")
    require(report["closure_max_abs"] <= 1e-5 and report["closure_relative_l2"] <= 1e-4, "CLOSURE_TOLERANCE")
    identities = [row["worker_identity"] for row in results]
    require(all(row["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and row["startup_matmul_backward"] == "PASS" for row in identities), "CUDA_STARTUP_CONTRACT")
    require(all(row["sys_executable_realpath"] == str((REPO.parent / ".conda/envs/vitnns/bin/python3.10").resolve()) for row in identities), "PYTHON_IDENTITY")
    require(report["gpu_workers"] >= 2 and len({row["gpu_uuid"] for row in identities}) >= 2, "MULTI_GPU_NOT_EXERCISED")
    require(report["throughput"]["single_tasks_per_second"] > 0 and report["throughput"]["multi_tasks_per_second"] > 0 and report["throughput"]["single_task_latency_p95_conservative_max"] >= report["throughput"]["single_task_latency_p50"], "THROUGHPUT_RECORD")
    require(len(report["ep1_checklist"]) == 18 and len(set(report["ep1_checklist"])) == 18, "EP1_CHECKLIST")
    require(report["gpu_hours_upper_observed"] <= 8 and report["retained_bytes"] < 14 * 1024 ** 3, "EP1_RESOURCE_CAP")
    require("passed" in report["cpu_tests"] and "failed" not in report["cpu_tests"], "CPU_REGRESSION")
    return {
        "status": "PASS_EP1_IMPLEMENTATION_AND_REGRESSION", "artifact_count": len(paths),
        "task_count_per_run": len(rows), "model_count": len(required_models), "gpu_worker_count": report["gpu_workers"],
        "max_closure_abs": report["closure_max_abs"], "max_fused_unfused_abs": report["fused_unfused_max_abs"],
        "production_outcome_access": False, "next_authorization": report["next_authorization"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    print(json.dumps(validate(parser.parse_args().root), sort_keys=True))
