#!/usr/bin/env python3
"""CPU-only independent validation of ENDPOINT-E3 structural closure."""
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

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
REPO = Path(__file__).resolve().parents[1]
GIT_ROOT = REPO.parent
ROOT = REPO / "reports/nns_vit_endpoint_e3_closure_r1_20260918"
EP0 = REPO / "manifests/nns_vit_endpoint/ep0_r1_20260918"
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_e3_closure_run_spec.json"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e2_atlas import array_sha256, digest, file_sha256  # noqa: E402
from vitnns.endpoint_e3_closure import (  # noqa: E402
    CLAIM_DECISION, CUT_ALIASES, PASS_STATUS, validate_run_spec,
)


def require(condition, code: str) -> None:
    if not condition:
        raise ValueError(code)


def load(path: Path):
    return json.loads(path.read_text())


def git_blob_sha(commit: str, path: str) -> str:
    value = subprocess.check_output(["git", "show", f"{commit}:vitnns_rebuild/{path}"], cwd=REPO)
    return hashlib.sha256(value).hexdigest()


def load_queue(queue: Path, results: Path, expected_tasks: int) -> tuple[dict[str, dict], list[dict]]:
    with sqlite3.connect(queue) as connection:
        rows = connection.execute(
            "SELECT task_id,payload_json,payload_digest,seed,state,attempt,result_path,result_digest "
            "FROM tasks ORDER BY task_id"
        ).fetchall()
        attempts = connection.execute(
            "SELECT task_id,attempt,status FROM attempts ORDER BY task_id,attempt"
        ).fetchall()
    require(len(rows) == expected_tasks and all(row[4] == "completed" for row in rows), "QUEUE_COVERAGE")
    require(len([row for row in attempts if row[2] == "COMPLETED"]) == expected_tasks, "QUEUE_ACCEPTED_ATTEMPTS")
    output, payloads = {}, []
    for task_id, payload_json, payload_digest, _seed, _state, attempt, relative, expected_digest in rows:
        require(hashlib.sha256(payload_json.encode()).hexdigest() == payload_digest, "QUEUE_PAYLOAD_DIGEST")
        payload = json.loads(payload_json)
        value = load(results / relative)
        unsigned = dict(value); observed = unsigned.pop("canonical_digest")
        require(observed == expected_digest == digest(unsigned), "QUEUE_RESULT_DIGEST")
        require(value["task_id"] == task_id and value["attempt"] == attempt, "QUEUE_RESULT_IDENTITY")
        output[task_id] = value["result"]
        payloads.append({"task_id": task_id, "payload": payload})
    return output, payloads


def validate_raw_task(result: dict) -> int:
    artifacts = result["artifacts"]
    require(len(artifacts) == 12, "RAW_BLOCK_COUNT")
    blocks = {row["block"]: row for row in result["blocks"]}
    for artifact in artifacts:
        path = REPO / artifact["path"]
        require(path.is_file() and path.stat().st_size == artifact["bytes"], "RAW_FILE")
        require(file_sha256(path) == artifact["sha256"], "RAW_FILE_HASH")
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.float32 and value.shape[1:] == (3, 224, 224), "RAW_ARRAY_CONTRACT")
        require(np.isfinite(value).all(), "RAW_ARRAY_NONFINITE")
        order = artifact["component_order"]
        require(value.shape[0] == len(order) and len(order) == len(set(order)), "RAW_ORDER")
        block = blocks[artifact["block"]]
        summaries = dict(block["components"]); summaries.update(block["reconstructions"])
        require(set(order) == set(summaries), "RAW_SUMMARY_ORDER")
        arrays = {name: value[index] for index, name in enumerate(order)}
        for name, array in arrays.items():
            summary = summaries[name]
            flat = array.reshape(-1)
            require(array_sha256(array) == summary["array_sha256"], "RAW_ARRAY_DIGEST")
            require(math.isclose(math.fsum(map(float, flat)), summary["signed_sum"], rel_tol=0, abs_tol=1e-10), "RAW_SIGNED_SUM")
            require(math.isclose(math.fsum(map(float, np.abs(flat))), summary["absolute_mass"], rel_tol=0, abs_tol=1e-10), "RAW_ABSOLUTE_MASS")
        for cut in block["cuts"]:
            summed = np.sum([arrays[name] for name in cut["component_ids"]], axis=0, dtype=np.float32)
            reference = arrays[cut["reconstruction_id"]]
            error = summed - reference
            maximum = float(np.max(np.abs(error)))
            norm = float(np.linalg.norm(reference))
            relative = float(np.linalg.norm(error) / max(norm, 1e-30))
            require(array_sha256(summed) == cut["summed_sha256"], "RAW_SUMMED_DIGEST")
            require(array_sha256(reference) == cut["reference_sha256"], "RAW_REFERENCE_DIGEST")
            require(math.isclose(maximum, cut["max_abs"], rel_tol=0, abs_tol=1e-12), "RAW_MAX_ABS")
            # NumPy and Torch use different parallel reduction trees for the
            # float32 L2 norms.  The raw arrays and elementwise residual are
            # already digest/exactly bound above; compare the derived scalar
            # with the separately registered handoff-audit tolerance.
            require(math.isclose(relative, cut["relative_l2"], rel_tol=2e-5, abs_tol=1e-10), "RAW_RELATIVE_L2")
            require(np.allclose(summed, reference, atol=1e-5, rtol=1e-4), "RAW_ELEMENTWISE_CLOSURE")
    return len(artifacts)


def validate(root: Path) -> dict:
    root = root.resolve()
    spec = load(SPEC_PATH); validate_run_spec(spec)
    report = load(root / "validation_report.json")
    terminal = load(root / "terminal_receipt.json")
    manifest = load(root / "artifact_manifest.json")
    require(report["status"] == terminal["status"] == PASS_STATUS, "STATUS")
    require(report["stage"] == terminal["stage"] == "ENDPOINT-E3", "STAGE")
    require(report["claim_decision"] == terminal["claim_decision"] == CLAIM_DECISION, "CLAIM_DECISION")
    require(report["claim_scope"] == terminal["claim_scope"] == "CORRECTNESS_ONLY_NOT_UTILITY", "CLAIM_SCOPE")
    require(terminal["validation_digest"] == digest(report), "VALIDATION_DIGEST")
    require(terminal["manifest_digest"] == manifest["canonical_digest"] == digest(manifest["files"]), "MANIFEST_DIGEST")
    paths = [row["path"] for row in manifest["files"]]
    require(len(paths) == len(set(paths)), "MANIFEST_DUPLICATE_PATH")
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        require(path.is_relative_to(root) and path.is_file(), "MANIFEST_PATH")
        require(path.stat().st_size == row["bytes"] and file_sha256(path) == row["sha256"], "ARTIFACT_HASH:" + row["path"])
    require(subprocess.run(["git", "merge-base", "--is-ancestor", report["producer_commit"], "HEAD"], cwd=REPO).returncode == 0, "PRODUCER_ANCESTRY")
    for row in report["implementation_bindings"]:
        require(row["commit"] == report["producer_commit"], "IMPLEMENTATION_COMMIT")
        require(git_blob_sha(row["commit"], row["path"]) == row["sha256"], "IMPLEMENTATION_BINDING:" + row["path"])
    e2 = load(REPO / "reports/nns_vit_endpoint_e2_atlas_r4_20260918/terminal_receipt.json")
    require(e2["status"] == "ENDPOINT_E2_PASS_NEXT_ENDPOINT_E3_REQUIRES_EXPLICIT_REQUEST", "E2_STATUS")
    require(e2["next_authorization"] == "ENDPOINT_E3_REQUIRES_EXPLICIT_USER_REQUEST", "E2_AUTHORIZATION")
    for key, relative in {
        "cohort_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json",
        "endpoint_e2_source_binding_sha256": "reports/nns_vit_endpoint_e2_atlas_r4_20260918/formal_source_binding.json",
        "endpoint_e2_terminal_sha256": "reports/nns_vit_endpoint_e2_atlas_r4_20260918/terminal_receipt.json",
        "endpoint_e2_validation_sha256": "reports/nns_vit_endpoint_e2_atlas_r4_20260918/validation_report.json",
        "implementation_contract_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/implementation_contract.json",
        "materialization_receipt_sha256": "reports/nns_vit_endpoint_e2_atlas_r1_20260918/materialization_receipt.json",
        "model_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
        "target_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json",
        "unified_plan_sha256": "docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md",
    }.items():
        require(file_sha256(REPO / relative) == spec["predecessors"][key], "PREDECESSOR_BINDING:" + key)
    material = load(REPO / spec["outputs"]["source_materialization"])
    require(material["status"] == "PASS_EVALUATION_MATERIALIZATION" and len(material["records"]) == 400, "MATERIALIZATION")
    require(material["records_digest"] == report["materialization_records_digest"] == digest(material["records"]), "MATERIALIZATION_DIGEST")
    for record in material["records"]:
        path = REPO / record["cache_path"]
        require(path.is_file() and path.stat().st_size == record["bytes"] and file_sha256(path) == record["sha256"], "IMAGE_BINDING:" + record["sample_id"])
    tasks = load(root / "task_manifest.json")
    require(tasks["task_count"] == 800 and tasks["positive_closure_evaluations"] == 76800, "TASK_SCOPE")
    require(tasks["negative_control_evaluations"] == 28800 and tasks["manifest_digest"] == digest(tasks["tasks"]), "TASK_MANIFEST")
    results, payloads = load_queue(root / "formal_queue.sqlite3", root / "formal_results", 800)
    require(payloads == tasks["tasks"] and tasks["queue_identity"]["manifest_digest"] == digest(payloads), "QUEUE_MANIFEST")
    expected_payload = {(model_id, row["split_order"]): row for model_id in ("vit_b16", "deit_s16") for row in material["records"]}
    positive_count = negative_count = raw_block_count = 0
    maximum_abs = maximum_relative = 0.0
    cut_counts = {name: 0 for name in CUT_ALIASES}
    branches = {"absolute_near_zero": 0, "relative_nonzero": 0}
    for task_id, result in results.items():
        frozen = expected_payload[(result["model_id"], result["split_order"])]
        require((result["sample_id"], result["class_id"], result["wnid"], result["image_sha256"]) == (frozen["sample_id"], frozen["class_id"], frozen["wnid"], frozen["sha256"]), "TASK_COHORT")
        require(result["stage"] == "ENDPOINT-E3" and result["claim_scope"] == "CORRECTNESS_ONLY_NOT_UTILITY", "RESULT_SCOPE")
        require(result["input_shape"] == [1, 3, 224, 224] and result["dtype"] == "torch.float32", "INPUT_CONTRACT")
        require(result["block_count"] == len(result["blocks"]) == 12 and [row["block"] for row in result["blocks"]] == list(range(12)), "BLOCK_COVERAGE")
        require(result["blocks_digest"] == result["numeric_digest"] == digest(result["blocks"]), "BLOCK_DIGEST")
        for block in result["blocks"]:
            signed = dict(block); observed = signed.pop("canonical_digest")
            require(observed == digest(signed), "BLOCK_CANONICAL_DIGEST")
            require(block["target_kind"] == "ground_truth_logit" and block["resolved_class_id"] == result["class_id"], "TARGET_IDENTITY")
            require(math.isfinite(block["target_value"]), "TARGET_FINITE")
            require(len(block["cuts"]) == 8 and {row["cut_id"] for row in block["cuts"]} == set(CUT_ALIASES), "CUT_COVERAGE")
            for summary in list(block["components"].values()) + list(block["reconstructions"].values()):
                require(summary["finite"] and summary["shape"] == [3, 224, 224] and summary["dtype"] == "float32", "SUMMARY_CONTRACT")
                require(all(math.isfinite(float(summary[key])) for key in ("signed_sum", "absolute_mass", "max_abs", "l2_norm")), "SUMMARY_FINITE")
            for cut in block["cuts"]:
                require(cut["pass"] and cut["branch"] in branches, "CUT_PASS")
                limit = 1e-5 + 1e-4 * block["reconstructions"][cut["reconstruction_id"]]["max_abs"]
                require(cut["max_abs"] <= limit, "CUT_ELEMENTWISE")
                if cut["reference_norm"] <= 1e-6:
                    require(cut["branch"] == "absolute_near_zero" and cut["max_abs"] <= 1e-5, "CUT_ABSOLUTE_BRANCH")
                else:
                    require(cut["branch"] == "relative_nonzero" and cut["relative_l2"] <= 1e-4, "CUT_RELATIVE_BRANCH")
                maximum_abs = max(maximum_abs, cut["max_abs"]); maximum_relative = max(maximum_relative, cut["relative_l2"]); cut_counts[cut["cut_id"]] += 1; branches[cut["branch"]] += 1; positive_count += 1
            require(len(block["negative_controls"]) == 3, "NEGATIVE_CONTROL_COVERAGE")
            for control in block["negative_controls"]:
                require(control["passed"] and control["affected_component_max_abs"] > 0, "NEGATIVE_CONTROL")
                require(control["checker_fixture_atol"] == control["checker_fixture_rtol"] == 0.0, "NEGATIVE_CHECKER_TOLERANCE")
                require(control["formal_positive_atol"] == 1e-5 and control["formal_positive_rtol"] == 1e-4, "POSITIVE_TOLERANCE_RECORD")
                negative_count += 1
        if result["split_order"] < 16:
            raw_block_count += validate_raw_task(result)
        else:
            require(result["artifacts"] == [], "RAW_RETENTION_SCOPE")
        identity = result["worker_identity"]
        require(identity["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and identity["startup_matmul_backward"] == "PASS", "CUDA_STARTUP")
        require(identity["deterministic_algorithms"] and not identity["cuda_matmul_allow_tf32"] and not identity["cudnn_allow_tf32"] and not identity["autocast"], "DETERMINISM")
    require(positive_count == report["positive_closure_evaluations"] == terminal["positive_closure_evaluation_count"] == 76800, "POSITIVE_COUNT")
    require(negative_count == report["negative_control_evaluations"] == terminal["negative_control_evaluation_count"] == 28800, "NEGATIVE_COUNT")
    require(raw_block_count == report["raw_component_map_tasks"] * 12 == 384, "RAW_COVERAGE")
    require(cut_counts == report["cut_counts"] and branches == report["closure_branch_counts"], "AGGREGATE_COUNTS")
    require(math.isclose(maximum_abs, report["maximum_closure_max_abs"], rel_tol=0, abs_tol=0), "MAX_ABS_AGGREGATE")
    require(math.isclose(maximum_relative, report["maximum_closure_relative_l2"], rel_tol=0, abs_tol=0), "MAX_REL_AGGREGATE")
    require(report["prohibited_methods"] == {
        "path_ig": "NOT_RUN_NOT_ALLOWED", "input_fidelity_stability": "NOT_RUN_NOT_ALLOWED",
        "activation_replacement": "NOT_RUN_NOT_ALLOWED", "finite_intervention": "NOT_RUN_NOT_ALLOWED",
        "utility_claim": "NOT_RUN_NOT_ALLOWED",
    }, "PROHIBITED_METHOD_SCOPE")
    require(report["formal_coverage"]["status"] == "PASS" and report["formal_coverage"]["completed_count"] == 800, "FORMAL_COVERAGE")
    require(report["gpu_hours_upper_observed"] <= 64 and report["formal_wall_seconds"] <= 72 * 3600, "TIME_CAP")
    require(report["peak_reserved_gpu_bytes_max"] <= 24 * 1024 ** 3 and report["peak_cpu_rss_bytes_max"] <= 32 * 1024 ** 3, "MEMORY_CAP")
    require(report["retained_bytes_before_report"] <= 128 * 1024 ** 3, "STORAGE_CAP")
    require(report["next_authorization"] == terminal["next_authorization"] == "ENDPOINT_E4_REQUIRES_EXPLICIT_USER_REQUEST", "NEXT_AUTHORIZATION")
    return {
        "status": PASS_STATUS, "evaluation_images": 400, "image_model_tasks": 800,
        "block_evaluations": 9600, "positive_closure_evaluations": positive_count,
        "negative_control_evaluations": negative_count, "raw_blocks_recomputed": raw_block_count,
        "maximum_closure_max_abs": maximum_abs,
        "maximum_closure_relative_l2": maximum_relative,
        "next_authorization": terminal["next_authorization"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    print(json.dumps(validate(parser.parse_args().root), sort_keys=True))
