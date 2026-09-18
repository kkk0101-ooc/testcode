#!/usr/bin/env python3
"""Independently validate the immutable ENDPOINT-E4 R2 recovery failure."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_e4_intervention_r2_20260918"
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_e4_intervention_recovery_r2.json"
PASS_STATUS = "ENDPOINT_E4_R2_PASS_EXECUTION_ANALYTIC_JVP_AND_FINITE_REPLACEMENT"
FAIL_STATUS = "ENDPOINT_E4_R2_FAIL_ANALYTIC_JVP_CONTRACT"
FAILED_TASK = "vit_b16__009"
SOURCE_FILES = [
    "config/nns_vit_endpoint/endpoint_e4_intervention_recovery_r2.json",
    "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE4恢复R2_20260918.md",
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
    "endpoint_e4_r1_artifact_manifest_sha256": "reports/nns_vit_endpoint_e4_intervention_r1_20260918/artifact_manifest.json",
    "endpoint_e4_r1_coverage_sha256": "reports/nns_vit_endpoint_e4_intervention_r1_20260918/formal_coverage_audit.json",
    "endpoint_e4_r1_terminal_sha256": "reports/nns_vit_endpoint_e4_intervention_r1_20260918/terminal_receipt.json",
    "endpoint_e4_r1_validation_sha256": "reports/nns_vit_endpoint_e4_intervention_r1_20260918/validation_report.json",
    "intervention_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/intervention_registry.json",
    "materialization_receipt_sha256": "reports/nns_vit_endpoint_e2_atlas_r1_20260918/materialization_receipt.json",
    "model_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
    "statistical_analysis_plan_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json",
    "target_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json",
    "unified_plan_sha256": "docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md",
}
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_e4_intervention import MAIN_FAMILIES, validate_recovery_run_spec  # noqa: E402


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def git_blob_sha(commit: str, relative: str) -> str:
    data = subprocess.check_output(["git", "show", f"{commit}:vitnns_rebuild/{relative}"], cwd=REPO)
    return hashlib.sha256(data).hexdigest()


def validate(root: Path) -> dict:
    root = root.resolve(); spec = load(SPEC_PATH); validate_recovery_run_spec(spec)
    require(spec["outputs"]["formal_root"] == root.relative_to(REPO).as_posix(), "SPEC_ROOT")
    manifest = load(root / "task_manifest.json"); startup = load(root / "startup_snapshot.json")
    coverage = load(root / "formal_coverage_audit.json")
    require(manifest["task_count"] == 800 and manifest["manifest_digest"] == digest(manifest["tasks"]), "TASK_MANIFEST")
    require(startup["producer_commit"] == startup["tracking_commit"] == startup["remote_commit"], "STARTUP_GIT")
    producer = startup["producer_commit"]
    require(subprocess.run(["git", "merge-base", "--is-ancestor", producer, "HEAD"], cwd=REPO).returncode == 0, "PRODUCER_ANCESTRY")
    for relative in SOURCE_FILES:
        require(subprocess.run(["git", "cat-file", "-e", f"{producer}:vitnns_rebuild/{relative}"], cwd=REPO).returncode == 0, "SOURCE_NOT_PUBLISHED:" + relative)
    for key, relative in BINDINGS.items():
        observed = git_blob_sha(producer, relative) if key == "unified_plan_sha256" else file_sha256(REPO / relative)
        require(observed == spec["predecessors"][key], "PREDECESSOR_BINDING:" + key)
    r1 = load(REPO / spec["recovery"]["source_root"] / "terminal_receipt.json")
    require(r1["status"] == spec["recovery"]["source_status"] == "ENDPOINT_E4_FAIL_LOCAL_DIRECTION_CONTRACT", "R1_STATUS")
    with sqlite3.connect(root / "formal_queue.sqlite3") as connection:
        tasks = connection.execute(
            "SELECT task_id,payload_json,payload_digest,seed,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id"
        ).fetchall()
        attempts = connection.execute("SELECT task_id,attempt,status FROM attempts ORDER BY task_id,attempt").fetchall()
        failures = connection.execute("SELECT key,value FROM metadata WHERE key LIKE 'failure:%' ORDER BY key").fetchall()
    require(len(tasks) == 800, "QUEUE_TASK_COUNT")
    states: dict[str, int] = {}; queue_payloads = []; completed_local = local_checks = main_count = secondary_count = zero_count = 0
    for task_id, payload_json, payload_digest, seed, state, attempt, result_path, result_digest in tasks:
        require(hashlib.sha256(payload_json.encode()).hexdigest() == payload_digest, "PAYLOAD_DIGEST")
        payload = json.loads(payload_json); queue_payloads.append({"task_id": task_id, "payload": payload})
        require(payload["stage"] == "ENDPOINT-E4-R2-RECOVERY" and payload["local_contract_mode"] == "analytic_jvp_identity_r2", "TASK_RECOVERY_SCOPE")
        states[state] = states.get(state, 0) + 1
        if state != "completed":
            require(task_id == FAILED_TASK and state == "failed", "UNEXPECTED_NONCOMPLETION")
            continue
        document = load(root / "formal_results" / result_path); signed = dict(document); observed = signed.pop("canonical_digest")
        require(digest(signed) == observed == result_digest, "RESULT_DIGEST")
        require(document["task_id"] == task_id and document["attempt"] == attempt and document["seed"] == int(seed), "RESULT_IDENTITY")
        result = document["result"]
        require(result["stage"] == "ENDPOINT-E4-R2-RECOVERY", "RESULT_STAGE")
        expected_main = 192 if result["model_id"] == "vit_b16" else 120
        expected_all = expected_main + (36 if result["split_order"] < 64 else 0)
        require(result["main_component_count"] == expected_main and result["all_component_count"] == expected_all, "COMPONENT_SCOPE")
        local = result["local_contract"]
        if result["include_local"]:
            completed_local += 1
            require(local["performed"] and local["all_pass"] and local["method"] == "analytic_jvp_identity_r2", "LOCAL_METHOD")
            require(len(local["components"]) == expected_all, "LOCAL_COUNT")
            for component in local["components"]:
                require(component["pass"] and "curve" not in component, "LOCAL_PASS")
                require(component["directional_prediction_absolute_error"] <= component["threshold"], "LOCAL_THRESHOLD")
                local_checks += 1
        else:
            require(local == {"performed": False, "components": [], "all_pass": None}, "LOCAL_OUTSIDE_SUBSET")
        for reference in ("normalized_zero", "blur_rgb"):
            components = result["finite_utility"][reference]["components"]
            require(len(components) == expected_all and len({row["component_id"] for row in components}) == expected_all, "FINITE_COMPONENT_COUNT")
            require(all(all(math.isfinite(float(row[key])) for key in ("a", "m", "p_ref", "effect", "absolute_effect")) for row in components), "FINITE_VALUES")
            main_count += sum(row["family"] in MAIN_FAMILIES for row in components)
            secondary_count += sum(row["family"] not in MAIN_FAMILIES for row in components)
        zero_count += len(result["zeroing_secondary"]["components"])
    require(queue_payloads == manifest["tasks"] and manifest["queue_identity"]["manifest_digest"] == digest(queue_payloads), "QUEUE_MANIFEST")
    require(states == {"completed": 799, "failed": 1}, "QUEUE_STATE_COUNTS")
    require([(row[0], row[1], row[2]) for row in attempts if row[0] == FAILED_TASK] == [(FAILED_TASK, 1, "RETRYABLE"), (FAILED_TASK, 2, "FAILED")], "FAILED_ATTEMPTS")
    require(len(failures) == 2 and all(json.loads(value) == {"error_type": "EndpointE4Error", "message": "LOCAL_DIRECTION_CONTRACT_FAILED"} for _, value in failures), "FAILURE_RECORDS")
    require(coverage["status"] == "FAIL" and coverage["completed_count"] == 799, "COVERAGE_STATUS")
    require(coverage["missing_task_ids"] == [FAILED_TASK] and coverage["duplicate_task_ids"] == [] and coverage["invalid"] == [], "COVERAGE_DETAIL")
    require((completed_local, local_checks, main_count, secondary_count, zero_count) == (31, 5916, 249216, 9144, 16728), "PARTIAL_COUNTS")
    output = {
        "status": FAIL_STATUS, "stage": "ENDPOINT-E4-R2-RECOVERY",
        "failed_gate": "ANALYTIC_JVP_IDENTITY", "failed_task_ids": [FAILED_TASK],
        "failed_task_count": 1, "failed_attempt_count": 2,
        "completed_tasks_excluded_from_scientific_interpretation": 799,
        "completed_local_tasks": completed_local, "completed_local_component_checks": local_checks,
        "partial_main_component_reference_evaluations": main_count,
        "partial_secondary_component_reference_evaluations": secondary_count,
        "partial_zeroing_evaluations": zero_count, "queue_state_counts": states,
        "r1_failure_preserved": True, "r1_partial_results_inherited": False,
        "scientific_utility_interpretation": "NOT_PERFORMED_DUE_INCOMPLETE_COVERAGE",
        "next_authorization": "NONE_PENDING_PROSPECTIVE_E4_RECOVERY_AMENDMENT",
    }
    terminal_path = root / "terminal_receipt.json"
    if terminal_path.exists():
        terminal = load(terminal_path); report = load(root / "validation_report.json")
        require(terminal["status"] == report["status"] == FAIL_STATUS, "SEALED_STATUS")
        require(terminal["validation_digest"] == digest(report), "SEALED_VALIDATION")
        artifact = load(root / "artifact_manifest.json")
        require(terminal["manifest_digest"] == artifact["canonical_digest"] == digest(artifact["files"]), "SEALED_MANIFEST")
        for row in artifact["files"]:
            path = root / row["path"]
            require(path.is_file() and path.stat().st_size == row["bytes"] and file_sha256(path) == row["sha256"], "SEALED_ARTIFACT_HASH")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    print(json.dumps(validate(parser.parse_args().root), sort_keys=True))
