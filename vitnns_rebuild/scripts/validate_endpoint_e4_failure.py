#!/usr/bin/env python3
"""Standard-library validation of the immutable ENDPOINT-E4 R1 failure root."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_e4_intervention_r1_20260918"
FAIL_STATUS = "ENDPOINT_E4_FAIL_LOCAL_DIRECTION_CONTRACT"
EXPECTED_FAILED = [f"deit_s16__{index:03d}" for index in range(16)]
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


def git_blob_file_sha256(commit: str, relative: str) -> str:
    data = subprocess.check_output(
        ["git", "show", f"{commit}:vitnns_rebuild/{relative}"], cwd=REPO,
    )
    return hashlib.sha256(data).hexdigest()


def validate(root: Path) -> dict:
    root = root.resolve()
    spec = load(REPO / "config/nns_vit_endpoint/endpoint_e4_intervention_run_spec.json")
    require(spec["stage"] == "ENDPOINT-E4" and spec["outputs"]["formal_root"] == root.relative_to(REPO).as_posix(), "SPEC_ROOT")
    manifest = load(root / "task_manifest.json"); startup = load(root / "startup_snapshot.json")
    require(manifest["task_count"] == 800 and manifest["manifest_digest"] == digest(manifest["tasks"]), "TASK_MANIFEST")
    require(startup["producer_commit"] == startup["tracking_commit"] == startup["remote_commit"], "STARTUP_GIT")
    producer = startup["producer_commit"]
    require(subprocess.run(["git", "merge-base", "--is-ancestor", producer, "HEAD"], cwd=REPO).returncode == 0, "PRODUCER_ANCESTRY")
    for relative in SOURCE_FILES:
        require(subprocess.run(["git", "cat-file", "-e", f"{producer}:vitnns_rebuild/{relative}"], cwd=REPO).returncode == 0, "SOURCE_NOT_PUBLISHED:" + relative)
    for key, relative in BINDINGS.items():
        observed = git_blob_file_sha256(producer, relative) if key == "unified_plan_sha256" else file_sha256(REPO / relative)
        require(observed == spec["predecessors"][key], "PREDECESSOR_BINDING:" + key)
    with sqlite3.connect(root / "formal_queue.sqlite3") as connection:
        tasks = connection.execute(
            "SELECT task_id,payload_json,payload_digest,seed,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id"
        ).fetchall()
        attempts = connection.execute("SELECT task_id,attempt,status FROM attempts ORDER BY task_id,attempt").fetchall()
        failures = connection.execute("SELECT key,value FROM metadata WHERE key LIKE 'failure:%' ORDER BY key").fetchall()
    require(len(tasks) == 800, "QUEUE_TASK_COUNT")
    queue_payloads = []
    states: dict[str, int] = {}
    completed_ids = []
    for task_id, payload_json, payload_digest, seed, state, attempt, result_path, result_digest in tasks:
        require(hashlib.sha256(payload_json.encode()).hexdigest() == payload_digest, "PAYLOAD_DIGEST")
        payload = json.loads(payload_json); queue_payloads.append({"task_id": task_id, "payload": payload})
        states[state] = states.get(state, 0) + 1
        if state == "completed":
            result = load(root / "formal_results" / result_path); signed = dict(result); observed = signed.pop("canonical_digest")
            require(digest(signed) == observed == result_digest, "RESULT_DIGEST")
            require(result["task_id"] == task_id and result["attempt"] == attempt and result["seed"] == int(seed), "RESULT_IDENTITY")
            require(not result["result"]["include_local"] and result["result"]["local_contract"] == {"performed": False, "components": [], "all_pass": None}, "COMPLETED_LOCAL_RESULT")
            completed_ids.append(task_id)
    require(queue_payloads == manifest["tasks"] and manifest["queue_identity"]["manifest_digest"] == digest(queue_payloads), "QUEUE_MANIFEST")
    require(states == {"claimed": 8, "completed": 73, "failed": 16, "pending": 703}, "QUEUE_STATE_COUNTS")
    failed_ids = sorted(row[0] for row in tasks if row[4] == "failed")
    require(failed_ids == EXPECTED_FAILED, "FAILED_TASK_SET")
    failed_attempts = [(task_id, attempt, status) for task_id, attempt, status in attempts if task_id in EXPECTED_FAILED]
    require(len(failed_attempts) == 32 and all(attempt in {1, 2} and status in {"RETRYABLE", "FAILED"} for _task, attempt, status in failed_attempts), "FAILED_ATTEMPTS")
    require(len(failures) == 32, "FAILURE_RECORD_COUNT")
    for key, value in failures:
        payload = json.loads(value)
        require(key.startswith("failure:deit_s16__0") and payload == {"error_type": "EndpointE4Error", "message": "LOCAL_DIRECTION_CONTRACT_FAILED"}, "FAILURE_CODE")
    require(all(task_id.startswith("deit_s16__") and int(task_id[-3:]) >= 16 for task_id in completed_ids), "EXCLUDED_COMPLETED_SCOPE")
    output = {
        "status": FAIL_STATUS, "stage": "ENDPOINT-E4",
        "failed_gate": "EP5-A_LOCAL_DIRECTION_CONTRACT", "failed_task_ids": failed_ids,
        "failed_task_count": 16, "failed_attempt_count": 32,
        "completed_tasks_excluded_from_scientific_interpretation": 73,
        "unfinished_tasks": states["claimed"] + states["pending"], "queue_state_counts": states,
        "scientific_utility_interpretation": "NOT_PERFORMED_DUE_LOCAL_GATE_FAILURE",
        "next_authorization": "NONE_PENDING_PROSPECTIVE_RECOVERY_AMENDMENT",
    }
    terminal_path = root / "terminal_receipt.json"
    if terminal_path.exists():
        terminal = load(terminal_path); report = load(root / "validation_report.json")
        require(terminal["status"] == report["status"] == FAIL_STATUS, "SEALED_STATUS")
        require(terminal["validation_digest"] == digest(report), "SEALED_VALIDATION_DIGEST")
        artifact = load(root / "artifact_manifest.json")
        require(terminal["manifest_digest"] == artifact["canonical_digest"] == digest(artifact["files"]), "SEALED_MANIFEST_DIGEST")
        for row in artifact["files"]:
            path = root / row["path"]
            require(path.is_file() and path.stat().st_size == row["bytes"] and file_sha256(path) == row["sha256"], "SEALED_ARTIFACT_HASH")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    print(json.dumps(validate(parser.parse_args().root), sort_keys=True))
