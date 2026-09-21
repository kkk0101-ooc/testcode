#!/usr/bin/env python3
"""Independently audit Step 5.1 from the frozen prompt contract to final artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RUN = ROOT / "step5_1_r7_20260921"
OUTPUT = ROOT / "step5_1_completion_r2_20260921"
EXPECTED_CLAIMS = {
    "CONTROLLED_CURVATURE_LOCALIZATION": "SUPPORTED",
    "NATURAL_NONLINEARITY_LOCALIZATION": "NOT_RUN_A1_GATE_CLOSED",
    "ATTRIBUTION_ERROR_LOCALIZATION": "SUPPORTED",
    "INCREMENTAL_OVER_SIMPLE_BASELINES": "NOT_SUPPORTED",
    "NUMERICAL_VALIDITY_AND_COST": "SUPPORTED",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    claimed = value.pop("canonical_digest", None)
    if claimed is None or digest(value) != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed
    return value


def signed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["canonical_digest"] = digest(result)
    return result


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError(code)


def same_number(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def conservative_rank(scores: Sequence[float], true_index: int, top_k: int) -> dict[str, Any]:
    true = float(scores[true_index])
    greater = sum(float(value) > true for value in scores)
    equal = sum(float(value) == true for value in scores)
    rank = greater + equal
    return {
        "rank": rank, "top1": rank == 1, "top_k": rank <= top_k,
        "reciprocal_rank": 1.0 / rank,
        "percentile_rank": 1.0 if len(scores) == 1 else 1.0 - (rank - 1) / (len(scores) - 1),
        "true_score": true, "maximum_score": max(float(value) for value in scores),
        "tie_count_at_true": equal,
    }


def compare_metrics(observed: Mapping[str, Any], expected: Mapping[str, Any], code: str) -> None:
    require(set(observed) == set(expected), code + "_KEYS")
    for key, value in expected.items():
        if isinstance(value, float):
            require(same_number(observed[key], value), code + "_" + key)
        else:
            require(observed[key] == value, code + "_" + key)


def aggregate(rows: Sequence[dict[str, Any]], model: str, role: str,
              strength: float, method: str) -> dict[str, Any]:
    selected = [condition["rank_metrics"][method]
                for row in rows if row["model_id"] == model
                for condition in row["conditions"]
                if condition["role"] == role and condition["strength"] == strength]
    require(len(selected) == 8, "AGGREGATE_COVERAGE")
    return {
        "n": 8,
        "top1_rate": math.fsum(float(row["top1"]) for row in selected) / 8,
        "top_k_rate": math.fsum(float(row["top_k"]) for row in selected) / 8,
        "mean_rank": math.fsum(float(row["rank"]) for row in selected) / 8,
        "mean_reciprocal_rank": math.fsum(float(row["reciprocal_rank"]) for row in selected) / 8,
        "mean_percentile_rank": math.fsum(float(row["percentile_rank"]) for row in selected) / 8,
    }


def run_tests() -> int:
    command = [
        sys.executable, "-m", "pytest", "-q",
        "vit_pcia_pefa/tests/test_step4_vit_adapter.py",
        "vit_pcia_pefa/tests/test_step5_pilot.py",
        "vit_pcia_pefa/tests/test_step5_1_localization.py",
        "vit_pcia_pefa/tests/test_step5_1_queue.py",
    ]
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    match = re.search(r"(\d+) passed", result.stdout)
    require(result.returncode == 0 and match is not None, "REGRESSION_TESTS")
    return int(match.group(1))


def build_audit() -> dict[str, Any]:
    protocol = verify_signed(RUN / "protocol.json")
    manifest = verify_signed(RUN / "task_manifest.json")
    binding = verify_signed(RUN / "source_binding.json")
    freeze = verify_signed(RUN / "freeze_receipt.json")
    inheritance = verify_signed(RUN / "inheritance_receipt.json")
    a0_decision = verify_signed(RUN / "a0_decision.json")
    launch = verify_signed(RUN / "a1_launch_terminal.json")
    analysis = verify_signed(RUN / "a1_analysis.json")
    c1 = verify_signed(RUN / "c1_result.json")
    claims = verify_signed(RUN / "claim_decisions.json")
    resources = verify_signed(RUN / "resource_ledger.json")
    evidence = verify_signed(RUN / "evidence_index.json")
    terminal = verify_signed(RUN / "terminal_receipt.json")
    prior_resources = verify_signed(ROOT / "step5_completion_20260921/resource_ledger.json")

    require(protocol["run_id"] == RUN.name and protocol["stage"] == "ENDPOINT_PCIA_PEFA_STEP5_1",
            "PROTOCOL_IDENTITY")
    require(protocol["status"] == "REPORTING_ONLY_RECOVERY_FROZEN", "R7_RECOVERY_STATUS")
    require(protocol["prior_results_immutable"]["step5"] == "NOT_SUPPORTED_EMPIRICAL_VALUE",
            "PRIOR_STEP5_DRIFT")
    require(protocol["evaluation_access"] is False and protocol["step6_authorized"] is False,
            "PROTOCOL_BOUNDARY")
    require(protocol["wrapper"]["use_port"] == "attn.proj.input_slot_0_head_slice",
            "USE_PORT_DRIFT")
    require(protocol["a1"]["methods"] == [
        "pefa_R_delta", "pcia_delta", "pcia_level", "endpoint_delta",
        "endpoint_level", "gradient_delta", "gradient_level",
        "activation_delta", "activation_level", "random"], "BASELINE_SET_DRIFT")
    require(protocol["integration"]["real_nodes"] == 512, "QUADRATURE_DRIFT")
    require(freeze["protocol_digest"] == protocol["canonical_digest"]
            and freeze["task_manifest_digest"] == manifest["canonical_digest"]
            and freeze["source_binding_digest"] == binding["canonical_digest"],
            "FREEZE_LINKAGE")
    require(not any((freeze["a0_started"], freeze["a1_started"], freeze["b1_started"],
                     freeze["c1_started"], freeze["evaluation_access"], freeze["step6_started"])),
            "FREEZE_PREACCESS_STATE")
    require(binding["outcomes_known_before_reporting_freeze"] is True
            and binding["outcome_reads_before_freeze"] == 16
            and binding["recovery"]["reporting_only"] is True, "RECOVERY_DISCLOSURE")

    source_checks = []
    for row in (*binding["files"], *binding["checkpoints"], *binding["images"]):
        path = REPO / row["path"]
        if row["path"] == "vit_pcia_pefa/执行规划.md":
            frozen = subprocess.check_output([
                "git", "show", f"{binding['producer_commit']}:{row['path']}"], cwd=REPO)
            require(len(frozen) == row["bytes"] and hashlib.sha256(frozen).hexdigest() == row["sha256"],
                    "FROZEN_PLAN_NOT_RECOVERABLE")
            source_checks.append({"path": row["path"], "status": "PASS_FROZEN_VERSION_RECOVERABLE"})
        else:
            require(path.is_file() and path.stat().st_size == row["bytes"]
                    and sha256(path) == row["sha256"], "SOURCE_BINDING:" + row["path"] )
            source_checks.append({"path": row["path"], "status": "PASS_CURRENT_BYTES"})
    plan = ROOT / "执行规划.md"
    plan_text = plan.read_text()
    for required in (
        "step5_1_r7_20260921", "NOT_SUPPORTED_INCREMENTAL_OVER_SIMPLE_BASELINES",
        "step5_1_completion_r2_20260921",
        "CONTROLLED_CURVATURE_LOCALIZATION = SUPPORTED",
        "NATURAL_NONLINEARITY_LOCALIZATION = NOT_RUN_A1_GATE_CLOSED",
        "INCREMENTAL_OVER_SIMPLE_BASELINES = NOT_SUPPORTED",
        "b1_authorized_by_gate=false", "14.182787541702375",
    ):
        require(required in plan_text, "MASTER_PLAN_MISSING:" + required)
    require("PROPOSED_NOT_FROZEN_NOT_STARTED" not in plan_text, "STALE_MASTER_PLAN_STATUS")

    tasks = manifest["a1_tasks"]
    require(len(manifest["a0_tasks"]) == 2 and len(tasks) == 16, "MANIFEST_COUNTS")
    require(sorted({int(row["split_order"]) for row in tasks}) == list(range(8, 16))
            and {row["split"] for row in tasks} == {"calibration"}, "A1_SELECTION")
    task_by_id = {row["task_id"]: row for row in tasks}
    require(len(task_by_id) == 16, "MANIFEST_DUPLICATE_TASK")
    normalized = [{"task_id": task_id,
                   "payload": {key: value for key, value in task_by_id[task_id].items()
                               if key != "task_id"}} for task_id in sorted(task_by_id)]

    documents = []
    queue_path = RUN / "a1_queue.sqlite3"
    with sqlite3.connect(queue_path) as connection:
        connection.row_factory = sqlite3.Row
        metadata = dict(connection.execute("SELECT key,value FROM metadata").fetchall())
        queue_rows = connection.execute("SELECT * FROM tasks ORDER BY task_id").fetchall()
        attempts = connection.execute("SELECT * FROM attempts ORDER BY task_id,attempt").fetchall()
    require(metadata["seed_namespace"] == "ENDPOINT_PCIA_PEFA|step5_1"
            and metadata["manifest_digest"] == digest(normalized), "QUEUE_METADATA")
    require(len(queue_rows) == 16 and len(attempts) == 16, "QUEUE_COVERAGE")
    require(all(row["status"] == "COMPLETED" and row["attempt"] == 1 for row in attempts),
            "QUEUE_ATTEMPTS")
    checkpoint_by_model = {row["model_id"]: row["sha256"] for row in binding["checkpoints"]}
    for row in queue_rows:
        task_id = row["task_id"]
        task = task_by_id[task_id]
        payload = json.loads(row["payload_json"])
        expected_seed = int.from_bytes(hashlib.sha256(
            f"ENDPOINT_PCIA_PEFA|step5_1|{task_id}".encode()).digest()[:8], "big")
        require(payload == normalized[[item["task_id"] for item in normalized].index(task_id)]["payload"],
                "QUEUE_PAYLOAD:" + task_id)
        require(row["payload_digest"] == hashlib.sha256(row["payload_json"].encode()).hexdigest()
                and int(row["seed"]) == expected_seed and row["state"] == "completed"
                and row["attempt"] == 1, "QUEUE_ROW:" + task_id)
        result_path = RUN / "a1_results" / row["result_path"]
        document = verify_signed(result_path)
        require(document["canonical_digest"] == row["result_digest"]
                and document["task_id"] == task_id and document["attempt"] == 1
                and document["seed"] == expected_seed, "RESULT_ENVELOPE:" + task_id)
        result = document["result"]
        require(result["status"] == "PASS_A1_TASK" and result["model_id"] == task["model_id"]
                and result["split"] == "calibration" and result["split_order"] == task["split_order"],
                "RESULT_IDENTITY:" + task_id)
        require(result["reproducibility"]["task_seed"] == expected_seed
                and result["reproducibility"]["input_sha256"] == task["image_sha256"]
                and result["reproducibility"]["checkpoint_sha256"] == checkpoint_by_model[task["model_id"]],
                "RESULT_REPRODUCIBILITY:" + task_id)
        worker = result["worker_identity"]
        require(worker["sys_executable_realpath"] == binding["sys_executable_realpath"]
                and worker["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
                and worker["deterministic_algorithms"] is True
                and worker["tf32"] is False and worker["autocast"] is False,
                "WORKER_IDENTITY:" + task_id)
        require(result["clean_closure"]["pass"] is True, "CLEAN_CLOSURE:" + task_id)
        require({(row["role"], float(row["strength"])) for row in result["conditions"]}
                == {(role, strength) for role in ("primary", "decoy")
                    for strength in (0.5, 1.0)}, "CONDITION_COVERAGE:" + task_id)
        for condition in result["conditions"]:
            require(condition["clean_parity"]["pass"] and condition["closure"]["pass"]
                    and condition["difference_direction"]["pass"], "CONDITION_VALIDITY:" + task_id)
            for method, scores in condition["methods"].items():
                recomputed = conservative_rank(scores, condition["modified_head"], result["top_k"])
                compare_metrics(condition["rank_metrics"][method], recomputed,
                                "RANK_METRIC:" + task_id + ":" + method)
            for metric in ("rank", "top1", "top_k", "percentile_rank"):
                require(condition["rank_metrics"]["pefa_R_delta"][metric]
                        == condition["rank_metrics"]["pcia_delta"][metric],
                        "PEFA_PCIA_RANK_NOT_TIED:" + task_id + ":" + metric)
        require({(row["mode"], float(row["strength"])) for row in result["controls"]}
                == {("bounded_quadratic", 0.0), ("linear_tangent", 1.0)},
                "CONTROL_COVERAGE:" + task_id)
        require(all(control["clean_parity"]["pass"] and control["closure"]["pass"]
                    and control["maximum_change_score"] <= protocol["numeric"]["real"]["atol"]
                    for control in result["controls"]), "CONTROL_VALIDITY:" + task_id)
        require(row["result_digest"] == document["canonical_digest"], "QUEUE_RESULT_DIGEST:" + task_id)
        documents.append(document)

    rows = [document["result"] for document in documents]
    recomputed_summaries = {}
    controlled_pass = True
    incremental_pass = True
    controlled_gate = protocol["a1"]["controlled_gate_each_model_each_strength"]
    incremental_gate = protocol["a1"]["incremental_gate_each_model_each_strength"]
    for model in protocol["models"]:
        for role in ("primary", "decoy"):
            for strength in protocol["wrapper"]["a1_strengths"]:
                key = f"{model}|{role}|{strength}"
                recomputed_summaries[key] = {method: aggregate(rows, model, role, strength, method)
                                             for method in protocol["a1"]["methods"]}
                for method, summary in recomputed_summaries[key].items():
                    compare_metrics(analysis["summaries"][key][method], summary,
                                    "SUMMARY:" + key + ":" + method)
                pefa = recomputed_summaries[key]["pefa_R_delta"]
                controlled_pass &= (pefa["top1_rate"] >= controlled_gate["pefa_top1_min"]
                                    and pefa["top_k_rate"] >= controlled_gate["pefa_top_k_min"]
                                    and pefa["mean_percentile_rank"] >= controlled_gate["pefa_mean_percentile_min"])
                for method, comparator in recomputed_summaries[key].items():
                    if method != "pefa_R_delta":
                        incremental_pass &= (
                            pefa["mean_percentile_rank"] - comparator["mean_percentile_rank"]
                            >= incremental_gate["mean_percentile_advantage_over_every_baseline"]
                            and pefa["top1_rate"] - comparator["top1_rate"]
                            >= incremental_gate["top1_rate_advantage_over_every_baseline"])
    require(controlled_pass is True and incremental_pass is False, "A1_GATE_RECOMPUTATION")
    require(all(not row["pass"] and row["advantages"]["pcia_delta"]
                == {"mean_percentile": 0.0, "top1_rate": 0.0}
                for row in analysis["incremental_checks"]), "PCIA_TIE_GATE")
    require(analysis["status"] == "VALID" and analysis["controls_valid"] is True
            and analysis["coverage"]["status"] == "PASS", "ANALYSIS_VALIDITY")

    for path in (RUN / "a0_fixture_result.json", RUN / "a0/deit_s16.json",
                 RUN / "a0/vit_b16.json"):
        result = verify_signed(path)
        require(result["status"].startswith("PASS_A0_") and result["dose_nondecreasing"] is True
                and all(row["pefa_R_rank"]["top1"] for row in result["rows"])
                and all(control["maximum_change_score"] == 0.0 for control in result["controls"]),
                "A0_RESULT:" + path.name)
    require(a0_decision["status"] == "PASS_A0_NUMERICAL_FEASIBILITY"
            and a0_decision["a1_authorized_by_gate"] is True, "A0_GATE")

    require(c1["run_id"] == "step5_1_r3_20260921" and c1["status"] == "SUPPORTED"
            and c1["claim_id"] == "ATTRIBUTION_ERROR_LOCALIZATION"
            and len(c1["rows"]) == 2, "C1_DECISION")
    for row in c1["rows"]:
        audit = row["contract_detection"]
        require(row["held_out_mutation"] == "consumer_head_slice_rotation"
                and audit["pass"] is False and audit["block_index"] == 5
                and audit["cut_id"] == "HEADS_WITH_BYPASS_FULL"
                and audit["category"] == "head"
                and all(error["code"] == "FAIL_PORT_ID_SEMANTIC_MISMATCH"
                        for error in audit["errors"]), "C1_LOCALIZATION")
    require(sha256(RUN / "c1_result.json") == sha256(ROOT / "step5_1_r3_20260921/c1_result.json"),
            "C1_PROVENANCE")

    require(inheritance["source_root"] == "step5_1_r6_20260921"
            and inheritance["source_terminal_digest"]
            == verify_signed(ROOT / "step5_1_r6_20260921/terminal_receipt.json")["canonical_digest"]
            and inheritance["accepted_a1_results"] == 16
            and inheritance["reporting_only"] is True, "INHERITANCE_SUMMARY")
    for row in inheritance["artifacts"]:
        target = RUN / row["path"]
        source = ROOT / inheritance["source_root"] / row["path"]
        require(target.stat().st_size == row["bytes"] and sha256(target) == row["sha256"]
                and sha256(source) == row["sha256"], "INHERITANCE_ARTIFACT:" + row["path"] )

    evidence_raw = [document["canonical_digest"] for document in documents]
    require(evidence["raw_result_count"] == 16
            and evidence["raw_result_digests"] == evidence_raw, "EVIDENCE_RAW_RESULTS")
    for row in evidence["files"]:
        path = REPO / row["path"]
        require(path.is_file() and path.stat().st_size == row["bytes"]
                and sha256(path) == row["sha256"], "EVIDENCE_FILE:" + row["path"] )

    worker_seconds = math.fsum(row["resources"]["wall_seconds"] for row in rows)
    step5_1_hours = math.fsum((0.003181305090905258, 0.09303328139810926,
                              0.25000801771111114, worker_seconds / 3600.0))
    cumulative_upper = math.fsum((
        prior_resources["cumulative_step5_h20_hours_upper_bound"], step5_1_hours))
    require(same_number(worker_seconds, resources["r5_a1_worker_seconds"])
            and same_number(step5_1_hours, resources["step5_1_h20_hours"])
            and same_number(cumulative_upper, resources["cumulative_step5_h20_hours_upper_bound"])
            and cumulative_upper <= protocol["resources"]["cumulative_step5_h20_hours_cap"],
            "RESOURCE_RECOMPUTATION")
    peak_gpu = max(row["resources"]["peak_reserved_gpu_bytes"] for row in rows)
    peak_rss = max(row["resources"]["peak_cpu_rss_bytes"] for row in rows)
    max_task_seconds = max(row["resources"]["wall_seconds"] for row in rows)
    retained_bytes = sum(path.stat().st_size for root in sorted(ROOT.glob("step5_1_r*_20260921"))
                         for path in root.rglob("*") if path.is_file())
    require(peak_gpu <= protocol["resources"]["gpu_memory_cap_bytes"]
            and peak_rss <= protocol["resources"]["cpu_rss_cap_bytes"]
            and max_task_seconds <= protocol["resources"]["task_timeout_seconds"]
            and retained_bytes <= protocol["resources"]["retained_bytes_cap"], "RESOURCE_CAPS")

    decision_map = {row["claim_id"]: row["decision"] for row in claims["decisions"]}
    require(decision_map == EXPECTED_CLAIMS, "CLAIM_DECISIONS")
    require(claims["status"] == "NOT_SUPPORTED_INCREMENTAL_OVER_SIMPLE_BASELINES"
            and claims["mainline_position"] == "ENDPOINT_EXTENSION_ONLY"
            and claims["prior_step5_decision_unchanged"] == "NOT_SUPPORTED_EMPIRICAL_VALUE"
            and claims["b1_authorized_by_gate"] is False
            and claims["confirmation_experiment_authorized"] is False
            and claims["step6_authorized"] is False, "FINAL_DECISION_BOUNDARY")
    require(terminal["status"] == "NOT_SUPPORTED_INCREMENTAL_OVER_SIMPLE_BASELINES"
            and terminal["claim_decisions_digest"] == claims["canonical_digest"]
            and terminal["analysis_digest"] == analysis["canonical_digest"]
            and terminal["resource_ledger_digest"] == resources["canonical_digest"]
            and terminal["evidence_index_digest"] == evidence["canonical_digest"]
            and terminal["b1_started"] is False and terminal["evaluation_access"] is False
            and terminal["step6_started"] is False, "TERMINAL_LINKAGE_OR_BOUNDARY")
    require(launch["status"] == "PASS_A1_TASK_EXECUTION"
            and launch["coverage"]["completed_count"] == 16
            and len(launch["selected_gpus"]) <= protocol["resources"]["maximum_gpus"],
            "A1_LAUNCH")
    b1_artifacts = [path for path in RUN.rglob("*")
                    if any(part == "b1" or part.startswith("b1_")
                           for part in path.relative_to(RUN).parts)]
    require(not b1_artifacts, "B1_ARTIFACT_PRESENT")

    failure_r1 = verify_signed(ROOT / "step5_1_r1_20260921/a0_launch_terminal.json")
    failure_r4 = verify_signed(ROOT / "step5_1_r4_20260921/a1_failure_receipt.json")
    require(failure_r1["status"] == "FAIL_A0_TASK_EXECUTION"
            and failure_r4["status"] == "TERMINAL_INFRASTRUCTURE_FAIL"
            and failure_r4["accepted_result_count"] == 0, "FAILED_ROOT_PRESERVATION")
    for amendment in range(2, 8):
        require((ROOT / f"step5_1_r{amendment}_recovery_amendment.md").is_file(),
                f"RECOVERY_AMENDMENT_R{amendment}")

    localization_source = (ROOT / "src/step5_1_localization.py").read_text()
    adapter_source = (ROOT / "src/vit_adapter.py").read_text()
    require("Compute source-summed R only for one frozen complete head cut" in localization_source
            and "projection_consumer_input" in adapter_source, "SOURCE_SUMMED_REAL_USE_PORT")
    test_count = run_tests()
    require(test_count == 23, "TEST_COUNT")

    checklist = [
        {"requirement": "5.1.1 independent prospective scope and immutable prior Step-5 result",
         "status": "PASS", "evidence": ["protocol.json", "source_binding.json", "执行规划.md"]},
        {"requirement": "5.1.2 source-summed R_e only at a real consumer use-port",
         "status": "PASS", "evidence": ["src/step5_1_localization.py", "src/vit_adapter.py"]},
        {"requirement": "5.1.3 A0/A1 controlled nonlinearity localization and simple baselines",
         "status": "PASS_CONTROLLED_NOT_INCREMENTAL",
         "evidence": ["a0_decision.json", "a1_analysis.json", "16 raw A1 results"]},
        {"requirement": "5.1.4 natural nonlinearity B1 only after all A1 gates",
         "status": "PASS_NOT_RUN_GATE_CLOSED",
         "evidence": ["claim_decisions.json", "terminal_receipt.json"]},
        {"requirement": "5.1.5 independent held-out attribution-error localization",
         "status": "PASS_SUPPORTED", "evidence": ["c1_result.json with original r3 provenance"]},
        {"requirement": "5.1.6 ordered gates, five claim decisions, and stop rules",
         "status": "PASS", "evidence": ["claim_decisions.json", "terminal_receipt.json"]},
        {"requirement": "5.1.7 claim boundary and unchanged ENDPOINT_EXTENSION_ONLY mainline",
         "status": "PASS", "evidence": ["执行规划.md", "claim_decisions.json"]},
        {"requirement": "task identity, first-attempt coverage, deterministic seed and byte-level result integrity",
         "status": "PASS", "evidence": ["task_manifest.json", "a1_queue.sqlite3", "evidence_index.json"]},
        {"requirement": "numeric, GPU, RSS, timeout, retained-byte and cumulative H20-hour caps",
         "status": "PASS", "evidence": ["resource_ledger.json", "prior Step-5 resource ledger"]},
        {"requirement": "no evaluation access, B1 start, confirmation experiment or Step 6",
         "status": "PASS", "evidence": ["protocol.json", "source_binding.json", "terminal_receipt.json"]},
        {"requirement": "failed roots preserved and recovered prospectively",
         "status": "PASS", "evidence": ["r1/r4 failure receipts", "r2-r7 recovery amendments"]},
        {"requirement": "regression and terminal validation", "status": "PASS",
         "evidence": [f"{test_count} pytest tests passed", "PASS_STEP5_1_TERMINAL_COVERAGE before living-plan update"]},
    ]
    return {
        "schema_version": 1, "run_id": "step5_1_completion_r2_20260921",
        "stage": "ENDPOINT_PCIA_PEFA_STEP5_1",
        "status": "PASS_STEP5_1_PROMPT_TO_ARTIFACT_AUDIT",
        "audited_root": RUN.name, "audited_terminal_digest": terminal["canonical_digest"],
        "protocol_digest": protocol["canonical_digest"],
        "prompt_to_artifact_checklist": checklist,
        "recomputed": {
            "a0_tasks": 2, "a1_tasks": 16, "accepted_first_attempt_tasks": 16,
            "controlled_groups": 8, "pefa_top1_rate_all_groups": 1.0,
            "pefa_top_k_rate_all_groups": 1.0, "pefa_mean_percentile_all_groups": 1.0,
            "pcia_delta_top1_rate_all_groups": 1.0,
            "pcia_delta_mean_percentile_all_groups": 1.0,
            "pefa_advantage_over_pcia_delta": {"top1_rate": 0.0, "mean_percentile": 0.0},
            "a1_worker_seconds": worker_seconds, "step5_1_h20_hours": step5_1_hours,
            "prior_step5_h20_hours_upper_bound": prior_resources["cumulative_step5_h20_hours_upper_bound"],
            "cumulative_step5_h20_hours_upper_bound": cumulative_upper,
            "peak_a1_reserved_gpu_bytes": peak_gpu, "peak_a1_cpu_rss_bytes": peak_rss,
            "max_a1_task_seconds": max_task_seconds, "retained_step5_1_bytes": retained_bytes,
        },
        "claims": EXPECTED_CLAIMS,
        "provenance_notes": {
            "a0_internal_run_id": "step5_1_r2_20260921",
            "c1_internal_run_id": "step5_1_r3_20260921",
            "a1_launch_internal_run_id": "step5_1_r5_20260921",
            "meaning": "byte-identical inherited artifacts retain their producing run_id",
            "frozen_plan": "recovered from producer commit; current plan is a post-terminal living-record update",
        },
        "boundaries": {
            "prior_step5_decision": "NOT_SUPPORTED_EMPIRICAL_VALUE",
            "mainline_position": "ENDPOINT_EXTENSION_ONLY",
            "b1_started": False, "natural_localization_evaluated": False,
            "evaluation_access": False, "confirmation_experiment_authorized": False,
            "step6_authorized": False, "step6_started": False,
        },
        "verification": {
            "canonical_json_digests_recomputed": True, "file_sha256_recomputed": True,
            "queue_sqlite_recomputed": True, "rank_metrics_recomputed_from_raw_results": True,
            "aggregate_gates_recomputed_from_raw_results": True,
            "resource_totals_recomputed": True, "source_bindings_verified": len(source_checks),
            "pytest_passed": test_count,
        },
        "commands": {
            "executed_reporting_recovery": [
                "STEP5_1_RUN_ID=step5_1_r7_20260921 .conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/freeze_step5_1.py",
                ".conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/inherit_step5_1_r7.py",
                ".conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/finalize_step5_1.py vit_pcia_pefa/step5_1_r7_20260921",
                ".conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/validate_step5_1.py vit_pcia_pefa/step5_1_r7_20260921",
            ],
            "current_verification": [
                ".conda/envs/vitnns/bin/python3.10 vit_pcia_pefa/audit_step5_1_completion.py --validate-only",
                ".conda/envs/vitnns/bin/python3.10 -m pytest -q vit_pcia_pefa/tests/test_step4_vit_adapter.py vit_pcia_pefa/tests/test_step5_pilot.py vit_pcia_pefa/tests/test_step5_1_localization.py vit_pcia_pefa/tests/test_step5_1_queue.py",
            ],
            "note": "The freeze, inheritance and finalization commands are historical execution records and must not be rerun over existing immutable roots.",
        },
    }


def validate_completion() -> dict[str, Any]:
    audit = verify_signed(OUTPUT / "completion_audit.json")
    evidence = verify_signed(OUTPUT / "evidence_index.json")
    terminal = verify_signed(OUTPUT / "terminal_receipt.json")
    require(audit["status"] == "PASS_STEP5_1_PROMPT_TO_ARTIFACT_AUDIT", "AUDIT_STATUS")
    require(terminal["completion_audit_digest"] == audit["canonical_digest"]
            and terminal["evidence_index_digest"] == evidence["canonical_digest"],
            "COMPLETION_LINKAGE")
    for row in evidence["files"]:
        path = REPO / row["path"]
        require(path.is_file() and path.stat().st_size == row["bytes"]
                and sha256(path) == row["sha256"], "COMPLETION_EVIDENCE:" + row["path"] )
    return {"status": "PASS_STEP5_1_COMPLETION_SEAL",
            "completion_terminal_digest": terminal["canonical_digest"],
            "audited_terminal_digest": terminal["audited_terminal_digest"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_completion(), sort_keys=True))
        return 0
    require(not OUTPUT.exists(), "COMPLETION_ROOT_EXISTS")
    audit = build_audit()
    audit["completed_unix_ns"] = time.time_ns()
    audit = signed(audit)
    evidence_paths = [
        ROOT / "执行规划.md", ROOT / "step5_1_protocol.md",
        ROOT / "audit_step5_1_completion.py", ROOT / "step5_1_r7_recovery_amendment.md",
        ROOT / "step5_1_completion_r2_amendment.md",
        ROOT / "step5_1_completion_20260921/terminal_receipt.json",
        ROOT / "step5_completion_20260921/resource_ledger.json",
        RUN / "protocol.json", RUN / "task_manifest.json", RUN / "source_binding.json",
        RUN / "freeze_receipt.json", RUN / "inheritance_receipt.json",
        RUN / "a0_decision.json", RUN / "a1_launch_terminal.json", RUN / "a1_analysis.json",
        RUN / "c1_result.json", RUN / "claim_decisions.json", RUN / "resource_ledger.json",
        RUN / "evidence_index.json", RUN / "terminal_receipt.json",
    ]
    evidence = signed({
        "schema_version": 1, "run_id": "step5_1_completion_r2_20260921",
        "files": [identity(path) for path in evidence_paths],
    })
    terminal = signed({
        "schema_version": 1, "run_id": "step5_1_completion_r2_20260921",
        "stage": "ENDPOINT_PCIA_PEFA_STEP5_1",
        "status": "STEP5_1_COMPLETE_NOT_SUPPORTED_INCREMENTAL_OVER_SIMPLE_BASELINES",
        "completion_audit_digest": audit["canonical_digest"],
        "evidence_index_digest": evidence["canonical_digest"],
        "audited_root": RUN.name,
        "audited_terminal_digest": audit["audited_terminal_digest"],
        "prior_step5_decision_unchanged": "NOT_SUPPORTED_EMPIRICAL_VALUE",
        "mainline_position": "ENDPOINT_EXTENSION_ONLY",
        "b1_started": False, "evaluation_access": False,
        "confirmation_experiment_authorized": False, "step6_started": False,
        "next_authorization": "NONE", "completed_unix_ns": time.time_ns(),
    })
    OUTPUT.mkdir()
    write_new(OUTPUT / "completion_audit.json", audit)
    write_new(OUTPUT / "evidence_index.json", evidence)
    write_new(OUTPUT / "terminal_receipt.json", terminal)
    print(json.dumps({"status": audit["status"], "terminal_status": terminal["status"],
                      "checklist_items": len(audit["prompt_to_artifact_checklist"]),
                      "step5_1_h20_hours": audit["recomputed"]["step5_1_h20_hours"],
                      "cumulative_step5_h20_hours_upper_bound":
                          audit["recomputed"]["cumulative_step5_h20_hours_upper_bound"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
