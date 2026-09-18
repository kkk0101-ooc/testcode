#!/usr/bin/env python3
"""Independent source-recomputing validator for ENDPOINT-E4 R3."""
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
ROOT = REPO / "reports/nns_vit_endpoint_e4_intervention_r3_20260918"
SOURCE = REPO / "reports/nns_vit_endpoint_e4_intervention_r2_20260918"
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_e4_intervention_recovery_r3.json"
PASS_STATUS = "ENDPOINT_E4_R3_PASS_EXECUTION_CONDITION_SCALED_JVP_AND_FINITE_REPLACEMENT"
SOURCE_FILES = [
    "config/nns_vit_endpoint/endpoint_e4_intervention_recovery_r3.json",
    "docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE4恢复R3_20260918.md",
    "src/vitnns/endpoint_e4_intervention.py",
    "scripts/run_endpoint_e4_intervention_worker.py",
    "scripts/run_endpoint_e4_recovery_r3.py",
    "scripts/validate_endpoint_e4_recovery_r3.py",
    "scripts/validate_endpoint_e4_recovery_r2_failure.py",
    "scripts/diagnose_endpoint_e4_r2_jvp.py",
    "scripts/validate_endpoint_unified_plan.py",
    "tests/test_endpoint_e4_intervention.py",
]
BINDINGS = {
    "cohort_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json",
    "endpoint_e3_r2_terminal_sha256": "reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json",
    "endpoint_e4_r2_artifact_manifest_sha256": "reports/nns_vit_endpoint_e4_intervention_r2_20260918/artifact_manifest.json",
    "endpoint_e4_r2_coverage_sha256": "reports/nns_vit_endpoint_e4_intervention_r2_20260918/formal_coverage_audit.json",
    "endpoint_e4_r2_diagnostic_sha256": "reports/nns_vit_endpoint_e4_r2_jvp_diagnostic_20260918/diagnostic.json",
    "endpoint_e4_r2_terminal_sha256": "reports/nns_vit_endpoint_e4_intervention_r2_20260918/terminal_receipt.json",
    "endpoint_e4_r2_validation_sha256": "reports/nns_vit_endpoint_e4_intervention_r2_20260918/validation_report.json",
    "intervention_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/intervention_registry.json",
    "materialization_receipt_sha256": "reports/nns_vit_endpoint_e2_atlas_r1_20260918/materialization_receipt.json",
    "model_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
    "statistical_analysis_plan_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json",
    "target_registry_sha256": "manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json",
    "unified_plan_sha256": "docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md",
}
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from validate_endpoint_e4_intervention import check_metric  # noqa: E402
from validate_endpoint_e4_recovery_r2_failure import validate as validate_r2_failure  # noqa: E402
from vitnns.endpoint_e2_atlas import digest, file_sha256  # noqa: E402
from vitnns.endpoint_e4_intervention import MAIN_FAMILIES, validate_recovery_r3_run_spec  # noqa: E402


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def git_blob_sha(commit: str, relative: str) -> str:
    data = subprocess.check_output(["git", "show", f"{commit}:vitnns_rebuild/{relative}"], cwd=REPO)
    return hashlib.sha256(data).hexdigest()


def validate_result(result: dict, frozen: dict, *, new: bool) -> tuple[int, int, int, int]:
    expected_stage = "ENDPOINT-E4-R3-RECOVERY" if new else "ENDPOINT-E4-R2-RECOVERY"
    require(result["stage"] == expected_stage, "RESULT_STAGE")
    require(result["claim_scope"] == "LOCAL_CORRECTNESS_AND_FINITE_REPLACEMENT_NO_FINAL_CLAIM_DECISION", "RESULT_CLAIM_SCOPE")
    require((result["sample_id"], result["class_id"], result["wnid"], result["image_sha256"]) == (frozen["sample_id"], frozen["class_id"], frozen["wnid"], frozen["sha256"]), "RESULT_COHORT")
    require(result["input_shape"] == [1, 3, 224, 224] and result["dtype"] == "torch.float32", "INPUT_CONTRACT")
    expected_main = 192 if result["model_id"] == "vit_b16" else 120
    expected_all = expected_main + (36 if result["split_order"] < 64 else 0)
    require(result["main_component_count"] == expected_main and result["all_component_count"] == expected_all, "COMPONENT_SCOPE")
    require(result["include_local"] == new and result["include_secondary"] == (result["split_order"] < 64), "SUBSET_SCOPE")
    local_count = 0; local = result["local_contract"]
    if new:
        require(local["performed"] and local["all_pass"] and local["method"] == "condition_scaled_jvp_r3", "LOCAL_METHOD")
        require(len(local["components"]) == expected_all and local["noop_max_abs"] <= 1e-4 + 1e-4 * abs(result["clean_score"]), "LOCAL_SCOPE")
        for component in local["components"]:
            require(component["pass"] and "curve" not in component, "LOCAL_PASS")
            require(component["atol"] == 1e-5 and component["rtol_absolute_dot_mass"] == 1e-5, "LOCAL_TOLERANCE")
            threshold = 1e-5 + 1e-5 * component["absolute_dot_mass"]
            require(math.isclose(component["threshold"], threshold, rel_tol=0, abs_tol=0), "LOCAL_THRESHOLD_VALUE")
            require(component["directional_prediction_absolute_error"] <= threshold, "LOCAL_THRESHOLD")
            local_count += 1
    else:
        require(local == {"performed": False, "components": [], "all_pass": None}, "INHERITED_LOCAL_SCOPE")
    main_count = secondary_count = 0
    for reference in ("normalized_zero", "blur_rgb"):
        finite = result["finite_utility"][reference]; components = finite["components"]
        require(len(components) == expected_all and len({item["component_id"] for item in components}) == expected_all, "FINITE_COMPONENT_COVERAGE")
        require(all(all(math.isfinite(float(item[key])) for key in ("a", "m", "map_l2", "output_norm", "gradient_norm", "gradient_activation", "reference_output_norm", "reference_displacement_norm", "p_ref", "replacement_score", "effect", "absolute_effect")) for item in components), "FINITE_NUMERIC")
        for family in MAIN_FAMILIES:
            family_rows = [item for item in components if item["family"] == family]
            require(len(family_rows) >= 24, "FAMILY_COMPONENT_COUNT")
            metrics = finite["per_image_metrics"][family]
            effects = [item["effect"] for item in family_rows]; absolute = [abs(value) for value in effects]
            for name, left, right in (
                ("m_vs_absolute_effect", [item["m"] for item in family_rows], absolute),
                ("a_vs_effect", [item["a"] for item in family_rows], effects),
                ("p_ref_vs_effect", [item["p_ref"] for item in family_rows], effects),
                ("abs_p_ref_vs_absolute_effect", [abs(item["p_ref"]) for item in family_rows], absolute),
                ("output_norm_vs_absolute_effect", [item["output_norm"] for item in family_rows], absolute),
                ("gradient_norm_vs_absolute_effect", [item["gradient_norm"] for item in family_rows], absolute),
                ("abs_gradient_activation_vs_absolute_effect", [abs(item["gradient_activation"]) for item in family_rows], absolute),
            ):
                check_metric(metrics[name], left, right, "METRIC:" + name)
            require(metrics["random_20"]["repeat_count"] == len(metrics["random_20"]["repeats"]) == 20, "RANDOM_REPEAT")
        main_count += sum(item["family"] in MAIN_FAMILIES for item in components)
        secondary_count += sum(item["family"] not in MAIN_FAMILIES for item in components)
    zeroing = result["zeroing_secondary"]
    require(zeroing["performed"] == result["include_secondary"], "ZEROING_SCOPE")
    expected_zero = (168 if result["model_id"] == "vit_b16" else 96) if result["include_secondary"] else 0
    require(len(zeroing["components"]) == expected_zero, "ZEROING_COUNT")
    identity = result["worker_identity"]
    require(identity["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and identity["startup_matmul_backward"] == "PASS", "CUDA_STARTUP")
    require(identity["deterministic_algorithms"] and not identity["cuda_matmul_allow_tf32"] and not identity["cudnn_allow_tf32"] and not identity["autocast"], "DETERMINISM")
    return local_count, main_count, secondary_count, expected_zero


def validate(root: Path) -> dict:
    root = root.resolve(); spec = load(SPEC_PATH); validate_recovery_r3_run_spec(spec)
    require(spec["outputs"]["formal_root"] == root.relative_to(REPO).as_posix(), "SPEC_ROOT")
    report = load(root / "validation_report.json"); terminal = load(root / "terminal_receipt.json"); manifest = load(root / "artifact_manifest.json")
    require(report["status"] == terminal["status"] == PASS_STATUS, "STATUS")
    require(report["stage"] == terminal["stage"] == "ENDPOINT-E4-R3-RECOVERY", "STAGE")
    require(report["claim_scope"] == terminal["claim_scope"] == "EXECUTION_PASS_NOT_FINAL_C4A_CLAIM_DECISION", "CLAIM_SCOPE")
    require(terminal["validation_digest"] == digest(report), "VALIDATION_DIGEST")
    require(terminal["manifest_digest"] == manifest["canonical_digest"] == digest(manifest["files"]), "MANIFEST_DIGEST")
    paths = [row["path"] for row in manifest["files"]]; require(len(paths) == len(set(paths)), "MANIFEST_DUPLICATE")
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        require(path.is_relative_to(root) and path.is_file(), "MANIFEST_PATH")
        require(path.stat().st_size == row["bytes"] and file_sha256(path) == row["sha256"], "ARTIFACT_HASH:" + row["path"])
    producer = report["producer_commit"]
    require(subprocess.run(["git", "merge-base", "--is-ancestor", producer, "HEAD"], cwd=REPO).returncode == 0, "PRODUCER_ANCESTRY")
    require({row["path"] for row in report["implementation_bindings"]} == set(SOURCE_FILES), "IMPLEMENTATION_COVERAGE")
    for row in report["implementation_bindings"]:
        require(row["commit"] == producer and git_blob_sha(producer, row["path"]) == row["sha256"], "IMPLEMENTATION_BINDING:" + row["path"])
    for key, relative in BINDINGS.items():
        observed = git_blob_sha(producer, relative) if key == "unified_plan_sha256" else file_sha256(REPO / relative)
        require(observed == spec["predecessors"][key], "PREDECESSOR_BINDING:" + key)
    source_validation = validate_r2_failure(SOURCE)
    require(source_validation["status"] == "ENDPOINT_E4_R2_FAIL_ANALYTIC_JVP_CONTRACT", "SOURCE_STATUS")
    source_revalidation = load(root / "source_revalidation.json")
    require(source_revalidation["status"] == "PASS_R2_SOURCE_REVALIDATED_FOR_R3" and source_revalidation["inherited_count"] == 768, "SOURCE_REVALIDATION")
    diagnostic = load(REPO / "reports/nns_vit_endpoint_e4_r2_jvp_diagnostic_20260918/diagnostic.json")
    require(diagnostic["status"] == "DIAGNOSTIC_ONLY_NOT_STAGE_RESULT" and diagnostic["task_id"] == "vit_b16__009", "DIAGNOSTIC_SCOPE")
    require(diagnostic["source_r2_terminal_sha256"] == file_sha256(SOURCE / "terminal_receipt.json") and diagnostic["r2_failed_component_count"] == 1 and diagnostic["condition_scaled_failed_component_count"] == 0, "DIAGNOSTIC_RESULT")
    diagnostic_blob = subprocess.check_output(["git", "show", f"{diagnostic['diagnostic_commit']}:vitnns_rebuild/scripts/diagnose_endpoint_e4_r2_jvp.py"], cwd=REPO)
    require(hashlib.sha256(diagnostic_blob).hexdigest() == diagnostic["diagnostic_script_sha256"], "DIAGNOSTIC_IMPLEMENTATION")
    material = load(REPO / spec["outputs"]["source_materialization"]); require(material["records_digest"] == report["materialization_records_digest"] == digest(material["records"]), "MATERIALIZATION")
    frozen = {(model, row["split_order"]): row for model in ("vit_b16", "deit_s16") for row in material["records"]}
    combined = load(root / "combined_result_index.json"); records = combined["records"]
    require(combined["record_count"] == 800 and combined["records_digest"] == digest(records), "COMBINED_INDEX")
    require(len(records) == len({row["task_id"] for row in records}) == 800, "COMBINED_UNION")
    origins = {"inherited_from_r2_source_revalidated": 0, "newly_run_in_r3": 0}
    totals = [0, 0, 0, 0]
    for row in records:
        require(set(row) == {"task_id", "origin", "source_result_path", "source_result_digest", "source_attempt"}, "INDEX_SCHEMA")
        origins[row["origin"]] += 1; order = int(row["task_id"].rsplit("__", 1)[1])
        new = row["origin"] == "newly_run_in_r3"
        require(new == (order < 16), "ORIGIN_PARTITION")
        path = (REPO / row["source_result_path"]).resolve()
        require(path.is_file() and (path.is_relative_to(root) if new else path.is_relative_to(SOURCE)), "INDEX_PATH")
        document = load(path); signed = dict(document); observed = signed.pop("canonical_digest")
        require(digest(signed) == observed == row["source_result_digest"], "INDEX_RESULT_DIGEST")
        require(document["task_id"] == row["task_id"] and document["attempt"] == row["source_attempt"], "INDEX_RESULT_IDENTITY")
        result = document["result"]; model = result["model_id"]
        counts = validate_result(result, frozen[(model, order)], new=new)
        totals = [left + right for left, right in zip(totals, counts)]
    require(origins == {"inherited_from_r2_source_revalidated": 768, "newly_run_in_r3": 32}, "ORIGIN_COUNTS")
    require(tuple(totals) == (6144, 249600, 9216, 16896), "GLOBAL_COUNTS")
    tasks = load(root / "task_manifest.json"); require(tasks["task_count"] == 32 and tasks["manifest_digest"] == digest(tasks["tasks"]), "RERUN_MANIFEST")
    with sqlite3.connect(root / "formal_queue.sqlite3") as connection:
        queue_rows = connection.execute("SELECT task_id,payload_json,payload_digest,seed,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id").fetchall()
        attempts = connection.execute("SELECT status FROM attempts").fetchall()
    queue_payloads = []
    new_index = {row["task_id"]: row for row in records if row["origin"] == "newly_run_in_r3"}
    require(len(queue_rows) == 32 and all(row[4] == "completed" for row in queue_rows), "RERUN_QUEUE")
    for task_id, payload_json, payload_digest, seed, state, attempt, result_path, result_digest in queue_rows:
        require(hashlib.sha256(payload_json.encode()).hexdigest() == payload_digest, "RERUN_PAYLOAD_DIGEST")
        payload = json.loads(payload_json); queue_payloads.append({"task_id": task_id, "payload": payload})
        require(payload["stage"] == "ENDPOINT-E4-R3-RECOVERY" and payload["local_contract_mode"] == "condition_scaled_jvp_r3", "RERUN_PAYLOAD_SCOPE")
        index_row = new_index[task_id]
        require((root / "formal_results" / result_path).relative_to(REPO).as_posix() == index_row["source_result_path"], "RERUN_RESULT_PATH")
        require(result_digest == index_row["source_result_digest"] and int(attempt) == index_row["source_attempt"], "RERUN_RESULT_BINDING")
    require(queue_payloads == tasks["tasks"] and tasks["queue_identity"]["manifest_digest"] == digest(queue_payloads), "RERUN_QUEUE_MANIFEST")
    require(len(attempts) == 32 and all(row[0] == "COMPLETED" for row in attempts), "RERUN_ATTEMPTS")
    require(report["inherited_task_count"] == terminal["inherited_task_count"] == 768, "REPORT_INHERITED")
    require(report["new_task_count"] == terminal["new_task_count"] == 32, "REPORT_NEW")
    require(report["local_component_evaluations"] == terminal["local_component_evaluation_count"] == totals[0], "REPORT_LOCAL")
    require(report["main_component_reference_evaluations"] == terminal["main_component_reference_evaluation_count"] == totals[1], "REPORT_MAIN")
    require(report["secondary_component_reference_evaluations"] == terminal["secondary_component_reference_evaluation_count"] == totals[2], "REPORT_SECONDARY")
    require(report["zeroing_secondary_evaluations"] == terminal["zeroing_secondary_evaluation_count"] == totals[3], "REPORT_ZERO")
    require(report["r1_failure_preserved"] and report["r2_failure_preserved"] and not report["r1_results_inherited"] and not report["r2_local_results_inherited"], "RECOVERY_BOUNDARIES")
    require(report["local_eta_evaluations"] == terminal["local_eta_evaluation_count"] == 0, "ETA_REMOVED")
    require(math.isclose(report["combined_wall_seconds_observed"], report["source_r2_resources"]["wall_seconds"] + report["new_formal_wall_seconds"], rel_tol=0, abs_tol=1e-12), "WALL_ACCOUNTING")
    require(report["combined_gpu_hours_observed"] <= 136 and report["combined_wall_seconds_observed"] <= 132 * 3600, "TIME_CAP")
    require(report["peak_reserved_gpu_bytes_max_new"] <= 24 * 1024 ** 3 and report["peak_cpu_rss_bytes_max_new"] <= 32 * 1024 ** 3, "MEMORY_CAP")
    require(report["retained_bytes_before_report"] <= 56 * 1024 ** 3, "STORAGE_CAP")
    require(report["deferred"] == {
        "c4a_bootstrap": "ENDPOINT-E6", "c4a_holm": "ENDPOINT-E6",
        "c4a_claim_decisions": "ENDPOINT-E6",
        "c4b_spatial_intervention": "ENDPOINT-E5_REQUIRES_EXPLICIT_USER_REQUEST",
        "path_ig_input_fidelity_stability": "NOT_RUN_NOT_ALLOWED",
    }, "DEFERRED_SCOPE")
    require(report["next_authorization"] == terminal["next_authorization"] == "ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST", "NEXT_AUTHORIZATION")
    return {
        "status": PASS_STATUS, "image_model_tasks": 800, "inherited_tasks": 768,
        "new_tasks": 32, "local_component_evaluations": totals[0],
        "main_component_reference_evaluations": totals[1],
        "secondary_component_reference_evaluations": totals[2],
        "zeroing_secondary_evaluations": totals[3],
        "next_authorization": terminal["next_authorization"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    print(json.dumps(validate(parser.parse_args().root), sort_keys=True))
