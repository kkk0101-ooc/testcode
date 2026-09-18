#!/usr/bin/env python3
"""Independent CPU handoff validator for ENDPOINT-E4."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import subprocess
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_e4_intervention_r1_20260918"
SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_e4_intervention_run_spec.json"
PASS_STATUS = "ENDPOINT_E4_PASS_EXECUTION_LOCAL_CONTRACT_AND_FINITE_REPLACEMENT"
MAIN_FAMILIES = ("heads", "routing_value", "residual_branches")
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


def git_blob_sha(commit: str, relative: str) -> str:
    data = subprocess.check_output(
        ["git", "show", f"{commit}:vitnns_rebuild/{relative}"], cwd=REPO,
    )
    return hashlib.sha256(data).hexdigest()


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values); start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0
        for position in range(start, stop):
            result[order[position]] = rank
        start = stop
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> tuple[float | None, str | None]:
    if len(left) != len(right) or len(left) < 8:
        return None, "NA_INSUFFICIENT_COMPONENTS"
    x, y = average_ranks(left), average_ranks(right)
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    centered_x = [value - mx for value in x]; centered_y = [value - my for value in y]
    denominator = math.sqrt(math.fsum(value * value for value in centered_x) * math.fsum(value * value for value in centered_y))
    if denominator == 0:
        return None, "NA_CONSTANT_VECTOR"
    return math.fsum(a * b for a, b in zip(centered_x, centered_y)) / denominator, None


def load_queue(root: Path) -> tuple[dict[str, dict], list[dict]]:
    queue = root / "formal_queue.sqlite3"; results_root = root / "formal_results"
    with sqlite3.connect(queue) as connection:
        rows = connection.execute(
            "SELECT task_id,payload_json,payload_digest,seed,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id"
        ).fetchall()
        attempts = connection.execute("SELECT task_id,attempt,status FROM attempts ORDER BY task_id,attempt").fetchall()
    require(len(rows) == 800, "QUEUE_TASK_COUNT")
    require(len([row for row in attempts if row[2] == "COMPLETED"]) == 800, "QUEUE_ACCEPTED_ATTEMPTS")
    output, payloads = {}, []
    for task_id, payload_json, payload_digest, seed, state, attempt, result_path, result_digest in rows:
        require(hashlib.sha256(payload_json.encode()).hexdigest() == payload_digest, "QUEUE_PAYLOAD_DIGEST")
        payload = json.loads(payload_json); payloads.append({"task_id": task_id, "payload": payload})
        require(state == "completed" and result_path, "QUEUE_INCOMPLETE")
        document = load(results_root / result_path); signed = dict(document); observed = signed.pop("canonical_digest")
        require(digest(signed) == observed == result_digest, "QUEUE_RESULT_DIGEST")
        require(document["task_id"] == task_id and document["attempt"] == attempt and document["seed"] == int(seed), "QUEUE_RESULT_IDENTITY")
        output[task_id] = document["result"]
    return output, payloads


def check_metric(observed: dict, left: Sequence[float], right: Sequence[float], code: str) -> None:
    value, reason = spearman(left, right)
    require(observed["n"] == len(left) and observed["reason"] == reason, code + "_SCHEMA")
    if value is None:
        require(observed["value"] is None, code + "_NA")
    else:
        require(math.isclose(observed["value"], value, rel_tol=1e-12, abs_tol=1e-12), code + "_VALUE")


def validate(root: Path) -> dict:
    root = root.resolve(); spec = load(SPEC_PATH)
    # Import-free structural validation of the closed production spec.
    require(set(spec) == {"schema_version", "stage", "authorization_token", "predecessors", "cohort", "models", "target", "local_contract", "finite_utility", "method_scope", "runtime", "outputs", "failure"}, "SPEC_SCHEMA")
    require(spec["stage"] == "ENDPOINT-E4" and spec["authorization_token"] == "ENDPOINT_E4_EXPLICIT_USER_REQUEST_20260918", "SPEC_IDENTITY")
    report = load(root / "validation_report.json"); terminal = load(root / "terminal_receipt.json"); manifest = load(root / "artifact_manifest.json")
    require(report["status"] == terminal["status"] == PASS_STATUS, "STATUS")
    require(report["stage"] == terminal["stage"] == "ENDPOINT-E4", "STAGE")
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
    for row in report["implementation_bindings"]:
        require(row["path"] in SOURCE_FILES and row["commit"] == producer, "IMPLEMENTATION_RECORD")
        require(git_blob_sha(producer, row["path"]) == row["sha256"], "IMPLEMENTATION_BINDING:" + row["path"])
    require({row["path"] for row in report["implementation_bindings"]} == set(SOURCE_FILES), "IMPLEMENTATION_COVERAGE")
    for key, relative in BINDINGS.items():
        observed = git_blob_sha(producer, relative) if key == "unified_plan_sha256" else file_sha256(REPO / relative)
        require(observed == spec["predecessors"][key], "PREDECESSOR_BINDING:" + key)
    e3 = load(REPO / "reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json")
    require(e3["status"] == "ENDPOINT_E3_PASS_CORRECTNESS_ONLY_NEXT_ENDPOINT_E4_REQUIRES_EXPLICIT_REQUEST", "E3_STATUS")
    material = load(REPO / spec["outputs"]["source_materialization"])
    require(material["records_digest"] == report["materialization_records_digest"] == digest(material["records"]), "MATERIALIZATION")
    tasks = load(root / "task_manifest.json"); results, payloads = load_queue(root)
    require(tasks["task_count"] == 800 and tasks["manifest_digest"] == digest(tasks["tasks"]), "TASK_MANIFEST")
    require(tasks["tasks"] == payloads and tasks["queue_identity"]["manifest_digest"] == digest(payloads), "QUEUE_MANIFEST")
    frozen = {(model, row["split_order"]): row for model in ("vit_b16", "deit_s16") for row in material["records"]}
    local_count = local_eta_count = main_count = secondary_count = zeroing_count = 0
    for task_id, result in results.items():
        row = frozen[(result["model_id"], result["split_order"])]
        require((result["sample_id"], result["class_id"], result["wnid"], result["image_sha256"]) == (row["sample_id"], row["class_id"], row["wnid"], row["sha256"]), "TASK_COHORT")
        require(result["stage"] == "ENDPOINT-E4" and result["claim_scope"] == "LOCAL_CORRECTNESS_AND_FINITE_REPLACEMENT_NO_FINAL_CLAIM_DECISION", "RESULT_SCOPE")
        require(result["input_shape"] == [1, 3, 224, 224] and result["dtype"] == "torch.float32", "INPUT_CONTRACT")
        expected_main = 192 if result["model_id"] == "vit_b16" else 120
        expected_all = expected_main + (36 if result["split_order"] < 64 else 0)
        require(result["main_component_count"] == expected_main and result["all_component_count"] == expected_all, "COMPONENT_SCOPE")
        require(result["include_local"] == (result["split_order"] < 16) and result["include_secondary"] == (result["split_order"] < 64), "SUBSET_SCOPE")
        local = result["local_contract"]
        if result["include_local"]:
            require(local["performed"] and local["all_pass"] and len(local["components"]) == expected_all, "LOCAL_SCOPE")
            require(local["noop_max_abs"] <= 1e-4 + 1e-4 * abs(result["clean_score"]), "NOOP_PARITY")
            for component in local["components"]:
                require(component["pass"] and len(component["curve"]) == 4, "LOCAL_PASS")
                require([item["eta"] for item in component["curve"]] == [0.01, 0.003, 0.001, 0.0003], "ETA_SCOPE")
                require(all(math.isfinite(item["estimate"]) and math.isfinite(item["absolute_error"]) for item in component["curve"]), "LOCAL_FINITE")
                require(component["minimum_absolute_error"] <= component["threshold"] and component["directional_prediction_absolute_error"] <= component["threshold"], "LOCAL_THRESHOLD")
                local_count += 1; local_eta_count += 4
        else:
            require(local == {"performed": False, "components": [], "all_pass": None}, "LOCAL_FORBIDDEN_OUTSIDE_SUBSET")
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
        if result["include_secondary"]:
            expected_zero = 168 if result["model_id"] == "vit_b16" else 96
            require(len(zeroing["components"]) == expected_zero, "ZEROING_COUNT"); zeroing_count += expected_zero
        else:
            require(zeroing["components"] == [], "ZEROING_FORBIDDEN")
        identity = result["worker_identity"]
        require(identity["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and identity["startup_matmul_backward"] == "PASS", "CUDA_STARTUP")
        require(identity["deterministic_algorithms"] and not identity["cuda_matmul_allow_tf32"] and not identity["cudnn_allow_tf32"] and not identity["autocast"], "DETERMINISM")
    require((local_count, local_eta_count, main_count, secondary_count, zeroing_count) == (6144, 24576, 249600, 9216, 16896), "GLOBAL_COUNTS")
    require(report["local_component_evaluations"] == terminal["local_component_evaluation_count"] == local_count, "REPORT_LOCAL_COUNT")
    require(report["local_eta_evaluations"] == terminal["local_eta_evaluation_count"] == local_eta_count, "REPORT_ETA_COUNT")
    require(report["main_component_reference_evaluations"] == terminal["main_component_reference_evaluation_count"] == main_count, "REPORT_MAIN_COUNT")
    require(report["secondary_component_reference_evaluations"] == terminal["secondary_component_reference_evaluation_count"] == secondary_count, "REPORT_SECONDARY_COUNT")
    require(report["zeroing_secondary_evaluations"] == terminal["zeroing_secondary_evaluation_count"] == zeroing_count, "REPORT_ZERO_COUNT")
    require(report["formal_coverage"]["status"] == "PASS" and report["formal_coverage"]["completed_count"] == 800, "FORMAL_COVERAGE")
    require(report["gpu_hours_upper_observed"] <= 136 and report["formal_wall_seconds"] <= 132 * 3600, "TIME_CAP")
    require(report["peak_reserved_gpu_bytes_max"] <= 24 * 1024 ** 3 and report["peak_cpu_rss_bytes_max"] <= 32 * 1024 ** 3, "MEMORY_CAP")
    require(report["retained_bytes_before_report"] <= 56 * 1024 ** 3, "STORAGE_CAP")
    require(report["deferred"] == {
        "c4a_bootstrap": "ENDPOINT-E6", "c4a_holm": "ENDPOINT-E6", "c4a_claim_decisions": "ENDPOINT-E6",
        "c4b_spatial_intervention": "ENDPOINT-E5_REQUIRES_EXPLICIT_USER_REQUEST",
        "path_ig_input_fidelity_stability": "NOT_RUN_NOT_ALLOWED",
    }, "DEFERRED_SCOPE")
    require(report["next_authorization"] == terminal["next_authorization"] == "ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST", "NEXT_AUTHORIZATION")
    return {
        "status": PASS_STATUS, "evaluation_images": 400, "image_model_tasks": 800,
        "main_component_reference_evaluations": main_count,
        "secondary_component_reference_evaluations": secondary_count,
        "local_component_evaluations": local_count, "local_eta_evaluations": local_eta_count,
        "zeroing_secondary_evaluations": zeroing_count,
        "next_authorization": terminal["next_authorization"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    print(json.dumps(validate(parser.parse_args().root), sort_keys=True))
