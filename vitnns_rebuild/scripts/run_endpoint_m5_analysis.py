#!/usr/bin/env python3
"""Run the frozen CPU-only Endpoint M5 reuse analysis."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_m5_analysis import METHOD_MATRIX, average_ranks, head_block_metrics, summarize

SPEC_PATH = REPO / "config/nns_vit_endpoint/endpoint_m5.json"
ROOT = REPO / "reports/nns_vit_endpoint_m5_r2_20260919"
M1 = REPO / "reports/nns_vit_endpoint_m1_r1_20260918"
M2 = REPO / "reports/nns_vit_endpoint_m2_r1_20260919"
M4 = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"
M4_REVIEW = REPO / "reports/nns_vit_endpoint_m4_review_r2_20260919"
EP2 = REPO / "reports/nns_vit_endpoint_ep2_r1_20260918"
M2_METHODS = ("endpoint_absolute", "endpoint_signed", "regional_taylor", "gradient_only", "displacement_only", "random", "center", "attention")


def require(condition, code):
    if not condition:
        raise ValueError(code)


def load(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_manifest(root, full_hash):
    manifest = load(root / "artifact_manifest.json")
    require(manifest["file_count"] == len(manifest["files"]), "MANIFEST_COUNT:" + root.name)
    failures = []
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            failures.append(row["path"])
        elif full_hash and sha(path) != row["sha256"]:
            failures.append(row["path"])
    require(not failures, "MANIFEST_FILES:" + root.name + ":" + ",".join(failures[:3]))
    return {"root": root.relative_to(REPO).as_posix(), "file_count": len(manifest["files"]), "full_sha256": full_hash}


def predecessor_gate(spec):
    for relative, expected in spec["source_bindings"].items():
        require(sha(REPO / relative) == expected, "SOURCE_BINDING:" + relative)
    m1_terminal = load(M1 / "terminal_receipt.json")
    m2_terminal = load(M2 / "terminal_receipt.json")
    m4_terminal = load(M4 / "terminal_receipt.json")
    review_terminal = load(M4_REVIEW / "terminal_receipt.json")
    ep2_terminal = load(EP2 / "terminal_receipt.json")
    require(m1_terminal["status"] == "ENDPOINT_M1_PASS_EXECUTION_AND_AUDIT", "M1_STATUS")
    require(m2_terminal["status"] == "ENDPOINT_M2_PASS_EXECUTION_AND_AUDIT", "M2_STATUS")
    require(m4_terminal["status"] == "ENDPOINT_M4_PASS_EXECUTION_AND_AUDIT", "M4_STATUS")
    require(review_terminal["status"] == "ENDPOINT_M4_REVIEW_R2_PASS_POST_HOC_SENSITIVITY", "M4_REVIEW_STATUS")
    require(ep2_terminal["status"] == "EP2_NUMERICAL_CALIBRATION_FAILED", "EP2_STATUS")
    for root, terminal in ((M1, m1_terminal), (M2, m2_terminal), (M4, m4_terminal), (M4_REVIEW, review_terminal)):
        require(sha(root / "artifact_manifest.json") == terminal["artifact_manifest_sha256"], "TERMINAL_MANIFEST:" + root.name)
    manifest_checks = [
        verify_manifest(M1, False),
        verify_manifest(M2, False),
        verify_manifest(M4, True),
        verify_manifest(M4_REVIEW, True),
    ]
    historical = spec["historical_stage_plan_binding"]
    payload = subprocess.check_output(
        ["git", "show", historical["commit"] + ":" + historical["path"]], cwd=REPO.parent
    )
    require(hashlib.sha256(payload).hexdigest() == historical["sha256"], "HISTORICAL_STAGE_PLAN")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    tracking = subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=REPO, text=True).strip()
    require(head == tracking, "UNPUBLISHED_ANALYZER")
    return {
        "status": "PASS",
        "producer_commit": head,
        "tracking_commit": tracking,
        "terminal_statuses": {
            "M1": m1_terminal["status"], "M2": m2_terminal["status"],
            "M4": m4_terminal["status"], "M4_review": review_terminal["status"],
            "EP2": ep2_terminal["status"],
        },
        "manifest_checks": manifest_checks,
        "historical_stage_plan_sha256": historical["sha256"],
        "live_stage_plan_sha256": sha(REPO / "docs/阶段M_endpointnnsvit.md"),
        "expected_postseal_stage_plan_drift": True,
    }


def reference_analysis():
    manifest = load(M1 / "metrics_manifest.json")
    require(manifest["task_count"] == len(manifest["records"]) == 800, "M1_METRICS_MANIFEST")
    expected = {row["task_id"]: row for row in manifest["records"]}
    image_rows = []
    endpoint_pairs = 0
    maximum_m_difference = 0.0
    maximum_a_difference = 0.0
    exact_taylor_blocks = 0
    exact_effect_blocks = 0
    reference_sensitive_taylor_components = 0
    reference_sensitive_effect_components = 0
    for path in sorted((M1 / "per_image").glob("*.json.gz")):
        task_id = path.name[:-8]
        require(task_id in expected and sha(path) == expected[task_id]["sha256"], "M1_PER_IMAGE:" + task_id)
        data = load(path)
        zero_rows = data["component_rows"]["normalized_zero"]
        blur_rows = data["component_rows"]["blur_rgb"]
        zero = {row["component_id"]: row for row in zero_rows}
        blur = {row["component_id"]: row for row in blur_rows}
        require(set(zero) == set(blur), "M1_COMPONENT_IDS:" + task_id)
        for component_id in zero:
            left, right = zero[component_id], blur[component_id]
            endpoint_pairs += 1
            maximum_m_difference = max(maximum_m_difference, abs(float(left["m"]) - float(right["m"])))
            maximum_a_difference = max(maximum_a_difference, abs(float(left["a"]) - float(right["a"])))
            reference_sensitive_taylor_components += not math.isclose(abs(float(left["p_ref"])), abs(float(right["p_ref"])), rel_tol=0, abs_tol=1e-12)
            reference_sensitive_effect_components += not math.isclose(float(left["absolute_effect"]), float(right["absolute_effect"]), rel_tol=0, abs_tol=1e-12)
        values, exact = head_block_metrics(zero_rows, blur_rows)
        exact_taylor_blocks += exact["taylor"]
        exact_effect_blocks += exact["finite_effect"]
        row = {
            "model_id": data["model_id"], "sample_id": data["sample_id"],
            "split_order": data["split_order"], **values,
        }
        row["taylor_minus_endpoint_normalized_zero"] = values["taylor_behavior_normalized_zero"] - values["endpoint_behavior_normalized_zero"]
        row["taylor_minus_endpoint_blur_rgb"] = values["taylor_behavior_blur_rgb"] - values["endpoint_behavior_blur_rgb"]
        image_rows.append(row)
    require(len(image_rows) == 800 and endpoint_pairs == 163200, "M1_SCOPE")
    require(maximum_m_difference == 0 and maximum_a_difference == 0, "ENDPOINT_REFERENCE_INVARIANCE")
    metrics = [key for key in image_rows[0] if key not in {"model_id", "sample_id", "split_order"}]
    summaries = []
    for model in ("vit_b16", "deit_s16"):
        rows = [row for row in image_rows if row["model_id"] == model]
        require(len(rows) == 400 and {row["split_order"] for row in rows} == set(range(400)), "M1_MODEL_SCOPE:" + model)
        for metric in metrics:
            summaries.append({"model_id": model, "metric": metric, **summarize([row[metric] for row in rows], "M5|" + model + "|" + metric)})
    output = {
        "status": "PASS_REUSE_ONLY_DESCRIPTIVE",
        "image_model_rows": len(image_rows),
        "endpoint_component_reference_pairs": endpoint_pairs,
        "head_block_reference_comparisons": 9600,
        "endpoint_invariance": {"maximum_m_absolute_difference": maximum_m_difference, "maximum_a_absolute_difference": maximum_a_difference, "exact": True},
        "reference_sensitivity": {
            "taylor_components_changed": reference_sensitive_taylor_components,
            "finite_effect_components_changed": reference_sensitive_effect_components,
            "total_components": endpoint_pairs,
            "exact_taylor_rank_blocks": exact_taylor_blocks,
            "exact_finite_effect_rank_blocks": exact_effect_blocks,
            "total_head_blocks": 9600,
        },
        "summaries": summaries,
        "new_inferential_tests": 0,
    }
    return image_rows, output


def m2_analysis():
    manifest = load(M2 / "artifact_manifest.json")
    entry = next(row for row in manifest["files"] if row["path"] == "image_values.json.gz")
    path = M2 / entry["path"]
    require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "M2_IMAGE_VALUES")
    source = load(path)
    rows = []
    selected = []
    for group in source:
        model, block, geometry, reference, outcome, metric = group["key"]
        if block not in (2, 5, 10) or geometry != "patch" or outcome not in ("full", "isolated") or metric != "auc":
            continue
        values = list(group["images"].values())
        require(len(values) == 400 and all(len(value) == len(M2_METHODS) for value in values), "M2_REUSE_SHAPE")
        endpoint = [float(value[M2_METHODS.index("endpoint_absolute")]) for value in values]
        taylor = [float(value[M2_METHODS.index("regional_taylor")]) for value in values]
        rows.append({
            "model_id": model, "block": block, "reference": reference, "outcome": outcome,
            "n_images": 400, "endpoint_auc_median": statistics.median(endpoint),
            "regional_taylor_auc_median": statistics.median(taylor),
            "endpoint_minus_taylor_median": statistics.median([a - b for a, b in zip(endpoint, taylor)]),
        })
        selected.append((model, block, reference, outcome))
    require(len(rows) == 24 and len(set(selected)) == 24, "M2_REUSE_GROUPS")
    report = load(M2 / "report_data.json")
    return rows, {
        "status": "PASS_RECOMPUTED_FROM_SEALED_IMAGE_VALUES", "groups": len(rows),
        "images_per_group": 400, "new_tests": 0,
        "inherited_patch_taylor_decisions": report["patch_taylor_comparisons"],
        "original_test_family": "672 bidirectional AUC tests; no new M5 family",
    }


def directory_bytes(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def cost_information():
    m1_task = load(M1 / "task_validation.json")
    m1_acquisition = load(M1 / "acquisition_receipt.json")
    m1_metrics = load(M1 / "metrics_manifest.json")
    m2_task = load(M2 / "task_validation.json")
    m2_acquisition = load(M2 / "acquisition_receipt.json")
    ep2_terminal = load(EP2 / "terminal_receipt.json")
    ep2_report = load(EP2 / "calibration_report.json")
    model_decisions = ep2_report["model_decisions"]
    return {
        "status": "DESCRIPTIVE_PACKAGE_AND_OPERATION_COSTS_NOT_METHOD_RUNTIME_BENCHMARK",
        "information_conditions": [
            {"method": "Endpoint", "operations": "observed-input forward plus target backward/VJP", "reference_state": "none external", "cache": "clean activations and target cotangent as required by the registered cut"},
            {"method": "Activation-space Taylor", "operations": "clean target backward plus reference/counterfactual activation forward", "reference_state": "required", "cache": "clean cotangent and clean/reference activations"},
            {"method": "Integrated Gradients", "operations": "M path-node forward/backward evaluations", "reference_state": "required", "cache": "accumulated input-gradient integral; optional path batches"},
            {"method": "Finite replacement", "operations": "one counterfactual forward per candidate replacement", "reference_state": "required", "cache": "candidate logits/effects and selector scores"},
        ],
        "observed_packages": {
            "M1": {"attempt_gpu_hours": m1_task["attempt_gpu_hours"], "acquisition_wall_seconds": m1_acquisition["wall_seconds"], "analysis_seconds": m1_metrics["seconds"], "new_conditions": m1_task["new_conditions"], "combined_component_reference_conditions": 326400, "local_bytes": directory_bytes(M1)},
            "M2": {"gpu_hours": m2_task["gpu_hours"], "acquisition_wall_seconds": m2_acquisition["wall_seconds"], "tasks": m2_task["task_count"], "candidates": m2_task["candidates"], "curves": m2_task["curves"], "points": m2_task["points"], "local_bytes": directory_bytes(M2)},
            "EP2_Path_IG_calibration": {"gpu_hours_upper_observed": ep2_report["gpu_hours_upper_observed"], "elapsed_seconds": ep2_terminal["elapsed_seconds"], "maximum_points": 512, "deit_s16": model_decisions["deit_s16"], "vit_b16": model_decisions["vit_b16"]},
        },
        "runtime_claim": "NOT_IDENTIFIABLE_FROM_COMBINED_M1_M2_PACKAGES",
        "reason": "The packages jointly execute multiple predictors, shared forwards, controls, random repetitions, and candidate interventions; they do not isolate comparable per-method runtimes.",
    }


def main():
    require(not any(name in sys.modules for name in ("torch", "timm", "PIL")), "PROHIBITED_IMPORT")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_NOT_CLEARED")
    require(not ROOT.exists(), "M5_ROOT_ALREADY_EXISTS")
    spec = load(SPEC_PATH)
    started = time.monotonic()
    gate = predecessor_gate(spec)
    ROOT.mkdir(parents=True)
    write_json(ROOT / "predecessor_validation.json", gate)
    write_json(ROOT / "freeze.json", {
        "stage": "ENDPOINT-M5-R2", "time_ns": time.time_ns(),
        "producer_commit": gate["producer_commit"], "config_sha256": sha(SPEC_PATH),
        "source_bindings": spec["source_bindings"], "recovery": spec["recovery"],
        "next_authorization": "STOP_AFTER_M5",
    })
    image_rows, reference_summary = reference_analysis()
    m2_rows, m2_receipt = m2_analysis()
    write_csv(ROOT / "reference_reuse_image_values.csv", image_rows)
    write_json(ROOT / "reference_reuse_summary.json", reference_summary)
    write_csv(ROOT / "m2_reuse_summary.csv", m2_rows)
    write_json(ROOT / "m2_reuse_receipt.json", m2_receipt)
    write_json(ROOT / "method_applicability_matrix.json", {"schema_version": 1, "methods": METHOD_MATRIX})
    write_json(ROOT / "cost_and_information.json", cost_information())
    write_json(ROOT / "analysis_receipt.json", {
        "status": "ENDPOINT_M5_ANALYSIS_COMPLETE", "mode": "CPU_ONLY_REUSE_ONLY",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_timm_pil_imported": False, "jpeg_reads": 0, "model_or_checkpoint_loads": 0,
        "new_gpu_production": False, "new_inferential_tests": 0,
        "image_model_rows": len(image_rows), "m2_reuse_groups": len(m2_rows),
        "elapsed_seconds": time.monotonic() - started,
    })
    print(json.dumps({"status": "ENDPOINT_M5_ANALYSIS_COMPLETE", "root": str(ROOT), "image_rows": len(image_rows), "m2_groups": len(m2_rows)}))


if __name__ == "__main__":
    main()
