#!/usr/bin/env python3
"""Independent standard-library audit of the Endpoint M5 reuse analysis."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m5_r2_20260919"
M1 = REPO / "reports/nns_vit_endpoint_m1_r1_20260918"
M2 = REPO / "reports/nns_vit_endpoint_m2_r1_20260919"
R1 = REPO / "reports/nns_vit_endpoint_m5_r1_20260919"
SPEC = REPO / "config/nns_vit_endpoint/endpoint_m5.json"
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


def ranks(values):
    values = [float(value) for value in values]
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        value = (start + stop - 1) / 2
        for offset in range(start, stop):
            result[order[offset]] = value
        start = stop
    return result


def rho(left, right):
    left = ranks(left); right = ranks(right)
    left_mean = math.fsum(left) / len(left); right_mean = math.fsum(right) / len(right)
    left = [value - left_mean for value in left]; right = [value - right_mean for value in right]
    denominator = math.sqrt(math.fsum(value * value for value in left) * math.fsum(value * value for value in right))
    require(denominator > 0, "UNDEFINED_RHO")
    return math.fsum(a * b for a, b in zip(left, right)) / denominator


def seed(key):
    return int.from_bytes(hashlib.sha256(("ENDPOINT_M5_20260919_V1|" + key).encode()).digest()[:8], "big")


def quantile(values, probability):
    position = (len(values) - 1) * probability
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] * (1 - fraction) + values[upper] * fraction)


def bootstrap(values, key, repetitions=10000):
    values = sorted(float(value) for value in values); count = len(values)
    rng = random.Random(seed(key)); draws = []
    require(count == 400, "BOOTSTRAP_IMAGE_COUNT")
    for _ in range(repetitions):
        lower_u = rng.betavariate(count // 2, count // 2 + 1)
        upper_u = lower_u + (1 - lower_u) * rng.betavariate(1, count // 2)
        draws.append((values[min(int(count * lower_u), count - 1)] + values[min(int(count * upper_u), count - 1)]) / 2)
    draws.sort()
    return [quantile(draws, 0.025), quantile(draws, 0.975)]


def close(left, right, code, tolerance=1e-14):
    require(math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance), code)


def verify_sources():
    spec = load(SPEC)
    for relative, expected in spec["source_bindings"].items():
        require(sha(REPO / relative) == expected, "SOURCE_BINDING:" + relative)
    recovery = spec["recovery"]
    require(sha(R1 / "predecessor_validation.json") == recovery["predecessor_validation_sha256"], "R1_PREDECESSOR")
    require(sha(R1 / "freeze.json") == recovery["freeze_sha256"], "R1_FREEZE")
    require({path.name for path in R1.iterdir()} == {"predecessor_validation.json", "freeze.json"}, "R1_FILE_SET")
    historical = spec["historical_stage_plan_binding"]
    payload = subprocess.check_output(["git", "show", historical["commit"] + ":" + historical["path"]], cwd=REPO.parent)
    require(hashlib.sha256(payload).hexdigest() == historical["sha256"], "HISTORICAL_PLAN")
    predecessor = load(ROOT / "predecessor_validation.json")
    require(predecessor["status"] == "PASS" and predecessor["producer_commit"] == predecessor["tracking_commit"], "PREDECESSOR_RECEIPT")
    require({row["root"]: row["file_count"] for row in predecessor["manifest_checks"]} == {
        "reports/nns_vit_endpoint_m1_r1_20260918": 1631,
        "reports/nns_vit_endpoint_m2_r1_20260919": 2436,
        "reports/nns_vit_endpoint_m4_r1_20260919": 838,
        "reports/nns_vit_endpoint_m4_review_r2_20260919": 30,
    }, "PREDECESSOR_MANIFEST_COUNTS")


def recompute_m1():
    manifest = load(M1 / "metrics_manifest.json")
    expected = {row["task_id"]: row["sha256"] for row in manifest["records"]}
    observed_rows = list(csv.DictReader((ROOT / "reference_reuse_image_values.csv").open()))
    observed = {(row["model_id"], int(row["split_order"])): row for row in observed_rows}
    require(len(expected) == len(observed) == 800, "M1_IMAGE_ROWS")
    component_pairs = changed_taylor = changed_effect = exact_taylor = exact_effect = 0
    metric_values = {}
    for path in sorted((M1 / "per_image").glob("*.json.gz")):
        task_id = path.name[:-8]
        require(sha(path) == expected[task_id], "M1_HASH:" + task_id)
        data = load(path); out = observed[(data["model_id"], data["split_order"])]
        zero = {row["component_id"]: row for row in data["component_rows"]["normalized_zero"]}
        blur = {row["component_id"]: row for row in data["component_rows"]["blur_rgb"]}
        require(set(zero) == set(blur), "COMPONENT_IDS")
        for identity in zero:
            component_pairs += 1
            require(float(zero[identity]["m"]) == float(blur[identity]["m"]) and float(zero[identity]["a"]) == float(blur[identity]["a"]), "ENDPOINT_INVARIANCE")
            changed_taylor += abs(float(zero[identity]["p_ref"])) != abs(float(blur[identity]["p_ref"]))
            changed_effect += float(zero[identity]["absolute_effect"]) != float(blur[identity]["absolute_effect"])
        block_values = {key: [] for key in ("taylor_reference_concordance", "finite_effect_reference_concordance", "endpoint_behavior_normalized_zero", "endpoint_behavior_blur_rgb", "taylor_behavior_normalized_zero", "taylor_behavior_blur_rgb")}
        for block in range(12):
            identities = sorted(identity for identity, row in zero.items() if row["operation"] == "head" and row["block"] == block)
            zrows = [zero[identity] for identity in identities]; brows = [blur[identity] for identity in identities]
            endpoint = [row["m"] for row in zrows]
            taylor_zero = [abs(row["p_ref"]) for row in zrows]; taylor_blur = [abs(row["p_ref"]) for row in brows]
            effect_zero = [row["absolute_effect"] for row in zrows]; effect_blur = [row["absolute_effect"] for row in brows]
            block_values["taylor_reference_concordance"].append(rho(taylor_zero, taylor_blur))
            block_values["finite_effect_reference_concordance"].append(rho(effect_zero, effect_blur))
            block_values["endpoint_behavior_normalized_zero"].append(rho(endpoint, effect_zero))
            block_values["endpoint_behavior_blur_rgb"].append(rho(endpoint, effect_blur))
            block_values["taylor_behavior_normalized_zero"].append(rho(taylor_zero, effect_zero))
            block_values["taylor_behavior_blur_rgb"].append(rho(taylor_blur, effect_blur))
            exact_taylor += ranks(taylor_zero) == ranks(taylor_blur)
            exact_effect += ranks(effect_zero) == ranks(effect_blur)
        values = {key: math.fsum(items) / 12 for key, items in block_values.items()}
        values["taylor_minus_endpoint_normalized_zero"] = values["taylor_behavior_normalized_zero"] - values["endpoint_behavior_normalized_zero"]
        values["taylor_minus_endpoint_blur_rgb"] = values["taylor_behavior_blur_rgb"] - values["endpoint_behavior_blur_rgb"]
        for key, value in values.items():
            close(out[key], value, "M1_IMAGE_VALUE:" + key)
            metric_values.setdefault((data["model_id"], key), []).append(value)
    summary = load(ROOT / "reference_reuse_summary.json")
    require(component_pairs == changed_taylor == changed_effect == 163200, "COMPONENT_SENSITIVITY_COUNTS")
    require(exact_taylor == 46 and exact_effect == 64, "EXACT_RANK_COUNTS")
    require(summary["new_inferential_tests"] == 0 and summary["head_block_reference_comparisons"] == 9600, "M1_SUMMARY_SCOPE")
    indexed = {(row["model_id"], row["metric"]): row for row in summary["summaries"]}
    require(len(indexed) == 16, "M1_SUMMARY_ROWS")
    for key, values in metric_values.items():
        row = indexed[key]
        close(row["median"], statistics.median(values), "M1_SUMMARY_MEDIAN")
        expected_ci = bootstrap(values, "M5|" + key[0] + "|" + key[1])
        close(row["ci95"][0], expected_ci[0], "M1_SUMMARY_CI_LOW")
        close(row["ci95"][1], expected_ci[1], "M1_SUMMARY_CI_HIGH")
    return len(observed_rows), component_pairs


def recompute_m2():
    manifest = load(M2 / "artifact_manifest.json")
    item = next(row for row in manifest["files"] if row["path"] == "image_values.json.gz")
    source_path = M2 / item["path"]
    require(source_path.stat().st_size == item["bytes"] and sha(source_path) == item["sha256"], "M2_SOURCE_HASH")
    source = load(source_path)
    observed_rows = list(csv.DictReader((ROOT / "m2_reuse_summary.csv").open()))
    observed = {(row["model_id"], int(row["block"]), row["reference"], row["outcome"]): row for row in observed_rows}
    require(len(observed) == 24, "M2_OUTPUT_GROUPS")
    checked = 0
    for group in source:
        model, block, geometry, reference, outcome, metric = group["key"]
        if block not in (2, 5, 10) or geometry != "patch" or outcome not in ("full", "isolated") or metric != "auc":
            continue
        values = list(group["images"].values())
        endpoint = [float(row[M2_METHODS.index("endpoint_absolute")]) for row in values]
        taylor = [float(row[M2_METHODS.index("regional_taylor")]) for row in values]
        row = observed[(model, block, reference, outcome)]
        close(row["endpoint_auc_median"], statistics.median(endpoint), "M2_ENDPOINT")
        close(row["regional_taylor_auc_median"], statistics.median(taylor), "M2_TAYLOR")
        close(row["endpoint_minus_taylor_median"], statistics.median([a - b for a, b in zip(endpoint, taylor)]), "M2_DIFFERENCE")
        checked += 1
    require(checked == 24, "M2_CHECKED_GROUPS")
    receipt = load(ROOT / "m2_reuse_receipt.json")
    require(receipt["new_tests"] == 0 and receipt["inherited_patch_taylor_decisions"] == {"full": {"endpoint_greater": 0, "baseline_greater": 11}, "isolated": {"endpoint_greater": 0, "baseline_greater": 12}}, "M2_DECISIONS")
    return checked


def verify_matrix_cost_boundary():
    matrix = load(ROOT / "method_applicability_matrix.json")
    required = {"method", "primary_question", "internal_scalar_target", "input_space_map", "external_intervention_reference", "local_jacobian_structural_closure", "finite_replacement_prediction"}
    require(matrix["schema_version"] == 1 and len(matrix["methods"]) == 7, "METHOD_COUNT")
    require(all(set(row) == required for row in matrix["methods"]), "METHOD_SCHEMA")
    require({row["method"] for row in matrix["methods"]} == {"Endpoint", "Activation-space Taylor", "Integrated Gradients", "Attention rollout", "Grad-CAM", "Chefer/AttnLRP", "Attention weights"}, "METHOD_IDENTITIES")
    cost = load(ROOT / "cost_and_information.json")
    require(cost["runtime_claim"] == "NOT_IDENTIFIABLE_FROM_COMBINED_M1_M2_PACKAGES", "RUNTIME_BOUNDARY")
    require(cost["observed_packages"]["M1"]["attempt_gpu_hours"] == load(M1 / "task_validation.json")["attempt_gpu_hours"], "M1_COST")
    require(cost["observed_packages"]["M2"]["gpu_hours"] == load(M2 / "task_validation.json")["gpu_hours"], "M2_COST")
    ep2 = cost["observed_packages"]["EP2_Path_IG_calibration"]
    require(ep2["maximum_points"] == 512 and ep2["deit_s16"]["selected_M"] == 512 and ep2["vit_b16"]["selected_M"] is None, "PATH_IG_BOUNDARY")
    receipt = load(ROOT / "analysis_receipt.json")
    require(receipt["mode"] == "CPU_ONLY_REUSE_ONLY" and receipt["cuda_visible_devices"] == "" and not receipt["torch_timm_pil_imported"], "CPU_ONLY")
    require(receipt["jpeg_reads"] == receipt["model_or_checkpoint_loads"] == receipt["new_inferential_tests"] == 0 and not receipt["new_gpu_production"], "ACCESS_BOUNDARY")


def main():
    require(not any(name in sys.modules for name in ("torch", "timm", "PIL")), "PROHIBITED_IMPORT")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_NOT_CLEARED")
    verify_sources()
    image_rows, components = recompute_m1()
    m2_groups = recompute_m2()
    verify_matrix_cost_boundary()
    report = {
        "status": "PASS", "mode": "CPU_ONLY_STANDARD_LIBRARY",
        "m1_per_image_hashes": 800, "image_rows_recomputed": image_rows,
        "component_reference_pairs_checked": components, "head_block_groups_checked": 9600,
        "bootstrap_summaries_recomputed": 16, "m2_groups_recomputed": m2_groups,
        "method_rows_checked": 7, "new_inferential_tests": 0,
        "r1_failure_chain": "PASS", "source_bindings": "PASS",
        "cpu_only": True,
    }
    path = ROOT / "independent_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
