#!/usr/bin/env python3
"""Independent CPU-only exhaustive validator for Endpoint M4."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"
SPEC = REPO / "config/nns_vit_endpoint/endpoint_m4.json"


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(condition, code: str):
    if not condition:
        raise ValueError(code)


def close(left, right, code: str, atol=1e-11):
    require((left is None and right is None) or (left is not None and right is not None and math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=atol)), code)


def median(values):
    values = [float(value) for value in values if value not in (None, "") and math.isfinite(float(value))]
    return None if not values else float(np.median(values))


def spearman(left, right):
    pairs = [(float(a), float(b)) for a, b in zip(left, right) if a is not None and b is not None]
    if len(pairs) < 4:
        return None
    a, b = (np.asarray(value) for value in zip(*pairs))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def independent_contrasts(result: dict) -> dict:
    attention = result["attention"]
    block = {}
    for index in range(12):
        heads = [row for row in attention if row["block"] == index]
        for metric in ("routing_share", "frozen_attention_endpoint_error"):
            block[(index, metric)] = median(row[metric] for row in heads)
        block[(index, "saturation_routing")] = spearman(
            [row["mean_query_saturation"] for row in heads], [row["routing_share"] for row in heads])
        block[(index, "saturation_error")] = spearman(
            [row["mean_query_saturation"] for row in heads], [row["frozen_attention_endpoint_error"] for row in heads])
    answer = {}
    for metric in ("routing_share", "frozen_attention_endpoint_error"):
        early = median(block[(index, metric)] for index in range(4))
        late = median(block[(index, metric)] for index in range(8, 12))
        answer[f"attention_{metric}_late_minus_early"] = None if early is None or late is None else late - early
    answer["saturation_routing"] = median(block[(index, "saturation_routing")] for index in range(12))
    answer["saturation_error"] = median(block[(index, "saturation_error")] for index in range(12))
    for kind in ("attention", "mlp"):
        rows = {row["block"]: row for row in result["residual"] if row["residual_type"] == kind}
        for metric in ("identity_share", "path_cancellation"):
            early = median(rows[index][metric] for index in range(4))
            late = median(rows[index][metric] for index in range(8, 12))
            answer[f"{kind}_{metric}_late_minus_early"] = None if early is None or late is None else late - early
    return answer


def main():
    require(not any(name in sys.modules for name in ("torch", "timm", "PIL")), "CPU_ONLY_IMPORTS")
    spec = load(SPEC)
    for relative, expected in spec["source_bindings"].items():
        require(sha(REPO / relative) == expected, "SOURCE_BINDING:" + relative)
    manifest = load(ROOT / "task_manifest.json")
    tasks = manifest["tasks"]
    require(len(tasks) == manifest["task_count"] == 800 and canonical_digest(tasks) == manifest["digest"], "TASK_MANIFEST")
    task_by_id = {row["task_id"]: row["payload"] for row in tasks}
    require(len(task_by_id) == 800, "TASK_IDS")
    with sqlite3.connect(ROOT / "queue.sqlite3") as connection:
        queue = connection.execute("SELECT task_id,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id").fetchall()
        attempts = connection.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status").fetchall()
    require(len(queue) == 800 and all(row[1] == "completed" for row in queue), "QUEUE_COMPLETION")
    require(dict(attempts).get("COMPLETED") == 800, "ACCEPTED_ATTEMPTS")
    source_by_id = {}; maximum = {
        "qkv_closure_max_abs": 0.0, "frozen_identity_max_abs": 0.0,
        "residual_closure_max_abs": 0.0, "explicit_jvp_relative_error": 0.0,
        "detach_gradient_max_abs": 0.0, "e3_tolerance_ratio": 0.0,
    }
    counts = {"attention_heads": 0, "residuals": 0, "jvp_direction_heads": 0,
              "negative_controls": 0, "detach_blocks": 0, "e3_scalar_crosschecks": 0,
              "na": 0}
    for task_id, _state, attempt, relative, result_digest in queue:
        envelope = load(ROOT / "results" / relative)
        unsigned = dict(envelope); observed = unsigned.pop("canonical_digest")
        require(canonical_digest(unsigned) == observed == result_digest, "RESULT_DIGEST:" + task_id)
        require(envelope["task_id"] == task_id and envelope["attempt"] == attempt, "RESULT_IDENTITY:" + task_id)
        row = envelope["result"]; task = task_by_id[task_id]
        require((row["model_id"], row["sample_id"], row["class_id"], row["image_sha256"]) ==
                (task["model_id"], task["sample_id"], task["class_id"], task["image_sha256"]), "TASK_RESULT_BINDING")
        heads = 12 if row["model_id"] == "vit_b16" else 6
        require({(item["block"], item["head"]) for item in row["attention"]} ==
                {(block, head) for block in range(12) for head in range(heads)}, "ATTENTION_SCOPE")
        require({(item["block"], item["residual_type"]) for item in row["residual"]} ==
                {(block, kind) for block in range(12) for kind in ("attention", "mlp")}, "RESIDUAL_SCOPE")
        require(len(row["attention_detach_validation"]) == (12 if task["detach_validation"] else 0), "DETACH_SCOPE")
        for item in row["attention"]:
            denominator = item["routing"]["absolute_mass"] + item["value"]["absolute_mass"]
            close(item["routing_share"], None if denominator == 0 else item["routing"]["absolute_mass"] / denominator, "ROUTING_SHARE")
            close(item["routing_value_path_cancellation"], None if denominator == 0 else 1 - item["full"]["absolute_mass"] / denominator, "ATTENTION_CANCELLATION")
            require(item["qkv_native_closure"]["pass"] and item["frozen_attention_identity"]["pass"], "ATTENTION_CLOSURE")
            require(item["directions"] == 8 and [value["direction"] for value in item["direction_records"]] == list(range(8)), "JVP_DIRECTIONS")
            require(item["explicit_jvp_relative_error"] is None or item["explicit_jvp_relative_error"] <= 1e-4, "JVP_TOLERANCE")
            maximum["qkv_closure_max_abs"] = max(maximum["qkv_closure_max_abs"], item["qkv_native_closure"]["max_abs"])
            maximum["frozen_identity_max_abs"] = max(maximum["frozen_identity_max_abs"], item["frozen_attention_identity"]["max_abs"])
            maximum["explicit_jvp_relative_error"] = max(maximum["explicit_jvp_relative_error"], item["explicit_jvp_relative_error"] or 0)
            counts["attention_heads"] += 1; counts["jvp_direction_heads"] += 8
            counts["na"] += sum(value is None for key, value in item.items() if key in {"routing_share", "frozen_attention_endpoint_error", "contracted_gradient_error", "routing_value_cosine", "frozen_attention_jvp_error"})
        for item in row["residual"]:
            denominator = item["identity"]["absolute_mass"] + item["branch"]["absolute_mass"]
            close(item["identity_share"], None if denominator == 0 else item["identity"]["absolute_mass"] / denominator, "IDENTITY_SHARE")
            close(item["branch_share"], None if denominator == 0 else item["branch"]["absolute_mass"] / denominator, "BRANCH_SHARE")
            close(item["path_cancellation"], None if denominator == 0 else 1 - item["output"]["absolute_mass"] / denominator, "RESIDUAL_CANCELLATION")
            require(item["closure"]["pass"] and item["negative_control"]["passed"] and item["negative_control"]["observed_failure_max_abs"] > 0, "RESIDUAL_VALIDATION")
            maximum["residual_closure_max_abs"] = max(maximum["residual_closure_max_abs"], item["closure"]["max_abs"])
            counts["residuals"] += 1; counts["negative_controls"] += 1
        for item in row["attention_detach_validation"]:
            require(item["attention_detach_value_gradient"]["pass"] and item["forward_parity"]["pass"], "DETACH_VALIDATION")
            maximum["detach_gradient_max_abs"] = max(maximum["detach_gradient_max_abs"], item["attention_detach_value_gradient"]["max_abs"])
            counts["detach_blocks"] += 1
        require(row["e3_crosscheck"]["status"] == "PASS" and row["e3_crosscheck"]["scalar_checks"] == 288, "E3_CROSSCHECK")
        counts["e3_scalar_crosschecks"] += 288
        maximum["e3_tolerance_ratio"] = max(maximum["e3_tolerance_ratio"], row["e3_crosscheck"]["maximum_tolerance_ratio"])
        identity = row["worker_identity"]
        require(identity["startup_matmul_backward"] == "PASS" and identity["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and identity["deterministic_algorithms"] and not identity["cuda_matmul_allow_tf32"] and not identity["cudnn_allow_tf32"] and not identity["autocast"], "WORKER_IDENTITY")
        source_by_id[task_id] = row
    require(counts["attention_heads"] == 86400 and counts["residuals"] == 19200 and counts["jvp_direction_heads"] == 691200, "TOTAL_COUNTS")
    require(counts["negative_controls"] == 19200 and counts["detach_blocks"] == 384 and counts["e3_scalar_crosschecks"] == 230400, "VALIDATION_COUNTS")
    image_rows = list(csv.DictReader((ROOT / "image_values.csv").open()))
    indexed = {(row["model_id"], row["sample_id"], row["domain"], row["group"], row["block"], row["metric"]): row["value"] for row in image_rows}
    require(len(indexed) == len(image_rows), "IMAGE_VALUE_DUPLICATE")
    independent = 0
    for task_id, result in source_by_id.items():
        expected = independent_contrasts(result)
        for metric, value in expected.items():
            observed = indexed[(result["model_id"], result["sample_id"], "contrast", "image", "", metric)]
            close(None if observed == "" else float(observed), value, "INDEPENDENT_IMAGE_CONTRAST:" + metric)
            independent += 1
    tests = load(ROOT / "m4_tests.json")
    require(tests["test_count"] == len(tests["tests"]) == 16, "TEST_COUNT")
    for model in ("vit_b16", "deit_s16"):
        attention = [row for row in tests["tests"] if row["model_id"] == model and row["domain"] == "attention"]
        require(len(attention) == 4 and len({row["test_id"] for row in attention}) == 4, "ATTENTION_HOLM_FAMILY")
        for kind in ("attention", "mlp"):
            residual = [row for row in tests["tests"] if row["model_id"] == model and row["domain"] == "residual" and row["residual_type"] == kind]
            require(len(residual) == 2 and len({row["test_id"] for row in residual}) == 2, "RESIDUAL_HOLM_FAMILY")
    analysis = load(ROOT / "analysis_receipt.json")
    require(analysis["mode"] == "CPU_ONLY" and analysis["cuda_visible_devices"] == "" and analysis["prohibited_imports_absent"], "ANALYSIS_BOUNDARY")
    a1 = load(ROOT / "m4_a1_analysis_receipt.json")
    require(a1["status"] == "PASS" and a1["mode"] == "CPU_ONLY" and a1["task_count"] == 800
            and a1["new_inferential_tests"] == 0 and a1["cuda_visible_devices"] == "", "A1_ANALYSIS")
    with (ROOT / "m4_a1_descriptive_summary.csv").open() as stream:
        a1_rows = list(csv.DictReader(stream))
    require(any(row["domain"] == "attention" and row["metric"] == "signed_magnitude" for row in a1_rows), "A1_SIGNED")
    require(any(row["domain"] == "residual_relation" and row["metric"] == "branch_share_path_cancellation_spearman" for row in a1_rows), "A1_RELATION")
    require(any(row["domain"] == "residual_type_contrast" for row in a1_rows), "A1_TYPE_CONTRAST")
    report = {"status": "PASS", "task_count": 800, **counts, "maximum_errors": maximum,
              "independent_image_contrasts": independent, "primary_tests": 16,
              "a1_descriptive_summary_rows": len(a1_rows), "source_bindings": "PASS",
              "queue_and_result_digests": "PASS", "cpu_only": True}
    path = ROOT / "validation_report.json"
    if path.exists():
        require(load(path) == report, "VALIDATION_REPORT_DRIFT")
    else:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
