#!/usr/bin/env python3
"""Independent source-recomputing audit of Endpoint M4 Review R2."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m4_review_r2_20260919"
M4 = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"
M1 = REPO / "reports/nns_vit_endpoint_m1_r1_20260918"
E3 = REPO / "reports/nns_vit_endpoint_e3_closure_r1_20260918"
SPEC = REPO / "config/nns_vit_endpoint/endpoint_m4_review_r1.json"


def load(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(condition, code):
    if not condition:
        raise ValueError(code)


def close(left, right, code, atol=1e-10):
    require((left is None and right is None) or
            (left is not None and right is not None and math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=atol)), code)


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


def patch_distribution(array):
    patch = np.abs(np.asarray(array, dtype=np.float64)).reshape(3, 14, 16, 14, 16).sum((0, 2, 4)).reshape(-1)
    return None if patch.sum() == 0 else patch / patch.sum()


def entropy(p):
    positive = p[p > 0]
    return float(-(positive * np.log(positive)).sum() / math.log(196))


def csv_rows(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    require(not any(name in sys.modules for name in ("torch", "timm")), "CPU_ONLY_IMPORTS")
    spec = load(SPEC)
    require(spec["stage"] == "ENDPOINT-M4-REVIEW-R2" and spec["runtime"]["CPU_only"], "SPEC_STATUS")
    for relative, expected in spec["source_bindings"].items():
        require(sha(REPO / relative) == expected, "SOURCE_BINDING:" + relative)
    failure_path = REPO / spec["recovery_predecessor"]["root"] / "failure_receipt.json"
    require(sha(failure_path) == spec["recovery_predecessor"]["failure_receipt_sha256"], "R1_FAILURE_HASH")
    failure = load(failure_path)
    require(failure["status"] == "FAIL_IMPLEMENTATION_NAMEERROR" and not failure["scientific_results_released"], "R1_FAILURE_STATUS")
    freeze = load(ROOT / "freeze.json")
    for relative, expected in freeze["sources"].items():
        require(sha(REPO / relative) == expected, "FROZEN_SOURCE:" + relative)
    receipt = load(ROOT / "analysis_receipt.json")
    require(receipt["status"] == "PASS_POST_HOC_REVIEW_ANALYSIS" and receipt["mode"] == "CPU_ONLY", "ANALYSIS_RECEIPT")
    require(not receipt["torch_or_timm_imported"] and receipt["jpeg_or_model_reads"] == 0 and receipt["new_primary_M4_tests"] == 0, "ANALYSIS_BOUNDARY")

    metric_manifest = load(M1 / "metrics_manifest.json")
    m1_hashes = {row["task_id"]: row["sha256"] for row in metric_manifest["records"]}
    require(len(m1_hashes) == 800, "M1_MANIFEST_COUNT")
    task_manifest = load(M4 / "task_manifest.json")["tasks"]
    m4_results = {}
    for task in task_manifest:
        task_id = task["task_id"]
        result_path = next((M4 / "results/attempts" / task_id).glob("attempt-*.json"))
        envelope = load(result_path); unsigned = dict(envelope); signature = unsigned.pop("canonical_digest")
        require(digest(unsigned) == signature, "M4_RESULT_DIGEST:" + task_id)
        m4_results[task_id] = envelope["result"]
        m1_path = M1 / "per_image" / f"{task_id}.json.gz"
        require(sha(m1_path) == m1_hashes[task_id], "M1_RESULT_HASH:" + task_id)
    require(len(m4_results) == 800, "M4_RESULT_COUNT")

    jvp_values = csv_rows("jvp_image_block_values.csv"); jvp_summary = csv_rows("jvp_block_summary.csv")
    require(len(jvp_values) == 38400 and len(jvp_summary) == 96, "JVP_COUNTS")
    index = {(row["task_id"], int(row["block"]), row["metric"]): float(row["image_head_median"]) for row in jvp_values}
    independent_jvp = 0
    for task_id, result in m4_results.items():
        for block in range(12):
            heads = [row for row in result["attention"] if row["block"] == block]
            expected = {
                "relative_error": median([row["frozen_attention_jvp_error"] for row in heads]),
                "absolute_error": median([row["frozen_attention_jvp_absolute_error"] for row in heads]),
                "full_response_denominator": median([math.sqrt(sum(direction["full_l2"] ** 2 for direction in row["direction_records"])) for row in heads]),
                "routing_numerator": median([math.sqrt(sum(direction["routing_l2"] ** 2 for direction in row["direction_records"])) for row in heads]),
            }
            for metric, value in expected.items():
                close(index[(task_id, block, metric)], value, "JVP_VALUE:" + metric); independent_jvp += 1
    for model in ("vit_b16", "deit_s16"):
        for metric in ("relative_error", "absolute_error", "full_response_denominator", "routing_numerator"):
            row = next(row for row in jvp_summary if row["model_id"] == model and row["block"] == "11" and row["metric"] == metric)
            observed = [value for (task, block, name), value in index.items() if m4_results[task]["model_id"] == model and block == 11 and name == metric]
            close(row["median"], np.median(observed), "JVP_SUMMARY:" + metric)
            close(row["q10"], np.quantile(observed, .1), "JVP_Q10:" + metric)

    sat_phase = csv_rows("saturation_image_phase_values.csv")
    sat_tests = load(ROOT / "saturation_sensitivity_tests.json")["tests"]
    require(len(sat_phase) == 57600 and len(sat_tests) == 36, "SATURATION_COUNTS")
    for model in ("vit_b16", "deit_s16"):
        family = [row for row in sat_tests if row["model_id"] == model]
        require(len(family) == 18, "SATURATION_FAMILY")
        ordered = sorted(family, key=lambda row: (row["p"], row["test_id"])); running = 0
        for rank, row in enumerate(ordered):
            running = max(running, min(1., (18 - rank) * row["p"]))
            close(row["holm_p"], running, "SATURATION_HOLM")
            values = [None if source["rho"] == "" else float(source["rho"]) for source in sat_phase
                      if source["model_id"] == model and source["phase"] == "all"
                      and "|".join((source["saturation"], source["outcome"], source["adjustment"])) == row["test_id"]]
            close(row["median"], median(values), "SATURATION_MEDIAN")

    residual_values = csv_rows("residual_spatial_image_values.csv")
    residual_tests = load(ROOT / "residual_spatial_tests.json")["tests"]
    require(len(residual_values) == 1600 and len(residual_tests) == 4, "RESIDUAL_COUNTS")
    for row in residual_tests:
        values = [float(source["late_minus_early_spatial_tv"]) for source in residual_values
                  if source["model_id"] == row["model_id"] and source["residual_type"] == row["residual_type"]]
        close(row["median"], np.median(values), "RESIDUAL_MEDIAN")
        sign = {"positive": sum(value > 1e-12 for value in values), "negative": sum(value < -1e-12 for value in values)}
        require(row["positive"] == sign["positive"] and row["negative"] == sign["negative"], "RESIDUAL_SIGNS")

    bridge_values = csv_rows("bridge_raw_map_values.csv")
    bridge_summary = csv_rows("bridge_block_summary.csv")
    require(len(bridge_values) == 384 and len(bridge_summary) == 120, "BRIDGE_COUNTS")
    raw_checked = 0
    for model in ("vit_b16", "deit_s16"):
        for order in range(16):
            task_id = f"{model}__{order:03d}"
            envelope = load(next((E3 / "formal_results/attempts" / task_id).glob("attempt-*.json")))
            unsigned = dict(envelope); signature = unsigned.pop("canonical_digest")
            require(digest(unsigned) == signature, "E3_RESULT_DIGEST:" + task_id)
            for artifact in envelope["result"]["artifacts"]:
                path = REPO / artifact["path"]; require(sha(path) == artifact["sha256"], "RAW_MAP_HASH")
                array = np.load(path, mmap_mode="r"); order_map = {name: artifact["component_order"].index(name) for name in ("attention", "value")}
                full, value = patch_distribution(array[order_map["attention"]]), patch_distribution(array[order_map["value"]])
                observed = next(row for row in bridge_values if row["task_id"] == task_id and int(row["block"]) == artifact["block"])
                close(observed["full_entropy"], entropy(full), "BRIDGE_ENTROPY")
                close(observed["full_value_patch_tv"], .5 * np.abs(full - value).sum(), "BRIDGE_TV")
                raw_checked += 1
    require(raw_checked == 384, "RAW_CHECK_COUNT")

    numeric = load(ROOT / "numeric_error_locations.json")["records"]
    require(len(numeric) == 4 and {row["metric"] for row in numeric} ==
            {"qkv_closure_max_abs", "frozen_identity_max_abs", "explicit_jvp_relative_error", "residual_closure_max_abs"}, "NUMERIC_LOCATIONS")
    figure = load(ROOT / "figure_manifest.json")["figures"]
    require(len(figure) == 8, "FIGURE_COUNT")
    for row in figure:
        report, paper = REPO / row["report_path"], REPO / row["paper_path"]
        require(sha(report) == sha(paper) == row["sha256"] and report.stat().st_size == paper.stat().st_size == row["bytes"], "FIGURE_HASH")
    report = {"status": "PASS", "mode": "CPU_ONLY", "m4_results": len(m4_results),
              "m1_control_files_hash_verified": len(m1_hashes), "independent_jvp_values": independent_jvp,
              "saturation_phase_rows": len(sat_phase), "saturation_tests": len(sat_tests),
              "residual_image_rows": len(residual_values), "residual_tests": len(residual_tests),
              "raw_map_hashes_and_metrics_recomputed": raw_checked, "figure_files": len(figure),
              "source_bindings": "PASS", "r1_failure_chain": "PASS"}
    path = ROOT / "independent_validation.json"
    if path.exists():
        require(load(path) == report, "VALIDATION_DRIFT")
    else:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
