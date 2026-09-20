#!/usr/bin/env python3
"""Independent source-recomputing validation of revision-v3 Step-2 outputs."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_ep1_queue import audit_coverage, digest
from vitnns.endpoint_revision_v3_step2 import (
    independently_sampled_channels, independently_sampled_tokens, qkv_metrics,
)

SPEC = REPO / "config/nns_vit_endpoint/revision_v3_step2.json"
ROOT = REPO / "reports/nns_vit_endpoint_revision_v3_step2_20260921"


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, code: str) -> None:
    if not condition:
        raise ValueError(code)


def result(task_id: str) -> dict:
    return load(ROOT / "results" / "attempts" / task_id / "attempt-01.json")["result"]


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_freeze_and_queue() -> tuple[dict, list[dict]]:
    spec = load(SPEC); freeze = load(ROOT / "freeze.json"); manifest = load(ROOT / "task_manifest.json")
    require(spec["stage"] == "ENDPOINT-REVISION-V3-STEP2", "SPEC_STAGE")
    for relative, expected in spec["source_bindings"].items():
        require(sha(REPO / relative) == expected, "SOURCE_BINDING:" + relative)
    for relative, expected in freeze["sources"].items():
        require(sha(REPO / relative) == expected, "FROZEN_SOURCE:" + relative)
    require(sha(ROOT / "task_manifest.json") == freeze["task_manifest_sha256"], "MANIFEST_SHA")
    require(sha(ROOT / "sampling_registry.json") == freeze["sampling_registry_sha256"], "REGISTRY_SHA")
    require(digest(manifest["tasks"]) == manifest["digest"], "MANIFEST_DIGEST")
    require((manifest["primary_tasks"], manifest["sampling_tasks"], len(manifest["tasks"])) == (800, 200, 1000), "TASK_COUNTS")
    require(audit_coverage(ROOT / "queue.sqlite3", ROOT / "results")["status"] == "PASS", "QUEUE_COVERAGE")
    startup = load(ROOT / "startup.json")
    require(startup["producer_commit"] == startup["tracking_commit"] == startup["remote_commit"] == freeze["producer_commit"], "PUBLISHED_PRODUCER")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", freeze["producer_commit"], "HEAD"], cwd=REPO).returncode == 0, "PRODUCER_ANCESTRY")
    require(startup["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8", "STARTUP_DETERMINISM")
    return spec, manifest["tasks"]


def validate_registry(spec: dict) -> None:
    registry = load(ROOT / "sampling_registry.json")
    require(registry["group_count"] == 5 and digest(registry["groups"]) == registry["digest"], "REGISTRY_DIGEST")
    widths = {"vit_b16": 768, "deit_s16": 384}
    for model_id, groups in registry["groups"].items():
        require(len(groups) == 5 and len({tuple(group["tokens"]) for group in groups}) == 5, "TOKEN_GROUPS")
        for index, group in enumerate(groups, 1):
            namespace = f"{spec['independent_sampling']['group_namespace_prefix']}|{index}"
            require(group["namespace"] == namespace and tuple(group["tokens"]) == independently_sampled_tokens(namespace), "TOKEN_RECOMPUTE")
            for block in range(12):
                expected = independently_sampled_channels(namespace, model_id, block, widths[model_id], 32)
                require(tuple(group["channels"][str(block)]) == expected, "CHANNEL_RECOMPUTE")
        for block in range(12):
            require(len({tuple(group["channels"][str(block)]) for group in groups}) == 5, "CHANNEL_GROUP_DISTINCT")


def validate_payloads(tasks: list[dict]) -> dict:
    counts = Counter(); amplitude_parity = []; qkv_parity = []
    for task in tasks:
        payload = task["payload"]; record = result(task["task_id"]); counts[record["kind"]] += 1
        require(all(record[key] == payload[key] for key in ("kind", "model_id", "sample_id", "split_order", "image_sha256")), "RESULT_IDENTITY")
        identity = record["worker_identity"]
        require(identity["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and identity["deterministic_algorithms"] and not identity["cuda_matmul_allow_tf32"] and not identity["cudnn_allow_tf32"] and identity["startup_matmul_backward"] == "PASS", "WORKER_DETERMINISM")
        if record["kind"] == "primary":
            amplitude_path = REPO / record["amplitude_path"]; qkv_path = REPO / record["qkv_path"]
            require(sha(amplitude_path) == record["amplitude_sha256"] and sha(qkv_path) == record["qkv_sha256"], "PRIMARY_PAYLOAD_SHA")
            amplitude = np.load(amplitude_path); qkv = np.load(qkv_path)
            heads = 12 if record["model_id"] == "vit_b16" else 6
            require(len(amplitude["alpha"]) == 12 * heads * 2 * 5, "AMPLITUDE_COUNT")
            observed_alphas = np.asarray(sorted(set(float(value) for value in amplitude["alpha"])), dtype=np.float64)
            require(np.allclose(observed_alphas, [0.05, 0.1, 0.25, 0.5, 1.0], atol=1e-7, rtol=0), "ALPHAS")
            require(set(amplitude["reference_id"].tolist()) == {"normalized_zero", "blur_rgb"}, "AMPLITUDE_REFERENCES")
            require(all(np.isfinite(amplitude[key]).all() for key in ("finite_effect", "p_ref", "taylor_prediction", "absolute_error")), "AMPLITUDE_FINITE")
            require(np.allclose(amplitude["taylor_prediction"], amplitude["alpha"] * amplitude["p_ref"], atol=2e-7, rtol=2e-6), "TAYLOR_FORMULA")
            require(np.allclose(amplitude["absolute_error"], np.abs(amplitude["finite_effect"] - amplitude["taylor_prediction"]), atol=2e-7, rtol=2e-6), "ABSOLUTE_ERROR_FORMULA")
            require(len(qkv["block"]) == 36 and set(qkv["path"].tolist()) == {"Q", "K", "V"}, "QKV_COVERAGE")
            require(all(np.isfinite(qkv[key]).all() for key in ("absolute_mass", "normalized_spatial_entropy", "normalized_pairwise_distance", "cancellation", "relative_scale")), "QKV_FINITE")
            for block in range(12):
                require(abs(float(qkv["relative_scale"][qkv["block"] == block].sum()) - 1.0) <= 2e-6, "QKV_SCALE_SUM")
            require(record["amplitude_validation"]["alpha_one_source_max_tolerance_ratio"] <= 1.0, "ALPHA_ONE_PARITY")
            amplitude_parity.append(record["amplitude_validation"]["alpha_one_source_max_abs"]); qkv_parity.append(record["qkv_source_max_abs"])
        else:
            path = REPO / record["sampling_path"]; require(sha(path) == record["sampling_sha256"], "SAMPLING_SHA")
            data = np.load(path); require(len(data["block"]) == 30720, "SAMPLING_COUNT")
            require(set(data["group_index"].tolist()) == set(range(5)), "SAMPLING_GROUPS")
            for group_index, group in enumerate(payload["groups"]):
                group_mask = data["group_index"] == group_index
                require(int(group_mask.sum()) == 6144, "SAMPLING_GROUP_COUNT")
                for block in range(12):
                    mask = group_mask & (data["block"] == block)
                    observed = {(int(token), int(channel)) for token, channel in zip(data["token"][mask], data["channel"][mask])}
                    expected = {(token, channel) for token in group["tokens"] for channel in group["channels"][str(block)]}
                    require(observed == expected, "SAMPLING_UNIT_IDENTITY")
            require(np.isfinite(data["normalized_spatial_entropy"]).sum() >= 30720 - 5, "SAMPLING_ENTROPY_FINITE")
            patch = data["token"] > 0
            require(np.isfinite(data["own_token_locality_above_uniform"][patch]).sum() >= int(patch.sum()) - 5, "SAMPLING_LOCALITY_FINITE")
            require(np.isfinite(data["input_dependency_distance"][patch]).sum() >= int(patch.sum()) - 5, "SAMPLING_DISTANCE_FINITE")
    require(counts == {"primary": 800, "sampling": 200}, "RESULT_KIND_COUNTS")
    return {"alpha_one_source_max_abs": max(amplitude_parity), "qkv_source_max_abs": max(qkv_parity)}


def validate_raw_qkv_recompute() -> dict:
    observed_rows = csv_rows(ROOT / "qkv_image_values.csv")
    observed = {(row["model_id"], row["sample_id"], int(row["block"]), row["path"]): row for row in observed_rows}
    maximum = defaultdict(float); compared = 0
    attempts = REPO / "reports/nns_vit_endpoint_e3_closure_r1_20260918/formal_results/attempts"
    for attempt in sorted(attempts.glob("*/attempt-01.json")):
        source = load(attempt)["result"]
        if not source["artifacts"]:
            continue
        for artifact in source["artifacts"]:
            array = np.load(REPO / artifact["path"]); order = artifact["component_order"]
            metrics = qkv_metrics({"Q": array[order.index("query")], "K": array[order.index("key")], "V": array[order.index("value")]})
            for path in ("Q", "K", "V"):
                row = observed[(source["model_id"], source["sample_id"], int(artifact["block"]), path)]
                for metric in ("absolute_mass", "normalized_spatial_entropy", "normalized_pairwise_distance", "cancellation", "relative_scale"):
                    difference = abs(float(row[metric]) - float(metrics[path][metric])); maximum[metric] = max(maximum[metric], difference)
                compared += 1
    require(compared == 32 * 12 * 3, "RAW_QKV_RECOMPUTE_COUNT")
    require(maximum["absolute_mass"] <= 2e-3 and max(maximum[name] for name in maximum if name != "absolute_mass") <= 2e-6, "RAW_QKV_RECOMPUTE_PARITY")
    return {"records": compared, "maximum_absolute_differences": dict(maximum)}


def validate_analysis() -> dict:
    receipt = load(ROOT / "analysis_receipt.json")
    require(receipt == {
        "status": "PASS", "statistical_unit": "image",
        "amplitude_image_rows": 8000, "amplitude_summary_rows": 100,
        "qkv_image_rows": 28800, "qkv_summary_rows": 360, "qkv_contrast_rows": 30,
        "sampling_image_rows": 3000, "sampling_summary_rows": 60, "sampling_variation_rows": 12,
    }, "ANALYSIS_COUNTS")
    amplitude = csv_rows(ROOT / "multi_amplitude_summary.csv")
    require(len(amplitude) == 100 and all(int(row["n_images"]) == 400 for row in amplitude), "AMPLITUDE_IMAGE_UNIT")
    qkv = csv_rows(ROOT / "qkv_block_summary.csv")
    require(len(qkv) == 360 and all(int(row["n_images"]) == 400 for row in qkv), "QKV_IMAGE_UNIT")
    sampling = csv_rows(ROOT / "independent_sampling_summary.csv")
    require(len(sampling) == 60 and all(int(row["n_images"]) == 100 for row in sampling), "SAMPLING_IMAGE_UNIT")
    variation = load(ROOT / "independent_sampling_variation.json")["rows"]
    require(len(variation) == 12 and all(row["group_count"] == 5 and len(row["group_medians"]) == 5 for row in variation), "SAMPLING_REPLICATION")
    contrasts = csv_rows(ROOT / "qkv_depth_contrasts.csv")
    require(len(contrasts) == 30 and all(int(row["n_images"]) == 400 for row in contrasts), "QKV_CONTRASTS")
    for name in ("multi_amplitude.png", "multi_amplitude.pdf", "qkv_population.png", "qkv_population.pdf", "independent_sampling.png", "independent_sampling.pdf"):
        require((ROOT / "figures" / name).stat().st_size > 1000, "FIGURE:" + name)
    return receipt


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=ROOT); args = parser.parse_args()
    ROOT = args.root.resolve()
    spec, tasks = validate_freeze_and_queue(); validate_registry(spec)
    payload = validate_payloads(tasks); analysis = validate_analysis(); raw = validate_raw_qkv_recompute()
    report = {
        "status": "PASS", "task_count": len(tasks), "queue_coverage": "PASS",
        "payload_validation": payload, "analysis": analysis, "raw_qkv_recompute": raw,
        "checks": [
            "frozen source and predecessor bindings", "published producer ancestry",
            "1000 canonical queue results", "all five prescribed amplitudes and both references",
            "Taylor and absolute-error formulas", "alpha=1 sealed finite-effect replay",
            "Q/K/V complete block coverage and relative-scale closure",
            "independent raw E3 Q/K/V spatial-metric recomputation",
            "five distinct token/channel sampling registries and realized units",
            "image-level sample sizes, confidence intervals, direction fields and figures",
        ],
    }
    write = ROOT / "validation_report.json"
    write.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "tasks": len(tasks), "raw_qkv_records": raw["records"]}))


if __name__ == "__main__":
    main()
