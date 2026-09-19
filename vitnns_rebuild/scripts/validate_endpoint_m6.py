#!/usr/bin/env python3
"""Independent CPU-only validation of Endpoint M6 selection and figures."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m6_r1_20260919"
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from run_endpoint_m6 import select_m2, select_m3, select_m4, select_model_difference  # noqa: E402


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, code):
    if not condition:
        raise ValueError(code)


def selected_identity(case):
    def identity(row):
        return tuple(row.get(key) for key in ("task_id", "model_id", "sample_id", "block", "token", "channel", "head", "residual_type"))
    if "selected_units" in case: return tuple(identity(row) for row in case["selected_units"])
    if "selected_heads" in case: return tuple(identity(row) for row in case["selected_heads"])
    if "selected_residuals" in case: return tuple(identity(row) for row in case["selected_residuals"])
    return (identity(case["selected_boundary"]),)


def m3_key(row):
    return "m3__" + "__".join(map(str, (row["task_id"], row["block"], row["token"], row["channel"])))


def main():
    require(not any(name in sys.modules for name in ("torch", "timm")), "CPU_ONLY_IMPORTS")
    spec = load(REPO / "config/nns_vit_endpoint/endpoint_m6.json"); selection = load(ROOT / "selection_manifest.json")
    for relative, expected in spec["source_bindings"].items(): require(sha(REPO / relative) == expected, "SOURCE_BINDING:" + relative)
    access = {}; m3, tasks, paths = select_m3(spec, access); independent = list(m3)
    independent.append(select_model_difference(tasks, paths, access)); independent.extend(select_m4(spec, access)); independent.append(select_m2(access))
    expected = {row["case_id"]: selected_identity(row) for row in independent}; observed = {row["case_id"]: selected_identity(row) for row in selection["cases"]}
    require(expected == observed and len(expected) == selection["case_count"] == 12, "INDEPENDENT_SELECTION")
    access_manifest = load(ROOT / "source_access_manifest.json"); recorded = {row["path"]: row for row in access_manifest["files"]}
    require(set(recorded) == set(access) and len(recorded) == access_manifest["file_count"], "ACCESS_SET")
    for relative, row in recorded.items():
        path = REPO / relative; require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "ACCESSED_SOURCE:" + relative)
    recompute = load(ROOT / "map_recompute_receipt.json")
    require(recompute["status"] == "PASS" and recompute["selection_manifest_sha256"] == sha(ROOT / "selection_manifest.json"), "RECOMPUTE_RECEIPT")
    require(recompute["worker_identity"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and recompute["worker_identity"]["deterministic_algorithms"]
            and not recompute["worker_identity"]["cuda_matmul_allow_tf32"] and not recompute["worker_identity"]["cudnn_allow_tf32"], "NUMERIC_IDENTITY")
    expected_m3_keys = {m3_key(unit) for case in selection["cases"] for unit in case.get("selected_units", [])}
    with np.load(ROOT / "raw_arrays.npz") as raw:
        keys = set(raw.files); maximum = 0.0; closure_count = 0
        for case_id in ("m4_routing_dominant", "m4_value_dominant", "m4_high_frozen_attention_error"):
            q, k, routing, value, full = (raw[case_id + "__" + name].astype(np.float64) for name in ("q", "k", "routing", "value", "full"))
            maximum = max(maximum, float(np.max(np.abs(q + k - routing))), float(np.max(np.abs(routing + value - full)))); closure_count += 2
        for case_id in ("m4_identity_dominant", "m4_branch_dominant", "m4_strong_path_cancellation"):
            identity, branch, output = (raw[case_id + "__" + name].astype(np.float64) for name in ("identity", "branch", "output"))
            maximum = max(maximum, float(np.max(np.abs(identity + branch - output)))); closure_count += 1
        require(maximum <= 1e-6, "RAW_MAP_CLOSURE")
        require({key for key in keys if key.startswith("m3__")} == expected_m3_keys, "M3_MAP_SET")
        require(sum(key.endswith("__attention_cls") for key in keys) == 3, "ATTENTION_MAP_COUNT")
        require(recompute["array_count"] == len(keys), "ARRAY_COUNT")
    figures = load(ROOT / "figure_manifest.json")
    require(len(figures["figures"]) == 5 and {row["figure_id"] for row in figures["figures"]} == set(spec["visualization"]["required_figures"]), "FIGURE_COVERAGE")
    figure_files = 0
    for figure in figures["figures"]:
        require({row["format"] for row in figure["files"]} == {"png", "pdf"}, "FIGURE_FORMATS")
        for row in figure["files"]:
            report, paper = REPO / row["report_path"], REPO / row["paper_path"]
            require(sha(report) == sha(paper) == row["sha256"] and report.stat().st_size == paper.stat().st_size == row["bytes"], "FIGURE_HASH")
            if row["format"] == "png":
                with Image.open(report) as image: require(image.width >= 1000 and image.height >= 300, "FIGURE_DIMENSION")
            figure_files += 1
    validation = {"status": "PASS_R2", "mode": "CPU_ONLY", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                  "amendment": "docs/endpoint_m6_validation_amendment_20260919.md",
                  "independently_reselected_cases": 12, "accessed_source_files": len(recorded), "raw_array_count": recompute["array_count"],
                  "unique_m3_map_keys": len(expected_m3_keys),
                  "recompute_summary_checks": recompute["recomputed_map_checks"], "raw_closure_checks": closure_count,
                  "maximum_raw_closure_error": maximum, "figures": 5, "figure_files": figure_files, "new_inferential_tests": 0}
    path = ROOT / "independent_validation.json"
    if path.exists(): require(load(path) == validation, "VALIDATION_DRIFT")
    else: path.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
