#!/usr/bin/env python3
"""Freeze sources and select all Endpoint M6 cases without viewing images."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m6_r1_20260919"
SPEC = REPO / "config/nns_vit_endpoint/endpoint_m6.json"
M3_ROOT = REPO / "reports/nns_vit_endpoint_m3_r1_20260919"
M4_ROOT = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"
M2_ROOT = REPO / "reports/nns_vit_endpoint_m2_r1_20260919"
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_m6 import patch_metrics, select_nearest_median, select_tail, spearman  # noqa: E402


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, code):
    if not condition:
        raise ValueError(code)


def artifact_index(root):
    manifest = load(root / "artifact_manifest.json")
    return {row["path"]: row for row in manifest["files"]}


def verified_artifact(path, root, index, access, stage):
    relative = path.relative_to(root).as_posix()
    require(relative in index, "UNSEALED_SOURCE:" + path.as_posix())
    row = index[relative]; observed = sha(path)
    require(path.stat().st_size == row["bytes"] and observed == row["sha256"], "SOURCE_ARTIFACT_DRIFT:" + path.as_posix())
    access[path.relative_to(REPO).as_posix()] = {"sha256": observed, "bytes": path.stat().st_size, "source_stage": stage}


def m3_row(task, path, index):
    with np.load(path) as source:
        row = {key: source[key][index].item() if source[key].ndim == 1 else source[key][index].tolist()
               for key in source.files}
    return {
        "task_id": task["task_id"], "model_id": task["payload"]["model_id"],
        "sample_id": task["payload"]["sample_id"], "image_sha256": task["payload"]["image_sha256"],
        "block": int(row["block"]), "token": int(row["token"]), "channel": int(row["channel"]),
        "signed_sum": float(row["signed_sum"]), "absolute_mass": float(row["absolute_mass"]),
        "l2_norm": float(row["l2_norm"]), "cancellation": float(row["cancellation"]),
        "normalized_spatial_entropy": float(row["normalized_spatial_entropy"]),
        "zero_gradient": bool(row["zero_gradient"]),
        **patch_metrics(row["patch_absolute_mass_196"], int(row["token"])),
        "patch_absolute_mass_196": row["patch_absolute_mass_196"],
        "patch_signed_sum_196": row["patch_signed_sum_196"],
        "source_path": path.relative_to(REPO).as_posix(), "source_sha256": sha(path),
    }


def numpy_tail(values, task_indexes, row_indexes, tasks, paths, metric, tail, fraction):
    values = np.concatenate(values).astype(np.float64, copy=False)
    task_indexes = np.concatenate(task_indexes); row_indexes = np.concatenate(row_indexes)
    require(len(values) == len(task_indexes) == len(row_indexes) and np.isfinite(values).all(), "M3_CANDIDATES")
    count = len(values); nominal = max(1, int(math.ceil(fraction * count)))
    threshold = float(np.partition(values, count - nominal)[count - nominal] if tail == "highest" else np.partition(values, nominal - 1)[nominal - 1])
    eligible = np.flatnonzero(values >= threshold if tail == "highest" else values <= threshold)
    def key(i):
        task = tasks[int(task_indexes[i])]; payload = task["payload"]
        with np.load(paths[int(task_indexes[i])]) as source:
            row = int(row_indexes[i])
            return (payload["sample_id"], payload["model_id"], int(source["block"][row]),
                    int(source["token"][row]), int(source["channel"][row]), "", "")
    chosen = min((int(i) for i in eligible), key=key); ti, ri = int(task_indexes[chosen]), int(row_indexes[chosen])
    return {
        "metric": metric, "tail": tail, "fraction": fraction, "universe_count": count,
        "nominal_tail_count": nominal, "inclusive_candidate_count": int(len(eligible)),
        "threshold": threshold, "selected_value": float(values[chosen]),
        "selected": m3_row(tasks[ti], paths[ti], ri),
    }


def select_m3(spec, access):
    tasks = load(M3_ROOT / "task_manifest.json")["tasks"]
    manifest = artifact_index(M3_ROOT); paths = []
    buckets = {name: {"v": [], "t": [], "r": []} for name in ("early_local", "late_entropy", "cancellation", "cls_patch")}
    for task_index, task in enumerate(tasks):
        path = REPO / ("reports/nns_vit_endpoint_m3_r1_20260919/task_payloads/" + task["task_id"] + "/metrics.npz")
        verified_artifact(path, M3_ROOT, manifest, access, "M3"); paths.append(path)
        with np.load(path) as source:
            block, token, channel = source["block"], source["token"], source["channel"]
            row_index = np.arange(len(block), dtype=np.int32)
            cancellation = source["cancellation"].astype(np.float64); mask = np.isfinite(cancellation)
            for key, values, use in [
                ("cancellation", cancellation, mask),
                ("late_entropy", source["normalized_spatial_entropy"].astype(np.float64), (token > 0) & (block >= 8)),
            ]:
                buckets[key]["v"].append(values[use]); buckets[key]["t"].append(np.full(use.sum(), task_index, np.int16)); buckets[key]["r"].append(row_index[use])
            mass = source["patch_absolute_mass_196"].astype(np.float64)
            totals = mass.sum(axis=1); locality = np.full(len(block), np.nan, dtype=np.float64)
            for token_id in sorted(set(int(x) for x in token if x > 0)):
                row, column = divmod(token_id - 1, 14); rr, cc = np.indices((14, 14))
                neighborhood = (np.maximum(abs(rr - row), abs(cc - column)) <= 1).reshape(-1)
                use = (token == token_id) & (totals > 0)
                locality[use] = mass[use][:, neighborhood].sum(axis=1) / totals[use]
            use = (token > 0) & (block <= 3) & np.isfinite(locality)
            buckets["early_local"]["v"].append(locality[use]); buckets["early_local"]["t"].append(np.full(use.sum(), task_index, np.int16)); buckets["early_local"]["r"].append(row_index[use])
            lookup = {(int(b), int(t), int(c)): i for i, (b, t, c) in enumerate(zip(block, token, channel))}
            values, indexes = [], []
            entropy = source["normalized_spatial_entropy"]
            for b in range(12):
                for c in sorted(set(int(x) for x in channel[block == b])):
                    left, right = lookup[(b, 0, c)], lookup[(b, 30, c)]
                    values.append(float(entropy[left]) - float(entropy[right])); indexes.append(right)
            buckets["cls_patch"]["v"].append(np.asarray(values)); buckets["cls_patch"]["t"].append(np.full(len(indexes), task_index, np.int16)); buckets["cls_patch"]["r"].append(np.asarray(indexes, np.int32))
    fraction = float(spec["selection"]["tail_fraction"]); cases = []
    definitions = (
        ("m3_early_local_patch", "early_local", "own_token_locality", "highest"),
        ("m3_late_dispersed_patch", "late_entropy", "normalized_spatial_entropy", "highest"),
        ("m3_cls_patch_contrast", "cls_patch", "cls_minus_patch_entropy_same_image_block_channel", "highest"),
        ("m3_high_signed_cancellation", "cancellation", "cancellation", "highest"),
    )
    for case_id, bucket_name, metric, tail in definitions:
        bucket = buckets[bucket_name]
        selection = numpy_tail(bucket["v"], bucket["t"], bucket["r"], tasks, paths, metric, tail, fraction)
        selected = selection["selected"]; units = [selected]
        if case_id == "m3_cls_patch_contrast":
            task_index = next(i for i, task in enumerate(tasks) if task["task_id"] == selected["task_id"])
            with np.load(paths[task_index]) as source:
                matches = np.flatnonzero((source["block"] == selected["block"]) & (source["token"] == 0) & (source["channel"] == selected["channel"]))
            require(len(matches) == 1, "CLS_PAIR"); units = [m3_row(tasks[task_index], paths[task_index], int(matches[0])), selected]
        cases.append({"case_id": case_id, "category": "neuron", "selection": selection, "selected_units": units})
    return cases, tasks, paths


def select_model_difference(tasks, paths, access=None):
    image_values_path = M3_ROOT / "image_values.csv"
    if access is not None:
        verified_artifact(image_values_path, M3_ROOT, artifact_index(M3_ROOT), access, "M3")
    rows = list(csv.DictReader(image_values_path.open()))
    by_key = {(row["model_id"], row["sample_id"]): row for row in rows}; samples = sorted(set(row["sample_id"] for row in rows))
    candidates = []
    for sample in samples:
        if ("vit_b16", sample) in by_key and ("deit_s16", sample) in by_key:
            value = float(by_key[("vit_b16", sample)]["late_minus_early_entropy"]) - float(by_key[("deit_s16", sample)]["late_minus_early_entropy"])
            candidates.append({"sample_id": sample, "model_id": "paired_models", "vit_minus_deit_late_minus_early_entropy": value})
    selection = select_nearest_median(candidates, "vit_minus_deit_late_minus_early_entropy")
    sample = selection["selected"]["sample_id"]; units = []
    for model in ("vit_b16", "deit_s16"):
        task_index = next(i for i, task in enumerate(tasks) if task["payload"]["model_id"] == model and task["payload"]["sample_id"] == sample)
        with np.load(paths[task_index]) as source:
            for block in (0, 11):
                indexes = np.flatnonzero((source["block"] == block) & (source["token"] == 30))
                values = source["normalized_spatial_entropy"][indexes].astype(np.float64); median = float(np.median(values))
                distance = np.abs(values - median); chosen = indexes[np.flatnonzero(distance == distance.min())[0]]
                unit = m3_row(tasks[task_index], paths[task_index], int(chosen)); unit["within_image_block_entropy_median"] = median
                units.append(unit)
    return {"case_id": "m3_model_layer_difference", "category": "model_layer", "selection": selection, "selected_units": units}


def select_m4(spec, access):
    manifest = artifact_index(M4_ROOT); task_manifest = load(M4_ROOT / "task_manifest.json")
    attention, residual = [], []
    for task in task_manifest["tasks"]:
        task_id = task["task_id"]; path = M4_ROOT / "results/attempts" / task_id / "attempt-01.json"
        verified_artifact(path, M4_ROOT, manifest, access, "M4")
        envelope = load(path); require(envelope["task_id"] == task_id, "M4_TASK_ID"); result = envelope["result"]
        base = {"task_id": task_id, "model_id": result["model_id"], "sample_id": result["sample_id"],
                "image_sha256": result["image_sha256"], "source_path": path.relative_to(REPO).as_posix(), "source_sha256": sha(path)}
        attention.extend({**base, **row} for row in result["attention"])
        residual.extend({**base, **row} for row in result["residual"])
    fraction = float(spec["selection"]["tail_fraction"]); cases = []
    for case_id, metric, tail in (
        ("m4_routing_dominant", "routing_share", "highest"),
        ("m4_value_dominant", "routing_share", "lowest"),
        ("m4_high_frozen_attention_error", "frozen_attention_endpoint_error", "highest"),
    ):
        selection = select_tail(attention, metric, tail, fraction); selected = dict(selection["selected"])
        selected.pop("direction_records", None); selection["selected"] = selected
        cases.append({"case_id": case_id, "category": "attention", "selection": selection, "selected_heads": [selected]})
    for case_id, metric, tail in (
        ("m4_identity_dominant", "identity_share", "highest"),
        ("m4_branch_dominant", "identity_share", "lowest"),
        ("m4_strong_path_cancellation", "path_cancellation", "highest"),
    ):
        selection = select_tail(residual, metric, tail, fraction); selected = selection["selected"]
        cases.append({"case_id": case_id, "category": "residual", "selection": selection, "selected_residuals": [selected]})
    return cases


def select_m2(access):
    manifest = artifact_index(M2_ROOT); tasks = load(M2_ROOT / "task_manifest.json")["tasks"]
    verified_artifact(M2_ROOT / "task_manifest.json", M2_ROOT, manifest, access, "M2")
    candidates, sources = [], {}
    for task in tasks:
        payload = task["payload"]
        if payload["model_id"] != "deit_s16" or int(payload["block"]) != 5:
            continue
        task_id = task["task_id"]; path = M2_ROOT / "results/attempts" / task_id / "attempt-01.json"
        verified_artifact(path, M2_ROOT, manifest, access, "M2"); result = load(path)["result"]
        sources[task_id] = (path, result, payload)
        for row in result["scores"]:
            if row["geometry"] == "patch" and row["reference"] == "normalized_zero":
                rho = spearman(row["scores"]["endpoint_absolute"], row["scores"]["regional_taylor"]); require(rho is not None, "M2_RANK_NA")
                candidates.append({"task_id": task_id, "model_id": "deit_s16", "sample_id": result["sample_id"], "block": 5,
                                   "head": int(row["head"]), "one_minus_spearman_endpoint_absolute_vs_regional_taylor": 1.0 - rho})
    selection = select_nearest_median(candidates, "one_minus_spearman_endpoint_absolute_vs_regional_taylor")
    chosen = selection["selected"]; path, result, payload = sources[chosen["task_id"]]
    score = next(row for row in result["scores"] if row["geometry"] == "patch" and row["reference"] == "normalized_zero" and int(row["head"]) == chosen["head"])
    curves = [row for row in result["curves"] if row["geometry"] == "patch" and row["reference"] == "normalized_zero" and int(row["head"]) == chosen["head"] and row["method"] in {"endpoint_absolute", "regional_taylor"}]
    require(len(curves) == 2, "M2_CURVES")
    chosen.update({"image_sha256": result["image_sha256"], "source_path": path.relative_to(REPO).as_posix(), "source_sha256": sha(path),
                   "endpoint_absolute_196": score["scores"]["endpoint_absolute"], "regional_taylor_196": score["scores"]["regional_taylor"],
                   "curves": curves, "cache_path": payload["cache_path"]})
    return {"case_id": "m2_semantic_boundary", "category": "semantic_boundary", "selection": selection, "selected_boundary": chosen}


def main():
    require("torch" not in sys.modules and "timm" not in sys.modules and "PIL" not in sys.modules, "SELECTION_IMPORT_BOUNDARY")
    require(not ROOT.exists(), "FRESH_ROOT_REQUIRED"); spec = load(SPEC)
    for relative, expected_sha in spec["source_bindings"].items(): require(sha(REPO / relative) == expected_sha, "SOURCE_BINDING:" + relative)
    statuses = {
        "M3": load(M3_ROOT / "terminal_receipt.json")["status"], "M4": load(M4_ROOT / "terminal_receipt.json")["status"],
        "M4_REVIEW_R2": load(REPO / "reports/nns_vit_endpoint_m4_review_r2_20260919/terminal_receipt.json")["status"],
        "M5": load(REPO / "reports/nns_vit_endpoint_m5_r2_20260919/terminal_receipt.json")["status"],
        "M2": load(M2_ROOT / "terminal_receipt.json")["status"],
    }
    expected = {"M3": "ENDPOINT_M3_PASS_EXECUTION_AND_AUDIT", "M4": "ENDPOINT_M4_PASS_EXECUTION_AND_AUDIT",
                "M4_REVIEW_R2": "ENDPOINT_M4_REVIEW_R2_PASS_POST_HOC_SENSITIVITY", "M5": "ENDPOINT_M5_PASS_EXECUTION_AND_AUDIT",
                "M2": "ENDPOINT_M2_PASS_EXECUTION_AND_AUDIT"}
    require(statuses == expected, "PREDECESSOR_STATUS")
    ROOT.mkdir(); source_files = [SPEC, REPO / "docs/endpoint_m6_execution_20260919.md", Path(__file__), REPO / "scripts/run_endpoint_m6_extract.py",
        REPO / "scripts/render_endpoint_m6.py", REPO / "scripts/validate_endpoint_m6.py", REPO / "scripts/validate_endpoint_m6_documents.py",
        REPO / "scripts/seal_endpoint_m6.py", REPO / "scripts/validate_stage_m_document_m6.py", REPO / "scripts/write_endpoint_m6_documents.py",
        REPO / "src/vitnns/endpoint_m6.py", REPO / "tests/test_endpoint_m6.py"]
    write_new(ROOT / "freeze.json", {"time_ns": time.time_ns(), "producer_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "sources": {path.relative_to(REPO).as_posix(): sha(path) for path in source_files}, "config_source_bindings": spec["source_bindings"], "selection_before_image_access": True})
    access = {}; m3_cases, tasks, paths = select_m3(spec, access); cases = list(m3_cases)
    cases.append(select_model_difference(tasks, paths, access)); cases.extend(select_m4(spec, access)); cases.append(select_m2(access))
    order = ["m3_early_local_patch", "m3_late_dispersed_patch", "m3_cls_patch_contrast", "m4_routing_dominant", "m4_value_dominant",
             "m4_high_frozen_attention_error", "m4_identity_dominant", "m4_branch_dominant", "m4_strong_path_cancellation",
             "m3_high_signed_cancellation", "m3_model_layer_difference", "m2_semantic_boundary"]
    by_id = {row["case_id"]: row for row in cases}; require(set(by_id) == set(order), "CASE_COVERAGE")
    write_new(ROOT / "source_access_manifest.json", {"file_count": len(access), "files": [{"path": path, **value} for path, value in sorted(access.items())]})
    write_new(ROOT / "source_validation.json", {"status": "PASS", "predecessor_statuses": statuses, "bound_sources": len(spec["source_bindings"]),
              "accessed_sealed_numeric_files": len(access), "mode": "CPU_ONLY_NO_JPEG_NO_MODEL", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "prohibited_imports_absent": True})
    write_new(ROOT / "selection_manifest.json", {"stage": "ENDPOINT-M6", "selection_status": "FROZEN_BEFORE_VISUALIZATION",
              "manual_visual_reselection": False, "case_count": 12, "case_order": order, "cases": [by_id[key] for key in order], "new_inferential_tests": 0})
    print(json.dumps({"status": "M6_SELECTION_FROZEN", "cases": len(cases), "sources": len(access)}))


if __name__ == "__main__":
    main()
