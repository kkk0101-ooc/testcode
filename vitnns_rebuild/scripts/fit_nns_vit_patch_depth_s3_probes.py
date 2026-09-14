#!/usr/bin/env python3
"""Fit all class-conditional S3 probes, cross-fit diagnostics, and seal weights."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa: E402


def terminal_for(index: int) -> Mapping[str, Any]:
    path = REPO / s3.ROOT_REL / "unit_terminals" / f"unit-{index:05d}.json"
    return json.loads(path.read_text())


def load_unit(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = int(row["schedule_index"])
    terminal = terminal_for(index)
    s3.require(s3.verify_signed(terminal), f"unit {index} terminal digest invalid")
    s3.require(terminal["payload"]["source_image_id"] == row["source_image_id"], f"unit {index} terminal source mismatch")
    s3.require(terminal["payload"]["terminal_state"] == "VALID_FEATURE", f"unit {index} is not VALID_FEATURE")
    unit_root = REPO / s3.PAYLOAD_REL / "units" / f"unit-{index:05d}"
    features = np.load(unit_root / "block_features.npy", allow_pickle=False)
    occupancy = np.load(unit_root / "occupancy.npy", allow_pickle=False)
    control = np.load(unit_root / "low_level_features.npy", allow_pickle=False)
    s3.require(features.shape == (12,196,768) and occupancy.shape == (4,196) and control.shape == (196,5), f"unit {index} array shape drift")
    return features, occupancy, control


def fit_candidate(train_images: list[np.ndarray], train_targets: list[np.ndarray], validation_images: list[np.ndarray], validation_targets: list[np.ndarray]) -> tuple[s3.RidgeModel, list[dict[str, float]]]:
    train_x = np.concatenate(train_images, axis=0).astype(np.float64)
    train_y = np.concatenate(train_targets, axis=0).astype(np.float64)
    decomposition = s3.ridge_decompose(train_x, train_y)
    ledger = []
    models = {}
    for alpha in s3.ALPHAS:
        model = s3.model_from_decomposition(decomposition, alpha)
        models[alpha] = model
        per_image = [float(np.mean((s3.predict_ridge(model, x) - y) ** 2, dtype=np.float64)) for x, y in zip(validation_images, validation_targets)]
        ledger.append({"alpha":float(alpha),"per_image_mse":per_image,"equal_image_weight_mse":math.fsum(per_image)/len(per_image)})
    selected = s3.select_alpha({row["alpha"]:row["equal_image_weight_mse"] for row in ledger})
    return models[selected], ledger


def crossfit(images: list[np.ndarray], targets: list[np.ndarray], source_ids: list[str], wnid: str, alpha: float) -> list[dict[str, Any]]:
    assignment = s3.crossfit_assignment(wnid, source_ids)
    index_by_source = {source_id:index for index, source_id in enumerate(source_ids)}
    rows = []
    for fold in range(3):
        held = [index_by_source[row["source_image_id"]] for row in assignment if row["fold"] == fold]
        fit = [index for index in range(6) if index not in held]
        model = s3.fit_ridge(np.concatenate([images[index] for index in fit]), np.concatenate([targets[index] for index in fit]), alpha)
        per_image = [float(np.mean((s3.predict_ridge(model, images[index]) - targets[index]) ** 2, dtype=np.float64)) for index in held]
        rows.append({"fold":fold,"fit_source_ids":[source_ids[index] for index in fit],"held_out_source_ids":[source_ids[index] for index in held],"per_image_mse":per_image,"equal_image_weight_mse":math.fsum(per_image)/2.0})
    s3.require(sorted(source for row in rows for source in row["held_out_source_ids"]) == sorted(source_ids), "cross-fit holdout completeness failure")
    return rows


def fit_class(wnid: str, rows: list[Mapping[str, Any]], producer_commit: str) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == "probe_train"]
    validation_rows = [row for row in rows if row["split"] == "probe_validation"]
    s3.require(len(train_rows) == 6 and len(validation_rows) == 2, f"{wnid} scheduled split count drift")
    train = [load_unit(row) for row in train_rows]
    validation = [load_unit(row) for row in validation_rows]
    train_target = [item[1][3].astype(np.float64) for item in train]
    validation_target = [item[1][3].astype(np.float64) for item in validation]
    feature_mean = np.empty((12,768), dtype="<f8")
    feature_scale = np.empty((12,768), dtype="<f8")
    feature_weight = np.empty((12,768), dtype="<f8")
    feature_intercept = np.empty(12, dtype="<f8")
    feature_alpha = np.empty(12, dtype="<f8")
    validation_ledger = []
    crossfit_ledger = []
    for block in range(12):
        model, ledger = fit_candidate([item[0][block] for item in train], train_target, [item[0][block] for item in validation], validation_target)
        feature_mean[block] = model.mean
        feature_scale[block] = model.scale
        feature_weight[block] = model.weight
        feature_intercept[block] = model.intercept
        feature_alpha[block] = model.alpha
        validation_ledger.append({"block":block,"candidates":ledger,"selected_alpha":model.alpha})
        crossfit_ledger.append({"block":block,"folds":crossfit([item[0][block] for item in train], train_target, [row["source_image_id"] for row in train_rows], wnid, model.alpha)})
    control_model, control_ledger = fit_candidate([item[2] for item in train], train_target, [item[2] for item in validation], validation_target)
    control_crossfit = crossfit([item[2] for item in train], train_target, [row["source_image_id"] for row in train_rows], wnid, control_model.alpha)
    weight_path = REPO / s3.PAYLOAD_REL / "weights" / f"{wnid}.npz"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = weight_path.with_name(weight_path.name + f".tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez(stream, feature_mean=feature_mean, feature_scale=feature_scale, feature_weight=feature_weight, feature_intercept=feature_intercept, feature_alpha=feature_alpha, control_mean=np.asarray(control_model.mean,dtype="<f8"), control_scale=np.asarray(control_model.scale,dtype="<f8"), control_weight=np.asarray(control_model.weight,dtype="<f8"), control_intercept=np.asarray(control_model.intercept,dtype="<f8"), control_alpha=np.asarray(control_model.alpha,dtype="<f8"))
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, weight_path); s3.fsync_directory(weight_path.parent)
    with np.load(weight_path, allow_pickle=False) as roundtrip:
        s3.require(np.array_equal(roundtrip["feature_weight"], feature_weight), "weight serialization is not byte-value exact")
    return {
        "annotation_wnid":wnid,"train_source_ids":[row["source_image_id"] for row in train_rows],"validation_source_ids":[row["source_image_id"] for row in validation_rows],
        "validation_ledger":validation_ledger,"control_validation_ledger":control_ledger,"selected_control_alpha":control_model.alpha,
        "crossfit":crossfit_ledger,"control_crossfit":control_crossfit,
        "weights":{"path":weight_path.relative_to(REPO).as_posix(),"bytes":weight_path.stat().st_size,"sha256":s3.sha_file(weight_path),"producer_commit":producer_commit,"serialization":"NPZ_little_endian_float64_allow_pickle_false"}
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=48)
    args = parser.parse_args()
    producer_commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=REPO, text=True).strip()
    schedule = json.loads((REPO / s3.SCHEDULE_REL).read_text())["payload"]["records"]
    by_class: dict[str, list[Mapping[str, Any]]] = {}
    for row in schedule:
        if row["split"] in ("probe_train","probe_validation"):
            by_class.setdefault(str(row["annotation_wnid"]), []).append(row)
    s3.require(len(by_class) == 872, "probe class count drift")
    result_root = REPO / s3.ROOT_REL / "probe_classes"
    result_root.mkdir(parents=True, exist_ok=True)
    missing = [wnid for wnid in sorted(by_class) if not (result_root / f"{wnid}.json").exists()]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = {executor.submit(fit_class, wnid, by_class[wnid], producer_commit):wnid for wnid in missing}
        for count, future in enumerate(as_completed(futures), 1):
            wnid = futures[future]
            result = future.result()
            document = s3.signed("s3_class_probe_terminal", "PASS", result, producer_commit=producer_commit)
            s3.write_json_new(result_root / f"{wnid}.json", document)
            if count % 25 == 0 or count == len(missing):
                print(json.dumps({"completed_this_launch":count,"missing_at_start":len(missing),"last_class":wnid,"elapsed_seconds":time.monotonic()-started}), flush=True)
    documents = [json.loads(path.read_text()) for path in sorted(result_root.glob("*.json"))]
    s3.require(len(documents) == 872 and all(s3.verify_signed(item) and item["status"] == "PASS" for item in documents), "class probe terminal completeness failed")
    weight_records = [item["payload"]["weights"] for item in documents]
    alpha_ledger = [{"annotation_wnid":item["payload"]["annotation_wnid"],"feature":[{"block":row["block"],"candidates":row["candidates"],"selected_alpha":row["selected_alpha"]} for row in item["payload"]["validation_ledger"]],"control_candidates":item["payload"]["control_validation_ledger"],"selected_control_alpha":item["payload"]["selected_control_alpha"]} for item in documents]
    normalization = [{"annotation_wnid":item["payload"]["annotation_wnid"],"weight_bundle_sha256":item["payload"]["weights"]["sha256"],"source_ids":item["payload"]["train_source_ids"],"fit_scope":"train_only"} for item in documents]
    crossfit_rows = [{"annotation_wnid":item["payload"]["annotation_wnid"],"feature":item["payload"]["crossfit"],"control":item["payload"]["control_crossfit"]} for item in documents]
    outputs = {
        "alpha_grid_validation_ledger.json": s3.signed("s3_alpha_validation_ledger", "PASS", {"records":alpha_ledger,"records_digest":s3.records_digest(alpha_ledger),"class_count":872,"selection_role":"validation_only","final_fit_role":"train_only"}, producer_commit=producer_commit),
        "normalization_manifest.json": s3.signed("s3_normalization_manifest", "PASS", {"records":normalization,"records_digest":s3.records_digest(normalization),"class_count":872,"fit_scope":"train_only_ddof_0_zero_scale_to_one"}, producer_commit=producer_commit),
        "class_block_probe_weight_manifest.json": s3.signed("s3_probe_weight_manifest", "PASS", {"records":weight_records,"records_digest":s3.records_digest(weight_records),"class_count":872,"block_count_per_class":12}, producer_commit=producer_commit),
        "low_level_control_weight_manifest.json": s3.signed("s3_control_weight_manifest", "PASS", {"records":weight_records,"records_digest":s3.records_digest(weight_records),"class_count":872,"features":["x","y","R_mean","G_mean","B_mean"]}, producer_commit=producer_commit),
        "train_only_crossfit_report.json": s3.signed("s3_crossfit_report", "PASS", {"records":crossfit_rows,"records_digest":s3.records_digest(crossfit_rows),"class_count":872,"held_out_exactly_once":True,"alpha_source":"validation_selected","formal_denominator_contribution":False}, producer_commit=producer_commit)
    }
    for name, document in outputs.items():
        s3.write_json_new(REPO / s3.ROOT_REL / name, document)
    print(json.dumps({"status":"PASS","class_count":872,"weight_records_digest":s3.records_digest(weight_records),"elapsed_seconds":time.monotonic()-started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
