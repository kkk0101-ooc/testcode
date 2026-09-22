#!/usr/bin/env python3
"""Independent CPU-only validator for N2-V2 low-level matching R1."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/n2_v2_matching_r2_20260922"
MANIFEST_PATH = OUT / "pair_manifest.json"
ARRAY_PATH = OUT / "pair_inputs.npz"
PRODUCTION_PATH = OUT / "production_receipt.json"
VALIDATION_PATH = OUT / "validation.json"
TERMINAL_PATH = OUT / "terminal_receipt.json"
ACCEPTED_PATH = ROOT / "evidence/n2_v2_i7_20260922/accepted_scene_only_assets.json"
CONTRACT_PATH = ROOT / "config/n2_v2/input_identifiability_contract.json"
LEDGER_PATH = ROOT / "manifests/n2_v2/legacy_lineage_exclusions.json"
ASSET_ROOT = ROOT / "evidence/n2_v2_i5_20260922"
RECIPIENT_IMAGE_ROOT = ASSET_ROOT / "private/recipient_images"
RECIPIENT_MASK_ROOT = ASSET_ROOT / "private/recipient_masks"
METRICS = ("L1", "L2", "RGB_MEAN_SHIFT", "HIGHPASS_L1", "BOUNDARY_L1")
RESIZE_SHORT = 248
CROP_SIZE = 224
EPSILON = 1e-8
TOLERANCE = 1e-10


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check(name: str, condition: bool, evidence: Any, checks: list[dict[str, Any]]) -> None:
    checks.append({"name": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})
    if not condition:
        raise AssertionError(name)


def preprocess(path: Path, *, mask: bool = False) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("L" if mask else "RGB")
        old_width, old_height = image.size
        if old_width <= old_height:
            width, height = RESIZE_SHORT, int(round(old_height * RESIZE_SHORT / old_width))
        else:
            width, height = int(round(old_width * RESIZE_SHORT / old_height)), RESIZE_SHORT
        interpolation = Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC
        resized = image.resize((width, height), resample=interpolation)
        left, top = (width - CROP_SIZE) // 2, (height - CROP_SIZE) // 2
        array = np.asarray(resized.crop((left, top, left + CROP_SIZE, top + CROP_SIZE)))
    if mask:
        return np.ascontiguousarray(array > 127, dtype=np.bool_)
    return np.ascontiguousarray(array, dtype=np.float32) / np.float32(255.0)


def composite(foreground: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    alpha = mask[..., None].astype(np.float32)
    return np.ascontiguousarray(alpha * foreground + (1.0 - alpha) * background, dtype=np.float32)


def metrics(first: np.ndarray, second: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    difference = second.astype(np.float64) - first.astype(np.float64)
    difference[foreground_mask] = 0.0
    grey = difference.mean(axis=-1)
    outer = ndimage.binary_dilation(foreground_mask, iterations=3)
    inner = ndimage.binary_erosion(foreground_mask, iterations=3)
    border = np.logical_xor(outer, inner)
    values = (
        float(np.abs(difference).mean()),
        float(np.sqrt(np.square(difference).mean())),
        float(np.linalg.norm(difference.mean(axis=(0, 1)))),
        float(np.abs(grey - ndimage.gaussian_filter(grey, 2.0)).mean()),
        float(np.abs(difference[border]).mean()) if border.any() else 0.0,
    )
    return np.asarray(values, dtype=np.float64)


def metric_components(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked = image.astype(np.float64).copy()
    masked[mask] = 0.0
    grey = masked.mean(axis=2)
    return masked, grey, ndimage.gaussian_filter(grey, 2.0)


def cached_metrics(
    first: tuple[np.ndarray, np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray, np.ndarray],
    border: np.ndarray,
) -> np.ndarray:
    difference = second[0] - first[0]
    grey = second[1] - first[1]
    lowpass = second[2] - first[2]
    return np.asarray((
        float(np.abs(difference).mean()),
        float(np.sqrt(np.square(difference).mean())),
        float(np.linalg.norm(difference.mean(axis=(0, 1)))),
        float(np.abs(grey - lowpass).mean()),
        float(np.abs(difference[border]).mean()) if border.any() else 0.0,
    ), dtype=np.float64)


def compare(left: np.ndarray, right: np.ndarray) -> tuple[float, float, np.ndarray]:
    left = np.maximum(left, EPSILON)
    right = np.maximum(right, EPSILON)
    ratios = np.maximum(left, right) / np.minimum(left, right)
    distance = float(np.linalg.norm(np.log(left / right)))
    return distance, float(ratios.max()), ratios


def collect_lineage(node: Any, groups: set[str], hashes: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "source_group_id" and isinstance(value, str):
                groups.add(value)
            if key in {"image_sha256", "annotation_sha256"} and isinstance(value, str):
                hashes.add(value)
            collect_lineage(value, groups, hashes)
    elif isinstance(node, list):
        for value in node:
            collect_lineage(value, groups, hashes)


def manifest_key(candidate: tuple[float, tuple[str, ...], int, int, int, int], recipients: list[dict[str, Any]], scenes: list[dict[str, Any]]) -> tuple[str, str, str, str]:
    _, _, recipient_index, base_index, semantic_index, null_index = candidate
    return (recipients[recipient_index]["source_group_id"], scenes[base_index]["source_group_id"], scenes[semantic_index]["source_group_id"], scenes[null_index]["source_group_id"])


def solve_assignment(candidates: list[tuple[float, tuple[str, ...], int, int, int, int]], indices: list[int], objective: np.ndarray, recipient_count: int, scene_count: int, time_limit: float) -> tuple[Any, list[int]]:
    rr: list[int] = []; cc: list[int] = []; vv: list[float] = []
    for column, original_index in enumerate(indices):
        candidate = candidates[original_index]
        for row in (0, 1 + candidate[2], 1 + recipient_count + candidate[3], 1 + recipient_count + candidate[4], 1 + recipient_count + candidate[5]):
            rr.append(row); cc.append(column); vv.append(1.0)
    matrix = sparse.coo_matrix((vv, (rr, cc)), shape=(1 + recipient_count + scene_count, len(indices))).tocsr()
    lower = np.zeros(matrix.shape[0]); lower[0] = 8
    upper = np.ones(matrix.shape[0]); upper[0] = 8
    result = milp(objective[indices], integrality=np.ones(len(indices)), bounds=Bounds(np.zeros(len(indices)), np.ones(len(indices))), constraints=LinearConstraint(matrix, lower, upper), options={"presolve": True, "mip_rel_gap": 0.0, "time_limit": time_limit})
    selected = [] if result.x is None else [indices[index] for index, value in enumerate(result.x) if value > 0.5]
    return result, selected


def main() -> None:
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES_MUST_BE_EMPTY")
    forbidden_modules = {"torch", "timm", "tensorflow", "jax"}
    if forbidden_modules.intersection(sys.modules):
        raise RuntimeError("MODEL_FRAMEWORK_IMPORTED")
    if VALIDATION_PATH.exists() or TERMINAL_PATH.exists():
        raise FileExistsError("VALIDATION_ALREADY_SEALED")
    print("[validator] start independent_cpu_only_recompute=true", flush=True)

    checks: list[dict[str, Any]] = []
    manifest = load_json(MANIFEST_PATH)
    contract = load_json(CONTRACT_PATH)
    accepted = load_json(ACCEPTED_PATH)
    ledger = load_json(LEDGER_PATH)
    production = load_json(PRODUCTION_PATH)
    matching = contract["matching"]
    policy = contract["asset_policy"]

    check("SOURCE_PATHS_EXACT", manifest["source_bindings"]["accepted_registry_path"] == str(ACCEPTED_PATH.relative_to(ROOT)) and manifest["source_bindings"]["contract_path"] == str(CONTRACT_PATH.relative_to(ROOT)) and manifest["source_bindings"]["legacy_lineage_ledger_path"] == str(LEDGER_PATH.relative_to(ROOT)), manifest["source_bindings"], checks)
    check("SOURCE_HASHES_RECOMPUTED", manifest["source_bindings"]["accepted_registry_sha256"] == sha_file(ACCEPTED_PATH) and manifest["source_bindings"]["contract_sha256"] == sha_file(CONTRACT_PATH) and manifest["source_bindings"]["legacy_lineage_ledger_sha256"] == sha_file(LEDGER_PATH) == accepted["legacy_lineage_ledger_sha256"], manifest["source_bindings"], checks)
    check("PRODUCTION_BINDINGS_RECOMPUTED", production["pair_manifest_sha256"] == sha_file(MANIFEST_PATH) and production["pair_inputs_sha256"] == sha_file(ARRAY_PATH), {"manifest": sha_file(MANIFEST_PATH), "arrays": sha_file(ARRAY_PATH)}, checks)
    check("CONTRACT_METRICS_AND_CALIPERS", matching["low_level_metrics"] == list(METRICS) and manifest["calipers"] == {"max_per_metric_ratio": 2.5, "max_aggregate_log_ratio_distance": 1.25, "max_unmatched_rate": 0.0, "background_max_reuse": 1}, manifest["calipers"], checks)
    check("FORBIDDEN_AUTHORIZATIONS_FALSE", not any(contract["authorization"].values()), contract["authorization"], checks)
    check("STOP_GATE_PRESENT", manifest["next_gate"] == "STOP_BEFORE_BEHAVIOR_GATE_AND_N3_REQUIRES_EXPLICIT_USER_AUTHORIZATION", manifest["next_gate"], checks)

    legacy_groups: set[str] = set(); legacy_hashes: set[str] = set()
    collect_lineage(ledger, legacy_groups, legacy_hashes)
    scenes: list[dict[str, Any]] = []
    scene_by_group: dict[str, int] = {}
    scene_path_set: set[str] = set()
    scene_overlap: list[str] = []
    for index, row in enumerate(accepted["records"]):
        path = ASSET_ROOT / row["image_path"]
        digest = sha_file(path)
        if digest != row["image_sha256"]:
            raise AssertionError(f"SCENE_HASH_{index}")
        if row["source_group_id"] in legacy_groups or digest in legacy_hashes:
            scene_overlap.append(row["asset_id"])
        scenes.append({**row, "path": str(path.relative_to(ROOT)), "array": preprocess(path)})
        scene_by_group[row["source_group_id"]] = index
        scene_path_set.add(str(path.relative_to(ROOT)))
    check("I7_SCENES_RECOMPUTED", len(scenes) == 43 and Counter(row["context_label"] for row in scenes) == Counter({"AQUATIC_CONTEXT": 19, "NON_AQUATIC_CONTEXT": 24}), {"count": len(scenes), "contexts": dict(Counter(row["context_label"] for row in scenes))}, checks)
    check("ALL_I7_SCENES_ZERO_LEGACY_INTERSECTION", not scene_overlap, scene_overlap, checks)

    mask_by_id: dict[str, Path] = {}
    for path in sorted(RECIPIENT_MASK_ROOT.iterdir()):
        if path.is_file():
            recipient_id = path.name.split("_", 1)[0]
            if recipient_id in mask_by_id:
                raise AssertionError("DUPLICATE_MASK_ID")
            mask_by_id[recipient_id] = path
    recipients: list[dict[str, Any]] = []
    recipient_by_group: dict[str, int] = {}
    recipient_overlap: list[str] = []
    recipient_paths: set[str] = set()
    mask_paths: set[str] = set()
    for image_path in sorted(RECIPIENT_IMAGE_ROOT.iterdir()):
        if not image_path.is_file():
            continue
        recipient_id = image_path.stem
        mask_path = mask_by_id[recipient_id]
        image_digest, mask_digest = sha_file(image_path), sha_file(mask_path)
        group = f"OPENIMAGES_TRAIN_IMAGE:{recipient_id}"
        if group in legacy_groups or image_digest in legacy_hashes or mask_digest in legacy_hashes:
            recipient_overlap.append(recipient_id)
        image = preprocess(image_path); mask = preprocess(mask_path, mask=True)
        if image.shape[:2] != mask.shape or not mask.any() or mask.all():
            raise AssertionError(f"RECIPIENT_GEOMETRY:{recipient_id}")
        recipients.append({"recipient_id": recipient_id, "source_group_id": group, "image": image, "mask": mask, "image_path": str(image_path.relative_to(ROOT)), "image_sha256": image_digest, "mask_path": str(mask_path.relative_to(ROOT)), "mask_sha256": mask_digest})
        recipient_by_group[group] = len(recipients) - 1
        recipient_paths.add(str(image_path.relative_to(ROOT))); mask_paths.add(str(mask_path.relative_to(ROOT)))
    check("TWELVE_RECIPIENTS_AND_MASKS_RECOMPUTED", len(recipients) == len(mask_by_id) == 12, {"recipient_count": len(recipients), "mask_count": len(mask_by_id)}, checks)
    check("ALL_RECIPIENTS_ZERO_LEGACY_INTERSECTION", not recipient_overlap, recipient_overlap, checks)
    print(f"[validator] inputs_verified scenes={len(scenes)} recipients={len(recipients)} elapsed={time.monotonic()-started:.1f}s", flush=True)

    accesses = manifest["input_access_manifest"]
    accessed_paths = {row["path"] for row in accesses}
    expected_paths = scene_path_set | recipient_paths | mask_paths | {str(ACCEPTED_PATH.relative_to(ROOT)), str(CONTRACT_PATH.relative_to(ROOT)), str(LEDGER_PATH.relative_to(ROOT))}
    check("INPUT_ACCESS_MANIFEST_EXACT_WHITELIST", accessed_paths == expected_paths and len(accesses) == len(expected_paths) == 70, {"count": len(accesses), "unexpected": sorted(accessed_paths - expected_paths), "missing": sorted(expected_paths - accessed_paths)}, checks)
    check("INPUT_ACCESS_HASHES_RECOMPUTED", all(sha_file(ROOT / row["path"]) == row["sha256"] for row in accesses), {"count": len(accesses)}, checks)

    aquatic = [i for i, row in enumerate(scenes) if row["context_label"] == "AQUATIC_CONTEXT"]
    non_aquatic = [i for i, row in enumerate(scenes) if row["context_label"] == "NON_AQUATIC_CONTEXT"]
    candidates: list[tuple[float, tuple[str, ...], int, int, int, int]] = []
    best_by_recipient: list[float | None] = [None] * len(recipients)
    for recipient_index, recipient in enumerate(recipients):
        border = ndimage.binary_dilation(recipient["mask"], iterations=3) ^ ndimage.binary_erosion(recipient["mask"], iterations=3)
        components = [metric_components(scene["array"], recipient["mask"]) for scene in scenes]
        for base_index in non_aquatic:
            base = components[base_index]
            semantic_values = {i: cached_metrics(base, components[i], border) for i in aquatic}
            null_values = {i: cached_metrics(base, components[i], border) for i in non_aquatic if i != base_index}
            for semantic_index, first in semantic_values.items():
                for null_index, second in null_values.items():
                    distance, max_ratio, _ = compare(first, second)
                    if distance <= float(matching["max_aggregate_log_ratio_distance"]) and max_ratio <= float(matching["max_per_metric_ratio"]):
                        tie = tuple(sha_text(x) for x in (recipient["source_group_id"], scenes[base_index]["source_group_id"], scenes[semantic_index]["source_group_id"], scenes[null_index]["source_group_id"]))
                        candidates.append((distance, tie, recipient_index, base_index, semantic_index, null_index))
                        prior = best_by_recipient[recipient_index]
                        best_by_recipient[recipient_index] = distance if prior is None else min(prior, distance)
        print(f"[validator] recipient={recipient_index+1}/{len(recipients)} id={recipient['recipient_id']} valid_candidates={len(candidates)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    candidates.sort(key=lambda row: (row[0], row[1]))
    check("VALID_CANDIDATE_COUNT_RECOMPUTED", len(candidates) == manifest["selection"]["caliper_valid_candidate_count"], {"count": len(candidates)}, checks)

    n = len(candidates)
    tie_rank = np.empty(n)
    for rank, index in enumerate(sorted(range(n), key=lambda i: candidates[i][1])):
        tie_rank[index] = rank / max(n - 1, 1)
    objective = np.asarray([row[0] for row in candidates]) + 1e-12 * tie_rank
    per_recipient = [[index for index, row in enumerate(candidates) if row[2] == recipient_index] for recipient_index in range(len(recipients))]
    upper_result = None; upper_solution: list[int] = []
    for limit in (64, 256, 1024):
        upper_indices = sorted({index for indices in per_recipient for index in indices[:limit]})
        upper_result, upper_solution = solve_assignment(candidates, upper_indices, objective, len(recipients), len(scenes), 30.0)
        print(f"[validator] upper_bound limit_per_recipient={limit} subset={len(upper_indices)} success={upper_result.success} selected={len(upper_solution)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if len(upper_solution) == 8:
            break
    check("FEASIBLE_UPPER_BOUND_RECOMPUTED", len(upper_solution) == 8, str(upper_result.message), checks)
    incumbent = float(objective[upper_solution].sum())
    recipient_minima = [min(objective[index] for index in indices) for indices in per_recipient]
    safe_indices = [index for index, candidate in enumerate(candidates) if objective[index] + sum(sorted(value for recipient_index, value in enumerate(recipient_minima) if recipient_index != candidate[2])[:7]) <= incumbent + 1e-12]
    check("SAFE_PRUNING_RECOMPUTED", len(safe_indices) == manifest["selection"]["solver"]["safe_pruned_candidate_count"], {"full": n, "retained": len(safe_indices), "incumbent": incumbent}, checks)
    result, solution_indices = solve_assignment(candidates, safe_indices, objective, len(recipients), len(scenes), 300.0)
    check("INDEPENDENT_GLOBAL_ASSIGNMENT_HAS_EXACT_SOLUTION", bool(result.success and result.x is not None), str(result.message), checks)
    print(f"[validator] exact_solver success={result.success} full={n} retained={len(safe_indices)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    solution = [candidates[index] for index in solution_indices]
    expected_selection = {manifest_key(row, recipients, scenes) for row in solution}
    actual_selection = {(row["recipient"]["source_group_id"], row["base_scene"]["source_group_id"], row["semantic_scene"]["source_group_id"], row["within_stratum_null_scene"]["source_group_id"]) for row in manifest["records"]}
    check("GLOBAL_MINIMUM_SELECTION_REPRODUCED", actual_selection == expected_selection and len(actual_selection) == 8, {"pair_count": len(actual_selection), "objective": float(result.fun)}, checks)

    archive = np.load(ARRAY_PATH, allow_pickle=False)
    max_metric_error = 0.0; max_array_error = 0.0; background_groups: list[str] = []
    selected_hashes: set[str] = set(); selected_groups: set[str] = set(); selected_recipient_groups: set[str] = set()
    for index, row in enumerate(manifest["records"]):
        recipient = recipients[recipient_by_group[row["recipient"]["source_group_id"]]]
        base_scene = scenes[scene_by_group[row["base_scene"]["source_group_id"]]]
        semantic_scene = scenes[scene_by_group[row["semantic_scene"]["source_group_id"]]]
        null_scene = scenes[scene_by_group[row["within_stratum_null_scene"]["source_group_id"]]]
        if base_scene["context_label"] != "NON_AQUATIC_CONTEXT" or semantic_scene["context_label"] != "AQUATIC_CONTEXT" or null_scene["context_label"] != "NON_AQUATIC_CONTEXT" or base_scene["source_group_id"] == null_scene["source_group_id"]:
            raise AssertionError(f"PAIR_SEMANTIC_ROLES_{index}")
        base_arm = composite(recipient["image"], recipient["mask"], base_scene["array"] )
        semantic_arm = composite(recipient["image"], recipient["mask"], semantic_scene["array"] )
        null_arm = composite(recipient["image"], recipient["mask"], null_scene["array"] )
        pair_id = row["pair_id"]
        for suffix, expected in (("base", base_arm), ("semantic", semantic_arm), ("within_stratum_null", null_arm), ("recipient_mask", recipient["mask"])):
            max_array_error = max(max_array_error, float(np.max(np.abs(archive[f"{pair_id}_{suffix}"].astype(np.float64) - expected.astype(np.float64)))))
        semantic_metric = metrics(base_arm, semantic_arm, recipient["mask"]); null_metric = metrics(base_arm, null_arm, recipient["mask"]); distance, max_ratio, ratios = compare(semantic_metric, null_metric)
        for metric_index, name in enumerate(METRICS):
            max_metric_error = max(max_metric_error, abs(float(semantic_metric[metric_index]) - row["semantic_metrics"][name]), abs(float(null_metric[metric_index]) - row["within_stratum_null_metrics"][name]), abs(float(ratios[metric_index]) - row["per_metric_ratios"][name]))
            if ratios[metric_index] > 2.5 + TOLERANCE:
                raise AssertionError(f"PER_METRIC_CALIPER_{index}_{name}")
        if abs(distance - row["aggregate_log_ratio_distance"]) > TOLERANCE or abs(max_ratio - row["max_per_metric_ratio"]) > TOLERANCE or distance > 1.25 + TOLERANCE or not row["matched"]:
            raise AssertionError(f"AGGREGATE_CALIPER_{index}")
        background_groups.extend([base_scene["source_group_id"], semantic_scene["source_group_id"], null_scene["source_group_id"]])
        selected_groups.update((recipient["source_group_id"], base_scene["source_group_id"], semantic_scene["source_group_id"], null_scene["source_group_id"]))
        selected_hashes.update((recipient["image_sha256"], recipient["mask_sha256"], base_scene["image_sha256"], semantic_scene["image_sha256"], null_scene["image_sha256"]))
        selected_recipient_groups.add(recipient["source_group_id"])
    archive.close()
    check("PAIR_ARRAYS_EXACTLY_RECOMPUTED", max_array_error == 0.0, {"max_absolute_error": max_array_error}, checks)
    check("ALL_FIVE_METRICS_AND_CALIPERS_RECOMPUTED", max_metric_error <= TOLERANCE and len(manifest["records"]) == 8 and all(row["matched"] for row in manifest["records"]), {"max_numeric_error": max_metric_error, "matched_count": len(manifest["records"])}, checks)
    observed_reuse = max(Counter(background_groups).values())
    check("BACKGROUND_REUSE_MAX_ONE", observed_reuse == 1 and manifest["selection"]["background_max_reuse_observed"] == 1, {"observed": observed_reuse}, checks)
    check("SELECTED_LINEAGE_ZERO_INTERSECTION", not (selected_groups & legacy_groups) and not (selected_hashes & legacy_hashes) and manifest["lineage_audit"]["zero_intersection"], {"group_intersection": sorted(selected_groups & legacy_groups), "hash_intersection": sorted(selected_hashes & legacy_hashes)}, checks)

    expected_reserve_indices = sorted((i for i, row in enumerate(recipients) if row["source_group_id"] not in selected_recipient_groups), key=lambda i: (math.inf if best_by_recipient[i] is None else best_by_recipient[i], sha_text(recipients[i]["source_group_id"])))
    expected_reserves = [recipients[i]["source_group_id"] for i in expected_reserve_indices]
    actual_reserves = [row["source_group_id"] for row in manifest["reserve_recipients"]]
    check("FOUR_RESERVES_FROZEN_IN_SCORE_THEN_HASH_ORDER", len(actual_reserves) == 4 and actual_reserves == expected_reserves and not (set(actual_reserves) & selected_recipient_groups), actual_reserves, checks)
    check("EIGHT_OF_EIGHT_AND_ZERO_UNMATCHED", manifest["status"] == "PASS_INPUT_IDENTIFIABILITY_FROZEN" and manifest["selection"]["final_pair_count"] == manifest["selection"]["matched_pair_count"] == 8 and manifest["selection"]["unmatched_rate"] == 0.0, manifest["selection"], checks)
    counters = manifest["prohibited_activity_counters"]
    check("PROHIBITED_ACTIVITY_COUNTERS_ZERO", set(counters) == {"checkpoint_reads", "model_loads", "inference_runs", "behavior_outcome_reads", "activation_runs", "endpoint_runs", "path_nns_runs", "jvp_runs", "intervention_runs", "gpu_operations"} and not any(counters.values()), counters, checks)

    validation = {
        "schema_version": 1,
        "stage_id": "N2_V2_LOW_LEVEL_MATCHING",
        "attempt_id": manifest["attempt_id"],
        "status": "PASS_N2_V2_MATCHING_VALIDATION",
        "checks": checks,
        "summary": {"check_count": len(checks), "pair_count": 8, "reserve_count": 4, "matched_count": 8, "unmatched_rate": 0.0, "background_max_reuse": observed_reuse, "legacy_lineage_intersection_count": 0, "max_metric_recompute_error": max_metric_error, "max_array_recompute_error": max_array_error},
        "validator_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "validator_sha256": sha_file(Path(__file__).resolve()),
        "python_executable_realpath": str(Path(sys.executable).resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "forbidden_modules_loaded": sorted(forbidden_modules.intersection(sys.modules)),
    }
    write_json(VALIDATION_PATH, validation)
    terminal = {
        "schema_version": 1,
        "stage_id": "N2_V2_LOW_LEVEL_MATCHING",
        "attempt_id": manifest["attempt_id"],
        "status": "PASS_INPUT_IDENTIFIABILITY_FROZEN",
        "pair_manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "pair_manifest_sha256": sha_file(MANIFEST_PATH),
        "pair_inputs_path": str(ARRAY_PATH.relative_to(ROOT)),
        "pair_inputs_sha256": sha_file(ARRAY_PATH),
        "production_receipt_path": str(PRODUCTION_PATH.relative_to(ROOT)),
        "production_receipt_sha256": sha_file(PRODUCTION_PATH),
        "validation_path": str(VALIDATION_PATH.relative_to(ROOT)),
        "validation_sha256": sha_file(VALIDATION_PATH),
        "gate_results": {"final_8_pairs": "PASS_8_OF_8", "per_metric_calipers": "PASS", "aggregate_caliper": "PASS", "background_reuse_max_1": "PASS", "legacy_lineage_zero_intersection": "PASS", "four_reserves_frozen": "PASS", "model_side_access": "NOT_RUN_NOT_AUTHORIZED"},
        "authorization": {"input_identifiability": True, "behavior_gate": False, "model_checkpoint_access": False, "n3": False},
        "next_gate": "STOP_BEFORE_BEHAVIOR_GATE_AND_N3_REQUIRES_EXPLICIT_USER_AUTHORIZATION",
    }
    write_json(TERMINAL_PATH, terminal)
    print(json.dumps({"status": terminal["status"], "checks": len(checks), "pairs": 8, "reserves": 4, "terminal_receipt": str(TERMINAL_PATH.relative_to(ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
