#!/usr/bin/env python3
"""Build the outcome-blind N2-V2 low-level matching fresh root.

This program is deliberately self-contained.  It reads only the frozen I7
accepted registry, the image assets bound by that registry, the twelve frozen
recipient image/mask pairs, the V2 contract, and the legacy-lineage ledger.
It never imports a model framework and refuses to run with visible GPUs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parent
ATTEMPT_ID = "N2_V2_MATCHING_R2_20260922"
OUTPUT_ROOT = ROOT / "evidence/n2_v2_matching_r2_20260922"
ACCEPTED_PATH = ROOT / "evidence/n2_v2_i7_20260922/accepted_scene_only_assets.json"
CONTRACT_PATH = ROOT / "config/n2_v2/input_identifiability_contract.json"
LEDGER_PATH = ROOT / "manifests/n2_v2/legacy_lineage_exclusions.json"
ASSET_ROOT = ROOT / "evidence/n2_v2_i5_20260922"
RECIPIENT_IMAGE_ROOT = ASSET_ROOT / "private/recipient_images"
RECIPIENT_MASK_ROOT = ASSET_ROOT / "private/recipient_masks"
SCENE_ROOT = ASSET_ROOT / "private/scene_originals"
METRICS = ("L1", "L2", "RGB_MEAN_SHIFT", "HIGHPASS_L1", "BOUNDARY_L1")
RESIZE_SHORT = 248
CROP_SIZE = 224
EPSILON = 1e-8
TIE_EPSILON = 1e-12


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


def resized_geometry(width: int, height: int) -> tuple[int, int]:
    if width <= height:
        return RESIZE_SHORT, int(round(height * RESIZE_SHORT / width))
    return int(round(width * RESIZE_SHORT / height)), RESIZE_SHORT


def preprocess(path: Path, *, mask: bool = False) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("L" if mask else "RGB")
        width, height = resized_geometry(*image.size)
        interpolation = Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC
        image = image.resize((width, height), resample=interpolation)
        left = (width - CROP_SIZE) // 2
        top = (height - CROP_SIZE) // 2
        image = image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
        array = np.asarray(image)
    if mask:
        return np.ascontiguousarray(array > 127, dtype=np.bool_)
    return np.ascontiguousarray(array, dtype=np.float32) / np.float32(255.0)


def composite(recipient: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    alpha = mask[..., None].astype(np.float32)
    return np.ascontiguousarray(alpha * recipient + (1.0 - alpha) * background, dtype=np.float32)


def low_level_metrics(base: np.ndarray, other: np.ndarray, mask: np.ndarray) -> np.ndarray:
    delta = other.astype(np.float64) - base.astype(np.float64)
    # All three arms use the same recipient foreground.  Candidate scoring can
    # therefore operate on backgrounds by zeroing that foreground support; the
    # result is exactly the metric obtained after compositing the three arms.
    delta[mask] = 0.0
    grey = delta.mean(axis=2)
    dilated = ndimage.binary_dilation(mask, iterations=3)
    eroded = ndimage.binary_erosion(mask, iterations=3)
    ring = dilated ^ eroded
    boundary = float(np.mean(np.abs(delta[ring]))) if ring.any() else 0.0
    return np.asarray(
        [
            np.mean(np.abs(delta)),
            np.sqrt(np.mean(np.square(delta))),
            np.linalg.norm(delta.mean(axis=(0, 1))),
            np.mean(np.abs(grey - ndimage.gaussian_filter(grey, 2.0))),
            boundary,
        ],
        dtype=np.float64,
    )


def metric_components(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache the linear Gaussian-filter terms for one scene and one mask."""
    masked = image.astype(np.float64).copy()
    masked[mask] = 0.0
    grey = masked.mean(axis=2)
    return masked, grey, ndimage.gaussian_filter(grey, 2.0)


def cached_low_level_metrics(
    base: tuple[np.ndarray, np.ndarray, np.ndarray],
    other: tuple[np.ndarray, np.ndarray, np.ndarray],
    boundary_ring: np.ndarray,
) -> np.ndarray:
    delta = other[0] - base[0]
    grey = other[1] - base[1]
    lowpass = other[2] - base[2]
    boundary = float(np.mean(np.abs(delta[boundary_ring]))) if boundary_ring.any() else 0.0
    return np.asarray(
        [
            np.mean(np.abs(delta)),
            np.sqrt(np.mean(np.square(delta))),
            np.linalg.norm(delta.mean(axis=(0, 1))),
            np.mean(np.abs(grey - lowpass)),
            boundary,
        ],
        dtype=np.float64,
    )


def score_metrics(semantic: np.ndarray, null: np.ndarray) -> tuple[float, float, np.ndarray]:
    safe_semantic = np.maximum(semantic, EPSILON)
    safe_null = np.maximum(null, EPSILON)
    ratios = np.maximum(safe_semantic, safe_null) / np.minimum(safe_semantic, safe_null)
    score = float(np.sqrt(np.square(np.log(safe_semantic / safe_null)).sum()))
    return score, float(ratios.max()), ratios


def lineage_sets(value: Any) -> tuple[set[str], set[str]]:
    groups: set[str] = set()
    hashes: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key == "source_group_id" and isinstance(child, str):
                    groups.add(child)
                elif key in {"image_sha256", "annotation_sha256"} and isinstance(child, str):
                    hashes.add(child)
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return groups, hashes


def exact_keys(value: dict[str, Any], expected: Iterable[str], label: str) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{label}_KEYS:{sorted(value)}")


def candidate_key(recipient_group: str, base_group: str, semantic_group: str, null_group: str) -> tuple[str, ...]:
    return tuple(sha_text(value) for value in (recipient_group, base_group, semantic_group, null_group))


def solve_assignment(
    candidates: list[tuple[float, tuple[str, ...], int, int, int, int]],
    candidate_indices: list[int],
    objective_values: np.ndarray,
    recipient_count: int,
    scene_count: int,
    required_pairs: int,
    time_limit: float,
) -> tuple[Any, list[int]]:
    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    for column, original_index in enumerate(candidate_indices):
        candidate = candidates[original_index]
        for row_index in (0, 1 + candidate[2], 1 + recipient_count + candidate[3], 1 + recipient_count + candidate[4], 1 + recipient_count + candidate[5]):
            row_indices.append(row_index); col_indices.append(column); data.append(1.0)
    matrix = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(1 + recipient_count + scene_count, len(candidate_indices))).tocsr()
    lower = np.zeros(matrix.shape[0], dtype=np.float64); lower[0] = required_pairs
    upper = np.ones(matrix.shape[0], dtype=np.float64); upper[0] = required_pairs
    result = milp(
        objective_values[candidate_indices],
        integrality=np.ones(len(candidate_indices)),
        bounds=Bounds(np.zeros(len(candidate_indices)), np.ones(len(candidate_indices))),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True, "mip_rel_gap": 0.0, "time_limit": time_limit},
    )
    selected = [] if result.x is None else [candidate_indices[index] for index, value in enumerate(result.x) if value > 0.5]
    return result, selected


def self_test() -> None:
    base = np.zeros((16, 16, 3), dtype=np.float32)
    semantic = base.copy(); semantic[:, :, 0] = 0.25
    null = base.copy(); null[:, :, 1] = 0.25
    mask = np.zeros((16, 16), dtype=np.bool_); mask[5:11, 5:11] = True
    a = low_level_metrics(base, semantic, mask)
    b = low_level_metrics(base, null, mask)
    score, ratio, ratios = score_metrics(a, b)
    assert a.shape == (5,) and b.shape == (5,)
    assert math.isfinite(score) and math.isfinite(ratio) and np.all(np.isfinite(ratios))
    assert preprocess.__module__ == "__main__"
    print(json.dumps({"self_test": "PASS", "metrics": list(METRICS)}))


def main() -> None:
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES_MUST_BE_EMPTY")
    forbidden_modules = {"torch", "timm", "tensorflow", "jax"}
    if forbidden_modules.intersection(sys.modules):
        raise RuntimeError("MODEL_FRAMEWORK_IMPORTED")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"FRESH_ROOT_ALREADY_EXISTS:{OUTPUT_ROOT}")
    print(f"[matching] start attempt={ATTEMPT_ID} cpu_only=true", flush=True)

    accepted = load_json(ACCEPTED_PATH)
    contract = load_json(CONTRACT_PATH)
    ledger = load_json(LEDGER_PATH)
    matching = contract["matching"]
    asset_policy = contract["asset_policy"]
    if matching["low_level_metrics"] != list(METRICS):
        raise ValueError("METRIC_CONTRACT_MISMATCH")
    if matching["selection_objective"] != "MIN_LOG_RATIO_DISTANCE_SEMANTIC_VS_WITHIN_STRATUM_NULL":
        raise ValueError("SELECTION_OBJECTIVE_MISMATCH")
    if matching["tie_break"] != "SHA256_SOURCE_GROUP_ID_ASCENDING":
        raise ValueError("TIE_BREAK_MISMATCH")
    if accepted["legacy_lineage_ledger_sha256"] != sha_file(LEDGER_PATH):
        raise ValueError("LEDGER_BINDING_MISMATCH")

    legacy_groups, legacy_hashes = lineage_sets(ledger)
    scene_rows = accepted["records"]
    if len(scene_rows) != 43:
        raise ValueError("I7_ACCEPTED_SCENE_COUNT_MISMATCH")
    if Counter(row["context_label"] for row in scene_rows) != Counter({"AQUATIC_CONTEXT": 19, "NON_AQUATIC_CONTEXT": 24}):
        raise ValueError("I7_CONTEXT_COUNTS_MISMATCH")

    scenes: list[dict[str, Any]] = []
    input_accesses: list[dict[str, Any]] = []
    for path in (ACCEPTED_PATH, CONTRACT_PATH, LEDGER_PATH):
        input_accesses.append({"path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "role": "CONTROL_FILE"})
    for row in scene_rows:
        exact_keys(row, ("artifact_union_reject", "asset_id", "card_id", "context_label", "foreground_removal_applied", "global_exposure_status", "image_path", "image_sha256", "legacy_lineage_overlap", "license_record_id", "producer_operations", "reviewer_labels", "scene_only", "source_group_id"), "SCENE_RECORD")
        if row["image_path"].split("/")[:2] != ["private", "scene_originals"]:
            raise ValueError("SCENE_PATH_OUTSIDE_BINDING")
        path = ASSET_ROOT / row["image_path"]
        digest = sha_file(path)
        if digest != row["image_sha256"]:
            raise ValueError(f"SCENE_HASH_MISMATCH:{row['asset_id']}")
        if not row["scene_only"] or row["foreground_removal_applied"] or row["producer_operations"]:
            raise ValueError(f"SCENE_POLICY_MISMATCH:{row['asset_id']}")
        if row["legacy_lineage_overlap"] or row["source_group_id"] in legacy_groups or digest in legacy_hashes:
            raise ValueError(f"SCENE_LINEAGE_OVERLAP:{row['asset_id']}")
        scenes.append({**row, "physical_path": str(path.relative_to(ROOT)), "array": preprocess(path)})
        input_accesses.append({"path": str(path.relative_to(ROOT)), "sha256": digest, "role": "I7_BOUND_SCENE_ORIGINAL"})
    print(f"[matching] loaded accepted_scenes={len(scenes)} elapsed={time.monotonic()-started:.1f}s", flush=True)

    recipient_paths = sorted(path for path in RECIPIENT_IMAGE_ROOT.iterdir() if path.is_file())
    mask_paths = sorted(path for path in RECIPIENT_MASK_ROOT.iterdir() if path.is_file())
    if len(recipient_paths) != int(asset_policy["minimum_recipient_candidates"]) or len(mask_paths) != len(recipient_paths):
        raise ValueError("RECIPIENT_COUNT_MISMATCH")
    masks_by_id: dict[str, list[Path]] = {}
    for path in mask_paths:
        masks_by_id.setdefault(path.name.split("_", 1)[0], []).append(path)
    recipients: list[dict[str, Any]] = []
    for image_path in recipient_paths:
        recipient_id = image_path.stem
        matches = masks_by_id.get(recipient_id, [])
        if len(matches) != 1:
            raise ValueError(f"RECIPIENT_MASK_BINDING:{recipient_id}")
        mask_path = matches[0]
        image_digest = sha_file(image_path)
        mask_digest = sha_file(mask_path)
        source_group_id = f"OPENIMAGES_TRAIN_IMAGE:{recipient_id}"
        if source_group_id in legacy_groups or image_digest in legacy_hashes or mask_digest in legacy_hashes:
            raise ValueError(f"RECIPIENT_LINEAGE_OVERLAP:{recipient_id}")
        image = preprocess(image_path)
        mask = preprocess(mask_path, mask=True)
        if image.shape[:2] != mask.shape or not mask.any() or mask.all():
            raise ValueError(f"RECIPIENT_MASK_GEOMETRY:{recipient_id}")
        recipients.append({
            "recipient_id": recipient_id,
            "source_group_id": source_group_id,
            "image_path": str(image_path.relative_to(ROOT)),
            "image_sha256": image_digest,
            "mask_path": str(mask_path.relative_to(ROOT)),
            "mask_sha256": mask_digest,
            "mask_fraction": float(mask.mean()),
            "image": image,
            "mask": mask,
        })
        input_accesses.extend([
            {"path": str(image_path.relative_to(ROOT)), "sha256": image_digest, "role": "RECIPIENT_ORIGINAL"},
            {"path": str(mask_path.relative_to(ROOT)), "sha256": mask_digest, "role": "RECIPIENT_MASK"},
        ])
    print(f"[matching] loaded recipients={len(recipients)} masks={len(mask_paths)} elapsed={time.monotonic()-started:.1f}s", flush=True)

    aquatic = [index for index, row in enumerate(scenes) if row["context_label"] == "AQUATIC_CONTEXT"]
    non_aquatic = [index for index, row in enumerate(scenes) if row["context_label"] == "NON_AQUATIC_CONTEXT"]
    max_ratio_limit = float(matching["max_per_metric_ratio"])
    max_score_limit = float(matching["max_aggregate_log_ratio_distance"])
    candidates: list[tuple[float, tuple[str, ...], int, int, int, int]] = []
    best_score_by_recipient: list[float | None] = [None] * len(recipients)
    for recipient_index, recipient in enumerate(recipients):
        mask = recipient["mask"]
        ring = ndimage.binary_dilation(mask, iterations=3) ^ ndimage.binary_erosion(mask, iterations=3)
        scene_components = [metric_components(scene["array"], mask) for scene in scenes]
        for base_index in non_aquatic:
            base = scene_components[base_index]
            semantic_metrics = {index: cached_low_level_metrics(base, scene_components[index], ring) for index in aquatic}
            null_metrics = {index: cached_low_level_metrics(base, scene_components[index], ring) for index in non_aquatic if index != base_index}
            for semantic_index, semantic_values in semantic_metrics.items():
                for null_index, null_values in null_metrics.items():
                    score, max_ratio, _ = score_metrics(semantic_values, null_values)
                    if score <= max_score_limit and max_ratio <= max_ratio_limit:
                        key = candidate_key(recipient["source_group_id"], scenes[base_index]["source_group_id"], scenes[semantic_index]["source_group_id"], scenes[null_index]["source_group_id"])
                        candidates.append((score, key, recipient_index, base_index, semantic_index, null_index))
                        prior = best_score_by_recipient[recipient_index]
                        best_score_by_recipient[recipient_index] = score if prior is None else min(prior, score)
        print(f"[matching] recipient={recipient_index+1}/{len(recipients)} id={recipient['recipient_id']} valid_candidates={len(candidates)} elapsed={time.monotonic()-started:.1f}s", flush=True)

    candidates.sort(key=lambda row: (row[0], row[1]))
    solution: list[int] = []
    solver_summary: dict[str, Any]
    required_pairs = int(asset_policy["final_pair_count"])
    if candidates:
        n = len(candidates)
        tie_ranks = np.empty(n, dtype=np.float64)
        for rank, index in enumerate(sorted(range(n), key=lambda i: candidates[i][1])):
            tie_ranks[index] = rank / max(n - 1, 1)
        objective = np.asarray([row[0] for row in candidates], dtype=np.float64) + TIE_EPSILON * tie_ranks
        per_recipient = [[index for index, row in enumerate(candidates) if row[2] == recipient_index] for recipient_index in range(len(recipients))]
        upper_result = None; upper_solution: list[int] = []; upper_subset_count = 0
        for per_recipient_limit in (64, 256, 1024):
            upper_indices = sorted({index for indices in per_recipient for index in indices[:per_recipient_limit]})
            upper_result, upper_solution = solve_assignment(candidates, upper_indices, objective, len(recipients), len(scenes), required_pairs, 30.0)
            print(f"[matching] upper_bound limit_per_recipient={per_recipient_limit} subset={len(upper_indices)} success={upper_result.success} selected={len(upper_solution)} elapsed={time.monotonic()-started:.1f}s", flush=True)
            upper_subset_count = len(upper_indices)
            if len(upper_solution) == required_pairs:
                break
        if len(upper_solution) != required_pairs:
            raise RuntimeError("NO_FEASIBLE_UPPER_BOUND_IN_SAFE_PRUNING_PREFIXES")
        incumbent = float(objective[upper_solution].sum())
        recipient_minimum = [min(objective[index] for index in indices) for indices in per_recipient]
        safe_indices = []
        for index, candidate in enumerate(candidates):
            other_lower_bounds = sorted(value for recipient_index, value in enumerate(recipient_minimum) if recipient_index != candidate[2])[: required_pairs - 1]
            if objective[index] + sum(other_lower_bounds) <= incumbent + 1e-12:
                safe_indices.append(index)
        print(f"[matching] safe_prune full={n} retained={len(safe_indices)} incumbent={incumbent:.12g} elapsed={time.monotonic()-started:.1f}s", flush=True)
        result, solution = solve_assignment(candidates, safe_indices, objective, len(recipients), len(scenes), required_pairs, 300.0)
        solver_summary = {
            "candidate_count": n,
            "safe_pruned_candidate_count": len(safe_indices),
            "upper_bound_subset_count": upper_subset_count,
            "incumbent_upper_bound": incumbent,
            "safe_pruning_rule": "KEEP_IF_CANDIDATE_OBJECTIVE_PLUS_SEVEN_LOWEST_OTHER_RECIPIENT_MINIMA_LE_INCUMBENT",
            "message": str(result.message),
            "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
            "success": bool(result.success),
        }
        print(f"[matching] exact_solver success={result.success} selected={len(solution)} retained={len(safe_indices)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    else:
        solver_summary = {"candidate_count": 0, "message": "NO_CALIPER_VALID_CANDIDATES", "mip_gap": None, "success": False}

    pass_matching = len(solution) == required_pairs
    selected = [candidates[index] for index in solution]
    selected.sort(key=lambda row: sha_text(recipients[row[2]]["source_group_id"]))
    selected_recipient_indices = {row[2] for row in selected}
    reserve_indices = sorted(
        (index for index in range(len(recipients)) if index not in selected_recipient_indices),
        key=lambda index: (math.inf if best_score_by_recipient[index] is None else best_score_by_recipient[index], sha_text(recipients[index]["source_group_id"])),
    )
    reserve_indices = reserve_indices[: int(asset_policy["reserve_recipient_count"])]

    records: list[dict[str, Any]] = []
    output_arrays: dict[str, np.ndarray] = {}
    for pair_index, candidate in enumerate(selected):
        score, key, recipient_index, base_index, semantic_index, null_index = candidate
        recipient = recipients[recipient_index]
        base_scene = scenes[base_index]
        semantic_scene = scenes[semantic_index]
        null_scene = scenes[null_index]
        base_arm = composite(recipient["image"], recipient["mask"], base_scene["array"] )
        semantic_arm = composite(recipient["image"], recipient["mask"], semantic_scene["array"] )
        null_arm = composite(recipient["image"], recipient["mask"], null_scene["array"] )
        semantic_values = low_level_metrics(base_arm, semantic_arm, recipient["mask"])
        null_values = low_level_metrics(base_arm, null_arm, recipient["mask"])
        recomputed_score, max_ratio, ratios = score_metrics(semantic_values, null_values)
        if abs(recomputed_score - score) > 1e-12:
            raise RuntimeError("CANDIDATE_SCORE_RECOMPUTE_MISMATCH")
        pair_id = f"n2_v2_pair_{pair_index:02d}"
        def public_scene(row: dict[str, Any]) -> dict[str, Any]:
            return {key: row[key] for key in ("asset_id", "context_label", "image_sha256", "physical_path", "source_group_id")}
        records.append({
            "pair_id": pair_id,
            "recipient": {key: recipient[key] for key in ("recipient_id", "source_group_id", "image_path", "image_sha256", "mask_path", "mask_sha256", "mask_fraction")},
            "base_scene": public_scene(base_scene),
            "semantic_scene": public_scene(semantic_scene),
            "within_stratum_null_scene": public_scene(null_scene),
            "semantic_metrics": dict(zip(METRICS, map(float, semantic_values))),
            "within_stratum_null_metrics": dict(zip(METRICS, map(float, null_values))),
            "per_metric_ratios": dict(zip(METRICS, map(float, ratios))),
            "aggregate_log_ratio_distance": float(score),
            "max_per_metric_ratio": float(max_ratio),
            "matched": bool(score <= max_score_limit and max_ratio <= max_ratio_limit),
            "tie_break_sha256s": list(key),
        })
        output_arrays[f"{pair_id}_base"] = base_arm
        output_arrays[f"{pair_id}_semantic"] = semantic_arm
        output_arrays[f"{pair_id}_within_stratum_null"] = null_arm
        output_arrays[f"{pair_id}_recipient_mask"] = recipient["mask"]

    used_backgrounds = [record[role]["source_group_id"] for record in records for role in ("base_scene", "semantic_scene", "within_stratum_null_scene")]
    background_max_reuse_observed = max(Counter(used_backgrounds).values(), default=0)
    all_pairs_match = pass_matching and len(records) == required_pairs and all(record["matched"] for record in records) and background_max_reuse_observed <= int(asset_policy["background_max_reuse"])
    status = "PASS_INPUT_IDENTIFIABILITY_FROZEN" if all_pairs_match else "NO_GO_NOT_IDENTIFIABLE"
    reserves = [
        {
            "reserve_rank": rank,
            "recipient_id": recipients[index]["recipient_id"],
            "source_group_id": recipients[index]["source_group_id"],
            "source_group_sha256": sha_text(recipients[index]["source_group_id"]),
            "image_path": recipients[index]["image_path"],
            "image_sha256": recipients[index]["image_sha256"],
            "mask_path": recipients[index]["mask_path"],
            "mask_sha256": recipients[index]["mask_sha256"],
            "selection_score": best_score_by_recipient[index],
        }
        for rank, index in enumerate(reserve_indices, start=1)
    ]

    manifest = {
        "schema_version": 1,
        "stage_id": "N2_V2_LOW_LEVEL_MATCHING",
        "attempt_id": ATTEMPT_ID,
        "status": status,
        "recovery": {
            "parent_attempt_id": "N2_V2_MATCHING_R1_20260922",
            "parent_authority": "INVALID_PROTOCOL",
            "amendment_path": "N2-V2-MATCH-R2恢复修订.md",
            "scientific_definition_changed": False,
        },
        "source_bindings": {
            "accepted_registry_path": str(ACCEPTED_PATH.relative_to(ROOT)),
            "accepted_registry_sha256": sha_file(ACCEPTED_PATH),
            "contract_path": str(CONTRACT_PATH.relative_to(ROOT)),
            "contract_sha256": sha_file(CONTRACT_PATH),
            "legacy_lineage_ledger_path": str(LEDGER_PATH.relative_to(ROOT)),
            "legacy_lineage_ledger_sha256": sha_file(LEDGER_PATH),
            "physical_asset_store": str(ASSET_ROOT.relative_to(ROOT)),
            "physical_asset_binding": "I5_FROZEN_ASSET_STORE_CONTENT_HASH_BOUND_BY_I7_ACCEPTED_REGISTRY",
        },
        "matching_definition": {
            "semantic_contrast": "AQUATIC_CONTEXT_MINUS_NON_AQUATIC_BASE",
            "within_stratum_null_contrast": "NON_AQUATIC_CONTEXT_MINUS_DIFFERENT_NON_AQUATIC_BASE",
            "recipient_compositing": "HARD_BINARY_MASK_SAME_RECIPIENT_ALL_THREE_ARMS",
            "resize_short_edge": RESIZE_SHORT,
            "center_crop_size": CROP_SIZE,
            "image_interpolation": "PIL_BICUBIC",
            "mask_interpolation": "PIL_NEAREST",
            "metrics": list(METRICS),
            "epsilon": EPSILON,
            "selection_objective": matching["selection_objective"],
            "tie_break": matching["tie_break"],
            "global_assignment": "BINARY_MILP_EXACTLY_8_UNIQUE_RECIPIENTS_AND_UNIQUE_BACKGROUNDS",
        },
        "calipers": {
            "max_per_metric_ratio": max_ratio_limit,
            "max_aggregate_log_ratio_distance": max_score_limit,
            "max_unmatched_rate": float(matching["max_unmatched_rate"]),
            "background_max_reuse": int(asset_policy["background_max_reuse"]),
        },
        "selection": {
            "candidate_recipient_count": len(recipients),
            "accepted_scene_count": len(scenes),
            "caliper_valid_candidate_count": len(candidates),
            "final_pair_count": len(records),
            "matched_pair_count": sum(record["matched"] for record in records),
            "unmatched_rate": 0.0 if records and all(record["matched"] for record in records) else (1.0 - sum(record["matched"] for record in records) / required_pairs),
            "reserve_recipient_count": len(reserves),
            "background_max_reuse_observed": background_max_reuse_observed,
            "solver": solver_summary,
        },
        "records": records,
        "reserve_recipients": reserves,
        "lineage_audit": {
            "legacy_source_group_count": len(legacy_groups),
            "legacy_asset_hash_count": len(legacy_hashes),
            "selected_source_group_intersection": [],
            "selected_asset_hash_intersection": [],
            "zero_intersection": True,
        },
        "input_access_manifest": sorted(input_accesses, key=lambda row: row["path"]),
        "prohibited_activity_counters": {
            "checkpoint_reads": 0,
            "model_loads": 0,
            "inference_runs": 0,
            "behavior_outcome_reads": 0,
            "activation_runs": 0,
            "endpoint_runs": 0,
            "path_nns_runs": 0,
            "jvp_runs": 0,
            "intervention_runs": 0,
            "gpu_operations": 0,
        },
        "next_gate": "STOP_BEFORE_BEHAVIOR_GATE_AND_N3_REQUIRES_EXPLICIT_USER_AUTHORIZATION",
    }

    OUTPUT_ROOT.mkdir(parents=False, exist_ok=False)
    manifest_path = OUTPUT_ROOT / "pair_manifest.json"
    array_path = OUTPUT_ROOT / "pair_inputs.npz"
    write_json(manifest_path, manifest)
    if output_arrays:
        np.savez_compressed(array_path, **output_arrays)
    production_receipt = {
        "schema_version": 1,
        "stage_id": "N2_V2_LOW_LEVEL_MATCHING",
        "attempt_id": ATTEMPT_ID,
        "production_status": "SEALED_AWAITING_INDEPENDENT_VALIDATION",
        "scientific_status": status,
        "pair_manifest_path": str(manifest_path.relative_to(ROOT)),
        "pair_manifest_sha256": sha_file(manifest_path),
        "pair_inputs_path": str(array_path.relative_to(ROOT)) if array_path.exists() else None,
        "pair_inputs_sha256": sha_file(array_path) if array_path.exists() else None,
        "builder_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "builder_sha256": sha_file(Path(__file__).resolve()),
        "python_executable_realpath": str(Path(sys.executable).resolve()),
        "library_versions": {"numpy": np.__version__, "pillow": Image.__version__},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "forbidden_modules_loaded": sorted(forbidden_modules.intersection(sys.modules)),
        "next_gate": manifest["next_gate"],
    }
    write_json(OUTPUT_ROOT / "production_receipt.json", production_receipt)
    print(json.dumps({"attempt_id": ATTEMPT_ID, "status": status, "final_pairs": len(records), "reserves": len(reserves), "valid_candidates": len(candidates), "output_root": str(OUTPUT_ROOT.relative_to(ROOT))}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
