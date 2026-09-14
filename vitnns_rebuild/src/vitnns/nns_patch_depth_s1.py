"""Frozen S1 pilot kernels for NNS patch/depth semantics.

This module contains deterministic geometry, P1/P2 controls, D1 reduction,
and D2 engineering-probe code.  GPU attribution remains delegated to the
already validated Stage Y grouped JVP/VJP implementation.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

STUDY_ID = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
STAGE_ID = "S1"
RUN_ID = "s1_r1_pilot_20260914"
PATCH_GRID = 14
PATCH_COUNT = 196
BLOCKS = tuple(range(12))
SHALLOW = (0, 1, 2, 3)
MIDDLE = (4, 5, 6, 7)
DEEP = (8, 9, 10, 11)
P1_SEED_NAMESPACE = "NNS_VIT_PATCH_DEPTH_S1_P1_POSITION_PERMUTATION_V1"
P2_SEED_NAMESPACE = "NNS_VIT_PATCH_DEPTH_S1_P2_MATCHED_CONTROL_V1"
TIE_NAMESPACE = "NNS_VIT_PATCH_DEPTH_S1_PATCH_TIE_V1"
PROBE_SEED_NAMESPACE = "NNS_VIT_PATCH_DEPTH_S1_PROBE_V1"


class S1Error(ValueError):
    pass


class S1TypedNA(S1Error):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def seed64(namespace: str, *parts: object) -> int:
    payload = namespace + "\0" + "\0".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def stable_descending(values: np.ndarray, namespace: str, *parts: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != PATCH_COUNT or not np.isfinite(array).all():
        raise S1Error("patch score must be finite [196]")
    tie = np.asarray([seed64(namespace, *parts, index) for index in range(PATCH_COUNT)], dtype=np.uint64)
    return np.lexsort((tie, -array))


def position_strata() -> np.ndarray:
    """Seven exact Chebyshev rings around the 14x14 image center."""
    result = np.empty((PATCH_GRID, PATCH_GRID), dtype=np.int64)
    for row in range(PATCH_GRID):
        for column in range(PATCH_GRID):
            result[row, column] = max(abs(2 * row - 13), abs(2 * column - 13)) // 2
    values = sorted(np.unique(result).tolist())
    if values != list(range(7)):
        raise S1Error("position strata construction drift")
    return result.reshape(-1)


POSITION_STRATA = position_strata()


def patch_occupancy(labels: np.ndarray, target_label: int) -> dict[str, np.ndarray]:
    value = np.asarray(labels)
    if value.shape != (224, 224) or not np.issubdtype(value.dtype, np.integer):
        raise S1TypedNA("NA_GEOMETRY_SIZE_MISMATCH", "mask must be integer [224,224]")
    allowed = (value == 0) | (value == 1000) | ((value >= 1) & (value <= 919))
    if not bool(allowed.all()):
        raise S1TypedNA("NA_MASK_LABEL_OUT_OF_RANGE", "mask contains an unsupported label")
    target = value == int(target_label)
    if not bool(target.any()):
        raise S1TypedNA("NA_TARGET_DISAPPEARED_AFTER_CROP", "target class is absent after crop")
    void = value == 1000
    if bool(void.any()):
        raise S1TypedNA("NA_VOID_MASK", "void pixels remain in the transformed mask")
    planes = {
        "background": value == 0,
        "other_category": (value != 0) & (value != 1000) & ~target,
        "void": void,
        "target": target,
    }
    result = {}
    for name, plane in planes.items():
        result[name] = plane.reshape(PATCH_GRID, 16, PATCH_GRID, 16).transpose(0, 2, 1, 3).mean((2, 3), dtype=np.float64).reshape(-1)
    if not np.allclose(sum(result.values()), 1.0, atol=0.0, rtol=0.0):
        raise S1Error("occupancy semantics do not close to one")
    return result


def source_patch_scores(matrix: np.ndarray, cls: np.ndarray) -> dict[str, Any]:
    t = np.asarray(matrix, dtype=np.float64)
    c = np.asarray(cls, dtype=np.float64)
    if t.shape != (PATCH_COUNT, PATCH_COUNT) or c.shape != (PATCH_COUNT,):
        raise S1Error("matrix/CLS geometry drift")
    if not np.isfinite(t).all() or not np.isfinite(c).all():
        raise S1Error("matrix/CLS contains NaN or Inf")
    signed = t.sum(1, dtype=np.float64) + c
    positive = np.maximum(signed, 0.0)
    negative = np.minimum(signed, 0.0)
    absolute = np.abs(signed)
    absolute_total = float(absolute.sum(dtype=np.float64))
    return {
        "signed": signed, "positive": positive, "negative": negative,
        "absolute": absolute, "signed_mass": float(signed.sum(dtype=np.float64)),
        "positive_mass": float(positive.sum(dtype=np.float64)),
        "negative_mass": float(negative.sum(dtype=np.float64)),
        "absolute_mass": absolute_total,
        "cancellation": None if absolute_total == 0.0 else 1.0 - abs(float(signed.sum(dtype=np.float64))) / absolute_total,
        "matrix_signed_mass": float(t.sum(dtype=np.float64)),
        "matrix_absolute_mass": float(np.abs(t).sum(dtype=np.float64)),
        "cls_signed_mass": float(c.sum(dtype=np.float64)),
        "cls_absolute_mass": float(np.abs(c).sum(dtype=np.float64)),
    }


def _stratified_permutation(source_id: str, block: int, repeat: int) -> np.ndarray:
    output = np.arange(PATCH_COUNT, dtype=np.int64)
    for stratum in range(7):
        indices = np.flatnonzero(POSITION_STRATA == stratum)
        generator = np.random.Generator(np.random.PCG64(seed64(P1_SEED_NAMESPACE, source_id, block, repeat, stratum)))
        output[indices] = generator.permutation(indices)
    return output


def p1_alignment(positive: np.ndarray, occupancy: np.ndarray, source_id: str, block: int, permutations: int) -> dict[str, Any]:
    weights = np.asarray(positive, dtype=np.float64).reshape(-1)
    target = np.asarray(occupancy, dtype=np.float64).reshape(-1)
    if weights.shape != (PATCH_COUNT,) or target.shape != (PATCH_COUNT,) or not np.isfinite(weights).all() or not np.isfinite(target).all():
        raise S1Error("P1 input contract drift")
    denominator = float(weights.sum(dtype=np.float64))
    if denominator <= 0.0:
        raise S1TypedNA("NA_ZERO_POSITIVE_MASS", "component has zero positive source-patch mass")
    raw = float(np.dot(weights, target) / denominator)
    controls = [float(np.dot(weights[_stratified_permutation(source_id, block, repeat)], target) / denominator) for repeat in range(int(permutations))]
    control_mean = math.fsum(controls) / len(controls)
    return {
        "status": "VALID", "raw_alignment": raw, "control_mean": control_mean,
        "corrected_alignment": raw - control_mean, "control_min": min(controls),
        "control_max": max(controls), "permutations": len(controls),
    }


def boundary_complexity(indices: Sequence[int]) -> int:
    selected = {int(item) for item in indices}
    if any(item not in range(PATCH_COUNT) for item in selected):
        raise S1Error("patch index out of range")
    boundary = 0
    for patch in selected:
        row, column = divmod(patch, PATCH_GRID)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            other_row, other_column = row + dy, column + dx
            if other_row not in range(PATCH_GRID) or other_column not in range(PATCH_GRID) or other_row * PATCH_GRID + other_column not in selected:
                boundary += 1
    return boundary


def matched_controls(selected: Sequence[int], source_id: str, block: int, budget: int, count: int, attempts: int, relative_boundary_tolerance: float) -> list[dict[str, Any]]:
    chosen = tuple(sorted(int(item) for item in selected))
    if len(chosen) != int(budget) or len(set(chosen)) != int(budget):
        raise S1Error("selected set does not equal frozen patch budget")
    per_stratum = {stratum: sum(POSITION_STRATA[item] == stratum for item in chosen) for stratum in range(7)}
    selected_boundary = boundary_complexity(chosen)
    tolerance = max(4, int(math.ceil(relative_boundary_tolerance * selected_boundary)))
    accepted: list[dict[str, Any]] = []
    seen = {chosen}
    # Dihedral transforms preserve the exact center-ring strata and boundary
    # complexity.  Try them in a hash-frozen order before random candidates.
    def transform(patch: int, transform_id: int) -> int:
        row, column = divmod(patch, PATCH_GRID)
        if transform_id >= 4:
            column = PATCH_GRID - 1 - column
        for _ in range(transform_id % 4):
            row, column = column, PATCH_GRID - 1 - row
        return row * PATCH_GRID + column
    transform_order = sorted(range(1, 8), key=lambda item: seed64(P2_SEED_NAMESPACE, source_id, block, budget, "dihedral", item))
    for transform_id in transform_order:
        candidate_tuple = tuple(sorted(transform(item, transform_id) for item in chosen))
        if candidate_tuple in seen:
            continue
        seen.add(candidate_tuple)
        accepted.append({
            "indices": list(candidate_tuple), "attempt": -transform_id, "boundary_complexity": selected_boundary,
            "boundary_absolute_error": 0, "overlap_count": len(set(candidate_tuple) & set(chosen)),
            "stratum_counts": [per_stratum[index] for index in range(7)],
        })
        if len(accepted) == int(count):
            return accepted
    for attempt in range(int(attempts)):
        candidate = []
        for stratum, needed in per_stratum.items():
            pool = np.flatnonzero(POSITION_STRATA == stratum)
            generator = np.random.Generator(np.random.PCG64(seed64(P2_SEED_NAMESPACE, source_id, block, budget, attempt, stratum)))
            candidate.extend(int(item) for item in generator.choice(pool, size=needed, replace=False))
        candidate_tuple = tuple(sorted(candidate))
        if candidate_tuple in seen:
            continue
        seen.add(candidate_tuple)
        boundary = boundary_complexity(candidate_tuple)
        error = abs(boundary - selected_boundary)
        if error <= tolerance:
            accepted.append({
                "indices": list(candidate_tuple), "attempt": attempt, "boundary_complexity": boundary,
                "boundary_absolute_error": error, "overlap_count": len(set(candidate_tuple) & set(chosen)),
                "stratum_counts": [per_stratum[index] for index in range(7)],
            })
            if len(accepted) == int(count):
                return accepted
    raise S1TypedNA("NA_MATCHED_CONTROL_UNAVAILABLE", "position/boundary matched controls could not be formed")


def topk_and_controls(positive: np.ndarray, source_id: str, block: int, budget: int, count: int, attempts: int, relative_boundary_tolerance: float) -> dict[str, Any]:
    order = stable_descending(positive, TIE_NAMESPACE, source_id, block, budget)
    selected = [int(item) for item in order[:int(budget)]]
    controls = matched_controls(selected, source_id, block, budget, count, attempts, relative_boundary_tolerance)
    return {
        "selected": selected, "selected_boundary_complexity": boundary_complexity(selected),
        "selected_positive_count": sum(float(np.asarray(positive)[item]) > 0.0 for item in selected),
        "controls": controls,
    }


def apply_patch_mask(batch_image: Any, sets: Sequence[Sequence[int]], operator: str) -> Any:
    import torch
    if batch_image.shape != (1, 3, 224, 224):
        raise S1Error("occlusion input must be [1,3,224,224]")
    outputs = batch_image.repeat(len(sets), 1, 1, 1)
    if operator == "normalized_zero":
        replacement = torch.zeros((len(sets), 3, 1, 1), device=outputs.device, dtype=outputs.dtype)
    elif operator == "image_channel_mean":
        replacement = batch_image.mean((2, 3), keepdim=True).repeat(len(sets), 1, 1, 1)
    else:
        raise S1Error("unknown mask operator")
    for row, indices in enumerate(sets):
        for patch in indices:
            y, x = divmod(int(patch), PATCH_GRID)
            outputs[row, :, y * 16:(y + 1) * 16, x * 16:(x + 1) * 16] = replacement[row]
    return outputs


def operator_artifact_score(image: Any, sets: Sequence[Sequence[int]], operator: str) -> float:
    """Outcome-free seam energy used only for S2 operator recommendation."""
    import torch
    masked = apply_patch_mask(image, sets, operator)
    original = image.repeat(masked.shape[0], 1, 1, 1)
    changed = (masked - original).abs()
    horizontal = changed[:, :, :, 1:] - changed[:, :, :, :-1]
    vertical = changed[:, :, 1:, :] - changed[:, :, :-1, :]
    return float(torch.cat((horizontal.abs().reshape(-1), vertical.abs().reshape(-1))).mean())


def d1_contrast(block_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_block = {int(row["block"]): row for row in block_rows if row.get("status") == "VALID"}
    if any(block not in by_block for block in BLOCKS):
        raise S1TypedNA("NA_INCOMPLETE_DEPTH_TRAJECTORY", "D1 requires all 12 valid blocks")
    values = {block: float(by_block[block]["corrected_alignment"]) for block in BLOCKS}
    mean = lambda blocks: math.fsum(values[block] for block in blocks) / len(blocks)
    return {
        "status": "VALID", "shallow_mean": mean(SHALLOW), "middle_mean": mean(MIDDLE),
        "deep_mean": mean(DEEP), "deep_minus_shallow": mean(DEEP) - mean(SHALLOW),
        "trajectory": [values[block] for block in BLOCKS], "scientific_decision": None,
    }


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    intercept: float
    alpha: float


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> RidgeModel:
    x = np.asarray(train_x, dtype=np.float64); y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[0] == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise S1Error("invalid ridge training arrays")
    mean = x.mean(0); scale = x.std(0); scale[scale == 0.0] = 1.0
    z = (x - mean) / scale; y_mean = float(y.mean()); centered = y - y_mean
    # Dual form is stable and much cheaper for 196 patches x 768 channels.
    dual = np.linalg.solve(z @ z.T + float(alpha) * np.eye(z.shape[0]), centered)
    weight = z.T @ dual
    return RidgeModel(mean, scale, weight, y_mean, float(alpha))


def predict_ridge(model: RidgeModel, value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != model.mean.size or not np.isfinite(x).all():
        raise S1Error("invalid ridge prediction array")
    return ((x - model.mean) / model.scale) @ model.weight + model.intercept


def probe_split(records: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    by_class: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_class.setdefault(str(record["annotation_wnid"]), []).append(record)
    train, validation = [], []
    for wnid in sorted(by_class):
        rows = sorted(by_class[wnid], key=lambda row: (canonical_digest([PROBE_SEED_NAMESPACE, row["source_image_id"]]), row["source_image_id"]))
        if len(rows) != 2:
            raise S1Error("pilot engineering probe requires exactly two images per class")
        train.append(rows[0]); validation.append(rows[1])
    if {row["source_image_id"] for row in train} & {row["source_image_id"] for row in validation}:
        raise S1Error("probe split leakage")
    return train, validation


def low_level_features(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.float64)
    if image.shape != (3, 224, 224):
        raise S1Error("low-level control expects RGB [3,224,224]")
    pooled = image.reshape(3, 14, 16, 14, 16).transpose(0, 1, 3, 2, 4).mean((3, 4)).reshape(3, PATCH_COUNT).T
    yy, xx = np.meshgrid((np.arange(14) + 0.5) / 14.0, (np.arange(14) + 0.5) / 14.0, indexing="ij")
    return np.column_stack((xx.reshape(-1), yy.reshape(-1), pooled))


def serialize_ridge(model: RidgeModel) -> dict[str, Any]:
    return {
        "alpha": model.alpha, "intercept": model.intercept,
        "mean_sha256": hashlib.sha256(np.ascontiguousarray(model.mean, dtype="<f8").tobytes()).hexdigest(),
        "scale_sha256": hashlib.sha256(np.ascontiguousarray(model.scale, dtype="<f8").tobytes()).hexdigest(),
        "weight_sha256": hashlib.sha256(np.ascontiguousarray(model.weight, dtype="<f8").tobytes()).hexdigest(),
        "feature_count": int(model.weight.size),
    }
