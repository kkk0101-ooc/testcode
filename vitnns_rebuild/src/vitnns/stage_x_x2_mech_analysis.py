"""Pure-NumPy X2-MECH operators for synthetic P0 tests and future M2."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

SSE_ATOL = 1e-10
SSE_RTOL = 1e-10


class X2MechAnalysisError(ValueError):
    pass


def seed64(namespace: str, *parts: object) -> int:
    text = "|".join((namespace, *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _map(value: np.ndarray, name: str = "map") -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise X2MechAnalysisError(f"{name} must be a numeric two-dimensional array")
    if not np.isfinite(array).all():
        raise X2MechAnalysisError(f"{name} contains non-finite values")
    return array.astype(np.float64, copy=False)


def _mask(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape or mask.dtype != np.bool_:
        raise X2MechAnalysisError("mask must be boolean and match map shape")
    return mask


def aggregate_blocks(value: np.ndarray, valid_mask: np.ndarray, block: int) -> dict[str, np.ndarray]:
    array = _map(value)
    mask = _mask(valid_mask, array.shape)
    if block <= 0 or array.shape[0] % block or array.shape[1] % block:
        raise X2MechAnalysisError("block must tile the map exactly")
    rows, cols = array.shape[0] // block, array.shape[1] // block
    values = array.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)
    masks = mask.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)
    complete = masks.all(axis=(2, 3))
    signed = values.sum(axis=(2, 3), dtype=np.float64)
    absolute = np.abs(values).sum(axis=(2, 3), dtype=np.float64)
    signed[~complete] = np.nan
    absolute[~complete] = np.nan
    return {"signed_sum": signed, "absolute_mass_sum": absolute, "valid_blocks": complete}


def top_fraction_indices(values: np.ndarray, valid: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    array = _map(values, "values")
    mask = _mask(valid, array.shape) & np.isfinite(array)
    indices = np.flatnonzero(mask.reshape(-1))
    if not len(indices) or not 0.0 < fraction <= 1.0:
        raise X2MechAnalysisError("top fraction needs valid units and a fraction in (0,1]")
    count = int(math.ceil(fraction * len(indices)))
    selected = np.argpartition(array.reshape(-1)[indices], -count)[-count:]
    return np.sort(indices[selected])


def top_fraction_jaccard(left: np.ndarray, right: np.ndarray, valid: np.ndarray, *, absolute: bool) -> float:
    a = np.abs(left) if absolute else np.asarray(left)
    b = np.abs(right) if absolute else np.asarray(right)
    ia = top_fraction_indices(a, valid)
    ib = top_fraction_indices(b, valid)
    return float(np.intersect1d(ia, ib).size / np.union1d(ia, ib).size)


def valid_block_centers(valid_blocks: np.ndarray, block: int) -> np.ndarray:
    valid = np.asarray(valid_blocks)
    if valid.ndim != 2 or valid.dtype != np.bool_:
        raise X2MechAnalysisError("valid_blocks must be a boolean matrix")
    row, col = np.nonzero(valid)
    return np.column_stack(((row + 0.5) * block - 0.5, (col + 0.5) * block - 0.5))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import spearmanr
    a = np.asarray(left).reshape(-1); b = np.asarray(right).reshape(-1)
    if len(a) < 2 or np.unique(a).size < 2 or np.unique(b).size < 2: return float("nan")
    value = float(spearmanr(a, b).statistic)
    return value


def _centroid(weights: np.ndarray, coordinates: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    total = float(weights.sum(dtype=np.float64))
    return fallback if total <= np.finfo(np.float64).tiny else (coordinates * weights[:, None]).sum(axis=0) / total


def stability_metrics(clean: np.ndarray, transformed: np.ndarray, valid_mask: np.ndarray, block: int) -> dict[str, float]:
    from skimage.metrics import structural_similarity
    a = _map(clean, "clean"); b = _map(transformed, "transformed")
    if a.shape != b.shape: raise X2MechAnalysisError("map shapes differ")
    mask = _mask(valid_mask, a.shape)
    left = aggregate_blocks(a, mask, block); right = aggregate_blocks(b, mask, block); valid = left["valid_blocks"]
    signed_a = left["signed_sum"][valid]; signed_b = right["signed_sum"][valid]
    mass_a = left["absolute_mass_sum"][valid]; mass_b = right["absolute_mass_sum"][valid]
    if not len(signed_a): raise X2MechAnalysisError("no complete valid units")
    signed_map_a = np.nan_to_num(left["signed_sum"], nan=0.0); signed_map_b = np.nan_to_num(right["signed_sum"], nan=0.0)
    row, col = np.nonzero(valid); r0, r1, c0, c1 = row.min(), row.max() + 1, col.min(), col.max() + 1
    crop_a = signed_map_a[r0:r1, c0:c1]; crop_b = signed_map_b[r0:r1, c0:c1]
    data_range = max(float(max(signed_a.max(), signed_b.max()) - min(signed_a.min(), signed_b.min())), 1e-12)
    win = min(7, crop_a.shape[0] if crop_a.shape[0] % 2 else crop_a.shape[0] - 1, crop_a.shape[1] if crop_a.shape[1] % 2 else crop_a.shape[1] - 1)
    if win < 3: raise X2MechAnalysisError("too few valid units for signed SSIM")
    ssim = float(structural_similarity(crop_a, crop_b, data_range=data_range, win_size=win))
    coordinates = valid_block_centers(valid, block); center = np.asarray([(a.shape[0] - 1) / 2, (a.shape[1] - 1) / 2])
    positive_a = np.maximum(signed_a, 0); positive_b = np.maximum(signed_b, 0)
    negative_a = np.maximum(-signed_a, 0); negative_b = np.maximum(-signed_b, 0)
    positive_shift = np.linalg.norm(_centroid(positive_a, coordinates, center) - _centroid(positive_b, coordinates, center))
    negative_shift = np.linalg.norm(_centroid(negative_a, coordinates, center) - _centroid(negative_b, coordinates, center))
    diagonal = math.hypot(*a.shape)
    mass_shift = np.linalg.norm(_centroid(mass_a, coordinates, center) - _centroid(mass_b, coordinates, center))
    return {
        "signed_spearman": _spearman(signed_a, signed_b),
        "top10_unit_signed_value_jaccard": top_fraction_jaccard(left["signed_sum"], right["signed_sum"], valid, absolute=False),
        "absolute_mass_spearman": _spearman(mass_a, mass_b),
        "top10_unit_absolute_mass_jaccard": top_fraction_jaccard(left["absolute_mass_sum"], right["absolute_mass_sum"], valid, absolute=False),
        "signed_ssim": ssim,
        "normalized_positive_negative_mass_displacement": float((positive_shift + negative_shift) / (2 * diagonal)),
        "absolute_mass_center_of_mass_pixel_error": float(mass_shift),
        "valid_unit_count": int(valid.sum()),
        "excluded_partial_unit_count": int(valid.size - valid.sum()),
    }


def sse_decomposition(clean: np.ndarray, transformed: np.ndarray, valid_mask: np.ndarray, block: int) -> dict[str, Any]:
    a = _map(clean, "clean")
    b = _map(transformed, "transformed")
    if a.shape != b.shape:
        raise X2MechAnalysisError("map shapes differ")
    mask = _mask(valid_mask, a.shape)
    aggregated = aggregate_blocks(b - a, mask, block)
    complete = aggregated["valid_blocks"]
    rows, cols = a.shape[0] // block, a.shape[1] // block
    differences = (b - a).reshape(rows, block, cols, block).transpose(0, 2, 1, 3)
    selected = differences[complete].astype(np.float64, copy=False)
    if not len(selected):
        raise X2MechAnalysisError("no complete valid blocks")
    means = selected.mean(axis=(1, 2), dtype=np.float64)
    between = float(np.sum((block * block) * means * means, dtype=np.float64))
    residual = selected - means[:, None, None]
    within = float(np.sum(residual * residual, dtype=np.float64))
    total = float(np.sum(selected * selected, dtype=np.float64))
    if not np.isclose(total, between + within, atol=SSE_ATOL, rtol=SSE_RTOL):
        raise X2MechAnalysisError("SSE identity failed")
    denominator = total if total > np.finfo(np.float64).tiny else 1.0
    clean_blocks = aggregate_blocks(a, mask, block)["signed_sum"][complete]
    transformed_blocks = aggregate_blocks(b, mask, block)["signed_sum"][complete]
    clean_residual = a.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)[complete]
    transformed_residual = b.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)[complete]
    clean_residual = clean_residual - clean_residual.mean(axis=(1, 2), keepdims=True)
    transformed_residual = transformed_residual - transformed_residual.mean(axis=(1, 2), keepdims=True)
    return {
        "sse_total": total, "sse_between": between, "sse_within": within,
        "between_error_fraction": between / denominator,
        "within_error_fraction": within / denominator,
        "valid_block_count": int(complete.sum()),
        "excluded_partial_block_count": int(complete.size - complete.sum()),
        "block_level_signed_rank_stability": _spearman(clean_blocks, transformed_blocks),
        "within_block_demeaned_residual_stability": _spearman(clean_residual, transformed_residual),
    }


def clean_structure_metrics(value: np.ndarray) -> dict[str, float]:
    array = _map(value); flat = array.reshape(-1); mass = np.abs(flat); total_mass = float(mass.sum(dtype=np.float64))
    if total_mass <= np.finfo(np.float64).tiny: raise X2MechAnalysisError("clean map has zero absolute mass")
    probability = mass / total_mass; positive = probability[probability > 0]; count = len(flat)
    top_count = int(math.ceil(0.1 * count)); top_share = float(np.sort(mass)[-top_count:].sum(dtype=np.float64) / total_mass)
    entropy = float(-np.sum(positive * np.log(positive), dtype=np.float64)); normalized_entropy = entropy / math.log(count)
    ordered = np.sort(mass); ranks = np.arange(1, count + 1, dtype=np.float64)
    gini = float((2 * np.sum(ranks * ordered, dtype=np.float64) / (count * total_mass)) - (count + 1) / count)
    variation = float((np.abs(np.diff(array, axis=0)).sum(dtype=np.float64) + np.abs(np.diff(array, axis=1)).sum(dtype=np.float64)) / total_mass)
    residuals = {}
    for block, name in ((4, "within_cell_4_residual_energy_fraction"), (16, "within_patch_16_residual_energy_fraction")):
        rows, cols = array.shape[0] // block, array.shape[1] // block
        blocks = array.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)
        residual = blocks - blocks.mean(axis=(2, 3), keepdims=True); denominator = float(np.sum(array * array, dtype=np.float64))
        residuals[name] = float(np.sum(residual * residual, dtype=np.float64) / denominator) if denominator > np.finfo(np.float64).tiny else 0.0
    return {
        "absolute_top10_pixel_mass_share": top_share,
        "normalized_shannon_entropy": normalized_entropy,
        "effective_support_fraction": float(math.exp(entropy) / count),
        "gini_coefficient": gini, "normalized_total_variation": variation,
        **residuals,
        "signed_cancellation_ratio": float(1.0 - abs(float(flat.sum(dtype=np.float64))) / total_mass),
    }


def transport_control(value: np.ndarray, valid_mask: np.ndarray, block: int,
                      forward: Callable[[np.ndarray], np.ndarray], inverse: Callable[[np.ndarray], np.ndarray]) -> dict[str, float]:
    source = _map(value); transported = _map(inverse(forward(source)), "transported")
    return stability_metrics(source, transported, valid_mask, block)


def association_summary(rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str, *,
                        model_key: str = "model_id", sample_key: str = "sample_id",
                        draws: int = 20000, namespace: str = "STAGE_X_X2_MECH_ASSOCIATION_V1_20260910") -> dict[str, Any]:
    if draws <= 0: raise X2MechAnalysisError("bootstrap draws must be positive")
    records = [dict(row) for row in rows]; models = sorted({str(row[model_key]) for row in records}); samples = sorted({str(row[sample_key]) for row in records})
    if not models or not samples: raise X2MechAnalysisError("association rows are empty")
    def estimate(selected: Sequence[str]) -> tuple[float, dict[str, float]]:
        per_model = {}
        for model in models:
            x = []; y = []
            by_sample = {str(row[sample_key]): row for row in records if str(row[model_key]) == model}
            for sample in selected:
                if sample in by_sample:
                    x.append(float(by_sample[sample][x_key])); y.append(float(by_sample[sample][y_key]))
            if len(x) >= 2: per_model[model] = _spearman(np.asarray(x), np.asarray(y))
        finite = [value for value in per_model.values() if math.isfinite(value)]
        if not finite: raise X2MechAnalysisError("association is not estimable")
        return float(np.mean(finite, dtype=np.float64)), per_model
    point, per_model = estimate(samples); rng = np.random.Generator(np.random.PCG64(seed64(namespace, x_key, y_key)))
    bootstrap_values = []; attempts = 0; maximum_attempts = draws * 100
    while len(bootstrap_values) < draws and attempts < maximum_attempts:
        attempts += 1
        try: bootstrap_values.append(estimate([samples[item] for item in rng.integers(0, len(samples), len(samples))])[0])
        except X2MechAnalysisError: continue
    if len(bootstrap_values) != draws: raise X2MechAnalysisError("insufficient finite synchronized cluster-bootstrap draws")
    bootstrap = np.asarray(bootstrap_values, dtype=np.float64)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    return {"effect": point, "ci_lower": float(lower), "ci_upper": float(upper),
            "per_model": per_model, "bootstrap_draws": draws, "cluster": "sample_id_synchronized_across_models",
            "seed": seed64(namespace, x_key, y_key), "rng": "NumPy_PCG64", "quantile": "linear",
            "degenerate_draw_policy": "skip_and_continue_to_fixed_finite_draw_count_max_100x_attempts",
            "confirmatory_p_value": None}


def forward_inverse_control(value: np.ndarray, transform: str) -> tuple[np.ndarray, np.ndarray]:
    """Exact synthetic controls; interpolating operators remain M1-lazy."""
    array = _map(value)
    valid = np.ones(array.shape, dtype=bool)
    if transform == "horizontal_flip":
        return np.fliplr(np.fliplr(array)).copy(), valid
    if transform.startswith("translate_x_"):
        amount = int(transform.removeprefix("translate_x_"))
        shifted = np.zeros_like(array); restored = np.zeros_like(array); valid[:] = False
        if amount > 0:
            shifted[:, amount:] = array[:, :-amount]; restored[:, :-amount] = shifted[:, amount:]; valid[:, :-amount] = True
        else:
            n = -amount; shifted[:, :-n] = array[:, n:]; restored[:, n:] = shifted[:, :-n]; valid[:, n:] = True
        return restored, valid
    if transform.startswith("translate_y_"):
        amount = int(transform.removeprefix("translate_y_"))
        shifted = np.zeros_like(array); restored = np.zeros_like(array); valid[:] = False
        if amount > 0:
            shifted[amount:, :] = array[:-amount, :]; restored[:-amount, :] = shifted[amount:, :]; valid[:-amount, :] = True
        else:
            n = -amount; shifted[:-n, :] = array[n:, :]; restored[n:, :] = shifted[:-n, :]; valid[n:, :] = True
        return restored, valid
    raise X2MechAnalysisError("synthetic transport supports only flip and integer translation")
