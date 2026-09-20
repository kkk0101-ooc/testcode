"""Frozen estimands and selection rules for Endpoint revision v3 Step 2."""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


def _score(namespace: str, *parts: object) -> bytes:
    return hashlib.sha256("|".join((namespace, *(str(x) for x in parts))).encode()).digest()


def independently_sampled_tokens(namespace: str, grid: int = 14) -> tuple[int, ...]:
    """Return CLS plus one outcome-blind hash draw in each fixed 3x5 stratum."""
    selected: list[int] = []
    for stratum_row in range(3):
        lo_r, hi_r = stratum_row * grid / 3, (stratum_row + 1) * grid / 3
        for stratum_column in range(5):
            lo_c, hi_c = stratum_column * grid / 5, (stratum_column + 1) * grid / 5
            candidates = [
                (r, c) for r in range(grid) for c in range(grid)
                if lo_r <= r < hi_r and lo_c <= c < hi_c
            ]
            row, column = min(
                candidates,
                key=lambda rc: (_score(namespace, "token", stratum_row, stratum_column, *rc), rc),
            )
            selected.append(1 + row * grid + column)
    if len(selected) != 15 or len(set(selected)) != 15:
        raise ValueError("TOKEN_SAMPLE_NOT_UNIQUE")
    return (0, *selected)


def independently_sampled_channels(
    namespace: str, model_id: str, block: int, width: int, count: int = 32,
) -> tuple[int, ...]:
    if not namespace or not model_id or block < 0 or not 0 < count <= width:
        raise ValueError("INVALID_CHANNEL_SAMPLE")
    selected = sorted(
        range(width), key=lambda channel: (_score(namespace, "channel", model_id, block, channel), channel),
    )[:count]
    return tuple(sorted(selected))


def interpolate(clean: Tensor, reference: Tensor, alpha: float) -> Tensor:
    if clean.shape != reference.shape or not 0.0 < float(alpha) <= 1.0:
        raise ValueError("INTERPOLATION_CONTRACT")
    return clean.detach() + float(alpha) * (reference.detach() - clean.detach())


def amplitude_errors(effect: float, p_ref: float, alpha: float) -> dict[str, float | None]:
    prediction = float(alpha) * float(p_ref)
    absolute_error = abs(float(effect) - prediction)
    relative_error = None if float(effect) == 0.0 else absolute_error / abs(float(effect))
    return {
        "taylor_prediction": prediction,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }


def _patch_absolute_mass(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (1, 3, 224, 224):
        array = array[0]
    if array.shape != (3, 224, 224) or not np.isfinite(array).all():
        raise ValueError("QKV_MAP_SHAPE_OR_FINITE")
    return np.abs(array).reshape(3, 14, 16, 14, 16).sum(axis=(0, 2, 4)).reshape(196)


def patch_pairwise_distance_matrix(grid: int = 14) -> np.ndarray:
    rows, columns = np.indices((grid, grid))
    coordinates = np.stack((rows.ravel(), columns.ravel()), axis=1).astype(np.float64)
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.square(difference).sum(axis=2)) / ((grid - 1) * math.sqrt(2.0))


PAIRWISE_DISTANCE_14 = patch_pairwise_distance_matrix()


def qkv_map_metrics(value: Tensor | np.ndarray) -> dict[str, float | None]:
    array = value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
    if array.shape == (1, 3, 224, 224):
        signed_array = np.asarray(array[0], dtype=np.float64)
    elif array.shape == (3, 224, 224):
        signed_array = np.asarray(array, dtype=np.float64)
    else:
        raise ValueError("QKV_MAP_SHAPE_OR_FINITE")
    patch_mass = _patch_absolute_mass(signed_array)
    absolute_mass = float(patch_mass.sum())
    signed_sum = float(signed_array.sum(dtype=np.float64))
    if absolute_mass == 0.0:
        return {
            "absolute_mass": 0.0, "signed_sum": signed_sum,
            "normalized_spatial_entropy": None, "normalized_pairwise_distance": None,
            "cancellation": None,
        }
    probability = patch_mass / absolute_mass
    positive = probability > 0
    entropy = float(-(probability[positive] * np.log(probability[positive])).sum() / math.log(196))
    distance = float(probability @ PAIRWISE_DISTANCE_14 @ probability)
    return {
        "absolute_mass": absolute_mass, "signed_sum": signed_sum,
        "normalized_spatial_entropy": entropy,
        "normalized_pairwise_distance": distance,
        "cancellation": float(1.0 - abs(signed_sum) / absolute_mass),
    }


def qkv_metrics(maps: Mapping[str, Tensor | np.ndarray]) -> dict[str, dict[str, float | None]]:
    if set(maps) != {"Q", "K", "V"}:
        raise ValueError("QKV_KEYS")
    result = {path: qkv_map_metrics(maps[path]) for path in ("Q", "K", "V")}
    denominator = sum(float(result[path]["absolute_mass"]) for path in result)
    for path in result:
        result[path]["relative_scale"] = (
            None if denominator == 0.0 else float(result[path]["absolute_mass"]) / denominator
        )
    return result


def compact_sampling_rows(rows: Iterable[dict]) -> dict[str, np.ndarray]:
    from .endpoint_m3_atlas import derived_metrics

    enriched = [derived_metrics(row) for row in rows]
    return {
        "block": np.asarray([row["block"] for row in enriched], dtype=np.int16),
        "token": np.asarray([row["token"] for row in enriched], dtype=np.int16),
        "channel": np.asarray([row["channel"] for row in enriched], dtype=np.int16),
        "normalized_spatial_entropy": np.asarray([
            np.nan if row["normalized_spatial_entropy"] is None else row["normalized_spatial_entropy"]
            for row in enriched
        ], dtype=np.float32),
        "own_token_locality_above_uniform": np.asarray([
            np.nan if row["own_token_locality_above_uniform"] is None else row["own_token_locality_above_uniform"]
            for row in enriched
        ], dtype=np.float32),
        "input_dependency_distance": np.asarray([
            np.nan if row["input_dependency_distance"] is None else row["input_dependency_distance"]
            for row in enriched
        ], dtype=np.float32),
    }


def average_ranks(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="stable")
    result = np.empty(len(data), dtype=np.float64)
    start = 0
    while start < len(data):
        stop = start + 1
        while stop < len(data) and data[order[stop]] == data[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return result


def spearman(left: Sequence[float], right: Sequence[float], minimum: int = 4) -> float | None:
    if len(left) != len(right) or len(left) < minimum:
        return None
    x, y = average_ranks(left), average_ranks(right)
    x -= x.mean(); y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator == 0.0 else float(x @ y / denominator)


def bootstrap_median(values: Sequence[float], key: str, replicates: int = 10000) -> list[float] | None:
    data = np.sort(np.asarray(values, dtype=np.float64))
    if not len(data):
        return None
    seed = int.from_bytes(_score("ENDPOINT_REVISION_V3_STEP2_BOOTSTRAP_20260921", key)[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(data), size=(replicates, len(data)))
    medians = np.median(data[indices], axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def scan_multi_amplitude(
    model, image: Tensor, references: Mapping[str, Tensor], class_id: int,
    alphas: Sequence[float], chunk_size: int = 8, progress=None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Scan all attention heads at fixed amplitudes on the same clean graph."""
    from .endpoint_e4_intervention import (
        _detach_edges, _run_in_chunks, capture_registered_edges, component_rows,
        component_tensor, dot64, edge_gradients,
    )

    num_heads = int(model.blocks[0].attn.num_heads)
    rows = [row for row in component_rows(num_heads) if row["operation"] == "head"]
    clean_capture = capture_registered_edges(model, image, class_id=class_id, enable_grad=True)
    clean_score = float(clean_capture.score.detach())
    gradients = edge_gradients(clean_capture)
    clean_edges = _detach_edges(clean_capture.edges)
    reference_edges = {
        name: _detach_edges(capture_registered_edges(model, value, class_id=class_id, enable_grad=False).edges)
        for name, value in references.items()
    }
    del clean_capture
    output_rows = []
    clean_control_max_abs = 0.0
    for reference_id in sorted(references):
        p_ref = {}
        for row in rows:
            block = int(row["block"]); component_id = row["component_id"]
            clean_value = component_tensor(clean_edges[block], row)
            reference_value = component_tensor(reference_edges[reference_id][block], row)
            p_ref[component_id] = dot64(
                component_tensor(gradients[block], row), clean_value.detach() - reference_value.detach(),
            )
        for alpha in alphas:
            replacements = {}
            for row in rows:
                block = int(row["block"]); component_id = row["component_id"]
                replacements[component_id] = interpolate(
                    component_tensor(clean_edges[block], row),
                    component_tensor(reference_edges[reference_id][block], row),
                    float(alpha),
                )
            observed_clean, scores, _ = _run_in_chunks(
                model, image, class_id=class_id, rows=rows, replacements=replacements,
                chunk_size=chunk_size, progress=progress,
            )
            clean_control_max_abs = max(clean_control_max_abs, abs(observed_clean - clean_score))
            for row in rows:
                component_id = row["component_id"]
                finite_effect = clean_score - scores[component_id]
                errors = amplitude_errors(finite_effect, p_ref[component_id], float(alpha))
                output_rows.append({
                    "reference_id": reference_id, "alpha": float(alpha),
                    "block": int(row["block"]), "head": int(row["head"]),
                    "finite_effect": finite_effect, "p_ref": p_ref[component_id], **errors,
                })
    packed = {
        "reference_id": np.asarray([row["reference_id"] for row in output_rows], dtype="U16"),
        "alpha": np.asarray([row["alpha"] for row in output_rows], dtype=np.float32),
        "block": np.asarray([row["block"] for row in output_rows], dtype=np.int8),
        "head": np.asarray([row["head"] for row in output_rows], dtype=np.int8),
        "finite_effect": np.asarray([row["finite_effect"] for row in output_rows], dtype=np.float32),
        "p_ref": np.asarray([row["p_ref"] for row in output_rows], dtype=np.float32),
        "taylor_prediction": np.asarray([row["taylor_prediction"] for row in output_rows], dtype=np.float32),
        "absolute_error": np.asarray([row["absolute_error"] for row in output_rows], dtype=np.float32),
        "relative_error": np.asarray([
            np.nan if row["relative_error"] is None else row["relative_error"] for row in output_rows
        ], dtype=np.float32),
    }
    return packed, {
        "clean_score": clean_score, "clean_control_max_abs": clean_control_max_abs,
        "component_count": len(rows), "record_count": len(output_rows),
    }


def scan_qkv_population(model, image: Tensor, class_id: int, blocks: Sequence[int], progress=None):
    """Compute full Q/K/V class-target maps and retain only declared statistics."""
    from .endpoint_vit import EndpointTarget, decompose_endpoint_vit

    target = EndpointTarget("ground_truth_logit", "", kind="ground_truth_logit", class_id=int(class_id))
    rows = []
    for block in blocks:
        decomposition = decompose_endpoint_vit(
            model, image, block=int(block), target=target, atol=1e-5, rtol=1e-4,
        )
        metrics = qkv_metrics({
            "Q": decomposition.components["query"],
            "K": decomposition.components["key"],
            "V": decomposition.components["value"],
        })
        for path in ("Q", "K", "V"):
            rows.append({"block": int(block), "path": path, **metrics[path]})
        del decomposition
        if progress is not None:
            progress()
    return {
        "block": np.asarray([row["block"] for row in rows], dtype=np.int8),
        "path": np.asarray([row["path"] for row in rows], dtype="U1"),
        **{name: np.asarray([
            np.nan if row[name] is None else row[name] for row in rows
        ], dtype=np.float32) for name in (
            "absolute_mass", "signed_sum", "normalized_spatial_entropy",
            "normalized_pairwise_distance", "cancellation", "relative_scale",
        )},
    }


def scan_independent_sampling_groups(
    model, image: Tensor, groups: Sequence[Mapping], chunk_size: int = 16, progress=None,
) -> dict[str, np.ndarray]:
    """Evaluate five predeclared samples on one shared model/image graph."""
    from .endpoint_m3_atlas import endpoint_chunk, summarize_maps

    captured = {}; handles = []
    for block in range(12):
        handles.append(model.blocks[block].register_forward_hook(
            lambda _module, _args, output, block=block: captured.__setitem__(block, output)
        ))
    clean = image.detach().clone().requires_grad_(True)
    records: dict[str, list[np.ndarray]] = {}
    try:
        with torch.enable_grad():
            model(clean)
        for group_index, group in enumerate(groups):
            tokens = tuple(int(token) for token in group["tokens"])
            for block in range(12):
                channels = tuple(int(channel) for channel in group["channels"][str(block)])
                units = tuple((token, channel) for token in tokens for channel in channels)
                maps = []
                for start in range(0, len(units), chunk_size):
                    maps.append(endpoint_chunk(captured[block], clean, units[start:start + chunk_size]).detach().cpu())
                    if progress is not None:
                        progress()
                compact = compact_sampling_rows(summarize_maps(torch.cat(maps), tokens, channels, block))
                compact["group_index"] = np.full(len(compact["block"]), group_index, dtype=np.int8)
                for key, value in compact.items():
                    records.setdefault(key, []).append(value)
        return {key: np.concatenate(values) for key, values in records.items()}
    finally:
        for handle in handles:
            handle.remove()
