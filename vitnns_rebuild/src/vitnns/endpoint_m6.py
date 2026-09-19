"""Deterministic selectors and selected-case map extraction for Endpoint M6."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


TIE_FIELDS = ("sample_id", "model_id", "block", "token", "channel", "head", "residual_type")


def tie_key(row: dict) -> tuple:
    return tuple(row.get(field, "") for field in TIE_FIELDS)


def select_tail(rows: Sequence[dict], metric: str, tail: str, fraction: float = 0.05) -> dict:
    """Select the smallest-ID member of an inclusive empirical tail."""
    valid = [row for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
    if not valid or tail not in {"highest", "lowest"} or not 0 < fraction <= 1:
        raise ValueError("TAIL_CONTRACT")
    ordered = sorted((float(row[metric]) for row in valid), reverse=tail == "highest")
    nominal = max(1, int(math.ceil(fraction * len(ordered))))
    threshold = ordered[nominal - 1]
    if tail == "highest":
        eligible = [row for row in valid if float(row[metric]) >= threshold]
    else:
        eligible = [row for row in valid if float(row[metric]) <= threshold]
    selected = min(eligible, key=tie_key)
    return {
        "metric": metric, "tail": tail, "fraction": fraction,
        "universe_count": len(valid), "nominal_tail_count": nominal,
        "inclusive_candidate_count": len(eligible), "threshold": threshold,
        "selected": selected,
    }


def select_nearest_median(rows: Sequence[dict], metric: str) -> dict:
    valid = [row for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
    if not valid:
        raise ValueError("MEDIAN_CONTRACT")
    median = float(np.median(np.asarray([float(row[metric]) for row in valid], dtype=np.float64)))
    distance = min(abs(float(row[metric]) - median) for row in valid)
    eligible = [row for row in valid if abs(float(row[metric]) - median) == distance]
    return {
        "metric": metric, "universe_count": len(valid), "median": median,
        "minimum_absolute_distance": distance, "tie_count": len(eligible),
        "selected": min(eligible, key=tie_key),
    }


def average_ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("RANK_INPUT")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("SPEARMAN_INPUT")
    a, b = average_ranks(left), average_ranks(right)
    a -= a.mean(); b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0 else float(np.dot(a, b) / denominator)


def patch_metrics(patch_absolute: Sequence[float], token: int) -> dict:
    mass = np.asarray(patch_absolute, dtype=np.float64)
    if mass.shape != (196,) or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("PATCH_MASS")
    total = float(mass.sum())
    if total == 0:
        return {"own_token_locality": None, "input_dependency_distance": None}
    if token == 0:
        return {"own_token_locality": None, "input_dependency_distance": None}
    p = mass.reshape(14, 14) / total
    row, column = divmod(int(token) - 1, 14)
    rr, cc = np.indices((14, 14))
    neighborhood = np.maximum(abs(rr - row), abs(cc - column)) <= 1
    distance = np.sqrt((rr - row) ** 2 + (cc - column) ** 2) / (13 * math.sqrt(2))
    return {
        "own_token_locality": float(p[neighborhood].sum()),
        "input_dependency_distance": float((p * distance).sum()),
    }


def summarize_map(array: np.ndarray) -> dict:
    value = np.asarray(array, dtype=np.float64)
    if value.shape != (3, 224, 224) or not np.isfinite(value).all():
        raise ValueError("MAP_CONTRACT")
    flat = value.reshape(-1)
    absolute = float(np.abs(flat).sum())
    signed = float(flat.sum())
    return {
        "signed_sum": signed, "absolute_mass": absolute,
        "l2_norm": float(np.linalg.norm(flat)),
        "map_cancellation": None if absolute == 0 else 1.0 - abs(signed) / absolute,
    }


def patch_absolute(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != (3, 224, 224):
        raise ValueError("MAP_CONTRACT")
    return np.abs(value).reshape(3, 14, 16, 14, 16).sum(axis=(0, 2, 4))


def patch_signed(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != (3, 224, 224):
        raise ValueError("MAP_CONTRACT")
    return value.reshape(3, 14, 16, 14, 16).sum(axis=(0, 2, 4))


def extract_selected_maps(model, image, class_id: int, attention: Iterable[dict], residual: Iterable[dict]) -> dict:
    """Recompute only selected M4 paths on one clean graph.

    Imports are local so CPU-only selection and validation never initialize Torch.
    """
    import torch
    from .endpoint_m4 import explicit_graph, _frozen_eval, _finite, _vjp

    attention = list(attention); residual = list(residual)
    clean = image.detach().clone().requires_grad_(True)
    observed = {}
    with _frozen_eval(model), explicit_graph(model, observed), torch.enable_grad():
        logits = model(clean); score = logits[:, int(class_id)]; _finite(logits, score)
        full_gradient = torch.autograd.grad(score.sum(), clean, retain_graph=True)[0][0].detach()
        full_map = clean.detach()[0] * full_gradient
        arrays = {"full_endpoint": full_map.cpu().numpy().astype(np.float32)}
        summaries = {}
        for case in attention:
            block, head, case_id = int(case["block"]), int(case["head"]), case["case_id"]
            edge = observed[block]
            gq, gk, gv, gav = torch.autograd.grad(
                score.sum(), (edge["query"], edge["key"], edge["value"], edge["av"]), retain_graph=True
            )
            components = {}
            for name, output, seed in (("q", edge["query"], gq), ("k", edge["key"], gk),
                                       ("value", edge["value"], gv), ("full", edge["av"], gav)):
                head_seed = torch.zeros_like(seed); head_seed[0, head] = seed[0, head].detach()
                gradient = _vjp(output, clean, head_seed)
                components[name] = clean.detach()[0] * gradient
            components["routing"] = components["q"] + components["k"]
            for name, value in components.items():
                key = f"{case_id}__{name}"; arrays[key] = value.cpu().numpy().astype(np.float32)
                summaries[key] = summarize_map(arrays[key])
            arrays[f"{case_id}__attention_cls"] = edge["attention"].detach()[0, head, 0, 1:].reshape(14, 14).cpu().numpy().astype(np.float32)
        for case in residual:
            block, kind, case_id = int(case["block"]), case["residual_type"], case["case_id"]
            edge = observed[block]
            target = edge["r1"] if kind == "attention" else edge["r2"]
            identity_edge = edge["u"] if kind == "attention" else edge["r1"]
            branch_edge = edge["attention_branch"] if kind == "attention" else edge["mlp_branch"]
            cotangent = torch.autograd.grad(score.sum(), target, retain_graph=True)[0]
            identity_gradient = _vjp(identity_edge, clean, cotangent)
            branch_gradient = _vjp(branch_edge, clean, cotangent)
            components = {
                "identity": clean.detach()[0] * identity_gradient,
                "branch": clean.detach()[0] * branch_gradient,
            }
            components["output"] = components["identity"] + components["branch"]
            for name, value in components.items():
                key = f"{case_id}__{name}"; arrays[key] = value.cpu().numpy().astype(np.float32)
                summaries[key] = summarize_map(arrays[key])
    return {"arrays": arrays, "summaries": summaries, "target_value": float(score.detach())}
