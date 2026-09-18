"""Frozen numerical calibration primitives for Endpoint Scanner EP2."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


class EP2Error(ValueError):
    pass


def array_digest(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def gauss_legendre(count: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    if type(count) is not int or count <= 0:
        raise EP2Error("GAUSS_LEGENDRE_COUNT")
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return (
        torch.as_tensor((nodes + 1.0) * 0.5, device=device, dtype=dtype),
        torch.as_tensor(weights * 0.5, device=device, dtype=torch.float64),
    )


def integrate_logit(
    model: nn.Module, image: Tensor, class_id: int, *, steps: int, alpha_chunk: int,
    progress: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if image.shape[0] != 1 or image.dtype is not torch.float32 or alpha_chunk not in {8, 4, 2, 1}:
        raise EP2Error("IG_INPUT_OR_CHUNK")
    if not 0 <= class_id < 1000:
        raise EP2Error("TARGET_CLASS")
    baseline = torch.zeros_like(image)
    difference = image - baseline
    nodes, weights = gauss_legendre(steps, image.device, image.dtype)
    accumulated = torch.zeros_like(image, dtype=torch.float32)
    model.eval()
    requires = [parameter.requires_grad for parameter in model.parameters()]
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            baseline_value = model(baseline)[:, class_id].detach().float()
            clean_value = model(image)[:, class_id].detach().float()
        for begin in range(0, steps, alpha_chunk):
            end = min(begin + alpha_chunk, steps)
            scaled = baseline.unsqueeze(0) + nodes[begin:end].reshape(-1, 1, 1, 1, 1) * difference.unsqueeze(0)
            scaled = scaled.reshape(-1, *image.shape[1:]).detach().requires_grad_(True)
            selected = model(scaled)[:, class_id]
            gradient = torch.autograd.grad(selected.sum(), scaled)[0].reshape(end - begin, *image.shape)
            weight = weights[begin:end].reshape(end - begin, *((1,) * image.ndim))
            accumulated.add_((gradient.float() * weight.float()).sum(0).detach())
            if progress is not None:
                progress()
        attribution = (difference.float() * accumulated).detach()
    finally:
        for parameter, state in zip(model.parameters(), requires):
            parameter.requires_grad_(state)
    delta = float((clean_value - baseline_value).item())
    attribution_sum = float(attribution.double().sum().item())
    residual = abs(attribution_sum - delta)
    finite = bool(torch.isfinite(attribution).all()) and all(math.isfinite(v) for v in (delta, attribution_sum, residual))
    return {
        "map": attribution.cpu(), "map_sha256": array_digest(attribution),
        "delta": delta, "attribution_sum": attribution_sum,
        "completeness_abs_residual": residual, "finite": finite,
    }


def integrate_logits(
    model: nn.Module, image: Tensor, class_ids: Sequence[int], *, steps: int, alpha_chunk: int,
    progress: Callable[[], None] | None = None,
) -> dict[int, dict[str, Any]]:
    """Integrate unique class logits while sharing each path forward."""
    classes = tuple(dict.fromkeys(int(value) for value in class_ids))
    if not classes or any(not 0 <= value < 1000 for value in classes):
        raise EP2Error("TARGET_CLASSES")
    if image.shape[0] != 1 or image.dtype is not torch.float32 or alpha_chunk not in {8, 4, 2, 1}:
        raise EP2Error("IG_INPUT_OR_CHUNK")
    baseline = torch.zeros_like(image); difference = image - baseline
    nodes, weights = gauss_legendre(steps, image.device, image.dtype)
    accumulators = {value: torch.zeros_like(image, dtype=torch.float32) for value in classes}
    requires = [parameter.requires_grad for parameter in model.parameters()]
    try:
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        with torch.no_grad():
            baseline_logits = model(baseline).detach().float()
            clean_logits = model(image).detach().float()
        for begin in range(0, steps, alpha_chunk):
            end = min(begin + alpha_chunk, steps)
            scaled = baseline.unsqueeze(0) + nodes[begin:end].reshape(-1, 1, 1, 1, 1) * difference.unsqueeze(0)
            scaled = scaled.reshape(-1, *image.shape[1:]).detach().requires_grad_(True)
            logits = model(scaled)
            weight = weights[begin:end].reshape(end - begin, *((1,) * image.ndim)).float()
            for index, class_id in enumerate(classes):
                gradient = torch.autograd.grad(
                    logits[:, class_id].sum(), scaled, retain_graph=index + 1 < len(classes),
                )[0].reshape(end - begin, *image.shape)
                accumulators[class_id].add_((gradient.float() * weight).sum(0).detach())
            if progress is not None: progress()
    finally:
        for parameter, state in zip(model.parameters(), requires): parameter.requires_grad_(state)
    results = {}
    for class_id in classes:
        attribution = (difference.float() * accumulators[class_id]).detach()
        delta = float((clean_logits[:, class_id] - baseline_logits[:, class_id]).item())
        attribution_sum = float(attribution.double().sum().item())
        residual = abs(attribution_sum - delta)
        finite = bool(torch.isfinite(attribution).all()) and all(math.isfinite(v) for v in (delta, attribution_sum, residual))
        results[class_id] = {
            "map": attribution.cpu(), "map_sha256": array_digest(attribution),
            "delta": delta, "attribution_sum": attribution_sum,
            "completeness_abs_residual": residual, "finite": finite,
        }
    return results


def compare_pair(lower: Mapping[str, Any], upper: Mapping[str, Any]) -> dict[str, Any]:
    left, right = lower["map"], upper["map"]
    relative = float((left.double() - right.double()).norm() / right.double().norm().clamp_min(1e-12))
    delta = float(lower["delta"]); residual = float(lower["completeness_abs_residual"])
    near_zero = abs(delta) < 1e-6
    completeness_value = residual if near_zero else residual / max(abs(delta), 1e-6)
    completeness_pass = completeness_value <= (1e-5 if near_zero else 0.01)
    finite = bool(lower["finite"] and upper["finite"] and math.isfinite(relative) and math.isfinite(completeness_value))
    return {
        "M": None, "map_M_sha256": lower["map_sha256"],
        "map_2M_sha256": upper["map_sha256"], "map_relative_l2": relative,
        "map_pass": finite and relative <= 0.02, "delta": delta,
        "completeness_abs_residual": residual,
        "completeness_value": completeness_value,
        "completeness_branch": "absolute_near_zero" if near_zero else "relative_nonzero",
        "completeness_pass": finite and completeness_pass, "finite": finite,
        "pair_pass": finite and relative <= 0.02 and completeness_pass,
    }


def select_model_M(
    records: Sequence[Mapping[str, Any]], candidates: Sequence[int] = (32, 64, 128, 256, 512),
) -> dict[str, Any]:
    targets = ("ground_truth_logit", "frozen_clean_top1_logit")
    rows = []
    selected = None
    for candidate in candidates:
        counts = {}
        for target in targets:
            subset = [row for row in records if row["target_kind"] == target and row["M"] == candidate]
            if len(subset) != 64:
                raise EP2Error(f"CALIBRATION_COVERAGE:{target}:{candidate}:{len(subset)}")
            counts[target] = {
                "finite_count": sum(bool(row["finite"]) for row in subset),
                "pass_count": sum(bool(row["pair_pass"]) for row in subset),
            }
        passed = all(counts[target]["finite_count"] == 64 and counts[target]["pass_count"] >= 61 for target in targets)
        rows.append({"M": candidate, "targets": counts, "model_pass": passed})
        if selected is None and passed:
            selected = candidate
    return {
        "selected_M": selected, "candidates": rows,
        "status": "PASS" if selected is not None else "EP2_NUMERICAL_CALIBRATION_FAILED",
    }
