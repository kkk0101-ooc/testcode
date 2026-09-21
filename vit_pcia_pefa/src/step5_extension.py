"""Primitives for the prospective Step-5 mechanism extension."""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from vit_pcia_pefa.src.pilot import endpoint_and_logit, spearman, stable_random_scores
from vit_pcia_pefa.src.vit_adapter import port_fields


def orthonormal_directions(value: Tensor, seed: int) -> tuple[Tensor, Tensor]:
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError("DIRECTION_INPUT_SHAPE")
    generator = torch.Generator(device=value.device); generator.manual_seed(int(seed))
    first = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
    second = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
    first = first / first.norm().clamp_min(1e-30)
    second = second - (second * first).sum() * first
    second = second / second.norm().clamp_min(1e-30)
    if abs(float((first.double() * second.double()).sum())) > 1e-6:
        raise ValueError("DIRECTION_ORTHOGONALITY")
    return first, second


def local_curl_scalars(
    model: nn.Module, base: Tensor, *, block_index: int, class_id: int,
    component_names: Sequence[str], first: Tensor, second: Tensor,
) -> dict[str, float]:
    requested = tuple(component_names)
    root = base.detach().requires_grad_(True)

    def field(value: Tensor) -> Tensor:
        current, names = port_fields(
            model, value, block_index=block_index, class_id=class_id,
            second_order=True, requested_names=requested,
        )
        if names != requested:
            raise ValueError("EXTENSION_PORT_ORDER")
        return current

    _primal, along_first = torch.autograd.functional.jvp(
        field, root, first, create_graph=False, strict=True,
    )
    _primal, along_second = torch.autograd.functional.jvp(
        field, root, second, create_graph=False, strict=True,
    )
    values = (along_first.double() * second.double().unsqueeze(0)).flatten(1).sum(1)
    values -= (along_second.double() * first.double().unsqueeze(0)).flatten(1).sum(1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("NONFINITE_LOCAL_CURL")
    return {name: float(values[index]) for index, name in enumerate(requested)}


def line_integral(
    model: nn.Module, start: Tensor, end: Tensor, *, block_index: int, class_id: int,
    component_names: Sequence[str], nodes: Sequence[float], weights: Sequence[float],
    progress: Callable[[], None] | None = None,
) -> Tensor:
    if start.shape != end.shape or len(nodes) != len(weights) or not nodes:
        raise ValueError("LINE_INTEGRAL_CONTRACT")
    requested = tuple(component_names)
    delta = end - start
    total = torch.zeros(len(requested), dtype=torch.float64, device=start.device)
    for node, weight in zip(nodes, weights):
        root = (start + float(node) * delta).detach().requires_grad_(True)
        fields, names = port_fields(
            model, root, block_index=block_index, class_id=class_id,
            second_order=False, requested_names=requested,
        )
        if names != requested:
            raise ValueError("EXTENSION_PORT_ORDER")
        total.add_((fields.double() * delta.double().unsqueeze(0)).flatten(1).sum(1),
                   alpha=float(weight))
        if progress is not None:
            progress()
    return total


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator == 0.0 else float(np.dot(left, right) / denominator)


def local_loop_analysis(
    model: nn.Module, base: Tensor, *, block_index: int, class_id: int,
    head_names: Sequence[str], seed: int, scales: Sequence[float],
    nodes: Sequence[float], weights: Sequence[float],
    progress: Callable[[], None] | None = None,
) -> dict[str, Any]:
    heads = tuple(head_names)
    components = ("identity", *heads, "__full__")
    first, second = orthonormal_directions(base, seed)
    kappa = local_curl_scalars(
        model, base, block_index=block_index, class_id=class_id,
        component_names=components, first=first, second=second,
    )
    random_scores = stable_random_scores(f"{seed}|loop", len(heads))
    scale_rows = []
    for epsilon in scales:
        eps = float(epsilon)
        corner_first = base + eps * first
        endpoint = corner_first + eps * second
        corner_second = base + eps * second
        edges = ((base, corner_first), (corner_first, endpoint),
                 (endpoint, corner_second), (corner_second, base))
        circulation = torch.zeros(len(components), dtype=torch.float64, device=base.device)
        for start, end in edges:
            circulation.add_(line_integral(
                model, start, end, block_index=block_index, class_id=class_id,
                component_names=components, nodes=nodes, weights=weights, progress=progress,
            ))
        head_observed = np.asarray([float(circulation[components.index(name)]) for name in heads])
        head_predicted = np.asarray([eps * eps * kappa[name] for name in heads])
        threshold_prediction = max(1e-12, float(np.max(np.abs(head_predicted))) * 1e-3)
        threshold_observed = max(1e-12, float(np.max(np.abs(head_observed))) * 1e-3)
        eligible = ((np.abs(head_predicted) >= threshold_prediction)
                    & (np.abs(head_observed) >= threshold_observed))
        sign_agreement = None if not eligible.any() else float(
            np.mean(np.sign(head_predicted[eligible]) == np.sign(head_observed[eligible]))
        )
        predicted_norm = float(np.linalg.norm(head_predicted))
        slope_error = None if predicted_norm == 0.0 else float(
            np.linalg.norm(head_observed - head_predicted) / predicted_norm
        )
        rank = spearman(np.abs(head_predicted).tolist(), np.abs(head_observed).tolist())
        random_rank = spearman(random_scores, np.abs(head_observed).tolist())
        cut_value = float(sum(circulation[components.index(name)] for name in ("identity", *heads)))
        full_value = float(circulation[components.index("__full__")])
        scale_rows.append({
            "epsilon": eps, "head_circulation": head_observed.tolist(),
            "head_local_prediction": head_predicted.tolist(),
            "absolute_rank_spearman": rank, "random_absolute_rank_spearman": random_rank,
            "rank_advantage_over_random": None if rank is None or random_rank is None else rank - random_rank,
            "signed_cosine": _cosine(head_predicted, head_observed),
            "eligible_port_count": int(eligible.sum()), "sign_agreement": sign_agreement,
            "relative_slope_error": slope_error,
            "head_circulation_l2": float(np.linalg.norm(head_observed)),
            "circulation_density_l2": float(np.linalg.norm(head_observed)) / (eps * eps),
            "complete_cut_circulation": cut_value, "full_gradient_circulation": full_value,
            "complete_cut_vs_full_abs_error": abs(cut_value - full_value),
            "identical_endpoint_max_abs": float((endpoint - (base + eps * second + eps * first)).abs().max()),
        })
    norms = np.asarray([row["head_circulation_l2"] for row in scale_rows], dtype=np.float64)
    scale_array = np.asarray([row["epsilon"] for row in scale_rows], dtype=np.float64)
    exponent = None
    if np.all(norms > 0) and len(norms) >= 2:
        exponent = float(np.polyfit(np.log(scale_array), np.log(norms), 1)[0])
    return {
        "direction_seed": int(seed), "component_ids": list(components),
        "head_ids": list(heads), "local_curl_kappa": kappa,
        "direction_norms": [float(first.double().norm()), float(second.double().norm())],
        "direction_inner_product": float((first.double() * second.double()).sum()),
        "scale_rows": scale_rows, "circulation_scaling_exponent": exponent,
    }


def boundary_directed_point(
    model: nn.Module, image: Tensor, *, class_id: int, mean: Tensor, std: Tensor,
    candidates_rgb: Sequence[float],
) -> tuple[Tensor, dict[str, Any]]:
    root = image.detach().requires_grad_(True)
    logits = model(root)[0]
    other = logits.detach().clone(); other[class_id] = -torch.inf
    competitor = int(other.argmax().item())
    fixed_margin = logits[class_id] - logits[competitor]
    gradient = torch.autograd.grad(fixed_margin, root)[0]
    clean_margin = float(fixed_margin.detach())
    direction = -math.copysign(1.0, clean_margin if clean_margin != 0.0 else 1.0) * gradient.sign()
    raw = (image * std + mean).clamp(0, 1)
    rows: list[tuple[Tensor, dict[str, Any]]] = []
    for epsilon in candidates_rgb:
        candidate = ((raw + float(epsilon) * direction).clamp(0, 1) - mean) / std
        with torch.no_grad():
            current = model(candidate)[0]
            alternatives = current.clone(); alternatives[class_id] = -torch.inf
            dynamic_margin = float(current[class_id] - alternatives.max())
            fixed_value = float(current[class_id] - current[competitor])
        crossed_fixed = (clean_margin == 0.0 or fixed_value == 0.0
                         or math.copysign(1.0, fixed_value) != math.copysign(1.0, clean_margin))
        crossed_dynamic = (clean_margin == 0.0 or dynamic_margin == 0.0
                           or math.copysign(1.0, dynamic_margin) != math.copysign(1.0, clean_margin))
        rows.append((candidate.detach(), {
            "epsilon_rgb": float(epsilon), "fixed_margin": fixed_value,
            "dynamic_margin": dynamic_margin, "crossed_fixed_margin": crossed_fixed,
            "crossed_dynamic_margin": crossed_dynamic,
        }))
    crossing = next((row for row in rows if row[1]["crossed_dynamic_margin"]), None)
    selected = crossing if crossing is not None else min(rows, key=lambda row: abs(row[1]["dynamic_margin"]))
    receipt = {
        "class_id": int(class_id), "clean_competitor_class_id": competitor,
        "clean_fixed_margin": clean_margin, "clean_dynamic_margin": float(
            logits.detach()[class_id] - other.max()),
        "candidate_rows": [row for _candidate, row in rows],
        "selected_epsilon_rgb": selected[1]["epsilon_rgb"],
        "selected_dynamic_margin": selected[1]["dynamic_margin"],
        "selected_crossed_fixed_margin": selected[1]["crossed_fixed_margin"],
        "selected_crossed_dynamic_margin": selected[1]["crossed_dynamic_margin"],
        "absolute_margin_reduced_or_crossed": (selected[1]["crossed_dynamic_margin"]
            or abs(selected[1]["dynamic_margin"]) < abs(float(logits.detach()[class_id] - other.max()))),
    }
    return selected[0], receipt


def _spatial_sums(value: Tensor, grid: int) -> Tensor:
    if value.ndim != 5 or value.shape[1:3] != (1, 3):
        raise ValueError("SPATIAL_SUM_SHAPE")
    height, width = value.shape[-2:]
    if height % grid or width % grid:
        raise ValueError("SPATIAL_GRID_DIVISIBILITY")
    reshaped = value.reshape(value.shape[0], 1, 3, grid, height // grid, grid, width // grid)
    return reshaped.sum(dim=(1, 2, 4, 6))


def source_resolved_pefa(
    model: nn.Module, image: Tensor, *, block_index: int, class_id: int,
    head_names: Sequence[str], grid: int, nodes: Sequence[float], weights: Sequence[float],
    seed: int, progress: Callable[[], None] | None = None,
) -> dict[str, Any]:
    requested = tuple(head_names)
    integrated_a = torch.zeros((len(requested), grid, grid), dtype=torch.float64, device=image.device)
    integrated_t = torch.zeros_like(integrated_a)
    contracted_a = torch.zeros(len(requested), dtype=torch.float64, device=image.device)
    contracted_t = torch.zeros_like(contracted_a)
    for node, weight in zip(nodes, weights):
        root = (image * float(node)).detach().requires_grad_(True)
        fields, names = port_fields(
            model, root, block_index=block_index, class_id=class_id,
            second_order=True, requested_names=requested,
        )
        if names != requested:
            raise ValueError("SOURCE_PORT_ORDER")
        a_groups = _spatial_sums(
            fields.detach().double() * image.double().unsqueeze(0), grid,
        )
        gradients = []
        for index in range(len(requested)):
            q_value = (root * fields[index]).sum()
            gradients.append(torch.autograd.grad(
                q_value, root, retain_graph=index + 1 < len(requested), create_graph=False,
            )[0].detach().double())
        q_gradients = torch.stack(gradients)
        t_groups = _spatial_sums(q_gradients * image.double().unsqueeze(0), grid)
        integrated_a.add_(a_groups, alpha=float(weight))
        integrated_t.add_(t_groups, alpha=float(weight))
        contracted_a.add_(
            (fields.detach().double() * image.double().unsqueeze(0)).flatten(1).sum(1),
            alpha=float(weight),
        )
        contracted_t.add_(
            (q_gradients * image.double().unsqueeze(0)).flatten(1).sum(1),
            alpha=float(weight),
        )
        if progress is not None:
            progress()
    integrated_r = integrated_t - integrated_a
    group_a = integrated_a.abs().sum(0).flatten().cpu().tolist()
    group_t = integrated_t.abs().sum(0).flatten().cpu().tolist()
    group_r = integrated_r.abs().sum(0).flatten().cpu().tolist()
    terminal, terminal_names = port_fields(
        model, image.detach().requires_grad_(True), block_index=block_index,
        class_id=class_id, second_order=False, requested_names=requested,
    )
    if terminal_names != requested:
        raise ValueError("SOURCE_TERMINAL_PORT_ORDER")
    terminal_q = (terminal.double() * image.double().unsqueeze(0)).flatten(1).sum(1)
    source_t_sum = integrated_t.sum((1, 2))
    source_a_sum = integrated_a.sum((1, 2))
    return {
        "grid": grid, "group_count": grid * grid, "head_ids": list(requested),
        "methods": {"pefa_R": group_r, "pefa_T": group_t, "conductance_pcia": group_a,
                    "random": stable_random_scores(f"{seed}|source|{grid}", grid * grid)},
        "source_sum_closure": {
            "A_max_abs": float((source_a_sum - contracted_a).abs().max()),
            "T_max_abs": float((source_t_sum - contracted_t).abs().max()),
            "R_max_abs": float(((source_t_sum - source_a_sum) - (contracted_t - contracted_a)).abs().max()),
        },
        "terminal_T_convergence": {
            "max_abs": float((source_t_sum - terminal_q).abs().max()),
            "relative_l2": float((source_t_sum - terminal_q).norm() / terminal_q.norm().clamp_min(1e-30)),
        },
    }


def spatial_baselines_and_effects(
    model: nn.Module, image: Tensor, *, baseline: Tensor, class_id: int, grid: int,
) -> dict[str, list[float]]:
    clean_endpoint, clean_logit = endpoint_and_logit(model, image, class_id, baseline=baseline)
    endpoint_scores, input_norms, endpoint_effects, logit_effects = [], [], [], []
    height, width = image.shape[-2:]
    for row in range(grid):
        for column in range(grid):
            mask = torch.zeros_like(image, dtype=torch.bool)
            mask[:, :, row * height // grid:(row + 1) * height // grid,
                 column * width // grid:(column + 1) * width // grid] = True
            endpoint_scores.append(float(clean_endpoint.masked_select(mask).double().abs().sum()))
            input_norms.append(float(image.masked_select(mask).double().norm()))
            perturbed = torch.where(mask, baseline, image)
            endpoint, logit = endpoint_and_logit(model, perturbed, class_id, baseline=baseline)
            endpoint_effects.append(float((endpoint - clean_endpoint).double().abs().sum()))
            logit_effects.append(abs(logit - clean_logit))
    return {
        "endpoint": endpoint_scores, "input_norm": input_norms,
        "endpoint_change_l1": endpoint_effects, "target_logit_abs_change": logit_effects,
    }


def source_correlations(methods: dict[str, list[float]], effects: dict[str, list[float]]) -> dict[str, dict[str, float | None]]:
    return {name: {effect: spearman(values, outcome) for effect, outcome in effects.items()}
            for name, values in methods.items()}
