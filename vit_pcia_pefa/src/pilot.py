"""Frozen Step-5 pilot primitives.

The module operates on registered use-port fields and small, explicit input
partitions.  It never materializes a full input Hessian.  The only dense
second-order object is a frozen 4 by 4 Integrated-Hessians comparator over
the four registered image quadrants.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from vit_pcia_pefa.src.vit_adapter import parity, port_fields, port_layout


def quadrant_masks(value: Tensor) -> tuple[Tensor, ...]:
    """Return the frozen 2 by 2 spatial partition as boolean masks."""
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
        raise ValueError("QUADRANT_INPUT_SHAPE")
    height, width = value.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError("QUADRANT_ODD_SPATIAL_SIZE")
    masks = []
    for row in range(2):
        for column in range(2):
            mask = torch.zeros_like(value, dtype=torch.bool)
            mask[:, :, row * height // 2:(row + 1) * height // 2,
                 column * width // 2:(column + 1) * width // 2] = True
            masks.append(mask)
    coverage = sum((mask.to(torch.int8) for mask in masks), torch.zeros_like(value, dtype=torch.int8))
    if not bool((coverage == 1).all()):
        raise ValueError("QUADRANT_NOT_PARTITION")
    return tuple(masks)


def path_integral_ports(
    model: nn.Module, *, start: Tensor, end: Tensor, block_index: int, class_id: int,
    nodes: Sequence[float], weights: Sequence[float],
    progress: Callable[[], None] | None = None,
    requested_names: Sequence[str] | None = None,
) -> tuple[Tensor, tuple[str, ...]]:
    """Integrate port-resolved PCIA over one straight path segment."""
    if start.shape != end.shape or len(nodes) != len(weights) or len(nodes) == 0:
        raise ValueError("PATH_CONTRACT")
    difference = end - start
    accumulated: Tensor | None = None
    names: tuple[str, ...] | None = None
    for node, weight in zip(nodes, weights):
        root = (start + float(node) * difference).detach().requires_grad_(True)
        fields, current_names = port_fields(
            model, root, block_index=block_index, class_id=class_id, second_order=False,
            requested_names=requested_names,
        )
        if names is None:
            names = current_names
        elif names != current_names:
            raise ValueError("PORT_ORDER_DRIFT")
        term = difference.unsqueeze(0).double() * fields.detach().double() * float(weight)
        accumulated = term if accumulated is None else accumulated.add(term)
        if progress is not None:
            progress()
    assert accumulated is not None and names is not None
    return accumulated, names


def formation_fields(
    model: nn.Module, *, image: Tensor, baseline: Tensor, waypoints: Sequence[Tensor],
    block_index: int, class_id: int, nodes: Sequence[float], weights: Sequence[float],
    progress: Callable[[], None] | None = None,
    requested_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return A, terminal T and R for a frozen piecewise-linear path."""
    points = (baseline, *tuple(waypoints), image)
    if any(point.shape != image.shape for point in points):
        raise ValueError("WAYPOINT_SHAPE")
    accumulated: Tensor | None = None
    names: tuple[str, ...] | None = None
    for start, end in zip(points[:-1], points[1:]):
        segment, current_names = path_integral_ports(
            model, start=start, end=end, block_index=block_index, class_id=class_id,
            nodes=nodes, weights=weights, progress=progress,
            requested_names=requested_names,
        )
        if names is None:
            names = current_names
        elif names != current_names:
            raise ValueError("SEGMENT_PORT_ORDER_DRIFT")
        accumulated = segment if accumulated is None else accumulated.add(segment)
    assert accumulated is not None and names is not None
    fields, terminal_names = port_fields(
        model, image.detach().requires_grad_(True), block_index=block_index,
        class_id=class_id, second_order=False,
        requested_names=requested_names,
    )
    if terminal_names != names:
        raise ValueError("TERMINAL_PORT_ORDER_DRIFT")
    terminal = (image - baseline).unsqueeze(0).double() * fields.detach().double()
    return {
        "D": fields.detach().double(), "A": accumulated, "T": terminal,
        "R": terminal - accumulated, "names": names,
    }


def cut_indexes(model: nn.Module, block_index: int, names: Sequence[str], cut_id: str) -> list[int]:
    layout = port_layout(model.blocks[block_index].attn.num_heads)
    if cut_id not in layout.cuts:
        raise ValueError("UNKNOWN_CUT")
    indexes = {name: index for index, name in enumerate(names)}
    return [indexes[name] for name in layout.cuts[cut_id]]


def tensor_mass(value: Tensor) -> dict[str, float | bool]:
    flat = value.detach().double().flatten()
    finite = bool(torch.isfinite(flat).all())
    signed = float(flat.sum())
    absolute = float(flat.abs().sum())
    return {
        "finite": finite, "signed_mass": signed, "absolute_mass": absolute,
        "cancellation_rate": 0.0 if absolute == 0.0 else 1.0 - abs(signed) / absolute,
    }


def port_categories(names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    def selected(predicate: Callable[[str], bool]) -> tuple[int, ...]:
        return tuple(index for index, name in enumerate(names) if name != "__full__" and predicate(name))
    categories = {
        "residual": selected(lambda name: name in {"identity", "carry"}),
        "attention_branch": selected(lambda name: name == "attention"),
        "query": selected(lambda name: name.startswith("query_head_")),
        "key": selected(lambda name: name.startswith("key_head_")),
        "value": selected(lambda name: name.startswith("value_head_")),
        "head": selected(lambda name: name.startswith("head_")),
        "projection": selected(lambda name: name == "projection"),
        "mlp": selected(lambda name: name in {"mlp", "mlp_fc1", "mlp_gelu", "mlp_fc2"}),
    }
    if any(not indexes for indexes in categories.values()):
        raise ValueError("EMPTY_PORT_CATEGORY")
    return categories


def representation_summary(value: Tensor, names: Sequence[str]) -> dict[str, Any]:
    """Signed/absolute mass, cancellation and port concentration."""
    if value.shape[0] != len(names):
        raise ValueError("REPRESENTATION_PORT_AXIS")
    port_values = value[:-1] if names[-1] == "__full__" else value
    port_names = names[:-1] if names[-1] == "__full__" else names
    per_port = {name: tensor_mass(port_values[index]) for index, name in enumerate(port_names)}
    masses = np.asarray([row["absolute_mass"] for row in per_port.values()], dtype=np.float64)
    total = float(masses.sum())
    concentration = 0.0 if total == 0.0 else float(np.square(masses / total).sum())
    categories = {}
    for category, indexes in port_categories(names).items():
        selected = port_values[list(indexes)]
        categories[category] = tensor_mass(selected)
    return {
        "overall": tensor_mass(port_values), "routing_concentration_hhi": concentration,
        "categories": categories, "per_port": per_port,
    }


def cut_representation_summary(
    value: Tensor, names: Sequence[str], component_names: Sequence[str],
) -> dict[str, Any]:
    indexes = {name: index for index, name in enumerate(names)}
    if not component_names or any(name not in indexes or name == "__full__" for name in component_names):
        raise ValueError("CUT_REPRESENTATION_COMPONENT")
    selected = value[[indexes[name] for name in component_names]]
    per_port = {name: tensor_mass(selected[index]) for index, name in enumerate(component_names)}
    masses = np.asarray([per_port[name]["absolute_mass"] for name in component_names], dtype=np.float64)
    total = float(masses.sum())
    return {
        "component_ids": list(component_names), "overall": tensor_mass(selected),
        "routing_concentration_hhi": 0.0 if total == 0.0 else float(np.square(masses / total).sum()),
        "per_port": per_port,
    }


def quadrant_scores(value: Tensor, masks: Sequence[Tensor], *, port_absolute: bool) -> list[float]:
    """Score four groups from [ports,1,C,H,W] or [1,C,H,W]."""
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if value.ndim != 5:
        raise ValueError("QUADRANT_SCORE_SHAPE")
    scores = []
    for mask in masks:
        expanded = mask.unsqueeze(0).expand(value.shape[0], *mask.shape)
        selected = value.masked_select(expanded)
        if port_absolute:
            scores.append(float(selected.double().abs().sum()))
        else:
            collapsed = value.sum(0)
            scores.append(float(collapsed.masked_select(mask).double().abs().sum()))
    return scores


def stable_random_scores(namespace: str, count: int) -> list[float]:
    values = []
    for index in range(count):
        raw = hashlib.sha256(f"{namespace}|{index}".encode()).digest()
        values.append(int.from_bytes(raw[:8], "big") / float(2**64))
    return values


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("RANK_INPUT")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    begin = 0
    while begin < len(array):
        end = begin + 1
        while end < len(array) and array[order[end]] == array[order[begin]]:
            end += 1
        ranks[order[begin:end]] = (begin + end - 1) / 2.0
        begin = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("SPEARMAN_LENGTH")
    x, y = _average_ranks(left), _average_ranks(right)
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator == 0.0 else float(np.dot(x, y) / denominator)


@contextmanager
def ablate_projection_head(model: nn.Module, block_index: int, head_index: int) -> Iterator[None]:
    attention = model.blocks[block_index].attn
    heads = int(attention.num_heads)
    if not 0 <= head_index < heads:
        raise ValueError("HEAD_INDEX")
    def hook(_module: nn.Module, args: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        value = args[0].clone()
        width = value.shape[-1] // heads
        value[..., head_index * width:(head_index + 1) * width] = 0
        return (value, *args[1:])
    handle = attention.proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def randomized_classifier(model: nn.Module, seed: int) -> Iterator[None]:
    """Temporarily replace the classifier by deterministic matched-scale noise."""
    head = model.get_classifier()
    if not isinstance(head, nn.Linear):
        raise ValueError("UNREGISTERED_CLASSIFIER")
    weight = head.weight.detach().clone()
    bias = None if head.bias is None else head.bias.detach().clone()
    generator = torch.Generator(device="cpu"); generator.manual_seed(int(seed))
    replacement_weight = torch.randn(weight.shape, generator=generator, dtype=torch.float64)
    replacement_weight.mul_(float(weight.detach().double().std())).to(dtype=weight.dtype)
    replacement_bias = None
    if bias is not None:
        replacement_bias = torch.randn(bias.shape, generator=generator, dtype=torch.float64)
        replacement_bias.mul_(float(bias.detach().double().std())).to(dtype=bias.dtype)
    with torch.no_grad():
        head.weight.copy_(replacement_weight.to(device=weight.device, dtype=weight.dtype))
        if bias is not None and replacement_bias is not None:
            head.bias.copy_(replacement_bias.to(device=bias.device, dtype=bias.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            head.weight.copy_(weight)
            if bias is not None:
                head.bias.copy_(bias)


def endpoint_and_logit(
    model: nn.Module, image: Tensor, class_id: int, *, baseline: Tensor,
    block_index: int | None = None, head_index: int | None = None,
) -> tuple[Tensor, float]:
    if (block_index is None) != (head_index is None):
        raise ValueError("ABLATION_PAIR")
    root = image.detach().requires_grad_(True)
    if block_index is None:
        logits = model(root)
    else:
        with ablate_projection_head(model, int(block_index), int(head_index)):
            logits = model(root)
    score = logits[:, class_id].sum()
    gradient = torch.autograd.grad(score, root)[0]
    endpoint = (root.detach() - baseline.detach()) * gradient.detach()
    if not bool(torch.isfinite(endpoint).all()) or not math.isfinite(float(score)):
        raise ValueError("NONFINITE_INTERVENTION")
    return endpoint, float(score.detach())


def projection_head_norms(model: nn.Module, image: Tensor, block_index: int) -> list[float]:
    observed: list[Tensor] = []
    attention = model.blocks[block_index].attn
    def hook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        observed.append(args[0].detach())
    handle = attention.proj.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(image)
    finally:
        handle.remove()
    if len(observed) != 1:
        raise ValueError("PROJECTION_CAPTURE_COUNT")
    value = observed[0]
    width = value.shape[-1] // attention.num_heads
    return [float(value[..., index * width:(index + 1) * width].double().norm())
            for index in range(attention.num_heads)]


def omega_directional_scores(
    model: nn.Module, image: Tensor, *, block_index: int, class_id: int,
    component_names: Sequence[str], seed: int,
) -> dict[str, float]:
    """Return ||(J_e^T-J_e)v||_2 without forming a Hessian."""
    generator = torch.Generator(device=image.device); generator.manual_seed(int(seed))
    direction = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
    direction = direction / direction.norm().clamp_min(1e-30)
    root = image.detach().requires_grad_(True)
    _fields, jvp = torch.autograd.functional.jvp(
        lambda value: port_fields(model, value, block_index=block_index, class_id=class_id)[0],
        root, direction, create_graph=False, strict=True,
    )
    current, names = port_fields(
        model, image.detach().requires_grad_(True), block_index=block_index,
        class_id=class_id, second_order=True,
    )
    indexes = {name: index for index, name in enumerate(names)}
    scores = {}
    second_root = image.detach().requires_grad_(True)
    current, current_names = port_fields(
        model, second_root, block_index=block_index, class_id=class_id, second_order=True,
    )
    if current_names != names:
        raise ValueError("OMEGA_PORT_ORDER_DRIFT")
    for position, name in enumerate(component_names):
        index = indexes[name]
        vjp = torch.autograd.grad(
            current[index], second_root, grad_outputs=direction,
            retain_graph=position + 1 < len(component_names), create_graph=False,
        )[0]
        omega_v = vjp - jvp[index]
        scores[name] = float(omega_v.detach().double().norm())
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("NONFINITE_OMEGA")
    return scores


def omega_directional_analysis(
    model: nn.Module, image: Tensor, *, block_index: int, class_id: int,
    component_names: Sequence[str], cut_component_names: Sequence[str], seed: int,
) -> dict[str, Any]:
    """Summarize one frozen directional probe of every registered port curl."""
    generator = torch.Generator(device=image.device); generator.manual_seed(int(seed))
    direction = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
    direction = direction / direction.norm().clamp_min(1e-30)
    root = image.detach().requires_grad_(True)
    _fields, jvp = torch.autograd.functional.jvp(
        lambda value: port_fields(model, value, block_index=block_index, class_id=class_id)[0],
        root, direction, create_graph=False, strict=True,
    )
    names = port_fields(
        model, image.detach().requires_grad_(True), block_index=block_index,
        class_id=class_id, second_order=False,
    )[1]
    indexes = {name: index for index, name in enumerate(names)}
    omega_fields = []
    for name in component_names:
        index = indexes[name]
        def component(root_value: Tensor, index: int = index) -> Tensor:
            return port_fields(
                model, root_value, block_index=block_index, class_id=class_id, second_order=True,
                requested_names=(name,),
            )[0][0]
        _output, vjp = torch.autograd.functional.vjp(
            component, root, v=direction, create_graph=False, strict=True,
        )
        omega_fields.append((vjp - jvp[index]).detach().double())
    stacked = torch.stack(omega_fields)
    indexes = {name: index for index, name in enumerate(component_names)}
    if not cut_component_names or any(name not in indexes for name in cut_component_names):
        raise ValueError("OMEGA_CUT_COMPONENT")
    cut_sum = stacked[[indexes[name] for name in cut_component_names]].sum(0)
    cut_zero_sum = parity(cut_sum, torch.zeros_like(cut_sum))
    with_full = torch.cat((stacked, torch.zeros_like(stacked[:1])), dim=0)
    analysis_names = (*tuple(component_names), "__full__")
    return {
        "direction_seed": int(seed),
        "scores_l2": {name: float(stacked[index].norm()) for index, name in enumerate(component_names)},
        "representation": representation_summary(with_full, analysis_names),
        "complete_cut_component_ids": list(cut_component_names),
        "complete_cut_zero_sum": cut_zero_sum,
    }


def omega_complete_cut_audit(
    model: nn.Module, image: Tensor, *, block_index: int, class_id: int, seed: int,
    declared_cut_component_names: Sequence[str],
    evaluation_component_names: Sequence[str] = ("identity", "attention"),
) -> dict[str, Any]:
    """Audit complete-cut zero curl without materializing every port graph.

    ``identity + attention`` is the registered coarse representation of the
    same complete cut as ``identity + sum(heads)``.  Evaluating its summed
    field directly avoids catastrophic float32 cancellation and the memory
    cost of retaining every per-head second-order graph.
    """
    if image.dtype != torch.float64:
        raise ValueError("OMEGA_AUDIT_REQUIRES_FLOAT64")
    declared = tuple(declared_cut_component_names)
    evaluated = tuple(evaluation_component_names)
    expected = tuple(port_layout(model.blocks[block_index].attn.num_heads).cuts["HEADS_WITH_BYPASS_FULL"])
    if declared != expected or evaluated != ("identity", "attention"):
        raise ValueError("OMEGA_AUDIT_CUT_IDENTITY")
    generator = torch.Generator(device=image.device); generator.manual_seed(int(seed))
    direction = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
    direction = direction / direction.norm().clamp_min(1e-30)

    def curl(names: tuple[str, ...]) -> Tensor:
        root = image.detach().requires_grad_(True)

        def field(value: Tensor) -> Tensor:
            values, current_names = port_fields(
                model, value, block_index=block_index, class_id=class_id,
                second_order=True, requested_names=names,
            )
            if current_names != names:
                raise ValueError("OMEGA_AUDIT_PORT_ORDER_DRIFT")
            return values.sum(0)

        _output, jvp = torch.autograd.functional.jvp(
            field, root, direction, create_graph=False, strict=True,
        )
        _output, vjp = torch.autograd.functional.vjp(
            field, root, v=direction, create_graph=False, strict=True,
        )
        return (vjp - jvp).detach().double()

    cut_curl = curl(evaluated)
    full_curl = curl(("__full__",))
    cut_zero = parity(cut_curl, torch.zeros_like(cut_curl))
    full_zero = parity(full_curl, torch.zeros_like(full_curl))
    cut_vs_full = parity(cut_curl, full_curl)
    return {
        "direction_seed": int(seed),
        "dtype": "float64",
        "declared_cut_component_ids": list(declared),
        "evaluation_component_ids": list(evaluated),
        "algebraic_equivalence": "identity + sum(heads) == identity + attention",
        "complete_cut_zero_sum": cut_zero,
        "full_gradient_zero_sum": full_zero,
        "complete_cut_vs_full_gradient": cut_vs_full,
        "pass": cut_zero["pass"] and full_zero["pass"] and cut_vs_full["pass"],
    }


def integrated_hessians_groups(
    model: nn.Module, *, image: Tensor, baseline: Tensor, class_id: int,
    nodes: Sequence[float], weights: Sequence[float], masks: Sequence[Tensor],
    progress: Callable[[], None] | None = None,
) -> Tensor:
    """Recursive IG over four group-scaling coordinates.

    This constructs only the registered 4 by 4 comparator, never an input
    Hessian.  Rows are the base-IG groups and columns the meta-attribution
    source groups.
    """
    if len(masks) != 4 or len(nodes) != len(weights) or len(nodes) == 0:
        raise ValueError("IH_GROUP_CONTRACT")
    difference = image - baseline
    directions = [difference * mask for mask in masks]
    result = torch.zeros((4, 4), dtype=torch.float64, device=image.device)
    for beta, beta_weight in zip(nodes, weights):
        for alpha, alpha_weight in zip(nodes, weights):
            scale = float(alpha) * float(beta)
            coordinate = torch.full((4,), scale, dtype=image.dtype, device=image.device, requires_grad=True)
            root = baseline + sum(coordinate[index] * directions[index] for index in range(4))
            score = model(root)[:, class_id].sum()
            gradient = torch.autograd.grad(score, coordinate, create_graph=True)[0]
            hessian_rows = []
            for row in range(4):
                hessian_rows.append(torch.autograd.grad(
                    gradient[row], coordinate, retain_graph=row < 3, create_graph=False,
                )[0])
            hessian = torch.stack(hessian_rows).detach().double()
            diagonal = torch.diag(gradient.detach().double())
            term = diagonal + float(alpha) * float(beta) * hessian
            result.add_(term * float(alpha_weight) * float(beta_weight))
            if progress is not None:
                progress()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("NONFINITE_INTEGRATED_HESSIANS")
    return result


def integrated_hessians_scores(matrix: Tensor) -> list[float]:
    if matrix.shape != (4, 4):
        raise ValueError("IH_MATRIX_SHAPE")
    return [float(matrix[:, index].abs().sum()) for index in range(4)]


def closure_summary(
    model: nn.Module, block_index: int, fields: Mapping[str, Any], cut_id: str,
) -> dict[str, Any]:
    indexes = cut_indexes(model, block_index, fields["names"], cut_id)
    A, T, R = (fields[name] for name in ("A", "T", "R"))
    full = len(fields["names"]) - 1
    checks = {
        "port_D_to_full_gradient": parity(fields["D"][indexes].sum(0), fields["D"][full]),
        "pcia_to_ig": parity(A[indexes].sum(0), A[full]),
        "pefa_to_endpoint": parity(T[indexes].sum(0), T[full]),
        "T_equals_A_plus_R": parity(T[indexes].sum(0), (A + R)[indexes].sum(0)),
        "endpoint_minus_ig_equals_R": parity(R[indexes].sum(0), T[full] - A[full]),
    }
    return {"pass": all(row["pass"] for row in checks.values()), "checks": checks}
