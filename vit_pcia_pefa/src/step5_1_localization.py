"""Step-5.1 controlled use-port nonlinearity and localization primitives.

The intervention is inserted at the actual input use of ``attn.proj``.  Its
clean value and first derivative are exactly the identity, while its bounded
residual has a non-zero second derivative away from the clean activation.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from vit_pcia_pefa.src.pilot import formation_fields
from vit_pcia_pefa.src.vit_adapter import parity, port_layout


def head_component_names(model: nn.Module, block_index: int) -> tuple[str, ...]:
    heads = int(model.blocks[block_index].attn.num_heads)
    return ("identity", *(f"head_{index}" for index in range(heads)), "__full__")


def capture_projection_input(model: nn.Module, image: Tensor, block_index: int) -> Tensor:
    """Capture the unmodified consumer input for one attention projection."""
    observed: list[Tensor] = []

    def save(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        if len(observed) != 0 or not args or not isinstance(args[0], Tensor):
            raise ValueError("PROJECTION_CAPTURE_AMBIGUOUS")
        observed.append(args[0].detach().clone())

    handle = model.blocks[block_index].attn.proj.register_forward_pre_hook(save)
    try:
        with torch.no_grad():
            model(image)
    finally:
        handle.remove()
    if len(observed) != 1:
        raise ValueError("PROJECTION_CAPTURE_COUNT")
    return observed[0]


def _scale_for_head(clean: Tensor, head_index: int, head_count: int) -> Tensor:
    if clean.ndim != 3 or clean.shape[-1] % head_count:
        raise ValueError("PROJECTION_HEAD_LAYOUT")
    width = clean.shape[-1] // head_count
    selected = clean[..., head_index * width:(head_index + 1) * width]
    # Detached, image-specific conditioning scale.  It is not a changing root.
    return selected.detach().double().square().mean().sqrt().clamp_min(1e-6).to(clean)


@contextmanager
def projection_head_wrapper(
    model: nn.Module, block_index: int, head_index: int, clean_projection_input: Tensor,
    *, strength: float, mode: str = "bounded_quadratic",
) -> Iterator[None]:
    """Insert a scoped wrapper at one real projection consumer use-port.

    For ``bounded_quadratic`` the selected slice is transformed as
    ``z + alpha*s*tanh(((z-z*)/s)^2)``.  At ``z=z*`` its residual value and
    Jacobian are zero.  ``linear_tangent`` is the exact affine tangent of that
    wrapper at the clean point, hence the identity; it is the prescribed
    no-curvature control.
    """
    attention = model.blocks[block_index].attn
    heads = int(attention.num_heads)
    if not 0 <= head_index < heads or not math.isfinite(strength) or strength < 0:
        raise ValueError("WRAPPER_ARGUMENT")
    if mode not in {"bounded_quadratic", "linear_tangent"}:
        raise ValueError("WRAPPER_MODE")
    clean = clean_projection_input.detach()
    scale = _scale_for_head(clean, head_index, heads)

    def transform(_module: nn.Module, args: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if not args or not isinstance(args[0], Tensor) or args[0].shape != clean.shape:
            raise ValueError("WRAPPER_INPUT_DRIFT")
        value = args[0]
        width = value.shape[-1] // heads
        begin, end = head_index * width, (head_index + 1) * width
        selected = value[..., begin:end]
        center = clean[..., begin:end].to(device=value.device, dtype=value.dtype)
        if mode == "linear_tangent" or strength == 0.0:
            replacement = selected
        else:
            delta = (selected - center) / scale
            replacement = selected + float(strength) * scale * torch.tanh(delta.square())
        output = torch.cat((value[..., :begin], replacement, value[..., end:]), dim=-1)
        return (output, *args[1:])

    handle = attention.proj.register_forward_pre_hook(transform, prepend=True)
    try:
        yield
    finally:
        handle.remove()


def endpoint_and_logit(model: nn.Module, image: Tensor, class_id: int) -> tuple[Tensor, float, int]:
    root = image.detach().requires_grad_(True)
    logits = model(root)
    score = logits[:, class_id].sum()
    gradient = torch.autograd.grad(score, root)[0]
    endpoint = root.detach().double() * gradient.detach().double()
    if not bool(torch.isfinite(endpoint).all()) or not math.isfinite(float(score)):
        raise ValueError("NONFINITE_ENDPOINT")
    return endpoint, float(score.detach()), int(logits.detach().argmax(1).item())


def cut_formation_fields(
    model: nn.Module, image: Tensor, *, block_index: int, class_id: int,
    nodes: Sequence[float], weights: Sequence[float], progress=None,
) -> dict[str, Any]:
    """Compute source-summed R only for one frozen complete head cut."""
    names = head_component_names(model, block_index)
    return formation_fields(
        model, image=image, baseline=torch.zeros_like(image), waypoints=(),
        block_index=block_index, class_id=class_id, nodes=nodes, weights=weights,
        progress=progress, requested_names=names,
    )


def clean_parity(
    clean: Mapping[str, Any], modified: Mapping[str, Any],
    *, clean_endpoint: Tensor, modified_endpoint: Tensor, clean_logit: float,
    modified_logit: float, clean_prediction: int, modified_prediction: int,
    atol: float, rtol: float, near_zero: float,
) -> dict[str, Any]:
    checks = {
        "endpoint": parity(modified_endpoint, clean_endpoint, atol=atol, rtol=rtol, near_zero=near_zero),
        "terminal_D": parity(modified["D"], clean["D"], atol=atol, rtol=rtol, near_zero=near_zero),
        "terminal_Q": parity(modified["T"], clean["T"], atol=atol, rtol=rtol, near_zero=near_zero),
    }
    logit_error = abs(modified_logit - clean_logit)
    logit_limit = atol + rtol * abs(clean_logit)
    return {
        "pass": (all(row["pass"] for row in checks.values())
                 and logit_error <= logit_limit and modified_prediction == clean_prediction),
        "checks": checks, "logit_abs_error": logit_error,
        "logit_limit": logit_limit, "prediction_equal": modified_prediction == clean_prediction,
    }


def _l1_per_port(value: Tensor) -> list[float]:
    return [float(row.detach().double().abs().sum()) for row in value]


def activation_head_norms(value: Tensor, head_count: int) -> list[float]:
    width = value.shape[-1] // head_count
    return [float(value[..., i * width:(i + 1) * width].double().norm()) for i in range(head_count)]


def localization_scores(
    clean: Mapping[str, Any], modified: Mapping[str, Any], *, clean_activation: Tensor,
    modified_activation: Tensor, random_namespace: str,
) -> dict[str, list[float]]:
    names = tuple(clean["names"]); other_names = tuple(modified["names"])
    if names != other_names or names[0] != "identity" or names[-1] != "__full__":
        raise ValueError("LOCALIZATION_PORT_ORDER")
    head_indexes = [i for i, name in enumerate(names) if name.startswith("head_")]
    clean_heads = clean_activation.shape[-1] // len(head_indexes)
    if clean_heads <= 0 or clean_heads * len(head_indexes) != clean_activation.shape[-1]:
        raise ValueError("LOCALIZATION_HEAD_LAYOUT")
    methods = {
        "pefa_R_delta": _l1_per_port(modified["R"][head_indexes] - clean["R"][head_indexes]),
        "pcia_delta": _l1_per_port(modified["A"][head_indexes] - clean["A"][head_indexes]),
        "endpoint_delta": _l1_per_port(modified["T"][head_indexes] - clean["T"][head_indexes]),
        "gradient_delta": _l1_per_port(modified["D"][head_indexes] - clean["D"][head_indexes]),
        "activation_delta": [abs(a - b) for a, b in zip(
            activation_head_norms(modified_activation, len(head_indexes)),
            activation_head_norms(clean_activation, len(head_indexes)),
        )],
        "endpoint_level": _l1_per_port(clean["T"][head_indexes]),
        "pcia_level": _l1_per_port(clean["A"][head_indexes]),
        "gradient_level": _l1_per_port(clean["D"][head_indexes]),
        "activation_level": activation_head_norms(clean_activation, len(head_indexes)),
        "random": [int.from_bytes(hashlib.sha256(
            f"{random_namespace}|{index}".encode()).digest()[:8], "big") / float(2**64)
                   for index in range(len(head_indexes))],
    }
    if not all(math.isfinite(item) for values in methods.values() for item in values):
        raise ValueError("NONFINITE_LOCALIZATION_SCORE")
    return methods


def rank_metrics(scores: Sequence[float], true_index: int, top_k: int) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not 0 <= true_index < len(values) or not 1 <= top_k <= len(values):
        raise ValueError("RANK_METRIC_INPUT")
    # Descending stable ranks; ties get the conservative worst rank.
    greater = int(np.sum(values > values[true_index]))
    equal = int(np.sum(values == values[true_index]))
    rank = greater + equal
    return {
        "rank": rank, "top1": rank == 1, "top_k": rank <= top_k,
        "reciprocal_rank": 1.0 / rank,
        "percentile_rank": 1.0 if len(values) == 1 else 1.0 - (rank - 1) / (len(values) - 1),
        "true_score": float(values[true_index]), "maximum_score": float(values.max()),
        "tie_count_at_true": equal,
    }


def difference_direction(clean: Mapping[str, Any], modified: Mapping[str, Any]) -> dict[str, float | bool]:
    names = tuple(clean["names"]); indexes = [i for i, name in enumerate(names) if name != "__full__"]
    delta_ports = (modified["R"][indexes] - clean["R"][indexes]).sum(0).flatten().double()
    delta_g = (modified["R"][-1] - clean["R"][-1]).flatten().double()
    denominator = float(delta_ports.norm() * delta_g.norm())
    cosine = 1.0 if denominator == 0.0 and torch.equal(delta_ports, delta_g) else (
        float(torch.dot(delta_ports, delta_g) / denominator) if denominator else 0.0
    )
    closure = parity(delta_ports, delta_g)
    return {
        "pass": bool(closure["pass"]), "cosine": cosine,
        "delta_g_l1": float(delta_g.abs().sum()),
        "delta_cut_l1": float(delta_ports.abs().sum()),
        "max_abs_closure_error": float(closure["max_abs"]),
    }


def audit_head_port_semantics(cut: Mapping[str, Any]) -> dict[str, Any]:
    """Check that ordered runtime head components retain their registered slices."""
    ports = list(cut.get("port_ids", ()))
    if cut.get("cut_id") != "HEADS_WITH_BYPASS_FULL" or len(ports) < 2:
        raise ValueError("HEAD_CUT_SCHEMA")
    errors = []
    identity = ports[0]
    if identity.get("input_slot") != 0 or identity.get("slice_spec") != "ALL":
        errors.append({"component": "identity", "code": "FAIL_PORT_ID_SEMANTIC_MISMATCH"})
    for index, port in enumerate(ports[1:]):
        expected = f"HEAD[{index}]"
        if (port.get("consumer_operator_id", "").split(".")[-1] != "proj"
                or port.get("input_slot") != 0 or port.get("slice_spec") != expected):
            errors.append({"component": f"head_{index}", "expected_slice_spec": expected,
                           "observed_slice_spec": port.get("slice_spec"),
                           "code": "FAIL_PORT_ID_SEMANTIC_MISMATCH"})
    return {"pass": not errors, "block_index": cut.get("block_index"),
            "cut_id": cut.get("cut_id"), "category": "head", "errors": errors}
