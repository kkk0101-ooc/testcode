"""Prospective Step-5.2 prediction and natural-head intervention primitives.

Predictions are computed only from the unmodified model.  The intervention
replaces one block-5 attention-head contribution by the affine tangent of the
complete normalized-residual-to-head mapping at the clean activation.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from vit_pcia_pefa.src.pilot import spearman
from vit_pcia_pefa.src.step5_1_localization import endpoint_and_logit
from vit_pcia_pefa.src.vit_adapter import parity


def capture_attention_context(
    model: nn.Module, image: Tensor, block_index: int,
) -> tuple[Tensor, Tensor]:
    """Capture normalized residual input and concatenated head output."""
    attention = model.blocks[block_index].attn
    inputs: list[Tensor] = []
    outputs: list[Tensor] = []

    def save_input(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        if inputs or not args or not isinstance(args[0], Tensor):
            raise ValueError("ATTENTION_INPUT_CAPTURE_AMBIGUOUS")
        inputs.append(args[0].detach().clone())

    def save_output(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        if outputs or not args or not isinstance(args[0], Tensor):
            raise ValueError("HEAD_OUTPUT_CAPTURE_AMBIGUOUS")
        outputs.append(args[0].detach().clone())

    first = attention.register_forward_pre_hook(save_input)
    second = attention.proj.register_forward_pre_hook(save_output)
    try:
        with torch.no_grad():
            model(image)
    finally:
        second.remove()
        first.remove()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("ATTENTION_CONTEXT_CAPTURE_COUNT")
    return inputs[0], outputs[0]


def attention_head_level_norms(
    model: nn.Module, image: Tensor, block_index: int, class_id: int,
) -> tuple[list[float], list[float]]:
    """Return ordinary activation and local cotangent L2 norms per head."""
    observed: list[Tensor] = []

    def save(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        if observed or not args or not isinstance(args[0], Tensor):
            raise ValueError("HEAD_NORM_CAPTURE_AMBIGUOUS")
        observed.append(args[0])

    handle = model.blocks[block_index].attn.proj.register_forward_pre_hook(save)
    try:
        root = image.detach().requires_grad_(True)
        logits = model(root)
        if len(observed) != 1:
            raise ValueError("HEAD_NORM_CAPTURE_COUNT")
        gradient = torch.autograd.grad(logits[:, class_id].sum(), observed[0])[0]
    finally:
        handle.remove()
    heads = int(model.blocks[block_index].attn.num_heads)
    activation = observed[0].detach(); gradient = gradient.detach()
    if activation.shape[-1] % heads:
        raise ValueError("HEAD_NORM_LAYOUT")
    width = activation.shape[-1] // heads
    activation_norms = [float(activation[..., i * width:(i + 1) * width].double().norm()) for i in range(heads)]
    gradient_norms = [float(gradient[..., i * width:(i + 1) * width].double().norm()) for i in range(heads)]
    return activation_norms, gradient_norms


def _natural_heads(attention: nn.Module, value: Tensor) -> Tensor:
    """Registered unfused timm attention through per-head AV outputs."""
    batch, tokens, channels = value.shape
    qkv = attention.qkv(value).reshape(
        batch, tokens, 3, attention.num_heads, attention.head_dim,
    ).permute(2, 0, 3, 1, 4)
    query, key, projected_value = qkv.unbind(0)
    query, key = attention.q_norm(query), attention.k_norm(key)
    probabilities = (query.mul(attention.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
    probabilities = attention.attn_drop(probabilities)
    return probabilities @ projected_value


def _head_to_projection_slice(heads: Tensor) -> Tensor:
    batch, head_count, tokens, width = heads.shape
    return heads.transpose(1, 2).reshape(batch, tokens, head_count * width)


@contextmanager
def natural_head_linearization(
    model: nn.Module, block_index: int, head_index: int, clean_attention_input: Tensor,
    *, strength: float,
) -> Iterator[None]:
    """Scale one natural head's nonlinear remainder by ``strength``.

    ``strength=1`` is the untouched natural mapping and ``strength=0`` is its
    exact clean-point affine tangent.  The replacement is installed at the
    actual ``attn.proj`` consumer input and remains visible to the registered
    PEFA cut capture.
    """
    attention = model.blocks[block_index].attn
    heads = int(attention.num_heads)
    if not 0 <= head_index < heads or not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("HEAD_LINEARIZATION_ARGUMENT")
    if getattr(attention, "fused_attn", True):
        raise ValueError("STRUCTURAL_BACKEND_REQUIRES_UNFUSED_ATTENTION")
    clean = clean_attention_input.detach().clone()
    state: dict[str, Tensor] = {}

    def remember_input(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        if not args or not isinstance(args[0], Tensor):
            raise ValueError("ATTENTION_INPUT_NOT_TENSOR")
        state["input"] = args[0]

    def replace_head(_module: nn.Module, args: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if not args or not isinstance(args[0], Tensor) or "input" not in state:
            raise ValueError("HEAD_LINEARIZATION_STATE")
        current_input = state.pop("input")
        natural = args[0]
        if current_input.shape != clean.shape or natural.shape[-1] % heads:
            raise ValueError("HEAD_LINEARIZATION_SHAPE")
        width = natural.shape[-1] // heads
        begin, end = head_index * width, (head_index + 1) * width

        # [B,H,N,D] -> the selected [B,N,D] head.  The explicit indexing is
        # kept in the closure so JVP covers Q/K/V, softmax and AV together.
        def selected_projection_slice(value: Tensor) -> Tensor:
            return _natural_heads(attention, value)[:, head_index]

        clean_head, tangent_head = torch.autograd.functional.jvp(
            selected_projection_slice, clean, (current_input - clean,),
            create_graph=True, strict=True,
        )
        affine = clean_head + tangent_head
        natural_head = natural[..., begin:end]
        # selected_projection_slice is [B,N,D], matching the concatenated slice.
        replacement = affine + float(strength) * (natural_head - affine)
        output = torch.cat((natural[..., :begin], replacement, natural[..., end:]), dim=-1)
        return (output, *args[1:])

    input_handle = attention.register_forward_pre_hook(remember_input, prepend=True)
    output_handle = attention.proj.register_forward_pre_hook(replace_head, prepend=True)
    try:
        yield
    finally:
        output_handle.remove()
        input_handle.remove()


def clean_prediction_scores(
    clean: Mapping[str, Any], *, activation_norms: Sequence[float],
    gradient_norms: Sequence[float], random_namespace: str,
) -> dict[str, list[float]]:
    names = tuple(clean["names"])
    indexes = [index for index, name in enumerate(names) if name.startswith("head_")]
    if names != head_component_names_from_names(names) or not indexes:
        raise ValueError("PREDICTION_PORT_ORDER")
    methods = {
        "pefa_R": [float(row.detach().double().abs().sum()) for row in clean["R"][indexes]],
        "pcia_A": [float(row.detach().double().abs().sum()) for row in clean["A"][indexes]],
        "endpoint_Q": [float(row.detach().double().abs().sum()) for row in clean["T"][indexes]],
        "gradient_norm": [float(value) for value in gradient_norms],
        "activation_norm": [float(value) for value in activation_norms],
        "random": [int.from_bytes(hashlib.sha256(
            f"{random_namespace}|{index}".encode()).digest()[:8], "big") / float(2**64)
                   for index in range(len(indexes))],
    }
    if any(len(scores) != len(indexes) for scores in methods.values()):
        raise ValueError("PREDICTION_SCORE_LENGTH")
    if not all(math.isfinite(value) for scores in methods.values() for value in scores):
        raise ValueError("NONFINITE_PREDICTION_SCORE")
    return methods


def head_component_names_from_names(names: Sequence[str]) -> tuple[str, ...]:
    heads = sum(name.startswith("head_") for name in names)
    return ("identity", *(f"head_{index}" for index in range(heads)), "__full__")


def signed_head_maps(clean: Mapping[str, Any]) -> dict[str, np.ndarray]:
    names = tuple(clean["names"]); indexes = [i for i, name in enumerate(names) if name.startswith("head_")]
    return {key: clean[key][indexes].detach().cpu().double().numpy() for key in ("R", "A", "T")}


def intervention_outcome(
    clean: Mapping[str, Any], modified: Mapping[str, Any], *, head_index: int, epsilon: float,
) -> dict[str, Any]:
    names = tuple(clean["names"]); indexes = [i for i, name in enumerate(names) if name.startswith("head_")]
    if names != tuple(modified["names"]) or not 0 <= head_index < len(indexes):
        raise ValueError("OUTCOME_PORT_CONTRACT")
    clean_g = clean["R"][-1].detach().double()
    delta_g = modified["R"][-1].detach().double() - clean_g
    negative_r = -clean["R"][indexes[head_index]].detach().double()
    denominator = float(negative_r.norm() * delta_g.norm())
    cosine = None if denominator == 0.0 else float(torch.dot(negative_r.flatten(), delta_g.flatten()) / denominator)
    active = (negative_r.abs() > epsilon) | (delta_g.abs() > epsilon)
    sign_agreement = None if not bool(active.any()) else float(
        (torch.sign(negative_r[active]) == torch.sign(delta_g[active])).double().mean()
    )
    return {
        "Y": float(delta_g.abs().sum() / (clean_g.abs().sum() + epsilon)),
        "delta_G_l1": float(delta_g.abs().sum()),
        "clean_G_l1": float(clean_g.abs().sum()),
        "negative_R_delta_G_cosine": cosine,
        "negative_R_delta_G_sign_agreement": sign_agreement,
        "delta_G": delta_g,
    }


def intervention_parity(
    clean: Mapping[str, Any], modified: Mapping[str, Any], *, clean_endpoint: Tensor,
    modified_endpoint: Tensor, clean_logit: float, modified_logit: float,
    clean_prediction: int, modified_prediction: int, atol: float, rtol: float,
    near_zero: float,
) -> dict[str, Any]:
    checks = {
        "input_gradient": parity(modified["D"][-1], clean["D"][-1], atol=atol, rtol=rtol, near_zero=near_zero),
        "endpoint": parity(modified_endpoint, clean_endpoint, atol=atol, rtol=rtol, near_zero=near_zero),
        "terminal_Q": parity(modified["T"][-1], clean["T"][-1], atol=atol, rtol=rtol, near_zero=near_zero),
    }
    logit_error = abs(modified_logit - clean_logit); logit_limit = atol + rtol * abs(clean_logit)
    return {
        "pass": all(row["pass"] for row in checks.values()) and logit_error <= logit_limit
                and modified_prediction == clean_prediction,
        "checks": checks, "logit_abs_error": logit_error, "logit_limit": logit_limit,
        "prediction_equal": modified_prediction == clean_prediction,
    }


def convergence_check(
    coarse: Mapping[str, Any], fine: Mapping[str, Any], *, atol: float, rtol: float, near_zero: float,
    rule: str = "strict_elementwise_and_global_relative",
) -> dict[str, Any]:
    indexes = [i for i, name in enumerate(fine["names"]) if name.startswith("head_")]
    if tuple(coarse["names"]) != tuple(fine["names"]):
        raise ValueError("CONVERGENCE_PORT_ORDER")
    checks = {
        "source_summed_R_e": parity(coarse["R"][indexes], fine["R"][indexes], atol=atol, rtol=rtol, near_zero=near_zero),
        "IG": parity(coarse["A"][-1], fine["A"][-1], atol=atol, rtol=rtol, near_zero=near_zero),
        "G": parity(coarse["R"][-1], fine["R"][-1], atol=atol, rtol=rtol, near_zero=near_zero),
    }
    if rule == "strict_elementwise_and_global_relative":
        for row in checks.values():
            row["gate_pass"] = row["pass"]
    elif rule == "global_relative_l2_or_absolute_near_zero":
        # Integrated signed maps contain many coordinates whose reference is
        # locally near zero.  Elementwise relative error is therefore not a
        # stable convergence gate for the map as a whole.  This rule retains
        # the elementwise diagnostic but gates nonzero tensors on their global
        # relative L2 error and genuinely near-zero tensors on max absolute
        # error.
        for row in checks.values():
            row["gate_pass"] = (
                row["max_abs"] <= atol if row["reference_norm"] <= near_zero
                else row["relative_l2"] <= rtol
            )
    else:
        raise ValueError("UNKNOWN_CONVERGENCE_RULE")
    return {"pass": all(row["gate_pass"] for row in checks.values()),
            "rule": rule, "checks": checks}


def rank_order(scores: Sequence[float]) -> list[int]:
    if not scores or not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("RANK_ORDER_INPUT")
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def prediction_metrics(scores: Sequence[float], outcomes: Sequence[float], top_k: int) -> dict[str, Any]:
    if len(scores) != len(outcomes) or not 1 <= top_k <= len(scores):
        raise ValueError("PREDICTION_METRIC_INPUT")
    predicted = rank_order(scores); actual = rank_order(outcomes)
    total = math.fsum(float(value) for value in outcomes)
    captured = math.fsum(float(outcomes[index]) for index in predicted[:top_k])
    return {
        "spearman": spearman(scores, outcomes),
        "predicted_top1": predicted[0], "actual_top1": actual[0],
        "top1_hit": predicted[0] == actual[0], "top_k": top_k,
        "top_k_effect_capture": 0.0 if total == 0.0 else captured / total,
        "predicted_order": predicted, "actual_order": actual,
    }


def bootstrap_mean_interval(
    values: Sequence[float], *, resamples: int, confidence: float, seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("BOOTSTRAP_VALUES")
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("BOOTSTRAP_CONTRACT")
    generator = np.random.default_rng(int(seed))
    samples = generator.choice(array, size=(int(resamples), len(array)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {"n": len(array), "mean": float(array.mean()), "confidence": confidence,
            "lower": float(np.quantile(samples, alpha)), "upper": float(np.quantile(samples, 1.0 - alpha)),
            "resamples": int(resamples), "seed": int(seed)}
