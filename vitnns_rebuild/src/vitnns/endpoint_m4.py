"""Endpoint M4 structured attention and residual mechanism kernels.

All maps are local input-coordinate VJPs for one frozen class logit.  Attention
is split per head into Q, K and V paths; residuals use the clean cotangent at
the output of each add, so the identity seed never includes the sibling branch.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from types import MethodType
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from .endpoint_nns import _frozen_eval
from .endpoint_m3_protocol import rademacher


class EndpointM4Error(ValueError):
    """A frozen M4 mathematical or numerical contract was violated."""


def _finite(*values: Tensor) -> None:
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise EndpointM4Error("NONFINITE_TENSOR")


@contextmanager
def explicit_graph(
    model: nn.Module, observed: dict[int, dict[str, Tensor]],
    *, detach_attention_blocks: Iterable[int] = (),
):
    """Expose the exact standard-timm attention and residual graph.

    Detaching a selected block's clean attention probabilities changes only its
    backward routing term; the forward tensor remains byte-identical.
    """
    detach = frozenset(int(value) for value in detach_attention_blocks)
    originals, handles = [], []
    try:
        for block_index, block in enumerate(model.blocks):
            attention = block.attn
            if getattr(attention, "fused_attn", True):
                raise EndpointM4Error("UNFUSED_ATTENTION_REQUIRED")
            originals.append((attention, attention.forward, "forward" in attention.__dict__))

            def forward(module: nn.Module, value: Tensor, block_index=block_index):
                batch, tokens, channels = value.shape
                qkv_projection = module.qkv(value)
                qkv = qkv_projection.reshape(
                    batch, tokens, 3, module.num_heads, module.head_dim,
                ).permute(2, 0, 3, 1, 4)
                query_projection, key_projection, value_projection = qkv.unbind(0)
                query = module.q_norm(query_projection)
                key = module.k_norm(key_projection)
                scores = (query * module.scale) @ key.transpose(-2, -1)
                probabilities = scores.softmax(dim=-1)
                effective_probabilities = probabilities.detach() if block_index in detach else probabilities
                after_dropout = module.attn_drop(effective_probabilities)
                av = after_dropout @ value_projection
                projection_input = av.transpose(1, 2).reshape(batch, tokens, channels)
                output = module.proj_drop(module.proj(projection_input))
                observed.setdefault(block_index, {}).update({
                    "query": query, "key": key, "value": value_projection,
                    "attention": probabilities, "av": av,
                    "projection_input": projection_input,
                })
                return output

            attention.forward = MethodType(forward, attention)

            def pre(name: str, block_index=block_index):
                def callback(_module, args):
                    observed.setdefault(block_index, {})[name] = args[0]
                return callback

            def post(name: str, block_index=block_index):
                def callback(_module, _args, output):
                    observed.setdefault(block_index, {})[name] = output
                return callback

            handles.extend((
                block.norm1.register_forward_pre_hook(pre("u")),
                block.drop_path1.register_forward_hook(post("attention_branch")),
                block.norm2.register_forward_pre_hook(pre("r1")),
                block.drop_path2.register_forward_hook(post("mlp_branch")),
                block.register_forward_hook(post("r2")),
            ))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for attention, original, had_instance in reversed(originals):
            if had_instance:
                attention.forward = original
            else:
                del attention.forward


def _summary(value: Tensor) -> dict:
    array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    absolute = math.fsum(map(float, np.abs(flat)))
    signed = math.fsum(map(float, flat))
    l2 = float(np.linalg.norm(flat))
    return {
        "signed_sum": signed, "absolute_mass": absolute, "l2_norm": l2,
        "positive_mass": math.fsum(map(float, np.maximum(flat, 0))),
        "negative_mass": math.fsum(map(float, np.maximum(-flat, 0))),
        "map_cancellation": None if absolute == 0 else 1.0 - abs(signed) / absolute,
        "zero_mass": absolute == 0,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _cosine(left: Tensor, right: Tensor) -> float | None:
    a, b = left.double().flatten(), right.double().flatten()
    denominator = float(a.norm() * b.norm())
    return None if denominator == 0 else float(torch.dot(a, b) / denominator)


def _patch_distribution(value: Tensor) -> Tensor | None:
    if value.shape != (3, 224, 224):
        raise EndpointM4Error("MAP_SHAPE")
    patch = value.abs().reshape(3, 14, 16, 14, 16).sum((0, 2, 4)).double().flatten()
    total = float(patch.sum())
    return None if total == 0 else patch / total


def _total_variation(left: Tensor, right: Tensor) -> float | None:
    p, q = _patch_distribution(left), _patch_distribution(right)
    return None if p is None or q is None else float(0.5 * (p - q).abs().sum())


def _relative_l2(error: Tensor, reference: Tensor) -> float | None:
    denominator = float(reference.double().norm())
    return None if denominator == 0 else float(error.double().norm() / denominator)


def _vjp(output: Tensor, clean: Tensor, seed: Tensor) -> Tensor:
    result = torch.autograd.grad(
        output, clean, grad_outputs=seed.detach(), retain_graph=True, create_graph=False,
    )[0][0]
    _finite(result)
    return result.detach()


def _head_vjps(output: Tensor, clean: Tensor, cotangent: Tensor, chunk_size: int) -> Tensor:
    if output.ndim != 4 or output.shape != cotangent.shape or output.shape[0] != 1:
        raise EndpointM4Error("HEAD_EDGE_SHAPE")
    heads = output.shape[1]
    values = []
    for start in range(0, heads, chunk_size):
        stop = min(heads, start + chunk_size)
        seeds = torch.zeros((stop - start, *output.shape), device=output.device, dtype=output.dtype)
        for lane, head in enumerate(range(start, stop)):
            seeds[lane, 0, head] = cotangent[0, head].detach()
        gradients = torch.autograd.grad(
            output, clean, grad_outputs=seeds, is_grads_batched=True,
            retain_graph=True, create_graph=False,
        )[0][:, 0]
        _finite(gradients)
        values.append(gradients.detach())
    return torch.cat(values)


def _parity(observed: Tensor, expected: Tensor) -> dict:
    error = observed - expected
    maximum = float(error.abs().max())
    reference_maximum = float(expected.abs().max())
    reference_norm = float(expected.double().norm())
    relative = _relative_l2(error, expected)
    elementwise = maximum <= 1e-5 + 1e-4 * reference_maximum
    branch = "absolute" if maximum <= 1e-5 else "relative"
    summary_pass = maximum <= 1e-5 or (relative is not None and relative <= 1e-4)
    return {
        "pass": bool(elementwise and summary_pass), "branch": branch,
        "max_abs": maximum, "relative_l2": relative,
        "reference_norm": reference_norm, "reference_max_abs": reference_maximum,
    }


def _attention_saturation(probabilities: Tensor) -> tuple[list[float], list[float]]:
    if probabilities.ndim != 4 or probabilities.shape[0] != 1:
        raise EndpointM4Error("ATTENTION_SHAPE")
    p = probabilities[0].double()
    entropy = -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1) / math.log(p.shape[-1])
    mean = 1.0 - entropy.mean(-1)
    cls = 1.0 - entropy[:, 0]
    return mean.cpu().tolist(), cls.cpu().tolist()


def scan_endpoint_maps(
    model: nn.Module, image: Tensor, *, class_id: int, chunk_size: int = 6,
    detach_validation: bool = False, progress=None,
) -> dict:
    """Compute per-head attention and both residual-add decompositions."""
    if image.shape != (1, 3, 224, 224) or image.dtype != torch.float32:
        raise EndpointM4Error("INPUT_CONTRACT")
    if len(model.blocks) != 12 or not 0 <= int(class_id) < 1000:
        raise EndpointM4Error("MODEL_OR_TARGET_CONTRACT")
    clean = image.detach().clone().requires_grad_(True)
    observed: dict[int, dict[str, Tensor]] = {}
    native_logits = None
    with _frozen_eval(model), torch.no_grad():
        native_logits = model(image.detach()).detach()
    with _frozen_eval(model), explicit_graph(model, observed), torch.enable_grad():
        logits = model(clean)
        score = logits[:, int(class_id)]
        _finite(logits, score)
        if not torch.allclose(logits.detach(), native_logits, atol=2e-6, rtol=2e-6):
            raise EndpointM4Error("FORWARD_PARITY")
        full_gradient = torch.autograd.grad(score.sum(), clean, retain_graph=True)[0][0].detach()
        full_map = clean.detach()[0] * full_gradient
        attention_rows, residual_rows, validation_rows, block_validation = [], [], [], []
        detach_references = {}
        for block in range(12):
            edge = observed[block]
            required = {"query", "key", "value", "attention", "av", "u",
                        "attention_branch", "r1", "mlp_branch", "r2"}
            if not required.issubset(edge):
                raise EndpointM4Error("CAPTURE_INCOMPLETE")
            gq, gk, gv, gav, g1, g2 = torch.autograd.grad(
                score.sum(), tuple(edge[name] for name in ("query", "key", "value", "av", "r1", "r2")),
                retain_graph=True, create_graph=False,
            )
            q_grad = _head_vjps(edge["query"], clean, gq, chunk_size)
            k_grad = _head_vjps(edge["key"], clean, gk, chunk_size)
            v_grad = _head_vjps(edge["value"], clean, gv, chunk_size)
            native_grad = _head_vjps(edge["av"], clean, gav, chunk_size)
            saturation, cls_saturation = _attention_saturation(edge["attention"].detach())
            q_map, k_map, v_map = clean.detach()[0] * q_grad, clean.detach()[0] * k_grad, clean.detach()[0] * v_grad
            routing_map = q_map + k_map
            explicit_full_map = routing_map + v_map
            native_map = clean.detach()[0] * native_grad
            aggregate = {
                "query": q_map.sum(0), "key": k_map.sum(0),
                "routing": routing_map.sum(0), "value": v_map.sum(0),
                "attention": native_map.sum(0),
            }
            for head in range(native_map.shape[0]):
                native_summary = _summary(native_map[head])
                routing_summary = _summary(routing_map[head])
                value_summary = _summary(v_map[head])
                denominator = routing_summary["absolute_mass"] + value_summary["absolute_mass"]
                path_cancel = None if denominator == 0 else 1.0 - native_summary["absolute_mass"] / denominator
                endpoint_error = native_map[head] - v_map[head]
                gradient_error = native_grad[head] - v_grad[head]
                row = {
                    "block": block, "head": head,
                    "q": _summary(q_map[head]), "k": _summary(k_map[head]),
                    "routing": routing_summary, "value": value_summary, "full": native_summary,
                    "routing_share": _ratio(routing_summary["absolute_mass"], denominator),
                    "frozen_attention_endpoint_error": _relative_l2(endpoint_error, native_map[head]),
                    "frozen_attention_absolute_error": float(endpoint_error.double().norm()),
                    "contracted_gradient_error": _relative_l2(gradient_error, native_grad[head]),
                    "contracted_gradient_absolute_error": float(gradient_error.double().norm()),
                    "routing_value_path_cancellation": path_cancel,
                    "routing_value_cosine": _cosine(routing_map[head], v_map[head]),
                    "mean_query_saturation": float(saturation[head]),
                    "cls_query_saturation": float(cls_saturation[head]),
                    "qkv_native_closure": _parity(explicit_full_map[head], native_map[head]),
                    "frozen_attention_identity": _parity(endpoint_error, routing_map[head]),
                }
                if not row["qkv_native_closure"]["pass"] or not row["frozen_attention_identity"]["pass"]:
                    raise EndpointM4Error(
                        f"ATTENTION_ENDPOINT_CLOSURE:{block}:{head}:"
                        f"{row['qkv_native_closure']}:{row['frozen_attention_identity']}"
                    )
                attention_rows.append(row)
            if detach_validation:
                detach_references[block] = {
                    "cotangent": gav.detach().cpu(), "value_gradient": v_grad.detach().cpu(),
                }
            identity = _vjp(edge["u"], clean, g1)
            attention_branch = _vjp(edge["attention_branch"], clean, g1)
            carry = _vjp(edge["r1"], clean, g2)
            mlp_branch = _vjp(edge["mlp_branch"], clean, g2)
            for residual_type, identity_gradient, branch_gradient in (
                ("attention", identity, attention_branch), ("mlp", carry, mlp_branch),
            ):
                identity_map = clean.detach()[0] * identity_gradient
                branch_map = clean.detach()[0] * branch_gradient
                output_map = identity_map + branch_map
                identity_summary, branch_summary = _summary(identity_map), _summary(branch_map)
                output_summary = _summary(output_map)
                denominator = identity_summary["absolute_mass"] + branch_summary["absolute_mass"]
                closure = _parity(output_map, full_map)
                omitted_error = _parity(branch_map, full_map)
                if identity_summary["absolute_mass"] > 0 and omitted_error["max_abs"] == 0:
                    raise EndpointM4Error("NEGATIVE_CONTROL_FALSE_PASS")
                if not closure["pass"]:
                    raise EndpointM4Error("RESIDUAL_ENDPOINT_CLOSURE")
                residual_rows.append({
                    "block": block, "residual_type": residual_type,
                    "identity": identity_summary, "branch": branch_summary, "output": output_summary,
                    "identity_share": _ratio(identity_summary["absolute_mass"], denominator),
                    "branch_share": _ratio(branch_summary["absolute_mass"], denominator),
                    "path_cancellation": None if denominator == 0 else 1.0 - output_summary["absolute_mass"] / denominator,
                    "identity_branch_cosine": _cosine(identity_map, branch_map),
                    "output_identity_patch_tv": _total_variation(output_map, identity_map),
                    "closure": closure,
                    "negative_control": {
                        "control": "omit_nonzero_identity_or_carry",
                        "component_nonzero": identity_summary["absolute_mass"] > 0,
                        "observed_failure_max_abs": omitted_error["max_abs"],
                        "passed": identity_summary["absolute_mass"] > 0 and omitted_error["max_abs"] > 0,
                    },
                })
                aggregate["identity" if residual_type == "attention" else "carry"] = identity_map
                aggregate["attention" if residual_type == "attention" else "mlp"] = branch_map
            block_validation.append({
                "block": block,
                "components": {name: _summary(value) for name, value in aggregate.items()},
                "head_sum_attention_closure": _parity(native_map.sum(0), aggregate["attention"]),
                "qk_routing_closure": _parity(aggregate["query"] + aggregate["key"], aggregate["routing"]),
                "routing_value_attention_closure": _parity(aggregate["routing"] + aggregate["value"], aggregate["attention"]),
            })
            if progress:
                progress()
        endpoint_digest = hashlib.sha256(full_map.cpu().contiguous().numpy().tobytes()).hexdigest()
    # Release the large original graph before optional detach checks.
    del observed, logits, score
    if detach_validation:
        for block in range(12):
            detached_input = image.detach().clone().requires_grad_(True)
            detached_observed: dict[int, dict[str, Tensor]] = {}
            with _frozen_eval(model), explicit_graph(model, detached_observed, detach_attention_blocks=(block,)), torch.enable_grad():
                detached_logits = model(detached_input)
                if not torch.allclose(detached_logits.detach(), native_logits, atol=2e-6, rtol=2e-6):
                    raise EndpointM4Error("DETACH_FORWARD_PARITY")
                reference = detach_references[block]
                cotangent = reference["cotangent"].to(detached_input.device)
                observed_grad = _head_vjps(detached_observed[block]["av"], detached_input, cotangent, chunk_size)
                expected_grad = reference["value_gradient"].to(observed_grad.device)
                validation_rows.append({
                    "block": block, "attention_detach_value_gradient": _parity(observed_grad, expected_grad),
                    "forward_parity": _parity(detached_logits.detach(), native_logits),
                })
                if not validation_rows[-1]["attention_detach_value_gradient"]["pass"]:
                    raise EndpointM4Error("ATTENTION_DETACH_PARITY")
            if progress:
                progress()
    return {
        "target_value": float(native_logits[0, int(class_id)]),
        "full_endpoint": _summary(full_map), "full_endpoint_sha256": endpoint_digest,
        "attention": attention_rows, "residual": residual_rows,
        "block_validation": block_validation,
        "attention_detach_validation": validation_rows,
        "forward_parity": _parity(native_logits, native_logits),
    }


def attention_jvp_diagnostics(
    model: nn.Module, image: Tensor, *, task_id: str, directions: int = 8, progress=None,
) -> list[dict]:
    """Compare native attention-output JVPs with explicit routing/value JVPs."""
    if directions != 8:
        raise EndpointM4Error("JVP_DIRECTION_COUNT")
    accumulators = {}
    for direction in range(directions):
        array = rademacher(tuple(image.shape), "ENDPOINT_M4_JVP_20260919", task_id, direction)
        tangent = torch.as_tensor(array, device=image.device, dtype=image.dtype)
        tangent = tangent / tangent.norm()

        def callback(value: Tensor):
            captured: dict[int, dict[str, Tensor]] = {}
            with explicit_graph(model, captured):
                model(value)
            output = []
            for block in range(12):
                edge = captured[block]
                output.extend((edge["query"], edge["key"], edge["value"], edge["attention"], edge["av"]))
            return tuple(output)

        with _frozen_eval(model), torch.enable_grad():
            primals, derivatives = torch.autograd.functional.jvp(
                callback, image.detach(), tangent, create_graph=False, strict=True,
            )
        for block in range(12):
            q, k, value, attention, native = primals[5 * block:5 * block + 5]
            dq, dk, dvalue, _dattention_native, dnative = derivatives[5 * block:5 * block + 5]
            dscores = (dq * model.blocks[block].attn.scale) @ k.transpose(-2, -1)
            dscores = dscores + (q * model.blocks[block].attn.scale) @ dk.transpose(-2, -1)
            dattention = attention * (dscores - (attention * dscores).sum(-1, keepdim=True))
            routing = dattention @ value
            frozen = attention @ dvalue
            explicit = routing + frozen
            for head in range(native.shape[1]):
                key_id = (block, head)
                acc = accumulators.setdefault(key_id, {
                    "full_sq": 0.0, "routing_sq": 0.0, "explicit_error_sq": 0.0,
                    "direction_records": [],
                })
                observed = dnative[0, head].double()
                route = routing[0, head].double()
                expected = explicit[0, head].double()
                acc["full_sq"] += float(observed.square().sum())
                acc["routing_sq"] += float(route.square().sum())
                acc["explicit_error_sq"] += float((observed - expected).square().sum())
                acc["direction_records"].append({
                    "direction": direction, "full_l2": float(observed.norm()),
                    "routing_l2": float(route.norm()),
                    "value_l2": float(frozen[0, head].double().norm()),
                    "explicit_error_l2": float((observed - expected).norm()),
                })
        if progress:
            progress()
    rows = []
    for (block, head), acc in sorted(accumulators.items()):
        denominator = math.sqrt(acc["full_sq"])
        rows.append({
            "block": block, "head": head, "directions": directions,
            "frozen_attention_jvp_error": None if denominator == 0 else math.sqrt(acc["routing_sq"]) / denominator,
            "frozen_attention_jvp_absolute_error": math.sqrt(acc["routing_sq"]),
            "explicit_jvp_relative_error": None if denominator == 0 else math.sqrt(acc["explicit_error_sq"]) / denominator,
            "explicit_jvp_absolute_error": math.sqrt(acc["explicit_error_sq"]),
            "direction_records": acc["direction_records"],
        })
    return rows


def scan_m4_task(
    model: nn.Module, image: Tensor, *, class_id: int, task_id: str,
    detach_validation: bool, chunk_size: int = 6, progress=None,
) -> dict:
    maps = scan_endpoint_maps(
        model, image, class_id=class_id, chunk_size=chunk_size,
        detach_validation=detach_validation, progress=progress,
    )
    jvp = attention_jvp_diagnostics(model, image, task_id=task_id, progress=progress)
    by_key = {(row["block"], row["head"]): row for row in jvp}
    for row in maps["attention"]:
        row.update(by_key[(row["block"], row["head"])])
    return maps
