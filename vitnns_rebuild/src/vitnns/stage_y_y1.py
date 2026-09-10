"""Stage Y Y1 complete-cut kernels, transforms, interventions, and accounting.

The routing implementation is exact and factorized.  It combines native
reverse cotangents with grouped forward JVPs whose tangent directions are the
196 disjoint input patches.  It never forms a pixel-by-hidden Jacobian.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .stage_x_nns_v2 import (
    ALPHA_CHUNK, CUT_PIXEL_ATOL, CUT_PIXEL_RTOL, CUT_TOTAL_ATOL,
    CUT_TOTAL_RTOL, ENDPOINT_LT, INPUT_ATOL, INPUT_RTOL, STEPS,
    NNSV2Error, array_digest as x1_array_digest, gauss_legendre,
)

PROTOCOL_ID = "STAGE_Y_VIT_BLOCK_LEVEL_NNS_MECHANISM_V1_20260910"
STAGE_ID = "Y1"
METHOD_ID = "stage_y_complete_cut_grouped_patch_jvp_native_vjp_v1"
PATCHES = 196
TOKENS = 197
BLOCKS = 12
HEADS = 12
CELLS = 56
TIE_NAMESPACE = "STAGE_Y_WITHIN_PATCH_TIE_V1_20260910"
TRANSFORMS = (
    "clean", "horizontal_flip", "translate_x_-8", "translate_x_8",
    "rotate_-10", "rotate_10", "brightness_0.9",
    "brightness_1.1", "gaussian_noise_0.01_seed_0",
)


class StageYY1Error(ValueError):
    """Closed-contract implementation or evidence failure."""


class StageYY1NumericalError(StageYY1Error):
    code = "FAILED_NUMERICAL"


class StageYY1BudgetError(StageYY1Error):
    code = "BLOCKED_BUDGET"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "$schema": "stage_y_y1_artifact.schema.json",
        "schema_version": 1,
        "document_type": document_type,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": status,
        "payload": dict(payload),
    }
    result["canonical_digest"] = canonical_digest(result)
    return result


def validate_signed(value: Mapping[str, Any]) -> None:
    expected = {"$schema", "schema_version", "document_type", "protocol_id", "stage_id", "status", "payload", "canonical_digest"}
    if set(value) != expected:
        raise StageYY1Error("artifact envelope is not closed")
    unsigned = dict(value)
    observed = unsigned.pop("canonical_digest")
    if observed != canonical_digest(unsigned):
        raise StageYY1Error("artifact canonical digest mismatch")
    if value["protocol_id"] != PROTOCOL_ID or value["stage_id"] != STAGE_ID:
        raise StageYY1Error("artifact protocol/stage drift")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(value: np.ndarray | torch.Tensor) -> str:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    array = np.ascontiguousarray(array)
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and not np.little_endian):
        array = array.byteswap().newbyteorder("<")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_scientific_array(value: np.ndarray, shape: Sequence[int], dtype: np.dtype = np.dtype("float32")) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != tuple(shape) or array.dtype != dtype or not array.flags.c_contiguous:
        raise StageYY1Error(f"scientific array contract drift: {array.shape}/{array.dtype}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise StageYY1Error("scientific array contains NaN/Inf")
    return array


def seed64(namespace: str, *parts: object) -> int:
    text = namespace + "\0" + "\0".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def tie_permutation(source_object_id: str, block: int, patch: int, seed_index: int) -> np.ndarray:
    seed = seed64(TIE_NAMESPACE, source_object_id, block, patch, seed_index)
    return np.random.Generator(np.random.PCG64(seed)).permutation(16)


def stable_descending(values: np.ndarray, namespace: str, *parts: object) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(data).all():
        raise StageYY1Error("rank input contains NaN/Inf")
    tie = np.asarray([seed64(namespace, *parts, index) for index in range(data.size)], dtype=np.uint64)
    return np.lexsort((tie, -data))


def mass_summary(value: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise StageYY1Error("mass input contains NaN/Inf")
    positive = float(np.maximum(array, 0.0).sum(dtype=np.float64))
    negative = float(np.minimum(array, 0.0).sum(dtype=np.float64))
    absolute = float(np.abs(array).sum(dtype=np.float64))
    signed_value = float(array.sum(dtype=np.float64))
    cancellation = None if absolute == 0 else 1.0 - abs(signed_value) / absolute
    return {
        "signed_conductance": signed_value, "positive_mass": positive,
        "negative_mass": negative, "absolute_mass": absolute,
        "cancellation": cancellation,
    }


def pool_cells(value: torch.Tensor) -> torch.Tensor:
    if value.shape != (3, 224, 224):
        raise StageYY1Error("cell pooling requires [3,224,224]")
    return value.double().sum(0).reshape(56, 4, 56, 4).sum((1, 3)).float().contiguous()


def pool_source_patches(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-3:] != (3, 224, 224):
        raise StageYY1Error("patch pooling requires trailing [3,224,224]")
    prefix = value.shape[:-3]
    return value.double().sum(-3).reshape(*prefix, 14, 16, 14, 16).sum((-3, -1)).reshape(*prefix, PATCHES)


def _rgb(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if image.shape != (1, 3, 224, 224) or image.dtype != torch.float32 or not bool(torch.isfinite(image).all()):
        raise StageYY1Error("transform input must be finite float32 [1,3,224,224]")
    mean = torch.tensor((0.5, 0.5, 0.5), device=image.device, dtype=image.dtype).reshape(1, 3, 1, 1)
    std = mean.clone()
    return (image * std + mean).clamp(0, 1), mean, std


def transform_image(image: torch.Tensor, condition: str, source_object_id: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the nine frozen Y0 transforms; Y1 empirical calls only clean/flip."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tf

    if condition not in TRANSFORMS:
        raise StageYY1Error(f"unknown frozen transform: {condition}")
    rgb, mean, std = _rgb(image)
    value = rgb[0]
    meta: dict[str, Any] = {"condition": condition, "inverse": "identity", "full_domain": True}
    if condition == "clean":
        result = value
    elif condition == "horizontal_flip":
        result = tf.hflip(value); meta["inverse"] = "horizontal_flip"
    elif condition.startswith("translate_x_"):
        amount = int(condition.removeprefix("translate_x_")); pad = abs(amount)
        padded = torch.nn.functional.pad(value, (pad, pad, 0, 0), mode="reflect")
        result = padded[:, :, pad - amount:pad - amount + 224]
        meta.update({"inverse": "translate_x", "amount": amount, "full_domain": False})
    elif condition.startswith("rotate_"):
        angle = float(condition.removeprefix("rotate_"))
        result = tf.rotate(value, angle, interpolation=InterpolationMode.BILINEAR, fill=0.5)
        meta.update({"inverse": "rotate", "angle": angle, "full_domain": False})
    elif condition.startswith("brightness_"):
        result = tf.adjust_brightness(value, float(condition.removeprefix("brightness_"))).clamp(0, 1)
    elif condition == "gaussian_noise_0.01_seed_0":
        seed = seed64("STAGE_Y_NOISE_V1_20260910", source_object_id, condition)
        noise = np.random.Generator(np.random.PCG64(seed)).normal(0, 0.01, tuple(value.shape)).astype(np.float32)
        result = (value + torch.from_numpy(noise).to(value.device)).clamp(0, 1)
        meta["seed"] = seed
    else:  # pragma: no cover - roster above is exhaustive
        raise StageYY1Error("unreachable transform")
    return ((result.unsqueeze(0) - mean) / std).contiguous(), meta


def inverse_map(value: torch.Tensor, metadata: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tf

    if value.shape != (224, 224) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise StageYY1Error("inverse map requires finite float32 [224,224]")
    image = value.unsqueeze(0); support = torch.ones_like(image)
    kind = metadata.get("inverse")
    if kind == "identity":
        restored, returned = image, support
    elif kind == "horizontal_flip":
        restored, returned = tf.hflip(image), tf.hflip(support)
    elif kind == "translate_x":
        amount = int(metadata["amount"]); restored = torch.zeros_like(image); returned = torch.zeros_like(support)
        if amount > 0:
            restored[:, :, :-amount] = image[:, :, amount:]; returned[:, :, :-amount] = 1
        else:
            count = -amount; restored[:, :, count:] = image[:, :, :-count]; returned[:, :, count:] = 1
    elif kind == "rotate":
        angle = float(metadata["angle"])
        transformed_support = tf.rotate(support, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
        restored = tf.rotate(image, -angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
        returned = tf.rotate(transformed_support, -angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
    else:
        raise StageYY1Error("malformed inverse metadata")
    return restored[0].cpu().numpy().astype(np.float32, copy=False), (returned[0] >= 0.999999).cpu().numpy()


@dataclass
class ForwardCapture:
    logits: torch.Tensor
    tensors: dict[str, torch.Tensor]
    heads: dict[int, torch.Tensor]


def native_forward_capture(model: nn.Module, image: torch.Tensor) -> ForwardCapture:
    """Reproduce the timm forward while exposing native complete-cut edges."""
    x = model.patch_embed(image)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)
    tensors: dict[str, torch.Tensor] = {"post_embedding": x}
    heads: dict[int, torch.Tensor] = {}
    for block_index, block in enumerate(model.blocks):
        prefix = f"b{block_index}"; tensors[prefix + ".input"] = x
        normalized = block.norm1(x); attention = block.attn
        batch, tokens, width = normalized.shape
        qkv = attention.qkv(normalized).reshape(batch, tokens, 3, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
        query, key, values = qkv.unbind(0); query, key = attention.q_norm(query), attention.k_norm(key)
        query = query * attention.scale
        weights = attention.attn_drop((query @ key.transpose(-2, -1)).softmax(dim=-1))
        mixed = weights @ values
        concatenated = mixed.transpose(1, 2).reshape(batch, tokens, width)
        projected = attention.proj(concatenated)
        projected = attention.proj_drop(projected)
        branch = block.drop_path1(block.ls1(projected))
        # Retain native value-mixed heads before concat/projection.  Their
        # cotangent is taken at the native concat input, which is exactly the
        # matching output-projection-column accounting and excludes the
        # projection bias's zero direct derivative.
        heads[block_index] = mixed
        tensors[prefix + ".head_concat"] = concatenated
        tensors[prefix + ".attention_branch"] = branch
        add1 = x + branch; tensors[prefix + ".add1"] = add1
        mlp = block.drop_path2(block.ls2(block.mlp(block.norm2(add1))))
        tensors[prefix + ".mlp_branch"] = mlp
        x = add1 + mlp; tensors[prefix + ".output"] = x
    x = model.norm(x); tensors["final_norm"] = x
    if model.attn_pool is not None or model.global_pool == "avg":
        raise StageYY1Error("Y1 requires native CLS pooling")
    classifier_input = model.fc_norm(x[:, 0])
    classifier_input = model.head_drop(classifier_input)
    tensors["classifier_input"] = classifier_input
    logits = model.head(classifier_input)
    return ForwardCapture(logits, tensors, heads)


def capture_keys() -> tuple[str, ...]:
    keys = ["post_embedding"]
    for block in range(BLOCKS):
        keys.extend((f"b{block}.input", f"b{block}.head_concat", f"b{block}.attention_branch", f"b{block}.add1", f"b{block}.mlp_branch", f"b{block}.output"))
    keys.extend(("final_norm", "classifier_input"))
    return tuple(keys)


def _token_dot(value: torch.Tensor, gradient: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (value * gradient * weights[:, None, None]).sum((0, 2))


def _total_dot(value: torch.Tensor, gradient: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (value * gradient * weights.reshape((-1,) + (1,) * (value.ndim - 1))).sum()


def routing_scalars(capture: ForwardCapture, cotangents: Mapping[str, torch.Tensor], weights: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    values: list[torch.Tensor] = []; layout: list[dict[str, Any]] = []
    def tokens(cut: str, component: str, tensor: torch.Tensor, gradient: torch.Tensor) -> None:
        start = sum(item.numel() for item in values); item = _token_dot(tensor, gradient, weights); values.append(item)
        layout.append({"cut": cut, "component": component, "start": start, "stop": start + item.numel(), "shape": list(item.shape)})
    tokens("post_patch_embedding_tokens", "tokens", capture.tensors["post_embedding"], cotangents["post_embedding"] )
    for block in range(BLOCKS):
        prefix = f"b{block}"
        tokens("block_input_tokens", prefix, capture.tensors[prefix + ".input"], cotangents[prefix + ".input"] )
        tokens("first_residual_addition", prefix + ".identity", capture.tensors[prefix + ".input"], cotangents[prefix + ".add1"] )
        head_gradient = cotangents[prefix + ".head_concat"].reshape(capture.heads[block].shape[0], capture.heads[block].shape[2], HEADS, -1).permute(0,2,1,3)
        for head in range(HEADS):
            tokens("first_residual_addition", prefix + f".head_{head}", capture.heads[block][:, head], head_gradient[:, head] )
        tokens("first_residual_addition", prefix + ".attention_aggregate", capture.tensors[prefix + ".head_concat"], cotangents[prefix + ".head_concat"] )
        tokens("second_residual_addition", prefix + ".post_attention_residual", capture.tensors[prefix + ".add1"], cotangents[prefix + ".output"] )
        tokens("second_residual_addition", prefix + ".mlp", capture.tensors[prefix + ".mlp_branch"], cotangents[prefix + ".output"] )
        tokens("block_output_tokens", prefix, capture.tensors[prefix + ".output"], cotangents[prefix + ".output"] )
    tokens("final_norm", "tokens", capture.tensors["final_norm"], cotangents["final_norm"] )
    start = sum(item.numel() for item in values)
    classifier = _total_dot(capture.tensors["classifier_input"], cotangents["classifier_input"], weights).reshape(1)
    values.append(classifier); layout.append({"cut": "classifier_input", "component": "cls", "start": start, "stop": start + 1, "shape": [1]})
    return torch.cat([item.reshape(-1) for item in values]), layout


def map_scalars(capture: ForwardCapture, cotangents: Mapping[str, torch.Tensor], weights: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    values: list[torch.Tensor] = []; labels: list[str] = []
    def add(label: str, tensor: torch.Tensor, gradient: torch.Tensor) -> None:
        labels.append(label); values.append(_total_dot(tensor, gradient, weights))
    add("post_embedding", capture.tensors["post_embedding"], cotangents["post_embedding"] )
    for block in range(BLOCKS):
        prefix = f"b{block}"
        add(prefix + ".input", capture.tensors[prefix + ".input"], cotangents[prefix + ".input"] )
        add(prefix + ".identity", capture.tensors[prefix + ".input"], cotangents[prefix + ".add1"] )
        head_gradient = cotangents[prefix + ".head_concat"].reshape(capture.heads[block].shape[0], capture.heads[block].shape[2], HEADS, -1).permute(0,2,1,3)
        for head in range(HEADS):
            add(prefix + f".head_{head}", capture.heads[block][:, head], head_gradient[:, head] )
        add(prefix + ".attention_aggregate", capture.tensors[prefix + ".head_concat"], cotangents[prefix + ".head_concat"] )
        add(prefix + ".post_attention_residual", capture.tensors[prefix + ".add1"], cotangents[prefix + ".output"] )
        add(prefix + ".mlp", capture.tensors[prefix + ".mlp_branch"], cotangents[prefix + ".output"] )
        add(prefix + ".output", capture.tensors[prefix + ".output"], cotangents[prefix + ".output"] )
    add("final_norm", capture.tensors["final_norm"], cotangents["final_norm"] )
    add("classifier_input", capture.tensors["classifier_input"], cotangents["classifier_input"] )
    return torch.stack(values), labels


def _patch_directions(difference: torch.Tensor, batch: int, patches: Sequence[int]) -> torch.Tensor:
    directions = torch.zeros((len(patches), batch, 3, 224, 224), device=difference.device, dtype=difference.dtype)
    for lane, patch in enumerate(patches):
        row, column = divmod(patch, 14)
        directions[lane, :, :, row * 16:(row + 1) * 16, column * 16:(column + 1) * 16] = difference[0, :, row * 16:(row + 1) * 16, column * 16:(column + 1) * 16]
    return directions


@dataclass
class IntegrationResult:
    attribution: torch.Tensor
    routes: torch.Tensor
    route_layout: list[dict[str, Any]]
    cell_maps: torch.Tensor
    map_labels: list[str]
    calls: dict[str, int]
    phase_telemetry: dict[str, Any]


def integrate_routes(
    model: nn.Module, image: torch.Tensor, baseline: torch.Tensor, target: int,
    steps: int, *, alpha_chunk: int = 4, source_patch_chunk: int = 4,
    component_vjp_chunk: int = 8,
) -> IntegrationResult:
    if image.shape != (1, 3, 224, 224) or baseline.shape != image.shape:
        raise StageYY1Error("Y1 integration shape drift")
    if alpha_chunk != 4 or source_patch_chunk <= 0 or component_vjp_chunk <= 0:
        raise StageYY1Error("Y1 frozen planner drift")
    nodes, quadrature_weights = gauss_legendre(steps, image.device, image.dtype)
    difference = image - baseline
    attribution = torch.zeros_like(image, dtype=torch.float64)
    routes: torch.Tensor | None = None; route_layout: list[dict[str, Any]] | None = None
    cell_maps: torch.Tensor | None = None; labels: list[str] | None = None
    calls = {"forward_calls": 0, "backward_calls": 0, "jvp_calls": 0, "jvp_direction_lanes": 0, "vjp_calls": 0, "quadrature_alpha_batches": 0}
    phase_telemetry: dict[str, Any] = {"forward_backward_wall_seconds": 0.0, "vjp_wall_seconds": 0.0, "jvp_wall_seconds": 0.0, "vjp_groups": [], "jvp_groups": []}
    keys = capture_keys()
    if BLOCKS == 12 and image.shape[-2:] == (224, 224):
        source_groups: list[tuple[int, ...]] = tuple((patch,) for patch in range(PATCHES))
    else:
        source_groups = [tuple(range(PATCHES))]
    for begin in range(0, steps, alpha_chunk):
        end = min(begin + alpha_chunk, steps); batch = end - begin
        scaled = (baseline.unsqueeze(0) + nodes[begin:end].reshape(-1, 1, 1, 1, 1) * difference.unsqueeze(0)).reshape(-1, 3, 224, 224).detach()
        scaled.requires_grad_(True)
        phase_started=time.monotonic();capture = native_forward_capture(model, scaled); calls["forward_calls"] += 1; calls["quadrature_alpha_batches"] += 1
        selected = capture.logits[:, target]
        tensors = [capture.tensors[key] for key in keys]
        gradients = torch.autograd.grad(selected.sum(), (scaled, *tensors), retain_graph=True)
        calls["backward_calls"] += 1
        phase_telemetry["forward_backward_wall_seconds"] += time.monotonic()-phase_started
        cotangents = {key: gradient.detach() for key, gradient in zip(keys, gradients[1:])}
        weights = quadrature_weights[begin:end].to(image.dtype)
        weighted_gradient = gradients[0].reshape(batch, 1, 3, 224, 224).double() * quadrature_weights[begin:end].reshape(batch, 1, 1, 1, 1)
        attribution.add_((weighted_gradient.sum(0) * difference.double()).detach())

        scalar_maps, current_labels = map_scalars(capture, cotangents, torch.ones_like(weights))
        if labels is None:
            labels = current_labels; cell_maps = torch.zeros((len(labels), CELLS, CELLS), dtype=torch.float64, device="cpu")
        elif labels != current_labels:
            raise StageYY1Error("map layout changed within integration")
        assert cell_maps is not None
        for start in range(0, scalar_maps.numel(), component_vjp_chunk):
            stop = min(start + component_vjp_chunk, scalar_maps.numel())
            group_started=time.monotonic()
            eye = torch.eye(stop - start, device=image.device, dtype=scalar_maps.dtype)
            derivative = torch.autograd.grad(scalar_maps[start:stop], scaled, grad_outputs=eye, is_grads_batched=True, retain_graph=True)[0]
            calls["vjp_calls"] += 1
            allocated = derivative.double() * difference.double().unsqueeze(0)
            allocated = allocated * quadrature_weights[begin:end].reshape(1,batch,1,1,1)
            pooled = allocated.sum(2).reshape(stop - start, batch, 56, 4, 56, 4).sum((1, 3, 5))
            cell_maps[start:stop].add_(pooled.detach().cpu())
            elapsed=time.monotonic()-group_started;phase_telemetry["vjp_wall_seconds"]+=elapsed;phase_telemetry["vjp_groups"].append({"alpha_begin":begin,"alpha_end":end,"component_start":start,"component_stop":stop,"component_labels":current_labels[start:stop],"wall_seconds":elapsed})

        def route_function(candidate: torch.Tensor) -> torch.Tensor:
            current = native_forward_capture(model, candidate)
            vector, _ = routing_scalars(current, cotangents, weights)
            return vector

        if route_layout is None:
            with torch.no_grad():
                _, route_layout = routing_scalars(capture, cotangents, weights)
                route_count = route_layout[-1]["stop"]
            routes = torch.zeros((PATCHES, route_count), dtype=torch.float64, device="cpu")
        assert routes is not None and route_layout is not None
        for start in range(0, len(source_groups), source_patch_chunk):
            groups = source_groups[start:start + source_patch_chunk]
            group_started=time.monotonic()
            directions = torch.stack([_patch_directions(difference, batch, group).sum(0) for group in groups])
            def one(direction: torch.Tensor) -> torch.Tensor:
                return torch.func.jvp(route_function, (scaled,), (direction,))[1]
            derivatives = torch.vmap(one, randomness="same")(directions)
            calls["jvp_calls"] += 1; calls["forward_calls"] += 1
            for local, group in enumerate(groups):
                routes[group[0]].add_(derivatives[local].detach().cpu().double())
            calls["jvp_direction_lanes"] += sum(len(group) for group in groups)
            elapsed=time.monotonic()-group_started;phase_telemetry["jvp_wall_seconds"]+=elapsed;phase_telemetry["jvp_groups"].append({"alpha_begin":begin,"alpha_end":end,"source_patch_groups":[list(group) for group in groups],"cut_ids":["post_patch_embedding_tokens","block_input_tokens","first_residual_addition","second_residual_addition","block_output_tokens","final_norm","classifier_input"],"wall_seconds":elapsed})
        del scaled, capture, selected, tensors, gradients, cotangents, scalar_maps
    assert routes is not None and route_layout is not None and cell_maps is not None and labels is not None
    return IntegrationResult(attribution[0].float().cpu(), routes.float(), route_layout, cell_maps.float(), labels, calls, phase_telemetry)


def _route_item(result: IntegrationResult, component: str, cut: str | None = None) -> torch.Tensor:
    matches = [row for row in result.route_layout if row["component"] == component and (cut is None or row["cut"] == cut)]
    if len(matches) != 1:
        raise StageYY1Error(f"route component is not unique: {component}")
    row = matches[0]
    return result.routes[:, row["start"]:row["stop"]].reshape(PATCHES, *row["shape"])


def _map_item(result: IntegrationResult, label: str) -> torch.Tensor:
    try:
        return result.cell_maps[result.map_labels.index(label)]
    except ValueError as error:
        raise StageYY1Error(f"missing cell component: {label}") from error


def materialize_arrays(result: IntegrationResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "input_attribution": result.attribution.numpy().astype("<f4", copy=False),
        "input_cell_map": pool_cells(result.attribution).numpy().astype("<f4", copy=False),
        "post_embedding_matrix": _route_item(result, "tokens", "post_patch_embedding_tokens")[:, 1:].numpy().astype("<f4", copy=False),
        "post_embedding_cls": _route_item(result, "tokens", "post_patch_embedding_tokens")[:, 0].numpy().astype("<f4", copy=False),
        "post_embedding_cell": _map_item(result, "post_embedding").numpy().astype("<f4", copy=False),
    }
    # The final global token component also uses the label "tokens"; resolve by cut.
    token_rows = [row for row in result.route_layout if row["component"] == "tokens"]
    post_row, final_row = token_rows
    final_route = result.routes[:, final_row["start"]:final_row["stop"]]
    arrays["final_norm_matrix"] = final_route[:, 1:].numpy().astype("<f4", copy=False)
    arrays["final_norm_cls"] = final_route[:, 0].numpy().astype("<f4", copy=False)
    arrays["final_norm_cell"] = _map_item(result, "final_norm").numpy().astype("<f4", copy=False)
    classifier = _route_item(result, "cls")[:, 0]
    arrays["classifier_input_cls"] = classifier.numpy().astype("<f4", copy=False)
    arrays["classifier_input_cell"] = _map_item(result, "classifier_input").numpy().astype("<f4", copy=False)
    for block in range(BLOCKS):
        prefix = f"b{block}"; out = f"b{block:02d}"
        for route_name, component in (("first_identity", prefix + ".identity"), ("first_attention", prefix + ".attention_aggregate"), ("second_residual", prefix + ".post_attention_residual"), ("second_mlp", prefix + ".mlp")):
            route = _route_item(result, component)
            arrays[f"{out}_{route_name}_matrix"] = route[:, 1:].numpy().astype("<f4", copy=False)
            arrays[f"{out}_{route_name}_cls"] = route[:, 0].numpy().astype("<f4", copy=False)
        # block input and output share a component name; resolve by cut.
        input_route = _route_item(result, prefix, "block_input_tokens")
        arrays[f"{out}_input_matrix"] = input_route[:, 1:].numpy().astype("<f4", copy=False)
        arrays[f"{out}_input_cls"] = input_route[:, 0].numpy().astype("<f4", copy=False)
        rows = [row for row in result.route_layout if row["component"] == prefix and row["cut"] == "block_output_tokens"]
        if len(rows) != 1:
            raise StageYY1Error("block output route layout drift")
        output_row = rows[0]
        output_route = result.routes[:, output_row["start"]:output_row["stop"]]
        arrays[f"{out}_output_matrix"] = output_route[:, 1:].numpy().astype("<f4", copy=False)
        arrays[f"{out}_output_cls"] = output_route[:, 0].numpy().astype("<f4", copy=False)
        head_routes = torch.stack([_route_item(result, prefix + f".head_{head}") for head in range(HEADS)])
        arrays[f"{out}_first_heads_matrix"] = head_routes[:, :, 1:].numpy().astype("<f4", copy=False)
        arrays[f"{out}_first_heads_cls"] = head_routes[:, :, 0].numpy().astype("<f4", copy=False)
        arrays[f"{out}_input_cell"] = _map_item(result, prefix + ".input").numpy().astype("<f4", copy=False)
        arrays[f"{out}_first_identity_cell"] = _map_item(result, prefix + ".identity").numpy().astype("<f4", copy=False)
        arrays[f"{out}_first_heads_cell"] = torch.stack([_map_item(result, prefix + f".head_{head}") for head in range(HEADS)]).numpy().astype("<f4", copy=False)
        arrays[f"{out}_first_attention_cell"] = _map_item(result, prefix + ".attention_aggregate").numpy().astype("<f4", copy=False)
        arrays[f"{out}_second_residual_cell"] = _map_item(result, prefix + ".post_attention_residual").numpy().astype("<f4", copy=False)
        arrays[f"{out}_second_mlp_cell"] = _map_item(result, prefix + ".mlp").numpy().astype("<f4", copy=False)
        arrays[f"{out}_output_cell"] = _map_item(result, prefix + ".output").numpy().astype("<f4", copy=False)
    for name, array in list(arrays.items()):
        array = np.ascontiguousarray(array, dtype="<f4")
        arrays[name] = array
        if array.dtype != np.dtype("float32") or not array.flags.c_contiguous or not np.isfinite(array).all():
            raise StageYY1Error(f"materialized array contract failed: {name}")
    return arrays


def validate_closure(arrays: Mapping[str, np.ndarray], delta: float) -> dict[str, Any]:
    input_attr = validate_scientific_array(arrays["input_attribution"], (3, 224, 224))
    input_cells = validate_scientific_array(arrays["input_cell_map"], (56, 56))
    input_patches = input_attr.astype(np.float64).sum(0).reshape(14, 16, 14, 16).sum((1, 3)).reshape(196)
    maximum = float(np.max(np.abs(input_attr)))
    pixel_threshold = CUT_PIXEL_ATOL + CUT_PIXEL_RTOL * maximum
    total_threshold = CUT_TOTAL_ATOL + CUT_TOTAL_RTOL * abs(delta)
    input_error = abs(float(input_attr.astype(np.float64).sum()) - delta)
    failures: list[str] = []; per_cut: dict[str, Any] = {}; head_max = 0.0; branch_max = 0.0; scalar_max = 0.0
    if input_error > INPUT_ATOL + INPUT_RTOL * abs(delta): failures.append("input_completeness")
    def check(name: str, matrix: np.ndarray, cls: np.ndarray, cells: np.ndarray) -> None:
        nonlocal scalar_max
        route = matrix.astype(np.float64).sum(1) + cls.astype(np.float64)
        patch_error = float(np.max(np.abs(route - input_patches)))
        cell_error = float(np.max(np.abs(cells.astype(np.float64) - input_cells.astype(np.float64))))
        total_error = abs(float(matrix.astype(np.float64).sum() + cls.astype(np.float64).sum()) - delta)
        scalar_error = abs(float(cells.astype(np.float64).sum()) - float(matrix.astype(np.float64).sum() + cls.astype(np.float64).sum()))
        scalar_max = max(scalar_max, scalar_error)
        passed = patch_error <= max(pixel_threshold * 16 * 16 * 3, total_threshold) and cell_error <= max(pixel_threshold * 16 * 3, total_threshold) and total_error <= total_threshold and scalar_error <= total_threshold
        per_cut[name] = {"patch_row_max_abs_error": patch_error, "cell_max_abs_error": cell_error, "total_abs_error": total_error, "scalar_map_matrix_abs_error": scalar_error, "pass": passed}
        if not passed: failures.append(name)
    check("post_patch_embedding_tokens", arrays["post_embedding_matrix"], arrays["post_embedding_cls"], arrays["post_embedding_cell"] )
    for block in range(BLOCKS):
        prefix = f"b{block:02d}"
        check(prefix + ".input", arrays[prefix + "_input_matrix"], arrays[prefix + "_input_cls"], arrays[prefix + "_input_cell"] )
        heads_matrix = arrays[prefix + "_first_heads_matrix"].astype(np.float64); heads_cls = arrays[prefix + "_first_heads_cls"].astype(np.float64); heads_cell = arrays[prefix + "_first_heads_cell"].astype(np.float64)
        attention_matrix = arrays[prefix + "_first_attention_matrix"].astype(np.float64); attention_cls = arrays[prefix + "_first_attention_cls"].astype(np.float64); attention_cell = arrays[prefix + "_first_attention_cell"].astype(np.float64)
        current_head = max(float(np.max(np.abs(heads_matrix.sum(0) - attention_matrix))), float(np.max(np.abs(heads_cls.sum(0) - attention_cls))), float(np.max(np.abs(heads_cell.sum(0) - attention_cell))))
        head_max = max(head_max, current_head)
        first_matrix = arrays[prefix + "_first_identity_matrix"].astype(np.float64) + attention_matrix
        first_cls = arrays[prefix + "_first_identity_cls"].astype(np.float64) + attention_cls
        first_cell = arrays[prefix + "_first_identity_cell"].astype(np.float64) + attention_cell
        check(prefix + ".first_residual", first_matrix.astype(np.float32), first_cls.astype(np.float32), first_cell.astype(np.float32))
        second_matrix = arrays[prefix + "_second_residual_matrix"].astype(np.float64) + arrays[prefix + "_second_mlp_matrix"].astype(np.float64)
        second_cls = arrays[prefix + "_second_residual_cls"].astype(np.float64) + arrays[prefix + "_second_mlp_cls"].astype(np.float64)
        second_cell = arrays[prefix + "_second_residual_cell"].astype(np.float64) + arrays[prefix + "_second_mlp_cell"].astype(np.float64)
        check(prefix + ".second_residual", second_matrix.astype(np.float32), second_cls.astype(np.float32), second_cell.astype(np.float32))
        check(prefix + ".output", arrays[prefix + "_output_matrix"], arrays[prefix + "_output_cls"], arrays[prefix + "_output_cell"] )
        branch_max = max(branch_max, float(np.max(np.abs(first_cell - input_cells))), float(np.max(np.abs(second_cell - input_cells))))
        if current_head > total_threshold: failures.append(prefix + ".head_sum")
    check("final_norm", arrays["final_norm_matrix"], arrays["final_norm_cls"], arrays["final_norm_cell"] )
    zero_matrix = np.zeros((196,196), dtype=np.float32)
    check("classifier_input", zero_matrix, arrays["classifier_input_cls"], arrays["classifier_input_cell"] )
    return {"status": "PASS" if not failures else "FAILED_NUMERICAL", "failures": failures, "input_abs_error": input_error, "input_threshold": INPUT_ATOL + INPUT_RTOL * abs(delta), "pixel_base_threshold": pixel_threshold, "total_threshold": total_threshold, "head_sum_max_abs_error": head_max, "branch_cell_max_abs_error": branch_max, "scalar_map_matrix_max_abs_error": scalar_max, "per_cut": per_cut}


@contextlib.contextmanager
def ablate_heads(model: nn.Module, block_index: int, heads: Iterable[int]):
    selected = tuple(sorted(set(int(head) for head in heads)))
    if block_index not in range(BLOCKS) or not selected or any(head not in range(HEADS) for head in selected):
        raise StageYY1Error("invalid head ablation coordinate")
    attention = model.blocks[block_index].attn
    def hook(_module: nn.Module, arguments: tuple[torch.Tensor, ...]):
        value = arguments[0].clone(); width = value.shape[-1] // attention.num_heads
        shaped = value.reshape(value.shape[0], value.shape[1], attention.num_heads, width)
        shaped[:, :, list(selected), :] = 0
        return (shaped.reshape_as(value),)
    handle = attention.proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def run_head_ablations(model: nn.Module, image: torch.Tensor, target: int, clean_logit: float, baseline_logit: float, head_maps: Mapping[int, np.ndarray], source_object_id: str) -> tuple[dict[str, Any], dict[str, int]]:
    denominator = max(abs(clean_logit - baseline_logit), 1e-6); blocks = []; forward_calls = 0
    with torch.no_grad():
        native = float(model(image)[0, target]); forward_calls += 1
        for block in range(BLOCKS):
            maps = validate_scientific_array(head_maps[block], (HEADS, 56, 56))
            allocation = np.abs(maps.astype(np.float64)).sum((1, 2))
            order = stable_descending(allocation, "STAGE_Y_HEAD_RANK_V1_20260910", source_object_id, block)
            losses = []
            for head in range(HEADS):
                with ablate_heads(model, block, [head]): score = float(model(image)[0, target])
                forward_calls += 1; losses.append({"head": head, "raw_logit_loss": native - score, "normalized_loss": (native - score) / denominator})
            curves = {"top": [], "bottom": []}
            for kind, ranking in (("top", order), ("bottom", order[::-1])):
                for count in (1,2,3,4,6,8,10,12):
                    chosen = [int(item) for item in ranking[:count]]
                    with ablate_heads(model, block, chosen): score = float(model(image)[0, target])
                    forward_calls += 1; curves[kind].append({"k": count, "heads": chosen, "raw_logit_loss": native - score, "normalized_loss": (native - score) / denominator})
            blocks.append({"block": block, "native_logit": native, "allocation_absolute_mass": allocation.tolist(), "rank_order": [int(item) for item in order], "single_head": losses, "curves": curves})
    return {"status": "PASS", "target": target, "normalization_denominator": denominator, "intervention_point": "native per-head value-mixed output before concat/output projection", "replacement": "zero same shape dtype", "blocks": blocks, "single_head_coordinates": BLOCKS * HEADS, "curve_points": BLOCKS * 2 * 8}, {"forward_calls": forward_calls}


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def hierarchical_order(cell_map: np.ndarray, source_object_id: str, block: int, control_seed: int | None = None) -> np.ndarray:
    cells = validate_scientific_array(cell_map, (56, 56)).reshape(14, 4, 14, 4).transpose(0,2,1,3).reshape(196,16)
    positive = np.maximum(cells.astype(np.float64), 0.0)
    patch_order = stable_descending(positive.sum(1), "STAGE_Y_PATCH_ORDER_V1_20260910", source_object_id, block)
    output = []
    for patch in patch_order:
        if control_seed is None:
            local = stable_descending(positive[patch], "STAGE_Y_CELL_ORDER_V1_20260910", source_object_id, block, int(patch))
        else:
            local = tie_permutation(source_object_id, block, int(patch), control_seed)
        patch_y, patch_x = divmod(int(patch), 14)
        for coordinate in local:
            cell_y, cell_x = divmod(int(coordinate), 4)
            output.append((patch_y * 4 + cell_y) * 56 + patch_x * 4 + cell_x)
    return np.asarray(output, dtype=np.int64)


def run_deletion_operator(model: nn.Module, image: torch.Tensor, target: int, clean_logit: float, baseline_logit: float, block_maps: Mapping[int, np.ndarray], source_object_id: str, batch_size: int = 64) -> tuple[dict[str, Any], dict[str, int]]:
    percentages = list(range(0, 101, 5)); counts = [_round_half_up(item * 3136 / 100) for item in percentages]
    denominator = max(abs(clean_logit - baseline_logit), 1e-6); rows = []; calls = 0
    def curve(order: np.ndarray) -> tuple[list[float], list[float]]:
        nonlocal calls
        batch = []
        for count in counts:
            mask = torch.ones((56 * 56,), device=image.device, dtype=image.dtype); mask[torch.as_tensor(order[:count], device=image.device)] = 0
            pixel = mask.reshape(56,56).repeat_interleave(4,0).repeat_interleave(4,1)
            batch.append(image * pixel.reshape(1,1,224,224))
        scores = []
        with torch.no_grad():
            for start in range(0, len(batch), batch_size):
                values = model(torch.cat(batch[start:start+batch_size], dim=0))[:, target]; calls += 1; scores.extend(float(item) for item in values)
        losses = [(clean_logit - score) / denominator for score in scores]
        auc = float(np.trapz(np.asarray(losses, dtype=np.float64), np.asarray(percentages, dtype=np.float64) / 100.0))
        return losses, [auc]
    for block in range(BLOCKS):
        value = validate_scientific_array(block_maps[block], (56,56))
        nns_losses, nns_auc = curve(hierarchical_order(value, source_object_id, block))
        controls = []
        for seed_index in range(32):
            losses, auc = curve(hierarchical_order(value, source_object_id, block, seed_index)); controls.append({"seed_index": seed_index, "losses": losses, "auc": auc[0]})
        median = float(np.median([row["auc"] for row in controls]))
        rows.append({"block": block, "percentiles": percentages, "selected_counts": counts, "nns_losses": nns_losses, "nns_auc": nns_auc[0], "controls": controls, "control_median_auc": median, "auc_gain": nns_auc[0] - median})
    return {"status": "PASS", "scientific_decision": None, "blocks": rows, "control_permutations_per_block": 32, "same_native_patch_order": True, "masking_baseline": "normalized_zero", "target": target}, {"forward_calls": calls}


def ceil_to(value: float, unit: float) -> float:
    return math.ceil((value - 1e-15) / unit) * unit


def project_budget(fixed: Mapping[str, float], repeats: Sequence[Mapping[str, Any]], retained_per_condition: int, fixed_bytes: int) -> dict[str, Any]:
    clean = [row for row in repeats if row["condition"] == "clean"]
    flip = [row for row in repeats if row["condition"] == "horizontal_flip"]
    if len(clean) != 2 or len(flip) != 2:
        raise StageYY1BudgetError("repeat measurements incomplete")
    metrics = ("gpu_active_seconds", "wall_seconds", "cpu_seconds")
    clean_unit = {key: max(float(row[key]) for row in clean) for key in metrics}
    flip_max = {key: max(float(row[key]) for row in flip) for key in metrics}
    increment = {key: max(0.0, flip_max[key] - clean_unit[key]) for key in metrics}
    def stage(units: int, perturbation: bool, storage_units: int) -> dict[str, Any]:
        raw = {}
        for key in metrics:
            unit = clean_unit[key] + (increment[key] if perturbation else 0.0)
            raw[key] = float(fixed.get(key, 0.0)) + units * unit
        proposed = {
            "h20_gpu_hours": ceil_to(1.5 * raw["gpu_active_seconds"] / 3600.0, 0.1),
            "wall_hours": ceil_to(1.5 * raw["wall_seconds"] / 3600.0, 1.0),
            "cpu_hours": ceil_to(1.5 * raw["cpu_seconds"] / 3600.0, 1.0),
            "storage_gib": ceil_to(1.5 * (fixed_bytes + storage_units * retained_per_condition) / 1024**3, 1.0),
        }
        return {"raw": raw, "proposed_caps": proposed}
    y2 = stage(100, False, 100); y3 = stage(20 * 8, True, 20 * 8)
    ceilings = {"y2": {"h20_gpu_hours":300.0,"wall_hours":168.0,"storage_gib":2048.0}, "y3": {"h20_gpu_hours":500.0,"wall_hours":240.0,"storage_gib":4096.0}}
    for name, value in (("y2", y2), ("y3", y3)):
        value["owner_ceiling"] = ceilings[name]
        value["status"] = "PASS" if all(value["proposed_caps"][key] <= ceiling for key, ceiling in ceilings[name].items()) else "BLOCKED_BUDGET"
    return {"safety_multiplier": 1.5, "fixed_cost": dict(fixed), "clean_unit_cost_maximum_rule": clean_unit, "maximum_horizontal_flip_repeat_cost": flip_max, "perturbation_incremental_cost": increment, "retained_bytes_per_condition": retained_per_condition, "fixed_bytes": fixed_bytes, "y2": y2, "y3": y3, "status": "PASS" if y2["status"] == y3["status"] == "PASS" else "BLOCKED_BUDGET"}
