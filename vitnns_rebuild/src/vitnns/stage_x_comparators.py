"""Auditable Stage X comparator adapters with frozen target semantics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import nn

from .stage_x_nns_v2 import (
    DisplayArray,
    NNSV2Result,
    ScientificArray,
    array_digest,
    gauss_legendre,
    input_ig,
)

PRIMARY = (
    "vit_grad_cam",
    "transformer_attribution_chefer",
    "attnlrp",
    "rise",
    "occlusion",
)
IDENTITY = "full_model_integrated_gradients"
ATTNLRP_CONV_GAMMAS = (0.1, 0.25, 100.0)
ATTNLRP_LINEAR_GAMMAS = (0.0, 0.01, 0.05, 0.1, 1.0)
ROOT = Path(__file__).resolve().parents[2]
DEPENDENCY_MANIFEST = ROOT / "third_party/stage_x/dependency_manifest.json"


class ComparatorError(ValueError):
    """A comparator target, source, forward, or output contract failed."""


class ComparatorSourceError(ComparatorError):
    status = "BLOCKED_COMPARATOR_SOURCE_IDENTITY"


class ComparatorForwardDriftError(ComparatorError):
    status = "NO_GO_METHOD_VALIDITY"


class ComparatorTargetDriftError(ComparatorError):
    status = "NO_GO_METHOD_VALIDITY"


class ComparatorGammaDriftError(ComparatorError):
    status = "NO_GO_METHOD_VALIDITY"


class _OfficialLayerNormRule(torch.autograd.Function):
    """Native LayerNorm forward with the official LXT identity-rule VJP."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, module: nn.LayerNorm) -> torch.Tensor:
        mean = value.mean(dim=-1, keepdim=True)
        variance = ((value - mean) ** 2).mean(dim=-1, keepdim=True)
        ctx.save_for_backward((variance + module.eps).sqrt(), module.weight)
        return nn.functional.layer_norm(
            value, module.normalized_shape, module.weight, module.bias, module.eps
        )

    @staticmethod
    def backward(ctx: Any, output_relevance: torch.Tensor) -> tuple[torch.Tensor, None]:
        standard_deviation, weight = ctx.saved_tensors
        scaled = output_relevance * weight / standard_deviation
        return scaled - scaled.mean(dim=-1, keepdim=True), None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest() -> Mapping[str, Any]:
    return json.loads(DEPENDENCY_MANIFEST.read_text(encoding="utf-8"))


def _manifest_record(relative: str) -> Mapping[str, Any]:
    matches = [row for row in _manifest()["files"] if row["path"] == relative]
    path = ROOT / relative
    if len(matches) != 1 or not path.is_file() or _sha(path) != matches[0]["sha256"]:
        raise ComparatorSourceError(f"vendored source identity drifted: {relative}")
    return matches[0]


def _local_source_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ComparatorSourceError(f"adapter source missing: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _validate_request(model: nn.Module, image: torch.Tensor, target_index: int) -> None:
    if image.ndim != 4 or image.shape[0] != 1 or not image.is_floating_point():
        raise ComparatorError("comparator requires one floating input image")
    if not bool(torch.isfinite(image).all()) or type(target_index) is not int or target_index < 0:
        raise ComparatorTargetDriftError("invalid comparator target request")
    if not isinstance(model, nn.Module):
        raise ComparatorError("model must be a torch module")


@contextmanager
def _frozen_model(model: nn.Module) -> Iterator[None]:
    states = [(module, module.training) for module in model.modules()]
    requires = [parameter.requires_grad for parameter in model.parameters()]
    try:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in zip(model.parameters(), requires):
            parameter.requires_grad_(state)
        for module, state in states:
            module.training = state


def _finite_scientific(value: torch.Tensor, semantics: str) -> torch.Tensor:
    result = value.detach().cpu().contiguous().float()
    ScientificArray(result, semantics)
    return result


def _quadrature_identity(steps: int, image: torch.Tensor) -> dict[str, Any]:
    nodes, weights = gauss_legendre(steps, torch.device("cpu"), torch.float64)
    return {
        "method": "gauss_legendre",
        "nodes_sha256": array_digest(nodes),
        "weights_sha256": array_digest(weights),
        "selected_steps": steps,
        "alpha_chunk": 4,
        "path": "straight",
        "baseline": "normalized_zero",
        "clean_input_sha256": array_digest(image),
    }


def full_model_ig_control(
    model: nn.Module,
    image: torch.Tensor,
    baseline: torch.Tensor,
    target_index: int,
    steps: int,
) -> dict[str, Any]:
    _validate_request(model, image, target_index)
    with _frozen_model(model):
        attribution = input_ig(model, image, baseline, target_index, steps, 4)
    return {
        "method_id": IDENTITY,
        "role": "INCLUDED_PRIMARY_IDENTITY_CONTROL",
        "target_index": target_index,
        "attribution": _finite_scientific(attribution, "signed_total_input_attribution"),
        "signed_rgb_map": _finite_scientific(attribution.sum(0), "signed_rgb_reduction"),
        "steps": steps,
        "quadrature": _quadrature_identity(steps, image),
        "baseline_sha256": array_digest(baseline),
        "superiority_test_allowed": False,
    }


def validate_identity(nns_result: NNSV2Result, ig_result: Mapping[str, Any]) -> dict[str, Any]:
    if ig_result.get("method_id", IDENTITY) != IDENTITY or ig_result.get("superiority_test_allowed", False):
        raise ComparatorTargetDriftError("IG role drifted from identity control")
    if nns_result.target.predicted_class != ig_result.get("target_index"):
        raise ComparatorTargetDriftError("identity-control target differs")
    if nns_result.steps != ig_result.get("steps"):
        raise ComparatorTargetDriftError("identity-control selected steps differ")
    other = ig_result.get("attribution")
    if not isinstance(other, torch.Tensor) or other.shape != nns_result.attribution.shape:
        raise ComparatorError("identity-control native shape differs")
    if not torch.allclose(other, nns_result.attribution, atol=1e-6, rtol=1e-6):
        raise ComparatorForwardDriftError("NNS-v2 total differs from IG identity control")
    return {
        "status": "PASS",
        "method_id": IDENTITY,
        "role": "INCLUDED_PRIMARY_IDENTITY_CONTROL",
        "atol": 1e-6,
        "rtol": 1e-6,
        "same_steps": True,
        "attribution_digest": array_digest(other),
        "superiority_test_allowed": False,
    }


def vit_grad_cam(model: nn.Module, image: torch.Tensor, target_index: int) -> dict[str, Any]:
    """ViT Grad-CAM on last-block norm1, excluding prefix tokens."""
    _validate_request(model, image, target_index)
    source_path = Path(importlib.util.find_spec("pytorch_grad_cam").origin).parent / "grad_cam.py"
    with _frozen_model(model):
        probe = image.detach().clone().requires_grad_(True)
        activation: torch.Tensor | None = None

        def capture(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            nonlocal activation
            activation = output

        handle = model.blocks[-1].norm1.register_forward_hook(capture)
        try:
            logits = model(probe)
            if target_index >= logits.shape[1]:
                raise ComparatorTargetDriftError("Grad-CAM target outside logits")
            assert activation is not None
            gradient = torch.autograd.grad(logits[:, target_index].sum(), activation)[0]
        finally:
            handle.remove()
        prefix = int(model.num_prefix_tokens)
        patch_activation = activation[:, prefix:]
        patch_gradient = gradient[:, prefix:]
        weights = patch_gradient.mean(dim=1)
        values = torch.relu((patch_activation * weights[:, None, :]).sum(dim=-1))[0]
    side = int(values.numel() ** 0.5)
    if side * side != values.numel():
        raise ComparatorError("Grad-CAM patch count is not square")
    return {
        "method_id": "vit_grad_cam",
        "class_targeted": True,
        "attribution": _finite_scientific(values.reshape(side, side), "native_patch_grid"),
        "target_index": target_index,
        "target_semantics": "unmodified_selected_raw_class_logit",
        "native_resolution": [side, side],
        "layer": f"blocks[{len(model.blocks) - 1}].norm1",
        "prefix_tokens_excluded": prefix,
        "minmax_normalization": False,
        "official_commit": "704393448a7b0c620ee2c2b9597723f1d7f17b3d",
        "official_source_file": "pytorch_grad_cam/grad_cam.py",
        "official_source_http_etag_sha256": "fab9d727f9d62cfc86365afee291a5cebfad84dc62218a933e3b9da3297ada0a",
        "license": "MIT",
        "source_file": _local_source_record(source_path),
    }


def chefer(model: nn.Module, image: torch.Tensor, target_index: int) -> dict[str, Any]:
    """Official Transformer Attribution attention-gradient rollout equation."""
    _validate_request(model, image, target_index)
    source = _manifest_record(
        "third_party/stage_x/transformer_explainability/baselines/ViT/ViT_LRP.py"
    )
    with _frozen_model(model):
        probe = image.detach().clone().requires_grad_(True)
        attentions: list[torch.Tensor] = []
        handles = [
            block.attn.attn_drop.register_forward_hook(
                lambda _module, _args, output: attentions.append(output)
            )
            for block in model.blocks
        ]
        try:
            logits = model(probe)
            if target_index >= logits.shape[1]:
                raise ComparatorTargetDriftError("Chefer target outside logits")
            if len(attentions) != len(model.blocks):
                raise ComparatorError("Chefer requires native unfused attention")
            gradients = torch.autograd.grad(logits[:, target_index].sum(), attentions)
        finally:
            for handle in handles:
                handle.remove()
    matrices = []
    for attention, gradient in zip(attentions, gradients):
        matrix = (attention * gradient).clamp_min(0).mean(dim=1)
        identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
        matrix = matrix + identity.unsqueeze(0)
        matrices.append(matrix / matrix.sum(dim=-1, keepdim=True))
    rollout = matrices[0]
    for matrix in matrices[1:]:
        rollout = matrix.bmm(rollout)
    values = rollout[0, 0, int(model.num_prefix_tokens):]
    side = int(values.numel() ** 0.5)
    if side * side != values.numel():
        raise ComparatorError("Chefer patch count is not square")
    return {
        "method_id": "transformer_attribution_chefer",
        "class_targeted": True,
        "attribution": _finite_scientific(values.reshape(side, side), "native_patch_grid"),
        "target_index": target_index,
        "target_semantics": "unmodified_selected_raw_class_logit",
        "native_resolution": [side, side],
        "layer_choice": "all_layers_start_0",
        "algorithm": "ReLU(attention_times_class_gradient), head_mean, identity_added_row_normalized_rollout",
        "official_source_file": source["path"],
        "official_source_sha256": source["sha256"],
        "official_commit": source["source_commit"],
        "license": "MIT",
    }


def rise(
    model: nn.Module,
    image: torch.Tensor,
    target_index: int,
    *,
    masks: int = 8000,
    grid: int = 7,
    p: float = 0.5,
    batch_size: int = 64,
    namespace: str = "STAGE_X_RISE_V1|",
) -> dict[str, Any]:
    """Official 7x7 Bernoulli masks, bilinear resize, and random shifts."""
    from skimage.transform import resize

    _validate_request(model, image, target_index)
    if masks <= 0 or grid <= 0 or not 0 < p < 1 or batch_size <= 0:
        raise ComparatorError("invalid RISE parameters")
    seed = int.from_bytes(
        hashlib.sha256(f"{namespace}{target_index}".encode()).digest()[:8], "big"
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    height, width = image.shape[-2:]
    cell = np.ceil(np.asarray([height, width]) / grid).astype(int)
    upsampled = (grid + 1) * cell
    low = rng.random((masks, grid, grid)) < p
    total = torch.zeros((height, width), device=image.device, dtype=torch.float64)
    with _frozen_model(model), torch.no_grad():
        for begin in range(0, masks, batch_size):
            stop = min(begin + batch_size, masks)
            current_np = np.empty((stop - begin, height, width), dtype=np.float32)
            for local, mask_index in enumerate(range(begin, stop)):
                expanded = resize(
                    low[mask_index].astype(np.float32), tuple(upsampled),
                    order=1, mode="reflect", anti_aliasing=False, preserve_range=True,
                )
                row = int(rng.integers(0, cell[0]))
                column = int(rng.integers(0, cell[1]))
                current_np[local] = expanded[row:row + height, column:column + width]
            current = torch.from_numpy(current_np[:, None]).to(image.device, image.dtype)
            logits = model(image * current)
            if target_index >= logits.shape[1]:
                raise ComparatorTargetDriftError("RISE target outside logits")
            scores = logits[:, target_index].double()
            total.add_((scores[:, None, None] * current[:, 0].double()).sum(0))
    return {
        "method_id": "rise",
        "class_targeted": True,
        "attribution": _finite_scientific(total / (masks * p), "input_resolution_mask_expectation"),
        "target_index": target_index,
        "target_semantics": "unmodified_selected_raw_class_logit",
        "native_resolution": [height, width],
        "grid": grid,
        "masks": masks,
        "p": p,
        "resize": "bilinear",
        "random_shift": True,
        "seed": seed,
        "official_commit": "d91ea006d4bb9b7990347fe97086bdc0f5c1fe10",
        "official_source_file": "RISE/explanations.py at pinned commit",
        "official_source_http_etag_sha256": "f9f2fef6995ed6138bce89d0cc8f83c695b21c51747b2320cedb0f7715e33928",
        "license": "MIT",
        "layer_choice": "none",
    }


def occlusion(
    model: nn.Module,
    image: torch.Tensor,
    target_index: int,
    *,
    window: int = 16,
    stride: int = 16,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Non-overlapping normalized-zero occlusion using raw-logit differences."""
    _validate_request(model, image, target_index)
    height, width = image.shape[-2:]
    if window <= 0 or stride <= 0 or height % stride or width % stride:
        raise ComparatorError("invalid occlusion geometry")
    coordinates = [(row, column) for row in range(0, height, stride) for column in range(0, width, stride)]
    values: list[torch.Tensor] = []
    with _frozen_model(model), torch.no_grad():
        clean_logits = model(image)
        if target_index >= clean_logits.shape[1]:
            raise ComparatorTargetDriftError("occlusion target outside logits")
        clean = clean_logits[0, target_index]
        for begin in range(0, len(coordinates), batch_size):
            current_coordinates = coordinates[begin:begin + batch_size]
            changed = image.expand(len(current_coordinates), -1, -1, -1).clone()
            for index, (row, column) in enumerate(current_coordinates):
                changed[index, :, row:row + window, column:column + window] = 0
            values.extend(clean - model(changed)[:, target_index])
    rows, columns = height // stride, width // stride
    map_value = torch.stack(values).reshape(rows, columns)
    return {
        "method_id": "occlusion",
        "class_targeted": True,
        "attribution": _finite_scientific(map_value, "method_defined_window_support"),
        "target_index": target_index,
        "target_semantics": "clean_minus_occluded_raw_class_logit",
        "native_resolution": [rows, columns],
        "window": window,
        "stride": stride,
        "replacement": "normalized_zero",
        "official_commit": "413e888e1e8f22cf8da1631cc9ca2133ce30a9db",
        "official_source_file": "captum.attr.Occlusion at pinned commit",
        "official_source_http_etag_sha256": "9f1f9d2f98c2976bab5104bcad17ca9dbc0831a1fc93abf0a6687489ae991824",
        "license": "BSD-3-Clause",
        "layer_choice": "none",
    }


@contextmanager
def _lxt_modules() -> Iterator[tuple[Any, Mapping[str, Any]]]:
    """Import retained LXT code without modifying its bytes or site packages."""
    rules_relative = "third_party/stage_x/attnlrp_lxt/lxt/efficient/rules.py"
    rules_record = _manifest_record(rules_relative)
    try:
        rules_spec = importlib.util.spec_from_file_location("stage_x_lxt_rules", ROOT / rules_relative)
        assert rules_spec and rules_spec.loader
        rules = importlib.util.module_from_spec(rules_spec)
        rules_spec.loader.exec_module(rules)
        yield rules, rules_record
    finally:
        sys.modules.pop("stage_x_lxt_rules", None)


@contextmanager
def _official_zennit_gradient_input_patch() -> Iterator[None]:
    """Apply and then restore the vendored LXT Zennit patch semantics."""
    from zennit.core import BasicHook, ParamMod, stabilize

    original_forward, original_backward = BasicHook.forward, BasicHook.backward

    def forward(self: Any, module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        self.stored_tensors["input"] = inputs
        self.stored_tensors["output"] = output

    def backward(self: Any, module: nn.Module, grad_input: Any, grad_output: Any) -> Any:
        if len(grad_output) != 1:
            raise ComparatorError("official LXT Zennit patch supports one output")
        output_relevance = grad_output[0] * self.stored_tensors["output"]
        output_relevance.requires_grad = True
        original_input = self.stored_tensors["input"][0].clone()
        inputs, outputs = [], []
        for input_modifier, parameter_modifier, output_modifier in zip(
            self.input_modifiers, self.param_modifiers, self.output_modifiers
        ):
            current_input = input_modifier(original_input).requires_grad_()
            with ParamMod.ensure(parameter_modifier)(module), torch.autograd.enable_grad():
                current_output = output_modifier(module.forward(current_input))
            inputs.append(current_input)
            outputs.append(current_output)
        gradients = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=self.gradient_mapper(output_relevance, outputs),
            create_graph=output_relevance.requires_grad,
        )
        relevance = self.reducer(inputs, gradients)
        relevance = relevance / stabilize(original_input, epsilon=1e-10)
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)

    BasicHook.forward, BasicHook.backward = forward, backward
    try:
        yield
    finally:
        BasicHook.forward, BasicHook.backward = original_forward, original_backward


@contextmanager
def _attnlrp_timm_rules(model: nn.Module, rules: Any) -> Iterator[None]:
    """Adapt official LXT ViT rules to equivalent timm operation surfaces."""
    originals: list[tuple[nn.Module, Any]] = []

    def replace(module: nn.Module, function: Any) -> None:
        originals.append((module, module.__dict__.get("forward")))
        module.forward = MethodType(function, module)

    def norm(module: nn.LayerNorm, value: torch.Tensor) -> torch.Tensor:
        return _OfficialLayerNormRule.apply(value, module)

    def gelu(module: nn.GELU, value: torch.Tensor) -> torch.Tensor:
        return rules.identity_rule_implicit(
            lambda tensor: nn.functional.gelu(tensor, approximate=module.approximate), value
        )

    def cp_attention(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = value.shape
        qkv = module.qkv(value).reshape(
            batch, tokens, 3, module.num_heads, module.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, val = qkv.unbind(0)
        query, key = module.q_norm(query), module.k_norm(key)
        # Official torchvision ViT recipe uses CP-LRP inside attention: Q and K
        # are detached, while the value path remains relevant.
        query = rules.stop_gradient(query)
        key = rules.stop_gradient(key)
        attention = ((query * module.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
        attention = module.attn_drop(attention)
        output = (attention @ val).transpose(1, 2).reshape(batch, tokens, channels)
        return module.proj_drop(module.proj(output))

    try:
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                replace(module, norm)
            elif isinstance(module, nn.GELU):
                replace(module, gelu)
        for block in model.blocks:
            replace(block.attn, cp_attention)
        yield
    finally:
        for module, original in reversed(originals):
            if original is None:
                module.__dict__.pop("forward", None)
            else:
                module.forward = original


def _validate_gamma(conv_gamma: float, linear_gamma: float) -> None:
    if conv_gamma not in ATTNLRP_CONV_GAMMAS or linear_gamma not in ATTNLRP_LINEAR_GAMMAS:
        raise ComparatorGammaDriftError("AttnLRP gamma is outside the frozen X1 grid")


def attnlrp_official_timm(
    model: nn.Module,
    image: torch.Tensor,
    target_index: int,
    *,
    conv_gamma: float,
    linear_gamma: float,
) -> dict[str, Any]:
    """Official ViT recipe: AttnLRP outside attention plus CP-LRP inside."""
    _validate_request(model, image, target_index)
    _validate_gamma(conv_gamma, linear_gamma)
    example = _manifest_record("third_party/stage_x/attnlrp_lxt/examples/vit_torch.py")
    vit_rules = _manifest_record("third_party/stage_x/attnlrp_lxt/lxt/efficient/models/vit_torch.py")
    patches = _manifest_record("third_party/stage_x/attnlrp_lxt/lxt/efficient/patches.py")
    zennit_patch = _manifest_record("third_party/stage_x/attnlrp_lxt/lxt/efficient/zennit_patches.py")
    if importlib.metadata.version("zennit") != "0.5.1":
        raise ComparatorSourceError("Zennit version drifted")
    from zennit.composites import LayerMapComposite
    from zennit.rules import Gamma

    with _frozen_model(model), _lxt_modules() as (rules, rules_record):
        with torch.no_grad():
            native = model(image).detach()
        with _attnlrp_timm_rules(model, rules):
            probe = image.detach().clone().requires_grad_(True)
            composite = LayerMapComposite([
                (nn.Conv2d, Gamma(conv_gamma)),
                (nn.Linear, Gamma(linear_gamma)),
            ])
            with _official_zennit_gradient_input_patch(), composite.context(model):
                patched = model(probe)
                if target_index >= patched.shape[1]:
                    raise ComparatorTargetDriftError("AttnLRP target outside logits")
                forward_error = float((native - patched.detach()).abs().max())
                if forward_error >= 1e-6:
                    raise ComparatorForwardDriftError("AttnLRP forward logits drifted")
                gradient = torch.autograd.grad(patched[:, target_index].sum(), probe)[0]
        relevance = (probe * gradient).sum(dim=1)[0]
    return {
        "method_id": "attnlrp",
        "exact_algorithm_name": "AttnLRP outside attention + CP-LRP inside attention",
        "class_targeted": True,
        "attribution": _finite_scientific(relevance, "signed_input_pixel_relevance"),
        "target_index": target_index,
        "target_semantics": "selected_raw_class_logit_relevance",
        "native_resolution": list(relevance.shape),
        "layer_choice": "full_model",
        "forward_parity_max_abs": forward_error,
        "forward_parity_threshold_lt": 1e-6,
        "conv_gamma": conv_gamma,
        "linear_gamma": linear_gamma,
        "gamma_selected": False,
        "gamma_selection_stage": "X2",
        "layernorm_rule": "official_identity_rule",
        "gelu_rule": "official_identity_rule_implicit",
        "attention_rule": "official_cp_lrp_stop_qk_inside_attention",
        "linear_conv_rule": "zennit_gamma",
        "custom_backward": True,
        "ordinary_input_times_gradient": False,
        "official_commit": rules_record["source_commit"],
        "license": "Clear-BSD with scientific-use patent limitation",
        "zennit_version": "0.5.1",
        "official_sources": [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in (example, vit_rules, patches, zennit_patch, rules_record)
        ],
    }


def attnlrp(
    model: nn.Module,
    image: torch.Tensor,
    target_index: int,
    *,
    conv_gamma: float,
    linear_gamma: float,
) -> dict[str, Any]:
    return attnlrp_official_timm(
        model, image, target_index, conv_gamma=conv_gamma, linear_gamma=linear_gamma
    )


attnlrp_zennit = attnlrp_official_timm


def validate_attnlrp_grid(
    model: nn.Module, image: torch.Tensor, target_index: int
) -> list[dict[str, Any]]:
    results = []
    for conv_gamma in ATTNLRP_CONV_GAMMAS:
        for linear_gamma in ATTNLRP_LINEAR_GAMMAS:
            result = attnlrp_official_timm(
                model, image, target_index,
                conv_gamma=conv_gamma, linear_gamma=linear_gamma,
            )
            results.append({
                "conv_gamma": conv_gamma,
                "linear_gamma": linear_gamma,
                "forward_parity_max_abs": result["forward_parity_max_abs"],
                "compatible": True,
                "selected": False,
            })
    return results


def attention_rollout(model: nn.Module, image: torch.Tensor) -> dict[str, Any]:
    """Class-agnostic mean-head attention rollout on the native unfused path."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ComparatorError("attention rollout requires one image")
    attentions: list[torch.Tensor] = []
    handles = [block.attn.attn_drop.register_forward_hook(
        lambda _module, _args, output: attentions.append(output)
    ) for block in model.blocks]
    try:
        with _frozen_model(model), torch.no_grad(): model(image)
    finally:
        for handle in handles: handle.remove()
    if len(attentions) != len(model.blocks):
        raise ComparatorError("attention rollout requires all native attentions")
    rollout = None
    for attention in attentions:
        matrix = attention.mean(1)
        identity = torch.eye(matrix.shape[-1],device=matrix.device,dtype=matrix.dtype)
        matrix = matrix + identity.unsqueeze(0); matrix = matrix / matrix.sum(-1,keepdim=True)
        rollout = matrix if rollout is None else matrix.bmm(rollout)
    assert rollout is not None
    values=rollout[0,0,int(model.num_prefix_tokens):]; side=int(values.numel()**0.5)
    if side*side!=values.numel(): raise ComparatorError("rollout patch count is not square")
    return {"method_id":"attention_rollout","attribution":_finite_scientific(values.reshape(side,side),"native_patch_grid"),"class_targeted":False,"target_index":None,"native_resolution":[side,side]}


def raw_cls_attention(model: nn.Module, image: torch.Tensor) -> dict[str, Any]:
    """Class-agnostic last-block mean-head CLS attention."""
    attention: torch.Tensor | None = None
    def capture(_module: nn.Module,_args: tuple[Any,...],output: torch.Tensor) -> None:
        nonlocal attention; attention=output
    handle=model.blocks[-1].attn.attn_drop.register_forward_hook(capture)
    try:
        with _frozen_model(model),torch.no_grad(): model(image)
    finally: handle.remove()
    if attention is None: raise ComparatorError("raw CLS attention was not captured")
    values=attention[0,:,0,int(model.num_prefix_tokens):].mean(0);side=int(values.numel()**0.5)
    if side*side!=values.numel(): raise ComparatorError("raw CLS patch count is not square")
    return {"method_id":"raw_cls_attention","attribution":_finite_scientific(values.reshape(side,side),"native_patch_grid"),"class_targeted":False,"target_index":None,"native_resolution":[side,side]}


def deterministic_random_map(height: int, width: int, identity: str) -> dict[str, Any]:
    seed=int.from_bytes(hashlib.sha256(("STAGE_X_RANDOM_MAP_V1|"+identity).encode()).digest()[:8],"big")
    value=torch.from_numpy(np.random.Generator(np.random.PCG64(seed)).random((height,width),dtype=np.float32))
    return {"method_id":"random_map","attribution":_finite_scientific(value,"input_resolution_seeded_random"),"class_targeted":False,"target_index":None,"native_resolution":[height,width],"seed":seed}


def center_prior(height: int, width: int) -> dict[str, Any]:
    rows=torch.arange(height,dtype=torch.float32)-(height-1)/2; cols=torch.arange(width,dtype=torch.float32)-(width-1)/2
    value=-torch.sqrt(rows[:,None].square()+cols[None,:].square())
    return {"method_id":"center_prior","attribution":_finite_scientific(value,"input_resolution_negative_center_distance"),"class_targeted":False,"target_index":None,"native_resolution":[height,width],"ties":"row_major"}


def edge_input_gradient_prior(model: nn.Module,image: torch.Tensor) -> dict[str,Any]:
    """Return separately identified Sobel edge and class-agnostic logit-sum gradient priors."""
    probe=image.detach().clone().requires_grad_(True)
    with _frozen_model(model): gradient=torch.autograd.grad(model(probe).sum(),probe)[0]
    gray=image.detach().mean(1,keepdim=True)
    kx=torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],device=image.device,dtype=image.dtype).reshape(1,1,3,3)
    ky=kx.transpose(-1,-2)
    edge=torch.sqrt(nn.functional.conv2d(gray,kx,padding=1).square()+nn.functional.conv2d(gray,ky,padding=1).square())[0,0]
    input_gradient=gradient.abs().sum(1)[0]
    return {"method_id":"edge_input_gradient_prior","edge_attribution":_finite_scientific(edge,"sobel_rgb_mean_edge_magnitude"),"input_gradient_attribution":_finite_scientific(input_gradient,"class_agnostic_logit_sum_gradient_magnitude"),"class_targeted":False,"target_index":None,"native_resolution":list(edge.shape),"reported_separately":True}


def architecture_compatibility(model: nn.Module, model_id: str) -> dict[str, Any]:
    blocks = getattr(model, "blocks", None)
    patch = getattr(model, "patch_embed", None)
    if blocks is None or len(blocks) != 12 or getattr(model, "num_prefix_tokens", 0) != 1 or patch is None:
        raise ComparatorError("architecture unsupported")
    if not all(hasattr(block.attn, "attn_drop") for block in blocks):
        raise ComparatorError("unfused attention hook surface missing")
    return {
        "model_id": model_id,
        "status": "PASS",
        "competitive_methods": list(PRIMARY),
        "identity_control": IDENTITY,
        "holm_family": list(PRIMARY),
        "depth": len(blocks),
        "patch_grid": list(patch.grid_size),
        "downloads_performed": False,
    }
