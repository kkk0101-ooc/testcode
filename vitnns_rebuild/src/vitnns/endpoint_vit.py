"""Original-input endpoint learning images from the native timm ViT graph.

Each scalar is differentiated through its full upstream graph at the observed
input. Hooks only observe tensors. No baseline forward or integration is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor, nn

from .endpoint_nns import _frozen_eval


@dataclass(frozen=True)
class EndpointTarget:
    name: str
    module: str
    kind: str = "patch_channel_sum"
    location: str = "output"
    channel: int = 0
    token: int = 0
    head: int | None = None
    class_id: int | None = None
    other_class_id: int | None = None


@dataclass(frozen=True)
class EndpointUnit:
    target: EndpointTarget
    value: Tensor
    gradient: Tensor
    attribution: Tensor
    token_values: Tensor | None
    attribution_sum: Tensor
    local_affine_intercept: Tensor


@dataclass(frozen=True)
class EndpointScan:
    logits: Tensor
    units: dict[str, EndpointUnit]
    forward_count: int


@dataclass(frozen=True)
class EndpointDecomposition:
    target: EndpointTarget
    value: Tensor
    full: Tensor
    components: dict[str, Tensor]
    cuts: dict[str, tuple[str, ...]]
    closure_max_abs: dict[str, float]
    closure_relative_l2: dict[str, float]


def _select(
    model: nn.Module, spec: EndpointTarget, value: Tensor,
) -> tuple[EndpointTarget, Tensor, Tensor | None]:
    if spec.kind in {"logit", "margin"}:
        if spec.module or spec.location != "output" or value.ndim != 2:
            raise ValueError("logit/margin targets must use the native model output")
        class_id = int(value[0].detach().argmax()) if spec.class_id is None else spec.class_id
        if not 0 <= class_id < value.shape[1]:
            raise ValueError("class_id out of range")
        spec = replace(spec, class_id=class_id)
        score = value[:, class_id]
        if spec.kind == "margin":
            if value.shape[1] < 2:
                raise ValueError("a margin requires two classes")
            other = spec.other_class_id
            if other is None:
                candidates = value[0].detach().clone()
                candidates[class_id] = -torch.inf
                other = int(candidates.argmax())
            if not 0 <= other < value.shape[1] or other == class_id:
                raise ValueError("margin classes must be different valid indices")
            spec = replace(spec, other_class_id=other)
            score = score - value[:, other]
        return spec, score, None

    if value.ndim != 3:
        raise ValueError(f"{spec.name}: expected [batch,tokens,channels]")
    channel = spec.channel
    if spec.kind == "head_channel":
        if not spec.module.endswith(".attn.proj") or spec.location != "input":
            raise ValueError("head targets must observe attn.proj input")
        attention = model.get_submodule(spec.module.rsplit(".", 1)[0])
        heads = attention.num_heads
        if value.shape[-1] % heads:
            raise ValueError("head dimensions do not divide the projection input")
        width = value.shape[-1] // heads
        if spec.head is None or not 0 <= spec.head < heads or not 0 <= channel < width:
            raise ValueError("head or within-head channel out of range")
        channel += spec.head * width
    elif spec.head is not None:
        raise ValueError("head is only valid for head_channel targets")
    if not 0 <= channel < value.shape[-1]:
        raise ValueError("channel out of range")
    token_values = value[:, :, channel]
    if spec.kind == "patch_channel_sum":
        prefix = 0 if spec.module == "patch_embed" else model.num_prefix_tokens
        if prefix >= value.shape[1]:
            raise ValueError("target has no patch tokens")
        score = token_values[:, prefix:].sum(1)
    elif spec.kind in {"token_channel", "head_channel"}:
        if not 0 <= spec.token < value.shape[1]:
            raise ValueError("token out of range")
        score = token_values[:, spec.token]
    else:
        raise ValueError(f"unknown target kind: {spec.kind}")
    return spec, score, token_values


def scan_endpoint_vit(
    model: nn.Module, inputs: Tensor, targets: Sequence[EndpointTarget],
) -> EndpointScan:
    """Scan one image with one native forward and one VJP per scalar target.

    ``inputs`` is the actual model-input tensor, including its normalization.
    All returned tensors are detached CPU tensors. Signed RGB attribution is
    preserved as [1,3,H,W]. A head target is an internal O_h coordinate before
    projection; a class target also includes projection and all later layers.
    The local affine intercept is a Taylor diagnostic, not a bias allocation.
    """
    if inputs.ndim != 4 or inputs.shape[0] != 1 or not inputs.is_floating_point():
        raise ValueError("scan expects one floating-point image [1,C,H,W]")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("input contains nonfinite values")
    targets = tuple(targets)
    if not targets or len({t.name for t in targets}) != len(targets):
        raise ValueError("targets must have unique nonempty names")
    if any(not t.name for t in targets):
        raise ValueError("target name is empty")
    if getattr(model, "grad_checkpointing", False):
        raise ValueError("disable gradient checkpointing for native tensor capture")
    modules = {}
    for spec in targets:
        if spec.location not in {"input", "output"}:
            raise ValueError("location must be input or output")
        if spec.module:
            modules[(spec.module, spec.location)] = model.get_submodule(spec.module)
        elif spec.kind not in {"logit", "margin"}:
            raise ValueError("only logit/margin targets use the model output")
    captured: dict[tuple[str, str], Tensor] = {}
    handles = []

    def save(key: tuple[str, str], value: Tensor) -> None:
        if key in captured:
            raise ValueError(f"module executed more than once: {key}")
        if not isinstance(value, Tensor):
            raise TypeError(f"hook target is not a tensor: {key}")
        captured[key] = value

    clean = inputs.detach().clone().requires_grad_(True)
    try:
        for key, module in modules.items():
            if key[1] == "input":
                def pre_hook(_module, args, key=key):
                    save(key, args[0])
                handles.append(module.register_forward_pre_hook(pre_hook))
            else:
                def post_hook(_module, _args, output, key=key):
                    save(key, output)
                handles.append(module.register_forward_hook(post_hook))
        with _frozen_eval(model), torch.enable_grad():
            logits = model(clean)
            if not isinstance(logits, Tensor) or logits.ndim != 2:
                raise ValueError("expected native classifier logits [1,classes]")
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite model logits")
            selected = []
            for spec in targets:
                value = captured[(spec.module, spec.location)] if spec.module else logits
                selected.append(_select(model, spec, value))
            units = {}
            for index, (spec, score, token_values) in enumerate(selected):
                if not bool(torch.isfinite(score).all()):
                    raise ValueError(f"nonfinite target: {spec.name}")
                gradient = torch.autograd.grad(
                    score.sum(), clean, retain_graph=index + 1 < len(selected),
                )[0]
                attribution = clean.detach() * gradient
                if not bool(torch.isfinite(attribution).all()):
                    raise ValueError(f"nonfinite learning image: {spec.name}")
                total = attribution.flatten(1).sum(1)
                units[spec.name] = EndpointUnit(
                    target=spec, value=score.detach().cpu(),
                    gradient=gradient.detach().cpu(), attribution=attribution.detach().cpu(),
                    token_values=None if token_values is None else token_values.detach().cpu(),
                    attribution_sum=total.detach().cpu(),
                    local_affine_intercept=(score.detach() - total).cpu(),
                )
            return EndpointScan(logits.detach().cpu(), units, forward_count=1)
    finally:
        for handle in handles:
            handle.remove()


def default_endpoint_targets(
    model: nn.Module, *, blocks: Sequence[int] = (0, 5, 11),
    channel: int = 0, head_block: int = 0, class_id: int | None = None,
) -> tuple[EndpointTarget, ...]:
    """Fixed channel scans, residual inputs/branches, every head, CLS and logits."""
    blocks = tuple(blocks)
    if not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("choose distinct blocks")
    if any(not 0 <= b < len(model.blocks) for b in (*blocks, head_block)):
        raise ValueError("block index out of range")
    targets = [EndpointTarget("patch_embedding", "patch_embed", channel=channel)]
    for block in blocks:
        for name, module, location in (
            ("input", "norm1", "input"),
            ("attention", "drop_path1", "output"),
            ("residual1", "norm2", "input"),
            ("mlp_fc1", "mlp.fc1", "output"),
            ("mlp_gelu", "mlp.act", "output"),
            ("mlp", "drop_path2", "output"),
            ("residual2", "", "output"),
        ):
            path = f"blocks.{block}" + (f".{module}" if module else "")
            targets.append(EndpointTarget(f"b{block}_{name}", path, location=location, channel=channel))
        # A fixed spatial token, distinct from channel sums and CLS head targets.
        targets.append(EndpointTarget(
            f"b{block}_unit_t1_c{channel}", f"blocks.{block}",
            kind="token_channel", channel=channel, token=model.num_prefix_tokens,
        ))
    for head in range(model.blocks[head_block].attn.num_heads):
        targets.append(EndpointTarget(
            f"b{head_block}_head{head}", f"blocks.{head_block}.attn.proj",
            kind="head_channel", location="input", head=head, channel=0, token=0,
        ))
    targets.extend((
        EndpointTarget("final_cls", "norm", kind="token_channel", channel=channel, token=0),
        EndpointTarget("class_logit", "", kind="logit", class_id=class_id),
        EndpointTarget("class_margin", "", kind="margin", class_id=class_id),
    ))
    return tuple(targets)


def attention_endpoint_components(
    model: nn.Module, inputs: Tensor, *, block: int = 0, head: int = 0,
    channel: int = 0, token: int = 0,
) -> dict[str, Tensor]:
    """Decompose one native O_h scalar at the Q/K/V projection output edges.

    The supported timm attention has Q, K, V as the complete input-dependent
    cut for O_h. Fixed biases remain in the forward. For attention with another
    input-dependent edge, closure must be extended before using this helper.
    Returned maps are first-order components, not intervention effects.
    """
    if inputs.ndim != 4 or inputs.shape[0] != 1 or not inputs.is_floating_point():
        raise ValueError("components expect one floating-point image")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("nonfinite input")
    if not 0 <= block < len(model.blocks):
        raise ValueError("block out of range")
    attention = model.blocks[block].attn
    if getattr(model, "grad_checkpointing", False):
        raise ValueError("disable gradient checkpointing")
    spec = EndpointTarget("head", f"blocks.{block}.attn.proj", kind="head_channel",
                          location="input", head=head, channel=channel, token=token)
    observed = {}
    handles = []
    def save_qkv(_module, _args, output):
        observed["qkv"] = output
    def save_output(_module, args):
        observed["output"] = args[0]
    clean = inputs.detach().clone().requires_grad_(True)
    try:
        handles.append(attention.qkv.register_forward_hook(save_qkv))
        handles.append(attention.proj.register_forward_pre_hook(save_output))
        with _frozen_eval(model), torch.enable_grad():
            model(clean)
            _, score, _ = _select(model, spec, observed["output"])
            qkv = observed["qkv"]
            full = torch.autograd.grad(score.sum(), clean, retain_graph=True)[0]
            edge_gradient = torch.autograd.grad(score.sum(), qkv, retain_graph=True)[0]
            parts = {"full": (clean.detach() * full).detach().cpu()}
            for index, name in enumerate(("query", "key", "value")):
                seed = torch.zeros_like(edge_gradient).reshape(*qkv.shape[:2], 3, -1)
                seed[:, :, index] = edge_gradient.reshape_as(seed)[:, :, index]
                gradient = torch.autograd.grad(
                    qkv, clean, grad_outputs=seed.reshape_as(qkv), retain_graph=index < 2,
                )[0]
                parts[name] = (clean.detach() * gradient).detach().cpu()
            if any(not bool(torch.isfinite(value).all()) for value in parts.values()):
                raise ValueError("nonfinite attention component")
            return parts
    finally:
        for handle in handles:
            handle.remove()


def decompose_endpoint_vit(
    model: nn.Module, inputs: Tensor, *, block: int,
    target: EndpointTarget, atol: float = 1e-9, rtol: float = 1e-7,
) -> EndpointDecomposition:
    """Complete edge cuts for one class logit/margin on a standard timm block.

    Each map is x times a VJP with the terminal target's downstream cotangent.
    At r=u+a, the identity seed is dF/dr, NOT dF/du (which also includes the
    branch). Attention head and Q/K/V cuts include identity as an extra edge.
    Downstream cotangents are held constant in these first-order VJPs.
    """
    from timm.models.vision_transformer import Attention, Block

    if inputs.ndim != 4 or inputs.shape[0] != 1 or not inputs.is_floating_point():
        raise ValueError("decomposition expects one floating-point image")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("nonfinite input")
    if not 0 <= block < len(model.blocks):
        raise ValueError("block out of range")
    current = model.blocks[block]
    if type(current) is not Block or type(current.attn) is not Attention:
        raise ValueError("edge contract requires the standard timm Block and Attention")
    if getattr(model, "grad_checkpointing", False):
        raise ValueError("disable gradient checkpointing")
    if target.kind not in {"logit", "margin"} or target.module:
        raise ValueError("decomposition target must be a class logit or margin")
    if atol < 0 or rtol < 0:
        raise ValueError("closure tolerances must be nonnegative")
    observed = {}
    handles = []
    bindings = {
        "u": (current.norm1, "input"),
        "attention": (current.drop_path1, "output"),
        "r1": (current.norm2, "input"),
        "mlp": (current.drop_path2, "output"),
        "r2": (current, "output"),
        "heads": (current.attn.proj, "input"),
        "qkv": (current.attn.qkv, "output"),
    }
    clean = inputs.detach().clone().requires_grad_(True)
    try:
        for name, (module, location) in bindings.items():
            if location == "input":
                def pre_hook(_m, args, name=name):
                    observed[name] = args[0]
                handles.append(module.register_forward_pre_hook(pre_hook))
            else:
                def post_hook(_m, _args, output, name=name):
                    observed[name] = output
                handles.append(module.register_forward_hook(post_hook))
        with _frozen_eval(model), torch.enable_grad():
            logits = model(clean)
            resolved, score, _ = _select(model, target, logits)
            torch.testing.assert_close(observed["r1"], observed["u"] + observed["attention"], atol=0, rtol=0)
            torch.testing.assert_close(observed["r2"], observed["r1"] + observed["mlp"], atol=0, rtol=0)
            full = clean.detach() * torch.autograd.grad(score.sum(), clean, retain_graph=True)[0]
            g1, g2, gh, gqkv = torch.autograd.grad(
                score.sum(), tuple(observed[key] for key in ("r1", "r2", "heads", "qkv")),
                retain_graph=True,
            )

            def vjp(key, seed):
                gradient = torch.autograd.grad(
                    observed[key], clean, grad_outputs=seed.detach(), retain_graph=True,
                )[0]
                return (clean.detach() * gradient).detach().cpu()

            components = {
                "identity": vjp("u", g1),
                "attention": vjp("attention", g1),
                "carry": vjp("r1", g2),
                "mlp": vjp("mlp", g2),
            }
            width = observed["heads"].shape[-1] // current.attn.num_heads
            for head in range(current.attn.num_heads):
                seed = torch.zeros_like(gh)
                seed[:, :, head * width:(head + 1) * width] = gh[:, :, head * width:(head + 1) * width]
                components[f"head_{head}"] = vjp("heads", seed)
            split = gqkv.reshape(*gqkv.shape[:2], 3, -1)
            for index, name in enumerate(("query", "key", "value")):
                seed = torch.zeros_like(split)
                seed[:, :, index] = split[:, :, index]
                components[name] = vjp("qkv", seed.reshape_as(gqkv))
            components["routing"] = components["query"] + components["key"]
            cuts = {
                "residual1": ("identity", "attention"),
                "residual2": ("carry", "mlp"),
                "heads_with_bypass": ("identity",) + tuple(f"head_{h}" for h in range(current.attn.num_heads)),
                "qkv_with_bypass": ("identity", "query", "key", "value"),
                "routing_value_with_bypass": ("identity", "routing", "value"),
            }
            full = full.detach().cpu()
            abs_errors, rel_errors = {}, {}
            if not bool(torch.isfinite(full).all()) or any(not bool(torch.isfinite(v).all()) for v in components.values()):
                raise ValueError("nonfinite decomposition")
            for name, keys in cuts.items():
                summed = sum((components[key] for key in keys), torch.zeros_like(full))
                error = summed - full
                abs_errors[name] = float(error.abs().max())
                rel_errors[name] = float(error.norm() / full.norm().clamp_min(1e-30))
                torch.testing.assert_close(summed, full, atol=atol, rtol=rtol)
            return EndpointDecomposition(resolved, score.detach().cpu(), full, components, cuts, abs_errors, rel_errors)
    finally:
        for handle in handles:
            handle.remove()
