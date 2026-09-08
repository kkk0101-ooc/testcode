"""NNS-v2 complete-cut path-Jacobian attribution for unmodified ViTs.

Scientific arrays and display arrays are separate types. Cut allocations are
streamed in deterministic batches, and retained component maps are accumulated
on CPU so the GPU never holds every layer, component, and input-pixel map.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

METHOD_ID = "nns_v2_complete_cut_path_jacobian_ig"
STEPS = (128, 256, 512, 1024, 2048, 4096, 8192)
ALPHA_CHUNK = 4
ENDPOINT_LT = 1e-6
INPUT_ATOL, INPUT_RTOL = 1e-4, 0.005
CUT_PIXEL_ATOL, CUT_PIXEL_RTOL = 2e-5, 0.005
CUT_TOTAL_ATOL, CUT_TOTAL_RTOL = 2e-4, 0.01
IDENTITY_ATOL = IDENTITY_RTOL = 1e-6
DEFAULT_MEMORY_CAP_BYTES = 48 * 1024**3


class NNSV2Error(ValueError):
    """Invalid input, graph contract, or retained evidence."""


class NNSV2ConvergenceError(NNSV2Error):
    status = "NO_GO_CLASS_LOGIT_COMPLETENESS"

    def __init__(self, message: str, attempts: Sequence[Mapping[str, Any]]):
        super().__init__(message)
        self.attempts = tuple(dict(item) for item in attempts)


class NNSV2MemoryPlanError(NNSV2Error):
    status = "BLOCKED_DETERMINISTIC_MEMORY_PLAN"


@dataclass(frozen=True)
class ScientificArray:
    value: torch.Tensor
    semantics: str

    def __post_init__(self) -> None:
        if self.value.dtype != torch.float32 or self.value.ndim not in (2, 3):
            raise NNSV2Error("scientific arrays must be native-rank float32")
        if not bool(torch.isfinite(self.value).all()):
            raise NNSV2Error("scientific arrays must be finite")


@dataclass(frozen=True)
class DisplayArray:
    value: np.ndarray
    colormap: str

    def __post_init__(self) -> None:
        if self.value.dtype != np.uint8 or self.value.ndim not in (2, 3):
            raise NNSV2Error("display arrays must be uint8 images")


@dataclass(frozen=True)
class TargetRecord:
    predicted_class: int
    true_class: int | None
    runner_up_class: int
    predicted_logit: float
    true_logit: float | None
    runner_up_logit: float
    clean_margin: float
    clean_logits_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "separate_target_denominators": True}


@dataclass(frozen=True)
class ComponentSpec:
    label: str
    tensor_key: str
    axis: int | None = None
    start: int | None = None
    stop: int | None = None
    cotangent_key: str | None = None

    def validate(self) -> None:
        if not self.label or not self.tensor_key:
            raise NNSV2Error("empty component identity")
        if self.axis is None:
            if self.start is not None or self.stop is not None:
                raise NNSV2Error("whole-tensor component has a range")
            return
        if (
            type(self.axis) is not int
            or type(self.start) is not int
            or type(self.stop) is not int
            or self.start < 0
            or self.stop <= self.start
        ):
            raise NNSV2Error("invalid component range")


@dataclass(frozen=True)
class CutSpec:
    cut_id: str
    components: tuple[ComponentSpec, ...]
    partial_tensor_keys: tuple[str, ...] = ()

    def validate(self) -> None:
        labels = {item.label for item in self.components}
        if not self.cut_id or not self.components or len(labels) != len(self.components):
            raise NNSV2Error("invalid cut identity or duplicate component")
        for item in self.components:
            item.validate()
        if not set(self.partial_tensor_keys) <= {item.tensor_key for item in self.components}:
            raise NNSV2Error("unknown partial tensor key")


@dataclass(frozen=True)
class MemoryPlan:
    alpha_chunk: int
    component_chunk: int
    input_elements: int
    temporary_vjp_bytes: int
    cap_bytes: int
    accumulation_device: str = "cpu"
    full_component_maps_on_gpu: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class CutResult:
    cut_id: str
    component_count: int
    component_totals: Mapping[str, float]
    component_maps: Mapping[str, torch.Tensor]
    retained_component_labels: tuple[str, ...]
    reconstructed: torch.Tensor
    pixel_max_abs_error: float
    total_abs_error: float
    passed: bool
    streamed: bool = True


@dataclass(frozen=True)
class NNSV2Result:
    attribution: torch.Tensor
    signed_rgb_map: torch.Tensor
    target: TargetRecord
    delta: float
    input_sum: float
    input_abs_error: float
    cuts: Mapping[str, CutResult]
    steps: int
    attempts: tuple[Mapping[str, Any], ...]
    converged: bool
    endpoint_max_abs_error: float
    identity_max_abs_error: float
    memory_plan: MemoryPlan
    constant_direct_terms: Mapping[str, float]


def array_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_input(image: torch.Tensor) -> None:
    if (
        image.ndim != 4
        or image.shape[0] != 1
        or image.dtype not in (torch.float32, torch.float64)
        or not bool(torch.isfinite(image).all())
    ):
        raise NNSV2Error("expected one finite floating input image")


def resolve_target(
    model: nn.Module, image: torch.Tensor, true_class: int | None = None
) -> TargetRecord:
    """Record prediction, truth, runner-up, and clean margin separately."""
    _validate_input(image)
    if true_class is not None and (type(true_class) is not int or true_class < 0):
        raise NNSV2Error("invalid true class")
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            logits = model(image)
    finally:
        for module, state in states:
            module.training = state
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] < 2:
        raise NNSV2Error("model did not return raw class logits")
    if not bool(torch.isfinite(logits).all()):
        raise NNSV2Error("raw logits are not finite")
    order = torch.argsort(logits[0], descending=True, stable=True)
    predicted, runner_up = int(order[0]), int(order[1])
    if true_class is not None and true_class >= logits.shape[1]:
        raise NNSV2Error("true class outside logits")
    return TargetRecord(
        predicted,
        true_class,
        runner_up,
        float(logits[0, predicted]),
        None if true_class is None else float(logits[0, true_class]),
        float(logits[0, runner_up]),
        float(logits[0, predicted] - logits[0, runner_up]),
        array_digest(logits),
    )


def gauss_legendre(
    count: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    if type(count) is not int or count <= 0:
        raise NNSV2Error("quadrature count must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return (
        torch.as_tensor((nodes + 1.0) / 2.0, device=device, dtype=dtype),
        torch.as_tensor(weights / 2.0, device=device, dtype=torch.float64),
    )


def deterministic_memory_plan(
    inputs: torch.Tensor,
    *,
    alpha_chunk: int = ALPHA_CHUNK,
    requested_component_chunk: int = 8,
    cap_bytes: int = DEFAULT_MEMORY_CAP_BYTES,
) -> MemoryPlan:
    if alpha_chunk != ALPHA_CHUNK and inputs.shape[-2:] == (224, 224):
        raise NNSV2MemoryPlanError("production alpha chunk must equal four")
    if min(alpha_chunk, requested_component_chunk, cap_bytes) <= 0:
        raise NNSV2MemoryPlanError("invalid deterministic memory request")
    elements = inputs.numel()
    bytes_per_component = alpha_chunk * elements * 8
    maximum = (cap_bytes // 8) // max(bytes_per_component, 1)
    chunk = min(requested_component_chunk, maximum)
    if chunk < 1:
        raise NNSV2MemoryPlanError("one streamed component exceeds the memory plan")
    return MemoryPlan(alpha_chunk, int(chunk), elements, int(bytes_per_component * chunk), cap_bytes)


def _select(tensor: torch.Tensor, spec: ComponentSpec) -> torch.Tensor:
    if spec.axis is None:
        return tensor
    axis = spec.axis if spec.axis >= 0 else tensor.ndim + spec.axis
    assert spec.start is not None and spec.stop is not None
    if axis <= 0 or axis >= tensor.ndim or spec.stop > tensor.shape[axis]:
        raise NNSV2Error("component outside captured tensor")
    index = [slice(None)] * tensor.ndim
    index[axis] = slice(spec.start, spec.stop)
    return tensor[tuple(index)]


def validate_partition(captures: Mapping[str, torch.Tensor], cut: CutSpec) -> None:
    cut.validate()
    grouped: dict[str, list[ComponentSpec]] = {}
    for spec in cut.components:
        if spec.tensor_key not in captures:
            raise NNSV2Error(f"missing cut tensor: {spec.tensor_key}")
        if spec.cotangent_key and spec.cotangent_key not in captures:
            raise NNSV2Error(f"missing cotangent tensor: {spec.cotangent_key}")
        grouped.setdefault(spec.tensor_key, []).append(spec)
    for key, items in grouped.items():
        tensor = captures[key]
        if any(item.axis is None for item in items):
            if len(items) != 1:
                raise NNSV2Error("whole component overlaps another component")
            continue
        axes = {item.axis if item.axis >= 0 else tensor.ndim + item.axis for item in items}
        if len(axes) != 1:
            raise NNSV2Error("one tensor was partitioned along mixed axes")
        axis = next(iter(axes))
        coverage = torch.zeros(tensor.shape[axis], dtype=torch.int16)
        for item in items:
            assert item.start is not None and item.stop is not None
            coverage[item.start:item.stop] += 1
        if bool(torch.any(coverage > 1)):
            raise NNSV2Error("overlapping component partition")
        if key not in cut.partial_tensor_keys and not bool(torch.all(coverage == 1)):
            raise NNSV2Error("incomplete component partition")


class TimmViTCutTracer:
    """Capture native timm ViT edges without modifying the forward graph."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.captures: dict[str, torch.Tensor] = {}
        self.stack = ExitStack()

    def _out(self, key: str) -> Callable[..., None]:
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            self.captures[key] = output
        return hook

    def _inp(self, key: str) -> Callable[..., None]:
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            self.captures[key] = args[0]
        return hook

    def __enter__(self) -> "TimmViTCutTracer":
        blocks = getattr(self.model, "blocks", None)
        if blocks is None or not len(blocks) or not hasattr(self.model, "norm_pre"):
            raise NNSV2Error("unsupported ViT architecture")
        handles = [self.model.norm_pre.register_forward_hook(self._out("post_embedding"))]
        for index, block in enumerate(blocks):
            prefix = f"b{index}"
            handles.extend((
                block.register_forward_pre_hook(self._inp(prefix + ".input")),
                block.register_forward_hook(self._out(prefix + ".output")),
                block.norm1.register_forward_hook(self._out(prefix + ".norm1")),
                block.attn.qkv.register_forward_hook(self._out(prefix + ".qkv")),
                block.attn.attn_drop.register_forward_hook(self._out(prefix + ".attention")),
                block.attn.proj.register_forward_pre_hook(self._inp(prefix + ".head_concat")),
                block.drop_path1.register_forward_hook(self._out(prefix + ".attn_projection")),
                block.norm2.register_forward_pre_hook(self._inp(prefix + ".add1")),
                block.norm2.register_forward_hook(self._out(prefix + ".norm2")),
                block.mlp.fc2.register_forward_pre_hook(self._inp(prefix + ".mlp_hidden")),
                block.drop_path2.register_forward_hook(self._out(prefix + ".mlp_output")),
            ))
        handles.extend((
            self.model.norm.register_forward_hook(self._out("final_norm")),
            self.model.head.register_forward_pre_hook(self._inp("classifier_input")),
        ))
        for handle in handles:
            self.stack.callback(handle.remove)
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stack.close()

    def forward(self, image: torch.Tensor, target: int) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        self.captures = {}
        logits = self.model(image)
        expected = 3 + 11 * len(self.model.blocks)
        if len(self.captures) != expected:
            raise NNSV2Error(f"incomplete native cut capture: {len(self.captures)}/{expected}")
        return logits[:, target], dict(self.captures)


def timm_cut_specs(
    captures: Mapping[str, torch.Tensor], blocks: int, heads: int | Sequence[int]
) -> tuple[CutSpec, ...]:
    """Build every frozen complete edge cut, including true residual edges."""
    head_counts = (heads,) * blocks if isinstance(heads, int) else tuple(heads)
    if len(head_counts) != blocks:
        raise NNSV2Error("head-count length differs from block count")
    tokens = captures["post_embedding"].shape[1]
    result: list[CutSpec] = [CutSpec("post_embedding", tuple(
        ComponentSpec("class_token" if token == 0 else f"patch_token_{token - 1}", "post_embedding", 1, token, token + 1)
        for token in range(tokens)
    ))]
    for index, head_count in enumerate(head_counts):
        prefix = f"b{index}"
        width = captures[prefix + ".qkv"].shape[-1] // 3
        if width % head_count:
            raise NNSV2Error("embedding width is not divisible by heads")
        head_dim = width // head_count
        residual_one = ComponentSpec("residual", prefix + ".input", cotangent_key=prefix + ".add1")
        residual_two = ComponentSpec("residual", prefix + ".add1", cotangent_key=prefix + ".output")
        qkv = tuple(
            ComponentSpec(f"{kind}_head_{head}", prefix + ".qkv", -1, offset + head * head_dim, offset + (head + 1) * head_dim)
            for kind, offset in (("q", 0), ("k", width), ("v", 2 * width))
            for head in range(head_count)
        )
        attention_value = tuple(
            ComponentSpec(f"attention_head_{head}", prefix + ".attention", 1, head, head + 1)
            for head in range(head_count)
        ) + tuple(
            ComponentSpec(f"value_head_{head}", prefix + ".qkv", -1, 2 * width + head * head_dim, 2 * width + (head + 1) * head_dim)
            for head in range(head_count)
        )
        hidden = captures[prefix + ".mlp_hidden"].shape[-1]
        result.extend((
            CutSpec(prefix + ".input", (ComponentSpec("all", prefix + ".input"),)),
            CutSpec(prefix + ".norm1", (residual_one, ComponentSpec("attention_branch_input", prefix + ".norm1"))),
            CutSpec(prefix + ".qkv", (residual_one, *qkv)),
            CutSpec(prefix + ".attention_v", (residual_one, *attention_value), (prefix + ".qkv",)),
            CutSpec(prefix + ".heads", (residual_one, *(ComponentSpec(f"head_{head}", prefix + ".head_concat", -1, head * head_dim, (head + 1) * head_dim) for head in range(head_count)))),
            CutSpec(prefix + ".attention_projection", (residual_one, ComponentSpec("attention_branch", prefix + ".attn_projection"))),
            CutSpec(prefix + ".add1", (ComponentSpec("all", prefix + ".add1"),)),
            CutSpec(prefix + ".norm2", (residual_two, ComponentSpec("mlp_branch_input", prefix + ".norm2"))),
            CutSpec(prefix + ".mlp_hidden", (residual_two, *(ComponentSpec(f"neuron_{neuron}", prefix + ".mlp_hidden", -1, neuron, neuron + 1) for neuron in range(hidden)))),
            CutSpec(prefix + ".mlp_output", (residual_two, ComponentSpec("mlp_branch", prefix + ".mlp_output"))),
            CutSpec(prefix + ".add2", (ComponentSpec("all", prefix + ".output"),)),
        ))
    result.extend((
        CutSpec("final_norm", (ComponentSpec("all", "final_norm"),)),
        CutSpec("classifier_input", (ComponentSpec("all", "classifier_input"),)),
    ))
    for cut in result:
        validate_partition(captures, cut)
    return tuple(result)


def representative_real_cut_ids(block_count: int) -> tuple[str, ...]:
    names: list[str] = ["post_embedding"]
    for index in range(block_count):
        prefix = f"b{index}"
        names.extend((prefix + ".norm1", prefix + ".qkv", prefix + ".attention_v", prefix + ".heads", prefix + ".attention_projection", prefix + ".norm2", prefix + ".mlp_output"))
    names.extend(("final_norm", "classifier_input"))
    return tuple(names)


def _unique_captures(captures: Mapping[str, torch.Tensor]) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    seen: set[int] = set()
    for tensor in captures.values():
        if id(tensor) not in seen:
            result.append(tensor)
            seen.add(id(tensor))
    return result


def _component_scalar(
    captures: Mapping[str, torch.Tensor], cotangents: Mapping[int, torch.Tensor | None], spec: ComponentSpec
) -> torch.Tensor:
    tensor = captures[spec.tensor_key]
    cotangent_tensor = captures[spec.cotangent_key] if spec.cotangent_key else tensor
    gradient = cotangents.get(id(cotangent_tensor))
    if gradient is None:
        raise NNSV2Error(f"disconnected complete-cut component: {spec.label}")
    return (_select(tensor, spec) * _select(gradient.detach(), spec)).sum()


def _aggregate_cut_scalar(
    captures: Mapping[str, torch.Tensor],
    cotangents: Mapping[int, torch.Tensor | None],
    cut: CutSpec,
) -> torch.Tensor:
    """Aggregate a complete partition without materializing per-neuron scalars."""
    grouped: dict[tuple[str, str | None], list[ComponentSpec]] = {}
    for spec in cut.components:
        grouped.setdefault((spec.tensor_key, spec.cotangent_key), []).append(spec)
    values = []
    for (tensor_key, cotangent_key), specs in grouped.items():
        tensor = captures[tensor_key]
        cotangent_tensor = captures[cotangent_key] if cotangent_key else tensor
        gradient = cotangents.get(id(cotangent_tensor))
        if gradient is None:
            raise NNSV2Error(f"disconnected complete-cut tensor: {tensor_key}")
        if len(specs) == 1 and specs[0].axis is None:
            values.append((tensor * gradient.detach()).sum())
            continue
        axis = specs[0].axis
        assert axis is not None
        normalized_axis = axis if axis >= 0 else tensor.ndim + axis
        ordered = sorted(specs, key=lambda item: item.start)
        assert ordered[0].start is not None and ordered[-1].stop is not None
        contiguous = all(
            left.stop == right.start for left, right in zip(ordered, ordered[1:])
        )
        if contiguous:
            union = ComponentSpec(
                "aggregate", tensor_key, normalized_axis,
                ordered[0].start, ordered[-1].stop, cotangent_key,
            )
            values.append((_select(tensor, union) * _select(gradient.detach(), union)).sum())
        else:
            values.append(sum(_component_scalar(captures, cotangents, spec) for spec in specs))
    return torch.stack(values).sum()


def integrate(
    forward_capture: Callable[[torch.Tensor, int], tuple[torch.Tensor, Mapping[str, torch.Tensor]]],
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    target: int,
    cuts: Sequence[CutSpec],
    steps: int = 32,
    alpha_chunk: int = ALPHA_CHUNK,
    component_chunk: int = 8,
    *,
    detailed_cut_ids: Iterable[str] | None = None,
    retained_components: Mapping[str, Iterable[str]] | str | None = None,
    memory_cap_bytes: int = DEFAULT_MEMORY_CAP_BYTES,
) -> tuple[torch.Tensor, Mapping[str, CutResult], MemoryPlan]:
    """Stream complete-cut allocation; retain maps only for an explicit L2 subset."""
    _validate_input(inputs)
    if baseline.shape != inputs.shape or baseline.dtype != inputs.dtype or baseline.device != inputs.device:
        raise NNSV2Error("baseline identity drift")
    plan = deterministic_memory_plan(
        inputs,
        alpha_chunk=alpha_chunk,
        requested_component_chunk=component_chunk,
        cap_bytes=memory_cap_bytes,
    )
    cut_tuple = tuple(cuts)
    if not cut_tuple or len({cut.cut_id for cut in cut_tuple}) != len(cut_tuple):
        raise NNSV2Error("cuts must be non-empty and uniquely named")
    detailed = set(detailed_cut_ids or ())
    if detailed - {cut.cut_id for cut in cut_tuple}:
        raise NNSV2Error("unknown detailed cut")
    if retained_components == "all":
        retain = {cut.cut_id: {item.label for item in cut.components} for cut in cut_tuple}
        detailed.update(retain)
    else:
        retain = {key: set(value) for key, value in (retained_components or {}).items()}
    for cut in cut_tuple:
        labels = {item.label for item in cut.components}
        if not retain.get(cut.cut_id, set()) <= labels:
            raise NNSV2Error(f"unknown retained component for {cut.cut_id}")
        if retain.get(cut.cut_id):
            detailed.add(cut.cut_id)

    nodes, weights = gauss_legendre(steps, inputs.device, inputs.dtype)
    difference = inputs - baseline
    total_accumulator = torch.zeros_like(inputs, dtype=torch.float64)
    reconstructions = {
        cut.cut_id: torch.zeros_like(inputs, device="cpu", dtype=torch.float64)
        for cut in cut_tuple
    }
    component_totals: dict[str, dict[str, float]] = {cut.cut_id: {} for cut in cut_tuple}
    component_maps: dict[str, dict[str, torch.Tensor]] = {
        cut.cut_id: {
            label: torch.zeros_like(inputs, device="cpu", dtype=torch.float64)
            for label in retain.get(cut.cut_id, set())
        }
        for cut in cut_tuple
    }
    for begin in range(0, steps, plan.alpha_chunk):
        end = min(begin + plan.alpha_chunk, steps)
        batch = end - begin
        scaled = (
            baseline.unsqueeze(0)
            + nodes[begin:end].reshape(-1, 1, 1, 1, 1) * difference.unsqueeze(0)
        ).reshape(-1, *inputs.shape[1:]).detach().requires_grad_(True)
        selected, captures = forward_capture(scaled, target)
        unique = _unique_captures(captures)
        gradients = torch.autograd.grad(
            selected.sum(), (scaled, *unique), retain_graph=True, allow_unused=True
        )
        cotangents = {id(tensor): gradient for tensor, gradient in zip(unique, gradients[1:])}
        weight = weights[begin:end].reshape(batch, *((1,) * inputs.ndim))
        input_gradient = gradients[0].reshape(batch, *inputs.shape).to(torch.float64)
        total_accumulator.add_((input_gradient * weight).sum(0))
        aggregate_scalars: list[torch.Tensor] = []
        aggregate_cuts: list[CutSpec] = []
        detailed_scalars: dict[str, list[torch.Tensor]] = {}
        for cut in cut_tuple:
            validate_partition(captures, cut)
            if cut.cut_id not in detailed:
                aggregate_scalars.append(_aggregate_cut_scalar(captures, cotangents, cut))
                aggregate_cuts.append(cut)
            else:
                detailed_scalars[cut.cut_id] = [
                    _component_scalar(captures, cotangents, item) for item in cut.components
                ]
        for start in range(0, len(aggregate_scalars), plan.component_chunk):
            stop = min(start + plan.component_chunk, len(aggregate_scalars))
            stack = torch.stack(aggregate_scalars[start:stop])
            eye = torch.eye(stop - start, device=scaled.device, dtype=stack.dtype)
            vjps = torch.autograd.grad(
                stack, scaled, grad_outputs=eye, is_grads_batched=True, retain_graph=True
            )[0].reshape(stop - start, batch, *inputs.shape)
            integrated = (vjps.double() * weight.unsqueeze(0)).sum(1)
            allocated = difference.double().unsqueeze(0) * integrated
            for local, cut in enumerate(aggregate_cuts[start:stop]):
                reconstructions[cut.cut_id].add_(allocated[local].detach().cpu())
        for cut in cut_tuple:
            if cut.cut_id not in detailed:
                continue
            scalars = detailed_scalars[cut.cut_id]
            for start in range(0, len(scalars), plan.component_chunk):
                stop = min(start + plan.component_chunk, len(scalars))
                stack = torch.stack(scalars[start:stop])
                eye = torch.eye(stop - start, device=scaled.device, dtype=stack.dtype)
                vjps = torch.autograd.grad(
                    stack, scaled, grad_outputs=eye, is_grads_batched=True, retain_graph=True
                )[0].reshape(stop - start, batch, *inputs.shape)
                integrated = (vjps.double() * weight.unsqueeze(0)).sum(1)
                allocated = difference.double().unsqueeze(0) * integrated
                reconstructions[cut.cut_id].add_(allocated.sum(0).detach().cpu())
                for local, spec in enumerate(cut.components[start:stop]):
                    value = allocated[local].detach().cpu()
                    component_totals[cut.cut_id][spec.label] = (
                        component_totals[cut.cut_id].get(spec.label, 0.0) + float(value.sum())
                    )
                    if spec.label in component_maps[cut.cut_id]:
                        component_maps[cut.cut_id][spec.label].add_(value)
        del scaled, selected, captures, unique, gradients, cotangents

    attribution = (difference.double() * total_accumulator).detach().cpu().float()
    results = {}
    for cut in cut_tuple:
        retained = {key: value[0].float() for key, value in component_maps[cut.cut_id].items()}
        results[cut.cut_id] = CutResult(
            cut.cut_id,
            len(cut.components),
            component_totals[cut.cut_id],
            retained,
            tuple(sorted(retained)),
            reconstructions[cut.cut_id][0].float(),
            float("nan"),
            float("nan"),
            False,
        )
    return attribution[0], results, plan


def input_ig(
    model: nn.Module,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    target: int,
    steps: int,
    alpha_chunk: int = ALPHA_CHUNK,
) -> torch.Tensor:
    """Independent full-model IG identity-control implementation."""
    _validate_input(inputs)
    if baseline.shape != inputs.shape or baseline.dtype != inputs.dtype or baseline.device != inputs.device:
        raise NNSV2Error("identity-control baseline drift")
    nodes, weights = gauss_legendre(steps, inputs.device, inputs.dtype)
    difference = inputs - baseline
    accumulator = torch.zeros_like(inputs, dtype=torch.float64)
    for begin in range(0, steps, alpha_chunk):
        end = min(begin + alpha_chunk, steps)
        batch = end - begin
        scaled = (
            baseline.unsqueeze(0)
            + nodes[begin:end].reshape(-1, 1, 1, 1, 1) * difference.unsqueeze(0)
        ).reshape(-1, *inputs.shape[1:]).detach().requires_grad_(True)
        selected = model(scaled)[:, target]
        gradient = torch.autograd.grad(selected.sum(), scaled)[0]
        weight = weights[begin:end].reshape(batch, *((1,) * inputs.ndim))
        accumulator.add_((gradient.reshape(batch, *inputs.shape).double() * weight).sum(0))
    return (difference.double() * accumulator).detach().cpu()[0].float()


def _finalize_cuts(
    cuts: Mapping[str, CutResult], attribution: torch.Tensor, delta: float
) -> dict[str, CutResult]:
    maximum = float(attribution.abs().max())
    result = {}
    for cut_id, cut in cuts.items():
        pixel_error = float((cut.reconstructed - attribution).abs().max())
        total_error = abs(float(cut.reconstructed.double().sum()) - delta)
        passed = (
            pixel_error <= CUT_PIXEL_ATOL + CUT_PIXEL_RTOL * maximum
            and total_error <= CUT_TOTAL_ATOL + CUT_TOTAL_RTOL * abs(delta)
        )
        result[cut_id] = CutResult(
            cut.cut_id,
            cut.component_count,
            cut.component_totals,
            cut.component_maps,
            cut.retained_component_labels,
            cut.reconstructed,
            pixel_error,
            total_error,
            passed,
        )
    return result


def explain(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    true_class: int | None = None,
    baseline: torch.Tensor | None = None,
    steps: Sequence[int] = STEPS,
    alpha_chunk: int = ALPHA_CHUNK,
    component_chunk: int = 8,
    cut_factory: Callable[[Mapping[str, torch.Tensor]], Sequence[CutSpec]] | None = None,
    selected_cut_ids: Iterable[str] | None = None,
    detailed_cut_ids: Iterable[str] | None = None,
    retained_components: Mapping[str, Iterable[str]] | str | None = None,
    memory_cap_bytes: int = DEFAULT_MEMORY_CAP_BYTES,
) -> NNSV2Result:
    """Explain the clean predicted raw logit and select first converged steps."""
    _validate_input(inputs)
    baseline = torch.zeros_like(inputs) if baseline is None else baseline
    if baseline.shape != inputs.shape or baseline.dtype != inputs.dtype or baseline.device != inputs.device:
        raise NNSV2Error("normalized-zero baseline drift")
    if not steps or tuple(sorted(set(steps))) != tuple(steps):
        raise NNSV2Error("step schedule must be unique and increasing")
    states = [(module, module.training) for module in model.modules()]
    requires = [parameter.requires_grad for parameter in model.parameters()]
    attempts: list[dict[str, Any]] = []
    try:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        target = resolve_target(model, inputs, true_class)
        with TimmViTCutTracer(model) as tracer:
            with torch.no_grad():
                clean, captures = tracer.forward(inputs, target.predicted_class)
                base, _ = tracer.forward(baseline, target.predicted_class)
                direct = model(inputs)[:, target.predicted_class]
            endpoint_error = float((clean - direct).abs().max())
            if endpoint_error >= ENDPOINT_LT:
                raise NNSV2Error("native tracer changed the selected raw logit")
            delta = float(clean - base)
            all_cuts = tuple(cut_factory(captures) if cut_factory else timm_cut_specs(
                captures, len(model.blocks), tuple(block.attn.num_heads for block in model.blocks)
            ))
            selected = set(selected_cut_ids or (cut.cut_id for cut in all_cuts))
            cuts = tuple(cut for cut in all_cuts if cut.cut_id in selected)
            if selected != {cut.cut_id for cut in cuts}:
                raise NNSV2Error("selected cut is not registered")
            for count in steps:
                attribution, provisional, plan = integrate(
                    tracer.forward, inputs, baseline, target.predicted_class, cuts,
                    count, alpha_chunk, component_chunk,
                    detailed_cut_ids=detailed_cut_ids,
                    retained_components=retained_components,
                    memory_cap_bytes=memory_cap_bytes,
                )
                cut_results = _finalize_cuts(provisional, attribution, delta)
                input_sum = float(attribution.double().sum())
                input_error = abs(input_sum - delta)
                identity = input_ig(model, inputs, baseline, target.predicted_class, count, alpha_chunk)
                identity_error = float((identity - attribution).abs().max())
                input_pass = input_error <= INPUT_ATOL + INPUT_RTOL * abs(delta)
                identity_pass = torch.allclose(identity, attribution, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)
                cut_pass = all(item.passed for item in cut_results.values())
                passed = input_pass and identity_pass and cut_pass
                attempts.append({
                    "steps": count,
                    "input_abs_error": input_error,
                    "identity_max_abs_error": identity_error,
                    "all_cuts_pass": cut_pass,
                    "pass": passed,
                })
                if passed:
                    if attribution.shape[0] not in (1, 3):
                        raise NNSV2Error("unexpected scientific attribution channels")
                    signed = attribution.sum(0).contiguous()
                    ScientificArray(attribution.contiguous(), "signed_total_input_attribution")
                    ScientificArray(signed, "signed_rgb_reduction")
                    return NNSV2Result(
                        attribution, signed, target, delta, input_sum, input_error,
                        cut_results, count, tuple(attempts), True, endpoint_error,
                        identity_error, plan,
                        {"bias": 0.0, "class_token": 0.0, "position_embedding": 0.0},
                    )
    finally:
        for parameter, state in zip(model.parameters(), requires):
            parameter.requires_grad_(state)
        for module, state in states:
            module.training = state
    raise NNSV2ConvergenceError(
        "NNS-v2 exhausted its schedule without complete-cut convergence", attempts
    )


def architecture_contract(model: nn.Module, model_id: str) -> dict[str, Any]:
    blocks = getattr(model, "blocks", None)
    patch = getattr(model, "patch_embed", None)
    if blocks is None or len(blocks) != 12 or getattr(model, "num_prefix_tokens", 0) != 1 or patch is None:
        raise NNSV2Error("unsupported architecture")
    if not all(
        hasattr(block.attn, "qkv") and hasattr(block.attn, "proj")
        and hasattr(block.attn, "attn_drop") and hasattr(block.mlp, "fc1")
        and hasattr(block.mlp, "fc2") for block in blocks
    ):
        raise NNSV2Error("registered cut surface is missing")
    return {
        "model_id": model_id,
        "depth": 12,
        "heads": [block.attn.num_heads for block in blocks],
        "embed_dim": model.embed_dim,
        "patch_size": list(patch.patch_size),
        "grid_size": list(patch.grid_size),
        "prefix_tokens": model.num_prefix_tokens,
        "required_capture_count": 3 + 11 * len(blocks),
        "fused_attention_requires_unfused_audit_mode": True,
        "layer_views_must_not_be_summed": True,
        "status": "PASS",
    }
