"""Auditable input attribution and fixed-attention interventions.

This module intentionally distinguishes three different questions:

* :func:`full_model_ig` attributes a final raw class logit.
* :func:`conditional_fixed_attention_model_ig` fixes one block's attention
  matrix, but still attributes a final class logit through *all* remaining
  model and residual paths.
* :func:`conditional_fixed_attention_value_path_ig` fixes that matrix and
  targets one scalar of the same block's pre-projection ``O = A @ V``.  The
  latter is a conditional Value-path attribution, not a complete attention
  attribution.

Integrated Gradients uses Gauss--Legendre quadrature, accumulates chunks
without retaining per-step graphs, and reports endpoint conservation.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .targets import TargetSpec, parse_target

Tensor = torch.Tensor
ForwardCallback = Callable[[Tensor], Tensor]
StepSpec = Union[int, str, Sequence[int]]

IG_STEP_SCHEDULES: Mapping[str, Tuple[int, ...]] = {
    "fast": (8, 16, 32),
    "default": (512, 1024, 2048, 4096, 8192),
    "audit": (512, 1024, 2048, 4096, 8192),
}


@dataclass(frozen=True)
class IGResult:
    """Integrated Gradients result.

    ``attribution`` is the native signed ``[B, C, H, W]`` tensor (channels
    are never reduced).  ``conservation`` has the exact keys ``delta``,
    ``sum``, ``abs``, ``rel``, and ``pass``, each containing a length-``B``
    tensor.  ``attempts`` and ``metadata`` contain JSON-compatible values.
    """

    attribution: Tensor
    conservation: Mapping[str, Tensor]
    steps: int
    converged: bool
    attempts: Tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]

    @property
    def delta(self) -> Tensor:
        return self.conservation["delta"]

    def to_dict(self, *, include_attribution: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "conservation": {
                key: value.detach().cpu().tolist()
                for key, value in self.conservation.items()
            },
            "steps": self.steps,
            "converged": self.converged,
            "attempts": [dict(attempt) for attempt in self.attempts],
            "metadata": dict(self.metadata),
        }
        if include_attribution:
            result["attribution"] = self.attribution.detach().cpu().tolist()
        return result


@dataclass(frozen=True)
class EndpointParityResult:
    """Serializable result of replaying an endpoint attention matrix."""

    passed: bool
    max_abs_error: float
    max_rel_error: float
    capture_calls: int
    intervention_calls: int
    attention_shape: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "capture_calls": self.capture_calls,
            "intervention_calls": self.intervention_calls,
            "attention_shape": list(self.attention_shape),
        }


def _resolve_steps(steps: StepSpec) -> Tuple[int, ...]:
    if isinstance(steps, bool):
        raise TypeError("steps must be an integer, schedule name, or integer sequence")
    if isinstance(steps, int):
        values = (steps,)
    elif isinstance(steps, str):
        try:
            values = IG_STEP_SCHEDULES[steps]
        except KeyError as exc:
            names = ", ".join(sorted(IG_STEP_SCHEDULES))
            raise ValueError(f"unknown IG step schedule {steps!r}; choose one of {names}") from exc
    elif isinstance(steps, Sequence):
        values = tuple(steps)
    else:
        raise TypeError("steps must be an integer, schedule name, or integer sequence")
    if not values:
        raise ValueError("steps schedule cannot be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("all IG step counts must be positive integers")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("an IG step schedule must be strictly increasing")
    return values


def _gauss_legendre_unit_interval(
    count: int, *, device: torch.device, dtype: torch.dtype
) -> Tuple[Tensor, Tensor]:
    """Return Gauss--Legendre nodes/weights on [0, 1] without NumPy."""

    roots = [0.0] * count
    weights = [0.0] * count
    half = (count + 1) // 2
    tolerance = 1e-15
    for index in range(half):
        root = math.cos(math.pi * (index + 0.75) / (count + 0.5))
        derivative = 0.0
        for _ in range(100):
            previous = 1.0
            current = root
            for degree in range(2, count + 1):
                following = (
                    (2.0 * degree - 1.0) * root * current
                    - (degree - 1.0) * previous
                ) / degree
                previous, current = current, following
            if count == 1:
                current, previous = root, 1.0
            derivative = count * (root * current - previous) / (root * root - 1.0)
            updated = root - current / derivative
            if abs(updated - root) <= tolerance:
                root = updated
                break
            root = updated
        weight = 2.0 / ((1.0 - root * root) * derivative * derivative)
        roots[index] = -root
        roots[count - 1 - index] = root
        weights[index] = weight
        weights[count - 1 - index] = weight
    nodes = torch.tensor([(value + 1.0) * 0.5 for value in roots], device=device, dtype=dtype)
    quadrature_weights = torch.tensor(
        [value * 0.5 for value in weights], device=device, dtype=torch.float32
    )
    return nodes, quadrature_weights


def _normalize_selected_output(output: Tensor, expected_batch: int) -> Tensor:
    if not isinstance(output, Tensor):
        raise TypeError("forward callback must return a Tensor")
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 1 or output.shape[0] != expected_batch:
        raise ValueError(
            "forward callback must return one scalar per input; "
            f"expected [{expected_batch}], got {tuple(output.shape)}"
        )
    if not output.is_floating_point():
        raise TypeError("forward callback output must be floating point")
    return output


@contextmanager
def _temporarily_freeze(parameters: Optional[Union[nn.Module, Iterable[nn.Parameter]]]) -> Iterator[None]:
    if parameters is None:
        yield
        return
    parameter_list = list(parameters.parameters() if isinstance(parameters, nn.Module) else parameters)
    original = [parameter.requires_grad for parameter in parameter_list]
    try:
        for parameter in parameter_list:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameter_list, original):
            parameter.requires_grad_(requires_grad)


@contextmanager
def _temporarily_eval(model: nn.Module) -> Iterator[None]:
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in states:
            module.training = training


def _prepare_inputs(inputs: Tensor, baselines: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a Tensor")
    if inputs.ndim != 4:
        raise ValueError(f"inputs must have native image shape [B, C, H, W], got {tuple(inputs.shape)}")
    if not inputs.is_floating_point():
        raise TypeError("inputs must be floating point")
    # Float32 is the audit/storage dtype.  Float64 is retained only when a
    # double-precision callback requires it; gradients and results accumulate
    # in float32.  Lower precision inputs are promoted.
    compute_dtype = torch.float64 if inputs.dtype == torch.float64 else torch.float32
    prepared_inputs = inputs.detach().to(dtype=compute_dtype)
    if baselines is None:
        prepared_baselines = torch.zeros_like(prepared_inputs)
    else:
        if not isinstance(baselines, Tensor):
            raise TypeError("baselines must be a Tensor or None")
        if not baselines.is_floating_point():
            raise TypeError("baselines must be floating point")
        prepared_baselines = baselines.detach().to(
            device=prepared_inputs.device, dtype=compute_dtype
        )
        try:
            prepared_baselines = torch.broadcast_to(
                prepared_baselines, prepared_inputs.shape
            ).clone()
        except RuntimeError as exc:
            raise ValueError(
                f"baselines shape {tuple(baselines.shape)} cannot broadcast to {tuple(inputs.shape)}"
            ) from exc
    if not torch.isfinite(prepared_inputs).all() or not torch.isfinite(prepared_baselines).all():
        raise ValueError("inputs and baselines must contain only finite values")
    return prepared_inputs, prepared_baselines


def _endpoint_values(forward_fn: ForwardCallback, endpoint: Tensor) -> Tensor:
    with torch.no_grad():
        value = _normalize_selected_output(forward_fn(endpoint), endpoint.shape[0])
    return value.detach().to(torch.float32)


def _integral_for_steps(
    forward_fn: ForwardCallback,
    inputs: Tensor,
    baselines: Tensor,
    *,
    steps: int,
    step_chunk_size: int,
) -> Tensor:
    batch_size = inputs.shape[0]
    difference = inputs - baselines
    nodes, weights = _gauss_legendre_unit_interval(
        steps, device=inputs.device, dtype=inputs.dtype
    )
    accumulated = torch.zeros_like(inputs, dtype=torch.float32)
    broadcast_dims = (1,) * inputs.ndim
    for start in range(0, steps, step_chunk_size):
        stop = min(start + step_chunk_size, steps)
        chunk_nodes = nodes[start:stop]
        chunk_weights = weights[start:stop]
        scaled = baselines.unsqueeze(0) + chunk_nodes.view(-1, *broadcast_dims) * difference.unsqueeze(0)
        chunk_steps = scaled.shape[0]
        scaled = scaled.reshape(chunk_steps * batch_size, *inputs.shape[1:]).detach()
        scaled.requires_grad_(True)
        selected = _normalize_selected_output(forward_fn(scaled), scaled.shape[0])
        if selected.requires_grad:
            gradient = torch.autograd.grad(
                selected.sum(), scaled, retain_graph=False, create_graph=False, allow_unused=True
            )[0]
        else:
            gradient = None
        if gradient is None:
            gradient = torch.zeros_like(scaled)
        gradient = gradient.reshape(chunk_steps, batch_size, *inputs.shape[1:]).to(torch.float32)
        weight_shape = (chunk_steps,) + (1,) * inputs.ndim
        accumulated.add_((gradient * chunk_weights.view(weight_shape)).sum(dim=0).detach())
        del selected, gradient, scaled
    return difference.to(torch.float32) * accumulated


def _integral_for_steps_float64_kahan(
    forward_fn: ForwardCallback,
    inputs: Tensor,
    baselines: Tensor,
    *,
    steps: int,
    step_chunk_size: int,
) -> Tensor:
    """Integrate a float32 model callback with float64 quadrature/Kahan sums.

    Only interpolation and numerical accumulation use float64.  Every callback
    input remains float32, so model parameters, forward/backward computation,
    target semantics, and the final stored attribution dtype are unchanged.
    """

    import numpy as np
    from scipy.special import roots_legendre

    if inputs.dtype is not torch.float32 or baselines.dtype is not torch.float32:
        raise TypeError(
            "float64-stable IG requires float32 model inputs and baselines"
        )
    roots, raw_weights = roots_legendre(steps)
    nodes_cpu = np.asarray((roots + 1.0) * 0.5, dtype=np.float64)
    weights_cpu = np.asarray(raw_weights * 0.5, dtype=np.float64)
    nodes = torch.from_numpy(nodes_cpu).to(device=inputs.device)
    weights = torch.from_numpy(weights_cpu).to(device=inputs.device)
    batch_size = inputs.shape[0]
    difference64 = (inputs - baselines).to(torch.float64)
    inputs64 = inputs.to(torch.float64)
    baselines64 = baselines.to(torch.float64)
    accumulated = torch.zeros_like(inputs64)
    compensation = torch.zeros_like(inputs64)
    broadcast_dims = (1,) * inputs.ndim
    for start in range(0, steps, step_chunk_size):
        stop = min(start + step_chunk_size, steps)
        chunk_nodes = nodes[start:stop]
        chunk_weights = weights[start:stop]
        scaled64 = (
            baselines64.unsqueeze(0)
            + chunk_nodes.view(-1, *broadcast_dims) * difference64.unsqueeze(0)
        )
        chunk_steps = scaled64.shape[0]
        scaled = (
            scaled64.reshape(chunk_steps * batch_size, *inputs.shape[1:])
            .to(torch.float32)
            .detach()
        )
        scaled.requires_grad_(True)
        selected = _normalize_selected_output(forward_fn(scaled), scaled.shape[0])
        if selected.requires_grad:
            gradient = torch.autograd.grad(
                selected.sum(), scaled, retain_graph=False,
                create_graph=False, allow_unused=True,
            )[0]
        else:
            gradient = None
        if gradient is None:
            gradient = torch.zeros_like(scaled)
        gradient64 = gradient.reshape(
            chunk_steps, batch_size, *inputs.shape[1:]
        ).to(torch.float64)
        weight_shape = (chunk_steps,) + (1,) * inputs.ndim
        weighted = (
            gradient64 * chunk_weights.view(weight_shape)
        ).sum(dim=0)
        corrected = weighted - compensation
        updated = accumulated + corrected
        compensation = (updated - accumulated) - corrected
        accumulated = updated
        del selected, gradient, gradient64, scaled, scaled64, weighted, corrected
    attribution64 = difference64 * accumulated
    return attribution64.to(torch.float32)


def _conservation(
    attribution: Tensor,
    delta: Tensor,
    *,
    atol: float,
    rtol: float,
) -> Dict[str, Tensor]:
    attribution_sum = attribution.flatten(1).sum(dim=1)
    absolute_error = (delta - attribution_sum).abs()
    tiny = torch.finfo(torch.float32).eps
    relative_error = absolute_error / delta.abs().clamp_min(tiny)
    passed = absolute_error <= (atol + rtol * delta.abs())
    return {
        "delta": delta.detach(),
        "sum": attribution_sum.detach(),
        "abs": absolute_error.detach(),
        "rel": relative_error.detach(),
        "pass": passed.detach(),
    }


def _conservation_float64_reduction(
    attribution: Tensor,
    delta: Tensor,
    *,
    atol: float,
    rtol: float,
) -> Dict[str, Tensor]:
    """Evaluate stored float32 attribution using a float64 final reduction."""

    attribution_sum = attribution.to(torch.float64).flatten(1).sum(dim=1)
    delta64 = delta.to(torch.float64)
    absolute_error = (delta64 - attribution_sum).abs()
    tiny = torch.finfo(torch.float64).eps
    relative_error = absolute_error / delta64.abs().clamp_min(tiny)
    passed = absolute_error <= (atol + rtol * delta64.abs())
    return {
        "delta": delta64.detach(),
        "sum": attribution_sum.detach(),
        "abs": absolute_error.detach(),
        "rel": relative_error.detach(),
        "pass": passed.detach(),
    }


def _serializable_attempt(steps: int, report: Mapping[str, Tensor]) -> Dict[str, Any]:
    return {
        "steps": steps,
        "delta": report["delta"].cpu().tolist(),
        "sum": report["sum"].cpu().tolist(),
        "abs": report["abs"].cpu().tolist(),
        "rel": report["rel"].cpu().tolist(),
        "pass": report["pass"].cpu().tolist(),
    }


def integrated_gradients(
    forward_fn: ForwardCallback,
    inputs: Tensor,
    baselines: Optional[Tensor] = None,
    *,
    steps: StepSpec = "default",
    step_chunk_size: int = 16,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    parameters: Optional[Union[nn.Module, Iterable[nn.Parameter]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> IGResult:
    """Compute signed image Integrated Gradients for a real callback.

    The callback is evaluated on actual interpolated tensors and must return
    one scalar per item.  A named or explicit schedule is tried in ascending
    order and stops at the first conservation pass.  Quadrature points are
    batched by ``step_chunk_size``; no individual-point graph is retained.
    Parameters supplied through ``parameters`` are frozen and restored even
    if attribution raises.
    """

    if not callable(forward_fn):
        raise TypeError("forward_fn must be callable")
    schedule = _resolve_steps(steps)
    if isinstance(step_chunk_size, bool) or not isinstance(step_chunk_size, int) or step_chunk_size <= 0:
        raise ValueError("step_chunk_size must be a positive integer")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative")
    prepared_inputs, prepared_baselines = _prepare_inputs(inputs, baselines)
    attempts: List[Mapping[str, Any]] = []
    final_attribution: Optional[Tensor] = None
    final_report: Optional[Mapping[str, Tensor]] = None
    final_steps = schedule[-1]
    with _temporarily_freeze(parameters):
        start_value = _endpoint_values(forward_fn, prepared_baselines)
        end_value = _endpoint_values(forward_fn, prepared_inputs)
        delta = end_value - start_value
        for step_count in schedule:
            attribution = _integral_for_steps(
                forward_fn,
                prepared_inputs,
                prepared_baselines,
                steps=step_count,
                step_chunk_size=step_chunk_size,
            )
            report = _conservation(attribution, delta, atol=atol, rtol=rtol)
            attempts.append(_serializable_attempt(step_count, report))
            final_attribution, final_report, final_steps = attribution, report, step_count
            if bool(report["pass"].all().item()):
                break
    assert final_attribution is not None and final_report is not None
    result_metadata: Dict[str, Any] = {
        "method": "integrated_gradients",
        "quadrature": "gauss_legendre",
        "integration_dtype": "float32",
        "input_shape": list(prepared_inputs.shape),
        "step_schedule": list(schedule),
        "step_chunk_size": step_chunk_size,
        "atol": float(atol),
        "rtol": float(rtol),
    }
    if metadata is not None:
        result_metadata.update(dict(metadata))
    return IGResult(
        attribution=final_attribution,
        conservation=final_report,
        steps=final_steps,
        converged=bool(final_report["pass"].all().item()),
        attempts=tuple(attempts),
        metadata=result_metadata,
    )


def integrated_gradients_float64_kahan(
    forward_fn: ForwardCallback,
    inputs: Tensor,
    baselines: Optional[Tensor] = None,
    *,
    steps: StepSpec,
    step_chunk_size: int = 8,
    atol: float = 1e-4,
    rtol: float = 0.01,
    parameters: Optional[Union[nn.Module, Iterable[nn.Parameter]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> IGResult:
    """Prospective float64-quadrature/Kahan IG for E3-R3 only."""

    if not callable(forward_fn):
        raise TypeError("forward_fn must be callable")
    schedule = _resolve_steps(steps)
    if (
        isinstance(step_chunk_size, bool)
        or not isinstance(step_chunk_size, int)
        or step_chunk_size <= 0
    ):
        raise ValueError("step_chunk_size must be a positive integer")
    prepared_inputs, prepared_baselines = _prepare_inputs(inputs, baselines)
    if prepared_inputs.dtype is not torch.float32:
        raise TypeError("E3-R3 model inputs must remain float32")
    attempts: List[Mapping[str, Any]] = []
    final_attribution: Optional[Tensor] = None
    final_report: Optional[Mapping[str, Tensor]] = None
    final_steps = schedule[-1]
    with _temporarily_freeze(parameters):
        start_value = _endpoint_values(forward_fn, prepared_baselines)
        end_value = _endpoint_values(forward_fn, prepared_inputs)
        delta = end_value - start_value
        for step_count in schedule:
            attribution = _integral_for_steps_float64_kahan(
                forward_fn, prepared_inputs, prepared_baselines,
                steps=step_count, step_chunk_size=step_chunk_size,
            )
            report = _conservation_float64_reduction(
                attribution, delta, atol=atol, rtol=rtol
            )
            attempts.append(_serializable_attempt(step_count, report))
            final_attribution, final_report, final_steps = (
                attribution, report, step_count
            )
            if bool(report["pass"].all().item()):
                break
    assert final_attribution is not None and final_report is not None
    result_metadata: Dict[str, Any] = {
        "method": "integrated_gradients",
        "quadrature": "gauss_legendre_scipy_roots_legendre",
        "model_compute_dtype": "float32",
        "quadrature_dtype": "float64",
        "accumulation_dtype": "float64",
        "accumulation_algorithm": "kahan_per_chunk",
        "stored_attribution_dtype": "float32",
        "conservation_reduction_dtype": "float64",
        "input_shape": list(prepared_inputs.shape),
        "step_schedule": list(schedule),
        "step_chunk_size": step_chunk_size,
        "atol": float(atol),
        "rtol": float(rtol),
    }
    if metadata is not None:
        result_metadata.update(dict(metadata))
    return IGResult(
        attribution=final_attribution, conservation=final_report,
        steps=final_steps, converged=bool(final_report["pass"].all().item()),
        attempts=tuple(attempts), metadata=result_metadata,
    )


def _resolve_attention(model: nn.Module, block_index: int) -> nn.Module:
    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise TypeError("block_index must be an integer")
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model must expose timm-style model.blocks")
    try:
        block = blocks[block_index]
    except (IndexError, TypeError) as exc:
        raise IndexError(f"invalid attention block_index {block_index}") from exc
    attention = getattr(block, "attn", None)
    if not isinstance(attention, nn.Module):
        raise AttributeError(f"model.blocks[{block_index}] must expose an attn module")
    if not isinstance(getattr(attention, "attn_drop", None), nn.Module):
        raise AttributeError(
            f"model.blocks[{block_index}].attn must expose an attn_drop module"
        )
    return attention


class _UnfusedMode:
    def __init__(self, attention: nn.Module) -> None:
        self.attention = attention
        self.had_attribute = hasattr(attention, "fused_attn")
        self.previous: Any = None

    def __enter__(self) -> None:
        if self.had_attribute:
            self.previous = getattr(self.attention, "fused_attn")
            setattr(self.attention, "fused_attn", False)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.had_attribute:
            setattr(self.attention, "fused_attn", self.previous)


class AttentionCapture:
    """Context manager capturing real unfused ``attn_drop`` output."""

    def __init__(self, model: nn.Module, block_index: int) -> None:
        self.attention_module = _resolve_attention(model, block_index)
        self.block_index = block_index
        self.call_count = 0
        self._attention: Optional[Tensor] = None
        self._handle: Any = None
        self._unfused = _UnfusedMode(self.attention_module)

    @property
    def attention(self) -> Tensor:
        if self._attention is None:
            raise RuntimeError("no attention was captured; the block may not have run")
        return self._attention

    def _hook(self, module: nn.Module, arguments: Tuple[Any, ...], output: Any) -> None:
        self.call_count += 1
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise RuntimeError(
                "attn_drop must return attention shaped [B, heads, queries, keys]"
            )
        self._attention = output.detach().clone()

    def __enter__(self) -> "AttentionCapture":
        if self._handle is not None:
            raise RuntimeError("AttentionCapture cannot be entered twice")
        self._unfused.__enter__()
        try:
            self._handle = self.attention_module.attn_drop.register_forward_hook(self._hook)
        except Exception:
            self._unfused.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._unfused.__exit__(exc_type, exc, traceback)


def capture_unfused_attention(model: nn.Module, block_index: int) -> AttentionCapture:
    return AttentionCapture(model, block_index)


class FixedAttentionIntervention:
    """Replace ``attn_drop`` output with a detached endpoint matrix."""

    def __init__(self, model: nn.Module, block_index: int, fixed_attention: Tensor) -> None:
        self.attention_module = _resolve_attention(model, block_index)
        if not isinstance(fixed_attention, Tensor):
            raise TypeError("fixed_attention must be a Tensor")
        if fixed_attention.ndim != 4:
            raise ValueError(
                "fixed_attention must have shape [B, heads, queries, keys], "
                f"got {tuple(fixed_attention.shape)}"
            )
        if not fixed_attention.is_floating_point():
            raise TypeError("fixed_attention must be floating point")
        if not torch.isfinite(fixed_attention).all():
            raise ValueError("fixed_attention must contain only finite values")
        self.fixed_attention = fixed_attention.detach().clone()
        self.block_index = block_index
        self.call_count = 0
        self._handle: Any = None
        self._unfused = _UnfusedMode(self.attention_module)

    def _expanded(self, output: Tensor) -> Tensor:
        fixed = self.fixed_attention
        if output.device != fixed.device:
            raise RuntimeError(
                f"fixed attention device {fixed.device} does not match attn_drop output {output.device}"
            )
        if output.dtype != fixed.dtype:
            raise RuntimeError(
                f"fixed attention dtype {fixed.dtype} does not match attn_drop output {output.dtype}"
            )
        if output.shape[1:] != fixed.shape[1:]:
            raise RuntimeError(
                "fixed attention non-batch shape mismatch: "
                f"fixed={tuple(fixed.shape)}, actual={tuple(output.shape)}"
            )
        actual_batch = output.shape[0]
        fixed_batch = fixed.shape[0]
        if fixed_batch <= 0 or actual_batch % fixed_batch != 0:
            raise RuntimeError(
                "attn_drop batch must be an integer multiple of fixed attention batch; "
                f"fixed={fixed_batch}, actual={actual_batch}"
            )
        repeats = actual_batch // fixed_batch
        return fixed if repeats == 1 else fixed.repeat(repeats, 1, 1, 1)

    def _hook(self, module: nn.Module, arguments: Tuple[Any, ...], output: Any) -> Tensor:
        self.call_count += 1
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise RuntimeError(
                "attn_drop must return attention shaped [B, heads, queries, keys]"
            )
        return self._expanded(output).detach()

    def __enter__(self) -> "FixedAttentionIntervention":
        if self._handle is not None:
            raise RuntimeError("FixedAttentionIntervention cannot be entered twice")
        self._unfused.__enter__()
        try:
            self._handle = self.attention_module.attn_drop.register_forward_hook(self._hook)
        except Exception:
            self._unfused.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._unfused.__exit__(exc_type, exc, traceback)


def fixed_attention_intervention(
    model: nn.Module, block_index: int, fixed_attention: Tensor
) -> FixedAttentionIntervention:
    return FixedAttentionIntervention(model, block_index, fixed_attention)


class _PreProjectionCapture:
    def __init__(self, attention: nn.Module) -> None:
        projection = getattr(attention, "proj", None)
        if not isinstance(projection, nn.Module):
            raise AttributeError("timm-style attention must expose a proj module")
        self.projection = projection
        self.call_count = 0
        self.latest: Optional[Tensor] = None
        self._handle: Any = None

    def _hook(self, module: nn.Module, arguments: Tuple[Any, ...]) -> None:
        self.call_count += 1
        if len(arguments) != 1 or not isinstance(arguments[0], Tensor):
            raise RuntimeError("attention proj must receive one Tensor")
        value = arguments[0]
        if value.ndim != 3:
            raise RuntimeError(
                f"attention pre-projection O must have shape [B, tokens, channels], got {tuple(value.shape)}"
            )
        self.latest = value

    def __enter__(self) -> "_PreProjectionCapture":
        self._handle = self.projection.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def scalar(
        self, *, before_calls: int, num_heads: int, head: int, query_token: int, head_channel: int
    ) -> Tensor:
        if self.call_count != before_calls + 1 or self.latest is None:
            raise RuntimeError(
                "the selected attention projection must run exactly once per callback forward"
            )
        value = self.latest
        batch, tokens, channels = value.shape
        if num_heads <= 0 or channels % num_heads != 0:
            raise RuntimeError(
                f"pre-projection channels {channels} are not divisible by num_heads={num_heads}"
            )
        head_width = channels // num_heads
        if not 0 <= head < num_heads:
            raise IndexError(f"head {head} is out of range [0, {num_heads})")
        if not 0 <= query_token < tokens:
            raise IndexError(f"query_token {query_token} is out of range [0, {tokens})")
        if not 0 <= head_channel < head_width:
            raise IndexError(f"head_channel {head_channel} is out of range [0, {head_width})")
        return value.reshape(batch, tokens, num_heads, head_width)[
            :, query_token, head, head_channel
        ]


def _capture_endpoint_attention(model: nn.Module, inputs: Tensor, block_index: int) -> Tuple[Tensor, int]:
    with capture_unfused_attention(model, block_index) as capture:
        with torch.no_grad():
            model(inputs)
    if capture.call_count != 1:
        raise RuntimeError(
            f"expected selected attn_drop once during endpoint capture, got {capture.call_count}"
        )
    return capture.attention, capture.call_count


def _require_tensor_model_output(output: Any) -> Tensor:
    if not isinstance(output, Tensor):
        raise TypeError("model must return a Tensor")
    return output


def full_model_ig(
    model: nn.Module,
    inputs: Tensor,
    baselines: Optional[Tensor],
    target: Union[TargetSpec, Mapping[str, Any]],
    *,
    steps: StepSpec = "default",
    step_chunk_size: int = 16,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> IGResult:
    """Attribute a final raw class logit through the unmodified model."""

    target_spec = parse_target(target)

    def callback(value: Tensor) -> Tensor:
        return target_spec.select(_require_tensor_model_output(model(value)))

    with _temporarily_eval(model):
        return integrated_gradients(
            callback,
            inputs,
            baselines,
            steps=steps,
            step_chunk_size=step_chunk_size,
            atol=atol,
            rtol=rtol,
            parameters=model,
            metadata={
                "scope": "full_model",
                "target": target_spec.to_dict(),
                "includes_residual_paths": True,
            },
        )


def conditional_fixed_attention_model_ig(
    model: nn.Module,
    inputs: Tensor,
    baselines: Optional[Tensor],
    target: Union[TargetSpec, Mapping[str, Any]],
    *,
    block_index: int,
    fixed_attention: Optional[Tensor] = None,
    steps: StepSpec = "default",
    step_chunk_size: int = 16,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> IGResult:
    """Attribute a class logit while fixing one attention endpoint.

    This remains a full-model attribution: residual and all other live paths
    contribute.  It is therefore deliberately not named a Value-path IG.
    """

    target_spec = parse_target(target)
    prepared_inputs, _ = _prepare_inputs(inputs, baselines)
    with _temporarily_eval(model):
        if fixed_attention is None:
            endpoint_attention, _ = _capture_endpoint_attention(
                model, prepared_inputs, block_index
            )
        else:
            endpoint_attention = fixed_attention
        with fixed_attention_intervention(
            model, block_index, endpoint_attention
        ) as intervention:

            def callback(value: Tensor) -> Tensor:
                return target_spec.select(_require_tensor_model_output(model(value)))

            result = integrated_gradients(
                callback,
                inputs,
                baselines,
                steps=steps,
                step_chunk_size=step_chunk_size,
                atol=atol,
                rtol=rtol,
                parameters=model,
                metadata={
                    "scope": "fixed_attention_model",
                    "target": target_spec.to_dict(),
                    "block_index": block_index,
                    "fixed_attention_shape": list(endpoint_attention.shape),
                    "includes_residual_paths": True,
                    "is_complete_attention_attribution": False,
                },
            )
            calls = intervention.call_count
    return replace(result, metadata={**result.metadata, "intervention_calls": calls})


def conditional_fixed_attention_value_path_ig(
    model: nn.Module,
    inputs: Tensor,
    baselines: Optional[Tensor] = None,
    *,
    block_index: int,
    head: int,
    query_token: int,
    head_channel: int,
    fixed_attention: Optional[Tensor] = None,
    steps: StepSpec = "default",
    step_chunk_size: int = 16,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> IGResult:
    """Attribute one internal fixed-``A`` Value-path scalar.

    The target is exactly
    ``pre_proj_O[B, query_token, head, head_channel]`` where ``O = A @ V``
    inside ``model.blocks[block_index].attn``.  The detached fixed ``A`` cuts
    Q/K gradients, and targeting pre-projection ``O`` excludes the block
    residual and later model paths.  Conservation ``delta`` is the endpoint
    difference of this internal scalar, never a class-logit difference.
    """

    for name, value in (
        ("head", head),
        ("query_token", query_token),
        ("head_channel", head_channel),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    prepared_inputs, _ = _prepare_inputs(inputs, baselines)
    attention_module = _resolve_attention(model, block_index)
    num_heads = getattr(attention_module, "num_heads", None)
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads <= 0:
        raise AttributeError("selected timm-style attention must expose positive integer num_heads")
    with _temporarily_eval(model):
        if fixed_attention is None:
            endpoint_attention, capture_calls = _capture_endpoint_attention(
                model, prepared_inputs, block_index
            )
        else:
            endpoint_attention = fixed_attention
            capture_calls = 0
        with fixed_attention_intervention(
            model, block_index, endpoint_attention
        ) as intervention, _PreProjectionCapture(attention_module) as pre_projection:

            def callback(value: Tensor) -> Tensor:
                before_intervention_calls = intervention.call_count
                before_calls = pre_projection.call_count
                model(value)
                if intervention.call_count != before_intervention_calls + 1:
                    raise RuntimeError(
                        "the selected attn_drop must run exactly once per callback forward"
                    )
                return pre_projection.scalar(
                    before_calls=before_calls,
                    num_heads=num_heads,
                    head=head,
                    query_token=query_token,
                    head_channel=head_channel,
                )

            result = integrated_gradients(
                callback,
                inputs,
                baselines,
                steps=steps,
                step_chunk_size=step_chunk_size,
                atol=atol,
                rtol=rtol,
                parameters=model,
                metadata={
                    "scope": "fixed_attention_value_path",
                    "target": {
                        "kind": "block_pre_projection_o_scalar",
                        "block_index": block_index,
                        "head": head,
                        "query_token": query_token,
                        "head_channel": head_channel,
                    },
                    "fixed_attention_shape": list(endpoint_attention.shape),
                    "capture_calls": capture_calls,
                    "includes_residual_paths": False,
                    "is_complete_attention_attribution": False,
                },
            )
            intervention_calls = intervention.call_count
            projection_calls = pre_projection.call_count
    return replace(
        result,
        metadata={
            **result.metadata,
            "intervention_calls": intervention_calls,
            "pre_projection_calls": projection_calls,
        },
    )


def fixed_attention_endpoint_parity(
    model: nn.Module,
    inputs: Tensor,
    *,
    block_index: int,
    fixed_attention: Optional[Tensor] = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> EndpointParityResult:
    """Check that replaying endpoint ``A`` reproduces endpoint output."""

    prepared_inputs, _ = _prepare_inputs(inputs, None)
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative")
    with _temporarily_eval(model), _temporarily_freeze(model):
        with capture_unfused_attention(model, block_index) as capture:
            with torch.no_grad():
                original = _require_tensor_model_output(model(prepared_inputs)).detach()
        if capture.call_count != 1:
            raise RuntimeError(
                f"expected selected attn_drop once during endpoint capture, got {capture.call_count}"
            )
        endpoint_attention = capture.attention if fixed_attention is None else fixed_attention
        with fixed_attention_intervention(
            model, block_index, endpoint_attention
        ) as intervention:
            with torch.no_grad():
                replayed = _require_tensor_model_output(model(prepared_inputs)).detach()
        if original.shape != replayed.shape:
            raise RuntimeError(
                f"model output shape changed under intervention: {tuple(original.shape)} vs {tuple(replayed.shape)}"
            )
    absolute = (original.to(torch.float32) - replayed.to(torch.float32)).abs()
    relative = absolute / original.to(torch.float32).abs().clamp_min(
        torch.finfo(torch.float32).eps
    )
    return EndpointParityResult(
        passed=bool(torch.allclose(original, replayed, atol=atol, rtol=rtol)),
        max_abs_error=float(absolute.max().item()) if absolute.numel() else 0.0,
        max_rel_error=float(relative.max().item()) if relative.numel() else 0.0,
        capture_calls=capture.call_count,
        intervention_calls=intervention.call_count,
        attention_shape=tuple(endpoint_attention.shape),
    )


# Short aliases for callers that prefer nouns while preserving the explicit
# public method names above.
capture_attention = capture_unfused_attention
intervene_fixed_attention = fixed_attention_intervention
endpoint_intervention_parity = fixed_attention_endpoint_parity


__all__ = [
    "AttentionCapture",
    "EndpointParityResult",
    "FixedAttentionIntervention",
    "IGResult",
    "IG_STEP_SCHEDULES",
    "capture_attention",
    "capture_unfused_attention",
    "conditional_fixed_attention_model_ig",
    "conditional_fixed_attention_value_path_ig",
    "endpoint_intervention_parity",
    "fixed_attention_endpoint_parity",
    "fixed_attention_intervention",
    "full_model_ig",
    "integrated_gradients",
    "intervene_fixed_attention",
]
