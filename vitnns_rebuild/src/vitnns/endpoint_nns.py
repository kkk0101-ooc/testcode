"""Endpoint learning images for the paper-formula NNS family.

This module intentionally does not call endpoint attribution Integrated
Gradients.  It evaluates a scalar target once at the endpoint and multiplies
the resulting input gradient by either the input or an explicitly supplied
input difference.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch
from torch import nn
from torch.nn import functional as F

Tensor = torch.Tensor
ScalarCallback = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class EndpointLearningImage:
    """One endpoint-Jacobian learning image and its explicit accounting."""

    attribution: Tensor
    gradient: Tensor
    multiplier: Tensor
    target: Tensor
    attribution_sum: Tensor
    response_delta: Tensor | None
    response_reconstruction_error: Tensor | None
    reference_mode: str


def _one_scalar_per_sample(output: Tensor, batch: int) -> Tensor:
    if not isinstance(output, Tensor):
        raise TypeError("callback must return a Tensor")
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 1 or output.shape[0] != batch:
        raise ValueError(
            "callback must return one scalar per sample; "
            f"expected [{batch}], got {tuple(output.shape)}"
        )
    if not output.is_floating_point() or not bool(torch.isfinite(output).all()):
        raise ValueError("callback output must be finite floating point")
    return output


@contextmanager
def _frozen_eval(
    parameters: nn.Module | Iterable[nn.Parameter] | None,
) -> Iterator[None]:
    modules = [] if not isinstance(parameters, nn.Module) else list(parameters.modules())
    parameter_list = [] if parameters is None else list(
        parameters.parameters() if isinstance(parameters, nn.Module) else parameters
    )
    training = [module.training for module in modules]
    requires_grad = [parameter.requires_grad for parameter in parameter_list]
    try:
        for module in modules:
            module.training = False
        for parameter in parameter_list:
            parameter.requires_grad_(False)
        yield
    finally:
        for module, state in zip(modules, training):
            module.training = state
        for parameter, state in zip(parameter_list, requires_grad):
            parameter.requires_grad_(state)


def endpoint_learning_image(
    callback: ScalarCallback,
    inputs: Tensor,
    *,
    reference: Tensor | None = None,
    parameters: nn.Module | Iterable[nn.Parameter] | None = None,
) -> EndpointLearningImage:
    """Return ``multiplier * grad(callback)(inputs)`` at the clean endpoint.

    With ``reference=None``, the multiplier is the input itself and the result
    reproduces the input initialization in the published NNS equations.  An
    explicit reference selects the baseline-relative endpoint extension; it
    changes the multiplier but never changes the endpoint where the gradient
    is evaluated.  Response reconstruction is reported as a diagnostic, not
    treated as a method axiom.
    """

    if not isinstance(inputs, Tensor) or inputs.ndim < 2:
        raise ValueError("inputs must be a rank-2-or-higher Tensor")
    if not inputs.is_floating_point() or not bool(torch.isfinite(inputs).all()):
        raise ValueError("inputs must be finite floating point")
    clean = inputs.detach().clone().requires_grad_(True)
    if reference is None:
        baseline = torch.zeros_like(clean)
        multiplier = clean.detach()
        reference_mode = "paper_input_initialization"
    else:
        if not isinstance(reference, Tensor) or not reference.is_floating_point():
            raise TypeError("reference must be a floating-point Tensor or None")
        try:
            baseline = torch.broadcast_to(
                reference.detach().to(device=clean.device, dtype=clean.dtype), clean.shape
            ).clone()
        except RuntimeError as error:
            raise ValueError("reference cannot broadcast to the input shape") from error
        if not bool(torch.isfinite(baseline).all()):
            raise ValueError("reference must be finite")
        multiplier = clean.detach() - baseline
        reference_mode = "baseline_relative_endpoint_extension"

    with _frozen_eval(parameters), torch.enable_grad():
        target = _one_scalar_per_sample(callback(clean), clean.shape[0])
        gradient = torch.autograd.grad(target.sum(), clean, create_graph=False)[0]
        attribution = multiplier * gradient
        with torch.no_grad():
            baseline_target = _one_scalar_per_sample(callback(baseline), clean.shape[0])

    dimensions = tuple(range(1, attribution.ndim))
    attribution_sum = attribution.sum(dim=dimensions)
    response_delta = target.detach() - baseline_target.detach()
    reconstruction_error = attribution_sum - response_delta
    return EndpointLearningImage(
        attribution=attribution.detach(),
        gradient=gradient.detach(),
        multiplier=multiplier.detach(),
        target=target.detach(),
        attribution_sum=attribution_sum.detach(),
        response_delta=response_delta,
        response_reconstruction_error=reconstruction_error.detach(),
        reference_mode=reference_mode,
    )


def paper_relu_weight_propagation(
    inputs: Tensor,
    weights: tuple[Tensor, ...],
    preactivations: tuple[Tensor, ...],
) -> Tensor:
    """Reproduce Eqs. 9--13 for a scalar-output feed-forward ReLU net.

    Biases affect the supplied preactivations and therefore the endpoint gates,
    but the paper's propagation matrices contain only weights.  The returned
    tensor has shape ``[batch, input_features]``.
    """

    if len(weights) == 0 or len(preactivations) != len(weights) - 1:
        raise ValueError("one preactivation is required for each hidden ReLU layer")
    if inputs.ndim != 2:
        raise ValueError("paper reproduction expects [batch,input_features]")
    coefficient = weights[0].unsqueeze(0).expand(inputs.shape[0], -1, -1)
    for layer, preactivation in zip(weights[1:], preactivations):
        if preactivation.ndim != 2 or preactivation.shape[0] != inputs.shape[0]:
            raise ValueError("preactivation batch or rank mismatch")
        gate = (preactivation > 0).to(dtype=inputs.dtype)
        coefficient = torch.einsum("oi,bi,bij->boj", layer, gate, coefficient)
    if coefficient.shape[1] != 1:
        raise ValueError("paper reproduction currently requires one scalar output")
    return inputs * coefficient[:, 0]


def finite_difference_gradient(
    callback: ScalarCallback, inputs: Tensor, *, epsilon: float = 1e-6
) -> Tensor:
    """Central finite-difference gradient for small validation fixtures."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    result = torch.empty_like(inputs)
    flat = result.reshape(inputs.shape[0], -1)
    for index in range(flat.shape[1]):
        plus = inputs.detach().clone().reshape(inputs.shape[0], -1)
        minus = plus.clone()
        plus[:, index] += epsilon
        minus[:, index] -= epsilon
        with torch.no_grad():
            y_plus = _one_scalar_per_sample(callback(plus.reshape_as(inputs)), inputs.shape[0])
            y_minus = _one_scalar_per_sample(callback(minus.reshape_as(inputs)), inputs.shape[0])
        flat[:, index] = (y_plus - y_minus) / (2.0 * epsilon)
    return result


def paper_cnn_conv_relu_maxpool_residual(
    inputs: Tensor,
    conv_weight: Tensor,
    conv_bias: Tensor,
    readout_weight: Tensor,
    residual_weight: Tensor,
    *,
    pool_kernel: int = 2,
) -> Tensor:
    """Explicit paper-formula endpoint map for the EP1 CNN fixture.

    The graph is ``conv -> ReLU -> non-overlapping MaxPool -> affine readout``
    plus a direct linear residual from the flattened input. Biases affect gates
    and winners but receive no input-coordinate seed. Tied MaxPool winners are
    rejected because this regression contract requires a unique winner.
    """
    if inputs.ndim != 4 or inputs.shape[0] != 1 or not inputs.is_floating_point():
        raise ValueError("paper CNN fixture expects one floating-point image")
    if conv_weight.ndim != 4 or conv_bias.shape != (conv_weight.shape[0],):
        raise ValueError("invalid convolution parameters")
    if conv_weight.shape[1] != inputs.shape[1] or pool_kernel <= 0:
        raise ValueError("convolution/input or pool geometry mismatch")
    with torch.no_grad():
        pre = F.conv2d(inputs, conv_weight, conv_bias)
        active = pre > 0
        height, width = pre.shape[-2:]
        if height % pool_kernel or width % pool_kernel:
            raise ValueError("pool must tile the convolution output")
        pooled_h, pooled_w = height // pool_kernel, width // pool_kernel
        expected_readout = conv_weight.shape[0] * pooled_h * pooled_w
        if readout_weight.numel() != expected_readout:
            raise ValueError("readout weight does not match pooled output")
        if residual_weight.numel() != inputs.numel():
            raise ValueError("residual weight does not match input")

        coefficient = residual_weight.reshape_as(inputs).clone()
        readout = readout_weight.reshape(conv_weight.shape[0], pooled_h, pooled_w)
        for channel in range(conv_weight.shape[0]):
            for pool_y in range(pooled_h):
                for pool_x in range(pooled_w):
                    y0, x0 = pool_y * pool_kernel, pool_x * pool_kernel
                    window = pre[0, channel, y0:y0 + pool_kernel, x0:x0 + pool_kernel]
                    maximum = window.max()
                    winners = (window == maximum).nonzero(as_tuple=False)
                    if winners.shape[0] != 1:
                        raise ValueError("MAXPOOL_TIE_NOT_ALLOWED")
                    wy, wx = (int(v) for v in winners[0])
                    out_y, out_x = y0 + wy, x0 + wx
                    if not bool(active[0, channel, out_y, out_x]):
                        continue
                    scale = readout[channel, pool_y, pool_x]
                    kh, kw = conv_weight.shape[-2:]
                    coefficient[0, :, out_y:out_y + kh, out_x:out_x + kw] += (
                        scale * conv_weight[channel]
                    )
        return inputs.detach() * coefficient
