from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from vitnns.endpoint_nns import (
    endpoint_learning_image,
    finite_difference_gradient,
    paper_cnn_conv_relu_maxpool_residual,
    paper_relu_weight_propagation,
)


def test_unbiased_relu_paper_equations_equal_gradient_times_input() -> None:
    x = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    w1 = torch.tensor([[1.0, 2.0], [-3.0, 1.0]], dtype=torch.float64)
    w2 = torch.tensor([[4.0, -2.0]], dtype=torch.float64)
    pre = x @ w1.T

    def target(value: torch.Tensor) -> torch.Tensor:
        return torch.relu(value @ w1.T) @ w2.T

    paper = paper_relu_weight_propagation(x, (w1, w2), (pre,))
    endpoint = endpoint_learning_image(target, x)
    torch.testing.assert_close(paper, endpoint.attribution, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(paper.sum(1), target(x)[:, 0], atol=1e-12, rtol=1e-12)


def test_bias_changes_gate_but_is_not_reconstructed_by_paper_equations() -> None:
    x = torch.tensor([[2.0]], dtype=torch.float64)

    def target(value: torch.Tensor) -> torch.Tensor:
        return torch.relu(value - 1.0)

    result = endpoint_learning_image(target, x)
    torch.testing.assert_close(result.attribution, torch.tensor([[2.0]], dtype=torch.float64))
    torch.testing.assert_close(result.response_delta, torch.tensor([1.0], dtype=torch.float64))
    torch.testing.assert_close(
        result.response_reconstruction_error, torch.tensor([1.0], dtype=torch.float64)
    )


def test_relu_boundary_uses_pytorch_zero_subgradient() -> None:
    x = torch.tensor([[1.0]], dtype=torch.float64)
    result = endpoint_learning_image(lambda value: torch.relu(value - 1.0), x)
    torch.testing.assert_close(result.gradient, torch.zeros_like(x))


def test_maxpool_selects_the_endpoint_winner() -> None:
    x = torch.tensor([[[[1.0, 3.0], [2.0, -1.0]]]], dtype=torch.float64)
    result = endpoint_learning_image(lambda value: nn.functional.max_pool2d(value, 2).flatten(1)[:, 0], x)
    expected = torch.zeros_like(x)
    expected[0, 0, 0, 1] = 3.0
    torch.testing.assert_close(result.attribution, expected)


def test_paper_cnn_formula_covers_bias_closed_gate_unique_pool_and_residual() -> None:
    x = torch.tensor([[[[0.2, -0.4, 0.7], [1.1, 0.3, -0.8], [0.6, -0.2, 0.9]]]], dtype=torch.float64)
    conv_weight = torch.tensor(
        [[[[0.4, -0.2], [0.1, 0.3]]], [[[-0.3, 0.2], [0.5, -0.1]]]],
        dtype=torch.float64,
    )
    conv_bias = torch.tensor([0.15, -2.0], dtype=torch.float64)
    readout = torch.tensor([0.7, -0.4], dtype=torch.float64)
    residual = torch.linspace(-0.2, 0.2, x.numel(), dtype=torch.float64)

    def target(value):
        hidden = nn.functional.relu(nn.functional.conv2d(value, conv_weight, conv_bias))
        pooled = nn.functional.max_pool2d(hidden, 2)
        return pooled.flatten(1) @ readout[:, None] + value.flatten(1) @ residual[:, None]

    paper = paper_cnn_conv_relu_maxpool_residual(x, conv_weight, conv_bias, readout, residual)
    independent = endpoint_learning_image(target, x).attribution
    torch.testing.assert_close(paper, independent, atol=1e-12, rtol=1e-12)
    assert paper.abs().sum() > 0
    assert bool((nn.functional.conv2d(x, conv_weight, conv_bias)[0, 1] < 0).all())


def test_paper_cnn_formula_rejects_maxpool_tie() -> None:
    x = torch.ones((1, 1, 3, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="MAXPOOL_TIE_NOT_ALLOWED"):
        paper_cnn_conv_relu_maxpool_residual(
            x, torch.ones((1, 1, 2, 2), dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            torch.zeros(x.numel(), dtype=torch.float64),
        )


def test_convolution_relu_matches_finite_difference() -> None:
    convolution = nn.Conv2d(1, 2, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        convolution.weight.copy_(torch.tensor(
            [[[[0.4, -0.2], [0.1, 0.3]]], [[[0.2, 0.5], [-0.4, 0.1]]]],
            dtype=torch.float64,
        ))
    x = torch.tensor([[[[0.8, -0.1], [0.3, 0.5]]]], dtype=torch.float64)
    callback = lambda value: torch.relu(convolution(value)).flatten(1).sum(1)
    endpoint = endpoint_learning_image(callback, x, parameters=convolution)
    finite = finite_difference_gradient(callback, x)
    torch.testing.assert_close(endpoint.gradient, finite, atol=2e-9, rtol=2e-8)


def test_residual_learning_image_adds_branch_endpoint_maps() -> None:
    x = torch.tensor([[0.4, -0.7, 1.2]], dtype=torch.float64)
    branch = lambda value: torch.tanh(value).sum(1)
    skip = lambda value: (0.25 * value).sum(1)
    total = lambda value: branch(value) + skip(value)
    branch_map = endpoint_learning_image(branch, x).attribution
    skip_map = endpoint_learning_image(skip, x).attribution
    total_map = endpoint_learning_image(total, x).attribution
    torch.testing.assert_close(total_map, branch_map + skip_map, atol=1e-12, rtol=1e-12)


class PrimitiveBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4, dtype=torch.float64)
        self.fc1 = nn.Linear(4, 7, dtype=torch.float64)
        self.fc2 = nn.Linear(7, 4, dtype=torch.float64)
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * 0.2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        branch = self.fc2(nn.functional.gelu(self.fc1(self.norm(value))))
        return (value + branch) @ torch.tensor([0.2, -0.3, 0.5, 0.7], dtype=value.dtype)


def test_layernorm_gelu_mlp_residual_matches_finite_difference() -> None:
    model = PrimitiveBlock()
    x = torch.tensor([[0.3, -0.8, 1.1, 0.4]], dtype=torch.float64)
    endpoint = endpoint_learning_image(model, x, parameters=model)
    finite = finite_difference_gradient(model, x, epsilon=1e-6)
    torch.testing.assert_close(endpoint.gradient, finite, atol=2e-8, rtol=2e-7)
    torch.testing.assert_close(endpoint.attribution, x * finite, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize(
    "callback",
    [
        lambda value: nn.functional.layer_norm(value, (4,)).square().sum(1),
        lambda value: nn.functional.gelu(value).square().sum(1),
        lambda value: value.softmax(-1)[:, 0],
    ],
)
def test_nonlinear_operator_endpoint_gradient_matches_finite_difference(callback) -> None:
    x = torch.tensor([[0.3, -0.8, 1.1, 0.4]], dtype=torch.float64)
    endpoint = endpoint_learning_image(callback, x)
    finite = finite_difference_gradient(callback, x)
    torch.testing.assert_close(endpoint.gradient, finite, atol=3e-8, rtol=3e-7)


def _attention_parts(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    wq = torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=tokens.dtype)
    wk = torch.tensor([[0.2, 0.5], [-0.4, 0.1]], dtype=tokens.dtype)
    wv = torch.tensor([[0.6, -0.1], [0.2, 0.3]], dtype=tokens.dtype)
    q, k, v = tokens @ wq, tokens @ wk, tokens @ wv
    if mode == "q":
        k, v = k.detach(), v.detach()
    elif mode == "k":
        q, v = q.detach(), v.detach()
    elif mode == "v":
        q, k = q.detach(), k.detach()
    elif mode != "full":
        raise ValueError(mode)
    attention = ((q @ k.transpose(-2, -1)) / math.sqrt(2.0)).softmax(-1)
    return (attention @ v)[:, 0, 1]


def test_attention_q_k_value_endpoint_components_sum_to_full_gradient() -> None:
    x = torch.tensor([[[0.2, -0.4], [0.7, 0.1], [-0.3, 0.9]]], dtype=torch.float64)
    full = endpoint_learning_image(lambda value: _attention_parts(value, "full"), x)
    parts = [endpoint_learning_image(lambda value, key=key: _attention_parts(value, key), x) for key in ("q", "k", "v")]
    component_gradient = sum((item.gradient for item in parts), torch.zeros_like(x))
    torch.testing.assert_close(full.gradient, component_gradient, atol=1e-12, rtol=1e-12)
    finite = finite_difference_gradient(lambda value: _attention_parts(value, "full"), x, epsilon=1e-6)
    torch.testing.assert_close(full.gradient, finite, atol=2e-9, rtol=2e-8)


def test_multihead_attention_endpoint_gradient_matches_finite_difference() -> None:
    attention = nn.MultiheadAttention(4, 2, dropout=0.0, batch_first=True, dtype=torch.float64)
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * 0.15)
    x = torch.tensor(
        [[[0.2, -0.4, 0.1, 0.8], [0.7, 0.1, -0.2, 0.3], [-0.3, 0.9, 0.5, -0.6]]],
        dtype=torch.float64,
    )

    def callback(value: torch.Tensor) -> torch.Tensor:
        output, _ = attention(value, value, value, need_weights=False)
        return output[:, 0, 2] + 0.3 * output[:, 2, 1]

    endpoint = endpoint_learning_image(callback, x, parameters=attention)
    finite = finite_difference_gradient(callback, x)
    torch.testing.assert_close(endpoint.gradient, finite, atol=3e-8, rtol=3e-7)


def test_reference_changes_multiplier_not_endpoint_gradient() -> None:
    x = torch.tensor([[0.6, -0.2]], dtype=torch.float64)
    baseline = torch.full_like(x, 0.1)
    callback = lambda value: torch.sin(value).sum(1)
    paper = endpoint_learning_image(callback, x)
    relative = endpoint_learning_image(callback, x, reference=baseline)
    torch.testing.assert_close(paper.gradient, relative.gradient)
    torch.testing.assert_close(relative.multiplier, x - baseline)


def test_invalid_callback_shape_fails_closed() -> None:
    with pytest.raises(ValueError, match="one scalar per sample"):
        endpoint_learning_image(lambda value: value, torch.ones(2, 3, dtype=torch.float64))
