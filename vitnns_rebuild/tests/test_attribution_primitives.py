from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from vitnns.attribution import integrated_gradients
from vitnns.targets import CLASS_LOGIT, TargetSpec, parse_target


def _assert_conserved(result, *, atol: float = 2e-5) -> None:
    assert result.attribution.dtype == torch.float32
    assert result.attribution.ndim == 4
    assert set(result.conservation) == {"delta", "sum", "abs", "rel", "pass"}
    torch.testing.assert_close(
        result.conservation["sum"], result.conservation["delta"], atol=atol, rtol=atol
    )
    assert result.converged
    assert bool(result.conservation["pass"].all())
    json.dumps(result.to_dict())


def test_class_logit_target_is_strict_and_serializable() -> None:
    target = TargetSpec.class_logit(2)
    assert target.to_dict() == {"kind": CLASS_LOGIT, "class_index": 2}
    assert TargetSpec.from_dict(target.to_dict()) == target
    assert parse_target(target.to_dict()) == target
    logits = torch.tensor([[0.0, 1.0, 2.0], [5.0, 4.0, 3.0]])
    torch.testing.assert_close(target.select(logits), torch.tensor([2.0, 3.0]))
    with pytest.raises(ValueError, match="only 'class_logit'"):
        TargetSpec(class_index=0, kind="probability")
    with pytest.raises(ValueError, match="extra"):
        TargetSpec.from_dict({"kind": CLASS_LOGIT, "class_index": 0, "temperature": 1})
    with pytest.raises(ValueError, match=r"requires \[B, classes\]"):
        target.select(torch.ones(3))


def test_linear_ig_is_exact_and_keeps_signed_rgb_channels() -> None:
    inputs = torch.tensor(
        [[[[1.0, -2.0]], [[3.0, 4.0]], [[-1.0, 2.0]]]], dtype=torch.float64
    )
    baselines = torch.zeros_like(inputs)
    weights = torch.tensor(
        [[[[2.0, 1.0]], [[0.5, 3.0]], [[-2.0, 4.0]]]], dtype=torch.float64
    )

    result = integrated_gradients(
        lambda value: (value * weights).flatten(1).sum(1),
        inputs,
        baselines,
        steps=3,
        step_chunk_size=2,
        atol=1e-6,
        rtol=1e-6,
    )

    torch.testing.assert_close(result.attribution, (inputs * weights).float(), atol=1e-6, rtol=1e-6)
    assert result.attribution.shape == inputs.shape
    assert (result.attribution < 0).any()
    _assert_conserved(result, atol=1e-6)


class PrimitiveStack(nn.Module):
    """Linear + LayerNorm + GELU + explicit residual."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(6, 6, bias=True, dtype=torch.float64)
        self.norm = nn.LayerNorm(6, dtype=torch.float64)
        self.act = nn.GELU()
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(
                [
                    [0.2, -0.1, 0.3, 0.0, 0.1, -0.2],
                    [0.0, 0.4, -0.2, 0.1, 0.2, 0.1],
                    [-0.3, 0.1, 0.2, 0.2, -0.1, 0.4],
                    [0.1, 0.0, 0.1, 0.3, 0.2, -0.2],
                    [0.2, 0.1, -0.1, -0.2, 0.3, 0.1],
                    [-0.1, 0.3, 0.2, 0.1, 0.0, 0.2],
                ], dtype=torch.float64
            ))
            self.linear.bias.copy_(torch.linspace(-0.1, 0.1, 6, dtype=torch.float64))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        flat = image.flatten(1)
        transformed = self.act(self.norm(self.linear(flat)))
        return (transformed + 0.35 * flat) @ torch.tensor(
            [0.7, -0.2, 0.4, 0.1, -0.5, 0.3], device=image.device, dtype=image.dtype
        )


def test_layernorm_gelu_residual_schedule_converges_and_restores_parameters() -> None:
    model = PrimitiveStack()
    inputs = torch.tensor(
        [[[[0.2, -0.7]], [[1.1, 0.3]], [[-0.5, 0.9]]]], dtype=torch.float64
    )
    baselines = torch.full_like(inputs, -0.15)
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]

    result = integrated_gradients(
        model,
        inputs,
        baselines,
        steps=(4, 8, 16, 32, 64),
        step_chunk_size=5,
        atol=3e-5,
        rtol=3e-5,
        parameters=model,
    )

    _assert_conserved(result, atol=3e-5)
    assert result.steps in (4, 8, 16, 32, 64)
    assert [attempt["steps"] for attempt in result.attempts] == [4, 8, 16, 32, 64][: len(result.attempts)]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert [parameter.requires_grad for parameter in model.parameters()] == original_requires_grad


class CompleteQKSoftmaxV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(3, 4, bias=False, dtype=torch.float64)
        self.k = nn.Linear(3, 4, bias=False, dtype=torch.float64)
        self.v = nn.Linear(3, 4, bias=False, dtype=torch.float64)
        generator = torch.Generator().manual_seed(4)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * 0.25)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # RGB pixels are tokens; this is the full differentiable Q/K-softmax-V path.
        tokens = image.permute(0, 2, 3, 1).reshape(image.shape[0], -1, 3)
        query, key, value = self.q(tokens), self.k(tokens), self.v(tokens)
        attention = (query @ key.transpose(-2, -1) / 2.0).softmax(dim=-1)
        output = attention @ value
        return output[:, 0, 1] + 0.2 * output[:, -1, 3]


def test_complete_qk_softmax_v_attention_conserves_rgb_sum() -> None:
    model = CompleteQKSoftmaxV()
    inputs = torch.tensor(
        [
            [
                [[0.2, 0.8, -0.3]],
                [[-0.6, 0.1, 0.5]],
                [[0.4, -0.2, 0.9]],
            ]
        ],
        dtype=torch.float64,
    )
    baseline = torch.zeros_like(inputs)

    result = integrated_gradients(
        model,
        inputs,
        baseline,
        steps=(8, 16, 32, 64),
        step_chunk_size=7,
        atol=3e-5,
        rtol=3e-5,
        parameters=model,
    )

    _assert_conserved(result, atol=3e-5)
    rgb_sum = result.attribution.sum(dim=(1, 2, 3))
    torch.testing.assert_close(rgb_sum, result.conservation["delta"], atol=3e-5, rtol=3e-5)


def test_fixed_attention_value_primitive_has_no_qk_dependence() -> None:
    inputs = torch.tensor(
        [[[[0.4, -0.2]], [[0.8, 0.1]], [[-0.3, 0.5]]]], dtype=torch.float64
    )
    fixed_attention = torch.tensor(
        [[[0.7, 0.3], [0.25, 0.75]]], dtype=torch.float64
    ).detach()
    value_weight = torch.tensor(
        [[0.2, -0.1], [0.4, 0.3], [-0.5, 0.7]], dtype=torch.float64
    )

    def fixed_value_path(image: torch.Tensor) -> torch.Tensor:
        tokens = image.permute(0, 2, 3, 1).reshape(image.shape[0], 2, 3)
        values = tokens @ value_weight
        repeated_attention = fixed_attention.repeat(image.shape[0], 1, 1)
        return (repeated_attention @ values)[:, 0, 1]

    result = integrated_gradients(
        fixed_value_path, inputs, torch.zeros_like(inputs), steps=4, atol=1e-6, rtol=1e-6
    )
    _assert_conserved(result, atol=1e-6)


def test_invalid_step_schedules_fail_closed() -> None:
    image = torch.ones(1, 1, 1, 1)
    callback = lambda value: value.flatten(1).sum(1)
    for invalid in (0, (), (8, 8), (16, 8), "missing"):
        with pytest.raises((TypeError, ValueError)):
            integrated_gradients(callback, image, steps=invalid)
