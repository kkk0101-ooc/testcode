from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from vitnns.attribution import (
    capture_unfused_attention,
    conditional_fixed_attention_model_ig,
    conditional_fixed_attention_value_path_ig,
    fixed_attention_endpoint_parity,
    fixed_attention_intervention,
)
from vitnns.targets import TargetSpec


class ToyTimmAttention(nn.Module):
    """Small, unfused timm-shaped QK-softmax-V attention module."""

    def __init__(self, dim: int = 4, num_heads: int = 2) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True
        self.qkv = nn.Linear(dim, dim * 3, bias=False, dtype=torch.float64)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, bias=False, dtype=torch.float64)
        generator = torch.Generator().manual_seed(123)
        with torch.no_grad():
            self.qkv.weight.copy_(
                torch.randn(self.qkv.weight.shape, generator=generator, dtype=torch.float64) * 0.3
            )
            self.proj.weight.copy_(
                torch.randn(self.proj.weight.shape, generator=generator, dtype=torch.float64) * 0.25
            )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, token_count, channels = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, token_count, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch, token_count, channels)
        return self.proj(output)


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4, dtype=torch.float64)
        self.attn = ToyTimmAttention()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.attn(self.norm(tokens))


class ToyTimmModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock()])
        self.head = nn.Linear(4, 3, bias=True, dtype=torch.float64)
        generator = torch.Generator().manual_seed(321)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.randn(self.head.weight.shape, generator=generator, dtype=torch.float64) * 0.2
            )
            self.head.bias.copy_(torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = image.permute(0, 2, 3, 1).reshape(image.shape[0], -1, 4)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens.mean(dim=1))


@pytest.fixture
def model() -> ToyTimmModel:
    return ToyTimmModel().eval()


@pytest.fixture
def image() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[0.2, -0.4, 0.7]],
                [[0.8, 0.1, -0.3]],
                [[-0.5, 0.9, 0.4]],
                [[0.3, -0.2, 0.6]],
            ]
        ],
        dtype=torch.float64,
    )


def _capture(model: ToyTimmModel, image: torch.Tensor) -> torch.Tensor:
    with capture_unfused_attention(model, 0) as capture:
        model(image)
    assert capture.call_count == 1
    assert capture.attention.shape == (1, 2, 3, 3)
    assert capture.attention.grad_fn is None
    return capture.attention


def test_capture_real_attn_drop_output_forces_unfused_then_restores(
    model: ToyTimmModel, image: torch.Tensor
) -> None:
    attn = model.blocks[0].attn
    assert attn.fused_attn is True
    hook_count = len(attn.attn_drop._forward_hooks)
    captured = _capture(model, image)
    assert attn.fused_attn is True
    assert len(attn.attn_drop._forward_hooks) == hook_count
    torch.testing.assert_close(captured.sum(dim=-1), torch.ones(1, 2, 3, dtype=torch.float64))


def test_endpoint_intervention_parity_and_hook_cleanup(
    model: ToyTimmModel, image: torch.Tensor
) -> None:
    attn_drop = model.blocks[0].attn.attn_drop
    initial_hooks = len(attn_drop._forward_hooks)
    result = fixed_attention_endpoint_parity(
        model, image, block_index=0, atol=1e-12, rtol=1e-12
    )
    assert result.passed
    assert result.max_abs_error <= 1e-12
    assert result.capture_calls == 1
    assert result.intervention_calls == 1
    assert result.attention_shape == (1, 2, 3, 3)
    assert result.to_dict()["attention_shape"] == [1, 2, 3, 3]
    assert len(attn_drop._forward_hooks) == initial_hooks
    assert model.blocks[0].attn.fused_attn is True


def test_fixed_attention_cuts_qk_gradient_but_preserves_v_path(
    model: ToyTimmModel, image: torch.Tensor
) -> None:
    endpoint = _capture(model, image)
    model.zero_grad(set_to_none=True)
    probe = image.detach().clone().requires_grad_(True)
    with fixed_attention_intervention(model, 0, endpoint) as intervention:
        logits = model(probe)
        logits[:, 1].sum().backward()
    gradient = model.blocks[0].attn.qkv.weight.grad
    assert gradient is not None
    dim = 4
    torch.testing.assert_close(gradient[: 2 * dim], torch.zeros_like(gradient[: 2 * dim]))
    assert gradient[2 * dim :].abs().sum() > 0
    assert probe.grad is not None and probe.grad.abs().sum() > 0
    assert intervention.call_count == 1
    assert len(model.blocks[0].attn.attn_drop._forward_hooks) == 0


@pytest.mark.parametrize(
    "transform, message",
    [
        (lambda value: value[:, :, :, :-1], "non-batch shape mismatch"),
        (lambda value: value.float(), "dtype"),
        (lambda value: value.repeat(2, 1, 1, 1), "integer multiple"),
    ],
)
def test_intervention_wrong_shape_or_dtype_fails_closed_and_cleans_hook(
    model: ToyTimmModel, image: torch.Tensor, transform, message: str
) -> None:
    endpoint = transform(_capture(model, image))
    attn_drop = model.blocks[0].attn.attn_drop
    initial_hooks = len(attn_drop._forward_hooks)
    with pytest.raises(RuntimeError, match=message):
        with fixed_attention_intervention(model, 0, endpoint):
            model(image)
    assert len(attn_drop._forward_hooks) == initial_hooks
    assert model.blocks[0].attn.fused_attn is True


def test_intervention_expands_endpoint_batch_for_ig_chunks(
    model: ToyTimmModel, image: torch.Tensor
) -> None:
    endpoint = _capture(model, image)
    repeated_images = image.repeat(5, 1, 1, 1)
    with fixed_attention_intervention(model, 0, endpoint) as intervention:
        output = model(repeated_images)
    assert output.shape == (5, 3)
    assert intervention.call_count == 1


def test_value_path_ig_targets_internal_preproj_scalar_and_conserves(
    model: ToyTimmModel, image: torch.Tensor
) -> None:
    baseline = torch.zeros_like(image)
    result = conditional_fixed_attention_value_path_ig(
        model,
        image,
        baseline,
        block_index=0,
        head=1,
        query_token=0,
        head_channel=1,
        steps=(16, 32, 64),
        step_chunk_size=3,
        atol=2e-5,
        rtol=2e-5,
    )

    assert result.converged
    assert result.attribution.shape == image.shape
    assert result.attribution.dtype == torch.float32
    torch.testing.assert_close(
        result.conservation["sum"], result.conservation["delta"], atol=2e-5, rtol=2e-5
    )
    assert result.metadata["scope"] == "fixed_attention_value_path"
    assert result.metadata["target"] == {
        "kind": "block_pre_projection_o_scalar",
        "block_index": 0,
        "head": 1,
        "query_token": 0,
        "head_channel": 1,
    }
    assert result.metadata["includes_residual_paths"] is False
    assert result.metadata["is_complete_attention_attribution"] is False


def test_fixed_endpoint_value_path_is_numerically_isolated_from_qk_rows(
    model: ToyTimmModel, image: torch.Tensor
) -> None:
    """With A* fixed, only the selected block's V rows may affect O=A*@V."""

    baseline = torch.zeros_like(image)
    endpoint_attention = _capture(model, image)
    qk_changed = deepcopy(model)
    v_changed = deepcopy(model)
    original_weight = model.blocks[0].attn.qkv.weight.detach().clone()
    channels = model.blocks[0].attn.qkv.in_features

    generator = torch.Generator().manual_seed(20260820)
    with torch.no_grad():
        qk_weight = qk_changed.blocks[0].attn.qkv.weight
        qk_weight[: 2 * channels].add_(
            torch.randn(
                qk_weight[: 2 * channels].shape,
                generator=generator,
                dtype=qk_weight.dtype,
                device=qk_weight.device,
            )
            * 0.75
        )
        # Scaling only V makes the expected path effect especially direct:
        # both the internal target delta and its IG map must scale by two.
        v_changed.blocks[0].attn.qkv.weight[2 * channels :].mul_(2.0)

    qk_weight = qk_changed.blocks[0].attn.qkv.weight.detach()
    v_weight = v_changed.blocks[0].attn.qkv.weight.detach()
    assert not torch.equal(qk_weight[: 2 * channels], original_weight[: 2 * channels])
    torch.testing.assert_close(
        qk_weight[2 * channels :], original_weight[2 * channels :], atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        v_weight[: 2 * channels], original_weight[: 2 * channels], atol=0.0, rtol=0.0
    )
    assert not torch.equal(v_weight[2 * channels :], original_weight[2 * channels :])

    common = {
        "baselines": baseline,
        "block_index": 0,
        "head": 1,
        "query_token": 0,
        "head_channel": 1,
        "fixed_attention": endpoint_attention,
        "steps": (32, 64),
        "step_chunk_size": 8,
        "atol": 2e-5,
        "rtol": 2e-5,
    }
    reference = conditional_fixed_attention_value_path_ig(model, image, **common)
    changed_qk = conditional_fixed_attention_value_path_ig(qk_changed, image, **common)
    changed_v = conditional_fixed_attention_value_path_ig(v_changed, image, **common)

    assert reference.converged and changed_qk.converged and changed_v.converged
    # Q/K are still evaluated by the toy attention, but the attn_drop hook
    # replaces their softmax result by detached A*.  Neither the target nor
    # any input-attribution element may consequently depend on Q/K rows.
    torch.testing.assert_close(
        changed_qk.conservation["delta"],
        reference.conservation["delta"],
        atol=1e-7,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        changed_qk.attribution, reference.attribution, atol=1e-7, rtol=1e-7
    )

    # In contrast, doubling only V rows doubles O=A*@V, its endpoint delta,
    # and every element of the corresponding IG map.
    torch.testing.assert_close(
        changed_v.conservation["delta"],
        2.0 * reference.conservation["delta"],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        changed_v.attribution,
        2.0 * reference.attribution,
        atol=2e-5,
        rtol=2e-5,
    )
    assert (
        changed_v.conservation["delta"] - reference.conservation["delta"]
    ).abs().max() > 1e-3
    assert (changed_v.attribution - reference.attribution).abs().max() > 1e-3


def test_fixed_attention_model_ig_is_explicitly_full_model_not_value_path(
    model: ToyTimmModel, image: torch.Tensor
) -> None:
    result = conditional_fixed_attention_model_ig(
        model,
        image,
        torch.zeros_like(image),
        TargetSpec.class_logit(1),
        block_index=0,
        steps=(8, 16, 32),
        step_chunk_size=4,
        atol=3e-5,
        rtol=3e-5,
    )
    assert result.converged
    assert result.metadata["scope"] == "fixed_attention_model"
    assert result.metadata["includes_residual_paths"] is True
    assert result.metadata["is_complete_attention_attribution"] is False


def test_value_path_indices_fail_closed(model: ToyTimmModel, image: torch.Tensor) -> None:
    with pytest.raises(IndexError, match="head 9"):
        conditional_fixed_attention_value_path_ig(
            model,
            image,
            torch.zeros_like(image),
            block_index=0,
            head=9,
            query_token=0,
            head_channel=0,
            steps=2,
        )
