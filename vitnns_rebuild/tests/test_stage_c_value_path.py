from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import vitnns.stage_c_value_path as value_path
from vitnns.attribution import (
    IGResult,
    conditional_fixed_attention_value_path_ig,
    integrated_gradients,
)
from vitnns.modeling import load_canonical_model
from vitnns.retention import ArtifactKind
from vitnns.stage_c_maps import (
    ComparisonLine,
    MethodResult,
    NAReason,
    SignSemantics,
)
from vitnns.stage_c_value_path import (
    DEGENERATE_DIRECTION,
    DEGENERATE_PATCH_SUPPORT,
    DEGENERATE_POSITIVE_PATCH_EVIDENCE,
    HEAD_DIM,
    IG_ATOL,
    IG_RTOL,
    IG_STEPS,
    IG_STEP_CHUNK_SIZE,
    METHOD_VERSION,
    PATCH_SUPPORT_MINIMUM,
    POSITIVE_PATCH_EVIDENCE_MINIMUM,
    STATUS_OK,
    FrozenHeadDirectionTarget,
    HeadCandidate,
    HeadDirectionResolution,
    MatchedTokenContribution,
    StageCValuePathConvergenceError,
    StageCValuePathError,
    all_head_candidates,
    chunk_head_candidates,
    conditional_fixed_clean_attention_head_direction_value_path_ig,
    matched_fixed_clean_attention_patch_token_contribution,
    matched_token_contributions,
    resolve_fixed_clean_attention_head_direction,
    resolve_fixed_clean_attention_head_directions,
    resolve_head_direction_from_endpoints,
)


TOKENS = 197
PATCHES = 196
PATCH_SHAPE = (14, 14)
TARGET_REF = "fixed_clean_attention_cls_head_direction"


class _ToyQKV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(tokens)
        return torch.cat((zeros, zeros, tokens * self.scale), dim=-1)


class _RegistrationFailProjection(nn.Identity):
    def __init__(self, *, fail_registration: bool = False) -> None:
        super().__init__()
        self.fail_registration = fail_registration

    def register_forward_pre_hook(self, hook, *args, **kwargs):
        handle = super().register_forward_pre_hook(hook, *args, **kwargs)
        if self.fail_registration:
            raise RuntimeError("synthetic projection hook registration failure")
        return handle


class _ToyAttention(nn.Module):
    def __init__(self, *, registration_failure: bool = False) -> None:
        super().__init__()
        self.num_heads = 12
        self.head_dim = 64
        self.fused_attn = True
        self.qkv = _ToyQKV()
        self.attn_drop = nn.Identity()
        self.proj = _RegistrationFailProjection(
            fail_registration=registration_failure
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, token_count, width = tokens.shape
        assert (token_count, width) == (TOKENS, 768)
        qkv = self.qkv(tokens)
        values = qkv[..., 2 * width :].reshape(
            batch, token_count, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        weights = torch.softmax(tokens[:, :, 0], dim=1)
        attention = torch.zeros(
            (batch, self.num_heads, token_count, token_count),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        attention[:, :, :, 0] = 1.0
        attention[:, :, 0, :] = weights[:, None, :]
        attention = self.attn_drop(attention)
        cls_heads = torch.einsum("bhk,bhkd->bhd", attention[:, :, 0], values)
        combined = torch.zeros_like(tokens)
        combined[:, 0] = cls_heads.reshape(batch, width)
        return self.proj(combined)


class _ToyBlock(nn.Module):
    def __init__(self, *, registration_failure: bool = False) -> None:
        super().__init__()
        self.attn = _ToyAttention(registration_failure=registration_failure)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + 0.05 * self.attn(tokens)


class _ToyValueModel(nn.Module):
    """Canonical-shape but cheap model for endpoint and lifecycle checks."""

    def __init__(self, *, registration_failure_block: int | None = None) -> None:
        super().__init__()
        self.unused = nn.Parameter(
            torch.tensor(2.0, dtype=torch.float32), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                _ToyBlock(registration_failure=index == registration_failure_block)
                for index in range(12)
            ]
        )
        self.forward_calls = 0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        patches = F.avg_pool2d(image[:, :1], kernel_size=16, stride=16)
        patch_scalars = patches.flatten(2).transpose(1, 2)
        patch_tokens = patch_scalars.repeat(1, 1, 768)
        cls_scalar = image.mean(dim=(1, 2, 3), keepdim=False).reshape(-1, 1, 1)
        cls = cls_scalar.repeat(1, 1, 768)
        tokens = torch.cat((cls, patch_tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.mean(dim=1)


def _clean_image() -> torch.Tensor:
    patch_values = torch.linspace(0.2, 1.4, PATCHES, dtype=torch.float32).reshape(
        PATCH_SHAPE
    )
    channel = patch_values.repeat_interleave(16, 0).repeat_interleave(16, 1)
    return torch.stack((channel, 0.5 * channel, -0.25 * channel)).unsqueeze(0)


def _hook_inventory(model: nn.Module) -> tuple[tuple[tuple[str, int, int], ...], ...]:
    result = []
    for module in model.modules():
        entries: list[tuple[str, int, int]] = []
        for name in (
            "_forward_pre_hooks",
            "_forward_hooks",
            "_backward_pre_hooks",
            "_backward_hooks",
        ):
            entries.extend(
                (name, int(key), id(callback))
                for key, callback in getattr(module, name).items()
            )
        result.append(tuple(entries))
    return tuple(result)


def _mode_flags(model: nn.Module) -> tuple[bool, ...]:
    return tuple(module.training for module in model.modules())


def _requires_grad_flags(model: nn.Module) -> tuple[bool, ...]:
    return tuple(parameter.requires_grad for parameter in model.parameters())


def _grad_snapshot(
    model: nn.Module,
) -> tuple[tuple[torch.Tensor | None, torch.Tensor | None], ...]:
    return tuple(
        (parameter.grad, None if parameter.grad is None else parameter.grad.detach().clone())
        for parameter in model.parameters()
    )


def _assert_grad_snapshot(
    model: nn.Module,
    snapshot: tuple[tuple[torch.Tensor | None, torch.Tensor | None], ...],
) -> None:
    for parameter, (original, values) in zip(model.parameters(), snapshot):
        assert parameter.grad is original
        if original is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, values, atol=0.0, rtol=0.0)


def _mixed_caller_state(model: _ToyValueModel) -> None:
    model.eval()
    model.blocks[0].train()
    parameters = tuple(model.parameters())
    parameters[0].requires_grad_(False)
    parameters[1].requires_grad_(True)
    parameters[1].grad = torch.tensor(7.25, dtype=torch.float32)



def _json_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_thaw(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolution_for_contributions(
    contributions: torch.Tensor,
) -> HeadDirectionResolution:
    """Make an endpoint whose analytic c_j values equal ``contributions``."""

    assert contributions.shape == (TOKENS,)
    assert contributions.dtype is torch.float32
    nonzero = torch.nonzero(contributions, as_tuple=False).flatten()
    assert nonzero.numel() > 0
    attention = torch.zeros(TOKENS, dtype=torch.float32)
    attention[nonzero] = 1.0 / float(nonzero.numel())
    clean = torch.zeros((TOKENS, 4), dtype=torch.float32)
    clean[nonzero, 0] = contributions[nonzero] / attention[nonzero]
    baseline = torch.zeros_like(clean)
    resolution = resolve_head_direction_from_endpoints(
        attention, clean, baseline, block_index=3, head_index=7
    )
    assert resolution.status == STATUS_OK
    assert resolution.u is not None
    assert resolution.u[0] > 0
    return resolution


def _sparse_contributions(*values: tuple[int, float]) -> torch.Tensor:
    result = torch.zeros(TOKENS, dtype=torch.float32)
    for index, value in values:
        result[index] = value
    assert result.sum() > 0
    return result


def test_public_candidate_space_is_exact_block_major_12_by_12_and_chunk_stable():
    candidates = all_head_candidates()

    assert len(candidates) == 144
    assert len(set(candidates)) == 144
    assert candidates == tuple(
        HeadCandidate(block, head)
        for block in range(12)
        for head in range(12)
    )
    assert candidates[0].to_dict() == {"block_index": 0, "head_index": 0}
    assert candidates[-1].to_dict() == {"block_index": 11, "head_index": 11}

    for chunk_size in (1, 7, 12, 17, 144, 200):
        chunks = chunk_head_candidates(candidates, chunk_size)
        assert tuple(candidate for chunk in chunks for candidate in chunk) == candidates
        assert all(0 < len(chunk) <= chunk_size for chunk in chunks)

    selected = (candidates[143], candidates[0], candidates[70])
    assert chunk_head_candidates(selected, 2) == (selected[:2], selected[2:])
    assert chunk_head_candidates((), 4) == ()


@pytest.mark.parametrize(
    ("factory", "exception", "message"),
    [
        (lambda: HeadCandidate(True, 0), TypeError, "block_index"),
        (lambda: HeadCandidate(-1, 0), StageCValuePathError, "block_index"),
        (lambda: HeadCandidate(12, 0), StageCValuePathError, "block_index"),
        (lambda: HeadCandidate(0, False), TypeError, "head_index"),
        (lambda: HeadCandidate(0, 12), StageCValuePathError, "head_index"),
        (lambda: chunk_head_candidates([], True), ValueError, "chunk_size"),
        (lambda: chunk_head_candidates([], 0), ValueError, "chunk_size"),
        (
            lambda: chunk_head_candidates([HeadCandidate(0, 0)] * 2, 1),
            StageCValuePathError,
            "duplicates",
        ),
    ],
)
def test_candidate_api_fails_closed(factory, exception, message):
    with pytest.raises(exception, match=message):
        factory()


def test_pure_direction_uses_fixed_clean_attention_at_both_endpoints_not_decoy():
    clean_attention = torch.tensor([0.75, 0.25], dtype=torch.float32)
    decoy_baseline_attention = torch.tensor([0.0, 1.0], dtype=torch.float32)
    clean_values = torch.tensor([[2.0, -1.0], [6.0, 3.0]], dtype=torch.float32)
    baseline_values = torch.tensor([[-2.0, 1.0], [10.0, -5.0]], dtype=torch.float32)

    resolved = resolve_head_direction_from_endpoints(
        clean_attention,
        clean_values,
        baseline_values,
        block_index=2,
        head_index=5,
    )
    expected_clean = clean_attention @ clean_values
    expected_baseline = clean_attention @ baseline_values
    forbidden_baseline = decoy_baseline_attention @ baseline_values
    expected_delta = expected_clean - expected_baseline

    assert isinstance(resolved, HeadDirectionResolution)
    assert resolved.candidate == HeadCandidate(2, 5)
    assert resolved.status == STATUS_OK
    assert resolved.na is None
    torch.testing.assert_close(resolved.A, clean_attention, atol=0.0, rtol=0.0)
    torch.testing.assert_close(resolved.g_clean, expected_clean, atol=0.0, rtol=0.0)
    torch.testing.assert_close(resolved.g_baseline, expected_baseline, atol=0.0, rtol=0.0)
    assert not torch.equal(resolved.g_baseline, forbidden_baseline)
    torch.testing.assert_close(resolved.d, expected_delta, atol=0.0, rtol=0.0)
    assert resolved.r == pytest.approx(float(torch.linalg.vector_norm(expected_delta)))
    torch.testing.assert_close(
        resolved.u, expected_delta / torch.linalg.vector_norm(expected_delta)
    )
    assert torch.dot(resolved.u, resolved.d).item() == pytest.approx(resolved.r)
    assert torch.dot(resolved.u, resolved.g_clean - resolved.g_baseline).item() > 0


def test_direction_sign_scale_and_owned_detached_frozen_endpoint_tensors():
    attention = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True)
    clean = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32, requires_grad=True
    )
    baseline = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    original_attention = attention.detach().clone()

    resolved = resolve_head_direction_from_endpoints(attention, clean, baseline)
    scaled = resolve_head_direction_from_endpoints(
        attention.detach(), clean.detach() * 3.0, baseline.detach()
    )
    reversed_direction = resolve_head_direction_from_endpoints(
        attention.detach(), baseline.detach(), clean.detach()
    )

    torch.testing.assert_close(scaled.u, resolved.u, atol=1e-7, rtol=1e-7)
    assert scaled.r == pytest.approx(3.0 * resolved.r)
    torch.testing.assert_close(reversed_direction.u, -resolved.u)
    assert reversed_direction.r == pytest.approx(resolved.r)
    assert torch.dot(reversed_direction.u, reversed_direction.d).item() > 0

    for tensor in (
        resolved.A,
        resolved.clean_values,
        resolved.baseline_values,
        resolved.d,
        resolved.u,
    ):
        assert tensor is not None
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None
    with torch.no_grad():
        attention.fill_(0.0)
        clean.fill_(99.0)
        baseline.fill_(-99.0)
    torch.testing.assert_close(resolved.A, original_attention, atol=0.0, rtol=0.0)
    torch.testing.assert_close(resolved.g_clean, torch.tensor([1.0, 1.0]))


@pytest.mark.parametrize(
    ("magnitude", "expected_status"),
    [
        (0.5e-6, DEGENERATE_DIRECTION),
        (1.0e-6, DEGENERATE_DIRECTION),
        (1.1e-6, STATUS_OK),
    ],
    ids=["below", "equal", "above"],
)
def test_direction_relative_threshold_is_inclusive_at_frozen_boundary(
    magnitude, expected_status
):
    resolved = resolve_head_direction_from_endpoints(
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([[magnitude, 0.0]], dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
    )

    assert resolved.tau == pytest.approx(1e-6)
    assert resolved.status == expected_status
    if expected_status == DEGENERATE_DIRECTION:
        assert resolved.u is None
        assert resolved.na is not None
        assert resolved.na.reason is NAReason.DEGENERATE_DIRECTION
        assert resolved.na.invalidating
    else:
        torch.testing.assert_close(resolved.u, torch.tensor([1.0, 0.0]))


def test_direction_threshold_is_relative_to_larger_endpoint_norm():
    attention = torch.tensor([1.0], dtype=torch.float32)
    baseline = torch.tensor([[1000.0, 0.0]], dtype=torch.float32)

    at_threshold = resolve_head_direction_from_endpoints(
        attention,
        baseline + torch.tensor([[0.001, 0.0]], dtype=torch.float32),
        baseline,
    )
    above = resolve_head_direction_from_endpoints(
        attention,
        baseline + torch.tensor([[0.0011, 0.0]], dtype=torch.float32),
        baseline,
    )

    expected_tau = 1e-6 * max(
        1.0,
        float(torch.linalg.vector_norm(at_threshold.g_clean)),
        float(torch.linalg.vector_norm(at_threshold.g_baseline)),
    )
    assert at_threshold.tau == pytest.approx(expected_tau)
    assert at_threshold.status == DEGENERATE_DIRECTION
    assert above.status == STATUS_OK


@pytest.mark.parametrize(
    "endpoint", ["attention_nan", "clean_inf", "baseline_nan", "zero"]
)
def test_nonfinite_and_zero_direction_return_structured_degenerate_direction(endpoint):
    attention = torch.tensor([0.4, 0.6], dtype=torch.float32)
    clean = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    baseline = clean.clone()
    if endpoint == "attention_nan":
        attention[0] = torch.nan
    elif endpoint == "clean_inf":
        clean[1, 1] = torch.inf
    elif endpoint == "baseline_nan":
        baseline[0, 0] = torch.nan

    result = resolve_head_direction_from_endpoints(attention, clean, baseline)

    assert result.status == DEGENERATE_DIRECTION
    assert result.u is None
    assert result.na is not None
    assert result.na.status == "NA"
    assert result.na.value is None
    assert result.na.reason is NAReason.DEGENERATE_DIRECTION
    assert result.na.to_dict()["reason"] == "degenerate_direction"


@pytest.mark.parametrize(
    ("attention", "clean", "baseline", "exception", "message"),
    [
        (torch.empty(0), torch.empty((0, 2)), torch.empty((0, 2)), StageCValuePathError, "non-empty"),
        (torch.ones((1, 2)), torch.ones((2, 2)), torch.zeros((2, 2)), StageCValuePathError, r"shape \[tokens\]"),
        (torch.ones(2), torch.ones(2), torch.zeros(2), StageCValuePathError, "values.*shape"),
        (torch.ones(2), torch.ones((3, 2)), torch.zeros((3, 2)), StageCValuePathError, "token count"),
        (torch.ones(2), torch.ones((2, 2)), torch.zeros((2, 3)), StageCValuePathError, "identical shapes"),
        (torch.tensor([1.0, -0.1]), torch.ones((2, 2)), torch.zeros((2, 2)), StageCValuePathError, "negative"),
        (torch.ones(2, dtype=torch.float64), torch.ones((2, 2)), torch.zeros((2, 2)), TypeError, "float32"),
        (torch.ones(2), torch.ones((2, 2), dtype=torch.float64), torch.zeros((2, 2)), TypeError, "float32"),
    ],
)
def test_pure_direction_structural_shape_and_dtype_fail_closed(
    attention, clean, baseline, exception, message
):
    with pytest.raises(exception, match=message):
        resolve_head_direction_from_endpoints(attention, clean, baseline)


def test_matched_all_197_contributions_manual_conservation_cls_and_patch_order():
    expected = torch.linspace(-0.1, 0.2, TOKENS, dtype=torch.float32)
    expected[0] = -0.375
    assert expected.sum() > 0
    resolution = _resolution_for_contributions(expected)

    result = matched_token_contributions(resolution)
    manual = resolution.A * torch.sum(
        (resolution.clean_values - resolution.baseline_values) * resolution.u,
        dim=1,
    )

    assert isinstance(result, MatchedTokenContribution)
    assert result.status == STATUS_OK
    assert result.na is None
    assert result.candidate == HeadCandidate(3, 7)
    assert result.token_contributions.shape == (TOKENS,)
    torch.testing.assert_close(result.contributions, manual, atol=1e-6, rtol=1e-6)
    assert result.cls_contribution == pytest.approx(float(manual[0]))
    torch.testing.assert_close(result.patch_contributions, manual[1:])
    torch.testing.assert_close(result.patch_map.flatten(), manual[1:])
    assert result.patch_map[0, 0] == pytest.approx(float(manual[1]))
    assert result.patch_map[13, 13] == pytest.approx(float(manual[196]))
    assert result.cls_contribution != pytest.approx(float(result.patch_map[0, 0]))
    assert result.conservation_sum == pytest.approx(float(manual.double().sum()))
    assert result.conservation_error <= result.conservation_tolerance
    assert result.conservation_sum == pytest.approx(resolution.r, abs=2e-5)


def test_synthetic_head_direction_uses_real_ig_core_and_conserves_exactly():
    image = torch.tensor(
        [[[[1.0, -2.0]], [[0.5, 3.0]], [[-1.5, 2.5]]]],
        dtype=torch.float32,
    )
    baseline = torch.zeros_like(image)
    attention = torch.tensor([0.25, 0.75], dtype=torch.float32)
    value_weight = torch.tensor(
        [[0.5, -0.25], [1.5, 0.75], [-0.5, 2.0]], dtype=torch.float32
    )

    def values(inputs: torch.Tensor) -> torch.Tensor:
        tokens = inputs.permute(0, 2, 3, 1).reshape(inputs.shape[0], 2, 3)
        return tokens @ value_weight

    clean_values = values(image)[0]
    baseline_values = values(baseline)[0]
    target = resolve_head_direction_from_endpoints(
        attention, clean_values, baseline_values
    )
    assert target.status == STATUS_OK and target.u is not None
    fixed_a = target.A.clone()
    frozen_u = target.u.clone()

    def callback(inputs: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            torch.sum(fixed_a[None, :, None] * values(inputs), dim=1)
            * frozen_u[None, :],
            dim=1,
        )

    result = integrated_gradients(
        callback,
        image,
        baseline,
        steps=(2, 4),
        step_chunk_size=2,
        atol=1e-6,
        rtol=1e-6,
    )

    assert result.converged
    assert result.steps == 2
    assert tuple(attempt["steps"] for attempt in result.attempts) == (2,)
    torch.testing.assert_close(
        result.conservation["delta"],
        torch.tensor([target.r]),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        result.conservation["sum"], result.conservation["delta"],
        atol=1e-6, rtol=1e-6,
    )
    torch.testing.assert_close(target.A, fixed_a, atol=0.0, rtol=0.0)
    torch.testing.assert_close(target.u, frozen_u, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("contributions", "support_relation", "support_flag"),
    [
        (_sparse_contributions((0, 100.0), (1, 1.0)), "below", True),
        (_sparse_contributions((0, 99.0), (1, 1.0)), "equal", False),
        (_sparse_contributions((0, 98.0), (1, 1.0)), "above", False),
    ],
    ids=["below", "equal", "above"],
)
def test_patch_support_uses_strict_point_zero_one_threshold_and_retains_flag(
    contributions, support_relation, support_flag
):
    result = matched_token_contributions(_resolution_for_contributions(contributions))

    expected = float(contributions[1:].abs().double().sum()) / float(
        contributions.abs().double().sum()
    )
    assert result.patch_support == pytest.approx(expected)
    if support_relation == "below":
        assert result.patch_support < PATCH_SUPPORT_MINIMUM
    elif support_relation == "equal":
        assert result.patch_support == PATCH_SUPPORT_MINIMUM
    else:
        assert result.patch_support > PATCH_SUPPORT_MINIMUM
    assert (DEGENERATE_PATCH_SUPPORT in result.flags) is support_flag
    assert result.status == STATUS_OK
    assert result.na is None


@pytest.mark.parametrize(
    ("contributions", "relation", "evidence_flag"),
    [
        (_sparse_contributions((0, 90.0), (1, -9.0), (2, 0.5)), "below", True),
        (_sparse_contributions((0, 90.0), (1, -9.0), (2, 1.0)), "equal", False),
        (_sparse_contributions((0, 90.0), (1, -9.0), (2, 2.0)), "above", False),
    ],
    ids=["below", "equal", "above"],
)
def test_positive_patch_evidence_strict_threshold_is_separate_and_retained(
    contributions, relation, evidence_flag
):
    result = matched_token_contributions(_resolution_for_contributions(contributions))

    expected = float(contributions[1:].clamp_min(0).double().sum()) / float(
        contributions.abs().double().sum()
    )
    assert result.positive_patch_evidence_fraction == pytest.approx(expected)
    if relation == "below":
        assert result.positive_patch_evidence_fraction < POSITIVE_PATCH_EVIDENCE_MINIMUM
    elif relation == "equal":
        assert result.positive_patch_evidence_fraction == POSITIVE_PATCH_EVIDENCE_MINIMUM
    else:
        assert result.positive_patch_evidence_fraction > POSITIVE_PATCH_EVIDENCE_MINIMUM
    assert (DEGENERATE_POSITIVE_PATCH_EVIDENCE in result.flags) is evidence_flag
    assert DEGENERATE_PATCH_SUPPORT not in result.flags
    assert result.status == STATUS_OK


def test_matched_degenerate_direction_is_retained_as_structured_result():
    degenerate = resolve_head_direction_from_endpoints(
        torch.tensor([1.0], dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
    )

    result = matched_token_contributions(degenerate)

    assert result.status == DEGENERATE_DIRECTION
    assert result.flags == (DEGENERATE_DIRECTION,)
    assert result.na is degenerate.na
    assert result.token_contributions is None
    assert result.cls_contribution is None
    assert result.patch_map is None


def test_matched_rejects_noncanonical_token_count_and_inconsistent_resolution():
    short = resolve_head_direction_from_endpoints(
        torch.tensor([0.5, 0.5], dtype=torch.float32),
        torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.float32),
    )
    with pytest.raises(StageCValuePathError, match="exactly 197"):
        matched_token_contributions(short)

    zeros = torch.zeros((TOKENS, 2), dtype=torch.float32)
    with pytest.raises(StageCValuePathError, match="normalized to one"):
        HeadDirectionResolution(
            candidate=HeadCandidate(0, 0),
            status=STATUS_OK,
            clean_attention=torch.zeros(TOKENS, dtype=torch.float32),
            clean_values=zeros,
            baseline_values=zeros,
            clean_head_output=torch.zeros(2, dtype=torch.float32),
            baseline_head_output=torch.zeros(2, dtype=torch.float32),
            direction_delta=torch.zeros(2, dtype=torch.float32),
            direction_norm=0.0,
            direction_unit=torch.tensor([1.0, 0.0], dtype=torch.float32),
            direction_threshold=1e-6,
        )


def _issued_target(
    *, block_index: int = 2, head_index: int = 3
) -> tuple[_ToyValueModel, torch.Tensor, FrozenHeadDirectionTarget]:
    model = _ToyValueModel().eval()
    image = _clean_image()
    target = resolve_fixed_clean_attention_head_direction(
        model, image, block_index, head_index
    )
    assert isinstance(target, FrozenHeadDirectionTarget)
    assert target.status == STATUS_OK
    return model, image, target


def _signed_native(total: float) -> torch.Tensor:
    native = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    native[0, 0, :16, :16] = total / 256.0
    return native


def _ig_result(
    target: FrozenHeadDirectionTarget,
    *,
    steps: int = 512,
    converged: bool = True,
    attribution: torch.Tensor | None = None,
    attempt_steps: tuple[int, ...] | None = None,
    metadata_updates: Mapping[str, Any] | None = None,
    report_delta: float | None = None,
) -> IGResult:
    native = _signed_native(target.r) if attribution is None else attribution
    report_sum_tensor = native.to(torch.float32).flatten(1).sum(dim=1)
    report_delta_tensor = torch.tensor(
        [target.r if report_delta is None else report_delta], dtype=torch.float32
    )
    report_abs_tensor = (report_delta_tensor - report_sum_tensor).abs()
    report_rel_tensor = report_abs_tensor / report_delta_tensor.abs().clamp_min(
        torch.finfo(torch.float32).eps
    )
    report_pass_tensor = report_abs_tensor <= (
        IG_ATOL + IG_RTOL * report_delta_tensor.abs()
    )
    delta = float(report_delta_tensor.item())
    attribution_sum = float(report_sum_tensor.item())
    absolute = float(report_abs_tensor.item())
    relative = float(report_rel_tensor.item())
    passes = bool(report_pass_tensor.item())
    if attempt_steps is None:
        attempt_steps = (
            IG_STEPS[: IG_STEPS.index(steps) + 1]
            if converged
            else IG_STEPS
        )
    attempts_list: list[dict[str, Any]] = []
    for index, step in enumerate(attempt_steps):
        if index == len(attempt_steps) - 1:
            attempt_sum = attribution_sum
            attempt_abs = absolute
            attempt_rel = relative
            attempt_pass = passes
        else:
            attempt_sum = delta + 1.0
            attempt_abs = abs(delta - attempt_sum)
            attempt_rel = attempt_abs / max(
                abs(delta), torch.finfo(torch.float32).eps
            )
            attempt_pass = False
        attempts_list.append(
            {
                "steps": step,
                "delta": [delta],
                "sum": [attempt_sum],
                "abs": [attempt_abs],
                "rel": [attempt_rel],
                "pass": [attempt_pass],
            }
        )
    attempts = tuple(attempts_list)
    metadata: dict[str, Any] = {
        "method": "integrated_gradients",
        "quadrature": "gauss_legendre",
        "integration_dtype": "float32",
        "input_shape": [1, 3, 224, 224],
        "step_schedule": list(IG_STEPS),
        "step_chunk_size": IG_STEP_CHUNK_SIZE,
        "atol": IG_ATOL,
        "rtol": IG_RTOL,
        "scope": "fixed_clean_attention_head_direction_value_path",
        "target": {
            "kind": TARGET_REF,
            "block_index": target.block_index,
            "head_index": target.head_index,
            "query_token": 0,
        },
        "includes_residual_paths": False,
        "is_complete_attention_attribution": False,
        "fixed_clean_attention": True,
        "frozen_direction": True,
    }
    metadata.update(dict(metadata_updates or {}))
    report = {
        "delta": report_delta_tensor,
        "sum": report_sum_tensor,
        "abs": report_abs_tensor,
        "rel": report_rel_tensor,
        "pass": report_pass_tensor,
    }
    return IGResult(
        attribution=native,
        conservation=report,
        steps=steps,
        converged=converged,
        attempts=attempts,
        metadata=metadata,
    )


def _assert_method_result(
    result: MethodResult, *, method_id: str, projection: str
) -> None:
    assert {field.name for field in dataclasses.fields(result)} == {
        "method_id",
        "method_version",
        "comparison_line",
        "explanation_target",
        "fidelity_target",
        "class_agnostic",
        "native_tensor",
        "metric_map",
        "projection",
        "sign_semantics",
        "valid_mask",
        "provenance",
        "na",
    }
    assert result.method_id == method_id
    assert result.method_version == METHOD_VERSION == "1.0.0"
    assert result.comparison_line is ComparisonLine.VALUE_PATH
    assert result.explanation_target == TARGET_REF
    assert result.fidelity_target == TARGET_REF
    assert result.class_agnostic is True
    assert result.projection == projection
    assert result.sign_semantics is SignSemantics.SIGNED
    assert result.na is None
    assert result.native_tensor is not None
    assert result.native_tensor.kind is ArtifactKind.RAW
    assert result.metric_map is not None
    assert result.metric_map.kind is ArtifactKind.METRIC_READY
    assert result.metric_map.shape == PATCH_SHAPE
    assert result.metric_map.dtype == np.dtype(np.float32)
    assert np.isfinite(result.metric_map.data).all()
    assert result.valid_mask.dtype == np.bool_
    assert bool(result.valid_mask.all())
    assert result.provenance["stage"] == "C4"
    assert result.provenance["comparison_line"] == "value_path"
    assert result.provenance["class_agnostic"] is True
    assert result.provenance["method_implementation_version"] == METHOD_VERSION
    assert result.provenance["target"]["target_kind"] == TARGET_REF
    json.dumps(_json_thaw(result.provenance), allow_nan=False)
    with pytest.raises(TypeError):
        result.provenance["tamper"] = True
    with pytest.raises(ValueError):
        result.valid_mask[0, 0] = False


def test_single_resolver_fixed_clean_a_endpoint_parity_and_target_binding():
    model, image, target = _issued_target(block_index=4, head_index=9)

    assert target.candidate == HeadCandidate(4, 9)
    assert target.status == STATUS_OK
    assert target.na is None
    assert target.A.shape == (TOKENS,)
    assert target.fixed_attention.shape == (1, TOKENS, TOKENS)
    assert target.clean_values.shape == (TOKENS, HEAD_DIM)
    assert target.baseline_values.shape == (TOKENS, HEAD_DIM)
    torch.testing.assert_close(
        target.g_clean, target.A @ target.clean_values, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        target.g_baseline, target.A @ target.baseline_values, atol=1e-6, rtol=1e-6
    )
    assert target.clean_value_max_abs_error < 1e-6
    assert target.clean_replay_max_abs_error < 1e-6
    assert target.clean_model_output_max_abs_error < 1e-6
    assert target.baseline_replay_max_abs_error < 1e-6
    assert target.clean_input_shape == tuple(image.shape)
    assert len(target.clean_input_sha256) == 64
    assert len(target.runtime_model_binding_sha256) == 64
    assert target.A.requires_grad is False
    assert target.u.requires_grad is False
    target_json = target.to_dict()
    assert target_json["attention_source"] == "clean_real_forward_attention_used"
    assert target_json["baseline_uses_clean_fixed_attention"] is True
    assert target_json["detached_and_frozen_before_ig"] is True
    assert target_json["clean_model_output_max_abs_error"] < 1e-6
    assert next(model.parameters()).device.type == "cpu"


def test_production_resolver_baseline_uses_fixed_clean_a_not_natural_baseline_a():
    model = _ToyValueModel().eval()
    # Give V(x0) a token-varying component so differing attention rows produce
    # observably different natural-vs-fixed baseline outputs.
    positions = torch.linspace(-0.2, 0.3, TOKENS).reshape(1, TOKENS, 1)

    def positional_value(_module, _arguments, output):
        changed = output.clone()
        changed[..., 2 * 768 :] += positions
        return changed

    handle = model.blocks[0].attn.qkv.register_forward_hook(positional_value)
    image = _clean_image()
    try:
        target = resolve_fixed_clean_attention_head_direction(model, image, 0, 3)
        baseline = torch.zeros_like(image)

        # Capture natural baseline A/V independently. The resolved g0 must use
        # clean A, never this natural baseline attention.
        attention_module = model.blocks[0].attn
        captures, _ = value_path._capture_real_endpoints(
            model, baseline, {0: attention_module}
        )
        baseline_values, baseline_attention, _ = value_path._validate_endpoint_block_data(
            captures[0], block_index=0
        )
        natural_baseline_g = baseline_attention[0, 3, 0] @ baseline_values[0, 3]
        fixed_baseline_g = target.A @ target.baseline_values

        torch.testing.assert_close(target.g_baseline, fixed_baseline_g)
        assert not torch.equal(target.A, baseline_attention[0, 3, 0])
        assert not torch.equal(target.g_baseline, natural_baseline_g)
        assert target.baseline_replay_max_abs_error < 1e-6
    finally:
        handle.remove()


def test_selective_chunked_resolver_preserves_request_order_and_values():
    model = _ToyValueModel().eval()
    image = _clean_image()
    candidates = (HeadCandidate(5, 11), HeadCandidate(1, 2), HeadCandidate(5, 0))

    one = resolve_fixed_clean_attention_head_directions(
        model, image, candidates, candidate_chunk_size=1
    )
    chunked = resolve_fixed_clean_attention_head_directions(
        model, image, candidates, candidate_chunk_size=12
    )

    assert tuple(target.candidate for target in one) == candidates
    assert tuple(target.candidate for target in chunked) == candidates
    for left, right in zip(one, chunked):
        torch.testing.assert_close(left.A, right.A, atol=0.0, rtol=0.0)
        torch.testing.assert_close(left.g_clean, right.g_clean, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            left.g_baseline, right.g_baseline, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(left.u, right.u, atol=0.0, rtol=0.0)
        assert left.r == right.r


def test_all_144_candidates_compute_identically_across_chunk_boundaries():
    model = _ToyValueModel().eval()
    image = _clean_image()
    candidates = all_head_candidates()

    one = resolve_fixed_clean_attention_head_directions(
        model, image, candidates, candidate_chunk_size=1
    )
    mixed = resolve_fixed_clean_attention_head_directions(
        model, image, candidates, candidate_chunk_size=17
    )

    assert len(one) == len(mixed) == 144
    for left, right in zip(one, mixed):
        assert left.candidate == right.candidate
        assert left.status == right.status
        torch.testing.assert_close(left.A, right.A, atol=0.0, rtol=0.0)
        torch.testing.assert_close(left.g_clean, right.g_clean, atol=0.0, rtol=0.0)
        torch.testing.assert_close(left.g_baseline, right.g_baseline, atol=0.0, rtol=0.0)
        torch.testing.assert_close(left.d, right.d, atol=0.0, rtol=0.0)
        assert left.r == right.r
        if left.u is None:
            assert right.u is None
        else:
            torch.testing.assert_close(left.u, right.u, atol=0.0, rtol=0.0)
        assert left.clean_model_output_max_abs_error == right.clean_model_output_max_abs_error
        assert left.baseline_replay_max_abs_error == right.baseline_replay_max_abs_error


@pytest.mark.parametrize(
    ("inputs", "candidates", "chunk_size", "exception", "message"),
    [
        (torch.zeros((1, 3, 223, 224)), (HeadCandidate(0, 0),), 1, ValueError, "shape"),
        (torch.zeros((1, 3, 224, 224), dtype=torch.float64), (HeadCandidate(0, 0),), 1, TypeError, "float32"),
        (torch.full((1, 3, 224, 224), torch.nan), (HeadCandidate(0, 0),), 1, ValueError, "finite"),
        (torch.zeros((1, 3, 224, 224)), (), 1, StageCValuePathError, "must not be empty"),
        (torch.zeros((1, 3, 224, 224)), (HeadCandidate(0, 0),), True, ValueError, "chunk_size"),
    ],
)
def test_resolver_input_candidate_and_chunk_validation(
    inputs, candidates, chunk_size, exception, message
):
    with pytest.raises(exception, match=message):
        resolve_fixed_clean_attention_head_directions(
            _ToyValueModel().eval(),
            inputs,
            candidates,
            candidate_chunk_size=chunk_size,
        )


def test_matched_wrapper_exact_method_result_cls_provenance_and_repeatability():
    _model, _image, target = _issued_target()

    first = matched_fixed_clean_attention_patch_token_contribution(
        target, provenance={"sample_id": "synthetic-1"}
    )
    second = matched_fixed_clean_attention_patch_token_contribution(target)
    analytic = matched_token_contributions(target)

    _assert_method_result(
        first,
        method_id="matched_fixed_clean_attention_patch_token_contribution",
        projection="patch_mass_to_pixel_uniform",
    )
    np.testing.assert_array_equal(first.native_tensor.data, analytic.patch_map.numpy())
    np.testing.assert_array_equal(first.metric_map.data, analytic.patch_map.numpy())
    np.testing.assert_array_equal(first.metric_map.data, second.metric_map.data)
    assert first.provenance["exact_method_name"] == (
        "matched fixed-clean-attention patch-token contribution"
    )
    assert first.provenance["caller_context"]["sample_id"] == "synthetic-1"
    retained = first.provenance["matched_contribution"]
    assert retained["token_contribution_count"] == TOKENS
    assert retained["cls_contribution"] == pytest.approx(analytic.cls_contribution)
    assert retained["token_sum"] == pytest.approx(analytic.conservation_sum)
    assert retained["conservation_abs_error"] <= retained["conservation_tolerance"]
    assert retained["degeneracy_reasons"] == analytic.flags
    assert first.provenance["algorithm"]["cls_given_spatial_location"] is False
    assert first.provenance["algorithm"]["projection_semantics"] == (
        "mass_conserving_uniform_distribution_not_native_pixel_evidence"
    )
    assert "class" not in first.provenance["algorithm"]["claim_boundary"]


def test_matched_wrapper_retains_cls_only_result_without_fake_spatial_location():
    model = _ToyValueModel().eval()
    # At block 0, channel 0 supplies every patch token while the global RGB
    # mean supplies CLS.  V therefore changes only at CLS for this input.
    image = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    image[:, 1] = 1.0
    target = resolve_fixed_clean_attention_head_direction(model, image, 0, 0)
    assert target.status == STATUS_OK
    result = matched_fixed_clean_attention_patch_token_contribution(target)

    assert np.count_nonzero(result.metric_map.data) == 0
    assert bool(result.valid_mask.all())
    retained = result.provenance["matched_contribution"]
    assert retained["cls_contribution"] == pytest.approx(target.r, rel=1e-5)
    assert retained["all_zero_patch_map_retained"] is True
    assert retained["degeneracy_reasons"] == (
        DEGENERATE_PATCH_SUPPORT, DEGENERATE_POSITIVE_PATCH_EVIDENCE
    )


def test_ig_wrapper_dispatches_frozen_schedule_keeps_a_u_fixed_and_projects_signed_rgb(
    monkeypatch,
):
    model, image, target = _issued_target()
    calls: dict[str, Any] = {}
    fixed_before = target.fixed_attention.clone()
    direction_before = target.u.clone()

    def fake_ig(forward_fn, inputs, baseline, **kwargs):
        calls["inputs_same_object"] = inputs is image
        calls["baseline"] = baseline.detach().clone()
        calls["kwargs"] = kwargs
        calls["modes"] = _mode_flags(model)
        calls["requires_grad"] = _requires_grad_flags(model)
        for value in (baseline, inputs, 0.25 * inputs, 0.75 * inputs):
            output = forward_fn(value)
            assert output.shape == (1,)
            torch.testing.assert_close(target.fixed_attention, fixed_before)
            torch.testing.assert_close(target.u, direction_before)
        return _ig_result(target)

    monkeypatch.setattr(value_path.attribution, "integrated_gradients", fake_ig)
    result = conditional_fixed_clean_attention_head_direction_value_path_ig(
        model, image, target, provenance={"sample_id": "ig-synthetic"}
    )

    # The wrapper owns a clone so a callback cannot mutate caller input.
    assert calls["inputs_same_object"] is False
    torch.testing.assert_close(calls["kwargs"]["parameters"].blocks[0].attn.qkv.scale, torch.tensor(1.0))
    torch.testing.assert_close(calls["baseline"], torch.zeros_like(image))
    assert calls["kwargs"]["steps"] == IG_STEPS
    assert calls["kwargs"]["step_chunk_size"] == 8
    assert calls["kwargs"]["atol"] == 1e-4
    assert calls["kwargs"]["rtol"] == 0.01
    assert calls["kwargs"]["parameters"] is model
    assert calls["kwargs"]["metadata"] == {
        "scope": "fixed_clean_attention_head_direction_value_path",
        "target": {
            "kind": TARGET_REF,
            "block_index": target.block_index,
            "head_index": target.head_index,
            "query_token": 0,
        },
        "includes_residual_paths": False,
        "is_complete_attention_attribution": False,
        "fixed_clean_attention": True,
        "frozen_direction": True,
    }
    assert not any(calls["modes"])
    assert not any(calls["requires_grad"])
    _assert_method_result(
        result,
        method_id="conditional_fixed_clean_attention_head_direction_value_path_ig",
        projection="pixel_mass_to_patch_sum",
    )
    expected = _signed_native(target.r)[0].numpy()
    np.testing.assert_array_equal(result.native_tensor.data, expected)
    assert result.metric_map.data[0, 0] == pytest.approx(target.r)
    assert np.count_nonzero(result.metric_map.data) == 1
    assert result.provenance["exact_method_name"] == (
        "NNS fixed-clean-attention head-direction Value-path IG"
    )
    assert result.provenance["ig_result"]["selected_steps"] == 512
    assert len(result.provenance["ig_result"]["attempts"]) == 1
    assert result.provenance["caller_context"]["sample_id"] == "ig-synthetic"
    torch.testing.assert_close(target.fixed_attention, fixed_before)
    torch.testing.assert_close(target.u, direction_before)


def test_production_wrapper_callback_with_real_ig_core_conserves(monkeypatch):
    """Exercise the production intervention/capture callback with real IG."""

    model, image, target = _issued_target()
    real_ig = integrated_gradients

    def short_real_ig(forward_fn, inputs, baseline, **kwargs):
        assert kwargs["steps"] == IG_STEPS
        assert kwargs["step_chunk_size"] == IG_STEP_CHUNK_SIZE
        short = real_ig(
            forward_fn,
            inputs,
            baseline,
            steps=(2, 4, 8),
            step_chunk_size=2,
            atol=1e-4,
            rtol=0.01,
            parameters=kwargs["parameters"],
            metadata={**kwargs["metadata"], "step_schedule": list(IG_STEPS)},
        )
        # Re-label only the cheap test's exact result as the first registered
        # schedule entry; production still receives and dispatches IG_STEPS.
        attempts = tuple(
            {**attempt, "steps": IG_STEPS[index]}
            for index, attempt in enumerate(short.attempts)
        )
        return dataclasses.replace(
            short,
            steps=attempts[-1]["steps"],
            attempts=attempts,
            metadata={
                **short.metadata,
                "step_schedule": list(IG_STEPS),
                "step_chunk_size": IG_STEP_CHUNK_SIZE,
            },
        )

    # Keep production metadata/schedule validation while making the analytic
    # toy integral cheap.  This replaces only dispatch policy, not the IG core.
    monkeypatch.setattr(value_path.attribution, "integrated_gradients", short_real_ig)
    result = conditional_fixed_clean_attention_head_direction_value_path_ig(
        model, image, target
    )
    repeated = conditional_fixed_clean_attention_head_direction_value_path_ig(
        model, image, target
    )

    assert result.na is None
    np.testing.assert_allclose(
        result.native_tensor.data, repeated.native_tensor.data, atol=1e-6, rtol=1e-5
    )
    np.testing.assert_allclose(
        result.metric_map.data, repeated.metric_map.data, atol=1e-6, rtol=1e-5
    )
    conservation = result.provenance["ig_result"]["conservation"]
    assert conservation["pass"] == (True,)
    assert conservation["abs"][0] <= IG_ATOL + IG_RTOL * abs(conservation["delta"][0])
    assert result.provenance["ig_result"]["metadata"]["intervention_calls"] >= 3
    assert result.provenance["ig_result"]["metadata"]["pre_projection_calls"] >= 3


def test_stage_b_single_channel_api_remains_compatible_after_c4_import():
    # The C4 implementation must not mutate or relabel the Stage B audit path.
    model = _ToyValueModel().eval()
    image = _clean_image()
    result = conditional_fixed_attention_value_path_ig(
        model, image, torch.zeros_like(image), block_index=0, head=1,
        query_token=0, head_channel=1, steps=(2, 4, 8),
        step_chunk_size=2, atol=2e-5, rtol=2e-5,
    )

    assert result.converged
    assert result.metadata["scope"] == "fixed_attention_value_path"
    assert result.metadata["target"]["kind"] == "block_pre_projection_o_scalar"
    assert result.metadata["target"]["head_channel"] == 1
    assert "direction" not in result.metadata["target"]
    torch.testing.assert_close(
        result.conservation["sum"], result.conservation["delta"],
        atol=2e-5, rtol=2e-5,
    )


def test_ig_wrapper_accepts_first_pass_at_each_exact_schedule_step(monkeypatch):
    model, image, target = _issued_target()
    seen: list[tuple[int, ...]] = []

    for selected in IG_STEPS:
        prefix = IG_STEPS[: IG_STEPS.index(selected) + 1]

        def fake(*_args, selected=selected, prefix=prefix, **_kwargs):
            seen.append(prefix)
            return _ig_result(target, steps=selected, attempt_steps=prefix)

        monkeypatch.setattr(value_path.attribution, "integrated_gradients", fake)
        result = conditional_fixed_clean_attention_head_direction_value_path_ig(
            model, image, target
        )
        assert result.provenance["ig_result"]["selected_steps"] == selected
        assert tuple(
            attempt["steps"]
            for attempt in result.provenance["ig_result"]["attempts"]
        ) == prefix

    assert seen == [IG_STEPS[: index + 1] for index in range(len(IG_STEPS))]


def test_ig_wrapper_nonconvergence_retains_all_five_attempts(monkeypatch):
    model, image, target = _issued_target()
    native = _signed_native(0.0)
    native[0, 0, 0, 0] = 1.0
    nonconverged = _ig_result(
        target,
        steps=IG_STEPS[-1],
        converged=False,
        attribution=native,
        attempt_steps=IG_STEPS,
    )
    monkeypatch.setattr(
        value_path.attribution, "integrated_gradients", lambda *_a, **_k: nonconverged
    )

    with pytest.raises(StageCValuePathConvergenceError) as captured:
        conditional_fixed_clean_attention_head_direction_value_path_ig(
            model, image, target
        )

    assert captured.value.ig_result is nonconverged
    assert captured.value.result is nonconverged
    assert captured.value.attempts is nonconverged.attempts
    assert tuple(attempt["steps"] for attempt in captured.value.attempts) == IG_STEPS
    assert all(attempt["pass"] == [False] for attempt in captured.value.attempts)


@pytest.mark.parametrize(
    ("factory", "exception", "message"),
    [
        (lambda target: object(), TypeError, "IGResult"),
        (
            lambda target: _ig_result(
                target,
                attribution=torch.zeros((1, 3, 224, 224)),
                report_delta=0.0,
            ),
            StageCValuePathError,
            "all zero",
        ),
        (
            lambda target: _ig_result(
                target, attribution=_signed_native(target.r).to(torch.float64)
            ),
            TypeError,
            "float32",
        ),
        (
            lambda target: _ig_result(
                target,
                attribution=torch.nn.functional.pad(
                    _signed_native(target.r)[:, :, :223], (0, 0, 0, 0)
                ),
            ),
            StageCValuePathError,
            "shape",
        ),
        (
            lambda target: _ig_result(
                target, metadata_updates={"step_chunk_size": 16}
            ),
            StageCValuePathError,
            "step_chunk_size",
        ),
        (
            lambda target: _ig_result(
                target, steps=1024, attempt_steps=(1024,)
            ),
            StageCValuePathError,
            "schedule prefix",
        ),
    ],
)
def test_ig_wrapper_fails_closed_on_invalid_controlled_result(
    monkeypatch, factory, exception, message
):
    model, image, target = _issued_target()
    _mixed_caller_state(model)
    caller_hook = model.blocks[5].register_forward_hook(
        lambda _module, _arguments, _output: None
    )
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)
    monkeypatch.setattr(
        value_path.attribution,
        "integrated_gradients",
        lambda *_args, **_kwargs: factory(target),
    )
    try:
        with pytest.raises(exception, match=message):
            conditional_fixed_clean_attention_head_direction_value_path_ig(
                model, image, target
            )
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
        assert caller_hook.id in model.blocks[5]._forward_hooks
    finally:
        caller_hook.remove()


def test_ig_endpoint_delta_identity_uses_registered_conservation_gate(monkeypatch):
    model, image, target = _issued_target()
    expected = float(torch.sum(target.u * target.d).item())
    bad = _ig_result(
        target,
        attribution=_signed_native(expected + 1.0),
        report_delta=expected + 1.0,
    )
    monkeypatch.setattr(
        value_path.attribution, "integrated_gradients", lambda *_a, **_k: bad
    )
    with pytest.raises(StageCValuePathError, match="endpoint delta"):
        conditional_fixed_clean_attention_head_direction_value_path_ig(
            model, image, target
        )


@pytest.mark.parametrize("should_raise", [False, True], ids=["success", "error"])
def test_ig_wrapper_restores_hooks_all_modes_requires_grad_and_grad_objects(
    monkeypatch, should_raise
):
    model, image, target = _issued_target()
    _mixed_caller_state(model)
    caller_hook = model.blocks[6].register_forward_pre_hook(
        lambda _module, _arguments: None
    )
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)

    def fake(received_forward, *_args, **_kwargs):
        assert not any(_mode_flags(model))
        assert not any(_requires_grad_flags(model))
        model.blocks[7].train()
        parameters = tuple(model.parameters())
        parameters[1].requires_grad_(False)
        parameters[1].grad.add_(91.0)
        received_forward(image)
        if should_raise:
            raise RuntimeError("synthetic controlled IG failure")
        return _ig_result(target)

    monkeypatch.setattr(value_path.attribution, "integrated_gradients", fake)
    try:
        if should_raise:
            with pytest.raises(RuntimeError, match="controlled IG failure"):
                conditional_fixed_clean_attention_head_direction_value_path_ig(
                    model, image, target
                )
        else:
            result = conditional_fixed_clean_attention_head_direction_value_path_ig(
                model, image, target
            )
            assert result.na is None
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
        assert caller_hook.id in model.blocks[6]._forward_pre_hooks
    finally:
        caller_hook.remove()


def test_caller_attention_hook_cannot_override_fixed_clean_attention():
    model, image, target = _issued_target(block_index=2, head_index=3)

    def hostile_attention_hook(_module, _arguments, output):
        changed = output.clone()
        changed[:, target.head_index] = 0.0
        changed[:, target.head_index, :, -1] = 1.0
        return changed

    handle = model.blocks[2].attn.attn_drop.register_forward_hook(
        hostile_attention_hook
    )
    try:
        attention_module = model.blocks[2].attn
        with value_path._SelectedHeadIntervention(
            attention_module, {target.head_index: target.fixed_attention}
        ) as intervention, value_path._EndpointCaptures(
            {2: attention_module}
        ) as captures:
            with torch.no_grad():
                model(image)
        captured = captures.finish()[2].attention_used
        torch.testing.assert_close(
            captured[:, target.head_index], target.fixed_attention,
            atol=0.0, rtol=0.0,
        )
        assert intervention.call_count == 1
        assert handle.id in attention_module.attn_drop._forward_hooks
    finally:
        handle.remove()


def test_resolver_restores_lifecycle_on_success_and_caller_hook_error():
    model = _ToyValueModel().eval()
    image = _clean_image()
    _mixed_caller_state(model)
    passive = model.blocks[2].attn.qkv.register_forward_hook(
        lambda _module, _arguments, _output: None
    )
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)
    try:
        target = resolve_fixed_clean_attention_head_direction(model, image, 2, 3)
        assert target.status == STATUS_OK
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
        assert passive.id in model.blocks[2].attn.qkv._forward_hooks
    finally:
        passive.remove()

    def mutate_and_fail(_module, _arguments, _output):
        model.blocks[8].train()
        parameter = tuple(model.parameters())[1]
        parameter.requires_grad_(False)
        parameter.grad.add_(-500.0)
        raise RuntimeError("synthetic caller endpoint hook failure")

    failing = model.blocks[2].attn.qkv.register_forward_hook(mutate_and_fail)
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)
    try:
        with pytest.raises(RuntimeError, match="caller endpoint hook failure"):
            resolve_fixed_clean_attention_head_direction(model, image, 2, 3)
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
        assert failing.id in model.blocks[2].attn.qkv._forward_hooks
    finally:
        failing.remove()


def test_resolver_owns_clean_input_and_reports_in_place_mutation():
    model = _ToyValueModel().eval()
    image = _clean_image()
    original = image.clone()

    def mutate_caller_input(_module, arguments):
        arguments[0].add_(0.25)

    handle = model.register_forward_pre_hook(mutate_caller_input)
    try:
        with pytest.raises(StageCValuePathError, match="private clean-input clone"):
            resolve_fixed_clean_attention_head_direction(model, image, 2, 3)
        # The model received a private clone, so caller-owned bytes survive.
        torch.testing.assert_close(image, original, atol=0.0, rtol=0.0)
        assert handle.id in model._forward_pre_hooks
    finally:
        handle.remove()


def test_resolver_cleans_partial_hook_registration_failure_and_restores_state():
    model = _ToyValueModel(registration_failure_block=4).eval()
    image = _clean_image()
    _mixed_caller_state(model)
    persistent = model.blocks[4].attn.qkv.register_forward_hook(
        lambda _module, _arguments, _output: None
    )
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)
    try:
        with pytest.raises(RuntimeError, match="projection hook registration failure"):
            resolve_fixed_clean_attention_head_direction(model, image, 4, 1)
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
        assert persistent.id in model.blocks[4].attn.qkv._forward_hooks
    finally:
        persistent.remove()


def test_ig_wrapper_cleans_partial_hook_registration_failure_and_restores_state():
    model, image, target = _issued_target(block_index=4, head_index=1)
    _mixed_caller_state(model)
    model.blocks[4].attn.proj.fail_registration = True
    persistent = model.blocks[4].attn.qkv.register_forward_hook(
        lambda _module, _arguments, _output: None
    )
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)
    try:
        with pytest.raises(RuntimeError, match="projection hook registration failure"):
            conditional_fixed_clean_attention_head_direction_value_path_ig(
                model, image, target
            )
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
        assert persistent.id in model.blocks[4].attn.qkv._forward_hooks
    finally:
        persistent.remove()


def test_frozen_target_rejects_direct_construction_tamper_input_and_model_drift(
    monkeypatch,
):
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        FrozenHeadDirectionTarget()

    model, image, target = _issued_target()
    changed_input = image.clone()
    changed_input[0, 0, 0, 0] += 1.0
    monkeypatch.setattr(
        value_path.attribution,
        "integrated_gradients",
        lambda *_args, **_kwargs: _ig_result(target),
    )
    with pytest.raises(StageCValuePathError, match="input bytes"):
        conditional_fixed_clean_attention_head_direction_value_path_ig(
            model, changed_input, target
        )

    original_status = target.resolution.status
    object.__setattr__(target.resolution, "status", DEGENERATE_DIRECTION)
    try:
        with pytest.raises(StageCValuePathError, match="mutated after issue"):
            matched_fixed_clean_attention_patch_token_contribution(target)
    finally:
        object.__setattr__(target.resolution, "status", original_status)

    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(StageCValuePathError, match="state changed"):
        matched_fixed_clean_attention_patch_token_contribution(target)


def test_degenerate_issued_target_wrappers_return_same_structured_na(monkeypatch):
    model = _ToyValueModel().eval()
    image = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    target = resolve_fixed_clean_attention_head_direction(model, image, 0, 0)
    assert target.status == DEGENERATE_DIRECTION
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("IG must not run for a degenerate direction")

    monkeypatch.setattr(value_path.attribution, "integrated_gradients", forbidden)
    nns = conditional_fixed_clean_attention_head_direction_value_path_ig(
        model, image, target
    )
    matched = matched_fixed_clean_attention_patch_token_contribution(target)

    assert called is False
    for result in (nns, matched):
        assert result.na is not None
        assert result.na.reason is NAReason.DEGENERATE_DIRECTION
        assert result.native_tensor is None
        assert result.metric_map is None
        assert not result.valid_mask.any()
        assert result.provenance["status"] == DEGENERATE_DIRECTION


@pytest.mark.integration
def test_real_cpu_resolver_smoke_endpoint_parity_a_v_and_token_identity(
    state_dict_path,
):
    """One cheap canonical smoke: endpoint resolver only, never high-step IG."""

    torch.manual_seed(20260821)
    model = load_canonical_model(
        state_dict_path, device="cpu", unfused_attention=True
    )
    image = torch.randn((1, 3, 224, 224), device="cpu", dtype=torch.float32)
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)

    target = resolve_fixed_clean_attention_head_direction(model, image, 4, 9)

    assert next(model.parameters()).device.type == "cpu"
    assert target.candidate == HeadCandidate(4, 9)
    assert target.A.device.type == "cpu"
    assert target.A.shape == (TOKENS,)
    assert target.clean_values.shape == (TOKENS, HEAD_DIM)
    assert target.baseline_values.shape == (TOKENS, HEAD_DIM)
    assert target.clean_values[0].shape == (HEAD_DIM,)
    assert target.clean_values[1:].shape == (PATCHES, HEAD_DIM)
    assert target.clean_value_max_abs_error < 1e-6
    assert target.clean_replay_max_abs_error < 1e-6
    assert target.clean_model_output_max_abs_error < 1e-6
    assert target.baseline_replay_max_abs_error < 1e-6
    torch.testing.assert_close(
        target.g_clean, target.A @ target.clean_values, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        target.g_baseline, target.A @ target.baseline_values, atol=1e-6, rtol=1e-5
    )
    assert _hook_inventory(model) == hooks_before
    assert _mode_flags(model) == modes_before
    assert _requires_grad_flags(model) == requires_before
    _assert_grad_snapshot(model, grads_before)
