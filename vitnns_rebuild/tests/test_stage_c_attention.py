from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np
import pytest
import torch
from torch import nn

from vitnns.extractor import ExtractionResult, LayerCapture, NativeViTExtractor
from vitnns.modeling import load_canonical_model
from vitnns.retention import ArtifactKind
from vitnns.stage_c_attention import (
    METHOD_VERSION,
    StageCAttentionError,
    attention_rollout_mean_head_residual_all_blocks,
    class_attention_maps_at_intervention,
    direct_raw_cls_attention_last_block_head_mean,
    direct_raw_cls_attention_per_head,
    rollout_attention_matrices,
)
from vitnns.stage_c_maps import ComparisonLine, MethodResult, SignSemantics


BLOCKS = 12
HEADS = 12
TOKENS = 197
PATCHES = 196
PATCH_SHAPE = (14, 14)
ATTENTION_SOURCE = "NativeViTExtractor.LayerCapture.attention_used"


class _ToyExtractor:
    def __init__(self, extraction):
        self.extraction = extraction
        self.inputs = None

    def extract(self, inputs):
        self.inputs = inputs
        return self.extraction


def _zero_attention(
    *,
    batch: int = 1,
    heads: int = HEADS,
    query_tokens: int = TOKENS,
    key_tokens: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    key_tokens = query_tokens if key_tokens is None else key_tokens
    return torch.zeros((batch, heads, query_tokens, key_tokens), dtype=dtype)


def _layer(attention_used: torch.Tensor) -> LayerCapture:
    # These intentionally-invalid sentinels make a wrapper fail if it consults
    # pre-dropout/recomputed attention instead of the authoritative value used
    # by the real forward.  C2 only needs attention_used from each capture.
    forbidden = torch.tensor(float("nan"))
    unused = torch.empty(0)
    return LayerCapture(
        block_input=unused,
        block_output=unused,
        q=unused,
        k=unused,
        v=unused,
        attention=forbidden,
        attention_used=attention_used,
        attention_recomputed=forbidden,
        scale=1.0,
    )


def _extraction(
    overrides: Mapping[int, torch.Tensor] | None = None,
    *,
    layer_count: int = BLOCKS,
) -> ExtractionResult:
    overrides = dict(overrides or {})
    # Expanded zero views keep the 12 x 12 x 197 x 197 toy geometry cheap.
    zero = torch.zeros((1, 1, TOKENS, TOKENS)).expand(1, HEADS, TOKENS, TOKENS)
    return ExtractionResult(
        logits=torch.zeros((1, 3)),
        layers=tuple(_layer(overrides.get(index, zero)) for index in range(layer_count)),
    )


def _expected_patch_map(row: torch.Tensor) -> np.ndarray:
    patches = row[1:]
    return (patches / patches.sum()).to(torch.float32).numpy().reshape(PATCH_SHAPE)


def _assert_method_result(
    result: MethodResult,
    *,
    method_id: str,
    comparison_line: ComparisonLine,
    fidelity_target: str | None,
    projection: str,
) -> None:
    assert isinstance(result, MethodResult)
    assert result.method_id == method_id
    assert result.method_version == METHOD_VERSION == "1.0.0"
    assert result.comparison_line is comparison_line
    assert result.explanation_target is None
    assert result.fidelity_target == fidelity_target
    assert result.class_agnostic is True
    assert result.projection == projection
    assert result.sign_semantics is SignSemantics.NONNEGATIVE
    assert result.na is None
    assert result.native_tensor is not None
    assert result.native_tensor.kind is ArtifactKind.RAW
    assert result.native_tensor.shape == PATCH_SHAPE
    assert result.native_tensor.dtype == np.dtype(np.float32)
    assert result.metric_map is not None
    assert result.metric_map.kind is ArtifactKind.METRIC_READY
    assert result.metric_map.shape == PATCH_SHAPE
    assert result.metric_map.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(result.native_tensor.data, result.metric_map.data)
    assert np.all(result.metric_map.data >= 0)
    assert float(result.metric_map.data.sum(dtype=np.float64)) == pytest.approx(
        1.0, abs=1e-6, rel=1e-6
    )
    assert result.valid_mask.dtype == np.bool_
    assert result.valid_mask.shape == PATCH_SHAPE
    assert bool(result.valid_mask.all())
    assert result.provenance["stage"] == "C2"
    assert result.provenance["method_implementation_version"] == METHOD_VERSION
    assert result.provenance["attention_source"] == ATTENTION_SOURCE
    assert result.provenance["class_agnostic"] is True


def test_per_head_direct_reads_only_attention_used_and_exact_matched_block_head():
    block_index, head_index = 7, 5
    selected = _zero_attention()
    patch_values = torch.arange(1, PATCHES + 1, dtype=torch.float32)
    selected[0, head_index, 0, 0] = 1_000_000.0  # Must be excluded.
    selected[0, head_index, 0, 1:] = patch_values
    selected[0, head_index + 1, 0, 1:] = patch_values.flip(0)

    decoy_block = _zero_attention()
    decoy_block[0, head_index, 0, 1] = 1.0
    selected_row_before = selected[0, head_index, 0].clone()
    caller_context = {
        "sample_id": "toy",
        "attention_source": "caller-cannot-override",
        "nested": {"indices": [block_index, head_index]},
    }
    result = direct_raw_cls_attention_per_head(
        _extraction({block_index - 1: decoy_block, block_index: selected}),
        block_index=block_index,
        head_index=head_index,
        provenance=caller_context,
    )

    np.testing.assert_array_equal(
        result.metric_map.data, _expected_patch_map(selected_row_before)
    )
    torch.testing.assert_close(selected[0, head_index, 0], selected_row_before)
    _assert_method_result(
        result,
        method_id="direct_raw_cls_attention_per_head",
        comparison_line=ComparisonLine.VALUE_PATH,
        fidelity_target=None,
        projection="raw_cls_attention_to_patch",
    )
    assert set(result.provenance) == {
        "stage",
        "method_implementation_version",
        "attention_source",
        "class_agnostic",
        "algorithm",
        "caller_context",
    }
    algorithm = result.provenance["algorithm"]
    assert algorithm["block_index"] == block_index
    assert algorithm["head_index"] == head_index
    assert algorithm["query_token"] == 0
    assert algorithm["exclude_cls_key"] is True
    assert algorithm["patch_count"] == PATCHES
    assert algorithm["head_fusion"] == "none"
    assert result.provenance["caller_context"]["attention_source"] == (
        "caller-cannot-override"
    )

    # Caller provenance is copied and deeply frozen, rather than being able to
    # smuggle mutable raw captures into or overwrite the method provenance.
    caller_context["nested"]["indices"].append(99)
    assert result.provenance["caller_context"]["nested"]["indices"] == (7, 5)
    with pytest.raises(TypeError):
        result.provenance["attention_source"] = "tampered"


def test_class_attention_intervention_recomputes_both_maps_and_binds_tensor_hash():
    used = _zero_attention()
    used[:, :, :, 1:] = torch.arange(1, PATCHES + 1, dtype=torch.float32)
    extractor = _ToyExtractor(_extraction({index: used for index in range(BLOCKS)}))
    intervention = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    identity = {
        "comparison_line": "class_logit",
        "target_ref": "clean_predicted_class_raw_logit",
        "clean_input_sha256": "a" * 64,
        "frozen_target": {"target_kind": "raw_logit"},
    }
    direct, rollout = class_attention_maps_at_intervention(
        extractor, intervention, fidelity_target_identity=identity,
        provenance={"sample_id": "toy"},
    )
    assert extractor.inputs is intervention
    for result in (direct, rollout):
        context = result.provenance["caller_context"]
        assert context["sample_id"] == "toy"
        assert context["fidelity_target_identity"]["target_ref"] == (
            "clean_predicted_class_raw_logit"
        )
        assert len(context["intervention_input_sha256"]) == 64


def test_per_head_direct_excludes_cls_preserves_patch_order_and_reshapes_196():
    attention_used = _zero_attention()
    row = attention_used[0, 3, 0]
    row[0] = 1e20
    row[1:] = torch.arange(PATCHES, dtype=torch.float32) + 0.5

    result = direct_raw_cls_attention_per_head(
        _extraction({2: attention_used}), block_index=2, head_index=3
    )

    expected = _expected_patch_map(row)
    np.testing.assert_array_equal(result.metric_map.data, expected)
    assert result.metric_map.data[0, 0] == expected[0, 0]
    assert result.metric_map.data[0, 13] == expected[0, 13]
    assert result.metric_map.data[1, 0] == expected[1, 0]
    assert result.metric_map.data[13, 13] == expected[13, 13]
    assert float(result.metric_map.data.sum(dtype=np.float64)) == pytest.approx(1.0)


def test_class_direct_uses_block_11_and_means_12_heads_before_normalizing():
    attention_used = _zero_attention()
    for head in range(HEADS):
        patch_mass = (head + 1) / 20.0
        attention_used[0, head, 0, 0] = 1.0 - patch_mass
        attention_used[0, head, 0, head + 1] = patch_mass

    decoy_block_10 = _zero_attention()
    decoy_block_10[0, :, 0, -1] = 1.0
    result = direct_raw_cls_attention_last_block_head_mean(
        _extraction({10: decoy_block_10, 11: attention_used})
    )

    raw_head_mean = attention_used[0].mean(dim=0)[0]
    expected = _expected_patch_map(raw_head_mean)
    np.testing.assert_array_equal(result.metric_map.data, expected)

    # Normalizing each head first would incorrectly weight all heads equally.
    wrong = torch.stack(
        [
            attention_used[0, head, 0, 1:]
            / attention_used[0, head, 0, 1:].sum()
            for head in range(HEADS)
        ]
    ).mean(dim=0).numpy().reshape(PATCH_SHAPE)
    assert not np.allclose(result.metric_map.data, wrong)

    _assert_method_result(
        result,
        method_id="direct_raw_cls_attention_last_block_head_mean",
        comparison_line=ComparisonLine.CLASS_LOGIT,
        fidelity_target="clean_predicted_class_raw_logit",
        projection="raw_cls_attention_to_patch",
    )
    algorithm = result.provenance["algorithm"]
    assert algorithm["block_index"] == 11
    assert algorithm["head_count"] == HEADS
    assert algorithm["head_fusion"] == (
        "arithmetic_mean_before_cls_exclusion_and_renormalization"
    )
    assert algorithm["query_token"] == 0
    assert algorithm["exclude_cls_key"] is True
    assert algorithm["patch_count"] == PATCHES


def test_rollout_core_hand_computes_head_mean_identity_and_row_normalization():
    attention = torch.tensor(
        [
            [
                [[0.0, 2.0], [1.0, 0.0]],
                [[0.0, 0.0], [3.0, 0.0]],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    before = attention.detach().clone()

    rollout = rollout_attention_matrices([attention])

    # mean_heads(A)=[[0,1],[2,0]], so I+mean(A)=[[1,1],[2,1]].
    expected = torch.tensor([[0.5, 0.5], [2.0 / 3.0, 1.0 / 3.0]], dtype=torch.float64)
    torch.testing.assert_close(rollout, expected, atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(rollout.sum(dim=-1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(attention.detach(), before)
    assert rollout.shape == (2, 2)
    assert rollout.dtype is torch.float64
    assert rollout.device.type == "cpu"
    assert rollout.requires_grad is False


def test_rollout_core_zero_attention_is_identity_and_single_layer_degenerates_to_b0():
    identity = rollout_attention_matrices([torch.zeros((1, 3, 4, 4))])
    torch.testing.assert_close(identity, torch.eye(4), atol=0, rtol=0)

    attention = torch.tensor(
        [[[[0.0, 3.0, 1.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]]]],
        dtype=torch.float32,
    )
    residual = torch.eye(3) + attention[0, 0]
    expected = residual / residual.sum(dim=-1, keepdim=True)
    actual = rollout_attention_matrices((attention,))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_rollout_core_left_multiplies_noncommuting_layers_in_frozen_order():
    a0 = torch.tensor(
        [[[[0.0, 1.0], [0.0, 1.0]]]], dtype=torch.float64
    )
    a1 = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float64
    )
    b0 = torch.tensor([[0.5, 0.5], [0.0, 1.0]], dtype=torch.float64)
    b1 = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float64)

    rollout = rollout_attention_matrices([a0, a1])

    torch.testing.assert_close(rollout, b1 @ b0, atol=0, rtol=0)
    assert not torch.allclose(rollout, b0 @ b1)
    torch.testing.assert_close(rollout.sum(dim=-1), torch.ones(2, dtype=torch.float64))
    assert bool(torch.isfinite(rollout).all())


def test_rollout_wrapper_uses_all_12_blocks_and_returns_exact_method_contract():
    first = _zero_attention()
    patches = torch.arange(1, PATCHES + 1, dtype=torch.float32)
    first[0, :, 0, 0] = 17.0
    first[0, :, 0, 1:] = patches
    last = _zero_attention()
    last[0, :, 0, -1] = 1.0
    # A non-identity block 11 makes omission or an early rollout cutoff
    # observably different from the frozen all-block result.
    result = attention_rollout_mean_head_residual_all_blocks(
        _extraction({0: first, 11: last}), provenance={"sample_id": "rollout-toy"}
    )

    zero = torch.zeros((1, HEADS, TOKENS, TOKENS), dtype=torch.float32)
    expected_rollout = rollout_attention_matrices(
        [first, *([zero] * (BLOCKS - 2)), last]
    )
    expected = _expected_patch_map(expected_rollout[0])
    np.testing.assert_array_equal(result.metric_map.data, expected)
    without_last = attention_rollout_mean_head_residual_all_blocks(
        _extraction({0: first})
    )
    assert not np.array_equal(result.metric_map.data, without_last.metric_map.data)
    _assert_method_result(
        result,
        method_id="attention_rollout_mean_head_residual_all_blocks",
        comparison_line=ComparisonLine.CLASS_LOGIT,
        fidelity_target="clean_predicted_class_raw_logit",
        projection="rollout_cls_to_patch",
    )
    algorithm = result.provenance["algorithm"]
    assert algorithm["blocks"] == tuple(range(BLOCKS))
    assert algorithm["head_count"] == HEADS
    assert algorithm["head_fusion"] == "arithmetic_mean"
    assert algorithm["residual"] == "identity_coefficient_1"
    assert algorithm["row_normalization"] == "after_identity_addition"
    assert algorithm["multiplication_order"] == "B11_times_B10_times_..._times_B0"
    assert algorithm["query_token"] == 0
    assert algorithm["exclude_cls_key"] is True
    assert algorithm["patch_count"] == PATCHES


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("block_index", True, TypeError, "block_index"),
        ("block_index", -1, StageCAttentionError, "block_index"),
        ("block_index", BLOCKS, StageCAttentionError, "block_index"),
        ("head_index", False, TypeError, "head_index"),
        ("head_index", -1, StageCAttentionError, "head_index"),
        ("head_index", HEADS, StageCAttentionError, "head_index"),
    ],
)
def test_per_head_direct_rejects_bool_negative_and_out_of_range_indices(
    field, value, exception, message
):
    kwargs = {"block_index": 0, "head_index": 0}
    kwargs[field] = value
    with pytest.raises(exception, match=message):
        direct_raw_cls_attention_per_head(_extraction(), **kwargs)


@pytest.mark.parametrize(
    ("attention", "message"),
    [
        (torch.zeros((HEADS, TOKENS, TOKENS)), "shape"),
        (_zero_attention(batch=2), "batch size 1"),
        (_zero_attention(heads=HEADS - 1), "expected 12 heads"),
        (_zero_attention(query_tokens=TOKENS - 1), "expected 197 tokens"),
        (_zero_attention(key_tokens=TOKENS - 1), "square token matrix"),
    ],
)
def test_direct_rejects_invalid_attention_rank_batch_head_and_token_geometry(
    attention, message
):
    with pytest.raises(StageCAttentionError, match=message):
        direct_raw_cls_attention_per_head(
            _extraction({0: attention}), block_index=0, head_index=0
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
        (float("-inf"), "non-finite"),
        (-0.25, "negative"),
    ],
)
def test_direct_rejects_nonfinite_or_negative_attention_used(value, message):
    attention_used = _zero_attention()
    attention_used[0, 0, 0, 1] = value
    with pytest.raises(StageCAttentionError, match=message):
        direct_raw_cls_attention_per_head(
            _extraction({0: attention_used}), block_index=0, head_index=0
        )


def test_direct_and_rollout_reject_zero_patch_mass_after_cls_exclusion():
    cls_only = _zero_attention()
    cls_only[0, :, 0, 0] = 1.0
    with pytest.raises(StageCAttentionError, match="finite and positive"):
        direct_raw_cls_attention_per_head(
            _extraction({0: cls_only}), block_index=0, head_index=0
        )
    with pytest.raises(StageCAttentionError, match="finite and positive"):
        direct_raw_cls_attention_last_block_head_mean(
            _extraction({11: cls_only})
        )

    # Zero attention makes every residual transition the identity, leaving no
    # CLS-to-patch mass in the final rollout.
    with pytest.raises(StageCAttentionError, match="finite and positive"):
        attention_rollout_mean_head_residual_all_blocks(_extraction())


@pytest.mark.parametrize(
    "wrapper",
    [
        lambda extraction: direct_raw_cls_attention_per_head(
            extraction, block_index=0, head_index=0
        ),
        direct_raw_cls_attention_last_block_head_mean,
        attention_rollout_mean_head_residual_all_blocks,
    ],
    ids=["per-head-direct", "class-direct", "rollout"],
)
@pytest.mark.parametrize("layer_count", [0, BLOCKS - 1, BLOCKS + 1])
def test_all_public_wrappers_reject_wrong_layer_count(wrapper, layer_count):
    with pytest.raises(StageCAttentionError, match="exactly 12 captured blocks"):
        wrapper(_extraction(layer_count=layer_count))


def test_public_wrappers_require_real_extraction_result_objects():
    with pytest.raises(TypeError, match="ExtractionResult"):
        direct_raw_cls_attention_per_head(
            {"layers": []}, block_index=0, head_index=0
        )
    with pytest.raises(TypeError, match="ExtractionResult"):
        direct_raw_cls_attention_last_block_head_mean(object())
    with pytest.raises(TypeError, match="ExtractionResult"):
        attention_rollout_mean_head_residual_all_blocks(object())


@pytest.mark.parametrize(
    ("bad_layer", "exception", "message"),
    [
        (torch.zeros((1, 2, 2)), StageCAttentionError, "shape"),
        (torch.zeros((2, 1, 2, 2)), StageCAttentionError, "batch size 1"),
        (torch.zeros((1, 0, 2, 2)), StageCAttentionError, "positive count"),
        (torch.zeros((1, 1, 2, 3)), StageCAttentionError, "square token matrix"),
        (torch.zeros((1, 1, 1, 1)), StageCAttentionError, "CLS plus patches"),
        (torch.zeros((1, 1, 2, 2), dtype=torch.int64), TypeError, "float32 or float64 dtype"),
        (torch.zeros((1, 1, 2, 2), dtype=torch.bool), TypeError, "float32 or float64 dtype"),
    ],
)
def test_rollout_core_rejects_invalid_rank_batch_head_token_and_dtype(
    bad_layer, exception, message
):
    with pytest.raises(exception, match=message):
        rollout_attention_matrices([bad_layer])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
        (float("-inf"), "non-finite"),
        (-1.0, "negative"),
    ],
)
def test_rollout_core_rejects_nonfinite_and_negative_attention(value, message):
    layer = torch.zeros((1, 1, 2, 2))
    layer[0, 0, 0, 1] = value
    with pytest.raises(StageCAttentionError, match=message):
        rollout_attention_matrices([layer])


def test_rollout_core_rejects_empty_implicit_or_inconsistent_layer_collections():
    with pytest.raises(StageCAttentionError, match="at least one layer"):
        rollout_attention_matrices([])
    with pytest.raises(TypeError, match="explicit sequence"):
        rollout_attention_matrices(torch.zeros((1, 1, 2, 2)))
    with pytest.raises(StageCAttentionError, match="expected 1 heads"):
        rollout_attention_matrices(
            [torch.zeros((1, 1, 2, 2)), torch.zeros((1, 2, 2, 2))]
        )
    with pytest.raises(StageCAttentionError, match="expected 2 tokens"):
        rollout_attention_matrices(
            [torch.zeros((1, 1, 2, 2)), torch.zeros((1, 1, 3, 3))]
        )
    with pytest.raises(StageCAttentionError, match="share device and floating dtype"):
        rollout_attention_matrices(
            [
                torch.zeros((1, 1, 2, 2), dtype=torch.float32),
                torch.zeros((1, 1, 2, 2), dtype=torch.float64),
            ]
        )


def test_provenance_rejects_non_mapping_nonfinite_and_raw_tensor_payloads():
    attention_used = _zero_attention()
    attention_used[0, 0, 0, 1] = 1.0
    extraction = _extraction({0: attention_used})

    with pytest.raises(TypeError, match="mapping"):
        direct_raw_cls_attention_per_head(
            extraction, block_index=0, head_index=0, provenance=["not", "mapping"]
        )
    with pytest.raises(ValueError, match="non-finite"):
        direct_raw_cls_attention_per_head(
            extraction, block_index=0, head_index=0, provenance={"loss": float("nan")}
        )
    with pytest.raises(TypeError, match="JSON-compatible"):
        direct_raw_cls_attention_per_head(
            extraction, block_index=0, head_index=0, provenance={"raw": attention_used}
        )


def test_method_result_retains_only_final_patch_maps_not_attention_cartesian_product():
    attention_used = _zero_attention()
    attention_used[0, 0, 0, 1:] = 1.0
    extraction = _extraction({0: attention_used})
    result = direct_raw_cls_attention_per_head(
        extraction, block_index=0, head_index=0
    )

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
    assert not any(isinstance(value, torch.Tensor) for value in vars(result).values())
    assert result.native_tensor.data.size == PATCHES
    assert result.metric_map.data.size == PATCHES
    assert result.valid_mask.size == PATCHES
    assert set(result.provenance["algorithm"]) == {
        "block_index",
        "head_index",
        "query_token",
        "head_fusion",
        "exclude_cls_key",
        "patch_count",
        "patch_mass_normalization",
        "comparison_label",
    }


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 2
        self.head_dim = 2
        self.scale = self.head_dim**-0.5
        self.fused_attn = False
        self.qkv = nn.Linear(4, 12, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, token_count, width = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, token_count, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = self.q_norm(query)
        key = self.k_norm(key)
        attention = ((query * self.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
        attention = self.attn_drop(attention)
        return (attention @ value).transpose(1, 2).reshape(batch, token_count, width)


class _TinyBlock(nn.Module):
    def __init__(self, *, fail_registration: bool = False) -> None:
        super().__init__()
        self.attn = _TinyAttention()
        self.fail_registration = fail_registration

    def register_forward_hook(self, *args, **kwargs):
        if self.fail_registration:
            raise RuntimeError("synthetic registration failure")
        return super().register_forward_hook(*args, **kwargs)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.attn(tokens)


class _TinyModel(nn.Module):
    def __init__(self, *, fail_registration: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(fail_registration=fail_registration)])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.mean(dim=1)


def _hook_inventory(model: nn.Module) -> tuple[tuple[int, ...], ...]:
    return tuple(
        (
            *module._forward_pre_hooks.keys(),
            -1,
            *module._forward_hooks.keys(),
        )
        for module in model.modules()
    )


def _training_flags(model: nn.Module) -> tuple[bool, ...]:
    return tuple(module.training for module in model.modules())


def test_extractor_restores_hooks_and_mode_after_success():
    tokens = torch.randn((1, 3, 4), generator=torch.Generator().manual_seed(9))
    model = _TinyModel().eval()
    # A caller-owned hook must survive cleanup unchanged.
    persistent = model.blocks[0].attn.qkv.register_forward_hook(
        lambda _module, _args, _output: None
    )
    try:
        hooks_before = _hook_inventory(model)
        modes_before = _training_flags(model)
        extracted = NativeViTExtractor(model).extract(tokens)
        assert len(extracted.layers) == 1
        assert torch.equal(
            extracted.layers[0].attention, extracted.layers[0].attention_used
        )
        assert _hook_inventory(model) == hooks_before
        assert _training_flags(model) == modes_before
        assert persistent.id in model.blocks[0].attn.qkv._forward_hooks
    finally:
        persistent.remove()


def test_extractor_restores_hooks_and_mode_when_existing_forward_hook_raises():
    model = _TinyModel().eval()

    def fail_from_caller_hook(_module, _args, _output):
        raise RuntimeError("synthetic caller forward-hook failure")

    persistent = model.blocks[0].attn.qkv.register_forward_hook(fail_from_caller_hook)
    try:
        hooks_before = _hook_inventory(model)
        modes_before = _training_flags(model)
        with pytest.raises(RuntimeError, match="caller forward-hook failure"):
            NativeViTExtractor(model).extract(torch.zeros((1, 3, 4)))
        assert _hook_inventory(model) == hooks_before
        assert _training_flags(model) == modes_before
        assert persistent.id in model.blocks[0].attn.qkv._forward_hooks
        assert model.blocks[0].attn.qkv._forward_hooks[persistent.id] is fail_from_caller_hook
    finally:
        persistent.remove()


def test_extractor_cleans_partial_registration_and_revalidates_eval_state():
    registration_failure = _TinyModel(fail_registration=True).eval()
    hooks_before = _hook_inventory(registration_failure)
    extractor = NativeViTExtractor(registration_failure)
    with pytest.raises(RuntimeError, match="synthetic registration failure"):
        extractor.extract(torch.zeros((1, 3, 4)))
    assert _hook_inventory(registration_failure) == hooks_before
    assert not any(_training_flags(registration_failure))

    changed_mode = _TinyModel().eval()
    extractor = NativeViTExtractor(changed_mode)
    changed_mode.train()
    hooks_before = _hook_inventory(changed_mode)
    modes_before = _training_flags(changed_mode)
    with pytest.raises(ValueError, match="every model module.*eval mode"):
        extractor.extract(torch.zeros((1, 3, 4)))
    assert _hook_inventory(changed_mode) == hooks_before
    assert _training_flags(changed_mode) == modes_before
    assert all(modes_before)

    mixed_mode = _TinyModel().eval()
    extractor = NativeViTExtractor(mixed_mode)
    mixed_mode.blocks[0].attn.attn_drop.train()
    assert mixed_mode.training is False
    hooks_before = _hook_inventory(mixed_mode)
    modes_before = _training_flags(mixed_mode)
    with pytest.raises(ValueError, match="every model module.*eval mode"):
        extractor.extract(torch.zeros((1, 3, 4)))
    assert _hook_inventory(mixed_mode) == hooks_before
    assert _training_flags(mixed_mode) == modes_before


@pytest.mark.integration
@torch.inference_mode()
def test_real_cpu_extractor_attention_used_and_direct_parity(state_dict_path):
    torch.manual_seed(20260821)
    model = load_canonical_model(
        state_dict_path, device="cpu", unfused_attention=True
    )
    assert next(model.parameters()).device.type == "cpu"
    hooks_before = _hook_inventory(model)
    modes_before = _training_flags(model)

    extraction = NativeViTExtractor(model).extract(
        torch.randn((1, 3, 224, 224), device="cpu")
    )

    assert _hook_inventory(model) == hooks_before
    assert _training_flags(model) == modes_before
    assert len(extraction.layers) == BLOCKS
    for layer in extraction.layers:
        assert layer.attention.device.type == "cpu"
        assert layer.attention_used.device.type == "cpu"
        assert torch.equal(layer.attention, layer.attention_used)

    block_index, head_index = 4, 9
    per_head = direct_raw_cls_attention_per_head(
        extraction, block_index=block_index, head_index=head_index
    )
    expected_per_head = _expected_patch_map(
        extraction.layers[block_index].attention_used[0, head_index, 0]
    )
    np.testing.assert_array_equal(per_head.metric_map.data, expected_per_head)

    class_direct = direct_raw_cls_attention_last_block_head_mean(extraction)
    expected_class = _expected_patch_map(
        extraction.layers[11].attention_used[0].mean(dim=0)[0]
    )
    np.testing.assert_array_equal(class_direct.metric_map.data, expected_class)
