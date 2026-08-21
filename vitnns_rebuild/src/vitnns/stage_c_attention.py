"""Stage C direct CLS-attention and mean-head residual rollout baselines.

The production entry points in this module consume :class:`ExtractionResult`
objects and read only ``LayerCapture.attention_used``.  They do not reconstruct
QK, softmax, or any Transformer block.  Both baselines are class agnostic; the
per-head direct map is only a different-target spatial-structure reference for
the Value-path line.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .extractor import ExtractionResult, LayerCapture
from .stage_c_maps import (
    ComparisonLine,
    MethodResult,
    PATCH_SHAPE,
    SignSemantics,
    raw_array,
    stage_c_metric_ready,
)
from .stage_c_class_logit import _tensor_sha256


METHOD_VERSION = "1.0.0"
MODEL_BLOCKS = 12
HEADS_PER_BLOCK = 12
TOKENS = 197
PATCH_TOKENS = 196
CLS_TOKEN_INDEX = 0
CLASS_DIRECT_BLOCK_INDEX = 11
_ATTENTION_SOURCE = "NativeViTExtractor.LayerCapture.attention_used"


class StageCAttentionError(ValueError):
    """Raised when captured attention violates the frozen C2 contract."""


def _strict_index(value: Any, *, name: str, upper_bound: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (not bool)")
    if value < 0 or value >= upper_bound:
        raise StageCAttentionError(
            f"{name} must be in [0, {upper_bound - 1}], got {value}"
        )
    return value


def _layers(extraction: ExtractionResult) -> tuple[LayerCapture, ...]:
    if not isinstance(extraction, ExtractionResult):
        raise TypeError(
            "C2 production methods require an ExtractionResult so attention "
            "comes from NativeViTExtractor"
        )
    layers = extraction.layers
    if not isinstance(layers, tuple) or len(layers) != MODEL_BLOCKS:
        raise StageCAttentionError(
            f"Stage C requires exactly {MODEL_BLOCKS} captured blocks"
        )
    if any(not isinstance(layer, LayerCapture) for layer in layers):
        raise TypeError("ExtractionResult.layers must contain LayerCapture objects")
    return layers


def _validate_attention_tensor(
    value: Any,
    *,
    name: str,
    expected_heads: int | None,
    expected_tokens: int | None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise StageCAttentionError(
            f"{name} must have shape [batch,heads,tokens,tokens], got {tuple(value.shape)}"
        )
    batch, heads, query_tokens, key_tokens = value.shape
    if batch != 1:
        raise StageCAttentionError(
            f"{name} requires single-sample batch size 1, got {batch}"
        )
    if heads <= 0 or (expected_heads is not None and heads != expected_heads):
        expected = expected_heads if expected_heads is not None else "a positive count"
        raise StageCAttentionError(
            f"{name} expected {expected} heads, got {heads}"
        )
    if query_tokens != key_tokens or query_tokens < 2:
        raise StageCAttentionError(
            f"{name} requires a square token matrix with CLS plus patches, "
            f"got {query_tokens}x{key_tokens}"
        )
    if expected_tokens is not None and query_tokens != expected_tokens:
        raise StageCAttentionError(
            f"{name} expected {expected_tokens} tokens, got {query_tokens}"
        )
    if value.dtype not in {torch.float32, torch.float64}:
        raise TypeError(
            f"{name} must have float32 or float64 dtype for fail-closed "
            f"normalization, got {value.dtype}"
        )
    detached = value.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise StageCAttentionError(f"{name} contains non-finite values")
    if bool((detached < 0).any().item()):
        raise StageCAttentionError(f"{name} contains negative attention values")
    return detached


def _normalize_cls_patch_mass(attention_row: torch.Tensor, *, name: str) -> np.ndarray:
    if attention_row.ndim != 1 or attention_row.shape[0] != TOKENS:
        raise StageCAttentionError(
            f"{name} must be one {TOKENS}-entry CLS attention row"
        )
    patch_mass = attention_row[1:]
    if patch_mass.shape[0] != PATCH_TOKENS:
        raise StageCAttentionError(
            f"{name} must contain exactly {PATCH_TOKENS} patch keys after CLS exclusion"
        )
    if not bool(torch.isfinite(patch_mass).all().item()):
        raise StageCAttentionError(f"{name} patch attention contains non-finite values")
    if bool((patch_mass < 0).any().item()):
        raise StageCAttentionError(f"{name} patch attention contains negative values")
    denominator = patch_mass.sum()
    denominator_value = float(denominator.item())
    if not np.isfinite(denominator_value) or denominator_value <= 0.0:
        raise StageCAttentionError(
            f"{name} patch attention mass must be finite and positive, "
            f"got {denominator_value}"
        )
    normalized = patch_mass / denominator
    if not bool(torch.isfinite(normalized).all().item()):
        raise StageCAttentionError(f"{name} normalized patch attention is non-finite")
    result = normalized.to(device="cpu", dtype=torch.float32).numpy().reshape(PATCH_SHAPE)
    if np.any(result < 0) or not np.all(np.isfinite(result)):
        raise StageCAttentionError(f"{name} normalized patch map is invalid")
    mass = float(np.sum(result, dtype=np.float64))
    if not np.isclose(mass, 1.0, atol=1e-6, rtol=1e-6):
        raise StageCAttentionError(
            f"{name} normalized float32 patch mass does not sum to one: {mass}"
        )
    return np.asarray(result, dtype=np.float32)


def _caller_context(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if provenance is None:
        return {}
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping or None")
    return dict(provenance)


def _result(
    patch_map: np.ndarray,
    *,
    method_id: str,
    comparison_line: ComparisonLine,
    fidelity_target: str | None,
    projection: str,
    algorithm: Mapping[str, Any],
    provenance: Mapping[str, Any] | None,
) -> MethodResult:
    values = np.asarray(patch_map, dtype=np.float32)
    if values.shape != PATCH_SHAPE:
        raise StageCAttentionError(
            f"C2 patch map must have shape {PATCH_SHAPE}, got {values.shape}"
        )
    method_provenance: dict[str, Any] = {
        "stage": "C2",
        "method_implementation_version": METHOD_VERSION,
        "attention_source": _ATTENTION_SOURCE,
        "class_agnostic": True,
        "algorithm": dict(algorithm),
    }
    context = _caller_context(provenance)
    if context:
        method_provenance["caller_context"] = context
    return MethodResult(
        method_id=method_id,
        method_version=METHOD_VERSION,
        comparison_line=comparison_line,
        explanation_target=None,
        fidelity_target=fidelity_target,
        class_agnostic=True,
        native_tensor=raw_array(values),
        metric_map=stage_c_metric_ready(values),
        projection=projection,
        sign_semantics=SignSemantics.NONNEGATIVE,
        valid_mask=np.ones(PATCH_SHAPE, dtype=np.bool_),
        provenance=method_provenance,
    )


def direct_raw_cls_attention_per_head(
    extraction: ExtractionResult,
    *,
    block_index: int,
    head_index: int,
    provenance: Mapping[str, Any] | None = None,
) -> MethodResult:
    """Return matched-block/head CLS-to-patch attention from ``attention_used``.

    This Value-path-line result is a different-target spatial-structure
    reference only.  It has neither an explanation target nor a fidelity
    target.
    """

    layers = _layers(extraction)
    block = _strict_index(block_index, name="block_index", upper_bound=MODEL_BLOCKS)
    head = _strict_index(head_index, name="head_index", upper_bound=HEADS_PER_BLOCK)
    attention_used = _validate_attention_tensor(
        layers[block].attention_used,
        name=f"blocks.{block}.attention_used",
        expected_heads=HEADS_PER_BLOCK,
        expected_tokens=TOKENS,
    )
    patch_map = _normalize_cls_patch_mass(
        attention_used[0, head, CLS_TOKEN_INDEX, :],
        name=f"blocks.{block}.heads.{head}.CLS",
    )
    return _result(
        patch_map,
        method_id="direct_raw_cls_attention_per_head",
        comparison_line=ComparisonLine.VALUE_PATH,
        fidelity_target=None,
        projection="raw_cls_attention_to_patch",
        algorithm={
            "block_index": block,
            "head_index": head,
            "query_token": CLS_TOKEN_INDEX,
            "head_fusion": "none",
            "exclude_cls_key": True,
            "patch_count": PATCH_TOKENS,
            "patch_mass_normalization": "divide_by_positive_patch_sum",
            "comparison_label": "spatial_structure_similarity_not_same_target_fidelity",
        },
        provenance=provenance,
    )


def direct_raw_cls_attention_last_block_head_mean(
    extraction: ExtractionResult,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> MethodResult:
    """Return block-11 arithmetic-head-mean CLS attention.

    Heads are averaged before the CLS key is removed and the remaining patch
    mass is normalized, exactly as frozen for the class-logit comparison line.
    The result is a class-agnostic ranking baseline, not a class explanation.
    """

    layers = _layers(extraction)
    attention_used = _validate_attention_tensor(
        layers[CLASS_DIRECT_BLOCK_INDEX].attention_used,
        name=f"blocks.{CLASS_DIRECT_BLOCK_INDEX}.attention_used",
        expected_heads=HEADS_PER_BLOCK,
        expected_tokens=TOKENS,
    )
    head_mean = attention_used[0].mean(dim=0)
    patch_map = _normalize_cls_patch_mass(
        head_mean[CLS_TOKEN_INDEX, :],
        name=f"blocks.{CLASS_DIRECT_BLOCK_INDEX}.head_mean.CLS",
    )
    return _result(
        patch_map,
        method_id="direct_raw_cls_attention_last_block_head_mean",
        comparison_line=ComparisonLine.CLASS_LOGIT,
        fidelity_target="clean_predicted_class_raw_logit",
        projection="raw_cls_attention_to_patch",
        algorithm={
            "block_index": CLASS_DIRECT_BLOCK_INDEX,
            "head_count": HEADS_PER_BLOCK,
            "query_token": CLS_TOKEN_INDEX,
            "head_fusion": "arithmetic_mean_before_cls_exclusion_and_renormalization",
            "exclude_cls_key": True,
            "patch_count": PATCH_TOKENS,
            "patch_mass_normalization": "divide_by_positive_patch_sum",
        },
        provenance=provenance,
    )


def rollout_attention_matrices(
    attention_used_by_layer: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Compute ``B_last @ ... @ B_0`` for a non-empty attention sequence.

    This pure numerical core permits small token/head counts so identity, head
    fusion, row normalization, multiplication order, and one-layer behavior can
    be tested analytically.  The production wrapper below separately enforces
    the frozen 12-head, 197-token, 12-block ViT geometry.
    """

    if isinstance(attention_used_by_layer, torch.Tensor):
        raise TypeError(
            "attention_used_by_layer must be an explicit sequence of per-layer tensors"
        )
    try:
        attentions = tuple(attention_used_by_layer)
    except TypeError as exc:
        raise TypeError("attention_used_by_layer must be a sequence") from exc
    if not attentions:
        raise StageCAttentionError("attention rollout requires at least one layer")

    rollout: torch.Tensor | None = None
    expected_heads: int | None = None
    expected_tokens: int | None = None
    expected_device: torch.device | None = None
    expected_dtype: torch.dtype | None = None
    for index, raw_attention in enumerate(attentions):
        attention = _validate_attention_tensor(
            raw_attention,
            name=f"attention_used_by_layer[{index}]",
            expected_heads=expected_heads,
            expected_tokens=expected_tokens,
        )
        _, heads, tokens, _ = attention.shape
        if expected_heads is None:
            expected_heads = heads
            expected_tokens = tokens
            expected_device = attention.device
            expected_dtype = attention.dtype
        elif attention.device != expected_device or attention.dtype != expected_dtype:
            raise StageCAttentionError(
                "all rollout layers must share device and floating dtype"
            )

        mean_attention = attention[0].mean(dim=0)
        identity = torch.eye(tokens, dtype=mean_attention.dtype, device=mean_attention.device)
        residual = identity + mean_attention
        row_mass = residual.sum(dim=-1, keepdim=True)
        if not bool(torch.isfinite(row_mass).all().item()) or bool((row_mass <= 0).any().item()):
            raise StageCAttentionError(
                f"rollout layer {index} has non-finite or non-positive residual row mass"
            )
        transition = residual / row_mass
        if not bool(torch.isfinite(transition).all().item()):
            raise StageCAttentionError(f"rollout layer {index} is non-finite after row normalization")
        expected_ones = torch.ones(tokens, dtype=transition.dtype, device=transition.device)
        if not torch.allclose(
            transition.sum(dim=-1), expected_ones, rtol=1e-6, atol=1e-6
        ):
            raise StageCAttentionError(f"rollout layer {index} rows do not sum to one")
        # Iterating from block 0 upward and left-multiplying yields exactly
        # B_last B_(last-1) ... B_0.
        rollout = transition if rollout is None else transition @ rollout

    if rollout is None:  # Defensive; the empty case is rejected above.
        raise RuntimeError("rollout construction produced no matrix")
    if not bool(torch.isfinite(rollout).all().item()) or bool((rollout < 0).any().item()):
        raise StageCAttentionError("rollout matrix is non-finite or negative")
    expected_ones = torch.ones(
        rollout.shape[0], dtype=rollout.dtype, device=rollout.device
    )
    if not torch.allclose(rollout.sum(dim=-1), expected_ones, rtol=1e-5, atol=1e-6):
        raise StageCAttentionError("rollout matrix rows do not sum to one")
    return rollout.detach().clone()


def attention_rollout_mean_head_residual_all_blocks(
    extraction: ExtractionResult,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> MethodResult:
    """Return frozen mean-head residual rollout over blocks 0 through 11."""

    layers = _layers(extraction)
    attentions: list[torch.Tensor] = []
    for block_index, layer in enumerate(layers):
        attentions.append(
            _validate_attention_tensor(
                layer.attention_used,
                name=f"blocks.{block_index}.attention_used",
                expected_heads=HEADS_PER_BLOCK,
                expected_tokens=TOKENS,
            )
        )
    rollout = rollout_attention_matrices(attentions)
    patch_map = _normalize_cls_patch_mass(
        rollout[CLS_TOKEN_INDEX, :], name="attention_rollout.CLS"
    )
    return _result(
        patch_map,
        method_id="attention_rollout_mean_head_residual_all_blocks",
        comparison_line=ComparisonLine.CLASS_LOGIT,
        fidelity_target="clean_predicted_class_raw_logit",
        projection="rollout_cls_to_patch",
        algorithm={
            "blocks": list(range(MODEL_BLOCKS)),
            "head_count": HEADS_PER_BLOCK,
            "head_fusion": "arithmetic_mean",
            "residual": "identity_coefficient_1",
            "row_normalization": "after_identity_addition",
            "multiplication_order": "B11_times_B10_times_..._times_B0",
            "query_token": CLS_TOKEN_INDEX,
            "exclude_cls_key": True,
            "patch_count": PATCH_TOKENS,
            "patch_mass_normalization": "divide_by_positive_patch_sum",
        },
        provenance=provenance,
    )


def class_attention_maps_at_intervention(
    extractor: Any,
    intervention_inputs: torch.Tensor,
    *,
    fidelity_target_identity: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> tuple[MethodResult, MethodResult]:
    """Recompute the two C2 class-line maps at one bound intervention."""

    if not hasattr(extractor, "extract") or not callable(extractor.extract):
        raise TypeError("extractor must expose a callable extract method")
    if not isinstance(intervention_inputs, torch.Tensor):
        raise TypeError("intervention_inputs must be a torch.Tensor")
    if intervention_inputs.dtype is not torch.float32 or tuple(
        intervention_inputs.shape
    ) != (1, 3, 224, 224):
        raise StageCAttentionError(
            "intervention_inputs must be float32 with shape [1,3,224,224]"
        )
    if not bool(torch.isfinite(intervention_inputs.detach()).all().item()):
        raise StageCAttentionError("intervention_inputs must be finite")
    if not isinstance(fidelity_target_identity, Mapping) or not fidelity_target_identity:
        raise TypeError("fidelity_target_identity must be a non-empty mapping")
    context = dict(provenance or {})
    context["fidelity_target_identity"] = dict(fidelity_target_identity)
    context["intervention_input_sha256"] = _tensor_sha256(intervention_inputs)
    caller_sha = _tensor_sha256(intervention_inputs)
    extraction = extractor.extract(intervention_inputs)
    if _tensor_sha256(intervention_inputs) != caller_sha:
        raise StageCAttentionError("extractor mutated intervention_inputs")
    direct = direct_raw_cls_attention_last_block_head_mean(
        extraction, provenance=context
    )
    rollout = attention_rollout_mean_head_residual_all_blocks(
        extraction, provenance=context
    )
    return direct, rollout


__all__ = [
    "CLASS_DIRECT_BLOCK_INDEX",
    "CLS_TOKEN_INDEX",
    "HEADS_PER_BLOCK",
    "METHOD_VERSION",
    "MODEL_BLOCKS",
    "PATCH_TOKENS",
    "StageCAttentionError",
    "TOKENS",
    "attention_rollout_mean_head_residual_all_blocks",
    "class_attention_maps_at_intervention",
    "direct_raw_cls_attention_last_block_head_mean",
    "direct_raw_cls_attention_per_head",
    "rollout_attention_matrices",
]
