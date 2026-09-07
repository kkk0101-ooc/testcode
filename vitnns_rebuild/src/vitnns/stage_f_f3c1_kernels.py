"""Frozen kernels for Stage F F3C1.

This module implements the prospective all-block fixed-clean-attention
conditional class-logit attribution and the common five-method evaluator.  It
contains no pilot/formal sample discovery and no rendering surface.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from . import attribution
from .stage_c_attention import (
    attention_rollout_mean_head_residual_all_blocks,
    direct_raw_cls_attention_last_block_head_mean,
)
from .stage_c_class_logit import (
    FrozenClassLogitTarget,
    StageCClassLogitIGConvergenceError,
    _verify_frozen_target,
    full_model_class_logit_ig,
    resolve_clean_class_logit_target,
    vit_grad_cam_block11_norm1,
)
from .stage_c_maps import pixel_to_patch_mass, raw_array, signed_rgb_channel_sum
from .stage_c_perturbations import tensor_sha256


METHOD_ID = "conditional_all_blocks_fixed_clean_attention_class_logit_ig"
METHODS = (
    METHOD_ID,
    "vit_grad_cam_block11_norm1",
    "full_model_class_logit_ig",
    "attention_rollout_mean_head_residual_all_blocks",
    "direct_raw_cls_attention_last_block_head_mean",
)
SIGNED_METHODS = frozenset({METHOD_ID, "full_model_class_logit_ig"})
CLASS_AGNOSTIC_METHODS = frozenset(
    {
        "attention_rollout_mean_head_residual_all_blocks",
        "direct_raw_cls_attention_last_block_head_mean",
    }
)
MODEL_BLOCKS = 12
HEADS = 12
TOKENS = 197
IG_SCHEDULE = (512, 1024, 2048, 4096, 8192)
IG_CHUNK = 8
IG_ATOL = 1e-4
IG_RTOL = 0.01
FRACTIONS = tuple(index / 20.0 for index in range(21))
TIE_REPEATS = 5
TIE_NAMESPACE = "STAGE_F_F3C_V1_TIE|"
STAT_NAMESPACE = "STAGE_F_F3C_V1_STATISTICS|"
BOOTSTRAP_RESAMPLES = 10_000
PERMUTATION_RESAMPLES = 100_000
PRIMARY_CONTRAST = "class_targeted_nns_minus_vit_grad_cam"
SECONDARY_CONTRASTS = (
    "class_targeted_nns_minus_full_model_ig",
    "class_targeted_nns_minus_attention_rollout",
    "class_targeted_nns_minus_raw_cls_attention",
)
_ISSUER = object()


class StageFF3C1KernelError(ValueError):
    pass


class StageFF3C1IGConvergenceError(StageFF3C1KernelError):
    def __init__(self, message: str, result: attribution.IGResult) -> None:
        super().__init__(message)
        self.ig_result = result
        self.attempts = result.attempts


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _model_signature(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    values: list[tuple[Any, ...]] = []
    for name, tensor in tuple(model.named_parameters()) + tuple(model.named_buffers()):
        values.append(
            (
                name, id(tensor), int(getattr(tensor, "_version", -1)),
                tuple(tensor.shape), str(tensor.dtype), str(tensor.device),
            )
        )
    return tuple(values)


def _attention_modules(model: nn.Module, expected_blocks: int) -> tuple[nn.Module, ...]:
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) != expected_blocks:
        raise StageFF3C1KernelError(
            f"all-block capture requires exactly {expected_blocks} blocks"
        )
    modules: list[nn.Module] = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        dropout = getattr(attention, "attn_drop", None)
        if not isinstance(dropout, nn.Module):
            raise StageFF3C1KernelError(
                f"blocks[{index}].attn.attn_drop is not observable"
            )
        if bool(getattr(attention, "fused_attn", False)):
            raise StageFF3C1KernelError(
                f"blocks[{index}] uses fused attention; post-softmax A is not observable"
            )
        if dropout.training:
            raise StageFF3C1KernelError(
                f"blocks[{index}].attn_drop must be in eval mode"
            )
        modules.append(dropout)
    return tuple(modules)


class FrozenAllBlockAttention:
    """Issuer-bound immutable set of all post-softmax attention tensors."""

    def __init__(
        self, *_args: Any, issuer: object | None = None, model: nn.Module | None = None,
        tensors: Sequence[torch.Tensor] = (), source_forward_id: str = "",
        expected_heads: int = HEADS, expected_tokens: int = TOKENS,
    ) -> None:
        if issuer is not _ISSUER or not isinstance(model, nn.Module):
            raise TypeError("FrozenAllBlockAttention can only be issued by clean capture")
        owned = tuple(value.detach().clone().contiguous() for value in tensors)
        if len(owned) != MODEL_BLOCKS:
            raise StageFF3C1KernelError("exactly 12 attention tensors are required")
        device = owned[0].device if owned else None
        for index, value in enumerate(owned):
            if value.shape != (1, expected_heads, expected_tokens, expected_tokens):
                raise StageFF3C1KernelError(
                    f"attention {index} shape drift: {tuple(value.shape)}"
                )
            if value.dtype is not torch.float32 or value.device != device:
                raise StageFF3C1KernelError(
                    "all frozen attentions must share float32 dtype and device"
                )
            if value.requires_grad or not bool(torch.isfinite(value).all().item()):
                raise StageFF3C1KernelError(
                    "frozen attention must be detached and finite"
                )
            if bool((value < 0).any().item()):
                raise StageFF3C1KernelError("post-softmax attention cannot be negative")
            row_sum = value.sum(dim=-1)
            if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=2e-6, rtol=2e-6):
                raise StageFF3C1KernelError("post-softmax attention rows must sum to one")
        self.__tensors = owned
        self.__hashes = tuple(tensor_sha256(value) for value in owned)
        self.__versions = tuple(int(value._version) for value in owned)
        self.__model_ref = weakref.ref(model)
        self.__model_signature = _model_signature(model)
        self.source_forward_id = source_forward_id
        self.expected_heads = expected_heads
        self.expected_tokens = expected_tokens
        self.device = str(device)

    @property
    def hashes(self) -> tuple[str, ...]:
        self.verify()
        return self.__hashes

    def manifest(self) -> dict[str, Any]:
        self.verify()
        return {
            "count": len(self.__tensors),
            "blocks": list(range(MODEL_BLOCKS)),
            "head_count": self.expected_heads,
            "token_count": self.expected_tokens,
            "dtype": "float32",
            "device": self.device,
            "source": "same_clean_forward_attn_drop_output_actually_consumed",
            "source_forward_id": self.source_forward_id,
            "detached_cloned_frozen": True,
            "manual_or_learned_head_weights": False,
            "tensor_sha256": list(self.__hashes),
            "canonical_digest": canonical_sha256(
                {"source_forward_id": self.source_forward_id, "hashes": self.__hashes}
            ),
        }

    def owned_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return defensive copies for tested read-only Stage C kernels."""

        self.verify()
        return tuple(value.detach().clone() for value in self.__tensors)

    def verify(self, model: nn.Module | None = None) -> None:
        bound = self.__model_ref()
        if bound is None or (model is not None and model is not bound):
            raise StageFF3C1KernelError("frozen attention model binding changed")
        if _model_signature(bound) != self.__model_signature:
            raise StageFF3C1KernelError("model state changed after attention capture")
        current = tuple(tensor_sha256(value) for value in self.__tensors)
        versions = tuple(int(value._version) for value in self.__tensors)
        if current != self.__hashes or versions != self.__versions:
            raise StageFF3C1KernelError("frozen attention tensor was mutated")

    @contextmanager
    def inject(self, model: nn.Module) -> Iterator[tuple[Any, ...]]:
        self.verify(model)
        with ExitStack() as stack:
            interventions = tuple(
                stack.enter_context(
                    attribution.fixed_attention_intervention(model, index, value)
                )
                for index, value in enumerate(self.__tensors)
            )
            yield interventions
        self.verify(model)


@dataclass(frozen=True)
class CleanCapture:
    target: FrozenClassLogitTarget
    attentions: FrozenAllBlockAttention
    logits: torch.Tensor
    clean_input_sha256: str

    def as_attention_extraction(self) -> Any:
        """Adapt the same-forward tensors to the existing C2 kernel input."""

        from .extractor import ExtractionResult, LayerCapture

        layers = []
        for value in self.attentions.owned_tensors():
            empty = torch.empty(0, dtype=value.dtype, device=value.device)
            layers.append(
                LayerCapture(
                    block_input=empty, block_output=empty, q=empty, k=empty, v=empty,
                    attention=value.detach().clone(), attention_used=value.detach().clone(),
                    attention_recomputed=value.detach().clone(), scale=1.0,
                )
            )
        return ExtractionResult(logits=self.logits.detach().clone(), layers=tuple(layers))


def capture_clean_target_and_all_attention(
    model: nn.Module, clean_inputs: torch.Tensor, *,
    official_truth_index: int | None = None, expected_heads: int = HEADS,
    expected_tokens: int = TOKENS,
) -> CleanCapture:
    if not isinstance(clean_inputs, torch.Tensor) or clean_inputs.shape != (1, 3, 224, 224):
        raise StageFF3C1KernelError("clean input must be [1,3,224,224]")
    if clean_inputs.dtype is not torch.float32:
        raise StageFF3C1KernelError("clean input must be float32")
    modules = _attention_modules(model, MODEL_BLOCKS)
    captured: list[torch.Tensor | None] = [None] * MODEL_BLOCKS
    calls = [0] * MODEL_BLOCKS
    logits_box: list[torch.Tensor] = []
    handles = []

    def attention_hook(index: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            calls[index] += 1
            if not isinstance(output, torch.Tensor):
                raise StageFF3C1KernelError("attn_drop output must be a tensor")
            captured[index] = output.detach().clone().contiguous()
        return hook

    def model_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise StageFF3C1KernelError("model output must be raw logits")
        logits_box.append(output.detach().clone())

    try:
        for index, module in enumerate(modules):
            handles.append(module.register_forward_hook(attention_hook(index)))
        handles.append(model.register_forward_hook(model_hook))
        target = resolve_clean_class_logit_target(
            model, clean_inputs, official_truth_index=official_truth_index
        )
    finally:
        for handle in reversed(handles):
            handle.remove()
    if calls != [1] * MODEL_BLOCKS or len(logits_box) != 1:
        raise StageFF3C1KernelError(
            "all 12 attentions and logits must come from exactly one clean forward"
        )
    if any(value is None for value in captured):
        raise StageFF3C1KernelError("clean forward did not capture every attention")
    logits = logits_box[0]
    forward_id = canonical_sha256(
        {
            "clean_input_sha256": target.clean_input_sha256,
            "logits_sha256": tensor_sha256(logits),
            "class_index": target.class_index,
        }
    )
    attentions = FrozenAllBlockAttention(
        issuer=_ISSUER, model=model, tensors=tuple(captured),
        source_forward_id=forward_id, expected_heads=expected_heads,
        expected_tokens=expected_tokens,
    )
    return CleanCapture(
        target=target, attentions=attentions, logits=logits,
        clean_input_sha256=target.clean_input_sha256,
    )


def conditional_logits(
    model: nn.Module, values: torch.Tensor, frozen: FrozenAllBlockAttention
) -> torch.Tensor:
    with frozen.inject(model):
        output = model(values)
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise StageFF3C1KernelError("conditional model must return [batch,classes] logits")
    if output.dtype is not torch.float32 or not bool(torch.isfinite(output).all().item()):
        raise StageFF3C1KernelError("conditional logits must be finite float32")
    return output


def validate_endpoint_parity(
    model: nn.Module, clean_inputs: torch.Tensor, capture: CleanCapture
) -> dict[str, Any]:
    replayed = conditional_logits(model, clean_inputs.detach(), capture.attentions).detach()
    if replayed.shape != capture.logits.shape:
        raise StageFF3C1KernelError("conditional endpoint logit shape drifted")
    error = (replayed - capture.logits).abs()
    maximum = float(error.max().item()) if error.numel() else 0.0
    passed = maximum < 1e-6
    result = {
        "status": "PASS" if passed else "FAIL",
        "criterion": "all_logits_max_absolute_error_strictly_less_than_1e-6",
        "max_absolute_error": maximum,
        "class_count": int(replayed.shape[1]),
        "attention_count": MODEL_BLOCKS,
    }
    if not passed:
        raise StageFF3C1KernelError(
            f"conditional endpoint parity failed: max_abs_error={maximum}"
        )
    return result


def conditional_class_logit_ig(
    model: nn.Module, clean_inputs: torch.Tensor, capture: CleanCapture, *,
    steps: Sequence[int] = IG_SCHEDULE, step_chunk_size: int = IG_CHUNK,
) -> dict[str, Any]:
    if tuple(steps) != IG_SCHEDULE:
        raise StageFF3C1KernelError("production conditional IG schedule drifted")
    if step_chunk_size != IG_CHUNK:
        raise StageFF3C1KernelError("production conditional IG chunk size drifted")
    return _conditional_class_logit_ig(
        model, clean_inputs, capture, steps=tuple(steps),
        step_chunk_size=step_chunk_size,
    )


def _conditional_class_logit_ig(
    model: nn.Module, clean_inputs: torch.Tensor, capture: CleanCapture, *,
    steps: Sequence[int], step_chunk_size: int,
) -> dict[str, Any]:
    _verify_frozen_target(capture.target, clean_inputs, model)
    parity = validate_endpoint_parity(model, clean_inputs, capture)
    baseline = torch.zeros_like(clean_inputs)
    with capture.attentions.inject(model) as interventions, torch.enable_grad():
        def callback(values: torch.Tensor) -> torch.Tensor:
            logits = model(values)
            return logits[:, capture.target.class_index]

        result = attribution.integrated_gradients(
            callback, clean_inputs, baseline, steps=tuple(steps),
            step_chunk_size=step_chunk_size, atol=IG_ATOL, rtol=IG_RTOL,
            parameters=model,
            metadata={
                "scope": "conditional_all_blocks_fixed_clean_attention_class_logit",
                "target": capture.target.target_spec.to_dict(),
                "attention_count": MODEL_BLOCKS,
                "attention_hashes": list(capture.attentions.hashes),
                "cut_paths": ["query", "key", "attention_softmax_variation"],
                "live_paths": [
                    "values", "head_concatenation", "attention_output_projection",
                    "residual", "mlp", "final_norm", "classifier_head",
                ],
            },
        )
        intervention_calls = [item.call_count for item in interventions]
    if len(set(intervention_calls)) != 1 or intervention_calls[0] <= 0:
        raise StageFF3C1KernelError("all attention interventions must cover every IG forward")
    if result.attribution.shape != (1, 3, 224, 224) or result.attribution.dtype is not torch.float32:
        raise StageFF3C1KernelError("conditional IG native attribution contract failed")
    if not bool(torch.isfinite(result.attribution).all().item()):
        raise StageFF3C1KernelError("conditional IG attribution is non-finite")
    if not result.converged:
        raise StageFF3C1IGConvergenceError(
            "conditional all-block IG exhausted its frozen schedule", result
        )
    native = result.attribution[0].detach().cpu().numpy().astype(np.float32, copy=True)
    metric = np.asarray(
        pixel_to_patch_mass(signed_rgb_channel_sum(raw_array(native))).data,
        dtype=np.float32,
    ).copy()
    if not np.isfinite(metric).all() or not np.count_nonzero(metric):
        raise StageFF3C1KernelError("conditional IG projected map is invalid or all zero")
    return {
        "method_id": METHOD_ID,
        "class_targeted": True,
        "class_agnostic": False,
        "conditional_functional": True,
        "native_signed_rgb": native,
        "patch_mass": metric,
        "sign_semantics": "signed",
        "endpoint_parity": parity,
        "ig_result": result,
        "intervention_calls_per_block": intervention_calls,
        "attention_manifest": capture.attentions.manifest(),
        "baseline": {
            "id": "normalized_zero", "normalized_value": 0.0,
            "canonical_rgb_value": 0.5, "is_black_image": False,
        },
    }


def synthetic_conditional_class_logit_ig(
    model: nn.Module, clean_inputs: torch.Tensor, capture: CleanCapture, *,
    steps: Sequence[int] = (8, 16, 32), step_chunk_size: int = 4,
) -> dict[str, Any]:
    """Synthetic-only small schedule; production callers cannot select it."""

    return _conditional_class_logit_ig(
        model, clean_inputs, capture, steps=tuple(steps),
        step_chunk_size=step_chunk_size,
    )


def _plain_method_result(result: Any) -> dict[str, Any]:
    native = np.asarray(result.native_tensor.data, dtype=np.float32).copy()
    metric = np.asarray(result.metric_map.data, dtype=np.float32).copy()
    return {
        "method_id": result.method_id,
        "class_targeted": not result.class_agnostic,
        "class_agnostic": bool(result.class_agnostic),
        "conditional_functional": False,
        "native": native, "patch_mass": metric,
        "sign_semantics": result.sign_semantics.value,
        "provenance": json.loads(
            json.dumps(
                _plain_json(result.provenance), sort_keys=True, allow_nan=False
            )
        ),
    }


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _class_target_identity(capture: CleanCapture) -> dict[str, Any]:
    return {
        "comparison_line": "class_logit",
        "target_ref": "clean_predicted_class_raw_logit",
        "clean_input_sha256": capture.target.clean_input_sha256,
        "frozen_target": capture.target.to_dict(),
    }


def collect_five_method_maps(
    model: nn.Module, clean_inputs: torch.Tensor, capture: CleanCapture, extraction: Any
) -> dict[str, dict[str, Any]]:
    from .stage_c_fidelity import ClassLogitScalarAdapter, fidelity_target_provenance_binding, resolve_target_response

    baseline = torch.zeros_like(clean_inputs)
    evaluator = ClassLogitScalarAdapter(model, clean_inputs, capture.target)
    response = resolve_target_response(evaluator, clean_inputs, baseline)
    if response.na is not None:
        raise StageFF3C1KernelError("unmodified class-logit endpoint is degenerate")
    context = fidelity_target_provenance_binding(response)
    nns = conditional_class_logit_ig(model, clean_inputs, capture)
    gradcam = _plain_method_result(
        vit_grad_cam_block11_norm1(model, clean_inputs, capture.target, provenance=context)
    )
    full_ig = _plain_method_result(
        full_model_class_logit_ig(model, clean_inputs, capture.target, provenance=context)
    )
    rollout = _plain_method_result(
        attention_rollout_mean_head_residual_all_blocks(extraction, provenance=context)
    )
    direct = _plain_method_result(
        direct_raw_cls_attention_last_block_head_mean(extraction, provenance=context)
    )
    observed = {item["method_id"]: item for item in (nns, gradcam, full_ig, rollout, direct)}
    if tuple(observed) != METHODS:
        raise StageFF3C1KernelError("five-method roster/order drifted")
    for method_id, item in observed.items():
        if item["class_agnostic"] is not (method_id in CLASS_AGNOSTIC_METHODS):
            raise StageFF3C1KernelError("method class-target label drifted")
    return observed


def tie_seed(sample_id: str, repeat: int) -> int:
    if not isinstance(sample_id, str) or not sample_id or type(repeat) is not int or not 0 <= repeat < 5:
        raise TypeError("tie seed requires sample_id and repeat 0..4")
    digest = hashlib.sha256(f"{TIE_NAMESPACE}{sample_id}|{repeat}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def tie_priority(seed: int, count: int = 196) -> np.ndarray:
    order = np.random.Generator(np.random.PCG64(seed)).permutation(count)
    priority = np.empty(count, dtype=np.int64)
    priority[order] = np.arange(count, dtype=np.int64)
    return priority


def method_ranking(patch_mass: np.ndarray, *, sigma: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(patch_mass)
    if value.dtype != np.float32 or value.shape != (14, 14) or not np.isfinite(value).all():
        raise StageFF3C1KernelError("ranking requires finite float32 14x14 patch mass")
    if sigma not in {-1, 1}:
        raise StageFF3C1KernelError("ranking sigma must be -1 or 1")
    scores = np.maximum(np.float32(sigma) * value, np.float32(0.0)).reshape(-1)
    if float(scores.sum(dtype=np.float64)) <= 0.0:
        raise StageFF3C1KernelError("method has zero target-oriented positive mass")
    priority = tie_priority(seed, scores.size)
    ranking = np.lexsort((priority, -scores.astype(np.float64))).astype(np.int64)
    return ranking, priority


def _actual_fractions(count: int = 196) -> tuple[float, ...]:
    counts = (0,) + tuple(int(math.ceil(value * count)) for value in FRACTIONS[1:])
    return tuple(value / count for value in counts)


def _intervention_batch(
    clean: torch.Tensor, baseline: torch.Tensor, ranking: np.ndarray, *, direction: str
) -> torch.Tensor:
    if direction not in {"insertion", "deletion"}:
        raise StageFF3C1KernelError("unknown fidelity direction")
    rows = []
    counts = (0,) + tuple(int(math.ceil(value * 196)) for value in FRACTIONS[1:])
    for count in counts:
        selected = np.zeros(196, dtype=np.bool_)
        selected[ranking[:count]] = True
        pixels = np.repeat(np.repeat(selected.reshape(14, 14), 16, axis=0), 16, axis=1)
        mask = torch.from_numpy(pixels.copy()).to(clean.device)[None, None].expand_as(clean)
        rows.append(torch.where(mask, clean, baseline) if direction == "insertion" else torch.where(mask, baseline, clean))
    return torch.cat(rows, dim=0)


def evaluate_common_fidelity(
    model: nn.Module, clean: torch.Tensor, capture: CleanCapture,
    methods: Mapping[str, Mapping[str, Any]], *, sample_id: str, eval_chunk: int = 8,
) -> dict[str, Any]:
    if tuple(methods) != METHODS:
        raise StageFF3C1KernelError("common evaluator requires the exact five-method roster")
    baseline = torch.zeros_like(clean)
    with torch.no_grad():
        clean_value = float(model(clean)[0, capture.target.class_index].item())
        baseline_value = float(model(baseline)[0, capture.target.class_index].item())
    delta = clean_value - baseline_value
    if not math.isfinite(delta) or abs(delta) <= 1e-6 * max(1.0, abs(clean_value), abs(baseline_value)):
        raise StageFF3C1KernelError("unmodified class-logit fidelity target is degenerate")
    unmodified_sigma = 1 if delta > 0 else -1
    nns_result = methods[METHOD_ID]["ig_result"]
    nns_delta = float(nns_result.conservation["delta"][0].item())
    nns_sigma = 1 if nns_delta > 0 else -1
    records: dict[str, Any] = {}
    actual = _actual_fractions()
    for method_id in METHODS:
        method = methods[method_id]
        sigma = (
            nns_sigma
            if method_id == METHOD_ID
            else unmodified_sigma
            if method_id in SIGNED_METHODS
            else 1
        )
        curves = []
        for repeat in range(TIE_REPEATS):
            seed = tie_seed(sample_id, repeat)
            ranking, priority = method_ranking(method["patch_mass"], sigma=sigma, seed=seed)
            directions: dict[str, Any] = {}
            for direction in ("insertion", "deletion"):
                batch = _intervention_batch(clean, baseline, ranking, direction=direction)
                values: list[float] = []
                with torch.no_grad():
                    for start in range(0, batch.shape[0], eval_chunk):
                        logits = model(batch[start:start + eval_chunk])
                        values.extend(float(x) for x in logits[:, capture.target.class_index].cpu().tolist())
                normalized = [unmodified_sigma * (value - baseline_value) / abs(delta) for value in values]
                expected_first = 0.0 if direction == "insertion" else 1.0
                expected_last = 1.0 if direction == "insertion" else 0.0
                if normalized[0] != expected_first or normalized[-1] != expected_last:
                    raise StageFF3C1KernelError("fidelity endpoints drifted")
                auc = float(np.trapz(np.asarray(normalized, np.float64), np.asarray(actual, np.float64)))
                directions[direction] = {
                    "responses": normalized, "auc": auc,
                    "actual_fractions": list(actual),
                }
            curves.append(
                {
                    "repeat": repeat, "seed": seed,
                    "tie_priority_sha256": array_digest(priority),
                    "ranking_sha256": array_digest(ranking),
                    "insertion": directions["insertion"],
                    "deletion": directions["deletion"],
                    "fidelity_score": 0.5 * (
                        directions["insertion"]["auc"] - directions["deletion"]["auc"]
                    ),
                }
            )
        records[method_id] = {
            "curves": curves,
            "fidelity_score_mean_over_five_ties": float(
                np.mean([item["fidelity_score"] for item in curves])
            ),
            "sigma": sigma,
        }
    priorities = [records[METHODS[0]]["curves"][i]["tie_priority_sha256"] for i in range(5)]
    for method_id in METHODS[1:]:
        if [records[method_id]["curves"][i]["tie_priority_sha256"] for i in range(5)] != priorities:
            raise StageFF3C1KernelError("tie priorities are not shared across methods")
    return {
        "target": _class_target_identity(capture),
        "target_class_index": capture.target.class_index,
        "baseline": "normalized_zero_canonical_rgb_0_5",
        "fractions": list(FRACTIONS), "tie_repeats": TIE_REPEATS,
        "tie_namespace": TIE_NAMESPACE,
        "auc": "trapezoid_over_actual_fraction",
        "score": "0.5*(insertion_auc-deletion_auc)",
        "methods": records,
    }


def _stat_seed(kind: str, contrast: str) -> int:
    digest = hashlib.sha256(f"{STAT_NAMESPACE}{kind}|{contrast}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def paired_sign_flip_p_value(
    differences: Sequence[float], contrast: str, *, resamples: int = PERMUTATION_RESAMPLES
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (50,) or not np.isfinite(values).all():
        raise StageFF3C1KernelError("paired inference requires exactly 50 finite image differences")
    if resamples != PERMUTATION_RESAMPLES:
        raise StageFF3C1KernelError("paired permutation resample count drifted")
    seed = _stat_seed("sign_flip", contrast)
    rng = np.random.Generator(np.random.PCG64(seed))
    observed = float(np.mean(values))
    exceed = 0
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        signs = rng.integers(0, 2, size=(count, 50), dtype=np.int8) * 2 - 1
        simulated = np.mean(signs * values[None, :], axis=1)
        exceed += int(np.count_nonzero(np.abs(simulated) >= abs(observed)))
    return {
        "contrast": contrast, "n": 50, "statistic": "paired_mean_difference",
        "observed": observed, "two_sided": True,
        "zero_differences": "retained_and_sign_flipped_with_zero_effect",
        "nonfinite_or_invalid_units": "fail_closed_no_filter_or_imputation",
        "resamples": resamples, "seed": seed,
        "rng": "numpy.random.Generator(PCG64)",
        "extreme_rule": "abs(simulated_mean)>=abs(observed_mean)",
        "add_one_correction": True, "p_value": (exceed + 1.0) / (resamples + 1.0),
    }


def bootstrap_mean_ci(
    differences: Sequence[float], contrast: str, *, resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (50,) or not np.isfinite(values).all():
        raise StageFF3C1KernelError("bootstrap requires exactly 50 finite image differences")
    if resamples != BOOTSTRAP_RESAMPLES:
        raise StageFF3C1KernelError("bootstrap resample count drifted")
    seed = _stat_seed("bootstrap_mean", contrast)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, 50, size=(resamples, 50), dtype=np.int64)
    distribution = np.mean(values[draws], axis=1)
    interval = np.quantile(distribution, [0.025, 0.975], method="linear")
    return {
        "contrast": contrast, "statistic": "paired_mean_difference",
        "resamples": resamples, "seed": seed, "unit": "image",
        "percentile": [0.025, 0.975], "quantile_method": "linear",
        "ci": [float(interval[0]), float(interval[1])],
        "draw_indices_sha256": array_digest(draws),
    }


def holm_adjust_secondary(raw_p_values: Mapping[str, float]) -> list[dict[str, Any]]:
    if tuple(raw_p_values) != SECONDARY_CONTRASTS:
        raise StageFF3C1KernelError("Holm family/order must be the frozen three contrasts")
    values = []
    for index, contrast in enumerate(SECONDARY_CONTRASTS):
        value = raw_p_values[contrast]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise StageFF3C1KernelError("Holm raw p-values must be finite in [0,1]")
        values.append((float(value), index, contrast))
    ordered = sorted(values, key=lambda item: (item[0], item[1]))
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, (value, _index, contrast) in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * value)
        adjusted[contrast] = min(1.0, running)
    return [
        {
            "contrast": contrast, "family_order": index,
            "raw_p_value": float(raw_p_values[contrast]),
            "adjusted_p_value": adjusted[contrast],
            "reject_alpha_0_05": adjusted[contrast] <= 0.05,
        }
        for index, contrast in enumerate(SECONDARY_CONTRASTS)
    ]


def summarize_contrast(differences: Sequence[float], contrast: str) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    allowed = (PRIMARY_CONTRAST,) + SECONDARY_CONTRASTS
    if contrast not in allowed:
        raise StageFF3C1KernelError("contrast is not in the frozen family")
    if values.shape != (50,) or not np.isfinite(values).all():
        raise StageFF3C1KernelError(
            "contrast summary requires 50 finite paired image values"
        )
    return {
        "contrast": contrast, "n": 50, "unit": "image",
        "per_image": values.tolist(), "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sample_sd_ddof_1": float(np.std(values, ddof=1)),
        "positive_count": int(np.count_nonzero(values > 0.0)),
        "bootstrap_mean_ci": bootstrap_mean_ci(values, contrast),
        "paired_two_sided_p_value": paired_sign_flip_p_value(values, contrast),
        "multiplicity_family": (
            None if contrast == PRIMARY_CONTRAST else "fixed_three_secondary_holm_family"
        ),
    }


def statistical_method_freeze() -> dict[str, Any]:
    payload = {
        "schema_version": 1, "document_type": "stage_f_f3c1_statistical_method_freeze",
        "status": "FROZEN_BEFORE_PILOT_OUTCOME_ACCESS",
        "formal_unit": "image", "formal_n": 50,
        "invalid_policy": "any_nonfinite_failed_or_na_required_unit_blocks_contrast_no_filter_no_imputation",
        "summary": ["per_image", "mean", "median", "sample_sd_ddof_1", "positive_count"],
        "bootstrap": {
            "statistic": "paired_mean_difference", "resamples": BOOTSTRAP_RESAMPLES,
            "interval": "percentile_95", "quantiles": [0.025, 0.975],
            "quantile_method": "linear", "seed_rule": "sha256_namespace_first8_big_endian",
        },
        "p_value": {
            "method": "monte_carlo_paired_sign_flip", "statistic": "paired_mean_difference",
            "two_sided": True, "resamples": PERMUTATION_RESAMPLES,
            "extreme_rule": "abs(simulated_mean)>=abs(observed_mean)",
            "add_one_correction": True, "zero_differences": "retained",
            "seed_rule": "sha256_namespace_first8_big_endian",
        },
        "primary_contrast": {"id": PRIMARY_CONTRAST, "multiplicity_adjusted": False},
        "secondary_holm_family": {
            "members_in_frozen_order": list(SECONDARY_CONTRASTS),
            "method": "Holm_step_down", "alpha": 0.05,
            "tie_break": "frozen_family_order",
        },
        "pilot_or_formal_outcomes_accessed_before_freeze": False,
    }
    payload["canonical_digest"] = canonical_sha256(payload)
    return payload


__all__ = [
    "BOOTSTRAP_RESAMPLES", "CleanCapture", "FRACTIONS",
    "FrozenAllBlockAttention", "IG_SCHEDULE", "METHODS", "METHOD_ID",
    "PERMUTATION_RESAMPLES", "PRIMARY_CONTRAST", "SECONDARY_CONTRASTS",
    "StageFF3C1IGConvergenceError", "StageFF3C1KernelError",
    "array_digest", "bootstrap_mean_ci",
    "capture_clean_target_and_all_attention", "collect_five_method_maps",
    "conditional_class_logit_ig", "conditional_logits",
    "evaluate_common_fidelity", "holm_adjust_secondary", "method_ranking",
    "paired_sign_flip_p_value", "statistical_method_freeze",
    "summarize_contrast", "synthetic_conditional_class_logit_ig",
    "tie_priority", "tie_seed",
    "validate_endpoint_parity",
]
