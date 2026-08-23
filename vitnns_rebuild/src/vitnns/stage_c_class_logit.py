"""Stage C adapters for frozen-class ViT Grad-CAM and full-model IG.

Both adapters consume the same :class:`FrozenClassLogitTarget`, resolved once
from clean raw logits.  Grad-CAM observes the real ``blocks[11].norm1``
activation, while the IG adapter delegates to :func:`attribution.full_model_ig`
without intervening in the model.  Neither method is an attention attribution.
"""

from __future__ import annotations

import hashlib
import math
import re
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from secrets import token_bytes
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from . import attribution
from .attribution import IGResult
from .stage_c_integrity import CheapTensorSnapshot, model_live_signature
from .stage_c_maps import (
    ComparisonLine,
    MethodResult,
    PATCH_SHAPE,
    SignSemantics,
    pixel_to_patch_mass,
    raw_array,
    signed_rgb_channel_sum,
    stage_c_metric_ready,
)
from .targets import TargetSpec


METHOD_VERSION = "1.0.0"
MODEL_BLOCKS = 12
TOKENS = 197
PATCH_TOKENS = 196
EMBED_DIM = 768
TARGET_LAYER = "blocks[11].norm1"
TARGET_REF = "clean_predicted_class_raw_logit"
IG_STEPS = (512, 1024, 2048, 4096, 8192)
IG_STEP_CHUNK_SIZE = 8
IG_ATOL = 1e-4
IG_RTOL = 0.01
_TARGET_ISSUER_SECRET = token_bytes(32)
_ISSUED_TARGETS: "weakref.WeakKeyDictionary[FrozenClassLogitTarget, tuple[Any, ...]]" = (
    weakref.WeakKeyDictionary()
)


class StageCClassLogitError(ValueError):
    """Raised when a C3 class-logit attribution violates its contract."""


class StageCClassLogitIGConvergenceError(StageCClassLogitError):
    """Raised after full-model IG exhausts the frozen schedule.

    The validated ``ig_result`` is retained verbatim so callers can persist
    the complete failed-attempt evidence even though no :class:`MethodResult`
    is emitted.
    """

    def __init__(self, message: str, ig_result: IGResult) -> None:
        super().__init__(message)
        self.ig_result = ig_result
        self.result = ig_result
        self.attempts = ig_result.attempts


@dataclass(frozen=True, init=False)
class FrozenScalarEvaluation:
    """One frozen-target scalar evaluated at an arbitrary normalized input."""

    value: float
    target_kind: str
    class_index: int
    clean_input_sha256: str
    intervention_input_sha256: str
    runtime_model_binding_sha256: str

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "FrozenScalarEvaluation can only be issued by "
            "evaluate_frozen_class_logit_target"
        )

    @classmethod
    def _create(
        cls,
        *,
        value: float,
        class_index: int,
        clean_input_sha256: str,
        intervention_input_sha256: str,
        runtime_model_binding_sha256: str,
        issuer_secret: bytes,
    ) -> "FrozenScalarEvaluation":
        if issuer_secret is not _TARGET_ISSUER_SECRET:
            raise TypeError("scalar evaluations can only be issued by the evaluator")
        instance = object.__new__(cls)
        for name, field_value in {
            "value": value,
            "target_kind": TARGET_REF,
            "class_index": class_index,
            "clean_input_sha256": clean_input_sha256,
            "intervention_input_sha256": intervention_input_sha256,
            "runtime_model_binding_sha256": runtime_model_binding_sha256,
        }.items():
            object.__setattr__(instance, name, field_value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("value must be numeric")
        value = float(self.value)
        if not math.isfinite(value):
            raise StageCClassLogitError("scalar evaluation value must be finite")
        object.__setattr__(self, "value", value)
        if self.target_kind != TARGET_REF:
            raise StageCClassLogitError("scalar target_kind is not the frozen class target")
        if type(self.class_index) is not int or self.class_index < 0:
            raise TypeError("class_index must be a non-negative integer (not bool)")
        for name in (
            "clean_input_sha256",
            "intervention_input_sha256",
            "runtime_model_binding_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "target_kind": self.target_kind,
            "class_index": self.class_index,
            "clean_input_sha256": self.clean_input_sha256,
            "intervention_input_sha256": self.intervention_input_sha256,
            "runtime_model_binding_sha256": self.runtime_model_binding_sha256,
            "class_index_frozen": True,
        }


def _strict_optional_class_index(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (not bool) or None")
    if value < 0:
        raise StageCClassLogitError(f"{name} must be non-negative")
    return value


def _validate_clean_input(inputs: Any) -> torch.Tensor:
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("clean inputs must be a torch.Tensor")
    if inputs.shape != (1, 3, 224, 224):
        raise StageCClassLogitError(
            "clean inputs must have shape [1,3,224,224], "
            f"got {tuple(inputs.shape)}"
        )
    if inputs.dtype is not torch.float32:
        raise TypeError(
            f"clean inputs must have frozen float32 dtype, got {inputs.dtype}"
        )
    if not bool(torch.isfinite(inputs.detach()).all().item()):
        raise StageCClassLogitError("clean inputs must contain only finite values")
    return inputs


def _tensor_sha256(value: torch.Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(item) for item in detached.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(detached.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _prebound_sha256(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256 or None")
    return value


def _model_state_signature(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    """Bind a target to cheap live model state, never tensor contents.

    The historical private name is retained because frozen target fields and
    downstream imports use it.  Full-content verification now lives only in
    :func:`stage_c_integrity.strong_model_signature` at model-load and
    execution-unit boundaries.
    """

    # ``requires_grad`` is caller lifecycle state, not part of the target's
    # persistent model binding.  Operation guards snapshot it separately and
    # verify it after the attribution-state context has restored the caller.
    return model_live_signature(
        model, include_lifecycle=False, include_requires_grad=False
    )


def _model_binding_sha256(signature: tuple[tuple[Any, ...], ...]) -> str:
    # This is deliberately labelled a runtime binding, not a reproducible
    # model artifact hash.  C6 provenance will bind the canonical state-dict
    # SHA-256; C3 needs only to reject a different or mutated live model.
    return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()


def _target_identity_snapshot(
    target: "FrozenClassLogitTarget",
) -> tuple[Any, ...]:
    """Capture every field whose mutation could change target semantics."""

    return (
        target._issuer_nonce,
        target.class_index,
        target.clean_raw_logit,
        target.class_count,
        target.clean_input_sha256,
        target.clean_input_shape,
        target.official_truth_index,
        target.clean_top1_correct,
        target.runtime_model_binding_sha256,
        id(target._model_ref),
        target._model_state_signature,
        id(target._clean_input_ref),
        id(target._clean_input_state),
        target._clean_input_state.signature,
    )


@contextmanager
def _cheap_runtime_integrity_guard(
    model: nn.Module,
    *,
    tensors: tuple[tuple[str, torch.Tensor], ...],
    target: "FrozenClassLogitTarget",
    context: str,
) -> Iterator[None]:
    """Guard one hot-path operation without reading tensor contents."""

    model_state = model_live_signature(model, include_lifecycle=True)
    tensor_states = tuple(
        (name, tensor, CheapTensorSnapshot.capture(tensor))
        for name, tensor in tensors
    )
    target_identity = _target_identity_snapshot(target)
    try:
        yield
    finally:
        violations: list[str] = []
        if model_live_signature(model, include_lifecycle=True) != model_state:
            violations.append("model parameter/buffer state")
        for name, tensor, snapshot in tensor_states:
            if not snapshot.verify(tensor).matched:
                violations.append(name)
        if (
            _ISSUED_TARGETS.get(target) != target_identity
            or _target_identity_snapshot(target) != target_identity
        ):
            violations.append("frozen class target")
        if violations:
            raise StageCClassLogitError(
                f"{context} mutated " + ", ".join(violations)
            )


@dataclass(frozen=True, init=False, eq=False)
class FrozenClassLogitTarget:
    """One clean-input argmax raw-logit target shared by C3 methods."""

    class_index: int
    clean_raw_logit: float
    class_count: int
    clean_input_sha256: str
    clean_input_shape: tuple[int, int, int, int]
    official_truth_index: int | None = None
    clean_top1_correct: bool | None = None
    runtime_model_binding_sha256: str = ""
    _model_ref: weakref.ReferenceType[nn.Module] = field(
        repr=False, compare=False
    )
    _model_state_signature: tuple[tuple[Any, ...], ...] = field(
        repr=False, compare=False
    )
    _clean_input_ref: weakref.ReferenceType[torch.Tensor] = field(
        repr=False, compare=False
    )
    _clean_input_state: CheapTensorSnapshot = field(repr=False, compare=False)
    _issuer_nonce: bytes = field(repr=False, compare=False)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "FrozenClassLogitTarget cannot be constructed directly; use "
            "resolve_clean_class_logit_target"
        )

    @classmethod
    def _create(
        cls,
        *,
        model: nn.Module,
        clean_inputs: torch.Tensor,
        class_index: int,
        clean_raw_logit: float,
        class_count: int,
        clean_input_sha256: str,
        clean_input_shape: tuple[int, int, int, int],
        official_truth_index: int | None,
        clean_top1_correct: bool | None,
        issuer_secret: bytes,
    ) -> "FrozenClassLogitTarget":
        if issuer_secret is not _TARGET_ISSUER_SECRET:
            raise TypeError(
                "frozen targets can only be issued by "
                "resolve_clean_class_logit_target"
            )
        signature = _model_state_signature(model)
        issuer_nonce = token_bytes(32)
        instance = object.__new__(cls)
        for name, value in {
            "class_index": class_index,
            "clean_raw_logit": clean_raw_logit,
            "class_count": class_count,
            "clean_input_sha256": clean_input_sha256,
            "clean_input_shape": clean_input_shape,
            "official_truth_index": official_truth_index,
            "clean_top1_correct": clean_top1_correct,
            "runtime_model_binding_sha256": _model_binding_sha256(signature),
            "_model_ref": weakref.ref(model),
            "_model_state_signature": signature,
            "_clean_input_ref": weakref.ref(clean_inputs),
            "_clean_input_state": CheapTensorSnapshot.capture(clean_inputs),
            "_issuer_nonce": issuer_nonce,
        }.items():
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        _ISSUED_TARGETS[instance] = _target_identity_snapshot(instance)
        return instance

    def __post_init__(self) -> None:
        if type(self.class_index) is not int or self.class_index < 0:
            raise TypeError("class_index must be a non-negative integer (not bool)")
        if type(self.class_count) is not int or self.class_count <= 0:
            raise TypeError("class_count must be a positive integer (not bool)")
        if self.class_index >= self.class_count:
            raise StageCClassLogitError(
                f"class_index {self.class_index} is outside {self.class_count} classes"
            )
        if isinstance(self.clean_raw_logit, bool) or not isinstance(
            self.clean_raw_logit, (int, float)
        ):
            raise TypeError("clean_raw_logit must be numeric")
        clean_raw_logit = float(self.clean_raw_logit)
        if not math.isfinite(clean_raw_logit):
            raise StageCClassLogitError("clean_raw_logit must be finite")
        object.__setattr__(self, "clean_raw_logit", clean_raw_logit)
        if not isinstance(self.clean_input_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.clean_input_sha256
        ):
            raise ValueError("clean_input_sha256 must be a lowercase SHA-256")
        if self.clean_input_shape != (1, 3, 224, 224):
            raise StageCClassLogitError(
                "clean_input_shape must be exactly (1,3,224,224)"
            )
        truth = _strict_optional_class_index(
            self.official_truth_index, name="official_truth_index"
        )
        if truth is not None and truth >= self.class_count:
            raise StageCClassLogitError(
                f"official_truth_index {truth} is outside {self.class_count} classes"
            )
        expected_correct = None if truth is None else truth == self.class_index
        if self.clean_top1_correct is not expected_correct:
            raise StageCClassLogitError(
                "clean_top1_correct must be None without official truth, or exactly "
                "match official_truth_index == class_index"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.runtime_model_binding_sha256):
            raise ValueError(
                "runtime_model_binding_sha256 must be a lowercase SHA-256"
            )
        if not isinstance(self._model_ref, weakref.ReferenceType):
            raise TypeError("frozen target has an invalid live-model reference")
        if not isinstance(self._model_state_signature, tuple):
            raise TypeError("frozen target has an invalid model-state signature")
        if not isinstance(self._clean_input_ref, weakref.ReferenceType):
            raise TypeError("frozen target has an invalid clean-input reference")
        if not isinstance(self._clean_input_state, CheapTensorSnapshot):
            raise TypeError("frozen target has an invalid clean-input snapshot")
        if not isinstance(self._issuer_nonce, bytes) or len(self._issuer_nonce) != 32:
            raise TypeError("frozen target has an invalid issuer nonce")

    @property
    def target_spec(self) -> TargetSpec:
        return TargetSpec.class_logit(self.class_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_kind": "raw_logit",
            "class_index": self.class_index,
            "class_index_source": "single_clean_input_argmax",
            "clean_raw_logit": self.clean_raw_logit,
            "class_count": self.class_count,
            "freeze_class_index_for_all_interventions": True,
            "official_truth_index": self.official_truth_index,
            "clean_top1_correct": self.clean_top1_correct,
            "exclude_misclassified": False,
            "clean_input_sha256": self.clean_input_sha256,
            "clean_input_shape": list(self.clean_input_shape),
            "runtime_model_binding": {
                "kind": "same_live_model_instance_and_tensor_state_versions",
                "sha256": self.runtime_model_binding_sha256,
                "is_canonical_model_artifact_hash": False,
            },
        }


@contextmanager
def _preserve_model_attribution_state(
    model: nn.Module, *, freeze_parameters: bool
) -> Iterator[None]:
    """Temporarily use eval/frozen parameters and restore caller state exactly."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    modules = tuple(model.modules())
    parameters = tuple(model.parameters())
    training_states = tuple(module.training for module in modules)
    requires_grad_states = tuple(parameter.requires_grad for parameter in parameters)
    # Attribution never needs parameter .grad.  Install private working clones
    # so even a caller-owned hook that writes ``parameter.grad`` cannot mutate
    # the caller's original tensor, version counter, layout, or inference flag.
    grad_states = tuple(
        (
            parameter.grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in parameters
    )
    try:
        for parameter, (_original_grad, working_grad) in zip(parameters, grad_states):
            parameter.grad = working_grad
        for module in modules:
            module.training = False
        if freeze_parameters:
            for parameter in parameters:
                parameter.requires_grad_(False)
        yield
    finally:
        for parameter, (original_grad, _working_grad) in zip(parameters, grad_states):
            parameter.grad = original_grad
        for parameter, requires_grad in zip(parameters, requires_grad_states):
            parameter.requires_grad_(requires_grad)
        for module, training in zip(modules, training_states):
            module.training = training


def _require_logits(output: Any, *, expected_classes: int | None = None) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError("model must return a raw-logit torch.Tensor")
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] <= 0:
        raise StageCClassLogitError(
            "model raw logits must have shape [1,classes], "
            f"got {tuple(output.shape)}"
        )
    if not output.is_floating_point():
        raise TypeError("model raw logits must be floating point")
    if expected_classes is not None and output.shape[1] != expected_classes:
        raise StageCClassLogitError(
            f"model class count changed from {expected_classes} to {output.shape[1]}"
        )
    if not bool(torch.isfinite(output.detach()).all().item()):
        raise StageCClassLogitError("model raw logits contain non-finite values")
    return output


def resolve_clean_class_logit_target(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    *,
    official_truth_index: int | None = None,
) -> FrozenClassLogitTarget:
    """Resolve the clean raw-logit argmax exactly once and freeze it."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    inputs = _validate_clean_input(clean_inputs)
    truth = _strict_optional_class_index(
        official_truth_index, name="official_truth_index"
    )
    model_state_before = _model_state_signature(model)
    with _preserve_model_attribution_state(model, freeze_parameters=True):
        with torch.no_grad():
            logits = _require_logits(model(inputs.detach()))
    if _model_state_signature(model) != model_state_before:
        raise StageCClassLogitError(
            "model parameter/buffer state changed during clean target resolution"
        )
    class_count = int(logits.shape[1])
    if truth is not None and truth >= class_count:
        raise StageCClassLogitError(
            f"official_truth_index {truth} is outside {class_count} classes"
        )
    # This is the sole class-selection operation.  C3 attribution wrappers
    # consume the resulting immutable index and never call argmax.
    class_index = int(torch.argmax(logits[0]).item())
    clean_raw_logit = float(logits[0, class_index].item())
    return FrozenClassLogitTarget._create(
        model=model,
        clean_inputs=inputs,
        class_index=class_index,
        clean_raw_logit=clean_raw_logit,
        class_count=class_count,
        clean_input_sha256=_tensor_sha256(inputs),
        clean_input_shape=tuple(inputs.shape),
        official_truth_index=truth,
        clean_top1_correct=None if truth is None else class_index == truth,
        issuer_secret=_TARGET_ISSUER_SECRET,
    )


def _verify_frozen_target(
    frozen_target: Any,
    clean_inputs: torch.Tensor,
    model: nn.Module,
    *,
    clean_input_sha256: str | None = None,
) -> FrozenClassLogitTarget:
    if not isinstance(frozen_target, FrozenClassLogitTarget):
        raise TypeError(
            "frozen_target must come from resolve_clean_class_logit_target"
        )
    if _ISSUED_TARGETS.get(frozen_target) != _target_identity_snapshot(
        frozen_target
    ):
        raise StageCClassLogitError(
            "frozen_target was not issued by the resolver or was mutated after issue"
        )
    if tuple(clean_inputs.shape) != frozen_target.clean_input_shape:
        raise StageCClassLogitError(
            "clean input shape does not match the frozen target context"
        )
    if frozen_target._clean_input_ref() is not clean_inputs:
        raise StageCClassLogitError(
            "clean input object does not match the frozen target context"
        )
    if not frozen_target._clean_input_state.verify(clean_inputs).matched:
        raise StageCClassLogitError(
            "clean input live state changed after target resolution"
        )
    clean_digest = _prebound_sha256(
        clean_input_sha256, name="clean_input_sha256"
    )
    if clean_digest is None:
        clean_digest = _tensor_sha256(clean_inputs)
    if clean_digest != frozen_target.clean_input_sha256:
        raise StageCClassLogitError(
            "clean input bytes do not match the frozen target context"
        )
    if frozen_target._model_ref() is not model:
        raise StageCClassLogitError(
            "model instance does not match the frozen target context"
        )
    current_signature = _model_state_signature(model)
    if current_signature != frozen_target._model_state_signature:
        raise StageCClassLogitError(
            "model parameter/buffer state changed after target resolution"
        )
    if (
        _model_binding_sha256(current_signature)
        != frozen_target.runtime_model_binding_sha256
    ):
        raise StageCClassLogitError(
            "model runtime binding hash does not match the frozen target context"
        )
    return frozen_target


def resolve_natural_prediction_observation(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    intervention_inputs: torch.Tensor,
    target: FrozenClassLogitTarget,
) -> dict[str, Any]:
    """Observe natural argmax at ``z`` while binding the frozen model context."""

    clean = _validate_clean_input(clean_inputs)
    intervention = _validate_clean_input(intervention_inputs)
    frozen = _verify_frozen_target(target, clean, model)
    caller_hash = _tensor_sha256(intervention)
    owned = intervention.detach().clone()
    with _preserve_model_attribution_state(model, freeze_parameters=True):
        with torch.no_grad():
            logits = _require_logits(model(owned), expected_classes=frozen.class_count)
            class_index = int(torch.argmax(logits[0]).item())
    _verify_frozen_target(frozen, clean, model)
    if _tensor_sha256(intervention) != caller_hash:
        raise StageCClassLogitError(
            "natural prediction observation mutated intervention_inputs"
        )
    return {
        "class_index": class_index, "class_count": frozen.class_count,
        "intervention_input_sha256": caller_hash,
        "runtime_model_binding_sha256": frozen.runtime_model_binding_sha256,
        "source": "natural_raw_logits_argmax",
    }


def evaluate_frozen_class_logit_target(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    intervention_inputs: torch.Tensor,
    target: FrozenClassLogitTarget,
    *,
    clean_input_sha256: str | None = None,
    intervention_input_sha256: str | None = None,
) -> FrozenScalarEvaluation:
    """Evaluate the originally resolved class logit at normalized ``z``.

    ``clean_inputs`` is validation context only.  Class selection is never
    repeated: the evaluator indexes the immutable resolver-issued class.
    """

    clean = _validate_clean_input(clean_inputs)
    intervention = _validate_clean_input(intervention_inputs)
    if clean.device != intervention.device:
        raise StageCClassLogitError(
            "intervention input device must match the clean input context"
        )
    clean_digest = _prebound_sha256(
        clean_input_sha256, name="clean_input_sha256"
    )
    intervention_digest = _prebound_sha256(
        intervention_input_sha256, name="intervention_input_sha256"
    )
    if clean_digest is None:
        clean_digest = _tensor_sha256(clean)
    if intervention_digest is None:
        intervention_digest = _tensor_sha256(intervention)
    frozen = _verify_frozen_target(
        target, clean, model, clean_input_sha256=clean_digest
    )
    owned_intervention = intervention.detach().clone()
    with _cheap_runtime_integrity_guard(
        model,
        tensors=(
            ("caller clean-input tensor", clean),
            ("caller intervention-input tensor", intervention),
            ("private intervention-input clone", owned_intervention),
        ),
        target=frozen,
        context="frozen class scalar evaluation",
    ):
        with _preserve_model_attribution_state(model, freeze_parameters=True):
            with torch.no_grad():
                logits = _require_logits(
                    model(owned_intervention), expected_classes=frozen.class_count
                )
            if logits.dtype is not torch.float32:
                raise TypeError(
                    f"model raw logits must have float32 dtype, got {logits.dtype}"
                )
            value = float(logits[0, frozen.class_index].item())
    _verify_frozen_target(
        frozen, clean, model, clean_input_sha256=clean_digest
    )
    if not math.isfinite(value):
        raise StageCClassLogitError("frozen class scalar is non-finite")
    return FrozenScalarEvaluation._create(
        value=value,
        class_index=frozen.class_index,
        clean_input_sha256=frozen.clean_input_sha256,
        intervention_input_sha256=intervention_digest,
        runtime_model_binding_sha256=frozen.runtime_model_binding_sha256,
        issuer_secret=_TARGET_ISSUER_SECRET,
    )


def _resolve_grad_cam_layer(model: nn.Module) -> nn.Module:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model must expose canonical blocks[11].norm1")
    try:
        block_count = len(blocks)
    except TypeError as exc:
        raise TypeError("model.blocks must be a sized module sequence") from exc
    if block_count != MODEL_BLOCKS:
        raise StageCClassLogitError(
            f"Grad-CAM requires exactly {MODEL_BLOCKS} model blocks, got {block_count}"
        )
    target_layer = getattr(blocks[11], "norm1", None)
    if not isinstance(target_layer, nn.Module):
        raise AttributeError("model.blocks[11] must expose norm1 as a module")
    return target_layer


class _ActivationCapture:
    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.call_count = 0
        self.activation: torch.Tensor | None = None
        self._handle: Any = None

    def _hook(
        self, _module: nn.Module, _arguments: tuple[Any, ...], output: Any
    ) -> None:
        self.call_count += 1
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{TARGET_LAYER} must return a Tensor")
        self.activation = output

    def __enter__(self) -> "_ActivationCapture":
        if self._handle is not None:
            raise RuntimeError("activation capture cannot be entered twice")
        existing_hook_ids = set(self.module._forward_hooks)
        try:
            self._handle = self.module.register_forward_hook(self._hook)
        except Exception:
            # A custom register_forward_hook implementation can call super()
            # and then raise before returning its handle.  Remove only hook IDs
            # created by this failed call; never clear caller-owned hooks.
            for hook_id in set(self.module._forward_hooks) - existing_hook_ids:
                self.module._forward_hooks.pop(hook_id, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def vit_grad_cam_patch_map(
    activation: torch.Tensor, gradient: torch.Tensor
) -> torch.Tensor:
    """Apply the frozen CLS exclusion, 196-token mean, sum, and ReLU."""

    for value, name in ((activation, "activation"), (gradient, "gradient")):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != TOKENS:
            raise StageCClassLogitError(
                f"{name} must have shape [1,{TOKENS},channels], got {tuple(value.shape)}"
            )
        if value.shape[2] <= 0:
            raise StageCClassLogitError(f"{name} must have at least one channel")
        if value.dtype is not torch.float32:
            raise TypeError(f"{name} must have float32 dtype, got {value.dtype}")
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise StageCClassLogitError(f"{name} contains non-finite values")
    if activation.shape != gradient.shape:
        raise StageCClassLogitError(
            "activation and gradient must have identical token/channel shape"
        )
    if activation.device != gradient.device:
        raise StageCClassLogitError(
            "activation and gradient must be on the same device"
        )

    # Index zero is CLS and is excluded from both operands.  The weights are
    # the arithmetic mean of target-logit gradients over exactly 196 patches.
    patch_activation = activation[:, 1:, :]
    patch_gradient = gradient[:, 1:, :]
    if patch_activation.shape[1] != PATCH_TOKENS:
        raise RuntimeError("CLS exclusion did not leave exactly 196 patch tokens")
    channel_weights = patch_gradient.mean(dim=1)
    weighted_channel_sum = (
        patch_activation * channel_weights[:, None, :]
    ).sum(dim=-1)
    cam = torch.relu(weighted_channel_sum).reshape(PATCH_SHAPE)
    cam = cam.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(cam).all().item()):
        raise StageCClassLogitError("Grad-CAM map contains non-finite values")
    if bool((cam < 0).any().item()):
        raise StageCClassLogitError("Grad-CAM ReLU produced a negative value")
    if int(torch.count_nonzero(cam).item()) == 0:
        raise StageCClassLogitError("Grad-CAM output is all zero")
    return cam


def _caller_context(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if provenance is None:
        return {}
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping or None")
    return dict(provenance)


def _base_provenance(
    *,
    exact_name: str,
    frozen_target: FrozenClassLogitTarget,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage": "C3",
        "exact_method_name": exact_name,
        "method_implementation_version": METHOD_VERSION,
        "comparison_line": "class_logit",
        "class_agnostic": False,
        "target": frozen_target.to_dict(),
        "target_resolution": "single_clean_raw_logits_argmax_then_frozen",
        "misclassified_samples_retained": True,
    }
    context = _caller_context(provenance)
    if context:
        result["caller_context"] = context
    return result


def vit_grad_cam_block11_norm1_at_intervention(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    intervention_inputs: torch.Tensor,
    frozen_target: FrozenClassLogitTarget,
    *,
    provenance: Mapping[str, Any] | None = None,
    intervention_input_sha256: str | None = None,
) -> MethodResult:
    """Compute frozen-class ViT Grad-CAM at arbitrary normalized ``z``."""

    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "ViT Grad-CAM cannot run inside torch.inference_mode()"
        )
    clean = _validate_clean_input(clean_inputs)
    inputs = _validate_clean_input(intervention_inputs)
    if clean.device != inputs.device:
        raise StageCClassLogitError(
            "intervention input device must match the clean input context"
        )
    intervention_digest = _prebound_sha256(
        intervention_input_sha256, name="intervention_input_sha256"
    )
    if intervention_digest is None:
        intervention_digest = _tensor_sha256(inputs)
    target = _verify_frozen_target(
        frozen_target, clean, model,
        clean_input_sha256=(
            frozen_target.clean_input_sha256
            if isinstance(frozen_target, FrozenClassLogitTarget) else None
        ),
    )
    target_layer = _resolve_grad_cam_layer(model)
    with _cheap_runtime_integrity_guard(
        model,
        tensors=(("caller intervention-input tensor", inputs),),
        target=target,
        context="Grad-CAM",
    ):
        with _preserve_model_attribution_state(
            model, freeze_parameters=True
        ), _ActivationCapture(target_layer) as capture, torch.enable_grad():
            probe = inputs.detach().clone().requires_grad_(True)
            logits = _require_logits(
                model(probe), expected_classes=target.class_count
            )
            if capture.call_count != 1 or capture.activation is None:
                raise StageCClassLogitError(
                    f"{TARGET_LAYER} must execute exactly once, got {capture.call_count}"
                )
            activation = capture.activation
            if activation.shape != (1, TOKENS, EMBED_DIM):
                raise StageCClassLogitError(
                    f"{TARGET_LAYER} must have shape [1,{TOKENS},{EMBED_DIM}], "
                    f"got {tuple(activation.shape)}"
                )
            selected_logit = target.target_spec.select(logits)
            if not selected_logit.requires_grad:
                raise StageCClassLogitError(
                    "frozen target logit is not differentiable from the target layer"
                )
            gradient = torch.autograd.grad(
                selected_logit.sum(), activation, retain_graph=False,
                create_graph=False, allow_unused=True,
            )[0]
            if gradient is None:
                raise StageCClassLogitError(
                    "frozen target logit does not depend on blocks[11].norm1"
                )
            cam = vit_grad_cam_patch_map(activation, gradient)
    _verify_frozen_target(
        target, clean, model, clean_input_sha256=target.clean_input_sha256
    )
    values = cam.to(device="cpu", dtype=torch.float32).numpy().copy()
    method_provenance = _base_provenance(
        exact_name=(
            "ViT Grad-CAM (blocks[11].norm1, clean predicted-class raw logit)"
        ),
        frozen_target=target,
        provenance=provenance,
    )
    method_provenance["intervention_input_sha256"] = intervention_digest
    method_provenance["algorithm"] = {
        "target_layer": TARGET_LAYER,
        "activation_layout": "batch_token_channel",
        "tokens": TOKENS,
        "exclude_cls_token": True,
        "patch_tokens": PATCH_TOKENS,
        "channel_weight": "arithmetic_mean_target_gradient_over_196_patch_tokens",
        "spatial_reduction": "weighted_activation_channel_sum",
        "activation": "relu",
        "minmax_normalization": False,
        "smoothing": False,
        "percentile_enhancement": False,
        "display_resize": False,
        "projection": "identity_14x14",
        "is_complete_attention_attribution": False,
    }
    return MethodResult(
        method_id="vit_grad_cam_block11_norm1",
        method_version=METHOD_VERSION,
        comparison_line=ComparisonLine.CLASS_LOGIT,
        explanation_target=TARGET_REF,
        fidelity_target=TARGET_REF,
        class_agnostic=False,
        native_tensor=raw_array(values),
        metric_map=stage_c_metric_ready(values),
        projection="identity_14x14",
        sign_semantics=SignSemantics.NONNEGATIVE,
        valid_mask=np.ones(PATCH_SHAPE, dtype=np.bool_),
        provenance=method_provenance,
    )


def vit_grad_cam_block11_norm1(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    frozen_target: FrozenClassLogitTarget,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> MethodResult:
    """Compute ViT Grad-CAM at the clean input for the frozen class."""

    return vit_grad_cam_block11_norm1_at_intervention(
        model, clean_inputs, clean_inputs, frozen_target, provenance=provenance,
        intervention_input_sha256=(
            frozen_target.clean_input_sha256
            if isinstance(frozen_target, FrozenClassLogitTarget) else None
        ),
    )


def _validate_ig_result(
    result: Any,
    *,
    target: FrozenClassLogitTarget,
    expected_delta: float | None = None,
) -> IGResult:
    if not isinstance(result, IGResult):
        raise TypeError("full_model_ig must return IGResult")
    if type(result.converged) is not bool:
        raise TypeError("IGResult.converged must be a boolean")
    if type(result.steps) is not int or result.steps not in IG_STEPS:
        raise StageCClassLogitError(
            f"full-model IG returned unregistered steps={result.steps!r}"
        )
    if not isinstance(result.attempts, tuple) or not result.attempts:
        raise StageCClassLogitError(
            "full-model IG must retain a non-empty tuple of attempts"
        )
    attempt_steps: list[int] = []
    attempt_passes: list[bool] = []
    for attempt_index, attempt in enumerate(result.attempts):
        if not isinstance(attempt, Mapping) or set(attempt) != {
            "steps", "delta", "sum", "abs", "rel", "pass"
        }:
            raise StageCClassLogitError(
                "full-model IG attempt has the wrong fields"
            )
        if type(attempt.get("steps")) is not int:
            raise StageCClassLogitError(
                "full-model IG attempt steps must be an integer"
            )
        attempt_steps.append(attempt["steps"])
        for key in ("delta", "sum", "abs", "rel"):
            values = attempt[key]
            if (
                not isinstance(values, list)
                or len(values) != 1
                or isinstance(values[0], bool)
                or not isinstance(values[0], (int, float))
                or not math.isfinite(float(values[0]))
            ):
                raise StageCClassLogitError(
                    f"full-model IG attempt {key} must be one finite numeric value"
                )
        attempt_delta = float(attempt["delta"][0])
        attempt_sum = float(attempt["sum"][0])
        attempt_abs = abs(attempt_delta - attempt_sum)
        attempt_rel = attempt_abs / max(
            abs(attempt_delta), torch.finfo(torch.float32).eps
        )
        if not math.isclose(
            float(attempt["abs"][0]), attempt_abs, abs_tol=1e-12, rel_tol=1e-6
        ):
            raise StageCClassLogitError(
                "full-model IG attempt absolute error is inconsistent"
            )
        if not math.isclose(
            float(attempt["rel"][0]), attempt_rel, abs_tol=1e-12, rel_tol=1e-6
        ):
            raise StageCClassLogitError(
                "full-model IG attempt relative error is inconsistent"
            )
        passes = attempt["pass"]
        if (
            not isinstance(passes, list)
            or len(passes) != 1
            or type(passes[0]) is not bool
        ):
            raise StageCClassLogitError(
                "full-model IG attempt pass must be one boolean"
            )
        expected_attempt_pass = attempt_abs <= (
            IG_ATOL + IG_RTOL * abs(attempt_delta)
        )
        if passes[0] is not expected_attempt_pass:
            raise StageCClassLogitError(
                "full-model IG attempt pass flag is inconsistent"
            )
        attempt_passes.append(passes[0])
    expected_attempts = list(IG_STEPS[: len(attempt_steps)])
    if attempt_steps != expected_attempts or result.steps != attempt_steps[-1]:
        raise StageCClassLogitError(
            "full-model IG attempts must be an exact frozen prefix of the "
            "schedule ending at result.steps"
        )
    passing_indices = [
        index for index, passed in enumerate(attempt_passes) if passed
    ]
    if result.converged:
        if passing_indices != [len(attempt_passes) - 1]:
            raise StageCClassLogitError(
                "full-model IG attempts must stop at the first conservation pass"
            )
    elif passing_indices or tuple(attempt_steps) != IG_STEPS:
        raise StageCClassLogitError(
            "nonconverged full-model IG must retain all five failed attempts"
        )
    native = result.attribution
    if not isinstance(native, torch.Tensor):
        raise TypeError("full-model IG attribution must be a Tensor")
    if native.shape != (1, 3, 224, 224):
        raise StageCClassLogitError(
            "full-model IG attribution must have shape [1,3,224,224], "
            f"got {tuple(native.shape)}"
        )
    if native.dtype is not torch.float32:
        raise TypeError(
            f"full-model IG attribution must be float32, got {native.dtype}"
        )
    if not bool(torch.isfinite(native.detach()).all().item()):
        raise StageCClassLogitError(
            "full-model IG attribution contains non-finite values"
        )
    if int(torch.count_nonzero(native.detach()).item()) == 0:
        raise StageCClassLogitError("full-model IG native attribution is all zero")

    required_report = {"delta", "sum", "abs", "rel", "pass"}
    if not isinstance(result.conservation, Mapping) or (
        set(result.conservation) != required_report
    ):
        raise StageCClassLogitError(
            "full-model IG conservation report has the wrong fields"
        )
    for name, value in result.conservation.items():
        if not isinstance(value, torch.Tensor) or value.shape != (1,):
            raise StageCClassLogitError(
                f"full-model IG conservation {name} must have shape [1]"
            )
        if name == "pass":
            if value.dtype is not torch.bool:
                raise TypeError(
                    "full-model IG conservation pass must have bool dtype"
                )
        else:
            if value.dtype is not torch.float32:
                raise TypeError(
                    f"full-model IG conservation {name} must be float32"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise StageCClassLogitError(
                    f"full-model IG conservation {name} is non-finite"
                )
    attribution_sum = native.flatten(1).sum(dim=1)
    report_delta = result.conservation["delta"].to(torch.float32)
    if expected_delta is not None and report_delta[0].item() != expected_delta:
        raise StageCClassLogitError(
            "full-model IG conservation delta does not equal t(z)-t(x0)"
        )
    report_sum = result.conservation["sum"].to(torch.float32)
    report_abs = result.conservation["abs"].to(torch.float32)
    report_rel = result.conservation["rel"].to(torch.float32)
    if not torch.allclose(attribution_sum, report_sum, atol=0.0, rtol=0.0):
        raise StageCClassLogitError(
            "full-model IG conservation sum does not equal attribution sum"
        )
    expected_abs = (report_delta - report_sum).abs()
    if not torch.allclose(expected_abs, report_abs, atol=0.0, rtol=0.0):
        raise StageCClassLogitError(
            "full-model IG conservation absolute error is inconsistent"
        )
    expected_rel = expected_abs / report_delta.abs().clamp_min(
        torch.finfo(torch.float32).eps
    )
    if not torch.allclose(expected_rel, report_rel, atol=0.0, rtol=0.0):
        raise StageCClassLogitError(
            "full-model IG conservation relative error is inconsistent"
        )
    expected_pass = expected_abs <= (IG_ATOL + IG_RTOL * report_delta.abs())
    if not torch.equal(expected_pass, result.conservation["pass"]):
        raise StageCClassLogitError(
            "full-model IG conservation pass flag is inconsistent"
        )
    if bool(result.conservation["pass"].all().item()) is not result.converged:
        raise StageCClassLogitError(
            "full-model IG converged flag contradicts conservation"
        )
    final_attempt = result.attempts[-1]
    for key in ("delta", "sum", "abs", "rel", "pass"):
        if final_attempt[key][0] != result.conservation[key][0].item():
            raise StageCClassLogitError(
                f"full-model IG final attempt {key} does not match conservation"
            )
    metadata = result.metadata
    if not isinstance(metadata, Mapping):
        raise TypeError("full-model IG metadata must be a mapping")
    expected_metadata = {
        "method": "integrated_gradients",
        "quadrature": "gauss_legendre",
        "integration_dtype": "float32",
        "input_shape": [1, 3, 224, 224],
        "step_schedule": list(IG_STEPS),
        "step_chunk_size": IG_STEP_CHUNK_SIZE,
        "atol": IG_ATOL,
        "rtol": IG_RTOL,
        "scope": "full_model",
        "target": target.target_spec.to_dict(),
        "includes_residual_paths": True,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise StageCClassLogitError(
                f"full-model IG metadata {key!r} does not match the frozen contract"
            )
    return result


def _ig_provenance_summary(result: IGResult) -> dict[str, Any]:
    return {
        "selected_steps": result.steps,
        "converged": result.converged,
        "attempts": [dict(attempt) for attempt in result.attempts],
        "conservation": {
            key: value.detach().to(device="cpu").tolist()
            for key, value in result.conservation.items()
        },
        "metadata": dict(result.metadata),
    }


def full_model_class_logit_ig(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    frozen_target: FrozenClassLogitTarget,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> MethodResult:
    """Adapt real full-model raw-class-logit IG to the Stage C map contract."""

    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "full-model class-logit IG cannot run inside torch.inference_mode()"
        )
    inputs = _validate_clean_input(clean_inputs)
    target = _verify_frozen_target(
        frozen_target, inputs, model,
        clean_input_sha256=(
            frozen_target.clean_input_sha256
            if isinstance(frozen_target, FrozenClassLogitTarget) else None
        ),
    )
    baseline = torch.zeros_like(inputs)

    with _preserve_model_attribution_state(
        model, freeze_parameters=True
    ), torch.enable_grad():
        result = attribution.full_model_ig(
            model,
            inputs,
            baseline,
            target.target_spec,
            steps=IG_STEPS,
            step_chunk_size=IG_STEP_CHUNK_SIZE,
            atol=IG_ATOL,
            rtol=IG_RTOL,
        )
    result = _validate_ig_result(result, target=target)
    _verify_frozen_target(
        target, inputs, model, clean_input_sha256=target.clean_input_sha256
    )
    if not result.converged:
        raise StageCClassLogitIGConvergenceError(
            "full-model class-logit IG did not converge on the complete "
            "frozen schedule",
            result,
        )

    native_values = (
        result.attribution[0].detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    )
    native = raw_array(native_values)
    metric = pixel_to_patch_mass(signed_rgb_channel_sum(native))
    if int(np.count_nonzero(metric.data)) == 0:
        raise StageCClassLogitError(
            "full-model IG signed channel-sum patch map is all zero"
        )

    method_provenance = _base_provenance(
        exact_name="full-model class-logit IG",
        frozen_target=target,
        provenance=provenance,
    )
    method_provenance["algorithm"] = {
        "model_scope": "real_unmodified_full_model",
        "scalar": "frozen_clean_predicted_class_raw_logit",
        "baseline": {
            "id": "zero_normalized",
            "normalized_value": 0.0,
            "canonical_rgb_value": 0.5,
            "is_black_image": False,
        },
        "quadrature": "gauss_legendre",
        "integration_dtype": "float32",
        "steps_schedule": list(IG_STEPS),
        "step_chunk_size": IG_STEP_CHUNK_SIZE,
        "atol": IG_ATOL,
        "rtol": IG_RTOL,
        "native_attribution": "signed_rgb_float32_3x224x224",
        "rgb_reduction": "signed_channel_sum_before_projection",
        "projection": "nonoverlapping_16x16_pixel_mass_to_patch_sum",
        "absolute_value_before_projection": False,
        "claim_boundary": "full_model_raw_class_logit_attribution",
        "is_complete_attention_attribution": False,
    }
    method_provenance["ig_result"] = _ig_provenance_summary(result)
    return MethodResult(
        method_id="full_model_class_logit_ig",
        method_version=METHOD_VERSION,
        comparison_line=ComparisonLine.CLASS_LOGIT,
        explanation_target=TARGET_REF,
        fidelity_target=TARGET_REF,
        class_agnostic=False,
        native_tensor=native,
        metric_map=metric,
        projection="pixel_mass_to_patch_sum",
        sign_semantics=SignSemantics.SIGNED,
        valid_mask=np.ones(PATCH_SHAPE, dtype=np.bool_),
        provenance=method_provenance,
    )


def full_model_class_logit_ig_at_intervention(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    intervention_inputs: torch.Tensor,
    frozen_target: FrozenClassLogitTarget,
    *,
    provenance: Mapping[str, Any] | None = None,
    intervention_input_sha256: str | None = None,
) -> MethodResult:
    """Run full-model IG from normalized zero to arbitrary ``z``."""

    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "full-model class-logit IG cannot run inside torch.inference_mode()"
        )
    clean = _validate_clean_input(clean_inputs)
    inputs = _validate_clean_input(intervention_inputs)
    if clean.device != inputs.device:
        raise StageCClassLogitError(
            "intervention input device must match the clean input context"
        )
    intervention_digest = _prebound_sha256(
        intervention_input_sha256, name="intervention_input_sha256"
    )
    if intervention_digest is None:
        intervention_digest = _tensor_sha256(inputs)
    target = _verify_frozen_target(
        frozen_target, clean, model,
        clean_input_sha256=(
            frozen_target.clean_input_sha256
            if isinstance(frozen_target, FrozenClassLogitTarget) else None
        ),
    )
    baseline = torch.zeros_like(inputs)
    owned_inputs = inputs.detach().clone()
    with _cheap_runtime_integrity_guard(
        model,
        tensors=(
            ("caller intervention-input tensor", inputs),
            ("private intervention-input clone", owned_inputs),
        ),
        target=target,
        context="full-model intervention IG",
    ):
        with _preserve_model_attribution_state(
            model, freeze_parameters=True
        ), torch.enable_grad():
            with torch.no_grad():
                endpoint = _require_logits(
                    model(owned_inputs.detach()), expected_classes=target.class_count
                )[0, target.class_index]
                baseline_endpoint = _require_logits(
                    model(baseline), expected_classes=target.class_count
                )[0, target.class_index]
                if endpoint.dtype is not torch.float32 or (
                    baseline_endpoint.dtype is not torch.float32
                ):
                    raise TypeError(
                        "model raw logits must have float32 dtype for intervention IG"
                    )
                expected_delta = float((endpoint - baseline_endpoint).item())
            result = attribution.full_model_ig(
                model, owned_inputs, baseline, target.target_spec,
                steps=IG_STEPS, step_chunk_size=IG_STEP_CHUNK_SIZE,
                atol=IG_ATOL, rtol=IG_RTOL,
            )
    result = _validate_ig_result(
        result, target=target, expected_delta=expected_delta
    )
    _verify_frozen_target(
        target, clean, model, clean_input_sha256=target.clean_input_sha256
    )
    if not result.converged:
        raise StageCClassLogitIGConvergenceError(
            "full-model class-logit IG did not converge on the complete "
            "frozen schedule",
            result,
        )
    native_values = result.attribution[0].detach().to(
        device="cpu", dtype=torch.float32
    ).numpy().copy()
    native = raw_array(native_values)
    metric = pixel_to_patch_mass(signed_rgb_channel_sum(native))
    if int(np.count_nonzero(metric.data)) == 0:
        raise StageCClassLogitError(
            "full-model IG signed channel-sum patch map is all zero"
        )
    method_provenance = _base_provenance(
        exact_name="full-model class-logit IG",
        frozen_target=target, provenance=provenance,
    )
    method_provenance["intervention_input_sha256"] = intervention_digest
    method_provenance["algorithm"] = {
        "model_scope": "real_unmodified_full_model",
        "scalar": "frozen_clean_predicted_class_raw_logit",
        "baseline": {
            "id": "zero_normalized", "normalized_value": 0.0,
            "canonical_rgb_value": 0.5, "is_black_image": False,
        },
        "quadrature": "gauss_legendre",
        "integration_dtype": "float32",
        "steps_schedule": list(IG_STEPS),
        "step_chunk_size": IG_STEP_CHUNK_SIZE, "atol": IG_ATOL,
        "rtol": IG_RTOL,
        "native_attribution": "signed_rgb_float32_3x224x224",
        "rgb_reduction": "signed_channel_sum_before_projection",
        "projection": "nonoverlapping_16x16_pixel_mass_to_patch_sum",
        "absolute_value_before_projection": False,
        "claim_boundary": "full_model_raw_class_logit_attribution",
        "is_complete_attention_attribution": False,
    }
    method_provenance["ig_result"] = _ig_provenance_summary(result)
    return MethodResult(
        method_id="full_model_class_logit_ig", method_version=METHOD_VERSION,
        comparison_line=ComparisonLine.CLASS_LOGIT, explanation_target=TARGET_REF,
        fidelity_target=TARGET_REF, class_agnostic=False, native_tensor=native,
        metric_map=metric, projection="pixel_mass_to_patch_sum",
        sign_semantics=SignSemantics.SIGNED,
        valid_mask=np.ones(PATCH_SHAPE, dtype=np.bool_),
        provenance=method_provenance,
    )


__all__ = [
    "EMBED_DIM",
    "FrozenClassLogitTarget",
    "FrozenScalarEvaluation",
    "IG_ATOL",
    "IG_RTOL",
    "IG_STEPS",
    "IG_STEP_CHUNK_SIZE",
    "METHOD_VERSION",
    "PATCH_TOKENS",
    "StageCClassLogitError",
    "StageCClassLogitIGConvergenceError",
    "TARGET_LAYER",
    "TOKENS",
    "full_model_class_logit_ig",
    "full_model_class_logit_ig_at_intervention",
    "evaluate_frozen_class_logit_target",
    "resolve_clean_class_logit_target",
    "resolve_natural_prediction_observation",
    "vit_grad_cam_block11_norm1",
    "vit_grad_cam_block11_norm1_at_intervention",
    "vit_grad_cam_patch_map",
]
