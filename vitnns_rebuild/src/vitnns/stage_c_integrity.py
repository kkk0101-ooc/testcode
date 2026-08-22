"""Layered runtime-integrity checks for Stage C.

The two layers in this module deliberately have different costs and threat
models:

* :func:`strong_model_signature` reads parameter, persistent-buffer and
  parameter-gradient bytes.  It is intended for model-load and execution-unit
  boundaries only.
* :func:`model_live_signature` and the cheap guards inspect identities, storage
  bindings, version counters and metadata without copying tensor contents to
  CPU.  They are safe to use around method, target, scalar and curve-point hot
  paths.

Keeping the strong signer as a module-level function is intentional: production
instrumentation and regression tests can count its calls directly.
"""

from __future__ import annotations

import hashlib
import weakref
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Any, Callable

import torch
from torch import nn


STRONG_SIGNATURE_ALGORITHM = "stage_c_strong_model_sha256_v1"
CHEAP_MODEL_GUARD_ALGORITHM = "stage_c_cheap_model_live_state_v1"
CHEAP_TENSOR_GUARD_ALGORITHM = "stage_c_cheap_tensor_live_state_v1"

_HOOK_ATTRIBUTES = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)

_STRONG_CHECKED = (
    "module_topology_and_object_identity",
    "all_module_training_modes",
    "all_module_forward_backward_and_state_dict_hook_identities",
    "attention_fused_attn_flags",
    "parameter_name_object_storage_metadata_version_and_content",
    "persistent_buffer_name_object_storage_metadata_version_and_content",
    "all_parameter_requires_grad_flags",
    "all_parameter_grad_identity_storage_metadata_and_content",
)

_CHEAP_MODEL_CHECKED = (
    "module_topology_and_object_identity",
    "parameter_and_buffer_object_storage_identity",
    "tensor_version_shape_dtype_device_layout",
    "parameter_requires_grad",
)

_CHEAP_TENSOR_CHECKED = (
    "tensor_object_and_storage_identity",
    "tensor_version_shape_dtype_device_layout_and_requires_grad",
)


class ModelIntegrityError(RuntimeError):
    """A model or guarded tensor crossed an integrity boundary changed."""

    def __init__(
        self,
        message: str | "IntegrityEvidence",
        *,
        evidence: "IntegrityEvidence | None" = None,
        body_exception: BaseException | None = None,
        restoration_evidence: "RestorationEvidence | None" = None,
    ) -> None:
        if isinstance(message, IntegrityEvidence):
            if evidence is not None:
                raise TypeError("evidence must not be supplied twice")
            evidence = message
            message = (
                "model integrity verification failed: "
                + ", ".join(evidence.differences)
            )
        if evidence is None:
            raise TypeError("evidence is required")
        super().__init__(message)
        self.evidence = evidence
        self.body_exception = body_exception
        self.restoration_evidence = restoration_evidence


@dataclass(frozen=True)
class IntegrityEvidence:
    """Serializable result of one integrity comparison.

    Runtime addresses and per-tensor content hashes are intentionally omitted
    from :meth:`to_dict`; the evidence records only aggregate digests and
    human-auditable difference labels.
    """

    algorithm: str
    checked: tuple[str, ...]
    entry_sha256: str
    exit_sha256: str | None
    matched: bool
    differences: tuple[str, ...]
    signature_calls: int
    verification_error_type: str | None = None
    verification_error_message: str | None = None
    body_exception_type: str | None = None

    @property
    def restored(self) -> bool:
        return self.matched

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "algorithm": self.algorithm,
            "checked": list(self.checked),
            "entry_sha256": self.entry_sha256,
            "exit_sha256": self.exit_sha256,
            "matched": self.matched,
            "restored": self.restored,
            "differences": list(self.differences),
            "signature_calls": self.signature_calls,
            "verification_error_type": self.verification_error_type,
            "verification_error_message": self.verification_error_message,
            "body_exception_type": self.body_exception_type,
        }


@dataclass(frozen=True)
class RestorationEvidence:
    """Result of restoring caller-owned lifecycle state before verification."""

    checked: tuple[str, ...]
    changes_detected: bool
    restored: bool
    differences: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "checked": list(self.checked),
            "changes_detected": self.changes_detected,
            "restored": self.restored,
            "differences": list(self.differences),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class _ModuleLifecycleState:
    name: str
    module: nn.Module
    training: bool
    hooks: tuple[tuple[str, Any, tuple[tuple[Any, Any], ...]], ...]
    fused_attn_present: bool
    fused_attn: Any
    full_backward_hook_present: bool
    full_backward_hook: Any


@dataclass(frozen=True)
class _ParameterLifecycleState:
    name: str
    parameter: nn.Parameter
    requires_grad: bool
    grad: torch.Tensor | None
    grad_metadata_signature: tuple[Any, ...] | None
    grad_content_sha256: str | None
    grad_storage_anchor: torch.Tensor | None
    grad_content: torch.Tensor | None


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _safe_version(value: torch.Tensor) -> int | str:
    try:
        return int(value._version)
    except RuntimeError:
        return "unavailable_inference_tensor"


def _storage_identity(value: torch.Tensor) -> tuple[int | str, int | str]:
    try:
        storage = value.untyped_storage()
        storage_identity: int | str = int(storage._cdata)
        storage_data_ptr: int | str = int(storage.data_ptr())
    except (AttributeError, RuntimeError):
        storage_identity = "unavailable"
        storage_data_ptr = "unavailable"
    return storage_identity, storage_data_ptr


def tensor_live_signature(
    value: torch.Tensor, *, include_requires_grad: bool = True
) -> tuple[Any, ...]:
    """Return a content-free descriptor for one live tensor."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a torch.Tensor")
    storage_identity, storage_data_ptr = _storage_identity(value)
    try:
        data_ptr: int | str = int(value.data_ptr())
    except RuntimeError:
        data_ptr = "unavailable"
    return (
        id(value),
        storage_identity,
        storage_data_ptr,
        data_ptr,
        _safe_version(value),
        tuple(value.shape),
        tuple(value.stride()),
        int(value.storage_offset()),
        str(value.dtype),
        str(value.device),
        str(value.layout),
        bool(value.requires_grad) if include_requires_grad else None,
    )


def _restorable_tensor_signature(value: torch.Tensor) -> tuple[Any, ...]:
    """Describe tensor metadata whose original object can be restored."""

    storage_identity, storage_data_ptr = _storage_identity(value)
    return (
        id(value),
        storage_identity,
        storage_data_ptr,
        tuple(value.shape),
        tuple(value.stride()),
        int(value.storage_offset()),
        str(value.dtype),
        str(value.device),
        str(value.layout),
        bool(value.requires_grad),
    )


def _tensor_content_sha256(value: torch.Tensor) -> str:
    """Hash exact tensor bytes after one deliberate CPU transfer."""

    detached = value.detach()
    if detached.layout is not torch.strided:
        detached = detached.to_dense()
    cpu = detached.to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(item) for item in cpu.shape).encode("ascii"))
    digest.update(b"|")
    # Viewing bytes supports dtypes such as bfloat16 that NumPy cannot expose
    # directly in every supported environment.
    if cpu.ndim == 0:
        cpu = cpu.reshape(1)
    digest.update(cpu.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _hook_signature(module: nn.Module) -> tuple[tuple[Any, ...], ...]:
    result: list[tuple[Any, ...]] = []
    for attribute in _HOOK_ATTRIBUTES:
        hooks = getattr(module, attribute, {})
        try:
            items = tuple(hooks.items())
        except AttributeError:
            items = ()
        result.append(
            (attribute, tuple((repr(key), id(callback)) for key, callback in items))
        )
    return tuple(result)


def _module_signature(
    name: str, module: nn.Module, *, include_lifecycle: bool
) -> tuple[Any, ...]:
    base: tuple[Any, ...] = (
        "module",
        name,
        id(module),
        _qualified_type(module),
        tuple(module._modules),
        tuple(module._parameters),
        tuple(
            (buffer_name, buffer_name not in module._non_persistent_buffers_set)
            for buffer_name in module._buffers
        ),
    )
    if not include_lifecycle:
        return base
    fused_attn = (
        ("present", bool(module.fused_attn))
        if hasattr(module, "fused_attn")
        else ("absent", None)
    )
    return base + (
        bool(module.training),
        _hook_signature(module),
        fused_attn,
        getattr(module, "_is_full_backward_hook", None),
    )


def _persistent_named_buffers(
    model: nn.Module,
) -> tuple[tuple[str, torch.Tensor], ...]:
    result: list[tuple[str, torch.Tensor]] = []
    for module_name, module in model.named_modules(remove_duplicate=False):
        for local_name, value in module._buffers.items():
            if value is None or local_name in module._non_persistent_buffers_set:
                continue
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            result.append((full_name, value))
    return tuple(result)


def _capture_lifecycle_state(
    model: nn.Module,
) -> tuple[
    tuple[_ModuleLifecycleState, ...], tuple[_ParameterLifecycleState, ...]
]:
    modules: list[_ModuleLifecycleState] = []
    for name, module in model.named_modules(remove_duplicate=False):
        hook_states: list[tuple[str, Any, tuple[tuple[Any, Any], ...]]] = []
        for attribute in _HOOK_ATTRIBUTES:
            hooks = getattr(module, attribute, None)
            if hooks is None:
                continue
            try:
                items = tuple(hooks.items())
            except AttributeError:
                continue
            hook_states.append((attribute, hooks, items))
        modules.append(
            _ModuleLifecycleState(
                name=name,
                module=module,
                training=bool(module.training),
                hooks=tuple(hook_states),
                fused_attn_present=hasattr(module, "fused_attn"),
                fused_attn=getattr(module, "fused_attn", None),
                full_backward_hook_present=hasattr(
                    module, "_is_full_backward_hook"
                ),
                full_backward_hook=getattr(
                    module, "_is_full_backward_hook", None
                ),
            )
        )
    parameters: list[_ParameterLifecycleState] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        grad = parameter.grad
        # Preserve a private byte copy for restoration while retaining the
        # original object/storage anchor.  This cost is unit-boundary-only.
        grad_content = None if grad is None else grad.detach().clone()
        parameters.append(
            _ParameterLifecycleState(
                name=name,
                parameter=parameter,
                requires_grad=bool(parameter.requires_grad),
                grad=grad,
                grad_metadata_signature=(
                    None if grad is None else _restorable_tensor_signature(grad)
                ),
                grad_content_sha256=(
                    None if grad is None else _tensor_content_sha256(grad)
                ),
                # A distinct TensorImpl keeps the original storage alive even
                # if caller code rebinds the original grad with ``set_``.
                grad_storage_anchor=(None if grad is None else grad.detach()),
                grad_content=grad_content,
            )
        )
    return tuple(modules), tuple(parameters)


def _restore_mapping(
    current: Any, expected_items: tuple[tuple[Any, Any], ...]
) -> None:
    current.clear()
    current.update(expected_items)


def _lifecycle_difference_labels(
    model: nn.Module,
    modules: tuple[_ModuleLifecycleState, ...],
    parameters: tuple[_ParameterLifecycleState, ...],
) -> tuple[str, ...]:
    differences: list[str] = []
    current_modules = tuple(model.named_modules(remove_duplicate=False))
    expected_modules = tuple((state.name, state.module) for state in modules)
    if len(current_modules) != len(expected_modules) or any(
        name != expected_name or module is not expected_module
        for (name, module), (expected_name, expected_module)
        in zip(current_modules, expected_modules)
    ):
        differences.append("module_topology")
    for state in modules:
        module = state.module
        if bool(module.training) != state.training:
            differences.append(f"module_mode:{state.name}")
        for attribute, expected_mapping, expected_items in state.hooks:
            current_mapping = getattr(module, attribute, None)
            if current_mapping is not expected_mapping or (
                current_mapping is not None
                and tuple(current_mapping.items()) != expected_items
            ):
                differences.append(f"module_hooks:{state.name}:{attribute}")
        if (
            hasattr(module, "fused_attn") != state.fused_attn_present
            or (
                state.fused_attn_present
                and getattr(module, "fused_attn") != state.fused_attn
            )
        ):
            differences.append(f"module_fused_attn:{state.name}")
        if (
            hasattr(module, "_is_full_backward_hook")
            != state.full_backward_hook_present
            or (
                state.full_backward_hook_present
                and getattr(module, "_is_full_backward_hook")
                != state.full_backward_hook
            )
        ):
            differences.append(f"module_full_backward_hook:{state.name}")
    current_parameters = tuple(model.named_parameters(remove_duplicate=False))
    expected_parameters = tuple(
        (state.name, state.parameter) for state in parameters
    )
    if len(current_parameters) != len(expected_parameters) or any(
        name != expected_name or parameter is not expected_parameter
        for (name, parameter), (expected_name, expected_parameter)
        in zip(current_parameters, expected_parameters)
    ):
        differences.append("parameter_topology")
    for state in parameters:
        parameter = state.parameter
        if bool(parameter.requires_grad) != state.requires_grad:
            differences.append(f"parameter_requires_grad:{state.name}")
        if parameter.grad is not state.grad:
            differences.append(f"parameter_grad_identity:{state.name}")
        elif parameter.grad is not None and (
            _restorable_tensor_signature(parameter.grad)
            != state.grad_metadata_signature
            or _tensor_content_sha256(parameter.grad)
            != state.grad_content_sha256
        ):
            differences.append(f"parameter_grad_state:{state.name}")
    return tuple(dict.fromkeys(differences))


def _restore_lifecycle_state(
    model: nn.Module,
    modules: tuple[_ModuleLifecycleState, ...],
    parameters: tuple[_ParameterLifecycleState, ...],
) -> RestorationEvidence:
    differences = _lifecycle_difference_labels(model, modules, parameters)
    errors: list[str] = []
    # Restore only caller-owned lifecycle state.  Parameter and persistent
    # buffer values/storage are never repaired here; the final strong compare
    # must fail closed for those changes.
    for state in modules:
        module = state.module
        try:
            module.training = state.training
        except Exception as error:
            errors.append(f"module_mode:{state.name}:{error}")
        for attribute, expected_mapping, expected_items in state.hooks:
            try:
                current_mapping = getattr(module, attribute, None)
                if current_mapping is not expected_mapping:
                    setattr(module, attribute, expected_mapping)
                    current_mapping = expected_mapping
                _restore_mapping(current_mapping, expected_items)
            except Exception as error:
                errors.append(f"module_hooks:{state.name}:{attribute}:{error}")
        try:
            if state.fused_attn_present:
                setattr(module, "fused_attn", state.fused_attn)
            elif hasattr(module, "fused_attn"):
                delattr(module, "fused_attn")
        except Exception as error:
            errors.append(f"module_fused_attn:{state.name}:{error}")
        try:
            if state.full_backward_hook_present:
                setattr(
                    module, "_is_full_backward_hook", state.full_backward_hook
                )
            elif hasattr(module, "_is_full_backward_hook"):
                delattr(module, "_is_full_backward_hook")
        except Exception as error:
            errors.append(f"module_full_backward_hook:{state.name}:{error}")
    for state in parameters:
        parameter = state.parameter
        try:
            parameter.requires_grad_(state.requires_grad)
        except Exception as error:
            errors.append(f"parameter_requires_grad:{state.name}:{error}")
        try:
            parameter.grad = state.grad
            if (
                state.grad is not None
                and state.grad_storage_anchor is not None
                and state.grad_content is not None
            ):
                with torch.no_grad():
                    state.grad.set_(state.grad_storage_anchor)
                    state.grad.copy_(state.grad_content)
        except Exception as error:
            errors.append(f"parameter_grad:{state.name}:{error}")
    remaining = _lifecycle_difference_labels(model, modules, parameters)
    return RestorationEvidence(
        checked=(
            "all_module_training_modes",
            "all_module_hook_mapping_identities_and_entries",
            "all_parameter_requires_grad_flags",
            "all_parameter_grad_identity_storage_metadata_and_content",
            "attention_fused_attn_flags",
        ),
        changes_detected=bool(differences),
        restored=not remaining and not errors,
        differences=differences,
        errors=tuple(errors or remaining),
    )


def model_live_signature(
    model: nn.Module,
    *,
    include_lifecycle: bool = False,
    include_requires_grad: bool = True,
) -> tuple[tuple[Any, ...], ...]:
    """Capture model topology and cheap live tensor state without contents."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    signature: list[tuple[Any, ...]] = [
        _module_signature(name, module, include_lifecycle=include_lifecycle)
        for name, module in model.named_modules(remove_duplicate=False)
    ]
    for name, parameter in model.named_parameters(remove_duplicate=False):
        signature.append(
            (
                "parameter",
                name,
                *tensor_live_signature(
                    parameter, include_requires_grad=include_requires_grad
                ),
            )
        )
        if include_lifecycle:
            signature.append(
                (
                    "parameter_grad",
                    name,
                    None
                    if parameter.grad is None
                    else _restorable_tensor_signature(parameter.grad),
                )
            )
    for name, buffer in model.named_buffers(remove_duplicate=False):
        signature.append(
            (
                "buffer",
                name,
                *tensor_live_signature(
                    buffer, include_requires_grad=include_requires_grad
                ),
            )
        )
    return tuple(signature)


def strong_model_signature(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    """Capture the full execution-unit model and lifecycle signature.

    This is the only Stage C model signer that reads tensor contents.  Do not
    call it from method, scalar, fidelity-point or IG chunk code.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    signature: list[tuple[Any, ...]] = [
        _module_signature(name, module, include_lifecycle=True)
        for name, module in model.named_modules(remove_duplicate=False)
    ]
    for name, parameter in model.named_parameters(remove_duplicate=False):
        signature.append(
            (
                "parameter",
                name,
                *tensor_live_signature(parameter),
                _tensor_content_sha256(parameter),
            )
        )
        grad = parameter.grad
        signature.append(
            (
                "parameter_grad",
                name,
                None
                if grad is None
                else (
                    *_restorable_tensor_signature(grad),
                    _tensor_content_sha256(grad),
                ),
            )
        )
    for name, buffer in _persistent_named_buffers(model):
        signature.append(
            (
                "persistent_buffer",
                name,
                *tensor_live_signature(buffer),
                _tensor_content_sha256(buffer),
            )
        )
    return tuple(signature)


def _signature_sha256(signature: tuple[tuple[Any, ...], ...]) -> str:
    return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()


def _signature_differences(
    before: tuple[tuple[Any, ...], ...],
    after: tuple[tuple[Any, ...], ...],
) -> tuple[str, ...]:
    if before == after:
        return ()
    try:
        before_by_key = {(item[0], item[1]): item for item in before}
        after_by_key = {(item[0], item[1]): item for item in after}
    except (IndexError, TypeError):
        return ("model_state",)
    differences: list[str] = []
    if set(before_by_key) != set(after_by_key):
        differences.append("model_topology")
    for key in sorted(set(before_by_key) & set(after_by_key), key=repr):
        if before_by_key[key] != after_by_key[key]:
            differences.append(f"{key[0]}:{key[1]}")
    return tuple(differences or ("model_state",))


@dataclass(frozen=True)
class StrongModelSnapshot:
    """One immutable full-content model snapshot."""

    _model_ref: weakref.ReferenceType[nn.Module]
    model_object_id: int
    signature: tuple[tuple[Any, ...], ...]
    sha256: str
    _module_lifecycle: tuple[_ModuleLifecycleState, ...]
    _parameter_lifecycle: tuple[_ParameterLifecycleState, ...]

    @classmethod
    def capture(cls, model: nn.Module) -> "StrongModelSnapshot":
        module_lifecycle, parameter_lifecycle = _capture_lifecycle_state(model)
        signature = strong_model_signature(model)
        return cls(
            _model_ref=weakref.ref(model),
            model_object_id=id(model),
            signature=signature,
            sha256=_signature_sha256(signature),
            _module_lifecycle=module_lifecycle,
            _parameter_lifecycle=parameter_lifecycle,
        )

    @property
    def digest(self) -> str:
        return self.sha256

    def restore_lifecycle(
        self, model: nn.Module | None = None
    ) -> RestorationEvidence:
        bound = self._model_ref() if model is None else model
        if bound is None:
            return RestorationEvidence(
                checked=("bound_model_lifetime",),
                changes_detected=True,
                restored=False,
                differences=("model_reference_expired",),
                errors=("model_reference_expired",),
            )
        if id(bound) != self.model_object_id:
            return RestorationEvidence(
                checked=("model_object_identity",),
                changes_detected=True,
                restored=False,
                differences=("model_object_identity",),
                errors=("model_object_identity",),
            )
        return _restore_lifecycle_state(
            bound, self._module_lifecycle, self._parameter_lifecycle
        )

    def compare_snapshot(
        self, current: "StrongModelSnapshot", *, signature_calls: int = 0
    ) -> IntegrityEvidence:
        if not isinstance(current, StrongModelSnapshot):
            raise TypeError("current must be a StrongModelSnapshot")
        differences = list(_signature_differences(self.signature, current.signature))
        if self.model_object_id != current.model_object_id:
            differences.insert(0, "model_object_identity")
        return IntegrityEvidence(
            algorithm=STRONG_SIGNATURE_ALGORITHM,
            checked=_STRONG_CHECKED,
            entry_sha256=self.sha256,
            exit_sha256=current.sha256,
            matched=not differences,
            differences=tuple(dict.fromkeys(differences)),
            signature_calls=signature_calls,
        )

    def verify(self, model: nn.Module | None = None) -> IntegrityEvidence:
        bound = self._model_ref() if model is None else model
        if bound is None:
            return IntegrityEvidence(
                algorithm=STRONG_SIGNATURE_ALGORITHM,
                checked=_STRONG_CHECKED,
                entry_sha256=self.sha256,
                exit_sha256=None,
                matched=False,
                differences=("model_reference_expired",),
                signature_calls=0,
            )
        try:
            # Verification needs only a fresh signature.  Avoid retaining a
            # second full lifecycle snapshot and duplicate gradient clones.
            current_signature = strong_model_signature(bound)
        except Exception as error:
            return IntegrityEvidence(
                algorithm=STRONG_SIGNATURE_ALGORITHM,
                checked=_STRONG_CHECKED,
                entry_sha256=self.sha256,
                exit_sha256=None,
                matched=False,
                differences=("strong_signature_capture_failed",),
                signature_calls=1,
                verification_error_type=_qualified_type(error),
                verification_error_message=str(error),
            )
        differences = list(
            _signature_differences(self.signature, current_signature)
        )
        if id(bound) != self.model_object_id:
            differences.insert(0, "model_object_identity")
        return IntegrityEvidence(
            algorithm=STRONG_SIGNATURE_ALGORITHM,
            checked=_STRONG_CHECKED,
            entry_sha256=self.sha256,
            exit_sha256=_signature_sha256(current_signature),
            matched=not differences,
            differences=tuple(dict.fromkeys(differences)),
            signature_calls=1,
        )

    def require_verified(
        self,
        model: nn.Module | None = None,
        *,
        body_exception: BaseException | None = None,
    ) -> IntegrityEvidence:
        evidence = self.verify(model)
        if not evidence.matched:
            raise ModelIntegrityError(
                "strong model integrity verification failed: "
                + ", ".join(evidence.differences),
                evidence=evidence,
                body_exception=body_exception,
            )
        return evidence


@dataclass(frozen=True)
class CheapModelSnapshot:
    _model_ref: weakref.ReferenceType[nn.Module]
    model_object_id: int
    signature: tuple[tuple[Any, ...], ...]
    sha256: str

    @classmethod
    def capture(cls, model: nn.Module) -> "CheapModelSnapshot":
        signature = model_live_signature(model, include_lifecycle=False)
        return cls(weakref.ref(model), id(model), signature, _signature_sha256(signature))

    def verify(self, model: nn.Module | None = None) -> IntegrityEvidence:
        bound = self._model_ref() if model is None else model
        if bound is None:
            return IntegrityEvidence(
                algorithm=CHEAP_MODEL_GUARD_ALGORITHM,
                checked=_CHEAP_MODEL_CHECKED,
                entry_sha256=self.sha256,
                exit_sha256=None,
                matched=False,
                differences=("model_reference_expired",),
                signature_calls=0,
            )
        try:
            current_signature = model_live_signature(
                bound, include_lifecycle=False
            )
        except Exception as error:
            return IntegrityEvidence(
                algorithm=CHEAP_MODEL_GUARD_ALGORITHM,
                checked=_CHEAP_MODEL_CHECKED,
                entry_sha256=self.sha256,
                exit_sha256=None,
                matched=False,
                differences=("cheap_signature_capture_failed",),
                signature_calls=0,
                verification_error_type=_qualified_type(error),
                verification_error_message=str(error),
            )
        current_sha256 = _signature_sha256(current_signature)
        differences = list(_signature_differences(self.signature, current_signature))
        if id(bound) != self.model_object_id:
            differences.insert(0, "model_object_identity")
        return IntegrityEvidence(
            algorithm=CHEAP_MODEL_GUARD_ALGORITHM,
            checked=_CHEAP_MODEL_CHECKED,
            entry_sha256=self.sha256,
            exit_sha256=current_sha256,
            matched=not differences,
            differences=tuple(dict.fromkeys(differences)),
            signature_calls=0,
        )


@dataclass(frozen=True)
class CheapTensorSnapshot:
    _tensor_ref: Callable[[], torch.Tensor | None]
    signature: tuple[Any, ...]
    sha256: str

    @classmethod
    def capture(cls, tensor: torch.Tensor) -> "CheapTensorSnapshot":
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        signature = tensor_live_signature(tensor)
        try:
            tensor_ref: Callable[[], torch.Tensor | None] = weakref.ref(tensor)
        except TypeError:
            tensor_ref = lambda: tensor
        return cls(tensor_ref, signature, _signature_sha256((signature,)))

    def verify(self, tensor: torch.Tensor | None = None) -> IntegrityEvidence:
        bound = self._tensor_ref() if tensor is None else tensor
        if bound is None:
            return IntegrityEvidence(
                algorithm=CHEAP_TENSOR_GUARD_ALGORITHM,
                checked=_CHEAP_TENSOR_CHECKED,
                entry_sha256=self.sha256,
                exit_sha256=None,
                matched=False,
                differences=("tensor_reference_expired",),
                signature_calls=0,
            )
        try:
            current = tensor_live_signature(bound)
        except Exception as error:
            return IntegrityEvidence(
                algorithm=CHEAP_TENSOR_GUARD_ALGORITHM,
                checked=_CHEAP_TENSOR_CHECKED,
                entry_sha256=self.sha256,
                exit_sha256=None,
                matched=False,
                differences=("cheap_tensor_signature_capture_failed",),
                signature_calls=0,
                verification_error_type=_qualified_type(error),
                verification_error_message=str(error),
            )
        current_sha256 = _signature_sha256((current,))
        matched = current == self.signature
        return IntegrityEvidence(
            algorithm=CHEAP_TENSOR_GUARD_ALGORITHM,
            checked=_CHEAP_TENSOR_CHECKED,
            entry_sha256=self.sha256,
            exit_sha256=current_sha256,
            matched=matched,
            differences=() if matched else ("tensor_live_state",),
            signature_calls=0,
        )


class CheapModelGuard(AbstractContextManager["CheapModelGuard"]):
    """Context manager that enforces a content-free model guard."""

    def __init__(self, model: nn.Module, *, context: str = "model operation") -> None:
        self.model = model
        self.context = context
        self.snapshot: CheapModelSnapshot | None = None
        self.evidence: IntegrityEvidence | None = None

    @classmethod
    def capture(
        cls, model: nn.Module, *, context: str = "model operation"
    ) -> "CheapModelGuard":
        guard = cls(model, context=context)
        guard.snapshot = CheapModelSnapshot.capture(model)
        return guard

    def verify(self, model: nn.Module | None = None) -> IntegrityEvidence:
        if self.snapshot is None:
            raise RuntimeError("cheap model guard has not been captured")
        self.evidence = self.snapshot.verify(self.model if model is None else model)
        return self.evidence

    def __enter__(self) -> "CheapModelGuard":
        if self.snapshot is None:
            self.snapshot = CheapModelSnapshot.capture(self.model)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        evidence = self.verify()
        if not evidence.matched:
            if exc is not None:
                evidence = replace(
                    evidence, body_exception_type=_qualified_type(exc)
                )
                self.evidence = evidence
            error = ModelIntegrityError(
                f"{self.context} changed cheap model live state: "
                + ", ".join(evidence.differences),
                evidence=evidence,
                body_exception=exc,
            )
            if exc is not None:
                raise error from exc
            raise error
        return False


class CheapTensorGuard(AbstractContextManager["CheapTensorGuard"]):
    """Context manager for one current hot-path tensor."""

    def __init__(
        self, tensor: torch.Tensor, *, context: str = "tensor operation"
    ) -> None:
        self.tensor = tensor
        self.context = context
        self.snapshot: CheapTensorSnapshot | None = None
        self.evidence: IntegrityEvidence | None = None

    @classmethod
    def capture(
        cls, tensor: torch.Tensor, *, context: str = "tensor operation"
    ) -> "CheapTensorGuard":
        guard = cls(tensor, context=context)
        guard.snapshot = CheapTensorSnapshot.capture(tensor)
        return guard

    def verify(self, tensor: torch.Tensor | None = None) -> IntegrityEvidence:
        if self.snapshot is None:
            raise RuntimeError("cheap tensor guard has not been captured")
        self.evidence = self.snapshot.verify(self.tensor if tensor is None else tensor)
        return self.evidence

    def __enter__(self) -> "CheapTensorGuard":
        if self.snapshot is None:
            self.snapshot = CheapTensorSnapshot.capture(self.tensor)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        evidence = self.verify()
        if not evidence.matched:
            if exc is not None:
                evidence = replace(
                    evidence, body_exception_type=_qualified_type(exc)
                )
                self.evidence = evidence
            error = ModelIntegrityError(
                f"{self.context} changed guarded tensor live state",
                evidence=evidence,
                body_exception=exc,
            )
            if exc is not None:
                raise error from exc
            raise error
        return False


class ModelIntegrityBoundary(AbstractContextManager["ModelIntegrityBoundary"]):
    """Strong entry/exit boundary for exactly one execution unit."""

    def __init__(
        self,
        model: nn.Module,
        *,
        reference_snapshot: StrongModelSnapshot | None = None,
    ) -> None:
        self.model = model
        self.reference_snapshot = reference_snapshot
        self.entry_snapshot: StrongModelSnapshot | None = None
        self.entry_evidence: IntegrityEvidence | None = None
        self.restoration_evidence: RestorationEvidence | None = None
        self.evidence: IntegrityEvidence | None = None

    def __enter__(self) -> "ModelIntegrityBoundary":
        self.entry_snapshot = StrongModelSnapshot.capture(self.model)
        if self.reference_snapshot is not None:
            self.entry_evidence = self.reference_snapshot.compare_snapshot(
                self.entry_snapshot, signature_calls=1
            )
            if not self.entry_evidence.matched:
                raise ModelIntegrityError(
                    "execution-unit entry does not match the canonical model snapshot",
                    evidence=self.entry_evidence,
                )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.entry_snapshot is None:
            raise RuntimeError("model integrity boundary was not entered")
        try:
            restoration = self.entry_snapshot.restore_lifecycle(self.model)
        except Exception as restoration_error:
            restoration = RestorationEvidence(
                checked=("execution_unit_lifecycle",),
                changes_detected=True,
                restored=False,
                differences=("lifecycle_restoration_failed",),
                errors=(
                    f"{_qualified_type(restoration_error)}: "
                    f"{restoration_error}",
                ),
            )
        self.restoration_evidence = restoration
        evidence = self.entry_snapshot.verify(self.model)
        # ``verify`` reports the one exit signing call.  Boundary evidence also
        # accounts for the entry capture, while the cached model-load snapshot
        # remains a separate process-level call.
        evidence = replace(evidence, signature_calls=2)
        if restoration.changes_detected or not restoration.restored:
            lifecycle_differences = tuple(
                f"lifecycle:{item}" for item in restoration.differences
            )
            if not restoration.restored:
                lifecycle_differences += ("lifecycle_restoration_failed",)
            evidence = replace(
                evidence,
                matched=False,
                differences=tuple(
                    dict.fromkeys(
                        (*evidence.differences, *lifecycle_differences)
                    )
                ),
            )
        if exc is not None:
            evidence = replace(evidence, body_exception_type=_qualified_type(exc))
        self.evidence = evidence
        if not evidence.matched:
            error = ModelIntegrityError(
                "execution-unit exit model integrity verification failed: "
                + ", ".join(evidence.differences),
                evidence=evidence,
                body_exception=exc,
                restoration_evidence=restoration,
            )
            if exc is not None:
                raise error from exc
            raise error
        return False


def capture(model: nn.Module) -> StrongModelSnapshot:
    """Short public spelling used by runner lifecycle integration."""

    return StrongModelSnapshot.capture(model)


def verify(
    snapshot: StrongModelSnapshot, model: nn.Module | None = None
) -> IntegrityEvidence:
    """Return structured evidence without raising on a state mismatch."""

    if not isinstance(snapshot, StrongModelSnapshot):
        raise TypeError("snapshot must be a StrongModelSnapshot")
    return snapshot.verify(model)


__all__ = [
    "CHEAP_MODEL_GUARD_ALGORITHM",
    "CHEAP_TENSOR_GUARD_ALGORITHM",
    "STRONG_SIGNATURE_ALGORITHM",
    "CheapModelGuard",
    "CheapModelSnapshot",
    "CheapTensorGuard",
    "CheapTensorSnapshot",
    "IntegrityEvidence",
    "ModelIntegrityBoundary",
    "ModelIntegrityError",
    "RestorationEvidence",
    "StrongModelSnapshot",
    "capture",
    "model_live_signature",
    "strong_model_signature",
    "tensor_live_signature",
    "verify",
]
