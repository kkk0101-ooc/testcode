"""Stage C execution orchestration.

The runner deliberately separates three concerns:

* :mod:`vitnns.stage_c_plan` owns the frozen run specification, expansion and
  conservative preflight;
* :mod:`vitnns.stage_c_artifacts` owns immutable attempt directories and their
  manifest/status commit protocol; and
* a :class:`StageCBackend` owns scientific computation.

That split makes the expensive production backend replaceable by a tiny CPU
backend in lifecycle tests without weakening resume or integrity checks.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import stat
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .stage_c_resources import (
    ArtifactByteMeasurement,
    ArtifactFileBytes,
    CallCounter,
    CallCounts,
    ResourceMeter,
    ResourceRecord,
    PhaseTimer,
    measurement_environment,
)


STAGE_C_RUNNER_SCHEMA_VERSION = 1
BATCH_STATUS_FILENAME = "batch_status.json"
FAILURE_TABLE_FILENAME = "failure_table.json"
EXECUTION_PLAN_FILENAME = "execution_plan.json"
PREFLIGHT_FILENAME = "preflight.json"
L0_LONG_TABLE_FILENAME = "l0_long_table.json"

UNIT_SUCCEEDED = "succeeded"
UNIT_FAILED = "failed"
UNIT_SKIPPED = "skipped_valid_complete"
UNIT_MISSING = "missing"
UNIT_INCOMPLETE = "incomplete"
UNIT_TAMPERED = "tampered"

_TERMINAL_STATES = frozenset({UNIT_SUCCEEDED, UNIT_FAILED, UNIT_SKIPPED})
_SHA256 = frozenset("0123456789abcdef")
_SIGNED_METHOD_IDS = frozenset({
    "conditional_fixed_clean_attention_head_direction_value_path_ig",
    "matched_fixed_clean_attention_patch_token_contribution",
    "full_model_class_logit_ig",
})


def _method_sign_semantics(method_id: str) -> str:
    return "signed" if method_id in _SIGNED_METHOD_IDS else "nonnegative"


class StageCRunnerError(RuntimeError):
    """Base error for Stage C execution orchestration."""


class StageCPreflightError(StageCRunnerError):
    """Execution was requested without a passing frozen preflight."""


class StageCBackendError(StageCRunnerError):
    """A backend returned data that violates the execution-unit contract."""


_SCIENTIFIC_INVALIDITY_CODES = frozenset({
    "empty_valid_region",
    "shape_mismatch",
    "non_finite",
    "degenerate_direction",
    "degenerate_target_delta",
    "missing_artifact",
    "audit_repeatability_failed",
    "endpoint_parity_failed",
    "ig_conservation_failed",
    "matched_token_conservation_failed",
})


@dataclass(frozen=True, slots=True)
class StageCPartialEvidence:
    """Ordered, stage-neutral evidence completed before invalidity."""

    l0_scalars: Mapping[str, Any] | None = None
    metric_ready_map: Any | None = None
    fidelity_curves: Any | None = None
    audit_evidence: Mapping[str, Any] | None = None


class StageCScientificInvalidity(StageCBackendError):
    """Nominal closed scientific failure emitted by the production runner."""

    def __init__(
        self, code: str, *, phase: str,
        l0_scalars: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        partial_evidence: StageCPartialEvidence | None = None,
        message: str | None = None,
    ) -> None:
        if code not in _SCIENTIFIC_INVALIDITY_CODES:
            raise ValueError("scientific invalidity code is outside the closed set")
        if not isinstance(phase, str) or not phase:
            raise TypeError("scientific invalidity phase must be non-empty")
        if partial_evidence is not None and not isinstance(
            partial_evidence, StageCPartialEvidence
        ):
            raise TypeError("partial_evidence must be StageCPartialEvidence")
        self.failure_code = code
        self.phase = phase
        self.l0_scalars = {} if l0_scalars is None else dict(l0_scalars)
        self.details = None if details is None else dict(details)
        self.partial_evidence = partial_evidence
        super().__init__(
            f"Stage C scientific invalidity: {code}" if message is None else message
        )


class StageCBatchValidationError(StageCRunnerError):
    """A batch cannot be aggregated because its integrity is invalid."""


def _jsonable(value: Any, context: str = "value") -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item(), context)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{context} keys must be non-empty strings")
            result[key] = _jsonable(item, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict(), context)
    raise TypeError(f"{context} is not JSON-compatible: {type(value).__name__}")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    plain = _jsonable(value, context)
    if not isinstance(plain, dict):
        raise TypeError(f"{context} must be a mapping")
    return plain


def _provenance_perturbation_parameters(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Separate each perturbation identifier from its parameter payload."""

    parameters: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(items):
        item = _mapping(raw, f"perturbations.items[{index}]")
        perturbation_id = item.pop("id", None)
        if not isinstance(perturbation_id, str) or not perturbation_id:
            raise StageCBackendError(
                f"perturbations.items[{index}].id must be a non-empty string"
            )
        if perturbation_id in parameters:
            raise StageCBackendError(
                f"duplicate perturbation id: {perturbation_id}"
            )
        parameters[perturbation_id] = item
    return parameters


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _unit_identity(unit: Any) -> dict[str, Any]:
    key = _attribute(unit, "key", unit)
    identity = _mapping(key, "execution unit key")
    required = {
        "schema_version", "sample_id", "method_id", "config_id", "block_index",
        "head_index", "target_ref", "canonical_digest",
    }
    if set(identity) != required:
        raise StageCBackendError(
            "execution unit key has the wrong fields: "
            f"expected {sorted(required)}, got {sorted(identity)}"
        )
    digest = identity["canonical_digest"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in _SHA256 for char in digest)
    ):
        raise StageCBackendError(
            "execution unit canonical_digest must be lowercase SHA-256"
        )
    try:
        from .stage_c_provenance import unit_identity_digest

        if unit_identity_digest(identity) != digest:
            raise StageCBackendError(
                "execution unit canonical_digest does not match its semantic fields"
            )
    except ImportError:
        pass
    return identity


def _unit_digest(unit: Any) -> str:
    return str(_unit_identity(unit)["canonical_digest"])


def _unit_audit(unit: Any) -> bool:
    value = _attribute(unit, "audit", None)
    if value is None:
        sample = _attribute(unit, "sample", {})
        value = _attribute(sample, "audit", False)
    if type(value) is not bool:
        raise StageCBackendError("execution unit audit flag must be boolean")
    return value


def _unit_line(unit: Any) -> str:
    value = _attribute(unit, "comparison_line", None)
    if hasattr(value, "value"):
        value = value.value
    if value not in {"value_path", "class_logit"}:
        raise StageCBackendError("execution unit comparison_line is invalid")
    return str(value)


def _plan_units(plan: Any) -> tuple[Any, ...]:
    units = _attribute(plan, "units", None)
    if units is None or isinstance(units, (str, bytes, Mapping)):
        raise TypeError("execution plan must expose an explicit units sequence")
    result = tuple(units)
    digests = [_unit_digest(unit) for unit in result]
    if len(digests) != len(set(digests)):
        raise StageCRunnerError("execution plan contains duplicate unit keys")
    return result


def _plan_digest(plan: Any) -> str:
    digest = _attribute(plan, "digest", None)
    if not isinstance(digest, str) or len(digest) != 64:
        raise StageCRunnerError("execution plan must expose a SHA-256 digest")
    return digest


def _plan_dict(plan: Any) -> dict[str, Any]:
    return _mapping(plan, "execution plan")


def _canonical_digest(value: Mapping[str, Any]) -> str:
    from .stage_c_provenance import canonical_sha256

    return canonical_sha256(value)


def _validated_preflight_binding(
    preflight: Mapping[str, Any], *, plan: Any | None
) -> dict[str, Any]:
    """Validate and return the dynamic provenance portion of a preflight."""

    document = _mapping(preflight, "preflight estimate")
    digest = document.get("canonical_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise StageCPreflightError(
            "preflight estimate must contain canonical_digest"
        )
    payload = {key: value for key, value in document.items() if key != "canonical_digest"}
    if _canonical_digest(payload) != digest:
        raise StageCPreflightError("preflight canonical_digest mismatch")
    plan_digest = document.get("plan_digest")
    if not isinstance(plan_digest, str) or len(plan_digest) != 64:
        raise StageCPreflightError("preflight plan_digest is invalid")
    if plan is not None and plan_digest != _plan_digest(plan):
        raise StageCPreflightError(
            "preflight plan_digest does not match the current execution plan"
        )
    return {"plan_digest": plan_digest, "canonical_digest": digest}


def _run_spec_digest(plan: Any) -> str | None:
    run_spec = _attribute(plan, "run_spec", None)
    if run_spec is not None:
        return _canonical_digest(_mapping(run_spec, "run spec"))
    return None


def _expected_provenance_bindings(
    *, plan: Any | None, preflight: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind an attempt to the active plan, run spec, and preflight."""

    preflight_binding = _validated_preflight_binding(preflight, plan=plan)
    plan_digest = (
        _plan_digest(plan) if plan is not None else preflight_binding["plan_digest"]
    )
    configuration: dict[str, Any] = {
        "execution_plan_sha256": plan_digest,
    }
    if plan is not None:
        run_spec_digest = _run_spec_digest(plan)
        if run_spec_digest is not None:
            configuration["run_spec_sha256"] = run_spec_digest
    return {
        "configuration": configuration,
        "resources": {"preflight": preflight_binding},
    }


def _runtime_git_state(
    *, plan: Any | None, backend: Any | None = None,
    observed_git_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    if observed_git_state is not None:
        return _mapping(observed_git_state, "observed_git_state")
    captured = getattr(backend, "code_state", None)
    if captured is not None:
        return _mapping(captured, "backend.code_state")
    # Generic toy plans intentionally have no repository binding.  Real plans
    # always come from the production planning module and are checked against
    # the current workspace for offline validation.
    if plan is not None and type(plan).__module__ == "vitnns.stage_c_plan":
        from .constants import REPO_ROOT
        from .stage_c_provenance import capture_git_state

        return capture_git_state(REPO_ROOT)
    return None


def _safe_output_path(value: Path | str, *, allow_missing: bool) -> Path:
    """Make a path absolute without resolving away symlink evidence."""

    path = Path(os.path.abspath(os.fspath(value)))
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if not os.path.lexists(current):
            if allow_missing:
                continue
            raise FileNotFoundError(current)
        if stat.S_ISLNK(current.lstat().st_mode):
            raise StageCRunnerError(
                f"symlink path component is forbidden: {current}"
            )
    return path


@dataclass(frozen=True)
class UnitPayload:
    """Backend output before the runner commits an immutable attempt.

    ``l0`` is embedded in one unit result record.  ``l1`` values must be
    float32 14x14 metric maps.  L2 is accepted only for registry-declared
    audit samples; L3 is a display-only leaf and is never returned by any
    quantitative reader.
    """

    l0: Mapping[str, Any]
    l1: Mapping[str, np.ndarray]
    l2: Mapping[str, Any] = field(default_factory=dict)
    l3: Mapping[str, bytes] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    lifecycle: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "l0", "l1", "l2", "l3", "provenance",
            "resources", "lifecycle",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"UnitPayload.{name} must be a mapping")
        _jsonable(self.l0, "UnitPayload.l0")
        _jsonable(self.resources, "UnitPayload.resources")
        for name, raw in self.l1.items():
            if not isinstance(name, str) or not name:
                raise TypeError("L1 names must be non-empty strings")
            value = np.asarray(raw)
            if (
                value.dtype != np.float32
                or value.shape != (14, 14)
                or not np.isfinite(value).all()
            ):
                raise StageCBackendError(
                    f"L1 {name!r} must be finite float32 with shape (14,14)"
                )
        for name, payload in self.l3.items():
            if not isinstance(name, str) or not name or not isinstance(payload, bytes):
                raise TypeError("L3 entries must map non-empty names to PNG bytes")

    @classmethod
    def coerce(cls, value: Any) -> "UnitPayload":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("backend.run must return UnitPayload or a mapping")
        allowed = {
            "l0", "l1", "l2", "l3", "provenance",
            "resources", "lifecycle",
        }
        unknown = set(value) - allowed
        if unknown:
            raise StageCBackendError(
                f"backend payload has unknown fields: {sorted(unknown)}"
            )
        return cls(
            l0=value.get("l0", {}),
            l1=value.get("l1", {}),
            l2=value.get("l2", {}),
            l3=value.get("l3", {}),
            provenance=value.get("provenance", {}),
            resources=value.get("resources", {}),
            lifecycle=value.get("lifecycle", {}),
        )


@runtime_checkable
class StageCBackend(Protocol):
    """Injectable scientific backend used by one execution unit."""

    def run(
        self, unit: Any, *, device: str, call_counter: CallCounter
    ) -> UnitPayload: ...


class ProductionStageCBackend:
    """Production adapter that composes the frozen C2--C5 primitives.

    The backend is intentionally lazy: construction performs no model or image
    I/O, so a rejected preflight cannot touch GPU state.  ``sample_loader`` is
    the explicit boundary that supplies one canonical RGB tensor and its
    registry row.  This keeps filesystem policy outside the scientific code
    and makes the exact same dispatcher usable by the CLI and C7 smoke tests.

    Fidelity, stability and residual calculations call their C5 public APIs;
    no formula is reproduced here.  Global class methods retain null block/head
    identities while their curve work consumes the explicit pairing context
    from each :class:`~vitnns.stage_c_plan.WorkItem`.
    """

    DEFAULT_FIDELITY_SEED_NAMESPACE = "stage-c-fidelity|20260821|"

    def __init__(
        self,
        *,
        sample_loader: Callable[[Any, Any], Any],
        model_loader: Callable[..., Any] | None = None,
        state_dict_path: Path | str | None = None,
        provenance_builder: Callable[..., Mapping[str, Any]] | None = None,
        config: Any = None,
        registry: Any = None,
        plan: Any = None,
        config_path: Path | str | None = None,
        code_state: Mapping[str, Any] | None = None,
        fidelity_seed_namespace: str | None = None,
        empty_cuda_cache_on_release: bool = False,
    ) -> None:
        if not callable(sample_loader):
            raise TypeError("sample_loader must be callable")
        if model_loader is not None and not callable(model_loader):
            raise TypeError("model_loader must be callable or None")
        self.sample_loader = sample_loader
        self.model_loader = model_loader
        self.state_dict_path = None if state_dict_path is None else Path(state_dict_path)
        self.provenance_builder = provenance_builder
        self.config = config
        self.registry = registry
        self.plan = plan
        self.config_path = None if config_path is None else Path(config_path)
        if code_state is None and config is not None and plan is not None:
            from .constants import REPO_ROOT
            from .stage_c_provenance import capture_git_state

            code_state = capture_git_state(REPO_ROOT)
        self.code_state = None if code_state is None else dict(code_state)
        namespace = (
            self.DEFAULT_FIDELITY_SEED_NAMESPACE
            if fidelity_seed_namespace is None else fidelity_seed_namespace
        )
        if not isinstance(namespace, str) or not namespace.endswith("|"):
            raise TypeError(
                "fidelity_seed_namespace must be a string ending in '|'"
            )
        if type(empty_cuda_cache_on_release) is not bool:
            raise TypeError(
                "empty_cuda_cache_on_release must be a boolean"
            )
        self.fidelity_seed_namespace = namespace
        self.empty_cuda_cache_on_release = empty_cuda_cache_on_release
        self._models: dict[str, Any] = {}
        self._model_load_snapshots: dict[str, Any] = {}
        self._samples: dict[tuple[str, str], dict[str, Any]] = {}
        self._perturbations: dict[tuple[str, str], Mapping[str, Any]] = {}
        self._extractions: dict[tuple[str, str], Any] = {}
        self._class_targets: dict[tuple[str, str], Any] = {}
        self._value_targets: dict[tuple[str, str, int, int], Any] = {}
        self._image_scope_lock = threading.Lock()
        self._active_image_scope: tuple[str, str] | None = None

    def release_sample(self, sample_id: str, device: str) -> None:
        """Release every sample-scoped cache entry for one image/device.

        Device-scoped models and their integrity snapshots intentionally remain
        resident.  The optional CUDA allocator cleanup is off by default and is
        never needed by CPU/synthetic callers.
        """

        if not isinstance(sample_id, str) or not sample_id:
            raise TypeError("sample_id must be a non-empty string")
        if not isinstance(device, str) or not device:
            raise TypeError("device must be a non-empty string")
        pair_key = (sample_id, device)
        for cache in (
            self._samples, self._perturbations, self._extractions,
            self._class_targets,
        ):
            cache.pop(pair_key, None)
        for key in tuple(self._value_targets):
            if key[:2] == pair_key:
                del self._value_targets[key]
        if self.empty_cuda_cache_on_release and device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    @contextmanager
    def image_scope(self, sample_id: str, *, device: str) -> Iterator["ProductionStageCBackend"]:
        """Exclusively own one image and release its caches on every exit."""

        if not isinstance(sample_id, str) or not sample_id:
            raise TypeError("sample_id must be a non-empty string")
        if not isinstance(device, str) or not device:
            raise TypeError("device must be a non-empty string")
        if not self._image_scope_lock.acquire(blocking=False):
            raise StageCBackendError(
                "ProductionStageCBackend is already owned by another image scope"
            )
        self._active_image_scope = (sample_id, device)
        try:
            yield self
        finally:
            try:
                self.release_sample(sample_id, device)
            finally:
                self._active_image_scope = None
                self._image_scope_lock.release()

    def _phase(self, name: str):
        timer = getattr(self, "_active_phase_timer", None)
        return nullcontext() if timer is None else timer.measure(name)

    def _model(self, device: str) -> Any:
        if device not in self._models:
            loader = self.model_loader
            if loader is None:
                from .modeling import load_canonical_model

                loader = load_canonical_model
            kwargs: dict[str, Any] = {
                "device": device, "verify_hash": True,
                "unfused_attention": True,
            }
            if self.state_dict_path is not None:
                kwargs["state_dict_path"] = self.state_dict_path
            self._models[device] = loader(**kwargs)
            from .modeling import canonical_preprocess_spec

            specification = canonical_preprocess_spec(self._models[device])
            if (
                specification.input_size != (3, 224, 224)
                or specification.resize_shorter_side != 248
                or specification.center_crop != (224, 224)
                or specification.interpolation != "bicubic"
                or specification.mean != (0.5, 0.5, 0.5)
                or specification.std != (0.5, 0.5, 0.5)
            ):
                raise StageCBackendError("canonical preprocessing contract drifted")
        if device not in self._model_load_snapshots:
            from .stage_c_integrity import StrongModelSnapshot

            with self._phase("integrity_checking"):
                self._model_load_snapshots[device] = StrongModelSnapshot.capture(
                    self._models[device]
                )
        return self._models[device]

    @staticmethod
    def _coerce_sample(value: Any, device: str) -> dict[str, Any]:
        if isinstance(value, Mapping):
            result = dict(value)
        elif isinstance(value, tuple) and len(value) in {2, 3}:
            result = {"canonical_rgb": value[0], "normalized_input": value[1]}
            if len(value) == 3:
                result["registry_row"] = value[2]
        else:
            raise TypeError(
                "sample_loader must return a mapping or "
                "(canonical_rgb, normalized_input[, registry_row])"
            )
        required = {"canonical_rgb", "normalized_input"}
        missing = required - set(result)
        if missing:
            raise StageCBackendError(
                f"sample_loader result is missing: {sorted(missing)}"
            )
        import torch

        canonical = result["canonical_rgb"]
        normalized = result["normalized_input"]
        if (
            not isinstance(canonical, torch.Tensor)
            or canonical.dtype is not torch.float32
            or tuple(canonical.shape) != (1, 3, 224, 224)
            or canonical.device.type != "cpu"
        ):
            raise StageCBackendError(
                "sample_loader canonical_rgb must be CPU float32 [1,3,224,224]"
            )
        if (
            not isinstance(normalized, torch.Tensor)
            or normalized.dtype is not torch.float32
            or tuple(normalized.shape) != (1, 3, 224, 224)
        ):
            raise StageCBackendError(
                "sample_loader normalized_input must be float32 [1,3,224,224]"
            )
        if not bool(torch.isfinite(canonical).all().item()) or bool(
            ((canonical < 0.0) | (canonical > 1.0)).any().item()
        ):
            raise StageCBackendError(
                "sample_loader canonical_rgb must be finite in [0,1]"
            )
        if not bool(torch.isfinite(normalized).all().item()):
            raise StageCBackendError("sample_loader normalized_input must be finite")
        result["canonical_rgb"] = canonical.detach().clone()
        result["normalized_input"] = normalized.detach().clone().to(device)
        return result

    def _sample_context(
        self, unit: Any, device: str, model: Any
    ) -> dict[str, Any]:
        key = (_unit_identity(unit)["sample_id"], device)
        if key not in self._samples:
            sample = _attribute(unit, "sample", unit)
            loaded = self.sample_loader(sample, model)
            self._samples[key] = self._coerce_sample(loaded, device)
        return self._samples[key]

    @staticmethod
    @contextmanager
    def _count_model_forwards(model: Any, counter: CallCounter) -> Iterator[None]:
        register = getattr(model, "register_forward_pre_hook", None)
        if not callable(register):
            raise TypeError("production model must support forward pre-hooks")
        handle = register(lambda _module, _args: counter.record_forward())
        try:
            yield
        finally:
            handle.remove()

    @staticmethod
    def _candidate(unit: Any) -> tuple[int, int]:
        identity = _unit_identity(unit)
        block, head = identity["block_index"], identity["head_index"]
        if type(block) is not int or type(head) is not int:
            raise StageCBackendError("per-head method requires a real block/head unit")
        return block, head

    @staticmethod
    def _method_result_dict(result: Any) -> dict[str, Any]:
        na = getattr(result, "na", None)
        provenance = _jsonable(getattr(result, "provenance", {}), "method provenance")
        return {
            "method_id": result.method_id,
            "method_version": result.method_version,
            "comparison_line": result.comparison_line.value,
            "target_ref": result.fidelity_target,
            "status": "NA" if na is not None else "OK",
            "na": None if na is None else na.to_dict(),
            "projection": result.projection,
            "sign_semantics": result.sign_semantics.value,
            "provenance_fragment": provenance,
        }

    @staticmethod
    def _metric_array(result: Any) -> np.ndarray:
        if result.metric_map is None:
            raise StageCBackendError(
                "structured-NA method result has no L1 map; refusing to fabricate zeros"
            )
        return np.array(result.metric_map.data, dtype=np.float32, copy=True)

    def _retained_metric_array(self, result: Any, response: Any) -> np.ndarray:
        """Return the metric-ready map retained by this backend.

        Stage C keeps its historical 14x14 representation.  Stage-specific
        adapters may override this hook and reuse the registered projection
        primitive without duplicating its formula.
        """

        del response
        return self._metric_array(result)

    def _retain_repeat_clean_maps(self) -> bool:
        """Whether repeat-clean metric maps belong in L1 retention."""

        return False

    def _make_unit_payload(
        self, *, l0: Mapping[str, Any], l1: Mapping[str, np.ndarray],
        l2: Mapping[str, Any], l3: Mapping[str, bytes] | None = None,
        provenance: Mapping[str, Any] | None = None,
        resources: Mapping[str, Any] | None = None,
        lifecycle: Mapping[str, Any] | None = None,
    ) -> UnitPayload:
        """Construct a payload through a retention-policy override point."""

        return UnitPayload(
            l0=l0, l1=l1, l2=l2, l3={} if l3 is None else l3,
            provenance={} if provenance is None else provenance,
            resources={} if resources is None else resources,
            lifecycle={} if lifecycle is None else lifecycle,
        )

    @staticmethod
    def _native_array(result: Any) -> np.ndarray | None:
        if result.native_tensor is None:
            return None
        return np.array(result.native_tensor.data, dtype=np.float32, copy=True)

    @staticmethod
    def _work_items(unit: Any) -> tuple[Any, ...]:
        return tuple(_attribute(unit, "work_items", ()))

    @staticmethod
    def _record_method_backward_calls(
        call_counter: CallCounter, method_id: str, result: Any
    ) -> None:
        """Account from the frozen method's retained execution metadata.

        C2--C5 do not accept a counter parameter. Their public IG results
        retain exact attempt schedules, while Grad-CAM has exactly one
        autograd call. Forward calls are observed independently by a root
        model hook, so nested APIs cannot be double-counted.
        """

        if method_id in {
            "conditional_fixed_clean_attention_head_direction_value_path_ig",
            "full_model_class_logit_ig",
        }:
            direct_attempts = getattr(result, "attempts", None)
            if isinstance(direct_attempts, (list, tuple)):
                attempts = direct_attempts
            else:
                provenance = getattr(result, "provenance", {})
                ig = (
                    provenance.get("ig_result", {})
                    if isinstance(provenance, Mapping) else {}
                )
                attempts = (
                    ig.get("attempts", []) if isinstance(ig, Mapping) else []
                )
            chunks = sum(
                math.ceil(int(attempt["steps"]) / 8)
                for attempt in attempts if isinstance(attempt, Mapping)
            )
            # Each quadrature chunk entails one autograd invocation.  The
            # public C2--C5 APIs do not accept a counter, so the retained,
            # validator-checked attempt schedule is the authoritative record.
            # An IG result must always contain at least one such chunk.
            if chunks <= 0:
                raise StageCBackendError(
                    "IG result is missing its retained attempt schedule"
                )
            call_counter.record_backward(chunks)
        elif method_id == "vit_grad_cam_block11_norm1":
            call_counter.record_backward(1)

    @staticmethod
    def _ig_execution_summary(result: Any) -> dict[str, Any] | None:
        from .attribution import IGResult

        if isinstance(result, IGResult):
            candidate = result.to_dict()
        else:
            provenance = getattr(result, "provenance", {})
            candidate = (
                provenance.get("ig_result")
                if isinstance(provenance, Mapping) else None
            )
        if not isinstance(candidate, Mapping):
            candidate = getattr(result, "ig_result", None)
            if isinstance(candidate, IGResult):
                candidate = candidate.to_dict()
        if not isinstance(candidate, Mapping):
            return None
        attempts = candidate.get("attempts", ())
        if not isinstance(attempts, (list, tuple)):
            raise StageCBackendError("IG execution evidence attempts are malformed")
        normalized = [_mapping(item, "IG attempt") for item in attempts]
        selected_steps = candidate.get("selected_steps", candidate.get("steps"))
        if type(selected_steps) is not int or selected_steps <= 0:
            raise StageCBackendError("IG execution evidence selected_steps is invalid")
        return {
            "selected_steps": selected_steps,
            "converged": candidate.get("converged"),
            "attempts": normalized,
            "metadata": _mapping(
                candidate.get("metadata", {}), "IG metadata"
            ),
        }

    def _run_method_observed(
        self, unit: Any, *, input_variant: str, repeat_index: int | None,
        model: Any, clean: Any, extraction: Any, target: Any,
        intervention: Any | None = None,
        target_identity: Mapping[str, Any] | None = None,
        intervention_input_sha256: str | None = None,
    ) -> Any:
        """Run one map and retain truthful call/IG evidence without patches."""

        counter = self._active_call_counter
        before = counter.snapshot()
        method_id = _unit_identity(unit)["method_id"]
        try:
            result = self._run_method(
                unit, model=model, clean=clean, extraction=extraction,
                target=target, intervention=intervention,
                target_identity=target_identity,
                intervention_input_sha256=intervention_input_sha256,
            )
            self._record_method_backward_calls(counter, method_id, result)
            after = counter.snapshot()
            ig_summary = self._ig_execution_summary(result)
            attempted_steps = (
                0 if ig_summary is None else sum(
                    int(item.get("steps", 0))
                    for item in ig_summary["attempts"]
                )
            )
            backward_calls = after.backward_calls - before.backward_calls
            self._active_map_execution_records.append({
                "method_id": method_id,
                "input_variant": input_variant,
                "repeat_index": repeat_index,
                "status": "completed",
                "ig_result": ig_summary,
                "attempted_steps_total": attempted_steps,
                "backward_invocations": backward_calls,
                "actual_calls": {
                    "accounting_unit": "batched_model_invocation",
                    "forward_calls": after.forward_calls - before.forward_calls,
                    "backward_calls": backward_calls,
                },
                "exception": None,
            })
            return result
        except BaseException as error:
            attached = getattr(error, "ig_result", None)
            if attached is not None:
                try:
                    self._record_method_backward_calls(counter, method_id, attached)
                except Exception:
                    pass
            after = counter.snapshot()
            ig_summary = (
                None if attached is None
                else self._ig_execution_summary(attached)
            )
            attempted_steps = (
                0 if ig_summary is None else sum(
                    int(item.get("steps", 0))
                    for item in ig_summary["attempts"]
                )
            )
            backward_calls = after.backward_calls - before.backward_calls
            self._active_map_execution_records.append({
                "method_id": method_id,
                "input_variant": input_variant,
                "repeat_index": repeat_index,
                "status": "failed",
                "ig_result": ig_summary,
                "attempted_steps_total": attempted_steps,
                "backward_invocations": backward_calls,
                "actual_calls": {
                    "accounting_unit": "batched_model_invocation",
                    "forward_calls": after.forward_calls - before.forward_calls,
                    "backward_calls": backward_calls,
                },
                "exception": {
                    "type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": str(error),
                },
            })
            raise

    @staticmethod
    def _needs(unit: Any, kind: str, variant: str | None = None) -> bool:
        for item in ProductionStageCBackend._work_items(unit):
            if _attribute(item, "kind") != kind:
                continue
            if variant is None or _attribute(item, "input_variant") == variant:
                return True
        return False

    def _run_method(
        self,
        unit: Any,
        *,
        model: Any,
        clean: Any,
        extraction: Any,
        target: Any,
        intervention: Any | None = None,
        target_identity: Mapping[str, Any] | None = None,
        intervention_input_sha256: str | None = None,
    ) -> Any:
        identity = _unit_identity(unit)
        method_id = identity["method_id"]
        inputs = clean if intervention is None else intervention
        context = {}
        if target_identity is not None:
            context["fidelity_target_identity"] = dict(target_identity)
        if intervention_input_sha256 is not None:
            context["intervention_input_sha256"] = intervention_input_sha256
        clean_sha256 = getattr(target, "clean_input_sha256", None)
        if method_id == "direct_raw_cls_attention_per_head":
            if intervention is not None:
                from .extractor import NativeViTExtractor
                extraction = NativeViTExtractor(model).extract(inputs)
            from .stage_c_attention import direct_raw_cls_attention_per_head

            block, head = self._candidate(unit)
            return direct_raw_cls_attention_per_head(
                extraction, block_index=block, head_index=head, provenance=context
            )
        if method_id == "direct_raw_cls_attention_last_block_head_mean":
            from .stage_c_attention import (
                class_attention_maps_at_intervention,
                direct_raw_cls_attention_last_block_head_mean,
            )
            if intervention is None:
                return direct_raw_cls_attention_last_block_head_mean(
                    extraction, provenance=context
                )
            return class_attention_maps_at_intervention(
                __import__("vitnns.extractor", fromlist=["NativeViTExtractor"]).NativeViTExtractor(model),
                inputs, fidelity_target_identity=target_identity,
                provenance=context,
            )[0]
        if method_id == "attention_rollout_mean_head_residual_all_blocks":
            from .stage_c_attention import (
                attention_rollout_mean_head_residual_all_blocks,
                class_attention_maps_at_intervention,
            )
            if intervention is None:
                return attention_rollout_mean_head_residual_all_blocks(
                    extraction, provenance=context
                )
            return class_attention_maps_at_intervention(
                __import__("vitnns.extractor", fromlist=["NativeViTExtractor"]).NativeViTExtractor(model),
                inputs, fidelity_target_identity=target_identity,
                provenance=context,
            )[1]
        if method_id == "vit_grad_cam_block11_norm1":
            from .stage_c_class_logit import (
                vit_grad_cam_block11_norm1,
                vit_grad_cam_block11_norm1_at_intervention,
            )
            return (
                vit_grad_cam_block11_norm1(model, clean, target, provenance=context)
                if intervention is None else
                vit_grad_cam_block11_norm1_at_intervention(
                    model, clean, inputs, target, provenance=context,
                    intervention_input_sha256=intervention_input_sha256,
                )
            )
        if method_id == "full_model_class_logit_ig":
            from .stage_c_class_logit import (
                full_model_class_logit_ig,
                full_model_class_logit_ig_at_intervention,
            )
            return (
                full_model_class_logit_ig(model, clean, target, provenance=context)
                if intervention is None else
                full_model_class_logit_ig_at_intervention(
                    model, clean, inputs, target, provenance=context,
                    intervention_input_sha256=intervention_input_sha256,
                )
            )
        if method_id == "conditional_fixed_clean_attention_head_direction_value_path_ig":
            from .stage_c_value_path import (
                conditional_fixed_clean_attention_head_direction_value_path_ig,
                conditional_fixed_clean_attention_head_direction_value_path_ig_at_intervention,
            )
            return (
                conditional_fixed_clean_attention_head_direction_value_path_ig(
                    model, clean, target, provenance=context
                ) if intervention is None else
                conditional_fixed_clean_attention_head_direction_value_path_ig_at_intervention(
                    model, clean, inputs, target, provenance=context,
                    clean_input_sha256=clean_sha256,
                    intervention_input_sha256=intervention_input_sha256,
                )
            )
        if method_id == "matched_fixed_clean_attention_patch_token_contribution":
            from .stage_c_value_path import (
                matched_fixed_clean_attention_patch_token_contribution,
                matched_fixed_clean_attention_patch_token_contribution_at_intervention,
            )
            return (
                matched_fixed_clean_attention_patch_token_contribution(
                    target, provenance=context
                ) if intervention is None else
                matched_fixed_clean_attention_patch_token_contribution_at_intervention(
                    model, clean, inputs, target, provenance=context,
                    clean_input_sha256=clean_sha256,
                    intervention_input_sha256=intervention_input_sha256,
                )
            )
        raise StageCBackendError(f"unregistered Stage C method: {method_id}")

    def run(
        self, unit: Any, *, device: str, call_counter: CallCounter
    ) -> UnitPayload:
        from .stage_c_integrity import ModelIntegrityBoundary

        model = self._model(device)
        if hasattr(self, "_active_map_execution_records"):
            raise StageCBackendError(
                "ProductionStageCBackend does not support concurrent execution"
            )
        self._active_map_execution_records: list[dict[str, Any]] = []
        self._active_call_counter = call_counter
        boundary = ModelIntegrityBoundary(
            model, reference_snapshot=self._model_load_snapshots[device]
        )
        try:
            try:
                with self._phase("integrity_checking"):
                    boundary.__enter__()
                try:
                    with self._count_model_forwards(model, call_counter):
                        payload = self._run_composed(
                            unit, device=device, call_counter=call_counter
                        )
                except BaseException as body_error:
                    with self._phase("integrity_checking"):
                        boundary.__exit__(
                            type(body_error), body_error, body_error.__traceback__
                        )
                    raise
                else:
                    with self._phase("integrity_checking"):
                        boundary.__exit__(None, None, None)
            except BaseException as error:
                evidence = getattr(error, "evidence", boundary.evidence)
                observability = {
                    "map_execution_records": list(
                        self._active_map_execution_records
                    ),
                    "lifecycle": {
                        "integrity": (
                            None if evidence is None else evidence.to_dict()
                        ),
                        "restoration": (
                            None
                            if boundary.restoration_evidence is None
                            else boundary.restoration_evidence.to_dict()
                        ),
                    },
                }
                setattr(error, "stage_c_execution_observability", observability)
                raise

            if boundary.evidence is None or not boundary.evidence.matched:
                raise StageCBackendError(
                    "execution-unit integrity boundary produced no success evidence"
                )
            l0 = dict(payload.l0)
            if "map_execution_records" in l0 or "lifecycle" in l0:
                raise StageCBackendError(
                    "backend payload attempted to overwrite runner observability"
                )
            l0["map_execution_records"] = list(
                self._active_map_execution_records
            )
            l0["lifecycle"] = {
                "integrity": boundary.evidence.to_dict(),
                "restoration": (
                    None
                    if boundary.restoration_evidence is None
                    else boundary.restoration_evidence.to_dict()
                ),
            }
            return self._make_unit_payload(
                l0=l0, l1=payload.l1, l2=payload.l2, l3=payload.l3,
                provenance=payload.provenance, resources=payload.resources,
                lifecycle=l0["lifecycle"],
            )
        finally:
            del self._active_map_execution_records
            del self._active_call_counter

    def run_with_phase_timer(
        self, unit: Any, *, device: str, call_counter: CallCounter,
        phase_timer: PhaseTimer,
    ) -> UnitPayload:
        """Runner-only adapter that exposes the unit phase timer explicitly."""

        if hasattr(self, "_active_phase_timer"):
            raise StageCBackendError(
                "ProductionStageCBackend does not support concurrent timing"
            )
        self._active_phase_timer = phase_timer
        try:
            with phase_timer.measure("attribution_or_map_compute"):
                return self.run(unit, device=device, call_counter=call_counter)
        finally:
            del self._active_phase_timer

    def _run_composed(
        self, unit: Any, *, device: str, call_counter: CallCounter
    ) -> UnitPayload:
        import torch
        from .extractor import NativeViTExtractor
        from .stage_c_class_logit import (
            resolve_clean_class_logit_target,
            resolve_natural_prediction_observation,
        )
        from .stage_c_fidelity import (
            ClassLogitScalarAdapter, CurveDirection, FidelityScale,
            RankingKind, ValuePathScalarAdapter, aggregate_five_curves,
            build_intervention_schedule, build_method_intervention_schedule,
            center_distance_ranking, evaluate_fidelity_curve,
            global_random_ranking, method_unit_masses, resolve_target_response,
        )
        from .stage_c_perturbations import (
            PERTURBATION_IDS, generate_stage_c_perturbations,
        )
        from .stage_c_stability import (
            aligned_map_result, evaluate_map_stability,
            prediction_observation_from_dict,
        )
        from .stage_c_value_path import (
            resolve_fixed_clean_attention_head_direction,
        )

        model = self._model(device)
        context = self._sample_context(unit, device, model)
        canonical = context["canonical_rgb"]
        clean = context["normalized_input"]
        row = context.get("registry_row", _attribute(unit, "sample", {}))
        truth = _attribute(row, "imagenet1k_index", None)
        identity = _unit_identity(unit)
        extraction_key = (identity["sample_id"], device)
        if extraction_key not in self._extractions:
            self._extractions[extraction_key] = NativeViTExtractor(model).extract(clean)
        extraction = self._extractions[extraction_key]
        line = _unit_line(unit)

        needs_prediction = any(
            _attribute(item, "kind") == "map"
            and _attribute(item, "input_variant") in PERTURBATION_IDS
            for item in self._work_items(unit)
        )
        if identity["target_ref"] is not None or needs_prediction:
            if extraction_key not in self._class_targets:
                self._class_targets[extraction_key] = resolve_clean_class_logit_target(
                    model, clean, official_truth_index=truth
                )
            class_target = self._class_targets[extraction_key]
        else:
            class_target = None
        if identity["target_ref"] is None:
            # The matched per-head raw-attention reference is deliberately
            # target-less.  Resolving a Value-path direction here would both
            # waste four forwards and invent scientific state the registry
            # explicitly says this method does not own.
            target = None
            evaluator = None
        elif line == "class_logit":
            target = class_target
            evaluator = ClassLogitScalarAdapter(model, clean, target)
        else:
            block, head = self._candidate(unit)
            target_key = (identity["sample_id"], device, block, head)
            if target_key not in self._value_targets:
                self._value_targets[target_key] = (
                    resolve_fixed_clean_attention_head_direction(
                        model, clean, block, head
                    )
                )
            target = self._value_targets[target_key]
            evaluator = ValuePathScalarAdapter(model, clean, target)
        baseline = torch.zeros_like(clean)
        curve_items = [
            item for item in self._work_items(unit)
            if _attribute(item, "kind") == "fidelity_curve_point"
        ]
        if identity["target_ref"] is None:
            response = None
            target_identity = None
        else:
            target_identity = evaluator.target_identity
            response = (
                resolve_target_response(evaluator, clean, baseline)
                if curve_items else None
            )
        clean_map_item = next((
            item for item in self._work_items(unit)
            if _attribute(item, "kind") == "map"
            and _attribute(item, "input_variant") == "clean"
        ), None)
        clean_repeat_index = (
            None if clean_map_item is None
            else _attribute(clean_map_item, "repeat_index")
        )
        clean_result = self._run_method_observed(
            unit, input_variant="clean", repeat_index=clean_repeat_index, model=model,
            clean=clean, extraction=extraction, target=target,
            target_identity=target_identity,
        )

        l0: dict[str, Any] = self._method_result_dict(clean_result)
        l0["sample_id"] = identity["sample_id"]
        l0["prediction"] = (
            {
                "clean_top1_index": class_target.class_index,
                "official_truth_index": class_target.official_truth_index,
                "clean_top1_correct": class_target.clean_top1_correct,
            }
            if class_target is not None
            else {
                "clean_top1_index": None,
                "official_truth_index": truth,
                "clean_top1_correct": None,
                "status": "not_requested_without_stability",
            }
        )
        if clean_result.na is not None:
            reason = getattr(clean_result.na, "reason", None)
            code = getattr(reason, "value", reason)
            if code not in _SCIENTIFIC_INVALIDITY_CODES:
                raise StageCBackendError(
                    "method returned an unregistered structured NA"
                )
            raise StageCScientificInvalidity(
                code, phase="clean_map", l0_scalars=l0,
                details={"structured_na": clean_result.na.to_dict()},
                partial_evidence=StageCPartialEvidence(l0_scalars=l0),
                message=(
                    "method returned an invalidating structured NA; retain as a "
                    "failed terminal unit instead of fabricating an L1 map"
                ),
            )
        l1 = {"clean": self._retained_metric_array(clean_result, response)}
        l2: dict[str, Any] = {}
        if _unit_audit(unit):
            native = self._native_array(clean_result)
            if native is not None:
                l2["clean_native"] = native

        needs_stability = any(
            _attribute(item, "kind") == "map"
            and _attribute(item, "input_variant") in PERTURBATION_IDS
            for item in self._work_items(unit)
        )
        perturbation_key = (identity["sample_id"], device)
        if needs_stability and perturbation_key not in self._perturbations:
            self._perturbations[perturbation_key] = generate_stage_c_perturbations(
                canonical,
                selection_order=int(_attribute(row, "selection_order", 0)),
            )
        bundle = self._perturbations.get(perturbation_key, {})
        stability_records: list[dict[str, Any]] = []
        if target_identity is None and needs_stability:
            # The matched per-head raw-attention method has deliberately no
            # scalar target.  Its perturbation work is nevertheless required:
            # retain each recomputed map, but never feed it into the targeted
            # C5 stability evaluator or label it as fidelity.
            from .stage_c_maps import (
                SignSemantics, SignView, patch_to_pixel_mass,
                pixel_to_patch_mass,
            )
            from .stage_c_metrics import (
                bidirectional_kl, hotspot_jaccard_top_10_percent,
                hotspot_jaccard_top_20_percent,
                jensen_shannon_divergence, pearson, spearman,
            )
            from .stage_c_perturbations import inverse_align_pixel_map

            clean_prediction = prediction_observation_from_dict(
                resolve_natural_prediction_observation(
                    model, clean, clean, class_target
                )
            )
            structural_records: list[dict[str, Any]] = []
            for perturbation_id in PERTURBATION_IDS:
                if not self._needs(unit, "map", perturbation_id):
                    continue
                perturbation = bundle[perturbation_id]
                intervention = perturbation.normalized_tensor.to(clean.device)
                perturbed = self._run_method_observed(
                    unit, input_variant=perturbation_id, repeat_index=None,
                    model=model, clean=clean, extraction=extraction,
                    target=target, intervention=intervention,
                    target_identity=None,
                    intervention_input_sha256=(
                        perturbation.normalized_tensor_sha256
                    ),
                )
                l1[perturbation_id] = self._metric_array(perturbed)
                aligned = pixel_to_patch_mass(
                    inverse_align_pixel_map(
                        patch_to_pixel_mass(perturbed.metric_map),
                        perturbation_id,
                    )
                )
                mask = perturbation.fully_valid_patch_mask
                metric_kwargs = {
                    "mask": mask, "view": SignView.POSITIVE,
                    "left_semantics": SignSemantics.NONNEGATIVE,
                    "right_semantics": SignSemantics.NONNEGATIVE,
                }
                perturbed_prediction = prediction_observation_from_dict(
                    resolve_natural_prediction_observation(
                        model, clean, intervention, class_target
                    )
                )
                kl = bidirectional_kl(
                    clean_result.metric_map, aligned, **metric_kwargs
                )
                structural_records.append({
                    "perturbation_id": perturbation_id,
                    "comparison_label": (
                        "spatial_structure_similarity_not_same_target_fidelity"
                    ),
                    "target_ref": None,
                    "normalized_input_sha256": (
                        perturbation.normalized_tensor_sha256
                    ),
                    "clean_natural_prediction": clean_prediction.to_dict(),
                    "perturbed_natural_prediction": (
                        perturbed_prediction.to_dict()
                    ),
                    "prediction_unchanged": (
                        clean_prediction.class_index
                        == perturbed_prediction.class_index
                    ),
                    "prediction_flipped": (
                        clean_prediction.class_index
                        != perturbed_prediction.class_index
                    ),
                    "metrics": {
                        "pearson": pearson(
                            clean_result.metric_map, aligned, **metric_kwargs
                        ).to_dict(),
                        "spearman": spearman(
                            clean_result.metric_map, aligned, **metric_kwargs
                        ).to_dict(),
                        "jensen_shannon_divergence": (
                            jensen_shannon_divergence(
                                clean_result.metric_map, aligned,
                                **metric_kwargs,
                            ).to_dict()
                        ),
                        "kl": kl.to_dict(),
                        "hotspot_jaccard_top_10_percent": (
                            hotspot_jaccard_top_10_percent(
                                clean_result.metric_map, aligned,
                                **metric_kwargs,
                            ).to_dict()
                        ),
                        "hotspot_jaccard_top_20_percent": (
                            hotspot_jaccard_top_20_percent(
                                clean_result.metric_map, aligned,
                                **metric_kwargs,
                            ).to_dict()
                        ),
                    },
                })
                if _unit_audit(unit):
                    native = self._native_array(perturbed)
                    if native is not None:
                        l2[f"{perturbation_id}_native"] = native
            l0["structural_perturbations"] = structural_records
        elif target_identity is not None and needs_stability:
            clean_prediction = prediction_observation_from_dict(
                resolve_natural_prediction_observation(model, clean, clean, class_target)
            )
            for perturbation_id in PERTURBATION_IDS:
                if not self._needs(unit, "map", perturbation_id):
                    continue
                perturbation = bundle[perturbation_id]
                intervention = perturbation.normalized_tensor.to(clean.device)
                perturbed = self._run_method_observed(
                    unit, input_variant=perturbation_id, repeat_index=None,
                    model=model, clean=clean, extraction=extraction,
                    target=target, intervention=intervention,
                    target_identity=target_identity,
                    intervention_input_sha256=(
                        perturbation.normalized_tensor_sha256
                    ),
                )
                l1[perturbation_id] = self._metric_array(perturbed)
                if target_identity is not None:
                    aligned = aligned_map_result(
                        clean_result, target_identity=target_identity,
                        perturbation=perturbation,
                        perturbed_method_result=perturbed,
                    )
                    perturbed_prediction = prediction_observation_from_dict(
                        resolve_natural_prediction_observation(
                            model, clean, intervention, class_target
                        )
                    )
                    stability_records.append(
                        evaluate_map_stability(
                            clean_result, aligned, target_identity=target_identity,
                            clean_prediction=clean_prediction,
                            perturbed_natural_prediction=perturbed_prediction,
                        ).to_dict()
                    )
                    if identity["method_id"] == (
                        "conditional_fixed_clean_attention_head_direction_value_path_ig"
                    ):
                        from .stage_c_maps import signed_rgb_channel_sum
                        from .stage_c_perturbations import inverse_align_pixel_map
                        from .stage_c_stability import (
                            patch_inner_residual, residual_signed_spearman,
                        )

                        clean_pixels = signed_rgb_channel_sum(clean_result.native_tensor)
                        perturbed_pixels = signed_rgb_channel_sum(perturbed.native_tensor)
                        aligned_pixels = inverse_align_pixel_map(
                            perturbed_pixels, perturbation_id
                        )
                        clean_residual = patch_inner_residual(
                            clean_pixels, method_id=clean_result.method_id,
                            method_version=clean_result.method_version,
                            comparison_line=clean_result.comparison_line,
                            target_identity=target_identity, perturbation=perturbation,
                        )
                        aligned_residual = patch_inner_residual(
                            aligned_pixels, method_id=perturbed.method_id,
                            method_version=perturbed.method_version,
                            comparison_line=perturbed.comparison_line,
                            target_identity=target_identity, perturbation=perturbation,
                        )
                        residual_record = {
                            "perturbation_id": perturbation_id,
                            "clean": clean_residual.to_dict(),
                            "aligned_perturbed": aligned_residual.to_dict(),
                            "signed_spearman": residual_signed_spearman(
                                clean_residual, aligned_residual
                            ).to_dict(),
                        }
                        l0.setdefault("residual_stability", []).append(
                            residual_record
                        )
                        if _unit_audit(unit):
                            l2[f"residual_clean_{perturbation_id}"] = np.array(
                                clean_residual.residual.data, dtype=np.float32, copy=True
                            )
                            l2[f"residual_aligned_{perturbation_id}"] = np.array(
                                aligned_residual.residual.data, dtype=np.float32, copy=True
                            )
        l0["stability"] = stability_records
        repeatability: list[dict[str, Any]] = []
        for item in self._work_items(unit):
            if (
                _attribute(item, "kind") != "map"
                or _attribute(item, "input_variant") != "repeat_clean"
            ):
                continue
            repeat_index = _attribute(item, "repeat_index")
            repeated = self._run_method_observed(
                unit, input_variant="repeat_clean",
                repeat_index=repeat_index, model=model, clean=clean,
                extraction=extraction, target=target,
                target_identity=target_identity,
            )
            if repeated.metric_map is None:
                raise StageCBackendError("repeatability result has no numeric map")
            reference = self._retained_metric_array(clean_result, response)
            observed = self._retained_metric_array(repeated, response)
            absolute = np.abs(reference.astype(np.float64) - observed.astype(np.float64))
            passed = bool(np.allclose(reference, observed, atol=1e-6, rtol=1e-5))
            item_record = {
                "repeat_index": repeat_index,
                "atol": 1e-6, "rtol": 1e-5,
                "max_abs_error": float(absolute.max()), "pass": passed,
            }
            repeatability.append(item_record)
            if self._retain_repeat_clean_maps():
                l1[f"repeat_clean_{repeat_index}"] = observed
            if _unit_audit(unit):
                repeated_native = self._native_array(repeated)
                if repeated_native is not None:
                    l2[f"repeat_{item_record['repeat_index']}_native"] = repeated_native
            if not passed:
                raise StageCScientificInvalidity(
                    "audit_repeatability_failed", phase="audit_repeatability",
                    l0_scalars={**l0, "repeatability": repeatability},
                    details={
                        "repeat_index": repeat_index,
                        "atol": 1e-6, "rtol": 1e-5,
                        "max_abs_error": item_record["max_abs_error"],
                    },
                    partial_evidence=StageCPartialEvidence(
                        l0_scalars={**l0, "repeatability": repeatability},
                        metric_ready_map=l1["clean"],
                    ),
                    message="audit clean-map repeatability failed",
                )
        l0["repeatability"] = repeatability

        if curve_items:
            fidelity_phase = self._phase("fidelity_compute")
        else:
            fidelity_phase = nullcontext()
        with fidelity_phase:
            try:
                self._run_fidelity_work(
                    unit=unit, identity=identity, clean=clean, baseline=baseline,
                    clean_result=clean_result, response=response, evaluator=evaluator,
                    curve_items=curve_items, l0=l0,
                )
            except StageCScientificInvalidity as error:
                # At this boundary the clean method and its metric-ready map
                # are already complete.  Retain only that closed prefix; any
                # unfinished curve and audit payload remains absent.
                if error.partial_evidence is None:
                    error.l0_scalars = dict(l0)
                    error.partial_evidence = StageCPartialEvidence(
                        l0_scalars=dict(l0), metric_ready_map=l1["clean"],
                    )
                raise
        l0["perturbations"] = [item.to_dict() for item in bundle.values()]
        l0["target_identity"] = (
            None if target_identity is None else _jsonable(target_identity)
        )
        l0["clean_tensor_hashes"] = {
            "canonical_rgb_sha256": __import__(
                "vitnns.stage_c_perturbations", fromlist=["tensor_sha256"]
            ).tensor_sha256(canonical),
            "normalized_tensor_sha256": __import__(
                "vitnns.stage_c_perturbations", fromlist=["tensor_sha256"]
            ).tensor_sha256(clean),
        }
        l0["fidelity_enabled"] = bool(curve_items)
        l0["stability_enabled"] = needs_stability
        return self._make_unit_payload(l0=l0, l1=l1, l2=l2)

    def _run_fidelity_work(
        self, *, unit: Any, identity: Mapping[str, Any], clean: Any,
        baseline: Any, clean_result: Any, response: Any, evaluator: Any,
        curve_items: Sequence[Any], l0: dict[str, Any],
    ) -> None:
        from .stage_c_fidelity import (
            CurveDirection, FidelityScale, RankingKind, aggregate_five_curves,
            build_intervention_schedule, build_method_intervention_schedule,
            center_distance_ranking, evaluate_fidelity_curve,
            fidelity_seed_for_namespace, global_random_ranking,
            method_unit_masses,
        )

        curve_records: list[dict[str, Any]] = []
        if curve_items:
            if response is None:
                raise StageCBackendError(
                    "structural-only method cannot execute fidelity work"
                )
            grouped: dict[tuple[Any, ...], list[Any]] = {}
            for item in curve_items:
                group = (
                    _attribute(item, "fidelity_scale"),
                    _attribute(item, "curve_direction"),
                    _attribute(item, "ranking_kind"),
                    _attribute(item, "replicate_index"),
                    _attribute(item, "pairing_block_index"),
                    _attribute(item, "pairing_head_index"),
                )
                grouped.setdefault(group, []).append(item)
            curve_objects: dict[tuple[Any, ...], Any] = {}
            for group, items in grouped.items():
                fraction_pairs = {
                    (_attribute(item, "fraction_index"), _attribute(item, "fraction"))
                    for item in items
                }
                if len(items) != 21 or fraction_pairs != {
                    (index, index / 20.0) for index in range(21)
                }:
                    raise StageCBackendError(
                        "fidelity work must contain complete 21-point curves"
                    )
                scale, direction, ranking_kind, replicate, block, head = group
                parsed_scale = FidelityScale(scale)
                parsed_direction = CurveDirection(direction)
                parsed_kind = RankingKind(ranking_kind)
                if parsed_kind is RankingKind.TARGET_ORIENTED_POSITIVE:
                    unit_map = method_unit_masses(clean_result, response, parsed_scale)
                    schedule = build_method_intervention_schedule(
                        clean, baseline, unit_map, response,
                        sample_id=identity["sample_id"], block=block, head=head,
                        replicate=replicate, direction=parsed_direction,
                        seed_namespace=self.fidelity_seed_namespace,
                    )
                else:
                    count = parsed_scale.unit_count
                    if parsed_kind is RankingKind.GLOBAL_RANDOM:
                        seed = fidelity_seed_for_namespace(
                            self.fidelity_seed_namespace, identity["sample_id"],
                            block, head, replicate,
                        )
                        ranking = global_random_ranking(count, seed)
                    else:
                        seed = replicate = None
                        ranking = center_distance_ranking(parsed_scale)
                    schedule = build_intervention_schedule(
                        clean, baseline, ranking, scale=parsed_scale,
                        direction=parsed_direction, ranking_kind=parsed_kind,
                        sample_id=identity["sample_id"], block=block, head=head,
                        seed=seed, replicate=replicate,
                        seed_namespace=self.fidelity_seed_namespace,
                    )
                curve = evaluate_fidelity_curve(
                    evaluator, response, schedule, method_id=identity["method_id"]
                )
                curve_objects[group] = curve
                curve_records.append(curve.to_dict())
            aggregates: list[dict[str, Any]] = []
            top20: list[dict[str, Any]] = []
            aggregate_groups: dict[tuple[Any, ...], dict[int, Any]] = {}
            for group, curve in curve_objects.items():
                scale, direction, ranking_kind, replicate, block, head = group
                if ranking_kind != RankingKind.TARGET_ORIENTED_POSITIVE.value:
                    continue
                aggregate_groups.setdefault(
                    (scale, direction, ranking_kind, block, head), {}
                )[replicate] = curve
            aggregate_objects: dict[tuple[Any, ...], Any] = {}
            for key, by_replicate in aggregate_groups.items():
                if set(by_replicate) != set(range(5)):
                    raise StageCBackendError(
                        "target-oriented fidelity requires all five paired repeats"
                    )
                aggregate = aggregate_five_curves(
                    by_replicate[index] for index in range(5)
                )
                aggregate_objects[key] = aggregate
                aggregates.append(aggregate.to_dict())
            # Include the explicit block/head pairing in this join.  Global
            # methods deliberately have null block/head in UnitKey, but their
            # fidelity WorkItems carry an explicit pairing context; neither
            # context may be mixed with another one even if scale/replicate
            # happen to match.
            top20_contexts = {
                (key[0], key[3], key[4]) for key in aggregate_objects
            }
            for scale, paired_block, paired_head in sorted(top20_contexts):
                for replicate in range(5):
                    insertion = next((
                        curve for group, curve in curve_objects.items()
                        if group[0] == scale and group[1] == "insertion"
                        and group[2] == RankingKind.TARGET_ORIENTED_POSITIVE.value
                        and group[3] == replicate
                        and group[4] == paired_block and group[5] == paired_head
                    ), None)
                    deletion = next((
                        curve for group, curve in curve_objects.items()
                        if group[0] == scale and group[1] == "deletion"
                        and group[2] == RankingKind.TARGET_ORIENTED_POSITIVE.value
                        and group[3] == replicate
                        and group[4] == paired_block and group[5] == paired_head
                    ), None)
                    if insertion is not None and deletion is not None:
                        top = __import__(
                            "vitnns.stage_c_fidelity", fromlist=["top20_from_curves"]
                        ).top20_from_curves(insertion, deletion)
                        top20.append(top.to_dict())
            l0["fidelity_aggregates"] = aggregates
            l0["top20"] = top20
        l0["fidelity_curves"] = curve_records

    def build_provenance(self, **context: Any) -> Mapping[str, Any]:
        if self.provenance_builder is not None:
            return self.provenance_builder(**context)
        return self._default_provenance(**context)

    def build_failure_provenance(self, **context: Any) -> Mapping[str, Any]:
        if self.provenance_builder is not None:
            return self.provenance_builder(**context)
        unit = context.get("unit")
        identity = _unit_identity(unit)
        if not any(
            item_id == identity["sample_id"]
            for item_id, _device in self._samples
        ):
            return self._pre_target_failure_provenance(**context)
        return self._default_provenance(**context)

    @staticmethod
    def _zero_baseline_sha256() -> str:
        """Return the frozen normalized-zero baseline's real tensor hash."""

        import torch
        from .stage_c_fidelity import tensor_sha256

        baseline = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        return tensor_sha256(baseline)

    def _default_provenance(
        self, *, unit: Any, payload: UnitPayload | None,
        status: Mapping[str, Any], failure: Mapping[str, Any] | None,
        resources: Mapping[str, Any], preflight: Mapping[str, Any],
        artifact_inventory: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Build the strict C6 provenance envelope from bound runtime inputs."""

        if self.config is None or self.registry is None or self.plan is None:
            raise StageCBackendError(
                "default production provenance requires config, registry, and plan"
            )
        from .constants import MODEL_NAME, MODEL_SHA256, REPO_ROOT, WORKSPACE_ROOT
        from .io_utils import sha256_file
        from .stage_c_config import DEFAULT_STAGE_C_CONFIG, DEFAULT_STAGE_C_SCHEMA
        from .stage_c_fidelity import fidelity_seed_for_namespace
        from .stage_c_plan import DEFAULT_REGISTRY_SCHEMA, DEFAULT_SELECTION_SCHEMA
        from .stage_c_provenance import (
            CANONICAL_RGB_HASH_DOMAIN, NORMALIZED_INPUT_HASH_DOMAIN,
            build_stage_c_provenance, canonical_sha256, capture_git_state,
        )

        identity = _unit_identity(unit)
        sample = _attribute(unit, "sample", {})
        sample_id = identity["sample_id"]
        runtime = self._samples.get((sample_id, "cpu"))
        if runtime is None:
            runtime = next(
                (value for (item_id, _device), value in self._samples.items()
                 if item_id == sample_id), None
            )
        if runtime is None:
            raise StageCBackendError(
                "sample context is unavailable for ordinary provenance"
            )
        row = runtime.get("registry_row", sample)
        scalars = {} if payload is None else dict(payload.l0)
        clean_hashes = scalars.get("clean_tensor_hashes", {})
        canonical_hash = clean_hashes.get("canonical_rgb_sha256")
        normalized_hash = clean_hashes.get("normalized_tensor_sha256")
        if not canonical_hash or not normalized_hash:
            from .stage_c_perturbations import tensor_sha256
            canonical_hash = tensor_sha256(runtime["canonical_rgb"])
            normalized_hash = tensor_sha256(runtime["normalized_input"])
        method_config = self.config.method_by_id[identity["method_id"]]
        target_identity = scalars.get("target_identity")
        if target_identity is None and identity["target_ref"] is not None:
            cached_target = (
                next((
                    value for (item_id, _device), value
                    in self._class_targets.items()
                    if item_id == sample_id
                ), None)
                if method_config.comparison_line == "class_logit"
                else next((
                    value for (item_id, _device, block, head), value
                    in self._value_targets.items()
                    if item_id == sample_id
                    and block == identity["block_index"]
                    and head == identity["head_index"]
                ), None)
            )
            if cached_target is not None:
                target_identity = {
                    "comparison_line": method_config.comparison_line,
                    "target_ref": identity["target_ref"],
                    "clean_input_sha256": cached_target.clean_input_sha256,
                    "frozen_target": cached_target.to_dict(),
                }
        method_result = scalars
        normalization_by_method = {
            "direct_raw_cls_attention_per_head": "patch_mass_renormalized_to_one_after_cls_exclusion",
            "direct_raw_cls_attention_last_block_head_mean": "patch_mass_renormalized_to_one_after_cls_exclusion",
            "attention_rollout_mean_head_residual_all_blocks": "patch_mass_renormalized_to_one_after_cls_exclusion",
            "vit_grad_cam_block11_norm1": "none",
            "full_model_class_logit_ig": "none_signed_mass_preserving",
            "conditional_fixed_clean_attention_head_direction_value_path_ig": "none_signed_mass_preserving",
            "matched_fixed_clean_attention_patch_token_contribution": "none_signed_mass_preserving",
        }
        fidelity_enabled = bool(scalars.get("fidelity_enabled", False))
        curve_item = next((
            item for item in self._work_items(unit)
            if _attribute(item, "kind") == "fidelity_curve_point"
        ), None)
        pairing = None
        seeds: list[int] = []
        if fidelity_enabled and curve_item is not None:
            block = int(_attribute(curve_item, "pairing_block_index"))
            head = int(_attribute(curve_item, "pairing_head_index"))
            pairing = {"block_index": block, "head_index": head}
            seeds = [
                fidelity_seed_for_namespace(
                    self.fidelity_seed_namespace, sample_id, block, head, index
                )
                for index in range(5)
            ]
        perturbation_records: list[dict[str, Any]] = []
        if scalars.get("stability_enabled"):
            parameters = _provenance_perturbation_parameters(
                self.config.perturbations["items"]
            )
            bundle = next(
                (value for (item_id, _device), value in self._perturbations.items()
                 if item_id == sample_id), {}
            )
            for perturbation_id, item in bundle.items():
                perturbation_records.append({
                    "id": perturbation_id,
                    "canonical_rgb": {
                        "domain": CANONICAL_RGB_HASH_DOMAIN,
                        "sha256": item.tensor_sha256,
                    },
                    "normalized_input": {
                        "domain": NORMALIZED_INPUT_HASH_DOMAIN,
                        "sha256": item.normalized_tensor_sha256,
                    },
                    "parameters": {
                        **dict(parameters[perturbation_id]),
                        **({"seed": 20260821 + int(_attribute(row, "selection_order"))}
                           if perturbation_id == "gaussian_rgb_std_0_01" else {}),
                    },
                    "shared_exact_tensor_across_methods": True,
                })
        config_path = self.config_path or DEFAULT_STAGE_C_CONFIG
        registry_dict = self.config.registry.to_dict()
        registry_binding = {
            key: registry_dict[key] for key in (
                "path", "sha256", "metadata_path", "metadata_sha256",
                "source_manifest_path", "source_manifest_sha256",
                "source_manifest_metadata_path",
                "source_manifest_metadata_sha256",
            )
        }
        selection = self.plan.selection.to_dict()
        selection_status = (
            "present" if selection["status"] == "present"
            else "not_applicable_pre_c8"
        )
        selection["status"] = selection_status
        return build_stage_c_provenance(
            unit_identity=identity, code=(
                dict(self.code_state) if self.code_state is not None
                else capture_git_state(REPO_ROOT)
            ),
            model={
                "name": MODEL_NAME,
                "state_dict_path": os.path.relpath(
                    (self.state_dict_path or Path("model.pth")).resolve(),
                    WORKSPACE_ROOT,
                ),
                "state_dict_sha256": MODEL_SHA256,
            },
            input_record={
                **_mapping(row, "registry row"),
                "canonical_rgb_sha256": canonical_hash,
                "normalized_tensor_sha256": normalized_hash,
            },
            dependencies={
                "lock_path": "requirements.lock.txt",
                "lock_sha256": sha256_file(REPO_ROOT / "requirements.lock.txt"),
            },
            registry=registry_binding,
            protocol={
                "path": self.config.protocol.path,
                "sha256": sha256_file(REPO_ROOT / self.config.protocol.path),
                "version": self.config.protocol.version,
            },
            acceptance={
                "path": "docs/stage_c_acceptance.md",
                "sha256": sha256_file(REPO_ROOT / "docs/stage_c_acceptance.md"),
            },
            configuration={
                "path": os.path.relpath(config_path.resolve(), REPO_ROOT),
                "sha256": sha256_file(config_path),
                "schema_path": "config/stage_c.schema.json",
                "schema_sha256": sha256_file(DEFAULT_STAGE_C_SCHEMA),
                "registry_schema_path": "config/stage_c_registry.schema.json",
                "registry_schema_sha256": sha256_file(DEFAULT_REGISTRY_SCHEMA),
                "selection_schema_path": "config/stage_c_selection.schema.json",
                "selection_schema_sha256": sha256_file(DEFAULT_SELECTION_SCHEMA),
                "run_spec_sha256": canonical_sha256(self.plan.run_spec.to_dict()),
                "execution_plan_sha256": self.plan.digest,
            },
            selection=selection,
            method={
                "id": method_config.id, "exact_name": method_config.exact_name,
                "version": method_result.get("method_version", "1.0.0"),
                "comparison_line": method_config.comparison_line,
                "explanation_target_ref": method_config.explanation_target_ref,
                "fidelity_target_ref": method_config.fidelity_target_ref,
                "class_agnostic": method_config.class_agnostic,
                "fidelity_eligible": method_config.fidelity_eligible,
                "structural_reference_only": method_config.structural_reference_only,
                "native_map": method_config.native_map,
            },
            target={
                "status": (
                    "resolved"
                    if target_identity is not None or identity["target_ref"] is None
                    else "unavailable"
                ),
                "reason": (
                    None if target_identity is not None
                    or identity["target_ref"] is None
                    else (str(failure.get("message")) if failure else "target unavailable")
                ),
                "comparison_line": method_config.comparison_line,
                "target_ref": identity["target_ref"],
                "clean_input_normalized_sha256": (
                    normalized_hash if target_identity is not None
                    or identity["target_ref"] is None else None
                ),
                "identity_sha256": (
                    None if target_identity is None else canonical_sha256(target_identity)
                ),
                "identity": target_identity,
            },
            map_record={
                "projection": method_config.projection_ref,
                "sign_semantics": method_result.get(
                    "sign_semantics", _method_sign_semantics(method_config.id)
                ),
                "normalization": normalization_by_method[method_config.id],
                "metric_ready_dtype": "float32",
                "metric_ready_shape": [14, 14],
            },
            baseline={
                **self.config.targets["baseline"],
                "normalized_tensor_sha256": self._zero_baseline_sha256(),
            },
            ig_schedule=dict(self.config.targets["ig"]),
            seeds={
                "base_seed": 20260821,
                "noise_seed": 20260821 + int(_attribute(row, "selection_order")),
                "noise_seed_rule": "base_seed_plus_registry_selection_order",
                "fidelity_enabled": fidelity_enabled,
                "fidelity_seed_namespace": (
                    self.fidelity_seed_namespace
                    + "sample_id|block|head|replicate"
                ),
                "fidelity_pairing_context": pairing,
                "fidelity_seeds": seeds,
            },
            perturbations={
                "enabled": bool(perturbation_records),
                "application_domain": CANONICAL_RGB_HASH_DOMAIN,
                "normalized_domain": NORMALIZED_INPUT_HASH_DOMAIN,
                "clean": {
                    "id": "clean",
                    "canonical_rgb": {
                        "domain": CANONICAL_RGB_HASH_DOMAIN,
                        "sha256": canonical_hash,
                    },
                    "normalized_input": {
                        "domain": NORMALIZED_INPUT_HASH_DOMAIN,
                        "sha256": normalized_hash,
                    },
                },
                "items": perturbation_records,
            },
            status=status, failure=failure,
            resources={"preflight": preflight, "actual": resources},
            retention={
                "audit": _unit_audit(unit),
                "display_is_never_metric_input": True,
                "l1_storage_dtype": "float32", "l2_audit_only": True,
                "l3_display_only": True,
                "forbidden_granularities": [
                    "per_quadrature_step_files", "per_pixel_files",
                    "per_dimension_files", "display_png_as_metric_input",
                ],
            },
            artifact_inventory=artifact_inventory,
        ).to_dict()

    def _pre_target_failure_provenance(
        self, *, unit: Any, payload: UnitPayload | None,
        status: Mapping[str, Any],
        failure: Mapping[str, Any], resources: Mapping[str, Any],
        preflight: Mapping[str, Any],
        artifact_inventory: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Build strict provenance when model/image preprocessing failed early."""

        if payload is not None:
            raise StageCBackendError(
                "pre-target failure provenance cannot contain a payload"
            )

        if self.config is None or self.registry is None or self.plan is None:
            raise StageCBackendError(
                "default production failure provenance requires config, registry, and plan"
            )
        from .constants import MODEL_NAME, MODEL_SHA256, REPO_ROOT, WORKSPACE_ROOT
        from .io_utils import sha256_file
        from .stage_c_config import DEFAULT_STAGE_C_CONFIG, DEFAULT_STAGE_C_SCHEMA
        from .stage_c_plan import DEFAULT_REGISTRY_SCHEMA, DEFAULT_SELECTION_SCHEMA
        from .stage_c_provenance import (
            CANONICAL_RGB_HASH_DOMAIN, NORMALIZED_INPUT_HASH_DOMAIN,
            build_stage_c_provenance, canonical_sha256, capture_git_state,
        )

        identity = _unit_identity(unit)
        sample_id = identity["sample_id"]
        row = next((
            item for item in self.registry.rows
            if _attribute(item, "sample_id") == sample_id
        ), None)
        if row is None:
            raise StageCBackendError(
                "execution unit sample is unavailable in the bound registry"
            )
        method = self.config.method_by_id[identity["method_id"]]
        normalization = {
            "direct_raw_cls_attention_per_head": "patch_mass_renormalized_to_one_after_cls_exclusion",
            "direct_raw_cls_attention_last_block_head_mean": "patch_mass_renormalized_to_one_after_cls_exclusion",
            "attention_rollout_mean_head_residual_all_blocks": "patch_mass_renormalized_to_one_after_cls_exclusion",
            "vit_grad_cam_block11_norm1": "none",
            "full_model_class_logit_ig": "none_signed_mass_preserving",
            "conditional_fixed_clean_attention_head_direction_value_path_ig": "none_signed_mass_preserving",
            "matched_fixed_clean_attention_patch_token_contribution": "none_signed_mass_preserving",
        }[method.id]
        sign = _method_sign_semantics(method.id)
        registry_data = self.config.registry.to_dict()
        selection = self.plan.selection.to_dict()
        selection["status"] = (
            "present" if selection["status"] == "present"
            else "not_applicable_pre_c8"
        )
        config_path = self.config_path or DEFAULT_STAGE_C_CONFIG
        null_clean = {
            "id": "clean",
            "canonical_rgb": {
                "domain": CANONICAL_RGB_HASH_DOMAIN, "sha256": None,
            },
            "normalized_input": {
                "domain": NORMALIZED_INPUT_HASH_DOMAIN, "sha256": None,
            },
        }
        return build_stage_c_provenance(
            unit_identity=identity,
            code=(dict(self.code_state) if self.code_state is not None
                  else capture_git_state(REPO_ROOT)),
            model={
                "name": MODEL_NAME,
                "state_dict_path": os.path.relpath(
                    (self.state_dict_path or Path("model.pth")).resolve(),
                    WORKSPACE_ROOT,
                ),
                "state_dict_sha256": MODEL_SHA256,
            },
            input_record={
                **_mapping(row, "registry row"),
                "canonical_rgb_sha256": None,
                "normalized_tensor_sha256": None,
            },
            dependencies={
                "lock_path": "requirements.lock.txt",
                "lock_sha256": sha256_file(REPO_ROOT / "requirements.lock.txt"),
            },
            registry={key: registry_data[key] for key in (
                "path", "sha256", "metadata_path", "metadata_sha256",
                "source_manifest_path", "source_manifest_sha256",
                "source_manifest_metadata_path",
                "source_manifest_metadata_sha256",
            )},
            protocol={
                "path": self.config.protocol.path,
                "sha256": sha256_file(REPO_ROOT / self.config.protocol.path),
                "version": self.config.protocol.version,
            },
            acceptance={
                "path": "docs/stage_c_acceptance.md",
                "sha256": sha256_file(REPO_ROOT / "docs/stage_c_acceptance.md"),
            },
            configuration={
                "path": os.path.relpath(config_path.resolve(), REPO_ROOT),
                "sha256": sha256_file(config_path),
                "schema_path": "config/stage_c.schema.json",
                "schema_sha256": sha256_file(DEFAULT_STAGE_C_SCHEMA),
                "registry_schema_path": "config/stage_c_registry.schema.json",
                "registry_schema_sha256": sha256_file(DEFAULT_REGISTRY_SCHEMA),
                "selection_schema_path": "config/stage_c_selection.schema.json",
                "selection_schema_sha256": sha256_file(DEFAULT_SELECTION_SCHEMA),
                "run_spec_sha256": canonical_sha256(self.plan.run_spec.to_dict()),
                "execution_plan_sha256": self.plan.digest,
            },
            selection=selection,
            method={
                "id": method.id, "exact_name": method.exact_name,
                "version": "1.0.0", "comparison_line": method.comparison_line,
                "explanation_target_ref": method.explanation_target_ref,
                "fidelity_target_ref": method.fidelity_target_ref,
                "class_agnostic": method.class_agnostic,
                "fidelity_eligible": method.fidelity_eligible,
                "structural_reference_only": method.structural_reference_only,
                "native_map": method.native_map,
            },
            target={
                "status": "unavailable", "reason": failure["message"],
                "comparison_line": method.comparison_line,
                "target_ref": identity["target_ref"],
                "clean_input_normalized_sha256": None,
                "identity_sha256": None, "identity": None,
            },
            map_record={
                "projection": method.projection_ref, "sign_semantics": sign,
                "normalization": normalization,
                "metric_ready_dtype": "float32",
                "metric_ready_shape": [14, 14],
            },
            baseline={
                **self.config.targets["baseline"],
                "normalized_tensor_sha256": self._zero_baseline_sha256(),
            },
            ig_schedule=dict(self.config.targets["ig"]),
            seeds={
                "base_seed": 20260821,
                "noise_seed": 20260821 + int(_attribute(row, "selection_order")),
                "noise_seed_rule": "base_seed_plus_registry_selection_order",
                "fidelity_enabled": False,
                "fidelity_seed_namespace": (
                    self.fidelity_seed_namespace
                    + "sample_id|block|head|replicate"
                ),
                "fidelity_pairing_context": None, "fidelity_seeds": [],
            },
            perturbations={
                "enabled": False,
                "application_domain": CANONICAL_RGB_HASH_DOMAIN,
                "normalized_domain": NORMALIZED_INPUT_HASH_DOMAIN,
                "clean": null_clean, "items": [],
            },
            status=status, failure=failure,
            resources={"preflight": preflight, "actual": resources},
            retention={
                "audit": _unit_audit(unit),
                "display_is_never_metric_input": True,
                "l1_storage_dtype": "float32", "l2_audit_only": True,
                "l3_display_only": True,
                "forbidden_granularities": [
                    "per_quadrature_step_files", "per_pixel_files",
                    "per_dimension_files", "display_png_as_metric_input",
                ],
            },
            artifact_inventory=artifact_inventory,
        ).to_dict()


@dataclass(frozen=True)
class UnitRunResult(Mapping[str, Any]):
    unit_key: Mapping[str, Any]
    state: str
    attempt_path: str | None
    resumed: bool
    failure: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.state not in _TERMINAL_STATES | {UNIT_INCOMPLETE, UNIT_TAMPERED}:
            raise ValueError(f"invalid unit run state: {self.state!r}")
        if type(self.resumed) is not bool:
            raise TypeError("resumed must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_key": _mapping(self.unit_key, "unit_key"),
            "state": self.state,
            "attempt_path": self.attempt_path,
            "resumed": self.resumed,
            "failure": None if self.failure is None else _mapping(self.failure, "failure"),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class BatchRunResult(Mapping[str, Any]):
    plan_digest: str
    output_root: str
    orchestration_complete: bool
    successful: bool
    unit_results: tuple[UnitRunResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_C_RUNNER_SCHEMA_VERSION,
            "document_type": "stage_c_batch_run_result",
            "plan_digest": self.plan_digest,
            "output_root": self.output_root,
            "orchestration_complete": self.orchestration_complete,
            "successful": self.successful,
            "unit_results": [item.to_dict() for item in self.unit_results],
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class BatchValidation(Mapping[str, Any]):
    plan_digest: str
    valid: bool
    orchestration_complete: bool
    successful: bool
    rows: tuple[Mapping[str, Any], ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_C_RUNNER_SCHEMA_VERSION,
            "document_type": "stage_c_batch_validation",
            "plan_digest": self.plan_digest,
            "valid": self.valid,
            "orchestration_complete": self.orchestration_complete,
            "successful": self.successful,
            "rows": [_mapping(row, "validation row") for row in self.rows],
            "errors": list(self.errors),
        }

    def require_valid(self) -> None:
        if not self.valid:
            raise StageCBatchValidationError(
                "Stage C batch validation failed: " + "; ".join(self.errors)
            )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def _failure(reason: str, phase: str, error: BaseException) -> dict[str, str]:
    message = str(error)
    if not message.strip():
        message = f"{type(error).__name__} raised without a message"
    return {
        "reason": reason,
        "phase": phase,
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": message,
    }


def _failure_observability(error: BaseException) -> dict[str, Any] | None:
    value = getattr(error, "stage_c_execution_observability", None)
    if value is None:
        return None
    try:
        return _mapping(value, "failure execution observability")
    except Exception as conversion_error:
        return {
            "capture_error": {
                "type": (
                    f"{type(conversion_error).__module__}."
                    f"{type(conversion_error).__qualname__}"
                ),
                "message": str(conversion_error),
            }
        }


def _unavailable_lifecycle(error: BaseException | None = None) -> dict[str, Any]:
    """Explicit evidence for injected/test backends without a model boundary."""

    digest = "0" * 64
    return {
        "integrity": {
            "schema_version": 1,
            "algorithm": "unavailable_non_model_backend",
            "checked": ["backend_exposes_no_model"],
            "entry_sha256": digest,
            "exit_sha256": digest,
            "matched": True,
            "restored": True,
            "differences": [],
            "signature_calls": 1,
            "verification_error_type": None,
            "verification_error_message": None,
            "body_exception_type": (
                None
                if error is None else
                f"{type(error).__module__}.{type(error).__qualname__}"
            ),
        },
        "restoration": None,
    }


def _lifecycle_from_payload_or_error(
    payload: UnitPayload | None, error: BaseException | None = None
) -> dict[str, Any]:
    if payload is not None and payload.lifecycle:
        return _mapping(payload.lifecycle, "payload lifecycle")
    if error is not None:
        evidence = _failure_observability(error)
        lifecycle = None if evidence is None else evidence.get("lifecycle")
        if isinstance(lifecycle, Mapping):
            return _mapping(lifecycle, "failure lifecycle")
    return _unavailable_lifecycle(error)


def _runner_telemetry(
    *, timer: PhaseTimer, device: str, lifecycle: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase_timings": timer.to_dict(),
        "lifecycle": _mapping(lifecycle, "runner lifecycle"),
        "measurement_environment": measurement_environment(device),
    }


def _empty_resource_record(
    *, device: str, status: str = "completed", error: BaseException | None = None
) -> ResourceRecord:
    is_cuda = str(device).startswith("cuda")
    exception_type = None
    exception_message = None
    if error is not None:
        exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
        exception_message = str(error)
        if not exception_message.strip():
            exception_message = (
                f"{type(error).__name__} raised without a message"
            )
    return ResourceRecord(
        status=status,
        wall_time_seconds=None,
        cuda_status=("measurement_failed" if is_cuda else "not_applicable_cpu"),
        cuda_device=(str(device) if is_cuda else None),
        cuda_peak_allocated_bytes=None,
        cuda_peak_reserved_bytes=None,
        calls=CallCounts(0, 0),
        artifacts=ArtifactByteMeasurement((), 0),
        exception_type=exception_type,
        exception_message=exception_message,
        instrumentation_errors=(),
    )


def _resource_dict(
    record: ResourceRecord,
    *,
    completed: bool,
    failure: Mapping[str, Any] | None,
    artifact_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = record.to_dict()
    files = [
        {"path": str(item["path"]), "bytes": int(item["bytes"])}
        for item in artifact_records
    ]
    result["artifact_bytes"] = sum(item["bytes"] for item in files)
    result["artifact_bytes_scope"] = "explicit_payload_file_list"
    result["artifact_root"] = None
    result["files"] = files
    if completed:
        result["status"] = "completed"
        result["exception"] = None
        result["instrumentation_errors"] = []
    else:
        result["status"] = "failed"
        if result.get("exception") is None:
            assert failure is not None
            result["exception"] = {
                "type": str(failure["type"]),
                "message": str(failure["message"]),
            }
    return result


def _check_payload_binding(unit: Any, payload: UnitPayload) -> None:
    line = _unit_line(unit)
    observed_line = payload.l0.get("comparison_line")
    if observed_line is not None and observed_line != line:
        raise StageCBackendError(
            f"backend crossed comparison lines: unit={line}, payload={observed_line}"
        )
    identity = _unit_identity(unit)
    observed_method = payload.l0.get("method_id")
    if observed_method is not None and observed_method != identity["method_id"]:
        raise StageCBackendError("backend payload method_id disagrees with its unit")
    observed_target = payload.l0.get("target_ref")
    if observed_target is not None and observed_target != identity["target_ref"]:
        raise StageCBackendError("backend payload target_ref disagrees with its unit")


def _invoke_backend(
    backend: Any, unit: Any, *, device: str, call_counter: CallCounter,
    phase_timer: PhaseTimer | None = None,
) -> UnitPayload:
    timed_function = getattr(backend, "run_with_phase_timer", None)
    function = (
        timed_function if phase_timer is not None and callable(timed_function)
        else getattr(backend, "run", backend if callable(backend) else None)
    )
    if not callable(function):
        raise TypeError("backend must be callable or expose run(unit, ...)")
    signature = inspect.signature(function)
    keywords: dict[str, Any] = {}
    if "device" in signature.parameters:
        keywords["device"] = device
    if "call_counter" in signature.parameters:
        keywords["call_counter"] = call_counter
    if phase_timer is not None and "phase_timer" in signature.parameters:
        keywords["phase_timer"] = phase_timer
        return UnitPayload.coerce(function(unit, **keywords))
    phase = (
        nullcontext() if phase_timer is None
        else phase_timer.measure("attribution_or_map_compute")
    )
    with phase:
        return UnitPayload.coerce(function(unit, **keywords))


def _passing_preflight(
    plan: Any, estimate: Any, *, allow_unissued_estimate: bool = False
) -> dict[str, Any]:
    candidate = estimate
    if candidate is None:
        candidate = _attribute(plan, "estimate", None)
    if candidate is None:
        raise StageCPreflightError(
            "a Stage C estimate is required before the output root is created"
        )
    data = _mapping(candidate, "preflight estimate")
    within = _attribute(candidate, "within_budget", data.get("within_budget"))
    if within is not True:
        violations = _attribute(candidate, "violations", data.get("violations", ()))
        raise StageCPreflightError(
            "Stage C hard-cap preflight failed"
            + (f": {list(violations)}" if violations else "")
        )
    # Invoke the planning module's authoritative guard when available.
    try:
        from .stage_c_plan import enforce_stage_c_budget
    except ImportError:
        enforce_stage_c_budget = None
    if enforce_stage_c_budget is not None:
        try:
            enforce_stage_c_budget(candidate)
        except (TypeError, ValueError) as error:
            if not allow_unissued_estimate:
                raise StageCPreflightError(
                    "execution requires an issuer-authenticated StageCEstimate"
                ) from error
    _validated_preflight_binding(data, plan=plan)
    return data


def _runtime_caps(caps: Mapping[str, Any] | None) -> dict[str, int]:
    values = dict(caps or {})
    return {
        "forward": int(values.get("forward_calls_per_execution_unit_max", 30000)),
        "backward": int(values.get("backward_calls_per_execution_unit_max", 30000)),
        "reserved": int(
            values.get("cuda_peak_reserved_bytes_per_execution_unit_max", 90 * 1024**3)
        ),
    }


def _enforce_actual_resources(record: ResourceRecord, caps: Mapping[str, Any] | None) -> None:
    limits = _runtime_caps(caps)
    if record.forward_calls > limits["forward"]:
        raise StageCBackendError("actual forward calls exceed the execution-unit cap")
    if record.backward_calls > limits["backward"]:
        raise StageCBackendError("actual backward calls exceed the execution-unit cap")
    reserved = record.cuda_peak_reserved_bytes
    if reserved is not None and reserved > limits["reserved"]:
        raise StageCBackendError("actual peak CUDA reserved bytes exceed the unit cap")


def _prepare_provenance(
    backend: Any,
    unit: Any,
    payload: UnitPayload | None,
    *,
    state: str,
    failure: Mapping[str, Any] | None,
    resources: Mapping[str, Any],
    preflight: Mapping[str, Any],
    artifact_inventory: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    def invoke(function: Callable[..., Any], context: Mapping[str, Any]) -> Any:
        parameters = inspect.signature(function).parameters.values()
        if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters):
            return function(**dict(context))
        names = {item.name for item in parameters}
        return function(**{name: item for name, item in context.items() if name in names})

    function_name = "build_provenance" if payload is not None else "build_failure_provenance"
    function = getattr(backend, function_name, None)
    if callable(function):
        context = {
            "unit": unit,
            "payload": payload,
            "status": {"state": state, "terminal": True},
            "failure": failure,
            "resources": resources,
            "preflight": preflight,
            "artifact_inventory": tuple(artifact_inventory),
        }
        provenance = invoke(function, context)
    elif payload is not None and payload.provenance:
        provenance = copy.deepcopy(_mapping(payload.provenance, "payload provenance"))
        bindings = provenance.get("bindings")
        if isinstance(bindings, dict):
            bindings["status"] = {"state": state, "terminal": True}
            bindings["failure"] = None if failure is None else dict(failure)
            bindings["resources"] = {"preflight": dict(preflight), "actual": dict(resources)}
            bindings["artifact_inventory"] = [dict(item) for item in artifact_inventory]
    else:
        # A backend exception may happen before a payload exists.  If the
        # backend can build ordinary provenance, use that same strict builder
        # for the failed terminal state rather than leaving the attempt
        # indefinitely running.
        fallback = getattr(backend, "build_provenance", None)
        if payload is None and callable(fallback):
            context = {
                "unit": unit, "payload": None,
                "status": {"state": state, "terminal": True},
                "failure": failure, "resources": resources,
                "preflight": preflight,
                "artifact_inventory": tuple(artifact_inventory),
            }
            provenance = invoke(fallback, context)
        else:
            return None
    if not isinstance(provenance, Mapping):
        raise TypeError("backend provenance builder must return a mapping")
    from .stage_c_provenance import validate_stage_c_provenance

    validate_stage_c_provenance(provenance, expected_unit=_unit_identity(unit))
    return provenance


def _unit_root(output_root: Path, unit: Any) -> Path:
    return output_root / "units" / _unit_digest(unit) / "attempts"


def _existing_resumable_attempt(
    unit_root: Path, unit: Any, *,
    expected_provenance_bindings: Mapping[str, Any],
    observed_git_state: Mapping[str, Any] | None,
    caps: Mapping[str, Any] | None,
) -> Path | None:
    from .stage_c_artifacts import attempt_directories, validate_execution_unit

    resumable: list[Path] = []
    for attempt in attempt_directories(unit_root):
        validation = validate_execution_unit(
            attempt, expected_unit_identity=_unit_identity(unit),
            expected_provenance_bindings=expected_provenance_bindings,
            observed_git_state=observed_git_state, caps=caps,
            require_runner_telemetry=True,
        )
        if validation.resumable:
            resumable.append(attempt)
    if len(resumable) > 1:
        raise StageCBatchValidationError(
            f"unit {_unit_digest(unit)} has multiple valid complete attempts"
        )
    return resumable[0] if resumable else None


def run_execution_unit(
    unit: Any,
    *,
    output_root: Path | str,
    backend: StageCBackend | Callable[..., Any],
    device: str = "cpu",
    resume: bool = True,
    caps: Mapping[str, Any] | None = None,
    preflight: Mapping[str, Any] | None = None,
    plan: Any | None = None,
    observed_git_state: Mapping[str, Any] | None = None,
    fault_injector: Callable[[str], Any] | str | Sequence[str] | None = None,
) -> UnitRunResult:
    """Run one already-preflighted unit in a new immutable attempt.

    Public batch/sample entry points always supply the passing preflight.  A
    direct caller must do the same; rejecting ``None`` prevents this low-level
    API from accidentally becoming a preflight bypass.
    """

    if preflight is None or preflight.get("within_budget") is not True:
        raise StageCPreflightError("run_execution_unit requires a passing preflight")
    identity = _unit_identity(unit)
    active_plan = plan or getattr(backend, "plan", None)
    dynamic_validation = active_plan is not None
    expected_bindings = _expected_provenance_bindings(
        plan=active_plan if dynamic_validation else None, preflight=preflight
    )
    git_state = _runtime_git_state(
        plan=active_plan if dynamic_validation else None, backend=(
            backend if dynamic_validation else None
        ), observed_git_state=observed_git_state
    )
    root = _safe_output_path(output_root, allow_missing=True)
    unit_root = _unit_root(root, unit)
    if resume:
        existing = _existing_resumable_attempt(
            unit_root, unit,
            expected_provenance_bindings=expected_bindings,
            observed_git_state=git_state, caps=caps,
        )
        if existing is not None:
            return UnitRunResult(identity, UNIT_SKIPPED, str(existing), True, None)

    from .stage_c_artifacts import (
        StageCArtifactWriter,
        next_attempt_directory,
        validate_execution_unit,
    )

    attempt = next_attempt_directory(unit_root)
    writer = StageCArtifactWriter(
        attempt, unit_identity=identity, audit=_unit_audit(unit), caps=caps,
        fault_injector=fault_injector,
    )
    meter = ResourceMeter(device=device, call_counter=CallCounter())
    timer = PhaseTimer(device=device)
    timer.start()
    payload: UnitPayload | None = None
    try:
        with meter:
            payload = _invoke_backend(
                backend, unit, device=device, call_counter=meter.call_counter,
                phase_timer=timer,
            )
        record = meter.record
        _enforce_actual_resources(record, caps)
        _check_payload_binding(unit, payload)
        lifecycle = _lifecycle_from_payload_or_error(payload)

        with timer.measure("artifact_finalize"):
            for name, value in payload.l1.items():
                writer.write_l1(name, value)
            for name, value in payload.l2.items():
                writer.write_l2(name, value)
            for name, value in payload.l3.items():
                writer.write_l3(name, value, failure=False)

            resource = _resource_dict(
                record, completed=True, failure=None,
                artifact_records=writer.artifacts,
            )
            l0_document = {
                "schema_version": STAGE_C_RUNNER_SCHEMA_VERSION,
                "document_type": "stage_c_execution_unit_l0",
                "unit_identity": identity,
                "state": "completed",
                "failure": None,
                "scalars": _mapping(payload.l0, "payload L0"),
                "resources": resource,
                "backend_resources": _mapping(
                    payload.resources, "payload backend resources"
                ),
                "lifecycle": lifecycle,
            }
            writer.write_l0("result", l0_document, kind="result")
            resource = _resource_dict(
                record, completed=True, failure=None,
                artifact_records=writer.artifacts,
            )
            provenance = _prepare_provenance(
                backend, unit, payload, state="completed", failure=None,
                resources=resource, preflight=preflight,
                artifact_inventory=writer.artifacts,
            )
            final = writer.finalize(provenance=provenance)
            validation = validate_execution_unit(
                attempt, expected_unit_identity=identity,
                expected_provenance_bindings=expected_bindings,
                observed_git_state=git_state, caps=caps,
            )
        timer.stop()
        writer.publish_runner_telemetry(
            _runner_telemetry(
                timer=timer, device=device, lifecycle=lifecycle,
            )
        )
        validation = validate_execution_unit(
            attempt, expected_unit_identity=identity,
            expected_provenance_bindings=expected_bindings,
            observed_git_state=git_state, caps=caps,
            require_runner_telemetry=True,
        )
        if not validation.resumable:
            return UnitRunResult(
                identity, UNIT_TAMPERED, str(attempt), False,
                {"reason": "post_finalize_validation_failed", "errors": list(validation.errors)},
            )
        if isinstance(final, Mapping) and final.get("complete") is not True:
            raise StageCRunnerError("artifact writer did not publish a complete status")
        return UnitRunResult(identity, UNIT_SUCCEEDED, str(attempt), False, None)
    except BaseException as error:
        failure = _failure("execution_unit_failed", "backend_or_persistence", error)
        failure_observability = _failure_observability(error)
        lifecycle = _lifecycle_from_payload_or_error(payload, error)
        try:
            record = meter.record
        except RuntimeError:
            record = _empty_resource_record(device=device, status="failed", error=error)
        resource = _resource_dict(
            record, completed=False, failure=failure, artifact_records=writer.artifacts
        )
        try:
            if not writer.terminal:
                with timer.measure("artifact_finalize"):
                    if failure_observability is not None:
                        writer.write_l0(
                            "execution_lifecycle", failure_observability,
                            kind="failure_evidence",
                        )
                    writer.prepare_failure(failure["reason"], failure)
                    resource = _resource_dict(
                        record, completed=False, failure=failure,
                        artifact_records=writer.artifacts,
                    )
                    provenance = _prepare_provenance(
                        backend, unit, payload, state="failed", failure=failure,
                        resources=resource, preflight=preflight,
                        artifact_inventory=writer.artifacts,
                    )
                    writer.fail(
                        failure["reason"], details=failure, provenance=provenance
                    )
                    validate_execution_unit(
                        attempt, expected_unit_identity=identity,
                        expected_provenance_bindings=expected_bindings,
                        observed_git_state=git_state, caps=caps,
                    )
            if not timer.stopped:
                timer.stop()
            if not writer.terminal:
                raise StageCRunnerError(
                    "terminal failure persistence did not publish a terminal state"
                )
            writer.publish_runner_telemetry(
                _runner_telemetry(
                    timer=timer, device=device, lifecycle=lifecycle,
                )
            )
            terminal_validation = validate_execution_unit(
                attempt, expected_unit_identity=identity,
                expected_provenance_bindings=expected_bindings,
                observed_git_state=git_state, caps=caps,
                require_runner_telemetry=True,
            )
            if (
                not terminal_validation.valid
                or terminal_validation.status != "failed"
                or not terminal_validation.terminal
                or terminal_validation.complete
                or writer.manifest.get("failure") != {
                    "reason": failure["reason"], "details": failure
                }
            ):
                raise StageCRunnerError(
                    "terminal failure validation failed: "
                    + (
                        "; ".join(terminal_validation.errors)
                        or "attempt is not a terminal failed state"
                    )
                )
            failed_result = UnitRunResult(
                identity, UNIT_FAILED, str(attempt), False, failure
            )
        except BaseException as persistence_error:
            combined = dict(failure)
            combined["persistence_error"] = {
                "type": f"{type(persistence_error).__module__}.{type(persistence_error).__qualname__}",
                "message": str(persistence_error),
            }
            incomplete_result = UnitRunResult(
                identity, UNIT_INCOMPLETE, str(attempt), False, combined
            )
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                error.stage_c_unit_result = incomplete_result
                raise error
            return incomplete_result
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            error.stage_c_unit_result = failed_result
            raise
        return failed_result


def _atomic_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish one deterministic batch-control JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _mapping(value, str(path)), indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _canonical_control_bytes(value: Mapping[str, Any], *, context: str) -> bytes:
    """Return the one accepted on-disk representation of an immutable control."""

    return (
        json.dumps(
            _mapping(value, context), indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _publish_or_validate_immutable_control(
    path: Path, value: Mapping[str, Any],
) -> None:
    """Create one immutable control, or require an exact existing match.

    A temporary regular file plus ``link`` provides no-replace publication:
    another process winning the destination race cannot be overwritten.
    Existing controls are accepted only when they are regular, non-symlink
    files using the exact canonical encoding and expected content.
    """

    expected = _canonical_control_bytes(value, context=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except FileExistsError:
            pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _validate_immutable_control(path, value)


def _validate_immutable_control(
    path: Path, value: Mapping[str, Any],
) -> None:
    """Require an already-published immutable control to match exactly."""

    expected = _canonical_control_bytes(value, context=str(path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StageCRunnerError(
            f"immutable batch control must be a non-symlink regular file: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise StageCRunnerError(
                    "immutable batch control must be a non-symlink regular "
                    f"file: {path}"
                )
            observed = handle.read()
    except BaseException:
        # ``fdopen`` owns and closes the descriptor once it succeeds.
        raise
    if observed != expected:
        # Parse only to distinguish non-canonical encoding from content drift;
        # neither condition authorizes replacement.
        try:
            parsed = _strict_json_bytes(observed, context=str(path))
        except Exception as error:
            raise StageCRunnerError(
                f"immutable batch control is invalid JSON: {path}: {error}"
            ) from error
        if parsed == _mapping(value, str(path)):
            raise StageCRunnerError(
                f"immutable batch control is not canonically encoded: {path}"
            )
        raise StageCRunnerError(
            f"immutable batch control content drift: {path}"
        )


def _publish_or_validate_batch_identity_controls(
    output_root: Path, plan: Any, preflight: Mapping[str, Any],
) -> None:
    _publish_or_validate_immutable_control(
        output_root / EXECUTION_PLAN_FILENAME, _plan_dict(plan)
    )
    _publish_or_validate_immutable_control(
        output_root / PREFLIGHT_FILENAME, preflight
    )


def _validate_batch_identity_controls(
    output_root: Path, plan: Any, preflight: Mapping[str, Any],
) -> None:
    _validate_immutable_control(
        output_root / EXECUTION_PLAN_FILENAME, _plan_dict(plan)
    )
    _validate_immutable_control(output_root / PREFLIGHT_FILENAME, preflight)


def _write_batch_controls(
    output_root: Path,
    plan: Any,
    preflight: Mapping[str, Any],
    results: Sequence[UnitRunResult],
) -> None:
    rows = []
    for result in results:
        status_sha256 = None
        if result.attempt_path is not None:
            status_path = Path(result.attempt_path) / "status.json"
            if status_path.is_file() and not status_path.is_symlink():
                status_sha256 = hashlib.sha256(status_path.read_bytes()).hexdigest()
        rows.append(
            {
                "unit_key": _mapping(result.unit_key, "unit key"),
                "state": result.state,
                "terminal": result.state in _TERMINAL_STATES,
                "complete": result.state in {UNIT_SUCCEEDED, UNIT_SKIPPED},
                "selected_attempt": result.attempt_path,
                "selected_status_sha256": status_sha256,
                "failure": (
                    None
                    if result.failure is None
                    else _mapping(result.failure, "failure")
                ),
            }
        )
    orchestration_complete = all(row["terminal"] for row in rows)
    successful = bool(rows) and all(row["complete"] for row in rows)
    _validate_batch_identity_controls(output_root, plan, preflight)
    _atomic_json_replace(
        output_root / FAILURE_TABLE_FILENAME,
        {
            "schema_version": 1,
            "document_type": "stage_c_failure_table",
            "plan_digest": _plan_digest(plan),
            "rows": rows,
        },
    )
    _atomic_json_replace(
        output_root / BATCH_STATUS_FILENAME,
        {
            "schema_version": 1,
            "document_type": "stage_c_batch_status",
            "plan_digest": _plan_digest(plan),
            "expected_unit_count": len(rows),
            "orchestration_complete": orchestration_complete,
            "successful": successful,
            "failure_table_sha256": hashlib.sha256(
                (output_root / FAILURE_TABLE_FILENAME).read_bytes()
            ).hexdigest(),
        },
    )


def _run_units(
    plan: Any,
    units: Sequence[Any],
    *,
    output_root: Path | str,
    backend: StageCBackend | Callable[..., Any],
    device: str,
    resume: bool,
    estimate: Any,
    caps: Mapping[str, Any] | None,
    fault_injector: Callable[[str], Any] | str | Sequence[str] | None,
    allow_unissued_estimate: bool,
) -> BatchRunResult:
    """Shared batch path; preflight is intentionally the first mutation gate."""

    preflight = _passing_preflight(
        plan, estimate, allow_unissued_estimate=allow_unissued_estimate
    )
    # No mkdir, attempt allocation, backend/model call, or batch-control write
    # may occur above this point.
    root = _safe_output_path(output_root, allow_missing=True)
    root.mkdir(parents=True, exist_ok=True)
    # Attempts and backend calls are permitted only after the immutable batch
    # identity controls are durably published (or exactly revalidated).
    _publish_or_validate_batch_identity_controls(root, plan, preflight)
    observed_git_state = _runtime_git_state(plan=plan, backend=backend)
    results: list[UnitRunResult] = []
    for unit in units:
        results.append(
            run_execution_unit(
                unit, output_root=root, backend=backend, device=device,
                resume=resume, caps=caps, preflight=preflight,
                plan=plan, observed_git_state=observed_git_state,
                fault_injector=fault_injector,
            )
        )
    _write_batch_controls(root, plan, preflight, results)
    validation = validate_stage_c_batch(
        root, plan, observed_git_state=observed_git_state
    )
    orchestration_complete = validation.orchestration_complete
    successful = validation.successful
    # Reflect aggregate-cap/integrity failures in the durable batch status.
    if not validation.valid:
        status_path = root / BATCH_STATUS_FILENAME
        status = _strict_json(status_path)
        status["orchestration_complete"] = orchestration_complete
        status["successful"] = False
        status["validation_errors"] = list(validation.errors)
        _atomic_json_replace(status_path, status)
    return BatchRunResult(
        _plan_digest(plan), str(root), orchestration_complete, successful,
        tuple(results),
    )


def run_stage_c_sample(
    plan: Any,
    sample_id: str,
    *,
    output_root: Path | str,
    backend: StageCBackend | Callable[..., Any],
    device: str = "cpu",
    resume: bool = True,
    estimate: Any = None,
    caps: Mapping[str, Any] | None = None,
    fault_injector: Callable[[str], Any] | str | Sequence[str] | None = None,
    _allow_unissued_estimate: bool = False,
) -> BatchRunResult:
    """Run every planned unit for exactly one explicitly named sample."""

    if not isinstance(sample_id, str) or not sample_id:
        raise TypeError("sample_id must be a non-empty string")
    all_units = _plan_units(plan)
    planned_samples = {
        _unit_identity(unit)["sample_id"] for unit in all_units
    }
    if sample_id not in planned_samples:
        raise StageCRunnerError(f"sample_id is not in the execution plan: {sample_id}")
    if planned_samples != {sample_id}:
        raise StageCRunnerError(
            "run_stage_c_sample requires an execution plan containing exactly "
            "the requested sample; create an explicit single-sample run spec"
        )
    units = all_units
    return _run_units(
        plan, units, output_root=output_root, backend=backend, device=device,
        resume=resume, estimate=estimate, caps=caps,
        fault_injector=fault_injector,
        allow_unissued_estimate=_allow_unissued_estimate,
    )


def run_stage_c_batch(
    plan: Any,
    *,
    output_root: Path | str,
    backend: StageCBackend | Callable[..., Any],
    device: str = "cpu",
    resume: bool = True,
    estimate: Any = None,
    caps: Mapping[str, Any] | None = None,
    fault_injector: Callable[[str], Any] | str | Sequence[str] | None = None,
    _allow_unissued_estimate: bool = False,
) -> BatchRunResult:
    """Run a complete expanded plan, retaining every per-unit outcome."""

    units = _plan_units(plan)
    if not units:
        raise StageCRunnerError("execution plan contains no units")
    return _run_units(
        plan, units, output_root=output_root, backend=backend, device=device,
        resume=resume, estimate=estimate, caps=caps,
        fault_injector=fault_injector,
        allow_unissued_estimate=_allow_unissued_estimate,
    )


def _strict_json(path: Path) -> Any:
    return _strict_json_bytes(path.read_bytes(), context=str(path))


def _strict_json_bytes(payload: bytes, *, context: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key in {context}: {key}")
            value[key] = item
        return value

    return json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number in {context}: {token}")
        ),
    )


def _classify_attempts(
    unit: Any, unit_root: Path, *,
    expected_provenance_bindings: Mapping[str, Any],
    observed_git_state: Mapping[str, Any] | None,
    caps: Mapping[str, Any] | None,
) -> tuple[str, str | None, tuple[dict[str, Any], ...], list[str]]:
    from .stage_c_artifacts import attempt_directories, validate_execution_unit

    validations = tuple(
        validate_execution_unit(
            path, expected_unit_identity=_unit_identity(unit),
            expected_provenance_bindings=expected_provenance_bindings,
            observed_git_state=observed_git_state, caps=caps,
            require_runner_telemetry=True,
        )
        for path in attempt_directories(unit_root)
    )
    attempts = tuple(item.to_dict() for item in validations)
    resumable = [item for item in validations if item.resumable]
    errors: list[str] = []
    if len(resumable) > 1:
        errors.append("multiple valid complete attempts for one execution unit")
        return UNIT_TAMPERED, None, attempts, errors
    if len(resumable) == 1:
        item = resumable[0]
        return UNIT_SUCCEEDED, str(item.path), attempts, errors
    if not validations:
        return UNIT_MISSING, None, attempts, errors
    latest = validations[-1]
    if latest.status == "failed" and latest.valid and latest.terminal:
        return UNIT_FAILED, str(latest.path), attempts, errors
    if latest.status == "running" or not latest.terminal:
        return UNIT_INCOMPLETE, str(latest.path), attempts, errors
    return UNIT_TAMPERED, str(latest.path), attempts, list(latest.errors)


def _validation_caps(root: Path) -> dict[str, Any]:
    path = root / PREFLIGHT_FILENAME
    if not path.is_file():
        return {}
    value = _strict_json(path)
    if not isinstance(value, Mapping):
        return {}
    caps = value.get("caps")
    effective = caps.get("effective") if isinstance(caps, Mapping) else None
    return dict(effective) if isinstance(effective, Mapping) else {}


def validate_stage_c_batch(
    output_root: Path | str, plan: Any, *,
    observed_git_state: Mapping[str, Any] | None = None,
) -> BatchValidation:
    """Validate a batch by left-joining disk state to every expected unit."""

    root = _safe_output_path(output_root, allow_missing=False)
    units = _plan_units(plan)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        preflight_document = _strict_json(root / PREFLIGHT_FILENAME)
        _validated_preflight_binding(preflight_document, plan=plan)
        effective_caps = _validation_caps(root)
    except Exception as error:
        preflight_document = {}
        effective_caps = {}
        errors.append(f"invalid preflight.json: {error}")
    expected_bindings = (
        _expected_provenance_bindings(plan=plan, preflight=preflight_document)
        if preflight_document else {"configuration": {
            "execution_plan_sha256": _plan_digest(plan)
        }}
    )
    git_state = _runtime_git_state(
        plan=plan, observed_git_state=observed_git_state
    )
    plan_path = root / EXECUTION_PLAN_FILENAME
    if not plan_path.is_file():
        errors.append("missing execution_plan.json")
    else:
        try:
            persisted_plan = _strict_json(plan_path)
            if persisted_plan != _plan_dict(plan):
                errors.append(
                    "execution_plan.json does not match the expected plan"
                )
        except Exception as error:
            errors.append(f"invalid execution_plan.json: {error}")
    selected_l2_by_sample: dict[str, int] = {}
    observed_digests: set[str] = set()
    for unit in units:
        identity = _unit_identity(unit)
        digest = identity["canonical_digest"]
        observed_digests.add(digest)
        state, selected, attempts, unit_errors = _classify_attempts(
            unit, _unit_root(root, unit),
            expected_provenance_bindings=expected_bindings,
            observed_git_state=git_state, caps=effective_caps,
        )
        rows.append(
            {
                "unit_key": identity,
                "state": state,
                "terminal": state in {UNIT_SUCCEEDED, UNIT_FAILED},
                "complete": state == UNIT_SUCCEEDED,
                "selected_attempt": selected,
                "attempts": list(attempts),
                "errors": unit_errors,
            }
        )
        if selected is not None:
            selected_validation = next(
                (item for item in attempts if item.get("path") == selected), None
            )
            if selected_validation is not None:
                selected_l2_by_sample[identity["sample_id"]] = (
                    selected_l2_by_sample.get(identity["sample_id"], 0)
                    + int(selected_validation.get("l2_bytes", 0))
                )
        errors.extend(f"{digest}: {message}" for message in unit_errors)

    units_root = root / "units"
    if units_root.exists():
        for child in units_root.iterdir():
            if child.is_symlink():
                errors.append(f"forbidden unit symlink: {child}")
            elif child.is_dir() and child.name not in observed_digests:
                errors.append(f"unexpected execution unit directory: {child.name}")

    failure_path = root / FAILURE_TABLE_FILENAME
    if not failure_path.is_file():
        errors.append("missing failure_table.json")
    else:
        try:
            table = _strict_json(failure_path)
            table_rows = table.get("rows") if isinstance(table, Mapping) else None
            if not isinstance(table_rows, list):
                raise TypeError("failure table rows must be a list")
            table_digests = [
                row.get("unit_key", {}).get("canonical_digest")
                for row in table_rows if isinstance(row, Mapping)
            ]
            if len(table_rows) != len(units) or set(table_digests) != observed_digests:
                errors.append(
                    "failure table has missing or unexpected rows; it must contain "
                    "exactly one row per expected unit"
                )
            if len(table_digests) != len(set(table_digests)):
                errors.append("failure table contains duplicate unit rows")
            disk_by_digest = {
                row["unit_key"]["canonical_digest"]: row for row in rows
            }
            for table_row in table_rows:
                if not isinstance(table_row, Mapping):
                    errors.append("failure table contains a non-object row")
                    continue
                digest = table_row.get("unit_key", {}).get("canonical_digest")
                observed = disk_by_digest.get(digest)
                if observed is None:
                    continue
                expected_state = (
                    UNIT_SKIPPED
                    if table_row.get("state") == UNIT_SKIPPED
                    else observed["state"]
                )
                if expected_state == UNIT_SKIPPED:
                    expected_state = UNIT_SUCCEEDED
                table_state = table_row.get("state")
                if table_state == UNIT_SKIPPED:
                    table_state = UNIT_SUCCEEDED
                if table_state != expected_state:
                    errors.append(f"failure table state disagrees for unit {digest}")
                if table_row.get("selected_attempt") != observed["selected_attempt"]:
                    errors.append(f"failure table selected attempt disagrees for unit {digest}")
                selected = observed["selected_attempt"]
                expected_status_sha256 = None
                if isinstance(selected, str):
                    selected_status = Path(selected) / "status.json"
                    if selected_status.is_file() and not selected_status.is_symlink():
                        expected_status_sha256 = hashlib.sha256(
                            selected_status.read_bytes()
                        ).hexdigest()
                if table_row.get("selected_status_sha256") != expected_status_sha256:
                    errors.append(
                        f"failure table selected status hash disagrees for unit {digest}"
                    )
        except Exception as error:
            errors.append(f"invalid failure_table.json: {error}")

    status_path = root / BATCH_STATUS_FILENAME
    if not status_path.is_file():
        errors.append("missing batch_status.json")
    else:
        try:
            status = _strict_json(status_path)
            if status.get("plan_digest") != _plan_digest(plan):
                errors.append("batch status plan digest mismatch")
            expected_hash = hashlib.sha256(failure_path.read_bytes()).hexdigest()
            if status.get("failure_table_sha256") != expected_hash:
                errors.append("batch status failure-table hash mismatch")
            if status.get("expected_unit_count") != len(units):
                errors.append("batch status expected-unit count mismatch")
        except Exception as error:
            errors.append(f"invalid batch_status.json: {error}")

    table_complete = not any(
        "failure table" in message.lower() for message in errors
    )
    orchestration_complete = (
        all(row["terminal"] for row in rows) and table_complete
    )
    successful = bool(rows) and all(row["complete"] for row in rows)
    if status_path.is_file():
        try:
            status = _strict_json(status_path)
            if status.get("orchestration_complete") != orchestration_complete:
                errors.append("batch status orchestration_complete is false or stale")
            if status.get("successful") != successful:
                errors.append("batch status successful is false or stale")
        except Exception:
            pass
    l2_cap = effective_caps.get("bytes_per_audit_sample_l2_max")
    if type(l2_cap) is int:
        for sample_id, byte_count in sorted(selected_l2_by_sample.items()):
            if byte_count > l2_cap:
                errors.append(
                    f"audit sample {sample_id} L2 bytes {byte_count} exceed cap {l2_cap}"
                )
    total_cap = effective_caps.get("stage_c_c8_c9_total_artifact_bytes_max")
    if type(total_cap) is int:
        try:
            from .stage_c_resources import measure_artifact_tree_bytes

            total_actual = measure_artifact_tree_bytes(root).total_bytes
        except Exception as error:
            errors.append(f"cannot measure complete batch artifact tree: {error}")
        else:
            if total_actual > total_cap:
                errors.append(
                    f"batch regular-file bytes {total_actual} exceed Stage C cap {total_cap}"
                )
    # Any hard-cap or control-integrity failure prevents both completion and
    # success even when individual unit attempt markers are terminal.
    if errors:
        if any(
            token in message.lower()
            for message in errors
            for token in ("cap", "preflight", "failure table", "batch status")
        ):
            orchestration_complete = False
        successful = False
    integrity_states = {UNIT_MISSING, UNIT_INCOMPLETE, UNIT_TAMPERED}
    valid = not errors and all(row["state"] not in integrity_states for row in rows)
    return BatchValidation(
        _plan_digest(plan), valid, orchestration_complete, successful,
        tuple(rows), tuple(errors),
    )


def _validated_l0(attempt: Path) -> dict[str, Any]:
    """Read L0 only.  This function intentionally has no L3 code path."""

    manifest = _strict_json(attempt / "manifest.json")
    records = manifest.get("artifacts", [])
    candidates = [
        record for record in records
        if isinstance(record, Mapping)
        and record.get("level") == "L0"
        and record.get("kind") == "result"
    ]
    if len(candidates) != 1:
        raise StageCBatchValidationError(
            f"attempt {attempt} must contain exactly one L0 result artifact"
        )
    path = attempt / str(candidates[0]["path"])
    document = _strict_json(path)
    if not isinstance(document, dict):
        raise StageCBatchValidationError("L0 result must be a JSON object")
    return document


def aggregate_stage_c_batch(
    output_root: Path | str,
    plan: Any,
    *,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Emit only validated L0 long rows; no selection or statistics occur."""

    validation = validate_stage_c_batch(output_root, plan)
    validation.require_valid()
    rows: list[dict[str, Any]] = []
    for row in validation.rows:
        identity = _mapping(row["unit_key"], "aggregate unit key")
        state = str(row["state"])
        record: dict[str, Any] = {
            "unit_key": identity,
            "state": state,
            "failure": None,
            "scalars": {},
        }
        if state == UNIT_SUCCEEDED:
            selected = row.get("selected_attempt")
            if not isinstance(selected, str):
                raise StageCBatchValidationError("successful row has no selected attempt")
            l0 = _validated_l0(Path(selected))
            record["failure"] = l0.get("failure")
            record["scalars"] = l0.get("scalars", {})
            record["resources"] = l0.get("resources", {})
        elif state == UNIT_FAILED:
            # Failure remains an explicit long-table row.  It is never dropped
            # or imputed to a numeric zero.
            selected = row.get("selected_attempt")
            if isinstance(selected, str):
                status = _strict_json(Path(selected) / "status.json")
                record["failure"] = status.get("failure")
        rows.append(record)
    document = {
        "schema_version": 1,
        "document_type": "stage_c_l0_long_table",
        "plan_digest": validation.plan_digest,
        "row_count": len(rows),
        "orchestration_complete": validation.orchestration_complete,
        "successful": validation.successful,
        "source_levels": ["L0"],
        "selection_performed": False,
        "bootstrap_performed": False,
        "decision_performed": False,
        "rows": rows,
    }
    if output_path is not None:
        destination = _safe_output_path(output_path, allow_missing=True)
        _atomic_json_replace(destination, document)
    return document


__all__ = [
    "BatchRunResult", "BatchValidation", "ProductionStageCBackend",
    "StageCBackend", "StageCBackendError", "StageCBatchValidationError",
    "StageCPartialEvidence", "StageCScientificInvalidity",
    "StageCPreflightError", "StageCRunnerError", "UnitPayload",
    "UnitRunResult", "aggregate_stage_c_batch", "run_execution_unit",
    "run_stage_c_batch", "run_stage_c_sample",
    "validate_stage_c_batch",
]
