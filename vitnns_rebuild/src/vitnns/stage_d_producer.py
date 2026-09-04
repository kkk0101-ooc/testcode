"""Stage D image-group producer and Stage C scientific-backend adapter.

This module owns only the narrow D4 boundary: validate one already-planned
image group, execute its NNS/matched pair in order, publish control-only
terminal notifications, and release all sample-scoped backend state.  Runtime
claims, attempts and artifact publication remain the caller's responsibility.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType, SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .fidelity_seed import derive_fidelity_seed
from .stage_c_runner import (
    ProductionStageCBackend, StageCBackendError, StageCPartialEvidence,
    StageCScientificInvalidity,
)
from .stage_d_plan import (
    BLOCK_INDEX, CURVE_FRACTIONS, DIRECTIONS, FIDELITY_SCALE, FINE_CELL_COUNT,
    FROZEN_FIDELITY_CONTEXT_DIGEST, FROZEN_MODEL_CONTEXT_DIGEST,
    FROZEN_PLAN_CONTEXT_DIGEST, FROZEN_TARGET_CONTEXT_DIGEST, HEAD_INDEX,
    METHOD_IDS, NNS_METHOD_ID, TARGET_ID,
    TIE_REPEAT_INDICES, StageDCurvePoint, StageDImageGroup,
    StageDExecutionPlan, StageDMapCoordinate, StageDUnit,
)


STAGE_D_FIDELITY_SEED_NAMESPACE = "stage-d-fidelity|20260821|"
STAGE_D_TERMINAL_STATES = frozenset({
    "TERMINAL_SUCCESS",
    "TERMINAL_SCIENTIFIC_FAILURE",
    "SKIPPED_VALID_TERMINAL",
})
STAGE_D_CONTROL_STATES = STAGE_D_TERMINAL_STATES | frozenset({
    "RETRY_PENDING", "BLOCKED",
})


class StageDProducerError(RuntimeError):
    """A group, backend, or unit result crossed the D4 contract."""


class StageDGroupBusyError(StageDProducerError):
    """The same backend is already owned by another image group."""


@runtime_checkable
class StageDGroupBackend(Protocol):
    """Backend lifecycle required by the image-group producer."""

    def image_scope(
        self, sample_id: str, *, device: str
    ) -> Any: ...

    def release_sample(self, sample_id: str, device: str) -> None: ...


@dataclass(frozen=True, slots=True)
class StageDGroupRunResult:
    """Control-plane result; scientific payloads deliberately stay opaque."""

    group_digest: str
    pair_binding_digest: str
    state: str
    unit_results: tuple[Any, ...]
    cache_released: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_digest": self.group_digest,
            "pair_binding_digest": self.pair_binding_digest,
            "state": self.state,
            "unit_results": [
                _control_projection(item, f"unit_results[{index}]")
                for index, item in enumerate(self.unit_results)
            ],
            "cache_released": self.cache_released,
        }


@dataclass(frozen=True, slots=True)
class _StageDBackendUnit:
    """Stage-C-compatible scientific view of one typed Stage D unit."""

    stage_d_unit: StageDUnit
    key: Mapping[str, Any]
    sample: Mapping[str, Any]
    comparison_line: str
    audit: bool
    work_items: tuple[Any, ...]
    pair_binding_digest: str


class StageDBackendPayload:
    """Stage D payload accepting only fine 56x56 L1 maps and no L3."""

    __slots__ = (
        "l0", "l1", "l2", "l3", "provenance", "resources",
        "lifecycle",
    )

    def __init__(
        self, *, l0: Mapping[str, Any], l1: Mapping[str, np.ndarray],
        l2: Mapping[str, Any] | None = None,
        l3: Mapping[str, bytes] | None = None,
        provenance: Mapping[str, Any] | None = None,
        resources: Mapping[str, Any] | None = None,
        lifecycle: Mapping[str, Any] | None = None,
    ) -> None:
        values = {
            "l0": l0, "l1": l1, "l2": {} if l2 is None else l2,
            "l3": {} if l3 is None else l3,
            "provenance": {} if provenance is None else provenance,
            "resources": {} if resources is None else resources,
            "lifecycle": {} if lifecycle is None else lifecycle,
        }
        if any(not isinstance(value, Mapping) for value in values.values()):
            raise TypeError("StageDBackendPayload fields must be mappings")
        if values["l3"]:
            raise StageDProducerError("Stage D retention forbids every L3 artifact")
        for name, raw in values["l1"].items():
            array = np.asarray(raw)
            if (
                not isinstance(name, str) or not name
                or array.dtype != np.float32
                or array.shape != (56, 56)
                or not np.isfinite(array).all()
            ):
                raise StageDProducerError(
                    "Stage D L1 maps must be finite float32 with shape (56,56)"
                )
        for name, value in values.items():
            object.__setattr__(self, name, value)


# Temporary compatibility alias for direct callers of the initial D4 adapter.
# Artifact persistence always imports its distinct payload type explicitly.
StageDUnitPayload = StageDBackendPayload


_RESOURCE_KEYS = (
    "elapsed_seconds", "forward_calls", "backward_calls",
    "l0_l1_payload_bytes", "l2_payload_bytes",
    "control_and_provenance_bytes", "cuda_peak_reserved_bytes",
)
_ZERO_RESOURCES: dict[str, int | float] = {
    "elapsed_seconds": 0.0, "forward_calls": 0, "backward_calls": 0,
    "l0_l1_payload_bytes": 0, "l2_payload_bytes": 0,
    "control_and_provenance_bytes": 0, "cuda_peak_reserved_bytes": 0,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_ID_RE = re.compile(r"^attempt-(?:000001|000002)$")
_RETRYABLE_INFRASTRUCTURE_CODES = frozenset({
    "SCHEDULER_PREEMPTION", "EXTERNAL_PROCESS_TERMINATION", "HOST_LOSS",
    "GPU_DEVICE_LOSS_OR_RESET", "TRANSIENT_STORAGE_IO_BEFORE_RESULT_PUBLISH",
    "WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH",
    "STALE_CLAIM_OWNER_CONFIRMED_DEAD",
})
_SCIENTIFIC_NA_CODES = frozenset({
    "empty_valid_region", "shape_mismatch", "non_finite",
    "degenerate_direction", "degenerate_target_delta",
    "missing_artifact",
})
_SCIENTIFIC_FAILURE_CODES = _SCIENTIFIC_NA_CODES | frozenset({
    "audit_repeatability_failed", "endpoint_parity_failed",
    "ig_conservation_failed", "matched_token_conservation_failed",
})


@dataclass(frozen=True, slots=True)
class StageDPartialEvidence:
    """Ordered scientific components completed before invalidity arose."""

    l0_scalars: Mapping[str, Any] | None = None
    metric_ready_map: np.ndarray | None = None
    fidelity_curves: Any | None = None
    audit_evidence: Mapping[str, Any] | None = None


class StageDScientificFailure(RuntimeError):
    """Explicit scientific invalidity carrying only bounded failure evidence."""

    failure_class = "SCIENTIFIC_INVALIDITY"

    def __init__(
        self, code: str, *, phase: str, l0_scalars: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        partial_evidence: StageDPartialEvidence | StageDBackendPayload | Any | None = None,
    ) -> None:
        if code not in _SCIENTIFIC_FAILURE_CODES:
            raise ValueError("scientific failure code is outside the closed set")
        if not isinstance(phase, str) or not phase:
            raise TypeError("scientific failure phase must be non-empty")
        self.failure_code = code
        self.phase = phase
        self.l0_scalars = {} if l0_scalars is None else dict(l0_scalars)
        self.details = None if details is None else dict(details)
        self.partial_evidence = partial_evidence
        super().__init__(f"Stage D scientific invalidity: {code}")


class StageDInfrastructureFailure(RuntimeError):
    """Explicit pre-publication infrastructure interruption."""

    failure_class = "INFRASTRUCTURE_FAILURE"

    def __init__(self, code: str, resources: Mapping[str, Any]) -> None:
        if code not in _RETRYABLE_INFRASTRUCTURE_CODES:
            raise ValueError("infrastructure failure code is outside the allowlist")
        self.failure_code = code
        self.resources = _resources(resources)
        self.result_manifest_published = False
        self.published_l0_l1_payload_count = 0
        super().__init__(f"Stage D infrastructure interruption: {code}")


def _plain(value: Any, context: str) -> Any:
    converter = getattr(value, "to_dict", None)
    if callable(converter) and not isinstance(value, Mapping):
        value = converter()
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise StageDProducerError(f"{context} contains a non-finite value")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise StageDProducerError(
                    f"{context} keys must be non-empty strings"
                )
            result[key] = _plain(child, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _plain(child, f"{context}[{index}]")
            for index, child in enumerate(value)
        ]
    raise StageDProducerError(f"{context} is not JSON-compatible")


def _resources(value: Any) -> dict[str, int | float]:
    if value is None:
        return dict(_ZERO_RESOURCES)
    if not isinstance(value, Mapping):
        converter = getattr(value, "to_dict", None)
        value = converter() if callable(converter) else value
    if not isinstance(value, Mapping):
        raise TypeError("resources must be a mapping")
    result = dict(_ZERO_RESOURCES)
    aliases = {
        "wall_time_seconds": "elapsed_seconds",
        "cuda_peak_reserved": "cuda_peak_reserved_bytes",
    }
    nested_calls = value.get("calls", {})
    nested_cuda = value.get("cuda", {})
    flattened = dict(value)
    if isinstance(nested_calls, Mapping):
        flattened.update({
            "forward_calls": nested_calls.get("forward_calls", 0),
            "backward_calls": nested_calls.get("backward_calls", 0),
        })
    if isinstance(nested_cuda, Mapping):
        flattened["cuda_peak_reserved_bytes"] = nested_cuda.get(
            "peak_reserved_bytes", 0
        ) or 0
    for raw_name, raw in flattened.items():
        name = aliases.get(raw_name, raw_name)
        if name not in result:
            continue
        if name == "elapsed_seconds":
            if type(raw) not in {int, float} or not np.isfinite(raw) or raw < 0:
                raise StageDProducerError("elapsed_seconds is invalid")
            result[name] = float(raw)
        elif type(raw) is not int or raw < 0:
            raise StageDProducerError(f"resource {name} is invalid")
        else:
            result[name] = raw
    return result


def _measured_resources(record: Any) -> dict[str, int | float]:
    return {
        "elapsed_seconds": float(record.wall_time_seconds or 0.0),
        "forward_calls": int(record.forward_calls),
        "backward_calls": int(record.backward_calls),
        "l0_l1_payload_bytes": 0, "l2_payload_bytes": 0,
        "control_and_provenance_bytes": 0,
        "cuda_peak_reserved_bytes": int(record.cuda_peak_reserved_bytes or 0),
    }


def _artifact_resource_bytes(
    records: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    l0_l1 = sum(
        int(item["bytes"]) for item in records
        if item["level"] in {"L0", "L1"}
        and item["kind"] != "provenance"
    )
    l2 = sum(int(item["bytes"]) for item in records if item["level"] == "L2")
    return l0_l1, l2


def _control_bytes(attempt_path: Path) -> int:
    return sum(
        path.stat().st_size for path in (
            attempt_path / "status.json", attempt_path / "manifest.json",
            attempt_path / "L0" / "provenance.json",
        )
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _control_state(value: Any, context: str) -> str:
    state = _field(value, "state", None)
    if hasattr(state, "value"):
        state = state.value
    if not isinstance(state, str):
        raise StageDProducerError(f"{context}.state must be a string")
    normalized = state.upper()
    if normalized not in STAGE_D_CONTROL_STATES:
        raise StageDProducerError(
            f"{context}.state is outside the closed Stage D control states"
        )
    return normalized


def _control_projection(value: Any, context: str) -> dict[str, Any]:
    """Return an allowlisted control view without inspecting science values."""

    state = _control_state(value, context)
    projection: dict[str, Any] = {"state": state}
    for name in (
        "unit_digest", "pair_binding_digest", "attempt_digest",
        "failure_class", "failure_code",
    ):
        item = _field(value, name, None)
        if item is not None:
            if not isinstance(item, str) or not item:
                raise StageDProducerError(f"{context}.{name} must be a string")
            projection[name] = item
    return projection


def _validate_group(group: Any) -> StageDImageGroup:
    if not isinstance(group, StageDImageGroup):
        raise TypeError("group must be a typed StageDImageGroup")
    # Reconstructing from the wire form checks every nested content digest.
    try:
        checked = StageDImageGroup.from_dict(group.to_dict())
    except Exception as error:
        raise StageDProducerError(f"invalid Stage D image group: {error}") from error
    if tuple(unit.method_id for unit in checked.units) != METHOD_IDS:
        raise StageDProducerError("group units must be ordered NNS then matched")
    if any(
        unit.pair_binding_digest != checked.pair_binding.canonical_digest
        for unit in checked.units
    ):
        raise StageDProducerError("group units do not share one pair binding")
    pair = checked.pair_binding
    if (
        pair.image_sha256 != checked.image_sha256
        or pair.model_context_digest != FROZEN_MODEL_CONTEXT_DIGEST
        or pair.target_context_digest != FROZEN_TARGET_CONTEXT_DIGEST
        or pair.fidelity_context_digest != FROZEN_FIDELITY_CONTEXT_DIGEST
        or pair.plan_context_digest != FROZEN_PLAN_CONTEXT_DIGEST
    ):
        raise StageDProducerError(
            "image/model/target/baseline/fractions/tie/operator binding drift"
        )
    return group


def _unit_coordinates(
    group: StageDImageGroup, unit: StageDUnit,
) -> tuple[tuple[StageDMapCoordinate, ...], tuple[StageDCurvePoint, ...]]:
    maps = tuple(
        item for item in group.map_coordinates
        if item.unit_digest == unit.canonical_digest
    )
    curves = tuple(
        item for item in group.iter_curve_points()
        if item.unit_digest == unit.canonical_digest
    )
    expected_map_kinds = (
        ("primary_clean_map", "audit_repeat_clean_map")
        if unit.audit else ("primary_clean_map",)
    )
    if tuple(item.map_kind for item in maps) != expected_map_kinds:
        raise StageDProducerError(
            "unit must have one primary clean map and exactly one audit repeat"
            if unit.audit else "unit must have exactly one primary clean map"
        )
    expected = [
        (direction, repeat, fraction_index, CURVE_FRACTIONS[fraction_index])
        for direction in DIRECTIONS
        for repeat in TIE_REPEAT_INDICES
        for fraction_index in range(len(CURVE_FRACTIONS))
    ]
    observed = [
        (item.direction, item.repeat_index, item.fraction_index, item.fraction)
        for item in curves
    ]
    if observed != expected or len({item.canonical_digest for item in curves}) != 210:
        raise StageDProducerError(
            "unit curve coordinates must be exactly 2*5*21 in frozen order"
        )
    for item in (*maps, *curves):
        if (
            item.pair_binding_digest != group.pair_binding.canonical_digest
            or item.sample_id != group.sample_id
            or item.selection_order != group.selection_order
            or item.method_id != unit.method_id
            or item.block_index != BLOCK_INDEX
            or item.head_index != HEAD_INDEX
            or item.target_id != TARGET_ID
        ):
            raise StageDProducerError(
                "unit work coordinate escaped its frozen pair context"
            )
    if any(
        item.fidelity_scale != FIDELITY_SCALE
        or item.fine_cell_count != FINE_CELL_COUNT
        for item in curves
    ):
        raise StageDProducerError("only fine 56x56 fidelity work is allowed")
    return maps, curves


def _backend_unit(group: StageDImageGroup, unit: StageDUnit) -> _StageDBackendUnit:
    """Map Stage D coordinates to the already-tested Stage C math dispatcher."""

    maps, curves = _unit_coordinates(group, unit)
    work_items: list[Any] = []
    for item in maps:
        work_items.append(SimpleNamespace(
            kind="map",
            input_variant=(
                "clean" if item.map_kind == "primary_clean_map"
                else "repeat_clean"
            ),
            repeat_index=item.clean_run_index,
        ))
    for item in curves:
        work_items.append(SimpleNamespace(
            kind="fidelity_curve_point", input_variant="clean",
            repeat_index=None, fidelity_scale=item.fidelity_scale,
            curve_direction=item.direction,
            ranking_kind="target_oriented_positive_tie_permuted",
            replicate_index=item.repeat_index,
            fraction_index=item.fraction_index, fraction=float(item.fraction),
            pairing_block_index=item.block_index,
            pairing_head_index=item.head_index,
            canonical_digest=item.canonical_digest,
        ))
    key = MappingProxyType({
        "schema_version": 1, "sample_id": unit.sample_id,
        "method_id": unit.method_id,
        "config_id": f"block_{unit.block_index:02d}_head_{unit.head_index:02d}",
        "block_index": unit.block_index, "head_index": unit.head_index,
        "target_ref": unit.target_id,
    })
    from .stage_c_provenance import unit_identity_digest

    key = MappingProxyType({**key, "canonical_digest": unit_identity_digest({
        **key, "canonical_digest": "0" * 64,
    })})
    sample = MappingProxyType({
        "sample_id": group.sample_id, "selection_order": group.selection_order,
        "image_sha256": group.image_sha256, "audit": group.audit,
        "relative_path": group.relative_path,
    })
    return _StageDBackendUnit(
        stage_d_unit=unit, key=key, sample=sample, comparison_line="value_path",
        audit=unit.audit, work_items=tuple(work_items),
        pair_binding_digest=unit.pair_binding_digest,
    )


class ProductionStageDBackend(ProductionStageCBackend):
    """Canonical Stage D facade over the registered Stage C dispatcher."""

    DEFAULT_FIDELITY_SEED_NAMESPACE = STAGE_D_FIDELITY_SEED_NAMESPACE

    def __init__(
        self, *, trace_mode: str | None = None,
        trace_builder: Callable[..., Any] | None = None,
        stage_d_model_binding: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        requested = kwargs.pop(
            "fidelity_seed_namespace", STAGE_D_FIDELITY_SEED_NAMESPACE
        )
        if requested != STAGE_D_FIDELITY_SEED_NAMESPACE:
            raise StageDProducerError(
                "ProductionStageDBackend requires the frozen Stage D seed namespace"
            )
        if trace_mode is not None:
            from .stage_d_engine import validate_trace_mode

            trace_mode = validate_trace_mode(trace_mode)
        if trace_builder is not None and not callable(trace_builder):
            raise TypeError("trace_builder must be callable or None")
        self._stage_d_configured_trace_mode = trace_mode
        self._stage_d_trace_builder = trace_builder
        self._stage_d_model_binding = (
            None if stage_d_model_binding is None
            else MappingProxyType(_plain(
                stage_d_model_binding, "stage_d_model_binding"
            ))
        )
        super().__init__(
            fidelity_seed_namespace=STAGE_D_FIDELITY_SEED_NAMESPACE, **kwargs
        )

    def _bound_model_record(self) -> dict[str, Any]:
        if self._stage_d_model_binding is not None:
            return dict(self._stage_d_model_binding)
        return super()._bound_model_record()

    def adapt_unit(
        self, group: StageDImageGroup, unit: StageDUnit
    ) -> Any:
        checked = _validate_group(group)
        selected = next((item for item in checked.units if item == unit), None)
        if selected is None:
            raise StageDProducerError("unit does not belong to the image group")
        return _backend_unit(checked, selected)

    @contextmanager
    def image_group_scope(
        self, group: StageDImageGroup, *, device: str
    ) -> Iterator["ProductionStageDBackend"]:
        checked = _validate_group(group)
        with super().image_scope(checked.sample_id, device=device):
            if hasattr(self, "_active_stage_d_group"):
                raise StageDGroupBusyError(
                    "ProductionStageDBackend already owns a Stage D image group"
                )
            self._active_stage_d_group = checked
            try:
                yield self
            finally:
                del self._active_stage_d_group

    def run_stage_d_unit(
        self, group: StageDImageGroup, unit: StageDUnit, *, device: str,
        call_counter: Any, trace_mode: str | None = None,
    ) -> StageDBackendPayload:
        active = getattr(self, "_active_stage_d_group", None)
        if active is not None and active.canonical_digest != group.canonical_digest:
            raise StageDProducerError(
                "requested unit belongs to a different active image group"
            )
        return self.run(
            unit, device=device, call_counter=call_counter,
            trace_mode=trace_mode,
        )

    def _retained_metric_array(self, result: Any, response: Any) -> np.ndarray:
        if response is None:
            raise StageDProducerError(
                "Stage D fine-map retention requires a resolved target response"
            )
        from .stage_c_fidelity import FidelityScale, method_unit_masses

        masses = method_unit_masses(result, response, FidelityScale.FINE_4).masses
        return np.array(masses, dtype=np.float32, copy=True)

    def _retain_repeat_clean_maps(self) -> bool:
        return True

    def _make_unit_payload(
        self, *, l0: Mapping[str, Any], l1: Mapping[str, np.ndarray],
        l2: Mapping[str, Any], l3: Mapping[str, bytes] | None = None,
        provenance: Mapping[str, Any] | None = None,
        resources: Mapping[str, Any] | None = None,
        lifecycle: Mapping[str, Any] | None = None,
    ) -> StageDBackendPayload:
        return StageDBackendPayload(
            l0=l0, l1=l1, l2=l2, l3=l3, provenance=provenance,
            resources=resources, lifecycle=lifecycle,
        )

    def _ig_execution_summary(self, result: Any) -> dict[str, Any] | None:
        """Skip attribution-trace serialization for canonical fast units."""

        if getattr(self, "_active_stage_d_trace_mode", None) == "fast":
            return None
        return super()._ig_execution_summary(result)

    def _method_result_dict(self, result: Any) -> dict[str, Any]:
        """Avoid materializing NNS provenance JSON on the fast path."""

        if (
            getattr(self, "_active_stage_d_trace_mode", None) == "fast"
            and getattr(result, "method_id", None) == NNS_METHOD_ID
        ):
            na = getattr(result, "na", None)
            comparison_line = getattr(result, "comparison_line", None)
            sign_semantics = getattr(result, "sign_semantics", None)
            return {
                "method_id": result.method_id,
                "method_version": result.method_version,
                "comparison_line": getattr(
                    comparison_line, "value", comparison_line
                ),
                "target_ref": result.fidelity_target,
                "status": "NA" if na is not None else "OK",
                "na": None if na is None else na.to_dict(),
                "projection": result.projection,
                "sign_semantics": getattr(
                    sign_semantics, "value", sign_semantics
                ),
            }
        return super()._method_result_dict(result)

    def run(
        self, unit: Any, *, device: str, call_counter: Any,
        trace_mode: str | None = None,
    ) -> StageDBackendPayload:
        if isinstance(unit, StageDUnit):
            group = getattr(self, "_active_stage_d_group", None)
            if group is None:
                raise StageDProducerError(
                    "a StageDUnit may run only inside image_group_scope"
                )
            unit = self.adapt_unit(group, unit)
        from .stage_d_engine import apply_trace_policy, resolve_trace_mode

        selected_trace_mode = resolve_trace_mode(
            unit, trace_mode=trace_mode,
            configured_mode=self._stage_d_configured_trace_mode,
        )
        if hasattr(self, "_active_stage_d_trace_mode"):
            raise StageDProducerError(
                "ProductionStageDBackend does not support concurrent trace execution"
            )
        self._active_stage_d_trace_mode = selected_trace_mode
        # UnitPayload performs Stage C's 14x14 validation, so invoke the
        # composed dispatcher directly under the same integrity/counter path
        # and rely on this subclass's StageDUnitPayload factory.
        try:
            try:
                payload = super().run(
                    unit, device=device, call_counter=call_counter
                )
            except StageCScientificInvalidity as error:
                partial = error.partial_evidence
                if partial is not None:
                    partial = StageDPartialEvidence(
                        l0_scalars=partial.l0_scalars,
                        metric_ready_map=partial.metric_ready_map,
                        fidelity_curves=partial.fidelity_curves,
                        audit_evidence=partial.audit_evidence,
                    )
                raise StageDScientificFailure(
                    error.failure_code, phase=error.phase,
                    l0_scalars=error.l0_scalars, details=error.details,
                    partial_evidence=partial,
                ) from error
            return apply_trace_policy(
                payload, unit=unit, trace_mode=selected_trace_mode,
                trace_builder=self._stage_d_trace_builder,
            )
        finally:
            del self._active_stage_d_trace_mode


def _context_field(context: Any, name: str) -> Any:
    if isinstance(context, Mapping):
        if name in context:
            return context[name]
    else:
        try:
            return getattr(context, name)
        except AttributeError:
            pass
    raise StageDProducerError(f"execution context lacks {name}")


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StageDProducerError(f"{context} must be a lowercase SHA-256")
    return value


def _validated_execution_inputs(
    unit: StageDUnit, *, group: StageDImageGroup, plan: StageDExecutionPlan,
    output_root: Path | str, context: Any, claim: Mapping[str, Any],
    retry_authorization: Mapping[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    checked_group = _validate_group(group)
    if not isinstance(plan, StageDExecutionPlan):
        raise TypeError("plan must be a typed StageDExecutionPlan")
    if unit not in checked_group.units or checked_group not in plan.groups:
        raise StageDProducerError("unit/group is not a member of the supplied plan")
    formal = _context_field(context, "formal")
    if type(formal) is not bool:
        raise StageDProducerError("execution context formal flag is invalid")
    if isinstance(context, Mapping):
        raise StageDProducerError(
            "direct execution requires a runtime-issued opaque context"
        )
    try:
        from . import stage_d_runtime as _stage_d_runtime

        issued = _stage_d_runtime._is_runtime_execution_capability(context)
    except (AttributeError, ImportError):
        issued = False
    if not issued:
        raise StageDProducerError(
            "formal execution context was not sealed by the Stage D runtime"
            if formal else
            "synthetic execution context was not issued by the D4 runtime"
        )
    if formal:
        if (
            _context_field(context, "execution_authorization")
            != "AUTHORIZED_STAGE_D_FIDELITY_SCOPE"
        ):
            raise StageDProducerError(
                "formal execution context lacks the D7 authorization binding"
            )
    else:
        if (
            _context_field(context, "authorization_scope")
            != "SYNTHETIC_D4_TEST_ONLY"
            or _context_field(context, "execution_authorization") != "NOT_ISSUED"
        ):
            raise StageDProducerError(
                "synthetic execution context crossed its authority boundary"
            )
    if _context_field(context, "plan_digest") != plan.canonical_digest:
        raise StageDProducerError("execution context plan binding drifted")
    shard_count = _context_field(context, "shard_count")
    shard_mapping = _context_field(context, "shard_mapping")
    if type(shard_count) is not int or shard_count <= 0:
        raise StageDProducerError("execution context shard_count is invalid")
    try:
        shard_id = shard_mapping[unit.selection_order]
    except (KeyError, IndexError, TypeError) as error:
        raise StageDProducerError("execution context shard mapping is incomplete") from error
    if shard_id != unit.selection_order % shard_count:
        raise StageDProducerError("execution context shard mapping drifted")

    required_claim = {
        "owner_token", "owner_token_digest", "group_owner_token_digest",
        "unit_digest", "group_digest", "attempt_id",
        "attempt_number", "attempt_path",
    }
    if not isinstance(claim, Mapping) or set(claim) != required_claim:
        raise StageDProducerError("claim fields are incomplete or unknown")
    if (
        claim["unit_digest"] != unit.canonical_digest
        or claim["group_digest"] != group.canonical_digest
        or claim["attempt_number"] not in {1, 2}
        or claim["attempt_id"] != f"attempt-{claim['attempt_number']:06d}"
        or _ATTEMPT_ID_RE.fullmatch(claim["attempt_id"]) is None
    ):
        raise StageDProducerError("claim coordinate/attempt binding drifted")
    owner = claim["owner_token"]
    if not isinstance(owner, str) or not owner:
        raise StageDProducerError("claim owner token is invalid")
    expected_owner = hashlib.sha256(owner.encode("utf-8")).hexdigest()
    if claim["owner_token_digest"] != expected_owner:
        raise StageDProducerError("claim owner token digest mismatch")
    _sha256(claim["group_owner_token_digest"], "claim group owner digest")

    root = Path(os.path.abspath(os.fspath(output_root)))
    attempt = Path(os.path.abspath(os.fspath(claim["attempt_path"])))
    expected = root / "units" / unit.canonical_digest / "attempts" / claim["attempt_id"]
    if attempt != expected:
        raise StageDProducerError("claim attempt path escaped output_root/unit")
    if retry_authorization is None:
        if claim["attempt_number"] != 1:
            raise StageDProducerError("attempt two requires retry authorization")
    else:
        retry = _plain(retry_authorization, "retry authorization")
        if not isinstance(retry, Mapping):
            raise StageDProducerError("retry authorization must be a mapping")
        if (
            claim["attempt_number"] != 2
            or retry.get("unit_digest") != unit.canonical_digest
            or retry.get("next_attempt_number") != 2
            or retry.get("failure_code") not in _RETRYABLE_INFRASTRUCTURE_CODES
        ):
            raise StageDProducerError("retry authorization binding drifted")
    return attempt, {"shard_id": shard_id, "shard_count": shard_count}


def _curve_array_from_records(
    records: Any, unit: StageDUnit,
) -> np.ndarray:
    if isinstance(records, np.ndarray):
        array = np.asarray(records)
        if array.dtype != np.float64 or array.shape != (2, 5, 21):
            raise StageDProducerError(
                "fidelity curve array must be float64 shape (2,5,21)"
            )
        if not np.isfinite(array).all():
            raise StageDProducerError("fidelity curve array contains non-finite values")
        return np.array(array, copy=True)
    if not isinstance(records, (list, tuple)) or len(records) != 10:
        raise StageDProducerError("backend must return exactly ten fidelity curves")
    expected = [
        (direction, repeat)
        for direction in DIRECTIONS for repeat in TIE_REPEAT_INDICES
    ]
    values: list[np.ndarray] = []
    for index, (record, coordinate) in enumerate(zip(records, expected)):
        if not isinstance(record, Mapping):
            raise StageDProducerError(f"fidelity curve {index} must be a mapping")
        direction = record.get("direction", record.get("curve_direction"))
        repeat = record.get("replicate", record.get("repeat_index"))
        fractions = record.get("curve_fractions")
        responses = record.get("responses")
        if (
            record.get("status", "OK") != "OK"
            or record.get("method_id") != unit.method_id
            or record.get("comparison_line") != "value_path"
            or record.get("target_ref") != unit.target_id
            or record.get("scale") != FIDELITY_SCALE
            or (direction, repeat) != coordinate
            or record.get("sample_id") != unit.sample_id
            or record.get("block") != unit.block_index
            or record.get("head") != unit.head_index
            or fractions != [float(item) for item in CURVE_FRACTIONS]
            or record.get("seed_namespace", STAGE_D_FIDELITY_SEED_NAMESPACE)
            != STAGE_D_FIDELITY_SEED_NAMESPACE
            or record.get("seed") != stage_d_fidelity_seed(
                unit.sample_id, unit.block_index, unit.head_index, repeat
            )
        ):
            raise StageDProducerError(
                f"fidelity curve {index} escaped its frozen Stage D coordinate"
            )
        array = np.asarray(responses, dtype=np.float64)
        if array.shape != (21,) or not np.isfinite(array).all():
            raise StageDProducerError(
                f"fidelity curve {index} responses must be 21 finite values"
            )
        values.append(array)
    return np.stack(values).reshape(2, 5, 21)


def _validate_artifact_payload(value: Any, unit: StageDUnit) -> Any:
    from .stage_d_artifacts import StageDUnitPayload as ArtifactPayload

    if not isinstance(value, ArtifactPayload):
        raise TypeError("value must be an artifact StageDUnitPayload")
    if value.audit != unit.audit:
        raise StageDProducerError(
            "artifact payload audit evidence disagrees with typed unit"
        )
    return value


def _backend_payload(value: Any, unit: StageDUnit) -> tuple[Any, dict[str, int | float]]:
    from .stage_d_artifacts import StageDUnitPayload as ArtifactPayload

    if isinstance(value, ArtifactPayload):
        return _validate_artifact_payload(value, unit), dict(_ZERO_RESOURCES)
    if not isinstance(value, StageDBackendPayload):
        raise TypeError(
            "Stage D backend must return StageDBackendPayload or artifact StageDUnitPayload"
        )
    if value.l3:
        raise StageDProducerError("Stage D backend payload contains forbidden L3")
    expected_l1 = {"clean", "repeat_clean_1"} if unit.audit else {"clean"}
    if set(value.l1) != expected_l1:
        raise StageDProducerError(
            "backend L1 must contain exactly primary clean and the audit-only repeat"
        )
    curve_data = value.l0.get("fidelity_curve_values", value.l0.get("fidelity_curves"))
    curves = _curve_array_from_records(curve_data, unit)
    audit_evidence = None
    if unit.audit:
        audit_evidence = {
            "repeat_clean_metric_ready_map": np.asarray(
                value.l1["repeat_clean_1"]
            ),
            **{key: np.asarray(item) for key, item in value.l2.items()},
        }
    elif value.l2:
        raise StageDProducerError("non-audit unit cannot retain L2 evidence")
    payload = ArtifactPayload(
        l0_scalars=value.l0, metric_ready_map=np.asarray(value.l1["clean"]),
        fidelity_curves=curves, audit_evidence=audit_evidence,
    )
    return payload, _resources(value.resources)


def _backend_call(
    backend: Any, group: StageDImageGroup, unit: StageDUnit, *, device: str,
    call_counter: Any,
) -> Any:
    runner = getattr(backend, "run_stage_d_unit", None)
    if callable(runner):
        return runner(group, unit, device=device, call_counter=call_counter)
    runner = getattr(backend, "run", None)
    if not callable(runner):
        raise TypeError("backend must implement run_stage_d_unit() or run()")
    return runner(unit, device=device, call_counter=call_counter)


def _translate_stage_c_scientific_error(
    error: BaseException, *, phase: str = "scientific_backend",
) -> StageDScientificFailure | None:
    """Translate only the frozen, closed Stage C scientific failures."""

    if not isinstance(error, StageCScientificInvalidity):
        return None
    if error.failure_code not in _SCIENTIFIC_FAILURE_CODES:
        return None
    partial = error.partial_evidence
    if partial is not None:
        if not isinstance(partial, StageCPartialEvidence):
            return None
        partial = StageDPartialEvidence(
            l0_scalars=partial.l0_scalars,
            metric_ready_map=partial.metric_ready_map,
            fidelity_curves=partial.fidelity_curves,
            audit_evidence=partial.audit_evidence,
        )
    return StageDScientificFailure(
        error.failure_code, phase=error.phase or phase,
        l0_scalars=error.l0_scalars, details=error.details,
        partial_evidence=partial,
    )


def _write_partial_evidence(
    writer: Any, partial: Any, unit: StageDUnit,
) -> tuple[dict[str, Any], ...]:
    if partial is None:
        return writer.artifacts
    if isinstance(partial, StageDBackendPayload):
        l0 = partial.l0 or None
        metric_map = partial.l1.get("clean")
        curves = partial.l0.get(
            "fidelity_curve_values", partial.l0.get("fidelity_curves")
        )
        audit = None
        if unit.audit and "repeat_clean_1" in partial.l1:
            audit = {
                "repeat_clean_metric_ready_map": partial.l1["repeat_clean_1"],
                **partial.l2,
            }
        partial = StageDPartialEvidence(
            l0_scalars=l0, metric_ready_map=metric_map,
            fidelity_curves=curves, audit_evidence=audit,
        )
    if not isinstance(partial, StageDPartialEvidence):
        raise TypeError("partial_evidence must be StageDPartialEvidence")
    # Closed retention order; a gap ends the prefix and later fields are
    # forbidden so provenance can state the exact produced/absent partition.
    components = (
        partial.l0_scalars, partial.metric_ready_map,
        partial.fidelity_curves, partial.audit_evidence,
    )
    seen_gap = False
    for index, component in enumerate(components):
        if component is None:
            seen_gap = True
        elif seen_gap:
            raise StageDProducerError(
                "partial evidence must be an ordered L0/map/curves/audit prefix"
            )
        elif index == 0:
            writer.write_l0_scalars(component)
        elif index == 1:
            writer.write_metric_ready_map(component)
        elif index == 2:
            writer.write_fidelity_curves(
                _curve_array_from_records(component, unit)
            )
        else:
            writer.write_audit_evidence(component)
    return writer.artifacts


def _provenance_kwargs(
    *, context: Any, unit: StageDUnit, group: StageDImageGroup,
    plan: StageDExecutionPlan, claim: Mapping[str, Any],
    retry_authorization: Mapping[str, Any] | None, shard: Mapping[str, int],
    terminal_status: str, failure: Mapping[str, Any] | None,
    resources: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    formal = _context_field(context, "formal")
    source_bindings = _context_field(context, "source_bindings")
    if isinstance(source_bindings, Mapping) and set(source_bindings) == {"status"}:
        # Synthetic permits intentionally have no reserve source authority.
        source_bindings = {
            name: {"status": source_bindings["status"]}
            for name in ("d0", "d1", "d2", "c10")
        }
    retry = (
        {"status": "NOT_A_RETRY"}
        if retry_authorization is None
        else _plain(retry_authorization, "retry authorization")
    )
    claim_binding = {
        "owner_token_digest": claim["owner_token_digest"],
        "group_owner_token_digest": claim["group_owner_token_digest"],
    }
    authorization_slot = _context_field(context, "authorization_slot")
    d7 = (
        None if not formal else {
            "status": _context_field(context, "execution_authorization"),
            "digest": __import__(
                "vitnns.stage_d_provenance", fromlist=["canonical_sha256"]
            ).canonical_sha256(authorization_slot),
        }
    )
    return {
        "mode": "formal" if formal else "SYNTHETIC_D4_TEST_ONLY",
        "producer": _context_field(context, "code_binding"),
        "dependencies": _context_field(context, "dependency_binding"),
        "model": _context_field(context, "model_binding"),
        "source_bindings": source_bindings, "plan": plan,
        "group": group, "unit": unit, "pair": group.pair_binding,
        "image_identity": (
            {"status": "VERIFIED_AFTER_D7_AUTHORIZATION",
             "image_sha256": group.image_sha256}
            if formal else {"status": "SYNTHETIC_NO_IMAGE_ACCESSED",
                            "fixture_id": group.sample_id}
        ),
        "method": {"id": unit.method_id, "version": "1.0.0"},
        "target": {"id": unit.target_id, "version": "1.0.0"},
        "operator": {"id": "fine_4x4_cell_projection", "version": "1.0.0"},
        "projection": {"id": "signed_56x56_fine_cell_mass"},
        "sign_semantics": {"id": "target_oriented_signed"},
        "normalization": {"id": "signed_endpoint_delta_zero_to_one"},
        "baseline": {"id": "normalized_zero_canonical_rgb_0_5"},
        "seed_contract": {
            "namespace": STAGE_D_FIDELITY_SEED_NAMESPACE,
            "base_seed": 20260821,
        },
        "attempt": {"number": claim["attempt_number"],
                     "id": claim["attempt_id"]},
        "retry": retry, "claim": claim_binding,
        "shard": {"id": shard["shard_id"], "count": shard["shard_count"]},
        "terminal": {"status": terminal_status, "terminal": True, "complete": True},
        "failure": failure, "resources": resources,
        "artifact_inventory": inventory, "d7_authorization": d7,
    }


def _status_sha256(attempt_path: Path) -> str:
    status = attempt_path / "status.json"
    return hashlib.sha256(status.read_bytes()).hexdigest()


def run_stage_d_execution_unit(
    unit: StageDUnit, *, group: StageDImageGroup, plan: StageDExecutionPlan,
    output_root: Path | str, backend: Any, context: Any,
    claim: Mapping[str, Any], retry_authorization: Mapping[str, Any] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run and atomically publish one claimed Stage D execution unit."""

    attempt_path, shard = _validated_execution_inputs(
        unit, group=group, plan=plan, output_root=output_root, context=context,
        claim=claim, retry_authorization=retry_authorization,
    )
    # Imports and filesystem mutation intentionally occur only after every
    # authorization, group, plan, claim and path binding above has passed.
    from .stage_c_resources import CallCounter, ResourceMeter
    from .stage_d_artifacts import StageDArtifactWriter
    from .stage_d_provenance import build_stage_d_provenance

    writer = StageDArtifactWriter(attempt_path, unit=unit, group=group)
    counter = CallCounter()
    meter = ResourceMeter(device=device, call_counter=counter)
    try:
        try:
            with meter:
                raw = _backend_call(
                    backend, group, unit, device=device, call_counter=counter
                )
        except (StageDScientificFailure, StageDInfrastructureFailure):
            raise
        except StageCBackendError as error:
            translated = _translate_stage_c_scientific_error(error)
            if translated is None:
                raise
            raise translated from error
        measured = meter.record
        payload, _ignored_backend_resources = _backend_payload(raw, unit)
        resources = _measured_resources(measured)
        writer.write_payload(payload)
        l0_l1, l2 = _artifact_resource_bytes(writer.artifacts)
        resources["l0_l1_payload_bytes"] = l0_l1
        resources["l2_payload_bytes"] = l2
        provenance = build_stage_d_provenance(**_provenance_kwargs(
            context=context, unit=unit, group=group, plan=plan, claim=claim,
            retry_authorization=retry_authorization, shard=shard,
            terminal_status="succeeded", failure=None, resources=resources,
            inventory=writer.artifacts,
        ))
        manifest = writer.succeed(provenance=provenance)
        disk_resources = dict(resources)
        disk_resources["control_and_provenance_bytes"] = _control_bytes(attempt_path)
        return {
            "state": "succeeded", "unit_digest": unit.canonical_digest,
            "pair_binding_digest": unit.pair_binding_digest,
            "attempt_id": claim["attempt_id"],
            "attempt_path": str(attempt_path),
            "status_sha256": _status_sha256(attempt_path),
            "failure_class": None, "failure_code": None,
            "result_manifest_published": True,
            "published_l0_l1_payload_count": sum(
                item["path"].startswith(("L0/", "L1/"))
                for item in manifest["artifacts"]
                if item["kind"] != "provenance"
            ),
            "inventory": {"artifact_count": len(manifest["artifacts"])},
            "resources": disk_resources,
        }
    except StageDScientificFailure as error:
        from .stage_d_engine import apply_partial_trace_policy

        trace_mode = unit.trace_mode
        retained_partial = apply_partial_trace_policy(
            error.partial_evidence, unit=unit, trace_mode=trace_mode
        )
        retained_failure_l0 = apply_partial_trace_policy(
            StageDPartialEvidence(l0_scalars=error.l0_scalars),
            unit=unit, trace_mode=trace_mode,
        ).l0_scalars
        failure: dict[str, Any] = {
            "code": error.failure_code, "phase": error.phase,
        }
        if error.details is not None:
            failure["details"] = error.details
        # Ask the writer to normalize produced/absent before provenance, then
        # build against the exact partial inventory it will seal.
        from .stage_d_provenance import (
            normalize_stage_d_scientific_failure,
            stage_d_scientific_failure_artifact,
        )
        produced_records = _write_partial_evidence(writer, retained_partial, unit)
        produced = [item["path"] for item in produced_records]
        normalized = normalize_stage_d_scientific_failure(
            failure, audit=unit.audit, produced=produced,
        )
        _normalized, failure_record, _raw = stage_d_scientific_failure_artifact(
            normalized, l0_scalars=retained_failure_l0, audit=unit.audit,
            produced=produced,
        )
        resources = _measured_resources(meter.record)
        provenance = build_stage_d_provenance(**_provenance_kwargs(
            context=context, unit=unit, group=group, plan=plan, claim=claim,
            retry_authorization=retry_authorization, shard=shard,
            terminal_status="scientific_failure", failure=normalized,
            resources=resources, inventory=[*produced_records, failure_record],
        ))
        manifest = writer.fail_scientifically_partial(
            failure=normalized, provenance=provenance,
            l0_scalars=retained_failure_l0,
        )
        disk_resources = dict(resources)
        disk_resources["l0_l1_payload_bytes"], disk_resources["l2_payload_bytes"] = (
            _artifact_resource_bytes(manifest["artifacts"])
        )
        disk_resources["control_and_provenance_bytes"] = _control_bytes(attempt_path)
        return {
            "state": "scientific_failed",
            "unit_digest": unit.canonical_digest,
            "pair_binding_digest": unit.pair_binding_digest,
            "attempt_id": claim["attempt_id"],
            "attempt_path": str(attempt_path),
            "status_sha256": _status_sha256(attempt_path),
            "failure_class": "SCIENTIFIC_INVALIDITY",
            "failure_code": error.failure_code,
            "result_manifest_published": True,
            "published_l0_l1_payload_count": sum(
                item["path"].startswith(("L0/", "L1/"))
                for item in manifest["artifacts"]
                if item["kind"] not in {"provenance", "scientific_failure"}
            ),
            "inventory": {"artifact_count": len(manifest["artifacts"])},
            "resources": disk_resources,
        }
    except StageDInfrastructureFailure as error:
        measured_resources = _measured_resources(meter.record)
        # Explicit infrastructure metadata may contribute storage/control
        # counts unavailable to the execution-boundary meter, but it cannot
        # replace observed elapsed/call/CUDA facts.
        supplied = error.resources
        for name in (
            "l0_l1_payload_bytes", "l2_payload_bytes",
            "control_and_provenance_bytes",
        ):
            measured_resources[name] = supplied[name]
        writer.mark_infrastructure_interruption(
            code=error.failure_code, resources=measured_resources
        )
        measured_resources["control_and_provenance_bytes"] = sum(
            path.stat().st_size for path in (
                attempt_path / "status.json", attempt_path / "manifest.json",
            )
        )
        return {
            "state": "infrastructure_failed",
            "unit_digest": unit.canonical_digest,
            "pair_binding_digest": unit.pair_binding_digest,
            "attempt_id": claim["attempt_id"],
            "attempt_path": str(attempt_path),
            "status_sha256": _status_sha256(attempt_path),
            "failure_class": "INFRASTRUCTURE_FAILURE",
            "failure_code": error.failure_code,
            "result_manifest_published": False,
            "published_l0_l1_payload_count": 0,
            "inventory": {"artifact_count": 0},
            "resources": measured_resources,
        }


run_stage_d_unit = run_stage_d_execution_unit


@contextmanager
def _backend_image_scope(
    backend: StageDGroupBackend, group: StageDImageGroup, device: str,
) -> Iterator[None]:
    group_scope = getattr(backend, "image_group_scope", None)
    scope = group_scope if callable(group_scope) else getattr(backend, "image_scope", None)
    if not callable(scope):
        raise TypeError("backend must implement image_scope()")
    try:
        subject = group if callable(group_scope) else group.sample_id
        with scope(subject, device=device):
            yield
    except Exception as error:
        # Normalize the Stage C backend's fail-closed concurrency error without
        # hiding exceptions raised by the unit runner itself.
        if "already owned by another image scope" in str(error):
            raise StageDGroupBusyError(str(error)) from error
        raise


def _group_state(results: Sequence[Any]) -> str:
    states = tuple(_control_state(item, "unit result") for item in results)
    if "BLOCKED" in states:
        return "BLOCKED"
    if "RETRY_PENDING" in states:
        return "RETRY_PENDING"
    if len(states) == 2 and all(item in STAGE_D_TERMINAL_STATES for item in states):
        return (
            "TERMINAL_SCIENTIFIC_FAILURE"
            if "TERMINAL_SCIENTIFIC_FAILURE" in states
            else "TERMINAL_SUCCESS"
        )
    raise StageDProducerError("group result has an invalid control-state combination")


def run_stage_d_image_group(
    group: StageDImageGroup, *, backend: StageDGroupBackend,
    run_unit: Callable[..., Any], device: str = "cpu",
    on_unit_terminal: Callable[[StageDUnit, Any], Any] | None = None,
) -> StageDGroupRunResult:
    """Run one complete image group under one exclusive cache scope.

    Scientific failure and retry-pending are control states, not Python
    exceptions, and therefore do not prevent the paired method from running.
    A workflow-integrity ``BLOCKED`` state is the sole control-state early stop.
    Raw scientific payloads are never inspected or serialized by this layer.
    """

    checked = _validate_group(group)
    if not callable(run_unit):
        raise TypeError("run_unit must be callable")
    if on_unit_terminal is not None and not callable(on_unit_terminal):
        raise TypeError("on_unit_terminal must be callable or None")
    if not isinstance(device, str) or not device:
        raise TypeError("device must be a non-empty string")

    results: list[Any] = []
    cache_released = False
    try:
        with _backend_image_scope(backend, checked, device):
            for unit in checked.units:
                # Validate exact map/curve coverage immediately before dispatch.
                _unit_coordinates(checked, unit)
                result = run_unit(unit, backend=backend, device=device)
                state = _control_state(result, "run_unit result")
                unit_digest = _field(result, "unit_digest", unit.canonical_digest)
                pair_digest = _field(
                    result, "pair_binding_digest", unit.pair_binding_digest
                )
                if (
                    unit_digest != unit.canonical_digest
                    or pair_digest != unit.pair_binding_digest
                ):
                    raise StageDProducerError(
                        "run_unit result does not bind the requested Stage D unit"
                    )
                results.append(result)
                if state in STAGE_D_TERMINAL_STATES and on_unit_terminal is not None:
                    on_unit_terminal(unit, result)
                if state == "BLOCKED":
                    break
        cache_released = True
    finally:
        # A conforming context manager releases in its own finally.  Calling
        # release here as a fallback would double-release and obscure failures.
        pass

    return StageDGroupRunResult(
        group_digest=checked.canonical_digest,
        pair_binding_digest=checked.pair_binding.canonical_digest,
        state=_group_state(results), unit_results=tuple(results),
        cache_released=cache_released,
    )


def stage_d_fidelity_seed(
    sample_id: str, block: int, head: int, repeat_index: int
) -> int:
    """Derive a Stage D tie seed through the shared fidelity primitive."""

    return derive_fidelity_seed(
        STAGE_D_FIDELITY_SEED_NAMESPACE, sample_id, block, head, repeat_index,
        repeat_count=5,
    )


__all__ = [
    "ProductionStageDBackend", "STAGE_D_CONTROL_STATES",
    "STAGE_D_FIDELITY_SEED_NAMESPACE", "STAGE_D_TERMINAL_STATES",
    "StageDGroupBackend", "StageDGroupBusyError",
    "StageDBackendPayload", "StageDGroupRunResult",
    "StageDInfrastructureFailure", "StageDProducerError",
    "StageDScientificFailure", "StageDUnitPayload",
    "run_stage_d_execution_unit", "run_stage_d_image_group",
    "run_stage_d_unit",
    "stage_d_fidelity_seed",
]
