"""Canonical Stage D backend with explicit NNS trace retention policy.

The scientific implementation remains owned by :mod:`stage_d_producer` and
the Stage C primitives it composes.  This module adds only the Stage D
``fast``/``full`` retention boundary: both modes execute the same attribution
and fidelity computation, while only ``full`` constructs and retains the
attribution trace.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any, Literal, cast

import numpy as np

from .stage_d_producer import (
    ProductionStageDBackend as _ProductionStageDBackend,
    StageDBackendPayload,
    StageDProducerError,
)


TraceMode = Literal["fast", "full"]
TRACE_MODES = frozenset({"fast", "full"})
NNS_METHOD_ID = (
    "conditional_fixed_clean_attention_head_direction_value_path_ig"
)
NNS_CONTRACT_KEYS = frozenset({
    "method_id", "method_version", "comparison_line", "target_ref",
    "projection", "sign_semantics",
})
NNS_TRACE_KEYS = frozenset({
    "provenance_fragment", "map_execution_records",
})


def validate_trace_mode(value: Any) -> TraceMode:
    """Return a closed, case-sensitive Stage D trace mode."""

    if not isinstance(value, str) or value not in TRACE_MODES:
        raise StageDProducerError(
            "trace_mode must be exactly 'fast' or 'full'"
        )
    return cast(TraceMode, value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _unit_method_id(unit: Any) -> Any:
    stage_d_unit = _field(unit, "stage_d_unit")
    if stage_d_unit is not None:
        return _field(stage_d_unit, "method_id")
    key = _field(unit, "key")
    if key is not None:
        method_id = _field(key, "method_id")
        if method_id is not None:
            return method_id
    return _field(unit, "method_id")


def expected_trace_mode(unit: Any) -> TraceMode:
    """Derive the only valid mode: audit units are full, all others fast."""

    stage_d_unit = _field(unit, "stage_d_unit")
    subject = stage_d_unit if stage_d_unit is not None else unit
    audit = _field(subject, "audit")
    if type(audit) is not bool:
        raise StageDProducerError(
            "Stage D trace policy requires a boolean unit.audit"
        )
    return "full" if audit else "fast"


def resolve_trace_mode(
    unit: Any, *, trace_mode: Any = None, configured_mode: Any = None,
) -> TraceMode:
    """Validate every declared mode against the immutable audit split."""

    expected = expected_trace_mode(unit)
    stage_d_unit = _field(unit, "stage_d_unit")
    unit_mode = _field(unit, "trace_mode")
    if unit_mode is None and stage_d_unit is not None:
        unit_mode = _field(stage_d_unit, "trace_mode")
    for source, value in (
        ("unit", unit_mode),
        ("backend", configured_mode),
        ("call", trace_mode),
    ):
        if value is None:
            continue
        selected = validate_trace_mode(value)
        if selected != expected:
            raise StageDProducerError(
                f"{source} trace_mode {selected!r} disagrees with unit.audit; "
                f"expected {expected!r}"
            )
    return expected


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_copy(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(child) for child in value]
    return value


def _default_trace_builder(
    *, unit: Any, l0: Mapping[str, Any], l2: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any], np.ndarray]:
    """Project already-computed evidence into the bounded full trace.

    No attribution or fidelity primitive is called here.  The numeric trace is
    the already-retained native clean attribution, so enabling trace cannot
    alter scientific computation, thresholds, seeds, or curve values.
    """

    method_id = _unit_method_id(unit)
    if method_id != NNS_METHOD_ID:
        raise StageDProducerError(
            "NNS trace may be constructed only for the registered NNS method"
        )
    native = l2.get("clean_native")
    if native is None:
        raise StageDProducerError(
            "full NNS trace requires the already-computed clean_native L2 array"
        )
    nns_trace = np.array(native, copy=True)
    if (
        nns_trace.size == 0
        or nns_trace.dtype.kind not in {"b", "i", "u", "f"}
        or (nns_trace.dtype.kind == "f" and not np.isfinite(nns_trace).all())
    ):
        raise StageDProducerError(
            "full NNS trace must be a non-empty finite numeric array"
        )
    contract = {
        name: _json_copy(l0.get(name))
        for name in (
            "method_id", "method_version", "comparison_line",
            "target_ref", "projection", "sign_semantics",
        )
    }
    trace = {
        "provenance_fragment": _json_copy(l0.get("provenance_fragment", {})),
        "map_execution_records": _json_copy(
            l0.get("map_execution_records", [])
        ),
    }
    return contract, trace, nns_trace


def _call_backend(
    backend: Any, method_name: str, *args: Any, trace_mode: TraceMode, **kwargs: Any
) -> Any:
    method = getattr(backend, method_name)
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    parameter = parameters.get("trace_mode")
    if parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }:
        kwargs["trace_mode"] = trace_mode
    return method(*args, **kwargs)


def apply_trace_policy(
    payload: StageDBackendPayload, *, unit: Any, trace_mode: Any,
    trace_builder: Callable[..., Any] | None = None,
) -> StageDBackendPayload:
    """Apply trace retention after, and without changing, scientific work."""

    selected = resolve_trace_mode(unit, trace_mode=trace_mode)
    if not isinstance(payload, StageDBackendPayload):
        raise TypeError("canonical Stage D backend requires StageDBackendPayload")
    l0 = dict(payload.l0)
    l2 = dict(payload.l2)
    is_nns = _unit_method_id(unit) == NNS_METHOD_ID
    provenance_fragment = l0.get("provenance_fragment", {})
    map_execution_records = l0.get("map_execution_records", [])
    # Never trust or leak trace-shaped fields supplied by a delegated backend.
    l0.pop("nns_contract", None)
    l0.pop("trace", None)
    l2.pop("nns_trace", None)
    l0["trace_mode"] = selected
    if is_nns and selected == "fast":
        l0.pop("provenance_fragment", None)
        l0.pop("map_execution_records", None)
        l2.clear()
    if selected == "full" and is_nns:
        builder = _default_trace_builder if trace_builder is None else trace_builder
        if not callable(builder):
            raise TypeError("trace_builder must be callable or None")
        built = builder(unit=unit, l0=l0, l2=l2)
        if not isinstance(built, tuple) or len(built) != 3:
            raise StageDProducerError(
                "trace_builder must return (nns_contract, trace, nns_trace)"
            )
        contract, trace, nns_trace = built
        if not isinstance(contract, Mapping) or not isinstance(trace, Mapping):
            raise StageDProducerError("nns_contract and trace must be mappings")
        if set(contract) != NNS_CONTRACT_KEYS or any(
            contract.get(name) != l0.get(name) for name in NNS_CONTRACT_KEYS
        ):
            raise StageDProducerError(
                "nns_contract does not bind the canonical six L0 fields"
            )
        if set(trace) != NNS_TRACE_KEYS:
            raise StageDProducerError(
                "trace must contain provenance_fragment and "
                "map_execution_records exactly"
            )
        if (
            trace.get("provenance_fragment") != provenance_fragment
            or trace.get("map_execution_records") != map_execution_records
        ):
            raise StageDProducerError(
                "trace does not bind the retained L0 execution evidence"
            )
        array = np.asarray(nns_trace)
        if (
            array.size == 0
            or array.dtype.kind not in {"b", "i", "u", "f"}
            or (array.dtype.kind == "f" and not np.isfinite(array).all())
        ):
            raise StageDProducerError(
                "nns_trace must be a non-empty finite numeric array"
            )
        l0["nns_contract"] = _json_copy(contract)
        l0["trace"] = _json_copy(trace)
        l2["nns_trace"] = np.array(array, copy=True)
    return StageDBackendPayload(
        l0=l0, l1=payload.l1, l2=l2, l3=payload.l3,
        provenance=payload.provenance, resources=payload.resources,
        lifecycle=payload.lifecycle,
    )


def apply_partial_trace_policy(
    partial: Any, *, unit: Any, trace_mode: Any,
) -> Any:
    """Apply the same no-trace boundary to retained failure prefixes."""

    selected = resolve_trace_mode(unit, trace_mode=trace_mode)
    if partial is None or _unit_method_id(unit) != NNS_METHOD_ID:
        return partial
    if selected == "full":
        return partial
    backend_payload = isinstance(partial, StageDBackendPayload)
    l0_value = (
        partial.l0 if backend_payload else getattr(partial, "l0_scalars", None)
    )
    l0 = None if l0_value is None else dict(l0_value)
    if l0 is not None:
        for key in (
            "nns_contract", "trace", "provenance_fragment",
            "map_execution_records",
        ):
            l0.pop(key, None)
        l0["trace_mode"] = selected
    audit_value = (
        partial.l2 if backend_payload else getattr(partial, "audit_evidence", None)
    )
    audit = None if audit_value is None else dict(audit_value)
    if backend_payload:
        audit = {}
    elif audit is not None:
        audit.pop("nns_trace", None)
        if not audit:
            audit = None
    if backend_payload:
        return StageDBackendPayload(
            l0={} if l0 is None else l0, l1=partial.l1,
            l2={} if audit is None else audit, l3=partial.l3,
            provenance=partial.provenance, resources=partial.resources,
            lifecycle=partial.lifecycle,
        )
    return type(partial)(
        l0_scalars=l0,
        metric_ready_map=getattr(partial, "metric_ready_map", None),
        fidelity_curves=getattr(partial, "fidelity_curves", None),
        audit_evidence=audit,
    )


# The producer owns the public production class.  Keeping these as identity
# aliases avoids an import cycle and guarantees old imports get trace policy.
CanonicalStageDBackend = _ProductionStageDBackend


def create_stage_d_backend(
    *, trace_mode: TraceMode | None = None, **kwargs: Any
) -> CanonicalStageDBackend:
    """Construct the canonical production backend."""

    return CanonicalStageDBackend(trace_mode=trace_mode, **kwargs)


def run_stage_d_canonical_backend(
    backend: Any, unit: Any, *, device: str, call_counter: Any,
    trace_mode: TraceMode,
) -> StageDBackendPayload:
    """Run a compatible backend and apply the canonical trace policy."""

    selected = validate_trace_mode(trace_mode)
    if isinstance(backend, CanonicalStageDBackend):
        return backend.run(
            unit, device=device, call_counter=call_counter, trace_mode=selected
        )
    raw = _call_backend(
        backend, "run", unit, device=device, call_counter=call_counter,
        trace_mode=selected,
    )
    return apply_trace_policy(raw, unit=unit, trace_mode=selected)


# New callers should use the explicit canonical name; this alias lets factory
# integrations migrate from the producer module without changing semantics.
ProductionStageDBackend = CanonicalStageDBackend


__all__ = [
    "CanonicalStageDBackend", "NNS_CONTRACT_KEYS", "NNS_METHOD_ID",
    "NNS_TRACE_KEYS",
    "ProductionStageDBackend", "TRACE_MODES", "TraceMode",
    "apply_partial_trace_policy", "apply_trace_policy",
    "create_stage_d_backend", "expected_trace_mode",
    "resolve_trace_mode", "run_stage_d_canonical_backend",
    "validate_trace_mode",
]
