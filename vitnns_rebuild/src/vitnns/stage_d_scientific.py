"""Model-free source validation of retained Stage D scientific evidence.

The validator in this module consumes only decoded artifact values.  In
particular, it deliberately does not import Stage C implementations, resolve a
target, or access a model/image.  Identities and numerical summaries are
recomputed when their source representation is retained.  Tensor-content
digests whose tensors are not retained remain opaque commitments: they are
checked for every sharing relationship promised by the frozen contract, but
are never described as independently recomputed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


NNS_METHOD_ID = (
    "conditional_fixed_clean_attention_head_direction_value_path_ig"
)
MATCHED_METHOD_ID = "matched_fixed_clean_attention_patch_token_contribution"
PRODUCTION_L0_KEYS = frozenset({
    "method_id", "method_version", "comparison_line", "target_ref",
    "status", "na", "projection", "sign_semantics",
    "provenance_fragment", "sample_id", "prediction", "stability",
    "map_execution_records", "lifecycle", "repeatability",
    "fidelity_aggregates", "top20", "fidelity_curves",
    "perturbations", "target_identity", "clean_tensor_hashes",
    "fidelity_enabled", "stability_enabled",
})
TRACE_MODES = frozenset({"fast", "full"})
NNS_CONTRACT_KEYS = frozenset({
    "method_id", "method_version", "comparison_line", "target_ref",
    "projection", "sign_semantics",
})
NNS_TRACE_KEYS = frozenset({
    "provenance_fragment", "map_execution_records",
})
_TRACELESS_PRODUCTION_L0_KEYS = (
    PRODUCTION_L0_KEYS - NNS_TRACE_KEYS
) | {"trace_mode"}
CURVE_KEYS = frozenset({
    "status", "method_id", "comparison_line", "target_ref",
    "target_identity_sha256", "scale", "direction", "ranking_kind",
    "sample_id", "block", "head", "replicate", "seed",
    "tie_priority_sha256", "source_unit_map_sha256", "score_sha256",
    "input_pair_sha256", "ranking_sha256", "intervention_sha256",
    "curve_fractions", "unit_counts", "actual_fractions", "responses",
    "auc", "response_clipped", "auc_integration", "seed_namespace",
})
AGGREGATE_KEYS = frozenset({
    "status", "replicate_role", "replicate_count",
    "standard_deviation", "curves", "response_mean", "response_std",
    "auc_mean", "auc_std",
})
TOP20_KEYS = frozenset({
    "status", "method_id", "comparison_line", "target_ref",
    "target_identity_sha256", "ranking_kind", "sample_id", "block",
    "head", "replicate", "seed", "tie_priority_sha256",
    "input_pair_sha256", "ranking_sha256", "scale",
    "fraction_requested", "k", "actual_fraction", "sufficiency",
    "sufficiency_gap", "comprehensiveness", "decision_gate",
})
INTEGRITY_KEYS = frozenset({
    "schema_version", "algorithm", "checked", "entry_sha256",
    "exit_sha256", "matched", "restored", "differences",
    "signature_calls", "verification_error_type",
    "verification_error_message", "body_exception_type",
})
RESTORATION_KEYS = frozenset({
    "schema_version", "checked", "changes_detected", "restored",
    "differences", "errors",
})
MAP_EXECUTION_KEYS = frozenset({
    "method_id", "input_variant", "repeat_index", "status",
    "ig_result", "attempted_steps_total", "backward_invocations",
    "actual_calls", "exception",
})
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PAIR_BINDING_KEYS = frozenset({
    "target_identity_sha256", "clean_input_sha256", "block_index",
    "head_index", "input_pair_sha256", "seeds",
    "model_binding_sha256", "plan_digest", "group_digest",
    "pair_digest", "image_identity_sha256", "target_contract_sha256",
    "operator_contract_sha256", "projection_contract_sha256",
    "sign_semantics_contract_sha256", "normalization_contract_sha256",
    "baseline_contract_sha256", "fractions_contract_sha256",
    "seed_contract_sha256",
})


@dataclass(frozen=True, slots=True)
class StageDScientificValidation:
    """Closed validation result; pair evidence contains hashes, not values."""

    errors: tuple[str, ...]
    pair_binding: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if not isinstance(self.errors, tuple) or any(
            not isinstance(item, str) or not item for item in self.errors
        ):
            raise TypeError("errors must be a tuple of non-empty strings")
        if self.pair_binding is None:
            return
        if not isinstance(self.pair_binding, Mapping):
            raise TypeError("pair_binding must be a mapping or None")
        owned: dict[str, Any] = {}
        for key, value in self.pair_binding.items():
            if not isinstance(key, str):
                raise TypeError("pair_binding keys must be strings")
            if isinstance(value, (list, tuple)):
                owned[key] = tuple(value)
            elif value is None or type(value) in {str, int, bool, float}:
                owned[key] = value
            else:
                raise TypeError("pair_binding values must be scalar or sequences")
        object.__setattr__(self, "pair_binding", MappingProxyType(owned))

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def pair_binding_evidence(self) -> Mapping[str, Any] | None:
        """Compatibility spelling used by the first D5 integration draft."""

        return self.pair_binding


def _field(unit: Any, name: str) -> Any:
    if isinstance(unit, Mapping):
        return unit.get(name)
    return getattr(unit, name, None)


def _trace_view(
    errors: list[str], l0: Mapping[str, Any], unit: Any, *, nns: bool,
) -> tuple[str | None, Mapping[str, Any] | None, Any]:
    """Validate the mode envelope and expose retained full-trace values.

    Payloads produced before the trace-mode contract remain readable.  Once
    either the unit or payload declares a mode, however, the new closed shape
    is mandatory and a full NNS payload cannot silently degrade to fast.
    """

    unit_mode = _field(unit, "trace_mode")
    payload_mode = l0.get("trace_mode")
    if payload_mode is None:
        # ``StageDUnit.trace_mode`` is a derived runtime policy added after
        # the original artifact schema.  Its presence must not reinterpret an
        # already-persisted legacy L0 envelope as a malformed new envelope.
        if set(l0) != PRODUCTION_L0_KEYS:
            errors.append("production_l0_envelope_invalid")
        fragment = l0.get("provenance_fragment")
        return None, fragment if isinstance(fragment, Mapping) else None, l0.get(
            "map_execution_records"
        )
    if unit_mode not in TRACE_MODES:
        errors.append("unit_trace_mode_invalid")
    if payload_mode not in TRACE_MODES:
        errors.append("trace_mode_invalid")
        return None, None, None
    if unit_mode is not None and unit_mode != payload_mode:
        errors.append("trace_mode_unit_mismatch")

    expected = (
        set(_TRACELESS_PRODUCTION_L0_KEYS)
        if payload_mode == "fast" and nns
        else set(PRODUCTION_L0_KEYS) | {"trace_mode"}
    )
    if payload_mode == "full" and nns:
        expected.update({"nns_contract", "trace"})
    if set(l0) != expected:
        errors.append("production_l0_envelope_invalid")

    if not nns:
        fragment = l0.get("provenance_fragment")
        return payload_mode, (
            fragment if isinstance(fragment, Mapping) else None
        ), l0.get("map_execution_records")
    if payload_mode == "fast":
        return payload_mode, None, None
    contract = l0.get("nns_contract")
    if (
        not isinstance(contract, Mapping)
        or set(contract) != NNS_CONTRACT_KEYS
        or any(contract.get(name) != l0.get(name) for name in NNS_CONTRACT_KEYS)
    ):
        errors.append("nns_contract_invalid")
    trace = l0.get("trace")
    if not isinstance(trace, Mapping) or set(trace) != NNS_TRACE_KEYS:
        errors.append("nns_trace_invalid")
        return payload_mode, None, None
    fragment = trace.get("provenance_fragment")
    records = trace.get("map_execution_records")
    if not isinstance(fragment, Mapping):
        errors.append("nns_trace_provenance_missing")
        fragment = None
    if not isinstance(records, list):
        errors.append("nns_trace_execution_records_missing")
        records = None
    if fragment != l0.get("provenance_fragment") or records != l0.get(
        "map_execution_records"
    ):
        errors.append("nns_trace_binding_invalid")
    return payload_mode, fragment, records


def _number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _digest_fields_valid(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                return False
            lowered = key.casefold()
            if lowered.endswith("sha256") or lowered.endswith("digest"):
                if child is not None and not _sha(child):
                    return False
            elif not _digest_fields_valid(child):
                return False
    elif isinstance(value, (list, tuple)):
        return all(_digest_fields_valid(child) for child in value)
    return True


def _identity_sha256(value: Mapping[str, Any]) -> str | None:
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError):
        return None
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Any) -> str | None:
    """Stage-D provenance canonical hash (UTF-8, not ASCII-only)."""

    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        return None
    return hashlib.sha256(raw).hexdigest()


def _provenance_pair_binding(
    errors: list[str], provenance: Any, unit: Any, fractions: list[float],
    namespace: str,
) -> dict[str, Any] | None:
    if not isinstance(provenance, Mapping):
        errors.append("attempt_provenance_missing")
        return None
    root_keys = {
        "schema_version", "document_type", "mode", "unit_identity",
        "unit_identity_sha256", "bindings", "canonical_digest",
    }
    if set(provenance) != root_keys:
        errors.append("attempt_provenance_schema_invalid")
        return None
    digest_source = dict(provenance)
    observed_digest = digest_source.pop("canonical_digest", None)
    if (
        provenance.get("schema_version") != 1
        or provenance.get("document_type")
        != "stage_d_execution_unit_provenance"
        or not _sha(observed_digest)
        or observed_digest != _canonical_sha256(digest_source)
    ):
        errors.append("attempt_provenance_identity_invalid")
    identity = provenance.get("unit_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("method_id") != _field(unit, "method_id")
        or identity.get("sample_id") != _field(unit, "sample_id")
        or identity.get("block_index") != _field(unit, "block_index")
        or identity.get("head_index") != _field(unit, "head_index")
        or identity.get("target_id") != _field(unit, "target_id")
        or identity.get("canonical_digest") != _field(unit, "canonical_digest")
        or provenance.get("unit_identity_sha256")
        != _field(unit, "canonical_digest")
    ):
        errors.append("attempt_provenance_unit_invalid")
    bindings = provenance.get("bindings")
    binding_keys = {
        "d7_authorization", "producer", "dependencies", "model",
        "source_bindings", "coordinates", "image_identity",
        "scientific_contract", "execution", "terminal", "failure",
        "resources", "artifact_inventory",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != binding_keys:
        errors.append("attempt_provenance_bindings_invalid")
        return None
    coordinates = bindings.get("coordinates")
    coordinate_keys = {
        "plan_digest", "group_digest", "pair_digest", "unit_digest",
        "work_item_digests", "work_item_count",
        "work_item_sequence_digest",
    }
    if (
        not isinstance(coordinates, Mapping) or set(coordinates) != coordinate_keys
        or any(not _sha(coordinates.get(name)) for name in (
            "plan_digest", "group_digest", "pair_digest", "unit_digest",
            "work_item_sequence_digest",
        ))
        or coordinates.get("pair_digest") != _field(unit, "pair_binding_digest")
        or coordinates.get("unit_digest") != _field(unit, "canonical_digest")
    ):
        errors.append("attempt_provenance_coordinates_invalid")
    scientific = bindings.get("scientific_contract")
    scientific_keys = {
        "method", "target", "block_index", "head_index", "operator",
        "projection", "sign_semantics", "normalization", "baseline",
        "fractions", "seed_contract",
    }
    expected = {
        "target": {"id": _field(unit, "target_id"), "version": "1.0.0"},
        "operator": {"id": "fine_4x4_cell_projection", "version": "1.0.0"},
        "projection": {"id": "signed_56x56_fine_cell_mass"},
        "sign_semantics": {"id": "target_oriented_signed"},
        "normalization": {"id": "signed_endpoint_delta_zero_to_one"},
        "baseline": {"id": "normalized_zero_canonical_rgb_0_5"},
        "fractions": fractions,
        "seed_contract": {"namespace": namespace, "base_seed": 20260821},
    }
    if (
        not isinstance(scientific, Mapping) or set(scientific) != scientific_keys
        or scientific.get("method") != {
            "id": _field(unit, "method_id"), "version": "1.0.0"
        }
        or scientific.get("block_index") != _field(unit, "block_index")
        or scientific.get("head_index") != _field(unit, "head_index")
        or any(scientific.get(name) != value for name, value in expected.items())
    ):
        errors.append("attempt_provenance_scientific_contract_invalid")
    model = bindings.get("model")
    if (
        not isinstance(model, Mapping) or not model
        or not _digest_fields_valid(model)
    ):
        errors.append("attempt_provenance_model_invalid")
    model_digest = _canonical_sha256(model) if isinstance(model, Mapping) else None
    image = bindings.get("image_identity")
    image_digest = _canonical_sha256(image) if isinstance(image, Mapping) else None
    if model_digest is None or image_digest is None:
        errors.append("attempt_provenance_shared_binding_invalid")
        return None
    return {
        "model_binding_sha256": model_digest,
        "plan_digest": coordinates.get("plan_digest") if isinstance(coordinates, Mapping) else None,
        "group_digest": coordinates.get("group_digest") if isinstance(coordinates, Mapping) else None,
        "pair_digest": coordinates.get("pair_digest") if isinstance(coordinates, Mapping) else None,
        "image_identity_sha256": image_digest,
        "target_contract_sha256": _canonical_sha256(expected["target"]),
        "operator_contract_sha256": _canonical_sha256(expected["operator"]),
        "projection_contract_sha256": _canonical_sha256(expected["projection"]),
        "sign_semantics_contract_sha256": _canonical_sha256(expected["sign_semantics"]),
        "normalization_contract_sha256": _canonical_sha256(expected["normalization"]),
        "baseline_contract_sha256": _canonical_sha256(expected["baseline"]),
        "fractions_contract_sha256": _canonical_sha256(expected["fractions"]),
        "seed_contract_sha256": _canonical_sha256(expected["seed_contract"]),
    }


def _array_sha256(value: np.ndarray) -> str:
    source = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(source.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(source.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(source.tobytes(order="C"))
    return digest.hexdigest()


def _seed(namespace: str, sample: str, block: int, head: int, repeat: int) -> int:
    raw = f"{namespace}{sample}|{block}|{head}|{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _tie_priority_sha256(unit_count: int, seed: int) -> str:
    random_order = np.random.default_rng(seed).permutation(unit_count)
    priority = np.empty(unit_count, dtype=np.int64)
    priority[random_order] = np.arange(unit_count, dtype=np.int64)
    return _array_sha256(priority)


def _target_oriented_ranking_sha256(metric: np.ndarray, seed: int) -> str:
    """Rebuild the frozen score-descending, tie-priority ranking."""

    scores = np.maximum(
        metric, np.float32(0.0),
    ).astype(np.float32, copy=False).reshape(-1)
    random_order = np.random.default_rng(seed).permutation(scores.size)
    priority = np.empty(scores.size, dtype=np.int64)
    priority[random_order] = np.arange(scores.size, dtype=np.int64)
    ranking = np.lexsort(
        (priority, -scores.astype(np.float64)),
    ).astype(np.int64)
    return _array_sha256(ranking)


def _normalized_zero_baseline_sha256() -> str:
    """Hash the contract-defined float32 [1,3,224,224] zero tensor."""

    shape = (1, 3, 224, 224)
    digest = hashlib.sha256()
    digest.update(b"torch.float32")
    digest.update(b"|")
    digest.update(",".join(str(item) for item in shape).encode("ascii"))
    digest.update(b"|")
    digest.update(bytes(math.prod(shape) * np.dtype(np.float32).itemsize))
    return digest.hexdigest()


NORMALIZED_ZERO_BASELINE_SHA256 = _normalized_zero_baseline_sha256()


def _input_pair_sha256(clean_sha256: str) -> str:
    source = f"{clean_sha256}|{NORMALIZED_ZERO_BASELINE_SHA256}".encode(
        "ascii"
    )
    return hashlib.sha256(source).hexdigest()


def _auc(x: Sequence[Any], y: Sequence[Any]) -> float | None:
    try:
        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if xx.shape != (21,) or yy.shape != (21,) or not (np.isfinite(xx).all() and np.isfinite(yy).all()):
        return None
    if xx[0] != 0.0 or xx[-1] != 1.0 or np.any(np.diff(xx) <= 0.0):
        return None
    return float(np.sum(np.diff(xx) * (yy[:-1] + yy[1:]) * 0.5, dtype=np.float64))


def _close(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    return _number(left) and _number(right) and math.isclose(
        float(left), float(right), abs_tol=atol, rel_tol=rtol
    )


def _artifact_shapes(
    errors: list[str], metric: Any, curves: Any, audit: Any, *, audit_unit: bool,
    trace_mode: str | None = None, nns: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    metric_array = np.asarray(metric) if isinstance(metric, np.ndarray) else None
    curve_array = np.asarray(curves) if isinstance(curves, np.ndarray) else None
    if metric_array is None or metric_array.dtype != np.float32 or metric_array.shape != (56, 56) or not np.isfinite(metric_array).all():
        errors.append("metric_ready_map_invalid")
        metric_array = None
    if curve_array is None or curve_array.dtype != np.float64 or curve_array.shape != (2, 5, 21) or not np.isfinite(curve_array).all():
        errors.append("fidelity_curve_array_invalid")
        curve_array = None
    if audit_unit:
        if not isinstance(audit, Mapping):
            errors.append("audit_evidence_missing")
        else:
            repeat = audit.get("repeat_clean_metric_ready_map")
            if not isinstance(repeat, np.ndarray) or repeat.dtype != np.float32 or repeat.shape != (56, 56) or not np.isfinite(repeat).all():
                errors.append("audit_repeat_map_invalid")
    elif audit is not None:
        errors.append("non_audit_evidence_present")
    nns_trace = audit.get("nns_trace") if isinstance(audit, Mapping) else None
    if trace_mode == "full" and nns:
        trace_array = (
            np.asarray(nns_trace) if isinstance(nns_trace, np.ndarray) else None
        )
        if (
            trace_array is None or trace_array.size == 0
            or trace_array.dtype.kind not in {"b", "i", "u", "f"}
            or (
                trace_array.dtype.kind == "f"
                and not np.isfinite(trace_array).all()
            )
        ):
            errors.append("nns_l2_trace_missing_or_invalid")
    elif trace_mode == "fast" and isinstance(audit, Mapping) and (
        "nns_trace" in audit
    ):
        errors.append("fast_nns_l2_trace_present")
    return metric_array, curve_array


def _endpoint_checks(errors: list[str], provenance: Mapping[str, Any], *, nns: bool, limit: float) -> None:
    target = provenance.get("target")
    if not isinstance(target, Mapping):
        errors.append("target_provenance_missing")
        return
    names = (
        "clean_value_max_abs_error", "clean_replay_max_abs_error",
        "clean_model_output_max_abs_error", "baseline_replay_max_abs_error",
    )
    values = [target.get(name) for name in names]
    if any(not _number(value) or float(value) < 0.0 or float(value) >= limit for value in values):
        errors.append("endpoint_parity_failed")
    direction_norm = target.get("direction_norm")
    direction_threshold = target.get("direction_threshold")
    if (
        not _number(direction_norm) or not _number(direction_threshold)
        or not float(direction_norm) > float(direction_threshold) >= 0.0
    ):
        errors.append("target_direction_invalid")
    if not nns:
        return
    parity = provenance.get("endpoint_parity")
    parity_names = (
        "clean_formula_max_abs_error",
        "clean_natural_vs_fixed_replay_max_abs_error",
        "clean_natural_vs_fixed_replay_model_output_max_abs_error",
        "baseline_fixed_replay_vs_clean_A_matmul_natural_V0_max_abs_error",
    )
    if (
        not isinstance(parity, Mapping) or parity.get("threshold") != limit
        or parity.get("comparison") != "strict_less_than"
        or any(parity.get(name) != value for name, value in zip(parity_names, values))
    ):
        errors.append("endpoint_parity_record_invalid")


def _ig_checks(errors: list[str], provenance: Mapping[str, Any], metric: np.ndarray | None) -> None:
    result = provenance.get("ig_result")
    if not isinstance(result, Mapping):
        errors.append("ig_evidence_missing")
        return
    attempts = result.get("attempts")
    schedule = [512, 1024, 2048, 4096, 8192]
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= len(schedule):
        errors.append("ig_attempt_schedule_invalid")
        return
    passes: list[bool] = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping) or set(attempt) != {"steps", "delta", "sum", "abs", "rel", "pass"} or attempt.get("steps") != schedule[index]:
            errors.append("ig_attempt_record_invalid")
            return
        scalar: dict[str, Any] = {}
        for name in ("delta", "sum", "abs", "rel"):
            item = attempt.get(name)
            scalar[name] = item[0] if isinstance(item, list) and len(item) == 1 else None
        passed = attempt.get("pass")
        passed_value = passed[0] if isinstance(passed, list) and len(passed) == 1 and type(passed[0]) is bool else None
        if any(not _number(scalar[name]) for name in scalar) or passed_value is None:
            errors.append("ig_attempt_record_invalid")
            return
        expected_abs = abs(float(scalar["delta"]) - float(scalar["sum"]))
        expected_rel = expected_abs / max(abs(float(scalar["delta"])), float(np.finfo(np.float32).eps))
        expected_pass = expected_abs <= 1e-4 + 0.01 * abs(float(scalar["delta"]))
        if not _close(scalar["abs"], expected_abs, atol=1e-12, rtol=1e-6) or not _close(scalar["rel"], expected_rel, atol=1e-12, rtol=1e-6) or passed_value is not expected_pass:
            errors.append("ig_attempt_conservation_invalid")
        passes.append(passed_value)
    if result.get("selected_steps") != attempts[-1]["steps"] or result.get("converged") is not True or passes != [False] * (len(passes) - 1) + [True]:
        errors.append("ig_convergence_invalid")
    final = result.get("conservation")
    if not isinstance(final, Mapping) or set(final) != {"delta", "sum", "abs", "rel", "pass"} or any(final.get(key) != attempts[-1].get(key) for key in final):
        errors.append("ig_final_conservation_invalid")
        return
    delta = final["delta"][0]
    total = final["sum"][0]
    target = provenance.get("target", {})
    direction_norm = target.get("direction_norm") if isinstance(target, Mapping) else None
    if not _close(delta, direction_norm, atol=1e-4, rtol=0.01):
        errors.append("ig_endpoint_delta_invalid")
    if metric is not None and not _close(float(metric.astype(np.float64).sum()), total, atol=1e-4, rtol=0.01):
        errors.append("ig_l1_conservation_invalid")


def _matched_checks(errors: list[str], provenance: Mapping[str, Any], metric: np.ndarray | None) -> None:
    record = provenance.get("matched_contribution")
    target = provenance.get("target")
    if not isinstance(record, Mapping) or not isinstance(target, Mapping):
        errors.append("matched_conservation_missing")
        return
    required = ("cls_contribution", "token_sum", "scalar_delta_f_z_minus_f_x0", "direction_norm", "conservation_abs_error", "conservation_tolerance")
    if record.get("status") != "OK" or record.get("token_contribution_count") != 197 or any(not _number(record.get(name)) for name in required):
        errors.append("matched_conservation_record_invalid")
        return
    delta = float(record["scalar_delta_f_z_minus_f_x0"])
    direction = float(record["direction_norm"])
    threshold = target.get("direction_threshold")
    if direction != delta or direction != target.get("direction_norm") or not _number(threshold) or not direction > float(threshold) >= 0.0:
        errors.append("matched_direction_invalid")
    expected_error = abs(float(record["token_sum"]) - delta)
    expected_tolerance = 1e-6 + 1e-5 * abs(delta)
    if not _close(record["conservation_abs_error"], expected_error, atol=1e-12, rtol=1e-7) or not _close(record["conservation_tolerance"], expected_tolerance, atol=1e-15, rtol=1e-12) or float(record["conservation_abs_error"]) > float(record["conservation_tolerance"]):
        errors.append("matched_token_conservation_failed")
    if metric is not None:
        reconstructed = float(metric.astype(np.float64).sum()) + float(record["cls_contribution"])
        if not _close(reconstructed, record["token_sum"], atol=1e-6, rtol=1e-5):
            errors.append("matched_l1_conservation_invalid")


def _curve_checks(
    errors: list[str], l0: Mapping[str, Any], metric: np.ndarray | None,
    persisted: np.ndarray | None, unit: Any, target_digest: str | None,
    *, namespace: str, clean_input_sha256: str | None,
) -> tuple[dict[str, Any], ...]:
    records = l0.get("fidelity_curves")
    if not isinstance(records, list) or len(records) != 10:
        errors.append("fidelity_curve_records_invalid")
        return ()
    fractions = [index / 20.0 for index in range(21)]
    counts = [0] + [math.ceil(fraction * 3136) for fraction in fractions[1:]]
    actual = [count / 3136 for count in counts]
    expected_coordinates = [(direction, repeat) for direction in ("insertion", "deletion") for repeat in range(5)]
    metric_hash = None if metric is None else _array_sha256(metric)
    score_hash = None
    if metric is not None:
        score_hash = _array_sha256(
            np.maximum(metric, np.float32(0.0)).astype(np.float32, copy=False)
        )
    accepted: list[dict[str, Any]] = []
    expected_input_pair = (
        _input_pair_sha256(clean_input_sha256)
        if _sha(clean_input_sha256) else None
    )
    for index, (raw, coordinate) in enumerate(zip(records, expected_coordinates)):
        if not isinstance(raw, Mapping) or set(raw) != CURVE_KEYS:
            errors.append(f"fidelity_curve_schema_invalid:{index}")
            continue
        record = dict(raw)
        direction, repeat = coordinate
        try:
            seed = _seed(
                namespace, _field(unit, "sample_id"),
                _field(unit, "block_index"), _field(unit, "head_index"),
                repeat,
            )
        except (AttributeError, TypeError, ValueError):
            errors.append(f"fidelity_curve_unit_coordinate_invalid:{index}")
            continue
        identity_ok = (
            record.get("status") == "OK" and record.get("method_id") == _field(unit, "method_id")
            and record.get("comparison_line") == "value_path" and record.get("target_ref") == _field(unit, "target_id")
            and record.get("sample_id") == _field(unit, "sample_id") and record.get("block") == _field(unit, "block_index")
            and record.get("head") == _field(unit, "head_index") and (record.get("direction"), record.get("replicate")) == coordinate
            and record.get("scale") == "fine_4x4_cell"
            and record.get("ranking_kind")
            == "target_oriented_positive_tie_permuted"
            and record.get("seed_namespace") == namespace and record.get("seed") == seed
            and record.get("target_identity_sha256") == target_digest
        )
        if not identity_ok:
            errors.append(f"fidelity_curve_identity_invalid:{index}")
        hashes = ("target_identity_sha256", "tie_priority_sha256", "source_unit_map_sha256", "score_sha256", "input_pair_sha256", "ranking_sha256")
        if any(not _sha(record.get(name)) for name in hashes) or not isinstance(record.get("intervention_sha256"), list) or len(record["intervention_sha256"]) != 21 or any(not _sha(item) for item in record["intervention_sha256"]):
            errors.append(f"fidelity_curve_hash_invalid:{index}")
        if record.get("tie_priority_sha256") != _tie_priority_sha256(3136, seed) or record.get("source_unit_map_sha256") != metric_hash or record.get("score_sha256") != score_hash:
            errors.append(f"fidelity_curve_source_invalid:{index}")
        if (
            metric is not None
            and record.get("ranking_sha256")
            != _target_oriented_ranking_sha256(metric, seed)
        ):
            errors.append(f"fidelity_curve_ranking_invalid:{index}")
        # The clean tensor hash is independently bound to the retained target
        # identity and the baseline bytes are fixed by the Stage D contract, so
        # the pair digest itself is recomputable without opening an image.
        input_pair = record.get("input_pair_sha256")
        if expected_input_pair is not None and input_pair != expected_input_pair:
            errors.append(f"fidelity_curve_input_pair_binding_invalid:{index}")
        # Intermediate intervention tensors are not retained.  Their digests
        # therefore remain opaque commitments.  The endpoint tensors are the
        # exception: their digests must bind the shared retained clean identity
        # and the contract-defined normalized-zero baseline.
        interventions = record.get("intervention_sha256")
        if (
            isinstance(interventions, list)
            and len(interventions) == 21
            and all(_sha(item) for item in interventions)
        ):
            expected_endpoints = (
                (NORMALIZED_ZERO_BASELINE_SHA256, clean_input_sha256)
                if direction == "insertion"
                else (clean_input_sha256, NORMALIZED_ZERO_BASELINE_SHA256)
            )
            if tuple((interventions[0], interventions[-1])) != expected_endpoints:
                errors.append(
                    f"fidelity_curve_intervention_endpoint_binding_invalid:{index}"
                )
        if record.get("curve_fractions") != fractions or record.get("unit_counts") != counts or record.get("actual_fractions") != actual:
            errors.append(f"fidelity_curve_axis_invalid:{index}")
        responses = record.get("responses")
        if not isinstance(responses, list) or len(responses) != 21 or any(not _number(item) for item in responses):
            errors.append(f"fidelity_curve_response_invalid:{index}")
            continue
        endpoints = (0.0, 1.0) if direction == "insertion" else (1.0, 0.0)
        if responses[0] != endpoints[0] or responses[-1] != endpoints[1]:
            errors.append(f"fidelity_curve_endpoint_invalid:{index}")
        calculated_auc = _auc(actual, responses)
        if calculated_auc is None or not _close(record.get("auc"), calculated_auc, atol=1e-12, rtol=1e-12):
            errors.append(f"fidelity_curve_auc_invalid:{index}")
        if record.get("response_clipped") is not False or record.get("auc_integration") != "trapezoid_over_actual_k_divided_by_units":
            errors.append(f"fidelity_curve_semantics_invalid:{index}")
        if persisted is not None and not np.array_equal(np.asarray(responses, dtype=np.float64), persisted[index // 5, index % 5]):
            errors.append(f"fidelity_curve_l1_mismatch:{index}")
        accepted.append(record)
    return tuple(accepted)


def _aggregate_checks(errors: list[str], l0: Mapping[str, Any], records: tuple[dict[str, Any], ...], persisted: np.ndarray | None) -> None:
    aggregates = l0.get("fidelity_aggregates")
    if not isinstance(aggregates, list) or len(aggregates) != 2 or len(records) != 10:
        errors.append("fidelity_aggregates_invalid")
        return
    for direction_index, raw in enumerate(aggregates):
        if not isinstance(raw, Mapping) or set(raw) != AGGREGATE_KEYS:
            errors.append(f"fidelity_aggregate_schema_invalid:{direction_index}")
            continue
        expected_curves = list(records[direction_index * 5:(direction_index + 1) * 5])
        if (raw.get("status") != "OK" or raw.get("replicate_role") != "paired_nuisance_resolution_not_independent_samples" or raw.get("replicate_count") != 5 or raw.get("standard_deviation") != "population_ddof_0_over_complete_five_replicates" or raw.get("curves") != expected_curves):
            errors.append(f"fidelity_aggregate_binding_invalid:{direction_index}")
        if persisted is None:
            continue
        matrix = persisted[direction_index]
        aucs = np.asarray([_auc(record["actual_fractions"], record["responses"]) for record in expected_curves], dtype=np.float64)
        expected = {
            "response_mean": matrix.mean(axis=0),
            "response_std": matrix.std(axis=0, ddof=0),
        }
        for name, value in expected.items():
            try:
                observed = np.asarray(raw.get(name), dtype=np.float64)
            except (TypeError, ValueError):
                observed = np.empty(0)
            if observed.shape != (21,) or not np.allclose(observed, value, atol=1e-12, rtol=1e-12):
                errors.append(f"fidelity_aggregate_{name}_invalid:{direction_index}")
        if not _close(raw.get("auc_mean"), float(aucs.mean()), atol=1e-12, rtol=1e-12) or not _close(raw.get("auc_std"), float(aucs.std(ddof=0)), atol=1e-12, rtol=1e-12):
            errors.append(f"fidelity_aggregate_auc_invalid:{direction_index}")


def _top20_checks(errors: list[str], l0: Mapping[str, Any], records: tuple[dict[str, Any], ...]) -> None:
    values = l0.get("top20")
    if not isinstance(values, list) or len(values) != 5 or len(records) != 10:
        errors.append("top20_invalid")
        return
    for repeat, raw in enumerate(values):
        insertion, deletion = records[repeat], records[5 + repeat]
        if not isinstance(raw, Mapping) or set(raw) != TOP20_KEYS:
            errors.append(f"top20_schema_invalid:{repeat}")
            continue
        paired = ("method_id", "comparison_line", "target_ref", "target_identity_sha256", "ranking_kind", "sample_id", "block", "head", "replicate", "seed", "tie_priority_sha256", "input_pair_sha256", "ranking_sha256", "scale")
        if any(raw.get(name) != insertion.get(name) or raw.get(name) != deletion.get(name) for name in paired):
            errors.append(f"top20_pair_binding_invalid:{repeat}")
        sufficiency = insertion["responses"][4]
        comprehensiveness = 1.0 - deletion["responses"][4]
        if (raw.get("status") != "OK" or raw.get("fraction_requested") != 0.2 or raw.get("k") != 628 or raw.get("actual_fraction") != 628 / 3136 or raw.get("sufficiency") != sufficiency or raw.get("sufficiency_gap") != 1.0 - sufficiency or raw.get("comprehensiveness") != comprehensiveness or raw.get("decision_gate") is not False):
            errors.append(f"top20_value_invalid:{repeat}")


def _execution_checks(
    errors: list[str], l0: Mapping[str, Any], unit: Any, *,
    trace_mode: str | None = None, trace_records: Any = None,
    scientific_fragment: Mapping[str, Any] | None = None,
) -> None:
    method = _field(unit, "method_id")
    lifecycle = l0.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or set(lifecycle) != {"integrity", "restoration"}:
        errors.append("lifecycle_invalid")
    else:
        integrity = lifecycle.get("integrity")
        restoration = lifecycle.get("restoration")
        if (
            not isinstance(integrity, Mapping) or set(integrity) != INTEGRITY_KEYS
            or integrity.get("schema_version") != 1
            or integrity.get("matched") is not True
            or integrity.get("restored") is not True
            or integrity.get("differences") != []
            or integrity.get("verification_error_type") is not None
            or integrity.get("verification_error_message") is not None
            or integrity.get("body_exception_type") is not None
            or not _sha(integrity.get("entry_sha256"))
            or integrity.get("exit_sha256") != integrity.get("entry_sha256")
            or type(integrity.get("signature_calls")) is not int
            or integrity.get("signature_calls") < 2
            or not isinstance(integrity.get("checked"), list)
            or any(
                not isinstance(item, str) or not item
                for item in integrity.get("checked", [])
            )
        ):
            errors.append("lifecycle_integrity_failed")
        if restoration is not None and (
            not isinstance(restoration, Mapping)
            or set(restoration) != RESTORATION_KEYS
            or restoration.get("schema_version") != 1
            or restoration.get("restored") is not True
            or restoration.get("errors") != []
            or not isinstance(restoration.get("checked"), list)
            or any(
                not isinstance(item, str) or not item
                for item in restoration.get("checked", [])
            )
            or type(restoration.get("changes_detected")) is not bool
            or not isinstance(restoration.get("differences"), list)
            or any(
                not isinstance(item, str) or not item
                for item in restoration.get("differences", [])
            )
        ):
            errors.append("lifecycle_restoration_failed")
    if trace_mode == "fast" and method == NNS_METHOD_ID:
        return
    records = (
        l0.get("map_execution_records")
        if trace_mode is None or method != NNS_METHOD_ID else trace_records
    )
    expected = [("clean", 0)] + (
        [("repeat_clean", 1)] if _field(unit, "audit") is True else []
    )
    if not isinstance(records, list) or len(records) != len(expected):
        errors.append("map_execution_records_invalid")
        return
    for record, coordinate in zip(records, expected):
        if not isinstance(record, Mapping) or set(record) != MAP_EXECUTION_KEYS or record.get("method_id") != _field(unit, "method_id") or (record.get("input_variant"), record.get("repeat_index")) != coordinate or record.get("status") != "completed" or record.get("exception") is not None:
            errors.append("map_execution_record_failed")
            continue
        calls = record.get("actual_calls")
        calls_valid = (
            isinstance(calls, Mapping)
            and set(calls) == {
                "accounting_unit", "forward_calls", "backward_calls"
            }
            and calls.get("accounting_unit") == "batched_model_invocation"
            and type(calls.get("forward_calls")) is int
            and calls.get("forward_calls") >= 0
            and type(calls.get("backward_calls")) is int
            and calls.get("backward_calls") >= 0
            and record.get("backward_invocations")
            == calls.get("backward_calls")
            and type(record.get("attempted_steps_total")) is int
            and record.get("attempted_steps_total") >= 0
        )
        if not calls_valid:
            errors.append("map_execution_call_accounting_invalid")
        forward_calls = calls.get("forward_calls") if isinstance(calls, Mapping) else None
        backward_calls = calls.get("backward_calls") if isinstance(calls, Mapping) else None
        if method == NNS_METHOD_ID:
            observed_ig = record.get("ig_result")
            retained = (
                l0.get("provenance_fragment")
                if trace_mode is None else scientific_fragment
            )
            retained_ig = (
                retained.get("ig_result")
                if isinstance(retained, Mapping) else None
            )
            observed_shape_valid = (
                isinstance(observed_ig, Mapping)
                and set(observed_ig)
                == {"selected_steps", "converged", "attempts", "metadata"}
            )
            attempts = observed_ig.get("attempts") if observed_shape_valid else None
            steps_valid = (
                isinstance(attempts, list) and bool(attempts)
                and all(
                    isinstance(item, Mapping)
                    and type(item.get("steps")) is int
                    and item.get("steps") > 0
                    for item in attempts
                )
            )
            if (
                not observed_shape_valid or not steps_valid
                or not isinstance(retained_ig, Mapping)
                or any(
                    observed_ig.get(name) != retained_ig.get(name)
                    for name in observed_ig
                )
            ):
                errors.append("map_execution_ig_evidence_missing")
            else:
                assert isinstance(attempts, list)
                steps = [item["steps"] for item in attempts]
                attempted = sum(steps)
                backward = sum(math.ceil(step / 8) for step in steps)
                metadata = observed_ig.get("metadata")
                expected_metadata = {
                    "method": "integrated_gradients",
                    "quadrature": "gauss_legendre",
                    "integration_dtype": "float32",
                    "input_shape": [1, 3, 224, 224],
                    "step_schedule": [512, 1024, 2048, 4096, 8192],
                    "step_chunk_size": 8, "atol": 1e-4, "rtol": 0.01,
                    "scope": "fixed_clean_attention_head_direction_value_path",
                    "target": {
                        "kind": _field(unit, "target_id"),
                        "block_index": _field(unit, "block_index"),
                        "head_index": _field(unit, "head_index"),
                        "query_token": 0,
                    },
                    "includes_residual_paths": False,
                    "is_complete_attention_attribution": False,
                    "fixed_clean_attention": True, "frozen_direction": True,
                    "intervention_calls": backward + 2,
                    "pre_projection_calls": backward + 2,
                }
                if (
                    not isinstance(metadata, Mapping)
                    or set(metadata) != set(expected_metadata)
                    or any(metadata.get(key) != value for key, value in expected_metadata.items())
                    or record.get("attempted_steps_total") != attempted
                    or record.get("backward_invocations") != backward
                    or backward_calls != backward
                    or forward_calls != backward + 4
                ):
                    errors.append("map_execution_ig_accounting_invalid")
        elif record.get("ig_result") is not None:
            errors.append("matched_map_execution_has_ig_evidence")
        elif (
            record.get("attempted_steps_total") != 0
            or record.get("backward_invocations") != 0
            or backward_calls != 0
            or forward_calls != 0
        ):
            errors.append("matched_map_execution_accounting_invalid")


def validate_stage_d_scientific_evidence(
    l0: Any, metric: Any, curves: Any, audit: Mapping[str, Any] | None, unit: Any,
    *, endpoint_parity_limit: float = 1e-6, repeatability_atol: float = 1e-6,
    repeatability_rtol: float = 1e-5,
    fidelity_seed_namespace: str = "stage-d-fidelity|20260821|",
    provenance: Mapping[str, Any] | None = None,
) -> StageDScientificValidation:
    """Validate one successful attempt and return hash-only pair evidence."""

    errors: list[str] = []
    audit_unit = _field(unit, "audit") is True
    if not isinstance(l0, Mapping):
        metric_array, curve_array = _artifact_shapes(
            errors, metric, curves, audit, audit_unit=audit_unit
        )
        return StageDScientificValidation(tuple(errors + ["l0_not_object"]), None)
    if set(l0) == {"endpoint_parity"}:
        metric_array, curve_array = _artifact_shapes(
            errors, metric, curves, audit, audit_unit=audit_unit
        )
        value = l0.get("endpoint_parity")
        if not _number(value) or abs(float(value)) >= endpoint_parity_limit:
            errors.append("synthetic_endpoint_parity_failed")
        if audit_unit and isinstance(audit, Mapping) and metric_array is not None:
            repeat = audit.get("repeat_clean_metric_ready_map")
            if isinstance(repeat, np.ndarray) and repeat.dtype == np.float32 and repeat.shape == (56, 56) and np.isfinite(repeat).all() and not np.allclose(metric_array, repeat, atol=repeatability_atol, rtol=repeatability_rtol):
                errors.append("audit_repeatability_failed")
        return StageDScientificValidation(tuple(errors), None)
    recognized_envelope = (
        set(l0) == PRODUCTION_L0_KEYS
        or "trace_mode" in l0
    )
    if not recognized_envelope:
        return StageDScientificValidation(
            tuple(errors + ["production_l0_envelope_invalid"]), None
        )
    method = _field(unit, "method_id")
    nns = method == NNS_METHOD_ID
    trace_mode, scientific_fragment, trace_records = _trace_view(
        errors, l0, unit, nns=nns
    )
    metric_array, curve_array = _artifact_shapes(
        errors, metric, curves, audit, audit_unit=audit_unit,
        trace_mode=trace_mode, nns=nns,
    )
    if method not in {NNS_METHOD_ID, MATCHED_METHOD_ID}:
        errors.append("unit_method_invalid")
    expected_projection = "pixel_mass_to_patch_sum" if nns else "patch_mass_to_pixel_uniform"
    prediction = l0.get("prediction")
    truth = (
        prediction.get("official_truth_index")
        if isinstance(prediction, Mapping) else None
    )
    top1 = (
        prediction.get("clean_top1_index")
        if isinstance(prediction, Mapping) else None
    )
    expected_correct = None if truth is None else top1 == truth
    prediction_valid = (
        isinstance(prediction, Mapping)
        and set(prediction) == {
            "clean_top1_index", "official_truth_index", "clean_top1_correct"
        }
        and type(prediction.get("clean_top1_index")) is int
        and 0 <= prediction.get("clean_top1_index") < 1000
        and (truth is None or (
            type(truth) is int and 0 <= truth < 1000
        ))
        and prediction.get("clean_top1_correct") is expected_correct
    )
    if (l0.get("status") != "OK" or l0.get("na") is not None or l0.get("method_id") != method or l0.get("method_version") != "1.0.0" or l0.get("target_ref") != _field(unit, "target_id") or l0.get("sample_id") != _field(unit, "sample_id") or l0.get("comparison_line") != "value_path" or l0.get("projection") != expected_projection or l0.get("sign_semantics") != "signed" or not prediction_valid or l0.get("fidelity_enabled") is not True or l0.get("stability_enabled") is not False or l0.get("stability") != [] or l0.get("perturbations") != []):
        errors.append("l0_scope_or_identity_invalid")

    attempt_provenance = provenance
    target_identity = l0.get("target_identity")
    clean_hashes = l0.get("clean_tensor_hashes")
    target_digest = _identity_sha256(target_identity) if isinstance(target_identity, Mapping) else None
    trace_optional = nns and trace_mode == "fast"
    if (
        (not trace_optional and not isinstance(scientific_fragment, Mapping))
        or not isinstance(target_identity, Mapping)
        or set(target_identity) != {
            "comparison_line", "target_ref", "clean_input_sha256",
            "frozen_target",
        }
        or not isinstance(clean_hashes, Mapping)
        or set(clean_hashes) != {
            "canonical_rgb_sha256", "normalized_tensor_sha256",
        }
        or any(
            not _sha(clean_hashes.get(name))
            for name in ("canonical_rgb_sha256", "normalized_tensor_sha256")
        )
    ):
        errors.append("target_identity_invalid")
    else:
        frozen = target_identity.get("frozen_target")
        candidate = frozen.get("candidate") if isinstance(frozen, Mapping) else None
        fragment_target = (
            scientific_fragment.get("target")
            if isinstance(scientific_fragment, Mapping) else None
        )
        if (target_identity.get("comparison_line") != "value_path" or target_identity.get("target_ref") != _field(unit, "target_id") or target_identity.get("clean_input_sha256") != clean_hashes.get("normalized_tensor_sha256") or (not trace_optional and frozen != fragment_target) or not _sha(target_identity.get("clean_input_sha256")) or not isinstance(candidate, Mapping) or candidate.get("block_index") != _field(unit, "block_index") or candidate.get("head_index") != _field(unit, "head_index")):
            errors.append("target_identity_binding_invalid")
    if isinstance(scientific_fragment, Mapping):
        _endpoint_checks(errors, scientific_fragment, nns=nns, limit=endpoint_parity_limit)
        if nns:
            _ig_checks(errors, scientific_fragment, metric_array)
        else:
            _matched_checks(errors, scientific_fragment, metric_array)
    _execution_checks(
        errors, l0, unit, trace_mode=trace_mode,
        trace_records=trace_records, scientific_fragment=scientific_fragment,
    )

    repeatability = l0.get("repeatability")
    if audit_unit:
        if not isinstance(repeatability, list) or len(repeatability) != 1 or not isinstance(repeatability[0], Mapping):
            errors.append("audit_repeatability_record_invalid")
        elif (
            isinstance(audit, Mapping) and metric_array is not None
            and isinstance(audit.get("repeat_clean_metric_ready_map"), np.ndarray)
            and audit["repeat_clean_metric_ready_map"].dtype == np.float32
            and audit["repeat_clean_metric_ready_map"].shape == (56, 56)
            and np.isfinite(audit["repeat_clean_metric_ready_map"]).all()
        ):
            record = repeatability[0]
            repeat = audit["repeat_clean_metric_ready_map"]
            difference = np.abs(metric_array.astype(np.float64) - repeat.astype(np.float64))
            passed = bool(np.allclose(metric_array, repeat, atol=repeatability_atol, rtol=repeatability_rtol))
            if set(record) != {"repeat_index", "atol", "rtol", "max_abs_error", "pass"} or record.get("repeat_index") != 1 or record.get("atol") != repeatability_atol or record.get("rtol") != repeatability_rtol or record.get("max_abs_error") != float(difference.max()) or record.get("pass") is not passed or not passed:
                errors.append("audit_repeatability_failed")
    elif repeatability != []:
        errors.append("non_audit_repeatability_present")

    records = _curve_checks(
        errors, l0, metric_array, curve_array, unit, target_digest,
        namespace=fidelity_seed_namespace,
        clean_input_sha256=(
            target_identity.get("clean_input_sha256")
            if isinstance(target_identity, Mapping) else None
        ),
    )
    _aggregate_checks(errors, l0, records, curve_array)
    _top20_checks(errors, l0, records)
    fractions = [index / 20.0 for index in range(21)]
    provenance_binding = _provenance_pair_binding(
        errors, attempt_provenance, unit, fractions, fidelity_seed_namespace
    )
    pair_evidence = None
    if target_digest is not None and len(records) == 10 and provenance_binding is not None:
        pair_evidence = {
            **provenance_binding,
            "target_identity_sha256": target_digest,
            "clean_input_sha256": target_identity.get("clean_input_sha256"),
            "block_index": _field(unit, "block_index"),
            "head_index": _field(unit, "head_index"),
            "input_pair_sha256": tuple(record.get("input_pair_sha256") for record in records),
            "seeds": tuple(record.get("seed") for record in records),
        }
    return StageDScientificValidation(tuple(errors), pair_evidence)


def validate_stage_d_pair_bindings(
    nns_binding: Mapping[str, Any] | None,
    matched_binding: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Check that two independently validated units share one experiment.

    Only target/schedule hashes and public coordinates enter the returned
    evidence, so callers can report the error codes without exposing response
    values or maps.  Input-pair and intervention hashes are opaque producer
    commitments because Stage D does not retain their source tensors; equality
    here verifies the frozen sharing contract, not their tensor contents.
    """

    errors: list[str] = []
    left = nns_binding
    right = matched_binding
    if left is None or right is None:
        return ("pair_binding_evidence_missing",)
    if (
        not isinstance(left, Mapping) or not isinstance(right, Mapping)
        or set(left) != PAIR_BINDING_KEYS or set(right) != PAIR_BINDING_KEYS
    ):
        return ("pair_binding_evidence_invalid",)
    digest_names = PAIR_BINDING_KEYS - {
        "block_index", "head_index", "input_pair_sha256", "seeds",
    }
    if (
        any(not _sha(left.get(name)) or not _sha(right.get(name)) for name in digest_names)
        or type(left.get("block_index")) is not int
        or type(right.get("block_index")) is not int
        or type(left.get("head_index")) is not int
        or type(right.get("head_index")) is not int
        or any(
            not isinstance(binding.get(name), tuple)
            or len(binding[name]) != 10
            for binding in (left, right)
            for name in ("input_pair_sha256", "seeds")
        )
        or any(
            not _sha(item)
            for binding in (left, right)
            for item in binding["input_pair_sha256"]
        )
        or any(
            type(item) is not int or not 0 <= item < 2**64
            for binding in (left, right) for item in binding["seeds"]
        )
    ):
        return ("pair_binding_evidence_invalid",)
    for name in (
        "target_identity_sha256", "clean_input_sha256",
        "block_index", "head_index", "input_pair_sha256", "seeds",
        "model_binding_sha256", "plan_digest", "group_digest",
        "pair_digest", "image_identity_sha256",
        "target_contract_sha256", "operator_contract_sha256",
        "projection_contract_sha256", "sign_semantics_contract_sha256",
        "normalization_contract_sha256", "baseline_contract_sha256",
        "fractions_contract_sha256", "seed_contract_sha256",
    ):
        if left.get(name) != right.get(name):
            errors.append(f"pair_{name}_mismatch")
    return tuple(errors)


# Short compatibility aliases for callers written against the initial draft.
validate_stage_d_scientific = validate_stage_d_scientific_evidence


def reconcile_stage_d_pair(
    nns: StageDScientificValidation, matched: StageDScientificValidation,
) -> tuple[str, ...]:
    if not isinstance(nns, StageDScientificValidation) or not isinstance(
        matched, StageDScientificValidation
    ):
        return ("pair_validation_type_invalid",)
    if not nns.valid or not matched.valid:
        return ("pair_member_scientifically_invalid",)
    return validate_stage_d_pair_bindings(nns.pair_binding, matched.pair_binding)


__all__ = [
    "MATCHED_METHOD_ID", "NNS_METHOD_ID", "PAIR_BINDING_KEYS",
    "PRODUCTION_L0_KEYS",
    "StageDScientificValidation", "reconcile_stage_d_pair",
    "validate_stage_d_pair_bindings", "validate_stage_d_scientific",
    "validate_stage_d_scientific_evidence",
]
