"""Synthetic contract tests for the canonical Stage D trace split."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vitnns.stage_d_engine as engine
from vitnns.stage_d_engine import (
    CanonicalStageDBackend,
    NNS_CONTRACT_KEYS,
    NNS_METHOD_ID,
    NNS_TRACE_KEYS,
)
from vitnns.stage_c_runner import ProductionStageCBackend
from vitnns.stage_d_producer import StageDBackendPayload, StageDProducerError
from vitnns.stage_d_producer import StageDPartialEvidence


MATCHED_METHOD_ID = "matched_fixed_clean_attention_patch_token_contribution"


def _unit(method_id: str = NNS_METHOD_ID, *, audit: bool = True):
    return SimpleNamespace(method_id=method_id, audit=audit)


def _payload(*, method_id: str = NNS_METHOD_ID) -> StageDBackendPayload:
    curves = [
        {"direction": "insertion", "responses": [0.0, 0.5, 1.0]},
        {"direction": "deletion", "responses": [1.0, 0.5, 0.0]},
    ]
    return StageDBackendPayload(
        l0={
            "method_id": method_id,
            "method_version": "synthetic-v1",
            "comparison_line": "value_path",
            "target_ref": "fixed_clean_attention_cls_head_direction",
            "projection": "pixel_mass_to_patch_sum",
            "sign_semantics": "signed",
            "fidelity_curves": curves,
            "provenance_fragment": {
                "ig_result": {"steps": 512, "converged": True}
            },
            "map_execution_records": [
                {"input_variant": "clean", "status": "completed"}
            ],
            # A delegated backend cannot smuggle stale trace fields through.
            "nns_contract": {"stale": True},
            "trace": {"stale": True},
        },
        l1={
            "clean": np.arange(56 * 56, dtype=np.float32).reshape(56, 56)
        },
        l2={
            "clean_native": np.arange(12, dtype=np.float32).reshape(3, 2, 2),
            "nns_trace": np.array([-1.0], dtype=np.float32),
        },
    )


def _backend(trace_builder):
    value = object.__new__(CanonicalStageDBackend)
    value._stage_d_configured_trace_mode = None
    value._stage_d_trace_builder = trace_builder
    return value


def test_canonical_backend_is_the_public_production_backend() -> None:
    from vitnns.stage_d_producer import ProductionStageDBackend

    assert CanonicalStageDBackend is ProductionStageDBackend
    backend = engine.create_stage_d_backend(
        sample_loader=lambda *_args: None
    )
    assert type(backend) is ProductionStageDBackend


def test_fast_never_constructs_or_retains_nns_trace(monkeypatch) -> None:
    source = _payload()
    calls: list[str] = []

    def forbidden_trace_builder(**_kwargs):
        calls.append("trace")
        raise AssertionError("fast mode constructed an NNS trace")

    monkeypatch.setattr(
        ProductionStageCBackend,
        "run",
        lambda _self, _unit, **_kwargs: source,
    )
    payload = _backend(forbidden_trace_builder).run(
        _unit(audit=False),
        device="cpu",
        call_counter=object(),
        trace_mode="fast",
    )

    assert calls == []
    assert payload.l0["trace_mode"] == "fast"
    assert "nns_contract" not in payload.l0
    assert "trace" not in payload.l0
    assert "provenance_fragment" not in payload.l0
    assert "map_execution_records" not in payload.l0
    assert "nns_trace" not in payload.l2
    assert payload.l2 == {}
    np.testing.assert_array_equal(payload.l1["clean"], source.l1["clean"])
    assert payload.l0["fidelity_curves"] == source.l0["fidelity_curves"]


def test_full_retains_canonical_nns_contract_trace_and_l2(monkeypatch) -> None:
    source = _payload()
    calls: list[dict] = []

    def build_trace(*, unit, l0, l2):
        calls.append({"unit": unit, "l0": l0, "l2": l2})
        return engine._default_trace_builder(unit=unit, l0=l0, l2=l2)

    monkeypatch.setattr(
        ProductionStageCBackend,
        "run",
        lambda _self, _unit, **_kwargs: source,
    )
    payload = _backend(build_trace).run(
        _unit(),
        device="cpu",
        call_counter=object(),
        trace_mode="full",
    )

    assert len(calls) == 1
    assert payload.l0["trace_mode"] == "full"
    assert set(payload.l0["nns_contract"]) == NNS_CONTRACT_KEYS
    assert payload.l0["nns_contract"]["method_id"] == NNS_METHOD_ID
    assert set(payload.l0["trace"]) == NNS_TRACE_KEYS
    assert payload.l0["trace"]["provenance_fragment"] == (
        source.l0["provenance_fragment"]
    )
    assert payload.l0["trace"]["map_execution_records"] == (
        source.l0["map_execution_records"]
    )
    assert payload.l0["provenance_fragment"] == (
        source.l0["provenance_fragment"]
    )
    assert payload.l0["map_execution_records"] == (
        source.l0["map_execution_records"]
    )
    np.testing.assert_array_equal(
        payload.l2["nns_trace"], source.l2["clean_native"]
    )
    assert payload.l2["nns_trace"] is not source.l2["clean_native"]
    assert set(source.l2) <= set(payload.l2)
    np.testing.assert_array_equal(payload.l1["clean"], source.l1["clean"])
    assert payload.l0["fidelity_curves"] == source.l0["fidelity_curves"]


@pytest.mark.parametrize(
    ("trace_mode", "audit"), [("fast", False), ("full", True)]
)
def test_matched_payload_never_builds_nns_trace(
    monkeypatch, trace_mode: str, audit: bool,
) -> None:
    source = _payload(method_id=MATCHED_METHOD_ID)
    calls: list[str] = []
    monkeypatch.setattr(
        ProductionStageCBackend,
        "run",
        lambda _self, _unit, **_kwargs: source,
    )
    payload = _backend(lambda **_kwargs: calls.append("trace")).run(
        _unit(MATCHED_METHOD_ID, audit=audit),
        device="cpu",
        call_counter=object(),
        trace_mode=trace_mode,
    )

    assert calls == []
    assert payload.l0["trace_mode"] == trace_mode
    assert "nns_contract" not in payload.l0
    assert "trace" not in payload.l0
    assert "nns_trace" not in payload.l2
    assert payload.l0["provenance_fragment"] == (
        source.l0["provenance_fragment"]
    )
    assert payload.l0["map_execution_records"] == (
        source.l0["map_execution_records"]
    )


@pytest.mark.parametrize(
    ("trace_mode", "audit"), [("fast", True), ("full", False)]
)
def test_explicit_trace_mode_must_match_audit_policy(
    trace_mode: str, audit: bool,
) -> None:
    with pytest.raises(StageDProducerError, match="disagrees with unit.audit"):
        engine.resolve_trace_mode(_unit(audit=audit), trace_mode=trace_mode)


@pytest.mark.parametrize("audit", [None, 0, 1, "yes"])
def test_trace_policy_requires_a_boolean_audit_flag(audit) -> None:
    with pytest.raises(StageDProducerError, match="boolean unit.audit"):
        engine.resolve_trace_mode(SimpleNamespace(audit=audit))


def test_fast_partial_failure_drops_only_nns_trace_material() -> None:
    partial = StageDPartialEvidence(
        l0_scalars={
            "status": "NA",
            "provenance_fragment": {"ig_result": {"attempts": []}},
            "map_execution_records": [{"status": "failed"}],
            "nns_contract": {"stale": True},
            "trace": {"stale": True},
        },
        metric_ready_map=np.zeros((56, 56), dtype=np.float32),
        audit_evidence={"nns_trace": np.ones(1, dtype=np.float32)},
    )

    filtered = engine.apply_partial_trace_policy(
        partial, unit=_unit(audit=False), trace_mode="fast"
    )

    assert filtered.l0_scalars == {"status": "NA", "trace_mode": "fast"}
    assert filtered.audit_evidence is None
    np.testing.assert_array_equal(
        filtered.metric_ready_map, partial.metric_ready_map
    )


def test_canonical_helper_revalidates_a_delegated_payload() -> None:
    payload = _payload()
    payload.l0["trace_mode"] = "fast"

    class Backend:
        def run(self, unit, *, device, call_counter, trace_mode):
            del unit, device, call_counter, trace_mode
            return payload

    filtered = engine.run_stage_d_canonical_backend(
        Backend(), _unit(audit=False), device="cpu", call_counter=object(),
        trace_mode="fast",
    )
    assert filtered.l0["trace_mode"] == "fast"
    assert not {
        "provenance_fragment", "map_execution_records",
        "nns_contract", "trace",
    } & set(filtered.l0)
    assert filtered.l2 == {}


def test_fast_backend_payload_partial_uses_its_native_shape() -> None:
    partial = _payload()
    filtered = engine.apply_partial_trace_policy(
        partial, unit=_unit(audit=False), trace_mode="fast"
    )
    assert isinstance(filtered, StageDBackendPayload)
    assert filtered.l0["trace_mode"] == "fast"
    assert "provenance_fragment" not in filtered.l0
    assert filtered.l2 == {}


@pytest.mark.parametrize("trace_mode", [None, "FAST", "trace", 1, []])
def test_trace_mode_is_a_closed_case_sensitive_enum(trace_mode) -> None:
    with pytest.raises(StageDProducerError, match="exactly 'fast' or 'full'"):
        engine.validate_trace_mode(trace_mode)
