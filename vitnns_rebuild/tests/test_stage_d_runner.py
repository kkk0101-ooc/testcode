"""Targeted trace-mode forwarding tests for the Stage D runtime."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vitnns.stage_d_runtime import (
    StageDRuntimeIntegrityError, _call_unit_runner, _trace_mode_for_unit,
)


def test_unit_trace_mode_uses_explicit_value_then_audit_default() -> None:
    assert _trace_mode_for_unit(
        SimpleNamespace(trace_mode="fast", audit=False)
    ) == "fast"
    assert _trace_mode_for_unit(
        SimpleNamespace(trace_mode="full", audit=True)
    ) == "full"
    assert _trace_mode_for_unit(SimpleNamespace(audit=False)) == "fast"
    assert _trace_mode_for_unit(SimpleNamespace(audit=True)) == "full"
    with pytest.raises(StageDRuntimeIntegrityError, match="unit.audit"):
        _trace_mode_for_unit(SimpleNamespace(trace_mode="fast", audit=True))
    with pytest.raises(StageDRuntimeIntegrityError, match="unit.audit"):
        _trace_mode_for_unit(SimpleNamespace(trace_mode="full", audit=False))
    with pytest.raises(StageDRuntimeIntegrityError, match="trace_mode"):
        _trace_mode_for_unit(SimpleNamespace(trace_mode="debug", audit=False))


def test_runtime_forwards_trace_mode_to_new_unit_runner() -> None:
    observed: list[str] = []

    def runner(unit, *, trace_mode, marker):
        del unit, marker
        observed.append(trace_mode)
        return {"ok": True}

    result = _call_unit_runner(
        runner, SimpleNamespace(trace_mode="fast", audit=False), marker=object(),
    )
    assert result == {"ok": True}
    assert observed == ["fast"]


def test_runtime_adapts_legacy_runner_backend_without_breaking_signature() -> None:
    observed: list[str] = []

    class Backend:
        def run_stage_d_unit(self, group, unit, *, trace_mode, device):
            del group, unit, device
            observed.append(trace_mode)
            return "done"

    def legacy_runner(unit, *, backend, group, device):
        return backend.run_stage_d_unit(group, unit, device=device)

    unit = SimpleNamespace(audit=True)
    result = _call_unit_runner(
        legacy_runner, unit, backend=Backend(), group=object(), device="cpu",
    )
    assert result == "done"
    assert observed == ["full"]


def test_runtime_preserves_fully_legacy_runner_and_backend() -> None:
    class Backend:
        def run(self, unit, *, device):
            del unit, device
            return "legacy"

    def runner(unit, *, backend, device):
        return backend.run(unit, device=device)

    assert _call_unit_runner(
        runner, SimpleNamespace(audit=False), backend=Backend(), device="cpu",
    ) == "legacy"


def test_runtime_does_not_inject_into_variadic_legacy_backend() -> None:
    observed: list[dict[str, object]] = []

    class Backend:
        def run(self, unit, **kwargs):
            del unit
            observed.append(dict(kwargs))
            return "legacy-variadic"

    def runner(unit, *, backend, device):
        return backend.run(unit, device=device)

    assert _call_unit_runner(
        runner, SimpleNamespace(audit=False), backend=Backend(), device="cpu",
    ) == "legacy-variadic"
    assert observed == [{"device": "cpu"}]


def test_variadic_wrapper_does_not_receive_trace_mode() -> None:
    observed: list[dict[str, object]] = []

    def legacy_target(unit, *, marker):
        del unit, marker
        return "legacy-target"

    def wrapper(*args, **kwargs):
        observed.append(dict(kwargs))
        return legacy_target(*args, **kwargs)

    assert _call_unit_runner(
        wrapper, SimpleNamespace(audit=False), marker=object(),
    ) == "legacy-target"
    assert "trace_mode" not in observed[0]
