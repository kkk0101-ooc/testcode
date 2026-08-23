from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import vitnns.stage_c_runner as runner
from vitnns.stage_c_provenance import canonical_sha256


def _control_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _unit_key() -> dict[str, Any]:
    from vitnns.stage_c_provenance import unit_identity_digest

    key = {
        "schema_version": 1,
        "sample_id": "calibration-1",
        "method_id": "full_model_class_logit_ig",
        "config_id": "global",
        "block_index": None,
        "head_index": None,
        "target_ref": "clean_predicted_class_raw_logit",
    }
    key["canonical_digest"] = "0" * 64
    key["canonical_digest"] = unit_identity_digest(key)
    return key


@dataclass(frozen=True)
class _Plan:
    units: tuple[dict[str, Any], ...]
    variant: str = "expected"

    @property
    def digest(self) -> str:
        return canonical_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "document_type": "test_execution_plan",
            "variant": self.variant,
            "units": list(self.units),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "canonical_digest": self.digest}


def _estimate(plan: _Plan, *, marker: str = "expected") -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "document_type": "stage_c_preflight_estimate",
        "plan_digest": plan.digest,
        "marker": marker,
        "within_budget": True,
        "violations": [],
        "caps": {"effective": {}},
    }
    value["canonical_digest"] = canonical_sha256(value)
    return value


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, root: Path | None = None,
) -> tuple[runner.BatchRunResult, list[str]]:
    plan = _Plan((_unit_key(),))
    calls: list[str] = []

    def fake_run(unit: Any, *, output_root: Path, **kwargs: Any) -> runner.UnitRunResult:
        calls.append("backend")
        # This models the first production convergence failure at the point
        # where its durable attempt is created: both batch identity controls
        # must already be present and exact.
        assert (output_root / runner.EXECUTION_PLAN_FILENAME).read_bytes() == (
            _control_bytes(plan.to_dict())
        )
        assert (output_root / runner.PREFLIGHT_FILENAME).read_bytes() == (
            _control_bytes(_estimate(plan))
        )
        return runner.UnitRunResult(
            unit, runner.UNIT_FAILED, None, False,
            {"reason": "ig_convergence_failed"},
        )

    monkeypatch.setattr(runner, "run_execution_unit", fake_run)
    result = runner.run_stage_c_batch(
        plan, output_root=tmp_path / "run" if root is None else root,
        backend=object(), estimate=_estimate(plan),
        _allow_unissued_estimate=True,
    )
    return result, calls


def test_first_failure_sees_controls_before_attempt_and_stays_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls = _run(tmp_path, monkeypatch)

    assert calls == ["backend"]
    assert result.unit_results[0].state == runner.UNIT_FAILED
    assert result.unit_results[0].state != runner.UNIT_INCOMPLETE


def test_existing_exact_canonical_controls_are_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    plan = _Plan((_unit_key(),))
    plan_path = root / runner.EXECUTION_PLAN_FILENAME
    preflight_path = root / runner.PREFLIGHT_FILENAME
    plan_path.write_bytes(_control_bytes(plan.to_dict()))
    preflight_path.write_bytes(_control_bytes(_estimate(plan)))
    plan_stat = plan_path.stat()
    preflight_stat = preflight_path.stat()

    result, calls = _run(tmp_path, monkeypatch, root=root)

    assert result.unit_results[0].state == runner.UNIT_FAILED
    assert calls == ["backend"]
    assert plan_path.stat().st_ino == plan_stat.st_ino
    assert preflight_path.stat().st_ino == preflight_stat.st_ino


@pytest.mark.parametrize("control", ["plan", "preflight"])
def test_control_drift_is_rejected_before_backend_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    plan = _Plan((_unit_key(),))
    if control == "plan":
        path = root / runner.EXECUTION_PLAN_FILENAME
        path.write_bytes(_control_bytes(_Plan(plan.units, "drift").to_dict()))
    else:
        path = root / runner.PREFLIGHT_FILENAME
        path.write_bytes(_control_bytes(_estimate(plan, marker="drift")))
    original = path.read_bytes()
    calls: list[str] = []

    def forbidden_backend(*args: Any, **kwargs: Any) -> runner.UnitRunResult:
        calls.append("backend")
        raise AssertionError("backend called before immutable-control validation")

    monkeypatch.setattr(runner, "run_execution_unit", forbidden_backend)
    with pytest.raises(runner.StageCRunnerError, match="content drift"):
        runner.run_stage_c_batch(
            plan, output_root=root, backend=object(), estimate=_estimate(plan),
            _allow_unissued_estimate=True,
        )

    assert calls == []
    assert path.read_bytes() == original


@pytest.mark.parametrize("kind", ["noncanonical", "symlink"])
def test_non_regular_or_noncanonical_control_is_rejected_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    plan = _Plan((_unit_key(),))
    path = root / runner.EXECUTION_PLAN_FILENAME
    if kind == "noncanonical":
        path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
        match = "not canonically encoded"
    else:
        target = tmp_path / "outside.json"
        target.write_bytes(_control_bytes(plan.to_dict()))
        path.symlink_to(target)
        match = "non-symlink regular file"
    calls: list[str] = []

    def forbidden_backend(*args: Any, **kwargs: Any) -> runner.UnitRunResult:
        calls.append("backend")
        raise AssertionError("backend called before immutable-control validation")

    monkeypatch.setattr(runner, "run_execution_unit", forbidden_backend)
    with pytest.raises(runner.StageCRunnerError, match=match):
        runner.run_stage_c_batch(
            plan, output_root=root, backend=object(), estimate=_estimate(plan),
            _allow_unissued_estimate=True,
        )
    assert calls == []
