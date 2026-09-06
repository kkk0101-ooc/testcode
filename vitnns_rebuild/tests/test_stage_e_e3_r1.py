"""E3-R1 prospective numerical-recovery and atomicity tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from jsonschema import Draft202012Validator

import vitnns.stage_e_e3_r1 as r1
from vitnns.attribution import IGResult
from vitnns.stage_d_provenance import canonical_sha256
from vitnns.stage_c_maps import (
    ComparisonLine, MethodResult, SignSemantics, raw_array, stage_c_metric_ready,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _unit() -> dict[str, object]:
    return {
        "canonical_digest": "a" * 64, "sample_id": "sample",
        "selection_order": 61, "stratum": "correct", "block": 4,
        "head": 2, "method": r1.e3.NNS_METHOD,
        "target": "fixed_clean_attention_cls_head_direction",
        "scale": "fine_4x4_cell",
        "coordinate_comparison_identity": r1.e3.comparison_identity(4, 2),
        "image_sha256": "b" * 64,
    }


def _failed_ig() -> IGResult:
    attempts = tuple({
        "steps": steps, "delta": [1.0], "sum": [0.0],
        "abs": [1.0], "rel": [1.0], "pass": [False],
    } for steps in r1.COMPLETE_SCHEDULE)
    return IGResult(
        attribution=torch.ones((1, 3, 224, 224), dtype=torch.float32),
        conservation={
            "delta": torch.tensor([1.0]), "sum": torch.tensor([0.0]),
            "abs": torch.tensor([1.0]), "rel": torch.tensor([1.0]),
            "pass": torch.tensor([False]),
        },
        steps=32768, converged=False, attempts=attempts,
        metadata={
            "step_schedule": list(r1.COMPLETE_SCHEDULE),
            "step_chunk_size": 8, "atol": 1e-4, "rtol": 0.01,
        },
    )


def _failure_result(unit: dict[str, object]) -> tuple[dict[str, object], np.ndarray]:
    diagnostic = np.ones((3, 224, 224), dtype=np.float32)
    error = SimpleNamespace(ig_result=_failed_ig())
    response = SimpleNamespace(to_dict=lambda: {
        "sigma": 1, "target_identity": {"frozen_target": {"runtime_model_binding": {"sha256": "x"}}},
    })
    target = {"frozen_target": {"runtime_model_binding": {"sha256": "x"}}}
    result = r1._failure_result(
        unit=unit, target_identity=target, response=response, error=error,
        elapsed_seconds=1.0, cpu_seconds=2.0, forward_calls=7,
        peak_cuda_bytes=8, clean_sha256="c" * 64, diagnostic=diagnostic,
    )
    return result, diagnostic


def test_config_schema_schedule_budget_and_e4_boundary() -> None:
    config = r1._validate_config(REPO_ROOT)
    Draft202012Validator.check_schema(json.loads((REPO_ROOT / r1.COMPLETION_SCHEMA_PATH).read_text()))
    assert config["canonical_digest"] == r1.EXPECTED_CONFIG_DIGEST
    assert tuple(config["numerical_recovery"]["complete_schedule"]) == r1.COMPLETE_SCHEDULE
    assert config["numerical_recovery"]["atol"] == 1e-4
    assert config["numerical_recovery"]["rtol"] == 0.01
    assert config["scope"]["pilot_orders"] == [61, 72]
    assert config["authorization"]["e4_started"] is False


def test_e3_v1_fail_and_516_2484_partition_are_preserved() -> None:
    result = r1.validate_e3_v1(REPO_ROOT)
    inherited, eligible = r1._partition(REPO_ROOT)
    assert result["status"] == "FAIL" and result["next_authorization"] == "NONE"
    assert len(inherited) == 516 and len(eligible) == 2484
    assert len([row for row in eligible if row["selection_order"] in r1.PILOT_ORDERS]) == 12


def test_r1_schedule_is_an_extension_and_historical_constant_is_unchanged() -> None:
    from vitnns import stage_c_value_path as value_path
    assert value_path.IG_STEPS == r1.ORIGINAL_SCHEDULE
    assert r1.COMPLETE_SCHEDULE == r1.ORIGINAL_SCHEDULE + (16384, 32768)
    assert "e3_r1_conditional" in value_path.e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig.__name__


def test_r1_wrapper_dispatches_only_the_frozen_extended_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    from vitnns import stage_c_value_path as value_path
    captured = {}
    native = np.ones((3, 224, 224), dtype=np.float32)
    metric = np.full((14, 14), 3.0 * 16.0 * 16.0, dtype=np.float32)
    result = MethodResult(
        method_id=value_path.NNS_METHOD_ID, method_version=value_path.METHOD_VERSION,
        comparison_line=ComparisonLine.VALUE_PATH,
        explanation_target=value_path.TARGET_REF, fidelity_target=value_path.TARGET_REF,
        class_agnostic=True, native_tensor=raw_array(native),
        metric_map=stage_c_metric_ready(metric), projection="pixel_mass_to_patch_sum",
        sign_semantics=SignSemantics.SIGNED, valid_mask=np.ones((14, 14), dtype=np.bool_),
        provenance={"intervention_input_sha256": "x", "kept": True},
    )
    def fake(*_args, **kwargs):
        captured.update(kwargs)
        return result
    monkeypatch.setattr(
        value_path,
        "conditional_fixed_clean_attention_head_direction_value_path_ig_at_intervention",
        fake,
    )
    target = SimpleNamespace(clean_input_sha256="c" * 64)
    observed = value_path.e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig(
        object(), object(), target, provenance={"stage": "E3-R1"},
    )
    assert tuple(captured["_steps_schedule"]) == r1.COMPLETE_SCHEDULE
    assert captured["_numerical_recovery_id"] == r1.RECOVERY_ID
    assert observed.provenance == {"kept": True}


def test_scientific_failure_is_atomic_and_source_validated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r1, "ROOT_PATH", "formal")
    unit = _unit()
    result, diagnostic = _failure_result(unit)
    path = r1._publish_attempt(tmp_path, unit, result, diagnostic)
    observed = r1._validate_r1_attempt(tmp_path, unit)
    assert path.is_dir() and observed["result"]["validity_status"] == "FAILED"
    assert observed["status"]["state"] == "scientific_failure"
    assert tuple(x["steps"] for x in observed["result"]["ig_result"]["attempts"]) == r1.COMPLETE_SCHEDULE
    with pytest.raises(r1.StageEE3R1Error, match="overwrite"):
        r1._publish_attempt(tmp_path, unit, result, diagnostic)


def test_original_schedule_only_failure_cannot_be_published_as_r1_terminal() -> None:
    unit = _unit()
    incomplete = _failed_ig()
    object.__setattr__(incomplete, "attempts", incomplete.attempts[:5])
    object.__setattr__(incomplete, "steps", 8192)
    error = SimpleNamespace(ig_result=incomplete)
    response = SimpleNamespace(to_dict=lambda: {"sigma": 1})
    target = {"frozen_target": {"runtime_model_binding": {"sha256": "x"}}}
    with pytest.raises(r1.StageEE3R1Error, match="exhaust"):
        r1._failure_result(
            unit=unit, target_identity=target, response=response, error=error,
            elapsed_seconds=1.0, cpu_seconds=2.0, forward_calls=7,
            peak_cuda_bytes=8, clean_sha256="c" * 64,
            diagnostic=np.ones((3, 224, 224), dtype=np.float32),
        )


def test_fault_before_rename_leaves_no_terminal_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r1, "ROOT_PATH", "formal")
    unit = _unit()
    result, diagnostic = _failure_result(unit)
    def fail(point: str) -> None:
        if point == "before_rename":
            raise RuntimeError("injected before rename")
    with pytest.raises(RuntimeError, match="before rename"):
        r1._publish_attempt(tmp_path, unit, result, diagnostic, fault_injector=fail)
    assert not r1._r1_attempt_paths(tmp_path, unit["canonical_digest"])[0].exists()
    attempts = tmp_path / "formal/units" / unit["canonical_digest"] / "attempts"
    assert not list(attempts.glob(".attempt-r1-*"))


def test_fault_after_rename_leaves_a_valid_terminal_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r1, "ROOT_PATH", "formal")
    unit = _unit()
    result, diagnostic = _failure_result(unit)
    def fail(point: str) -> None:
        if point == "after_rename":
            raise RuntimeError("injected after rename")
    with pytest.raises(RuntimeError, match="after rename"):
        r1._publish_attempt(tmp_path, unit, result, diagnostic, fault_injector=fail)
    observed = r1._validate_r1_attempt(tmp_path, unit)
    assert observed["status"]["terminal"] is True


def test_incomplete_or_tampered_failure_cannot_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r1, "ROOT_PATH", "formal")
    unit = _unit()
    result, diagnostic = _failure_result(unit)
    root = r1._publish_attempt(tmp_path, unit, result, diagnostic)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["payload_sha256"] = "0" * 64
    manifest["canonical_digest"] = canonical_sha256({key: value for key, value in manifest.items() if key != "canonical_digest"})
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(r1.StageEE3R1Error, match="control binding"):
        r1._validate_r1_attempt(tmp_path, unit)


def test_bulk_requires_passing_pilot_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r1, "PILOT_GATE_PATH", "formal/gate.json")
    monkeypatch.setattr(r1, "BULK_AUTH_PATH", "formal/auth.json")
    root = tmp_path / "formal"
    root.mkdir()
    for name, value in (
        ("gate.json", r1._signed({"status": "FAIL"})),
        ("auth.json", r1._signed({"status": "NOT_AUTHORIZED"})),
    ):
        (root / name).write_bytes(r1.canonical_json_bytes(value) + b"\n")
    monkeypatch.setattr(r1, "validate_initialized", lambda _repo: {"status": "PASS"})
    with pytest.raises(r1.StageEE3R1Error, match="not authorized"):
        r1._run_orders(tmp_path, orders=(3,), device="cpu", phase="BULK")


def test_partial_union_cannot_enter_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unit = _unit()
    monkeypatch.setattr(r1, "_source_attempt", lambda _repo, _unit: None)
    with pytest.raises(r1.StageEE3R1Error, match="missing terminal union"):
        r1._combined_new_rows(tmp_path, {"units": [unit]})


def test_no_e4_execution_surface() -> None:
    public = {name for name in dir(r1) if name.startswith(("run_", "start_", "execute_"))}
    assert public == {"run_pilot", "run_bulk_shard"}
