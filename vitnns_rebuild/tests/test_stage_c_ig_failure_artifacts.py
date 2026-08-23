"""Focused semantic validation for retained production IG failures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vitnns.attribution import IGResult
from vitnns.stage_c_artifacts import validate_execution_unit
from vitnns.stage_c_class_logit import StageCClassLogitIGConvergenceError
from vitnns.stage_c_config import load_stage_c_config
from vitnns.stage_c_plan import (
    build_execution_plan, estimate_stage_c_run, load_stage_c_registry,
    make_run_spec,
)
from vitnns.stage_c_perturbations import tensor_sha256
from vitnns.stage_c_provenance import canonical_sha256
from vitnns.stage_c_runner import ProductionStageCBackend, run_execution_unit

SAMPLE = "ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG"
METHOD = "full_model_class_logit_ig"
ERROR_TYPE = (
    "vitnns.stage_c_class_logit.StageCClassLogitIGConvergenceError"
)
STEPS = (512, 1024, 2048, 4096, 8192)


def _canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def _ig(*, converged: bool, class_index: int = 17):
    selected = 512 if converged else STEPS[-1]
    schedule = STEPS[:1] if converged else STEPS
    passed = converged
    return IGResult(
        attribution=torch.zeros((1, 3, 1, 1), dtype=torch.float32),
        conservation={
            "delta": torch.tensor([1.0]), "sum": torch.tensor([1.0 if passed else 0.0]),
            "abs": torch.tensor([0.0 if passed else 1.0]),
            "rel": torch.tensor([0.0 if passed else 1.0]),
            "pass": torch.tensor([passed]),
        },
        steps=selected, converged=converged, attempts=tuple(
            {
                "steps": step, "delta": [1.0],
                "sum": [1.0 if passed and step == selected else 0.0],
                "abs": [0.0 if passed and step == selected else 1.0],
                "rel": [0.0 if passed and step == selected else 1.0],
                "pass": [bool(passed and step == selected)],
            } for step in schedule
        ),
        metadata={
            "method": "integrated_gradients", "quadrature": "gauss_legendre",
            "integration_dtype": "float32", "input_shape": [1, 3, 224, 224],
            "step_schedule": list(STEPS), "step_chunk_size": 8,
            "atol": 1e-4, "rtol": 1e-2, "scope": "full_model",
            "target": {"kind": "class_logit", "class_index": class_index},
            "includes_residual_paths": True,
        },
    )


def _fixture(monkeypatch, tmp_path, failed_variant="clean"):
    config = load_stage_c_config()
    registry = load_stage_c_registry(config, verify_live_assets=False)
    plan = build_execution_plan(config, registry, make_run_spec(
        sample_ids=[SAMPLE], method_ids=[METHOD], value_config_source="none",
        value_candidates=[], fidelity_pairing_context=None,
        audit_repeatability_runs=2, audit_repeatability_scope="clean_map_only",
        run_registered_fidelity=False, run_sanity=False,
        run_stability=failed_variant != "clean",
    ))
    estimate = estimate_stage_c_run(config, plan).to_dict()
    unit = plan.units[0]
    model = torch.nn.Identity()
    clean = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    clean_sha256 = tensor_sha256(clean)
    class_index = 17
    target = SimpleNamespace(
        class_index=class_index, clean_input_sha256=clean_sha256,
        to_dict=lambda: {
            "target_kind": "raw_logit", "class_index": class_index,
            "class_index_source": "single_clean_input_argmax",
            "clean_raw_logit": 1.0, "class_count": 1000,
            "freeze_class_index_for_all_interventions": True,
            "official_truth_index": None, "clean_top1_correct": None,
            "exclude_misclassified": False, "clean_input_sha256": clean_sha256,
            "clean_input_shape": [1, 3, 224, 224],
            "runtime_model_binding": {"kind": "test", "sha256": "b" * 64,
                                      "is_canonical_model_artifact_hash": False},
        },
    )
    backend = ProductionStageCBackend(
        sample_loader=lambda *_args: pytest.fail("cached sample only"),
        model_loader=lambda **_kwargs: model, config=config, registry=registry,
        plan=plan, code_state={"commit": "d" * 40, "dirty": False},
    )
    backend._models["cpu"] = model
    backend._samples[(SAMPLE, "cpu")] = {
        "canonical_rgb": clean, "normalized_input": clean,
        "registry_row": registry.by_sample_id()[SAMPLE],
    }
    backend._class_targets[(SAMPLE, "cpu")] = target
    failure_result = _ig(converged=False, class_index=class_index)
    error = StageCClassLogitIGConvergenceError("full IG failed", failure_result)

    def composed(received, *, device, call_counter):
        variants = [(item.input_variant, item.repeat_index) for item in received.work_items
                    if item.kind == "map"]
        assert failed_variant in [item[0] for item in variants]
        for variant, repeat in variants:
            result = failure_result if variant == failed_variant else _ig(
                converged=True, class_index=class_index
            )
            overhead = 2 if variant in {"clean", "repeat_clean"} else 4
            chunks = 1984 if not result.converged else 64
            def invoke(*_args, _result=result, **_kwargs):
                backend._active_call_counter.record_forward(chunks + overhead)
                if not _result.converged:
                    raise error
                return SimpleNamespace(provenance={"ig_result": _result.to_dict()})
            monkeypatch.setattr(backend, "_run_method", invoke)
            backend._run_method_observed(
                received, input_variant=variant, repeat_index=repeat, model=model,
                clean=clean, extraction=object(), target=target,
                target_identity={"comparison_line": "class_logit"},
            )
            if variant == failed_variant:
                pytest.fail("failed observed map unexpectedly returned")

    monkeypatch.setattr(backend, "_run_composed", composed)
    root = tmp_path / failed_variant
    root.mkdir()
    (root / "execution_plan.json").write_bytes(_canonical(plan.to_dict()))
    result = run_execution_unit(unit, output_root=root, backend=backend, plan=plan,
                                preflight=estimate, device="cpu", resume=False)
    assert "persistence_error" not in result.failure, result.failure
    attempt = Path(result.attempt_path)
    expected = {
        "configuration": {"execution_plan_sha256": plan.canonical_digest,
                          "run_spec_sha256": canonical_sha256(plan.run_spec.to_dict())},
        "resources": {"preflight": {"plan_digest": plan.canonical_digest,
                                        "canonical_digest": estimate["canonical_digest"]}},
    }
    return SimpleNamespace(attempt=attempt, unit=unit, expected=expected,
                           caps=config.budget.to_dict(), code=backend.code_state)


def _validate(fixture):
    return validate_execution_unit(
        fixture.attempt, expected_unit_identity=fixture.unit.key.to_dict(),
        expected_provenance_bindings=(
            fixture.expected if hasattr(fixture, "expected")
            else fixture.expected_bindings
        ),
        observed_git_state=(
            fixture.code if hasattr(fixture, "code")
            else fixture.observed_git_state
        ), caps=fixture.caps,
        require_runner_telemetry=True,
    )


def _resign(fixture, mutate):
    attempt = fixture.attempt
    lifecycle_path = attempt / "L0/execution_lifecycle.json"
    manifest_path = attempt / "manifest.json"
    provenance_path = attempt / "L0/provenance.json"
    status_path = attempt / "status.json"
    lifecycle = json.loads(lifecycle_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    provenance = json.loads(provenance_path.read_bytes())
    status = json.loads(status_path.read_bytes())
    mutate(lifecycle, manifest, provenance, status)
    lifecycle_bytes = _canonical(lifecycle)
    lifecycle_path.write_bytes(lifecycle_bytes)
    inventory = manifest["artifacts"]
    record = next(x for x in inventory if x["path"] == "L0/execution_lifecycle.json")
    record.update(bytes=len(lifecycle_bytes), sha256=hashlib.sha256(lifecycle_bytes).hexdigest())
    declared = [dict(x) for x in inventory if x["path"] != "L0/provenance.json"]
    provenance["bindings"]["artifact_inventory"] = declared
    actual = provenance["bindings"]["resources"]["actual"]
    actual["files"] = [{"path": x["path"], "bytes": x["bytes"]} for x in declared]
    actual["artifact_bytes"] = sum(x["bytes"] for x in declared)
    provenance_bytes = _canonical(provenance)
    provenance_path.write_bytes(provenance_bytes)
    p_record = next(x for x in inventory if x["path"] == "L0/provenance.json")
    p_record.update(bytes=len(provenance_bytes), sha256=hashlib.sha256(provenance_bytes).hexdigest())
    manifest["provenance_sha256"] = p_record["sha256"]
    manifest["runner_telemetry_sha256"] = canonical_sha256(status["runner_telemetry"])
    manifest_bytes = _canonical(manifest)
    manifest_path.write_bytes(manifest_bytes)
    status["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    status_path.write_bytes(_canonical(status))


@pytest.mark.parametrize("variant", ["clean", "translate_right_down_4", "repeat_clean"])
def test_full_model_failure_profiles_accept_variant_specific_counts(monkeypatch, tmp_path, variant):
    fixture = _fixture(monkeypatch, tmp_path, variant)
    assert _validate(fixture).valid


@pytest.mark.parametrize(
    "mutate,error_token",
    [
        (lambda record: record["ig_result"]["metadata"]["target"].update(
            class_index=18), "metadata"),
        (lambda record: record.update(ig_result=None), "mismatch"),
        (lambda record: record.update(unexpected=True), "unknown fields"),
    ],
    ids=["provenance-bound-target", "deleted-evidence", "closed-schema"],
)
def test_full_model_failure_semantic_resign_is_rejected(
    monkeypatch, tmp_path, mutate, error_token,
):
    fixture = _fixture(monkeypatch, tmp_path)
    _resign(fixture, lambda life, *_: mutate(life["map_execution_records"][-1]))
    result = _validate(fixture)
    assert not result.valid
    assert any(error_token in error for error in result.errors)


def test_nns_joint_method_and_terminal_type_resign_cannot_hide_ig_evidence(monkeypatch, tmp_path):
    # Reuse the established production NNS fixture without modifying its test module.
    from test_stage_c_runner import _run_value_path_convergence_failure
    fixture = _run_value_path_convergence_failure(monkeypatch, tmp_path)
    def mutate(life, manifest, provenance, status):
        fake = "example.OtherError"
        life["map_execution_records"][-1]["method_id"] = "other_method"
        life["map_execution_records"][-1]["exception"]["type"] = fake
        life["lifecycle"]["integrity"]["body_exception_type"] = fake
        manifest["failure"]["details"]["type"] = fake
        provenance["bindings"]["failure"]["type"] = fake
        status["runner_telemetry"]["lifecycle"]["integrity"]["body_exception_type"] = fake
    _resign(fixture, mutate)
    result = _validate(fixture)
    assert not result.valid
    assert any("mismatch" in error or "method_id" in error for error in result.errors)
