"""Synthetic-only contract tests for the Stage D image-group producer."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vitnns.stage_c_runner import (
    ProductionStageCBackend, StageCBackendError, StageCPartialEvidence,
    StageCScientificInvalidity,
)
from vitnns.stage_d_plan import METHOD_IDS, StageDExecutionPlan
from vitnns.stage_d_producer import (
    ProductionStageDBackend, STAGE_D_FIDELITY_SEED_NAMESPACE,
    StageDBackendPayload, StageDInfrastructureFailure,
    StageDPartialEvidence, StageDScientificFailure,
    run_stage_d_execution_unit,
    run_stage_d_image_group, stage_d_fidelity_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plan() -> StageDExecutionPlan:
    # Reading the metadata-only typed plan is permitted by the D4 guard.
    from vitnns.stage_d_plan import build_stage_d_plan

    def load(relative: str) -> dict:
        return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))

    return build_stage_d_plan(
        load("config/stage_d.json"),
        load("config/stage_d_run_spec.json"),
        load("manifests/stage_d_fidelity_v1/registry.json"),
    )


class FakeBackend:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.active = False
        self.release_count = 0

    @contextmanager
    def image_scope(self, sample_id: str, *, device: str):
        if self.active:
            raise StageCBackendError(
                "ProductionStageCBackend is already owned by another image scope"
            )
        self.active = True
        self.events.append(("enter", sample_id))
        try:
            yield self
        finally:
            self.release_sample(sample_id, device)
            self.active = False

    def release_sample(self, sample_id: str, device: str) -> None:
        self.events.append(("release", sample_id))
        self.release_count += 1


def _result(unit, state: str, **extra):
    return SimpleNamespace(
        state=state, unit_digest=unit.canonical_digest,
        pair_binding_digest=unit.pair_binding_digest, **extra,
    )


def test_group_order_pair_work_coordinates_audit_and_control_only_result(plan):
    group = next(item for item in plan.groups if item.audit)
    backend = FakeBackend()
    calls = []
    callbacks = []

    def run_unit(unit, *, backend, device):
        assert backend.active and device == "cpu"
        calls.append(unit)
        return _result(unit, "TERMINAL_SUCCESS", auc=123.0, map=[999.0])

    result = run_stage_d_image_group(
        group, backend=backend, run_unit=run_unit,
        on_unit_terminal=lambda unit, value: callbacks.append(
            (unit.method_id, value.state, backend.active)
        ),
    )

    assert [unit.method_id for unit in calls] == list(METHOD_IDS)
    assert {unit.pair_binding_digest for unit in calls} == {
        group.pair_binding.canonical_digest
    }
    assert callbacks == [
        (METHOD_IDS[0], "TERMINAL_SUCCESS", True),
        (METHOD_IDS[1], "TERMINAL_SUCCESS", True),
    ]
    assert backend.events == [("enter", group.sample_id), ("release", group.sample_id)]
    assert result.state == "TERMINAL_SUCCESS" and result.cache_released
    public = result.to_dict()
    assert set(public) == {
        "group_digest", "pair_binding_digest", "state",
        "unit_results", "cache_released",
    }
    assert all("auc" not in item and "map" not in item for item in public["unit_results"])

    adapted = ProductionStageDBackend(sample_loader=lambda *_: None).adapt_unit(
        group, group.units[0]
    )
    maps = [item for item in adapted.work_items if item.kind == "map"]
    curves = [
        item for item in adapted.work_items
        if item.kind == "fidelity_curve_point"
    ]
    assert [(item.input_variant, item.repeat_index) for item in maps] == [
        ("clean", 0), ("repeat_clean", 1),
    ]
    assert len(curves) == 2 * 5 * 21
    assert [(item.curve_direction, item.replicate_index, item.fraction_index) for item in curves] == [
        (direction, repeat, fraction)
        for direction in ("insertion", "deletion")
        for repeat in range(5) for fraction in range(21)
    ]
    assert {item.fidelity_scale for item in curves} == {"fine_4x4_cell"}


def test_scientific_failure_and_retry_pending_continue_paired_unit(plan):
    group = plan.groups[0]
    for first_state, expected_state in (
        ("TERMINAL_SCIENTIFIC_FAILURE", "TERMINAL_SCIENTIFIC_FAILURE"),
        ("RETRY_PENDING", "RETRY_PENDING"),
    ):
        backend = FakeBackend()
        calls = []

        def run_unit(unit, *, backend, device):
            calls.append(unit.method_id)
            state = first_state if len(calls) == 1 else "TERMINAL_SUCCESS"
            return _result(unit, state)

        result = run_stage_d_image_group(group, backend=backend, run_unit=run_unit)
        assert calls == list(METHOD_IDS)
        assert result.state == expected_state
        assert backend.release_count == 1


def test_blocked_stops_pair_and_exception_always_releases(plan):
    group = plan.groups[0]
    backend = FakeBackend()
    calls = []

    result = run_stage_d_image_group(
        group, backend=backend,
        run_unit=lambda unit, **_: (
            calls.append(unit.method_id) or _result(unit, "BLOCKED")
        ),
    )
    assert calls == [METHOD_IDS[0]]
    assert result.state == "BLOCKED"
    assert backend.release_count == 1

    failing = FakeBackend()
    with pytest.raises(RuntimeError, match="synthetic crash"):
        run_stage_d_image_group(
            group, backend=failing,
            run_unit=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("synthetic crash")
            ),
        )
    assert failing.release_count == 1


def test_backend_scope_rejects_concurrency_and_long_sequence_releases_caches(plan):
    backend = ProductionStageCBackend(sample_loader=lambda *_: None)
    sample = plan.groups[0].sample_id
    other = plan.groups[1].sample_id
    entered = threading.Event()
    leave = threading.Event()

    def owner():
        with backend.image_scope(sample, device="cpu"):
            entered.set()
            leave.wait(timeout=5)

    thread = threading.Thread(target=owner)
    thread.start()
    assert entered.wait(timeout=5)
    with pytest.raises(StageCBackendError, match="already owned"):
        with backend.image_scope(other, device="cpu"):
            pass
    leave.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    backend._models["cpu"] = object()
    backend._model_load_snapshots["cpu"] = object()
    for index in range(25):
        sample_id = f"synthetic-{index}"
        with backend.image_scope(sample_id, device="cpu"):
            pair = (sample_id, "cpu")
            backend._samples[pair] = {"synthetic": True}
            backend._perturbations[pair] = {}
            backend._extractions[pair] = object()
            backend._class_targets[pair] = object()
            backend._value_targets[(*pair, 0, 9)] = object()
        assert not backend._samples
        assert not backend._perturbations
        assert not backend._extractions
        assert not backend._class_targets
        assert not backend._value_targets
    assert set(backend._models) == {"cpu"}
    assert set(backend._model_load_snapshots) == {"cpu"}


def test_stage_d_seed_golden_namespace_and_fine_retention(plan):
    first_sample = plan.groups[0].sample_id
    assert STAGE_D_FIDELITY_SEED_NAMESPACE == "stage-d-fidelity|20260821|"
    assert stage_d_fidelity_seed(first_sample, 0, 9, 0) == 6554433287525610283
    backend = ProductionStageDBackend(sample_loader=lambda *_: None)
    assert backend.fidelity_seed_namespace == STAGE_D_FIDELITY_SEED_NAMESPACE
    assert issubclass(ProductionStageDBackend, ProductionStageCBackend)
    with pytest.raises(Exception, match="frozen Stage D seed namespace"):
        ProductionStageDBackend(
            sample_loader=lambda *_: None,
            fidelity_seed_namespace="stage-c-fidelity|20260821|",
        )
    payload = StageDBackendPayload(
        l0={}, l1={"clean": np.zeros((56, 56), dtype=np.float32)}
    )
    assert payload.l1["clean"].shape == (56, 56)
    with pytest.raises(Exception, match="56,56"):
        StageDBackendPayload(
            l0={}, l1={"clean": np.zeros((14, 14), dtype=np.float32)}
        )
    with pytest.raises(Exception, match="L3"):
        StageDBackendPayload(
            l0={}, l1={"clean": np.zeros((56, 56), dtype=np.float32)},
            l3={"forbidden.png": b"synthetic"},
        )


def _execution_inputs(plan, tmp_path, *, audit=False):
    from vitnns.stage_d_runtime import _issue_synthetic_d4_test_context
    from vitnns.stage_d_plan import estimate_stage_d_run, preflight_stage_d

    group = (
        next(item for item in plan.groups if item.audit)
        if audit else next(item for item in plan.groups if not item.audit)
    )
    unit = group.units[0]
    root = tmp_path / "runtime-root"
    attempt = root / "units" / unit.canonical_digest / "attempts" / "attempt-000001"
    owner = "synthetic-owner-token"
    import hashlib
    claim = {
        "owner_token": owner,
        "owner_token_digest": hashlib.sha256(owner.encode()).hexdigest(),
        "group_owner_token_digest": "a" * 64,
        "unit_digest": unit.canonical_digest,
        "group_digest": group.canonical_digest,
        "attempt_id": "attempt-000001", "attempt_number": 1,
        "attempt_path": str(attempt),
    }
    config = json.loads((REPO_ROOT / "config/stage_d.json").read_text())
    estimate = estimate_stage_d_run(config, plan)
    preflight = preflight_stage_d(estimate)
    context, _gate = _issue_synthetic_d4_test_context(
        plan, estimate, preflight, shard_count=500
    )
    return group, unit, root, context, claim


class EvidenceBackend:
    def __init__(self, outcome="success"):
        self.outcome = outcome
        self.calls = 0

    def run(self, unit, *, device, call_counter):
        self.calls += 1
        if self.outcome == "scientific":
            call_counter.record_forward(2)
            call_counter.record_backward(1)
            raise StageDScientificFailure(
                "degenerate_target_delta", phase="target_response",
                l0_scalars={"status": "NA"},
            )
        if self.outcome == "infrastructure":
            call_counter.record_forward(1)
            raise StageDInfrastructureFailure(
                "SCHEDULER_PREEMPTION", {"elapsed_seconds": 0.25}
            )
        call_counter.record_forward(3)
        call_counter.record_backward(2)
        method = unit.method_id
        records = []
        for direction in ("insertion", "deletion"):
            for repeat in range(5):
                records.append({
                    "status": "OK", "method_id": method,
                    "comparison_line": "value_path",
                    "target_ref": unit.target_id, "scale": "fine_4x4_cell",
                    "direction": direction, "replicate": repeat,
                    "sample_id": unit.sample_id, "block": unit.block_index,
                    "head": unit.head_index,
                    "curve_fractions": [index / 20.0 for index in range(21)],
                    "seed_namespace": STAGE_D_FIDELITY_SEED_NAMESPACE,
                    "seed": stage_d_fidelity_seed(
                        unit.sample_id, unit.block_index, unit.head_index, repeat
                    ),
                    "responses": [float(index) / 20.0 for index in range(21)],
                })
        l1 = {"clean": np.zeros((56, 56), dtype=np.float32)}
        if unit.audit:
            l1["repeat_clean_1"] = np.ones((56, 56), dtype=np.float32)
        return StageDBackendPayload(
            l0={"status": "OK", "fidelity_curves": records},
            l1=l1, l2={"raw": np.ones((2, 3), dtype=np.float32)} if unit.audit else {},
            resources={"forward_calls": 999},
        )


def test_default_execution_unit_success_publishes_real_evidence(plan, tmp_path):
    group, unit, root, context, claim = _execution_inputs(plan, tmp_path)
    backend = EvidenceBackend()
    result = run_stage_d_execution_unit(
        unit, group=group, plan=plan, output_root=root, backend=backend,
        context=context, claim=claim, device="cpu",
    )
    assert result["state"] == "succeeded"
    assert result["status_sha256"] == __import__("hashlib").sha256(
        (Path(claim["attempt_path"]) / "status.json").read_bytes()
    ).hexdigest()
    assert result["resources"]["forward_calls"] == 3
    assert result["resources"]["backward_calls"] == 2
    assert result["resources"]["l0_l1_payload_bytes"] > 0
    assert "fidelity_curves" not in result and "l0" not in result


def test_default_execution_unit_audit_moves_repeat_map_to_l2(plan, tmp_path):
    group, unit, root, context, claim = _execution_inputs(plan, tmp_path, audit=True)
    result = run_stage_d_execution_unit(
        unit, group=group, plan=plan, output_root=root,
        backend=EvidenceBackend(), context=context, claim=claim,
    )
    assert result["state"] == "succeeded"
    with np.load(Path(claim["attempt_path"]) / "L2/audit_evidence.npz") as data:
        assert set(data.files) == {"raw", "repeat_clean_metric_ready_map"}
        np.testing.assert_array_equal(
            data["repeat_clean_metric_ready_map"],
            np.ones((56, 56), dtype=np.float32),
        )


def test_default_execution_unit_scientific_and_infrastructure_failures(plan, tmp_path):
    group, unit, root, context, claim = _execution_inputs(plan, tmp_path / "scientific")
    scientific = run_stage_d_execution_unit(
        unit, group=group, plan=plan, output_root=root,
        backend=EvidenceBackend("scientific"), context=context, claim=claim,
    )
    assert scientific["state"] == "scientific_failed"
    assert scientific["failure_class"] == "SCIENTIFIC_INVALIDITY"
    assert scientific["result_manifest_published"] is True
    assert scientific["resources"]["elapsed_seconds"] >= 0.0
    assert scientific["resources"]["forward_calls"] == 2
    assert scientific["resources"]["backward_calls"] == 1

    group, unit, root, context, claim = _execution_inputs(plan, tmp_path / "infra")
    infrastructure = run_stage_d_execution_unit(
        unit, group=group, plan=plan, output_root=root,
        backend=EvidenceBackend("infrastructure"), context=context, claim=claim,
    )
    assert infrastructure["state"] == "infrastructure_failed"
    assert infrastructure["failure_code"] == "SCHEDULER_PREEMPTION"
    assert infrastructure["result_manifest_published"] is False
    assert infrastructure["resources"]["elapsed_seconds"] >= 0.0
    assert infrastructure["resources"]["forward_calls"] == 1
    with pytest.raises(ValueError, match="allowlist"):
        StageDInfrastructureFailure("SCIENTIFIC_INVALIDITY", {})


def test_curve_record_adapter_rejects_wrong_order_and_seed(plan, tmp_path):
    group, unit, root, context, claim = _execution_inputs(plan, tmp_path / "order")
    backend = EvidenceBackend()
    original_run = backend.run

    def wrong_order(unit, **kwargs):
        payload = original_run(unit, **kwargs)
        curves = list(payload.l0["fidelity_curves"]); curves.reverse()
        return StageDBackendPayload(l0={**payload.l0, "fidelity_curves": curves}, l1=payload.l1)

    backend.run = wrong_order
    with pytest.raises(Exception, match="frozen Stage D coordinate"):
        run_stage_d_execution_unit(
            unit, group=group, plan=plan, output_root=root, backend=backend,
            context=context, claim=claim,
        )

    group, unit, root, context, claim = _execution_inputs(plan, tmp_path / "seed")
    backend = EvidenceBackend()
    original_run = backend.run

    def wrong_seed(unit, **kwargs):
        payload = original_run(unit, **kwargs)
        curves = [dict(item) for item in payload.l0["fidelity_curves"]]
        curves[0]["seed"] += 1
        return StageDBackendPayload(l0={**payload.l0, "fidelity_curves": curves}, l1=payload.l1)

    backend.run = wrong_seed
    with pytest.raises(Exception, match="frozen Stage D coordinate"):
        run_stage_d_execution_unit(
            unit, group=group, plan=plan, output_root=root, backend=backend,
            context=context, claim=claim,
        )


def test_default_execution_unit_validates_metadata_before_backend_or_attempt(plan, tmp_path):
    group, unit, root, context, claim = _execution_inputs(plan, tmp_path)
    backend = EvidenceBackend()
    claim["unit_digest"] = "0" * 64
    with pytest.raises(Exception, match="claim coordinate"):
        run_stage_d_execution_unit(
            unit, group=group, plan=plan, output_root=root, backend=backend,
            context=context, claim=claim,
        )
    assert backend.calls == 0
    assert not Path(claim["attempt_path"]).exists()


def test_default_execution_unit_does_not_reclassify_plain_exception(plan, tmp_path):
    group, unit, root, context, claim = _execution_inputs(plan, tmp_path)

    class CrashingBackend(EvidenceBackend):
        def run(self, unit, *, device, call_counter):
            self.calls += 1
            raise RuntimeError("scientific-looking secret 91.25")

    backend = CrashingBackend()
    with pytest.raises(RuntimeError, match="scientific-looking secret"):
        run_stage_d_execution_unit(
            unit, group=group, plan=plan, output_root=root, backend=backend,
            context=context, claim=claim,
        )
    assert backend.calls == 1
    status = json.loads(
        (Path(claim["attempt_path"]) / "status.json").read_text()
    )
    assert status["status"] == "running"
    assert "91.25" not in json.dumps(status)


def test_stage_c_scientific_translation_is_closed_and_pair_can_continue(plan, tmp_path):
    group, unit, root, context, claim = _execution_inputs(plan, tmp_path / "typed")

    class TypedScientific(EvidenceBackend):
        def run(self, unit, *, device, call_counter):
            self.calls += 1
            raise StageCScientificInvalidity(
                "degenerate_direction", phase="target_resolution"
            )

    result = run_stage_d_execution_unit(
        unit, group=group, plan=plan, output_root=root,
        backend=TypedScientific(), context=context, claim=claim,
    )
    assert result["state"] == "scientific_failed"
    assert result["failure_code"] == "degenerate_direction"

    group, unit, root, context, claim = _execution_inputs(plan, tmp_path / "audit")

    for message in (
        "audit clean-map repeatability failed",
        "actual peak CUDA reserved bytes exceed the unit cap",
        "canonical preprocessing contract drifted",
        "CUDA out of memory",
        "unknown backend failure",
    ):
        group, unit, root, context, claim = _execution_inputs(
            plan, tmp_path / message.replace(" ", "-")
        )

        class NotScientific(EvidenceBackend):
            def run(self, unit, *, device, call_counter):
                raise StageCBackendError(message)

        with pytest.raises(StageCBackendError, match=message):
            run_stage_d_execution_unit(
                unit, group=group, plan=plan, output_root=root,
                backend=NotScientific(), context=context, claim=claim,
            )


def test_scientific_partial_evidence_is_written_before_failure(plan, tmp_path):
    group, unit, root, context, claim = _execution_inputs(plan, tmp_path)

    class PartialFailure(EvidenceBackend):
        def run(self, unit, *, device, call_counter):
            call_counter.record_forward(1)
            raise StageDScientificFailure(
                "ig_conservation_failed", phase="ig_conservation",
                l0_scalars={"status": "NA"},
                partial_evidence=StageDPartialEvidence(
                    l0_scalars={"partial": True},
                    metric_ready_map=np.ones((56, 56), dtype=np.float32),
                ),
            )

    result = run_stage_d_execution_unit(
        unit, group=group, plan=plan, output_root=root,
        backend=PartialFailure(), context=context, claim=claim,
    )
    attempt = Path(claim["attempt_path"])
    assert result["state"] == "scientific_failed"
    assert result["published_l0_l1_payload_count"] == 2
    assert (attempt / "L0/result.json").is_file()
    assert (attempt / "L1/metric_ready_map.npy").is_file()
    assert not (attempt / "L1/fidelity_curves.npz").exists()
    status = json.loads((attempt / "status.json").read_text())
    assert status["failure"]["produced"] == [
        "L0/result.json", "L1/metric_ready_map.npy"
    ]


def test_production_adapter_translates_typed_stage_c_partial_evidence(
    plan, tmp_path, monkeypatch,
):
    group, unit, root, context, claim = _execution_inputs(plan, tmp_path)
    partial = StageCPartialEvidence(
        l0_scalars={"status": "NA"},
        metric_ready_map=np.ones((56, 56), dtype=np.float32),
    )

    def fail_scientifically(_backend, _unit, *, device, call_counter):
        del device, call_counter
        raise StageCScientificInvalidity(
            "audit_repeatability_failed", phase="audit_repeatability",
            l0_scalars={"status": "NA"}, partial_evidence=partial,
        )

    monkeypatch.setattr(ProductionStageCBackend, "run", fail_scientifically)
    backend = ProductionStageDBackend(sample_loader=lambda *_args: None)
    backend._active_stage_d_group = group
    result = run_stage_d_execution_unit(
        unit, group=group, plan=plan, output_root=root, backend=backend,
        context=context, claim=claim,
    )
    assert result["state"] == "scientific_failed"
    assert result["failure_code"] == "audit_repeatability_failed"
    assert result["published_l0_l1_payload_count"] == 2
    attempt = Path(claim["attempt_path"])
    assert (attempt / "L0/result.json").is_file()
    assert (attempt / "L1/metric_ready_map.npy").is_file()


def test_direct_fake_formal_and_synthetic_contexts_fail_before_backend_writer(plan, tmp_path):
    group, unit, root, issued, claim = _execution_inputs(plan, tmp_path / "base")
    base = {name: getattr(issued, name) for name in (
        "formal", "authorization_scope", "execution_authorization",
        "authorization_slot", "shard_count", "shard_mapping",
        "plan_digest", "code_binding", "dependency_binding",
        "model_binding", "source_bindings",
    )}
    fake_formal = {
        **base, "formal": True,
        "execution_authorization": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE",
    }
    backend = EvidenceBackend()
    with pytest.raises(Exception, match="opaque context"):
        run_stage_d_execution_unit(
            unit, group=group, plan=plan, output_root=root, backend=backend,
            context=fake_formal, claim=claim,
        )
    assert backend.calls == 0 and not Path(claim["attempt_path"]).exists()
    group, unit, root, issued, claim = _execution_inputs(
        plan, tmp_path / "forged-formal-object"
    )
    forged_formal = SimpleNamespace(
        **{
            **base, "formal": True,
            "execution_authorization": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE",
        },
        _d7_validated_context=True,
    )
    backend = EvidenceBackend()
    with pytest.raises(Exception, match="not sealed"):
        run_stage_d_execution_unit(
            unit, group=group, plan=plan, output_root=root, backend=backend,
            context=forged_formal, claim=claim,
        )
    assert backend.calls == 0 and not Path(claim["attempt_path"]).exists()
    group, unit, root, issued, claim = _execution_inputs(plan, tmp_path / "synthetic")
    fake_synthetic = SimpleNamespace(**base)
    backend = EvidenceBackend()
    with pytest.raises(Exception, match="not issued"):
        run_stage_d_execution_unit(
            unit, group=group, plan=plan, output_root=root, backend=backend,
            context=fake_synthetic, claim=claim,
        )
    assert backend.calls == 0 and not Path(claim["attempt_path"]).exists()
