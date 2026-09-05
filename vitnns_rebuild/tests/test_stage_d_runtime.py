"""Synthetic-only tests for the Stage D D4 runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vitnns.stage_d_plan import (
    build_stage_d_plan, canonical_json_bytes, canonical_sha256,
    estimate_stage_d_run, preflight_stage_d,
)
from vitnns.stage_d_producer import (
    StageDInfrastructureFailure, StageDScientificFailure,
)
from vitnns.stage_d_artifacts import StageDUnitPayload
from vitnns.stage_d_runtime import (
    NONRETRYABLE_FAILURE_CLASSES, RETRYABLE_INFRASTRUCTURE_FAILURE_CODES,
    RUNTIME_CHECKPOINT_FILENAME, StageDConcurrentClaimError,
    StageDExecutionAuthorizationRequired, StageDRecoveryAuthorizationRequired,
    StageDRuntimeBlockedError, StageDRuntimeIntegrityError,
    _expected_lifecycle_state, _issue_synthetic_d4_test_context,
    initialize_stage_d_access_root,
    initialize_stage_d_output_root, recover_stage_d_claim, run_stage_d_shard,
    validate_stage_d_access_root,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ZERO = {
    "elapsed_seconds": 0.0, "forward_calls": 0, "backward_calls": 0,
    "l0_l1_payload_bytes": 0, "l2_payload_bytes": 0,
    "control_and_provenance_bytes": 0, "cuda_peak_reserved_bytes": 0,
}


@pytest.fixture(scope="module")
def controls():
    config = json.loads((REPO_ROOT / "config/stage_d.json").read_text())
    run_spec = json.loads((REPO_ROOT / "config/stage_d_run_spec.json").read_text())
    registry = json.loads(
        (REPO_ROOT / "manifests/stage_d_fidelity_v1/registry.json").read_text()
    )
    plan = build_stage_d_plan(config, run_spec, registry)
    estimate = estimate_stage_d_run(config, plan)
    return plan, estimate, preflight_stage_d(estimate)


class FakeBackend:
    def __init__(self, behavior):
        self.behavior = behavior
        self.active = None
        self.order: list[int] = []

    def image_scope(self, sample_id, *, device):
        backend = self

        class Scope:
            def __enter__(self):
                assert backend.active is None
                backend.active = (sample_id, device)

            def __exit__(self, *_args):
                backend.active = None

        return Scope()

    def run_stage_d_unit(self, group, unit, *, device, call_counter):
        del group, device, call_counter
        self.order.append(unit.method_index)
        return self.behavior(unit)


def payload() -> StageDUnitPayload:
    return StageDUnitPayload(
        l0_scalars={
            "endpoint_parity": 0.0,
        },
        metric_ready_map=np.zeros((56, 56), dtype=np.float32),
        fidelity_curves=np.zeros((2, 5, 21), dtype=np.float64),
    )


def permit(controls, count=500):
    return _issue_synthetic_d4_test_context(*controls, shard_count=count)


def init_root(controls, tmp_path: Path, count=500):
    plan, estimate, preflight = controls
    context, gate = permit(controls, count)
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "synthetic-runtime"
    initialize_stage_d_output_root(
        root, plan, estimate, preflight,
        execution_context=context, execution_gate=gate,
    )
    return root, context, gate


def run(controls, root, context, gate, factory, **kwargs):
    plan, estimate, preflight = controls
    return run_stage_d_shard(
        plan, output_root=root, estimate=estimate, preflight=preflight,
        shard_id=0, execution_context=context, execution_gate=gate,
        backend_factory=factory, **kwargs,
    )


def checkpoint(root: Path) -> dict[str, Any]:
    return json.loads((root / RUNTIME_CHECKPOINT_FILENAME).read_text())


def access_contracts(plan):
    registry = json.loads(
        (REPO_ROOT / "manifests/stage_d_fidelity_v1/registry.json").read_text()
    )
    by_id = {row["sample_id"]: row for row in registry["rows"]}
    return [by_id[group.sample_id] for group in plan.groups]


def synthetic_observation(group, contract):
    digest = __import__("hashlib").sha256(group.sample_id.encode()).hexdigest()
    return {
        "source_kind": "SYNTHETIC_TEST_FIXTURE",
        "source_payload_sha256": digest, "image_sha256": digest,
        **{key: contract[key] for key in (
            "size_bytes", "width", "height", "mode", "image_format",
            "official_ilsvrc2012_class_id", "official_wnid",
            "imagenet1k_index", "label_source",
        )},
    }


def formal_observation(group, contract):
    return {
        **synthetic_observation(group, contract),
        "source_payload_sha256": group.image_sha256,
        "image_sha256": group.image_sha256,
    }


def formal_permit(controls, count=500):
    import vitnns.stage_d_runtime as runtime

    plan, estimate, preflight = controls
    synthetic, _ = permit(controls, count)
    values = {name: getattr(synthetic, name) for name in (
        "authorization_slot", "shard_count", "shard_mapping",
        "plan_digest", "budget_digest", "preflight_digest",
        "code_binding", "dependency_binding", "model_binding",
        "source_bindings",
    )}
    values.update({
        "formal": True, "authorization_scope": "D7_FORMAL_TEST_FIXTURE",
        "execution_authorization": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE",
        "access_attestation": {
            "reserve_image_bytes_accessed": False,
            "model_loaded_for_identity_verification": True,
            "model_used_for_inference": False, "cuda_accessed": False,
            "outcome_accessed": False,
        },
    })
    values["canonical_digest"] = canonical_sha256({
        **values, "shard_mapping": [
            values["shard_mapping"][index] for index in range(500)
        ],
    })
    formal = type("FormalContext", (), values)()

    class Gate:
        def validate_execution_context(self, context, **_kwargs):
            assert context is formal
            return formal

    binding = runtime._normalize_execution_context(
        formal, plan=plan, estimate=estimate, preflight=preflight
    )
    return formal, Gate(), binding


def resign(root: Path, value: dict[str, Any]) -> None:
    value["canonical_digest"] = canonical_sha256({
        key: child for key, child in value.items() if key != "canonical_digest"
    })
    (root / RUNTIME_CHECKPOINT_FILENAME).write_bytes(canonical_json_bytes(value) + b"\n")


def test_gate_and_controls_precede_root_and_backend(controls, tmp_path: Path) -> None:
    plan, estimate, preflight = controls
    root = tmp_path / "never-created"
    calls: list[str] = []
    with pytest.raises(StageDExecutionAuthorizationRequired):
        initialize_stage_d_output_root(root, plan, estimate, preflight)
    with pytest.raises(StageDExecutionAuthorizationRequired):
        run_stage_d_shard(
            plan, output_root=root, estimate=estimate, preflight=preflight,
            shard_id=0, backend_factory=lambda **_kwargs: calls.append("backend"),
        )
    assert calls == [] and not root.exists()


def test_continuation_authorization_is_bound_to_exact_requested_shard(
    controls, tmp_path: Path,
) -> None:
    plan, estimate, preflight = controls
    calls: list[str] = []
    continuation = {"authorized_next_shard": 1}
    with pytest.raises(
        StageDExecutionAuthorizationRequired, match="different shard",
    ):
        run_stage_d_shard(
            plan, output_root=tmp_path / "never-opened",
            estimate=estimate, preflight=preflight, shard_id=2,
            execution_context=continuation,
            backend_factory=lambda **_kwargs: calls.append("backend"),
        )
    assert calls == []


def test_d8_initializer_durable_resume_and_no_unit_execution(controls, tmp_path: Path) -> None:
    plan, estimate, preflight = controls
    context, gate = permit(controls, 8)
    root = tmp_path / "synthetic-d8-init"
    contracts = access_contracts(plan)
    calls: list[int] = []
    stopped = False

    def validator(group, contract):
        lifecycle = json.loads((root / "access/lifecycle.json").read_text())
        events = sorted((root / "access/journal").iterdir())
        assert lifecycle["state"] == "ACCESS_STARTED"
        assert lifecycle["journal_entry_count"] == len(events)
        assert json.loads(events[-1].read_text())["event"] in {
            "ASSET_ACCESS_INTENT", "ASSET_ACCESS_RESUMED",
        }
        calls.append(group.selection_order)
        return synthetic_observation(group, contract)

    def interrupt(point, selection_order):
        nonlocal stopped
        if point == "after_asset_validated_event" and selection_order == 9 and not stopped:
            stopped = True
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        initialize_stage_d_access_root(
            root, plan, estimate, preflight, execution_context=context,
            execution_gate=gate, asset_contracts=contracts,
            asset_validator=validator, fault_injector=interrupt,
        )
    partial = validate_stage_d_access_root(
        root, plan, estimate, preflight, execution_context=context,
        execution_gate=gate, asset_contracts=contracts, require_runnable=False,
        repair_lag=True,
    )
    assert partial["state"] == "ACCESS_STARTED"
    assert partial["validated_asset_count"] == 10
    result = initialize_stage_d_access_root(
        root, plan, estimate, preflight, execution_context=context,
        execution_gate=gate, asset_contracts=contracts, asset_validator=validator,
    )
    assert result["state"] == "RUNNABLE"
    assert calls == list(range(500))
    assert not list((root / "units").glob("*/attempts/*"))


def test_d8_initializer_failure_is_durable_and_blocked(controls, tmp_path: Path) -> None:
    plan, estimate, preflight = controls
    context, gate = permit(controls, 8)
    root = tmp_path / "synthetic-d8-blocked"
    contracts = access_contracts(plan)

    def validator(group, contract):
        if group.selection_order == 3:
            raise ValueError("synthetic digest mismatch")
        return synthetic_observation(group, contract)

    with pytest.raises(StageDRuntimeBlockedError):
        initialize_stage_d_access_root(
            root, plan, estimate, preflight, execution_context=context,
            execution_gate=gate, asset_contracts=contracts, asset_validator=validator,
        )
    blocked = validate_stage_d_access_root(
        root, plan, estimate, preflight, execution_context=context,
        execution_gate=gate, asset_contracts=contracts, require_runnable=False,
    )
    assert blocked["state"] == "BLOCKED"
    assert blocked["validated_asset_count"] == 3
    assert blocked["failure_code"] == "LIVE_ASSET_VALIDATION_FAILED"
    assert not list((root / "units").glob("*/attempts/*"))
    with pytest.raises(StageDRuntimeBlockedError):
        initialize_stage_d_access_root(
            root, plan, estimate, preflight, execution_context=context,
            execution_gate=gate, asset_contracts=contracts, asset_validator=validator,
        )


def test_formal_run_shard_requires_runnable_access_gate(controls, tmp_path: Path) -> None:
    import vitnns.stage_d_runtime as runtime

    plan, estimate, preflight = controls
    synthetic, _ = permit(controls, 8)
    values = {name: getattr(synthetic, name) for name in (
        "authorization_slot", "shard_count", "shard_mapping",
        "plan_digest", "budget_digest", "preflight_digest",
        "code_binding", "dependency_binding", "model_binding",
        "source_bindings",
    )}
    values.update({
        "formal": True, "authorization_scope": "D7_FORMAL_TEST_FIXTURE",
        "execution_authorization": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE",
        "access_attestation": {
            "reserve_image_bytes_accessed": False,
            "model_loaded_for_identity_verification": True,
            "model_used_for_inference": False, "cuda_accessed": False,
            "outcome_accessed": False,
        },
    })
    values["canonical_digest"] = canonical_sha256({
        **values, "shard_mapping": [values["shard_mapping"][i] for i in range(500)],
    })
    formal = type("FormalContext", (), values)()

    class Gate:
        def validate_execution_context(self, context, **_kwargs):
            assert context is formal
            return formal

    binding = runtime._normalize_execution_context(
        formal, plan=plan, estimate=estimate, preflight=preflight
    )
    root = runtime._initialize_validated_output_root(
        tmp_path / "formal-before-access", plan, estimate, preflight, binding
    )
    backend_calls: list[str] = []
    with pytest.raises(StageDRuntimeIntegrityError, match="access controls"):
        run_stage_d_shard(
            plan, output_root=root, estimate=estimate, preflight=preflight,
            shard_id=0, execution_context=formal, execution_gate=Gate(),
            backend_factory=lambda **_kwargs: backend_calls.append("called"),
            asset_contracts=access_contracts(plan),
        )
    assert backend_calls == []


def test_formal_access_gate_survives_prior_shard_checkpoint_progress(
    controls, tmp_path: Path,
) -> None:
    plan, estimate, preflight = controls
    formal, gate, _binding = formal_permit(controls, 500)
    root = tmp_path / "formal-continuation"
    contracts = access_contracts(plan)
    initialize_stage_d_access_root(
        root, plan, estimate, preflight, execution_context=formal,
        execution_gate=gate, asset_contracts=contracts,
        asset_validator=formal_observation,
    )
    backend0 = FakeBackend(lambda _unit: payload())
    first = run_stage_d_shard(
        plan, output_root=root, estimate=estimate, preflight=preflight,
        shard_id=0, execution_context=formal, execution_gate=gate,
        backend_factory=lambda **_kwargs: backend0,
        asset_contracts=contracts,
    )
    assert first.state == "completed"
    assert first.counts["terminal_units"] == 2

    validated = validate_stage_d_access_root(
        root, plan, estimate, preflight, execution_context=formal,
        execution_gate=gate, asset_contracts=contracts,
    )
    assert validated["state"] == "RUNNABLE"

    backend1 = FakeBackend(lambda _unit: payload())
    second = run_stage_d_shard(
        plan, output_root=root, estimate=estimate, preflight=preflight,
        shard_id=1, execution_context=formal, execution_gate=gate,
        backend_factory=lambda **_kwargs: backend1,
        asset_contracts=contracts,
    )
    assert second.state == "completed"
    assert second.counts["terminal_units"] == 2
    assert backend0.order == backend1.order == [0, 1]


def test_formal_continuation_rejects_access_start_checkpoint_tamper_before_backend(
    controls, tmp_path: Path,
) -> None:
    plan, estimate, preflight = controls
    formal, gate, _binding = formal_permit(controls, 500)
    root = tmp_path / "formal-continuation-tamper"
    contracts = access_contracts(plan)
    initialize_stage_d_access_root(
        root, plan, estimate, preflight, execution_context=formal,
        execution_gate=gate, asset_contracts=contracts,
        asset_validator=formal_observation,
    )
    backend0 = FakeBackend(lambda _unit: payload())
    run_stage_d_shard(
        plan, output_root=root, estimate=estimate, preflight=preflight,
        shard_id=0, execution_context=formal, execution_gate=gate,
        backend_factory=lambda **_kwargs: backend0, asset_contracts=contracts,
    )
    lifecycle_path = root / "access/lifecycle.json"
    lifecycle = json.loads(lifecycle_path.read_text())
    lifecycle["runtime_checkpoint_at_access_start"] = "f" * 64
    lifecycle["canonical_digest"] = canonical_sha256({
        key: value for key, value in lifecycle.items()
        if key != "canonical_digest"
    })
    lifecycle_path.write_bytes(canonical_json_bytes(lifecycle) + b"\n")
    backend_calls: list[str] = []
    with pytest.raises(StageDRuntimeIntegrityError, match="access lifecycle binding"):
        run_stage_d_shard(
            plan, output_root=root, estimate=estimate, preflight=preflight,
            shard_id=1, execution_context=formal, execution_gate=gate,
            backend_factory=lambda **_kwargs: backend_calls.append("called"),
            asset_contracts=contracts,
        )
    assert backend_calls == []
    assert not list(
        (root / "units" / plan.groups[1].units[0].canonical_digest / "attempts")
        .glob("attempt-*")
    )


def test_access_journal_tamper_is_rejected(controls, tmp_path: Path) -> None:
    plan, estimate, preflight = controls
    context, gate = permit(controls, 8)
    root = tmp_path / "synthetic-d8-tamper"
    contracts = access_contracts(plan)
    initialize_stage_d_access_root(
        root, plan, estimate, preflight, execution_context=context,
        execution_gate=gate, asset_contracts=contracts,
        asset_validator=synthetic_observation,
    )
    path = root / "access/journal/000010.json"
    event = json.loads(path.read_text())
    event["asset"]["sample_id"] = "tampered"
    event["canonical_digest"] = canonical_sha256({
        key: value for key, value in event.items() if key != "canonical_digest"
    })
    path.write_bytes(canonical_json_bytes(event) + b"\n")
    with pytest.raises(StageDRuntimeIntegrityError):
        validate_stage_d_access_root(
            root, plan, estimate, preflight, execution_context=context,
            execution_gate=gate, asset_contracts=contracts,
        )


def test_runtime_seals_formal_gate_result_before_backend_and_unit(controls) -> None:
    import vitnns.stage_d_runtime as runtime

    plan, estimate, preflight = controls
    synthetic, _synthetic_gate = permit(controls)
    values = {
        name: getattr(synthetic, name)
        for name in (
            "authorization_slot", "shard_count", "shard_mapping",
            "plan_digest", "budget_digest", "preflight_digest",
            "code_binding", "dependency_binding", "model_binding",
            "source_bindings", "access_attestation",
        )
    }
    values.update({
        "formal": True, "authorization_scope": "D7_FORMAL_TEST_FIXTURE",
        "execution_authorization": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE",
        "access_attestation": {
            "reserve_image_bytes_accessed": False,
            "model_loaded_for_identity_verification": True,
            "model_used_for_inference": False,
            "cuda_accessed": False, "outcome_accessed": False,
        },
    })
    values["canonical_digest"] = canonical_sha256({
        **values, "shard_mapping": [
            values["shard_mapping"][index] for index in range(500)
        ],
    })
    gate_result = type("ForgableGateResult", (), values)()

    # Once a gate result passes complete normalization, runtime replaces it
    # with a private capability before either execution callback receives it.
    binding = runtime._normalize_execution_context(
        gate_result, plan=plan, estimate=estimate, preflight=preflight
    )
    sealed = runtime._seal_execution_context(gate_result, binding)
    assert sealed is not gate_result
    assert runtime._is_runtime_execution_capability(sealed) is True
    assert runtime._is_runtime_execution_capability(gate_result) is False

    forged = type("ForgedFormalCapability", (), {
        **values, "_d7_validated_context": True,
        "_issuer": runtime._RUNTIME_CAPABILITY_ISSUER,
    })()
    assert runtime._is_runtime_execution_capability(forged) is False

    observed: list[object] = []
    runtime._call_backend_factory(
        lambda *, context, **_kwargs: observed.append(context),
        context=sealed, device="cpu",
    )
    (lambda *, context: observed.append(context))(context=sealed)
    assert observed == [sealed, sealed]


def test_real_attempt_source_validation_half_pair_retry_and_no_leak(
    controls, tmp_path: Path,
) -> None:
    root, context, gate = init_root(controls, tmp_path)
    initial = checkpoint(root)
    assert len(initial["groups"]) == 500
    assert len(initial["units"]) == len(initial["failure_rows"]) == 1000
    assert [row["shard_id"] for row in initial["groups"][:4]] == [0, 1, 2, 3]

    backend = FakeBackend(lambda unit: (
        (_ for _ in ()).throw(StageDScientificFailure(
            "degenerate_direction", phase="attribution",
            l0_scalars={"finite": True},
        )) if unit.method_index == 0 else
        (_ for _ in ()).throw(StageDInfrastructureFailure(
            "SCHEDULER_PREEMPTION", ZERO
        ))
    ))
    first = run(controls, root, context, gate, lambda **_kwargs: backend)
    assert first.state == "retry_pending"
    assert backend.order == [0, 1]
    current = checkpoint(root)
    rows = current["units"][:2]
    assert [row["state"] for row in rows] == ["scientific_failed", "retry_pending"]
    assert all(row["attempts"][0]["attempt_inventory_digest"] for row in rows)

    retry = {
        "schema_version": 1, "document_type": "stage_d_runtime_retry_authorization",
        "status": "COORDINATOR_AUTHORIZED_RETRY",
        "unit_digest": rows[1]["unit_digest"],
        "prior_attempt_digest": rows[1]["attempts"][0]["attempt_inventory_digest"],
        "failure_code": "SCHEDULER_PREEMPTION", "next_attempt_number": 2,
        "execution_context_digest": current["execution_context"]["runtime_binding_digest"],
    }
    retry["canonical_digest"] = canonical_sha256(retry)
    backend.behavior = lambda _unit: payload()

    def lying_runner(*args, **kwargs):
        from vitnns.stage_d_producer import run_stage_d_execution_unit

        result = run_stage_d_execution_unit(*args, **kwargs)
        # Runtime must derive these from the attempt tree, not trust producer
        # claims.  The underlying real artifact is a valid terminal attempt.
        result["result_manifest_published"] = False
        result["published_l0_l1_payload_count"] = 0
        result["resources"] = ZERO
        return result

    second = run(
        controls, root, context, gate, lambda **_kwargs: backend,
        retry_authorizations=[retry], unit_runner=lying_runner,
    )
    assert second.state == "completed"
    assert backend.order == [0, 1, 1]
    assert second.counts["scientific_failed_units"] == 1
    assert second.counts["succeeded_units"] == 1
    assert second.counts["skipped_terminal_units"] == 1
    rendered = json.dumps(second.to_dict()).lower()
    assert all(word not in rendered for word in ("auc", "delta", "prediction", "logit"))

    factories: list[str] = []
    assert run(
        controls, root, context, gate,
        lambda **_kwargs: factories.append("constructed"),
    ).state == "completed"
    assert factories == []


def test_resume_rejects_tampered_extra_and_missing_attempts_before_backend(
    controls, tmp_path: Path,
) -> None:
    root, context, gate = init_root(controls, tmp_path)
    backend = FakeBackend(lambda _unit: payload())
    run(controls, root, context, gate, lambda **_kwargs: backend)
    calls: list[str] = []
    first_attempt = next((root / "units").glob("*/attempts/attempt-000001"))
    (first_attempt / "status.json").write_bytes(b"{}\n")
    with pytest.raises(StageDRuntimeIntegrityError, match="source-recomputing|source evidence"):
        run(controls, root, context, gate, lambda **_kwargs: calls.append("backend"))
    assert calls == []

    # A self-resigned checkpoint cannot legitimize an unrecorded attempt.
    root2, context2, gate2 = init_root(controls, tmp_path / "extra")
    unit_root = next((root2 / "units").iterdir()) / "attempts"
    (unit_root / "attempt-000001").mkdir()
    with pytest.raises(StageDRuntimeIntegrityError, match="unrecorded attempt"):
        run(controls, root2, context2, gate2, lambda **_kwargs: calls.append("backend"))

    root3, context3, gate3 = init_root(controls, tmp_path / "missing")
    value = checkpoint(root3)
    value["units"][0]["state"] = "succeeded"
    value["units"][0]["terminal"] = True
    value["groups"][0]["state"] = "claimed"
    value["groups"][0]["owner_token_digest"] = "a" * 64
    value["shards"][0]["state"] = "leased"
    value["shards"][0]["owner_token_digest"] = "b" * 64
    resign(root3, value)
    with pytest.raises(StageDRuntimeIntegrityError):
        run(controls, root3, context3, gate3, lambda **_kwargs: calls.append("backend"))
    assert calls == []


def test_crash_claim_is_persisted_rereadable_and_not_stolen(controls, tmp_path: Path) -> None:
    root, context, gate = init_root(controls, tmp_path)
    backend = FakeBackend(lambda _unit: (_ for _ in ()).throw(RuntimeError("boom 1.23")))
    with pytest.raises(RuntimeError, match="boom"):
        run(controls, root, context, gate, lambda **_kwargs: backend)
    current = checkpoint(root)
    assert current["shards"][0]["state"] == "leased"
    assert current["groups"][0]["state"] == "claimed"
    assert current["units"][0]["state"] == "claimed"
    with pytest.raises(StageDConcurrentClaimError):
        run(controls, root, context, gate, lambda **_kwargs: backend)
    with pytest.raises(StageDRecoveryAuthorizationRequired):
        recover_stage_d_claim(
            controls[0], output_root=root, estimate=controls[1], preflight=controls[2],
            unit_digest=current["units"][0]["unit_digest"],
            owner_token_digest=current["units"][0]["owner_token_digest"],
            failure_code="WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH",
            execution_context=context, execution_gate=gate,
        )


def test_checkpoint_projection_and_runner_publication_claims_fail_closed(
    controls, tmp_path: Path,
) -> None:
    root, context, gate = init_root(controls, tmp_path)
    value = checkpoint(root)
    value["lifecycle"]["terminal_units"] = 1
    resign(root, value)
    with pytest.raises(StageDRuntimeIntegrityError, match="lifecycle counters"):
        run(controls, root, context, gate, lambda **_kwargs: FakeBackend(lambda _: payload()))


def test_lifecycle_complete_requires_all_shard_leases_released() -> None:
    shards = [
        {
            "shard_id": 0,
            "state": "completed",
            "terminal": True,
            "owner_token_digest": None,
        },
        {
            "shard_id": 1,
            "state": "leased",
            "terminal": False,
            "owner_token_digest": "a" * 64,
        },
    ]
    assert _expected_lifecycle_state("running", 2, shards) == "running"
    shards[1] = {**shards[1], "state": "completed", "terminal": True}
    assert _expected_lifecycle_state("running", 3, shards) == "complete"


def test_coordinator_recovery_of_missing_attempt_is_durable(controls, tmp_path: Path) -> None:
    root, context, gate = init_root(controls, tmp_path)
    backend = FakeBackend(lambda _unit: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        run(controls, root, context, gate, lambda **_kwargs: backend)
    before = checkpoint(root)
    row = before["units"][0]

    class Recovery:
        def validate_recovery_authorization(
            self, authorization, *, unit, attempt, owner_token_digest,
            failure_code, execution_context_digest, checkpoint_digest,
        ):
            assert authorization == {"coordinator": True}
            assert unit.canonical_digest == row["unit_digest"]
            assert attempt["attempt_id"] == "attempt-000001"
            assert owner_token_digest == row["owner_token_digest"]
            assert failure_code == "WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH"
            assert checkpoint_digest == before["canonical_digest"]
            return type("Permit", (), {
                "canonical_digest": canonical_sha256({
                    "unit": unit.canonical_digest,
                    "context": execution_context_digest,
                })
            })()

    digest = recover_stage_d_claim(
        controls[0], output_root=root, estimate=controls[1], preflight=controls[2],
        unit_digest=row["unit_digest"],
        owner_token_digest=row["owner_token_digest"],
        failure_code="WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH",
        recovery_authorization={"coordinator": True}, recovery_gate=Recovery(),
        execution_context=context, execution_gate=gate,
    )
    after = checkpoint(root)
    assert after["canonical_digest"] == digest
    assert after["units"][0]["state"] == "retry_pending"
    assert (
        after["units"][0]["attempts"][0]["failure_code"]
        == "WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH"
    )
    assert (
        Path(after["units"][0]["attempts"][0]["attempt_path"])
        / "status.json"
    ).is_file()
    with pytest.raises(StageDRuntimeBlockedError):
        recover_stage_d_claim(
            controls[0], output_root=root, estimate=controls[1],
            preflight=controls[2], unit_digest=row["unit_digest"],
            owner_token_digest=row["owner_token_digest"],
            failure_code="WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH",
            recovery_authorization={"coordinator": True}, recovery_gate=Recovery(),
            execution_context=context, execution_gate=gate,
        )


def test_frozen_failure_taxonomy() -> None:
    assert len(RETRYABLE_INFRASTRUCTURE_FAILURE_CODES) == 7
    assert len(NONRETRYABLE_FAILURE_CLASSES) == 7
    assert set(RETRYABLE_INFRASTRUCTURE_FAILURE_CODES).isdisjoint(
        NONRETRYABLE_FAILURE_CLASSES
    )


def _force_small_l0_cap(controls, maximum: int):
    from dataclasses import replace

    estimate = controls[1]
    caps = dict(estimate.launch_safety_caps)
    caps["l0_l1_bytes_per_execution_unit_max"] = maximum
    # This is a runtime-only injected estimate used to exercise online cap
    # disposition after the real attempt has already been validated.
    altered = replace(estimate, launch_safety_caps=caps)
    object.__setattr__(altered, "canonical_digest", canonical_sha256(altered._payload()))
    preflight = preflight_stage_d(altered)
    return controls[0], altered, preflight


def test_cap_breach_preserves_attempt_fact_but_blocks_unit_and_root(
    controls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, context, gate = init_root(controls, tmp_path)
    import vitnns.stage_d_runtime as runtime

    real_caps = runtime._resource_cap_errors
    monkeypatch.setattr(
        runtime, "_resource_cap_errors",
        lambda result, estimate: ["l0_l1_bytes_per_execution_unit_max"],
    )
    backend = FakeBackend(lambda _unit: payload())
    summary = run(controls, root, context, gate, lambda **_kwargs: backend)
    assert summary.state == "blocked"
    value = checkpoint(root)
    assert value["units"][0]["state"] == "blocked"
    assert value["units"][0]["failure_class"] == "LAUNCH_SAFETY_CAP_BREACH"
    assert value["units"][0]["attempts"][0]["state"] == "succeeded"
    monkeypatch.setattr(runtime, "_resource_cap_errors", real_caps)
    calls: list[str] = []
    with pytest.raises(StageDRuntimeBlockedError):
        run(controls, root, context, gate, lambda **_kwargs: calls.append("backend"))
    assert calls == []


@pytest.mark.parametrize("fault", ["after_manifest", "after_status", "before_checkpoint"])
def test_recovery_reconciles_publication_faults(
    controls, tmp_path: Path, fault: str,
) -> None:
    root, context, gate = init_root(controls, tmp_path / fault)
    backend = FakeBackend(lambda _unit: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        run(controls, root, context, gate, lambda **_kwargs: backend)
    before = checkpoint(root)
    row = before["units"][0]

    class Recovery:
        def validate_recovery_authorization(self, *_args, **kwargs):
            return type("Permit", (), {
                "canonical_digest": canonical_sha256({
                    "unit": kwargs["unit"].canonical_digest, "fault": fault,
                })
            })()

    arguments = dict(
        plan=controls[0], output_root=root, estimate=controls[1], preflight=controls[2],
        unit_digest=row["unit_digest"], owner_token_digest=row["owner_token_digest"],
        failure_code="WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH",
        recovery_authorization={"coordinator": True}, recovery_gate=Recovery(),
        execution_context=context, execution_gate=gate,
    )
    with pytest.raises(RuntimeError, match="recovery crash"):
        recover_stage_d_claim(**arguments, fault_injector=fault)
    assert checkpoint(root)["units"][0]["state"] == "claimed"
    recover_stage_d_claim(**arguments)
    assert checkpoint(root)["units"][0]["state"] == "retry_pending"


def test_second_infrastructure_attempt_exhaustion_is_reReadable(
    controls, tmp_path: Path,
) -> None:
    root, context, gate = init_root(controls, tmp_path)
    backend = FakeBackend(lambda _unit: (_ for _ in ()).throw(
        StageDInfrastructureFailure("SCHEDULER_PREEMPTION", ZERO)
    ))
    run(controls, root, context, gate, lambda **_kwargs: backend)
    first = checkpoint(root)
    row = first["units"][0]
    retry = {
        "schema_version": 1, "document_type": "stage_d_runtime_retry_authorization",
        "status": "COORDINATOR_AUTHORIZED_RETRY",
        "unit_digest": row["unit_digest"],
        "prior_attempt_digest": row["attempts"][0]["attempt_inventory_digest"],
        "failure_code": row["failure_code"], "next_attempt_number": 2,
        "execution_context_digest": first["execution_context"]["runtime_binding_digest"],
    }
    retry["canonical_digest"] = canonical_sha256(retry)
    summary = run(
        controls, root, context, gate, lambda **_kwargs: backend,
        retry_authorizations=[retry],
    )
    assert summary.state == "blocked"
    final = checkpoint(root)
    assert final["units"][0]["state"] == "blocked"
    assert [item["state"] for item in final["units"][0]["attempts"]] == [
        "infrastructure_failed", "infrastructure_failed",
    ]
    calls: list[str] = []
    with pytest.raises(StageDRuntimeBlockedError):
        run(controls, root, context, gate, lambda **_kwargs: calls.append("backend"))
    assert calls == []
