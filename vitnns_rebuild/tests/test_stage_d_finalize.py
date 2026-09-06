"""Synthetic-only tests for Stage D D5 finalization and validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import vitnns.stage_d_finalize as stage_d_finalize_module
import vitnns.stage_d_artifacts as stage_d_artifacts_module
import vitnns.stage_d_runtime as stage_d_runtime_module
from vitnns.stage_d_artifacts import StageDUnitPayload
from vitnns.stage_d_finalize import (
    StageDFinalizationBlockedError,
    StageDFinalizationIntegrityError,
    finalize_stage_d_root,
    validate_stage_d_root,
)
from vitnns.stage_d_plan import (
    build_stage_d_plan,
    canonical_json_bytes,
    canonical_sha256,
    estimate_stage_d_run,
    preflight_stage_d,
)
from vitnns.stage_d_producer import (
    StageDInfrastructureFailure,
    StageDScientificFailure,
)
from vitnns.stage_d_runtime import (
    _issue_synthetic_d4_test_context,
    initialize_stage_d_output_root,
    run_stage_d_shard,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_FILES = {
    "evidence_report.json",
    "reconciliation.json",
    "source_inventory.json",
    "payload_manifest.json",
    "status.json",
}
RESULT_KEYS = {
    "package_complete", "workflow_status", "scientific_validity",
    "analysis_authorized", "plan_digest", "final_status_sha256",
    "root_inventory", "actual_cost_audit",
    "evidence_report", "reconciliation", "source_inventory",
    "payload_manifest", "whole_root_ordered_inventory",
}
ZERO_RESOURCES = {
    "elapsed_seconds": 0.0, "forward_calls": 0, "backward_calls": 0,
    "l0_l1_payload_bytes": 0, "l2_payload_bytes": 0,
    "control_and_provenance_bytes": 0, "cuda_peak_reserved_bytes": 0,
}
STAGING_PREFIX = ".d5-final-staging-"


@pytest.fixture(scope="module", autouse=True)
def _fast_synthetic_durability():
    """Keep content/hash/schema coverage while avoiding 1,000 tmpfs fsyncs."""

    patch = pytest.MonkeyPatch()
    patch.setattr(stage_d_runtime_module.os, "fsync", lambda _fd: None)
    patch.setattr(stage_d_artifacts_module.os, "fsync", lambda _fd: None)
    patch.setattr(stage_d_finalize_module.os, "fsync", lambda _fd: None)
    yield
    patch.undo()


@pytest.fixture(scope="module")
def controls():
    config = json.loads((REPO_ROOT / "config/stage_d.json").read_text())
    run_spec = json.loads(
        (REPO_ROOT / "config/stage_d_run_spec.json").read_text()
    )
    registry = json.loads(
        (REPO_ROOT / "manifests/stage_d_fidelity_v1/registry.json").read_text()
    )
    plan = build_stage_d_plan(config, run_spec, registry)
    estimate = estimate_stage_d_run(config, plan)
    return plan, estimate, preflight_stage_d(estimate)


class FakeBackend:
    """Deterministic in-memory producer; it performs no image/model/GPU work."""

    def __init__(self, behavior: Callable[[Any], StageDUnitPayload]) -> None:
        self.behavior = behavior
        self.active: tuple[str, str] | None = None
        self.calls = 0

    def image_scope(self, sample_id: str, *, device: str):
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
        self.calls += 1
        return self.behavior(unit)


def _payload(unit: Any, *, endpoint_parity: float = 0.0) -> StageDUnitPayload:
    metric_map = np.zeros((56, 56), dtype=np.float32)
    return StageDUnitPayload(
        l0_scalars={"endpoint_parity": endpoint_parity},
        metric_ready_map=metric_map,
        fidelity_curves=np.zeros((2, 5, 21), dtype=np.float64),
        audit_evidence=(
            {
                "raw_attribution": np.ones((1,), dtype=np.float32),
                "repeat_clean_metric_ready_map": metric_map.copy(),
            }
            if unit.audit else None
        ),
    )


def _build_full_root(
    controls: tuple[Any, Any, Any], root: Path,
    behavior: Callable[[Any], StageDUnitPayload],
) -> Path:
    plan, estimate, preflight = controls
    context, gate = _issue_synthetic_d4_test_context(
        plan, estimate, preflight, shard_count=1,
    )
    initialize_stage_d_output_root(
        root, plan, estimate, preflight,
        execution_context=context, execution_gate=gate,
    )
    backend = FakeBackend(behavior)
    summary = _run_linear_fixture_shard(
        controls, root, context=context, gate=gate, backend=backend,
    )
    assert summary.state == "completed"
    assert summary.counts["terminal_groups"] == 500
    assert summary.counts["terminal_units"] == 1000
    assert backend.calls == 1000
    return root


def _run_linear_fixture_shard(
    controls: tuple[Any, Any, Any], root: Path, *, context: Any, gate: Any,
    backend: FakeBackend, retry_authorizations: Any = (),
):
    """Run the real D4 producer while deferring quadratic checkpoint I/O.

    Attempts, artifacts, manifests, status, and provenance still use production
    writers and source validation.  Only repeated whole-checkpoint derivation is
    deferred until the shard releases, followed by one unpatched full D4 read.
    """

    plan, estimate, preflight = controls
    original_validate_checkpoint = stage_d_runtime_module._validate_checkpoint
    original_read_checkpoint = stage_d_runtime_module._read_valid_checkpoint_locked
    original_replace_checkpoint = stage_d_runtime_module._replace_checkpoint
    initial = json.loads(
        (root / stage_d_runtime_module.RUNTIME_CHECKPOINT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    context_binding = initial["execution_context"]
    current = original_read_checkpoint(
        root, plan, estimate, preflight, context_binding,
    )
    published = False

    def fast_read_checkpoint(
        checked_root, checked_plan, checked_estimate, checked_preflight,
        checked_context_binding, *, transitioning_unit_digest=None,
    ):
        if checked_root != root:
            return original_read_checkpoint(
                checked_root, checked_plan, checked_estimate,
                checked_preflight, checked_context_binding,
                transitioning_unit_digest=transitioning_unit_digest,
            )
        assert checked_context_binding == context_binding
        return current

    def validate_checkpoint_once_then_trust(document, **kwargs):
        if kwargs.get("root") != root:
            return original_validate_checkpoint(document, **kwargs)
        return document

    def fast_replace_checkpoint(
        checked_root, previous, changed, **kwargs,
    ):
        nonlocal current, published
        if checked_root != root:
            return original_replace_checkpoint(
                checked_root, previous, changed, **kwargs,
            )
        assert previous is current and changed is current
        prior_digest = previous["canonical_digest"]
        next_document = changed
        next_document["generation"] = previous["generation"] + 1
        next_document["previous_checkpoint_digest"] = prior_digest
        next_document["canonical_digest"] = canonical_sha256({
            "generation": next_document["generation"],
            "previous_checkpoint_digest": prior_digest,
        })
        current = next_document
        released = (
            all(row["state"] != "leased" for row in next_document["shards"])
            and all(row["state"] != "claimed" for row in next_document["groups"])
            and all(row["state"] != "claimed" for row in next_document["units"])
        )
        if released:
            next_document["failure_rows"] = stage_d_runtime_module._failure_rows(
                next_document["units"]
            )
            next_document["resources"] = stage_d_runtime_module._sum_resources(
                next_document["units"]
            )
            lifecycle = next_document["lifecycle"]
            next_document["lifecycle"] = stage_d_runtime_module._lifecycle_projection(
                lifecycle["state"], next_document["shards"],
                next_document["groups"], next_document["units"],
                lifecycle["blocked_reasons"],
            )
            next_document.pop("canonical_digest", None)
            next_document["canonical_digest"] = canonical_sha256(next_document)
            stage_d_runtime_module._atomic_replace_checkpoint(
                checked_root / stage_d_runtime_module.RUNTIME_CHECKPOINT_FILENAME,
                next_document,
            )
            published = True
        return current

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            stage_d_runtime_module, "_validate_checkpoint",
            validate_checkpoint_once_then_trust,
        )
        patch.setattr(
            stage_d_runtime_module, "_read_valid_checkpoint_locked",
            fast_read_checkpoint,
        )
        patch.setattr(
            stage_d_runtime_module, "_replace_checkpoint",
            fast_replace_checkpoint,
        )
        summary = run_stage_d_shard(
            plan, output_root=root, estimate=estimate, preflight=preflight,
            shard_id=0, execution_context=context, execution_gate=gate,
            backend_factory=lambda **_kwargs: backend,
            retry_authorizations=retry_authorizations,
        )
    assert published is True
    persisted = original_read_checkpoint(
        root, plan, estimate, preflight, context_binding,
    )
    assert persisted["canonical_digest"] == summary.checkpoint_digest
    return summary


@pytest.fixture(scope="module")
def valid_root(controls, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_full_root(
        controls, tmp_path_factory.mktemp("d5-valid") / "execution-root",
        _payload,
    )


@pytest.fixture(scope="module")
def authorized_retry_root(
    controls, tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str, dict[str, Any]]:
    plan, estimate, preflight = controls
    root = tmp_path_factory.mktemp("d5-retry") / "execution-root"
    context, gate = _issue_synthetic_d4_test_context(
        plan, estimate, preflight, shard_count=1,
    )
    initialize_stage_d_output_root(
        root, plan, estimate, preflight,
        execution_context=context, execution_gate=gate,
    )
    retried_digest = plan.units[0].canonical_digest
    interrupted = False

    def first_behavior(unit):
        nonlocal interrupted
        if unit.canonical_digest == retried_digest and not interrupted:
            interrupted = True
            raise StageDInfrastructureFailure(
                "SCHEDULER_PREEMPTION", ZERO_RESOURCES,
            )
        return _payload(unit)

    backend = FakeBackend(first_behavior)
    first = _run_linear_fixture_shard(
        controls, root, context=context, gate=gate, backend=backend,
    )
    assert first.state == "retry_pending"
    checkpoint = _read_checkpoint(root)
    retry_row = next(
        row for row in checkpoint["units"]
        if row["unit_digest"] == retried_digest
    )
    assert retry_row["state"] == "retry_pending"
    assert len(retry_row["attempts"]) == 1
    retry = {
        "schema_version": 1,
        "document_type": "stage_d_runtime_retry_authorization",
        "status": "COORDINATOR_AUTHORIZED_RETRY",
        "unit_digest": retried_digest,
        "prior_attempt_digest": (
            retry_row["attempts"][0]["attempt_inventory_digest"]
        ),
        "failure_code": retry_row["failure_code"],
        "next_attempt_number": 2,
        "execution_context_digest": (
            checkpoint["execution_context"]["runtime_binding_digest"]
        ),
    }
    retry["canonical_digest"] = canonical_sha256(retry)

    backend.behavior = _payload
    second = _run_linear_fixture_shard(
        controls, root, context=context, gate=gate, backend=backend,
        retry_authorizations=[retry],
    )
    assert second.state == "completed"
    assert second.counts["terminal_groups"] == 500
    assert second.counts["terminal_units"] == 1000
    assert backend.calls == 1001
    final_checkpoint = _read_checkpoint(root)
    retry_row = next(
        row for row in final_checkpoint["units"]
        if row["unit_digest"] == retried_digest
    )
    assert retry_row["state"] == "succeeded"
    assert len(retry_row["attempts"]) == 2
    assert retry_row["retry_authorization_digest"] == retry["canonical_digest"]
    assert retry_row["attempts"][1]["retry_authorization_digest"] == (
        retry["canonical_digest"]
    )
    return root, retried_digest, retry


def _copy_root(source: Path, tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copytree(
        source, destination, copy_function=shutil.copy2,
        ignore=shutil.ignore_patterns("final"),
    )

    # D4 deliberately binds absolute attempt paths.  Rebase only those
    # transport fields and their enclosing canonical digests after copying;
    # all source attempt bytes remain independent, single-link copies.
    checkpoint_path = destination / "runtime_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    for row in checkpoint["units"]:
        for attempt in row["attempts"]:
            attempt["attempt_path"] = str(
                destination / "units" / row["unit_digest"]
                / "attempts" / attempt["attempt_id"]
            )
            attempt["canonical_digest"] = canonical_sha256({
                key: value for key, value in attempt.items()
                if key != "canonical_digest"
            })
    checkpoint["canonical_digest"] = canonical_sha256({
        key: value for key, value in checkpoint.items()
        if key != "canonical_digest"
    })
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint) + b"\n")
    assert all(
        path.stat().st_nlink == 1
        for path in destination.rglob("*") if path.is_file()
    )
    return destination


def _read_checkpoint(root: Path) -> dict[str, Any]:
    return json.loads(
        (root / "runtime_checkpoint.json").read_text(encoding="utf-8")
    )


def _write_checkpoint(root: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["canonical_digest"] = canonical_sha256({
        key: value for key, value in checkpoint.items()
        if key != "canonical_digest"
    })
    (root / "runtime_checkpoint.json").write_bytes(
        canonical_json_bytes(checkpoint) + b"\n"
    )


def _minimal_checkpoint_root(
    controls: tuple[Any, Any, Any], root: Path, *, formal: bool = False,
) -> Path:
    plan, estimate, preflight = controls
    context, gate = _issue_synthetic_d4_test_context(
        plan, estimate, preflight, shard_count=1,
    )
    initialize_stage_d_output_root(
        root, plan, estimate, preflight,
        execution_context=context, execution_gate=gate,
    )
    if formal:
        checkpoint = _read_checkpoint(root)
        checkpoint["execution_context"]["formal"] = True
        _write_checkpoint(root, checkpoint)
    return root


def _formal_checkpoint_only_root(
    controls: tuple[Any, Any, Any], root: Path,
) -> Path:
    """Build only the controls needed to prove pre-traversal refusal."""

    plan, estimate, preflight = controls
    context, _gate = _issue_synthetic_d4_test_context(
        plan, estimate, preflight, shard_count=1,
    )
    context_binding = stage_d_runtime_module._normalize_execution_context(
        context, plan=plan, estimate=estimate, preflight=preflight,
    )
    checkpoint = stage_d_runtime_module._initial_checkpoint(
        plan, estimate, preflight, context_binding,
    )
    checkpoint["execution_context"]["formal"] = True
    checkpoint["canonical_digest"] = canonical_sha256({
        key: value for key, value in checkpoint.items()
        if key != "canonical_digest"
    })
    root.mkdir()
    for name in (
        stage_d_runtime_module.LIFECYCLE_LOCK_FILENAME,
        stage_d_runtime_module.CHECKPOINT_LOCK_FILENAME,
        stage_d_runtime_module.ARTIFACT_ACTIVITY_LOCK_FILENAME,
    ):
        (root / name).write_bytes(b"")
    (root / stage_d_runtime_module.RUNTIME_CHECKPOINT_FILENAME).write_bytes(
        canonical_json_bytes(checkpoint) + b"\n"
    )
    return root


def _authorized_formal_checkpoint_only_root(
    controls: tuple[Any, Any, Any], root: Path,
) -> tuple[Path, dict[str, str]]:
    plan, estimate, preflight = controls
    context, _gate = _issue_synthetic_d4_test_context(
        plan, estimate, preflight, shard_count=8,
    )
    synthetic = stage_d_runtime_module._normalize_execution_context(
        context, plan=plan, estimate=estimate, preflight=preflight,
    )
    slot = {
        "status": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE",
        "authorization_canonical_digest": "3479f9457aa924fe189591e836e38bb1a65cce8543e3a879477025782659afc9",
        "authority_ref": "refs/archive/stage-d-fidelity-v1-d8-0-rebind-r1-authority",
        "authority_commit": "SELF",
    }
    producer = {
        "commit": "4c4dd4af010afdc72e26f10cb8e6519e2448cf40",
        "tree": "a27651064facd955f78ef3f517a0934f3e1ace58",
        "clean": True,
    }
    dependencies = {
        "path": "requirements.lock.txt",
        "lock_sha256": "81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984",
    }
    model = {
        "path": "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth",
        "state_dict_sha256": "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d",
        "lfs_oid_sha256": "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d",
        "canonical_tensor_digest": "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd",
        "parameter_count": 86567656,
        "key_count": 152,
        "identity": "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718",
    }
    source_bindings = {
        "d0": {
            "status": "RECOVERY_V2_R3_PASS",
            "canonical_digest": "124b404297c99e5a1913511a51caedec241ce05804949b390a14719814eb6c11",
        },
        "d1": {
            "status": "RECOVERY_V2_R3_METADATA_PASS",
            "canonical_digest": "00ad4968256606e95022f22d7502990223274738ca8066526134cae1945cbf40",
        },
        "d2": {
            "status": "RECOVERY_V2_R3_PASS",
            "canonical_digest": "5b3c8497dccda450b1b2682364c4f07bcf017e657d999ccb1d765aae9eec224c",
        },
        "c10": {
            "status": "BLOCKED_NO_GO_NOVELTY_HISTORICAL_SOURCE",
            "canonical_digest": "efc4e880793166fdf2e7869396cc0d0df9699e27c53071aa83c83ddce472e849",
            "downstream_authorized": False,
        },
    }
    access = {
        "reserve_image_bytes_accessed": False,
        "model_loaded_for_identity_verification": True,
        "model_used_for_inference": False,
        "cuda_accessed": False,
        "outcome_accessed": False,
    }
    gate_payload = {
        "formal": True,
        "authorization_scope": "D7_RECOVERY_V2_FIDELITY_ONLY",
        "execution_authorization": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE",
        "authorization_slot": slot,
        "shard_count": 8,
        "shard_mapping": [index % 8 for index in range(500)],
        "plan_digest": plan.canonical_digest,
        "budget_digest": estimate.canonical_digest,
        "preflight_digest": preflight.canonical_digest,
        "code_binding": producer,
        "dependency_binding": dependencies,
        "model_binding": model,
        "source_bindings": source_bindings,
        "access_attestation": access,
    }
    provenance_bindings = {
        "d7_authorization": {
            "status": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE",
            "digest": canonical_sha256(slot),
        },
        "producer": producer,
        "dependencies": dependencies,
        "model": model,
        "source_bindings": source_bindings,
    }
    context_binding = {
        "formal": True,
        "authorization_scope": "D7_RECOVERY_V2_FIDELITY_ONLY",
        "execution_authorization": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE",
        "authorization_slot_digest": canonical_sha256(slot),
        "shard_count": 8,
        "shard_mapping": [index % 8 for index in range(500)],
        "plan_digest": plan.canonical_digest,
        "budget_digest": estimate.canonical_digest,
        "preflight_digest": preflight.canonical_digest,
        "code_binding_digest": canonical_sha256(producer),
        "dependency_binding_digest": canonical_sha256(dependencies),
        "model_binding_digest": canonical_sha256(model),
        "source_bindings_digest": canonical_sha256(source_bindings),
        "access_attestation": access,
        "gate_context_digest": canonical_sha256(gate_payload),
        "provenance_bindings": provenance_bindings,
    }
    context_binding["runtime_binding_digest"] = canonical_sha256(context_binding)
    checkpoint = stage_d_runtime_module._initial_checkpoint(
        plan, estimate, preflight, synthetic,
    )
    checkpoint["execution_context"] = context_binding
    checkpoint["canonical_digest"] = canonical_sha256({
        key: value for key, value in checkpoint.items()
        if key != "canonical_digest"
    })
    root.mkdir()
    for name in (
        stage_d_runtime_module.LIFECYCLE_LOCK_FILENAME,
        stage_d_runtime_module.CHECKPOINT_LOCK_FILENAME,
        stage_d_runtime_module.ARTIFACT_ACTIVITY_LOCK_FILENAME,
    ):
        (root / name).write_bytes(b"")
    (root / stage_d_runtime_module.RUNTIME_CHECKPOINT_FILENAME).write_bytes(
        canonical_json_bytes(checkpoint) + b"\n"
    )
    return root, {
        "authorization_slot_digest": context_binding["authorization_slot_digest"],
        "gate_context_digest": context_binding["gate_context_digest"],
        "runtime_binding_digest": context_binding["runtime_binding_digest"],
        "code_binding_digest": context_binding["code_binding_digest"],
        "dependency_binding_digest": context_binding["dependency_binding_digest"],
        "model_binding_digest": context_binding["model_binding_digest"],
        "source_bindings_digest": context_binding["source_bindings_digest"],
    }


@pytest.fixture(scope="module")
def formal_checkpoint_root(
    controls, tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    return _formal_checkpoint_only_root(
        controls, tmp_path_factory.mktemp("d5-formal-control") / "root",
    )


def _source_attempt(root: Path, unit: Any, number: int = 1) -> Path:
    return (
        root / "units" / unit.canonical_digest / "attempts"
        / f"attempt-{number:06d}"
    )


def _retry_attempt(root: Path, unit_digest: str) -> Path:
    return root / "units" / unit_digest / "attempts" / "attempt-000002"


def _staging_entries(root: Path) -> list[Path]:
    return [child for child in root.iterdir() if child.name.startswith(STAGING_PREFIX)]


def _bytes_by_name(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_retained_root_descriptor_detects_path_aba(tmp_path: Path) -> None:
    root = tmp_path / "execution-root"
    root.mkdir()
    with stage_d_finalize_module._retained_root(root) as (
        descriptor_root, descriptor, identity,
    ):
        displaced = tmp_path / "displaced-root"
        root.rename(displaced)
        root.mkdir()
        assert descriptor_root.stat().st_ino == displaced.stat().st_ino
        with pytest.raises(
            StageDFinalizationIntegrityError,
            match="identity changed",
        ):
            stage_d_finalize_module._verify_root_binding(
                root, descriptor, identity,
            )


def test_descriptor_bound_publish_rejects_staging_path_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "execution-root"
    root.mkdir()
    staging_name = STAGING_PREFIX + "race"
    staging = root / staging_name
    staging.mkdir()
    expected = stage_d_finalize_module._root_identity(staging)
    staging.rename(root / "displaced-staging")
    staging.mkdir()
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    retained = os.open(
        root / "displaced-staging", os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        with pytest.raises(
            StageDFinalizationIntegrityError,
            match="staging directory identity changed",
        ):
            stage_d_finalize_module._publish_directory_no_replace(
                descriptor, staging_name, "final", retained,
            )
    finally:
        os.close(retained)
        os.close(descriptor)
    assert not (root / "final").exists()


def test_staging_creation_failure_removes_only_its_own_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "execution-root"
    root.mkdir()
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    real_open = stage_d_finalize_module.os.open

    def fail_staging_open(path, flags, *args, **kwargs):
        if (
            isinstance(path, str)
            and path.startswith(STAGING_PREFIX)
            and kwargs.get("dir_fd") == descriptor
        ):
            raise OSError("injected retained staging open failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(stage_d_finalize_module.os, "open", fail_staging_open)
    try:
        with pytest.raises(OSError, match="injected retained staging"):
            stage_d_finalize_module._create_staging_directory(descriptor)
    finally:
        os.close(descriptor)
    assert _staging_entries(root) == []


def _rewrite_retry_provenance(
    root: Path, unit_digest: str,
    mutate: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    attempt = _retry_attempt(root, unit_digest)
    provenance_path = attempt / "L0/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    mutate(provenance)
    provenance["canonical_digest"] = canonical_sha256({
        key: value for key, value in provenance.items()
        if key != "canonical_digest"
    })
    provenance_raw = canonical_json_bytes(provenance) + b"\n"
    provenance_path.write_bytes(provenance_raw)

    manifest_path = attempt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        row for row in manifest["artifacts"]
        if row["path"] == "L0/provenance.json"
    )
    record["bytes"] = len(provenance_raw)
    record["sha256"] = hashlib.sha256(provenance_raw).hexdigest()
    manifest["provenance_sha256"] = record["sha256"]
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    manifest_path.write_bytes(manifest_raw)

    status_path = attempt / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    status_path.write_bytes(canonical_json_bytes(status) + b"\n")
    return provenance


def _resign_retry_attempt_checkpoint(
    root: Path, unit_digest: str, *, retry_digest: str | None = None,
) -> None:
    checkpoint = _read_checkpoint(root)
    row = next(
        item for item in checkpoint["units"]
        if item["unit_digest"] == unit_digest
    )
    attempt = _retry_attempt(root, unit_digest)
    retry_row = row["attempts"][1]
    if retry_digest is not None:
        row["retry_authorization_digest"] = retry_digest
        retry_row["retry_authorization_digest"] = retry_digest
    retry_row["status_sha256"] = hashlib.sha256(
        (attempt / "status.json").read_bytes()
    ).hexdigest()
    inventory = [
        {
            "path": path.relative_to(attempt).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(attempt.rglob("*")) if path.is_file()
    ]
    retry_row["attempt_inventory_digest"] = canonical_sha256({
        "files": inventory
    })
    retry_row["resources"]["control_and_provenance_bytes"] = sum(
        (attempt / relative).stat().st_size
        for relative in ("status.json", "manifest.json", "L0/provenance.json")
    )
    retry_row["canonical_digest"] = canonical_sha256({
        key: value for key, value in retry_row.items()
        if key != "canonical_digest"
    })
    row["resources"] = {
        key: (
            max(attempt_row["resources"][key] for attempt_row in row["attempts"])
            if key == "cuda_peak_reserved_bytes" else
            sum(attempt_row["resources"][key] for attempt_row in row["attempts"])
        )
        for key in ZERO_RESOURCES
    }
    checkpoint["resources"] = {
        key: (
            max(unit_row["resources"][key] for unit_row in checkpoint["units"])
            if key == "cuda_peak_reserved_bytes" else
            sum(unit_row["resources"][key] for unit_row in checkpoint["units"])
        )
        for key in ZERO_RESOURCES
    }
    _write_checkpoint(root, checkpoint)


def test_full_500_1000_root_finalizes_and_independently_validates(
    controls, valid_root: Path,
) -> None:
    finalized = finalize_stage_d_root(valid_root, *controls)
    validated = validate_stage_d_root(valid_root, *controls)

    assert set(finalized) == set(validated) == RESULT_KEYS
    assert finalized.to_dict() == validated.to_dict()
    assert finalized.package_complete is True
    assert finalized.analysis_authorized is False
    assert finalized.scientific_validity == "PASS"
    assert finalized.workflow_status == "PASS"
    assert finalized.plan_digest == controls[0].canonical_digest
    assert set(path.name for path in (valid_root / "final").iterdir()) == FINAL_FILES
    assert finalized.reconciliation["expected_image_count"] == 500
    assert finalized.reconciliation["expected_unit_count"] == 1000
    assert finalized.reconciliation["image_count"] == 500
    assert finalized.reconciliation["unit_count"] == 1000
    assert len(finalized.reconciliation["rows"]) == 1000
    assert finalized.evidence_report["counts"]["images"] == 500
    assert finalized.evidence_report["counts"]["groups"] == 500
    assert finalized.evidence_report["counts"]["units"] == 1000
    assert finalized.evidence_report["counts"]["attempts"] == 1000
    assert finalized.evidence_report["counts"]["scientific_failures"] == 0
    assert finalized.reconciliation["expected_counts"] == (
        finalized.reconciliation["observed_counts"]
    )
    assert finalized.reconciliation["checkpoint_failure_table_reconciled"] is True
    assert finalized.reconciliation["checkpoint_resources_reconciled"] is True
    assert finalized.reconciliation["no_rows_dropped"] is True
    assert finalized.reconciliation["zero_imputation_performed"] is False
    assert len(finalized.reconciliation["group_rows"]) == 500
    assert len(finalized.reconciliation["failure_rows"]) == 1000
    assert finalized.reconciliation["na_rows"] == ()
    retention = finalized.reconciliation["retention"]
    assert retention["observed"]["L0_unit_coverage"] == 1000
    assert retention["observed"]["L1_unit_coverage"] == 1000
    assert retention["observed"]["L2_unit_coverage"] == 10
    assert retention["observed"]["L3_file_count"] == 0
    assert finalized.evidence_report["analysis_performed"] is False
    assert finalized.evidence_report["bootstrap_performed"] is False
    assert finalized.evidence_report["decision_performed"] is False
    assert set(finalized.evidence_report) == {
        "schema_version", "document_type", "package_complete",
        "workflow_status", "scientific_validity", "analysis_authorized",
        "plan_digest", "estimate_digest", "preflight_digest",
        "execution_context_digest", "counts", "scientific_failures",
        "authorization", "analysis_boundary", "actual_cost_audit",
        "analysis_performed", "bootstrap_performed",
        "decision_performed", "reconciliation_digest",
        "source_inventory_digest", "canonical_digest",
    }
    cost = finalized.actual_cost_audit
    assert cost["status"] == "PASS_COST"
    assert cost["evaluated"] is True
    assert cost["actual_cost_includes_all_attempts_and_audit_repeats"] is True
    assert cost["resource_totals"] == finalized.reconciliation["resources"]
    assert cost["source_file_count"] == finalized.source_inventory["file_count"]
    assert set(finalized.reconciliation) == {
        "schema_version", "document_type", "plan_digest",
        "expected_image_count", "expected_unit_count", "image_count",
        "unit_count", "expected_counts", "observed_counts",
        "state_counts", "group_rows", "rows", "failure_rows",
        "na_rows", "resources", "retention",
        "checkpoint_failure_table_reconciled",
        "checkpoint_resources_reconciled", "no_rows_dropped",
        "zero_imputation_performed", "canonical_digest",
    }
    assert set(finalized.source_inventory) == {
        "schema_version", "document_type", "root_scope", "files",
        "file_count", "total_bytes", "canonical_digest",
    }
    assert set(finalized.payload_manifest) == {
        "schema_version", "document_type", "immutable",
        "artifact_count", "artifacts", "canonical_digest",
    }
    for document in (
        finalized.evidence_report, finalized.reconciliation,
        finalized.source_inventory, finalized.payload_manifest,
    ):
        assert document["canonical_digest"] == canonical_sha256({
            key: value for key, value in document.items()
            if key != "canonical_digest"
        })

    status_path = valid_root / "final/status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert set(status) == {
        "schema_version", "document_type", "terminal",
        "package_complete", "scientific_validity", "workflow_status",
        "plan_digest", "payload_manifest_sha256",
        "sealed_root_inventory", "canonical_digest",
    }
    assert status["canonical_digest"] == canonical_sha256({
        key: value for key, value in status.items()
        if key != "canonical_digest"
    })
    assert finalized.final_status_sha256 == hashlib.sha256(
        status_path.read_bytes()
    ).hexdigest()
    inventory_paths = [
        row["path"] for row in finalized.whole_root_ordered_inventory
    ]
    assert inventory_paths == sorted(inventory_paths)
    assert set(f"final/{name}" for name in FINAL_FILES) <= set(inventory_paths)
    assert finalized.root_inventory["file_count"] == len(inventory_paths)

    serialized = json.dumps(finalized.to_dict()).casefold()
    assert all(
        token not in serialized
        for token in ("auc", "prediction", "logit", "ranking")
    )


def test_complete_scientific_failure_seals_complete_but_invalid(
    controls, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    failed_digest = controls[0].units[0].canonical_digest
    def behavior(unit):
        if unit.canonical_digest == failed_digest:
            raise StageDScientificFailure(
                "degenerate_direction", phase="attribution",
                l0_scalars={"status": "NA"},
            )
        return _payload(unit)

    root = _build_full_root(
        controls,
        tmp_path_factory.mktemp("d5-failed") / "execution-root",
        behavior,
    )
    finalized = finalize_stage_d_root(root, *controls)
    independently_validated = validate_stage_d_root(root, *controls)

    assert finalized.package_complete is True
    assert finalized.analysis_authorized is False
    assert finalized.scientific_validity == "FAIL"
    assert finalized.workflow_status == "PASS"
    assert independently_validated.to_dict() == finalized.to_dict()
    assert finalized.evidence_report["counts"]["scientific_failures"] == 1
    expected = finalized.reconciliation["expected_counts"]
    observed = finalized.reconciliation["observed_counts"]
    assert {
        name: expected[name]
        for name in (
            "execution_units", "map_work_items", "fidelity_curves",
            "fidelity_curve_points", "logical_work_items",
        )
    } == {
        "execution_units": 1000,
        "map_work_items": 1010,
        "fidelity_curves": 10000,
        "fidelity_curve_points": 210000,
        "logical_work_items": 211010,
    }
    assert {
        name: expected[name] - observed[name]
        for name in (
            "execution_units", "map_work_items", "fidelity_curves",
            "fidelity_curve_points", "logical_work_items",
        )
    } == {
        "execution_units": 0,
        "map_work_items": 1,
        "fidelity_curves": 10,
        "fidelity_curve_points": 210,
        "logical_work_items": 211,
    }
    assert finalized.reconciliation["state_counts"] == {
        "scientific_failed": 1, "succeeded": 999,
    }
    assert list(finalized.reconciliation["na_rows"]) == [{
        "unit_digest": failed_digest, "reason": "degenerate_direction",
    }]
    reasons = [
        reason
        for row in finalized.evidence_report["scientific_failures"]
        for reason in row["reasons"]
    ]
    assert "degenerate_direction" in reasons
    assert "synthetic_endpoint_parity_failed" not in reasons
    assert finalized.reconciliation["no_rows_dropped"] is True
    assert finalized.reconciliation["zero_imputation_performed"] is False


def test_checkpoint_control_accepts_formal_only_with_d9_authority(
    controls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, digests = _authorized_formal_checkpoint_only_root(
        controls, tmp_path / "formal-control",
    )

    with pytest.raises(StageDFinalizationBlockedError, match="formal roots"):
        stage_d_finalize_module._checkpoint_control(root, *controls)

    for name, digest in digests.items():
        monkeypatch.setattr(
            stage_d_finalize_module, "_FORMAL_" + name.upper(), digest,
        )
    checkpoint = stage_d_finalize_module._checkpoint_control(
        root, *controls, allow_formal=True,
    )
    assert checkpoint["execution_context"]["formal"] is True
    assert checkpoint["execution_context"]["execution_authorization"] == (
        "AUTHORIZED_STAGE_D_FIDELITY_SCOPE"
    )


def test_authorized_infrastructure_retry_finalizes_and_validates_exactly(
    controls, authorized_retry_root: tuple[Path, str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    source, retried_digest, retry = authorized_retry_root
    root = _copy_root(source, tmp_path, "authorized-retry")
    finalized = finalize_stage_d_root(root, *controls)
    validated = validate_stage_d_root(root, *controls)

    assert validated.to_dict() == finalized.to_dict()
    assert finalized.package_complete is True
    assert finalized.workflow_status == "PASS"
    assert finalized.evidence_report["counts"]["attempts"] == 1001
    assert finalized.reconciliation["expected_counts"]["attempts"] == 1001
    assert finalized.reconciliation["observed_counts"]["attempts"] == 1001
    row = next(
        item for item in finalized.reconciliation["rows"]
        if item["unit_digest"] == retried_digest
    )
    assert row["retry_authorization_digest"] == retry["canonical_digest"]
    assert len(row["attempts"]) == 2
    assert row["attempts"][0]["state"] == "infrastructure_failed"
    assert row["attempts"][0]["result_manifest_published"] is False
    assert row["attempts"][0]["published_l0_l1_payload_count"] == 0
    assert row["attempts"][1]["state"] == "succeeded"
    assert row["attempts"][1]["retry_authorization_digest"] == (
        retry["canonical_digest"]
    )
    assert row["attempts"][1]["provenance"]["execution"]["retry"] == retry


@pytest.mark.parametrize(
    "tamper",
    [
        "checkpoint_retry_authorization_digest",
        "provenance_retry_authorization",
        "provenance_prior_attempt_digest",
    ],
)
@pytest.mark.parametrize("operation", ["finalize", "validate"])
def test_finalize_and_validate_reject_tampered_authorized_retry_chain(
    controls, authorized_retry_root: tuple[Path, str, dict[str, Any]],
    tmp_path: Path, tamper: str, operation: str,
) -> None:
    source, retried_digest, _retry = authorized_retry_root
    root = _copy_root(source, tmp_path, f"{operation}-{tamper}")
    if operation == "validate":
        finalize_stage_d_root(root, *controls)
    replacement = "f" * 64

    if tamper == "checkpoint_retry_authorization_digest":
        checkpoint = _read_checkpoint(root)
        row = next(
            item for item in checkpoint["units"]
            if item["unit_digest"] == retried_digest
        )
        row["retry_authorization_digest"] = replacement
        _write_checkpoint(root, checkpoint)
    else:
        def mutate(provenance: dict[str, Any]) -> None:
            retry = provenance["bindings"]["execution"]["retry"]
            if tamper == "provenance_retry_authorization":
                retry["canonical_digest"] = replacement
            else:
                retry["prior_attempt_digest"] = replacement
                retry["canonical_digest"] = canonical_sha256({
                    key: value for key, value in retry.items()
                    if key != "canonical_digest"
                })

        provenance = _rewrite_retry_provenance(root, retried_digest, mutate)
        retry_digest = (
            provenance["bindings"]["execution"]["retry"]["canonical_digest"]
            if tamper == "provenance_prior_attempt_digest" else None
        )
        _resign_retry_attempt_checkpoint(
            root, retried_digest, retry_digest=retry_digest,
        )

    with pytest.raises(
        StageDFinalizationIntegrityError,
        match="retry|authorization|provenance|checkpoint|attempt",
    ):
        (
            finalize_stage_d_root if operation == "finalize"
            else validate_stage_d_root
        )(root, *controls)
    assert (root / "final").exists() is (operation == "validate")


@pytest.mark.parametrize(
    "mutation, match",
    [
        (
            lambda root, unit: (_source_attempt(root, unit) / "L1/fidelity_curves.npz").unlink(),
            "missing|source|attempt|inventory",
        ),
        (
            lambda root, unit: (_source_attempt(root, unit) / "L1/fidelity_curves.npz").write_bytes(b"not-an-npz"),
            "source|attempt|hash|payload|inventory",
        ),
    ],
)
def test_finalize_rejects_missing_or_tampered_source_evidence(
    controls, valid_root: Path, tmp_path: Path, mutation, match: str,
) -> None:
    root = _copy_root(valid_root, tmp_path, "source-tamper")
    mutation(root, controls[0].units[0])
    with pytest.raises(StageDFinalizationIntegrityError, match=match):
        finalize_stage_d_root(root, *controls)
    assert not (root / "final").exists()


def test_active_claim_blocks_without_publishing_final_controls(
    controls, tmp_path: Path,
) -> None:
    root = _minimal_checkpoint_root(controls, tmp_path / "active")
    checkpoint = _read_checkpoint(root)
    checkpoint["generation"] = 1
    checkpoint["previous_checkpoint_digest"] = checkpoint["canonical_digest"]
    checkpoint["lifecycle"].update({
        "state": "running", "active_shard_leases": 1,
    })
    checkpoint["shards"][0].update({
        "state": "leased", "owner_token_digest": "a" * 64,
    })
    _write_checkpoint(root, checkpoint)

    with pytest.raises(StageDFinalizationBlockedError, match="active|claim"):
        finalize_stage_d_root(root, *controls)
    assert not (root / "final").exists()


@pytest.mark.parametrize("unsafe_kind", ["extra", "symlink", "hardlink"])
def test_finalize_rejects_extra_symlink_and_hardlink(
    controls, valid_root: Path, tmp_path: Path, unsafe_kind: str,
) -> None:
    root = _copy_root(valid_root, tmp_path, unsafe_kind)
    attempt = _source_attempt(root, controls[0].units[0])
    if unsafe_kind == "extra":
        (attempt / "extra.dat").write_bytes(b"extra")
    elif unsafe_kind == "symlink":
        (attempt / "extra-link").symlink_to("status.json")
    else:
        os.link(attempt / "status.json", attempt / "extra-hardlink")

    with pytest.raises(StageDFinalizationIntegrityError):
        finalize_stage_d_root(root, *controls)
    assert not (root / "final").exists()


def test_finalize_rejects_extra_empty_attempt_directory(
    controls, valid_root: Path, tmp_path: Path,
) -> None:
    root = _copy_root(valid_root, tmp_path, "extra-empty-directory")
    (_source_attempt(root, controls[0].units[0]) / "L0" / "extra-empty").mkdir()

    with pytest.raises(StageDFinalizationIntegrityError, match="directory|closed"):
        finalize_stage_d_root(root, *controls)
    assert not (root / "final").exists()


def test_final_controls_are_closed_and_never_overwritten(
    controls, valid_root: Path, tmp_path: Path,
) -> None:
    root = _copy_root(valid_root, tmp_path, "immutable")
    finalize_stage_d_root(root, *controls)
    final_root = root / "final"
    before = _bytes_by_name(final_root)

    with pytest.raises(
        (StageDFinalizationBlockedError, StageDFinalizationIntegrityError),
        match="exist|overwrite|seal|closed|extra|layout",
    ):
        finalize_stage_d_root(root, *controls)
    assert _bytes_by_name(final_root) == before

    (final_root / "unknown.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(StageDFinalizationIntegrityError, match="closed|extra|layout"):
        validate_stage_d_root(root, *controls)


@pytest.mark.parametrize("operation", ["finalize", "validate"])
def test_formal_root_is_rejected_before_source_traversal_and_never_published(
    controls, formal_checkpoint_root: Path,
    monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    root = formal_checkpoint_root
    final_root = root / "final"

    def root_tree_traversal_is_forbidden(*_args, **_kwargs):
        pytest.fail("formal-root refusal must precede root-tree traversal")

    monkeypatch.setattr(
        stage_d_finalize_module, "_root_layout", root_tree_traversal_is_forbidden,
    )
    monkeypatch.setattr(
        stage_d_finalize_module._runtime,
        "_validate_checkpoint",
        root_tree_traversal_is_forbidden,
    )
    with pytest.raises(
        StageDFinalizationBlockedError, match="formal|authority",
    ):
        (
            finalize_stage_d_root if operation == "finalize"
            else validate_stage_d_root
        )(root, *controls)
    assert not final_root.exists()
    assert _staging_entries(root) == []


@pytest.mark.parametrize(
    "drifted_plan",
    [
        lambda plan: replace(plan, canonical_digest="f" * 64),
        lambda plan: replace(plan, coordinate_digest="f" * 64),
    ],
)
def test_plan_digest_or_coordinate_drift_is_rejected_before_root_access(
    controls, monkeypatch: pytest.MonkeyPatch, drifted_plan,
) -> None:
    plan = drifted_plan(controls[0])

    def root_access_is_forbidden(*_args, **_kwargs):
        pytest.fail("invalid plan must be rejected before output-root access")

    monkeypatch.setattr(
        stage_d_finalize_module, "_safe_root", root_access_is_forbidden,
    )
    with pytest.raises(
        StageDFinalizationIntegrityError,
        match="invalid Stage D control inputs|canonical|coordinate",
    ):
        finalize_stage_d_root(
            Path("/root-must-not-be-inspected"), plan,
            controls[1], controls[2],
        )


@pytest.mark.parametrize("failed_name", ["source_inventory.json", "status.json"])
def test_staging_write_failure_is_hidden_cleaned_and_retryable(
    controls, valid_root: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, failed_name: str,
) -> None:
    root = _copy_root(valid_root, tmp_path, f"staging-{failed_name}")
    real_atomic_create = stage_d_finalize_module._atomic_create_at
    writes: list[str] = []

    def fail_one_staging_write(
        directory_descriptor: int, name: str, raw: bytes,
    ) -> None:
        writes.append(name)
        if name == failed_name:
            raise OSError(f"injected staging write failure: {failed_name}")
        real_atomic_create(directory_descriptor, name, raw)

    monkeypatch.setattr(
        stage_d_finalize_module, "_atomic_create_at", fail_one_staging_write,
    )
    with pytest.raises(OSError, match="injected staging write failure"):
        finalize_stage_d_root(root, *controls)
    assert failed_name in writes
    assert not (root / "final").exists()
    assert _staging_entries(root) == []

    monkeypatch.setattr(
        stage_d_finalize_module, "_atomic_create_at", real_atomic_create,
    )
    finalized = finalize_stage_d_root(root, *controls)
    assert finalized.package_complete is True
    assert set(path.name for path in (root / "final").iterdir()) == FINAL_FILES
    assert validate_stage_d_root(root, *controls).to_dict() == finalized.to_dict()


@pytest.mark.parametrize("existing_kind", ["complete", "partial"])
def test_existing_final_directory_is_never_overwritten(
    controls, valid_root: Path, tmp_path: Path, existing_kind: str,
) -> None:
    root = _copy_root(valid_root, tmp_path, f"existing-{existing_kind}")
    final_root = root / "final"
    if existing_kind == "complete":
        finalize_stage_d_root(root, *controls)
    else:
        final_root.mkdir()
        (final_root / "evidence_report.json").write_bytes(b"partial-final\n")
    before = _bytes_by_name(final_root)

    with pytest.raises(StageDFinalizationBlockedError, match="final|claim|reuse"):
        finalize_stage_d_root(root, *controls)
    assert _bytes_by_name(final_root) == before
    assert _staging_entries(root) == []


@pytest.mark.parametrize(
    "target",
    ["evidence_report.json", "reconciliation.json", "payload_manifest.json"],
)
def test_validator_recomputes_closed_outputs_from_source(
    controls, valid_root: Path, tmp_path: Path, target: str,
) -> None:
    root = _copy_root(valid_root, tmp_path, "final-tamper")
    finalize_stage_d_root(root, *controls)
    path = root / "final" / target
    document = json.loads(path.read_text(encoding="utf-8"))
    document["unexpected"] = True
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StageDFinalizationIntegrityError, match="source|recomput|schema|control"):
        validate_stage_d_root(root, *controls)


def test_independent_validator_rejects_post_finalize_source_tamper(
    controls, valid_root: Path, tmp_path: Path,
) -> None:
    root = _copy_root(valid_root, tmp_path, "post-final-source-tamper")
    finalize_stage_d_root(root, *controls)
    path = _source_attempt(root, controls[0].units[0]) / "status.json"
    path.write_bytes(b"{}\n")

    with pytest.raises(StageDFinalizationIntegrityError, match="source|attempt|status"):
        validate_stage_d_root(root, *controls)
