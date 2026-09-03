from __future__ import annotations

from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from vitnns import stage_c_c8_salvage_repair as repair
from vitnns import stage_c_c8_salvage_repair_selection as selection


def _unit(index: int):
    digest = f"{index:064x}"
    key = SimpleNamespace(canonical_digest=digest)
    key.to_dict = lambda: {"canonical_digest": digest}
    return SimpleNamespace(key=key, work_item_counts={}, audit=False)


def _plan(count: int = 2180):
    return SimpleNamespace(
        canonical_digest="a" * 64,
        units=tuple(_unit(index) for index in range(count)),
    )


def _record(index: int, origin: str, *, state: str = "succeeded"):
    return {
        "plan_index": index,
        "unit_key": {"canonical_digest": f"{index:064x}"},
        "origin": origin,
        "state": state,
        "validation_errors": [],
    }


def _three_way_audit():
    original = [
        _record(index, repair.ORIGINAL_SOURCE_ORIGIN)
        for index in range(1460)
    ]
    prior = [
        _record(
            index, repair.PRIOR_GAP_ORIGIN,
            state="failed" if index == 1460 else "succeeded",
        )
        for index in range(1460, 2029)
    ]
    repair_units = [f"{index:064x}" for index in range(2029, 2180)]
    return {
        "status": "PASS",
        "plan_digest": "a" * 64,
        "original_records": original,
        "prior_records": prior,
        "repair_unit_digests": repair_units,
        "origin_counts": {
            repair.ORIGINAL_SOURCE_ORIGIN: 1460,
            repair.PRIOR_GAP_ORIGIN: 569,
            repair.REPAIR_ORIGIN: 151,
        },
    }


def _no_lock(*_args, **_kwargs):
    @contextmanager
    def locked():
        yield

    return locked()


def test_repair_origins_are_the_authority_contract_strings():
    assert repair.ORIGINAL_SOURCE_ORIGIN == (
        "inherited_from_original_interrupted_root"
    )
    assert repair.PRIOR_GAP_ORIGIN == (
        "inherited_from_prior_partial_salvage_root"
    )
    assert repair.REPAIR_ORIGIN == "newly_run_in_salvage_repair_root"
    assert len({
        repair.ORIGINAL_SOURCE_ORIGIN,
        repair.PRIOR_GAP_ORIGIN,
        repair.REPAIR_ORIGIN,
    }) == 3


def test_source_audit_derives_exact_partition_and_does_not_trust_prior_states(
    monkeypatch, tmp_path,
):
    plan = _plan()
    digests = [unit.key.canonical_digest for unit in plan.units]
    original = digests[:1460]
    prior = digests[1460:2029]
    repair_set = digests[2029:]
    authority = SimpleNamespace(
        repo_root=tmp_path, source_root=tmp_path / "original",
        prior_gap_root=tmp_path / "prior",
        document={
            "original_interrupted_root": {"evidence_commit": "1" * 40},
            "prior_partial_root": {"evidence_commit": "2" * 40},
        },
        original_salvage_record=SimpleNamespace(launch_commit="3" * 40),
    )
    prior_states = {digest: "pending" for digest in digests[1460:]}
    prior_states[prior[0]] = "failed"
    for digest in prior[1:]:
        prior_states[digest] = "succeeded"
    # This reported incomplete attempt is invalid and therefore belongs to
    # the source-recomputed repair set, not the inherited set.
    prior_states[repair_set[0]] = "incomplete"

    monkeypatch.setattr(repair, "_authority", lambda value: value)
    custody_checks = []
    monkeypatch.setattr(
        repair, "_require_custody_clean",
        lambda repo, root, revision: custody_checks.append(
            (repo, root, revision)
        ),
    )
    monkeypatch.setattr(repair, "_validate_verifier", lambda *_args: {})
    monkeypatch.setattr(
        repair, "validate_c8_execution_plan",
        lambda *_args, **_kwargs: {"valid": True, "violations": []},
    )
    monkeypatch.setattr(
        repair, "_tree_inventory",
        lambda _repo, root, **_kwargs: {
            "digest": "4" * 64 if root == authority.source_root else "5" * 64,
            "total_bytes": 1,
        },
    )

    def read(path):
        if path.name == "c8_launch_authorization.json":
            return {
                "source": {
                    "execution_git_state": {
                        "commit": "6" * 40, "dirty": False,
                    },
                },
            }
        if path.name == "failure_table.json":
            return {
                "rows": [
                    {
                        "unit_key": {"canonical_digest": digest},
                        "state": "succeeded" if index < 1460 else "pending",
                    }
                    for index, digest in enumerate(digests)
                ],
            }
        raise AssertionError(f"unexpected read: {path}")

    monkeypatch.setattr(repair, "_read", read)
    monkeypatch.setattr(
        repair, "_execution_identity",
        lambda _authority, commit: {"commit": commit, "dirty": False},
    )
    monkeypatch.setattr(
        repair, "_prior_scope",
        lambda *_args: {
            "inherited_unit_digests": original,
            "gap_unit_digests": digests[1460:],
            "states": prior_states,
        },
    )

    def attempts(path):
        digest = path.parent.name
        index = int(digest, 16)
        if index < 2029 or index == 2029:
            return [path / "attempt-0001"]
        return []

    monkeypatch.setattr(repair, "attempt_directories", attempts)

    def validate(_attempt, _unit, *, origin, plan_index, **_kwargs):
        if plan_index == 2029:
            return (
                _record(plan_index, origin, state="rejected"),
                ["invalid prior attempt"],
            )
        state = "failed" if plan_index == 1460 else "succeeded"
        return _record(plan_index, origin, state=state), []

    monkeypatch.setattr(repair, "_validate_attempt", validate)

    audit = repair.audit_c8_salvage_repair_sources(
        authority, plan, object(), object(), verifier_identity={},
    )

    assert custody_checks == [(
        authority.repo_root, authority.source_root, "1" * 40,
    )]
    assert audit["counts"] == {
        "original_succeeded": 1460, "prior_succeeded": 568,
        "prior_terminal_failed": 1, "prior_terminal": 569,
        "repair": 151, "expected": 2180,
    }
    assert audit["prior_failed_unit_digests"] == [prior[0]]
    assert audit["prior_records"][0]["state"] == "failed"
    assert prior[0] not in audit["repair_unit_digests"]
    assert audit["repair_unit_digests"] == repair_set
    assert audit["rejected_prior_attempts"] == [{
        "plan_index": 2029,
        "unit_key": {"canonical_digest": repair_set[0]},
        "attempt_path": str(
            authority.prior_gap_root / "units" / repair_set[0]
            / "attempts" / "attempt-0001"
        ),
        "errors": ["invalid prior attempt"],
    }]


def test_invalid_reported_failed_attempt_is_rejected_into_repair(
    monkeypatch, tmp_path,
):
    plan = _plan()
    digests = [unit.key.canonical_digest for unit in plan.units]
    original = digests[:1460]
    valid_prior = digests[1460:2029]
    invalid_failed = digests[2029]
    repair_set = digests[2030:]
    authority = SimpleNamespace(
        repo_root=tmp_path, source_root=tmp_path / "original",
        prior_gap_root=tmp_path / "prior",
        document={
            "original_interrupted_root": {"evidence_commit": "1" * 40},
            "prior_partial_root": {"evidence_commit": "2" * 40},
        },
        original_salvage_record=SimpleNamespace(launch_commit="3" * 40),
    )
    prior_states = {digest: "pending" for digest in digests[1460:]}
    for digest in valid_prior[1:]:
        prior_states[digest] = "succeeded"
    prior_states[valid_prior[0]] = "failed"
    prior_states[invalid_failed] = "failed"
    monkeypatch.setattr(repair, "_authority", lambda value: value)
    monkeypatch.setattr(
        repair, "_require_custody_clean", lambda *_args: None,
    )
    monkeypatch.setattr(repair, "_validate_verifier", lambda *_args: {})
    monkeypatch.setattr(
        repair, "validate_c8_execution_plan",
        lambda *_args, **_kwargs: {"valid": True, "violations": []},
    )
    monkeypatch.setattr(
        repair, "_tree_inventory",
        lambda *_args, **_kwargs: {"digest": "4" * 64, "total_bytes": 1},
    )

    def read(path):
        if path.name == "c8_launch_authorization.json":
            return {
                "source": {
                    "execution_git_state": {
                        "commit": "6" * 40, "dirty": False,
                    },
                },
            }
        return {
            "rows": [
                {
                    "unit_key": {"canonical_digest": digest},
                    "state": "succeeded" if index < 1460 else "pending",
                }
                for index, digest in enumerate(digests)
            ],
        }

    monkeypatch.setattr(repair, "_read", read)
    monkeypatch.setattr(
        repair, "_execution_identity",
        lambda _authority, commit: {"commit": commit, "dirty": False},
    )
    monkeypatch.setattr(
        repair, "_prior_scope",
        lambda *_args: {
            "inherited_unit_digests": original,
            "gap_unit_digests": digests[1460:],
            "states": prior_states,
        },
    )
    monkeypatch.setattr(
        repair, "attempt_directories",
        lambda path: (
            [path / "attempt-0001"]
            if int(path.parent.name, 16) <= 2029 else []
        ),
    )

    def validate(_attempt, _unit, *, origin, plan_index, **_kwargs):
        if plan_index == 2029:
            return (
                _record(plan_index, origin, state="rejected"),
                ["failed attempt is not valid terminal scientific evidence"],
            )
        return _record(
            plan_index, origin,
            state="failed" if plan_index == 1460 else "succeeded",
        ), []

    monkeypatch.setattr(repair, "_validate_attempt", validate)

    audit = repair.audit_c8_salvage_repair_sources(
        authority, plan, object(), object(), verifier_identity={},
    )

    assert invalid_failed in audit["repair_unit_digests"]
    assert audit["repair_unit_digests"] == [invalid_failed, *repair_set]
    assert audit["rejected_prior_attempts"][0]["unit_key"] == {
        "canonical_digest": invalid_failed,
    }


@pytest.mark.parametrize(
    ("failure_type", "expected_state", "valid"),
    [
        (
            "vitnns.stage_c_value_path.StageCValuePathConvergenceError",
            "failed", True,
        ),
        ("builtins.RuntimeError", "rejected", False),
    ],
)
def test_only_valid_terminal_scientific_failure_is_inheritable(
    monkeypatch, tmp_path, failure_type, expected_state, valid,
):
    attempt = tmp_path / "units" / ("1" * 64) / "attempts" / "attempt-0001"
    (attempt / "L0").mkdir(parents=True)
    files = {
        "manifest.json": {"artifacts": []},
        "status.json": {
            "status": "failed", "terminal": True, "complete": False,
            "failure": {"details": {"type": failure_type}},
            "runner_telemetry": {},
        },
        "L0/provenance.json": {
            "bindings": {"code": {"commit": "7" * 40, "dirty": False}},
        },
        "L0/result.json": {"scalars": {}, "resources": {}},
    }
    for relative, value in files.items():
        (attempt / relative).write_text(json.dumps(value), encoding="utf-8")
    checked = SimpleNamespace(
        errors=(), valid=True, status="failed", terminal=True,
        complete=False, resumable=False, manifest_sha256="8" * 64,
        unit_identity_sha256="9" * 64, l0_l1_bytes=1, total_bytes=2,
    )
    monkeypatch.setattr(
        repair, "validate_execution_unit", lambda *_args, **_kwargs: checked,
    )
    monkeypatch.setattr(repair, "_expected_bindings", lambda *_args: {})

    record, errors = repair._validate_attempt(
        attempt, _unit(1), plan=object(),
        estimate=SimpleNamespace(caps={"effective": {}}),
        authority=object(), observed_git_state={},
        execution_identity={"commit": "7" * 40, "dirty": False},
        verifier={}, origin=repair.PRIOR_GAP_ORIGIN, plan_index=1,
    )

    assert record["state"] == expected_state
    assert (errors == []) is valid


def test_runtime_executes_only_authoritative_repair_set_and_skips_terminals(
    monkeypatch, tmp_path,
):
    plan = _plan(6)
    repair_set = [f"{index:064x}" for index in (1, 3, 5)]
    scope = {
        "original_unit_digests": [],
        "prior_unit_digests": [],
        "prior_failed_unit_digests": [],
        "repair_unit_digests": repair_set,
        "repair_execution_identity": {
            "commit": "c" * 40, "tree": "d" * 40,
            "source_subtree_path": "src/vitnns",
            "source_subtree_tree": "e" * 40, "dirty": False,
        },
        "states": {
            repair_set[0]: "succeeded",
            repair_set[1]: "failed",
            repair_set[2]: "pending",
        },
        "claim_tokens": {},
        "canonical_digest": "b" * 64,
    }
    authority = SimpleNamespace(
        repair_root=tmp_path, launch_commit="c" * 40,
    )
    monkeypatch.setattr(repair, "_authority", lambda value: value)
    monkeypatch.setattr(repair, "_scope", lambda *_args: scope)
    monkeypatch.setattr(
        repair, "_validate_partition", lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(repair, "_lifecycle_lock", _no_lock)
    monkeypatch.setattr(repair, "_checkpoint_lock", _no_lock)
    monkeypatch.setattr(repair, "_artifact_activity_lock", _no_lock)
    monkeypatch.setattr(
        repair, "_read_preflight",
        lambda *_args: SimpleNamespace(
            caps={"effective": {}}, to_dict=lambda: {},
        ),
    )
    monkeypatch.setattr(repair, "_publish_scope", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        repair, "_plan_digests",
        lambda current: [unit.key.canonical_digest for unit in current.units],
    )
    monkeypatch.setattr(
        repair, "_runtime_git_state", lambda _authority, _scope, raw: dict(raw),
    )
    monkeypatch.setattr(repair, "_require_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        repair, "attempt_directories", lambda *_args: [tmp_path / "attempt"],
    )
    monkeypatch.setattr(repair, "_capture_verifier", lambda *_args: {})
    monkeypatch.setattr(
        repair, "_validate_attempt",
        lambda _attempt, unit, **_kwargs: (
            _record(int(unit.key.canonical_digest, 16), repair.REPAIR_ORIGIN),
            [],
        ),
    )
    seen = []

    def run(unit, **kwargs):
        seen.append((unit.key.canonical_digest, kwargs))
        return SimpleNamespace(
            state="succeeded",
            to_dict=lambda: {"state": "succeeded"},
        )

    monkeypatch.setattr(repair, "run_execution_unit", run)
    backend = SimpleNamespace(trusted_recovery_context=authority)
    results = repair.run_c8_salvage_repair_shard(
        authority, plan, backend, shard_index=0, shard_count=1, device="cpu",
        runtime_git_state={"commit": authority.launch_commit, "dirty": False},
    )

    assert [item[0] for item in seen] == [repair_set[2]]
    assert seen[0][1]["resume"] is False
    assert [item["state"] for item in results] == [
        "succeeded", "failed", "succeeded",
    ]


def test_partition_validation_rejects_non_authoritative_runtime_unit():
    plan = _plan()
    audit = _three_way_audit()
    scope = {
        "original_unit_digests": [
            row["unit_key"]["canonical_digest"]
            for row in audit["original_records"]
        ],
        "prior_unit_digests": [
            row["unit_key"]["canonical_digest"]
            for row in audit["prior_records"]
        ],
        "prior_failed_unit_digests": [f"{1460:064x}"],
        "repair_unit_digests": list(audit["repair_unit_digests"]),
        "repair_execution_identity": {
            "commit": "1" * 40, "tree": "2" * 40,
            "source_subtree_path": "src/vitnns",
            "source_subtree_tree": "3" * 40, "dirty": False,
        },
        "states": {
            digest: "pending" for digest in audit["repair_unit_digests"]
        },
        "claim_tokens": {},
    }
    stolen = scope["original_unit_digests"][0]
    scope["repair_unit_digests"][0] = stolen
    scope["states"] = {
        digest: "pending" for digest in scope["repair_unit_digests"]
    }

    with pytest.raises(
        repair.RecoveryC8SalvageRepairError,
        match="partition|repair|scope|overlap",
    ):
        repair._validate_partition(SimpleNamespace(), plan, scope)


def test_finalize_blocks_only_nonterminal_repair_states(monkeypatch, tmp_path):
    digests = [f"{index:064x}" for index in range(151)]
    states = {digest: "succeeded" for digest in digests}
    states[digests[0]] = "failed"
    states[digests[1]] = "incomplete"
    scope = {
        "repair_unit_digests": digests, "states": states,
        "claim_tokens": {},
    }
    authority = SimpleNamespace(repair_root=tmp_path)
    monkeypatch.setattr(repair, "_authority", lambda value: value)
    monkeypatch.setattr(repair, "_scope", lambda *_args: scope)
    monkeypatch.setattr(
        repair, "_validate_partition", lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(repair, "_lifecycle_lock", _no_lock)
    monkeypatch.setattr(repair, "_checkpoint_lock", _no_lock)
    monkeypatch.setattr(repair, "_artifact_activity_lock", _no_lock)
    monkeypatch.setattr(
        repair, "build_c8_salvage_repair_combined_manifest",
        lambda *_args, **_kwargs: pytest.fail("blocked finalize built manifest"),
    )

    result = repair.finalize_c8_salvage_repair(
        authority, object(), object(),
    )

    assert result["status"] == "BLOCKED"
    assert result["manifest_written"] is False
    assert result["repair_state_counts"] == {
        "failed": 1, "incomplete": 1, "succeeded": 149,
    }


def test_finalize_accepts_succeeded_and_valid_failed_terminal_records(
    monkeypatch, tmp_path,
):
    digests = [f"{index:064x}" for index in range(151)]
    states = {digest: "succeeded" for digest in digests}
    states[digests[0]] = "failed"
    scope = {
        "repair_unit_digests": digests, "states": states,
        "claim_tokens": {},
    }
    authority = SimpleNamespace(
        repair_root=tmp_path / "repair", reports_root=tmp_path / "reports",
        combined_manifest_path=tmp_path / "reports" / "manifest.json",
    )
    manifest = {
        "status": "PASS",
        "records": [
            _record(0, repair.REPAIR_ORIGIN, state="failed"),
            _record(1, repair.REPAIR_ORIGIN),
        ],
    }
    monkeypatch.setattr(repair, "_authority", lambda value: value)
    monkeypatch.setattr(repair, "_scope", lambda *_args: scope)
    monkeypatch.setattr(
        repair, "_validate_partition", lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(repair, "_lifecycle_lock", _no_lock)
    monkeypatch.setattr(repair, "_checkpoint_lock", _no_lock)
    monkeypatch.setattr(repair, "_artifact_activity_lock", _no_lock)
    monkeypatch.setattr(repair, "_capture_verifier", lambda *_args: {})
    monkeypatch.setattr(repair, "_publish_scope", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        repair, "build_c8_salvage_repair_combined_manifest",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(repair, "_create_or_accept", lambda *_args: None)

    assert repair.finalize_c8_salvage_repair(
        authority, object(), object(),
    ) is manifest


def test_validate_rebuilds_from_sources_and_rejects_persisted_drift(
    monkeypatch, tmp_path,
):
    authority = SimpleNamespace(
        repair_root=tmp_path, combined_manifest_path=tmp_path / "manifest.json",
    )
    persisted = {"status": "PASS", "records": []}
    rebuilt = {"status": "PASS", "records": [{"state": "failed"}]}
    scope = {
        "repair_unit_digests": [], "states": {}, "claim_tokens": {},
    }
    monkeypatch.setattr(repair, "_authority", lambda value: value)
    monkeypatch.setattr(repair, "_scope", lambda *_args: scope)
    monkeypatch.setattr(repair, "_require_finalized_snapshot", lambda *_args: {})
    monkeypatch.setattr(repair, "_lifecycle_lock", _no_lock)
    monkeypatch.setattr(repair, "_checkpoint_lock", _no_lock)
    monkeypatch.setattr(repair, "_artifact_activity_lock", _no_lock)
    monkeypatch.setattr(repair, "_read", lambda *_args: persisted)
    monkeypatch.setattr(repair, "_validate_verifier", lambda *_args: {})
    monkeypatch.setattr(
        repair, "build_c8_salvage_repair_combined_manifest",
        lambda *_args, **_kwargs: rebuilt,
    )

    with pytest.raises(
        repair.RecoveryC8SalvageRepairError,
        match="source recomputation|differs",
    ):
        repair.validate_c8_salvage_repair_from_source(
            authority, object(), object(),
        )


def _identity(digit: str):
    return {
        "commit": digit * 40, "tree": digit * 40,
        "source_subtree_path": "src/vitnns",
        "source_subtree_tree": digit * 40, "dirty": False,
    }


def _selection_ready_record(index, origin, identity, *, state="succeeded"):
    record = _record(index, origin, state=state)
    record.update({
        "execution_git_identity": identity,
        "hashes": {
            "manifest_sha256": "a" * 64,
            "status_sha256": "b" * 64,
            "provenance_sha256": "c" * 64,
            "result_sha256": "d" * 64,
            "unit_identity_sha256": f"{index:064x}",
        },
        "scalars": {},
        "provenance": {
            "bindings": {
                "code": {"commit": identity["commit"], "dirty": False},
            },
        },
        "resources": {}, "runner_telemetry": {},
        "artifact_inventory": [], "work_item_counts": {},
        "audit": False, "attempt_path": f"/attempts/{index}",
        "l0_l1_cap_accounting_bytes": 0,
        "level_bytes": {
            "L0": 0, "L1": 0, "L2": 0, "L3": 0, "control": 0,
        },
        "total_regular_file_bytes": 0,
    })
    return record


def test_combined_manifest_exposes_selection_required_three_origin_fields(
    monkeypatch, tmp_path,
):
    plan = _plan()
    identities = {
        repair.ORIGINAL_SOURCE_ORIGIN: _identity("1"),
        repair.PRIOR_GAP_ORIGIN: _identity("2"),
        repair.REPAIR_ORIGIN: _identity("3"),
    }
    original = [
        _selection_ready_record(
            index, repair.ORIGINAL_SOURCE_ORIGIN,
            identities[repair.ORIGINAL_SOURCE_ORIGIN],
        )
        for index in range(1460)
    ]
    prior = [
        _selection_ready_record(
            index, repair.PRIOR_GAP_ORIGIN,
            identities[repair.PRIOR_GAP_ORIGIN],
            state="failed" if index == 1460 else "succeeded",
        )
        for index in range(1460, 2029)
    ]
    repair_digests = [f"{index:064x}" for index in range(2029, 2180)]
    inventory = {"digest": "4" * 64, "total_bytes": 2}
    audit = {
        "canonical_digest": "5" * 64,
        "original_root_inventory": inventory,
        "prior_gap_root_inventory": inventory,
        "original_records": original, "prior_records": prior,
    }
    estimate = SimpleNamespace(to_dict=lambda: {"measured": True})
    authority = SimpleNamespace(
        repair_root=tmp_path / "repair", repo_root=tmp_path,
        path=tmp_path / "authority.json", file_sha256="6" * 64,
        canonical_digest="7" * 64, launch_commit="8" * 40,
        document={"original_salvage_authority": {"canonical_digest": "9" * 64}},
    )
    scope = {
        "repair_unit_digests": repair_digests,
        "repair_execution_identity": identities[repair.REPAIR_ORIGIN],
        "states": {digest: "succeeded" for digest in repair_digests},
    }
    monkeypatch.setattr(repair, "_authority", lambda value: value)
    monkeypatch.setattr(repair, "_scope", lambda *_args: scope)
    monkeypatch.setattr(repair, "_validate_verifier", lambda *_args: {
        "commit": "a" * 40, "dirty": False,
    })
    monkeypatch.setattr(repair, "_read_preflight", lambda *_args: estimate)
    monkeypatch.setattr(repair, "_estimate", lambda *_args: estimate)
    monkeypatch.setattr(
        repair, "audit_c8_salvage_repair_sources",
        lambda *_args, **_kwargs: audit,
    )
    monkeypatch.setattr(
        repair, "_validate_partition",
        lambda *_args, **_kwargs: {
            "path": "audit.json", "file_sha256": "b" * 64,
            "canonical_digest": "5" * 64,
        },
    )
    monkeypatch.setattr(
        repair, "attempt_directories", lambda path: [path / "attempt-0001"],
    )
    monkeypatch.setattr(
        repair, "_validate_attempt",
        lambda _attempt, _unit_value, *, plan_index, **_kwargs: (
            _selection_ready_record(
                plan_index, repair.REPAIR_ORIGIN,
                identities[repair.REPAIR_ORIGIN],
            ),
            [],
        ),
    )
    monkeypatch.setattr(repair, "_tree_inventory", lambda *_args: inventory)
    monkeypatch.setattr(
        repair, "_control_hashes", lambda *_args: {"scope.json": "c" * 64},
    )
    monkeypatch.setattr(
        repair, "_c7_report_binding",
        lambda *_args: {"source_batch_control_bytes": 1},
    )

    manifest = repair.build_c8_salvage_repair_combined_manifest(
        authority, plan, object(),
        verifier_identity={"commit": "a" * 40, "dirty": False},
    )
    adapted, adapted_identities, adapted_counts = selection._adapt_records(
        manifest
    )

    assert len(adapted) == 2180
    assert manifest["origin_counts"] == {
        repair.ORIGINAL_SOURCE_ORIGIN: 1460,
        repair.PRIOR_GAP_ORIGIN: 569,
        repair.REPAIR_ORIGIN: 151,
    }
    assert set(manifest["producer_identities"]) == {
        repair.ORIGINAL_SOURCE_ORIGIN, repair.PRIOR_GAP_ORIGIN,
        repair.REPAIR_ORIGIN,
    }
    assert set(manifest["root_inventories"]) == set(
        manifest["producer_identities"]
    )
    assert adapted_counts == manifest["origin_counts"]
    assert adapted_identities == manifest["producer_identities"]
    assert manifest["records"][1460]["state"] == "failed"
    assert manifest["records"][1460]["validation_errors"] == []
    assert manifest["budget_evidence"]["bindings"] == {
        "source_tree_total_bytes": 6,
        "control_file_sha256": {"scope.json": "c" * 64},
        "c7_report_binding": {"source_batch_control_bytes": 1},
    }
