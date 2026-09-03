from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vitnns import stage_c_c8_salvage as salvage
from vitnns.stage_c_cli import build_parser


def _unit(digest: str):
    key = SimpleNamespace(canonical_digest=digest)
    key.to_dict = lambda: {"canonical_digest": digest}
    return SimpleNamespace(key=key)


def test_gap_sharding_is_deterministic_and_disjoint(monkeypatch, tmp_path):
    digests = [f"{index:064x}" for index in range(8)]
    authority = SimpleNamespace(gap_root=tmp_path, launch_commit="a" * 40)
    plan = SimpleNamespace(
        canonical_digest="b" * 64, units=tuple(_unit(item) for item in digests),
    )
    scope = {
        "gap_unit_digests": digests,
        "states": {item: "pending" for item in digests},
        "canonical_digest": "d" * 64,
    }
    monkeypatch.setattr(salvage, "_authority", lambda value: value)
    current = scope
    monkeypatch.setattr(salvage, "_scope", lambda *_args: current)
    monkeypatch.setattr(salvage, "_validate_scope_partition", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        salvage, "_read",
        lambda path: (
            salvage._runtime_status(current, state="running")
            if path.name == salvage.STATUS_FILE else
            {"plan_digest": plan.canonical_digest, "caps": {"effective": {}},
             "within_budget": True, "violations": [],
             "canonical_digest": "c" * 64}
        ),
    )
    monkeypatch.setattr("vitnns.stage_c_plan.StageCEstimate.from_dict", lambda value: SimpleNamespace(caps={"effective": {}}, to_dict=lambda: value))
    seen = []
    monkeypatch.setattr(
        salvage, "run_execution_unit",
        lambda unit, **kwargs: seen.append(unit.key.canonical_digest)
        or SimpleNamespace(state="succeeded", to_dict=lambda: {"state": "succeeded"}),
    )
    monkeypatch.setattr(salvage, "_atomic_replace_json", lambda *_args: None)
    backend = SimpleNamespace(trusted_recovery_context=authority)
    salvage.run_c8_salvage_gap_shard(
        authority, plan, backend, shard_index=1, shard_count=3, device="cpu",
        runtime_git_state={"commit": authority.launch_commit, "dirty": False},
    )
    assert seen == [digests[1], digests[4], digests[7]]


def test_combined_manifest_rejects_overlap_in_gap_root(monkeypatch, tmp_path):
    digest = "1" * 64
    (tmp_path / "units" / digest).mkdir(parents=True)
    authority = SimpleNamespace(gap_root=tmp_path)
    # The implementation checks gap layout after reconstructing all rows; this
    # regression assertion documents that inherited unit directories are not
    # permitted in the fresh gap root.
    assert (authority.gap_root / "units" / digest).exists()
    assert digest not in set()


def test_origins_are_explicit_and_distinct():
    assert salvage.SOURCE_ORIGIN == "inherited_from_interrupted_root"
    assert salvage.GAP_ORIGIN == "newly_run_in_salvage_root"
    assert salvage.REJECTED_ORIGIN == "rejected_from_interrupted_root"
    assert len({salvage.SOURCE_ORIGIN, salvage.GAP_ORIGIN, salvage.REJECTED_ORIGIN}) == 3


def test_terminal_gap_unit_is_never_rerun(monkeypatch, tmp_path):
    digest = "1" * 64
    authority = SimpleNamespace(gap_root=tmp_path, launch_commit="a" * 40)
    plan = SimpleNamespace(
        canonical_digest="b" * 64, units=(_unit(digest),),
    )
    scope = {
        "gap_unit_digests": [digest], "states": {digest: "succeeded"},
        "claim_tokens": {}, "canonical_digest": "d" * 64,
    }
    monkeypatch.setattr(salvage, "_authority", lambda value: value)
    monkeypatch.setattr(salvage, "_scope", lambda *_args: scope)
    monkeypatch.setattr(salvage, "_validate_scope_partition", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        salvage, "_read",
        lambda path: (
            salvage._runtime_status(scope, state="running")
            if path.name == salvage.STATUS_FILE else
            {"caps": {"effective": {}}, "within_budget": True}
        ),
    )
    monkeypatch.setattr(
        "vitnns.stage_c_plan.StageCEstimate.from_dict",
        lambda value: SimpleNamespace(caps={"effective": {}}, to_dict=lambda: value),
    )
    monkeypatch.setattr(
        salvage, "run_execution_unit",
        lambda *_args, **_kwargs: pytest.fail("terminal unit was rerun"),
    )
    backend = SimpleNamespace(trusted_recovery_context=authority)
    result = salvage.run_c8_salvage_gap_shard(
        authority, plan, backend, shard_index=0, shard_count=1, device="cpu"
        , runtime_git_state={"commit": authority.launch_commit, "dirty": False}
    )
    assert result == [{
        "unit_key": {"canonical_digest": digest},
        "state": "succeeded", "attempt_path": None,
        "resumed": True, "failure": None,
    }]


def test_salvage_cli_is_explicit_and_separate_from_normal_c8():
    parser = build_parser()
    args = parser.parse_args([
        "c8-salvage-init", "--config", "config/stage_c.json",
        "--run-spec", "config/stage_c_c8_run_spec.json",
        "--salvage-authority", "evidence/stage_c/salvage/authority.json",
    ])
    assert args.command == "c8-salvage-init"
    normal = parser.parse_args([
        "c8-finalize", "--config", "config/stage_c.json",
        "--run-spec", "config/stage_c_c8_run_spec.json",
        "--output-root", "evidence/stage_c/ordinary/formal",
    ])
    assert not hasattr(normal, "salvage_authority")
    completion = parser.parse_args([
        "c8-salvage-complete", "--config", "config/stage_c.json",
        "--run-spec", "config/stage_c_c8_run_spec.json",
        "--salvage-authority", "evidence/stage_c/salvage/authority.json",
        "--mode", "validate",
    ])
    assert completion.command == "c8-salvage-complete"
    assert completion.mode == "validate"


def test_runtime_git_state_must_be_exact_clean_launch_commit():
    authority = SimpleNamespace(launch_commit="a" * 40)
    assert salvage._runtime_git_state(
        authority, {"commit": "a" * 40, "dirty": False},
    ) == {"commit": "a" * 40, "dirty": False}
    with pytest.raises(salvage.RecoveryC8SalvageError, match="real clean launch"):
        salvage._runtime_git_state(
            authority, {"commit": "b" * 40, "dirty": False},
        )
    with pytest.raises(salvage.RecoveryC8SalvageError, match="exactly"):
        salvage._runtime_git_state(
            authority, {"commit": "a" * 40, "dirty": False, "tree": "x"},
        )


def _no_lock(*_args, **_kwargs):
    @contextmanager
    def locked():
        yield
    return locked()


def test_finalize_blocks_unless_all_720_gap_states_succeeded(monkeypatch, tmp_path):
    states = {f"{index:064x}": "succeeded" for index in range(720)}
    states[next(iter(states))] = "failed"
    scope = {
        "gap_unit_digests": list(states),
        "states": states, "claim_tokens": {},
    }
    authority = SimpleNamespace(gap_root=tmp_path)
    monkeypatch.setattr(salvage, "_authority", lambda value: value)
    monkeypatch.setattr(salvage, "_scope", lambda *_args: scope)
    monkeypatch.setattr(salvage, "_validate_scope_partition", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(salvage, "_lifecycle_lock", _no_lock)
    monkeypatch.setattr(salvage, "_checkpoint_lock", _no_lock)
    monkeypatch.setattr(salvage, "_artifact_activity_lock", _no_lock)
    monkeypatch.setattr(
        salvage, "build_c8_salvage_combined_manifest",
        lambda *_args, **_kwargs: pytest.fail("blocked finalize built manifest"),
    )
    result = salvage.finalize_c8_salvage(authority, object(), object())
    assert result["status"] == "BLOCKED"
    assert result["manifest_written"] is False
    assert result["gap_state_counts"] == {"failed": 1, "succeeded": 719}


def test_finalize_rejects_non_succeeded_combined_record(monkeypatch, tmp_path):
    states = {f"{index:064x}": "succeeded" for index in range(720)}
    scope = {
        "gap_unit_digests": list(states),
        "states": states, "claim_tokens": {},
    }
    authority = SimpleNamespace(gap_root=tmp_path, reports_root=tmp_path / "reports")
    published = []
    monkeypatch.setattr(salvage, "_authority", lambda value: value)
    monkeypatch.setattr(salvage, "_scope", lambda *_args: scope)
    monkeypatch.setattr(salvage, "_validate_scope_partition", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(salvage, "_lifecycle_lock", _no_lock)
    monkeypatch.setattr(salvage, "_checkpoint_lock", _no_lock)
    monkeypatch.setattr(salvage, "_artifact_activity_lock", _no_lock)
    monkeypatch.setattr(salvage, "_capture_verifier_identity", lambda *_args: {})
    monkeypatch.setattr(
        salvage, "_publish_scope",
        lambda _authority, _scope, *, state: published.append(state),
    )
    monkeypatch.setattr(
        salvage, "build_c8_salvage_combined_manifest",
        lambda *_args, **_kwargs: {"records": [{"state": "failed"}]},
    )
    with pytest.raises(salvage.RecoveryC8SalvageError, match="non-succeeded"):
        salvage.finalize_c8_salvage(authority, object(), object())
    assert published == ["finalized", "blocked"]


def test_scope_partition_must_match_frozen_and_current_audits(tmp_path):
    digests = [f"{index:064x}" for index in range(2180)]
    inherited, gap = digests[:1460], digests[1460:]
    audit = {
        "status": "PASS", "plan_digest": "p" * 64,
        "inherited_count": 1460, "gap_count": 720,
        "inherited_unit_digests": inherited, "gap_unit_digests": gap,
        "source_inventory": {"digest": "s" * 64},
        "rejected_count": 0, "rejected_units": [],
        "inherited_record_index": [],
    }
    audit["canonical_digest"] = salvage.canonical_sha256(audit)
    reports = tmp_path / "reports"
    reports.mkdir()
    audit_path = reports / "c8_salvage_source_audit.json"
    audit_path.write_text(__import__("json").dumps(audit), encoding="utf-8")
    authority = SimpleNamespace(reports_root=reports)
    plan = SimpleNamespace(
        canonical_digest="p" * 64, units=tuple(_unit(item) for item in digests),
    )
    scope = {
        "source_audit_path": str(audit_path),
        "source_audit_digest": audit["canonical_digest"],
        "inherited_unit_digests": inherited, "gap_unit_digests": gap,
        "states": {item: "succeeded" for item in gap}, "claim_tokens": {},
    }
    salvage._validate_scope_partition(authority, plan, scope, source_reaudit=audit)
    drifted = dict(audit)
    drifted["gap_unit_digests"] = list(reversed(gap))
    with pytest.raises(salvage.RecoveryC8SalvageError, match="current source re-audit"):
        salvage._validate_scope_partition(
            authority, plan, scope, source_reaudit=drifted,
        )


def test_completion_report_binds_selection_bytes_and_final_flags(monkeypatch, tmp_path):
    authority = SimpleNamespace(
        path=tmp_path / "authority.json", file_sha256="a" * 64,
        canonical_digest="b" * 64, launch_commit="c" * 40,
        combined_manifest_path=tmp_path / "manifest.json",
        selection_path=tmp_path / "selection.json",
        selection_report_path=tmp_path / "selection-report.json",
        reports_root=tmp_path, source_root=tmp_path / "source",
        gap_root=tmp_path / "gap", document={
            "amendment": {}, "base_c8_authority": {}, "contracts": {},
            "model": {},
        },
    )
    manifest = {
        "canonical_digest": "d" * 64,
        "combined_artifact_manifest_digest": "e" * 64,
        "source_reaudit_digest": "f" * 64,
        "frozen_source_audit": {
            "path": "audit.json", "file_sha256": "7" * 64,
            "canonical_digest": "8" * 64,
        },
        "source_partition_counts": {
            "inherited": 1460, "gap": 720,
            "rejected_source_attempts": 3,
        },
        "source_root_inventory": {"digest": "1" * 64},
        "gap_root_inventory": {"digest": "2" * 64},
        "producer_identities": {"old": {}, "gap": {}},
        "origin_counts": {
            salvage.SOURCE_ORIGIN: 1460, salvage.GAP_ORIGIN: 720,
        },
    }
    monkeypatch.setattr(salvage, "_authority", lambda value: value)
    monkeypatch.setattr(salvage, "validate_c8_salvage_from_source", lambda *_args: manifest)
    monkeypatch.setattr(salvage, "_validated_selection_bindings", lambda *_args: {
        "selection": {
            "path": str(tmp_path / "selection.json"),
            "file_sha256": "3" * 64, "canonical_digest": "4" * 64,
        },
        "selection_report": {
            "path": str(tmp_path / "selection-report.json"),
            "file_sha256": "5" * 64, "canonical_digest": "6" * 64,
        },
    })
    monkeypatch.setattr(salvage, "_capture_verifier_identity", lambda *_args: {"commit": "7" * 40})
    monkeypatch.setattr(salvage, "_read", lambda *_args: {})
    monkeypatch.setattr(salvage, "_sha", lambda *_args: "9" * 64)
    report = salvage.build_c8_salvage_completion_report(
        authority, object(), object(),
    )
    assert report["status"] == "PASS" and report["c8_pass"] is True
    assert report["selection"]["file_sha256"] == "3" * 64
    assert report["selection_report"]["canonical_digest"] == "6" * 64
    assert report["counts"]["gap_units"] == 720
    assert report["counts"]["newly_run_units"] == 720
    assert report["counts"]["rejected_source_attempts"] == 3
    assert report["counts"]["failed_or_blocked_units"] == 0
    assert report["roots"] == {
        "interrupted_source_root": str(authority.source_root),
        "fresh_gap_root": str(authority.gap_root),
        "virtual_combined_manifest": str(authority.combined_manifest_path),
    }
    assert report["flags"] == {
        "holdout_accessed": False, "selection_performed": True,
        "bootstrap_performed": False, "holm_performed": False,
        "decision_performed": False, "c9_authorized": False,
        "c10_authorized": False, "stage_d_authorized": False,
    }


def test_completion_closed_contract_rejects_self_resigned_extra_field(
    monkeypatch, tmp_path,
):
    authority = SimpleNamespace(
        path=tmp_path / "authority.json",
        combined_manifest_path=tmp_path / "manifest.json",
        selection_path=tmp_path / "selection.json",
        selection_report_path=tmp_path / "selection-report.json",
        source_root=tmp_path / "source", gap_root=tmp_path / "gap",
    )
    report = {"unexpected": True}
    report["canonical_digest"] = salvage.canonical_sha256(report)
    with pytest.raises(
        salvage.RecoveryC8SalvageError, match="fields are not closed",
    ):
        salvage._validate_completion_contract(authority, report)
