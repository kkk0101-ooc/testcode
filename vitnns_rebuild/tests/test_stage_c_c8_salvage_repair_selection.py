from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from vitnns import stage_c_c8_salvage_repair_selection as repair
from vitnns.stage_c_provenance import canonical_sha256


SHA = "a" * 64
COMMITS = {
    repair.ORIGINAL_SOURCE_ORIGIN: "1" * 40,
    repair.PRIOR_GAP_ORIGIN: "2" * 40,
    repair.REPAIR_ORIGIN: "3" * 40,
}
GENERATOR = {
    "commit": "4" * 40,
    "tree": "8" * 40,
    "source_subtree_path": "src/vitnns",
    "source_subtree_tree": "9" * 40,
    "dirty": False,
    "authorization_launch_commit": "0" * 40,
}


def _identity(origin):
    digit = {
        repair.ORIGINAL_SOURCE_ORIGIN: "5",
        repair.PRIOR_GAP_ORIGIN: "6",
        repair.REPAIR_ORIGIN: "7",
    }[origin]
    return {
        "commit": COMMITS[origin],
        "tree": digit * 40,
        "source_subtree_path": "src/vitnns",
        "source_subtree_tree": digit * 40,
        "dirty": False,
    }


def _authority(tmp_path=None):
    root = tmp_path
    return SimpleNamespace(
        canonical_digest="b" * 64,
        prior_authority_digest="c" * 64,
        selection_path=(None if root is None else root / "selection.json"),
        selection_report_path=(None if root is None else root / "report.json"),
    )


def _common_bindings(commit):
    return {
        "code": {"commit": commit, "dirty": False},
        "model": {"state_dict_sha256": "1" * 64},
        "dependencies": {"lock_sha256": "2" * 64},
        "registry": {"sha256": "3" * 64, "metadata_sha256": "4" * 64},
        "protocol": {"sha256": "5" * 64},
        "acceptance": {"sha256": "6" * 64},
        "configuration": {"sha256": "7" * 64, "schema_sha256": "8" * 64},
    }


def _raw_record(index, origin, *, state="succeeded"):
    identity = _identity(origin)
    sample_id = repair.base.CALIBRATION_SAMPLE_IDS[index % 5]
    method_id = repair.base.VALUE_METHOD_IDS[index % 3]
    block, head = divmod(index % 144, 12)
    hashes = {
        "manifest_sha256": SHA,
        "status_sha256": SHA,
        "provenance_sha256": SHA,
        "result_sha256": SHA,
        "unit_identity_sha256": f"{index:064x}",
    }
    return {
        "plan_index": index,
        "unit_key": {
            "canonical_digest": f"{index:064x}",
            "sample_id": sample_id,
            "method_id": method_id,
            "block_index": block,
            "head_index": head,
        },
        "origin": origin,
        "state": state,
        "execution_code_commit": identity["commit"],
        "execution_git_identity": identity,
        "hashes": hashes,
        "scalars": {},
        "provenance": {"bindings": _common_bindings(identity["commit"])},
        "resources": {
            "status": "completed",
            "calls": {"forward_calls": 1, "backward_calls": 1},
            "cuda": {"peak_reserved_bytes": 1},
        },
        "runner_telemetry": {
            "lifecycle": {
                "integrity": {
                    "matched": True, "restored": True,
                    "entry_sha256": SHA, "exit_sha256": SHA,
                },
                "restoration": {"restored": True, "changes_detected": False},
            },
        },
        "artifact_inventory": [{"level": "L0", "path": "L0/result.json", "bytes": 1}],
        "work_item_counts": {},
        "audit": False,
        "validation_errors": [],
        "attempt_path": f"/attempts/{index}",
        "l0_l1_cap_accounting_bytes": 1,
        "level_bytes": {"L0": 1, "L1": 0, "L2": 0, "L3": 0, "control": 0},
        "total_regular_file_bytes": 1,
    }


def _full_manifest(*, failed_index=0):
    records = []
    for index in range(2180):
        if index < 1460:
            origin = repair.ORIGINAL_SOURCE_ORIGIN
        elif index < 2029:
            origin = repair.PRIOR_GAP_ORIGIN
        else:
            origin = repair.REPAIR_ORIGIN
        records.append(_raw_record(
            index, origin,
            state="failed" if index == failed_index else "succeeded",
        ))
    return {
        "status": "PASS",
        "plan_digest": "d" * 64,
        "canonical_digest": "e" * 64,
        "combined_artifact_manifest_digest": "f" * 64,
        "origin_counts": {
            repair.ORIGINAL_SOURCE_ORIGIN: 1460,
            repair.PRIOR_GAP_ORIGIN: 569,
            repair.REPAIR_ORIGIN: 151,
        },
        "producer_identities": {
            origin: _identity(origin) for origin in repair.ORIGINS
        },
        "root_inventories": {
            origin: {"digest": str(index) * 64, "total_bytes": 10}
            for index, origin in enumerate(repair.ORIGINS, 1)
        },
        "verifier": {
            "commit": "9" * 40,
            "tree": "8" * 40,
            "source_subtree_path": "src/vitnns",
            "source_subtree_tree": "7" * 40,
            "dirty": False,
            "authorization_launch_commit": "0" * 40,
            "evidence_worktree_dirty": False,
            "verified_at_utc": "2026-09-03T00:00:00Z",
            "rules_version": "recovery-c8-salvage-repair-v1",
        },
        "budget_evidence": {
            "bindings": {
                "source_tree_total_bytes": 30,
                "control_file_sha256": {"control.json": SHA},
                "c7_report_binding": {"source_batch_control_bytes": 1},
            },
        },
        "records": records,
    }


def test_adapter_preserves_valid_terminal_failure_and_three_origins():
    records, identities, counts = repair._adapt_records(_full_manifest())
    assert records[0]["state"] == "failed"
    assert records[0]["trusted_by_batch_validator"] is True
    assert records[0]["validation_errors"] == []
    assert any(
        item.endswith("unit_not_succeeded")
        for item in repair.base._record_validity(records[0], "failed")
    )
    assert set(identities) == set(repair.ORIGINS)
    assert counts == {
        repair.ORIGINAL_SOURCE_ORIGIN: 1460,
        repair.PRIOR_GAP_ORIGIN: 569,
        repair.REPAIR_ORIGIN: 151,
    }


def _selection_records(*, failed=(11, 5)):
    result = []
    for index in range(2180):
        result.append({
            "state": (
                "failed" if index == 0 and failed is not None else "succeeded"
            ),
            "trusted_by_batch_validator": True,
            "validation_errors": [],
            "unit_key": {
                "canonical_digest": f"{index:064x}",
                "sample_id": repair.base.CALIBRATION_SAMPLE_IDS[index % 5],
                "method_id": repair.base.VALUE_METHOD_IDS[index % 3],
                "block_index": None if failed is None else failed[0],
                "head_index": None if failed is None else failed[1],
            },
            "provenance": {"bindings": _common_bindings(COMMITS[repair.REPAIR_ORIGIN])},
        })
    return result


def _candidate(records, block, head):
    failed = any(
        record["state"] == "failed"
        and record["unit_key"].get("block_index") == block
        and record["unit_key"].get("head_index") == head
        for record in records
    )
    reference = repair.base._candidate_ref(block, head)
    candidate = {
        **reference,
        "eligible": not failed,
        "validity_failures": ["terminal_failed"] if failed else [],
        "rounded_rank_tuple": (
            None if failed else [5, 1.0, 1.0, 1.0, -block, -head]
        ),
        "rank_within_depth_band": None,
        "selected_depth_band_winner": False,
        "selected_primary": False,
    }
    failures = [] if not failed else [{
        "candidate_key": reference["candidate_key"],
        "sample_id": repair.base.CALIBRATION_SAMPLE_IDS[0],
        "reason": "terminal_failed",
    }]
    return candidate, {**reference, "per_sample": []}, failures


@pytest.fixture
def derived(monkeypatch):
    authority = _authority()
    manifest = _full_manifest(failed_index=137)
    monkeypatch.setattr(repair.base, "_build_candidate", _candidate)
    monkeypatch.setattr(repair, "_global_class_failures", lambda _records: [])
    monkeypatch.setattr(
        repair, "_budget", lambda *_args: {
            "status": "PASS", "reason": None,
            "exact_unit_count": 2500, "projection_safety_factor": 1.25,
        },
    )
    monkeypatch.setattr(
        repair, "capture_c8_salvage_repair_verifier_identity",
        lambda *_args: dict(GENERATOR),
    )
    selected, report = repair._derive(
        authority, manifest, generator_git_state=GENERATOR,
    )
    return authority, manifest, selected, report


def test_per_head_terminal_failure_is_local_and_selection_passes(derived):
    _authority_value, _manifest, selected, report = derived
    assert selected is not None
    assert report["status"] == "PASS"
    failed = next(
        item for item in report["candidate_records"]
        if item["block_index"] == 11 and item["head_index"] == 5
    )
    assert failed["eligible"] is False
    assert report["evidence_summary"]["terminal_failed_unit_count"] == 1
    assert report["evidence_summary"]["trusted_terminal_unit_count"] == 2180
    assert len(report["depth_band_winners"]) == 4


def test_build_expands_simple_generator_to_full_captured_identity(
    derived, monkeypatch,
):
    authority, manifest, selected, report = derived
    calls = []
    monkeypatch.setattr(
        repair, "validate_c8_salvage_repair_from_source",
        lambda actual, plan, config: manifest,
    )

    def capture(actual, phase):
        calls.append((actual, phase))
        return dict(GENERATOR)

    monkeypatch.setattr(
        repair, "capture_c8_salvage_repair_verifier_identity", capture,
    )
    rebuilt = repair.build_c8_salvage_repair_selection(
        authority, object(), object(),
        generator_git_state={"commit": GENERATOR["commit"], "dirty": False},
    )
    assert rebuilt == (selected, report)
    assert calls == [(authority, "selection_assembler")]


def test_global_class_failure_blocks_without_selection(derived, monkeypatch):
    authority, manifest, _selected, _report = derived
    monkeypatch.setattr(repair, "_global_class_failures", lambda _records: [{
        "candidate_key": None, "sample_id": "sample", "reason": "global",
        "unit_key": {},
    }])
    selected, report = repair._derive(
        authority, manifest, generator_git_state=GENERATOR,
    )
    assert selected is None
    assert report["status"] == "BLOCKED"
    assert report["blocked_reasons"] == ["global_class_unit_failure"]
    assert report["selection_performed"] is False


def test_empty_depth_band_blocks_without_selection(derived, monkeypatch):
    authority, manifest, _selected, _report = derived

    def empty_band(records, block, head):
        candidate, audit, failures = _candidate(records, block, head)
        if block < 3:
            candidate["eligible"] = False
            candidate["rounded_rank_tuple"] = None
        return candidate, audit, failures

    monkeypatch.setattr(repair.base, "_build_candidate", empty_band)
    selected, report = repair._derive(
        authority, manifest, generator_git_state=GENERATOR,
    )
    assert selected is None
    assert "depth_band_0_has_no_eligible_candidate" in report["blocked_reasons"]


def test_budget_failure_blocks_without_selection(derived, monkeypatch):
    authority, manifest, _selected, _report = derived
    monkeypatch.setattr(
        repair, "_budget",
        lambda *_args: {"status": "BLOCKED", "reason": "post_c8_frozen_cap_failure"},
    )
    selected, report = repair._derive(
        authority, manifest, generator_git_state=GENERATOR,
    )
    assert selected is None
    assert "post_c8_frozen_cap_failure" in report["blocked_reasons"]


def test_validator_source_recomputes_and_rejects_self_resigned_tamper(
    derived, monkeypatch,
):
    authority, manifest, selected, report = derived
    calls = []
    monkeypatch.setattr(
        repair, "validate_c8_salvage_repair_from_source",
        lambda actual, plan, config: calls.append((actual, plan, config)) or manifest,
    )
    plan, config = object(), object()
    repair.validate_c8_salvage_repair_selection_report(
        authority, plan, config, report, selected,
        generator_git_state={"commit": GENERATOR["commit"], "dirty": False},
    )
    assert calls == [(authority, plan, config)]

    tampered = copy.deepcopy(report)
    tampered["candidate_records"][0]["eligible"] = False
    tampered["canonical_digest"] = canonical_sha256({
        key: value for key, value in tampered.items()
        if key != "canonical_digest"
    })
    with pytest.raises(
        repair.RecoveryC8SalvageRepairError, match="source recomputation",
    ):
        repair.validate_c8_salvage_repair_selection_report(
            authority, plan, config, tampered, selected,
            generator_git_state={"commit": GENERATOR["commit"], "dirty": False},
        )


def test_validator_reuses_persisted_full_assembler_identity(
    derived, monkeypatch,
):
    authority, manifest, selected, report = derived
    seen = []
    monkeypatch.setattr(
        repair, "validate_c8_salvage_repair_from_source", lambda *_args: manifest,
    )
    original = repair._derive

    def derive(*args, **kwargs):
        seen.append(kwargs["generator_git_state"])
        return original(*args, **kwargs)

    monkeypatch.setattr(repair, "_derive", derive)
    repair.validate_c8_salvage_repair_selection_report(
        authority, object(), object(), report, selected,
        generator_git_state={"commit": GENERATOR["commit"], "dirty": False},
    )
    assert seen == [report["assembler_identity"]]


def test_three_origin_or_nonterminal_drift_fails_closed():
    manifest = _full_manifest()
    manifest["origin_counts"].pop(repair.REPAIR_ORIGIN)
    with pytest.raises(
        repair.RecoveryC8SalvageRepairError, match="origin counts",
    ):
        repair._adapt_records(manifest)

    manifest = _full_manifest()
    manifest["records"][0]["state"] = "incomplete"
    with pytest.raises(
        repair.RecoveryC8SalvageRepairError, match="not terminal",
    ):
        repair._adapt_records(manifest)


def test_writer_is_idempotent_and_blocked_never_writes_selection(
    derived, monkeypatch, tmp_path,
):
    _authority_value, manifest, selected, report = derived
    authority = _authority(tmp_path)
    monkeypatch.setattr(
        repair, "validate_c8_salvage_repair_from_source", lambda *_args: manifest,
    )
    first = repair.write_c8_salvage_repair_selection_outputs(
        authority, object(), object(), selected, report,
        generator_git_state={"commit": GENERATOR["commit"], "dirty": False},
    )
    second = repair.write_c8_salvage_repair_selection_outputs(
        authority, object(), object(), selected, report,
        generator_git_state={"commit": GENERATOR["commit"], "dirty": False},
    )
    assert first == second
    assert first["status"] == "PASS"

    blocked_authority = _authority(tmp_path / "blocked")
    monkeypatch.setattr(repair, "_global_class_failures", lambda _records: [{
        "candidate_key": None, "sample_id": "sample", "reason": "global",
        "unit_key": {},
    }])
    blocked_selection, blocked_report = repair._derive(
        blocked_authority, manifest, generator_git_state=GENERATOR,
    )
    result = repair.write_c8_salvage_repair_selection_outputs(
        blocked_authority, object(), object(), blocked_selection, blocked_report,
        generator_git_state={"commit": GENERATOR["commit"], "dirty": False},
    )
    assert result["status"] == "BLOCKED"
    assert result["selection_sha256"] is None
    assert not blocked_authority.selection_path.exists()
