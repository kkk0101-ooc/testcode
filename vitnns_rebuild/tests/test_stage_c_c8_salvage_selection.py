from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

from vitnns import stage_c_c8_salvage_selection as selection
from vitnns.stage_c_c8_salvage import GAP_ORIGIN, SOURCE_ORIGIN
from vitnns.stage_c_provenance import canonical_sha256


SOURCE_COMMIT = "1" * 40
GAP_COMMIT = "2" * 40
LAUNCH_COMMIT = "3" * 40
GENERATOR_COMMIT = "4" * 40


def _identity(commit, tree, subtree):
    return {
        "commit": commit,
        "tree": tree * 40,
        "source_subtree_path": "src/vitnns",
        "source_subtree_tree": subtree * 40,
        "dirty": False,
    }


def _generator():
    return {
        **_identity(GENERATOR_COMMIT, "9", "8"),
        "verified_at_utc": "2026-09-03T00:00:00Z",
        "rules_version": "recovery-c8-salvage-v1",
    }


def _authority():
    return SimpleNamespace(
        canonical_digest="c" * 64,
        launch_commit=LAUNCH_COMMIT,
        document={
            "source_interrupted_root": {
                "execution_commit": SOURCE_COMMIT,
            },
        },
    )


def _origin_record(origin, identity):
    return {
        "origin": origin,
        "execution_code_commit": identity["commit"],
        "execution_git_identity": identity,
        "provenance": {
            "bindings": {
                "code": {"commit": identity["commit"], "dirty": False},
            },
        },
    }


def _manifest():
    inherited = _identity(SOURCE_COMMIT, "5", "6")
    gap = _identity(GAP_COMMIT, "7", "8")
    return {
        "canonical_digest": "d" * 64,
        "combined_artifact_manifest_digest": "e" * 64,
        "source_root_inventory": {"digest": "f" * 64, "total_bytes": 10},
        "gap_root_inventory": {"digest": "0" * 64, "total_bytes": 20},
        "origin_counts": {SOURCE_ORIGIN: 1, GAP_ORIGIN: 1},
        "producer_identities": {
            "interrupted_root_producer": _identity("a" * 40, "1", "2"),
            "salvage_authority_producer": _identity("b" * 40, "3", "4"),
            "interrupted_execution": inherited,
            "gap_execution": gap,
        },
        "records": [
            _origin_record(SOURCE_ORIGIN, inherited),
            _origin_record(GAP_ORIGIN, gap),
        ],
    }


def _scientific_records():
    common = {
        "model": {"state_dict_sha256": "a" * 64},
        "dependencies": {"lock_sha256": "b" * 64},
        "registry": {"sha256": "c" * 64, "metadata_sha256": "d" * 64},
        "protocol": {"sha256": "e" * 64},
        "acceptance": {"sha256": "f" * 64},
        "configuration": {"sha256": "0" * 64, "schema_sha256": "1" * 64},
    }
    records = []
    for sample_id in selection.base.CALIBRATION_SAMPLE_IDS:
        for method_id in selection.base.CLASS_METHOD_IDS:
            records.append({
                "unit_key": {
                    "canonical_digest": hashlib.sha256(
                        (sample_id + method_id).encode()
                    ).hexdigest(),
                    "sample_id": sample_id,
                    "method_id": method_id,
                },
                "state": "succeeded",
                "trusted_by_batch_validator": True,
                "validation_errors": [],
                "provenance": {"bindings": common},
            })
    return records


def _candidate(_records, block, head):
    reference = selection.base._candidate_ref(block, head)
    candidate = {
        **reference,
        "eligible": True,
        "rounded_rank_tuple": [5, 1.0, 1.0, 1.0, -block, -head],
        "median_delta_fid_fine4": 1.0,
        "rank_within_depth_band": None,
        "selected_depth_band_winner": False,
        "selected_primary": False,
    }
    return candidate, {**reference, "per_sample": []}, []


@pytest.fixture
def derived(monkeypatch):
    authority = _authority()
    manifest = _manifest()
    generator = _generator()
    monkeypatch.setattr(
        selection, "_validate_verifier_identity",
        lambda _authority, value: dict(value),
    )
    monkeypatch.setattr(
        selection, "_capture_verifier_identity", lambda _authority: generator,
    )
    monkeypatch.setattr(
        selection, "_scientific_records", lambda _manifest: _scientific_records(),
    )
    monkeypatch.setattr(selection.base, "_build_candidate", _candidate)
    monkeypatch.setattr(selection.base, "_record_validity", lambda *_args: [])
    monkeypatch.setattr(
        selection, "_post_c8_budget_from_source",
        lambda *_args: {
            "status": "PASS",
            "reason": None,
            "exact_unit_count": 2500,
            "projection_safety_factor": 1.25,
            "projection": {"all_hard_caps_pass": True},
        },
    )
    selected, report = selection._derive_c8_salvage_selection(
        authority, manifest, generator_git_state=generator,
    )
    return authority, manifest, generator, selected, report


def _resign(selected, report):
    selected["canonical_digest"] = canonical_sha256({
        key: value for key, value in selected.items()
        if key != "canonical_digest"
    })
    report["selection"] = copy.deepcopy(selected)
    report["selection_digest"] = selected["canonical_digest"]
    report["selection_file_sha256"] = hashlib.sha256(
        selection._pretty_json_bytes(selected)
    ).hexdigest()
    report["canonical_digest"] = canonical_sha256({
        key: value for key, value in report.items()
        if key != "canonical_digest"
    })


def test_derive_binds_real_identities_and_selection_hash(derived):
    authority, _manifest_value, generator, selected, report = derived
    bindings = selected["input_bindings"]
    assert bindings["retained_execution_commit"] == SOURCE_COMMIT
    assert bindings["gap_execution_commit"] == GAP_COMMIT
    assert bindings["gap_execution_commit"] != authority.launch_commit
    assert bindings["assembler_identity"] == generator
    assert bindings["verifier_identity"] == generator
    assert report["selection_generator_identity"] == generator
    assert report["selection_file_sha256"] == hashlib.sha256(
        selection._pretty_json_bytes(selected)
    ).hexdigest()


def test_derive_allows_real_generator_at_launch_commit(derived, monkeypatch):
    authority, manifest, generator, _selected, _report = derived
    launch_generator = {**generator, "commit": authority.launch_commit}
    selected, report = selection._derive_c8_salvage_selection(
        authority, manifest, generator_git_state=launch_generator,
    )
    assert selected["input_bindings"]["assembler_commit"] == LAUNCH_COMMIT
    assert report["verifier_identity"] == launch_generator


def test_validator_recomputes_source_and_accepts_exact_outputs(
    derived, monkeypatch,
):
    authority, manifest, generator, selected, report = derived
    calls = []
    monkeypatch.setattr(selection, "_authority", lambda value: value)
    monkeypatch.setattr(
        selection, "validate_c8_salvage_from_source",
        lambda actual_authority, plan, config: calls.append(
            (actual_authority, plan, config)
        ) or manifest,
    )
    plan, config = object(), object()
    selection.validate_c8_salvage_selection_report(
        authority, plan, config, report, selected,
        generator_git_state={"commit": GENERATOR_COMMIT, "dirty": False},
    )
    assert calls == [(authority, plan, config)]


def test_validator_reuses_persisted_verifier_identity(derived, monkeypatch):
    authority, manifest, generator, selected, report = derived
    seen = []
    monkeypatch.setattr(selection, "_authority", lambda value: value)
    monkeypatch.setattr(
        selection, "validate_c8_salvage_from_source", lambda *_args: manifest,
    )
    original = selection._derive_c8_salvage_selection

    def derive(*args, **kwargs):
        seen.append(kwargs["generator_git_state"])
        return original(*args, **kwargs)

    monkeypatch.setattr(selection, "_derive_c8_salvage_selection", derive)
    selection.validate_c8_salvage_selection_report(
        authority, object(), object(), report, selected,
        generator_git_state={"commit": GENERATOR_COMMIT, "dirty": False},
    )
    assert seen == [generator]


@pytest.mark.parametrize(
    "target, expected",
    [
        ("candidate", "candidate 0"),
        ("audit", "candidate audit 0"),
        ("binding", "input_bindings"),
        ("primary", "primary"),
    ],
)
def test_validator_rejects_self_resigned_semantic_tamper(
    derived, monkeypatch, target, expected,
):
    authority, manifest, generator, selected, report = derived
    selected = copy.deepcopy(selected)
    report = copy.deepcopy(report)
    if target == "candidate":
        selected["candidate_records"][0]["median_delta_fid_fine4"] = 2.0
    elif target == "audit":
        report["candidate_audit_records"][0]["per_sample"] = [{"x": 2.0}]
    elif target == "binding":
        selected["input_bindings"]["model_sha256"] = "9" * 64
        report["input_bindings"] = copy.deepcopy(selected["input_bindings"])
    else:
        selected["primary"] = copy.deepcopy(selected["depth_band_winners"][1])
    _resign(selected, report)
    monkeypatch.setattr(selection, "_authority", lambda value: value)
    monkeypatch.setattr(
        selection, "validate_c8_salvage_from_source", lambda *_args: manifest,
    )
    with pytest.raises(selection.RecoveryC8SalvageError, match=expected):
        selection.validate_c8_salvage_selection_report(
            authority, object(), object(), report, selected,
            generator_git_state={"commit": GENERATOR_COMMIT, "dirty": False},
        )


def test_derive_fails_closed_on_global_class_or_budget_failure(
    derived, monkeypatch,
):
    authority, manifest, generator, _selected, _report = derived
    monkeypatch.setattr(
        selection.base, "_record_validity", lambda *_args: ["bad"],
    )
    with pytest.raises(selection.RecoveryC8SalvageError, match="global class-unit"):
        selection._derive_c8_salvage_selection(
            authority, manifest, generator_git_state=generator,
        )


def test_selection_publication_is_idempotent_after_report_first_interruption(
    derived, monkeypatch, tmp_path,
):
    authority, _manifest_value, _generator, selected, report = derived
    authority.selection_report_path = tmp_path / "report.json"
    authority.selection_path = tmp_path / "selection.json"
    monkeypatch.setattr(selection, "_authority", lambda value: value)
    authority.selection_report_path.write_bytes(
        selection._pretty_json_bytes(report)
    )
    result = selection.write_c8_salvage_selection_outputs(
        authority, selected, report,
    )
    assert authority.selection_path.read_bytes() == selection._pretty_json_bytes(
        selected
    )
    assert result == selection.write_c8_salvage_selection_outputs(
        authority, selected, report,
    )


def test_selection_publication_rejects_existing_drift(
    derived, monkeypatch, tmp_path,
):
    authority, _manifest_value, _generator, selected, report = derived
    authority.selection_report_path = tmp_path / "report.json"
    authority.selection_path = tmp_path / "selection.json"
    authority.selection_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(selection, "_authority", lambda value: value)
    with pytest.raises(
        selection.RecoveryC8SalvageError, match="existing selection document",
    ):
        selection.write_c8_salvage_selection_outputs(
            authority, selected, report,
        )
