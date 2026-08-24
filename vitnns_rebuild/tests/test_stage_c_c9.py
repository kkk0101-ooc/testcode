from __future__ import annotations

import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from vitnns import stage_c_c9 as c9
from vitnns import stage_c_artifacts
from vitnns.stage_c_config import load_stage_c_config
from vitnns.stage_c_plan import (
    AUDIT_SAMPLE_IDS,
    CALIBRATION_SAMPLE_IDS,
    CLASS_METHOD_IDS,
    VALUE_METHOD_IDS,
    ExecutionPlan,
    ValueCandidate,
    build_execution_plan,
    estimate_stage_c_run,
    load_run_spec,
    load_stage_c_registry,
    load_stage_c_selection,
)
from vitnns.stage_c_provenance import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "config/stage_c_c9_run_spec.json"
LIVE_CODE = {"commit": "a" * 40, "dirty": False}
WINNERS = (
    ValueCandidate(0, 9),
    ValueCandidate(4, 2),
    ValueCandidate(6, 10),
    ValueCandidate(11, 2),
)
PRIMARY = WINNERS[0]
ALL_METHODS = (*VALUE_METHOD_IDS, *CLASS_METHOD_IDS)
EXPECTED_COUNTS = {
    "samples": 20,
    "audit_samples": 2,
    "execution_units": 320,
    "fidelity_eligible_units": 240,
    "structural_reference_units": 80,
    "work_items": 223_072,
    "map_runs": 1_312,
    "clean_map_runs": 320,
    "perturbation_map_runs": 960,
    "repeatability_map_runs": 32,
    "fidelity_curves": 10_560,
    "fidelity_curve_points": 221_760,
    "target_tie_curve_points": 100_800,
    "random_sanity_curve_points": 100_800,
    "center_sanity_curve_points": 20_160,
    "logical_scalar_evaluations": 221_760,
}
EXPECTED_SAMPLE_IDS = (
    "ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00002607.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00005279.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00007489.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00009888.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00012573.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00014787.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00018380.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00020970.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00023229.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00025927.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00028908.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00031099.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00033512.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00035680.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00038565.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00041511.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00044187.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00047079.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00049970.JPEG",
)
CALIBRATION_POSITIONS = (0, 4, 9, 14, 19)
AUDIT_POSITIONS = (0, 19)


@pytest.fixture(scope="module")
def exact():
    """Build the real C9 plan without validating or opening live JPEGs."""

    config = load_stage_c_config()
    registry = load_stage_c_registry(config, verify_live_assets=False)
    selection = load_stage_c_selection(config, required=True)
    run_spec = load_run_spec(SPEC)
    plan = build_execution_plan(
        config, registry, run_spec, selection=selection
    )
    return config, registry, selection, plan


@pytest.fixture(scope="module")
def formal(exact):
    config, registry, selection, plan = exact
    return SimpleNamespace(
        config=config, registry=registry, selection=selection, plan=plan,
        estimate=estimate_stage_c_run(config, plan),
    )


@pytest.fixture(scope="module")
def small(formal):
    plan = ExecutionPlan.create(
        run_spec=formal.plan.run_spec, registry=formal.plan.registry,
        selection=formal.plan.selection, units=formal.plan.units[:2],
    )
    return SimpleNamespace(
        config=formal.config, plan=plan,
        estimate=estimate_stage_c_run(formal.config, plan),
    )


def _resign(document):
    value = json.loads(json.dumps(document))
    value.pop("canonical_digest", None)
    value["canonical_digest"] = canonical_sha256(value)
    return value


def test_exact_c9_plan_has_frozen_scope_winners_primary_and_counts(exact):
    config, registry, selection, plan = exact

    result = c9.validate_c9_execution_plan(plan, config)

    expected_ids = list(EXPECTED_SAMPLE_IDS)
    assert result["valid"] is True
    assert result["counts"] == EXPECTED_COUNTS
    assert len(plan.units) == 20 * (4 * 3 + 4) == 320
    assert tuple(plan.run_spec.sample_ids) == tuple(expected_ids)
    assert tuple(row.sample_id for row in registry.rows) == EXPECTED_SAMPLE_IDS
    assert selection.depth_band_winners == WINNERS
    assert selection.primary == PRIMARY
    assert plan.selection.depth_band_winners == WINNERS
    assert plan.selection.primary == PRIMARY
    assert plan.run_spec.value_config_source == "selection_band_winners"
    assert plan.run_spec.value_candidates == ()
    assert plan.run_spec.fidelity_pairing_context == PRIMARY

    samples = {unit.sample.sample_id: unit.sample for unit in plan.units}
    assert list(samples) == expected_ids
    assert Counter(item.role for item in samples.values()) == {
        "calibration": 5,
        "holdout": 15,
    }
    assert {item.sample_id for item in samples.values() if item.audit} == set(
        AUDIT_SAMPLE_IDS
    )
    assert {item.sample_id for item in samples.values() if item.role == "calibration"} == set(
        CALIBRATION_SAMPLE_IDS
    )
    assert Counter(unit.sample.sample_id for unit in plan.units) == Counter(
        {sample_id: 16 for sample_id in expected_ids}
    )
    assert tuple(
        index for index, item in enumerate(samples.values())
        if item.role == "calibration"
    ) == CALIBRATION_POSITIONS
    assert tuple(
        index for index, item in enumerate(samples.values()) if item.audit
    ) == AUDIT_POSITIONS


def test_exact_c9_plan_does_not_multiply_global_methods_by_winners(exact):
    _config, registry, _selection, plan = exact

    value_units = [unit for unit in plan.units if unit.comparison_line == "value_path"]
    global_units = [unit for unit in plan.units if unit.comparison_line == "class_logit"]
    assert len(value_units) == 20 * 4 * 3 == 240
    assert len(global_units) == 20 * 4 == 80
    assert {(unit.key.block_index, unit.key.head_index) for unit in value_units} == {
        (0, 9), (4, 2), (6, 10), (11, 2)
    }
    assert all(
        unit.key.block_index is None and unit.key.head_index is None
        for unit in global_units
    )
    assert Counter(
        (unit.sample.sample_id, unit.key.method_id) for unit in global_units
    ) == Counter(
        (row.sample_id, method_id)
        for row in registry.rows
        for method_id in CLASS_METHOD_IDS
    )
    assert {
        (item.pairing_block_index, item.pairing_head_index)
        for unit in global_units
        for item in unit.work_items
        if item.kind == "fidelity_curve_point"
    } == {(0, 9)}


def test_c9_plan_validator_rejects_order_and_scope_drift(exact):
    config, _registry, _selection, plan = exact
    document = plan.to_dict()
    document["units"][0], document["units"][1] = (
        document["units"][1],
        document["units"][0],
    )
    altered = ExecutionPlan.from_dict(_resign(document))

    result = c9.validate_c9_execution_plan(altered, config)

    assert result["valid"] is False
    assert any("order" in item.lower() for item in result["violations"])


def test_pending_failure_table_covers_all_320_units_in_plan_order(exact):
    _config, _registry, _selection, plan = exact

    rows = c9._pending_rows(plan)

    assert len(rows) == 320
    assert [row["unit_key"] for row in rows] == [
        unit.key.to_dict() for unit in plan.units
    ]
    assert all(
        row["state"] == "pending"
        and row["terminal"] is False
        and row["complete"] is False
        and row["claim_token"] is None
        and row["selected_attempt"] is None
        for row in rows
    )


def _synthetic_handoff():
    value = {
        "schema_version": 1,
        "document_type": "stage_c_c8_to_c9_handoff",
        "status": "PASS",
        "integrity_valid": True,
        "c8_root": "/synthetic/c8",
        "c8_report_path": "/synthetic/c8-report.json",
        "c8_report_file_sha256": c9.C8_REPORT_FILE_SHA256,
        "c8_report_canonical_digest": c9.C8_REPORT_CANONICAL_DIGEST,
        "c8_recovery_selection_path": "/synthetic/recovery-selection.json",
        "c8_recovery_selection_file_sha256": c9.C8_SELECTION_FILE_SHA256,
        "tracked_selection_path": str(
            (ROOT / "selection/stage_c_selection.json").absolute()
        ),
        "tracked_selection_file_sha256": c9.C8_SELECTION_FILE_SHA256,
        "selection_canonical_digest": c9.C8_SELECTION_CANONICAL_DIGEST,
        "c8_producer_code": {
            "commit": c9.C8_PRODUCER_COMMIT, "dirty": False,
        },
        "c8_generator_code": {
            "commit": c9.C8_GENERATOR_COMMIT, "dirty": False,
        },
        "c8_completion_commit": c9.C8_COMPLETION_COMMIT,
        "c8_control_file_sha256": dict(c9.C8_CONTROL_FILE_SHA256),
        "c8_execution_plan_sha256": c9.C8_EXECUTION_PLAN_SHA256,
        "c8_run_spec_sha256": c9.C8_RUN_SPEC_SHA256,
        "c8_preflight_sha256": c9.C8_PREFLIGHT_SHA256,
        "c8_calibration_artifact_manifest_sha256": (
            c9.C8_CALIBRATION_ARTIFACT_MANIFEST_SHA256
        ),
        "c8_calibration_evidence_sha256": (
            c9.C8_CALIBRATION_EVIDENCE_SHA256
        ),
        "c8_source_tree_total_bytes": c9.C8_SOURCE_TREE_TOTAL_BYTES,
        "c8_retained_artifact_bytes": c9.C8_SOURCE_TREE_TOTAL_BYTES,
        "c9_projected_artifact_bytes": c9.C9_PROJECTED_ARTIFACT_BYTES,
        "c8_c9_projected_artifact_bytes": c9.C8_C9_PROJECTED_ARTIFACT_BYTES,
        "c9_projected_h20_gpu_hours": c9.C9_PROJECTED_H20_GPU_HOURS,
        "c8_c9_projected_h20_gpu_hours": c9.C8_C9_PROJECTED_H20_GPU_HOURS,
    }
    value["canonical_digest"] = canonical_sha256(value)
    return value


def _synthetic_authorization(root, plan, estimate):
    handoff = _synthetic_handoff()
    value = {
        "schema_version": 1,
        "document_type": "stage_c_c9_launch_authorization",
        "status": "PASS",
        "holdout_accessed": False,
        "selection_performed": False,
        "bootstrap_performed": False,
        "holm_performed": False,
        "decision_performed": False,
        "c8_handoff": handoff,
        "c8_handoff_sha256": handoff["canonical_digest"],
        "c9_execution_plan_sha256": plan.canonical_digest,
        "c9_run_spec_sha256": canonical_sha256(plan.run_spec.to_dict()),
        "c9_preflight_sha256": estimate.canonical_digest,
        "c9_output_root": str(Path(root).absolute()),
        "code": dict(LIVE_CODE),
    }
    value["canonical_digest"] = canonical_sha256(value)
    return value


def _install_small_gates(monkeypatch, authorization):
    monkeypatch.setattr(
        c9, "validate_c9_execution_plan",
        lambda *_args: {"valid": True, "violations": []},
    )
    monkeypatch.setattr(
        c9, "_validate_c9_preflight", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        c9, "_require_live_git_state",
        lambda *_args, **_kwargs: dict(LIVE_CODE),
    )
    monkeypatch.setattr(
        c9, "_validate_commit_lineage", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        c9, "_revalidate_handoff_from_authorization",
        lambda *_args, **_kwargs: authorization["c8_handoff"],
    )


def _initialize_small(tmp_path, small, monkeypatch):
    root = tmp_path / "c9"
    authorization = _synthetic_authorization(
        root, small.plan, small.estimate
    )
    _install_small_gates(monkeypatch, authorization)
    c9.initialize_c9_output_root(
        root, small.plan, small.estimate, authorization,
        config=small.config, observed_git_state=LIVE_CODE, repo_root=ROOT,
    )
    return SimpleNamespace(
        root=root, plan=small.plan, estimate=small.estimate,
        authorization=authorization, config=small.config,
    )


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _make_attempt(root, digest):
    attempt = (
        Path(root) / "units" / digest / "attempts" / "attempt-000001"
    ).absolute()
    attempt.mkdir(parents=True)
    (attempt / "status.json").write_text("{}\n", encoding="utf-8")
    return attempt


def _valid_attempt():
    return SimpleNamespace(
        valid=True, status="complete", complete=True, terminal=True,
        resumable=True, errors=(), l0_l1_bytes=0, l2_bytes=0,
        total_bytes=0,
    )


def test_launch_authorization_rejects_re_signed_c8_handoff_tamper(exact):
    config, _registry, _selection, plan = exact
    estimate = estimate_stage_c_run(config, plan)
    root = ROOT / "never-created-c9-test-root"
    authorization = _synthetic_authorization(root, plan, estimate)
    tampered = json.loads(json.dumps(authorization))
    tampered["c8_handoff"]["c8_generator_code"]["commit"] = "0" * 40
    tampered["c8_handoff"] = _resign(tampered["c8_handoff"])
    tampered["c8_handoff_sha256"] = tampered["c8_handoff"]["canonical_digest"]
    tampered = _resign(tampered)

    with pytest.raises(c9.StageCBenchmarkIntegrityError):
        c9.validate_c9_launch_authorization(
            tampered, plan=plan, estimate=estimate, output_root=root
        )


def test_build_launch_authorization_binds_handoff_preflight_and_fresh_root(
    exact, tmp_path, monkeypatch
):
    config, _registry, _selection, plan = exact
    handoff = _synthetic_handoff()
    root = tmp_path / "new-c9-root"
    monkeypatch.setattr(
        c9, "validate_c8_handoff", lambda **_kwargs: handoff
    )
    monkeypatch.setattr(
        c9, "_require_live_git_state",
        lambda *_args, **_kwargs: dict(LIVE_CODE),
    )
    monkeypatch.setattr(c9, "_validate_commit_lineage", lambda **_kwargs: None)

    authorization, estimate = c9.build_c9_launch_authorization(
        plan, config, c8_root="/synthetic/c8",
        c8_report_path="/synthetic/c8-report.json",
        recovery_selection_path="/synthetic/recovery-selection.json",
        output_root=root, observed_git_state=LIVE_CODE, repo_root=ROOT,
    )

    assert authorization["c8_handoff"] == handoff
    assert authorization["c8_handoff_sha256"] == handoff["canonical_digest"]
    assert authorization["c9_execution_plan_sha256"] == plan.canonical_digest
    assert authorization["c9_preflight_sha256"] == estimate.canonical_digest
    assert authorization["c9_output_root"] == str(root.absolute())
    assert authorization["code"] == LIVE_CODE
    assert estimate.counts == EXPECTED_COUNTS
    assert estimate.compute["projected_h20_gpu_hours"] == pytest.approx(
        21.893448287103737
    )
    assert not root.exists()
    c9.validate_c9_launch_authorization(
        authorization, plan=plan, estimate=estimate, output_root=root
    )


def test_launch_authorization_rejects_any_c10_flag(exact):
    config, _registry, _selection, plan = exact
    estimate = estimate_stage_c_run(config, plan)
    root = ROOT / "never-created-c9-test-root"

    for flag in ("selection_performed", "bootstrap_performed", "holm_performed", "decision_performed"):
        authorization = _synthetic_authorization(root, plan, estimate)
        authorization[flag] = True
        authorization = _resign(authorization)
        with pytest.raises(c9.StageCBenchmarkIntegrityError):
            c9.validate_c9_launch_authorization(
                authorization, plan=plan, estimate=estimate, output_root=root
            )


def test_c8_handoff_document_accepts_only_frozen_evidence_bindings():
    handoff = _synthetic_handoff()
    c9._validate_handoff_document(handoff)

    assert handoff["status"] == "PASS"
    assert handoff["integrity_valid"] is True
    assert handoff["c8_report_file_sha256"] == c9.C8_REPORT_FILE_SHA256
    assert handoff["c8_report_canonical_digest"] == c9.C8_REPORT_CANONICAL_DIGEST
    assert handoff["tracked_selection_file_sha256"] == c9.C8_SELECTION_FILE_SHA256
    assert handoff["c8_recovery_selection_file_sha256"] == c9.C8_SELECTION_FILE_SHA256
    assert handoff["c8_producer_code"] == {
        "commit": c9.C8_PRODUCER_COMMIT, "dirty": False
    }
    assert handoff["c8_generator_code"] == {
        "commit": c9.C8_GENERATOR_COMMIT, "dirty": False
    }
    assert handoff["c9_projected_artifact_bytes"] == 440_416_200
    assert handoff["c8_c9_projected_artifact_bytes"] == 3_967_765_325
    assert handoff["c9_projected_h20_gpu_hours"] == pytest.approx(
        21.893448287103737
    )
    assert handoff["c8_c9_projected_h20_gpu_hours"] == pytest.approx(
        171.04256474299797
    )
    assert handoff["canonical_digest"] == canonical_sha256({
        key: value for key, value in handoff.items()
        if key != "canonical_digest"
    })


def test_full_c8_source_recomputation_is_required_only_for_authorization(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(c9, "_validate_c8_source_controls", lambda _root: None)
    monkeypatch.setattr(c9, "_validate_commit_lineage", lambda **_kwargs: None)
    monkeypatch.setattr(
        c9, "measure_artifact_tree_bytes",
        lambda _root: SimpleNamespace(
            total_bytes=c9.C8_SOURCE_TREE_TOTAL_BYTES
        ),
    )
    monkeypatch.setattr(
        c9, "_validate_c8_source_evidence",
        lambda *args: calls.append(args),
    )

    handoff = c9.validate_c8_handoff(
        c8_root=c9.C8_SOURCE_ROOT,
        c8_report_path=c9.C8_REPORT_PATH,
        recovery_selection_path=c9.C8_RECOVERY_SELECTION_PATH,
        tracked_selection_path=ROOT / "selection/stage_c_selection.json",
        repo_root=ROOT,
    )

    assert handoff["status"] == "PASS"
    assert len(calls) == 1
    assert calls[0][0] == c9.C8_SOURCE_ROOT
    assert calls[0][1] == c9.C8_REPORT_PATH
    assert calls[0][2] == ROOT / "selection/stage_c_selection.json"


def test_c8_handoff_rejects_tampered_report_before_semantics(tmp_path):
    report = tmp_path / "c8-report.json"
    recovery = tmp_path / "recovery-selection.json"
    tracked = tmp_path / "tracked-selection.json"
    report.write_text("{}\n", encoding="utf-8")
    recovery.write_text("{}\n", encoding="utf-8")
    tracked.write_text("{}\n", encoding="utf-8")

    with pytest.raises(c9.StageCBenchmarkIntegrityError, match="SHA-256|hash"):
        c9.validate_c8_handoff(
            c8_root=tmp_path / "c8",
            c8_report_path=report,
            recovery_selection_path=recovery,
            tracked_selection_path=tracked,
            repo_root=ROOT,
        )


def test_c9_launch_rehashes_every_frozen_input(exact, monkeypatch):
    config, _registry, _selection, _plan = exact
    calls = []
    by_path = {
        c9.DEFAULT_STATE_DICT: c9.FROZEN_INPUT_SHA256["model_sha256"],
        c9.REPO_ROOT / config.registry.path:
            c9.FROZEN_INPUT_SHA256["registry_sha256"],
        c9.REPO_ROOT / config.registry.metadata_path:
            c9.FROZEN_INPUT_SHA256["registry_metadata_sha256"],
        c9.DEFAULT_STAGE_C_CONFIG:
            c9.FROZEN_INPUT_SHA256["config_sha256"],
        c9.DEFAULT_STAGE_C_SCHEMA:
            c9.FROZEN_INPUT_SHA256["config_schema_sha256"],
        c9.DEFAULT_REGISTRY_SCHEMA:
            c9.FROZEN_INPUT_SHA256["registry_schema_sha256"],
        c9.DEFAULT_SELECTION_SCHEMA:
            c9.FROZEN_INPUT_SHA256["selection_schema_sha256"],
        c9.REPO_ROOT / config.protocol.path:
            c9.FROZEN_INPUT_SHA256["protocol_sha256"],
        c9.DEFAULT_ACCEPTANCE:
            c9.FROZEN_INPUT_SHA256["acceptance_sha256"],
        c9.DEFAULT_DEPENDENCY_LOCK:
            c9.FROZEN_INPUT_SHA256["dependency_lock_sha256"],
        c9.REPO_ROOT / config.registry.source_manifest_path:
            c9.FROZEN_INPUT_SHA256["stage_a_manifest_sha256"],
    }

    def digest(path):
        path = Path(path)
        calls.append(path)
        return by_path[path]

    monkeypatch.setattr(c9, "sha256_file", digest)
    c9._validate_frozen_input_hashes(
        config=config, tracked_selection_path=ROOT / "selection/stage_c_selection.json"
    )
    assert set(calls) == set(by_path)

    by_path[c9.DEFAULT_STAGE_C_SCHEMA] = "0" * 64
    with pytest.raises(c9.StageCBenchmarkIntegrityError, match="config_schema"):
        c9._validate_frozen_input_hashes(
            config=config,
            tracked_selection_path=ROOT / "selection/stage_c_selection.json",
        )


def test_initialize_requires_a_fresh_root_before_writing_controls(
    exact, tmp_path, monkeypatch
):
    config, _registry, _selection, plan = exact
    estimate = estimate_stage_c_run(config, plan)
    root = tmp_path / "c9"
    root.mkdir()
    sentinel = root / "belongs-to-user.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    authorization = _synthetic_authorization(root, plan, estimate)
    monkeypatch.setattr(
        c9, "validate_c9_execution_plan",
        lambda *_args: {"valid": True, "violations": []},
    )
    monkeypatch.setattr(c9, "validate_c9_launch_authorization", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(c9, "_validate_c9_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(c9, "_require_live_git_state", lambda *_args, **_kwargs: dict(LIVE_CODE))
    monkeypatch.setattr(c9, "_validate_commit_lineage", lambda **_kwargs: None)
    monkeypatch.setattr(
        c9, "_revalidate_handoff_from_authorization",
        lambda *_args, **_kwargs: authorization["c8_handoff"],
    )

    with pytest.raises(c9.StageCBenchmarkBlockedError, match="must not exist"):
        c9.initialize_c9_output_root(
            root, plan, estimate, authorization, config=config,
            observed_git_state=LIVE_CODE, repo_root=ROOT,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert list(root.iterdir()) == [sentinel]


def test_initialize_atomically_rejects_root_claimed_during_c8_revalidation(
    exact, tmp_path, monkeypatch
):
    config, _registry, _selection, plan = exact
    estimate = estimate_stage_c_run(config, plan)
    root = tmp_path / "c9-raced"
    authorization = _synthetic_authorization(root, plan, estimate)
    monkeypatch.setattr(
        c9, "validate_c9_execution_plan",
        lambda *_args: {"valid": True, "violations": []},
    )
    monkeypatch.setattr(
        c9, "validate_c9_launch_authorization",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        c9, "_validate_c9_preflight", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        c9, "_require_live_git_state",
        lambda *_args, **_kwargs: dict(LIVE_CODE),
    )
    monkeypatch.setattr(
        c9, "_validate_commit_lineage", lambda **_kwargs: None
    )

    def claim_root(*_args, **_kwargs):
        root.mkdir()
        (root / "competing-owner.txt").write_text(
            "keep\n", encoding="utf-8"
        )
        return authorization["c8_handoff"]

    monkeypatch.setattr(
        c9, "_revalidate_handoff_from_authorization", claim_root
    )

    with pytest.raises(
        c9.StageCBenchmarkBlockedError, match="claimed during"
    ):
        c9.initialize_c9_output_root(
            root, plan, estimate, authorization, config=config,
            observed_git_state=LIVE_CODE, repo_root=ROOT,
        )

    assert [path.name for path in root.iterdir()] == ["competing-owner.txt"]
    assert (root / "competing-owner.txt").read_text(encoding="utf-8") == "keep\n"


def test_small_initialize_writes_one_pending_row_per_unit(
    small, tmp_path, monkeypatch
):
    initialized = _initialize_small(tmp_path, small, monkeypatch)

    table = _read(initialized.root / "failure_table.json")
    status = _read(initialized.root / "batch_status.json")
    assert len(table["rows"]) == len(initialized.plan.units) == 2
    assert [row["unit_key"] for row in table["rows"]] == [
        unit.key.to_dict() for unit in initialized.plan.units
    ]
    assert all(row["state"] == "pending" for row in table["rows"])
    assert status["expected_unit_count"] == 2
    assert status["lifecycle_state"] == "running"
    assert status["active_claim_count"] == 0
    assert status["orchestration_complete"] is False


def test_shard_partition_is_exact_disjoint_and_rejects_empty_shards(
    exact,
):
    _config, _registry, _selection, plan = exact
    shards = [
        c9._shard_units(plan, shard_index=index, shard_count=7)
        for index in range(7)
    ]
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert sum(map(len, shards)) == 320
    assert len({unit.key.canonical_digest for shard in shards for unit in shard}) == 320
    rebuilt = [None] * 320
    for shard_index, shard in enumerate(shards):
        for offset, unit in enumerate(shard):
            rebuilt[shard_index + offset * 7] = unit
    assert rebuilt == list(plan.units)

    for index, count in ((-1, 2), (2, 2), (0, 0), (0, 321)):
        with pytest.raises(ValueError, match="non-empty zero-based shard"):
            c9._shard_units(plan, shard_index=index, shard_count=count)


def test_small_shard_passes_through_runner_without_resume(
    small, tmp_path, monkeypatch
):
    initialized = _initialize_small(tmp_path, small, monkeypatch)
    calls = []
    monkeypatch.setattr(
        stage_c_artifacts, "validate_execution_unit",
        lambda *_args, **_kwargs: _valid_attempt(),
    )
    monkeypatch.setattr(
        c9, "load_c9_initialized_state",
        lambda *_args, **_kwargs: (
            initialized.estimate, initialized.authorization
        ),
    )
    monkeypatch.setattr(
        c9, "_combined_artifact_errors",
        lambda *_args, **_kwargs: [],
    )

    def run(unit, **kwargs):
        digest = unit.key.canonical_digest
        attempt = _make_attempt(initialized.root, digest)
        calls.append((unit, kwargs))
        return SimpleNamespace(
            state="succeeded", attempt_path=str(attempt), failure=None
        )

    monkeypatch.setattr(c9, "run_execution_unit", run)
    result = c9.run_c9_shard(
        initialized.plan, output_root=initialized.root, backend=object(),
        estimate=initialized.estimate, shard_index=0, shard_count=2,
        observed_git_state=LIVE_CODE,
    )

    assert len(result) == 1
    assert calls[0][0] is initialized.plan.units[0]
    assert calls[0][1]["resume"] is False
    assert calls[0][1]["preflight"] == initialized.estimate.to_dict()
    assert calls[0][1]["preflight"] is not initialized.estimate
    assert calls[0][1]["observed_git_state"] == LIVE_CODE
    rows = _read(initialized.root / "failure_table.json")["rows"]
    assert rows[0]["state"] == "succeeded"
    assert len(rows[0]["claim_token"]) == 64
    assert rows[1]["state"] == "pending"


def test_shard_stops_immediately_when_runner_raises_and_marks_incomplete(
    small, tmp_path, monkeypatch
):
    initialized = _initialize_small(tmp_path, small, monkeypatch)
    monkeypatch.setattr(
        c9, "load_c9_initialized_state",
        lambda *_args, **_kwargs: (
            initialized.estimate, initialized.authorization
        ),
    )
    monkeypatch.setattr(
        c9, "_combined_artifact_errors", lambda *_args, **_kwargs: []
    )
    calls = []

    def fail(unit, **_kwargs):
        calls.append(unit)
        raise RuntimeError("synthetic crash")

    monkeypatch.setattr(c9, "run_execution_unit", fail)
    with pytest.raises(c9.StageCBenchmarkBlockedError, match="new empty root"):
        c9.run_c9_shard(
            initialized.plan, output_root=initialized.root, backend=object(),
            estimate=initialized.estimate, shard_index=0, shard_count=1,
            observed_git_state=LIVE_CODE,
        )

    assert len(calls) == 1
    rows = _read(initialized.root / "failure_table.json")["rows"]
    assert rows[0]["state"] == "incomplete"
    assert rows[1]["state"] == "pending"
    assert _read(initialized.root / "batch_status.json")["lifecycle_state"] == "blocked"


def test_small_shard_rejects_checkpoint_tamper_before_backend(
    small, tmp_path, monkeypatch
):
    initialized = _initialize_small(tmp_path, small, monkeypatch)
    table = _read(initialized.root / "failure_table.json")
    table["plan_digest"] = "0" * 64
    c9._c8._atomic_replace_json(initialized.root / "failure_table.json", table)
    calls = []
    monkeypatch.setattr(
        c9, "load_c9_initialized_state",
        lambda *_args, **_kwargs: (
            initialized.estimate, initialized.authorization
        ),
    )
    monkeypatch.setattr(
        c9, "run_execution_unit",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(c9.StageCBenchmarkIntegrityError, match="failure table"):
        c9.run_c9_shard(
            initialized.plan, output_root=initialized.root, backend=object(),
            estimate=initialized.estimate, shard_index=0, shard_count=2,
            observed_git_state=LIVE_CODE,
        )

    assert calls == []


@pytest.mark.parametrize("state", ["incomplete", "tampered"])
def test_any_unsuccessful_attempt_blocks_the_root_before_another_shard(
    small, tmp_path, monkeypatch, state
):
    initialized = _initialize_small(tmp_path, small, monkeypatch)
    unit = initialized.plan.units[0]
    token = c9._claim_c9_unit(initialized.root, initialized.plan, unit)
    attempt = _make_attempt(initialized.root, unit.key.canonical_digest)
    row = {
        **c9._pending_rows(initialized.plan)[0],
        "state": state, "terminal": state == "failed",
        "complete": False, "claim_token": token,
        "selected_attempt": str(attempt),
        "selected_status_sha256": "b" * 64,
        "failure": {"reason": "synthetic"},
        "l0_l1_bytes": 0, "l2_bytes": 0, "total_bytes": 0,
    }
    monkeypatch.setattr(
        c9, "_combined_artifact_errors", lambda *_args, **_kwargs: []
    )

    c9._publish_checkpoint(
        initialized.root, initialized.plan, row,
        caps=initialized.estimate.caps["effective"], claim_token=token,
        handoff=initialized.authorization["c8_handoff"],
    )

    assert _read(initialized.root / "batch_status.json")["lifecycle_state"] == "blocked"
    calls = []
    monkeypatch.setattr(
        c9, "run_execution_unit",
        lambda *_args, **_kwargs: calls.append(True),
    )
    with pytest.raises(c9.StageCBenchmarkIntegrityError, match="lifecycle"):
        c9.run_c9_shard(
            initialized.plan, output_root=initialized.root, backend=object(),
            estimate=initialized.estimate, shard_index=1, shard_count=2,
            observed_git_state=LIVE_CODE,
        )
    assert calls == []


def test_terminal_scientific_failure_keeps_root_running_for_other_units(
    small, tmp_path, monkeypatch
):
    initialized = _initialize_small(tmp_path, small, monkeypatch)
    failed_unit = initialized.plan.units[0]
    token = c9._claim_c9_unit(
        initialized.root, initialized.plan, failed_unit
    )
    failed_attempt = _make_attempt(
        initialized.root, failed_unit.key.canonical_digest
    )
    failed_row = {
        **c9._pending_rows(initialized.plan)[0],
        "state": "failed", "terminal": True, "complete": False,
        "claim_token": token, "selected_attempt": str(failed_attempt),
        "selected_status_sha256": "b" * 64,
        "failure": {"reason": "scientific_method_failure"},
        "l0_l1_bytes": 0, "l2_bytes": 0, "total_bytes": 0,
    }
    monkeypatch.setattr(
        c9, "_combined_artifact_errors", lambda *_args, **_kwargs: []
    )
    c9._publish_checkpoint(
        initialized.root, initialized.plan, failed_row,
        caps=initialized.estimate.caps["effective"], claim_token=token,
        handoff=initialized.authorization["c8_handoff"],
    )
    assert _read(
        initialized.root / "batch_status.json"
    )["lifecycle_state"] == "running"

    monkeypatch.setattr(
        c9, "load_c9_initialized_state",
        lambda *_args, **_kwargs: (
            initialized.estimate, initialized.authorization
        ),
    )
    monkeypatch.setattr(
        stage_c_artifacts, "validate_execution_unit",
        lambda *_args, **_kwargs: _valid_attempt(),
    )

    def succeed(unit, **_kwargs):
        attempt = _make_attempt(
            initialized.root, unit.key.canonical_digest
        )
        return SimpleNamespace(
            state="succeeded", attempt_path=str(attempt), failure=None
        )

    monkeypatch.setattr(c9, "run_execution_unit", succeed)
    result = c9.run_c9_shard(
        initialized.plan, output_root=initialized.root, backend=object(),
        estimate=initialized.estimate, shard_index=1, shard_count=2,
        observed_git_state=LIVE_CODE,
    )

    assert len(result) == 1
    rows = _read(initialized.root / "failure_table.json")["rows"]
    assert [row["state"] for row in rows] == ["failed", "succeeded"]
    status = _read(initialized.root / "batch_status.json")
    assert status["lifecycle_state"] == "running"
    # The production completion predicate intentionally remains exact-320;
    # this two-unit fixture proves continuation without impersonating a formal
    # completed batch.
    assert status["orchestration_complete"] is False
    assert status["successful"] is False


def test_primary_validity_uses_exactly_three_primary_units_for_all_holdouts(
    formal, monkeypatch
):
    rows = _terminal_rows(formal.plan)
    calls = []
    shared = []

    def evidence(row, *, unit, observed_git_state, caps):
        assert observed_git_state == LIVE_CODE
        calls.append(unit)
        return {
            "unit_key": unit.key.to_dict(),
            "scalars": {
                "clean_tensor_hashes": {"canonical": "a" * 64},
                "perturbations": [],
                "target_identity": {"class_index": 1},
            },
            "artifact_inventory": [],
        }

    monkeypatch.setattr(c9, "_read_success_evidence", evidence)
    monkeypatch.setattr(
        c9, "_sample_candidate_metrics",
        lambda _records, _sample_id, _block, _head: ({}, []),
    )

    def shared_inputs(records, **_kwargs):
        shared.extend(records)
        return []

    monkeypatch.setattr(c9, "_shared_input_errors", shared_inputs)

    valid, failures, evidence_rows, shared_errors = (
        c9._primary_holdout_validity(
            rows, formal.plan, Path("/synthetic/c9"), LIVE_CODE, {}
        )
    )

    assert valid == list(c9.C9_HOLDOUT_SAMPLE_IDS)
    assert failures == []
    assert shared_errors == []
    assert len(evidence_rows) == 320
    assert len(shared) == 320
    assert len(calls) == 320
    primary_calls = [
        unit for unit in calls
        if unit.sample.role == "holdout"
        and unit.key.method_id in VALUE_METHOD_IDS
        and (unit.key.block_index, unit.key.head_index) == (0, 9)
    ]
    assert len(primary_calls) == 45
    assert {unit.key.method_id for unit in primary_calls} == set(VALUE_METHOD_IDS)
    assert all(
        unit.sample.role == "holdout"
        and (unit.key.block_index, unit.key.head_index) == (0, 9)
        for unit in primary_calls
    )


def test_shared_input_validator_requires_exact_16_way_hashes_targets_and_roles(
    monkeypatch,
):
    sample_id = c9.C9_SAMPLE_IDS[0]
    records = []
    prediction = {
        "clean_top1_index": 7, "official_truth_index": 3,
        "clean_top1_correct": False,
    }
    for method_id in ALL_METHODS:
        candidates = WINNERS if method_id in VALUE_METHOD_IDS else (None,)
        for candidate in candidates:
            block = None if candidate is None else candidate.block_index
            head = None if candidate is None else candidate.head_index
            target = {
                "clean_input_sha256": "1" * 64,
                "comparison_line": (
                    "value_path" if candidate is not None else "class_logit"
                ),
                "target_ref": (
                    "fixed_clean_attention_cls_head_direction"
                    if candidate is not None
                    else "clean_predicted_class_raw_logit"
                ),
                "frozen_target": {
                    "runtime_model_binding": {
                        "kind": "same_live_model_instance_and_tensor_state_versions",
                        "sha256": f"{len(records):064x}",
                        "is_canonical_model_artifact_hash": False,
                    },
                    **({"candidate": {"block_index": block, "head_index": head}}
                       if candidate is not None else {"class_index": 7}),
                },
            }
            records.append({
                "unit_key": {
                    "sample_id": sample_id, "method_id": method_id,
                    "block_index": block, "head_index": head,
                    "config_id": (
                        "global" if candidate is None
                        else f"block_{block:02d}_head_{head:02d}"
                    ),
                },
                "scalars": {
                    "clean_tensor_hashes": {
                        "canonical_rgb_sha256": "2" * 64,
                        "normalized_tensor_sha256": "1" * 64,
                    },
                    "perturbations": [
                        {
                            "perturbation_id": perturbation_id,
                            "tensor_sha256": f"{index + 3:064x}",
                            "normalized_tensor_sha256": f"{index + 6:064x}",
                        }
                        for index, perturbation_id in enumerate(c9.PERTURBATION_IDS)
                    ],
                    "target_identity": target, "prediction": prediction,
                    "repeatability": [{
                        "repeat_index": 1, "atol": 1e-6, "rtol": 1e-5,
                        "pass": True,
                    }],
                },
                "artifact_inventory": [],
            })
    assert len(records) == 16
    monkeypatch.setattr(c9, "_record_validity", lambda *_args, **_kwargs: [])
    baseline_errors = c9._shared_input_errors(records)
    assert not any(error.startswith(f"{sample_id}:") for error in baseline_errors)
    records[-1]["scalars"]["prediction"] = {
        **prediction, "clean_top1_index": 8,
    }
    errors = c9._shared_input_errors(records)
    assert any(
        error.startswith(f"{sample_id}:")
        and "prediction context is not shared" in error
        for error in errors
    )


def _terminal_rows(plan, root=Path("/synthetic")):
    rows = c9._pending_rows(plan)
    for row in rows:
        row.update({
            "state": "succeeded",
            "terminal": True,
            "complete": True,
            "claim_token": "a" * 64,
            "selected_attempt": str(
                (Path(root) / "units" / row["unit_key"]["canonical_digest"]
                 / "attempts" / "attempt-000001").absolute()
            ),
            "selected_status_sha256": "b" * 64,
            "l0_l1_bytes": 0,
            "l2_bytes": 0,
            "total_bytes": 0,
        })
    return rows


def _passing_post_c9_budget():
    projection = {
        "hard_cap_checks": [
            {
                "cap_key": key, "observed": 0,
                "cap": c9.FROZEN_CAPS[key], "measured": True,
                "comparison": "observed <= cap", "status": "PASS",
            }
            for key in c9.HARD_CAP_KEYS
        ],
        "all_hard_caps_pass": True,
    }
    value = {
        "schema_version": 1,
        "document_type": "stage_c_c9_post_run_budget",
        "status": "PASS", "blocked_reasons": [],
        "source_unit_counts": {
            "c8_expected": 2180, "c8_succeeded": 2179,
            "c9_expected": 320, "c9_succeeded": 320,
            "budget_measurement_expected": 2499,
            "budget_measurement_usable": 2499,
        },
        "measurement_boundary": {
            "c8_source_regular_file_bytes": 0,
            "c9_source_regular_file_bytes_before_report": 0,
            "prospective_final_report_bytes": 0,
            "c9_retained_bytes_with_report": 0,
            "combined_retained_bytes_with_report": 0,
            "c8_source_control_bytes": 0,
            "c9_source_control_bytes": 0,
            "actual_combined_cap_check": {
                "cap_key": "stage_c_c8_c9_total_artifact_bytes_max",
                "observed": 0,
                "cap": c9.FROZEN_CAPS[
                    "stage_c_c8_c9_total_artifact_bytes_max"
                ],
                "comparison": "observed <= cap", "status": "PASS",
            },
        },
        "projection": projection,
    }
    return _resign(value)


def _finalize_without_artifacts(
    monkeypatch, tmp_path, formal, *, valid_ids, failures
):
    root = tmp_path / "c9-final"
    root.mkdir()
    for name in c9._CONTROL_NAMES:
        (root / name).write_text("{}\n", encoding="utf-8")
    rows = _terminal_rows(formal.plan, root)
    authorization = _synthetic_authorization(
        root, formal.plan, formal.estimate
    )
    status = c9._batch_status(
        formal.plan, rows, "f" * 64, lifecycle_state="running"
    )
    captured = {}
    monkeypatch.setattr(
        c9, "_lifecycle_lock", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(
        c9, "_checkpoint_lock", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(
        c9, "_artifact_activity_lock",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        c9, "load_c9_initialized_state",
        lambda *_args, **_kwargs: (formal.estimate, authorization),
    )
    monkeypatch.setattr(
        c9, "_validate_checkpoint_controls",
        lambda *_args, **_kwargs: (rows, status),
    )
    monkeypatch.setattr(
        c9, "_scan_rows", lambda *_args, **_kwargs: (rows, [])
    )
    monkeypatch.setattr(
        c9, "_validate_closed_layout", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        c9, "_combined_artifact_errors", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        c9, "_revalidate_handoff_from_authorization",
        lambda *_args, **kwargs: (
            kwargs.get("source_evidence_out", []).append({})
            if kwargs.get("source_evidence_out") is not None else None
        ) or authorization["c8_handoff"],
    )
    monkeypatch.setattr(
        c9, "_primary_holdout_validity",
        lambda *_args, **_kwargs: (list(valid_ids), list(failures), [], []),
    )
    monkeypatch.setattr(
        c9, "_build_post_c9_budget",
        lambda *_args, **_kwargs: _passing_post_c9_budget(),
    )
    def capture(path, value):
        captured[Path(path).name] = json.loads(json.dumps(value))

    monkeypatch.setattr(c9._c8, "_atomic_create_json", capture)
    report = c9.finalize_c9_output_root(
        root, formal.plan, observed_git_state=LIVE_CODE
    )
    assert captured[c9.FINAL_REPORT_FILENAME] == report
    return report


def test_finalize_emits_evidence_only_and_never_c10_outputs(
    formal, tmp_path, monkeypatch
):
    report = _finalize_without_artifacts(
        monkeypatch, tmp_path, formal,
        valid_ids=c9.C9_HOLDOUT_SAMPLE_IDS, failures=[],
    )

    assert report["status"] == "PASS"
    assert report["expected_unit_count"] == 320
    assert report["terminal_unit_count"] == 320
    assert report["state_counts"] == {"succeeded": 320}
    assert report["primary_holdout_valid_count"] == 15
    assert report["primary_holdout_complete"] is True
    assert report["primary_holdout_failures"] == []
    assert report["selection_performed"] is False
    assert report["bootstrap_performed"] is False
    assert report["holm_performed"] is False
    assert report["decision_performed"] is False
    assert set(report["bindings"]) == {
        "output_root", "code", "c8_handoff_sha256",
        "execution_plan_sha256", "run_spec_sha256",
        "preflight_sha256", "authorization_sha256",
        "control_file_sha256",
    }
    assert set(report["bindings"]["control_file_sha256"]) == {
        "execution_plan.json", "preflight.json",
        "failure_table.json", "batch_status.json",
        c9.AUTHORIZATION_FILENAME,
    }
    assert report["bindings"]["c8_handoff_sha256"] == (
        _synthetic_handoff()["canonical_digest"]
    )
    assert not any(
        token in json.dumps(report).upper()
        for token in (
            '"GO"', "NO_GO_VALIDITY", "NO_GO_NOVELTY",
            "NO_GO_COST", "INCONCLUSIVE",
        )
    )
    c9.validate_c9_final_report(report, plan=formal.plan)


def test_finalize_fails_closed_when_primary_is_only_14_of_15_valid(
    formal, tmp_path, monkeypatch
):
    invalid_id = c9.C9_HOLDOUT_SAMPLE_IDS[-1]
    report = _finalize_without_artifacts(
        monkeypatch, tmp_path, formal,
        valid_ids=c9.C9_HOLDOUT_SAMPLE_IDS[:-1],
        failures=[{
            "sample_id": invalid_id,
            "reasons": ["primary_validity_failed"],
        }],
    )

    assert report["status"] == "BLOCKED"
    assert report["orchestration_complete"] is True
    assert report["successful"] is True
    assert report["primary_holdout_valid_count"] == 14
    assert report["primary_holdout_complete"] is False
    assert report["primary_holdout_failures"] == [{
        "sample_id": invalid_id,
        "reasons": ["primary_validity_failed"],
    }]
    assert "NO_GO_VALIDITY" not in json.dumps(report)
    c9.validate_c9_final_report(report, plan=formal.plan)


@pytest.mark.parametrize(
    "flag",
    ["selection_performed", "bootstrap_performed",
     "holm_performed", "decision_performed"],
)
def test_final_report_rejects_re_signed_c10_flags(
    formal, tmp_path, monkeypatch, flag
):
    report = _finalize_without_artifacts(
        monkeypatch, tmp_path, formal,
        valid_ids=c9.C9_HOLDOUT_SAMPLE_IDS, failures=[],
    )
    report[flag] = True
    report = _resign(report)

    with pytest.raises(c9.StageCBenchmarkIntegrityError, match="forbidden C10"):
        c9.validate_c9_final_report(report, plan=formal.plan)


def test_final_report_rejects_re_signed_decision_payload(
    formal, tmp_path, monkeypatch
):
    report = _finalize_without_artifacts(
        monkeypatch, tmp_path, formal,
        valid_ids=c9.C9_HOLDOUT_SAMPLE_IDS, failures=[],
    )
    report["errors"] = ["NO_GO_VALIDITY"]
    report["status"] = "BLOCKED"
    report["integrity_valid"] = False
    report["successful"] = False
    report = _resign(report)

    with pytest.raises(c9.StageCBenchmarkIntegrityError, match="forbidden C10"):
        c9.validate_c9_final_report(report, plan=formal.plan)


def test_final_report_rejects_re_signed_row_state_flag_inconsistency(
    formal, tmp_path, monkeypatch
):
    report = _finalize_without_artifacts(
        monkeypatch, tmp_path, formal,
        valid_ids=c9.C9_HOLDOUT_SAMPLE_IDS, failures=[],
    )
    report["rows"][0]["complete"] = False
    report = _resign(report)

    with pytest.raises(c9.StageCBenchmarkIntegrityError):
        c9.validate_c9_final_report(report, plan=formal.plan)


def test_final_report_rejects_re_signed_success_claim_with_failed_row(
    formal, tmp_path, monkeypatch
):
    report = _finalize_without_artifacts(
        monkeypatch, tmp_path, formal,
        valid_ids=c9.C9_HOLDOUT_SAMPLE_IDS, failures=[],
    )
    row = report["rows"][0]
    row["state"] = "failed"
    row["complete"] = False
    row["failure"] = {"reason": "synthetic"}
    report["state_counts"] = {"failed": 1, "succeeded": 319}
    report["status"] = "BLOCKED"
    report = _resign(report)

    with pytest.raises(c9.StageCBenchmarkIntegrityError):
        c9.validate_c9_final_report(report, plan=formal.plan)


@pytest.mark.parametrize("state", ["missing", "incomplete", "tampered"])
def test_nonterminal_final_states_never_count_as_completed_execution(
    formal, tmp_path, monkeypatch, state
):
    report = _finalize_without_artifacts(
        monkeypatch, tmp_path, formal,
        valid_ids=c9.C9_HOLDOUT_SAMPLE_IDS, failures=[],
    )
    row = report["rows"][0]
    row.update({"state": state, "terminal": False, "complete": False})
    if state == "missing":
        row.update({
            "selected_attempt": None, "selected_status_sha256": None,
            "l0_l1_bytes": None, "l2_bytes": None, "total_bytes": None,
        })
    report["terminal_unit_count"] = 319
    report["orchestration_complete"] = False
    report["successful"] = False
    report["integrity_valid"] = False
    report["state_counts"] = {state: 1, "succeeded": 319}
    report["status"] = "BLOCKED"
    report["errors"] = [
        f"{row['unit_key']['canonical_digest']}: explicit {state} evidence"
    ]
    report["canonical_digest"] = c9._self_digest(report)

    c9.validate_c9_final_report(report, plan=formal.plan)

    row["terminal"] = True
    report["terminal_unit_count"] = 320
    report["orchestration_complete"] = True
    report["canonical_digest"] = c9._self_digest(report)
    with pytest.raises(c9.StageCBenchmarkIntegrityError, match="state flags"):
        c9.validate_c9_final_report(report, plan=formal.plan)


def test_post_c9_budget_rejects_unmeasured_or_failed_cap():
    budget = _passing_post_c9_budget()
    budget["projection"]["hard_cap_checks"][4]["measured"] = False
    budget["projection"]["hard_cap_checks"][4]["status"] = "BLOCKED"
    budget["projection"]["all_hard_caps_pass"] = False
    budget = _resign(budget)

    with pytest.raises(c9.StageCBenchmarkIntegrityError):
        c9._validate_post_c9_budget(budget)


def _budget_record(unit, *, seconds=10.0):
    digest = "d" * 64
    return {
        "unit_key": unit.key.to_dict(), "state": "succeeded",
        "trusted_by_batch_validator": True,
        "attempt_path": "/synthetic/attempt-000001",
        "resources": {
            "calls": {
                "accounting_unit": "batched_model_invocation",
                "forward_calls": 10, "backward_calls": 5,
            },
            "cuda": {
                "status": "measured", "peak_reserved_bytes": 1024,
            },
        },
        "runner_telemetry": {
            "schema_version": 1,
            "phase_timings": {
                "clock": "perf_counter",
                "synchronization": "device_before_and_after_each_phase",
                "phases_are_mutually_exclusive": False,
                "total_boundary": (
                    "unit_start_through_first_complete_validation_"
                    "excluding_telemetry_publication"
                ),
                "synchronized_total_wall_time_seconds": seconds,
                "phases": {
                    name: {"wall_time_seconds": 1.0, "invocations": 1}
                    for name in (
                        "attribution_or_map_compute", "fidelity_compute",
                        "integrity_checking", "artifact_finalize",
                    )
                },
            },
            "lifecycle": {
                "integrity": {
                    "schema_version": 1,
                    "algorithm": "stage_c_strong_model_sha256_v1",
                    "checked": ["parameter_and_persistent_buffer_content"],
                    "entry_sha256": digest, "exit_sha256": digest,
                    "matched": True, "restored": True, "differences": [],
                    "signature_calls": 2, "verification_error_type": None,
                    "verification_error_message": None,
                    "body_exception_type": None,
                },
                "restoration": {
                    "schema_version": 1,
                    "checked": ["module_modes", "hooks", "grad", "requires_grad"],
                    "changes_detected": False, "restored": True,
                    "differences": [], "errors": [],
                },
            },
            "measurement_environment": {
                "device_type": "cuda", "device": "cuda:0",
                "device_name": "NVIDIA H20",
                "cuda_runtime_version": "12.1", "cudnn_version": 8902,
            },
        },
        "work_item_counts": dict(unit.work_item_counts),
        "l0_l1_cap_accounting_bytes": 4096,
        "level_bytes": {
            "L0": 1024, "L1": 1024, "L2": 0,
            "L3": 1024, "control": 1024,
        },
        "total_regular_file_bytes": 4096,
    }


def test_post_c9_budget_recomputes_ten_caps_from_c8_plus_c9_actuals(
    formal, tmp_path, monkeypatch
):
    root = tmp_path / "budget-root"
    root.mkdir()
    for name in c9._CONTROL_NAMES:
        (root / name).write_text("{}\n", encoding="utf-8")
    rows = _terminal_rows(formal.plan, root)
    c9_records = [_budget_record(unit) for unit in formal.plan.units]
    c8_unit = formal.plan.units[0]
    c8_records = [_budget_record(c8_unit) for _ in range(2179)]
    c8_records.append({
        "unit_key": c8_unit.key.to_dict(), "state": "failed",
        "trusted_by_batch_validator": True,
    })
    storage_model = _read(
        c9.C8_REPORT_PATH
    )["bindings"]["c7_report_binding"]["recomputed_estimate"]["storage"]
    c8_root = tmp_path / "c8-budget-root"
    c8_root.mkdir()
    for name in c9.C8_CONTROL_FILE_SHA256:
        (c8_root / name).write_text("{}\n", encoding="utf-8")
    c8_evidence = {
        "source_root": str(c8_root),
        "records": c8_records,
        "bindings": {"source_tree_total_bytes": 10_000},
        "c7_report_binding": {
            "recomputed_estimate": {"storage": storage_model},
            "source_batch_control_bytes": 100,
        },
    }
    monkeypatch.setattr(
        c9, "measure_artifact_tree_bytes",
        lambda _root: SimpleNamespace(total_bytes=10_000),
    )

    budget = c9._build_post_c9_budget(
        root, formal.plan, rows, c9_records, c8_evidence, c9.FROZEN_CAPS
    )

    assert budget["status"] == "PASS"
    assert budget["source_unit_counts"] == {
        "c8_expected": 2180, "c8_succeeded": 2179,
        "c9_expected": 320, "c9_succeeded": 320,
        "budget_measurement_expected": 2499,
        "budget_measurement_usable": 2499,
    }
    checks = budget["projection"]["hard_cap_checks"]
    assert [item["cap_key"] for item in checks] == list(c9.HARD_CAP_KEYS)
    assert all(item["measured"] and item["status"] == "PASS" for item in checks)
    assert budget["projection"]["time"]["slowest_valid_measured_unit_seconds"] == 10.0
    c9._validate_post_c9_budget(budget)


def test_finalize_and_validate_bind_real_report_bytes_to_fresh_budget(
    formal, tmp_path, monkeypatch
):
    root = tmp_path / "c9-fixed-point"
    root.mkdir()
    rows = _terminal_rows(formal.plan, root)
    authorization = _synthetic_authorization(
        root, formal.plan, formal.estimate
    )
    table = c9._failure_table(formal.plan, rows)
    c9._c8._atomic_create_json(root / "execution_plan.json", {})
    c9._c8._atomic_create_json(root / "preflight.json", {})
    c9._c8._atomic_create_json(
        root / c9.AUTHORIZATION_FILENAME, authorization
    )
    c9._c8._atomic_create_json(root / "failure_table.json", table)
    c9._c8._atomic_create_json(
        root / "batch_status.json",
        c9._batch_status(
            formal.plan, rows,
            c9._file_sha256(root / "failure_table.json"),
            lifecycle_state="running",
        ),
    )

    c9_records = [_budget_record(unit) for unit in formal.plan.units]
    c8_unit = formal.plan.units[0]
    c8_records = [_budget_record(c8_unit) for _ in range(2179)]
    c8_records.append({
        "unit_key": c8_unit.key.to_dict(),
        "state": "failed",
        "trusted_by_batch_validator": True,
    })
    c8_root = tmp_path / "c8-fixed-point-source"
    c8_root.mkdir()
    for name in c9.C8_CONTROL_FILE_SHA256:
        (c8_root / name).write_text("{}\n", encoding="utf-8")
    storage_model = _read(
        c9.C8_REPORT_PATH
    )["bindings"]["c7_report_binding"]["recomputed_estimate"]["storage"]
    c8_evidence = {
        "source_root": str(c8_root),
        "records": c8_records,
        "bindings": {
            "source_tree_total_bytes": (
                c9.measure_artifact_tree_bytes(c8_root).total_bytes
            ),
        },
        "c7_report_binding": {
            "recomputed_estimate": {"storage": storage_model},
            "source_batch_control_bytes": sum(
                (c8_root / name).stat().st_size
                for name in c9.C8_CONTROL_FILE_SHA256
            ),
        },
    }

    monkeypatch.setattr(
        c9, "load_c9_initialized_state",
        lambda *_args, **_kwargs: (formal.estimate, authorization),
    )
    monkeypatch.setattr(
        c9, "_scan_rows", lambda *_args, **_kwargs: (rows, [])
    )
    monkeypatch.setattr(
        c9, "_validate_closed_layout", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        c9, "_primary_holdout_validity",
        lambda *_args, **_kwargs: (
            list(c9.C9_HOLDOUT_SAMPLE_IDS), [], c9_records, []
        ),
    )

    source_revalidation_count = 0

    def revalidate_source(_handoff, **kwargs):
        nonlocal source_revalidation_count
        source_revalidation_count += 1
        assert kwargs["measure_source_tree"] is True
        kwargs["source_evidence_out"].append(c8_evidence)
        return authorization["c8_handoff"]

    monkeypatch.setattr(
        c9, "_revalidate_handoff_from_authorization", revalidate_source
    )
    real_budget_builder = c9._build_post_c9_budget
    budget_build_phases = []

    def build_budget(*args, **kwargs):
        budget_build_phases.append(
            "validate" if (root / c9.FINAL_REPORT_FILENAME).exists()
            else "finalize"
        )
        return real_budget_builder(*args, **kwargs)

    monkeypatch.setattr(c9, "_build_post_c9_budget", build_budget)

    report = c9.finalize_c9_output_root(
        root, formal.plan, observed_git_state=LIVE_CODE
    )
    report_path = root / c9.FINAL_REPORT_FILENAME
    report_size = report_path.stat().st_size
    assert report == _read(report_path)
    assert report["post_c9_budget"]["measurement_boundary"][
        "prospective_final_report_bytes"
    ] == report_size

    validated = c9.validate_c9_output_root(
        root, formal.plan, observed_git_state=LIVE_CODE
    )

    assert validated == report
    assert source_revalidation_count == 2
    assert budget_build_phases == ["finalize", "validate"]
    assert validated["post_c9_budget"]["measurement_boundary"][
        "combined_retained_bytes_with_report"
    ] == (
        c8_evidence["bindings"]["source_tree_total_bytes"]
        + c9.measure_artifact_tree_bytes(root).total_bytes
    )
