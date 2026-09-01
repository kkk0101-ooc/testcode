from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from types import SimpleNamespace

import pytest

from vitnns import stage_c_artifacts
from vitnns import stage_c_c8 as c8
from vitnns import recovery_c8_launch
from vitnns.recovery_v2 import load_recovery_v2_binding
from vitnns.stage_c_config import load_stage_c_config
from vitnns.stage_c_plan import (
    StageCEstimate,
    build_execution_plan,
    estimate_stage_c_run,
    load_run_spec,
    load_stage_c_registry,
)
from vitnns.stage_c_provenance import canonical_sha256
from vitnns.recovery_c8_launch import RecoveryC8LaunchRecord


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "config/stage_c_c8_run_spec.json"
LIVE_CODE = {"commit": "a" * 40, "dirty": False}
ARCHIVE_TREE = "b" * 40
RECOVERY_CODE = {"commit": "c" * 40, "dirty": False}
RECOVERY_PROJECTIONS = {
    "c7_slowest_unit_seconds": 80.01139638799941,
    "c7_peak_reserved_bytes": 2_833_252_352,
    "c7_projected_c8_h20_gpu_hours": 60.564181988138444,
    "c7_projected_c8_artifact_bytes": 4_289_580_675,
    "c7_projected_c8_c9_artifact_bytes": 4_841_884_075,
    "c7_projected_c8_l2_bytes_per_audit": 488_599_600,
}


@pytest.fixture(scope="module")
def exact():
    config = load_stage_c_config()
    # The historical workspace-relative selection sources are intentionally
    # absent in a recovery checkout.  Planning through the checked-in recovery
    # binding keeps this fixture calibration-only without reading live images.
    registry = load_stage_c_registry(
        config, verify_live_assets=False,
        recovery_binding=load_recovery_v2_binding(verify_assets=False),
    )
    plan = build_execution_plan(config, registry, load_run_spec(SPEC))
    return config, plan


@pytest.fixture(scope="module")
def formal(exact):
    config, plan = exact
    estimate = estimate_stage_c_run(
        config,
        plan,
        measured_slowest_unit_seconds=c8.C7_SLOWEST_UNIT_SECONDS,
        measured_peak_reserved_bytes_per_unit=c8.C7_PEAK_RESERVED_BYTES,
    )
    return config, plan, estimate


def _authorization(root, plan, estimate, *, code=LIVE_CODE):
    value = {
        "schema_version": 1,
        "document_type": "stage_c_c8_launch_authorization",
        "status": "PASS",
        "holdout_accessed": False,
        "c7_report_path": str((Path(root).parent / "c7.json").absolute()),
        "c7_report_file_sha256": c8.C7_REPORT_FILE_SHA256,
        "c7_report_canonical_digest": c8.C7_REPORT_CANONICAL_DIGEST,
        "c7_code_commit": c8.C7_CODE_COMMIT,
        "c7_archive_ref": c8.C7_ARCHIVE_REF,
        "c7_archive_tree": ARCHIVE_TREE,
        "c7_slowest_unit_seconds": c8.C7_SLOWEST_UNIT_SECONDS,
        "c7_peak_reserved_bytes": c8.C7_PEAK_RESERVED_BYTES,
        "c7_projected_c8_h20_gpu_hours": c8.C7_PROJECTED_C8_H20_GPU_HOURS,
        "c7_projected_c8_artifact_bytes": c8.C7_PROJECTED_C8_ARTIFACT_BYTES,
        "c7_projected_c8_l2_bytes_per_audit": (
            c8.C7_PROJECTED_C8_L2_BYTES_PER_AUDIT
        ),
        "c8_execution_plan_sha256": plan.canonical_digest,
        "c8_preflight_sha256": estimate.canonical_digest,
        "c8_output_root": str(Path(root).absolute()),
        "code": dict(code),
    }
    value["canonical_digest"] = canonical_sha256(value)
    return value


def _install_evidence(monkeypatch, *, live_code=LIVE_CODE):
    """Install controlled evidence while retaining every production check."""

    calls = {"preflight": [], "live_git": [], "c7": []}
    validate_preflight = c8._validate_formal_preflight

    def tracked_preflight(estimate, plan, *, require_issuer):
        calls["preflight"].append(require_issuer)
        return validate_preflight(
            estimate, plan, require_issuer=require_issuer
        )

    def capture(repo_root):
        calls["live_git"].append(Path(repo_root))
        return dict(live_code)

    def validate_c7(path, *, repo_root=None):
        calls["c7"].append((Path(path), repo_root))
        return {
            "canonical_digest": c8.C7_REPORT_CANONICAL_DIGEST,
            "c8_validated_archive_ref": {"tree": ARCHIVE_TREE},
        }

    monkeypatch.setattr(c8, "_validate_formal_preflight", tracked_preflight)
    monkeypatch.setattr(c8, "capture_git_state", capture)
    monkeypatch.setattr(c8, "validate_fixed_c7_report", validate_c7)
    return calls


@pytest.fixture(scope="module")
def initialized_template(tmp_path_factory, formal):
    """Pay the cost of serializing the full C8 plan only once."""

    config, plan, estimate = formal
    root = tmp_path_factory.mktemp("c8-template") / "root"
    authorization = _authorization(root, plan, estimate)
    monkeypatch = pytest.MonkeyPatch()
    calls = _install_evidence(monkeypatch)
    try:
        result = c8.initialize_c8_output_root(
            root,
            plan,
            estimate,
            authorization,
            config=config,
            observed_git_state=LIVE_CODE,
        )
    finally:
        monkeypatch.undo()
    return SimpleNamespace(
        root=result, authorization=authorization, evidence=calls
    )


@pytest.fixture
def initialized(tmp_path, formal, initialized_template, monkeypatch):
    """Clone immutable controls without rebuilding the 1.3M-work-item plan."""

    config, plan, estimate = formal
    root = tmp_path / "c8"
    root.mkdir()
    for name in c8._CONTROL_NAMES:
        source = initialized_template.root / name
        destination = root / name
        if name in {"execution_plan.json", "preflight.json"}:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        else:
            shutil.copy2(source, destination)
    authorization = _authorization(root, plan, estimate)
    c8._atomic_replace_json(root / c8.AUTHORIZATION_FILENAME, authorization)
    evidence = _install_evidence(monkeypatch)
    return SimpleNamespace(
        config=config, plan=plan, estimate=estimate, root=root,
        authorization=authorization, evidence=evidence,
    )


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _valid_attempt():
    return SimpleNamespace(
        valid=True,
        status="complete",
        complete=True,
        terminal=True,
        resumable=True,
        errors=(),
        l0_l1_bytes=0,
        l2_bytes=0,
        total_bytes=0,
    )


def _make_attempt(root, digest, *, number=1):
    attempt = (
        Path(root) / "units" / digest / "attempts"
        / f"attempt-{number:06d}"
    ).absolute()
    attempt.mkdir(parents=True)
    (attempt / "status.json").write_text("{}\n", encoding="utf-8")
    return attempt


def _recovery_record(tmp_path, binding):
    repo = tmp_path / "repo"
    report_path = repo / "evidence/c7/reports/stage_c_benchmark_report.json"
    launch_path = repo / "evidence/c8/recovery_c8_launch_record.json"
    schema_path = repo / "config/recovery_c8_launch_record.schema.json"
    formal_root = repo / "evidence/c8/formal"
    report_path.parent.mkdir(parents=True)
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "canonical_digest": "1" * 64,
        "status": "PASS",
        "integrity_valid": True,
        "projections": {
            "time": {
                "slowest_valid_measured_unit_seconds":
                    RECOVERY_PROJECTIONS["c7_slowest_unit_seconds"],
                "C8_h20_gpu_hours":
                    RECOVERY_PROJECTIONS["c7_projected_c8_h20_gpu_hours"],
            },
            "measured_maxima": {
                "cuda_peak_reserved_bytes_per_unit":
                    RECOVERY_PROJECTIONS["c7_peak_reserved_bytes"],
                "c8_projected_l2_bytes_per_audit_sample":
                    RECOVERY_PROJECTIONS["c7_projected_c8_l2_bytes_per_audit"],
            },
            "storage": {"projected_bytes": {
                "C8": RECOVERY_PROJECTIONS["c7_projected_c8_artifact_bytes"],
                "C8_C9": RECOVERY_PROJECTIONS["c7_projected_c8_c9_artifact_bytes"],
            }},
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    launch_path.write_text("{}", encoding="utf-8")
    schema_path.write_text("{}", encoding="utf-8")
    c7_launch = SimpleNamespace(binding=binding)
    document = {
        "c7_launch": {
            "path": "evidence/c7/recovery_c7_launch_record.json",
            "file_sha256": "2" * 64,
            "canonical_digest": "3" * 64,
            "launch_commit": "4" * 40,
            "producer_commit": "5" * 40,
            "producer_tree": "6" * 40,
        },
        "c7_report": {
            "path": report_path.relative_to(repo).as_posix(),
            "file_sha256": c8._file_sha256(report_path),
            "canonical_digest": report["canonical_digest"],
            "report_commit": "7" * 40,
            "status": "PASS",
            "source_commit": "4" * 40,
        },
        "model": {"identity": "recovery", "state_dict_sha256": "8" * 64},
        "producer_git": {
            "commit": "9" * 40, "tree": "a" * 40,
            "source_subtree_path": "src/vitnns",
            "source_subtree_tree": "b" * 40, "dirty": False,
        },
        "canonical_digest": "d" * 64,
    }
    return RecoveryC8LaunchRecord(
        path=launch_path, schema_path=schema_path,
        document=MappingProxyType(document), binding=binding,
        c7_launch_record=c7_launch, model_path=repo / "model.pth",
        data_root=repo / "data", devkit_path=repo / "devkit.tar.gz",
        formal_root=formal_root, reports_root=repo / "evidence/c8/reports",
        selection_root=repo / "evidence/c8/selection",
        report_path=repo / "evidence/c8/reports/report.json",
        selection_path=repo / "evidence/c8/selection/selection.json",
        model_identity=MappingProxyType(document["model"]),
        file_sha256="e" * 64, launch_commit=RECOVERY_CODE["commit"],
        repo_root=repo,
    ), report


def test_checked_in_c8_plan_is_exact(exact):
    config, plan = exact
    result = c8.validate_c8_execution_plan(plan, config)
    assert result["valid"] is True
    assert result["counts"] == c8.C8_EXPECTED_COUNTS
    assert len(plan.units) == 5 * 436 == 2180
    assert {unit.sample.sample_id for unit in plan.units} == set(
        c8.CALIBRATION_SAMPLE_IDS
    )
    assert not any(unit.sample.role == "holdout" for unit in plan.units)


def test_c8_plan_validator_rejects_order_and_scope_drift(exact):
    config, plan = exact
    document = plan.to_dict()
    document["units"][0], document["units"][1] = (
        document["units"][1],
        document["units"][0],
    )
    document.pop("canonical_digest")
    document["canonical_digest"] = canonical_sha256(document)
    altered = type(plan).from_dict(document)
    result = c8.validate_c8_execution_plan(altered, config)
    assert result["valid"] is False
    assert any("ordered unit" in item for item in result["violations"])


def test_measured_preflight_uses_c7_slowest_and_peak(monkeypatch, exact):
    config, plan = exact
    captured = {}
    expected = SimpleNamespace(
        within_budget=True,
        canonical_digest="a" * 64,
        compute={
            "projected_h20_gpu_hours": c8.C7_PROJECTED_C8_H20_GPU_HOURS
        },
    )

    def fake_estimate(config_arg, plan_arg, **kwargs):
        captured.update(kwargs)
        assert config_arg is config and plan_arg is plan
        return expected

    monkeypatch.setattr(c8, "estimate_stage_c_run", fake_estimate)
    report = {
        "projections": {
            "time": {
                "slowest_valid_measured_unit_seconds": (
                    c8.C7_SLOWEST_UNIT_SECONDS
                )
            },
            "measured_maxima": {
                "cuda_peak_reserved_bytes_per_unit": c8.C7_PEAK_RESERVED_BYTES
            },
        }
    }
    assert c8.build_c8_measured_preflight(plan, config, report) is expected
    assert captured == {
        "measured_slowest_unit_seconds": c8.C7_SLOWEST_UNIT_SECONDS,
        "measured_peak_reserved_bytes_per_unit": c8.C7_PEAK_RESERVED_BYTES,
    }


def test_initialize_revalidates_issuer_live_git_and_c7_and_writes_pending_rows(
    formal, initialized_template
):
    _config, _plan, _estimate = formal
    root = initialized_template.root
    table = _read(root / "failure_table.json")
    status = _read(root / "batch_status.json")

    assert initialized_template.evidence["preflight"] == [True]
    assert len(initialized_template.evidence["live_git"]) == 1
    assert initialized_template.evidence["c7"] == [
        (Path(initialized_template.authorization["c7_report_path"]), None)
    ]
    assert len(table["rows"]) == 2180
    assert all(
        row["state"] == "pending"
        and row["claim_token"] is None
        and row["selected_attempt"] is None
        for row in table["rows"]
    )
    assert status["expected_unit_count"] == 2180
    assert status["lifecycle_state"] == "running"
    assert status["active_claim_count"] == 0
    assert status["orchestration_complete"] is False
    assert status["failure_table_sha256"] == _sha256(
        root / "failure_table.json"
    )


def test_serialized_preflight_cannot_authorize_a_shard(initialized, monkeypatch):
    restored = StageCEstimate.from_dict(initialized.estimate.to_dict())
    monkeypatch.setattr(
        c8, "run_execution_unit",
        lambda *_args, **_kwargs: pytest.fail("backend was called"),
    )

    with pytest.raises(
        c8.StageCBenchmarkIntegrityError, match="freshly issuer-authenticated"
    ):
        c8.run_c8_shard(
            initialized.plan,
            output_root=initialized.root,
            backend=object(),
            estimate=restored,
            unit_digests=[initialized.plan.units[0].key.canonical_digest],
            observed_git_state=LIVE_CODE,
        )

    assert initialized.evidence["preflight"] == [True]
    assert initialized.evidence["live_git"] == []


def test_shard_revalidates_evidence_and_checkpoints_one_attempt(
    initialized, monkeypatch
):
    calls = []
    digest = initialized.plan.units[0].key.canonical_digest
    monkeypatch.setattr(
        stage_c_artifacts, "validate_execution_unit",
        lambda *_args, **_kwargs: _valid_attempt(),
    )

    def run(unit, **kwargs):
        attempt = _make_attempt(initialized.root, digest)
        # Match stage_c_runner.run_execution_unit's Mapping contract.  A
        # StageCEstimate has no .get(), so exercising this lookup protects the
        # formal shard boundary from passing the authenticated object itself.
        assert kwargs["preflight"].get("within_budget") is True
        calls.append((unit, kwargs))
        return SimpleNamespace(
            state="succeeded", attempt_path=str(attempt), failure=None
        )

    monkeypatch.setattr(c8, "run_execution_unit", run)
    result = c8.run_c8_shard(
        initialized.plan,
        output_root=initialized.root,
        backend=object(),
        estimate=initialized.estimate,
        unit_digests=[digest],
        observed_git_state=LIVE_CODE,
    )

    assert result[0].state == "succeeded"
    assert calls[0][1]["resume"] is False
    assert calls[0][1]["preflight"] == initialized.estimate.to_dict()
    assert calls[0][1]["preflight"] is not initialized.estimate
    assert calls[0][1]["observed_git_state"] == LIVE_CODE
    assert initialized.evidence["preflight"] == [True]
    assert len(initialized.evidence["live_git"]) == 1
    table = _read(initialized.root / "failure_table.json")
    row = table["rows"][0]
    assert row["state"] == "succeeded"
    assert len(row["claim_token"]) == 64
    assert row["selected_attempt"].endswith("attempt-000001")
    assert all(item["state"] == "pending" for item in table["rows"][1:])


def test_two_threads_competing_for_one_unit_run_backend_and_attempt_once(
    initialized, monkeypatch
):
    digest = initialized.plan.units[0].key.canonical_digest
    entered_backend = threading.Event()
    release_backend = threading.Event()
    backend_calls = []
    results = []
    errors = []
    monkeypatch.setattr(
        stage_c_artifacts, "validate_execution_unit",
        lambda *_args, **_kwargs: _valid_attempt(),
    )

    def run(unit, **_kwargs):
        backend_calls.append(unit.key.canonical_digest)
        attempt = _make_attempt(initialized.root, digest)
        entered_backend.set()
        assert release_backend.wait(timeout=20)
        return SimpleNamespace(
            state="succeeded", attempt_path=str(attempt), failure=None
        )

    def worker():
        try:
            results.append(
                c8.run_c8_shard(
                    initialized.plan,
                    output_root=initialized.root,
                    backend=object(),
                    estimate=initialized.estimate,
                    unit_digests=[digest],
                    observed_git_state=LIVE_CODE,
                )
            )
        except BaseException as error:  # captured for assertion in main thread
            errors.append(error)

    monkeypatch.setattr(c8, "run_execution_unit", run)
    first = threading.Thread(target=worker, name="c8-first")
    second = threading.Thread(target=worker, name="c8-second")
    first.start()
    assert entered_backend.wait(timeout=20)
    second.start()
    second.join(timeout=20)
    release_backend.set()
    first.join(timeout=20)
    second.join(timeout=20)

    assert not first.is_alive() and not second.is_alive()
    assert backend_calls == [digest]
    attempts = stage_c_artifacts.attempt_directories(
        initialized.root / "units" / digest / "attempts"
    )
    assert len(attempts) == 1
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], c8.StageCBenchmarkIntegrityError)
    assert "cannot be claimed from 'claimed'" in str(errors[0])


def test_activity_lock_allows_concurrent_writers_and_excludes_online_scan(
    initialized, monkeypatch
):
    digests = [
        initialized.plan.units[index].key.canonical_digest for index in (0, 1)
    ]
    second_writing = threading.Event()
    release_second = threading.Event()
    exclusive_requested = threading.Event()
    exclusive_entered = threading.Event()
    active_writers = set()
    state_guard = threading.Lock()
    local = threading.local()
    backend_calls = []
    measurements = []
    results = []
    errors = []
    real_activity_lock = c8._artifact_activity_lock

    @contextmanager
    def observed_activity_lock(root, *, exclusive):
        if exclusive:
            exclusive_requested.set()
        with real_activity_lock(root, exclusive=exclusive):
            local.exclusive = exclusive
            if exclusive:
                exclusive_entered.set()
            try:
                yield
            finally:
                del local.exclusive

    def run(unit, **_kwargs):
        digest = unit.key.canonical_digest
        assert local.exclusive is False
        with state_guard:
            active_writers.add(digest)
            backend_calls.append(digest)
        try:
            if digest == digests[1]:
                second_writing.set()
                assert release_second.wait(timeout=20)
            else:
                assert second_writing.is_set()
                with state_guard:
                    assert active_writers == set(digests)
        finally:
            with state_guard:
                active_writers.remove(digest)
        return SimpleNamespace(
            state="incomplete",
            attempt_path=None,
            failure={"reason": "synthetic incomplete unit"},
        )

    def measure(root):
        assert Path(root) == initialized.root
        assert local.exclusive is True
        with state_guard:
            assert active_writers == set()
        measurements.append(Path(root))
        return SimpleNamespace(total_bytes=0)

    def worker(digest):
        try:
            results.append(
                c8.run_c8_shard(
                    initialized.plan, output_root=initialized.root,
                    backend=object(), estimate=initialized.estimate,
                    unit_digests=[digest], observed_git_state=LIVE_CODE,
                )
            )
        except BaseException as error:  # captured for assertion in main thread
            errors.append(error)

    monkeypatch.setattr(c8, "_artifact_activity_lock", observed_activity_lock)
    monkeypatch.setattr(c8, "run_execution_unit", run)
    monkeypatch.setattr(c8, "measure_artifact_tree_bytes", measure)
    second = threading.Thread(target=worker, args=(digests[1],), name="c8-writer-2")
    first = threading.Thread(target=worker, args=(digests[0],), name="c8-writer-1")
    second.start()
    assert second_writing.wait(timeout=20)
    first.start()
    assert exclusive_requested.wait(timeout=20)

    # Writer 1 has returned and requested the exclusive tree-scan lock, but it
    # cannot enter or measure while writer 2 still owns a shared activity lock.
    assert exclusive_entered.is_set() is False
    assert measurements == []
    release_second.set()
    first.join(timeout=20)
    second.join(timeout=20)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert len(results) == 2
    assert set(backend_calls) == set(digests)
    assert len(measurements) == 2
    rows = _read(initialized.root / "failure_table.json")["rows"]
    assert [rows[index]["state"] for index in (0, 1)] == [
        "incomplete", "incomplete"
    ]


def test_failure_table_and_batch_status_tamper_are_rejected(
    initialized, monkeypatch
):
    root = initialized.root
    digest = initialized.plan.units[0].key.canonical_digest
    table_path = root / "failure_table.json"
    status_path = root / "batch_status.json"
    original_table = table_path.read_bytes()
    backend_calls = []
    monkeypatch.setattr(
        c8, "run_execution_unit",
        lambda *_args, **_kwargs: backend_calls.append(True),
    )

    table = _read(table_path)
    table["plan_digest"] = "0" * 64
    c8._atomic_replace_json(table_path, table)
    with pytest.raises(
        c8.StageCBenchmarkIntegrityError, match="failure table header drifted"
    ):
        c8.run_c8_shard(
            initialized.plan, output_root=root, backend=object(),
            estimate=initialized.estimate, unit_digests=[digest],
            observed_git_state=LIVE_CODE,
        )

    table_path.write_bytes(original_table)
    status = _read(status_path)
    status["active_claim_count"] = 1
    c8._atomic_replace_json(status_path, status)
    with pytest.raises(
        c8.StageCBenchmarkIntegrityError,
        match="batch status does not bind the current failure table",
    ):
        c8.run_c8_shard(
            initialized.plan, output_root=root, backend=object(),
            estimate=initialized.estimate, unit_digests=[digest],
            observed_git_state=LIVE_CODE,
        )

    assert backend_calls == []


def test_runner_selected_attempt_outside_claimed_unit_is_rejected(
    initialized, monkeypatch, tmp_path
):
    digest = initialized.plan.units[0].key.canonical_digest
    external = (tmp_path / "external" / "attempt-000001").absolute()

    def run(*_args, **_kwargs):
        external.mkdir(parents=True)
        return SimpleNamespace(
            state="succeeded", attempt_path=str(external), failure=None
        )

    monkeypatch.setattr(c8, "run_execution_unit", run)
    with pytest.raises(
        c8.StageCBenchmarkIntegrityError,
        match="attempt outside the claimed unit",
    ):
        c8.run_c8_shard(
            initialized.plan, output_root=initialized.root, backend=object(),
            estimate=initialized.estimate, unit_digests=[digest],
            observed_git_state=LIVE_CODE,
        )

    row = _read(initialized.root / "failure_table.json")["rows"][0]
    assert row["state"] == "claimed"
    assert row["selected_attempt"] is None
    assert external.is_dir()


def test_online_full_tree_storage_cap_blocks_immediately(
    initialized, monkeypatch
):
    digest = initialized.plan.units[0].key.canonical_digest
    cap = initialized.estimate.caps["effective"][
        "stage_c_c8_c9_total_artifact_bytes_max"
    ]
    backend_calls = []
    measurements = []
    activity_modes = []
    real_activity_lock = c8._artifact_activity_lock

    @contextmanager
    def observed_activity_lock(root, *, exclusive):
        with real_activity_lock(root, exclusive=exclusive):
            activity_modes.append(exclusive)
            try:
                yield
            finally:
                activity_modes.pop()

    def run(*_args, **_kwargs):
        assert activity_modes == [False]
        backend_calls.append(True)
        (initialized.root / "unregistered-residue.bin").write_bytes(b"residue")
        return SimpleNamespace(
            state="incomplete",
            attempt_path=None,
            failure={"reason": "synthetic incomplete unit"},
        )

    def measure(root):
        assert Path(root) == initialized.root
        assert (initialized.root / "unregistered-residue.bin").is_file()
        assert activity_modes == [True]
        measurements.append(Path(root))
        return SimpleNamespace(total_bytes=cap + 1)

    monkeypatch.setattr(c8, "_artifact_activity_lock", observed_activity_lock)
    monkeypatch.setattr(c8, "run_execution_unit", run)
    monkeypatch.setattr(c8, "measure_artifact_tree_bytes", measure)
    with pytest.raises(
        c8.StageCBenchmarkBlockedError,
        match="online artifact cap/integrity check failed",
    ) as raised:
        c8.run_c8_shard(
            initialized.plan, output_root=initialized.root, backend=object(),
            estimate=initialized.estimate, unit_digests=[digest],
            observed_git_state=LIVE_CODE,
        )

    assert backend_calls == [True]
    assert measurements == [initialized.root]
    assert any(
        f"artifact tree bytes {cap + 1} exceed cap {cap}" in reason
        for reason in raised.value.reasons
    )
    row = _read(initialized.root / "failure_table.json")["rows"][0]
    status = _read(initialized.root / "batch_status.json")
    assert row["state"] == "tampered"
    assert row["failure"]["reason"] == "online_cap_or_integrity_failure"
    assert status["lifecycle_state"] == "blocked"
    assert status["orchestration_complete"] is False


def test_finalize_enforces_closed_layout_and_rescans_all_units(initialized):
    root = initialized.root
    digest = initialized.plan.units[0].key.canonical_digest
    _make_attempt(root, digest, number=1)
    _make_attempt(root, digest, number=2)
    with c8._artifact_activity_lock(root, exclusive=False):
        pass
    (root / "unexpected.txt").write_text("not registered\n", encoding="utf-8")

    result = c8.finalize_c8_output_root(
        root, initialized.plan, observed_git_state=LIVE_CODE
    )

    assert result["status"] == "BLOCKED"
    assert result["orchestration_complete"] is False
    assert any("expected exactly one attempt, found 2" in item for item in result["errors"])
    assert "unexpected C8 root entry: unexpected.txt" in result["errors"]
    assert not any(
        c8._ARTIFACT_ACTIVITY_LOCK_FILENAME in item for item in result["errors"]
    )
    assert any("units directory coverage mismatch" in item for item in result["errors"])
    assert result["rows"][0]["state"] == "tampered"
    assert result["rows"][1]["state"] == "missing"
    status = _read(root / "batch_status.json")
    assert status["lifecycle_state"] == "finalized"
    assert initialized.evidence["preflight"] == [False]
    assert len(initialized.evidence["live_git"]) == 1


def test_finalize_refuses_an_active_claim_without_changing_lifecycle(initialized):
    unit = initialized.plan.units[0]
    token = c8._claim_c8_unit(initialized.root, initialized.plan, unit)

    with pytest.raises(
        c8.StageCBenchmarkBlockedError, match="while execution units are claimed"
    ):
        c8.finalize_c8_output_root(
            initialized.root, initialized.plan, observed_git_state=LIVE_CODE
        )

    row = _read(initialized.root / "failure_table.json")["rows"][0]
    status = _read(initialized.root / "batch_status.json")
    assert row["state"] == "claimed"
    assert row["claim_token"] == token
    assert status["lifecycle_state"] == "running"
    assert status["active_claim_count"] == 1
    assert initialized.evidence["preflight"] == [False]


def test_forged_supplied_git_state_cannot_replace_live_repository_state(
    initialized, monkeypatch
):
    actual = {"commit": "c" * 40, "dirty": False}
    captures = []
    monkeypatch.setattr(
        c8, "capture_git_state",
        lambda root: captures.append(Path(root)) or actual,
    )
    monkeypatch.setattr(
        c8, "run_execution_unit",
        lambda *_args, **_kwargs: pytest.fail("backend was called"),
    )

    with pytest.raises(
        c8.StageCBenchmarkIntegrityError,
        match="supplied C8 Git state differs from the live repository",
    ):
        c8.run_c8_shard(
            initialized.plan, output_root=initialized.root, backend=object(),
            estimate=initialized.estimate,
            unit_digests=[initialized.plan.units[0].key.canonical_digest],
            observed_git_state=LIVE_CODE,
        )

    assert len(captures) == 1
    assert _read(initialized.root / "failure_table.json")["rows"][0][
        "state"
    ] == "pending"


def test_checkpoint_lifecycle_and_activity_lock_symlinks_are_rejected(
    initialized, monkeypatch, tmp_path
):
    root = initialized.root
    unit = initialized.plan.units[0]
    target = tmp_path / "outside-lock-target"
    target.write_text("unchanged", encoding="utf-8")
    checkpoint = root / c8._CHECKPOINT_LOCK_FILENAME
    checkpoint.symlink_to(target)

    with pytest.raises(
        c8.StageCBenchmarkIntegrityError,
        match="cannot safely open C8 checkpoint lock",
    ):
        c8._claim_c8_unit(root, initialized.plan, unit)
    assert target.read_text(encoding="utf-8") == "unchanged"

    checkpoint.unlink()
    lifecycle = root / c8._LIFECYCLE_LOCK_FILENAME
    lifecycle.symlink_to(target)
    monkeypatch.setattr(
        c8, "run_execution_unit",
        lambda *_args, **_kwargs: pytest.fail("backend was called"),
    )
    with pytest.raises(
        c8.StageCBenchmarkIntegrityError,
        match="cannot safely open C8 lifecycle lock",
    ):
        c8.run_c8_shard(
            initialized.plan, output_root=root, backend=object(),
            estimate=initialized.estimate,
            unit_digests=[unit.key.canonical_digest],
            observed_git_state=LIVE_CODE,
        )
    assert target.read_text(encoding="utf-8") == "unchanged"

    lifecycle.unlink()
    activity = root / c8._ARTIFACT_ACTIVITY_LOCK_FILENAME
    activity.symlink_to(target)
    with pytest.raises(
        c8.StageCBenchmarkIntegrityError,
        match="cannot safely open C8 artifact-activity lock",
    ):
        with c8._artifact_activity_lock(root, exclusive=False):
            pytest.fail("activity lock symlink was followed")
    assert target.read_text(encoding="utf-8") == "unchanged"


def test_fixed_report_hash_is_checked_before_semantics(monkeypatch, tmp_path):
    path = tmp_path / "report.json"
    path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(c8, "C7_REPORT_FILE_SHA256", "f" * 64)
    with pytest.raises(c8.StageCBenchmarkIntegrityError, match="file SHA-256"):
        c8.validate_fixed_c7_report(path)


def test_recovery_preflight_uses_authority_bound_report_values(
    monkeypatch, tmp_path, exact,
):
    config, plan = exact
    binding = load_recovery_v2_binding(verify_assets=False)
    record, report = _recovery_record(tmp_path, binding)
    captured = {}
    expected = SimpleNamespace(
        within_budget=True, violations=(), canonical_digest="f" * 64,
        compute={
            "projected_h20_gpu_hours":
                RECOVERY_PROJECTIONS["c7_projected_c8_h20_gpu_hours"],
        },
    )
    monkeypatch.setattr(
        recovery_c8_launch, "load_recovery_c8_launch_record",
        lambda *_args, **_kwargs: record,
    )

    def fake_estimate(config_arg, plan_arg, **kwargs):
        assert config_arg is config and plan_arg is plan
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(c8, "estimate_stage_c_run", fake_estimate)
    assert c8.build_c8_measured_preflight(
        plan, config, report, trusted_recovery_context=record,
    ) is expected
    assert captured == {
        "measured_slowest_unit_seconds":
            RECOVERY_PROJECTIONS["c7_slowest_unit_seconds"],
        "measured_peak_reserved_bytes_per_unit":
            RECOVERY_PROJECTIONS["c7_peak_reserved_bytes"],
    }


def test_recovery_authorization_projects_typed_authority_without_archive(
    monkeypatch, tmp_path,
):
    binding = load_recovery_v2_binding(verify_assets=False)
    record, report = _recovery_record(tmp_path, binding)
    plan = SimpleNamespace(canonical_digest="f" * 64)
    estimate = SimpleNamespace(canonical_digest="0" * 64)
    monkeypatch.setattr(
        recovery_c8_launch, "load_recovery_c8_launch_record",
        lambda *_args, **_kwargs: record,
    )
    validations = []
    monkeypatch.setattr(
        c8, "_validate_formal_preflight",
        lambda *args, **kwargs: validations.append((args, kwargs)),
    )

    authorization = c8._recovery_authorization(
        record, report, plan, estimate, record.formal_root, RECOVERY_CODE,
    )
    c8.validate_c8_launch_authorization(
        authorization, plan=plan, estimate=estimate,
        output_root=record.formal_root, trusted_recovery_context=record,
    )

    assert authorization["authority_kind"] == "recovery_c8_launch_record"
    assert authorization["projections"] == RECOVERY_PROJECTIONS
    assert authorization["source"]["execution_git_state"] == RECOVERY_CODE
    assert not any("archive" in key for key in authorization)
    assert not Path(
        authorization["recovery_c8_launch_record"]["path"]
    ).is_absolute()
    assert validations[-1][1]["projections"] == RECOVERY_PROJECTIONS

    damaged = json.loads(json.dumps(authorization))
    damaged["projections"]["c7_slowest_unit_seconds"] += 1.0
    damaged["canonical_digest"] = canonical_sha256({
        key: value for key, value in damaged.items()
        if key != "canonical_digest"
    })
    with pytest.raises(
        c8.StageCBenchmarkIntegrityError, match="report projections drifted"
    ):
        c8.validate_c8_launch_authorization(
            damaged, plan=plan, estimate=estimate,
            output_root=record.formal_root, trusted_recovery_context=record,
        )


def test_all_c8_attempt_validation_paths_receive_recovery_context(
    monkeypatch, tmp_path,
):
    digest = "a" * 64
    identity = {
        "schema_version": 1, "sample_id": "sample",
        "method_id": "method", "config_id": "config",
        "block_index": None, "head_index": None,
        "target_ref": None, "canonical_digest": digest,
    }
    key = SimpleNamespace(
        canonical_digest=digest, to_dict=lambda: dict(identity),
    )
    unit = SimpleNamespace(key=key)
    plan = SimpleNamespace(
        canonical_digest="b" * 64, units=(unit,),
        run_spec=SimpleNamespace(to_dict=lambda: {"schema_version": 1}),
    )
    caps = {
        "bytes_per_l0_l1_execution_unit_max": 1,
        "bytes_per_audit_sample_l2_max": 1,
        "stage_c_c8_c9_total_artifact_bytes_max": 1,
    }
    estimate = SimpleNamespace(
        canonical_digest="c" * 64, caps={"effective": caps},
    )
    attempt = _make_attempt(tmp_path, digest)
    context = object()
    received = []

    def validate(*_args, **kwargs):
        received.append(kwargs.get("trusted_recovery_context"))
        return _valid_attempt()

    monkeypatch.setattr(stage_c_artifacts, "validate_execution_unit", validate)
    monkeypatch.setattr(
        stage_c_artifacts, "attempt_directories", lambda *_args: (attempt,),
    )
    monkeypatch.setattr(
        c8, "measure_artifact_tree_bytes",
        lambda *_args: SimpleNamespace(total_bytes=0),
    )
    c8._result_row(
        unit, SimpleNamespace(
            state="succeeded", attempt_path=str(attempt), failure=None,
        ), claim_token="d" * 64, caps=caps, root=tmp_path, plan=plan,
        estimate=estimate, observed_git_state=LIVE_CODE,
        trusted_recovery_context=context,
    )
    c8._selected_attempt_totals(
        tmp_path, [{"selected_attempt": str(attempt), "unit_key": identity}],
        caps, trusted_recovery_context=context,
    )
    c8._scan_rows(
        tmp_path, plan, estimate, observed_git_state=LIVE_CODE,
        trusted_recovery_context=context,
    )
    assert received == [context, context, context, context]
