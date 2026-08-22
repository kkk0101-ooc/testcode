from __future__ import annotations

import copy
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import vitnns.stage_c_benchmark as benchmark
from vitnns.constants import MODEL_NAME, MODEL_SHA256, REPO_ROOT
from vitnns.stage_c_config import load_stage_c_config
from vitnns.stage_c_plan import (
    CLASS_METHOD_IDS,
    VALUE_METHOD_IDS,
    build_execution_plan,
    estimate_stage_c_run,
    load_run_spec,
    load_stage_c_registry,
)


FORMAL_SPEC_PATH = REPO_ROOT / "config/stage_c_c7_run_spec.json"
DIAGNOSTIC_SPEC_PATH = (
    REPO_ROOT / "config/stage_c_c7_diagnostic_run_spec.json"
)
FORMAL_CANDIDATES = (
    (0, 0),
    (1, 5),
    (2, 11),
    (3, 5),
    (4, 11),
    (5, 0),
    (6, 11),
    (7, 0),
    (8, 5),
    (9, 0),
    (10, 5),
    (11, 11),
)
FORMAL_COUNTS = {
    "execution_units": 40,
    "map_runs": 200,
    "fidelity_eligible_units": 28,
    "fidelity_curves": 1232,
    "fidelity_curve_points": 25872,
    "work_items": 26072,
}
DIAGNOSTIC_COUNTS = {
    "execution_units": 1,
    "map_runs": 5,
    "fidelity_eligible_units": 1,
    "fidelity_curves": 44,
    "fidelity_curve_points": 924,
    "work_items": 929,
}
DIAGNOSTIC_METHOD = (
    "conditional_fixed_clean_attention_head_direction_value_path_ig"
)
ABANDONED_DIAGNOSTIC_COMMIT = (
    "357dd9d4a8b6dd2d43ff5bfbc85cc93a68145226"
)
NEW_TEST_COMMIT = "6a1b9c2d3e4f5061728394a5b6c7d8e9f0011223"
ALL_METHODS = VALUE_METHOD_IDS + CLASS_METHOD_IDS
IG_METHODS = {
    "conditional_fixed_clean_attention_head_direction_value_path_ig",
    "full_model_class_logit_ig",
}


@pytest.fixture(scope="module")
def config():
    return load_stage_c_config()


@pytest.fixture(scope="module")
def registry(config):
    return load_stage_c_registry(config)


@pytest.fixture(scope="module")
def formal_plan(config, registry):
    return build_execution_plan(config, registry, load_run_spec(FORMAL_SPEC_PATH))


@pytest.fixture(scope="module")
def diagnostic_plan(config, registry):
    return build_execution_plan(
        config, registry, load_run_spec(DIAGNOSTIC_SPEC_PATH)
    )


@pytest.fixture(scope="module")
def formal_estimate(config, formal_plan):
    return estimate_stage_c_run(config, formal_plan)


def _aggregate_plan_counts(plan: Any) -> dict[str, int]:
    work = Counter()
    for unit in plan.units:
        work.update(unit.work_item_counts)
    return {
        "execution_units": len(plan.units),
        "map_runs": work["map_runs"],
        "fidelity_eligible_units": sum(
            unit.fidelity_eligible for unit in plan.units
        ),
        "fidelity_curves": work["fidelity_curves"],
        "fidelity_curve_points": work["fidelity_curve_points"],
        "work_items": work["work_items"],
    }


def _candidate_pairs(plan: Any) -> tuple[tuple[int, int], ...]:
    return tuple(
        (candidate.block_index, candidate.head_index)
        for candidate in plan.run_spec.value_candidates
    )


def test_checked_in_formal_spec_is_exact_and_expands_frozen_scope(
    config, formal_plan
):
    scope = benchmark.validate_c7_execution_plan(formal_plan, config)

    assert _candidate_pairs(formal_plan) == FORMAL_CANDIDATES
    assert tuple(formal_plan.run_spec.method_ids) == ALL_METHODS
    assert _aggregate_plan_counts(formal_plan) == FORMAL_COUNTS
    assert scope["valid"] is True
    assert scope["violations"] == []
    assert scope["counts"] == FORMAL_COUNTS
    assert benchmark.C7_EXPECTED_COUNTS == FORMAL_COUNTS
    assert tuple(benchmark.C7_VALUE_CANDIDATES) == FORMAL_CANDIDATES


def test_checked_in_diagnostic_spec_is_exact_one_nns_full_workload(
    config, diagnostic_plan
):
    validate = getattr(
        benchmark,
        "validate_c7_diagnostic_execution_plan",
        benchmark.validate_c7_diagnostic_plan,
    )
    scope = validate(diagnostic_plan, config)
    unit = diagnostic_plan.units[0]

    assert diagnostic_plan.run_spec.sample_ids == (
        "ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG",
    )
    assert diagnostic_plan.run_spec.method_ids == (DIAGNOSTIC_METHOD,)
    assert _candidate_pairs(diagnostic_plan) == ((0, 0),)
    assert unit.key.method_id == DIAGNOSTIC_METHOD
    assert (unit.key.block_index, unit.key.head_index) == (0, 0)
    assert _aggregate_plan_counts(diagnostic_plan) == DIAGNOSTIC_COUNTS
    assert scope["valid"] is True
    assert scope["violations"] == []
    assert scope["counts"] == DIAGNOSTIC_COUNTS


def test_formal_and_diagnostic_validators_do_not_accept_each_others_scope(
    config, formal_plan, diagnostic_plan
):
    validate_diagnostic = getattr(
        benchmark,
        "validate_c7_diagnostic_execution_plan",
        benchmark.validate_c7_diagnostic_plan,
    )

    formal_as_diagnostic = validate_diagnostic(formal_plan, config)
    diagnostic_as_formal = benchmark.validate_c7_execution_plan(
        diagnostic_plan, config
    )

    assert formal_as_diagnostic["valid"] is False
    assert formal_as_diagnostic["violations"]
    assert diagnostic_as_formal["valid"] is False
    assert diagnostic_as_formal["violations"]


def test_frozen_downstream_counts_factors_and_caps_match_config(config):
    assert benchmark.C8_EXECUTION_UNITS == 5 * (144 * 3 + 4) == 2180
    assert benchmark.C9_EXECUTION_UNITS == 20 * (4 * 3 + 4) == 320
    assert benchmark.C8_C9_EXECUTION_UNITS == 2180 + 320 == 2500
    assert benchmark.STAGE_D_EXECUTION_UNITS == 500 * (4 * 3 + 4) == 8000
    assert benchmark.DIAGNOSTIC_WALL_LIMIT_SECONDS == 720

    caps = dict(benchmark.FROZEN_CAPS)
    configured = config.budget.to_dict()
    assert caps == {key: configured[key] for key in caps}
    assert caps["projection_safety_factor"] == 1.25
    assert caps["storage_control_overhead_factor"] == 1.25


def _runner_telemetry(
    *, total_wall: float = 720.0, lifecycle_restored: bool = True
) -> dict[str, Any]:
    digest = "d" * 64
    return {
        "schema_version": 1,
        "phase_timings": {
            "clock": "perf_counter",
            "synchronization": "device_before_and_after_each_phase",
            "phases_are_mutually_exclusive": False,
            "total_boundary": benchmark.RUNNER_TOTAL_BOUNDARY,
            "synchronized_total_wall_time_seconds": total_wall,
            "phases": {
                name: {"wall_time_seconds": 1.0, "invocations": 1}
                for name in benchmark.RUNNER_PHASE_KEYS
            },
        },
        "lifecycle": {
            "integrity": {
                "schema_version": 1,
                "algorithm": "stage_c_strong_model_sha256_v1",
                "checked": ["parameter_and_persistent_buffer_content"],
                "entry_sha256": digest,
                "exit_sha256": digest if lifecycle_restored else "e" * 64,
                "matched": lifecycle_restored,
                "restored": lifecycle_restored,
                "differences": [] if lifecycle_restored else ["parameter:weight"],
                "signature_calls": 2,
                "verification_error_type": None,
                "verification_error_message": None,
                "body_exception_type": None,
            },
            "restoration": {
                "schema_version": 1,
                "checked": ["module_modes", "hooks", "grad", "requires_grad"],
                "changes_detected": False,
                "restored": lifecycle_restored,
                "differences": [],
                "errors": [],
            },
        },
        "measurement_environment": {
            "device_type": "cuda",
            "device": "cuda:0",
            "device_name": "NVIDIA H20",
            "cuda_runtime_version": "12.1",
            "cudnn_version": 8902,
        },
    }


def _projection_record(
    *,
    method_id: str = DIAGNOSTIC_METHOD,
    total_wall: float = 720.0,
    backend_wall: float = 1.0,
    forward_calls: int = 31,
    backward_calls: int = 29,
    peak_reserved_bytes: int = 43,
    non_l2_bytes: int = 17,
    l2_bytes: int = 19,
) -> dict[str, Any]:
    return {
        "state": "succeeded",
        "trusted_by_batch_validator": True,
        "unit_key": {
            "sample_id": "toy-audit-sample",
            "method_id": method_id,
            "canonical_digest": "c" * 64,
        },
        "attempt_path": "/tmp/new-c7/units/u/attempts/attempt-000001",
        "resources": {
            # This deliberately differs from the runner total.  It represents
            # the old backend-only meter and must not drive formal projection.
            "wall_time_seconds": backend_wall,
            "calls": {
                "accounting_unit": "batched_model_invocation",
                "forward_calls": forward_calls,
                "backward_calls": backward_calls,
            },
            "cuda": {
                "status": "measured",
                "device": "cuda:0",
                "peak_allocated_bytes": peak_reserved_bytes // 2,
                "peak_reserved_bytes": peak_reserved_bytes,
            },
        },
        "runner_telemetry": _runner_telemetry(total_wall=total_wall),
        "work_item_counts": {"work_items": 929, "map_runs": 5},
        "l0_l1_cap_accounting_bytes": non_l2_bytes,
        "level_bytes": {
            "L0": 2,
            "L1": 3,
            "L2": l2_bytes,
            "L3": 5,
            "control": 7,
        },
        "total_regular_file_bytes": non_l2_bytes + l2_bytes,
    }


def _wide_caps() -> dict[str, int | float]:
    result = {key: 10**30 for key in benchmark.HARD_CAP_KEYS}
    result.update({
        "projection_safety_factor": 1.25,
        "storage_control_overhead_factor": 1.25,
    })
    return result


def _project(
    record: dict[str, Any], caps: dict[str, int | float] | None = None
) -> dict[str, Any]:
    return benchmark.project_benchmark_budget(
        [record],
        caps=_wide_caps() if caps is None else caps,
        c6_storage_payload_bytes=0,
        source_batch_regular_file_bytes=record["total_regular_file_bytes"],
        source_unit_count=1,
        source_batch_control_bytes=0,
    )


def _cap_check(projection: dict[str, Any], key: str) -> dict[str, Any]:
    return next(
        item for item in projection["hard_cap_checks"]
        if item["cap_key"] == key
    )


def test_projection_uses_exact_counts_factor_and_synchronized_runner_total():
    record = _projection_record(total_wall=720.0, backend_wall=1.0)
    projection = _project(record)

    assert projection["exact_unit_counts"] == {
        "C8": 2180,
        "C9": 320,
        "C8_C9": 2500,
        "StageD": 8000,
        "StageD_images": 500,
    }
    assert projection["time"]["projection_safety_factor"] == 1.25
    assert projection["time"]["slowest_valid_measured_unit_seconds"] == 720.0
    assert projection["time"]["C8_h20_gpu_hours"] == 545.0
    assert projection["time"]["C9_h20_gpu_hours"] == 80.0
    assert projection["time"]["C8_C9_h20_gpu_hours"] == 625.0
    assert projection["time"]["StageD_h20_gpu_hours"] == 2000.0
    assert projection["time"]["StageD_h20_gpu_hours_per_image"] == 4.0
    assert projection["time"]["measurement_source"].startswith(
        "status.runner_telemetry"
    )


@pytest.mark.parametrize("cap_key", benchmark.HARD_CAP_KEYS)
def test_every_hard_cap_allows_equality_and_blocks_one_over(cap_key):
    record = _projection_record()
    baseline = _project(record)
    observed = _cap_check(baseline, cap_key)["observed"]
    assert type(observed) in {int, float} and observed > 1

    equality_caps = _wide_caps()
    equality_caps[cap_key] = observed
    equality = _cap_check(_project(record, equality_caps), cap_key)
    assert equality["comparison"] == "observed <= cap"
    assert equality["observed"] == equality["cap"]
    assert equality["status"] == benchmark.STATUS_PASS

    one_over_caps = _wide_caps()
    one_over_caps[cap_key] = observed - 1
    one_over = _cap_check(_project(record, one_over_caps), cap_key)
    assert one_over["observed"] == one_over["cap"] + 1
    assert one_over["status"] == benchmark.STATUS_BLOCKED


@pytest.mark.parametrize(
    ("damage", "blocked_cap_keys"),
    [
        (
            lambda record: record["resources"].update({
                "cuda": {
                    "status": "measurement_failed",
                    "device": "cuda:0",
                    "peak_allocated_bytes": None,
                    "peak_reserved_bytes": None,
                }
            }),
            {"cuda_peak_reserved_bytes_per_execution_unit_max"},
        ),
        (
            lambda record: record.update({"runner_telemetry": None}),
            {
                "stage_c_c8_c9_h20_gpu_hours_max",
                "stage_d_500_h20_gpu_hours_max",
                "stage_d_h20_gpu_hours_per_image_max",
            },
        ),
        (
            lambda record: record["runner_telemetry"].update({
                "lifecycle": None
            }),
            {
                "stage_c_c8_c9_h20_gpu_hours_max",
                "stage_d_500_h20_gpu_hours_max",
                "stage_d_h20_gpu_hours_per_image_max",
            },
        ),
        (
            lambda record: record["resources"].update({"calls": {}}),
            {
                "forward_calls_per_execution_unit_max",
                "backward_calls_per_execution_unit_max",
            },
        ),
    ],
    ids=("cuda", "phase-telemetry", "lifecycle", "calls"),
)
def test_missing_measurement_evidence_fails_closed(damage, blocked_cap_keys):
    record = _projection_record()
    damage(record)
    projection = _project(record)

    for key in blocked_cap_keys:
        check = _cap_check(projection, key)
        assert check["measured"] is False
        assert check["status"] == benchmark.STATUS_BLOCKED
    assert projection["all_hard_caps_pass"] is False


H20_ENVIRONMENT = {
    "schema_version": 1,
    "hostname": "toy-host",
    "platform": "toy-platform",
    "python_version": "3.10.20",
    "torch_version": "2.2.2+cu121",
    "device_type": "cuda",
    "device": "cuda:0",
    "cuda_available": True,
    "device_name": "NVIDIA H20",
    "cuda_device_name": "NVIDIA H20",
    "cuda_total_memory_bytes": 96 * 1024**3,
    "cuda_compute_capability": [9, 0],
    "cuda_runtime_version": "12.1",
    "cudnn_version": 8902,
    "cuda_driver_version": "535.104.05",
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cublas_workspace_config": ":4096:8",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "seed": 20260821,
}


def _ig_result() -> dict[str, Any]:
    return {
        "selected_steps": 512,
        "converged": True,
        "attempts": [{"steps": 512, "pass": [True]}],
        "metadata": {"step_chunk_size": 8},
    }


def _map_records(method_id: str) -> list[dict[str, Any]]:
    variants = (
        ("clean", 0),
        ("translate_right_down_4", None),
        ("center_crop_216_resize_224", None),
        ("gaussian_rgb_std_0_01", None),
        ("repeat_clean", 1),
    )
    return [
        {
            "method_id": method_id,
            "input_variant": variant,
            "repeat_index": repeat_index,
            "status": "completed",
            "ig_result": _ig_result() if method_id in IG_METHODS else None,
            "attempted_steps_total": (
                512 if method_id in IG_METHODS else None
            ),
            "backward_invocations": (
                64 if method_id in IG_METHODS else None
            ),
        }
        for variant, repeat_index in variants
    ]


def _shared_provenance(plan_digest: str | None = None) -> dict[str, Any]:
    digest = "a" * 64
    return {
        "code": {"commit": NEW_TEST_COMMIT, "dirty": False},
        "model": {
            "name": MODEL_NAME,
            "state_dict_sha256": MODEL_SHA256,
        },
        "input": {"image_sha256": digest},
        "dependencies": {"lock_sha256": digest},
        "registry": {"sha256": digest},
        "protocol": {"sha256": digest},
        "acceptance": {"sha256": digest},
        "configuration": {
            "sha256": digest,
            **(
                {} if plan_digest is None
                else {
                    "execution_plan_sha256": plan_digest,
                    "run_spec_sha256": None,
                }
            ),
        },
    }


def _diagnostic_mapping(plan: Any, wall: float, root: Path) -> dict[str, Any]:
    record = _projection_record(total_wall=wall)
    record["unit_key"] = plan.units[0].key.to_dict()
    record["attempt_path"] = str(
        (
            root / "units" / plan.units[0].key.canonical_digest
            / "attempts" / "attempt-000001"
        ).resolve()
    )
    record["map_execution_records"] = _map_records(DIAGNOSTIC_METHOD)
    record["shared_provenance"] = _shared_provenance(plan.canonical_digest)
    record["shared_provenance"]["configuration"]["run_spec_sha256"] = (
        benchmark._canonical_digest(plan.run_spec.to_dict())
    )
    record["shared_provenance_sha256"] = benchmark._canonical_digest(
        record["shared_provenance"]
    )
    validation = {
        "schema_version": 1,
        "document_type": "stage_c_batch_validation",
        "plan_digest": plan.canonical_digest,
        "valid": True,
        "orchestration_complete": True,
        "successful": True,
        "rows": [],
        "errors": [],
    }
    return {
        "source_root": str(root.resolve()),
        "batch_validation": validation,
        "validation_exception": None,
        "controls": {
            name: {
                "path": str((root / name).resolve()),
                "present": True,
                "bytes": 1,
                "sha256": "b" * 64,
                "document": {},
                "error": None,
            }
            for name in (
                "execution_plan.json",
                "preflight.json",
                "failure_table.json",
                "batch_status.json",
            )
        },
        "records": [record],
        "mapping_errors": [],
    }


def _diagnostic_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, diagnostic_plan: Any,
    config: Any, *, wall: float, mutate_record=None, environment=None,
) -> dict[str, Any]:
    root = tmp_path / "diagnostic-source"
    mapping = _diagnostic_mapping(diagnostic_plan, wall, root)
    if mutate_record is not None:
        mutate_record(mapping["records"][0])
    monkeypatch.setattr(benchmark, "map_execution_records", lambda *_a, **_k: mapping)
    monkeypatch.setattr(benchmark, "_source_layout_errors", lambda *_a, **_k: [])
    monkeypatch.setattr(
        benchmark, "_measure_source_tree",
        lambda _root: (
            1234,
            4,
            [
                {
                    "path": "synthetic-evidence.json",
                    "bytes": 1234,
                    "sha256": "f" * 64,
                }
            ],
            [],
        ),
    )
    return benchmark.build_c7_diagnostic_gate(
        root,
        diagnostic_plan,
        config=config,
        environment=H20_ENVIRONMENT if environment is None else environment,
        run_spec_path=DIAGNOSTIC_SPEC_PATH,
        run_spec_file_sha256=hashlib.sha256(
            DIAGNOSTIC_SPEC_PATH.read_bytes()
        ).hexdigest(),
        observed_git_state={"commit": NEW_TEST_COMMIT, "dirty": False},
        report_path=tmp_path / "diagnostic-gate.json",
    )


@pytest.mark.parametrize(
    ("wall", "expected_status"),
    [
        (719.999, benchmark.STATUS_PASS),
        (720.0, benchmark.STATUS_BLOCKED),
        (720.001, benchmark.STATUS_BLOCKED),
    ],
)
def test_diagnostic_gate_uses_exclusive_720_boundary(
    monkeypatch, tmp_path, diagnostic_plan, config, wall, expected_status
):
    report = _diagnostic_report(
        monkeypatch, tmp_path, diagnostic_plan, config, wall=wall
    )

    assert report["status"] == expected_status
    assert report["diagnostic_measurement"] == {
        "source": (
            "status.runner_telemetry.phase_timings."
            "synchronized_total_wall_time_seconds"
        ),
        "synchronized_total_wall_time_seconds": wall,
        "limit_seconds_exclusive": 720.0,
        "comparison": "wall_time_seconds < 720",
        "status": expected_status,
    }
    assert report["exit_class"] == (
        benchmark.EXIT_CLASS_SUCCESS
        if expected_status == benchmark.STATUS_PASS
        else benchmark.EXIT_CLASS_PREFLIGHT
    )
    benchmark.validate_c7_diagnostic_gate(report)


@pytest.mark.parametrize(
    ("case", "mutate_record", "environment"),
    [
        ("phase", lambda record: record.update({"runner_telemetry": None}), None),
        (
            "lifecycle",
            lambda record: record["runner_telemetry"].update({
                "lifecycle": None
            }),
            None,
        ),
        (
            "ig",
            lambda record: record.update({"map_execution_records": []}),
            None,
        ),
        (
            "cuda",
            None,
            {**H20_ENVIRONMENT, "cuda_available": False},
        ),
    ],
)
def test_diagnostic_gate_missing_required_evidence_is_fail_closed(
    monkeypatch, tmp_path, diagnostic_plan, config, case, mutate_record,
    environment,
):
    report = _diagnostic_report(
        monkeypatch, tmp_path, diagnostic_plan, config, wall=1.0,
        mutate_record=mutate_record, environment=environment,
    )

    assert report["status"] == benchmark.STATUS_BLOCKED, case
    assert report["blocked_reasons"], case
    assert report["blocking_kind"] in {"integrity", "performance"}
    benchmark.validate_c7_diagnostic_gate(report)


def _resign(document: dict[str, Any]) -> dict[str, Any]:
    document["canonical_digest"] = benchmark._self_digest(document)
    return document


def test_diagnostic_gate_detects_plain_tamper_and_resigned_semantic_forgery(
    monkeypatch, tmp_path, diagnostic_plan, config
):
    report = _diagnostic_report(
        monkeypatch, tmp_path, diagnostic_plan, config, wall=12.5
    )
    tampered = copy.deepcopy(report)
    tampered["diagnostic_measurement"][
        "synchronized_total_wall_time_seconds"
    ] = 12.0
    with pytest.raises(
        benchmark.StageCBenchmarkIntegrityError, match="canonical_digest mismatch"
    ):
        benchmark.validate_c7_diagnostic_gate(tampered)

    forged = _resign(tampered)
    with pytest.raises(
        benchmark.StageCBenchmarkIntegrityError,
        match="semantic|measurement|runner",
    ):
        benchmark.validate_c7_diagnostic_gate(forged)


def test_fresh_output_root_rejects_old_retry_and_nonempty_paths(tmp_path):
    assert benchmark.require_fresh_c7_output_root(
        tmp_path / "new-formal-root", kind="formal"
    ) == (tmp_path / "new-formal-root").resolve()

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "old-evidence.json").write_text("{}", encoding="utf-8")
    with pytest.raises(benchmark.StageCBenchmarkBlockedError, match="not fresh"):
        benchmark.require_fresh_c7_output_root(nonempty, kind="formal")

    polluted = tmp_path / "formal-retry01"
    with pytest.raises(
        benchmark.StageCBenchmarkBlockedError, match="old-attempt"
    ):
        benchmark.require_fresh_c7_output_root(polluted, kind="formal")


def _formal_mapping(plan: Any, estimate: Any, root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    shared = _shared_provenance(plan.canonical_digest)
    shared["configuration"]["run_spec_sha256"] = benchmark._canonical_digest(
        plan.run_spec.to_dict()
    )
    shared_digest = benchmark._canonical_digest(shared)
    for index, unit in enumerate(plan.units):
        record = _projection_record(
            method_id=unit.key.method_id,
            total_wall=1.0 + index / 100.0,
            backend_wall=10_000.0 + index,
        )
        record.update({
            "unit_key": unit.key.to_dict(),
            "attempt_path": str((
                root / "units" / unit.key.canonical_digest
                / "attempts" / "attempt-000001"
            ).resolve()),
            "work_item_counts": unit.work_item_counts,
            "map_execution_records": _map_records(unit.key.method_id),
            "shared_provenance": copy.deepcopy(shared),
            "shared_provenance_sha256": shared_digest,
            "hashes": {
                "manifest_sha256": "1" * 64,
                "status_sha256": "2" * 64,
                "selected_status_sha256": "2" * 64,
                "provenance_sha256": "3" * 64,
                "result_sha256": "4" * 64,
                "unit_identity_sha256": unit.key.canonical_digest,
            },
        })
        records.append(record)
    validation = {
        "schema_version": 1,
        "document_type": "stage_c_batch_validation",
        "plan_digest": plan.canonical_digest,
        "valid": True,
        "orchestration_complete": True,
        "successful": True,
        "rows": [],
        "errors": [],
    }
    controls = {
        name: {
            "path": str((root / name).resolve()),
            "present": True,
            "bytes": 1,
            "sha256": "5" * 64,
            "document": (
                estimate.to_dict() if name == "preflight.json" else {}
            ),
            "error": None,
        }
        for name in (
            "execution_plan.json",
            "preflight.json",
            "failure_table.json",
            "batch_status.json",
        )
    }
    return {
        "source_root": str(root.resolve()),
        "batch_validation": validation,
        "validation_exception": None,
        "controls": controls,
        "records": records,
        "mapping_errors": [],
    }


def _formal_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, formal_plan: Any,
    formal_estimate: Any, config: Any, *, mapping=None, source_bytes=4000,
    layout_errors=(),
) -> dict[str, Any]:
    root = tmp_path / "formal-source"
    diagnostic_plan = build_execution_plan(
        config, load_stage_c_registry(config),
        load_run_spec(DIAGNOSTIC_SPEC_PATH),
    )
    gate = _diagnostic_report(
        monkeypatch, tmp_path, diagnostic_plan, config, wall=1.0
    )
    gate_path = tmp_path / "diagnostic-gate-for-formal.json"
    benchmark._atomic_write_json(gate_path, gate)
    # The diagnostic helper patches report-only readers; restore before the
    # formal helper installs its own source evidence.
    monkeypatch.undo()
    authorization = benchmark.require_c7_launch_gate(
        gate_path, formal_plan, config=config, output_root=root,
        observed_git_state={"commit": NEW_TEST_COMMIT, "dirty": False},
        run_spec_path=FORMAL_SPEC_PATH,
    )
    benchmark.write_c7_launch_authorization(root, authorization)
    evidence = (
        _formal_mapping(formal_plan, formal_estimate, root)
        if mapping is None else mapping
    )
    monkeypatch.setattr(
        benchmark, "map_execution_records", lambda *_a, **_k: evidence
    )
    monkeypatch.setattr(
        benchmark, "_source_layout_errors",
        lambda *_a, **_k: list(layout_errors),
    )
    if source_bytes is not None:
        monkeypatch.setattr(
            benchmark, "_measure_source_tree",
            lambda _root: (
                source_bytes,
                sum(
                    item["bytes"]
                    for item in {
                        **evidence["controls"],
                        benchmark.LAUNCH_AUTHORIZATION_FILENAME: {
                            "bytes": (
                                root / benchmark.LAUNCH_AUTHORIZATION_FILENAME
                            ).stat().st_size
                        },
                    }.values()
                ),
                [
                    {
                        "path": "synthetic-evidence.json",
                        "bytes": source_bytes,
                        "sha256": "f" * 64,
                    }
                ],
                [],
            ),
        )
    return benchmark.build_benchmark_report(
        root,
        formal_plan,
        config=config,
        environment=H20_ENVIRONMENT,
        run_spec_path=FORMAL_SPEC_PATH,
        run_spec_file_sha256=hashlib.sha256(
            FORMAL_SPEC_PATH.read_bytes()
        ).hexdigest(),
        recomputed_estimate=formal_estimate,
        observed_git_state={"commit": NEW_TEST_COMMIT, "dirty": False},
        c7_diagnostic_gate=gate_path,
        report_path=tmp_path / "formal-report.json",
    )


def test_formal_report_uses_status_runner_total_not_backend_only_wall(
    monkeypatch, tmp_path, formal_plan, formal_estimate, config
):
    report = _formal_report(
        monkeypatch, tmp_path, formal_plan, formal_estimate, config
    )

    assert report["status"] == benchmark.STATUS_PASS
    assert report["integrity_valid"] is True
    assert report["blocked_reasons"] == []
    assert len(report["unit_records"]) == 40
    slowest = report["projections"]["time"]
    assert slowest["slowest_valid_measured_unit_seconds"] == pytest.approx(1.39)
    assert slowest["slowest_valid_measured_unit"][
        "unit_key"
    ] == formal_plan.units[-1].key.to_dict()
    assert slowest["measurement_source"] == (
        "status.runner_telemetry.phase_timings."
        "synchronized_total_wall_time_seconds"
    )
    assert max(
        record["resources"]["wall_time_seconds"]
        for record in report["unit_records"]
    ) > 10_000
    assert report["selection_performed"] is False
    assert report["bootstrap_performed"] is False
    assert report["decision_performed"] is False
    assert report["holdout_accessed"] is False
    benchmark.validate_benchmark_report(report)


def test_formal_report_detects_plain_tamper_and_resigned_semantic_forgery(
    monkeypatch, tmp_path, formal_plan, formal_estimate, config
):
    report = _formal_report(
        monkeypatch, tmp_path, formal_plan, formal_estimate, config
    )
    tampered = copy.deepcopy(report)
    tampered["scope"]["counts"]["execution_units"] = 39
    with pytest.raises(
        benchmark.StageCBenchmarkIntegrityError, match="canonical_digest mismatch"
    ):
        benchmark.validate_benchmark_report(tampered)

    forged = _resign(tampered)
    with pytest.raises(
        benchmark.StageCBenchmarkIntegrityError, match="semantic validation"
    ):
        benchmark.validate_benchmark_report(forged)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["projections"]["exact_unit_counts"].update(
            {"C8": 2179}
        ),
        lambda report: report["projections"]["hard_cap_checks"][0].update(
            {"cap": benchmark.FROZEN_CAPS[benchmark.HARD_CAP_KEYS[0]] + 1}
        ),
        lambda report: report["unit_records"][0].update({
            "attempt_path": (
                "/synthetic/archive/c7-overhash-357dd9d/"
                "attempt-20260822"
            )
        }),
    ],
    ids=("downstream-count", "frozen-cap", "old-attempt-path"),
)
def test_formal_report_rejects_resigned_derived_or_path_forgery(
    monkeypatch, tmp_path, formal_plan, formal_estimate, config, mutate
):
    report = _formal_report(
        monkeypatch, tmp_path, formal_plan, formal_estimate, config
    )
    mutate(report)
    _resign(report)

    with pytest.raises(
        benchmark.StageCBenchmarkIntegrityError, match="semantic validation"
    ):
        benchmark.validate_benchmark_report(report)


def test_formal_report_rejects_resigned_old_commit(
    monkeypatch, tmp_path, formal_plan, formal_estimate, config
):
    report = _formal_report(
        monkeypatch, tmp_path, formal_plan, formal_estimate, config
    )
    forged = copy.deepcopy(report)
    forged["bindings"]["observed_git_state"]["commit"] = (
        ABANDONED_DIAGNOSTIC_COMMIT
    )
    _resign(forged)

    with pytest.raises(
        benchmark.StageCBenchmarkIntegrityError,
        match="Git binding|abandoned|forbidden",
    ):
        benchmark.validate_benchmark_report(forged)


def test_formal_report_rejects_resigned_consistently_bound_abandoned_commit(
    monkeypatch, tmp_path, formal_plan, formal_estimate, config
):
    report = _formal_report(
        monkeypatch, tmp_path, formal_plan, formal_estimate, config
    )
    forged = copy.deepcopy(report)
    old_code = {
        "commit": ABANDONED_DIAGNOSTIC_COMMIT,
        "dirty": False,
    }
    forged["bindings"]["observed_git_state"] = old_code
    forged["bindings"]["shared_provenance"]["code"] = old_code
    forged["bindings"]["shared_provenance_sha256"] = (
        benchmark._canonical_digest(
            forged["bindings"]["shared_provenance"]
        )
    )
    for record in forged["unit_records"]:
        record["shared_provenance"]["code"] = old_code
        record["shared_provenance_sha256"] = forged["bindings"][
            "shared_provenance_sha256"
        ]
    _resign(forged)

    with pytest.raises(
        benchmark.StageCBenchmarkIntegrityError,
        match="abandoned|forbidden|357dd9d|old",
    ):
        benchmark.validate_benchmark_report(forged)


def test_launch_gate_rejects_resigned_abandoned_commit(
    monkeypatch, tmp_path, diagnostic_plan, formal_plan, config
):
    gate = _diagnostic_report(
        monkeypatch, tmp_path, diagnostic_plan, config, wall=12.5
    )
    assert gate["status"] == benchmark.STATUS_PASS
    forged = copy.deepcopy(gate)
    forged["bindings"]["shared_provenance"]["code"] = {
        "commit": ABANDONED_DIAGNOSTIC_COMMIT,
        "dirty": False,
    }
    forged["bindings"]["shared_provenance_sha256"] = (
        benchmark._canonical_digest(
            forged["bindings"]["shared_provenance"]
        )
    )
    _resign(forged)

    with pytest.raises(
        benchmark.StageCBenchmarkIntegrityError, match="Git|provenance"
    ):
        benchmark.require_c7_launch_gate(
            forged,
            formal_plan,
            config=config,
            output_root=tmp_path / "new-formal-output",
            observed_git_state={"commit": NEW_TEST_COMMIT, "dirty": False},
            run_spec_path=FORMAL_SPEC_PATH,
        )


def test_unselected_attempt_bytes_are_counted_but_never_used_as_measurement(
    monkeypatch, tmp_path, formal_plan, formal_estimate, config
):
    root = tmp_path / "formal-source"
    mapping = _formal_mapping(formal_plan, formal_estimate, root)
    selected = mapping["records"][0]["attempt_path"]
    mapping["records"][0]["attempts"] = [
        {"path": selected, "status": "complete"},
        {
            "path": "/synthetic/formal/unselected/attempt-000000",
            "status": "failed",
            "runner_telemetry": _runner_telemetry(total_wall=100_000.0),
        },
    ]
    unselected_attempt_bytes = 123_456
    report = _formal_report(
        monkeypatch, tmp_path, formal_plan, formal_estimate, config,
        mapping=mapping, source_bytes=unselected_attempt_bytes,
        layout_errors=("unselected historical attempt pollutes formal root",),
    )

    assert report["status"] == benchmark.STATUS_BLOCKED
    assert report["measurement_boundary"]["all_regular_file_bytes"] == (
        unselected_attempt_bytes
    )
    assert report["projections"]["storage"][
        "source_batch_regular_file_bytes"
    ] == unselected_attempt_bytes
    assert report["projections"]["measurement_completeness"][
        "usable_units"
    ] == 40
    assert len(report["unit_records"]) == 40
    assert report["unit_records"][0]["attempt_path"] == selected
    assert report["projections"]["time"][
        "slowest_valid_measured_unit_seconds"
    ] == pytest.approx(1.39)
    assert max(
        record["total_regular_file_bytes"]
        for record in report["unit_records"]
    ) < unselected_attempt_bytes


def test_resigned_blocked_diagnostic_cannot_forge_existing_binding(
    monkeypatch, tmp_path, diagnostic_plan, config
):
    report = _diagnostic_report(
        monkeypatch, tmp_path, diagnostic_plan, config, wall=720.0
    )
    assert report["status"] == benchmark.STATUS_BLOCKED
    report["bindings"]["execution_plan_sha256"] = "e" * 64
    _resign(report)

    with pytest.raises(
        benchmark.StageCBenchmarkIntegrityError, match="semantic validation"
    ):
        benchmark.validate_c7_diagnostic_gate(report)


def test_resigned_blocked_formal_cannot_forge_source_bytes(
    monkeypatch, tmp_path, formal_plan, formal_estimate, config
):
    report = _formal_report(
        monkeypatch, tmp_path, formal_plan, formal_estimate, config,
        layout_errors=("synthetic source pollution",),
    )
    assert report["status"] == benchmark.STATUS_BLOCKED
    report["measurement_boundary"]["all_regular_file_bytes"] += 1
    _resign(report)

    with pytest.raises(
        benchmark.StageCBenchmarkIntegrityError, match="semantic validation"
    ):
        benchmark.validate_benchmark_report(report)
