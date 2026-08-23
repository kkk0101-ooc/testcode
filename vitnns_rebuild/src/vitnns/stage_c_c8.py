"""Fail-closed planning and orchestration helpers for formal Stage C8.

The module deliberately does not load images or initialize a model.  It pins
the C7 evidence used to authorize C8, validates the complete ordered C8 plan,
initializes a new root, and coordinates immutable execution-unit attempts.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .constants import REPO_ROOT
from .stage_c_benchmark import (
    FROZEN_CAPS,
    HARD_CAP_KEYS,
    STATUS_PASS,
    StageCBenchmarkBlockedError,
    StageCBenchmarkIntegrityError,
    validate_benchmark_report,
)
from .stage_c_plan import (
    AUDIT_SAMPLE_IDS,
    CALIBRATION_SAMPLE_IDS,
    CENTER_RANKING,
    CLASS_METHOD_IDS,
    CURVE_DIRECTIONS,
    CURVE_POINT_KIND,
    VALUE_METHOD_IDS,
    ExecutionPlan,
    FIDELITY_SCALES,
    MAP_KIND,
    RANDOM_RANKING,
    StageCEstimate,
    TARGET_RANKING,
    WorkItem,
    enforce_stage_c_budget,
    estimate_stage_c_run,
)
from .stage_c_perturbations import PERTURBATION_IDS
from .stage_c_provenance import canonical_sha256, capture_git_state
from .stage_c_resources import measure_artifact_tree_bytes
from .stage_c_runner import run_execution_unit

SCHEMA_VERSION = 1
C8_EXECUTION_UNITS = 2180
C8_UNITS_PER_IMAGE = 436
C8_EXPECTED_COUNTS = {
    "execution_units": 2180,
    "map_runs": 9592,
    "fidelity_eligible_units": 1460,
    "structural_reference_units": 720,
    "fidelity_curves": 64240,
    "fidelity_curve_points": 1_349_040,
    "work_items": 1_358_632,
}

C7_REPORT_FILE_SHA256 = "d33e4639da522460e8a2e83219b4d3a68bff50cf2adbfda97545a69ed471610c"
C7_REPORT_CANONICAL_DIGEST = "ae32465c79f9b0a0d24de4eaaf07d2f7b93e481d1188b62cac49f5aeb75eba7f"
C7_CODE_COMMIT = "314ae5e7f30b5d41f51d3b0d359992d7f0a95fdd"
C7_ARCHIVE_REF = "refs/heads/archive/c7-benchmark-314ae5e"
C7_SLOWEST_UNIT_SECONDS = 76.84819071099628
C7_PEAK_RESERVED_BYTES = 2_833_252_352
C7_PROJECTED_C8_H20_GPU_HOURS = 58.1698110242958
C7_PROJECTED_C8_ARTIFACT_BYTES = 4_289_463_500
C7_PROJECTED_C8_C9_ARTIFACT_BYTES = 4_841_749_700
C7_PROJECTED_C8_L2_BYTES_PER_AUDIT = 488_599_600

AUTHORIZATION_FILENAME = "c8_launch_authorization.json"
_CHECKPOINT_LOCK_FILENAME = ".c8-checkpoint.lock"
_LIFECYCLE_LOCK_FILENAME = ".c8-lifecycle.lock"
_ARTIFACT_ACTIVITY_LOCK_FILENAME = ".c8-artifact-activity.lock"
_CONTROL_NAMES = (
    "execution_plan.json",
    "preflight.json",
    "failure_table.json",
    "batch_status.json",
    AUTHORIZATION_FILENAME,
)
_ALL_METHODS = (*VALUE_METHOD_IDS, *CLASS_METHOD_IDS)
_SHA256 = frozenset("0123456789abcdef")
_TERMINAL_STATES = frozenset({"succeeded", "failed"})
_CHECKPOINT_STATES = frozenset({
    "pending", "claimed", "succeeded", "failed", "missing",
    "incomplete", "tampered",
})
_ROW_KEYS = frozenset({
    "unit_key", "state", "terminal", "complete",
    "selected_attempt", "selected_status_sha256", "failure",
    "claim_token", "l0_l1_bytes", "l2_bytes", "total_bytes",
})
_LIFECYCLE_STATES = frozenset({
    "running", "blocked", "finalizing", "finalized",
})
_ATTEMPT_NAME = re.compile(r"attempt-[0-9]{6}\Z")
_STATUS_KEYS = frozenset({
    "schema_version", "document_type", "plan_digest",
    "expected_unit_count", "orchestration_complete", "successful",
    "failure_table_sha256", "lifecycle_state",
    "active_claim_count", "validation_errors",
})


def _plain(value: Any) -> dict[str, Any]:
    method = getattr(value, "to_dict", None)
    value = method() if callable(method) else value
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping or expose to_dict()")
    return json.loads(json.dumps(value, allow_nan=False))


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise StageCBenchmarkIntegrityError(
                    f"duplicate JSON key in {path}: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except StageCBenchmarkIntegrityError:
        raise
    except Exception as error:
        raise StageCBenchmarkIntegrityError(f"invalid JSON document {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageCBenchmarkIntegrityError(f"JSON document is not an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _plain(value), indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _checkpoint_lock(root: Path) -> Iterator[None]:
    """Serialize shared checkpoint updates across independent shard processes."""

    lock_path = root / _CHECKPOINT_LOCK_FILENAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise StageCBenchmarkIntegrityError(
            f"cannot safely open C8 checkpoint lock: {error}"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise StageCBenchmarkIntegrityError(
                "C8 checkpoint lock must be a regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        descriptor_stat = os.fstat(descriptor)
        path_stat = lock_path.lstat()
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            path_stat.st_dev, path_stat.st_ino
        ):
            raise StageCBenchmarkIntegrityError(
                "C8 checkpoint lock changed while being acquired"
            )
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _lifecycle_lock(root: Path, *, exclusive: bool) -> Iterator[None]:
    """Serialize finalization against all concurrently running shards."""

    lock_path = root / _LIFECYCLE_LOCK_FILENAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise StageCBenchmarkIntegrityError(
            f"cannot safely open C8 lifecycle lock: {error}"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise StageCBenchmarkIntegrityError(
                "C8 lifecycle lock must be a regular file"
            )
        fcntl.flock(
            descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        )
        descriptor_stat = os.fstat(descriptor)
        path_stat = lock_path.lstat()
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            path_stat.st_dev, path_stat.st_ino
        ):
            raise StageCBenchmarkIntegrityError(
                "C8 lifecycle lock changed while being acquired"
            )
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _artifact_activity_lock(root: Path, *, exclusive: bool) -> Iterator[None]:
    """Keep whole-tree accounting disjoint from concurrent artifact writes."""

    lock_path = root / _ARTIFACT_ACTIVITY_LOCK_FILENAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise StageCBenchmarkIntegrityError(
            f"cannot safely open C8 artifact-activity lock: {error}"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise StageCBenchmarkIntegrityError(
                "C8 artifact-activity lock must be a regular file"
            )
        fcntl.flock(
            descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        )
        descriptor_stat = os.fstat(descriptor)
        path_stat = lock_path.lstat()
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            path_stat.st_dev, path_stat.st_ino
        ):
            raise StageCBenchmarkIntegrityError(
                "C8 artifact-activity lock changed while being acquired"
            )
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _counts(plan_doc: Mapping[str, Any]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    units = plan_doc.get("units", [])
    for unit in units if isinstance(units, list) else []:
        counts = unit.get("work_item_counts", {}) if isinstance(unit, Mapping) else {}
        for key in ("map_runs", "fidelity_curves", "fidelity_curve_points", "work_items"):
            if type(counts.get(key)) is int:
                totals[key] += counts[key]
        totals["fidelity_eligible_units"] += int(
            unit.get("fidelity_eligible") is True
        )
        totals["structural_reference_units"] += int(
            unit.get("structural_reference_only") is True
        )
    totals["execution_units"] = len(units) if isinstance(units, list) else 0
    return {key: int(totals[key]) for key in C8_EXPECTED_COUNTS}


def _expected_unit_work_counts(method_id: str, audit: bool) -> dict[str, int]:
    maps = 5 if audit else 4
    curves = 0 if method_id == VALUE_METHOD_IDS[0] else 44
    points = curves * 21
    return {
        "work_items": maps + points,
        "map_runs": maps,
        "clean_map_runs": 1,
        "perturbation_map_runs": 3,
        "repeatability_map_runs": 1 if audit else 0,
        "fidelity_curves": curves,
        "fidelity_curve_points": points,
        "target_tie_curve_points": 420 if curves else 0,
        "random_sanity_curve_points": 420 if curves else 0,
        "center_sanity_curve_points": 84 if curves else 0,
        "logical_scalar_evaluations": points,
    }


def _expected_unit_work_items(
    method_id: str, audit: bool, block: int | None, head: int | None,
) -> list[dict[str, Any]]:
    items = [
        WorkItem(
            MAP_KIND, "clean", 0, None, None, None, None, None,
            None, None, None,
        )
    ]
    items.extend(
        WorkItem(
            MAP_KIND, perturbation_id, None, None, None, None, None,
            None, None, None, None,
        )
        for perturbation_id in PERTURBATION_IDS
    )
    if audit:
        items.append(
            WorkItem(
                MAP_KIND, "repeat_clean", 1, None, None, None, None,
                None, None, None, None,
            )
        )
    if method_id != VALUE_METHOD_IDS[0]:
        pairing_block = 0 if block is None else block
        pairing_head = 0 if head is None else head
        rankings = [
            *((TARGET_RANKING, replicate) for replicate in range(5)),
            *((RANDOM_RANKING, replicate) for replicate in range(5)),
            (CENTER_RANKING, None),
        ]
        items.extend(
            WorkItem(
                CURVE_POINT_KIND, "clean", None, scale, direction,
                ranking, replicate, fraction_index, fraction_index / 20.0,
                pairing_block, pairing_head,
            )
            for scale in FIDELITY_SCALES
            for direction in CURVE_DIRECTIONS
            for ranking, replicate in rankings
            for fraction_index in range(21)
        )
    return [item.to_dict() for item in items]


def validate_c8_execution_plan(plan: ExecutionPlan, config: Any) -> dict[str, Any]:
    """Validate the exact ordered five-image, all-candidate C8 workload."""

    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")
    document = ExecutionPlan.from_dict(plan.to_dict()).to_dict()
    spec = document["run_spec"]
    units = document["units"]
    violations: list[str] = []
    expected_spec = {
        "schema_version": 1,
        "document_type": "stage_c_run_spec",
        "sample_ids": list(CALIBRATION_SAMPLE_IDS),
        "method_ids": list(_ALL_METHODS),
        "value_config_source": "calibration_all_144",
        "value_candidates": [],
        "fidelity_pairing_context": {"block_index": 0, "head_index": 0},
        "audit_repeatability_runs": 2,
        "audit_repeatability_scope": "clean_map_only",
        "run_registered_fidelity": True,
        "run_sanity": True,
        "run_stability": True,
    }
    if spec != expected_spec:
        violations.append("C8 RunSpec differs from the exact frozen document")
    selection = document.get("selection", {})
    if (
        selection.get("status") != "not_applicable_pre_c8"
        or selection.get("sha256") is not None
        or selection.get("primary") is not None
        or selection.get("depth_band_winners") != []
    ):
        violations.append("C8 calibration must not consume an existing selection")
    observed_counts = _counts(document)
    if observed_counts != C8_EXPECTED_COUNTS:
        violations.append(f"C8 expanded counts mismatch: {observed_counts!r}")
    method_counts = Counter(unit.get("key", {}).get("method_id") for unit in units)
    expected_method_counts = {
        **{method: 720 for method in VALUE_METHOD_IDS},
        **{method: 5 for method in CLASS_METHOD_IDS},
    }
    if method_counts != Counter(expected_method_counts):
        violations.append("C8 per-method unit counts drifted")
    per_sample = Counter(unit.get("sample", {}).get("sample_id") for unit in units)
    if per_sample != Counter(
        {sample: C8_UNITS_PER_IMAGE for sample in CALIBRATION_SAMPLE_IDS}
    ):
        violations.append("C8 must contain exactly 436 units per calibration image")
    audit_ids = {
        unit.get("sample", {}).get("sample_id")
        for unit in units
        if unit.get("sample", {}).get("audit") is True
    }
    if audit_ids != set(AUDIT_SAMPLE_IDS):
        violations.append("C8 audit scope differs from the exact two audit IDs")
    if any(unit.get("sample", {}).get("role") != "calibration" for unit in units):
        violations.append("C8 execution plan contains a holdout unit")

    expected_order: list[tuple[str, str, int | None, int | None]] = []
    for sample_id in CALIBRATION_SAMPLE_IDS:
        for method_id in _ALL_METHODS:
            if method_id in VALUE_METHOD_IDS:
                expected_order.extend(
                    (sample_id, method_id, block, head)
                    for block in range(12)
                    for head in range(12)
                )
            else:
                expected_order.append((sample_id, method_id, None, None))
    observed_order = [
        (
            unit.get("key", {}).get("sample_id"),
            unit.get("key", {}).get("method_id"),
            unit.get("key", {}).get("block_index"),
            unit.get("key", {}).get("head_index"),
        )
        for unit in units
    ]
    if observed_order != expected_order:
        violations.append("C8 ordered unit expansion drifted")
    for unit in units:
        key = unit.get("key", {})
        sample = unit.get("sample", {})
        expected = _expected_unit_work_counts(
            str(key.get("method_id")), sample.get("sample_id") in AUDIT_SAMPLE_IDS
        )
        if unit.get("work_item_counts") != expected:
            violations.append(
                f"C8 unit work counts drifted: {key.get('canonical_digest')}"
            )
        expected_items = _expected_unit_work_items(
            str(key.get("method_id")),
            sample.get("sample_id") in AUDIT_SAMPLE_IDS,
            key.get("block_index"), key.get("head_index"),
        )
        if unit.get("work_items") != expected_items:
            violations.append(
                f"C8 ordered work items drifted: {key.get('canonical_digest')}"
            )
    config_methods = [item.id for item in getattr(config, "methods", ())]
    if config_methods != list(_ALL_METHODS):
        violations.append("config method order/set drifted")
    return {
        "valid": not violations,
        "violations": violations,
        "counts": observed_counts,
        "sample_ids": list(CALIBRATION_SAMPLE_IDS),
        "method_ids": list(_ALL_METHODS),
        "execution_plan_sha256": plan.canonical_digest,
    }


def _exact_projection(report: Mapping[str, Any], path: Sequence[str], expected: Any) -> bool:
    value: Any = report
    for key in path:
        if not isinstance(value, Mapping):
            return False
        value = value.get(key)
    return type(value) is type(expected) and value == expected


def _validate_c7_archive_ref(
    *, repo_root: Path | str | None = None, git_runner: Any = subprocess.run
) -> dict[str, str]:
    root = Path.cwd() if repo_root is None else Path(repo_root)
    try:
        ref_commit = git_runner(
            ["git", "rev-parse", "--verify", f"{C7_ARCHIVE_REF}^{{commit}}"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit_tree = git_runner(
            ["git", "rev-parse", f"{C7_CODE_COMMIT}^{{tree}}"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        ref_tree = git_runner(
            ["git", "rev-parse", f"{C7_ARCHIVE_REF}^{{tree}}"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise StageCBenchmarkIntegrityError(
            f"cannot resolve the frozen C7 archive reference: {error}"
        ) from error
    if ref_commit != C7_CODE_COMMIT or ref_tree != commit_tree:
        raise StageCBenchmarkIntegrityError(
            "C7 archive reference does not resolve to the frozen code commit/content"
        )
    if not all(len(item) == 40 and set(item) <= _SHA256 for item in (ref_commit, ref_tree)):
        raise StageCBenchmarkIntegrityError("C7 archive reference returned malformed Git IDs")
    return {
        "ref": C7_ARCHIVE_REF, "commit": ref_commit, "tree": ref_tree
    }


def validate_fixed_c7_report(
    path: Path | str, *, repo_root: Path | str | None = None
) -> dict[str, Any]:
    """Validate the one frozen C7 report and its retained code reference."""

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise StageCBenchmarkIntegrityError(
            "C7 report is missing, non-regular, or a symlink"
        )
    if _file_sha256(source) != C7_REPORT_FILE_SHA256:
        raise StageCBenchmarkIntegrityError(
            "C7 report file SHA-256 differs from the frozen C8 input"
        )
    report = _strict_json(source)
    validate_benchmark_report(report)
    errors: list[str] = []
    if report.get("canonical_digest") != C7_REPORT_CANONICAL_DIGEST:
        errors.append("canonical digest mismatch")
    if report.get("status") != STATUS_PASS or report.get("integrity_valid") is not True:
        errors.append("report is not an integrity-valid PASS")
    validation = report.get("batch_validation", {})
    if (
        validation.get("valid") is not True
        or validation.get("orchestration_complete") is not True
        or validation.get("successful") is not True
    ):
        errors.append("C7 batch is not valid, complete, and successful")
    records = report.get("unit_records")
    if (
        not isinstance(records, list)
        or len(records) != 40
        or any(
            not isinstance(row, Mapping)
            or row.get("state") != "succeeded"
            or row.get("trusted_by_batch_validator") is not True
            for row in records
        )
    ):
        errors.append("C7 report does not contain 40/40 usable units")
    completeness = report.get("projections", {}).get("measurement_completeness", {})
    if completeness != {
        "bytes_measured": True,
        "calls_measured": True,
        "cuda_measured": True,
        "expected_units": 40,
        "timing_measured": True,
        "usable_units": 40,
    }:
        errors.append("C7 measurement completeness differs from 40/40 measured")
    checks = report.get("projections", {}).get("hard_cap_checks", [])
    if (
        not isinstance(checks, list)
        or len(checks) != len(HARD_CAP_KEYS)
        or [item.get("cap_key") for item in checks if isinstance(item, Mapping)]
        != list(HARD_CAP_KEYS)
        or any(
            not isinstance(item, Mapping)
            or item.get("status") != STATUS_PASS
            or item.get("measured") is not True
            or item.get("cap") != FROZEN_CAPS.get(item.get("cap_key"))
            for item in checks
        )
    ):
        errors.append("C7 report does not retain ten measured frozen-cap PASS checks")
    bindings = report.get("bindings", {})
    observed_code = bindings.get("observed_git_state") if isinstance(bindings, Mapping) else None
    shared_code = (
        bindings.get("shared_provenance", {}).get("code")
        if isinstance(bindings, Mapping)
        and isinstance(bindings.get("shared_provenance"), Mapping)
        else None
    )
    launch_code = (
        bindings.get("launch_authorization", {}).get("observed_git_state")
        if isinstance(bindings, Mapping)
        and isinstance(bindings.get("launch_authorization"), Mapping)
        else None
    )
    expected_code = {"commit": C7_CODE_COMMIT, "dirty": False}
    if observed_code != expected_code or shared_code != expected_code or launch_code != expected_code:
        errors.append("C7 report code bindings do not equal the clean frozen commit")
    exact_values = (
        (("projections", "time", "slowest_valid_measured_unit_seconds"), C7_SLOWEST_UNIT_SECONDS),
        (("projections", "time", "C8_h20_gpu_hours"), C7_PROJECTED_C8_H20_GPU_HOURS),
        (("projections", "measured_maxima", "cuda_peak_reserved_bytes_per_unit"), C7_PEAK_RESERVED_BYTES),
        (("projections", "measured_maxima", "c8_projected_l2_bytes_per_audit_sample"), C7_PROJECTED_C8_L2_BYTES_PER_AUDIT),
        (("projections", "storage", "projected_bytes", "C8"), C7_PROJECTED_C8_ARTIFACT_BYTES),
        (("projections", "storage", "projected_bytes", "C8_C9"), C7_PROJECTED_C8_C9_ARTIFACT_BYTES),
    )
    for projection_path, expected in exact_values:
        if not _exact_projection(report, projection_path, expected):
            errors.append(f"C7 projection drifted: {'.'.join(projection_path)}")
    recomputed_hours = (
        C7_SLOWEST_UNIT_SECONDS * C8_EXECUTION_UNITS
        * float(FROZEN_CAPS["projection_safety_factor"]) / 3600.0
    )
    if recomputed_hours != C7_PROJECTED_C8_H20_GPU_HOURS:
        errors.append("C7-to-C8 time projection arithmetic is not exact")
    for flag in (
        "holdout_accessed", "selection_performed",
        "bootstrap_performed", "decision_performed",
    ):
        if report.get(flag) is not False:
            errors.append(f"C7 report {flag} must be false")
    if errors:
        raise StageCBenchmarkIntegrityError(
            "invalid frozen C7 report: " + "; ".join(errors)
        )
    archive = _validate_c7_archive_ref(repo_root=repo_root)
    result = dict(report)
    result["c8_validated_archive_ref"] = archive
    return result


def build_c8_measured_preflight(
    plan: ExecutionPlan, config: Any, c7_report: Mapping[str, Any]
) -> StageCEstimate:
    scope = validate_c8_execution_plan(plan, config)
    if not scope["valid"]:
        raise StageCBenchmarkIntegrityError(
            "invalid C8 plan: " + "; ".join(scope["violations"])
        )
    if not _exact_projection(
        c7_report, ("projections", "time", "slowest_valid_measured_unit_seconds"),
        C7_SLOWEST_UNIT_SECONDS,
    ):
        raise StageCBenchmarkIntegrityError("C7 slowest-unit measurement is not frozen")
    if not _exact_projection(
        c7_report, ("projections", "measured_maxima", "cuda_peak_reserved_bytes_per_unit"),
        C7_PEAK_RESERVED_BYTES,
    ):
        raise StageCBenchmarkIntegrityError("C7 CUDA peak measurement is not frozen")
    estimate = estimate_stage_c_run(
        config, plan,
        measured_slowest_unit_seconds=C7_SLOWEST_UNIT_SECONDS,
        measured_peak_reserved_bytes_per_unit=C7_PEAK_RESERVED_BYTES,
    )
    projected = estimate.compute.get("projected_h20_gpu_hours")
    if projected != C7_PROJECTED_C8_H20_GPU_HOURS:
        raise StageCBenchmarkIntegrityError(
            "C8 measured preflight does not prove the frozen 58.1698110242958 hours"
        )
    if not estimate.within_budget:
        raise StageCBenchmarkBlockedError(
            "C8 measured preflight exceeds frozen caps",
            reasons=list(estimate.violations),
        )
    return estimate


def _validate_formal_preflight(
    estimate: StageCEstimate, plan: ExecutionPlan, *, require_issuer: bool,
) -> None:
    if not isinstance(estimate, StageCEstimate):
        raise StageCBenchmarkIntegrityError("C8 preflight is not a StageCEstimate")
    if require_issuer:
        try:
            enforce_stage_c_budget(estimate)
        except Exception as error:
            raise StageCBenchmarkIntegrityError(
                f"C8 preflight is not freshly issuer-authenticated: {error}"
            ) from error
    if (
        estimate.plan_digest != plan.canonical_digest
        or estimate.within_budget is not True
        or tuple(estimate.violations) != ()
        or dict(estimate.caps.get("effective", {})) != FROZEN_CAPS
        or dict(estimate.caps.get("test_overrides", {})) != {}
        or estimate.caps.get("config_mutated") is not False
        or tuple(estimate.caps.get("unmeasured_caps", ())) != ()
        or estimate.compute.get("measured_slowest_unit_seconds")
        != C7_SLOWEST_UNIT_SECONDS
        or estimate.compute.get("measured_peak_reserved_bytes_per_unit")
        != C7_PEAK_RESERVED_BYTES
        or estimate.compute.get("projected_h20_gpu_hours")
        != C7_PROJECTED_C8_H20_GPU_HOURS
    ):
        raise StageCBenchmarkIntegrityError(
            "C8 preflight is not the exact measured, frozen-cap estimate"
        )


def _require_live_git_state(
    expected: Mapping[str, Any] | None = None, *,
    supplied: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    live = _plain(capture_git_state(REPO_ROOT if repo_root is None else repo_root))
    if (
        set(live) != {"commit", "dirty"}
        or live.get("dirty") is not False
        or not isinstance(live.get("commit"), str)
        or len(live["commit"]) != 40
        or set(live["commit"]) - _SHA256
    ):
        raise StageCBenchmarkBlockedError(
            "C8 formal execution requires the live Git worktree to be clean"
        )
    if supplied is not None and _plain(supplied) != live:
        raise StageCBenchmarkIntegrityError(
            "supplied C8 Git state differs from the live repository"
        )
    if expected is not None and _plain(expected) != live:
        raise StageCBenchmarkIntegrityError(
            "live C8 Git state differs from launch authorization"
        )
    return live


def build_c8_launch_authorization(
    plan: ExecutionPlan,
    config: Any,
    *,
    c7_report_path: Path | str,
    output_root: Path | str,
    observed_git_state: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> tuple[dict[str, Any], StageCEstimate]:
    report = validate_fixed_c7_report(c7_report_path, repo_root=repo_root)
    estimate = build_c8_measured_preflight(plan, config, report)
    code = _require_live_git_state(
        supplied=observed_git_state, repo_root=repo_root
    )
    authorization = {
        "schema_version": 1,
        "document_type": "stage_c_c8_launch_authorization",
        "status": "PASS",
        "holdout_accessed": False,
        "c7_report_path": str(Path(c7_report_path).absolute()),
        "c7_report_file_sha256": C7_REPORT_FILE_SHA256,
        "c7_report_canonical_digest": C7_REPORT_CANONICAL_DIGEST,
        "c7_code_commit": C7_CODE_COMMIT,
        "c7_archive_ref": C7_ARCHIVE_REF,
        "c7_archive_tree": report["c8_validated_archive_ref"]["tree"],
        "c7_slowest_unit_seconds": C7_SLOWEST_UNIT_SECONDS,
        "c7_peak_reserved_bytes": C7_PEAK_RESERVED_BYTES,
        "c7_projected_c8_h20_gpu_hours": C7_PROJECTED_C8_H20_GPU_HOURS,
        "c7_projected_c8_artifact_bytes": C7_PROJECTED_C8_ARTIFACT_BYTES,
        "c7_projected_c8_l2_bytes_per_audit": C7_PROJECTED_C8_L2_BYTES_PER_AUDIT,
        "c8_execution_plan_sha256": plan.canonical_digest,
        "c8_preflight_sha256": estimate.canonical_digest,
        "c8_output_root": str(Path(output_root).absolute()),
        "code": code,
    }
    authorization["canonical_digest"] = canonical_sha256(authorization)
    validate_c8_launch_authorization(
        authorization, plan=plan, estimate=estimate, output_root=output_root
    )
    return authorization, estimate


def validate_c8_launch_authorization(
    value: Mapping[str, Any],
    *,
    plan: ExecutionPlan,
    estimate: StageCEstimate,
    output_root: Path | str | None = None,
) -> None:
    expected = {
        "schema_version", "document_type", "status",
        "holdout_accessed", "c7_report_path",
        "c7_report_file_sha256", "c7_report_canonical_digest",
        "c7_code_commit", "c7_archive_ref", "c7_archive_tree",
        "c7_slowest_unit_seconds", "c7_peak_reserved_bytes",
        "c7_projected_c8_h20_gpu_hours",
        "c7_projected_c8_artifact_bytes",
        "c7_projected_c8_l2_bytes_per_audit",
        "c8_execution_plan_sha256", "c8_preflight_sha256",
        "c8_output_root", "code", "canonical_digest",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise StageCBenchmarkIntegrityError("C8 launch authorization fields drifted")
    payload = {key: item for key, item in value.items() if key != "canonical_digest"}
    if value.get("canonical_digest") != canonical_sha256(payload):
        raise StageCBenchmarkIntegrityError("C8 launch authorization digest mismatch")
    fixed = {
        "schema_version": 1,
        "document_type": "stage_c_c8_launch_authorization",
        "status": "PASS",
        "holdout_accessed": False,
        "c7_report_file_sha256": C7_REPORT_FILE_SHA256,
        "c7_report_canonical_digest": C7_REPORT_CANONICAL_DIGEST,
        "c7_code_commit": C7_CODE_COMMIT,
        "c7_archive_ref": C7_ARCHIVE_REF,
        "c7_slowest_unit_seconds": C7_SLOWEST_UNIT_SECONDS,
        "c7_peak_reserved_bytes": C7_PEAK_RESERVED_BYTES,
        "c7_projected_c8_h20_gpu_hours": C7_PROJECTED_C8_H20_GPU_HOURS,
        "c7_projected_c8_artifact_bytes": C7_PROJECTED_C8_ARTIFACT_BYTES,
        "c7_projected_c8_l2_bytes_per_audit": C7_PROJECTED_C8_L2_BYTES_PER_AUDIT,
        "c8_execution_plan_sha256": plan.canonical_digest,
        "c8_preflight_sha256": estimate.canonical_digest,
    }
    if any(value.get(key) != expected_value for key, expected_value in fixed.items()):
        raise StageCBenchmarkIntegrityError("C8 launch authorization frozen binding drifted")
    for key in ("c7_archive_tree",):
        item = value.get(key)
        if not isinstance(item, str) or len(item) != 40 or set(item) - _SHA256:
            raise StageCBenchmarkIntegrityError(f"C8 launch authorization {key} is invalid")
    code = value.get("code")
    if (
        not isinstance(code, Mapping)
        or set(code) != {"commit", "dirty"}
        or code.get("dirty") is not False
        or not isinstance(code.get("commit"), str)
        or len(code["commit"]) != 40
        or set(code["commit"]) - _SHA256
    ):
        raise StageCBenchmarkIntegrityError(
            "C8 launch authorization requires clean Git evidence"
        )
    if output_root is not None and Path(str(value.get("c8_output_root"))).absolute() != Path(output_root).absolute():
        raise StageCBenchmarkIntegrityError(
            "C8 launch authorization output-root binding drifted"
        )


def _require_fresh_root(output_root: Path | str) -> Path:
    root = Path(os.path.abspath(os.fspath(output_root)))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise StageCBenchmarkBlockedError("C8 output root contains a symlink")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise StageCBenchmarkBlockedError(
            "C8 output root must be missing or empty"
        )
    return root


def _pending_rows(plan: ExecutionPlan) -> list[dict[str, Any]]:
    return [
        {
            "unit_key": unit.key.to_dict(),
            "state": "pending",
            "terminal": False,
            "complete": False,
            "selected_attempt": None,
            "selected_status_sha256": None,
            "failure": None,
            "claim_token": None,
            "l0_l1_bytes": None,
            "l2_bytes": None,
            "total_bytes": None,
        }
        for unit in plan.units
    ]


def _failure_table(plan: ExecutionPlan, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "document_type": "stage_c_failure_table",
        "plan_digest": plan.canonical_digest,
        "rows": [_plain(row) for row in rows],
    }


def _batch_status(
    plan: ExecutionPlan, rows: Sequence[Mapping[str, Any]], failure_sha256: str,
    *, validation_errors: Sequence[str] = (),
    lifecycle_state: str = "running",
) -> dict[str, Any]:
    if lifecycle_state not in _LIFECYCLE_STATES:
        raise StageCBenchmarkIntegrityError("invalid C8 lifecycle state")
    terminal = all(row.get("state") in _TERMINAL_STATES for row in rows)
    finalized = lifecycle_state == "finalized"
    successful = bool(rows) and all(row.get("state") == "succeeded" for row in rows)
    return {
        "schema_version": 1,
        "document_type": "stage_c_batch_status",
        "plan_digest": plan.canonical_digest,
        "expected_unit_count": len(rows),
        "orchestration_complete": finalized and terminal,
        "successful": finalized and successful and not validation_errors,
        "failure_table_sha256": failure_sha256,
        "lifecycle_state": lifecycle_state,
        "active_claim_count": sum(
            row.get("state") == "claimed" for row in rows
        ),
        "validation_errors": list(validation_errors),
    }


def initialize_c8_output_root(
    output_root: Path | str,
    plan: ExecutionPlan,
    preflight: StageCEstimate,
    authorization: Mapping[str, Any],
    *,
    config: Any,
    observed_git_state: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> Path:
    scope = validate_c8_execution_plan(plan, config)
    if not scope["valid"]:
        raise StageCBenchmarkIntegrityError(
            "invalid C8 plan: " + "; ".join(scope["violations"])
        )
    validate_c8_launch_authorization(
        authorization, plan=plan, estimate=preflight, output_root=output_root
    )
    _validate_formal_preflight(preflight, plan, require_issuer=True)
    live_code = _require_live_git_state(
        authorization["code"], supplied=observed_git_state,
        repo_root=repo_root,
    )
    if live_code != authorization["code"]:
        raise StageCBenchmarkIntegrityError("C8 initialization code binding drifted")
    report = validate_fixed_c7_report(
        authorization["c7_report_path"], repo_root=repo_root
    )
    if (
        report["canonical_digest"] != authorization["c7_report_canonical_digest"]
        or report["c8_validated_archive_ref"]["tree"]
        != authorization["c7_archive_tree"]
    ):
        raise StageCBenchmarkIntegrityError(
            "C8 initialization C7 evidence/archive binding drifted"
        )
    root = _require_fresh_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = _pending_rows(plan)
    table = _failure_table(plan, rows)
    _atomic_create_json(root / "execution_plan.json", plan.to_dict())
    _atomic_create_json(root / "preflight.json", preflight.to_dict())
    _atomic_create_json(root / AUTHORIZATION_FILENAME, authorization)
    _atomic_create_json(root / "failure_table.json", table)
    _atomic_create_json(
        root / "batch_status.json",
        _batch_status(plan, rows, _file_sha256(root / "failure_table.json")),
    )
    return root


def _validate_initialized_controls(
    root: Path, plan: ExecutionPlan, estimate: StageCEstimate, *,
    require_issuer: bool = True, validate_checkpoint: bool = True,
) -> dict[str, Any]:
    for name in _CONTROL_NAMES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise StageCBenchmarkIntegrityError(
                f"C8 root lacks safe initialized control {name}"
            )
    if _strict_json(root / "execution_plan.json") != plan.to_dict():
        raise StageCBenchmarkIntegrityError("C8 initialized execution plan drifted")
    if _strict_json(root / "preflight.json") != estimate.to_dict():
        raise StageCBenchmarkIntegrityError("C8 initialized preflight drifted")
    _validate_formal_preflight(estimate, plan, require_issuer=require_issuer)
    authorization = _strict_json(root / AUTHORIZATION_FILENAME)
    validate_c8_launch_authorization(
        authorization, plan=plan, estimate=estimate, output_root=root
    )
    if validate_checkpoint:
        _validate_checkpoint_controls(root, plan)
    return authorization


def _exception_failure(error: BaseException) -> dict[str, str]:
    return {
        "reason": "c8_unit_runner_raised",
        "phase": "run_execution_unit",
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
    }


def _result_row(
    unit: Any, result: Any, *, claim_token: str,
    caps: Mapping[str, Any], root: Path, plan: ExecutionPlan,
    estimate: StageCEstimate, observed_git_state: Mapping[str, Any],
) -> dict[str, Any]:
    from .stage_c_artifacts import validate_execution_unit

    state = getattr(result, "state", None)
    attempt_path = getattr(result, "attempt_path", None)
    failure = getattr(result, "failure", None)
    if isinstance(result, Mapping):
        state = result.get("state", state)
        attempt_path = result.get("attempt_path", attempt_path)
        failure = result.get("failure", failure)
    observed_state = state
    if state not in {"succeeded", "failed", "incomplete", "tampered"}:
        state = "incomplete"
        failure = {
            "reason": "invalid_unit_runner_result_state",
            "observed_state": observed_state,
        }
    status_sha256 = None
    if isinstance(attempt_path, str):
        attempt = Path(attempt_path)
        expected_parent = (
            root / "units" / unit.key.canonical_digest / "attempts"
        ).absolute()
        if (
            not attempt.is_absolute() or attempt.parent != expected_parent
            or _ATTEMPT_NAME.fullmatch(attempt.name) is None
            or attempt.is_symlink() or not attempt.is_dir()
        ):
            raise StageCBenchmarkIntegrityError(
                "C8 runner returned an attempt outside the claimed unit"
            )
        status_path = attempt / "status.json"
        if status_path.is_file() and not status_path.is_symlink():
            status_sha256 = _file_sha256(status_path)
    validation = None
    if isinstance(attempt_path, str):
        expected_bindings = {
            "configuration": {
                "execution_plan_sha256": plan.canonical_digest,
                "run_spec_sha256": canonical_sha256(plan.run_spec.to_dict()),
            },
            "resources": {
                "preflight": {
                    "plan_digest": plan.canonical_digest,
                    "canonical_digest": estimate.canonical_digest,
                }
            },
        }
        validation = validate_execution_unit(
            attempt, expected_unit_identity=unit.key.to_dict(),
            expected_provenance_bindings=expected_bindings,
            observed_git_state=observed_git_state, caps=caps,
            require_runner_telemetry=True,
        )
        if state == "succeeded" and not validation.resumable:
            state = "tampered"
            failure = {
                "reason": "post_unit_validation_failed",
                "errors": list(validation.errors),
            }
        elif state == "failed" and not (
            validation.valid and validation.status == "failed"
            and validation.terminal and not validation.complete
        ):
            state = "tampered"
            failure = {
                "reason": "failed_attempt_validation_failed",
                "errors": list(validation.errors),
            }
    return {
        "unit_key": unit.key.to_dict(),
        "state": state if isinstance(state, str) else "incomplete",
        "terminal": state in _TERMINAL_STATES,
        "complete": state == "succeeded",
        "selected_attempt": attempt_path if isinstance(attempt_path, str) else None,
        "selected_status_sha256": status_sha256,
        "failure": None if failure is None else _plain(failure),
        "claim_token": claim_token,
        "l0_l1_bytes": None if validation is None else validation.l0_l1_bytes,
        "l2_bytes": None if validation is None else validation.l2_bytes,
        "total_bytes": None if validation is None else validation.total_bytes,
    }


def _exception_row(
    unit: Any, error: BaseException, *, claim_token: str,
    caps: Mapping[str, Any], root: Path, plan: ExecutionPlan,
    estimate: StageCEstimate, observed_git_state: Mapping[str, Any],
) -> dict[str, Any]:
    retained = getattr(error, "stage_c_unit_result", None)
    if retained is not None:
        return _result_row(
            unit, retained, claim_token=claim_token, caps=caps, root=root,
            plan=plan, estimate=estimate,
            observed_git_state=observed_git_state,
        )
    return {
        "unit_key": unit.key.to_dict(),
        "state": "incomplete",
        "terminal": False,
        "complete": False,
        "selected_attempt": None,
        "selected_status_sha256": None,
        "failure": _exception_failure(error),
        "claim_token": claim_token,
        "l0_l1_bytes": None,
        "l2_bytes": None,
        "total_bytes": None,
    }


def _read_checkpoint_rows(root: Path, plan: ExecutionPlan) -> list[dict[str, Any]]:
    table = _strict_json(root / "failure_table.json")
    if (
        set(table) != {"schema_version", "document_type", "plan_digest", "rows"}
        or table.get("schema_version") != 1
        or table.get("document_type") != "stage_c_failure_table"
        or table.get("plan_digest") != plan.canonical_digest
        or not isinstance(table.get("rows"), list)
    ):
        raise StageCBenchmarkIntegrityError("C8 failure table header drifted")
    rows = table["rows"]
    expected = [unit.key.canonical_digest for unit in plan.units]
    observed = [
        row.get("unit_key", {}).get("canonical_digest")
        for row in rows if isinstance(row, Mapping)
    ]
    if len(rows) != C8_EXECUTION_UNITS or observed != expected:
        raise StageCBenchmarkIntegrityError(
            "C8 failure table must contain the exact ordered 2180 unit rows"
        )
    normalized = [_plain(row) for row in rows]
    for index, row in enumerate(normalized):
        if set(row) != _ROW_KEYS or row.get("state") not in _CHECKPOINT_STATES:
            raise StageCBenchmarkIntegrityError(
                f"C8 failure table row {index} fields/state drifted"
            )
        if row["unit_key"] != plan.units[index].key.to_dict():
            raise StageCBenchmarkIntegrityError(
                f"C8 failure table row {index} identity drifted"
            )
        state = row["state"]
        expected_terminal = state in _TERMINAL_STATES
        expected_complete = state == "succeeded"
        if (
            type(row["terminal"]) is not bool
            or type(row["complete"]) is not bool
            or row["terminal"] is not expected_terminal
            or row["complete"] is not expected_complete
        ):
            raise StageCBenchmarkIntegrityError(
                f"C8 failure table row {index} state flags drifted"
            )
        token = row["claim_token"]
        if state == "pending":
            if any(row[name] is not None for name in (
                "selected_attempt", "selected_status_sha256", "failure",
                "claim_token", "l0_l1_bytes", "l2_bytes", "total_bytes",
            )):
                raise StageCBenchmarkIntegrityError(
                    f"C8 pending row {index} contains execution evidence"
                )
        elif state in {"claimed", "succeeded", "failed", "incomplete"} and (
            not isinstance(token, str) or len(token) != 64
            or set(token) - _SHA256
        ):
            raise StageCBenchmarkIntegrityError(
                f"C8 non-pending row {index} lacks a valid claim token"
            )
        selected = row["selected_attempt"]
        status_sha = row["selected_status_sha256"]
        byte_fields = (row["l0_l1_bytes"], row["l2_bytes"], row["total_bytes"])
        if selected is None:
            if status_sha is not None or any(value is not None for value in byte_fields):
                raise StageCBenchmarkIntegrityError(
                    f"C8 row {index} has partial selected-attempt evidence"
                )
            if state in {"succeeded", "failed"}:
                raise StageCBenchmarkIntegrityError(
                    f"C8 terminal row {index} lacks a selected attempt"
                )
        else:
            if not isinstance(selected, str):
                raise StageCBenchmarkIntegrityError(
                    f"C8 row {index} selected attempt is invalid"
                )
            expected_parent = (
                root / "units" / row["unit_key"]["canonical_digest"] / "attempts"
            ).absolute()
            selected_path = Path(selected)
            if (
                not selected_path.is_absolute()
                or selected_path.parent != expected_parent
                or _ATTEMPT_NAME.fullmatch(selected_path.name) is None
                or selected_path.is_symlink()
            ):
                raise StageCBenchmarkIntegrityError(
                    f"C8 row {index} selected attempt escapes its unit"
                )
            if status_sha is not None and (
                not isinstance(status_sha, str) or len(status_sha) != 64
                or set(status_sha) - _SHA256
            ):
                raise StageCBenchmarkIntegrityError(
                    f"C8 row {index} selected status hash is invalid"
                )
            if state in {"succeeded", "failed"} and status_sha is None:
                raise StageCBenchmarkIntegrityError(
                    f"C8 terminal row {index} lacks a selected status hash"
                )
            if any(type(value) is not int or value < 0 for value in byte_fields):
                raise StageCBenchmarkIntegrityError(
                    f"C8 row {index} artifact-byte evidence is invalid"
                )
            if row["total_bytes"] != row["l0_l1_bytes"] + row["l2_bytes"]:
                raise StageCBenchmarkIntegrityError(
                    f"C8 row {index} artifact-byte totals disagree"
                )
    return normalized


def _validate_checkpoint_controls(
    root: Path, plan: ExecutionPlan, *,
    allowed_lifecycle: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_checkpoint_rows(root, plan)
    status = _strict_json(root / "batch_status.json")
    if set(status) != _STATUS_KEYS:
        raise StageCBenchmarkIntegrityError(
            "C8 batch status fields drifted"
        )
    lifecycle = status.get("lifecycle_state")
    if lifecycle not in _LIFECYCLE_STATES:
        raise StageCBenchmarkIntegrityError("C8 batch lifecycle state is invalid")
    if allowed_lifecycle is not None and lifecycle not in set(allowed_lifecycle):
        raise StageCBenchmarkIntegrityError(
            f"C8 batch lifecycle {lifecycle!r} forbids this operation"
        )
    errors = status.get("validation_errors")
    if not isinstance(errors, list) or any(
        not isinstance(item, str) or not item for item in errors
    ):
        raise StageCBenchmarkIntegrityError(
            "C8 batch validation_errors are invalid"
        )
    expected = _batch_status(
        plan, rows, _file_sha256(root / "failure_table.json"),
        validation_errors=errors, lifecycle_state=lifecycle,
    )
    if status != expected:
        raise StageCBenchmarkIntegrityError(
            "C8 batch status does not bind the current failure table"
        )
    return rows, status


def _validate_closed_layout(
    root: Path, plan: ExecutionPlan, *, require_all_units: bool,
) -> list[str]:
    """Reject unregistered files/directories throughout the C8 root."""

    errors: list[str] = []
    allowed_root = {
        *_CONTROL_NAMES, _CHECKPOINT_LOCK_FILENAME, _LIFECYCLE_LOCK_FILENAME,
        _ARTIFACT_ACTIVITY_LOCK_FILENAME,
        "units",
    }
    for child in root.iterdir():
        if child.name not in allowed_root:
            errors.append(f"unexpected C8 root entry: {child.name}")
        if child.is_symlink():
            errors.append(f"forbidden C8 root symlink: {child.name}")
    units_root = root / "units"
    expected = {unit.key.canonical_digest for unit in plan.units}
    observed: set[str] = set()
    if units_root.exists():
        if units_root.is_symlink() or not units_root.is_dir():
            errors.append("C8 units entry is not a safe directory")
            return errors
        for unit_root in units_root.iterdir():
            if unit_root.is_symlink() or not unit_root.is_dir():
                errors.append(f"unexpected or unsafe C8 units entry: {unit_root.name}")
                continue
            observed.add(unit_root.name)
            if unit_root.name not in expected:
                errors.append(f"unexpected C8 unit directory: {unit_root.name}")
                continue
            children = list(unit_root.iterdir())
            if len(children) != 1 or children[0].name != "attempts" or (
                children[0].is_symlink() or not children[0].is_dir()
            ):
                errors.append(f"C8 unit layout drifted: {unit_root.name}")
                continue
            for attempt in children[0].iterdir():
                if (
                    attempt.is_symlink() or not attempt.is_dir()
                    or _ATTEMPT_NAME.fullmatch(attempt.name) is None
                ):
                    errors.append(
                        f"unexpected C8 attempt entry: {unit_root.name}/{attempt.name}"
                    )
    if require_all_units and observed != expected:
        missing = sorted(expected - observed)
        errors.append(
            f"C8 units directory coverage mismatch; missing={len(missing)}"
        )
    return errors


def _claim_c8_unit(root: Path, plan: ExecutionPlan, unit: Any) -> str:
    """Atomically claim a never-attempted unit before backend execution."""

    from .stage_c_artifacts import attempt_directories

    digest = unit.key.canonical_digest
    with _checkpoint_lock(root):
        rows, _status = _validate_checkpoint_controls(
            root, plan, allowed_lifecycle=("running",)
        )
        indexes = {
            item["unit_key"]["canonical_digest"]: index
            for index, item in enumerate(rows)
        }
        if digest not in indexes:
            raise StageCBenchmarkIntegrityError("claimed unit is not in the C8 plan")
        index = indexes[digest]
        previous = rows[index]
        if previous.get("state") != "pending":
            raise StageCBenchmarkIntegrityError(
                f"C8 unit {digest} cannot be claimed from "
                f"{previous.get('state')!r}"
            )
        attempts = attempt_directories(root / "units" / digest / "attempts")
        if attempts:
            raise StageCBenchmarkIntegrityError(
                f"C8 unit {digest} has an attempt before claim"
            )
        claim_token = secrets.token_hex(32)
        rows[index] = {
            **previous,
            "state": "claimed",
            "claim_token": claim_token,
        }
        table = _failure_table(plan, rows)
        _atomic_replace_json(root / "failure_table.json", table)
        _atomic_replace_json(
            root / "batch_status.json",
            _batch_status(plan, rows, _file_sha256(root / "failure_table.json")),
        )
        return claim_token


def _selected_attempt_totals(
    root: Path, rows: Sequence[Mapping[str, Any]], caps: Mapping[str, Any]
) -> list[str]:
    from .stage_c_artifacts import validate_execution_unit

    errors: list[str] = []
    l2_by_sample: Counter[str] = Counter()
    for row in rows:
        selected = row.get("selected_attempt")
        if not isinstance(selected, str):
            continue
        attempt = Path(selected)
        if not attempt.is_absolute():
            errors.append("selected attempt path is not absolute")
            continue
        checked = validate_execution_unit(
            attempt, expected_unit_identity=row.get("unit_key"), caps=caps,
            require_runner_telemetry=True,
        )
        if checked.l0_l1_bytes > caps["bytes_per_l0_l1_execution_unit_max"]:
            errors.append(
                f"{row['unit_key']['canonical_digest']}: per-unit L0/L1 cap exceeded"
            )
        sample_id = row.get("unit_key", {}).get("sample_id")
        if sample_id in AUDIT_SAMPLE_IDS:
            l2_by_sample[str(sample_id)] += checked.l2_bytes
        elif checked.l2_bytes:
            errors.append(
                f"{row['unit_key']['canonical_digest']}: non-audit unit retained L2"
            )
    l2_cap = caps["bytes_per_audit_sample_l2_max"]
    for sample_id, byte_count in sorted(l2_by_sample.items()):
        if byte_count > l2_cap:
            errors.append(
                f"audit sample {sample_id} L2 bytes {byte_count} exceed cap {l2_cap}"
            )
    try:
        total_bytes = measure_artifact_tree_bytes(root).total_bytes
    except Exception as error:
        errors.append(f"cannot measure C8 artifact tree: {error}")
    else:
        total_cap = caps["stage_c_c8_c9_total_artifact_bytes_max"]
        if total_bytes > total_cap:
            errors.append(
                f"C8 artifact tree bytes {total_bytes} exceed cap {total_cap}"
            )
    return errors


def _publish_checkpoint(
    root: Path, plan: ExecutionPlan, row: Mapping[str, Any], *,
    caps: Mapping[str, Any], claim_token: str,
) -> None:
    digest = row.get("unit_key", {}).get("canonical_digest")
    with _checkpoint_lock(root):
        rows, checkpoint_status = _validate_checkpoint_controls(
            root, plan, allowed_lifecycle=("running", "blocked")
        )
        indexes = {
            item["unit_key"]["canonical_digest"]: index
            for index, item in enumerate(rows)
        }
        if digest not in indexes:
            raise StageCBenchmarkIntegrityError("checkpoint row is not in the C8 plan")
        previous = rows[indexes[digest]]
        if (
            previous.get("state") != "claimed"
            or previous.get("claim_token") != claim_token
            or row.get("claim_token") != claim_token
        ):
            raise StageCBenchmarkIntegrityError(
                f"C8 unit {digest} checkpoint claim/state mismatch"
            )
        rows[indexes[digest]] = _plain(row)
        completed = rows[indexes[digest]]
        cap_errors: list[str] = []
        if completed.get("l0_l1_bytes") is not None and (
            completed["l0_l1_bytes"]
            > caps["bytes_per_l0_l1_execution_unit_max"]
        ):
            cap_errors.append(f"{digest}: per-unit L0/L1 cap exceeded")
        sample_id = completed.get("unit_key", {}).get("sample_id")
        if sample_id not in AUDIT_SAMPLE_IDS and (completed.get("l2_bytes") or 0):
            cap_errors.append(f"{digest}: non-audit unit retained L2")
        l2_total = sum(
            row_item.get("l2_bytes") or 0 for row_item in rows
            if row_item.get("unit_key", {}).get("sample_id") == sample_id
        )
        if (
            sample_id in AUDIT_SAMPLE_IDS
            and l2_total > caps["bytes_per_audit_sample_l2_max"]
        ):
            cap_errors.append(
                f"audit sample {sample_id} L2 bytes {l2_total} exceed cap "
                f"{caps['bytes_per_audit_sample_l2_max']}"
            )
        # The frozen storage cap applies to every regular file retained under
        # the formal root, not merely to the attempts already represented by
        # terminal checkpoint rows.  Measure the tree after every completed
        # unit so controls, locks, partial concurrent attempts, and any
        # unexpected residue are all included in the online hard-cap gate.
        with _artifact_activity_lock(root, exclusive=True):
            try:
                tree_total = measure_artifact_tree_bytes(root).total_bytes
            except Exception as error:
                cap_errors.append(
                    f"cannot measure C8 artifact tree after {digest}: {error}"
                )
            else:
                total_cap = caps["stage_c_c8_c9_total_artifact_bytes_max"]
                if tree_total > total_cap:
                    cap_errors.append(
                        f"C8 artifact tree bytes {tree_total} exceed cap {total_cap}"
                    )
        if cap_errors:
            failed = rows[indexes[digest]]
            failed["state"] = "tampered"
            failed["terminal"] = False
            failed["complete"] = False
            failed["failure"] = {
                "reason": "online_cap_or_integrity_failure",
                "errors": cap_errors,
            }
        retained_errors = [
            *checkpoint_status["validation_errors"], *cap_errors,
        ]
        retained_errors = list(dict.fromkeys(retained_errors))
        table = _failure_table(plan, rows)
        _atomic_replace_json(root / "failure_table.json", table)
        _atomic_replace_json(
            root / "batch_status.json",
            _batch_status(
                plan, rows, _file_sha256(root / "failure_table.json"),
                validation_errors=retained_errors,
                lifecycle_state=(
                    "blocked"
                    if checkpoint_status["lifecycle_state"] == "blocked"
                    or retained_errors
                    else "running"
                ),
            ),
        )
        if cap_errors:
            raise StageCBenchmarkBlockedError(
                "C8 online artifact cap/integrity check failed", reasons=cap_errors
            )


def run_c8_shard(
    plan: ExecutionPlan,
    *,
    output_root: Path | str,
    backend: Any,
    estimate: StageCEstimate,
    unit_digests: Sequence[str],
    device: str = "cpu",
    caps: Mapping[str, Any] | None = None,
    observed_git_state: Mapping[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Run a disjoint shard and durably checkpoint after every unit.

    Formal C8 never retries in the same root: every unit digest may transition
    exactly once from ``pending``.  Ordinary unit-launch exceptions become an
    explicit incomplete row and the shard continues.  Process-control
    exceptions are checkpointed and then re-raised.
    """

    if not isinstance(plan, ExecutionPlan) or not isinstance(estimate, StageCEstimate):
        raise TypeError("plan and estimate must be formal Stage C objects")
    root = Path(os.path.abspath(os.fspath(output_root)))
    # Static controls are immutable.  Checkpoint controls are mutable across
    # shard processes and must only be read while holding their lock; reading
    # failure_table.json and batch_status.json between their two atomic
    # replacements would otherwise create a false tamper failure.
    authorization = _validate_initialized_controls(
        root, plan, estimate, validate_checkpoint=False
    )
    live_code = _require_live_git_state(
        authorization["code"], supplied=observed_git_state
    )
    requested = tuple(unit_digests)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("unit_digests must be non-empty and unique")
    by_digest = {unit.key.canonical_digest: unit for unit in plan.units}
    if any(item not in by_digest for item in requested):
        raise ValueError("shard contains an unknown unit digest")
    effective_caps = dict(estimate.caps["effective"])
    if caps is not None and dict(caps) != effective_caps:
        raise StageCBenchmarkIntegrityError(
            "C8 shard caps must equal the measured preflight caps"
        )
    results: list[Any] = []
    with _lifecycle_lock(root, exclusive=False):
        with _checkpoint_lock(root):
            _validate_checkpoint_controls(
                root, plan, allowed_lifecycle=("running",)
            )
        for digest in requested:
            unit = by_digest[digest]
            claim_token = _claim_c8_unit(root, plan, unit)
            result: Any | None = None
            try:
                with _artifact_activity_lock(root, exclusive=False):
                    result = run_execution_unit(
                        unit, output_root=root, backend=backend, device=device,
                        resume=False, caps=effective_caps, preflight=estimate,
                        plan=plan, observed_git_state=live_code,
                    )
            except (KeyboardInterrupt, SystemExit) as error:
                row = _exception_row(
                    unit, error, claim_token=claim_token, caps=effective_caps,
                    root=root, plan=plan, estimate=estimate,
                    observed_git_state=live_code,
                )
                try:
                    _publish_checkpoint(
                        root, plan, row, caps=effective_caps,
                        claim_token=claim_token,
                    )
                except BaseException as checkpoint_error:
                    if hasattr(error, "add_note"):
                        try:
                            error.add_note(
                                "C8 interrupt checkpoint also failed: "
                                f"{type(checkpoint_error).__module__}."
                                f"{type(checkpoint_error).__qualname__}: "
                                f"{checkpoint_error}"
                            )
                        except BaseException:
                            pass
                raise
            except Exception as error:
                row = _exception_row(
                    unit, error, claim_token=claim_token, caps=effective_caps,
                    root=root, plan=plan, estimate=estimate,
                    observed_git_state=live_code,
                )
                _publish_checkpoint(
                    root, plan, row, caps=effective_caps,
                    claim_token=claim_token,
                )
                results.append({
                    "unit_key": unit.key.to_dict(),
                    "state": "incomplete",
                    "attempt_path": None,
                    "resumed": False,
                    "failure": row["failure"],
                })
                continue
            row = _result_row(
                unit, result, claim_token=claim_token, caps=effective_caps,
                root=root, plan=plan, estimate=estimate,
                observed_git_state=live_code,
            )
            _publish_checkpoint(
                root, plan, row, caps=effective_caps, claim_token=claim_token
            )
            results.append(result)
    return tuple(results)


def _scan_rows(
    root: Path, plan: ExecutionPlan, estimate: StageCEstimate,
    *, observed_git_state: Mapping[str, Any],
    claim_tokens: Mapping[str, str | None] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    from .stage_c_artifacts import attempt_directories, validate_execution_unit

    expected = {
        "configuration": {
            "execution_plan_sha256": plan.canonical_digest,
            "run_spec_sha256": canonical_sha256(plan.run_spec.to_dict()),
        },
        "resources": {
            "preflight": {
                "plan_digest": plan.canonical_digest,
                "canonical_digest": estimate.canonical_digest,
            }
        },
    }
    caps = dict(estimate.caps["effective"])
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    expected_digests = {unit.key.canonical_digest for unit in plan.units}
    units_root = root / "units"
    if units_root.exists():
        for child in units_root.iterdir():
            if child.is_symlink() or not child.is_dir():
                errors.append(f"unexpected or unsafe C8 units entry: {child.name}")
            elif child.name not in expected_digests:
                errors.append(f"unexpected C8 unit directory: {child.name}")
    for unit in plan.units:
        digest = unit.key.canonical_digest
        selected = None
        state = "missing"
        failure = None
        status_hash = None
        l0_l1_bytes = l2_bytes = total_bytes = None
        try:
            attempts = attempt_directories(root / "units" / digest / "attempts")
        except Exception as error:
            attempts = ()
            errors.append(f"{digest}: cannot inspect attempts: {error}")
        if len(attempts) != 1:
            errors.append(f"{digest}: expected exactly one attempt, found {len(attempts)}")
            state = "missing" if not attempts else "tampered"
        else:
            selected = str(attempts[0])
            status_path = attempts[0] / "status.json"
            if status_path.is_file() and not status_path.is_symlink():
                status_hash = _file_sha256(status_path)
            checked = validate_execution_unit(
                attempts[0], expected_unit_identity=unit.key.to_dict(),
                expected_provenance_bindings=expected,
                observed_git_state=observed_git_state, caps=caps,
                require_runner_telemetry=True,
            )
            l0_l1_bytes = checked.l0_l1_bytes
            l2_bytes = checked.l2_bytes
            total_bytes = checked.total_bytes
            if not checked.valid:
                state = "tampered"
                errors.extend(f"{digest}: {item}" for item in checked.errors)
            elif checked.status == "complete" and checked.complete:
                state = "succeeded"
            elif checked.status == "failed" and checked.terminal:
                state = "failed"
                failure = _strict_json(status_path).get("failure")
            else:
                state = "incomplete"
                errors.append(f"{digest}: sole attempt is not terminal")
        rows.append({
            "unit_key": unit.key.to_dict(),
            "state": state,
            "terminal": state in _TERMINAL_STATES,
            "complete": state == "succeeded",
            "selected_attempt": selected,
            "selected_status_sha256": status_hash,
            "failure": failure,
            "claim_token": (
                None if claim_tokens is None else claim_tokens.get(digest)
            ),
            "l0_l1_bytes": l0_l1_bytes,
            "l2_bytes": l2_bytes,
            "total_bytes": total_bytes,
        })
    errors.extend(_selected_attempt_totals(root, rows, caps))
    return rows, errors


def finalize_c8_output_root(
    output_root: Path | str,
    plan: ExecutionPlan,
    *,
    observed_git_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rescan exact one-attempt layout and atomically publish final controls."""

    root = Path(os.path.abspath(os.fspath(output_root)))
    # The lifecycle lock waits for all shards.  The exclusive activity lock
    # additionally proves that no artifact writer outside the normal shard
    # call stack is active while finalization scans the tree.
    with (
        _lifecycle_lock(root, exclusive=True),
        _checkpoint_lock(root),
        _artifact_activity_lock(root, exclusive=True),
    ):
        preflight = _strict_json(root / "preflight.json")
        estimate = StageCEstimate.from_dict(preflight)
        authorization = _validate_initialized_controls(
            root, plan, estimate, require_issuer=False,
            validate_checkpoint=False,
        )
        code = _require_live_git_state(
            authorization["code"], supplied=observed_git_state
        )
        current_rows, current_status = _validate_checkpoint_controls(
            root, plan, allowed_lifecycle=("running", "blocked")
        )
        if current_status["active_claim_count"]:
            raise StageCBenchmarkBlockedError(
                "C8 cannot finalize while execution units are claimed"
            )
        finalizing = _batch_status(
            plan, current_rows, _file_sha256(root / "failure_table.json"),
            validation_errors=current_status["validation_errors"],
            lifecycle_state="finalizing",
        )
        _atomic_replace_json(root / "batch_status.json", finalizing)
        claim_tokens = {
            row["unit_key"]["canonical_digest"]: row["claim_token"]
            for row in current_rows
        }
        rows, errors = _scan_rows(
            root, plan, estimate, observed_git_state=code,
            claim_tokens=claim_tokens,
        )
        # An online hard-cap or integrity failure is an immutable part of this
        # attempt.  A later clean rescan must not erase the reason the batch
        # entered the blocked lifecycle.
        errors = [*current_status["validation_errors"], *errors]
        errors.extend(_validate_closed_layout(root, plan, require_all_units=True))
        errors = list(dict.fromkeys(errors))
        table = _failure_table(plan, rows)
        _atomic_replace_json(root / "failure_table.json", table)
        status = _batch_status(
            plan, rows, _file_sha256(root / "failure_table.json"),
            validation_errors=errors,
            lifecycle_state="finalized",
        )
        _atomic_replace_json(root / "batch_status.json", status)
    complete = status["orchestration_complete"] and not errors
    success = status["successful"] and not errors
    return {
        "status": "PASS" if success else "BLOCKED",
        "orchestration_complete": complete,
        "successful": success,
        "errors": errors,
        "rows": rows,
    }


__all__ = [
    "AUTHORIZATION_FILENAME", "C7_ARCHIVE_REF", "C7_CODE_COMMIT",
    "C7_PEAK_RESERVED_BYTES", "C7_PROJECTED_C8_ARTIFACT_BYTES",
    "C7_PROJECTED_C8_H20_GPU_HOURS",
    "C7_PROJECTED_C8_L2_BYTES_PER_AUDIT",
    "C7_REPORT_CANONICAL_DIGEST", "C7_REPORT_FILE_SHA256",
    "C7_SLOWEST_UNIT_SECONDS", "C8_EXECUTION_UNITS",
    "C8_EXPECTED_COUNTS", "C8_UNITS_PER_IMAGE",
    "build_c8_launch_authorization", "build_c8_measured_preflight",
    "finalize_c8_output_root", "initialize_c8_output_root",
    "run_c8_shard", "validate_c8_execution_plan",
    "validate_c8_launch_authorization", "validate_fixed_c7_report",
]
