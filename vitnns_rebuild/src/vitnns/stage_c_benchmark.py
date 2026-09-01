"""Read-only C7 benchmark gates, evidence mapping, and budget reports.

The module owns no model execution.  It consumes the immutable Stage C runner
tree, independently binds the small control/evidence files used by a report,
and applies the frozen C7 launch and downstream-budget rules.  Diagnostic and
formal reports must live outside their measured source trees so report writing
cannot change the measurement.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, ROUND_CEILING
from pathlib import Path, PurePath
from typing import Any

from .stage_c_plan import (
    AUDIT_SAMPLE_IDS,
    CLASS_METHOD_IDS,
    VALUE_METHOD_IDS,
    ExecutionPlan,
    StageCEstimate,
)


SCHEMA_VERSION = 1
STATUS_PASS = "PASS"
STATUS_BLOCKED = "BLOCKED"
EXIT_CLASS_SUCCESS = "success"
EXIT_CLASS_PREFLIGHT = "preflight"
EXIT_CLASS_TERMINAL = "terminal"
EXIT_CLASS_INTEGRITY = "integrity"

BENCHMARK_DOCUMENT_TYPE = "stage_c_c7_benchmark_report"
DIAGNOSTIC_DOCUMENT_TYPE = "stage_c_c7_diagnostic_gate"
LAUNCH_DOCUMENT_TYPE = "stage_c_c7_launch_gate"
REPORT_FILENAME = "stage_c_benchmark_report.json"
DIAGNOSTIC_GATE_FILENAME = "stage_c_c7_diagnostic_gate.json"
LAUNCH_AUTHORIZATION_FILENAME = "c7_launch_authorization.json"

DIAGNOSTIC_WALL_LIMIT_SECONDS = 720.0
C8_EXECUTION_UNITS = 2180
C9_EXECUTION_UNITS = 320
C8_C9_EXECUTION_UNITS = C8_EXECUTION_UNITS + C9_EXECUTION_UNITS
STAGE_D_EXECUTION_UNITS = 8000
STAGE_D_IMAGES = 500
STAGE_D_EXECUTION_UNITS_PER_IMAGE = 16
STAGE_C_SEED = 20260821
ABANDONED_C7_DIAGNOSTIC_COMMIT = (
    "357dd9d4a8b6dd2d43ff5bfbc85cc93a68145226"
)
IG_STEPS_SCHEDULE = (512, 1024, 2048, 4096, 8192)
IG_STEP_CHUNK_SIZE = 8

C7_VALUE_CANDIDATES = (
    (0, 0), (1, 5), (2, 11),
    (3, 5), (4, 11), (5, 0),
    (6, 11), (7, 0), (8, 5),
    (9, 0), (10, 5), (11, 11),
)
C7_FIDELITY_PAIRING = (0, 0)
C7_EXPECTED_COUNTS = {
    "execution_units": 40,
    "map_runs": 200,
    "fidelity_eligible_units": 28,
    "fidelity_curves": 1232,
    "fidelity_curve_points": 25872,
    "work_items": 26072,
}
C7_DIAGNOSTIC_METHOD_ID = (
    "conditional_fixed_clean_attention_head_direction_value_path_ig"
)
C7_DIAGNOSTIC_SAMPLE_ID = AUDIT_SAMPLE_IDS[0]
C7_DIAGNOSTIC_CANDIDATE = (0, 0)
C7_DIAGNOSTIC_EXPECTED_COUNTS = {
    "execution_units": 1,
    "map_runs": 5,
    "fidelity_eligible_units": 1,
    "fidelity_curves": 44,
    "fidelity_curve_points": 924,
    "work_items": 929,
}

HARD_CAP_KEYS = (
    "bytes_per_l0_l1_execution_unit_max",
    "bytes_per_audit_sample_l2_max",
    "stage_c_c8_c9_total_artifact_bytes_max",
    "stage_d_500_total_artifact_bytes_max",
    "forward_calls_per_execution_unit_max",
    "backward_calls_per_execution_unit_max",
    "cuda_peak_reserved_bytes_per_execution_unit_max",
    "stage_c_c8_c9_h20_gpu_hours_max",
    "stage_d_500_h20_gpu_hours_max",
    "stage_d_h20_gpu_hours_per_image_max",
)
FROZEN_CAPS: dict[str, int | float] = {
    "bytes_per_l0_l1_execution_unit_max": 64 * 1024 * 1024,
    "bytes_per_audit_sample_l2_max": 1024 * 1024 * 1024,
    "stage_c_c8_c9_total_artifact_bytes_max": 32 * 1024**3,
    "stage_d_500_total_artifact_bytes_max": 64 * 1024**3,
    "forward_calls_per_execution_unit_max": 30_000,
    "backward_calls_per_execution_unit_max": 30_000,
    "cuda_peak_reserved_bytes_per_execution_unit_max": 90 * 1024**3,
    "stage_c_c8_c9_h20_gpu_hours_max": 1000.0,
    "stage_d_500_h20_gpu_hours_max": 2000.0,
    "stage_d_h20_gpu_hours_per_image_max": 4.0,
    "projection_safety_factor": 1.25,
    "storage_control_overhead_factor": 1.25,
}
# Kept as a compatibility spelling for callers that treated the old design as
# an implementation detail.  New code should use the public constant above.
_FROZEN_CAPS = FROZEN_CAPS

RUNNER_PHASE_KEYS = (
    "attribution_or_map_compute",
    "fidelity_compute",
    "integrity_checking",
    "artifact_finalize",
)
RUNNER_TOTAL_BOUNDARY = (
    "unit_start_through_first_complete_validation_excluding_telemetry_publication"
)
SHARED_PROVENANCE_KEYS = (
    "code", "model", "input", "dependencies", "registry",
    "protocol", "acceptance", "configuration",
)
_ALL_METHODS = tuple((*VALUE_METHOD_IDS, *CLASS_METHOD_IDS))
_IG_METHODS = frozenset({
    C7_DIAGNOSTIC_METHOD_ID,
    "full_model_class_logit_ig",
})
_LEVELS = ("L0", "L1", "L2", "L3")
_CONTROL_FILENAMES = (
    "execution_plan.json", "preflight.json",
    "failure_table.json", "batch_status.json",
)
_FORMAL_CONTROL_FILENAMES = (*_CONTROL_FILENAMES, LAUNCH_AUTHORIZATION_FILENAME)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CUDA_DEVICE_RE = re.compile(r"^cuda:(0|[1-9][0-9]*)$")
_ATTEMPT_RE = re.compile(r"^attempt-[0-9]{6}$")
_POLLUTED_PATH_RE = re.compile(
    r"(?:^|[-_.\/])(?:archive|archived|c7[-_]?overhash|357dd9d|"
    r"retry[0-9]*|old[-_]?attempt|attempt[-_]?20[0-9]{6})"
    r"(?:$|[-_.\/])",
    re.IGNORECASE,
)


class StageCBenchmarkError(ValueError):
    """Base error for the C7 reporting contract."""


class StageCBenchmarkIntegrityError(StageCBenchmarkError):
    """A source tree, gate, report, hash, or binding is malformed."""


class StageCBenchmarkBlockedError(StageCBenchmarkError):
    """A valid gate prevents formal launch without claiming corruption."""

    def __init__(
        self, message: str, *, report: Mapping[str, Any] | None = None,
        reasons: Iterable[str] = (),
    ) -> None:
        super().__init__(message)
        self.report = None if report is None else _json_value(report, "report")
        supplied = tuple(str(item) for item in reasons if str(item))
        if not supplied and self.report is not None:
            blocked = self.report.get("blocked_reasons", ())
            if isinstance(blocked, list):
                supplied = tuple(str(item) for item in blocked if str(item))
        self.reasons = supplied or (message,)


# Alternate name used by some runner integrations.
StageCLaunchBlockedError = StageCBenchmarkBlockedError


def _json_value(value: Any, context: str = "value") -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageCBenchmarkIntegrityError(
                f"{context} contains a non-finite number"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise StageCBenchmarkIntegrityError(
                    f"{context} keys must be non-empty strings"
                )
            result[key] = _json_value(item, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict(), context)
    raise StageCBenchmarkIntegrityError(
        f"{context} is not JSON-compatible: {type(value).__name__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value, "document"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_digest(value: Mapping[str, Any]) -> str:
    return _canonical_digest({
        key: item for key, item in value.items() if key != "canonical_digest"
    })


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise StageCBenchmarkIntegrityError(
                    f"duplicate JSON key in {path}: {key}"
                )
            result[key] = item
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                StageCBenchmarkIntegrityError(
                    f"non-finite JSON value in {path}: {token}"
                )
            ),
        )
    except StageCBenchmarkIntegrityError:
        raise
    except Exception as error:
        raise StageCBenchmarkIntegrityError(
            f"cannot parse strict JSON {path}: {error}"
        ) from error


def _plain_plan(plan: ExecutionPlan) -> dict[str, Any]:
    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")
    try:
        return ExecutionPlan.from_dict(plan.to_dict()).to_dict()
    except Exception as error:
        raise StageCBenchmarkIntegrityError(
            f"execution plan failed its canonical round trip: {error}"
        ) from error


def _config_dict(config: Any) -> dict[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    value = to_dict() if callable(to_dict) else config
    if not isinstance(value, Mapping):
        raise TypeError("config must be a StageCConfig or mapping")
    return _json_value(value, "config")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _polluted_path(path: Path | str) -> bool:
    return _POLLUTED_PATH_RE.search(str(path)) is not None


def _require_unpolluted_path(path: Path, context: str) -> None:
    if _polluted_path(path):
        raise StageCBenchmarkIntegrityError(
            f"{context} contains a forbidden archive/old-attempt path marker"
        )


def _lexical_absolute_path(value: Path | str) -> Path:
    """Return an absolute path without following any symbolic links."""

    return Path(os.path.abspath(os.fspath(value)))


def _path_traverses_symlink(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _require_authorized_recovery_path(
    value: Path | str, expected: Path | str, context: str,
) -> Path:
    """Match a launch-authorized path without accepting symlink aliases."""

    actual = _lexical_absolute_path(value)
    authorized = _lexical_absolute_path(expected)
    if actual != authorized:
        raise StageCBenchmarkIntegrityError(
            f"{context} disagrees with RecoveryC7LaunchRecord outputs"
        )
    if _path_traverses_symlink(actual):
        raise StageCBenchmarkIntegrityError(
            f"{context} traverses a symlink"
        )
    return actual


def _plan_counts(document: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    units = document.get("units")
    if not isinstance(units, list):
        return {key: 0 for key in C7_EXPECTED_COUNTS}
    counts["execution_units"] = len(units)
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        item_counts = unit.get("work_item_counts")
        if not isinstance(item_counts, Mapping):
            continue
        for key in (
            "map_runs", "fidelity_curves",
            "fidelity_curve_points", "work_items",
        ):
            value = item_counts.get(key)
            if type(value) is int:
                counts[key] += value
        counts["fidelity_eligible_units"] += int(
            unit.get("fidelity_eligible") is True
        )
    return {key: int(counts[key]) for key in C7_EXPECTED_COUNTS}


def _validate_common_run_flags(spec: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    if spec.get("value_config_source") != "explicit":
        violations.append("value_config_source must be explicit")
    if spec.get("audit_repeatability_runs") != 2:
        violations.append("audit repeatability must contain two clean runs")
    if spec.get("audit_repeatability_scope") != "clean_map_only":
        violations.append("audit repeatability scope must be clean_map_only")
    for field in (
        "run_registered_fidelity", "run_sanity", "run_stability",
    ):
        if spec.get(field) is not True:
            violations.append(f"{field} must be true")
    if spec.get("fidelity_pairing_context") != {
        "block_index": C7_FIDELITY_PAIRING[0],
        "head_index": C7_FIDELITY_PAIRING[1],
    }:
        violations.append("fidelity pairing must be block 0 head 0")
    return violations


def validate_c7_execution_plan(
    plan: ExecutionPlan, config: Any,
) -> dict[str, Any]:
    """Return the exact formal C7 scope and fail-closed violations."""

    document = _plain_plan(plan)
    config_value = _config_dict(config)
    spec = document["run_spec"]
    units = document["units"]
    violations = _validate_common_run_flags(spec)
    sample_ids = spec.get("sample_ids")
    if sample_ids != [C7_DIAGNOSTIC_SAMPLE_ID]:
        violations.append(
            "formal C7 requires exactly the frozen calibration audit sample"
        )
    if spec.get("method_ids") != list(_ALL_METHODS):
        violations.append("formal C7 requires the seven frozen methods in order")
    candidate_values = spec.get("value_candidates")
    expected_candidates = [
        {"block_index": block, "head_index": head}
        for block, head in C7_VALUE_CANDIDATES
    ]
    if candidate_values != expected_candidates:
        violations.append("formal C7 candidates do not equal the exact K=12 set")
    if document.get("selection", {}).get("status") != "not_applicable_pre_c8":
        violations.append("C7 cannot consume or create a C8 selection")
    if {unit.get("sample", {}).get("sample_id") for unit in units} != {
        C7_DIAGNOSTIC_SAMPLE_ID
    }:
        violations.append("formal units do not cover exactly the declared audit sample")
    if any(
        unit.get("sample", {}).get("audit") is not True
        or unit.get("sample", {}).get("role") != "calibration"
        for unit in units
    ):
        violations.append("every C7 unit must bind the calibration audit sample")
    counts = _plan_counts(document)
    if counts != C7_EXPECTED_COUNTS:
        violations.append(
            f"formal C7 expanded counts mismatch: {counts!r}"
        )
    method_counts = Counter(
        unit.get("key", {}).get("method_id") for unit in units
    )
    expected_method_counts = {
        **{method: 12 for method in VALUE_METHOD_IDS},
        **{method: 1 for method in CLASS_METHOD_IDS},
    }
    if dict(method_counts) != expected_method_counts:
        violations.append("formal C7 per-method execution-unit counts drifted")
    method_ids = [item.get("id") for item in config_value.get("methods", [])]
    if method_ids != list(_ALL_METHODS):
        violations.append("config method order/set differs from the frozen methods")
    bands = config_value.get("layers_heads", {}).get("depth_bands")
    expected_bands = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    if bands != expected_bands:
        violations.append("config depth bands differ from the frozen four bands")
    coverage: list[dict[str, Any]] = []
    for index, band in enumerate(expected_bands):
        selected = [
            {"block_index": block, "head_index": head}
            for block, head in C7_VALUE_CANDIDATES if block in band
        ]
        coverage.append({
            "depth_band": index, "blocks": band,
            "candidates": selected, "covered": len(selected) == 3,
        })
    return {
        "valid": not violations,
        "violations": violations,
        "sample_ids": _json_value(sample_ids, "sample_ids"),
        "method_ids": _json_value(spec.get("method_ids"), "method_ids"),
        "candidate_policy": {
            "result_independent": True,
            "candidate_count": len(candidate_values)
            if isinstance(candidate_values, list) else 0,
            "candidates": expected_candidates,
            "block_coverage": "all_12_blocks_exactly_once",
            "head_strata": [0, 5, 11],
            "depth_band_coverage": coverage,
            "forbidden_selection_inputs": [
                "attribution_results", "convergence_results",
                "runtime_results", "prediction_correctness",
                "image_content", "heatmaps", "holdout_data",
            ],
        },
        "counts": counts,
        "execution_plan_sha256": plan.canonical_digest,
        "config_sha256": _canonical_digest(config_value),
    }


def validate_c7_diagnostic_execution_plan(
    plan: ExecutionPlan, config: Any,
) -> dict[str, Any]:
    """Validate the exact one-unit audit NNS calibration workload."""

    document = _plain_plan(plan)
    config_value = _config_dict(config)
    spec = document["run_spec"]
    units = document["units"]
    violations = _validate_common_run_flags(spec)
    if spec.get("sample_ids") != [C7_DIAGNOSTIC_SAMPLE_ID]:
        violations.append("diagnostic requires the frozen audit calibration sample")
    if spec.get("method_ids") != [C7_DIAGNOSTIC_METHOD_ID]:
        violations.append("diagnostic requires only the frozen NNS method")
    if spec.get("value_candidates") != [{
        "block_index": C7_DIAGNOSTIC_CANDIDATE[0],
        "head_index": C7_DIAGNOSTIC_CANDIDATE[1],
    }]:
        violations.append("diagnostic requires exactly block 0 head 0")
    if document.get("selection", {}).get("status") != "not_applicable_pre_c8":
        violations.append("diagnostic cannot consume or create a C8 selection")
    counts = _plan_counts(document)
    expected_counts = C7_DIAGNOSTIC_EXPECTED_COUNTS
    if counts != expected_counts:
        violations.append(f"diagnostic expanded counts mismatch: {counts!r}")
    if len(units) != 1:
        violations.append("diagnostic execution plan must contain exactly one unit")
        unit = None
    else:
        unit = units[0]
        key = unit.get("key", {})
        sample = unit.get("sample", {})
        if (
            key.get("method_id") != C7_DIAGNOSTIC_METHOD_ID
            or key.get("block_index") != 0
            or key.get("head_index") != 0
            or sample.get("sample_id") != C7_DIAGNOSTIC_SAMPLE_ID
            or sample.get("audit") is not True
            or sample.get("role") != "calibration"
        ):
            violations.append("diagnostic unit identity is not the frozen audit NNS unit")
        if unit.get("fidelity_eligible") is not True:
            violations.append("diagnostic NNS unit must retain the full fidelity workload")
    method_ids = [item.get("id") for item in config_value.get("methods", [])]
    if method_ids != list(_ALL_METHODS):
        violations.append("diagnostic config is not the frozen Stage C config")
    return {
        "valid": not violations,
        "violations": violations,
        "sample_ids": _json_value(spec.get("sample_ids"), "sample_ids"),
        "method_ids": _json_value(spec.get("method_ids"), "method_ids"),
        "candidate": {"block_index": 0, "head_index": 0},
        "counts": counts,
        "unit": None if unit is None else _json_value(unit, "diagnostic unit"),
        "execution_plan_sha256": plan.canonical_digest,
        "config_sha256": _canonical_digest(config_value),
    }


validate_c7_diagnostic_plan = validate_c7_diagnostic_execution_plan


def capture_benchmark_environment(device: str = "cuda:0") -> dict[str, Any]:
    """Capture the report process environment without mutating torch state."""

    if not isinstance(device, str) or not device:
        raise TypeError("device must be a non-empty string")
    try:
        import torch
    except Exception as error:  # pragma: no cover - deployment dependency
        raise StageCBenchmarkIntegrityError(
            f"cannot import torch for environment capture: {error}"
        ) from error

    cuda_requested = _CUDA_DEVICE_RE.fullmatch(device) is not None
    cuda_available = bool(torch.cuda.is_available())
    device_name = None
    total_memory = None
    compute_capability = None
    driver_version = None
    if cuda_requested and cuda_available:
        try:
            index = int(device.split(":", 1)[1])
            properties = torch.cuda.get_device_properties(index)
            device_name = str(properties.name)
            total_memory = int(properties.total_memory)
            compute_capability = [int(properties.major), int(properties.minor)]
            try:
                driver_version = str(torch._C._cuda_getDriverVersion())
            except (AttributeError, RuntimeError):
                try:
                    output = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=driver_version",
                            "--format=csv,noheader",
                            f"--id={index}",
                        ],
                        check=True, capture_output=True, text=True, timeout=10,
                    ).stdout.splitlines()
                    driver_version = output[0].strip() if output else None
                except Exception:
                    driver_version = None
        except Exception:
            device_name = None
            total_memory = None
            compute_capability = None
    cudnn_version = (
        torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available() else None
    )
    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    return {
        "schema_version": 1,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "device_type": "cuda" if cuda_requested else "cpu",
        "device": device,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "cuda_device_name": device_name,
        "cuda_total_memory_bytes": total_memory,
        "cuda_compute_capability": compute_capability,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "cuda_driver_version": driver_version,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_allow_tf32": (
            None if matmul_backend is None
            else bool(matmul_backend.allow_tf32)
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "seed": STAGE_C_SEED,
    }


def _environment_errors(environment: Any) -> list[str]:
    if not isinstance(environment, Mapping):
        return ["benchmark environment must be an object"]
    errors: list[str] = []
    device = environment.get("device")
    device_type = environment.get("device_type", "cuda" if isinstance(device, str) and device.startswith("cuda:") else None)
    name = environment.get("device_name", environment.get("cuda_device_name"))
    capability = environment.get("cuda_compute_capability")
    if not isinstance(device, str) or _CUDA_DEVICE_RE.fullmatch(device) is None:
        errors.append("benchmark device must be an explicit cuda:<index>")
    if device_type != "cuda":
        errors.append("benchmark device_type must be cuda")
    if environment.get("cuda_available") is not True:
        errors.append("CUDA was not available to the benchmark process")
    if not isinstance(name, str) or re.fullmatch(
        r"(?:NVIDIA )?H20(?: NVL)?", name.strip(), flags=re.IGNORECASE
    ) is None:
        errors.append("device_name is not an NVIDIA H20")
    if capability != [9, 0]:
        errors.append("H20 CUDA compute capability must be [9, 0]")
    if not isinstance(environment.get("cuda_runtime_version"), str):
        errors.append("CUDA runtime version was not measured")
    if environment.get("cudnn_version") is None:
        errors.append("cuDNN version was not measured")
    if environment.get("deterministic_algorithms") is not True:
        errors.append("deterministic algorithms were not enabled")
    if environment.get("cudnn_benchmark") is not False:
        errors.append("cuDNN benchmarking was not disabled")
    if environment.get("cublas_workspace_config") not in {":4096:8", ":16:8"}:
        errors.append("CUBLAS_WORKSPACE_CONFIG is not deterministic")
    if environment.get("cuda_matmul_allow_tf32") is not False:
        errors.append("CUDA matmul TF32 was not disabled")
    if environment.get("cudnn_allow_tf32") is not False:
        errors.append("cuDNN TF32 was not disabled")
    if environment.get("seed") != STAGE_C_SEED:
        errors.append("benchmark seed is not the frozen Stage C seed")
    return errors


def _control_evidence(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    value: dict[str, Any] = {
        "path": str(path), "present": False, "bytes": None,
        "sha256": None, "document": None, "error": None,
    }
    if path.is_symlink() or not path.is_file():
        value["error"] = "missing, non-regular, or symlink control"
        return value
    try:
        value.update({
            "present": True, "bytes": path.stat().st_size,
            "sha256": _sha256_file(path), "document": _strict_json(path),
        })
    except Exception as error:
        value["error"] = str(error)
    return value


def _source_controls(
    root: Path, names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    return {name: _control_evidence(root, name) for name in names}


def _safe_attempt(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    attempt = candidate.resolve()
    if not _path_is_within(attempt, root):
        return None
    if attempt.is_symlink() or not attempt.is_dir():
        return None
    if attempt.name and _ATTEMPT_RE.fullmatch(attempt.name) is None:
        return None
    try:
        attempt.relative_to((root / "units").resolve())
    except ValueError:
        return None
    return attempt


def _read_small_json(path: Path, name: str, errors: list[str]) -> tuple[Any, str | None]:
    if path.is_symlink() or not path.is_file():
        errors.append(f"missing or unsafe {name}")
        return {}, None
    try:
        return _strict_json(path), _sha256_file(path)
    except Exception as error:
        errors.append(f"invalid {name}: {error}")
        return {}, None


def _measure_attempt_from_manifest(
    attempt: Path, manifest: Any,
) -> tuple[dict[str, int], int, list[str]]:
    """Verify manifest inventory and sum exact C6 levels plus controls."""

    errors: list[str] = []
    levels = {level: 0 for level in _LEVELS}
    records = manifest.get("artifacts") if isinstance(manifest, Mapping) else None
    if not isinstance(records, list):
        return {**levels, "control": 0}, 0, [
            "manifest artifacts must be an array"
        ]
    expected_files = {"manifest.json", "status.json"}
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(f"artifact[{index}] is not an object")
            continue
        level = record.get("level")
        relative = record.get("path")
        byte_count = record.get("bytes")
        digest = record.get("sha256")
        if (
            level not in levels or not isinstance(relative, str)
            or not relative or type(byte_count) is not int or byte_count < 0
            or not _is_sha256(digest)
        ):
            errors.append(f"artifact[{index}] metadata is invalid")
            continue
        if relative in seen:
            errors.append(f"duplicate artifact path: {relative}")
            continue
        seen.add(relative)
        pure = PurePath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.parts[0] != level:
            errors.append(f"unsafe or level-mismatched artifact path: {relative}")
            continue
        path = (attempt / relative).resolve()
        if not _path_is_within(path, attempt) or path.is_symlink() or not path.is_file():
            errors.append(f"missing or unsafe artifact: {relative}")
            continue
        observed = path.stat().st_size
        if observed != byte_count or _sha256_file(path) != digest:
            errors.append(f"artifact byte/hash mismatch: {relative}")
        levels[level] += observed
        expected_files.add(relative)
    controls = 0
    for name in ("manifest.json", "status.json"):
        path = attempt / name
        if path.is_symlink() or not path.is_file():
            errors.append(f"missing or unsafe attempt control: {name}")
        else:
            controls += path.stat().st_size
    actual_files: set[str] = set()
    for path in attempt.rglob("*"):
        if path.is_symlink():
            errors.append(f"attempt contains a symlink: {path}")
        elif path.is_file():
            actual_files.add(path.relative_to(attempt).as_posix())
    if actual_files != expected_files:
        errors.append(
            "attempt regular-file inventory mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    result = {**levels, "control": controls}
    return result, sum(result.values()), errors


def _telemetry_errors(
    telemetry: Any, *, environment: Mapping[str, Any] | None = None,
    require_success: bool = True,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(telemetry, Mapping):
        return ["status runner_telemetry is missing or not an object"]
    if set(telemetry) != {
        "schema_version", "phase_timings", "lifecycle",
        "measurement_environment",
    }:
        errors.append("runner_telemetry fields differ from the frozen shape")
    if telemetry.get("schema_version") != 1:
        errors.append("runner_telemetry schema_version is not 1")
    timings = telemetry.get("phase_timings")
    if not isinstance(timings, Mapping):
        errors.append("runner phase_timings is missing")
    else:
        if timings.get("clock") != "perf_counter":
            errors.append("runner phase clock is not perf_counter")
        if timings.get("synchronization") != "device_before_and_after_each_phase":
            errors.append("runner phase synchronization contract drifted")
        if timings.get("phases_are_mutually_exclusive") is not False:
            errors.append("runner phases must declare non-exclusive timing")
        if timings.get("total_boundary") != RUNNER_TOTAL_BOUNDARY:
            errors.append("runner synchronized total boundary drifted")
        total = timings.get("synchronized_total_wall_time_seconds")
        if (
            isinstance(total, bool) or not isinstance(total, (int, float))
            or not math.isfinite(float(total)) or float(total) < 0.0
        ):
            errors.append("runner synchronized total wall time is invalid")
        phases = timings.get("phases")
        if not isinstance(phases, Mapping) or set(phases) != set(RUNNER_PHASE_KEYS):
            errors.append("runner phases do not equal the frozen ordered phase set")
        else:
            for name in RUNNER_PHASE_KEYS:
                item = phases[name]
                if not isinstance(item, Mapping) or set(item) != {
                    "wall_time_seconds", "invocations"
                }:
                    errors.append(f"runner phase {name} has the wrong fields")
                    continue
                wall = item.get("wall_time_seconds")
                invocations = item.get("invocations")
                if (
                    isinstance(wall, bool) or not isinstance(wall, (int, float))
                    or not math.isfinite(float(wall)) or float(wall) < 0.0
                ):
                    errors.append(f"runner phase {name} wall time is invalid")
                elif (
                    isinstance(total, (int, float))
                    and not isinstance(total, bool) and float(wall) > float(total)
                ):
                    errors.append(f"runner phase {name} exceeds synchronized total")
                if type(invocations) is not int or invocations < 0:
                    errors.append(f"runner phase {name} invocations is invalid")
    lifecycle = telemetry.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or set(lifecycle) != {
        "integrity", "restoration"
    }:
        errors.append(
            "runner lifecycle must contain integrity and restoration evidence"
        )
    else:
        integrity = lifecycle.get("integrity")
        required = {
            "schema_version", "algorithm", "checked",
            "entry_sha256", "exit_sha256", "matched", "restored",
            "differences", "signature_calls", "verification_error_type",
            "verification_error_message", "body_exception_type",
        }
        if not isinstance(integrity, Mapping) or set(integrity) != required:
            errors.append("runner integrity evidence fields drifted")
        elif (
            integrity.get("schema_version") != 1
            or integrity.get("algorithm") != "stage_c_strong_model_sha256_v1"
            or not isinstance(integrity.get("checked"), list)
            or not integrity.get("checked")
            or integrity.get("matched") is not True
            or integrity.get("restored") is not True
            or integrity.get("differences") != []
            or not _is_sha256(integrity.get("entry_sha256"))
            or integrity.get("exit_sha256") != integrity.get("entry_sha256")
            or integrity.get("signature_calls") != 2
            or integrity.get("verification_error_type") is not None
            or integrity.get("verification_error_message") is not None
            or (require_success and integrity.get("body_exception_type") is not None)
            or (
                not require_success
                and integrity.get("body_exception_type") is not None
                and (
                    not isinstance(integrity.get("body_exception_type"), str)
                    or not integrity.get("body_exception_type")
                )
            )
        ):
            errors.append("runner lifecycle did not prove strong exit restoration")
        restoration = lifecycle.get("restoration")
        restoration_fields = {
            "schema_version", "checked", "changes_detected",
            "restored", "differences", "errors",
        }
        if (
            not isinstance(restoration, Mapping)
            or set(restoration) != restoration_fields
            or restoration.get("schema_version") != 1
            or restoration.get("restored") is not True
            or type(restoration.get("changes_detected")) is not bool
            or not isinstance(restoration.get("checked"), list)
            or not restoration.get("checked")
            or not isinstance(restoration.get("differences"), list)
            or restoration.get("changes_detected")
            is not bool(restoration.get("differences"))
            or restoration.get("errors") != []
        ):
            errors.append(
                "runner lifecycle restoration evidence is missing or unsuccessful"
            )
    measured = telemetry.get("measurement_environment")
    expected_measurement_keys = {
        "device_type", "device", "device_name",
        "cuda_runtime_version", "cudnn_version",
    }
    if not isinstance(measured, Mapping) or set(measured) != expected_measurement_keys:
        errors.append("runner measurement_environment has the wrong shape")
    elif environment is not None:
        expected = {
            "device_type": environment.get("device_type", "cuda"),
            "device": environment.get("device"),
            "device_name": environment.get(
                "device_name", environment.get("cuda_device_name")
            ),
            "cuda_runtime_version": environment.get("cuda_runtime_version"),
            "cudnn_version": environment.get("cudnn_version"),
        }
        if dict(measured) != expected:
            errors.append("execution measurement environment differs from report environment")
    return errors


def _normalize_telemetry_from_status(
    status: Mapping[str, Any], resources: Mapping[str, Any],
) -> Any:
    telemetry = status.get("runner_telemetry")
    if telemetry is not None:
        return telemetry
    # Transitional read compatibility only.  A formal/diagnostic PASS still
    # rejects this path because it cannot bind telemetry through status.
    actual = resources.get("actual") if isinstance(resources, Mapping) else None
    if isinstance(actual, Mapping):
        candidate = actual.get("runner_telemetry")
        if candidate is not None:
            return candidate
        phase_timings = actual.get("phase_timings")
        lifecycle = actual.get("lifecycle")
        measurement = actual.get("measurement_environment")
        if all(item is not None for item in (phase_timings, lifecycle, measurement)):
            return {
                "schema_version": 1, "phase_timings": phase_timings,
                "lifecycle": lifecycle,
                "measurement_environment": measurement,
            }
    return None


def _map_records_from_sources(
    result: Mapping[str, Any], resources: Mapping[str, Any],
) -> list[dict[str, Any]]:
    scalars = result.get("scalars")
    candidates: Any = None
    if isinstance(scalars, Mapping):
        candidates = scalars.get("map_execution_records")
    if candidates is None:
        actual = resources.get("actual") if isinstance(resources, Mapping) else None
        if isinstance(actual, Mapping):
            candidates = actual.get("map_execution_records")
    if not isinstance(candidates, list):
        return []
    result_records: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        record = _json_value(item, "map_execution_record")
        if record.get("ig_result") is None:
            fragment = record.get("provenance_fragment")
            if isinstance(fragment, Mapping):
                record["ig_result"] = fragment.get("ig_result")
        result_records.append(record)
    return result_records


def _read_attempt_evidence(
    attempt: Path, identity: Mapping[str, Any], *, trusted: bool, state: str,
    selected_status_sha256: Any,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest, manifest_hash = _read_small_json(
        attempt / "manifest.json", "manifest.json", errors
    )
    status, status_hash = _read_small_json(
        attempt / "status.json", "status.json", errors
    )
    provenance, provenance_hash = _read_small_json(
        attempt / "L0/provenance.json", "L0/provenance.json", errors
    )
    result_path = attempt / "L0/result.json"
    if state == "succeeded" or result_path.is_file():
        result, result_hash = _read_small_json(
            result_path, "L0/result.json", errors
        )
    else:
        result, result_hash = {}, None
    if not isinstance(manifest, Mapping):
        errors.append("manifest is not an object")
        manifest = {}
    if not isinstance(status, Mapping):
        errors.append("status is not an object")
        status = {}
    if not isinstance(provenance, Mapping):
        errors.append("provenance is not an object")
        provenance = {}
    if not isinstance(result, Mapping):
        errors.append("result is not an object")
        result = {}
    if status.get("manifest_sha256") != manifest_hash:
        errors.append("status manifest hash differs from manifest bytes")
    if selected_status_sha256 != status_hash:
        errors.append("failure table selected_status_sha256 differs from status bytes")
    if manifest.get("provenance_sha256") != provenance_hash:
        errors.append("manifest provenance hash differs from provenance bytes")
    if manifest.get("unit_identity") != dict(identity):
        errors.append("manifest unit identity differs from the execution plan")
    result_records = [
        item for item in manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("path") == "L0/result.json"
    ]
    if state == "succeeded" and (
        len(result_records) != 1 or result_records[0].get("sha256") != result_hash
    ):
        errors.append("manifest result hash differs from L0/result.json bytes")
    levels, total_bytes, measurement_errors = _measure_attempt_from_manifest(
        attempt, manifest
    )
    errors.extend(measurement_errors)
    bindings = provenance.get("bindings")
    if not isinstance(bindings, Mapping):
        errors.append("provenance bindings are missing")
        bindings = {}
    resource_binding = bindings.get("resources")
    if not isinstance(resource_binding, Mapping):
        errors.append("provenance resource binding is missing")
        resource_binding = {}
    actual_resources = resource_binding.get("actual")
    if not isinstance(actual_resources, Mapping):
        errors.append("provenance actual resources are missing")
        actual_resources = {}
    telemetry = _normalize_telemetry_from_status(status, resource_binding)
    if "runner_telemetry" not in status:
        errors.append("runner telemetry is not bound directly by status.json")
    map_records = _map_records_from_sources(result, resource_binding)
    shared = {
        key: _json_value(bindings[key], f"bindings.{key}")
        for key in SHARED_PROVENANCE_KEYS if key in bindings
    }
    artifact_inventory = _json_value(
        manifest.get("artifacts", []), "artifact inventory"
    )
    return {
        "unit_key": _json_value(identity, "unit_key"),
        "state": state,
        "trusted_by_batch_validator": bool(trusted and not errors),
        "attempt_path": str(attempt),
        "hashes": {
            "manifest_sha256": manifest_hash,
            "status_sha256": status_hash,
            "selected_status_sha256": selected_status_sha256,
            "provenance_sha256": provenance_hash,
            "result_sha256": result_hash,
            "unit_identity_sha256": manifest.get("unit_identity_sha256"),
        },
        "resources": _json_value(actual_resources, "actual resources"),
        "runner_telemetry": _json_value(telemetry, "runner telemetry")
        if telemetry is not None else None,
        "level_bytes": levels,
        "l0_l1_cap_accounting_bytes": (
            levels["L0"] + levels["L1"]
            + levels["L3"] + levels["control"]
        ),
        "total_regular_file_bytes": total_bytes,
        "map_execution_records": map_records,
        "artifact_inventory": artifact_inventory,
        "shared_provenance": shared,
        "shared_provenance_sha256": _canonical_digest(shared)
        if len(shared) == len(SHARED_PROVENANCE_KEYS) else None,
        "failure": status.get("failure", result.get("failure")),
        "errors": errors,
    }, errors


def _failure_rows_by_digest(control: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    document = control.get("document")
    rows = document.get("rows") if isinstance(document, Mapping) else None
    if not isinstance(rows, list):
        return {}, ["failure table rows are unavailable"]
    result: dict[str, Any] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"failure table row {index} is not an object")
            continue
        identity = row.get("unit_key")
        digest = identity.get("canonical_digest") if isinstance(identity, Mapping) else None
        if not _is_sha256(digest) or digest in result:
            errors.append(f"failure table row {index} identity is invalid or duplicate")
            continue
        result[digest] = row
    return result, errors


def map_execution_records(
    output_root: Path | str, plan: ExecutionPlan, *,
    observed_git_state: Mapping[str, Any] | None = None,
    trusted_recovery_context: Any | None = None,
) -> dict[str, Any]:
    """Map the current runner batch without dropping failed evidence."""

    _plain_plan(plan)
    authority = _trusted_recovery_launch_record(trusted_recovery_context)
    root = Path(output_root).resolve()
    controls = _source_controls(root, _CONTROL_FILENAMES)
    mapping_errors: list[str] = []
    try:
        from .stage_c_runner import validate_stage_c_batch

        validation = validate_stage_c_batch(
            root, plan, observed_git_state=observed_git_state,
            trusted_recovery_context=authority,
        ).to_dict()
        validation_exception = None
    except Exception as error:
        validation_exception = (
            f"{type(error).__module__}.{type(error).__qualname__}: {error}"
        )
        validation = {
            "schema_version": 1,
            "document_type": "stage_c_batch_validation",
            "plan_digest": plan.canonical_digest, "valid": False,
            "orchestration_complete": False, "successful": False,
            "rows": [], "errors": [validation_exception],
        }
    failure_by_digest, table_errors = _failure_rows_by_digest(
        controls["failure_table.json"]
    )
    mapping_errors.extend(table_errors)
    batch_status = controls["batch_status.json"].get("document")
    if not isinstance(batch_status, Mapping):
        mapping_errors.append("batch status is unavailable")
    elif (
        batch_status.get("failure_table_sha256")
        != controls["failure_table.json"].get("sha256")
    ):
        mapping_errors.append("batch status failure-table hash is invalid")
    rows = validation.get("rows")
    if not isinstance(rows, list):
        rows = []
        mapping_errors.append("batch validation rows are unavailable")
    plan_by_digest = {unit.key.canonical_digest: unit for unit in plan.units}
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            mapping_errors.append("batch validation contains a non-object row")
            continue
        identity = row.get("unit_key")
        identity = identity if isinstance(identity, Mapping) else {}
        digest = identity.get("canonical_digest")
        state = row.get("state")
        failure_row = failure_by_digest.get(digest)
        status_hash = (
            failure_row.get("selected_status_sha256")
            if isinstance(failure_row, Mapping) else None
        )
        attempt = _safe_attempt(root, row.get("selected_attempt"))
        if attempt is None:
            errors = _json_value(row.get("errors", []), "validation errors")
            if state == "succeeded":
                errors.append("successful row has no safe selected attempt")
            records.append({
                "unit_key": _json_value(identity, "unit_key"),
                "state": state, "trusted_by_batch_validator": False,
                "attempt_path": row.get("selected_attempt"),
                "hashes": {"selected_status_sha256": status_hash},
                "runner_telemetry": None, "errors": errors,
                "attempts": _json_value(row.get("attempts", []), "attempts"),
                "validation_errors": _json_value(row.get("errors", []), "errors"),
            })
            mapping_errors.extend(f"{digest}: {item}" for item in errors)
            continue
        evidence, evidence_errors = _read_attempt_evidence(
            attempt, identity, trusted=bool(validation.get("valid")),
            state=str(state), selected_status_sha256=status_hash,
        )
        unit = plan_by_digest.get(digest)
        if unit is None:
            evidence_errors.append("validation row is absent from execution plan")
        else:
            evidence["work_item_counts"] = unit.work_item_counts
            evidence["audit"] = unit.audit
        evidence["attempts"] = _json_value(row.get("attempts", []), "attempts")
        evidence["validation_errors"] = _json_value(row.get("errors", []), "errors")
        evidence["errors"] = evidence_errors
        evidence["trusted_by_batch_validator"] = bool(
            validation.get("valid") and not evidence_errors
        )
        records.append(evidence)
        mapping_errors.extend(f"{digest}: {item}" for item in evidence_errors)
    return {
        "source_root": str(root),
        "batch_validation": validation,
        "validation_exception": validation_exception,
        "controls": controls,
        "records": records,
        "mapping_errors": list(dict.fromkeys(mapping_errors)),
    }


_DOWNSTREAM_PROFILES = {
    "C8": {
        "images": 5, "value_candidates_per_image": 144,
        "value_methods": 3, "class_methods": 4, "audit_samples": 2,
    },
    "C9": {
        "images": 20, "value_candidates_per_image": 4,
        "value_methods": 3, "class_methods": 4, "audit_samples": 2,
    },
    "StageD": {
        "images": 500, "value_candidates_per_image": 4,
        "value_methods": 3, "class_methods": 4, "audit_samples": 2,
    },
}


def _downstream_profile(name: str) -> dict[str, Any]:
    raw = _DOWNSTREAM_PROFILES[name]
    value_units = raw["images"] * raw["value_candidates_per_image"]
    class_units = raw["images"]
    per_method = {
        **{method: value_units for method in VALUE_METHOD_IDS},
        **{method: class_units for method in CLASS_METHOD_IDS},
    }
    audit_value = raw["audit_samples"] * raw["value_candidates_per_image"]
    audit_class = raw["audit_samples"]
    audit_per_method = {
        **{method: audit_value for method in VALUE_METHOD_IDS},
        **{method: audit_class for method in CLASS_METHOD_IDS},
    }
    total = sum(per_method.values())
    return {
        **raw,
        "formula": (
            f"{raw['images']}*({raw['value_candidates_per_image']}*"
            f"{raw['value_methods']}+{raw['class_methods']})"
        ),
        "execution_units": total,
        "per_method_execution_units": per_method,
        "audit_execution_units": sum(audit_per_method.values()),
        "audit_execution_units_per_sample": (
            raw["value_candidates_per_image"] * raw["value_methods"]
            + raw["class_methods"]
        ),
        "per_method_audit_execution_units": audit_per_method,
    }


def _method_from_record(record: Mapping[str, Any]) -> str | None:
    identity = record.get("unit_key")
    method = identity.get("method_id") if isinstance(identity, Mapping) else None
    return method if method in _ALL_METHODS else None


def _ceil_factor(value: int, factor: float) -> int:
    return int(
        (Decimal(value) * Decimal(str(factor))).to_integral_value(
            rounding=ROUND_CEILING
        )
    )


def _nonnegative_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 0:
        raise StageCBenchmarkIntegrityError(f"{context} must be a non-negative integer")
    return value


def _modeled_storage_profiles(
    records: Sequence[Mapping[str, Any]], storage_model: Mapping[str, Any] | None,
    *, c6_storage_payload_bytes: int, source_unit_count: int,
) -> tuple[dict[str, int], dict[str, int], int, dict[str, Any]]:
    """Derive exact C6 model operands, with a conservative blocked fallback."""

    if not isinstance(storage_model, Mapping):
        per_unit = math.ceil(c6_storage_payload_bytes / source_unit_count)
        return (
            {method: per_unit for method in _ALL_METHODS},
            {method: 0 for method in _ALL_METHODS},
            0,
            {
                "mode": "aggregate_fallback",
                "payload_bytes_per_unit_ceiling": per_unit,
            },
        )
    estimate_model = storage_model.get("estimate_model")
    l0 = storage_model.get("L0")
    l1 = storage_model.get("L1")
    l2 = storage_model.get("L2")
    l3 = storage_model.get("L3")
    control = storage_model.get("control")
    if not all(isinstance(item, Mapping) for item in (
        estimate_model, l0, l1, l2, l3, control,
    )):
        raise StageCBenchmarkIntegrityError("C6 storage model sections are malformed")
    required_operands = (
        "l0_base_bytes_per_unit", "l0_bytes_per_work_item",
        "l1_float32_patch_map_bytes",
        "l2_shared_attention_bytes_per_audit_sample",
    )
    operands = {
        key: _nonnegative_int(estimate_model.get(key), f"estimate_model.{key}")
        for key in required_operands
    }
    l1_total = _nonnegative_int(l1.get("estimated_bytes"), "storage.L1.estimated_bytes")
    l2_total = _nonnegative_int(l2.get("estimated_bytes"), "storage.L2.estimated_bytes")
    l3_total = _nonnegative_int(l3.get("estimated_bytes"), "storage.L3.estimated_bytes")
    control_total = _nonnegative_int(
        control.get("estimated_bytes"), "storage.control.estimated_bytes"
    )
    control_files = _nonnegative_int(control.get("file_count"), "storage.control.file_count")
    manifest_files = _nonnegative_int(
        control.get("unit_manifest_files"), "storage.control.unit_manifest_files"
    )
    status_files = _nonnegative_int(
        control.get("unit_status_files"), "storage.control.unit_status_files"
    )
    batch_files = _nonnegative_int(
        control.get("batch_control_files"), "storage.control.batch_control_files"
    )
    if control_files == 0:
        raise StageCBenchmarkIntegrityError("C6 control file count cannot be zero")

    # A storage model is an aggregate over the plan that produced it.  In
    # particular, a C7 aggregate must never be decomposed with C8 record
    # counts: doing so compares the 40-unit C7 L1 total with thousands of C8
    # maps.  Manifest/status counts and the L1 file count are the C6-bound
    # unit/map operands.  Older injectable test models did not retain the L1
    # file count, so keep their former same-scale interpretation as a narrow
    # compatibility path.
    c6_map_count = l1.get("file_count")
    has_c6_scale = type(c6_map_count) is int
    if has_c6_scale:
        c6_map_count = _nonnegative_int(c6_map_count, "storage.L1.file_count")
        if manifest_files == 0 or manifest_files != status_files:
            raise StageCBenchmarkIntegrityError(
                "C6 unit manifest/status counts must be equal and non-zero"
            )
        c6_unit_count = manifest_files
        if (
            c6_unit_count != C7_EXPECTED_COUNTS["execution_units"]
            or c6_map_count != C7_EXPECTED_COUNTS["map_runs"]
        ):
            raise StageCBenchmarkIntegrityError(
                "C6 storage model unit/map counts differ from the frozen C7 plan"
            )
    else:
        c6_unit_count = source_unit_count
        c6_map_count = sum(
            int(record.get("work_item_counts", {}).get("map_runs", 0))
            for record in records
        )
    exact_l1_payload = c6_map_count * operands["l1_float32_patch_map_bytes"]
    if l1_total < exact_l1_payload:
        raise StageCBenchmarkIntegrityError(
            "C6 L1 estimate is below its exact map payload"
        )
    l1_container_total = l1_total - exact_l1_payload
    if has_c6_scale and l1_container_total % c6_unit_count:
        raise StageCBenchmarkIntegrityError(
            "C6 L1 container bytes do not divide across source units"
        )
    l1_container_per_unit = math.ceil(l1_container_total / c6_unit_count)
    l3_per_unit = math.ceil(l3_total / c6_unit_count)
    unit_control = math.ceil(
        control_total * (manifest_files + status_files)
        / control_files / c6_unit_count
    )
    batch_control = math.ceil(control_total / control_files) * batch_files

    # The C6 model consists of one metadata envelope per audit unit, each
    # method's native clean/perturb/repeat maps, and one shared-attention term.
    # Formal C7 has a frozen 12/12/12/1/1/1/1 method mix and five maps per
    # unit.  Use that source-plan profile when the C6 scale is present.  This
    # also means a failed C8 unit (or even a method with no usable C8 unit)
    # cannot erase that method's conservative downstream model.
    c6_method_counts = {
        **{method: 12 for method in VALUE_METHOD_IDS},
        **{method: 1 for method in CLASS_METHOD_IDS},
    }
    c6_method_work = {
        method: {
            "work_items": (
                5 if method == "direct_raw_cls_attention_per_head" else 929
            ),
            "map_runs": 5,
        }
        for method in _ALL_METHODS
    }
    if has_c6_scale:
        source_profiles = [
            (method, c6_method_work[method]["map_runs"], count)
            for method, count in c6_method_counts.items()
        ]
    else:
        source_profiles = [
            (
                _method_from_record(record),
                int(record.get("work_item_counts", {}).get("map_runs", 0)),
                1,
            )
            for record in records
        ]
    modeled_native_l2 = sum(
        count * maps * (
            (3 * 224 * 224 * 4) if method in _IG_METHODS else (14 * 14 * 4)
        )
        for method, maps, count in source_profiles
    )
    shared_l2 = operands["l2_shared_attention_bytes_per_audit_sample"]
    if l2_total < modeled_native_l2 + shared_l2:
        raise StageCBenchmarkIntegrityError(
            "C6 L2 estimate is below native maps plus shared attention"
        )
    l2_metadata_per_unit = math.ceil(
        (l2_total - modeled_native_l2 - shared_l2) / c6_unit_count
    )
    non_l2: dict[str, int] = {}
    l2_by_method: dict[str, int] = {}
    for method in _ALL_METHODS:
        method_records = [
            record for record in records if _method_from_record(record) == method
        ]
        if has_c6_scale:
            work_profiles = [c6_method_work[method]]
        else:
            work_profiles = [
                record.get("work_item_counts") for record in method_records
            ]
        modeled: list[int] = []
        for counts in work_profiles:
            if not isinstance(counts, Mapping):
                continue
            work = counts.get("work_items")
            maps = counts.get("map_runs")
            if type(work) is not int or type(maps) is not int:
                continue
            modeled.append(
                operands["l0_base_bytes_per_unit"]
                + work * operands["l0_bytes_per_work_item"]
                + l1_container_per_unit
                + maps * operands["l1_float32_patch_map_bytes"]
                + l3_per_unit + unit_control
            )
        if not modeled:
            raise StageCBenchmarkIntegrityError(
                f"C6 model has no source work counts for {method}"
            )
        non_l2[method] = max(modeled)
        maps = max(int(item["map_runs"]) for item in work_profiles)
        native = (3 * 224 * 224 * 4) if method in _IG_METHODS else (14 * 14 * 4)
        l2_by_method[method] = l2_metadata_per_unit + maps * native
    # Shared attention is charged once per audit sample.  Assigning it to the
    # stable rollout profile makes downstream multiplication explicit.
    l2_by_method["attention_rollout_mean_head_residual_all_blocks"] += shared_l2
    model_details = {
        "mode": "exact_c6_operands",
        "source_storage_model": _json_value(storage_model, "C6 storage model"),
        "estimate_model": operands,
        "l1_container_bytes_per_unit_ceiling": l1_container_per_unit,
        "l2_unit_metadata_bytes_per_unit_ceiling": l2_metadata_per_unit,
        "l3_bytes_per_unit_ceiling": l3_per_unit,
        "unit_control_bytes_per_unit_ceiling": unit_control,
        "modeled_batch_control_bytes": batch_control,
    }
    # Keep the historical C7 same-scale projection byte-for-byte reproducible.
    # Cross-scale consumers additionally expose the source scale that was used
    # to decompose the aggregate model.
    if source_unit_count != c6_unit_count:
        model_details.update({
            "source_execution_units": c6_unit_count,
            "source_map_file_count": c6_map_count,
        })
    return non_l2, l2_by_method, batch_control, model_details


def _cap_check(
    key: str, observed: int | float | None, cap: int | float, *,
    unit: str, formula: str, measured: bool,
) -> dict[str, Any]:
    passed = bool(
        measured and observed is not None
        and not isinstance(observed, bool)
        and isinstance(observed, (int, float))
        and math.isfinite(float(observed))
        and observed <= cap
    )
    return {
        "cap_key": key, "observed": observed, "cap": cap,
        "unit": unit, "formula": formula, "measured": bool(measured),
        "comparison": "observed <= cap",
        "status": STATUS_PASS if passed else STATUS_BLOCKED,
    }


def _record_total_wall(record: Mapping[str, Any]) -> float | None:
    telemetry = record.get("runner_telemetry")
    timings = telemetry.get("phase_timings") if isinstance(telemetry, Mapping) else None
    total = (
        timings.get("synchronized_total_wall_time_seconds")
        if isinstance(timings, Mapping) else None
    )
    if (
        isinstance(total, bool) or not isinstance(total, (int, float))
        or not math.isfinite(float(total)) or float(total) < 0.0
    ):
        return None
    return float(total)


def project_benchmark_budget(
    records: Sequence[Mapping[str, Any]], *, caps: Mapping[str, Any],
    c6_storage_payload_bytes: int | None = None,
    source_batch_regular_file_bytes: int | None = None,
    source_unit_count: int | None = None,
    c6_storage_model: Mapping[str, Any] | None = None,
    source_batch_control_bytes: int = 0,
) -> dict[str, Any]:
    """Project C8/C9/Stage D and evaluate all ten frozen hard caps.

    Missing actual time, call, CUDA, or byte evidence is represented as an
    unmeasured ``BLOCKED`` cap check; it is never replaced with zero.  Equality
    passes every cap.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
    if source_unit_count is None:
        source_unit_count = len(records)
    if type(source_unit_count) is not int or source_unit_count <= 0:
        raise StageCBenchmarkIntegrityError("source_unit_count must be positive")
    c6_storage_payload_bytes = (
        0 if c6_storage_payload_bytes is None else c6_storage_payload_bytes
    )
    source_batch_regular_file_bytes = (
        0 if source_batch_regular_file_bytes is None
        else source_batch_regular_file_bytes
    )
    _nonnegative_int(c6_storage_payload_bytes, "c6_storage_payload_bytes")
    _nonnegative_int(
        source_batch_regular_file_bytes, "source_batch_regular_file_bytes"
    )
    _nonnegative_int(source_batch_control_bytes, "source_batch_control_bytes")
    missing_caps = [key for key in HARD_CAP_KEYS if key not in caps]
    if missing_caps:
        raise StageCBenchmarkIntegrityError(f"missing hard caps: {missing_caps}")
    if (
        caps.get("projection_safety_factor") != 1.25
        or caps.get("storage_control_overhead_factor") != 1.25
    ):
        raise StageCBenchmarkIntegrityError(
            "projection and storage overhead factors must equal 1.25"
        )
    for key in HARD_CAP_KEYS:
        value = caps[key]
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value <= 0
        ):
            raise StageCBenchmarkIntegrityError(f"cap {key} is invalid")
    factor = float(caps["projection_safety_factor"])
    overhead = float(caps["storage_control_overhead_factor"])
    usable = [
        record for record in records
        if record.get("trusted_by_batch_validator") is True
        and record.get("state") == "succeeded"
    ]
    walls: list[tuple[float, str, dict[str, Any], str | None]] = []
    forward: list[int] = []
    backward: list[int] = []
    reserved: list[int] = []
    non_l2: list[int] = []
    totals: list[int] = []
    l2_by_sample: Counter[str] = Counter()
    measured_non_l2: dict[str, list[int]] = defaultdict(list)
    measured_l2: dict[str, list[int]] = defaultdict(list)
    telemetry_valid: list[bool] = []
    for record in usable:
        telemetry_valid.append(not _telemetry_errors(record.get("runner_telemetry")))
        wall = _record_total_wall(record)
        identity = record.get("unit_key")
        identity = identity if isinstance(identity, Mapping) else {}
        digest = str(identity.get("canonical_digest", ""))
        if wall is not None:
            walls.append((wall, digest, _json_value(identity, "unit key"), record.get("attempt_path")))
        resources = record.get("resources")
        resources = resources if isinstance(resources, Mapping) else {}
        calls = resources.get("calls")
        calls = calls if isinstance(calls, Mapping) else {}
        fwd, bwd = calls.get("forward_calls"), calls.get("backward_calls")
        if (
            calls.get("accounting_unit") == "batched_model_invocation"
            and type(fwd) is int and fwd >= 0 and type(bwd) is int and bwd >= 0
        ):
            forward.append(fwd)
            backward.append(bwd)
        cuda = resources.get("cuda")
        cuda = cuda if isinstance(cuda, Mapping) else {}
        peak = cuda.get("peak_reserved_bytes")
        if cuda.get("status") == "measured" and type(peak) is int and peak >= 0:
            reserved.append(peak)
        cap_bytes = record.get("l0_l1_cap_accounting_bytes")
        total_bytes = record.get("total_regular_file_bytes")
        levels = record.get("level_bytes")
        levels = levels if isinstance(levels, Mapping) else {}
        method = _method_from_record(record)
        if type(cap_bytes) is int and cap_bytes >= 0:
            non_l2.append(cap_bytes)
            if method is not None:
                measured_non_l2[method].append(cap_bytes)
        if type(total_bytes) is int and total_bytes >= 0:
            totals.append(total_bytes)
        l2_bytes = levels.get("L2")
        if type(l2_bytes) is int and l2_bytes >= 0:
            if method is not None:
                measured_l2[method].append(l2_bytes)
            sample = identity.get("sample_id")
            if isinstance(sample, str):
                l2_by_sample[sample] += l2_bytes
    timing_measured = (
        len(usable) == source_unit_count == len(walls)
        and len(telemetry_valid) == source_unit_count and all(telemetry_valid)
    )
    calls_measured = len(usable) == source_unit_count == len(forward) == len(backward)
    cuda_measured = len(usable) == source_unit_count == len(reserved)
    bytes_measured = len(usable) == source_unit_count == len(non_l2) == len(totals)
    slowest_entry = max(walls, key=lambda item: (item[0], item[1])) if walls else None
    slowest = slowest_entry[0] if slowest_entry else None
    modeled_non_l2, modeled_l2, modeled_batch, model_details = (
        _modeled_storage_profiles(
            records, c6_storage_model,
            c6_storage_payload_bytes=c6_storage_payload_bytes,
            source_unit_count=source_unit_count,
        )
    )
    method_profiles: dict[str, Any] = {}
    for method in _ALL_METHODS:
        measured_non = max(measured_non_l2[method], default=0)
        measured_audit = max(measured_l2[method], default=0)
        method_profiles[method] = {
            "c6_modeled_non_l2_bytes_per_unit": modeled_non_l2[method],
            "measured_non_l2_bytes_per_unit_max": measured_non,
            "conservative_non_l2_bytes_per_unit": max(
                modeled_non_l2[method], measured_non
            ),
            "c6_modeled_l2_bytes_per_audit_unit": modeled_l2[method],
            "measured_l2_bytes_per_audit_unit_max": measured_audit,
            "conservative_l2_bytes_per_audit_unit": max(
                modeled_l2[method], measured_audit
            ),
        }
    measured_batch_per_unit = math.ceil(source_batch_control_bytes / source_unit_count)
    conservative_batch_per_unit = max(
        measured_batch_per_unit, math.ceil(modeled_batch / source_unit_count)
    )
    profiles = {name: _downstream_profile(name) for name in _DOWNSTREAM_PROFILES}
    components: dict[str, Any] = {}
    raw_storage: dict[str, int] = {}
    for name, profile in profiles.items():
        non_l2_bytes = sum(
            method_profiles[method]["conservative_non_l2_bytes_per_unit"] * count
            for method, count in profile["per_method_execution_units"].items()
        )
        l2_bytes = sum(
            method_profiles[method]["conservative_l2_bytes_per_audit_unit"] * count
            for method, count in profile["per_method_audit_execution_units"].items()
        )
        batch_bytes = conservative_batch_per_unit * profile["execution_units"]
        raw = non_l2_bytes + l2_bytes + batch_bytes
        components[name] = {
            "non_l2_unit_bytes": non_l2_bytes, "l2_audit_bytes": l2_bytes,
            "batch_control_bytes": batch_bytes, "before_overhead_bytes": raw,
        }
        raw_storage[name] = raw
    components["C8_C9"] = {
        key: components["C8"][key] + components["C9"][key]
        for key in components["C8"]
    }
    raw_storage["C8_C9"] = raw_storage["C8"] + raw_storage["C9"]
    projected_storage = {
        name: _ceil_factor(raw, overhead) for name, raw in raw_storage.items()
    }

    def hours(count: int) -> float | None:
        return None if slowest is None else slowest * count * factor / 3600.0

    projected_time = {
        "C8_h20_gpu_hours": hours(C8_EXECUTION_UNITS),
        "C9_h20_gpu_hours": hours(C9_EXECUTION_UNITS),
        "C8_C9_h20_gpu_hours": hours(C8_C9_EXECUTION_UNITS),
        "StageD_h20_gpu_hours": hours(STAGE_D_EXECUTION_UNITS),
        "StageD_h20_gpu_hours_per_image": (
            None if slowest is None else
            slowest * STAGE_D_EXECUTION_UNITS_PER_IMAGE * factor / 3600.0
        ),
    }
    projected_c8_l2_per_audit = components["C8"]["l2_audit_bytes"] // 2
    checks = [
        _cap_check(HARD_CAP_KEYS[0], max(non_l2, default=None), caps[HARD_CAP_KEYS[0]], unit="bytes", formula="max finalized unit L0+L1+L3+manifest+status", measured=bytes_measured),
        _cap_check(HARD_CAP_KEYS[1], projected_c8_l2_per_audit, caps[HARD_CAP_KEYS[1]], unit="bytes", formula="method-profile C6/measured maximum for 436 units in one C8 audit sample", measured=bytes_measured and bool(measured_l2)),
        _cap_check(HARD_CAP_KEYS[2], projected_storage["C8_C9"], caps[HARD_CAP_KEYS[2]], unit="bytes", formula="C8+C9 exact method profiles and audit-only L2, then 1.25 once", measured=bytes_measured),
        _cap_check(HARD_CAP_KEYS[3], projected_storage["StageD"], caps[HARD_CAP_KEYS[3]], unit="bytes", formula="StageD exact method profiles and two-audit-only L2, then 1.25 once", measured=bytes_measured),
        _cap_check(HARD_CAP_KEYS[4], max(forward, default=None), caps[HARD_CAP_KEYS[4]], unit="batched_model_invocations", formula="maximum actual forward calls per unit", measured=calls_measured),
        _cap_check(HARD_CAP_KEYS[5], max(backward, default=None), caps[HARD_CAP_KEYS[5]], unit="batched_model_invocations", formula="maximum actual backward calls per unit", measured=calls_measured),
        _cap_check(HARD_CAP_KEYS[6], max(reserved, default=None), caps[HARD_CAP_KEYS[6]], unit="bytes", formula="maximum measured CUDA peak reserved per unit", measured=cuda_measured),
        _cap_check(HARD_CAP_KEYS[7], projected_time["C8_C9_h20_gpu_hours"], caps[HARD_CAP_KEYS[7]], unit="H20-GPU-hours", formula="slowest synchronized unit * 2500 * 1.25 / 3600", measured=timing_measured),
        _cap_check(HARD_CAP_KEYS[8], projected_time["StageD_h20_gpu_hours"], caps[HARD_CAP_KEYS[8]], unit="H20-GPU-hours", formula="slowest synchronized unit * 8000 * 1.25 / 3600", measured=timing_measured),
        _cap_check(HARD_CAP_KEYS[9], projected_time["StageD_h20_gpu_hours_per_image"], caps[HARD_CAP_KEYS[9]], unit="H20-GPU-hours/image", formula="slowest synchronized unit * 16 * 1.25 / 3600", measured=timing_measured),
    ]
    return {
        "exact_unit_counts": {
            "C8": C8_EXECUTION_UNITS, "C9": C9_EXECUTION_UNITS,
            "C8_C9": C8_C9_EXECUTION_UNITS,
            "StageD": STAGE_D_EXECUTION_UNITS,
            "StageD_images": STAGE_D_IMAGES,
        },
        "downstream_profiles": profiles,
        "time": {
            "slowest_valid_measured_unit_seconds": slowest,
            "slowest_valid_measured_unit": None if slowest_entry is None else {
                "unit_key": slowest_entry[2],
                "unit_key_sha256": slowest_entry[1],
                "attempt_path": slowest_entry[3],
                "wall_time_seconds": slowest_entry[0],
            },
            "measurement_source": (
                "status.runner_telemetry.phase_timings."
                "synchronized_total_wall_time_seconds"
            ),
            "projection_safety_factor": factor,
            **projected_time,
        },
        "storage": {
            "c6_exact_model_payload_bytes": c6_storage_payload_bytes,
            "c6_model_operands": model_details,
            "source_batch_regular_file_bytes": source_batch_regular_file_bytes,
            "source_batch_control_bytes": source_batch_control_bytes,
            "method_profiles": method_profiles,
            "measured_batch_control_bytes_per_source_unit_ceiling": measured_batch_per_unit,
            "conservative_batch_control_bytes_per_unit": conservative_batch_per_unit,
            "components": components,
            "control_provenance_overhead_factor": overhead,
            "projected_bytes": projected_storage,
        },
        "measured_maxima": {
            "l0_l1_cap_accounting_bytes_per_unit": max(non_l2, default=None),
            "l2_bytes_per_audit_sample": max(l2_by_sample.values(), default=None),
            "c8_projected_l2_bytes_per_audit_sample": projected_c8_l2_per_audit,
            "forward_calls_per_unit": max(forward, default=None),
            "backward_calls_per_unit": max(backward, default=None),
            "cuda_peak_reserved_bytes_per_unit": max(reserved, default=None),
        },
        "measurement_completeness": {
            "expected_units": source_unit_count,
            "usable_units": len(usable),
            "timing_measured": timing_measured,
            "calls_measured": calls_measured,
            "cuda_measured": cuda_measured,
            "bytes_measured": bytes_measured,
        },
        "hard_cap_checks": checks,
        "all_hard_caps_pass": all(
            item["status"] == STATUS_PASS for item in checks
        ),
    }


def require_fresh_c7_output_root(
    output_root: Path | str, *, kind: str,
) -> Path:
    """Require a missing or existing-empty C7 output root.

    Freshness failure is a valid preflight block, not source corruption.  The
    helper is deliberately non-mutating so the runner remains the sole owner
    of output-root creation.
    """

    if kind not in {"diagnostic", "formal"}:
        raise ValueError("kind must be 'diagnostic' or 'formal'")
    root = Path(output_root)
    try:
        resolved = root.resolve()
    except OSError as error:
        raise StageCBenchmarkBlockedError(
            f"cannot resolve {kind} output root: {error}",
            reasons=[f"{kind} output root cannot be resolved"],
        ) from error
    reasons: list[str] = []
    if root.is_symlink():
        reasons.append(f"{kind} output root cannot be a symlink")
    if _polluted_path(resolved):
        reasons.append(
            f"{kind} output root contains an archive/old-attempt path marker"
        )
    if resolved.exists():
        if not resolved.is_dir():
            reasons.append(f"{kind} output root exists and is not a directory")
        else:
            try:
                if any(resolved.iterdir()):
                    reasons.append(f"{kind} output root is not fresh and empty")
            except OSError as error:
                reasons.append(f"{kind} output root cannot be inspected: {error}")
    if reasons:
        report = {
            "schema_version": 1, "document_type": LAUNCH_DOCUMENT_TYPE,
            "kind": kind, "status": STATUS_BLOCKED,
            "integrity_valid": True, "blocking_kind": "freshness",
            "exit_class": EXIT_CLASS_PREFLIGHT,
            "blocked_reasons": reasons, "output_root": str(resolved),
        }
        raise StageCBenchmarkBlockedError(
            "; ".join(reasons), report=report, reasons=reasons
        )
    return resolved


def _measure_source_tree(
    root: Path,
) -> tuple[int, int, list[dict[str, Any]], list[str]]:
    total = 0
    control_bytes = 0
    files: list[dict[str, Any]] = []
    errors: list[str] = []
    if root.is_symlink() or not root.is_dir():
        return 0, 0, [], ["source root is missing, non-directory, or symlink"]
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            errors.append(f"source tree contains a symlink: {relative}")
            continue
        if path.is_file():
            size = path.stat().st_size
            total += size
            files.append({
                "path": relative, "bytes": size, "sha256": _sha256_file(path),
            })
            if relative in _FORMAL_CONTROL_FILENAMES:
                control_bytes += size
    return total, control_bytes, files, errors


def _source_layout_errors(
    root: Path, plan: ExecutionPlan, records: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    _require_unpolluted_path(root, "source output root")
    expected_top = {*_CONTROL_FILENAMES, "units"}
    if len(plan.units) == C7_EXPECTED_COUNTS["execution_units"]:
        expected_top.add(LAUNCH_AUTHORIZATION_FILENAME)
    try:
        actual_top = {path.name for path in root.iterdir()}
    except OSError as error:
        return [f"cannot inspect source root: {error}"]
    if actual_top != expected_top:
        errors.append(
            "source root top-level inventory mismatch: "
            f"missing={sorted(expected_top - actual_top)}, "
            f"extra={sorted(actual_top - expected_top)}"
        )
    units_root = root / "units"
    expected_units = {unit.key.canonical_digest for unit in plan.units}
    if units_root.is_symlink() or not units_root.is_dir():
        errors.append("source units root is missing or unsafe")
        return errors
    actual_units = {path.name for path in units_root.iterdir() if path.is_dir()}
    non_directories = [path.name for path in units_root.iterdir() if not path.is_dir()]
    if actual_units != expected_units or non_directories:
        errors.append(
            "source unit directory inventory differs from plan: "
            f"missing={sorted(expected_units - actual_units)}, "
            f"extra={sorted(actual_units - expected_units)}, "
            f"non_directories={sorted(non_directories)}"
        )
    record_by_digest = {
        record.get("unit_key", {}).get("canonical_digest"): record
        for record in records if isinstance(record.get("unit_key"), Mapping)
    }
    for digest in sorted(expected_units):
        unit_root = units_root / digest
        expected_unit_children = {"attempts"}
        if unit_root.is_symlink() or not unit_root.is_dir():
            errors.append(f"unit root {digest} is missing or unsafe")
            continue
        actual_children = {path.name for path in unit_root.iterdir()}
        if actual_children != expected_unit_children:
            errors.append(f"unit root {digest} has unexpected entries")
        attempts = unit_root / "attempts"
        if attempts.is_symlink() or not attempts.is_dir():
            errors.append(f"unit {digest} attempts directory is missing or unsafe")
            continue
        attempt_dirs = [
            path for path in attempts.iterdir() if path.is_dir() and not path.is_symlink()
        ]
        unsafe = [path for path in attempts.iterdir() if path not in attempt_dirs]
        if len(attempt_dirs) != 1 or unsafe:
            errors.append(
                f"unit {digest} must contain exactly one safe attempt directory"
            )
            continue
        record = record_by_digest.get(digest)
        selected = record.get("attempt_path") if isinstance(record, Mapping) else None
        if selected != str(attempt_dirs[0].resolve()):
            errors.append(f"unit {digest} selected attempt differs from its only attempt")
    return errors


def _run_spec_binding(
    plan: ExecutionPlan, run_spec_path: Path | str | None,
    run_spec_file_sha256: str | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    semantic = _canonical_digest(plan.run_spec.to_dict())
    if run_spec_path is None:
        return {
            "path": None, "file_sha256": run_spec_file_sha256,
            "semantic_sha256": semantic,
        }, ["run-spec path is required"]
    path = Path(run_spec_path).resolve()
    _require_unpolluted_path(path, "run-spec path")
    if path.is_symlink() or not path.is_file():
        errors.append("run-spec file is missing or unsafe")
        observed = None
    else:
        observed = _sha256_file(path)
        if run_spec_file_sha256 != observed:
            errors.append("run-spec file SHA-256 binding is invalid")
        try:
            value = _strict_json(path)
            if value != plan.run_spec.to_dict():
                errors.append("run-spec file content differs from the execution plan")
        except Exception as error:
            errors.append(f"run-spec content is invalid: {error}")
    return {
        "path": str(path), "file_sha256": run_spec_file_sha256,
        "observed_file_sha256": observed, "semantic_sha256": semantic,
    }, errors


def _git_binding_errors(
    observed_git_state: Any, shared: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(observed_git_state, Mapping) or set(observed_git_state) != {
        "commit", "dirty"
    }:
        return ["observed Git state has the wrong shape"]
    commit = observed_git_state.get("commit")
    dirty = observed_git_state.get("dirty")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        errors.append("observed Git commit is invalid")
    elif commit == ABANDONED_C7_DIAGNOSTIC_COMMIT:
        errors.append(
            "Git binding uses the abandoned C7 diagnostic commit and is forbidden"
        )
    if type(dirty) is not bool:
        errors.append("observed Git dirty flag is invalid")
    elif dirty:
        errors.append("C7 requires a clean current Git worktree")
    if shared is not None and shared.get("code") != dict(observed_git_state):
        errors.append("shared provenance code binding differs from observed Git state")
    return errors


def _shared_provenance(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    values: list[dict[str, Any]] = []
    for record in records:
        value = record.get("shared_provenance")
        if not isinstance(value, Mapping) or set(value) != set(SHARED_PROVENANCE_KEYS):
            errors.append("a unit lacks the exact shared provenance binding set")
            continue
        values.append(dict(value))
    if not values:
        return {}, errors or ["no shared provenance evidence was mapped"]
    first = values[0]
    if any(value != first for value in values[1:]):
        errors.append("unit shared provenance bindings drift within the batch")
    return first, errors


def _trusted_recovery_launch_record(value: Any) -> Any | None:
    """Extract the validated launch authority; never trust projected JSON."""

    if value is None:
        return None
    from .recovery_c7_launch import RecoveryC7LaunchRecord
    from .recovery_v2 import RecoveryV2Binding

    if isinstance(value, RecoveryC7LaunchRecord):
        record = value
        binding = record.binding
    elif isinstance(value, Mapping) and set(value) == {
        "recovery_binding", "recovery_launch_record",
    }:
        binding = value["recovery_binding"]
        record = value["recovery_launch_record"]
    else:
        raise StageCBenchmarkIntegrityError(
            "trusted recovery context must be a validated "
            "RecoveryC7LaunchRecord or its exact validated binding pair"
        )
    if not isinstance(record, RecoveryC7LaunchRecord):
        raise StageCBenchmarkIntegrityError(
            "trusted recovery context lacks a validated "
            "RecoveryC7LaunchRecord"
        )
    if not isinstance(binding, RecoveryV2Binding) or record.binding is not binding:
        raise StageCBenchmarkIntegrityError(
            "trusted recovery context is not the binding owned by its "
            "RecoveryC7LaunchRecord"
        )
    return record


def _recovery_gate_path(record: Any) -> Path:
    return Path(record.reports_root) / DIAGNOSTIC_GATE_FILENAME


def _recovery_report_path(record: Any) -> Path:
    return Path(record.reports_root) / REPORT_FILENAME


def _stage_c_model_binding_errors(
    model: Any, *, trusted_recovery_context: Any | None,
) -> list[str]:
    """Validate a historical or context-bound recovery model block."""

    try:
        from .stage_c_provenance import validate_stage_c_model_binding

        authority = _trusted_recovery_launch_record(trusted_recovery_context)

        validate_stage_c_model_binding(
            model,
            trusted_recovery_context=authority,
            # The provenance helper applies this requirement only to the
            # recovery extension shape.  Historical three-field bindings keep
            # their frozen name/hash contract without needing a context.
            require_trusted_recovery_context=True,
        )
    except Exception as error:
        return [f"shared provenance model binding is invalid: {error}"]
    return []


def _require_stage_c_model_binding(
    model: Any, *, trusted_recovery_context: Any | None, context: str,
) -> None:
    errors = _stage_c_model_binding_errors(
        model, trusted_recovery_context=trusted_recovery_context,
    )
    if errors:
        raise StageCBenchmarkIntegrityError(f"{context}: {errors[0]}")


def _shared_model_from_report(value: Any) -> Any:
    bindings = value.get("bindings") if isinstance(value, Mapping) else None
    shared = (
        bindings.get("shared_provenance")
        if isinstance(bindings, Mapping) else None
    )
    return shared.get("model") if isinstance(shared, Mapping) else None


def _frozen_caps(config: Any) -> tuple[dict[str, Any], list[str]]:
    config_value = _config_dict(config)
    budget = config_value.get("budget")
    if not isinstance(budget, Mapping):
        return {}, ["config budget is missing"]
    result = _json_value(budget, "config budget")
    errors = [
        f"config changed frozen cap/factor {key}"
        for key, expected in FROZEN_CAPS.items() if result.get(key) != expected
    ]
    return result, errors


def _estimate_binding(
    estimate: StageCEstimate | Mapping[str, Any] | None, plan: ExecutionPlan,
    preflight: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    if estimate is None:
        return None, ["freshly recomputed C7 estimate is required"]
    try:
        document = (
            estimate.to_dict() if isinstance(estimate, StageCEstimate)
            else StageCEstimate.from_dict(estimate).to_dict()
        )
    except Exception as error:
        return None, [f"recomputed estimate is invalid: {error}"]
    errors: list[str] = []
    if document.get("plan_digest") != plan.canonical_digest:
        errors.append("recomputed estimate plan digest differs from the plan")
    if not isinstance(preflight, Mapping) or document != dict(preflight):
        errors.append("recomputed estimate differs from source preflight.json")
    return document, errors


def _ig_errors(record: Mapping[str, Any]) -> list[str]:
    method = _method_from_record(record)
    maps = record.get("map_execution_records")
    if not isinstance(maps, list):
        return ["map_execution_records is missing"]
    expected_variants = [
        ("clean", 0),
        ("translate_right_down_4", None),
        ("center_crop_216_resize_224", None),
        ("gaussian_rgb_std_0_01", None),
        ("repeat_clean", 1),
    ]
    actual_variants = [
        (item.get("input_variant"), item.get("repeat_index"))
        for item in maps if isinstance(item, Mapping)
    ]
    errors: list[str] = []
    if actual_variants != expected_variants:
        errors.append("map execution coverage/order differs from five frozen maps")
    if method not in _IG_METHODS:
        if any(isinstance(item, Mapping) and item.get("ig_result") is not None for item in maps):
            errors.append("non-IG method claims IG attempt evidence")
        return errors
    for index, item in enumerate(maps):
        if not isinstance(item, Mapping):
            errors.append(f"map record {index} is not an object")
            continue
        ig = item.get("ig_result")
        if not isinstance(ig, Mapping):
            errors.append(f"map record {index} lacks IG result provenance")
            continue
        attempts = ig.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            errors.append(f"map record {index} IG attempts are missing")
            continue
        steps: list[int] = []
        passes: list[bool] = []
        for attempt_index, attempt in enumerate(attempts):
            if not isinstance(attempt, Mapping):
                errors.append(f"map {index} IG attempt {attempt_index} is malformed")
                continue
            step = attempt.get("steps")
            passed = attempt.get("pass")
            if type(step) is not int:
                errors.append(f"map {index} IG attempt steps are invalid")
            else:
                steps.append(step)
            if not isinstance(passed, list) or not passed or any(type(v) is not bool for v in passed):
                errors.append(f"map {index} IG attempt pass evidence is invalid")
            else:
                passes.append(all(passed))
        if steps != list(IG_STEPS_SCHEDULE[:len(steps)]):
            errors.append(f"map {index} IG attempts are not the frozen schedule prefix")
        if not steps or ig.get("selected_steps") != steps[-1]:
            errors.append(f"map {index} IG selected_steps is not its final attempt")
        if ig.get("converged") is not True:
            errors.append(f"map {index} successful IG did not converge")
        if passes != [False] * (len(passes) - 1) + [True]:
            errors.append(f"map {index} IG did not stop on first conservation pass")
        metadata = ig.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("step_chunk_size") != 8:
            errors.append(f"map {index} IG chunk size is not frozen at 8")
        attempted = sum(steps)
        backward = sum(math.ceil(step / 8) for step in steps)
        if item.get("attempted_steps_total") != attempted:
            errors.append(f"map {index} attempted_steps_total is inconsistent")
        if item.get("backward_invocations") != backward:
            errors.append(f"map {index} backward invocation evidence is inconsistent")
    return errors


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _json_value(document, "report"), indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        ) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _external_report_path(
    report_path: Path | str | None, source_root: Path, errors: list[str],
) -> Path | None:
    if report_path is None:
        errors.append("external report path is required")
        return None
    destination = Path(report_path).resolve()
    _require_unpolluted_path(destination, "report path")
    if _path_is_within(destination, source_root):
        errors.append("report must be external to the measured source root")
    return destination


def _blocked_projection(caps: Mapping[str, Any], reason: str) -> dict[str, Any]:
    checks = [
        _cap_check(
            key, None, caps.get(key, FROZEN_CAPS[key]),
            unit=(
                "bytes" if "bytes" in key else
                "batched_model_invocations" if "calls" in key else
                "H20-GPU-hours/image" if "per_image" in key else
                "H20-GPU-hours"
            ),
            formula=f"unavailable: {reason}", measured=False,
        )
        for key in HARD_CAP_KEYS
    ]
    return {
        "exact_unit_counts": {
            "C8": 2180, "C9": 320, "C8_C9": 2500,
            "StageD": 8000, "StageD_images": 500,
        },
        "downstream_profiles": {
            name: _downstream_profile(name) for name in _DOWNSTREAM_PROFILES
        },
        "time": {"slowest_valid_measured_unit_seconds": None},
        "storage": {}, "measured_maxima": {},
        "measurement_completeness": {"usable_units": 0},
        "hard_cap_checks": checks, "all_hard_caps_pass": False,
    }


def _document_exit_class(
    *, status: str, integrity_valid: bool, blocking_kind: str | None,
) -> str:
    if not integrity_valid:
        return EXIT_CLASS_INTEGRITY
    if status == STATUS_PASS:
        return EXIT_CLASS_SUCCESS
    if blocking_kind in {"performance", "budget", "environment", "preflight", "freshness"}:
        return EXIT_CLASS_PREFLIGHT
    return EXIT_CLASS_TERMINAL


def _report_header(
    *, document_type: str, status: str, integrity_valid: bool,
    blocking_kind: str | None, reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "document_type": document_type,
        "status": status, "integrity_valid": integrity_valid,
        "blocking_kind": blocking_kind,
        "exit_class": _document_exit_class(
            status=status, integrity_valid=integrity_valid,
            blocking_kind=blocking_kind,
        ),
        "blocked_reasons": list(dict.fromkeys(str(item) for item in reasons if str(item))),
    }


def _validate_self_digest(value: Mapping[str, Any], context: str) -> None:
    digest = value.get("canonical_digest")
    if not _is_sha256(digest):
        raise StageCBenchmarkIntegrityError(f"{context} canonical_digest is invalid")
    if _self_digest(value) != digest:
        raise StageCBenchmarkIntegrityError(f"{context} canonical_digest mismatch")


def _unit_identity_errors(identity: Any) -> list[str]:
    if not isinstance(identity, Mapping):
        return ["unit identity is not an object"]
    expected = {
        "schema_version", "sample_id", "method_id", "config_id",
        "block_index", "head_index", "target_ref", "canonical_digest",
    }
    if set(identity) != expected:
        return ["unit identity fields differ from UnitKey contract"]
    try:
        from .stage_c_provenance import unit_identity_digest

        digest = unit_identity_digest(identity)
    except Exception as error:
        return [f"unit identity is malformed: {error}"]
    if digest != identity.get("canonical_digest"):
        return ["unit identity canonical digest is invalid"]
    return []


def _source_controls_errors(
    controls: Any, *, plan_digest: str | None, require_complete: bool,
    source_root: str | None, names: Sequence[str],
) -> list[str]:
    if not isinstance(controls, Mapping) or set(controls) != set(names):
        return ["source control binding set is invalid"]
    errors: list[str] = []
    for name in names:
        item = controls[name]
        if not isinstance(item, Mapping) or set(item) != {
            "path", "present", "bytes", "sha256", "error"
        }:
            errors.append(f"source control {name} evidence shape is invalid")
            continue
        present = item.get("present")
        if type(present) is not bool:
            errors.append(f"source control {name} present flag is invalid")
        elif not present and not require_complete:
            continue
        elif (
            present is not True or item.get("error") is not None
            or type(item.get("bytes")) is not int or item["bytes"] < 0
            or not _is_sha256(item.get("sha256"))
            or not isinstance(item.get("path"), str)
        ):
            errors.append(f"source control {name} is not a valid bound file")
        elif (
            not isinstance(source_root, str)
            or Path(item["path"]).resolve() != (Path(source_root) / name).resolve()
            or _polluted_path(item["path"])
        ):
            errors.append(f"source control {name} path is not canonical")
    return errors


def _control_bytes_sum(
    controls: Any, names: Sequence[str] = _CONTROL_FILENAMES,
) -> int | None:
    if not isinstance(controls, Mapping):
        return None
    sizes: list[int] = []
    for name in names:
        item = controls.get(name)
        if not isinstance(item, Mapping) or type(item.get("bytes")) is not int:
            return None
        sizes.append(item["bytes"])
    return sum(sizes)


def _normalize_source_inventory(
    files: Any, root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not isinstance(files, list):
        return [], ["source regular-file inventory is not an array"]
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(files):
        if isinstance(item, Mapping):
            normalized.append(_json_value(item, f"source file[{index}]"))
            continue
        if not isinstance(item, str):
            errors.append(f"source file[{index}] is malformed")
            continue
        relative = PurePath(item)
        path = (root / item).resolve()
        if (
            relative.is_absolute() or ".." in relative.parts
            or not _path_is_within(path, root) or path.is_symlink()
            or not path.is_file()
        ):
            errors.append(f"source file[{index}] cannot be measured safely")
            continue
        normalized.append({
            "path": item, "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    return normalized, errors


def _common_existing_evidence_errors(
    value: Mapping[str, Any], *, formal: bool, require_complete: bool,
    trusted_recovery_context: Any | None = None,
) -> list[str]:
    """Validate every retained cross-link for PASS and BLOCKED reports."""

    errors: list[str] = []
    scope = value.get("scope")
    bindings = value.get("bindings")
    boundary = value.get("measurement_boundary")
    if not isinstance(scope, Mapping) or not isinstance(bindings, Mapping):
        return ["scope or bindings are missing"]
    plan_digest = bindings.get("execution_plan_sha256")
    if not _is_sha256(plan_digest) or scope.get("execution_plan_sha256") != plan_digest:
        errors.append("scope/execution-plan binding is inconsistent")
    plan_document = bindings.get("execution_plan")
    if not isinstance(plan_document, Mapping):
        if require_complete:
            errors.append("embedded execution plan evidence is missing")
    else:
        try:
            parsed_plan = ExecutionPlan.from_dict(plan_document)
        except Exception as error:
            errors.append(f"embedded execution plan is invalid: {error}")
        else:
            if parsed_plan.canonical_digest != plan_digest:
                errors.append("embedded execution plan digest differs from binding")
            expected_scope_counts = (
                C7_EXPECTED_COUNTS if formal else C7_DIAGNOSTIC_EXPECTED_COUNTS
            )
            if _plan_counts(parsed_plan.to_dict()) != expected_scope_counts:
                errors.append("embedded execution plan counts differ from frozen scope")
    config_digest = bindings.get("config_sha256")
    if not _is_sha256(config_digest) or scope.get("config_sha256") != config_digest:
        errors.append("scope/config binding is inconsistent")
    run_spec = bindings.get("run_spec")
    if not isinstance(run_spec, Mapping) or not _is_sha256(
        run_spec.get("semantic_sha256")
    ):
        errors.append("run-spec binding is incomplete or inconsistent")
    elif require_complete or run_spec.get("file_sha256") is not None:
        if (
            not _is_sha256(run_spec.get("file_sha256"))
            or run_spec.get("observed_file_sha256") != run_spec.get("file_sha256")
        ):
            errors.append("run-spec file binding is incomplete or inconsistent")
    if isinstance(plan_document, Mapping) and isinstance(run_spec, Mapping):
        embedded_spec = plan_document.get("run_spec")
        if (
            not isinstance(embedded_spec, Mapping)
            or _canonical_digest(embedded_spec) != run_spec.get("semantic_sha256")
        ):
            errors.append("embedded plan/run-spec semantic binding is inconsistent")
    shared = bindings.get("shared_provenance")
    shared_digest = bindings.get("shared_provenance_sha256")
    if (not shared) and not require_complete:
        pass
    elif not isinstance(shared, Mapping) or set(shared) != set(SHARED_PROVENANCE_KEYS):
        errors.append("shared provenance binding set is invalid")
    elif _canonical_digest(shared) != shared_digest:
        errors.append("shared provenance digest is invalid")
    else:
        errors.extend(_git_binding_errors(bindings.get("observed_git_state"), shared))
        configuration = shared.get("configuration")
        if not isinstance(configuration, Mapping) or configuration.get("execution_plan_sha256") != plan_digest:
            errors.append("shared provenance plan binding is invalid")
        elif configuration.get("run_spec_sha256") != run_spec.get("semantic_sha256"):
            errors.append("shared provenance run-spec binding is invalid")
        errors.extend(_stage_c_model_binding_errors(
            shared.get("model"),
            trusted_recovery_context=trusted_recovery_context,
        ))
    source_root = bindings.get("source_root")
    if not isinstance(source_root, str) or _polluted_path(source_root):
        errors.append("source root binding is invalid or polluted")
    control_names = _FORMAL_CONTROL_FILENAMES if formal else _CONTROL_FILENAMES
    errors.extend(_source_controls_errors(
        bindings.get("source_controls"), plan_digest=plan_digest,
        require_complete=require_complete, source_root=source_root,
        names=control_names,
    ))
    validation = value.get("batch_validation")
    if isinstance(validation, Mapping) and validation.get("plan_digest") != plan_digest:
        errors.append("batch validation plan digest differs from report binding")
    if not isinstance(boundary, Mapping):
        errors.append("measurement boundary is missing")
    else:
        source_root = boundary.get("source_batch_root")
        if source_root != bindings.get("source_root"):
            errors.append("measurement source root differs from report binding")
        report_path = boundary.get("report_path")
        if require_complete or report_path is not None:
            if (
                not isinstance(source_root, str) or not isinstance(report_path, str)
                or _path_is_within(Path(report_path), Path(source_root))
                or boundary.get("report_external_to_source_batch") is not True
            ):
                errors.append("report is not external to its measured source root")
        total = boundary.get(
            "all_regular_file_bytes" if formal else "source_regular_file_bytes"
        )
        files = boundary.get("source_regular_files")
        if not isinstance(files, list) or any(
            not isinstance(item, Mapping) or set(item) != {"path", "bytes", "sha256"}
            or not isinstance(item.get("path"), str)
            or PurePath(item["path"]).is_absolute()
            or ".." in PurePath(item["path"]).parts
            or _polluted_path(item["path"])
            or type(item.get("bytes")) is not int or item["bytes"] < 0
            or not _is_sha256(item.get("sha256"))
            for item in files
        ):
            errors.append("source regular-file inventory is invalid or polluted")
        elif (
            len({item["path"] for item in files}) != len(files)
            or sum(item["bytes"] for item in files) != total
        ):
            errors.append("source regular-file inventory paths/bytes are inconsistent")
        control = boundary.get("source_batch_control_bytes")
        if type(total) is not int or total < 0 or type(control) is not int or control < 0 or control > total:
            errors.append("source/control byte boundary is invalid")
        observed_control_sum = _control_bytes_sum(
            bindings.get("source_controls"), control_names
        )
        if observed_control_sum is not None and control != observed_control_sum:
            errors.append("source control byte sum differs from measurement boundary")
    return errors


def _base_report_shape_errors(
    value: Mapping[str, Any], document_type: str, expected: set[str],
) -> None:
    if set(value) != expected:
        raise StageCBenchmarkIntegrityError(
            f"report fields differ: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    if value.get("schema_version") != 1 or value.get("document_type") != document_type:
        raise StageCBenchmarkIntegrityError("report identity is invalid")
    status = value.get("status")
    reasons = value.get("blocked_reasons")
    if status not in {STATUS_PASS, STATUS_BLOCKED} or not isinstance(reasons, list):
        raise StageCBenchmarkIntegrityError("report status/reasons are invalid")
    if (status == STATUS_PASS) != (not reasons):
        raise StageCBenchmarkIntegrityError("report status and blocked reasons disagree")
    if any(not isinstance(item, str) or not item for item in reasons) or len(
        reasons
    ) != len(set(reasons)):
        raise StageCBenchmarkIntegrityError(
            "report blocked reasons must be unique non-empty strings"
        )
    if type(value.get("integrity_valid")) is not bool:
        raise StageCBenchmarkIntegrityError("report integrity_valid must be boolean")
    blocking_kind = value.get("blocking_kind")
    if status == STATUS_PASS and blocking_kind is not None:
        raise StageCBenchmarkIntegrityError("PASS report cannot have a blocking_kind")
    if status == STATUS_BLOCKED and blocking_kind not in {
        "integrity", "terminal", "performance", "budget",
        "environment", "preflight", "freshness",
    }:
        raise StageCBenchmarkIntegrityError("BLOCKED report blocking_kind is invalid")
    if value["integrity_valid"] is (blocking_kind == "integrity"):
        raise StageCBenchmarkIntegrityError(
            "integrity_valid and blocking_kind=integrity disagree"
        )
    expected_exit = _document_exit_class(
        status=status, integrity_valid=value["integrity_valid"],
        blocking_kind=blocking_kind,
    )
    if value.get("exit_class") != expected_exit:
        raise StageCBenchmarkIntegrityError("report exit_class is inconsistent")
    _validate_self_digest(value, "report")


def build_c7_diagnostic_gate(
    output_root: Path | str, plan: ExecutionPlan, *, config: Any,
    environment: Mapping[str, Any],
    run_spec_path: Path | str | None = None,
    run_spec_file_sha256: str | None = None,
    observed_git_state: Mapping[str, Any] | None = None,
    report_path: Path | str | None = None,
    trusted_recovery_context: Any | None = None,
) -> dict[str, Any]:
    """Build a self-hashed gate from the exact one-unit diagnostic batch."""

    root = Path(output_root).resolve()
    integrity_errors: list[str] = []
    terminal_errors: list[str] = []
    performance_errors: list[str] = []
    scope = validate_c7_diagnostic_plan(plan, config)
    if not scope["valid"]:
        integrity_errors.extend(scope["violations"])
    destination = _external_report_path(report_path, root, integrity_errors)
    run_spec_binding, run_spec_errors = _run_spec_binding(
        plan, run_spec_path, run_spec_file_sha256
    )
    integrity_errors.extend(run_spec_errors)
    mapped = map_execution_records(
        root, plan, observed_git_state=observed_git_state,
        trusted_recovery_context=trusted_recovery_context,
    )
    records = mapped["records"]
    integrity_errors.extend(mapped["mapping_errors"])
    layout_errors = _source_layout_errors(root, plan, records)
    integrity_errors.extend(layout_errors)
    source_bytes, source_control_bytes, source_files, tree_errors = (
        _measure_source_tree(root)
    )
    integrity_errors.extend(tree_errors)
    source_files, inventory_errors = _normalize_source_inventory(source_files, root)
    integrity_errors.extend(inventory_errors)
    bound_control_bytes = _control_bytes_sum(mapped["controls"])
    if bound_control_bytes is not None:
        if root.is_dir() and source_control_bytes != bound_control_bytes:
            integrity_errors.append(
                "measured batch-control bytes differ from bound control evidence"
            )
        source_control_bytes = bound_control_bytes
    validation = mapped["batch_validation"]
    if validation.get("valid") is not True:
        integrity_errors.extend(
            str(item) for item in validation.get("errors", [])
        )
    if validation.get("orchestration_complete") is not True:
        terminal_errors.append("diagnostic orchestration did not reach every terminal state")
    if validation.get("successful") is not True:
        terminal_errors.append("diagnostic NNS execution did not succeed")
    if len(records) != 1:
        integrity_errors.append("diagnostic source must map exactly one unit record")
        record: Mapping[str, Any] | None = None
    else:
        record = records[0]
        state = record.get("state")
        integrity_errors.extend(_telemetry_errors(
            record.get("runner_telemetry"), environment=environment,
            require_success=state == "succeeded",
        ))
        if state == "succeeded":
            integrity_errors.extend(_ig_errors(record))
        elif state == "failed" and record.get("failure") is None:
            integrity_errors.append("diagnostic failed without failure evidence")
        if record.get("trusted_by_batch_validator") is not True:
            integrity_errors.append("diagnostic unit is not trusted by batch validation")
    shared, shared_errors = _shared_provenance(records)
    integrity_errors.extend(shared_errors)
    integrity_errors.extend(_stage_c_model_binding_errors(
        shared.get("model") if isinstance(shared, Mapping) else None,
        trusted_recovery_context=trusted_recovery_context,
    ))
    shared_digest = _canonical_digest(shared) if shared else None
    for record in records:
        if (
            record.get("shared_provenance_sha256") is None
            and record.get("shared_provenance") == shared
        ):
            record["shared_provenance_sha256"] = shared_digest
    integrity_errors.extend(_git_binding_errors(observed_git_state, shared))
    environment_errors = _environment_errors(environment)
    if environment_errors:
        performance_errors.extend(environment_errors)
    total_wall = _record_total_wall(record) if record is not None else None
    if total_wall is None:
        performance_errors.append("diagnostic synchronized total wall time is unmeasured")
    elif total_wall >= DIAGNOSTIC_WALL_LIMIT_SECONDS:
        performance_errors.append(
            f"diagnostic synchronized wall {total_wall} is not < "
            f"{DIAGNOSTIC_WALL_LIMIT_SECONDS} seconds"
        )
    if integrity_errors:
        blocking_kind = "integrity"
        reasons = integrity_errors + terminal_errors + performance_errors
        integrity_valid = False
    elif terminal_errors:
        blocking_kind = "terminal"
        reasons = terminal_errors + performance_errors
        integrity_valid = True
    elif performance_errors:
        blocking_kind = "performance"
        reasons = performance_errors
        integrity_valid = True
    else:
        blocking_kind = None
        reasons = []
        integrity_valid = True
    status = STATUS_PASS if not reasons else STATUS_BLOCKED
    document: dict[str, Any] = {
        **_report_header(
            document_type=DIAGNOSTIC_DOCUMENT_TYPE, status=status,
            integrity_valid=integrity_valid, blocking_kind=blocking_kind,
            reasons=reasons,
        ),
        "scope": scope,
        "bindings": {
            "source_root": str(root),
            "execution_plan_sha256": plan.canonical_digest,
            "run_spec": run_spec_binding,
            "config_sha256": _canonical_digest(_config_dict(config)),
            "observed_git_state": _json_value(
                observed_git_state, "observed_git_state"
            ) if observed_git_state is not None else None,
            "source_controls": {
                name: {key: item.get(key) for key in ("path", "present", "bytes", "sha256", "error")}
                for name, item in mapped["controls"].items()
            },
            "execution_plan": plan.to_dict(),
            "shared_provenance": shared,
            "shared_provenance_sha256": shared_digest,
        },
        "environment": _json_value(environment, "environment"),
        "batch_validation": validation,
        "unit_record": None if record is None else _json_value(record, "unit record"),
        "diagnostic_measurement": {
            "source": (
                "status.runner_telemetry.phase_timings."
                "synchronized_total_wall_time_seconds"
            ),
            "synchronized_total_wall_time_seconds": total_wall,
            "limit_seconds_exclusive": DIAGNOSTIC_WALL_LIMIT_SECONDS,
            "comparison": "wall_time_seconds < 720",
            "status": (
                STATUS_PASS if total_wall is not None
                and total_wall < DIAGNOSTIC_WALL_LIMIT_SECONDS else STATUS_BLOCKED
            ),
        },
        "measurement_boundary": {
            "source_batch_root": str(root),
            "source_regular_file_bytes": source_bytes,
            "source_batch_control_bytes": source_control_bytes,
            "source_regular_files": source_files,
            "report_path": None if destination is None else str(destination),
            "report_external_to_source_batch": (
                destination is not None and not _path_is_within(destination, root)
            ),
            "diagnostic_is_forbidden_as_formal_budget_input": True,
        },
        "selection_performed": False, "bootstrap_performed": False,
        "decision_performed": False, "holdout_accessed": False,
    }
    document["canonical_digest"] = _self_digest(document)
    return document


def validate_c7_diagnostic_gate(
    value: Mapping[str, Any], *, trusted_recovery_context: Any | None = None,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("diagnostic gate must be a mapping")
    expected = {
        "schema_version", "document_type", "status", "integrity_valid",
        "blocking_kind", "exit_class", "blocked_reasons", "scope",
        "bindings", "environment", "batch_validation", "unit_record",
        "diagnostic_measurement", "measurement_boundary",
        "selection_performed", "bootstrap_performed",
        "decision_performed", "holdout_accessed", "canonical_digest",
    }
    _base_report_shape_errors(value, DIAGNOSTIC_DOCUMENT_TYPE, expected)
    common_errors = _common_existing_evidence_errors(
        value, formal=False, require_complete=value.get("integrity_valid") is True,
        trusted_recovery_context=trusted_recovery_context,
    )
    for field in (
        "selection_performed", "bootstrap_performed",
        "decision_performed", "holdout_accessed",
    ):
        if value.get(field) is not False:
            raise StageCBenchmarkIntegrityError(f"diagnostic gate {field} must be false")
    measurement = value.get("diagnostic_measurement")
    if not isinstance(measurement, Mapping):
        raise StageCBenchmarkIntegrityError("diagnostic measurement is missing")
    expected_measurement_fields = {
        "source", "synchronized_total_wall_time_seconds",
        "limit_seconds_exclusive", "comparison", "status",
    }
    if set(measurement) != expected_measurement_fields:
        raise StageCBenchmarkIntegrityError(
            "diagnostic measurement fields differ from the frozen shape"
        )
    if (
        measurement.get("source")
        != "status.runner_telemetry.phase_timings.synchronized_total_wall_time_seconds"
        or measurement.get("limit_seconds_exclusive") != 720.0
        or measurement.get("comparison") != "wall_time_seconds < 720"
    ):
        raise StageCBenchmarkIntegrityError(
            "diagnostic measurement source/limit/comparison semantics drifted"
        )
    wall = measurement.get("synchronized_total_wall_time_seconds")
    expected_measurement_pass = (
        not isinstance(wall, bool) and isinstance(wall, (int, float))
        and math.isfinite(float(wall)) and float(wall) < 720.0
    )
    if (measurement.get("status") == STATUS_PASS) != expected_measurement_pass:
        raise StageCBenchmarkIntegrityError(
            "diagnostic measurement verdict is inconsistent with exclusive 720 limit"
        )
    record = value.get("unit_record")
    if record is not None:
        if not isinstance(record, Mapping):
            common_errors.append("diagnostic unit record is malformed")
        else:
            common_errors.extend(_unit_identity_errors(record.get("unit_key")))
            telemetry_wall = _record_total_wall(record)
            if telemetry_wall != wall:
                common_errors.append(
                    "diagnostic measurement wall differs from status runner telemetry"
                )
            bindings = value.get("bindings")
            source_root = bindings.get("source_root") if isinstance(bindings, Mapping) else None
            identity = record.get("unit_key")
            digest = identity.get("canonical_digest") if isinstance(identity, Mapping) else None
            attempt = record.get("attempt_path")
            expected_parent = (
                Path(source_root) / "units" / str(digest) / "attempts"
                if isinstance(source_root, str) else None
            )
            if (
                not isinstance(attempt, str) or _polluted_path(attempt)
                or expected_parent is None
                or Path(attempt).resolve().parent != expected_parent.resolve()
                or _ATTEMPT_RE.fullmatch(Path(attempt).name) is None
            ):
                common_errors.append("diagnostic attempt path is not canonical")
            shared_digest = bindings.get("shared_provenance_sha256") if isinstance(bindings, Mapping) else None
            if record.get("shared_provenance_sha256") != shared_digest:
                common_errors.append("diagnostic unit/shared provenance digest differs")
    scope = value.get("scope")
    if value.get("integrity_valid") is True and (
        not isinstance(scope, Mapping)
        or scope.get("valid") is not True
        or scope.get("violations") != []
        or scope.get("counts") != C7_DIAGNOSTIC_EXPECTED_COUNTS
        or scope.get("candidate") != {"block_index": 0, "head_index": 0}
    ):
        common_errors.append("integrity-valid diagnostic scope is not exact")
    validation = value.get("batch_validation")
    if value.get("integrity_valid") is True and (
        not isinstance(validation, Mapping)
        or validation.get("valid") is not True
        or validation.get("orchestration_complete") is not True
    ):
        common_errors.append("integrity-valid diagnostic batch is invalid/incomplete")
    if value.get("integrity_valid") is True:
        environment_errors = _environment_errors(value.get("environment"))
        if value.get("blocking_kind") != "performance":
            common_errors.extend(environment_errors)
        if isinstance(record, Mapping) and record.get("state") == "succeeded":
            common_errors.extend(_telemetry_errors(
                record.get("runner_telemetry"),
                environment=value.get("environment")
                if isinstance(value.get("environment"), Mapping) else None,
            ))
            common_errors.extend(_ig_errors(record))
    if common_errors:
        raise StageCBenchmarkIntegrityError(
            "diagnostic semantic validation failed: " + "; ".join(common_errors)
        )
    if value.get("status") == STATUS_PASS:
        if value.get("integrity_valid") is not True:
            raise StageCBenchmarkIntegrityError("PASS diagnostic is not integrity-valid")
        scope = value.get("scope")
        if not isinstance(scope, Mapping) or (
            scope.get("valid") is not True
            or scope.get("counts") != C7_DIAGNOSTIC_EXPECTED_COUNTS
            or scope.get("candidate") != {"block_index": 0, "head_index": 0}
        ):
            raise StageCBenchmarkIntegrityError("PASS diagnostic scope is not exact")
        record = value.get("unit_record")
        if not isinstance(record, Mapping):
            raise StageCBenchmarkIntegrityError("PASS diagnostic lacks unit evidence")
        errors = _telemetry_errors(
            record.get("runner_telemetry"),
            environment=value.get("environment")
            if isinstance(value.get("environment"), Mapping) else None,
        ) + _ig_errors(record)
        if errors or record.get("trusted_by_batch_validator") is not True:
            raise StageCBenchmarkIntegrityError(
                "PASS diagnostic semantic validation failed: " + "; ".join(errors)
            )


def write_c7_diagnostic_gate(
    output_path: Path | str, output_root: Path | str, plan: ExecutionPlan, *,
    config: Any, environment: Mapping[str, Any],
    run_spec_path: Path | str, run_spec_file_sha256: str,
    observed_git_state: Mapping[str, Any],
    trusted_recovery_context: Any | None = None,
) -> dict[str, Any]:
    authority = _trusted_recovery_launch_record(trusted_recovery_context)
    if authority is not None:
        destination = _require_authorized_recovery_path(
            output_path, _recovery_gate_path(authority),
            "diagnostic gate destination",
        )
        root = _require_authorized_recovery_path(
            output_root, authority.diagnostic_root,
            "diagnostic output root",
        )
    else:
        destination = Path(output_path).resolve()
        root = Path(output_root).resolve()
    if _path_is_within(destination, root):
        raise StageCBenchmarkIntegrityError(
            "diagnostic gate must be external to the measured source root"
        )
    report = build_c7_diagnostic_gate(
        root, plan, config=config, environment=environment,
        run_spec_path=run_spec_path,
        run_spec_file_sha256=run_spec_file_sha256,
        observed_git_state=observed_git_state, report_path=destination,
        trusted_recovery_context=authority,
    )
    validate_c7_diagnostic_gate(
        report, trusted_recovery_context=authority,
    )
    _atomic_write_json(destination, report)
    return report


def _load_gate(
    value: Path | str | Mapping[str, Any],
) -> tuple[dict[str, Any], str | None, str | None]:
    if isinstance(value, Mapping):
        return _json_value(value, "gate"), None, None
    if not isinstance(value, (str, Path)):
        raise TypeError("gate_path_or_mapping must be a path or mapping")
    path = Path(value).resolve()
    _require_unpolluted_path(path, "diagnostic gate path")
    if path.is_symlink() or not path.is_file():
        raise StageCBenchmarkIntegrityError(
            "diagnostic gate path is missing, non-regular, or a symlink"
        )
    document = _strict_json(path)
    if not isinstance(document, dict):
        raise StageCBenchmarkIntegrityError("diagnostic gate document is not an object")
    return document, str(path), _sha256_file(path)


def _formal_contains_diagnostic_unit(
    gate: Mapping[str, Any], formal_plan: ExecutionPlan,
) -> tuple[bool, str | None]:
    record = gate.get("unit_record")
    diagnostic_identity = (
        record.get("unit_key") if isinstance(record, Mapping) else None
    )
    if not isinstance(diagnostic_identity, Mapping):
        return False, "diagnostic gate has no unit identity"
    formal_matches = [
        unit for unit in formal_plan.units
        if unit.key.canonical_digest == diagnostic_identity.get("canonical_digest")
    ]
    if len(formal_matches) != 1:
        return False, "diagnostic unit is not contained exactly once in formal plan"
    diagnostic_unit = gate.get("scope", {}).get("unit")
    formal_unit = formal_matches[0].to_dict()
    if diagnostic_unit != formal_unit:
        return False, "diagnostic unit workload differs from its formal-plan unit"
    return True, None


def require_c7_launch_gate(
    gate_path_or_mapping: Path | str | Mapping[str, Any],
    formal_plan: ExecutionPlan, *, config: Any, output_root: Path | str,
    observed_git_state: Mapping[str, Any],
    run_spec_path: Path | str | None = None,
    trusted_recovery_context: Any | None = None,
) -> dict[str, Any]:
    """Validate the diagnostic gate and authorize one fresh formal root."""

    authority = _trusted_recovery_launch_record(trusted_recovery_context)
    if authority is not None:
        if isinstance(gate_path_or_mapping, Mapping):
            raise StageCBenchmarkIntegrityError(
                "recovery C7 diagnostic gate must be loaded from the "
                "RecoveryC7LaunchRecord output path"
            )
        _require_authorized_recovery_path(
            gate_path_or_mapping, _recovery_gate_path(authority),
            "diagnostic gate path",
        )
        _require_authorized_recovery_path(
            output_root, authority.formal_root, "formal output root",
        )
    scope = validate_c7_execution_plan(formal_plan, config)
    if not scope["valid"]:
        raise StageCBenchmarkIntegrityError(
            "formal plan is not the exact C7 plan: "
            + "; ".join(scope["violations"])
        )
    gate, gate_path, gate_file_sha256 = _load_gate(gate_path_or_mapping)
    validate_c7_diagnostic_gate(
        gate, trusted_recovery_context=authority,
    )
    if gate.get("integrity_valid") is not True:
        raise StageCBenchmarkIntegrityError(
            "diagnostic gate reports invalid source integrity"
        )
    binding = gate.get("bindings")
    if not isinstance(binding, Mapping):
        raise StageCBenchmarkIntegrityError("diagnostic gate bindings are missing")
    shared = binding.get("shared_provenance")
    _require_stage_c_model_binding(
        shared.get("model") if isinstance(shared, Mapping) else None,
        trusted_recovery_context=authority,
        context="diagnostic launch gate",
    )
    git_errors = _git_binding_errors(observed_git_state, shared if isinstance(shared, Mapping) else None)
    if git_errors:
        raise StageCBenchmarkIntegrityError("; ".join(git_errors))
    config_sha = _canonical_digest(_config_dict(config))
    if binding.get("config_sha256") != config_sha:
        raise StageCBenchmarkIntegrityError("diagnostic gate config binding differs")
    if (
        not isinstance(shared, Mapping)
        or shared.get("configuration", {}).get("execution_plan_sha256")
        != binding.get("execution_plan_sha256")
    ):
        raise StageCBenchmarkIntegrityError(
            "diagnostic shared provenance does not bind its diagnostic plan"
        )
    contains, reason = _formal_contains_diagnostic_unit(gate, formal_plan)
    if not contains:
        raise StageCBenchmarkIntegrityError(reason or "diagnostic/formal unit mismatch")
    if run_spec_path is None:
        raise StageCBenchmarkIntegrityError("formal run-spec path is required")
    if authority is not None:
        contract = authority.document.get("c7_contracts", {}).get(
            "formal_run_spec", {}
        )
        declared = contract.get("path") if isinstance(contract, Mapping) else None
        if not isinstance(declared, str):
            raise StageCBenchmarkIntegrityError(
                "RecoveryC7LaunchRecord lacks formal run-spec authority"
            )
        _require_authorized_recovery_path(
            run_spec_path, Path(authority.repo_root) / declared,
            "formal run-spec path",
        )
    path = Path(run_spec_path).resolve()
    _require_unpolluted_path(path, "formal run-spec path")
    if path.is_symlink() or not path.is_file():
        raise StageCBenchmarkIntegrityError("formal run-spec file is missing or unsafe")
    formal_run_spec_file_sha256 = _sha256_file(path)
    if _strict_json(path) != formal_plan.run_spec.to_dict():
        raise StageCBenchmarkIntegrityError(
            "formal run-spec file differs from the formal execution plan"
        )
    measurement = gate.get("diagnostic_measurement")
    wall = measurement.get("synchronized_total_wall_time_seconds") if isinstance(measurement, Mapping) else None
    blocked: list[str] = []
    if gate.get("status") != STATUS_PASS:
        blocked.extend(str(item) for item in gate.get("blocked_reasons", []))
    if (
        isinstance(wall, bool) or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall)) or float(wall) >= 720.0
    ):
        blocked.append("diagnostic synchronized wall must be measured and < 720 seconds")
    try:
        formal_root = require_fresh_c7_output_root(output_root, kind="formal")
    except StageCBenchmarkBlockedError as error:
        blocked.extend(error.reasons)
        formal_root = Path(output_root).resolve()
    result = {
        "schema_version": 1, "document_type": LAUNCH_DOCUMENT_TYPE,
        "status": STATUS_PASS if not blocked else STATUS_BLOCKED,
        "integrity_valid": True,
        "blocking_kind": None if not blocked else "preflight",
        "exit_class": EXIT_CLASS_SUCCESS if not blocked else EXIT_CLASS_PREFLIGHT,
        "blocked_reasons": list(dict.fromkeys(blocked)),
        "gate_path": gate_path,
        "gate_file_sha256": gate_file_sha256,
        "gate_canonical_digest": gate.get("canonical_digest"),
        "diagnostic_execution_plan_sha256": binding.get("execution_plan_sha256"),
        "formal_execution_plan_sha256": formal_plan.canonical_digest,
        "formal_run_spec_path": str(path),
        "formal_run_spec_file_sha256": formal_run_spec_file_sha256,
        "formal_run_spec_semantic_sha256": _canonical_digest(
            formal_plan.run_spec.to_dict()
        ),
        "formal_output_root": str(formal_root),
        "diagnostic_unit_contained_identically": True,
        "observed_git_state": _json_value(observed_git_state, "observed Git state"),
    }
    result["canonical_digest"] = _self_digest(result)
    if blocked:
        raise StageCBenchmarkBlockedError(
            "formal C7 launch is blocked: " + "; ".join(blocked),
            report=result, reasons=blocked,
        )
    return result


def validate_c7_launch_authorization(
    value: Mapping[str, Any], *, trusted_recovery_context: Any | None = None,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("launch authorization must be a mapping")
    expected = {
        "schema_version", "document_type", "status", "integrity_valid",
        "blocking_kind", "exit_class", "blocked_reasons",
        "gate_path", "gate_file_sha256", "gate_canonical_digest",
        "diagnostic_execution_plan_sha256",
        "formal_execution_plan_sha256", "formal_run_spec_path",
        "formal_run_spec_file_sha256", "formal_run_spec_semantic_sha256",
        "formal_output_root", "diagnostic_unit_contained_identically",
        "observed_git_state", "canonical_digest",
    }
    if set(value) != expected:
        raise StageCBenchmarkIntegrityError("launch authorization fields drifted")
    _validate_self_digest(value, "launch authorization")
    if (
        value.get("schema_version") != 1
        or value.get("document_type") != LAUNCH_DOCUMENT_TYPE
        or value.get("status") != STATUS_PASS
        or value.get("integrity_valid") is not True
        or value.get("blocking_kind") is not None
        or value.get("exit_class") != EXIT_CLASS_SUCCESS
        or value.get("blocked_reasons") != []
        or value.get("diagnostic_unit_contained_identically") is not True
    ):
        raise StageCBenchmarkIntegrityError(
            "launch authorization is not an unblocked integrity-valid PASS"
        )
    for key in (
        "gate_file_sha256", "gate_canonical_digest",
        "diagnostic_execution_plan_sha256",
        "formal_execution_plan_sha256", "formal_run_spec_file_sha256",
        "formal_run_spec_semantic_sha256",
    ):
        if not _is_sha256(value.get(key)):
            raise StageCBenchmarkIntegrityError(
                f"launch authorization {key} is invalid"
            )
    git_errors = _git_binding_errors(value.get("observed_git_state"))
    if git_errors:
        raise StageCBenchmarkIntegrityError(
            "launch authorization Git binding is invalid: "
            + "; ".join(git_errors)
        )
    for key in ("gate_path", "formal_run_spec_path", "formal_output_root"):
        raw = value.get(key)
        if not isinstance(raw, str) or not raw or _polluted_path(raw):
            raise StageCBenchmarkIntegrityError(
                f"launch authorization {key} is invalid or polluted"
            )
    authority = _trusted_recovery_launch_record(trusted_recovery_context)
    if authority is not None:
        gate_path = _require_authorized_recovery_path(
            value["gate_path"], _recovery_gate_path(authority),
            "launch authorization diagnostic gate path",
        )
        _require_authorized_recovery_path(
            value["formal_output_root"], authority.formal_root,
            "launch authorization formal output root",
        )
        if not gate_path.is_file():
            raise StageCBenchmarkIntegrityError(
                "launch authorization diagnostic gate is missing or unsafe"
            )
        if _sha256_file(gate_path) != value["gate_file_sha256"]:
            raise StageCBenchmarkIntegrityError(
                "launch authorization diagnostic gate file binding differs"
            )
        gate = _strict_json(gate_path)
        if not isinstance(gate, Mapping):
            raise StageCBenchmarkIntegrityError(
                "launch authorization diagnostic gate is not an object"
            )
        validate_c7_diagnostic_gate(
            gate, trusted_recovery_context=authority,
        )
        if gate.get("canonical_digest") != value["gate_canonical_digest"]:
            raise StageCBenchmarkIntegrityError(
                "launch authorization diagnostic gate digest differs"
            )


def write_c7_launch_authorization(
    output_root: Path | str, authorization: Mapping[str, Any], *,
    trusted_recovery_context: Any | None = None,
) -> dict[str, Any]:
    """Claim a fresh formal root with the validated diagnostic launch proof."""

    authority = _trusted_recovery_launch_record(trusted_recovery_context)
    if authority is not None:
        _require_authorized_recovery_path(
            output_root, authority.formal_root, "formal output root",
        )
    root = require_fresh_c7_output_root(output_root, kind="formal")
    value = _json_value(authorization, "launch authorization")
    validate_c7_launch_authorization(
        value, trusted_recovery_context=authority,
    )
    if Path(value["formal_output_root"]).resolve() != root:
        raise StageCBenchmarkIntegrityError(
            "launch authorization formal output root binding differs"
        )
    path = root / LAUNCH_AUTHORIZATION_FILENAME
    _atomic_write_json(path, value)
    return value


def _bind_formal_launch_authorization(
    root: Path, plan: ExecutionPlan, *, diagnostic_gate_path: Path | str | None,
    observed_git_state: Mapping[str, Any] | None,
    run_spec_path: Path | str | None,
    trusted_recovery_context: Any | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    auth_path = root / LAUNCH_AUTHORIZATION_FILENAME
    if auth_path.is_symlink() or not auth_path.is_file():
        return None, None, ["formal source lacks launch authorization"]
    try:
        authorization = _strict_json(auth_path)
        if not isinstance(authorization, Mapping):
            raise StageCBenchmarkIntegrityError("authorization is not an object")
        validate_c7_launch_authorization(
            authorization,
            trusted_recovery_context=trusted_recovery_context,
        )
    except Exception as error:
        return None, None, [f"formal launch authorization is invalid: {error}"]
    if diagnostic_gate_path is None:
        return dict(authorization), None, ["diagnostic gate path is required for formal report"]
    gate_path = Path(diagnostic_gate_path).resolve()
    _require_unpolluted_path(gate_path, "formal report diagnostic gate path")
    if gate_path.is_symlink() or not gate_path.is_file():
        return dict(authorization), None, ["formal report diagnostic gate is missing or unsafe"]
    try:
        gate = _strict_json(gate_path)
        if not isinstance(gate, Mapping):
            raise StageCBenchmarkIntegrityError("gate is not an object")
        validate_c7_diagnostic_gate(
            gate, trusted_recovery_context=trusted_recovery_context,
        )
    except Exception as error:
        return dict(authorization), None, [f"formal diagnostic gate is invalid: {error}"]
    if gate.get("status") != STATUS_PASS:
        errors.append("formal diagnostic gate is not PASS")
    gate_file_sha256 = _sha256_file(gate_path)
    if (
        authorization.get("gate_path") != str(gate_path)
        or authorization.get("gate_file_sha256") != gate_file_sha256
        or authorization.get("gate_canonical_digest") != gate.get("canonical_digest")
    ):
        errors.append("launch authorization diagnostic gate binding differs")
    if authorization.get("formal_execution_plan_sha256") != plan.canonical_digest:
        errors.append("launch authorization formal plan binding differs")
    if Path(str(authorization.get("formal_output_root"))).resolve() != root:
        errors.append("launch authorization formal root binding differs")
    if authorization.get("observed_git_state") != observed_git_state:
        errors.append("launch authorization current Git binding differs")
    if run_spec_path is None:
        errors.append("formal run-spec path is required to bind launch authorization")
    else:
        spec_path = Path(run_spec_path).resolve()
        if (
            authorization.get("formal_run_spec_path") != str(spec_path)
            or not spec_path.is_file() or spec_path.is_symlink()
            or authorization.get("formal_run_spec_file_sha256") != _sha256_file(spec_path)
            or authorization.get("formal_run_spec_semantic_sha256")
            != _canonical_digest(plan.run_spec.to_dict())
        ):
            errors.append("launch authorization formal run-spec binding differs")
    return dict(authorization), dict(gate), errors


def build_benchmark_report(
    output_root: Path | str, plan: ExecutionPlan, *, config: Any,
    environment: Mapping[str, Any],
    run_spec_path: Path | str | None = None,
    run_spec_file_sha256: str | None = None,
    recomputed_estimate: StageCEstimate | Mapping[str, Any] | None = None,
    observed_git_state: Mapping[str, Any] | None = None,
    c7_diagnostic_gate: Path | str | None = None,
    report_path: Path | str | None = None,
    trusted_recovery_context: Any | None = None,
    **legacy: Any,
) -> dict[str, Any]:
    """Build the formal exact-40-unit C7 PASS/BLOCKED report."""

    # Two obsolete caller fields are accepted only to ease a rolling CLI/test
    # update.  They cannot replace the current observed state or add evidence.
    unknown_legacy = set(legacy) - {
        "working_tree_snapshot", "bindings", "lifecycle_evidence",
        "diagnostic_gate_path",
    }
    if unknown_legacy:
        raise TypeError(f"unknown benchmark report arguments: {sorted(unknown_legacy)}")
    if legacy.get("working_tree_snapshot") is not None and observed_git_state is not None:
        if dict(legacy["working_tree_snapshot"]) != dict(observed_git_state):
            raise StageCBenchmarkIntegrityError(
                "working_tree_snapshot differs from observed_git_state"
            )
    legacy_gate = legacy.get("diagnostic_gate_path")
    if c7_diagnostic_gate is not None and legacy_gate is not None:
        if Path(c7_diagnostic_gate).resolve() != Path(legacy_gate).resolve():
            raise StageCBenchmarkIntegrityError(
                "c7_diagnostic_gate and diagnostic_gate_path disagree"
            )
    diagnostic_gate_path = (
        c7_diagnostic_gate if c7_diagnostic_gate is not None else legacy_gate
    )
    root = Path(output_root).resolve()
    integrity_errors: list[str] = []
    terminal_errors: list[str] = []
    budget_errors: list[str] = []
    environment_errors: list[str] = []
    scope = validate_c7_execution_plan(plan, config)
    if not scope["valid"]:
        integrity_errors.extend(scope["violations"])
    destination = _external_report_path(report_path, root, integrity_errors)
    authorization, diagnostic_gate, launch_errors = _bind_formal_launch_authorization(
        root, plan, diagnostic_gate_path=diagnostic_gate_path,
        observed_git_state=observed_git_state, run_spec_path=run_spec_path,
        trusted_recovery_context=trusted_recovery_context,
    )
    integrity_errors.extend(launch_errors)
    run_spec_binding, spec_errors = _run_spec_binding(
        plan, run_spec_path, run_spec_file_sha256
    )
    integrity_errors.extend(spec_errors)
    mapped = map_execution_records(
        root, plan, observed_git_state=observed_git_state,
        trusted_recovery_context=trusted_recovery_context,
    )
    records = mapped["records"]
    controls = {
        **mapped["controls"],
        LAUNCH_AUTHORIZATION_FILENAME: _control_evidence(
            root, LAUNCH_AUTHORIZATION_FILENAME
        ),
    }
    integrity_errors.extend(mapped["mapping_errors"])
    try:
        integrity_errors.extend(_source_layout_errors(root, plan, records))
    except StageCBenchmarkIntegrityError as error:
        integrity_errors.append(str(error))
    source_bytes, source_control_bytes, source_files, tree_errors = _measure_source_tree(root)
    integrity_errors.extend(tree_errors)
    source_files, inventory_errors = _normalize_source_inventory(source_files, root)
    integrity_errors.extend(inventory_errors)
    bound_control_bytes = _control_bytes_sum(controls, _FORMAL_CONTROL_FILENAMES)
    if bound_control_bytes is not None:
        if root.is_dir() and source_control_bytes != bound_control_bytes:
            integrity_errors.append(
                "measured batch-control bytes differ from bound control evidence"
            )
        source_control_bytes = bound_control_bytes
    validation = mapped["batch_validation"]
    if validation.get("valid") is not True:
        integrity_errors.extend(str(item) for item in validation.get("errors", []))
    if validation.get("orchestration_complete") is not True:
        terminal_errors.append("formal source batch orchestration is incomplete")
    if validation.get("successful") is not True:
        terminal_errors.append("one or more formal execution units did not succeed")
    if len(records) != C7_EXPECTED_COUNTS["execution_units"]:
        integrity_errors.append("formal report must map exactly 40 unit records")
    shared, shared_errors = _shared_provenance(records)
    integrity_errors.extend(shared_errors)
    integrity_errors.extend(_stage_c_model_binding_errors(
        shared.get("model") if isinstance(shared, Mapping) else None,
        trusted_recovery_context=trusted_recovery_context,
    ))
    diagnostic_model = _shared_model_from_report(diagnostic_gate)
    if shared and diagnostic_model is not None and shared.get("model") != diagnostic_model:
        integrity_errors.append(
            "formal shared provenance model differs from diagnostic gate"
        )
    integrity_errors.extend(_git_binding_errors(observed_git_state, shared))
    environment_errors.extend(_environment_errors(environment))
    for record in records:
        digest = record.get("unit_key", {}).get("canonical_digest", "<unknown>")
        state = record.get("state")
        if state in {"succeeded", "failed"} and (
            record.get("trusted_by_batch_validator") is not True
        ):
            integrity_errors.append(f"unit {digest} is not trusted by batch validation")
        if state == "failed":
            telemetry_errors = _telemetry_errors(
                record.get("runner_telemetry"), environment=environment,
                require_success=False,
            )
            if telemetry_errors:
                integrity_errors.extend(
                    f"unit {digest}: {item}" for item in telemetry_errors
                )
            if record.get("failure") is None:
                integrity_errors.append(f"unit {digest} failed without failure evidence")
        elif state == "succeeded":
            integrity_errors.extend(
                f"unit {digest}: {item}" for item in _telemetry_errors(
                    record.get("runner_telemetry"), environment=environment
                )
            )
            integrity_errors.extend(
                f"unit {digest}: {item}" for item in _ig_errors(record)
            )
        else:
            integrity_errors.append(f"unit {digest} has non-terminal state {state!r}")
    preflight = controls["preflight.json"].get("document")
    estimate_document, estimate_errors = _estimate_binding(
        recomputed_estimate, plan, preflight
    )
    integrity_errors.extend(estimate_errors)
    caps, cap_binding_errors = _frozen_caps(config)
    integrity_errors.extend(cap_binding_errors)
    c6_payload = 0
    storage_model = None
    if estimate_document is not None:
        storage_model = estimate_document.get("storage")
        if isinstance(storage_model, Mapping):
            payload = storage_model.get("payload_bytes_before_control_overhead")
            if type(payload) is int and payload >= 0:
                c6_payload = payload
            else:
                integrity_errors.append("preflight C6 storage payload is invalid")
        else:
            integrity_errors.append("preflight C6 storage model is unavailable")
    try:
        projections = project_benchmark_budget(
            records, caps=caps or FROZEN_CAPS,
            c6_storage_payload_bytes=c6_payload,
            source_batch_regular_file_bytes=source_bytes,
            source_unit_count=len(plan.units),
            c6_storage_model=storage_model,
            source_batch_control_bytes=source_control_bytes,
        )
    except Exception as error:
        integrity_errors.append(f"budget projection cannot be recomputed: {error}")
        projections = _blocked_projection(caps or FROZEN_CAPS, str(error))
    if projections.get("all_hard_caps_pass") is not True:
        budget_errors.append("one or more frozen hard caps failed or were unmeasured")
    integrity_errors = list(dict.fromkeys(item for item in integrity_errors if item))
    terminal_errors = list(dict.fromkeys(terminal_errors))
    environment_errors = list(dict.fromkeys(environment_errors))
    budget_errors = list(dict.fromkeys(budget_errors))
    if integrity_errors:
        integrity_valid = False
        blocking_kind = "integrity"
        reasons = integrity_errors + terminal_errors + environment_errors + budget_errors
    elif terminal_errors:
        integrity_valid = True
        blocking_kind = "terminal"
        reasons = terminal_errors + environment_errors + budget_errors
    elif environment_errors:
        integrity_valid = True
        blocking_kind = "environment"
        reasons = environment_errors + budget_errors
    elif budget_errors:
        integrity_valid = True
        blocking_kind = "budget"
        reasons = budget_errors
    else:
        integrity_valid = True
        blocking_kind = None
        reasons = []
    status = STATUS_PASS if not reasons else STATUS_BLOCKED
    document: dict[str, Any] = {
        **_report_header(
            document_type=BENCHMARK_DOCUMENT_TYPE, status=status,
            integrity_valid=integrity_valid, blocking_kind=blocking_kind, reasons=reasons,
        ),
        "scope": scope,
        "bindings": {
            "source_root": str(root),
            "execution_plan_sha256": plan.canonical_digest,
            "run_spec": run_spec_binding,
            "config_sha256": _canonical_digest(_config_dict(config)),
            "recomputed_estimate_sha256": (
                None if estimate_document is None else estimate_document.get("canonical_digest")
            ),
            "recomputed_estimate": estimate_document,
            "observed_git_state": _json_value(
                observed_git_state, "observed Git state"
            ) if observed_git_state is not None else None,
            "source_controls": {
                name: {key: item.get(key) for key in ("path", "present", "bytes", "sha256", "error")}
                for name, item in controls.items()
            },
            "execution_plan": plan.to_dict(),
            "launch_authorization": authorization,
            "launch_authorization_file_sha256": controls[
                LAUNCH_AUTHORIZATION_FILENAME
            ].get("sha256"),
            "diagnostic_gate": diagnostic_gate,
            "diagnostic_gate_path": (
                None if diagnostic_gate_path is None
                else str(Path(diagnostic_gate_path).resolve())
            ),
            "diagnostic_gate_file_sha256": (
                None if diagnostic_gate_path is None
                or not Path(diagnostic_gate_path).is_file()
                else _sha256_file(Path(diagnostic_gate_path).resolve())
            ),
            "shared_provenance": shared,
            "shared_provenance_sha256": _canonical_digest(shared) if shared else None,
        },
        "environment": _json_value(environment, "environment"),
        "batch_validation": validation,
        "unit_records": records,
        "projections": projections,
        "measurement_boundary": {
            "source_batch_root": str(root),
            "all_regular_file_bytes": source_bytes,
            "source_batch_control_bytes": source_control_bytes,
            "source_regular_files": source_files,
            "report_path": None if destination is None else str(destination),
            "report_external_to_source_batch": (
                destination is not None and not _path_is_within(destination, root)
            ),
            "source_is_single_current_clean_commit_batch": (
                not integrity_errors and isinstance(observed_git_state, Mapping)
                and observed_git_state.get("dirty") is False
            ),
        },
        "selection_performed": False, "bootstrap_performed": False,
        "decision_performed": False, "holdout_accessed": False,
    }
    document["canonical_digest"] = _self_digest(document)
    return document


def _semantic_benchmark_errors(
    value: Mapping[str, Any], *, trusted_recovery_context: Any | None = None,
) -> list[str]:
    errors: list[str] = []
    scope = value.get("scope")
    if not isinstance(scope, Mapping) or (
        scope.get("valid") is not True
        or scope.get("violations") != []
        or scope.get("counts") != C7_EXPECTED_COUNTS
        or scope.get("candidate_policy", {}).get("candidates")
        != [{"block_index": b, "head_index": h} for b, h in C7_VALUE_CANDIDATES]
    ):
        errors.append("formal C7 scope is not exact")
    records = value.get("unit_records")
    if not isinstance(records, list) or len(records) != 40:
        errors.append("formal report must contain exactly 40 unit records")
        records = []
    bindings_for_plan = value.get("bindings")
    embedded_plan_value = (
        bindings_for_plan.get("execution_plan")
        if isinstance(bindings_for_plan, Mapping) else None
    )
    expected_unit_identities: dict[str, dict[str, Any]] = {}
    if isinstance(embedded_plan_value, Mapping):
        try:
            embedded_plan = ExecutionPlan.from_dict(embedded_plan_value)
        except Exception as error:
            errors.append(f"formal embedded execution plan is invalid: {error}")
        else:
            expected_unit_identities = {
                unit.key.canonical_digest: unit.key.to_dict()
                for unit in embedded_plan.units
            }
    digests: list[Any] = []
    method_counts: Counter[str] = Counter()
    shared_digests: set[Any] = set()
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("formal unit record is not an object")
            continue
        identity = record.get("unit_key")
        digest = identity.get("canonical_digest") if isinstance(identity, Mapping) else None
        errors.extend(_unit_identity_errors(identity))
        if expected_unit_identities.get(str(digest)) != identity:
            errors.append(f"unit {digest} identity is absent from bound execution plan")
        digests.append(digest)
        method = _method_from_record(record)
        method_counts[str(method)] += 1
        attempt_path = record.get("attempt_path")
        source_root = (
            value.get("measurement_boundary", {}).get("source_batch_root")
            if isinstance(value.get("measurement_boundary"), Mapping) else None
        )
        expected_parent = (
            Path(source_root) / "units" / str(digest) / "attempts"
            if isinstance(source_root, str) else None
        )
        if (
            not isinstance(attempt_path, str) or not attempt_path
            or _polluted_path(attempt_path)
            or expected_parent is None
            or Path(attempt_path).resolve().parent != expected_parent.resolve()
            or _ATTEMPT_RE.fullmatch(Path(attempt_path).name) is None
        ):
            errors.append(f"unit {digest} attempt path is unsafe or outside source root")
        if record.get("state") != "succeeded" or record.get("trusted_by_batch_validator") is not True:
            errors.append(f"unit {digest} is not trusted and successful")
        telemetry_errors = _telemetry_errors(
            record.get("runner_telemetry"),
            environment=value.get("environment")
            if isinstance(value.get("environment"), Mapping) else None,
        )
        errors.extend(f"unit {digest}: {item}" for item in telemetry_errors)
        errors.extend(f"unit {digest}: {item}" for item in _ig_errors(record))
        levels = record.get("level_bytes")
        if not isinstance(levels, Mapping) or set(levels) != {*_LEVELS, "control"}:
            errors.append(f"unit {digest} level byte evidence is malformed")
        elif (
            record.get("total_regular_file_bytes") != sum(levels.values())
            or record.get("l0_l1_cap_accounting_bytes")
            != levels["L0"] + levels["L1"] + levels["L3"] + levels["control"]
        ):
            errors.append(f"unit {digest} byte arithmetic is inconsistent")
        hashes = record.get("hashes")
        if not isinstance(hashes, Mapping) or any(
            not _is_sha256(hashes.get(key))
            for key in (
                "manifest_sha256", "status_sha256",
                "selected_status_sha256", "provenance_sha256",
                "result_sha256", "unit_identity_sha256",
            )
        ) or hashes.get("status_sha256") != hashes.get("selected_status_sha256"):
            errors.append(f"unit {digest} hash chain evidence is malformed")
        shared_digests.add(record.get("shared_provenance_sha256"))
    if len(digests) != len(set(digests)):
        errors.append("formal unit records contain duplicate identities")
    if set(digests) != set(expected_unit_identities):
        errors.append("formal unit identity set differs from bound execution plan")
    expected_method_counts = {
        **{method: 12 for method in VALUE_METHOD_IDS},
        **{method: 1 for method in CLASS_METHOD_IDS},
    }
    if dict(method_counts) != expected_method_counts:
        errors.append("formal per-method unit counts are invalid")
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping):
        errors.append("formal report bindings are missing")
    else:
        shared = bindings.get("shared_provenance")
        shared_digest = bindings.get("shared_provenance_sha256")
        if (
            not isinstance(shared, Mapping) or set(shared) != set(SHARED_PROVENANCE_KEYS)
            or _canonical_digest(shared) != shared_digest
            or shared_digests != {shared_digest}
        ):
            errors.append("formal shared provenance binding is inconsistent")
        if shared.get("code") != bindings.get("observed_git_state"):
            errors.append("formal observed Git binding differs from provenance")
        if not isinstance(bindings.get("observed_git_state"), Mapping) or bindings["observed_git_state"].get("dirty") is not False:
            errors.append("formal Git binding is not clean")
        errors.extend(
            _git_binding_errors(
                bindings.get("observed_git_state"),
                shared if isinstance(shared, Mapping) else None,
            )
        )
        errors.extend(_stage_c_model_binding_errors(
            shared.get("model") if isinstance(shared, Mapping) else None,
            trusted_recovery_context=trusted_recovery_context,
        ))
    environment_errors = _environment_errors(value.get("environment"))
    errors.extend(environment_errors)
    projections = value.get("projections")
    if not isinstance(projections, Mapping):
        errors.append("formal projections are missing")
    else:
        expected_counts = {
            "C8": 2180, "C9": 320, "C8_C9": 2500,
            "StageD": 8000, "StageD_images": 500,
        }
        if projections.get("exact_unit_counts") != expected_counts:
            errors.append("formal downstream exact unit counts are invalid")
        expected_profiles = {
            name: _downstream_profile(name) for name in _DOWNSTREAM_PROFILES
        }
        if projections.get("downstream_profiles") != expected_profiles:
            errors.append("formal downstream profile formulas are invalid")
        checks = projections.get("hard_cap_checks")
        if (
            not isinstance(checks, list) or len(checks) != 10
            or [item.get("cap_key") for item in checks if isinstance(item, Mapping)]
            != list(HARD_CAP_KEYS)
            or any(
                not isinstance(item, Mapping)
                or item.get("cap") != FROZEN_CAPS[item.get("cap_key")]
                or item.get("comparison") != "observed <= cap"
                or item.get("measured") is not True
                or item.get("status") != STATUS_PASS
                or item.get("observed") > item.get("cap")
                for item in checks
            )
        ):
            errors.append("formal hard-cap checks are incomplete or failed")
        if projections.get("all_hard_caps_pass") is not True:
            errors.append("formal projection does not pass all hard caps")
        time = projections.get("time")
        slowest = max(
            records, key=lambda record: (_record_total_wall(record) or -1.0),
            default=None,
        )
        if slowest is not None:
            seconds = _record_total_wall(slowest)
            identity = slowest.get("unit_key")
            expected_slowest = {
                "unit_key": identity,
                "unit_key_sha256": identity.get("canonical_digest")
                if isinstance(identity, Mapping) else None,
                "attempt_path": slowest.get("attempt_path"),
                "wall_time_seconds": seconds,
            }
            if not isinstance(time, Mapping) or time.get("slowest_valid_measured_unit") != expected_slowest:
                errors.append("formal slowest synchronized unit evidence is inconsistent")
            expected_times = {
                "C8_h20_gpu_hours": seconds * 2180 * 1.25 / 3600,
                "C9_h20_gpu_hours": seconds * 320 * 1.25 / 3600,
                "C8_C9_h20_gpu_hours": seconds * 2500 * 1.25 / 3600,
                "StageD_h20_gpu_hours": seconds * 8000 * 1.25 / 3600,
                "StageD_h20_gpu_hours_per_image": seconds * 16 * 1.25 / 3600,
            }
            if not isinstance(time, Mapping) or any(time.get(key) != expected for key, expected in expected_times.items()):
                errors.append("formal time projection arithmetic is inconsistent")
    boundary = value.get("measurement_boundary")
    if not isinstance(boundary, Mapping) or (
        boundary.get("report_external_to_source_batch") is not True
        or boundary.get("source_is_single_current_clean_commit_batch") is not True
    ):
        errors.append("formal measurement boundary is invalid")
    return errors


def _formal_projection_recompute_errors(value: Mapping[str, Any]) -> list[str]:
    records = value.get("unit_records")
    projections = value.get("projections")
    boundary = value.get("measurement_boundary")
    if not isinstance(records, list) or not isinstance(projections, Mapping) or not isinstance(boundary, Mapping):
        return ["formal projection operands are unavailable"]
    storage = projections.get("storage")
    if not isinstance(storage, Mapping):
        return ["formal storage projection operands are unavailable"]
    errors: list[str] = []
    if storage.get("source_batch_regular_file_bytes") != boundary.get("all_regular_file_bytes"):
        errors.append("projected source bytes differ from measurement boundary")
    if storage.get("source_batch_control_bytes") != boundary.get("source_batch_control_bytes"):
        errors.append("projected control bytes differ from measurement boundary")
    bindings = value.get("bindings")
    estimate = bindings.get("recomputed_estimate") if isinstance(bindings, Mapping) else None
    if not isinstance(estimate, Mapping):
        return ["bound recomputed estimate evidence is unavailable"]
    try:
        parsed_estimate = StageCEstimate.from_dict(estimate).to_dict()
    except Exception as error:
        return [f"bound recomputed estimate is invalid: {error}"]
    if parsed_estimate.get("canonical_digest") != bindings.get("recomputed_estimate_sha256"):
        errors.append("bound recomputed estimate digest differs from binding")
    source_model = parsed_estimate.get("storage")
    expected_payload = (
        source_model.get("payload_bytes_before_control_overhead")
        if isinstance(source_model, Mapping) else None
    )
    if storage.get("c6_exact_model_payload_bytes") != expected_payload:
        errors.append("projection C6 payload differs from bound estimate")
    model_operands = storage.get("c6_model_operands")
    if (
        not isinstance(model_operands, Mapping)
        or model_operands.get("source_storage_model") != source_model
    ):
        errors.append("projection C6 storage model differs from bound estimate")
    try:
        recomputed = project_benchmark_budget(
            records, caps=FROZEN_CAPS,
            c6_storage_payload_bytes=expected_payload,
            source_batch_regular_file_bytes=boundary["all_regular_file_bytes"],
            source_unit_count=len(records), c6_storage_model=source_model,
            source_batch_control_bytes=boundary["source_batch_control_bytes"],
        )
    except Exception as error:
        errors.append(f"formal projection cannot be independently recomputed: {error}")
    else:
        if recomputed != projections:
            errors.append("formal projection differs from independent recomputation")
    return errors


def _existing_formal_record_errors(value: Mapping[str, Any]) -> list[str]:
    records = value.get("unit_records")
    if not isinstance(records, list):
        return ["formal unit_records is not an array"]
    errors: list[str] = []
    bindings = value.get("bindings")
    source_root = bindings.get("source_root") if isinstance(bindings, Mapping) else None
    shared_digest = (
        bindings.get("shared_provenance_sha256")
        if isinstance(bindings, Mapping) else None
    )
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(f"formal unit_records[{index}] is not an object")
            continue
        identity_errors = _unit_identity_errors(record.get("unit_key"))
        errors.extend(f"unit[{index}]: {item}" for item in identity_errors)
        identity = record.get("unit_key")
        digest = identity.get("canonical_digest") if isinstance(identity, Mapping) else None
        if isinstance(digest, str):
            if digest in seen:
                errors.append(f"duplicate formal unit identity {digest}")
            seen.add(digest)
        attempt = record.get("attempt_path")
        if attempt is not None:
            expected_parent = (
                Path(source_root) / "units" / str(digest) / "attempts"
                if isinstance(source_root, str) else None
            )
            if (
                not isinstance(attempt, str) or _polluted_path(attempt)
                or expected_parent is None
                or Path(attempt).resolve().parent != expected_parent.resolve()
                or _ATTEMPT_RE.fullmatch(Path(attempt).name) is None
            ):
                errors.append(f"unit {digest} retained attempt path is noncanonical")
        hashes = record.get("hashes")
        if isinstance(hashes, Mapping):
            for key, digest_value in hashes.items():
                if digest_value is not None and not _is_sha256(digest_value):
                    errors.append(f"unit {digest} retained hash {key} is invalid")
            if (
                hashes.get("status_sha256") is not None
                and hashes.get("selected_status_sha256")
                != hashes.get("status_sha256")
            ):
                errors.append(f"unit {digest} retained selected status hash differs")
            if (
                hashes.get("unit_identity_sha256") is not None
                and hashes.get("unit_identity_sha256") != digest
            ):
                errors.append(f"unit {digest} retained identity hash differs")
        if record.get("shared_provenance_sha256") is not None and (
            record.get("shared_provenance_sha256") != shared_digest
        ):
            errors.append(f"unit {digest} retained shared provenance digest differs")
        levels = record.get("level_bytes")
        if isinstance(levels, Mapping):
            if set(levels) != {*_LEVELS, "control"} or any(
                type(item) is not int or item < 0 for item in levels.values()
            ):
                errors.append(f"unit {digest} retained level bytes are malformed")
            elif (
                record.get("total_regular_file_bytes") is not None
                and record.get("total_regular_file_bytes") != sum(levels.values())
            ):
                errors.append(f"unit {digest} retained total bytes are inconsistent")
    return errors


def validate_benchmark_report(
    value: Mapping[str, Any], *, trusted_recovery_context: Any | None = None,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("benchmark report must be a mapping")
    expected = {
        "schema_version", "document_type", "status", "integrity_valid",
        "blocking_kind", "exit_class", "blocked_reasons", "scope",
        "bindings", "environment", "batch_validation", "unit_records",
        "projections", "measurement_boundary", "selection_performed",
        "bootstrap_performed", "decision_performed", "holdout_accessed",
        "canonical_digest",
    }
    _base_report_shape_errors(value, BENCHMARK_DOCUMENT_TYPE, expected)
    common_errors = _common_existing_evidence_errors(
        value, formal=True, require_complete=value.get("integrity_valid") is True,
        trusted_recovery_context=trusted_recovery_context,
    )
    common_errors.extend(_existing_formal_record_errors(value))
    bindings = value.get("bindings")
    if isinstance(bindings, Mapping):
        authorization = bindings.get("launch_authorization")
        if authorization is not None:
            try:
                validate_c7_launch_authorization(
                    authorization,
                    trusted_recovery_context=trusted_recovery_context,
                )
            except Exception as error:
                common_errors.append(
                    f"retained launch authorization is invalid: {error}"
                )
        diagnostic_gate = bindings.get("diagnostic_gate")
        if diagnostic_gate is not None:
            try:
                validate_c7_diagnostic_gate(
                    diagnostic_gate,
                    trusted_recovery_context=trusted_recovery_context,
                )
            except Exception as error:
                common_errors.append(f"retained diagnostic gate is invalid: {error}")
        if isinstance(authorization, Mapping) and isinstance(diagnostic_gate, Mapping):
            if (
                authorization.get("gate_canonical_digest")
                != diagnostic_gate.get("canonical_digest")
            ):
                common_errors.append(
                    "retained launch authorization diagnostic gate digest differs"
                )
            if (
                authorization.get("gate_file_sha256")
                != bindings.get("diagnostic_gate_file_sha256")
            ):
                common_errors.append(
                    "retained launch authorization diagnostic gate file binding differs"
                )
        if isinstance(diagnostic_gate, Mapping) and (
            diagnostic_gate.get("status") != STATUS_PASS
            or diagnostic_gate.get("integrity_valid") is not True
        ):
            common_errors.append(
                "retained diagnostic gate is not an integrity-valid PASS"
            )
        if (
            _shared_model_from_report(value) is not None
            and _shared_model_from_report(diagnostic_gate) is not None
            and _shared_model_from_report(value)
            != _shared_model_from_report(diagnostic_gate)
        ):
            common_errors.append(
                "formal shared provenance model differs from diagnostic gate"
            )
        estimate = bindings.get("recomputed_estimate")
        estimate_sha = bindings.get("recomputed_estimate_sha256")
        if estimate is not None:
            try:
                parsed = StageCEstimate.from_dict(estimate).to_dict()
            except Exception as error:
                common_errors.append(f"retained recomputed estimate is invalid: {error}")
            else:
                if parsed.get("canonical_digest") != estimate_sha:
                    common_errors.append("retained recomputed estimate digest differs")
                if parsed.get("plan_digest") != bindings.get("execution_plan_sha256"):
                    common_errors.append("retained recomputed estimate plan differs")
        elif value.get("integrity_valid") is True:
            common_errors.append("integrity-valid report lacks recomputed estimate evidence")
    for field in (
        "selection_performed", "bootstrap_performed",
        "decision_performed", "holdout_accessed",
    ):
        if value.get(field) is not False:
            raise StageCBenchmarkIntegrityError(f"C7 report {field} must be false")
    # A BLOCKED report may omit unavailable evidence, but every retained
    # cross-link and derived projection must remain internally truthful.
    projections = value.get("projections")
    if isinstance(projections, Mapping) and isinstance(projections.get("storage"), Mapping) and projections["storage"]:
        common_errors.extend(_formal_projection_recompute_errors(value))
    if common_errors:
        raise StageCBenchmarkIntegrityError(
            "benchmark semantic validation failed: " + "; ".join(common_errors)
        )
    if value.get("status") == STATUS_PASS:
        errors = _semantic_benchmark_errors(
            value, trusted_recovery_context=trusted_recovery_context,
        )
        if errors:
            raise StageCBenchmarkIntegrityError(
                "benchmark PASS semantic validation failed: " + "; ".join(errors)
            )


def write_benchmark_report(
    output_path: Path | str, output_root: Path | str, plan: ExecutionPlan, *,
    config: Any, environment: Mapping[str, Any],
    run_spec_path: Path | str, run_spec_file_sha256: str,
    recomputed_estimate: StageCEstimate | Mapping[str, Any],
    observed_git_state: Mapping[str, Any],
    c7_diagnostic_gate: Path | str,
    trusted_recovery_context: Any | None = None,
) -> dict[str, Any]:
    authority = _trusted_recovery_launch_record(trusted_recovery_context)
    if authority is not None:
        destination = _require_authorized_recovery_path(
            output_path, _recovery_report_path(authority),
            "benchmark report destination",
        )
        root = _require_authorized_recovery_path(
            output_root, authority.formal_root, "formal output root",
        )
        _require_authorized_recovery_path(
            c7_diagnostic_gate, _recovery_gate_path(authority),
            "diagnostic gate path",
        )
    else:
        destination = Path(output_path).resolve()
        root = Path(output_root).resolve()
    if _path_is_within(destination, root):
        raise StageCBenchmarkIntegrityError(
            "benchmark report must be external to the measured source root"
        )
    report = build_benchmark_report(
        root, plan, config=config, environment=environment,
        run_spec_path=run_spec_path,
        run_spec_file_sha256=run_spec_file_sha256,
        recomputed_estimate=recomputed_estimate,
        observed_git_state=observed_git_state,
        c7_diagnostic_gate=c7_diagnostic_gate, report_path=destination,
        trusted_recovery_context=authority,
    )
    validate_benchmark_report(
        report, trusted_recovery_context=authority,
    )
    _atomic_write_json(destination, report)
    return report


# Verbose aliases preserve discoverability for external report consumers.
build_stage_c_benchmark_report = build_benchmark_report
write_stage_c_benchmark_report = write_benchmark_report
validate_stage_c_benchmark_report = validate_benchmark_report


__all__ = [
    "BENCHMARK_DOCUMENT_TYPE", "C7_DIAGNOSTIC_CANDIDATE",
    "C7_DIAGNOSTIC_EXPECTED_COUNTS", "C7_DIAGNOSTIC_METHOD_ID",
    "C7_DIAGNOSTIC_SAMPLE_ID", "C7_EXPECTED_COUNTS",
    "C7_FIDELITY_PAIRING", "C7_VALUE_CANDIDATES",
    "C8_C9_EXECUTION_UNITS", "C8_EXECUTION_UNITS",
    "C9_EXECUTION_UNITS", "DIAGNOSTIC_DOCUMENT_TYPE",
    "DIAGNOSTIC_GATE_FILENAME", "DIAGNOSTIC_WALL_LIMIT_SECONDS",
    "FROZEN_CAPS", "HARD_CAP_KEYS", "REPORT_FILENAME",
    "STATUS_BLOCKED", "STATUS_PASS", "STAGE_D_EXECUTION_UNITS",
    "STAGE_D_IMAGES", "StageCBenchmarkBlockedError",
    "StageCBenchmarkError", "StageCBenchmarkIntegrityError",
    "StageCLaunchBlockedError", "build_benchmark_report",
    "build_c7_diagnostic_gate", "build_stage_c_benchmark_report",
    "capture_benchmark_environment", "map_execution_records",
    "project_benchmark_budget", "require_c7_launch_gate",
    "require_fresh_c7_output_root", "validate_benchmark_report",
    "validate_c7_diagnostic_execution_plan",
    "validate_c7_diagnostic_gate", "validate_c7_diagnostic_plan",
    "validate_c7_execution_plan", "validate_stage_c_benchmark_report",
    "write_benchmark_report", "write_c7_diagnostic_gate",
    "write_stage_c_benchmark_report",
]

