"""Fail-closed orchestration for the frozen Stage C9 pilot.

This module is intentionally an evidence producer, not a statistical reader.
It validates the completed C8 hand-off before any C9 root is created, reuses
the C8 immutable-attempt machinery, and emits a closed 320-row evidence
report.  It never performs bootstrap, Holm correction, or a C10 decision.
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
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .constants import DEFAULT_STATE_DICT, REPO_ROOT
from . import stage_c_c8 as _c8
from .io_utils import sha256_file
from .stage_c_config import DEFAULT_STAGE_C_CONFIG, DEFAULT_STAGE_C_SCHEMA
from .stage_c_benchmark import (
    FROZEN_CAPS, HARD_CAP_KEYS, StageCBenchmarkBlockedError,
    StageCBenchmarkIntegrityError, project_benchmark_budget,
)
from .stage_c_plan import (
    AUDIT_SAMPLE_IDS, CALIBRATION_SAMPLE_IDS, CENTER_RANKING,
    CLASS_METHOD_IDS, CURVE_DIRECTIONS, CURVE_POINT_KIND, ExecutionPlan,
    DEFAULT_ACCEPTANCE, DEFAULT_DEPENDENCY_LOCK, DEFAULT_REGISTRY_SCHEMA,
    DEFAULT_SELECTION_SCHEMA,
    FIDELITY_SCALES, MAP_KIND, RANDOM_RANKING, StageCEstimate,
    TARGET_RANKING, VALUE_METHOD_IDS, WorkItem, enforce_stage_c_budget,
    estimate_stage_c_run,
)
from .stage_c_perturbations import PERTURBATION_IDS
from .stage_c_provenance import canonical_sha256, capture_git_state
from .stage_c_resources import measure_artifact_tree_bytes
from .stage_c_runner import run_execution_unit
from .stage_c_selection import (
    _record_validity, _sample_candidate_metrics,
    build_c8_calibration_evidence,
    validate_stage_c_selection_document,
    validate_stage_c_selection_report,
)

SCHEMA_VERSION = 1
C9_EXECUTION_UNITS = 320
C9_UNITS_PER_IMAGE = 16
C9_EXPECTED_COUNTS = {
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

C9_SAMPLE_IDS = (
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
C9_HOLDOUT_SAMPLE_IDS = tuple(
    sample_id for sample_id in C9_SAMPLE_IDS
    if sample_id not in CALIBRATION_SAMPLE_IDS
)
C9_WINNERS = ((0, 9), (4, 2), (6, 10), (11, 2))
C9_PRIMARY = (0, 9)
_ALL_METHODS = (*VALUE_METHOD_IDS, *CLASS_METHOD_IDS)

# Frozen C8 hand-off.  File SHA-256 is over exact bytes; canonical digests are
# over the corresponding JSON object without presentation whitespace.
C8_PRODUCER_COMMIT = "1fd25392968c9ccfde6c066f6544e9c4557ac4f0"
C8_GENERATOR_COMMIT = "d7f2ef3bca8c22e1fc1ae38d9333fd2cd3e11579"
C8_COMPLETION_COMMIT = "c03f01ff2feb0f250b49974688b5fa8d511161b1"
C8_REPORT_FILE_SHA256 = "d921d95ebe756e8cec859faea96578f8b3da766e9049e0440f570487d2e40b44"
C8_REPORT_CANONICAL_DIGEST = "9eb0df50be93320d7d6279852441c5bc103eaef146b394ff8d128cc64d2c9a6c"
C8_SELECTION_FILE_SHA256 = "5dff9b5aec77cab22b157b9bf2b5d63b2252b7e50ebe728250206d655a142254"
C8_SELECTION_CANONICAL_DIGEST = "6615acb46a97b1a7f34402bbe07d1de542ff12c7f82f321eb00160a3168eae48"
RECOVERY_C8_SELECTION_FILE_SHA256 = (
    "5d26c78bb94278c7f12d7bd88b1a37cfb97352f3d0bde5e1c4e52f8e811c7982"
)
C8_EXECUTION_PLAN_SHA256 = "ece22206df300d4099c888d9ce83be83bbf926eaf4d5003059694912881f0892"
C8_RUN_SPEC_SHA256 = "42f22bb4992dd560382acd48a5e5ec2fe47b0345835e9201636943fb22e7e30b"
C8_PREFLIGHT_SHA256 = "175d5c9c8461dd5b082792a57ffb7f627c62f2c540114b8789039ebc334b7a2b"
C8_CALIBRATION_ARTIFACT_MANIFEST_SHA256 = "705b92c6101b5b4354eb3d2b922aced8d781dcf5ee6d3b7198a6132d9a495788"
C8_CALIBRATION_EVIDENCE_SHA256 = "08daa501d3e18ba927180156537de86a3c62fac6176dde3f3a6a37f0916e1620"
C8_SOURCE_TREE_TOTAL_BYTES = 1_964_120_421
C8_CONTROL_FILE_SHA256 = {
    "execution_plan.json": "da24efd1a700c59ae378a87b7b5b575f5d09edcbf40111c515f1321b2db899f9",
    "preflight.json": "c7d3a9baa1b95c2c6b1fdd9af70b9714c18db9230e34e437a401e94a329899f2",
    "c8_launch_authorization.json": "86a60e0a3c915b1295a803ace0c9ac6c414ef99a0dac7a879b30abd8663999fe",
    "failure_table.json": "deacdadd2c7ebe42b253bd1da86a264664e1d4a885c95cb8ed92e2ecfce327d0",
    "batch_status.json": "7c37bb76ae42d445912b953e582a02b30a16107ba9d5ce43e3fa1f4199f6a708",
}
C8_SLOWEST_UNIT_SECONDS = 197.04103458393365
C8_PEAK_RESERVED_BYTES = 4_412_407_808
C8_PROJECTED_ARTIFACT_BYTES = 3_527_349_125
C9_PROJECTED_ARTIFACT_BYTES = 440_416_200
C8_C9_PROJECTED_ARTIFACT_BYTES = 3_967_765_325
C9_PROJECTED_H20_GPU_HOURS = 21.893448287103737
C8_C9_PROJECTED_H20_GPU_HOURS = 171.04256474299797
STAGE_D_PROJECTED_ARTIFACT_BYTES = 8_700_496_200
STAGE_D_PROJECTED_H20_GPU_HOURS = 547.3362071775935
STAGE_D_PROJECTED_H20_GPU_HOURS_PER_IMAGE = 1.094672414355187
FROZEN_INPUT_SHA256 = {
    "model_sha256": "2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0",
    "registry_sha256": "9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0",
    "registry_metadata_sha256": "bedb9df9a77807c86da0b898f8d44767c8276f21ab69821965f2b30d7184a4af",
    "config_sha256": "0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669",
    "config_schema_sha256": "94209b7c8a63a62ef27f3396810c8d5be8f6fbbf0a4df22e663b3661b9ed3f4b",
    "registry_schema_sha256": "4ce0ac8cd5bff93ae2b323c3e0447a87fc825c3243ad46c54c9b48e05e9b2d6d",
    "selection_schema_sha256": "867a3cf22519b2a045d81751a20c5cd52570ddeadede26a2da2bb79973f0b59d",
    "protocol_sha256": "ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8",
    "acceptance_sha256": "f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd",
    "dependency_lock_sha256": "81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984",
    "stage_a_manifest_sha256": "042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89",
}

C8_SOURCE_ROOT = Path(
    "/data00/home/yeziqi.831/testcode/vitnns_c8_formal_1fd2539_20260823"
)
C8_REPORT_PATH = Path(
    "/data00/home/yeziqi.831/testcode/"
    "vitnns_c8_selection_report_final_d7f2ef3_20260824.json"
)
C8_RECOVERY_SELECTION_PATH = Path(
    "/data00/home/yeziqi.831/testcode/"
    "vitnns_c8_selection_recovery_d7f2ef3_20260824.json"
)

DEFAULT_TRACKED_SELECTION = REPO_ROOT / "selection/stage_c_selection.json"
AUTHORIZATION_FILENAME = "c9_launch_authorization.json"
FINAL_REPORT_FILENAME = "c9_evidence_report.json"
_CHECKPOINT_LOCK_FILENAME = ".c9-checkpoint.lock"
_LIFECYCLE_LOCK_FILENAME = ".c9-lifecycle.lock"
_ARTIFACT_ACTIVITY_LOCK_FILENAME = ".c9-artifact-activity.lock"
_CONTROL_NAMES = (
    "execution_plan.json", "preflight.json", "failure_table.json",
    "batch_status.json", AUTHORIZATION_FILENAME,
)
_SHA256 = frozenset("0123456789abcdef")
_COMMIT = re.compile(r"^[0-9a-f]{40}\Z")
_ATTEMPT_NAME = re.compile(r"attempt-[0-9]{6}\Z")
_TERMINAL_STATES = frozenset({"succeeded", "failed"})
_FINAL_STATES = frozenset({
    "succeeded", "failed", "missing", "incomplete", "tampered",
})
_CHECKPOINT_STATES = frozenset({
    "pending", "claimed", "succeeded", "failed", "missing",
    "incomplete", "tampered",
})
_LIFECYCLE_STATES = frozenset({
    "running", "blocked", "finalizing", "finalized",
})
_ROW_KEYS = frozenset({
    "unit_key", "state", "terminal", "complete",
    "selected_attempt", "selected_status_sha256", "failure",
    "claim_token", "l0_l1_bytes", "l2_bytes", "total_bytes",
})
_STATUS_KEYS = frozenset({
    "schema_version", "document_type", "plan_digest",
    "expected_unit_count", "orchestration_complete", "successful",
    "failure_table_sha256", "lifecycle_state",
    "active_claim_count", "validation_errors",
})
_HANDOFF_KEYS = frozenset({
    "schema_version", "document_type", "status", "integrity_valid",
    "c8_root", "c8_report_path", "c8_report_file_sha256",
    "c8_report_canonical_digest", "c8_recovery_selection_path",
    "c8_recovery_selection_file_sha256", "tracked_selection_path",
    "tracked_selection_file_sha256", "selection_canonical_digest",
    "c8_producer_code", "c8_generator_code", "c8_completion_commit",
    "c8_control_file_sha256", "c8_execution_plan_sha256",
    "c8_run_spec_sha256", "c8_preflight_sha256",
    "c8_calibration_artifact_manifest_sha256",
    "c8_calibration_evidence_sha256",
    "c8_source_tree_total_bytes", "c8_retained_artifact_bytes",
    "c9_projected_artifact_bytes", "c8_c9_projected_artifact_bytes",
    "c9_projected_h20_gpu_hours", "c8_c9_projected_h20_gpu_hours",
    "canonical_digest",
})
_AUTHORIZATION_KEYS = frozenset({
    "schema_version", "document_type", "status", "holdout_accessed",
    "selection_performed", "bootstrap_performed", "holm_performed",
    "decision_performed", "c8_handoff", "c8_handoff_sha256",
    "c9_execution_plan_sha256", "c9_run_spec_sha256",
    "c9_preflight_sha256", "c9_output_root", "code",
    "canonical_digest",
})
_REPORT_KEYS = frozenset({
    "schema_version", "document_type", "status", "integrity_valid",
    "orchestration_complete", "successful", "expected_unit_count",
    "terminal_unit_count", "state_counts",
    "primary_holdout_valid_count", "primary_holdout_expected_count",
    "primary_holdout_complete", "primary_holdout_failures",
    "bootstrap_performed", "holm_performed", "decision_performed",
    "selection_performed", "rows", "errors", "bindings",
    "post_c9_budget",
    "canonical_digest",
})
_REPORT_BINDING_KEYS = frozenset({
    "output_root", "code", "c8_handoff_sha256",
    "execution_plan_sha256", "run_spec_sha256", "preflight_sha256",
    "authorization_sha256", "control_file_sha256",
})
_POST_C9_BUDGET_KEYS = frozenset({
    "schema_version", "document_type", "status",
    "blocked_reasons", "source_unit_counts",
    "measurement_boundary", "projection", "canonical_digest",
})


def _plain(value: Any) -> Any:
    method = getattr(value, "to_dict", None)
    value = method() if callable(method) else value
    return json.loads(json.dumps(value, allow_nan=False))


def _self_digest(value: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: item for key, item in value.items() if key != "canonical_digest"
    })


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_regular(path: Path, context: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_file():
        raise StageCBenchmarkIntegrityError(
            f"{context} is missing, non-regular, or a symlink"
        )
    return path


def _require_safe_existing_root(output_root: Path | str) -> Path:
    root = Path(os.path.abspath(os.fspath(output_root)))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        if not os.path.lexists(current):
            raise StageCBenchmarkIntegrityError(
                f"C9 output root component is missing: {current}"
            )
        if stat.S_ISLNK(current.lstat().st_mode):
            raise StageCBenchmarkIntegrityError(
                f"C9 output root contains a symlink: {current}"
            )
    if not root.is_dir():
        raise StageCBenchmarkIntegrityError(
            "C9 output root is not a directory"
        )
    return root


def _strict_json(path: Path) -> dict[str, Any]:
    return _c8._strict_json(path)


def _sha(value: Any, context: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str) or len(value) != length
        or set(value) - _SHA256
    ):
        raise StageCBenchmarkIntegrityError(f"{context} is not lowercase hex")
    return value


def _validate_frozen_input_hashes(
    *, config: Any, tracked_selection_path: Path | str,
) -> None:
    """Rehash every frozen C9 launch input without opening image pixels."""

    if config.to_dict() != _strict_json(DEFAULT_STAGE_C_CONFIG):
        raise StageCBenchmarkIntegrityError(
            "C9 config object is not the frozen repository config"
        )
    protocol_path = REPO_ROOT / config.protocol.path
    registry_path = REPO_ROOT / config.registry.path
    metadata_path = REPO_ROOT / config.registry.metadata_path
    source_manifest_path = REPO_ROOT / config.registry.source_manifest_path
    observed = {
        "model_sha256": sha256_file(DEFAULT_STATE_DICT),
        "registry_sha256": sha256_file(registry_path),
        "registry_metadata_sha256": sha256_file(metadata_path),
        "config_sha256": sha256_file(DEFAULT_STAGE_C_CONFIG),
        "config_schema_sha256": sha256_file(DEFAULT_STAGE_C_SCHEMA),
        "registry_schema_sha256": sha256_file(DEFAULT_REGISTRY_SCHEMA),
        "selection_schema_sha256": sha256_file(DEFAULT_SELECTION_SCHEMA),
        "protocol_sha256": sha256_file(protocol_path),
        "acceptance_sha256": sha256_file(DEFAULT_ACCEPTANCE),
        "dependency_lock_sha256": sha256_file(DEFAULT_DEPENDENCY_LOCK),
        "stage_a_manifest_sha256": sha256_file(source_manifest_path),
    }
    expected = dict(FROZEN_INPUT_SHA256)
    if observed != expected:
        drifted = [
            key for key in expected if observed.get(key) != expected[key]
        ]
        raise StageCBenchmarkIntegrityError(
            f"C9 frozen input hash drifted: {', '.join(drifted)}"
        )
    tracked = _strict_json(_safe_regular(
        Path(tracked_selection_path), "tracked C8 selection"
    ))
    selection_hashes = tracked.get("input_hashes")
    if not isinstance(selection_hashes, Mapping) or any(
        selection_hashes.get(key) != value
        for key, value in FROZEN_INPUT_SHA256.items()
        if key not in {
            "registry_schema_sha256", "selection_schema_sha256",
            "stage_a_manifest_sha256",
        }
    ):
        raise StageCBenchmarkIntegrityError(
            "tracked selection input hashes differ from frozen C9 inputs"
        )
    if config.registry.source_manifest_sha256 != FROZEN_INPUT_SHA256[
        "stage_a_manifest_sha256"
    ]:
        raise StageCBenchmarkIntegrityError(
            "Stage C config does not bind the frozen manifest/schema hashes"
        )


def _counts(plan: ExecutionPlan) -> dict[str, int]:
    totals: Counter[str] = Counter()
    totals["samples"] = len({unit.sample.sample_id for unit in plan.units})
    totals["audit_samples"] = len({
        unit.sample.sample_id for unit in plan.units if unit.sample.audit
    })
    for unit in plan.units:
        counts = unit.work_item_counts
        for key in (
            "work_items", "map_runs", "clean_map_runs",
            "perturbation_map_runs", "repeatability_map_runs",
            "fidelity_curves", "fidelity_curve_points",
            "target_tie_curve_points", "random_sanity_curve_points",
            "center_sanity_curve_points", "logical_scalar_evaluations",
        ):
            totals[key] += int(counts[key])
        totals["fidelity_eligible_units"] += int(unit.fidelity_eligible)
        totals["structural_reference_units"] += int(
            unit.structural_reference_only
        )
    totals["execution_units"] = len(plan.units)
    return {key: int(totals[key]) for key in C9_EXPECTED_COUNTS}


def _expected_work_items(
    method_id: str, *, audit: bool, block: int | None, head: int | None,
) -> list[dict[str, Any]]:
    items = [
        WorkItem(MAP_KIND, "clean", 0, None, None, None, None, None,
                 None, None, None)
    ]
    items.extend(
        WorkItem(MAP_KIND, perturbation_id, None, None, None, None, None,
                 None, None, None, None)
        for perturbation_id in PERTURBATION_IDS
    )
    if audit:
        items.append(
            WorkItem(MAP_KIND, "repeat_clean", 1, None, None, None, None,
                     None, None, None, None)
        )
    if method_id != VALUE_METHOD_IDS[0]:
        pairing_block, pairing_head = (
            C9_PRIMARY if block is None else (block, int(head))
        )
        rankings = [
            *((TARGET_RANKING, repeat) for repeat in range(5)),
            *((RANDOM_RANKING, repeat) for repeat in range(5)),
            (CENTER_RANKING, None),
        ]
        items.extend(
            WorkItem(
                CURVE_POINT_KIND, "clean", None, scale, direction, ranking,
                repeat, fraction_index, fraction_index / 20.0,
                pairing_block, pairing_head,
            )
            for scale in FIDELITY_SCALES
            for direction in CURVE_DIRECTIONS
            for ranking, repeat in rankings
            for fraction_index in range(21)
        )
    return [item.to_dict() for item in items]


def validate_c9_execution_plan(plan: ExecutionPlan, config: Any) -> dict[str, Any]:
    """Validate the exact ordered 20-image/four-winner C9 plan."""

    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")
    document = ExecutionPlan.from_dict(plan.to_dict()).to_dict()
    violations: list[str] = []
    expected_spec = {
        "schema_version": 1,
        "document_type": "stage_c_run_spec",
        "sample_ids": list(C9_SAMPLE_IDS),
        "method_ids": list(_ALL_METHODS),
        "value_config_source": "selection_band_winners",
        "value_candidates": [],
        "fidelity_pairing_context": {
            "block_index": C9_PRIMARY[0], "head_index": C9_PRIMARY[1],
        },
        "audit_repeatability_runs": 2,
        "audit_repeatability_scope": "clean_map_only",
        "run_registered_fidelity": True,
        "run_sanity": True,
        "run_stability": True,
    }
    if document["run_spec"] != expected_spec:
        violations.append("C9 RunSpec differs from the exact frozen document")
    selection = document.get("selection", {})
    observed_winners = [
        (item.get("block_index"), item.get("head_index"))
        for item in selection.get("depth_band_winners", [])
        if isinstance(item, Mapping)
    ]
    primary = selection.get("primary")
    observed_primary = (
        primary.get("block_index"), primary.get("head_index")
    ) if isinstance(primary, Mapping) else None
    if (
        selection.get("status") != "present"
        or selection.get("sha256") not in {
            C8_SELECTION_FILE_SHA256, RECOVERY_C8_SELECTION_FILE_SHA256,
        }
        or observed_winners != list(C9_WINNERS)
        or observed_primary != C9_PRIMARY
    ):
        violations.append("C9 selection is not the frozen four-winner selection")
    counts = _counts(plan)
    if counts != C9_EXPECTED_COUNTS:
        violations.append(f"C9 expanded counts mismatch: {counts!r}")
    method_counts = Counter(unit.key.method_id for unit in plan.units)
    expected_method_counts = {
        **{method: 80 for method in VALUE_METHOD_IDS},
        **{method: 20 for method in CLASS_METHOD_IDS},
    }
    if method_counts != Counter(expected_method_counts):
        violations.append("C9 per-method unit counts drifted")
    per_sample = Counter(unit.sample.sample_id for unit in plan.units)
    if per_sample != Counter({item: C9_UNITS_PER_IMAGE for item in C9_SAMPLE_IDS}):
        violations.append("C9 must contain exactly 16 units per frozen image")
    role_by_id = {
        unit.sample.sample_id: unit.sample.role for unit in plan.units
    }
    if (
        {item for item, role in role_by_id.items() if role == "calibration"}
        != set(CALIBRATION_SAMPLE_IDS)
        or {item for item, role in role_by_id.items() if role == "holdout"}
        != set(C9_HOLDOUT_SAMPLE_IDS)
    ):
        violations.append("C9 roles are not the exact 5 calibration + 15 holdout")
    audit_ids = {
        unit.sample.sample_id for unit in plan.units if unit.sample.audit
    }
    if audit_ids != set(AUDIT_SAMPLE_IDS):
        violations.append("C9 audit scope differs from the frozen two calibration IDs")
    expected_order: list[tuple[str, str, int | None, int | None]] = []
    for sample_id in C9_SAMPLE_IDS:
        for method_id in _ALL_METHODS:
            if method_id in VALUE_METHOD_IDS:
                expected_order.extend(
                    (sample_id, method_id, block, head)
                    for block, head in C9_WINNERS
                )
            else:
                expected_order.append((sample_id, method_id, None, None))
    observed_order = [
        (unit.key.sample_id, unit.key.method_id, unit.key.block_index,
         unit.key.head_index)
        for unit in plan.units
    ]
    if observed_order != expected_order:
        violations.append("C9 ordered unit expansion drifted")
    for unit in plan.units:
        expected_counts = _c8._expected_unit_work_counts(
            unit.key.method_id, unit.sample.audit
        )
        if unit.work_item_counts != expected_counts:
            violations.append(
                f"C9 unit work counts drifted: {unit.key.canonical_digest}"
            )
        if [item.to_dict() for item in unit.work_items] != _expected_work_items(
            unit.key.method_id, audit=unit.sample.audit,
            block=unit.key.block_index, head=unit.key.head_index,
        ):
            violations.append(
                f"C9 ordered work items drifted: {unit.key.canonical_digest}"
            )
    if [item.id for item in getattr(config, "methods", ())] != list(_ALL_METHODS):
        violations.append("config method order/set drifted")
    return {
        "valid": not violations, "violations": violations,
        "counts": counts, "sample_ids": list(C9_SAMPLE_IDS),
        "method_ids": list(_ALL_METHODS),
        "winners": [list(item) for item in C9_WINNERS],
        "primary": list(C9_PRIMARY),
        "execution_plan_sha256": plan.canonical_digest,
    }


def _validate_commit_lineage(
    *, repo_root: Path | str | None = None, current_commit: str | None = None,
    git_runner: Any = subprocess.run,
) -> None:
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    try:
        for commit in (
            C8_PRODUCER_COMMIT, C8_GENERATOR_COMMIT, C8_COMPLETION_COMMIT,
        ):
            observed = git_runner(
                ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
                cwd=root, check=True, capture_output=True, text=True,
            ).stdout.strip()
            if observed != commit:
                raise ValueError(f"commit resolution drifted: {commit}")
        for older, newer in (
            (C8_PRODUCER_COMMIT, C8_GENERATOR_COMMIT),
            (C8_GENERATOR_COMMIT, C8_COMPLETION_COMMIT),
        ):
            git_runner(
                ["git", "merge-base", "--is-ancestor", older, newer],
                cwd=root, check=True, capture_output=True, text=True,
            )
        if current_commit is not None:
            git_runner(
                ["git", "merge-base", "--is-ancestor",
                 C8_COMPLETION_COMMIT, current_commit],
                cwd=root, check=True, capture_output=True, text=True,
            )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise StageCBenchmarkIntegrityError(
            f"C8 producer/generator/completion lineage is invalid: {error}"
        ) from error


def _validate_c8_source_controls(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise StageCBenchmarkIntegrityError("C8 source root is missing or unsafe")
    for name, expected in C8_CONTROL_FILE_SHA256.items():
        path = _safe_regular(root / name, f"C8 source control {name}")
        if _file_sha256(path) != expected:
            raise StageCBenchmarkIntegrityError(
                f"C8 source control hash drifted: {name}"
            )
    status = _strict_json(root / "batch_status.json")
    table = _strict_json(root / "failure_table.json")
    rows = table.get("rows")
    if (
        status.get("schema_version") != 1
        or status.get("document_type") != "stage_c_batch_status"
        or status.get("plan_digest") != C8_EXECUTION_PLAN_SHA256
        or status.get("expected_unit_count") != 2180
        or status.get("lifecycle_state") != "finalized"
        or status.get("orchestration_complete") is not True
        or status.get("active_claim_count") != 0
        or status.get("validation_errors") != []
        or status.get("failure_table_sha256")
        != C8_CONTROL_FILE_SHA256["failure_table.json"]
        or not isinstance(rows, list) or len(rows) != 2180
        or any(
            not isinstance(row, Mapping)
            or row.get("state") not in _TERMINAL_STATES
            for row in rows
        )
    ):
        raise StageCBenchmarkIntegrityError(
            "C8 source controls do not prove one complete terminal batch"
        )
    failures = [row for row in rows if row.get("state") == "failed"]
    if len(failures) != 1:
        raise StageCBenchmarkIntegrityError(
            "C8 source must retain its exact one terminal failure"
        )
    failed_key = failures[0].get("unit_key", {})
    if (
        failed_key.get("sample_id") != CALIBRATION_SAMPLE_IDS[-1]
        or failed_key.get("method_id") != VALUE_METHOD_IDS[1]
        or failed_key.get("block_index") != 11
        or failed_key.get("head_index") != 5
    ):
        raise StageCBenchmarkIntegrityError("C8 retained failure identity drifted")


def _validate_c8_source_evidence(
    root: Path, report_path: Path, tracked_selection_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the C8 report from all 2,180 immutable unit attempts.

    This expensive check is deliberately confined to C9 authorization.  It
    reads control/L0 evidence and lets the existing batch validator hash every
    retained artifact, but it never opens a source JPEG or interprets L3 PNGs.
    Shards subsequently revalidate the immutable authorization and compact C8
    controls instead of repeating this scan.
    """

    from .stage_c_config import load_stage_c_config

    try:
        c8_plan = ExecutionPlan.from_dict(_strict_json(root / "execution_plan.json"))
        config = load_stage_c_config()
        evidence = build_c8_calibration_evidence(
            root, c8_plan, config,
            observed_git_state={
                "commit": C8_PRODUCER_COMMIT, "dirty": False,
            },
            c7_report_binding=report["bindings"]["c7_report_binding"],
        )
        validate_stage_c_selection_report(
            report, evidence=evidence,
            input_hashes=report["bindings"]["input_hashes"],
            generator_git_state={
                "commit": C8_GENERATOR_COMMIT, "dirty": False,
            },
        )
        tracked = _strict_json(tracked_selection_path)
    except Exception as error:
        raise StageCBenchmarkIntegrityError(
            f"C8 full source-evidence recomputation failed: {error}"
        ) from error
    if tracked != report.get("selection_document"):
        raise StageCBenchmarkIntegrityError(
            "C8 full source-evidence recomputation changed the PASS report"
        )
    return evidence


def _validate_handoff_document(value: Mapping[str, Any]) -> None:
    if isinstance(value, Mapping) and value.get("document_type") == (
        "stage_c_c8_salvage_repair_to_c9_handoff"
    ):
        from .recovery_c9_launch import validate_recovery_handoff_document

        try:
            validate_recovery_handoff_document(value)
        except Exception as error:
            raise StageCBenchmarkIntegrityError(
                f"recovery C8 handoff is invalid: {error}"
            ) from error
        return
    if not isinstance(value, Mapping) or set(value) != _HANDOFF_KEYS:
        raise StageCBenchmarkIntegrityError("C8 handoff fields drifted")
    if value.get("canonical_digest") != _self_digest(value):
        raise StageCBenchmarkIntegrityError("C8 handoff canonical digest mismatch")
    fixed = {
        "schema_version": 1,
        "document_type": "stage_c_c8_to_c9_handoff",
        "status": "PASS", "integrity_valid": True,
        "c8_report_file_sha256": C8_REPORT_FILE_SHA256,
        "c8_report_canonical_digest": C8_REPORT_CANONICAL_DIGEST,
        "c8_recovery_selection_file_sha256": C8_SELECTION_FILE_SHA256,
        "tracked_selection_file_sha256": C8_SELECTION_FILE_SHA256,
        "selection_canonical_digest": C8_SELECTION_CANONICAL_DIGEST,
        "c8_producer_code": {"commit": C8_PRODUCER_COMMIT, "dirty": False},
        "c8_generator_code": {"commit": C8_GENERATOR_COMMIT, "dirty": False},
        "c8_completion_commit": C8_COMPLETION_COMMIT,
        "c8_control_file_sha256": C8_CONTROL_FILE_SHA256,
        "c8_execution_plan_sha256": C8_EXECUTION_PLAN_SHA256,
        "c8_run_spec_sha256": C8_RUN_SPEC_SHA256,
        "c8_preflight_sha256": C8_PREFLIGHT_SHA256,
        "c8_calibration_artifact_manifest_sha256": (
            C8_CALIBRATION_ARTIFACT_MANIFEST_SHA256
        ),
        "c8_calibration_evidence_sha256": C8_CALIBRATION_EVIDENCE_SHA256,
        "c8_source_tree_total_bytes": C8_SOURCE_TREE_TOTAL_BYTES,
        "c8_retained_artifact_bytes": C8_SOURCE_TREE_TOTAL_BYTES,
        "c9_projected_artifact_bytes": C9_PROJECTED_ARTIFACT_BYTES,
        "c8_c9_projected_artifact_bytes": C8_C9_PROJECTED_ARTIFACT_BYTES,
        "c9_projected_h20_gpu_hours": C9_PROJECTED_H20_GPU_HOURS,
        "c8_c9_projected_h20_gpu_hours": C8_C9_PROJECTED_H20_GPU_HOURS,
    }
    if any(value.get(key) != expected for key, expected in fixed.items()):
        raise StageCBenchmarkIntegrityError("C8 handoff frozen binding drifted")
    for key in (
        "c8_root", "c8_report_path", "c8_recovery_selection_path",
        "tracked_selection_path",
    ):
        path = value.get(key)
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise StageCBenchmarkIntegrityError(f"C8 handoff {key} is not absolute")


def _build_c8_handoff(
    *, c8_root: Path | str, c8_report_path: Path | str,
    recovery_selection_path: Path | str,
    tracked_selection_path: Path | str, repo_root: Path | str | None,
    measure_source_tree: bool,
    source_evidence_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(c8_root)))
    report_path = _safe_regular(Path(c8_report_path), "C8 PASS report")
    recovery_path = _safe_regular(
        Path(recovery_selection_path), "C8 recovery selection"
    )
    tracked_path = _safe_regular(
        Path(tracked_selection_path), "tracked C8 selection"
    )
    if _file_sha256(report_path) != C8_REPORT_FILE_SHA256:
        raise StageCBenchmarkIntegrityError("C8 PASS report byte hash drifted")
    if (
        _file_sha256(recovery_path) != C8_SELECTION_FILE_SHA256
        or _file_sha256(tracked_path) != C8_SELECTION_FILE_SHA256
        or recovery_path.read_bytes() != tracked_path.read_bytes()
    ):
        raise StageCBenchmarkIntegrityError(
            "C8 recovery/tracked selection bytes differ from the frozen file"
        )
    report = _strict_json(report_path)
    recovery = _strict_json(recovery_path)
    tracked = _strict_json(tracked_path)
    try:
        validate_stage_c_selection_report(
            report, generator_git_state={
                "commit": C8_GENERATOR_COMMIT, "dirty": False,
            },
        )
        validate_stage_c_selection_document(tracked)
        validate_stage_c_selection_document(recovery)
    except Exception as error:
        raise StageCBenchmarkIntegrityError(
            f"C8 report/selection semantic validation failed: {error}"
        ) from error
    bindings = report.get("bindings", {})
    budget = report.get("post_c8_budget", {})
    projection = budget.get("projection", {}) if isinstance(budget, Mapping) else {}
    time = projection.get("time", {}) if isinstance(projection, Mapping) else {}
    storage = projection.get("storage", {}) if isinstance(projection, Mapping) else {}
    expected_winners = [
        {
            "candidate_key": f"block_{block:02d}_head_{head:02d}",
            "block_index": block, "head_index": head,
            "depth_band": block // 3,
        }
        for block, head in C9_WINNERS
    ]
    if (
        report.get("canonical_digest") != C8_REPORT_CANONICAL_DIGEST
        or report.get("status") != "PASS"
        or report.get("integrity_valid") is not True
        or report.get("holdout_accessed") is not False
        or any(report.get(flag) is not False for flag in (
            "bootstrap_performed", "holm_performed", "decision_performed",
        ))
        or report.get("depth_band_winners") != expected_winners
        or report.get("primary") != expected_winners[0]
        or report.get("selection_document") != tracked
        or recovery != tracked
        or canonical_sha256(tracked) != C8_SELECTION_CANONICAL_DIGEST
        or report.get("selection_document_sha256")
        != C8_SELECTION_CANONICAL_DIGEST
        or bindings.get("source_root") != str(root)
        or bindings.get("code")
        != {"commit": C8_PRODUCER_COMMIT, "dirty": False}
        or bindings.get("selection_generator_code")
        != {"commit": C8_GENERATOR_COMMIT, "dirty": False}
        or bindings.get("control_file_sha256") != C8_CONTROL_FILE_SHA256
        or bindings.get("execution_plan_sha256") != C8_EXECUTION_PLAN_SHA256
        or bindings.get("run_spec_sha256") != C8_RUN_SPEC_SHA256
        or bindings.get("preflight_sha256") != C8_PREFLIGHT_SHA256
        or bindings.get("calibration_artifact_manifest_sha256")
        != C8_CALIBRATION_ARTIFACT_MANIFEST_SHA256
        or bindings.get("calibration_evidence_sha256")
        != C8_CALIBRATION_EVIDENCE_SHA256
        or bindings.get("source_tree_total_bytes") != C8_SOURCE_TREE_TOTAL_BYTES
        or budget.get("status") != "PASS"
        or budget.get("exact_unit_count") != 2500
        or time.get("slowest_valid_measured_unit_seconds")
        != C8_SLOWEST_UNIT_SECONDS
        or time.get("C9_h20_gpu_hours") != C9_PROJECTED_H20_GPU_HOURS
        or time.get("C8_C9_h20_gpu_hours")
        != C8_C9_PROJECTED_H20_GPU_HOURS
        or storage.get("projected_bytes", {}).get("C8")
        != C8_PROJECTED_ARTIFACT_BYTES
        or storage.get("projected_bytes", {}).get("C9")
        != C9_PROJECTED_ARTIFACT_BYTES
        or storage.get("projected_bytes", {}).get("C8_C9")
        != C8_C9_PROJECTED_ARTIFACT_BYTES
    ):
        raise StageCBenchmarkIntegrityError(
            "C8 PASS report does not contain the exact frozen C9 handoff"
        )
    checks = projection.get("hard_cap_checks")
    if (
        not isinstance(checks, list) or len(checks) != len(HARD_CAP_KEYS)
        or [item.get("cap_key") for item in checks if isinstance(item, Mapping)]
        != list(HARD_CAP_KEYS)
        or any(
            not isinstance(item, Mapping) or item.get("status") != "PASS"
            or item.get("measured") is not True
            or item.get("cap") != FROZEN_CAPS.get(item.get("cap_key"))
            for item in checks
        )
    ):
        raise StageCBenchmarkIntegrityError(
            "C8 handoff lacks ten measured frozen-cap PASS checks"
        )
    _validate_c8_source_controls(root)
    if measure_source_tree:
        measured = measure_artifact_tree_bytes(root).total_bytes
        if measured != C8_SOURCE_TREE_TOTAL_BYTES:
            raise StageCBenchmarkIntegrityError(
                "C8 retained artifact tree byte count drifted"
            )
        source_evidence = _validate_c8_source_evidence(
            root, report_path, tracked_path, report
        )
        if source_evidence_out is not None:
            source_evidence_out.append(source_evidence)
    _validate_commit_lineage(repo_root=repo_root)
    handoff: dict[str, Any] = {
        "schema_version": 1,
        "document_type": "stage_c_c8_to_c9_handoff",
        "status": "PASS", "integrity_valid": True,
        "c8_root": str(root),
        "c8_report_path": str(report_path),
        "c8_report_file_sha256": C8_REPORT_FILE_SHA256,
        "c8_report_canonical_digest": C8_REPORT_CANONICAL_DIGEST,
        "c8_recovery_selection_path": str(recovery_path),
        "c8_recovery_selection_file_sha256": C8_SELECTION_FILE_SHA256,
        "tracked_selection_path": str(tracked_path),
        "tracked_selection_file_sha256": C8_SELECTION_FILE_SHA256,
        "selection_canonical_digest": C8_SELECTION_CANONICAL_DIGEST,
        "c8_producer_code": {"commit": C8_PRODUCER_COMMIT, "dirty": False},
        "c8_generator_code": {"commit": C8_GENERATOR_COMMIT, "dirty": False},
        "c8_completion_commit": C8_COMPLETION_COMMIT,
        "c8_control_file_sha256": dict(C8_CONTROL_FILE_SHA256),
        "c8_execution_plan_sha256": C8_EXECUTION_PLAN_SHA256,
        "c8_run_spec_sha256": C8_RUN_SPEC_SHA256,
        "c8_preflight_sha256": C8_PREFLIGHT_SHA256,
        "c8_calibration_artifact_manifest_sha256": (
            C8_CALIBRATION_ARTIFACT_MANIFEST_SHA256
        ),
        "c8_calibration_evidence_sha256": C8_CALIBRATION_EVIDENCE_SHA256,
        "c8_source_tree_total_bytes": C8_SOURCE_TREE_TOTAL_BYTES,
        "c8_retained_artifact_bytes": C8_SOURCE_TREE_TOTAL_BYTES,
        "c9_projected_artifact_bytes": C9_PROJECTED_ARTIFACT_BYTES,
        "c8_c9_projected_artifact_bytes": C8_C9_PROJECTED_ARTIFACT_BYTES,
        "c9_projected_h20_gpu_hours": C9_PROJECTED_H20_GPU_HOURS,
        "c8_c9_projected_h20_gpu_hours": C8_C9_PROJECTED_H20_GPU_HOURS,
    }
    handoff["canonical_digest"] = _self_digest(handoff)
    _validate_handoff_document(handoff)
    return handoff


def validate_c8_handoff(
    *, c8_root: Path | str, c8_report_path: Path | str,
    recovery_selection_path: Path | str,
    tracked_selection_path: Path | str = DEFAULT_TRACKED_SELECTION,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Recompute the complete, read-only C8-to-C9 launch evidence."""

    return _build_c8_handoff(
        c8_root=c8_root, c8_report_path=c8_report_path,
        recovery_selection_path=recovery_selection_path,
        tracked_selection_path=tracked_selection_path, repo_root=repo_root,
        measure_source_tree=True,
    )


def build_c9_measured_preflight(
    plan: ExecutionPlan, config: Any, c8_handoff: Mapping[str, Any],
) -> StageCEstimate:
    scope = validate_c9_execution_plan(plan, config)
    if not scope["valid"]:
        raise StageCBenchmarkIntegrityError(
            "invalid C9 plan: " + "; ".join(scope["violations"])
        )
    _validate_handoff_document(c8_handoff)
    projected_c9_hours = c8_handoff["c9_projected_h20_gpu_hours"]
    projected_combined_hours = c8_handoff["c8_c9_projected_h20_gpu_hours"]
    projected_stage_d_bytes = c8_handoff.get(
        "stage_d_projected_artifact_bytes", STAGE_D_PROJECTED_ARTIFACT_BYTES
    )
    projected_stage_d_hours = c8_handoff.get(
        "stage_d_projected_h20_gpu_hours", STAGE_D_PROJECTED_H20_GPU_HOURS
    )
    projected_stage_d_per_image = c8_handoff.get(
        "stage_d_projected_h20_gpu_hours_per_image",
        STAGE_D_PROJECTED_H20_GPU_HOURS_PER_IMAGE,
    )
    slowest = c8_handoff.get(
        "measured_slowest_unit_seconds", C8_SLOWEST_UNIT_SECONDS
    )
    peak = c8_handoff.get(
        "measured_peak_reserved_bytes_per_unit", C8_PEAK_RESERVED_BYTES
    )
    if (
        c8_handoff["c8_c9_projected_artifact_bytes"]
        > FROZEN_CAPS["stage_c_c8_c9_total_artifact_bytes_max"]
        or projected_combined_hours
        > FROZEN_CAPS["stage_c_c8_c9_h20_gpu_hours_max"]
        or projected_stage_d_bytes
        > FROZEN_CAPS["stage_d_500_total_artifact_bytes_max"]
        or projected_stage_d_hours
        > FROZEN_CAPS["stage_d_500_h20_gpu_hours_max"]
        or projected_stage_d_per_image
        > FROZEN_CAPS["stage_d_h20_gpu_hours_per_image_max"]
    ):
        raise StageCBenchmarkBlockedError(
            "C8+C9 or Stage D frozen projection exceeds a hard cap"
        )
    estimate = estimate_stage_c_run(
        config, plan,
        measured_slowest_unit_seconds=slowest,
        measured_peak_reserved_bytes_per_unit=peak,
    )
    if (
        dict(estimate.counts) != C9_EXPECTED_COUNTS
        or estimate.compute.get("projected_h20_gpu_hours")
        != projected_c9_hours
        or estimate.compute.get("measured_peak_reserved_bytes_per_unit")
        != peak
        or estimate.storage.get("estimated_all_regular_file_bytes")
        > C9_PROJECTED_ARTIFACT_BYTES
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 measured preflight differs from the frozen C8 projection"
        )
    if not estimate.within_budget:
        raise StageCBenchmarkBlockedError(
            "C9 measured preflight exceeds frozen caps",
            reasons=list(estimate.violations),
        )
    return estimate


def _validate_c9_preflight(
    estimate: StageCEstimate, plan: ExecutionPlan, *, require_issuer: bool,
) -> None:
    if not isinstance(estimate, StageCEstimate):
        raise StageCBenchmarkIntegrityError("C9 preflight is not a StageCEstimate")
    if require_issuer:
        try:
            enforce_stage_c_budget(estimate)
        except Exception as error:
            raise StageCBenchmarkIntegrityError(
                f"C9 preflight is not freshly issuer-authenticated: {error}"
            ) from error
    slowest = estimate.compute.get("measured_slowest_unit_seconds")
    peak = estimate.compute.get("measured_peak_reserved_bytes_per_unit")
    projected = estimate.compute.get("projected_h20_gpu_hours")
    expected_projected = (
        None if not isinstance(slowest, (int, float)) or isinstance(slowest, bool)
        else slowest * C9_EXECUTION_UNITS
        * FROZEN_CAPS["projection_safety_factor"] / 3600.0
    )
    if (
        estimate.plan_digest != plan.canonical_digest
        or dict(estimate.counts) != C9_EXPECTED_COUNTS
        or estimate.within_budget is not True
        or tuple(estimate.violations) != ()
        or dict(estimate.caps.get("effective", {})) != FROZEN_CAPS
        or dict(estimate.caps.get("test_overrides", {})) != {}
        or estimate.caps.get("config_mutated") is not False
        or tuple(estimate.caps.get("unmeasured_caps", ())) != ()
        or not isinstance(slowest, (int, float)) or isinstance(slowest, bool)
        or slowest <= 0
        or type(peak) is not int or peak <= 0
        or projected != expected_projected
        or peak > FROZEN_CAPS["cuda_peak_reserved_bytes_per_execution_unit_max"]
        or estimate.compute.get("maximum_forward_calls_per_execution_unit")
        > FROZEN_CAPS["forward_calls_per_execution_unit_max"]
        or estimate.compute.get("maximum_backward_calls_per_execution_unit")
        > FROZEN_CAPS["backward_calls_per_execution_unit_max"]
        or estimate.storage.get("maximum_l0_l1_bytes_per_execution_unit")
        > FROZEN_CAPS["bytes_per_l0_l1_execution_unit_max"]
        or estimate.storage.get("maximum_l2_bytes_per_audit_sample")
        > FROZEN_CAPS["bytes_per_audit_sample_l2_max"]
        or estimate.storage.get("estimated_all_regular_file_bytes")
        > C9_PROJECTED_ARTIFACT_BYTES
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 preflight is not the exact measured frozen-cap estimate"
        )
    # Cumulative and Stage-D projections belong to the bound C8 handoff and
    # are checked by build/validation of the launch authorization.


def _require_live_git_state(
    expected: Mapping[str, Any] | None = None, *,
    supplied: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    live = _plain(capture_git_state(REPO_ROOT if repo_root is None else repo_root))
    if (
        not isinstance(live, Mapping) or set(live) != {"commit", "dirty"}
        or live.get("dirty") is not False
        or not isinstance(live.get("commit"), str)
        or _COMMIT.fullmatch(live["commit"]) is None
    ):
        raise StageCBenchmarkBlockedError(
            "C9 formal execution requires a clean Git worktree"
        )
    if supplied is not None and _plain(supplied) != live:
        raise StageCBenchmarkIntegrityError(
            "supplied C9 Git state differs from the live repository"
        )
    if expected is not None and _plain(expected) != live:
        raise StageCBenchmarkIntegrityError(
            "live C9 Git state differs from launch authorization"
        )
    return dict(live)


def validate_c9_launch_authorization(
    value: Mapping[str, Any], *, plan: ExecutionPlan, estimate: StageCEstimate,
    output_root: Path | str | None = None,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _AUTHORIZATION_KEYS:
        raise StageCBenchmarkIntegrityError("C9 launch authorization fields drifted")
    if value.get("canonical_digest") != _self_digest(value):
        raise StageCBenchmarkIntegrityError(
            "C9 launch authorization canonical digest mismatch"
        )
    for flag in (
        "holdout_accessed", "selection_performed", "bootstrap_performed",
        "holm_performed", "decision_performed",
    ):
        if value.get(flag) is not False:
            raise StageCBenchmarkIntegrityError(
                f"C9 authorization forbidden phase flag is true: {flag}"
            )
    if (
        value.get("schema_version") != 1
        or value.get("document_type") != "stage_c_c9_launch_authorization"
        or value.get("status") != "PASS"
        or value.get("c9_execution_plan_sha256") != plan.canonical_digest
        or value.get("c9_run_spec_sha256")
        != canonical_sha256(plan.run_spec.to_dict())
        or value.get("c9_preflight_sha256") != estimate.canonical_digest
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 launch authorization frozen binding drifted"
        )
    handoff = value.get("c8_handoff")
    _validate_handoff_document(handoff)
    if value.get("c8_handoff_sha256") != handoff["canonical_digest"]:
        raise StageCBenchmarkIntegrityError("C9 authorization handoff digest drifted")
    if handoff.get("document_type") == (
        "stage_c_c8_salvage_repair_to_c9_handoff"
    ):
        projected = {
            "stage_c_c8_c9_total_artifact_bytes_max": handoff[
                "c8_c9_projected_artifact_bytes"
            ],
            "stage_c_c8_c9_h20_gpu_hours_max": handoff[
                "c8_c9_projected_h20_gpu_hours"
            ],
            "stage_d_500_total_artifact_bytes_max": handoff[
                "stage_d_projected_artifact_bytes"
            ],
            "stage_d_500_h20_gpu_hours_max": handoff[
                "stage_d_projected_h20_gpu_hours"
            ],
            "stage_d_h20_gpu_hours_per_image_max": handoff[
                "stage_d_projected_h20_gpu_hours_per_image"
            ],
        }
        if any(projected[key] > FROZEN_CAPS[key] for key in projected):
            raise StageCBenchmarkIntegrityError(
                "recovery C9 handoff exceeds a frozen hard cap"
            )
    code = value.get("code")
    if (
        not isinstance(code, Mapping) or set(code) != {"commit", "dirty"}
        or code.get("dirty") is not False
        or not isinstance(code.get("commit"), str)
        or _COMMIT.fullmatch(code["commit"]) is None
        or code["commit"] == C8_COMPLETION_COMMIT
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 authorization requires a new clean execution commit"
        )
    root_value = value.get("c9_output_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise StageCBenchmarkIntegrityError("C9 authorization root is not absolute")
    if output_root is not None and Path(root_value) != Path(
        os.path.abspath(os.fspath(output_root))
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 launch authorization output-root binding drifted"
        )


def _require_fresh_root(output_root: Path | str) -> Path:
    root = Path(os.path.abspath(os.fspath(output_root)))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise StageCBenchmarkBlockedError(
                "C9 output root contains a symlink"
            )
    if os.path.lexists(root):
        raise StageCBenchmarkBlockedError(
            "C9 output root must not exist before formal initialization"
        )
    return root


def build_c9_launch_authorization(
    plan: ExecutionPlan, config: Any, *, c8_root: Path | str,
    c8_report_path: Path | str, recovery_selection_path: Path | str,
    output_root: Path | str,
    tracked_selection_path: Path | str = DEFAULT_TRACKED_SELECTION,
    observed_git_state: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> tuple[dict[str, Any], StageCEstimate]:
    _validate_frozen_input_hashes(
        config=config, tracked_selection_path=tracked_selection_path,
    )
    handoff = validate_c8_handoff(
        c8_root=c8_root, c8_report_path=c8_report_path,
        recovery_selection_path=recovery_selection_path,
        tracked_selection_path=tracked_selection_path, repo_root=repo_root,
    )
    estimate = build_c9_measured_preflight(plan, config, handoff)
    code = _require_live_git_state(
        supplied=observed_git_state, repo_root=repo_root
    )
    _validate_commit_lineage(repo_root=repo_root, current_commit=code["commit"])
    root = _require_fresh_root(output_root)
    authorization: dict[str, Any] = {
        "schema_version": 1,
        "document_type": "stage_c_c9_launch_authorization",
        "status": "PASS", "holdout_accessed": False,
        "selection_performed": False, "bootstrap_performed": False,
        "holm_performed": False, "decision_performed": False,
        "c8_handoff": handoff,
        "c8_handoff_sha256": handoff["canonical_digest"],
        "c9_execution_plan_sha256": plan.canonical_digest,
        "c9_run_spec_sha256": canonical_sha256(plan.run_spec.to_dict()),
        "c9_preflight_sha256": estimate.canonical_digest,
        "c9_output_root": str(root), "code": code,
    }
    authorization["canonical_digest"] = _self_digest(authorization)
    validate_c9_launch_authorization(
        authorization, plan=plan, estimate=estimate, output_root=root
    )
    return authorization, estimate


def build_recovery_c9_launch_authorization(
    plan: ExecutionPlan, config: Any, *, recovery_record: Any,
    output_root: Path | str,
    observed_git_state: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> tuple[dict[str, Any], StageCEstimate]:
    from .recovery_c9_launch import (
        RecoveryC9LaunchRecord, build_runtime_handoff,
    )

    if type(recovery_record) is not RecoveryC9LaunchRecord:
        raise StageCBenchmarkIntegrityError(
            "validated recovery C9 launch record required"
        )
    if Path(output_root).absolute() != recovery_record.output_root:
        raise StageCBenchmarkIntegrityError(
            "C9 output root differs from recovery authority"
        )
    handoff = build_runtime_handoff(recovery_record, full_source=True)
    estimate = build_c9_measured_preflight(plan, config, handoff)
    code = _require_live_git_state(
        supplied=observed_git_state, repo_root=repo_root
    )
    if code["commit"] != recovery_record.launch_commit:
        raise StageCBenchmarkIntegrityError(
            "recovery C9 execution must start at the authority-only commit"
        )
    root = _require_fresh_root(output_root)
    authorization: dict[str, Any] = {
        "schema_version": 1,
        "document_type": "stage_c_c9_launch_authorization",
        "status": "PASS", "holdout_accessed": False,
        "selection_performed": False, "bootstrap_performed": False,
        "holm_performed": False, "decision_performed": False,
        "c8_handoff": handoff,
        "c8_handoff_sha256": handoff["canonical_digest"],
        "c9_execution_plan_sha256": plan.canonical_digest,
        "c9_run_spec_sha256": canonical_sha256(plan.run_spec.to_dict()),
        "c9_preflight_sha256": estimate.canonical_digest,
        "c9_output_root": str(root), "code": code,
    }
    authorization["canonical_digest"] = _self_digest(authorization)
    validate_c9_launch_authorization(
        authorization, plan=plan, estimate=estimate, output_root=root
    )
    return authorization, estimate


@contextmanager
def _safe_lock(root: Path, name: str, *, exclusive: bool) -> Iterator[None]:
    path = root / name
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise StageCBenchmarkIntegrityError(
            f"cannot safely open C9 lock {name}: {error}"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise StageCBenchmarkIntegrityError(
                f"C9 lock {name} must be a regular file"
            )
        fcntl.flock(
            descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        )
        descriptor_stat = os.fstat(descriptor)
        path_stat = path.lstat()
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            path_stat.st_dev, path_stat.st_ino
        ):
            raise StageCBenchmarkIntegrityError(
                f"C9 lock {name} changed while being acquired"
            )
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _checkpoint_lock(root: Path) -> Iterator[None]:
    return _safe_lock(root, _CHECKPOINT_LOCK_FILENAME, exclusive=True)


def _lifecycle_lock(root: Path, *, exclusive: bool) -> Iterator[None]:
    return _safe_lock(root, _LIFECYCLE_LOCK_FILENAME, exclusive=exclusive)


def _artifact_activity_lock(root: Path, *, exclusive: bool) -> Iterator[None]:
    return _safe_lock(root, _ARTIFACT_ACTIVITY_LOCK_FILENAME, exclusive=exclusive)


def _pending_rows(plan: ExecutionPlan) -> list[dict[str, Any]]:
    return _c8._pending_rows(plan)


def _failure_table(
    plan: ExecutionPlan, rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _c8._failure_table(plan, rows)


def _batch_status(
    plan: ExecutionPlan, rows: Sequence[Mapping[str, Any]],
    failure_sha256: str, *, validation_errors: Sequence[str] = (),
    lifecycle_state: str = "running",
) -> dict[str, Any]:
    if lifecycle_state not in _LIFECYCLE_STATES:
        raise StageCBenchmarkIntegrityError("invalid C9 lifecycle state")
    terminal = all(row.get("terminal") is True for row in rows)
    finalized = lifecycle_state == "finalized"
    successful = bool(rows) and all(
        row.get("state") == "succeeded" for row in rows
    )
    return {
        "schema_version": 1,
        "document_type": "stage_c_batch_status",
        "plan_digest": plan.canonical_digest,
        "expected_unit_count": len(plan.units),
        "orchestration_complete": finalized and terminal,
        "successful": finalized and successful and not validation_errors,
        "failure_table_sha256": failure_sha256,
        "lifecycle_state": lifecycle_state,
        "active_claim_count": sum(
            row.get("state") == "claimed" for row in rows
        ),
        "validation_errors": list(validation_errors),
    }


def _read_checkpoint_rows(
    root: Path, plan: ExecutionPlan,
) -> list[dict[str, Any]]:
    table = _strict_json(root / "failure_table.json")
    if (
        set(table) != {"schema_version", "document_type", "plan_digest", "rows"}
        or table.get("schema_version") != 1
        or table.get("document_type") != "stage_c_failure_table"
        or table.get("plan_digest") != plan.canonical_digest
        or not isinstance(table.get("rows"), list)
    ):
        raise StageCBenchmarkIntegrityError("C9 failure table header drifted")
    rows = table["rows"]
    expected = [unit.key.canonical_digest for unit in plan.units]
    observed = [
        row.get("unit_key", {}).get("canonical_digest")
        for row in rows if isinstance(row, Mapping)
    ]
    if len(rows) != len(plan.units) or observed != expected:
        raise StageCBenchmarkIntegrityError(
            "C9 failure table must contain every exact ordered plan row"
        )
    normalized = [_plain(row) for row in rows]
    for index, row in enumerate(normalized):
        if set(row) != _ROW_KEYS or row.get("state") not in _CHECKPOINT_STATES:
            raise StageCBenchmarkIntegrityError(
                f"C9 failure row {index} fields/state drifted"
            )
        if row["unit_key"] != plan.units[index].key.to_dict():
            raise StageCBenchmarkIntegrityError(
                f"C9 failure row {index} identity drifted"
            )
        state = row["state"]
        if (
            type(row["terminal"]) is not bool
            or type(row["complete"]) is not bool
            or row["terminal"] is not (state in _TERMINAL_STATES)
            or row["complete"] is not (state == "succeeded")
        ):
            raise StageCBenchmarkIntegrityError(
                f"C9 failure row {index} state flags drifted"
            )
        token = row["claim_token"]
        if state == "pending":
            if any(row[name] is not None for name in (
                "selected_attempt", "selected_status_sha256", "failure",
                "claim_token", "l0_l1_bytes", "l2_bytes", "total_bytes",
            )):
                raise StageCBenchmarkIntegrityError(
                    f"C9 pending row {index} contains execution evidence"
                )
        if state in {"claimed", "succeeded", "failed", "incomplete"} and (
            not isinstance(token, str) or len(token) != 64 or set(token) - _SHA256
        ):
            raise StageCBenchmarkIntegrityError(
                f"C9 non-pending row {index} lacks a valid claim token"
            )
        selected = row["selected_attempt"]
        byte_fields = (row["l0_l1_bytes"], row["l2_bytes"], row["total_bytes"])
        if selected is None:
            if row["selected_status_sha256"] is not None or any(
                item is not None for item in byte_fields
            ):
                raise StageCBenchmarkIntegrityError(
                    f"C9 row {index} has partial attempt evidence"
                )
            if state in _TERMINAL_STATES:
                raise StageCBenchmarkIntegrityError(
                    f"C9 terminal row {index} lacks an attempt"
                )
        else:
            selected_path = Path(selected) if isinstance(selected, str) else Path(".")
            expected_parent = (
                root / "units" / row["unit_key"]["canonical_digest"] / "attempts"
            ).absolute()
            if (
                not isinstance(selected, str) or not selected_path.is_absolute()
                or selected_path.parent != expected_parent
                or _ATTEMPT_NAME.fullmatch(selected_path.name) is None
                or selected_path.is_symlink()
            ):
                raise StageCBenchmarkIntegrityError(
                    f"C9 row {index} selected attempt escapes its unit"
                )
            status_hash = row["selected_status_sha256"]
            if status_hash is not None:
                _sha(status_hash, f"C9 row {index} selected status hash")
            if state in _TERMINAL_STATES and status_hash is None:
                raise StageCBenchmarkIntegrityError(
                    f"C9 terminal row {index} lacks status hash"
                )
            if any(type(item) is not int or item < 0 for item in byte_fields):
                raise StageCBenchmarkIntegrityError(
                    f"C9 row {index} artifact byte evidence is invalid"
                )
            if row["total_bytes"] != row["l0_l1_bytes"] + row["l2_bytes"]:
                raise StageCBenchmarkIntegrityError(
                    f"C9 row {index} artifact byte totals disagree"
                )
    return normalized


def _validate_checkpoint_controls(
    root: Path, plan: ExecutionPlan, *,
    allowed_lifecycle: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_checkpoint_rows(root, plan)
    status = _strict_json(root / "batch_status.json")
    if set(status) != _STATUS_KEYS:
        raise StageCBenchmarkIntegrityError("C9 batch status fields drifted")
    lifecycle = status.get("lifecycle_state")
    if lifecycle not in _LIFECYCLE_STATES:
        raise StageCBenchmarkIntegrityError("C9 lifecycle state is invalid")
    if allowed_lifecycle is not None and lifecycle not in set(allowed_lifecycle):
        raise StageCBenchmarkIntegrityError(
            f"C9 lifecycle {lifecycle!r} forbids this operation"
        )
    errors = status.get("validation_errors")
    if not isinstance(errors, list) or any(
        not isinstance(item, str) or not item for item in errors
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 batch validation_errors are invalid"
        )
    expected = _batch_status(
        plan, rows, _file_sha256(root / "failure_table.json"),
        validation_errors=errors, lifecycle_state=lifecycle,
    )
    if status != expected:
        raise StageCBenchmarkIntegrityError(
            "C9 batch status does not bind the current failure table"
        )
    return rows, status


def _validate_closed_layout(
    root: Path, plan: ExecutionPlan, *, require_all_units: bool,
) -> list[str]:
    errors: list[str] = []
    allowed_root = {
        *_CONTROL_NAMES, FINAL_REPORT_FILENAME, _CHECKPOINT_LOCK_FILENAME,
        _LIFECYCLE_LOCK_FILENAME, _ARTIFACT_ACTIVITY_LOCK_FILENAME, "units",
    }
    for child in root.iterdir():
        if child.name not in allowed_root:
            errors.append(f"unexpected C9 root entry: {child.name}")
        if child.is_symlink():
            errors.append(f"forbidden C9 root symlink: {child.name}")
        elif child.is_file():
            try:
                if child.stat().st_nlink != 1:
                    errors.append(
                        f"C9 root file has multiple hard links: {child.name}"
                    )
            except OSError as error:
                errors.append(f"cannot stat C9 root file {child.name}: {error}")
    units_root = root / "units"
    expected = {unit.key.canonical_digest for unit in plan.units}
    observed: set[str] = set()
    if units_root.exists():
        if units_root.is_symlink() or not units_root.is_dir():
            errors.append("C9 units entry is not a safe directory")
            return errors
        for unit_root in units_root.iterdir():
            if unit_root.is_symlink() or not unit_root.is_dir():
                errors.append(f"unexpected C9 units entry: {unit_root.name}")
                continue
            observed.add(unit_root.name)
            if unit_root.name not in expected:
                errors.append(f"unexpected C9 unit directory: {unit_root.name}")
                continue
            children = list(unit_root.iterdir())
            if len(children) != 1 or children[0].name != "attempts" or (
                children[0].is_symlink() or not children[0].is_dir()
            ):
                errors.append(f"C9 unit layout drifted: {unit_root.name}")
                continue
            attempts = list(children[0].iterdir())
            if len(attempts) != 1:
                errors.append(
                    f"C9 unit {unit_root.name} must have exactly one attempt"
                )
            for attempt in attempts:
                if (
                    attempt.is_symlink() or not attempt.is_dir()
                    or _ATTEMPT_NAME.fullmatch(attempt.name) is None
                ):
                    errors.append(
                        f"unexpected C9 attempt: {unit_root.name}/{attempt.name}"
                    )
                    continue
                for path in attempt.rglob("*"):
                    if path.is_symlink():
                        errors.append(
                            f"forbidden C9 attempt symlink: "
                            f"{path.relative_to(root).as_posix()}"
                        )
                    elif path.is_file():
                        try:
                            if path.stat().st_nlink != 1:
                                errors.append(
                                    "C9 attempt file has multiple hard links: "
                                    f"{path.relative_to(root).as_posix()}"
                                )
                        except OSError as error:
                            errors.append(
                                f"cannot stat C9 attempt file "
                                f"{path.relative_to(root).as_posix()}: {error}"
                            )
    if require_all_units and observed != expected:
        errors.append(
            f"C9 units directory coverage mismatch; missing={len(expected-observed)}"
        )
    return errors


def _revalidate_handoff_from_authorization(
    handoff: Mapping[str, Any], *, repo_root: Path | str | None = None,
    measure_source_tree: bool = False,
    source_evidence_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _validate_handoff_document(handoff)
    if handoff.get("document_type") == (
        "stage_c_c8_salvage_repair_to_c9_handoff"
    ):
        from .recovery_c9_launch import validate_recovery_c8_handoff

        return validate_recovery_c8_handoff(
            handoff, full_source=measure_source_tree,
            evidence_out=source_evidence_out,
        )
    observed = _build_c8_handoff(
        c8_root=handoff["c8_root"],
        c8_report_path=handoff["c8_report_path"],
        recovery_selection_path=handoff["c8_recovery_selection_path"],
        tracked_selection_path=handoff["tracked_selection_path"],
        repo_root=repo_root, measure_source_tree=measure_source_tree,
        source_evidence_out=source_evidence_out,
    )
    if observed != dict(handoff):
        raise StageCBenchmarkIntegrityError("persisted C8 handoff has drifted")
    return observed


def _recovery_runtime_context(authorization: Mapping[str, Any]) -> Any | None:
    handoff = authorization.get("c8_handoff", {})
    if not isinstance(handoff, Mapping) or handoff.get("document_type") != (
        "stage_c_c8_salvage_repair_to_c9_handoff"
    ):
        return None
    from .recovery_c9_launch import load_recovery_c9_launch_record

    authority = handoff["recovery_c9_authority"]
    return load_recovery_c9_launch_record(
        REPO_ROOT / authority["path"], require_tracked=True,
        full_c8_validation=False,
    ).repair_record


def initialize_c9_output_root(
    output_root: Path | str, plan: ExecutionPlan, preflight: StageCEstimate,
    authorization: Mapping[str, Any], *, config: Any,
    observed_git_state: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> Path:
    scope = validate_c9_execution_plan(plan, config)
    if not scope["valid"]:
        raise StageCBenchmarkIntegrityError(
            "invalid C9 plan: " + "; ".join(scope["violations"])
        )
    validate_c9_launch_authorization(
        authorization, plan=plan, estimate=preflight, output_root=output_root
    )
    _validate_c9_preflight(preflight, plan, require_issuer=True)
    live = _require_live_git_state(
        authorization["code"], supplied=observed_git_state,
        repo_root=repo_root,
    )
    if authorization["c8_handoff"].get("document_type") == (
        "stage_c_c8_salvage_repair_to_c9_handoff"
    ):
        from .recovery_c9_launch import load_recovery_c9_launch_record

        authority_path = authorization["c8_handoff"][
            "recovery_c9_authority"
        ]["path"]
        load_recovery_c9_launch_record(
            Path(repo_root or REPO_ROOT) / authority_path,
            repo_root=repo_root or REPO_ROOT, require_tracked=True,
        )
    else:
        _validate_commit_lineage(repo_root=repo_root, current_commit=live["commit"])
    # Resolve freshness before the comparatively expensive C8 source-tree
    # measurement.  This is still read-only and occurs before mkdir/writes.
    root = _require_fresh_root(output_root)
    _revalidate_handoff_from_authorization(
        authorization["c8_handoff"], repo_root=repo_root,
        measure_source_tree=True,
    )
    try:
        # Claim the exact formal-root pathname atomically after every launch
        # gate has passed.  An empty directory created in the gap is still a
        # competing owner and must never be adopted.
        root.mkdir(parents=False, exist_ok=False)
    except FileExistsError as error:
        raise StageCBenchmarkBlockedError(
            "C9 output root was claimed during formal initialization"
        ) from error
    rows = _pending_rows(plan)
    table = _failure_table(plan, rows)
    try:
        _c8._atomic_create_json(root / "execution_plan.json", plan.to_dict())
        _c8._atomic_create_json(root / "preflight.json", preflight.to_dict())
        _c8._atomic_create_json(root / AUTHORIZATION_FILENAME, authorization)
        _c8._atomic_create_json(root / "failure_table.json", table)
        _c8._atomic_create_json(
            root / "batch_status.json",
            _batch_status(
                plan, rows, _file_sha256(root / "failure_table.json")
            ),
        )
    except BaseException:
        # Never delete a partially initialized formal root; it is durable
        # failure evidence and cannot be retried in place.
        raise
    return root


def load_c9_initialized_state(
    output_root: Path | str, plan: ExecutionPlan, *,
    observed_git_state: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None, require_checkpoint: bool = True,
) -> tuple[StageCEstimate, dict[str, Any]]:
    root = _require_safe_existing_root(output_root)
    for name in _CONTROL_NAMES:
        _safe_regular(root / name, f"C9 initialized control {name}")
    if _strict_json(root / "execution_plan.json") != plan.to_dict():
        raise StageCBenchmarkIntegrityError("C9 persisted execution plan drifted")
    estimate = StageCEstimate.from_dict(_strict_json(root / "preflight.json"))
    _validate_c9_preflight(estimate, plan, require_issuer=False)
    authorization = _strict_json(root / AUTHORIZATION_FILENAME)
    validate_c9_launch_authorization(
        authorization, plan=plan, estimate=estimate, output_root=root
    )
    live = _require_live_git_state(
        authorization["code"], supplied=observed_git_state, repo_root=repo_root
    )
    if authorization["c8_handoff"].get("document_type") != (
        "stage_c_c8_salvage_repair_to_c9_handoff"
    ):
        _validate_commit_lineage(repo_root=repo_root, current_commit=live["commit"])
    _revalidate_handoff_from_authorization(
        authorization["c8_handoff"], repo_root=repo_root,
        measure_source_tree=False,
    )
    if require_checkpoint:
        with _checkpoint_lock(root):
            _validate_checkpoint_controls(root, plan)
    return estimate, authorization


# Compatibility name for focused tests that treat the C8 and C9 measured
# preflight validators uniformly.
_validate_formal_preflight = _validate_c9_preflight


def _claim_c9_unit(root: Path, plan: ExecutionPlan, unit: Any) -> str:
    from .stage_c_artifacts import attempt_directories

    digest = unit.key.canonical_digest
    with _checkpoint_lock(root):
        rows, _status = _validate_checkpoint_controls(
            root, plan, allowed_lifecycle=("running",)
        )
        indexes = {
            row["unit_key"]["canonical_digest"]: index
            for index, row in enumerate(rows)
        }
        if digest not in indexes:
            raise StageCBenchmarkIntegrityError(
                "claimed unit is not in the C9 plan"
            )
        index = indexes[digest]
        previous = rows[index]
        if previous["state"] != "pending":
            raise StageCBenchmarkIntegrityError(
                f"C9 unit {digest} cannot be claimed from "
                f"{previous['state']!r}"
            )
        if attempt_directories(root / "units" / digest / "attempts"):
            raise StageCBenchmarkIntegrityError(
                f"C9 unit {digest} has an attempt before claim"
            )
        token = secrets.token_hex(32)
        rows[index] = {**previous, "state": "claimed", "claim_token": token}
        table = _failure_table(plan, rows)
        _c8._atomic_replace_json(root / "failure_table.json", table)
        _c8._atomic_replace_json(
            root / "batch_status.json",
            _batch_status(
                plan, rows, _file_sha256(root / "failure_table.json")
            ),
        )
        return token


def _combined_artifact_errors(
    root: Path, handoff: Mapping[str, Any], caps: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        c9_bytes = measure_artifact_tree_bytes(root).total_bytes
    except Exception as error:
        return [f"cannot measure C9 artifact tree: {error}"]
    c8_bytes = handoff.get("c8_retained_artifact_bytes")
    if type(c8_bytes) is not int or c8_bytes < 0:
        return ["C8 retained artifact byte binding is invalid"]
    combined = c8_bytes + c9_bytes
    cap = caps.get("stage_c_c8_c9_total_artifact_bytes_max")
    if type(cap) is not int or cap <= 0:
        errors.append("C8+C9 artifact cap is invalid")
    elif combined > cap:
        errors.append(
            f"C8+C9 artifact bytes {combined} exceed cap {cap}"
        )
    return errors


def _publish_checkpoint(
    root: Path, plan: ExecutionPlan, row: Mapping[str, Any], *,
    caps: Mapping[str, Any], claim_token: str, handoff: Mapping[str, Any],
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
            raise StageCBenchmarkIntegrityError(
                "checkpoint row is not in the C9 plan"
            )
        previous = rows[indexes[digest]]
        if (
            previous.get("state") != "claimed"
            or previous.get("claim_token") != claim_token
            or row.get("claim_token") != claim_token
        ):
            raise StageCBenchmarkIntegrityError(
                f"C9 unit {digest} checkpoint claim/state mismatch"
            )
        rows[indexes[digest]] = _plain(row)
        completed = rows[indexes[digest]]
        cap_errors: list[str] = []
        if (
            completed.get("l0_l1_bytes") is not None
            and completed["l0_l1_bytes"]
            > caps["bytes_per_l0_l1_execution_unit_max"]
        ):
            cap_errors.append(f"{digest}: per-unit L0/L1 cap exceeded")
        sample_id = completed.get("unit_key", {}).get("sample_id")
        if sample_id not in AUDIT_SAMPLE_IDS and (completed.get("l2_bytes") or 0):
            cap_errors.append(f"{digest}: non-audit unit retained L2")
        l2_total = sum(
            item.get("l2_bytes") or 0 for item in rows
            if item.get("unit_key", {}).get("sample_id") == sample_id
        )
        if (
            sample_id in AUDIT_SAMPLE_IDS
            and l2_total > caps["bytes_per_audit_sample_l2_max"]
        ):
            cap_errors.append(
                f"audit sample {sample_id} L2 bytes {l2_total} exceed cap "
                f"{caps['bytes_per_audit_sample_l2_max']}"
            )
        with _artifact_activity_lock(root, exclusive=True):
            cap_errors.extend(_combined_artifact_errors(root, handoff, caps))
        if cap_errors:
            failed = rows[indexes[digest]]
            failed["state"] = "tampered"
            failed["terminal"] = False
            failed["complete"] = False
            failed["failure"] = {
                "reason": "online_cap_or_integrity_failure",
                "errors": cap_errors,
            }
            completed = failed
        retained = list(dict.fromkeys([
            *checkpoint_status["validation_errors"], *cap_errors,
        ]))
        # A validated failed attempt is a truthful terminal scientific result
        # and does not prevent the remaining distinct units from running.
        # Incomplete/tampered evidence invalidates this root and stops further
        # claims; repair always starts from another empty root.
        root_failure = completed.get("state") in {"incomplete", "tampered"}
        table = _failure_table(plan, rows)
        _c8._atomic_replace_json(root / "failure_table.json", table)
        _c8._atomic_replace_json(
            root / "batch_status.json",
            _batch_status(
                plan, rows, _file_sha256(root / "failure_table.json"),
                validation_errors=retained,
                lifecycle_state=(
                    "blocked" if checkpoint_status["lifecycle_state"] == "blocked"
                    or retained or root_failure else "running"
                ),
            ),
        )
        if cap_errors:
            raise StageCBenchmarkBlockedError(
                "C9 online artifact cap/integrity check failed",
                reasons=cap_errors,
            )


def _shard_units(
    plan: ExecutionPlan, *, shard_index: int, shard_count: int,
) -> tuple[Any, ...]:
    if (
        type(shard_index) is not int or type(shard_count) is not int
        or shard_count < 1 or shard_count > len(plan.units)
        or shard_index < 0 or shard_index >= shard_count
    ):
        raise ValueError(
            "shard_index/shard_count must select a non-empty zero-based shard"
        )
    return tuple(
        unit for index, unit in enumerate(plan.units)
        if index % shard_count == shard_index
    )


def run_c9_shard(
    plan: ExecutionPlan, *, output_root: Path | str, backend: Any,
    estimate: StageCEstimate, shard_index: int, shard_count: int,
    device: str = "cpu", caps: Mapping[str, Any] | None = None,
    observed_git_state: Mapping[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Run one deterministic round-robin shard with no caller unit filter."""

    if not isinstance(plan, ExecutionPlan) or not isinstance(estimate, StageCEstimate):
        raise TypeError("plan and estimate must be formal Stage C objects")
    selected = _shard_units(
        plan, shard_index=shard_index, shard_count=shard_count
    )
    root = _require_safe_existing_root(output_root)
    persisted_estimate, authorization = load_c9_initialized_state(
        root, plan, observed_git_state=observed_git_state,
        require_checkpoint=False,
    )
    if persisted_estimate.to_dict() != estimate.to_dict():
        raise StageCBenchmarkIntegrityError(
            "supplied C9 preflight differs from initialized root"
        )
    effective_caps = dict(estimate.caps["effective"])
    if caps is not None and dict(caps) != effective_caps:
        raise StageCBenchmarkIntegrityError(
            "C9 shard caps must equal measured preflight caps"
        )
    live = _plain(authorization["code"])
    handoff = authorization["c8_handoff"]
    results: list[Any] = []
    with _lifecycle_lock(root, exclusive=False):
        with _checkpoint_lock(root):
            rows, _status = _validate_checkpoint_controls(
                root, plan, allowed_lifecycle=("running",)
            )
            by_digest = {
                row["unit_key"]["canonical_digest"]: row for row in rows
            }
            unavailable = [
                unit.key.canonical_digest for unit in selected
                if by_digest[unit.key.canonical_digest]["state"] != "pending"
            ]
            if unavailable:
                raise StageCBenchmarkIntegrityError(
                    "C9 shard includes units already claimed or attempted: "
                    + ", ".join(unavailable)
                )
        for unit in selected:
            token = _claim_c9_unit(root, plan, unit)
            try:
                with _artifact_activity_lock(root, exclusive=False):
                    result = run_execution_unit(
                        unit, output_root=root, backend=backend, device=device,
                        resume=False, caps=effective_caps,
                        preflight=estimate.to_dict(), plan=plan,
                        observed_git_state=live,
                    )
            except (KeyboardInterrupt, SystemExit) as error:
                row = _c8._exception_row(
                    unit, error, claim_token=token, caps=effective_caps,
                    root=root, plan=plan, estimate=estimate,
                    observed_git_state=live,
                    trusted_recovery_context=getattr(
                        backend, "trusted_recovery_context", None
                    ),
                )
                try:
                    _publish_checkpoint(
                        root, plan, row, caps=effective_caps,
                        claim_token=token, handoff=handoff,
                    )
                except BaseException as checkpoint_error:
                    if hasattr(error, "add_note"):
                        error.add_note(
                            "C9 interrupt checkpoint also failed: "
                            f"{type(checkpoint_error).__module__}."
                            f"{type(checkpoint_error).__qualname__}: "
                            f"{checkpoint_error}"
                        )
                raise
            except Exception as error:
                row = _c8._exception_row(
                    unit, error, claim_token=token, caps=effective_caps,
                    root=root, plan=plan, estimate=estimate,
                    observed_git_state=live,
                    trusted_recovery_context=getattr(
                        backend, "trusted_recovery_context", None
                    ),
                )
                _publish_checkpoint(
                    root, plan, row, caps=effective_caps, claim_token=token,
                    handoff=handoff,
                )
                results.append({
                    "unit_key": unit.key.to_dict(), "state": "incomplete",
                    "attempt_path": None, "resumed": False,
                    "failure": row["failure"],
                })
                raise StageCBenchmarkBlockedError(
                    "C9 root became incomplete; remaining work requires a "
                    "new empty root",
                    reasons=(f"{unit.key.canonical_digest}: incomplete",),
                ) from error
            row = _c8._result_row(
                unit, result, claim_token=token, caps=effective_caps,
                root=root, plan=plan, estimate=estimate,
                observed_git_state=live,
                trusted_recovery_context=getattr(
                    backend, "trusted_recovery_context", None
                ),
            )
            _publish_checkpoint(
                root, plan, row, caps=effective_caps, claim_token=token,
                handoff=handoff,
            )
            results.append(result)
            if row["state"] in {"incomplete", "tampered"}:
                raise StageCBenchmarkBlockedError(
                    "C9 root became non-resumable; remaining work requires "
                    "a new empty root",
                    reasons=(
                        f"{unit.key.canonical_digest}: {row['state']}",
                    ),
                )
    return tuple(results)


def _scan_rows(
    root: Path, plan: ExecutionPlan, estimate: StageCEstimate, *,
    observed_git_state: Mapping[str, Any],
    claim_tokens: Mapping[str, str | None] | None = None,
    trusted_recovery_context: Any | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    from .stage_c_artifacts import attempt_directories, validate_execution_unit

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
    caps = dict(estimate.caps["effective"])
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
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
            errors.append(
                f"{digest}: expected exactly one attempt, found {len(attempts)}"
            )
            state = "missing" if not attempts else "tampered"
        else:
            selected = str(attempts[0])
            status_path = attempts[0] / "status.json"
            if status_path.is_file() and not status_path.is_symlink():
                status_hash = _file_sha256(status_path)
            checked = validate_execution_unit(
                attempts[0], expected_unit_identity=unit.key.to_dict(),
                expected_provenance_bindings=expected_bindings,
                observed_git_state=observed_git_state, caps=caps,
                require_runner_telemetry=True,
                trusted_recovery_context=trusted_recovery_context,
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
            "unit_key": unit.key.to_dict(), "state": state,
            "terminal": state in _TERMINAL_STATES,
            "complete": state == "succeeded",
            "selected_attempt": selected,
            "selected_status_sha256": status_hash, "failure": failure,
            "claim_token": (
                None if claim_tokens is None else claim_tokens.get(digest)
            ),
            "l0_l1_bytes": l0_l1_bytes, "l2_bytes": l2_bytes,
            "total_bytes": total_bytes,
        })
    return rows, errors


def _read_success_evidence(
    row: Mapping[str, Any], *, unit: Any, observed_git_state: Mapping[str, Any],
    caps: Mapping[str, Any], trusted_recovery_context: Any | None = None,
) -> dict[str, Any]:
    from .stage_c_artifacts import validate_execution_unit

    selected = row.get("selected_attempt")
    if not isinstance(selected, str):
        raise StageCBenchmarkIntegrityError("successful C9 row lacks attempt path")
    attempt = Path(selected)
    checked = validate_execution_unit(
        attempt, expected_unit_identity=unit.key.to_dict(),
        observed_git_state=observed_git_state, caps=caps,
        require_runner_telemetry=True,
        trusted_recovery_context=trusted_recovery_context,
    )
    if not checked.resumable:
        raise StageCBenchmarkIntegrityError(
            f"C9 successful attempt is not fully valid: {unit.key.canonical_digest}"
        )
    result = _strict_json(attempt / "L0/result.json")
    provenance = _strict_json(attempt / "L0/provenance.json")
    manifest = _strict_json(attempt / "manifest.json")
    scalars = result.get("scalars")
    if not isinstance(scalars, Mapping):
        raise StageCBenchmarkIntegrityError("C9 successful result lacks scalars")
    return {
        "plan_index": None, "unit_key": unit.key.to_dict(),
        "state": "succeeded", "trusted_by_batch_validator": True,
        "attempt_path": str(attempt),
        "hashes": {
            "manifest_sha256": _file_sha256(attempt / "manifest.json"),
            "status_sha256": _file_sha256(attempt / "status.json"),
            "selected_status_sha256": row["selected_status_sha256"],
            "provenance_sha256": _file_sha256(attempt / "L0/provenance.json"),
            "result_sha256": _file_sha256(attempt / "L0/result.json"),
            "unit_identity_sha256": manifest.get("unit_identity_sha256"),
        },
        "scalars": _plain(scalars), "provenance": provenance,
        "resources": _plain(result.get("resources", {})),
        "runner_telemetry": _plain(
            _strict_json(attempt / "status.json").get("runner_telemetry", {})
        ),
        "artifact_inventory": _plain(manifest.get("artifacts", [])),
        "work_item_counts": _plain(unit.work_item_counts),
        "audit": unit.sample.audit, "validation_errors": [],
        "l0_l1_cap_accounting_bytes": checked.l0_l1_bytes,
        "level_bytes": {
            "L0": sum(
                item["bytes"] for item in manifest.get("artifacts", [])
                if item.get("level") == "L0"
            ),
            "L1": sum(
                item["bytes"] for item in manifest.get("artifacts", [])
                if item.get("level") == "L1"
            ),
            "L2": checked.l2_bytes,
            "L3": sum(
                item["bytes"] for item in manifest.get("artifacts", [])
                if item.get("level") == "L3"
            ),
            "control": checked.l0_l1_bytes - sum(
                item["bytes"] for item in manifest.get("artifacts", [])
                if item.get("level") in {"L0", "L1", "L3"}
            ),
        },
        "total_regular_file_bytes": checked.total_bytes,
    }


def _shared_input_errors(
    evidence: Sequence[Mapping[str, Any]],
    *, expected_success_by_sample: Mapping[str, int] | None = None,
) -> list[str]:
    """Prove one canonical/perturbation tensor set and frozen targets per image."""

    errors: list[str] = []
    by_sample: dict[str, list[Mapping[str, Any]]] = {}
    for record in evidence:
        by_sample.setdefault(record["unit_key"]["sample_id"], []).append(record)
    for sample_id in C9_SAMPLE_IDS:
        records = by_sample.get(sample_id, [])
        expected_successes = (
            C9_UNITS_PER_IMAGE if expected_success_by_sample is None
            else expected_success_by_sample.get(sample_id, 0)
        )
        if len(records) != expected_successes:
            errors.append(
                f"{sample_id}: shared evidence covers {len(records)} of "
                f"{expected_successes} successful units"
            )
            continue
        # A valid terminal failure is evidence, not corruption.  It makes C9
        # unsuccessful/BLOCKED, while shared-tensor equality is checked over
        # the full sixteen-unit set whenever that sample fully succeeded.
        if expected_successes != C9_UNITS_PER_IMAGE:
            continue
        clean_values = [
            record["scalars"].get("clean_tensor_hashes")
            for record in records
        ]
        clean = {json.dumps(value, sort_keys=True) for value in clean_values}
        if (
            len(clean) != 1
            or any(
                not isinstance(value, Mapping)
                or set(value) != {
                    "canonical_rgb_sha256", "normalized_tensor_sha256",
                }
                or any(
                    not isinstance(digest, str) or len(digest) != 64
                    or set(digest) - _SHA256
                    for digest in value.values()
                )
                for value in clean_values
            )
        ):
            errors.append(f"{sample_id}: canonical clean tensor hashes are not shared")
        perturbation_values = [
            [
                {
                    "perturbation_id": item.get("perturbation_id"),
                    "tensor_sha256": item.get("tensor_sha256"),
                    "normalized_tensor_sha256": item.get("normalized_tensor_sha256"),
                }
                for item in record["scalars"].get("perturbations", [])
                if isinstance(item, Mapping)
            ]
            for record in records
        ]
        perturbations = {
            json.dumps(value, sort_keys=True) for value in perturbation_values
        }
        if (
            len(perturbations) != 1
            or any(
                [item.get("perturbation_id") for item in value]
                != list(PERTURBATION_IDS)
                or any(
                    not isinstance(item.get(name), str)
                    or len(item[name]) != 64 or set(item[name]) - _SHA256
                    for item in value
                    for name in ("tensor_sha256", "normalized_tensor_sha256")
                )
                for value in perturbation_values
            )
        ):
            errors.append(f"{sample_id}: perturbation tensor hashes are not shared")
        class_targets = set()
        for record in records:
            if record["unit_key"]["method_id"] not in CLASS_METHOD_IDS:
                continue
            identity = record["scalars"].get("target_identity")
            try:
                # runtime_model_binding is a process-local mutation guard; the
                # scientific class target must otherwise be identical.
                from .stage_c_selection import _scientific_target_identity

                identity = _scientific_target_identity(identity)
            except Exception:
                identity = None
            class_targets.add(json.dumps(identity, sort_keys=True))
        if len(class_targets) != 1 or "null" in class_targets:
            errors.append(f"{sample_id}: class target is not frozen across methods")
        class_target = None
        class_prediction = None
        if len(class_targets) == 1 and "null" not in class_targets:
            class_target = json.loads(next(iter(class_targets)))
            class_predictions = {
                json.dumps(record["scalars"].get("prediction"), sort_keys=True)
                for record in records
                if record["unit_key"]["method_id"] in CLASS_METHOD_IDS
            }
            if len(class_predictions) == 1 and "null" not in class_predictions:
                class_prediction = json.loads(next(iter(class_predictions)))
            else:
                errors.append(
                    f"{sample_id}: class prediction context is not shared"
                )
        value_targets: dict[tuple[int, int], set[str]] = {}
        for record in records:
            if record["unit_key"]["method_id"] not in VALUE_METHOD_IDS[1:]:
                continue
            pair = (
                record["unit_key"]["block_index"],
                record["unit_key"]["head_index"],
            )
            identity = record["scalars"].get("target_identity")
            try:
                from .stage_c_selection import _scientific_target_identity

                identity = _scientific_target_identity(identity)
            except Exception:
                identity = None
            value_targets.setdefault(pair, set()).add(
                json.dumps(identity, sort_keys=True)
            )
        if set(value_targets) != set(C9_WINNERS):
            errors.append(f"{sample_id}: Value target winner coverage drifted")
        for block, head in C9_WINNERS:
            targets = value_targets.get((block, head), set())
            if len(targets) != 1 or "null" in targets:
                errors.append(
                    f"{sample_id}: Value target is not frozen for "
                    f"block {block} head {head}"
                )
            elif class_target is not None:
                value_target = json.loads(next(iter(targets)))
                if (
                    value_target.get("clean_input_sha256")
                    != class_target.get("clean_input_sha256")
                ):
                    errors.append(
                        f"{sample_id}: class/Value clean model target context "
                        f"differs for block {block} head {head}"
                    )
        if class_prediction is not None:
            for record in records:
                prediction = record["scalars"].get("prediction")
                if prediction != class_prediction:
                    errors.append(
                        f"{sample_id}: clean prediction/ground truth context "
                        "is not shared across all 16 units"
                    )
                    break
        for record in records:
            errors.extend(_record_validity(
                record,
                f"{sample_id}:{record['unit_key']['method_id']}:"
                f"{record['unit_key']['config_id']}",
            ))
            expected_repeats = 1 if sample_id in AUDIT_SAMPLE_IDS else 0
            repeats = record["scalars"].get("repeatability")
            if (
                not isinstance(repeats, list)
                or len(repeats) != expected_repeats
                or any(
                    not isinstance(item, Mapping)
                    or item.get("repeat_index") != 1
                    or item.get("atol") != 1e-6
                    or item.get("rtol") != 1e-5
                    or item.get("pass") is not True
                    for item in repeats
                )
            ):
                errors.append(
                    f"{sample_id}:{record['unit_key']['method_id']}:"
                    "repeatability evidence drifted"
                )
        for record in records:
            for artifact in record["artifact_inventory"]:
                if (
                    artifact.get("level") == "L2"
                    and sample_id not in AUDIT_SAMPLE_IDS
                ):
                    errors.append(f"{sample_id}: non-audit unit retained L2")
                if artifact.get("level") == "L3" and (
                    artifact.get("display_only") is not True
                    or artifact.get("metric_input") is not False
                ):
                    errors.append(f"{sample_id}: L3 entered the quantitative path")
    return errors


def _shared_errors_by_sample(errors: Sequence[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for error in errors:
        sample_id = next(
            (item for item in C9_SAMPLE_IDS if error.startswith(f"{item}:")),
            None,
        )
        if sample_id is not None:
            result.setdefault(sample_id, []).append(error[len(sample_id) + 1:].strip())
    return result


def _primary_holdout_validity(
    rows: Sequence[Mapping[str, Any]], plan: ExecutionPlan, root: Path,
    observed_git_state: Mapping[str, Any], caps: Mapping[str, Any],
    trusted_recovery_context: Any | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Validate, but never aggregate, primary evidence for all 15 holdouts."""

    by_digest = {
        row["unit_key"]["canonical_digest"]: row for row in rows
    }
    failures: list[dict[str, Any]] = []
    valid_ids: list[str] = []
    successful_evidence: dict[str, dict[str, Any]] = {}
    shared_read_errors: list[str] = []
    for index, unit in enumerate(plan.units):
        row = by_digest.get(unit.key.canonical_digest)
        if row is None or row.get("state") != "succeeded":
            continue
        try:
            record = _read_success_evidence(
                row, unit=unit, observed_git_state=observed_git_state, caps=caps
                , trusted_recovery_context=trusted_recovery_context
            )
            record["plan_index"] = index
            successful_evidence[unit.key.canonical_digest] = record
        except Exception as error:
            shared_read_errors.append(
                f"{unit.key.sample_id}:{unit.key.canonical_digest}: "
                f"shared-input evidence: {error}"
            )
    units_by_sample: dict[str, list[Any]] = {item: [] for item in C9_HOLDOUT_SAMPLE_IDS}
    for unit in plan.units:
        if (
            unit.sample.sample_id in units_by_sample
            and unit.key.method_id in VALUE_METHOD_IDS
            and (unit.key.block_index, unit.key.head_index) == C9_PRIMARY
        ):
            units_by_sample[unit.sample.sample_id].append(unit)
    for sample_id in C9_HOLDOUT_SAMPLE_IDS:
        units = units_by_sample[sample_id]
        sample_records: list[dict[str, Any]] = []
        reasons: list[str] = []
        if len(units) != len(VALUE_METHOD_IDS):
            reasons.append("primary_unit_coverage_invalid")
        for unit in units:
            row = by_digest.get(unit.key.canonical_digest)
            if row is None or row.get("state") != "succeeded":
                reasons.append(f"{unit.key.method_id}:unit_not_succeeded")
                continue
            try:
                record = successful_evidence[unit.key.canonical_digest]
                sample_records.append(record)
            except Exception as error:
                reasons.append(f"{unit.key.method_id}:{error}")
        if not reasons and len(sample_records) == len(VALUE_METHOD_IDS):
            try:
                _summary, scientific_failures = _sample_candidate_metrics(
                    {
                        record["unit_key"]["method_id"]: record
                        for record in sample_records
                    }, sample_id, *C9_PRIMARY,
                )
                reasons.extend(reason for _sample, reason in scientific_failures)
            except Exception as error:
                reasons.append(f"primary_validity:{error}")
        if reasons:
            failures.append({
                "sample_id": sample_id,
                "reasons": list(dict.fromkeys(reasons)),
            })
        else:
            valid_ids.append(sample_id)
    # Shared clean/perturbation/class-target identity is a contract across all
    # 16 successful units for an image, not merely the three primary units.
    shared_evidence = list(successful_evidence.values())
    expected_success_by_sample = Counter(
        row["unit_key"]["sample_id"]
        for row in rows if row.get("state") == "succeeded"
    )
    shared_errors = [
        *shared_read_errors,
        *_shared_input_errors(
            shared_evidence,
            expected_success_by_sample=expected_success_by_sample,
        ),
    ]
    if len(shared_evidence) != sum(row.get("state") == "succeeded" for row in rows):
        shared_errors.append("successful unit shared-input evidence coverage drifted")
    shared_by_sample = _shared_errors_by_sample(shared_errors)
    if shared_by_sample:
        by_sample_failure = {item["sample_id"]: item for item in failures}
        for sample_id in C9_HOLDOUT_SAMPLE_IDS:
            reasons = shared_by_sample.get(sample_id)
            if not reasons:
                continue
            if sample_id in valid_ids:
                valid_ids.remove(sample_id)
            if sample_id in by_sample_failure:
                current = by_sample_failure[sample_id]["reasons"]
                current.extend(item for item in reasons if item not in current)
            else:
                failure = {
                    "sample_id": sample_id,
                    "reasons": list(dict.fromkeys(reasons)),
                }
                failures.append(failure)
                by_sample_failure[sample_id] = failure
        failures.sort(key=lambda item: C9_HOLDOUT_SAMPLE_IDS.index(item["sample_id"]))
    return valid_ids, failures, shared_evidence, shared_errors


def _blocked_post_c9_budget(reason: str) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": 1,
        "document_type": "stage_c_c9_post_run_budget",
        "status": "BLOCKED",
        "blocked_reasons": [reason],
        "source_unit_counts": {
            "c8_expected": 2180, "c8_succeeded": None,
            "c9_expected": C9_EXECUTION_UNITS, "c9_succeeded": None,
            "budget_measurement_expected": 2499,
            "budget_measurement_usable": None,
        },
        "measurement_boundary": None,
        "projection": None,
    }
    document["canonical_digest"] = _self_digest(document)
    return document


def _build_post_c9_budget(
    root: Path, plan: ExecutionPlan, rows: Sequence[Mapping[str, Any]],
    c9_evidence: Sequence[Mapping[str, Any]],
    c8_evidence: Mapping[str, Any], caps: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute all ten caps from validated C8 and C9 unit evidence."""

    if dict(caps) != FROZEN_CAPS:
        raise StageCBenchmarkIntegrityError(
            "post-C9 budget caps differ from the frozen caps"
        )
    c8_records = c8_evidence.get("records")
    if not isinstance(c8_records, list) or len(c8_records) != 2180:
        raise StageCBenchmarkIntegrityError(
            "post-C9 budget requires all 2,180 C8 source records"
        )
    c8_succeeded = sum(
        record.get("state") == "succeeded"
        and record.get("trusted_by_batch_validator") is True
        for record in c8_records if isinstance(record, Mapping)
    )
    if c8_succeeded != 2179:
        raise StageCBenchmarkIntegrityError(
            "post-C9 budget C8 usable-unit count drifted"
        )
    by_digest = {
        record.get("unit_key", {}).get("canonical_digest"): record
        for record in c9_evidence if isinstance(record, Mapping)
    }
    if len(by_digest) != len(c9_evidence):
        raise StageCBenchmarkIntegrityError(
            "post-C9 budget C9 evidence contains duplicate identities"
        )
    c9_records: list[dict[str, Any]] = []
    for unit, row in zip(plan.units, rows, strict=True):
        digest = unit.key.canonical_digest
        record = by_digest.get(digest)
        if row.get("state") == "succeeded":
            if record is None:
                raise StageCBenchmarkIntegrityError(
                    f"post-C9 budget lacks successful evidence for {digest}"
                )
            c9_records.append(_plain(record))
        else:
            if record is not None:
                raise StageCBenchmarkIntegrityError(
                    f"post-C9 budget has evidence for non-success {digest}"
                )
            c9_records.append({
                "unit_key": unit.key.to_dict(),
                "state": row.get("state"),
                "trusted_by_batch_validator": False,
            })
    c9_succeeded = len(c9_evidence)
    c7_binding = c8_evidence.get("c7_report_binding")
    estimate = (
        c7_binding.get("recomputed_estimate")
        if isinstance(c7_binding, Mapping) else None
    )
    storage_model = (
        estimate.get("storage") if isinstance(estimate, Mapping) else None
    )
    payload = (
        storage_model.get("payload_bytes_before_control_overhead")
        if isinstance(storage_model, Mapping) else None
    )
    if type(payload) is not int or payload < 0:
        raise StageCBenchmarkIntegrityError(
            "post-C9 budget lacks the frozen C6 storage operands"
        )
    report_path = root / FINAL_REPORT_FILENAME
    c9_tree_bytes = measure_artifact_tree_bytes(root).total_bytes
    if report_path.exists():
        if report_path.is_symlink() or not report_path.is_file():
            raise StageCBenchmarkIntegrityError(
                "C9 final report path is not a regular file"
            )
        c9_tree_bytes -= report_path.stat().st_size
    c9_control_bytes = sum(
        _safe_regular(root / name, f"C9 budget control {name}").stat().st_size
        for name in _CONTROL_NAMES
    )
    c8_root = Path(str(c8_evidence.get("source_root", "")))
    if not c8_root.is_absolute():
        raise StageCBenchmarkIntegrityError(
            "post-C9 budget C8 source root is invalid"
        )
    control_paths = c8_evidence.get("control_file_paths")
    if isinstance(control_paths, list):
        c8_control_bytes = sum(
            _safe_regular(Path(name), "C8 salvage budget control").stat().st_size
            for name in control_paths
        )
    else:
        c8_control_bytes = sum(
            _safe_regular(c8_root / name, f"C8 budget control {name}").stat().st_size
            for name in C8_CONTROL_FILE_SHA256
        )
    c8_tree_bytes = c8_evidence.get("bindings", {}).get(
        "source_tree_total_bytes"
    )
    if type(c8_tree_bytes) is not int or c8_tree_bytes < 0:
        raise StageCBenchmarkIntegrityError(
            "post-C9 budget C8 tree-byte binding is invalid"
        )
    combined_records = [*_plain(c8_records), *c9_records]
    projection = project_benchmark_budget(
        combined_records, caps=caps,
        c6_storage_payload_bytes=payload,
        source_batch_regular_file_bytes=c8_tree_bytes + c9_tree_bytes,
        # C8 intentionally retains one trusted terminal failure.  The 2,179
        # valid C8 units plus all 320 C9 units are the complete measurement set.
        source_unit_count=2499, c6_storage_model=storage_model,
        source_batch_control_bytes=c8_control_bytes + c9_control_bytes,
    )
    usable = projection.get("measurement_completeness", {}).get("usable_units")
    actual_combined = c8_tree_bytes + c9_tree_bytes
    actual_cap = caps["stage_c_c8_c9_total_artifact_bytes_max"]
    actual_check = {
        "cap_key": "stage_c_c8_c9_total_artifact_bytes_max",
        "observed": actual_combined, "cap": actual_cap,
        "comparison": "observed <= cap",
        "status": "PASS" if actual_combined <= actual_cap else "BLOCKED",
    }
    blocked = [
        item["cap_key"] for item in projection["hard_cap_checks"]
        if item.get("status") != "PASS"
    ]
    if actual_check["status"] != "PASS":
        blocked.append("actual_retained_stage_c_c8_c9_artifact_bytes")
    document = {
        "schema_version": 1,
        "document_type": "stage_c_c9_post_run_budget",
        "status": "PASS" if not blocked else "BLOCKED",
        "blocked_reasons": blocked,
        "source_unit_counts": {
            "c8_expected": 2180, "c8_succeeded": c8_succeeded,
            "c9_expected": C9_EXECUTION_UNITS,
            "c9_succeeded": c9_succeeded,
            "budget_measurement_expected": 2499,
            "budget_measurement_usable": usable,
        },
        "measurement_boundary": {
            "c8_source_regular_file_bytes": c8_tree_bytes,
            "c9_source_regular_file_bytes_before_report": c9_tree_bytes,
            "prospective_final_report_bytes": 0,
            "c9_retained_bytes_with_report": c9_tree_bytes,
            "combined_retained_bytes_with_report": actual_combined,
            "c8_source_control_bytes": c8_control_bytes,
            "c9_source_control_bytes": c9_control_bytes,
            "actual_combined_cap_check": actual_check,
        },
        "projection": projection,
    }
    document["canonical_digest"] = _self_digest(document)
    return document


def _bind_final_report_size(
    report: dict[str, Any], *, caps: Mapping[str, Any],
) -> None:
    """Reach a fixed point for the report's own retained-byte evidence."""

    if report["post_c9_budget"].get("measurement_boundary") is None:
        report["canonical_digest"] = _self_digest(report)
        return
    for _ in range(16):
        serialized_bytes = len(_c8._json_bytes(report))
        budget = _plain(report["post_c9_budget"])
        boundary = budget.get("measurement_boundary")
        if not isinstance(boundary, Mapping):
            report["canonical_digest"] = _self_digest(report)
            return
        boundary = dict(boundary)
        if boundary.get("prospective_final_report_bytes") == serialized_bytes:
            return
        c8_bytes = boundary["c8_source_regular_file_bytes"]
        c9_before = boundary["c9_source_regular_file_bytes_before_report"]
        combined = c8_bytes + c9_before + serialized_bytes
        cap = caps["stage_c_c8_c9_total_artifact_bytes_max"]
        actual_status = "PASS" if combined <= cap else "BLOCKED"
        boundary.update({
            "prospective_final_report_bytes": serialized_bytes,
            "c9_retained_bytes_with_report": c9_before + serialized_bytes,
            "combined_retained_bytes_with_report": combined,
            "actual_combined_cap_check": {
                "cap_key": "stage_c_c8_c9_total_artifact_bytes_max",
                "observed": combined, "cap": cap,
                "comparison": "observed <= cap", "status": actual_status,
            },
        })
        budget["measurement_boundary"] = boundary
        projected_failures = [
            item["cap_key"]
            for item in budget["projection"]["hard_cap_checks"]
            if item.get("status") != "PASS"
        ]
        if actual_status != "PASS":
            projected_failures.append(
                "actual_retained_stage_c_c8_c9_artifact_bytes"
            )
        budget["blocked_reasons"] = list(dict.fromkeys(projected_failures))
        budget["status"] = "PASS" if not budget["blocked_reasons"] else "BLOCKED"
        budget["canonical_digest"] = _self_digest(budget)
        report["post_c9_budget"] = budget
        report["status"] = (
            "PASS" if report["integrity_valid"]
            and report["orchestration_complete"] and report["successful"]
            and report["primary_holdout_complete"]
            and budget["status"] == "PASS" else "BLOCKED"
        )
        report["canonical_digest"] = _self_digest(report)
    raise StageCBenchmarkIntegrityError(
        "C9 final report byte-size binding did not converge"
    )


def _validate_post_c9_budget(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _POST_C9_BUDGET_KEYS:
        raise StageCBenchmarkIntegrityError(
            "C9 post-run budget fields drifted"
        )
    if (
        value.get("schema_version") != 1
        or value.get("document_type") != "stage_c_c9_post_run_budget"
        or value.get("status") not in {"PASS", "BLOCKED"}
        or value.get("canonical_digest") != _self_digest(value)
        or not isinstance(value.get("blocked_reasons"), list)
        or any(
            not isinstance(reason, str) or not reason
            for reason in value.get("blocked_reasons", [])
        )
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 post-run budget header/digest is invalid"
        )
    counts = value.get("source_unit_counts")
    if not isinstance(counts, Mapping) or set(counts) != {
        "c8_expected", "c8_succeeded", "c9_expected",
        "c9_succeeded", "budget_measurement_expected",
        "budget_measurement_usable",
    } or counts.get("c8_expected") != 2180 or counts.get("c9_expected") != 320:
        raise StageCBenchmarkIntegrityError(
            "C9 post-run budget source counts are invalid"
        )
    projection = value.get("projection")
    boundary = value.get("measurement_boundary")
    if projection is None or boundary is None:
        if value.get("status") != "BLOCKED" or not value["blocked_reasons"]:
            raise StageCBenchmarkIntegrityError(
                "unmeasured C9 post-run budget must be BLOCKED"
            )
        return
    expected_boundary_keys = {
        "c8_source_regular_file_bytes",
        "c9_source_regular_file_bytes_before_report",
        "prospective_final_report_bytes",
        "c9_retained_bytes_with_report",
        "combined_retained_bytes_with_report",
        "c8_source_control_bytes", "c9_source_control_bytes",
        "actual_combined_cap_check",
    }
    if not isinstance(boundary, Mapping) or set(boundary) != expected_boundary_keys:
        raise StageCBenchmarkIntegrityError(
            "C9 post-run budget measurement boundary drifted"
        )
    byte_fields = (
        "c8_source_regular_file_bytes",
        "c9_source_regular_file_bytes_before_report",
        "prospective_final_report_bytes",
        "c9_retained_bytes_with_report",
        "combined_retained_bytes_with_report",
        "c8_source_control_bytes", "c9_source_control_bytes",
    )
    if any(type(boundary.get(name)) is not int or boundary[name] < 0 for name in byte_fields):
        raise StageCBenchmarkIntegrityError(
            "C9 post-run budget byte boundary is invalid"
        )
    if (
        boundary["c9_retained_bytes_with_report"]
        != boundary["c9_source_regular_file_bytes_before_report"]
        + boundary["prospective_final_report_bytes"]
        or boundary["combined_retained_bytes_with_report"]
        != boundary["c8_source_regular_file_bytes"]
        + boundary["c9_retained_bytes_with_report"]
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 post-run budget byte arithmetic is inconsistent"
        )
    checks = projection.get("hard_cap_checks")
    if (
        not isinstance(checks, list) or len(checks) != len(HARD_CAP_KEYS)
        or [item.get("cap_key") for item in checks if isinstance(item, Mapping)]
        != list(HARD_CAP_KEYS)
        or any(
            not isinstance(item, Mapping)
            or item.get("cap") != FROZEN_CAPS.get(item.get("cap_key"))
            or item.get("comparison") != "observed <= cap"
            for item in checks
        )
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 post-run budget does not contain the exact ten cap checks"
        )
    projection_pass = (
        all(
            item.get("status") == "PASS" and item.get("measured") is True
            and isinstance(item.get("observed"), (int, float))
            and not isinstance(item.get("observed"), bool)
            and item["observed"] <= item["cap"]
            for item in checks
        )
        and projection.get("all_hard_caps_pass") is True
    )
    actual = boundary.get("actual_combined_cap_check")
    actual_pass = (
        isinstance(actual, Mapping) and actual.get("status") == "PASS"
        and set(actual) == {
            "cap_key", "observed", "cap", "comparison", "status",
        }
        and actual.get("cap_key")
        == "stage_c_c8_c9_total_artifact_bytes_max"
        and actual.get("comparison") == "observed <= cap"
        and actual.get("cap")
        == FROZEN_CAPS["stage_c_c8_c9_total_artifact_bytes_max"]
        and type(actual.get("observed")) is int
        and actual.get("observed")
        == boundary["combined_retained_bytes_with_report"]
        and actual["observed"] <= actual["cap"]
    )
    expected_pass = (
        projection_pass and actual_pass and counts.get("c8_succeeded") == 2179
        and counts.get("c9_succeeded") == 320
        and counts.get("budget_measurement_expected") == 2499
        and counts.get("budget_measurement_usable") == 2499
    )
    if (value.get("status") == "PASS") is not expected_pass:
        raise StageCBenchmarkIntegrityError(
            "C9 post-run budget status is inconsistent"
        )
    if (value.get("status") == "PASS") is (bool(value["blocked_reasons"])):
        raise StageCBenchmarkIntegrityError(
            "C9 post-run budget blocked reasons are inconsistent"
        )


def _report_bindings(
    root: Path, plan: ExecutionPlan, estimate: StageCEstimate,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "output_root": str(root), "code": _plain(authorization["code"]),
        "c8_handoff_sha256": authorization["c8_handoff_sha256"],
        "execution_plan_sha256": plan.canonical_digest,
        "run_spec_sha256": canonical_sha256(plan.run_spec.to_dict()),
        "preflight_sha256": estimate.canonical_digest,
        "authorization_sha256": authorization["canonical_digest"],
        # Finalization calls this only after the mutable controls have reached
        # their sealed finalized state, so the report can bind all controls.
        "control_file_sha256": {
            name: _file_sha256(root / name) for name in _CONTROL_NAMES
        },
    }


def _validate_report_bindings(
    value: Any, *, plan: ExecutionPlan | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REPORT_BINDING_KEYS:
        raise StageCBenchmarkIntegrityError(
            "C9 final report binding fields drifted"
        )
    output_root = value.get("output_root")
    if not isinstance(output_root, str) or not Path(output_root).is_absolute():
        raise StageCBenchmarkIntegrityError(
            "C9 final report output root is not absolute"
        )
    code = value.get("code")
    if (
        not isinstance(code, Mapping) or set(code) != {"commit", "dirty"}
        or code.get("dirty") is not False
        or not isinstance(code.get("commit"), str)
        or _COMMIT.fullmatch(code["commit"]) is None
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 final report code binding is invalid"
        )
    for name in (
        "c8_handoff_sha256", "execution_plan_sha256",
        "run_spec_sha256", "preflight_sha256",
        "authorization_sha256",
    ):
        _sha(value.get(name), f"C9 final report bindings.{name}")
    controls = value.get("control_file_sha256")
    expected_controls = set(_CONTROL_NAMES)
    if not isinstance(controls, Mapping) or set(controls) != expected_controls:
        raise StageCBenchmarkIntegrityError(
            "C9 final report control hash fields drifted"
        )
    for name in expected_controls:
        _sha(controls[name], f"C9 final report control {name}")
    if plan is not None and (
        value.get("execution_plan_sha256") != plan.canonical_digest
        or value.get("run_spec_sha256")
        != canonical_sha256(plan.run_spec.to_dict())
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 final report plan/run-spec binding drifted"
        )
    return value


def _validate_report_row(
    row: Any, *, index: int, unit: Any | None, output_root: Path,
) -> None:
    if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
        raise StageCBenchmarkIntegrityError(
            f"C9 final report row {index} fields drifted"
        )
    state = row.get("state")
    if state not in _FINAL_STATES:
        raise StageCBenchmarkIntegrityError(
            f"C9 final report row {index} is not explicitly terminal"
        )
    if unit is not None and row.get("unit_key") != unit.key.to_dict():
        raise StageCBenchmarkIntegrityError(
            f"C9 final report row {index} identity drifted"
        )
    if (
        type(row.get("terminal")) is not bool
        or type(row.get("complete")) is not bool
        or row["terminal"] is not (state in _TERMINAL_STATES)
        or row["complete"] is not (state == "succeeded")
    ):
        raise StageCBenchmarkIntegrityError(
            f"C9 final report row {index} state flags drifted"
        )
    token = row.get("claim_token")
    if state in {"claimed", "succeeded", "failed", "incomplete"} and (
        not isinstance(token, str) or len(token) != 64 or set(token) - _SHA256
    ):
        raise StageCBenchmarkIntegrityError(
            f"C9 final report row {index} claim token is invalid"
        )
    if token is not None and (
        not isinstance(token, str) or len(token) != 64 or set(token) - _SHA256
    ):
        raise StageCBenchmarkIntegrityError(
            f"C9 final report row {index} claim token is malformed"
        )
    selected = row.get("selected_attempt")
    byte_fields = (
        row.get("l0_l1_bytes"), row.get("l2_bytes"),
        row.get("total_bytes"),
    )
    if selected is None:
        if row.get("selected_status_sha256") is not None or any(
            item is not None for item in byte_fields
        ):
            raise StageCBenchmarkIntegrityError(
                f"C9 final report row {index} has partial attempt evidence"
            )
        if state in _TERMINAL_STATES:
            raise StageCBenchmarkIntegrityError(
                f"C9 final report terminal row {index} lacks an attempt"
            )
    else:
        if not isinstance(selected, str):
            raise StageCBenchmarkIntegrityError(
                f"C9 final report row {index} attempt path is invalid"
            )
        selected_path = Path(selected)
        digest = row.get("unit_key", {}).get("canonical_digest")
        expected_parent = (
            output_root / "units" / str(digest) / "attempts"
        ).absolute()
        if (
            not selected_path.is_absolute()
            or selected_path.parent != expected_parent
            or _ATTEMPT_NAME.fullmatch(selected_path.name) is None
        ):
            raise StageCBenchmarkIntegrityError(
                f"C9 final report row {index} attempt escapes its unit"
            )
        status_hash = row.get("selected_status_sha256")
        if status_hash is not None:
            _sha(status_hash, f"C9 final report row {index} status hash")
        if state in _TERMINAL_STATES and status_hash is None:
            raise StageCBenchmarkIntegrityError(
                f"C9 final report terminal row {index} lacks status hash"
            )
        if any(type(item) is not int or item < 0 for item in byte_fields):
            raise StageCBenchmarkIntegrityError(
                f"C9 final report row {index} byte evidence is invalid"
            )
        if row["total_bytes"] != row["l0_l1_bytes"] + row["l2_bytes"]:
            raise StageCBenchmarkIntegrityError(
                f"C9 final report row {index} byte totals disagree"
            )
    failure = row.get("failure")
    if state == "succeeded" and failure is not None:
        raise StageCBenchmarkIntegrityError(
            f"C9 final report successful row {index} has a failure"
        )
    if state == "failed" and not isinstance(failure, Mapping):
        raise StageCBenchmarkIntegrityError(
            f"C9 final report failed row {index} lacks failure evidence"
        )


def _contains_c10_output(value: Any) -> bool:
    forbidden = {
        "decision", "go", "no_go_validity", "no_go_novelty",
        "no_go_cost", "inconclusive", "bootstrap_ci", "holm_results",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in forbidden or _contains_c10_output(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_c10_output(item) for item in value)
    return isinstance(value, str) and value.lower() in forbidden


def validate_c9_final_report(
    value: Mapping[str, Any], *, plan: ExecutionPlan | None = None,
    output_root: Path | str | None = None,
    estimate: StageCEstimate | None = None,
    authorization: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _REPORT_KEYS:
        raise StageCBenchmarkIntegrityError("C9 final report fields drifted")
    if value.get("canonical_digest") != _self_digest(value):
        raise StageCBenchmarkIntegrityError("C9 final report digest mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("document_type") != "stage_c_c9_evidence_report"
        or value.get("status") not in {"PASS", "BLOCKED"}
        or type(value.get("integrity_valid")) is not bool
        or type(value.get("orchestration_complete")) is not bool
        or type(value.get("successful")) is not bool
        or value.get("expected_unit_count") != C9_EXECUTION_UNITS
        or value.get("primary_holdout_expected_count") != 15
    ):
        raise StageCBenchmarkIntegrityError("C9 final report header drifted")
    for flag in (
        "bootstrap_performed", "holm_performed", "decision_performed",
        "selection_performed",
    ):
        if value.get(flag) is not False:
            raise StageCBenchmarkIntegrityError(
                f"C9 final report performed forbidden C10 phase: {flag}"
            )
    if _contains_c10_output(value):
        raise StageCBenchmarkIntegrityError(
            "C9 final report contains forbidden C10 output"
        )
    _validate_post_c9_budget(value.get("post_c9_budget"))
    bindings = _validate_report_bindings(value.get("bindings"), plan=plan)
    if output_root is not None:
        root = Path(os.path.abspath(os.fspath(output_root)))
        if bindings["output_root"] != str(root):
            raise StageCBenchmarkIntegrityError(
                "C9 final report output-root binding drifted"
            )
        expected_controls = {
            name: _file_sha256(
                _safe_regular(root / name, f"C9 finalized control {name}")
            )
            for name in _CONTROL_NAMES
        }
        if dict(bindings["control_file_sha256"]) != expected_controls:
            raise StageCBenchmarkIntegrityError(
                "C9 final report immutable control bytes drifted"
            )
    if (
        estimate is not None
        and bindings["preflight_sha256"] != estimate.canonical_digest
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 final report preflight binding drifted"
        )
    if authorization is not None and (
        bindings["authorization_sha256"]
        != authorization.get("canonical_digest")
        or bindings["c8_handoff_sha256"]
        != authorization.get("c8_handoff_sha256")
        or dict(bindings["code"]) != dict(authorization.get("code", {}))
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 final report authorization binding drifted"
        )
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != C9_EXECUTION_UNITS:
        raise StageCBenchmarkIntegrityError("C9 report must retain 320 rows")
    if plan is not None and len(plan.units) != C9_EXECUTION_UNITS:
        raise StageCBenchmarkIntegrityError("C9 report plan is not 320 units")
    output_root = Path(str(bindings["output_root"]))
    for index, row in enumerate(rows):
        _validate_report_row(
            row, index=index, unit=None if plan is None else plan.units[index],
            output_root=output_root,
        )
    counts = Counter(row.get("state") for row in rows)
    if value.get("state_counts") != dict(sorted(counts.items())):
        raise StageCBenchmarkIntegrityError("C9 report state counts drifted")
    terminal = sum(row.get("terminal") is True for row in rows)
    if (
        value.get("terminal_unit_count") != terminal
        or value.get("orchestration_complete") is not (terminal == 320)
    ):
        raise StageCBenchmarkIntegrityError("C9 report terminal count drifted")
    if terminal != C9_EXECUTION_UNITS:
        for row in rows:
            if row["state"] in _TERMINAL_STATES:
                continue
            digest = row["unit_key"]["canonical_digest"]
            if not any(
                digest in error
                for error in value.get("errors", [])
            ):
                raise StageCBenchmarkIntegrityError(
                    "C9 nonterminal row lacks an explicit report error"
                )
    valid_count = value.get("primary_holdout_valid_count")
    failures = value.get("primary_holdout_failures")
    if (
        type(valid_count) is not int or not 0 <= valid_count <= 15
        or not isinstance(failures, list) or len(failures) != 15 - valid_count
        or value.get("primary_holdout_complete") is not (valid_count == 15)
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 primary 15/15 validity accounting drifted"
        )
    errors = value.get("errors")
    if not isinstance(errors, list) or any(
        not isinstance(item, str) or not item for item in errors
    ) or len(errors) != len(set(errors)):
        raise StageCBenchmarkIntegrityError("C9 final report errors are invalid")
    failure_ids: list[str] = []
    for index, failure in enumerate(failures):
        if (
            not isinstance(failure, Mapping)
            or set(failure) != {"sample_id", "reasons"}
            or failure.get("sample_id") not in C9_HOLDOUT_SAMPLE_IDS
            or not isinstance(failure.get("reasons"), list)
            or not failure["reasons"]
            or any(
                not isinstance(reason, str) or not reason
                for reason in failure["reasons"]
            )
            or len(failure["reasons"]) != len(set(failure["reasons"]))
        ):
            raise StageCBenchmarkIntegrityError(
                f"C9 primary failure row {index} is invalid"
            )
        failure_ids.append(failure["sample_id"])
    expected_failure_order = [
        sample_id for sample_id in C9_HOLDOUT_SAMPLE_IDS
        if sample_id in set(failure_ids)
    ]
    if failure_ids != expected_failure_order or len(failure_ids) != len(set(failure_ids)):
        raise StageCBenchmarkIntegrityError(
            "C9 primary failure IDs are duplicated or out of frozen order"
        )
    derived_complete = terminal == C9_EXECUTION_UNITS
    derived_integrity = derived_complete and not errors
    derived_success = derived_complete and all(
        row["state"] == "succeeded" for row in rows
    )
    if (
        value["integrity_valid"] is not derived_integrity
        or value["successful"] is not derived_success
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 final report success/integrity flags drifted from rows"
        )
    if valid_count == 15 and failures:
        raise StageCBenchmarkIntegrityError(
            "C9 final report claims 15/15 primary validity with failures"
        )
    expected_pass = (
        value["integrity_valid"] and value["orchestration_complete"]
        and value["successful"] and valid_count == 15 and not errors
        and value["post_c9_budget"]["status"] == "PASS"
    )
    if (value["status"] == "PASS") is not expected_pass:
        raise StageCBenchmarkIntegrityError("C9 final report status is inconsistent")


def finalize_c9_output_root(
    output_root: Path | str, plan: ExecutionPlan, *,
    observed_git_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal the root and publish evidence only; never run C10 statistics."""

    root = _require_safe_existing_root(output_root)
    with (
        _lifecycle_lock(root, exclusive=True),
        _checkpoint_lock(root),
        _artifact_activity_lock(root, exclusive=True),
    ):
        estimate, authorization = load_c9_initialized_state(
            root, plan, observed_git_state=observed_git_state,
            require_checkpoint=False,
        )
        current_rows, current_status = _validate_checkpoint_controls(
            root, plan, allowed_lifecycle=(
                "running", "blocked", "finalizing", "finalized",
            )
        )
        if current_status["active_claim_count"]:
            raise StageCBenchmarkBlockedError(
                "C9 cannot finalize while execution units are claimed"
            )
        if current_status["lifecycle_state"] != "finalized":
            _c8._atomic_replace_json(
                root / "batch_status.json",
                _batch_status(
                    plan, current_rows,
                    _file_sha256(root / "failure_table.json"),
                    validation_errors=current_status["validation_errors"],
                    lifecycle_state="finalizing",
                ),
            )
        tokens = {
            row["unit_key"]["canonical_digest"]: row["claim_token"]
            for row in current_rows
        }
        rows, errors = _scan_rows(
            root, plan, estimate, observed_git_state=authorization["code"],
            claim_tokens=tokens,
            trusted_recovery_context=(
                _recovery_runtime_context(authorization)
            ),
        )
        errors = [*current_status["validation_errors"], *errors]
        errors.extend(_validate_closed_layout(root, plan, require_all_units=True))
        # Revalidate all C8 source artifacts again at the final C9 boundary.
        # This intentionally pays the full scan once more, after all C9 work
        # and before evidence publication.
        c8_evidence_values: list[dict[str, Any]] = []
        try:
            _revalidate_handoff_from_authorization(
                authorization["c8_handoff"], measure_source_tree=True,
                source_evidence_out=c8_evidence_values,
            )
        except Exception as error:
            errors.append(f"C8 handoff changed before C9 finalize: {error}")
        valid_ids, primary_failures, evidence, shared_errors = (
            _primary_holdout_validity(
                rows, plan, root, authorization["code"],
                dict(estimate.caps["effective"]),
                trusted_recovery_context=_recovery_runtime_context(
                    authorization
                ),
            )
        )
        errors.extend(shared_errors)
        errors = list(dict.fromkeys(errors))
        table = _failure_table(plan, rows)
        _c8._atomic_replace_json(root / "failure_table.json", table)
        status = _batch_status(
            plan, rows, _file_sha256(root / "failure_table.json"),
            validation_errors=errors, lifecycle_state="finalized",
        )
        _c8._atomic_replace_json(root / "batch_status.json", status)
        try:
            if len(c8_evidence_values) != 1:
                raise StageCBenchmarkIntegrityError(
                    "C8 source evidence was not captured exactly once"
                )
            post_c9_budget = _build_post_c9_budget(
                root, plan, rows, evidence, c8_evidence_values[0],
                dict(estimate.caps["effective"]),
            )
        except Exception as error:
            budget_error = f"post-C9 budget recomputation failed: {error}"
            post_c9_budget = _blocked_post_c9_budget(budget_error)
            errors = list(dict.fromkeys([*errors, budget_error]))
            status = _batch_status(
                plan, rows, _file_sha256(root / "failure_table.json"),
                validation_errors=errors, lifecycle_state="finalized",
            )
            _c8._atomic_replace_json(root / "batch_status.json", status)
        terminal_count = sum(row["terminal"] is True for row in rows)
        integrity_valid = not errors and terminal_count == C9_EXECUTION_UNITS
        successful = status["successful"] and not errors
        report: dict[str, Any] = {
            "schema_version": 1,
            "document_type": "stage_c_c9_evidence_report",
            "status": (
                "PASS" if integrity_valid and successful
                and len(valid_ids) == 15 else "BLOCKED"
            ),
            "integrity_valid": integrity_valid,
            "orchestration_complete": terminal_count == C9_EXECUTION_UNITS,
            "successful": successful,
            "expected_unit_count": C9_EXECUTION_UNITS,
            "terminal_unit_count": terminal_count,
            "state_counts": dict(sorted(Counter(
                row["state"] for row in rows
            ).items())),
            "primary_holdout_valid_count": len(valid_ids),
            "primary_holdout_expected_count": 15,
            "primary_holdout_complete": len(valid_ids) == 15,
            "primary_holdout_failures": primary_failures,
            "bootstrap_performed": False, "holm_performed": False,
            "decision_performed": False, "selection_performed": False,
            "rows": rows, "errors": errors,
            "post_c9_budget": post_c9_budget,
            "bindings": _report_bindings(
                root, plan, estimate, authorization
            ),
        }
        report["canonical_digest"] = _self_digest(report)
        _bind_final_report_size(
            report, caps=dict(estimate.caps["effective"]),
        )
        validate_c9_final_report(
            report, plan=plan, output_root=root, estimate=estimate,
            authorization=authorization,
        )
        report_path = root / FINAL_REPORT_FILENAME
        if report_path.exists():
            existing = _strict_json(_safe_regular(
                report_path, "C9 final evidence report"
            ))
            if existing != report:
                raise StageCBenchmarkIntegrityError(
                    "existing C9 final report differs from recomputed evidence"
                )
        else:
            _c8._atomic_create_json(report_path, report)
    return report


def validate_c9_output_root(
    output_root: Path | str, plan: ExecutionPlan, *,
    observed_git_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read-only validation of an already finalized C9 evidence root."""

    root = _require_safe_existing_root(output_root)
    estimate, authorization = load_c9_initialized_state(
        root, plan, observed_git_state=observed_git_state,
        require_checkpoint=False,
    )
    checkpoint_rows, status = _validate_checkpoint_controls(
        root, plan, allowed_lifecycle=("finalized",)
    )
    if status["active_claim_count"]:
        raise StageCBenchmarkIntegrityError(
            "finalized C9 root retains active claims"
        )
    report = _strict_json(_safe_regular(
        root / FINAL_REPORT_FILENAME, "C9 final evidence report"
    ))
    validate_c9_final_report(
        report, plan=plan, output_root=root, estimate=estimate,
        authorization=authorization,
    )
    rows, scan_errors = _scan_rows(
        root, plan, estimate, observed_git_state=authorization["code"],
        claim_tokens={
            row["unit_key"]["canonical_digest"]: row["claim_token"]
            for row in checkpoint_rows
        },
        trusted_recovery_context=_recovery_runtime_context(authorization),
    )
    validation_errors = [*scan_errors]
    validation_errors.extend(
        _validate_closed_layout(root, plan, require_all_units=True)
    )
    valid_ids, primary_failures, _evidence, shared_errors = (
        _primary_holdout_validity(
            rows, plan, root, authorization["code"],
            dict(estimate.caps["effective"]),
            trusted_recovery_context=_recovery_runtime_context(authorization),
        )
    )
    validation_errors.extend(shared_errors)
    validation_errors = list(dict.fromkeys(validation_errors))
    if rows != report["rows"]:
        raise StageCBenchmarkIntegrityError(
            "C9 report rows differ from a fresh artifact scan"
        )
    if (
        len(valid_ids) != report["primary_holdout_valid_count"]
        or primary_failures != report["primary_holdout_failures"]
        or validation_errors != report["errors"]
    ):
        raise StageCBenchmarkIntegrityError(
            "C9 report validity/errors differ from fresh evidence"
        )
    c8_evidence_values: list[dict[str, Any]] = []
    _revalidate_handoff_from_authorization(
        authorization["c8_handoff"], measure_source_tree=True,
        source_evidence_out=c8_evidence_values,
    )
    if len(c8_evidence_values) != 1:
        raise StageCBenchmarkIntegrityError(
            "C8 source evidence was not captured exactly once"
        )
    recomputed_budget = _build_post_c9_budget(
        root, plan, rows, _evidence, c8_evidence_values[0],
        dict(estimate.caps["effective"]),
    )
    report_size = (root / FINAL_REPORT_FILENAME).stat().st_size
    boundary = recomputed_budget["measurement_boundary"]
    boundary["prospective_final_report_bytes"] = report_size
    boundary["c9_retained_bytes_with_report"] += report_size
    boundary["combined_retained_bytes_with_report"] += report_size
    actual = boundary["actual_combined_cap_check"]
    actual["observed"] += report_size
    actual["status"] = (
        "PASS" if actual["observed"] <= actual["cap"] else "BLOCKED"
    )
    recomputed_budget["blocked_reasons"] = [
        item["cap_key"]
        for item in recomputed_budget["projection"]["hard_cap_checks"]
        if item.get("status") != "PASS"
    ]
    if actual["status"] != "PASS":
        recomputed_budget["blocked_reasons"].append(
            "actual_retained_stage_c_c8_c9_artifact_bytes"
        )
    recomputed_budget["status"] = (
        "PASS" if not recomputed_budget["blocked_reasons"] else "BLOCKED"
    )
    recomputed_budget["canonical_digest"] = _self_digest(recomputed_budget)
    if recomputed_budget != report["post_c9_budget"]:
        raise StageCBenchmarkIntegrityError(
            "C9 post-run budget differs from fresh source evidence"
        )
    expected_control_hashes = {
        name: _file_sha256(root / name) for name in _CONTROL_NAMES
    }
    if report.get("bindings", {}).get("control_file_sha256") != expected_control_hashes:
        raise StageCBenchmarkIntegrityError(
            "C9 final report control-file bindings drifted"
        )
    # The finalized report is itself included in the live combined-cap check
    # above, closing the pre-publication accounting gap.
    return report


__all__ = [
    "AUTHORIZATION_FILENAME", "FINAL_REPORT_FILENAME",
    "C8_SOURCE_ROOT", "C8_REPORT_PATH", "C8_RECOVERY_SELECTION_PATH",
    "C8_PRODUCER_COMMIT", "C8_GENERATOR_COMMIT",
    "C8_COMPLETION_COMMIT", "C8_REPORT_FILE_SHA256",
    "C8_REPORT_CANONICAL_DIGEST", "C8_SELECTION_FILE_SHA256",
    "C8_SELECTION_CANONICAL_DIGEST",
    "C8_CALIBRATION_ARTIFACT_MANIFEST_SHA256",
    "C8_CALIBRATION_EVIDENCE_SHA256", "C8_CONTROL_FILE_SHA256",
    "C8_SOURCE_TREE_TOTAL_BYTES", "C8_PROJECTED_ARTIFACT_BYTES",
    "C9_PROJECTED_ARTIFACT_BYTES", "C8_C9_PROJECTED_ARTIFACT_BYTES",
    "C9_PROJECTED_H20_GPU_HOURS", "C8_C9_PROJECTED_H20_GPU_HOURS",
    "C9_EXECUTION_UNITS", "C9_UNITS_PER_IMAGE",
    "C9_EXPECTED_COUNTS", "C9_SAMPLE_IDS", "C9_HOLDOUT_SAMPLE_IDS",
    "C9_WINNERS", "C9_PRIMARY",
    "build_c9_launch_authorization", "build_c9_measured_preflight",
    "finalize_c9_output_root", "initialize_c9_output_root",
    "load_c9_initialized_state", "run_c9_shard",
    "validate_c8_handoff", "validate_c9_execution_plan",
    "validate_c9_final_report", "validate_c9_launch_authorization",
    "validate_c9_output_root",
]
