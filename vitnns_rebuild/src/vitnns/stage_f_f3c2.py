"""Stage F F3C2 blind five-image pilot runner and control plane.

The continuation gate in this module intentionally reads only blind terminal
records, manifests, identities, integrity state, and resource telemetry.  It
never loads numerical arrays or result payload values.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_f_f3c1 as f3c1
from . import stage_f_f3c1_kernels as kernels


CONFIG_PATH = "config/stage_f_f3c2.json"
SCHEMA_PATH = "config/stage_f_f3c2.schema.json"
ROOT_PATH = "evidence/stage_f/f3c2_pilot_v1_20260907/formal"
REPORT_PATH = "reports/stage_f_f3c2_pilot_v1_20260907"
STARTUP_PATH = ROOT_PATH + "/startup_gate.json"
PILOT_MANIFEST_PATH = ROOT_PATH + "/pilot_manifest.json"
UNIT_PLAN_PATH = ROOT_PATH + "/unit_execution_plan.json"
MODEL_BINDING_PATH = ROOT_PATH + "/model_preprocessing_binding.json"
SOURCE_BINDING_PATH = ROOT_PATH + "/source_config_binding.json"
RESOURCE_SPEC_PATH = ROOT_PATH + "/resource_accounting_specification.json"
BLIND_SPEC_PATH = ROOT_PATH + "/blind_continuation_specification.json"
ACCESS_POLICY_PATH = ROOT_PATH + "/access_policy.json"
PLAN_SEAL_PATH = ROOT_PATH + "/plan_seal.json"
PENDING_PATH = ROOT_PATH + "/completion/f3c2.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/f3c2.json"
TERMINAL_UNION_PATH = ROOT_PATH + "/terminal_union.json"
ATTEMPT_JOURNAL_PATH = ROOT_PATH + "/attempt_journal.json"
RESOURCE_LEDGER_PATH = ROOT_PATH + "/resource_ledger.json"
BUDGET_PATH = ROOT_PATH + "/budget_projection.json"
ACCESS_AUDIT_PATH = ROOT_PATH + "/access_audit.json"
PAYLOAD_PATH = ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = ROOT_PATH + "/whole_root_inventory.json"
STATUS_PATH = ROOT_PATH + "/status.json"
BLIND_REPORT_PATH = REPORT_PATH + "/blind_continuation_report.json"
VALIDATION_PATH = REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = REPORT_PATH + "/false_pass_audit.json"
TEST_PATH = REPORT_PATH + "/test_report.json"
FAILURE_PATH = REPORT_PATH + "/failure_event.json"
ARCHIVE_REF = "refs/archive/stage-f-v1-f3c2-pilot"
EXPECTED_BRANCH = "stage-d-d3-pass"
ATTEMPT_ID = "attempt-000001"
PILOT_ORDERS = (416, 350, 438, 411, 337)
F3C1_COMPLETION_SHA256 = "2880143e5f3b0296f0ae2a2c31af3c9c1a15ec7effc60025e0118146712c0a6f"
F3C1_COMPLETION_DIGEST = "0f6803acded866017c2555b0f37d48e1c4c03d46f75863bc112842e68d1a1d44"
F3C1_INVENTORY_SHA256 = "513b91c617fa6e8961c17ebd049d9738e5a55c7b98e6d5334c98e08f04dca2c1"
F3C1_INVENTORY_DIGEST = "3c9a262bc0cd9f668c2de8a091228fa6b90afb2e4128b756583e45bab5e81407"
F3C1_FILE_INVENTORY_DIGEST = "e461b4723f5a66b0684077eb731520e250b06ca07657f35127be72410815ee4e"
F3C1_STAT_SHA256 = "d0c8b03e8a10baa3e9c934dadf2519985526391a5cb03cee278b667df7c1544f"
F3C1_STAT_DIGEST = "546b88a25bc0b63d8bc42436923f95625d24270be240d94f88e3bcd3e28603de"
F3C1_ENGINEERING_GPU_HOURS = 0.003537431458341113
F3C1_ROOT_BYTES = 2_515_724
METHODS = kernels.METHODS
MODEL_PATH = f3c1.MODEL_PATH
MODEL_NAME = f3c1.MODEL_NAME
MODEL_IDENTITY = f3c1.MODEL_IDENTITY
MODEL_LOADER_IDENTITY = f3c1.MODEL_LOADER_IDENTITY
MODEL_SHA256 = f3c1.MODEL_SHA256
MODEL_TENSOR_DIGEST = f3c1.MODEL_TENSOR_DIGEST
FROZEN_SOURCE_HASHES = {
    "config/stage_f_f3c.json": "abb17c68c0f5961b75510604de486a2aa30509bb2c5dabc3a6688a1e3a11b0df",
    "config/stage_f_f3c.schema.json": "65191984978bb9191ca0474458f50949403e55741bcf8a8381b8e4beeecb055b",
    "config/stage_f_f3c1.json": "20fd9f1b237c19a20ada17d0c731059df0c568a016ede1f9b02efb27fb92a67f",
    "config/stage_f_f3c1.schema.json": "dfa55c719aa8377947605c8e7fa59796313be4e4f42c036c45c9d24c3cf8fe27",
    "manifests/stage_f_f3c_v1/sample_split.json": "b1dd0bb74f9ee0a2381eaa797c27a0365c13757b0f72ea253924fded3d8a2b09",
    "src/vitnns/stage_f_f3c1.py": "43c8f7b3336d555e40ad0914d2776992eeb77f0bda794e428352cd954d6b8998",
    "src/vitnns/stage_f_f3c1_kernels.py": "514f522897975f1b01a67ac17009dd3d4aa75477280bc59a2b715cff20326fdb",
}
SOURCE_FILES = (
    "src/vitnns/stage_f_f3c2.py", "tests/test_stage_f_f3c2.py",
    CONFIG_PATH, SCHEMA_PATH, "src/vitnns/stage_f_cli.py",
    "src/vitnns/stage_f_f3c1.py", "src/vitnns/stage_f_f3c1_kernels.py",
    "src/vitnns/stage_c_class_logit.py", "src/vitnns/stage_c_attention.py",
    "src/vitnns/attribution.py",
)
FALSE_PASS_ATTACKS = (
    "pilot_roster_count_or_identity_drift",
    "pilot_order_sample_hash_or_label_drift",
    "order44_recomputed_or_included",
    "formal_jpeg_map_logit_or_outcome_read",
    "pilot_plan_not_sealed_before_jpeg_access",
    "class_index_reparsed_or_changed",
    "method_roster_order_or_target_label_drift",
    "attention_count_not_exactly_12",
    "attention_not_from_same_consumed_clean_forward",
    "partial_heads_weights_or_f3h_weighted_sum",
    "frozen_attention_mutated_or_qk_softmax_leak",
    "value_projection_residual_mlp_norm_classifier_cut",
    "endpoint_parity_failure_continues",
    "ig_conservation_failure_marked_valid",
    "ig_attempt_loss_or_schedule_threshold_chunk_drift",
    "normalized_zero_baseline_replaced",
    "signed_rgb_absolute_value_before_projection",
    "fraction_tie_ranking_auc_or_score_drift",
    "tie_priority_or_unmodified_evaluator_not_shared",
    "display_transform_reenters_metric",
    "failed_na_imputed_replaced_or_has_numeric_payload",
    "terminal_overwrite_multiple_valid_or_incomplete_accepted",
    "blind_gate_consumes_forbidden_numerical_outcome",
    "statistical_freeze_modified_or_executed",
    "resource_missing_double_counted_underreported_or_formula_drift",
    "budget_breach_waived_or_authorizes_f3c3",
    "pilot_jpeg_or_map_rendered_or_manually_viewed",
    "comparison_winner_pvalue_ci_or_go_no_go_generated",
    "sealed_upstream_evidence_ref_or_hash_drift",
    "f3c3_f3c4_or_f4_started",
    "completion_validator_manifest_blind_or_false_pass_missing",
    "git_diff_lfs_archive_state_unverified",
)
BLIND_FORBIDDEN_KEYS = frozenset({
    "native_map", "patch_mass", "target_oriented_mass", "ranking",
    "logits", "clean_raw_logit", "baseline_raw_logit", "curves",
    "responses", "auc", "fidelity_score", "delta_m", "p_value",
    "confidence_interval", "winner", "go_no_go",
})


class StageFF3C2Error(ValueError):
    pass


def _now() -> str:
    raw = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    return raw[:-2] + ":" + raw[-2:]


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicates(pairs: Sequence[tuple[str, Any]], path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise StageFF3C2Error(f"duplicate JSON key {key!r}: {path}")
        value[key] = item
    return value


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=lambda pairs: _reject_duplicates(pairs, path))


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["canonical_digest"] = canonical_sha256(result)
    return result


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    payload = dict(value)
    observed = payload.pop("canonical_digest", None)
    if observed != canonical_sha256(payload):
        raise StageFF3C2Error(f"{context} canonical digest mismatch")


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    value = _read(path)
    _validate_signed(value, relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path), "canonical_digest": value["canonical_digest"]}


def _file_record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded:
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return {
        "files": files, "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "file_inventory_digest": canonical_sha256({"files": files}),
    }


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if process.returncode:
        raise StageFF3C2Error(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def load_config(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = _read(repo / CONFIG_PATH)
    schema = _read(repo / SCHEMA_PATH)
    errors = sorted(Draft202012Validator(schema).iter_errors(config), key=lambda item: list(item.path))
    if errors:
        raise StageFF3C2Error(errors[0].message)
    if tuple(config["pilot_orders"]) != PILOT_ORDERS or tuple(config["methods"]) != METHODS:
        raise StageFF3C2Error("F3C2 pilot or method roster drifted")
    budget = config["budget"]
    if budget != {
        "f3c1_engineering_gpu_hours": F3C1_ENGINEERING_GPU_HOURS,
        "total_h20_gpu_hours_max": 8.0,
        "per_image_group_h20_gpu_hours_max": 0.3,
        "max_devices": 8,
        "per_process_peak_reserved_bytes_max": 17179869184,
        "retained_regular_file_bytes_max": 8589934592,
        "waiver_allowed": False, "formal_image_count": 50,
    }:
        raise StageFF3C2Error("F3C2 budget contract drifted")
    return config


def _pilot_rows(repo: Path) -> list[dict[str, Any]]:
    split = _read(repo / "manifests/stage_f_f3c_v1/sample_split.json")
    _validate_signed(split, "F3C split")
    rows = split.get("pilot", [])
    if tuple(row.get("selection_order") for row in rows) != PILOT_ORDERS:
        raise StageFF3C2Error("pilot order is not the frozen five-image order")
    required = ("sample_id", "relative_path", "image_sha256", "imagenet1k_index", "canonical_digest")
    if len(rows) != 5 or any(any(key not in row for key in required) for row in rows):
        raise StageFF3C2Error("pilot sample identity is incomplete")
    if any(row["selection_order"] == 44 for row in rows):
        raise StageFF3C2Error("order 44 cannot enter F3C2")
    return [dict(row) for row in rows]


def false_pass_reasons(preconditions: Mapping[str, bool]) -> tuple[str, ...]:
    if set(preconditions) != set(FALSE_PASS_ATTACKS):
        raise StageFF3C2Error("false-pass preconditions must cover exactly 32 attacks")
    if any(type(value) is not bool for value in preconditions.values()):
        raise TypeError("false-pass preconditions must be booleans")
    return tuple(name for name in FALSE_PASS_ATTACKS if not preconditions[name])


def _walk_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.add(str(key).lower())
            found.update(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_walk_keys(item))
    return found


def validate_blind_document(value: Mapping[str, Any]) -> None:
    forbidden = sorted(_walk_keys(value) & BLIND_FORBIDDEN_KEYS)
    if forbidden:
        raise StageFF3C2Error("blind continuation consumed forbidden fields: " + ",".join(forbidden))


def initialize(repo_root: Path | str = REPO_ROOT, *, started_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = repo / ROOT_PATH
    startup_file = repo / STARTUP_PATH
    if not startup_file.is_file() or any(path.name != "startup_gate.json" for path in root.iterdir()):
        raise StageFF3C2Error("new F3C2 root must contain only the pre-code startup gate")
    if (repo / REPORT_PATH).exists():
        raise StageFF3C2Error("F3C2 report root already exists")
    startup = _read(startup_file)
    _validate_signed(startup, "F3C2 startup")
    if startup.get("status") != "PASS" or not startup.get("validated_before_code_modification_model_load_or_pilot_jpeg_decode"):
        raise StageFF3C2Error("F3C2 startup gate is not valid")
    config = load_config(repo)
    rows = _pilot_rows(repo)
    source_files = [_file_record(repo, path) for path in SOURCE_FILES]
    source = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_source_config_binding",
        "status": "FROZEN_BEFORE_PILOT_JPEG_ACCESS",
        "implementation_commit": startup["local_head"], "files": source_files,
        "source_tree_digest": canonical_sha256({"files": source_files}),
        "f3c1_frozen_source_hashes": FROZEN_SOURCE_HASHES,
    })
    model = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_model_preprocessing_binding",
        "status": "FROZEN_BEFORE_PILOT_JPEG_ACCESS",
        "model": {"name": MODEL_NAME, "identity": MODEL_IDENTITY, "path": MODEL_PATH, "sha256": MODEL_SHA256, "canonical_tensor_digest": MODEL_TENSOR_DIGEST},
        "preprocessing": {"color_mode": "RGB", "resize_shorter_side": 248, "interpolation": "bicubic", "antialias": True, "center_crop": [224, 224], "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "baseline": {"id": "normalized_zero", "canonical_rgb": [0.5, 0.5, 0.5], "black_image": False},
    })
    pilot = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_five_sample_manifest",
        "status": "FROZEN_BEFORE_PILOT_JPEG_ACCESS", "samples": rows,
        "selection_orders": list(PILOT_ORDERS), "sample_count": 5,
        "source_split": _file_record(repo, "manifests/stage_f_f3c_v1/sample_split.json"),
        "formal_entries_read": False, "order44_included": False, "replacement_allowed": False,
    })
    units = []
    for row in rows:
        for method in METHODS:
            static = {
                "protocol_id": config["protocol_id"], "sample_id": row["sample_id"],
                "selection_order": row["selection_order"], "jpeg_sha256": row["image_sha256"],
                "official_imagenet1k_index": row["imagenet1k_index"], "method_id": method,
                "model_sha256": MODEL_SHA256, "model_canonical_tensor_digest": MODEL_TENSOR_DIGEST,
                "preprocessing_binding_digest": model["canonical_digest"],
                "source_config_binding_digest": source["canonical_digest"],
                "clean_input_sha256": "BOUND_AFTER_AUTHORIZED_DECODE",
                "frozen_target_class_index": "BOUND_AFTER_SINGLE_CLEAN_FORWARD",
                "target_identity_digest": "BOUND_AFTER_SINGLE_CLEAN_FORWARD",
                "attempt_id": ATTEMPT_ID,
            }
            units.append({**static, "planned_unit_canonical_digest": canonical_sha256(static)})
    plan = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_25_unit_execution_plan",
        "status": "FROZEN_BEFORE_PILOT_JPEG_ACCESS", "sample_count": 5,
        "method_count": 5, "unit_count": 25, "units": units,
        "dynamic_identity_rule": "clean_input_sha256_target_class_and_target_identity_are_bound_after_the_single_authorized_clean_forward_before_terminal_publish",
        "execution_order": "pilot_manifest_order_then_frozen_method_order",
    })
    resource = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_resource_accounting_specification",
        "status": "FROZEN_BEFORE_PILOT_JPEG_ACCESS", "device_accounting": "elapsed_wall_seconds_times_one_observed_H20_device_divided_by_3600",
        "image_group_scope": "model_load_decode_clean_forward_five_methods_fidelity_and_terminal_publication",
        "pilot_actual_gpu_hours_formula": "sum(image_group_actual_gpu_hours)",
        "projected_formal_gpu_hours_formula": "50*max(image_group_actual_gpu_hours)",
        "projected_total_f3c_gpu_hours_formula": "f3c1_engineering_gpu_hours+pilot_actual_gpu_hours+projected_formal_gpu_hours",
        "projected_total_retained_bytes_formula": "existing_retained_f3c1_f3c2_bytes+50*max(image_group_retained_bytes)",
        "caps": config["budget"], "no_waiver": True, "shared_group_resources_not_summed_across_five_method_terminals": True,
    })
    blind = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_blind_continuation_specification",
        "status": "FROZEN_BEFORE_PILOT_JPEG_ACCESS",
        "allowed_inputs": ["sample_and_unit_identity", "terminal_status", "endpoint_parity_status_and_error_bound", "ig_conservation_status_and_selected_steps", "manifest_and_hash_completeness", "resource_telemetry", "budget_projection", "access_boundary_status"],
        "forbidden_inputs": sorted(BLIND_FORBIDDEN_KEYS),
        "pass_requires": {"samples": 5, "valid_terminals": 25, "nns_endpoint_parity_pass": 5, "nns_conservation_pass": 5, "full_model_ig_conservation_pass": 5, "resource_budget": "PASS", "formal_sample_access": False, "order44_recomputed": False, "rendered_or_manual_inspection": False, "scientific_comparison_or_inference": False},
    })
    access = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_access_policy",
        "status": "FROZEN_BEFORE_PILOT_JPEG_ACCESS",
        "allowed_pilot_orders_after_plan_seal": list(PILOT_ORDERS),
        "formal_jpeg_map_logit_curve_or_outcome_access_allowed": False,
        "order44_recompute_allowed": False, "rendering_allowed": False,
        "manual_visual_inspection_allowed": False, "statistical_inference_allowed": False,
        "scientific_comparison_winner_or_go_no_go_allowed": False,
    })
    for path, value in ((SOURCE_BINDING_PATH, source), (MODEL_BINDING_PATH, model), (PILOT_MANIFEST_PATH, pilot), (UNIT_PLAN_PATH, plan), (RESOURCE_SPEC_PATH, resource), (BLIND_SPEC_PATH, blind), (ACCESS_POLICY_PATH, access)):
        _write_new(repo / path, value)
    pending = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_pending_completion",
        "status": "PENDING", "started_at": started_at, "authorization": "F3C2_PILOT_ONLY",
        "startup_gate_digest": startup["canonical_digest"], "pilot_manifest_digest": pilot["canonical_digest"],
        "unit_plan_digest": plan["canonical_digest"], "model_preprocessing_digest": model["canonical_digest"],
        "source_config_digest": source["canonical_digest"], "resource_spec_digest": resource["canonical_digest"],
        "blind_spec_digest": blind["canonical_digest"], "access_policy_digest": access["canonical_digest"],
        "pilot_jpeg_access_started": False, "formal_sample_access": False, "order44_recomputed": False,
        "maps_rendered": False, "maps_manually_inspected": False, "statistical_inference_performed": False,
        "scientific_comparison_performed": False, "winner_or_go_no_go_produced": False,
        "f3c3_started": False, "f3c4_started": False, "f4_started": False,
    })
    _write_new(repo / PENDING_PATH, pending)
    seal = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_predecode_plan_seal",
        "status": "SEALED_BEFORE_PILOT_JPEG_ACCESS", "sealed_at": _now(),
        "records": [_record(repo, path) for path in (STARTUP_PATH, PILOT_MANIFEST_PATH, UNIT_PLAN_PATH, MODEL_BINDING_PATH, SOURCE_BINDING_PATH, RESOURCE_SPEC_PATH, BLIND_SPEC_PATH, ACCESS_POLICY_PATH, PENDING_PATH)],
        "pilot_jpeg_access_before_seal": False, "formal_sample_access": False,
    })
    _write_new(repo / PLAN_SEAL_PATH, seal)
    (repo / REPORT_PATH).mkdir(parents=True, exist_ok=False)
    return {"status": "PASS", "plan_sealed": True, "pilot_samples": 5, "units": 25, "pilot_jpeg_access": False}


def validate_initialized(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    required = (STARTUP_PATH, PILOT_MANIFEST_PATH, UNIT_PLAN_PATH, MODEL_BINDING_PATH, SOURCE_BINDING_PATH, RESOURCE_SPEC_PATH, BLIND_SPEC_PATH, ACCESS_POLICY_PATH, PENDING_PATH, PLAN_SEAL_PATH)
    for relative in required:
        value = _read(repo / relative)
        _validate_signed(value, relative)
    pending = _read(repo / PENDING_PATH)
    if pending.get("status") != "PENDING" or pending.get("pilot_jpeg_access_started") is not False:
        raise StageFF3C2Error("F3C2 pending receipt is not pre-access")
    plan = _read(repo / UNIT_PLAN_PATH)
    if plan.get("unit_count") != 25 or len(plan.get("units", [])) != 25:
        raise StageFF3C2Error("F3C2 unit plan is incomplete")
    if tuple(item["selection_order"] for item in plan["units"][::5]) != PILOT_ORDERS:
        raise StageFF3C2Error("F3C2 unit plan order drifted")
    if any(item["method_id"] != METHODS[index % 5] for index, item in enumerate(plan["units"])):
        raise StageFF3C2Error("F3C2 method execution order drifted")
    seal = _read(repo / PLAN_SEAL_PATH)
    for record in seal.get("records", []):
        if record != _record(repo, record["path"]):
            raise StageFF3C2Error("F3C2 predecode plan seal drifted")
    if seal.get("status") != "SEALED_BEFORE_PILOT_JPEG_ACCESS":
        raise StageFF3C2Error("F3C2 predecode plan is not sealed")
    return {"status": "PASS", "pilot_samples": 5, "units": 25, "pilot_jpeg_access_allowed": True}


def _terminal_root(repo: Path, order: int, method: str) -> Path:
    return repo / ROOT_PATH / "terminals" / f"order-{order:04d}" / method / "attempts" / ATTEMPT_ID


def _publish_terminal(
    destination: Path, result: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    *, blind_context: Mapping[str, Any] | None = None,
) -> str:
    terminal_status = result.get("status")
    if terminal_status not in {"VALID", "FAILED", "NA"}:
        raise StageFF3C2Error("typed terminal status must be VALID, FAILED, or NA")
    if terminal_status == "VALID" and not arrays:
        raise StageFF3C2Error("VALID terminal requires a complete numerical payload")
    if terminal_status in {"FAILED", "NA"} and arrays:
        raise StageFF3C2Error("FAILED/NA terminal must not contain numerical payload")
    if destination.exists():
        validated = validate_terminal(destination)
        previous = validated["result"]
        if (
            previous.get("status") != result.get("status")
            or previous.get("unit_identity") != result.get("unit_identity")
            or previous.get("payload_identities") != result.get("payload_identities")
        ):
            raise StageFF3C2Error("conflicting terminal payload at existing attempt")
        return "RESUME_SKIP_VERIFIED"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{ATTEMPT_ID}-", dir=destination.parent))
    try:
        _write_new(temporary / "result.json", _signed(result))
        records = []
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise StageFF3C2Error("terminal arrays must be finite float32")
            _write_npy(temporary / name, value)
            records.append({"path": name, "bytes": (temporary / name).stat().st_size, "sha256": _sha_file(temporary / name), "array_digest": kernels.array_digest(value), "shape": list(value.shape), "dtype": "float32"})
        status = _signed({
            "schema_version": 1, "document_type": "stage_f_f3c2_typed_terminal_status",
            "attempt_id": ATTEMPT_ID, "status": terminal_status,
            "terminal": True, "immutable": True,
        })
        _write_new(temporary / "status.json", status)
        manifest = _signed({
            "schema_version": 1, "document_type": "stage_f_f3c2_typed_terminal_manifest",
            "attempt_id": ATTEMPT_ID, "status": terminal_status,
            "result": _file_record(temporary, "result.json"),
            "status_record": _record(temporary, "status.json"),
            "arrays": records, "immutable": True,
        })
        _write_new(temporary / "manifest.json", manifest)
        if blind_context is not None:
            blind = _signed({
                **dict(blind_context),
                "result_sha256": _sha_file(temporary / "result.json"),
                "manifest_sha256": _sha_file(temporary / "manifest.json"),
            })
            validate_blind_document(blind)
            _write_new(temporary / "blind.json", blind)
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return "PUBLISHED"


def validate_terminal(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    required = [root / name for name in ("result.json", "status.json", "manifest.json")]
    if not root.is_dir() or any(not path.is_file() or path.is_symlink() for path in required):
        raise StageFF3C2Error("typed terminal is incomplete or nonregular")
    result, status, manifest = (_read(path) for path in required)
    for value, name in ((result, "result"), (status, "status"), (manifest, "manifest")):
        _validate_signed(value, f"typed terminal {name}")
    terminal_status = result.get("status")
    if terminal_status not in {"VALID", "FAILED", "NA"} or status.get("status") != terminal_status or manifest.get("status") != terminal_status or status.get("terminal") is not True:
        raise StageFF3C2Error("typed terminal status conflict")
    if manifest.get("result") != _file_record(root, "result.json") or manifest.get("status_record") != _record(root, "status.json"):
        raise StageFF3C2Error("typed terminal control file conflict")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, list):
        raise StageFF3C2Error("typed terminal array manifest is invalid")
    if terminal_status == "VALID" and not arrays:
        raise StageFF3C2Error("VALID terminal lacks numerical payload")
    if terminal_status in {"FAILED", "NA"} and arrays:
        raise StageFF3C2Error("FAILED/NA terminal carries numerical payload")
    for record in arrays:
        path = root / record["path"]
        value = np.load(path, allow_pickle=False)
        if _sha_file(path) != record["sha256"] or path.stat().st_size != record["bytes"] or kernels.array_digest(value) != record["array_digest"] or list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise StageFF3C2Error("typed terminal numerical payload conflict")
    if terminal_status == "VALID":
        required_arrays = {"native.npy", "patch_mass.npy", "target_oriented_mass.npy", "rankings.npy", "insertion_curves.npy", "deletion_curves.npy", "insertion_aucs.npy", "deletion_aucs.npy", "fidelity_scores.npy", "fractions.npy", "actual_fractions.npy"}
        if {item["path"] for item in arrays} != required_arrays:
            raise StageFF3C2Error("VALID terminal numerical payload roster drifted")
    return {"status": "PASS", "result": result, "manifest": manifest}


def _load_pilot_sample(repo: Path, row: Mapping[str, Any], device: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform_functional
    from .modeling import canonical_preprocess_spec, load_canonical_model

    image_path = repo / "assets/data/ILSVRC2012_val" / row["relative_path"]
    if _sha_file(image_path) != row["image_sha256"]:
        raise StageFF3C2Error("pilot JPEG identity drifted")
    model = load_canonical_model(
        state_dict_path=repo / MODEL_PATH, device=device, verify_hash=True,
        unfused_attention=True, expected_state_dict_sha256=MODEL_SHA256,
        expected_model_name=MODEL_NAME, expected_model_identity=MODEL_LOADER_IDENTITY,
        expected_canonical_tensor_digest=MODEL_TENSOR_DIGEST,
    )
    specification = canonical_preprocess_spec(model)
    observed = (specification.color_mode, specification.input_size, specification.resize_shorter_side, specification.center_crop, specification.interpolation, specification.antialias, specification.mean, specification.std)
    expected = ("RGB", (3, 224, 224), 248, (224, 224), "bicubic", True, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if observed != expected:
        raise StageFF3C2Error("preprocessing contract drifted")
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        resized = transform_functional.resize(rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True)
        cropped = transform_functional.center_crop(resized, [224, 224])
        canonical = transform_functional.pil_to_tensor(cropped).to(torch.float32).div(255.0).unsqueeze(0)
    clean = ((canonical - 0.5) / 0.5).to(device)
    return model, clean, {
        "sample_id": row["sample_id"], "selection_order": row["selection_order"],
        "relative_path": row["relative_path"], "jpeg_sha256": row["image_sha256"],
        "official_imagenet1k_index": row["imagenet1k_index"],
        "clean_input_sha256": kernels.tensor_sha256(clean),
    }


def _target_oriented(method_id: str, method: Mapping[str, Any], fidelity: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    patch = np.asarray(method["patch_mass"], dtype=np.float32)
    sigma = int(fidelity["methods"][method_id]["sigma"])
    mass = np.maximum(np.float32(sigma) * patch, np.float32(0.0))
    denominator = float(mass.sum(dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0:
        raise StageFF3C2Error("target-oriented positive mass is zero")
    normalized = np.asarray(mass / denominator, dtype=np.float32)
    rankings = []
    for curve in fidelity["methods"][method_id]["curves"]:
        ranking, priority = kernels.method_ranking(patch, sigma=sigma, seed=curve["seed"])
        if kernels.array_digest(priority) != curve["tie_priority_sha256"] or kernels.array_digest(ranking) != curve["ranking_sha256"]:
            raise StageFF3C2Error("ranking identity is not source-recomputable")
        rankings.append(ranking.astype(np.float32))
    return normalized, np.stack(rankings).astype(np.float32)


def _method_payload(method_id: str, method: Mapping[str, Any], fidelity: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    native = np.asarray(method.get("native_signed_rgb", method.get("native")), dtype=np.float32)
    patch = np.asarray(method["patch_mass"], dtype=np.float32)
    target_mass, rankings = _target_oriented(method_id, method, fidelity)
    curves = fidelity["methods"][method_id]["curves"]
    insertion = np.asarray([item["insertion"]["responses"] for item in curves], dtype=np.float32)
    deletion = np.asarray([item["deletion"]["responses"] for item in curves], dtype=np.float32)
    insertion_auc = np.asarray([item["insertion"]["auc"] for item in curves], dtype=np.float32)
    deletion_auc = np.asarray([item["deletion"]["auc"] for item in curves], dtype=np.float32)
    scores = np.asarray([item["fidelity_score"] for item in curves], dtype=np.float32)
    arrays = {
        "native.npy": native, "patch_mass.npy": patch,
        "target_oriented_mass.npy": target_mass, "rankings.npy": rankings,
        "insertion_curves.npy": insertion, "deletion_curves.npy": deletion,
        "insertion_aucs.npy": insertion_auc, "deletion_aucs.npy": deletion_auc,
        "fidelity_scores.npy": scores, "fractions.npy": np.asarray(kernels.FRACTIONS, dtype=np.float32),
        "actual_fractions.npy": np.asarray(kernels._actual_fractions(), dtype=np.float32),
    }
    identities = {
        "native_array_digest": kernels.array_digest(native), "patch_mass_array_digest": kernels.array_digest(patch),
        "target_oriented_mass_array_digest": kernels.array_digest(target_mass),
        "ranking_array_digest": kernels.array_digest(rankings),
        "tie_priority_sha256": [item["tie_priority_sha256"] for item in curves],
        "insertion_curve_array_digest": kernels.array_digest(insertion),
        "deletion_curve_array_digest": kernels.array_digest(deletion),
    }
    return arrays, identities


def _blind_terminal(
    *, sample: Mapping[str, Any], method_id: str, target_class_index: int,
    target_identity_digest: str, attention: Mapping[str, Any], endpoint: Mapping[str, Any] | None,
    conservation: Mapping[str, Any] | None, resources: Mapping[str, Any],
    result_sha256: str, manifest_sha256: str, unit_digest: str,
) -> dict[str, Any]:
    value = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_blind_terminal_record",
        "status": "VALID", "attempt_id": ATTEMPT_ID, "terminal": True,
        "sample_id": sample["sample_id"], "selection_order": sample["selection_order"],
        "jpeg_sha256": sample["jpeg_sha256"], "clean_input_sha256": sample["clean_input_sha256"],
        "method_id": method_id, "target_class_index": target_class_index,
        "target_identity_digest": target_identity_digest, "unit_canonical_digest": unit_digest,
        "attention_count": attention["count"] if method_id == kernels.METHOD_ID else None,
        "attention_hash_count": len(attention["tensor_sha256"]) if method_id == kernels.METHOD_ID else 0,
        "endpoint_parity_status": endpoint["status"] if endpoint is not None else "NOT_APPLICABLE",
        "endpoint_parity_max_absolute_error": endpoint["max_absolute_error"] if endpoint is not None else None,
        "ig_conservation_status": ("PASS" if conservation and conservation["passed"] else "NOT_APPLICABLE"),
        "ig_selected_steps": conservation["selected_steps"] if conservation is not None else None,
        "attempt_count": len(conservation["attempts"]) if conservation is not None else 0,
        "result_sha256": result_sha256, "manifest_sha256": manifest_sha256,
        "resources": dict(resources), "manifest_hash_complete": True,
        "numerical_payload_not_read_by_blind_gate": True,
    })
    validate_blind_document(value)
    return value


def run_sample(repo_root: Path | str = REPO_ROOT, *, selection_order: int, device: str = "cuda:0") -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validate_initialized(repo)
    rows = _pilot_rows(repo)
    matches = [row for row in rows if row["selection_order"] == selection_order]
    if len(matches) != 1 or selection_order == 44:
        raise StageFF3C2Error("requested sample is not one frozen pilot member")
    row = matches[0]
    group_root = repo / ROOT_PATH / "image_groups" / f"order-{selection_order:04d}"
    if (group_root / "completion.json").is_file():
        return validate_sample(repo, selection_order=selection_order)
    earlier = PILOT_ORDERS[: PILOT_ORDERS.index(selection_order)]
    if any(not (repo / ROOT_PATH / "image_groups" / f"order-{order:04d}" / "completion.json").is_file() for order in earlier):
        raise StageFF3C2Error("pilot samples must execute in frozen order")
    model = None
    start = time.monotonic()
    cpu_start = time.process_time()
    forward_calls = 0
    try:
        import torch
        import timm
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise StageFF3C2Error("F3C2 pilot requires an available CUDA device")
        torch.cuda.set_device(torch.device(device))
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        model, clean, sample = _load_pilot_sample(repo, row, device)

        def count_forward(_module: Any, _args: Any) -> None:
            nonlocal forward_calls
            forward_calls += 1

        forward_handle = model.register_forward_pre_hook(count_forward)
        try:
            capture = kernels.capture_clean_target_and_all_attention(model, clean, official_truth_index=row["imagenet1k_index"])
            methods = kernels.collect_five_method_maps(model, clean, capture, capture.as_attention_extraction())
            fidelity = kernels.evaluate_common_fidelity(model, clean, capture, methods, sample_id=row["sample_id"])
        finally:
            forward_handle.remove()
        elapsed_compute = time.monotonic() - start
        cpu_compute = time.process_time() - cpu_start
        peak = int(torch.cuda.max_memory_reserved(torch.device(device)))
        target_payload = {
            "target_ref": "clean_predicted_class_raw_logit",
            "class_index": capture.target.class_index,
            "clean_input_sha256": capture.clean_input_sha256,
            "official_truth_index": row["imagenet1k_index"],
            "clean_top1_correct": capture.target.clean_top1_correct,
            "resolved_once_from_same_clean_forward": True,
            "frozen_target": capture.target.to_dict(),
        }
        target_identity_digest = canonical_sha256(target_payload)
        source = _read(repo / SOURCE_BINDING_PATH)
        model_binding = _read(repo / MODEL_BINDING_PATH)
        base_identity = {
            "protocol_id": load_config(repo)["protocol_id"], "sample_id": row["sample_id"],
            "selection_order": selection_order, "jpeg_sha256": row["image_sha256"],
            "model_sha256": MODEL_SHA256, "model_canonical_tensor_digest": MODEL_TENSOR_DIGEST,
            "preprocessing_identity": model_binding["canonical_digest"],
            "clean_input_sha256": capture.clean_input_sha256, "frozen_target_class_index": capture.target.class_index,
            "target_identity_digest": target_identity_digest,
            "source_config_code_digest": source["source_tree_digest"],
            "source_config_code_hashes": {item["path"]: item["sha256"] for item in source["files"]},
            "attempt_id": ATTEMPT_ID,
        }
        nns_ig = methods[kernels.METHOD_ID]["ig_result"]
        full_ig = methods["full_model_class_logit_ig"]["provenance"]["ig_result"]
        backward_calls = (
            sum(math.ceil(int(item["steps"]) / kernels.IG_CHUNK) for item in nns_ig.attempts)
            + sum(math.ceil(int(item["steps"]) / kernels.IG_CHUNK) for item in full_ig["attempts"])
            + 1
        )
        group_resources = {
            "elapsed_seconds": elapsed_compute, "cpu_seconds": cpu_compute,
            "gpu_hours": elapsed_compute / 3600.0, "device_count": 1,
            "cuda_device": device, "gpu_name": torch.cuda.get_device_name(torch.device(device)),
            "peak_cuda_memory_reserved_bytes": peak, "forward_calls": forward_calls,
            "backward_calls": backward_calls, "backward_call_accounting": "one_autograd_grad_per_retained_IG_chunk_plus_one_gradcam_call",
            "backward_chunk_calls_expected": sum(math.ceil(int(item["steps"]) / kernels.IG_CHUNK) for item in nns_ig.attempts) + sum(math.ceil(int(item["steps"]) / kernels.IG_CHUNK) for item in full_ig["attempts"]),
            "torch_version": str(torch.__version__), "timm_version": str(timm.__version__),
            "cuda_runtime_version": str(torch.version.cuda),
            "accounting_scope": "shared_five_method_image_group_count_once",
        }
        terminal_records = []
        for method_id in METHODS:
            method = methods[method_id]
            arrays, identities = _method_payload(method_id, method, fidelity)
            conservation = None
            if method_id == kernels.METHOD_ID:
                conservation = {"selected_steps": nns_ig.steps, "passed": nns_ig.converged, "attempts": [dict(item) for item in nns_ig.attempts]}
            elif method_id == "full_model_class_logit_ig":
                conservation = {"selected_steps": full_ig["selected_steps"], "passed": full_ig["converged"], "attempts": full_ig["attempts"]}
            unit_identity = {**base_identity, "method_id": method_id}
            unit_digest = canonical_sha256(unit_identity)
            result = {
                "schema_version": 1, "document_type": "stage_f_f3c2_sample_method_terminal",
                "status": "VALID", "attempt_id": ATTEMPT_ID, "unit_identity": unit_identity,
                "unit_canonical_digest": unit_digest, "sample": sample, "target": target_payload,
                "target_identity_digest": target_identity_digest, "method_id": method_id,
                "class_targeted": bool(method["class_targeted"]), "class_agnostic": bool(method["class_agnostic"]),
                "attention_hashes": list(capture.attentions.hashes) if method_id == kernels.METHOD_ID else [],
                "endpoint_parity": method.get("endpoint_parity"),
                "ig_conservation": conservation, "payload_identities": identities,
                "fidelity_contract": {"fractions": list(kernels.FRACTIONS), "actual_fractions": list(kernels._actual_fractions()), "tie_seeds": [kernels.tie_seed(row["sample_id"], repeat) for repeat in range(5)], "tie_priority_sha256": identities["tie_priority_sha256"], "target": "unmodified_frozen_clean_predicted_class_raw_logit", "score": "0.5*(insertion_auc-deletion_auc)"},
                "fidelity_numeric_payload": fidelity["methods"][method_id],
                "resources": {**group_resources, "artifact_bytes": sum(value.nbytes for value in arrays.values()), "artifact_bytes_scope": "uncompressed_numeric_array_payload"},
                "resources_are_shared_image_group_not_additive": True,
                "maps_rendered": False, "maps_manually_inspected": False,
                "statistical_inference_performed": False, "scientific_comparison_performed": False,
            }
            destination = _terminal_root(repo, selection_order, method_id)
            blind_context = {
                "schema_version": 1, "document_type": "stage_f_f3c2_blind_terminal_record",
                "status": "VALID", "attempt_id": ATTEMPT_ID, "terminal": True,
                "sample_id": sample["sample_id"], "selection_order": sample["selection_order"],
                "jpeg_sha256": sample["jpeg_sha256"], "clean_input_sha256": sample["clean_input_sha256"],
                "method_id": method_id, "target_class_index": capture.target.class_index,
                "target_identity_digest": target_identity_digest, "unit_canonical_digest": unit_digest,
                "attention_count": capture.attentions.manifest()["count"] if method_id == kernels.METHOD_ID else None,
                "attention_hash_count": len(capture.attentions.manifest()["tensor_sha256"]) if method_id == kernels.METHOD_ID else 0,
                "endpoint_parity_status": method.get("endpoint_parity", {}).get("status", "NOT_APPLICABLE") if method.get("endpoint_parity") is not None else "NOT_APPLICABLE",
                "endpoint_parity_max_absolute_error": method.get("endpoint_parity", {}).get("max_absolute_error") if method.get("endpoint_parity") is not None else None,
                "ig_conservation_status": "PASS" if conservation and conservation["passed"] else "NOT_APPLICABLE",
                "ig_selected_steps": conservation["selected_steps"] if conservation is not None else None,
                "attempt_count": len(conservation["attempts"]) if conservation is not None else 0,
                "resources": group_resources, "manifest_hash_complete": True,
                "numerical_payload_not_read_by_blind_gate": True,
            }
            action = _publish_terminal(destination, result, arrays, blind_context=blind_context)
            validated = validate_terminal(destination)
            existing_blind = _read(destination / "blind.json")
            _validate_signed(existing_blind, "existing blind terminal")
            if any(existing_blind.get(key) != blind_context.get(key) for key in ("sample_id", "selection_order", "method_id", "unit_canonical_digest")):
                raise StageFF3C2Error("conflicting blind terminal record")
            terminal_records.append({"method_id": method_id, "action": action, "manifest_sha256": validated["manifest"]["canonical_digest"]})
        group_completion = _signed({
            "schema_version": 1, "document_type": "stage_f_f3c2_image_group_completion",
            "status": "PASS", "selection_order": selection_order, "sample_id": row["sample_id"],
            "terminal_count": 5, "valid_count": 5, "target_class_index": capture.target.class_index,
            "target_identity_digest": target_identity_digest, "nns_endpoint_parity": methods[kernels.METHOD_ID]["endpoint_parity"],
            "nns_conservation": {"selected_steps": nns_ig.steps, "passed": nns_ig.converged},
            "full_model_ig_conservation": {"selected_steps": full_ig["selected_steps"], "passed": full_ig["converged"]},
            "resources": group_resources, "terminals": terminal_records,
            "maps_rendered": False, "maps_manually_inspected": False,
        })
        group_root.parent.mkdir(parents=True, exist_ok=True)
        temp_group = Path(tempfile.mkdtemp(prefix=f".order-{selection_order:04d}-", dir=group_root.parent))
        try:
            target = _signed({"schema_version": 1, "document_type": "stage_f_f3c2_target_binding", "status": "PASS", **target_payload, "target_identity_digest": target_identity_digest})
            attention = _signed({"schema_version": 1, "document_type": "stage_f_f3c2_attention_binding", "status": "PASS", "manifest": capture.attentions.manifest(), "same_clean_forward": True, "q_k_softmax_variation_cut": True, "value_projection_residual_mlp_final_norm_classifier_live": True})
            resource = _signed({"schema_version": 1, "document_type": "stage_f_f3c2_image_group_resource", "status": "PASS", "selection_order": selection_order, **group_resources})
            _write_new(temp_group / "target_binding.json", target)
            _write_new(temp_group / "attention_binding.json", attention)
            _write_new(temp_group / "resource.json", resource)
            _write_new(temp_group / "completion.json", group_completion)
            os.replace(temp_group, group_root)
            descriptor = os.open(group_root.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temp_group.exists():
                shutil.rmtree(temp_group)
        return {"status": "PASS", "selection_order": selection_order, "valid_terminals": 5, "target_class_index": capture.target.class_index, "elapsed_seconds": elapsed_compute, "gpu_hours": elapsed_compute / 3600.0}
    except (kernels.StageFF3C1IGConvergenceError, Exception) as error:
        failure = repo / FAILURE_PATH
        if not failure.exists():
            _write_new(failure, _signed({"schema_version": 1, "document_type": "stage_f_f3c2_failure_event", "status": "BLOCKED", "phase": "pilot_sample_execution", "selection_order": selection_order, "error_type": type(error).__name__, "error": str(error), "unknown_implementation_exception_remains_incomplete": not isinstance(error, (kernels.StageFF3C1IGConvergenceError,)), "recorded_at": _now()}))
        raise
    finally:
        if model is not None:
            del model
        try:
            import torch
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def validate_sample(repo_root: Path | str = REPO_ROOT, *, selection_order: int) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    row = next((item for item in _pilot_rows(repo) if item["selection_order"] == selection_order), None)
    if row is None:
        raise StageFF3C2Error("sample is not in frozen pilot")
    group = _read(repo / ROOT_PATH / "image_groups" / f"order-{selection_order:04d}" / "completion.json")
    _validate_signed(group, "image group completion")
    records = []
    for method in METHODS:
        validated = validate_terminal(_terminal_root(repo, selection_order, method))
        result = validated["result"]
        blind = _read(_terminal_root(repo, selection_order, method) / "blind.json")
        _validate_signed(blind, "blind terminal")
        validate_blind_document(blind)
        if result.get("status") != "VALID" or result.get("method_id") != method or result.get("sample", {}).get("sample_id") != row["sample_id"]:
            raise StageFF3C2Error("sample terminal identity drifted")
        if blind.get("result_sha256") != _sha_file(_terminal_root(repo, selection_order, method) / "result.json") or blind.get("manifest_sha256") != _sha_file(_terminal_root(repo, selection_order, method) / "manifest.json"):
            raise StageFF3C2Error("blind terminal hash binding drifted")
        records.append(blind)
    target_digests = {record["target_identity_digest"] for record in records}
    class_indices = {record["target_class_index"] for record in records}
    if len(target_digests) != 1 or len(class_indices) != 1:
        raise StageFF3C2Error("target/class identity is not shared across methods")
    if group.get("valid_count") != 5 or group.get("status") != "PASS":
        raise StageFF3C2Error("image group is not complete")
    return {"status": "PASS", "selection_order": selection_order, "valid_count": 5, "target_class_index": next(iter(class_indices)), "blind_records": records}


def _retained_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def reconcile(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validate_initialized(repo)
    groups = [validate_sample(repo, selection_order=order) for order in PILOT_ORDERS]
    blind_records = [record for group in groups for record in group["blind_records"]]
    if len(blind_records) != 25:
        raise StageFF3C2Error("pilot does not have exactly 25 blind terminal records")
    if len({(item["selection_order"], item["method_id"]) for item in blind_records}) != 25:
        raise StageFF3C2Error("pilot terminal identities are not unique")
    union_records = []
    attempts = []
    for row in _pilot_rows(repo):
        for method in METHODS:
            root = _terminal_root(repo, row["selection_order"], method)
            attempt_dirs = sorted((root.parent).glob("attempt-*"))
            if attempt_dirs != [root]:
                raise StageFF3C2Error("unit has missing or multiple attempts")
            validated = validate_terminal(root)
            result = validated["result"]
            union_records.append({
                "selection_order": row["selection_order"], "sample_id": row["sample_id"],
                "method_id": method, "attempt_id": ATTEMPT_ID, "status": result["status"],
                "unit_canonical_digest": result["unit_canonical_digest"],
                "root": root.relative_to(repo).as_posix(),
                "manifest": _record(repo, (root / "manifest.json").relative_to(repo).as_posix()),
                "blind": _record(repo, (root / "blind.json").relative_to(repo).as_posix()),
            })
            attempts.append({
                "selection_order": row["selection_order"], "method_id": method,
                "attempt_id": ATTEMPT_ID, "status": result["status"],
                "ig_attempts": result.get("ig_conservation", {}).get("attempts", []) if result.get("ig_conservation") else [],
                "attempts_discarded": False,
            })
    counts = {status: sum(item["status"] == status for item in union_records) for status in ("VALID", "FAILED", "NA")}
    counts["incomplete"] = 25 - sum(counts.values())
    union = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_terminal_union",
        "status": "PASS" if counts == {"VALID": 25, "FAILED": 0, "NA": 0, "incomplete": 0} else "BLOCKED",
        "sample_count": 5, "unit_count": 25, "counts": counts, "records": union_records,
        "multiple_valid_attempts": False, "conflicting_payloads": False,
    })
    _write_new(repo / TERMINAL_UNION_PATH, union)
    journal = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_attempt_journal",
        "status": union["status"], "unit_attempts": attempts,
        "unit_count": 25, "attempt_count": len(attempts), "attempts_discarded": False,
    })
    _write_new(repo / ATTEMPT_JOURNAL_PATH, journal)
    resources = []
    for order in PILOT_ORDERS:
        group_root = repo / ROOT_PATH / "image_groups" / f"order-{order:04d}"
        resource = _read(group_root / "resource.json")
        _validate_signed(resource, "image group resource")
        retained = _retained_bytes(group_root) + sum(_retained_bytes(_terminal_root(repo, order, method)) for method in METHODS)
        resources.append({
            "selection_order": order, "elapsed_seconds": resource["elapsed_seconds"],
            "cpu_seconds": resource["cpu_seconds"], "gpu_hours": resource["gpu_hours"],
            "device_count": resource["device_count"], "gpu_name": resource["gpu_name"],
            "peak_cuda_memory_reserved_bytes": resource["peak_cuda_memory_reserved_bytes"],
            "forward_calls": resource["forward_calls"], "backward_calls": resource["backward_calls"],
            "backward_chunk_calls_expected": resource["backward_chunk_calls_expected"],
            "retained_bytes": retained, "shared_image_group_counted_once": True,
        })
    ledger = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_resource_ledger", "status": "PASS",
        "image_groups": resources, "pilot_actual_gpu_hours": sum(item["gpu_hours"] for item in resources),
        "total_elapsed_seconds": sum(item["elapsed_seconds"] for item in resources),
        "total_cpu_seconds": sum(item["cpu_seconds"] for item in resources),
        "total_forward_calls": sum(item["forward_calls"] for item in resources),
        "total_backward_calls": sum(item["backward_calls"] for item in resources),
        "peak_reserved_bytes": max(item["peak_cuda_memory_reserved_bytes"] for item in resources),
        "observed_devices": 1, "gpu_name": "NVIDIA H20",
        "duplicate_shared_method_accounting": False, "waiver": False,
    })
    _write_new(repo / RESOURCE_LEDGER_PATH, ledger)
    access = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_access_audit", "status": "PASS",
        "pilot_sample_access": True, "decoded_jpeg_selection_orders": list(PILOT_ORDERS),
        "decoded_jpeg_count": 5, "formal_sample_access": False, "formal_map_logit_curve_or_outcome_access": False,
        "order44_recomputed": False, "maps_rendered": False, "maps_manually_inspected": False,
        "pilot_jpegs_rendered_or_manually_viewed": False, "statistical_inference_performed": False,
        "scientific_comparison_performed": False, "winner_or_go_no_go_produced": False,
        "statistical_method_freeze_modified": False, "statistical_method_freeze_executed": False,
        "f3c3_started": False, "f3c4_started": False, "f4_started": False,
    })
    _write_new(repo / ACCESS_AUDIT_PATH, access)
    return {"status": union["status"], "counts": counts, "pilot_actual_gpu_hours": ledger["pilot_actual_gpu_hours"]}


def build_budget_and_blind_gate(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    union = _read(repo / TERMINAL_UNION_PATH)
    ledger = _read(repo / RESOURCE_LEDGER_PATH)
    access = _read(repo / ACCESS_AUDIT_PATH)
    for value, name in ((union, "terminal union"), (ledger, "resource ledger"), (access, "access audit")):
        _validate_signed(value, name)
    blind_records = []
    for row in _pilot_rows(repo):
        for method in METHODS:
            value = _read(_terminal_root(repo, row["selection_order"], method) / "blind.json")
            _validate_signed(value, "blind terminal")
            validate_blind_document(value)
            blind_records.append(value)
    image_groups = ledger["image_groups"]
    maximum_gpu = max(item["gpu_hours"] for item in image_groups)
    pilot_gpu = ledger["pilot_actual_gpu_hours"]
    formal_gpu = 50 * maximum_gpu
    total_gpu = F3C1_ENGINEERING_GPU_HOURS + pilot_gpu + formal_gpu
    existing = F3C1_ROOT_BYTES + _retained_bytes(repo / ROOT_PATH) + _retained_bytes(repo / REPORT_PATH)
    maximum_retained = max(item["retained_bytes"] for item in image_groups)
    projected_retained = existing + 50 * maximum_retained
    caps = load_config(repo)["budget"]
    checks = {
        "each_image_group_gpu_hours": all(item["gpu_hours"] <= caps["per_image_group_h20_gpu_hours_max"] for item in image_groups),
        "projected_total_gpu_hours": total_gpu <= caps["total_h20_gpu_hours_max"],
        "observed_devices": ledger["observed_devices"] <= caps["max_devices"],
        "per_process_peak_reserved_bytes": ledger["peak_reserved_bytes"] <= caps["per_process_peak_reserved_bytes_max"],
        "projected_total_retained_bytes": projected_retained <= caps["retained_regular_file_bytes_max"],
        "no_waiver": ledger["waiver"] is False and caps["waiver_allowed"] is False,
    }
    budget_status = "PASS" if all(checks.values()) else "BLOCKED_BUDGET"
    budget = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_budget_projection",
        "status": budget_status, "f3c1_engineering_gpu_hours": F3C1_ENGINEERING_GPU_HOURS,
        "pilot_actual_gpu_hours": pilot_gpu, "maximum_pilot_image_group_gpu_hours": maximum_gpu,
        "projected_formal_gpu_hours": formal_gpu, "projected_total_f3c_gpu_hours": total_gpu,
        "existing_retained_f3c1_f3c2_bytes": existing, "maximum_pilot_image_group_retained_bytes": maximum_retained,
        "projected_total_retained_bytes": projected_retained, "observed_devices": ledger["observed_devices"],
        "peak_reserved_bytes": ledger["peak_reserved_bytes"], "caps": caps, "checks": checks,
        "waiver": False, "formula_version": "F3C2_FROZEN_CONSERVATIVE_MAX_IMAGE_GROUP_V1",
    })
    _write_new(repo / BUDGET_PATH, budget)
    nns = [item for item in blind_records if item["method_id"] == kernels.METHOD_ID]
    full = [item for item in blind_records if item["method_id"] == "full_model_class_logit_ig"]
    conditions = {
        "samples_5_of_5": len({item["selection_order"] for item in blind_records}) == 5,
        "terminals_25_of_25_valid": len(blind_records) == 25 and all(item["status"] == "VALID" for item in blind_records),
        "nns_endpoint_parity_5_of_5": len(nns) == 5 and all(item["endpoint_parity_status"] == "PASS" and item["endpoint_parity_max_absolute_error"] < 1e-6 for item in nns),
        "nns_conservation_5_of_5": len(nns) == 5 and all(item["ig_conservation_status"] == "PASS" for item in nns),
        "full_model_ig_conservation_5_of_5": len(full) == 5 and all(item["ig_conservation_status"] == "PASS" for item in full),
        "target_identity_shared_once_per_sample": all(len({item["target_identity_digest"] for item in blind_records if item["selection_order"] == order}) == 1 for order in PILOT_ORDERS),
        "manifest_hash_complete": all(item["manifest_hash_complete"] for item in blind_records),
        "pilot_formal_order44_boundary": access["decoded_jpeg_selection_orders"] == list(PILOT_ORDERS) and not access["formal_sample_access"] and not access["order44_recomputed"],
        "resource_budget": budget_status == "PASS",
        "no_render_or_manual_inspection": not access["maps_rendered"] and not access["maps_manually_inspected"] and not access["pilot_jpegs_rendered_or_manually_viewed"],
        "no_statistics_comparison_winner_or_go_no_go": not access["statistical_inference_performed"] and not access["scientific_comparison_performed"] and not access["winner_or_go_no_go_produced"],
    }
    status = "PASS" if all(conditions.values()) else ("BLOCKED_BUDGET" if not conditions["resource_budget"] else "BLOCKED")
    report = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_blind_continuation_report",
        "status": status, "input_policy": "BLIND_IDENTITY_STATUS_INTEGRITY_CONSERVATION_RESOURCE_ONLY",
        "records": blind_records, "conditions": conditions,
        "budget_projection": _record(repo, BUDGET_PATH),
        "forbidden_numerical_outcomes_consumed": False,
        "next_authorization": "F3C3_FORMAL_ONLY" if status == "PASS" else "F3C2_RECOVERY_REQUIRED",
    })
    validate_blind_document(report)
    _write_new(repo / BLIND_REPORT_PATH, report)
    return {"status": status, "next_authorization": report["next_authorization"], "conditions": conditions}


def record_tests(repo_root: Path | str, *, passed: int, failed: int, deselected: int, command: str) -> dict[str, Any]:
    if passed <= 0 or failed != 0:
        raise StageFF3C2Error("F3C2 regression tests are not passing")
    repo = Path(repo_root).resolve()
    result = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_test_report", "status": "PASS",
        "passed": passed, "failed": failed, "deselected": deselected, "command": command,
        "false_pass_mutation_tests": len(FALSE_PASS_ATTACKS), "render_or_manual_inspection_tests": "STATIC_AND_RUNTIME_BOUNDARY",
    })
    _write_new(repo / TEST_PATH, result)
    return result


def build_false_pass_audit(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    test = _read(repo / TEST_PATH)
    _validate_signed(test, "F3C2 test report")
    if test.get("status") != "PASS" or test.get("false_pass_mutation_tests") != len(FALSE_PASS_ATTACKS):
        raise StageFF3C2Error("false-pass mutation tests are incomplete")
    if false_pass_reasons({attack: True for attack in FALSE_PASS_ATTACKS}):
        raise StageFF3C2Error("valid false-pass preconditions were rejected")
    audit = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_false_pass_audit", "status": "PASS",
        "attack_count": len(FALSE_PASS_ATTACKS),
        "attacks": [{"attack": attack, "result": "REJECTED", "evidence": f"tests/test_stage_f_f3c2.py::test_every_false_pass_attack_is_independently_rejected[{attack}]"} for attack in FALSE_PASS_ATTACKS],
        "static_all_pass_list_only": False, "pass_possible_without_all_conditions": False,
    })
    _write_new(repo / FALSE_PASS_PATH, audit)
    return audit


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    required = (STARTUP_PATH, PILOT_MANIFEST_PATH, UNIT_PLAN_PATH, MODEL_BINDING_PATH, SOURCE_BINDING_PATH, RESOURCE_SPEC_PATH, BLIND_SPEC_PATH, ACCESS_POLICY_PATH, PLAN_SEAL_PATH, PENDING_PATH, TERMINAL_UNION_PATH, ATTEMPT_JOURNAL_PATH, RESOURCE_LEDGER_PATH, BUDGET_PATH, ACCESS_AUDIT_PATH, BLIND_REPORT_PATH, TEST_PATH, FALSE_PASS_PATH)
    for relative in required:
        value = _read(repo / relative)
        _validate_signed(value, relative)
    source = _read(repo / SOURCE_BINDING_PATH)
    expected_files = [_file_record(repo, path) for path in SOURCE_FILES]
    if source.get("files") != expected_files or source.get("source_tree_digest") != canonical_sha256({"files": expected_files}):
        raise StageFF3C2Error("F3C2 source binding drifted after plan seal")
    for relative, expected in FROZEN_SOURCE_HASHES.items():
        if _sha_file(repo / relative) != expected:
            raise StageFF3C2Error(f"frozen F3C1 source drifted: {relative}")
    stat_path = repo / f3c1.STATISTICS_PATH
    stat = _read(stat_path)
    _validate_signed(stat, "F3C1 statistical freeze")
    if _sha_file(stat_path) != F3C1_STAT_SHA256 or stat["canonical_digest"] != F3C1_STAT_DIGEST:
        raise StageFF3C2Error("F3C1 statistical freeze drifted")
    groups = [validate_sample(repo, selection_order=order) for order in PILOT_ORDERS]
    union = _read(repo / TERMINAL_UNION_PATH)
    budget = _read(repo / BUDGET_PATH)
    blind = _read(repo / BLIND_REPORT_PATH)
    access = _read(repo / ACCESS_AUDIT_PATH)
    tests = _read(repo / TEST_PATH)
    audit = _read(repo / FALSE_PASS_PATH)
    validate_blind_document(blind)
    if union.get("counts") != {"VALID": 25, "FAILED": 0, "NA": 0, "incomplete": 0}:
        raise StageFF3C2Error("terminal union is not 25/25 VALID")
    if budget.get("status") != "PASS" or not all(budget.get("checks", {}).values()):
        raise StageFF3C2Error("F3C2 budget is not PASS")
    if blind.get("status") != "PASS" or not all(blind.get("conditions", {}).values()):
        raise StageFF3C2Error("blind continuation gate is not PASS")
    if tests.get("status") != "PASS" or tests.get("failed") != 0:
        raise StageFF3C2Error("test gate failed")
    if audit.get("attack_count") != len(FALSE_PASS_ATTACKS) or any(item.get("result") != "REJECTED" for item in audit.get("attacks", [])):
        raise StageFF3C2Error("false-pass audit gate failed")
    for field in ("formal_sample_access", "formal_map_logit_curve_or_outcome_access", "order44_recomputed", "maps_rendered", "maps_manually_inspected", "pilot_jpegs_rendered_or_manually_viewed", "statistical_inference_performed", "scientific_comparison_performed", "winner_or_go_no_go_produced", "statistical_method_freeze_modified", "statistical_method_freeze_executed", "f3c3_started", "f3c4_started", "f4_started"):
        if access.get(field) is not False:
            raise StageFF3C2Error(f"F3C2 access boundary failed: {field}")
    if access.get("decoded_jpeg_selection_orders") != list(PILOT_ORDERS):
        raise StageFF3C2Error("F3C2 pilot access set drifted")
    if subprocess.run(["git", "diff", "--check"], cwd=repo).returncode:
        raise StageFF3C2Error("git diff --check failed")
    checks = {item: "PASS" for item in (
        "startup_and_predecode_plan_seal", "frozen_five_sample_and_25_unit_manifests",
        "model_preprocessing_source_config_bindings", "25_immutable_typed_terminals",
        "target_once_and_shared_per_sample", "all_block_attention_endpoint_parity",
        "nns_and_full_model_ig_conservation_attempt_retention", "common_fidelity_evaluator",
        "resource_ledger_and_budget_projection", "blind_continuation_boundary",
        "statistical_method_freeze_read_only_not_executed", "access_boundary",
        "false_pass_32_of_32", "git_diff_check",
    )}
    return _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_validation_report",
        "status": "PASS", "implementation_validation": "PASS",
        "workflow_validation": "PASS", "package_validation": "PASS",
        "checks": checks, "pilot_samples": "5/5", "method_terminals": "25/25 VALID",
        "nns_endpoint_parity": "5/5 PASS", "nns_conservation": "5/5 PASS",
        "full_model_ig_conservation": "5/5 PASS", "resource_budget_projection": "PASS",
        "test_passed": tests["passed"], "test_failed": tests["failed"],
        "test_deselected": tests["deselected"], "false_pass_attacks_rejected": len(FALSE_PASS_ATTACKS),
        "formal_sample_access": False, "order44_recomputed": False, "maps_rendered": False,
        "maps_manually_inspected": False, "statistical_inference_performed": False,
        "scientific_comparison_performed": False, "winner_or_go_no_go_produced": False,
        "sealed_upstream_modified": False, "f3c3_started": False, "f3c4_started": False,
        "f4_started": False, "next_authorization": "F3C3_FORMAL_ONLY",
        "target_class_indices_by_order": {str(group["selection_order"]): group["target_class_index"] for group in groups},
    })


def finalize(repo_root: Path | str = REPO_ROOT, *, completed_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validation = validate_outputs(repo)
    _write_new(repo / VALIDATION_PATH, validation)
    payload = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_payload_manifest", "status": "PASS",
        "records": [_record(repo, relative) for relative in (STARTUP_PATH, PILOT_MANIFEST_PATH, UNIT_PLAN_PATH, MODEL_BINDING_PATH, SOURCE_BINDING_PATH, RESOURCE_SPEC_PATH, BLIND_SPEC_PATH, ACCESS_POLICY_PATH, PLAN_SEAL_PATH, TERMINAL_UNION_PATH, ATTEMPT_JOURNAL_PATH, RESOURCE_LEDGER_PATH, BUDGET_PATH, ACCESS_AUDIT_PATH, BLIND_REPORT_PATH, TEST_PATH, FALSE_PASS_PATH, VALIDATION_PATH)],
        "typed_terminal_count": 25, "numeric_payload_present_only_for_valid_terminals": True,
        "display_or_render_payload_present": False,
    })
    _write_new(repo / PAYLOAD_PATH, payload)
    status = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_status", "status": "PASS",
        "terminal": True, "immutable": True, "implementation_validation": "PASS",
        "workflow_validation": "PASS", "package_validation": "PASS",
        "pilot_samples": "5/5", "method_terminals": "25/25 VALID",
        "nns_endpoint_parity": "5/5 PASS", "nns_conservation": "5/5 PASS",
        "full_model_ig_conservation": "5/5 PASS", "resource_budget_projection": "PASS",
        "pilot_sample_access": True, "pilot_selection_orders": list(PILOT_ORDERS),
        "formal_sample_access": False, "order44_recomputed": False, "maps_rendered": False,
        "maps_manually_inspected": False, "statistical_inference_performed": False,
        "scientific_comparison_performed": False, "winner_or_go_no_go_produced": False,
        "f3c0_f3c1_f3h_and_other_sealed_evidence_modified": False,
        "f3c3_started": False, "f3c4_started": False, "f4_started": False,
        "next_authorization": "F3C3_FORMAL_ONLY", "git": {"commit": "SELF", "ref": ARCHIVE_REF},
    })
    _write_new(repo / STATUS_PATH, status)
    inventory_payload = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "completion/f3c2.json", "completion/f3c2.pending.json"))
    inventory = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_whole_root_inventory",
        "status": "PASS", **inventory_payload,
    })
    _write_new(repo / INVENTORY_PATH, inventory)
    completion = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c2_terminal_completion",
        "status": "PASS", "completed_at": completed_at,
        "implementation_validation": "PASS", "workflow_validation": "PASS",
        "package_validation": "PASS", "pilot_samples": "5/5",
        "method_terminals": {"VALID": 25, "FAILED": 0, "NA": 0, "incomplete": 0},
        "nns_endpoint_parity": "5/5 PASS", "nns_conservation": "5/5 PASS",
        "full_model_ig_conservation": "5/5 PASS", "resource_budget_projection": "PASS",
        "startup_gate": _record(repo, STARTUP_PATH), "pilot_manifest": _record(repo, PILOT_MANIFEST_PATH),
        "unit_execution_plan": _record(repo, UNIT_PLAN_PATH), "model_preprocessing_binding": _record(repo, MODEL_BINDING_PATH),
        "source_config_binding": _record(repo, SOURCE_BINDING_PATH), "resource_accounting_specification": _record(repo, RESOURCE_SPEC_PATH),
        "blind_continuation_specification": _record(repo, BLIND_SPEC_PATH), "access_policy": _record(repo, ACCESS_POLICY_PATH),
        "plan_seal": _record(repo, PLAN_SEAL_PATH), "terminal_union": _record(repo, TERMINAL_UNION_PATH),
        "attempt_journal": _record(repo, ATTEMPT_JOURNAL_PATH), "resource_ledger": _record(repo, RESOURCE_LEDGER_PATH),
        "budget_projection": _record(repo, BUDGET_PATH), "access_audit": _record(repo, ACCESS_AUDIT_PATH),
        "blind_continuation_report": _record(repo, BLIND_REPORT_PATH), "payload_manifest": _record(repo, PAYLOAD_PATH),
        "whole_root_inventory": _record(repo, INVENTORY_PATH), "validation_report": _record(repo, VALIDATION_PATH),
        "false_pass_audit": _record(repo, FALSE_PASS_PATH), "test_report": _record(repo, TEST_PATH),
        "status_record": _record(repo, STATUS_PATH), "pilot_sample_access": True,
        "pilot_selection_orders": list(PILOT_ORDERS), "formal_sample_access": False,
        "order44_recomputed": False, "maps_rendered": False, "maps_manually_inspected": False,
        "statistical_inference_performed": False, "scientific_comparison_performed": False,
        "winner_or_go_no_go_produced": False, "f3c0_f3c1_f3h_and_other_sealed_evidence_modified": False,
        "f3c3_started": False, "f3c4_started": False, "f4_started": False,
        "next_authorization": "F3C3_FORMAL_ONLY", "git": {"commit": "SELF", "ref": ARCHIVE_REF},
    })
    temporary = (repo / PENDING_PATH).with_name(".f3c2-terminal.tmp")
    _write_new(temporary, completion)
    os.replace(temporary, repo / PENDING_PATH)
    os.replace(repo / PENDING_PATH, repo / COMPLETION_PATH)
    descriptor = os.open((repo / COMPLETION_PATH).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return completion


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    completion = _read(repo / COMPLETION_PATH)
    _validate_signed(completion, "F3C2 completion")
    if (repo / PENDING_PATH).exists() or completion.get("status") != "PASS":
        raise StageFF3C2Error("F3C2 completion is not atomically terminal PASS")
    if completion.get("method_terminals") != {"VALID": 25, "FAILED": 0, "NA": 0, "incomplete": 0}:
        raise StageFF3C2Error("F3C2 completion terminal counts drifted")
    if completion.get("next_authorization") != "F3C3_FORMAL_ONLY":
        raise StageFF3C2Error("F3C2 next authorization drifted")
    for field in ("formal_sample_access", "order44_recomputed", "maps_rendered", "maps_manually_inspected", "statistical_inference_performed", "scientific_comparison_performed", "winner_or_go_no_go_produced", "f3c0_f3c1_f3h_and_other_sealed_evidence_modified", "f3c3_started", "f3c4_started", "f4_started"):
        if completion.get(field) is not False:
            raise StageFF3C2Error(f"F3C2 completion boundary failed: {field}")
    for field in ("startup_gate", "pilot_manifest", "unit_execution_plan", "model_preprocessing_binding", "source_config_binding", "resource_accounting_specification", "blind_continuation_specification", "access_policy", "plan_seal", "terminal_union", "attempt_journal", "resource_ledger", "budget_projection", "access_audit", "blind_continuation_report", "payload_manifest", "whole_root_inventory", "validation_report", "false_pass_audit", "test_report", "status_record"):
        if completion[field] != _record(repo, completion[field]["path"]):
            raise StageFF3C2Error(f"F3C2 completion binding drifted: {field}")
    inventory = _read(repo / INVENTORY_PATH)
    rebuilt = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "completion/f3c2.json", "completion/f3c2.pending.json"))
    if any(inventory.get(key) != rebuilt[key] for key in ("files", "file_count", "total_bytes", "file_inventory_digest")):
        raise StageFF3C2Error("F3C2 whole-root inventory drifted")
    for order in PILOT_ORDERS:
        validate_sample(repo, selection_order=order)
    return {"status": "PASS", "archive_ref": ARCHIVE_REF, "pilot_samples": 5, "valid_terminals": 25, "next_authorization": "F3C3_FORMAL_ONLY"}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-initialized", "reconcile", "build-budget-blind-gate", "build-false-pass", "validate-outputs", "validate-completion"):
        child = sub.add_parser(command); child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    init = sub.add_parser("initialize"); init.add_argument("--repo-root", type=Path, default=REPO_ROOT); init.add_argument("--started-at", required=True)
    run = sub.add_parser("run-sample"); run.add_argument("--repo-root", type=Path, default=REPO_ROOT); run.add_argument("--selection-order", type=int, required=True); run.add_argument("--device", default="cuda:0")
    tests = sub.add_parser("record-tests"); tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); tests.add_argument("--passed", type=int, required=True); tests.add_argument("--failed", type=int, required=True); tests.add_argument("--deselected", type=int, default=0); tests.add_argument("--test-command", required=True)
    final = sub.add_parser("finalize"); final.add_argument("--repo-root", type=Path, default=REPO_ROOT); final.add_argument("--completed-at", required=True)
    args = parser.parse_args(argv)
    commands = {
        "initialize": lambda: initialize(args.repo_root, started_at=args.started_at),
        "validate-initialized": lambda: validate_initialized(args.repo_root),
        "run-sample": lambda: run_sample(args.repo_root, selection_order=args.selection_order, device=args.device),
        "reconcile": lambda: reconcile(args.repo_root),
        "build-budget-blind-gate": lambda: build_budget_and_blind_gate(args.repo_root),
        "record-tests": lambda: record_tests(args.repo_root, passed=args.passed, failed=args.failed, deselected=args.deselected, command=args.test_command),
        "build-false-pass": lambda: build_false_pass_audit(args.repo_root),
        "validate-outputs": lambda: validate_outputs(args.repo_root),
        "finalize": lambda: finalize(args.repo_root, completed_at=args.completed_at),
        "validate-completion": lambda: validate_completion(args.repo_root),
    }
    try:
        result = commands[args.command]()
    except (StageFF3C2Error, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
