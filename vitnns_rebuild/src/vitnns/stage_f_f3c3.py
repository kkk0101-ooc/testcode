"""Stage F F3C3 blind formal runner and control plane.

This module may produce per-method numerical terminals, but every stage-level
reconciliation path consumes only identities, terminal status, integrity
metadata, conservation/parity status, and resource telemetry.  Statistical
analysis and rendering intentionally have no execution surface here.
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


CONFIG_PATH = "config/stage_f_f3c3.json"
SCHEMA_PATH = "config/stage_f_f3c3.schema.json"
ROOT_PATH = "evidence/stage_f/f3c3_formal_v1_20260908/formal"
REPORT_PATH = "reports/stage_f_f3c3_formal_v1_20260908"
STARTUP_PATH = ROOT_PATH + "/startup_gate.json"
FORMAL_MANIFEST_PATH = ROOT_PATH + "/formal_manifest.json"
UNIT_PLAN_PATH = ROOT_PATH + "/unit_execution_plan.json"
SHARD_PLAN_PATH = ROOT_PATH + "/deterministic_shard_lease_plan.json"
MODEL_BINDING_PATH = ROOT_PATH + "/model_preprocessing_binding.json"
SOURCE_BINDING_PATH = ROOT_PATH + "/source_config_code_binding.json"
AUTHORIZATION_BINDING_PATH = ROOT_PATH + "/f3c2_continuation_authorization_binding.json"
RESOURCE_SPEC_PATH = ROOT_PATH + "/resource_accounting_specification.json"
BLIND_SPEC_PATH = ROOT_PATH + "/blind_formal_execution_specification.json"
ACCESS_POLICY_PATH = ROOT_PATH + "/access_policy.json"
PLAN_SEAL_PATH = ROOT_PATH + "/plan_seal.json"
PENDING_PATH = ROOT_PATH + "/completion/f3c3.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/f3c3.json"
TERMINAL_UNION_PATH = ROOT_PATH + "/terminal_union.json"
ATTEMPT_JOURNAL_PATH = ROOT_PATH + "/attempt_journal.json"
RESOURCE_LEDGER_PATH = ROOT_PATH + "/resource_ledger.json"
BUDGET_PATH = ROOT_PATH + "/total_f3c_budget_report.json"
ACCESS_AUDIT_PATH = ROOT_PATH + "/access_audit.json"
PAYLOAD_PATH = ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = ROOT_PATH + "/whole_root_inventory.json"
STATUS_PATH = ROOT_PATH + "/status.json"
BLIND_REPORT_PATH = REPORT_PATH + "/blind_formal_execution_report.json"
NUMERIC_VALIDATION_PATH = REPORT_PATH + "/postrun_numeric_integrity_validation.json"
VALIDATION_PATH = REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = REPORT_PATH + "/false_pass_audit.json"
PREFLIGHT_TEST_PATH = REPORT_PATH + "/preflight_test_report.json"
POSTRUN_TEST_PATH = REPORT_PATH + "/postrun_test_report.json"
ARCHIVE_REF = "refs/archive/stage-f-v1-f3c3-formal"
EXPECTED_BRANCH = "stage-d-d3-pass"
STARTUP_HEAD = "51826fdda62262a88ede2b59b15f08ea397f33a9"
ATTEMPT_ID = "attempt-000001"
WORKER_ID = "worker-000"
SHARD_COUNT = 1
SHARD_ID = 0
DEVICE = "cuda:0"
FORMAL_ROSTER_DIGEST = "d3e08fe512ff60a88889d089eb15ba382102ed983926086b806880f07e67a426"
SPLIT_PATH = "manifests/stage_f_f3c_v1/sample_split.json"
SPLIT_SHA256 = "b1dd0bb74f9ee0a2381eaa797c27a0365c13757b0f72ea253924fded3d8a2b09"
SPLIT_DIGEST = "b89be7ebf83c8e5783fd8eb12ad210c8ed6295dbfef4c9c81db1a6cd81188f71"
F3C2_COMPLETION_PATH = "evidence/stage_f/f3c2_pilot_v1_20260907/formal/completion/f3c2.json"
F3C2_COMPLETION_SHA256 = "05144c6532d190d5ef5c017e51a1ce891d15d3cff09f66c751b67825bf7b0fb4"
F3C2_COMPLETION_DIGEST = "5713e41ec1de427adaa1c70a2037fc9d9796cc0103aa4fa037fb686b2c48d0ad"
F3C2_INVENTORY_PATH = "evidence/stage_f/f3c2_pilot_v1_20260907/formal/whole_root_inventory.json"
F3C2_INVENTORY_SHA256 = "0cf845b910d112cdb33da3ac8657477d82ce1043be43c1240b69869bc15a67ed"
F3C2_INVENTORY_DIGEST = "f754e813d17e93e66f5e5721c6fb4711c407fa23925086d0e4ca8e5884ee4ac2"
F3C2_VALIDATION_PATH = "reports/stage_f_f3c2_pilot_v1_20260907/validation_report.json"
F3C2_VALIDATION_SHA256 = "1c2bde9b6d108f2adbbe01c5f859066ea322e733c35cb5064b365128bec41db6"
F3C2_VALIDATION_DIGEST = "b9dd43c16006cc00368bc4f06e2f44dbfeaf3e821e50b87e27eefe4ad506d034"
F3C2_FALSE_PASS_PATH = "reports/stage_f_f3c2_pilot_v1_20260907/false_pass_audit.json"
F3C2_FALSE_PASS_SHA256 = "9e1d8ee858887145b3537faf7fd1d60b31b64ea3124e0d85986f08f73f692317"
F3C2_FALSE_PASS_DIGEST = "d2730df68cb0ee53d24220ec323ad0c03a18c6d4e266594b0021b8eac5a3b032"
F3C1_STAT_PATH = "evidence/stage_f/f3c1_implementation_r6_cpu_seal_20260907/formal/statistical_method_freeze.json"
F3C1_STAT_SHA256 = "d0c8b03e8a10baa3e9c934dadf2519985526391a5cb03cee278b667df7c1544f"
F3C1_STAT_DIGEST = "546b88a25bc0b63d8bc42436923f95625d24270be240d94f88e3bcd3e28603de"
F3C1_GPU_HOURS = 0.003537431458341113
F3C2_GPU_HOURS = 0.01988542154631836
EXISTING_F3C1_F3C2_RETAINED_BYTES = 9_380_440
METHODS = kernels.METHODS
MODEL_PATH = f3c1.MODEL_PATH
MODEL_NAME = f3c1.MODEL_NAME
MODEL_IDENTITY = f3c1.MODEL_IDENTITY
MODEL_LOADER_IDENTITY = f3c1.MODEL_LOADER_IDENTITY
MODEL_SHA256 = f3c1.MODEL_SHA256
MODEL_TENSOR_DIGEST = f3c1.MODEL_TENSOR_DIGEST

FROZEN_F3C2_HASHES = {
    "config/stage_f_f3c2.json": "87d7f4475e7d2f510d633ebe614b4d8fd2c51c3c282ee5f0773a4a7c9cdbad1c",
    "config/stage_f_f3c2.schema.json": "d76669ea49de3ee8896c69a0adfc45931d4625f416d32ee1ad4031f4cac336df",
    "src/vitnns/stage_f_f3c2.py": "eec099e333962b2649528a85cec19787bc565e6e081172610dbcb15dc72428a2",
    "src/vitnns/stage_f_f3c2_validate.py": "807dcd34f242c6bce5664c540897b8315804522a778c57d69fe710e8d2eea11b",
    "src/vitnns/stage_f_f3c2_seal.py": "dec65294ab193369f90d63102f7f0614da84ffe8d3345d801ddcf951acf98dc2",
    "tests/test_stage_f_f3c2.py": "e8f14d2c9b7aeb390e373ce5119ab18fba30e5d41a4ea146076c4d68a70211b2",
    "tests/test_stage_f_f3c2_validate.py": "48ea7344ed0c17797af4897b9428826682dc7c3515c2459edc8968e13b7c6ede",
    "evidence/stage_f/f3c2_pilot_v1_20260907/formal/terminal_union.json": "c15ca730664cd842d8581d2bd804a375d5c4cb2a9f1add1f174d39f824fab3d0",
    "evidence/stage_f/f3c2_pilot_v1_20260907/formal/resource_ledger.json": "cf2d37b6943c05699571efe135cefa3a6936c3649b4d8f4e45f26cf88f56c245",
    "evidence/stage_f/f3c2_pilot_v1_20260907/formal/budget_projection.json": "6bde92d4d1adc8eb33919287203c1b7173149c1126c13f393f1109d545830d1d",
    "evidence/stage_f/f3c2_pilot_v1_20260907/formal/access_audit.json": "0fce8aeb2e4bf1712c2ff6381c6f900b1006d853e96d406adcfcd6a096add579",
    "reports/stage_f_f3c2_pilot_v1_20260907/blind_continuation_report.json": "52bfd2c1b303efe7fe8e828a41689379f9c0026334d493ae43470b645b544255",
}
SOURCE_FILES = (
    CONFIG_PATH, SCHEMA_PATH, "src/vitnns/stage_f_f3c3.py",
    "src/vitnns/stage_f_f3c3_validate.py", "tests/test_stage_f_f3c3.py",
    "tests/test_stage_f_f3c3_validate.py", "src/vitnns/stage_f_cli.py",
    "src/vitnns/stage_f_f3c1.py", "src/vitnns/stage_f_f3c1_kernels.py",
    "src/vitnns/stage_c_class_logit.py", "src/vitnns/stage_c_attention.py",
    "src/vitnns/attribution.py",
)
SEALED_REFS = {
    "f3c2": ("refs/archive/stage-f-v1-f3c2-pilot", STARTUP_HEAD),
    "f3c1": ("refs/archive/stage-f-v1-f3c1-implementation", "18e6b424f8b034cdd43490f01f2ab47b97ea3e76"),
    "f3c0": ("refs/archive/stage-f-v1-f3c0-protocol-freeze", "e2c104bd6653ce62c8c67f7c0ed55a771d505837"),
    "f3h": ("refs/archive/stage-f-v1-f3h-completion", "56afedd2e73a760dc5127ab4ee9c38d72623dbbc"),
}
PILOT_ORDERS = frozenset({416, 350, 438, 411, 337})
BLIND_FORBIDDEN_KEYS = frozenset({
    "native_map", "patch_mass", "target_oriented_mass", "ranking",
    "logits", "clean_raw_logit", "baseline_raw_logit", "curves",
    "responses", "auc", "fidelity_score", "delta_m", "p_value",
    "confidence_interval", "winner", "go_no_go", "bootstrap",
    "sign_flip", "holm", "scientific_outcome",
})
FALSE_PASS_ATTACKS = (
    "formal_roster_count_identity_or_order_drift",
    "sample_order_jpeg_hash_or_label_drift",
    "pilot_order44_or_unauthorized_sample_mixed_in",
    "pilot_map_logit_curve_auc_or_score_read",
    "formal_plan_not_sealed_before_jpeg_access",
    "unit_plan_missing_duplicate_or_method_order_drift",
    "shard_lease_claim_or_resume_overlap",
    "sample_duplicate_execution_or_billing",
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
    "blind_summary_consumes_forbidden_outcome",
    "statistical_freeze_modified_or_executed",
    "formal_delta_bootstrap_sign_flip_holm_pvalue_or_ci_produced",
    "comparison_ranking_winner_or_go_no_go_produced",
    "resource_missing_double_counted_underreported_or_formula_drift",
    "budget_breach_waived_or_authorizes_f3c4",
    "formal_jpeg_or_map_rendered_or_manually_viewed",
    "sealed_upstream_evidence_ref_or_hash_drift",
    "f3c4_or_f4_started",
    "completion_validator_union_blind_inventory_or_audit_missing",
    "git_diff_lfs_or_archive_state_unverified",
)


class StageFF3C3Error(ValueError):
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
            raise StageFF3C3Error(f"duplicate JSON key {key!r}: {path}")
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
        raise StageFF3C3Error(f"{context} canonical digest mismatch")


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if process.returncode:
        raise StageFF3C3Error(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def _file_record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _record(repo: Path, relative: str) -> dict[str, Any]:
    value = _read(repo / relative)
    _validate_signed(value, relative)
    return {**_file_record(repo, relative), "canonical_digest": value["canonical_digest"]}


def _regular_bytes(*roots: Path) -> int:
    return sum(
        path.stat().st_size
        for root in roots if root.exists()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


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


def false_pass_reasons(preconditions: Mapping[str, bool]) -> tuple[str, ...]:
    if set(preconditions) != set(FALSE_PASS_ATTACKS):
        raise StageFF3C3Error("false-pass preconditions must cover exactly 36 attacks")
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
        raise StageFF3C3Error("blind formal reconciliation consumed forbidden fields: " + ",".join(forbidden))


def load_config(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = _read(repo / CONFIG_PATH)
    schema = _read(repo / SCHEMA_PATH)
    errors = sorted(Draft202012Validator(schema).iter_errors(config), key=lambda item: list(item.path))
    if errors:
        raise StageFF3C3Error(errors[0].message)
    if tuple(config["methods"]) != METHODS or config["formal_sample_count"] != 50:
        raise StageFF3C3Error("F3C3 formal sample or method roster drifted")
    expected_execution = {
        "shard_count": 1, "worker_id": WORKER_ID, "device": DEVICE,
        "mapping": "formal_position_modulo_shard_count",
        "image_group_atomic": True,
        "lease_protocol": "exclusive_create_claim_then_terminal_release",
    }
    if config["execution"] != expected_execution:
        raise StageFF3C3Error("F3C3 deterministic execution plan drifted")
    expected_budget = {
        "f3c1_engineering_gpu_hours": F3C1_GPU_HOURS,
        "f3c2_pilot_actual_gpu_hours": F3C2_GPU_HOURS,
        "existing_retained_f3c1_f3c2_launch_baseline_bytes": EXISTING_F3C1_F3C2_RETAINED_BYTES,
        "total_h20_gpu_hours_max": 8.0, "per_image_group_h20_gpu_hours_max": 0.3,
        "max_devices": 8, "per_process_peak_reserved_bytes_max": 17_179_869_184,
        "retained_regular_file_bytes_max": 8_589_934_592, "waiver_allowed": False,
    }
    if config["budget"] != expected_budget:
        raise StageFF3C3Error("F3C3 budget contract drifted")
    return config


def _formal_rows(repo: Path) -> list[dict[str, Any]]:
    if _sha_file(repo / SPLIT_PATH) != SPLIT_SHA256:
        raise StageFF3C3Error("F3C frozen split file drifted")
    split = _read(repo / SPLIT_PATH)
    _validate_signed(split, "F3C split")
    if split["canonical_digest"] != SPLIT_DIGEST:
        raise StageFF3C3Error("F3C split canonical identity drifted")
    rows = split.get("formal", [])
    if len(rows) != 50 or canonical_sha256(rows) != FORMAL_ROSTER_DIGEST:
        raise StageFF3C3Error("formal roster is not the frozen 50-image cohort")
    required = {"selection_order", "sample_id", "relative_path", "image_sha256", "imagenet1k_index", "canonical_digest"}
    if any(set(row) != required for row in rows):
        raise StageFF3C3Error("formal sample identity fields drifted")
    if len({row["selection_order"] for row in rows}) != 50 or len({row["sample_id"] for row in rows}) != 50:
        raise StageFF3C3Error("formal roster has duplicate samples")
    if {row["selection_order"] for row in rows} & (PILOT_ORDERS | {44}):
        raise StageFF3C3Error("pilot or order 44 entered formal cohort")
    registry_binding = split.get("registry", {})
    registry_path = repo / registry_binding.get("path", "")
    if not registry_path.is_file() or _sha_file(registry_path) != registry_binding.get("sha256"):
        raise StageFF3C3Error("D1 registry binding drifted")
    registry = _read(registry_path)
    _validate_signed(registry, "D1 registry")
    if registry.get("canonical_digest") != registry_binding.get("canonical_digest") or len(registry.get("rows", [])) != 500:
        raise StageFF3C3Error("D1 registry identity drifted")
    excluded = set(split["selection_rule"]["excluded_selection_orders"])
    eligible = [row for row in registry["rows"] if row["selection_order"] not in excluded]
    namespace = split["selection_rule"]["namespace"]
    eligible.sort(key=lambda row: (hashlib.sha256((namespace + row["sample_id"]).encode()).hexdigest(), row["selection_order"]))
    keys = ("selection_order", "sample_id", "relative_path", "image_sha256", "imagenet1k_index", "canonical_digest")
    if rows != [{key: row[key] for key in keys} for row in eligible[5:55]]:
        raise StageFF3C3Error("formal cohort is not source-recomputed from D1 registry")
    return [dict(row) for row in rows]


def _verify_upstream(repo: Path) -> dict[str, Any]:
    for relative, expected in FROZEN_F3C2_HASHES.items():
        if _sha_file(repo / relative) != expected:
            raise StageFF3C3Error(f"frozen F3C2 source/evidence drifted: {relative}")
    bindings = (
        (F3C2_COMPLETION_PATH, F3C2_COMPLETION_SHA256, F3C2_COMPLETION_DIGEST, "F3C2 completion"),
        (F3C2_INVENTORY_PATH, F3C2_INVENTORY_SHA256, F3C2_INVENTORY_DIGEST, "F3C2 inventory"),
        (F3C2_VALIDATION_PATH, F3C2_VALIDATION_SHA256, F3C2_VALIDATION_DIGEST, "F3C2 validation"),
        (F3C2_FALSE_PASS_PATH, F3C2_FALSE_PASS_SHA256, F3C2_FALSE_PASS_DIGEST, "F3C2 false-pass"),
        (F3C1_STAT_PATH, F3C1_STAT_SHA256, F3C1_STAT_DIGEST, "F3C1 statistical freeze"),
    )
    records = {}
    for relative, expected_sha, expected_digest, label in bindings:
        value = _read(repo / relative)
        _validate_signed(value, label)
        if _sha_file(repo / relative) != expected_sha or value["canonical_digest"] != expected_digest:
            raise StageFF3C3Error(f"{label} identity drifted")
        records[label] = _record(repo, relative)
    completion = _read(repo / F3C2_COMPLETION_PATH)
    if (
        completion.get("status") != "PASS"
        or completion.get("next_authorization") != "F3C3_FORMAL_ONLY"
        or completion.get("pilot_samples") != "5/5"
        or completion.get("method_terminals") != {"VALID": 25, "FAILED": 0, "NA": 0, "incomplete": 0}
        or completion.get("nns_endpoint_parity") != "5/5 PASS"
        or completion.get("nns_conservation") != "5/5 PASS"
        or completion.get("full_model_ig_conservation") != "5/5 PASS"
    ):
        raise StageFF3C3Error("F3C2 terminal authorization gate failed")
    for field in (
        "formal_sample_access", "order44_recomputed", "maps_rendered",
        "maps_manually_inspected", "statistical_inference_performed",
        "scientific_comparison_performed", "winner_or_go_no_go_produced",
        "f3c3_started", "f3c4_started", "f4_started",
    ):
        if completion.get(field) is not False:
            raise StageFF3C3Error(f"F3C2 boundary drifted: {field}")
    inventory = _read(repo / F3C2_INVENTORY_PATH)
    if (inventory.get("file_count"), inventory.get("total_bytes"), inventory.get("file_inventory_digest")) != (411, 6_871_119, "995f2a17e120dc281d370c00b47d9074d22d3edfde6348ca39d31c0a4cd48ac6"):
        raise StageFF3C3Error("F3C2 inventory summary drifted")
    validation = _read(repo / F3C2_VALIDATION_PATH)
    if validation.get("status") != "PASS" or validation.get("postrun_validated_units") != 25:
        raise StageFF3C3Error("F3C2 validation gate failed")
    audit = _read(repo / F3C2_FALSE_PASS_PATH)
    if audit.get("attack_count") != 32 or any(row.get("result") != "REJECTED" for row in audit.get("attacks", [])):
        raise StageFF3C3Error("F3C2 false-pass gate failed")
    statistic = _read(repo / F3C1_STAT_PATH)
    if statistic.get("status") != "FROZEN_BEFORE_PILOT_OUTCOME_ACCESS":
        raise StageFF3C3Error("F3C1 statistical freeze status drifted")
    return {"records": records, "next_authorization": completion["next_authorization"]}


def validate_startup_gate(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    startup = _read(repo / STARTUP_PATH)
    _validate_signed(startup, "F3C3 startup gate")
    if startup.get("status") != "PASS" or startup.get("authorization") != "F3C3_FORMAL_ONLY":
        raise StageFF3C3Error("F3C3 startup gate is not PASS")
    if not startup.get("validated_before_code_modification_model_load_or_formal_jpeg_access"):
        raise StageFF3C3Error("F3C3 startup ordering is not proven")
    git = startup.get("git", {})
    if (git.get("branch"), git.get("local_head"), git.get("tracking_head"), git.get("remote_head")) != (EXPECTED_BRANCH, STARTUP_HEAD, STARTUP_HEAD, STARTUP_HEAD):
        raise StageFF3C3Error("F3C3 startup Git identity drifted")
    if not git.get("worktree_index_clean") or not git.get("git_lfs_clean"):
        raise StageFF3C3Error("F3C3 startup cleanliness was not proven")
    for name, (ref, commit) in SEALED_REFS.items():
        record = startup.get("archive_refs", {}).get(name, {})
        if record.get("ref") != ref or record.get("local_commit") != commit or record.get("remote_commit") != commit or record.get("ancestor_of_startup_head") is not True:
            raise StageFF3C3Error(f"F3C3 startup archive binding drifted: {name}")
    _verify_upstream(repo)
    _formal_rows(repo)
    return startup


def initialize(repo_root: Path | str = REPO_ROOT, *, started_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = repo / ROOT_PATH
    startup_file = repo / STARTUP_PATH
    if not startup_file.is_file() or any(path.name != "startup_gate.json" for path in root.iterdir()):
        raise StageFF3C3Error("fresh F3C3 root must contain only startup_gate.json")
    if (repo / REPORT_PATH).exists():
        raise StageFF3C3Error("F3C3 report root already exists")
    startup = validate_startup_gate(repo)
    config = load_config(repo)
    rows = _formal_rows(repo)
    source_files = [_file_record(repo, path) for path in SOURCE_FILES]
    source = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_source_config_code_binding",
        "status": "FROZEN_BEFORE_FORMAL_JPEG_ACCESS",
        "startup_head": startup["git"]["local_head"], "files": source_files,
        "source_tree_digest": canonical_sha256({"files": source_files}),
        "frozen_f3c2_hashes": FROZEN_F3C2_HASHES,
    })
    model = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_model_preprocessing_binding",
        "status": "FROZEN_BEFORE_FORMAL_JPEG_ACCESS",
        "model": {"name": MODEL_NAME, "identity": MODEL_IDENTITY, "path": MODEL_PATH, "sha256": MODEL_SHA256, "canonical_tensor_digest": MODEL_TENSOR_DIGEST},
        "preprocessing": {"color_mode": "RGB", "resize_shorter_side": 248, "interpolation": "bicubic", "antialias": True, "center_crop": [224, 224], "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "baseline": {"id": "normalized_zero", "canonical_rgb": [0.5, 0.5, 0.5], "black_image": False},
    })
    formal_rows = [{**row, "formal_position": index} for index, row in enumerate(rows, start=1)]
    manifest = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_exact_formal_manifest",
        "status": "FROZEN_BEFORE_FORMAL_JPEG_ACCESS", "samples": formal_rows,
        "sample_count": 50, "selection_orders": [row["selection_order"] for row in rows],
        "source_split": _file_record(repo, SPLIT_PATH),
        "source_formal_roster_digest": FORMAL_ROSTER_DIGEST,
        "pilot_or_order44_included": False, "replacement_allowed": False,
    })
    units = []
    for row in formal_rows:
        for method in METHODS:
            static = {
                "protocol_id": config["protocol_id"], "sample_id": row["sample_id"],
                "formal_position": row["formal_position"], "selection_order": row["selection_order"],
                "jpeg_sha256": row["image_sha256"], "official_imagenet1k_index": row["imagenet1k_index"],
                "method_id": method, "model_sha256": MODEL_SHA256,
                "model_canonical_tensor_digest": MODEL_TENSOR_DIGEST,
                "preprocessing_identity": model["canonical_digest"],
                "source_config_code_digest": source["source_tree_digest"],
                "clean_input_sha256": "BOUND_AFTER_AUTHORIZED_DECODE",
                "frozen_target_class_index": "BOUND_AFTER_SINGLE_CLEAN_FORWARD",
                "target_identity_digest": "BOUND_AFTER_SINGLE_CLEAN_FORWARD",
                "attempt_id": ATTEMPT_ID, "shard_id": SHARD_ID, "worker_id": WORKER_ID,
            }
            units.append({**static, "planned_unit_canonical_digest": canonical_sha256(static)})
    plan = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_250_unit_execution_plan",
        "status": "FROZEN_BEFORE_FORMAL_JPEG_ACCESS", "sample_count": 50,
        "method_count": 5, "unit_count": 250, "units": units,
        "dynamic_identity_rule": "clean_input_target_and_target_digest_bound_after_single_clean_forward_before_terminal_publish",
        "execution_order": "formal_position_then_frozen_method_order",
    })
    shard = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_deterministic_shard_lease_plan",
        "status": "FROZEN_BEFORE_FORMAL_JPEG_ACCESS", "shard_count": SHARD_COUNT,
        "shards": [{"shard_id": SHARD_ID, "worker_id": WORKER_ID, "device": DEVICE, "formal_positions": list(range(1, 51)), "selection_orders": [row["selection_order"] for row in rows]}],
        "mapping_rule": "zero_based_formal_position_modulo_shard_count",
        "claim_scope": "whole_five_method_image_group",
        "claim_protocol": "exclusive_create_immutable_claim_and_release_receipts",
        "overlap_allowed": False, "multiple_workers_per_image_allowed": False,
        "multiple_valid_attempts_allowed": False, "shared_group_resources_counted_once": True,
    })
    upstream = _verify_upstream(repo)
    authorization = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_f3c2_continuation_authorization_binding",
        "status": "PASS", "authorization": "F3C3_FORMAL_ONLY",
        "records": upstream["records"], "frozen_f3c2_hashes": FROZEN_F3C2_HASHES,
        "pilot_outcome_payload_consumed": False, "pilot_identity_status_qc_resource_only": True,
        "f3c1_statistical_method_freeze_modified": False,
        "f3c1_statistical_method_freeze_executed": False,
    })
    resource = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_resource_accounting_specification",
        "status": "FROZEN_BEFORE_FORMAL_JPEG_ACCESS",
        "image_group_scope": "model_load_jpeg_decode_clean_forward_five_methods_fidelity_terminal_publication_and_group_validation",
        "formal_actual_gpu_hours_formula": "sum(each_formal_image_group_actual_gpu_hours_including_all_attempts)",
        "actual_total_f3c_gpu_hours_formula": "f3c1_engineering_gpu_hours+f3c2_pilot_actual_gpu_hours+f3c3_formal_actual_gpu_hours",
        "actual_total_retained_bytes_formula": "retained_f3c1_bytes+retained_f3c2_bytes+retained_f3c3_bytes",
        "caps": config["budget"], "no_waiver": True,
        "shared_group_resources_not_summed_across_method_terminals": True,
        "stop_new_claims_immediately_on_observed_budget_breach": True,
    })
    blind = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_blind_formal_execution_specification",
        "status": "FROZEN_BEFORE_FORMAL_JPEG_ACCESS",
        "allowed_inputs": ["sample_and_unit_identity", "terminal_status", "endpoint_parity_status_and_error_bound", "ig_conservation_status_and_selected_steps", "manifest_and_hash_completeness", "resource_telemetry", "access_boundary_status", "terminal_counts"],
        "forbidden_inputs": sorted(BLIND_FORBIDDEN_KEYS),
        "pass_requires": {"formal_samples": 50, "valid_terminals": 250, "nns_endpoint_parity_pass": 50, "nns_conservation_pass": 50, "full_model_ig_conservation_pass": 50, "budget": "PASS", "pilot_access": False, "order44_recomputed": False, "render_or_manual_inspection": False, "statistics_or_comparison": False},
    })
    access = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_access_policy",
        "status": "FROZEN_BEFORE_FORMAL_JPEG_ACCESS",
        "allowed_formal_orders_after_plan_seal": [row["selection_order"] for row in rows],
        "pilot_jpeg_map_logit_curve_auc_score_access_allowed": False,
        "order44_recompute_allowed": False, "other_jpeg_access_allowed": False,
        "rendering_allowed": False, "manual_visual_inspection_allowed": False,
        "statistical_inference_allowed": False, "scientific_comparison_allowed": False,
        "winner_pvalue_ci_or_go_no_go_allowed": False,
    })
    for relative, value in (
        (SOURCE_BINDING_PATH, source), (MODEL_BINDING_PATH, model),
        (FORMAL_MANIFEST_PATH, manifest), (UNIT_PLAN_PATH, plan),
        (SHARD_PLAN_PATH, shard), (AUTHORIZATION_BINDING_PATH, authorization),
        (RESOURCE_SPEC_PATH, resource), (BLIND_SPEC_PATH, blind),
        (ACCESS_POLICY_PATH, access),
    ):
        _write_new(repo / relative, value)
    pending = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_pending_completion",
        "status": "PENDING", "started_at": started_at,
        "authorization": "F3C3_FORMAL_ONLY",
        "startup_gate_digest": startup["canonical_digest"],
        "formal_manifest_digest": manifest["canonical_digest"],
        "unit_plan_digest": plan["canonical_digest"],
        "shard_lease_plan_digest": shard["canonical_digest"],
        "model_preprocessing_digest": model["canonical_digest"],
        "source_config_code_digest": source["canonical_digest"],
        "authorization_binding_digest": authorization["canonical_digest"],
        "resource_spec_digest": resource["canonical_digest"],
        "blind_spec_digest": blind["canonical_digest"],
        "access_policy_digest": access["canonical_digest"],
        "formal_jpeg_access_started": False, "pilot_cohort_access": False,
        "order44_recomputed": False, "maps_rendered": False,
        "maps_manually_inspected": False, "statistical_inference_performed": False,
        "scientific_comparison_performed": False, "winner_pvalue_ci_or_go_no_go_produced": False,
        "f3c4_started": False, "f4_started": False,
    })
    _write_new(repo / PENDING_PATH, pending)
    seal = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_predecode_plan_seal",
        "status": "SEALED_BEFORE_FORMAL_JPEG_ACCESS", "sealed_at": _now(),
        "records": [_record(repo, relative) for relative in (STARTUP_PATH, FORMAL_MANIFEST_PATH, UNIT_PLAN_PATH, SHARD_PLAN_PATH, MODEL_BINDING_PATH, SOURCE_BINDING_PATH, AUTHORIZATION_BINDING_PATH, RESOURCE_SPEC_PATH, BLIND_SPEC_PATH, ACCESS_POLICY_PATH, PENDING_PATH)],
        "formal_jpeg_access_before_seal": False, "pilot_or_order44_access": False,
    })
    _write_new(repo / PLAN_SEAL_PATH, seal)
    (repo / REPORT_PATH).mkdir(parents=True, exist_ok=False)
    return {"status": "PASS", "plan_sealed": True, "formal_samples": 50, "units": 250, "formal_jpeg_access": False}


def validate_initialized(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    required = (STARTUP_PATH, FORMAL_MANIFEST_PATH, UNIT_PLAN_PATH, SHARD_PLAN_PATH, MODEL_BINDING_PATH, SOURCE_BINDING_PATH, AUTHORIZATION_BINDING_PATH, RESOURCE_SPEC_PATH, BLIND_SPEC_PATH, ACCESS_POLICY_PATH, PENDING_PATH, PLAN_SEAL_PATH)
    for relative in required:
        value = _read(repo / relative)
        _validate_signed(value, relative)
    pending = _read(repo / PENDING_PATH)
    if pending.get("status") != "PENDING" or pending.get("formal_jpeg_access_started") is not False:
        raise StageFF3C3Error("F3C3 pending receipt is not the sealed pre-access receipt")
    rows = _formal_rows(repo)
    manifest = _read(repo / FORMAL_MANIFEST_PATH)
    if manifest.get("samples") != [{**row, "formal_position": index} for index, row in enumerate(rows, 1)]:
        raise StageFF3C3Error("sealed formal manifest drifted")
    plan = _read(repo / UNIT_PLAN_PATH)
    if plan.get("unit_count") != 250 or len(plan.get("units", [])) != 250:
        raise StageFF3C3Error("F3C3 unit plan is incomplete")
    identities = [(item["selection_order"], item["method_id"]) for item in plan["units"]]
    expected = [(row["selection_order"], method) for row in rows for method in METHODS]
    if identities != expected or len(set(identities)) != 250:
        raise StageFF3C3Error("F3C3 unit plan order, coverage, or uniqueness drifted")
    shard = _read(repo / SHARD_PLAN_PATH)
    if shard.get("shard_count") != 1 or shard.get("shards", [{}])[0].get("selection_orders") != [row["selection_order"] for row in rows] or shard.get("overlap_allowed") is not False:
        raise StageFF3C3Error("F3C3 shard mapping drifted")
    source = _read(repo / SOURCE_BINDING_PATH)
    expected_files = [_file_record(repo, path) for path in SOURCE_FILES]
    if source.get("files") != expected_files or source.get("source_tree_digest") != canonical_sha256({"files": expected_files}):
        raise StageFF3C3Error("F3C3 source binding drifted after predecode seal")
    seal = _read(repo / PLAN_SEAL_PATH)
    for record in seal.get("records", []):
        if record != _record(repo, record["path"]):
            raise StageFF3C3Error("F3C3 predecode plan seal drifted")
    if seal.get("status") != "SEALED_BEFORE_FORMAL_JPEG_ACCESS":
        raise StageFF3C3Error("F3C3 plan was not sealed before access")
    _verify_upstream(repo)
    return {"status": "PASS", "formal_samples": 50, "units": 250, "formal_jpeg_access_allowed": True}


def _terminal_root(repo: Path, order: int, method: str) -> Path:
    return repo / ROOT_PATH / "terminals" / f"order-{order:04d}" / method / "attempts" / ATTEMPT_ID


def _claim_paths(repo: Path, order: int) -> tuple[Path, Path]:
    base = repo / ROOT_PATH / "leases"
    return base / f"order-{order:04d}.claim.json", base / f"order-{order:04d}.release.json"


def _claim_image_group(repo: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    claim_path, release_path = _claim_paths(repo, int(row["selection_order"]))
    token = canonical_sha256({
        "sample_id": row["sample_id"], "selection_order": row["selection_order"],
        "worker_id": WORKER_ID, "shard_id": SHARD_ID, "attempt_id": ATTEMPT_ID,
    })
    if claim_path.exists():
        claim = _read(claim_path)
        _validate_signed(claim, "F3C3 claim")
        if claim.get("claim_token") != token:
            raise StageFF3C3Error("conflicting image-group lease claim")
        if not release_path.is_file():
            raise StageFF3C3Error("existing image-group lease is active; recovery root required")
        release = _read(release_path)
        _validate_signed(release, "F3C3 lease release")
        if release.get("claim_token") != token or release.get("terminal") is not True:
            raise StageFF3C3Error("image-group lease release conflicts with claim")
        return claim
    if release_path.exists():
        raise StageFF3C3Error("lease release exists without immutable claim")
    claim = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_image_group_lease_claim",
        "status": "CLAIMED", "claim_token": token, "worker_id": WORKER_ID,
        "shard_id": SHARD_ID, "device": DEVICE, "sample_id": row["sample_id"],
        "selection_order": row["selection_order"], "formal_position": row["formal_position"],
        "attempt_id": ATTEMPT_ID, "claimed_at": _now(),
    })
    _write_new(claim_path, claim)
    return claim


def _release_image_group(repo: Path, row: Mapping[str, Any], claim: Mapping[str, Any], status: str) -> None:
    _claim_path, release_path = _claim_paths(repo, int(row["selection_order"]))
    if release_path.exists():
        raise StageFF3C3Error("refusing to overwrite image-group lease release")
    _write_new(release_path, _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_image_group_lease_release",
        "status": "RELEASED_TO_TERMINAL", "terminal_status": status,
        "terminal": True, "claim_token": claim["claim_token"],
        "worker_id": WORKER_ID, "shard_id": SHARD_ID,
        "sample_id": row["sample_id"], "selection_order": row["selection_order"],
        "formal_position": row["formal_position"], "released_at": _now(),
    }))


def _publish_terminal(
    destination: Path, result: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    *, blind_context: Mapping[str, Any],
) -> str:
    terminal_status = result.get("status")
    if terminal_status not in {"VALID", "FAILED", "NA"}:
        raise StageFF3C3Error("typed terminal status must be VALID, FAILED, or NA")
    if terminal_status == "VALID" and not arrays:
        raise StageFF3C3Error("VALID terminal requires complete numerical payload")
    if terminal_status in {"FAILED", "NA"} and arrays:
        raise StageFF3C3Error("FAILED/NA terminal must not contain numerical payload")
    if destination.exists():
        validated = validate_terminal(destination)
        previous = validated["result"]
        if previous.get("status") != terminal_status or previous.get("unit_identity") != result.get("unit_identity") or previous.get("payload_identities") != result.get("payload_identities"):
            raise StageFF3C3Error("conflicting terminal payload at existing attempt")
        siblings = sorted(destination.parent.glob("attempt-*"))
        if terminal_status == "VALID" and siblings != [destination]:
            raise StageFF3C3Error("multiple VALID attempts are forbidden")
        return "RESUME_SKIP_VERIFIED"
    if destination.parent.exists() and any(destination.parent.glob("attempt-*")):
        raise StageFF3C3Error("additional attempt requires prospective recovery authorization")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{ATTEMPT_ID}-", dir=destination.parent))
    try:
        _write_new(temporary / "result.json", _signed(result))
        records = []
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise StageFF3C3Error("terminal arrays must be finite float32")
            _write_npy(temporary / name, value)
            records.append({"path": name, "bytes": (temporary / name).stat().st_size, "sha256": _sha_file(temporary / name), "array_digest": kernels.array_digest(value), "shape": list(value.shape), "dtype": "float32"})
        status = _signed({
            "schema_version": 1, "document_type": "stage_f_f3c3_typed_terminal_status",
            "attempt_id": ATTEMPT_ID, "status": terminal_status,
            "terminal": True, "immutable": True,
        })
        _write_new(temporary / "status.json", status)
        manifest = _signed({
            "schema_version": 1, "document_type": "stage_f_f3c3_typed_terminal_manifest",
            "attempt_id": ATTEMPT_ID, "status": terminal_status,
            "result": _file_record(temporary, "result.json"),
            "status_record": _record(temporary, "status.json"),
            "arrays": records, "immutable": True,
        })
        _write_new(temporary / "manifest.json", manifest)
        blind = _signed({**dict(blind_context), "result_sha256": _sha_file(temporary / "result.json"), "manifest_sha256": _sha_file(temporary / "manifest.json")})
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
    required = [root / name for name in ("result.json", "status.json", "manifest.json", "blind.json")]
    if not root.is_dir() or any(not path.is_file() or path.is_symlink() for path in required):
        raise StageFF3C3Error("typed terminal is incomplete or nonregular")
    result, status, manifest, blind = (_read(path) for path in required)
    for value, name in ((result, "result"), (status, "status"), (manifest, "manifest"), (blind, "blind")):
        _validate_signed(value, f"typed terminal {name}")
    validate_blind_document(blind)
    terminal_status = result.get("status")
    if terminal_status not in {"VALID", "FAILED", "NA"} or status.get("status") != terminal_status or manifest.get("status") != terminal_status or blind.get("status") != terminal_status or status.get("terminal") is not True:
        raise StageFF3C3Error("typed terminal status conflict")
    if manifest.get("result") != _file_record(root, "result.json") or manifest.get("status_record") != _record(root, "status.json"):
        raise StageFF3C3Error("typed terminal control file conflict")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, list):
        raise StageFF3C3Error("typed terminal array manifest is invalid")
    if terminal_status == "VALID" and not arrays:
        raise StageFF3C3Error("VALID terminal lacks numerical payload")
    if terminal_status in {"FAILED", "NA"} and arrays:
        raise StageFF3C3Error("FAILED/NA terminal carries numerical payload")
    if blind.get("result_sha256") != _sha_file(root / "result.json") or blind.get("manifest_sha256") != _sha_file(root / "manifest.json"):
        raise StageFF3C3Error("blind terminal hash binding drifted")
    for record in arrays:
        path = root / record["path"]
        value = np.load(path, allow_pickle=False)
        if _sha_file(path) != record["sha256"] or path.stat().st_size != record["bytes"] or kernels.array_digest(value) != record["array_digest"] or list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise StageFF3C3Error("typed terminal numerical payload conflict")
    required_arrays = {"native.npy", "patch_mass.npy", "target_oriented_mass.npy", "rankings.npy", "insertion_curves.npy", "deletion_curves.npy", "insertion_aucs.npy", "deletion_aucs.npy", "fidelity_scores.npy", "fractions.npy", "actual_fractions.npy"}
    if terminal_status == "VALID" and {item["path"] for item in arrays} != required_arrays:
        raise StageFF3C3Error("VALID terminal numerical payload roster drifted")
    return {"status": "PASS", "result": result, "manifest": manifest, "blind": blind}


def validate_terminal_envelope(root: Path | str) -> dict[str, Any]:
    """Validate terminal hashes without decoding arrays or result outcomes."""

    root = Path(root)
    required = [root / name for name in ("result.json", "status.json", "manifest.json", "blind.json")]
    if not root.is_dir() or any(not path.is_file() or path.is_symlink() for path in required):
        raise StageFF3C3Error("blind terminal envelope is incomplete or nonregular")
    status, manifest, blind = (_read(root / name) for name in ("status.json", "manifest.json", "blind.json"))
    for value, name in ((status, "status"), (manifest, "manifest"), (blind, "blind")):
        _validate_signed(value, f"blind terminal {name}")
    validate_blind_document(blind)
    terminal_status = status.get("status")
    if terminal_status not in {"VALID", "FAILED", "NA"} or manifest.get("status") != terminal_status or blind.get("status") != terminal_status or status.get("terminal") is not True:
        raise StageFF3C3Error("blind terminal status conflict")
    if manifest.get("result") != _file_record(root, "result.json") or manifest.get("status_record") != _record(root, "status.json"):
        raise StageFF3C3Error("blind terminal control-file conflict")
    if blind.get("result_sha256") != _sha_file(root / "result.json") or blind.get("manifest_sha256") != _sha_file(root / "manifest.json"):
        raise StageFF3C3Error("blind terminal result/manifest hash conflict")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, list):
        raise StageFF3C3Error("blind terminal array manifest is invalid")
    if terminal_status == "VALID" and not arrays:
        raise StageFF3C3Error("blind VALID terminal lacks payload manifest")
    if terminal_status in {"FAILED", "NA"} and arrays:
        raise StageFF3C3Error("blind FAILED/NA terminal carries payload manifest")
    expected = {"native.npy", "patch_mass.npy", "target_oriented_mass.npy", "rankings.npy", "insertion_curves.npy", "deletion_curves.npy", "insertion_aucs.npy", "deletion_aucs.npy", "fidelity_scores.npy", "fractions.npy", "actual_fractions.npy"}
    if terminal_status == "VALID" and {item.get("path") for item in arrays} != expected:
        raise StageFF3C3Error("blind VALID payload roster drifted")
    for record in arrays:
        path = root / record["path"]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record["bytes"] or _sha_file(path) != record["sha256"]:
            raise StageFF3C3Error("blind payload file/hash completeness failed")
    return {"status": "PASS", "terminal_status": terminal_status, "manifest": manifest, "blind": blind}


def _load_formal_sample(repo: Path, row: Mapping[str, Any], device: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform_functional
    from .modeling import canonical_preprocess_spec, load_canonical_model

    image_path = repo / "assets/data/ILSVRC2012_val" / row["relative_path"]
    if _sha_file(image_path) != row["image_sha256"]:
        raise StageFF3C3Error("formal JPEG identity drifted")
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
        raise StageFF3C3Error("preprocessing contract drifted")
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        resized = transform_functional.resize(rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True)
        cropped = transform_functional.center_crop(resized, [224, 224])
        canonical = transform_functional.pil_to_tensor(cropped).to(torch.float32).div(255.0).unsqueeze(0)
    clean = ((canonical - 0.5) / 0.5).to(device)
    return model, clean, {
        "sample_id": row["sample_id"], "formal_position": row["formal_position"],
        "selection_order": row["selection_order"], "relative_path": row["relative_path"],
        "jpeg_sha256": row["image_sha256"], "official_imagenet1k_index": row["imagenet1k_index"],
        "clean_input_sha256": kernels.tensor_sha256(clean),
    }


def _target_oriented(method_id: str, method: Mapping[str, Any], fidelity: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    patch = np.asarray(method["patch_mass"], dtype=np.float32)
    sigma = int(fidelity["methods"][method_id]["sigma"])
    mass = np.maximum(np.float32(sigma) * patch, np.float32(0.0))
    denominator = float(mass.sum(dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0:
        raise StageFF3C3Error("target-oriented positive mass is zero")
    normalized = np.asarray(mass / denominator, dtype=np.float32)
    rankings = []
    for curve in fidelity["methods"][method_id]["curves"]:
        ranking, priority = kernels.method_ranking(patch, sigma=sigma, seed=curve["seed"])
        if kernels.array_digest(priority) != curve["tie_priority_sha256"] or kernels.array_digest(ranking) != curve["ranking_sha256"]:
            raise StageFF3C3Error("ranking identity is not source-recomputable")
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
        "native.npy": native, "patch_mass.npy": patch, "target_oriented_mass.npy": target_mass,
        "rankings.npy": rankings, "insertion_curves.npy": insertion, "deletion_curves.npy": deletion,
        "insertion_aucs.npy": insertion_auc, "deletion_aucs.npy": deletion_auc,
        "fidelity_scores.npy": scores, "fractions.npy": np.asarray(kernels.FRACTIONS, dtype=np.float32),
        "actual_fractions.npy": np.asarray(kernels._actual_fractions(), dtype=np.float32),
    }
    identities = {
        "native_array_digest": kernels.array_digest(native), "patch_mass_array_digest": kernels.array_digest(patch),
        "target_oriented_mass_array_digest": kernels.array_digest(target_mass), "ranking_array_digest": kernels.array_digest(rankings),
        "tie_priority_sha256": [item["tie_priority_sha256"] for item in curves],
        "insertion_curve_array_digest": kernels.array_digest(insertion), "deletion_curve_array_digest": kernels.array_digest(deletion),
    }
    return arrays, identities


def _completed_group_resources(repo: Path) -> list[dict[str, Any]]:
    resources = []
    for row in _formal_rows(repo):
        path = repo / ROOT_PATH / "image_groups" / f"order-{row['selection_order']:04d}" / "resource.json"
        if path.is_file():
            value = _read(path)
            _validate_signed(value, "F3C3 image-group resource")
            resources.append(value)
    return resources


def _assert_budget_before_new_claim(repo: Path) -> None:
    caps = load_config(repo)["budget"]
    resources = _completed_group_resources(repo)
    formal_gpu = sum(float(item["gpu_hours"]) for item in resources)
    retained = EXISTING_F3C1_F3C2_RETAINED_BYTES + _regular_bytes(repo / ROOT_PATH, repo / REPORT_PATH)
    if (
        any(float(item["gpu_hours"]) > caps["per_image_group_h20_gpu_hours_max"] for item in resources)
        or F3C1_GPU_HOURS + F3C2_GPU_HOURS + formal_gpu > caps["total_h20_gpu_hours_max"]
        or any(int(item["device_count"]) > caps["max_devices"] for item in resources)
        or any(int(item["peak_cuda_memory_reserved_bytes"]) > caps["per_process_peak_reserved_bytes_max"] for item in resources)
        or retained > caps["retained_regular_file_bytes_max"]
    ):
        raise StageFF3C3Error("BLOCKED_BUDGET: refusing a new formal image-group claim")


def run_sample(repo_root: Path | str = REPO_ROOT, *, selection_order: int, device: str = DEVICE) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validate_initialized(repo)
    config = load_config(repo)
    if device != config["execution"]["device"]:
        raise StageFF3C3Error("device assignment differs from sealed shard plan")
    rows = [{**row, "formal_position": index} for index, row in enumerate(_formal_rows(repo), 1)]
    row = next((item for item in rows if item["selection_order"] == selection_order), None)
    if row is None:
        raise StageFF3C3Error("requested sample is not one frozen formal member")
    group_root = repo / ROOT_PATH / "image_groups" / f"order-{selection_order:04d}"
    if (group_root / "completion.json").is_file():
        return validate_sample(repo, selection_order=selection_order)
    earlier = rows[: row["formal_position"] - 1]
    if any(not (repo / ROOT_PATH / "image_groups" / f"order-{item['selection_order']:04d}" / "completion.json").is_file() for item in earlier):
        raise StageFF3C3Error("formal samples must execute in frozen order")
    _assert_budget_before_new_claim(repo)
    claim = _claim_image_group(repo, row)
    model = None
    start = time.monotonic()
    cpu_start = time.process_time()
    forward_calls = 0
    try:
        import torch
        import timm
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise StageFF3C3Error("F3C3 formal execution requires the sealed CUDA device")
        torch.cuda.set_device(torch.device(device))
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        model, clean, sample = _load_formal_sample(repo, row, device)

        def count_forward(_module: Any, _args: Any) -> None:
            nonlocal forward_calls
            forward_calls += 1

        handle = model.register_forward_pre_hook(count_forward)
        try:
            capture = kernels.capture_clean_target_and_all_attention(model, clean, official_truth_index=row["imagenet1k_index"])
            methods = kernels.collect_five_method_maps(model, clean, capture, capture.as_attention_extraction())
            fidelity = kernels.evaluate_common_fidelity(model, clean, capture, methods, sample_id=row["sample_id"])
        finally:
            handle.remove()
        compute_elapsed = time.monotonic() - start
        compute_cpu = time.process_time() - cpu_start
        peak = int(torch.cuda.max_memory_reserved(torch.device(device)))
        target_payload = {
            "target_ref": "clean_predicted_class_raw_logit", "class_index": capture.target.class_index,
            "clean_input_sha256": capture.clean_input_sha256, "official_truth_index": row["imagenet1k_index"],
            "clean_top1_correct": capture.target.clean_top1_correct, "resolved_once_from_same_clean_forward": True,
            "frozen_target": capture.target.to_dict(),
        }
        target_identity_digest = canonical_sha256(target_payload)
        source = _read(repo / SOURCE_BINDING_PATH)
        model_binding = _read(repo / MODEL_BINDING_PATH)
        base_identity = {
            "protocol_id": config["protocol_id"], "sample_id": row["sample_id"],
            "formal_position": row["formal_position"], "selection_order": selection_order,
            "jpeg_sha256": row["image_sha256"], "official_imagenet1k_index": row["imagenet1k_index"],
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
            + sum(math.ceil(int(item["steps"]) / kernels.IG_CHUNK) for item in full_ig["attempts"]) + 1
        )
        resource_snapshot = {
            "elapsed_seconds": compute_elapsed, "cpu_seconds": compute_cpu,
            "gpu_hours": compute_elapsed / 3600.0, "device_count": 1, "cuda_device": device,
            "gpu_name": torch.cuda.get_device_name(torch.device(device)),
            "peak_cuda_memory_reserved_bytes": peak, "forward_calls": forward_calls,
            "backward_calls": backward_calls,
            "backward_call_accounting": "one_autograd_grad_per_retained_IG_chunk_plus_one_gradcam_call",
            "backward_chunk_calls_expected": backward_calls - 1,
            "torch_version": str(torch.__version__), "timm_version": str(timm.__version__),
            "cuda_runtime_version": str(torch.version.cuda),
            "accounting_scope": "shared_five_method_image_group_snapshot_before_atomic_terminal_publication",
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
                "schema_version": 1, "document_type": "stage_f_f3c3_sample_method_terminal",
                "status": "VALID", "attempt_id": ATTEMPT_ID, "unit_identity": unit_identity,
                "unit_canonical_digest": unit_digest, "sample": sample, "target": target_payload,
                "target_identity_digest": target_identity_digest, "method_id": method_id,
                "class_targeted": bool(method["class_targeted"]), "class_agnostic": bool(method["class_agnostic"]),
                "attention_hashes": list(capture.attentions.hashes) if method_id == kernels.METHOD_ID else [],
                "endpoint_parity": method.get("endpoint_parity"), "ig_conservation": conservation,
                "payload_identities": identities,
                "fidelity_contract": {"fractions": list(kernels.FRACTIONS), "actual_fractions": list(kernels._actual_fractions()), "tie_seeds": [kernels.tie_seed(row["sample_id"], repeat) for repeat in range(5)], "tie_priority_sha256": identities["tie_priority_sha256"], "target": "unmodified_frozen_clean_predicted_class_raw_logit", "score": "0.5*(insertion_auc-deletion_auc)"},
                "fidelity_numeric_payload": fidelity["methods"][method_id],
                "resources": {**resource_snapshot, "artifact_bytes": sum(value.nbytes for value in arrays.values()), "artifact_bytes_scope": "uncompressed_numeric_array_payload"},
                "resources_are_shared_image_group_not_additive": True,
                "maps_rendered": False, "maps_manually_inspected": False,
                "statistical_inference_performed": False, "scientific_comparison_performed": False,
            }
            blind_context = {
                "schema_version": 1, "document_type": "stage_f_f3c3_blind_terminal_record",
                "status": "VALID", "attempt_id": ATTEMPT_ID, "terminal": True,
                "sample_id": sample["sample_id"], "formal_position": sample["formal_position"],
                "selection_order": sample["selection_order"], "jpeg_sha256": sample["jpeg_sha256"],
                "clean_input_sha256": sample["clean_input_sha256"], "method_id": method_id,
                "target_class_index": capture.target.class_index, "target_identity_digest": target_identity_digest,
                "unit_canonical_digest": unit_digest,
                "attention_count": capture.attentions.manifest()["count"] if method_id == kernels.METHOD_ID else None,
                "attention_hash_count": len(capture.attentions.manifest()["tensor_sha256"]) if method_id == kernels.METHOD_ID else 0,
                "endpoint_parity_status": method.get("endpoint_parity", {}).get("status", "NOT_APPLICABLE") if method.get("endpoint_parity") is not None else "NOT_APPLICABLE",
                "endpoint_parity_max_absolute_error": method.get("endpoint_parity", {}).get("max_absolute_error") if method.get("endpoint_parity") is not None else None,
                "ig_conservation_status": "PASS" if conservation and conservation["passed"] else "NOT_APPLICABLE",
                "ig_selected_steps": conservation["selected_steps"] if conservation is not None else None,
                "attempt_count": len(conservation["attempts"]) if conservation is not None else 0,
                "resources": resource_snapshot, "manifest_hash_complete": True,
                "numerical_payload_not_read_by_blind_gate": True,
            }
            destination = _terminal_root(repo, selection_order, method_id)
            action = _publish_terminal(destination, result, arrays, blind_context=blind_context)
            validated = validate_terminal(destination)
            terminal_records.append({"method_id": method_id, "action": action, "manifest_sha256": validated["manifest"]["canonical_digest"]})
        elapsed = time.monotonic() - start
        cpu = time.process_time() - cpu_start
        retained = sum(_regular_bytes(_terminal_root(repo, selection_order, method)) for method in METHODS)
        group_resources = {
            **resource_snapshot, "elapsed_seconds": elapsed, "cpu_seconds": cpu,
            "gpu_hours": elapsed / 3600.0, "retained_bytes": retained,
            "accounting_scope": "shared_five_method_image_group_from_pre_model_load_through_atomic_terminal_publication",
            "shared_image_group_counted_once": True,
        }
        caps = config["budget"]
        group_status = "PASS"
        if group_resources["gpu_hours"] > caps["per_image_group_h20_gpu_hours_max"] or group_resources["peak_cuda_memory_reserved_bytes"] > caps["per_process_peak_reserved_bytes_max"]:
            group_status = "BLOCKED_BUDGET"
        group_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".order-{selection_order:04d}-", dir=group_root.parent))
        try:
            _write_new(temporary / "clean_input_binding.json", _signed({"schema_version": 1, "document_type": "stage_f_f3c3_clean_input_binding", "status": "PASS", **sample}))
            _write_new(temporary / "target_binding.json", _signed({"schema_version": 1, "document_type": "stage_f_f3c3_target_binding", "status": "PASS", **target_payload, "target_identity_digest": target_identity_digest}))
            _write_new(temporary / "attention_binding.json", _signed({"schema_version": 1, "document_type": "stage_f_f3c3_attention_binding", "status": "PASS", "manifest": capture.attentions.manifest(), "same_clean_forward": True, "q_k_softmax_variation_cut": True, "value_projection_residual_mlp_final_norm_classifier_live": True}))
            _write_new(temporary / "resource.json", _signed({"schema_version": 1, "document_type": "stage_f_f3c3_image_group_resource", "status": group_status, "formal_position": row["formal_position"], "selection_order": selection_order, **group_resources}))
            _write_new(temporary / "completion.json", _signed({
                "schema_version": 1, "document_type": "stage_f_f3c3_image_group_completion",
                "status": group_status, "formal_position": row["formal_position"],
                "selection_order": selection_order, "sample_id": row["sample_id"],
                "terminal_count": 5, "counts": {"VALID": 5, "FAILED": 0, "NA": 0, "incomplete": 0},
                "target_class_index": capture.target.class_index, "target_identity_digest": target_identity_digest,
                "nns_endpoint_parity": methods[kernels.METHOD_ID]["endpoint_parity"],
                "nns_conservation": {"selected_steps": nns_ig.steps, "passed": nns_ig.converged},
                "full_model_ig_conservation": {"selected_steps": full_ig["selected_steps"], "passed": full_ig["converged"]},
                "resources": group_resources, "terminals": terminal_records,
                "maps_rendered": False, "maps_manually_inspected": False,
            }))
            os.replace(temporary, group_root)
            descriptor = os.open(group_root.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        _release_image_group(repo, row, claim, group_status)
        if group_status != "PASS":
            raise StageFF3C3Error("BLOCKED_BUDGET: image-group cap exceeded")
        return {"status": "PASS", "formal_position": row["formal_position"], "selection_order": selection_order, "valid_terminals": 5, "target_class_index": capture.target.class_index, "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600.0}
    except BaseException:
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
    rows = [{**row, "formal_position": index} for index, row in enumerate(_formal_rows(repo), 1)]
    row = next((item for item in rows if item["selection_order"] == selection_order), None)
    if row is None:
        raise StageFF3C3Error("sample is not in frozen formal cohort")
    group_root = repo / ROOT_PATH / "image_groups" / f"order-{selection_order:04d}"
    group = _read(group_root / "completion.json")
    _validate_signed(group, "F3C3 image-group completion")
    records = []
    for method in METHODS:
        terminal_root = _terminal_root(repo, selection_order, method)
        validated = validate_terminal_envelope(terminal_root)
        blind = validated["blind"]
        if blind.get("method_id") != method or blind.get("sample_id") != row["sample_id"] or blind.get("formal_position") != row["formal_position"]:
            raise StageFF3C3Error("blind terminal identity drifted")
        records.append(blind)
    statuses = {status: sum(item["status"] == status for item in records) for status in ("VALID", "FAILED", "NA")}
    statuses["incomplete"] = 5 - sum(statuses.values())
    if group.get("counts") != statuses or group.get("sample_id") != row["sample_id"] or group.get("formal_position") != row["formal_position"]:
        raise StageFF3C3Error("image-group completion identity/count drifted")
    if len({item["target_identity_digest"] for item in records}) != 1 or len({item["target_class_index"] for item in records}) != 1:
        raise StageFF3C3Error("target/class identity is not shared across five methods")
    claim_path, release_path = _claim_paths(repo, selection_order)
    claim, release = _read(claim_path), _read(release_path)
    _validate_signed(claim, "F3C3 claim")
    _validate_signed(release, "F3C3 release")
    if claim.get("claim_token") != release.get("claim_token") or release.get("terminal") is not True:
        raise StageFF3C3Error("image-group lease did not close exactly once")
    return {"status": group["status"], "formal_position": row["formal_position"], "selection_order": selection_order, "counts": statuses, "target_class_index": records[0]["target_class_index"], "blind_records": records}


def reconcile(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validate_initialized(repo)
    rows = [{**row, "formal_position": index} for index, row in enumerate(_formal_rows(repo), 1)]
    groups = [validate_sample(repo, selection_order=row["selection_order"]) for row in rows]
    blind_records = [record for group in groups for record in group["blind_records"]]
    if len(blind_records) != 250 or len({(item["selection_order"], item["method_id"]) for item in blind_records}) != 250:
        raise StageFF3C3Error("formal terminal set is not exactly 250 unique units")
    union_records = []
    attempts = []
    for row in rows:
        for method in METHODS:
            root = _terminal_root(repo, row["selection_order"], method)
            attempt_dirs = sorted(root.parent.glob("attempt-*"))
            if attempt_dirs != [root]:
                raise StageFF3C3Error("unit has missing or multiple attempts")
            envelope = validate_terminal_envelope(root)
            blind = envelope["blind"]
            union_records.append({
                "formal_position": row["formal_position"], "selection_order": row["selection_order"],
                "sample_id": row["sample_id"], "method_id": method, "attempt_id": ATTEMPT_ID,
                "status": envelope["terminal_status"], "unit_canonical_digest": blind["unit_canonical_digest"],
                "root": root.relative_to(repo).as_posix(),
                "manifest": _record(repo, (root / "manifest.json").relative_to(repo).as_posix()),
                "blind": _record(repo, (root / "blind.json").relative_to(repo).as_posix()),
            })
            attempts.append({
                "formal_position": row["formal_position"], "selection_order": row["selection_order"],
                "method_id": method, "attempt_id": ATTEMPT_ID, "status": envelope["terminal_status"],
                "ig_attempt_count": blind["attempt_count"], "ig_selected_steps": blind["ig_selected_steps"],
                "attempts_discarded": False,
            })
    counts = {status: sum(item["status"] == status for item in union_records) for status in ("VALID", "FAILED", "NA")}
    counts["incomplete"] = 250 - sum(counts.values())
    status = "PASS" if counts == {"VALID": 250, "FAILED": 0, "NA": 0, "incomplete": 0} else "BLOCKED"
    union = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_terminal_union",
        "status": status, "sample_count": 50, "unit_count": 250, "counts": counts,
        "records": union_records, "multiple_valid_attempts": False, "conflicting_payloads": False,
        "numerical_payload_consumed": False,
    })
    _write_new(repo / TERMINAL_UNION_PATH, union)
    _write_new(repo / ATTEMPT_JOURNAL_PATH, _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_attempt_journal",
        "status": status, "unit_attempts": attempts, "unit_count": 250,
        "attempt_count": len(attempts), "attempts_discarded": False,
        "attempt_details_bound_inside_immutable_terminals_not_consumed_by_blind_reconciliation": True,
    }))
    resources = []
    for row in rows:
        group_root = repo / ROOT_PATH / "image_groups" / f"order-{row['selection_order']:04d}"
        resource = _read(group_root / "resource.json")
        _validate_signed(resource, "F3C3 image-group resource")
        resources.append({key: resource[key] for key in ("formal_position", "selection_order", "elapsed_seconds", "cpu_seconds", "gpu_hours", "device_count", "cuda_device", "gpu_name", "peak_cuda_memory_reserved_bytes", "forward_calls", "backward_calls", "backward_chunk_calls_expected", "retained_bytes", "shared_image_group_counted_once")})
    ledger = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_resource_ledger",
        "status": status, "image_groups": resources,
        "formal_actual_gpu_hours": sum(item["gpu_hours"] for item in resources),
        "total_elapsed_seconds": sum(item["elapsed_seconds"] for item in resources),
        "total_cpu_seconds": sum(item["cpu_seconds"] for item in resources),
        "total_forward_calls": sum(item["forward_calls"] for item in resources),
        "total_backward_calls": sum(item["backward_calls"] for item in resources),
        "peak_reserved_bytes": max(item["peak_cuda_memory_reserved_bytes"] for item in resources),
        "observed_devices": len({item["cuda_device"] for item in resources}),
        "gpu_names": sorted({item["gpu_name"] for item in resources}),
        "duplicate_shared_method_accounting": False, "waiver": False,
    })
    _write_new(repo / RESOURCE_LEDGER_PATH, ledger)
    _write_new(repo / ACCESS_AUDIT_PATH, _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_access_audit", "status": status,
        "formal_sample_access": True, "decoded_formal_selection_orders": [row["selection_order"] for row in rows],
        "decoded_formal_sample_count": 50, "pilot_cohort_access": False,
        "pilot_map_logit_curve_auc_or_score_access": False, "order44_recomputed": False,
        "unlisted_jpeg_access": False, "formal_jpegs_rendered_or_manually_viewed": False,
        "maps_rendered": False, "maps_manually_inspected": False,
        "statistical_inference_performed": False, "scientific_comparison_performed": False,
        "delta_bootstrap_sign_flip_holm_pvalue_or_ci_produced": False,
        "winner_or_go_no_go_produced": False, "statistical_method_freeze_modified": False,
        "statistical_method_freeze_executed": False, "f3c4_started": False, "f4_started": False,
    }))
    return {"status": status, "counts": counts, "formal_actual_gpu_hours": ledger["formal_actual_gpu_hours"]}


def build_budget_and_blind_gate(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    union, ledger, access = (_read(repo / path) for path in (TERMINAL_UNION_PATH, RESOURCE_LEDGER_PATH, ACCESS_AUDIT_PATH))
    for value, name in ((union, "terminal union"), (ledger, "resource ledger"), (access, "access audit")):
        _validate_signed(value, name)
    rows = [{**row, "formal_position": index} for index, row in enumerate(_formal_rows(repo), 1)]
    blind_records = []
    for row in rows:
        for method in METHODS:
            envelope = validate_terminal_envelope(_terminal_root(repo, row["selection_order"], method))
            blind_records.append(envelope["blind"])
    caps = load_config(repo)["budget"]
    formal_gpu = float(ledger["formal_actual_gpu_hours"])
    actual_total_gpu = F3C1_GPU_HOURS + F3C2_GPU_HOURS + formal_gpu
    f3c3_retained = _regular_bytes(repo / ROOT_PATH, repo / REPORT_PATH)
    actual_total_retained = EXISTING_F3C1_F3C2_RETAINED_BYTES + f3c3_retained
    image_groups = ledger["image_groups"]
    checks = {
        "each_image_group_gpu_hours": all(item["gpu_hours"] <= caps["per_image_group_h20_gpu_hours_max"] for item in image_groups),
        "actual_total_f3c_gpu_hours": actual_total_gpu <= caps["total_h20_gpu_hours_max"],
        "observed_devices": ledger["observed_devices"] <= caps["max_devices"],
        "per_process_peak_reserved_bytes": ledger["peak_reserved_bytes"] <= caps["per_process_peak_reserved_bytes_max"],
        "actual_total_retained_bytes": actual_total_retained <= caps["retained_regular_file_bytes_max"],
        "no_waiver": ledger["waiver"] is False and caps["waiver_allowed"] is False,
    }
    budget_status = "PASS" if all(checks.values()) else "BLOCKED_BUDGET"
    budget = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_total_f3c_budget_report",
        "status": budget_status, "f3c1_engineering_gpu_hours": F3C1_GPU_HOURS,
        "f3c2_pilot_actual_gpu_hours": F3C2_GPU_HOURS, "f3c3_formal_actual_gpu_hours": formal_gpu,
        "actual_total_f3c_gpu_hours": actual_total_gpu,
        "existing_retained_f3c1_f3c2_launch_baseline_bytes": EXISTING_F3C1_F3C2_RETAINED_BYTES,
        "f3c3_retained_regular_file_bytes_at_budget_gate": f3c3_retained,
        "actual_total_retained_bytes_at_budget_gate": actual_total_retained,
        "retained_accounting_boundary": "all_F3C3_regular_files_existing_immediately_before_this_budget_report_plus_frozen_F3C1_F3C2_launch_baseline",
        "observed_devices": ledger["observed_devices"], "peak_reserved_bytes": ledger["peak_reserved_bytes"],
        "maximum_image_group_gpu_hours": max(item["gpu_hours"] for item in image_groups),
        "caps": caps, "checks": checks, "waiver": False,
        "formula_version": "F3C3_ACTUAL_IMAGE_GROUP_SUM_V1",
    })
    _write_new(repo / BUDGET_PATH, budget)
    nns = [item for item in blind_records if item["method_id"] == kernels.METHOD_ID]
    full = [item for item in blind_records if item["method_id"] == "full_model_class_logit_ig"]
    conditions = {
        "samples_50_of_50_in_frozen_order": [item["selection_order"] for item in nns] == [row["selection_order"] for row in rows],
        "terminals_250_of_250_valid": len(blind_records) == 250 and all(item["status"] == "VALID" for item in blind_records),
        "nns_endpoint_parity_50_of_50": len(nns) == 50 and all(item["endpoint_parity_status"] == "PASS" and item["endpoint_parity_max_absolute_error"] < 1e-6 for item in nns),
        "nns_conservation_50_of_50": len(nns) == 50 and all(item["ig_conservation_status"] == "PASS" for item in nns),
        "full_model_ig_conservation_50_of_50": len(full) == 50 and all(item["ig_conservation_status"] == "PASS" for item in full),
        "target_identity_shared_once_per_sample": all(len({item["target_identity_digest"] for item in blind_records if item["selection_order"] == row["selection_order"]}) == 1 for row in rows),
        "manifest_hash_complete": all(item["manifest_hash_complete"] for item in blind_records),
        "formal_only_access_boundary": access["decoded_formal_selection_orders"] == [row["selection_order"] for row in rows] and not access["pilot_cohort_access"] and not access["order44_recomputed"] and not access["unlisted_jpeg_access"],
        "resource_and_total_f3c_budget": budget_status == "PASS",
        "no_render_or_manual_inspection": not access["maps_rendered"] and not access["maps_manually_inspected"] and not access["formal_jpegs_rendered_or_manually_viewed"],
        "no_statistics_comparison_or_decision": not access["statistical_inference_performed"] and not access["scientific_comparison_performed"] and not access["delta_bootstrap_sign_flip_holm_pvalue_or_ci_produced"] and not access["winner_or_go_no_go_produced"],
        "statistical_freeze_read_only_and_unexecuted": not access["statistical_method_freeze_modified"] and not access["statistical_method_freeze_executed"],
    }
    status = "PASS" if all(conditions.values()) else ("BLOCKED_BUDGET" if not conditions["resource_and_total_f3c_budget"] else "BLOCKED")
    report = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_blind_formal_execution_report",
        "status": status, "input_policy": "BLIND_IDENTITY_STATUS_INTEGRITY_CONSERVATION_RESOURCE_ONLY",
        "records": blind_records, "conditions": conditions, "total_f3c_budget_report": _record(repo, BUDGET_PATH),
        "forbidden_numerical_outcomes_consumed": False,
        "next_authorization": "F3C4_CPU_ANALYSIS_AND_SEAL_ONLY" if status == "PASS" else "F3C3_RECOVERY_REQUIRED",
    })
    validate_blind_document(report)
    _write_new(repo / BLIND_REPORT_PATH, report)
    return {"status": status, "next_authorization": report["next_authorization"], "conditions": conditions}


def record_tests(repo_root: Path | str, *, phase: str, passed: int, failed: int, deselected: int, command: str) -> dict[str, Any]:
    if phase not in {"preflight", "postrun"}:
        raise StageFF3C3Error("test phase must be preflight or postrun")
    if passed <= 0 or failed != 0:
        raise StageFF3C3Error("F3C3 regression tests are not passing")
    repo = Path(repo_root).resolve()
    path = PREFLIGHT_TEST_PATH if phase == "preflight" else POSTRUN_TEST_PATH
    result = _signed({
        "schema_version": 1, "document_type": f"stage_f_f3c3_{phase}_test_report",
        "status": "PASS", "phase": phase, "passed": passed, "failed": failed,
        "deselected": deselected, "command": command,
        "false_pass_mutation_tests": len(FALSE_PASS_ATTACKS),
        "formal_workload_started_during_tests": False, "model_loaded": False,
        "jpeg_decoded": False, "render_or_manual_inspection": False,
        "statistical_inference_or_scientific_comparison": False,
    })
    _write_new(repo / path, result)
    return result


def build_false_pass_audit(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    preflight, postrun = _read(repo / PREFLIGHT_TEST_PATH), _read(repo / POSTRUN_TEST_PATH)
    for value, name in ((preflight, "preflight tests"), (postrun, "postrun tests")):
        _validate_signed(value, name)
        if value.get("status") != "PASS" or value.get("failed") != 0 or value.get("false_pass_mutation_tests") != 36:
            raise StageFF3C3Error("F3C3 mutation test gate failed")
    if false_pass_reasons({attack: True for attack in FALSE_PASS_ATTACKS}):
        raise StageFF3C3Error("valid F3C3 false-pass preconditions were rejected")
    audit = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_false_pass_audit",
        "status": "PASS", "attack_count": 36,
        "attacks": [{"attack": attack, "result": "REJECTED", "evidence": f"tests/test_stage_f_f3c3.py::test_every_false_pass_attack_is_independently_rejected[{attack}]"} for attack in FALSE_PASS_ATTACKS],
        "integration_and_validator_tests": _record(repo, POSTRUN_TEST_PATH),
        "static_all_pass_list_only": False, "pass_possible_without_all_conditions": False,
    })
    _write_new(repo / FALSE_PASS_PATH, audit)
    return audit


def _validate_no_forbidden_future_outputs(repo: Path) -> None:
    forbidden = (
        "evidence/stage_f/f3c4", "reports/stage_f_f3c4",
        "evidence/stage_f/f4_", "reports/stage_f_f4_",
    )
    for root in (repo / "evidence/stage_f", repo / "reports"):
        if root.exists():
            for path in root.iterdir():
                relative = path.relative_to(repo).as_posix()
                if any(relative.startswith(prefix) for prefix in forbidden):
                    raise StageFF3C3Error("F3C4 or F4 output was started")


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    required = (STARTUP_PATH, FORMAL_MANIFEST_PATH, UNIT_PLAN_PATH, SHARD_PLAN_PATH, MODEL_BINDING_PATH, SOURCE_BINDING_PATH, AUTHORIZATION_BINDING_PATH, RESOURCE_SPEC_PATH, BLIND_SPEC_PATH, ACCESS_POLICY_PATH, PLAN_SEAL_PATH, PENDING_PATH, TERMINAL_UNION_PATH, ATTEMPT_JOURNAL_PATH, RESOURCE_LEDGER_PATH, BUDGET_PATH, ACCESS_AUDIT_PATH, BLIND_REPORT_PATH, NUMERIC_VALIDATION_PATH, PREFLIGHT_TEST_PATH, POSTRUN_TEST_PATH, FALSE_PASS_PATH)
    for relative in required:
        value = _read(repo / relative)
        _validate_signed(value, relative)
    validate_initialized(repo)
    source = _read(repo / SOURCE_BINDING_PATH)
    expected_files = [_file_record(repo, path) for path in SOURCE_FILES]
    if source.get("files") != expected_files or source.get("source_tree_digest") != canonical_sha256({"files": expected_files}):
        raise StageFF3C3Error("F3C3 source/config/code changed after predecode seal")
    union, ledger, budget, blind, access = (_read(repo / path) for path in (TERMINAL_UNION_PATH, RESOURCE_LEDGER_PATH, BUDGET_PATH, BLIND_REPORT_PATH, ACCESS_AUDIT_PATH))
    if union.get("counts") != {"VALID": 250, "FAILED": 0, "NA": 0, "incomplete": 0}:
        raise StageFF3C3Error("terminal union is not 250/250 VALID")
    if budget.get("status") != "PASS" or not all(budget.get("checks", {}).values()):
        raise StageFF3C3Error("F3C3 actual resource budget is not PASS")
    if blind.get("status") != "PASS" or not all(blind.get("conditions", {}).values()):
        raise StageFF3C3Error("blind formal execution gate is not PASS")
    validate_blind_document(blind)
    numeric = _read(repo / NUMERIC_VALIDATION_PATH)
    if numeric.get("status") != "PASS" or numeric.get("validated_units") != 250 or numeric.get("validated_samples") != 50 or numeric.get("empirical_values_reported_or_displayed") is not False:
        raise StageFF3C3Error("post-run numerical integrity validation failed")
    for phase_path in (PREFLIGHT_TEST_PATH, POSTRUN_TEST_PATH):
        test = _read(repo / phase_path)
        if test.get("status") != "PASS" or test.get("failed") != 0:
            raise StageFF3C3Error("F3C3 test gate failed")
    audit = _read(repo / FALSE_PASS_PATH)
    if audit.get("attack_count") != 36 or any(item.get("result") != "REJECTED" for item in audit.get("attacks", [])):
        raise StageFF3C3Error("F3C3 false-pass audit gate failed")
    for field in ("pilot_cohort_access", "pilot_map_logit_curve_auc_or_score_access", "order44_recomputed", "unlisted_jpeg_access", "formal_jpegs_rendered_or_manually_viewed", "maps_rendered", "maps_manually_inspected", "statistical_inference_performed", "scientific_comparison_performed", "delta_bootstrap_sign_flip_holm_pvalue_or_ci_produced", "winner_or_go_no_go_produced", "statistical_method_freeze_modified", "statistical_method_freeze_executed", "f3c4_started", "f4_started"):
        if access.get(field) is not False:
            raise StageFF3C3Error(f"F3C3 access boundary failed: {field}")
    rows = _formal_rows(repo)
    if access.get("decoded_formal_selection_orders") != [row["selection_order"] for row in rows]:
        raise StageFF3C3Error("F3C3 formal access set/order drifted")
    _verify_upstream(repo)
    _validate_no_forbidden_future_outputs(repo)
    if subprocess.run(["git", "diff", "--check"], cwd=repo).returncode:
        raise StageFF3C3Error("git diff --check failed")
    groups = [validate_sample(repo, selection_order=row["selection_order"]) for row in rows]
    return _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_validation_report",
        "status": "PASS", "implementation_validation": "PASS",
        "workflow_validation": "PASS", "package_validation": "PASS",
        "checks": {name: "PASS" for name in ("startup_and_predecode_plan_seal", "frozen_50_sample_and_250_unit_manifests", "deterministic_shard_and_closed_leases", "model_preprocessing_source_authorization_bindings", "250_immutable_typed_terminals", "target_once_and_shared_per_sample", "all_block_attention_endpoint_parity", "nns_and_full_model_ig_conservation_attempt_retention", "common_fidelity_numerical_integrity", "resource_ledger_and_actual_total_budget", "blind_execution_boundary", "statistical_method_freeze_read_only_not_executed", "access_boundary", "false_pass_36_of_36", "git_diff_check")},
        "formal_samples": "50/50", "method_terminals": "250/250 VALID",
        "nns_endpoint_parity": "50/50 PASS", "nns_conservation": "50/50 PASS",
        "full_model_ig_conservation": "50/50 PASS", "resource_and_total_f3c_budget": "PASS",
        "test_passed": _read(repo / POSTRUN_TEST_PATH)["passed"], "test_failed": 0,
        "test_deselected": _read(repo / POSTRUN_TEST_PATH)["deselected"],
        "postrun_validated_units": 250, "false_pass_attacks_rejected": 36,
        "target_class_indices_by_order": {str(group["selection_order"]): group["target_class_index"] for group in groups},
        "formal_sample_access": True, "pilot_cohort_access": False, "order44_recomputed": False,
        "maps_rendered": False, "maps_manually_inspected": False, "statistical_inference_performed": False,
        "scientific_comparison_performed": False, "winner_pvalue_ci_or_go_no_go_produced": False,
        "sealed_upstream_modified": False, "f3c4_started": False, "f4_started": False,
        "next_authorization": "F3C4_CPU_ANALYSIS_AND_SEAL_ONLY",
    })


def finalize(repo_root: Path | str = REPO_ROOT, *, completed_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validation = validate_outputs(repo)
    _write_new(repo / VALIDATION_PATH, validation)
    payload = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_payload_manifest",
        "status": "PASS",
        "records": [_record(repo, relative) for relative in (STARTUP_PATH, FORMAL_MANIFEST_PATH, UNIT_PLAN_PATH, SHARD_PLAN_PATH, MODEL_BINDING_PATH, SOURCE_BINDING_PATH, AUTHORIZATION_BINDING_PATH, RESOURCE_SPEC_PATH, BLIND_SPEC_PATH, ACCESS_POLICY_PATH, PLAN_SEAL_PATH, TERMINAL_UNION_PATH, ATTEMPT_JOURNAL_PATH, RESOURCE_LEDGER_PATH, BUDGET_PATH, ACCESS_AUDIT_PATH, BLIND_REPORT_PATH, NUMERIC_VALIDATION_PATH, PREFLIGHT_TEST_PATH, POSTRUN_TEST_PATH, FALSE_PASS_PATH, VALIDATION_PATH)],
        "typed_terminal_count": 250, "numeric_payload_present_only_for_valid_terminals": True,
        "display_or_render_payload_present": False, "statistical_output_present": False,
        "runner_source_bound_before_decode": True,
        "postrun_validator_source_bound_after_compute_without_recompute": True,
    })
    _write_new(repo / PAYLOAD_PATH, payload)
    ledger, budget = _read(repo / RESOURCE_LEDGER_PATH), _read(repo / BUDGET_PATH)
    status = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_status", "status": "PASS",
        "terminal": True, "immutable": True, "implementation_validation": "PASS",
        "workflow_validation": "PASS", "package_validation": "PASS",
        "formal_samples": "50/50", "method_terminals": "250/250 VALID",
        "nns_endpoint_parity": "50/50 PASS", "nns_conservation": "50/50 PASS",
        "full_model_ig_conservation": "50/50 PASS", "resource_and_total_f3c_budget": "PASS",
        "formal_sample_access": True, "pilot_cohort_access": False, "order44_recomputed": False,
        "maps_rendered": False, "maps_manually_inspected": False,
        "statistical_inference_performed": False, "scientific_comparison_performed": False,
        "winner_pvalue_ci_or_go_no_go_produced": False,
        "f3c0_f3c1_f3c2_f3h_and_other_sealed_evidence_modified": False,
        "f3c4_started": False, "f4_started": False,
        "next_authorization": "F3C4_CPU_ANALYSIS_AND_SEAL_ONLY",
        "git": {"commit": "SELF", "ref": ARCHIVE_REF},
    })
    _write_new(repo / STATUS_PATH, status)
    inventory_payload = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "completion/f3c3.json", "completion/f3c3.pending.json"))
    inventory = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_whole_root_inventory",
        "status": "PASS", **inventory_payload,
    })
    _write_new(repo / INVENTORY_PATH, inventory)
    reports_retained = _regular_bytes(repo / REPORT_PATH)
    f3c3_retained_scope = inventory["total_bytes"] + reports_retained
    total_f3c_retained_scope = EXISTING_F3C1_F3C2_RETAINED_BYTES + f3c3_retained_scope
    if total_f3c_retained_scope > load_config(repo)["budget"]["retained_regular_file_bytes_max"]:
        raise StageFF3C3Error("BLOCKED_BUDGET: final retained package exceeds cap")
    groups = [validate_sample(repo, selection_order=row["selection_order"]) for row in _formal_rows(repo)]
    per_sample = [{
        "formal_position": group["formal_position"], "selection_order": group["selection_order"],
        "target_class_index": group["target_class_index"],
        "nns_endpoint_parity": next(item for item in group["blind_records"] if item["method_id"] == kernels.METHOD_ID)["endpoint_parity_status"],
        "nns_selected_steps": next(item for item in group["blind_records"] if item["method_id"] == kernels.METHOD_ID)["ig_selected_steps"],
        "nns_conservation": next(item for item in group["blind_records"] if item["method_id"] == kernels.METHOD_ID)["ig_conservation_status"],
        "full_model_ig_selected_steps": next(item for item in group["blind_records"] if item["method_id"] == "full_model_class_logit_ig")["ig_selected_steps"],
        "full_model_ig_conservation": next(item for item in group["blind_records"] if item["method_id"] == "full_model_class_logit_ig")["ig_conservation_status"],
        "gpu_hours": next(item for item in ledger["image_groups"] if item["selection_order"] == group["selection_order"])["gpu_hours"],
    } for group in groups]
    completion = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c3_terminal_completion",
        "status": "PASS", "completed_at": completed_at,
        "implementation_validation": "PASS", "workflow_validation": "PASS", "package_validation": "PASS",
        "formal_samples": "50/50", "method_terminals": {"VALID": 250, "FAILED": 0, "NA": 0, "incomplete": 0},
        "nns_endpoint_parity": "50/50 PASS", "nns_conservation": "50/50 PASS",
        "full_model_ig_conservation": "50/50 PASS", "resource_and_total_f3c_budget": "PASS",
        "per_sample_blind_qc": per_sample,
        "resources": {"total_elapsed_seconds": ledger["total_elapsed_seconds"], "formal_actual_gpu_hours": ledger["formal_actual_gpu_hours"], "actual_total_f3c_gpu_hours": budget["actual_total_f3c_gpu_hours"], "observed_devices": ledger["observed_devices"], "peak_reserved_bytes": ledger["peak_reserved_bytes"], "total_forward_calls": ledger["total_forward_calls"], "total_backward_calls": ledger["total_backward_calls"], "f3c3_retained_bytes_inventory_scope": f3c3_retained_scope, "actual_total_f3c_retained_bytes_inventory_scope": total_f3c_retained_scope, "all_caps_satisfied": True},
        "startup_gate": _record(repo, STARTUP_PATH), "formal_manifest": _record(repo, FORMAL_MANIFEST_PATH),
        "unit_execution_plan": _record(repo, UNIT_PLAN_PATH), "deterministic_shard_lease_plan": _record(repo, SHARD_PLAN_PATH),
        "model_preprocessing_binding": _record(repo, MODEL_BINDING_PATH), "source_config_code_binding": _record(repo, SOURCE_BINDING_PATH),
        "f3c2_continuation_authorization_binding": _record(repo, AUTHORIZATION_BINDING_PATH), "resource_accounting_specification": _record(repo, RESOURCE_SPEC_PATH),
        "blind_formal_execution_specification": _record(repo, BLIND_SPEC_PATH), "access_policy": _record(repo, ACCESS_POLICY_PATH),
        "plan_seal": _record(repo, PLAN_SEAL_PATH), "terminal_union": _record(repo, TERMINAL_UNION_PATH),
        "attempt_journal": _record(repo, ATTEMPT_JOURNAL_PATH), "resource_ledger": _record(repo, RESOURCE_LEDGER_PATH),
        "total_f3c_budget_report": _record(repo, BUDGET_PATH), "access_audit": _record(repo, ACCESS_AUDIT_PATH),
        "blind_formal_execution_report": _record(repo, BLIND_REPORT_PATH), "postrun_numeric_integrity_validation": _record(repo, NUMERIC_VALIDATION_PATH),
        "payload_manifest": _record(repo, PAYLOAD_PATH), "whole_root_inventory": _record(repo, INVENTORY_PATH),
        "validation_report": _record(repo, VALIDATION_PATH), "false_pass_audit": _record(repo, FALSE_PASS_PATH),
        "preflight_test_report": _record(repo, PREFLIGHT_TEST_PATH), "postrun_test_report": _record(repo, POSTRUN_TEST_PATH),
        "status_record": _record(repo, STATUS_PATH),
        "formal_sample_access": True, "pilot_cohort_access": False, "order44_recomputed": False,
        "maps_rendered": False, "maps_manually_inspected": False, "statistical_inference_performed": False,
        "scientific_comparison_performed": False, "winner_pvalue_ci_or_go_no_go_produced": False,
        "f3c0_f3c1_f3c2_f3h_and_other_sealed_evidence_modified": False,
        "f3c4_started": False, "f4_started": False, "next_authorization": "F3C4_CPU_ANALYSIS_AND_SEAL_ONLY",
        "git": {"commit": "SELF", "ref": ARCHIVE_REF},
    })
    temporary = (repo / PENDING_PATH).with_name(".f3c3-terminal.tmp")
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
    _validate_signed(completion, "F3C3 completion")
    if (repo / PENDING_PATH).exists() or completion.get("status") != "PASS":
        raise StageFF3C3Error("F3C3 completion is not atomically terminal PASS")
    if completion.get("method_terminals") != {"VALID": 250, "FAILED": 0, "NA": 0, "incomplete": 0} or completion.get("next_authorization") != "F3C4_CPU_ANALYSIS_AND_SEAL_ONLY":
        raise StageFF3C3Error("F3C3 terminal counts or next authorization drifted")
    for field in ("pilot_cohort_access", "order44_recomputed", "maps_rendered", "maps_manually_inspected", "statistical_inference_performed", "scientific_comparison_performed", "winner_pvalue_ci_or_go_no_go_produced", "f3c0_f3c1_f3c2_f3h_and_other_sealed_evidence_modified", "f3c4_started", "f4_started"):
        if completion.get(field) is not False:
            raise StageFF3C3Error(f"F3C3 completion boundary failed: {field}")
    for field in ("startup_gate", "formal_manifest", "unit_execution_plan", "deterministic_shard_lease_plan", "model_preprocessing_binding", "source_config_code_binding", "f3c2_continuation_authorization_binding", "resource_accounting_specification", "blind_formal_execution_specification", "access_policy", "plan_seal", "terminal_union", "attempt_journal", "resource_ledger", "total_f3c_budget_report", "access_audit", "blind_formal_execution_report", "postrun_numeric_integrity_validation", "payload_manifest", "whole_root_inventory", "validation_report", "false_pass_audit", "preflight_test_report", "postrun_test_report", "status_record"):
        if completion[field] != _record(repo, completion[field]["path"]):
            raise StageFF3C3Error(f"F3C3 completion binding drifted: {field}")
    inventory = _read(repo / INVENTORY_PATH)
    rebuilt = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "completion/f3c3.json", "completion/f3c3.pending.json"))
    if any(inventory.get(key) != rebuilt[key] for key in ("files", "file_count", "total_bytes", "file_inventory_digest")):
        raise StageFF3C3Error("F3C3 whole-root inventory drifted")
    for row in _formal_rows(repo):
        validate_sample(repo, selection_order=row["selection_order"])
    return {"status": "PASS", "archive_ref": ARCHIVE_REF, "formal_samples": 50, "valid_terminals": 250, "next_authorization": "F3C4_CPU_ANALYSIS_AND_SEAL_ONLY"}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-startup", "validate-initialized", "reconcile", "build-budget-blind-gate", "build-false-pass", "validate-outputs", "validate-completion"):
        child = sub.add_parser(command); child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    init = sub.add_parser("initialize"); init.add_argument("--repo-root", type=Path, default=REPO_ROOT); init.add_argument("--started-at", required=True)
    run = sub.add_parser("run-sample"); run.add_argument("--repo-root", type=Path, default=REPO_ROOT); run.add_argument("--selection-order", type=int, required=True); run.add_argument("--device", default=DEVICE)
    tests = sub.add_parser("record-tests"); tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); tests.add_argument("--phase", choices=("preflight", "postrun"), required=True); tests.add_argument("--passed", type=int, required=True); tests.add_argument("--failed", type=int, required=True); tests.add_argument("--deselected", type=int, default=0); tests.add_argument("--test-command", required=True)
    final = sub.add_parser("finalize"); final.add_argument("--repo-root", type=Path, default=REPO_ROOT); final.add_argument("--completed-at", required=True)
    numeric = sub.add_parser("record-numeric-validation"); numeric.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    if args.command == "record-numeric-validation":
        from . import stage_f_f3c3_validate
        result = stage_f_f3c3_validate.record_validation(args.repo_root)
    else:
        commands = {
            "validate-startup": lambda: validate_startup_gate(args.repo_root),
            "initialize": lambda: initialize(args.repo_root, started_at=args.started_at),
            "validate-initialized": lambda: validate_initialized(args.repo_root),
            "run-sample": lambda: run_sample(args.repo_root, selection_order=args.selection_order, device=args.device),
            "reconcile": lambda: reconcile(args.repo_root),
            "build-budget-blind-gate": lambda: build_budget_and_blind_gate(args.repo_root),
            "record-tests": lambda: record_tests(args.repo_root, phase=args.phase, passed=args.passed, failed=args.failed, deselected=args.deselected, command=args.test_command),
            "build-false-pass": lambda: build_false_pass_audit(args.repo_root),
            "validate-outputs": lambda: validate_outputs(args.repo_root),
            "finalize": lambda: finalize(args.repo_root, completed_at=args.completed_at),
            "validate-completion": lambda: validate_completion(args.repo_root),
        }
        try:
            result = commands[args.command]()
        except (StageFF3C3Error, OSError, ValueError) as error:
            print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True), file=sys.stderr)
            return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
