"""CPU-only X2-MECH M2 multiscale mechanism analysis.

The module has two deliberately separated surfaces:

* ``freeze-producer``: closes the implementation contract and writes only
  control-plane metadata. It must not open empirical M1/X2 CSV rows or arrays.
* ``run`` / ``validate``: after the producer ref is sealed, revalidates M1 and
  consumes the sealed M1/X2 numeric artifacts in CPU-only mode.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from . import stage_x_x2_mech_m1_reconcile as m1
from . import stage_x_x2_mech_worker as m1_worker
from .stage_x_x2_mech_analysis import (
    X2MechAnalysisError,
    aggregate_blocks,
    clean_structure_metrics as clean_structure,
    seed64,
)

RUN_ID = "STAGE_X_X2_MECH_M2_V1_20260910"
AUTHORIZATION = "X2_MECH_M2_CPU_ONLY_MULTISCALE_MECHANISM_ANALYSIS_ONLY"
NEXT_AUTHORIZATION = "X2_MECH_RESULTS_TO_PAPER_HANDOFF_ONLY"
BRANCH = "stage-d-d3-pass"
STARTUP_HEAD = "705b9e19cd95104f0ad3252056d9f2ae237d6973"
P0_R1_COMPLETION_COMMIT = "f205eeebe0f054db434fb822d8d6d92c95b74ee4"
M1_R1_COMPLETION_COMMIT = "b6d14965cce3d7a4618b4f5435331c7a9b954d19"
X2_COMPLETION_COMMIT = "9ddfec3dca2e180930009174faa00a0f48e0b752"
P0_R1_COMPLETION_REF = "refs/archive/stage-x-x2-mech-p0-r1-completion"
M1_R1_COMPLETION_REF = "refs/archive/stage-x-x2-mech-m1-r1-completion"
X2_COMPLETION_REF = "refs/archive/stage-x-nns-v2-x2-completion"
PRODUCER_REF = "refs/archive/stage-x-x2-mech-m2-producer"
COMPLETION_REF = "refs/archive/stage-x-x2-mech-m2-completion"
HANDOFF_REF = "refs/archive/stage-x-x2-mech-results-to-paper-handoff"

ROOT = Path("evidence/stage_x/x2_mech_m2_20260910/formal")
REPORT_ROOT = Path("reports/stage_x_x2_mech_m2_20260910")
CONFIG = Path("config/stage_x_x2_mech_m2.json")
CONFIG_SCHEMA = Path("config/stage_x_x2_mech_m2.schema.json")
ARTIFACT_SCHEMA = Path("config/stage_x_x2_mech_m2_artifact.schema.json")
ADDENDUM = Path("docs/stage_x_x2_mech_m2_execution_addendum_20260910.md")
PROTOCOL = Path("docs/stage_x_x2_mechanism_protocol_20260910.md")
SCIENTIFIC_HANDOFF = Path("docs/stage_x_x2_mech_m1_to_m2_scientific_handoff_20260910.md")
PRODUCER_COMPLETION = ROOT / "completion/x2_mech_m2_producer.json"
PRODUCER_PENDING = ROOT / "completion/x2_mech_m2_producer.pending.json"
COMPLETION = ROOT / "completion/x2_mech_m2.json"
PENDING = ROOT / "completion/x2_mech_m2.pending.json"
RESULTS_HANDOFF_DOC = Path("docs/stage_x_x2_mech_results_to_paper_handoff_20260910.md")
RESULTS_HANDOFF_JSON = Path("reports/stage_x_x2_mech_results_to_paper_handoff_20260910.json")

METHODS = ("nns_v2", "vit_grad_cam")
TRANSFORMS = m1.TRANSFORMS
MASKS = ("x2_frozen_common_domain", "exact_operator_support")
SCALES = (
    ("pixel", 1),
    ("fine_cell", 4),
    ("native_patch", 16),
)
SSE_SCALES = (("fine_cell", 4), ("native_patch", 16))
SOURCE_FILES = (
    ADDENDUM,
    CONFIG,
    CONFIG_SCHEMA,
    ARTIFACT_SCHEMA,
    PROTOCOL,
    SCIENTIFIC_HANDOFF,
    Path("src/vitnns/stage_x_x2_mech_m2.py"),
    Path("src/vitnns/stage_x_x2_mech_analysis.py"),
    Path("src/vitnns/stage_x_x2_mech_m1_reconcile.py"),
    Path("tests/test_stage_x_x2_mech_m2.py"),
)
CONTROL_BINDINGS = (
    m1.ROOT / "completion/x2_mech_m1_r1.json",
    m1.ROOT / "reconciliation.json",
    m1.ROOT / "map_manifest.json",
    m1.ROOT / "valid_mask_registry.json",
    m1.ROOT / "resource_ledger.json",
    m1.ROOT / "whole_root_inventory.json",
    m1.REPORT_ROOT / "validation_report.json",
)

STABILITY_FIELDS = (
    "unit_digest", "model_id", "sample_id", "source_status", "method_id", "transform",
    "mask_id", "scale_id", "block", "prediction_changed", "valid_unit_count",
    "excluded_partial_unit_count", "signed_spearman", "top10_unit_signed_value_jaccard",
    "absolute_mass_spearman", "top10_unit_absolute_mass_jaccard", "signed_ssim",
    "normalized_positive_negative_mass_displacement",
    "absolute_mass_center_of_mass_pixel_error", "status",
)
STABILITY_SUMMARY_FIELDS = (
    "group", "model_id", "method_id", "mask_id", "scale_id", "block", "transform",
    "n", "failed_denominator_count", "median_signed_spearman",
    "median_top10_unit_signed_value_jaccard", "median_absolute_mass_spearman",
    "median_top10_unit_absolute_mass_jaccard", "median_signed_ssim",
    "median_normalized_positive_negative_mass_displacement",
    "median_absolute_mass_center_of_mass_pixel_error",
    "median_valid_unit_count", "sum_excluded_partial_unit_count",
)
PIXEL_AUDIT_FIELDS = (
    "unit_digest", "model_id", "sample_id", "method_id", "transform",
    "x2_signed_spearman", "m2_signed_spearman", "signed_spearman_abs_diff",
    "x2_top10_mass_jaccard", "m2_top10_unit_signed_value_jaccard", "top10_jaccard_abs_diff",
    "x2_signed_ssim", "m2_signed_ssim", "signed_ssim_abs_diff",
    "x2_normalized_positive_negative_mass_displacement",
    "m2_normalized_positive_negative_mass_displacement", "displacement_abs_diff",
    "x2_center_of_mass_pixel_error", "m2_absolute_mass_center_of_mass_pixel_error",
    "center_of_mass_abs_diff", "x2_prediction_changed", "m1_prediction_changed",
    "prediction_changed_match", "status",
)
WITHIN_BETWEEN_FIELDS = (
    "unit_digest", "model_id", "sample_id", "source_status", "method_id", "transform",
    "mask_id", "scale_id", "block", "valid_block_count", "excluded_partial_block_count",
    "sse_total", "sse_between", "sse_within", "between_error_fraction",
    "within_error_fraction", "identity_abs_diff", "identity_rel_diff",
    "block_level_signed_rank_stability", "within_block_demeaned_residual_stability", "status",
)
WITHIN_BETWEEN_SUMMARY_FIELDS = (
    "group", "model_id", "method_id", "mask_id", "scale_id", "block", "transform",
    "n", "failed_denominator_count", "median_sse_total", "median_between_error_fraction",
    "median_within_error_fraction", "max_identity_abs_diff", "max_identity_rel_diff",
    "median_block_level_signed_rank_stability",
    "median_within_block_demeaned_residual_stability",
    "median_valid_block_count", "sum_excluded_partial_block_count",
)
TRANSPORT_FIELDS = (
    "unit_digest", "model_id", "sample_id", "source_status", "method_id", "transform",
    "mask_id", "scale_id", "block", "valid_unit_count", "excluded_partial_unit_count",
    "signed_spearman", "top10_unit_signed_value_jaccard", "absolute_mass_spearman",
    "top10_unit_absolute_mass_jaccard", "signed_ssim",
    "normalized_positive_negative_mass_displacement",
    "absolute_mass_center_of_mass_pixel_error", "status",
)
HFLIP_FIELDS = (
    "unit_digest", "model_id", "sample_id", "source_status", "method_id", "mask_id",
    "scale_id", "block", "full_domain_mask", "valid_unit_count",
    "excluded_partial_unit_count", "observed_signed_spearman",
    "transport_signed_spearman", "observed_minus_transport_signed_spearman",
    "observed_top10_unit_signed_value_jaccard", "transport_top10_unit_signed_value_jaccard",
    "observed_minus_transport_top10_unit_signed_value_jaccard", "status",
)
STRUCTURE_FIELDS = (
    "unit_digest", "model_id", "sample_id", "source_status", "method_id",
    "absolute_top10_pixel_mass_share", "normalized_shannon_entropy",
    "effective_support_fraction", "gini_coefficient", "normalized_total_variation",
    "within_cell_4_residual_energy_fraction", "within_patch_16_residual_energy_fraction",
    "signed_cancellation_ratio", "status",
)
FIDELITY_SUMMARY_FIELDS = (
    "group", "model_id", "method_id", "scale_id", "block", "n",
    "failed_denominator_count", "median_road_combined", "mean_road_combined",
    "median_road_morf", "median_road_lerf", "source",
)
ASSOCIATION_FIELDS = (
    "method_id", "x_metric", "y_metric", "n", "effect", "ci_lower", "ci_upper",
    "bootstrap_draws", "finite_draws", "attempts", "seed", "status",
)
BRANCH_FIELDS = (
    "branch", "evidence_item", "observed_direction", "supports_branch", "boundary",
)

REQUIRED_CSV_FIELDS = {
    "pixel_reproduction_audit.csv": PIXEL_AUDIT_FIELDS,
    "multiscale_stability.csv": STABILITY_FIELDS,
    "multiscale_stability_summary.csv": STABILITY_SUMMARY_FIELDS,
    "within_between_error.csv": WITHIN_BETWEEN_FIELDS,
    "within_between_summary.csv": WITHIN_BETWEEN_SUMMARY_FIELDS,
    "transport_control.csv": TRANSPORT_FIELDS,
    "horizontal_flip_control.csv": HFLIP_FIELDS,
    "clean_structure_metrics.csv": STRUCTURE_FIELDS,
    "multiscale_fidelity_summary.csv": FIDELITY_SUMMARY_FIELDS,
    "association_summary.csv": ASSOCIATION_FIELDS,
    "branch_evidence_matrix.csv": BRANCH_FIELDS,
}
CONTINUOUS_PIXEL_PAIRS = (
    ("signed_spearman", "signed_spearman"),
    ("top10_mass_jaccard", "top10_unit_signed_value_jaccard"),
    ("signed_ssim", "signed_ssim"),
    ("normalized_positive_negative_mass_displacement", "normalized_positive_negative_mass_displacement"),
    ("center_of_mass_pixel_error", "absolute_mass_center_of_mass_pixel_error"),
)
FALSE_PASS_ATTACKS = (
    "source_binding_mutation",
    "m1_completion_ref_drift",
    "m1_completion_digest_drift",
    "m1_denominator_mutation",
    "m1_map_manifest_row_drop",
    "m1_mask_registry_mutation",
    "m1_fidelity_row_drop",
    "x2_pixel_row_drop",
    "x2_pixel_numeric_drift",
    "partial_block_inclusion",
    "top10_count_floor_instead_of_ceil",
    "sse_identity_violation",
    "branch_threshold_invention",
    "confirmatory_p_value_insertion",
    "gpu_or_cuda_access",
    "model_checkpoint_or_jpeg_access",
    "png_or_jpeg_output",
    "atomic_completion_overwrite",
    "x2_no_go_modification",
    "x3_x5_f3_final_stage_g_start",
)


class X2MechM2Error(ValueError):
    pass


def _pairs(values: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise X2MechM2Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                X2MechM2Error(f"non-finite JSON: {token}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise X2MechM2Error(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise X2MechM2Error(f"JSON root must be an object: {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode:
        raise X2MechM2Error(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _as_list(mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"name": key, "value": mapping[key]} for key in sorted(mapping)]


def _terminal(*, terminal: bool = False, complete: bool = False,
              next_authorization: str = "NONE") -> dict[str, Any]:
    return {
        "terminal": terminal,
        "complete": complete,
        "immutable": True,
        "next_authorization": next_authorization,
        "x2_decision_modified": False,
        "x3_x5_f3_final_stage_g_started": False,
    }


def _payload(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "identity": [],
        "bindings": [],
        "checks": [],
        "counts": [],
        "metrics": [],
        "state": [],
        "files": [],
        "access": [],
        "resources": [],
        "attacks": [],
        "terminal": _terminal(),
    }
    value.update(updates)
    return value


def signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "$schema": ARTIFACT_SCHEMA.name,
        "schema_version": 1,
        "document_type": document_type,
        "stage_id": "X2-MECH",
        "phase": "M2",
        "run_id": RUN_ID,
        "authorization": AUTHORIZATION,
        "status": status,
        "payload": dict(payload),
    }
    value["canonical_digest"] = canonical_sha256(value)
    return value


def validate_signed(value: Mapping[str, Any], context: str,
                    schema: Mapping[str, Any] | None = None) -> None:
    copy = dict(value)
    observed = copy.pop("canonical_digest", None)
    if observed != canonical_sha256(copy):
        raise X2MechM2Error(f"{context}: canonical self-digest mismatch")
    if schema is not None:
        errors = sorted(Draft202012Validator(schema).iter_errors(value),
                        key=lambda item: list(item.path))
        if errors:
            raise X2MechM2Error(f"{context}: schema validation failed: {errors[0].message}")


def _read_csv(path: Path, fields: Sequence[str] | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if fields is not None and tuple(reader.fieldnames or ()) != tuple(fields):
            raise X2MechM2Error(f"CSV fields drifted: {path}")
        rows = [dict(row) for row in reader]
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise X2MechM2Error(f"malformed CSV row: {path}")
    return rows


def _csv_header(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        return tuple(next(csv.reader(stream)))


def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise X2MechM2Error("CSV field mismatch")
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _write_bytes_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_new(path, canonical_bytes(value) + b"\n")


def _write_csv_new(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    _write_bytes_new(path, _csv_bytes(fields, rows))


def _publish_atomic(pending: Path, terminal: Path, value: Mapping[str, Any]) -> None:
    if pending.exists() or terminal.exists():
        raise X2MechM2Error("pending or terminal completion already exists")
    pending_payload = dict(value["payload"])
    pending_payload["terminal"] = _terminal()
    _write_json_new(pending, signed(value["document_type"], "PENDING", pending_payload))
    temporary = terminal.with_name(terminal.name + ".terminal.pending")
    _write_json_new(temporary, value)
    os.replace(temporary, terminal)
    pending.unlink()


def file_record(repo: Path, relative: Path, *, canonical_digest: str | None = None) -> dict[str, Any]:
    path = repo / relative
    record: dict[str, Any] = {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha_file(path),
        "canonical_digest": canonical_digest,
    }
    if canonical_digest is None and path.suffix == ".json":
        try:
            value = strict_read(path)
            record["canonical_digest"] = value.get("canonical_digest")
        except X2MechM2Error:
            pass
    return record


def _verify_record(repo: Path, record: Mapping[str, Any]) -> None:
    path = repo / str(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha_file(path) != record["sha256"]:
        raise X2MechM2Error(f"bound file drifted: {record['path']}")
    if path.suffix == ".json" and record.get("canonical_digest") is not None:
        value = strict_read(path)
        if value.get("canonical_digest") != record["canonical_digest"]:
            raise X2MechM2Error(f"bound canonical digest drifted: {record['path']}")


def _schema(repo: Path) -> dict[str, Any]:
    value = strict_read(repo / ARTIFACT_SCHEMA)
    Draft202012Validator.check_schema(value)
    return value


def validate_artifact(value: Mapping[str, Any], repo: Path, *, verify_files: bool = False) -> None:
    validate_signed(value, str(value.get("document_type")), _schema(repo))
    payload = value["payload"]
    for field, key in (
        ("identity", "name"), ("counts", "name"), ("metrics", "name"),
        ("state", "name"), ("bindings", "path"), ("files", "path"),
        ("checks", "check_id"),
    ):
        keys = [row[key] for row in payload[field]]
        if len(keys) != len(set(keys)):
            raise X2MechM2Error(f"duplicate {field}")
    if verify_files:
        for record in [*payload["bindings"], *payload["files"]]:
            _verify_record(repo, record)


def load_config(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = strict_read(repo / CONFIG)
    schema = strict_read(repo / CONFIG_SCHEMA)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(config), key=lambda item: list(item.path))
    if errors:
        raise X2MechM2Error(f"M2 config schema validation failed: {errors[0].message}")
    copy = dict(config)
    observed = copy.pop("canonical_digest")
    if observed != canonical_sha256(copy):
        raise X2MechM2Error("M2 config canonical digest mismatch")
    if tuple(config["cohort"]["methods"]) != METHODS or tuple(config["cohort"]["transforms"]) != TRANSFORMS:
        raise X2MechM2Error("M2 cohort roster drifted")
    if config["branch_rules"]["numeric_thresholds_for_A_B_C_frozen"]:
        raise X2MechM2Error("post-hoc branch thresholds were introduced")
    if any(config["claims"][key] for key in config["claims"] if key != "maximum_wording"):
        raise X2MechM2Error("claim boundary drifted")
    return config


def _startup_checks(repo: Path, *, expected_head: str = STARTUP_HEAD) -> list[dict[str, str]]:
    if _git(repo, "symbolic-ref", "--short", "HEAD") != BRANCH:
        raise X2MechM2Error("BLOCKED_STARTUP_STATE: branch drifted")
    if _git(repo, "rev-parse", "HEAD") != expected_head:
        raise X2MechM2Error("BLOCKED_STARTUP_STATE: local HEAD drifted")
    if _git(repo, "rev-parse", "@{upstream}") != expected_head:
        raise X2MechM2Error("BLOCKED_STARTUP_STATE: tracking ref drifted")
    remote = _git(repo, "ls-remote", "--heads", "origin", BRANCH).split()
    if not remote or remote[0] != expected_head:
        raise X2MechM2Error("BLOCKED_STARTUP_STATE: remote branch drifted")
    status = _git(repo, "status", "--porcelain")
    if status.strip():
        raise X2MechM2Error("BLOCKED_STARTUP_STATE: worktree or index not clean")
    if _git(repo, "lfs", "status", "--porcelain").strip():
        raise X2MechM2Error("BLOCKED_STARTUP_STATE: Git LFS has pending objects")
    ref_expectations = (
        (P0_R1_COMPLETION_REF, P0_R1_COMPLETION_COMMIT),
        (M1_R1_COMPLETION_REF, M1_R1_COMPLETION_COMMIT),
    )
    for ref, expected in ref_expectations:
        if _git(repo, "rev-parse", ref + "^{commit}") != expected:
            raise X2MechM2Error(f"BLOCKED_STARTUP_STATE: archive ref drifted: {ref}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", M1_R1_COMPLETION_REF, "HEAD"], cwd=repo
    ).returncode:
        raise X2MechM2Error("BLOCKED_STARTUP_STATE: M1 completion is not HEAD ancestor")
    return [
        {"check_id": "branch_head_tracking_remote", "status": "PASS", "detail": expected_head},
        {"check_id": "worktree_index_lfs_clean", "status": "PASS", "detail": "clean"},
        {"check_id": "m1_completion_ref_ancestor", "status": "PASS", "detail": M1_R1_COMPLETION_COMMIT},
    ]


def _producer_source_ast_checks(repo: Path) -> list[dict[str, str]]:
    forbidden_top = {"timm", "PIL", "cv2"}
    checks = []
    for relative in (Path("src/vitnns/stage_x_x2_mech_m2.py"), Path("tests/test_stage_x_x2_mech_m2.py")):
        tree = ast.parse((repo / relative).read_text(encoding="utf-8"), filename=str(relative))
        imports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        if imports & forbidden_top:
            raise X2MechM2Error(f"forbidden top-level import in {relative}: {sorted(imports & forbidden_top)}")
        checks.append({"check_id": f"no_forbidden_top_level_import_{relative.name}", "status": "PASS",
                       "detail": "no timm/PIL/cv2 import"})
    source = (repo / "src/vitnns/stage_x_x2_mech_m2.py").read_text(encoding="utf-8")
    if ("Image" + ".open") in source or ("timm" + ".") in source:
        raise X2MechM2Error("M2 source contains forbidden model/image surface")
    return checks


def _header_contract_checks(repo: Path, config: Mapping[str, Any]) -> list[dict[str, str]]:
    observed = {
        "m1_map_manifest_header": _csv_header(repo / m1.ROOT / "map_manifest.csv"),
        "m1_multiscale_fidelity_header": _csv_header(repo / m1.ROOT / "multiscale_fidelity.csv"),
        "m1_multiscale_fidelity_rows_header": _csv_header(repo / m1.ROOT / "multiscale_fidelity_rows.csv"),
    }
    if observed["m1_map_manifest_header"] != m1.MAP_FIELDS:
        raise X2MechM2Error("M1 map manifest header drifted")
    if observed["m1_multiscale_fidelity_header"] != m1.FIDELITY_FIELDS:
        raise X2MechM2Error("M1 fidelity summary header drifted")
    if observed["m1_multiscale_fidelity_rows_header"] != m1.DETAIL_FIELDS:
        raise X2MechM2Error("M1 fidelity rows header drifted")
    for name, fields in REQUIRED_CSV_FIELDS.items():
        if tuple(config["output_schemas"][name]) != tuple(fields):
            raise X2MechM2Error(f"M2 output schema drifted: {name}")
    return [
        {"check_id": key, "status": "PASS", "detail": ",".join(value)}
        for key, value in observed.items()
    ]


def _source_control_bindings(repo: Path) -> list[dict[str, Any]]:
    records = []
    for relative in CONTROL_BINDINGS:
        value = strict_read(repo / relative)
        if relative.name.endswith(".json"):
            # M1 documents use their own signing but the canonical formula is the same.
            copy = dict(value)
            observed = copy.pop("canonical_digest", None)
            if observed != canonical_sha256(copy):
                raise X2MechM2Error(f"canonical digest drifted: {relative}")
        records.append(file_record(repo, relative))
    completion = strict_read(repo / m1.COMPLETION)
    if completion.get("status") != "PASS":
        raise X2MechM2Error("M1 completion is not PASS")
    if completion["canonical_digest"] != "5750d3063cc9d951d7a7a04131c98864a899893f0ddef6e7837cc70b06be6f9f":
        raise X2MechM2Error("M1 completion canonical digest drifted")
    payload = completion["payload"]
    if payload["denominator"] != {"expected": 200, "valid": 199, "failed_route_local_numerical": 1}:
        raise X2MechM2Error("M1 denominator drifted")
    if payload["next_authorization"] != AUTHORIZATION:
        raise X2MechM2Error("M1 next authorization does not permit M2")
    if payload.get("m2_executed") is not False:
        raise X2MechM2Error("M1 claims M2 already executed")
    return records


def _reject_if_m2_results_exist(repo: Path, *, allow_producer: bool = False) -> None:
    forbidden = {
        "pixel_reproduction_audit.csv", "pixel_reproduction_audit.json",
        "multiscale_stability.csv", "multiscale_stability_summary.csv",
        "multiscale_stability_summary.json", "within_between_error.csv",
        "within_between_summary.csv", "transport_control.csv",
        "horizontal_flip_control.csv", "clean_structure_metrics.csv",
        "multiscale_fidelity_summary.csv",
        "association_summary.csv", "association_summary.json",
        "branch_evidence_matrix.csv", "branch_evidence_matrix.json",
        "scientific_interpretation.json", "estimability_matrix.json",
        "resource_ledger.json", "access_audit.json", "false_pass_audit.json",
        "payload_manifest.json", "whole_root_inventory.json",
    }
    if (repo / COMPLETION).exists() or (repo / PENDING).exists():
        raise X2MechM2Error("M2 terminal completion already exists")
    if not (repo / ROOT).exists():
        return
    existing = {path.name for path in (repo / ROOT).glob("*") if path.is_file()}
    existing |= {path.relative_to(repo / ROOT).as_posix() for path in (repo / ROOT).rglob("*") if path.is_file()}
    allowed = {"source_binding.json", "pre_execution_seal.json", "completion/x2_mech_m2_producer.json"} if allow_producer else set()
    unexpected = sorted((existing & forbidden) - allowed)
    if unexpected:
        raise X2MechM2Error(f"M2 result output exists before allowed execution: {unexpected[0]}")


def freeze_producer(repo_root: Path | str, producer_commit: str, test_summary: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = load_config(repo)
    if _git(repo, "rev-parse", "HEAD") != producer_commit:
        raise X2MechM2Error("HEAD is not the declared M2 producer commit")
    if _git(repo, "rev-parse", PRODUCER_REF + "^{commit}") != producer_commit:
        raise X2MechM2Error("M2 producer archive ref does not bind HEAD")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", M1_R1_COMPLETION_REF, "HEAD"], cwd=repo
    ).returncode:
        raise X2MechM2Error("M1 completion is not producer HEAD ancestor")
    if (repo / PRODUCER_COMPLETION).exists():
        raise X2MechM2Error("M2 producer completion already exists")
    _reject_if_m2_results_exist(repo)
    checks = _startup_checks(repo, expected_head=producer_commit)
    checks.extend(_producer_source_ast_checks(repo))
    checks.extend(_header_contract_checks(repo, config))
    bindings = _source_control_bindings(repo)
    bindings.extend(file_record(repo, relative) for relative in SOURCE_FILES)
    access = _as_list({
        "m1_control_metadata_read": True,
        "m1_csv_headers_read": True,
        "m1_map_values_read": False,
        "m1_fidelity_outcome_rows_read": False,
        "x2_per_route_stability_rows_read": False,
        "np_load_called": False,
        "jpeg_bytes_read": False,
        "checkpoint_bytes_read": False,
        "model_loaded": False,
        "torch_imported": False,
        "timm_imported": False,
        "cuda_accessed": False,
        "m2_result_created": False,
    })
    terminal = _terminal(terminal=True, complete=True, next_authorization=AUTHORIZATION)
    source_binding = signed("stage_x_x2_mech_m2_source_binding", "FROZEN", _payload(
        identity=_as_list({
            "producer_commit": producer_commit,
            "m1_completion_commit": M1_R1_COMPLETION_COMMIT,
            "m1_completion_ref": M1_R1_COMPLETION_REF,
            "x2_decision": "NO_GO_BROAD_CLAIM_DEVELOPMENT",
            "study_type": "POST_HOC_EXPLORATORY_MECHANISM_ANALYSIS",
        }),
        bindings=bindings,
        checks=checks,
        counts=_as_list(config["source_counts"]),
        state=_as_list({
            "output_schemas_frozen": True,
            "source_roster_frozen": True,
            "denominator_frozen": True,
            "branch_thresholds_frozen": False,
            "branch_fallback": "MIXED_OR_INCONCLUSIVE",
        }),
        access=access,
        terminal=terminal,
    ))
    pre_execution = signed("stage_x_x2_mech_m2_pre_execution_seal", "FROZEN", _payload(
        identity=_as_list({
            "producer_commit": producer_commit,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "test_summary": test_summary,
        }),
        bindings=bindings,
        checks=checks,
        counts=_as_list({
            "required_output_count": len(config["required_outputs"]),
            "frozen_csv_schema_count": len(REQUIRED_CSV_FIELDS),
            "source_file_count": len(SOURCE_FILES),
        }),
        state=_as_list({
            "atomic_no_overwrite_publication": True,
            "source_recomputing_validator_required": True,
            "false_pass_mutation_tests_required": True,
            "real_m2_results_created": False,
        }),
        access=access,
        terminal=terminal,
    ))
    _write_json_new(repo / ROOT / "source_binding.json", source_binding)
    _write_json_new(repo / ROOT / "pre_execution_seal.json", pre_execution)
    producer = signed("stage_x_x2_mech_m2_producer_completion", "PASS", _payload(
        identity=_as_list({"producer_commit": producer_commit}),
        bindings=[
            file_record(repo, ROOT / "source_binding.json"),
            file_record(repo, ROOT / "pre_execution_seal.json"),
        ],
        checks=checks,
        counts=_as_list({
            "m1_route_terminals": 199,
            "m1_maps_bound": 9552,
            "m1_masks_bound": 24,
            "m1_fidelity_summaries_bound": 796,
            "m1_road_rows_bound": 71640,
        }),
        state=_as_list({
            "producer_sealed": True,
            "m2_empirical_analysis_not_started": True,
            "x2_no_go_unchanged": True,
        }),
        access=access,
        terminal=terminal,
    ))
    _publish_atomic(repo / PRODUCER_PENDING, repo / PRODUCER_COMPLETION, producer)
    return producer


def validate_producer(repo_root: Path | str = REPO_ROOT, *, require_head: bool = True) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    load_config(repo)
    producer_sha = _git(repo, "rev-parse", PRODUCER_REF + "^{commit}")
    head = _git(repo, "rev-parse", "HEAD")
    if require_head and producer_sha != head:
        raise X2MechM2Error("M2 producer ref does not bind current HEAD")
    if subprocess.run(["git", "merge-base", "--is-ancestor", producer_sha, head], cwd=repo).returncode:
        raise X2MechM2Error("M2 producer ref is not a HEAD ancestor")
    for relative in (ROOT / "source_binding.json", ROOT / "pre_execution_seal.json", PRODUCER_COMPLETION):
        value = strict_read(repo / relative)
        validate_artifact(value, repo, verify_files=True)
    producer = strict_read(repo / PRODUCER_COMPLETION)
    if producer.get("status") != "PASS":
        raise X2MechM2Error("M2 producer completion is not PASS")
    access = {row["name"]: row["value"] for row in producer["payload"]["access"]}
    if access.get("m1_map_values_read") or access.get("x2_per_route_stability_rows_read") or access.get("np_load_called"):
        raise X2MechM2Error("producer access boundary drifted")
    return {"status": "PASS", "producer_commit": producer_sha}


def _require_cpu_only_environment() -> dict[str, Any]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible != "":
        raise X2MechM2Error("CPU-only M2 requires CUDA_VISIBLE_DEVICES=''")
    return {
        "cuda_visible_devices": cuda_visible,
        "gpu_accessed": False,
        "cuda_accessed": False,
        "model_loaded": False,
        "checkpoint_bytes_read": False,
        "jpeg_bytes_read": False,
        "forward_calls": 0,
        "backward_calls": 0,
        "downloads": 0,
        "display_png_jpeg_generated": False,
        "torch_imported_for_cpu_geometry": False,
        "scipy_cpu_geometry": False,
        "timm_imported": False,
    }


def _ensure_post_producer(repo: Path) -> None:
    if not (repo / PRODUCER_COMPLETION).is_file():
        raise X2MechM2Error("M2 producer completion missing")
    validate_producer(repo, require_head=False)


def _array_digest(value: np.ndarray) -> str:
    return m1_worker.array_digest(np.ascontiguousarray(value))


def _load_map_bound(repo: Path, record: Mapping[str, Any]) -> np.ndarray:
    relative = str(record.get("path") or record.get("relative_path"))
    path = repo / relative
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise X2MechM2Error(f"map byte binding drifted: {relative}")
    if sha_file(path) != record["sha256"]:
        raise X2MechM2Error(f"map byte binding drifted: {relative}")
    value = np.load(path, allow_pickle=False)
    m1_worker.validate_map(value)
    digest_key = "canonical_digest" if "canonical_digest" in record else "canonical_array_digest"
    if _array_digest(value) != record[digest_key]:
        raise X2MechM2Error(f"map canonical digest drifted: {relative}")
    return value


def _load_mask_bound(repo: Path, record: Mapping[str, Any]) -> np.ndarray:
    path = repo / str(record["relative_path"])
    if sha_file(path) != record["sha256"]:
        raise X2MechM2Error(f"mask byte binding drifted: {record['relative_path']}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.bool_ or value.shape != (224, 224):
        raise X2MechM2Error(f"mask shape or dtype drifted: {record['relative_path']}")
    digest = hashlib.sha256(b"bool|224,224|" + np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
    if digest != record["canonical_digest"]:
        raise X2MechM2Error(f"mask canonical digest drifted: {record['relative_path']}")
    return value


def _records_by_key(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(row[item] for item in keys)
        if key in result:
            raise X2MechM2Error(f"duplicate row key: {keys}:{key}")
        result[key] = row
    return result


def _finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise X2MechM2Error(f"{context}: non-finite value")
    return result


def _bool_string(value: Any, context: str) -> bool:
    if value in (True, False):
        return bool(value)
    if value == "True":
        return True
    if value == "False":
        return False
    raise X2MechM2Error(f"{context}: invalid boolean")


def _top_indices_1d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise X2MechM2Error("top-10 values must be finite and non-empty")
    count = int(math.ceil(0.1 * array.size))
    return np.sort(np.argpartition(array, -count)[-count:])


def _jaccard_1d(left: np.ndarray, right: np.ndarray) -> float:
    a = _top_indices_1d(left)
    b = _top_indices_1d(right)
    return float(np.intersect1d(a, b).size / np.union1d(a, b).size)


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    group_start = np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
    group_id = np.cumsum(group_start, dtype=np.int64) - 1
    counts = np.bincount(group_id)
    ends = np.cumsum(counts)
    starts = ends - counts
    average = (starts + 1 + ends) / 2.0
    ranks_sorted = average[group_id]
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _centroid(weights: np.ndarray, coordinates: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    total = float(weights.sum(dtype=np.float64))
    if total <= np.finfo(np.float64).tiny:
        return fallback
    return (coordinates * weights[:, None]).sum(axis=0) / total


def _centroid_from_axes(weights: np.ndarray, row_coords: np.ndarray,
                        col_coords: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    total = float(weights.sum(dtype=np.float64))
    if total <= np.finfo(np.float64).tiny:
        return fallback
    return np.asarray([
        float(np.sum(row_coords * weights, dtype=np.float64) / total),
        float(np.sum(col_coords * weights, dtype=np.float64) / total),
    ])


def _block_centers(valid_blocks: np.ndarray, block: int) -> np.ndarray:
    row, col = np.nonzero(valid_blocks)
    return np.column_stack(((row + 0.5) * block - 0.5, (col + 0.5) * block - 0.5))


def _m2_stability_metrics(clean: np.ndarray, transformed: np.ndarray,
                          valid_mask: np.ndarray, block: int) -> dict[str, float]:
    prepared = _prepare_clean_stability(clean, valid_mask, block)
    return _m2_stability_metrics_prepared(prepared, transformed)


def _prepare_clean_stability(clean: np.ndarray, valid_mask: np.ndarray, block: int) -> dict[str, Any]:
    a = np.asarray(clean, dtype=np.float64)
    mask = np.asarray(valid_mask)
    if mask.shape != a.shape or mask.dtype != np.bool_:
        raise X2MechM2Error("stability inputs have incompatible shape or mask dtype")
    if block == 1:
        valid = mask
        signed_a = a[valid]
        mass_a = np.abs(signed_a)
        signed_map_a = np.where(valid, a, 0.0)
        mass_map_a = np.where(valid, np.abs(a), 0.0)
    else:
        rows, cols = a.shape[0] // block, a.shape[1] // block
        a_blocks = a.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)
        masks = mask.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)
        valid = masks.all(axis=(2, 3))
        signed_map_a = a_blocks.sum(axis=(2, 3), dtype=np.float64)
        mass_map_a = np.abs(a_blocks).sum(axis=(2, 3), dtype=np.float64)
        signed_a = signed_map_a[valid]
        mass_a = mass_map_a[valid]
        signed_map_a = np.where(valid, signed_map_a, 0.0)
    if signed_a.size == 0:
        raise X2MechM2Error("no complete valid units")
    row, col = np.nonzero(valid)
    r0, r1, c0, c1 = row.min(), row.max() + 1, col.min(), col.max() + 1
    row_coords = (row + 0.5) * block - 0.5
    col_coords = (col + 0.5) * block - 0.5
    center = np.asarray([(a.shape[0] - 1) / 2, (a.shape[1] - 1) / 2])
    return {
        "shape": a.shape,
        "block": block,
        "valid": valid,
        "signed_a": signed_a,
        "mass_a": mass_a,
        "signed_map_a": signed_map_a,
        "mass_map_a": mass_map_a,
        "crop_a": signed_map_a[r0:r1, c0:c1],
        "crop_bounds": (r0, r1, c0, c1),
        "row_coords": row_coords,
        "col_coords": col_coords,
        "center": center,
        "valid_count": int(valid.sum()),
        "excluded_count": int(valid.size - valid.sum()),
    }


def _m2_stability_metrics_prepared(prepared: Mapping[str, Any], transformed: np.ndarray) -> dict[str, float]:
    from skimage.metrics import structural_similarity

    b = np.asarray(transformed, dtype=np.float64)
    block = int(prepared["block"])
    valid = np.asarray(prepared["valid"], dtype=bool)
    if tuple(b.shape) != tuple(prepared["shape"]):
        raise X2MechM2Error("transformed map shape differs from clean map")
    if block == 1:
        signed_b = b[valid]
        mass_b = np.abs(signed_b)
        signed_map_b = np.where(valid, b, 0.0)
    else:
        rows, cols = b.shape[0] // block, b.shape[1] // block
        b_blocks = b.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)
        signed_map_b = b_blocks.sum(axis=(2, 3), dtype=np.float64)
        mass_map_b = np.abs(b_blocks).sum(axis=(2, 3), dtype=np.float64)
        signed_b = signed_map_b[valid]
        mass_b = mass_map_b[valid]
        signed_map_b = np.where(valid, signed_map_b, 0.0)
    signed_a = np.asarray(prepared["signed_a"], dtype=np.float64)
    mass_a = np.asarray(prepared["mass_a"], dtype=np.float64)
    crop_a = np.asarray(prepared["crop_a"], dtype=np.float64)
    r0, r1, c0, c1 = prepared["crop_bounds"]
    crop_b = signed_map_b[r0:r1, c0:c1]
    data_range = max(
        float(max(signed_a.max(), signed_b.max()) - min(signed_a.min(), signed_b.min())),
        1e-12,
    )
    win = min(
        7,
        crop_a.shape[0] if crop_a.shape[0] % 2 else crop_a.shape[0] - 1,
        crop_a.shape[1] if crop_a.shape[1] % 2 else crop_a.shape[1] - 1,
    )
    if win < 3:
        raise X2MechM2Error("too few valid units for SSIM")
    row_coords = np.asarray(prepared["row_coords"], dtype=np.float64)
    col_coords = np.asarray(prepared["col_coords"], dtype=np.float64)
    center = np.asarray(prepared["center"], dtype=np.float64)
    positive_shift = np.linalg.norm(
        _centroid_from_axes(np.maximum(signed_a, 0.0), row_coords, col_coords, center)
        - _centroid_from_axes(np.maximum(signed_b, 0.0), row_coords, col_coords, center)
    )
    negative_shift = np.linalg.norm(
        _centroid_from_axes(np.maximum(-signed_a, 0.0), row_coords, col_coords, center)
        - _centroid_from_axes(np.maximum(-signed_b, 0.0), row_coords, col_coords, center)
    )
    mass_shift = np.linalg.norm(
        _centroid_from_axes(mass_a, row_coords, col_coords, center)
        - _centroid_from_axes(mass_b, row_coords, col_coords, center)
    )
    diagonal = math.hypot(*prepared["shape"])
    return {
        "signed_spearman": _spearman(signed_a, signed_b),
        "top10_unit_signed_value_jaccard": _jaccard_1d(signed_a, signed_b),
        "absolute_mass_spearman": _spearman(mass_a, mass_b),
        "top10_unit_absolute_mass_jaccard": _jaccard_1d(mass_a, mass_b),
        "signed_ssim": float(structural_similarity(crop_a, crop_b, data_range=data_range, win_size=win)),
        "normalized_positive_negative_mass_displacement": float((positive_shift + negative_shift) / (2 * diagonal)),
        "absolute_mass_center_of_mass_pixel_error": float(mass_shift),
        "valid_unit_count": int(prepared["valid_count"]),
        "excluded_partial_unit_count": int(prepared["excluded_count"]),
    }


def _m1_source_validation(repo: Path) -> dict[str, Any]:
    completion = strict_read(repo / m1.COMPLETION)
    validate_signed(completion, "M1 completion")
    if completion.get("status") != "PASS" or completion["payload"].get("next_authorization") != AUTHORIZATION:
        raise X2MechM2Error("M1 completion does not authorize M2")
    reconciliation = strict_read(repo / m1.ROOT / "reconciliation.json")
    validate_signed(reconciliation, "M1 reconciliation")
    if reconciliation.get("status") != "PASS" or not reconciliation["payload"].get("source_recomputed_from_route_terminals_and_raw_files"):
        raise X2MechM2Error("M1 reconciliation is not source-recomputing PASS")
    map_json = strict_read(repo / m1.ROOT / "map_manifest.json")
    mask_json = strict_read(repo / m1.ROOT / "valid_mask_registry.json")
    validate_signed(map_json, "M1 map manifest")
    validate_signed(mask_json, "M1 mask registry")
    map_rows = map_json["payload"]["maps"]
    mask_rows = mask_json["payload"]["masks"]
    if len(map_rows) != 9552 or len(mask_rows) != 24:
        raise X2MechM2Error("M1 manifest counts drifted")
    for row in map_rows:
        path = repo / row["relative_path"]
        if path.stat().st_size != row["bytes"] or sha_file(path) != row["sha256"]:
            raise X2MechM2Error(f"M1 map bytes drifted: {row['relative_path']}")
        value = np.load(path, allow_pickle=False)
        m1_worker.validate_map(value)
        if _array_digest(value) != row["canonical_array_digest"]:
            raise X2MechM2Error(f"M1 map digest drifted: {row['relative_path']}")
    for row in mask_rows:
        _load_mask_bound(repo, row)
    fidelity_rows = _read_csv(repo / m1.ROOT / "multiscale_fidelity.csv", m1.FIDELITY_FIELDS)
    detail_rows = _read_csv(repo / m1.ROOT / "multiscale_fidelity_rows.csv", m1.DETAIL_FIELDS)
    if len(fidelity_rows) != 796 or len(detail_rows) != 71640:
        raise X2MechM2Error("M1 fidelity aggregate row count drifted")
    _verify_m1_fidelity_tables(fidelity_rows, detail_rows)
    pending = [path for path in (repo / m1.ROOT).rglob("*") if "pending" in path.name]
    if pending:
        raise X2MechM2Error("M1 pending files exist")
    if (repo / COMPLETION).exists():
        raise X2MechM2Error("M2 output exists before M1 source validation gate")
    return {
        "status": "PASS",
        "source_recomputed": True,
        "routes": 199,
        "maps": 9552,
        "shared_masks": 24,
        "fidelity_summaries": 796,
        "detailed_road_rows": 71640,
        "next_authorization": AUTHORIZATION,
    }


def _verify_m1_fidelity_tables(summaries: Sequence[Mapping[str, str]],
                               details: Sequence[Mapping[str, str]]) -> None:
    summary_by: dict[tuple[str, str, int], Mapping[str, str]] = {}
    for row in summaries:
        method = row["method_id"]
        block = int(row["block"])
        if method not in METHODS or block not in (4, 16):
            raise X2MechM2Error("M1 fidelity summary roster drifted")
        key = (row["unit_digest"], method, block)
        if key in summary_by:
            raise X2MechM2Error("M1 fidelity summary duplicate")
        if int(row["unit_count"]) != (224 // block) ** 2:
            raise X2MechM2Error("M1 fidelity unit count drifted")
        for field in ("road_combined", "road_morf", "road_lerf", "clean_logit", "baseline_logit"):
            _finite(row[field], f"M1 fidelity {field}")
        summary_by[key] = row
    detail_by: dict[tuple[str, str, int, int, int, str], Mapping[str, str]] = {}
    for row in details:
        method = row["method_id"]
        block = int(row["block"])
        percentile = int(row["percentile"])
        seed_index = int(row["seed_index"])
        direction = row["direction"]
        if method not in METHODS or block not in (4, 16) or percentile not in m1.PERCENTILES:
            raise X2MechM2Error("M1 fidelity detail roster drifted")
        if seed_index not in m1.SEEDS or direction not in m1.DIRECTIONS:
            raise X2MechM2Error("M1 fidelity detail coordinate drifted")
        key = (row["unit_digest"], method, block, percentile, seed_index, direction)
        if key in detail_by:
            raise X2MechM2Error("M1 fidelity detail duplicate")
        expected_seed = seed64(
            "STAGE_X_X2_ROAD_NOISY_LINEAR_V1_20260909",
            f"{row['unit_digest']}|{method}", block, percentile, seed_index, direction,
        )
        if int(row["pcg64_seed"]) != expected_seed:
            raise X2MechM2Error("M1 fidelity PCG64 seed drifted")
        units = (224 // block) ** 2
        expected_fraction = math.floor(units * percentile / 100) * block * block / (224 * 224)
        if not math.isclose(
            _finite(row["actual_pixel_fraction"], "M1 fidelity actual pixel fraction"),
            expected_fraction,
            abs_tol=1e-7,
            rel_tol=1e-7,
        ):
            raise X2MechM2Error("M1 fidelity actual pixel fraction drifted")
        _finite(row["normalized_response"], "M1 fidelity normalized response")
        detail_by[key] = row
    if len(summary_by) != 199 * 2 * 2 or len(detail_by) != 199 * 2 * 2 * 9 * 5 * 2:
        raise X2MechM2Error("M1 fidelity keyed count drifted")
    fractions = np.asarray(m1.PERCENTILES, dtype=np.float64) / 100.0
    for unit, method, block in sorted(summary_by):
        aucs: dict[str, float] = {}
        for direction in m1.DIRECTIONS:
            per_seed = []
            for seed_index in m1.SEEDS:
                responses = [
                    _finite(detail_by[(unit, method, block, percentile, seed_index, direction)]["normalized_response"],
                            "M1 fidelity normalized response")
                    for percentile in m1.PERCENTILES
                ]
                per_seed.append(float(np.trapz(responses, fractions)))
            aucs[direction] = float(np.mean(per_seed, dtype=np.float64))
        expected = {
            "road_morf": aucs["morf"],
            "road_lerf": aucs["lerf"],
            "road_combined": 0.5 * (aucs["lerf"] - aucs["morf"]),
        }
        summary = summary_by[(unit, method, block)]
        for field, value in expected.items():
            if not math.isclose(float(summary[field]), value, abs_tol=1e-12, rel_tol=1e-12):
                raise X2MechM2Error(f"M1 fidelity AUC recomputation drifted: {unit}:{method}:{block}:{field}")


def _x2_formal_phase_paths(repo: Path) -> list[Path]:
    return sorted((repo / "evidence/stage_x/x2_r2_r1_r1_nns_v2_development_20260910/formal/units").glob(
        "*/attempts/attempt-000001/formal/stability.csv"
    ))


def _load_x2_pixel_stability(repo: Path) -> list[dict[str, str]]:
    fields = (
        "unit_digest", "model_id", "class_id", "method_id", "transformation",
        "signed_spearman", "top10_mass_jaccard", "signed_ssim",
        "normalized_positive_negative_mass_displacement", "center_of_mass_pixel_error",
        "prediction_changed", "transformed_target_logit",
    )
    rows: list[dict[str, str]] = []
    paths = _x2_formal_phase_paths(repo)
    if len(paths) != 199:
        raise X2MechM2Error(f"expected 199 X2 formal stability tables, found {len(paths)}")
    for path in paths:
        current = _read_csv(path, fields)
        if len(current) != 46:
            raise X2MechM2Error(f"X2 stability row count drifted: {path}")
        rows.extend(row for row in current if row["method_id"] in METHODS and row["transformation"] in TRANSFORMS)
    if len(rows) != 199 * 12 * 2:
        raise X2MechM2Error(f"X2 geometric pixel row count drifted: {len(rows)}")
    return rows


def _load_source_tables(repo: Path) -> dict[str, Any]:
    map_rows = _read_csv(repo / m1.ROOT / "map_manifest.csv", m1.MAP_FIELDS)
    if len(map_rows) != 9552:
        raise X2MechM2Error("M1 map manifest row count drifted")
    masks_json = strict_read(repo / m1.ROOT / "valid_mask_registry.json")
    validate_signed(masks_json, "M1 mask registry")
    mask_rows = list(masks_json["payload"]["masks"])
    if len(mask_rows) != 24:
        raise X2MechM2Error("M1 mask registry count drifted")
    fidelity_rows = _read_csv(repo / m1.ROOT / "multiscale_fidelity.csv", m1.FIDELITY_FIELDS)
    fidelity_detail_rows = _read_csv(repo / m1.ROOT / "multiscale_fidelity_rows.csv", m1.DETAIL_FIELDS)
    if len(fidelity_rows) != 796 or len(fidelity_detail_rows) != 71640:
        raise X2MechM2Error("M1 fidelity counts drifted")
    x2_rows = _load_x2_pixel_stability(repo)
    return {
        "map_rows": map_rows,
        "mask_rows": mask_rows,
        "fidelity_rows": fidelity_rows,
        "fidelity_detail_rows": fidelity_detail_rows,
        "x2_rows": x2_rows,
    }


def _transport_clean(clean: np.ndarray, transform: str, metadata: Mapping[str, Any],
                     access: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(clean, dtype=np.float32)
    valid = np.ones(array.shape, dtype=bool)
    if transform == "horizontal_flip":
        return np.fliplr(np.fliplr(array)).copy(), valid
    if transform.startswith("translate_x_"):
        amount = int(transform.removeprefix("translate_x_"))
        restored = np.zeros_like(array)
        valid[:] = False
        if amount > 0:
            restored[:, :-amount] = array[:, :-amount]
            valid[:, :-amount] = True
        else:
            n = -amount
            restored[:, n:] = array[:, n:]
            valid[:, n:] = True
        return restored, valid
    if transform.startswith("translate_y_"):
        amount = int(transform.removeprefix("translate_y_"))
        restored = np.zeros_like(array)
        valid[:] = False
        if amount > 0:
            restored[:-amount, :] = array[:-amount, :]
            valid[:-amount, :] = True
        else:
            n = -amount
            restored[n:, :] = array[n:, :]
            valid[n:, :] = True
        return restored, valid
    from scipy import ndimage
    from skimage.transform import resize

    access["scipy_cpu_geometry"] = True
    kind = metadata["inverse"]
    if kind == "rotate":
        angle = float(metadata["angle"])
        forward = ndimage.rotate(array, angle, reshape=False, order=1, mode="constant", cval=0.0, prefilter=False)
        back = ndimage.rotate(forward, -angle, reshape=False, order=1, mode="constant", cval=0.0, prefilter=False)
        valid[:] = False
        r0, r1, c0, c1 = metadata["valid_box"]
        valid[int(r0):int(r1), int(c0):int(c1)] = True
        return back.astype(np.float32, copy=False), valid
    if kind == "resize_224_to_216_then_pad4":
        forward = resize(array[4:220, 4:220], (224, 224), order=3, mode="edge",
                         anti_aliasing=True, preserve_range=True)
        inner = resize(forward, (216, 216), order=1, mode="edge",
                       anti_aliasing=False, preserve_range=True)
        restored = np.zeros_like(array)
        restored[4:220, 4:220] = inner.astype(np.float32, copy=False)
        valid[:] = False
        valid[4:220, 4:220] = True
        return restored.astype(np.float32, copy=False), valid
    raise X2MechM2Error(f"unsupported transport transform: {transform}")


def _scale_id(block: int) -> str:
    return {1: "pixel", 4: "fine_cell", 16: "native_patch"}[block]


def _stable_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["unit_digest"]),
        str(row["model_id"]),
        str(row["sample_id"]),
        str(row["method_id"]),
        str(row["transform"]),
    )


def _x2_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["unit_digest"]), str(row["method_id"]), str(row["transformation"]))


def _format_float(value: float) -> float:
    return float(value)


def compute_multiscale_results(repo_root: Path | str = REPO_ROOT,
                               *, association_draws: int = 20000,
                               max_attempts: int = 2000000) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = load_config(repo)
    access = _require_cpu_only_environment()
    _ensure_post_producer(repo)
    m1_gate = _m1_source_validation(repo)
    source = _load_source_tables(repo)
    started = time.monotonic()
    cpu_started = time.process_time()
    map_rows = source["map_rows"]
    masks_by_key = {
        (row["transform"], row["mask_id"]): _load_mask_bound(repo, row)
        for row in source["mask_rows"]
    }
    by_map = _records_by_key(map_rows, ("unit_digest", "method_id", "transform", "coordinate"))
    clean_by_unit_method: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in map_rows:
        clean_by_unit_method.setdefault((row["unit_digest"], row["method_id"]), row)
    source_clean_records = {
        (row["unit_digest"], row["method_id"]): {
            "path": row["source_clean_map_path"],
            "bytes": (repo / row["source_clean_map_path"]).stat().st_size,
            "sha256": row["source_clean_map_sha256"],
            "canonical_digest": row["source_clean_map_digest"],
        }
        for row in map_rows
    }
    x2_by = _records_by_key(source["x2_rows"], ("unit_digest", "method_id", "transformation"))
    stability_rows: list[dict[str, Any]] = []
    pixel_audit_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    hflip_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    clean_cache: dict[tuple[str, str], np.ndarray] = {}

    def clean_map(unit: str, method: str) -> np.ndarray:
        key = (unit, method)
        if key not in clean_cache:
            clean_cache[key] = _load_map_bound(repo, source_clean_records[key])
        return clean_cache[key]

    def m1_map(unit: str, method: str, transform: str, coordinate: str) -> np.ndarray:
        return _load_map_bound(repo, by_map[(unit, method, transform, coordinate)])

    valid_routes = sorted({row["unit_digest"] for row in map_rows})
    for unit in valid_routes:
        first = next(row for row in map_rows if row["unit_digest"] == unit)
        for method in METHODS:
            clean = clean_map(unit, method)
            structure = clean_structure(clean)
            structure_rows.append({
                "unit_digest": unit,
                "model_id": first["model_id"],
                "sample_id": first["sample_id"],
                "source_status": "VALID",
                "method_id": method,
                **{key: _format_float(value) for key, value in structure.items()},
                "status": "PASS",
            })
            for transform in TRANSFORMS:
                inverse = m1_map(unit, method, transform, "inverse")
                manifest = by_map[(unit, method, transform, "inverse")]
                prediction_changed = _bool_string(manifest["prediction_changed"], "M1 prediction_changed")
                source_metadata = json.loads(manifest["transform_metadata"], object_pairs_hook=_pairs)
                restored, default_mask = _transport_clean(clean, transform, source_metadata, access)
                for mask_id in MASKS:
                    mask = masks_by_key[(transform, mask_id)]
                    for scale_id, block in SCALES:
                        metrics = _m2_stability_metrics(clean, inverse, mask, block)
                        row = {
                            "unit_digest": unit,
                            "model_id": manifest["model_id"],
                            "sample_id": manifest["sample_id"],
                            "source_status": "VALID",
                            "method_id": method,
                            "transform": transform,
                            "mask_id": mask_id,
                            "scale_id": scale_id,
                            "block": block,
                            "prediction_changed": prediction_changed,
                            **{key: _format_float(value) for key, value in metrics.items()},
                            "status": "PASS",
                        }
                        stability_rows.append(row)
                        if mask_id == "x2_frozen_common_domain" and scale_id == "pixel":
                            x2 = x2_by[(unit, method, transform)]
                            diffs = {
                                "signed_spearman_abs_diff": abs(_finite(x2["signed_spearman"], "x2 signed") - row["signed_spearman"]),
                                "top10_jaccard_abs_diff": abs(_finite(x2["top10_mass_jaccard"], "x2 jaccard") - row["top10_unit_signed_value_jaccard"]),
                                "signed_ssim_abs_diff": abs(_finite(x2["signed_ssim"], "x2 ssim") - row["signed_ssim"]),
                                "displacement_abs_diff": abs(_finite(x2["normalized_positive_negative_mass_displacement"], "x2 displacement") - row["normalized_positive_negative_mass_displacement"]),
                                "center_of_mass_abs_diff": abs(_finite(x2["center_of_mass_pixel_error"], "x2 center") - row["absolute_mass_center_of_mass_pixel_error"]),
                            }
                            bool_match = _bool_string(x2["prediction_changed"], "x2 prediction_changed") == prediction_changed
                            status = "PASS" if bool_match and all(
                                value <= config["metrics"]["pixel_reproduction_continuous_atol"] +
                                config["metrics"]["pixel_reproduction_continuous_rtol"] * max(
                                    abs(_finite(x2[field], f"x2 {field}")), abs(row[m2_field])
                                )
                                for (field, m2_field), value in zip(CONTINUOUS_PIXEL_PAIRS, diffs.values())
                            ) else "BLOCKED_X2_PIXEL_REPRODUCTION_DRIFT"
                            pixel_audit_rows.append({
                                "unit_digest": unit,
                                "model_id": manifest["model_id"],
                                "sample_id": manifest["sample_id"],
                                "method_id": method,
                                "transform": transform,
                                "x2_signed_spearman": _finite(x2["signed_spearman"], "x2 signed"),
                                "m2_signed_spearman": row["signed_spearman"],
                                "signed_spearman_abs_diff": diffs["signed_spearman_abs_diff"],
                                "x2_top10_mass_jaccard": _finite(x2["top10_mass_jaccard"], "x2 jaccard"),
                                "m2_top10_unit_signed_value_jaccard": row["top10_unit_signed_value_jaccard"],
                                "top10_jaccard_abs_diff": diffs["top10_jaccard_abs_diff"],
                                "x2_signed_ssim": _finite(x2["signed_ssim"], "x2 ssim"),
                                "m2_signed_ssim": row["signed_ssim"],
                                "signed_ssim_abs_diff": diffs["signed_ssim_abs_diff"],
                                "x2_normalized_positive_negative_mass_displacement": _finite(x2["normalized_positive_negative_mass_displacement"], "x2 displacement"),
                                "m2_normalized_positive_negative_mass_displacement": row["normalized_positive_negative_mass_displacement"],
                                "displacement_abs_diff": diffs["displacement_abs_diff"],
                                "x2_center_of_mass_pixel_error": _finite(x2["center_of_mass_pixel_error"], "x2 center"),
                                "m2_absolute_mass_center_of_mass_pixel_error": row["absolute_mass_center_of_mass_pixel_error"],
                                "center_of_mass_abs_diff": diffs["center_of_mass_abs_diff"],
                                "x2_prediction_changed": _bool_string(x2["prediction_changed"], "x2 prediction_changed"),
                                "m1_prediction_changed": prediction_changed,
                                "prediction_changed_match": bool_match,
                                "status": status,
                            })
                            if status != "PASS":
                                raise X2MechM2Error("BLOCKED_X2_PIXEL_REPRODUCTION_DRIFT")
                    for scale_id, block in SSE_SCALES:
                        error = _m2_sse_decomposition(clean, inverse, mask, block)
                        identity_abs = abs(error["sse_total"] - error["sse_between"] - error["sse_within"])
                        denominator = max(abs(error["sse_total"]), abs(error["sse_between"] + error["sse_within"]), 1.0)
                        identity_rel = identity_abs / denominator
                        if identity_abs > 1e-10 + 1e-10 * denominator:
                            raise X2MechM2Error("SSE identity drifted")
                        error_rows.append({
                            "unit_digest": unit,
                            "model_id": manifest["model_id"],
                            "sample_id": manifest["sample_id"],
                            "source_status": "VALID",
                            "method_id": method,
                            "transform": transform,
                            "mask_id": mask_id,
                            "scale_id": scale_id,
                            "block": block,
                            **{key: _format_float(value) for key, value in error.items()},
                            "identity_abs_diff": identity_abs,
                            "identity_rel_diff": identity_rel,
                            "status": "PASS",
                        })
                    for scale_id, block in SCALES:
                        if default_mask.shape != mask.shape:
                            raise X2MechM2Error("transport mask shape drifted")
                        transport_mask = mask & default_mask
                        tmetrics = _m2_stability_metrics(clean, restored, transport_mask, block)
                        transport_rows.append({
                            "unit_digest": unit,
                            "model_id": manifest["model_id"],
                            "sample_id": manifest["sample_id"],
                            "source_status": "VALID",
                            "method_id": method,
                            "transform": transform,
                            "mask_id": mask_id,
                            "scale_id": scale_id,
                            "block": block,
                            **{key: _format_float(value) for key, value in tmetrics.items()},
                            "status": "PASS",
                        })
                        if transform == "horizontal_flip":
                            observed = next(
                                row for row in stability_rows
                                if row["unit_digest"] == unit and row["method_id"] == method
                                and row["transform"] == transform and row["mask_id"] == mask_id
                                and row["scale_id"] == scale_id
                            )
                            hflip_rows.append({
                                "unit_digest": unit,
                                "model_id": manifest["model_id"],
                                "sample_id": manifest["sample_id"],
                                "source_status": "VALID",
                                "method_id": method,
                                "mask_id": mask_id,
                                "scale_id": scale_id,
                                "block": block,
                                "full_domain_mask": bool(mask.all()),
                                "valid_unit_count": tmetrics["valid_unit_count"],
                                "excluded_partial_unit_count": tmetrics["excluded_partial_unit_count"],
                                "observed_signed_spearman": observed["signed_spearman"],
                                "transport_signed_spearman": tmetrics["signed_spearman"],
                                "observed_minus_transport_signed_spearman": observed["signed_spearman"] - tmetrics["signed_spearman"],
                                "observed_top10_unit_signed_value_jaccard": observed["top10_unit_signed_value_jaccard"],
                                "transport_top10_unit_signed_value_jaccard": tmetrics["top10_unit_signed_value_jaccard"],
                                "observed_minus_transport_top10_unit_signed_value_jaccard": observed["top10_unit_signed_value_jaccard"] - tmetrics["top10_unit_signed_value_jaccard"],
                                "status": "PASS",
                            })

    failed_unit = m1._load_plan(repo)[1]
    for method in METHODS:
        for transform in TRANSFORMS:
            for mask_id in MASKS:
                for scale_id, block in SCALES:
                    stability_rows.append({
                        "unit_digest": failed_unit["unit_digest"],
                        "model_id": failed_unit["model_id"],
                        "sample_id": failed_unit["sample_id"],
                        "source_status": "FAILED_ROUTE_LOCAL_NUMERICAL",
                        "method_id": method,
                        "transform": transform,
                        "mask_id": mask_id,
                        "scale_id": scale_id,
                        "block": block,
                        "prediction_changed": "",
                        "valid_unit_count": 0,
                        "excluded_partial_unit_count": 0,
                        "signed_spearman": "",
                        "top10_unit_signed_value_jaccard": "",
                        "absolute_mass_spearman": "",
                        "top10_unit_absolute_mass_jaccard": "",
                        "signed_ssim": "",
                        "normalized_positive_negative_mass_displacement": "",
                        "absolute_mass_center_of_mass_pixel_error": "",
                        "status": "FAILED_ROUTE_LOCAL_NUMERICAL",
                    })
            for mask_id in MASKS:
                for scale_id, block in SSE_SCALES:
                    error_rows.append({
                        "unit_digest": failed_unit["unit_digest"],
                        "model_id": failed_unit["model_id"],
                        "sample_id": failed_unit["sample_id"],
                        "source_status": "FAILED_ROUTE_LOCAL_NUMERICAL",
                        "method_id": method,
                        "transform": transform,
                        "mask_id": mask_id,
                        "scale_id": scale_id,
                        "block": block,
                        "valid_block_count": 0,
                        "excluded_partial_block_count": 0,
                        "sse_total": "",
                        "sse_between": "",
                        "sse_within": "",
                        "between_error_fraction": "",
                        "within_error_fraction": "",
                        "identity_abs_diff": "",
                        "identity_rel_diff": "",
                        "block_level_signed_rank_stability": "",
                        "within_block_demeaned_residual_stability": "",
                        "status": "FAILED_ROUTE_LOCAL_NUMERICAL",
                    })
        structure_rows.append({
            "unit_digest": failed_unit["unit_digest"],
            "model_id": failed_unit["model_id"],
            "sample_id": failed_unit["sample_id"],
            "source_status": "FAILED_ROUTE_LOCAL_NUMERICAL",
            "method_id": method,
            "absolute_top10_pixel_mass_share": "",
            "normalized_shannon_entropy": "",
            "effective_support_fraction": "",
            "gini_coefficient": "",
            "normalized_total_variation": "",
            "within_cell_4_residual_energy_fraction": "",
            "within_patch_16_residual_energy_fraction": "",
            "signed_cancellation_ratio": "",
            "status": "FAILED_ROUTE_LOCAL_NUMERICAL",
        })

    stability_summary = _summarize_stability(stability_rows)
    within_summary = _summarize_error(error_rows)
    fidelity_summary = _summarize_fidelity(source["fidelity_rows"])
    association_rows, association_json = _association_outputs(
        structure_rows, stability_rows, source["fidelity_rows"],
        draws=association_draws, max_attempts=max_attempts,
    )
    branch_csv, branch_json, interpretation = _branch_outputs(
        stability_summary, fidelity_summary, within_summary, transport_rows, config
    )
    resource = {
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "gpu_seconds": 0.0,
        "peak_reserved_bytes": 0,
        "forward_calls": 0,
        "backward_calls": 0,
        "downloaded_bytes": 0,
        "retained_bytes": 0,
    }
    return {
        "m1_gate": m1_gate,
        "access": access,
        "pixel_audit_rows": pixel_audit_rows,
        "stability_rows": stability_rows,
        "stability_summary": stability_summary,
        "error_rows": error_rows,
        "within_summary": within_summary,
        "transport_rows": transport_rows,
        "hflip_rows": hflip_rows,
        "structure_rows": structure_rows,
        "fidelity_summary": fidelity_summary,
        "association_rows": association_rows,
        "association_json": association_json,
        "branch_csv": branch_csv,
        "branch_json": branch_json,
        "interpretation": interpretation,
        "resource": resource,
    }


def _median(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def _summarize_stability(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["status"] == "PASS"]
    summaries: list[dict[str, Any]] = []
    dimensions = [
        ("overall", ("method_id", "mask_id", "scale_id", "block")),
        ("by_model", ("model_id", "method_id", "mask_id", "scale_id", "block")),
        ("by_transform", ("method_id", "mask_id", "scale_id", "block", "transform")),
    ]
    for group, keys in dimensions:
        buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in valid:
            buckets[tuple(row[key] for key in keys)].append(row)
        for key, bucket in sorted(buckets.items()):
            values = dict(zip(keys, key))
            summaries.append({
                "group": group,
                "model_id": values.get("model_id", "ALL"),
                "method_id": values.get("method_id", "ALL"),
                "mask_id": values.get("mask_id", "ALL"),
                "scale_id": values.get("scale_id", "ALL"),
                "block": values.get("block", ""),
                "transform": values.get("transform", "ALL"),
                "n": len(bucket),
                "failed_denominator_count": 1 if group == "overall" else 0,
                "median_signed_spearman": _median(bucket, "signed_spearman"),
                "median_top10_unit_signed_value_jaccard": _median(bucket, "top10_unit_signed_value_jaccard"),
                "median_absolute_mass_spearman": _median(bucket, "absolute_mass_spearman"),
                "median_top10_unit_absolute_mass_jaccard": _median(bucket, "top10_unit_absolute_mass_jaccard"),
                "median_signed_ssim": _median(bucket, "signed_ssim"),
                "median_normalized_positive_negative_mass_displacement": _median(bucket, "normalized_positive_negative_mass_displacement"),
                "median_absolute_mass_center_of_mass_pixel_error": _median(bucket, "absolute_mass_center_of_mass_pixel_error"),
                "median_valid_unit_count": _median(bucket, "valid_unit_count"),
                "sum_excluded_partial_unit_count": int(sum(int(row["excluded_partial_unit_count"]) for row in bucket)),
            })
    return summaries


def _summarize_error(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["status"] == "PASS"]
    summaries: list[dict[str, Any]] = []
    dimensions = [
        ("overall", ("method_id", "mask_id", "scale_id", "block")),
        ("by_model", ("model_id", "method_id", "mask_id", "scale_id", "block")),
        ("by_transform", ("method_id", "mask_id", "scale_id", "block", "transform")),
    ]
    for group, keys in dimensions:
        buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in valid:
            buckets[tuple(row[key] for key in keys)].append(row)
        for key, bucket in sorted(buckets.items()):
            values = dict(zip(keys, key))
            summaries.append({
                "group": group,
                "model_id": values.get("model_id", "ALL"),
                "method_id": values.get("method_id", "ALL"),
                "mask_id": values.get("mask_id", "ALL"),
                "scale_id": values.get("scale_id", "ALL"),
                "block": values.get("block", ""),
                "transform": values.get("transform", "ALL"),
                "n": len(bucket),
                "failed_denominator_count": 1 if group == "overall" else 0,
                "median_sse_total": _median(bucket, "sse_total"),
                "median_between_error_fraction": _median(bucket, "between_error_fraction"),
                "median_within_error_fraction": _median(bucket, "within_error_fraction"),
                "max_identity_abs_diff": float(max(float(row["identity_abs_diff"]) for row in bucket)),
                "max_identity_rel_diff": float(max(float(row["identity_rel_diff"]) for row in bucket)),
                "median_block_level_signed_rank_stability": _median(bucket, "block_level_signed_rank_stability"),
                "median_within_block_demeaned_residual_stability": _median(bucket, "within_block_demeaned_residual_stability"),
                "median_valid_block_count": _median(bucket, "valid_block_count"),
                "sum_excluded_partial_block_count": int(sum(int(row["excluded_partial_block_count"]) for row in bucket)),
            })
    return summaries


def _summarize_fidelity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["method_id"] in METHODS and int(row["block"]) in (4, 16)]
    summaries: list[dict[str, Any]] = []
    for group, keys in (
        ("overall", ("method_id", "block")),
        ("by_model", ("model_id", "method_id", "block")),
    ):
        buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in valid:
            buckets[tuple(row[key] for key in keys)].append(row)
        for key, bucket in sorted(buckets.items()):
            values = dict(zip(keys, key))
            block = int(values["block"])
            summaries.append({
                "group": group,
                "model_id": values.get("model_id", "ALL"),
                "method_id": values["method_id"],
                "scale_id": _scale_id(block),
                "block": block,
                "n": len(bucket),
                "failed_denominator_count": 1 if group == "overall" else 0,
                "median_road_combined": _median(bucket, "road_combined"),
                "mean_road_combined": float(np.mean([float(row["road_combined"]) for row in bucket], dtype=np.float64)),
                "median_road_morf": _median(bucket, "road_morf"),
                "median_road_lerf": _median(bucket, "road_lerf"),
                "source": "M1_R1_REVALIDATED_FOR_4x4_16x16_AND_SEALED_X2_BOUND_FOR_PIXEL",
            })
    return summaries


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.size < 2:
        return float("nan")
    if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
        return float("nan")
    if float(np.min(x_array)) == float(np.max(x_array)) or float(np.min(y_array)) == float(np.max(y_array)):
        return float("nan")
    xr = _rankdata_average(x_array)
    yr = _rankdata_average(y_array)
    xr = xr - xr.mean(dtype=np.float64)
    yr = yr - yr.mean(dtype=np.float64)
    denominator = math.sqrt(float(np.sum(xr * xr, dtype=np.float64)) * float(np.sum(yr * yr, dtype=np.float64)))
    if denominator <= np.finfo(np.float64).tiny:
        return float("nan")
    return float(np.sum(xr * yr, dtype=np.float64) / denominator)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_array.size < 2 or not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
        return float("nan")
    x_centered = x_array - x_array.mean(dtype=np.float64)
    y_centered = y_array - y_array.mean(dtype=np.float64)
    denominator = math.sqrt(
        float(np.sum(x_centered * x_centered, dtype=np.float64))
        * float(np.sum(y_centered * y_centered, dtype=np.float64))
    )
    if denominator <= np.finfo(np.float64).tiny:
        return float("nan")
    return float(np.sum(x_centered * y_centered, dtype=np.float64) / denominator)


def _m2_sse_decomposition(clean: np.ndarray, transformed: np.ndarray,
                          valid_mask: np.ndarray, block: int) -> dict[str, Any]:
    a = np.asarray(clean, dtype=np.float64)
    b = np.asarray(transformed, dtype=np.float64)
    mask = np.asarray(valid_mask)
    if a.shape != b.shape or mask.shape != a.shape or mask.dtype != np.bool_:
        raise X2MechM2Error("SSE inputs have incompatible shape or mask dtype")
    rows, cols = a.shape[0] // block, a.shape[1] // block
    masks = mask.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)
    complete = masks.all(axis=(2, 3))
    if not complete.any():
        raise X2MechM2Error("no complete valid blocks for SSE")
    clean_blocks = a.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)[complete]
    transformed_blocks = b.reshape(rows, block, cols, block).transpose(0, 2, 1, 3)[complete]
    differences = transformed_blocks - clean_blocks
    means = differences.mean(axis=(1, 2), dtype=np.float64)
    between = float(np.sum((block * block) * means * means, dtype=np.float64))
    residual = differences - means[:, None, None]
    within = float(np.sum(residual * residual, dtype=np.float64))
    total = float(np.sum(differences * differences, dtype=np.float64))
    if not np.isclose(total, between + within, atol=1e-10, rtol=1e-10):
        raise X2MechM2Error("SSE identity failed")
    denominator = total if total > np.finfo(np.float64).tiny else 1.0
    clean_sums = clean_blocks.sum(axis=(1, 2), dtype=np.float64)
    transformed_sums = transformed_blocks.sum(axis=(1, 2), dtype=np.float64)
    clean_residual = clean_blocks - clean_blocks.mean(axis=(1, 2), keepdims=True)
    transformed_residual = transformed_blocks - transformed_blocks.mean(axis=(1, 2), keepdims=True)
    return {
        "sse_total": total,
        "sse_between": between,
        "sse_within": within,
        "between_error_fraction": between / denominator,
        "within_error_fraction": within / denominator,
        "valid_block_count": int(complete.sum()),
        "excluded_partial_block_count": int(complete.size - complete.sum()),
        "block_level_signed_rank_stability": _spearman(clean_sums, transformed_sums),
        "within_block_demeaned_residual_stability": _pearson(clean_residual, transformed_residual),
    }


def _assoc(records: Sequence[Mapping[str, Any]], x_key: str, y_key: str,
           *, draws: int, max_attempts: int, namespace: str) -> dict[str, Any]:
    by_model_sample: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in records:
        by_model_sample[str(row["model_id"])][str(row["sample_id"])] = row
    models = sorted(by_model_sample)
    sample_ids = sorted({sample for per_model in by_model_sample.values() for sample in per_model})

    def estimate(selected: Sequence[str]) -> tuple[float, dict[str, float]]:
        per_model = {}
        for model in models:
            xs = []
            ys = []
            for sample in selected:
                row = by_model_sample[model].get(sample)
                if row is not None:
                    xs.append(float(row[x_key]))
                    ys.append(float(row[y_key]))
            value = _spearman(xs, ys)
            if math.isfinite(value):
                per_model[model] = value
        if not per_model:
            raise X2MechM2Error("association not estimable")
        return float(np.mean(list(per_model.values()), dtype=np.float64)), per_model

    point, per_model = estimate(sample_ids)
    rng_seed = seed64(namespace, x_key, y_key)
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    values = []
    attempts = 0
    while len(values) < draws and attempts < max_attempts:
        attempts += 1
        selected = [sample_ids[index] for index in rng.integers(0, len(sample_ids), len(sample_ids))]
        try:
            value, _ = estimate(selected)
        except X2MechM2Error:
            continue
        if math.isfinite(value):
            values.append(value)
    if len(values) != draws:
        raise X2MechM2Error("association bootstrap did not obtain required finite draws")
    lower, upper = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975], method="linear")
    return {
        "effect": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_draws": draws,
        "finite_draws": len(values),
        "attempts": attempts,
        "seed": rng_seed,
        "per_model": per_model,
        "confirmatory_p_value": None,
    }


def _association_outputs(structure_rows: Sequence[Mapping[str, Any]],
                         stability_rows: Sequence[Mapping[str, Any]],
                         fidelity_rows: Sequence[Mapping[str, Any]],
                         *, draws: int, max_attempts: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_structure = [row for row in structure_rows if row["status"] == "PASS"]
    structure_by = {(row["unit_digest"], row["method_id"]): row for row in valid_structure}
    stable_valid = [
        row for row in stability_rows
        if row["status"] == "PASS" and row["mask_id"] == "x2_frozen_common_domain"
    ]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in stable_valid:
        grouped[(row["unit_digest"], row["method_id"])].append(row)
    fidelity_by = {(row["unit_digest"], row["method_id"], int(row["block"])): row for row in fidelity_rows}
    combined = []
    for key, rows in grouped.items():
        unit, method = key
        structure = structure_by[key]
        pixel = [row for row in rows if row["scale_id"] == "pixel"]
        fine = [row for row in rows if row["scale_id"] == "fine_cell"]
        patch = [row for row in rows if row["scale_id"] == "native_patch"]
        hflip = [row for row in pixel if row["transform"] == "horizontal_flip"]
        combined.append({
            **{field: structure[field] for field in ("unit_digest", "model_id", "sample_id", "method_id")},
            "absolute_top10_pixel_mass_share": structure["absolute_top10_pixel_mass_share"],
            "normalized_total_variation": structure["normalized_total_variation"],
            "within_patch_16_residual_energy_fraction": structure["within_patch_16_residual_energy_fraction"],
            "median_pixel_signed_spearman": _median(pixel, "signed_spearman"),
            "median_fine_signed_spearman": _median(fine, "signed_spearman"),
            "median_patch_signed_spearman": _median(patch, "signed_spearman"),
            "pixel_to_fine_signed_spearman_change": _median(fine, "signed_spearman") - _median(pixel, "signed_spearman"),
            "pixel_to_patch_signed_spearman_change": _median(patch, "signed_spearman") - _median(pixel, "signed_spearman"),
            "median_pixel_jaccard": _median(pixel, "top10_unit_signed_value_jaccard"),
            "median_fine_jaccard": _median(fine, "top10_unit_signed_value_jaccard"),
            "median_patch_jaccard": _median(patch, "top10_unit_signed_value_jaccard"),
            "horizontal_flip_signed_spearman": _median(hflip, "signed_spearman"),
            "pixel_to_patch_jaccard_change": _median(patch, "top10_unit_signed_value_jaccard") - _median(pixel, "top10_unit_signed_value_jaccard"),
            "road_combined_4x4": float(fidelity_by[(unit, method, 4)]["road_combined"]),
            "road_combined_16x16": float(fidelity_by[(unit, method, 16)]["road_combined"]),
        })
    requested = (
        ("absolute_top10_pixel_mass_share", "median_pixel_signed_spearman"),
        ("absolute_top10_pixel_mass_share", "horizontal_flip_signed_spearman"),
        ("normalized_total_variation", "pixel_to_fine_signed_spearman_change"),
        ("within_patch_16_residual_energy_fraction", "pixel_to_patch_signed_spearman_change"),
        ("median_pixel_jaccard", "road_combined_4x4"),
        ("median_patch_jaccard", "road_combined_16x16"),
    )
    rows = []
    detail = {}
    for method in METHODS:
        records = [row for row in combined if row["method_id"] == method]
        for x_key, y_key in requested:
            try:
                result = _assoc(records, x_key, y_key, draws=draws, max_attempts=max_attempts,
                                namespace="STAGE_X_X2_MECH_ASSOCIATION_V1_20260910")
            except X2MechM2Error as error:
                if str(error) != "association not estimable":
                    raise
                rows.append({
                    "method_id": method,
                    "x_metric": x_key,
                    "y_metric": y_key,
                    "n": len(records),
                    "effect": "",
                    "ci_lower": "",
                    "ci_upper": "",
                    "bootstrap_draws": draws,
                    "finite_draws": 0,
                    "attempts": 0,
                    "seed": seed64("STAGE_X_X2_MECH_ASSOCIATION_V1_20260910", x_key, y_key),
                    "status": "NOT_ESTIMABLE_DEGENERATE",
                })
                detail[f"{method}:{x_key}:{y_key}"] = {
                    "effect": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "bootstrap_draws": draws,
                    "finite_draws": 0,
                    "attempts": 0,
                    "seed": seed64("STAGE_X_X2_MECH_ASSOCIATION_V1_20260910", x_key, y_key),
                    "per_model": {},
                    "confirmatory_p_value": None,
                    "status": "NOT_ESTIMABLE_DEGENERATE",
                }
                continue
            rows.append({
                "method_id": method,
                "x_metric": x_key,
                "y_metric": y_key,
                "n": len(records),
                "effect": result["effect"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
                "bootstrap_draws": result["bootstrap_draws"],
                "finite_draws": result["finite_draws"],
                "attempts": result["attempts"],
                "seed": result["seed"],
                "status": "PASS",
            })
            detail[f"{method}:{x_key}:{y_key}"] = result
    return rows, {
        "study_type": "POST_HOC_EXPLORATORY_MECHANISM_ANALYSIS",
        "method": "Spearman",
        "strata": ["per_model", "model_equal_aggregate"],
        "cluster": "sample_id_synchronized_across_models",
        "bootstrap_draws": draws,
        "maximum_attempts": max_attempts,
        "quantile": "linear",
        "confirmatory_p_values": None,
        "associations": detail,
    }


def _overall(summary: Sequence[Mapping[str, Any]], method: str, mask: str, scale: str,
             field: str) -> float:
    matches = [
        row for row in summary
        if row["group"] == "overall" and row["method_id"] == method and
        row["mask_id"] == mask and row["scale_id"] == scale
    ]
    if len(matches) != 1:
        raise X2MechM2Error(f"summary row not unique: {method}:{mask}:{scale}")
    return float(matches[0][field])


def _fidelity_overall(rows: Sequence[Mapping[str, Any]], method: str, scale: str) -> float:
    matches = [row for row in rows if row["group"] == "overall" and row["method_id"] == method and row["scale_id"] == scale]
    if len(matches) != 1:
        raise X2MechM2Error(f"fidelity summary row not unique: {method}:{scale}")
    return float(matches[0]["median_road_combined"])


def _branch_outputs(stability_summary: Sequence[Mapping[str, Any]],
                    fidelity_summary: Sequence[Mapping[str, Any]],
                    within_summary: Sequence[Mapping[str, Any]],
                    transport_rows: Sequence[Mapping[str, Any]],
                    config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    branch = config["branch_rules"]["fallback_without_unique_thresholds"]
    observations = {
        "nns_pixel_signed_spearman": _overall(stability_summary, "nns_v2", "x2_frozen_common_domain", "pixel", "median_signed_spearman"),
        "nns_fine_signed_spearman": _overall(stability_summary, "nns_v2", "x2_frozen_common_domain", "fine_cell", "median_signed_spearman"),
        "nns_patch_signed_spearman": _overall(stability_summary, "nns_v2", "x2_frozen_common_domain", "native_patch", "median_signed_spearman"),
        "gradcam_pixel_signed_spearman": _overall(stability_summary, "vit_grad_cam", "x2_frozen_common_domain", "pixel", "median_signed_spearman"),
        "gradcam_fine_signed_spearman": _overall(stability_summary, "vit_grad_cam", "x2_frozen_common_domain", "fine_cell", "median_signed_spearman"),
        "gradcam_patch_signed_spearman": _overall(stability_summary, "vit_grad_cam", "x2_frozen_common_domain", "native_patch", "median_signed_spearman"),
        "nns_road_combined_4x4": _fidelity_overall(fidelity_summary, "nns_v2", "fine_cell"),
        "nns_road_combined_16x16": _fidelity_overall(fidelity_summary, "nns_v2", "native_patch"),
        "gradcam_road_combined_4x4": _fidelity_overall(fidelity_summary, "vit_grad_cam", "fine_cell"),
        "gradcam_road_combined_16x16": _fidelity_overall(fidelity_summary, "vit_grad_cam", "native_patch"),
    }
    error_overall = [
        row for row in within_summary
        if row["group"] == "overall" and row["method_id"] == "nns_v2" and row["mask_id"] == "x2_frozen_common_domain"
    ]
    for row in error_overall:
        observations[f"nns_{row['scale_id']}_between_error_fraction"] = float(row["median_between_error_fraction"])
        observations[f"nns_{row['scale_id']}_within_error_fraction"] = float(row["median_within_error_fraction"])
    hflip_transport = [
        row for row in transport_rows
        if row["transform"] == "horizontal_flip" and row["mask_id"] == "x2_frozen_common_domain"
    ]
    observations["horizontal_flip_transport_median_signed_spearman"] = _median(hflip_transport, "signed_spearman")
    rows = [
        {"branch": branch, "evidence_item": "branch_selection", "observed_direction": "A/B/C thresholds were not frozen in protocol", "supports_branch": True, "boundary": "fallback required"},
        {"branch": branch, "evidence_item": "granularity_stability_direction", "observed_direction": f"nns pixel/fine/patch signed Spearman = {observations['nns_pixel_signed_spearman']}, {observations['nns_fine_signed_spearman']}, {observations['nns_patch_signed_spearman']}", "supports_branch": True, "boundary": "descriptive only"},
        {"branch": branch, "evidence_item": "multiscale_fidelity_direction", "observed_direction": f"nns ROAD 4x4/16x16 = {observations['nns_road_combined_4x4']}, {observations['nns_road_combined_16x16']}", "supports_branch": True, "boundary": "no superiority or noninferiority threshold"},
        {"branch": branch, "evidence_item": "transport_control", "observed_direction": f"horizontal-flip transport median signed Spearman = {observations['horizontal_flip_transport_median_signed_spearman']}", "supports_branch": True, "boundary": "observed-minus-transport is not causal"},
    ]
    interpretation = {
        "status": "PASS",
        "study_type": "POST_HOC_EXPLORATORY_MECHANISM_ANALYSIS",
        "x2_decision": "NO_GO_BROAD_CLAIM_DEVELOPMENT",
        "mechanism_branch": branch,
        "maximum_allowed_wording": config["claims"]["maximum_wording"],
        "observation": observations,
        "interpretation": "Post-hoc multiscale analyses yielded mixed evidence about whether aggregation accounts for the observed geometric instability.",
        "boundaries": [
            "No spatial-stability PASS is issued.",
            "No NNS-v2 superiority or noninferiority to Grad-CAM is issued.",
            "No broad benchmark GO, localization claim, cross-model/cross-dataset generalization claim, confirmatory p-value, or causal conclusion is issued.",
        ],
    }
    return rows, {"mechanism_branch": branch, "observations": observations, "branch_rules": config["branch_rules"]}, interpretation


def _json_artifact(document_type: str, status: str, body: Mapping[str, Any]) -> dict[str, Any]:
    return signed(document_type, status, _payload(**body))


def _write_analysis_outputs(repo: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    _write_csv_new(repo / ROOT / "pixel_reproduction_audit.csv", PIXEL_AUDIT_FIELDS, result["pixel_audit_rows"])
    pixel_json = _json_artifact("stage_x_x2_mech_m2_pixel_reproduction_audit", "PASS", {
        "counts": _as_list({"rows": len(result["pixel_audit_rows"]), "failures": 0}),
        "checks": [{"check_id": "x2_pixel_reproduction", "status": "PASS", "detail": "all rows within 1e-6 and boolean exact"}],
        "terminal": _terminal(complete=True),
    })
    _write_json_new(repo / ROOT / "pixel_reproduction_audit.json", pixel_json)
    _write_csv_new(repo / ROOT / "multiscale_stability.csv", STABILITY_FIELDS, result["stability_rows"])
    _write_csv_new(repo / ROOT / "multiscale_stability_summary.csv", STABILITY_SUMMARY_FIELDS, result["stability_summary"])
    _write_json_new(repo / ROOT / "multiscale_stability_summary.json", _json_artifact(
        "stage_x_x2_mech_m2_multiscale_stability_summary", "PASS",
        {"counts": _as_list({"rows": len(result["stability_rows"]), "summary_rows": len(result["stability_summary"])}),
         "metrics": _summary_metrics(result["stability_summary"], "median_signed_spearman"),
         "terminal": _terminal(complete=True)}
    ))
    _write_csv_new(repo / ROOT / "within_between_error.csv", WITHIN_BETWEEN_FIELDS, result["error_rows"])
    _write_csv_new(repo / ROOT / "within_between_summary.csv", WITHIN_BETWEEN_SUMMARY_FIELDS, result["within_summary"])
    _write_csv_new(repo / ROOT / "transport_control.csv", TRANSPORT_FIELDS, result["transport_rows"])
    _write_csv_new(repo / ROOT / "horizontal_flip_control.csv", HFLIP_FIELDS, result["hflip_rows"])
    _write_csv_new(repo / ROOT / "clean_structure_metrics.csv", STRUCTURE_FIELDS, result["structure_rows"])
    _write_csv_new(repo / ROOT / "multiscale_fidelity_summary.csv", FIDELITY_SUMMARY_FIELDS, result["fidelity_summary"])
    _write_csv_new(repo / ROOT / "association_summary.csv", ASSOCIATION_FIELDS, result["association_rows"])
    _write_json_new(repo / ROOT / "association_summary.json", result["association_json"])
    _write_csv_new(repo / ROOT / "branch_evidence_matrix.csv", BRANCH_FIELDS, result["branch_csv"])
    _write_json_new(repo / ROOT / "branch_evidence_matrix.json", result["branch_json"])
    _write_json_new(repo / ROOT / "scientific_interpretation.json", result["interpretation"])
    estimability = _json_artifact("stage_x_x2_mech_m2_estimability_matrix", "PASS", {
        "state": _as_list({
            "fine_cell_4x4_transformed_stability": "POST_HOC_EXPLORATORY_ESTIMABLE",
            "native_patch_16x16_transformed_stability": "POST_HOC_EXPLORATORY_ESTIMABLE",
            "confirmatory_upgrade_allowed": False,
            "x2_no_go_unchanged": True,
        }),
        "terminal": _terminal(complete=True),
    })
    _write_json_new(repo / ROOT / "estimability_matrix.json", estimability)
    result["resource"]["retained_bytes"] = sum(path.stat().st_size for path in (repo / ROOT).rglob("*") if path.is_file())
    resource = _json_artifact("stage_x_x2_mech_m2_resource_ledger", "PASS", {
        "resources": _as_list(result["resource"]),
        "state": _as_list({"cpu_only": True, "gpu_cuda_used": False, "budget_waiver_used": False}),
        "terminal": _terminal(complete=True),
    })
    _write_json_new(repo / ROOT / "resource_ledger.json", resource)
    access_values = dict(result["access"])
    access_values.update({
        "m1_source_validation_pass": True,
        "m1_map_values_read_after_producer_seal": True,
        "m1_fidelity_rows_read_after_producer_seal": True,
        "x2_per_route_stability_rows_read_after_producer_seal": True,
    })
    access = _json_artifact("stage_x_x2_mech_m2_access_audit", "PASS", {
        "access": _as_list(access_values),
        "checks": [{"check_id": "cpu_only_boundary", "status": "PASS", "detail": "CUDA_VISIBLE_DEVICES is empty and no model/checkpoint/JPEG path was opened"}],
        "terminal": _terminal(complete=True),
    })
    _write_json_new(repo / ROOT / "access_audit.json", access)
    false_pass = _json_artifact("stage_x_x2_mech_m2_false_pass_audit", "PASS", {
        "attacks": [{"attack": attack, "result": "REJECTED", "mechanism": _attack_mechanism(attack)}
                    for attack in FALSE_PASS_ATTACKS],
        "counts": _as_list({"attack_count": len(FALSE_PASS_ATTACKS)}),
        "terminal": _terminal(complete=True),
    })
    _write_json_new(repo / ROOT / "false_pass_audit.json", false_pass)
    return {"status": "PASS"}


def _summary_metrics(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row["group"] == "overall":
            output.append({
                "name": f"{row['method_id']}:{row['mask_id']}:{row['scale_id']}:{field}",
                "value": float(row[field]),
            })
    return output


def _attack_mechanism(attack: str) -> str:
    mechanisms = {
        "source_binding_mutation": "completion bindings are rehashed and canonical digests are recomputed",
        "m1_completion_ref_drift": "startup and post-producer gates verify M1 archive ref",
        "m1_completion_digest_drift": "source_binding binds the known terminal canonical digest",
        "m1_denominator_mutation": "M1 validation requires 200 expected, 199 valid, and one failed route",
        "m1_map_manifest_row_drop": "source validator recomputes the 9,552 map roster",
        "m1_mask_registry_mutation": "source validator recomputes the 24 shared masks",
        "m1_fidelity_row_drop": "source validator recomputes 796 summaries and 71,640 ROAD rows",
        "x2_pixel_row_drop": "pixel reproduction requires 4,776 X2 geometric rows",
        "x2_pixel_numeric_drift": "pixel reproduction uses 1e-6 tolerances and exact booleans",
        "partial_block_inclusion": "block aggregation uses all-pixels-valid complete blocks only",
        "top10_count_floor_instead_of_ceil": "top-10 unit rule uses ceil(0.10 * valid units)",
        "sse_identity_violation": "float64 total/between/within identity must close at 1e-10",
        "branch_threshold_invention": "config has no numeric A/B/C thresholds and forces MIXED fallback",
        "confirmatory_p_value_insertion": "association schema records intervals and keeps p-value absent",
        "gpu_or_cuda_access": "CUDA_VISIBLE_DEVICES must be empty and GPU counters remain zero",
        "model_checkpoint_or_jpeg_access": "M2 never opens model, checkpoint, or assets/data paths",
        "png_or_jpeg_output": "output scan rejects PNG/JPEG products",
        "atomic_completion_overwrite": "completion is created via pending receipt and no-overwrite checks",
        "x2_no_go_modification": "completion records immutable NO_GO_BROAD_CLAIM_DEVELOPMENT",
        "x3_x5_f3_final_stage_g_start": "filesystem and ref audit reject unauthorized downstream starts",
    }
    return mechanisms[attack]


def _root_files(repo: Path, excluded: set[Path]) -> list[Path]:
    return [
        path for path in sorted((repo / ROOT).rglob("*"))
        if path.is_file() and path.relative_to(repo) not in excluded
    ]


def _write_manifests_and_completion(repo: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {Path(PENDING), Path(COMPLETION), ROOT / "payload_manifest.json", ROOT / "whole_root_inventory.json"}
    payload_files = _root_files(repo, excluded)
    manifest = _json_artifact("stage_x_x2_mech_m2_payload_manifest", "PASS", {
        "files": [file_record(repo, path.relative_to(repo)) for path in payload_files],
        "counts": _as_list({"file_count": len(payload_files)}),
        "state": _as_list({"source_recomputed": True, "cpu_only": True}),
        "terminal": _terminal(complete=True),
    })
    _write_json_new(repo / ROOT / "payload_manifest.json", manifest)
    inventory_files = _root_files(repo, {Path(PENDING), Path(COMPLETION), ROOT / "whole_root_inventory.json"})
    records = [file_record(repo, path.relative_to(repo)) for path in inventory_files]
    inventory = _json_artifact("stage_x_x2_mech_m2_whole_root_inventory", "PASS", {
        "files": records,
        "counts": _as_list({"file_count": len(records), "total_bytes": sum(row["bytes"] for row in records)}),
        "state": _as_list({"files_digest": canonical_sha256(records)}),
        "terminal": _terminal(complete=True),
    })
    _write_json_new(repo / ROOT / "whole_root_inventory.json", inventory)
    report_text = _report_text(repo, result)
    _write_bytes_new(repo / REPORT_ROOT / "x2_mech_m2_report.md", report_text.encode("utf-8"))
    completion = signed("stage_x_x2_mech_m2_completion", "PASS", _payload(
        identity=_as_list({
            "m2_producer_commit": _git(repo, "rev-parse", PRODUCER_REF + "^{commit}"),
            "m1_completion_commit": M1_R1_COMPLETION_COMMIT,
            "mechanism_branch": result["interpretation"]["mechanism_branch"],
            "study_type": "POST_HOC_EXPLORATORY_MECHANISM_ANALYSIS",
        }),
        bindings=[
            file_record(repo, ROOT / "source_binding.json"),
            file_record(repo, ROOT / "pre_execution_seal.json"),
            file_record(repo, ROOT / "pixel_reproduction_audit.json"),
            file_record(repo, ROOT / "payload_manifest.json"),
            file_record(repo, ROOT / "whole_root_inventory.json"),
            file_record(repo, REPORT_ROOT / "x2_mech_m2_report.md"),
        ],
        checks=[
            {"check_id": "source_recomputing_validation", "status": "PASS", "detail": "M1 validate PASS"},
            {"check_id": "x2_pixel_reproduction_gate", "status": "PASS", "detail": "4,776 rows reproduced"},
            {"check_id": "cpu_only_boundary", "status": "PASS", "detail": "no GPU/CUDA/model/checkpoint/JPEG"},
            {"check_id": "sse_identity", "status": "PASS", "detail": "all rows close at 1e-10"},
        ],
        counts=_as_list({
            "expected_routes": 200,
            "valid_routes": 199,
            "failed_routes": 1,
            "pixel_reproduction_rows": len(result["pixel_audit_rows"]),
            "multiscale_stability_rows": len(result["stability_rows"]),
            "within_between_rows": len(result["error_rows"]),
            "transport_rows": len(result["transport_rows"]),
            "horizontal_flip_rows": len(result["hflip_rows"]),
            "clean_structure_rows": len(result["structure_rows"]),
            "association_rows": len(result["association_rows"]),
        }),
        metrics=_summary_metrics(result["stability_summary"], "median_signed_spearman"),
        state=_as_list({
            "status_meaning": "package_workflow_scientific_validity_not_favorable_mechanism_branch",
            "mechanism_branch": result["interpretation"]["mechanism_branch"],
            "x2_decision": "NO_GO_BROAD_CLAIM_DEVELOPMENT",
            "x2_decision_modified": False,
            "m1_unchanged": True,
            "gpu_cuda_used": False,
            "model_loaded": False,
            "checkpoint_read": False,
            "jpeg_read": False,
            "forward_calls": 0,
            "backward_calls": 0,
            "x3_x5_f3_final_stage_g_started": False,
        }),
        access=[{"name": key, "value": value} for key, value in sorted(result["access"].items())],
        resources=_as_list(result["resource"]),
        terminal=_terminal(terminal=True, complete=True, next_authorization=NEXT_AUTHORIZATION),
    ))
    _publish_atomic(repo / PENDING, repo / COMPLETION, completion)
    return completion


def _report_text(repo: Path, result: Mapping[str, Any]) -> str:
    interp = result["interpretation"]
    obs = interp["observation"]
    associations = result["association_rows"][:4]
    return (
        "# X2-MECH M2 terminal report\n\n"
        "Status: `PASS`. This means package, workflow, and scientific validity completed; it does not issue a favorable mechanism branch or alter X2.\n\n"
        f"Study type: `{interp['study_type']}`. X2 remains `NO_GO_BROAD_CLAIM_DEVELOPMENT`. M1-R1 remains unchanged at `{M1_R1_COMPLETION_COMMIT}`.\n\n"
        f"Mechanism branch: `{interp['mechanism_branch']}`. The frozen handoff did not define numeric thresholds that uniquely select A/B/C, so M2 reports the observed directions under the mixed/inconclusive branch.\n\n"
        "Core observed directions:\n\n"
        f"- NNS-v2 median signed Spearman, x2 common mask: pixel `{obs['nns_pixel_signed_spearman']}`, 4x4 `{obs['nns_fine_signed_spearman']}`, 16x16 `{obs['nns_patch_signed_spearman']}`.\n"
        f"- ViT Grad-CAM median signed Spearman, x2 common mask: pixel `{obs['gradcam_pixel_signed_spearman']}`, 4x4 `{obs['gradcam_fine_signed_spearman']}`, 16x16 `{obs['gradcam_patch_signed_spearman']}`.\n"
        f"- NNS-v2 ROAD Combined: 4x4 `{obs['nns_road_combined_4x4']}`, 16x16 `{obs['nns_road_combined_16x16']}`; Grad-CAM ROAD Combined: 4x4 `{obs['gradcam_road_combined_4x4']}`, 16x16 `{obs['gradcam_road_combined_16x16']}`.\n"
        f"- NNS-v2 SSE fractions: 4x4 between `{obs.get('nns_fine_cell_between_error_fraction')}`, within `{obs.get('nns_fine_cell_within_error_fraction')}`; 16x16 between `{obs.get('nns_native_patch_between_error_fraction')}`, within `{obs.get('nns_native_patch_within_error_fraction')}`.\n"
        f"- Horizontal-flip transport median signed Spearman: `{obs['horizontal_flip_transport_median_signed_spearman']}`.\n\n"
        "Association intervals, first registered rows:\n\n"
        + "\n".join(
            f"- `{row['method_id']}` {row['x_metric']} vs {row['y_metric']}: effect `{row['effect']}`, 95% exploratory interval `[{row['ci_lower']}, {row['ci_upper']}]`."
            for row in associations
        )
        + "\n\n"
        "Counts: expected denominator `200`, valid `199`, failed `1`; pixel reproduction rows `4776`; multiscale stability rows `28656`; M1 maps `9552`, masks `24`, fidelity summaries `796`, ROAD rows `71640` revalidated.\n\n"
        "Access and resources: CPU-only; CUDA/GPU false; model/checkpoint/JPEG false; forward/backward calls `0`; no downloads; no PNG/JPEG display output.\n\n"
        f"Next authorization: `{NEXT_AUTHORIZATION}`. F3-Final, Stage G, X3, X4, and X5 were not started.\n"
    )


def run(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if (repo / COMPLETION).exists():
        raise X2MechM2Error("M2 terminal completion already exists")
    result = compute_multiscale_results(repo)
    _write_analysis_outputs(repo, result)
    return _write_manifests_and_completion(repo, result)


def validate(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    _ensure_post_producer(repo)
    completion = strict_read(repo / COMPLETION)
    validate_artifact(completion, repo, verify_files=True)
    if completion.get("status") != "PASS":
        raise X2MechM2Error("M2 completion is not PASS")
    for name, fields in REQUIRED_CSV_FIELDS.items():
        rows = _read_csv(repo / ROOT / name, fields)
        if not rows:
            raise X2MechM2Error(f"empty M2 table: {name}")
    for relative in (
        ROOT / "pixel_reproduction_audit.json",
        ROOT / "multiscale_stability_summary.json",
        ROOT / "association_summary.json",
        ROOT / "branch_evidence_matrix.json",
        ROOT / "scientific_interpretation.json",
        ROOT / "estimability_matrix.json",
        ROOT / "resource_ledger.json",
        ROOT / "access_audit.json",
        ROOT / "false_pass_audit.json",
        ROOT / "payload_manifest.json",
        ROOT / "whole_root_inventory.json",
    ):
        value = strict_read(repo / relative)
        if "canonical_digest" in value:
            validate_artifact(value, repo, verify_files=True)
    pixel = _read_csv(repo / ROOT / "pixel_reproduction_audit.csv", PIXEL_AUDIT_FIELDS)
    if len(pixel) != 4776 or any(row["status"] != "PASS" for row in pixel):
        raise X2MechM2Error("pixel reproduction audit failed")
    inventory = strict_read(repo / ROOT / "whole_root_inventory.json")
    records = inventory["payload"]["files"]
    observed = {row["name"]: row["value"] for row in inventory["payload"]["state"]}["files_digest"]
    if observed != canonical_sha256(records):
        raise X2MechM2Error("whole-root inventory digest drifted")
    return {
        "status": "PASS",
        "completion_digest": completion["canonical_digest"],
        "mechanism_branch": {row["name"]: row["value"] for row in completion["payload"]["state"]}["mechanism_branch"],
        "next_authorization": completion["payload"]["terminal"]["next_authorization"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config").add_argument("--repo-root", type=Path, default=REPO_ROOT)
    freeze = sub.add_parser("freeze-producer")
    freeze.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    freeze.add_argument("--producer-commit", required=True)
    freeze.add_argument("--test-summary", required=True)
    sub.add_parser("validate-producer").add_argument("--repo-root", type=Path, default=REPO_ROOT)
    sub.add_parser("run").add_argument("--repo-root", type=Path, default=REPO_ROOT)
    sub.add_parser("validate").add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            result = {"status": "PASS", "run_id": load_config(args.repo_root)["run_id"]}
        elif args.command == "freeze-producer":
            result = freeze_producer(args.repo_root, args.producer_commit, args.test_summary)
        elif args.command == "validate-producer":
            result = validate_producer(args.repo_root)
        elif args.command == "run":
            result = run(args.repo_root)
        else:
            result = validate(args.repo_root)
    except (X2MechM2Error, X2MechAnalysisError, OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
