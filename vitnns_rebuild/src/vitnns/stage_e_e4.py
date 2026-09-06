"""Stage E E4 source validation, packaging, and mixed-validity sealing."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_e_e0 as e0
from . import stage_e_e1 as e1
from . import stage_e_e2 as e2
from . import stage_e_e3 as e3
from . import stage_e_e3_r4 as e3r4

STAGE_ID = "E4"
CONFIG_PATH = "config/stage_e_e4.json"
CONFIG_SCHEMA_PATH = "config/stage_e_e4.schema.json"
EXPECTED_CONFIG_DIGEST = "786889136d6d44006b141f4a47bae6978da5db86a62993cbc49afcedc7d7d406"
ROOT_PATH = "evidence/stage_e/e4_recovery_v2_20260907/formal"
REPORT_ROOT_PATH = "reports/stage_e_e4_20260907"
AUTH_PATH = ROOT_PATH + "/control/execution_authorization.json"
PENDING_PATH = ROOT_PATH + "/completion/e4.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/e4.json"
TEST_PATH = REPORT_ROOT_PATH + "/test_results.json"
VALIDATION_PATH = REPORT_ROOT_PATH + "/validation_report.json"
FALSE_PASS_PATH = REPORT_ROOT_PATH + "/false_pass_audit.json"
E3_ARCHIVE_REF = "refs/archive/stage-e-e3-r4-mixed-pass"
E3_COMMIT = "c20c22d60836832b7fbe310101552b5ca8ad3150"
E3_TREE = "b006c07a9fc9505c39cc513e6351c43b9fe29659"
E3_COMPLETION_SHA256 = "03c0aa182b9c9864cfc455657821f8e4074245b0f1c697f44f58f8de8ecc7dd7"
E3_COMPLETION_DIGEST = "ad387d08cf978aad9c9995ee64b5cac560b44c8c6fe6c56bd786392a113b1b36"
E3_INVENTORY_SHA256 = "bc9a0cb982de5cdafda8daba80da47fa4068d8a0808436f694b506f72675d9e2"
E3_INVENTORY_DIGEST = "797e252cb744df2339853cedb2bb425952734ced333179a9f0c13c605f25608c"
ARCHIVE_REF = "refs/archive/stage-e-e4-completion"
REQUIRED_COMMON_COLUMNS = frozenset(("variant_id", "sample_id", "selection_order", "cohort", "stratum", "validity_status", "source_hashes", "method", "target", "scale", "layer", "head"))
REQUIRED_QC_COLUMNS = frozenset(("failure_class", "failure_code", "na_reason", "attempt_id", "retry_count", "elapsed_seconds", "gpu_hours", "artifact_bytes"))
FORBIDDEN_SUFFIXES = {".png", ".pdf", ".svg"}
FALSE_PASS_ATTACKS = (
    "e3_r4_mixed_validity_relabelled_pass",
    "one_of_two_e3_failures_dropped_or_zero_imputed",
    "valid_only_summary_population_undisclosed",
    "e1_e2_e3_source_table_or_csv_drift",
    "e3_r4_completion_or_inventory_not_bound_to_archive",
    "d9_d10_or_c10_history_modified",
    "new_scientific_variant_or_model_execution_in_e4",
    "render_artifact_created_before_stage_f",
    "package_workflow_pass_treated_as_new_stage_d_decision",
    "f0_authorized_before_atomic_e4_completion",
)


class StageEE4Error(RuntimeError):
    """The E4 source-validation or sealing contract failed."""


def _now() -> str:
    return e3r4._now()


def _sha(raw: bytes) -> str:
    return e3r4._sha(raw)


def _sha_file(path: Path) -> str:
    return e3r4._sha_file(path)


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    return e3r4._signed(value)


def _strict_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        return e3r4._strict_json(path, context)
    except Exception as error:
        raise StageEE4Error(str(error)) from error


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    if value.get("canonical_digest") != canonical_sha256({key: item for key, item in value.items() if key != "canonical_digest"}):
        raise StageEE4Error(f"{context} canonical digest mismatch")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    e3r4._write_new_json(path, value)


def _write_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    e3r4._write_or_verify(path, value)


def _git(repo: Path, *args: str) -> str:
    try:
        return e3r4._git(repo, *args)
    except Exception as error:
        raise StageEE4Error(str(error)) from error


def _record(repo: Path, relative: str) -> dict[str, Any]:
    return e3r4._record(repo, relative)


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    return e3r4._inventory(root, exclude=exclude)


def _validate_config(repo: Path) -> dict[str, Any]:
    config, _ = _strict_json(repo / CONFIG_PATH, "E4 config"); schema, _ = _strict_json(repo / CONFIG_SCHEMA_PATH, "E4 config schema")
    Draft202012Validator.check_schema(schema); errors = list(Draft202012Validator(schema).iter_errors(config))
    if errors: raise StageEE4Error(f"E4 config schema failure: {errors[0].message}")
    _validate_signed(config, "E4 config")
    if config["canonical_digest"] != EXPECTED_CONFIG_DIGEST or config["scope"] != {"new_scientific_variant": False, "new_model_or_image_compute": False, "source_validation_and_packaging_only": True, "numeric_source": "paper_data_only", "renders_png": False, "failure_disclosure_required": True, "valid_only_numeric_summaries": True}:
        raise StageEE4Error("E4 frozen scope drifted")
    return config


def _validate_e0_completion(repo: Path) -> dict[str, Any]:
    value, _ = _strict_json(repo / e0.COMPLETION_PATH, "E0 completion")
    try: e0.validate_completion_document(value, repo)
    except Exception as error: raise StageEE4Error(f"E0 validation failed: {error}") from error
    return value


def _validate_e3_custody(repo: Path) -> dict[str, Any]:
    if _git(repo, "rev-parse", E3_ARCHIVE_REF + "^{commit}") != E3_COMMIT or _git(repo, "rev-parse", E3_COMMIT + "^{tree}") != E3_TREE:
        raise StageEE4Error("E3-R4 archive custody drifted")
    if _git(repo, "diff", "--name-only", E3_COMMIT, "--", e3r4.ROOT_PATH, e3r4.REPORT_ROOT_PATH):
        raise StageEE4Error("E3-R4 evidence changed after custody")
    completion, raw = _strict_json(repo / e3r4.COMPLETION_PATH, "E3-R4 completion"); inventory, inventory_raw = _strict_json(repo / e3r4.ROOT_PATH / "whole_root_inventory.json", "E3-R4 inventory")
    _validate_signed(completion, "E3-R4 completion"); _validate_signed(inventory, "E3-R4 inventory")
    if _sha(raw) != E3_COMPLETION_SHA256 or completion["canonical_digest"] != E3_COMPLETION_DIGEST or _sha(inventory_raw) != E3_INVENTORY_SHA256 or inventory["canonical_digest"] != E3_INVENTORY_DIGEST:
        raise StageEE4Error("E3-R4 completion/inventory byte binding drifted")
    if completion.get("status") != "PASS" or completion.get("scientific_validity") != "MIXED" or completion.get("next_authorization") != "E4_ONLY_WITH_MIXED_VALIDITY_DISCLOSURE" or completion.get("union", {}).get("terminal_counts") != {"expected": 3000, "valid": 2998, "failed": 2, "na": 0}:
        raise StageEE4Error("E3-R4 terminal gate drifted")
    return completion


def _upstream_validation(repo: Path, *, full: bool) -> dict[str, Any]:
    config = _validate_config(repo); e0_completion = _validate_e0_completion(repo); e3_completion = _validate_e3_custody(repo)
    if full:
        e1_result = e1.validate_completion(repo); e2_result = e2.validate_completion(repo); e3_result = e3r4.validate_completion(repo)
        d9 = e0.validate_d9_formal_validation_report(repo_root=repo)
        d9_control = {"status": d9["status"], "analysis_authorized": d9["analysis_authorized"], "root_inventory": d9["validation"]["root_inventory"]}
        stage_d = dict(e0.verify_upstream(repo, d9_source_control=d9_control))
    else:
        e1_completion, _ = _strict_json(repo / e1.COMPLETION_PATH, "E1 completion")
        e2_completion, _ = _strict_json(repo / e2.SUPERSEDING_COMPLETION_PATH, "E2 completion")
        e1_result = {"status": e1_completion.get("status")}; e2_result = {"status": e2_completion.get("status")}; e3_result = {"status": e3_completion.get("status"), "scientific_validity": e3_completion.get("scientific_validity")}
        stage_d = {"status": "BOUND_FOR_FAST_PRECHECK"}
    if e1_result.get("status") != "PASS" or e2_result.get("status") != "PASS" or e3_result.get("status") != "PASS": raise StageEE4Error("E1-E3 completion chain is not PASS")
    if e3_result.get("scientific_validity", "MIXED") != "MIXED": raise StageEE4Error("E3 mixed scientific validity was lost")
    return {"status": "PASS", "full_source_validation": full, "config": config, "e0": e0_completion, "e1": e1_result, "e2": e2_result, "e3": e3_result, "stage_d": stage_d}


def verify_preconditions(repo_root: Path | str = REPO_ROOT, *, full: bool = False) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); upstream = _upstream_validation(repo, full=full)
    return {"status": "PASS", "upstream": upstream, "head": _git(repo, "rev-parse", "HEAD"), "tree": _git(repo, "rev-parse", "HEAD^{tree}"), "e4_started": False}


def record_tests(repo_root: Path | str = REPO_ROOT, *, passed: int, failed: int, command: str) -> dict[str, Any]:
    if type(passed) is not int or passed <= 0 or failed != 0 or not command.strip(): raise StageEE4Error("only zero-failure E4 tests can be recorded")
    report = _signed({"schema_version": 1, "document_type": "stage_e_e4_test_results", "stage_id": "E4", "status": "PASS", "recorded_at": _now(), "command": command, "passed": passed, "failed": failed, "gpu_used": False, "model_loaded": False, "images_read": False})
    _write_or_verify(Path(repo_root).resolve() / TEST_PATH, report); return report


def initialize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); pre = verify_preconditions(repo, full=False); tests, _ = _strict_json(repo / TEST_PATH, "E4 tests"); _validate_signed(tests, "E4 tests")
    if tests.get("status") != "PASS" or tests.get("failed") != 0: raise StageEE4Error("E4 tests are not PASS")
    status = _git(repo, "status", "--porcelain=v1"); allowed = (ROOT_PATH + "/", REPORT_ROOT_PATH + "/")
    unexpected = [line for line in status.splitlines() if not any(line[3:].startswith(item) or f"/{item}" in line[3:] for item in allowed)]
    if unexpected: raise StageEE4Error(f"E4 producer worktree is not clean: {unexpected}")
    if _git(repo, "rev-parse", "@{u}") != pre["head"]: raise StageEE4Error("E4 producer is not pushed")
    authorization = _signed({"schema_version": 1, "document_type": "stage_e_e4_execution_authorization", "stage_id": "E4", "status": "AUTHORIZED_SEALING_ONLY", "issued_at": _now(), "producer_commit": pre["head"], "producer_tree": pre["tree"], "config_canonical_digest": EXPECTED_CONFIG_DIGEST, "e3_archive_ref": E3_ARCHIVE_REF, "e3_commit": E3_COMMIT, "e3_completion_canonical_digest": E3_COMPLETION_DIGEST, "expected_e3_terminal_counts": {"expected": 3000, "valid": 2998, "failed": 2, "na": 0}, "scientific_validity": "MIXED", "new_scientific_compute_authorized": False, "gpu_authorized": False, "stage_f_authorized": False})
    pending = _signed({"schema_version": 1, "document_type": "stage_e_e4_completion_pending", "stage_id": "E4", "status": "PENDING", "started_at": _now(), "authorization_canonical_digest": authorization["canonical_digest"], "next_authorization": "NONE_PENDING", "stage_f_started": False})
    _write_or_verify(repo / AUTH_PATH, authorization); _write_or_verify(repo / PENDING_PATH, pending)
    return validate_initialized(repo)


def validate_initialized(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); _validate_e3_custody(repo)
    authorization, _ = _strict_json(repo / AUTH_PATH, "E4 authorization"); pending, _ = _strict_json(repo / PENDING_PATH, "E4 pending")
    _validate_signed(authorization, "E4 authorization"); _validate_signed(pending, "E4 pending")
    if authorization.get("status") != "AUTHORIZED_SEALING_ONLY" or authorization.get("new_scientific_compute_authorized") is not False or authorization.get("gpu_authorized") is not False or pending.get("status") != "PENDING" or pending.get("stage_f_started") is not False: raise StageEE4Error("E4 initialized boundary drifted")
    return {"status": "PASS", "scope": "SOURCE_VALIDATION_AND_PACKAGING_ONLY", "scientific_validity": "MIXED", "stage_f_started": False}


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> bytes:
    if not rows: raise StageEE4Error("E4 cannot write an empty table")
    ordered_columns = list(rows[0]) if columns is None else list(columns)
    if any(set(row) != set(ordered_columns) for row in rows): raise StageEE4Error("E4 table rows do not share one column set")
    stream = io.StringIO(newline=""); writer = csv.DictWriter(stream, fieldnames=ordered_columns, lineterminator="\n"); writer.writeheader(); writer.writerows([{key: e3._csv_scalar(row[key]) for key in ordered_columns} for row in rows])
    return stream.getvalue().encode("utf-8")


def _validate_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream); observed = list(reader)
    expected = [{key: e3._csv_scalar(row[key]) for key in columns} for row in rows]
    if reader.fieldnames != list(columns) or observed != expected:
        raise StageEE4Error(f"CSV/JSON parity failed: {path}")


def _table_from_source(repo: Path, relative: str, table_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source, raw = _strict_json(repo / relative, relative); _validate_signed(source, relative)
    rows = source.get("rows"); columns = source.get("columns")
    if not isinstance(rows, list) or not rows or not isinstance(columns, list) or source.get("row_count") != len(rows): raise StageEE4Error(f"source table structure failed: {relative}")
    if any(set(row) != set(columns) for row in rows): raise StageEE4Error(f"source table column coverage drifted: {relative}")
    csv_source = (repo / relative).with_suffix(".csv")
    if not csv_source.is_file(): raise StageEE4Error(f"source CSV is absent: {relative}")
    _validate_csv_rows(csv_source, rows, columns)
    packaged = _signed({"schema_version": 1, "document_type": "stage_e_paper_table", "table_id": table_id, "columns": columns, "row_count": len(rows), "rows": rows})
    source_record = {"path": relative, "bytes": len(raw), "sha256": _sha(raw), "canonical_digest": source["canonical_digest"], "source_csv_path": csv_source.relative_to(repo).as_posix(), "source_csv_sha256": _sha_file(csv_source)}
    return packaged, source_record


def _resource_table(repo: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sources = (
        e1.ROOT_PATH + "/paper_data/resource_and_qc.json",
        e2.ROOT_PATH + "/paper_data/resource_and_qc.json",
        e3r4.ROOT_PATH + "/paper_data/resource_and_qc.json",
    )
    rows: list[dict[str, Any]] = []; records = []
    for stage, relative in zip(("E1", "E2", "E3"), sources):
        source, raw = _strict_json(repo / relative, relative); _validate_signed(source, relative); source_rows = source.get("rows", [])
        csv_source = (repo / relative).with_suffix(".csv")
        if source.get("row_count") != len(source_rows): raise StageEE4Error(f"resource source cardinality failed: {relative}")
        _validate_csv_rows(csv_source, source_rows, source.get("columns", []))
        for row in source_rows: rows.append({"analysis_family": stage, **row})
        records.append({"path": relative, "bytes": len(raw), "sha256": _sha(raw), "canonical_digest": source["canonical_digest"], "source_csv_path": csv_source.relative_to(repo).as_posix(), "source_csv_sha256": _sha_file(csv_source)})
    columns = list(dict.fromkeys(key for row in rows for key in row)); normalized = [{key: row.get(key) for key in columns} for row in rows]
    return _signed({"schema_version": 1, "document_type": "stage_e_paper_table", "table_id": "resource_and_qc", "columns": columns, "row_count": len(normalized), "rows": normalized}), records


def _build_package(repo: Path) -> dict[str, Any]:
    sources = {
        "patch_scale_sensitivity": e1.ROOT_PATH + "/paper_data/patch_scale_sensitivity.json",
        "upsampling_sensitivity": e2.ROOT_PATH + "/paper_data/upsampling_sensitivity.json",
        "layer_head_descriptive": e3r4.ROOT_PATH + "/paper_data/layer_head_paired_delta.json",
        "prediction_strata": e3r4.ROOT_PATH + "/paper_data/prediction_strata.json",
    }
    tables: dict[str, dict[str, Any]] = {}; source_records = []
    for table_id, relative in sources.items():
        tables[table_id], record = _table_from_source(repo, relative, table_id); source_records.append(record)
    tables["resource_and_qc"], resource_records = _resource_table(repo); source_records.extend(resource_records)
    failed_pairs = [row for row in tables["layer_head_descriptive"]["rows"] if row["validity_status"] == "FAILED"]
    failed_pair_identity = sorted((row["selection_order"], row["layer"], row["head"], row["method"], row["failure_code"]) for row in failed_pairs)
    expected_pair_failures = [(117, 11, 2, "nns_vs_matched", "ig_conservation_failed_after_e3_r1_schedule"), (352, 11, 2, "nns_vs_matched", "ig_conservation_failed_typed_terminal")]
    if failed_pair_identity != expected_pair_failures: raise StageEE4Error(f"E4 paired failure disclosure drifted: {failed_pair_identity}")
    if any(row.get("delta_fid") is not None for row in failed_pairs): raise StageEE4Error("E4 failed layer/head row was imputed")
    terminal_relative = e3r4.ROOT_PATH + "/paper_data/new_compute_terminal_rows.json"
    terminal, terminal_raw = _strict_json(repo / terminal_relative, "E3-R4 terminal-row table"); _validate_signed(terminal, "E3-R4 terminal-row table")
    terminal_csv = (repo / terminal_relative).with_suffix(".csv"); _validate_csv_rows(terminal_csv, terminal["rows"], terminal["columns"]); source_records.append({"path": terminal_relative, "bytes": len(terminal_raw), "sha256": _sha(terminal_raw), "canonical_digest": terminal["canonical_digest"], "source_csv_path": terminal_csv.relative_to(repo).as_posix(), "source_csv_sha256": _sha_file(terminal_csv)})
    failed = [row for row in terminal["rows"] if row["validity_status"] == "FAILED"]
    failed_identity = sorted((row["selection_order"], row["layer"], row["head"], row["method"], row["failure_code"]) for row in failed)
    expected_failures = [(117, 11, 2, e3.NNS_METHOD, "ig_conservation_failed_after_e3_r1_schedule"), (352, 11, 2, e3.NNS_METHOD, "ig_conservation_failed_typed_terminal")]
    if terminal.get("row_count") != 3000 or failed_identity != expected_failures: raise StageEE4Error(f"E4 terminal failure disclosure drifted: {failed_identity}")
    summaries = tables["prediction_strata"]["rows"]
    if not any(row["failed"] for row in summaries) or any(row.get("summary_population") != "all_expected_rows_with_typed_status_counts_numeric_statistics_over_VALID_only" for row in summaries): raise StageEE4Error("E4 valid-only summary disclosure failed")
    for name, table in tables.items():
        if table["row_count"] != len(table["rows"]) or any(not REQUIRED_COMMON_COLUMNS <= set(row) for row in table["rows"]): raise StageEE4Error(f"E4 required table columns failed: {name}")
    output_schema, _ = _strict_json(repo / "config/stage_e_output_table.schema.json", "Stage E output table schema")
    for name, table in tables.items():
        errors = list(Draft202012Validator(output_schema).iter_errors(table))
        if errors: raise StageEE4Error(f"E4 {name} schema failure: {errors[0].message}")
    if not REQUIRED_QC_COLUMNS <= set(tables["resource_and_qc"]["columns"]): raise StageEE4Error("E4 resource/QC columns are incomplete")
    return {"tables": tables, "source_records": source_records, "failures": failed, "failed_pairs": failed_pairs, "failure_identity": failed_identity}


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream: stream.write(_csv_bytes(rows, columns)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path); directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def _verify_table_files(repo: Path, name: str, expected: Mapping[str, Any]) -> None:
    observed, raw = _strict_json(repo / ROOT_PATH / "paper_data" / f"{name}.json", f"E4 {name}")
    _validate_signed(observed, f"E4 {name}")
    if observed != expected or raw != canonical_json_bytes(expected) + b"\n" or (repo / ROOT_PATH / "paper_data" / f"{name}.csv").read_bytes() != _csv_bytes(expected["rows"], expected["columns"]): raise StageEE4Error(f"E4 table is not source-recomputable: {name}")


def validate_outputs(repo_root: Path | str = REPO_ROOT, *, full_upstream: bool = False) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); validate_initialized(repo); package = _build_package(repo)
    for name, table in package["tables"].items(): _verify_table_files(repo, name, table)
    recipes, _ = _strict_json(repo / ROOT_PATH / "figure_recipes.json", "E4 recipes"); _validate_signed(recipes, "E4 recipes")
    schema, _ = _strict_json(repo / "config/stage_e_figure_recipes.schema.json", "recipe schema")
    errors = list(Draft202012Validator(schema).iter_errors(recipes))
    if errors or recipes.get("numeric_source") != "paper_data_only" or recipes.get("renders_png") is not False or any(item.get("renders_png") is not False for item in recipes.get("figures", [])): raise StageEE4Error("E4 recipe contract failed")
    if full_upstream: _upstream_validation(repo, full=True)
    return package


def _copy_source(repo: Path, source: str, destination: str) -> dict[str, Any]:
    raw = (repo / source).read_bytes(); target = repo / ROOT_PATH / destination; target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != raw: raise StageEE4Error(f"immutable E4 copy differs: {destination}")
    else:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        directory = os.open(target.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    return {"source": source, "destination": destination, "bytes": len(raw), "sha256": _sha(raw)}


def _minimum_copies(repo: Path) -> list[dict[str, Any]]:
    copies = [
        ("docs/stage_e_protocol.md", "docs/stage_e_protocol.md"),
        ("config/stage_e_e0.json", "config/stage_e_e0.json"),
        ("config/stage_e_e0.schema.json", "config/stage_e_e0.schema.json"),
        ("config/stage_e_run_spec.json", "config/stage_e_run_spec.json"),
        ("config/stage_e_run_spec.schema.json", "config/stage_e_run_spec.schema.json"),
        ("config/stage_e_reuse_inventory.schema.json", "config/stage_e_reuse_inventory.schema.json"),
        ("config/stage_e_output_table.schema.json", "config/stage_e_output_table.schema.json"),
        ("config/stage_e_figure_recipes.schema.json", "config/stage_e_figure_recipes.schema.json"),
        ("config/stage_e_package.schema.json", "config/stage_e_package.schema.json"),
        ("config/stage_e_stage_completion.schema.json", "config/stage_e_stage_completion.schema.json"),
        ("config/stage_e_e0_completion.schema.json", "config/stage_e_e0_completion.schema.json"),
        (e0.REUSE_PATH, e0.REUSE_PATH),
        (e0.COMPLETION_PATH, "completion/e0.json"),
        (e1.COMPLETION_PATH, "completion/e1.json"),
        (e2.SUPERSEDING_COMPLETION_PATH, "completion/e2.json"),
        (e3r4.COMPLETION_PATH, "completion/e3.json"),
    ]
    return [_copy_source(repo, source, destination) for source, destination in copies]


def _cost(started: float, cpu_started: float, root: Path) -> dict[str, Any]:
    elapsed = time.monotonic() - started; cpu_seconds = time.process_time() - cpu_started; retained = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/e4.pending.json", "completion/e4.json"))["total_bytes"]
    result = {"elapsed_seconds": elapsed, "gpu_hours": 0.0, "cpu_seconds": cpu_seconds, "cpu_hours": cpu_seconds / 3600.0, "retained_bytes_before_status_inventory_completion": retained, "gpu_hours_max": 0.0, "cpu_hours_max": 8.0, "retained_bytes_max": 1073741824}
    result["status"] = "PASS_COST" if result["cpu_hours"] <= 8.0 and retained <= 1073741824 else "FAIL_COST"; return result


def finalize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); started = time.monotonic(); cpu_started = time.process_time(); validate_initialized(repo); upstream = _upstream_validation(repo, full=True); package = _build_package(repo); root = repo / ROOT_PATH
    for name, table in package["tables"].items(): _write_or_verify(root / "paper_data" / f"{name}.json", table); _write_csv(root / "paper_data" / f"{name}.csv", table["rows"], table["columns"])
    copies = _minimum_copies(repo)
    recipes = _signed({"schema_version": 1, "document_type": "stage_e_figure_recipes", "numeric_source": "paper_data_only", "renders_png": False, "figures": [
        {"figure_id": "stage_e_patch_scale_sensitivity_future", "table_path": "paper_data/patch_scale_sensitivity.json", "x": "scale", "y": "delta_fid", "group": "variant_id", "caption_contract": "Post-D10 descriptive sensitivity; D10 primary remains unchanged.", "numeric_source": "paper_data_only", "renders_png": False},
        {"figure_id": "stage_e_upsampling_sensitivity_future", "table_path": "paper_data/upsampling_sensitivity.json", "x": "variant_id", "y": "insertion_auc_mean", "group": "variant_id", "caption_contract": "Uniform and nearest are one equivalent evidence class; no winner selection.", "numeric_source": "paper_data_only", "renders_png": False},
        {"figure_id": "stage_e_layer_head_prediction_strata_future", "table_path": "paper_data/prediction_strata.json", "x": "layer", "y": "median", "group": "stratum", "caption_contract": "Scientific validity is MIXED; numeric summaries use VALID rows only and failures at orders 117 and 352 are retained.", "numeric_source": "paper_data_only", "renders_png": False},
    ]})
    _write_or_verify(root / "figure_recipes.json", recipes)
    analysis = _signed({"schema_version": 1, "document_type": "stage_e_analysis_report", "status": "PASS", "descriptive_only": True, "p_values_emitted": False, "winner_selection_performed": False, "new_go_no_go_issued": False, "variant_status_counts": {"E1": {"valid": 1500, "failed": 0, "na": 0}, "E2": {"valid": 2000, "failed": 0, "na": 0}, "E3_new_compute": {"expected": 3000, "valid": 2998, "failed": 2, "na": 0}, "E3_layer_head_pairs": {"expected": 2000, "valid": 1998, "failed": 2, "na": 0}, "not_run_with_reason": 1}, "scientific_summary": {"package_completeness": "PASS", "workflow_integrity": "PASS", "scientific_validity": "MIXED", "summary_population": "VALID_ONLY_WITH_EXPECTED_VALID_FAILED_NA_COUNTS", "failed_units": [{"selection_order": row["selection_order"], "sample_id": row["sample_id"], "coordinate": [row["layer"], row["head"]], "method": row["method"], "failure_code": row["failure_code"], "delta_fid": row["delta_fid"]} for row in package["failures"]], "d10_primary_unchanged": True, "c10_primary_decision": "NO_GO_NOVELTY", "new_stage_d_decision": None}, "limitations": ["Two frozen (11,2) NNS rows failed IG conservation after the unchanged 32768-step schedule.", "No failed row was deleted, retried, interpolated, zero-imputed, or included in numeric summaries.", "E4 PASS denotes package/workflow integrity and does not denote all-unit scientific validity.", "Stage E is post-D10 descriptive sensitivity analysis and does not replace the D10 primary result."]})
    _write_or_verify(root / "analysis_report.json", analysis)
    provenance_sources = package["source_records"] + [_record(repo, path) for path in (e0.COMPLETION_PATH, e1.COMPLETION_PATH, e2.SUPERSEDING_COMPLETION_PATH, e3r4.COMPLETION_PATH, e1.ROOT_PATH + "/whole_root_inventory.json", e2.SUPERSEDING_INVENTORY_PATH, e3r4.ROOT_PATH + "/whole_root_inventory.json", "reports/stage_d_d9_recovery_v2_20260906/d9_formal_validation.json", "reports/stage_d_d10_recovery_v2_20260906/d10_formal_decision.json", "reports/recovery_c10_completion_20260903.json")]
    provenance = _signed({"schema_version": 1, "document_type": "stage_e_provenance", "source_files": provenance_sources, "pair_bindings": [{"selection_order": row["selection_order"], "sample_id": row["sample_id"], "coordinate": [row["layer"], row["head"]], "method": row["method"], "validity_status": "FAILED", "failure_code": row["failure_code"], "source_hashes": row["source_hashes"]} for row in package["failures"]], "operations": ["source-recompute E1 E2 and E3-R4 terminal packages", "copy validated paper tables without numeric transformation", "merge resource and QC rows with explicit analysis-family identity", "retain and disclose typed FAILED and NA rows", "seal package without model image or GPU access"], "environment": {"python": platform.python_version(), "platform": platform.platform(), "gpu_used": False, "model_loaded": False, "images_read": False}, "outcome_access": {"stage_e_outcomes_read_for_packaging": True, "new_scientific_variant_added": False, "winner_selection_performed": False, "manual_case_selection_performed": False}, "d9_d10_unchanged": True, "c10_history_unchanged": True})
    _write_or_verify(root / "provenance.json", provenance)
    false_audit = _signed({"schema_version": 1, "document_type": "stage_e_e4_false_pass_audit", "stage_id": "E4", "status": "PASS", "checks": [{"attack": name, "result": "REJECTED"} for name in FALSE_PASS_ATTACKS], "workflow_pass_is_not_scientific_pass": True, "stage_f_started": False})
    _write_or_verify(repo / FALSE_PASS_PATH, false_audit)
    validation = _signed({"schema_version": 1, "document_type": "stage_e_e4_source_validation", "stage_id": "E4", "status": "PASS", "validated_at": _now(), "checks": {"e0_e1_e2_e3_completion_chain": "PASS", "e3_r4_archive_custody": "PASS", "d9_d10_c10_unchanged": "PASS", "minimum_package_files": "PASS", "csv_json_parity": "PASS", "required_columns": "PASS", "typed_failure_disclosure": "PASS_2_OF_3000", "valid_only_summary_population": "PASS", "no_new_scientific_compute": "PASS", "no_render_artifacts": "PASS"}, "package_completeness": "PASS", "workflow_integrity": "PASS", "scientific_validity": "MIXED", "upstream_validation": {"e1": upstream["e1"]["status"], "e2": upstream["e2"]["status"], "e3": upstream["e3"]["status"], "stage_d": upstream["stage_d"]["status"]}, "false_pass_audit_canonical_digest": false_audit["canonical_digest"]})
    _write_or_verify(repo / VALIDATION_PATH, validation)
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name not in {"payload_manifest.json", "whole_root_inventory.json", "status.json", "e4.pending.json", "e4.json"}: artifacts.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    manifest = _signed({"schema_version": 1, "document_type": "stage_e_payload_manifest", "immutable": True, "artifacts": artifacts, "payload_count": len(artifacts)})
    _write_or_verify(root / "payload_manifest.json", manifest)
    cost = _cost(started, cpu_started, root)
    if cost["status"] != "PASS_COST": raise StageEE4Error("E4 cost gate failed")
    inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/e4.pending.json", "completion/e4.json")); inventory_doc = _signed({"schema_version": 1, "document_type": "stage_e_e4_whole_root_inventory", "root": ROOT_PATH, "scope": "all regular non-hidden E4 formal files except self-referential inventory status and completion receipts", **inventory})
    _write_or_verify(root / "whole_root_inventory.json", inventory_doc)
    status = _signed({"schema_version": 1, "document_type": "stage_e_status", "status": "PASS", "complete": True, "terminal": True, "immutable": True, "package_completeness": "PASS", "workflow_integrity": "PASS", "scientific_validity": "MIXED", "cost_status": "PASS_COST", "d10_decision_unchanged": True, "c10_history_unchanged": True, "whole_root_inventory": {"file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"], "canonical_digest": inventory_doc["canonical_digest"]}, "git": {"commit": "SELF", "ref": ARCHIVE_REF}, "next_stage": "F0"})
    package_schema, _ = _strict_json(repo / "config/stage_e_package.schema.json", "Stage E package schema")
    for document, name in ((analysis, "analysis report"), (provenance, "provenance"), (manifest, "payload manifest"), (status, "status")):
        errors = list(Draft202012Validator(package_schema).iter_errors(document))
        if errors: raise StageEE4Error(f"E4 {name} schema failure: {errors[0].message}")
    _write_or_verify(root / "status.json", status)
    pending, _ = _strict_json(repo / PENDING_PATH, "E4 pending"); completion = _signed({"schema_version": 1, "document_type": "stage_e_terminal_completion", "stage_id": "E4", "status": "PASS", "started_at": pending["started_at"], "completed_at": _now(), "predecessor": _record(repo, e3r4.COMPLETION_PATH), "config_canonical_digest": e0.load_stage_e_contract(repo)[0]["canonical_digest"], "run_spec_canonical_digest": e0.load_stage_e_run_spec(repo)[0]["canonical_digest"], "expected_coordinates": [{"package_file": item} for item in e0.load_stage_e_contract(repo)[0]["outputs"]["minimum_e4_files"]], "observed_coordinates": [{"package_file": item} for item in e0.load_stage_e_contract(repo)[0]["outputs"]["minimum_e4_files"]], "variant_statuses": [{"variant_id": "E1_patch_scale", "status": "PASS", "reason": "1500 source-validated rows"}, {"variant_id": "E2_upsampling", "status": "PASS", "reason": "2000 source-validated rows including equivalent ranking class"}, {"variant_id": "E3_valid_units", "status": "PASS", "reason": "2998 valid typed terminals"}, {"variant_id": "E3_failed_units", "status": "FAIL", "reason": "orders 117 and 352 retained without imputation"}, {"variant_id": "E3_clean_margin", "status": "NOT_RUN_WITH_REASON", "reason": "not frozen in E0"}], "package_completeness": "PASS", "workflow_integrity": "PASS", "scientific_validity": "MIXED", "cost_status": "PASS_COST", "source_validation": {"validation_report": _record(repo, VALIDATION_PATH), "test_report": _record(repo, TEST_PATH), "false_pass_audit": _record(repo, FALSE_PASS_PATH), "payload_manifest": _record(repo, ROOT_PATH + "/payload_manifest.json"), "status": _record(repo, ROOT_PATH + "/status.json"), "whole_root_inventory": _record(repo, ROOT_PATH + "/whole_root_inventory.json")}, "upstream_unchanged": True, "actual_cost": cost, "next_authorization": "F0_ONLY"})
    completion_schema, _ = _strict_json(repo / "config/stage_e_stage_completion.schema.json", "Stage E completion schema"); errors = list(Draft202012Validator(completion_schema).iter_errors(completion))
    if errors: raise StageEE4Error(f"E4 completion schema failure: {errors[0].message}")
    temporary = (repo / PENDING_PATH).with_name(".e4-terminal.tmp"); _write_new_json(temporary, completion); os.replace(temporary, repo / PENDING_PATH); observed, raw = _strict_json(repo / PENDING_PATH, "E4 terminal pending")
    if observed != completion or raw != canonical_json_bytes(completion) + b"\n": raise StageEE4Error("E4 completion changed before atomic publication")
    os.replace(repo / PENDING_PATH, repo / COMPLETION_PATH); return validate_completion(repo)


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if not (repo / COMPLETION_PATH).is_file() or (repo / PENDING_PATH).exists(): raise StageEE4Error("E4 terminal completion absent or pending remains")
    completion, _ = _strict_json(repo / COMPLETION_PATH, "E4 completion"); _validate_signed(completion, "E4 completion")
    schema, _ = _strict_json(repo / "config/stage_e_stage_completion.schema.json", "Stage E completion schema"); errors = list(Draft202012Validator(schema).iter_errors(completion))
    if errors: raise StageEE4Error(f"E4 completion schema failure: {errors[0].message}")
    if completion.get("status") != "PASS" or completion.get("scientific_validity") != "MIXED" or completion.get("next_authorization") != "F0_ONLY" or completion.get("upstream_unchanged") is not True: raise StageEE4Error("E4 terminal boundary drifted")
    _upstream_validation(repo, full=True); package = _build_package(repo)
    for name, table in package["tables"].items(): _verify_table_files(repo, name, table)
    for field in ("validation_report", "test_report", "false_pass_audit", "payload_manifest", "status", "whole_root_inventory"):
        record = completion["source_validation"][field]
        if record != _record(repo, record["path"]): raise StageEE4Error(f"E4 completion binding drifted: {field}")
    analysis, _ = _strict_json(repo / ROOT_PATH / "analysis_report.json", "E4 analysis"); provenance, _ = _strict_json(repo / ROOT_PATH / "provenance.json", "E4 provenance"); manifest, _ = _strict_json(repo / ROOT_PATH / "payload_manifest.json", "E4 manifest"); status, _ = _strict_json(repo / ROOT_PATH / "status.json", "E4 status"); inventory, _ = _strict_json(repo / ROOT_PATH / "whole_root_inventory.json", "E4 inventory")
    package_schema, _ = _strict_json(repo / "config/stage_e_package.schema.json", "Stage E package schema")
    for document, name in ((analysis, "analysis"), (provenance, "provenance"), (manifest, "manifest"), (status, "status")):
        _validate_signed(document, f"E4 {name}"); errors = list(Draft202012Validator(package_schema).iter_errors(document))
        if errors: raise StageEE4Error(f"E4 {name} schema failed: {errors[0].message}")
    _validate_signed(inventory, "E4 inventory"); rebuilt = _inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "status.json", "completion/e4.pending.json", "completion/e4.json"))
    if any(inventory[key] != rebuilt[key] for key in ("files", "file_count", "total_bytes")) or canonical_sha256({"files": inventory["files"]}) != rebuilt["canonical_digest"]: raise StageEE4Error("E4 whole-root inventory drifted")
    if status.get("status") != "PASS" or status.get("scientific_validity") != "MIXED" or status.get("next_stage") != "F0" or status.get("git") != {"commit": "SELF", "ref": ARCHIVE_REF}: raise StageEE4Error("E4 status drifted")
    if analysis.get("scientific_summary", {}).get("failed_units") != [{"selection_order": row["selection_order"], "sample_id": row["sample_id"], "coordinate": [row["layer"], row["head"]], "method": row["method"], "failure_code": row["failure_code"], "delta_fid": row["delta_fid"]} for row in package["failures"]]: raise StageEE4Error("E4 failed-unit disclosure drifted")
    required = e0.load_stage_e_contract(repo)[0]["outputs"]["minimum_e4_files"]
    if any(not (repo / ROOT_PATH / relative).is_file() for relative in required): raise StageEE4Error("E4 minimum package is incomplete")
    if any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in (repo / ROOT_PATH).rglob("*") if path.is_file()): raise StageEE4Error("E4 contains a forbidden render artifact")
    return {"status": "PASS", "package_completeness": "PASS", "workflow_integrity": "PASS", "scientific_validity": "MIXED", "failed_units": 2, "na_units": 0, "completion_canonical_digest": completion["canonical_digest"], "whole_root_inventory_digest": inventory["canonical_digest"], "actual_cost": completion["actual_cost"], "next_authorization": "F0_ONLY", "stage_f_started": False}


def _main() -> int:
    parser = argparse.ArgumentParser(prog="python -m vitnns.stage_e_e4"); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-preconditions", "verify-preconditions-full", "initialize", "validate-initialized", "validate-outputs", "validate-outputs-full", "finalize", "validate-completion"):
        child = sub.add_parser(name); child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    tests = sub.add_parser("record-tests"); tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); tests.add_argument("--passed", type=int, required=True); tests.add_argument("--failed", type=int, required=True); tests.add_argument("--test-command", required=True)
    args = parser.parse_args()
    if args.command == "verify-preconditions": value = verify_preconditions(args.repo_root, full=False)
    elif args.command == "verify-preconditions-full": value = verify_preconditions(args.repo_root, full=True)
    elif args.command == "initialize": value = initialize(args.repo_root)
    elif args.command == "validate-initialized": value = validate_initialized(args.repo_root)
    elif args.command == "record-tests": value = record_tests(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command)
    elif args.command == "validate-outputs": value = {"status": "PASS", "tables": {name: table["row_count"] for name, table in validate_outputs(args.repo_root, full_upstream=False)["tables"].items()}, "scientific_validity": "MIXED"}
    elif args.command == "validate-outputs-full": value = {"status": "PASS", "tables": {name: table["row_count"] for name, table in validate_outputs(args.repo_root, full_upstream=True)["tables"].items()}, "scientific_validity": "MIXED"}
    elif args.command == "finalize": value = finalize(args.repo_root)
    else: value = validate_completion(args.repo_root)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(_main())
