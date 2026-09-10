"""Source-recomputed post-failure custody for the immutable X2 attempt."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from .stage_f_f3c1_kernels import array_digest

ROOT = Path("evidence/stage_x/x2_nns_v2_development_20260909/formal")
REPORT_ROOT = Path("reports/stage_x_x2_nns_v2_development_20260909")
RECEIPT = ROOT / "post_failure/x2_no_go_method_validity.json"
REPORT = REPORT_ROOT / "x2_no_go_method_validity_report.md"
SCHEMA = Path("config/stage_x_x2_failure_receipt.schema.json")


class X2FailureError(ValueError):
    pass


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in rows:
        if key in value:
            raise X2FailureError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            X2FailureError(f"non-finite JSON: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise X2FailureError(f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(repo: Path, relative: Path) -> dict[str, Any]:
    path = repo / relative
    return {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_self(value: Mapping[str, Any], context: str) -> None:
    plain = dict(value)
    observed = plain.pop("canonical_digest", None)
    if observed != canonical_sha256(plain):
        raise X2FailureError(f"canonical digest mismatch: {context}")


def _existing_inventory(repo: Path) -> tuple[list[dict[str, Any]], str]:
    base = repo / ROOT
    files = []
    for path in sorted(item for item in base.rglob("*") if item.is_file() and item != repo / RECEIPT):
        relative = path.relative_to(base).as_posix()
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files, canonical_sha256(files)


def recompute(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = repo / ROOT
    sample_path = root / "development_sample_manifest.json"
    unit_path = root / "development_unit_manifest.json"
    samples_doc, units_doc = read(sample_path), read(unit_path)
    validate_self(samples_doc, sample_path.as_posix())
    validate_self(units_doc, unit_path.as_posix())
    samples = samples_doc["payload"]["samples"]
    units = units_doc["payload"]["units"]
    if len(samples) != len({row["source_object_id"] for row in samples}) or len(samples) != 40:
        raise X2FailureError("source sample manifest is not exactly 40 unique IDs")
    if len(units) != len({row["unit_digest"] for row in units}) or len(units) != 200:
        raise X2FailureError("source unit manifest is not exactly 200 unique routes")

    diagnostic: list[dict[str, Any]] = []
    published: list[str] = []
    partial: list[str] = []
    pending: list[str] = []
    for unit in units:
        identity = {key: unit[key] for key in ("model_id", "sample_id", "sample_sha256", "development_rank")}
        if unit["unit_digest"] != canonical_sha256(identity):
            raise X2FailureError(f"unit digest drift: {unit['unit_digest']}")
        route_root = root / "units" / unit["unit_digest"]
        clean = route_root / "attempts/attempt-000001/clean"
        terminal = route_root / "attempts/attempt-000001/terminal"
        if terminal.exists():
            published.append(unit["unit_digest"])
        if route_root.exists() and any("pending" in path.name for path in route_root.rglob("*")):
            pending.append(unit["unit_digest"])
        if clean.is_dir():
            result_path, manifest_path = clean / "result.json", clean / "manifest.json"
            result, manifest = read(result_path), read(manifest_path)
            validate_self(result, result_path.as_posix())
            validate_self(manifest, manifest_path.as_posix())
            if result["unit_digest"] != unit["unit_digest"] or manifest["unit_digest"] != unit["unit_digest"]:
                raise X2FailureError("diagnostic unit identity drift")
            for record in manifest["arrays"]:
                array_path = clean / record["path"]
                value = np.load(array_path, allow_pickle=False)
                observed = {
                    "path": record["path"], "bytes": array_path.stat().st_size,
                    "sha256": sha256_file(array_path), "dtype": str(value.dtype),
                    "shape": list(value.shape), "array_digest": array_digest(value),
                }
                if observed != record:
                    raise X2FailureError(f"diagnostic array drift: {array_path}")
            files = sorted(path for path in clean.rglob("*") if path.is_file())
            tree_rows = [
                (path.relative_to(clean).as_posix(), path.stat().st_size, sha256_file(path))
                for path in files
            ]
            diagnostic.append({
                **identity, "unit_digest": unit["unit_digest"], "shard": unit["shard"],
                "file_count": len(files), "tree_digest": canonical_sha256(tree_rows),
                "state": "DIAGNOSTIC_ONLY_INVALIDATED_BY_GLOBAL_METHOD_VALIDITY_STOP",
            })
        elif route_root.exists() and not terminal.exists():
            partial.append(unit["unit_digest"])

    completed = {row["unit_digest"] for row in diagnostic}
    interrupted = []
    for shard in range(4):
        rows = [row for row in units if row["shard"] == shard]
        unfinished = next(row for row in rows if row["unit_digest"] not in completed)
        interrupted.append({key: unfinished[key] for key in ("unit_digest", "model_id", "sample_id", "sample_sha256", "development_rank", "shard")})
    files, inventory_digest = _existing_inventory(repo)
    if (len(diagnostic), len(published), len(partial), len(pending), len(files)) != (11, 0, 0, 0, 331):
        raise X2FailureError("original X2 source counts differ from the frozen failure facts")
    return {
        "$schema": SCHEMA.name, "schema_version": 1,
        "document_type": "stage_x_x2_post_failure_terminal_receipt",
        "stage_id": "X2", "attempt_id": "STAGE_X_X2_V1_20260909",
        "status": "NO_GO_METHOD_VALIDITY",
        "failure_code": "X2_ARTIFACT_PAYLOAD_SCHEMA_NOT_CLOSED",
        "source_root": ROOT.as_posix(), "report_root": REPORT_ROOT.as_posix(),
        "manifest_sources": {
            "sample_manifest": file_record(repo, ROOT / "development_sample_manifest.json"),
            "unit_manifest": file_record(repo, ROOT / "development_unit_manifest.json"),
            "open_artifact_schema": file_record(repo, Path("config/stage_x_x2_artifact.schema.json")),
        },
        "counts": {
            "development_images": 40, "expected_routes": 200,
            "complete_diagnostic_routes": 11, "interrupted_attempts": 4,
            "not_started_total": 189, "never_started": 185,
            "published_unit_terminals": 0,
            "existing_file_count_before_receipt": len(files),
            "existing_bytes_before_receipt": sum(row["bytes"] for row in files),
        },
        "diagnostic_routes": sorted(diagnostic, key=lambda row: row["unit_digest"]),
        "interrupted_routes": sorted(interrupted, key=lambda row: row["shard"]),
        "interruption_evidence": {
            "captured_process_count": 4,
            "pids": [3246053, 3246054, 3246055, 3246058],
            "commands": [
                f"../.conda/envs/vitnns/bin/python -u -m vitnns.stage_x_x2_worker run-clean --repo-root . --shard {index} --device cuda:{index}"
                for index in range(4)
            ],
            "derivation": "one live worker per frozen shard at global stop; interrupted identity is the first manifest-ordered route in that shard without a clean or terminal publication",
        },
        "states": {
            "diagnostic_route_state": "DIAGNOSTIC_ONLY_INVALIDATED_BY_GLOBAL_METHOD_VALIDITY_STOP",
            "interrupted_route_state": "NOT_STARTED_DUE_TO_GLOBAL_METHOD_VALIDITY_STOP_INTERRUPTED_IN_MEMORY",
            "remaining_route_state": "NOT_STARTED_DUE_TO_GLOBAL_METHOD_VALIDITY_STOP_NEVER_STARTED",
            "unexecuted_unit_terminals_created": False,
        },
        "scientific_seals": {"gamma_selected": False, "margins_sealed": False, "futility_computed": False, "stability_completed": False, "formal_budget_projected": False},
        "custody": {"existing_tree_inventory_digest": inventory_digest, "existing_files_unchanged": True, "partial_atomic_directories": 0, "pending_publications": 0, "source_recomputed": True},
        "authorization": {"next_authorization": "X2_R1_NNS_V2_DEVELOPMENT_AND_FUTILITY_RECOVERY_ONLY", "source": "OWNER_EXPLICIT_20260909", "created_by_failed_x2": False},
        "downstream": {"x3_started": False, "x4_started": False, "x5_started": False},
    }


def validate(value: Mapping[str, Any], repo: Path) -> None:
    schema = read(repo / SCHEMA)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: list(error.absolute_path))
    if errors:
        raise X2FailureError(errors[0].message)
    validate_self(value, "X2 failure receipt")


def seal(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    expected = recompute(repo)
    expected["canonical_digest"] = canonical_sha256(expected)
    path = repo / RECEIPT
    if path.exists():
        observed = read(path)
        validate(observed, repo)
        if observed != expected:
            raise X2FailureError("existing X2 failure receipt does not source-recompute")
        return observed
    validate(expected, repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(expected) + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    report = (
        "# Stage X X2 failed-attempt custody\n\n"
        "Terminal: `NO_GO_METHOD_VALIDITY` / "
        "`X2_ARTIFACT_PAYLOAD_SCHEMA_NOT_CLOSED`. Eleven completed routes are "
        "diagnostic-only; four routes were interrupted in memory; 185 never "
        "started; zero unit terminals were published. No X2 outcome is eligible "
        "for X2-R1 selection or analysis. X3-X5 were not started.\n"
    ).encode("utf-8")
    report_path = repo / REPORT
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(report); stream.flush(); os.fsync(stream.fileno())
    return expected
