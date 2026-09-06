"""D9 formal-root sealing report for the recovery-v2 Stage D campaign.

This module is a CPU-only coordinator helper.  It consumes the D8 completion
record and the D9 finalizer output, then writes a small tracked report that
binds the local sealed evidence root without copying its ignored payloads.
It does not run D10 aggregation, bootstrap, decision, paper export, model
construction, CUDA, or image/JPEG access.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .constants import REPO_ROOT
from .stage_d_config import strict_json_bytes
from .stage_d_d8_completion import (
    CAMPAIGN_COMPLETION_PATH,
    COMPLETION_REF as D8_COMPLETION_REF,
    validate_d8_campaign_completion,
)
from .stage_d_finalize import FINAL_FILES, StageDRootValidation
from .stage_d_plan import canonical_json_bytes


D9_ID = "STAGE_D_RECOVERY_V2_D9_20260906"
DOCUMENT_TYPE = "stage_d_d9_formal_validation"
REPORT_ROOT = "reports/stage_d_d9_recovery_v2_20260906"
REPORT_PATH = f"{REPORT_ROOT}/d9_formal_validation.json"
COMPLETION_REF = "refs/archive/stage-d-fidelity-v1-d9-completion"
PRODUCER_REF = "refs/archive/stage-d-fidelity-v1-d9-producer"
D8_COMPLETION_COMMIT = "fafcae85e94137933fbacf3b3819cdc461f23855"
D8_S07_COMPLETION_COMMIT = "1474b103124c5f68187960e305410e114718a350"
D8_S07_COMPLETION_REF = "refs/archive/stage-d-fidelity-v1-d8-s07-completion"
FORMAL_ROOT = "evidence/stage_d/d8_recovery_v2_20260904/formal_r2"
BLOCKED_ROOT = "evidence/stage_d/d8_recovery_v2_20260904/formal"
C10_STATUS_PRESERVED = {
    "acceptance": "BLOCKED",
    "primary_decision": "NO_GO_NOVELTY",
    "overall_spatial": "FAIL_SPATIAL",
    "holm": "HOLM_BLOCKED_UNDERSPECIFIED",
    "downstream_authorized": False,
}


class StageDD9CompletionError(ValueError):
    """D9 cannot publish a complete formal validation report."""


def _plain(value: Any, context: str = "value") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise StageDD9CompletionError(f"{context} has a non-string key")
            result[key] = _plain(child, f"{context}.{key}")
        return result
    if isinstance(value, tuple):
        return [_plain(child, context) for child in value]
    if isinstance(value, list):
        return [_plain(child, context) for child in value]
    return value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _semantic_digest(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes({
            key: value for key, value in document.items()
            if key != "canonical_digest"
        })
    ).hexdigest()


def canonical_file_bytes(document: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(document) + b"\n"


def _read_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise StageDD9CompletionError(f"cannot read {context}: {path}") from error
    try:
        value = strict_json_bytes(raw, context)
    except Exception as error:
        raise StageDD9CompletionError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise StageDD9CompletionError(f"{context} must be a JSON object")
    if raw != canonical_file_bytes(value):
        raise StageDD9CompletionError(f"{context} is not canonical JSON")
    return value, raw


def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *args), cwd=repo, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        raise StageDD9CompletionError(
            f"git {' '.join(args)} failed: {error.stderr.strip()}"
        ) from error
    return completed.stdout.strip()


def _require_clean_git_state(repo: Path) -> dict[str, Any]:
    branch = _git(repo, "symbolic-ref", "--short", "HEAD")
    head = _git(repo, "rev-parse", "HEAD")
    upstream = _git(repo, "rev-parse", "@{u}^{commit}")
    archive = _git(repo, "rev-parse", f"{D8_COMPLETION_REF}^{{commit}}")
    s07 = _git(repo, "rev-parse", f"{D8_S07_COMPLETION_REF}^{{commit}}")
    remote_lines = _git(repo, "ls-remote", "--heads", "origin", "stage-d-d3-pass").splitlines()
    remote = remote_lines[0].split()[0] if len(remote_lines) == 1 else None
    status = _git(repo, "status", "--branch", "--porcelain=v2")
    lfs_status = _git(repo, "lfs", "status")
    ancestor_ok = subprocess.run(
        ("git", "merge-base", "--is-ancestor", D8_COMPLETION_COMMIT, head),
        cwd=repo,
    ).returncode == 0
    if (
        branch != "stage-d-d3-pass"
        or not ancestor_ok
        or upstream != D8_COMPLETION_COMMIT
        or archive != D8_COMPLETION_COMMIT
        or remote != D8_COMPLETION_COMMIT
        or s07 != D8_S07_COMPLETION_COMMIT
        or any(line and not line.startswith("# ") for line in status.splitlines())
        or "Git LFS objects to be pushed" not in lfs_status
        or "Git LFS objects to be committed" not in lfs_status
        or "Git LFS objects not staged for commit" not in lfs_status
        or any(
            line.strip().startswith(("*", "A ", "M ", "D ", "R ", "C ", "?"))
            for line in lfs_status.splitlines()
        )
    ):
        raise StageDD9CompletionError("D9 start Git/LFS state is not clean and authorized")
    return {
        "branch": branch,
        "d9_producer_head": head,
        "d9_producer_ref": PRODUCER_REF,
        "local_head_is_descendant_of_d8_completion": ancestor_ok,
        "tracking_ref": "origin/stage-d-d3-pass",
        "tracking_sha_before_d9_completion": upstream,
        "remote_ref": "refs/heads/stage-d-d3-pass",
        "remote_sha_before_d9_completion": remote,
        "d8_completion_ref": D8_COMPLETION_REF,
        "d8_completion_ref_sha": archive,
        "d8_s07_completion_ref": D8_S07_COMPLETION_REF,
        "d8_s07_completion_ref_sha": s07,
        "worktree_index_clean": True,
        "status_porcelain_v2": status.splitlines(),
        "lfs_status_clean": True,
    }


def _final_file_records(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name in FINAL_FILES:
        path = root / "final" / name
        document, raw = _read_json(path, f"D9 final {name}")
        rows[name] = {
            "path": f"{FORMAL_ROOT}/final/{name}",
            "file_sha256": _sha(raw),
            "canonical_digest": document["canonical_digest"],
        }
    return rows


def _analysis_path(scientific_validity: str) -> str:
    return (
        "D10_FULL_ANALYSIS_AND_DECISION"
        if scientific_validity == "PASS"
        else "D10_NO_GO_VALIDITY_ONLY"
    )


def _d10_prompt(scientific_validity: str) -> str:
    if scientific_validity == "PASS":
        return (
            "Run Stage D D10 only after revalidating "
            f"`{REPORT_PATH}` and the sealed root `{FORMAL_ROOT}`. "
            "D10 may read validated sealed L0/L1 numeric evidence and D9 QC/cost "
            "records, then compute the pre-registered aggregate statistics, "
            "fidelity decision, and allowed machine-readable paper data. Do not "
            "read JPEGs, load a model, use GPU/CUDA, modify the D9 root, or "
            "generate qualitative overlays/contact sheets/Stage F display material."
        )
    return (
        "Do not run D10 aggregate/statistics. Diagnose the D9 scientific "
        f"validity failures in `{REPORT_PATH}` and the sealed root `{FORMAL_ROOT}`; "
        "preserve all evidence in place and do not rerun D8 unless a new formal "
        "repair protocol is approved."
    )


def build_d9_formal_validation_report(
    validation: StageDRootValidation | Mapping[str, Any],
    *, repo_root: Path | str = REPO_ROOT,
    git_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    validated = _plain(validation.to_dict() if hasattr(validation, "to_dict") else validation)
    if not isinstance(validated, dict):
        raise StageDD9CompletionError("D9 validation result must be a mapping")
    evidence = validated.get("evidence_report")
    reconciliation = validated.get("reconciliation")
    source_inventory = validated.get("source_inventory")
    if not isinstance(evidence, Mapping) or not isinstance(reconciliation, Mapping) or not isinstance(source_inventory, Mapping):
        raise StageDD9CompletionError("D9 validation result omits required final documents")
    if (
        validated.get("package_complete") is not True
        or validated.get("workflow_status") != "PASS"
        or evidence.get("authorization", {}).get("formal") is not True
        or evidence.get("analysis_authorized") is not True
        or evidence.get("analysis_performed") is not False
        or evidence.get("bootstrap_performed") is not False
        or evidence.get("decision_performed") is not False
    ):
        raise StageDD9CompletionError("D9 validation result is not a sealed formal root")
    cost = evidence.get("actual_cost_audit")
    if not isinstance(cost, Mapping) or cost.get("status") not in {"PASS_COST", "FAIL_COST"}:
        raise StageDD9CompletionError("D9 cost audit is missing or invalid")
    d8 = validate_d8_campaign_completion(repo_root=repo)
    final_root = repo / FORMAL_ROOT
    blocked_root = repo / BLOCKED_ROOT
    if not blocked_root.is_dir():
        raise StageDD9CompletionError("initial blocked root is not retained")
    final_records = _final_file_records(final_root)
    git_record = dict(git_state) if git_state is not None else _require_clean_git_state(repo)
    document: dict[str, Any] = {
        "schema_version": 1,
        "document_type": DOCUMENT_TYPE,
        "d9_id": D9_ID,
        "status": "PASS",
        "formal_root": str(final_root),
        "relative_formal_root": FORMAL_ROOT,
        "retained_blocked_root": {
            "path": str(blocked_root),
            "relative_path": BLOCKED_ROOT,
            "retained": True,
            "reused": False,
            "modified_by_d9": False,
        },
        "d8_completion": {
            "path": CAMPAIGN_COMPLETION_PATH,
            "file_sha256": _sha((repo / CAMPAIGN_COMPLETION_PATH).read_bytes()),
            "canonical_digest": d8["canonical_digest"],
            "archive_ref": D8_COMPLETION_REF,
            "commit": D8_COMPLETION_COMMIT,
            "final_checkpoint_generation": d8["final_checkpoint"]["generation"],
            "final_checkpoint_canonical_digest": d8["final_checkpoint"]["canonical_digest"],
        },
        "final_files": final_records,
        "validation": {
            "package_complete": True,
            "workflow_status": "PASS",
            "scientific_validity": validated["scientific_validity"],
            "analysis_authorized": validated["analysis_authorized"],
            "analysis_authorization": evidence["authorization"]["analysis_authorization"],
            "d10_allowed_path": _analysis_path(validated["scientific_validity"]),
            "final_status_sha256": validated["final_status_sha256"],
            "root_inventory": validated["root_inventory"],
            "reconciliation_digest": reconciliation["canonical_digest"],
            "source_inventory_digest": source_inventory["canonical_digest"],
            "payload_manifest_digest": validated["payload_manifest"]["canonical_digest"],
        },
        "qc": {
            "expected_image_groups": 500,
            "observed_image_groups": reconciliation["image_count"],
            "expected_units": 1000,
            "observed_units": reconciliation["unit_count"],
            "expected_counts": reconciliation["expected_counts"],
            "observed_counts": reconciliation["observed_counts"],
            "state_counts": reconciliation["state_counts"],
            "na_rows_count": len(reconciliation["na_rows"]),
            "scientific_failure_rows_count": len(evidence["scientific_failures"]),
            "failure_rows_count": len(reconciliation["failure_rows"]),
            "no_rows_dropped": reconciliation["no_rows_dropped"],
            "zero_imputation_performed": reconciliation["zero_imputation_performed"],
            "checkpoint_failure_table_reconciled": reconciliation[
                "checkpoint_failure_table_reconciled"
            ],
            "checkpoint_resources_reconciled": reconciliation[
                "checkpoint_resources_reconciled"
            ],
        },
        "retention": reconciliation["retention"],
        "actual_cost_audit": cost,
        "scope_boundaries": {
            "cpu_only": True,
            "jpeg_read": False,
            "model_constructed_or_loaded": False,
            "cuda_or_gpu_used": False,
            "d8_shards_run": False,
            "retry_attempts_created": False,
            "d10_aggregate_statistics_decision_run": False,
            "delta_fid_aggregate_computed": False,
            "median_positive_count_or_bootstrap_computed": False,
            "paper_data_figures_overlays_or_contact_sheets_generated": False,
        },
        "state_separation": {
            "package_completeness": "PASS",
            "workflow_integrity": "PASS",
            "scientific_validity": validated["scientific_validity"],
            "cost_status": cost["status"],
        },
        "c10_status_preserved": C10_STATUS_PRESERVED,
        "analysis_authorized": validated["analysis_authorized"],
        "next_task": "D10" if validated["analysis_authorized"] else "D9_DIAGNOSIS",
        "next_prompt": _d10_prompt(validated["scientific_validity"]),
        "git": git_record,
    }
    document["canonical_digest"] = _semantic_digest(document)
    return MappingProxyType(document)


def write_d9_formal_validation_report(
    validation: StageDRootValidation | Mapping[str, Any],
    *, repo_root: Path | str = REPO_ROOT,
    path: Path | str = REPORT_PATH,
    git_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document = build_d9_formal_validation_report(
        validation, repo_root=repo, git_state=git_state,
    )
    target = repo / os.fspath(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise StageDD9CompletionError(f"refusing to overwrite D9 report: {target}")
    target.write_bytes(canonical_file_bytes(document))
    return document


def validate_d9_formal_validation_report(
    *, repo_root: Path | str = REPO_ROOT,
    path: Path | str = REPORT_PATH,
    validation: StageDRootValidation | Mapping[str, Any] | None = None,
    git_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    relative = os.fspath(path)
    document, raw = _read_json(repo / relative, "D9 formal validation report")
    if raw != canonical_file_bytes(document):
        raise StageDD9CompletionError("D9 report bytes are not canonical")
    if document.get("canonical_digest") != _semantic_digest(document):
        raise StageDD9CompletionError("D9 report self digest mismatch")
    if validation is None:
        expected = dict(document)
    else:
        if git_state is None:
            git_state = document.get("git")
        expected = build_d9_formal_validation_report(
            validation, repo_root=repo, git_state=git_state,
        )
    if document != expected:
        raise StageDD9CompletionError("D9 report is not source-recomputable")
    return MappingProxyType(document)


__all__ = [
    "COMPLETION_REF",
    "D9_ID",
    "PRODUCER_REF",
    "REPORT_PATH",
    "StageDD9CompletionError",
    "build_d9_formal_validation_report",
    "canonical_file_bytes",
    "validate_d9_formal_validation_report",
    "write_d9_formal_validation_report",
]
