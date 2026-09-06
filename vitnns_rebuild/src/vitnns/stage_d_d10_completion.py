"""D10 formal analysis completion report for Stage D recovery-v2.

This module is a CPU-only postprocessing helper.  It consumes an already
validated D10 analysis package and paper package, plus the frozen D9 validation
report.  It never opens JPEGs, models, CUDA state, or the D8/D9 execution root.
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
from .stage_d_d9_completion import (
    C10_STATUS_PRESERVED,
    COMPLETION_REF as D9_COMPLETION_REF,
    D9_ID,
    D8_COMPLETION_COMMIT,
    D8_COMPLETION_REF,
    PRODUCER_REF as D9_PRODUCER_REF,
    REPORT_PATH as D9_REPORT_PATH,
)
from .stage_d_paper import (
    ALL_FILES,
    analysis_control_from_package,
    validate_stage_d_analysis_package,
    validate_stage_d_paper_package,
)
from .stage_d_provenance import canonical_json_bytes, canonical_sha256


D10_ID = "STAGE_D_RECOVERY_V2_D10_20260906"
DOCUMENT_TYPE = "stage_d_d10_formal_decision"
REPORT_ROOT = "reports/stage_d_d10_recovery_v2_20260906"
REPORT_PATH = f"{REPORT_ROOT}/d10_formal_decision.json"
ANALYSIS_ROOT = "evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis"
PAPER_ROOT = "evidence/stage_d/d10_recovery_v2_20260906/formal_r2/paper"
COMPLETION_REF = "refs/archive/stage-d-fidelity-v1-d10-completion"
D9_COMPLETION_REPORT_SHA256 = (
    "bbfb6b5ad879116f6849e6f0a058780fdfd5e54b656e62f0674a3c3184c810d5"
)
D9_COMPLETION_REPORT_CANONICAL_DIGEST = (
    "50e2c1455534e5b10aeacb79fac42d128317e0e943c817addf47671f599ea206"
)
D9_SEALED_ROOT_INVENTORY_DIGEST = (
    "76d7d532049a5c90a09c74ba56b019042d7f2f0174ccb439bd8f67713514cc5a"
)
D9_PRODUCER_COMMIT = "33be20d13e0848ed10c0e9875cfa23fe2fa2bd7c"


class StageDD10CompletionError(ValueError):
    """D10 cannot publish a complete formal decision report."""


def _plain(value: Any, context: str = "value") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise StageDD10CompletionError(f"{context} has a non-string key")
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
        raise StageDD10CompletionError(f"cannot read {context}: {path}") from error
    try:
        value = strict_json_bytes(raw, context)
    except Exception as error:
        raise StageDD10CompletionError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise StageDD10CompletionError(f"{context} must be a JSON object")
    if raw != canonical_file_bytes(value):
        raise StageDD10CompletionError(f"{context} is not canonical JSON")
    return value, raw


def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *args), cwd=repo, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        raise StageDD10CompletionError(
            f"git {' '.join(args)} failed: {error.stderr.strip()}"
        ) from error
    return completed.stdout.strip()


def _lfs_clean(status: str) -> bool:
    if (
        "Git LFS objects to be pushed" not in status
        or "Git LFS objects to be committed" not in status
        or "Git LFS objects not staged for commit" not in status
    ):
        return False
    return not any(
        line.strip().startswith(("*", "A ", "M ", "D ", "R ", "C ", "?"))
        for line in status.splitlines()
    )


def _git_state(repo: Path) -> dict[str, Any]:
    branch = _git(repo, "symbolic-ref", "--short", "HEAD")
    head = _git(repo, "rev-parse", "HEAD")
    upstream = _git(repo, "rev-parse", "@{u}^{commit}")
    d9_completion = _git(repo, "rev-parse", f"{D9_COMPLETION_REF}^{{commit}}")
    d8_completion = _git(repo, "rev-parse", f"{D8_COMPLETION_REF}^{{commit}}")
    d9_producer = _git(repo, "rev-parse", f"{D9_PRODUCER_REF}^{{commit}}")
    remote_lines = _git(
        repo, "ls-remote", "--heads", "origin", "stage-d-d3-pass"
    ).splitlines()
    remote = remote_lines[0].split()[0] if len(remote_lines) == 1 else None
    status = _git(repo, "status", "--branch", "--porcelain=v2")
    lfs_status = _git(repo, "lfs", "status")
    clean = (
        branch == "stage-d-d3-pass"
        and head == d9_completion
        and upstream == d9_completion
        and remote == d9_completion
        and d8_completion == D8_COMPLETION_COMMIT
        and d9_producer == D9_PRODUCER_COMMIT
        and not any(line and not line.startswith("# ") for line in status.splitlines())
        and _lfs_clean(lfs_status)
    )
    if not clean:
        raise StageDD10CompletionError("D10 start Git/LFS state is not clean and authorized")
    return {
        "branch": branch,
        "local_head_at_start": head,
        "tracking_ref": "origin/stage-d-d3-pass",
        "tracking_sha_at_start": upstream,
        "remote_ref": "refs/heads/stage-d-d3-pass",
        "remote_sha_at_start": remote,
        "d9_completion_ref": D9_COMPLETION_REF,
        "d9_completion_ref_sha": d9_completion,
        "d8_completion_ref": D8_COMPLETION_REF,
        "d8_completion_ref_sha": d8_completion,
        "d9_producer_ref": D9_PRODUCER_REF,
        "d9_producer_ref_sha": d9_producer,
        "worktree_index_clean_at_start": True,
        "status_porcelain_v2_at_start": status.splitlines(),
        "lfs_status_clean_at_start": True,
    }


def _file_record(path: Path, relative: str) -> dict[str, Any]:
    document, raw = _read_json(path, relative)
    return {
        "path": relative,
        "file_sha256": _sha(raw),
        "canonical_digest": document["canonical_digest"],
    }


def _package_records(root: Path, relative_root: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in ALL_FILES:
        if not name.endswith(".json"):
            continue
        records[name] = _file_record(root / name, f"{relative_root}/{name}")
    return records


def _csv_records(root: Path, relative_root: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in ALL_FILES:
        if not name.endswith(".csv"):
            continue
        path = root / name
        raw = path.read_bytes()
        records[name] = {
            "path": f"{relative_root}/{name}",
            "file_sha256": _sha(raw),
            "bytes": len(raw),
        }
    return records


def _require_d9_report(repo: Path) -> Mapping[str, Any]:
    report, raw = _read_json(repo / D9_REPORT_PATH, "D9 formal validation report")
    if (
        _sha(raw) != D9_COMPLETION_REPORT_SHA256
        or report.get("canonical_digest") != D9_COMPLETION_REPORT_CANONICAL_DIGEST
        or _semantic_digest(report) != D9_COMPLETION_REPORT_CANONICAL_DIGEST
        or report.get("status") != "PASS"
        or report.get("analysis_authorized") is not True
        or report.get("c10_status_preserved") != C10_STATUS_PRESERVED
    ):
        raise StageDD10CompletionError("D9 validation report does not match D10 entry requirements")
    validation = report.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("package_complete") is not True
        or validation.get("workflow_status") != "PASS"
        or validation.get("scientific_validity") != "PASS"
        or validation.get("analysis_authorized") is not True
        or validation.get("analysis_authorization") != "D9_AUTHORIZED_D10_FULL_ANALYSIS"
        or validation.get("root_inventory", {}).get("canonical_digest") != D9_SEALED_ROOT_INVENTORY_DIGEST
    ):
        raise StageDD10CompletionError("D9 validation report does not authorize D10")
    return report


def _analysis_summary(analysis: Mapping[str, Any]) -> dict[str, Any]:
    summary = analysis["summary"]
    bootstrap = analysis["bootstrap"]
    if not isinstance(summary, Mapping) or not isinstance(bootstrap, Mapping):
        raise StageDD10CompletionError("D10 analysis lacks summary/bootstrap")
    return {
        "metric_id": summary["metric_id"],
        "n": summary["n"],
        "positive_count": summary["positive_count"],
        "mean": summary["mean"],
        "median": summary["median"],
        "sample_sd_ddof_1": summary["standard_deviation_sample_ddof_1"],
        "bootstrap_ci_95": [
            summary["bootstrap_ci_lower"], summary["bootstrap_ci_upper"],
        ],
        "bootstrap_draw_indices_sha256": bootstrap["draw_indices_sha256"],
        "bootstrap_distribution_sha256": bootstrap["distribution_sha256"],
    }


def build_d10_formal_decision_report(
    *, repo_root: Path | str = REPO_ROOT,
    analysis_root: Path | str = ANALYSIS_ROOT,
    paper_root: Path | str = PAPER_ROOT,
    d9_revalidation: Mapping[str, Any],
    git_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    if not isinstance(d9_revalidation, Mapping):
        raise StageDD10CompletionError("D9 revalidation result must be a mapping")
    if (
        d9_revalidation.get("status") != "PASS"
        or d9_revalidation.get("package_complete") is not True
        or d9_revalidation.get("scientific_validity") != "PASS"
        or d9_revalidation.get("analysis_authorized") is not True
        or d9_revalidation.get("root_inventory", {}).get("canonical_digest")
        != D9_SEALED_ROOT_INVENTORY_DIGEST
    ):
        raise StageDD10CompletionError("D9 revalidation did not authorize D10")
    d9_report = _require_d9_report(repo)
    analysis_path = repo / os.fspath(analysis_root)
    paper_path = repo / os.fspath(paper_root)
    analysis_validation = validate_stage_d_analysis_package(
        analysis_path, repo_root=repo
    )
    paper_validation = validate_stage_d_paper_package(
        paper_path, repo_root=repo
    )
    if (
        analysis_validation.get("formal_result") is not True
        or paper_validation.get("formal_result") is not True
        or analysis_validation.get("analysis_canonical_digest")
        != paper_validation.get("analysis_canonical_digest")
    ):
        raise StageDD10CompletionError("D10 packages are not matching formal packages")
    analysis = analysis_control_from_package(analysis_path, repo_root=repo)
    if (
        analysis.get("validity_status") != "PASS"
        or analysis.get("c10_reference", {}).get("overall_decision") != "NO_GO_NOVELTY"
        or analysis.get("c10_reference", {}).get("workflow_status") != "BLOCKED"
        or analysis.get("c10_reference", {}).get("spatial_status") != "FAIL_SPATIAL"
        or analysis.get("c10_reference", {}).get("holm_status") != "HOLM_BLOCKED_UNDERSPECIFIED"
        or analysis.get("provenance", {}).get("formal_result") is not True
        or analysis.get("provenance", {}).get("jpeg_read") is not False
        or analysis.get("provenance", {}).get("model_loaded") is not False
        or analysis.get("provenance", {}).get("cuda_or_gpu_used") is not False
    ):
        raise StageDD10CompletionError("D10 analysis package violates formal scope")

    document: dict[str, Any] = {
        "schema_version": 1,
        "document_type": DOCUMENT_TYPE,
        "d10_id": D10_ID,
        "status": "PASS",
        "formal_postprocessing_root": str(analysis_path.parent),
        "analysis_root": str(analysis_path),
        "paper_root": str(paper_path),
        "relative_analysis_root": os.fspath(analysis_root),
        "relative_paper_root": os.fspath(paper_root),
        "d9_revalidation": _plain(d9_revalidation),
        "d9_report": {
            "path": D9_REPORT_PATH,
            "file_sha256": D9_COMPLETION_REPORT_SHA256,
            "canonical_digest": D9_COMPLETION_REPORT_CANONICAL_DIGEST,
            "d9_id": D9_ID,
        },
        "d9_sealed_root": {
            "relative_path": d9_report["relative_formal_root"],
            "status_file_sha256": d9_report["final_files"]["status.json"]["file_sha256"],
            "status_canonical_digest": d9_report["final_files"]["status.json"]["canonical_digest"],
            "root_inventory_digest": d9_report["validation"]["root_inventory"]["canonical_digest"],
            "modified_by_d10": False,
        },
        "analysis_package": {
            **_plain(analysis_validation),
            "json_files": _package_records(analysis_path, os.fspath(analysis_root)),
            "csv_files": _csv_records(analysis_path, os.fspath(analysis_root)),
        },
        "paper_package": {
            **_plain(paper_validation),
            "json_files": _package_records(paper_path, os.fspath(paper_root)),
            "csv_files": _csv_records(paper_path, os.fspath(paper_root)),
        },
        "statistics": _analysis_summary(analysis),
        "truth_table": {
            "validity_status": analysis["validity_status"],
            "fidelity_status": analysis["fidelity_status"],
            "cost_status": analysis["cost_status"],
            "overall_decision": analysis["overall_decision"],
            "positive_count_pass_threshold": 400,
            "median_pass_threshold": 0.02,
            "ci_lower_pass_threshold_exclusive": 0,
            "ci_upper_failure_threshold_exclusive": 0.02,
        },
        "paper_artifacts": {
            "machine_readable_only": True,
            "numeric_figures_only": True,
            "qualitative_overlays_generated": False,
            "contact_sheets_generated": False,
            "stage_f_display_material_generated": False,
            "figures_rendered": 0,
            "figure_recipes_path": f"{os.fspath(paper_root)}/figure_recipes.json",
            "paper_data_paths": [
                f"{os.fspath(paper_root)}/paper_data/{name}"
                for name in (
                    "main_fidelity_table.csv",
                    "main_fidelity_table.json",
                    "cohort_comparison.csv",
                    "cohort_comparison.json",
                    "per_sample_delta.csv",
                    "per_sample_delta.json",
                    "resource_and_qc.csv",
                    "resource_and_qc.json",
                    "curve_summary.csv",
                    "curve_summary.json",
                )
            ],
        },
        "scope_boundaries": {
            "cpu_only": True,
            "jpeg_read": False,
            "model_constructed_or_loaded": False,
            "cuda_or_gpu_used": False,
            "d9_root_modified": False,
            "d8_shards_run": False,
            "retry_attempts_created": False,
            "qualitative_overlay_contact_sheet_or_stage_f_material_generated": False,
        },
        "c10_status_preserved": C10_STATUS_PRESERVED,
        "next_task_recommendation": (
            "Prepare Stage E interpretation and reporting for the D10 decision; "
            "keep Stage F qualitative display under a separate predeclared display protocol."
        ),
        "git": dict(git_state) if git_state is not None else _git_state(repo),
    }
    document["canonical_digest"] = _semantic_digest(document)
    return MappingProxyType(document)


def write_d10_formal_decision_report(
    *, repo_root: Path | str = REPO_ROOT,
    path: Path | str = REPORT_PATH,
    analysis_root: Path | str = ANALYSIS_ROOT,
    paper_root: Path | str = PAPER_ROOT,
    d9_revalidation: Mapping[str, Any],
    git_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document = build_d10_formal_decision_report(
        repo_root=repo,
        analysis_root=analysis_root,
        paper_root=paper_root,
        d9_revalidation=d9_revalidation,
        git_state=git_state,
    )
    target = repo / os.fspath(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise StageDD10CompletionError(f"refusing to overwrite D10 report: {target}")
    target.write_bytes(canonical_file_bytes(document))
    return document


def validate_d10_formal_decision_report(
    *, repo_root: Path | str = REPO_ROOT,
    path: Path | str = REPORT_PATH,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    relative = os.fspath(path)
    document, raw = _read_json(repo / relative, "D10 formal decision report")
    if raw != canonical_file_bytes(document):
        raise StageDD10CompletionError("D10 report bytes are not canonical")
    if document.get("canonical_digest") != _semantic_digest(document):
        raise StageDD10CompletionError("D10 report self digest mismatch")
    expected = build_d10_formal_decision_report(
        repo_root=repo,
        analysis_root=document["relative_analysis_root"],
        paper_root=document["relative_paper_root"],
        d9_revalidation=document["d9_revalidation"],
        git_state=document["git"],
    )
    if document != expected:
        raise StageDD10CompletionError("D10 report is not source-recomputable")
    return MappingProxyType(document)


__all__ = [
    "ANALYSIS_ROOT",
    "COMPLETION_REF",
    "D10_ID",
    "DOCUMENT_TYPE",
    "PAPER_ROOT",
    "REPORT_PATH",
    "StageDD10CompletionError",
    "build_d10_formal_decision_report",
    "canonical_file_bytes",
    "validate_d10_formal_decision_report",
    "write_d10_formal_decision_report",
]
