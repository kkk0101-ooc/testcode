"""Launch authority and C8-salvage handoff for recovery-v2 C9."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .io_utils import sha256_file
from .stage_c_provenance import canonical_sha256, capture_git_state

SCHEMA_VERSION = 1
DOCUMENT_TYPE = "vitnns_recovery_c9_launch_record"
RECORD_ID = "RECOVERY_C9_SALVAGE_C8_20260903"
STATUS = "AUTHORIZED_FOR_RECOVERY_C9_ONLY"
AMENDMENT_ID = "RECOVERY-C9-SALVAGE-C8-20260903"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "config/recovery_c9_launch_record.schema.json"
DEFAULT_AMENDMENT_PATH = REPO_ROOT / "docs/recovery_c9_amendment_20260903.md"
DEFAULT_C8_ROOT = REPO_ROOT / "evidence/stage_c/c8_salvage_repair_recovery_v2_de71d55"
DEFAULT_C8_AUTHORITY = DEFAULT_C8_ROOT / "recovery_c8_salvage_repair_launch_record.json"
C8_ARCHIVED_CUSTODY_COMMIT = "82825131a598d4fffc22ca761e4d9a30b5d35666"
C8_LAUNCH_COMMIT = "6b98c17c4108ce2d0a0891c66399e4c1c89bf424"
C8_PRODUCER_COMMIT = "de71d55d09c1d82489ca2495479e1cb56a1c0a81"
C8_AUTHORITY_FILE_SHA256 = "5632b3d8c786c5f02bd3c85a570394262d96ba0a6cb9d24361ac08853e63d66a"
C8_AUTHORITY_CANONICAL_DIGEST = "197f53c51b4dc4c84c352de946d6d270b72e47c5f6142dc5d6dd425cc15cfd92"
C8_COMPLETION_FILE_SHA256 = "974e9366f240018d6118edc47dfbd623d9690cb407961fcf73437cee07c7898b"
C8_COMPLETION_CANONICAL_DIGEST = "4e31223f9f4e8113b035d9d8ddeb221bec916278864873180593d106684d4612"
C8_MANIFEST_FILE_SHA256 = "c6bd5f4f4b79e1b6dd67c6d53fc43e5283aca2db51b791f7c462ac5a30009101"
C8_MANIFEST_CANONICAL_DIGEST = "52ea4e8c959d73f5c0b8e3a8b46cd4a7232042e25053a46a5d86dd9166f83811"
C8_ARTIFACT_DIGEST = "4a7f1bd35aadfb704196625c709f21c1fd51159dbae8d5b1fb9dae8f95043257"
C8_SELECTION_FILE_SHA256 = "5d26c78bb94278c7f12d7bd88b1a37cfb97352f3d0bde5e1c4e52f8e811c7982"
C8_SELECTION_CANONICAL_DIGEST = "6915f7c4383c91ed5fcc465a52d3582eebdb6cd325c7bc2107b03c7996c87620"
C8_SELECTION_REPORT_FILE_SHA256 = "dcb875911c0285a28fa87dc4ec8226672e15ed39d36d4e437e89c47aa9123aae"
C8_SELECTION_REPORT_CANONICAL_DIGEST = "b0a1c557fa734673f0acc126f83a826bf493e2861788003c95476ef784ab44ce"
C8_SOURCE_REAUDIT_DIGEST = "86d946ecb096104978cf5c9b2a7bd5a6c33af91ad222c1fb49afbf1f6f5cd925"
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
WINNERS = ((0, 9), (4, 2), (6, 10), (11, 2))
PRIMARY = (0, 9)
ORIGIN_COUNTS = {
    "inherited_from_original_interrupted_root": 1460,
    "inherited_from_prior_partial_salvage_root": 569,
    "newly_run_in_salvage_repair_root": 151,
}
TERMINAL_COUNTS = {"failed": 1, "succeeded": 2179}
ROOT_INVENTORY_DIGESTS = {
    "inherited_from_original_interrupted_root": "25c4321a2067022ff21bb40616cd29a21aa5d5889956c1925d905b618bcd734d",
    "inherited_from_prior_partial_salvage_root": "56b3eda4a23ea15506252aae38bc7e93be53f0942f5dde306140e92d44cdae2e",
    "newly_run_in_salvage_repair_root": "c00af1b2993d482c99d8393199c34ce66bce99c7e88209e0befcf7233b59886f",
}
PRODUCER_COMMITS = {
    "inherited_from_original_interrupted_root": "10633700a268170bd7c6f0ee1c7a52f8d4d97610",
    "inherited_from_prior_partial_salvage_root": "6a6655b604f7310e298490837cac7e66ebbf2744",
    "newly_run_in_salvage_repair_root": C8_LAUNCH_COMMIT,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class RecoveryC9LaunchError(ValueError):
    """Raised when the recovery C9 authority or C8 handoff is invalid."""


def _plain(value: Any) -> Any:
    method = getattr(value, "to_dict", None)
    value = method() if callable(method) else value
    return json.loads(json.dumps(value, allow_nan=False))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryC9LaunchError(
            f"recovery C9 git {' '.join(args)} failed: {error}"
        ) from error


def _read(path: Path, context: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RecoveryC9LaunchError(f"duplicate key in {context}: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite {token}")
            ),
        )
    except Exception as error:
        raise RecoveryC9LaunchError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryC9LaunchError(f"{context} must be an object")
    return value


def _path(repo: Path, raw: Path | str, context: str, *, exists: bool = True) -> Path:
    candidate = Path(raw)
    result = Path(os.path.abspath(os.fspath(candidate if candidate.is_absolute() else repo / candidate)))
    try:
        result.relative_to(repo)
    except ValueError as error:
        raise RecoveryC9LaunchError(f"{context} escapes repository") from error
    current = Path(result.anchor)
    for part in result.parts[1:]:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise RecoveryC9LaunchError(f"{context} contains a symlink")
    if exists and not result.exists():
        raise RecoveryC9LaunchError(f"{context} is missing")
    return result


def _relative(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def _require_hash(path: Path, expected: str, context: str) -> None:
    if sha256_file(path) != expected:
        raise RecoveryC9LaunchError(f"{context} SHA-256 drifted")


def _self_digest(value: Mapping[str, Any]) -> str:
    return canonical_sha256({k: v for k, v in value.items() if k != "canonical_digest"})


def _capture_producer(repo: Path) -> dict[str, Any]:
    state = capture_git_state(repo)
    if state.get("dirty") is not False:
        raise RecoveryC9LaunchError("recovery C9 producer worktree must be clean")
    commit = str(state["commit"])
    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    source = (repo / "src/vitnns").relative_to(top).as_posix()
    return {
        "commit": commit, "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "source_subtree_path": "src/vitnns",
        "source_subtree_tree": _git(repo, "rev-parse", f"HEAD:{source}"),
        "dirty": False,
    }


def _load_repair(repo: Path, *, verify_custody: bool) -> Any:
    from .recovery_c8_salvage_repair import (
        load_recovery_c8_salvage_repair_launch_record,
    )
    return load_recovery_c8_salvage_repair_launch_record(
        DEFAULT_C8_AUTHORITY, repo_root=repo, require_tracked=True,
        require_fresh_outputs=False,
        archived_custody_commit=C8_ARCHIVED_CUSTODY_COMMIT,
        expected_launch_commit=C8_LAUNCH_COMMIT,
        verify_archived_custody=verify_custody,
    )


def load_salvage_selection_state(record: Any) -> Any:
    from .stage_c_plan import SelectionState, ValueCandidate
    path = Path(record.repair_record.selection_path)
    _require_hash(path, C8_SELECTION_FILE_SHA256, "C8 salvage selection")
    selection = _read(path, "C8 salvage selection")
    if (
        selection.get("document_type") != "stage_c_c8_salvage_repair_selection"
        or selection.get("holdout_accessed") is not False
        or selection.get("canonical_digest") != C8_SELECTION_CANONICAL_DIGEST
        or _self_digest(selection) != C8_SELECTION_CANONICAL_DIGEST
    ):
        raise RecoveryC9LaunchError("C8 salvage selection identity drifted")
    winners = tuple(
        ValueCandidate.from_selection_ref(item, f"salvage winner {index}")
        for index, item in enumerate(selection.get("depth_band_winners", ()))
    )
    primary = ValueCandidate.from_selection_ref(
        selection.get("primary"), "salvage primary"
    )
    if (
        tuple((item.block_index, item.head_index) for item in winners) != WINNERS
        or (primary.block_index, primary.head_index) != PRIMARY
    ):
        raise RecoveryC9LaunchError("C8 salvage winners or primary drifted")
    return SelectionState(
        1, "present", _relative(record.repo_root, path), None,
        C8_SELECTION_FILE_SHA256, primary, winners,
    )


def recovery_c9_provenance_context(record: Any) -> Any:
    if type(record) is not RecoveryC9LaunchRecord:
        raise RecoveryC9LaunchError("validated recovery C9 authority required")
    validate_recovery_c9_runtime_record(record)
    return record.repair_record


def _adapt_c8_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    from .stage_c_c8_salvage_repair_selection import _adapt_records
    records, _identities, _counts = _adapt_records(manifest)
    return records


def _control_paths(repair: Any, manifest: Mapping[str, Any]) -> list[str]:
    paths: list[Path] = []
    for name in repair.document["original_interrupted_root"]["control_sha256"]:
        paths.append(repair.source_root / name)
    prior_map = {
        "execution_plan.json": repair.prior_gap_root / "execution_plan.json",
        "preflight.json": repair.prior_gap_root / "preflight.json",
        "salvage_scope.json": repair.prior_gap_root / "salvage_scope.json",
        "salvage_status.json": repair.prior_gap_root / "salvage_status.json",
        "c8_salvage_source_audit.json": repair.prior_gap_root.parent / "reports/c8_salvage_source_audit.json",
        "c8_partial_scientific_failure_diagnosis.json": repair.prior_gap_root.parent / "reports/c8_partial_scientific_failure_diagnosis.json",
    }
    paths.extend(prior_map.values())
    paths.extend(repair.repair_root / name for name in (
        "execution_plan.json", "preflight.json",
        "salvage_repair_scope.json", "salvage_repair_status.json",
    ))
    paths.append(repair.reports_root / "c8_salvage_repair_source_audit.json")
    expected = manifest.get("budget_evidence", {}).get("bindings", {}).get("control_file_sha256", {})
    if len(paths) != len(expected):
        raise RecoveryC9LaunchError("C8 salvage control inventory drifted")
    return [str(path) for path in paths]


def _validate_salvage_payload(
    record: Any, *, full_source: bool,
    evidence_out: list[dict[str, Any]] | None = None,
    include_authority: bool = True,
) -> dict[str, Any]:
    repair = record.repair_record
    root = record.c8_root
    completion_path = repair.combined_report_path
    manifest_path = repair.combined_manifest_path
    selection_path = repair.selection_path
    selection_report_path = repair.selection_report_path
    for path, digest, label in (
        (repair.path, C8_AUTHORITY_FILE_SHA256, "C8 repair authority"),
        (completion_path, C8_COMPLETION_FILE_SHA256, "C8 completion"),
        (manifest_path, C8_MANIFEST_FILE_SHA256, "C8 combined manifest"),
        (selection_path, C8_SELECTION_FILE_SHA256, "C8 selection"),
        (selection_report_path, C8_SELECTION_REPORT_FILE_SHA256, "C8 selection report"),
    ):
        _require_hash(path, digest, label)
    completion = _read(completion_path, "C8 completion")
    manifest = _read(manifest_path, "C8 combined manifest")
    selection = _read(selection_path, "C8 selection")
    selection_report = _read(selection_report_path, "C8 selection report")
    if full_source:
        from .stage_c_c8_salvage_repair import (
            validate_c8_salvage_repair_from_source,
        )
        from .stage_c_c8_salvage_repair_selection import _derive
        from .stage_c_config import load_stage_c_config
        from .stage_c_plan import build_execution_plan, load_run_spec, load_stage_c_registry
        config = load_stage_c_config(REPO_ROOT / "config/stage_c.json")
        registry = load_stage_c_registry(
            config, recovery_binding=repair.binding, verify_live_assets=False,
        )
        plan = build_execution_plan(
            config, registry,
            load_run_spec(REPO_ROOT / "config/stage_c_c8_run_spec.json"),
            selection=None,
        )
        manifest = validate_c8_salvage_repair_from_source(
            repair, plan, config,
        )
        expected_selection, expected_selection_report = _derive(
            repair, manifest,
            generator_git_state=selection_report["assembler_identity"],
        )
        if (
            expected_selection != selection
            or expected_selection_report != selection_report
        ):
            raise RecoveryC9LaunchError(
                "C8 selection differs from source recomputation"
            )
    winner_pairs = tuple(
        (item.get("block_index"), item.get("head_index"))
        for item in selection.get("depth_band_winners", ())
        if isinstance(item, Mapping)
    )
    primary = selection.get("primary", {})
    if (
        completion.get("status") != "PASS" or completion.get("c8_pass") is not True
        or completion.get("canonical_digest") != C8_COMPLETION_CANONICAL_DIGEST
        or manifest.get("canonical_digest") != C8_MANIFEST_CANONICAL_DIGEST
        or manifest.get("combined_artifact_manifest_digest") != C8_ARTIFACT_DIGEST
        or manifest.get("source_reaudit_digest") != C8_SOURCE_REAUDIT_DIGEST
        or manifest.get("origin_counts") != ORIGIN_COUNTS
        or manifest.get("terminal_state_counts") != TERMINAL_COUNTS
        or len(manifest.get("records", ())) != 2180
        or winner_pairs != WINNERS
        or (primary.get("block_index"), primary.get("head_index")) != PRIMARY
        or selection_report.get("status") != "PASS"
        or selection_report.get("canonical_digest") != C8_SELECTION_REPORT_CANONICAL_DIGEST
        or selection.get("canonical_digest") != C8_SELECTION_CANONICAL_DIGEST
        or any(completion.get("flags", {}).get(flag) is not False for flag in (
            "holdout_accessed", "bootstrap_performed", "holm_performed",
            "decision_performed", "c9_authorized", "c10_authorized",
            "stage_d_authorized",
        ))
    ):
        raise RecoveryC9LaunchError("C8 salvage PASS binding drifted")
    inventories = manifest.get("root_inventories", {})
    identities = manifest.get("producer_identities", {})
    if (
        {key: value.get("digest") for key, value in inventories.items()}
        != ROOT_INVENTORY_DIGESTS
        or {key: value.get("commit") for key, value in identities.items()}
        != PRODUCER_COMMITS
    ):
        raise RecoveryC9LaunchError("C8 salvage origins or inventories drifted")
    post = selection_report.get("post_c8_budget", {})
    projection = post.get("projection", {})
    time = projection.get("time", {})
    storage = projection.get("storage", {})
    maxima = projection.get("measured_maxima", {})
    source_bytes = manifest.get("budget_evidence", {}).get("bindings", {}).get("source_tree_total_bytes")
    handoff = {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_repair_to_c9_handoff",
        "status": "PASS", "integrity_valid": True,
        "c8_root": str(root),
        "c8_report_path": str(completion_path),
        "c8_report_file_sha256": C8_COMPLETION_FILE_SHA256,
        "c8_report_canonical_digest": C8_COMPLETION_CANONICAL_DIGEST,
        "c8_recovery_selection_path": str(selection_path),
        "c8_recovery_selection_file_sha256": C8_SELECTION_FILE_SHA256,
        "tracked_selection_path": str(selection_path),
        "tracked_selection_file_sha256": C8_SELECTION_FILE_SHA256,
        "selection_canonical_digest": C8_SELECTION_CANONICAL_DIGEST,
        "c8_completion_commit": C8_ARCHIVED_CUSTODY_COMMIT,
        "c8_execution_plan_sha256": manifest.get("plan_digest"),
        "c8_run_spec_sha256": canonical_sha256(_read(REPO_ROOT / "config/stage_c_c8_run_spec.json", "C8 run spec")),
        "c8_source_tree_total_bytes": source_bytes,
        "c8_retained_artifact_bytes": source_bytes,
        "c9_projected_artifact_bytes": storage.get("projected_bytes", {}).get("C9"),
        "c8_c9_projected_artifact_bytes": storage.get("projected_bytes", {}).get("C8_C9"),
        "c9_projected_h20_gpu_hours": time.get("C9_h20_gpu_hours"),
        "c8_c9_projected_h20_gpu_hours": time.get("C8_C9_h20_gpu_hours"),
        "measured_slowest_unit_seconds": time.get("slowest_valid_measured_unit_seconds"),
        "measured_peak_reserved_bytes_per_unit": maxima.get("cuda_peak_reserved_bytes_per_unit"),
        "stage_d_projected_artifact_bytes": storage.get("projected_bytes", {}).get("StageD"),
        "stage_d_projected_h20_gpu_hours": time.get("StageD_h20_gpu_hours"),
        "stage_d_projected_h20_gpu_hours_per_image": time.get("StageD_h20_gpu_hours_per_image"),
        "salvage_binding": {
            "repair_authority_file_sha256": C8_AUTHORITY_FILE_SHA256,
            "repair_authority_canonical_digest": C8_AUTHORITY_CANONICAL_DIGEST,
            "repair_launch_commit": C8_LAUNCH_COMMIT,
            "archived_custody_commit": C8_ARCHIVED_CUSTODY_COMMIT,
            "combined_manifest_file_sha256": C8_MANIFEST_FILE_SHA256,
            "combined_manifest_canonical_digest": C8_MANIFEST_CANONICAL_DIGEST,
            "combined_artifact_manifest_digest": C8_ARTIFACT_DIGEST,
            "source_reaudit_digest": C8_SOURCE_REAUDIT_DIGEST,
            "selection_report_file_sha256": C8_SELECTION_REPORT_FILE_SHA256,
            "selection_report_canonical_digest": C8_SELECTION_REPORT_CANONICAL_DIGEST,
            "origin_counts": ORIGIN_COUNTS, "terminal_state_counts": TERMINAL_COUNTS,
            "root_inventory_digests": ROOT_INVENTORY_DIGESTS,
            "producer_commits": PRODUCER_COMMITS,
        },
    }
    if include_authority:
        handoff["recovery_c9_authority"] = {
            "path": _relative(record.repo_root, record.path),
            "file_sha256": record.file_sha256,
            "canonical_digest": record.canonical_digest,
            "launch_commit": record.launch_commit,
            "producer_commit": record.producer_commit,
            "producer_source_subtree_tree": record.document["producer_git"]["source_subtree_tree"],
        }
        handoff["canonical_digest"] = canonical_sha256(handoff)
        validate_recovery_handoff_document(handoff)
    if evidence_out is not None:
        evidence_out.append({
            "source_root": str(root),
            "bindings": {"source_tree_total_bytes": source_bytes},
            "c7_report_binding": _plain(manifest["budget_evidence"]["c7_report_binding"]),
            "records": _adapt_c8_records(manifest),
            "control_file_paths": _control_paths(repair, manifest),
        })
    return handoff


def validate_recovery_handoff_document(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "document_type", "status", "integrity_valid",
        "c8_root", "c8_report_path", "c8_report_file_sha256",
        "c8_report_canonical_digest", "c8_recovery_selection_path",
        "c8_recovery_selection_file_sha256", "tracked_selection_path",
        "tracked_selection_file_sha256", "selection_canonical_digest",
        "c8_completion_commit", "c8_execution_plan_sha256",
        "c8_run_spec_sha256", "c8_source_tree_total_bytes",
        "c8_retained_artifact_bytes", "c9_projected_artifact_bytes",
        "c8_c9_projected_artifact_bytes", "c9_projected_h20_gpu_hours",
        "c8_c9_projected_h20_gpu_hours", "measured_slowest_unit_seconds",
        "measured_peak_reserved_bytes_per_unit",
        "stage_d_projected_artifact_bytes",
        "stage_d_projected_h20_gpu_hours",
        "stage_d_projected_h20_gpu_hours_per_image", "salvage_binding",
        "recovery_c9_authority", "canonical_digest",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RecoveryC9LaunchError("recovery C8 handoff fields drifted")
    if (
        value.get("schema_version") != 1
        or value.get("document_type") != "stage_c_c8_salvage_repair_to_c9_handoff"
        or value.get("status") != "PASS"
        or value.get("integrity_valid") is not True
        or value.get("canonical_digest") != _self_digest(value)
        or value.get("c8_report_file_sha256") != C8_COMPLETION_FILE_SHA256
        or value.get("c8_report_canonical_digest") != C8_COMPLETION_CANONICAL_DIGEST
        or value.get("c8_recovery_selection_file_sha256") != C8_SELECTION_FILE_SHA256
        or value.get("selection_canonical_digest") != C8_SELECTION_CANONICAL_DIGEST
        or value.get("c8_completion_commit") != C8_ARCHIVED_CUSTODY_COMMIT
        or value.get("c8_root") != str(DEFAULT_C8_ROOT)
    ):
        raise RecoveryC9LaunchError("recovery C8 handoff identity drifted")


@dataclass(frozen=True)
class RecoveryC9LaunchRecord:
    path: Path
    schema_path: Path
    document: Mapping[str, Any]
    repair_record: Any
    c8_root: Path
    output_root: Path
    model_path: Path
    binding: Any
    c7_launch_record: Any
    launch_commit: str
    producer_commit: str
    file_sha256: str
    repo_root: Path = REPO_ROOT

    @property
    def canonical_digest(self) -> str:
        return str(self.document["canonical_digest"])


def _resolve_launch(repo: Path, path: Path, document: Mapping[str, Any]) -> str:
    producer = document["producer_git"]["commit"]
    launch = document["launch_record_commit"]
    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    relative = path.relative_to(top).as_posix()
    if launch == "SELF":
        launch = _git(repo, "log", "-n", "1", "--format=%H", "HEAD", "--", f":(top){relative}")
    if _git(repo, "rev-list", "--parents", "-n", "1", launch).split() != [launch, producer]:
        raise RecoveryC9LaunchError("C9 authority must directly follow producer")
    changed = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", producer, launch).splitlines()
    if changed != [relative]:
        raise RecoveryC9LaunchError("C9 authority commit must be launch-only")
    return launch


def _validate_runtime(record: RecoveryC9LaunchRecord) -> None:
    state = capture_git_state(record.repo_root)
    if state.get("dirty") is not False:
        raise RecoveryC9LaunchError("C9 runtime worktree must be clean")
    current = str(state["commit"])
    _git(record.repo_root, "merge-base", "--is-ancestor", record.launch_commit, current)
    top = Path(_git(record.repo_root, "rev-parse", "--show-toplevel"))
    source = (record.repo_root / "src/vitnns").relative_to(top).as_posix()
    if _git(record.repo_root, "rev-parse", f"{current}:{source}") != record.document["producer_git"]["source_subtree_tree"]:
        raise RecoveryC9LaunchError("C9 runtime source differs from producer")


def _validate_document(document: Mapping[str, Any]) -> None:
    required = {
        "$schema", "schema_version", "document_type", "record_id",
        "status", "amendment", "producer_git", "launch_record_commit",
        "c8_salvage", "model", "contracts", "outputs", "scope",
        "canonical_digest",
    }
    if set(document) != required or document.get("canonical_digest") != _self_digest(document):
        raise RecoveryC9LaunchError("C9 authority fields or digest drifted")
    if (
        document.get("schema_version") != 1
        or document.get("document_type") != DOCUMENT_TYPE
        or document.get("record_id") != RECORD_ID
        or document.get("status") != STATUS
        or document.get("scope") != {
            "c9_authorized": True, "holdout_access_authorized": True,
            "c10_authorized": False, "stage_d_authorized": False,
            "stage_d_reserve_access_authorized": False,
        }
    ):
        raise RecoveryC9LaunchError("C9 authority identity or scope drifted")
    producer = document.get("producer_git")
    if (
        not isinstance(producer, Mapping)
        or set(producer) != {
            "commit", "tree", "source_subtree_path",
            "source_subtree_tree", "dirty",
        }
        or any(_COMMIT.fullmatch(str(producer.get(key))) is None for key in (
            "commit", "tree", "source_subtree_tree",
        ))
        or producer.get("source_subtree_path") != "src/vitnns"
        or producer.get("dirty") is not False
    ):
        raise RecoveryC9LaunchError("C9 producer identity drifted")
    if document.get("model") != {
        "identity": "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718",
        "state_dict_path": (
            "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"
        ),
        "state_dict_sha256": MODEL_SHA256,
        "canonical_tensor_digest": MODEL_TENSOR_DIGEST,
        "historical_artifact_recovered": False,
    }:
        raise RecoveryC9LaunchError("C9 recovery-v2 model binding drifted")
    contracts = document.get("contracts")
    expected_contracts = {
        "recovery_binding", "stage_c_config", "c9_run_spec",
        "protocol", "acceptance", "authority_schema",
    }
    if not isinstance(contracts, Mapping) or set(contracts) != expected_contracts:
        raise RecoveryC9LaunchError("C9 contract set drifted")
    for key, item in contracts.items():
        if (
            not isinstance(item, Mapping) or set(item) != {"path", "file_sha256"}
            or not isinstance(item.get("path"), str)
            or _SHA256.fullmatch(str(item.get("file_sha256"))) is None
        ):
            raise RecoveryC9LaunchError(f"C9 contract binding is invalid: {key}")
    outputs = document.get("outputs")
    if (
        not isinstance(outputs, Mapping) or set(outputs) != {"formal_root"}
        or not isinstance(outputs.get("formal_root"), str)
        or not outputs["formal_root"].startswith("evidence/stage_c/c9_recovery_v2_")
        or not outputs["formal_root"].endswith("/formal")
    ):
        raise RecoveryC9LaunchError("C9 formal output root is invalid")


def generate_recovery_c9_launch_record(
    *, repo_root: Path | str, launch_record_path: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    launch = _path(repo, launch_record_path, "C9 authority", exists=False)
    output = _path(repo, output_root, "C9 output root", exists=False)
    if os.path.lexists(launch) or os.path.lexists(output):
        raise RecoveryC9LaunchError("C9 authority and formal root must be fresh")
    producer = _capture_producer(repo)
    repair = _load_repair(repo, verify_custody=True)
    temporary = RecoveryC9LaunchRecord(
        launch, DEFAULT_SCHEMA_PATH, {"producer_git": producer}, repair, DEFAULT_C8_ROOT, output,
        repair.model_path, repair.binding, repair.c7_launch_record, "SELF",
        producer["commit"], "0" * 64, repo,
    )
    # Full source recomputation occurs before authority bytes are written.
    completion = _validate_salvage_payload(
        temporary, full_source=True, include_authority=False,
    )
    amendment = _path(repo, DEFAULT_AMENDMENT_PATH, "C9 amendment")
    schema = _path(repo, DEFAULT_SCHEMA_PATH, "C9 authority schema")
    model = repo / "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"
    _require_hash(model, MODEL_SHA256, "recovery-v2 model")
    contracts = {}
    for key, relative in {
        "recovery_binding": "config/recovery_v2_binding.json",
        "stage_c_config": "config/stage_c.json",
        "c9_run_spec": "config/stage_c_c9_run_spec.json",
        "protocol": "docs/阶段C_protocol.md",
        "acceptance": "docs/stage_c_acceptance.md",
        "authority_schema": "config/recovery_c9_launch_record.schema.json",
    }.items():
        path = repo / relative
        contracts[key] = {"path": relative, "file_sha256": sha256_file(path)}
    document = {
        "$schema": DEFAULT_SCHEMA_PATH.name, "schema_version": 1,
        "document_type": DOCUMENT_TYPE, "record_id": RECORD_ID,
        "status": STATUS,
        "amendment": {
            "id": AMENDMENT_ID, "path": _relative(repo, amendment),
            "file_sha256": sha256_file(amendment),
            "containing_commit": producer["commit"],
        },
        "producer_git": producer, "launch_record_commit": "SELF",
        "c8_salvage": _plain(completion),
        "model": {
            "identity": "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718",
            "state_dict_path": _relative(repo, model),
            "state_dict_sha256": MODEL_SHA256,
            "canonical_tensor_digest": MODEL_TENSOR_DIGEST,
            "historical_artifact_recovered": False,
        },
        "contracts": contracts,
        "outputs": {"formal_root": _relative(repo, output)},
        "scope": {
            "c9_authorized": True, "holdout_access_authorized": True,
            "c10_authorized": False, "stage_d_authorized": False,
            "stage_d_reserve_access_authorized": False,
        },
    }
    document["canonical_digest"] = _self_digest(document)
    _validate_document(document)
    Draft202012Validator(_read(schema, "C9 authority schema")).validate(document)
    return document


def write_recovery_c9_launch_record(**kwargs: Any) -> Path:
    repo = Path(kwargs["repo_root"]).resolve(strict=True)
    path = _path(repo, kwargs["launch_record_path"], "C9 authority", exists=False)
    document = generate_recovery_c9_launch_record(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return path


def load_recovery_c9_launch_record(
    path: Path | str, *, repo_root: Path | str = REPO_ROOT,
    require_tracked: bool = True, full_c8_validation: bool = False,
) -> RecoveryC9LaunchRecord:
    repo = Path(repo_root).resolve(strict=True)
    launch = _path(repo, path, "C9 authority")
    schema = _path(repo, DEFAULT_SCHEMA_PATH, "C9 authority schema")
    document = _read(launch, "C9 authority")
    _validate_document(document)
    Draft202012Validator(_read(schema, "C9 authority schema")).validate(document)
    producer = document["producer_git"]
    for commit in (producer["commit"], C8_ARCHIVED_CUSTODY_COMMIT, C8_LAUNCH_COMMIT):
        if _git(repo, "rev-parse", f"{commit}^{{commit}}") != commit:
            raise RecoveryC9LaunchError("required custody commit is unavailable")
    if _git(repo, "rev-parse", f"{producer['commit']}^{{tree}}") != producer["tree"]:
        raise RecoveryC9LaunchError("C9 producer tree drifted")
    for record in document["contracts"].values():
        target = _path(repo, record["path"], "C9 contract")
        _require_hash(target, record["file_sha256"], "C9 contract")
    amendment = document["amendment"]
    amendment_path = _path(repo, amendment["path"], "C9 amendment")
    _require_hash(amendment_path, amendment["file_sha256"], "C9 amendment")
    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    amendment_top = amendment_path.relative_to(top).as_posix()
    if (
        amendment.get("containing_commit") != producer["commit"]
        or subprocess.run(
            ["git", "-C", str(repo), "show",
             f"{producer['commit']}:{amendment_top}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=120,
        ).stdout != amendment_path.read_bytes()
    ):
        raise RecoveryC9LaunchError("C9 amendment is not bound to producer")
    model_path = _path(repo, document["model"]["state_dict_path"], "C9 model")
    _require_hash(model_path, MODEL_SHA256, "C9 model")
    repair = _load_repair(repo, verify_custody=full_c8_validation)
    launch_commit = (
        _resolve_launch(repo, launch, document) if require_tracked
        else producer["commit"]
    )
    output = _path(repo, document["outputs"]["formal_root"], "C9 formal root", exists=False)
    result = RecoveryC9LaunchRecord(
        launch, schema, _freeze(document), repair, DEFAULT_C8_ROOT, output,
        model_path, repair.binding, repair.c7_launch_record, launch_commit,
        producer["commit"], sha256_file(launch), repo,
    )
    if require_tracked:
        _validate_runtime(result)
    observed = _validate_salvage_payload(
        result, full_source=full_c8_validation, include_authority=False,
    )
    expected = _plain(document["c8_salvage"])
    if observed != expected:
        raise RecoveryC9LaunchError("C8 salvage handoff differs from C9 authority")
    return result


def validate_recovery_c9_runtime_record(record: RecoveryC9LaunchRecord) -> RecoveryC9LaunchRecord:
    authoritative = load_recovery_c9_launch_record(
        record.path, repo_root=record.repo_root, require_tracked=True,
        full_c8_validation=False,
    )
    if authoritative != record:
        raise RecoveryC9LaunchError("C9 authority differs from its source bytes")
    return authoritative


def validate_recovery_c8_handoff(
    handoff: Mapping[str, Any], *, full_source: bool,
    evidence_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_recovery_handoff_document(handoff)
    authority = handoff["recovery_c9_authority"]
    path = REPO_ROOT / authority["path"]
    record = load_recovery_c9_launch_record(
        path, require_tracked=True, full_c8_validation=False,
    )
    observed = _validate_salvage_payload(
        record, full_source=full_source, evidence_out=evidence_out,
    )
    if observed != dict(handoff):
        raise RecoveryC9LaunchError("persisted recovery C8 handoff drifted")
    return observed


def build_runtime_handoff(record: RecoveryC9LaunchRecord, *, full_source: bool) -> dict[str, Any]:
    _validate_runtime(record)
    return _validate_salvage_payload(record, full_source=full_source)


__all__ = [
    "RecoveryC9LaunchError", "RecoveryC9LaunchRecord",
    "generate_recovery_c9_launch_record", "write_recovery_c9_launch_record",
    "load_recovery_c9_launch_record", "validate_recovery_c9_runtime_record",
    "validate_recovery_c8_handoff", "validate_recovery_handoff_document",
    "load_salvage_selection_state",
    "recovery_c9_provenance_context",
]
