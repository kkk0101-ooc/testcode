#!/usr/bin/env python3
"""Stage Y2 orchestration: metadata seal, worker launch, and reconcile.

`seal-cohort` is Python-standard-library-only and cannot decode images, load a
model, import Torch/timm, or access CUDA. The empirical path lives in the
separate stage_y_y2_worker module and requires a completed pre-execution seal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from vitnns.constants import REPO_ROOT
from vitnns import stage_y_y2 as y2

EXPECTED_REFS = {
    y2.Y0_REF: "a6efa1426902708d83de1d83ef257d524cacf95b",
    y2.Y1_BLOCKED_REF: "d200a5b3de2e46e5d6de2ef07a6c0f5230b1c082",
    y2.Y1_R1_REF: y2.EXPECTED_BASE,
}
SOURCE_FILES = (
    Path("docs/stage_y_y2_clean_cohort_production_20260911.md"),
    Path("config/stage_y/y2_run_spec.json"),
    Path("config/stage_y/y2_run_spec.schema.json"),
    Path("config/stage_y/y2_artifact.schema.json"),
    Path("config/stage_y/y2_unit_terminal.schema.json"),
    Path("config/stage_y/y2_completion.schema.json"),
    Path("config/stage_y/y2_startup_gate.schema.json"),
    Path("config/stage_y/y2_shard_manifest.schema.json"),
    Path("config/stage_y/y2_parallel_schedule.json"),
    Path("docs/stage_y_y2_p0_parallel_schedule_amendment_20260911.md"),
    Path("docs/stage_y_y2_p0_r1_elastic_idle_h20_schedule_20260911.md"),
    Path("src/vitnns/stage_y_y2.py"),
    Path("src/vitnns/stage_y_y2_worker.py"),
    Path("scripts/run_stage_y_y2.py"),
    Path("scripts/validate_stage_y_y2.py"),
    Path("tests/test_stage_y_y2.py"),
    Path("scripts/validate_stage_y_y2_p0.py"),
    Path("tests/test_stage_y_y2_p0.py"),
    Path("scripts/validate_stage_y_y2_p0_r1.py"),
    Path("tests/test_stage_y_y2_p0_r1.py"),
)
R1_SOURCE_FILES = SOURCE_FILES + (
    Path("docs/stage_y_y2_r1_cublas_determinism_recovery_20260911.md"),
    Path("config/stage_y/y2_r1_run_spec.json"),
    Path("config/stage_y/y2_r1_run_spec.schema.json"),
    Path("config/stage_y/y2_r1_artifact.schema.json"),
    Path("scripts/validate_stage_y_y2_r1.py"),
    Path("tests/test_stage_y_y2_r1.py"),
)
ORIGINAL_Y2_REF = "refs/archive/stage-y-y2-clean-cohort-evidence-blocked"
ORIGINAL_Y2_COMMIT = "1a621724b32b208036fb5f2a3993a48697e933c8"
R1_PASS_REF = "refs/archive/stage-y-y2-r1-clean-cohort-evidence"
R1_BLOCKED_REF = "refs/archive/stage-y-y2-r1-clean-cohort-evidence-blocked"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise y2.StageYY2ProvenanceError(message)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def repo_file_record(repo: Path, relative: Path) -> dict[str, Any]:
    result = y2.file_record(repo / relative)
    result["path"] = relative.as_posix()
    return result


def remote_ref(repo: Path, ref: str) -> str:
    output = y2.git(repo, "ls-remote", "origin", ref).split()
    if len(output) != 2 or output[1] != ref:
        raise y2.StageYY2ProvenanceError(f"remote ref missing or ambiguous: {ref}")
    return output[0]


def _history_binding(repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for ref, expected in EXPECTED_REFS.items():
        require(y2.git(repo, "rev-parse", ref) == expected, f"local archive ref drift: {ref}")
        require(remote_ref(repo, ref) == expected, f"remote archive ref drift: {ref}")
    y0_completion = repo_file_record(repo, Path("evidence/stage_y/y0_protocol_20260910/formal/completion/y0.json"))
    y0_inventory = repo_file_record(repo, Path("evidence/stage_y/y0_protocol_20260910/formal/whole_root_inventory.json"))
    y1_completion_path = Path("evidence/stage_y/y1_validation_pilot_20260910/formal/completion/y1.json")
    y1_failure_path = Path("evidence/stage_y/y1_validation_pilot_20260910/formal/failure.json")
    y1_completion = y2.read_json(repo / y1_completion_path)
    y1_failure = y2.read_json(repo / y1_failure_path)
    require(y1_completion["status"] == "BLOCKED_BUDGET", "original Y1 completion status drift")
    require(y1_completion["payload"]["next_authorization"] == "NONE", "original Y1 authorization drift")
    require(y1_failure["payload"]["failure_layer"] == "RESOURCE_TELEMETRY", "original Y1 failure layer drift")
    diff = subprocess.run(["git", "diff", "--exit-code", EXPECTED_REFS[y2.Y1_BLOCKED_REF], "--", "evidence/stage_y/y1_validation_pilot_20260910/formal"], cwd=repo).returncode
    require(diff == 0, "original Y1 formal root differs from blocked archive commit")
    r1_path = Path("evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/completion/y1_r1.json")
    r1 = y2.read_json(repo / r1_path)
    require(r1["status"] == "PASS", "Y1-R1 completion is not PASS")
    require(r1["payload"]["completion_basis"] == "PASS_WITH_EXPLICIT_OWNER_RESOURCE_TELEMETRY_WAIVER", "Y1-R1 basis drift")
    require(r1["payload"]["next_authorization"] == y2.TOKEN, "Y1-R1 authorization drift")
    y0 = {"status": "PASS", "archive_ref": y2.Y0_REF, "commit": EXPECTED_REFS[y2.Y0_REF], "completion": y0_completion, "whole_root_inventory": y0_inventory}
    original = {"status": "BLOCKED_BUDGET", "failure_layer": "RESOURCE_TELEMETRY", "next_authorization": "NONE", "archive_ref": y2.Y1_BLOCKED_REF, "commit": EXPECTED_REFS[y2.Y1_BLOCKED_REF], "completion": repo_file_record(repo, y1_completion_path), "failure": repo_file_record(repo, y1_failure_path), "formal_tree_exact_at_blocked_commit": True}
    r1_binding = {"status": "PASS", "completion_basis": "PASS_WITH_EXPLICIT_OWNER_RESOURCE_TELEMETRY_WAIVER", "archive_ref": y2.Y1_R1_REF, "commit": y2.EXPECTED_BASE, "completion": repo_file_record(repo, r1_path), "conservative_budget_receipt": repo_file_record(repo, Path("evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/conservative_budget_receipt.json")), "waiver_scope": "HISTORICAL_PHASE_LEVEL_TEMETRY_ONLY", "waiver_applies_to_y2": False}
    return y0, original, r1_binding


def _git_tree_digest(repo: Path, commit: str, paths: Sequence[Path]) -> str:
    prefix = repo.relative_to(repo.parent).as_posix()
    args = ["ls-tree", "-r", "--full-tree", commit, "--"] + [f"{prefix}/{path.as_posix()}" for path in paths]
    return hashlib.sha256(y2.git(repo, *args).encode("utf-8")).hexdigest()


def _materialized_tree_digest(repo: Path, paths: Sequence[Path]) -> tuple[str, int]:
    rows = []
    for relative_root in paths:
        for path in sorted((repo / relative_root).rglob("*")):
            if path.is_file():
                rows.append(y2.file_record(path))
    return y2.canonical_digest({"files": rows}), len(rows)


def _validate_original_y2(repo: Path, *, recompute_release: bool) -> dict[str, Any]:
    root = repo / y2.ORIGINAL_ROOT
    require(y2.git(repo, "rev-parse", ORIGINAL_Y2_REF) == ORIGINAL_Y2_COMMIT, "original Y2 local archive ref drift")
    require(remote_ref(repo, ORIGINAL_Y2_REF) == ORIGINAL_Y2_COMMIT, "original Y2 remote archive ref drift")
    protected = (y2.ORIGINAL_ROOT, y2.ORIGINAL_REPORT_ROOT)
    diff = subprocess.run(["git", "diff", "--exit-code", ORIGINAL_Y2_COMMIT, "--", *(path.as_posix() for path in protected)], cwd=repo).returncode
    require(diff == 0, "original Y2 root/report differs from immutable archive commit")
    completion = y2.read_json(root / "completion/y2.json")
    failure = y2.read_json(root / "failure.json")
    reconciliation = y2.read_json(root / "reconciliation.json")
    for document in (completion, failure, reconciliation):
        y2.validate_signed(document, allow_historical_y2=True)
    require(completion["status"] == "BLOCKED_INFRASTRUCTURE", "original Y2 status drift")
    require(completion["payload"]["valid_clean_count"] == 0 and completion["payload"]["formal_outcome_access"] is True and completion["payload"]["next_authorization"] == "NONE", "original Y2 completion semantics drift")
    require(failure["payload"] == {"failure_type": "BLOCKED_INFRASTRUCTURE", "message": "source-recomputed reconciliation did not pass", "next_authorization": "NONE", "phase": "RECONCILIATION", "recovery_requires_new_root": True}, "original Y2 failure receipt drift")
    require(reconciliation["payload"]["terminal_counts"] == {"VALID": 0, "FAILED_NUMERICAL": 0, "BLOCKED_INFRASTRUCTURE": 100, "BLOCKED_BUDGET": 0}, "original Y2 terminal counts drift")
    terminals = sorted(root.glob("shards/shard-*/units/unit-*/completion/unit.json"))
    require(len(terminals) == 100, "original Y2 does not retain exactly 100 unit terminals")
    expected_failure = None; terminal_records = []
    for path in terminals:
        document = y2.read_json(path); y2.validate_signed(document, allow_historical_y2=True)
        payload = document["payload"]
        require(document["status"] == "BLOCKED_INFRASTRUCTURE", "original Y2 unit status is not homogeneous infrastructure failure")
        require(payload["target_identity"] == {} and payload["quadrature_attempts"] == [], "original Y2 failure reached scientific target or quadrature")
        require(payload["array_records"] == [] and payload["scientific_record"] is None, "original Y2 failure retained scientific arrays/results")
        code = payload["failure_code"]
        require("uses CuBLAS" in code and "CUBLAS_WORKSPACE_CONFIG=:4096:8" in code, "original Y2 failure is not the frozen cuBLAS environment failure")
        if expected_failure is None: expected_failure = code
        require(code == expected_failure, "original Y2 failure codes are not byte-identical")
        terminal_records.append({"selection_order": payload["selection_order"], **y2.file_record(path)})
    require(sorted(row["selection_order"] for row in terminal_records) == list(range(100)), "original Y2 selection-order coverage drift")
    cohort = y2.read_json(root / "cohort_registry.json"); y2.validate_signed(cohort, allow_historical_y2=True)
    proof = y2.read_json(root / "cohort_selection_proof.json"); y2.validate_signed(proof, allow_historical_y2=True)
    audit = y2.read_json(root / "audit_subset_registry.json"); y2.validate_signed(audit, allow_historical_y2=True)
    perturbation = y2.read_json(root / "perturbation_subset_registry.json"); y2.validate_signed(perturbation, allow_historical_y2=True)
    require(cohort["canonical_digest"] == "156eab4bd86ff9979abc6acf8b06af3aa5a90e90cd02edef7713539b147ce277", "original cohort canonical digest drift")
    require(cohort["payload"]["selection_digest"] == proof["payload"]["selection_digest"] == "8b3faa6d437bda38e963985ac12f14f3b440a3e4021915081f1d5e1b1c80d2cf", "original selection digest drift")
    require(cohort["payload"]["count"] == 100 and cohort["payload"]["distinct_true_class_count"] == 100, "original cohort cardinality drift")
    require(audit["payload"]["count"] == 10 and perturbation["payload"]["count"] == 20, "original subset cardinality drift")
    source_recomputed = None
    if recompute_release:
        exclusion, excluded = y2.exclusion_union(repo)
        candidates, metadata = y2.enumerate_candidates(repo)
        selected, selected_proof = y2.select_cohort(candidates, excluded)
        recomputed_audit = y2.subset_registry(selected, y2.AUDIT_NAMESPACE, 10, "audit_10")
        recomputed_perturbation = y2.subset_registry(selected, y2.PERTURBATION_NAMESPACE, 20, "perturbation_20")
        require(selected == cohort["payload"]["records"], "raw-source cohort differs from original Y2 registry")
        require(selected_proof["selection_digest"] == cohort["payload"]["selection_digest"], "raw-source selection digest drift")
        require(recomputed_audit == audit["payload"] and recomputed_perturbation == perturbation["payload"], "raw-source subset identity drift")
        source_recomputed = {"candidate_metadata": metadata, "exclusion_union_digest": exclusion["union_digest"], "selection_digest": selected_proof["selection_digest"]}
    materialized_digest, file_count = _materialized_tree_digest(repo, protected)
    return {
        "status": "PASS", "archive_ref": ORIGINAL_Y2_REF, "commit": ORIGINAL_Y2_COMMIT,
        "git_tree_digest": _git_tree_digest(repo, ORIGINAL_Y2_COMMIT, protected),
        "materialized_files_digest": materialized_digest, "materialized_file_count": file_count,
        "completion": repo_file_record(repo, y2.ORIGINAL_ROOT / "completion/y2.json"),
        "failure": repo_file_record(repo, y2.ORIGINAL_ROOT / "failure.json"),
        "whole_root_inventory": repo_file_record(repo, y2.ORIGINAL_ROOT / "whole_root_inventory.json"),
        "report": repo_file_record(repo, y2.ORIGINAL_REPORT_ROOT / "y2_report.md"),
        "terminal_count": 100, "terminal_records_digest": y2.canonical_digest(terminal_records),
        "failure_code": expected_failure, "no_scientific_arrays": True,
        "empty_target_identity": True, "empty_quadrature_attempts": True,
        "cohort_registry": repo_file_record(repo, y2.ORIGINAL_ROOT / "cohort_registry.json"),
        "cohort_canonical_digest": cohort["canonical_digest"],
        "selection_digest": cohort["payload"]["selection_digest"],
        "audit_subset_digest": audit["payload"]["subset_digest"],
        "perturbation_subset_digest": perturbation["payload"]["subset_digest"],
        "original_resource_totals": reconciliation["payload"]["resource_totals"],
        "original_elapsed_seconds": reconciliation["payload"]["overall_stage_wall_seconds"],
        "source_recomputed": source_recomputed,
    }


def _startup(repo: Path, *, allow_authorized_changes: bool = False) -> dict[str, Any]:
    branch = y2.git(repo, "symbolic-ref", "--short", "HEAD")
    head = y2.git(repo, "rev-parse", "HEAD")
    upstream = y2.git(repo, "rev-parse", "@{upstream}")
    remote = remote_ref(repo, f"refs/heads/{y2.EXPECTED_BRANCH}")
    require(branch == y2.EXPECTED_BRANCH and head == upstream == remote, "Y2 startup branch binding drift")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", y2.EXPECTED_BASE, head], cwd=repo).returncode == 0, "Y1-R1 is not an ancestor of Y2 startup HEAD")
    p0_ref = "refs/archive/stage-y-y2-p0-parallel-schedule-freeze"
    p0_commit = "66a0dfb889fdb0d45d9e566ed422ba9a40bd30a1"
    require(y2.git(repo, "rev-parse", p0_ref) == remote_ref(repo, p0_ref) == p0_commit, "Y2-P0 archive binding drift")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", p0_commit, head], cwd=repo).returncode == 0, "Y2-P0 is not an ancestor of Y2 startup HEAD")
    r1_ref = "refs/archive/stage-y-y2-p0-r1-elastic-schedule-freeze"
    r1_commit = "e6321a17a416b693f02f12bc0c7c329c21db5307"
    require(y2.git(repo, "rev-parse", r1_ref) == remote_ref(repo, r1_ref) == r1_commit, "Y2-P0-R1 archive binding drift")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", r1_commit, head], cwd=repo).returncode == 0, "Y2-P0-R1 is not an ancestor of startup HEAD")
    status = y2.git(repo, "status", "--porcelain")
    allowed = tuple(path.as_posix() for path in (R1_SOURCE_FILES if y2.RECOVERY_MODE else SOURCE_FILES))
    if allow_authorized_changes:
        changed = []
        repo_prefix = repo.relative_to(repo.parent).as_posix() + "/"
        for line in status.splitlines():
            path = line[2:].lstrip().split(" -> ")[-1]
            if path.startswith(repo_prefix): path = path[len(repo_prefix):]
            changed.append(path)
        require(all(path in allowed for path in changed), f"unexplained startup path: {sorted(set(changed) - set(allowed))}")
    else:
        require(not status, "Y2 startup worktree/index must be clean")
    lfs = y2.git(repo, "lfs", "status")
    if allow_authorized_changes:
        lfs_paths = []
        for line in lfs.splitlines():
            match = re.match(r"\s+(.+?) \(Git:", line)
            if match:
                path = match.group(1)
                if path.startswith(repo_prefix): path = path[len(repo_prefix):]
                lfs_paths.append(path)
        require(all(path in allowed for path in lfs_paths), f"unexplained Git LFS path: {sorted(set(lfs_paths) - set(allowed))}")
    else:
        require("(LFS:" not in lfs and "(Git:" not in lfs, "Git LFS is not clean")
    refs_that_must_be_absent = (R1_PASS_REF, R1_BLOCKED_REF) if y2.RECOVERY_MODE else ("refs/archive/stage-y-y2-clean-cohort-evidence", "refs/archive/stage-y-y2-clean-cohort-evidence-blocked")
    for ref in refs_that_must_be_absent:
        require(subprocess.run(["git", "show-ref", "--verify", "--quiet", ref], cwd=repo).returncode != 0, f"Y2 archive ref already exists: {ref}")
    return {"status": "PASS", "branch": branch, "base_commit": head, "tracking_commit": upstream, "remote_branch_commit": remote, "p0_archive_commit": p0_commit, "p0_r1_archive_commit": r1_commit, "initial_worktree_clean_before_y2_edits": not bool(status), "initial_index_clean_before_y2_edits": not any(line[0] != " " for line in status.splitlines()), "initial_lfs_clean_before_y2_edits": True, "current_authorized_y2_paths": list(allowed), "unexplained_current_paths": [], "archive_ref_absent_before_y2": True}


def _pre_selection(repo: Path) -> dict[str, Any]:
    paths = (y2.RUN_SPEC, y2.PLAN, y2.TAR_PATH, y2.LABEL_PATH, y2.DEVKIT_PATH, *y2.EXCLUSION_PATHS)
    records = [repo_file_record(repo, path) for path in paths]
    spec = y2.read_json(repo / y2.RUN_SPEC)
    require(spec["stage_token"] == y2.TOKEN and spec["selection"]["metadata_only_before_seal"] is True, "Y2 run spec drift")
    require(spec["hard_caps"]["h20_gpu_hours"] == 64.0 and spec["hard_caps"]["cpu_hours"] == 218.0, "Y2 aggregate hard-cap drift")
    require(spec["execution"]["minimum_device_count"] == 1 and spec["execution"]["maximum_device_count"] == 8, "Y2 elastic device bounds drift")
    return {"status": "PASS", "sealed_at": now(), "source_files": records, "source_bundle_digest": y2.canonical_digest(records), "allowed_selection_fields": ["dataset_release_id","class_id","source_object_id","file_sha256"], "forbidden_outcome_fields": ["prediction","correctness","raw_logit","confidence","attribution","scientific_metric","runtime","visual_appearance"], "candidate_image_decode_count": 0, "model_load_count": 0}


def seal_cohort(repo: Path) -> dict[str, Any]:
    root = repo / y2.ROOT
    report = repo / y2.REPORT_ROOT
    require(not root.exists() and not report.exists(), "Y2 formal/report root already exists")
    startup = _startup(repo)
    y0, original_y1, y1_r1 = _history_binding(repo)
    root.joinpath("completion").mkdir(parents=True)
    report.mkdir(parents=True)
    started = now()
    y2.write_json_new(root / "completion/y2.pending.json", y2.signed("stage_y_y2_pending_completion", "PENDING", {"started_at": started, "terminal_published": False, "next_authorization": "NONE_WHILE_PENDING"}))
    y2.write_json_new(root / "startup_gate.json", y2.signed("stage_y_y2_startup_gate", "PASS", startup))
    y2.write_json_new(root / "y0_binding.json", y2.signed("stage_y_y2_y0_binding", "PASS", y0))
    y2.write_json_new(root / "y1_r1_binding.json", y2.signed("stage_y_y2_y1_r1_binding", "PASS", {"original_y1": original_y1, "y1_r1": y1_r1}))
    access_policy = {"status": "PASS", "before_cohort_seal": {"candidate_raw_byte_sha256": "ALLOWED", "image_decode": "FORBIDDEN", "model_or_checkpoint_load": "FORBIDDEN", "outcome_access": "FORBIDDEN"}, "after_cohort_seal": {"decoded_source_objects": "exactly cohort_registry records", "conditions": ["clean"], "device_policy": "lock 1..8 eligible idle NVIDIA H20 devices once at startup", "one_process_per_physical_gpu": True, "worker_internal_device": "cuda:0"}, "display_extensions_forbidden": [".png",".svg",".pdf",".jpg",".jpeg",".JPEG"], "later_stages_forbidden": ["Y3","Y4","Y5","F3-Final","Stage G"]}
    y2.write_json_new(root / "access_policy.json", y2.signed("stage_y_y2_access_policy", "PASS", access_policy))
    implementation_files = [repo_file_record(repo, path) for path in SOURCE_FILES]
    y2.write_json_new(root / "implementation_seal.json", y2.signed("stage_y_y2_implementation_seal", "PASS", {"source_files": implementation_files, "files_digest": y2.canonical_digest({"files": implementation_files}), "frozen_before_formal_cohort_access": True}))
    pre = _pre_selection(repo)
    y2.write_json_new(root / "pre_selection_plan_seal.json", y2.signed("stage_y_y2_pre_selection_plan_seal", "PASS", pre))
    exclusion, exclusions = y2.exclusion_union(repo)
    y2.write_json_new(root / "exclusion_union.json", y2.signed("stage_y_y2_exclusion_union", "PASS", exclusion))
    candidates, metadata = y2.enumerate_candidates(repo)
    selected, proof = y2.select_cohort(candidates, exclusions)
    proof["candidate_metadata"] = metadata
    proof["exclusion_union_digest"] = exclusion["union_digest"]
    cohort = {"status": "PASS", "sealed_at": now(), "dataset_release_id": y2.RELEASE_ID, "namespace": y2.NAMESPACE, "count": 100, "distinct_true_class_count": 100, "selection_digest": proof["selection_digest"], "records": selected, "outcome_access_before_seal": False}
    audit = y2.subset_registry(selected, y2.AUDIT_NAMESPACE, 10, "audit_10")
    perturbation = y2.subset_registry(selected, y2.PERTURBATION_NAMESPACE, 20, "perturbation_20")
    y2.write_json_new(root / "cohort_registry.json", y2.signed("stage_y_y2_cohort_registry", "PASS", cohort))
    y2.write_json_new(root / "cohort_selection_proof.json", y2.signed("stage_y_y2_cohort_selection_proof", "PASS", proof))
    y2.write_json_new(root / "audit_subset_registry.json", y2.signed("stage_y_y2_audit_subset_registry", "PASS", audit))
    y2.write_json_new(root / "perturbation_subset_registry.json", y2.signed("stage_y_y2_perturbation_subset_registry", "PASS", perturbation))
    seals = [repo_file_record(repo, path) for path in (y2.RUN_SPEC, y2.PLAN, y2.ROOT/"implementation_seal.json", y2.ROOT/"pre_selection_plan_seal.json", y2.ROOT/"exclusion_union.json", y2.ROOT/"cohort_registry.json", y2.ROOT/"cohort_selection_proof.json", y2.ROOT/"audit_subset_registry.json", y2.ROOT/"perturbation_subset_registry.json")]
    pre_execution = {"status": "PASS", "sealed_at": now(), "sealed_before_any_image_decode_or_model_load": True, "candidate_image_decode_count": 0, "model_or_checkpoint_load_count": 0, "scientific_outcome_access_count": 0, "sealed_files": seals, "seal_bundle_digest": y2.canonical_digest(seals), "authorized_next_action_within_y2": "LOCK_ELIGIBLE_IDLE_H20_ROSTER"}
    y2.write_json_new(root / "pre_execution_plan_seal.json", y2.signed("stage_y_y2_pre_execution_plan_seal", "PASS", pre_execution))
    access = {"status": "PASS", "phase": "COHORT_SEALED_PRE_EXECUTION", "candidate_raw_byte_hash_count": 50000, "candidate_raw_bytes_hashed": metadata["raw_bytes_hashed"], "image_decode_count": 0, "decoded_source_object_ids": [], "model_load_count": 0, "checkpoint_load_count": 0, "gpu_access_count": 0, "download_bytes": 0, "empirical_conditions": [], "later_stages_started": []}
    y2.write_json_new(root / "access_ledger.json", y2.signed("stage_y_y2_access_ledger", "PASS", access))
    for relative in ("implementation_seal.json", "pre_selection_plan_seal.json", "exclusion_union.json", "cohort_registry.json", "cohort_selection_proof.json", "audit_subset_registry.json", "perturbation_subset_registry.json", "pre_execution_plan_seal.json"):
        (root / relative).chmod(0o444)
    descriptor = os.open(root, os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)
    return {"status": "PASS", "phase": "COHORT_SEALED_PRE_EXECUTION", "cohort_count": 100, "selection_digest": proof["selection_digest"], "audit_subset_digest": audit["subset_digest"], "perturbation_subset_digest": perturbation["subset_digest"], "next_action": "validate-startup"}


def _load_recovery_validator(repo: Path):
    path = repo / "scripts/validate_stage_y_y2_r1.py"
    spec = importlib.util.spec_from_file_location("stage_y_y2_r1_validator", path)
    require(spec is not None and spec.loader is not None, "cannot load Y2-R1 validator")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def recovery_preflight(repo: Path) -> dict[str, Any]:
    y2.activate_recovery_contract()
    root = repo / y2.ROOT; report = repo / y2.REPORT_ROOT
    require(not root.exists() and not report.exists(), "Y2-R1 formal/report root already exists")
    startup = _startup(repo, allow_authorized_changes=True)
    original = _validate_original_y2(repo, recompute_release=True)
    validator = _load_recovery_validator(repo)
    focused = validator.run_focused_tests(repo)
    false_pass = validator.false_pass_audit(repo)
    require(focused["status"] == false_pass["status"] == "PASS", "Y2-R1 focused tests or false-pass audit failed")
    checkpoint_sha = y2.sha_file(repo / Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"))
    require(checkpoint_sha == "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d", "checkpoint identity drift")
    synthetic_probe_rows = _probe_devices(repo, _eligible_gpu_rows(repo), checkpoint_sha)
    synthetic_probe = {
        "status": "PASS", "temporary_output_only": True,
        "physical_indices": [row["physical_index"] for row in synthetic_probe_rows],
        "numeric_probe_digest": synthetic_probe_rows[0]["numeric_probe_digest"],
        "environment_digest": synthetic_probe_rows[0]["environment_digest"],
        "python_executable_realpath": os.path.realpath(sys.executable),
        "cublas_workspace_config": y2.REQUIRED_CUBLAS_WORKSPACE_CONFIG,
        "cublas_matmul": True, "backward": True, "exact_repeat": True,
        "cross_device_numeric_parity": True,
    }
    root.joinpath("completion").mkdir(parents=True)
    report.mkdir(parents=True)
    y2.write_json_new(root / "completion/y2_r1.pending.json", y2.signed(f"{y2.DOCUMENT_PREFIX}_pending_completion", "PENDING", {"started_at": now(), "terminal_published": False, "next_authorization": "NONE_WHILE_PENDING"}))
    y2.write_json_new(root / "startup_gate.json", y2.signed(f"{y2.DOCUMENT_PREFIX}_startup_gate", "PASS", startup))
    y2.write_json_new(root / "original_y2_immutable_binding.json", y2.signed(f"{y2.DOCUMENT_PREFIX}_original_y2_immutable_binding", "PASS", original))
    reuse_paths = ("cohort_registry.json", "cohort_selection_proof.json", "audit_subset_registry.json", "perturbation_subset_registry.json", "exclusion_union.json")
    reuse = {
        "source_root": y2.ORIGINAL_ROOT.as_posix(), "copied": False,
        "artifacts": [repo_file_record(repo, y2.ORIGINAL_ROOT / name) for name in reuse_paths],
        "cohort_canonical_digest": original["cohort_canonical_digest"],
        "selection_digest": original["selection_digest"],
        "audit_subset_digest": original["audit_subset_digest"],
        "perturbation_subset_digest": original["perturbation_subset_digest"],
        "source_recomputed_from_official_release": True, "scientific_scope_changed": False,
    }
    y2.write_json_new(root / "cohort_reuse_binding.json", y2.signed(f"{y2.DOCUMENT_PREFIX}_cohort_reuse_binding", "PASS", reuse))
    implementation_files = [repo_file_record(repo, path) for path in R1_SOURCE_FILES]
    implementation = {"source_files": implementation_files, "files_digest": y2.canonical_digest({"files": implementation_files}), "frozen_before_formal_retry_access": True, "original_y2_source_snapshot_changed": False}
    y2.write_json_new(root / "implementation_seal.json", y2.signed(f"{y2.DOCUMENT_PREFIX}_implementation_seal", "PASS", implementation))
    preflight = {
        "status": "PASS", "completed_at": now(), "original_y2_binding": repo_file_record(repo, y2.ROOT / "original_y2_immutable_binding.json"),
        "cohort_reuse_binding": repo_file_record(repo, y2.ROOT / "cohort_reuse_binding.json"),
        "implementation_seal": repo_file_record(repo, y2.ROOT / "implementation_seal.json"),
        "focused_tests": focused, "false_pass_audit": false_pass,
        "synthetic_cublas_probe": synthetic_probe, "formal_worker_started": False,
        "authorized_next_action_within_y2_r1": "VALIDATE_STARTUP_CUBLAS_PROBE",
    }
    y2.write_json_new(root / "recovery_preflight.json", y2.signed(f"{y2.DOCUMENT_PREFIX}_recovery_preflight", "PASS", preflight))
    return {"status": "PASS", "original_terminal_count": original["terminal_count"], "cohort_canonical_digest": original["cohort_canonical_digest"], "selection_digest": original["selection_digest"], "focused_test_count": focused["test_count"], "false_pass_attack_count": false_pass["attack_count"], "next_action": "validate-startup"}


def build_shards(repo: Path) -> dict[str, Any]:
    formal = repo / y2.ROOT
    cohort_path = repo / y2.ORIGINAL_ROOT / "cohort_registry.json" if y2.RECOVERY_MODE else formal / "cohort_registry.json"
    cohort = y2.read_json(cohort_path)
    gate = y2.read_json(formal / "multi_gpu_startup_gate.json"); y2.validate_signed(gate)
    worker_count = y2.validate_worker_count(gate["payload"]["locked_worker_count"])
    manifests = y2.build_shard_manifests(cohort, formal, worker_count, gate["payload"]["resource_projection"])
    return {"status": "PASS", "shard_count": len(manifests), "shard_unit_counts": [row["payload"]["unit_count"] for row in manifests], "next_action": "run-shard --shard-id N"}


def _device_probe_script() -> str:
    return """import hashlib,json,os,platform,sys
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 before Torch import')
import torch,torchvision,timm
torch.cuda.set_device(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
def run(dtype):
    a=torch.linspace(-1.0,1.0,4096,dtype=dtype,device='cuda:0').reshape(64,64).requires_grad_(True)
    b=torch.linspace(0.75,-0.25,4096,dtype=dtype,device='cuda:0').reshape(64,64).requires_grad_(True)
    out=torch.matmul(a,b)
    loss=out.square().sum()
    loss.backward()
    return tuple(t.detach().cpu().contiguous() for t in (out,a.grad,b.grad))
first=[run(torch.float32),run(torch.float64)]
second=[run(torch.float32),run(torch.float64)]
if not all(torch.equal(x,y) for left,right in zip(first,second) for x,y in zip(left,right)):
    raise RuntimeError('repeated deterministic cuBLAS matmul/backward was not exact')
h=hashlib.sha256()
for group in first:
    for tensor in group:
        h.update(str(tensor.dtype).encode());h.update(b'\\0');h.update(str(tuple(tensor.shape)).encode());h.update(b'\\0');h.update(tensor.numpy().tobytes(order='C'))
p=torch.cuda.get_device_properties(0)
print(json.dumps({'python_executable_realpath':os.path.realpath(sys.executable),'python_version':platform.python_version(),'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,'timm_version':timm.__version__,'cuda_version':torch.version.cuda,'device_name':torch.cuda.get_device_name(0),'total_memory_bytes':p.total_memory,'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),'deterministic_warn_only_enabled':torch.is_deterministic_algorithms_warn_only_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'numeric_probe_digest':h.hexdigest(),'probe_operations':['torch.matmul_float32','torch.matmul_float64','backward','exact_repeat']}))"""


def _eligible_gpu_rows(repo: Path) -> list[dict[str, Any]]:
    process = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"], cwd=repo, text=True, capture_output=True)
    if process.returncode != 0: raise y2.StageYY2ScheduleError("nvidia-smi unavailable: " + process.stderr.strip())
    rows = []
    for line in process.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        require(len(values) == 7, "malformed nvidia-smi row")
        rows.append({"physical_index": int(values[0]), "device_name": values[1], "gpu_uuid": values[2], "pci_bus_id": values[3], "total_memory_mib": int(values[4]), "free_memory_mib": int(values[5]), "driver": values[6]})
    rows.sort(key=lambda row: row["physical_index"])
    compute = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], cwd=repo, text=True, capture_output=True)
    if compute.returncode != 0: raise y2.StageYY2ScheduleError("cannot query active GPU compute processes: " + compute.stderr.strip())
    busy_uuids = {line.split(",", 1)[0].strip() for line in compute.stdout.splitlines() if line.strip() and not line.lstrip().startswith("No running")}
    return y2.select_idle_h20_devices(rows, busy_uuids)


def _probe_devices(repo: Path, rows: list[dict[str, Any]], checkpoint_sha: str) -> list[dict[str, Any]]:
    probes = []
    for row in rows:
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(row["physical_index"])
        env["CUBLAS_WORKSPACE_CONFIG"] = y2.REQUIRED_CUBLAS_WORKSPACE_CONFIG
        probe = subprocess.run([sys.executable, "-c", _device_probe_script()], cwd=repo, env=env, text=True, capture_output=True)
        if probe.returncode != 0: raise y2.StageYY2ScheduleError(f"device probe failed for {row['physical_index']}: {probe.stderr.strip()}")
        value = json.loads(probe.stdout); require(value["device_name"] == "NVIDIA H20", "probe device-name drift")
        require(int(value.pop("total_memory_bytes")) >= 21 * 1024 ** 3, "probe memory below 21 GiB")
        numeric = value.pop("numeric_probe_digest")
        operations = value.pop("probe_operations")
        require(operations == ["torch.matmul_float32", "torch.matmul_float64", "backward", "exact_repeat"], "cuBLAS probe coverage drift")
        identity = {**value, "driver": row["driver"], "checkpoint_sha256": checkpoint_sha, "numeric_contract_digest": "stage-y-y1-grouped-jvp-vjp-frozen-v1"}
        require(identity["python_executable_realpath"] == os.path.realpath(sys.executable), "probe Python executable differs from launcher")
        row["environment_identity"] = identity
        row["environment_digest"] = y2.environment_digest(identity)
        row["numeric_probe_digest"] = numeric
        row["probe_operations"] = operations
        probes.append(identity)
    require(len({y2.canonical_digest(item) for item in probes}) == 1 and len({row["numeric_probe_digest"] for row in rows}) == 1, "cross-device environment or numeric parity failure")
    return rows


def validate_startup(repo: Path) -> dict[str, Any]:
    """Future formal gate. This is deliberately never called by P0 tests."""
    formal = repo / y2.ROOT
    cohort_marker = formal / "cohort_reuse_binding.json" if y2.RECOVERY_MODE else formal / "cohort_registry.json"
    require(cohort_marker.is_file(), "cohort is not sealed")
    require(not (formal / "multi_gpu_startup_gate.json").exists(), "startup roster is already locked")
    require(not list((formal / "shards").glob("shard-*/units/unit-*/completion/unit.json")), "Y2 unit outcome already exists")
    if y2.RECOVERY_MODE:
        preflight = y2.read_json(formal / "recovery_preflight.json"); y2.validate_signed(preflight)
        require(preflight["status"] == "PASS" and preflight["payload"]["formal_worker_started"] is False, "Y2-R1 recovery preflight is not PASS")
    ps = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True)
    stale = []
    for line in ps.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and int(parts[0]) != os.getpid() and re.search(r"stage_y_y2_worker|run_stage_y_y2.py\s+(run-shard|run-all)", parts[1]): stale.append({"pid": int(parts[0]), "command": parts[1]})
    require(not stale, "stale Y2 worker/PID exists")
    rows = _eligible_gpu_rows(repo)
    worker_count = y2.validate_worker_count(len(rows))
    require(len({row["gpu_uuid"] for row in rows}) == len({row["pci_bus_id"] for row in rows}) == worker_count, "GPU UUID/PCI identities not unique")
    checkpoint_sha = y2.sha_file(repo / Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"))
    require(checkpoint_sha == "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d", "checkpoint identity drift")
    rows = _probe_devices(repo, rows, checkpoint_sha)
    ledger = y2.read_json(repo / Path("evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/conservative_budget_receipt.json"))
    projection = y2.schedule_from_source_ledger(ledger, worker_count)["y2"]
    gate = y2.signed(f"{y2.DOCUMENT_PREFIX}_multi_gpu_startup_gate", "PASS", {"locked_worker_count": worker_count, "selection_policy": "all eligible idle H20 devices in ascending physical-index order, capped at eight", "devices": rows, "checkpoint_sha256": checkpoint_sha, "model_identity": "vit_base_patch16_224.augreg2_in21k_ft_in1k", "numeric_contract_digest": "stage-y-y1-grouped-jvp-vjp-frozen-v1", "cross_device_numeric_parity": True, "probe_performed_cublas_matmul": True, "probe_performed_backward": True, "probe_exact_repeat": True, "python_executable_realpath": os.path.realpath(sys.executable), "cublas_workspace_config": y2.REQUIRED_CUBLAS_WORKSPACE_CONFIG, "isolated_shard_roots": True, "shared_mutable_output": False, "dynamic_rebalancing": False, "stale_worker_pids": stale, "formal_unit_outcome_count_before_launch": 0, "resource_projection": projection, "started_unix_ns": __import__("time").time_ns()})
    y2.write_json_new(formal / "multi_gpu_startup_gate.json", gate)
    return {"status": "PASS", "locked_worker_count": worker_count, "physical_indices": [row["physical_index"] for row in rows], "next_action": "build-shards"}


def run_shard(repo: Path, shard_id: int) -> dict[str, Any]:
    gate = y2.read_json(repo / y2.ROOT / "multi_gpu_startup_gate.json"); y2.validate_signed(gate)
    worker_count = y2.validate_worker_count(gate["payload"]["locked_worker_count"])
    require(shard_id in y2.shard_ids(worker_count), "invalid shard id")
    physical = gate["payload"]["devices"][shard_id]["physical_index"]
    pending = repo / y2.ROOT / "shards" / f"shard-{shard_id:02d}" / "completion/shard.pending.json"
    y2.write_json_new(pending, y2.signed(f"{y2.DOCUMENT_PREFIX}_shard_pending", "PENDING", {"shard_id": shard_id, "shard_count": worker_count, "launcher_pid": os.getpid(), "terminal_published": False}))
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(physical)
    env["CUBLAS_WORKSPACE_CONFIG"] = y2.REQUIRED_CUBLAS_WORKSPACE_CONFIG
    if y2.RECOVERY_MODE: env["STAGE_Y2_R1_RECOVERY"] = "1"
    env["STAGE_Y2_LAUNCH_AUTHORIZATION"] = gate["canonical_digest"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(repo / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    process = subprocess.run([sys.executable, "-m", "vitnns.stage_y_y2_worker", "--repo-root", str(repo), "--shard-id", str(shard_id)], cwd=repo, env=env, text=True, capture_output=True)
    if process.returncode != 0:
        _publish_failed_shard(repo, shard_id, worker_count, process.stdout.strip(), process.stderr.strip())
        raise y2.StageYY2InfrastructureError(process.stdout.strip() or process.stderr.strip() or "shard worker failed")
    return json.loads(process.stdout)


def _publish_failed_shard(repo: Path, shard_id: int, worker_count: int, stdout: str, stderr: str) -> None:
    shard_root = repo / y2.ROOT / "shards" / f"shard-{shard_id:02d}"
    pending = shard_root / "completion/shard.pending.json"; terminal = shard_root / "completion/shard.json"
    if not pending.is_file() or terminal.exists():
        return
    counts = {state: 0 for state in y2.UNIT_TERMINAL_STATES}
    for path in shard_root.glob("units/unit-*/completion/unit.json"):
        document = y2.read_json(path); counts[document["status"]] += 1
    manifest = y2.read_json(shard_root / "shard_manifest.json")
    identity_path = shard_root / "worker_identity.json"
    identity = y2.read_json(identity_path)["payload"] if identity_path.is_file() else None
    failure = y2.signed(f"{y2.DOCUMENT_PREFIX}_shard_terminal", "BLOCKED_INFRASTRUCTURE", {"shard_id": shard_id, "shard_count": worker_count, "unit_count": sum(counts.values()), "expected_unit_count": manifest["payload"]["unit_count"], "terminal_counts": counts, "worker_identity": identity, "resource_caps": manifest["payload"]["resource_caps"], "stdout": stdout, "stderr": stderr, "failure_code": "WORKER_PROCESS_NONZERO_EXIT", "completed_unix_ns": __import__("time").time_ns()})
    y2.publish_terminal(pending, terminal, failure)


def run_all(repo: Path) -> dict[str, Any]:
    formal = repo / y2.ROOT
    gate = y2.read_json(formal / "multi_gpu_startup_gate.json"); y2.validate_signed(gate)
    worker_count = y2.validate_worker_count(gate["payload"]["locked_worker_count"])
    require(all((formal / "shards" / f"shard-{shard:02d}/shard_manifest.json").is_file() for shard in y2.shard_ids(worker_count)), "locked shard manifest roster incomplete")
    require(not (formal / "worker_launch_roster.json").exists() and not (formal / "worker_launch_results.json").exists(), "formal worker launch already recorded")
    commands = []
    for shard in y2.shard_ids(worker_count):
        physical = gate["payload"]["devices"][shard]["physical_index"]
        commands.append({"shard_id": shard, "physical_index": physical, "cuda_visible_devices": str(physical)})
    y2.write_json_new(formal / "worker_launch_roster.json", y2.signed(f"{y2.DOCUMENT_PREFIX}_worker_launch_roster", "PASS", {"locked_worker_count": worker_count, "commands": commands, "concurrent_launch": True, "dynamic_rebalancing": False}))
    for row in commands:
        pending = formal / "shards" / f"shard-{row['shard_id']:02d}" / "completion/shard.pending.json"
        y2.write_json_new(pending, y2.signed(f"{y2.DOCUMENT_PREFIX}_shard_pending", "PENDING", {"shard_id": row["shard_id"], "shard_count": worker_count, "launcher_pid": os.getpid(), "terminal_published": False}))
    processes = []
    try:
        for row in commands:
            env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = row["cuda_visible_devices"]
            env["CUBLAS_WORKSPACE_CONFIG"] = y2.REQUIRED_CUBLAS_WORKSPACE_CONFIG
            if y2.RECOVERY_MODE: env["STAGE_Y2_R1_RECOVERY"] = "1"
            env["STAGE_Y2_LAUNCH_AUTHORIZATION"] = gate["canonical_digest"]
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = str(repo / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            process = subprocess.Popen([sys.executable, "-m", "vitnns.stage_y_y2_worker", "--repo-root", str(repo), "--shard-id", str(row["shard_id"])], cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append((row, process))
    except Exception as error:
        for _row, process in processes:
            if process.poll() is None:
                process.terminate()
        for _row, process in processes:
            process.communicate()
        for row in commands:
            _publish_failed_shard(repo, row["shard_id"], worker_count, "", f"launcher spawn failure: {type(error).__name__}:{error}")
        raise y2.StageYY2InfrastructureError("worker fleet launch failed before the locked roster was fully started") from error
    results = []
    for row, process in processes:
        stdout, stderr = process.communicate()
        results.append({**row, "pid": process.pid, "returncode": process.returncode, "stdout": stdout.strip(), "stderr": stderr.strip()})
        if process.returncode != 0:
            _publish_failed_shard(repo, row["shard_id"], worker_count, stdout.strip(), stderr.strip())
    status = "PASS" if all(row["returncode"] == 0 for row in results) else "BLOCKED_INFRASTRUCTURE"
    y2.write_json_new(formal / "worker_launch_results.json", y2.signed(f"{y2.DOCUMENT_PREFIX}_worker_launch_results", status, {"locked_worker_count": worker_count, "results": results, "all_workers_joined": True, "dynamic_rebalancing": False}))
    if status != "PASS":
        raise y2.StageYY2InfrastructureError("one or more locked shard workers failed; inspect worker_launch_results.json")
    return {"status": status, "locked_worker_count": worker_count, "next_action": "reconcile"}


def reconcile(repo: Path) -> dict[str, Any]:
    result = y2.reconcile_shards(repo / y2.ROOT)
    y2.write_json_new(repo / y2.ROOT / "reconciliation.json", result)
    return {"status": result["status"], **result["payload"]}


def _seal_terminal(repo: Path, status: str, reconciliation: Mapping[str, Any] | None, failure: Mapping[str, Any] | None) -> dict[str, Any]:
    root = repo / y2.ROOT; report_root = repo / y2.REPORT_ROOT
    pending_name = "y2_r1.pending.json" if y2.RECOVERY_MODE else "y2.pending.json"
    terminal_name = "y2_r1.json" if y2.RECOVERY_MODE else "y2.json"
    pending = root / "completion" / pending_name
    require(pending.is_file() and not (root / "completion" / terminal_name).exists(), "Y2 terminal transition is unavailable")
    valid_count = 0 if reconciliation is None else int(reconciliation["payload"]["valid_count"])
    outcome_count = len(list(root.glob("shards/shard-*/units/unit-*/completion/unit.json")))
    if failure is not None:
        y2.write_json_new(root / "failure.json", y2.signed(f"{y2.DOCUMENT_PREFIX}_failure", status, failure))
    summary = y2.signed(f"{y2.DOCUMENT_PREFIX}_terminal_access_summary", status, {
        "formal_unit_terminal_count": outcome_count,
        "formal_outcome_access": outcome_count > 0,
        "conditions_observed": ["clean"] if outcome_count else [],
        "non_clean_condition_count": 0,
        "bootstrap_executed": False, "holm_executed": False,
        "claim_decision_executed": False, "later_stages_started": [],
    })
    y2.write_json_new(root / "terminal_access_summary.json", summary)
    report_root.mkdir(parents=True, exist_ok=True)
    report_name = "y2_r1_report.md" if y2.RECOVERY_MODE else "y2_report.md"
    report_path = report_root / report_name
    report_text = (f"# Stage {'Y2-R1' if y2.RECOVERY_MODE else 'Y2'} terminal report\n\nStatus: **{status}**. Valid clean units: {valid_count}. "
                   f"Formal unit terminals: {outcome_count}. No non-clean condition, bootstrap, Holm procedure, or claim decision ran.\n\n"
                   f"Next authorization: `{'STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY' if status == 'PASS' else 'NONE'}`.\n")
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(report_text); stream.flush(); os.fsync(stream.fileno())
    inventory = y2.whole_root_inventory(root, status)
    y2.write_json_new(root / "whole_root_inventory.json", inventory)
    completion = y2.signed(f"{y2.DOCUMENT_PREFIX}_terminal_completion", status, {
        "terminal": True,
        "reconciliation": None if reconciliation is None else repo_file_record(repo, y2.ROOT / "reconciliation.json"),
        "failure": None if failure is None else repo_file_record(repo, y2.ROOT / "failure.json"),
        "whole_root_inventory": repo_file_record(repo, y2.ROOT / "whole_root_inventory.json"),
        "terminal_access_summary": repo_file_record(repo, y2.ROOT / "terminal_access_summary.json"),
        "report": repo_file_record(repo, y2.REPORT_ROOT / report_name),
        "valid_clean_count": valid_count,
        "formal_outcome_access": outcome_count > 0,
        "conditions": ["clean"],
        "bootstrap_executed": False, "holm_executed": False,
        "claim_decisions": [],
        "next_authorization": "STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY" if status == "PASS" else "NONE",
    })
    y2.publish_terminal(pending, root / "completion" / terminal_name, completion)
    return {"status": status, "valid_clean_count": valid_count, "next_authorization": completion["payload"]["next_authorization"]}


def seal_control_failure(repo: Path, error: Exception, phase: str) -> dict[str, Any]:
    status = getattr(error, "code", "BLOCKED_INFRASTRUCTURE")
    if status not in ("BLOCKED_COHORT", "BLOCKED_PROVENANCE", "BLOCKED_RESOURCE_SCHEDULE", "BLOCKED_INFRASTRUCTURE", "BLOCKED_BUDGET", "BLOCKED_SCIENTIFIC_VALIDITY"):
        status = "BLOCKED_INFRASTRUCTURE"
    failure = {"phase": phase, "failure_type": type(error).__name__, "message": str(error), "recovery_requires_new_root": True, "next_authorization": "NONE"}
    return _seal_terminal(repo, status, None, failure)


def seal(repo: Path) -> dict[str, Any]:
    reconciliation = y2.read_json(repo / y2.ROOT / "reconciliation.json"); y2.validate_signed(reconciliation)
    pending_name = "y2_r1.pending.json" if y2.RECOVERY_MODE else "y2.pending.json"
    pending = repo / y2.ROOT / "completion" / pending_name
    status = reconciliation["status"]
    require(status in ("PASS", "BLOCKED_BUDGET", "BLOCKED_INFRASTRUCTURE", "BLOCKED_SCIENTIFIC_VALIDITY"), "unsupported reconciliation terminal")
    failure = None if status == "PASS" else {"phase": "RECONCILIATION", "failure_type": status, "message": "source-recomputed reconciliation did not pass", "recovery_requires_new_root": True, "next_authorization": "NONE"}
    return _seal_terminal(repo, status, reconciliation, failure)


def validate(repo: Path, publication: bool = False) -> dict[str, Any]:
    path = repo / ("scripts/validate_stage_y_y2_r1.py" if y2.RECOVERY_MODE else "scripts/validate_stage_y_y2.py")
    spec = importlib.util.spec_from_file_location("stage_y_y2_formal_validator", path)
    require(spec is not None and spec.loader is not None, "cannot load formal Y2 validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(repo, publication=publication, allow_blocked=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal-cohort", "recovery-preflight", "validate-startup", "build-shards", "run-shard", "run-all", "reconcile", "seal", "validate", "validate-publication"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--recovery-r1", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    if args.recovery_r1 or args.command == "recovery-preflight":
        y2.activate_recovery_contract()
    try:
        if args.command == "seal-cohort": result = seal_cohort(repo)
        elif args.command == "recovery-preflight": result = recovery_preflight(repo)
        elif args.command == "build-shards": result = build_shards(repo)
        elif args.command == "validate-startup": result = validate_startup(repo)
        elif args.command == "run-shard":
            require(args.shard_id is not None, "run-shard requires --shard-id"); result = run_shard(repo, args.shard_id)
        elif args.command == "run-all": result = run_all(repo)
        elif args.command == "reconcile": result = reconcile(repo)
        elif args.command == "seal": result = seal(repo)
        else: result = validate(repo, args.command == "validate-publication")
    except Exception as error:
        pending_name = "y2_r1.pending.json" if y2.RECOVERY_MODE else "y2.pending.json"
        terminal_name = "y2_r1.json" if y2.RECOVERY_MODE else "y2.json"
        pending = repo / y2.ROOT / "completion" / pending_name
        terminal = repo / y2.ROOT / "completion" / terminal_name
        if args.command not in ("validate", "validate-publication") and pending.is_file() and not terminal.exists():
            try:
                blocked = seal_control_failure(repo, error, args.command)
                print(json.dumps({**blocked, "error": str(error)}, sort_keys=True))
                return 2
            except Exception as seal_error:
                print(json.dumps({"status": "BLOCKED_INFRASTRUCTURE", "error": str(error), "terminal_seal_error": str(seal_error), "next_authorization": "NONE"}, sort_keys=True))
                return 2
        print(json.dumps({"status": getattr(error, "code", "BLOCKED"), "error": str(error), "next_authorization": "NONE"}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
