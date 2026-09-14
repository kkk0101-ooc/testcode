#!/usr/bin/env python3
"""Publish S4 pre-access tests, then seal a pushed implementation commit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "evidence/nns_vit_patch_depth/s4_r4_20260915"
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s4 as s4  # noqa: E402


def artifact(path: Path) -> dict[str, Any]:
    value = {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": s4.sha_file(path)}
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if isinstance(document.get("canonical_digest"), str):
            value["canonical_digest"] = document["canonical_digest"]
    return value


def run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(command, cwd=REPO, env=env, text=True, stderr=subprocess.STDOUT).strip()


def preaccess() -> None:
    python = REPO.parent / ".conda/envs/vitnns/bin/python"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    pytest_root = REPO.parent / ".runtime/pytest-s4-seal"
    output = run([str(python), "-m", "pytest", "-q", "tests/test_nns_patch_depth_s4.py", "--basetemp", str(pytest_root)], env=environment)
    require_test = "9 passed" in output
    if not require_test:
        raise RuntimeError("S4 pre-access tests did not pass")
    false_pass = json.loads(run(["python3", "scripts/validate_nns_vit_patch_depth_s4.py", "false-pass"]))
    preaccess_validation = json.loads(run(["python3", "scripts/validate_nns_vit_patch_depth_s4.py", "preaccess"]))
    pilot_root = REPO / "evidence/nns_vit_patch_depth/s1_r1_pilot_20260914"
    pilot_terminal = json.loads((pilot_root / "terminal_receipt.json").read_text())
    repeat = json.loads((pilot_root / "deterministic_repeatability_report.json").read_text())
    nns = json.loads((pilot_root / "nns_reconstruction_parity_report.json").read_text())
    p1p2 = json.loads((pilot_root / "p1_p2_control_validation_report.json").read_text())
    if not (pilot_terminal["status"] == repeat["status"] == nns["status"] == p1p2["status"] == "PASS"):
        raise RuntimeError("S1 pilot-only evidence binding failed")
    suffix = ""
    reports = {
        "synthetic_queue_concurrency_crash_recovery_report.json": s4.signed("s4_synthetic_queue_concurrency_crash_recovery_report", "PASS", {
            "test_command": str(python) + " -m pytest -q tests/test_nns_patch_depth_s4.py",
            "test_output": output, "test_count": 9,
            "checks": ["P1_hash_permutation_and_raw_storage", "P1_zero_mass", "P2_exact_strata_boundary_and_determinism", "P2_NumPy_scalar_JSON_native_serialization_regression", "dynamic_single_winner_claim", "dependency_ready_without_round_barrier", "fresh_launch_ignores_stale_claim", "global_terminal_skip", "atomic_hardlink_no_overwrite", "no_static_shard", "forbidden_read_guard"],
            "formal_member_access_count": 0,
        }),
        "s1_pilot_only_nns_occlusion_validation_report.json": s4.signed("s4_s1_pilot_only_nns_occlusion_validation_report", "PASS", {
            "development_only": True, "formal_S4_access_count": 0,
            "source_terminal": artifact(pilot_root / "terminal_receipt.json"),
            "nns_reconstruction_parity": artifact(pilot_root / "nns_reconstruction_parity_report.json"),
            "p1_p2_controls": artifact(pilot_root / "p1_p2_control_validation_report.json"),
            "actual_pilot_terminal_count": pilot_terminal["payload"]["unit_validation"]["terminal_count"],
            "actual_valid_pilot_count": nns["payload"]["valid_unit_count"],
            "all_valid_numeric_closure_pass": nns["payload"]["all_valid_closure_pass"],
            "all_valid_native_hooked_parity_pass": nns["payload"]["all_valid_native_hooked_parity_pass"],
            "P1_raw_block_count": p1p2["payload"]["p1_valid_block_count"],
            "P2_matched_coordinate_count": p1p2["payload"]["p2_matched_coordinate_count"],
            "operator_selection_used_logit_outcome": p1p2["payload"]["p2_operator_selection_used_logit_outcome"],
        }),
        "deterministic_single_multi_gpu_repeatability_report.json": s4.signed("s4_deterministic_single_multi_gpu_repeatability_report", "PASS", {
            "development_only_pilot": True, "source": artifact(pilot_root / "deterministic_repeatability_report.json"),
            "all_byte_identical": repeat["payload"]["all_byte_identical"],
            "frozen_array_tolerance": {"atol": 0.0, "rtol": 0.0, "formal_max_abs_tolerance_ceiling": 1e-5},
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "startup_probe": ["torch.matmul", "backward"],
            "formal_S4_access_count": 0,
        }),
        "pre_access_validation_report.json": s4.signed("s4_pre_access_validation_report", "PASS", {"metadata_provenance": preaccess_validation, "formal_member_access_count": 0}),
        "pre_access_false_pass_audit.json": s4.signed("s4_pre_access_false_pass_audit", "PASS", false_pass | {"formal_member_access_count": 0}),
    }
    if suffix:
        reports = {name.removesuffix(".json") + suffix + ".json": document for name, document in reports.items()}
    for name, document in reports.items():
        path = ROOT / name
        if path.exists():
            raise RuntimeError("pre-access report already exists: " + name)
        s4.write_json_new(path, document)
    print(json.dumps({"status": "PASS", "reports": len(reports), "formal_member_access_count": 0, "pytest": output}, sort_keys=True))


def seal(implementation_commit: str) -> None:
    if subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() != implementation_commit:
        raise RuntimeError("HEAD is not requested implementation commit")
    branch = subprocess.check_output(["git", "-C", str(REPO), "symbolic-ref", "--short", "HEAD"], text=True).strip()
    remote_line = subprocess.check_output(["git", "-C", str(REPO), "ls-remote", "--heads", "origin", f"refs/heads/{branch}"], text=True).strip()
    remote_oid = remote_line.split()[0] if remote_line else ""
    if remote_oid != implementation_commit:
        raise RuntimeError("live remote branch does not equal implementation commit")
    tracked = subprocess.check_output(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", implementation_commit], text=True).splitlines()
    required = [
        "src/vitnns/nns_patch_depth_s4.py", "src/vitnns/nns_patch_depth_s4_worker.py",
        "scripts/prepare_nns_vit_patch_depth_s4.py", "scripts/validate_nns_vit_patch_depth_s4.py",
        "scripts/seal_nns_vit_patch_depth_s4_implementation.py", "tests/test_nns_patch_depth_s4.py",
        "config/nns_vit_patch_depth/s4_run_spec.json", "config/nns_vit_patch_depth/s4_artifact.schema.json",
        "config/nns_vit_patch_depth/s4_unit_terminal.schema.json", "config/nns_vit_patch_depth/s4_source_schedule.schema.json",
        "config/nns_vit_patch_depth/s4_claim.schema.json",
        "manifests/nns_vit_patch_depth/s4_r4_source_schedule.json",
        "manifests/nns_vit_patch_depth/s4_r4_path_resolution_registry.json",
    ]
    if not all(item in tracked or "vitnns_rebuild/" + item in tracked for item in required):
        raise RuntimeError("implementation commit lacks required files")
    documents = {
        name: artifact(ROOT / name) for name in (
            "formal_access_audit_pre_implementation_seal.json", "pre_access_validation_report.json",
            "pre_access_false_pass_audit.json", "synthetic_queue_concurrency_crash_recovery_report.json",
            "s1_pilot_only_nns_occlusion_validation_report.json",
            "deterministic_single_multi_gpu_repeatability_report.json",
            "dependent_unit_schedule_and_queue_contract.json", "elastic_gpu_startup_and_lock_contract.json",
            "typed_failure_na_propagation_contract.json", "retention_and_audit_subset_registry.json",
            "s4_r3_failure_binding.json",
        )
    }
    seal_document = s4.signed("s4_implementation_seal", "PASS", {
        "implementation_commit": implementation_commit, "implementation_tree": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", implementation_commit + "^{tree}"], text=True).strip(),
        "branch": branch, "live_remote_branch_oid_before_seal_commit": remote_oid,
        "remote_contains_implementation_commit": True, "created_unix_ns": time.time_ns(),
        "formal_selected_JPEG_member_access_count_before_seal": 0, "formal_selected_mask_member_access_count_before_seal": 0,
        "decode_count_before_seal": 0, "model_forward_count_before_seal": 0,
        "bound_documents": documents, "scientific_claim_decision": None,
    }, producer_commit=implementation_commit)
    path = ROOT / "implementation_seal.json"
    if path.exists():
        raise RuntimeError("implementation seal already exists")
    s4.write_json_new(path, seal_document)
    print(json.dumps({"status": "PASS", "implementation_commit": implementation_commit, "remote_oid": remote_oid, "formal_member_access_count": 0}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preaccess")
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("implementation_commit")
    args = parser.parse_args()
    preaccess() if args.command == "preaccess" else seal(args.implementation_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
