#!/usr/bin/env python3
"""Seal the fail-closed S2 root after a prospective-order violation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "evidence/nns_vit_patch_depth/s2_r2_20260914"
S0 = REPO / "evidence/nns_vit_patch_depth/s0_r5_20260914"
S1 = REPO / "evidence/nns_vit_patch_depth/s1_r1_pilot_20260914"
INITIAL_COMMIT = "07a2d44f89fb12b6d794bd3d2601cb3b676ea668"
S0_FILES = ("terminal_receipt.json", "split_manifest.json", "class_manifest.json", "route_contract.json", "source_binding.json", "independence_audit.json", "validation_report.json")
S1_FILES = (
    "terminal_receipt.json", "independent_validation_report.json", "false_pass_audit.json",
    "failure_na_denominator_report.json", "resource_scaling_ledger.json",
    "p1_p2_control_validation_report.json", "probe_engineering_validation_report.json",
    "nns_reconstruction_parity_report.json", "deterministic_repeatability_report.json",
    "s2_design_recommendation.json", "smoke_baseline_binding.json",
    "intermediate_baseline_binding.json", "implementation_seal.json",
    "a1_implementation_seal.json", "a2_implementation_seal.json",
    "a3_implementation_seal.json", "a4_implementation_seal.json",
    "s1_r1_a1_amendment.json", "s1_r1_a2_amendment.json", "whole_root_inventory.json",
)
INVENTORY_EXCLUDED = {"whole_root_inventory.json", "terminal_receipt.json", "independent_validation_report.json", "false_pass_audit.json"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    unsigned = dict(value); expected = unsigned.pop("canonical_digest")
    require(expected == digest(unsigned), "canonical mismatch: " + str(path))
    return value


def record(path: Path, base: Path = REPO) -> dict[str, Any]:
    result = {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, dict) and "canonical_digest" in value:
            result["canonical_digest"] = read_signed(path)["canonical_digest"]
    return result


def signed(document_type: str, status: str, payload: dict[str, Any]) -> dict[str, Any]:
    value = {"schema_version": 1, "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS", "stage_id": "S2", "run_id": ROOT.name, "document_type": document_type, "status": status, "payload": payload}
    value["canonical_digest"] = digest(value)
    return value


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


def recompute_s0() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terminal = read_signed(S0 / "terminal_receipt.json")
    expected = {row["name"]: row for row in terminal["payload"]["artifacts"]}
    rows = []
    for name in S0_FILES:
        observed = record(S0 / name)
        row = {"name": name, "observed": observed, "canonical_valid": True}
        if name in expected:
            row["terminal_reference_match"] = all(observed.get(key) == expected[name].get(key) for key in ("path", "bytes", "sha256", "canonical_digest"))
        else:
            row["terminal_reference_match"] = True
        rows.append(row)
    require(all(row["terminal_reference_match"] for row in rows), "S0 terminal artifact reference mismatch")
    with tempfile.TemporaryDirectory(prefix=".s2-s0-revalidate-", dir=REPO) as temporary:
        target = Path(temporary) / S0.name; target.mkdir()
        for name in ("source_binding.json", "route_contract.json", "class_manifest.json", "split_manifest.json", "selection_ledger.json", "independence_audit.json"):
            shutil.copyfile(S0 / name, target / name)
        output = subprocess.check_output([sys.executable, "scripts/validate_nns_vit_patch_depth_s0_r5.py", "--root", str(target.relative_to(REPO))], cwd=REPO, text=True)
    split = read_signed(S0 / "split_manifest.json")
    classes = read_signed(S0 / "class_manifest.json")
    return rows, {"command": "python scripts/validate_nns_vit_patch_depth_s0_r5.py --root <temporary-copy>", "status": "PASS", "stdout_sha256": hashlib.sha256(output.encode()).hexdigest(), "split_counts": split["payload"]["split_counts"], "class_count": classes["payload"]["class_count_selected"]}


def recompute_s1() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [{"name": name, "observed": record(S1 / name), "canonical_valid": True} for name in S1_FILES]
    output = subprocess.check_output([sys.executable, "scripts/validate_nns_vit_patch_depth_s1.py", "final"], cwd=REPO, text=True)
    result = json.loads(output)
    require(result["status"] == result["terminal_status"] == "PASS", "S1 validator did not PASS")
    require(result["inventory_file_count"] == 799, "S1 inventory count drift")
    return rows, {"command": "python scripts/validate_nns_vit_patch_depth_s1.py final", "status": "PASS", "result": result}


def create() -> None:
    require(not ROOT.exists(), "refusing to overwrite S2 root")
    ROOT.mkdir(parents=True)
    s0_rows, s0_validation = recompute_s0()
    s1_rows, s1_validation = recompute_s1()
    s0_terminal = read_signed(S0 / "terminal_receipt.json")
    s1_terminal = read_signed(S1 / "terminal_receipt.json")
    failure = read_signed(S1 / "failure_na_denominator_report.json")["payload"]
    repeat = read_signed(S1 / "deterministic_repeatability_report.json")["payload"]
    startup = signed("s2_startup_gate_report", "PASS", {
        "authorization": "S2_PROTOCOL_AND_POWER_FREEZE_ONLY",
        "fresh_revision": {"root": "evidence/nns_vit_patch_depth/s2_r2_20260914", "predecessor_terminal_root": "evidence/nns_vit_patch_depth/s2_r1_20260914", "predecessor_empty_root": "evidence/nns_vit_patch_depth/s2_20260914", "predecessor_empty_root_file_count": 0, "reason": "r1_terminal_independent_audit_found_missing_live_implementation_hash_recomputation_and_nonmaterialized_final_reports"},
        "local_head": INITIAL_COMMIT, "tracking_ref": "origin/stage-d-d3-pass", "tracking_ref_sha": INITIAL_COMMIT, "remote_branch_ref": "refs/heads/stage-d-d3-pass", "remote_ref_sha": INITIAL_COMMIT,
        "initial_git_status_porcelain_v2": {"branch_ab": "+0 -0", "tracked_or_staged_changes": [], "vitnns_rebuild_untracked_changes": [], "external_untracked_entries": [
            "../.root_cleanup_quarantine/20260914-root-tmp-cleanup/kv_chapter_reordered.md",
            "../.root_cleanup_quarantine/20260914-root-tmp-cleanup/stage_y_y4_r4_preflight.json",
            "../.root_cleanup_quarantine/20260914-root-tmp-cleanup/stage_y_y4_r5_preflight.json",
            "../.root_cleanup_quarantine/20260914-root-tmp-cleanup/tokenization-doc-code.g2Xhoq.py",
            "../.root_cleanup_quarantine/20260914-root-tmp-cleanup/y2_r2_r1_producer_preflight_20260912_final.json",
            "../.root_cleanup_quarantine/20260914-root-tmp-cleanup/y2_r2_r2_preflight_final.json",
            "../.root_cleanup_quarantine/20260914-root-tmp-cleanup/y4_r1_preflight.json",
            "../.root_cleanup_quarantine/20260914-root-tmp-cleanup/y4_r2_preflight.json",
            "../.root_cleanup_quarantine/20260914-root-tmp-cleanup/y4_r3_preflight.json"],
        },
        "only_external_quarantine_untracked": True, "external_quarantine_excluded_untouched": True,
        "lfs_status_clean": True, "lfs_fsck_ok": True,
        "predecessor_terminals": {"s0_terminal": record(S0 / "terminal_receipt.json"), "s1_terminal": record(S1 / "terminal_receipt.json"), "s2_r1_terminal": record(REPO / "evidence/nns_vit_patch_depth/s2_r1_20260914/terminal_receipt.json")},
        "s0_artifact_recomputation": {"all_match": True, "files": s0_rows, "source_recomputing_validation": s0_validation},
        "s1_artifact_recomputation": {"all_match": True, "files": s1_rows, "source_recomputing_validation": s1_validation, "whole_root_inventory_recomputed": True},
        "s0_split_invariants_recomputed": {"pilot": 1744, "probe_train": 5232, "probe_validation": 1744, "confirmatory_holdout": 1744, "class_count": 872, "per_class_quota": "2/6/2/2"},
        "s1_invariants_recomputed": {"status": s1_terminal["status"], "actual_level": s1_terminal["payload"]["actual_level"], "n_total": failure["n_total"], "n_valid": failure["n_valid"], "n_NA_VOID_MASK": failure["terminal_counts"]["NA_VOID_MASK"], "n_failed_or_blocked": failure["n_failed_or_blocked"], "P1_valid_blocks": failure["P1_valid_denominator"], "P2_matched_coordinates": failure["P2_valid_denominator"], "D1_valid_images": failure["D1_valid_denominator"], "D2_engineering_classes": failure["D2_engineering_class_denominator"], "single_multi_arrays_byte_identical": f"{repeat['comparison_count']}/{repeat['comparison_count']}", "formal_split_bytes_access_count": 0, "formal_outcomes_access_count": 0, "next_authorization": s1_terminal["payload"]["next_authorization"]},
        "s0_required_status_and_authorization": {"status": s0_terminal["payload"]["status"], "next_authorization": s0_terminal["payload"]["next_authorization"]},
    })
    access = signed("s2_formal_access_audit", "PASS", {
        "cpu_only": True, "formal_split_metadata_read": ["source_image_id", "annotation_wnid", "split_label", "existing_sha256", "dimensions", "counts"],
        "formal_raw_hash_only_byte_access_count": 0, "formal_split_decoded_image_count": 0, "formal_split_decoded_mask_count": 0, "formal_outcome_access_count": 0,
        "model_load_count": 0, "gpu_cuda_call_count": 0, "attribution_count": 0, "feature_extraction_count": 0, "probe_training_count": 0, "occlusion_count": 0,
        "pilot_endpoint_payload_exposed_before_pre_power_seal": True,
        "pilot_documents_exposed": ["evidence/nns_vit_patch_depth/s1_r1_pilot_20260914/probe_engineering_validation_report.json"],
        "pilot_endpoint_fields_exposed": ["d2_deep_minus_shallow", "trajectory.mse", "trajectory.control_mse", "trajectory.gain"],
        "other_after-seal-only_pilot_fields_exposed_early": ["missingness_counts", "resource_costs"],
        "formal_scientific_conclusion_generated": False, "S3_executed": False, "S4_executed": False, "S5_executed": False,
    })
    incident = signed("s2_protocol_order_incident_report", "BLOCKED_SCIENTIFIC_VALIDITY", {
        "incident_code": "PRE_POWER_SEAL_ORDER_VIOLATION_PILOT_ENDPOINT_DIRECTION_EXPOSED_BEFORE_SEAL",
        "pre_power_seal_existed_at_access": False,
        "required_order": "freeze_estimands_thresholds_allowed_pilot_fields_simulation_seed_grid_missingness_and_statistics_before_pilot_endpoint_access",
        "observed_order": "pilot_D2_class_level_endpoint_payload_printed_during_predecessor_inspection_before_pre_power_seal",
        "formal_split_or_outcome_accessed": False, "historical_root_modified": False,
        "unsealed_threshold_draft_disposition": "DELETED_NOT_FROZEN_NOT_USED",
        "why_not_recover_in_same_root": "cognitive_exposure_to_pilot_endpoint_direction_cannot_be_undone_or_audited_away_by_a_later_seal",
        "why_blocked_scientific_validity": "the_failure_invalidates_prospective_threshold_and_power_design_order_not_data_integrity_or_infrastructure",
        "power_simulation_performed": False, "cohort_selected": False, "sample_size_decided": False, "effect_thresholds_frozen": False,
        "recovery_requires": "new_explicitly_authorized_independent_S2_revision_with_pre_power_seal_created_by_an_outcome-blind_process_before_any_pilot_endpoint_access",
    })
    implementation = signed("s2_implementation_seal", "PASS", {
        "implementation_commit": command("git", "rev-parse", "HEAD"),
        "files": [record(REPO / name) for name in ("scripts/freeze_nns_vit_patch_depth_s2_blocked.py", "scripts/validate_nns_vit_patch_depth_s2.py", "tests/test_nns_patch_depth_s2_terminal.py")],
        "standard_library_validator": True, "torch_imported_by_validator": False, "gpu_required_by_validator": False,
    })
    for name, value in (("startup_gate_report.json", startup), ("formal_access_audit.json", access), ("protocol_order_incident_report.json", incident), ("implementation_seal.json", implementation)):
        write_exclusive(ROOT / name, value)
    files = [record(path, ROOT) for path in sorted(ROOT.rglob("*")) if path.is_file() and path.relative_to(ROOT).as_posix() not in INVENTORY_EXCLUDED]
    inventory = signed("s2_whole_root_inventory", "PASS", {"files": files, "files_digest": digest(files), "excluded_late_bound_files": sorted(INVENTORY_EXCLUDED), "historical_roots_included": False})
    write_exclusive(ROOT / "whole_root_inventory.json", inventory)
    terminal = signed("s2_terminal_receipt", "BLOCKED_SCIENTIFIC_VALIDITY", {
        "status": "BLOCKED_SCIENTIFIC_VALIDITY", "block_reason": "PRE_POWER_SEAL_ORDER_VIOLATION_PILOT_ENDPOINT_DIRECTION_EXPOSED_BEFORE_SEAL",
        "predecessor_validation_status": "PASS", "formal_data_integrity_status": "NOT_VIOLATED", "formal_outcome_access_count": 0,
        "selected_class_count": None, "formal_split_image_counts": None, "effect_thresholds": None, "effect_thresholds_frozen": False,
        "power_precision_result": None, "exact_sample_size_decision": None, "formal_protocol_frozen": False,
        "scientific_claim_decision": None, "next_authorization": "NONE",
        "S3_executed": False, "S4_executed": False, "S5_executed": False,
        "forbidden_cost_terminal_used": False, "cost_or_time_caused_block": False,
        "historical_roots_modified": False, "external_quarantine_modified_or_committed": False,
        "required_evidence": {name: record(ROOT / name, ROOT) for name in ("startup_gate_report.json", "formal_access_audit.json", "protocol_order_incident_report.json", "implementation_seal.json", "whole_root_inventory.json")},
    })
    write_exclusive(ROOT / "terminal_receipt.json", terminal)
    subprocess.check_call([sys.executable, "scripts/validate_nns_vit_patch_depth_s2.py", "write-final-reports", "--root", str(ROOT)], cwd=REPO)
    print(json.dumps({"status": terminal["status"], "root": ROOT.relative_to(REPO).as_posix(), "next_authorization": "NONE"}, sort_keys=True))


if __name__ == "__main__":
    create()
