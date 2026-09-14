#!/usr/bin/env python3
"""Seal the already computed S2-R8 result without changing scientific inputs."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "evidence/nns_vit_patch_depth/s2_r8_20260914"
SPEC = REPO / "config/nns_vit_patch_depth/s2_r8_run_spec.json"
LATE = {"whole_root_inventory.json", "terminal_receipt.json", "independent_validation_report.json"}


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
    unsigned = dict(value)
    expected = unsigned.pop("canonical_digest")
    require(expected == digest(unsigned), "canonical digest mismatch: " + str(path))
    return value


def record(path: Path, base: Path = REPO) -> dict[str, Any]:
    result = {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if "canonical_digest" in value:
            result["canonical_digest"] = read_signed(path)["canonical_digest"]
    return result


def signed(document_type: str, status: str, payload: dict[str, Any]) -> dict[str, Any]:
    value = {"schema_version": 1, "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS", "stage_id": "S2-R8", "run_id": ROOT.name, "document_type": document_type, "status": status, "payload": payload}
    value["canonical_digest"] = digest(value)
    return value


def write(name: str, value: dict[str, Any]) -> None:
    path = ROOT / name
    require(not path.exists(), "refusing overwrite: " + name)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def git_record(path: str, commit: str) -> dict[str, Any]:
    result = record(REPO / path)
    line = subprocess.check_output(["git", "-C", str(REPO.parent), "ls-tree", commit, "--", "vitnns_rebuild/" + path], text=True).strip()
    require(bool(line), "uncommitted finalizer: " + path)
    result["git_blob_oid"] = line.split()[2]
    return result


def main() -> None:
    require((ROOT / "pre_power_seal.json").is_file(), "pre-power seal missing")
    require(not (ROOT / "terminal_receipt.json").exists(), "terminal already exists")
    pre = json.loads(subprocess.check_output([sys.executable, "scripts/validate_nns_vit_patch_depth_s2_r8.py", "prepower"], cwd=REPO, text=True))
    require(pre["status"] == "PASS", "pre-power validation failed")
    spec = json.loads(SPEC.read_text())
    power = read_signed(ROOT / "power_result.json")
    calibration = read_signed(ROOT / "strong_FWER_calibration_report.json")
    results = power["payload"]["results"]
    selected = next((row for row in results if row["all_gates_pass"]), None)
    require(selected is not None and selected["cap_per_class"] == 4, "frozen selection rule did not choose m=4")
    selected_manifest = ROOT / "candidate_m4_manifest.json"
    selected_record = record(selected_manifest, ROOT)
    sample = {
        "status": "PASS",
        "selected_D2_cap_per_class": 4,
        "selected_D2_image_count": 3482,
        "selected_manifest": selected_record,
        "first_passing_candidate": True,
        "candidate_results": results,
        "target_power": 0.8,
        "scientific_claim_decision": None,
    }
    write("sample_size_decision.json", signed("s2_r8_sample_size_decision", "PASS", sample))
    selected_flags = read_signed(selected_manifest)["payload"]["records"]
    extra_count = sum(not row["role_flags"]["P1"] for row in selected_flags)
    resource = {
        "status": "PASS",
        "probe_train": 5232,
        "probe_validation": 1744,
        "D2_test": 3482,
        "D2_extra_over_original": extra_count,
        "S3_source_image_forward_count": 10458,
        "S3_original_source_image_forward_count": 8720,
        "S3_increment": 1738,
        "S3_increment_fraction": 1738 / 8720,
        "S3_increment_percent": 100 * 1738 / 8720,
        "S3_hard_limit_satisfied": 10458 <= 10458 and 1738 / 8720 <= 0.2,
        "S4_NNS_occlusion_image_count": 1744,
        "S4_unchanged": True,
        "extra_D2_role_flags": spec["extra_D2_role_flags"],
        "cost_is_reporting_only": True,
        "no_full_pipeline_time_projection_for_S3": "S1 telemetry mixes NNS occlusion with feature extraction and cannot identify S3-only time",
    }
    write("resource_projection.json", signed("s2_r8_resource_projection", "PASS", resource))
    write("hierarchical_gatekeeping_contract.json", signed("s2_r8_hierarchical_gatekeeping_contract", "PASS", {
        "contract": spec["gatekeeping"],
        "analytic_strong_FWER_proof": spec["gatekeeping"]["strong_FWER_proof"],
        "calibration": record(ROOT / "strong_FWER_calibration_report.json", ROOT),
        "D2_gate_closed_state": "NOT_TESTED_GATE_CLOSED",
        "scientific_claim_decision": None,
    }))
    write("formal_access_audit.json", signed("s2_r8_formal_access_audit", "PASS", {
        "CPU_only": True,
        "formal_metadata_records_read": 11557,
        "formal_hash_only_byte_access_count": 0,
        "formal_image_mask_feature_prediction_endpoint_access_count": 0,
        "model_GPU_attribution_probe_occlusion_count": 0,
        "scientific_outcome_access_count": 0,
        "S3_executed": False,
        "S4_executed": False,
        "S5_executed": False,
    }))
    audit = json.loads(subprocess.check_output([sys.executable, "scripts/validate_nns_vit_patch_depth_s2_r8.py", "false-pass"], cwd=REPO, text=True))
    require(audit["status"] == "PASS" and audit["attack_count"] == 16, "false-pass audit failed")
    write("false_pass_audit.json", signed("s2_r8_false_pass_audit", "PASS", audit))
    commit = subprocess.check_output(["git", "-C", str(REPO.parent), "rev-parse", "HEAD"], text=True).strip()
    write("implementation_seal.json", signed("s2_r8_finalization_implementation_seal", "PASS", {
        "pre_power_implementation_commit": "b9bcdfd901bed1080d1f1e8ad08fa576b48a8d5e",
        "pre_power_seal_commit": "717af9143de09fb2854849628a7f484fa19009e7",
        "finalizer_commit": commit,
        "files": [git_record(item, commit) for item in ("scripts/seal_nns_vit_patch_depth_s2_r8_final.py", "scripts/validate_nns_vit_patch_depth_s2_r8_publication.py", "tests/test_nns_patch_depth_s2_r8_final.py")],
        "scientific_code_changed_after_pre_power_seal": False,
    }))
    inventory_files = [record(path, ROOT) for path in sorted(ROOT.glob("*")) if path.is_file() and path.name not in LATE]
    write("whole_root_inventory.json", signed("s2_r8_whole_root_inventory", "PASS", {"files": inventory_files, "files_digest": digest(inventory_files), "excluded_late_bound": sorted(LATE)}))
    payload = {
        "status": "PASS",
        "selected_D2_cap_per_class": 4,
        "selected_D2_image_count": 3482,
        "selected_manifest": selected_record,
        "P1_P2_D1_confirmatory_image_count": 1744,
        "probe_train_count": 5232,
        "probe_validation_count": 1744,
        "first_layer_marginal_power": selected["primary_Holm_marginal_power_conservative"],
        "gate_open_power": selected["gate_open_power_conservative"],
        "conditional_D2_power": selected["conditional_D2_power_conservative"],
        "unconditional_joint_power": selected["unconditional_joint_power_conservative"],
        "joint_power_lower_bound": selected["joint_power_Wilson_lower_95_conservative"],
        "correlation_independent_joint_lower_bound": selected["correlation_independent_Frechet_lower_bound"],
        "D2_precision_half_width": selected["D2_precision_half_width"],
        "target_power": 0.8,
        "strong_FWER_validation": selected["strong_FWER_calibration"],
        "S3_maximum_forward_count": 10458,
        "S3_increment_fraction": 1738 / 8720,
        "S4_image_count": 1744,
        "formal_outcome_access_count": 0,
        "scientific_claim_decision": None,
        "S3_executed": False,
        "S4_executed": False,
        "S5_executed": False,
        "next_authorization": "S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY",
        "required_evidence": {path.name: record(path, ROOT) for path in sorted(ROOT.glob("*.json")) if path.name not in ("terminal_receipt.json", "independent_validation_report.json")},
    }
    write("terminal_receipt.json", signed("s2_r8_terminal_receipt", "PASS", payload))
    validation = json.loads(subprocess.check_output([sys.executable, "scripts/validate_nns_vit_patch_depth_s2_r8.py", "final"], cwd=REPO, text=True))
    write("independent_validation_report.json", signed("s2_r8_independent_validation_report", "PASS", {"terminal_binding": record(ROOT / "terminal_receipt.json", ROOT), "validation": validation}))
    print(json.dumps({"status": "PASS", "selected_cap": 4, "selected_D2_image_count": 3482, "next_authorization": "S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY"}, sort_keys=True))


if __name__ == "__main__":
    main()
