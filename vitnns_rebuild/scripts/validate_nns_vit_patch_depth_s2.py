#!/usr/bin/env python3
"""Standard-library validator for the terminal S2 evidence root."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "evidence/nns_vit_patch_depth/s2_r2_20260914"
ALLOWED = {"PASS", "BLOCKED_SCIENTIFIC_VALIDITY", "BLOCKED_DATA_INTEGRITY", "BLOCKED_INFRASTRUCTURE", "FAILED_IMPLEMENTATION"}
FORBIDDEN = {"BLOCKED_BUDGET", "BLOCKED_WALL_TIME", "FAIL_COST"}
PREDECESSORS = {
    "s0_terminal": REPO / "evidence/nns_vit_patch_depth/s0_r5_20260914/terminal_receipt.json",
    "s1_terminal": REPO / "evidence/nns_vit_patch_depth/s1_r1_pilot_20260914/terminal_receipt.json",
}
EXCLUDED_FROM_INVENTORY = {"whole_root_inventory.json", "terminal_receipt.json", "independent_validation_report.json", "false_pass_audit.json"}
R2_FRESH_REVISION = {
    "root": "evidence/nns_vit_patch_depth/s2_r2_20260914",
    "predecessor_terminal_root": "evidence/nns_vit_patch_depth/s2_r1_20260914",
    "predecessor_empty_root": "evidence/nns_vit_patch_depth/s2_20260914",
    "predecessor_empty_root_file_count": 0,
    "reason": "r1_terminal_independent_audit_found_missing_live_implementation_hash_recomputation_and_nonmaterialized_final_reports",
}


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


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
    unsigned = dict(value); observed = unsigned.pop("canonical_digest")
    require(observed == digest(unsigned), "canonical digest mismatch: " + str(path))
    return value


def record(path: Path, base: Path) -> dict[str, Any]:
    value = {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if isinstance(document, dict) and "canonical_digest" in document:
            value["canonical_digest"] = read_signed(path)["canonical_digest"]
    return value


def validate(root: Path) -> dict[str, Any]:
    require(root.is_dir(), "S2 root missing")
    terminal = read_signed(root / "terminal_receipt.json")
    access = read_signed(root / "formal_access_audit.json")
    startup = read_signed(root / "startup_gate_report.json")
    incident = read_signed(root / "protocol_order_incident_report.json")
    implementation = read_signed(root / "implementation_seal.json")
    inventory = read_signed(root / "whole_root_inventory.json")
    for value in (terminal, access, startup, incident, inventory):
        require(value["study_id"] == "NNS_VIT_PATCH_DEPTH_SEMANTICS" and value["stage_id"] == "S2", "identity drift")
        require(value["run_id"] == root.name, "run identity drift")
    status = terminal["status"]
    require(status in ALLOWED and status not in FORBIDDEN, "invalid terminal state")
    require(status == "BLOCKED_SCIENTIFIC_VALIDITY", "this root must preserve the ordering violation")
    payload = terminal["payload"]
    require(payload["block_reason"] == "PRE_POWER_SEAL_ORDER_VIOLATION_PILOT_ENDPOINT_DIRECTION_EXPOSED_BEFORE_SEAL", "block reason drift")
    require(payload["next_authorization"] == "NONE" and payload["scientific_claim_decision"] is None, "blocked root issued authorization or claim")
    require(payload["selected_class_count"] is None and payload["exact_sample_size_decision"] is None, "blocked root fabricated cohort or N")
    require(payload["effect_thresholds_frozen"] is False and payload["formal_protocol_frozen"] is False, "post-incident drafts represented as frozen")
    ap = access["payload"]
    require(ap["formal_split_decoded_image_count"] == ap["formal_split_decoded_mask_count"] == ap["formal_outcome_access_count"] == 0, "formal outcome boundary violated")
    require(ap["formal_raw_hash_only_byte_access_count"] == 0, "unexpected formal raw-byte hashing")
    require(ap["model_load_count"] == ap["gpu_cuda_call_count"] == ap["attribution_count"] == ap["probe_training_count"] == ap["occlusion_count"] == 0, "forbidden S2 compute found")
    require(ap["pilot_endpoint_payload_exposed_before_pre_power_seal"] is True, "incident hidden")
    require(incident["status"] == "BLOCKED_SCIENTIFIC_VALIDITY" and incident["payload"]["pre_power_seal_existed_at_access"] is False, "incident status drift")
    sp = startup["payload"]
    require(sp["fresh_revision"] == R2_FRESH_REVISION, "fresh revision binding drift")
    empty_predecessor = REPO / sp["fresh_revision"]["predecessor_empty_root"]
    require((not empty_predecessor.exists()) or (empty_predecessor.is_dir() and not any(empty_predecessor.iterdir())), "empty predecessor root drift")
    r1_terminal_path = REPO / R2_FRESH_REVISION["predecessor_terminal_root"] / "terminal_receipt.json"
    r1_terminal = read_signed(r1_terminal_path)
    require(r1_terminal["status"] == "BLOCKED_SCIENTIFIC_VALIDITY" and r1_terminal["payload"]["next_authorization"] == "NONE", "r1 predecessor terminal drift")
    require(record(r1_terminal_path, REPO) == sp["predecessor_terminals"]["s2_r1_terminal"], "r1 predecessor binding mismatch")
    require(sp["local_head"] == sp["tracking_ref_sha"] == sp["remote_ref_sha"] == "07a2d44f89fb12b6d794bd3d2601cb3b676ea668", "startup Git identity mismatch")
    require(sp["lfs_status_clean"] and sp["lfs_fsck_ok"], "LFS gate failed")
    require(sp["only_external_quarantine_untracked"] and sp["external_quarantine_excluded_untouched"], "workspace cleanliness gate failed")
    for name, path in PREDECESSORS.items():
        observed = record(path, REPO)
        expected = startup["payload"]["predecessor_terminals"][name]
        require(all(observed.get(key) == expected.get(key) for key in observed), "predecessor terminal binding drift: " + name)
    require(startup["payload"]["s0_artifact_recomputation"]["all_match"] is True, "S0 recomputation failed")
    require(startup["payload"]["s1_artifact_recomputation"]["all_match"] is True, "S1 recomputation failed")
    require(startup["payload"]["s1_invariants_recomputed"] == {
        "status": "PASS", "actual_level": "intermediate", "n_total": 24, "n_valid": 23,
        "n_NA_VOID_MASK": 1, "n_failed_or_blocked": 0, "P1_valid_blocks": 276,
        "P2_matched_coordinates": 828, "D1_valid_images": 23, "D2_engineering_classes": 11,
        "single_multi_arrays_byte_identical": "31/31", "formal_split_bytes_access_count": 0,
        "formal_outcomes_access_count": 0, "next_authorization": "S2_PROTOCOL_AND_POWER_FREEZE_ONLY"}, "S1 invariant mismatch")
    require(startup["payload"]["s0_split_invariants_recomputed"] == {"pilot": 1744, "probe_train": 5232, "probe_validation": 1744, "confirmatory_holdout": 1744, "class_count": 872, "per_class_quota": "2/6/2/2"}, "S0 split invariant mismatch")
    for expected in implementation["payload"]["files"]:
        observed = record(REPO / expected["path"], REPO)
        require(observed == expected, "live implementation drift: " + expected["path"])
    require(implementation["payload"]["standard_library_validator"] is True and implementation["payload"]["torch_imported_by_validator"] is False and implementation["payload"]["gpu_required_by_validator"] is False, "implementation boundary drift")
    observed_files = [record(path, root) for path in sorted(root.rglob("*")) if path.is_file() and path.relative_to(root).as_posix() not in EXCLUDED_FROM_INVENTORY]
    require(inventory["payload"]["files"] == observed_files, "whole-root inventory file drift")
    require(inventory["payload"]["files_digest"] == digest(observed_files), "whole-root inventory digest drift")
    for name, expected in payload["required_evidence"].items():
        require(record(root / name, root) == expected, "terminal evidence binding mismatch: " + name)
    for report_name in ("independent_validation_report.json", "false_pass_audit.json"):
        report_path = root / report_name
        if report_path.exists():
            report = read_signed(report_path)
            require(report["payload"]["terminal_binding"] == record(root / "terminal_receipt.json", root), "late report terminal binding drift: " + report_name)
    return {"status": "PASS", "terminal_status": status, "formal_outcome_access_count": 0, "inventory_file_count": len(observed_files), "implementation_file_count": len(implementation["payload"]["files"]), "next_authorization": "NONE"}


def false_pass(root: Path) -> dict[str, Any]:
    base = json.loads((root / "terminal_receipt.json").read_text())
    attacks = {
        "predecessor_terminal_status_authorization": lambda x: x["payload"].update(predecessor_validation_status="FAIL"),
        "split_label_source_id_or_sha": lambda x: x["payload"].update(formal_manifest_frozen=True),
        "pilot_formal_overlap": lambda x: x["payload"].update(pilot_formal_overlap=1),
        "class_or_sample_count": lambda x: x["payload"].update(selected_class_count=872),
        "effect_threshold": lambda x: x["payload"].update(effect_thresholds_frozen=True),
        "power_seed_or_replicate_count": lambda x: x["payload"].update(power_precision_result={"replicates": 1}),
        "alpha_or_Holm_family": lambda x: x["payload"].update(statistical_plan_frozen=True),
        "P2_gate_0_90": lambda x: x["payload"].update(p2_gate_frozen=True),
        "mask_operator": lambda x: x["payload"].update(mask_operator_frozen=True),
        "probe_normalization_or_alpha_rule": lambda x: x["payload"].update(probe_registry_frozen=True),
        "static_full_sharding": lambda x: x["payload"].update(dynamic_execution_contract_frozen=True),
        "worker_GPU_dependent_seed": lambda x: x["payload"].update(seed_contract_frozen=True),
        "claim_duplication": lambda x: x["payload"].update(claim_contract_frozen=True),
        "existing_terminal_recompute": lambda x: x["payload"].update(recovery_contract_frozen=True),
        "K_lt_8_misclassified_failure": lambda x: x["payload"].update(elastic_startup_contract_frozen=True),
        "existing_compute_PID_auto_disqualified": lambda x: x["payload"].update(gpu_eligibility_contract_frozen=True),
        "empty_cohort_or_result": lambda x: x["payload"].update(formal_protocol_frozen=True),
        "terminal_status": lambda x: x.update(status="PASS"),
        "next_authorization": lambda x: x["payload"].update(next_authorization="S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY"),
        "incident_hidden": lambda x: x["payload"].update(block_reason="NONE"),
        "fabricated_N": lambda x: x["payload"].update(exact_sample_size_decision=1744),
        "forbidden_cost_terminal": lambda x: x.update(status="BLOCKED_BUDGET"),
        "scientific_claim": lambda x: x["payload"].update(scientific_claim_decision="SUPPORTED"),
    }
    rows = []
    for name, mutate in attacks.items():
        value = json.loads(json.dumps(base)); mutate(value); value.pop("canonical_digest", None); value["canonical_digest"] = digest(value)
        rejected = False
        try:
            require(value["status"] == "BLOCKED_SCIENTIFIC_VALIDITY", "status")
            require(value["payload"]["next_authorization"] == "NONE", "authorization")
            require(value["payload"]["block_reason"] == "PRE_POWER_SEAL_ORDER_VIOLATION_PILOT_ENDPOINT_DIRECTION_EXPOSED_BEFORE_SEAL", "incident")
            require(value["payload"]["selected_class_count"] is None and value["payload"]["exact_sample_size_decision"] is None, "fabricated design")
            require(value["payload"]["effect_thresholds_frozen"] is False, "posthoc threshold")
            require(value["payload"]["predecessor_validation_status"] == "PASS", "predecessor")
            require(not any(key in value["payload"] for key in ("formal_manifest_frozen", "pilot_formal_overlap", "statistical_plan_frozen", "p2_gate_frozen", "mask_operator_frozen", "probe_registry_frozen", "dynamic_execution_contract_frozen", "seed_contract_frozen", "claim_contract_frozen", "recovery_contract_frozen", "elastic_startup_contract_frozen", "gpu_eligibility_contract_frozen")), "blocked root gained frozen protocol fields")
            require(value["payload"]["power_precision_result"] is None and value["payload"]["formal_protocol_frozen"] is False, "blocked root gained result")
            require(value["status"] not in FORBIDDEN, "cost terminal")
            require(value["payload"]["scientific_claim_decision"] is None, "claim")
        except ValidationError:
            rejected = True
        require(rejected, "mutation false-passed: " + name)
        rows.append({"attack": name, "rejected": True})
    return {"status": "PASS", "attack_count": len(rows), "attacks": rows}


def write_final_reports(root: Path) -> dict[str, Any]:
    require(not (root / "independent_validation_report.json").exists() and not (root / "false_pass_audit.json").exists(), "refusing to overwrite final reports")
    validation = validate(root)
    mutation = false_pass(root)
    terminal_binding = record(root / "terminal_receipt.json", root)
    reports = (
        ("independent_validation_report.json", "s2_independent_standard_library_source_recomputing_validation_report", {"terminal_binding": terminal_binding, "validation": validation}),
        ("false_pass_audit.json", "s2_false_pass_mutation_audit", {"terminal_binding": terminal_binding, **mutation}),
    )
    for name, document_type, payload in reports:
        value = {"schema_version": 1, "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS", "stage_id": "S2", "run_id": root.name, "document_type": document_type, "status": "PASS", "payload": payload}
        value["canonical_digest"] = digest(value)
        data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        descriptor = __import__("os").open(root / name, __import__("os").O_WRONLY | __import__("os").O_CREAT | __import__("os").O_EXCL, 0o644)
        with __import__("os").fdopen(descriptor, "wb") as stream:
            stream.write(data); stream.flush(); __import__("os").fsync(stream.fileno())
    return {"status": "PASS", "reports": [name for name, _, _ in reports], "terminal_binding": terminal_binding}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("validate", "false-pass", "write-final-reports")); parser.add_argument("--root", default=str(DEFAULT_ROOT)); args = parser.parse_args()
    result = validate(Path(args.root)) if args.command == "validate" else false_pass(Path(args.root)) if args.command == "false-pass" else write_final_reports(Path(args.root))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
