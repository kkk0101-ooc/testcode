#!/usr/bin/env python3
"""Standard-library metadata/provenance validation for formal S4."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "evidence/nns_vit_patch_depth/s4_r4_20260915"
SCHEDULE = REPO / "manifests/nns_vit_patch_depth/s4_r4_source_schedule.json"
SPEC = REPO / "config/nns_vit_patch_depth/s4_run_spec.json"
S3_TERMINAL = REPO / "evidence/nns_vit_patch_depth/s3_r7_20260915/terminal_receipt.json"
EXPECTED_S3_COMMIT = "a2fa81082ba9b49e0aa44407cbbb373de18accff"
EXPECTED_S3_SHA = "8c1a4f82d3bb553246cc4c019e4e2286a8dfb7dfd8b058265af93ad5f1e192d4"
EXPECTED_S3_DIGEST = "1b6f7308f135dc69ad0146635a31eb529cfac703710fe5a340c009c56ce1215d"
EXPECTED_M4_SHA = "9309a73c1cc2d91cc4c6ca2dc6a13e5b6ed3fed680ed20bc35e1152fc259e4f8"
EXPECTED_M4_DIGEST = "2260a21c949f364eafc21fc3d722243a65c2f05aa82873c9aeb7779a8b1c3f6b"
EXPECTED_RECORDS_DIGEST = "d5e9d5a4873d51b6ddedfe3ed2c9cb08be50fac4a8cfc053d8782c83ffafbd16"
ALLOWED_OUTER_UNTRACKED_PREFIX = "../.root_cleanup_quarantine/"


class ValidationError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ValidationError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path: Path) -> str:
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"file changed while hashing: {path}")
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing required file: {path.relative_to(REPO)}")
    return json.loads(path.read_text())


def verify_signed(value: Mapping[str, Any], *, run_id: str = "s4_r4_20260915") -> None:
    expected = {"$schema", "schema_version", "study_id", "stage_id", "run_id", "document_type", "status", "payload", "canonical_digest"}
    if "producer_commit" in value:
        expected.add("producer_commit")
    require(set(value) == expected, f"open or missing artifact envelope fields for {value.get('document_type')}")
    unsigned = dict(value)
    claimed = unsigned.pop("canonical_digest")
    require(claimed == digest(unsigned), f"canonical digest mismatch for {value.get('document_type')}")
    require(value["study_id"] == "NNS_VIT_PATCH_DEPTH_SEMANTICS" and value["stage_id"] == "S4" and value["run_id"] == run_id, "stage identity drift")


def validate_git() -> dict[str, Any]:
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    subprocess.check_call(["git", "-C", str(REPO), "merge-base", "--is-ancestor", EXPECTED_S3_COMMIT, head])
    tracking = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--symbolic-full-name", "@{upstream}"], text=True).strip()
    tracking_oid = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "@{upstream}"], text=True).strip()
    subprocess.check_call(["git", "-C", str(REPO), "merge-base", "--is-ancestor", EXPECTED_S3_COMMIT, tracking_oid])
    return {"HEAD": head, "tracking_ref": tracking, "tracking_oid": tracking_oid}


def validate_preaccess_documents() -> dict[str, Any]:
    git_state = validate_git()
    s3 = load(S3_TERMINAL)
    require(sha_file(S3_TERMINAL) == EXPECTED_S3_SHA, "S3 terminal SHA drift")
    require(s3["canonical_digest"] == EXPECTED_S3_DIGEST and s3["status"] == "PASS", "S3 status/digest drift")
    require(s3["payload"]["next_authorization"] == "S4_NNS_AND_OCCLUSION_EXECUTION_ONLY", "S3 next authorization drift")
    require(s3["payload"]["terminal_count"] == 10458 and s3["payload"]["test_prediction_count"] == 3424, "S3 counts drift")
    m4_path = REPO / "evidence/nns_vit_patch_depth/s2_r8_20260914/candidate_m4_manifest.json"
    m4 = load(m4_path)
    require(sha_file(m4_path) == EXPECTED_M4_SHA and m4["canonical_digest"] == EXPECTED_M4_DIGEST and m4["payload"]["records_digest"] == EXPECTED_RECORDS_DIGEST, "candidate_m4 identity drift")
    schedule = load(SCHEDULE)
    verify_signed(schedule)
    rows = schedule["payload"]["records"]
    require(len(rows) == 1744 and [row["schedule_index"] for row in rows] == list(range(1744)), "schedule count/order drift")
    require(schedule["payload"]["records_digest"] == digest(rows), "schedule records digest drift")
    require(len({row["source_image_id"] for row in rows}) == 1744 and len({row["image_sha256"] for row in rows}) == 1744, "schedule source/hash uniqueness drift")
    class_counts = Counter(row["target_model_class_index"] for row in rows)
    require(len(class_counts) == 872 and set(class_counts.values()) == {2}, "schedule class balance drift")
    required_flags = ("S4_NNS", "S4_occlusion", "P1", "P2", "D1", "D2")
    require(all(all(row["role_flags"].get(key) is True for key in required_flags) for row in rows), "required S4 role flag drift")
    require(not any(any(row["role_flags"].get(key) for key in ("pilot", "probe_train", "probe_validation")) for row in rows), "forbidden role in S4")
    expected = {
        "root_creation_receipt.json", "s3_r7_input_binding.json", "candidate_m4_s4_role_binding.json",
        "model_preprocessing_target_registry.json", "nns_numeric_registry.json", "p1_permutation_registry.json",
        "p2_operator_control_registry.json", "dependent_unit_schedule_and_queue_contract.json",
        "elastic_gpu_startup_and_lock_contract.json", "typed_failure_na_propagation_contract.json",
        "retention_and_audit_subset_registry.json", "resource_and_scaling_projection.json",
        "formal_access_audit_pre_implementation_seal.json", "startup_git_environment_audit.json",
        "s4_r3_failure_binding.json", "completion/s4.pending.json",
    }
    for name in expected:
        document = load(ROOT / name)
        verify_signed(document)
    access = load(ROOT / "formal_access_audit_pre_implementation_seal.json")["payload"]
    require(all(access[key] == 0 for key in ("S4_selected_JPEG_member_access_count", "S4_selected_mask_member_access_count", "decode_count", "model_forward_count", "NNS_count", "occlusion_count", "S3_D2_prediction_error_numeric_read_count", "cohort_level_inference_count", "S5_execution_count", "S6_execution_count")), "pre-seal access is nonzero")
    spec = load(SPEC)
    require(spec["unit_topology"]["scheduled_unit_terminals"] == 5232 and spec["unit_topology"]["static_shards"] is False, "unit topology drift")
    require(spec["scientific_claim_decision"] is None and spec["next_authorization_on_pass"] == "S5_D2_MISSING_CLASS_RECONCILIATION_GATE_ONLY", "scientific boundary drift")
    require(spec["nns"]["quadrature_schedule"] == [128, 256, 512, 1024, 2048, 4096, 8192], "quadrature drift")
    require(spec["p1"]["namespace"] == "NNS_VIT_PATCH_DEPTH_S2_R4_P1_POSITION_PERMUTATION_V1" and spec["p1"]["permutations"] == 64, "P1 drift")
    require(spec["p2"]["patch_budgets"] == [4, 8, 16] and spec["p2"]["primary_operator"] == "image_channel_mean" and spec["p2"]["sensitivity_operator"] == "normalized_zero", "P2 drift")
    schemas = [load(REPO / "config/nns_vit_patch_depth/s4_artifact.schema.json"), load(REPO / "config/nns_vit_patch_depth/s4_unit_terminal.schema.json"), load(REPO / "config/nns_vit_patch_depth/s4_source_schedule.schema.json"), load(REPO / "config/nns_vit_patch_depth/s4_claim.schema.json")]
    require(all(value.get("additionalProperties") is False for value in schemas), "top-level schema not closed")
    source = (REPO / "src/vitnns/nns_patch_depth_s4.py").read_text()
    worker = (REPO / "src/vitnns/nns_patch_depth_s4_worker.py").read_text()
    require("schedule_index %" not in source + worker and "index % K" not in source + worker, "static shard implementation detected")
    require("CUBLAS_WORKSPACE_CONFIG" in worker and "@ right" in worker and "backward()" in worker, "deterministic startup probe drift")
    return {"status": "PASS", "git": git_state, "source_count": 1744, "class_count": 872, "pre_seal_formal_access_count": 0, "closed_schema_count": len(schemas)}


def unit_paths() -> list[Path]:
    return sorted(ROOT.glob("unit_terminals/*/unit-*.json"))


def validate_unit(value: Mapping[str, Any], path: Path, schedule: list[dict[str, Any]]) -> None:
    verify_signed(value)
    require(value["document_type"] == "s4_unit_terminal", f"wrong document type: {path}")
    payload = value["payload"]
    index = payload["schedule_index"]
    require(type(index) is int and 0 <= index < 1744, f"unit schedule index invalid: {path}")
    row = schedule[index]
    require(payload["source_id"] == row["source_image_id"] and payload["image_sha256"] == row["image_sha256"] and payload["mask_sha256"] == row["annotation_sha256"], f"unit source identity drift: {path}")
    require(payload["terminal_state"] == value["status"], f"unit status mismatch: {path}")
    require(payload["S3_D2_prediction_error_numeric_reads"] == 0 and payload["cohort_level_inference_count"] == 0 and payload["scientific_claim_decision"] is None, f"unit boundary violation: {path}")
    require(payload["formal_archive_member_reads_by_worker"] == 0, f"worker directly accessed archive: {path}")


def validate_final() -> dict[str, Any]:
    pre = validate_preaccess_documents()
    schedule = load(SCHEDULE)["payload"]["records"]
    paths = unit_paths()
    require(len(paths) == 5232, "S4 unit terminal count is not 5232")
    seen = set()
    counts: Counter[str] = Counter()
    types: Counter[str] = Counter()
    for path in paths:
        value = load(path)
        validate_unit(value, path, schedule)
        key = (value["payload"]["unit_type"], value["payload"]["schedule_index"])
        require(key not in seen, "duplicate immutable unit terminal")
        seen.add(key)
        counts[value["status"]] += 1
        types[value["payload"]["unit_type"]] += 1
    require(types == Counter({"NNS_IMAGE": 1744, "OCCLUSION_PRIMARY": 1744, "OCCLUSION_SENSITIVITY": 1744}), "unit type coverage drift")
    completions = sorted(ROOT.glob("source_completions/source-*.json"))
    require(len(completions) == 1744, "source completion coverage drift")
    for index, path in enumerate(completions):
        value = load(path)
        verify_signed(value)
        require(value["payload"]["schedule_index"] == index and value["payload"]["all_three_units_terminal"] is True, "source completion drift")
    access = load(ROOT / "phased_formal_access_audit.json")["payload"]
    require(access["authorized_source_count"] == 1744 and access["D2_only_member_read_count"] == 0, "formal access cohort drift")
    require(access["S3_D2_prediction_error_numeric_read_count"] == 0 and access["cohort_level_inference_count"] == 0 and access["S5_execution_count"] == 0 and access["S6_execution_count"] == 0, "formal scientific boundary drift")
    terminal = load(ROOT / "terminal_receipt.json")
    verify_signed(terminal)
    require(terminal["status"] in ("PASS", "BLOCKED_SCIENTIFIC_VALIDITY", "BLOCKED_DATA_INTEGRITY", "BLOCKED_INFRASTRUCTURE", "FAILED_IMPLEMENTATION"), "forbidden stage terminal")
    require(terminal["payload"]["scientific_claim_decision"] is None and terminal["payload"]["cohort_level_inference_count"] == 0, "terminal scientific boundary drift")
    return {"status": "PASS", "terminal_status": terminal["status"], "terminal_count": len(paths), "source_completion_count": len(completions), "unit_type_counts": dict(types), "state_counts": dict(counts), "preaccess": pre}


def false_pass() -> dict[str, Any]:
    base_spec = load(SPEC)
    base_schedule = load(SCHEDULE)
    attacks: list[tuple[str, Callable[[], None]]] = []
    def reject(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            return
        raise ValidationError("false-pass attack accepted: " + name)
    def bad_signed(document: dict[str, Any]) -> None:
        verify_signed(document)
    for field, replacement in (("status", "FAIL"), ("canonical_digest", "0" * 64)):
        value = copy.deepcopy(load(S3_TERMINAL))
        value[field] = replacement
        attacks.append(("S3_" + field, lambda value=value: require(value["status"] == "PASS" and value["canonical_digest"] == EXPECTED_S3_DIGEST, "rejected")))
    mutations = [
        ("cohort_count", lambda d: d["payload"].update(source_image_count=1743)),
        ("cohort_order", lambda d: d["payload"]["records"].reverse()),
        ("source_id", lambda d: d["payload"]["records"][0].update(source_image_id="tampered")),
        ("image_sha", lambda d: d["payload"]["records"][0].update(image_sha256="0" * 64)),
        ("role_flags", lambda d: d["payload"]["records"][0]["role_flags"].update(P1=False)),
    ]
    for name, mutate in mutations:
        value = copy.deepcopy(base_schedule)
        mutate(value)
        attacks.append((name, lambda value=value: bad_signed(value)))
    spec_mutations = [
        ("backbone", lambda d: d["model"].update(name="other")),
        ("checkpoint", lambda d: d["model"].update(checkpoint_sha256="0" * 64)),
        ("target", lambda d: d["preprocessing"].update(target="predicted_class")),
        ("component", lambda d: d["nns"].update(component="other")),
        ("geometry", lambda d: d["preprocessing"].update(image="nearest")),
        ("score_formula", lambda d: d["nns"].update(source_patch_score="wrong")),
        ("aggregation", lambda d: d["nns"].update(aggregation_order="positive_then_sum")),
        ("block11", lambda d: d["nns"].update(block11_zero_patch_destination="drop")),
        ("quadrature", lambda d: d["nns"].update(quadrature_schedule=[128])),
        ("chunks", lambda d: d["nns"].update(alpha_chunk=8)),
        ("P1_namespace", lambda d: d["p1"].update(namespace="wrong")),
        ("P1_permutations", lambda d: d["p1"].update(permutations=63)),
        ("P1_strata", lambda d: d["p1"].update(position_strata="none")),
        ("P1_tie", lambda d: d["p1"].update(tie_rule="random")),
        ("P2_budgets", lambda d: d["p2"].update(patch_budgets=[4])),
        ("P2_operator", lambda d: d["p2"].update(primary_operator="normalized_zero")),
        ("P2_controls", lambda d: d["p2"].update(controls_per_coordinate=3)),
        ("P2_matching", lambda d: d["p2"].update(position_match="approximate")),
        ("P2_attempts", lambda d: d["p2"].update(maximum_attempts=1)),
        ("outcome_operator_selection", lambda d: d["p2"].update(outcome_dependent_selection=True)),
        ("static_shard", lambda d: d["unit_topology"].update(static_shards=True)),
        ("worker_seed", lambda d: d["scientific_randomness"]["allowed_seed_inputs"].append("worker_id")),
        ("K_less_8_fail", lambda d: d["parallel"].update(K_min=8)),
        ("eight_to_seven", lambda d: d["parallel"].update(K_max=7)),
        ("PID_exclusion", lambda d: d["parallel"].update(existing_compute_pid_is_disqualifying=True)),
        ("cost_gate", lambda d: d["resource_projection"].update(reporting_and_reservation_only=False)),
        ("cohort_inference", lambda d: d.update(scientific_claim_decision="SUPPORTED")),
        ("S5_execution", lambda d: d["access_boundary"].update(S5_execution_count=1)),
        ("S6_execution", lambda d: d["access_boundary"].update(S6_execution_count=1)),
        ("next_authorization", lambda d: d.update(next_authorization_on_pass="S5_ANALYSIS")),
    ]
    def validate_frozen_spec(value: Mapping[str, Any]) -> None:
        require(value == base_spec, "frozen spec rejected")
    for name, mutate in spec_mutations:
        value = copy.deepcopy(base_spec)
        mutate(value)
        attacks.append((name, lambda value=value: validate_frozen_spec(value)))
    # Execution-mechanism attacks are checked directly from immutable source.
    source = (REPO / "src/vitnns/nns_patch_depth_s4.py").read_text()
    worker = (REPO / "src/vitnns/nns_patch_depth_s4_worker.py").read_text()
    mechanisms = {
        "duplicate_claim": "O_CREAT | os.O_EXCL" in source, "existing_terminal_recompute": "terminal_path(root, unit_type, index).is_file()" in source,
        "stale_claim_recovery": "launch_root" in source, "CUBLAS_before_Torch": "_runtime_gate()" in worker and "import torch" in worker,
        "startup_matmul_backward": "@ right" in worker and "backward()" in worker,
        "S3_D2_read_guard": "test_prediction_error_manifest.json" in worker, "nonempty_cohort": len(base_schedule["payload"]["records"]) == 1744,
        "gpu_hours_not_divided_by_K": base_spec["resource_projection"]["reporting_and_reservation_only"] is True,
    }
    require(all(mechanisms.values()), "mechanism false-pass coverage drift")
    for name, fn in attacks:
        reject(name, fn)
    return {"status": "PASS", "attack_count": len(attacks) + len(mechanisms), "attacks": [name for name, _ in attacks] + list(mechanisms), "all_rejected": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preaccess", "final", "false-pass"))
    args = parser.parse_args()
    result = validate_preaccess_documents() if args.command == "preaccess" else validate_final() if args.command == "final" else false_pass()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
