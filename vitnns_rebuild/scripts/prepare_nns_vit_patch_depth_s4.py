#!/usr/bin/env python3
"""Create the fresh S4 pre-access evidence root and frozen schedule."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s4 as s4  # noqa: E402


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def artifact(path: Path) -> dict:
    value = {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": s4.sha_file(path)}
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if isinstance(document.get("canonical_digest"), str):
            value["canonical_digest"] = document["canonical_digest"]
    return value


def main() -> int:
    root = REPO / s4.ROOT_REL
    payload = REPO / s4.PAYLOAD_REL
    if root.exists() or payload.exists():
        raise SystemExit("fresh S4 root unavailable; use a prospective revision rather than overwrite")
    s3_path = REPO / s4.S3_TERMINAL_REL
    s4.require(s4.sha_file(s3_path) == s4.S3_TERMINAL_SHA256, "S3-R7 terminal SHA drift")
    s3 = json.loads(s3_path.read_text())
    s4.require(s3["canonical_digest"] == s4.S3_TERMINAL_DIGEST and s3["status"] == "PASS", "S3-R7 terminal status or digest drift")
    s4.require(s3["payload"]["next_authorization"] == "S4_NNS_AND_OCCLUSION_EXECUTION_ONLY", "S3-R7 authorization drift")
    subprocess.check_call(["git", "-C", str(REPO), "merge-base", "--is-ancestor", s4.S3_COMMIT, "HEAD"])
    schedule, resolution = s4.build_schedule(REPO)
    audit = s4.audit_subset_indices(schedule)
    schedule_document = s4.signed("s4_source_schedule", "FROZEN", {
        "source_image_count": 1744, "class_count": 872, "images_per_class": 2,
        "source_id_unique": True, "image_sha256_unique": True,
        "candidate_m4_D2_only_excluded_count": 1738,
        "pilot_probe_train_probe_validation_excluded": True,
        "order_rule": "candidate_m4_record_order_filtered_by_all_six_S4_roles",
        "records_digest": s4.records_digest(schedule), "audit_subset_schedule_indices": audit,
        "audit_subset_namespace": s4.AUDIT_NAMESPACE, "records": schedule,
    })
    resolution_document = s4.signed("s4_path_resolution_registry", "FROZEN", {
        "join_key": ["image_sha256", "annotation_sha256", "annotation_wnid", "image_width", "image_height", "mask_width", "mask_height"],
        "one_to_one": True, "record_count": 1744,
        "selection_sort_exclusion_replacement_use": False,
        "records_digest": s4.records_digest(resolution), "records": resolution,
    })
    s4.write_json_new(REPO / s4.SCHEDULE_REL, schedule_document)
    s4.write_json_new(REPO / s4.PATH_REGISTRY_REL, resolution_document)
    root.mkdir(parents=True)
    payload.mkdir(parents=True)
    now = time.time_ns()
    branch = git("symbolic-ref", "--short", "HEAD")
    tracking = git("rev-parse", "--symbolic-full-name", "@{upstream}")
    documents = {
        "root_creation_receipt.json": s4.signed("s4_root_creation_receipt", "PASS", {
            "root": s4.ROOT_REL.as_posix(), "large_array_root": s4.PAYLOAD_REL.as_posix(),
            "predecessor_root": "evidence/nns_vit_patch_depth/s3_r7_20260915",
            "creation_reason": "AUTHORIZED_FRESH_S4_FORMAL_EXECUTION_ROOT",
            "created_unix_ns": now, "formal_access_status": "PRE_IMPLEMENTATION_SEAL_ZERO_ACCESS",
            "predecessor_terminal_sha256": s4.S3_TERMINAL_SHA256,
            "predecessor_terminal_canonical_digest": s4.S3_TERMINAL_DIGEST,
            "immutable_predecessor_roots_modified": False,
        }),
        "s3_r7_input_binding.json": s4.signed("s4_s3_r7_input_binding", "PASS", {
            "publication_commit": s4.S3_COMMIT, "terminal": artifact(s3_path),
            "terminal_status": s3["status"], "next_authorization": s3["payload"]["next_authorization"],
            "terminal_count": s3["payload"]["terminal_count"], "test_prediction_count": s3["payload"]["test_prediction_count"],
            "publication_validator_D2_prediction_count_alias": s3["payload"]["test_prediction_count"],
            "numeric_prediction_error_values_read": False,
        }),
        "candidate_m4_s4_role_binding.json": s4.signed("s4_candidate_m4_role_binding", "PASS", {
            "source": artifact(REPO / s4.SELECTED_REL), "required_role_flags": ["S4_NNS", "S4_occlusion", "P1", "P2", "D1", "D2"],
            "source_count": 1744, "class_count": 872, "images_per_class": 2,
            "D2_only_excluded_count": 1738, "pilot_probe_train_probe_validation_excluded": True,
            "schedule": artifact(REPO / s4.SCHEDULE_REL), "path_resolution": artifact(REPO / s4.PATH_REGISTRY_REL),
        }),
        "model_preprocessing_target_registry.json": s4.signed("s4_model_preprocessing_target_registry", "FROZEN", json.loads((REPO / s4.SPEC_REL).read_text())["model"] | {"preprocessing": json.loads((REPO / s4.SPEC_REL).read_text())["preprocessing"]}),
        "nns_numeric_registry.json": s4.signed("s4_nns_numeric_registry", "FROZEN", json.loads((REPO / s4.SPEC_REL).read_text())["nns"]),
        "p1_permutation_registry.json": s4.signed("s4_p1_permutation_registry", "FROZEN", json.loads((REPO / s4.SPEC_REL).read_text())["p1"]),
        "p2_operator_control_registry.json": s4.signed("s4_p2_operator_control_registry", "FROZEN", json.loads((REPO / s4.SPEC_REL).read_text())["p2"] | {"control_namespace": s4.P2_NAMESPACE, "tie_namespace": s4.P2_TIE_NAMESPACE}),
        "dependent_unit_schedule_and_queue_contract.json": s4.signed("s4_dependent_unit_schedule_and_queue_contract", "FROZEN", json.loads((REPO / s4.SPEC_REL).read_text())["unit_topology"] | {"atomicity": json.loads((REPO / s4.SPEC_REL).read_text())["atomicity"]}),
        "elastic_gpu_startup_and_lock_contract.json": s4.signed("s4_elastic_gpu_startup_and_lock_contract", "FROZEN", json.loads((REPO / s4.SPEC_REL).read_text())["parallel"]),
        "typed_failure_na_propagation_contract.json": s4.signed("s4_typed_failure_na_propagation_contract", "FROZEN", {"states": list(s4.TERMINAL_STATES), "upstream_NNS_NA": "both_occlusion_placeholders_publish_NA_UPSTREAM_NNS", "zero_fill_or_substitution": False}),
        "retention_and_audit_subset_registry.json": s4.signed("s4_retention_and_audit_subset_registry", "FROZEN", json.loads((REPO / s4.SPEC_REL).read_text())["retention"] | {"audit_subset_schedule_indices": audit}),
        "resource_and_scaling_projection.json": s4.signed("s4_resource_and_scaling_projection", "PASS", json.loads((REPO / s4.SPEC_REL).read_text())["resource_projection"] | {
            "available_disk_bytes": shutil.disk_usage(REPO).free, "estimated_retained_bytes": int(15.8 * 1024**3),
            "estimated_cache_bytes": sum(int(row["image_bytes"]) for row in schedule) + 1744 * 1024 * 1024,
            "estimated_temporary_bytes": int(32 * 1024**3), "independent_validation_GPU_hours_reference": 0.56,
            "independent_validation_CPU_hours_reference": 8.0, "cost_is_scientific_gate": False,
        }),
        "formal_access_audit_pre_implementation_seal.json": s4.signed("s4_formal_access_audit_pre_implementation_seal", "PASS", {
            "S4_selected_JPEG_member_access_count": 0, "S4_selected_mask_member_access_count": 0,
            "decode_count": 0, "model_forward_count": 0, "NNS_count": 0, "occlusion_count": 0,
            "S3_D2_prediction_error_numeric_read_count": 0, "cohort_level_inference_count": 0,
            "S5_execution_count": 0, "S6_execution_count": 0,
            "command_level_forbidden_read_guard_implemented": True, "formal_access_status": "SEALED_ZERO",
        }),
        "startup_git_environment_audit.json": s4.signed("s4_startup_git_environment_audit", "PASS", {
            "local_HEAD": git("rev-parse", "HEAD"), "branch": branch, "tracking_ref": tracking,
            "tracking_oid": git("rev-parse", "@{upstream}"), "git_executable": shutil.which("git"),
            "sys_executable_realpath": os.path.realpath(sys.executable), "S3_publication_ancestor_of_HEAD": True,
            "outer_root_cleanup_quarantine_excluded_read_only": True, "other_unexplained_changes_at_startup": 0,
        }),
        "completion/s4.pending.json": s4.signed("s4_stage_pending", "PENDING", {"created_unix_ns": now, "formal_access_count": 0, "terminal_published": False}),
    }
    for name, document in documents.items():
        s4.write_json_new(root / name, document)
    print(json.dumps({"status": "PASS", "root": root.relative_to(REPO).as_posix(), "source_count": len(schedule), "audit_subset_count": len(audit), "formal_member_access_count": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
