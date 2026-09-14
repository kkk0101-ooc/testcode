#!/usr/bin/env python3
"""Create the metadata-only S3 schedule and prospective pre-access evidence root."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa: E402


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()


def record(path: str) -> dict[str, object]:
    return s3.artifact_record(REPO, REPO / path)


def main() -> int:
    root = REPO / s3.ROOT_REL
    payload = REPO / s3.PAYLOAD_REL
    s3.require(not root.exists(), f"fresh root already exists: {root}")
    s3.require(not payload.exists(), f"fresh payload root already exists: {payload}")
    head = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--short", "HEAD")
    tracking = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    tracking_oid = git("rev-parse", tracking)
    subprocess.run(["git", "merge-base", "--is-ancestor", s3.PREDECESSOR_COMMIT, "HEAD"], cwd=REPO, check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", s3.PREDECESSOR_COMMIT, tracking], cwd=REPO, check=True)
    terminal_path = REPO / "evidence/nns_vit_patch_depth/s2_r8_20260914/terminal_receipt.json"
    terminal = json.loads(terminal_path.read_text())
    s3.require(s3.sha_file(terminal_path) == s3.PREDECESSOR_TERMINAL_SHA256, "S2-R8 terminal SHA drift")
    s3.require(terminal["canonical_digest"] == s3.PREDECESSOR_TERMINAL_DIGEST, "S2-R8 terminal canonical digest drift")
    s3.require(terminal["status"] == "PASS" and terminal["payload"]["next_authorization"] == "S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY", "S2-R8 does not authorize S3")
    s3.require(terminal["payload"]["formal_outcome_access_count"] == 0, "S2-R8 formal access count drift")
    s3.require(not terminal["payload"]["S3_executed"] and not terminal["payload"]["S4_executed"] and not terminal["payload"]["S5_executed"], "S2-R8 execution flags drift")
    rows = s3.build_schedule(REPO)
    independence = s3.independence_report(rows)
    root.mkdir(parents=True)
    payload.mkdir(parents=True)
    schedule = s3.signed("s3_source_schedule", "FROZEN", {"record_count": len(rows), "records": rows, "records_digest": s3.records_digest(rows), "split_counts": {split: sum(row["split"] == split for row in rows) for split in ("probe_train","probe_validation","d2_test")}, "class_count": 872, "status": "FROZEN_METADATA_ONLY"})
    s3.write_json_new(REPO / s3.SCHEDULE_REL, schedule)
    binding_files = [
        "evidence/nns_vit_patch_depth/s2_r8_20260914/terminal_receipt.json",
        "evidence/nns_vit_patch_depth/s2_r8_20260914/candidate_m4_manifest.json",
        "evidence/nns_vit_patch_depth/s2_r8_20260914/sample_size_decision.json",
        "evidence/nns_vit_patch_depth/s2_r8_20260914/power_result.json",
        "evidence/nns_vit_patch_depth/s2_r8_20260914/resource_projection.json",
        "evidence/nns_vit_patch_depth/s2_r8_20260914/formal_access_audit.json",
        "evidence/nns_vit_patch_depth/s2_r8_20260914/implementation_seal.json",
        "evidence/nns_vit_patch_depth/s2_r8_20260914/whole_root_inventory.json",
        "evidence/nns_vit_patch_depth/s2_r8_20260914/independent_validation_report.json",
        "config/nns_vit_patch_depth/s2_r8_run_spec.json",
        "scripts/freeze_nns_vit_patch_depth_s2_r8.py",
        "scripts/run_nns_vit_patch_depth_s2_r8.py",
        "scripts/seal_nns_vit_patch_depth_s2_r8_final.py",
        "scripts/validate_nns_vit_patch_depth_s2_r8.py",
        "scripts/validate_nns_vit_patch_depth_s2_r8_publication.py"
    ]
    bindings = [record(path) for path in binding_files]
    for item in bindings:
        tree_path = f"vitnns_rebuild/{item['path']}"
        tree_bytes = subprocess.check_output(["git", "show", f"{s3.PREDECESSOR_COMMIT}:{tree_path}"], cwd=REPO)
        if tree_bytes.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            pointer = tree_bytes.decode().splitlines()
            oid = next(line.split(":", 1)[1] for line in pointer if line.startswith("oid sha256:"))
            size = int(next(line.split(" ", 1)[1] for line in pointer if line.startswith("size ")))
            s3.require(oid == item["sha256"] and size == item["bytes"], f"S2-R8 LFS pointer identity drift: {item['path']}")
            item["git_tree_storage"] = "LFS_POINTER_OID_AND_SIZE_VERIFIED"
        else:
            s3.require(s3.sha_bytes(tree_bytes) == item["sha256"], f"S2-R8 Git-tree identity drift: {item['path']}")
            item["git_tree_storage"] = "ORDINARY_BLOB_BYTES_VERIFIED"
        item["git_tree_identity_verified"] = True
    documents = {
        "root_creation_receipt.json": s3.signed("s3_root_creation", "FROZEN", {"root": s3.ROOT_REL.as_posix(), "root_reason": "fresh S3-R3 recovery root after metadata-only LFS pointer signature failure", "predecessor": "evidence/nns_vit_patch_depth/s3_r2_20260914", "scientific_predecessor": "s2_r8_20260914", "created_time_ns": time.time_ns()}),
        "route_and_authorization_contract.json": s3.signed("s3_route_authorization", "PASS", {"authorization": "S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY", "forbidden_stages": ["S4","S5"], "forbidden_operations": ["NNS","attribution","occlusion","P1_permutation","P2_matched_controls","cohort_level_D2_inference"], "pass_next_authorization": "S4_NNS_AND_OCCLUSION_EXECUTION_ONLY", "nonpass_next_authorization": "NONE"}),
        "s2_r8_input_binding.json": s3.signed("s3_predecessor_binding", "PASS", {"publication_commit": s3.PREDECESSOR_COMMIT, "local_head": head, "branch": branch, "tracking_ref": tracking, "tracking_oid": tracking_oid, "files": bindings, "terminal_status": "PASS", "formal_outcome_access_count": 0, "S3_executed": False, "S4_executed": False, "S5_executed": False}),
        "selected_manifest_binding.json": s3.signed("s3_selected_manifest_binding", "PASS", {"binding": {"path":"evidence/nns_vit_patch_depth/s2_r8_20260914/candidate_m4_manifest.json","sha256":s3.SELECTED_SHA256,"canonical_digest":s3.SELECTED_DIGEST,"records_digest":s3.SELECTED_RECORDS_DIGEST,"image_count":3482,"class_count":872,"cap_per_class":4,"S3_forward_count":10458,"S4_image_count":1744}, "candidate_m2_m3_forbidden": True, "r7_11557_manifest_forbidden": True}),
        "split_role_manifest.json": s3.signed("s3_split_role_manifest", "FROZEN", {"schedule_path": s3.SCHEDULE_REL.as_posix(), "schedule_sha256": s3.sha_file(REPO / s3.SCHEDULE_REL), "schedule_canonical_digest": schedule["canonical_digest"], "schedule_records_digest": schedule["payload"]["records_digest"], "counts": schedule["payload"]["split_counts"], "role_rule": "D2 extras remain excluded from P1 P2 D1 S4"}),
        "independence_audit.json": s3.signed("s3_independence_audit", "PASS", independence),
        "data_archive_path_registry.json": s3.signed("s3_data_registry", "FROZEN", {"probe_train_validation": {"identity":"evidence/nns_vit_patch_depth/s0_r5_20260914/split_manifest.json","jpeg":"frozen image_local_path","mask":"frozen annotation_archive_path"}, "d2_test": {"identity":"candidate_m4_manifest.json","resolution_join":"S0-R3 image SHA plus annotation SHA plus WNID plus image/mask dimensions exactly one-to-one","jpeg_archive":{"path":"assets/imagenet/ILSVRC2012_img_val.tar","bytes":6744924160,"sha256":s3.JPEG_ARCHIVE_SHA256},"mask_archive":{"path":"assets/imagenet/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip","sha256":s3.MASK_ARCHIVE_SHA256}}, "pre_test_archive_stream_hash_count": 0}),
        "model_checkpoint_registry.json": s3.signed("s3_model_registry", "FROZEN", {"name":"vit_base_patch16_224.augreg2_in21k_ft_in1k","checkpoint":{"path":"assets/models/vit_base_patch16_224_weights.semantic_candidate.pth","bytes":346329934,"sha256":s3.MODEL_SHA256,"canonical_tensor_digest":s3.MODEL_TENSOR_DIGEST},"eval":True,"all_parameters_frozen":True}),
        "preprocessing_feature_registry.json": s3.signed("s3_preprocessing_registry", "FROZEN", {"spec_path": s3.SPEC_REL.as_posix(), "single_forward_all_12_blocks": True, "arrays": json.loads((REPO / s3.SPEC_REL).read_text())["arrays"], "low_level_formula_frozen": True}),
        "ridge_training_registry.json": s3.signed("s3_ridge_registry", "FROZEN", {"details": json.loads((REPO / s3.SPEC_REL).read_text())["ridge"], "test_participation_in_fit": False}),
        "train_crossfit_registry.json": s3.signed("s3_crossfit_registry", "FROZEN", {"details": json.loads((REPO / s3.SPEC_REL).read_text())["crossfit"], "formal_denominator_contribution": False}),
        "dynamic_execution_contract.json": s3.signed("s3_dynamic_execution_contract", "FROZEN", {"details": json.loads((REPO / s3.SPEC_REL).read_text())["parallel"], "claim_identity": "schedule_index_plus_source_ID", "completion_identity": "schedule_index_plus_source_ID_plus_unique_immutable_terminal"}),
        "resource_projection.json": s3.signed("s3_resource_projection", "FROZEN", {"projection": {"available_data_disk_bytes": int(os.statvfs(REPO).f_bavail * os.statvfs(REPO).f_frsize), "feature_payload_bytes": 10458 * (12*196*768*4 + 4*196*4 + 196*5*4), "prediction_payload_bytes": 3482 * (12*196*4 + 196*4 + 12*8 + 8), "archive_cache_temporary_bytes": 2_000_000_000, "estimated_cumulative_GPU_hours": 1.75, "estimated_CPU_hours": 8.0, "independent_numeric_validation_CPU_hours": 2.0, "elapsed_reference_hours_by_K": {str(k): 1.75/k + 0.25 for k in range(1,9)}, "termination_role": "reporting_and_engineering_only"}}),
        "formal_access_audit_pre_access.json": s3.signed("s3_formal_access_audit", "PASS", {"phase":"pre_access", "access": {"S2_predecessor_formal_outcome_access_count":0,"train_JPEG_decode_count":0,"train_mask_decode_count":0,"validation_JPEG_decode_count":0,"validation_mask_decode_count":0,"D2_test_JPEG_raw_byte_access_count":0,"D2_test_mask_raw_byte_access_count":0,"D2_test_archive_member_access_count":0,"D2_test_decode_count":0,"D2_test_feature_extraction_count":0,"D2_test_prediction_count":0,"cohort_level_D2_aggregation_count":0,"per_image_D2_deep_minus_shallow_materialization_count":0,"formal_cohort_estimate_count":0,"formal_p_value_count":0,"formal_CI_count":0,"Holm_decision_count":0,"gatekeeping_decision_count":0,"scientific_claim_decision_count":0,"S4_NNS_count":0,"S4_occlusion_count":0,"S5_execution_count":0}}),
        "completion_s3.pending.json": s3.signed("s3_completion_pending", "PENDING", {"authorization":"S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY", "next_authorization":None, "scientific_claim_decision":None, "created_time_ns":time.time_ns()})
    }
    for name, document in documents.items():
        s3.write_json_new(root / name, document)
    print(json.dumps({"status":"PASS","root":root.relative_to(REPO).as_posix(),"schedule_count":len(rows),"schedule_records_digest":schedule["payload"]["records_digest"],"formal_image_mask_access_count":0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
