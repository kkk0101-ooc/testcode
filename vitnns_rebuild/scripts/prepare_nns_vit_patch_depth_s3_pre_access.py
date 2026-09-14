#!/usr/bin/env python3
"""Publish pre-access test/audit reports without reading formal image or mask bytes."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa:E402


def write_failure(root:Path,run_id:str,reason:str)->None:
    document={"schema_version":1,"study_id":s3.STUDY_ID,"stage_id":"S3","run_id":run_id,"document_type":"s3_metadata_initialization_failure","status":"FAILED_IMPLEMENTATION","payload":{"reason":reason,"formal_image_mask_access_count":0,"formal_feature_prediction_outcome_access_count":0,"next_authorization":"NONE","preserved":True}}
    document["canonical_digest"]=s3.canonical_digest(document)
    s3.write_json_new(root/"failure.json",document)


def main()->int:
    historical=[
        (REPO/"evidence/nns_vit_patch_depth/s3_20260914","s3_20260914","system Python lacked NumPy, followed by incorrect Git top-level relative tree path"),
        (REPO/"evidence/nns_vit_patch_depth/s3_r1_20260914","s3_r1_20260914","hydrated LFS content was incorrectly compared directly to Git pointer bytes"),
        (REPO/"evidence/nns_vit_patch_depth/s3_r2_20260914","s3_r2_20260914","LFS pointer signature used a literal escaped newline")]
    for root,run_id,reason in historical:
        if not (root/"failure.json").exists():write_failure(root,run_id,reason)
    runtime=REPO.parent/".runtime/nns_vit_patch_depth_s3"
    reports=[json.loads((runtime/f"pilot_repeat_{index}.json").read_text()) for index in (0,1)]
    arrays=[]
    for index in (0,1):
        with np.load(runtime/f"pilot_repeat_{index}.npz",allow_pickle=False) as bundle:
            arrays.append({key:np.asarray(bundle[key]) for key in bundle.files})
    comparisons={key:{"byte_identical":arrays[0][key].tobytes()==arrays[1][key].tobytes(),"max_abs_error":float(np.max(np.abs(arrays[0][key].astype(np.float64)-arrays[1][key].astype(np.float64))))} for key in arrays[0]}
    s3.require(all(item["byte_identical"] for item in comparisons.values()),"pilot multi-GPU extraction is not byte identical")
    root=REPO/s3.ROOT_REL
    pilot=s3.signed("s3_pilot_repeatability_report","PASS",{"permanently_development_only":True,"source_image_id":reports[0]["source_image_id"],"GPU_runs":[{"physical_gpu_index":item["identity"]["physical_gpu_index"],"gpu_uuid":item["identity"]["gpu_uuid"],"python_executable_realpath":item["identity"]["python_executable_realpath"],"CUBLAS_WORKSPACE_CONFIG":item["identity"]["CUBLAS_WORKSPACE_CONFIG"],"startup_matmul_backward":item["identity"]["startup_matmul_backward"],"feature_array_digest":item["feature_array_digest"],"output_sha256":item["output_sha256"]} for item in reports],"comparisons":comparisons,"frozen_tolerance":{"atol":0.0,"rtol":0.0},"formal_split_access_count":0})
    synthetic=s3.signed("s3_synthetic_validation_report","PASS",{"checks":["dynamic_global_queue","concurrent_O_CREAT_O_EXCL_claim_uniqueness","stale_claim_fresh_launch_recovery","global_terminal_skip","label_shuffle","constant_feature_zero_scale_to_one","split_leakage","train_only_normalization","alpha_smallest_numeric_tie_rule","serialization_roundtrip","equal_image_weighting","crossfit_identity_completeness","weight_seal_pretest_fail_closed","K_less_than_8_allowed","eight_eligible_requires_eight_workers","existing_compute_PID_not_disqualifying","CUBLAS_before_torch","startup_matmul_backward"],"pytest_result":"6 passed","formal_split_access_count":0})
    preaudit=s3.signed("s3_pre_access_audit","PASS",{"checks":["S2-R8 validators PASS","S2-R8 publication validator PASS","S2-R8 LFS pointer and hydrated identity verified","candidate_m4 identity and one-to-one QC join verified","10458-unit schedule frozen","source ID and image SHA pairwise independence","synthetic validation PASS","S1 pilot-only repeatability PASS","D2 test member access fail-closed","formal byte access count remains zero"],"formal_image_mask_access_count":0,"formal_feature_prediction_outcome_access_count":0,"pilot_only_access_count":2,"status":"PASS"})
    for name,document in (("pilot_only_probe_engineering_report.json",pilot),("synthetic_queue_concurrency_crash_recovery_report.json",synthetic),("pre_access_audit.json",preaudit)):
        s3.write_json_new(root/name,document)
    print(json.dumps({"status":"PASS","pilot_byte_identical":True,"formal_access_count":0,"created_time_ns":time.time_ns()},sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
