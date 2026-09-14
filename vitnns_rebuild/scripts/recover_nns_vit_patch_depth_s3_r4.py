#!/usr/bin/env python3
"""CPU-only S3-R4 rebind and scalar serialization recovery."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa:E402
SOURCE_ROOT=REPO/"evidence/nns_vit_patch_depth/s3_r3_20260914";TARGET_ROOT=REPO/"evidence/nns_vit_patch_depth/s3_r4_20260915";TARGET_PAYLOAD=REPO/"artifacts/nns_vit_patch_depth/s3_r4_20260915"
def sign(kind:str,status:str,payload:Mapping[str,Any],commit:str)->dict[str,Any]:
    value={"$schema":"s3_artifact.schema.json","schema_version":1,"study_id":s3.STUDY_ID,"stage_id":"S3","run_id":"s3_r4_20260914","document_type":kind,"status":status,"producer_commit":commit,"payload":s3.jsonable(dict(payload))};value["canonical_digest"]=s3.canonical_digest(value);return value
def main()->int:
    s3.require(not TARGET_ROOT.exists() and not TARGET_PAYLOAD.exists(),"fresh S3-R4 root already exists")
    s3.require(os.environ.get("CUDA_VISIBLE_DEVICES","")=="","S3-R4 recovery must clear CUDA_VISIBLE_DEVICES")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip();TARGET_ROOT.mkdir(parents=True);TARGET_PAYLOAD.mkdir(parents=True)
    schedule=json.loads((REPO/s3.SCHEDULE_REL).read_text())["payload"]["records"]
    rebind=[];counts={};corrected=[];started=time.monotonic()
    for row in schedule:
        index=int(row["schedule_index"]);source_path=SOURCE_ROOT/"unit_terminals"/f"unit-{index:05d}.json";source=json.loads(source_path.read_text());s3.require(s3.verify_signed(source),f"source terminal digest {index}");payload=dict(source["payload"]);s3.require(payload["source_image_id"]==row["source_image_id"],f"source identity {index}")
        arrays=[]
        for record in payload["arrays"]:
            item=dict(record);path=REPO/item["path"];s3.require(path.is_file() and s3.sha_file(path)==item["sha256"],f"source array SHA {index}")
            value=np.load(path,allow_pickle=False);s3.require(s3.array_digest(value)==item["canonical_array_digest"],f"source canonical array digest {index}")
            if payload["split"]=="d2_test" and payload["terminal_state"]=="VALID_FEATURE" and Path(item["path"]).name=="control_mse.npy":continue
            arrays.append(item)
        if payload["split"]=="d2_test" and payload["terminal_state"]=="VALID_FEATURE":
            by_name={Path(item["path"]).name:item for item in payload["arrays"]};occupancy=np.load(REPO/by_name["occupancy.npy"]["path"],allow_pickle=False)[3].astype(np.float64);prediction=np.load(REPO/by_name["control_prediction.npy"]["path"],allow_pickle=False).astype(np.float64);value=np.asarray(np.mean((prediction-occupancy)**2,dtype=np.float64),dtype="<f8")
            new_record=s3.write_array_atomic(TARGET_PAYLOAD/"corrected_control_mse"/f"unit-{index:05d}.npy",value,dtype="<f8",axes=[],producer_commit=commit,source_identity=row["source_image_id"],recomputation_rule="per-image mean squared error of sealed low-level prediction and sealed target occupancy; no cohort aggregation")
            new_record["path"]=(TARGET_PAYLOAD/"corrected_control_mse"/f"unit-{index:05d}.npy").relative_to(REPO).as_posix();s3.require(new_record["shape"]==[],"corrected scalar shape drift");arrays.append(new_record);corrected.append({"schedule_index":index,"source_image_id":row["source_image_id"],"source_array":by_name["control_mse.npy"],"corrected_array":new_record,"numeric_value_unchanged":bool(np.array_equal(np.load(REPO/by_name["control_mse.npy"]["path"],allow_pickle=False).reshape(()),value))})
        payload["arrays"]=sorted(arrays,key=lambda item:item["path"]);payload["producer_commit"]=commit;payload["launch_id"]="s3_r4_cpu_rebind_launch_001";payload["worker_id"]="cpu-rebind-00";payload["worker_pid"]=os.getpid();payload["physical_gpu_index"]=0;payload["gpu_uuid"]="CPU_ONLY_NO_GPU";payload["started_time_ns"]=time.time_ns();payload["completed_time_ns"]=time.time_ns();payload["python_executable_realpath"]=os.path.realpath(sys.executable)
        target=TARGET_ROOT/"unit_terminals"/f"unit-{index:05d}.json";s3.write_json_new(target,sign("s3_unit_terminal",payload["terminal_state"],payload,commit));state=payload["terminal_state"];counts[state]=counts.get(state,0)+1;rebind.append({"schedule_index":index,"source_terminal":{"path":source_path.relative_to(REPO).as_posix(),"sha256":s3.sha_file(source_path),"canonical_digest":source["canonical_digest"]},"target_terminal":{"path":target.relative_to(REPO).as_posix(),"sha256":s3.sha_file(target),"canonical_digest":json.loads(target.read_text())["canonical_digest"]},"correction":"CONTROL_MSE_TRUE_SCALAR" if payload["split"]=="d2_test" and state=="VALID_FEATURE" else "IDENTITY_REBIND"})
    s3.require(len(rebind)==10458 and len(corrected)==3424,"S3-R4 recovery count drift")
    documents={
      "root_creation_receipt.json":sign("s3_r4_root_creation","PASS",{"root":TARGET_ROOT.relative_to(REPO).as_posix(),"large_array_root":TARGET_PAYLOAD.relative_to(REPO).as_posix(),"predecessor":"evidence/nns_vit_patch_depth/s3_r3_20260914","reason":"fresh recovery for scalar serialization shape","formal_image_mask_access_count":0},commit),
      "rebind_manifest.json":sign("s3_r4_rebind_manifest","PASS",{"records":rebind,"records_digest":s3.records_digest(rebind),"record_count":10458,"identity_rebind_count":10458-3424,"scalar_correction_count":3424,"model_forward_count":0,"formal_image_mask_access_count":0},commit),
      "control_mse_scalar_correction_manifest.json":sign("s3_r4_scalar_correction_manifest","PASS",{"records":corrected,"records_digest":s3.records_digest(corrected),"record_count":3424,"all_numeric_values_unchanged":all(item["numeric_value_unchanged"] for item in corrected),"source_shape":[1],"target_shape":[],"dtype":"float64","cohort_aggregation_count":0},commit),
      "weight_and_test_gate_binding.json":sign("s3_r4_weight_gate_binding","PASS",{"probe_weight_seal":s3.artifact_record(REPO,SOURCE_ROOT/"probe_weight_seal.json"),"test_access_gate":s3.artifact_record(REPO,SOURCE_ROOT/"test_access_gate.json"),"weights_or_predictions_changed":False,"alpha_or_normalization_changed":False},commit),
      "formal_access_audit.json":sign("s3_r4_formal_access_audit","PASS",{"recovery_additional_image_raw_byte_access_count":0,"recovery_additional_mask_raw_byte_access_count":0,"recovery_additional_archive_member_access_count":0,"recovery_additional_decode_count":0,"recovery_additional_feature_extraction_count":0,"recovery_additional_prediction_count":0,"recovery_per_image_MSE_recompute_count":3424,"cohort_level_D2_aggregation_count":0,"per_image_D2_deep_minus_shallow_materialization_count":0,"formal_cohort_estimate_count":0,"formal_p_value_count":0,"formal_CI_count":0,"Holm_decision_count":0,"gatekeeping_decision_count":0,"scientific_claim_decision_count":0,"S4_NNS_count":0,"S4_occlusion_count":0,"S5_execution_count":0},commit),
      "recovery_launch.json":sign("s3_r4_recovery_launch","PASS",{"launch_id":"s3_r4_cpu_rebind_launch_001","CPU_only":True,"CUDA_VISIBLE_DEVICES":"","terminal_count":10458,"scalar_correction_count":3424,"elapsed_seconds":time.monotonic()-started,"fresh_launch":True},commit)
    }
    for name,doc in documents.items():s3.write_json_new(TARGET_ROOT/name,doc)
    print(json.dumps({"status":"PASS","terminal_count":10458,"scalar_correction_count":3424,"states":counts,"elapsed_seconds":time.monotonic()-started},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
