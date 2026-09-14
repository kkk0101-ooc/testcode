#!/usr/bin/env python3
"""CPU-only bit-exact scalar reserialization into fresh S3-R5 roots."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa:E402
SOURCE=REPO/"evidence/nns_vit_patch_depth/s3_r3_20260914";ROOT=REPO/"evidence/nns_vit_patch_depth/s3_r5_20260915";PAYLOAD=REPO/"artifacts/nns_vit_patch_depth/s3_r5_20260915"
def sign(kind:str,status:str,payload:Mapping[str,Any],commit:str)->dict[str,Any]:
    value={"$schema":"s3_artifact.schema.json","schema_version":1,"study_id":s3.STUDY_ID,"stage_id":"S3","run_id":"s3_r5_20260915","document_type":kind,"status":status,"producer_commit":commit,"payload":s3.jsonable(dict(payload))};value["canonical_digest"]=s3.canonical_digest(value);return value
def main()->int:
    s3.require(not ROOT.exists() and not PAYLOAD.exists(),"fresh S3-R5 root unavailable");s3.require(os.environ.get("CUDA_VISIBLE_DEVICES","")=="","CUDA_VISIBLE_DEVICES must be empty")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip();ROOT.mkdir(parents=True);PAYLOAD.mkdir(parents=True);schedule=json.loads((REPO/s3.SCHEDULE_REL).read_text())["payload"]["records"]
    rebind=[];corrections=[];counts={};started=time.monotonic()
    for row in schedule:
        index=int(row["schedule_index"]);source_path=SOURCE/"unit_terminals"/f"unit-{index:05d}.json";source=json.loads(source_path.read_text());s3.require(s3.verify_signed(source),f"source terminal digest {index}");payload=dict(source["payload"]);s3.require(payload["source_image_id"]==row["source_image_id"],f"source identity {index}")
        arrays=[];old_control=None
        for record in payload["arrays"]:
            item=dict(record);path=REPO/item["path"];s3.require(path.is_file() and s3.sha_file(path)==item["sha256"],f"source array SHA {index}");value=np.load(path,allow_pickle=False);s3.require(s3.array_digest(value)==item["canonical_array_digest"],f"source array digest {index}")
            if payload["split"]=="d2_test" and payload["terminal_state"]=="VALID_FEATURE" and Path(item["path"]).name=="control_mse.npy":old_control=(item,value);continue
            arrays.append(item)
        correction="IDENTITY_REBIND"
        if old_control is not None:
            old_record,old_value=old_control;s3.require(old_value.shape==(1,) and old_value.dtype==np.float64 and np.isfinite(old_value).all(),"source scalar candidate contract")
            value=np.asarray(old_value[0],dtype="<f8");new_path=PAYLOAD/"corrected_control_mse"/f"unit-{index:05d}.npy";new=s3.write_array_atomic(new_path,value,dtype="<f8",axes=[],producer_commit=commit,source_identity=row["source_image_id"],recomputation_rule="bit-exact reshape of the sole sealed S3-R3 float64 control MSE element; no numeric recomputation")
            new["path"]=new_path.relative_to(REPO).as_posix();s3.require(new["shape"]==[] and np.asarray(old_value[0]).tobytes()==np.load(new_path,allow_pickle=False).tobytes(),"bit-exact scalar recovery failure");arrays.append(new);correction="CONTROL_MSE_BIT_EXACT_SCALAR_RESERIALIZATION";corrections.append({"schedule_index":index,"source_image_id":row["source_image_id"],"source_array":old_record,"corrected_array":new,"value_bytes_identical":True})
        payload.update({"arrays":sorted(arrays,key=lambda x:x["path"]),"producer_commit":commit,"launch_id":"s3_r5_cpu_rebind_launch_001","worker_id":"cpu-rebind-00","worker_pid":os.getpid(),"physical_gpu_index":0,"gpu_uuid":"CPU_ONLY_NO_GPU","started_time_ns":time.time_ns(),"completed_time_ns":time.time_ns(),"python_executable_realpath":os.path.realpath(sys.executable)})
        target=ROOT/"unit_terminals"/f"unit-{index:05d}.json";s3.write_json_new(target,sign("s3_unit_terminal",payload["terminal_state"],payload,commit));state=payload["terminal_state"];counts[state]=counts.get(state,0)+1;rebind.append({"schedule_index":index,"source_terminal":{"path":source_path.relative_to(REPO).as_posix(),"sha256":s3.sha_file(source_path),"canonical_digest":source["canonical_digest"]},"target_terminal":{"path":target.relative_to(REPO).as_posix(),"sha256":s3.sha_file(target),"canonical_digest":json.loads(target.read_text())["canonical_digest"]},"correction":correction})
    s3.require(len(rebind)==10458 and len(corrections)==3424,"R5 count drift")
    r4_failure=sign("s3_r4_terminal_receipt","FAILED_IMPLEMENTATION",{"reason":"float32 prediction recomputation did not preserve original float64 control MSE bits","numeric_value_unchanged_count":0,"numeric_value_changed_count":3424,"formal_image_mask_access_count":0,"next_authorization":"NONE"},commit);s3.write_json_new(REPO/"evidence/nns_vit_patch_depth/s3_r4_20260915/failure.json",r4_failure)
    docs={"root_creation_receipt.json":sign("s3_r5_root_creation","PASS",{"root":ROOT.relative_to(REPO).as_posix(),"large_array_root":PAYLOAD.relative_to(REPO).as_posix(),"predecessor":"evidence/nns_vit_patch_depth/s3_r4_20260915","scientific_source":"evidence/nns_vit_patch_depth/s3_r3_20260914","reason":"fresh bit-exact scalar serialization recovery"},commit),"rebind_manifest.json":sign("s3_r5_rebind_manifest","PASS",{"records":rebind,"records_digest":s3.records_digest(rebind),"record_count":10458,"scalar_correction_count":3424,"formal_image_mask_access_count":0,"model_forward_count":0},commit),"control_mse_scalar_correction_manifest.json":sign("s3_r5_scalar_manifest","PASS",{"records":corrections,"records_digest":s3.records_digest(corrections),"record_count":3424,"all_value_bytes_identical":True,"source_shape":[1],"target_shape":[],"dtype":"float64","cohort_aggregation_count":0},commit),"formal_access_audit.json":sign("s3_r5_access_audit","PASS",{"recovery_additional_image_raw_byte_access_count":0,"recovery_additional_mask_raw_byte_access_count":0,"recovery_additional_archive_member_access_count":0,"recovery_additional_decode_count":0,"recovery_additional_feature_extraction_count":0,"recovery_additional_prediction_count":0,"cohort_level_D2_aggregation_count":0,"per_image_D2_deep_minus_shallow_materialization_count":0,"formal_estimate_pvalue_CI_Holm_gatekeeping_scientific_decision_count":0,"S4_NNS_count":0,"S4_occlusion_count":0,"S5_execution_count":0},commit),"recovery_launch.json":sign("s3_r5_recovery_launch","PASS",{"launch_id":"s3_r5_cpu_rebind_launch_001","CPU_only":True,"CUDA_VISIBLE_DEVICES":"","terminal_count":10458,"scalar_correction_count":3424,"elapsed_seconds":time.monotonic()-started,"fresh_launch":True},commit)}
    for name,doc in docs.items():s3.write_json_new(ROOT/name,doc)
    print(json.dumps({"status":"PASS","terminal_count":10458,"scalar_correction_count":3424,"states":counts,"elapsed_seconds":time.monotonic()-started},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
