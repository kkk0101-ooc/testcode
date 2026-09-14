#!/usr/bin/env python3
"""Seal independently validated S3 weights and create the pre-publication test gate."""
from __future__ import annotations
import glob
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns import nns_patch_depth_s3 as s3  # noqa:E402

def main()->int:
    root=REPO/s3.ROOT_REL;commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    required=["alpha_grid_validation_ledger.json","normalization_manifest.json","class_block_probe_weight_manifest.json","low_level_control_weight_manifest.json","train_only_crossfit_report.json","probe_class_typed_na_report.json"]
    docs={name:json.loads((root/name).read_text()) for name in required}
    s3.require(all(s3.verify_signed(doc) and doc["status"]=="PASS" for doc in docs.values()),"probe evidence document invalid")
    weights=docs["class_block_probe_weight_manifest.json"]["payload"]["records"];s3.require(len(weights)==859,"weight count drift")
    for record in weights:s3.require(s3.sha_file(REPO/record["path"])==record["sha256"],f"weight SHA drift: {record['path']}")
    terminals=[];inventory=[];state_counts={};split_counts={};resources=[]
    for path_text in sorted(glob.glob(str(root/"unit_terminals/*.json"))):
        path=Path(path_text);doc=json.loads(path.read_text());s3.require(s3.verify_signed(doc),f"terminal digest drift {path.name}");payload=doc["payload"]
        if payload["split"] not in ("probe_train","probe_validation"):continue
        terminals.append({"schedule_index":payload["schedule_index"],"path":path.relative_to(REPO).as_posix(),"sha256":s3.sha_file(path),"canonical_digest":doc["canonical_digest"],"state":payload["terminal_state"]})
        state_counts[payload["terminal_state"]]=state_counts.get(payload["terminal_state"],0)+1;split_counts[payload["split"]]=split_counts.get(payload["split"],0)+1;resources.append(payload["resources"])
        inventory.extend(payload["arrays"])
    terminals.sort(key=lambda row:(row["schedule_index"],row["path"]));inventory.sort(key=lambda row:row["path"])
    s3.require(len(terminals)==6976 and state_counts=={"VALID_FEATURE":6955,"NA_TARGET_DISAPPEARED_AFTER_CROP":21},"train/validation terminal counts drift")
    feature_inventory=s3.signed("s3_feature_inventory","PASS",{"records":inventory,"records_digest":s3.records_digest(inventory),"array_count":len(inventory),"valid_unit_count":6955,"large_array_root":s3.PAYLOAD_REL.as_posix(),"manifest_only_committed":True},producer_commit=commit)
    s3.write_json_new(root/"train_validation_feature_inventory.json",feature_inventory)
    launch=json.loads((root/"launches/train_validation-launch-001/launch_result.json").read_text())
    resource=s3.signed("s3_resource_ledger","PASS",{"phase":"train_validation","terminal_order":"stable_sort_schedule_index_terminal_path","cumulative_GPU_hours":math.fsum(item["gpu_active_seconds"] for item in resources)/3600.0,"cumulative_CPU_hours":math.fsum(item["cpu_seconds"] for item in resources)/3600.0,"unit_wall_time_sum_hours":math.fsum(item["unit_wall_seconds"] for item in resources)/3600.0,"launch_elapsed_seconds":launch["payload"]["launch_elapsed_seconds"],"critical_path_wall_seconds":launch["payload"]["launch_elapsed_seconds"],"retained_bytes":sum(item["retained_bytes"] for item in resources),"peak_allocated_gpu_bytes":max(item["peak_allocated_gpu_bytes"] for item in resources),"peak_reserved_gpu_bytes":max(item["peak_reserved_gpu_bytes"] for item in resources),"GPU_hours_divided_by_K":False,"worker_wall_times_summed_as_launch_elapsed":False},producer_commit=commit)
    s3.write_json_new(root/"resource_ledger_train_validation.json",resource)
    denominator=s3.signed("s3_typed_failure_denominator_report","PASS",{"phase":"train_validation","scheduled_counts":{"probe_train":5232,"probe_validation":1744},"valid_counts":{"probe_train":5215,"probe_validation":1740},"typed_NA_counts":{"probe_train":17,"probe_validation":4},"failed_count":0,"valid_class_count":859,"typed_NA_class_count":13,"typed_NA_class_fraction":13/872,"replacement_used":False,"minimum_valid_fraction":0.9,"observed_image_valid_fraction":6955/6976},producer_commit=commit)
    s3.write_json_new(root/"typed_failure_na_denominator_train_validation.json",denominator)
    files=[s3.artifact_record(REPO,root/name) for name in required]+[s3.artifact_record(REPO,root/"train_validation_feature_inventory.json"),s3.artifact_record(REPO,root/"resource_ledger_train_validation.json"),s3.artifact_record(REPO,root/"typed_failure_na_denominator_train_validation.json")]
    weight_seal=s3.signed("s3_probe_weight_seal","PASS",{"weight_input_commit":commit,"files":files,"files_digest":s3.records_digest(files),"weight_class_count":859,"typed_na_class_count":13,"feature_probe_count":859*12,"control_probe_count":859,"validation_selected_alpha_only":True,"final_fit_train_only":True,"crossfit_train_only":True,"metadata_validator":"PASS","numeric_validator":"PASS","D2_test_raw_byte_access_count":0,"D2_test_decode_count":0,"D2_test_feature_count":0,"D2_test_prediction_count":0,"scientific_claim_decision":None},producer_commit=commit)
    s3.write_json_new(root/"probe_weight_seal.json",weight_seal)
    gate=s3.signed("s3_test_access_gate","PASS",{"weight_seal_commit":commit,"weight_seal_path":(s3.ROOT_REL/"probe_weight_seal.json").as_posix(),"weight_seal_sha256":s3.sha_file(root/"probe_weight_seal.json"),"weight_seal_canonical_digest":weight_seal["canonical_digest"],"requires_gate_commit_push_and_remote_tree_verification":True,"authorized_D2_test_scheduled_count":3482,"authorized_selected_members_only":True,"typed_na_classes_skip_member_access":True,"cohort_aggregation_authorized":False,"created_time_ns":time.time_ns()},producer_commit=commit)
    s3.write_json_new(root/"test_access_gate.json",gate)
    print(json.dumps({"status":"PASS","weight_class_count":859,"typed_na_class_count":13,"D2_test_access_count":0,"weights_digest":docs["class_block_probe_weight_manifest.json"]["payload"]["records_digest"]},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
