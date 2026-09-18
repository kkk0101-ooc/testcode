#!/usr/bin/env python3
"""CPU-only independent validation of Endpoint Scanner EP2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"reports/nns_vit_endpoint_ep2_r1_20260918";EP0=REPO/"manifests/nns_vit_endpoint/ep0_r1_20260918";EP1_COMMIT="ea7b7437f1e58389868d9cb6206d7e9681a81233"
sys.path.insert(0,str(REPO/"src"))
from vitnns.endpoint_ep2 import select_model_M  # noqa:E402

def require(c,m):
    if not c:raise ValueError(m)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def load(path):return json.loads(path.read_text())

def validate(root:Path)->dict:
    root=root.resolve();report=load(root/"calibration_report.json");terminal=load(root/"terminal_receipt.json");manifest=load(root/"artifact_manifest.json");material=load(root/"materialization_receipt.json");tasks=load(root/"task_manifest.json");coverage=load(root/"coverage_audit.json")
    require(report["status"]==terminal["status"]=="PASS_EP2_CALIBRATION_AND_SEAL","EP2_STATUS");require(report["scientific_results"]==terminal["scientific_results"]=="NOT_RUN_CALIBRATION_ONLY","SCIENTIFIC_SCOPE");require(not report["evaluation_access"] and not terminal["evaluation_access"] and not report["ep3_started"] and not terminal["ep3_started"],"ACCESS_OR_STAGE")
    require(subprocess.run(["git","merge-base","--is-ancestor",EP1_COMMIT,report["producer_commit"]],cwd=REPO).returncode==0,"EP1_ANCESTRY");require(terminal["producer_commit"]==report["producer_commit"],"PRODUCER_BINDING");require(terminal["calibration_report_digest"]==digest(report),"REPORT_DIGEST");require(terminal["manifest_digest"]==manifest["canonical_digest"]==digest(manifest["files"]),"MANIFEST_DIGEST")
    for row in manifest["files"]:
        path=(root/row["path"]).resolve();require(path.is_relative_to(root) and path.is_file(),"MANIFEST_PATH");require(path.stat().st_size==row["bytes"] and sha(path)==row["sha256"],"ARTIFACT_HASH:"+row["path"])
    for row in report["implementation_bindings"]:require(sha(REPO/row["path"])==row["sha256"],"IMPLEMENTATION_DRIFT:"+row["path"]);require(subprocess.run(["git","cat-file","-e",f"{report['producer_commit']}:vitnns_rebuild/{row['path']}"],cwd=REPO).returncode==0,"IMPLEMENTATION_NOT_COMMITTED")
    cohort=load(EP0/"cohort_registry.json");expected=sorted((row for row in cohort["rows"] if row["split"]=="calibration"),key=lambda row:row["split_order"]);require(len(expected)==64 and material["authorized_member_count"]==material["archive_member_reads"]==64,"CALIBRATION_COUNT");require(material["discovery_member_reads"]==material["evaluation_member_reads"]==material["replacement_count"]==0,"OUT_OF_SCOPE_ACCESS");require(material["records_digest"]==digest(material["records"]),"MATERIALIZATION_DIGEST")
    for frozen,actual in zip(expected,material["records"]):
        require((actual["split_order"],actual["sample_id"],actual["class_id"],actual["wnid"],actual["source_object_id"])==(frozen["split_order"],frozen["sample_id"],frozen["class_id"],frozen["wnid"],frozen["source_object_id"]),"COHORT_DRIFT");path=REPO/actual["cache_path"];require(path.stat().st_size==actual["bytes"] and sha(path)==actual["sha256"],"CACHE_HASH")
    require(tasks["task_count"]==128 and tasks["calibration_images"]==64 and tasks["model_count"]==tasks["targets_per_task"]==2 and not tasks["evaluation_access"],"TASK_SCOPE");require(tasks["manifest_digest"]==digest(tasks["tasks"]),"TASK_DIGEST");require(coverage["status"]=="PASS" and coverage["task_count"]==coverage["completed_count"]==128 and not coverage["missing_task_ids"] and not coverage["invalid"] and not coverage["duplicate_task_ids"],"COVERAGE")
    with sqlite3.connect(root/"queue.sqlite3") as connection:queue=connection.execute("SELECT task_id,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id").fetchall();attempts=connection.execute("SELECT task_id,attempt,status FROM attempts ORDER BY task_id,attempt").fetchall()
    require(len(queue)==128 and all(row[1]=="completed" for row in queue) and len([row for row in attempts if row[2]=="COMPLETED"])==128,"QUEUE")
    task_results=[]
    for task_id,_state,attempt,relative,expected_digest in queue:
        value=load(root/"results"/relative);unsigned=dict(value);observed=unsigned.pop("canonical_digest");require(observed==expected_digest==digest(unsigned) and value["task_id"]==task_id and value["attempt"]==attempt,"RESULT_IDENTITY");task_results.append(value["result"])
    require({row["model_id"] for row in task_results}=={"vit_b16","deit_s16"} and all(row["input_shape"]==[1,3,224,224] and row["dtype"]=="torch.float32" and row["alpha_chunk"]==8 and row["target_batch_probe"]==16 and row["retained_maps"] is False and row["finite"] for row in task_results),"TASK_RESULT_CONTRACT")
    decisions={}
    for mid in ("vit_b16","deit_s16"):
        rows=[];subset=[task for task in task_results if task["model_id"]==mid];require(len(subset)==64 and sorted(row["split_order"] for row in subset)==list(range(64)),"MODEL_IMAGE_COVERAGE")
        for task in subset:
            require(len(task["records"])==10,"PAIR_RECORD_COUNT")
            for row in task["records"]:rows.append({**row,"sample_id":task["sample_id"],"split_order":task["split_order"]})
        decisions[mid]=select_model_M(rows)
    require(decisions==report["model_decisions"],"DECISION_RECOMPUTE");require(all(value["status"]=="PASS" and value["selected_M"] in [32,64,128,256,512] for value in decisions.values()),"MODEL_SELECTION");require(report["selected_parameters"]=={mid:{"M":value["selected_M"],"alpha_chunk":8,"target_batch":16} for mid,value in decisions.items()},"PARAMETER_SEAL")
    identities=[row["worker_identity"] for row in task_results];require(all(row["CUBLAS_WORKSPACE_CONFIG"]==":4096:8" and row["startup_matmul_backward"]=="PASS" and row["tf32"] is False for row in identities),"CUDA_IDENTITY");require(1<=report["workers"]<=8 and len({row["gpu_uuid"] for row in identities})==report["workers"],"WORKER_COUNT");require(report["gpu_hours_upper_observed"]<=16 and report["peak_reserved_gpu_bytes_max"]<=25769803776 and report["retained_bytes_before_report"]<=25769803776,"RESOURCE_CAP")
    return {"status":report["status"],"producer_commit":report["producer_commit"],"calibration_images":64,"tasks":128,"models":report["selected_parameters"],"workers":report["workers"],"evaluation_access":False,"next_authorization":report["next_authorization"]}
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("root",nargs="?",type=Path,default=ROOT);print(json.dumps(validate(p.parse_args().root),sort_keys=True))
