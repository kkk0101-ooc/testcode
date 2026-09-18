#!/usr/bin/env python3
"""Orchestrate the frozen Endpoint Scanner EP2 calibration stage."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

REPO=Path(__file__).resolve().parents[1];PYTHON=REPO.parent/".conda/envs/vitnns/bin/python";EP0=REPO/"manifests/nns_vit_endpoint/ep0_r1_20260918";EP1=REPO/"reports/nns_vit_endpoint_ep1_r11_20260918";SPEC=REPO/"config/nns_vit_endpoint/ep2_run_spec.json";RUNTIME=REPO.parent/".runtime/nns_vit_endpoint_ep2"
sys.path.insert(0,str(REPO/"src"))
from vitnns.endpoint_ep1_queue import audit_coverage,canonical,create_queue  # noqa:E402
from vitnns.endpoint_ep2 import select_model_M  # noqa:E402

ALLOWED_UNTRACKED={"config/nns_vit_cut_intervention/p0_c.json","manifests/nns_vit_cut_intervention/p0_c_operation_registry.json","scripts/validate_nns_vit_cut_intervention_p0_c.py","src/vitnns/nns_vit_cut_intervention_p0_c.py","src/vitnns/nns_vit_cut_intervention_p0_c_worker.py","tests/test_nns_vit_cut_intervention_p0_c.py"}
SOURCE_FILES=["config/nns_vit_endpoint/ep2_run_spec.json","docs/NNS_ViT_Endpoint_Scanner_EP2执行方案_20260918.md","src/vitnns/endpoint_ep2.py","scripts/build_endpoint_ep2_materialization.py","scripts/run_endpoint_ep2_worker.py","scripts/run_endpoint_ep2.py","scripts/validate_endpoint_ep2.py","tests/test_endpoint_ep2.py"]

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def load(path):return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
def git(*args):return subprocess.check_output(["git",*args],cwd=REPO,text=True).strip()
def remote(ref):
    out=subprocess.check_output(["git","ls-remote","origin",ref],cwd=REPO,text=True).split();return out[0] if len(out)==2 else ""

def startup():
    spec=load(SPEC);head=git("rev-parse","HEAD");branch=git("symbolic-ref","--short","HEAD")
    if branch!="stage-d-d3-pass" or head!=git("rev-parse","@{upstream}") or head!=remote("refs/heads/stage-d-d3-pass"):raise RuntimeError("EP2_PRODUCER_NOT_PUBLISHED")
    if subprocess.run(["git","merge-base","--is-ancestor",spec["predecessor_commit"],head],cwd=REPO).returncode:raise RuntimeError("EP1_NOT_ANCESTOR")
    validation=subprocess.run([str(PYTHON),"scripts/validate_endpoint_ep1.py"],cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),text=True,capture_output=True)
    if validation.returncode:raise RuntimeError("EP1_VALIDATION_FAILED:"+validation.stderr)
    status=git("status","--porcelain","--untracked-files=all");changed=[]
    repo_prefix=REPO.relative_to(REPO.parent).as_posix()+"/"
    for line in status.splitlines():
        path=line[3:].split(" -> ")[-1]
        if path.startswith(repo_prefix):path=path[len(repo_prefix):]
        changed.append(path)
    if set(changed)-ALLOWED_UNTRACKED:raise RuntimeError("UNEXPLAINED_WORKTREE:"+str(sorted(set(changed)-ALLOWED_UNTRACKED)))
    for relative in SOURCE_FILES:
        if subprocess.run(["git","cat-file","-e",f"{head}:vitnns_rebuild/{relative}"],cwd=REPO).returncode:raise RuntimeError("SOURCE_NOT_COMMITTED:"+relative)
    return spec,head,validation.stdout.splitlines()

def gpus():
    query=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.free,utilization.gpu,driver_version","--format=csv,noheader,nounits"],text=True);proc=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_memory","--format=csv,noheader,nounits"],text=True);active={}
    for line in proc.splitlines():
        if line.strip():u,p,m=[x.strip() for x in line.split(",")];active.setdefault(u,[]).append({"pid":int(p),"used_memory_mib":int(m)})
    rows=[]
    for line in query.splitlines():
        i,u,n,pci,total,free,util,driver=[x.strip() for x in line.split(",")];rows.append({"physical_index":int(i),"gpu_uuid":u,"name":n,"pci_bus_id":pci,"memory_total_mib":int(total),"memory_free_mib":int(free),"utilization_percent":int(util),"driver_version":driver,"existing_compute_processes":active.get(u,[]),"eligible":n=="NVIDIA H20" and int(free)>=32768 and int(util)<=10})
    return rows

def launch(queue,results,selected,root):
    processes=[];streams=[];started=time.monotonic()
    for index,gpu in enumerate(selected):
        worker=f"worker-{index:02d}";log=root/"logs"/f"{worker}.log";log.parent.mkdir(parents=True,exist_ok=True);stream=log.open("xb");streams.append(stream);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu["physical_index"]),CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONPATH=str(REPO/"src"),PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4");cmd=[str(PYTHON),"scripts/run_endpoint_ep2_worker.py","--queue",str(queue),"--results",str(results),"--worker-id",worker,"--physical-index",str(gpu["physical_index"]),"--gpu-uuid",gpu["gpu_uuid"]];processes.append((worker,gpu,subprocess.Popen(cmd,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)))
    exits=[]
    for (worker,gpu,process),stream in zip(processes,streams):code=process.wait();stream.flush();os.fsync(stream.fileno());stream.close();exits.append({"worker_id":worker,"gpu_uuid":gpu["gpu_uuid"],"return_code":code})
    return exits,time.monotonic()-started

def main():
    p=argparse.ArgumentParser();p.add_argument("--max-gpus",type=int,default=8);a=p.parse_args();spec,head,prior=startup();root=REPO/spec["outputs"]["formal_root"];cache=REPO/spec["outputs"]["cache_root"]
    if root.exists() or cache.exists():raise RuntimeError("EP2_ROOT_OR_CACHE_EXISTS")
    if shutil.disk_usage(REPO).free<spec["runtime"]["minimum_free_disk_bytes"]:raise RuntimeError("PAUSED_RESOURCE_CAP_DISK")
    root.mkdir(parents=True);RUNTIME.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    material=subprocess.run([str(PYTHON),"scripts/build_endpoint_ep2_materialization.py","--cache-root",str(cache),"--receipt",str(root/"materialization_receipt.json")],cwd=REPO,text=True,capture_output=True)
    if material.returncode:raise RuntimeError("MATERIALIZATION_FAILED:"+material.stderr)
    materialization=load(root/"materialization_receipt.json");models=load(EP0/"model_registry.json")["models"];tasks=[]
    for row in materialization["records"]:
        for model in models:
            tasks.append({"task_id":f"{model['model_id']}__{row['split_order']:02d}","payload":{"model_id":model["model_id"],"timm_name":model["timm_name"],"checkpoint":model["checkpoint"]["path"],"checkpoint_sha256":model["checkpoint"]["sha256"],"checkpoint_bytes":model["checkpoint"]["bytes"],"sample_id":row["sample_id"],"split_order":row["split_order"],"class_id":row["class_id"],"wnid":row["wnid"],"cache_path":row["cache_path"],"image_sha256":row["sha256"],"image_bytes":row["bytes"],"mean":model["preprocessing"]["mean"],"std":model["preprocessing"]["std"],"alpha_chunk":8}})
    queue=root/"queue.sqlite3";identity=create_queue(queue,tasks);write(root/"task_manifest.json",{"tasks":tasks,**identity,"task_count":128,"calibration_images":64,"model_count":2,"targets_per_task":2,"evaluation_access":False})
    snapshot=gpus();eligible=[row for row in snapshot if row["eligible"]];idle=[row for row in eligible if not row["existing_compute_processes"]];selected=(idle if idle else eligible)[:max(1,min(a.max_gpus,8))]
    if not selected:raise RuntimeError("BLOCKED_INFRASTRUCTURE_NO_H20")
    lockroot=RUNTIME/"gpu_locks";lockroot.mkdir(parents=True,exist_ok=True);locks=[];locked=[]
    for gpu in selected:
        stream=(lockroot/f"{gpu['gpu_uuid']}.lock").open("a+")
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:stream.close();continue
        stream.seek(0);stream.truncate();stream.write(json.dumps({"pid":os.getpid(),"head":head,"root":str(root)}));stream.flush();os.fsync(stream.fileno());locks.append(stream);locked.append(gpu)
    selected=locked
    if not selected:raise RuntimeError("BLOCKED_INFRASTRUCTURE_NO_LOCKABLE_H20")
    write(root/"startup_snapshot.json",{"hostname":socket.gethostname(),"producer_commit":head,"remote_commit":remote("refs/heads/stage-d-d3-pass"),"ep1_validation":prior,"gpu_snapshot":snapshot,"selected_gpus":selected,"free_disk_bytes":shutil.disk_usage(REPO).free,"formal_image_reads":{"calibration":64,"discovery":0,"evaluation":0}})
    try:exits,worker_seconds=launch(queue,root/"results",selected,root)
    finally:
        for stream in locks:fcntl.flock(stream,fcntl.LOCK_UN);stream.close()
    coverage=audit_coverage(queue,root/"results");write(root/"coverage_audit.json",coverage)
    if any(row["return_code"] for row in exits) or coverage["status"]!="PASS":raise RuntimeError("EP2_WORKER_OR_COVERAGE_FAILED")
    task_results=[load(path)["result"] for path in sorted((root/"results").rglob("*.json"))];decisions={};flat=[]
    for mid in ("vit_b16","deit_s16"):
        subset=[]
        for task in task_results:
            if task["model_id"]!=mid:continue
            for row in task["records"]:subset.append({**row,"sample_id":task["sample_id"],"split_order":task["split_order"]})
        decisions[mid]=select_model_M(subset);flat.extend({"model_id":mid,**row} for row in subset)
    status="PASS_EP2_CALIBRATION_AND_SEAL" if all(value["status"]=="PASS" for value in decisions.values()) else "EP2_NUMERICAL_CALIBRATION_FAILED"
    gpu_hours=worker_seconds*len(selected)/3600;retained=sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    report={"status":status,"producer_commit":head,"scientific_results":"NOT_RUN_CALIBRATION_ONLY","evaluation_access":False,"ep3_started":False,"materialization_records_digest":materialization["records_digest"],"task_manifest_digest":identity["manifest_digest"],"coverage":coverage,"model_decisions":decisions,"selected_parameters":{mid:{"M":value["selected_M"],"alpha_chunk":8,"target_batch":16} for mid,value in decisions.items()},"records":flat,"workers":len(selected),"worker_exits":exits,"worker_wall_seconds":worker_seconds,"gpu_hours_upper_observed":gpu_hours,"peak_reserved_gpu_bytes_max":max(row["peak_reserved_gpu_bytes"] for row in task_results),"retained_bytes_before_report":retained,"implementation_bindings":[{"path":path,"sha256":sha(REPO/path)} for path in SOURCE_FILES],"next_authorization":"EP3_REQUIRES_EXPLICIT_USER_REQUEST_AND_PUBLISHED_EP2_PASS" if status.startswith("PASS") else "NONE_EP2_RECOVERY_REQUIRED"}
    write(root/"calibration_report.json",report);files=[{"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.name not in {"artifact_manifest.json","terminal_receipt.json"}];manifest={"files":files,"canonical_digest":digest(files)};write(root/"artifact_manifest.json",manifest);terminal={"status":status,"scientific_results":report["scientific_results"],"evaluation_access":False,"ep3_started":False,"producer_commit":head,"calibration_report_digest":digest(report),"manifest_digest":manifest["canonical_digest"],"next_authorization":report["next_authorization"],"elapsed_seconds":time.monotonic()-started};write(root/"terminal_receipt.json",terminal);print(json.dumps(terminal,sort_keys=True));return 0 if status.startswith("PASS") else 3
if __name__=="__main__":raise SystemExit(main())
