#!/usr/bin/env python3
"""One GPU worker for EP2 64-image numerical calibration."""
from __future__ import annotations

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import hashlib
import io
import json
import platform
import sqlite3
import sys
import time
from pathlib import Path

import timm
import torch
import torchvision
from PIL import Image
from timm.layers import set_fused_attn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns.endpoint_ep1_queue import canonical,claim_task,complete_task,heartbeat  # noqa:E402
from vitnns.endpoint_ep2 import compare_pair,integrate_logits  # noqa:E402
from vitnns.endpoint_vit import EndpointTarget,scan_endpoint_vit  # noqa:E402
from vitnns.recovery_v2 import canonical_state_dict_tensor_digest  # noqa:E402

def sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def load_model(spec,device):
    path=REPO/spec["checkpoint"]
    if path.stat().st_size!=spec["checkpoint_bytes"] or sha_file(path)!=spec["checkpoint_sha256"]:raise ValueError("CHECKPOINT_IDENTITY")
    with path.open("rb") as f:loaded=torch.load(f,map_location="cpu",weights_only=True)
    state=loaded["model"] if isinstance(loaded,dict) and "model" in loaded else loaded
    set_fused_attn(False);model=timm.create_model(spec["timm_name"],pretrained=False).eval();out=model.load_state_dict(state,strict=True)
    if out.missing_keys or out.unexpected_keys:raise ValueError("STRICT_LOAD")
    for block in model.blocks:block.attn.fused_attn=False
    return model.to(device),{"tensor_digest":canonical_state_dict_tensor_digest(state),"key_count":len(state)}

def abandon_task(path,claim,error):
    with sqlite3.connect(path,isolation_level=None,timeout=30) as connection:
        connection.execute("PRAGMA journal_mode=WAL");connection.execute("BEGIN IMMEDIATE")
        current=connection.execute("SELECT state,attempt,fencing_token,worker_id,device_uuid FROM tasks WHERE task_id=?",(claim["task_id"],)).fetchone()
        valid=current is not None and current[0]=="claimed" and (int(current[1]),int(current[2]),current[3],current[4])==(claim["attempt"],claim["fencing_token"],claim["worker_id"],claim["device_uuid"])
        if not valid:connection.rollback();raise RuntimeError("STALE_FENCING_TOKEN")
        terminal=claim["attempt"]>=2;connection.execute("UPDATE attempts SET status=?,completed_ns=? WHERE task_id=? AND attempt=?",("FAILED" if terminal else "RETRYABLE",time.time_ns(),claim["task_id"],claim["attempt"]));connection.execute("UPDATE tasks SET state=?,fencing_token=NULL,worker_id=NULL,device_uuid=NULL,lease_until_ns=NULL,heartbeat_ns=NULL WHERE task_id=?",("failed" if terminal else "pending",claim["task_id"]));connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",(f"failure:{claim['task_id']}:{claim['attempt']}",canonical(error).decode()));connection.commit()

def image_tensor(task,device):
    path=REPO/task["cache_path"];data=path.read_bytes()
    if len(data)!=task["image_bytes"] or hashlib.sha256(data).hexdigest()!=task["image_sha256"]:raise ValueError("IMAGE_IDENTITY")
    with Image.open(io.BytesIO(data)) as source:
        if source.format!="JPEG":raise ValueError("IMAGE_FORMAT")
        rgb=source.convert("RGB");original_size=list(rgb.size);resized=TF.resize(rgb,248,interpolation=InterpolationMode.BICUBIC,antialias=True);cropped=TF.center_crop(resized,[224,224]);value=TF.pil_to_tensor(cropped).float().div(255).unsqueeze(0)
    mean=torch.tensor(task["mean"]).reshape(1,3,1,1);std=torch.tensor(task["std"]).reshape(1,3,1,1)
    return ((value-mean)/std).to(device),original_size

def execute(model,task,claim,queue,device):
    image,original_size=image_tensor(task,device);baseline=torch.zeros_like(image)
    with torch.no_grad():clean_logits=model(image);classes={"ground_truth_logit":task["class_id"],"frozen_clean_top1_logit":int(clean_logits[0].argmax())}
    target_batch=[EndpointTarget(f"unit_{i}",f"blocks.{i%12}",kind="token_channel",token=1,channel=i % model.embed_dim) for i in range(16)]
    torch.cuda.reset_peak_memory_stats();probe=scan_endpoint_vit(model,image,target_batch)
    if len(probe.units)!=16:raise ValueError("TARGET_BATCH_16")
    maps={};records=[];started=time.monotonic();last_heartbeat=time.monotonic()
    def progress():
        nonlocal last_heartbeat
        if time.monotonic()-last_heartbeat>=25:
            heartbeat(queue,claim);last_heartbeat=time.monotonic()
    for steps in (32,64,128,256,512,1024):
        current=integrate_logits(model,image,classes.values(),steps=steps,alpha_chunk=task["alpha_chunk"],progress=progress)
        maps[steps]=current
    for name,class_id in classes.items():
        for steps in (32,64,128,256,512):
            row=compare_pair(maps[steps][class_id],maps[2*steps][class_id]);row["M"]=steps;row.update({"target_kind":name,"class_id":class_id});records.append(row)
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    return {"model_id":task["model_id"],"sample_id":task["sample_id"],"split_order":task["split_order"],"class_id":task["class_id"],"clean_top1":classes["frozen_clean_top1_logit"],"image_sha256":task["image_sha256"],"original_size_wh":original_size,"input_shape":list(image.shape),"dtype":str(image.dtype),"alpha_chunk":task["alpha_chunk"],"target_batch_probe":16,"records":records,"finite":all(row["finite"] for row in records),"elapsed_seconds":elapsed,"peak_reserved_gpu_bytes":torch.cuda.max_memory_reserved(),"retained_maps":False}

def main():
    p=argparse.ArgumentParser();p.add_argument("--queue",type=Path,required=True);p.add_argument("--results",type=Path,required=True);p.add_argument("--worker-id",required=True);p.add_argument("--physical-index",type=int,required=True);p.add_argument("--gpu-uuid",required=True);a=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(a.physical_index):raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;device=torch.device("cuda:0")
    probe=torch.arange(16,device=device,dtype=torch.float32).reshape(4,4).requires_grad_();(probe@probe.T).sum().backward();torch.cuda.synchronize()
    identity={"sys_executable_realpath":str(Path(sys.executable).resolve()),"python_version":platform.python_version(),"torch_version":torch.__version__,"torchvision_version":torchvision.__version__,"timm_version":timm.__version__,"cuda_version":torch.version.cuda,"gpu_uuid":a.gpu_uuid,"physical_index":a.physical_index,"CUBLAS_WORKSPACE_CONFIG":os.environ["CUBLAS_WORKSPACE_CONFIG"],"startup_matmul_backward":"PASS","tf32":False}
    models={};completed=0
    while True:
        claim=claim_task(a.queue,worker_id=a.worker_id,device_uuid=a.gpu_uuid,lease_seconds=180,max_attempts=2)
        if claim is None:break
        task=claim["payload"]
        if task["model_id"] not in models: ambigu=load_model(task,device);models[task["model_id"]]=ambigu
        model,binding=models[task["model_id"]]
        try:
            result=execute(model,task,claim,a.queue,device);result.update({"worker_identity":identity,"model_binding":binding});complete_task(a.queue,a.results,claim,result);completed+=1
        except Exception as error:
            abandon_task(a.queue,claim,{"error_type":type(error).__name__,"message":str(error)});print(json.dumps({"status":"TASK_FAILED","task_id":claim["task_id"],"attempt":claim["attempt"],"error_type":type(error).__name__,"message":str(error)},sort_keys=True),flush=True)
    print(json.dumps({"status":"PASS","completed":completed,"worker":a.worker_id},sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
