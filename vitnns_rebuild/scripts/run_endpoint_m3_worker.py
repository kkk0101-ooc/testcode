#!/usr/bin/env python3
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("CUBLAS_PREIMPORT")
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np,torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from run_endpoint_e4_intervention_worker import load_model,load_inputs,worker_identity
from vitnns.endpoint_ep1_queue import claim_task,complete_task,heartbeat
from vitnns.endpoint_m3_atlas import scan_model,pack_rows
from vitnns.endpoint_m3_protocol import rademacher

def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--worker-id",required=True);p.add_argument("--physical-index",type=int,required=True);p.add_argument("--gpu-uuid",required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;device=torch.device("cuda:0");identity=worker_identity(device,a);models={}
 while not (a.root/"STOP_FAILURE").exists():
  claim=claim_task(a.root/"queue.sqlite3",worker_id=a.worker_id,device_uuid=a.gpu_uuid,lease_seconds=3600,max_attempts=1)
  if claim is None:return
  task=claim["payload"];started=time.monotonic();last=started
  try:
   if task["model_id"] not in models:models[task["model_id"]]=load_model(task,device)
   model,binding=models[task["model_id"]];image,_,_=load_inputs(task,device)
   def progress():
    nonlocal last
    if time.monotonic()-last>20:heartbeat(a.root/"queue.sqlite3",claim,lease_seconds=3600);last=time.monotonic()
   direction_arrays=tuple(rademacher((3,224,224),task["direction_namespace"],task["model_id"],task["sample_id"],i)/np.sqrt(3*224*224) for i in range(task["directions"]))
   epsilons=tuple(task["finite_difference_epsilons_float32"]);torch.cuda.reset_peak_memory_stats();rows,full,validation=scan_model(model,image,tuple(task["tokens"]),{int(k):tuple(v) for k,v in task["channels"].items()},task["gradient_chunk_size"],progress,task["save_full_maps"],direction_arrays,epsilons)
   task_dir=a.root/"task_payloads"/claim["task_id"];task_dir.mkdir(parents=True)
   metrics_path=task_dir/"metrics.npz";np.savez_compressed(metrics_path,**pack_rows(rows))
   full_path=None
   if full:full_path=task_dir/"full_maps.npz";np.savez_compressed(full_path,**full)
   result={"stage":"ENDPOINT-M3","model_id":task["model_id"],"sample_id":task["sample_id"],"image_sha256":task["image_sha256"],"cohort_order":task["cohort_order"],"target_count":len(rows),"zero_gradient_count":sum(r["zero_gradient"] for r in rows),"metrics_path":metrics_path.relative_to(REPO).as_posix(),"metrics_sha256":hashlib.sha256(metrics_path.read_bytes()).hexdigest(),"full_maps_path":None if full_path is None else full_path.relative_to(REPO).as_posix(),"full_maps_sha256":None if full_path is None else hashlib.sha256(full_path.read_bytes()).hexdigest(),"validation":validation,"model_binding":binding,"worker_identity":identity,"seconds":time.monotonic()-started,"peak_gpu_bytes":torch.cuda.max_memory_reserved(),"peak_cpu_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
   complete_task(a.root/"queue.sqlite3",a.root/"results",claim,result);print(json.dumps({"task":claim["task_id"],"seconds":result["seconds"],"peak_gpu_bytes":result["peak_gpu_bytes"]}),flush=True)
  except Exception as e:
   (a.root/f"failure_{claim['task_id']}.json").write_text(json.dumps({"type":type(e).__name__,"message":str(e)},indent=2));(a.root/"STOP_FAILURE").touch();raise
if __name__=="__main__":main()
