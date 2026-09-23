#!/usr/bin/env python3
import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('CUBLAS_PREIMPORT')
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from run_endpoint_e4_intervention_worker import load_model,load_inputs,worker_identity
from vitnns.revision_v6_coordinate import scan
from vitnns.endpoint_ep1_queue import claim_task,complete_task,heartbeat

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker-id',required=True);p.add_argument('--physical-index',type=int,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
 root=a.root;protocol=json.loads((root/'protocol.json').read_text())
 for row in json.loads((root/'source_bindings.json').read_text())['scripts']:
  if sha(R/row['path'])!=row['sha256']:raise ValueError('FROZEN_SCRIPT_DRIFT:'+row['path'])
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 device=torch.device('cuda:0');identity=worker_identity(device,a);models={}
 queue=root/('pilot_queue.sqlite3' if a.pilot else 'queue.sqlite3');dest=root/('pilot' if a.pilot else 'tasks');dest.mkdir(exist_ok=True)
 while not (root/'STOP_FAILURE').exists():
  claim=claim_task(queue,worker_id=a.worker_id,device_uuid=a.gpu_uuid,lease_seconds=1800,max_attempts=1)
  if claim is None:return
  start=time.monotonic();task=claim['payload'];last=start
  try:
   if task['model_id'] not in models:
    models[task['model_id']]=load_model(task,device)[0]
   def progress():
    nonlocal last
    if time.monotonic()-last>20:heartbeat(queue,claim,lease_seconds=1800);last=time.monotonic()
    if time.monotonic()-start>1800:raise TimeoutError('TASK_BUDGET')
   image,_,_=load_inputs(task,device);torch.cuda.reset_peak_memory_stats();packed,raw=scan(models[task['model_id']],image,task,progress,retain_raw=a.pilot)
   expected=18432 if task['kind']=='units' else 252
   assert len(packed['block'])==expected and set(packed)=={'block','object','coordinate','token','channel','patch_mass'} and packed['patch_mass'].shape==(expected,196)
   path=dest/(claim['task_id']+'.npz');np.savez_compressed(path,**packed)
   rawpath=None
   if raw:rawpath=dest/(claim['task_id']+'_raw.npz');np.savez_compressed(rawpath,**raw)
   result=dict(schema_version=1,task_id=claim['task_id'],model_id=task['model_id'],image=task['sample_id'],kind=task['kind'],image_sha256=task['image_sha256'],rows=expected,output=str(path.relative_to(R)),output_sha256=sha(path),raw_output=None if rawpath is None else str(rawpath.relative_to(R)),raw_sha256=None if rawpath is None else sha(rawpath),seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_reserved(),peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,environment=identity)
   complete_task(queue,dest/'receipts',claim,result);print(json.dumps(dict(task=claim['task_id'],seconds=result['seconds'],bytes=path.stat().st_size)),flush=True)
  except Exception as e:
   (root/'STOP_FAILURE').write_text(str(e));(root/('failure_'+claim['task_id']+'.json')).write_text(json.dumps(dict(type=type(e).__name__,message=str(e),task=claim['task_id'])));raise
if __name__=='__main__':main()
