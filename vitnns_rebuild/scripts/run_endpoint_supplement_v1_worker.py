#!/usr/bin/env python3
import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('CUBLAS_PREIMPORT')
import argparse,fcntl,gzip,hashlib,json,sys,time,traceback
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np,torch,timm
from timm.layers import set_fused_attn
from run_endpoint_e4_intervention_worker import load_model,load_inputs,worker_identity
from vitnns.endpoint_ep1_queue import claim_task,complete_task,heartbeat,canonical
from vitnns.endpoint_m3_atlas import scan_model,pack_rows
from vitnns.endpoint_m4 import attention_jvp_diagnostics
from vitnns.endpoint_supplement_v1 import scan_endpoint_maps,feature_distributions
from vitnns.recovery_v2 import canonical_state_dict_tensor_digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--physical-index',type=int,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--worker-id',required=True);a=p.parse_args()
 lock=open(REPO/f'.endpoint_supplement_gpu{a.physical_index}.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(a.physical_index)
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 device=torch.device('cuda:0');identity=worker_identity(device,a);models={}
 sources=json.loads((a.root/'source_bindings.json').read_text())
 for rel in ('src/vitnns/endpoint_supplement_v1.py','scripts/run_endpoint_supplement_v1_worker.py'):
  assert hashlib.sha256((REPO/rel).read_bytes()).hexdigest()==sources[rel]
 while not (a.root/'STOP_FAILURE').exists():
  claim=claim_task(a.root/'queue.sqlite3',worker_id=a.worker_id,device_uuid=a.gpu_uuid,lease_seconds=1800,max_attempts=1)
  if claim is None:return
  t=claim['payload'];started=time.monotonic();last=started
  try:
   key=(t['model_id'],t['model_state'],t['init_seed'])
   if key not in models:
    if t['model_state']=='trained':model,binding=load_model(t,device)
    else:
     set_fused_attn(False);torch.manual_seed(t['init_seed']);model=timm.create_model(t['timm_name'],pretrained=False).eval()
     for b in model.blocks:b.attn.fused_attn=False
     binding=dict(init_seed=t['init_seed'],tensor_digest=canonical_state_dict_tensor_digest(model.state_dict()),parameter_count=sum(p.numel() for p in model.parameters()))
     model=model.to(device)
    models[key]=(model,binding)
   model,binding=models[key];image,_,_=load_inputs(t,device)
   def progress():
    nonlocal last
    if time.monotonic()-last>20:heartbeat(a.root/'queue.sqlite3',claim,lease_seconds=1800);last=time.monotonic()
   torch.cuda.reset_peak_memory_stats();out=a.root/'payloads'/claim['task_id'];out.mkdir(parents=True)
   result=dict(study=t['study'],model_id=t['model_id'],model_state=t['model_state'],init_seed=t['init_seed'],sample_id=t['sample_id'],image_sha256=t['image_sha256'],cohort_order=t['cohort_order'],split_order=t['split_order'],binding=binding,worker_identity=identity)
   if t['study']=='random':
    rows,_,validation=scan_model(model,image,tuple(t['tokens']),{int(b):tuple(cs) for b,cs in t['channels'].items()},16,progress)
    np.savez_compressed(out/'m3.npz',**pack_rows(rows));result['m3_validation']=validation;result['m3_target_count']=len(rows)
    result['m4']=scan_endpoint_maps(model,image,class_id=t['class_id'],task_id=claim['task_id'],progress=progress)
    result['jvp']=attention_jvp_diagnostics(model,image,task_id=f"{t['model_id']}__{t['split_order']:03d}",progress=progress)
   elif t['study']=='target':
    with torch.no_grad():logits=model(image)[0];indices=torch.argsort(logits,descending=True,stable=True)[:2].tolist()
    top,runner=indices;result.update(gt=t['class_id'],top1=top,runnerup=runner,correct=top==t['class_id'])
    shared=dict(blocks=(0,5,11),task_id=f"{t['model_id']}__{t['split_order']:03d}",progress=progress)
    gt=scan_endpoint_maps(model,image,class_id=t['class_id'],**shared)
    result['targets']={'gt':gt,'top1':gt if result['correct'] else scan_endpoint_maps(model,image,class_id=top,**shared),'margin':scan_endpoint_maps(model,image,class_id=top,runnerup=runner,**shared)}
   else:
    rows=feature_distributions(model,image,t['channels'],claim['task_id']);progress()
    (out/'feature.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0));result['feature_count']=len(rows)
   result['payload_files']=[dict(path=str(f.relative_to(REPO)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(out.iterdir())]
   result.update(seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_reserved())
   complete_task(a.root/'queue.sqlite3',a.root/'results',claim,result)
   print(json.dumps(dict(task=claim['task_id'],seconds=result['seconds'])),flush=True)
  except Exception as e:
   (a.root/f"failure_{claim['task_id']}.json").write_text(json.dumps(dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()),indent=2));(a.root/'STOP_FAILURE').touch();raise
if __name__=='__main__':main()
