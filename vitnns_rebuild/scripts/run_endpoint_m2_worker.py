#!/usr/bin/env python3
import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    raise RuntimeError('CUBLAS_PREIMPORT')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from run_endpoint_e4_intervention_worker import load_model,load_inputs,worker_identity
from vitnns.endpoint_ep1_queue import claim_task,complete_task,heartbeat,digest
from vitnns.endpoint_m2_compute import scan,close
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker-id',required=True);p.add_argument('--physical-index',type=int,required=True)
    p.add_argument('--gpu-uuid',required=True);args=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(args.physical_index)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;device=torch.device('cuda:0')
    identity=worker_identity(device,args);models={}
    while not (args.root/'STOP_FAILURE').exists():
        claim=claim_task(args.root/'queue.sqlite3',worker_id=args.worker_id,
            device_uuid=args.gpu_uuid,lease_seconds=1800,max_attempts=1)
        if claim is None: return
        task=claim['payload'];started=time.monotonic();last=started
        try:
            if task['model_id'] not in models: models[task['model_id']]=load_model(task,device)
            model,binding=models[task['model_id']];image,refs,original=load_inputs(task,device)
            def progress():
                nonlocal last
                if time.monotonic()-last>20:
                    heartbeat(args.root/'queue.sqlite3',claim,lease_seconds=1800);last=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            result=scan(model,image,refs,task,progress)
            source_bytes=(REPO/task['e3_path']).read_bytes()
            assert hashlib.sha256(source_bytes).hexdigest()==task['e3_sha256']
            source=json.loads(source_bytes);canonical=source.pop('canonical_digest')
            assert digest(source)==canonical
            e3=source['result']['blocks'][task['block']]['components']
            parity=[]
            for scores in result['scores']:
                if scores['geometry']=='patch' and scores['reference']=='normalized_zero':
                    h=scores['head'];values=scores['scores']
                    expected=e3[f'head_{h}']['absolute_mass']
                    observed=sum(values['endpoint_absolute'])
                    parity.append(dict(head=h,error=close(observed,expected,'E3_MASS'),expected=expected,observed=observed))
            result['e3_parity']=parity
            torch.cuda.synchronize()
            result.update({k:task[k] for k in ('model_id','sample_id','split_order','class_id','block','image_sha256')})
            result.update(worker_identity=identity,model_binding=binding,seconds=time.monotonic()-started,
                peak_gpu_bytes=torch.cuda.max_memory_reserved(),reference_identity={k:hashlib.sha256(
                    v.cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in refs.items()})
            complete_task(args.root/'queue.sqlite3',args.root/'results',claim,result)
            print(json.dumps(dict(task=claim['task_id'],seconds=result['seconds'],peak_gpu_bytes=result['peak_gpu_bytes'])),flush=True)
        except Exception as error:
            record=dict(task=claim['task_id'],error_type=type(error).__name__,message=str(error))
            (args.root/('failure_'+claim['task_id']+'.json')).write_text(json.dumps(record,indent=2))
            (args.root/'STOP_FAILURE').touch()
            raise

if __name__=='__main__': main()
