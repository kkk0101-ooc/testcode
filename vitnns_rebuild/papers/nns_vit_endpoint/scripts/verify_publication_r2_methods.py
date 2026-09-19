#!/usr/bin/env python3
"""Independent native-forward recomputation of six qualitative method comparisons."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import sys
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'papers/nns_vit_endpoint/publication_r2_20260920'
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from run_endpoint_e4_intervention_worker import load_model,load_inputs

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    tasks=json.loads((REPO/'reports/nns_vit_endpoint_e4_intervention_r2_20260918/task_manifest.json').read_text())['tasks'];specs={t['task_id']:t['payload'] for t in tasks};results=[]
    for mid in ('deit_s16','vit_b16'):
        model,_=load_model(specs[mid+'__014'],torch.device('cpu'))
        if mid=='vit_b16':model.double()
        receipt=json.loads((ROOT/'evidence/compute_receipt.json').read_text())
        orders=receipt['figure19_orders'] if mid=='deit_s16' else sorted({4,14,15,*receipt['figure20_orders']})
        for order in orders:
            task=mid+f'__{order:03d}';spec=specs[task];d=np.load(ROOT/'assets'/f'{task}.npz');meta=json.loads((ROOT/'assets'/f'{task}.json').read_text());saved={};handles=[]
            def hook(k):
                def fn(m,args,out):saved[k]=out
                return fn
            for b,block in enumerate(model.blocks):handles.append(block.attn.attn_drop.register_forward_hook(hook(f'a{b}')))
            handles.append(model.blocks[-1].norm1.register_forward_hook(hook('norm')))
            if order==14:
                for b in (0,11):handles.append(model.blocks[b].register_forward_hook(hook(f'b{b}')))
            x,_,_=load_inputs(spec,torch.device('cpu'));x=x.to(next(model.parameters()).dtype).detach().requires_grad_(True);logits=model(x);f=logits[0,spec['class_id']]
            gx,g=torch.autograd.grad(f,(x,saved['norm']),retain_graph=True)
            endpoint=(x*gx)[0].detach().numpy()
            alpha=g[0,1:].detach().numpy().sum(0)/196
            cam=np.maximum(np.einsum('td,d->t',saved['norm'][0,1:].detach().numpy(),alpha),0).reshape(14,14)
            matrices=[saved[f'a{b}'][0].detach().double().numpy().mean(0) for b in range(12)]
            roll=np.eye(197)
            for a in matrices:
                transition=a+np.eye(197);transition/=transition.sum(1)[:,None];roll=transition.dot(roll)
            outputs={'class_endpoint':endpoint,'gradcam':cam,'raw_attention':matrices[-1][0,1:].reshape(14,14),'rollout':roll[0,1:].reshape(14,14)}
            errors={}
            for key,a in outputs.items():
                b=d[key];delta=float(np.max(np.abs(a-b)));errors[key]=delta
                assert np.allclose(a,b,rtol=2e-5,atol=1e-8),(task,key,delta)
            if order==14:
                selected={r['name']:r['channel'] for r in meta['selected_channels']}
                for b in (0,11):
                    h=saved[f'b{b}'][:,1:];ch=int(h.detach().var(1,unbiased=False).mean(0).argmax());assert ch==selected[f'b{b}']
                    assert np.array_equal(h[0,:,ch].detach().numpy().reshape(14,14),d[f'b{b}.energy.feature'])
                    s=h[:,:,ch].square().sum()/2;g0=torch.autograd.grad(s,x,retain_graph=True)[0]
                    a=(x*g0)[0].detach().numpy();assert np.allclose(a,d[f'b{b}.energy.endpoint'],rtol=2e-5,atol=1e-8)
            for h in handles:h.remove()
            results.append(dict(task_id=task,maximum_absolute_errors=errors,layer_checks=[0,11] if order==14 else []));print(task,errors,flush=True)
    report=dict(status='PASS',native_graph=True,comparison_tasks=len(results),layer_pair_checks=4,results=results)
    (ROOT/'qc/method_verification.json').write_text(json.dumps(report,indent=2)+chr(10))

if __name__=='__main__':main()
