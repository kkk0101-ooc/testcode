#!/usr/bin/env python3
"""Post-result implementation diagnostic for the observed identity/Taylor contrast."""
import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
    raise RuntimeError('CUBLAS_PREIMPORT')
import json
from pathlib import Path
import sys
import time
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from run_endpoint_e4_intervention_worker import load_model,load_inputs
from vitnns.endpoint_m1_compute import capture,replace,missing_rows
from vitnns.endpoint_e4_intervention import component_tensor

torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
root=REPO/'reports/nns_vit_endpoint_m1_r1_20260918'
tasks=json.loads((root/'task_manifest.json').read_text())['tasks']
rows=[];started=time.monotonic()
for model_id in ('vit_b16','deit_s16'):
    task=next(t['payload'] for t in tasks if t['payload']['model_id']==model_id and t['payload']['split_order']==0)
    model,_=load_model(task,torch.device('cuda:0'));model.double()
    x,refs,_=load_inputs(task,torch.device('cuda:0'));x=x.double();refs={k:v.double() for k,v in refs.items()}
    _,clean,grad=capture(model,x,task['class_id'],True)
    reference={k:capture(model,v,task['class_id'])[1] for k,v in refs.items()}
    for r in missing_rows(model.blocks[0].attn.num_heads,True):
        if r['block'] not in (0,5,11):continue
        block=r['block'];cid=r['component_id'];z=component_tensor(clean[block],r);g=component_tensor(grad[block],r)
        for ref in refs:
            zr=component_tensor(reference[ref][block],r);d=z-zr;analytic=float((g*d).sum());curve=[]
            for eta in (.01,.001,.0001):
                plus=replace(model,x,task['class_id'],[r],{cid:z+eta*d})[cid][1]
                minus=replace(model,x,task['class_id'],[r],{cid:z-eta*d})[cid][1]
                estimate=(plus-minus)/(2*eta)
                curve.append(dict(eta=eta,finite_difference=estimate,error=abs(estimate-analytic)))
            condition=float((g*d).abs().sum());tol=1e-6+1e-5*condition
            row=dict(model=model_id,split_order=0,component=cid,reference=ref,dtype='float64',
                     gradient_point='clean',displacement='z_clean-z_reference',analytic=analytic,
                     condition=condition,tolerance=tol,curve=curve,pass_local=min(v['error'] for v in curve)<=tol)
            if r['operation'].startswith('identity'):
                b=model.blocks[block];original=b.forward;op=r['operation']
                def manual(u):
                    a=b.drop_path1(b.ls1(b.attn(b.norm1(u))))
                    r1=(zr if op=='identity1' else u)+a
                    m=b.drop_path2(b.ls2(b.mlp(b.norm2(r1))))
                    return (zr if op=='identity2' else r1)+m
                b.forward=manual
                try:
                    with torch.no_grad():expected=float(model(x)[0,task['class_id']])
                finally:b.forward=original
                actual=replace(model,x,task['class_id'],[r],{cid:zr})[cid][1]
                row['manual_skip_reference_error']=abs(actual-expected)
                row['pass_reference']=abs(actual-expected)<=1e-8
            rows.append(row)
    del model,clean,grad,reference;torch.cuda.empty_cache()
report=dict(status='PASS' if all(r['pass_local'] and r.get('pass_reference',True) for r in rows) else 'FAIL',
            scope='post_hoc_implementation_diagnostic_not_new_utility_evidence',
            selection='fixed_evaluation_order_0_each_model_blocks_0_5_11',
            checks=rows,check_count=len(rows),seconds=time.monotonic()-started,
            caveat='higher_order_effects_not_inferred_from_ranking_difference; no universal_Taylor_claim')
with (root/'taylor_implementation_diagnostic.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
assert report['status']=='PASS'
