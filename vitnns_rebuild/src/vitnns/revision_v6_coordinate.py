"""Three-coordinate local differential maps on the unchanged fixed targets."""
import numpy as np
import torch
from .endpoint_m4 import explicit_graph
from .endpoint_nns import _frozen_eval
COORDINATES=('normalized_endpoint','raw_gradient','common_origin')
OBJECTS=('Q','K','V','attention_output','attention_identity','mlp_output','mlp_identity')
def scan(model,image,task,progress=lambda:None,retain_raw=False):
    clean=image.detach().clone().requires_grad_(True)
    sigma=torch.tensor(task['std'],device=image.device).reshape(3,1,1)
    mu=torch.tensor(task['mean'],device=image.device).reshape(3,1,1)
    weights=(clean.detach()[0],1/sigma,clean.detach()[0]+mu/sigma)
    rows=[];patches=[];raw={};observed={};handles=[];post={}
    def emit(grad,block,obj,tokens,channels):
        if grad.ndim==3:grad=grad[None]
        if not bool(torch.isfinite(grad).all()):raise ValueError('NONFINITE_GRADIENT')
        for coord,w in zip(COORDINATES,weights):
            maps=grad*w
            pm=maps.abs().reshape(len(maps),3,14,16,14,16).sum((1,3,5)).reshape(len(maps),196).detach().cpu().numpy()
            if not np.isfinite(pm).all():raise ValueError('NONFINITE_PATCH_MASS')
            patches.extend(pm)
            rows.extend((block,obj,coord,t,c) for t,c in zip(tokens,channels))
            if retain_raw:
                # Frozen first scalar in each chunk, plus every structural map, checks spatial aggregation directly.
                key=str(len(rows)-len(maps));raw[key]=maps[0].detach().cpu().numpy()
        progress()
    try:
        for b in range(12):handles.append(model.blocks[b].register_forward_hook(lambda m,a,y,b=b:post.__setitem__(b,y)))
        with _frozen_eval(model),explicit_graph(model,observed),torch.enable_grad():
            logits=model(clean);score=logits[0,task['class_id']]
            if task['kind']=='units':
                for b in range(12):
                    units=[(t,c) for t in task['tokens'] for c in task['channels'][str(b)]]
                    for start in range(0,len(units),16):
                        batch=units[start:start+16];values=torch.stack([post[b][0,t,c] for t,c in batch])
                        g=torch.autograd.grad(values,clean,grad_outputs=torch.eye(len(batch),device=image.device),is_grads_batched=True,retain_graph=True)[0][:,0]
                        emit(g,b,'unit',[t for t,c in batch],[c for t,c in batch])
            else:
                full=torch.autograd.grad(score,clean,retain_graph=True)[0][0]
                for b in range(12):
                    e=observed[b]
                    gq,gk,gv,g1,g2=torch.autograd.grad(score,tuple(e[k] for k in ['query','key','value','r1','r2']),retain_graph=True)
                    for obj,key,g in [('Q','query',gq),('K','key',gk),('V','value',gv),('attention_identity','u',g1),('mlp_identity','r1',g2)]:
                        grad=torch.autograd.grad(e[key],clean,grad_outputs=g.detach(),retain_graph=True)[0][0]
                        emit(grad,b,obj,[-1],[-1])
                    emit(full,b,'attention_output',[-1],[-1]);emit(full,b,'mlp_output',[-1],[-1])
        packed=dict(block=np.array([r[0] for r in rows],dtype=np.int8),object=np.array([r[1] for r in rows],dtype='U20'),coordinate=np.array([r[2] for r in rows],dtype='U24'),token=np.array([r[3] for r in rows],dtype=np.int16),channel=np.array([r[4] for r in rows],dtype=np.int16),patch_mass=np.asarray(patches,dtype=np.float32))
        return packed,raw
    finally:
        for handle in handles:handle.remove()
