"""Frozen-unit neuron maps for revision v5; learning maps and class paths stay distinct."""
import math
import numpy as np
import torch
from .endpoint_m3_atlas import endpoint_chunk

def statistics(maps):
    """Keep absolute mass before RGB/patch reduction, with exact-zero NA."""
    a=maps.detach().double();n=len(a)
    absolute=a.abs().flatten(1).sum(1);signed=a.flatten(1).sum(1)
    patch=a.abs().reshape(n,3,14,16,14,16).sum((1,3,5)).flatten(1)
    patch_signed=a.reshape(n,3,14,16,14,16).sum((1,3,5)).flatten(1)
    p=patch/absolute[:,None];logp=torch.where(p>0,p.log(),torch.zeros_like(p))
    entropy=-(p*logp).sum(1)/math.log(196)
    l2=a.flatten(1).norm(dim=1)
    sparse=(math.sqrt(a[0].numel())-absolute/l2)/(math.sqrt(a[0].numel())-1)
    return {k:v.cpu().numpy() for k,v in dict(patch_mass=patch,patch_signed=patch_signed,absolute_mass=absolute,signed_sum=signed,entropy=entropy,sparsity=sparse,cancellation=1-signed.abs()/absolute).items()}

def neuron_scan(model,image,targets,blocks=(0,3,7,11),chunk=8):
    captured={};handles=[];backward=0
    for b in blocks:
        for kind,module in [('post',model.blocks[b]),('mlp',model.blocks[b].mlp.act)]:
            def save(_m,_a,out,key=(b,kind)):captured[key]=out
            handles.append(module.register_forward_hook(save))
    clean=image.detach().clone().requires_grad_(True)
    packed={};parity=[]
    try:
        logits=model(clean)
        for b in blocks:
            for kind in ('post','mlp'):
                units=tuple((u['token'],u['channel']) for u in targets[str(b)][kind])
                tensor=captured[b,kind];maps=[]
                for start in range(0,len(units),chunk):
                    maps.append(endpoint_chunk(tensor,clean,units[start:start+chunk]).detach());backward+=1
                maps=torch.cat(maps)
                t,c=units[0];g=torch.autograd.grad(tensor[0,t,c],clean,retain_graph=True)[0];backward+=1
                expected=clean.detach()[0]*g[0]
                torch.testing.assert_close(maps[0],expected,atol=1e-5,rtol=1e-4)
                parity.append(float((maps[0]-expected).abs().max()))
                prefix=f'b{b:02d}_{kind}_'
                for name,v in statistics(maps).items():packed[prefix+name]=v
                packed[prefix+'activation']=np.asarray([float(tensor[0,t,c].detach()) for t,c in units])
                packed[prefix+'units']=np.asarray(units,dtype=np.int16)
        packed['logits']=logits.detach().cpu().numpy()[0]
        # Fixed, model-independent RGB descriptor for E1 visual matching.
        packed['rgb_descriptor']=torch.nn.functional.adaptive_avg_pool2d(image.detach(),(8,8)).cpu().numpy().reshape(-1)
        return packed,dict(forward_calls=1,backward_calls=backward,scalar_vjp_parity_max=max(parity))
    finally:
        for h in handles:h.remove()
