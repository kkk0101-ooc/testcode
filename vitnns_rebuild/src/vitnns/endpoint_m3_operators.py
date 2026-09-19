"""Analytic directional derivatives for the M3 ViT operator audit."""
from __future__ import annotations
import math
import torch
from torch import Tensor


def layer_norm(x: Tensor, dx: Tensor, weight: Tensor, bias: Tensor, eps: float=1e-6):
    mean=x.mean(-1,keepdim=True); centered=x-mean; var=centered.square().mean(-1,keepdim=True); inv=(var+eps).rsqrt()
    y=centered*inv*weight+bias
    dmean=dx.mean(-1,keepdim=True); dvar=2*(centered*dx).mean(-1,keepdim=True)
    dy=((dx-dmean)*inv-centered*dvar*inv.pow(3)/2)*weight
    return y,dy

def gelu(x: Tensor, dx: Tensor):
    y=torch.nn.functional.gelu(x,approximate="none")
    derivative=.5*(1+torch.erf(x/math.sqrt(2)))+x*torch.exp(-x.square()/2)/math.sqrt(2*math.pi)
    return y,derivative*dx

def softmax(x: Tensor, dx: Tensor):
    y=x.softmax(-1); return y,y*(dx-(y*dx).sum(-1,keepdim=True))

def qk_scaled(inputs, directions):
    q,k=inputs; dq,dk=directions; scale=1/math.sqrt(q.shape[-1])
    return q@k.transpose(-2,-1)*scale,(dq@k.transpose(-2,-1)+q@dk.transpose(-2,-1))*scale

def av(inputs, directions):
    a,v=inputs; da,dv=directions; return a@v,da@v+a@dv

def head_concat(x: Tensor, dx: Tensor):
    # [heads,tokens,head_dim] -> [tokens,heads*head_dim]
    return x.permute(1,0,2).reshape(x.shape[1],-1),dx.permute(1,0,2).reshape(x.shape[1],-1)

def affine(x: Tensor, dx: Tensor, weight: Tensor, bias: Tensor):
    return x@weight.T+bias,dx@weight.T

def residual(inputs, directions):
    a,b=inputs; da,db=directions; return a+b,da+db

def mlp(x: Tensor, dx: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor):
    hidden=x@w1.T+b1; dhidden=dx@w1.T; activated,dactivated=gelu(hidden,dhidden)
    return activated@w2.T+b2,dactivated@w2.T

def cases(dtype=torch.float64):
    """Return operator name, primal tuple, differentiable function and analytic JVP."""
    g=torch.Generator().manual_seed(77123)
    rand=lambda *shape:torch.randn(*shape,generator=g,dtype=dtype)
    wln,bln=rand(7),rand(7); wout,bout=rand(5,12),rand(5); w1,b1,w2,b2=rand(11,7),rand(11),rand(7,11),rand(7)
    entries=[]
    def single(name,x,function,analytic): entries.append((name,(x,),lambda value:function(value),lambda values,directions:analytic(values[0],directions[0])))
    single("layer_norm",rand(4,7),lambda x:torch.nn.functional.layer_norm(x,(7,),wln,bln,1e-6),lambda x,dx:layer_norm(x,dx,wln,bln)[1])
    single("gelu",rand(3,7),lambda x:torch.nn.functional.gelu(x,approximate="none"),lambda x,dx:gelu(x,dx)[1])
    single("softmax",rand(2,3,7),lambda x:x.softmax(-1),lambda x,dx:softmax(x,dx)[1])
    q,k=rand(2,5,4),rand(2,6,4); entries.append(("qk_scaled",(q,k),lambda q,k:q@k.transpose(-2,-1)/math.sqrt(4),lambda z,d:qk_scaled(z,d)[1]))
    a=rand(2,5,6).softmax(-1); v=rand(2,6,4); entries.append(("av",(a,v),lambda a,v:a@v,lambda z,d:av(z,d)[1]))
    single("head_concat",rand(3,5,4),lambda x:x.permute(1,0,2).reshape(5,12),lambda x,dx:head_concat(x,dx)[1])
    single("output_projection",rand(5,12),lambda x:x@wout.T+bout,lambda x,dx:affine(x,dx,wout,bout)[1])
    x,y=rand(4,7),rand(4,7); entries.append(("residual_add",(x,y),lambda x,y:x+y,lambda z,d:residual(z,d)[1]))
    single("mlp",rand(4,7),lambda x:torch.nn.functional.gelu(x@w1.T+b1,approximate="none")@w2.T+b2,lambda x,dx:mlp(x,dx,w1,b1,w2,b2)[1])
    single("final_layer_norm",rand(4,7),lambda x:torch.nn.functional.layer_norm(x,(7,),wln,bln,1e-6),lambda x,dx:layer_norm(x,dx,wln,bln)[1])
    single("classifier",rand(4,12),lambda x:x@wout.T+bout,lambda x,dx:affine(x,dx,wout,bout)[1])
    return entries
