#!/usr/bin/env python3
"""Compare independent analytic component maps with the actual scanner on tiny models."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from timm.models.vision_transformer import Block
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from vitnns.revision_v6_analytic import reference,fanout_fixture,validate_use_maps
from vitnns.endpoint_vit import decompose_endpoint_vit,EndpointTarget
class Tiny(torch.nn.Module):
 def __init__(self,pre_ln=True,mlp=True,heads=2):
  super().__init__();self.blocks=torch.nn.ModuleList([Block(4,heads,mlp_ratio=2,qkv_bias=True).double()]);self.blocks[0].attn.fused_attn=False
  if not pre_ln:self.blocks[0].norm1=torch.nn.Identity()
  with torch.no_grad():
   for i,(name,p) in enumerate(self.blocks[0].named_parameters()):
    v=torch.arange(p.numel(),dtype=torch.float64).reshape(p.shape)
    p.copy_((1+.12*torch.sin(v+i)) if 'norm' in name and name.endswith('weight') else .23*torch.sin(v*.71+i+.3)+.04)
   if not mlp:self.blocks[0].mlp.fc2.weight.zero_();self.blocks[0].mlp.fc2.bias.zero_()
  self.register_buffer('readout',torch.tensor([[.8,-.3,.6,.1],[-.2,.9,.4,-.7],[.5,.2,-.8,1.1]],dtype=torch.float64))
  self.eval()
 def forward(self,x):
  z=self.blocks[0](x.reshape(1,3,4));s=(z*self.readout).sum((1,2));return torch.stack([s,-s],dim=1)
def run():
 torch.set_num_threads(1);rows=[];arrays={};directions=np.random.default_rng(2026092301).normal(size=(3,3,4));directions/=np.linalg.norm(directions,axis=(1,2))[:,None,None]
 x=np.array([[.31,-.73,1.17,.62],[-1.12,.43,.86,-.28],[.54,1.23,-.47,.91]],dtype=np.float64)
 for label,pre,mlp,heads in [('attention_single',False,False,1),('attention_double',False,False,2),('preln_residual',True,True,2)]:
  model=Tiny(pre,mlp,heads);params={k:v.detach().numpy() for k,v in model.blocks[0].named_parameters()};params['readout']=model.readout.numpy()
  expected,score=reference(x,params,heads,pre,mlp)
  result=decompose_endpoint_vit(model,torch.tensor(x).reshape(1,1,3,4),block=0,target=EndpointTarget('s','',kind='logit',class_id=0),atol=1e-11,rtol=1e-9)
  observed={k:v.numpy().ravel() for k,v in result.components.items()};observed['full']=result.full.numpy().ravel();observed['branch_reconstruction']=sum(observed[k] for k in ['query','key','value']);observed['with_bypass']=observed['branch_reconstruction']+observed['identity']
  for key,value in expected.items():
   err=observed[key]-value;absolute=float(abs(err).max());relative=float(np.linalg.norm(err)/np.linalg.norm(value));ok=bool(np.allclose(observed[key],value,atol=1e-11,rtol=1e-9));assert ok,(label,key,absolute,relative)
   rows.append(dict(fixture=label,component=key,max_abs_error=absolute,relative_l2=relative,atol=1e-11,rtol=1e-9,passed=ok));arrays[label+'__'+key+'__analytic']=value;arrays[label+'__'+key+'__scanner']=observed[key]
  for i,d in enumerate(directions):
   eps=1e-5;plus=reference(x+eps*d,params,heads,pre,mlp)[1];minus=reference(x-eps*d,params,heads,pre,mlp)[1];fd=(plus-minus)/(2*eps);analytic=float(np.sum(expected['full'].reshape(x.shape)/x*d));error=abs(fd-analytic);assert error<=1e-8+1e-6*abs(analytic)
   rows.append(dict(fixture=label,component='direction_'+str(i),max_abs_error=error,relative_l2=error/max(abs(analytic),1e-30),atol=1e-8,rtol=1e-6,passed=True))
 mutations,toy=fanout_fixture()
 # Swap Q/K labels: sum closure remains intact, component verification must fail.
 expected={('qkv','attention',i,0):arrays['preln_residual__'+key+'__analytic'] for i,key in enumerate(['query','key','value'])}
 correct=[dict(use=list(k),map=v) for k,v in expected.items()];validate_use_maps(correct,expected)
 wrong=[dict(use=r['use'],map=correct[[1,0,2][i]]['map']) for i,r in enumerate(correct)]
 try:validate_use_maps(wrong,expected)
 except ValueError as e:mutations.append(dict(mutation='attention_wrong_qk_slot',rejected=True,reason=str(e),correct_pass=True))
 else:raise AssertionError('ATTENTION_WRONG_SLOT_FALSE_PASS')
 return rows,arrays,mutations,toy
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();assert (a.root/'protocol.json').is_file() and not (a.root/'analytic_reference_results.json').exists()
 start=time.monotonic();rows,arrays,mutations,toy=run();np.savez_compressed(a.root/'analytic_reference_arrays.npz',**arrays)
 (a.root/'analytic_reference_results.json').write_text(json.dumps(dict(schema_version=1,status='PASS',cpu_only=True,seconds=time.monotonic()-start,rows=rows,array_sha256=hashlib.sha256((a.root/'analytic_reference_arrays.npz').read_bytes()).hexdigest()),indent=2)+chr(10))
 (a.root/'mutation_results.json').write_text(json.dumps(dict(schema_version=1,status='PASS',rows=mutations,fanout_maps=toy),indent=2)+chr(10));print('analytic',len(rows),'mutations',len(mutations))
