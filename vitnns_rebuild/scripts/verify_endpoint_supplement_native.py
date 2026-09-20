#!/usr/bin/env python3
"""Independent native graph recomputation for both real architectures."""
import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('CUBLAS_PREIMPORT')
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np,torch,timm
from timm.layers import set_fused_attn
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from run_endpoint_e4_intervention_worker import load_model,load_inputs
from vitnns.recovery_v2 import canonical_state_dict_tensor_digest
ROOT=REPO/'reports/nns_vit_endpoint_supplement_v1_20260920';torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
tasks=json.loads((ROOT/'task_manifest.json').read_text())['tasks'];device=torch.device('cuda:0');checks=[]
for model_id in ('vit_b16','deit_s16'):
 t=next(r['payload'] for r in tasks if r['payload']['model_id']==model_id and r['payload']['cohort_order']==0);model,binding=load_model(t,device);image,_,_=load_inputs(t,device);x=image.detach().clone().requires_grad_(True);captured={};handle=model.blocks[5].register_forward_hook(lambda _m,_a,o:captured.update(h=o))
 logits=model(x);order=torch.argsort(logits[0],descending=True,stable=True);top,runner=order[:2].tolist();gt=t['class_id'];grads=[]
 for c in (gt,top,runner):grads.append(torch.autograd.grad(logits[0,c],x,retain_graph=True)[0])
 stored=json.loads((ROOT/f'results/attempts/target__{model_id}__000/attempt-01.json').read_text())['result']
 for name,g in [('gt',grads[0]),('top1',grads[1]),('margin',grads[1]-grads[2])]:
  e=(x.detach()*g).detach().double().cpu().numpy();summary=stored['targets'][name]['full_endpoint'];values=dict(absolute_mass=float(abs(e).sum()),signed_sum=float(e.sum()),l2_norm=float(np.linalg.norm(e)))
  ratios={k:abs(v-summary[k])/(2e-5+2e-4*abs(summary[k])) for k,v in values.items()};assert max(ratios.values())<=1
  checks.append(dict(model=model_id,kind='native_target',target=name,max_tolerance_ratio=max(ratios.values())))
 feature=json.loads(gzip.decompress((ROOT/f'payloads/feature__{model_id}__000/feature.json.gz').read_bytes()));r=next(r for r in feature if r['block']==5);c=r['channel'];h=captured['h'][0,1:,c];score=.5*h.square().sum();g=torch.autograd.grad(score,x)[0];e=(x.detach()*g).abs().double();mass=e.reshape(3,14,16,14,16).sum((0,2,4)).flatten();p=(mass/mass.sum()).cpu().numpy();q=h.detach().double().square();q=(q/q.sum()).cpu().numpy();err=float(np.max(abs(p-r['p'])));assert err<2e-6 and np.max(abs(q-r['q']))<1e-8;checks.append(dict(model=model_id,kind='native_feature',block=5,channel=c,p_max_abs=err));handle.remove()
 del model,x,logits,grads
 seed=202609201;set_fused_attn(False);torch.manual_seed(seed);random=timm.create_model(t['timm_name'],pretrained=False).eval();record=json.loads((ROOT/f'results/attempts/random{seed}__{model_id}__000/attempt-01.json').read_text())['result'];assert canonical_state_dict_tensor_digest(random.state_dict())==record['binding']['tensor_digest'];random=random.to(device)
 for b in random.blocks:b.attn.fused_attn=False
 hmap={};handles=[random.blocks[b].register_forward_hook(lambda _m,_a,o,b=b:hmap.update({b:o})) for b in (0,11)];x=image.detach().clone().requires_grad_(True);random(x);npz=np.load(ROOT/f'payloads/random{seed}__{model_id}__000/m3.npz')
 for b in (0,11):
  for token in (0,t['tokens'][1]):
   c=t['channels'][str(b)][0];g=torch.autograd.grad(hmap[b][0,token,c],x,retain_graph=True)[0];mass=(x.detach()*g).abs().reshape(3,14,16,14,16).sum((0,2,4)).flatten().detach().cpu().numpy();i=np.flatnonzero((npz['block']==b)&(npz['token']==token)&(npz['channel']==c));assert len(i)==1;reference=npz['patch_absolute_mass_196'][i[0]];ratio=float(np.max(abs(mass-reference)/(2e-5+2e-4*abs(reference))));assert ratio<=1;checks.append(dict(model=model_id,kind='native_random_unit',block=b,token=token,channel=c,max_tolerance_ratio=ratio))
 for h in handles:h.remove()
 del random,x,hmap
result=dict(status='PASS',checks=checks,check_count=len(checks),purpose='Native unpatched real-model gradients independently reproduce margins, energy pairs and selected random-model units')
(ROOT/'native_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',checks=len(checks))))
