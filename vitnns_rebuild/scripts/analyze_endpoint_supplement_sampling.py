#!/usr/bin/env python3
"""M3 sampling sensitivity on the frozen 32-image supplement cohort; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/'reports/nns_vit_endpoint_supplement_v1_20260920'
from analyze_endpoint_supplement_v1 import estimate,write,rng_for

def metric_cube(d):
 mass=d['patch_absolute_mass_196'].astype(float);p=mass/mass.sum(1)[:,None];tokens=d['token'];coords=np.indices((14,14)).reshape(2,196).T;tc=np.stack(((tokens-1)//14,(tokens-1)%14),1);dist=np.abs(tc[:,None]-coords[None]);local=(p*(dist.max(2)<=1)).sum(1);distance=(p*np.sqrt((dist**2).sum(2))/(13*np.sqrt(2))).sum(1)
 local[tokens==0]=np.nan;distance[tokens==0]=np.nan
 return np.stack([d['normalized_spatial_entropy'],local,distance,d['cancellation']],1).reshape(12,16,32,4)
def trends(cube,blocks=None,token_indices=None,channel_indices=None):
 x=cube if blocks is None else cube[:,blocks]
 if token_indices is not None:x=x[:,:,token_indices]
 if channel_indices is not None:x=x[:,:,:,channel_indices]
 patch=np.nanmedian(x[:,:,1:].reshape(x.shape[0],x.shape[1],-1,4),axis=2)
 cls=np.nanmedian(x[:,:,0,:,0],axis=2);gap=cls-patch[:,:,0]
 bs=list(range(12)) if blocks is None else blocks;early=[i for i,b in enumerate(bs) if b<4];late=[i for i,b in enumerate(bs) if b>=8]
 values=np.nanmedian(patch[:,late],axis=1)-np.nanmedian(patch[:,early],axis=1)
 return np.concatenate([values,np.nanmedian(gap,axis=1)[:,None]],axis=1)
def main():
 inherited=json.loads((ROOT/'inherited.json').read_text());manifest=json.loads((REPO/'reports/nns_vit_endpoint_m3_r1_20260919/artifact_manifest.json').read_text());expected={r['path']:r['sha256'] for r in manifest['files']};results=[]
 for model in ('vit_b16','deit_s16'):
  records=[r for r in inherited if r['payload']['model_id']==model and r['payload']['cohort_order']<32];cubes=[]
  for r in records:
   base=REPO/'reports/nns_vit_endpoint_m3_r1_20260919';relative=f"task_payloads/{r['task_id']}/metrics.npz";p=base/relative;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[relative]
   d=np.load(p);assert len(d['block'])==6144;cubes.append(metric_cube(d))
  cube=np.stack(cubes);reference=trends(cube);rng=rng_for(model,'sampling');channels=[];tokens=[]
  for _ in range(256):
   channels.append(trends(cube,channel_indices=rng.integers(32,size=32)).mean(0))
   tokens.append(trends(cube,token_indices=np.r_[0,rng.integers(1,16,size=15)]).mean(0))
  loo_blocks=[trends(cube,blocks=[b for b in range(12) if b!=drop]).mean(0) for drop in range(12)]
  loo_strata=[trends(cube,token_indices=[t for t in range(16) if t!=drop]).mean(0) for drop in range(1,16)]
  for i,metric in enumerate(('entropy','locality','distance','cancellation','cls_minus_patch_entropy')):
   results.append(dict(model=model,metric=metric,n_images=32,channels=32,patch_strata=15,reference=estimate(reference[:,i],model+metric),channel_bootstrap_mean_range=np.quantile(np.array(channels)[:,i],[.025,.975]).tolist(),token_stratum_bootstrap_mean_range=np.quantile(np.array(tokens)[:,i],[.025,.975]).tolist(),leave_one_block_out_range=[float(np.min(np.array(loo_blocks)[:,i])),float(np.max(np.array(loo_blocks)[:,i]))],leave_one_stratum_out_range=[float(np.min(np.array(loo_strata)[:,i])),float(np.max(np.array(loo_strata)[:,i]))]))
 write('sampling_analysis.json',dict(scope='Existing M3 data on the frozen 32-image supplement cohort; all original 32 channels and 15 patch strata; resampling is not independent channel-seed replication',unit_bootstrap_repetitions=256,image_bootstrap_repetitions=2000,results=results))
 print(json.dumps(dict(status='COMPLETE',metrics=len(results))))
if __name__=='__main__':main()
