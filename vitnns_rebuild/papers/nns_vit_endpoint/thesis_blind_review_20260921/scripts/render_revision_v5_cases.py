#!/usr/bin/env python3
"""CPU-only signed closure pages from frozen sloth-bear arrays."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];R=P.parents[2];PUB=R/'papers/nns_vit_endpoint/publication_r2_20260920';O=P/'figures/revision_v5';O.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=R/'reports/nns_vit_endpoint_e3_closure_r1_20260918/formal_results/attempts/deit_s16__014/attempt-01.json'
source=json.loads((PUB/'evidence/figure_sources.json').read_text());assert sha(f)==source[str(f.relative_to(R))]
d=json.loads(f.read_text())['result'];rgbp=PUB/'assets/deit_s16__014.npz';assert sha(rgbp)==source[str(rgbp.relative_to(R))]
with np.load(rgbp) as z:rgb=z['input_rgb']
raw={};sources={str(f.relative_to(R)):sha(f),str(rgbp.relative_to(R)):sha(rgbp)}
for a in d['artifacts']:
 q=R/a['path'];assert sha(q)==a['sha256'];sources[a['path']]=sha(q);v=np.load(q);raw[a['block']]={k:v[i].astype(np.float64) for i,k in enumerate(a['component_order'])}
# Descriptive case selection: fixed block 5 plus strongest QKV cancellation among the 12 frozen blocks.
ratios={b:1-np.abs(z['attention']).sum()/sum(np.abs(z[k]).sum() for k in ('query','key','value')) for b,z in raw.items()}
selected=[5,max(ratios,key=lambda b:(ratios[b],-b))]
plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
receipt=[]
for ci,b in enumerate(selected):
 z=raw[b];recon=z['identity']+z['query']+z['key']+z['value'];error=recon-z['L_F'];assert np.max(abs(error))<1e-5+1e-4*np.max(abs(z['L_F']))
 panels=[('Input',None),('Total signed',z['L_F']),('Identity',z['identity']),('Query',z['query']),('Key',z['key']),('Value',z['value']),('Signed reconstruction',recon),('Reconstruction error',error)]
 fig,axes=plt.subplots(2,4,figsize=(11,7.2));scale=max(np.max(abs(v.sum(0))) for _,v in panels[1:-1]);error_scale=max(np.max(abs(error.sum(0))),1e-15)
 for ax,(name,value) in zip(axes.flat,panels):
  if value is None:ax.imshow(rgb)
  else:
   limit=error_scale if name=='Reconstruction error' else scale
   shown=np.sign(value.sum(0))*np.log1p(99*abs(value.sum(0))/limit)/np.log(100)
   ax.imshow(shown,cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
  ax.set_title(name);ax.axis('off')
 fig.suptitle(f'DeiT-S/16 | sloth bear, target 297 | block {b} (zero-based)')
 fig.text(.5,.025,f'Signed RGB sums; log display kappa99, common raw scale ±{scale:.3g}. Error uses ±{error_scale:.3g}.  QKV cancellation={ratios[b]:.4f}.\nRaw RGB maximum closure error={np.max(abs(error)):.3g}; maps are not independently normalized.',ha='center',fontsize=9)
 fig.subplots_adjust(left=.025,right=.975,bottom=.14,top=.86,wspace=.08,hspace=.30)
 name=f'signed_case_{ci+1}'
 for suffix in ['png','pdf']:fig.savefig(O/f'{name}.{suffix}',dpi=180)
 plt.close(fig)
 # Store a compact, independently recheckable artifact with the original image and signed arrays.
 np.savez_compressed(O/f'{name}_arrays.npz',input_rgb=rgb,total=z['L_F'],identity=z['identity'],query=z['query'],key=z['key'],value=z['value'],reconstruction=recon,error=error)
 receipt.append(dict(case=ci+1,block=b,selection='fixed block5' if ci==0 else 'maximum QKV cancellation among12 blocks; descriptive',qkv_cancellation=ratios[b],closure_max=float(np.max(abs(error))),files={f'{name}.{ext}':sha(O/f'{name}.{ext}') for ext in ('png','pdf')},arrays_sha256=sha(O/f'{name}_arrays.npz')))
(O/'case_receipt.json').write_text(json.dumps(dict(status='RENDERED_FROM_SEALED_ARRAYS',sources=sources,cases=receipt),indent=2)+'\n');print(receipt)
