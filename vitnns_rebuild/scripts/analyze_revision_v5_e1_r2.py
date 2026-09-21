#!/usr/bin/env python3
"""Class-level analysis of the prospectively frozen WordNet-matched E1-R2 study."""
import csv, hashlib, json, math, os, sys
from collections import defaultdict
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.special import rel_entr
R=Path(__file__).resolve().parents[1];M=R/'manifests/nns_vit_revision_v5_e1_r2';ROOT=R/'reports/nns_vit_revision_v5_e1_r2_r1_20260921';OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'src'))
from vitnns.endpoint_m1_statistics import bootstrap_median
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(a):
 a=np.asarray(a,float);d=a.sum(-1,keepdims=True);return np.divide(a,d,out=np.full_like(a,np.nan),where=d!=0)
def js(p,q):
 m=(p+q)/2
 with np.errstate(invalid='ignore',divide='ignore'):return .5*np.sum(rel_entr(p,m)+rel_entr(q,m),axis=-1)/math.log(2)
def summary(values,key):
 a=np.asarray(values,float);a=a[np.isfinite(a)];ci=bootstrap_median(a,key);pos=int((a>1e-12).sum());neg=int((a< -1e-12).sum());n=pos+neg;p=1. if not n else min(1.,2*math.fsum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n)
 return dict(n=len(a),median=float(np.median(a)),ci95_low=ci[0],ci95_high=ci[1],p=p,positive=pos,negative=neg,ties=len(a)-n)
def dump(path,values):
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
cohort=json.loads((M/'cohorts.json').read_text());matching={r['wnid']:r for r in json.loads((M/'matching.json').read_text())['rows']};byclass=defaultdict(list);byimage={}
for r in cohort['rows']:byclass[r['wnid']].append(r['sample_id']);byimage[r['sample_id']]=r
sources={};image_rows=[];class_rows=[];activation_rows=[]
for model in ('deit_s16','vit_b16'):
 data={}
 for sample,row in byimage.items():
  stem=model+'__'+sample.split('.')[0];receipt=ROOT/'e1'/(stem+'.json');r=json.loads(receipt.read_text());payload=R/r['output'];assert r['status']=='VALID' and sha(payload)==r['output_sha256'];sources[str(receipt.relative_to(R))]=sha(receipt);sources[str(payload.relative_to(R))]=sha(payload)
  with np.load(payload,allow_pickle=False) as z:
   data[sample]={}
   for b in (0,3,7,11):
    for kind in ('post','mlp'):
     for suffix in ('patch_mass','activation','units'):
      field=f'b{b:02d}_{kind}_{suffix}';data[sample][field]=z[field]
 for b in (0,3,7,11):
  for kind in ('post','mlp'):
   field=f'b{b:02d}_{kind}_patch_mass';maps={s:norm(v[field]) for s,v in data.items()}
   activations=np.asarray([data[s][f'b{b:02d}_{kind}_activation'] for s in data]);class_ids=np.asarray([byimage[s]['class_id'] for s in data]);overall=activations.mean(0);den=((activations-overall)**2).sum(0);between=sum(5*(activations[class_ids==c].mean(0)-overall)**2 for c in set(class_ids));eta=np.divide(between,den,out=np.full(activations.shape[1],np.nan),where=den!=0)
   first=next(iter(data.values()))[f'b{b:02d}_{kind}_units']
   for unit in range(activations.shape[1]):activation_rows.append(dict(model=model,block=b,kind=kind,unit=unit,token=int(first[unit,0]),channel=int(first[unit,1]),activation_mean=float(overall[unit]),activation_sd=float(activations[:,unit].std()),activation_eta_squared=float(eta[unit])))
   for wnid,current in byclass.items():
    matched=byclass[matching[wnid]['matched_wnid']];values=[]
    for image in current:
     same=[x for x in current if x!=image];same_sim=float(np.nanmean([np.nanmean(1-js(maps[image],maps[x])) for x in same]));matched_sim=float(np.nanmean([np.nanmean(1-js(maps[image],maps[x])) for x in matched]));value=same_sim-matched_sim;values.append(value);image_rows.append(dict(model=model,image=image,wnid=wnid,matched_wnid=matching[wnid]['matched_wnid'],wordnet_distance=matching[wnid]['wordnet_distance'],block=b,kind=kind,same_similarity=same_sim,matched_similarity=matched_sim,same_minus_matched=value))
    class_rows.append(dict(model=model,wnid=wnid,matched_wnid=matching[wnid]['matched_wnid'],wordnet_distance=matching[wnid]['wordnet_distance'],block=b,kind=kind,value=float(np.mean(values))))
summaries=[]
for model in ('deit_s16','vit_b16'):
 family=[]
 for b in (0,3,7,11):
  for kind in ('post','mlp'):
   vals=[r['value'] for r in class_rows if r['model']==model and r['block']==b and r['kind']==kind];family.append(dict(model=model,block=b,kind=kind,**summary(vals,f'E1_R2|{model}|{b}|{kind}')))
 order=sorted(family,key=lambda r:r['p']);prev=0
 for i,r in enumerate(order):prev=max(prev,min(1.,(len(order)-i)*r['p']));r['holm_p']=prev;r['decision']='SUPPORTED_DIRECTION' if prev<=.05 and r['median']>0 else 'OPPOSITE_DIRECTION' if prev<=.05 and r['median']<0 else 'NOT_SUPPORTED'
 summaries.extend(family)
dump(OUT/'image_values.csv',image_rows);dump(OUT/'class_values.csv',class_rows);dump(OUT/'summary.csv',summaries);dump(OUT/'activation_selectivity.csv',activation_rows)
receipt=dict(status='ANALYZED_PROSPECTIVE_E1_R2',protocol_sha256=sha(M/'protocol.json'),matching_sha256=sha(M/'matching.json'),task_count=len(sources)//2,sources=sources,outputs={p.name:sha(p) for p in OUT.glob('*.csv')})
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'task_count':receipt['task_count'],'summary':summaries},indent=2))
