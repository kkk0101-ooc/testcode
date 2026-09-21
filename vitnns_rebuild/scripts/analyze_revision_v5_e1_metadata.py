#!/usr/bin/env python3
"""Outcome-blind ImageNet-hierarchy matching sensitivity for E1."""
import io, tarfile, json, csv, hashlib, math, sys, os
from pathlib import Path
from collections import defaultdict, deque
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import scipy.io
from scipy.special import rel_entr

R=Path(__file__).resolve().parents[1]
RUN=R/'reports/nns_vit_revision_v5_r1_20260921'
M=R/'manifests/nns_vit_revision_v5'
O=RUN/'analysis'
sys.path.insert(0,str(R/'src'))
from vitnns.endpoint_m1_statistics import bootstrap_median

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(*x):return hashlib.sha256(('NNS_VIT_REVISION_V5_E1_METADATA_SENSITIVITY|20260921|'+'|'.join(map(str,x))).encode()).hexdigest()
def normalized(a):
 a=np.asarray(a,float);d=a.sum(-1,keepdims=True);return np.divide(a,d,out=np.full_like(a,np.nan),where=d!=0)
def js(p,q):
 m=(p+q)/2
 with np.errstate(invalid='ignore',divide='ignore'):return .5*np.sum(rel_entr(p,m)+rel_entr(q,m),axis=-1)/math.log(2)
def summarize(a,k):
 a=np.asarray(a,float);a=a[np.isfinite(a)];ci=bootstrap_median(a,k);pos=int((a>1e-12).sum());neg=int((a< -1e-12).sum());n=pos+neg;p=1. if not n else min(1.,2*math.fsum(math.comb(n,j) for j in range(min(pos,neg)+1))/2**n)
 return dict(n=len(a),median=float(np.median(a)),ci95_low=ci[0],ci95_high=ci[1],p=p,positive=pos,negative=neg,ties=len(a)-n)
def dump(path,rows):
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

cohort=json.loads((M/'cohorts.json').read_text());e1=[r for r in cohort['rows'] if r['cohort']=='e1'];wnids=sorted({r['wnid'] for r in e1})
tar=R/'assets/imagenet/ILSVRC2012_devkit_t12.tar.gz'
with tarfile.open(tar) as archive:
 data=archive.extractfile('ILSVRC2012_devkit_t12/data/meta.mat').read()
nodes=scipy.io.loadmat(io.BytesIO(data),squeeze_me=True,struct_as_record=False)['synsets'];byid={int(x.ILSVRC2012_ID):x for x in nodes};bywnid={str(x.WNID):int(x.ILSVRC2012_ID) for x in nodes};graph=defaultdict(set)
for node in nodes:
 for child in np.atleast_1d(node.children):graph[int(node.ILSVRC2012_ID)].add(int(child));graph[int(child)].add(int(node.ILSVRC2012_ID))
def distance(a,b):
 q=deque([(bywnid[a],0)]);seen={bywnid[a]};target=bywnid[b]
 while q:
  n,d=q.popleft()
  if n==target:return d
  for v in graph[n]:
   if v not in seen:seen.add(v);q.append((v,d+1))
 raise ValueError('DISCONNECTED_WORDNET')
dist={(a,b):distance(a,b) for a in wnids for b in wnids if a!=b}
match={a:min((b for b in wnids if b!=a),key=lambda b:(dist[a,b],key('class_tie',a,b))) for a in wnids}
byclass=defaultdict(list)
for r in e1:byclass[r['wnid']].append(r['sample_id'])
result_rows=[];class_rows=[]
for model in ('deit_s16','vit_b16'):
 payload={}
 for r in e1:
  p=RUN/'e1'/(model+'__'+r['sample_id'].split('.')[0]+'.npz')
  with np.load(p,allow_pickle=False) as z:payload[r['sample_id']]={f'b{b:02d}_{kind}_patch_mass':z[f'b{b:02d}_{kind}_patch_mass'] for b in (0,3,7,11) for kind in ('post','mlp')}
 for b in (0,3,7,11):
  for kind in ('post','mlp'):
   field=f'b{b:02d}_{kind}_patch_mass';norm={i:normalized(payload[i][field]) for i in payload}
   for wnid in wnids:
    target=byclass[wnid];other=byclass[match[wnid]]
    values=[]
    for image in target:
     same=[j for j in target if j!=image]
     same_similarity=float(np.nanmean([np.nanmean(1-js(norm[image],norm[j])) for j in same]))
     matched_similarity=float(np.nanmean([np.nanmean(1-js(norm[image],norm[j])) for j in other]))
     values.append(same_similarity-matched_similarity)
     result_rows.append(dict(model=model,image=image,wnid=wnid,matched_wnid=match[wnid],wordnet_distance=dist[wnid,match[wnid]],block=b,kind=kind,same_minus_metadata_matched=values[-1]))
    class_rows.append(dict(model=model,wnid=wnid,matched_wnid=match[wnid],wordnet_distance=dist[wnid,match[wnid]],block=b,kind=kind,value=float(np.mean(values))))
summary=[]
for model in ('deit_s16','vit_b16'):
 family=[]
 for b in (0,3,7,11):
  for kind in ('post','mlp'):
   vals=[r['value'] for r in class_rows if r['model']==model and r['block']==b and r['kind']==kind];row=dict(model=model,block=b,kind=kind,**summarize(vals,f'{model}|{b}|{kind}'));family.append(row)
 order=sorted(family,key=lambda r:r['p']);prev=0
 for i,r in enumerate(order):prev=max(prev,min(1.,(len(order)-i)*r['p']));r['holm_p']=prev;r['decision']='SUPPORTED_DIRECTION' if prev<=.05 and r['median']>0 else 'OPPOSITE_DIRECTION' if prev<=.05 and r['median']<0 else 'NOT_SUPPORTED'
 summary.extend(family)
dump(O/'e1_metadata_image_values.csv',result_rows);dump(O/'e1_metadata_class_values.csv',class_rows);dump(O/'e1_metadata_summary.csv',summary)
receipt=dict(status='POST_HOC_OUTCOME_BLIND_METADATA_SENSITIVITY',matching='shortest undirected distance in ILSVRC2012 devkit WordNet hierarchy; SHA256 tie break; average five matched-class images',meta_tar_sha256=sha(tar),selected_classes=len(wnids),sources={'cohorts.json':sha(M/'cohorts.json')},outputs={p.name:sha(p) for p in O.glob('e1_metadata_*.csv')})
(O/'e1_metadata_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'summary':summary},indent=2))
