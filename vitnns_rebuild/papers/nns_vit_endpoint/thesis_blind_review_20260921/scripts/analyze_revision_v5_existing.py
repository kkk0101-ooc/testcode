#!/usr/bin/env python3
"""Recompute v5 manuscript baselines and uncertainty from sealed image-level data."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,gzip,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];R=P.parents[2];O=P/'evidence/revision_v5_existing';O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R/'src'))
from vitnns.endpoint_m1_statistics import metrics,random_metrics,bootstrap_median
SOURCES={};MANIFESTS={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def sealed(root,rel):
 if root not in MANIFESTS:
  t=read(root/'terminal_receipt.json');assert 'PASS' in t['status'];assert sha(root/'artifact_manifest.json')==t['artifact_manifest_sha256']
  MANIFESTS[root]={r['path']:r for r in read(root/'artifact_manifest.json')['files']}
 p=root/rel;assert sha(p)==MANIFESTS[root][rel]['sha256'];SOURCES[str(p.relative_to(R))]=sha(p);return p

def summary(values,key):
 a=np.asarray(values,float);n=len(a);a=a[np.isfinite(a)]
 if not len(a):return dict(n_total=n,n=0,median=None,ci95_low=None,ci95_high=None)
 ci=bootstrap_median(a,key);return dict(n_total=n,n=len(a),median=float(np.median(a)),ci95_low=float(ci[0]),ci95_high=float(ci[1]))
def dump(name,rows):
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def m1():
 root=R/'reports/nns_vit_endpoint_m1_r1_20260918';rows=[];summ=defaultdict(list);diff=defaultdict(list);classify=defaultdict(list)
 for f in sorted((root/'per_image').glob('*.json.gz')):
  d=json.loads(gzip.decompress(sealed(root,str(f.relative_to(root))).read_bytes()))
  for ref,comps in d['component_rows'].items():
   scores=defaultdict(list)
   for b in range(12):
    part=sorted([r for r in comps if r['operation']=='head' and r['block']==b],key=lambda r:r['component_id']);y=[r['absolute_effect'] for r in part]
    predictors={'endpoint':[r['m'] for r in part],'activation_gradient':[abs(r['gradient_activation']) for r in part],'taylor':[abs(r['p_ref']) for r in part]}
    for name,x in predictors.items():
     for metric,v in metrics(x,y).items():scores[name,metric].append(v)
    # Validate archived 1000-permutation means; recompute analytic expectation for random capture and regret.
    archived=next(r for r in d['comparisons'][ref] if r['family']=='head' and r['stratum']==f'block_{b:02d}')
    for metric,v in archived['metrics']['random'].items():scores['random',metric].append(v)
   archived=next(r for r in d['comparisons'][ref] if r['family']=='head' and r['stratum']=='within_block_mean')
   for (name,metric),values in scores.items():
    value=float(np.mean(values)) if all(v is not None for v in values) else np.nan
    expected=archived['metrics'][name][metric]
    assert (expected is None and not np.isfinite(value)) or abs(value-expected)<1e-12
    row=dict(model=d['model_id'],image=d['sample_id'],reference=ref,predictor=name,metric=metric,value=value);rows.append(row)
    summ[d['model_id'],ref,name,metric].append(value)
   for baseline in ['activation_gradient','taylor','random']:
    for metric in ('spearman','capture_at_1','regret_at_1'):
     a=archived['metrics']['endpoint'][metric];b=archived['metrics'][baseline][metric]
     diff[d['model_id'],ref,baseline,metric].append(np.nan if a is None or b is None else a-b)
  print('M1',len(rows)//56,flush=True) if len(rows)%11200==0 else None
 assert len(list((root/'per_image').glob('*.json.gz')))==800
 dump('m1_head_image_values.csv',rows)
 dump('m1_head_summary.csv',[dict(model=k[0],reference=k[1],predictor=k[2],metric=k[3],**summary(v,str(k))) for k,v in sorted(summ.items())])
 dump('m1_head_paired.csv',[dict(model=k[0],reference=k[1],baseline=k[2],metric=k[3],**summary(v,str(k))) for k,v in sorted(diff.items())])

def amplitude():
 root=R/'reports/nns_vit_endpoint_revision_v3_step2_r2_20260921';tasks=read(sealed(root,'task_manifest.json'))['tasks'];rows=[];allvalues=defaultdict(list)
 for t in tasks:
  if t['payload']['kind']!='primary':continue
  d=read(sealed(root,'results/attempts/'+t['task_id']+'/attempt-01.json'))['result'];p=R/d['amplitude_path'];assert sha(p)==d['amplitude_sha256'];SOURCES[str(p.relative_to(R))]=sha(p)
  with np.load(p) as z:
   if not rows:print('AMPLITUDE_KEYS',z.files,flush=True)
   for ref in np.unique(z['reference_id']):
    for alpha in np.unique(z['alpha']):
     mask=(z['reference_id']==ref)&(z['alpha']==alpha)
     effect=z['finite_effect'][mask];pred=z['taylor_prediction'][mask];absolute=np.abs(effect-pred);relative=np.divide(absolute,np.abs(effect),out=np.full_like(absolute,np.nan),where=effect!=0)
     for metric,values in [('absolute_error',absolute),('relative_error',relative)]:
      finite=values[np.isfinite(values)]
      for q in (.5,.9,.95,.99):
       rows.append(dict(model=d['model_id'],image=d['sample_id'],reference=str(ref),alpha=float(alpha),metric=metric,quantile=q,value=float(np.quantile(finite,q)) if len(finite) else np.nan))
     # Within-image effect quartiles preserve all small-effect observations.
     order=np.argsort(np.abs(effect),kind='stable')
     for binid,indices in enumerate(np.array_split(order,4)):
      for metric,values in [('absolute_error',absolute),('relative_error',relative)]:
       finite=values[indices];finite=finite[np.isfinite(finite)];allvalues[d['model_id'],str(ref),float(alpha),binid,metric].append(float(np.median(finite)) if len(finite) else np.nan)
 dump('amplitude_image_quantiles.csv',rows)
 sums=defaultdict(list)
 for r in rows:sums[r['model'],r['reference'],r['alpha'],r['metric'],r['quantile']].append(r['value'])
 dump('amplitude_quantile_summary.csv',[dict(model=k[0],reference=k[1],alpha=k[2],metric=k[3],quantile=k[4],**summary(v,str(k))) for k,v in sorted(sums.items())])
 dump('amplitude_effect_strata.csv',[dict(model=k[0],reference=k[1],alpha=k[2],effect_quartile=k[3],metric=k[4],**summary(v,str(k))) for k,v in sorted(allvalues.items())])

def main():
 m1();amplitude()
 (O/'source_receipt.json').write_text(json.dumps(dict(status='COMPLETE_EXISTING_DATA_REANALYSIS',sources=SOURCES,outputs={p.name:sha(p) for p in O.glob('*.csv')},script_sha256=sha(Path(__file__))),indent=2)+'\n')
if __name__=='__main__':main()
