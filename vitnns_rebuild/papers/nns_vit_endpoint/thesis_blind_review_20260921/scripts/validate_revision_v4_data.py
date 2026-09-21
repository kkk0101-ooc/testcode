#!/usr/bin/env python3
"""Independent CPU audit of every depth estimate, interval, and test family."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,binomtest
P=Path(__file__).resolve().parents[1];R=P.parents[2];D=P/'evidence/revision_v4'
checks=[]
def check(name,ok,detail=''):
 checks.append(dict(name=name,passed=bool(ok),detail=detail))
 if not ok:raise AssertionError((name,detail))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((D/n).open()))
def key(r):return r['model'],r['group'],r['metric']
def numeric(s):return float(s) if s not in ('',None) else np.nan
def med(a):
 a=np.asarray(a,dtype=float);a=a[np.isfinite(a)];return float(np.median(a)) if a.size else np.nan

def main():
 receipt=json.loads((D/'analysis_receipt.json').read_text())
 for rel,h in receipt['source_sha256'].items():check('source:'+rel,sha(R/rel)==h)
 for rel,h in receipt['output_sha256'].items():check('output:'+rel,sha(D/rel)==h)
 check('producer_bound',sha(P/'scripts/analyze_revision_v4_depth.py')==receipt['script_sha256'])
 check('plan_bound',sha(P/'notes/revision_v4_plan.md')==receipt['plan_sha256'])
 arrays=defaultdict(list);ids=defaultdict(list)
 for r in rows('image_profiles.csv'):
  arrays[key(r)].append([numeric(r[f'block_{b}']) for b in range(12)]);ids[key(r)].append(r['sample_id'])
 arrays={k:np.array(v) for k,v in arrays.items()}
 check('all_98_families',len(arrays)==98)
 check('all_30200_profiles',sum(len(a) for a in arrays.values())==30200)
 check('all_profiles_complete',all(np.isfinite(a).all() for a in arrays.values()))
 for k,a in arrays.items():check('unique_expected_images:'+str(k),len(set(ids[k]))==len(a)==(100 if k[1].startswith('sample_') else 400))
 # Independent formula-level recomputation; supplied seeds are also checked against the declared keys.
 def stats(r,a,suffix):
  prefix='|'.join(key(r));seed=int.from_bytes(hashlib.sha256(('ENDPOINT_V4_DEPTH_20260921|'+prefix+'|'+suffix).encode()).digest()[:8],'big')
  check('seed:'+prefix+'|'+suffix,int(r['seed'])==seed)
  x=np.sort(np.asarray(a)[np.isfinite(a)]);n=len(x)
  check('counts:'+prefix+'|'+suffix,int(r['n_total'])==len(a) and int(r['n_valid'])==n and int(r['n_missing'])==len(a)-n)
  rng=np.random.Generator(np.random.PCG64(seed))
  if n%2:
   u=rng.beta((n+1)/2,(n+1)/2,10000);draw=x[np.floor(n*u).astype(int)]
  else:
   u=rng.beta(n/2,n/2+1,10000);v=u+(1-u)*rng.beta(1,n/2,10000);draw=(x[np.floor(n*u).astype(int)]+x[np.floor(n*v).astype(int)])/2
  values=[med(x),*np.quantile(draw,[.025,.975],method='linear')]
  observed=[float(r[f]) for f in ['median','ci95_low','ci95_high']]
  check('estimates:'+prefix+'|'+suffix,np.allclose(values,observed,atol=2e-13,rtol=2e-13),str((values,observed)) if not np.allclose(values,observed,atol=2e-13,rtol=2e-13) else '')
 for r in rows('block_summary.csv'):stats(r,arrays[key(r)][:,int(r['block'])],'block_'+r['block'])
 trends={k:np.array([spearmanr(range(12),a).statistic if np.ptp(a)>0 else np.nan for a in ar]) for k,ar in arrays.items()}
 for r in rows('trend_summary.csv'):
  k=key(r);stats(r,trends[k],'spearman');a=arrays[k]
  check('monotone_counts:'+str(k),int(r['nondecreasing_images'])==np.all(np.diff(a,axis=1)>=0,axis=1).sum() and int(r['nonincreasing_images'])==np.all(np.diff(a,axis=1)<=0,axis=1).sum())
 for r in rows('phase_endpoint_contrasts.csv'):
  a=arrays[key(r)];p={n:np.median(a[:,bs],axis=1) for n,bs in [('early',slice(0,4)),('middle',slice(4,8)),('late',slice(8,12))]};c=r['contrast']
  if c=='block11_minus_block0':v=a[:,11]-a[:,0];suffix='endpoint'
  elif '_minus_' in c:u,w=c.split('_minus_');v=p[u]-p[w];suffix=c
  else:v=p[c];suffix=c
  stats(r,v,suffix)
 adj=defaultdict(list)
 for r in rows('adjacent_differences.csv'):
  a=arrays[key(r)];b=int(r['block']);d=a[:,b+1]-a[:,b];stats(r,d,'adjacent_'+str(b))
  pos=int((d>1e-12).sum());neg=int((d< -1e-12).sum());p=binomtest(pos,pos+neg,.5).pvalue if pos+neg else 1.
  check('exact_sign:'+str(key(r))+str(b),math.isclose(p,float(r['p_two_sided']),abs_tol=2e-14,rel_tol=2e-12) and pos==int(r['positive']) and neg==int(r['negative']))
  adj[key(r)].append(r)
 for k,a in adj.items():
  check('11_adjacent:'+str(k),sorted(int(r['block']) for r in a)==list(range(11)))
  prev=0
  for i,r in enumerate(sorted(a,key=lambda r:(float(r['p_two_sided']),int(r['block'])))):
   prev=max(prev,min(1.,(11-i)*float(r['p_two_sided'])))
   check('holm:'+str(k)+r['block'],math.isclose(prev,float(r['holm_p']),abs_tol=1e-15) and (r['holm_reject_005']=='True')==(prev<=.05))
 # Producer input parity for all source CSV rows plus original per-unit NPZ blocks.
 indexed={(k,s):a for k,ar in arrays.items() for s,a in zip(ids[k],ar)}
 qroot=R/'reports/nns_vit_endpoint_revision_v3_step2_r2_20260921'
 qmetrics=['absolute_mass','normalized_spatial_entropy','normalized_pairwise_distance','cancellation','relative_scale']
 with (qroot/'qkv_image_values.csv').open() as f:
  for r in csv.DictReader(f):
   for m in qmetrics:
    k=(r['model_id'],'qkv_'+r['path'],m);check('qkv:'+str(k)+r['sample_id']+r['block'],indexed[k,r['sample_id']][int(r['block'])]==float(r[m]))
 manifest=json.loads((qroot/'task_manifest.json').read_text())
 for t in manifest['tasks']:
  if t['payload']['kind']!='sampling':continue
  r=json.loads((qroot/'results/attempts'/t['task_id']/'attempt-01.json').read_text())['result']
  with np.load(R/r['sampling_path'],allow_pickle=False) as data:
   for gi in range(5):
    for b in range(12):
     mask=(data['group_index']==gi)&(data['block']==b)
     for m in ['normalized_spatial_entropy','own_token_locality_above_uniform','input_dependency_distance']:
      a=data[m][mask];k=(r['model_id'],'sample_'+str(gi+1),m)
      check('unit_aggregation:'+t['task_id']+str((gi,b,m)),indexed[k,r['sample_id']][b]==med(a))
 # M4 image values and descriptive masses have different aggregation origins, audited independently.
 root=R/'reports/nns_vit_endpoint_m4_r1_20260919'
 for fn in ['image_values.csv','m4_a1_image_values.csv']:
  with (root/fn).open() as f:
   for r in csv.DictReader(f):
    k=(r['model_id'],r['domain']+'_'+r['group'],r['metric'])
    if k not in arrays:continue
    check('m4_input:'+fn+str(k)+r['sample_id']+r['block'],indexed[k,r['sample_id']][int(r['block'])]==float(r['value']))
 # Independent analytic counterexamples and undefined-value behavior.
 rho=spearmanr(range(12),[0]*12).statistic
 check('constant_trend_undefined',not np.isfinite(rho))
 nonmono=np.array([0,3,1,4,2,5,3,6,4,7,5,8]);check('endpoint_not_monotonic',nonmono[-1]>nonmono[0] and (np.diff(nonmono)<0).any())
 # Avoid hundreds of thousands of redundant detail rows: retain counts and every failure.
 out=dict(status='PASS',checks=len(checks),families=len(arrays),all_estimates_intervals_seeds_recomputed=True,all_adjacent_tests_recomputed=True,all_source_input_rows_checked=True,all_sampling_unit_blocks_recomputed=True,errors=[],sources=receipt['source_sha256'],outputs={p.name:sha(p) for p in sorted(D.glob('*.csv'))},validator_sha256=sha(Path(__file__)))
 (P/'qc/revision_v4_data_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('sources','outputs')}))
if __name__=='__main__':main()
