#!/usr/bin/env python3
"""Reconstruct fair random AUCs directly from sealed isolated scores, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/'reports/nns_vit_endpoint_supplement_v1_20260920'

def write(name,d):(ROOT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(name,rows):
 with (ROOT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def auc(points):return math.fsum((x1-x0)*(y1+y0)/2 for (x0,y0),(x1,y1) in zip(points,points[1:]))/.25
def estimate(v,rng):
 v=np.asarray(v,dtype=float);v=v[np.isfinite(v)];draw=v[rng.integers(len(v),size=(2000,len(v)))];means=draw.mean(1);medians=np.median(draw,axis=1)
 return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),mean_ci95=np.quantile(means,[.025,.975]).tolist(),median_ci95=np.quantile(medians,[.025,.975]).tolist(),q10=float(np.quantile(v,.1)),q90=float(np.quantile(v,.9)))
def main():
 sources=json.loads((ROOT/'source_bindings.json').read_text());paths=sorted(p for p in sources if 'e5_spatial' in p and '/attempts/' in p)
 assert len(paths)==800
 images=[];repeats=[];curve_error=0.;old_error=0.;jensen_min=1.;rng=np.random.default_rng(20260920)
 for path in paths:
  raw=(REPO/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sources[path]
  d=json.loads(raw)['result']
  for ref,group in d['references'].items():
   curves=group['curves'];heads=d['heads'];method_auc={};random_head=[];random_old=[];oracle=[]
   for h in heads:
    per=[c for c in curves if c['head']==h];env={}
    for c in per:
     points=[(p['actual_area_fraction'],abs(d['clean_score']-p['isolated_score'])) for p in c['points']]
     value=auc(points);curve_error=max(curve_error,abs(value-c['absolute_isolated_effect_auc_div_0_25']))
     for area,effect in points:env[area]=max(env.get(area,0),effect)
     if c['method']!='random_regions':method_auc.setdefault(c['method'],[]).append(value)
    rc=sorted((c for c in per if c['method']=='random_regions'),key=lambda c:c['repeat']);assert [c['repeat'] for c in rc]==list(range(20))
    signed=np.array([[d['clean_score']-p['isolated_score'] for p in c['points']] for c in rc]);areas=[p['actual_area_fraction'] for p in rc[0]['points']]
    vals=np.array([auc(list(zip(areas,np.abs(v)))) for v in signed]);old=auc(list(zip(areas,np.abs(signed.mean(0)))))
    jensen_min=min(jensen_min,float(vals.mean()-old));random_head.append(vals);random_old.append(old);oracle.append(auc(sorted(env.items())))
   r=np.array(random_head);per_repeat=r.mean(0);new=float(r.mean());old=float(np.mean(random_old));scores={k:float(np.mean(v)) for k,v in method_auc.items()}
   # Independent head randomizations imply SE(mean heads) from sum head variances.
   se=float(np.sqrt(np.var(r,axis=1,ddof=1).sum()/20)/len(heads));ep=scores['endpoint_component'];tay=scores['regional_taylor'];ora=float(np.mean(oracle))
   for key,val in {**scores,'random_regions':old}.items():old_error=max(old_error,abs(val-group['image_method_scores'][key]))
   identity=dict(model=d['model_id'],reference=ref,sample_id=d['sample_id'])
   images.append(dict(**identity,endpoint=ep,taylor=tay,random_corrected=new,random_legacy=old,random_mc_se=se,random_relative_mc_se=se/new if new else 0.,endpoint_minus_random=ep-new,taylor_minus_random=tay-new,endpoint_minus_random_legacy=ep-old,taylor_minus_random_legacy=tay-old,oracle=ora,random_regret=(ora-new)/ora if ora else None,split_half_difference=float(per_repeat[:10].mean()-per_repeat[10:].mean()),loo_max_deviation=float(np.max(np.abs((per_repeat.sum()-per_repeat)/19-new))),clean_logit=d['clean_score']))
   for rep in range(20):repeats.append(dict(**identity,repeat=rep,random_auc=float(per_repeat[rep])))
 assert curve_error<1e-12 and old_error<1e-12 and jensen_min>=-1e-12
 summaries=[];repeat_summary=[];stability=[]
 for model in ('vit_b16','deit_s16'):
  for ref in ('normalized_zero','blur_rgb'):
   rs=[r for r in images if r['model']==model and r['reference']==ref];assert len(rs)==400
   summaries.append(dict(model=model,reference=ref,**{key:estimate([r[key] for r in rs],rng) for key in rs[0] if key not in ('model','reference','sample_id')}))
   grid=np.array([[r['random_auc'] for r in repeats if r['model']==model and r['reference']==ref and r['repeat']==k] for k in range(20)]).T
   for k in range(20):repeat_summary.append(dict(model=model,reference=ref,repeat=k,cohort_mean=float(grid[:,k].mean()),cohort_median=float(np.median(grid[:,k]))))
   center=grid.mean(1);draws={}
   for k in (5,10,20):
    values=[];deltas=[]
    for _ in range(2000):
     sampled=np.take_along_axis(grid,rng.integers(20,size=(400,k)),axis=1).mean(1)
     values.append(float(sampled.mean()));deltas.append(float(np.median(np.abs(sampled-center))))
    draws[str(k)]=dict(cohort_mean_mc_ci95=np.quantile(values,[.025,.975]).tolist(),median_image_absolute_mc_error=float(np.median(deltas)))
   # Fixed-image MC interval complements image bootstrap; never represents image uncertainty.
   ep_mean=float(np.mean([r['endpoint'] for r in rs]));tay_mean=float(np.mean([r['taylor'] for r in rs]));ci=draws['20']['cohort_mean_mc_ci95']
   stability.append(dict(model=model,reference=ref,resampling=draws,relative_mc_se_median=float(np.median([r['random_relative_mc_se'] for r in rs])),relative_mc_se_q90=float(np.quantile([r['random_relative_mc_se'] for r in rs],.9)),endpoint_difference_mc_ci95=[ep_mean-ci[1],ep_mean-ci[0]],taylor_difference_mc_ci95=[tay_mean-ci[1],tay_mean-ci[0]],scope='MC uncertainty conditional on observed 20 repeats and fixed images; no universal sufficiency claim'))
 table('baseline_image_values.csv',images);table('baseline_repeat_image_values.csv',repeats);table('baseline_repeat_cohort_values.csv',repeat_summary)
 m2path='reports/nns_vit_endpoint_m2_r1_20260919/auc_tests.json';raw=(REPO/m2path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sources[m2path];m2=json.loads(raw)
 selected=[r for r in m2['tests'] if r['baseline']=='regional_taylor' and r['direction']=='baseline_greater' and r['geometry']=='patch']
 if not selected:
  print('M2 geometries',sorted(set(r['geometry'] for r in m2['tests'])));raise ValueError('M2_GEOMETRY')
 assert len(selected)==24
 counts={outcome:sum(r['reject'] for r in selected if r['outcome']==outcome) for outcome in ('full','isolated')};assert counts==dict(full=11,isolated=12)
 write('baseline_analysis.json',dict(status='COMPLETE_CORRECTED_ESTIMAND',source_tasks=800,image_reference_rows=len(images),repeat_rows=len(repeats),curve_auc_max_error=curve_error,legacy_score_max_error=old_error,jensen_min_difference=jensen_min,summary=summaries,repeat_stability=stability))
 write('m2_inherited.json',dict(source=m2path,source_sha256=sources[m2path],holm_family=672,tests=selected,taylor_greater_counts=counts,original_C4b='NOT_SUPPORTED'))
 print(json.dumps(dict(status='COMPLETE',rows=len(images),curve_error=curve_error,counts=counts)))
if __name__=='__main__':main()
