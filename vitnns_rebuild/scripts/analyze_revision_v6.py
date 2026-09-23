#!/usr/bin/env python3
"""CPU-only independent recomputation of coordinate estimands and threshold sensitivity."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dumpcsv(path,rows,fields=None):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def interval(values,key):
 values=np.asarray(values,float);rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('2026092302|'+str(key)).encode()).digest()[:8],'big'))
 draws=[]
 for start in range(0,10000,250):draws.extend(np.median(values[rng.integers(len(values),size=(250,len(values)))],axis=1))
 return np.quantile(draws,[.025,.975]).tolist()
def metrics(patch,tokens):
 pm=patch.astype(np.float64);s=pm.sum(1);p=np.divide(pm,s[:,None],out=np.zeros_like(pm),where=s[:,None]>0)
 entropy=-(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0)).sum(1)/np.log(196);entropy[s==0]=np.nan
 locality=np.full(len(pm),np.nan);distance=locality.copy();rr,cc=np.indices((14,14));rr=rr.ravel();cc=cc.ravel()
 for t in set(tokens[tokens>0]):
  ix=np.flatnonzero((tokens==t)&(s>0));tr,tc=divmod(int(t)-1,14);dr=abs(rr-tr);dc=abs(cc-tc);mask=np.maximum(dr,dc)<=1
  locality[ix]=p[ix][:,mask].sum(1)-mask.sum()/196;distance[ix]=p[ix]@np.sqrt(dr*dr+dc*dc)/(13*np.sqrt(2))
 return p,dict(entropy=entropy,locality=locality,distance=distance)
def recompute(root,pilot=False):
 folder=root/('pilot' if pilot else 'tasks');bindings=json.loads((root/'cohort_bindings.json').read_text());tasks={t['task_id']:t for t in bindings['tasks']}
 ids=bindings['pilot_task_ids'] if pilot else list(tasks);image_rows=[];sources=[];raw_checks=0;raw_max=0.
 for tid in ids:
  t=tasks[tid]['payload'];path=folder/(tid+'.npz');assert path.is_file(),tid
  with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
  assert set(d)=={'block','object','coordinate','token','channel','patch_mass'}
  n=18432 if t['kind']=='units' else 252;assert len(d['block'])==n and d['patch_mass'].shape==(n,196) and np.isfinite(d['patch_mass']).all() and (d['patch_mass']>=0).all()
  if pilot:
   with np.load(folder/(tid+'_raw.npz'),allow_pickle=False) as raw:
    for key in raw.files:
     v=raw[key];expected=np.abs(v).reshape(3,14,16,14,16).sum((0,2,4)).ravel();observed=d['patch_mass'][int(key)];err=float(abs(expected-observed).max());raw_max=max(raw_max,err);assert np.allclose(expected,observed,atol=1e-6,rtol=2e-5);raw_checks+=1
  p,mm=metrics(d['patch_mass'],d['token'])
  for coordinate in ['normalized_endpoint','raw_gradient','common_origin']:
   for obj in (['unit'] if t['kind']=='units' else ['Q','K','V','attention','mlp']):
    if obj in ['attention','mlp']:
     ix=np.flatnonzero((d['coordinate']==coordinate)&(d['object']==obj+'_output'));jx=np.flatnonzero((d['coordinate']==coordinate)&(d['object']==obj+'_identity'));assert np.array_equal(d['block'][ix],d['block'][jx]);values=.5*np.abs(p[ix]-p[jx]).sum(1);values[(d['patch_mass'][ix].sum(1)==0)|(d['patch_mass'][jx].sum(1)==0)]=np.nan;data={'output_identity_tv':values};blocks=d['block'][ix]
    else:
     ix=np.flatnonzero((d['coordinate']==coordinate)&(d['object']==obj));blocks=d['block'][ix];data={k:v[ix] for k,v in mm.items() if obj=='unit' or k=='entropy'}
    for metric,values in data.items():
     for contrast,earlymask,latemask in [('late_minus_early',blocks<=3,blocks>=8),('block11_minus_block0',blocks==0,blocks==11)]:
      e=values[earlymask];l=values[latemask];ve=e[np.isfinite(e)];vl=l[np.isfinite(l)];early=float(np.median(ve)) if len(ve) else None;late=float(np.median(vl)) if len(vl) else None
      image_rows.append(dict(model=t['model_id'],image=t['sample_id'],coordinate=coordinate,object=obj,metric=metric,contrast=contrast,early=early,late=late,value=None if early is None or late is None else late-early,valid_units=len(ve)+len(vl),expected_units=len(e)+len(l)))
  sources.append(dict(path=str(path.relative_to(R)),sha256=sha(path),bytes=path.stat().st_size))
 if pilot:return dict(schema_version=1,status='PASS',tasks=len(ids),rows=len(image_rows),raw_checks=raw_checks,raw_patch_max_abs_error=raw_max,sources=sources)
 groups=defaultdict(list)
 for r in image_rows:groups[tuple(r[k] for k in ['model','coordinate','object','metric','contrast'])].append(r['value'])
 summary=[]
 for key,values in sorted(groups.items()):
  valid=[v for v in values if v is not None];ci=interval(valid,key);m=float(np.median(valid));summary.append(dict(zip(['model','coordinate','object','metric','contrast'],key),n=len(valid),missing=len(values)-len(valid),median=m,ci_low=ci[0],ci_high=ci[1],direction='positive' if ci[0]>0 else 'negative' if ci[1]<0 else 'crosses_zero'))
 dumpcsv(root/'coordinate_sensitivity_image_values.csv',image_rows);dumpcsv(root/'coordinate_sensitivity_summary.csv',summary)
 return dict(schema_version=1,status='PASS',tasks=len(ids),rows=len(image_rows),summaries=len(summary),sources=sources)
def thresholds(root):
 p=R/'reports/nns_vit_endpoint_e6_release_r3_20260918/per_image_statistics.json';d=json.loads(p.read_text())['rows'];f=json.loads((p.parent/'family_decisions.json').read_text())['families'];output=[]
 for family in f:
  if family['claim']!='C4a':continue
  test=family['test_results'][0];values=[row['c4a'][family['component_family']]['primary']['value'] for row in d if row['model_id']==family['model_id'] and row['reference_id']==family['reference_id']];values=[v for v in values if v is not None];m=float(np.median(values));assert abs(m-test['estimate'])<1e-12
  ci=interval(values,family['family_id'])
  for threshold in [.05,.10,.15]:output.append(dict(family=family['family_id'],metric='component_rho',threshold=threshold,n=len(values),median=m,ci_low=ci[0],ci_high=ci[1],point_exceeds=m>=threshold,ci_exceeds=ci[0]>threshold,original_composite=family['status']))
 p2=R/'papers/nns_vit_endpoint/publication_r2_20260920/evidence/supplement_v1/baseline_image_values.csv'
 with p2.open() as stream:rows=list(csv.DictReader(stream))
 for model in ['vit_b16','deit_s16']:
  for ref in ['normalized_zero','blur_rgb']:
   for baseline in ['taylor','random_corrected']:
    vals=[float(r['endpoint'])-float(r[baseline]) for r in rows if r['model']==model and r['reference']==ref];family=model+'|'+ref+'|'+baseline;ci=interval(vals,family);m=float(np.median(vals))
    for threshold in [0,.025,.05,.10]:output.append(dict(family=family,metric='regional_auc_gain',threshold=threshold,n=len(vals),median=m,ci_low=ci[0],ci_high=ci[1],point_exceeds=m>=threshold,ci_exceeds=ci[0]>threshold,original_composite='NOT_SUPPORTED_all_baselines'))
 dumpcsv(root/'threshold_sensitivity.csv',output)
 return [dict(path=str(p.relative_to(R)),sha256=sha(p),bytes=p.stat().st_size) for p in [p,p2]]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();result=recompute(a.root,a.pilot)
 if not a.pilot:result['threshold_sources']=thresholds(a.root)
 (a.root/('pilot_validation.json' if a.pilot else 'independent_recalculation.json')).write_text(json.dumps(result,indent=2)+chr(10));print({k:v for k,v in result.items() if k not in ['sources','threshold_sources']})
