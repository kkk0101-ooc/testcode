#!/usr/bin/env python3
"""Recompute Tables 6-10 from immutable component and replacement records, on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,gzip,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from analyze_revision_v6 import interval,dumpcsv
OUT=R/'reports/nns_vit_endpoint_revision_v6_r1_20260923'
SOURCES=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p,expected=None):
 raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if expected:assert digest==expected,str(p)
 SOURCES.append(dict(path=str(p.relative_to(R)),sha256=digest,bytes=len(raw)))
 return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
def rho(x,y):
 a=rankdata(x);b=rankdata(y)
 return float(np.corrcoef(a,b)[0,1]) if len(a)>=4 and np.ptp(a)>0 and np.ptp(b)>0 else None
def stat(vals,key):
 valid=[v for v in vals if v is not None];ci=interval(valid,key)
 return dict(n=len(valid),missing=len(vals)-len(valid),median=float(np.median(valid)),ci_low=ci[0],ci_high=ci[1])
def metric(x,y):
 order=np.argsort(-np.asarray(x),kind='stable');effect=np.asarray(y);oracle=np.sort(effect)[::-1]
 return dict(rho=rho(x,y),capture1=float(effect[order[0]]/oracle[0]),capture3=float(effect[order[:3]].sum()/oracle[:3].sum()),regret=float(oracle[0]-effect[order[0]]))
def auc(y):return float(np.trapz(y,[0,.0625,.125,.25])/.25)
def main():
 groups=defaultdict(list);image=[];headgroups=defaultdict(list)
 root=R/'reports/nns_vit_endpoint_m1_r1_20260918';manifest=read(root/'metrics_manifest.json');expected={r['task_id']:r['sha256'] for r in manifest['records']}
 old=read(R/'reports/nns_vit_endpoint_e6_release_r3_20260918/per_image_statistics.json');lookup={(r['model_id'],r['reference_id'],r['sample_id']):r for r in old['rows']};maxdiff=0.
 for tid in sorted(expected):
  d=read(root/'per_image'/(tid+'.json.gz'),expected[tid]);model=d['model_id'];sid=d['sample_id']
  for ref,rows in d['component_rows'].items():
   for family,ops in [('heads',{'head'}),('routing_value',{'routing','value'}),('residual_branches',{'attention','mlp'})]:
    part=sorted([r for r in rows if r['operation'] in ops],key=lambda r:r['component_id']);value=rho([r['m'] for r in part],[r['absolute_effect'] for r in part]);target=lookup[model,ref,sid]['c4a'][family]['primary']['value'];assert value is not None and target is not None;maxdiff=max(maxdiff,abs(value-target));assert abs(value-target)<1e-12;groups[model,ref,family].append(value)
   local=defaultdict(list)
   for b in range(12):
    part=sorted([r for r in rows if r['operation']=='head' and r['block']==b],key=lambda r:r['component_id']);effect=[r['absolute_effect'] for r in part]
    for method,field in [('local_map','m'),('taylor','p_ref'),('activation_gradient','gradient_activation')]:
     for name,value in metric([abs(r[field]) for r in part],effect).items():local[method,name].append(value)
   for key,values in local.items():
    val=float(np.mean(values));headgroups[(model,ref)+key].append(val);image.append(dict(model=model,reference=ref,image=sid,method=key[0],metric=key[1],value=val))
 table6=[dict(model=k[0],reference=k[1],family=k[2],**stat(v,k)) for k,v in sorted(groups.items())]
 table7=[dict(model=k[0],reference=k[1],method=k[2],metric=k[3],**stat(v,k)) for k,v in sorted(headgroups.items())]
 dumpcsv(OUT/'table6_recomputed.csv',table6);dumpcsv(OUT/'table7_recomputed.csv',table7);dumpcsv(OUT/'table7_image_values.csv',image)
 root=R/'reports/nns_vit_endpoint_e5_spatial_r1_20260918';manifest=read(root/'artifact_manifest.json');paths=[r for r in manifest['files'] if r['path'].startswith('formal_results/attempts/') and r['path'].endswith('.json')];assert len(paths)==800
 groups=defaultdict(list);paired=defaultdict(list);rowsout=[];curve_error=0.
 for ent in paths:
  d=read(root/ent['path'],ent['sha256'])['result'];model=d['model_id'];sid=d['sample_id']
  for ref,entry in d['references'].items():
   per=defaultdict(list);envelope=defaultdict(lambda:np.zeros(4))
   for c in entry['curves']:
    effects=np.array([abs(d['clean_score']-pt['isolated_score']) for pt in c['points']]);value=auc(effects);curve_error=max(curve_error,abs(value-c['absolute_isolated_effect_auc_div_0_25']));per[c['method'],c['head']].append(value);envelope[c['head']]=np.maximum(envelope[c['head']],effects)
   scores={m:float(np.mean([np.mean(per[m,h]) for h in d['heads']])) for m in sorted({k[0] for k in per})}
   oracle=float(np.mean([auc(envelope[h]) for h in d['heads']]))
   for m,value in scores.items():
    for metric0,val in [('auc',value),('regret',1-value/oracle),('logit_ratio',value/abs(d['clean_score']))]:groups[model,ref,m,metric0].append(val)
    paired[model,ref,m].append(scores['endpoint_component']-value)
    rowsout.append(dict(model=model,reference=ref,image=sid,method=m,auc=value,oracle=oracle,regret=1-value/oracle,logit_ratio=value/abs(d['clean_score']),clean_logit=d['clean_score']))
   groups[model,ref,'empirical_oracle','auc'].append(oracle);groups[model,ref,'clean_logit','logit'].append(d['clean_score'])
 dumpcsv(OUT/'table8_recomputed.csv',[dict(model=k[0],reference=k[1],baseline=k[2],**stat(v,k)) for k,v in sorted(paired.items()) if k[2]!='endpoint_component'])
 dumpcsv(OUT/'table9_10_recomputed.csv',[dict(model=k[0],reference=k[1],method=k[2],metric=k[3],**stat(v,k)) for k,v in sorted(groups.items())])
 dumpcsv(OUT/'regional_recomputed_image_values.csv',rowsout)
 assert curve_error<1e-12
 out=dict(schema_version=1,status='PASS',component_rho_max_error=maxdiff,curve_auc_max_error=curve_error,source_files=len(SOURCES),sources=SOURCES,interval_identity='new descriptive image bootstrap; original scientific decisions unchanged',oracle_definition='pointwise maximum across evaluated region sets at each budget, then integrate and average heads; envelope may join different rankings')
 (OUT/'table_recalculation.json').write_text(json.dumps(out,indent=2)+chr(10));print({k:v for k,v in out.items() if k!='sources'})
if __name__=='__main__':main()
