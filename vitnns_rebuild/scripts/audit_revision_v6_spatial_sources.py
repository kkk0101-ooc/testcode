#!/usr/bin/env python3
"""Recompute retained spatial results from raw columnar records, never producer summaries."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,json,hashlib,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'));sys.path.insert(0,str(R/'papers/nns_vit_endpoint/scripts'))
from analyze_revision_v6 import metrics,interval,dumpcsv
from validate_gxi_step4 import m3_bootstrap_median
O=R/'reports/nns_vit_endpoint_revision_v6_r1_20260923';sources=[]
def read(p):
 b=p.read_bytes();sources.append(dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)));return json.loads(b)
def main():
 rows=[];groups=defaultdict(list);m3=R/'reports/nns_vit_endpoint_m3_r1_20260919';tasks=read(m3/'task_manifest.json')['tasks']
 for task in tasks:
  meta=read(m3/'results/attempts'/task['task_id']/'attempt-01.json')['result'];p=R/meta['metrics_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['metrics_sha256']
  with np.load(p,allow_pickle=False) as z:
   pp,metric=metrics(z['patch_absolute_mass_196'],z['token']);metric['entropy']=z['normalized_spatial_entropy'].astype(float)
   for name,vals in metric.items():
    e=vals[z['block']<=3];l=vals[z['block']>=8];value=float(np.nanmedian(l)-np.nanmedian(e));groups[task['payload']['model_id'],name].append(value)
    rows.append(dict(model=task['payload']['model_id'],image=task['payload']['sample_id'],metric=name,value=value))
 summ=[]
 for key,vals in sorted(groups.items()):
  ci=m3_bootstrap_median(vals,'M3|'+key[0]+'|late_minus_early_'+key[1]);summ.append(dict(model=key[0],metric=key[1],n=len(vals),median=float(np.median(vals)),ci_low=ci[0],ci_high=ci[1]))
 dumpcsv(O/'legacy_m3_recomputed.csv',summ)
 e3=R/'reports/nns_vit_endpoint_e3_closure_r1_20260918';positive=negative=0;targets=defaultdict(int)
 for p in sorted((e3/'formal_results/attempts').glob('*/attempt-01.json')):
  d=read(p)['result']
  for b in d['blocks']:
   for c in b['cuts']:
    assert c['pass'];positive+=1;targets[c['reconstruction_id']]+=1
   for c in b['negative_controls']:assert c['passed'];negative+=1
 assert positive==76800 and negative==28800
 fidelity=read(m3/'m3a_fidelity.json');groups=defaultdict(list)
 for row in fidelity['records']:groups[row['model'],row['dtype']].append(row['relative_l2'])
 numeric=[dict(model=k[0],dtype=k[1],n=len(v),max_relative_l2=max(v)) for k,v in sorted(groups.items())]
 dumpcsv(O/'legacy_identity_recomputed.csv',numeric)
 out=dict(schema_version=1,status='PASS',m3_tasks=len(tasks),positive_closures=positive,negative_rejections=negative,reconstruction_counts=dict(targets),fidelity_comparisons=sum(len(v) for v in groups.values()),sources=sources)
 (O/'legacy_spatial_source_audit.json').write_text(json.dumps(out,indent=2)+chr(10));print({k:v for k,v in out.items() if k!='sources'})
if __name__=='__main__':main()
