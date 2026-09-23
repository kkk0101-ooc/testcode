#!/usr/bin/env python3
"""Recompute M4 phase contrasts from per-head and residual primitive records."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from analyze_revision_v6 import dumpcsv
from vitnns.endpoint_m4_statistics import bootstrap_median
O=R/'reports/nns_vit_endpoint_revision_v6_r1_20260923'
def main():
 groups=defaultdict(list);sources=[]
 for p in sorted((R/'reports/nns_vit_endpoint_m4_r1_20260919/results/attempts').glob('*/attempt-01.json')):
  data=p.read_bytes();d=json.loads(data)['result'];sources.append(dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)));model=d['model_id']
  for metric in ['routing_share','frozen_attention_endpoint_error']:
   vals=[float(np.median([r[metric] for r in d['attention'] if r['block']==b])) for b in range(12)]
   groups[model,'attention',metric].append(float(np.median(vals[8:])-np.median(vals[:4])))
  for kind in ['attention','mlp']:
   vals=[next(r['identity_share'] for r in d['residual'] if r['block']==b and r['residual_type']==kind) for b in range(12)]
   groups[model,kind,'identity_share'].append(float(np.median(vals[8:])-np.median(vals[:4])))
 rows=[]
 for key,vals in sorted(groups.items()):
  model,kind,metric=key
  if metric=='identity_share':seed=f'M4|test|{model}|{kind}|late_minus_early_identity_share'
  else:seed=f'M4|test|{model}|late_minus_early_{metric}'
  ci=bootstrap_median(vals,seed);rows.append(dict(model=model,object=kind,metric=metric,n=len(vals),median=float(np.median(vals)),ci_low=ci[0],ci_high=ci[1]))
 dumpcsv(O/'legacy_m4_contrasts_recomputed.csv',rows)
 (O/'legacy_m4_source_audit.json').write_text(json.dumps(dict(schema_version=1,status='PASS',sources=sources,contrasts=rows),indent=2)+chr(10));print(len(sources),len(rows))
if __name__=='__main__':main()
