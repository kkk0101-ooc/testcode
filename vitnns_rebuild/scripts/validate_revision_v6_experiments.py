#!/usr/bin/env python3
"""Fail-closed validation of v6 inputs, coordinate coverage and analytic evidence."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import jsonschema
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();checks=[]
 def check(name,value,detail=''):checks.append(dict(name=name,passed=bool(value),detail=str(detail)))
 freeze=json.loads((root/'protocol_freeze.json').read_text())
 for rel,expected in freeze['sha256'].items():check('freeze:'+rel,sha(root/rel)==expected)
 sources=json.loads((root/'source_bindings.json').read_text())
 for row in sources['scripts']+sources['inputs']:check('source:'+row['path'],sha(R/row['path'])==row['sha256'])
 addendum=json.loads((root/'preexecution_implementation_addendum.json').read_text())
 for row in addendum['sources']:check('implementation_addendum:'+row['path'],sha(R/row['path'])==row['sha256'])
 acquisition=json.loads((root/'production_acquisition.json').read_text());check('production_complete',acquisition['status']=='PASS' and acquisition['states']=={'completed':1000})
 check('resource_budget',acquisition['gpu_hours_reserved']<32 and acquisition['wall_seconds']<43200)
 tasks=json.loads((root/'cohort_bindings.json').read_text())['tasks'];allids={t['task_id'] for t in tasks};seen=set();outputs=[];environments=set()
 for p in (root/'tasks/receipts/attempts').glob('*/attempt-01.json'):
  d=json.loads(p.read_text());unsigned=dict(d);digest=unsigned.pop('canonical_digest');observed=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();check('canonical:'+d['task_id'],observed==digest)
  r=d['result'];seen.add(r['task_id']);required={'schema_version','task_id','model_id','image','kind','image_sha256','rows','output','output_sha256','raw_output','raw_sha256','seconds','peak_gpu_bytes','peak_cpu_rss_bytes','environment'}
  check('closed_receipt:'+r['task_id'],set(r)==required)
  t=next(t for t in tasks if t['task_id']==r['task_id'])['payload'];check('image_identity:'+r['task_id'],t['sample_id']==r['image'] and t['image_sha256']==r['image_sha256'])
  path=R/r['output'];check('output_hash:'+r['task_id'],sha(path)==r['output_sha256'])
  with np.load(path,allow_pickle=False) as z:
   check('array_keys:'+r['task_id'],set(z.files)=={'block','object','coordinate','token','channel','patch_mass'})
   n=18432 if r['kind']=='units' else 252
   check('array_rows:'+r['task_id'],len(z['block'])==n and z['patch_mass'].shape==(n,196) and z['patch_mass'].dtype==np.float32)
   check('finite:'+r['task_id'],np.isfinite(z['patch_mass']).all() and (z['patch_mass']>=0).all())
   actual=set(zip(z['block'].tolist(),z['object'].tolist(),z['coordinate'].tolist(),z['token'].tolist(),z['channel'].tolist()))
   coordinates=['normalized_endpoint','raw_gradient','common_origin']
   if r['kind']=='units':expected={(b,'unit',c,tok,ch) for b in range(12) for c in coordinates for tok in t['tokens'] for ch in t['channels'][str(b)]}
   else:expected={(b,obj,c,-1,-1) for b in range(12) for obj in ['Q','K','V','attention_output','attention_identity','mlp_output','mlp_identity'] for c in coordinates}
   check('exact_target_coordinate_coverage:'+r['task_id'],actual==expected and len(actual)==n)
  env=r['environment'];check('determinism:'+r['task_id'],env['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and env['startup_matmul_backward']=='PASS' and env['deterministic_algorithms'] and not env['cuda_matmul_allow_tf32']);environments.add(env['gpu_uuid'])
  outputs.append(dict(path=str(path.relative_to(root)),sha256=r['output_sha256'],bytes=path.stat().st_size))
 check('exact_task_set',seen==allids and len(seen)==1000)
 analytic=json.loads((root/'analytic_reference_results.json').read_text());mutation=json.loads((root/'mutation_results.json').read_text())
 for data,name in [(analytic,'analytic_reference'),(mutation,'mutation')]:
  try:jsonschema.validate(data,json.loads((root/(name+'.schema.json')).read_text()));ok=True
  except jsonschema.ValidationError as e:ok=False
  check('closed_schema:'+name,ok)
 check('analytic_arrays_hash',sha(root/'analytic_reference_arrays.npz')==analytic['array_sha256'])
 with np.load(root/'analytic_reference_arrays.npz',allow_pickle=False) as z:
  for row in analytic['rows']:
   if row['component'].startswith('direction_'):continue
   base=row['fixture']+'__'+row['component'];ref=z[base+'__analytic'];obs=z[base+'__scanner'];error=obs-ref;absolute=float(abs(error).max());relative=float(np.linalg.norm(error)/np.linalg.norm(ref))
   check('analytic_direct:'+base,np.allclose(obs,ref,atol=row['atol'],rtol=row['rtol']) and absolute==row['max_abs_error'] and relative==row['relative_l2'])
 check('mutation_coverage',len(mutation['rows'])==4 and all(r['rejected'] and r['correct_pass'] for r in mutation['rows']) and {'wrong_slot','merged_fanout'}<={r['mutation'] for r in mutation['rows']})
 recomputed=json.loads((root/'independent_recalculation.json').read_text());check('independent_complete',recomputed['status']=='PASS' and recomputed['tasks']==1000)
 import csv
 with (root/'coordinate_sensitivity_summary.csv').open() as f:rows=list(csv.DictReader(f))
 check('all_summary_groups',len(rows)==96 and all(int(r['n'])+int(r['missing'])==(100 if r['object']=='unit' else 400) for r in rows))
 check('all_directions_retained',all(r['direction'] in {'positive','negative','crosses_zero'} for r in rows))
 errors=[r for r in checks if not r['passed']];out=dict(schema_version=1,status='PASS_VALIDATION' if not errors else 'FAIL',checks=checks,errors=errors,task_count=len(seen),gpu_count=len(environments),scientific_support='see coordinate decisions; validation does not require a positive direction')
 (root/'validation_report.json').write_text(json.dumps(out,indent=2)+chr(10));print(dict(status=out['status'],checks=len(checks),errors=errors[:20]));return bool(errors)
if __name__=='__main__':raise SystemExit(main())
