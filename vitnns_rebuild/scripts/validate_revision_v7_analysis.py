#!/usr/bin/env python3
"""Independent validation for the revision-v7 CPU-only evidence root."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/"reports/nns_vit_endpoint_revision_v7_r4_20260923"
ANALYSIS=REPO/"scripts/run_revision_v7_analysis.py"
VALIDATOR=Path(__file__).resolve()
TEST_FILE=REPO/"tests/test_revision_v7.py"
E4_INDEX=REPO/"reports/nns_vit_endpoint_e4_intervention_r3_20260918/combined_result_index.json"
E5_ROOT=REPO/"reports/nns_vit_revision_v5_r1_20260921/e5"
E5_ANALYSIS=REPO/"reports/nns_vit_revision_v5_r1_20260921/analysis/e5_image_values.csv"
E6_PER_IMAGE=REPO/"reports/nns_vit_endpoint_e6_release_r3_20260918/per_image_statistics.json"
EXPECTED_STATUS="REVISION_V7_REANALYSIS_PASS_WITH_ALL_STRATA_REPORTED"
EPS=1e-12


def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def record(path:Path)->dict[str,Any]:
 return {"path":path.relative_to(REPO).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path)}


def ranks(values:Sequence[float])->np.ndarray:
 x=np.asarray(values,np.float64);order=np.argsort(x,kind='stable');out=np.empty(len(x),np.float64);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  out[order[i:j]]=(i+j-1)/2;i=j
 return out


def rho(left:Sequence[float],right:Sequence[float])->float|None:
 x=np.asarray(left,np.float64);y=np.asarray(right,np.float64)
 if len(x)<4 or not np.isfinite(x).all() or not np.isfinite(y).all():return None
 x=ranks(x);y=ranks(y);x-=x.mean();y-=y.mean();den=float(np.linalg.norm(x)*np.linalg.norm(y))
 return None if den==0 else float(x@y/den)


def residual_rho(rows:list[dict[str,Any]],controls:tuple[str,...])->float|None:
 x=ranks([r['m'] for r in rows]);y=ranks([r['absolute_effect'] for r in rows]);cols=[np.ones(len(rows))]
 for key in controls:
  levels=sorted({r[key] for r in rows},key=str)
  cols.extend(np.asarray([float(r[key]==level) for r in rows]) for level in levels[1:])
 design=np.column_stack(cols)
 if len(rows)-np.linalg.matrix_rank(design)<3 or np.linalg.cond(design)>1e12:return None
 x-=design@np.linalg.lstsq(design,x,rcond=None)[0];y-=design@np.linalg.lstsq(design,y,rcond=None)[0];den=float(np.linalg.norm(x)*np.linalg.norm(y))
 return None if den<=EPS else float(x@y/den)


def csv_rows(name:str)->list[dict[str,str]]:
 with (ROOT/name).open() as f:return list(csv.DictReader(f))


def table6_key(row:dict[str,str])->tuple[str,str,str,str,str]:
 return row['model'],row['reference'],row['image'],row['metric'],row['block']


def validate_table6(errors:list[str])->float:
 observed={table6_key(r):r for r in csv_rows('table6_stratified_image_values.csv')};old=json.loads(E6_PER_IMAGE.read_text());legacy={(r['model_id'],r['reference_id'],r['sample_id']):r for r in old['rows']}
 index=json.loads(E4_INDEX.read_text());maximum=0.;expected_rows=0
 for item in index['records']:
  result=json.loads((REPO/item['source_result_path']).read_text())['result'];model=result['model_id'];image=result['sample_id']
  for ref in ('normalized_zero','blur_rgb'):
   rows=result['finite_utility'][ref]['components']
   for family,ops in (('heads',{'head'}),('routing_value',{'routing','value'}),('residual_branches',{'attention','mlp'})):
    part=[r for r in rows if r['operation'] in ops];value=rho([r['m'] for r in part],[r['absolute_effect'] for r in part]);target=float(observed[model,ref,image,'historical_pooled_'+family,'']['value']);maximum=max(maximum,abs(value-target),abs(value-legacy[model,ref,image]['c4a'][family]['primary']['value']));expected_rows+=1
   for metric,op in (('A_only_across_blocks','routing'),('V_only_across_blocks','value'),('attention_branch_only_across_blocks','attention'),('mlp_branch_only_across_blocks','mlp')):
    part=sorted((r for r in rows if r['operation']==op),key=lambda r:r['block']);value=rho([r['m'] for r in part],[r['absolute_effect'] for r in part]);cell=observed[model,ref,image,metric,''];maximum=max(maximum,abs(value-float(cell['value'])));expected_rows+=1
   block_values=[]
   for block in range(12):
    part=sorted((r for r in rows if r['operation']=='head' and r['block']==block),key=lambda r:r['component_id']);value=rho([r['m'] for r in part],[r['absolute_effect'] for r in part]);cell=observed[model,ref,image,'heads_within_block',str(block)];maximum=max(maximum,abs(value-float(cell['value'])));block_values.append(np.clip(value,-1+EPS,1-EPS));expected_rows+=1
   value=float(np.tanh(np.mean(np.arctanh(block_values))));maximum=max(maximum,abs(value-float(observed[model,ref,image,'heads_within_block_fisher_mean','']['value'])));expected_rows+=1
   for ops,prefix in (({'routing','value'},'A_V'),({'attention','mlp'},'residual')):
    part=[r for r in rows if r['operation'] in ops]
    for controls,suffix in ((('operation',),'operation'),(('block',),'block'),(('operation','block'),'operation_and_block')):
     value=residual_rho(part,controls);cell=observed[model,ref,image,f'{prefix}_adjusted_{suffix}_rank_correlation',''];maximum=max(maximum,abs(value-float(cell['value'])));expected_rows+=1
   by={(r['block'],r['operation']):r for r in rows if r['operation'] in {'routing','value'}};signs=[]
   for block in range(12):
    a=by[block,'routing'];v=by[block,'value'];product=(a['m']-v['m'])*(a['absolute_effect']-v['absolute_effect'])
    if product!=0:signs.append(product>0)
   value=float(np.mean(signs));maximum=max(maximum,abs(value-float(observed[model,ref,image,'A_V_paired_concordance','']['value'])));expected_rows+=1
 if len(observed)!=expected_rows:errors.append(f'TABLE6_ROW_COUNT:{len(observed)}!={expected_rows}')
 if maximum>1e-12:errors.append(f'TABLE6_VALUE_MISMATCH:{maximum}')
 return maximum


def validate_h4(errors:list[str])->float:
 observed={(r['model'],r['image'],int(r['block'])):r for r in csv_rows('h4_recomputed_image_values.csv')};old={}
 with E5_ANALYSIS.open() as f:
  for r in csv.DictReader(f):old[r['model'],r['image'],r['hypothesis']]=float(r['value'])
 maximum=0.;count=0
 for metadata in sorted(E5_ROOT.glob('*.json')):
  rec=json.loads(metadata.read_text());path=REPO/rec['output'];diff=[]
  if sha256(path)!=rec['output_sha256']:errors.append('E5_HASH:'+rec['task_id']);continue
  with np.load(path,allow_pickle=False) as z:
   for block in (0,3,7,11):
    x=z[f'b{block:02d}_heads_absolute_mass'];a=np.abs(z[f'b{block:02d}_head_activation_gradient']);y=np.abs(z[f'b{block:02d}_normalized_zero_head_effect']);pr=rho(x,y);ar=rho(a,y);value=pr-ar;diff.append(value);row=observed[rec['model_id'],rec['image'],block];maximum=max(maximum,abs(pr-float(row['path_map_mass_rho'])),abs(ar-float(row['activation_gradient_rho'])),abs(value-float(row['difference'])));count+=1
  h4=float(np.mean(diff));maximum=max(maximum,abs(h4-old[rec['model_id'],rec['image'],'H4']))
 if len(observed)!=count or count!=6400:errors.append(f'H4_ROW_COUNT:{len(observed)}:{count}')
 if maximum>1e-12:errors.append(f'H4_VALUE_MISMATCH:{maximum}')
 return maximum


def validate_schema(errors:list[str])->None:
 schema=json.loads((ROOT/'analysis_schema.json').read_text())
 if not schema.get('all_json_objects_closed'):errors.append('SCHEMA_NOT_CLOSED')
 for value in schema['json'].values():
  if value.get('additionalProperties') is not False or set(value.get('required',()))!=set(value.get('properties',())):errors.append('JSON_SCHEMA_OPEN')
 for name,entry in schema['csv'].items():
  path=ROOT/name
  if not path.exists():errors.append('MISSING_CSV:'+name);continue
  with path.open() as f:header=next(csv.reader(f))
  if header!=entry['header'] or entry.get('additional_columns') is not False:errors.append('CSV_SCHEMA:'+name)


def main()->int:
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(sys.modules).intersection({'torch','timm','PIL'}):raise ValueError('CPU_ONLY_BOUNDARY')
 errors=[]
 receipt=json.loads((ROOT/'stage1_receipt.json').read_text())
 for key,path in (('script_sha256',ANALYSIS),('validator_sha256',VALIDATOR),('test_sha256',TEST_FILE)):
  if receipt[key]!=sha256(path):errors.append('STAGE1_CODE_HASH_DRIFT:'+key)
 validate_schema(errors);table_error=validate_table6(errors);h4_error=validate_h4(errors)
 expected={'table6_stratified_image_values.csv':43200,'table6_stratified_summary.csv':108,'table6_scale_diagnostics.csv':384,'h4_recomputed_image_values.csv':6400,'h4_recomputed_summary.csv':16,'pvalue_audit.csv':72,'locality_radius_sensitivity.csv':12,'residual_tv_quality_diagnostics.csv':16}
 for name,count in expected.items():
  rows=csv_rows(name)
  if len(rows)!=count:errors.append(f'COUNT:{name}:{len(rows)}!={count}')
 pvalues=csv_rows('pvalue_audit.csv')
 if any(r['display']=='p=0' or (r['display']=='p<10^{-6}' and not float(r['holm_p'])<1e-6) for r in pvalues):errors.append('PVALUE_DISPLAY')
 concordance=[r for r in csv_rows('table6_stratified_summary.csv') if r['metric']=='A_V_paired_concordance']
 if len(concordance)!=4 or any(r['raw_p'] or r['holm_p'] for r in concordance):errors.append('CONCORDANCE_MUST_BE_DESCRIPTIVE')
 quality=csv_rows('residual_tv_quality_diagnostics.csv')
 if any(int(r['n'])>400 for r in quality):errors.append('QUALITY_UNIT_NOT_IMAGE')
 counter=json.loads((ROOT/'cut_definition_counterexample.json').read_text())
 if counter['structural_cut_valid'] or counter['pointwise_derivative_on_omitted_branch']!=0 or counter['pointwise_closure_error_after_omission']!=0:errors.append('COUNTEREXAMPLE')
 outputs=[ROOT/name for name in expected]
 report={"schema_version":1,"status":"PASS" if not errors else "FAIL","checks":[{"id":"closed_schemas","status":"PASS" if not any('SCHEMA' in x for x in errors) else "FAIL","detail":"all declared JSON objects and CSV headers are closed"},{"id":"table6_independent_recalculation","status":"PASS" if table_error<=1e-12 else "FAIL","detail":f"max_abs_error={table_error:.17g}"},{"id":"h4_independent_recalculation","status":"PASS" if h4_error<=1e-12 else "FAIL","detail":f"max_abs_error={h4_error:.17g}"},{"id":"coverage","status":"PASS" if not any(x.startswith('COUNT:') for x in errors) else "FAIL","detail":json.dumps(expected,sort_keys=True)}],"errors":errors,"cpu_only":True,"gpu_model_imports":0,"old_family_max_abs_error":table_error,"h4_max_abs_error":h4_error,"output_sha256":[record(p) for p in outputs]}
 (ROOT/'validation_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 if errors:
  print(json.dumps(report,indent=2));return 1
 manifest=json.loads((ROOT/'artifact_manifest.json').read_text());files=[ROOT/r['path'].split('/')[-1] for r in manifest['files'] if not r['path'].endswith('/validation_report.json')]+[ROOT/'validation_report.json'];manifest={"schema_version":1,"files":[record(p) for p in files]};(ROOT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 receipt={"schema_version":1,"status":EXPECTED_STATUS,"cpu_only":True,"gpu_inference_runs":0,"table6_images":800,"h4_images":1600,"artifact_manifest_sha256":sha256(ROOT/'artifact_manifest.json'),"validation_report_sha256":sha256(ROOT/'validation_report.json'),"old_sealed_roots_modified":False,"next":"REVISION_V7_STAGE3_INTEGRATION_ONLY"};(ROOT/'terminal_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps({"status":EXPECTED_STATUS,"checks":len(report['checks']),"table6_max_error":table_error,"h4_max_error":h4_error}))
 return 0


if __name__=='__main__':raise SystemExit(main())
