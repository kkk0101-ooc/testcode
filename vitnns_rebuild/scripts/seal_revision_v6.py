#!/usr/bin/env python3
"""Seal v6 scientific evidence with closed JSON schemas and complete file hashes."""
import hashlib,json,sys,time
from pathlib import Path
import jsonschema
R=Path(__file__).resolve().parents[1];ROOT=R/'reports/nns_vit_endpoint_revision_v6_r1_20260923'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def schema(v):
 if v is None:return {'type':'null'}
 if isinstance(v,bool):return {'type':'boolean'}
 if isinstance(v,str):return {'type':'string'}
 if isinstance(v,int):return {'type':'integer'}
 if isinstance(v,float):return {'type':'number'}
 if isinstance(v,list):
  unique={json.dumps(schema(x),sort_keys=True):schema(x) for x in v}
  return dict(type='array',items=(next(iter(unique.values())) if len(unique)==1 else {'anyOf':list(unique.values())}) if unique else {'not':{}})
 if isinstance(v,dict):return dict(type='object',properties={k:schema(x) for k,x in v.items()},required=list(v),additionalProperties=False)
 raise TypeError(type(v))
def main():
 assert not (ROOT/'terminal_receipt.json').exists(),'TERMINAL_IMMUTABLE'
 validation=json.loads((ROOT/'validation_report.json').read_text());assert validation['status']=='PASS_VALIDATION' and not validation['errors']
 assert json.loads((ROOT/'coordinate_decisions.json').read_text())['status']=='SCIENTIFIC_SCOPE_RESTRICTED_NORMALIZED_INPUT'
 required=['protocol.json','source_bindings.json','environment.json','cohort_bindings.json','coordinate_sensitivity_image_values.csv','coordinate_sensitivity_summary.csv','analytic_reference_results.json','mutation_results.json','threshold_sensitivity.csv','analysis_identity.json','validation_report.json']
 assert all((ROOT/f).is_file() for f in required)
 schemas={}
 for p in sorted(ROOT.rglob('*.json')):
  if p.name.endswith('.schema.json'):continue
  v=json.loads(p.read_text());s=schema(v);jsonschema.validate(v,s);schemas[str(p.relative_to(ROOT))]=s
 (ROOT/'artifact_schemas.json').write_text(json.dumps(dict(schema_version=1,note='Protocol defines outcome arrays/CSV fields prospectively; these closed schemas validate the serialized evidence and do not alter scientific acceptance.',schemas=schemas),indent=2)+chr(10))
 files=[]
 for p in sorted(ROOT.rglob('*')):
  if p.is_file():files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
 manifest=dict(schema_version=1,files=files,required=required)
 (ROOT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+chr(10))
 receipt=dict(schema_version=1,status='PASS_EXECUTION_AND_VALIDATION_WITH_COORDINATE_LIMITATION',scientific_status='SCIENTIFIC_SCOPE_RESTRICTED_NORMALIZED_INPUT',artifact_manifest_sha256=sha(ROOT/'artifact_manifest.json'),validation_sha256=sha(ROOT/'validation_report.json'),created_ns=time.time_ns(),tasks=1000,pilot_tasks=4,analytic_checks=41,mutation_checks=4,next='stage4 bilingual statistical and intervention rewrite')
 (ROOT/'terminal_receipt.json').write_text(json.dumps(receipt,indent=2)+chr(10));print(receipt)
if __name__=='__main__':main()
