#!/usr/bin/env python3
"""Create Step-4 validation, cost and decision artifacts from attempt2."""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write(path,value):
 if path.exists():raise RuntimeError('OUTPUT_EXISTS:'+path.name)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 binding=json.loads((ROOT/'step4_attempt2_source_binding.json').read_text());models=[json.loads((ROOT/f'step4_runtime/attempt2_{mid}.json').read_text()) for mid in ('vit_b16','deit_s16')]
 drift=[]
 for row in binding['files']:
  p=REPO/row['path'];actual=(p.stat().st_size,sha(p))
  if actual!=(row['bytes'],row['sha256']):drift.append({'path':row['path'],'expected_bytes':row['bytes'],'actual_bytes':actual[0],'expected_sha256':row['sha256'],'actual_sha256':actual[1]})
 summaries=[]
 for d in models:
  cut_fail=[c['cut_id'] for c in d['primary_cuts'] if not c['pass']];block_fail=[b['block_index'] for b in d['all_blocks_first_order'] if not b['pass']];neg_fail=[n['control_id'] for n in d['negative_controls'] if not n['pass']]
  result_path=ROOT/f"step4_runtime/attempt2_{d['model_id']}.json"
  summaries.append({'model_id':d['model_id'],'status':d['status'],'result_path':result_path.relative_to(REPO).as_posix(),'result_sha256':sha(result_path),'sample_id':d['sample_id'],'class_id':d['class_id'],'primary_cut_pass_count':len(d['primary_cuts'])-len(cut_fail),'primary_cut_count':len(d['primary_cuts']),'failed_primary_cuts':cut_fail,'all_block_pass_count':len(d['all_blocks_first_order'])-len(block_fail),'all_block_count':len(d['all_blocks_first_order']),'failed_blocks':block_fail,'negative_control_pass_count':len(d['negative_controls'])-len(neg_fail),'negative_control_count':len(d['negative_controls']),'failed_negative_controls':neg_fail,'independent_batched_ig_diagnostic':d['independent_batched_ig_diagnostic']})
 validation={'schema_version':1,'stage':'ENDPOINT_PCIA_PEFA_STEP4','attempt':'attempt2','status':'PASS_VIT_PORT_FIDELITY_AND_IDENTITIES' if not drift and all(d['status']=='PASS' for d in models) else 'FAIL','source_binding_canonical_digest':binding['canonical_digest'],'source_binding_drift':drift,'models':summaries,'identity_families':['port Endpoint closure','PCIA to path IG','source-summed PEFA to Endpoint','T equals A plus R','complete-cut omega zero sum'],'raw_coordinate_shape':[1,3,224,224],'aggregation_used_for_closure':False,'source_coordinate_resolution':'one exact all-coordinate contraction','port_registry':'vit_port_registry.json','port_registry_sha256':sha(ROOT/'vit_port_registry.json'),'terminal_attempts':['step4_attempt0_receipt.json','step4_attempt1_receipt.json'],'step5_started':False};validation['canonical_digest']=canon(validation);write(ROOT/'vit_validation.json',validation)
 observed=[];projected=0.0
 for d in models:
  costs=d['costs'];seconds=costs['source_summed_pefa']['wall_seconds'];projection=seconds*64*12/3600;projected+=projection
  observed.append({'model_id':d['model_id'],'endpoint':costs['endpoint'],'ig':costs['ig'],'pcia':costs['pcia'],'source_summed_pefa':costs['source_summed_pefa'],'relative_wall_cost':{'vs_endpoint':seconds/costs['endpoint']['wall_seconds'],'vs_ig':seconds/costs['ig']['wall_seconds'],'vs_pcia':seconds/costs['pcia']['wall_seconds']},'projected_64_images_12_blocks_h20_hours':projection,'peak_cpu_rss_bytes':d['process_peak_cpu_rss_bytes']})
 retained=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
 attempt0=[json.loads((ROOT/f'step4_runtime/{mid}.json').read_text()) for mid in ('vit_b16','deit_s16')]
 attempt1=[json.loads((ROOT/f'step4_runtime/attempt1_{mid}.json').read_text()) for mid in ('vit_b16','deit_s16')]
 diagnostics=[json.loads((ROOT/f'step4_runtime/{mid}_quadrature_diagnostic.json').read_text()) for mid in ('vit_b16','deit_s16')]
 attempt0_seconds=sum(sum(d['costs'][k]['wall_seconds'] for k in ('endpoint','ig','pcia','pefa')) for d in attempt0)
 attempt1_seconds=sum(d['costs']['endpoint']['wall_seconds']+d['costs']['ig']['wall_seconds']+d['costs']['source_summed_pefa']['wall_seconds'] for d in attempt1)
 attempt2_seconds=sum(d['costs']['endpoint']['wall_seconds']+d['costs']['ig']['wall_seconds']+d['costs']['source_summed_pefa']['wall_seconds'] for d in models)
 quadrature_seconds=sum(sum(row['wall_seconds'] for row in d['rows']) for d in diagnostics)
 interactive_diagnostics_upper_bound_seconds=600.0
 gpu_seconds_observed=attempt0_seconds+attempt1_seconds+attempt2_seconds+quadrature_seconds+interactive_diagnostics_upper_bound_seconds
 cost={'schema_version':1,'stage':'ENDPOINT_PCIA_PEFA_STEP4','attempt':'attempt2','status':'PASS_COST_GATE' if projected<=16 and gpu_seconds_observed/3600<=16 and max(x['source_summed_pefa']['peak_reserved_gpu_bytes'] for x in observed)<=24*1024**3 and retained<=32*1024**3 else 'NO_GO_COST','observed':observed,'aggregate_observed':{'attempt0_seconds':attempt0_seconds,'attempt1_seconds':attempt1_seconds,'attempt2_seconds':attempt2_seconds,'quadrature_diagnostic_seconds':quadrature_seconds,'interactive_diagnostics_conservative_upper_bound_seconds':interactive_diagnostics_upper_bound_seconds,'h20_hours_upper_bound':gpu_seconds_observed/3600,'frozen_h20_hours_cap':16,'pass':gpu_seconds_observed/3600<=16},'projected_production':{'images':64,'models':2,'blocks_per_model':12,'targets':1,'h20_hours':projected,'frozen_h20_hours_cap':16,'pass':projected<=16},'resource_caps':{'wall_hours':24,'retained_gib':32,'gpu_memory_gib_per_worker':24,'cpu_rss_gib_per_worker':32,'task_seconds':1800},'retained_bytes_at_finalize':retained,'temporary_bytes_at_finalize':sum(p.stat().st_size for p in (ROOT/'step4_runtime').rglob('*') if p.is_file()),'raw_hessian_materialized':False,'raw_port_by_input_by_input_materialized':False,'notes':['Counts are logical model/derivative evaluations, not CUDA kernel launches.','Independent IG uses alpha batch 8; PCIA uses per-node batch 1. Their ViT-B float32 difference is retained as a non-gating numerical diagnostic.','Aggregate observed usage includes both failed attempts, independent quadrature diagnostics, final attempt, and a conservative 600-second allowance for interactive GPU probes.']};cost['canonical_digest']=canon(cost);write(ROOT/'cost_report.json',cost)
 print(json.dumps({'validation':validation['status'],'cost':cost['status'],'projected_h20_hours':projected,'source_drift':len(drift)},sort_keys=True));return 0 if validation['status'].startswith('PASS') and cost['status'].startswith('PASS') else 3
if __name__=='__main__':raise SystemExit(main())
