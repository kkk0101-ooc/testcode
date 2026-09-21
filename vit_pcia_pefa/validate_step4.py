#!/usr/bin/env python3
"""Independent artifact and requirement audit for Step 4."""
from __future__ import annotations
import ast,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent
def need(ok,msg):
 if not ok:raise AssertionError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def load(name):return json.loads((ROOT/name).read_text())
def main():
 binding=load('step4_attempt2_source_binding.json');need(binding['status']=='FROZEN_PROSPECTIVE','binding status');stored=binding.pop('canonical_digest');need(canon(binding)==stored,'binding digest');binding['canonical_digest']=stored
 for row in binding['files']:
  p=REPO/row['path'];need(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'source drift:'+row['path'])
 for row in binding['checkpoints']:
  p=REPO/row['path'];need(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'checkpoint drift:'+row['model_id'])
 registry=load('vit_port_registry.json');stored=registry.pop('canonical_digest');need(canon(registry)==stored,'port registry digest');registry['canonical_digest']=stored;need(registry['cut_count']==216 and len(registry['cuts'])==216,'cut registry coverage')
 fields=set(registry['port_id_fields']);need(fields=={'model_id','graph_revision','block_index','producer_tensor_id','consumer_operator_id','input_slot','invocation_index','slice_spec'},'port fields')
 need({r['model_id'] for r in registry['cuts']}=={'vit_b16','deit_s16'},'registry models');need({r['block_index'] for r in registry['cuts']}==set(range(12)),'registry blocks');need(len({r['cut_id'] for r in registry['cuts']})==9,'registry families')
 results=[load(f'step4_runtime/attempt2_{mid}.json') for mid in ('vit_b16','deit_s16')]
 for d in results:
  need(d['status']=='PASS','model result:'+d['model_id']);need(d['split']=='discovery' and d['split_order']==0,'sample scope');need(len(d['primary_cuts'])==9 and all(c['pass'] for c in d['primary_cuts']),'cut gate');need(len(d['all_blocks_first_order'])==12 and all(x['pass'] for x in d['all_blocks_first_order']),'all blocks');need(len(d['negative_controls'])==9 and all(x['pass'] for x in d['negative_controls']),'negative controls');need(d['source_resolution']=='one exact all-coordinate contraction','resolution');need(d['path_ig_reference'].startswith('PCIA __full__'),'ig reference')
  for c in d['primary_cuts']:
   need(c['identity_families']==5,'five identities');need(set(c['identities'])=={'endpoint_D_closure','pcia_to_ig','pefa_to_endpoint','T_equals_A_plus_R','endpoint_minus_ig_equals_R','omega_zero_sum'},'identity keys');need(all(v['pass'] for k,v in c['identities'].items() if k!='omega_zero_sum'),'identity fail');need(len(c['identities']['omega_zero_sum'])==2 and all(v['pass'] for v in c['identities']['omega_zero_sum']),'omega')
  ids={n['control_id'] for n in d['negative_controls']};need({'omit_residual_identity','duplicate_residual_identity','omit_attention_head','swap_residual_input_slots','duplicate_serial_mlp_cuts','detach_second_order_omega','omit_qkv_value_head_pcia','merge_qkv_consumer_cotangents','omit_qkv_value_head_pefa'}==ids,'negative IDs');need(d['worker_identity']['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and d['worker_identity']['startup_matmul_backward']=='PASS','runtime identity');need(not d['raw_hessian_materialized'] and not d['raw_port_by_input_by_input_materialized'],'materialization')
 validation=load('vit_validation.json');stored=validation.pop('canonical_digest');need(canon(validation)==stored,'validation digest');validation['canonical_digest']=stored;need(validation['status']=='PASS_VIT_PORT_FIDELITY_AND_IDENTITIES' and not validation['source_binding_drift'],'validation status');need(not validation['aggregation_used_for_closure'] and not validation['step5_started'],'scope')
 cost=load('cost_report.json');stored=cost.pop('canonical_digest');need(canon(cost)==stored,'cost digest');cost['canonical_digest']=stored;need(cost['status']=='PASS_COST_GATE','cost status');need(cost['aggregate_observed']['pass'] and cost['aggregate_observed']['h20_hours_upper_bound']<=16,'observed budget');need(cost['projected_production']['pass'] and cost['projected_production']['h20_hours']<=16,'projected budget');need(cost['retained_bytes_at_finalize']<=32*1024**3,'storage');need(max(x['source_summed_pefa']['peak_reserved_gpu_bytes'] for x in cost['observed'])<=24*1024**3,'gpu memory');need(max(x['peak_cpu_rss_bytes'] for x in cost['observed'])<=32*1024**3,'cpu rss');need(not cost['raw_hessian_materialized'] and not cost['raw_port_by_input_by_input_materialized'],'cost materialization')
 attempt0=load('step4_attempt0_receipt.json');attempt1=load('step4_attempt1_receipt.json');need(attempt0['immutable_evidence'] and 'FAIL_QUADRATURE_CLOSURE' in attempt0['failure_codes'] and 'NO_GO_COST' in attempt0['failure_codes'],'attempt0');need(attempt1['immutable_evidence'] and attempt1['status']=='TERMINAL_PARTIAL_FAIL_QUADRATURE_CLOSURE','attempt1')
 binding1=load('step4_attempt1_source_binding.json');stored1=binding1.pop('canonical_digest');need(canon(binding1)==stored1==attempt1['source_binding_canonical_digest'],'attempt1 binding digest');binding1['canonical_digest']=stored1
 for row in validation['models']:
  p=REPO/row['result_path'];need(p.is_file() and sha(p)==row['result_sha256'],'result drift:'+row['model_id'])
 source=ast.parse((ROOT/'src/vit_adapter.py').read_text());calls=[n for n in ast.walk(source) if isinstance(n,ast.Call)];need(any(isinstance(n.func,ast.Attribute) and n.func.attr=='jvp' for n in calls),'JVP absent');need(any(isinstance(n.func,ast.Attribute) and n.func.attr=='vjp' for n in calls),'VJP absent');text=(ROOT/'src/vit_adapter.py').read_text();need('torch.autograd.functional.hessian' not in text and 'torch.autograd.functional.jacobian' not in text,'explicit Hessian/Jacobian')
 tests=subprocess.run([str(REPO/'.conda/envs/vitnns/bin/python'),'-m','pytest','-q','vit_pcia_pefa/tests/test_step4_vit_adapter.py'],cwd=REPO,env={'CUDA_VISIBLE_DEVICES':'','PATH':'/usr/bin:/bin','PYTHONPATH':'.:vitnns_rebuild/src'},capture_output=True,text=True);need(tests.returncode==0,'unit tests:'+tests.stdout+tests.stderr)
 report={'status':'PASS_STEP4_ARTIFACT_AND_COVERAGE_AUDIT','source_files_verified':len(binding['files']),'registered_cuts_verified':216,'models_verified':2,'primary_cuts_verified':18,'all_blocks_verified':24,'identity_families_per_cut':5,'negative_controls_verified':18,'unit_tests':tests.stdout.strip(),'validation_sha256':sha(ROOT/'vit_validation.json'),'cost_report_sha256':sha(ROOT/'cost_report.json'),'prompt_audit_requirements_verified':22}
 print(json.dumps(report,sort_keys=True,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
