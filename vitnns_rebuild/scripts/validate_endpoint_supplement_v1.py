#!/usr/bin/env python3
"""Coverage, immutable source, numerical and paper checks for the actual request."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,gzip,hashlib,json,re,sqlite3,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/'reports/nns_vit_endpoint_supplement_v1_20260920';PAPER=REPO/'papers/nns_vit_endpoint/publication_r2_20260920'
checks=[]
def check(k,v,detail=''):checks.append(dict(name=k,passed=bool(v),detail=detail))
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--paper',action='store_true');a=ap.parse_args();freeze=load(ROOT/'freeze.json');tasks=load(ROOT/'task_manifest.json')['tasks'];sources=load(ROOT/'source_bindings.json')
 check('frozen_456_tasks',freeze['task_count']==len(tasks)==456)
 check('manifest_digest',hashlib.sha256(canonical(tasks)).hexdigest()==freeze['task_manifest_digest'])
 check('source_bindings_digest',sha(ROOT/'source_bindings.json')==freeze['source_bindings_sha256'])
 for rel,expected in sources.items():
  p=REPO/rel;check('source_sha:'+rel,p.is_file() and sha(p)==expected)
  if rel.endswith('artifact_manifest.json'):
   d=load(p)
   if 'canonical_digest' in d:
    unsigned=dict(d);exp=unsigned.pop('canonical_digest');check('source_canonical:'+rel,hashlib.sha256(canonical(unsigned)).hexdigest()==exp or hashlib.sha256(canonical(d['files'])).hexdigest()==exp)
 with sqlite3.connect(ROOT/'queue.sqlite3') as c:states=dict(c.execute('select state,count(*) from tasks group by state'))
 check('queue_complete',states=={'completed':456},str(states))
 counts={};unit_count=0;feature_count=0;target_images={};random_states={};result_seconds=0.
 for task in tasks:
  tid=task['task_id'];t=task['payload'];p=ROOT/'results/attempts'/tid/'attempt-01.json';check('task_exists:'+tid,p.exists())
  if not p.exists():continue
  e=load(p);u=dict(e);ex=u.pop('canonical_digest');check('task_canonical:'+tid,hashlib.sha256(canonical(u)).hexdigest()==ex);d=e['result'];counts[d['study']]=counts.get(d['study'],0)+1
  check('task_identity:'+tid,all(d[k]==t[k] for k in ('study','model_id','init_seed','sample_id','image_sha256','cohort_order','split_order')))
  check('worker_contract:'+tid,d['worker_identity']['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and d['worker_identity']['deterministic_algorithms'] and not d['worker_identity']['cuda_matmul_allow_tf32'] and d['worker_identity']['startup_matmul_backward']=='PASS')
  result_seconds+=d['seconds']
  for f in d['payload_files']:check('payload:'+f['path'],sha(REPO/f['path'])==f['sha256'])
  if d['study']=='random':
   random_states.setdefault((d['model_id'],d['init_seed']),set()).add(d['binding']['tensor_digest']);npz=np.load(ROOT/'payloads'/tid/'m3.npz');unit_count+=len(npz['block'])
   expected={(b,token,c) for b in range(12) for token in t['tokens'] for c in t['channels'][str(b)]};observed=set(zip(npz['block'].tolist(),npz['token'].tolist(),npz['channel'].tolist()))
   check('random_m3_units:'+tid,observed==expected and len(npz['block'])==3072)
   check('random_m3_finite:'+tid,all(np.isfinite(npz[k]).all() for k in ('patch_absolute_mass_196','patch_signed_sum_196','absolute_mass','l2_norm')))
   scans=[d['m4']];blocks=range(12);jvp=d['jvp'];check('8_jvp_directions:'+tid,all(r['directions']==8 and len(r['direction_records'])==8 for r in jvp))
   check('jvp_identity:'+tid,all(r['explicit_jvp_relative_error'] is None or r['explicit_jvp_relative_error']<=1e-4 for r in jvp))
   check('m3_scalar_parity:'+tid,all(c['batched_vjp_scalar_vjp_max_abs']<=2e-5+2e-4*c['scalar_endpoint_max_abs'] for c in d['m3_validation']['checks']))
  elif d['study']=='target':
   target_images.setdefault(d['model_id'],set()).add(d['sample_id']);check('target_scope:'+tid,set(d['targets'])=={'gt','top1','margin'} and d['correct']==(d['gt']==d['top1']) and d['runnerup']!=d['top1'])
   if d['correct']:check('correct_gt_top1_equal:'+tid,d['targets']['gt']==d['targets']['top1'])
   scans=list(d['targets'].values());blocks=(0,5,11)
  else:
   rows=json.loads(gzip.decompress((ROOT/'payloads'/tid/'feature.json.gz').read_bytes()));feature_count+=len(rows);check('feature_units:'+tid,{(r['block'],r['channel']) for r in rows}=={(b,c) for b in range(12) for c in t['channels'][str(b)]})
   for i,r in enumerate(rows):
    check(f'feature_distribution:{tid}:{i}',all(v is None or (len(v)==196 and np.isfinite(v).all() and np.min(v)>=0 and np.isclose(sum(v),1,atol=1e-9)) for v in [r[k] for k in ('p','q','q_wrong_channel','q_cross_layer','q_permuted','q_random_channel')]))
    check(f'feature_controls:{tid}:{i}',r['wrong_channel']!=r['channel'] and r['random_channel']!=r['channel'] and r['cross_layer']==(r['block']+1)%12 and sorted(r['q'])==sorted(r['q_permuted']))
   continue
  heads=12 if d['model_id']=='vit_b16' else 6
  for index,scan in enumerate(scans):
   check(f'all_heads:{tid}:{index}',{(r['block'],r['head']) for r in scan['attention']}=={(b,h) for b in blocks for h in range(heads)})
   check(f'both_residuals:{tid}:{index}',{(r['block'],r['residual_type']) for r in scan['residual']}=={(b,k) for b in blocks for k in ('attention','mlp')})
   check(f'closures:{tid}:{index}',all(r['qkv_native_closure']['pass'] and r['frozen_attention_identity']['pass'] for r in scan['attention']) and all(r['closure']['pass'] for r in scan['residual']) and all(r[k]['pass'] for r in scan['block_validation'] for k in ('head_sum_attention_closure','qk_routing_closure','routing_value_attention_closure')))
 check('study_counts',counts==dict(random=192,feature=64,target=200));check('six_distinct_initializations',len(random_states)==6 and all(len(v)==1 for v in random_states.values()) and len(set.union(*random_states.values()))==6)
 check('random_m3_count',unit_count==589824);check('feature_unit_count',feature_count==12288);check('target_100_per_model',all(len(v)==100 for v in target_images.values()) and len(target_images)==2)
 for name in ('baseline_analysis.json','m2_inherited.json','training_analysis.json','target_analysis.json','feature_analysis.json','sampling_analysis.json'):
  check('analysis_exists:'+name,(ROOT/name).exists())
 if all((ROOT/n).exists() for n in ('training_analysis.json','target_analysis.json','feature_analysis.json')):
  train=load(ROOT/'training_analysis.json');check('training_metrics',len(train['comparisons'])==102)
  check('hierarchical_three_seeds',all(r['n_seeds']==3 and r['n_images']==32 for r in train['comparisons'] if r['metric'] not in ('saturation_routing','saturation_error')))
  target=load(ROOT/'target_analysis.json');check('target_strata',set(r['subset'] for r in target['summary'])=={'all','correct','incorrect'})
  check('target_all_metrics',{'routing_scale','value_error','jvp_discrepancy','saturation_routing','saturation_error','attention_identity_scale','mlp_identity_scale','attention_output_identity_tv','mlp_output_identity_tv','target_contracted_jvp'}.issubset({r['metric'] for r in target['summary']}))
  feature=load(ROOT/'feature_analysis.json');check('four_negative_controls',set(r['control'] for r in feature['contrasts'])=={'wrong_channel','cross_layer','permuted','random_channel'})
  base=load(ROOT/'baseline_analysis.json');check('baseline_raw_reconstruction',base['source_tasks']==800 and base['curve_auc_max_error']<1e-12 and base['legacy_score_max_error']<1e-12 and base['jensen_min_difference']>=-1e-12)
  check('baseline_repeat_count',base['repeat_rows']==32000);check('m2_counts',load(ROOT/'m2_inherited.json')['taylor_greater_counts']==dict(full=11,isolated=12))
 if a.paper:
  for lang in ('zh','en'):
   text=(PAPER/lang/'manuscript.md').read_text();sup=(PAPER/lang/'supplement.md').read_text();new=(PAPER/lang/'supplement_experiments_v1.md').read_text()
   check('main_supplement_section:'+lang,'5.9' in text and 'supplement_experiments_v1.md' in text)
   check('old_estimator_removed:'+lang,'随机基线先平均 20 条 signed effect' not in text and 'random baseline averages 20 signed-effect' not in text)
   for section in ('S9.','S10.','S11.','S12.','S13.','S14.'):check('supplement_section:'+lang+section,section in new)
   for p in (PAPER/lang/'manuscript.md',PAPER/lang/'supplement.md',PAPER/lang/'supplement_experiments_v1.md'):
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
     if not target.startswith(('http','#')):check('paper_link:'+str(p)+target,(p.parent/target.split('#')[0]).is_file())
  zh=(PAPER/'zh/supplement_experiments_v1.md').read_text();en=(PAPER/'en/supplement_experiments_v1.md').read_text()
  check('supplement_equations_bilingual',re.findall(r'\$\$(.*?)\$\$',zh,re.S)==re.findall(r'\$\$(.*?)\$\$',en,re.S));check('supplement_tables_bilingual',[l for l in zh.splitlines() if l.startswith('|')]==[l for l in en.splitlines() if l.startswith('|')])
  portable=load(PAPER/'evidence/supplement_v1/manifest.json')
  check('portable_supplement_revision',portable['revision']=='ENDPOINT-SUPPLEMENT-V1')
  for f in portable['files']:check('portable_supplement_hash:'+f['path'],sha(PAPER/f['path'])==f['sha256'] and (PAPER/f['path']).stat().st_size==f['bytes'])
  check('portable_required_evidence',{'evidence/supplement_v1/native_verification.json','evidence/supplement_v1/independent_statistics_verification.json','evidence/supplement_v1/validation_report.json','evidence/supplement_v1/figure_manifest.json'}.issubset({f['path'] for f in portable['files']}))
 errors=[r for r in checks if not r['passed']];report=dict(status='PASS' if not errors else 'FAIL',check_count=len(checks),errors=errors,counts=counts,total_task_seconds=result_seconds,gpu_compute_hours=result_seconds/3600,checks=checks)
 report_path=ROOT/('paper_validation.json' if a.paper else 'validation_report.json')
 report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 if a.paper and not errors:
  native=load(ROOT/'native_verification.json');independent=load(ROOT/'independent_statistics_verification.json');tests=load(ROOT/'test_results.json')
  terminal=dict(status='ENDPOINT_SUPPLEMENT_V1_PASS_COMPLETE',scientific_status='SUPPLEMENTARY_CONTROLS_COMPLETE_DESCRIPTIVE_NOT_RETROACTIVE_CONFIRMATORY',task_count=456,random_initialization_tasks=192,target_robustness_tasks=200,feature_tasks=64,random_m3_targets=589824,feature_energy_targets=12288,paper_validation_checks=report['check_count'],native_recomputation_checks=native['check_count'],independent_statistics_checks=independent['check_count'],test_count=tests['passed'],gpu_compute_hours=report['gpu_compute_hours'],source_roots_immutable=True,optional_P2_P3='NOT_EXECUTED_MINIMUM_PACKAGE',next_authorization='STOP_AFTER_ENDPOINT_SUPPLEMENT_V1',evidence_sha256={name:sha(ROOT/name) for name in ('freeze.json','acquisition_receipt.json','validation_report.json','paper_validation.json','native_verification.json','independent_statistics_verification.json','test_results.json')})
  (ROOT/'terminal_receipt.json').write_text(json.dumps(terminal,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2));return bool(errors)
if __name__=='__main__':sys.exit(main())
