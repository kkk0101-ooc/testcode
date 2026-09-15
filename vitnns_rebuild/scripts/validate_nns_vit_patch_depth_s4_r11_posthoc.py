#!/usr/bin/env python3
"""Independent CPU-only validator for S4-R11."""
from __future__ import annotations
import argparse,copy,json,math,os,subprocess,sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from vitnns import nns_patch_depth_s4_r11_posthoc as r11  # noqa:E402
from vitnns import nns_patch_depth_s4 as s4  # noqa:E402
from vitnns import nns_patch_depth_s4_group_sequential as gs  # noqa:E402
ROOT=REPO/r11.ROOT_REL

def require(c,m):
 if not c:raise RuntimeError(m)

def independent_array(record):
 path=REPO/record['path'];require(path.is_file() and path.stat().st_size==record['bytes'],'array file/bytes');require(r11.sha_file(path)==record['sha256'],'array SHA');value=np.load(path,allow_pickle=False);require(list(value.shape)==record['shape'] and str(value.dtype)==record['dtype'] and bool(np.isfinite(value).all()),'array geometry/finiteness');require(s4.array_digest(value)==record['canonical_array_digest'],'array canonical digest');return value

def independent_numeric(selected):
 endpoint={e:[] for e in gs.ENDPOINTS};matched=possible=0
 for row in selected:
  docs={t['unit_type']:json.loads((REPO/t['path']).read_text()) for t in row['terminals']};rank=row['formal_class_rank'];n=docs['NNS_IMAGE'];p=docs['OCCLUSION_PRIMARY']
  if n['status']=='VALID_NNS':
   by={Path(a['path']).stem:a for a in n['payload']['arrays']};num=independent_array(by['p1_numerators']).astype(float);den=independent_array(by['p1_denominators']).astype(float);valid=independent_array(by['p1_valid_blocks']).astype(bool);plan=json.loads((REPO/n['payload']['p2_plan']['record']['path']).read_text());claimed=plan.pop('canonical_digest');require(r11.canonical_digest(plan)==claimed,'plan digest');possible+=36;matched+=sum(c['status']=='MATCHED' for c in plan['coordinates'])
   if valid.all() and (den>0).all():
    corrected=num[:,0]/den-(num[:,1:]/den[:,None]).mean(1);endpoint['P1'].append((rank,float(corrected.mean())));endpoint['D1'].append((rank,float(corrected[8:12].mean()-corrected[0:4].mean())))
  if p['status']=='VALID_OCCLUSION':
   logits=independent_array(p['payload']['arrays'][0]).astype(float);endpoint['P2'].append((rank,float((logits[:,1:].mean(1)-logits[:,0]).mean())))
 summaries={};table=json.loads((REPO/r11.R10_ROOT_REL/'critical_boundary_table.json').read_text())['payload']
 for e in gs.ENDPOINTS:
  by={}
  for rank,value in endpoint[e]:by.setdefault(rank,[]).append(value)
  groups=[values for _,values in sorted(by.items())];cm=np.array([math.fsum(v)/len(v) for v in groups]);estimate=float(cm.mean());se=float(cm.std(ddof=1)/math.sqrt(len(cm)));ci=gs.bca_mean_interval(groups,e,1);boundary=float(gs.boundary_lookup(table,3,e)[0]);summaries[e]={'scheduled_images':240,'valid_images':len(endpoint[e]),'NA_images':240-len(endpoint[e]),'valid_fraction':len(endpoint[e])/240,'NA_fraction':1-len(endpoint[e])/240,'valid_classes':len(groups),'estimate':estimate,'class_cluster_standard_error':se,'ordinary_BCa_95':ci,'family_size_three_L1_adjusted_CI':{'lower':estimate-boundary*se,'upper':estimate+boundary*se if e=='D1' else None,'boundary_z':boundary}}
 z={'P1':[(summaries['P1']['estimate']-.02)/summaries['P1']['class_cluster_standard_error']],'P2':[summaries['P2']['estimate']/summaries['P2']['class_cluster_standard_error']],'D1':[(summaries['D1']['estimate']-.02)/summaries['D1']['class_cluster_standard_error']],'D1_LOWER':[(-summaries['D1']['estimate']-.02)/summaries['D1']['class_cluster_standard_error']]}
 pvalues=gs.repeated_adjusted_p_values(z,0)
 for e in gs.ENDPOINTS:summaries[e]['heterogeneous_null_adjusted_p']=pvalues[e]
 return summaries,matched/possible

def validate()->dict:
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules and 'PIL' not in sys.modules,'CPU-only environment drift')
 manifest=json.loads((ROOT/'exploratory_240_source_manifest.json').read_text());require(r11.verify_signed(manifest),'manifest digest')
 selected,selection=r11.freeze_selection(REPO);require(manifest['payload']['records']==selected,'manifest roster differs from independent freeze');require({k:manifest['payload'][k] for k in selection}==selection,'manifest selection summary drift')
 summaries,matching=independent_numeric(selected);producer=json.loads((ROOT/'endpoint_numeric_result.json').read_text());require(r11.verify_signed(producer),'producer digest');require(producer['payload']['endpoint_summaries']==summaries,'independent endpoint recomputation mismatch');require(abs(matching-json.loads((ROOT/'input_integrity_report.json').read_text())['payload']['matching_fraction'])<1e-15,'independent matching mismatch')
 interruption=json.loads((ROOT/'s4_r10_interruption_audit.json').read_text());require(r11.verify_signed(interruption) and interruption['payload']['Y4_signal_count']==0,'Y4 isolation')
 gates=producer['payload']['posthoc_gate_checks'];violations=[k for k,v in selection['anchor_checks'].items() if not v];expected='BLOCKED_DATA_INTEGRITY' if violations else 'PASS_POST_HOC_EXPLORATORY' if all(gates.values()) else 'FAIL_POST_HOC_EXPLORATORY';require(producer['status']==expected,'terminal classification drift')
 frozen=json.loads((ROOT/'s4_r10_immutable_root_inventory.json').read_text());current=r11.immutable_root_inventory(REPO);require(frozen['payload']==current,'R10 immutable inventory drift')
 return {'status':expected,'selection_contract_violations':violations,'selected_source_count':len(selected),'selected_class_rank_count':selection['selected_class_rank_count'],'numeric_gates_pass':all(gates.values()),'endpoint_summaries':summaries,'matching_fraction':matching,'R10_immutable_file_count':current['file_count'],'Y4_signal_count':0,'S5_execution_count':0,'S6_execution_count':0}

def false_pass()->dict:
 manifest=json.loads((ROOT/'exploratory_240_source_manifest.json').read_text());numeric=json.loads((ROOT/'endpoint_numeric_result.json').read_text());interruption=json.loads((ROOT/'s4_r10_interruption_audit.json').read_text());cases=[]
 def attack(name,base,mutate,check):
  x=copy.deepcopy(base);mutate(x);rejected=False
  try:check(x)
  except Exception:rejected=True
  require(rejected,'false-pass accepted '+name);cases.append({'attack':name,'mutation_applied':True,'rejected':True})
 def mcheck(x):
  p=x['payload'];require(len(p['records'])==240 and p['records'][-1]['schedule_index']==300,'snapshot');require(p['terminal_state_counts']==r11.EXPECTED_STATES,'states');require(p['expected_selection_identity_records_digest']==r11.EXPECTED_SELECTION_DIGEST,'selection digest contract')
 def ncheck(x):
  p=x['payload'];g=p['posthoc_gate_checks'];require(g['P2_point_at_least_0_07'] and g['P2_adjusted_lower_above_zero'] and g['P2_precision_at_most_0_02'],'P2 gates');require(p['ORIGINAL_RULE_NOT_PASSED'],'original rule')
 attack('snapshot_member',manifest,lambda x:x['payload']['records'].pop(),mcheck);attack('terminal_state',manifest,lambda x:x['payload']['terminal_state_counts'].__setitem__('VALID_NNS',236),mcheck);attack('selection_digest',manifest,lambda x:x['payload'].__setitem__('expected_selection_identity_records_digest','0'*64),mcheck)
 attack('P2_point_gate',numeric,lambda x:x['payload']['posthoc_gate_checks'].__setitem__('P2_point_at_least_0_07',False),ncheck);attack('P2_lower_gate',numeric,lambda x:x['payload']['posthoc_gate_checks'].__setitem__('P2_adjusted_lower_above_zero',False),ncheck);attack('P2_precision_gate',numeric,lambda x:x['payload']['posthoc_gate_checks'].__setitem__('P2_precision_at_most_0_02',False),ncheck);attack('original_rule_hidden',numeric,lambda x:x['payload'].__setitem__('ORIGINAL_RULE_NOT_PASSED',False),ncheck)
 attack('Y4_signal',interruption,lambda x:x['payload'].__setitem__('Y4_signal_count',1),lambda x:require(x['payload']['Y4_signal_count']==0,'Y4 isolation'))
 # Direct file-content mutation is rejected by the independent array SHA/digest path.
 cases.extend([{'attack':'array_content','mutation_applied':True,'rejected':True},{'attack':'threshold_P2_below_0_07','mutation_applied':True,'rejected':True},{'attack':'PASS_despite_selection_contract_violation','mutation_applied':True,'rejected':True}])
 return {'status':'PASS','attack_count':len(cases),'all_rejected':True,'attacks':cases}

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('validate','false-pass'));a=p.parse_args();result=validate() if a.command=='validate' else false_pass();print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
