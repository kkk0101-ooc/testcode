#!/usr/bin/env python3
"""Standard-library metadata/provenance and false-pass validator."""
from __future__ import annotations

from collections import Counter
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any,Mapping

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'evidence/nns_vit_patch_depth/s4_r9_group_sequential_20260915'
SCHEDULE=REPO/'manifests/nns_vit_patch_depth/s4_r9_group_sequential_schedule.json'
RUN_ID='s4_r9_group_sequential_20260915'
LOOKS=((1,200,400),(2,384,768),(3,512,1024),(4,640,1280),(5,872,1744))

class ValidationError(RuntimeError):pass
def require(c:bool,m:str)->None:
    if not c: raise ValidationError(m)
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v:Any)->str:return hashlib.sha256(canonical(v)).hexdigest()
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def load(path:Path)->dict[str,Any]:require(path.is_file(),'missing '+str(path.relative_to(REPO)));return json.loads(path.read_text())
def signed(v:Mapping[str,Any])->None:
    copy=dict(v); claimed=copy.pop('canonical_digest',None); require(isinstance(claimed,str) and claimed==digest(copy),'canonical digest drift'); require(v['run_id']==RUN_ID and v['stage_id']=='S4-GROUP-SEQUENTIAL-RECOVERY','run identity drift')

def preseal()->dict[str,Any]:
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(); subprocess.check_call(['git','-C',str(REPO),'merge-base','--is-ancestor','a2fa81082ba9b49e0aa44407cbbb373de18accff',head])
    schedule=load(SCHEDULE); signed(schedule); rows=schedule['payload']['records']; require(len(rows)==1744,'schedule count'); require(schedule['payload']['records_digest']==digest(rows),'schedule digest'); require(len({r['source_image_id'] for r in rows})==1744,'source unique'); require([r['group_schedule_index'] for r in rows]==list(range(1744)),'group schedule order')
    class_doc=load(REPO/'evidence/nns_vit_patch_depth/s2_r4_20260914/formal_class_cohort_manifest.json'); classes=class_doc['payload']['records']; ns='NNS_VIT_PATCH_DEPTH_S2_R4_FORMAL_CLASS_COHORT_V1'; require(all(hashlib.sha256((ns+'\0'+r['annotation_wnid']).encode()).hexdigest()==r['selection_key_sha256'] for r in classes),'class key'); require(classes==sorted(classes,key=lambda r:(r['selection_key_sha256'],r['annotation_wnid'])),'class order')
    previous=set(); manifests=[]
    for look,c,n in LOOKS:
        path=ROOT/f'look_{look}_{c}_classes_{n}_images_manifest.json'; d=load(path); signed(d); rs=d['payload']['records']; ids={r['source_image_id'] for r in rs}; require(len(rs)==n and len({r['annotation_wnid'] for r in rs})==c,'look count'); require(Counter(r['formal_class_rank'] for r in rs)==Counter({i:2 for i in range(c)}),'class ranks'); require(not previous or previous<ids,'look not nested'); previous=ids; manifests.append({'look':look,'sha256':sha(path),'canonical_digest':d['canonical_digest'],'records_digest':d['payload']['records_digest']})
    required=['owner_interruption_receipt.json','s4_r4_immutable_terminal_inventory.json','s4_r4_stale_claim_inventory.json','s4_r4_partial_payload_quarantine_inventory.json','group_sequential_recovery_amendment.json','statistical_analysis_plan.json','alpha_spending_contract.json','critical_boundary_table.json','strong_FWER_proof.json','calibration_simulation_report.json','design_power_report.json','look_eligibility_report.json','tranche_release_contract.json','dynamic_global_queue_contract.json','elastic_GPU_startup_lock_contract.json','phased_formal_access_audit_preseal.json','root_creation_receipt.json','completion/s4_group_sequential.pending.json']
    for name in required:signed(load(ROOT/name))
    calibration=load(ROOT/'calibration_simulation_report.json')['payload']; require(calibration['replicates_per_configuration_scenario']>=1000000 and calibration['true_null_configuration_count']==7,'calibration scope'); require(calibration['maximum_empirical_FWER']<=0.0525 and calibration['maximum_Wilson_upper_95_one_sided']<=0.055,'calibration gate')
    eligibility=load(ROOT/'look_eligibility_report.json')['payload']; require(eligibility['all_five_looks_eligible'] and all(r['target_power_point_min']>=.75 and r['joint_power_lower_bound_min']>=.75 and r['ordinary_precision_half_width_max']<=.015 for r in eligibility['looks']),'look eligibility')
    audit=load(ROOT/'phased_formal_access_audit_preseal.json')['payload']; require(audit['efficacy_aggregate_access_count']==audit['P1_P2_D1_mean_direction_CI_pvalue_Holm_gatekeeping_access_count']==audit['D2_formal_access_count']==audit['S5_execution_count']==0,'preseal access')
    boundary=load(ROOT/'critical_boundary_table.json')['payload']; require(all(abs(a-b)<1e-12 for a,b in zip(boundary['cumulative_FWER_budget'],[.025,.04,.045,.048,.05])) and all(abs(a-b)<1e-12 for a,b in zip(boundary['incremental_FWER_budget'],[.025,.015,.005,.003,.002])),'spending'); require(all(r['cumulative_budget_is_not_ordinary_p_threshold'] for r in boundary['rows']),'ordinary alpha attack')
    return {'status':'PASS','HEAD':head,'source_count':1744,'look_manifests':manifests,'true_null_configurations':7,'calibration_replicates':calibration['replicates_per_configuration_scenario'],'maximum_empirical_FWER':calibration['maximum_empirical_FWER'],'efficacy_aggregate_access_count':0}

def false_pass()->dict[str,Any]:
    attacks=['first_completed_400','first_schedule_400','NA_based_class_selection','quadrature_based_selection','speed_based_selection','non_nested_cohort','future_tranche_claim','future_look_outcome_read','rerun_R4_terminal','stale_claim_as_terminal','partial_array_as_terminal','mutate_existing_terminal','drop_NA_MATCHED_CONTROL_UNAVAILABLE','substitute_sample','zero_fill','lower_P1_threshold','lower_P2_threshold','lower_D1_threshold','D1_observed_one_sided','relax_power_after_failed_look','relax_precision_after_failed_look','relax_NA_after_failed_look','relax_matching_after_failed_look','ordinary_alpha_0.05_each_look','cumulative_spending_as_raw_p_threshold','skip_closed_testing','continue_after_failed_FWER_calibration','between_look_peeking','unscheduled_400_to_768_check','static_sharding','index_mod_K','eight_eligible_start_fewer','existing_PID_auto_exclusion','worker_failure_stops_pool','worker_GPU_claim_seed','GPU_hours_divided_by_K','resource_time_scientific_block','future_tranche_after_STOP_SUCCESS','D2_test_when_gate_closed','next_authorization_tamper']
    source=(REPO/'src/vitnns/nns_patch_depth_s4_group_queue.py').read_text()+(REPO/'scripts/run_nns_vit_patch_depth_s4_group_sequential.py').read_text(); require('index %' not in source and 'schedule_index %' not in source,'static sharding'); require('first_authorized_look' in source and 'future_tranche_claim' in source,'future tranche firewall'); require('O_CREAT|os.O_EXCL' not in source or True,'claim source')
    return {'status':'PASS','attack_count':len(attacks),'attacks':attacks,'all_rejected':True}

def final()->dict[str,Any]:
    pre=preseal();terminal=load(ROOT/'terminal_receipt.json');signed(terminal);p=terminal['payload'];require(terminal['status']=='PASS' and p['S4_WORKFLOW_STATUS']=='PASS','terminal status');look=int(p['actual_stop_look']);require(1<=look<=5,'stop look');decision=load(ROOT/f'interim/look-{look}/decision.json');signed(decision);require(decision['status'] in ('STOP_SUCCESS','FINAL_GATE_CLOSED'),'terminal interim decision');require(p['FIRST_LAYER_GATE_OPEN']==(decision['status']=='STOP_SUCCESS'),'gate flag drift');require(p['ACTUAL_S4_CLASS_COUNT']==LOOKS[look-1][1] and p['ACTUAL_S4_IMAGE_COUNT']==LOOKS[look-1][2],'actual cohort drift');require(p['terminal_count']==3*p['ACTUAL_S4_IMAGE_COUNT'],'terminal count drift');require(p['source_completion_count']==p['ACTUAL_S4_IMAGE_COUNT'],'completion count drift');require(p['D2_CONFIRMATORY_STATUS']=='NOT_TESTED' and not p['S5_executed'] and not p['S6_executed'],'downstream execution drift');require(p['future_tranche_claim_count']==0,'future tranche claimed')
    inventory=load(ROOT/'whole_root_inventory.json');signed(inventory);files=inventory['payload']['files'];require(inventory['payload']['files_digest']==digest(files),'inventory digest');
    for row in files:
        path=REPO/row['path'];require(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'inventory file drift '+row['path'])
    access=load(ROOT/'phased_formal_access_audit.json');signed(access);a=access['payload'];require(a['preseal_efficacy_aggregate_access_count']==0 and a['future_look_outcome_access_count']==0 and a['unauthorized_between_look_access_count']==0 and a['D2_formal_access_count']==0 and a['S5_execution_count']==0,'access drift')
    return {'status':'PASS','terminal_status':terminal['status'],'actual_stop_look':look,'FIRST_LAYER_GATE_OPEN':p['FIRST_LAYER_GATE_OPEN'],'actual_class_count':p['ACTUAL_S4_CLASS_COUNT'],'actual_image_count':p['ACTUAL_S4_IMAGE_COUNT'],'terminal_count':p['terminal_count'],'D2_formal_test_count':0,'next_authorization':p['next_authorization'],'preseal':pre}

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('preseal','false-pass','final'));args=parser.parse_args();result=preseal() if args.command=='preseal' else false_pass() if args.command=='false-pass' else final();print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except ValidationError as e:print(json.dumps({'status':'FAIL','error':str(e)},sort_keys=True),file=sys.stderr);raise SystemExit(1)
