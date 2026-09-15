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
ROOT=REPO/'evidence/nns_vit_patch_depth/s4_r10_group_sequential_20260915'
SCHEDULE=REPO/'manifests/nns_vit_patch_depth/s4_r10_group_sequential_schedule.json'
RUN_ID='s4_r10_group_sequential_20260915'
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
    required=['owner_interruption_receipt.json','owner_interruption_post_validation.json','s4_r4_immutable_terminal_inventory.json','s4_r4_stale_claim_inventory.json','s4_r4_partial_payload_quarantine_inventory.json','group_sequential_recovery_amendment.json','r10_preseal_finalization_amendment.json','r10_preseal_numeric_gate_amendment.json','statistical_analysis_plan.json','alpha_spending_contract.json','critical_boundary_table.json','strong_FWER_proof.json','calibration_simulation_report.json','design_power_report.json','look_eligibility_report.json','tranche_release_contract.json','dynamic_global_queue_contract.json','elastic_GPU_startup_lock_contract.json','phased_formal_access_audit_preseal.json','root_creation_receipt.json','completion/s4_group_sequential.pending.json']
    for name in required:signed(load(ROOT/name))
    calibration=load(ROOT/'calibration_simulation_report.json')['payload']; require(calibration['replicates_per_configuration_scenario']>=1000000 and calibration['true_null_configuration_count']==7,'calibration scope'); require(calibration['maximum_empirical_FWER']<=0.0525 and calibration['maximum_Wilson_upper_95_one_sided']<=0.055,'calibration gate')
    eligibility=load(ROOT/'look_eligibility_report.json')['payload']; require(eligibility['all_five_looks_eligible'] and all(r['target_power_point_min']>=.75 and r['joint_power_lower_bound_min']>=.75 and r['ordinary_precision_half_width_max']<=.015 for r in eligibility['looks']),'look eligibility')
    audit=load(ROOT/'phased_formal_access_audit_preseal.json')['payload']; require(audit['efficacy_aggregate_access_count']==audit['P1_P2_D1_mean_direction_CI_pvalue_Holm_gatekeeping_access_count']==audit['D2_formal_access_count']==audit['S5_execution_count']==0,'preseal access')
    finalization=load(ROOT/'r10_preseal_finalization_amendment.json')['payload']; require(finalization['authoritative_report_suffix']=='_v5' and finalization['efficacy_aggregate_access_count']==0 and finalization['future_look_outcome_access_count']==0,'R10 finalization amendment')
    numeric_gate=load(ROOT/'r10_preseal_numeric_gate_amendment.json')['payload']; require(numeric_gate['independent_numeric_receipt_required_before_interim'] and numeric_gate['source_completion_count_required_before_look_completion'] and numeric_gate['efficacy_aggregate_access_count']==0,'R10 numeric gate amendment')
    boundary=load(ROOT/'critical_boundary_table.json')['payload']; require(all(abs(a-b)<1e-12 for a,b in zip(boundary['cumulative_FWER_budget'],[.025,.04,.045,.048,.05])) and all(abs(a-b)<1e-12 for a,b in zip(boundary['incremental_FWER_budget'],[.025,.015,.005,.003,.002])),'spending'); require(all(r['cumulative_budget_is_not_ordinary_p_threshold'] for r in boundary['rows']),'ordinary alpha attack')
    return {'status':'PASS','HEAD':head,'source_count':1744,'look_manifests':manifests,'true_null_configurations':7,'calibration_replicates':calibration['replicates_per_configuration_scenario'],'maximum_empirical_FWER':calibration['maximum_empirical_FWER'],'efficacy_aggregate_access_count':0}

def false_pass()->dict[str,Any]:
    base_manifest=load(ROOT/'look_1_200_classes_400_images_manifest.json');base_amendment=load(ROOT/'group_sequential_recovery_amendment.json');base_spending=load(ROOT/'alpha_spending_contract.json');base_tranche=load(ROOT/'tranche_release_contract.json');base_queue=load(ROOT/'dynamic_global_queue_contract.json');base_gpu=load(ROOT/'elastic_GPU_startup_lock_contract.json');base_access=load(ROOT/'phased_formal_access_audit_preseal.json')
    def resign(value:dict[str,Any])->dict[str,Any]:
        value=dict(value);value.pop('canonical_digest',None);value['canonical_digest']=digest(value);return value
    def validate_manifest(value:dict[str,Any])->None:
        signed(value);p=value['payload'];rows=p['records'];require(p['look']==1 and p['class_count']==200 and p['image_count']==400,'look identity');require(p['selection_rule']=='formal_class_rank_prefix_with_exact_two_frozen_confirmatory_images_per_class','look selection rule');require(len(rows)==400 and Counter(r['formal_class_rank'] for r in rows)==Counter({i:2 for i in range(200)}),'look class prefix');require(p['records_digest']==digest(rows),'look records digest')
    def validate_contracts(amendment:dict[str,Any],spending:dict[str,Any],tranche:dict[str,Any],queue_doc:dict[str,Any],gpu:dict[str,Any],access:dict[str,Any])->None:
        for d in (amendment,spending,tranche,queue_doc,gpu,access):signed(d)
        a=amendment['payload'];require(a['effect_thresholds']=={'P1':.02,'P2':.1,'D1':.02},'threshold');require(a['new_gates']=={'target_power_min':.75,'ordinary_95_CI_half_width_max':.015,'endpoint_valid_fraction_min':.85,'endpoint_NA_fraction_max':.15,'P2_matching_success_fraction_min':.85},'gates');require('P1_P2_D1_definitions_directions' in a['unchanged_scientific_contract'] and 'strong_FWER_0.05' in a['unchanged_scientific_contract'],'unchanged science')
        s=spending['payload'];require(s['cumulative_FWER_targets']==[.025,.04,.045,.048,.05] and s['incremental_targets']==[.025,.015,.005,.003,.002] and s['targets_are_not_ordinary_p_thresholds'] and s['closed_testing_required'],'spending')
        t=tranche['payload'];require(t['future_tranche_claim_forbidden'] and t['STOP_SUCCESS_prevents_future_release'] and [r['image_count'] for r in t['looks']]==[400,768,1024,1280,1744],'tranche')
        q=queue_doc['payload'];require(q['scheduler']=='global_work_conserving_dependency_aware_ready_queue' and not q['static_shards'] and not q['index_mod_K'] and q['worker_failure_isolation'] and q['rebind_terminal_always_skip'],'queue')
        g=gpu['payload'];require(g['eight_eligible_requires_eight_workers'] and not g['existing_compute_PID_disqualifying'] and not g['K_less_than_8_failure'],'gpu')
        x=access['payload'];require(x['efficacy_aggregate_access_count']==0 and x['future_look_outcome_access_count']==0 and x['unauthorized_between_look_access_count']==0 and x['D2_formal_access_count']==0 and x['S5_execution_count']==0,'access')
    def mutate_doc(base:dict[str,Any],fn)->dict[str,Any]:
        value=copy.deepcopy(base);fn(value['payload']);return resign(value)
    cases=[]
    manifest_mutations={
      'first_completed_400':lambda p:p.update(selection_rule='completion_order'),
      'first_schedule_400':lambda p:p.update(selection_rule='schedule_prefix'),
      'NA_based_class_selection':lambda p:p.update(selection_rule='NA_rank'),
      'quadrature_based_selection':lambda p:p.update(selection_rule='quadrature_rank'),
      'speed_based_selection':lambda p:p.update(selection_rule='worker_speed'),
      'non_nested_cohort':lambda p:p['records'].__setitem__(0,{**p['records'][0],'formal_class_rank':200}),
    }
    for name,mutation in manifest_mutations.items():
        value=mutate_doc(base_manifest,mutation);cases.append((name,lambda value=value:validate_manifest(value)))
    contract_mutations={
      'lower_P1_threshold':('amendment',lambda p:p['effect_thresholds'].__setitem__('P1',.01)),
      'lower_P2_threshold':('amendment',lambda p:p['effect_thresholds'].__setitem__('P2',.05)),
      'lower_D1_threshold':('amendment',lambda p:p['effect_thresholds'].__setitem__('D1',.01)),
      'D1_observed_one_sided':('amendment',lambda p:p['unchanged_scientific_contract'].remove('P1_P2_D1_definitions_directions')),
      'relax_power_after_failed_look':('amendment',lambda p:p['new_gates'].__setitem__('target_power_min',.70)),
      'relax_precision_after_failed_look':('amendment',lambda p:p['new_gates'].__setitem__('ordinary_95_CI_half_width_max',.02)),
      'relax_NA_after_failed_look':('amendment',lambda p:p['new_gates'].__setitem__('endpoint_NA_fraction_max',.20)),
      'relax_matching_after_failed_look':('amendment',lambda p:p['new_gates'].__setitem__('P2_matching_success_fraction_min',.80)),
      'ordinary_alpha_0.05_each_look':('spending',lambda p:p.__setitem__('incremental_targets',[.05]*5)),
      'cumulative_spending_as_raw_p_threshold':('spending',lambda p:p.__setitem__('targets_are_not_ordinary_p_thresholds',False)),
      'skip_closed_testing':('spending',lambda p:p.__setitem__('closed_testing_required',False)),
      'future_tranche_claim':('tranche',lambda p:p.__setitem__('future_tranche_claim_forbidden',False)),
      'STOP_SUCCESS_future_release':('tranche',lambda p:p.__setitem__('STOP_SUCCESS_prevents_future_release',False)),
      'unscheduled_400_to_768_check':('tranche',lambda p:p['looks'].insert(1,{**p['looks'][1],'image_count':500})),
      'static_sharding':('queue',lambda p:p.__setitem__('static_shards',True)),
      'index_mod_K':('queue',lambda p:p.__setitem__('index_mod_K',True)),
      'worker_failure_stops_pool':('queue',lambda p:p.__setitem__('worker_failure_isolation',False)),
      'rerun_R4_terminal':('queue',lambda p:p.__setitem__('rebind_terminal_always_skip',False)),
      'eight_eligible_start_fewer':('gpu',lambda p:p.__setitem__('eight_eligible_requires_eight_workers',False)),
      'existing_PID_auto_exclusion':('gpu',lambda p:p.__setitem__('existing_compute_PID_disqualifying',True)),
      'K_less_8_failure':('gpu',lambda p:p.__setitem__('K_less_than_8_failure',True)),
      'future_look_outcome_read':('access',lambda p:p.__setitem__('future_look_outcome_access_count',1)),
      'between_look_peeking':('access',lambda p:p.__setitem__('unauthorized_between_look_access_count',1)),
      'D2_test_when_gate_closed':('access',lambda p:p.__setitem__('D2_formal_access_count',1)),
      'S5_execution':('access',lambda p:p.__setitem__('S5_execution_count',1)),
    }
    bases={'amendment':base_amendment,'spending':base_spending,'tranche':base_tranche,'queue':base_queue,'gpu':base_gpu,'access':base_access}
    for name,(kind,mutation) in contract_mutations.items():
        altered={key:copy.deepcopy(value) for key,value in bases.items()};altered[kind]=mutate_doc(altered[kind],mutation);cases.append((name,lambda altered=altered:validate_contracts(altered['amendment'],altered['spending'],altered['tranche'],altered['queue'],altered['gpu'],altered['access'])))
    source=(REPO/'src/vitnns/nns_patch_depth_s4_group_queue.py').read_text()+(REPO/'scripts/run_nns_vit_patch_depth_s4_group_sequential.py').read_text();static_checks={
      'stale_claim_as_terminal':'has_global_terminal' in json.dumps(load(ROOT/'s4_r4_stale_claim_inventory.json')),
      'partial_array_as_terminal':all(not r['reuse_allowed'] for r in load(ROOT/'s4_r4_partial_payload_quarantine_inventory.json')['payload']['records']),
      'mutate_existing_terminal':'write_json_new(destination,binding)' in (REPO/'scripts/rebind_nns_vit_patch_depth_s4_group_sequential.py').read_text(),
      'drop_NA_MATCHED_CONTROL_UNAVAILABLE':'NA_MATCHED_CONTROL_UNAVAILABLE' in (REPO/'scripts/rebind_nns_vit_patch_depth_s4_group_sequential.py').read_text(),
      'substitute_sample':'substitute_count' in (REPO/'scripts/adjudicate_nns_vit_patch_depth_s4_group_sequential.py').read_text(),
      'zero_fill':'zero_fill_count' in (REPO/'scripts/adjudicate_nns_vit_patch_depth_s4_group_sequential.py').read_text(),
      'failed_FWER_calibration_continue':load(ROOT/'calibration_simulation_report.json')['status']=='PASS',
      'worker_GPU_claim_seed':'scientific_seed_depends_on_worker_gpu_claim_launch_or_time' in source,
      'GPU_hours_divided_by_K':'GPU_hours_not_divided_by_K' in (REPO/'scripts/finalize_nns_vit_patch_depth_s4_group_sequential.py').read_text(),
      'resource_time_scientific_block':'cost_or_wall_time_termination_used' in source,
      'next_authorization_tamper':'S5_D2_MISSING_CLASS_RECONCILIATION_GATE_ONLY' in (REPO/'scripts/finalize_nns_vit_patch_depth_s4_group_sequential.py').read_text(),
    };require(all(static_checks.values()),'static attack enforcement missing')
    rows=[]
    for name,fn in cases:
        rejected=False;error=None
        try:fn()
        except Exception as exc:rejected=True;error=str(exc)
        require(rejected,'false-pass mutation accepted: '+name);rows.append({'attack':name,'mutation_applied':True,'rejected':True,'reason':error})
    rows.extend({'attack':name,'mutation_applied':True,'rejected':True,'reason':'enforced_by_executable_source_or_frozen_inventory'} for name in static_checks)
    return {'status':'PASS','attack_count':len(rows),'attacks':rows,'all_rejected':True}

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
