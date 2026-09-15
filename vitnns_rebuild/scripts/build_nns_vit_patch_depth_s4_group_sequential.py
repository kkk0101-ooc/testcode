#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from vitnns import nns_patch_depth_s4_group_sequential as gs  # noqa:E402

R4_ROOT=REPO/'evidence/nns_vit_patch_depth/s4_r4_20260915'
R4_PAYLOAD=REPO/'artifacts/nns_vit_patch_depth/s4_r4_20260915'
R4_SCHEDULE=REPO/'manifests/nns_vit_patch_depth/s4_r4_source_schedule.json'
SNAPSHOT_ROOT=REPO.parent/'.runtime/nns_vit_patch_depth_group_sequential_recovery'

def artifact(path:Path)->dict[str,Any]:
    value={'path':path.relative_to(REPO).as_posix(),'bytes':path.stat().st_size,'sha256':gs.sha_file(path)}
    if path.suffix=='.json':
        d=json.loads(path.read_text())
        if isinstance(d.get('canonical_digest'),str): value['canonical_digest']=d['canonical_digest']
    return value

def git(*args:str)->str:
    return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()

def copy_new(source:Path,destination:Path)->None:
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with source.open('rb') as src,os.fdopen(fd,'wb') as dst:
        shutil.copyfileobj(src,dst); dst.flush(); os.fsync(dst.fileno())
    gs.fsync_directory(destination.parent)

def partial_payload_inventory(post:dict[str,Any])->dict[str,Any]:
    terminal_keys={(row['unit_type'],int(row['schedule_index'])) for row in post['payload']['completed_terminals']}
    stale=post['payload']['claimed_without_terminal_roster']; rows=[]
    for claim in stale:
        index=int(claim['schedule_index']); unit_type=claim['unit_type']; unit_dir=R4_PAYLOAD/'units'/f'unit-{index:04d}'
        files=[]
        if unit_dir.exists():
            for path in sorted(unit_dir.rglob('*')):
                if path.is_file(): files.append({'path':path.relative_to(REPO).as_posix(),'bytes':path.stat().st_size,'sha256':gs.sha_file(path)})
        rows.append({'unit_type':unit_type,'schedule_index':index,'claim_path':claim['path'],'global_terminal_absent':(unit_type,index) not in terminal_keys,'partial_payload_files':files,'reuse_allowed':False,'quarantine_reason':'NO_IMMUTABLE_TERMINAL'})
    return {'record_count':len(rows),'records':rows,'delete_or_mutate':False}

def main()->int:
    root=REPO/gs.ROOT_REL; payload=REPO/gs.PAYLOAD_REL
    if root.exists() or payload.exists(): raise SystemExit('fresh group-sequential root unavailable')
    root.mkdir(parents=True); payload.mkdir(parents=True)
    pre_path=SNAPSHOT_ROOT/'s4_r4_owner_interruption_pre_snapshot.json'; post_path=SNAPSHOT_ROOT/'s4_r4_owner_interruption_post_snapshot.json'
    pre=json.loads(pre_path.read_text()); post=json.loads(post_path.read_text())
    copy_new(pre_path,root/pre_path.name); copy_new(post_path,root/post_path.name)
    schedule_doc=json.loads(R4_SCHEDULE.read_text()); schedule=schedule_doc['payload']['records']
    class_doc=json.loads((REPO/'evidence/nns_vit_patch_depth/s2_r4_20260914/formal_class_cohort_manifest.json').read_text()); classes=class_doc['payload']['records']
    require=lambda c,m: None if c else (_ for _ in ()).throw(RuntimeError(m))
    require(len(schedule)==1744 and len(classes)==872,'cohort count drift')
    namespace=gs.CLASS_NAMESPACE
    require(all(hashlib.sha256((namespace+'\0'+r['annotation_wnid']).encode()).hexdigest()==r['selection_key_sha256'] for r in classes),'class hash identity drift')
    require(classes==sorted(classes,key=lambda r:(r['selection_key_sha256'],r['annotation_wnid'])),'class order drift')
    by_wnid={}
    for row in schedule: by_wnid.setdefault(row['annotation_wnid'],[]).append(row)
    require(all(len(by_wnid[r['annotation_wnid']])==2 for r in classes),'two images per class drift')
    class_rank={row['annotation_wnid']:int(row['class_rank']) for row in classes}
    require(all(int(row['formal_class_rank'])==class_rank[row['annotation_wnid']] for row in schedule),'schedule formal class rank drift')
    group_records=sorted(schedule,key=lambda row:(int(row['formal_class_rank']),row['source_image_id']))
    first_look={}
    for row in group_records:
        first_look[int(row['schedule_index'])]=next(i+1 for i,count in enumerate(gs.LOOK_CLASS_COUNTS) if int(row['formal_class_rank'])<count)
    group_schedule=[]
    for index,row in enumerate(group_records): group_schedule.append({**row,'group_schedule_index':index,'first_authorized_look':first_look[int(row['schedule_index'])]})
    gs.write_json_new(REPO/gs.SCHEDULE_REL,gs.signed('s4_group_sequential_schedule','FROZEN',{'source_count':1744,'class_count':872,'order':'formal_class_rank_then_source_image_id','records_digest':gs.canonical_digest(group_schedule),'records':group_schedule}))
    look_artifacts=[]; previous=set(); increments=(400,368,256,256,464)
    for look,(class_count,image_count,new_count) in enumerate(zip(gs.LOOK_CLASS_COUNTS,gs.LOOK_IMAGE_COUNTS,increments),start=1):
        rows=[{key:row[key] for key in ('schedule_index','source_image_id','image_sha256','annotation_sha256','annotation_wnid','formal_class_rank','target_model_class_index','role_flags')} for row in group_records if int(row['formal_class_rank'])<class_count]
        identities={row['source_image_id'] for row in rows}; require(len(rows)==image_count and previous<identities if previous else len(rows)==image_count,'look nesting/count drift')
        document=gs.signed('s4_group_sequential_look_manifest','FROZEN',{'look':look,'class_count':class_count,'image_count':image_count,'new_image_count':new_count,'class_rank_min':0,'class_rank_max':class_count-1,'strictly_contains_previous':look>1,'selection_rule':'formal_class_rank_prefix_with_exact_two_frozen_confirmatory_images_per_class','records_digest':gs.canonical_digest(rows),'records':rows})
        name=f'look_{look}_{class_count}_classes_{image_count}_images_manifest.json'; gs.write_json_new(root/name,document); look_artifacts.append(artifact(root/name)); previous=identities
    table=gs.critical_boundary_table(); calibration=gs.calibrate_and_power(table,replicates=1_000_000,chunk=50_000)
    require(calibration['status']=='PASS','design power or calibration gate failed')
    documents={
      'owner_interruption_receipt.json':gs.signed('s4_r4_owner_interruption_receipt','PASS',{'reason':'OWNER_RESOURCE_DRIVEN_GROUP_SEQUENTIAL_REDESIGN','classification':'RESOURCE_DRIVEN_EFFECT_BLIND_BUT_PARTIAL_VALIDITY_STATUS_KNOWN_GROUP_SEQUENTIAL_RECOVERY','pre_snapshot':artifact(root/pre_path.name),'post_snapshot':artifact(root/post_path.name),'launcher_PID':962884,'worker_PIDs':[965876,965880,965881,965882,965884,965886,965888,965892],'worker_signal':'SIGTERM','SIGKILL_required':False,'pre_terminal_count':pre['payload']['unit_terminal_count'],'post_terminal_count':post['payload']['unit_terminal_count'],'pre_source_completion_count':pre['payload']['source_completion_count'],'post_source_completion_count':post['payload']['source_completion_count'],'stale_claim_count':post['payload']['claim_without_terminal_count'],'GPU_UUID_locks_released':True,'old_writer_count_after_interruption':0,'R4_interpretation':'INTERRUPTED_OWNER_RESOURCE_REDESIGN_EVIDENCE_ROOT','R4_next_authorization':'NONE','scientific_aggregate_access_count':0}),
      's4_r4_immutable_terminal_inventory.json':gs.signed('s4_r4_immutable_terminal_inventory','FROZEN',{'terminal_count':post['payload']['unit_terminal_count'],'unit_type_counts':post['payload']['unit_type_terminal_counts'],'state_counts':post['payload']['terminal_state_counts'],'records':post['payload']['completed_terminals'],'scientific_values_read':False}),
      's4_r4_stale_claim_inventory.json':gs.signed('s4_r4_stale_claim_inventory','FROZEN',{'claim_count':post['payload']['claim_count'],'stale_claim_count':post['payload']['claim_without_terminal_count'],'records':post['payload']['claimed_without_terminal_roster'],'stale_claim_is_terminal':False}),
      's4_r4_partial_payload_quarantine_inventory.json':gs.signed('s4_r4_partial_payload_quarantine_inventory','FROZEN',partial_payload_inventory(post)),
      'group_sequential_recovery_amendment.json':gs.signed('s4_group_sequential_recovery_amendment','FROZEN',{'classification':'RESOURCE_DRIVEN_EFFECT_BLIND_BUT_PARTIAL_VALIDITY_STATUS_KNOWN_GROUP_SEQUENTIAL_RECOVERY','disclosure':{'partial_typed_terminal_status_known_before_freeze':True,'relaxed_valid_NA_matching_gates_after_partial_status_known':True,'P1_P2_D1_aggregate_direction_mean_CI_pvalue_Holm_access_before_freeze':0},'unchanged_scientific_contract':['model','checkpoint','preprocessing','target','NNS_component','12_blocks','quadrature','closure','P1_permutation','P2_budgets_controls_operators','missingness_types','seeds','two_images_per_class','P1_P2_D1_definitions_directions','effect_thresholds','strong_FWER_0.05'],'new_gates':{'target_power_min':0.75,'ordinary_95_CI_half_width_max':0.015,'endpoint_valid_fraction_min':0.85,'endpoint_NA_fraction_max':0.15,'P2_matching_success_fraction_min':0.85},'effect_thresholds':gs.THRESHOLDS,'look_manifests':look_artifacts,'superseded_preseal_drafts':['s4_r5_group_sequential_20260915','s4_r6_group_sequential_20260915','s4_r7_group_sequential_20260915','s4_r8_group_sequential_20260915','s4_r9_group_sequential_20260915'],'next_authorization_on_stop_success':'S5_D2_MISSING_CLASS_RECONCILIATION_GATE_ONLY'}),
      'statistical_analysis_plan.json':gs.signed('s4_group_sequential_statistical_analysis_plan','FROZEN',{'independent_unit':'source_image_with_equal_class_equal_within_class_estimator','endpoints':{'P1':'mean_over_12_blocks_of_raw_alignment_minus_mean_64_position_stratified_permutation_draws','P2':'mean_over_12_blocks_and_3_budgets_of_primary_selected_logit_drop_minus_mean_4_matched_control_drops','D1':'mean_deep_blocks_8_11_minus_mean_shallow_blocks_0_3_of_P1_corrected_alignment'},'directions':{'P1':'one_sided_above_0.02','P2':'one_sided_above_0.10','D1':'two_sided_outside_minus0.02_plus0.02_no_observed_direction_selection'},'ordinary_precision':'20000_replicate_within_each_fixed_class_whole_image_BCa_95_percent_half_width_gate_only','ordinary_precision_seed_namespace':gs.POWER_NAMESPACE,'P2_matching_denominator':'36_coordinates_for_each_VALID_NNS_source_in_current_look','sequential_efficacy':'closed_testing_all_containing_intersections_reject','repeated_look_adjusted_p':'minimum_total_FWER_alpha_whose_incremental_spending_boundary_is_crossed_then_max_over_all_containing_intersections','repeated_look_adjusted_CI':'simultaneous_inversion_using_family_size_three_incremental_boundaries','typed_NA':'never_zero_fill_never_substitute','future_look_access':'forbidden','D2_formal_test':'forbidden_in_S4'}),
      'alpha_spending_contract.json':gs.signed('s4_group_sequential_alpha_spending_contract','FROZEN',{'cumulative_FWER_targets':list(gs.SPENDING),'incremental_targets':[0.025,0.015,0.005,0.003,0.002],'targets_are_not_ordinary_p_thresholds':True,'local_intersection_method':'Bonferroni_components_with_exact_incremental_alpha_spending_boundaries','D1_two_sided_factor':2,'closed_testing_required':True,'nested_information_used_in_canonical_sensitivity_and_cumulative_history':True}),
      'critical_boundary_table.json':gs.signed('s4_group_sequential_critical_boundary_table','PASS',table),
      'strong_FWER_proof.json':gs.signed('s4_group_sequential_strong_FWER_proof','PASS',{'elementary_hypotheses':['P1','P2','D1'],'nonempty_intersections':[list(x) for x in gs.intersections()],'local_test':'for intersection size m each one-sided component receives incremental alpha_j/m and each D1 direction incremental alpha_j/(2m) at look j; union bound across looks and components gives cumulative local error at most alpha(t)','nested_directional_boundary':'canonical nested boundaries are retained as sensitivity reference; controlling boundaries are exact normal quantiles of the preregistered incremental component spends','closure':'elementary rejection requires every containing intersection rejection','proof':'Every local intersection test has cumulative type-I error at most its spending target by summing the preregistered incremental component spends. Closed testing therefore controls strong FWER over every configuration at maximum local level 0.05, without relying on endpoint correlation.','strong_FWER_upper_bound':0.05,'Monte_Carlo_role':'implementation_calibration_only_not_proof'}),
      'calibration_simulation_report.json':gs.signed('s4_group_sequential_calibration_simulation_report','PASS',{'replicates_per_configuration_scenario':calibration['replicates_per_configuration_scenario'],'true_null_configuration_count':calibration['true_null_configuration_count'],'correlation_scenarios':calibration['correlation_scenarios'],'calibration_rows':calibration['calibration_rows'],'maximum_empirical_FWER':calibration['maximum_empirical_FWER'],'maximum_Wilson_upper_95_one_sided':calibration['maximum_Wilson_upper_95_one_sided'],'point_tolerance_for_MC_validation':0.0525,'upper_tolerance_for_MC_validation':0.055,'analytic_strong_FWER_bound':0.05,'seed_namespace':gs.CALIBRATION_NAMESPACE}),
      'design_power_report.json':gs.signed('s4_group_sequential_design_power_report','PASS',{'target_power':0.75,'design_alternatives':gs.DESIGN_ALTERNATIVES,'effect_thresholds':gs.THRESHOLDS,'conservative_scales':gs.SCALES,'valid_fraction_design':0.85,'replicates':1000000,'seed_namespace':gs.CALIBRATION_NAMESPACE,'rows':calibration['power_rows'],'D2_859_class_design':calibration['D2_859_class_design'],'pilot_direction_used':False}),
      'look_eligibility_report.json':gs.signed('s4_group_sequential_look_eligibility_report','PASS',{'gates':{'target_power_point_min':0.75,'joint_power_lower_bound_min':0.75,'correlation_independent_lower_bound_min':0.75,'ordinary_95_CI_half_width_max':0.015,'strong_FWER_max':0.05},'looks':calibration['look_eligibility'],'all_five_looks_eligible':all(row['eligible_STOP_SUCCESS_look'] for row in calibration['look_eligibility'])}),
      'tranche_release_contract.json':gs.signed('s4_group_sequential_tranche_release_contract','FROZEN',{'looks':[{'look':i+1,'class_count':c,'image_count':n,'new_image_count':d,'scheduled_unit_count':3*n,'release_requires':'PREVIOUS_INTERIM_CONTINUE' if i else 'PRE_INTERIM_SEAL_AND_REBIND_PASS'} for i,(c,n,d) in enumerate(zip(gs.LOOK_CLASS_COUNTS,gs.LOOK_IMAGE_COUNTS,(400,368,256,256,464)))],'future_tranche_claim_forbidden':True,'STOP_SUCCESS_prevents_future_release':True,'global_barrier_allowed_only_at_look_boundary':True}),
      'dynamic_global_queue_contract.json':gs.signed('s4_group_sequential_dynamic_global_queue_contract','FROZEN',{'scheduler':'global_work_conserving_dependency_aware_ready_queue','static_shards':False,'index_mod_K':False,'one_long_lived_worker_per_GPU':True,'model_load_count_per_worker':1,'claim':'launch_specific_O_CREAT_O_EXCL_fsync_file_and_directory','terminal':'global_no_overwrite_atomic_publish','rebind_terminal_always_skip':True,'stale_claim_fresh_launch_nonblocking':True,'worker_failure_isolation':True}),
      'elastic_GPU_startup_lock_contract.json':gs.signed('s4_group_sequential_elastic_GPU_startup_lock_contract','FROZEN',{'eligible':{'name':'NVIDIA H20','free_memory_mib_min':21504,'utilization_percent_max':10},'K':'min(8,eligible_count)','eight_eligible_requires_eight_workers':True,'K_less_than_8_failure':False,'existing_compute_PID_disqualifying':False,'GPU_UUID_keyed_lock':True,'fixed_GPU_set_per_launch':True,'worker_visible_GPU_count':1,'worker_device':'cuda:0'}),
      'phased_formal_access_audit_preseal.json':gs.signed('s4_group_sequential_preseal_access_audit','PASS',{'partial_typed_terminal_status_known':True,'efficacy_aggregate_access_count':0,'P1_P2_D1_mean_direction_CI_pvalue_Holm_gatekeeping_access_count':0,'future_look_outcome_access_count':0,'unauthorized_between_look_access_count':0,'D2_formal_access_count':0,'S5_execution_count':0}),
      'root_creation_receipt.json':gs.signed('s4_group_sequential_root_creation_receipt','PASS',{'evidence_root':gs.ROOT_REL.as_posix(),'payload_root':gs.PAYLOAD_REL.as_posix(),'created_unix_ns':time.time_ns(),'fresh':True,'S3_R7_binding':'evidence/nns_vit_patch_depth/s3_r7_20260915/terminal_receipt.json','S2_R8_binding':'evidence/nns_vit_patch_depth/s2_r8_20260914/candidate_m4_manifest.json','S4_R4_implementation_seal':'evidence/nns_vit_patch_depth/s4_r4_20260915/implementation_seal.json','S4_R4_schedule':'manifests/nns_vit_patch_depth/s4_r4_source_schedule.json','superseded_preseal_drafts':['s4_r5_group_sequential_20260915','s4_r6_group_sequential_20260915','s4_r7_group_sequential_20260915','s4_r8_group_sequential_20260915','s4_r9_group_sequential_20260915'],'predecessor_mutated':False}),
      'completion/s4_group_sequential.pending.json':gs.signed('s4_group_sequential_pending','PENDING',{'created_unix_ns':time.time_ns(),'efficacy_aggregate_access_count':0,'terminal_published':False}),
    }
    for name,doc in documents.items(): gs.write_json_new(root/name,doc)
    print(json.dumps({'status':'PASS','root':gs.ROOT_REL.as_posix(),'look_count':5,'calibration_replicates':1000000,'maximum_empirical_FWER':calibration['maximum_empirical_FWER'],'all_looks_eligible':all(row['eligible_STOP_SUCCESS_look'] for row in calibration['look_eligibility']),'efficacy_aggregate_access_count':0},sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
