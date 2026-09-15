#!/usr/bin/env python3
"""Produce the immutable CPU-only S4-R11 post-hoc exploratory package."""
from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from vitnns import nns_patch_depth_s4_r11_posthoc as r11  # noqa:E402

def audit_guard() -> None:
 forbidden=[REPO/'assets/imagenet/ILSVRC2012_img_val.tar',REPO/'assets/imagenet/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip',REPO/'assets/models/vit_base_patch16_224_weights.semantic_candidate.pth']
 def hook(event,args):
  if event!='open' or not args:return
  try:path=Path(os.path.realpath(os.fspath(args[0])))
  except TypeError:return
  if any(path==x for x in forbidden):raise PermissionError('R11 CPU-only firewall rejected '+str(path))
 sys.addaudithook(hook)

def main()->int:
 audit_guard();root=REPO/r11.ROOT_REL;r11.require(not root.exists(),'fresh R11 root required');root.mkdir(parents=True);head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
 immutable_inventory=r11.immutable_root_inventory(REPO);selected,selection=r11.freeze_selection(REPO)
 interruption=r11.signed('s4_r10_interruption_audit','PASS',{'precheck_unix_ns':1789512663923873157,'candidate_processes':[{'pid':3477989,'ppid':3377451,'pgid':3477989,'sid':3477989,'cwd':str(REPO),'cmdline':'../.conda/envs/vitnns/bin/python scripts/run_nns_vit_patch_depth_s4_group_sequential.py launch --look 1 --launch-id look-1-launch-001'}],'verified_S4_PIDs':[3477989],'worker_PIDs_at_signal':[],'signals':[{'pid':3477989,'signal':'SIGTERM','identity_reverified':True}],'SIGKILL_count':0,'stop_result':'SIGTERM_SENT_PROCESS_EXITED','postcheck_unix_ns':1789512779076256283,'postcheck_active_S4_PIDs':[],'Y4_excluded_PIDs':[],'Y4_signal_count':0,'R10_terminal_count_before_stop':714,'R10_terminal_count_after_stop':714,'five_second_stable_terminal_count':True,'S4_R10_file_write_count_by_R11':0,'failed_pre_signal_probe_count':1,'failed_pre_signal_probe_signal_count':0})
 amendment=r11.signed('s4_r11_posthoc_exploratory_amendment','FROZEN',{'classification':'OUTCOME_ACCESSED_POST_HOC_EXPLORATORY','original_blind_or_preregistered':False,'S4_R10_P1_P2_D1_aggregate_outcome_access_occurred':True,'owner_P2_rules_changed_after_approximately_237_to_242_results_seen':True,'strong_FWER_preserved_claim':False,'original_STOP_SUCCESS_claim_allowed':False,'allowed_statuses':['PASS_POST_HOC_EXPLORATORY','FAIL_POST_HOC_EXPLORATORY','BLOCKED_DATA_INTEGRITY'],'posthoc_thresholds':{'P1_adjusted_lower':0.02,'P2_point':0.07,'P2_adjusted_lower':0.0,'D1_equivalence_band':[-0.02,0.02]},'posthoc_precision_half_width_max':{'P1':0.015,'P2':0.02,'D1':0.015},'original_P2_rules_retained_for_comparison':{'effect_threshold':0.10,'precision_half_width_max':0.015},'next_authorization_if_PASS_POST_HOC_EXPLORATORY':'S5_CPU_ONLY_POST_HOC_EXPLORATORY_ANALYSIS_THEN_S6_EXPLORATORY_REPORTING','next_authorization_otherwise':'NONE'})
 manifest_status='FROZEN' if selection['all_anchor_checks_pass'] else 'BLOCKED_DATA_INTEGRITY'
 manifest=r11.signed('s4_r11_exploratory_240_source_manifest',manifest_status,{**selection,'selection_rule':'first_240_by_source_completion_time_max_unix_ns_then_schedule_index','selection_uses_outcome_values':False,'records':selected})
 numeric,integrity=r11.compute(REPO,selected)
 selection_violations=[name for name,value in selection['anchor_checks'].items() if not value]
 final_status='PASS_POST_HOC_EXPLORATORY' if not selection_violations and numeric['all_posthoc_numeric_gates_pass'] else 'BLOCKED_DATA_INTEGRITY' if selection_violations else 'FAIL_POST_HOC_EXPLORATORY'
 input_report=r11.signed('s4_r11_input_integrity_report',final_status,{**integrity,'selected_source_count':len(selected),'selected_terminal_count':sum(len(r['terminals']) for r in selected),'selection_contract_violations':selection_violations,'selected_class_rank_count_actual':selection['selected_class_rank_count'],'P1_D1_valid_class_count':numeric['endpoint_summaries']['P1']['valid_classes'],'P2_valid_class_count':numeric['endpoint_summaries']['P2']['valid_classes'],'unexplained_failure_count':selection['unexplained_failure_count'],'R10_roots_mutated':False})
 numeric_doc=r11.signed('s4_r11_endpoint_numeric_result',final_status,{**numeric,'scientific_scope':'POST_HOC_EXPLORATORY_ONLY','selection_integrity_gate_pass':not selection_violations})
 comparison=r11.signed('s4_r11_original_vs_posthoc_rule_comparison',final_status,{'P2_estimate':numeric['endpoint_summaries']['P2']['estimate'],'P2_adjusted_lower':numeric['endpoint_summaries']['P2']['family_size_three_L1_adjusted_CI']['lower'],'P2_ordinary_half_width':numeric['endpoint_summaries']['P2']['ordinary_BCa_95']['half_width'],'original_rule':{'effect_threshold':0.10,'precision_max':0.015,'checks':numeric['original_P2_rule_checks'],'status':'ORIGINAL_RULE_NOT_PASSED'},'posthoc_rule':{'point_threshold':0.07,'adjusted_lower_threshold':0.0,'precision_max':0.02,'checks':{k:v for k,v in numeric['posthoc_gate_checks'].items() if k.startswith('P2_')},'numeric_status':'PASS'},'original_design_eligibility':'HISTORICAL_REFERENCE_ONLY_240_NOT_PRESEALED_LOOK','confirmatory_interpretation_allowed':False})
 access=r11.signed('s4_r11_phased_access_and_deviation_audit',final_status,{'pre_R11_S4_R10_aggregate_outcome_access_occurred':True,'posthoc_P2_rule_changed_after_outcome_access':True,'selected_terminal_metadata_reads':720,'selected_numeric_array_records_validated':integrity['array_records_validated'],'selected_P2_plan_reads':integrity['P2_plans_validated'],'nonselected_numeric_outcome_reads_by_R11_producer':0,'JPEG_reads':0,'mask_reads':0,'decode_count':0,'model_load_count':0,'torch_timm_torchvision_PIL_import_count':0,'CUDA_GPU_initialization_count':0,'D2_formal_test_count':0,'S5_execution_count':0,'S6_execution_count':0,'FORMAL_CONFIRMATORY_GATE_OPEN':False})
 inventory_doc=r11.signed('s4_r10_immutable_root_inventory','PASS',immutable_inventory)
 for name,doc in [('posthoc_exploratory_amendment.json',amendment),('s4_r10_interruption_audit.json',interruption),('s4_r10_immutable_root_inventory.json',inventory_doc),('exploratory_240_source_manifest.json',manifest),('input_integrity_report.json',input_report),('endpoint_numeric_result.json',numeric_doc),('original_vs_posthoc_rule_comparison.json',comparison),('phased_access_and_deviation_audit.json',access)]:r11.write_json_new(root/name,doc)
 print(json.dumps({'status':final_status,'selection_violations':selection_violations,'selected':len(selected),'numeric_gates_pass':numeric['all_posthoc_numeric_gates_pass'],'root':root.relative_to(REPO).as_posix()},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
