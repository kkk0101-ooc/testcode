#!/usr/bin/env python3
"""Publish pre-interim reports and seal the remotely published implementation."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/'evidence/nns_vit_patch_depth/s4_r9_group_sequential_20260915';sys.path.insert(0,str(REPO/'src'))
from vitnns import nns_patch_depth_s4_group_sequential as gs  # noqa:E402
def artifact(path:Path)->dict[str,Any]:
 v={'path':path.relative_to(REPO).as_posix(),'bytes':path.stat().st_size,'sha256':gs.sha_file(path)};d=json.loads(path.read_text()) if path.suffix=='.json' else {};
 if isinstance(d.get('canonical_digest'),str):v['canonical_digest']=d['canonical_digest']
 return v
def run(args:list[str],env=None)->str:return subprocess.check_output(args,cwd=REPO,text=True,env=env,stderr=subprocess.STDOUT).strip()
def preseal()->None:
 python=REPO.parent/'.conda/envs/vitnns/bin/python';env={**os.environ,'PYTHONPATH':str(REPO/'src')};test=run([str(python),'-m','pytest','-q','tests/test_nns_patch_depth_s4_group_sequential.py','--basetemp',str(REPO.parent/'.runtime/pytest-s4-group-sequential')],env)
 if '9 passed' not in test:raise RuntimeError('group-sequential test count/status drift: '+test)
 validation=json.loads(run(['python3','scripts/validate_nns_vit_patch_depth_s4_group_sequential.py','preseal']));attacks=json.loads(run(['python3','scripts/validate_nns_vit_patch_depth_s4_group_sequential.py','false-pass']))
 suffix='_final' if (ROOT/'pre_interim_validation_report.json').exists() else ''
 docs={'synthetic_group_sequential_validation_report.json':gs.signed('s4_group_sequential_synthetic_validation','PASS',{'pytest_output':test,'checks':['nested_class_rank_looks','future_tranche_firewall','rebind_terminal_skip','stale_claim_nonblocking','closed_testing_all_intersections','incremental_alpha_boundaries','adjusted_p_inversion','BCa_within_class_whole_image_precision','STOP_SUCCESS_blocks_future_release'],'efficacy_aggregate_access_count':0}),'pre_interim_validation_report.json':gs.signed('s4_group_sequential_pre_interim_validation','PASS',validation),'false_pass_mutation_audit.json':gs.signed('s4_group_sequential_false_pass_mutation_audit','PASS',attacks)}
 if suffix:docs={name[:-5]+suffix+'.json':doc for name,doc in docs.items()}
 for name,doc in docs.items():gs.write_json_new(ROOT/name,doc)
 print(json.dumps({'status':'PASS','pytest':test,'false_pass_attacks':attacks['attack_count'],'efficacy_aggregate_access_count':0},sort_keys=True))
def seal(commit:str)->None:
 head=run(['git','rev-parse','HEAD']);require=head==commit
 if not require:raise RuntimeError('HEAD mismatch')
 branch=run(['git','symbolic-ref','--short','HEAD']);remote=run(['git','ls-remote','--heads','origin',f'refs/heads/{branch}']).split()[0]
 if remote!=commit:raise RuntimeError('implementation not live on remote')
 required=['src/vitnns/nns_patch_depth_s4_group_sequential.py','src/vitnns/nns_patch_depth_s4_group_queue.py','scripts/build_nns_vit_patch_depth_s4_group_sequential.py','scripts/rebind_nns_vit_patch_depth_s4_group_sequential.py','scripts/run_nns_vit_patch_depth_s4_group_sequential.py','scripts/validate_nns_vit_patch_depth_s4_group_sequential.py','scripts/seal_nns_vit_patch_depth_s4_group_sequential.py','scripts/adjudicate_nns_vit_patch_depth_s4_group_sequential.py','scripts/finalize_nns_vit_patch_depth_s4_group_sequential.py','tests/test_nns_patch_depth_s4_group_sequential.py','config/nns_vit_patch_depth/s4_group_sequential_artifact.schema.json','config/nns_vit_patch_depth/s4_group_sequential_look_manifest.schema.json','config/nns_vit_patch_depth/s4_group_sequential_interim.schema.json','manifests/nns_vit_patch_depth/s4_r9_group_sequential_schedule.json'];tracked=run(['git','ls-tree','-r','--name-only',commit]).splitlines()
 if not all(x in tracked or 'vitnns_rebuild/'+x in tracked for x in required):raise RuntimeError('implementation tree incomplete')
 names=['owner_interruption_receipt.json','s4_r4_immutable_terminal_inventory.json','s4_r4_stale_claim_inventory.json','s4_r4_partial_payload_quarantine_inventory.json','group_sequential_recovery_amendment.json','statistical_analysis_plan.json','alpha_spending_contract.json','critical_boundary_table.json','strong_FWER_proof.json','calibration_simulation_report.json','design_power_report.json','look_eligibility_report.json','tranche_release_contract.json','dynamic_global_queue_contract.json','elastic_GPU_startup_lock_contract.json','phased_formal_access_audit_preseal.json','synthetic_group_sequential_validation_report_final.json','pre_interim_validation_report_final.json','false_pass_mutation_audit_final.json']
 doc=gs.signed('s4_group_sequential_implementation_seal','PASS',{'implementation_commit':commit,'implementation_tree':run(['git','rev-parse',commit+'^{tree}']),'branch':branch,'live_remote_branch_oid_before_seal_commit':remote,'bound_documents':{name:artifact(ROOT/name) for name in names},'look_manifests':[artifact(path) for path in sorted(ROOT.glob('look_*_manifest.json'))],'efficacy_aggregate_access_count_before_seal':0,'interim_execution_count_before_seal':0,'D2_formal_access_count':0,'S5_execution_count':0,'created_unix_ns':time.time_ns()},producer_commit=commit);gs.write_json_new(ROOT/'implementation_seal.json',doc);print(json.dumps({'status':'PASS','implementation_commit':commit,'remote_oid':remote},sort_keys=True))
def main()->int:
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True);s.add_parser('preseal');q=s.add_parser('seal');q.add_argument('commit');a=p.parse_args();preseal() if a.command=='preseal' else seal(a.commit);return 0
if __name__=='__main__':raise SystemExit(main())
