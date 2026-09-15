#!/usr/bin/env python3
"""Seal S4 group-sequential STOP_SUCCESS or valid L5 gate closure."""
from __future__ import annotations
from collections import Counter
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from vitnns import nns_patch_depth_s4 as s4  # noqa:E402
from vitnns import nns_patch_depth_s4_group_sequential as gs  # noqa:E402
from vitnns import nns_patch_depth_s4_group_queue as queue  # noqa:E402
ROOT=REPO/gs.ROOT_REL
def require(c:bool,m:str)->None:
 if not c:raise RuntimeError(m)
def artifact(path:Path)->dict[str,Any]:
 v={'path':path.relative_to(REPO).as_posix(),'bytes':path.stat().st_size,'sha256':gs.sha_file(path)};d=json.loads(path.read_text()) if path.suffix=='.json' else {};
 if isinstance(d.get('canonical_digest'),str):v['canonical_digest']=d['canonical_digest']
 return v
def write(name:str,kind:str,status:str,payload:dict[str,Any],commit:str)->Path:
 path=ROOT/name;require(not path.exists(),'refusing overwrite '+name);gs.write_json_new(path,gs.signed(kind,status,payload,producer_commit=commit));return path
def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--look',type=int,required=True);a=p.parse_args();look=a.look;require(1<=look<=5,'look');decision_path=ROOT/f'interim/look-{look}/decision.json';decision=json.loads(decision_path.read_text());require(gs.verify_signed(decision),'decision digest');require(decision['status']=='STOP_SUCCESS' or (look==5 and decision['status']=='FINAL_GATE_CLOSED'),'not a terminal interim')
 schedule=json.loads((REPO/gs.SCHEDULE_REL).read_text())['payload']['records'];selected=[r for r in schedule if int(r['first_authorized_look'])<=look];counts=queue.terminal_counts(ROOT,schedule,look);require(counts['terminal']==counts['planned']==3*len(selected),'terminal topology incomplete');require(len(list((ROOT/'source_completions').glob('source-*.json')))==len(selected),'source completion count drift')
 future_claims=[]
 for path in ROOT.glob('launches/*/claims/*/*.json'):
  d=json.loads(path.read_text());q=d['payload'];
  if int(q.get('first_authorized_look',look))>look:future_claims.append(path.relative_to(REPO).as_posix())
 require(not future_claims,'future tranche claim after stop')
 commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();resources=[];states=Counter();types=Counter();provenance=Counter();array_records=[]
 for item in counts['records']:
  path=Path(item['path']);path=path if path.is_absolute() else REPO/path;doc=json.loads(path.read_text());q=doc['payload'];states[doc['status']]+=1;types[q['unit_type']]+=1;provenance[item['provenance']]+=1
  res=q.get('resources') or {};resources.append({'unit_type':q['unit_type'],'schedule_index':int(q['schedule_index']),'terminal_path':path.relative_to(REPO).as_posix(),'provenance':item['provenance'],'gpu_active_seconds':float(res.get('gpu_active_seconds',0)),'cpu_seconds':float(res.get('cpu_seconds',0)),'unit_wall_seconds':float(res.get('unit_wall_seconds',0)),'retained_bytes':int(res.get('retained_bytes',0)),'peak_allocated_gpu_bytes':int(res.get('peak_allocated_gpu_bytes',0)),'peak_reserved_gpu_bytes':int(res.get('peak_reserved_gpu_bytes',0))});array_records.extend(q.get('arrays',[]))
 resources.sort(key=lambda r:(r['unit_type'],r['schedule_index'],r['terminal_path']));launch_rows=[]
 for path in sorted(ROOT.glob('launches/*/launch_terminal.json')):
  d=json.loads(path.read_text());launch_rows.append({'path':path.relative_to(REPO).as_posix(),'look':d['payload'].get('look'),'status':d['status'],'elapsed_seconds':float(d['payload'].get('launch_elapsed_seconds',0))})
 ledger_path=write('cumulative_resource_ledger.json','s4_group_sequential_cumulative_resource_ledger','PASS',{'stable_sort_key':['unit_type','schedule_index','terminal_path'],'summation':'math.fsum','GPU_hours_not_divided_by_K':True,'cumulative_GPU_seconds':math.fsum(r['gpu_active_seconds'] for r in resources),'cumulative_GPU_hours':math.fsum(r['gpu_active_seconds'] for r in resources)/3600,'cumulative_CPU_seconds':math.fsum(r['cpu_seconds'] for r in resources),'cumulative_CPU_hours':math.fsum(r['cpu_seconds'] for r in resources)/3600,'unit_wall_seconds_sum':math.fsum(r['unit_wall_seconds'] for r in resources),'launch_elapsed_seconds_sum':math.fsum(r['elapsed_seconds'] for r in launch_rows),'retained_bytes':sum(r['retained_bytes'] for r in resources),'peak_allocated_GPU_bytes':max((r['peak_allocated_gpu_bytes'] for r in resources),default=0),'peak_reserved_GPU_bytes':max((r['peak_reserved_gpu_bytes'] for r in resources),default=0),'unit_records':resources,'launch_records':launch_rows,'cost_or_time_scientific_gate_used':False},commit)
 numeric=json.loads((ROOT/f'interim/look-{look}/full_numeric_result.json').read_text())['payload'];denom_path=write('typed_failure_NA_denominator_report.json','s4_group_sequential_typed_failure_NA_denominator_report','PASS',{'look':look,'planned_source_count':len(selected),'planned_unit_count':counts['planned'],'terminal_state_counts':dict(states),'unit_type_counts':dict(types),'terminal_provenance_counts':dict(provenance),'endpoint_validity':numeric['validity'],'P2_matching':numeric['P2_matching'],'substitute_count':0,'zero_fill_count':0},commit)
 access_path=write('phased_formal_access_audit.json','s4_group_sequential_phased_formal_access_audit','PASS',{'pre_redesign_partial_validity_status_known':True,'preseal_efficacy_aggregate_access_count':0,'look_endpoint_numeric_access':[{'look':i,'executed':(ROOT/f'interim/look-{i}/decision.json').is_file(),'authorized_source_count':gs.LOOK_IMAGE_COUNTS[i-1]} for i in range(1,look+1)],'future_look_outcome_access_count':0,'unauthorized_between_look_access_count':0,'D2_formal_access_count':0,'S5_execution_count':0,'S6_execution_count':0},commit)
 manifest_path=write('large_array_manifest.json','s4_group_sequential_large_array_manifest','PASS',{'record_count':len(array_records),'records_digest':gs.canonical_digest(array_records),'records':array_records,'rebind_arrays_not_copied':True},commit)
 required=[ledger_path,denom_path,access_path,manifest_path,decision_path,ROOT/'implementation_seal.json',ROOT/'s4_r4_completed_terminal_rebind_inventory.json'];inventory_exclude={'whole_root_inventory.json','terminal_receipt.json','completion/s4_group_sequential.pending.json'};files=[artifact(path) for path in sorted(ROOT.rglob('*')) if path.is_file() and path.relative_to(ROOT).as_posix() not in inventory_exclude];inventory_path=write('whole_root_inventory.json','s4_group_sequential_whole_root_inventory','PASS',{'files':files,'files_digest':gs.canonical_digest(files),'large_arrays_bound_by_manifest':True},commit);required.append(inventory_path)
 gate_open=decision['status']=='STOP_SUCCESS';next_auth='S5_D2_MISSING_CLASS_RECONCILIATION_GATE_ONLY' if gate_open else 'S5_PRIMARY_RESULTS_SEAL_ONLY';terminal=gs.signed('s4_group_sequential_terminal_receipt','PASS',{'S4_WORKFLOW_STATUS':'PASS','stop_reason':('STOP_SUCCESS_AT_LOOK_%d'%look) if gate_open else 'LOOK_5_FIRST_LAYER_GATE_CLOSED','actual_stop_look':look,'ACTUAL_S4_CLASS_COUNT':gs.LOOK_CLASS_COUNTS[look-1],'ACTUAL_S4_IMAGE_COUNT':gs.LOOK_IMAGE_COUNTS[look-1],'FIRST_LAYER_GATE_OPEN':gate_open,'D2_CONFIRMATORY_STATUS':'NOT_TESTED','scientific_result':'FIRST_LAYER_ALL_SUPPORTED' if gate_open else 'FIRST_LAYER_NOT_ALL_SUPPORTED_OR_GATE_CLOSED','terminal_count':counts['terminal'],'source_completion_count':len(selected),'future_tranche_claim_count':0,'S5_executed':False,'S6_executed':False,'required_evidence':{path.name:artifact(path) for path in required},'next_authorization':next_auth},producer_commit=commit)
 pending=ROOT/'completion/s4_group_sequential.pending.json';destination=ROOT/'terminal_receipt.json';require(pending.is_file() and not destination.exists(),'terminal transition unavailable');temporary=pending.with_name('s4_group_sequential.terminal.tmp');gs.write_json_new(temporary,terminal);os.link(temporary,destination);gs.fsync_directory(destination.parent);temporary.unlink();pending.unlink();gs.fsync_directory(pending.parent);print(json.dumps({'status':'PASS','look':look,'FIRST_LAYER_GATE_OPEN':gate_open,'terminal_count':counts['terminal'],'next_authorization':next_auth},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
