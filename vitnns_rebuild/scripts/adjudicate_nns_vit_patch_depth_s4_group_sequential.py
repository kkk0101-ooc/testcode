#!/usr/bin/env python3
"""CPU-only look-boundary adjudicator with a future-outcome read firewall."""
from __future__ import annotations

from collections import Counter
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from vitnns import nns_patch_depth_s4_group_sequential as gs  # noqa:E402
from vitnns import nns_patch_depth_s4_group_queue as queue  # noqa:E402
from vitnns import nns_patch_depth_s4 as s4  # noqa:E402
ROOT=REPO/gs.ROOT_REL

def require(c:bool,m:str)->None:
    if not c:raise RuntimeError(m)

def install_firewall(allowed:set[int])->None:
    unit_pattern=re.compile(r'(?:unit-|source-)(\d{4})(?:\D|$)')
    protected=(str((REPO/'evidence/nns_vit_patch_depth/s4_r4_20260915').resolve()),str(ROOT.resolve()),str((REPO/'artifacts/nns_vit_patch_depth/s4_r4_20260915').resolve()),str((REPO/gs.PAYLOAD_REL).resolve()))
    def hook(event:str,args:tuple[Any,...])->None:
        if event!='open' or not args:return
        try:path=os.path.realpath(os.fspath(args[0]))
        except TypeError:return
        if not any(path==root or path.startswith(root+os.sep) for root in protected):return
        match=unit_pattern.search(path)
        if match and int(match.group(1)) not in allowed:raise PermissionError('future-look outcome firewall rejected '+path)
    sys.addaudithook(hook)

def arrays(terminal:dict[str,Any])->dict[str,dict[str,Any]]:
    return {Path(row['path']).stem:row for row in terminal['payload'].get('arrays',[])}
def load_array(record:dict[str,Any])->np.ndarray:
    path=REPO/record['path'];require(path.is_file(),'array missing');require(gs.sha_file(path)==record['sha256'],'array SHA drift');value=np.load(path,allow_pickle=False);require(list(value.shape)==record['shape'] and str(value.dtype)==record['dtype'],'array shape/dtype drift');require(bool(np.isfinite(value).all()),'array nonfinite');require(s4.array_digest(value)==record['canonical_array_digest'],'array digest drift');return value

def endpoint_rows(look:int,schedule:list[dict[str,Any]])->tuple[dict[str,list[dict[str,Any]]],dict[str,Any]]:
    selected=[row for row in schedule if int(row['formal_class_rank'])<gs.LOOK_CLASS_COUNTS[look-1]];out={e:[] for e in gs.ENDPOINTS};states=Counter();matched=0;possible=0;failures=[]
    for row in selected:
        index=int(row['schedule_index']);resolved={u:queue.resolved_terminal(ROOT,u,index) for u in s4.UNIT_TYPES};require(all(v is not None for v in resolved.values()),f'incomplete topology {index}')
        docs={u:resolved[u][1] for u in s4.UNIT_TYPES};states.update(d['status'] for d in docs.values())
        for u,d in docs.items():
            if d['status'] in ('FAILED_IMPLEMENTATION','BLOCKED_DATA_INTEGRITY','FAILED_NNS','FAILED_OCCLUSION'):failures.append({'schedule_index':index,'unit_type':u,'status':d['status']})
        nns=docs['NNS_IMAGE'];
        if nns['status']=='VALID_NNS':
            by=arrays(nns);num=load_array(by['p1_numerators']).astype(np.float64);den=load_array(by['p1_denominators']).astype(np.float64);valid=load_array(by['p1_valid_blocks']).astype(bool);possible+=36
            plan_path=REPO/nns['payload']['p2_plan']['record']['path'];plan=json.loads(plan_path.read_text());matched+=sum(c['status']=='MATCHED' for c in plan['coordinates'])
            if bool(valid.all()) and bool((den>0).all()):
                corrected=num[:,0]/den-(num[:,1:]/den[:,None]).mean(1);p1=float(corrected.mean());d1=float(corrected[8:12].mean()-corrected[0:4].mean());out['P1'].append({'class_rank':int(row['formal_class_rank']),'schedule_index':index,'value':p1});out['D1'].append({'class_rank':int(row['formal_class_rank']),'schedule_index':index,'value':d1})
        primary=docs['OCCLUSION_PRIMARY']
        if primary['status']=='VALID_OCCLUSION':
            rec=next(iter(arrays(primary).values()));logits=load_array(rec).astype(np.float64);require(logits.shape==(36,5),'P2 logits geometry');effect=logits[:,1:].mean(1)-logits[:,0];out['P2'].append({'class_rank':int(row['formal_class_rank']),'schedule_index':index,'value':float(effect.mean())})
    return out,{'unit_state_counts':dict(states),'failures':failures,'P2_matched_coordinate_count':matched,'P2_possible_coordinate_count':possible,'P2_matching_success_fraction':None if possible==0 else matched/possible,'selected_source_count':len(selected)}

def aggregate(rows:list[dict[str,Any]],scheduled:int,endpoint:str,look:int)->dict[str,Any]:
    by_class={}
    for row in rows:by_class.setdefault(row['class_rank'],[]).append(row['value'])
    groups=[values for _,values in sorted(by_class.items())];class_values=np.asarray([math.fsum(values)/len(values) for values in groups],dtype=np.float64);n=len(rows);classes=len(class_values);estimate=float(class_values.mean()) if classes else None;se=float(class_values.std(ddof=1)/math.sqrt(classes)) if classes>1 else None;bca=None if classes<3 else gs.bca_mean_interval(groups,endpoint,look)
    return {'scheduled_images':scheduled,'valid_images':n,'NA_images':scheduled-n,'valid_fraction':n/scheduled,'NA_fraction':1-n/scheduled,'valid_classes':classes,'estimate':estimate,'class_cluster_standard_error':se,'ordinary_95_CI_BCa':bca,'ordinary_95_CI_half_width':None if bca is None else bca['half_width']}

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument('--look',type=int,required=True);args=parser.parse_args();look=args.look;require(1<=look<=5,'look')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA_VISIBLE_DEVICES not empty');require('torch' not in sys.modules and 'PIL' not in sys.modules,'forbidden import before adjudication')
    manifest=next(ROOT.glob(f'look_{look}_*_manifest.json'));look_doc=json.loads(manifest.read_text());allowed={int(r['schedule_index']) for r in look_doc['payload']['records']};install_firewall(allowed)
    schedule=json.loads((REPO/gs.SCHEDULE_REL).read_text())['payload']['records'];completion=json.loads((ROOT/f'look_{look}_completion_receipt.json').read_text());require(completion['status']=='PASS','look completion absent')
    numeric_validation=json.loads((ROOT/f'look_{look}_numeric_validation_report.json').read_text());require(gs.verify_signed(numeric_validation) and numeric_validation['status']=='PASS' and numeric_validation['payload']['look']==look,'independent numeric validation absent')
    endpoint,engineering=endpoint_rows(look,schedule);summaries={e:aggregate(endpoint[e],gs.LOOK_IMAGE_COUNTS[look-1],e,look) for e in gs.ENDPOINTS};validity={e:{'valid_fraction':summaries[e]['valid_fraction'],'NA_fraction':summaries[e]['NA_fraction'],'pass':summaries[e]['valid_fraction']>=.85 and summaries[e]['NA_fraction']<=.15} for e in gs.ENDPOINTS};validity['P2_matching']={'fraction':engineering['P2_matching_success_fraction'],'pass':engineering['P2_matching_success_fraction'] is not None and engineering['P2_matching_success_fraction']>=.85}
    precision={e:{'ordinary_95_CI_half_width':summaries[e]['ordinary_95_CI_half_width'],'pass':summaries[e]['ordinary_95_CI_half_width'] is not None and summaries[e]['ordinary_95_CI_half_width']<=.015} for e in gs.ENDPOINTS}
    history={e:[] for e in (*gs.ENDPOINTS,'D1_LOWER')};estimate_history={e:[] for e in gs.ENDPOINTS};standard_errors={e:[] for e in gs.ENDPOINTS}
    for prior in range(1,look):
        prior_path=ROOT/f'interim/look-{prior}/full_numeric_result.json'
        # Historical look results are required for repeated-look inference.
        # The firewall permits them because every prior look is a strict subset
        # of the current allowed source set.
        d=json.loads(prior_path.read_text())['payload'];
        for e in gs.ENDPOINTS:history[e].append(d['z_statistics'][e]);estimate_history[e].append(d['endpoint_summaries'][e]['estimate']);standard_errors[e].append(d['endpoint_summaries'][e]['class_cluster_standard_error'])
        history['D1_LOWER'].append(d['z_statistics']['D1_LOWER'])
    z={}
    for e in ('P1','P2'):
        s=summaries[e];z[e]=(s['estimate']-gs.THRESHOLDS[e])/s['class_cluster_standard_error'];history[e].append(z[e]);estimate_history[e].append(s['estimate']);standard_errors[e].append(s['class_cluster_standard_error'])
    s=summaries['D1'];z['D1']=(s['estimate']-gs.THRESHOLDS['D1'])/s['class_cluster_standard_error'];z['D1_LOWER']=(-s['estimate']-gs.THRESHOLDS['D1'])/s['class_cluster_standard_error'];history['D1'].append(z['D1']);history['D1_LOWER'].append(z['D1_LOWER']);estimate_history['D1'].append(s['estimate']);standard_errors['D1'].append(s['class_cluster_standard_error'])
    table=json.loads((ROOT/'critical_boundary_table.json').read_text())['payload'];closed=gs.crossed_by_look(history,table,look-1);adjusted_p=gs.repeated_adjusted_p_values(history,look-1);adjusted_ci={e:gs.repeated_adjusted_ci(estimate_history[e],standard_errors[e],e,look-1,table) for e in gs.ENDPOINTS};ci_consistent={'P1':adjusted_ci['P1']['lower']>gs.THRESHOLDS['P1'],'P2':adjusted_ci['P2']['lower']>gs.THRESHOLDS['P2'],'D1':adjusted_ci['D1']['lower']>gs.THRESHOLDS['D1'] or adjusted_ci['D1']['upper']<-gs.THRESHOLDS['D1']}
    eligibility=json.loads((ROOT/'look_eligibility_report.json').read_text())['payload']['looks'][look-1];integrity={'complete_topology':completion['payload']['terminal_count']==completion['payload']['planned_unit_count'],'unexplained_failures':engineering['failures'],'no_unexplained_failures':not engineering['failures'],'substitute_count':0,'zero_fill_count':0,'array_validation':numeric_validation['status'],'numeric_validation_receipt':(ROOT/f'look_{look}_numeric_validation_report.json').relative_to(REPO).as_posix()};efficacy={'elementary_rejections':closed['elementary_rejections'],'all_three_rejected':all(closed['elementary_rejections'].values()),'adjusted_p_values':adjusted_p,'adjusted_CIs':adjusted_ci,'adjusted_CI_threshold_consistency':ci_consistent,'all_adjusted_CIs_consistent':all(ci_consistent.values()),'local_intersections':closed['local_intersections']}
    all_validity=all(item['pass'] for item in validity.values());all_gates=integrity['complete_topology'] and integrity['no_unexplained_failures'] and all_validity and all(item['pass'] for item in precision.values()) and efficacy['all_three_rejected'] and efficacy['all_adjusted_CIs_consistent'] and eligibility['eligible_STOP_SUCCESS_look']
    decision='BLOCKED_DATA_INTEGRITY' if not integrity['complete_topology'] or not integrity['no_unexplained_failures'] else 'INVALID' if not all_validity else 'STOP_SUCCESS' if all_gates else ('CONTINUE_TO_NEXT_LOOK' if look<5 else 'FINAL_GATE_CLOSED');numeric=gs.signed('s4_group_sequential_interim_full_numeric_result',decision,{'look':look,'endpoint_summaries':summaries,'z_statistics':z,'integrity':integrity,'validity':validity,'precision':precision,'sequential_efficacy':efficacy,'design_eligibility':eligibility,'P2_matching':{'matched':engineering['P2_matched_coordinate_count'],'possible':engineering['P2_possible_coordinate_count'],'fraction':engineering['P2_matching_success_fraction']},'future_look_outcome_access_count':0,'unauthorized_between_look_access_count':0,'D2_formal_test_count':0,'S5_execution_count':0,'scientific_claim_scope':'P1_P2_D1_FIRST_LAYER_ONLY'})
    directory=ROOT/f'interim/look-{look}';gs.write_json_new(directory/'full_numeric_result.json',numeric);decision_doc=gs.signed('s4_group_sequential_interim_decision',decision,{'look':look,'decision':decision,'integrity':{k:v for k,v in integrity.items() if k!='unexplained_failures'},'validity':{k:{'pass':v['pass']} for k,v in validity.items()},'precision':{k:{'pass':v['pass']} for k,v in precision.items()},'sequential_efficacy':{'all_three_rejected':efficacy['all_three_rejected'],'all_adjusted_CIs_consistent':efficacy['all_adjusted_CIs_consistent']},'design_eligibility':{'pass':eligibility['eligible_STOP_SUCCESS_look']},'future_look_outcome_access_count':0,'D2_formal_test_count':0,'S5_execution_count':0});gs.write_json_new(directory/'decision.json',decision_doc);print(json.dumps({'decision':decision,'look':look},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
