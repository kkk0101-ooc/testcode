#!/usr/bin/env python3
"""Post-seal strict validation and rebind of S4-R4 immutable terminals."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from vitnns import nns_patch_depth_s4 as s4  # noqa:E402
from vitnns import nns_patch_depth_s4_group_sequential as gs  # noqa:E402
from vitnns import nns_patch_depth_s4_group_queue as queue  # noqa:E402

R4_ROOT=REPO/'evidence/nns_vit_patch_depth/s4_r4_20260915'
ROOT=REPO/gs.ROOT_REL
ALLOWED=set(('VALID_NNS','VALID_OCCLUSION','NA_GEOMETRY_SIZE_MISMATCH','NA_TARGET_DISAPPEARED_AFTER_CROP','NA_MASK_LABEL_OUT_OF_RANGE','NA_VOID_MASK','NA_ZERO_POSITIVE_MASS','NA_MATCHED_CONTROL_UNAVAILABLE','NA_INCOMPLETE_DEPTH_TRAJECTORY','NA_UPSTREAM_NNS'))
EXPECTED_PRODUCER='acb338d8589a2a6af10187e16afca835630d5b0f'

def require(c:bool,m:str)->None:
    if not c: raise RuntimeError(m)

def artifact(path:Path)->dict[str,Any]:
    d=json.loads(path.read_text()) if path.suffix=='.json' else None
    v={'path':path.relative_to(REPO).as_posix(),'bytes':path.stat().st_size,'sha256':gs.sha_file(path)}
    if isinstance(d,dict) and isinstance(d.get('canonical_digest'),str):v['canonical_digest']=d['canonical_digest']
    return v

def verify_array(rec:dict[str,Any])->dict[str,Any]:
    path=REPO/rec['path']; require(path.is_file(),'missing array '+rec['path']); require(path.stat().st_size==int(rec['bytes']),'array bytes drift '+rec['path']); require(gs.sha_file(path)==rec['sha256'],'array SHA drift '+rec['path'])
    arr=np.load(path,allow_pickle=False); require(list(arr.shape)==rec['shape'] and str(arr.dtype)==rec['dtype'],'array shape/dtype drift '+rec['path'])
    if np.issubdtype(arr.dtype,np.number): require(bool(np.isfinite(arr).all()),'array nonfinite '+rec['path'])
    require(s4.array_digest(arr)==rec['canonical_array_digest'],'array digest drift '+rec['path'])
    return {'path':rec['path'],'sha256':rec['sha256'],'canonical_array_digest':rec['canonical_array_digest'],'shape':rec['shape'],'dtype':rec['dtype'],'bytes':rec['bytes']}

def main()->int:
    seal_path=ROOT/'implementation_seal.json'; require(seal_path.is_file(),'new implementation seal absent'); seal=json.loads(seal_path.read_text()); require(gs.verify_signed(seal) and seal['status']=='PASS','new implementation seal invalid')
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(); branch=subprocess.check_output(['git','-C',str(REPO),'symbolic-ref','--short','HEAD'],text=True).strip(); remote=subprocess.check_output(['git','-C',str(REPO),'ls-remote','--heads','origin',f'refs/heads/{branch}'],text=True).split()[0]
    require(head==remote,'new seal not live on remote')
    schedule=json.loads((REPO/gs.SCHEDULE_REL).read_text())['payload']['records']; by_index={int(row['schedule_index']):row for row in schedule}
    r4_seal_path=R4_ROOT/'implementation_seal.json'; r4_seal=json.loads(r4_seal_path.read_text()); require(s4.verify_signed(r4_seal) and r4_seal['status']=='PASS','R4 implementation seal invalid')
    terminals=sorted(R4_ROOT.glob('unit_terminals/*/unit-*.json')); require(len(terminals)==69,'R4 completed terminal count drift')
    rows=[]; states=Counter(); types=Counter(); array_count=0; seen=set()
    for path in terminals:
        doc=json.loads(path.read_text()); require(s4.verify_signed(doc),'terminal canonical digest drift '+str(path)); require(doc['status'] in ALLOWED,'terminal state not rebindable '+doc['status']); p=doc['payload']; index=int(p['schedule_index']); unit_type=p['unit_type']; key=(unit_type,index); require(key not in seen,'duplicate R4 terminal'); seen.add(key); source=by_index[index]
        require(p['source_id']==source['source_image_id'] and p['image_sha256']==source['image_sha256'] and p['mask_sha256']==source['annotation_sha256'] and int(p['class_target'])==int(source['target_model_class_index']),'terminal source identity drift')
        require(p['role_flags']==source['role_flags'],'terminal role flags drift'); require(p['producer_commit']==EXPECTED_PRODUCER and doc.get('producer_commit')==EXPECTED_PRODUCER,'producer commit drift')
        require(p.get('S3_D2_prediction_error_numeric_reads')==0 and p.get('cohort_level_inference_count')==0 and p.get('scientific_claim_decision') is None,'forbidden downstream access in R4 terminal')
        if unit_type=='NNS_IMAGE': require(p['component']=='first_residual_addition_all_head_aggregate_attention_branch' and p['operator'] is None,'NNS identity drift')
        elif unit_type=='OCCLUSION_PRIMARY': require(p['operator']=='image_channel_mean','primary operator drift')
        elif unit_type=='OCCLUSION_SENSITIVITY': require(p['operator']=='normalized_zero','sensitivity operator drift')
        else: raise RuntimeError('unit type drift')
        arrays=[verify_array(rec) for rec in p.get('arrays',[])]; array_count+=len(arrays)
        if doc['status']=='VALID_NNS':
            require(p['nns']['closure']['status']=='PASS' and p['nns']['blocks']==list(range(12)),'NNS closure/depth drift'); require(p['nns']['block11_matrix_zero_mass_explicit'] and p['nns']['block11_cls_routing_preserved'],'block11 drift')
            plan=p['p2_plan']['record']; plan_path=REPO/plan['path']; require(gs.sha_file(plan_path)==plan['sha256'],'P2 plan SHA drift'); plan_doc=json.loads(plan_path.read_text()); claimed=plan_doc.pop('canonical_digest'); require(gs.canonical_digest(plan_doc)==claimed==plan['canonical_digest'],'P2 plan digest drift')
        dep=p.get('dependency_terminal')
        if unit_type!='NNS_IMAGE':
            require(isinstance(dep,dict),'occlusion dependency missing'); dep_path=REPO/dep['path']; require(dep_path.is_file() and gs.sha_file(dep_path)==dep['sha256'],'dependency terminal SHA drift')
        binding=gs.signed('s4_r4_completed_unit_rebind','PASS',{'unit_type':unit_type,'schedule_index':index,'source_id':p['source_id'],'formal_class_rank':int(source['formal_class_rank']),'first_authorized_look':int(source['first_authorized_look']),'source_terminal_path':path.relative_to(REPO).as_posix(),'source_terminal_sha256':gs.sha_file(path),'source_terminal_canonical_digest':doc['canonical_digest'],'source_terminal_status':doc['status'],'source_producer_commit':EXPECTED_PRODUCER,'S4_R4_implementation_seal':artifact(r4_seal_path),'arrays':arrays,'array_count':len(arrays),'numeric_values_used_for_protocol_or_look_selection':False,'recompute_forbidden':True})
        destination=queue.rebind_path(ROOT,unit_type,index); gs.write_json_new(destination,binding)
        states[doc['status']]+=1; types[unit_type]+=1; rows.append({'unit_type':unit_type,'schedule_index':index,'source_id':p['source_id'],'formal_class_rank':int(source['formal_class_rank']),'status':doc['status'],'rebind_path':destination.relative_to(REPO).as_posix(),'rebind_sha256':gs.sha_file(destination),'source_terminal_path':path.relative_to(REPO).as_posix(),'source_terminal_sha256':gs.sha_file(path),'source_terminal_canonical_digest':doc['canonical_digest'],'array_count':len(arrays)})
    by_source={}
    for row in rows: by_source.setdefault(row['schedule_index'],[]).append(row)
    complete=0
    for index,items in sorted(by_source.items()):
        if {item['unit_type'] for item in items}==set(s4.UNIT_TYPES):
            source=by_index[index]; doc=gs.signed('s4_group_sequential_source_completion','COMPLETE',{'schedule_index':index,'source_id':source['source_image_id'],'formal_class_rank':int(source['formal_class_rank']),'provenance':'S4_R4_VALIDATED_REBIND','unit_terminals':items,'all_three_units_terminal':True,'scientific_claim_decision':None}); gs.write_json_new(ROOT/'source_completions'/f'source-{index:04d}.json',doc); complete+=1
    inventory=gs.signed('s4_r4_completed_terminal_rebind_inventory','PASS',{'source_root':R4_ROOT.relative_to(REPO).as_posix(),'source_terminal_count':len(terminals),'rebound_terminal_count':len(rows),'rebound_source_completion_count':complete,'unit_type_counts':dict(types),'state_counts':dict(states),'array_record_count':array_count,'records':sorted(rows,key=lambda r:(r['unit_type'],r['schedule_index'])),'existing_terminal_rerun_count':0,'formal_numeric_aggregate_access_count':0,'future_look_numeric_access_count':0})
    gs.write_json_new(ROOT/'s4_r4_completed_terminal_rebind_inventory.json',inventory)
    print(json.dumps({'status':'PASS','rebound_terminal_count':len(rows),'source_completion_count':complete,'unit_type_counts':dict(types),'state_counts':dict(states),'array_record_count':array_count,'existing_terminal_rerun_count':0},sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
