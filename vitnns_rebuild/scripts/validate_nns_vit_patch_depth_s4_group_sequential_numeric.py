#!/usr/bin/env python3
"""Independent current-look numeric integrity validator; no aggregation."""
from __future__ import annotations
import argparse,json,os,re,subprocess,sys
from collections import Counter
from pathlib import Path
from typing import Any
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from vitnns import nns_patch_depth_s4 as s4  # noqa:E402
from vitnns import nns_patch_depth_s4_group_sequential as gs  # noqa:E402
from vitnns import nns_patch_depth_s4_group_queue as queue  # noqa:E402
ROOT=REPO/gs.ROOT_REL
def require(c:bool,m:str)->None:
 if not c:raise RuntimeError(m)
def firewall(allowed:set[int])->None:
 pattern=re.compile(r'(?:unit-|source-)(\d{4})(?:\D|$)');protected=(str((REPO/'evidence/nns_vit_patch_depth/s4_r4_20260915').resolve()),str(ROOT.resolve()),str((REPO/'artifacts/nns_vit_patch_depth/s4_r4_20260915').resolve()),str((REPO/gs.PAYLOAD_REL).resolve()))
 def hook(event:str,args:tuple[Any,...])->None:
  if event!='open' or not args:return
  try:path=os.path.realpath(os.fspath(args[0]))
  except TypeError:return
  if any(path==r or path.startswith(r+os.sep) for r in protected):
   match=pattern.search(path)
   if match and int(match.group(1)) not in allowed:raise PermissionError('future look firewall rejected '+path)
 sys.addaudithook(hook)
def array(rec:dict[str,Any],shape:tuple[int,...],dtype:str)->np.ndarray:
 path=REPO/rec['path'];require(path.is_file() and path.stat().st_size==rec['bytes'],'array file/bytes');require(gs.sha_file(path)==rec['sha256'],'array SHA');value=np.load(path,allow_pickle=False);require(value.shape==shape and str(value.dtype)==dtype,'array geometry');require(bool(np.isfinite(value).all()),'array finiteness');require(s4.array_digest(value)==rec['canonical_array_digest'],'array digest');return value
def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--look',type=int,required=True);a=p.parse_args();look=a.look;require(1<=look<=5,'look');require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA visibility');require('torch' not in sys.modules and 'PIL' not in sys.modules,'forbidden imports')
 schedule=json.loads((REPO/gs.SCHEDULE_REL).read_text())['payload']['records'];selected=[r for r in schedule if int(r['formal_class_rank'])<gs.LOOK_CLASS_COUNTS[look-1]];allowed={int(r['schedule_index']) for r in selected};firewall(allowed);states=Counter();types=Counter();arrays_validated=0;audit=0;matched=0;possible=0
 for row in selected:
  index=int(row['schedule_index'])
  for unit_type in s4.UNIT_TYPES:
   resolved=queue.resolved_terminal(ROOT,unit_type,index);require(resolved is not None,'missing terminal');path,doc,provenance=resolved;payload=doc['payload'];require(payload['source_id']==row['source_image_id'] and payload['image_sha256']==row['image_sha256'] and payload['mask_sha256']==row['annotation_sha256'],'source identity');states[doc['status']]+=1;types[unit_type]+=1
   by={Path(rec['path']).stem:rec for rec in payload.get('arrays',[])}
   if doc['status']=='VALID_NNS':
    scores=array(by['source_scores'],(4,12,196),'float32');occ=array(by['occupancy'],(4,196),'float32');array(by['block_features'],(12,196,768),'float32');perms=array(by['p1_permutation_indices'],(12,64,196),'int16');num=array(by['p1_numerators'],(12,65),'float64');den=array(by['p1_denominators'],(12,),'float64');flags=array(by['p1_valid_blocks'],(12,),'uint8');arrays_validated+=7;require(np.allclose(scores[1],np.maximum(scores[0],0),atol=0,rtol=0) and np.allclose(scores[2],np.minimum(scores[0],0),atol=0,rtol=0),'score decomposition');require(np.allclose(occ.astype(np.float64).sum(0),1,atol=0,rtol=0),'occupancy closure');require(np.allclose(den,scores[1].astype(np.float64).sum(1),rtol=1e-6,atol=1e-8),'P1 denominator');require(doc['payload']['nns']['closure']['status']=='PASS' and doc['payload']['nns']['blocks']==list(range(12)),'NNS closure');require(doc['payload']['nns']['block11_matrix_zero_mass_explicit'],'block11')
    for block in range(12):
     for repeat in range(64):
      for stratum in range(7):
       members=np.flatnonzero(s4.POSITION_STRATA==stratum);require(sorted(perms[block,repeat,members].tolist())==members.tolist(),'P1 strata')
    plan_record=payload['p2_plan']['record'];plan_path=REPO/plan_record['path'];require(gs.sha_file(plan_path)==plan_record['sha256'],'plan SHA');plan=json.loads(plan_path.read_text());claim=plan.pop('canonical_digest');require(gs.canonical_digest(plan)==claim,'plan digest');possible+=36;matched+=sum(c['status']=='MATCHED' for c in plan['coordinates']);require(len(plan['coordinates'])==36,'plan count')
    if 'audit_first_attention_T' in by:
     T=array(by['audit_first_attention_T'],(12,196,196),'float32');C=array(by['audit_first_attention_C'],(12,196),'float32');require(float(np.abs(T[11]).sum(dtype=np.float64))==0.0,'audit block11');audit+=1;arrays_validated+=2
   elif doc['status']=='VALID_OCCLUSION':
    require(len(by)==1,'occlusion array count');array(next(iter(by.values())),(36,5),'float32');arrays_validated+=1;require(payload['coordinate_plan_shared_between_operators'] and not payload['operator_selected_from_logit_drop'],'operator plan')
 require(types==Counter({'NNS_IMAGE':len(selected),'OCCLUSION_PRIMARY':len(selected),'OCCLUSION_SENSITIVITY':len(selected)}),'type counts')
 result={'look':look,'source_count':len(selected),'unit_count':3*len(selected),'state_counts':dict(states),'arrays_validated':arrays_validated,'audit_subset_array_pairs_validated':audit,'P2_matched_coordinate_count':matched,'P2_possible_coordinate_count':possible,'future_look_outcome_access_count':0,'cohort_level_aggregate_count':0,'CUDA_VISIBLE_DEVICES':'','model_load_count':0,'JPEG_mask_decode_count':0}
 commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();receipt=gs.signed('s4_group_sequential_numeric_validation','PASS',result,producer_commit=commit);destination=ROOT/f'look_{look}_numeric_validation_report.json';gs.write_json_new(destination,receipt);print(json.dumps({'status':'PASS',**result},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
