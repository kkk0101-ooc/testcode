"""CPU-only S4-R12 selection-anchor recovery primitives."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

from . import nns_patch_depth_s4 as s4
from . import nns_patch_depth_s4_group_sequential as gs

STUDY_ID = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
STAGE_ID = "S4-SELECTION-ANCHOR-RECOVERY"
RUN_ID = "s4_r12_selection_anchor_recovery_20260916"
ROOT_REL = Path("evidence/nns_vit_patch_depth/s4_r12_selection_anchor_recovery_20260916")
R10_ROOT_REL = Path("evidence/nns_vit_patch_depth/s4_r10_group_sequential_20260915")
R11_ROOT_REL = Path("evidence/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916")
SCHEDULE_REL = Path("manifests/nns_vit_patch_depth/s4_r10_group_sequential_schedule.json")
FULL_RECORDS_DIGEST = "3bb68cb6bb627f4a84f1a6d7bd54e26d430541b4ae4cd565a6fe3eb2d46863e9"
MEMBERSHIP_DIGEST = "ad0c10e5057d767ef4329f81b22c5f1569afb7e820c7f88bb5199da7e03cb9b2"
CUTOFF_NS = 1789511492710807249
CUTOFF_INDEX = 300
EXPECTED_COUNTS = {"selected_sources": 240, "selected_terminals": 720, "selected_distinct_class_ranks": 122, "P1_D1_valid_classes": 120, "P2_valid_classes": 119}
EXPECTED_STATES = {"VALID_NNS":235,"NA_VOID_MASK":5,"VALID_OCCLUSION":448,"NA_MATCHED_CONTROL_UNAVAILABLE":22,"NA_UPSTREAM_NNS":10}
R11_BINDINGS = {
 "terminal_receipt.json": ("eaabb5cc0912ea7b6bc632300e1214bcedbc16febe7c2fc1b11e2d71e6efa6a4", "651a8ce172f34b6894f098fe2ab2e3910d11d56353bffd19d312c0140f92ccac"),
 "exploratory_240_source_manifest.json": ("81e3414cd04681bfc626b4cca836caa3cc2052b916aba18454b43eca08fcd1f5", "94b44455674f7fd68a85bfefc97ca1677d09beefdb7fa056f1998b1f728974f0"),
 "endpoint_numeric_result.json": ("1357713798be111b6f971552b3d9fff32d64986d4f723ad0696b6c8174df49f4", "323d0758404226efdcb8cffab621c185818b30bc6bee898fdfeb31e8f0c68334"),
 "independent_validation_report.json": ("85e52a1de3b091351068b7adcbf4e0acc71b421ddd248f68d14f887bb614e72f", "baf90e27eef2ed8c3495a3fc34cee5d565fa84f35e467bb7620aa237ad60821b"),
 "s4_r10_immutable_root_inventory.json": ("c5d0183f09455f6d544e4fd20b218e9fae06d4aa9466e9d907aa9645f93a1119", "96cbc818bac5b4e4d8df62b90991071211407b8b7a57e3d594483392bd8d3b3d"),
}
EXPECTED_NUMERIC = {
 "P1": {"estimate":.0626003110,"lower":.0573913097,"upper":.0678940724,"half_width":.0052937614,"adjusted_lower":.0514420599,"adjusted_p":1.8768816516782683e-19},
 "P2": {"estimate":.0749396239,"lower":.0570841929,"upper":.0927266631,"half_width":.0178554310,"adjusted_lower":.0347448291,"adjusted_p":1.613666525930498e-05},
 "D1": {"estimate":.0363141999,"lower":.0316151874,"upper":.0409722310,"half_width":.0046990126,"adjusted_lower":.0258156526,"adjusted_upper":.0468127473,"adjusted_p":4.136584313500504e-05},
}
FAILURES={"FAILED_IMPLEMENTATION","BLOCKED_DATA_INTEGRITY","FAILED_NNS","FAILED_OCCLUSION"}

class R12Error(RuntimeError): pass
def require(c:bool,m:str)->None:
 if not c:raise R12Error(m)
def canonical_bytes(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def canonical_digest(v:Any)->str:return hashlib.sha256(canonical_bytes(v)).hexdigest()
def sha_file(path:Path)->str:
 before=path.stat();h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 after=path.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'file changed while hashing '+str(path));return h.hexdigest()
def signed(kind:str,status:str,payload:Mapping[str,Any],commit:str|None=None)->dict[str,Any]:
 d={"schema_version":1,"study_id":STUDY_ID,"stage_id":STAGE_ID,"run_id":RUN_ID,"document_type":kind,"status":status,"payload":dict(payload)}
 if commit:d['producer_commit']=commit
 d['canonical_digest']=canonical_digest(d);return d
def verify_signed(d:Mapping[str,Any])->bool:
 x=dict(d);c=x.pop('canonical_digest',None);return isinstance(c,str) and c==canonical_digest(x)
def write_json_new(path:Path,d:Any)->None:
 path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'wb') as f:f.write(json.dumps(d,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b'\n');f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def artifact(repo:Path,path:Path)->dict[str,Any]:
 d=json.loads(path.read_text()) if path.suffix=='.json' else {};v={"path":path.relative_to(repo).as_posix(),"bytes":path.stat().st_size,"sha256":sha_file(path)}
 if isinstance(d.get('canonical_digest'),str):v['canonical_digest']=d['canonical_digest']
 return v

def verify_r11_bindings(repo:Path)->dict[str,Any]:
 rows=[];root=repo/R11_ROOT_REL
 for name,(sha,digest) in R11_BINDINGS.items():
  path=root/name;require(path.is_file() and sha_file(path)==sha,'R11 SHA drift '+name);doc=json.loads(path.read_text());require(doc['canonical_digest']==digest,'R11 canonical binding drift '+name)
  copy=dict(doc);claim=copy.pop('canonical_digest');require(canonical_digest(copy)==claim,'R11 canonical recomputation drift '+name);rows.append(artifact(repo,path))
 terminal=json.loads((root/'terminal_receipt.json').read_text());require(terminal['status']=='BLOCKED_DATA_INTEGRITY','R11 blocked status not preserved')
 inv=json.loads((root/'s4_r10_immutable_root_inventory.json').read_text())['payload'];require(inv['file_count']==8684 and inv['records_digest']=='eb1557b9747d4c0ac31dc3930c3fbe3c9374a564ae8c6d7947a90a00be726d41','R10 inventory binding drift')
 return {'bound_files':rows,'R11_terminal_status':terminal['status'],'R10_inventory_file_count':inv['file_count'],'R10_inventory_records_digest':inv['records_digest']}

def corrected_identity(records:Sequence[Mapping[str,Any]])->dict[str,Any]:
 projection=[{'schedule_index':int(r['schedule_index']),'formal_class_rank':int(r['formal_class_rank']),'source_image_id':r['source_id']} for r in records]
 return {'full_manifest_records_digest':canonical_digest(list(records)),'membership_projection_digest':canonical_digest(projection),'membership_projection':projection}

def verify_selection(repo:Path)->tuple[list[dict[str,Any]],dict[str,Any]]:
 manifest=json.loads((repo/R11_ROOT_REL/'exploratory_240_source_manifest.json').read_text());require(manifest['canonical_digest']==R11_BINDINGS['exploratory_240_source_manifest.json'][1],'R11 manifest digest')
 records=manifest['payload']['records'];identity=corrected_identity(records);schedule=json.loads((repo/SCHEDULE_REL).read_text())['payload']['records'];r10=repo/R10_ROOT_REL;resolved=[]
 for source in schedule:
  if int(source['formal_class_rank'])>=200:continue
  terms=[]
  for unit_type in s4.UNIT_TYPES:
   new=r10/'unit_terminals'/unit_type.lower()/f"unit-{int(source['schedule_index']):04d}.json"
   if new.is_file():path=new;doc=json.loads(path.read_text());provenance='NEW'
   else:
    bind=r10/'rebind_terminals'/unit_type.lower()/f"unit-{int(source['schedule_index']):04d}.json"
    if not bind.is_file():terms=[];break
    b=json.loads(bind.read_text());require(gs.verify_signed(b),'R10 rebind digest');path=repo/b['payload']['source_terminal_path'];doc=json.loads(path.read_text());provenance='REBIND'
   require((gs.verify_signed(doc) or s4.verify_signed(doc)),'terminal canonical digest');terms.append((path,doc,provenance))
  if len(terms)==3:resolved.append((max(t[1]['payload']['completed_unix_ns'] for t in terms),int(source['schedule_index'])))
 resolved.sort();expected_indices=[x[1] for x in resolved[:240]];actual_indices=[int(r['schedule_index']) for r in records];states=Counter(t['status'] for r in records for t in r['terminals']);failures=sum(t['status'] in FAILURES for r in records for t in r['terminals'])
 checks={"R11_records_unchanged":sha_file(repo/R11_ROOT_REL/'exploratory_240_source_manifest.json')==R11_BINDINGS['exploratory_240_source_manifest.json'][0],"first_240_selection_exact":actual_indices==expected_indices,"source_count_240":len(records)==240,"terminal_count_720":sum(len(r['terminals']) for r in records)==720,"selected_distinct_class_ranks_122":len({r['formal_class_rank'] for r in records})==122,"cutoff_time":records[-1]['source_completion_time_max_unix_ns']==CUTOFF_NS,"cutoff_index":records[-1]['schedule_index']==CUTOFF_INDEX,"terminal_states":dict(states)==EXPECTED_STATES,"unexplained_failures_zero":failures==0,"full_manifest_records_digest":identity['full_manifest_records_digest']==FULL_RECORDS_DIGEST,"membership_projection_digest":identity['membership_projection_digest']==MEMBERSHIP_DIGEST}
 return records,{**identity,'selected_source_count':len(records),'selected_terminal_count':sum(len(r['terminals']) for r in records),'selected_distinct_class_rank_count':len({r['formal_class_rank'] for r in records}),'cutoff_completion_unix_ns':records[-1]['source_completion_time_max_unix_ns'],'cutoff_schedule_index':records[-1]['schedule_index'],'terminal_state_counts':dict(states),'unexplained_failure_count':failures,'checks':checks,'all_checks_pass':all(checks.values())}

def _array(repo:Path,rec:Mapping[str,Any])->np.ndarray:
 path=repo/rec['path'];require(path.is_file() and path.stat().st_size==rec['bytes'],'array file/bytes');require(sha_file(path)==rec['sha256'],'array SHA');a=np.load(path,allow_pickle=False);require(list(a.shape)==rec['shape'] and str(a.dtype)==rec['dtype'] and bool(np.isfinite(a).all()),'array geometry/finiteness');require(s4.array_digest(a)==rec['canonical_array_digest'],'array digest');return a

def compute(repo:Path,records:Sequence[Mapping[str,Any]])->tuple[dict[str,Any],dict[str,Any]]:
 endpoint={e:[] for e in gs.ENDPOINTS};states=Counter();arrays=plans=matched=possible=0
 for row in records:
  docs={}
  for tr in row['terminals']:
   path=repo/tr['path'];require(path.is_file() and path.stat().st_size==tr['bytes'] and sha_file(path)==tr['sha256'],'terminal bytes/SHA');doc=json.loads(path.read_text());require((gs.verify_signed(doc) or s4.verify_signed(doc)) and doc['canonical_digest']==tr['canonical_digest'],'terminal canonical');require(doc['payload']['source_id']==row['source_id'] and doc['payload']['schedule_index']==row['schedule_index'],'terminal source identity');docs[tr['unit_type']]=doc;states[doc['status']]+=1
  require(set(docs)==set(s4.UNIT_TYPES),'terminal topology');n=docs['NNS_IMAGE'];rank=int(row['formal_class_rank'])
  if n['status']=='VALID_NNS':
   by={Path(a['path']).stem:a for a in n['payload']['arrays']}
   for rec in n['payload']['arrays']:_array(repo,rec);arrays+=1
   num=_array(repo,by['p1_numerators']).astype(float);den=_array(repo,by['p1_denominators']).astype(float);valid=_array(repo,by['p1_valid_blocks']).astype(bool)
   plan_record=n['payload']['p2_plan']['record'];path=repo/plan_record['path'];require(sha_file(path)==plan_record['sha256'],'plan SHA');plan=json.loads(path.read_text());claim=plan.pop('canonical_digest');require(canonical_digest(plan)==claim==plan_record['canonical_digest'],'plan digest');plans+=1;possible+=36;matched+=sum(c['status']=='MATCHED' for c in plan['coordinates'])
   if valid.all() and (den>0).all():cor=num[:,0]/den-(num[:,1:]/den[:,None]).mean(1);endpoint['P1'].append((rank,float(cor.mean())));endpoint['D1'].append((rank,float(cor[8:12].mean()-cor[0:4].mean())))
  for unit in ('OCCLUSION_PRIMARY','OCCLUSION_SENSITIVITY'):
   d=docs[unit];dep=d['payload']['dependency_terminal'];require(dep['canonical_digest']==n['canonical_digest'] and sha_file(repo/dep['path'])==dep['sha256'],'dependency binding')
   for rec in d['payload'].get('arrays',[]):a=_array(repo,rec);arrays+=1
   if unit=='OCCLUSION_PRIMARY' and d['status']=='VALID_OCCLUSION':endpoint['P2'].append((rank,float((a.astype(float)[:,1:].mean(1)-a.astype(float)[:,0]).mean())))
 table=json.loads((repo/R10_ROOT_REL/'critical_boundary_table.json').read_text())['payload'];summary={}
 for e in gs.ENDPOINTS:
  by={}
  for rank,v in endpoint[e]:by.setdefault(rank,[]).append(v)
  groups=[v for _,v in sorted(by.items())];cm=np.array([math.fsum(v)/len(v) for v in groups]);est=float(cm.mean());se=float(cm.std(ddof=1)/math.sqrt(len(cm)));bca=gs.bca_mean_interval(groups,e,1);b=float(gs.boundary_lookup(table,3,e)[0]);summary[e]={'scheduled_images':240,'valid_images':len(endpoint[e]),'NA_images':240-len(endpoint[e]),'valid_fraction':len(endpoint[e])/240,'NA_fraction':1-len(endpoint[e])/240,'valid_classes':len(groups),'estimate':est,'class_cluster_standard_error':se,'ordinary_BCa_95':bca,'family_size_three_L1_adjusted_CI':{'lower':est-b*se,'upper':est+b*se if e=='D1' else None,'boundary_z':b}}
 z={'P1':[(summary['P1']['estimate']-.02)/summary['P1']['class_cluster_standard_error']],'P2':[summary['P2']['estimate']/summary['P2']['class_cluster_standard_error']],'D1':[(summary['D1']['estimate']-.02)/summary['D1']['class_cluster_standard_error']],'D1_LOWER':[(-summary['D1']['estimate']-.02)/summary['D1']['class_cluster_standard_error']]}
 p=gs.repeated_adjusted_p_values(z,0);closed=gs.crossed_by_look(z,table,0)
 for e in gs.ENDPOINTS:summary[e]['heterogeneous_null_adjusted_p']=p[e]
 anchors={}
 for e,x in EXPECTED_NUMERIC.items():
  s=summary[e];obs={'estimate':s['estimate'],'lower':s['ordinary_BCa_95']['lower'],'upper':s['ordinary_BCa_95']['upper'],'half_width':s['ordinary_BCa_95']['half_width'],'adjusted_lower':s['family_size_three_L1_adjusted_CI']['lower'],'adjusted_p':s['heterogeneous_null_adjusted_p']}
  if e=='D1':obs['adjusted_upper']=s['family_size_three_L1_adjusted_CI']['upper']
  anchors[e]=all(abs(obs[k]-x[k])<=max(5e-10,abs(x[k])*5e-8) for k in x)
 matching=matched/possible;gates={'complete_720_terminal_topology':sum(states.values())==720,'no_unexplained_failure':not any(s in FAILURES for s in states),'P1_valid_fraction':summary['P1']['valid_fraction']>=.85 and summary['P1']['NA_fraction']<=.15,'P2_valid_fraction':summary['P2']['valid_fraction']>=.85 and summary['P2']['NA_fraction']<=.15,'D1_valid_fraction':summary['D1']['valid_fraction']>=.85 and summary['D1']['NA_fraction']<=.15,'P2_matching_fraction':matching>=.85,'P1_precision':summary['P1']['ordinary_BCa_95']['half_width']<=.015,'P2_precision':summary['P2']['ordinary_BCa_95']['half_width']<=.02,'D1_precision':summary['D1']['ordinary_BCa_95']['half_width']<=.015,'P1_adjusted_lower':summary['P1']['family_size_three_L1_adjusted_CI']['lower']>.02,'P2_point':summary['P2']['estimate']>=.07,'P2_adjusted_lower':summary['P2']['family_size_three_L1_adjusted_CI']['lower']>0,'D1_adjusted_interval':summary['D1']['family_size_three_L1_adjusted_CI']['lower']>.02 or summary['D1']['family_size_three_L1_adjusted_CI']['upper']<-.02,'closed_testing_all_rejected':all(closed['elementary_rejections'].values()),'numeric_anchors':all(anchors.values())}
 original={'P2_point_above_0_10':summary['P2']['estimate']>.10,'P2_adjusted_lower_above_0_10':summary['P2']['family_size_three_L1_adjusted_CI']['lower']>.10,'P2_half_width_at_most_0_015':summary['P2']['ordinary_BCa_95']['half_width']<=.015}
 return {'endpoint_summaries':summary,'heterogeneous_null_z':z,'closed_testing':closed,'numeric_anchor_checks':anchors,'posthoc_gate_checks':gates,'all_posthoc_gates_pass':all(gates.values()),'original_P2_rule_checks':original,'ORIGINAL_RULE_NOT_PASSED':not all(original.values())},{'terminal_state_counts':dict(states),'array_records_validated':arrays,'P2_plans_validated':plans,'matching_count':matched,'matching_possible':possible,'matching_fraction':matching}
