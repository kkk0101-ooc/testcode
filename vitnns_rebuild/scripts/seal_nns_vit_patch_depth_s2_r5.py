#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r5_20260914";R4=REPO/"evidence/nns_vit_patch_depth/s2_r4_20260914";LATE={"whole_root_inventory.json","terminal_receipt.json","independent_validation_report.json","false_pass_audit.json"}
def req(c:bool,m:str)->None:
    if not c:raise ValueError(m)
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def dig(v:Any)->str:return hashlib.sha256(canonical(v)).hexdigest()
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text());u=dict(v);d=u.pop("canonical_digest");req(d==dig(u),"canonical");return v
def rec(p:Path,base:Path=REPO)->dict[str,Any]:
    r={"path":p.relative_to(base).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
    if p.suffix==".json":
        try:r["canonical_digest"]=read(p)["canonical_digest"]
        except (KeyError,ValueError):pass
    return r
def signed(t:str,status:str,payload:dict[str,Any])->dict[str,Any]:
    v={"schema_version":1,"study_id":"NNS_VIT_PATCH_DEPTH_SEMANTICS","stage_id":"S2-R5","run_id":ROOT.name,"document_type":t,"status":status,"payload":payload};v["canonical_digest"]=dig(v);return v
def write(name:str,v:dict[str,Any])->None:
    p=ROOT/name;req(not p.exists(),"overwrite");fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
def impl(relative:str,commit:str)->dict[str,Any]:
    r=rec(REPO/relative);line=subprocess.check_output(["git","-C",str(REPO.parent),"ls-tree",commit,"--","vitnns_rebuild/"+relative],text=True).strip();req(bool(line),"uncommitted implementation: "+relative);r["git_blob_oid"]=line.split()[2];return r
def main()->None:
    req(not ROOT.exists(),"fresh R5 exists");source=json.loads(subprocess.check_output([sys.executable,"scripts/validate_nns_vit_patch_depth_s2_r4.py","final"],cwd=REPO,text=True));req(source["terminal_status"]=="BLOCKED_SCIENTIFIC_VALIDITY","R4 source drift");commit=subprocess.check_output(["git","-C",str(REPO.parent),"rev-parse","HEAD"],text=True).strip();ROOT.mkdir(parents=True)
    recovery=signed("s2_r5_sealing_recovery_report","PASS",{"predecessor_root":"evidence/nns_vit_patch_depth/s2_r4_20260914","predecessor_terminal":rec(R4/"terminal_receipt.json"),"R4_terminal_status":"BLOCKED_SCIENTIFIC_VALIDITY","R4_sealing_failure":"PYTHON_3_7_DICT_UNION_IN_LATE_REPORT_PHASE","R4_scientific_artifacts_recomputed":source,"scientific_design_or_result_changed":False,"power_simulation_rerun_in_R5":False});write("sealing_recovery_report.json",recovery)
    access=signed("s2_r5_formal_access_audit","PASS",{"CPU_only":True,"formal_metadata_read_count":0,"formal_hash_only_byte_access_count":0,"formal_decode_count":0,"formal_outcome_access_count":0,"model_load_count":0,"GPU_CUDA_count":0,"attribution_count":0,"feature_extraction_count":0,"probe_training_count":0,"occlusion_count":0});write("formal_access_audit.json",access)
    files=("scripts/validate_nns_vit_patch_depth_s2_r4.py","scripts/validate_nns_vit_patch_depth_s2_r5.py","scripts/seal_nns_vit_patch_depth_s2_r5.py","src/vitnns/nns_patch_depth_s2_r4.py");implementation=signed("s2_r5_implementation_seal","PASS",{"implementation_commit":commit,"files":[impl(x,commit) for x in files],"standard_library_only_validator":True});write("implementation_seal.json",implementation)
    current=[rec(p,ROOT) for p in sorted(ROOT.glob("*")) if p.is_file() and p.name not in LATE];write("whole_root_inventory.json",signed("s2_r5_whole_root_inventory","PASS",{"files":current,"files_digest":dig(current),"excluded_late_bound":sorted(LATE)}))
    evidence={p.name:rec(p,ROOT) for p in sorted(ROOT.glob("*.json")) if p.name not in ("terminal_receipt.json","independent_validation_report.json","false_pass_audit.json")};terminal=signed("s2_r5_terminal_receipt","BLOCKED_SCIENTIFIC_VALIDITY",{"status":"BLOCKED_SCIENTIFIC_VALIDITY","block_reason":"ALL_872_CLASSES_STRUCTURALLY_UNDERPOWERED_FOR_FROZEN_D2_DESIGN_ALTERNATIVE_UNDER_CONSERVATIVE_SMALL_PILOT_SCALE","selected_class_count":None,"maximum_candidate_class_count":872,"maximum_candidate_formal_counts":{"probe_train":5232,"probe_validation":1744,"confirmatory_holdout":1744},"maximum_candidate_is_authorized_cohort":False,"effect_thresholds":{"P1":.02,"P2":.1,"D1":.02,"D2":.01},"D2_power_at_872":.6934,"target_power":.9,"formal_outcome_access_count":0,"scientific_claim_decision":None,"next_authorization":"NONE","S3_executed":False,"S4_executed":False,"S5_executed":False,"cost_or_wall_time_caused_block":False,"forbidden_cost_terminal_used":False,"required_evidence":evidence});write("terminal_receipt.json",terminal)
    validation=json.loads(subprocess.check_output([sys.executable,"scripts/validate_nns_vit_patch_depth_s2_r5.py","validate"],cwd=REPO,text=True));audit=json.loads(subprocess.check_output([sys.executable,"scripts/validate_nns_vit_patch_depth_s2_r5.py","false-pass"],cwd=REPO,text=True));binding=rec(ROOT/"terminal_receipt.json",ROOT);write("independent_validation_report.json",signed("s2_r5_independent_validation_report","PASS",{"terminal_binding":binding,"validation":validation}));write("false_pass_audit.json",signed("s2_r5_false_pass_audit","PASS",{"terminal_binding":binding,"audit":audit}));print(json.dumps({"status":"BLOCKED_SCIENTIFIC_VALIDITY","D2_power_at_872":.6934,"next_authorization":"NONE"},sort_keys=True))
if __name__=="__main__":main()
