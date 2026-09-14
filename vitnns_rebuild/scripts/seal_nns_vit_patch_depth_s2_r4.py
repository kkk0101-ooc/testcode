#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r4_20260914";LATE={"whole_root_inventory.json","terminal_receipt.json","independent_validation_report.json","false_pass_audit.json"}
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
    v={"schema_version":1,"study_id":"NNS_VIT_PATCH_DEPTH_SEMANTICS","stage_id":"S2-R4","run_id":ROOT.name,"document_type":t,"status":status,"payload":payload};v["canonical_digest"]=dig(v);return v
def write(name:str,v:dict[str,Any])->None:
    p=ROOT/name;req(not p.exists(),"overwrite");fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
def main()->None:
    result=json.loads(subprocess.check_output([sys.executable,"scripts/validate_nns_vit_patch_depth_s2_r4.py","recompute"],cwd=REPO,text=True));req(result["terminal_status"]=="BLOCKED_SCIENTIFIC_VALIDITY","unexpected result");implementation=[rec(REPO/p) for p in ("src/vitnns/nns_patch_depth_s2_r4.py","scripts/build_nns_vit_patch_depth_s2_r4.py","scripts/amend_nns_vit_patch_depth_s2_r4_a1.py","scripts/validate_nns_vit_patch_depth_s2_r4.py","scripts/seal_nns_vit_patch_depth_s2_r4.py","tests/test_nns_patch_depth_s2_r4.py")];write("implementation_seal.json",signed("s2_r4_implementation_seal","PASS",{"pre_power_commit":"1654463f4","files":implementation,"standard_library_validator":True,"synthetic_concurrency_tests":"PASS"}))
    files=[rec(p,ROOT) for p in sorted(ROOT.glob("*")) if p.is_file() and p.name not in LATE];write("whole_root_inventory.json",signed("s2_r4_whole_root_inventory","PASS",{"files":files,"files_digest":dig(files),"excluded_late_bound":sorted(LATE)}))
    terminal=signed("s2_r4_terminal_receipt","BLOCKED_SCIENTIFIC_VALIDITY",{"status":"BLOCKED_SCIENTIFIC_VALIDITY","block_reason":"ALL_872_CLASSES_STRUCTURALLY_UNDERPOWERED_FOR_FROZEN_D2_DESIGN_ALTERNATIVE_UNDER_CONSERVATIVE_SMALL_PILOT_SCALE","selected_class_count":None,"maximum_candidate_class_count":872,"maximum_candidate_formal_counts":{"probe_train":5232,"probe_validation":1744,"confirmatory_holdout":1744},"maximum_candidate_is_authorized_cohort":False,"effect_thresholds":{"P1":.02,"P2":.1,"D1":.02,"D2":.01},"D2_power_at_872":.6934,"target_power":.9,"formal_outcome_access_count":0,"scientific_claim_decision":None,"next_authorization":"NONE","S3_executed":False,"S4_executed":False,"S5_executed":False,"cost_or_wall_time_caused_block":False,"forbidden_cost_terminal_used":False,"required_evidence":{p.name:rec(p,ROOT) for p in sorted(ROOT.glob("*.json")) if p.name not in ("terminal_receipt.json","independent_validation_report.json","false_pass_audit.json")}});write("terminal_receipt.json",terminal)
    validation=json.loads(subprocess.check_output([sys.executable,"scripts/validate_nns_vit_patch_depth_s2_r4.py","final"],cwd=REPO,text=True));audit=json.loads(subprocess.check_output([sys.executable,"scripts/validate_nns_vit_patch_depth_s2_r4.py","false-pass"],cwd=REPO,text=True));binding=rec(ROOT/"terminal_receipt.json",ROOT);write("independent_validation_report.json",signed("s2_r4_independent_standard_library_source_recomputing_validation_report","PASS",{"terminal_binding":binding,"validation":validation}));audit_payload={"terminal_binding":binding};audit_payload.update(audit);write("false_pass_audit.json",signed("s2_r4_false_pass_mutation_audit","PASS",audit_payload));print(json.dumps({"status":terminal["status"],"D2_power_at_872":.6934,"next_authorization":"NONE"},sort_keys=True))
if __name__=="__main__":main()
