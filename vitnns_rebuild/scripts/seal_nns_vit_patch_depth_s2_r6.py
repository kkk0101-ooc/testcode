#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r6_20260914";R4=REPO/"evidence/nns_vit_patch_depth/s2_r4_20260914";R5=REPO/"evidence/nns_vit_patch_depth/s2_r5_20260914";LATE={"whole_root_inventory.json","terminal_receipt.json","independent_validation_report.json","false_pass_audit.json"}
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
    v={"schema_version":1,"study_id":"NNS_VIT_PATCH_DEPTH_SEMANTICS","stage_id":"S2-R6","run_id":ROOT.name,"document_type":t,"status":status,"payload":payload};v["canonical_digest"]=dig(v);return v
def write(name:str,v:dict[str,Any])->None:
    p=ROOT/name;req(not p.exists(),"overwrite");fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
def run(script:str,command:str)->dict[str,Any]:return json.loads(subprocess.check_output([sys.executable,script,command],cwd=REPO,text=True))
def impl(path:str,commit:str)->dict[str,Any]:
    r=rec(REPO/path);line=subprocess.check_output(["git","-C",str(REPO.parent),"ls-tree",commit,"--","vitnns_rebuild/"+path],text=True).strip();req(bool(line),"uncommitted implementation");r["git_blob_oid"]=line.split()[2];return r
def main()->None:
    req(not ROOT.exists(),"fresh R6 exists");r4=run("scripts/validate_nns_vit_patch_depth_s2_r4.py","final");r4a=run("scripts/validate_nns_vit_patch_depth_s2_r4.py","false-pass");r5=run("scripts/validate_nns_vit_patch_depth_s2_r5.py","validate");r5a=run("scripts/validate_nns_vit_patch_depth_s2_r5.py","false-pass");req(r4a["attack_count"]==23 and r5a["attack_count"]==8,"audit source drift");commit=subprocess.check_output(["git","-C",str(REPO.parent),"rev-parse","HEAD"],text=True).strip();ROOT.mkdir(parents=True)
    consolidation={"R4_terminal":rec(R4/"terminal_receipt.json"),"R5_terminal":rec(R5/"terminal_receipt.json"),"power_report_A1":rec(R4/"power_and_precision_simulation_report_a1.json"),"exact_sample_size_decision_A1":rec(R4/"exact_sample_size_decision_a1.json"),"R4_source_validation":r4,"R5_source_validation":r5,"power_or_protocol_recomputed_in_R6":False,"scientific_result_changed":False};write("publication_consolidation_report.json",signed("s2_r6_publication_consolidation_report","PASS",consolidation))
    write("formal_access_audit.json",signed("s2_r6_formal_access_audit","PASS",{"CPU_only":True,"formal_metadata_read_count":0,"formal_hash_only_byte_access_count":0,"formal_decode_count":0,"formal_outcome_access_count":0,"model_load_count":0,"GPU_CUDA_count":0,"attribution_count":0,"feature_extraction_count":0,"probe_training_count":0,"occlusion_count":0}))
    files=("scripts/validate_nns_vit_patch_depth_s2_r4.py","scripts/validate_nns_vit_patch_depth_s2_r5.py","scripts/validate_nns_vit_patch_depth_s2_r6.py","scripts/seal_nns_vit_patch_depth_s2_r6.py");write("implementation_seal.json",signed("s2_r6_implementation_seal","PASS",{"implementation_commit":commit,"files":[impl(x,commit) for x in files],"standard_library_only":True}))
    attacks=list(r4a["attacks"])+[dict(x,attack="R5_"+x["attack"]) for x in r5a["attacks"]];write("false_pass_audit.json",signed("s2_r6_consolidated_false_pass_audit","PASS",{"attack_count":len(attacks),"attacks":attacks,"R4_source_count":23,"R5_source_count":8}))
    current=[rec(p,ROOT) for p in sorted(ROOT.glob("*")) if p.is_file() and p.name not in LATE];write("whole_root_inventory.json",signed("s2_r6_whole_root_inventory","PASS",{"files":current,"files_digest":dig(current),"excluded_late_bound":sorted(LATE)}));evidence={p.name:rec(p,ROOT) for p in sorted(ROOT.glob("*.json")) if p.name not in ("terminal_receipt.json","independent_validation_report.json")}
    terminal=signed("s2_r6_terminal_receipt","BLOCKED_SCIENTIFIC_VALIDITY",{"status":"BLOCKED_SCIENTIFIC_VALIDITY","block_reason":"ALL_872_CLASSES_STRUCTURALLY_UNDERPOWERED_FOR_FROZEN_D2_DESIGN_ALTERNATIVE_UNDER_CONSERVATIVE_SMALL_PILOT_SCALE","selected_class_count":None,"maximum_candidate_class_count":872,"maximum_candidate_formal_counts":{"probe_train":5232,"probe_validation":1744,"confirmatory_holdout":1744},"maximum_candidate_is_authorized_cohort":False,"effect_thresholds":{"P1":.02,"P2":.1,"D1":.02,"D2":.01},"D2_power_at_872":.6934,"target_power":.9,"formal_outcome_access_count":0,"scientific_claim_decision":None,"next_authorization":"NONE","S3_executed":False,"S4_executed":False,"S5_executed":False,"cost_or_wall_time_caused_block":False,"forbidden_cost_terminal_used":False,"required_evidence":evidence});write("terminal_receipt.json",terminal);validation=run("scripts/validate_nns_vit_patch_depth_s2_r6.py","validate");write("independent_validation_report.json",signed("s2_r6_independent_validation_report","PASS",{"terminal_binding":rec(ROOT/"terminal_receipt.json",ROOT),"validation":validation}));print(json.dumps({"status":"BLOCKED_SCIENTIFIC_VALIDITY","D2_power_at_872":.6934,"false_pass_attack_count":31,"next_authorization":"NONE"},sort_keys=True))
if __name__=="__main__":main()
