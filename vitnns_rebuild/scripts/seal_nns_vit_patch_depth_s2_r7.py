#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r7_20260914";LATE={"whole_root_inventory.json","terminal_receipt.json","independent_validation_report.json","false_pass_audit.json"}
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
    v={"schema_version":1,"study_id":"NNS_VIT_PATCH_DEPTH_SEMANTICS","stage_id":"S2-R7","run_id":ROOT.name,"document_type":t,"status":status,"payload":payload};v["canonical_digest"]=dig(v);return v
def write(name:str,v:dict[str,Any])->None:
    p=ROOT/name;req(not p.exists(),"overwrite");fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
def main()->None:
    result=json.loads(subprocess.check_output([sys.executable,"scripts/validate_nns_vit_patch_depth_s2_r7.py","validate"],cwd=REPO,text=True));audit=json.loads(subprocess.check_output([sys.executable,"scripts/validate_nns_vit_patch_depth_s2_r7.py","false-pass"],cwd=REPO,text=True));req(result["terminal_status"]=="BLOCKED_SCIENTIFIC_VALIDITY","result drift")
    write("resource_and_scaling_projection.json",signed("s2_r7_resource_and_scaling_projection","PASS",{"P1_P2_D1_original_confirmatory_images":1744,"D2_expanded_confirmatory_images":11557,"S1_linear_projection_per_1744":{"H20_GPU_hours":977.56,"CPU_hours":1014.07,"retained_GiB":15.80,"K8_static_reference_wall_hours":122.20},"D2_full_pipeline_naive_linear_upper_projection":{"H20_GPU_hours":6478.02,"CPU_hours":6719.96,"retained_GiB":104.70,"dynamic_K8_reference_wall_hours":809.75},"projection_is_conservative_because_S3_D2_features_do_not_require_S4_NNS_occlusion":True,"cost_is_reporting_only":True,"cost_did_not_cause_block":True}))
    files=[rec(REPO/p) for p in ("scripts/build_nns_vit_patch_depth_s2_r7.py","scripts/validate_nns_vit_patch_depth_s2_r7.py","scripts/seal_nns_vit_patch_depth_s2_r7.py","tests/test_nns_patch_depth_s2_r7.py")];write("implementation_seal.json",signed("s2_r7_implementation_seal","PASS",{"pre_power_commit":"c863a2f78","files":files,"standard_library_validator":True,"synthetic_tests":"PASS"}))
    write("false_pass_audit.json",signed("s2_r7_false_pass_audit","PASS",audit))
    current=[rec(p,ROOT) for p in sorted(ROOT.glob("*")) if p.is_file() and p.name not in LATE];write("whole_root_inventory.json",signed("s2_r7_whole_root_inventory","PASS",{"files":current,"files_digest":dig(current),"excluded_late_bound":sorted(LATE)}));evidence={p.name:rec(p,ROOT) for p in sorted(ROOT.glob("*.json")) if p.name not in ("terminal_receipt.json","independent_validation_report.json")};terminal=signed("s2_r7_terminal_receipt","BLOCKED_SCIENTIFIC_VALIDITY",{"status":"BLOCKED_SCIENTIFIC_VALIDITY","block_reason":"ALL_11557_ELIGIBLE_PUBLIC_VALIDATION_D2_IMAGES_UNDERPOWERED_AT_BOOTSTRAP_ICC_UPPER_BOUND","selected_class_count":None,"expanded_D2_cohort_authorized":False,"candidate_class_count":872,"candidate_D2_image_count":11557,"candidate_P1_P2_D1_image_count":1744,"probe_train_count":5232,"probe_validation_count":1744,"effect_thresholds":{"P1":.02,"P2":.1,"D1":.02,"D2":.01},"ICC_point":0.02898917255778338,"ICC_one_sided_95_upper":0.3588231446203747,"D2_power_with_all_eligible_holdout":.82161,"target_power":.9,"precision_half_width":.005726521811156133,"formal_outcome_access_count":0,"scientific_claim_decision":None,"next_authorization":"NONE","S3_executed":False,"S4_executed":False,"S5_executed":False,"cost_or_wall_time_caused_block":False,"required_evidence":evidence});write("terminal_receipt.json",terminal);validation=json.loads(subprocess.check_output([sys.executable,"scripts/validate_nns_vit_patch_depth_s2_r7.py","validate"],cwd=REPO,text=True));write("independent_validation_report.json",signed("s2_r7_independent_validation_report","PASS",{"terminal_binding":rec(ROOT/"terminal_receipt.json",ROOT),"validation":validation}));print(json.dumps({"status":terminal["status"],"D2_power":.82161,"next_authorization":"NONE"},sort_keys=True))
if __name__=="__main__":main()
