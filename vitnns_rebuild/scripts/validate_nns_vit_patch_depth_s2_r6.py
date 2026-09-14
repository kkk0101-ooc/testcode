#!/usr/bin/env python3
"""Validate final S2-R6 consolidated publication evidence."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r6_20260914";R4=REPO/"evidence/nns_vit_patch_depth/s2_r4_20260914";R5=REPO/"evidence/nns_vit_patch_depth/s2_r5_20260914";LATE={"whole_root_inventory.json","terminal_receipt.json","independent_validation_report.json","false_pass_audit.json"}
class Error(ValueError):pass
def req(c:bool,m:str)->None:
    if not c:raise Error(m)
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def dig(v:Any)->str:return hashlib.sha256(canonical(v)).hexdigest()
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text());u=dict(v);d=u.pop("canonical_digest");req(d==dig(u),"canonical mismatch: "+str(p));return v
def rec(p:Path,base:Path=REPO)->dict[str,Any]:
    r={"path":p.relative_to(base).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
    if p.suffix==".json":
        try:r["canonical_digest"]=read(p)["canonical_digest"]
        except (KeyError,Error):pass
    return r
def run(script:str,command:str)->dict[str,Any]:return json.loads(subprocess.check_output([sys.executable,script,command],cwd=REPO,text=True))
def validate()->dict[str,Any]:
    r4=run("scripts/validate_nns_vit_patch_depth_s2_r4.py","final");r4_audit=run("scripts/validate_nns_vit_patch_depth_s2_r4.py","false-pass");r5=run("scripts/validate_nns_vit_patch_depth_s2_r5.py","validate");r5_audit=run("scripts/validate_nns_vit_patch_depth_s2_r5.py","false-pass")
    req(r4["terminal_status"]==r5["terminal_status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and r4["D2_power_at_872"]==r5["D2_power_at_872"]==.6934,"source result drift");req(r4_audit["attack_count"]==23 and r5_audit["attack_count"]==8,"source mutation count drift")
    consolidation=read(ROOT/"publication_consolidation_report.json");access=read(ROOT/"formal_access_audit.json");implementation=read(ROOT/"implementation_seal.json");audit=read(ROOT/"false_pass_audit.json");terminal=read(ROOT/"terminal_receipt.json");inventory=read(ROOT/"whole_root_inventory.json")
    req(consolidation["payload"]["R4_terminal"]==rec(R4/"terminal_receipt.json") and consolidation["payload"]["R5_terminal"]==rec(R5/"terminal_receipt.json"),"source terminal binding drift");req(consolidation["payload"]["power_report_A1"]==rec(R4/"power_and_precision_simulation_report_a1.json"),"power binding drift")
    req(audit["payload"]["attack_count"]==31 and len(audit["payload"]["attacks"])==31 and all(x["rejected"] for x in audit["payload"]["attacks"]),"consolidated false-pass drift");req([x["attack"] for x in audit["payload"]["attacks"]]==[x["attack"] for x in r4_audit["attacks"]]+["R5_"+x["attack"] for x in r5_audit["attacks"]],"consolidated attack identity drift")
    a=access["payload"];req(all(a[k]==0 for k in ("formal_metadata_read_count","formal_hash_only_byte_access_count","formal_decode_count","formal_outcome_access_count","model_load_count","GPU_CUDA_count","attribution_count","feature_extraction_count","probe_training_count","occlusion_count")),"R6 access violation")
    for item in implementation["payload"]["files"]:
        observed=rec(REPO/item["path"]);req(all(observed[k]==item[k] for k in ("path","bytes","sha256")),"implementation drift");line=subprocess.check_output(["git","-C",str(REPO.parent),"ls-tree",implementation["payload"]["implementation_commit"],"--","vitnns_rebuild/"+item["path"]],text=True).strip();req(bool(line) and line.split()[2]==item["git_blob_oid"],"implementation Git drift")
    p=terminal["payload"];req(terminal["status"]==p["status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and p["next_authorization"]=="NONE","terminal drift");req(p["selected_class_count"] is None and p["maximum_candidate_class_count"]==872 and p["D2_power_at_872"]==.6934 and p["target_power"]==.9,"scientific block drift");req(p["effect_thresholds"]=={"P1":.02,"P2":.1,"D1":.02,"D2":.01} and p["scientific_claim_decision"] is None and not p["S3_executed"],"threshold claim or route drift")
    files=[rec(path,ROOT) for path in sorted(ROOT.glob("*")) if path.is_file() and path.name not in LATE];req(files==inventory["payload"]["files"] and dig(files)==inventory["payload"]["files_digest"],"inventory drift");req(all(rec(ROOT/name,ROOT)==item for name,item in p["required_evidence"].items()),"terminal evidence drift")
    return {"status":"PASS","terminal_status":"BLOCKED_SCIENTIFIC_VALIDITY","D2_power_at_872":.6934,"target_power":.9,"false_pass_attack_count":31,"formal_outcome_access_count":0,"next_authorization":"NONE"}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("command",choices=("validate",));p.parse_args();print(json.dumps(validate(),sort_keys=True))
if __name__=="__main__":main()
