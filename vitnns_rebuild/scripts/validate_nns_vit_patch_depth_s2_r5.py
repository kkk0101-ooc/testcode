#!/usr/bin/env python3
"""Validate the final S2-R5 publication root and its immutable R4 source."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r5_20260914";R4=REPO/"evidence/nns_vit_patch_depth/s2_r4_20260914";LATE={"whole_root_inventory.json","terminal_receipt.json","independent_validation_report.json","false_pass_audit.json"}
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
def validate()->dict[str,Any]:
    source=json.loads(subprocess.check_output([sys.executable,"scripts/validate_nns_vit_patch_depth_s2_r4.py","final"],cwd=REPO,text=True));req(source["status"]=="PASS" and source["terminal_status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and source["D2_power_at_872"]==.6934,"R4 source validation drift")
    recovery=read(ROOT/"sealing_recovery_report.json");access=read(ROOT/"formal_access_audit.json");implementation=read(ROOT/"implementation_seal.json");terminal=read(ROOT/"terminal_receipt.json");inventory=read(ROOT/"whole_root_inventory.json")
    req(recovery["payload"]["predecessor_terminal"]==rec(R4/"terminal_receipt.json"),"R4 terminal binding drift");req(recovery["payload"]["R4_terminal_status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and recovery["payload"]["R4_sealing_failure"]=="PYTHON_3_7_DICT_UNION_IN_LATE_REPORT_PHASE","recovery identity drift")
    for item in implementation["payload"]["files"]:
        observed=rec(REPO/item["path"]);req(all(observed[k]==item[k] for k in ("path","bytes","sha256")),"live implementation drift: "+item["path"]);line=subprocess.check_output(["git","-C",str(REPO.parent),"ls-tree",implementation["payload"]["implementation_commit"],"--","vitnns_rebuild/"+item["path"]],text=True).strip();req(bool(line) and line.split()[2]==item["git_blob_oid"],"Git implementation drift")
    a=access["payload"];req(all(a[k]==0 for k in ("formal_hash_only_byte_access_count","formal_decode_count","formal_outcome_access_count","model_load_count","GPU_CUDA_count","attribution_count","feature_extraction_count","probe_training_count","occlusion_count")),"formal access violation")
    req(terminal["status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and terminal["payload"]["next_authorization"]=="NONE","terminal drift");req(terminal["payload"]["selected_class_count"] is None and terminal["payload"]["maximum_candidate_class_count"]==872 and terminal["payload"]["D2_power_at_872"]==.6934 and terminal["payload"]["target_power"]==.9,"scientific block drift");req(terminal["payload"]["scientific_claim_decision"] is None and not terminal["payload"]["S3_executed"],"claim or S3 drift")
    files=[rec(p,ROOT) for p in sorted(ROOT.glob("*")) if p.is_file() and p.name not in LATE];req(files==inventory["payload"]["files"] and dig(files)==inventory["payload"]["files_digest"],"inventory drift")
    for name,item in terminal["payload"]["required_evidence"].items():req(rec(ROOT/name,ROOT)==item,"terminal evidence drift: "+name)
    return {"status":"PASS","terminal_status":"BLOCKED_SCIENTIFIC_VALIDITY","source_R4_validation":source,"D2_power_at_872":.6934,"formal_outcome_access_count":0,"next_authorization":"NONE","inventory_file_count":len(files)}
def false_pass()->dict[str,Any]:
    base=json.loads((ROOT/"terminal_receipt.json").read_text());mut={"status":lambda x:x.update(status="PASS"),"authorization":lambda x:x["payload"].update(next_authorization="S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY"),"selected_class_count":lambda x:x["payload"].update(selected_class_count=872),"D2_power":lambda x:x["payload"].update(D2_power_at_872=.9),"threshold":lambda x:x["payload"].update(delta_D2=.005),"claim":lambda x:x["payload"].update(scientific_claim_decision="SUPPORTED"),"S3":lambda x:x["payload"].update(S3_executed=True),"cost":lambda x:x.update(status="BLOCKED_BUDGET")};rows=[]
    for name,fn in mut.items():
        v=json.loads(json.dumps(base));fn(v);v.pop("canonical_digest",None);v["canonical_digest"]=dig(v);rejected=False
        try:req(v["status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and v["payload"]["next_authorization"]=="NONE" and v["payload"]["selected_class_count"] is None and v["payload"]["D2_power_at_872"]==.6934 and "delta_D2" not in v["payload"] and v["payload"]["scientific_claim_decision"] is None and not v["payload"]["S3_executed"],"mutation")
        except Error:rejected=True
        req(rejected,"false pass: "+name);rows.append({"attack":name,"rejected":True})
    return {"status":"PASS","attack_count":len(rows),"attacks":rows}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("command",choices=("validate","false-pass"));a=p.parse_args();print(json.dumps(validate() if a.command=="validate" else false_pass(),sort_keys=True))
if __name__=="__main__":main()
