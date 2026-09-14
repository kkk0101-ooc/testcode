#!/usr/bin/env python3
"""Publication-level validation for a terminal S2-R8 root."""
from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r8_20260914";LATE={"whole_root_inventory.json","terminal_receipt.json","independent_validation_report.json"}
def req(c:bool,m:str)->None:
    if not c:raise ValueError(m)
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def dig(v:Any)->str:return hashlib.sha256(canonical(v)).hexdigest()
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text());u=dict(v);d=u.pop("canonical_digest");req(d==dig(u),"canonical mismatch: "+str(p));return v
def rec(p:Path,base:Path=REPO)->dict[str,Any]:
    r={"path":p.relative_to(base).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
    if p.suffix==".json":
        try:r["canonical_digest"]=read(p)["canonical_digest"]
        except (KeyError,ValueError):pass
    return r
def validate()->dict[str,Any]:
    scientific=json.loads(subprocess.check_output([sys.executable,"scripts/validate_nns_vit_patch_depth_s2_r8.py","final"],cwd=REPO,text=True));req(scientific["status"]=="PASS" and scientific["terminal_status"]=="PASS" and scientific["selected_cap"]==4,"scientific validator drift")
    implementation=read(ROOT/"implementation_seal.json");inventory=read(ROOT/"whole_root_inventory.json");terminal=read(ROOT/"terminal_receipt.json");independent=read(ROOT/"independent_validation_report.json");audit=read(ROOT/"false_pass_audit.json")
    for item in implementation["payload"]["files"]:
        observed=rec(REPO/item["path"]);req(all(observed[k]==item[k] for k in ("path","bytes","sha256")),"live finalizer drift");line=subprocess.check_output(["git","-C",str(REPO.parent),"ls-tree",implementation["payload"]["finalizer_commit"],"--","vitnns_rebuild/"+item["path"]],text=True).strip();req(line and line.split()[2]==item["git_blob_oid"],"finalizer Git blob drift")
    files=[rec(p,ROOT) for p in sorted(ROOT.glob("*")) if p.is_file() and p.name not in LATE];req(files==inventory["payload"]["files"] and dig(files)==inventory["payload"]["files_digest"],"inventory drift")
    for name,item in terminal["payload"]["required_evidence"].items():req(rec(ROOT/name,ROOT)==item,"terminal evidence drift: "+name)
    req(audit["payload"]["attack_count"]==16 and all(x["rejected"] for x in audit["payload"]["attacks"]),"false-pass drift")
    req(independent["payload"]["terminal_binding"]==rec(ROOT/"terminal_receipt.json",ROOT),"independent terminal binding drift")
    return {"status":"PASS","terminal_status":"PASS","selected_cap":4,"selected_D2_image_count":3482,"next_authorization":"S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY","formal_outcome_access_count":0,"inventory_file_count":len(files)}
def main()->None:print(json.dumps(validate(),sort_keys=True))
if __name__=="__main__":main()
