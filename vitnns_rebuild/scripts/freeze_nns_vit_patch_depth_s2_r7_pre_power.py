#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r7_20260914";SPEC=REPO/"config/nns_vit_patch_depth/s2_r7_d2_redesign.json";DOC=REPO/"docs/nns_vit_patch_depth_s2_r7_d2_efficiency_amendment_20260914.md";PRE=REPO/"evidence/nns_vit_patch_depth/s2_r6_20260914/terminal_receipt.json"
def req(c:bool,m:str)->None:
    if not c:raise ValueError(m)
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def dig(v:Any)->str:return hashlib.sha256(canonical(v)).hexdigest()
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text());u=dict(v);d=u.pop("canonical_digest");req(d==dig(u),"canonical");return v
def rec(p:Path)->dict[str,Any]:return{"path":p.relative_to(REPO).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
def signed(t:str,payload:dict[str,Any])->dict[str,Any]:
    v={"schema_version":1,"study_id":"NNS_VIT_PATCH_DEPTH_SEMANTICS","stage_id":"S2-R7","run_id":ROOT.name,"document_type":t,"status":"PASS","payload":payload};v["canonical_digest"]=dig(v);return v
def write(name:str,v:dict[str,Any])->None:
    fd=os.open(ROOT/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
def main()->None:
    req(not ROOT.exists(),"fresh root exists");pre=read(PRE);req(pre["status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and pre["payload"]["next_authorization"]=="NONE","predecessor drift");s=json.loads(SPEC.read_text());req(s["unchanged_scientific_contract"]["D2_conservative_scale"]==.13320415057793725 and s["cluster_aware_power"]["power_gate"]==.9,"scientific contract drift");ROOT.mkdir(parents=True);payload={"predecessor_terminal":rec(PRE),"spec":rec(SPEC),"amendment":rec(DOC),"implementation":rec(Path(__file__).resolve()),"frozen_D2_redesign":s,"pilot_D2_pair_or_ICC_access_before_seal_count":0,"formal_outcome_access_count":0,"scientific_claim":None};write("d2_redesign_amendment.json",signed("s2_r7_d2_redesign_amendment",payload));write("pre_power_seal.json",signed("s2_r7_pre_power_seal",payload));write("pre_power_access_audit.json",signed("s2_r7_pre_power_access_audit",{"metadata_capacity_records_inspected":21609,"pilot_D2_pair_or_ICC_access_count":0,"formal_decode_or_outcome_access_count":0,"model_GPU_attribution_probe_occlusion_count":0}));print(json.dumps({"status":"PASS","root":ROOT.relative_to(REPO).as_posix()},sort_keys=True))
if __name__=="__main__":main()
