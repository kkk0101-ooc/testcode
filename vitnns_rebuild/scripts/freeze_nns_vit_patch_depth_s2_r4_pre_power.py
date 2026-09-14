#!/usr/bin/env python3
"""Create the S2-R4 pre-power seal without reading pilot endpoints."""
from __future__ import annotations
import hashlib, json, os, subprocess
from pathlib import Path
from typing import Any

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r4_20260914"
CONFIG=REPO/"config/nns_vit_patch_depth/s2_r4_pre_power_protocol.json"
SCHEMA=REPO/"config/nns_vit_patch_depth/s2_r4_pre_power_protocol.schema.json"
AMENDMENT=REPO/"docs/nns_vit_patch_depth_s2_r4_recovery_amendment_20260914.md"
PREDECESSOR=REPO/"evidence/nns_vit_patch_depth/s2_r3_20260914/terminal_receipt.json"

def require(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def canonical(v:Any)->bytes: return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(v:Any)->str: return hashlib.sha256(canonical(v)).hexdigest()
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def read_signed(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text()); u=dict(v); d=u.pop("canonical_digest"); require(d==digest(u),"canonical mismatch: "+str(p)); return v
def record(p:Path)->dict[str,Any]: return {"path":p.relative_to(REPO).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
def signed(t:str,status:str,payload:dict[str,Any])->dict[str,Any]:
    v={"schema_version":1,"study_id":"NNS_VIT_PATCH_DEPTH_SEMANTICS","stage_id":"S2-R4","run_id":ROOT.name,"document_type":t,"status":status,"payload":payload}; v["canonical_digest"]=digest(v); return v
def write(p:Path,v:dict[str,Any])->None:
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f: f.write(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n"); f.flush(); os.fsync(f.fileno())
def main()->None:
    require(not ROOT.exists(),"fresh root already exists")
    predecessor=read_signed(PREDECESSOR); require(predecessor["status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and predecessor["payload"]["next_authorization"]=="NONE","predecessor drift")
    spec=json.loads(CONFIG.read_text()); require(spec["pre_power"]["effect_thresholds"]=={"P1":0.02,"P2":0.1,"D1":0.02,"D2":0.01},"threshold drift")
    forbidden=json.dumps(spec["pre_power"],sort_keys=True); require(all(x not in forbidden for x in ("pilot_mean","pilot_p_value","pilot_sign")),"forbidden pilot input in seal")
    root_status=subprocess.check_output(["git","-C",str(REPO.parent),"status","--porcelain","--","vitnns_rebuild"],text=True).splitlines()
    allowed={"?? vitnns_rebuild/config/nns_vit_patch_depth/s2_r4_pre_power_protocol.json","?? vitnns_rebuild/config/nns_vit_patch_depth/s2_r4_pre_power_protocol.schema.json","?? vitnns_rebuild/docs/nns_vit_patch_depth_s2_r4_recovery_amendment_20260914.md","?? vitnns_rebuild/scripts/freeze_nns_vit_patch_depth_s2_r4_pre_power.py"}
    require(set(root_status)<=allowed,"unexpected vitnns_rebuild change before pre-power seal: "+repr(root_status))
    ROOT.mkdir(parents=True)
    payload={"predecessor_terminal":record(PREDECESSOR),"protocol":record(CONFIG),"schema":record(SCHEMA),"amendment":record(AMENDMENT),"implementation":record(Path(__file__).resolve()),"effect_thresholds":spec["pre_power"]["effect_thresholds"],"threshold_basis":spec["pre_power"]["threshold_units_and_basis"],"candidate_class_counts":spec["pre_power"]["candidate_class_counts"],"power_algorithm":{k:spec["pre_power"][k] for k in ("selection_rule","power_seed_namespace","calibration_seed_namespace","power_replicates","calibration_replicates","target_power_per_endpoint_under_worst_case_Holm_slot","design_alternative","precision_gate","type_I_gate","coverage_gate","D2_small_pilot_handling")},"statistics":spec["statistics"],"allowed_pilot_fields_after_seal":spec["pre_power"]["allowed_pilot_fields_after_seal"],"forbidden_power_inputs":spec["pre_power"]["forbidden_power_inputs"],"new_pilot_endpoint_or_nuisance_access_before_seal_count":0,"formal_outcome_access_count":0,"historical_exposure_acknowledged":True,"scientific_claim_decision":None}
    write(ROOT/"recovery_amendment.json",signed("s2_r4_recovery_amendment","PASS",payload))
    write(ROOT/"pre_power_effect_threshold_seal.json",signed("s2_r4_pre_power_effect_threshold_seal","PASS",payload))
    write(ROOT/"pre_power_access_audit.json",signed("s2_r4_pre_power_access_audit","PASS",{"new_pilot_endpoint_or_nuisance_access_count":0,"formal_metadata_access_count":0,"formal_hash_only_byte_access_count":0,"formal_decode_count":0,"formal_outcome_access_count":0,"model_load_count":0,"GPU_CUDA_count":0,"attribution_count":0,"probe_training_count":0,"occlusion_count":0,"historical_exposure_acknowledged":True}))
    print(json.dumps({"status":"PASS","root":ROOT.relative_to(REPO).as_posix(),"thresholds":spec["pre_power"]["effect_thresholds"]},sort_keys=True))
if __name__=="__main__": main()
