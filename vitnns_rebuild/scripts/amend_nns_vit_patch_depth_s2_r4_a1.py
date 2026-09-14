#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,os,sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"));from vitnns import nns_patch_depth_s2_r4 as core
ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r4_20260914";SPEC=REPO/"config/nns_vit_patch_depth/s2_r4_pre_power_protocol.json";SCHEMA=REPO/"config/nns_vit_patch_depth/s2_r4_artifact_envelope.schema.json";DOC=REPO/"docs/nns_vit_patch_depth_s2_r4_a1_implementation_amendment_20260914.md"
def require(c:bool,m:str)->None:
    if not c:raise ValueError(m)
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(v:Any)->str:return hashlib.sha256(canonical(v)).hexdigest()
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text());u=dict(v);d=u.pop("canonical_digest");require(d==digest(u),"canonical mismatch");return v
def rec(p:Path)->dict[str,Any]:return{"path":p.relative_to(REPO).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
def signed(t:str,status:str,payload:dict[str,Any])->dict[str,Any]:
    v={"schema_version":1,"study_id":"NNS_VIT_PATCH_DEPTH_SEMANTICS","stage_id":"S2-R4","run_id":ROOT.name,"document_type":t,"status":status,"payload":payload};v["canonical_digest"]=digest(v);return v
def write(name:str,t:str,status:str,payload:dict[str,Any])->None:
    p=ROOT/name;require(not p.exists(),"refusing overwrite");fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(json.dumps(signed(t,status,payload),sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
def main()->None:
    spec=json.loads(SPEC.read_text());seal=read(ROOT/"pre_power_effect_threshold_seal.json");require(seal["payload"]["protocol"]["sha256"]==sha(SPEC),"pre-power drift");old=read(ROOT/"power_and_precision_simulation_report.json");n=read(ROOT/"pilot_nuisance_extraction_report.json")["payload"];variances=n["sample_variances"];scales=dict(n["conservative_scales"]);scales["D2"]=max(max(math.sqrt(v) for v in variances.values()),math.sqrt(2.5*variances["D2"]));corr=n["four_endpoint_class_level_correlation_matrix"];thresholds=spec["pre_power"]["effect_thresholds"];alts=spec["pre_power"]["design_alternative"];margins={"P1":alts["P1"]-thresholds["P1"],"P2":alts["P2"]-thresholds["P2"],"D1":alts["D1_absolute"]-thresholds["D1"],"D2":alts["D2_absolute"]-thresholds["D2"]};results=[]
    for count in spec["pre_power"]["candidate_class_counts"]:
        r=core.simulate_power(count,scales,corr,margins,n["missingness"]["conservative"],spec["pre_power"]["power_replicates"]);r["power_gate"]=all(v>=.9 for v in r["power"].values());r["precision_gate"]=all(r["precision_half_width"][k]<=thresholds[k] for k in core.ENDPOINTS);r["validity_gate"]=r["conservative_valid_images"]>=max(200,math.ceil(.9*r["scheduled_images"]));results.append(r)
    cal=old["payload"]["calibration"];eligible=[r for r in results if r["power_gate"] and r["precision_gate"] and r["validity_gate"] and cal["type_I_gate_pass"] and cal["coverage_gate_pass"]];selected=eligible[0]["class_count"] if eligible else None;status="PASS" if selected else "BLOCKED_SCIENTIFIC_VALIDITY"
    amendment={"amendment_id":"S2-R4-A1","status":"PASS","pre_power_thresholds_changed":False,"scientific_algorithm_changed":False,"implementation_correction":"D2_scale_is_max_largest_empirical_endpoint_SD_and_sqrt_2.5_times_D2_variance","schema_correction":"closed_signed_artifact_envelope_plus_exact_key_checks_in_validator","old_reports_immutable":True,"old_power_report":rec(ROOT/"power_and_precision_simulation_report.json"),"schema":rec(SCHEMA),"document":rec(DOC),"implementation":rec(Path(__file__).resolve())};write("s2_r4_a1_implementation_amendment.json","s2_r4_a1_implementation_amendment","PASS",amendment)
    write("power_and_precision_simulation_report_a1.json","s2_r4_a1_power_and_precision_simulation_report",status,{"status":status,"supersedes":rec(ROOT/"power_and_precision_simulation_report.json"),"thresholds":thresholds,"design_alternatives":alts,"margins_beyond_threshold":margins,"conservative_scales":scales,"candidate_results":results,"calibration":cal,"selected_class_count":selected,"selection_rule":spec["pre_power"]["selection_rule"],"pilot_endpoint_direction_used":False})
    write("exact_sample_size_decision_a1.json","s2_r4_a1_exact_sample_size_decision",status,{"status":status,"supersedes":rec(ROOT/"exact_sample_size_decision.json"),"selected_class_count":selected,"exact_formal_image_counts":None if selected is None else {"probe_train":6*selected,"probe_validation":2*selected,"confirmatory_holdout":2*selected},"maximum_candidate_evaluated":872,"maximum_candidate_result":results[-1],"reason":"smallest_candidate_meets_all_gates" if selected else "all_872_classes_fail_frozen_scientific_power_or_precision_gate","cost_used_to_reduce_N":False})
    print(json.dumps({"status":status,"D2_conservative_scale":scales["D2"],"D2_power_at_872":results[-1]["power"]["D2"]},sort_keys=True))
if __name__=="__main__":main()
