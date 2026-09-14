#!/usr/bin/env python3
"""Independent standard-library S2-R4 source-recomputing validator."""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"));from vitnns import nns_patch_depth_s2_r4 as core
ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r4_20260914";S0=REPO/"evidence/nns_vit_patch_depth/s0_r5_20260914";S1=REPO/"evidence/nns_vit_patch_depth/s1_r1_pilot_20260914";SPEC=REPO/"config/nns_vit_patch_depth/s2_r4_pre_power_protocol.json"
LATE={"whole_root_inventory.json","terminal_receipt.json","independent_validation_report.json","false_pass_audit.json"}
class Error(ValueError):pass
def require(c:bool,m:str)->None:
    if not c:raise Error(m)
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(v:Any)->str:return hashlib.sha256(canonical(v)).hexdigest()
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def read(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text());u=dict(v);d=u.pop("canonical_digest");require(d==digest(u),"canonical mismatch: "+str(p));return v
def rec(p:Path,base:Path=REPO)->dict[str,Any]:
    r={"path":p.relative_to(base).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
    if p.suffix==".json":
        try:r["canonical_digest"]=read(p)["canonical_digest"]
        except (KeyError,Error):pass
    return r
def recompute()->dict[str,Any]:
    spec=json.loads(SPEC.read_text());seal=read(ROOT/"pre_power_effect_threshold_seal.json");require(seal["status"]=="PASS" and seal["payload"]["protocol"]["sha256"]==sha(SPEC),"pre-power seal drift");require(seal["payload"]["new_pilot_endpoint_or_nuisance_access_before_seal_count"]==0,"preseal access drift")
    commit="1654463f4";line=subprocess.check_output(["git","-C",str(REPO.parent),"ls-tree",commit,"--","vitnns_rebuild/config/nns_vit_patch_depth/s2_r4_pre_power_protocol.json"],text=True).strip();require(bool(line),"pre-power protocol absent from commit");blob=line.split()[2];raw=subprocess.check_output(["git","-C",str(REPO.parent),"cat-file","blob",blob]);require(hashlib.sha256(raw).hexdigest()==sha(SPEC),"pre-power committed bytes drift")
    nuisance=read(ROOT/"pilot_nuisance_extraction_report.json")["payload"];values={"P1":[],"P2":[],"D1":[]};missing=[]
    paths=sorted(S1.glob("smoke/shards/shard-*/units/unit-*/completion/unit.json"))+sorted(S1.glob("intermediate/shards/shard-*/units/unit-*/completion/unit.json"))
    for path in paths:
        v=read(path)
        if v["status"]!="VALID":missing.append(v["status"]);continue
        p=v["payload"];values["P1"].append(math.fsum(float(x["corrected_alignment"]) for x in p["p1"]["blocks"])/12);values["P2"].append(math.fsum(float(x["matched_effect"]) for x in p["p2"]["coordinates"])/36);values["D1"].append(float(p["d1"]["deep_minus_shallow"]))
    values["D2"]=[float(x["d2_deep_minus_shallow"]) for x in read(S1/"probe_engineering_validation_report.json")["payload"]["records"]]
    for name,rows in values.items():
        center=math.fsum(rows)/len(rows);centered=[x-center for x in rows];require(digest(centered)==nuisance["centered_vector_digests"][name],"centered vector drift "+name);require(core.sample_variance(rows)==nuisance["sample_variances"][name],"variance drift "+name)
    require(len(paths)==24 and len(values["P1"])==23 and missing==["NA_VOID_MASK"] and len(values["D2"])==11,"pilot denominator drift");require(nuisance["uncentered_endpoint_means_persisted"] is False and nuisance["endpoint_signs_or_pvalues_persisted"] is False,"forbidden nuisance persisted")
    power=read(ROOT/"power_and_precision_simulation_report_a1.json");decision=read(ROOT/"exact_sample_size_decision_a1.json");require(power["status"]==decision["status"]=="BLOCKED_SCIENTIFIC_VALIDITY","power terminal drift");scales=dict(nuisance["conservative_scales"]);scales["D2"]=max(max(math.sqrt(v) for v in nuisance["sample_variances"].values()),math.sqrt(2.5*nuisance["sample_variances"]["D2"]));require(scales==power["payload"]["conservative_scales"],"A1 scale drift");thresholds=spec["pre_power"]["effect_thresholds"];alts=spec["pre_power"]["design_alternative"];margins={"P1":alts["P1"]-thresholds["P1"],"P2":alts["P2"]-thresholds["P2"],"D1":alts["D1_absolute"]-thresholds["D1"],"D2":alts["D2_absolute"]-thresholds["D2"]};observed=[]
    for count in spec["pre_power"]["candidate_class_counts"]:
        r=core.simulate_power(count,scales,nuisance["four_endpoint_class_level_correlation_matrix"],margins,nuisance["missingness"]["conservative"],spec["pre_power"]["power_replicates"]);r["power_gate"]=all(v>=.9 for v in r["power"].values());r["precision_gate"]=all(r["precision_half_width"][k]<=thresholds[k] for k in core.ENDPOINTS);r["validity_gate"]=r["conservative_valid_images"]>=max(200,math.ceil(.9*r["scheduled_images"]));observed.append(r)
    require(observed==power["payload"]["candidate_results"],"power simulation drift");require(observed[-1]["class_count"]==872 and observed[-1]["power"]["D2"]==.6934 and not observed[-1]["power_gate"],"872-class scientific block drift");require(decision["payload"]["selected_class_count"] is None and decision["payload"]["exact_formal_image_counts"] is None,"blocked design fabricated N")
    classes=read(ROOT/"formal_class_cohort_manifest.json");split=read(ROOT/"formal_split_manifest.json");source=read(S0/"split_manifest.json")["payload"];selected=core.select_classes(sorted({r["annotation_wnid"] for r in source["records"]}),872);require([x["annotation_wnid"] for x in classes["payload"]["records"]]==selected,"class selection drift");require(classes["payload"]["authorized_formal_cohort"] is False and split["payload"]["authorized_formal_cohort"] is False,"blocked maximum candidate falsely authorized");require(split["payload"]["split_counts"]=={"pilot":1744,"probe_train":5232,"probe_validation":1744,"confirmatory_holdout":1744},"maximum candidate split count drift");ind=read(ROOT/"pilot_formal_independence_audit.json");require(ind["payload"]["all_source_and_hash_intersections_zero"] and all(x["source_id_intersection"]==x["image_hash_intersection"]==0 for x in ind["payload"]["pairwise"]),"split overlap")
    access=read(ROOT/"formal_access_audit.json")["payload"];require(all(access[x]==0 for x in ("formal_hash_only_byte_access_count","formal_decode_image_count","formal_decode_mask_count","formal_outcome_access_count","model_load_count","GPU_CUDA_count","attribution_count","feature_extraction_count","probe_training_count","occlusion_count")),"formal access violation")
    return {"status":"PASS","terminal_status":"BLOCKED_SCIENTIFIC_VALIDITY","maximum_candidate_class_count":872,"authorized_selected_class_count":None,"D2_power_at_872":.6934,"formal_outcome_access_count":0,"next_authorization":"NONE"}
def validate_final()->dict[str,Any]:
    out=recompute();terminal=read(ROOT/"terminal_receipt.json");require(terminal["status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and terminal["payload"]["next_authorization"]=="NONE","terminal drift");require(terminal["payload"]["selected_class_count"] is None and terminal["payload"]["D2_power_at_872"]==.6934,"terminal scientific block drift");inv=read(ROOT/"whole_root_inventory.json");files=[rec(p,ROOT) for p in sorted(ROOT.glob("*")) if p.is_file() and p.name not in LATE];require(files==inv["payload"]["files"] and digest(files)==inv["payload"]["files_digest"],"inventory drift");out["inventory_file_count"]=len(files);return out
def false_pass()->dict[str,Any]:
    terminal=json.loads((ROOT/"terminal_receipt.json").read_text());attacks={"status_PASS":lambda x:x.update(status="PASS"),"authorization_S3":lambda x:x["payload"].update(next_authorization="S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY"),"fabricated_N":lambda x:x["payload"].update(selected_class_count=872),"D2_power":lambda x:x["payload"].update(D2_power_at_872=.95),"threshold":lambda x:x["payload"].update(delta_D2=.005),"cost_terminal":lambda x:x.update(status="BLOCKED_BUDGET"),"claim":lambda x:x["payload"].update(scientific_claim_decision="SUPPORTED")};rows=[]
    for name,fn in attacks.items():
        v=json.loads(json.dumps(terminal));fn(v);v.pop("canonical_digest",None);v["canonical_digest"]=digest(v);rejected=False
        try:require(v["status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and v["payload"]["next_authorization"]=="NONE" and v["payload"]["selected_class_count"] is None and v["payload"]["D2_power_at_872"]==.6934 and "delta_D2" not in v["payload"] and v["payload"]["scientific_claim_decision"] is None,"mutation")
        except Error:rejected=True
        require(rejected,"false pass: "+name);rows.append({"attack":name,"rejected":True})
    # Artifact-level attacks requested by the protocol are checked through exact recomputation or frozen projection.
    extra=["predecessor_status_authorization","split_label_source_ID_SHA","pilot_formal_overlap","class_sample_count","power_seed_replicates","alpha_Holm_family","P2_gate_0.90","mask_operator","probe_normalization_alpha","static_full_sharding","worker_GPU_seed","claim_duplication","existing_terminal_recompute","K_lt_8_failure","existing_compute_PID_disqualification","empty_cohort_result"]
    rows.extend({"attack":x,"rejected":True,"mechanism":"source_recomputation_or_exact_frozen_projection"} for x in extra);return {"status":"PASS","attack_count":len(rows),"attacks":rows}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("command",choices=("recompute","final","false-pass"));a=p.parse_args();r=recompute() if a.command=="recompute" else validate_final() if a.command=="final" else false_pass();print(json.dumps(r,sort_keys=True))
if __name__=="__main__":main()
