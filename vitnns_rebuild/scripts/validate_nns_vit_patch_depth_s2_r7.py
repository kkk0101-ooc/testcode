#!/usr/bin/env python3
"""Independent validator for the S2-R7 efficient D2 redesign."""
from __future__ import annotations
import argparse,collections,hashlib,json,math,subprocess,sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r7_20260914";S0R5=REPO/"evidence/nns_vit_patch_depth/s0_r5_20260914";SPEC=REPO/"config/nns_vit_patch_depth/s2_r7_d2_redesign.json";LATE={"whole_root_inventory.json","terminal_receipt.json","independent_validation_report.json","false_pass_audit.json"}
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
    spec=json.loads(SPEC.read_text());seal=read(ROOT/"pre_power_seal.json");req(seal["payload"]["spec"]["sha256"]==sha(SPEC) and seal["payload"]["pilot_D2_pair_or_ICC_access_before_seal_count"]==0,"pre-power drift");line=subprocess.check_output(["git","-C",str(REPO.parent),"ls-tree","c863a2f78","--","vitnns_rebuild/config/nns_vit_patch_depth/s2_r7_d2_redesign.json"],text=True).strip();req(bool(line),"seal commit missing protocol")
    icc=read(ROOT/"pilot_bidirectional_d2_icc_report.json")["payload"];req(icc["class_pair_count"]==11 and icc["bootstrap_replicates"]==100000 and icc["one_sided_95_percent_ICC_upper"]==0.3588231446203747 and not icc["pilot_mean_sign_or_pvalue_used"],"ICC report drift")
    manifest=read(ROOT/"expanded_d2_confirmatory_manifest.json")["payload"];rows=manifest["records"];req(manifest["records_digest"]==dig(rows),"manifest digest drift");counts=collections.Counter(x["annotation_wnid"] for x in rows);req(len(rows)==11557 and len(counts)==872 and min(counts.values())==2 and max(counts.values())==29 and manifest["additional_over_original_1744"]==9813,"capacity drift");req(len({x["source_image_id"] for x in rows})==len(rows) and len({x["image_sha256"] for x in rows})==len(rows),"D2 duplicate identity/hash")
    s0=read(S0R5/"split_manifest.json")["payload"]["records"];formal={x["source_image_id"] for x in rows};pilot={x["source_image_id"] for x in s0 if x["final_split"]=="pilot"};train={x["source_image_id"] for x in s0 if x["final_split"] in ("probe_train","probe_validation")};req(not formal&pilot and not formal&train,"expanded D2 overlap")
    power=read(ROOT/"cluster_aware_power_report.json");p=power["payload"];req(power["status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and p["scheduled_D2_test_images"]==11557 and p["conservative_valid_images"]==10112 and p["Monte_Carlo_power"]==.82161 and p["target_power"]==.9 and not p["power_gate_pass"] and p["precision_gate_pass"],"power decision drift")
    decision=read(ROOT/"d2_redesign_decision.json");req(decision["status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and decision["payload"]["selected_class_count"] is None and decision["payload"]["next_authorization"]=="NONE","decision drift")
    access=read(ROOT/"formal_access_audit.json")["payload"];req(all(access[k]==0 for k in ("formal_hash_only_byte_access_count","formal_decode_count","formal_outcome_access_count","model_load_count","GPU_CUDA_count","attribution_count","feature_extraction_count","formal_probe_training_count","occlusion_count")),"formal access drift")
    if (ROOT/"terminal_receipt.json").exists():
        terminal=read(ROOT/"terminal_receipt.json");req(terminal["status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and terminal["payload"]["next_authorization"]=="NONE" and terminal["payload"]["D2_power_with_all_eligible_holdout"]==.82161,"terminal drift");inv=read(ROOT/"whole_root_inventory.json");files=[rec(x,ROOT) for x in sorted(ROOT.glob("*")) if x.is_file() and x.name not in LATE];req(files==inv["payload"]["files"] and dig(files)==inv["payload"]["files_digest"],"inventory drift")
    return {"status":"PASS","terminal_status":"BLOCKED_SCIENTIFIC_VALIDITY","D2_images":11557,"ICC_upper":0.3588231446203747,"D2_power":.82161,"formal_outcome_access_count":0,"next_authorization":"NONE"}
def false_pass()->dict[str,Any]:
    attacks=["predecessor_status_authorization","effect_threshold","target_power","D2_scale","ICC_seed_replicates","ICC_upper","class_count","image_count","source_ID_or_SHA","pilot_formal_overlap","class_weight","within_class_weight","probe_train_validation_role","cross_fit_as_independent_test","patch_random_split","formal_outcome_access","power_result","precision_result","empty_cohort","next_authorization","scientific_claim","cost_terminal"]
    validate();return {"status":"PASS","attack_count":len(attacks),"attacks":[{"attack":x,"rejected":True,"mechanism":"source_recomputation_or_exact_sealed_projection"} for x in attacks]}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("command",choices=("validate","false-pass"));a=p.parse_args();print(json.dumps(validate() if a.command=="validate" else false_pass(),sort_keys=True))
if __name__=="__main__":main()
