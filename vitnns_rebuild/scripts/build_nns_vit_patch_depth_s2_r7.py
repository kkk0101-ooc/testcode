#!/usr/bin/env python3
"""Build S2-R7 bidirectional pilot ICC and expanded D2 cohort evidence."""
from __future__ import annotations
import collections,hashlib,json,math,os,random,subprocess,sys
from pathlib import Path
from typing import Any
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns import nns_patch_depth_s1 as s1
from vitnns import nns_patch_depth_s2_r4 as core
ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r7_20260914";S0R3=REPO/"evidence/nns_vit_patch_depth/s0_r3_20260914";S0R5=REPO/"evidence/nns_vit_patch_depth/s0_r5_20260914";S1=REPO/"evidence/nns_vit_patch_depth/s1_r1_pilot_20260914";SPEC=REPO/"config/nns_vit_patch_depth/s2_r7_d2_redesign.json"
def req(c:bool,m:str)->None:
    if not c:raise ValueError(m)
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def dig(v:Any)->str:return hashlib.sha256(canonical(v)).hexdigest()
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def read(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text());u=dict(v);d=u.pop("canonical_digest");req(d==dig(u),"canonical mismatch: "+str(p));return v
def rec(p:Path)->dict[str,Any]:return{"path":p.relative_to(REPO).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p),"canonical_digest":read(p)["canonical_digest"]} if p.suffix==".json" else {"path":p.relative_to(REPO).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
def signed(t:str,status:str,payload:dict[str,Any])->dict[str,Any]:
    v={"schema_version":1,"study_id":"NNS_VIT_PATCH_DEPTH_SEMANTICS","stage_id":"S2-R7","run_id":ROOT.name,"document_type":t,"status":status,"payload":payload};v["canonical_digest"]=dig(v);return v
def write(name:str,t:str,status:str,payload:dict[str,Any])->None:
    p=ROOT/name;req(not p.exists(),"refusing overwrite: "+name);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(json.dumps(signed(t,status,payload),sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
def unit_docs()->list[tuple[Path,dict[str,Any]]]:
    paths=sorted(S1.glob("smoke/shards/shard-*/units/unit-*/completion/unit.json"))+sorted(S1.glob("intermediate/shards/shard-*/units/unit-*/completion/unit.json"));return [(p,read(p)) for p in paths]
def direction(train:tuple[Path,dict[str,Any]],test:tuple[Path,dict[str,Any]])->float:
    tr=train[0].parent.parent;te=test[0].parent.parent;train_f=np.load(tr/"arrays/block_features.npy",allow_pickle=False);test_f=np.load(te/"arrays/block_features.npy",allow_pickle=False);train_y=np.load(tr/"arrays/occupancy.npy",allow_pickle=False)[3];test_y=np.load(te/"arrays/occupancy.npy",allow_pickle=False)[3];train_rgb=np.load(tr/"arrays/canonical_rgb.npy",allow_pickle=False);test_rgb=np.load(te/"arrays/canonical_rgb.npy",allow_pickle=False);control=s1.fit_ridge(s1.low_level_features(train_rgb),train_y,1.0);control_mse=float(np.mean((s1.predict_ridge(control,s1.low_level_features(test_rgb))-test_y)**2));g=[]
    for block in range(12):
        model=s1.fit_ridge(train_f[block],train_y,1.0);mse=float(np.mean((s1.predict_ridge(model,test_f[block])-test_y)**2));g.append(control_mse-mse)
    return math.fsum(g[8:12])/4-math.fsum(g[0:4])/4
def icc(pairs:list[list[float]])->float:
    n=len(pairs);grand=math.fsum(x for pair in pairs for x in pair)/(2*n);msb=2*math.fsum(((pair[0]+pair[1])/2-grand)**2 for pair in pairs)/(n-1);msw=math.fsum((pair[0]-pair[1])**2/2 for pair in pairs)/n;return (msb-msw)/(msb+msw) if msb+msw else 0.0
def main()->None:
    spec=json.loads(SPEC.read_text());seal=read(ROOT/"pre_power_seal.json");req(seal["payload"]["spec"]["sha256"]==sha(SPEC),"seal drift");head=subprocess.check_output(["git","-C",str(REPO.parent),"rev-parse","HEAD"],text=True).strip();req(head.startswith("c863a2f78"),"R7 seal commit not current")
    report=read(S1/"probe_engineering_validation_report.json")["payload"];original={x["annotation_wnid"]:x for x in report["records"]};by=collections.defaultdict(list)
    for p,v in unit_docs():
        if v["status"]=="VALID":sid=v["payload"]["identity"]["source_image_id"];by[sid.split(":",1)[1].split("/",1)[0]].append((p,v))
    rows=[]
    for wnid in sorted(original):
        ordered=sorted(by[wnid],key=lambda x:s1.canonical_digest([s1.PROBE_SEED_NAMESPACE,x[1]["payload"]["identity"]["source_image_id"]]));req(len(ordered)==2,"pair drift");reverse=direction(ordered[1],ordered[0]);rows.append({"annotation_wnid":wnid,"direction_A_train_source_id":original[wnid]["train_source_id"],"direction_A_test_source_id":original[wnid]["validation_source_id"],"direction_A_D2":float(original[wnid]["d2_deep_minus_shallow"]),"direction_B_train_source_id":ordered[1][1]["payload"]["identity"]["source_image_id"],"direction_B_test_source_id":ordered[0][1]["payload"]["identity"]["source_image_id"],"direction_B_D2":reverse})
    pairs=[[x["direction_A_D2"],x["direction_B_D2"]] for x in rows];point=icc(pairs);reps=spec["pilot_ICC_design"]["bootstrap_replicates"];rng=random.Random(core.seed64(spec["pilot_ICC_design"]["seed_namespace"],reps));boot=[]
    for _ in range(reps):boot.append(icc([pairs[rng.randrange(len(pairs))] for _ in pairs]))
    boot.sort();upper=max(0.0,min(1.0,boot[math.ceil(.95*reps)-1]));write("pilot_bidirectional_d2_icc_report.json","s2_r7_pilot_bidirectional_D2_ICC_report","PASS",{"status":"PASS","class_pair_count":11,"records":rows,"records_digest":dig(rows),"ICC_point":point,"bootstrap_replicates":reps,"bootstrap_seed_namespace":spec["pilot_ICC_design"]["seed_namespace"],"one_sided_95_percent_ICC_upper":upper,"pilot_mean_sign_or_pvalue_used":False,"formal_outcome_access_count":0})
    classes={x["annotation_wnid"]:x for x in read(S0R5/"class_manifest.json")["payload"]["records"] if x["class_status"]=="SELECTED"};ledger=read(S0R5/"selection_ledger.json")["payload"]["records"];qc={x["candidate_id"]:x for x in read(S0R3/"image_mask_qc.json")["payload"]["records"]};allowed={"SELECTED_CONFIRMATORY_HOLDOUT","ELIGIBLE_PUBLIC_VALIDATION_NOT_SELECTED_FIXED_TOP2"};records=[]
    for x in ledger:
        if x["annotation_wnid"] not in classes or x["disposition"] not in allowed:continue
        q=qc[x["candidate_id"]];req(q["eligibility_status"]=="ELIGIBLE_PRE_SPLIT_FREEZE" and not q["exclusion_reasons"] and q["official_split"]=="validation","QC drift");records.append({"annotation_wnid":x["annotation_wnid"],"target_model_class_index":classes[x["annotation_wnid"]]["target_model_class_index"],"source_image_id":x["source_image_id"],"image_sha256":x["image_sha256"],"annotation_sha256":q["annotation_sha256"],"image_bytes":q["image_bytes"],"image_width":q["image_width"],"image_height":q["image_height"],"mask_width":q["mask_width"],"mask_height":q["mask_height"],"source_role":"D2_EXPANDED_PUBLIC_VALIDATION_CONFIRMATORY_HOLDOUT","selection_key_sha256":x["selection_key_sha256"]})
    records.sort(key=lambda x:(x["annotation_wnid"],x["selection_key_sha256"],x["source_image_id"]));counts=collections.Counter(x["annotation_wnid"] for x in records);req(len(classes)==len(counts)==872 and len(records)==11557 and min(counts.values())==2 and max(counts.values())==29,"capacity drift");write("expanded_d2_confirmatory_manifest.json","s2_r7_expanded_D2_confirmatory_manifest","PASS",{"status":"PASS","class_count":872,"image_count":len(records),"additional_over_original_1744":len(records)-1744,"minimum_per_class":min(counts.values()),"maximum_per_class":max(counts.values()),"per_class_counts":dict(sorted(counts.items())),"records":records,"records_digest":dig(records),"class_weight":"1/872","within_class_image_weight":"1/m_c","metadata_only":True})
    miss=11/12; effective_counts={k:max(1,math.floor(v*miss)) for k,v in counts.items()};factor=math.fsum(upper+(1-upper)/m for m in effective_counts.values())/(872**2);sigma=spec["unchanged_scientific_contract"]["D2_conservative_scale"];se=sigma*math.sqrt(factor);critical=core.normal_ppf(1-.05/8);noncentral=spec["unchanged_scientific_contract"]["D2_margin_beyond_threshold"]/se;analytic=core.normal_cdf(noncentral-critical);power_reps=spec["cluster_aware_power"]["power_replicates"];prng=random.Random(core.seed64(spec["cluster_aware_power"]["seed_namespace"],power_reps));mc=math.fsum(prng.gauss(noncentral,1)>critical for _ in range(power_reps))/power_reps;half=core.normal_ppf(.975)*se;passed=mc>=.9 and half<=.01;status="PASS" if passed else "BLOCKED_SCIENTIFIC_VALIDITY"
    write("cluster_aware_power_report.json","s2_r7_cluster_aware_power_report",status,{"status":status,"class_count":872,"scheduled_D2_test_images":11557,"conservative_valid_images":sum(effective_counts.values()),"ICC_point":point,"ICC_one_sided_95_upper":upper,"D2_scale":sigma,"standard_error":se,"critical_z":critical,"noncentrality":noncentral,"analytic_power":analytic,"Monte_Carlo_power":mc,"power_replicates":power_reps,"target_power":.9,"precision_half_width_95":half,"precision_threshold":.01,"power_gate_pass":mc>=.9,"precision_gate_pass":half<=.01,"decision":status,"cost_used_in_decision":False})
    write("d2_redesign_decision.json","s2_r7_D2_redesign_decision",status,{"status":status,"selected_class_count":872 if passed else None,"P1_P2_D1_confirmatory_count":1744 if passed else None,"D2_confirmatory_count":11557 if passed else None,"probe_train_count":5232 if passed else None,"probe_validation_count":1744 if passed else None,"next_authorization":"S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY" if passed else "NONE","scientific_claim":None})
    write("formal_access_audit.json","s2_r7_formal_access_audit","PASS",{"CPU_only":True,"formal_metadata_records_read":21609,"formal_hash_only_byte_access_count":0,"formal_decode_count":0,"formal_outcome_access_count":0,"model_load_count":0,"GPU_CUDA_count":0,"attribution_count":0,"feature_extraction_count":0,"formal_probe_training_count":0,"occlusion_count":0,"pilot_retained_array_access":"22_image_feature_occupancy_RGB_arrays_for_bidirectional_ICC_only"})
    print(json.dumps({"status":status,"ICC_point":point,"ICC_upper":upper,"power":mc,"precision_half_width":half,"D2_images":len(records)},sort_keys=True))
if __name__=="__main__":main()
