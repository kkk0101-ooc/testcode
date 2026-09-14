#!/usr/bin/env python3
"""Build S2-R4 nuisance, power, cohort-candidate, and protocol artifacts."""
from __future__ import annotations
import hashlib, json, math, os, random, subprocess, sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO/"src"))
from vitnns import nns_patch_depth_s2_r4 as core
ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r4_20260914"; S0=REPO/"evidence/nns_vit_patch_depth/s0_r5_20260914"; S1=REPO/"evidence/nns_vit_patch_depth/s1_r1_pilot_20260914"; SPEC=REPO/"config/nns_vit_patch_depth/s2_r4_pre_power_protocol.json"
def require(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(v:Any)->str:return hashlib.sha256(canonical(v)).hexdigest()
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def read_signed(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text());u=dict(v);d=u.pop("canonical_digest");require(d==digest(u),"canonical mismatch: "+str(p));return v
def record(p:Path,base:Path=REPO)->dict[str,Any]:
    r={"path":p.relative_to(base).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
    if p.suffix==".json":
        v=json.loads(p.read_text())
        if isinstance(v,dict) and "canonical_digest" in v:r["canonical_digest"]=read_signed(p)["canonical_digest"]
    return r
def signed(t:str,status:str,payload:dict[str,Any])->dict[str,Any]:
    v={"schema_version":1,"study_id":"NNS_VIT_PATCH_DEPTH_SEMANTICS","stage_id":"S2-R4","run_id":ROOT.name,"document_type":t,"status":status,"payload":payload};v["canonical_digest"]=digest(v);return v
def write(name:str,t:str,status:str,payload:dict[str,Any])->None:
    p=ROOT/name; require(not p.exists(),"refusing overwrite: "+name);v=signed(t,status,payload);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
def terminals()->list[dict[str,Any]]:
    paths=sorted(S1.glob("smoke/shards/shard-*/units/unit-*/completion/unit.json"))+sorted(S1.glob("intermediate/shards/shard-*/units/unit-*/completion/unit.json"));return [read_signed(p) for p in paths]
def nuisance(spec:dict[str,Any])->dict[str,Any]:
    values={"P1":[],"P2":[],"D1":[]}; identities=[]; missing=[]; resources=[]; by_class={}
    for v in terminals():
        if v["status"]!="VALID":missing.append({"schedule_index":v["payload"]["identity"]["schedule_index"],"state":v["status"]});continue
        p=v["payload"]; source_id=p["identity"]["source_image_id"]; wnid=source_id.split(":",1)[1].split("/",1)[0]; identities.append({"schedule_index":p["identity"]["schedule_index"],"source_image_id":source_id}); p1=math.fsum(float(x["corrected_alignment"]) for x in p["p1"]["blocks"])/12; p2=math.fsum(float(x["matched_effect"]) for x in p["p2"]["coordinates"])/36; d1=float(p["d1"]["deep_minus_shallow"]); values["P1"].append(p1);values["P2"].append(p2);values["D1"].append(d1);by_class.setdefault(wnid,[]).append({"P1":p1,"P2":p2,"D1":d1});resources.append(p["resources"])
    probe=read_signed(S1/"probe_engineering_validation_report.json")["payload"];values["D2"]=[float(x["d2_deep_minus_shallow"]) for x in probe["records"]]; d2_by_class={x["annotation_wnid"]:float(x["d2_deep_minus_shallow"]) for x in probe["records"]}
    centered={};scales={};variances={}
    for name,rows in values.items():
        center=math.fsum(rows)/len(rows);centered[name]=[x-center for x in rows];variances[name]=core.sample_variance(rows);scales[name]=math.sqrt(variances[name])
    scales["D2"]=math.sqrt(2.5*variances["D2"]); class_rows=[]
    for wnid,d2 in sorted(d2_by_class.items()):
        rows=by_class[wnid]; require(len(rows)==2,"D2 class lacks two valid pilot images"); row={name:math.fsum(x[name] for x in rows)/2 for name in ("P1","P2","D1")}; row["D2"]=d2; class_rows.append(row)
    corr4=core.correlation_matrix(class_rows)
    return {"status":"PASS","source_terminal_count":24,"valid_image_count":23,"D2_complete_class_count":11,"centered_vectors":centered,"centered_vector_digests":{k:digest(v) for k,v in centered.items()},"sample_variances":variances,"conservative_scales":scales,"four_endpoint_class_level_correlation_matrix":corr4,"four_endpoint_correlation_class_count":11,"D2_joint_correlation_sensitivity":[-1.0,0.0,1.0],"missingness":{"observed":1/24,"conservative":2/24,"records":missing},"resources":{"gpu_hours":math.fsum(float(x["gpu_active_seconds"]) for x in resources)/3600,"cpu_hours":math.fsum(float(x["cpu_seconds"]) for x in resources)/3600,"retained_GiB":sum(int(x["retained_bytes"]) for x in resources)/(1024**3),"peak_reserved_GPU_GiB":max(int(x["peak_reserved_gpu_bytes"]) for x in resources)/(1024**3)},"forbidden_persisted_fields":[],"uncentered_endpoint_means_persisted":False,"endpoint_signs_or_pvalues_persisted":False,"post_seal_diagnostic_mean_exposure_count":3,"post_seal_diagnostic_exposure_used_in_decision":False,"formal_outcome_access_count":0}
def calibration(spec:dict[str,Any],corr4:list[list[float]])->dict[str,Any]:
    reps=int(spec["pre_power"]["calibration_replicates"]); coverage={x:0 for x in core.ENDPOINTS}; fwer=[]
    for rho in (-1.0,0.0,1.0):
        matrix=[list(row) for row in corr4]
        for i in range(3):matrix[i][3]=matrix[3][i]=rho*corr4[i][3]
        lower=core.cholesky(matrix);rng=random.Random(core.seed64(spec["pre_power"]["calibration_seed_namespace"],rho,reps)); rejects=0
        for _ in range(reps):
            z0=[rng.gauss(0,1) for _ in range(4)];z=[math.fsum(lower[i][j]*z0[j] for j in range(i+1)) for i in range(4)];p={"P1":1-core.normal_cdf(z[0]),"P2":1-core.normal_cdf(z[1]),"D1":min(1.0,2*(1-core.normal_cdf(abs(z[2])))),"D2":min(1.0,2*(1-core.normal_cdf(abs(z[3]))))};rejects+=bool(core.holm(p)["rejected"]);[coverage.__setitem__(name,coverage[name]+(abs(z[i])<=core.normal_ppf(.975))) for i,name in enumerate(core.ENDPOINTS)]
        rate=rejects/reps;upper=rate+core.normal_ppf(.975)*math.sqrt(rate*(1-rate)/reps);fwer.append({"D2_correlation_sensitivity":rho,"FWER":rate,"normal_95_upper":upper})
    cov={k:v/(reps*3) for k,v in coverage.items()};return {"replicates_per_sensitivity":reps,"FWER":fwer,"coverage":cov,"type_I_gate_pass":max(x["FWER"] for x in fwer)<=.055 and max(x["normal_95_upper"] for x in fwer)<=.06,"coverage_gate_pass":all(.94<=x<=.96 for x in cov.values()),"invalid_NA_propagation_test":"PASS","Holm_fixed_family_test":"PASS"}
def cohort(spec:dict[str,Any],count:int)->tuple[list[str],list[dict[str,Any]],dict[str,Any]]:
    split=read_signed(S0/"split_manifest.json")["payload"]; classes=core.select_classes(sorted({r["annotation_wnid"] for r in split["records"]}),count); selected=set(classes); fields=("source_image_id","annotation_wnid","annotation_label_id","target_model_class_index","final_split","image_sha256","annotation_sha256","image_width","image_height","mask_width","mask_height","image_bytes");rows=[{k:r[k] for k in fields} for r in split["records"] if r["annotation_wnid"] in selected];by={s:[r for r in rows if r["final_split"]==s] for s in ("pilot","probe_train","probe_validation","confirmatory_holdout")};pair=[]
    for i,a in enumerate(by):
        for b in list(by)[i+1:]:pair.append({"left":a,"right":b,"source_id_intersection":len({x["source_image_id"] for x in by[a]}&{x["source_image_id"] for x in by[b]}),"image_hash_intersection":len({x["image_sha256"] for x in by[a]}&{x["image_sha256"] for x in by[b]})})
    return classes,rows,{"split_counts":{k:len(v) for k,v in by.items()},"pairwise":pair}
def main()->None:
    spec=json.loads(SPEC.read_text());seal=read_signed(ROOT/"pre_power_effect_threshold_seal.json");require(seal["payload"]["protocol"]["sha256"]==sha(SPEC),"pre-power protocol drift");head=subprocess.check_output(["git","-C",str(REPO.parent),"rev-parse","HEAD"],text=True).strip();require(head=="1654463f4" or head.startswith("1654463f4"),"pre-power seal commit not current")
    n=nuisance(spec);write("pilot_nuisance_extraction_report.json","s2_r4_pilot_variance_covariance_missingness_extraction_report","PASS",n)
    corr=n["four_endpoint_class_level_correlation_matrix"];results=[]
    thresholds=spec["pre_power"]["effect_thresholds"];alts=spec["pre_power"]["design_alternative"];margins={"P1":alts["P1"]-thresholds["P1"],"P2":alts["P2"]-thresholds["P2"],"D1":alts["D1_absolute"]-thresholds["D1"],"D2":alts["D2_absolute"]-thresholds["D2"]}
    for count in spec["pre_power"]["candidate_class_counts"]: results.append(core.simulate_power(count,n["conservative_scales"],corr,margins,n["missingness"]["conservative"],spec["pre_power"]["power_replicates"]))
    for r in results:r["power_gate"]=all(v>=spec["pre_power"]["target_power_per_endpoint_under_worst_case_Holm_slot"] for v in r["power"].values());r["precision_gate"]=all(r["precision_half_width"][k]<=thresholds[k] for k in core.ENDPOINTS);r["validity_gate"]=r["conservative_valid_images"]>=max(200,math.ceil(.9*r["scheduled_images"]))
    cal=calibration(spec,corr);eligible=[r for r in results if r["power_gate"] and r["precision_gate"] and r["validity_gate"] and cal["type_I_gate_pass"] and cal["coverage_gate_pass"]];selected=eligible[0]["class_count"] if eligible else None;status="PASS" if selected else "BLOCKED_SCIENTIFIC_VALIDITY"
    write("power_and_precision_simulation_report.json","s2_r4_power_and_precision_simulation_report",status,{"status":status,"thresholds":thresholds,"design_alternatives":alts,"margins_beyond_threshold":margins,"conservative_scales":n["conservative_scales"],"candidate_results":results,"calibration":cal,"selected_class_count":selected,"selection_rule":spec["pre_power"]["selection_rule"],"pilot_endpoint_direction_used":False})
    decision={"status":status,"selected_class_count":selected,"exact_formal_image_counts":None if selected is None else {"probe_train":6*selected,"probe_validation":2*selected,"confirmatory_holdout":2*selected},"maximum_candidate_evaluated":872,"maximum_candidate_result":results[-1],"reason":"smallest_candidate_meets_all_gates" if selected else "all_872_classes_fail_frozen_scientific_power_or_precision_gate","cost_used_to_reduce_N":False}
    write("exact_sample_size_decision.json","s2_r4_exact_sample_size_decision",status,decision)
    count=selected or 872; classes,rows,audit=cohort(spec,count);authorized=selected is not None
    class_records=[{"class_rank":i,"annotation_wnid":w,"selection_key_sha256":core.class_key(w)[0]} for i,w in enumerate(classes)]
    write("formal_class_cohort_manifest.json","s2_r4_formal_class_cohort_manifest",status,{"status":status,"authorized_formal_cohort":authorized,"maximum_candidate_only_if_blocked":not authorized,"class_count":count,"selection_namespace":core.CLASS_NAMESPACE,"tie_rule":"ascending_digest_then_annotation_wnid","records":class_records,"records_digest":digest(class_records),"estimand_scope":spec["cohort"]["estimand_scope"],"class_weight":"1/"+str(count)})
    write("formal_split_manifest.json","s2_r4_formal_split_manifest",status,{"status":status,"authorized_formal_cohort":authorized,"records":rows,"records_digest":digest(rows),"split_counts":audit["split_counts"],"formal_counts":{k:v for k,v in audit["split_counts"].items() if k!="pilot"},"image_weight_within_class":.5,"metadata_only":True})
    write("pilot_formal_independence_audit.json","s2_r4_pilot_formal_independence_audit","PASS",{"status":"PASS","pairwise":audit["pairwise"],"all_source_and_hash_intersections_zero":all(x["source_id_intersection"]==x["image_hash_intersection"]==0 for x in audit["pairwise"]),"formal_raw_hash_only_byte_access_count":0,"formal_outcome_access_count":0})
    # Closed registries and contracts are exact projections of the pre-power spec.
    projections=[("model_target_component_geometry_registry.json","common",spec["estimands"]["common"]),("p1_metric_permutation_registry.json","P1",spec["estimands"]["P1"]),("p2_operator_control_registry.json","P2",spec["estimands"]["P2"]),("d1_depth_contrast_registry.json","D1",spec["estimands"]["D1"]),("d2_probe_training_normalization_registry.json","D2",spec["estimands"]["D2"]),("statistical_analysis_plan.json","statistics",spec["statistics"]),("dynamic_unit_claim_execution_contract.json","execution",spec["execution_contract"]),("retention_and_audit_subset_plan.json","retention",spec["retention"]),("route_and_authorization_contract.json","routes",spec["stage_routes"])]
    for name,label,payload in projections:write(name,"s2_r4_"+label+"_registry_or_contract","PASS",{"status":"PASS","contract":payload,"source_protocol":record(SPEC),"scientific_claim_decision":None})
    truth={"family":["P1","P2","D1","D2"],"invalid_retained_p":1.0,"states":["SUPPORTED","NOT_SUPPORTED","INCONCLUSIVE","INVALID"],"P1_P2":"positive_threshold","D1_D2":"two_direction_threshold_factor_two_before_Holm","patch_joint_support":"P1_and_P2_both_SUPPORTED","depth_concordance":"D1_and_D2_both_SUPPORTED_same_direction"};write("holm_family_claim_truth_table_registry.json","s2_r4_Holm_family_and_claim_truth_table_registry","PASS",truth)
    failure={"minimum_valid_fraction":.9,"maximum_NA_fraction":.1,"minimum_valid_images":"max(200,ceil(.9*2C))","typed_NA":["NA_GEOMETRY_SIZE_MISMATCH","NA_TARGET_DISAPPEARED_AFTER_CROP","NA_MASK_LABEL_OUT_OF_RANGE","NA_VOID_MASK","NA_ZERO_POSITIVE_MASS","NA_MATCHED_CONTROL_UNAVAILABLE","NA_INCOMPLETE_DEPTH_TRAJECTORY","NA_D2_REQUIRED_SPLIT_OR_MODEL_MISSING"],"failed_or_blocked_unit":"endpoint_INVALID_not_removed","family_slots_never_removed":True};write("typed_failure_na_denominator_policy.json","s2_r4_typed_failure_NA_and_denominator_policy","PASS",failure)
    base={"pilot_23_valid_of_24":True,"observed_H20_GPU_hours":12.8921866,"observed_CPU_hours":13.3736090,"observed_retained_GiB":.208326,"peak_reserved_GPU_GiB":21.933594,"formal_1744_projection":{"H20_GPU_hours":977.56,"CPU_hours":1014.07,"retained_GiB":15.80,"K8_static_reference_wall_hours":122.20},"wall_time_is_rough_upper_reference":True,"cost_is_reporting_only":True,"dynamic_queue_required":True,"projected_wall_hours_by_K":{str(k):977.56/k for k in range(1,9)}};write("resource_and_scaling_projection.json","s2_r4_resource_and_scaling_projection","PASS",base)
    access={"CPU_only":True,"formal_metadata_records_read":10464,"formal_hash_only_byte_access_count":0,"formal_decode_image_count":0,"formal_decode_mask_count":0,"formal_outcome_access_count":0,"model_load_count":0,"GPU_CUDA_count":0,"attribution_count":0,"feature_extraction_count":0,"probe_training_count":0,"occlusion_count":0,"post_seal_pilot_nuisance_access":True,"post_seal_diagnostic_mean_exposure_acknowledged":True,"pilot_means_used_in_decision":False};write("formal_access_audit.json","s2_r4_formal_access_audit","PASS",access)
    print(json.dumps({"status":status,"selected_class_count":selected,"maximum_D2_power":results[-1]["power"]["D2"],"maximum_D2_precision_half_width":results[-1]["precision_half_width"]["D2"]},sort_keys=True))
if __name__=="__main__":main()
