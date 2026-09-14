#!/usr/bin/env python3
"""S2-R8 candidate-manifest freezer and sealed one-shot power runner."""
from __future__ import annotations
import argparse,hashlib,json,math,os,random,subprocess,sys
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"));from vitnns import nns_patch_depth_s2_r8 as core
ROOT=REPO/"evidence/nns_vit_patch_depth/s2_r8_20260914";R7=REPO/"evidence/nns_vit_patch_depth/s2_r7_20260914";R4=REPO/"evidence/nns_vit_patch_depth/s2_r4_20260914";SPEC=REPO/"config/nns_vit_patch_depth/s2_r8_run_spec.json"
def req(c:bool,m:str)->None:
    if not c:raise ValueError(m)
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def dig(v:Any)->str:return hashlib.sha256(canonical(v)).hexdigest()
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text());u=dict(v);d=u.pop("canonical_digest");req(d==dig(u),"canonical mismatch: "+str(p));return v
def rec(p:Path,base:Path=REPO)->dict[str,Any]:return{"path":p.relative_to(base).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p),"canonical_digest":read(p)["canonical_digest"]}
def signed(t:str,status:str,payload:dict[str,Any])->dict[str,Any]:
    v={"schema_version":1,"study_id":"NNS_VIT_PATCH_DEPTH_SEMANTICS","stage_id":"S2-R8","run_id":ROOT.name,"document_type":t,"status":status,"payload":payload};v["canonical_digest"]=dig(v);return v
def write(name:str,t:str,status:str,payload:dict[str,Any])->None:
    p=ROOT/name;req(not p.exists(),"refusing overwrite: "+name);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(json.dumps(signed(t,status,payload),sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
def freeze_manifests()->dict[str,Any]:
    req(not ROOT.exists(),"R8 root exists");ROOT.mkdir(parents=True);spec=json.loads(SPEC.read_text());source=read(R7/"expanded_d2_confirmatory_manifest.json")["payload"];by={}
    for row in source["records"]:by.setdefault(row["annotation_wnid"],[]).append(row)
    original=read(REPO/"evidence/nns_vit_patch_depth/s0_r5_20260914/split_manifest.json")["payload"]["records"];original_ids={x["source_image_id"] for x in original if x["final_split"]=="confirmatory_holdout"};prior=set();prior_digest=None;counts={};bindings={}
    for cap in spec["candidate_manifests"]["caps"]:
        records=[]
        for wnid in sorted(by):
            rows=sorted(by[wnid],key=lambda x:(x["selection_key_sha256"],x["source_image_id"]))[:cap]
            req({x["source_image_id"] for x in rows} >= ({x["source_image_id"] for x in by[wnid]}&original_ids),"original two not included")
            for x in rows:
                old=x["source_image_id"] in original_ids;y=dict(x);y["D2_cap"]=cap;y["role_flags"]={"P1":old,"P2":old,"D1":old,"D2":True,"S4_NNS":old,"S4_occlusion":old};records.append(y)
        ids={x["source_image_id"] for x in records};req(ids==original_ids if cap==2 else prior<ids,"candidate nesting or original-m2 binding drift");payload={"status":"FROZEN_CANDIDATE","cap_per_class":cap,"class_count":len(by),"image_count":len(records),"records":records,"records_digest":dig(records),"previous_manifest_records_digest":prior_digest,"strictly_contains_previous":cap>2,"outcome_blind":True,"S3_forward_count":5232+1744+len(records),"S3_increment_fraction":(len(records)-1744)/8720,"S4_image_count":1744};name=f"candidate_m{cap}_manifest.json";write(name,"s2_r8_exact_candidate_manifest","PASS",payload);counts[str(cap)]=len(records);bindings[str(cap)]=rec(ROOT/name,ROOT);prior=ids;prior_digest=payload["records_digest"]
    req(counts=={"2":1744,"3":2614,"4":3482},"candidate counts drift");return {"counts":counts,"bindings":bindings}
def simulate_candidate(cap:int,counts:list[int],spec:dict[str,Any],corr:list[list[float]],scales:dict[str,float])->dict[str,Any]:
    effective=[max(1,math.floor(11*m/12)) for m in counts];se={"P1":scales["P1"]/math.sqrt(1598),"P2":scales["P2"]/math.sqrt(1598),"D1":scales["D1"]/math.sqrt(1598),"D2":core.cluster_se(scales["D2"],spec["unchanged"]["D2_ICC"],effective)};margin={"P1":.02,"P2":.1,"D1":.02,"D2":.01};mu=[margin[x]/se[x] for x in core.ENDPOINTS];reps=spec["power"]["replicates"];scenario_rows=[];marginal_analytic={"P1":1-core.normal_cdf(core.normal_ppf(1-.05/3)-mu[0]),"P2":1-core.normal_cdf(core.normal_ppf(1-.05/3)-mu[1]),"D1":1-core.normal_cdf(core.normal_ppf(1-.05/6)-mu[2]),"D2":1-core.normal_cdf(core.normal_ppf(1-.05/2)-mu[3])};frechet=max(0.,math.fsum(marginal_analytic.values())-3)
    for scenario,matrix in core.correlation_scenarios(corr).items():
        lower=core.cholesky(matrix);rng=random.Random(core.seed64(spec["power"]["seed_namespace"],cap,scenario,reps));gate=joint=0;marg={x:0 for x in core.ENDPOINTS}
        for _ in range(reps):
            base=[rng.gauss(0,1) for _ in range(4)];z=[mu[i]+math.fsum(lower[i][j]*base[j] for j in range(i+1)) for i in range(4)];p=core.pvalues(z);g,j=core.decisions(z);gate+=g;joint+=j
            primary=core.holm_primary({name:p[name] for name in core.PRIMARY}); rejected={row["endpoint"] for row in primary["rows"] if row["reject"]}
            for name in core.PRIMARY:marg[name]+=name in rejected
            marg["D2"]+=p["D2"]<=.05
        scenario_rows.append({"scenario":scenario,"marginal_power":{x:marg[x]/reps for x in core.ENDPOINTS},"gate_open_power":gate/reps,"conditional_D2_power":joint/gate if gate else 0.,"unconditional_joint_power":joint/reps,"unconditional_joint_power_Wilson_lower_95":core.wilson_lower(joint,reps),"joint_success_count":joint,"gate_open_count":gate})
    worst=min(scenario_rows,key=lambda x:(x["unconditional_joint_power"],x["scenario"]));half=core.normal_ppf(.975)*se["D2"];primary_marginal={name:min(row["marginal_power"][name] for row in scenario_rows) for name in core.PRIMARY};return {"cap_per_class":cap,"class_count":872,"scheduled_D2_images":sum(counts),"conservative_valid_D2_images":sum(effective),"standard_errors":se,"marginal_power_analytic":marginal_analytic,"primary_Holm_marginal_power_conservative":primary_marginal,"gate_open_power_conservative":min(x["gate_open_power"] for x in scenario_rows),"conditional_D2_power_conservative":min(x["conditional_D2_power"] for x in scenario_rows),"unconditional_joint_power_conservative":worst["unconditional_joint_power"],"joint_power_Wilson_lower_95_conservative":min(x["unconditional_joint_power_Wilson_lower_95"] for x in scenario_rows),"correlation_independent_Frechet_lower_bound":frechet,"D2_precision_half_width":half,"scenarios":scenario_rows}
def calibrate_strong_fwer(cap:int,ses:dict[str,float],spec:dict[str,Any],corr:list[list[float]])->dict[str,Any]:
    reps=spec["calibration"]["replicates"];margins={"P1":.02,"P2":.1,"D1":.02,"D2":.01};noncentral={name:margins[name]/ses[name] for name in core.ENDPOINTS};configs=[tuple(core.ENDPOINTS[i] for i in range(4) if mask&(1<<i)) for mask in range(1,16)];rows=[];coverage={name:0 for name in core.ENDPOINTS}
    for scenario,matrix in core.correlation_scenarios(corr).items():
        lower=core.cholesky(matrix);rng=random.Random(core.seed64(spec["calibration"]["seed_namespace"],cap,scenario,reps));errors={config:0 for config in configs}
        for _ in range(reps):
            base=[rng.gauss(0,1) for _ in range(4)];noise=[math.fsum(lower[i][j]*base[j] for j in range(i+1)) for i in range(4)]
            for i,name in enumerate(core.ENDPOINTS):coverage[name]+=abs(noise[i])<=core.normal_ppf(.975)
            for true_nulls in configs:
                z=[noise[i] if name in true_nulls else noise[i]+noncentral[name] for i,name in enumerate(core.ENDPOINTS)];p=core.pvalues(z);primary=core.holm_primary({name:p[name] for name in core.PRIMARY});rejected={row["endpoint"] for row in primary["rows"] if row["reject"]}
                if primary["all_reject"] and p["D2"]<=.05:rejected.add("D2")
                errors[true_nulls]+=bool(rejected&set(true_nulls))
        for true_nulls in configs:
            count=errors[true_nulls];rate=count/reps;rows.append({"cap_per_class":cap,"scenario":scenario,"true_nulls":list(true_nulls),"error_count":count,"FWER":rate,"Wilson_upper_95":core.wilson_upper(count,reps)})
    cov={name:coverage[name]/(reps*3) for name in core.ENDPOINTS};return {"replicates_per_configuration_scenario":reps,"true_null_configuration_count":15,"scenario_count":3,"rows":rows,"maximum_FWER":max(row["FWER"] for row in rows),"maximum_Wilson_upper_95":max(row["Wilson_upper_95"] for row in rows),"coverage":cov,"point_gate_pass":max(row["FWER"] for row in rows)<=.0525,"upper_gate_pass":max(row["Wilson_upper_95"] for row in rows)<=.055,"coverage_gate_pass":all(.948<=value<=.952 for value in cov.values())}
def compute()->dict[str,Any]:
    spec=json.loads(SPEC.read_text());seal=read(ROOT/"pre_power_seal.json");req(seal["payload"]["real_power_computed_before_seal"] is False,"seal order");r4=read(R4/"pilot_nuisance_extraction_report.json")["payload"];scales={"P1":r4["conservative_scales"]["P1"],"P2":r4["conservative_scales"]["P2"],"D1":r4["conservative_scales"]["D1"],"D2":spec["unchanged"]["D2_scale"]};corr=r4["four_endpoint_class_level_correlation_matrix"];results=[]
    for cap in spec["candidate_manifests"]["caps"]:
        m=read(ROOT/f"candidate_m{cap}_manifest.json")["payload"];counts=list(__import__("collections").Counter(x["annotation_wnid"] for x in m["records"]).values());results.append(simulate_candidate(cap,counts,spec,corr,scales))
    calibrations=[]
    for r in results:
        calibration=calibrate_strong_fwer(r["cap_per_class"],r["standard_errors"],spec,corr);calibrations.append(calibration);r["strong_FWER_calibration"]={key:calibration[key] for key in ("maximum_FWER","maximum_Wilson_upper_95","coverage","point_gate_pass","upper_gate_pass","coverage_gate_pass")};r["all_gates_pass"]=min(r["primary_Holm_marginal_power_conservative"].values())>=.8 and r["gate_open_power_conservative"]>=.8 and r["conditional_D2_power_conservative"]>=.8 and r["unconditional_joint_power_conservative"]>=.8 and r["joint_power_Wilson_lower_95_conservative"]>=.8 and r["correlation_independent_Frechet_lower_bound"]>=.8 and r["D2_precision_half_width"]<=.01 and calibration["point_gate_pass"] and calibration["upper_gate_pass"] and calibration["coverage_gate_pass"]
    selected=next((r for r in results if r["all_gates_pass"]),None);status="PASS" if selected else "BLOCKED_SCIENTIFIC_VALIDITY";write("strong_FWER_calibration_report.json","s2_r8_strong_FWER_calibration_report","PASS",{"status":"PASS","analytic_proof":spec["gatekeeping"]["strong_FWER_proof"],"all_15_nonempty_true_null_configurations_tested":True,"calibrations":calibrations});write("power_result.json","s2_r8_hierarchical_power_result",status,{"status":status,"target_power":.8,"replicates":spec["power"]["replicates"],"results":results,"selected_cap":None if selected is None else selected["cap_per_class"],"selection_rule":spec["power"]["PASS_rule"]})
    return {"status":status,"selected":selected,"results":results}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("command",choices=("freeze-manifests","compute"));a=p.parse_args();print(json.dumps(freeze_manifests() if a.command=="freeze-manifests" else compute(),sort_keys=True))
if __name__=="__main__":main()
