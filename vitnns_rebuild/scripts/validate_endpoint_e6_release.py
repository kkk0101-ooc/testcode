#!/usr/bin/env python3
"""Independent source-recomputing validator for ENDPOINT-E6."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/"reports/nns_vit_endpoint_e6_release_r3_20260918"
SPEC_PATH=REPO/"config/nns_vit_endpoint/endpoint_e6_release_run_spec.json"
E4_R3=REPO/"reports/nns_vit_endpoint_e4_intervention_r3_20260918"
E4_R2=REPO/"reports/nns_vit_endpoint_e4_intervention_r2_20260918"
E5=REPO/"reports/nns_vit_endpoint_e5_spatial_r1_20260918"
PASS="ENDPOINT_E6_R3_PASS_STATISTICS_AND_PAPER_RELEASE_COMPLETE"
NEXT="NONE_ENDPOINT_PRIMARY_TRACK_COMPLETE"
SOURCE_CODE=["config/nns_vit_endpoint/endpoint_e6_release_run_spec.json","docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE6执行合同_20260918.md","docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE6恢复R2_20260918.md","docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE6封存验证器修订R1_20260918.md","docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE6恢复R3_20260918.md","src/vitnns/endpoint_e6_release.py","scripts/run_endpoint_e6_release.py","scripts/validate_endpoint_e6_release.py","tests/test_endpoint_e6_release.py"]
VALIDATOR_PATH="scripts/validate_endpoint_e6_release.py"
VALIDATOR_AMENDMENT="docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE6恢复R3_20260918.md"
REFERENCE_PATHS={
 "endpoint_e6_r1_failure_sha256":"reports/nns_vit_endpoint_e6_release_r1_20260918/failure_receipt.json",
 "endpoint_e6_r2_terminal_sha256":"reports/nns_vit_endpoint_e6_release_r2_20260918/terminal_receipt.json","endpoint_e6_r2_manifest_sha256":"reports/nns_vit_endpoint_e6_release_r2_20260918/artifact_manifest.json",
 "endpoint_e6_r2_validation_sha256":"reports/nns_vit_endpoint_e6_release_r2_20260918/validation_report.json","endpoint_e6_r2_reproduction_sha256":"reports/nns_vit_endpoint_e6_release_r2_20260918/reproduction_manifest.json",
 "endpoint_e6_r2_family_decisions_sha256":"reports/nns_vit_endpoint_e6_release_r2_20260918/family_decisions.json","endpoint_e6_r2_paper_en_sha256":"manuscript/en/endpoint_scanner_results_release.md",
 "endpoint_e6_r2_paper_zh_sha256":"manuscript/zh/endpoint_scanner_results_release.md","endpoint_e6_r2_evidence_matrix_sha256":"manuscript/notes/endpoint_scanner_e6_evidence_matrix.md",
 "endpoint_e6_r2_writing_audit_sha256":"manuscript/qc/endpoint_scanner_e6_defensive_writing_audit.md",
 "cohort_registry_sha256":"manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json","statistical_analysis_plan_sha256":"manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json",
 "endpoint_e4_r2_artifact_manifest_sha256":"reports/nns_vit_endpoint_e4_intervention_r2_20260918/artifact_manifest.json","endpoint_e4_r3_artifact_manifest_sha256":"reports/nns_vit_endpoint_e4_intervention_r3_20260918/artifact_manifest.json",
 "endpoint_e4_r3_combined_index_sha256":"reports/nns_vit_endpoint_e4_intervention_r3_20260918/combined_result_index.json","endpoint_e4_r3_terminal_sha256":"reports/nns_vit_endpoint_e4_intervention_r3_20260918/terminal_receipt.json",
 "endpoint_e4_r3_validation_sha256":"reports/nns_vit_endpoint_e4_intervention_r3_20260918/validation_report.json","endpoint_e5_artifact_manifest_sha256":"reports/nns_vit_endpoint_e5_spatial_r1_20260918/artifact_manifest.json",
 "endpoint_e5_coverage_sha256":"reports/nns_vit_endpoint_e5_spatial_r1_20260918/formal_coverage_audit.json","endpoint_e5_task_manifest_sha256":"reports/nns_vit_endpoint_e5_spatial_r1_20260918/task_manifest.json",
 "endpoint_e5_terminal_sha256":"reports/nns_vit_endpoint_e5_spatial_r1_20260918/terminal_receipt.json","endpoint_e5_validation_sha256":"reports/nns_vit_endpoint_e5_spatial_r1_20260918/validation_report.json",
 "endpoint_ep1_r11_terminal_sha256":"reports/nns_vit_endpoint_ep1_r11_20260918/terminal_receipt.json","endpoint_ep1_r11_validation_sha256":"reports/nns_vit_endpoint_ep1_r11_20260918/validation_report.json",
 "endpoint_e2_r4_terminal_sha256":"reports/nns_vit_endpoint_e2_atlas_r4_20260918/terminal_receipt.json","endpoint_e2_r4_validation_sha256":"reports/nns_vit_endpoint_e2_atlas_r4_20260918/validation_report.json",
 "endpoint_e3_r2_terminal_sha256":"reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json","endpoint_e3_r2_validation_sha256":"reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json",
 "endpoint_e4_r1_terminal_sha256":"reports/nns_vit_endpoint_e4_intervention_r1_20260918/terminal_receipt.json","endpoint_e4_r2_terminal_sha256":"reports/nns_vit_endpoint_e4_intervention_r2_20260918/terminal_receipt.json",
 "unified_plan_sha256":"docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md"}


def require(value: bool, code: str) -> None:
    if not value: raise ValueError(code)


def canonical(value: Any) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def digest(value: Any) -> str: return hashlib.sha256(canonical(value)).hexdigest()


def sha(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> Any: return json.loads(path.read_text())


def git_blob_sha(commit: str, relative: str) -> str:
    data=subprocess.check_output(["git","show",f"{commit}:vitnns_rebuild/{relative}"],cwd=REPO)
    return hashlib.sha256(data).hexdigest()


def ranks(values: Sequence[float]) -> np.ndarray:
    a=np.asarray(values,dtype=np.float64);require(a.ndim==1 and a.size>0 and np.isfinite(a).all(),"RANK_INPUT")
    order=np.argsort(a,kind="stable");result=np.empty(a.size,dtype=np.float64);start=0
    while start<a.size:
        stop=start+1
        while stop<a.size and a[order[stop]]==a[order[start]]:stop+=1
        result[order[start:stop]]=(start+stop-1)/2;start=stop
    return result


def corr(left: Sequence[float],right: Sequence[float]) -> dict[str,Any]:
    x=np.asarray(left,dtype=np.float64);y=np.asarray(right,dtype=np.float64);require(x.size==y.size and x.size>=8 and np.isfinite(x).all() and np.isfinite(y).all(),"CORRELATION_INPUT")
    x=ranks(x);y=ranks(y);x-=x.mean();y-=y.mean();denom=float(np.sqrt(np.dot(x,x)*np.dot(y,y)))
    return {"value":None,"n":int(x.size),"reason":"NA_CONSTANT_VECTOR"} if denom==0 else {"value":float(np.dot(x,y)/denom),"n":int(x.size),"reason":None}


def auc(values: Sequence[float]) -> float:
    a=np.abs(np.asarray(values,dtype=np.float64));require(a.shape==(4,) and np.isfinite(a).all(),"AUC_INPUT");return float((.0625*(a[0]+a[1])/2+.0625*(a[1]+a[2])/2+.125*(a[2]+a[3])/2)/.25)


def sign(values: Sequence[float]) -> dict[str,Any]:
    a=np.asarray(values,dtype=np.float64);require(a.ndim==1 and np.isfinite(a).all(),"SIGN_INPUT");positive=int(np.sum(a>1e-12));negative=int(np.sum(a< -1e-12));n=positive+negative;ties=int(a.size-n)
    p=1.0 if n==0 else min(1.0,float(2*sum(math.comb(n,i) for i in range(min(positive,negative)+1))/(2**n)))
    return {"n_total":int(a.size),"n_nonzero":n,"n_positive":positive,"n_negative":negative,"n_tie":ties,"raw_p_value":p}


def bootstrap(columns: Mapping[str,Sequence[float]],family_id: str) -> dict[str,Any]:
    arrays={k:np.asarray(v,dtype=np.float64) for k,v in columns.items()};sizes={v.size for v in arrays.values()};require(len(sizes)==1 and all(np.isfinite(v).all() for v in arrays.values()),"BOOTSTRAP_INPUT");n=next(iter(sizes));seed=int.from_bytes(hashlib.sha256(f"NNS_VIT_ENDPOINT_EP0_R1_20260918|bootstrap|{family_id}".encode()).digest()[:8],"big");rng=np.random.Generator(np.random.PCG64(seed));draws={k:np.empty(10000) for k in arrays}
    for start in range(0,10000,500):
        index=rng.integers(0,n,size=(min(500,10000-start),n),endpoint=False)
        for key,value in arrays.items():draws[key][start:start+index.shape[0]]=np.median(value[index],axis=1)
    return {"seed":seed,"replicates":10000,"n":n,"results":{k:{"estimate":float(np.median(v)),"ci_lower":float(np.quantile(draws[k],.025,method="linear")),"ci_upper":float(np.quantile(draws[k],.975,method="linear"))} for k,v in arrays.items()}}


def load_sources() -> tuple[dict[str,dict],dict[str,dict]]:
    combined=load(E4_R3/"combined_result_index.json");m3={r["path"]:r for r in load(E4_R3/"artifact_manifest.json")["files"]};m2={r["path"]:r for r in load(E4_R2/"artifact_manifest.json")["files"]};e4={}
    for record in combined["records"]:
        relative=Path(record["source_result_path"]);path=REPO/relative
        if path.is_relative_to(E4_R3): manifest=m3.get(path.relative_to(E4_R3).as_posix())
        elif path.is_relative_to(E4_R2): manifest=m2.get(path.relative_to(E4_R2).as_posix())
        else: raise ValueError("E4_PATH")
        require(manifest is not None and sha(path)==manifest["sha256"] and path.stat().st_size==manifest["bytes"],"E4_MANIFEST");document=load(path);copy=dict(document);observed=copy.pop("canonical_digest");require(digest(copy)==observed==record["source_result_digest"],"E4_DIGEST");e4[record["task_id"]]=document["result"]
    m5={r["path"]:r for r in load(E5/"artifact_manifest.json")["files"]};e5={}
    for model in ("vit_b16","deit_s16"):
        for order in range(400):
            relative=f"formal_results/attempts/{model}__{order:03d}/attempt-01.json";path=E5/relative;manifest=m5[relative];require(sha(path)==manifest["sha256"] and path.stat().st_size==manifest["bytes"],"E5_MANIFEST");document=load(path);copy=dict(document);observed=copy.pop("canonical_digest");require(digest(copy)==observed,"E5_DIGEST");e5[document["task_id"]]=document["result"]
    require(len(e4)==len(e5)==800,"SOURCE_COVERAGE");return e4,e5


def image_rows(e4:dict[str,dict],e5:dict[str,dict]) -> list[dict[str,Any]]:
    rows=[]
    for model in ("vit_b16","deit_s16"):
        for order in range(400):
            task=f"{model}__{order:03d}";left=e4[task];right=e5[task];require((left["sample_id"],left["split_order"],left["image_sha256"])==(right["sample_id"],right["split_order"],right["image_sha256"]),"SOURCE_IDENTITY")
            for reference in ("normalized_zero","blur_rgb"):
                c4a={}
                for family in ("heads","routing_value","residual_branches"):
                    components=[r for r in left["finite_utility"][reference]["components"] if r["family"]==family];primary=corr([r["m"] for r in components],[abs(r["effect"]) for r in components]);secondary={"a_vs_effect":corr([r["a"] for r in components],[r["effect"] for r in components]),"p_ref_vs_effect":corr([r["p_ref"] for r in components],[r["effect"] for r in components]),"abs_p_ref_vs_absolute_effect":corr([abs(r["p_ref"]) for r in components],[abs(r["effect"]) for r in components]),"output_norm_vs_absolute_effect":corr([r["output_norm"] for r in components],[abs(r["effect"]) for r in components]),"gradient_norm_vs_absolute_effect":corr([r["gradient_norm"] for r in components],[abs(r["effect"]) for r in components]),"abs_gradient_activation_vs_absolute_effect":corr([abs(r["gradient_activation"]) for r in components],[abs(r["effect"]) for r in components])};c4a[family]={"primary":primary,"secondary":secondary,"component_count":len(components)}
                curves=right["references"][reference]["curves"];scores={}
                for method in ("endpoint_component","total_gxi","raw_gradient","regional_taylor"):
                    values=[]
                    for head in right["heads"]:
                        selected=[r for r in curves if r["method"]==method and r["head"]==head];require(len(selected)==1,"C4B_CURVE");values.append(auc([p["isolated_effect"] for p in selected[0]["points"]]))
                    scores[method]=float(np.mean(values))
                random=[]
                for head in right["heads"]:
                    selected=sorted((r for r in curves if r["method"]=="random_regions" and r["head"]==head),key=lambda r:r["repeat"]);require(len(selected)==20,"RANDOM_COVERAGE");random.append(auc([float(np.mean([r["points"][i]["isolated_effect"] for r in selected])) for i in range(4)]))
                scores["random_regions"]=float(np.mean(random));paired={key:scores["endpoint_component"]-scores[key] for key in ("total_gxi","raw_gradient","regional_taylor","random_regions")}
                rows.append({"task_id":task,"model_id":model,"split_order":order,"sample_id":left["sample_id"],"reference_id":reference,"c4a":c4a,"c4b":{"scores":scores,"paired":paired}})
    require(len(rows)==1600,"ROW_COVERAGE");return rows


def recompute_inference(rows:list[dict],plan:dict) -> tuple[list[dict],list[dict],list[dict]]:
    test_plan={r["test_id"]:r for r in plan["tests"]};families=[];tests=[]
    for family in plan["families"]:
        selected=[r for r in rows if r["model_id"]==family["model"] and r["reference_id"]==family["replacement"]];fid=family["family_id"]
        if family["claim"]=="C4a":values=[r["c4a"][family["component_family"]]["primary"]["value"] for r in selected if r["c4a"][family["component_family"]]["primary"]["reason"] is None];columns={fid:values}
        else:columns={test_id:[r["c4b"]["paired"][test_id.split(".vs_",1)[1]] for r in selected] for test_id in family["tests"]}
        b=bootstrap(columns,fid);ftests=[]
        for test_id,values in columns.items():
            ci=b["results"][test_id];row={"test_id":test_id,"family_id":fid,"claim":family["claim"],"model_id":family["model"],"reference_id":family["replacement"],"component_family":family["component_family"],"baseline":test_id.split(".vs_",1)[1] if ".vs_" in test_id else None,"estimate":ci["estimate"],"ci_lower":ci["ci_lower"],"ci_upper":ci["ci_upper"],"ci_width":ci["ci_upper"]-ci["ci_lower"],"bootstrap_seed":b["seed"],"bootstrap_replicates":10000,"min_nonzero_n":test_plan[test_id]["min_nonzero_n"],**sign(values)};ftests.append(row);tests.append(row)
        families.append({"family_id":fid,"claim":family["claim"],"model_id":family["model"],"reference_id":family["replacement"],"component_family":family["component_family"],"n_valid":len(next(iter(columns.values()))),"n_valid_min":family["n_valid_min"],"exact_coverage":len(selected)==400,"delta_min":family["delta_min"],"ci_width_max":family["ci_width_max"],"tests":[r["test_id"] for r in ftests]})
    order=plan["holm"]["test_ids"];position={v:i for i,v in enumerate(order)};sorted_tests=sorted(tests,key=lambda r:(r["raw_p_value"],r["test_id"]));running=0.;holm=[]
    for index,row in enumerate(sorted_tests):running=max(running,min(1.,(28-index)*row["raw_p_value"]));holm.append({"test_id":row["test_id"],"frozen_order":position[row["test_id"]],"holm_rank":index,"raw_p_value":row["raw_p_value"],"adjusted_p_value":running,"reject_alpha_0_05":running<=.05})
    adjusted={r["test_id"]:r for r in holm}
    for row in tests:row["holm_adjusted_p_value"]=adjusted[row["test_id"]]["adjusted_p_value"];row["holm_reject"]=adjusted[row["test_id"]]["reject_alpha_0_05"]
    for family in families:
        selected=[r for r in tests if r["family_id"]==family["family_id"]];inconclusive=[]
        if not family["exact_coverage"]:inconclusive.append("EXACT_COVERAGE_INCOMPLETE")
        if family["n_valid"]<family["n_valid_min"]:inconclusive.append("N_VALID_BELOW_MIN")
        if any(r["n_nonzero"]<r["min_nonzero_n"] for r in selected):inconclusive.append("N_NONZERO_BELOW_MIN")
        if any(r["ci_width"]>family["ci_width_max"] for r in selected):inconclusive.append("CI_WIDTH_ABOVE_MAX")
        supported=all(r["estimate"]>=family["delta_min"] and r["estimate"]>0 and r["holm_reject"] for r in selected)
        family["status"]="INCONCLUSIVE" if inconclusive else ("SUPPORTED" if supported else "NOT_SUPPORTED");family["decision_reasons"]=inconclusive or (["ALL_REGISTERED_SUPPORT_CONDITIONS_MET"] if supported else ["ONE_OR_MORE_REGISTERED_SUPPORT_CONDITIONS_FAILED"]);family["test_results"]=selected
    return families,tests,holm


def validate(root:Path) -> dict[str,Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not set(sys.modules).intersection({"torch","timm","PIL"}),"CPU_ONLY");root=root.resolve();spec=load(SPEC_PATH);report=load(root/"validation_report.json");terminal=load(root/"terminal_receipt.json");manifest=load(root/"artifact_manifest.json")
    require(report["status"]==terminal["status"]==PASS and report["stage"]==terminal["stage"]=="ENDPOINT-E6-R3-RECOVERY" and report["next_authorization"]==terminal["next_authorization"]==NEXT,"TERMINAL_STATUS");require(terminal["validation_digest"]==digest(report) and terminal["manifest_digest"]==manifest["canonical_digest"]==digest(manifest["files"]),"TERMINAL_DIGEST");require(not (root/"terminal_receipt.pending.json").exists() and not (root/"failure_receipt.json").exists(),"ATOMIC_TERMINAL")
    r1=load(REPO/REFERENCE_PATHS["endpoint_e6_r1_failure_sha256"]);require(r1=={"error":"name 'selected' is not defined","error_type":"NameError","next_authorization":"NONE","stage":"ENDPOINT-E6","status":"ENDPOINT_E6_FAIL_CLOSED"},"R1_FAILURE_PRESERVATION")
    paths=[r["path"] for r in manifest["files"]];require(len(paths)==len(set(paths)),"MANIFEST_DUPLICATE")
    for row in manifest["files"]:path=root/row["path"];require(path.is_file() and path.stat().st_size==row["bytes"] and sha(path)==row["sha256"],"MANIFEST_FILE:"+row["path"])
    producer=terminal["producer_commit"];require(subprocess.run(["git","merge-base","--is-ancestor",producer,"HEAD"],cwd=REPO).returncode==0,"PRODUCER_ANCESTRY")
    reproduction=load(root/"reproduction_manifest.json");require(reproduction["producer_commit"]==producer and [r["path"] for r in reproduction["source_code"]]==SOURCE_CODE,"SOURCE_COVERAGE")
    for row in reproduction["source_code"]:
        require(git_blob_sha(producer,row["path"])==row["sha256"],"SOURCE_BINDING:"+row["path"])
        if row["path"]!=VALIDATOR_PATH: require(sha(REPO/row["path"])==row["sha256"],"SOURCE_WORKTREE_DRIFT:"+row["path"])
    validator_commit=subprocess.check_output(["git","log","-1","--format=%H","--",f"vitnns_rebuild/{VALIDATOR_PATH}"],cwd=REPO,text=True).strip()
    require(git_blob_sha(validator_commit,VALIDATOR_PATH)==sha(Path(__file__)) and git_blob_sha(validator_commit,VALIDATOR_AMENDMENT)==sha(REPO/VALIDATOR_AMENDMENT),"VALIDATOR_AMENDMENT_BINDING")
    require(spec["predecessors"].keys()==REFERENCE_PATHS.keys(),"PREDECESSOR_KEYS")
    for key,relative in REFERENCE_PATHS.items():observed=git_blob_sha(producer,relative) if key=="unified_plan_sha256" else sha(REPO/relative);require(observed==spec["predecessors"][key],"PREDECESSOR:"+key)
    require(load(REPO/REFERENCE_PATHS["endpoint_ep1_r11_terminal_sha256"])["status"]=="PASS_EP1_IMPLEMENTATION_AND_REGRESSION","EP1_STATUS");require(load(REPO/REFERENCE_PATHS["endpoint_e2_r4_terminal_sha256"])["status"].startswith("ENDPOINT_E2_PASS"),"E2_STATUS");require(load(REPO/REFERENCE_PATHS["endpoint_e3_r2_terminal_sha256"])["status"].startswith("ENDPOINT_E3_PASS_CORRECTNESS_ONLY"),"E3_STATUS");require(load(REPO/REFERENCE_PATHS["endpoint_e4_r1_terminal_sha256"])["status"]=="ENDPOINT_E4_FAIL_LOCAL_DIRECTION_CONTRACT","E4_R1_HISTORY");require(load(REPO/REFERENCE_PATHS["endpoint_e4_r2_terminal_sha256"])["status"]=="ENDPOINT_E4_R2_FAIL_ANALYTIC_JVP_CONTRACT","E4_R2_HISTORY");require(load(REPO/REFERENCE_PATHS["endpoint_e4_r3_terminal_sha256"])["status"].startswith("ENDPOINT_E4_R3_PASS"),"E4_R3_STATUS");e5_terminal=load(REPO/REFERENCE_PATHS["endpoint_e5_terminal_sha256"]);require(e5_terminal["status"].startswith("ENDPOINT_E5_PASS") and e5_terminal["next_authorization"]=="ENDPOINT_E6_REQUIRES_EXPLICIT_USER_REQUEST","E5_STATUS")
    audit=load(root/"source_read_audit.json");require(audit["cpu_only"] and audit["cuda_visible_devices"]=="" and audit["forbidden_imports_loaded"]==[] and audit["image_files_read"]==audit["model_files_read"]==0 and audit["read_count"]==len(audit["reads"]),"READ_AUDIT")
    combined=load(E4_R3/"combined_result_index.json");expected_reads={SPEC_PATH.relative_to(REPO).as_posix(),*REFERENCE_PATHS.values(),*{row["source_result_path"] for row in combined["records"]},*{f"reports/nns_vit_endpoint_e5_spatial_r1_20260918/formal_results/attempts/{model}__{order:03d}/attempt-01.json" for model in ("vit_b16","deit_s16") for order in range(400)}}
    require({r["path"] for r in audit["reads"]}==expected_reads and len(audit["reads"])==len(expected_reads),"READ_WHITELIST")
    for row in audit["reads"]:path=REPO/row["path"];require(path.is_file() and path.stat().st_size==row["bytes"] and sha(path)==row["sha256"],"READ_HASH:"+row["path"])
    e4,e5=load_sources();rows=image_rows(e4,e5);per=load(root/"per_image_statistics.json");require(per["row_count"]==1600 and per["rows_digest"]==digest(rows) and per["rows"]==rows,"PER_IMAGE_RECOMPUTE");plan=load(REPO/spec["inputs"]["statistical_plan"]);families,tests,holm=recompute_inference(rows,plan);decision=load(root/"family_decisions.json");require(decision["family_count"]==16 and decision["families"]==families,"FAMILY_RECOMPUTE");require(load(root/"holm_report.json")["tests"]==holm and load(root/"sign_test_report.json")["test_count"]==28 and load(root/"bootstrap_report.json")["family_count"]==16,"INFERENCE_REPORTS")
    aggregate={claim:("INVALID" if any(r["status"]=="INVALID" for r in families if r["claim"]==claim) else "INCONCLUSIVE" if any(r["status"]=="INCONCLUSIVE" for r in families if r["claim"]==claim) else "SUPPORTED" if all(r["status"]=="SUPPORTED" for r in families if r["claim"]==claim) else "NOT_SUPPORTED") for claim in ("C4a","C4b")};require(decision["aggregate_claims"]==report["aggregate_claims"]==aggregate and terminal["C4a"]==aggregate["C4a"] and terminal["C4b"]==aggregate["C4b"],"AGGREGATE_DECISION")
    require(report["evaluation_images"]==terminal["evaluation_image_count"]==400 and report["family_count"]==terminal["family_count"]==16 and report["test_count"]==terminal["test_count"]==report["holm_family_count"]==28 and report["gpu_hours"]==0 and not report["new_empirical_compute"] and report["image_decode_count"]==report["model_load_count"]==0,"REPORT_COUNTS")
    for row in reproduction["paper_release_files"]:path=REPO/row["path"];require(path.is_file() and path.stat().st_size==row["bytes"] and sha(path)==row["sha256"],"PAPER_FILE")
    en=(REPO/spec["outputs"]["paper_release_en"]).read_text();zh=(REPO/spec["outputs"]["paper_release_zh"]).read_text();require(f"**C4a {aggregate['C4a']}**" in en and f"**C4b {aggregate['C4b']}**" in en and f"**C4a {aggregate['C4a']}**" in zh and f"**C4b {aggregate['C4b']}**" in zh,"BILINGUAL_DECISIONS");require(en.count("| C4")==zh.count("| C4")==16,"BILINGUAL_FAMILY_TABLE")
    require(r"\n\n" not in en and r"\n\n" not in zh and en.count("\n\n")>=4 and zh.count("\n\n")>=4,"BILINGUAL_MARKDOWN_FORMAT")
    for row in reproduction["figures"]:path=root/row["path"];text=path.read_text();require(path.suffix==".svg" and path.stat().st_size==row["bytes"] and sha(path)==row["sha256"] and text.count("<circle")== (12 if "c4a" in path.name else 16),"FIGURE")
    require(report["path_track_status"]=="EP2_R1_PATH_NUMERICAL_CALIBRATION_FAILED_IMMUTABLE" and load(root/"claim_decisions.json")["Path_IG"]["status"]==report["path_track_status"],"PATH_BOUNDARY")
    return {"status":PASS,"families":16,"tests":28,"C4a":aggregate["C4a"],"C4b":aggregate["C4b"],"per_image_rows":1600,"next_authorization":NEXT}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("root",nargs="?",type=Path,default=ROOT);print(json.dumps(validate(parser.parse_args().root),sort_keys=True))
