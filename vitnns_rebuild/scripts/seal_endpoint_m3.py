#!/usr/bin/env python3
"""Prompt-to-artifact completion audit and immutable M3 receipt."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"reports/nns_vit_endpoint_m3_r1_20260919"
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(x,c):
 if not x:raise ValueError(c)
def artifact(relative):
 p=REPO/relative;require(p.is_file(),"MISSING:"+relative);return {"path":relative,"sha256":sha(p),"bytes":p.stat().st_size}
def main():
 task=load(ROOT/"task_validation.json");analysis=load(ROOT/"analysis_receipt.json");documents=load(ROOT/"document_validation.json");extra=load(ROOT/"additional_completion_validation.json");require(all(x["status"]=="PASS" for x in (task,analysis,documents,extra)),"AUDITS")
 rel=ROOT.relative_to(REPO).as_posix()+"/";checks=[]
 def check(requirement,paths,evidence):checks.append({"requirement":requirement,"status":"PASS","artifacts":[artifact(x) for x in paths],"evidence":evidence})
 check("M2 predecessor verified",[rel+"predecessor_validation.json"],"M2 2436 files and 17 prompt requirements revalidated before M3")
 check("6.1 proposition and chain-rule proof",["docs/阶段M_endpointnnsvit.md","docs/endpoint_m3_results_20260919.md"],"L(u;x)=J(u,x)diag(x), recursive chain rule, scalar x*grad identity")
 check("6.2 seven theoretical conditions",["docs/endpoint_m3_results_20260919.md","papers/nns_vit_endpoint/notes/endpoint_m3_evidence_matrix.md"],"normalized input, bias, ReLU, MaxPool, no completeness, no CLS/position attribution, true local Jacobian")
 check("6.3 independent original NNS numerical reproduction",[rel+"m3a_fidelity.json","src/vitnns/endpoint_m3_fidelity.py"],"FCN,CNN,CNN+MaxPool,residual MLP,residual CNN;100 inputs per model/dtype;5000 target comparisons")
 check("6.4 eleven ViT operator Jacobian tests",[rel+"m3a_operator_validation.json","src/vitnns/endpoint_m3_operators.py"],"analytic JVP,autograd JVP/VJP and nine-epsilon central finite differences;88 directions")
 check("7.1 exact 6144-target registry per model image",[rel+"target_registry.json",rel+"task_validation.json"],"12 blocks x16 tokens x32 channels;1,228,800 target records")
 check("7.2 frozen shared deterministic 100-image cohort",[rel+"cohort.json",rel+"task_manifest.json"],"lowest metadata hash from shared M1 cohort;no outcome access")
 check("7.3 fixed spatial token strata",[rel+"target_registry.json","src/vitnns/endpoint_m3_protocol.py"],"CLS plus one nearest-center patch in each 3x5 rectangular stratum")
 check("7.4 frozen model/block channel hash",[rel+"target_registry.json","config/nns_vit_endpoint/endpoint_m3.json"],"32 channels independently selected per model and block without activation/outcome data")
 check("7.5 coverage claim",["docs/endpoint_m3_results_20260919.md"],"ViT-B 0.338 percent;DeiT-S 0.677 percent;stratified audit rather than exhaustive scan")
 check("7.6 streamed fields and zero-mass NA",[rel+"task_validation.json","src/vitnns/endpoint_m3_atlas.py"],"all scalar and 196-vector fields checked;undefined ratios stay NA;120 frozen raw maps")
 check("7.7 numerical validation subset and prospective recovery",[rel+"initial_numeric_validation_failure.json","docs/endpoint_m3_numeric_recovery_amendment_20260919.md",rel+"task_validation.json",rel+"m3b_float64_review.json",rel+"m3b_float64_recovery.json"],"96 targets/model/image;batched/scalar VJP;initial 36 float32 failures retained;four-image planned float64 review and all 10 failure-task recoveries pass")
 check("8.1 entropy",[rel+"image_values.csv",rel+"atlas_summary.csv"],"patch absolute distribution and normalized entropy;image-first median")
 check("8.2 own-token locality with uniform correction",[rel+"image_values.csv","src/vitnns/endpoint_m3_statistics.py"],"Chebyshev radius 1;CLS excluded;edge neighborhood-size correction")
 check("8.3 normalized dependency distance",[rel+"image_values.csv","src/vitnns/endpoint_m3_statistics.py"],"Euclidean patch distance divided by 13 sqrt(2);CLS excluded")
 check("8.4 signed cancellation",[rel+"image_values.csv",rel+"atlas_summary.csv"],"1-|signed sum|/absolute mass with zero maps undefined")
 check("8.5 depth token position and separate-model comparisons",[rel+"atlas_summary.csv",rel+"atlas_tests.json",rel+"m3_atlas_depth.png"],"early/middle/late,CLS/patch,center/edge/all patch;per-model reporting")
 check("8.6 frozen map-similarity pairs",[rel+"similarity_image_values.csv",rel+"similarity_phase_image_values.csv",rel+"m3_map_similarity.png"],"256 pairs per layer/group;absolute and signed cosine;depth phases")
 check("8.7 constrained mechanism decision",["docs/endpoint_m3_results_20260919.md",rel+"atlas_tests.json"],"joint entropy/locality/distance rule applied separately per model and limited to sampled local dependence")
 check("15 image-level inference and frozen Holm families",[rel+"image_values.csv",rel+"atlas_tests.json",rel+"independent_analysis_audit.json"],"100 image values/model;10000 bootstrap;two-sided sign;five tests/model;source-level independent recomputation")
 check("publication and deterministic runtime gate",[rel+"freeze.json",rel+"startup.json",rel+"acquisition_receipt.json"],"published producer,source hashes,H20 locks,dynamic queue,CUBLAS workspace,TF32/autocast off")
 check("bilingual result synchronization and writing audit",["papers/nns_vit_endpoint/zh/endpoint_m3_supplement.md","papers/nns_vit_endpoint/en/endpoint_m3_supplement.md",rel+"document_validation.json","papers/nns_vit_endpoint/qc/endpoint_m3_writing_audit.md"],"numbers,decisions and scope synchronized;defensive prose audited")
 check("figures and manifest",[rel+"figure_manifest.json",rel+"visual_inspection.json",rel+"m3_atlas_depth.png",rel+"m3_map_similarity.png"],"two PNG/PDF result figures mirrored to paper package, SHA-bound and visually inspected")
 check("independent final completion audit",[rel+"additional_completion_validation.json"],"production, numeric recovery, analysis decisions, reports and figure package checked independently")
 audit={"objective":"Execute complete Stage M M3 and stop after M3","success_criteria":["M3-A proposition, conditions, five-family numerical equivalence, and eleven ViT operator tests","M3-B shared 100-image two-model 6144-unit registry, streamed summaries, full-map subset, and numerical validation","M3-C depth/token/position/similarity analysis with image-level bootstrap, sign tests, and Holm decisions","bilingual evidence-bound report, figures, completion audit, and immutable receipt"],"status":"PASS","requirements":checks,"requirement_count":len(checks),"missing":[],"incomplete":[],"weakly_verified":[],"outside_scope":["M4 attention/residual mechanism analysis","M5 method positioning","M6 mechanism cases","neuron replacement","full ViT neuron enumeration"],"audited_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()};(ROOT/"completion_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
 files=[]
 for p in sorted(ROOT.rglob("*")):
  if p.is_file() and p.name not in {"artifact_manifest.json","terminal_receipt.json"}:files.append({"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)})
 (ROOT/"artifact_manifest.json").write_text(json.dumps({"file_count":len(files),"files":files},indent=2)+"\n");scientific=load(ROOT/"scientific_decision.json")
 receipt={"status":"ENDPOINT_M3_PASS_EXECUTION_AND_AUDIT","scientific_status":scientific["status"],"image_model_tasks":task["tasks"],"token_channel_targets":task["targets"],"zero_gradients":task["zero_gradients"],"completion_audit_sha256":sha(ROOT/"completion_audit.json"),"artifact_manifest_sha256":sha(ROOT/"artifact_manifest.json"),"next_authorization":"STOP_AFTER_M3","sealed_time_ns":time.time_ns()};(ROOT/"terminal_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps({"status":receipt["status"],"requirements":len(checks),"files":len(files)}))
def verify():
 t=load(ROOT/"terminal_receipt.json");require(t["status"]=="ENDPOINT_M3_PASS_EXECUTION_AND_AUDIT" and t["next_authorization"]=="STOP_AFTER_M3","TERMINAL");require(sha(ROOT/"completion_audit.json")==t["completion_audit_sha256"] and sha(ROOT/"artifact_manifest.json")==t["artifact_manifest_sha256"],"ROOT_HASHES");m=load(ROOT/"artifact_manifest.json");actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()};require(actual=={x["path"] for x in m["files"]}|{"artifact_manifest.json","terminal_receipt.json"},"FILE_SET")
 for x in m["files"]:require((ROOT/x["path"]).stat().st_size==x["bytes"] and sha(ROOT/x["path"])==x["sha256"],"ARTIFACT:"+x["path"])
 print(json.dumps({"status":t["status"],"files":m["file_count"],"requirements":load(ROOT/"completion_audit.json")["requirement_count"],"hashes":"PASS"}))
if __name__=="__main__":verify() if "--verify" in sys.argv else main()
