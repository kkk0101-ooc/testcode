#!/usr/bin/env python3
"""Independent exhaustive validation of M3 production and numerical subsets."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OPENBLAS_NUM_THREADS"]="1"
from collections import Counter
import argparse,hashlib,json,math,sqlite3,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from run_endpoint_m3 import ROOT,SPEC,load,sha,write
from vitnns.endpoint_ep1_queue import audit_coverage,digest
from vitnns.endpoint_m3_protocol import selected_tokens,selected_channels

def require(x,code):
 if not x:raise ValueError(code)
def best_fd(curves,atol,rtol):
 n=len(curves[0]["per_target_analytic"]);passed=[]
 for i in range(n):passed.append(min(abs(c["per_target_finite_difference"][i]-c["per_target_analytic"][i])/(atol+rtol*abs(c["per_target_analytic"][i])) for c in curves))
 return max(passed)
def main():
 spec=load(SPEC);freeze=load(ROOT/"freeze.json");tasks=load(ROOT/"task_manifest.json")["tasks"];require(len(tasks)==200 and digest(tasks)==load(ROOT/"task_manifest.json")["digest"],"TASK_MANIFEST");require(audit_coverage(ROOT/"queue.sqlite3",ROOT/"results")["status"]=="PASS","QUEUE")
 require(load(ROOT/"m3a_fidelity.json")["status"]=="PASS" and load(ROOT/"m3a_fidelity.json")["comparison_count"]==5000,"M3A_FIDELITY");require(load(ROOT/"m3a_operator_validation.json")["status"]=="PASS" and load(ROOT/"m3a_operator_validation.json")["record_count"]==88,"M3A_OPERATORS")
 cohort=load(ROOT/"cohort.json")["images"];require(len(cohort)==100 and len({(x["sample_id"],x["image_sha256"]) for x in cohort})==100,"COHORT");registry=load(ROOT/"target_registry.json");require(tuple(registry["tokens"])==selected_tokens(),"TOKENS")
 target_total=zero_total=full_maps=0;max_vjp=0;max_fd=0;records=[]
 for task in tasks:
  p=task["payload"];result_path=ROOT/"results"/"attempts"/task["task_id"]/"attempt-01.json";document=load(result_path);canonical=document.pop("canonical_digest");require(digest(document)==canonical,"RESULT_CANONICAL");r=document["result"]
  require(all(r[k]==p[k] for k in ("model_id","sample_id","image_sha256","cohort_order")),"TASK_IDENTITY");require(r["target_count"]==6144,"TARGET_COUNT");require(r["model_binding"]["checkpoint_sha256"]==p["checkpoint_sha256"] and r["model_binding"]["tensor_digest"]==p["tensor_digest"],"MODEL_BINDING")
  ident=r["worker_identity"];require(ident["CUBLAS_WORKSPACE_CONFIG"]==":4096:8" and ident["deterministic_algorithms"] and not ident["cuda_matmul_allow_tf32"] and not ident["cudnn_allow_tf32"] and ident["startup_matmul_backward"]=="PASS","NUMERIC_IDENTITY")
  mp=REPO/r["metrics_path"];require(sha(mp)==r["metrics_sha256"],"METRICS_SHA");archive=np.load(mp);require(set(archive.files)=={"block","token","channel","signed_sum","absolute_mass","l2_norm","positive_mass","negative_mass","cancellation","normalized_spatial_entropy","zero_gradient","numeric_status","patch_absolute_mass_196","patch_signed_sum_196"},"METRICS_SCHEMA");data={k:archive[k] for k in archive.files};require(len(data["block"])==6144 and data["patch_absolute_mass_196"].shape==(6144,196),"METRICS_SHAPE");require(np.isfinite(data["absolute_mass"]).all() and np.isfinite(data["patch_absolute_mass_196"]).all() and (data["numeric_status"]==1).all(),"METRICS_FINITE")
  require(np.allclose(data["patch_absolute_mass_196"].sum(1),data["absolute_mass"],atol=1e-3,rtol=2e-5),"MASS_RECONSTRUCTION");zero=data["zero_gradient"];require(np.isnan(data["cancellation"][zero]).all() and np.isnan(data["normalized_spatial_entropy"][zero]).all(),"ZERO_NA");require(np.isfinite(data["cancellation"][~zero]).all() and np.isfinite(data["normalized_spatial_entropy"][~zero]).all(),"NONZERO_FINITE")
  checks=r["validation"]["checks"];require(len(checks)==96,"VJP_CHECK_COUNT");ratios=[c["batched_vjp_scalar_vjp_max_abs"]/(spec["validation_subset"]["batched_scalar_atol"]+spec["validation_subset"]["batched_scalar_rtol"]*c["scalar_endpoint_max_abs"]) for c in checks];max_vjp=max(max_vjp,max(ratios));require(max(ratios)<=1,"BATCHED_SCALAR_VJP")
  curves=r["validation"]["finite_difference"];require(len(curves)==12,"FD_CURVE_COUNT");bydir={d:[c for c in curves if c["direction"]==d] for d in range(3)};fd=max(best_fd(x,spec["validation_subset"]["finite_difference_atol_float32"],spec["validation_subset"]["finite_difference_rtol_float32"]) for x in bydir.values());max_fd=max(max_fd,fd)
  if r["full_maps_path"]:
   fp=REPO/r["full_maps_path"];require(sha(fp)==r["full_maps_sha256"] and len(np.load(fp).files)==6,"FULL_MAPS");full_maps+=len(np.load(fp).files)
  target_total+=r["target_count"];zero_total+=r["zero_gradient_count"];records.append({"task_id":task["task_id"],"result_sha256":sha(result_path),"metrics_sha256":r["metrics_sha256"]})
 require(full_maps==120,"FULL_MAP_COUNT")
 # Preregistered float64 review: validate every target at its best frozen epsilon.
 review=load(ROOT/"m3b_float64_review.json");ratios=[]
 for record in review["records"]:
  for direction in record["directions"]:
   curves=direction["curves"];n=len(curves[0]["analytic"])
   for i in range(n):ratios.append(min(abs(c["finite_difference"][i]-c["analytic"][i])/(spec["validation_subset"]["finite_difference_atol_float64"]+spec["validation_subset"]["finite_difference_rtol_float64"]*abs(c["analytic"][i])) for c in curves))
 require(len(review["records"])==4 and max(ratios)<=1,"FLOAT64_REVIEW");review["status"]="PASS";review["max_best_tolerance_ratio"]=max(ratios);write(ROOT/"m3b_float64_review.json",review)
 # The recovery population is mechanically bound to the immutable float32 failure receipt.
 initial=load(ROOT/"initial_numeric_validation_failure.json");require(initial["status"]=="FAIL_NUMERIC_FLOAT32_FINITE_DIFFERENCE" and initial["failed_condition_count"]==36 and initial["failed_task_count"]==10,"INITIAL_FAILURE");recovery=load(ROOT/"m3b_float64_recovery.json");require(recovery["source_failure_sha256"]==sha(ROOT/"initial_numeric_validation_failure.json") and recovery["selected_task_ids"]==initial["failed_task_ids"],"RECOVERY_SELECTION");recovery_ratios=[]
 for record in recovery["records"]:
  for direction in record["directions"]:
   curves=direction["curves"];n=len(curves[0]["analytic"])
   for i in range(n):recovery_ratios.append(min(abs(c["finite_difference"][i]-c["analytic"][i])/(spec["validation_subset"]["finite_difference_atol_float64"]+spec["validation_subset"]["finite_difference_rtol_float64"]*abs(c["analytic"][i])) for c in curves))
 require(len(recovery["records"])==10 and len(recovery_ratios)==2880 and max(recovery_ratios)<=1,"FLOAT64_RECOVERY");recovery["status"]="PASS_FLOAT64_RECOVERY_WITH_RETAINED_FLOAT32_FAILURE";recovery["max_best_tolerance_ratio"]=max(recovery_ratios);write(ROOT/"m3b_float64_recovery.json",recovery)
 report={"status":"PASS","numeric_status":"PASS_FLOAT64_RECOVERY_WITH_RETAINED_FLOAT32_FAILURE","tasks":200,"targets":target_total,"zero_gradients":zero_total,"full_maps":full_maps,"batched_scalar_max_tolerance_ratio":max_vjp,"float32_fd_failed_conditions":initial["failed_condition_count"],"float32_fd_failed_tasks":initial["failed_task_count"],"float32_fd_max_best_tolerance_ratio":max_fd,"float64_fd_max_best_tolerance_ratio":max(ratios),"float64_recovery_max_best_tolerance_ratio":max(recovery_ratios),"records":records};write(ROOT/"task_validation.json",report);print(json.dumps({k:v for k,v in report.items() if k!="records"}))
if __name__=="__main__":main()
