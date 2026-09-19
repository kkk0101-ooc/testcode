#!/usr/bin/env python3
"""Record the immutable initial M3 float32 finite-difference failures."""
import hashlib,json
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"reports/nns_vit_endpoint_m3_r1_20260919"
def load(p):return json.loads(p.read_text())
def main():
 spec=load(REPO/"config/nns_vit_endpoint/endpoint_m3.json");atol=spec["validation_subset"]["finite_difference_atol_float32"];rtol=spec["validation_subset"]["finite_difference_rtol_float32"];tasks=load(ROOT/"task_manifest.json")["tasks"];failures=[];total=0
 for task in tasks:
  r=load(ROOT/"results"/"attempts"/task["task_id"]/"attempt-01.json")["result"];curves=r["validation"]["finite_difference"];tokens=task["payload"]["tokens"][:2];units=[(b,t,c) for b in range(12) for t in tokens for c in task["payload"]["channels"][str(b)][:4]]
  for direction in range(spec["validation_subset"]["directions"]):
   part=[x for x in curves if x["direction"]==direction]
   for index,(block,token,channel) in enumerate(units):
    total+=1;analytic=part[0]["per_target_analytic"][index];candidates=[{"epsilon":x["epsilon"],"finite_difference":x["per_target_finite_difference"][index],"tolerance_ratio":abs(x["per_target_finite_difference"][index]-analytic)/(atol+rtol*abs(analytic))} for x in part];best=min(candidates,key=lambda x:(x["tolerance_ratio"],x["epsilon"]))
    if best["tolerance_ratio"]>1:failures.append({"task_id":task["task_id"],"model_id":r["model_id"],"sample_id":r["sample_id"],"image_sha256":r["image_sha256"],"direction":direction,"block":block,"token":token,"channel":channel,"analytic":analytic,"best_float32":best,"curve":candidates})
 result={"status":"FAIL_NUMERIC_FLOAT32_FINITE_DIFFERENCE","source_validation_command":"CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 python scripts/validate_endpoint_m3.py","source_validation_error":"ValueError: FLOAT32_FD","total_target_direction_conditions":total,"failed_condition_count":len(failures),"failed_task_count":len({x["task_id"] for x in failures}),"failed_task_ids":sorted({x["task_id"] for x in failures}),"atol":atol,"rtol":rtol,"selection_rule":"all and only frozen float32 finite-difference failures; no atlas outcome or scientific statistic read","failures":failures}
 path=ROOT/"initial_numeric_validation_failure.json"
 if path.exists():
  if load(path)!=result:raise ValueError("IMMUTABLE_FAILURE_DRIFT")
 else:path.write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({k:result[k] for k in ("status","total_target_direction_conditions","failed_condition_count","failed_task_count")}))
if __name__=="__main__":main()
