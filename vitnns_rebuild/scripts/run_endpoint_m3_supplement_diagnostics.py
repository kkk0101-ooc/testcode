#!/usr/bin/env python3
"""M3-SUP cancellation-scale and finite-difference diagnostics."""
import csv,json,math,os,sys
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OPENBLAS_NUM_THREADS"]="1"
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns.endpoint_m3_statistics import bootstrap_median
from vitnns.endpoint_m3_supplement import cancellation_metrics,tolerance_ratio
SOURCE=REPO/"reports/nns_vit_endpoint_m3_r1_20260919";ROOT=REPO/"reports/nns_vit_endpoint_m3_sup_r2_20260919"
def load(p):return json.loads(p.read_text())
def dump(path,rows):
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,separators=(",",":")) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
def main():
 full=load(SOURCE/"full_map_manifest.json");map_rows=[]
 for record in full["records"]:
  z=np.load(REPO/record["path"]);model=record["task_id"].split("__")[0]
  for key in z.files:
   b,t,c=(int(key[1:3]),int(key[5:8]),int(key[10:14]));map_rows.append({"task_id":record["task_id"],"model_id":model,"map_key":key,"block":b,"token":t,"channel":c,**cancellation_metrics(z[key])})
 dump(ROOT/"cancellation_map_values.csv",map_rows);summary=[]
 metrics=("within_rgb_channels","within_patch_after_rgb_sum","across_patch_signed","whole_image_cancellation","net_ratio")
 for key in sorted({(r["model_id"],r["block"],r["token"]==0) for r in map_rows}):
  for metric in metrics:
   values=[r[metric] for r in map_rows if (r["model_id"],r["block"],r["token"]==0)==key and r[metric] is not None];summary.append({"model":key[0],"block":key[1],"token_type":"cls" if key[2] else "patch","metric":metric,"n":len(values),"median":float(np.median(values)),"ci95":bootstrap_median(values,"M3SUP|cancel|"+"|".join(map(str,key))+"|"+metric,10000)})
 dump(ROOT/"cancellation_summary.csv",summary)
 failure=load(SOURCE/"initial_numeric_validation_failure.json");recovery=load(SOURCE/"m3b_float64_recovery.json");by_task={r["task_id"]:r for r in recovery["records"]};diag=[]
 for item in failure["failures"]:
  record=by_task[item["task_id"]];index=next(i for i,t in enumerate(record["targets"]) if (t["block"],t["token"],t["channel"])==(item["block"],item["token"],item["channel"]));curve64=next(x for x in record["directions"] if x["direction"]==item["direction"])["curves"];a64=curve64[0]["analytic"][index];float64=[{"epsilon":x["epsilon"],"finite_difference":x["finite_difference"][index],"tolerance_ratio":tolerance_ratio(x["finite_difference"][index],a64,1e-7,1e-4)} for x in curve64];best64=min(float64,key=lambda x:x["tolerance_ratio"]);diag.append({"task_id":item["task_id"],"model_id":item["model_id"],"sample_id":item["sample_id"],"block":item["block"],"token":item["token"],"channel":item["channel"],"direction":item["direction"],"analytic_float32":item["analytic"],"analytic_float64":a64,"absolute_analytic_magnitude_float64":abs(a64),"best_float32_tolerance_ratio":item["best_float32"]["tolerance_ratio"],"best_float64_tolerance_ratio":best64["tolerance_ratio"],"best_float32_epsilon":item["best_float32"]["epsilon"],"best_float64_epsilon":best64["epsilon"],"float32_curve":item["curve"],"float64_curve":float64,"operator_attribution":"NOT_IDENTIFIABLE_POST_BLOCK_TARGET"})
 dump(ROOT/"finite_difference_diagnostics.csv",diag);write={"status":"PASS","float32_failure_conditions":len(diag),"float64_recovered_conditions":sum(r["best_float64_tolerance_ratio"]<=1 for r in diag),"cancellation_maps":len(map_rows),"survival_identity_max_error":max(r["survival_identity_error"] for r in map_rows),"fd_operator_attribution":"NOT_IDENTIFIABLE_POST_BLOCK_TARGET"};(ROOT/"diagnostic_receipt.json").write_text(json.dumps(write,indent=2)+"\n");print(json.dumps(write))
if __name__=="__main__":main()
