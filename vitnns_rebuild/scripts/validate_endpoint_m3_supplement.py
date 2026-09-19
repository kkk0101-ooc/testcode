#!/usr/bin/env python3
"""Independent source-to-artifact validation for M3-SUP-R2."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OPENBLAS_NUM_THREADS"]="1"
import csv,hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns.endpoint_m3_supplement import cancellation_metrics,tolerance_ratio
from vitnns.endpoint_m3_statistics import bootstrap_median
SOURCE=REPO/"reports/nns_vit_endpoint_m3_r1_20260919";ROOT=REPO/"reports/nns_vit_endpoint_m3_sup_r2_20260919";SPEC=REPO/"config/nns_vit_endpoint/endpoint_m3_supplement_r2.json"
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(x,c):
 if not x:raise ValueError(c)
def close(a,b,c,tol=1e-12):require(math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol),c)
def csvrows(name):
 with (ROOT/name).open() as f:return list(csv.DictReader(f))
def main():
 spec=load(SPEC);freeze=load(ROOT/"freeze.json")
 for p,s in spec["source_bindings"].items():require(sha(REPO/p)==s,"SOURCE:"+p)
 for p,s in freeze["sources"].items():
  if p not in {"scripts/render_endpoint_m3_supplement.py","scripts/write_endpoint_m3_supplement.py","scripts/validate_endpoint_m3_supplement.py"}:require(sha(REPO/p)==s,"FROZEN:"+p)
 require(freeze["producer_commit"]==freeze["remote_commit"] and subprocess.run(["git","merge-base","--is-ancestor",freeze["producer_commit"],"HEAD"],cwd=REPO).returncode==0,"ANCESTRY")
 # M3-A distributions and operator tables.
 fidelity=load(SOURCE/"m3a_fidelity.json")["records"];published=csvrows("m3a_error_distribution.csv")
 for row in published:
  if row["group"]=="overall":part=fidelity
  elif row["group"]=="dtype":part=[r for r in fidelity if r["dtype"]==row["key"]]
  elif row["group"]=="model":part=[r for r in fidelity if r["model"]==row["key"]]
  else:
   model,dtype=row["key"].split("|");part=[r for r in fidelity if r["model"]==model and r["dtype"]==dtype]
  a=np.asarray([r[row["metric"]] for r in part]);expected=[a.min(),np.median(a),np.quantile(a,.9),np.quantile(a,.99),a.max()]
  require(int(row["n"])==len(a),"ERROR_N")
  for key,value in zip(("minimum","median","q90","q99","maximum"),expected):close(row[key],value,"ERROR_QUANTILE")
 operators=load(SOURCE/"m3a_operator_validation.json");oprows=csvrows("operator_validation_table.csv");require(len(oprows)==11,"OP_ROWS")
 for row in oprows:
  part=[r for r in operators["records"] if r["operator"]==row["operator"]];require(len(part)==int(row["directions"])==8 and row["dtype"]=="float64","OP_SCOPE");close(row["max_analytic_autograd_jvp_relative_error"],max(x["analytic_autograd_jvp_relative_error"] for x in part),"OP_JVP");close(row["max_jvp_vjp_identity_relative_error"],max(x["jvp_vjp_identity_relative_error"] for x in part),"OP_DUAL");close(row["max_best_finite_difference_relative_error"],max(x["finite_difference_best_relative_error"] for x in part),"OP_FD")
 # Recompute block and CLS-patch values from all 200 source tasks.
 block_rows=csvrows("block_image_values.csv");cls_rows=csvrows("cls_patch_image_values.csv");block_index={(r["model_id"],r["sample_id"],int(r["block"]),r["metric"]):r for r in block_rows};cls_index={(r["model_id"],r["sample_id"],r["stratum"]):r for r in cls_rows};tasks=load(SOURCE/"task_manifest.json")["tasks"]
 for task in tasks:
  result=load(SOURCE/"results/attempts"/task["task_id"]/"attempt-01.json")["result"];z=np.load(REPO/result["metrics_path"]);block=z["block"];token=z["token"];entropy=z["normalized_spatial_entropy"].astype(float);cancellation=z["cancellation"].astype(float);mass=z["patch_absolute_mass_196"].astype(float);locality=np.full(len(block),np.nan);distance=np.full(len(block),np.nan)
  for i in np.flatnonzero(token>0):
   p=mass[i]/mass[i].sum();tr,tc=divmod(int(token[i])-1,14);rr,cc=np.indices((14,14));dr=np.abs(rr-tr);dc=np.abs(cc-tc);near=np.maximum(dr,dc)<=1;locality[i]=p.reshape(14,14)[near].sum()-near.sum()/196;distance[i]=(p.reshape(14,14)*np.sqrt(dr**2+dc**2)/(13*np.sqrt(2))).sum()
  arrays={"entropy":entropy,"locality":locality,"distance":distance,"cancellation":cancellation,"net_ratio":1-cancellation};diff=[]
  for b in range(12):
   for name,a in arrays.items():close(block_index[(result["model_id"],result["sample_id"],b,name)]["median"],np.nanmedian(a[block==b]),"BLOCK_VALUE")
   cls=np.median(entropy[(block==b)&(token==0)]);patch=np.median(entropy[(block==b)&(token>0)]);diff.append(float(cls-patch));close(cls_index[(result["model_id"],result["sample_id"],f"block_{b:02d}")]["paired_difference"],diff[-1],"CLS_BLOCK")
  for name,ids in (("early",range(4)),("middle",range(4,8)),("late",range(8,12)),("all",range(12))):close(cls_index[(result["model_id"],result["sample_id"],name)]["paired_difference"],np.median([diff[i] for i in ids]),"CLS_PHASE")
 # Independently reconstruct image-level medians and deterministic bootstrap tables.
 block_summary=csvrows("block_summary.csv");cls_summary=csvrows("cls_patch_summary.csv")
 for row in block_summary:
  values=[float(r["median"]) for r in block_rows if r["model_id"]==row["model"] and int(r["block"])==int(row["block"]) and r["metric"]==row["metric"]];close(row["median"],np.median(values),"BLOCK_SUMMARY");require(json.loads(row["ci95"])==bootstrap_median(values,"M3SUP|block|"+"|".join((row["model"],row["block"],row["metric"])),10000),"BLOCK_BOOTSTRAP")
 for row in cls_summary:
  values=[float(r["paired_difference"]) for r in cls_rows if r["model_id"]==row["model"] and r["stratum"]==row["stratum"]];close(row["median_difference"],np.median(values),"CLS_SUMMARY");require(json.loads(row["ci95"])==bootstrap_median(values,"M3SUP|cls_patch|"+"|".join((row["model"],row["stratum"])),10000),"CLS_BOOTSTRAP");require((int(row["positive"]),int(row["negative"]),int(row["ties"]))==(sum(v>1e-12 for v in values),sum(v< -1e-12 for v in values),sum(abs(v)<=1e-12 for v in values)),"CLS_SIGNS")
 # Cancellation from all 120 raw maps and all 36 FD failures.
 cmap=csvrows("cancellation_map_values.csv");cindex={(r["task_id"],r["map_key"]):r for r in cmap};full=load(SOURCE/"full_map_manifest.json")
 for record in full["records"]:
  z=np.load(REPO/record["path"])
  for key in z.files:
   expected=cancellation_metrics(z[key]);published0=cindex[(record["task_id"],key)]
   for field,value in expected.items():close(published0[field],value,"CANCELLATION")
 diag=csvrows("finite_difference_diagnostics.csv");failure=load(SOURCE/"initial_numeric_validation_failure.json");recovery=load(SOURCE/"m3b_float64_recovery.json");by_failure={(r["task_id"],r["block"],r["token"],r["channel"],r["direction"]):r for r in failure["failures"]};by_recovery={r["task_id"]:r for r in recovery["records"]};require(len(diag)==len(by_failure)==36,"FD_COUNT")
 for row in diag:
  key=(row["task_id"],int(row["block"]),int(row["token"]),int(row["channel"]),int(row["direction"]));source=by_failure[key];require(json.loads(row["float32_curve"])==source["curve"] and row["operator_attribution"]=="NOT_IDENTIFIABLE_POST_BLOCK_TARGET","FD_FLOAT32");record=by_recovery[row["task_id"]];index=next(i for i,t in enumerate(record["targets"]) if (t["block"],t["token"],t["channel"])==key[1:4]);curve=next(x for x in record["directions"] if x["direction"]==key[4])["curves"];analytic=curve[0]["analytic"][index];expected=[{"epsilon":x["epsilon"],"finite_difference":x["finite_difference"][index],"tolerance_ratio":tolerance_ratio(x["finite_difference"][index],analytic,1e-7,1e-4)} for x in curve];require(json.loads(row["float64_curve"])==expected and float(row["best_float64_tolerance_ratio"])==min(x["tolerance_ratio"] for x in expected)<=1,"FD_FLOAT64")
 review=load(ROOT/"cancellation_float64_review.json");require(review["status"]=="PASS" and review["target_count"]==24 and review["max_cancellation_metric_absolute_difference"]<=2e-6,"FLOAT64_CANCEL")
 report={"status":"PASS","fidelity_records":5000,"operator_records":88,"source_tasks":200,"block_image_rows":len(block_rows),"block_summary_rows":len(block_summary),"cls_patch_image_rows":len(cls_rows),"cls_patch_summary_rows":len(cls_summary),"raw_maps":len(cmap),"fd_failures":len(diag),"float64_cancellation_targets":review["target_count"],"new_confirmatory_tests":0};(ROOT/"independent_validation.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
if __name__=="__main__":main()
