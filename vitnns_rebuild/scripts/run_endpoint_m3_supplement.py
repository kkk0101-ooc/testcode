#!/usr/bin/env python3
"""Prepare and execute the CPU-only M3-SUP-R1 descriptive supplement."""
import argparse,csv,hashlib,json,math,os,sys,time
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OPENBLAS_NUM_THREADS"]="1"
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns.endpoint_m3_statistics import bootstrap_median
SPEC=REPO/"config/nns_vit_endpoint/endpoint_m3_supplement_r1.json";SOURCE=REPO/"reports/nns_vit_endpoint_m3_r1_20260919";ROOT=REPO/"reports/nns_vit_endpoint_m3_sup_r1_20260919"
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+"\n")
def dump_csv(path,rows):
 with path.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,separators=(",",":")) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
def quantiles(values):
 a=np.asarray(values,float);return {"n":len(a),"minimum":float(a.min()),"median":float(np.median(a)),"q90":float(np.quantile(a,.9)),"q99":float(np.quantile(a,.99)),"maximum":float(a.max())}
def prepare():
 spec=load(SPEC)
 if ROOT.exists():raise ValueError("FRESH_ROOT_REQUIRED")
 for path,expected in spec["source_bindings"].items():
  if sha(REPO/path)!=expected:raise ValueError("SOURCE_DRIFT:"+path)
 terminal=load(SOURCE/"terminal_receipt.json");assert terminal["status"]=="ENDPOINT_M3_PASS_EXECUTION_AND_AUDIT" and terminal["next_authorization"]=="STOP_AFTER_M3"
 ROOT.mkdir();write(ROOT/"source_validation.json",{"status":"PASS","source_terminal":terminal["status"],"source_bindings":spec["source_bindings"],"source_root_immutable":True})
 sources=[SPEC,Path(__file__),REPO/"docs/endpoint_m3_supplement_protocol_20260919.md",
          REPO/"scripts/run_endpoint_m3_supplement_gpu.py",
          REPO/"scripts/run_endpoint_m3_supplement_diagnostics.py",
          REPO/"src/vitnns/endpoint_m3_supplement.py",
          REPO/"tests/test_endpoint_m3_supplement.py"]
 write(ROOT/"freeze.json",{"status":"FROZEN","time_ns":time.time_ns(),"sources":{p.relative_to(REPO).as_posix():sha(p) for p in sources},"source_terminal_sha256":sha(SOURCE/"terminal_receipt.json")});print("M3_SUP_PREPARED")
def execute():
 spec=load(SPEC);freeze=load(ROOT/"freeze.json")
 for p,s in freeze["sources"].items():
  if sha(REPO/p)!=s:raise ValueError("FROZEN_SOURCE_DRIFT:"+p)
 fidelity=load(SOURCE/"m3a_fidelity.json")["records"];rows=[]
 metrics=("relative_l2","max_abs","max_tolerance_ratio")
 groups=[("overall","all",fidelity)]
 for dtype in ("float32","float64"):groups.append(("dtype",dtype,[r for r in fidelity if r["dtype"]==dtype]))
 for model in ("fcn","cnn","cnn_maxpool","residual_mlp","residual_cnn"):
  groups.append(("model",model,[r for r in fidelity if r["model"]==model]))
  for dtype in ("float32","float64"):groups.append(("model_dtype",model+"|"+dtype,[r for r in fidelity if r["model"]==model and r["dtype"]==dtype]))
 for group,key,part in groups:
  for metric in metrics:rows.append({"group":group,"key":key,"metric":metric,**quantiles([r[metric] for r in part])})
 dump_csv(ROOT/"m3a_error_distribution.csv",rows)
 operators=[]
 op=load(SOURCE/"m3a_operator_validation.json")
 for name in op["operators"]:
  part=[r for r in op["records"] if r["operator"]==name];operators.append({"operator":name,"directions":len(part),"dtype":"float64","max_analytic_autograd_jvp_relative_error":max(r["analytic_autograd_jvp_relative_error"] for r in part),"max_jvp_vjp_identity_relative_error":max(r["jvp_vjp_identity_relative_error"] for r in part),"max_best_finite_difference_relative_error":max(r["finite_difference_best_relative_error"] for r in part),"epsilons":[x["epsilon"] for x in part[0]["finite_difference_curve"]],"analytic_jacobian":True,"autograd_jvp_vjp":True})
 dump_csv(ROOT/"operator_validation_table.csv",operators)
 tasks=load(SOURCE/"task_manifest.json")["tasks"];image_rows=[];cls_patch=[]
 strata={"all":range(12),"early":range(4),"middle":range(4,8),"late":range(8,12),**{f"block_{b:02d}":[b] for b in range(12)}}
 for task in tasks:
  result=load(SOURCE/"results/attempts"/task["task_id"]/"attempt-01.json")["result"];archive=np.load(REPO/result["metrics_path"]);data={k:archive[k] for k in archive.files};block=data["block"];token=data["token"];mass=data["patch_absolute_mass_196"].astype(float);entropy=data["normalized_spatial_entropy"].astype(float);cancellation=data["cancellation"].astype(float);net_ratio=1-cancellation;locality=np.full(len(block),np.nan);distance=np.full(len(block),np.nan)
  for i in np.flatnonzero(token>0):
   p=mass[i]/mass[i].sum();tr,tc=divmod(int(token[i])-1,14);rr,cc=np.indices((14,14));dr=np.abs(rr-tr);dc=np.abs(cc-tc);near=np.maximum(dr,dc)<=1;locality[i]=p.reshape(14,14)[near].sum()-near.sum()/196;distance[i]=(p.reshape(14,14)*np.sqrt(dr**2+dc**2)/(13*np.sqrt(2))).sum()
  arrays={"entropy":entropy,"locality":locality,"distance":distance,"cancellation":cancellation,"net_ratio":net_ratio}
  for b in range(12):
   for metric,array in arrays.items():
    valid=(block==b)&np.isfinite(array);image_rows.append({"model_id":result["model_id"],"sample_id":result["sample_id"],"block":b,"metric":metric,"valid_units":int(valid.sum()),"median":float(np.median(array[valid]))})
  for stratum,blocks in strata.items():
   scope=np.isin(block,list(blocks));cls=entropy[scope&(token==0)&np.isfinite(entropy)];patch=entropy[scope&(token>0)&np.isfinite(entropy)];cls_patch.append({"model_id":result["model_id"],"sample_id":result["sample_id"],"stratum":stratum,"cls_units":len(cls),"patch_units":len(patch),"cls_median_entropy":float(np.median(cls)),"patch_median_entropy":float(np.median(patch)),"paired_difference":float(np.median(cls)-np.median(patch))})
 dump_csv(ROOT/"block_image_values.csv",image_rows);dump_csv(ROOT/"cls_patch_image_values.csv",cls_patch)
 block_summary=[]
 for key in sorted({(r["model_id"],r["block"],r["metric"]) for r in image_rows}):
  values=[r["median"] for r in image_rows if (r["model_id"],r["block"],r["metric"])==key];block_summary.append({"model":key[0],"block":key[1],"metric":key[2],"images":len(values),"median":float(np.median(values)),"ci95":bootstrap_median(values,"M3SUP|block|"+"|".join(map(str,key)),10000)})
 cls_summary=[]
 for key in sorted({(r["model_id"],r["stratum"]) for r in cls_patch}):
  part=[r for r in cls_patch if (r["model_id"],r["stratum"])==key];values=[r["paired_difference"] for r in part];cls_summary.append({"model":key[0],"stratum":key[1],"images":len(values),"median_difference":float(np.median(values)),"ci95":bootstrap_median(values,"M3SUP|cls_patch|"+"|".join(key),10000),"positive":sum(v>1e-12 for v in values),"negative":sum(v< -1e-12 for v in values),"ties":sum(abs(v)<=1e-12 for v in values)})
 dump_csv(ROOT/"block_summary.csv",block_summary);dump_csv(ROOT/"cls_patch_summary.csv",cls_summary)
 write(ROOT/"analysis_receipt.json",{"status":"PASS","m3a_error_rows":len(rows),"operator_rows":len(operators),"block_image_rows":len(image_rows),"block_summary_rows":len(block_summary),"cls_patch_image_rows":len(cls_patch),"cls_patch_summary_rows":len(cls_summary),"new_confirmatory_tests":0});print(json.dumps(load(ROOT/"analysis_receipt.json")))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--execute",action="store_true");a=p.parse_args();prepare() if a.prepare else execute() if a.execute else p.error("choose one")
