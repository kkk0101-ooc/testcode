#!/usr/bin/env python3
"""Independent source-to-table audit of M3 image-level statistics."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OPENBLAS_NUM_THREADS"]="1"
import csv,json,math,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"reports/nns_vit_endpoint_m3_r1_20260919"
def load(p):return json.loads(p.read_text())
def require(x,c):
 if not x:raise ValueError(c)
def close(a,b,c):require(math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-12),c)
def metric_values(data):
 block=data["block"];token=data["token"];zero=data["zero_gradient"];entropy=data["normalized_spatial_entropy"].astype(float);cancellation=data["cancellation"].astype(float);mass=data["patch_absolute_mass_196"].astype(float);locality=np.full(len(block),np.nan);distance=np.full(len(block),np.nan)
 for i in np.flatnonzero((token>0)&~zero):
  p=mass[i]/mass[i].sum();r,c=divmod(int(token[i])-1,14);rr,cc=np.indices((14,14));dr=np.abs(rr-r);dc=np.abs(cc-c);mask=np.maximum(dr,dc)<=1;locality[i]=p.reshape(14,14)[mask].sum()-mask.sum()/196;distance[i]=(p.reshape(14,14)*np.sqrt(dr**2+dc**2)/(13*np.sqrt(2))).sum()
 arrays={"entropy":entropy,"locality":locality,"distance":distance,"cancellation":cancellation};result={}
 for phase,blocks in {"early":range(4),"middle":range(4,8),"late":range(8,12)}.items():
  for name,array in arrays.items():result[f"{phase}_{name}"]=float(np.median(array[np.isin(block,list(blocks))&np.isfinite(array)]))
 for kind,mask0 in (("cls",token==0),("patch",token>0)):
  for name,array in arrays.items():
   valid=mask0&np.isfinite(array);result[f"{kind}_{name}"]=None if not valid.any() else float(np.median(array[valid]))
 for name in arrays:result[f"late_minus_early_{name}"]=result[f"late_{name}"]-result[f"early_{name}"]
 result["cls_minus_patch_entropy"]=result["cls_entropy"]-result["patch_entropy"];return result
def main():
 tasks=load(ROOT/"task_manifest.json")["tasks"];computed=[]
 for task in tasks:
  r=load(ROOT/"results"/"attempts"/task["task_id"]/"attempt-01.json")["result"];archive=np.load(REPO/r["metrics_path"]);data={k:archive[k] for k in archive.files};row=metric_values(data);row.update(model_id=r["model_id"],sample_id=r["sample_id"]);computed.append(row)
 observed=list(csv.DictReader((ROOT/"image_values.csv").open()));index={(r["model_id"],r["sample_id"]):r for r in computed};require(len(observed)==len(index)==200,"IMAGE_ROWS")
 fields=("early_entropy","middle_entropy","late_entropy","early_locality","middle_locality","late_locality","early_distance","middle_distance","late_distance","early_cancellation","middle_cancellation","late_cancellation","cls_entropy","patch_entropy","late_minus_early_entropy","late_minus_early_locality","late_minus_early_distance","late_minus_early_cancellation","cls_minus_patch_entropy")
 for row in observed:
  expected=index[(row["model_id"],row["sample_id"])]
  for field in fields:close(row[field],expected[field],"IMAGE_VALUE:"+field)
 published=load(ROOT/"atlas_tests.json")["tests"];tests=[]
 for model in ("deit_s16","vit_b16"):
  family=[]
  for contrast in ("late_minus_early_entropy","late_minus_early_locality","late_minus_early_distance","late_minus_early_cancellation","cls_minus_patch_entropy"):
   values=np.asarray([r[contrast] for r in computed if r["model_id"]==model]);positive=int((values>1e-12).sum());negative=int((values< -1e-12).sum());n=positive+negative;p=1.0 if not n else float(binomtest(positive,n,.5).pvalue);family.append({"test_id":f"{model}|{contrast}","model":model,"contrast":contrast,"n":len(values),"median_difference":float(np.median(values)),"positive":positive,"negative":negative,"ties":len(values)-n,"p":p})
  order=sorted(range(5),key=lambda i:(family[i]["p"],family[i]["test_id"]));previous=0
  for rank,i in enumerate(order):previous=max(previous,min(1,(5-rank)*family[i]["p"]));family[i]["holm_p"]=previous;family[i]["reject"]=previous<.05
  tests.extend(family)
 require(tests==published,"TEST_RECOMPUTE");report={"status":"PASS","source_tasks":200,"image_rows_recomputed":200,"fields_per_image_checked":len(fields),"tests_recomputed":10,"holm_families":2};(ROOT/"independent_analysis_audit.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
if __name__=="__main__":main()
