#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OPENBLAS_NUM_THREADS"]="1"
import csv,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from run_endpoint_m3 import ROOT,SPEC,load,write
from vitnns.endpoint_m3_protocol import similarity_pairs
from vitnns.endpoint_m3_statistics import image_metric_values,summarize_image_values,cosine

def dump_csv(path,rows):
 with path.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,separators=(",",":")) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
def main():
 spec=load(SPEC);tasks=load(ROOT/"task_manifest.json")["tasks"];image_values=[];similarity=[]
 for ti,task in enumerate(tasks):
  tid=task["task_id"];r=load(ROOT/"results"/"attempts"/tid/"attempt-01.json")["result"];archive=np.load(REPO/r["metrics_path"]);data={k:archive[k] for k in archive.files};v=image_metric_values(data);v.update(model_id=r["model_id"],sample_id=r["sample_id"],image_sha256=r["image_sha256"]);image_values.append(v)
  tokens=tuple(task["payload"]["tokens"]);model=r["model_id"]
  for block in range(12):
   channels=tuple(task["payload"]["channels"][str(block)]);index={(int(data["token"][i]),int(data["channel"][i])):i for i in np.flatnonzero(data["block"]==block)}
   for group in ("same_token_different_channel","different_token"):
    pairs=similarity_pairs(model,block,tokens,channels,group,spec["analysis"]["similarity_pairs_per_layer_group"],spec["analysis"]["similarity_namespace"]);absolute=[];signed=[]
    for left,right in pairs:
     i,j=index[left],index[right];absolute.append(cosine(data["patch_absolute_mass_196"][i],data["patch_absolute_mass_196"][j]));signed.append(cosine(data["patch_signed_sum_196"][i],data["patch_signed_sum_196"][j]))
    for metric,values in (("patch_absolute_cosine",absolute),("patch_signed_cosine",signed)):
     valid=[x for x in values if x is not None];similarity.append({"model_id":model,"sample_id":r["sample_id"],"block":block,"phase":"early" if block<4 else "middle" if block<8 else "late","group":group,"metric":metric,"valid_pairs":len(valid),"na_pairs":len(values)-len(valid),"median":None if not valid else float(np.median(valid))})
  if ti%20==0:print("analyzed",ti+1,flush=True)
 summaries,tests=summarize_image_values(image_values,spec["analysis"]["bootstrap_repetitions"]);dump_csv(ROOT/"image_values.csv",image_values);dump_csv(ROOT/"atlas_summary.csv",summaries);dump_csv(ROOT/"similarity_image_values.csv",similarity);write(ROOT/"atlas_tests.json",{"test_count":len(tests),"family_size_per_model":5,"tests":tests})
 # Separate similarity description: image medians by exact layer/group and phase-level image aggregates.
 sim_summary=[]
 for key in sorted({(r["model_id"],r["block"],r["group"],r["metric"]) for r in similarity}):
  values=[r["median"] for r in similarity if (r["model_id"],r["block"],r["group"],r["metric"])==key and r["median"] is not None];sim_summary.append({"model":key[0],"block":key[1],"phase":"early" if key[1]<4 else "middle" if key[1]<8 else "late","group":key[2],"metric":key[3],"images":len(values),"median":None if not values else float(np.median(values))})
 # Explicit early/middle/late comparison: within each image, median the four layer summaries.
 sim_phase=[]
 for key in sorted({(r["model_id"],r["sample_id"],r["phase"],r["group"],r["metric"]) for r in similarity}):
  values=[r["median"] for r in similarity if (r["model_id"],r["sample_id"],r["phase"],r["group"],r["metric"])==key and r["median"] is not None];sim_phase.append({"model_id":key[0],"sample_id":key[1],"phase":key[2],"group":key[3],"metric":key[4],"valid_layers":len(values),"median":None if not values else float(np.median(values))})
 dump_csv(ROOT/"similarity_summary.csv",sim_summary);dump_csv(ROOT/"similarity_phase_image_values.csv",sim_phase);write(ROOT/"analysis_receipt.json",{"status":"PASS","image_records":len(image_values),"summary_rows":len(summaries),"tests":len(tests),"similarity_image_rows":len(similarity),"similarity_phase_image_rows":len(sim_phase),"similarity_summary_rows":len(sim_summary),"statistical_unit":"image","within_image_aggregation":"median"});print(json.dumps(load(ROOT/"analysis_receipt.json")))
if __name__=="__main__":main()
