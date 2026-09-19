#!/usr/bin/env python3
"""Float64 cancellation review for the pre-frozen M3 full-map targets."""
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("CUBLAS_PREIMPORT")
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"));sys.path.insert(0,str(REPO/"scripts"))
from run_endpoint_e4_intervention_worker import load_model,load_inputs
from vitnns.endpoint_m3_supplement import cancellation_metrics
SOURCE=REPO/"reports/nns_vit_endpoint_m3_r1_20260919";ROOT=REPO/"reports/nns_vit_endpoint_m3_sup_r1_20260919"
def load(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");a=p.parse_args();device=torch.device(a.device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;tasks=load(SOURCE/"task_manifest.json")["tasks"];selected=[]
 for model in ("deit_s16","vit_b16"):selected.extend([x for x in tasks if x["payload"]["model_id"]==model][:2])
 rows=[];models={};started=time.monotonic()
 for taskrow in selected:
  task=taskrow["payload"];model_id=task["model_id"]
  if model_id not in models:models[model_id]=load_model(task,device)[0].double()
  model=models[model_id];image,_,_=load_inputs(task,device);clean=image.double().detach().requires_grad_(True);captured={};handles=[model.blocks[b].register_forward_hook(lambda _m,_a,out,b=b:captured.__setitem__(b,out)) for b in (0,5,11)];model(clean)
  float32_path=SOURCE/"task_payloads"/taskrow["task_id"]/"full_maps.npz";stored=np.load(float32_path)
  for b in (0,5,11):
   channel=task["channels"][str(b)][0]
   for token in task["tokens"][:2]:
    key=f"b{b:02d}_t{token:03d}_c{channel:04d}";gradient=torch.autograd.grad(captured[b][0,token,channel],clean,retain_graph=True)[0][0];map64=(clean.detach()[0]*gradient).cpu().numpy();map32=stored[key];m32=cancellation_metrics(map32);m64=cancellation_metrics(map64);record={"task_id":taskrow["task_id"],"model_id":model_id,"sample_id":task["sample_id"],"block":b,"token":token,"channel":channel,"map_key":key,"float32":m32,"float64":m64,"absolute_map_max_difference":float(np.max(np.abs(map32.astype(np.float64)-map64)))}
    for field in ("within_rgb_channels","within_patch_after_rgb_sum","across_patch_signed","whole_image_cancellation","net_ratio"):record[field+"_absolute_difference"]=abs(m32[field]-m64[field])
    rows.append(record)
  for h in handles:h.remove()
 result={"status":"PASS","dtype":"float64","selected_task_ids":[x["task_id"] for x in selected],"target_count":len(rows),"records":rows,"max_cancellation_metric_absolute_difference":max(r[k] for r in rows for k in r if k.endswith("_absolute_difference") and k!="absolute_map_max_difference"),"wall_seconds":time.monotonic()-started};(ROOT/"cancellation_float64_review.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:result[k] for k in ("status","target_count","max_cancellation_metric_absolute_difference","wall_seconds")}))
if __name__=="__main__":main()
