#!/usr/bin/env python3
"""Float64 recovery for all tasks selected by immutable float32 failures."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from run_endpoint_e4_intervention_worker import load_model,load_inputs
from vitnns.endpoint_m3_protocol import rademacher
ROOT=REPO/"reports/nns_vit_endpoint_m3_r1_20260919"
def load(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");a=p.parse_args();device=torch.device(a.device);spec=load(REPO/"config/nns_vit_endpoint/endpoint_m3.json");failure=load(ROOT/"initial_numeric_validation_failure.json");manifest={x["task_id"]:x["payload"] for x in load(ROOT/"task_manifest.json")["tasks"]};expected=sorted({x["task_id"] for x in failure["failures"]});torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;records=[];models={}
 for task_id in expected:
  task=manifest[task_id];model_id=task["model_id"]
  if model_id not in models:models[model_id]=load_model(task,device)[0].double()
  model=models[model_id];image,_,_=load_inputs(task,device);image=image.double();captured={};handles=[model.blocks[b].register_forward_hook(lambda _m,_a,out,b=b:captured.__setitem__(b,out)) for b in range(12)];clean=image.detach().clone().requires_grad_(True);model(clean);units=[(b,t,c) for b in range(12) for t in task["tokens"][:2] for c in task["channels"][str(b)][:4]];directions=[torch.as_tensor(rademacher((3,224,224),task["direction_namespace"],model_id,task["sample_id"],i)/np.sqrt(3*224*224),device=device,dtype=torch.float64).unsqueeze(0) for i in range(task["directions"])];analytic=[]
  for b,t,c in units:
   gradient=torch.autograd.grad(captured[b][0,t,c],clean,retain_graph=True)[0];analytic.append([float((gradient*d).sum()) for d in directions])
  def values(x):
   with torch.no_grad():model(x)
   return torch.stack([captured[b][0,t,c] for b,t,c in units])
  output=[]
  for direction,d in enumerate(directions):
   expected_values=torch.tensor([x[direction] for x in analytic],device=device,dtype=torch.float64);curves=[]
   for epsilon in spec["validation_subset"]["finite_difference_epsilons_float64"]:
    observed=(values(clean.detach()+epsilon*d)-values(clean.detach()-epsilon*d))/(2*epsilon);curves.append({"epsilon":epsilon,"finite_difference":observed.cpu().tolist(),"analytic":expected_values.cpu().tolist()})
   output.append({"direction":direction,"curves":curves})
  for h in handles:h.remove()
  records.append({"task_id":task_id,"model_id":model_id,"sample_id":task["sample_id"],"image_sha256":task["image_sha256"],"targets":[{"block":b,"token":t,"channel":c} for b,t,c in units],"directions":output});print(task_id,flush=True)
 result={"status":"PENDING_VALIDATION","amendment":"docs/endpoint_m3_numeric_recovery_amendment_20260919.md","source_failure_sha256":hashlib.sha256((ROOT/"initial_numeric_validation_failure.json").read_bytes()).hexdigest(),"selected_task_ids":expected,"records":records};(ROOT/"m3b_float64_recovery.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"tasks":len(records),"conditions":len(records)*96*3}))
if __name__=="__main__":main()
