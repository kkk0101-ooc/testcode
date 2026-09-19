#!/usr/bin/env python3
"""Small float64 central-difference review for M3's frozen 96-unit subset."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from run_endpoint_e4_intervention_worker import load_model,load_inputs
from vitnns.endpoint_m3_protocol import rademacher

def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();root=a.root;spec=json.loads((REPO/"config/nns_vit_endpoint/endpoint_m3.json").read_text());tasks=json.loads((root/"task_manifest.json").read_text())["tasks"];review=[];device=torch.device(a.device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for model_id in spec["models"]:
  selected=[t for t in tasks if t["payload"]["model_id"]==model_id][:spec["validation_subset"]["float64_review_images_per_model"]];model=None
  for taskrow in selected:
   task=taskrow["payload"]
   if model is None:model,_=load_model(task,device);model.double()
   image,_,_=load_inputs(task,device);image=image.double();captured={};handles=[model.blocks[b].register_forward_hook(lambda _m,_a,out,b=b:captured.__setitem__(b,out)) for b in range(12)]
   clean=image.detach().clone().requires_grad_(True);model(clean);units=[(b,t,c) for b in range(12) for t in task["tokens"][:2] for c in task["channels"][str(b)][:4]]
   analytic=[]
   directions=[torch.as_tensor(rademacher((3,224,224),task["direction_namespace"],model_id,task["sample_id"],i)/np.sqrt(3*224*224),device=device,dtype=torch.float64).unsqueeze(0) for i in range(task["directions"])]
   for b,t,c in units:
    gradient=torch.autograd.grad(captured[b][0,t,c],clean,retain_graph=True)[0];analytic.append([float((gradient*d).sum()) for d in directions])
   def values(x):
    with torch.no_grad():model(x)
    return torch.stack([captured[b][0,t,c] for b,t,c in units])
   per_direction=[]
   for direction,d in enumerate(directions):
    expected=torch.tensor([x[direction] for x in analytic],device=device,dtype=torch.float64);curves=[]
    for epsilon in spec["validation_subset"]["finite_difference_epsilons_float64"]:
     fd=(values(clean.detach()+epsilon*d)-values(clean.detach()-epsilon*d))/(2*epsilon);curves.append({"epsilon":epsilon,"finite_difference":fd.cpu().tolist(),"analytic":expected.cpu().tolist()})
    per_direction.append({"direction":direction,"curves":curves})
   for h in handles:h.remove()
   review.append({"model_id":model_id,"sample_id":task["sample_id"],"image_sha256":task["image_sha256"],"targets":[{"block":b,"token":t,"channel":c} for b,t,c in units],"directions":per_direction})
 result={"status":"PENDING_VALIDATION","dtype":"float64","records":review};path=root/"m3b_float64_review.json";path.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"records":len(review),"directions":sum(len(r["directions"]) for r in review)}))
if __name__=="__main__":main()
