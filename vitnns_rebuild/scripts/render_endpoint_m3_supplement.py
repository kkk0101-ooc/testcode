#!/usr/bin/env python3
import csv,hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"reports/nns_vit_endpoint_m3_sup_r2_20260919";PAPER=REPO/"papers/nns_vit_endpoint/figures/m3_sup_r2"
def rows(name):
 with (ROOT/name).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,name,description,paper_number):
 PAPER.mkdir(parents=True,exist_ok=True);files=[];paper=[]
 for suffix in ("png","pdf"):
  p=ROOT/f"{name}.{suffix}";q=PAPER/f"{name}.{suffix}";fig.savefig(p,dpi=190,bbox_inches="tight");shutil.copyfile(p,q);files.append({"path":p.relative_to(REPO).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size});paper.append({"path":q.relative_to(REPO).as_posix(),"sha256":sha(q),"bytes":q.stat().st_size})
 plt.close(fig);return {"id":name,"paper_number":paper_number,"description":description,"files":files,"paper_files":paper}
def main():
 figs=[];colors={"vit_b16":"#2266aa","deit_s16":"#cc6633"};block=rows("block_summary.csv")
 fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
 for ax,metric in zip(axes.flat,("entropy","locality","distance","net_ratio")):
  for model in colors:
   part=sorted((x for x in block if x["model"]==model and x["metric"]==metric),key=lambda x:int(x["block"]));x=np.array([int(r["block"]) for r in part]);y=np.array([float(r["median"]) for r in part]);ci=np.array([json.loads(r["ci95"]) for r in part]);ax.plot(x,y,"o-",label=model,color=colors[model],ms=3);ax.fill_between(x,ci[:,0],ci[:,1],color=colors[model],alpha=.14)
  ax.set_title(metric);ax.set_xticks(range(12));ax.grid(alpha=.2);ax.set_xlabel("block")
 axes[0,0].legend();fig.suptitle("M3: 12-block spatial broadening and net-ratio trajectories");figs.append(save(fig,"m3_sup_block_trajectories","Image-level medians and 95% bootstrap intervals over all twelve blocks.",29))
 cls=rows("cls_patch_summary.csv");fig,ax=plt.subplots(figsize=(9,4))
 for model in colors:
  part=[next(x for x in cls if x["model"]==model and x["stratum"]==f"block_{b:02d}") for b in range(12)];y=np.array([float(x["median_difference"]) for x in part]);ci=np.array([json.loads(x["ci95"]) for x in part]);ax.plot(range(12),y,"o-",label=model,color=colors[model]);ax.fill_between(range(12),ci[:,0],ci[:,1],color=colors[model],alpha=.14)
 ax.axhline(0,color="black",lw=.8);ax.set(xlabel="block",ylabel="CLS - patch entropy",title="Image-level paired CLS-patch entropy difference");ax.set_xticks(range(12));ax.grid(alpha=.2);ax.legend();figs.append(save(fig,"m3_sup_cls_patch_by_block","Paired image-level CLS-minus-patch entropy by block.",30))
 cancel=rows("cancellation_summary.csv");fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,model in zip(axes,colors):
  for metric,label in (("within_rgb_channels","RGB channel"),("within_patch_after_rgb_sum","within patch"),("across_patch_signed","across patch"),("whole_image_cancellation","whole image")):
   part=sorted((x for x in cancel if x["model"]==model and x["token_type"]=="patch" and x["metric"]==metric),key=lambda x:int(x["block"]));ax.plot([int(x["block"]) for x in part],[float(x["median"]) for x in part],"o-",label=label)
  ax.set(title=model,xlabel="block",ylabel="cancellation ratio");ax.set_xticks((0,5,11));ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle("Cancellation decomposed across nested spatial scales");figs.append(save(fig,"m3_sup_cancellation_scales","Raw-map cancellation at RGB, within-patch, across-patch, and whole-image scales.",31))
 err=rows("m3a_error_distribution.csv");fig,ax=plt.subplots(figsize=(8,4));models=("fcn","cnn","cnn_maxpool","residual_mlp","residual_cnn");x=np.arange(len(models));w=.35
 for j,dtype in enumerate(("float32","float64")):
  part=[next(r for r in err if r["group"]=="model_dtype" and r["key"]==m+"|"+dtype and r["metric"]=="relative_l2") for m in models];ax.bar(x+(j-.5)*w,[max(float(r["q99"]),1e-20) for r in part],w,label=dtype)
 ax.set_yscale("log");ax.set_xticks(x,models,rotation=20);ax.set_ylabel("99th-percentile relative L2");ax.set_title("Original-NNS vs Endpoint numerical equivalence");ax.legend();ax.grid(axis="y",alpha=.2);figs.append(save(fig,"m3_sup_equivalence_errors","99th-percentile relative L2 by network and precision.",32))
 fd=rows("finite_difference_diagnostics.csv");fig,ax=plt.subplots(figsize=(7,4))
 for dtype,color in (("float32","#bb3344"),("float64","#2266aa")):
  by={}
  for r in fd:
   a=float(r["analytic_"+dtype])
   for point in json.loads(r[dtype+"_curve"]):by.setdefault(float(point["epsilon"]),[]).append(abs(float(point["finite_difference"])-a))
  eps=sorted(by);ax.plot(eps,[np.median(by[e]) for e in eps],"o-",label=dtype,color=color);ax.fill_between(eps,[np.quantile(by[e],.1) for e in eps],[np.quantile(by[e],.9) for e in eps],color=color,alpha=.15)
 ax.set_xscale("log");ax.set_yscale("log");ax.set(xlabel="epsilon",ylabel="absolute FD-JVP error",title="Finite-difference diagnostics for the 36 float32 failures");ax.grid(alpha=.2);ax.legend();figs.append(save(fig,"m3_sup_fd_diagnostics","Median and 10th-90th percentile absolute FD error across the retained failure set.",33))
 manifest={"stage":"ENDPOINT-M3-SUP-R2","figures":figs};(ROOT/"figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");(REPO/"papers/nns_vit_endpoint/figure_manifest_m3_sup_r2.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps({"figures":len(figs),"files":sum(len(x["files"]) for x in figs)}))
if __name__=="__main__":main()
