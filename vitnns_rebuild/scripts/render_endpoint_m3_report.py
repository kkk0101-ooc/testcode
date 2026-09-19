#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import csv,hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"reports/nns_vit_endpoint_m3_r1_20260919"
PAPER=REPO/"papers/nns_vit_endpoint/figures/m3"
def rows(name):
 with (ROOT/name).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,name,description):
 records=[];paper_records=[];PAPER.mkdir(parents=True,exist_ok=True)
 for suffix in ("png","pdf"):
  path=ROOT/f"{name}.{suffix}";paper=PAPER/f"{name}.{suffix}";fig.savefig(path,dpi=180,bbox_inches="tight");shutil.copyfile(path,paper);records.append({"path":path.relative_to(REPO).as_posix(),"sha256":sha(path),"bytes":path.stat().st_size});paper_records.append({"path":paper.relative_to(REPO).as_posix(),"sha256":sha(paper),"bytes":paper.stat().st_size})
 plt.close(fig);return {"id":name,"description":description,"files":records,"paper_files":paper_records}
def main():
 summary=rows("atlas_summary.csv");similarity=rows("similarity_summary.csv");figures=[];models=("vit_b16","deit_s16");colors={"vit_b16":"#2266aa","deit_s16":"#cc6633"}
 fig,axes=plt.subplots(2,2,figsize=(9,6));metrics=("entropy","locality","distance","cancellation")
 for ax,metric in zip(axes.flat,metrics):
  for model in models:
   vals=[]
   for phase in ("early","middle","late"):
    row=next(r for r in summary if r["model"]==model and r["metric"]==f"{phase}_{metric}");vals.append(float(row["median"]))
   ax.plot((0,1,2),vals,marker="o",label=model,color=colors[model])
  ax.set_xticks((0,1,2),("early","middle","late"));ax.set_title(metric);ax.grid(alpha=.2)
 axes[0,0].legend();fig.suptitle("M3 token-channel Endpoint atlas: image-level medians");figures.append(save(fig,"m3_atlas_depth","Image-level median atlas metrics across frozen depth phases."))
 fig,axes=plt.subplots(1,2,figsize=(9,3.5))
 for ax,metric in zip(axes,("patch_absolute_cosine","patch_signed_cosine")):
  for model in models:
   for group,style in (("same_token_different_channel","-"),("different_token","--")):
    vals=[]
    for block in range(12):
     row=next(r for r in similarity if r["model"]==model and int(r["block"])==block and r["group"]==group and r["metric"]==metric);vals.append(float(row["median"]))
    ax.plot(range(12),vals,style,color=colors[model],label=model+" "+("same token" if group.startswith("same") else "different token"))
  ax.set_title(metric);ax.set_xlabel("block");ax.grid(alpha=.2)
 axes[0].legend(fontsize=7);fig.suptitle("Frozen unit-pair map similarity");figures.append(save(fig,"m3_map_similarity","Median image-level cosine for 256 frozen pairs per block and group."))
 manifest={"stage":"ENDPOINT-M3","figures":figures};(ROOT/"figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");(REPO/"papers/nns_vit_endpoint/figure_manifest_m3.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps({"figures":len(figures),"files":sum(len(x["files"]) for x in figures)}))
if __name__=="__main__":main()
