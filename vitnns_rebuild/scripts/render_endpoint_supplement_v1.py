#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['SOURCE_DATE_EPOCH']='1789862400'
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/'reports/nns_vit_endpoint_supplement_v1_20260920';OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'svg.fonttype':'none','pdf.fonttype':42})
COLORS=['#1764a0','#cc7927','#238880','#9870ad','#758497'];models=['vit_b16','deit_s16'];labels={'vit_b16':'ViT-B/16','deit_s16':'DeiT-S/16'};figures=[]
def load(n):return json.loads((ROOT/n).read_text())
def save(fig,n,slug,caption):
 files={}
 for ext in ('png','pdf','svg'):
  p=OUT/f'sup_{n:02d}_{slug}.{ext}';fig.savefig(p,dpi=190,bbox_inches='tight',facecolor='white')
  if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
  files[ext]=dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 figures.append(dict(number=n,slug=slug,caption=caption,files=files));plt.close(fig)
# Corrected baseline and paired effects, each group contains 400 independent images.
b=load('baseline_analysis.json')['summary'];fig,ax=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True);x=np.arange(4)
for j,(key,label) in enumerate([('random_legacy','Legacy abs(mean signed)'),('random_corrected','Mean absolute random'),('endpoint','Endpoint'),('taylor','Taylor')]):
 ax[0].bar(x+(j-1.5)*.2,[r[key]['median'] for r in b],.19,label=label,color=COLORS[j])
for j,(key,label) in enumerate([('endpoint_minus_random','Endpoint - random'),('taylor_minus_random','Taylor - random')]):
 vals=np.array([r[key]['median'] for r in b]);ci=np.array([r[key]['median_ci95'] for r in b]);ax[1].errorbar(x+(j-.5)*.13,vals,yerr=np.vstack((vals-ci[:,0],ci[:,1]-vals)),fmt='o',capsize=4,label=label,color=COLORS[j])
for a in ax:a.set_xticks(x,['ViT zero','ViT blur','DeiT zero','DeiT blur']);a.legend(fontsize=8);a.set_axisbelow(True)
ax[0].set_ylabel('Median absolute-effect AUC');ax[1].set_ylabel('Paired median AUC difference');ax[1].axhline(0,color='black',lw=.8);fig.suptitle('Absolute value before random-repeat averaging changes the comparison',fontsize=14)
save(fig,1,'random_correction','400 images per model/reference; intervals bootstrap images. Legacy curves are retained only to quantify estimator correction.')
# Training depth curves, observed seed range distinct from image uncertainty.
for n,study,metrics in [(2,'m3',[('entropy','Spatial entropy'),('locality','Own-token locality'),('distance','Normalized distance')]),(3,'m4',[('routing_scale','Routing scale'),('value_error','Value-only error'),('attention_identity_scale','Attention identity'),('mlp_identity_scale','MLP identity')])]:
 rows=list(csv.DictReader((ROOT/f'training_{study}_block_values.csv').open()));fig,axes=plt.subplots(2,len(metrics),figsize=(4*len(metrics),6.3),sharex=True,constrained_layout=True)
 for i,model in enumerate(models):
  for j,(metric,label) in enumerate(metrics):
   ax=axes[i,j];trained=[];random=[]
   for block in range(12):
    per=[r for r in rows if r['model']==model and int(r['block'])==block];trained.append(np.mean([float(r[metric]) for r in per if r['state']=='trained']));random.append([np.mean([float(r[metric]) for r in per if r['state']==seed]) for seed in ('202609201','202609202','202609203')])
   random=np.array(random);ax.plot(range(12),trained,color=COLORS[0],lw=2,label='Trained');ax.plot(range(12),random.mean(1),color=COLORS[1],lw=2,label='Random seed mean');ax.fill_between(range(12),random.min(1),random.max(1),color=COLORS[1],alpha=.18,label='Range of 3 seed means');ax.set_title(labels[model]+' | '+label);ax.set_xticks([0,3,5,8,11]);ax.set_xlabel('Block')
 axes[0,0].legend(fontsize=8);fig.suptitle('Training and initialization controls | 32 shared images per model',fontsize=14)
 save(fig,n,study+'_training','Within-image unit medians, then image means; shading is the observed range of three seed means, not a confidence interval.')
# Target-conditioned changes across B0 and B11; separate correct/incorrect.
t=load('target_analysis.json')['summary'];metrics=[('routing_scale','Routing scale'),('value_error','Value-only error'),('attention_identity_scale','Attention identity'),('mlp_identity_scale','MLP identity')];fig,axes=plt.subplots(2,4,figsize=(15,6.7),constrained_layout=True)
for i,model in enumerate(models):
 for j,(metric,label) in enumerate(metrics):
  ax=axes[i,j]
  for k,target in enumerate(('gt','top1','margin')):
   rows=[next(r for r in t if r['model']==model and r['subset']==group and r['target']==target and r['metric']==metric and r['quantity']=='late_minus_early') for group in ('correct','incorrect')];v=np.array([r['mean'] for r in rows]);ci=np.array([r['ci95'] for r in rows]);ax.errorbar(np.arange(2)+(k-1)*.14,v,yerr=np.vstack((v-ci[:,0],ci[:,1]-v)),fmt='o',capsize=3,color=COLORS[k],label=target)
  ax.axhline(0,color='black',lw=.7);ax.set_xticks([0,1],['Correct','Incorrect']);ax.set_title(labels[model]+' | '+label);ax.set_ylabel('B11 - B0')
axes[0,0].legend(fontsize=8);fig.suptitle('Direction persists across GT, top1 and margin; magnitude depends on target',fontsize=14)
save(fig,4,'target_robustness','95% image-bootstrap intervals; ViT-B correct/incorrect n=92/8, DeiT-S n=86/14. Correct GT/top1 points overlap by construction.')
# All four similarity metrics across controls.
f=load('feature_analysis.json')['summary'];fig,axes=plt.subplots(2,4,figsize=(15,6.4),constrained_layout=True);controls=['paired','wrong_channel','cross_layer','permuted','random_channel']
for i,model in enumerate(models):
 for j,metric in enumerate(('spearman','cosine','js_distance','top20_overlap')):
  ax=axes[i,j];rows=[next(r for r in f if r['model']==model and r['phase']=='all' and r['metric']==metric and r['control']==c) for c in controls];v=np.array([r['mean'] for r in rows]);ci=np.array([r['ci95'] for r in rows]);ax.bar(range(5),v,color=COLORS);ax.errorbar(range(5),v,yerr=np.vstack((v-ci[:,0],ci[:,1]-v)),fmt='none',color='black',capsize=3);ax.set_xticks(range(5),['Paired','Wrong ch.','Cross layer','Permuted','Random ch.'],rotation=35,ha='right',fontsize=8);ax.set_title(labels[model]+' | '+metric)
fig.suptitle('Feature energy and Endpoint mass relate beyond four pairing controls',fontsize=14)
save(fig,5,'feature_relation','32 images/model; 12 blocks and 16 frozen channels/block; within-image medians precede image bootstrap. Lower Jensen-Shannon distance means greater similarity.')
(ROOT/'figure_manifest.json').write_text(json.dumps(dict(figures=figures),indent=2)+'\n');print(json.dumps(dict(figures=len(figures))))
