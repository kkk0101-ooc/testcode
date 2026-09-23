#!/usr/bin/env python3
"""Correct the class-target residual diagram, keeping signed source maps intact."""
import hashlib,json,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from render_revision_v6 import R,P,box,arrow,BLUE,TEAL,ORANGE,INK

def main():
 manifest=json.loads((P/'figure_manifest.json').read_text());row=next(x for x in manifest['figures'] if x['number']==7)
 meta=json.loads((R/'reports/nns_vit_endpoint_e3_closure_r1_20260918/formal_results/attempts/deit_s16__014/attempt-01.json').read_text())['result'];ar=next(r for r in meta['artifacts'] if r['block']==5);path=R/ar['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==ar['sha256'];arr=np.load(path);maps=dict(zip(ar['component_order'],arr))
 with np.load(P/'assets/deit_s16__014.npz') as z:photo=z['input_rgb']
 fig=plt.figure(figsize=(13,7));fig.suptitle('Residual identity + attention contribution = Total Endpoint',fontsize=18,color=INK)
 ax=fig.add_axes([.04,.58,.92,.30]);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
 box(ax,.05,.32,.14,.27,'u',BLUE);box(ax,.31,.60,.23,.27,'attention f(u)',TEAL);box(ax,.66,.32,.10,.27,'+',ORANGE);box(ax,.84,.32,.13,.27,'target s',BLUE)
 arrow(ax,(.19,.52),(.30,.72),TEAL);arrow(ax,(.55,.72),(.70,.60),TEAL)
 ax.plot([.13,.13,.71,.71],[.31,.10,.10,.31],color=BLUE,lw=2);arrow(ax,(.71,.18),(.71,.31));arrow(ax,(.77,.45),(.83,.45));ax.text(.42,.15,'identity bypass',ha='center',color=BLUE)
 for i,(label,value) in enumerate([('Input',photo),('Identity local map',maps['identity']),('Attention contribution',maps['attention']),('Total Endpoint',maps['L_F'])]):
  a=fig.add_axes([.025+i*.245,.08,.23,.40]);a.axis('off');a.set_title(label,fontsize=12)
  if i==0:a.imshow(value)
  else:
   v=np.maximum(value.sum(0),0);q=np.quantile(v[v>0],.99) if np.any(v>0) else 1;a.imshow(np.sqrt(np.clip(v/q,0,1)),cmap='turbo',vmin=0,vmax=1)
 fig.text(.5,.035,'Signed arrays add exactly within tolerance; positive display panels are independently normalized.',ha='center',fontsize=10)
 for ext,rec in row['files'].items():
  p=P/rec['path'];fig.savefig(p,dpi=250,facecolor='white');rec['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 plt.close(fig);row['title']='Residual global reconstruction of Total Endpoint'
 (P/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+chr(10))
 bound=json.loads((P.parent/'figure_manifest.json').read_text());b=next(x for x in bound['figures'] if x['figure']==7)
 for ext,rec in row['files'].items():b[ext]['sha256']=rec['sha256']
 (P.parent/'figure_manifest.json').write_text(json.dumps(bound,indent=2)+chr(10));print('residual figure corrected',path)
if __name__=='__main__':main()
