#!/usr/bin/env python3
"""Render revision-v6 concept figures without reading experimental outcomes."""
import hashlib,json,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
R=Path(__file__).resolve().parents[3]
P=R/'papers/nns_vit_endpoint/publication_r2_20260920'
BLUE='#28628C';TEAL='#168B89';ORANGE='#D18132';RED='#BE4248';INK='#21384A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','pdf.fonttype':42})
def tx(s):return s.replace('§',chr(92))
def box(ax,x,y,w,h,text,color=BLUE,size=12,dashed=False):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.008',facecolor='#F5F9FC',edgecolor=color,linewidth=1.5,linestyle='--' if dashed else '-'))
 ax.text(x+w/2,y+h/2,tx(text),ha='center',va='center',color=INK,fontsize=size,linespacing=1.5)
def arrow(ax,a,b,color=BLUE,dashed=False):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,linewidth=1.7,color=color,linestyle='--' if dashed else '-'))
def canvas(title):
 f,a=plt.subplots(figsize=(14,6.5));f.subplots_adjust(left=.025,right=.975,bottom=.04,top=.85);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');f.suptitle(title,fontsize=19,color=INK,y=.96);return f,a
captions={
1:{'zh':'左侧为固定目标和输入点的 Total Endpoint。中间保留原生 use 身份与输入坐标，并将 attention 分支内部的微分分组单独定义。右侧分别比较空间分布与规定有限替换效应。全局闭合须包含 residual bypass；分支分组只重构 attention branch contribution。',
'en':'Left: Total Endpoint for one fixed target and input point. Middle: retain native-use identity and input coordinates, with a separate definition for differential partitions inside attention. Right: compare spatial distributions and prescribed finite replacement effects. Global closure includes the residual bypass; a branch partition reconstructs only the attention branch contribution.'},
2:{'zh':'左栏区分共享 producer 的两个 consumer slots。中栏的蓝色下路是 residual identity bypass，上路为 attention branch。绿色虚线框只包含 branch partition，闭合到 attention branch contribution A_F。红色虚线表示全局切面，包含 identity 及完整 branch partition，闭合到 Total Endpoint L_F。右栏列出两种不同的重构目标。',
'en':'Left: two consumer slots of a shared producer remain distinct. Middle: the lower blue path is the residual identity bypass; the upper path is attention. The green dashed region contains a branch partition and reconstructs attention branch contribution A_F. The red dashed global cut includes identity and the complete branch partition and reconstructs Total Endpoint L_F. Right: the two reconstruction targets.'}}
def save(f,number,title,manifest):
 row=next(x for x in manifest['figures'] if x['number']==number);row['title']=title
 for ext,rec in row['files'].items():
  p=P/rec['path'];f.savefig(p,dpi=250,facecolor='white');rec['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 plt.close(f)
 main={1:1,2:2,6:3}[number]
 for lang in ['zh','en']:
  p=P/lang/'manuscript.md';s=p.read_text()
  if number in captions:
   caption=captions[number][lang];s=re.sub('(?m)^'+re.escape('**Figure '+str(main)+'.** ')+'.*$',lambda m:'**Figure '+str(main)+'.** '+caption,s);p.write_text(s)
  else:caption=re.search('(?m)^'+re.escape('**Figure '+str(main)+'.** ')+'(.*)$',s).group(1)
  row['caption_'+lang]=caption

def main():
 manifest=json.loads((P/'figure_manifest.json').read_text())
 f,a=canvas('Local differential maps: representation, reconstruction, and finite replacement')
 box(a,.02,.43,.22,.31,'Total Endpoint'+chr(10)+'$E_s=x§odot§nabla_xs$'+chr(10)+'fixed target s and input x',BLUE)
 box(a,.31,.53,.30,.30,'Edge-conditioned local maps'+chr(10)+'native use e × input coordinate i',TEAL)
 box(a,.31,.14,.30,.23,'Attention branch partitions'+chr(10)+'Q + K + V = routing + value',TEAL,dashed=True)
 box(a,.70,.59,.27,.24,'Spatial distributions'+chr(10)+'on fixed sampled targets',BLUE)
 box(a,.70,.15,.27,.28,'Finite replacement effect'+chr(10)+'forward A, V, head, branch'+chr(10)+'specified reference',ORANGE)
 arrow(a,(.24,.61),(.30,.68));arrow(a,(.61,.68),(.69,.71));arrow(a,(.61,.27),(.69,.28),ORANGE)
 a.text(.16,.22,'Global reconstruction'+chr(10)+'identity + branch = total',ha='center',color=RED,fontsize=12)
 arrow(a,(.46,.51),(.46,.39),TEAL)
 save(f,1,'Local differential maps and prescribed replacement',manifest)
 f,a=canvas('Branch partitions and global cuts have different reconstruction targets')
 box(a,.015,.12,.21,.72,'Shared producer z'+chr(10)*2+'consumer A · slot 0'+chr(10)+'consumer B · slot 1'+chr(10)*2+'retain each invocation',BLUE,size=11,dashed=True)
 box(a,.28,.40,.08,.13,'u',BLUE)
 box(a,.41,.51,.26,.28,'attention branch'+chr(10)+'Q / K / V; all heads',TEAL,dashed=True)
 box(a,.71,.40,.055,.13,'+',ORANGE);box(a,.83,.40,.13,.13,'target s',BLUE)
 arrow(a,(.36,.51),(.405,.66),TEAL);arrow(a,(.675,.65),(.73,.54),TEAL)
 a.plot([.32,.32,.73,.73],[.39,.20,.20,.39],color=BLUE,lw=2);arrow(a,(.73,.26),(.73,.39),BLUE)
 a.text(.50,.25,'identity bypass',color=BLUE,ha='center');arrow(a,(.77,.465),(.825,.465))
 a.plot([.69,.69],[.13,.87],color=RED,linestyle='--',lw=2)
 a.text(.66,.925,'global cut = identity + complete branch partition',color=RED,ha='center',fontsize=12)
 a.text(.54,.46,'branch only → A_F',color=TEAL,ha='center',fontsize=11)
 a.text(.82,.085,'A_F = attention contribution'+chr(10)+'L_F = Total Endpoint',ha='center',fontsize=11,color=INK)
 save(f,2,'Residual bypass distinguishes branch and global reconstruction',manifest)
 f,a=canvas('Attention differentials reconstruct the attention branch contribution')
 a.text(.01,.87,'Normal computation',fontsize=13,color=BLUE)
 for x,label in [(.04,'Q, K'),(.29,'S = Q Kᵀ / √d'),(.56,'A = softmax(S)'),(.82,'O = A V')]:box(a,x,.66,.15,.13,label,BLUE,11)
 for x,y in [(.19,.29),(.44,.56),(.71,.82)]:arrow(a,(x,.725),(y,.725))
 a.text(.01,.50,'Local differential propagation',fontsize=13,color=TEAL)
 box(a,.04,.24,.20,.17,'dS = dQ Kᵀ / √d'+chr(10)+'     + Q dKᵀ / √d',TEAL,11)
 box(a,.32,.24,.25,.17,'da = [diag(a) − a aᵀ] ds'+chr(10)+'row-wise softmax Jacobian',TEAL,11)
 box(a,.66,.24,.30,.17,'dO = (dA) V + A dV'+chr(10)+'routing-induced + value-induced',ORANGE,11)
 arrow(a,(.245,.325),(.315,.325),TEAL);arrow(a,(.575,.325),(.655,.325),TEAL)
 a.text(.50,.07,'Head concat / output projection / downstream cotangent → attention branch contribution; add bypass for Total Endpoint',ha='center',fontsize=11,color=RED)
 save(f,6,'Attention branch differential partitions',manifest)
 # Figure 4 caption was corrected without modifying its empirical map panels.
 row=next(x for x in manifest['figures'] if x['number']==7)
 for lang in ['zh','en']:row['caption_'+lang]=re.search('(?m)^'+re.escape('**Figure 4.** ')+'(.*)$',(P/lang/'manuscript.md').read_text()).group(1)
 manifest['revision']='PUBLICATION-R2-REVISION-V6'
 (P/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+chr(10))
 binding=json.loads((P.parent/'figure_manifest.json').read_text());binding['source_revision']=manifest['revision']
 for row in manifest['figures']:
  target=next(x for x in binding['figures'] if x['figure']==row['number'])
  for ext,rec in row['files'].items():target[ext]['sha256']=rec['sha256']
 (P.parent/'figure_manifest.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2)+chr(10))
 for lang in ['zh','en']:
  s=(P/lang/'manuscript.md').read_text();s=s.replace('](../figures/','](../publication_r2_20260920/figures/').replace('](../references.md)','](../publication_r2_20260920/references.md)').replace('](supplement.md)','](../publication_r2_20260920/'+lang+'/supplement.md)').replace('](supplement_experiments_v1.md)','](../publication_r2_20260920/'+lang+'/supplement_experiments_v1.md)')
  (P.parent/lang/'full_draft.md').write_text(s)
 print('Rendered concept package figures 1, 2, 6; synchronized manifests and captions')
if __name__=='__main__':main()
