#!/usr/bin/env python3
"""R4: separate ViT representation, token attribution and subpatch expansion."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig-nns-unified-paper-r4'
import argparse
from functools import lru_cache
import html
import json
from pathlib import Path
import shutil
import sys
import render_nns_vit_unified_paper_figures_r1 as base
from nns_vit_unified_r4_captions import TITLES, CAPTIONS_ZH, CAPTIONS_EN, ANALYSIS
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

REPO = base.REPO
ROOT = REPO / 'reports/nns_vit_unified_paper_figures_r4_20260916'
R2 = REPO / 'reports/nns_vit_unified_paper_figures_r2_20260916'
R3 = REPO / 'reports/nns_vit_unified_paper_figures_r3_20260916'
CONFIG = REPO / 'config/nns_vit_unified_paper_figures_r4.json'
S = base.SOURCES
BLUE, ORANGE = '#1178B4', '#D97904'
INK, MUTED, LIGHT = '#172033', '#637083', '#E8EEF4'
NL = chr(10)
LAYOUT = []
NNS_A = np.asarray([[-.4,.2,.8,-.1],[.3,1.2,2.4,.5],[-.2,.9,1.7,.4],[-.3,.1,.6,-.2]])
NNS_B = np.asarray([[.2,-.1,.4,.1],[.8,1.4,.6,-.2],[.3,.7,-.3,.2],[.1,-.4,.2,.5]])
LOCAL_RANK = np.asarray([[14,10,6,13],[9,3,1,7],[15,4,2,8],[16,11,5,12]])


def text(ax,x,y,value,size=9,color=INK,weight='normal',ha='left',**kwargs):
    return ax.text(x,y,value,fontsize=size,color=color,weight=weight,ha=ha,va='center',**kwargs)


def math(value):
    return value.replace('@',chr(92))


def canvas(height):
    fig=plt.figure(figsize=(12,height))
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,12),ylim=(0,height)); ax.axis('off')
    return fig,ax


def arrow(ax,start,end,color=MUTED):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,linewidth=1.6,color=color))


def panel(ax,x,y,label,title,color=BLUE):
    text(ax,x,y,label,11,color=color,weight='bold')
    text(ax,x+.29,y,title,11,weight='bold')


def grid(ax,values,x,y,width,cmap='Blues',vmin=0,vmax=1,labels=None,edge=INK):
    values=np.asarray(values); nr,nc=values.shape; height=width*nr/nc
    ax.imshow(values,extent=(x,x+width,y,y+height),origin='upper',cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest',aspect='auto',zorder=2)
    for k in range(nr+1): ax.plot([x,x+width],[y+k*height/nr]*2,color='white',lw=.6,zorder=3)
    for k in range(nc+1): ax.plot([x+k*width/nc]*2,[y,y+height],color='white',lw=.6,zorder=3)
    ax.add_patch(Rectangle((x,y),width,height,fill=False,ec=edge,lw=.8,zorder=4))
    if labels is not None:
        for r in range(nr):
            for c in range(nc):
                color='white' if (values[r,c]-vmin)/(vmax-vmin)>.68 else INK
                text(ax,x+(c+.5)*width/nc,y+height-(r+.5)*height/nr,str(labels[r][c]),8,color=color,ha='center',zorder=5)


@lru_cache(maxsize=1)
def original():
    manifest=S.json(base.F3C3/'formal_manifest.json')
    record=next(r for r in manifest['samples'] if int(r['formal_position'])==1)
    path=REPO/'assets/data/ILSVRC2012_val'/record['relative_path']
    base.require_sha(path,record['image_sha256'],'Frozen R4 schematic original')
    return base.crop_224(path)


def photo(ax,arr,x,y,width,patch_grid=False):
    ax.imshow(arr,extent=(x,x+width,y,y+width),aspect='auto',interpolation='nearest',zorder=2)
    if patch_grid:
        for k in range(15):
            ax.plot([x+k*width/14]*2,[y,y+width],color='white',lw=.4,alpha=.65,zorder=3)
            ax.plot([x,x+width],[y+k*width/14]*2,color='white',lw=.4,alpha=.65,zorder=3)


def figure_01():
    fig,ax=canvas(7.5)
    text(ax,.25,7.19,'One trained ViT, two attribution readouts',17,weight='bold')
    text(ax,.25,6.82,'Different patches can have different scores; uniform expansion acts inside each patch.',10,color=MUTED)
    panel(ax,.25,6.35,'a','The model represents structure within and between patches')
    photo(ax,original(),.35,4.55,1.38,patch_grid=True)
    text(ax,1.04,4.29,'224 × 224 image',8.5,ha='center')
    arrow(ax,(1.87,5.25),(2.18,5.25))
    grid(ax,np.arange(16).reshape(4,4)%7,2.31,5.00,.77,vmax=8)
    text(ax,2.69,4.68,'16 × 16 × 3 patch',8,ha='center')
    arrow(ax,(3.18,5.39),(3.58,5.39))
    grid(ax,[[.1,.7,.3,.9,.4,.2,.8,.5]],3.74,5.31,1.18)
    text(ax,4.33,5.84,'Patch embedding',9,weight='bold',ha='center')
    text(ax,4.33,4.94,'A feature vector per token',8.4,ha='center')
    text(ax,4.33,4.57,'Pixel positions can contribute differently.',8,color=MUTED,ha='center')
    arrow(ax,(5.06,5.39),(5.43,5.39))
    grid(ax,[[.2,.7,.1],[.5,.2,.3],[.1,.3,.6]],5.61,5.03,.72,vmax=.8)
    text(ax,6.90,5.79,'Self-attention',9,weight='bold')
    text(ax,6.90,5.39,'Weights vary across tokens,',8.4)
    text(ax,6.90,5.09,'inputs, blocks and heads.',8.4)
    arrow(ax,(8.93,5.39),(9.27,5.39))
    text(ax,10.46,5.74,'Select one internal target',9,weight='bold',ha='center')
    text(ax,10.46,5.22,math('$f(z)=u^@top g(z)$'),14,color=BLUE,ha='center')
    text(ax,10.46,4.70,'Same target for both methods',8,ha='center',color=MUTED)
    ax.plot([.25,11.75],[3.99]*2,color=LIGHT,lw=1.1)
    panel(ax,.25,3.66,'b','NNS: contributions in input coordinates')
    panel(ax,6.16,3.66,'c','Matched token baseline',ORANGE)
    text(ax,.55,3.24,'Within-patch detail is computed from input gradients.',8.7,color=MUTED)
    text(ax,6.46,3.24,'Patch baseline: one token score → uniform cells in its patch',8.7,color=MUTED)
    for xpos,values,label in [(.64,NNS_A,'Input patch A'),(2.41,NNS_B,'Input patch B')]:
        grid(ax,values,xpos,1.64,1.14,cmap='RdBu_r',vmin=-2.5,vmax=2.5,edge=BLUE)
        text(ax,xpos+.57,2.98,label,8.5,weight='bold',ha='center')
    text(ax,4.00,2.52,'16 fine cells',9,weight='bold')
    text(ax,4.00,2.19,'per patch',9)
    text(ax,4.00,1.76,'Each cell: 4 × 4 pixels',8,color=MUTED)
    text(ax,.64,1.24,'Different input locations may receive different signed scores.',8.5)
    ax.imshow(np.linspace(-2.5,2.5,100)[None,:],extent=(.64,1.78,.77,.87),cmap='RdBu_r',vmin=-2.5,vmax=2.5,aspect='auto',zorder=2)
    text(ax,1.97,.82,'negative ← 0 → positive',8,color=MUTED)
    for xpos,score,label in [(6.53,16,'A'),(8.28,8,'B')]:
        per_cell=score/16
        grid(ax,np.full((4,4),per_cell),xpos,1.64,1.14,cmap='Oranges',vmax=1.2,labels=np.full((4,4),f'{per_cell:g}'),edge=ORANGE)
        text(ax,xpos+.57,2.98,f'Token {label}: c{label} = {score}',8.5,weight='bold',ha='center')
        text(ax,xpos+.57,1.33,f'Patch {label}: {per_cell:g} per cell',8.4,ha='center')
    text(ax,9.79,2.47,'A ≠ B',12,color=ORANGE,weight='bold')
    text(ax,9.79,2.09,'Different between patches',8)
    text(ax,9.79,1.78,'Equal within each patch',8)
    text(ax,6.53,.83,'Uniformity comes from this readout rule.',8.7,color=ORANGE,weight='bold')
    text(ax,.25,.32,'Illustrative grids and scores. The two readouts are computed independently; patch-wise totals are not constrained to match.',8,color=MUTED)
    return fig


def figure_02():
    fig,ax=canvas(8.05)
    text(ax,.25,7.72,'One internal target, two contribution coordinates',17,weight='bold')
    text(ax,.25,7.35,'Choose a block/head and define the scalar before computing either explanation.',10,color=MUTED)
    ax.add_patch(Rectangle((.25,5.82),11.5,1.16,fc='#F3F6F9',ec='none',zorder=0))
    text(ax,.45,6.68,math('$g(z)=@sum_{j=0}^{196}A(x)_{0j}V_j(z),@qquad f(z)=u^@top g(z)$'),12)
    text(ax,7.85,6.68,math('$u=@frac{g(x)-g(x_0)}{@Vert g(x)-g(x_0)@Vert_2}$'),12)
    text(ax,.45,6.11,'A(x): clean CLS attention; A and u stay fixed for this explanation.',9,color=MUTED)
    text(ax,8.24,6.11,'Value features V(z) vary with the input.',8.5,color=MUTED)
    panel(ax,.25,5.41,'a','NNS · input-space integrated gradients')
    panel(ax,6.26,5.41,'b','Matched token baseline',ORANGE)
    arr=original()
    for xpos,alpha,label in [(.66,0,'Reference x₀'),(2.23,.5,'Path input xα'),(3.80,1,'Clean image x')]:
        photo(ax,np.rint(127.5+(arr.astype(float)-127.5)*alpha).astype(np.uint8),xpos,4.22,.72)
        text(ax,xpos+.36,4.02,label,8,ha='center')
    arrow(ax,(1.49,4.58),(2.13,4.58),BLUE); arrow(ax,(3.06,4.58),(3.70,4.58),BLUE)
    text(ax,2.88,3.63,math('$x_@alpha=x_0+@alpha(x-x_0)$'),12,ha='center')
    text(ax,2.88,3.11,math('$R_i=(x_i-x_{0i})@int_0^1@partial_i f(x_@alpha)@,d@alpha$'),12,color=BLUE,ha='center')
    text(ax,.55,2.53,'Sum RGB channels with sign, then sum into fine cells.',9)
    arrow(ax,(2.05,2.30),(2.05,1.96),BLUE)
    grid(ax,NNS_A,1.51,.83,1.04,cmap='RdBu_r',vmin=-2.5,vmax=2.5,edge=BLUE)
    text(ax,2.82,1.61,'Input-coordinate scores',9,color=BLUE,weight='bold')
    text(ax,2.82,1.26,'Within-patch values can differ.',8.5)
    text(ax,2.82,.91,'Computed from f, x and x₀.',8.5)
    text(ax,1.51,.63,'Blue: negative · red: positive',7.7,color=MUTED)
    ax.plot([5.96]*2,[.82,5.43],color=LIGHT,lw=1.1)
    text(ax,6.54,4.87,'Evaluate clean and reference Value features',9.5,weight='bold')
    text(ax,8.99,4.24,math('$c_j=A(x)_{0j}@,u^@top[V_j(x)-V_j(x_0)]$'),12,color=ORANGE,ha='center')
    text(ax,6.54,3.73,'Attention weight × target-directed Value change',9,color=MUTED)
    arrow(ax,(8.89,3.43),(8.89,3.14),ORANGE)
    text(ax,6.54,2.89,'196 patch-token scalars',10,weight='bold')
    text(ax,9.34,2.89,'CLS scalar kept separately',8.7,color=MUTED)
    text(ax,6.54,2.38,'Example token A',9)
    text(ax,7.32,1.80,math('$c_A=16$'),15,color=ORANGE,ha='center')
    arrow(ax,(8.18,1.77),(8.78,1.77),ORANGE)
    grid(ax,np.ones((4,4)),8.96,1.24,1.04,cmap='Oranges',vmax=1.2,labels=np.ones((4,4),dtype=int),edge=ORANGE)
    text(ax,10.22,1.94,'One patch',9,weight='bold')
    text(ax,10.22,1.56,'16 fine cells',8.5)
    text(ax,8.99,.89,math('$c_j/16$ per fine cell'),9,color=ORANGE,ha='center')
    text(ax,.25,.43,'Reference x₀: normalized zero / RGB 0.5. Diagrams and numbers are illustrative.',8,color=MUTED)
    text(ax,.25,.17,'Deep tokens mix spatial information: input-patch attribution sums need not equal the indexed token contributions.',8,color=MUTED)
    return fig


def figure_03():
    fig,ax=canvas(7.75)
    text(ax,.25,7.43,'Does the finer ranking predict the internal response?',17,weight='bold')
    text(ax,.25,7.06,'Compare the two explanations using the same target, image and intervention area.',10,color=MUTED)
    panel(ax,.25,6.56,'a','Choose the spatial unit')
    panel(ax,4.05,6.56,'b','Rank fine cells')
    panel(ax,7.66,6.56,'c','Restore or remove')
    grid(ax,(np.arange(196).reshape(14,14)%13)/13,.49,4.84,1.31)
    ax.add_patch(Rectangle((.49+5*1.31/14,4.84+6*1.31/14),1.31/14,1.31/14,fill=False,ec=ORANGE,lw=1.8,zorder=6))
    arrow(ax,(1.88,5.47),(2.13,5.47),ORANGE)
    grid(ax,np.ones((4,4))*.4,2.23,5.03,.97,edge=ORANGE)
    text(ax,1.15,4.58,'14 × 14 patches',8.7,ha='center')
    text(ax,2.72,4.76,'ONE 16 × 16-pixel patch',8,weight='bold',ha='center')
    text(ax,2.72,4.39,'16 cells of 4 × 4 pixels',8.2,ha='center')
    text(ax,.49,3.77,'Native scale',9,weight='bold')
    text(ax,.49,3.41,'196 units · 16 × 16 pixels per unit',8.8)
    text(ax,.49,2.85,'Fine scale',9,weight='bold',color=BLUE)
    text(ax,.49,2.49,'3,136 units · 4 × 4 pixels per unit',8.8)
    text(ax,.49,1.98,'The same unit size is used for both methods.',8,color=MUTED)
    grid(ax,(17-LOCAL_RANK)/16,4.25,4.84,1.10,labels=LOCAL_RANK,edge=BLUE)
    text(ax,5.57,5.64,'NNS',10,color=BLUE,weight='bold')
    text(ax,5.57,5.27,'Can rank cells',8.6)
    text(ax,5.57,4.98,'within a patch',8.6)
    text(ax,4.25,4.48,'Illustrative local order: 1 = first',8,color=MUTED)
    grid(ax,np.ones((4,4)),4.25,2.95,1.10,cmap='Oranges',vmax=1.2,labels=np.ones((4,4),dtype=int),edge=ORANGE)
    text(ax,5.57,3.78,'Token baseline',9,color=ORANGE,weight='bold')
    text(ax,5.57,3.43,'Same score within',8.5)
    text(ax,5.57,3.13,'this patch',8.5)
    text(ax,4.25,2.55,'Example: cA = 16 → score 1 per cell',8,color=MUTED)
    text(ax,4.25,2.10,'Actual ranking uses all 3,136 cells.',8.4,weight='bold')
    text(ax,4.25,1.78,'Ties: 5 paired permutations for both methods.',8,color=MUTED)
    arr=original(); order=np.argsort(LOCAL_RANK.ravel(),kind='stable')
    for row,y,color in [('Insertion',5.05,BLUE),('Deletion',3.55,ORANGE)]:
        text(ax,7.88,y+.92,row,10,color=color,weight='bold')
        for col,k in enumerate((0,8,16)):
            mask=np.zeros(16,dtype=bool); mask[order[:k]]=True
            mask=np.repeat(np.repeat(mask.reshape(4,4),56,axis=0),56,axis=1)
            if row=='Deletion': mask=~mask
            shown=np.where(mask[:,:,None],arr,128).astype(np.uint8); xpos=7.88+col*1.13
            photo(ax,shown,xpos,y,.77); text(ax,xpos+.385,y-.20,('0','½','1')[col],8,ha='center')
            if col<2: arrow(ax,(xpos+.82,y+.39),(xpos+1.09,y+.39),color)
    text(ax,7.88,2.91,'Better ranking: higher insertion AUC',8.3)
    text(ax,7.88,2.58,'and lower deletion AUC.',8.3)
    text(ax,7.88,2.14,'21 fractions · evaluate the same internal f',8.3,weight='bold')
    text(ax,7.88,1.79,'Masks are enlarged schematics.',8,color=MUTED)
    ax.plot([.25,11.75],[1.42]*2,color=LIGHT,lw=1.1)
    text(ax,.49,1.12,'Average 5 repeats within each image',9,weight='bold')
    arrow(ax,(4.02,1.12),(4.49,1.12))
    text(ax,4.68,1.12,'ΔFIDᵢ = Fᵢ(NNS) − Fᵢ(token)',11,color=BLUE)
    arrow(ax,(8.03,1.12),(8.49,1.12))
    text(ax,8.65,1.12,'Image median + 95% CI',10,weight='bold')
    text(ax,.49,.66,'F = ½(AUC insertion − AUC deletion)',10)
    text(ax,5.14,.66,'Paired bootstrap uses images as independent units.',8.6,color=MUTED)
    text(ax,.25,.22,'Illustrative maps, ranks and masks. Gray reference: normalized zero / RGB 0.5. Actual intervention fraction is used for AUC.',8,color=MUTED)
    return fig


BUILDERS={1:figure_01,2:figure_02,3:figure_03}


def save(number,slug):
    fig=BUILDERS[number](); fig.canvas.draw(); renderer=fig.canvas.get_renderer(); bounds=fig.bbox; outside=[]
    for ax in fig.axes:
        for artist in ax.texts:
            box=artist.get_window_extent(renderer)
            if box.x0<bounds.x0-1 or box.y0<bounds.y0-1 or box.x1>bounds.x1+1 or box.y1>bounds.y1+1: outside.append(artist.get_text())
    LAYOUT.append({'number':number,'text_outside_canvas':outside,'text_count':sum(len(ax.texts) for ax in fig.axes)})
    if outside: raise base.RenderError('Text outside canvas: '+repr(outside))
    return base.save_figure(fig,number,slug)


def snapshot(root):
    return [{'path':p.relative_to(REPO).as_posix(),'bytes':p.stat().st_size,'sha256':base.sha256_file(p)} for p in sorted(root.rglob('*')) if p.is_file()]


def sections(path,marker):
    return {int(p.split(':',1)[0].split(NL,1)[0]):p.split(NL,1)[1].strip() for p in path.read_text(encoding='utf-8').split(marker)[1:]}


def write_docs(records):
    for lang,overrides in [('zh',CAPTIONS_ZH),('en',CAPTIONS_EN)]:
        prior=sections(R2/f'captions_{lang}.md','## Figure '); S.add(R2/f'captions_{lang}.md','inherited_caption_source')
        prior[7]=(R3/f'captions_{lang}.md').read_text().strip(); S.add(R3/f'captions_{lang}.md','R3_figure_7_caption')
        prior.update(overrides); lines=[f'# Unified paper figures R4 · {lang}','']
        for n in range(1,15): lines.extend([f'## Figure {n}','',prior[n],''])
        (ROOT/f'captions_{lang}.md').write_text(NL.join(lines),encoding='utf-8')
    prior=sections(R2/'figure_analysis.md','## Image #'); S.add(R2/'figure_analysis.md','inherited_analysis')
    prior[7]=(R3/'figure_analysis.md').read_text().split(NL,1)[1].strip(); S.add(R3/'figure_analysis.md','R3_figure_7_analysis')
    prior.update(ANALYSIS); lines=['# 统一论文图 R4：逐图分析','','Figure 1–3 重画；Figure 7 沿用 R3；其余图沿用 R2。','']
    for r in records: lines.extend([f'## Image #{r["number"]}: '+r['title'],'',prior[r['number']],''])
    (ROOT/'figure_analysis.md').write_text(NL.join(lines),encoding='utf-8')
    lines=['# Unified paper figures R4','','[浏览完整画廊](gallery.html) · [中文图注](captions_zh.md) · [逐图解读](figure_analysis.md)','','| Figure | Title | Revision | PNG | PDF | SVG |','|---:|---|---|---|---|---|']; cards=[]
    for r in records:
        n=r['number']; links={o['format']:Path(o['path']).parent.name+'/'+Path(o['path']).name for o in r['outputs']}
        revision='R4 重画' if n<4 else '继承 R3' if n==7 else '继承 R2'
        lines.append('| '+str(n)+' | '+r['title']+' | '+revision+' | '+' | '.join('['+fmt+']('+links[fmt]+')' for fmt in ['PNG','PDF','SVG'])+' |')
        explanation=CAPTIONS_ZH[n] if n in CAPTIONS_ZH else prior[n].split(NL)[0].removeprefix('- 图能说明什么：')
        before='<details><summary>查看上一版</summary><img loading="lazy" src="../'+R2.name+'/'+links['PNG']+'" alt="Previous figure '+str(n)+'"></details>' if n<4 else ''
        downloads=' · '.join('<a href="'+links[fmt]+'">'+fmt+'</a>' for fmt in ['PNG','PDF','SVG'])
        cards.append('<article id="figure-'+str(n)+'"><div class="caption"><span>'+revision+'</span><h2>Figure '+str(n)+' · '+html.escape(r['title'])+'</h2><p>'+html.escape(explanation)+'</p><div class="downloads">'+downloads+'</div></div><a href="'+links['PNG']+'"><img loading="lazy" src="'+links['PNG']+'" alt="Figure '+str(n)+'"></a>'+before+'</article>')
    (ROOT/'figure_index.md').write_text(NL.join(lines)+NL,encoding='utf-8')
    gallery='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NNS ViT · R4 Figure 1–14</title><style>
body{margin:0;background:#edf2f6;color:#172033;font:16px/1.65 system-ui,sans-serif}header{padding:24px 4vw;background:#172033;color:white}header h1{margin:0;font-size:26px}header p{margin:8px 0;color:#d0deeb}nav{display:flex;gap:16px;flex-wrap:wrap}nav a{color:#b9def7}main{max-width:1500px;margin:auto;padding:24px}article{background:white;padding:26px;margin-bottom:28px}h2{font-size:21px;margin:5px 0 10px}.caption p{max-width:1050px;color:#526071}.caption span{font-size:13px;font-weight:700;color:#1178B4}.downloads{margin:12px 0}a{color:#1178B4}img{width:100%;height:auto}details{padding:15px 0;border-top:1px solid #d9e1ea}summary{cursor:pointer;color:#637083}@media(max-width:750px){main{padding:10px}article{padding:14px}}</style><header><h1>NNS ViT · R4 · Figure 1–14</h1><p>Figure 1–3：模型表征、token 归因与 patch 内展开。点击图片查看完整分辨率。</p><nav>'''
    gallery+=' '.join('<a href="#figure-'+str(n)+'">Figure '+str(n)+'</a>' for n in range(1,15))
    gallery+='</nav></header><main>'+NL.join(cards)+'</main></html>'
    (ROOT/'gallery.html').write_text(gallery,encoding='utf-8')
    (ROOT/'README.md').write_text('''# NNS ViT unified paper figures R4

完整 Figure 1–14，全部图在当前目录提供 PNG（400 dpi）、PDF、SVG 实体文件，共 42 个导出。

- [浏览画廊](gallery.html)
- [完整图表索引](figure_index.md)
- [中文图注](captions_zh.md) / [English captions](captions_en.md)
- [逐图分析](figure_analysis.md)

Figure 1–3 重新绘制，区分模型表征与归因读出，明确“不同 patch 可以不同，各 patch 内均匀展开”，并分别画出 NNS 和 token 基线的计算路径。数字网格和 mask 都是方法示意。

Figure 7 逐字节继承 R3 的分组柱状图与局部放大，其余 Figure 4–14 逐字节继承 R2。中英文图注、逐图说明均按此继承关系组装；既有实证数值保持原值。

渲染器：scripts/render_nns_vit_unified_paper_figures_r4.py。验证器：scripts/validate_nns_vit_unified_paper_figures_r4.py。仅使用 CPU；封存后的图包不可原地重画。
''',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--draft',action='store_true'); args=parser.parse_args()
    if ROOT.exists() and ((ROOT/'terminal_receipt.json').exists() or not args.draft): raise base.RenderError('Only an unsealed R4 draft may be updated')
    config=S.json(CONFIG); base.verify_signed_json(config,'R4 config'); base.configure(config); base.OUTPUT=ROOT
    plt.rcParams['svg.hashsalt']='nns-vit-unified-r4'; ROOT.mkdir(parents=True,exist_ok=True)
    before=snapshot(R2)+snapshot(R3)
    for path,role in [(Path(__file__),'renderer_source'),(Path(base.__file__),'export_helper'),(Path(__file__).with_name('nns_vit_unified_r4_captions.py'),'caption_source'),(Path(__file__).with_name('validate_nns_vit_unified_paper_figures_r4.py'),'validator_source'),(REPO/config['design_revision'],'design_revision'),(REPO/'docs/阶段C_protocol.md','method_definition'),(REPO/'src/vitnns/stage_c_value_path.py','method_implementation_read_only')]: S.add(path,role)
    old=S.json(R2/'render_manifest.json'); base.verify_signed_json(old,'R2 manifest')
    newer=S.json(R3/'render_manifest.json'); base.verify_signed_json(newer,'R3 manifest')
    for root in (R2,R3):
        for name in ('terminal_receipt.json','validation_report.json','manual_visual_qc.json'):
            value=S.json(root/name); base.verify_signed_json(value,'predecessor '+name)
            if not value['status'].startswith('PASS'): raise base.RenderError('Unvalidated predecessor')
    records=[]
    for spec in old['figures']:
        n=spec['number']; source_revision='R4' if n<4 else 'R3' if n==7 else 'R2'
        if n<4: outputs=save(n,spec['slug'])
        else:
            outputs=[]
            for out in newer['outputs'] if n==7 else spec['outputs']:
                source=REPO/out['path']; base.require_sha(source,out['sha256'],'Inherited '+source_revision+' figure'); S.add(source,'inherited_figure')
                dest=ROOT/source.parent.name/source.name; dest.parent.mkdir(exist_ok=True,parents=True); shutil.copyfile(source,dest)
                outputs.append({**out,'path':dest.relative_to(REPO).as_posix()})
        records.append({'number':n,'slug':spec['slug'],'title':TITLES.get(n,spec['title']),'role':spec['role'],'revision':source_revision,'redesigned':n<4,'outputs':outputs})
        print(f'Figure {n:02d}: {source_revision}',flush=True)
    write_docs(records)
    numeric=S.json(R2/'numeric_display_data.json'); base.verify_signed_json(numeric,'R2 numeric')
    numeric7=S.json(R3/'numeric_display_data.json'); base.verify_signed_json(numeric7,'R3 numeric')
    if numeric['figures']['7']!=numeric7['series']: raise base.RenderError('R2/R3 Figure 7 numeric mismatch')
    base.write_json(ROOT/'numeric_display_data.json',base.signed({'schema_version':1,'revision':'R4','figures':numeric['figures'],'schematics':{'status':'ILLUSTRATIVE_ONLY','token_scores':[16,8],'uniform_cells_per_patch':16,'uniform_cell_scores':[1,.5],'nns_patch_A':NNS_A.tolist(),'nns_patch_B':NNS_B.tolist(),'nns_patch_sums_equal_token_scores_assumed':False,'local_rank_example':LOCAL_RANK.tolist()},'new_inference':False}))
    if before!=snapshot(R2)+snapshot(R3): raise base.RenderError('Predecessor root changed during rendering')
    base.write_json(ROOT/'predecessor_snapshot.json',base.signed({'files':before,'files_digest':base.canonical_digest(before),'unchanged_after_render':True}))
    base.write_json(ROOT/'layout_audit.json',base.signed({'status':'PASS','records':LAYOUT,'scope':'text_canvas_bounds; visual overlap checked separately'}))
    sources=S.inventory()
    base.write_json(ROOT/'input_inventory.json',base.signed({'source_files':sources,'source_files_digest':base.canonical_digest(sources)}))
    base.write_json(ROOT/'render_manifest.json',base.signed({'schema_version':1,'revision':'R4','status':'RENDERED_PENDING_VISUAL_REVIEW','producer_commit':base.git('rev-parse','HEAD'),'source_identity':'file hashes include working-tree renderer sources','figure_count':14,'artifact_count':42,'figures':records,'source_files':sources,'source_files_digest':base.canonical_digest(sources),'cpu_only':True,'new_scientific_inference':False,'redesigned_figures':[1,2,3],'inherited_R3_figures':[7],'inherited_R2_figures':[4,5,6,8,9,10,11,12,13,14],'prior_roots_modified':False}))
    base.write_json(ROOT/'cpu_access_audit.json',base.signed({'status':'PASS_CPU_ONLY_PRESENTATION','CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'torch_imported':'torch' in sys.modules,'timm_imported':'timm' in sys.modules,'new_inference':False,'jpeg_scope':'Frozen F3C3 formal position 1 for schematic display only'}))
    print(json.dumps({'status':'RENDERED_PENDING_VISUAL_REVIEW','root':str(ROOT),'figures':14,'artifacts':42}))
    return 0


if __name__=='__main__': raise SystemExit(main())
