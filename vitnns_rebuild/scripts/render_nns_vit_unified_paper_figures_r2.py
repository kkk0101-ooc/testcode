#!/usr/bin/env python3
"""Presentation-only R2: diagrams, empirical distributions and compact comparisons."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import json
import shutil
from pathlib import Path

import render_nns_vit_unified_paper_figures_r1 as base
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import numpy as np

REPO = base.REPO
ROOT = REPO / 'reports/nns_vit_unified_paper_figures_r2_20260916'
OLD = REPO / 'reports/nns_vit_unified_paper_figures_r1_20260916'
CONFIG = REPO / 'config/nns_vit_unified_paper_figures_r2.json'
BLUE, ORANGE, GREEN, PURPLE = '#1178B4', '#D97904', '#16966F', '#9B63A8'
INK, MUTED, LIGHT = '#172033', '#637083', '#E8EEF4'
S = base.SOURCES
NUMERIC = {}
NL = chr(10)


def text(ax, x, y, value, size=9, color=INK, weight='normal', ha='left', **kw):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight, ha=ha, va='center', **kw)


def mathtext(value):
    return value.replace('@', chr(92))


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height)); ax.axis('off')
    return fig, ax


def connector(ax, start, end, color=MUTED, connectionstyle='arc3,rad=0', lw=1.7):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=13,
                                linewidth=lw, color=color, connectionstyle=connectionstyle))


def tile(ax, values, x, y, width, cmap='Blues', vmin=None, vmax=None, grid=True):
    values = np.asarray(values)
    height = width * values.shape[0] / values.shape[1]
    ax.imshow(values, extent=(x, x + width, y, y + height), origin='upper',
              cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', aspect='auto', zorder=2)
    if grid:
        for k in range(values.shape[0]+1):
            yy = y + height*k/values.shape[0]
            ax.plot([x, x+width], [yy, yy], color='white', lw=.65, zorder=3)
        for k in range(values.shape[1]+1):
            xx = x + width*k/values.shape[1]
            ax.plot([xx, xx], [y, y+height], color='white', lw=.65, zorder=3)
    return height


def frozen_original():
    rows = S.json(base.F3C3 / 'formal_manifest.json')['samples']
    row = next(row for row in rows if int(row['formal_position']) == 1)
    path = REPO / 'assets/data/ILSVRC2012_val' / row['relative_path']
    base.require_sha(path, row['image_sha256'], 'R2 frozen schematic original')
    return base.crop_224(path)


def photo(ax, arr, x, y, width, n_grid=0):
    ax.imshow(arr, extent=(x, x+width, y, y+width), aspect='auto', interpolation='nearest', zorder=2)
    if n_grid:
        for k in range(n_grid+1):
            ax.plot([x+k*width/n_grid]*2, [y,y+width], color='white', alpha=.65, lw=.4, zorder=3)
            ax.plot([x,x+width], [y+k*width/n_grid]*2, color='white', alpha=.65, lw=.4, zorder=3)


def heading(ax, x, y, number, title):
    text(ax, x, y, number, size=10, color=BLUE, weight='bold')
    text(ax, x+.33, y, title, size=10, weight='bold')


TOY = np.asarray([[.2,.7,1.0,.3], [.4,1.5,3.0,.8], [.1,.7,2.2,.3], [.2,.3,.6,.4]])
SIGNED_TOY = np.asarray([[-.3,.7,1.,-.4], [.4,1.5,3.,.8], [-.5,.7,2.2,.3], [-.2,-.3,.6,-.4]])


def figure_01(config):
    fig, ax = canvas(10.4, 4.15)
    text(ax,.15,3.94,'What does a patch-level explanation leave inside a patch?',14,weight='bold')
    for x,num,title in [( .15,'01','Image'),(2.12,'02','Internal pathway'),(4.8,'03','Spatial readout'),(7.6,'04','Intervention')]:
        heading(ax,x,3.43,num,title)
    arr=frozen_original(); photo(ax,arr,.18,1.40,1.55,n_grid=14)
    ax.add_patch(Rectangle((.18+6*1.55/14,1.40+6*1.55/14),1.55/14,1.55/14,fill=False,ec=ORANGE,lw=1.8,zorder=5))
    text(ax,.96,1.13,'224 × 224 image',8,ha='center')
    text(ax,.96,.85,'14 × 14 patch tokens',8,color=MUTED,ha='center')
    connector(ax,(1.80,2.20),(2.15,2.20))
    for k in range(12):
        ax.add_patch(Rectangle((2.30+k*.125,1.68),.085,1.20,fc=BLUE if k==0 else LIGHT,ec='none'))
    text(ax,3.05,3.02,'ViT-B/16',10,weight='bold',ha='center')
    text(ax,3.05,1.45,'12 blocks',8,ha='center',color=MUTED)
    for k in range(12):
        ax.add_patch(Circle((2.36+(k%6)*.26,.99-(k//6)*.22),.063,fc=BLUE if k==9 else LIGHT,ec='none'))
    text(ax,3.07,.53,'Select a fixed block / head',8,ha='center')
    connector(ax,(3.93,2.20),(4.55,2.20))
    tile(ax,TOY,4.80,2.12,.90,vmin=0,vmax=3)
    tile(ax,np.full((4,4),TOY.mean()),4.80,.77,.90,cmap='Oranges',vmin=0,vmax=3)
    text(ax,5.91,2.71,'NNS',11,color=BLUE,weight='bold')
    text(ax,5.91,2.39,'Within-patch detail',8)
    text(ax,5.91,1.34,'Patch baseline',10,color=ORANGE,weight='bold')
    text(ax,5.91,1.02,'Uniform allocation',8)
    text(ax,5.96,.50,'Same target and intervention budget',7.8,ha='center',color=MUTED)
    connector(ax,(7.10,2.58),(7.62,2.23),BLUE)
    connector(ax,(7.10,1.25),(7.62,1.94),ORANGE)
    for j,frac in enumerate((0,.5,1)):
        masked=arr.copy(); masked[:,:int(224*(1-frac))]=128
        photo(ax,masked,7.80+j*.79,2.02,.65)
    text(ax,8.94,2.91,'Restore / remove ranked regions',8.5,ha='center')
    connector(ax,(7.87,1.85),(9.99,1.85))
    text(ax,8.96,1.52,'Paired image-level fidelity',10,ha='center',weight='bold')
    text(ax,8.96,1.12,'ΔFID = F(NNS) − F(patch)',10,ha='center',color=BLUE)
    text(ax,8.96,.55,'Scale · heads · depth · branches',8,ha='center',color=MUTED)
    return fig


def figure_02(config):
    fig, ax=canvas(10.4,4.9)
    text(ax,.15,4.66,'Value-path attribution: one fixed target, two spatial readouts',14,weight='bold')
    heading(ax,.15,4.13,'a','Fix the internal target')
    heading(ax,3.62,4.13,'b','Integrate along the input path')
    heading(ax,7.28,4.13,'c','Resolve spatial contributions')
    attention=np.asarray([[.1,.3,.8,.2,.5]])
    tile(ax,attention,.29,3.45,1.58,vmin=0,vmax=1)
    text(ax,.28,3.20,'Clean CLS attention  A(x)[0, j]',8)
    connector(ax,(2.12,3.43),(2.80,3.78),BLUE,lw=2.2)
    text(ax,2.48,3.15,'Direction u',8,ha='center')
    text(ax,.29,2.73,mathtext('$g(z)=@sum_{j=0}^{196}A(x)_{0j}V(z)_j$'),11)
    text(ax,.29,2.13,mathtext('$u=@frac{g(x)-g(x_0)}{@Vert g(x)-g(x_0)@Vert_2}$'),12)
    text(ax,.29,1.51,mathtext('$f(z)=u^@top g(z)$'),14,color=BLUE)
    text(ax,.29,.93,'A and u stay fixed along the path.',8,color=MUTED)
    text(ax,.29,.55,'Value features vary with the input.',8,color=MUTED)
    arr=frozen_original()
    for xx,alpha in [(3.77,0),(4.83,.5),(5.89,1)]:
        photo(ax,(128+(arr.astype(float)-128)*alpha).astype(np.uint8),xx,3.05,.66)
    connector(ax,(4.48,3.38),(4.78,3.38)); connector(ax,(5.54,3.38),(5.84,3.38))
    for xx,label in [(4.10,'$x_0$'),(5.16,mathtext('$x_@alpha$')),(6.22,'$x$')]: text(ax,xx,2.82,label,10,ha='center')
    text(ax,5.21,2.45,mathtext('$x_@alpha=x_0+@alpha(x-x_0)$'),11,ha='center')
    text(ax,5.21,1.77,mathtext('$R_i=(x_i-x_{0i})@int_0^1@partial_i f(x_@alpha)@,d@alpha$'),11,ha='center',color=BLUE)
    text(ax,5.19,.93,'Signed input-space IG',9,ha='center',weight='bold')
    text(ax,5.19,.55,'RGB channels are summed with sign.',8,ha='center',color=MUTED)
    tile(ax,SIGNED_TOY,7.42,2.96,.88,cmap='RdBu_r',vmin=-3,vmax=3)
    text(ax,8.52,3.63,'NNS',11,color=BLUE,weight='bold')
    text(ax,8.52,3.29,'Input-space contributions',8)
    tile(ax,np.full((4,4),TOY.mean()),7.42,1.72,.88,cmap='Oranges',vmin=0,vmax=3)
    text(ax,8.52,2.37,'Patch baseline',10,color=ORANGE,weight='bold')
    text(ax,8.52,2.01,r'$c_j/16$ per fine cell',9)
    text(ax,7.42,1.19,mathtext('$c_j=A(x)_{0j}u^@top[V(x)_j-V(x_0)_j]$'),9.5)
    text(ax,7.42,.78,'196 patch contributions + separate CLS scalar',8,color=MUTED)
    # Explicit sign key for the illustrative signed readout.
    tile(ax,np.linspace(-3,3,80)[None,:],7.43,.37,.84,cmap='RdBu_r',vmin=-3,vmax=3,grid=False)
    text(ax,8.42,.40,'negative  ←  0  →  positive',7.5,color=MUTED)
    return fig


def figure_03(config):
    fig,ax=canvas(10.4,4.5)
    text(ax,.15,4.27,'From spatial ranking to a paired image-level effect',14,weight='bold')
    heading(ax,.15,3.77,'a','Choose the intervention unit')
    heading(ax,3.49,3.77,'b','Restore and remove in rank order')
    heading(ax,7.38,3.77,'c','Aggregate by image')
    tile(ax,np.arange(196).reshape(14,14)%9,.22,1.98,1.37,cmap='Blues',vmin=0,vmax=12)
    ax.add_patch(Rectangle((.22+6*1.37/14,1.98+6*1.37/14),1.37/14,1.37/14,fill=False,ec=ORANGE,lw=1.7,zorder=4))
    connector(ax,(1.65,2.66),(2.06,2.66),ORANGE)
    tile(ax,TOY,2.13,2.31,.97,vmin=0,vmax=3)
    text(ax,.91,1.70,'14 × 14 native grid',8,ha='center')
    text(ax,.91,1.40,'196 patches',8,ha='center',color=MUTED)
    text(ax,2.62,2.05,'One 16 × 16 patch',8,ha='center')
    text(ax,2.62,1.73,'16 cells of 4 × 4 pixels',8,ha='center')
    text(ax,1.64,.94,'Fine: 3,136 cells   ·   Native: 196 patches',8.5,ha='center',weight='bold')
    arr=frozen_original(); rank=np.argsort(TOY.ravel())[::-1]
    for row,y in [('Insertion',2.71),('Deletion',1.68)]:
        text(ax,3.55,y+.76,row,9,weight='bold',color=BLUE if row=='Insertion' else ORANGE)
        for col,k in enumerate((0,4,8,16)):
            mask=np.zeros(16,dtype=bool); mask[rank[:k]]=True; mask=np.repeat(np.repeat(mask.reshape(4,4),56,0),56,1)
            if row=='Deletion': mask=~mask
            masked=np.where(mask[:,:,None],arr,128).astype(np.uint8)
            photo(ax,masked,3.61+col*.82,y,.67)
    for col,lab in enumerate(('0','¼','½','1')): text(ax,3.94+col*.82,1.40,lab,8,ha='center')
    text(ax,5.16,.96,'21 fractions · 5 paired tie repeats',8.5,ha='center',weight='bold')
    text(ax,5.16,.64,'Gray reference: normalized zero / RGB 0.5',7.5,ha='center',color=MUTED)
    for j,y in enumerate((3.13,2.75,2.37)):
        text(ax,7.46,y,f'Image {j+1}',8)
        ax.add_patch(Rectangle((8.24,y-.10),.32,.20,fc=BLUE,ec='none'))
        text(ax,8.70,y,'−',11,ha='center')
        ax.add_patch(Rectangle((8.91,y-.10),.32,.20,fc=ORANGE,ec='none'))
        text(ax,9.44,y,f'→  Δ{j+1}',9)
    text(ax,8.40,3.49,'NNS',8,color=BLUE,ha='center',weight='bold')
    text(ax,9.07,3.49,'Patch',8,color=ORANGE,ha='center',weight='bold')
    connector(ax,(8.77,2.07),(8.77,1.78))
    text(ax,8.77,1.49,'Median across images',10,ha='center',weight='bold')
    text(ax,8.77,1.04,'Paired image bootstrap → 95% CI',8.5,ha='center')
    text(ax,5.20,.21,'F = ½(AUC insertion − AUC deletion)          ΔFIDᵢ = Fᵢ(NNS) − Fᵢ(patch)',11,ha='center')
    return fig


def figure_04(config):
    rows=S.csv(base.D10/'paper_data/per_sample_delta.csv')
    values=np.asarray([float(r['delta_fid']) for r in rows if r['status']=='VALID'])
    summary=S.json(base.D10/'analysis.json')['summary']
    med,lo,hi=(float(summary[k]) for k in ('median','bootstrap_ci_lower','bootstrap_ci_upper'))
    if len(values)!=summary['n'] or not np.isclose(np.median(values),med,rtol=0,atol=1e-12): raise base.RenderError('D10 summary mismatch')
    NUMERIC['4']={'n':len(values),'positive_count':int(np.sum(values>0)),'median':med,'ci':[lo,hi],'values':values.tolist()}
    fig,(left,right)=plt.subplots(1,2,figsize=(7.8,3.35),constrained_layout=True,gridspec_kw={'width_ratios':[1.05,1]})
    left.hist(values,bins=np.linspace(0,np.ceil(values.max()),25),color=BLUE,alpha=.82,edgecolor='white',lw=.5)
    left.axvspan(lo,hi,color=ORANGE,alpha=.25,zorder=3); left.axvline(med,color=ORANGE,lw=1.8,zorder=4)
    left.set(xlabel='Per-image ΔFID',ylabel='Number of images',title='(a) Effect distribution')
    left.text(.97,.96,f'Median {med:.3f}'+NL+f'95% CI [{lo:.3f}, {hi:.3f}]',transform=left.transAxes,ha='right',va='top',fontsize=8.5,bbox={'fc':'white','ec':'none','alpha':.9,'pad':4})
    x=np.sort(values); right.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=BLUE,lw=2)
    right.axvspan(lo,hi,color=ORANGE,alpha=.18)
    right.scatter([med],[.5],color=ORANGE,s=35,zorder=4)
    right.set(xlabel='Per-image ΔFID',ylabel='Cumulative proportion',title='(b) All 500 paired effects',ylim=(0,1.04))
    right.text(.95,.13,'500 / 500 positive',transform=right.transAxes,color=BLUE,weight='bold',ha='right',fontsize=11)
    for ax in (left,right): base.clean_axis(ax); ax.spines['left'].set_visible(False); ax.tick_params(length=0); ax.set_xlim(0,np.ceil(values.max()))
    return fig


def jitter_ids(ids, width=.22):
    return np.asarray([(int(base.hashlib.sha256(str(i).encode()).hexdigest()[:8],16)/0xffffffff-.5)*width*2 for i in ids])


def figure_06(config):
    f2a=S.json(base.F2/'plot_data/f2_a_plot_data.json')['plot_payload']
    f2b=S.json(base.F2/'plot_data/f2_b_plot_data.json')['plot_payload']
    fig,axes=plt.subplots(1,3,figsize=(9.6,3.5),constrained_layout=True)
    for row,color,label in zip(f2a['delta_fid_series'],[BLUE,GREEN,ORANGE],['Fine · 4 × 4 cells','Native · NNS','Native · patch baseline']):
        base.ecdf(axes[0],row['values'],color,label,'--' if color==ORANGE else '-')
    axes[0].set(xlabel='ΔFID',ylabel='Cumulative proportion',title='(a) Spatial scale'); axes[0].legend(frameon=False,fontsize=7,loc='lower right')
    NUMERIC['6']={}
    for ax,metric,title in zip(axes[1:],['insertion_auc_mean','deletion_auc_mean'],['(b) Insertion AUC','(c) Deletion AUC']):
        records=[]
        for j,(series,color) in enumerate(zip(f2b['panels'][metric],[MUTED,GREEN,PURPLE])):
            vals=np.asarray(series['values']); q1,med,q3=np.percentile(vals,[25,50,75])
            x=j+jitter_ids(series['sample_ids'],.23)
            ax.scatter(x,vals,s=5,alpha=.20,color=color,edgecolors='none',rasterized=True)
            ax.add_patch(Rectangle((j-.20,q1),.40,q3-q1,facecolor='white',edgecolor=color,lw=1.2,zorder=4))
            ax.add_patch(Rectangle((j-.20,q1),.40,q3-q1,facecolor=color,alpha=.13,edgecolor='none',zorder=4))
            ax.plot([j-.2,j+.2],[med,med],color=color,lw=2.2,zorder=5)
            records.append({'method':series['method'],'n':len(vals),'median':float(med),'q1':float(q1),'q3':float(q3),'values':vals.tolist()})
        NUMERIC['6'][metric]=records
        ax.set_xticks(range(3),['Uniform ='+NL+'nearest','Bilinear','Bicubic']); ax.set(xlim=(-.5,2.5),ylabel='Image-level AUC',title=title)
        ax.text(.97,.98,'500 images · box = IQR',transform=ax.transAxes,ha='right',va='top',fontsize=7,color=MUTED)
    for ax in axes: base.clean_axis(ax); ax.spines['left'].set_visible(False); ax.tick_params(length=0)
    return fig


def figure_07(config):
    p=S.json(base.F2/'plot_data/f2_c_plot_data.json')['plot_payload']
    groups=[next(g for g in p['panels'] if g['stratum']==s) for s in ['overall','correct','misclassified']]
    series=[next(r for r in g['series'] if r['identity']['metric']=='delta_fid') for g in groups]
    data=np.asarray([r['y'] for r in series]); NUMERIC['7']=[{'stratum':g['stratum'],**{k:s[k] for k in ['y','lower','upper','source_counts']}} for g,s in zip(groups,series)]
    fig=plt.figure(figsize=(8.4,3.75)); ax=fig.add_axes([.20,.19,.76,.60])
    cmap=LinearSegmentedColormap.from_list('matrix',['#F0F6FA','#B9D8EB',BLUE])
    ax.imshow(data,cmap=cmap,vmin=0,vmax=2.5,aspect='auto')
    ax.set_xticks(range(4),['Block 0 / head 9','Block 4 / head 2','Block 6 / head 10','Block 11 / head 2'])
    ax.xaxis.tick_top(); ax.tick_params(axis='both',length=0,pad=9,labelsize=8)
    ax.set_yticks(range(3),['Overall','Correct','Misclassified'])
    for row,s in enumerate(series):
        for col in range(4):
            c='white' if data[row,col]>1.3 else INK
            ax.text(col,row-.20,f"{s['y'][col]:.3f}",ha='center',va='center',color=c,fontsize=13,weight='bold')
            ax.text(col,row+.08,f"[{s['lower'][col]:.3f}, {s['upper'][col]:.3f}]",ha='center',va='center',color=c,fontsize=8)
            counts=s['source_counts'][col]
            ax.text(col,row+.32,f"n = {counts['valid']} / {counts['n']}",ha='center',va='center',color=c,fontsize=7.3)
    for k in range(1,4): ax.axvline(k-.5,color='white',lw=6)
    for k in range(1,3): ax.axhline(k-.5,color='white',lw=6)
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.text(.02,.96,'One view of layer/head specificity and prediction strata',fontsize=13,weight='bold',va='top',color=INK)
    fig.text(.20,.88,'Median ΔFID   ·   descriptive 95% CI   ·   valid / expected images',fontsize=8,color=MUTED)
    cbax=fig.add_axes([.66,.065,.28,.027]); cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,2.5),cmap=cmap),cax=cbax,orientation='horizontal',ticks=[0,1.25,2.5]); cb.outline.set_visible(False); cb.ax.tick_params(length=0,labelsize=7)
    fig.text(.20,.075,'Shared linear color scale: median ΔFID',fontsize=8,color=MUTED)
    return fig


def effect_bars(ax, estimates, lows, highs, labels, colors, xlabel):
    """Signed bars from zero; outlined rectangles show the *actual* CI endpoints."""
    y=np.arange(len(labels))[::-1]
    for yy,est,lo,hi,label,color in zip(y,estimates,lows,highs,labels,colors):
        ax.barh(yy,est,height=.31,color=color,alpha=.72,zorder=2)
        ax.add_patch(Rectangle((lo,yy-.25),hi-lo,.50,facecolor=color,alpha=.15,edgecolor='none',zorder=3))
        ax.add_patch(Rectangle((lo,yy-.25),hi-lo,.50,facecolor='none',edgecolor=color,lw=1.1,zorder=4))
        ax.plot([est],[yy],marker='D',ms=4,color=color,mec='white',mew=.6,zorder=5)
        ax.text(est,yy+.35,f'{est:+.3f}',fontsize=8,ha='center',va='bottom',weight='bold',color=color)
    ax.axvline(0,color=MUTED,lw=.8)
    ax.set_yticks(y,labels); ax.set_ylim(-.55,len(labels)-.22); ax.set_xlabel(xlabel)
    ax.spines['left'].set_visible(False); ax.tick_params(length=0); base.clean_axis(ax,grid='x')


def figure_11(config):
    scores=base.as_float_rows(S.csv(base.F3C4/'paper_data/per_image_method_scores.csv'),['image_mean_score'])
    contrasts=base.as_float_rows(S.csv(base.F3C4/'paper_data/contrast_summary.csv'),['mean','bootstrap_ci_95_lower','bootstrap_ci_95_upper'])
    curves=base.as_float_rows(S.csv(base.F3C4/'paper_data/curve_summary.csv'),['actual_fraction','median'])
    fig=plt.figure(figsize=(9.5,5.65),constrained_layout=True); gs=fig.add_gridspec(2,2,width_ratios=[1,1.48])
    dist=fig.add_subplot(gs[:,0]); eff=fig.add_subplot(gs[0,1]); curve=fig.add_subplot(gs[1,1])
    methods=base.METHODS; labels=['NNS','Grad-CAM','Full IG','Rollout','Raw CLS']; colors=[BLUE,ORANGE,GREEN,PURPLE,'#C29B21']
    for j,(method,color) in enumerate(zip(methods,colors)):
        rows=[r for r in scores if r['method_id']==method]; vals=np.asarray([r['image_mean_score'] for r in rows]); q1,med,q3=np.percentile(vals,[25,50,75])
        dist.scatter(j+jitter_ids([r['sample_id'] for r in rows],.20),vals,s=11,alpha=.45,color=color,edgecolors='none')
        dist.add_patch(Rectangle((j-.19,q1),.38,q3-q1,ec=color,fc='white',lw=1,zorder=3))
        dist.plot([j-.19,j+.19],[med,med],color=color,lw=2,zorder=4)
    dist.set_xticks(range(5),labels,rotation=28,ha='right'); dist.set(ylabel='Image-level fidelity F',title='(a) Fifty paired images'); base.clean_axis(dist)
    estimates=[r['mean'] for r in contrasts]; lows=[r['bootstrap_ci_95_lower'] for r in contrasts]; highs=[r['bootstrap_ci_95_upper'] for r in contrasts]
    NUMERIC['11']=[{'contrast_id':r['contrast_id'],'mean':e,'ci':[lo,hi],'n':int(r['n'])} for r,e,lo,hi in zip(contrasts,estimates,lows,highs)]
    # Join by source image when independently checking every paired mean.
    by={(r['sample_id'],r['method_id']):r['image_mean_score'] for r in scores}
    ids=sorted({r['sample_id'] for r in scores})
    for method,expected in zip(methods[1:],estimates):
        if not np.isclose(np.mean([by[(i,methods[0])]-by[(i,method)] for i in ids]),expected,rtol=0,atol=1e-12): raise base.RenderError('F3C paired contrast drift')
    effect_bars(eff,estimates,lows,highs,['NNS − Grad-CAM','NNS − full IG','NNS − rollout','NNS − raw CLS'],[ORANGE,MUTED,BLUE,MUTED],'Paired mean difference · outlined band = 95% CI')
    eff.set_title('(b) Direction and size of the contrasts',loc='left')
    eff.set_xlim(-.082,.076)
    for method,label,color in zip(methods,labels,colors):
        selected=sorted([r for r in curves if r['method_id']==method and r['direction']=='deletion'],key=lambda r:r['actual_fraction'])
        curve.plot([r['actual_fraction'] for r in selected],[r['median'] for r in selected],label=label,color=color)
    curve.set(xlabel='Deleted fraction',ylabel='Normalized target response',title='(c) Deletion trajectories'); base.clean_axis(curve); curve.legend(frameon=False,ncol=3,fontsize=7,loc='lower left')
    return fig


def figure_13(config):
    rows=base.as_float_rows(S.csv(base.Y4/'branch_allocation_summary.csv'),['block','estimate','ci_lower','ci_upper'])
    tests=S.json(base.Y4/'hypothesis_test_table.json'); p=tests.get('payload',tests); tests=p.get('rows',p.get('records',[]))
    selected_tests=[next(r for r in tests if r['hypothesis_id']==h) for h in ['B4','B5']]
    fig,(left,right)=plt.subplots(1,2,figsize=(8.6,3.55),gridspec_kw={'width_ratios':[1.4,1]},constrained_layout=True)
    for metric,label,color in [('first_attention_share','Attention branch',BLUE),('second_mlp_share','MLP branch',ORANGE)]:
        rs=sorted([r for r in rows if r['metric']==metric],key=lambda r:r['block']); x=[r['block'] for r in rs]; y=[r['estimate'] for r in rs]
        left.plot(x,y,marker='o',ms=4,color=color,label=label); left.fill_between(x,[r['ci_lower'] for r in rs],[r['ci_upper'] for r in rs],color=color,alpha=.13,lw=0)
    left.set_xticks(range(12)); left.set(xlabel='ViT block',ylabel='Absolute-mass allocation share',title='(a) Branch allocation across depth'); left.legend(frameon=False); base.clean_axis(left)
    estimates=[float(r['estimate']) for r in selected_tests]; lows=[float(r['ci_lower']) for r in selected_tests]; highs=[float(r['ci_upper']) for r in selected_tests]
    NUMERIC['13']=[{'hypothesis_id':r['hypothesis_id'],'estimate':e,'ci':[lo,hi]} for r,e,lo,hi in zip(selected_tests,estimates,lows,highs)]
    effect_bars(right,estimates,lows,highs,['Attention','MLP'],[BLUE,ORANGE],'Deep − shallow allocation share')
    right.set(title='(b) Redistribution with depth',xlim=(-.29,.21),ylim=(-.60,1.70))
    right.text(.02,.98,'Outlined band: 95% CI',transform=right.transAxes,fontsize=7.5,color=MUTED,va='top')
    return fig


def figure_14(config):
    fig=base.figure_14(config); ax=fig.axes[0]; ax.clear(); ax.axis('off')
    data=S.json(base.S6/'figure_data.json')['payload']['endpoint_rows']; NUMERIC['14']=[{'endpoint':r['endpoint'],'estimate':r['estimate'],'ci':[r['ordinary_BCa_95']['lower'],r['ordinary_BCa_95']['upper']],'n_valid':r['n_valid'],'n_total':r['n_total'],'n_NA':r['n_NA']} for r in data]
    ax.set_title('(a) Endpoint estimates and coverage',loc='left',weight='bold')
    names=['P1 · Target alignment','P2 · Matched occlusion','D1 · Alignment depth','D2 · Probe depth']
    cells=[]
    for name,r in zip(names,data):
        ci=r['ordinary_BCa_95']; cells.append([name,f"{r['estimate']:.4f}",f"[{ci['lower']:.4f}, {ci['upper']:.4f}]",f"{r['n_valid']} / {r['n_total']}"])
    table=ax.table(cellText=cells,colLabels=['Endpoint','Estimate','BCa 95% CI','Valid / total'],colWidths=[.36,.16,.29,.19],bbox=[0,.16,1,.78],cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(7.5)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('white'); cell.set_linewidth(3); cell.set_facecolor(LIGHT if r==0 else ('#F5F8FB' if r%2 else 'white'))
        if r==0: cell.get_text().set_weight('bold')
        if c==0: cell.get_text().set_ha('left')
        if c==1 and r: cell.get_text().set_color(PURPLE if r==4 else BLUE); cell.get_text().set_weight('bold')
    ax.text(0,.06,'Endpoint-specific units; ordinary intervals describe precision.',transform=ax.transAxes,fontsize=7.5,color=MUTED)
    return fig


BUILDERS={1:figure_01,2:figure_02,3:figure_03,4:figure_04,6:figure_06,7:figure_07,11:figure_11,13:figure_13,14:figure_14}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--draft',action='store_true',help='Allow replacement of an unsealed R2 draft only.'); parser.add_argument('--figures',type=int,nargs='+'); args=parser.parse_args()
    if ROOT.exists() and ((ROOT/'terminal_receipt.json').exists() or not args.draft): raise base.RenderError('R2 exists; only an unsealed draft may be updated')
    config=S.json(CONFIG); base.verify_signed_json(config,'R2 config'); base.OUTPUT=ROOT
    base.configure(config); plt.rcParams.update({'axes.titleweight':'bold','axes.titlelocation':'left','axes.titlepad':10,'svg.hashsalt':'nns-unified-r2'})
    S.add(Path(__file__),'renderer_source'); S.add(Path(__file__).with_name('validate_nns_vit_unified_paper_figures_r2.py'),'validator_source'); S.add(Path(base.__file__),'inherited_renderer_source'); S.add(REPO/config['design_revision'],'design_amendment')
    if args.figures:
        for number in args.figures:
            spec=config['figures'][number-1]; base.save_figure(BUILDERS[number](config),number,spec['slug']); print('draft',number,flush=True)
        return 0
    old=S.json(OLD/'render_manifest.json'); base.verify_signed_json(old,'R1 manifest')
    for record in old['source_files']:
        path=next((p for p in [REPO/record['path'],REPO.parent/record['path']] if p.is_file()),None)
        if path is None: raise base.RenderError('Missing inherited source')
        base.require_sha(path,record['sha256'],'R1 inherited source'); S.add(path,record['role'])
    for path in sorted((REPO.parent/'相关文章').glob('*/assets/figures/*')):
        if path.is_file(): S.add(path,'visual_grammar_reference_only')
    for path in sorted((REPO.parent/'相关文章').glob('*/zh/*.md')): S.add(path,'reference_article_text')
    records=[]
    for spec in config['figures']:
        n=spec['number']
        if n in BUILDERS:
            outputs=base.save_figure(BUILDERS[n](config),n,spec['slug'])
        else:
            outputs=[]
            for out in old['figures'][n-1]['outputs']:
                source=REPO/out['path']; base.require_sha(source,out['sha256'],'Inherited R1 output'); S.add(source,'unchanged_R1_figure')
                dest=ROOT/source.parent.name/source.name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,dest)
                outputs.append({**out,'path':dest.relative_to(REPO).as_posix()})
        records.append({'number':n,'slug':spec['slug'],'role':spec['role'],'title':base.FIGURE_TITLES[n],'redesigned':n in BUILDERS,'outputs':outputs}); print(f'Rendered Figure {n:02d}',flush=True)
    write_docs(records)
    base.write_json(ROOT/'numeric_display_data.json',base.signed({'schema_version':1,'document_type':'r2_numeric_display_data','figures':NUMERIC,'new_inference':False}))
    sources=S.inventory()
    manifest=base.signed({'schema_version':1,'document_type':'nns_vit_unified_paper_figure_manifest','status':'RENDERED_PENDING_VISUAL_REVIEW','revision_id':config['revision_id'],'producer_commit':base.git('rev-parse','HEAD'),'figure_count':14,'figures':records,'source_files':sources,'source_files_digest':base.canonical_digest(sources),'cpu_only':True,'new_scientific_inference':False,'old_evidence_or_report_root_modified':False})
    base.write_json(ROOT/'render_manifest.json',manifest)
    base.write_json(ROOT/'input_inventory.json',base.signed({'source_files':sources,'source_files_digest':base.canonical_digest(sources)}))
    base.write_json(ROOT/'cpu_access_audit.json',base.signed({'status':'PASS_CPU_ONLY_PRESENTATION','CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'torch_imported':'torch' in base.sys.modules,'timm_imported':'timm' in base.sys.modules,'new_inference':False,'jpeg_scope':'first frozen F3C3 formal position, reused from Figure 10'}))
    print(json.dumps({'status':'RENDERED_PENDING_VISUAL_REVIEW','root':str(ROOT),'figures':len(records)}))
    return 0


def write_docs(records):
    lines=['# Unified paper figures R2', '', '[Before/after gallery](gallery.html)', '', '| Figure | Title | PNG | PDF | SVG |', '|---:|---|---|---|---|']
    for r in records:
        links=['['+o['format']+']('+Path(o['path']).parent.name+'/'+Path(o['path']).name+')' for o in r['outputs']]
        lines.append('| '+str(r['number'])+' | '+r['title']+' | '+' | '.join(links)+' |')
    (ROOT/'figure_index.md').write_text(NL.join(lines)+NL,encoding='utf-8')
    cards=[]
    for r in records:
        name=next(Path(o['path']).name for o in r['outputs'] if o['format']=='PNG')
        state='重新设计' if r['redesigned'] else '沿用 R1'
        cards.append('<article id="figure-'+str(r['number'])+'"><h2>Figure '+str(r['number'])+' · '+r['title']+'</h2><p>'+state+'</p><div class="pair"><figure><figcaption>R1</figcaption><img loading="lazy" src="../'+OLD.name+'/figures_png/'+name+'"></figure><figure><figcaption>R2</figcaption><img loading="lazy" src="figures_png/'+name+'"></figure></div></article>')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>NNS ViT · Figure R2 review</title>
<style>body{margin:0;background:#edf2f6;color:#172033;font:16px/1.5 system-ui,sans-serif}header{padding:24px 4vw;background:#172033;color:white;position:sticky;top:0;z-index:1}header h1{margin:0;font-size:24px}nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px}nav a{color:#b9def7}main{max-width:1700px;margin:auto;padding:24px}article{background:white;padding:24px;margin-bottom:28px;scroll-margin-top:130px}h2{font-size:20px;margin:0}p{color:#637083;margin:6px 0 16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}figcaption{font-weight:700;border-bottom:1px solid #d9e1ea;padding:8px 0;margin-bottom:16px}img{width:100%;height:auto}.new-only .pair{grid-template-columns:1fr}.new-only .pair figure:first-child{display:none}button{margin-top:10px;border:0;border-radius:4px;padding:8px 14px;cursor:pointer;font:inherit;background:#b9def7}@media(max-width:900px){.pair{grid-template-columns:1fr}header{position:static}}</style>
<header><h1>NNS ViT · Figure 1–14 · R2 视觉修订</h1><nav>'''
    html+=' '.join('<a href="#figure-'+str(n)+'">'+str(n)+'</a>' for n in range(1,15))
    html+='''</nav><button onclick="document.body.classList.toggle('new-only')">切换：R1/R2 对照 ↔ 只看 R2</button></header><main>'''
    html+=NL.join(cards)+'</main></html>'
    (ROOT/'gallery.html').write_text(html,encoding='utf-8')
    # Caption revisions are kept next to the implementation to regenerate the complete package.
    from nns_vit_unified_r2_captions import CAPTIONS_ZH, CAPTIONS_EN, ANALYSIS
    S.add(Path(__file__).with_name('nns_vit_unified_r2_captions.py'),'caption_source')
    for language,overrides,builder in [('zh',CAPTIONS_ZH,base.build_captions_zh),('en',CAPTIONS_EN,base.build_captions_en)]:
        original=builder(); parts=original.split('## Figure ')[1:]; inherited={int(p.split(NL,1)[0]):p.split(NL,1)[1].strip() for p in parts}
        lines=[f'# Unified paper figures R2 · {language}', '']
        for number in range(1,15): lines.extend([f'## Figure {number}','',overrides.get(number,inherited[number]),''])
        (ROOT/f'captions_{language}.md').write_text(NL.join(lines),encoding='utf-8')
    original=base.build_analysis(); parts=original.split('## Image #')[1:]; inherited={int(p.split(':',1)[0]):p.split(NL,1)[1].strip() for p in parts}
    lines=['# 统一论文图 R2：逐图分析','','R2 为展示修订；正式估计、区间和科学状态沿用封存源。','']
    for number in range(1,15): lines.extend([f'## Image #{number}: {base.FIGURE_TITLES[number]}','',ANALYSIS.get(number,inherited[number]),''])
    (ROOT/'figure_analysis.md').write_text(NL.join(lines),encoding='utf-8')
    (ROOT/'README.md').write_text(NL.join(['# NNS ViT unified paper figures R2','','Redesigned: Figures 1–4, 6–7, 11, 13–14. Figures 5, 8–10 and 12 are inherited byte-for-byte from R1.','','- [Figure index](figure_index.md)','- [Chinese captions](captions_zh.md)','- [English captions](captions_en.md)','- [Per-figure analysis](figure_analysis.md)','','PNG (400 dpi), PDF and SVG are supplied for all 14 figures. Quantitative values come from sealed CSV/JSON sources. Method diagrams use one fixed original image and explicitly illustrative grids.','','The renderer requires CPU only and does no new model inference, intervention or hypothesis testing. Run scripts/render_nns_vit_unified_paper_figures_r2.py from a fresh root. Use --draft only while the root has no terminal receipt. Run scripts/validate_nns_vit_unified_paper_figures_r2.py after actual visual review. A sealed root is immutable.','']),encoding='utf-8')


if __name__=='__main__':
    raise SystemExit(main())
