#!/usr/bin/env python3
"""Figure 7 R3: grouped bars on a common axis, with an explicit small-effect inset."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from pathlib import Path
import json
import sys

import render_nns_vit_unified_paper_figures_r1 as base
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np

REPO = base.REPO
ROOT = REPO / 'reports/nns_vit_unified_paper_figures_r3_20260916'
PREDECESSOR = REPO / 'reports/nns_vit_unified_paper_figures_r2_20260916'
SOURCE = base.F2 / 'plot_data/f2_c_plot_data.json'
SLUG = 'layer_head_and_prediction_strata'
NL = chr(10)


def render(payload):
    groups = [next(g for g in payload['panels'] if g['stratum'] == name)
              for name in ('overall', 'correct', 'misclassified')]
    series = [next(s for s in g['series'] if s['identity']['metric'] == 'delta_fid') for g in groups]
    colors = ['#9AABBB', '#1679B0', '#D68B35']
    labels = ['Overall · n=500', 'Correct · n=418', 'Misclassified · n=82']
    fig = plt.figure(figsize=(8.6, 4.65))
    ax = fig.add_axes([.085, .18, .885, .66])
    inset = fig.add_axes([.49, .46, .455, .345])
    x = np.arange(4, dtype=float)
    width = .21
    for i, (s, color, label) in enumerate(zip(series, colors, labels)):
        y = np.asarray(s['y']); lo = np.asarray(s['lower']); hi = np.asarray(s['upper'])
        locations = x + (i-1)*width
        ax.bar(locations, y, width*.88, color=color, label=label, zorder=3,
               edgecolor='white', linewidth=.5)
        ax.errorbar(locations, y, yerr=[y-lo, hi-y], fmt='none', ecolor='#394958',
                    capsize=2.4, capthick=.85, elinewidth=.9, zorder=4)
        inset.bar(locations[1:], y[1:], width*.88, color=color, zorder=3,
                  edgecolor='white', linewidth=.5)
        inset.errorbar(locations[1:], y[1:], yerr=[y[1:]-lo[1:], hi[1:]-y[1:]],
                       fmt='none', ecolor='#394958', capsize=2.4, capthick=.85,
                       elinewidth=.9, zorder=4)
        ax.text(locations[0], hi[0]+.06, f'{y[0]:.2f}', ha='center', va='bottom',
                fontsize=8, color='#394958')
    ax.set(xlim=(-.58, 3.55), ylim=(0, 3.05), ylabel='Median ΔFID')
    ax.set_xticks(x, ['Block 0'+NL+'head 9', 'Block 4'+NL+'head 2',
                     'Block 6'+NL+'head 10', 'Block 11'+NL+'head 2*'])
    ax.set_yticks(np.arange(0,3.01,.5))
    ax.set_xlabel('Frozen layer / head coordinate', labelpad=10)
    inset.set(xlim=(.55,3.45), ylim=(0,.10))
    inset.set_xticks([1,2,3], ['B4 / H2','B6 / H10','B11 / H2*'])
    inset.set_yticks([0,.025,.05,.075,.10], ['0','.025','.050','.075','.100'])
    inset.set_title('Detail of the other three coordinates', fontsize=8.5, loc='left', pad=8)
    inset.text(1.0, 1.025, 'ΔFID', transform=inset.transAxes, ha='right', va='bottom',
               fontsize=7.5, color='#637083')
    for axis in (ax,inset):
        axis.set_axisbelow(True)
        axis.grid(axis='y', color='#E1E7ED', linewidth=.65)
        axis.spines['left'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_color('#9AABBB')
        axis.tick_params(axis='both', length=0, labelsize=8, pad=6)
    # Explicitly mark the repeated region; both axes start at zero.
    ax.add_patch(Rectangle((.55,0),2.90,.10,fc='#E8EEF4',ec='#9AABBB',lw=.8,
                           linestyle=(0,(3,3)),alpha=.6,zorder=1))
    fig.add_artist(ConnectionPatch(xyA=(3.45,.10),coordsA=ax.transData,
                                   xyB=(1,0),coordsB=inset.transAxes,
                                   color='#B1BDC9',lw=.8,linestyle=(0,(3,3)),zorder=1))
    ax.text(1.70,.22,'Small-effect region enlarged above',fontsize=7.5,
             color='#637083',ha='center',va='bottom')
    fig.text(.085,.965,'Layer/head specificity across prediction strata',
              fontsize=13,weight='bold',color='#172033',va='top')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.077,.917),
                ncol=3,frameon=False,fontsize=8.2,handlelength=1.4,columnspacing=2.0)
    fig.text(.085,.035,'Bars: median · whiskers: descriptive 95% CI',fontsize=7.5,color='#637083')
    fig.text(.97,.035,'* Valid images: 498 overall / 417 correct / 81 misclassified',
              fontsize=7.2,color='#637083',ha='right')
    numeric = [{'stratum': g['stratum'], **{k:s[k] for k in ('y','lower','upper','source_counts')}}
               for g,s in zip(groups,series)]
    return fig,numeric


CAPTION_ZH = 'Figure 7. 四个冻结 layer/head 坐标的 NNS−patch baseline median ΔFID。灰、蓝、橙色分别表示 overall、correct、misclassified，柱高为 median，误差线为原 Stage E 图像级配对 bootstrap 描述性 95% CI。主图所有坐标共用从零起始的纵轴，右上 inset 将后三个坐标按 0–0.10 的线性纵轴放大；两处呈现的是同一组数值，不能重复计为新观测。总体 cohort 为 500，正确/误分类分别为 418/82；block 11/head 2 的有效数为 498/417/81，其余坐标全部有效，两个 typed failures 仍保留在分母中。坐标在 Stage C 冻结，三组包含重叠样本，不是三个独立实验，也未检验组间等效性。区间沿用 Stage E 的描述性 bootstrap namespace，与 Figure 4 的正式主分析区间略有差异。'
CAPTION_EN = 'Figure 7. Median NNS-minus-patch-baseline ΔFID at four frozen layer/head coordinates. Gray, blue and orange bars identify the overall, correct and misclassified strata; whiskers are the original Stage E descriptive paired-image-bootstrap 95% CIs. The main axis starts at zero and is shared across all coordinates. The inset repeats the last three coordinates on an explicitly labeled linear 0–0.10 scale, also starting at zero. These repeated marks are not additional observations. Expected image counts are 500/418/82; block 11/head 2 has 498/417/81 valid images, with both typed failures retained in the denominators. Other coordinates have complete coverage. Coordinates were frozen in Stage C. Strata overlap and are not independent experiments; similar directions do not establish equivalence. Stage E descriptive intervals use a different bootstrap namespace from the primary Figure 4 analysis.'


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--draft',action='store_true'); args=parser.parse_args()
    if ROOT.exists() and ((ROOT/'terminal_receipt.json').exists() or not args.draft):
        raise base.RenderError('Only an unsealed draft may be updated')
    config=base.read_config(); base.configure(config); base.OUTPUT=ROOT
    payload=base.SOURCES.json(SOURCE)['plot_payload']
    fig,numeric=render(payload); outputs=base.save_figure(fig,7,SLUG)
    base.SOURCES.add(Path(__file__),'renderer_source')
    base.SOURCES.add(Path(base.__file__),'inherited_export_helper')
    base.SOURCES.add(Path(__file__).with_name('validate_nns_vit_figure_07_r3.py'),'validator_source')
    base.SOURCES.add(REPO/'docs/unified_paper_figure_07_r3_design_20260916.md','design_revision')
    base.SOURCES.add(PREDECESSOR/'terminal_receipt.json','predecessor_terminal')
    base.SOURCES.add(PREDECESSOR/'numeric_display_data.json','predecessor_numeric_data')
    old=base.SOURCES.json(PREDECESSOR/'render_manifest.json')
    sources=base.SOURCES.inventory()
    base.write_json(ROOT/'numeric_display_data.json',base.signed({'figure':7,'series':numeric,'scale':'linear','main_ylim':[0,3.05],'inset_ylim':[0,.10],'inset_coordinates':[1,2,3],'new_inference':False}))
    base.write_json(ROOT/'render_manifest.json',base.signed({'schema_version':1,'figure':7,'revision':'R3','status':'RENDERED_PENDING_VISUAL_REVIEW','producer_commit':base.git('rev-parse','HEAD'),'outputs':outputs,'sources':sources,'sources_digest':base.canonical_digest(sources),'cpu_only':True,'torch_imported':'torch' in sys.modules,'timm_imported':'timm' in sys.modules,'new_inference':False,'prior_roots_modified':False}))
    (ROOT/'captions_zh.md').write_text(CAPTION_ZH+NL,encoding='utf-8')
    (ROOT/'captions_en.md').write_text(CAPTION_EN+NL,encoding='utf-8')
    analysis='# Image #7: 分组柱状图与局部放大'+NL*2+'三组结果共用四个横轴坐标，直接比较 overall、correct 和 misclassified。完整纵轴保留 block 0/head 9 与其他路径的真实效应差距；右上 inset 让小效应及其区间可读，避免对数轴或断轴。颜色始终表示预测分层，未添加新统计检验。'+NL*2+CAPTION_ZH+NL
    (ROOT/'figure_analysis.md').write_text(analysis,encoding='utf-8')
    lines=['# Unified paper figure index · R3','','本次仅更新 Figure 7；其余图沿用 R2。','','| Figure | PNG | PDF | SVG |','|---:|---|---|---|']
    for entry in old['figures']:
        n=entry['number']; links=[]
        for out in outputs if n==7 else entry['outputs']:
            p=Path(out['path']); target=(p.parent.name+'/'+p.name) if n==7 else '../'+PREDECESSOR.name+'/'+p.parent.name+'/'+p.name
            links.append('['+out['format']+']('+target+')')
        lines.append('| '+str(n)+' | '+' | '.join(links)+' |')
    (ROOT/'figure_index.md').write_text(NL.join(lines)+NL,encoding='utf-8')
    (ROOT/'README.md').write_text('# Figure 7 · R3'+NL*2+'Grouped bars with a shared zero baseline and an explicit 0–0.10 inset. PNG/PDF/SVG, bilingual captions and unchanged source values are supplied. Other figures remain in R2; see [full figure index](figure_index.md).'+NL,encoding='utf-8')
    print(json.dumps({'status':'RENDERED_PENDING_VISUAL_REVIEW','root':str(ROOT)}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
