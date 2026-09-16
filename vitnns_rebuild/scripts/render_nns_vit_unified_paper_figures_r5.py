#!/usr/bin/env python3
"""R5 Figure 1: internal response, measured signed map, intervention evidence."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLCONFIGDIR']='/tmp/mplconfig-nns-unified-paper-r5'
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import render_nns_vit_unified_paper_figures_r4 as previous

base=previous.base
REPO=base.REPO
ROOT=REPO/'reports/nns_vit_unified_paper_figures_r5_20260916'
OLD=previous.ROOT
S=base.SOURCES
NL=chr(10)
TITLE='NNS links internal ViT responses to image evidence'
CAPTION_ZH='NNS 将选定内部响应与输入区域的贡献联系起来。(a) 对同一个已训练 ViT-B/16 选择一个内部 head 的响应。(b) 展示 F3 预先冻结的 median-effect 案例（selection_order=44，ILSVRC2012_val_00002287.JPEG，block 0/head 9）的真实 NNS signed fine-cell map；红色为对固定 head-direction Value-path 标量的正贡献，蓝色为负贡献。原图与地图一一对应，非新模型计算或示意归因。为同时显示小幅正负结构，颜色采用 sign(m)·log(1+99·|m|/L)/log(100)，L=max|m|；地图为 56×56，每 cell 对应 4×4 输入像素，最近邻显示。该变换仅用于显示。(c) 正式独立 500 图 cohort 中，500/500 个 NNS−同目标 token 空间基线 ΔFID 为正；每个方格表示一张图像的正差值，方格不编码效应大小。median ΔFID=2.3082，正式图像级配对 bootstrap 95% CI=[2.1111,2.4452]。这里的 F 采用对正向贡献排序的 fine-cell insertion/deletion；该总体结果不为案例中每个局部 signed cell 提供单独因果验证。首图说明内部解释用途及其一项实证支持；计算和评价分别见 Figure 2–3，完整分布见 Figure 4–5。'
CAPTION_EN='NNS links a selected internal ViT response to input contributions. (a) Select one head response in the same trained ViT-B/16. (b) Measured signed NNS fine-cell map for the previously frozen median-effect example (selection_order 44, ILSVRC2012_val_00002287.JPEG, block 0/head 9). Red and blue encode positive and negative contributions to the fixed head-direction Value-path scalar. The image and attribution belong to the same recorded sample. For display, colors use sign(m)·log(1+99·|m|/L)/log(100), L=max|m|, on the recorded 56×56 map, with nearest-neighbor display; each cell represents 4×4 input pixels. (c) All 500 paired NNS-minus-target-matched-token-baseline ΔFID values in the independent formal cohort are positive. Each square represents one positive image-level difference, not its magnitude. Median ΔFID is 2.3082, with the original paired-image-bootstrap 95% CI [2.1111,2.4452]. F evaluates positive-ranking fine-cell insertion/deletion; the cohort result does not individually validate the causal role of every signed cell. Figure 1 presents an internal-explanation use and its empirical support; Figures 2–3 specify the computation and evaluation, and Figures 4–5 show the full results.'
ANALYSIS='''首图只回答一个问题：NNS 能帮助我们分析模型内部的什么？左侧选择同一个已训练 ViT 中的一个 head；中间把这个 head 的内部响应追溯到输入区域，并展示真实正负贡献；右侧用正式 500 图实验说明该归因排序在图像干预中有用。

中间使用既有 median-effect 冻结案例，不按新图美观程度挑选。红蓝表示 signed contribution 的方向；signed-log 仅帮助读图，不改变实验数值。右侧 500 个方格来自所有 500 个逐图 ΔFID 的符号，方格相同大小不表示效应相同。

这个用途不等于“只要格子更细就有创新”。Value-path 的 IG 计算细节在 Figure 2，干预协议在 Figure 3；NNS-v2 的完整 class-logit 成分分解另在 Methods 与 Figure 13 对应分析中说明。首图不把一个 head 响应称作完整类别输出，也不借单个 signed 局部推出因果作用。'''


def load_evidence():
    manifest=S.json(base.F3/'panel_manifest.json'); base.verify_signed_json(manifest,'F3 panel manifest')
    row=next(r for r in manifest['panels'] if r.get('selection_order')==44 and r.get('sources'))
    if row['selection_rules']!=['median_effect'] or row['coordinate']!=[0,9]: raise base.RenderError('Frozen case mismatch')
    source=next(r for r in row['sources'] if r['method']==base.NNS_F3)
    path=REPO/source['source_map_path']; base.require_sha(path,source['source_map_file_sha256'],'Measured NNS map')
    raw=S.array(path)
    if raw.shape!=(56,56): raise base.RenderError('Expected recorded 4x4-cell map')
    jpeg=REPO/row['jpeg']['path']; base.require_sha(jpeg,row['jpeg']['file_sha256'],'Frozen original')
    photo=base.crop_224(jpeg)
    summary=S.json(base.D10/'analysis.json')['summary']
    samples=S.csv(base.D10/'paper_data/per_sample_delta.csv')
    values=[float(r['delta_fid']) for r in samples if r['status']=='VALID']
    if len(values)!=summary['n'] or sum(v>0 for v in values)!=summary['positive_count'] or abs(float(np.median(values))-summary['median'])>1e-12: raise base.RenderError('D10 source mismatch')
    numeric={'sample_id':row['sample_id'],'selection_order':44,'selection_rule':'median_effect','coordinate':[0,9],
             'map_path':source['source_map_path'],'map_sha256':source['source_map_file_sha256'],'map_shape':list(raw.shape),
             'jpeg_path':row['jpeg']['path'],'jpeg_sha256':row['jpeg']['file_sha256'],
             'display_transform':'signed_log1p','kappa':99,'max_abs':float(np.max(np.abs(raw))),'interpolation':'nearest',
             'n':len(values),'positive_count':sum(v>0 for v in values),'median':summary['median'],
             'ci':[summary['bootstrap_ci_lower'],summary['bootstrap_ci_upper']],'delta_fid_values':values,
             'target':'fixed_clean_attention_cls_head_direction','endpoint':'fine_4x4_cell_delta_fid'}
    return photo,raw,numeric


def draw(photo,raw,numeric):
    fig,ax=previous.canvas(4.8)
    t=previous.text; arrow=previous.arrow; BLUE=previous.BLUE; INK=previous.INK; MUTED=previous.MUTED
    t(ax,.25,4.49,TITLE,16,weight='bold')
    t(ax,.25,4.08,'Select a head. Trace signed input contributions. Test their intervention ranking.',10,color=MUTED)
    previous.panel(ax,.25,3.58,'a','Choose an internal response')
    previous.panel(ax,4.21,3.58,'b','Locate input contributions')
    previous.panel(ax,8.19,3.58,'c','Test the ranking')
    previous.photo(ax,photo,.35,1.62,1.35)
    t(ax,1.02,1.35,'Recorded input image',8.3,ha='center')
    arrow(ax,(1.78,2.30),(2.09,2.30))
    for k in range(6):
        ax.add_patch(Rectangle((2.20+k*.17,1.78),.12,.98,fc=BLUE if k==0 else '#DDE7EF',ec='none'))
    t(ax,2.69,3.04,'Trained ViT',9,weight='bold',ha='center')
    for k in range(12):
        ax.add_patch(Circle((2.23+(k%6)*.17,1.45-(k//6)*.19),.045,fc=BLUE if k==9 else '#DDE7EF'))
    t(ax,2.69,.95,'Selected head response',8,ha='center')
    arrow(ax,(3.31,2.30),(4.10,2.30),BLUE)
    t(ax,3.70,2.55,'NNS',8.8,color=BLUE,weight='bold',ha='center')
    limit=numeric['max_abs']; scaled=np.sign(raw)*np.log1p(99*np.abs(raw)/limit)/np.log1p(99)
    ax.imshow(scaled,extent=(4.55,6.73,1.02,3.20),cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest',aspect='auto')
    t(ax,5.64,.76,'Measured signed contribution map',8.3,ha='center')
    ax.imshow(np.linspace(-1,1,120)[None,:],extent=(4.85,6.43,.47,.56),cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    t(ax,4.71,.51,'−',11,color=BLUE,ha='center'); t(ax,6.59,.51,'+',11,color='#B2182B',ha='center')
    arrow(ax,(6.95,2.30),(7.96,2.30),BLUE)
    t(ax,7.45,2.67,'Image',8,ha='center',color=MUTED); t(ax,7.45,2.47,'interventions',8,ha='center',color=MUTED)
    t(ax,9.95,3.00,'500 / 500 images',20,color=BLUE,weight='bold',ha='center')
    t(ax,9.95,2.63,'Higher fidelity than the matched token baseline',8.3,ha='center')
    # One observed sign per square, in the original formal CSV order.
    colors=[BLUE if value>0 else '#B2182B' if value<0 else '#DDE7EF' for value in numeric['delta_fid_values']]
    for i,color in enumerate(colors):
        col=i%25; row=i//25
        ax.add_patch(Rectangle((8.58+col*.108,2.32-(row+1)*.056),.085,.039,fc=color,ec='none'))
    t(ax,9.93,1.00,'Median ΔFID  '+f"{numeric['median']:.4f}",10,weight='bold',ha='center')
    t(ax,9.93,.70,'95% CI  ['+f"{numeric['ci'][0]:.4f}, {numeric['ci'][1]:.4f}"+']',9,ha='center')
    t(ax,.25,.20,'Example: frozen median-effect case, block 0 / head 9. Cohort result: same internal target, 4 × 4-pixel interventions.',8,color=MUTED)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); bounds=fig.bbox
    outside=[a.get_text() for a in ax.texts if not bounds.contains(*a.get_window_extent(renderer).get_points()[0]) or not bounds.contains(*a.get_window_extent(renderer).get_points()[1])]
    if outside: raise base.RenderError('Text outside Figure 1 canvas: '+repr(outside))
    return fig


def docs(records):
    for language,caption in [('zh',CAPTION_ZH),('en',CAPTION_EN)]:
        path=OLD/f'captions_{language}.md'; S.add(path,'inherited_captions'); sections=previous.sections(path,'## Figure '); sections[1]=caption
        lines=[f'# Unified paper figures R5 · {language}','']
        for n in range(1,15): lines.extend([f'## Figure {n}','',sections[n],''])
        (ROOT/f'captions_{language}.md').write_text(NL.join(lines),encoding='utf-8')
    S.add(OLD/'figure_analysis.md','inherited_analysis'); sections=previous.sections(OLD/'figure_analysis.md','## Image #'); sections[1]=ANALYSIS
    lines=['# 统一论文图 R5：逐图分析','','Figure 1 强调内部解释用途，Figure 2–14 沿用 R4。','']
    for r in records: lines.extend([f"## Image #{r['number']}: {r['title']}",'',sections[r['number']],''])
    (ROOT/'figure_analysis.md').write_text(NL.join(lines),encoding='utf-8')
    index=['# Unified paper figures R5','','[完整画廊](gallery.html) · [中文图注](captions_zh.md) · [逐图解读](figure_analysis.md)','','| Figure | Title | Revision | PNG | PDF | SVG |','|---:|---|---|---|---|---|']; cards=[]
    for r in records:
        links={o['format']:Path(o['path']).parent.name+'/'+Path(o['path']).name for o in r['outputs']}; n=r['number']
        revision='R5 重画' if n==1 else '继承 R4'
        index.append('| '+str(n)+' | '+r['title']+' | '+revision+' | '+' | '.join('['+k+']('+links[k]+')' for k in ('PNG','PDF','SVG'))+' |')
        cards.append('<article id="figure-'+str(n)+'"><h2>Figure '+str(n)+' · '+r['title']+'</h2><p>'+revision+'</p><a href="'+links['PNG']+'"><img loading="lazy" src="'+links['PNG']+'" alt="Figure '+str(n)+'"></a><p>'+' · '.join('<a href="'+links[k]+'">'+k+'</a>' for k in ('PNG','PDF','SVG'))+'</p></article>')
    (ROOT/'figure_index.md').write_text(NL.join(index)+NL,encoding='utf-8')
    gallery='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NNS ViT R5</title><style>body{margin:0;background:#edf2f6;color:#172033;font:16px/1.6 system-ui}header{background:#172033;color:white;padding:24px 4vw}h1{font-size:26px;margin:0}nav{display:flex;gap:16px;flex-wrap:wrap}nav a{color:#b9def7}main{max-width:1500px;margin:auto;padding:24px}article{padding:24px;margin-bottom:24px;background:white}h2{font-size:20px}img{width:100%;height:auto}a{color:#1178B4}</style><header><h1>NNS ViT · R5 · Figure 1–14</h1><p>Figure 1：将内部响应追溯到输入证据。Figure 2–3：计算与评价。</p><nav>'''
    gallery+=' '.join('<a href="#figure-'+str(n)+'">'+str(n)+'</a>' for n in range(1,15))
    gallery+='</nav></header><main>'+NL.join(cards)+'</main></html>'
    (ROOT/'gallery.html').write_text(gallery,encoding='utf-8')
    (ROOT/'README.md').write_text('''# NNS ViT unified paper figures R5

完整 Figure 1–14，PNG（400 dpi）/PDF/SVG 共 42 个实体文件。

- [完整画廊](gallery.html)
- [图表索引](figure_index.md)
- [中文图注](captions_zh.md) / [English captions](captions_en.md)
- [逐图解读](figure_analysis.md)

Figure 1 采用“内部 head → 实测正负输入贡献 → 500 图干预证据”呈现研究用途。案例是已冻结的 median-effect 样本，signed map 为原有 56×56 fine-cell 归因；没有新增模型计算。Figure 2–14 连同图注和说明逐字节或逐节继承 R4。

CPU 渲染入口为 scripts/render_nns_vit_unified_paper_figures_r5.py，验证入口为 scripts/validate_nns_vit_unified_paper_figures_r5.py。最终视觉判断记录在 manual_visual_qc.json；终态根不原地重画。
''',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--draft',action='store_true'); args=parser.parse_args()
    if ROOT.exists() and ((ROOT/'terminal_receipt.json').exists() or not args.draft): raise base.RenderError('Only unsealed R5 drafts may be updated')
    config=S.json(previous.CONFIG); base.configure(config); base.OUTPUT=ROOT; plt.rcParams['svg.hashsalt']='nns-unified-r5'; ROOT.mkdir(parents=True,exist_ok=True)
    before=previous.snapshot(OLD)
    for p,role in [(Path(__file__),'renderer'),(Path(base.__file__),'export_helper'),(Path(previous.__file__),'geometry_helper'),(Path(__file__).with_name('validate_nns_vit_unified_paper_figures_r5.py'),'validator'),(REPO/'docs/unified_paper_figure_01_r5_design_20260916.md','design'),(REPO/'docs/nns_paper_framing_audit_20260916.md','framing_audit')]: S.add(p,role)
    old=S.json(OLD/'render_manifest.json'); base.verify_signed_json(old,'R4 manifest')
    for name in ('terminal_receipt.json','validation_report.json','manual_visual_qc.json'):
        v=S.json(OLD/name); base.verify_signed_json(v,'R4 '+name)
        if not v['status'].startswith('PASS'): raise base.RenderError('R4 not validated')
    photo,raw,numeric=load_evidence(); records=[]
    for spec in old['figures']:
        n=spec['number']
        if n==1: outputs=base.save_figure(draw(photo,raw,numeric),1,spec['slug'])
        else:
            outputs=[]
            for out in spec['outputs']:
                source=REPO/out['path']; base.require_sha(source,out['sha256'],'R4 inherited output'); S.add(source,'inherited_R4_figure')
                dest=ROOT/source.parent.name/source.name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,dest)
                outputs.append({**out,'path':dest.relative_to(REPO).as_posix()})
        records.append({**spec,'title':TITLE if n==1 else spec['title'],'revision':'R5' if n==1 else 'R4','redesigned':n==1,'outputs':outputs})
        print(f'Figure {n:02d}: '+('redrawn' if n==1 else 'copied'),flush=True)
    docs(records)
    old_numeric=S.json(OLD/'numeric_display_data.json'); base.verify_signed_json(old_numeric,'R4 numeric')
    base.write_json(ROOT/'numeric_display_data.json',base.signed({'revision':'R5','figure_1':numeric,'inherited_R4_numeric':old_numeric,'new_inference':False}))
    if before!=previous.snapshot(OLD): raise base.RenderError('R4 root changed')
    base.write_json(ROOT/'predecessor_snapshot.json',base.signed({'files':before,'files_digest':base.canonical_digest(before)}))
    sources=S.inventory()
    base.write_json(ROOT/'render_manifest.json',base.signed({'revision':'R5','status':'RENDERED_PENDING_VISUAL_REVIEW','producer_commit':base.git('rev-parse','HEAD'),'figure_count':14,'artifact_count':42,'redesigned_figures':[1],'figures':records,'source_files':sources,'source_files_digest':base.canonical_digest(sources),'cpu_only':True,'new_inference':False}))
    base.write_json(ROOT/'cpu_access_audit.json',base.signed({'status':'PASS_CPU_ONLY_PRESENTATION','CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'torch_imported':'torch' in sys.modules,'timm_imported':'timm' in sys.modules,'new_inference':False,'jpeg_scope':'frozen F3 median-effect case order 44 only'}))
    print(json.dumps({'root':str(ROOT),'status':'RENDERED_PENDING_VISUAL_REVIEW','figures':14}))
    return 0


if __name__=='__main__': raise SystemExit(main())
