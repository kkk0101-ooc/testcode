#!/usr/bin/env python3
"""Assemble a complete, source-bound paper figure set with real module outputs."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import html
import re
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import render_nns_vit_unified_paper_figures_r7 as g
import validate_nns_vit_unified_paper_figures_r4 as common

REPO = common.REPO
ROOT = REPO/'reports/nns_vit_unified_paper_figures_r9_20260917'
OLD = REPO/'reports/nns_vit_unified_paper_figures_r8_20260916'
LEARNING = REPO/'reports/nns_vit_learning_image_figures_15_21_20260917'
DETAIL = REPO/'reports/nns_vit_attention_residual_examples_20260917'
FORWARD = REPO/'reports/nns_vit_forward_learning_image_comparison_20260917'
PREDECESSORS = [OLD, LEARNING, DETAIL, FORWARD]
SOURCES = {}
LAYOUT = []
DISPLAY = []
TITLES = {2:'A learning image keeps the input origins of an internal response',
          19:'Class-component learning images and token responses',
          22:'Attention: Value and routing derivatives',
          23:'Residual: contributions to a downstream target',
          24:'Normal forward and learning images: step-by-step overview',
          25:'Patch embedding: matching values and learning images',
          26:'Attention: matching values and learning images',
          27:'Residual: matching values and learning images',
          28:'Real learning images through three ViT modules',
          29:'Patch embedding: real channel learning images',
          30:'Attention: real channel learning images',
          31:'Residual: real channel learning images'}
SLUGS = {2:'value_path_construction',24:'forward_learning_overview',25:'forward_learning_patch_embedding',
         26:'forward_learning_attention',27:'forward_learning_residual',28:'real_module_learning_image_progression',
         29:'real_patch_embedding_learning_images',30:'real_attention_learning_images',31:'real_residual_learning_images'}
NAMES = ['patch_embedding','attention','residual']
LABELS = ['Patch embedding','Attention · B0','Residual add · B0']
CHANNELS = [0,255,511,767]


def register(path, role='source'):
    path = Path(path).resolve()
    key = os.path.relpath(path, REPO)
    SOURCES[key] = {'path':key, 'bytes':path.stat().st_size,'sha256':common.sha(path),'role':role}


def checked(path):
    register(path)
    value = common.read(path)
    if not common.verify(value):
        raise ValueError('Digest: '+str(path))
    return value


def resolve_output(root, record):
    path = REPO/record['path']
    if not path.exists(): path = root/record['path']
    if common.sha(path) != record['sha256']:
        raise ValueError('Output changed: '+str(path))
    return path


def inherit(root, spec, number, slug=None):
    outputs = []
    for record in spec['outputs']:
        source = resolve_output(root,record)
        register(source,'inherited_figure')
        filename = source.name if slug is None else f'figure_{number:02d}_{slug}.{record["format"].lower()}'
        target = ROOT/source.parent.name/filename
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        outputs.append({**record,'path':target.relative_to(REPO).as_posix(),
                        'inherited_from':source.relative_to(REPO).as_posix()})
    return {'number':number,'title':TITLES.get(number,spec.get('title',str(number))),
            'kind':spec.get('kind','inherited_paper_figure'),'origin':root.relative_to(REPO).as_posix(),
            'new_render':False,'outputs':outputs}


def draw_heat(axis, raw, cmap, key, signed=False):
    raw = np.asarray(raw,dtype=float)
    if signed:
        limit = float(np.max(np.abs(raw)))
        scaled = np.sign(raw)*np.log1p(9*np.abs(raw)/max(limit,1e-30))/np.log(10)
        axis.imshow(scaled,cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
        rule = 'signed log1p kappa9 maxabs; no clipping'
    else:
        positive = np.maximum(raw,0)
        limit = float(np.percentile(positive,99)) or float(positive.max())
        scaled = np.clip(positive/max(limit,1e-30),0,1)
        axis.imshow(scaled,cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        rule = 'positive P99 with max fallback; display-only clipping'
    axis.axis('off')
    DISPLAY.append({'panel':key,'scale':rule,'limit':limit,'shape':list(raw.shape),
                    'clipped_count':0 if signed else int((np.maximum(raw,0)>limit).sum())})


def fig2(photo,maps,receipt):
    fig,ax = g.canvas(12,6.1)
    t = g.text
    t(ax,.32,5.73,TITLES[2],18,weight='bold')
    t(ax,.32,5.27,'Real input  →  selected ViT response  →  contributions at the original input locations',10.5,color=g.MUTED)
    g.image_tile(ax,photo,.45,2.13,2.45)
    t(ax,1.675,1.86,'Input image',11,weight='bold',ha='center')
    g.arrow(ax,(3.02,3.36),(3.68,3.36))
    g.box(ax,3.83,2.08,3.15,2.5,face=g.LIGHT,edge=g.BLUE)
    t(ax,5.405,4.25,'Trained ViT · Block 0',12,weight='bold',ha='center')
    t(ax,5.405,3.82,'Patch embedding → Attention',11,ha='center')
    t(ax,5.405,3.39,'Channel 0 · 196 patch tokens',10.5,ha='center')
    t(ax,5.405,2.96,r'$s(x)=\sum_{t=1}^{196} O_{t,0}(x)$',14,ha='center')
    delta=receipt['target_deltas'][1][0]
    t(ax,5.405,2.45,f'Response change Δs = {delta:.4f}',11,color=g.BLUE,weight='bold',ha='center')
    g.arrow(ax,(7.10,3.36),(7.80,3.36),g.BLUE)
    raw=maps[1,0];limit=float(np.abs(raw).max())
    signed=np.sign(raw)*np.log1p(9*np.abs(raw)/limit)/np.log(10)
    ax.imshow(signed,extent=(8.0,10.8,1.94,4.74),cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest',aspect='auto')
    t(ax,9.4,1.66,'One learning image · 224 × 224',11,weight='bold',ha='center')
    ax.imshow(np.linspace(-1,1,129)[None],extent=(8.5,10.3,1.29,1.40),cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    t(ax,8.39,1.35,'−',10,ha='right');t(ax,10.41,1.35,'+',10)
    t(ax,9.4,1.08,'Signed path contribution · white = zero',9,color=g.MUTED,ha='center')
    ax.plot([.35,11.65],[.85,.85],color='#DCE5ED')
    t(ax,.4,.52,r'$L_s(i)=(x_i-x_{0i})\int_0^1\partial_i s(x_0+\alpha(x-x_0))\,\mathrm{d}\alpha$',13)
    t(ax,8.0,.52,f'Sum of raw map = {raw.sum():.4f}',11,color=g.BLUE,weight='bold')
    t(ax,.4,.15,'One fixed input and channel · gray reference · full attention derivatives · no spatial smoothing',9,color=g.MUTED)
    DISPLAY.append({'panel':'figure2','scale':'signed log1p kappa9 maxabs; no clipping','limit':limit,'shape':[224,224],'clipped_count':0})
    return fig


def real_panel(number,photo,maps,tokens,cmap,signed=False):
    if number == 28:
        entries=[(i,0) for i in range(3)]
        headings=LABELS
        subtitle='Same input and channel 0 · 196 patch-token responses per module'
    else:
        module=number-29
        entries=[(module,c) for c in range(4)]
        headings=[f'Channel {c}' for c in CHANNELS]
        subtitle=LABELS[module]+' · fixed channel indices · same input and gray reference'
    count=len(entries)
    fig=plt.figure(figsize=(12,6.4))
    grid=fig.add_gridspec(2,count+2,left=.025,right=.985,bottom=.13,top=.81,
                         width_ratios=[.76,.20]+[1]*count,wspace=.08,hspace=.23)
    inp=fig.add_subplot(grid[:,0]);inp.imshow(photo);inp.axis('off');inp.set_title('Input',fontsize=11,weight='bold')
    for col,((module,ch),heading) in enumerate(zip(entries,headings)):
        a=fig.add_subplot(grid[0,col+2]);b=fig.add_subplot(grid[1,col+2])
        draw_heat(a,maps[module,ch],cmap,f'{number}_{module}_{ch}_learning_'+str(signed),signed)
        response=tokens[NAMES[module]][:,CHANNELS[ch]].reshape(14,14)
        draw_heat(b,response,cmap,f'{number}_{module}_{ch}_token',True)
        a.set_title(heading,fontsize=12,weight='bold',pad=8)
        if col==0:
            a.text(-.08,.5,'Learning image\ninput coordinates',transform=a.transAxes,ha='right',va='center',fontsize=9,rotation=90)
            b.text(-.08,.5,'Token response\ninternal indices',transform=b.transAxes,ha='right',va='center',fontsize=9,rotation=90)
    fig.suptitle(TITLES[number]+(' · signed companion' if signed else ''),fontsize=18,weight='bold',y=.96)
    fig.text(.53,.877,subtitle,fontsize=10.5,ha='center',color=g.MUTED)
    fig.text(.53,.066,('Top: complete signed response contributions' if signed else 'Top: positive response contributions')+' · 224 × 224     Bottom: signed output values · 14 × 14',fontsize=10,ha='center',color=g.MUTED)
    fig.text(.53,.025,'Bottom: blue < 0, white = 0, red > 0 · per-panel scales · B0 residual is before the MLP',fontsize=9,ha='center',color=g.MUTED)
    return fig


def save(fig,number,slug):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    outside=[]
    for ax in fig.axes:
        for text in ax.texts:
            if not all(fig.bbox.contains(*point) for point in text.get_window_extent(renderer).get_points()):
                outside.append(text.get_text())
    if outside: raise ValueError(f'Figure {number} out of bounds: {outside}')
    LAYOUT.append({'number':number,'text_outside_canvas':[]})
    return g.base.save_figure(fig,number,slug)


def strip_image_links(text):
    text=text.split('## 实测图的来源')[0].split('## 共同显示与来源说明')[0]
    return re.sub(r'!\[[^\]]*\]\([^\n]+\)','',text).strip()


def captions(receipt):
    zh=common.sections(OLD/'captions_zh.md','## Figure ')
    en=common.sections(OLD/'captions_en.md','## Figure ')
    for lang,values in [('zh',zh),('en',en)]:
        values.update(common.sections(LEARNING/f'captions_{lang}.md','## Figure '))
        for n in values:values[n]=strip_image_links(values[n])
    analyses=common.sections(OLD/'figure_analysis.md','## Image #')
    analyses.update(common.sections(LEARNING/'figure_analysis.md','## Image #'))
    extra=common.sections(DETAIL/'figure_analysis.md','## Image #')
    for n in [19,22,23]:
        analyses[n]=zh[n]=strip_image_links(extra[n])
    zh[2]='真实输入经过 ViT 的 patch embedding 和 B0 attention 后，选定 channel 0 的 196 个 patch-token 输出之和作为内部响应 s。右侧只画一张完整的 224×224 signed learning image，所有值来自本次实际输入路径积分。它保留输入位置，各 RGB 通道贡献相加，整图之和恢复相对灰色参考的响应变化。红/蓝编码同一张图的符号，采用 max-abs signed-log(kappa=9)、nearest，无空间平滑。输入固定为既有 Y2 队列首图，channel 0 在计算前按索引固定。本图定义的是模块自身响应；Figure 3–7 的正式实验仍使用原先指定的 B0/H9 条件 Value-path 目标。'
    en[2]='A real response learning image. For the fixed input ILSVRC2012_val_00007922.JPEG, s is the sum of channel 0 over the 196 patch-token outputs of block-0 attention, including output projection. The full 224×224 signed map is input IG relative to normalized-zero gray, with all attention derivatives active. RGB contributions are summed; the raw map sum reconstructs the response difference. A single map encodes both signs, using max-absolute signed-log(kappa=9), nearest display and no smoothing. Channel 0 was fixed by index before computation. Figures 3–7 retain the distinct preregistered B0/H9 conditional Value-path target.'
    analyses[2]=zh[2]
    en[3]=en[3].replace('the internal scalar f defined in Figure 2','the original B0/H9 CLS head-direction scalar with its clean attention fixed')
    zh[3]=zh[3].replace('Figure 2 定义的内部 f','原协议 B0/H9 的固定 clean-attention head-direction 标量 f')
    zh[3]+=' 本图干预仍针对原协议 B0/H9 的固定 clean-attention Value-path 标量；R9 Figure 2 改用真实模块响应演示通用定义，两者目标分别注明。'
    zh[19]=analyses[19]='同一类别目标下，六个 block 的 attention 分支正输入贡献与对应 token RMS。上排为既有 56×56 input-cell 成分图，下排为 196 个 patch-token 输出跨通道的 RMS。输入固定为 Y2 队列首图、类别目标 235；采用逐面板 P99 与 nearest。完整 signed/负幅度图保留于本包 supplement。上排解释最终类别的分支成分；Figure 28–31 解释模块自身响应。'
    for values in [zh,en,analyses]:
        for k,v in list(values.items()):
            values[k]=v.replace('matched patch baseline','target-matched patch-token baseline with uniform within-patch allocation')
    en[19]='Positive class-component learning images (top) and attention-output token RMS (bottom) for blocks 0,2,4,6,8,11. The first frozen Y2 input and class 235 are held fixed. Top maps retain the original 56×56 input cells; bottom responses are 14×14 patch-token indices. Each nonnegative panel uses P99 scaling with max fallback and nearest display. The signed and negative companion views are included in the supplement. Unlike Figures 28–31, these are downstream class-component maps, not module-own-response maps.'
    en[22]='Analytic two-token attention example with input [1,3], zero reference, scalar Q/K/V weights 1 and head dimension 1. The input derivative includes A dV and (dA)V. Integrated full-path attribution differs from attribution in the model with clean attention fixed. This is an analytic illustration.'
    en[23]='Analytic downstream-target example: r=x1, b=2x1−x2, z=r+b and F=z² at input [2,1] relative to zero. Residual and branch component maps are [10,0] and [20,−5], summing to [30,−5] and output difference 25. These include the downstream derivative of F, unlike branch-own-response maps.'
    step_zh={24:'同一小型解析链的总览：输入 [1,1;3,3] 依次经过线性 patch embedding、完整 softmax attention 和 residual。上排是正常响应，下排是同名响应 learning image，虚线逐一对应。',25:'四个单坐标输入图通过 [0.5,0.5] 权重形成两张 token 学习图。图值相加分别得到响应 1 和 3，输入位置保留。',26:'正常值使用 A 组合 V；本例输入 patch 映射为线性，learning image 可用沿路径平均的完整 Jacobian 系数 C 组合。C 含 Q/K/V 路径，不等于 clean attention。',27:'同一参考和路径下，residual 的响应图逐位置相加，L(z)=L(h)+L(o)。相加发生在原始值上，着色在之后。'}
    step_en={24:'One analytic chain pairs normal forward values with response learning images: input [1,1;3,3], shared linear patch projection, full softmax attention, and residual addition.',25:'Four coordinate seed images produce two token images with shared weights [0.5,0.5]. Their sums equal token responses 1 and 3 while retaining input coordinates.',26:'Forward values mix V using A. For this linear patch-input example, response images combine using path-averaged full Jacobian coefficients C, including Q/K/V derivatives; C is not clean attention. This endpoint combination is not a general propagation rule through nonlinear upstream layers.',27:'For a common input path and reference, residual response images add at each input coordinate: L(z)=L(h)+L(o). Raw values are added before color mapping.'}
    for n in range(24,28):
        zh[n]=step_zh[n]+' 原始四图全部原样纳入 R9。这是算子示意，不是训练后 ViT 实测；真实输出见 Figure 28–31。'
        en[n]=step_en[n]+' These four diagrams are inherited byte-for-byte. The chain omits position, bias, LayerNorm, MLP and classifier for explanation; real trained-ViT outputs appear in Figures 28–31.'
        analyses[n]=zh[n]
    common_zh='固定输入 ILSVRC2012_val_00007922.JPEG，规范化灰色参考为零；patch 输出与 B0 attention、B0 第一次 residual 相加前后的真实响应。每个目标为指定通道在 196 个 patch-token 输出上的总和，CLS 参与前向但不计入这项空间聚合。上排为该响应的输入 IG 正贡献，224×224，RGB 先带符号求和；下排为相同通道的原始 signed token 值，排为 14×14。上排各面板正部 P99（零则 max）缩放，下排各面板 max-abs signed-log(kappa=9)，均用 nearest。完整 signed learning image 与 RGB 原始数组在补充目录。该单样本图不度量总体层间趋势或 feature-map 相关性。'
    common_en='The fixed first Y2 input, ILSVRC2012_val_00007922.JPEG, uses normalized-zero gray as reference. Each target sums one channel over the 196 patch-token module outputs; CLS participates in forward attention but is excluded from this spatial output sum. Top: positive parts of RGB-summed response IG in 224×224 input coordinates, with per-panel P99 scaling (max fallback). Bottom: original signed output-channel values at 14×14 token indices, with per-panel max-absolute signed-log(kappa=9). All displays use nearest interpolation. Complete signed companions and raw RGB arrays are retained. This is one-input module visualization, not a population depth or feature-map correlation test.'
    for n in range(28,32):
        lead='同一通道 0 连续经过三个模块。' if n==28 else f'{LABELS[n-29]} 的固定通道 0、255、511、767。'
        zh[n]=lead+common_zh
        en[n]=('Channel 0 is followed through patch embedding, B0 attention output and the first residual add before the MLP. ' if n==28 else LABELS[n-29]+' is shown at fixed channel indices 0,255,511,767. ')+common_en
        analyses[n]=zh[n]+f' 128 点与 64 点积分对照通过，最大完整性误差 {np.max(receipt["attempts"][-1]["completeness_abs"]):.3e}；原始 residual 相加误差 {receipt["attempts"][-1]["residual_additivity_max_abs"]:.3e}。'
    return zh,en,{n:strip_image_links(v) for n,v in analyses.items()}


def docs(figures,receipt):
    zh,en,analysis=captions(receipt)
    for name,values,marker in [('captions_zh.md',zh,'## Figure '),('captions_en.md',en,'## Figure '),('figure_analysis.md',analysis,'## Image #')]:
        lines=['# R9 · 31 张论文图','', '[画廊](gallery.html) · [图表索引](figure_index.md) · [论文大纲](../../docs/论文大纲_NNS_ViT_R9_20260917.md)','']
        for spec in figures:
            n=spec['number'];lines += [marker+str(n)+(': '+spec['title'] if marker.endswith('#') else ''),'',values[n],'']
            if marker.endswith('#'):
                path=next(o['path'] for o in spec['outputs'] if o['format']=='PNG')
                lines += [f'![Figure {n}]({Path(path).relative_to(ROOT.relative_to(REPO)).as_posix()})','']
        (ROOT/name).write_text('\n'.join(lines),encoding='utf-8')
    index=['# R9 完整论文图集：Figure 1–31','','[画廊](gallery.html) · [逐图说明](figure_analysis.md) · [论文大纲](../../docs/论文大纲_NNS_ViT_R9_20260917.md)','','图号保留已有引用；推荐阅读顺序是概念 → 方法对照 → 真实模块输出 → 定量实验。全部 31 张均提供 PNG/PDF/SVG。','','| 图号 | 内容 | PNG | PDF | SVG |','|---:|---|---|---|---|']
    cards=[]
    groups=[('概念与逐步计算',[1,2,15,16,17,18,24,25,26,27]),('真实模块输出',[28,29,30,31,19,20,21]),('实验结果',[3,4,5,6,7,8,9,10,11,12,13,14]),('导数与下游目标细节',[22,23])]
    for f in figures:
        links={o['format']:Path(o['path']).relative_to(ROOT.relative_to(REPO)).as_posix() for o in f['outputs']}
        index.append(f'| {f["number"]} | {f["title"]} | '+' | '.join(f'[{fmt}]({links[fmt]})' for fmt in ['PNG','PDF','SVG'])+' |')
    for title,numbers in groups:
        cards.append(f'<h2>{title}</h2>')
        for n in numbers:
            f=figures[n-1];links={o['format']:Path(o['path']).relative_to(ROOT.relative_to(REPO)).as_posix() for o in f['outputs']}
            cards.append(f'<article id="figure-{n}"><h3>Figure {n} · {html.escape(f["title"])}</h3><a href="{links["PNG"]}"><img loading="lazy" src="{links["PNG"]}" alt="Figure {n}"></a><p>{html.escape(zh[n])}</p><p>'+ ' · '.join(f'<a href="{links[fmt]}">{fmt}</a>' for fmt in ['PNG','PDF','SVG'])+'</p></article>')
    cards.append('<h2>完整 signed 视图及原始图库</h2>')
    for folder in ['supplement','reference_gallery','all_heads']:
        for p in sorted((ROOT/folder).rglob('*.png')):
            rel=p.relative_to(ROOT).as_posix()
            cards.append(f'<details><summary>{html.escape(rel)}</summary><img loading="lazy" src="{rel}" alt="Companion figure"></details>')
    gallery='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NNS-ViT R9</title><style>body{font:16px/1.6 system-ui;color:#172B3A;background:#eef3f6;max-width:1500px;margin:auto;padding:24px}article,details{background:white;padding:20px;margin:22px 0}img{width:100%}a{color:#137EAB}nav{position:sticky;top:0;background:#eef3f6;padding:12px;z-index:2}</style><h1>NNS-ViT · R9 · 31 张论文图</h1><p>概念、逐步计算、三模块真实 learning image 与完整实验图集。</p><p><a href="figure_index.md">图表索引</a> · <a href="figure_analysis.md">逐图说明</a> · <a href="../../docs/论文大纲_NNS_ViT_R9_20260917.md">论文大纲</a></p><nav>'+ ' · '.join(f'<a href="#figure-{n}">{n}</a>' for n in range(1,32))+'</nav>'+''.join(cards)+'</html>'
    (ROOT/'gallery.html').write_text(gallery,encoding='utf-8')
    (ROOT/'figure_index.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    (ROOT/'README.md').write_text('# NNS-ViT 完整论文图集 R9\n\n[画廊](gallery.html) · [图表索引](figure_index.md) · [逐图说明](figure_analysis.md) · [论文大纲](../../docs/论文大纲_NNS_ViT_R9_20260917.md)\n\n31 张编号图、93 个 PNG/PDF/SVG 导出。Figure 2 使用真实 ViT 模块响应和一张完整 learning image；Figure 24–27 是全部四张上下对照图；Figure 28–31 是本次 CPU 计算的 patch embedding、attention 和 residual 真实响应图。Figure 19 采用已有正贡献候选版。\n\n新模块图固定既有首图、B0 和四个通道。原生输入贡献分辨率为 224×224，内部 token 图为 14×14，完整 signed 视图见 supplement，原始 RGB 归因和两阶积分结果见 source_arrays。已有类别成分图仍保留其 56×56 cell 与 class-target 定义。\n\n旧四个图包原样保留；新计算为单输入方法展示，D/E/F/X/Y/S 总体数值继承已有证据。\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');args=parser.parse_args()
    if (ROOT/'terminal_receipt.json').exists(): raise ValueError('Sealed R9 root')
    before={p.relative_to(REPO).as_posix():common.inventory(p) for p in PREDECESSORS}
    manifests={p:checked(p/'render_manifest.json') for p in PREDECESSORS}
    for p in PREDECESSORS:
        for name in ['terminal_receipt.json','validation_report.json','manual_visual_qc.json']:
            v=checked(p/name)
            if not v['status'].startswith('PASS'):raise ValueError('Predecessor not passed')
    receipt=checked(ROOT/'module_compute_receipt.json')
    if receipt['status']!='PASS_SINGLE_IMAGE_RESPONSE_MAPS':raise ValueError('Module maps not passed')
    for row in receipt['source_files']:
        p=REPO/row['path'];assert common.sha(p)==row['sha256'];register(p,'module_compute_source')
    for p in [Path(__file__),REPO/'scripts/validate_nns_vit_unified_paper_figures_r9.py',Path(g.__file__),Path(g.base.__file__),Path(common.__file__),REPO/'config/nns_vit_unified_paper_figures_r4.json']:
        register(p,'implementation')
    for p in (REPO.parent/'论文1/论文架构').glob('*'):
        if p.is_file() and not p.name.startswith('~'):register(p,'read_historical_reference')
    for p in (REPO.parent/'相关文章/Understanding neural network through neuron level visualization/zh').glob('*.md'):register(p,'read_reference_paper')
    register(REPO.parent/'相关文章/Understanding neural network through neuron level visualization/assets/figures/fig-13.jpg','layout_reference')
    for p in REPO.glob('docs/*20260917.md'):
        if any(k in p.name for k in ['learning_image','正常前向']):register(p,'method_reference')
    for p in PREDECESSORS:
        for name in ['captions_zh.md','captions_en.md','figure_analysis.md','numeric_display_data.json','numeric_data.json']:
            if (p/name).exists():register(p/name,'inherited_caption_or_data')
    cmap=g.base.configure(common.read(REPO/'config/nns_vit_unified_paper_figures_r4.json'))
    g.base.OUTPUT=ROOT;plt.rcParams['svg.hashsalt']='nns-vit-r9'
    photo=np.load(ROOT/'source_arrays/original_rgb.npy',allow_pickle=False)
    maps=np.load(ROOT/'source_arrays/module_response_maps.npy',allow_pickle=False)
    tokens={name:np.load(ROOT/'source_arrays'/f'{name}_clean_tokens.npy',allow_pickle=False) for name in NAMES}
    figures=[]
    for spec in manifests[OLD]['figures']:
        n=spec['number']
        if n==2:
            outputs=save(fig2(photo,maps,receipt),2,SLUGS[2])
            figures.append({'number':2,'title':TITLES[2],'kind':'real_response_definition','new_render':True,'outputs':outputs})
        else:figures.append(inherit(OLD,spec,n))
    for spec in manifests[LEARNING]['figures']:
        n=spec['number']
        if n==19:spec=next(s for s in manifests[DETAIL]['figures'] if s['number']==19)
        figures.append(inherit(DETAIL if n==19 else LEARNING,spec,n))
    for spec in manifests[DETAIL]['figures']:
        if spec['number'] in [22,23]:figures.append(inherit(DETAIL,spec,spec['number']))
    for n,spec in enumerate(manifests[FORWARD]['figures'],24):figures.append(inherit(FORWARD,spec,n,SLUGS[n]))
    for n in range(28,32):
        outputs=save(real_panel(n,photo,maps,tokens,cmap),n,SLUGS[n])
        figures.append({'number':n,'title':TITLES[n],'kind':'real_module_response','new_render':True,'outputs':outputs})
        (ROOT/'supplement').mkdir(exist_ok=True)
        f=real_panel(n,photo,maps,tokens,cmap,signed=True)
        for ext in ['png','pdf','svg']:f.savefig(ROOT/'supplement'/f'figure_{n:02d}_signed.{ext}',dpi=400,bbox_inches='tight')
        plt.close(f)
    for source_dir,target_dir in [(OLD/'reference_gallery',ROOT/'reference_gallery'),(OLD/'supplement',ROOT/'supplement/value_path_details'),(LEARNING/'all_heads',ROOT/'all_heads')]:
        for p in sorted(source_dir.rglob('*')):
            if p.is_file():
                register(p,'reference_gallery');target=target_dir/p.relative_to(source_dir);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    original19=next(s for s in manifests[LEARNING]['figures'] if s['number']==19)
    for out in original19['outputs']:
        p=resolve_output(LEARNING,out);register(p,'complete_signed_companion');shutil.copyfile(p,ROOT/'supplement'/p.name)
    (ROOT/'real_outputs').mkdir(exist_ok=True)
    for mi,name in enumerate(NAMES):
        for ci,ch in enumerate(CHANNELS):
            raw=maps[mi,ci];pos=np.maximum(raw,0);lim=float(np.percentile(pos,99)) or float(pos.max())
            plt.imsave(ROOT/'real_outputs'/f'{name}_channel_{ch:03d}_positive.png',np.clip(pos/max(lim,1e-30),0,1),cmap=cmap,vmin=0,vmax=1)
            limit=float(np.abs(raw).max());s=np.sign(raw)*np.log1p(9*np.abs(raw)/limit)/np.log(10)
            plt.imsave(ROOT/'real_outputs'/f'{name}_channel_{ch:03d}_signed.png',s,cmap='RdBu_r',vmin=-1,vmax=1)
    figures.sort(key=lambda f:f['number']);docs(figures,receipt)
    for p in PREDECESSORS:
        if before[p.relative_to(REPO).as_posix()]!=common.inventory(p):raise ValueError('Predecessor changed')
    common.write(ROOT/'predecessor_snapshot.json',common.signed({'roots':before}))
    common.write(ROOT/'numeric_display_data.json',common.signed({'module_compute_receipt_sha256':common.sha(ROOT/'module_compute_receipt.json'),'display':DISPLAY,'new_cohort_statistics':False}))
    common.write(ROOT/'source_arrays_manifest.json',common.signed({'files':common.inventory(ROOT/'source_arrays')}))
    sources=[SOURCES[k] for k in sorted(SOURCES)]
    common.write(ROOT/'render_manifest.json',common.signed({'revision':'R9','status':'RENDERED_PENDING_VISUAL_REVIEW','figure_count':31,'artifact_count':93,'figures':figures,'source_files':sources,'source_files_digest':common.canonical(sources),'cpu_only':True,'new_inference':'single-input module-response illustration only'}))
    common.write(ROOT/'layout_audit.json',common.signed({'status':'PASS','records':LAYOUT}))
    print('R9 rendered: 31 figures / 93 exports',flush=True)


if __name__=='__main__':main()
