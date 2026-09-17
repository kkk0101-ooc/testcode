#!/usr/bin/env python3
"""R8 signed learning-image illustration; other figures inherit R7."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig-nns-unified-r8')

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle

import render_nns_vit_unified_paper_figures_r7 as prior

base, common = prior.base, prior.common
REPO = base.REPO
ROOT = REPO / 'reports/nns_vit_unified_paper_figures_r8_20260916'
OLD = prior.ROOT
TITLE = 'A learning image locates signed input contributions'
DESIGN = REPO / 'docs/unified_paper_figures_r8_design_20260916.md'
SOURCES = base.SOURCES
CAPTION_ZH = 'ViT 中的 signed learning image。左侧是原图和选中的 block 0/head 10 的 CLS head-direction 响应；中间红蓝图表示输入各位置相对灰色参考、沿输入路径累计的带符号贡献。红色为正、蓝色为负、白色为零。右侧放大同一框选区域，分别保留正值和负值幅度；两幅图使用同一个幅值上限，白色表示该 cell 不属于该符号。图来自 F3H 已有 order 44 样本的真实 float32 56×56 map，每 cell 对应 4×4 输入像素。显示使用 sign(R)·log1p(9·abs(R)/L)/log(10)，L=max(abs(R))；全图及局部共用该尺度，采用 nearest 显示，无空间插值和平滑。放大区域是源图行 4–19、列 0–15（零起始）。B0/H10 和放大框是查看结果后为讲解符号而选择的展示案例，不参与总体效果评价。该样本的原 median-effect 身份对应 B0/H9；本图 H10 与 Figure 3–7 的主实验 H9 不是同一坐标。公式给出单 head 条件 Value-path 的目标和输入路径积分，符号不等于独立遮挡该区域后的因果效应。'
CAPTION_EN = 'A signed learning image in a ViT. The input and selected block-0/head-10 CLS head-direction response are followed by its input-space map: red denotes positive path attribution, blue negative attribution and white zero. The two enlarged panels split the same boxed region into positive and negative magnitude under one shared limit. The recorded order-44 F3H map is float32 56×56, with 4×4 input pixels per cell. Display uses sign(R) log1p(9 abs(R)/L)/log(10), L=max(abs(R)), shared by the full map and crops, with nearest display and no spatial smoothing. The crop covers zero-based rows 4–19 and columns 0–15. H10 and the crop were selected after inspecting results for explanation of sign, not cohort evaluation. The original median-effect designation belongs to B0/H9; H10 is a different illustrative coordinate from the primary H9 experiment in Figures 3–7. The formulas define conditional Value-path attribution to one head. Attribution sign is not an independently measured regional ablation effect.'
ANALYSIS = '这次只解释一件事：一个 head 的响应，怎样对应到输入图像的不同位置。左边选定内部响应，中间把它对应的 learning image 画成红蓝图：红色为正贡献，蓝色为负贡献，白色为零。\n\n右边两张图放大的是同一个框。正图保留红色格子，负图保留蓝色格子；同一个格子不会同时出现在两边。正负可能紧挨着，所以这里保留原始小格，不把它们平滑成相似的轮廓。全图和放大图共用色阶。\n\n例子换成了已有 F3H 的 B0/H10，是看过结果后为讲解选的。主实验仍是 B0/H9；这张图不提供 H10 的新效果结论。这里的“正负”指相对参考输入累计的贡献，也不直接表示单独遮掉这个位置，类别分数就一定升或降。'


def load_case():
    manifest = SOURCES.json(base.F3 / 'panel_manifest.json')
    base.verify_signed_json(manifest, 'F3 case manifest')
    case = next(row for row in manifest['panels'] if row.get('selection_order') == 44 and row.get('sources'))
    inventory = SOURCES.json(base.F3H / 'signed_map_inventory.json')
    base.verify_signed_json(inventory, 'F3H maps')
    source = next(row for row in inventory['maps'] if (row['block'], row['head'], row['method']) == (0, 10, base.NNS_F3))
    path = REPO / source['source_path']
    base.require_sha(path, source['source_file_sha256'], 'B0/H10 map')
    raw = SOURCES.array(path)
    if raw.shape != (56, 56) or raw.dtype != np.float32 or prior.array_digest(raw) != source['signed_map_sha256']:
        raise base.RenderError('Source map shape, dtype or digest mismatch')
    photo_path = REPO / case['jpeg']['path']
    base.require_sha(photo_path, case['jpeg']['file_sha256'], 'Source photograph')
    photo = base.crop_224(photo_path)
    if prior.array_digest(photo.astype(np.float32) / np.float32(255)) != case['jpeg']['display_array_sha256']:
        raise base.RenderError('Photograph crop mismatch')
    positive, negative = np.maximum(raw, 0), np.maximum(-raw, 0)
    if np.any((positive > 0) & (negative > 0)) or not np.array_equal(positive - negative, raw):
        raise base.RenderError('Signed split does not reconstruct')
    metadata = {'selection_order': 44, 'sample_id': case['sample_id'], 'coordinate': [0, 10],
                'source': source, 'jpeg': case['jpeg'], 'native_shape': [56, 56], 'cell_pixels': [4, 4],
                'target': 'fixed_clean_attention_cls_head_direction',
                'selection': 'post_hoc_display_choice_after_inspection_of_all_144_coordinates',
                'original_median_effect_coordinate': [0, 9], 'primary_result_coordinate': [0, 9],
                'crop_row_col_bounds_exclusive': [4, 20, 0, 16], 'display': 'signed_log1p',
                'kappa': 9, 'limit': float(np.abs(raw).max()), 'interpolation': 'nearest',
                'spatial_smoothing': False, 'positive_negative_cell_overlap': 0,
                'exact_split_reconstruction': True, 'new_model_inference': False}
    return photo, raw, metadata


def draw(photo, raw, metadata):
    figure, axis = prior.canvas(12, 6.3)
    text, arrow = prior.text, prior.arrow
    text(axis, .3, 5.92, 'What is a learning image in a ViT?', 20, weight='bold')
    text(axis, .3, 5.48, 'Choose an internal response. Map its signed contributions back to the input.', 11, color=prior.MUTED)
    prior.image_tile(axis, photo, .35, 2.77, 1.87)
    text(axis, 1.285, 2.46, 'Input image', 11, weight='bold', ha='center')
    arrow(axis, (2.30, 3.71), (2.75, 3.71))
    prior.box(axis, 2.83, 2.99, 1.57, 1.47, face=prior.LIGHT, edge=prior.BLUE)
    text(axis, 3.615, 4.16, 'ViT · Block 0', 10.5, weight='bold', ha='center')
    for head in range(12):
        xpos, ypos = 3.04 + .23 * (head % 6), 3.76 - .26 * (head // 6)
        axis.add_patch(Circle((xpos, ypos), .075, fc=prior.BLUE if head == 10 else '#CDDCE7'))
    text(axis, 3.615, 3.19, 'CLS · Head 10', 10, color=prior.BLUE, ha='center')
    text(axis, 3.615, 2.64, 'Selected response', 9.5, ha='center')
    arrow(axis, (4.49, 3.71), (4.94, 3.71), prior.BLUE)
    signed = np.sign(raw) * np.log1p(metadata['kappa'] * np.abs(raw).astype(float) / metadata['limit']) / np.log1p(metadata['kappa'])
    axis.imshow(signed, extent=(5.04, 7.50, 2.49, 4.95), cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest', aspect='auto')
    text(axis, 6.27, 2.20, 'Signed learning image', 11, weight='bold', ha='center')
    top, bottom, left, right = metadata['crop_row_col_bounds_exclusive']
    corner = (5.04 + left * 2.46 / 56, 2.49 + (56 - bottom) * 2.46 / 56)
    axis.add_patch(Rectangle(corner, (right-left)*2.46/56, (bottom-top)*2.46/56, fill=False, ec='#263746', lw=1.7))
    text(axis, 9.92, 4.68, 'Same region, enlarged', 11, weight='bold', ha='center')
    arrow(axis, (7.61, 3.83), (8.21, 3.83), prior.MUTED)
    crop = signed[top:bottom, left:right]
    for xpos, values, color, label in [(8.34, np.maximum(crop, 0), '#B2182B', 'Positive (+)'),
                                      (10.09, np.maximum(-crop, 0), '#2166AC', 'Negative (−)')]:
        palette = LinearSegmentedColormap.from_list(label, ['#FFFFFF', color])
        prior.image_tile(axis, values, xpos, 2.76, 1.49, palette)
        axis.add_patch(Rectangle((xpos, 2.76), 1.49, 1.49, fill=False, ec='#CED6DD', lw=.7))
        text(axis, xpos+.745, 2.46, label, 10.5, color=color, weight='bold', ha='center')
    text(axis, 9.83, 2.08, 'White = no contribution of that sign', 9, color=prior.MUTED, ha='center')
    axis.imshow(np.linspace(-1,1,128)[None,:], extent=(5.31,7.23,1.77,1.88), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
    text(axis, 5.24, 1.82, '−', 10, ha='right'); text(axis, 7.31, 1.82, '+', 10)
    text(axis, 6.27, 1.55, 'Shared signed scale · zero at white', 8.5, color=prior.MUTED, ha='center')
    axis.plot([.32,11.67],[1.22,1.22], color='#DCE5ED', lw=1)
    text(axis, .35, .94, 'Fixed head target', 9, color=prior.MUTED)
    text(axis, .35, .57, r'$f(z)=u^{\top}\sum_j A(x)_{\mathrm{CLS},j}V(z)_j$', 12)
    text(axis, 5.25, .94, 'Path attribution to input coordinate i', 9, color=prior.MUTED)
    text(axis, 5.25, .57, r'$R_i=(x_i-x_{0i})\int_0^1\partial_i f(x_0+\alpha(x-x_0))\,d\alpha$', 12)
    text(axis, .35, .18, 'Recorded B0/H10 example chosen for illustration · original cells, no spatial smoothing', 8.5, color=prior.MUTED)
    return figure


def write_docs(records):
    for language, caption in [('zh', CAPTION_ZH), ('en', CAPTION_EN)]:
        source = OLD / f'captions_{language}.md'; SOURCES.add(source)
        sections = common.sections(source, '## Figure '); sections[2] = caption
        lines = [f'# NNS ViT R8 · {language}', '']
        for number in range(1,15): lines.extend([f'## Figure {number}', '', sections[number], ''])
        (ROOT / f'captions_{language}.md').write_text('\n'.join(lines), encoding='utf-8')
    source = OLD / 'figure_analysis.md'; SOURCES.add(source)
    sections = common.sections(source, '## Image #'); sections[2] = ANALYSIS
    lines = ['# R8：这 14 张图分别在说什么', '', 'Figure 2 改为单个响应的红蓝 learning image，其余图沿用 R7。', '']
    index = ['# NNS ViT R8 · Figure 1–14', '', '[画廊](gallery.html) · [逐图说明](figure_analysis.md)', '',
             '| Figure | 内容 | PNG | PDF | SVG |', '|---:|---|---|---|---|']
    cards = []
    for record in records:
        number = record['number']; title = record['title']
        lines.extend([f'## Image #{number}: {title}', '', sections[number], ''])
        links = {out['format']: str(Path(out['path']).relative_to(ROOT.relative_to(REPO))) for out in record['outputs']}
        index.append(f'| {number} | {title} | '+' | '.join(f'[{ext}]({links[ext]})' for ext in ('PNG','PDF','SVG'))+' |')
        cards.append(f'<article id="figure-{number}"><h2>Figure {number} · {html.escape(title)}</h2><a href="{links["PNG"]}"><img loading="lazy" src="{links["PNG"]}" alt="Figure {number}"></a><p>'+ ' · '.join(f'<a href="{links[ext]}">{ext}</a>' for ext in ('PNG','PDF','SVG'))+'</p></article>')
    (ROOT/'figure_analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    (ROOT/'figure_index.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    refs = common.read(ROOT/'reference_gallery_manifest.json')['records']
    for record in refs:
        if record['path'].endswith('.png'):
            cards.append(f'<details><summary>{html.escape(Path(record["path"]).name)}</summary><img loading="lazy" src="{record["path"]}" alt="Original figure"></details>')
    gallery = '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NNS ViT R8</title><style>body{font:16px/1.6 system-ui;background:#edf2f6;color:#172B3A;max-width:1500px;margin:auto;padding:24px}article,details{background:white;padding:20px;margin:20px 0}img{width:100%}a{color:#1875A5}h2{font-size:21px}</style><h1>NNS ViT R8 · Figure 1–14</h1><p>Figure 2：红蓝 learning image。附原始 F3H/F3C 图库。</p><nav>'+' · '.join(f'<a href="#figure-{number}">{number}</a>' for number in range(1,15))+'</nav>'+ '\n'.join(cards)+'</html>'
    (ROOT/'gallery.html').write_text(gallery,encoding='utf-8')
    (ROOT/'README.md').write_text('# NNS ViT unified paper figures R8\n\n[画廊](gallery.html) · [图表索引](figure_index.md) · [逐图说明](figure_analysis.md)\n\nFigure 2 重画为 B0/H10 的 signed learning image；另十三张图、50 张原图库 PNG 和详细方法补充原样继承 R7。14 张主图均有 PNG/PDF/SVG。\n\n样本与 head 是事后展示选择；正式主实验仍绑定 B0/H9。全部处理仅使用已有图像和地图，不启动模型计算。\n\n[方法、learning image 与结论映射](../../docs/论文写作前置条件与结论证据图表映射_20260916.md)\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--draft',action='store_true'); args = parser.parse_args()
    if ROOT.exists() and ((ROOT/'terminal_receipt.json').exists() or not args.draft):
        raise base.RenderError('Only an unsealed R8 draft may be updated')
    ROOT.mkdir(parents=True,exist_ok=True)
    before = common.inventory(OLD)
    old = SOURCES.json(OLD/'render_manifest.json'); base.verify_signed_json(old,'R7 manifest')
    for name in ['terminal_receipt.json','validation_report.json','manual_visual_qc.json']:
        value = SOURCES.json(OLD/name);base.verify_signed_json(value,name)
        if not value['status'].startswith('PASS'): raise base.RenderError('R7 not passed')
    base.configure(SOURCES.json(REPO/'config/nns_vit_unified_paper_figures_r4.json'))
    base.OUTPUT = ROOT; plt.rcParams['svg.hashsalt']='nns-unified-r8'
    for path in [Path(__file__),Path(__file__).with_name('validate_nns_vit_unified_paper_figures_r8.py'),Path(prior.__file__),Path(base.__file__),Path(common.__file__),DESIGN]: SOURCES.add(path,'implementation_or_design')
    photo,raw,metadata = load_case(); records = []; layout = []
    for spec in old['figures']:
        number = spec['number']
        if number == 2:
            figure = draw(photo,raw,metadata);figure.canvas.draw();renderer=figure.canvas.get_renderer()
            outside=[artist.get_text() for axis in figure.axes for artist in axis.texts if not all(figure.bbox.contains(*point) for point in artist.get_window_extent(renderer).get_points())]
            if outside: raise base.RenderError('Text outside canvas: '+str(outside))
            layout.append({'number':2,'text_outside_canvas':outside})
            outputs=base.save_figure(figure,number,spec['slug'])
        else:
            outputs=[]
            for output in spec['outputs']:
                source=REPO/output['path'];base.require_sha(source,output['sha256'],'Inherited figure');SOURCES.add(source,'inherited_R7_figure')
                target=ROOT/source.parent.name/source.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
                outputs.append({**output,'path':target.relative_to(REPO).as_posix()})
        records.append({**spec,'title':TITLE if number==2 else spec['title'],'redesigned':number==2,'revision':'R8' if number==2 else 'R7','outputs':outputs})
        print(f'Figure {number:02d}: '+('redrawn' if number==2 else 'copied'),flush=True)
    refs=SOURCES.json(OLD/'reference_gallery_manifest.json');base.verify_signed_json(refs,'R7 references')
    for record in refs['records']:
        source=OLD/record['path'];base.require_sha(source,record['sha256'],'Reference');SOURCES.add(source,'inherited_reference')
        target=ROOT/record['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    common.write(ROOT/'reference_gallery_manifest.json',refs)
    write_docs(records)
    inherited=SOURCES.json(OLD/'numeric_display_data.json');base.verify_signed_json(inherited,'R7 numeric data')
    common.write(ROOT/'numeric_display_data.json',common.signed({'revision':'R8','figure_2':metadata,'inherited_R7_numeric':inherited,'new_inference':False}))
    if before!=common.inventory(OLD):raise base.RenderError('R7 changed')
    common.write(ROOT/'predecessor_snapshot.json',common.signed({'root':OLD.relative_to(REPO).as_posix(),'files':before,'files_digest':common.canonical(before)}))
    sources=SOURCES.inventory()
    common.write(ROOT/'input_inventory.json',common.signed({'source_files':sources}))
    common.write(ROOT/'render_manifest.json',common.signed({'revision':'R8','status':'RENDERED_PENDING_VISUAL_REVIEW','figure_count':14,'artifact_count':42,'redesigned_figures':[2],'figures':records,'source_files':sources,'source_files_digest':common.canonical(sources),'cpu_only':True,'new_inference':False}))
    common.write(ROOT/'layout_audit.json',common.signed({'status':'PASS','records':layout}))
    common.write(ROOT/'cpu_access_audit.json',common.signed({'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'torch_imported':'torch' in sys.modules,'timm_imported':'timm' in sys.modules,'new_inference':False}))


if __name__=='__main__':main()
