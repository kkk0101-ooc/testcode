#!/usr/bin/env python3
"""CPU-only analytic attention/residual examples and a positive Figure 19 view."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MPLCONFIGDIR','/tmp/nns-attention-residual-mpl')
import argparse
import html
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import render_learning_image_figures_15_21 as previous

common=previous.common
graphics=previous.graphics
REPO=common.REPO
ROOT=REPO/'reports/nns_vit_attention_residual_examples_20260917'
OLD=previous.ROOT
SOURCES={}


def add_source(path):
    relative=path.relative_to(REPO).as_posix()
    SOURCES[relative]={'path':relative,'bytes':path.stat().st_size,'sha256':common.sha(path)}


def read(path):
    add_source(path)
    result=common.read(path)
    if not common.verify(result):raise ValueError('Canonical digest mismatch: '+str(path))
    return result


def attention_forward(value):
    query=value[0]
    scores=query*value
    unnormalized=np.exp(scores-scores.max())
    attention=unnormalized/unnormalized.sum()
    return float(attention@value),attention,scores


def attention_jacobians(value):
    _,attention,_=attention_forward(value)
    probability=attention[1]
    difference=value[1]-value[0]
    score_gradient=np.array([value[1]-2*value[0],value[0]])
    weights_path=difference*probability*(1-probability)*score_gradient
    return attention,weights_path


def compute_examples(steps=256):
    inputs=np.array([1.,3.]);baseline=np.zeros(2)
    nodes,weights=np.polynomial.legendre.leggauss(steps)
    alpha=(nodes+1)/2;weights=weights/2
    contribution_value=np.zeros(2);contribution_attention=np.zeros(2)
    for fraction,weight in zip(alpha,weights):
        value_path,attention_path=attention_jacobians(fraction*inputs)
        contribution_value+=weight*inputs*value_path
        contribution_attention+=weight*inputs*attention_path
    clean,probability,scores=attention_forward(inputs)
    gradient_value,gradient_attention=attention_jacobians(inputs)
    residual_input=np.array([2.,1.])
    skip_map=np.zeros(2);branch_map=np.zeros(2)
    for fraction,weight in zip(alpha,weights):
        point=fraction*residual_input
        sum_value=3*point[0]-point[1]
        downstream=2*sum_value
        skip_map+=weight*residual_input*downstream*np.array([1.,0.])
        branch_map+=weight*residual_input*downstream*np.array([2.,-1.])
    return {
        'status':'ANALYTIC_EXAMPLES_NOT_TRAINED_VIT_RESULTS','integration_steps':steps,
        'attention':{'input':inputs.tolist(),'baseline':baseline.tolist(),'query':1.,'keys':inputs.tolist(),'values':inputs.tolist(),
                     'scores':scores.tolist(),'clean_attention':probability.tolist(),'clean_output':clean,
                     'baseline_output':attention_forward(baseline)[0],
                     'clean_value_gradient':gradient_value.tolist(),'clean_attention_gradient':gradient_attention.tolist(),
                     'value_route_map':contribution_value.tolist(),'attention_route_map':contribution_attention.tolist(),
                     'full_map':(contribution_value+contribution_attention).tolist(),
                     'frozen_clean_attention_map':(probability*inputs).tolist()},
        'residual':{'input':residual_input.tolist(),'baseline':[0.,0.],'skip_output':2.,'branch_output':3.,'sum_output':5.,'final_output':25.,
                    'skip_target_map':skip_map.tolist(),'branch_target_map':branch_map.tolist(),
                    'total_target_map':(skip_map+branch_map).tolist(),'branch_self_response_map':[4.,-1.]}}


def cells(axis,values,xpos,ypos,width=2.3,height=.77,limit=None):
    values=np.asarray(values,dtype=float).reshape(1,-1)
    maximum=limit if limit is not None else max(1.,float(abs(values).max()))
    axis.imshow(values,extent=(xpos,xpos+width,ypos,ypos+height),cmap='RdBu_r',vmin=-maximum,vmax=maximum,interpolation='nearest',aspect='auto')
    for index,value in enumerate(values[0]):
        text=f'{value:.4f}' if abs(value-round(value))>1e-8 else f'{value:.0f}'
        previous.text(axis,xpos+(index+.5)*width/values.shape[1],ypos+height/2,text,11,color='white' if abs(value)>.70*maximum else previous.INK,ha='center')
    axis.add_patch(Rectangle((xpos,ypos),width,height,fill=False,ec='#BBC7D1',lw=.8))


def figure22(examples):
    data=examples['attention'];text=previous.text;arrow=graphics.arrow
    figure,axis=previous.header('Attention: what changes when the input changes?',
        'Two input tokens · one selected query · scalar projection weights W_Q = W_K = W_V = 1.',7.25)
    for xpos,heading in [(.35,'1 · Compute attention'),(4.35,'2 · Separate two routes'),(8.37,'3 · Obtain an input map')]:text(axis,xpos,5.94,heading,12,weight='bold')
    text(axis,.4,5.34,'Input: x = [1, 3]     Reference: [0, 0]',10.5)
    text(axis,.4,4.91,'Query q = 1     Keys K = [1, 3]',10.5)
    text(axis,.4,4.50,'Values V = [1, 3]     Head dimension = 1',10)
    text(axis,.4,3.98,r'$s=qK=[1,3]$',14)
    text(axis,.4,3.48,r'$A=\mathrm{softmax}(s)$',13)
    cells(axis,data['clean_attention'],.43,2.43,2.70,.71,limit=1.)
    text(axis,.43,2.17,'Weights for token 1 and token 2',9,color=previous.MUTED)
    text(axis,.43,1.72,r'$o=0.1192\times1+0.8808\times3$',11)
    text(axis,.43,1.28,'Output o = 2.7616',13,color=previous.BLUE,weight='bold')
    text(axis,4.38,5.18,r'$\mathrm{d}o=A\,\mathrm{d}V+(\mathrm{d}A)V$',15)
    text(axis,4.38,4.48,'At the clean input:',10.5,weight='bold')
    text(axis,4.38,4.01,'Value route gradient   [0.1192, 0.8808]',10)
    text(axis,4.38,3.58,'Weight route gradient [0.2100, 0.2100]',10)
    text(axis,4.38,2.95,'Input path: x(α) = [α, 3α]',12,color=previous.BLUE)
    text(axis,4.38,2.43,'Both routes change along this path.',10)
    text(axis,4.38,1.91,'Integrate the two gradients, then',10)
    text(axis,4.38,1.55,'multiply by each input difference.',10)
    arrow(axis,(7.44,3.20),(8.15,3.20),previous.BLUE)
    for ypos,key,label in [(4.33,'value_route_map','Value route'),(2.89,'attention_route_map','Weight route'),(1.45,'full_map','Full attention learning image')]:
        text(axis,8.4,ypos+1.01,label,10.5,weight='bold')
        cells(axis,data[key],8.42,ypos,2.92,.78,limit=2.8)
    text(axis,11.63,3.98,'+',17,ha='center',color=previous.BLUE)
    text(axis,11.63,2.53,'=',17,ha='center',color=previous.BLUE)
    axis.plot([.35,11.65],[.91,.91],color='#DCE5ED')
    text(axis,.4,.60,'Same total, different input allocation: frozen clean A gives [0.1192, 2.6424].',10.5)
    text(axis,.4,.26,'Illustrative attention module. All displayed input maps keep two input locations; no model training or image selection.',9,color=previous.MUTED)
    return figure


def figure23(examples):
    data=examples['residual'];text=previous.text;arrow=graphics.arrow
    figure,axis=previous.header('Residual branches: count each route once',
        'Analytic example: r = x₁, b = 2x₁ − x₂, z = r + b, final target F = z².',6.60)
    graphics.box(axis,.4,3.84,1.55,.94,face=graphics.LIGHT);text(axis,1.175,4.31,'x = [2, 1]',12,weight='bold',ha='center')
    graphics.box(axis,3.2,4.25,1.64,.62,face='#E6F2F8');text(axis,4.02,4.56,'r = 2',12,ha='center')
    graphics.box(axis,3.2,3.21,1.64,.62,face='#F8ECDF');text(axis,4.02,3.52,'b = 3',12,ha='center')
    arrow(axis,(2.03,4.45),(3.05,4.56),previous.BLUE);arrow(axis,(2.03,4.04),(3.05,3.52),graphics.COLORS[1])
    arrow(axis,(4.94,4.56),(6.03,4.15),previous.BLUE);arrow(axis,(4.94,3.52),(6.03,4.02),graphics.COLORS[1])
    text(axis,6.30,4.10,'+',22,ha='center')
    graphics.box(axis,7.07,3.69,1.33,.85,face=graphics.LIGHT);text(axis,7.74,4.12,'z = 5',12,ha='center')
    arrow(axis,(6.56,4.10),(6.97,4.10))
    arrow(axis,(8.51,4.10),(9.30,4.10));text(axis,8.92,4.56,'square',9,ha='center')
    graphics.box(axis,9.46,3.69,1.90,.85,face='#E8F2F7');text(axis,10.41,4.12,'F = 25',15,weight='bold',ha='center')
    text(axis,.40,2.73,'Along x(α) = [2α, α]:  z = 5α,  ∂F/∂z = 10α.',12)
    for xpos,key,label in [(.45,'skip_target_map','Via the skip edge'),(4.24,'branch_target_map','Via the computed branch'),(8.03,'total_target_map','Learning image for F')]:
        text(axis,xpos,2.22,label,11,weight='bold');cells(axis,data[key],xpos,1.06,2.85,.85,limit=30.)
    text(axis,3.64,1.50,'+',20,ha='center',color=previous.BLUE);text(axis,7.43,1.50,'=',20,ha='center',color=previous.BLUE)
    text(axis,.45,.65,'10 + 20 − 5 = 25 = F(x) − F(0)',12,weight='bold',color=previous.BLUE)
    text(axis,.45,.26,'Branch-alone response map: [4, −1]. Its contribution to F is [20, −5], including the downstream derivative.',9.8,color=previous.MUTED)
    return figure


def positive_figure19():
    old_manifest=read(OLD/'source_arrays_manifest.json')
    numeric=read(OLD/'numeric_display_data.json')
    bindings={Path(row['path']).stem:row for row in numeric['source']['array_bindings']}
    local={row['path']:row for row in old_manifest['files']}
    columns=[0,2,4,6,8,11];records=[]
    config_path=REPO/'config/nns_vit_unified_paper_figures_r4.json';add_source(config_path)
    cmap=graphics.base.configure(common.read(config_path))
    figure,axes=plt.subplots(2,6,figsize=(12,5.25))
    figure.subplots_adjust(left=.16,right=.99,top=.77,bottom=.10,wspace=.08,hspace=.13)
    for column,block in enumerate(columns):
        source=bindings[f'b{block:02d}_first_attention_cell'];path=REPO/source['path']
        if common.sha(path)!=source['sha256']:raise ValueError('Learning image source mismatch')
        add_source(path);raw=np.load(path,allow_pickle=False)
        response_path=OLD/'source_arrays'/f'b{block:02d}_attention_output.npy'
        if common.sha(response_path)!=local[response_path.name]['sha256']:raise ValueError('Token response source mismatch')
        add_source(response_path);response=np.load(response_path,allow_pickle=False)[1:]
        rms=np.sqrt(np.mean(response.astype(float)**2,axis=1)).reshape(14,14)
        previous.heat(axes[0,column],raw,cmap,None,records,f'19_positive_B{block}')
        previous.heat(axes[1,column],rms,cmap,None,records,f'19_RMS_B{block}')
        axes[0,column].set_title(f'Block {block}',fontsize=12,weight='bold')
    for row,label in enumerate(['Positive input\ncontribution','Token response\nRMS']):axes[row,0].text(-.08,.5,label,transform=axes[row,0].transAxes,ha='right',va='center',fontsize=11,weight='bold')
    photo_path=OLD/'source_arrays/original_rgb.npy'
    if common.sha(photo_path)!=local[photo_path.name]['sha256']:raise ValueError('Photograph mismatch')
    add_source(photo_path)
    inset=figure.add_axes([.02,.82,.065,.15]);inset.imshow(np.load(photo_path,allow_pickle=False));inset.axis('off')
    figure.suptitle('Learning images and token responses',fontsize=18,weight='bold',y=.976)
    figure.text(.57,.89,'Positive contribution view · same image and class target',ha='center',fontsize=11,color=previous.MUTED)
    figure.text(.57,.032,'Input coordinates above · internal token indices below · per-panel P99 display',ha='center',fontsize=10,color=previous.MUTED)
    return figure,records


CAPTIONS={
22:'两个 token 的解析 attention 例子。固定选第一个 token 作 query，输入 [1,3]、参考 [0,0]，Q/K/V 标量投影权重均为 1，d_k=1。先算 qK、softmax 和输出，再将输入导数分为 A dV 与 (dA)V 两项。沿直线路径 [α,3α] 采用 Gauss–Legendre 数值积分，得到两个输入位置的完整归因。冻结 clean attention 时的图是另一个条件函数的归因，不等于完整路径中 A dV 那一项。本图是解析示意，不是已训练 ViT 的实测结果。',
23:'解析 residual 例子。输入 [2,1]、参考 [0,0]，旁路 r=x₁，分支 b=2x₁−x₂，相加 z=r+b，最终目标 F=z²。沿路径 z=5α，下游导数为 10α。分别经过旁路和分支的输入贡献为 [10,0] 和 [20,−5]，和为 [30,−5]，总和 25 恢复目标差值。分支对 F 的贡献包括下游导数，因此不同于它自身响应的 [4,−1]。两格固定表示输入位置 1 和 2。',
19:'Figure 19 的正贡献正文候选版：上排是同一类别目标下的 attention 分支正输入贡献，下排是该分支在 patch tokens 上的 RMS。样本、原始地图、六个 block、逐面板 P99 和 nearest 显示均沿用旧 Figure 19；只去除负幅度行。完整 signed 数据和负图保留在原图组。低值不表示“不重要”，可能为负贡献或较小正贡献。'}
SLUGS={22:'attention_routes_worked_example',23:'residual_routes_worked_example',19:'positive_learning_images_reading_view'}


def write_docs(figures,examples):
    lines=['# Attention 与 residual：详细例子和图','',
        '[详细公式与逐步计算](../../docs/attention_residual_learning_image_详解_20260917.md) · [画廊](gallery.html) · [原 Figure 15–21](../nns_vit_learning_image_figures_15_21_20260917/gallery.html)','']
    cards=[]
    for record in figures:
        number=record['number'];png=next(row for row in record['outputs'] if row['format']=='PNG')
        target=str(Path(png['path']).relative_to(ROOT.relative_to(REPO)))
        lines.extend([f'## Image #{number}', '',CAPTIONS[number],'',f'![Figure {number}]({target})',''])
        links={row['format']:str(Path(row['path']).relative_to(ROOT.relative_to(REPO))) for row in record['outputs']}
        cards.append(f'<article><h2>Figure {number}</h2><p>{html.escape(CAPTIONS[number])}</p><a href="{target}"><img src="{target}" alt="Figure {number}"></a><p>'+ ' · '.join(f'<a href="{path}">{fmt}</a>' for fmt,path in links.items())+'</p></article>')
    (ROOT/'figure_analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    (ROOT/'README.md').write_text('# Attention / residual learning image 补充\n\n[画廊](gallery.html) · [逐图说明](figure_analysis.md) · [详细公式](../../docs/attention_residual_learning_image_详解_20260917.md)\n\n两张解析图 Figure 22–23，加一张 Figure 19 正贡献候选版，PNG/PDF/SVG 共九个文件。旧封存图组不修改。\n',encoding='utf-8')
    gallery='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Attention and residual</title><style>body{max-width:1500px;margin:auto;padding:24px;background:#edf2f6;color:#172B3A;font:16px/1.6 system-ui}article{background:white;padding:24px;margin:24px 0}img{width:100%}a{color:#1875A5}</style><h1>Attention / residual · 详细例子</h1><a href="../../docs/attention_residual_learning_image_详解_20260917.md">公式与推导</a>'+''.join(cards)+'</html>'
    (ROOT/'gallery.html').write_text(gallery,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');args=parser.parse_args()
    if ROOT.exists() and ((ROOT/'terminal_receipt.json').exists() or not args.draft):raise ValueError('Only unsealed drafts can be updated')
    ROOT.mkdir(parents=True,exist_ok=True)
    before=common.inventory(OLD)
    for name in ['validation_report.json','terminal_receipt.json']:
        value=read(OLD/name)
        if not value['status'].startswith('PASS'):raise ValueError('Previous figure package not validated')
    for path in [Path(__file__),Path(__file__).with_name('validate_attention_residual_examples.py'),Path(previous.__file__),Path(graphics.__file__),Path(graphics.base.__file__),Path(common.__file__),REPO/'docs/attention_residual_learning_image_详解_20260917.md']:add_source(path)
    config=REPO/'config/nns_vit_unified_paper_figures_r4.json';add_source(config)
    graphics.base.configure(common.read(config));graphics.base.OUTPUT=ROOT;plt.rcParams['svg.hashsalt']='attention-residual-examples'
    examples=compute_examples();figures=[];layout=[];positive_records=[]
    for number in [22,23,19]:
        if number==22:figure=figure22(examples)
        elif number==23:figure=figure23(examples)
        else:figure,positive_records=positive_figure19()
        figure.canvas.draw();renderer=figure.canvas.get_renderer()
        outside=[artist.get_text() for axis in figure.axes for artist in axis.texts if not all(figure.bbox.contains(*point) for point in artist.get_window_extent(renderer).get_points())]
        if outside:raise ValueError(f'Figure {number} text outside canvas: {outside}')
        outputs=graphics.base.save_figure(figure,number,SLUGS[number])
        figures.append({'number':number,'kind':'analytic_example' if number!=19 else 'positive_display_only','outputs':outputs})
        layout.append({'number':number,'text_outside_canvas':outside});print(f'Figure {number} rendered',flush=True)
    write_docs(figures,examples)
    common.write(ROOT/'numeric_display_data.json',common.signed({'examples':examples,'positive_figure19_panels':positive_records}))
    common.write(ROOT/'layout_audit.json',common.signed({'records':layout}))
    if before!=common.inventory(OLD):raise ValueError('Previous root changed')
    common.write(ROOT/'predecessor_snapshot.json',common.signed({'root':OLD.relative_to(REPO).as_posix(),'files':before,'files_digest':common.canonical(before)}))
    sources=[SOURCES[key] for key in sorted(SOURCES)]
    common.write(ROOT/'render_manifest.json',common.signed({'figures':figures,'source_files':sources,'source_files_digest':common.canonical(sources),'status':'PENDING_VISUAL_REVIEW','new_model_forward_calls':0,'new_model_attribution_calls':0}))
    common.write(ROOT/'runtime.json',common.signed({'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'torch_imported':'torch' in sys.modules,'new_model_forward_calls':0,'new_model_attribution_calls':0}))


if __name__=='__main__':main()
