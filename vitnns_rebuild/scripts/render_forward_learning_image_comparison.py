#!/usr/bin/env python3
"""Paired ViT-forward / response-learning-image diagrams for one analytic chain."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MPLCONFIGDIR','/tmp/nns-forward-learning-image-mpl')
import argparse
import html
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import validate_nns_vit_unified_paper_figures_r4 as common

REPO=common.REPO
ROOT=REPO/'reports/nns_vit_forward_learning_image_comparison_20260917'
OLD=REPO/'reports/nns_vit_attention_residual_examples_20260917'
GUIDE=REPO/'docs/ViT正常前向与NNS学习图像逐步对照_20260917.md'
BLUE,ORANGE,INK,MUTED='#1178AA','#C97622','#172B3A','#667789'
PALETTE=[BLUE,ORANGE]
SOURCES={}


def source(path):
    path=Path(path)
    key=os.path.relpath(path,REPO)
    SOURCES[key]={'path':key,'sha256':common.sha(path),'bytes':path.stat().st_size}


def forward_and_jacobians(inputs):
    projection=np.array([[.5,.5,0.,0.],[0.,0.,.5,.5]])
    hidden=projection@inputs
    scores=np.outer(hidden,hidden)
    probability=np.exp(scores-scores.max(axis=1,keepdims=True))
    probability/=probability.sum(axis=1,keepdims=True)
    attention=probability@hidden
    score_jacobian=np.empty((2,2,4));probability_jacobian=np.empty((2,2,4))
    attention_jacobian=np.empty((2,4))
    for query in range(2):
        for key in range(2):
            score_jacobian[query,key]=hidden[key]*projection[query]+hidden[query]*projection[key]
        softmax_jacobian=np.diag(probability[query])-np.outer(probability[query],probability[query])
        probability_jacobian[query]=softmax_jacobian@score_jacobian[query]
        attention_jacobian[query]=probability[query]@projection+hidden@probability_jacobian[query]
    values={'x':inputs,'h':hidden,'q':hidden,'k':hidden,'v':hidden,'scores':scores,'a':probability,'o':attention,'z':hidden+attention}
    jacobians={'x':np.eye(4),'h':projection,'q':projection,'k':projection,'v':projection,
               'scores':score_jacobian,'a':probability_jacobian,'o':attention_jacobian,'z':projection+attention_jacobian}
    return values,jacobians


def compute(steps=256):
    inputs=np.array([1.,1.,3.,3.]);baseline=np.zeros(4)
    values,_=forward_and_jacobians(inputs);base_values,_=forward_and_jacobians(baseline)
    nodes,weights=np.polynomial.legendre.leggauss(steps);fractions=(nodes+1)/2;weights=weights/2
    maps={name:np.zeros(np.asarray(value).shape+(4,)) for name,value in values.items()}
    for fraction,weight in zip(fractions,weights):
        _,jacobians=forward_and_jacobians(baseline+fraction*(inputs-baseline))
        for name in maps:maps[name]+=weight*jacobians[name]*(inputs-baseline)
    for name in maps:
        if not np.allclose(maps[name].sum(-1),values[name]-base_values[name],rtol=0,atol=1e-11):
            raise ValueError('Response-map completeness failed: '+name)
    return {'kind':'analytic_module_chain_not_trained_ViT_measurement','input_shape':[2,2],
            'patches':[[0,1],[2,3]],'reference':baseline.tolist(),'input':inputs.tolist(),
            'patch_projection_weights':[.5,.5],'scalar_qkv_projection_weights':[1,1,1],
            'bias':0,'position_embedding':0,'head_dimension':1,'head_count':1,
            'normalization_and_mlp':'omitted_to_isolate_the_three_requested_modules',
            'meaning':'each map explains its own corresponding scalar response relative to the common input reference',
            'attention_learning_image_coefficients':(maps['o'][:,[0,2]]/np.array([.5,1.5])).tolist(),
            'integration_steps':steps,'values':{key:value.tolist() for key,value in values.items()},
            'baseline_values':{key:value.tolist() for key,value in base_values.items()},
            'learning_images':{key:value.reshape(value.shape[:-1]+(2,2)).tolist() for key,value in maps.items()}}


def txt(axis,xpos,ypos,value,size=11,color=INK,weight='normal',ha='left'):
    return axis.text(xpos,ypos,value,fontsize=size,color=color,weight=weight,ha=ha,va='center',
                     zorder=5,bbox={'facecolor':'white','edgecolor':'none','alpha':.86,'pad':.3} if ypos>3.3 and ypos<4.5 else None)


def arrow(axis,start,end,color=MUTED,dashed=False):
    axis.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,linewidth=1.4,
                                 linestyle='--' if dashed else '-',color=color,zorder=1))


def panel(axis,xpos,ypos,width,height,edge=BLUE,face='#F4F8FB'):
    axis.add_patch(FancyBboxPatch((xpos,ypos),width,height,boxstyle='round,pad=0.01,rounding_size=0.06',
                                 facecolor=face,edgecolor=edge,lw=1.3,zorder=2))


def canvas(title,subtitle,height=7.0,width=16):
    figure=plt.figure(figsize=(width,height))
    axis=figure.add_axes([0,0,1,1]);axis.set(xlim=(0,width),ylim=(0,height));axis.axis('off')
    txt(axis,.32,height-.36,title,19,weight='bold')
    txt(axis,.32,height-.79,subtitle,10.5,color=MUTED)
    return figure,axis


def number(value):
    return str(int(round(value))) if abs(value-round(value))<1e-9 else f'{value:.4f}'


def node(axis,center,ypos,label,value,color=BLUE,width=1.45):
    panel(axis,center-width/2,ypos-.34,width,.68,color)
    txt(axis,center,ypos,f'{label} = {number(value)}',11,color=color,weight='bold',ha='center')


def image_map(axis,center,ypos,value,label,color=BLUE,width=1.12,show_sum=True):
    array=np.asarray(value,dtype=float).reshape(2,2)
    left=center-width/2
    axis.imshow(array,extent=(left,left+width,ypos,ypos+width),cmap='Blues',vmin=0,vmax=3,
                interpolation='nearest',aspect='auto',zorder=3)
    axis.add_patch(Rectangle((left,ypos),width,width,fill=False,ec=color,lw=1.5,zorder=4))
    for row in range(2):
        for column in range(2):
            val=array[row,column]
            txt(axis,left+(column+.5)*width/2,ypos+(1.5-row)*width/2,number(val),8.7,
                color='white' if val>=1.8 else INK,ha='center')
    txt(axis,center,ypos+width+.19,label,10.5,color=color,weight='bold',ha='center')
    if show_sum:txt(axis,center,ypos-.22,'sum = '+number(float(array.sum())),9,color=MUTED,ha='center')


def pair_maps(axis,centers,ypos,values,prefix,width=1.12):
    for index,center in enumerate(centers):image_map(axis,center,ypos,values[index],f'L({prefix}{index+1})',PALETTE[index],width)


def pair_nodes(axis,centers,ypos,values,prefix,width=1.45):
    for index,center in enumerate(centers):node(axis,center,ypos,f'{prefix}{index+1}',values[index],PALETTE[index],width)


def ties(axis,centers,upper,lower):
    for index,center in enumerate(centers):arrow(axis,(center,upper),(center,lower),PALETTE[index],True)


def row_labels(axis,top,bottom):
    txt(axis,.28,top,'ViT',14,weight='bold')
    txt(axis,.28,top-.30,'normal forward',9,color=MUTED)
    txt(axis,.28,bottom,'NNS-ViT',13,weight='bold',color=BLUE)
    txt(axis,.28,bottom-.32,'learning images',9,color=MUTED)


def overview(data):
    value=data['values'];maps=data['learning_images']
    figure,axis=canvas('The same computation, two aligned views',
        'Read left to right: input → patch embedding → attention → residual. Dashed lines pair each value with its learning image.',8)
    row_labels(axis,5.45,2.45)
    stages=[(2.8,'Input'),(6.15,'Patch tokens'),(9.83,'Attention output'),(13.52,'Residual output')]
    for center,label in stages:txt(axis,center,6.46,label,12,weight='bold',ha='center')
    image_map(axis,2.8,4.72,value['x'],'x',BLUE,1.35,False)
    groups=[[5.37,6.94],[9.04,10.61],[12.74,14.31]]
    for centers,key in zip(groups,['h','o','z']):
        pair_nodes(axis,centers,5.45,value[key],key,1.42)
        pair_maps(axis,centers,1.65,maps[key],key,1.25)
        ties(axis,centers,5.02,3.18)
    for index,seed in enumerate(maps['x']):
        center=2.20+(index%2)*1.22;ypos=2.57-(index//2)*1.27
        image_map(axis,center,ypos,seed,f'L(x{index+1})',PALETTE[(index//2)],.84,False)
    arrow(axis,(2.8,4.5),(2.8,3.77),BLUE,True)
    for start,end,label in [(3.89,4.5,'embed'),(7.77,8.18,'attend'),(11.45,11.9,'add')]:
        arrow(axis,(start,5.45),(end,5.45));txt(axis,(start+end)/2,5.98,label,9,ha='center')
        arrow(axis,(start,2.30),(end,2.30),BLUE)
    axis.plot([6.15,6.15,13.52,13.52],[5.87,6.14,6.14,5.87],color='#8495A6',lw=1.1)
    txt(axis,10.9,6.30,'skip connection: + h',8.5,color=MUTED,ha='center')
    axis.plot([6.15,6.15,13.52,13.52],[1.65,1.12,1.12,1.65],color=BLUE,lw=1.1)
    txt(axis,9.8,1.26,'skip learning image: + L(h)',8.5,color=BLUE,ha='center')
    txt(axis,2.8,.82,'Four input values → four seed images',9.5,ha='center',color=MUTED)
    txt(axis,10.0,.82,'Each token scalar → one 2 × 2 input-coordinate image',10.5,ha='center',color=BLUE)
    txt(axis,.32,.28,'Same 2 × 2 reference-zero example in all four diagrams. These are response maps, not class-output component allocations.',10,color=MUTED)
    return figure


def embedding(data):
    value=data['values'];maps=data['learning_images'];figure,axis=canvas('Patch embedding: scalar tokens above, learning images below',
        'One 2 × 2 input; each row is one toy patch. The same projection weights [0.5, 0.5] apply to both patches.',7.6)
    row_labels(axis,5.33,2.10)
    input_centers=[2.7,4.32,6.30,7.92]
    for index,center in enumerate(input_centers):
        node(axis,center,5.3,f'x{index+1}',value['x'][index],PALETTE[index//2],1.24)
        image_map(axis,center,1.5,maps['x'][index],f'L(x{index+1})',PALETTE[index//2],1.15)
        arrow(axis,(center,4.86),(center,2.95),PALETTE[index//2],True)
    txt(axis,3.51,5.97,'Patch 1: top row',11,color=BLUE,ha='center')
    txt(axis,7.11,5.97,'Patch 2: bottom row',11,color=ORANGE,ha='center')
    output_centers=[11.42,14.02]
    pair_nodes(axis,output_centers,5.3,value['h'],'h',1.60)
    pair_maps(axis,output_centers,1.5,maps['h'],'h',1.55)
    ties(axis,output_centers,4.86,3.48)
    arrow(axis,(8.72,5.3),(10.2,5.3));txt(axis,9.46,5.91,'W_E',12,weight='bold',ha='center')
    arrow(axis,(8.72,2.25),(10.2,2.25),BLUE);txt(axis,9.46,2.71,'same weights',9,ha='center',color=BLUE)
    txt(axis,11.42,4.31,'h1 = 0.5x1 + 0.5x2',10,ha='center',color=BLUE)
    txt(axis,14.02,4.31,'h2 = 0.5x3 + 0.5x4',10,ha='center',color=ORANGE)
    txt(axis,5.30,.76,'Zero outside the corresponding input position',10,ha='center',color=MUTED)
    txt(axis,12.72,.76,'L(h1) = 0.5 L(x1) + 0.5 L(x2)',10,ha='center',color=BLUE)
    txt(axis,.32,.27,'The normal output has two numbers. NNS keeps a complete input-position image for each number.',10.5,color=MUTED)
    return figure


def attention(data):
    value=data['values'];maps=data['learning_images'];figure,axis=canvas('Attention: the input and output units keep their matching images',
        'Same tokens h = [1, 3]. One head; W_Q = W_K = W_V = 1. Both query outputs are shown.',7.6)
    row_labels(axis,5.24,2.13)
    inputs=[2.92,4.97];outputs=[11.85,14.15]
    pair_nodes(axis,inputs,5.30,value['h'],'h',1.65)
    pair_maps(axis,inputs,1.30,maps['h'],'h',1.55)
    ties(axis,inputs,4.87,3.32)
    pair_nodes(axis,outputs,5.30,value['o'],'o',1.97)
    pair_maps(axis,outputs,1.30,maps['o'],'o',1.55)
    ties(axis,outputs,4.87,3.32)
    panel(axis,6.55,4.24,3.5,1.82,edge=MUTED)
    txt(axis,8.30,5.80,'Q = K = V = [1, 3]',11,ha='center',weight='bold')
    txt(axis,8.30,5.37,'A = softmax(QKᵀ)',11,ha='center')
    txt(axis,8.30,4.98,'[ 0.1192   0.8808 ]',11,ha='center')
    txt(axis,8.30,4.60,'[ 0.0025   0.9975 ]',11,ha='center')
    arrow(axis,(5.88,5.3),(6.36,5.3));arrow(axis,(10.20,5.3),(10.75,5.3))
    txt(axis,12.99,6.09,'Normal output: o = A h',11,ha='center',weight='bold')
    panel(axis,6.55,1.08,3.5,2.12,edge=BLUE,face='#F1F7FA')
    txt(axis,8.30,2.91,'Contribution coefficients C',11,color=BLUE,weight='bold',ha='center')
    coefficients=data['attention_learning_image_coefficients']
    txt(axis,8.30,2.48,f'[ {coefficients[0][0]:.4f}   {coefficients[0][1]:.4f} ]',11,ha='center')
    txt(axis,8.30,2.10,f'[ {coefficients[1][0]:.4f}   {coefficients[1][1]:.4f} ]',11,ha='center')
    txt(axis,8.30,1.58,'L(o_t) = Σ_j C_tj L(h_j)',10.5,ha='center',color=BLUE)
    txt(axis,8.30,1.26,'C: average Jacobian along this path',8.8,ha='center')
    arrow(axis,(5.88,2.10),(6.36,2.10),BLUE);arrow(axis,(10.20,2.10),(10.75,2.10),BLUE)
    arrow(axis,(8.30,4.07),(8.30,3.35),MUTED,True)
    txt(axis,3.95,3.92,'Input: one image for each token',10,ha='center',color=MUTED)
    txt(axis,12.99,3.92,'Output: one image for each query output',10,ha='center',color=MUTED)
    txt(axis,.32,.69,'Token 1 starts in the top row; its attention-output image includes both input patches.',11,color=BLUE)
    txt(axis,.32,.25,'This toy has linear patch inputs: path-averaged coefficients combine their images exactly. C includes Q, K and V derivatives.',9.5,color=MUTED)
    return figure


def residual(data):
    value=data['values'];maps=data['learning_images'];figure,axis=canvas('Residual: add values above, add the corresponding images below',
        'The two inputs are the original tokens h and the attention outputs o from the previous diagram. z = h + o.',7.3)
    row_labels(axis,5.08,2.2)
    groups=[[2.74,4.49],[7.45,9.20],[12.15,13.90]]
    for centers,key,label in zip(groups,['h','o','z'],['Identity / skip input','Attention branch input','Residual output']):
        txt(axis,sum(centers)/2,5.90,label,12,ha='center',weight='bold')
        pair_nodes(axis,centers,5.03,value[key],key,1.60)
        pair_maps(axis,centers,1.58,maps[key],key,1.38)
        ties(axis,centers,4.61,3.43)
    for xpos,symbol in [(5.98,'+'),(10.67,'=')]:
        txt(axis,xpos,5.03,symbol,25,color=INK,ha='center');txt(axis,xpos,2.24,symbol,25,color=BLUE,ha='center')
    txt(axis,12.98,4.12,'z1 = 1 + 2.7616 = 3.7616',10,ha='center')
    txt(axis,12.98,3.73,'z2 = 3 + 2.9951 = 5.9951',10,ha='center')
    txt(axis,8.0,.91,'L(z1) = L(h1) + L(o1)         L(z2) = L(h2) + L(o2)',13,color=BLUE,ha='center')
    txt(axis,.32,.33,'Same input coordinates, same reference. Addition is elementwise. No extra final class target is introduced in this diagram.',10,color=MUTED)
    return figure


DRAWERS={'01_overview':overview,'02_patch_embedding':embedding,'03_attention':attention,'04_residual':residual}
ZH={'01_overview':'先看这张：正常前向与 learning image 的两排总览',
    '02_patch_embedding':'Patch embedding：四个输入值与两张 token 学习图',
    '03_attention':'Attention：输入 token 图怎样对应到输出 token 图',
    '04_residual':'Residual：上面两个数相加，下面两张图逐格相加'}
NOTES={
    '01_overview':'只看上下对应关系。上排是同一个小网络的正常输入和输出；下排每一张 2×2 小图，都对应正上方同名标量。虚线负责上下对应，横向从 patch embedding 走到 attention，再走到 residual。图中不切换样本，也不切换到另一个最终分类目标。',
    '02_patch_embedding':'上排：原输入的第一行 [1,1] 平均得到 h1=1，第二行 [3,3] 平均得到 h2=3。下排：h1 对应的图是 [0.5,0.5;0,0]，h2 对应的图是 [0,0;1.5,1.5]。每张图仍保留整张原图的四个位置；不属于这个 patch 的位置在这里是零。',
    '03_attention':'上排输入 h1=1、h2=3，用 attention 矩阵 A 得到 o1≈2.7616、o2≈2.9951。下排左侧就是上一张图得到的 L(h1)、L(h2)，用图中明确列出的贡献系数 C 逐格加权，得到各输出单元的 L(o1)、L(o2)。o1 的图从只含上半输入扩展到两块输入都有贡献。C 沿路径追踪 Q/K/V，在这个线性 patch 输入例子中可直接组合两张图；一般非线性上游需要保留路径上的位置导数。',
    '04_residual':'这次上排和下排执行同一种“逐项相加”关系。上排 z1=h1+o1、z2=h2+o2；下排 L(z1)=L(h1)+L(o1)、L(z2)=L(h2)+L(o2)，对同一位置的格子相加。图中两个 residual 输出都能从前一张 attention 图接过来。'}


def save(figure,key):
    records=[]
    for extension in ['png','pdf','svg']:
        path=ROOT/f'figures_{extension}'/f'{key}.{extension}';path.parent.mkdir(exist_ok=True)
        figure.savefig(path,dpi=320,facecolor='white',bbox_inches='tight',pad_inches=.06)
        records.append({'format':extension.upper(),'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':common.sha(path)})
    plt.close(figure)
    return records


def documentation(records):
    index=['# 正常 ViT 与 NNS learning image：两排逐步对照','',
        '[画廊](gallery.html) · [完整读图说明](../../docs/ViT正常前向与NNS学习图像逐步对照_20260917.md)','',
        '这四张图使用同一个 2×2 输入和同一条模块链。先看总览，再按 Patch embedding → Attention → Residual 阅读。','']
    cards=[]
    for record in records:
        key=record['key'];links={row['format']:row['path'] for row in record['outputs']}
        index.extend([f'## {ZH[key]}','',NOTES[key],'',f'![{ZH[key]}]({links["PNG"]})','',
                      ' · '.join(f'[{fmt}]({path})' for fmt,path in links.items()),''])
        cards.append(f'<article id="{key}"><h2>{html.escape(ZH[key])}</h2><p>{NOTES[key]}</p><a href="{links["PNG"]}"><img src="{links["PNG"]}" alt="{key}"></a><p>'+ ' · '.join(f'<a href="{path}">{fmt}</a>' for fmt,path in links.items())+'</p></article>')
    (ROOT/'figure_analysis.md').write_text('\n'.join(index),encoding='utf-8')
    (ROOT/'README.md').write_text('# ViT forward / NNS learning-image comparison\n\n[画廊](gallery.html) · [逐图说明](figure_analysis.md) · [详细定义](../../docs/ViT正常前向与NNS学习图像逐步对照_20260917.md)\n\n四张新图，每张 PNG/PDF/SVG。统一数学例子，所有显示地图解释正上方单元自身的响应，参考输入为零。展示核心是“正常数值”和“同名 learning image”上下对应。\n\n本例保留 patch 线性映射、完整 softmax attention、residual add；省略 bias/position、LayerNorm、MLP 等以讲解三个算子，不是已训练 ViT 的实测输入输出。\n',encoding='utf-8')
    html_text='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ViT 与 NNS：上下对照</title><style>body{max-width:1700px;margin:auto;padding:24px;background:#edf2f6;color:#172B3A;font:16px/1.6 system-ui}article{margin:24px 0;padding:24px;background:white}img{width:100%;height:auto}a{color:#1178AA}h2{font-size:22px}</style><h1>同一计算，两排对应</h1><p>上排：正常 ViT 的输入输出。下排：这些单元的 learning image。</p><a href="../../docs/ViT正常前向与NNS学习图像逐步对照_20260917.md">完整说明</a>'+''.join(cards)+'</html>'
    (ROOT/'gallery.html').write_text(html_text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');args=parser.parse_args()
    if ROOT.exists() and ((ROOT/'terminal_receipt.json').exists() or not args.draft):raise ValueError('Only unsealed drafts may be rewritten')
    ROOT.mkdir(parents=True,exist_ok=True)
    before=common.inventory(OLD)
    for name in ['terminal_receipt.json','validation_report.json']:
        path=OLD/name;source(path);value=common.read(path)
        if not common.verify(value) or not value['status'].startswith('PASS'):raise ValueError('Prior figure package not validated')
    for path in [Path(__file__),Path(__file__).with_name('validate_forward_learning_image_comparison.py'),Path(common.__file__),GUIDE,
                 REPO.parent/'相关文章/Understanding neural network through neuron level visualization/assets/figures/fig-02.jpg',
                 REPO/'src/vitnns/stage_x_nns_v2.py']:
        source(path)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'path','svg.hashsalt':'forward-learning-image-comparison'})
    data=compute();records=[];layout=[]
    for key,drawer in DRAWERS.items():
        figure=drawer(data);figure.canvas.draw();renderer=figure.canvas.get_renderer()
        outside=[artist.get_text() for axis in figure.axes for artist in axis.texts if not all(figure.bbox.contains(*point) for point in artist.get_window_extent(renderer).get_points())]
        if outside:raise ValueError(f'{key}: text outside canvas: {outside}')
        records.append({'key':key,'title':ZH[key],'outputs':save(figure,key)})
        layout.append({'key':key,'text_outside_canvas':outside});print(key+' rendered',flush=True)
    documentation(records)
    common.write(ROOT/'numeric_data.json',common.signed(data))
    common.write(ROOT/'layout_audit.json',common.signed({'records':layout}))
    if before!=common.inventory(OLD):raise ValueError('Prior root changed')
    common.write(ROOT/'predecessor_snapshot.json',common.signed({'root':OLD.relative_to(REPO).as_posix(),'files':before,'files_digest':common.canonical(before)}))
    sources=[SOURCES[key] for key in sorted(SOURCES)]
    common.write(ROOT/'render_manifest.json',common.signed({'figures':records,'source_files':sources,'source_files_digest':common.canonical(sources),'status':'PENDING_VISUAL_REVIEW','schematic':True}))
    common.write(ROOT/'runtime.json',common.signed({'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'torch_imported':'torch' in sys.modules,'new_trained_model_forwards':0,'new_trained_model_attributions':0}))


if __name__=='__main__':main()
