#!/usr/bin/env python3
"""Explanatory figures and source-bound learning images for a fixed ViT input."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MPLCONFIGDIR','/tmp/mplconfig-learning-images-20260917')

import argparse
import hashlib
import html
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image

import render_nns_vit_unified_paper_figures_r7 as graphics
import validate_nns_vit_unified_paper_figures_r4 as common

REPO=common.REPO
ROOT=REPO/'reports/nns_vit_learning_image_figures_15_21_20260917'
BASE=REPO/'reports/nns_vit_unified_paper_figures_r8_20260916'
Y2=REPO/'evidence/stage_y/y2_r2_r2_clean_formal_20260912/formal'
DESIGN=REPO/'docs/learning_image_figures_15_21_design_20260917.md'
MODEL=REPO/'assets/models/vit_base_patch16_224_weights.semantic_candidate.pth'
TAR=REPO/'assets/imagenet/ILSVRC2012_img_val.tar'
SOURCE_RECORDS={}
BLUE,INK,MUTED=graphics.BLUE,graphics.INK,graphics.MUTED
SLUGS={15:'learning_image_correspondence',16:'learning_image_initialization',17:'learning_image_generation',18:'learning_image_vit_operators',19:'learning_images_and_token_responses',20:'learning_images_across_heads',21:'learning_image_component_sum'}
TITLES={15:'One internal component, one input-space image',16:'Keep each input value at its own location',17:'Build and combine learning images',18:'From patch projection to transformer components',19:'Learning images and internal token responses',20:'Different heads, the same image and class target',21:'Components reconstruct the same input attribution'}


def register(path,role='source'):
    path=Path(path).resolve()
    relative=os.path.relpath(path,REPO)
    SOURCE_RECORDS[relative]={'path':relative,'bytes':path.stat().st_size,'sha256':common.sha(path),'role':role}
    return SOURCE_RECORDS[relative]


def read_signed(path):
    register(path)
    value=common.read(path)
    if not common.verify(value):raise ValueError('Bad canonical digest: '+str(path))
    return value


def array_digest(value):
    array=np.ascontiguousarray(value)
    digest=hashlib.sha256()
    for chunk in [str(array.dtype).encode(),b'\0',str(tuple(array.shape)).encode(),b'\0',array.tobytes()]:digest.update(chunk)
    return digest.hexdigest()


def toy_values():
    delta=np.array([[1.,2.],[3.,4.]])
    seeds=np.zeros((4,2,2))
    for index in range(4):seeds[index].flat[index]=delta.flat[index]
    unit_a=np.array([[1.,2.],[0.,0.]])
    unit_b=np.array([[0.,0.],[6.,-4.]])
    return {'input':(delta+1).tolist(),'baseline':np.ones((2,2)).tolist(),'delta':delta.tolist(),
            'seeds':seeds.tolist(),'unit_a':unit_a.tolist(),'unit_b':unit_b.tolist(),
            'output':(2*unit_a+.5*unit_b).tolist(),'output_scalar':7.,
            'status':'ANALYTIC_LINEAR_EXAMPLE_NOT_A_TRAINED_VIT_RESULT'}


def load_existing():
    manifest=read_signed(Y2/'unified_100_unit_manifest.json')
    row=min(manifest['payload']['records'],key=lambda item:item['selection_order'])
    if row['selection_order']!=0 or row['status']!='VALID':raise ValueError('First cohort sample unavailable')
    terminal_path=REPO/row['source_terminal']['path']
    if common.sha(terminal_path)!=row['source_terminal']['sha256']:raise ValueError('Terminal mismatch')
    terminal=read_signed(terminal_path)['payload']
    scientific_path=REPO/row['scientific_result']['path']
    if common.sha(scientific_path)!=row['scientific_result']['sha256']:raise ValueError('Scientific result mismatch')
    scientific=read_signed(scientific_path)['payload']
    by_name={Path(record['path']).stem:record for record in terminal['array_records']}
    wanted=['input_attribution','input_cell_map']
    for block in range(12):
        wanted.extend(f'b{block:02d}_{suffix}' for suffix in ['first_heads_cell','first_attention_cell','first_identity_cell','second_residual_cell','second_mlp_cell'])
    arrays={};bindings=[]
    for name in wanted:
        record=by_name[name];path=REPO/record['path']
        if common.sha(path)!=record['sha256']:raise ValueError('Map file mismatch: '+name)
        array=np.load(path,allow_pickle=False)
        if list(array.shape)!=record['shape'] or array.dtype.name!=record['dtype'] or not np.isfinite(array).all() or array_digest(array)!=record['canonical_digest']:raise ValueError('Map array mismatch: '+name)
        register(path,'existing_class_logit_component');arrays[name]=array;bindings.append(record)
    source_id=row['source_object_id'];name=source_id.split('/')[-1]
    with tarfile.open(TAR,'r:') as archive:
        members=[member for member in archive if member.name==name or member.name.endswith('/'+name)]
        if len(members)!=1:raise ValueError('Image tar member mismatch')
        member=members[0];raw=archive.extractfile(member).read()
    if hashlib.sha256(raw).hexdigest()!=terminal['source_binding']['file_sha256']:raise ValueError('Source image hash mismatch')
    (ROOT/'source_arrays').mkdir(parents=True,exist_ok=True)
    with Image.open(io.BytesIO(raw)) as image:
        image=image.convert('RGB');width,height=image.size
        size=(248,int(248*height/width)) if width<height else (int(248*width/height),248)
        resized=image.resize(size,Image.Resampling.BICUBIC)
        left=int(round((resized.width-224)/2));top=int(round((resized.height-224)/2))
        photo=np.asarray(resized.crop((left,top,left+224,top+224))).copy()
    np.save(ROOT/'source_arrays/original_rgb.npy',photo,allow_pickle=False)
    metadata={'selection_order':0,'source_object_id':source_id,'selection':'first_frozen_Y2_cohort_sample',
              'class_id':terminal['target_identity']['class_id'],'target_rule':terminal['target_identity']['rule'],
              'checkpoint_sha256':terminal['checkpoint_sha256'],'source_terminal':row['source_terminal'],
              'scientific_result':row['scientific_result'],'array_bindings':bindings,
              'tar_member':{'archive':str(TAR.relative_to(REPO)),'name':member.name,'offset_data':member.offset_data,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},
              'preprocessing':{'resize':[int(value) for value in size],'crop':[left,top,left+224,top+224],'normalization':'(RGB/255 - 0.5)/0.5'},
              'recorded_clean_logit':scientific['clean_logit'],'recorded_baseline_logit':scientific['baseline_logit'],
              'recorded_delta':scientific['delta'],'recorded_closure':scientific['closure']}
    return photo,arrays,metadata


def read_responses(photo,metadata):
    import torch
    sys.path.insert(0,str(REPO/'src'))
    from vitnns.modeling import load_canonical_model
    torch.set_num_threads(4)
    torch.manual_seed(0)
    register(MODEL,'existing_model_for_single_CPU_forward')
    model=load_canonical_model(MODEL,device='cpu',unfused_attention=True,
        expected_state_dict_sha256=metadata['checkpoint_sha256'],
        expected_model_name='vit_base_patch16_224.augreg2_in21k_ft_in1k',
        expected_model_identity='F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES',
        expected_canonical_tensor_digest='150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd').eval()
    tensor=(torch.from_numpy(photo.copy()).permute(2,0,1).float().unsqueeze(0)/255-.5)/.5
    captured={};handles=[]
    def hook(name):
        def save(module,inputs,output):captured[name]=output.detach().cpu().numpy()[0].copy()
        return save
    for block,module in enumerate(model.blocks):handles.append(module.drop_path1.register_forward_hook(hook(f'b{block:02d}_attention_output')))
    try:
        with torch.no_grad():logits=model(tensor)[0].cpu().numpy()
    finally:
        for handle in handles:handle.remove()
    target=int(logits.argmax());error=abs(float(logits[metadata['class_id']])-metadata['recorded_clean_logit'])
    if target!=metadata['class_id'] or error>1e-4:raise ValueError(f'CPU target parity failed: {target}, {error}')
    for name,array in captured.items():np.save(ROOT/'source_arrays'/f'{name}.npy',array,allow_pickle=False)
    np.save(ROOT/'source_arrays/logits.npy',logits,allow_pickle=False)
    metadata['cpu_forward']={'device':'cpu','forward_calls':1,'backward_calls':0,'new_IG_calls':0,'target':target,'target_logit':float(logits[target]),'logit_absolute_error':error,'atol':1e-4,'torch_version':torch.__version__,'threads':4}
    journal_path=ROOT/'forward_capture_journal.json'
    journal=common.read(journal_path)['attempts'] if journal_path.exists() else []
    journal.append(dict(metadata['cpu_forward']))
    common.write(journal_path,common.signed({'attempts':journal,'total_executed_forward_calls':len(journal),'retained_forward_capture_count':1,'purpose':'forward-only component-response capture, separate from inherited attribution computation'}))
    return captured


def text(axis,xpos,ypos,value,size=11,color=INK,weight='normal',ha='left'):
    return graphics.text(axis,xpos,ypos,value,size,color,weight,ha)


def header(title,subtitle,height=5.3):
    figure,axis=graphics.canvas(12,height)
    text(axis,.3,height-.38,title,19,weight='bold')
    text(axis,.3,height-.85,subtitle,10.5,color=MUTED)
    return figure,axis


def matrix(axis,values,xpos,ypos,width=1.12,limit=6):
    values=np.asarray(values)
    axis.imshow(values,extent=(xpos,xpos+width,ypos,ypos+width),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest',aspect='auto')
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value=values[row,column]
            text(axis,xpos+(column+.5)*width/values.shape[1],ypos+width-(row+.5)*width/values.shape[0],f'{value:g}',12,color='white' if abs(value)>.65*limit else INK,ha='center')
    axis.add_patch(Rectangle((xpos,ypos),width,width,fill=False,ec='#ABB8C2',lw=.8))


def figure15(toy):
    figure,axis=header(TITLES[15],'The network contains values. The scanner attaches an image to each selected component.')
    positions=[.6,3.65,6.7,9.75]
    for xpos,label,value in zip(positions,['Input change','Unit a','Unit b','Output'],['1, 2, 3, 4','Δa = 3','Δb = 2','Δf = 7']):
        graphics.box(axis,xpos,3.1,1.7,.72,face=graphics.LIGHT,edge=BLUE)
        text(axis,xpos+.85,3.46,value,12,weight='bold',ha='center')
        text(axis,xpos+.85,4.03,label,11,ha='center')
        graphics.arrow(axis,(xpos+.85,3),(xpos+.85,2.3),BLUE,dashed=True)
    for column,(key,xpos) in enumerate(zip(['delta','unit_a','unit_b','output'],positions)):
        matrix(axis,toy[key],xpos+.18,.85,1.34)
    text(axis,.55,.35,'Each image uses the same four input locations. Example values are illustrative.',10,color=MUTED)
    return figure


def figure16(toy):
    figure,axis=header(TITLES[16],'Initialization: one nonzero input coordinate per seed image.')
    matrix(axis,toy['input'],.4,2.7,1.15);text(axis,.98,2.45,'Input x',10,ha='center')
    text(axis,1.79,3.25,'−',22)
    matrix(axis,toy['baseline'],2.15,2.7,1.15);text(axis,2.72,2.45,'Reference x₀',10,ha='center')
    text(axis,3.6,3.25,'=',20)
    matrix(axis,toy['delta'],4.13,2.7,1.15);text(axis,4.7,2.45,'Input difference',10,ha='center')
    graphics.arrow(axis,(5.55,3.27),(6.12,3.27),BLUE)
    for index,seed in enumerate(toy['seeds']):
        xpos=6.45+index*1.33;matrix(axis,seed,xpos,2.7,1.06)
        text(axis,xpos+.53,2.43,f'Seed {index+1}',10,ha='center')
        axis.add_patch(Circle((xpos+.53,1.65),.18,fc=graphics.COLORS[index%3],ec='none'))
        graphics.arrow(axis,(xpos+.53,2.15),(xpos+.53,1.94),MUTED,dashed=True)
        text(axis,xpos+.53,1.15,f'coordinate {index+1}',9,ha='center')
    text(axis,.4,1.65,r'$S_i=(x_i-x_{0i})e_i$',16,color=BLUE)
    text(axis,.4,1.08,r'$\sum_i S_i=x-x_0$',16)
    text(axis,.4,.37,'ViT adds a reference difference to the original NNS seed-image idea. No image is spatially rearranged.',10,color=MUTED)
    return figure


def figure17(toy):
    figure,axis=header(TITLES[17],'An analytic linear example: a = x₁ + x₂, b = 2x₃ − x₄, f = 2a + 0.5b.',6.2)
    matrix(axis,toy['delta'],.4,3.35,1.3)
    text(axis,1.05,3.04,'Input difference',10,ha='center')
    graphics.arrow(axis,(1.95,4.05),(3.15,4.05),BLUE)
    text(axis,2.55,4.4,'to unit a',10,ha='center')
    matrix(axis,toy['unit_a'],3.45,3.35,1.3)
    text(axis,4.10,3.04,'Learning image of a',10,ha='center')
    graphics.arrow(axis,(1.95,3.65),(3.15,1.91),graphics.COLORS[1])
    matrix(axis,toy['unit_b'],3.45,1.17,1.3)
    text(axis,4.10,.87,'Learning image of b',10,ha='center')
    graphics.arrow(axis,(4.95,3.99),(7.17,3.25),BLUE);text(axis,5.72,4.19,'× 2',13,color=BLUE)
    graphics.arrow(axis,(4.95,1.83),(7.17,2.77),graphics.COLORS[1]);text(axis,5.68,1.65,'× 0.5',13,color=graphics.COLORS[1])
    matrix(axis,toy['output'],7.46,2.32,1.62)
    text(axis,8.27,1.98,'Output learning image',11,weight='bold',ha='center')
    text(axis,9.6,3.48,'2 + 4 + 3 − 2',13);text(axis,9.6,2.99,'= 7',19,color=BLUE,weight='bold')
    text(axis,.4,.35,'For a nonlinear ViT, the input-to-unit coefficient is integrated along the reference-to-image path.',10,color=MUTED)
    return figure


def figure18():
    figure,axis=header(TITLES[18],'Linear patch mixing, input-dependent attention, and complete residual accounting.',6.4)
    for xpos,label in [(.35,'1 · Patch embedding'),(4.38,'2 · Attention'),(8.38,'3 · Residual branch')]:text(axis,xpos,4.94,label,12,weight='bold')
    matrix(axis,[[1,2],[3,4]],.45,3.35,1.1)
    graphics.arrow(axis,(1.72,3.9),(2.18,3.9),BLUE)
    matrix(axis,[[1,0],[0,2]],2.35,3.35,1.1)
    text(axis,.45,2.99,'One token channel: weights [1, 0, 0, 0.5]',9)
    text(axis,.45,2.47,r'$h_t=W_E x_{P_t}+b+p_t$',13)
    text(axis,.45,1.88,r'$\Delta h_t=W_E\Delta x_{P_t}$',13,color=BLUE)
    text(axis,.45,1.31,'Local learning image: [1, 0; 0, 2], sum = 3.',9,color=MUTED)
    text(axis,.45,.97,'Bias and position stay in the model.',9,color=MUTED)
    text(axis,4.38,4.06,r'$O=A(x)V(x)$',19,color=BLUE)
    text(axis,4.38,3.23,r'$\mathrm{d}O=(\mathrm{d}A)V+A\,\mathrm{d}V$',13)
    text(axis,4.38,2.55,'Full model: retain both terms.',10.5)
    text(axis,4.38,1.98,'Fixed-attention view: retain A dV.',10.5)
    text(axis,4.38,1.30,'Softmax / LayerNorm / GELU:',9.5,color=MUTED)
    text(axis,4.38,.96,'use derivatives along the path.',9.5,color=MUTED)
    graphics.box(axis,8.49,3.52,1.12,.65,face=graphics.LIGHT);text(axis,9.05,3.85,'Identity',11,ha='center')
    graphics.box(axis,10.08,3.52,1.12,.65,face='#E8F2F7');text(axis,10.64,3.85,'Heads',11,ha='center')
    graphics.arrow(axis,(9.05,3.4),(9.80,2.89));graphics.arrow(axis,(10.64,3.4),(10.08,2.89))
    text(axis,9.94,2.64,'+',22,ha='center')
    text(axis,8.45,2.06,r'$\mathcal{L}_{\mathrm{id}}+\sum_h\mathcal{L}_h=\mathcal{L}_{\mathrm{total}}$',12)
    text(axis,8.45,1.38,'Count each route exactly once.',10,color=MUTED)
    text(axis,.35,.36,'Seed images are a mathematical representation; autograd evaluates the full input path, not each seed in isolation.',9.5,color=MUTED)
    return figure


def heat(axis,value,cmap,limits,records,label):
    raw=np.asarray(value,dtype=float)
    positive=np.maximum(raw,0)
    upper=float(np.percentile(positive,99))
    if upper==0:upper=float(positive.max())
    normalized=np.clip(positive/upper,0,1) if upper else np.zeros_like(positive)
    axis.imshow(normalized,cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
    axis.set_xticks([]);axis.set_yticks([])
    for spine in axis.spines.values():spine.set_visible(False)
    records.append({'panel':label,'shape':list(raw.shape),'scale':'positive_P99_with_max_fallback','upper':upper,'clipped_cells':int((positive>upper).sum()),'raw_sha256':array_digest(np.asarray(value))})


def figure19(photo,arrays,responses,cmap,display):
    blocks=[0,2,4,6,8,11]
    figure,axes=plt.subplots(3,6,figsize=(12,7.2))
    figure.subplots_adjust(left=.18,right=.99,top=.83,bottom=.09,wspace=.08,hspace=.14)
    for column,block in enumerate(blocks):
        raw=arrays[f'b{block:02d}_first_attention_cell']
        response=responses[f'b{block:02d}_attention_output'][1:]
        rms=np.sqrt(np.mean(response.astype(float)**2,axis=1)).reshape(14,14)
        heat(axes[0,column],raw,cmap,None,display,f'19_B{block}_learning_positive')
        heat(axes[1,column],-raw,cmap,None,display,f'19_B{block}_learning_negative')
        heat(axes[2,column],rms,cmap,None,display,f'19_B{block}_token_RMS')
        axes[0,column].set_title(f'Block {block}',fontsize=12,weight='bold',pad=8)
    for row,label in enumerate(['Learning image\npositive','Learning image\nnegative magnitude','Token response\nRMS']):
        axes[row,0].text(-.10,.5,label,transform=axes[row,0].transAxes,ha='right',va='center',fontsize=10.5,weight='bold')
    figure.suptitle(TITLES[19],fontsize=18,weight='bold',y=.978)
    figure.text(.56,.919,'Same input · same class target · attention branch at each depth',ha='center',fontsize=10.5,color=MUTED)
    inset=figure.add_axes([.015,.86,.065,.10]);inset.imshow(photo);inset.axis('off')
    figure.text(.56,.039,'Top two rows: input coordinates (56 × 56)     Bottom: internal token indices (14 × 14)',ha='center',fontsize=10,color=MUTED)
    return figure


def figure20(arrays,cmap,display):
    blocks=[0,5,11];heads=[0,5,11]
    figure,axes=plt.subplots(3,3,figsize=(9.4,8.5))
    figure.subplots_adjust(left=.10,right=.98,top=.87,bottom=.08,wspace=.06,hspace=.08)
    for row,head in enumerate(heads):
        for column,block in enumerate(blocks):
            heat(axes[row,column],arrays[f'b{block:02d}_first_heads_cell'][head],cmap,None,display,f'20_B{block}_H{head}')
            if row==0:axes[row,column].set_title(f'Block {block}',fontsize=12,weight='bold')
            if column==0:axes[row,column].text(-.06,.5,f'H{head}',transform=axes[row,column].transAxes,ha='right',va='center',fontsize=12,weight='bold')
    figure.suptitle(TITLES[20],fontsize=17,weight='bold',y=.97)
    figure.text(.5,.925,'Class-logit component learning images · positive support',fontsize=11,color=MUTED,ha='center')
    figure.text(.5,.028,'Fixed first / middle / last indices · H labels denote separate heads in each block',fontsize=10,color=MUTED,ha='center')
    return figure


def closure_checks(arrays):
    total=arrays['input_cell_map'].astype(float)
    records=[]
    for block in range(12):
        prefix=f'b{block:02d}_'
        head_sum=arrays[prefix+'first_heads_cell'].astype(float).sum(0)
        attention=arrays[prefix+'first_attention_cell'].astype(float)
        identity=arrays[prefix+'first_identity_cell'].astype(float)
        mlp=arrays[prefix+'second_mlp_cell'].astype(float)
        residual=arrays[prefix+'second_residual_cell'].astype(float)
        records.append({'block':block,'head_sum_to_attention_max_abs':float(np.abs(head_sum-attention).max()),
                        'first_cut_to_input_max_abs':float(np.abs(identity+attention-total).max()),
                        'second_cut_to_input_max_abs':float(np.abs(residual+mlp-total).max())})
    threshold=2e-5+.005*float(np.abs(total).max())
    if any(value>threshold for record in records for key,value in record.items() if key!='block'):raise ValueError('Component reconstruction failed')
    return {'atol_plus_rtol_threshold':threshold,'records':records}


def figure21(arrays,checks,display):
    block=5;prefix=f'b{block:02d}_'
    identity=arrays[prefix+'first_identity_cell'];attention=arrays[prefix+'first_attention_cell']
    head_sum=arrays[prefix+'first_heads_cell'].astype(float).sum(0)
    total=arrays['input_cell_map'];reconstructed=identity.astype(float)+head_sum
    values=[identity,head_sum,reconstructed,total]
    limit=max(float(np.abs(value).max()) for value in values)
    figure,axis=header(TITLES[21],'One fixed block and one class target. Add signed raw values before coloring.',4.6)
    for xpos,value,label in zip([.35,3.30,6.25,9.20],values,['Residual path','All 12 heads','Their raw sum','Total input IG']):
        scaled=np.sign(value)*np.log1p(9*np.abs(value)/limit)/np.log(10)
        axis.imshow(scaled,extent=(xpos,xpos+2.28,1.0,3.28),cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest',aspect='auto')
        text(axis,xpos+1.14,.76,label,10.5,weight='bold',ha='center')
        display.append({'panel':'21_'+label,'scale':'shared_signed_log_kappa9','upper':limit,'raw_sha256':array_digest(value)})
    for xpos,label in [(2.82,'+'),(5.77,'='),(8.72,'≈')]:text(axis,xpos,2.12,label,21,color=BLUE,ha='center')
    error=float(np.abs(reconstructed-total).max())
    text(axis,.35,.24,f'Block 5 · maximum raw cell reconstruction error: {error:.2e} · shared signed display scale',10,color=MUTED)
    return figure


EXPLANATIONS={
15:'把网络里的一个数和一张输入位置图配起来。上面是内部响应的变化量，下面说明这次变化来自输入的哪几个位置。四张图始终用同一套位置坐标。这张采用可手算的线性示例，不是 ViT 实测结果。',
16:'先用原图减去参考图，得到这次需要解释的输入变化。然后拆成四张小图，每张只保留一个位置，其余位置为零；四张相加正好恢复输入差值。真实 RGB 输入对应每个像素的三个通道。实现用自动微分计算这些位置的贡献，不必真的存下十几万张稀疏图。',
17:'沿用上一图的输入差值。单元 a 的图是 [1,2;0,0]，单元 b 的图是 [0,0;6,-4]。输出用 2a+0.5b，因此输出图是 [2,4;3,-2]，总和为 7。你可以看见数值被重新加权，但原来的位置始终没丢。ViT 的系数还随输入变化，所以实际方法要沿参考到原图的路径累计。',
18:'Patch embedding 可以先按线性连接理解；位置编码和偏置仍保留在模型里。到了 attention，输入既会改变注意力权重 A，也会改变 Value，完整方法需要两部分导数。最后 residual 旁路也要算，才能把贡献账算完整。旧的固定 attention 图只研究其中一个条件视角。',
19:'这张对应原论文 Figure 13 的上下对照读法。上面两行是同一类别目标的 attention 分支 learning image：正贡献和负贡献分开看；下面是同一分支输出在各个 token 上的响应强度。前者回答“输入哪里贡献了多少”，后者回答“内部哪个 token 的向量较大”。它们使用不同坐标，不能要求长得一样。每列换一个 block，始终保持同一张输入和同一个类别目标。',
20:'固定同一输入和最终类别，比较不同 head 分到的输入贡献。每个小图是一个 head 的成分 learning image，这比换不同 head 自身的响应目标更适合看共同任务下的差异。图中保留 H0/H5/H11、B0/B5/B11 这些固定位置，完整逐 block 12-head 图另附。颜色逐面板缩放，亮度不用于排名。',
21:'同一个 block 的 residual 旁路图，加上全部 12 个 head 的图，会恢复总输入贡献图。这里先对原始正负数求和，再着色；不是把几张颜色图片叠起来。它展示 learning image 怎样组合成更大组件，同时能检查是否漏掉某条路径。'
}

CAPTIONS_EN={
15:'One internal response is paired with one map in input coordinates. The illustrated linear example uses input difference [1,2;3,4], unit changes 3 and 2, and output change 7. All four maps retain the same input locations. This is an analytic example, not a measured ViT.',
16:'Input-position initialization. The reference is [1,1;1,1] and the input is [2,3;4,5]. Their difference is split into four coordinate images whose sum restores the difference. This is an equivalent mathematical representation for path attribution; the implementation does not forward each sparse seed independently through the nonlinear model.',
17:'Generation in an analytic linear network. For a=x1+x2, b=2x3−x4 and f=2a+0.5b, the learning images of a and b are [1,2;0,0] and [0,0;6,−4]. Their weighted combination is [2,4;3,−2], summing to the output change 7. For nonlinear ViT targets the corresponding coefficients are integrated along the input path.',
18:'ViT operator adaptation. A patch channel with weights [1,0,0,0.5] produces the local contribution map [1,0;0,2] from input differences [1,2;3,4]. Position and bias terms remain in the forward model. Attention requires both dA and dV terms on the full path; the fixed-attention condition suppresses dA. Head and identity components jointly form a complete cut. The drawing is schematic.',
19:'Measured class-conditioned attention-branch learning images and token responses at blocks 0,2,4,6,8,11. The input is the first frozen Y2 cohort sample, ILSVRC2012_val_00007922.JPEG, with clean predicted-class logit target 235. The first two rows show positive and negative-magnitude input-cell contributions, respectively, from sealed 56×56 component arrays. The last row shows RMS across the 768 channels of the corresponding attention branch output at each of the 196 patch tokens; it is obtained by one retained CPU forward with matching target/logit. Input coordinates and internal token indices are different coordinate systems. Each nonnegative panel uses its own P99 scale and nearest display. No correspondence strength or model-wide trend is inferred from this single input.',
20:'Class-logit component learning images for heads 0,5,11 at blocks 0,5,11, on the same image and target as Figure 19. Positive signed input-cell contributions are shown with per-panel P99 scaling. These fixed index subsets do not rank heads by appearance or performance; every head and both signs are included in the attached 12 block pages. Head numbers in different blocks identify different parameter sets.',
21:'Component reconstruction at block 5 for the same class target. The raw residual map plus the sum of all twelve raw head maps reconstructs the total input-cell IG map to numerical precision. Addition is performed before coloring. All panels share sign(m) log1p(9 abs(m)/L)/log(10), where L is the maximum absolute raw value across the four panels. This checks complete-cut accounting, not a new fidelity or causal test.'
}


def docs(figures,metadata):
    lines=['# NNS-ViT 与 learning image：Figure 15–21','',
        'NNS-ViT 是把 NNS 的内部单元可视化思路用于 Vision Transformer 的方法体系；它给选定的内部响应或组件配一张保持输入坐标的贡献图。learning image 就是这张图。', '',
        '本组 Figure 15–18 用数学例子讲清原理；Figure 19–21 使用完整类别目标下的真实 NNS-v2 组件图。原 Figure 1–14 在 [R8 画廊](../nns_vit_unified_paper_figures_r8_20260916/gallery.html)。','']
    index=['# Figure 15–21：新增 learning image 图组','','[画廊](gallery.html) · [说人话说明](figure_analysis.md) · [原 Figure 1–14](../nns_vit_unified_paper_figures_r8_20260916/gallery.html)','','| Figure | 内容 | PNG | PDF | SVG |','|---:|---|---|---|---|']
    cards=[]
    for record in figures:
        number=record['number'];links={item['format']:str(Path(item['path']).relative_to(ROOT.relative_to(REPO))) for item in record['outputs']}
        index.append(f'| {number} | {record["title"]} | '+' | '.join(f'[{fmt}]({links[fmt]})' for fmt in ['PNG','PDF','SVG'])+' |')
        lines.extend([f'## Image #{number}: {record["title"]}','',EXPLANATIONS[number],'',f'![Figure {number}]({links["PNG"]})',''])
        cards.append(f'<article id="figure-{number}"><h2>Figure {number} · {html.escape(record["title"])}</h2><p>{EXPLANATIONS[number]}</p><a href="{links["PNG"]}"><img src="{links["PNG"]}" loading="lazy" alt="Figure {number}"></a><p>'+ ' · '.join(f'<a href="{links[fmt]}">{fmt}</a>' for fmt in ['PNG','PDF','SVG'])+'</p></article>')
    lines.extend(['## 实测图的来源','',f'Y2 冻结队列第一张有效样本：{metadata["source_object_id"]}，selection_order=0，类别目标={metadata["class_id"]}。',
        '归因地图来自已有封存数据；对应 token 响应由同一模型的一次 CPU 前向取得，类别与目标 logit 已核对。没有重新计算归因或干预。',
        '图 19–20 的非负显示分别按 P99 缩放，保持原始格子；图 21 使用统一 signed-log 色阶。展示结果不用于新的总体统计。',''])
    (ROOT/'figure_analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    (ROOT/'figure_index.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    for language,captions in [('zh',EXPLANATIONS),('en',CAPTIONS_EN)]:
        content=[f'# Learning image figures 15–21 · {language}','']
        for number in range(15,22):content.extend([f'## Figure {number}','',captions[number],''])
        if language=='zh':content.extend(['## 共同显示与来源说明','',
            'Figure 15–18 是解析示意，不是实测网络结果。Figure 19–21 使用队列首个冻结样本 ILSVRC2012_val_00007922.JPEG、clean predicted-class logit 235 的已有 NNS-v2 组件地图。',
            '56×56 map 每个 cell 对应 4×4 输入像素。图 19 下排 RMS 来自原生 attention 分支输出的 196 个 patch tokens，跨 768 通道计算平方平均后开根号；并不等同于类贡献或 CNN 的 feature map。',
            '图 19–20 与附带全 head 图逐面板正部 P99 缩放（P99=0 时用最大值），nearest 显示，负部先取幅度；图 21 为原始带符号值相加后，用共同 max-abs 上限的 signed-log(kappa=9) 显示。',''])
        (ROOT/f'captions_{language}.md').write_text('\n'.join(content),encoding='utf-8')
    gallery='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NNS-ViT learning image</title><style>body{max-width:1500px;margin:auto;padding:24px;background:#eef3f6;color:#172B3A;font:16px/1.6 system-ui}article{padding:24px;background:white;margin:22px 0}img{width:100%}a{color:#1875A5}h2{font-size:22px}</style><h1>NNS-ViT · Learning image · Figure 15–21</h1><p><a href="../nns_vit_unified_paper_figures_r8_20260916/gallery.html">原 Figure 1–14</a> · <a href="figure_analysis.md">逐图说明</a></p>'+''.join(cards)
    gallery+='<h2>每个 block 的全部 12 个 head</h2>'
    for path in sorted((ROOT/'all_heads').glob('*.png')):
        relative=path.relative_to(ROOT).as_posix();gallery+=f'<details><summary>{path.stem}</summary><img loading="lazy" src="{relative}" alt="All 12 heads"></details>'
    (ROOT/'gallery.html').write_text(gallery+'</html>',encoding='utf-8')
    (ROOT/'README.md').write_text('# NNS-ViT learning image 补充图\n\n[完整画廊](gallery.html) · [图表索引](figure_index.md) · [逐图说明](figure_analysis.md)\n\n新增 Figure 15–21，每张 PNG/PDF/SVG，另附 12 页全 head 正负图。已有 Figure 1–14 保持原样，新画廊可导航到原图包。\n\n[来源阅读笔记](../../docs/learning_image_source_reading_20260917.md)\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');parser.add_argument('--reuse-forward',action='store_true');args=parser.parse_args()
    if ROOT.exists() and ((ROOT/'terminal_receipt.json').exists() or not args.draft):raise ValueError('Only unsealed drafts can be updated')
    ROOT.mkdir(parents=True,exist_ok=True)
    before=common.inventory(BASE)
    for name in ['terminal_receipt.json','validation_report.json']:
        value=read_signed(BASE/name)
        if not value['status'].startswith('PASS'):raise ValueError('R8 is not passed')
    for path in [Path(__file__),Path(__file__).with_name('validate_learning_image_figures_15_21.py'),DESIGN,Path(graphics.__file__),Path(graphics.base.__file__),Path(common.__file__)]:register(path,'implementation_or_design')
    register(REPO/'src/vitnns/modeling.py','model_loader')
    register(REPO/'src/vitnns/stage_y_y1.py','component_map_definition')
    references=[REPO.parent/'论文1/参考文章.pdf',*sorted((REPO.parent/'论文1/论文架构').glob('*'))]
    for path in references:
        if path.is_file() and not path.name.startswith('~$'):register(path,'historical_reading_reference')
    config=common.read(REPO/'config/nns_vit_unified_paper_figures_r4.json');register(REPO/'config/nns_vit_unified_paper_figures_r4.json')
    cmap=graphics.base.configure(config);graphics.base.OUTPUT=ROOT;plt.rcParams['svg.hashsalt']='learning-images-15-21'
    photo,arrays,metadata=load_existing()
    if args.reuse_forward:
        old=common.read(ROOT/'numeric_display_data.json')
        if not common.verify(old):raise ValueError('Cached forward metadata digest mismatch')
        metadata['cpu_forward']=old['source']['cpu_forward'];register(MODEL,'existing_model_for_single_CPU_forward')
        responses={f'b{block:02d}_attention_output':np.load(ROOT/'source_arrays'/f'b{block:02d}_attention_output.npy',allow_pickle=False) for block in range(12)}
    else:responses=read_responses(photo,metadata)
    print('Source maps and CPU forward verified',flush=True)
    checks=closure_checks(arrays);toy=toy_values();display=[];figures=[];layout=[]
    common.write(ROOT/'numeric_display_data.json',common.signed({'toy':toy,'source':metadata,'closure':checks,'display':display}))
    for number in range(15,22):
        if number==18:figure=figure18()
        elif number<18:figure=globals()[f'figure{number}'](toy)
        elif number==19:figure=figure19(photo,arrays,responses,cmap,display)
        elif number==20:figure=figure20(arrays,cmap,display)
        else:figure=figure21(arrays,checks,display)
        figure.canvas.draw();renderer=figure.canvas.get_renderer()
        outside=[artist.get_text() for axis in figure.axes for artist in axis.texts if not all(figure.bbox.contains(*point) for point in artist.get_window_extent(renderer).get_points())]
        if outside:raise ValueError(f'Figure {number} text out of bounds: {outside}')
        outputs=graphics.base.save_figure(figure,number,SLUGS[number])
        figures.append({'number':number,'title':TITLES[number],'outputs':outputs,'kind':'analytic_explanation' if number<19 else 'measured_component_visualization'})
        layout.append({'number':number,'text_outside_canvas':outside});print(f'Figure {number} rendered',flush=True)
    (ROOT/'all_heads').mkdir(exist_ok=True)
    for block in range(12):
        figure,axes=plt.subplots(2,12,figsize=(18,3.4));figure.subplots_adjust(left=.035,right=.99,top=.82,bottom=.08,wspace=.06,hspace=.10)
        for head in range(12):
            for row,sign in enumerate([1,-1]):heat(axes[row,head],sign*arrays[f'b{block:02d}_first_heads_cell'][head],cmap,None,display,f'all_B{block}_H{head}_sign{sign}')
            axes[0,head].set_title(f'H{head}',fontsize=9)
        for row,label in enumerate(['+','−']):axes[row,0].text(-.12,.5,label,transform=axes[row,0].transAxes,ha='right',va='center',fontsize=12)
        figure.suptitle(f'Block {block} · all heads · fixed class {metadata["class_id"]}',fontsize=14)
        figure.savefig(ROOT/'all_heads'/f'block_{block:02d}.png',dpi=200,bbox_inches='tight');plt.close(figure)
    docs(figures,metadata)
    common.write(ROOT/'numeric_display_data.json',common.signed({'toy':toy,'source':metadata,'closure':checks,'display':display}))
    common.write(ROOT/'layout_audit.json',common.signed({'status':'PASS','records':layout}))
    if before!=common.inventory(BASE):raise ValueError('R8 changed')
    common.write(ROOT/'predecessor_snapshot.json',common.signed({'root':str(BASE.relative_to(REPO)),'files':before,'files_digest':common.canonical(before)}))
    sources=[SOURCE_RECORDS[key] for key in sorted(SOURCE_RECORDS)]
    common.write(ROOT/'render_manifest.json',common.signed({'status':'PENDING_VISUAL_REVIEW','figure_numbers':list(range(15,22)),'figures':figures,'source_files':sources,'source_files_digest':common.canonical(sources),'base_figures_1_14_modified':False}))
    common.write(ROOT/'source_arrays_manifest.json',common.signed({'files':common.inventory(ROOT/'source_arrays')}))
    common.write(ROOT/'runtime.json',common.signed({'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'device':'cpu','new_forward_calls_this_render':0 if args.reuse_forward else 1,'retained_forward_capture_count':1,'new_attribution_calls':0,'sys_executable':str(Path(sys.executable).resolve())}))
    common.write(ROOT/'all_heads_manifest.json',common.signed({'files':common.inventory(ROOT/'all_heads')}))


if __name__=='__main__':main()
