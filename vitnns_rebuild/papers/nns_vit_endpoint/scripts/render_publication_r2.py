#!/usr/bin/env python3
"""Expanded paper figures with original inputs, blue learning images and conceptual diagrams."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['SOURCE_DATE_EPOCH']='1789862400'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle,Rectangle

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'papers/nns_vit_endpoint/publication_r2_20260920'
BLUE='#195D9C';TEAL='#218D8A';ORANGE='#D08332';INK='#18324B';GRAY='#617185'
CMAP=LinearSegmentedColormap.from_list('nns_blue',[(0,'#07077F'),(.2,'#0019C6'),(.42,'#00AEE8'),(.65,'#FFF000'),(.82,'#FF6A00'),(1,'#D90000')])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':'#C4CDD6','axes.spines.top':False,'axes.spines.right':False,'xtick.color':GRAY,'ytick.color':GRAY,'grid.color':'#E5ECF2','svg.fonttype':'none','svg.hashsalt':'endpoint-publication-r2','pdf.fonttype':42})
FIGS=[];DISPLAYS=[];SOURCES={}
NUMBERING={12:13,13:14,14:15,15:17,16:12,17:16}
SAMPLE_LABELS={1:'Langur',4:'Mantis',7:'Offshore platform',13:'Stethoscope',14:'Sloth bear',15:'Little blue heron'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):
    SOURCES[str(p.relative_to(REPO))]=sha(p);return p
def load(p):return json.loads(bind(p).read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+chr(10))
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rows(p):return list(csv.DictReader(bind(p).open()))
def positive(raw):
    a=np.asarray(raw,dtype=float).squeeze()
    return np.maximum(a.sum(0) if a.ndim==3 else a,0)
def show(ax,value,label='',kind='learning',source='',interpolation='nearest'):
    a=np.asarray(value)
    if kind=='photo':ax.imshow(a);ax.set_title(label,pad=7);ax.axis('off');return
    if kind=='signed':
        v=a.astype(float).squeeze();v=v.sum(0) if v.ndim==3 else v
        lim=float(np.quantile(np.abs(v),.99)) or float(np.abs(v).max()) or 1
        shown=np.clip(v/lim,-1,1);ax.imshow(shown,cmap='RdBu_r',vmin=-1,vmax=1,interpolation=interpolation)
    else:
        v=np.abs(a).sum(0) if kind=='absolute' and a.ndim==3 else positive(a)
        # Positive-value P99 avoids a zero threshold for sparse coordinate maps.
        nz=v[v>0];lim=float(np.quantile(nz,.99)) if nz.size else 1
        gamma=.5 if kind in ('learning','absolute') else 1.0
        shown=np.clip(v/lim,0,1)**gamma;ax.imshow(shown,cmap=CMAP,vmin=0,vmax=1,interpolation=interpolation)
    ax.set_title(label,pad=7,fontsize=10);ax.axis('off')
    DISPLAYS.append(dict(figure=len(FIGS)+1,label=label,kind=kind,source=source,raw_sha256=arrsha(a),shape=list(a.shape),limit=lim,gamma=.5 if kind in ('learning','absolute') else 1.0,clipped=int((np.abs(v)>lim).sum()),display_sha256=arrsha(shown),interpolation=interpolation))
def title(fig,text,sub=''):
    fig.suptitle(text,x=.035,y=.99,ha='left',fontsize=17,weight='bold')
    if sub:fig.text(.035,.94,sub,fontsize=10,color=GRAY,va='top')
def finish(fig,n,slug,zh,en,role='qualitative'):
    original_number=n;n=NUMBERING.get(n,n)
    files={}
    for ext in ('png','pdf','svg'):
        p=ROOT/f'figures/{ext}/fig_{n:02d}_{slug}.{ext}'
        fig.savefig(p,dpi=220 if ext=='png' else 160,bbox_inches='tight',pad_inches=.14,facecolor='white')
        if ext=='svg':p.write_text(chr(10).join(line.rstrip() for line in p.read_text().splitlines())+chr(10))
        files[ext]=dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
    FIGS.append(dict(number=n,original_number=original_number,slug=slug,title=fig._suptitle.get_text(),caption_zh=zh,caption_en=en,role=role,files=files));plt.close(fig);print('figure',n,slug,flush=True)
def canvas(size=(12,6)):
    fig=plt.figure(figsize=size);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');return fig,ax
def box(ax,x,y,w,h,text,color=BLUE,fill='#F0F6FB',fontsize=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.012',facecolor=fill,edgecolor=color,lw=1.4))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,color=INK)
def arrow(ax,a,b,color=BLUE,style='-',lw=1.8):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=color,lw=lw,linestyle=style,connectionstyle='arc3'))
def inset(fig,x,y,w,h,value,label='',kind='learning',source=''):
    ax=fig.add_axes([x,y,w,h]);show(ax,value,label,kind,source);return ax
def matrix(ax,x,y,w,h,a,label='',color=BLUE):
    a=np.asarray(a);nr,nc=a.shape
    for i in range(nr):
        for j in range(nc):
            xx=x+j*w/nc;yy=y+(nr-1-i)*h/nr
            ax.add_patch(Rectangle((xx,yy),w/nc,h/nr,facecolor='#EAF3FB' if a[i,j]!=0 else 'white',edgecolor=color,lw=1))
            ax.text(xx+w/nc/2,yy+h/nr/2,f'{a[i,j]:g}',ha='center',va='center',fontsize=12)
    if label:ax.text(x+w/2,y+h+.018,label,ha='center',fontsize=11,color=color)
def e3(model,b):
    p=REPO/f'reports/nns_vit_endpoint_e3_closure_r1_20260918/formal_results/attempts/{model}__014/attempt-01.json'
    d=load(p)['result'];r=next(r for r in d['artifacts'] if r.get('block')==b)
    q=bind(REPO/r['path']);assert sha(q)==r['sha256'];a=np.load(q,mmap_mode='r')
    return {k:np.array(a[i]) for i,k in enumerate(r['component_order'])}
def patchmax(raw):
    p=positive(raw).reshape(14,16,14,16).max((1,3)).astype('float32')
    return np.array(Image.fromarray(p).resize((224,224),Image.Resampling.BILINEAR))

def main():
    receipt=load(ROOT/'evidence/compute_receipt.json');assert receipt['task_count']>=6
    data={};meta={}
    for r in receipt['tasks']:
        p=bind(ROOT/'assets'/f"{r['task_id']}.npz");assert sha(p)==r['array_sha256'];data[r['task_id']]=np.load(p);meta[r['task_id']]=r
    bear=data['deit_s16__014'];photo=bear['input_rgb'];parts=e3('deit_s16',5)
    # 1. Conceptual overview: ordinary forward computation above and input-coordinate scanner below.
    fig,ax=canvas((12,6.6));title(fig,'Seeing a Vision Transformer through learning images')
    inset(fig,.025,.36,.16,.30,photo,'Input image','photo')
    box(ax,.25,.62,.17,.16,'Patch tokens');box(ax,.48,.62,.20,.16,'Transformer blocks');box(ax,.75,.62,.19,.16,'Class prediction')
    for x0,x1 in [(.19,.24),(.43,.47),(.69,.74)]:arrow(ax,(x0,.7),(x1,.7))
    ax.text(.26,.84,'Model computation',fontsize=12,color=BLUE,weight='bold')
    ax.text(.25,.39,'NNS-ViT: retain the image coordinates of each target',fontsize=12,color=TEAL,weight='bold')
    for i,(name,key) in enumerate([('Patch embedding','patch.energy.endpoint'),('Intermediate unit','b5.energy.endpoint'),('Class output','class_endpoint')]):
        x=.25+i*.245;inset(fig,x,.08,.19,.24,bear[key],name,source='deit_s16__014/'+key);arrow(ax,(x+.095,.61),(x+.095,.36),TEAL,'--')
    finish(fig,1,'concept_overview','同一输入经过正常前向计算得到预测；NNS-ViT 为内部目标与输出目标提供保留输入位置的学习图像。虚线表示目标与学习图像的对应关系。','The same input produces a prediction through the ordinary forward computation. NNS-ViT associates internal and output targets with learning images in the original input coordinates. Dashed lines link targets to their images.','concept')
    # 2. Explicit four-coordinate example, values and learning images paired.
    x=np.array([1.,2.,3.,4.]);W=np.array([[1.,.5,0.,0.],[0.,0.,.5,1.]]);h=W@x;v=np.array([1.,.5]);E=x*(v@W)
    fig,ax=canvas((12,7));title(fig,'A neuron and its learning image carry different information')
    ax.text(.025,.75,'Forward\nvalues',fontsize=13,color=BLUE);ax.text(.025,.27,'Learning\nimages',fontsize=13,color=TEAL)
    positions=[[(.25,.8),(.25,.67),(.25,.54),(.25,.41)],[(.56,.73),(.56,.49)],[(.86,.61)]]
    for (xx,yy),val in zip(positions[0],x):ax.add_patch(Circle((xx,yy),.035,fc='white',ec=BLUE,lw=2));ax.text(xx,yy,f'{val:g}',ha='center',va='center')
    for (xx,yy),val in zip(positions[1],h):ax.add_patch(Circle((xx,yy),.04,fc='#EAF3FB',ec=TEAL,lw=2));ax.text(xx,yy,f'{val:g}',ha='center',va='center')
    ax.add_patch(Circle(positions[2][0],.045,fc='#FFF4E6',ec=ORANGE,lw=2));ax.text(.86,.61,f'{v@h:g}',ha='center',va='center')
    for i in range(4):
        for j in range(2):
            if W[j,i]:arrow(ax,(.29,positions[0][i][1]),(.515,positions[1][j][1]),GRAY);ax.text(.4,(positions[0][i][1]+positions[1][j][1])/2,f'{W[j,i]:g}',fontsize=9)
    for j in range(2):
        arrow(ax,(.60,positions[1][j][1]),(.81,.61),GRAY)
        ax.text(.70,(positions[1][j][1]+.61)/2+.025,f'{v[j]:g}',fontsize=10)
    for i in range(4):
        a=np.zeros(4);a[i]=x[i];matrix(ax,.17+i*.115,.12,.085,.14,a.reshape(2,2))
    matrix(ax,.66,.24,.09,.10,(x*W[0]).reshape(2,2),'Unit 1',TEAL)
    matrix(ax,.66,.065,.09,.10,(x*W[1]).reshape(2,2),'Unit 2',TEAL)
    matrix(ax,.84,.13,.10,.14,E.reshape(2,2),'Output',ORANGE)
    arrow(ax,(.61,.19),(.65,.29));arrow(ax,(.61,.19),(.65,.115))
    arrow(ax,(.76,.29),(.83,.23));arrow(ax,(.76,.115),(.83,.17))
    ax.text(.20,.015,'Each hidden unit has its own learning image; both contribute to the output image.',ha='left',fontsize=10)
    finish(fig,2,'neuron_learning_image','四坐标线性示例：上方为神经元值，下方为相同计算对应的输入坐标学习图像。此例用于解释传播规则，数值由图中权重直接计算。','A four-coordinate linear example. The upper diagram shows neuron values; the lower matrices preserve each input coordinate through the same computation. Values follow directly from the displayed weights.','concept')
    # 3. Initialization.
    fig,ax=canvas((12,4.6));title(fig,'Initialization: preserve both value and position')
    matrix(ax,.045,.27,.16,.38,x.reshape(2,2),'Input x');ax.text(.245,.44,'=',fontsize=25)
    for i in range(4):
        a=np.zeros(4);a[i]=x[i];xx=.31+i*.17;matrix(ax,xx,.27,.12,.38,a.reshape(2,2),f'Coordinate {i+1}')
        if i<3:ax.text(xx+.135,.44,'+',fontsize=22)
    ax.text(.30,.08,r'$\mathcal{L}(x;x)=\mathrm{diag}(x)$',fontsize=17);ax.text(.60,.08,'Each coordinate starts in its own image location.',fontsize=11)
    finish(fig,3,'coordinate_initialization','输入值初始化为对角 learning-image 矩阵；拆分后每个坐标的位置保持不变。RGB 输入对每个通道分别进行相同初始化。','Input values initialize a diagonal learning-image matrix. Each split coordinate retains its location; RGB channels follow the same initialization independently.','concept')
    # 4. Generation and chain rule.
    fig,ax=canvas((12,5.2));title(fig,'Generation: combine learning images with local derivatives')
    a=x*W[0];b=x*W[1]
    matrix(ax,.055,.35,.17,.34,a.reshape(2,2),'Learning image of h1',BLUE);ax.text(.25,.51,'× 1  +',fontsize=18)
    matrix(ax,.36,.35,.17,.34,b.reshape(2,2),'Learning image of h2',TEAL);ax.text(.55,.51,'× 0.5  =',fontsize=18)
    matrix(ax,.76,.35,.17,.34,E.reshape(2,2),'Learning image of s',ORANGE)
    ax.text(.06,.16,r'$\mathcal{L}(s;x)=\sum_k \frac{\partial s}{\partial h_k}\mathcal{L}(h_k;x)$',fontsize=17)
    ax.text(.57,.16,r'$E_s(x)=x\odot\nabla_x s(x)$',fontsize=18,color=BLUE)
    finish(fig,4,'local_derivative_generation','局部导数作为组合系数，学习图像逐层传播后与输入乘目标梯度一致。这里展示已定义线性示例的完整数值运算。','Local derivatives weight the incoming learning images. Layerwise propagation equals input times target gradient. The figure shows the complete numerical operation for the defined linear example.','concept')
    # 5. Patch embedding as a spatially restricted linear map.
    fig,ax=canvas((12,5.8));title(fig,'From pixels to tokens without losing image coordinates')
    ia=inset(fig,.035,.25,.25,.49,photo,'Image divided into 16 × 16 patches','photo')
    for k in range(1,14):ia.axvline(k*16-.5,color='white',lw=.35,alpha=.65);ia.axhline(k*16-.5,color='white',lw=.35,alpha=.65)
    box(ax,.36,.42,.22,.22,'Flatten one patch\n\nLinear projection W',fontsize=12);arrow(ax,(.29,.53),(.35,.53))
    box(ax,.67,.59,.27,.13,'Token coordinate z(t, c)',TEAL);arrow(ax,(.59,.54),(.68,.64))
    seed=np.zeros((224,224));seed[96:112,96:112]=1;inset(fig,.70,.15,.19,.31,seed,'Input-coordinate support')
    arrow(ax,(.8,.58),(.8,.47),TEAL,'--');ax.text(.035,.075,r'$z_{t,c}=w_c^\top x_t+b_c\qquad E_{t,c}=x\odot w_{t,c}$',fontsize=15)
    finish(fig,5,'patch_embedding','Patch embedding 可写为局部线性映射。其 token-channel 坐标的学习图像保留在原输入网格；示意支持区域对应一个 patch，偏置不形成图像坐标项。','Patch embedding is a local linear map. A token-channel learning image retains the original input grid; the illustrated support is one patch. Biases have no independent image-coordinate term.','concept')
    # 6. Attention graph rather than a text list.
    fig,ax=canvas((12,6));title(fig,'Attention separates routing from value transmission')
    box(ax,.035,.40,.13,.18,'Input X')
    for yy,lab in [(.72,'Query Q'),(.44,'Key K'),(.16,'Value V')]:box(ax,.25,yy,.16,.13,lab);arrow(ax,(.17,.49),(.24,yy+.065),GRAY)
    box(ax,.50,.60,.19,.16,'softmax(QKᵀ / √d)',TEAL,fontsize=11);arrow(ax,(.42,.78),(.5,.71));arrow(ax,(.42,.5),(.5,.64))
    box(ax,.76,.37,.17,.17,'O = A V',ORANGE);arrow(ax,(.60,.59),(.78,.55),TEAL);arrow(ax,(.42,.22),(.78,.37),BLUE)
    ax.text(.45,.055,r'$\mathrm{d}O=(\mathrm{d}A)V+A\mathrm{d}V$',fontsize=16)
    ax.text(.58,.52,'Q/K: routing',color=TEAL,fontsize=11)
    ax.text(.50,.18,'V: value transmission',color=BLUE,fontsize=11)
    finish(fig,6,'attention_differential','Q/K 的变化通过 attention weights 改变 routing；V 的变化通过当前权重传播。两条微分路径相加重构完整 attention 微分，head 下游 cotangent 包含 output projection。','Q/K changes alter routing through attention weights; V changes propagate through the current weights. The two differential paths sum to the full attention differential, with output projection included in the downstream cotangent.','concept')
    # 7. Residual, actual maps underneath.
    fig,ax=canvas((12,6.2));title(fig,'Residual connections retain and transform input dependence')
    box(ax,.06,.66,.15,.13,'u');box(ax,.36,.75,.23,.12,'Branch f(u)',ORANGE);box(ax,.77,.66,.16,.13,'y = u + f(u)',TEAL,fontsize=11)
    arrow(ax,(.22,.73),(.35,.8));arrow(ax,(.60,.80),(.76,.73));arrow(ax,(.22,.68),(.76,.68),BLUE)
    ax.text(.38,.61,'Identity edge uses the output cotangent gᵧ',fontsize=11,color=GRAY)
    for i,(lab,key) in enumerate([('Original image',None),('Identity','identity'),('Attention branch','attention'),('Residual output','L_F')]):
        inset(fig,.04+i*.245,.13,.20,.34,photo if key is None else parts[key],lab,'photo' if key is None else 'learning',key or '')
    ax.text(.25,.04,r'$E_{\mathrm{output}}=E_{\mathrm{identity}}+E_{\mathrm{branch}}$',fontsize=16)
    finish(fig,7,'residual_principle','上方为 residual 的两条真实边，下方为同一懒熊输入和类别目标的两路径学习图像。等式对 signed maps 成立，蓝底正部仅供空间阅读。','The residual has two distinct computational edges. Below are their learning images for the same sloth-bear input and class target. Additivity holds for signed maps; blue positive views show spatial structure.','concept')
    # 8. Internal target types, larger panels with clear photograph.
    fig,axes=plt.subplots(2,4,figsize=(12,7.5));fig.subplots_adjust(top=.85,bottom=.08,wspace=.12,hspace=.24)
    title(fig,'Different targets reveal different parts of the same input')
    entries=[('Input',None),('Patch embedding','patch.energy.endpoint'),('Block 0 energy','b0.energy.endpoint'),('Block 5 energy','b5.energy.endpoint'),('Block 11 energy','b11.energy.endpoint'),('Attention projection','b5.attn.energy.endpoint'),('MLP output','b5.fc2.energy.endpoint'),('Class logit','class_endpoint')]
    for ax,(lab,key) in zip(axes.flat,entries):show(ax,photo if key is None else bear[key],lab,'photo' if key is None else 'learning',key or '')
    finish(fig,8,'internal_targets','同一图像的不同内部与类别目标。Energy targets 使用前向空间方差最大的通道；这与单个 token-channel 目标和最终类别目标区分。','Different internal and class targets for one image. Energy targets use the channel of highest forward spatial variance and are distinct from individual token-channel and final class targets.')
    # 9–10. Twelve layers, paired real forward maps.
    for number,model in [(9,'deit_s16'),(10,'vit_b16')]:
        d=data[model+'__014'];mm={r['name']:r['channel'] for r in meta[model+'__014']['selected_channels']}
        fig,axes=plt.subplots(4,7,figsize=(15,9));fig.subplots_adjust(top=.87,bottom=.06,wspace=.05,hspace=.30)
        title(fig,('DeiT-S/16' if model=='deit_s16' else 'ViT-B/16')+': learning images through all twelve blocks','Each column pairs one energy-target learning image with its actual forward channel map')
        for phase in range(2):
            show(axes[phase*2,0],d['input_rgb'],'Original input','photo');axes[phase*2+1,0].axis('off');axes[phase*2+1,0].text(.1,.55,'Top: learning image\nBottom: feature map',fontsize=10)
            for j in range(6):
                b=phase*6+j;name=f'b{b}';show(axes[phase*2,j+1],d[name+'.energy.endpoint'],f'Block {b} · c{mm[name]}',source=model+'/'+name)
                show(axes[phase*2+1,j+1],d[name+'.energy.feature'],'Forward feature',kind='feature',source=model+'/'+name)
        finish(fig,number,model+'_layerwise','同一懒熊输入跨全部 12 层的学习图像及真实前向特征。每层通道按前向空间方差选择；上排为该通道 energy scalar 的输入图，下排为该通道原生 14×14 activation 的正部。各面板独立缩放。','Learning images and actual forward features across all 12 blocks for the same sloth-bear input. Each block selects a channel by forward spatial variance. The upper row shows its energy-target input map, and the lower row the positive native 14×14 activation. Panels are independently scaled.')
    # 11. Module chain paired features.
    fig,axes=plt.subplots(6,7,figsize=(15,13));fig.subplots_adjust(top=.91,bottom=.04,wspace=.06,hspace=.32)
    title(fig,'Inside a block: normalization, attention and the MLP')
    for i,b in enumerate((0,5,11)):
        show(axes[2*i,0],photo,f'Block {b} · input','photo');axes[2*i+1,0].axis('off');axes[2*i+1,0].text(.05,.5,'Learning image\n↕\nForward feature',fontsize=10)
        for j,key in enumerate(('norm1','attn','norm2','fc1','gelu','fc2')):
            name=f'b{b}.{key}';show(axes[2*i,j+1],bear[name+'.energy.endpoint'],key,source=name);show(axes[2*i+1,j+1],bear[name+'.energy.feature'],'Feature',kind='feature',source=name)
    finish(fig,11,'module_learning_images','DeiT-S blocks 0、5、11 内部模块的学习图像和真实前向 activation。每个模块独立选择高空间方差通道并定义 energy target；LayerNorm、attention projection、MLP 的 fc1/GELU/fc2 清楚对应。','Learning images and actual forward activations inside DeiT-S blocks 0, 5 and 11. Each module independently selects a high-spatial-variance channel and defines an energy target, covering LayerNorm, attention projection and MLP fc1/GELU/fc2.')
    # 12. All heads at three depths.
    fig,axes=plt.subplots(3,7,figsize=(15,7.6));fig.subplots_adjust(top=.87,bottom=.06,wspace=.05,hspace=.18)
    title(fig,'Attention heads distribute dependence differently')
    for i,b in enumerate((0,5,11)):
        p=e3('deit_s16',b);show(axes[i,0],photo,f'Block {b} · input','photo')
        for h in range(6):show(axes[i,h+1],p[f'head_{h}'],f'Head {h}',source=f'e3/deit/b{b}/h{h}')
    finish(fig,12,'all_heads','DeiT-S 在 blocks 0、5、11 的全部六个 heads，对同一个 ground-truth logit 的输入依赖图。原图逐行保留，无按地图观感筛选 head。','All six DeiT-S heads at blocks 0, 5 and 11, conditioned on the same ground-truth logit. Each row retains the original image, and no heads are selected by visual appearance.')
    # 13. Same class target across models with identifiable inputs.
    fig,axes=plt.subplots(3,3,figsize=(9.7,9));fig.subplots_adjust(top=.9,bottom=.05,wspace=.1,hspace=.2)
    title(fig,'Class-target learning images across two models')
    for i,(order,label) in enumerate([(4,'Mantis'),(14,'Sloth bear'),(15,'Little blue heron')]):
        show(axes[i,0],data[f'deit_s16__{order:03d}']['input_rgb'],label,'photo')
        for j,m in enumerate(('deit_s16','vit_b16')):show(axes[i,j+1],data[f'{m}__{order:03d}']['class_endpoint'],'DeiT-S/16' if j==0 else 'ViT-B/16',source=m+f'/{order}/class')
    finish(fig,13,'cross_model_examples','三个主体可辨的展示样本及两模型的同类别 Endpoint。样本为展示选择，模型间颜色强度不作为量值比较。','Three presentation-selected images with recognizable subjects and same-class Endpoints from both models. Color intensities are not used for cross-model magnitude comparisons.')
    # 14. Routing decomposition on same clear bear.
    fig,axes=plt.subplots(2,5,figsize=(13,6));fig.subplots_adjust(top=.86,bottom=.08,wspace=.07,hspace=.2)
    title(fig,'Routing and value on the same image and class target')
    for i,(b,labels) in enumerate([(0,('query','key','routing','value')),(11,('query','key','routing','value'))]):
        p=e3('deit_s16',b);show(axes[i,0],photo,f'Block {b} · input','photo')
        for j,k in enumerate(labels):show(axes[i,j+1],p[k],k.title(),source=f'e3/deit/b{b}/{k}')
    finish(fig,14,'routing_value_examples','同一懒熊输入在浅层和深层的 Q、K、routing 与 value 路径。蓝底保留正部细节；signed closure 和绝对质量来自完整原始 arrays。','Q, K, routing and value paths at early and late blocks for the same sloth-bear image. Blue displays retain positive detail; signed closure and absolute masses use complete original arrays.')
    # 15. Two residuals in same block.
    fig,axes=plt.subplots(2,4,figsize=(11,6.5));fig.subplots_adjust(top=.85,bottom=.08,wspace=.08,hspace=.2)
    title(fig,'Two residual sublayers, one downstream target')
    for i,(label,keys) in enumerate([('Attention',('identity','attention','L_F')),('MLP',('carry','mlp','L_F'))]):
        show(axes[i,0],photo,label+' residual','photo')
        for j,(lab,k) in enumerate(zip(('Identity','Branch','Output'),keys)):show(axes[i,j+1],parts[k],lab,source='e3/deit/b5/'+k)
    finish(fig,15,'residual_examples','同一 block 5 中 attention 与 MLP residual 对同一类别目标的两条路径。两行的 identity 指向不同 residual 输入边。正部图不可直接相加，完整 signed maps 满足闭合。','Attention and MLP residual paths in block 5 for the same class target. The identity paths refer to different residual input edges. Positive displays are not additive; the complete signed maps satisfy closure.')
    # 16–18. Quantitative trajectories are retained; sparse correctness/forest plots become tables.
    def curves(ax,rr,metric,label,extra=None):
        for model,color in [('vit_b16',BLUE),('deit_s16',ORANGE)]:
            r=sorted([r for r in rr if r.get('model_id',r.get('model'))==model and r['metric']==metric and (not extra or all(r.get(k)==v for k,v in extra.items()))],key=lambda r:int(r['block']))
            assert len(r)==12,(metric,model);y=[float(v['median']) for v in r];ci=np.array([json.loads(v['ci95']) for v in r]);ax.plot(range(12),y,lw=2.2,color=color,label='ViT-B/16' if model=='vit_b16' else 'DeiT-S/16');ax.fill_between(range(12),ci[:,0],ci[:,1],color=color,alpha=.12)
        ax.set(xlabel='Block',ylabel=label,xticks=[0,3,5,8,11]);ax.grid(axis='y');ax.legend(frameon=False,fontsize=9)
    configs=[(16,'spatial_broadening','Spatial dependence broadens with depth','nns_vit_endpoint_m3_sup_r2_20260919/block_summary.csv',[('entropy','Normalized entropy',None),('locality','Own-token locality',None),('distance','Normalized distance',None)]),
      (17,'attention_depth','Deep attention departs from Value-only propagation','nns_vit_endpoint_m4_r1_20260919/attention_block_summary.csv',[('routing_share','Routing relative scale',None),('frozen_attention_endpoint_error','Endpoint approximation error',None),('frozen_attention_jvp_error','JVP discrepancy',None)]),
      (18,'residual_depth','Attention and MLP residuals follow distinct trends','nns_vit_endpoint_m4_r1_20260919/residual_block_summary.csv',[('identity_share','Attention identity scale',{'residual_type':'attention'}),('identity_share','MLP identity scale',{'residual_type':'mlp'}),('path_cancellation','MLP path cancellation',{'residual_type':'mlp'})])]
    for n,slug,t,path,metrics in configs:
        rr=rows(REPO/'reports'/path);fig,axes=plt.subplots(1,3,figsize=(12,4.1));fig.subplots_adjust(top=.76,bottom=.19,wspace=.32);title(fig,t)
        for ax,(metric,label,extra) in zip(axes,metrics):curves(ax,rr,metric,label,extra)
        finish(fig,n,slug,'完整 cohort 的图像级层深轨迹，阴影为 95% bootstrap 区间。空间分析每模型 100 图，attention/residual 每模型 400 图；推断沿用原冻结分析。','Image-level depth trajectories from the full cohorts, with 95% bootstrap intervals. Spatial analyses use 100 images per model and attention/residual analyses 400; inference follows the original analysis.','quantitative')
    # 19–20. Actual method comparison; original photo in every row.
    comparison_orders={'deit_s16':receipt['figure19_orders'],'vit_b16':receipt['figure20_orders']}
    for n,model in [(19,'deit_s16'),(20,'vit_b16')]:
        fig,axes=plt.subplots(3,6,figsize=(15,8));fig.subplots_adjust(top=.88,bottom=.06,wspace=.035,hspace=.18)
        title(fig,('DeiT-S/16' if model=='deit_s16' else 'ViT-B/16')+': learning images and other visual explanations')
        for i,order in enumerate(comparison_orders[model]):
            d=data[f'{model}__{order:03d}'];items=[(SAMPLE_LABELS[order]+('\nOriginal' if i==0 else ''),d['input_rgb'],'photo'),('Endpoint',d['class_endpoint'],'learning'),('Endpoint\npatch-max display',patchmax(d['class_endpoint']),'score'),('Grad-CAM',d['gradcam'],'score'),('Attention rollout',d['rollout'],'score'),('Raw attention',d['raw_attention'],'score')]
            for ax,(lab,v,kind) in zip(axes[i],items):show(ax,v,lab if i==0 else '',kind,source=f'{model}/{order}/{lab}',interpolation='bilinear' if kind=='score' else 'nearest')
            axes[i,0].set_title(SAMPLE_LABELS[order]+('\nOriginal' if i==0 else ''),pad=7,fontsize=10)
        selection_note=(' Figure 20 examples were selected by the presealed Endpoint/input-detail readability rule before comparison methods were computed.' if model=='vit_b16' else '')
        selection_zh=(' Figure 20 的三例在计算比较方法前，已按预先封存的 Endpoint/输入细节可读性规则选定。' if model=='vit_b16' else '')
        finish(fig,n,model+'_method_comparison','同输入实际计算的 Endpoint、patch-max 显示变体、Grad-CAM、attention rollout 与 raw attention。Endpoint/Grad-CAM 使用同一 GT logit；后两种 attention 方法不依赖类别。所有图采用相同蓝底色系并独立归一化，只比较表示形态与粒度。'+selection_zh,'Actual same-input Endpoint, its patch-max display variant, Grad-CAM, attention rollout and raw attention. Endpoint/Grad-CAM use the same GT logit; attention methods are class-agnostic. A common blue palette and independent normalization compare spatial form and granularity, not quantitative superiority.'+selection_note)
    # 21. Utility curves and absolute effect scale.
    e5=load(REPO/'reports/nns_vit_endpoint_e5_spatial_r1_20260918/formal_results/attempts/deit_s16__014/attempt-01.json')['result']
    stats=load(REPO/'reports/nns_vit_endpoint_paper_revision_v1_20260920/analysis.json')
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.4),gridspec_kw={'width_ratios':[.75,1.2,1.5]});fig.subplots_adjust(top=.78,bottom=.21,wspace=.35);title(fig,'Spatial intervention evaluates a different property')
    show(axes[0],photo,'Example input','photo')
    for method,color,label in [('endpoint_component',BLUE,'Endpoint'),('regional_taylor',ORANGE,'Regional Taylor')]:
        curves0=[c for c in e5['references']['normalized_zero']['curves'] if c['method']==method]
        vals=np.mean([[abs(p['isolated_effect']) for p in c['points']] for c in curves0],axis=0)
        axes[1].plot([0,.0625,.125,.25],vals,color=color,lw=2,label=label)
    axes[1].set(xlabel='Replaced area fraction',ylabel='Absolute logit change');axes[1].legend(frameon=False,fontsize=9);axes[1].grid(axis='y')
    groups=[(m,r) for m in ('vit_b16','deit_s16') for r in ('normalized_zero','blur_rgb')]
    for j,(method,label,color) in enumerate([('endpoint_component','Endpoint',BLUE),('regional_taylor','Taylor',ORANGE),('empirical_oracle','Candidate oracle',TEAL)]):
        vals=[next(v['auc']['median'] for v in stats['spatial']['summary'] if (v['model'],v['reference'],v['method'])==(m,r,method)) for m,r in groups];axes[2].bar(np.arange(4)+(j-1)*.24,vals,.23,color=color,label=label)
    axes[2].set(xticks=range(4),xticklabels=['ViT\nzero','ViT\nblur','DeiT\nzero','DeiT\nblur'],ylabel='Median absolute-effect AUC');axes[2].legend(frameon=False,fontsize=8);axes[2].grid(axis='y');axes[2].set_axisbelow(True)
    finish(fig,21,'intervention_boundary','左侧示例的有限区域替换曲线与右侧完整 400 图 family 的绝对 AUC。候选 oracle 只在已评价区域集合中取最优；数学闭合与区域干预选择属于不同性质。','Finite region-replacement curves for the example and absolute AUC over complete 400-image families. The candidate oracle is restricted to evaluated region sets; mathematical closure and spatial intervention selection measure different properties.','quantitative')
    # 22. Complementary views make the blue display interpretation explicit.
    fig,axes=plt.subplots(2,4,figsize=(11,6.6));fig.subplots_adjust(top=.84,bottom=.08,wspace=.08,hspace=.25);title(fig,'Reading positive structure without discarding signed information')
    for i,(name,key) in enumerate([('Class target','class_endpoint'),('Block 5 energy','b5.energy.endpoint')]):
        show(axes[i,0],photo,name,'photo');show(axes[i,1],bear[key],'Positive learning image',source=key);show(axes[i,2],bear[key],'Signed learning image','signed',key);show(axes[i,3],bear[key],'Absolute mass','absolute',key)
    finish(fig,22,'signed_and_blue','同一原始图的蓝底正部、signed 与 RGB absolute mass 三种视图。正部用于读取空间细节，signed 保留方向，absolute 保留通道抵消前的总幅值。','Blue positive, signed and RGB absolute-mass views of the same original arrays. The positive view displays spatial detail, signed maps retain direction, and absolute maps preserve magnitude before channel cancellation.')
    FIGS.sort(key=lambda r:r['number'])
    for row in DISPLAYS:row['figure']=NUMBERING.get(row['figure'],row['figure'])
    write(ROOT/'figure_manifest.json',dict(revision='PUBLICATION-R2',figure_count=len(FIGS),figures=FIGS))
    write(ROOT/'evidence/display_receipt.json',dict(policy='positive RGB sum; positive-value P99; blue high-intensity palette; no smoothing for pixel Endpoint',panels=DISPLAYS))
    write(ROOT/'evidence/figure_sources.json',SOURCES)
    cards=''.join(f'<article><h2>Figure {r["number"]}. {r["title"]}</h2><img src="{r["files"]["png"]["path"]}"><p>{r["caption_zh"]}</p><p>{r["caption_en"]}</p><a href="{r["files"]["pdf"]["path"]}">PDF</a> · <a href="{r["files"]["svg"]["path"]}">SVG</a></article>' for r in FIGS)
    (ROOT/'gallery.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>NNS-ViT paper figures</title><style>body{max-width:1280px;margin:40px auto;background:#eef3f8;color:#18324b;font:16px system-ui;padding:20px}article{background:white;padding:28px;margin:28px 0;border-radius:12px}img{width:100%}h2{font-size:20px}p{line-height:1.6}a{color:#195d9c}</style><h1>NNS-ViT · learning images</h1><p>22 figures · conceptual explanation, layerwise views, actual method comparisons</p>'+cards+'</html>')
    tx=np.array([1.,2.,3.,4.]);tw=np.array([[1.,.5,0.,0.],[0.,0.,.5,1.]]);tv=np.array([1.,.5])
    write(ROOT/'evidence/concept_numeric_example.json',dict(x=tx.tolist(),W=tw.tolist(),v=tv.tolist(),h=(tw@tx).tolist(),s=float(tv@tw@tx),endpoint=(tx*(tv@tw)).tolist()))
    print('COMPLETE',len(FIGS),len(DISPLAYS))

if __name__=='__main__':main()
