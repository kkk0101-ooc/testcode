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
def inset(fig,x,y,w,h,value,label='',kind='learning',source='',interpolation='nearest'):
    ax=fig.add_axes([x,y,w,h]);show(ax,value,label,kind,source,interpolation);return ax
def matrix(ax,x,y,w,h,a,label='',color=BLUE):
    a=np.asarray(a);nr,nc=a.shape
    for i in range(nr):
        for j in range(nc):
            xx=x+j*w/nc;yy=y+(nr-1-i)*h/nr
            ax.add_patch(Rectangle((xx,yy),w/nc,h/nr,facecolor='#EAF3FB' if a[i,j]!=0 else 'white',edgecolor=color,lw=1))
            ax.text(xx+w/nc/2,yy+h/nr/2,f'{a[i,j]:g}',ha='center',va='center',fontsize=12)
    if label:ax.text(x+w/2,y+h+.018,label,ha='center',fontsize=11,color=color)
def dashed_panel(ax,x,y,w,h,label='',color='#9AA3AC'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.018',
        facecolor='white',edgecolor=color,lw=1.5,linestyle=(0,(5,4))))
    if label:ax.text(x+.018,y+h-.032,label,ha='left',va='top',fontsize=12,style='italic',weight='bold',color=INK)
def node(ax,x,y,text,color,r=.034,fill='white'):
    ax.add_patch(Circle((x,y),r,facecolor=fill,edgecolor=color,lw=2.0));ax.text(x,y,text,ha='center',va='center',fontsize=11)
def dotted_link(ax,a,b,color,lw=1.5):
    ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,linestyle=(0,(2,3)))
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
    # 1. Step-3 method overview: primitive -> dual index -> valid cut -> mechanism question.
    fig,ax=canvas((14,6.8));title(fig,'From total Gradient × Input to auditable dual-indexed maps','One registered scalar target s · one observed input x · one frozen computational graph')
    panels=[(.02,'1 · Total primitive'),(.265,'2 · Expand by use edge'),(.51,'3 · Reconstruct by a valid cut'),(.755,'4 · Ask structural questions')]
    for xx,label in panels:dashed_panel(ax,xx,.16,.225,.68,label)
    # Total map: the established Gradient × Input primitive.
    grid=np.array([[.05,.20,.45,.10],[.15,.75,1.0,.25],[.08,.55,.68,.16],[.03,.12,.30,.06]])
    inset(fig,.055,.39,.15,.27,grid,r'$x\odot\nabla_xs(x)$','score','concept-total')
    ax.text(.132,.29,'one input-indexed map',ha='center',fontsize=10,color=GRAY)
    ax.text(.132,.225,r'$i$ retained · $e$ summed out',ha='center',fontsize=11,color=INK)
    # Edge-indexed stack: same input coordinates, concrete use-edge identities.
    maps=[grid*np.array([[1,.7,.4,.2]]).T,grid*np.array([[.2,.5,.9,.6]]),grid*np.array([[.6,.3,.2,.8]])]
    labels=[r'$\mathcal{L}_{e_Q,i}$',r'$\mathcal{L}_{e_K,i}$',r'$\mathcal{L}_{e_V,i}$']
    for j,(value,label0,color) in enumerate(zip(maps,labels,(BLUE,TEAL,ORANGE))):
        yy=.615-j*.145;inset(fig,.292,yy-.055,.085,.105,value,'','score','concept-edge-'+str(j));ax.text(.392,yy,label0,fontsize=12,color=color,va='center')
    ax.text(.377,.225,r'joint index $[e,i]$ · conditioned on $s$',ha='center',fontsize=11,color=INK)
    # Three concrete cuts all reconstruct the same signed local differential.
    for j,(label0,terms,color) in enumerate([
        (r'$C_{\mathrm{attn}}$','Q + K + V',TEAL),
        (r'$C_{\mathrm{head}}$','head 1 + ··· + head H',BLUE),
        (r'$C_{\mathrm{res}}$','identity + branch',ORANGE),
    ]):
        yy=.66-j*.125;box(ax,.535,yy-.039,.055,.070,label0,color,'#F7FAFC',10);ax.text(.605,yy,terms,fontsize=10.5,va='center',color=INK)
    ax.text(.622,.265,r'$\sum_{e\in C}\mathcal{L}_{e,i}$',ha='center',fontsize=13,color=INK)
    ax.text(.622,.205,r'$=\;x_i\,\partial_i s(x)$',ha='center',fontsize=13,color='#C62828',weight='bold')
    # Mechanism questions require the joint edge-by-input structure.
    for j,(label0,color) in enumerate([('routing ↔ value',TEAL),('identity ↔ branch',ORANGE),('space × path × depth',BLUE)]):
        box(ax,.79,.62-j*.14,.155,.095,label0,color,'#F7FAFC',10.5)
    ax.text(.867,.225,'mechanistic comparisons',ha='center',fontsize=10,color=GRAY)
    for a,b in [((.245,.50),(.26,.50)),((.49,.50),(.505,.50)),((.735,.50),(.75,.50))]:arrow(ax,a,b,'#C62828',style=(0,(5,3)),lw=1.8)
    # Explicit degenerations: each deletion removes one class of answer.
    ax.text(.035,.075,r'drop $i$ → edge scalar',fontsize=10.5,color=GRAY)
    ax.text(.286,.075,r'drop $e$ → total G×I',fontsize=10.5,color=GRAY)
    ax.text(.527,.075,'drop cut contract → unverified accounting',fontsize=10.5,color=GRAY)
    ax.text(.784,.075,r'drop registry $s$ → no target atlas',fontsize=10.5,color=GRAY)
    finish(fig,1,'concept_overview','对每个注册目标，已有的总 Gradient × Input 被展开为同时保留 consumer-use edge 与输入坐标的局部微分图组。合法 attention、head 或 residual 切面在 signed 输入坐标上重构同一总量；路径身份与空间位置共同支持层深机制比较。下方列出删除输入索引、边索引、切面合同或目标 registry 后的退化。','For each registered target, the established total Gradient × Input map is expanded into local differential maps jointly indexed by consumer-use edge and input coordinate. Valid attention, head, or residual cuts reconstruct the same signed total, while joint path and spatial identity support depth-wise mechanistic comparisons. The bottom row shows the loss caused by removing the input index, edge index, cut contract, or target registry.','concept')
    # 2. Consumer-use identity and the structural-cut closure contract.
    fig,ax=canvas((14,7.2));title(fig,'Valid differential cuts partition live paths exactly once','Consumer-use identity prevents shared producers, input slots and repeated invocations from being merged')
    dashed_panel(ax,.02,.12,.265,.75,'A · One producer, two concrete uses')
    node(ax,.075,.59,r'$z$',BLUE,.035,'#EEF5FB');ax.text(.075,.665,'producer tensor',ha='center',fontsize=10,color=GRAY)
    box(ax,.155,.64,.105,.105,'consumer A\nslot 0 · call 1',TEAL,'#EAF7F6',9.5);box(ax,.155,.38,.105,.105,'consumer B\nslot 1 · call 2',ORANGE,'#FFF2E3',9.5)
    arrow(ax,(.112,.60),(.15,.68),TEAL);arrow(ax,(.112,.575),(.15,.43),ORANGE)
    ax.text(.045,.29,r'$e=(\mathrm{producer},\mathrm{consumer},$',fontsize=10.5)
    ax.text(.065,.245,r'$\mathrm{input\ slot},\mathrm{invocation})$',fontsize=10.5)
    ax.text(.052,.18,'same tensor ≠ same use edge',fontsize=10.5,color='#C62828',weight='bold')
    dashed_panel(ax,.31,.12,.455,.75,'B · Registered cuts through a ViT block')
    # Schematic live graph. Red dashed paths stress complete input-to-target routes.
    node(ax,.345,.50,r'$x$',BLUE,.028,'#EEF5FB')
    for yy,lab,color in [(.70,'Q',TEAL),(.56,'K',TEAL),(.42,'V',ORANGE)]:box(ax,.405,yy-.035,.07,.07,lab,color,'#F7FAFC',11);arrow(ax,(.373,.50),(.40,yy),color,lw=1.1)
    box(ax,.51,.58,.09,.13,'routing\nQ + K',TEAL,'#EAF7F6',9.5);box(ax,.51,.38,.09,.11,'value\nV',ORANGE,'#FFF2E3',9.5)
    arrow(ax,(.48,.70),(.505,.66),TEAL,lw=1.1);arrow(ax,(.48,.56),(.505,.62),TEAL,lw=1.1);arrow(ax,(.48,.42),(.505,.435),ORANGE,lw=1.1)
    box(ax,.63,.52,.08,.12,'heads\n$1…H$',BLUE,'#EEF5FB',10);arrow(ax,(.605,.64),(.625,.595),TEAL,lw=1.1);arrow(ax,(.605,.435),(.625,.555),ORANGE,lw=1.1)
    node(ax,.70,.34,'id',BLUE,.025,'#EEF5FB');node(ax,.735,.49,'+',ORANGE,.028,'#FFF2E3');node(ax,.735,.72,r'$s$',TEAL,.03,'#EAF7F6')
    arrow(ax,(.713,.575),(.73,.525),ORANGE,lw=1.1);arrow(ax,(.37,.475),(.695,.34),BLUE,lw=1.1);arrow(ax,(.72,.35),(.735,.455),BLUE,lw=1.1);arrow(ax,(.735,.52),(.735,.685),TEAL,lw=1.3)
    ax.plot([.492,.492],[.27,.78],color='#C62828',lw=1.35,linestyle=(0,(5,3)));ax.text(.492,.80,r'$C_{attn}$',ha='center',fontsize=9.5,color='#C62828')
    ax.plot([.615,.615],[.27,.78],color='#C62828',lw=1.35,linestyle=(0,(5,3)));ax.text(.615,.80,r'$C_{head}$',ha='center',fontsize=9.5,color='#C62828')
    ax.plot([.685,.755],[.35,.50],color='#C62828',lw=1.35,linestyle=(0,(5,3)));ax.text(.682,.285,r'$C_{res}={id,branch}$',fontsize=9.5,color='#C62828')
    ax.text(.335,.195,r'$C_{attn}=\{Q,K,V\}=\{routing,value\}$',fontsize=9.5)
    ax.text(.335,.155,r'$C_{head}=\{h_1,\ldots,h_H\}$',fontsize=9.5)
    dashed_panel(ax,.79,.12,.19,.75,'C · Closure and audit')
    ax.text(.885,.70,r'$\mathcal{L}_{e,i}=x_i[J_{z_e,x}^{\mathsf{T}}\bar z_e^{(s)}]_i$',ha='center',fontsize=10.5)
    ax.text(.885,.61,r'$\sum_{e\in C}\mathcal{L}_{e,i}=x_i\partial_i s$',ha='center',fontsize=12,color='#C62828',weight='bold')
    for j,(mark,label0,color) in enumerate([('✓','each live path once',TEAL),('×','omit nonzero edge','#C62828'),('×','duplicate identity','#C62828')]):
        yy=.48-j*.105;ax.text(.82,yy,mark,fontsize=17,color=color,weight='bold',va='center');ax.text(.85,yy,label0,fontsize=10,va='center')
    ax.text(.81,.19,'closed on raw signed\ninput coordinates',fontsize=10,color=GRAY,linespacing=1.35)
    finish(fig,2,'neuron_learning_image','左栏用 producer、consumer、input slot 与 invocation 区分同一张量的两次实际使用。中栏给出 attention Q/K/V、multi-head 与 residual identity/branch 的合法切面；每条有效输入到目标路径在一个切面上恰好穿过一次。右栏给出双索引定义、signed closure 以及已封存的漏项和重复记账失败模式。','The left panel distinguishes two concrete uses of the same tensor by producer, consumer, input slot, and invocation. The middle panel shows valid attention Q/K/V, multi-head, and residual identity/branch cuts; every live input-to-target path crosses a cut exactly once. The right panel states the dual-indexed definition, signed closure, and the sealed omission and duplicate-accounting failure modes.','concept')
    # 3. Reference-paper-style coordinate initialization.
    x=np.array([1.,2.,3.,4.]);W=np.array([[1.,.5,0.,0.],[0.,0.,.5,1.]]);v=np.array([1.,.5]);E=x*(v@W)
    colors=['#E67E22','#2E8B57','#8E44AD','#D4A017']
    fig,ax=canvas((12,6.6));title(fig,'Initialization: split the input, then assign one image to each coordinate')
    dashed_panel(ax,.025,.075,.51,.795,'1 · Split the input');dashed_panel(ax,.58,.075,.395,.795,'2 · Assign learning images')
    matrix(ax,.065,.37,.15,.31,x.reshape(2,2),r'$x$',INK);ax.text(.24,.50,'=',fontsize=25)
    for i,(val,c) in enumerate(zip(x,colors)):
        a=np.zeros(4);a[i]=val;yy=.70-i*.155;matrix(ax,.32,yy-.075,.105,.145,a.reshape(2,2),fr'$I_{i+1}$',c)
        if i<3:ax.text(.45,yy-.01,'+',fontsize=20)
        matrix(ax,.64,yy-.075,.105,.145,a.reshape(2,2),fr'$I_{i+1}^{0}$',c);dotted_link(ax,(.755,yy),(.845,yy),c);node(ax,.88,yy,fr'$O_{i+1}^{0}$',c,.03)
    arrow(ax,(.52,.49),(.575,.49),GRAY,lw=2.2)
    ax.text(.08,.105,r'$x=I_1+I_2+I_3+I_4$',fontsize=14);ax.text(.635,.105,r'$\mathcal{L}(x;x)=\mathrm{diag}(x)$',fontsize=15)
    finish(fig,3,'coordinate_initialization','初始化分为拆分与分配两步：输入矩阵被拆成四张稀疏坐标图，并按同色虚线分配给四个输入节点。每张图同时保存输入值与其位置。','Initialization has two steps: split the input into four sparse coordinate images and assign them to input nodes using matching colored links. Each image preserves both value and position.','concept')
    # 4. Reference Fig. 4 style: muted intermediate graph, highlighted complete path coefficients.
    fig,ax=canvas((12,6.5));title(fig,'Generation: collect every input path to the selected target')
    dashed_panel(ax,.035,.13,.93,.70,'Learning-image generation for target s')
    for i,c in enumerate(colors):matrix(ax,.07,.625-i*.105,.09,.085,(np.eye(4)[i]*x[i]).reshape(2,2),fr'$I_{i+1}^{0}$',c)
    hidden_pos=[(.49,.65),(.49,.38)];target_pos=(.86,.515)
    for j,pos in enumerate(hidden_pos):box(ax,pos[0]-.055,pos[1]-.055,.11,.11,fr'$I_{j+1}^{1}$','#B6BDC4','#F7F8F9',13)
    box(ax,target_pos[0]-.06,target_pos[1]-.06,.12,.12,r'$I_s$',ORANGE,'#FFF4E8',14)
    for i,c in enumerate(colors):
        yy=.6675-i*.105
        for j,pos in enumerate(hidden_pos):
            if W[j,i]:arrow(ax,(.17,yy),(pos[0]-.062,pos[1]),'#BCC2C8',lw=.9)
    for pos in hidden_pos:arrow(ax,(pos[0]+.062,pos[1]),(target_pos[0]-.067,target_pos[1]),'#BCC2C8',lw=1.0)
    # First and last composed paths are highlighted, as in the reference-paper convention.
    coeff=v@W
    for i in (0,3):
        yy=.6675-i*.105;ax.plot([.16,.72,.80],[yy,yy,target_pos[1]],color='#C62828',lw=1.35,linestyle=(0,(5,3)))
        arrow(ax,(.80,target_pos[1]),(target_pos[0]-.067,target_pos[1]),'#C62828',style=(0,(5,3)),lw=1.35)
        ax.text(.67,yy+.018,fr'$c_{i+1}={coeff[i]:g}$',fontsize=10,color='#C62828')
    ax.text(.69,.555,r'$c_2=0.5$',fontsize=10,color='#C62828');ax.text(.69,.475,r'$c_3=0.25$',fontsize=10,color='#C62828');ax.text(.735,.515,r'$\vdots$',fontsize=15,color='#C62828')
    matrix(ax,.78,.19,.16,.21,E.reshape(2,2),r'$I_s=\sum_i c_i I_i^0$',ORANGE)
    ax.text(.25,.20,r'$c_i=\frac{\partial s}{\partial x_i}$',fontsize=15,color='#C62828');ax.text(.24,.075,r'$I_s=\mathcal{L}(s;x),\qquad E_s=x\odot\nabla_xs$',fontsize=16)
    finish(fig,4,'local_derivative_generation','中间网络以灰色显示，红色虚线路径突出每个输入坐标到目标的合成局部系数。右下矩阵是四张输入 learning images 的加权和，与输入乘目标梯度一致。','The intermediate graph is muted while red dashed paths highlight composed local coefficients from every input coordinate to the target. The lower-right matrix is the weighted sum of input learning images and equals input times target gradient.','concept')
    # 5. Reference Fig. 5 style: native patch operation and equivalent sparse linear map.
    fig,ax=canvas((13,6.6));title(fig,'Patch embedding is a local projection and a sparse global linear map')
    dashed_panel(ax,.02,.12,.46,.75,'Native patch projection');dashed_panel(ax,.52,.12,.46,.75,'Sparse linear form')
    ia=inset(fig,.045,.35,.21,.39,photo,'Input image','photo')
    for k in range(1,14):ia.axvline(k*16-.5,color='white',lw=.28,alpha=.55);ia.axhline(k*16-.5,color='white',lw=.28,alpha=.55)
    ia.add_patch(Rectangle((6*16-.5,6*16-.5),16,16,fill=False,edgecolor='#D62728',lw=2.2))
    ax.text(.27,.55,'×',fontsize=25);box(ax,.31,.47,.13,.16,'Patch row\n$w_c^\top$',ORANGE,'#FFF2E3',12);arrow(ax,(.255,.545),(.30,.545),GRAY)
    box(ax,.32,.25,.12,.12,r'$z_{t,c}$',TEAL,'#EAF7F6',13);arrow(ax,(.375,.46),(.38,.38),TEAL)
    ax.text(.055,.20,r'$z_{t,c}=w_c^\top x_t+b_c$',fontsize=14)
    # Global patch grid and sparse row-matrix view.
    grid=np.zeros((14,14));grid[6,6]=1;gx=.55;gy=.405;gw=.19;gh=.31
    for i in range(14):
        for j in range(14):
            color='#F05A28' if (i,j)==(6,6) else '#EAF1F6'
            ax.add_patch(Rectangle((gx+j*gw/14,gy+(13-i)*gh/14),gw/14,gh/14,facecolor=color,edgecolor='white',lw=.18))
    ax.text(gx+gw/2,gy+gh+.025,'Image coordinates',ha='center',fontsize=11)
    arrow(ax,(.75,.61),(.79,.61),GRAY)
    # Sparse projection row with one active 16x16 block.
    ax.add_patch(Rectangle((.80,.50),.145,.21,facecolor='#F4F6F8',edgecolor=GRAY,lw=1.2))
    for j in range(7):ax.add_patch(Rectangle((.805+j*.019,.56),.016,.09,facecolor=('#F05A28' if j==3 else 'white'),edgecolor='#A9B2BA',lw=.45))
    ax.text(.872,.755,'Sparse row $W_{t,c}$',ha='center',fontsize=11);ax.text(.872,.525,'nonzero only in patch t',ha='center',fontsize=9,color=GRAY)
    arrow(ax,(.872,.49),(.872,.39),TEAL)
    inset(fig,.80,.16,.145,.20,grid,'Learning-image support','score','patch-support',interpolation='nearest')
    ax.text(.55,.19,r'$E_{t,c}=x\odot W_{t,c}$',fontsize=15);ax.text(.55,.14,'Bias changes the scalar value but has no input-coordinate term.',fontsize=9.5,color=GRAY)
    finish(fig,5,'patch_embedding','左栏以真实图像说明 patch embedding 的局部投影；右栏将同一运算写成全局输入上的稀疏线性行。橙色区域在 token、稀疏权重与 learning-image support 之间一一对应。','The left panel shows native patch projection on a real image; the right expresses the same operation as a sparse row over global input coordinates. The orange region corresponds across token support, sparse weights and learning-image support.','concept')
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
    # 7. Paired forward/residual path and aligned real learning images.
    fig,ax=canvas((13,7.2));title(fig,'Residual connections preserve an identity path and add a transformed path','Top: native computation · Bottom: corresponding input-coordinate learning images')
    dashed_panel(ax,.025,.54,.95,.34,'Residual computation');dashed_panel(ax,.025,.055,.95,.405,'NNS-ViT path images')
    node(ax,.14,.70,r'$u$',BLUE,.038,'#EEF5FB');box(ax,.38,.73,.19,.12,r'Branch $f(u)$',ORANGE,'#FFF2E3',13);node(ax,.75,.70,r'$+$',TEAL,.035,'#E9F7F5');node(ax,.89,.70,r'$y$',TEAL,.038,'#E9F7F5')
    arrow(ax,(.18,.73),(.37,.79),ORANGE);arrow(ax,(.58,.79),(.71,.72),ORANGE);arrow(ax,(.18,.67),(.71,.68),BLUE);arrow(ax,(.79,.70),(.85,.70),TEAL)
    ax.text(.41,.60,r'$y=u+f(u)$',fontsize=15);ax.text(.61,.60,r'$g_y=\nabla_yF$',fontsize=13,color=GRAY)
    panels=[('Original input',photo,'photo',''),('Identity path',parts['identity'],'learning','identity'),('Branch path',parts['attention'],'learning','attention'),('Residual output',parts['L_F'],'learning','L_F')]
    centers=[]
    for i,(lab,value,kind,source) in enumerate(panels):
        xx=.045+i*.235;inset(fig,xx,.125,.18,.235,value,lab,kind,source);centers.append(xx+.09)
    dotted_link(ax,(.14,.66),(centers[1],.385),BLUE);dotted_link(ax,(.475,.73),(centers[2],.385),ORANGE);dotted_link(ax,(.89,.66),(centers[3],.385),TEAL)
    ax.text(.49,.245,r'$+$',fontsize=24,color=TEAL);ax.text(.72,.245,r'$=$',fontsize=24,color=TEAL)
    arrow(ax,(centers[1]+.095,.245),(centers[2]-.10,.245),'#C62828',style=(0,(5,3)),lw=1.3);arrow(ax,(centers[2]+.095,.245),(centers[3]-.10,.245),'#C62828',style=(0,(5,3)),lw=1.3)
    ax.text(.28,.075,r'$E_y^{\mathrm{signed}}=E_{\mathrm{identity}}+E_{\mathrm{branch}}$',fontsize=15);ax.text(.67,.075,'All paths use the same output cotangent.',fontsize=10,color=GRAY)
    finish(fig,7,'residual_principle','上方显示 residual 的 identity 与 branch 两条真实计算边；下方用同色虚线连接到同一输入、同一类别目标的路径 learning images。完整 signed maps 相加得到 residual output，蓝底正部用于阅读空间结构。','The upper graph shows the native identity and branch edges; matching dashed links connect them to path learning images for the same input and class target. Complete signed maps add to the residual output, while blue positive views display spatial structure.','concept')
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
    main_map=[dict(main_number=i+1,package_number=n) for i,n in enumerate((1,2,6,7,8,12,15,16,17,18,21))]
    manifest=dict(revision='PUBLICATION-R2-STEP3',figure_count=len(FIGS),main_text_figure_count=11,main_text_figure_map=main_map,figures=FIGS)
    write(ROOT/'figure_manifest.json',manifest)
    root_relative=ROOT.relative_to(REPO).as_posix()
    binding={'source_revision':'PUBLICATION-R2-STEP3','figure_count':len(FIGS),'package':root_relative,'figures':[]}
    for row in FIGS:
        bound={'figure':row['number'],'slug':row['slug']}
        for extension,record in row['files'].items():bound[extension]={'path':f"{root_relative}/{record['path']}",'sha256':record['sha256']}
        binding['figures'].append(bound)
    write(ROOT.parent/'figure_manifest.json',binding)
    write(ROOT/'evidence/display_receipt.json',dict(policy='positive RGB sum; positive-value P99; blue high-intensity palette; no smoothing for pixel Endpoint',panels=DISPLAYS))
    write(ROOT/'evidence/figure_sources.json',SOURCES)
    cards=''.join(f'<article><h2>Figure {r["number"]}. {r["title"]}</h2><img src="{r["files"]["png"]["path"]}"><p>{r["caption_zh"]}</p><p>{r["caption_en"]}</p><a href="{r["files"]["pdf"]["path"]}">PDF</a> · <a href="{r["files"]["svg"]["path"]}">SVG</a></article>' for r in FIGS)
    (ROOT/'gallery.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>NNS-ViT paper figures</title><style>body{max-width:1280px;margin:40px auto;background:#eef3f8;color:#18324b;font:16px system-ui;padding:20px}article{background:white;padding:28px;margin:28px 0;border-radius:12px}img{width:100%}h2{font-size:20px}p{line-height:1.6}a{color:#195d9c}</style><h1>NNS-ViT · learning images</h1><p>22 figures · conceptual explanation, layerwise views, actual method comparisons</p>'+cards+'</html>')
    tx=np.array([1.,2.,3.,4.]);tw=np.array([[1.,.5,0.,0.],[0.,0.,.5,1.]]);tv=np.array([1.,.5])
    write(ROOT/'evidence/concept_numeric_example.json',dict(x=tx.tolist(),W=tw.tolist(),v=tv.tolist(),h=(tw@tx).tolist(),s=float(tv@tw@tx),endpoint=(tx*(tv@tw)).tolist()))
    print('COMPLETE',len(FIGS),len(DISPLAYS))

if __name__=='__main__':main()
