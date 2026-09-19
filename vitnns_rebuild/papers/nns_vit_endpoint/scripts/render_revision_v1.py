#!/usr/bin/env python3
"""Publication revision from sealed arrays and tables, without model inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from PIL import Image

REPO=Path(__file__).resolve().parents[3]
PAPER=REPO/'papers/nns_vit_endpoint'
OUT=REPO/'reports/nns_vit_endpoint_paper_revision_v1_20260920'
R2=REPO/'reports/nns_vit_endpoint_all_figures_r2_20260919'
M3=REPO/'reports/nns_vit_endpoint_m3_sup_r2_20260919'
M4=REPO/'reports/nns_vit_endpoint_m4_r1_20260919'
M6=REPO/'reports/nns_vit_endpoint_m6_r1_20260919'
BLUE='#2372A4';ORANGE='#C57724';INK='#172B42';GRAY='#63758A'
COLORS={'vit_b16':BLUE,'deit_s16':ORANGE}
LABELS={'vit_b16':'ViT-B/16','deit_s16':'DeiT-S/16'}
SOURCES={};PANELS=[];FIGURES=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#B8C3CD','axes.labelcolor':INK,'text.color':INK,'xtick.color':GRAY,
 'ytick.color':GRAY,'grid.color':'#E5EBF0','pdf.fonttype':42,'svg.fonttype':'none',
 'savefig.facecolor':'white','axes.titlepad':9})

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def bind(path,expected=None):
    digest=sha(path)
    if expected: assert digest==expected,(path,digest,expected)
    SOURCES[str(path.relative_to(REPO))]=dict(sha256=digest,bytes=path.stat().st_size)
    return path

def load(path): return json.loads(bind(path).read_text())
def rows(path): return list(csv.DictReader(bind(path).open()))
def write(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def title(fig,number,text,subtitle):
    fig.suptitle(f'{number:02d}  {text}',x=.04,y=.99,ha='left',fontsize=16,fontweight='bold')
    fig.text(.04,.94,subtitle,fontsize=9.5,color=GRAY,va='top')

def save(fig,number,slug,description):
    records={}
    for suffix in ('png','pdf','svg'):
        d=OUT/f'figures_{suffix}';d.mkdir(exist_ok=True)
        p=d/f'figure_{number:02d}_{slug}.{suffix}'
        fig.savefig(p,dpi=200 if suffix=='png' else 150,bbox_inches='tight',pad_inches=.15)
        records[suffix]=dict(path=str(p.relative_to(REPO)),sha256=sha(p),bytes=p.stat().st_size)
    FIGURES.append(dict(figure=number,slug=slug,description=description,**records));plt.close(fig)
    print('rendered',number,flush=True)

def curve(ax,data,metric,label,**filters):
    for model in COLORS:
        selected=[r for r in data if r.get('model',r.get('model_id'))==model and r.get('metric')==metric
                  and all(r.get(k)==v for k,v in filters.items())]
        selected.sort(key=lambda r:int(r['block']));assert len(selected)==12,(metric,model,filters)
        x=[int(r['block']) for r in selected];y=[float(r['median']) for r in selected]
        ci=np.array([json.loads(r['ci95']) for r in selected])
        ax.plot(x,y,color=COLORS[model],label=LABELS[model],lw=2,marker='o',ms=3)
        ax.fill_between(x,ci[:,0],ci[:,1],color=COLORS[model],alpha=.13,lw=0)
    ax.set(xlabel='Block',ylabel=label,xticks=[0,3,5,8,11]);ax.grid(axis='y');ax.legend(frameon=False,fontsize=8)

def limits(values):
    s=np.concatenate([np.abs(np.asarray(v,dtype=float).sum(0)).ravel() for v in values])
    a=np.concatenate([np.abs(np.asarray(v,dtype=float)).sum(0).ravel() for v in values])
    return max(float(np.quantile(s,.99)),1e-30),max(float(np.quantile(a,.99)),1e-30)

def map_panel(ax,value,view,label,scale,group):
    raw=np.asarray(value,dtype=float).squeeze();assert raw.shape==(3,224,224)
    data=raw.sum(0) if view=='signed' else np.abs(raw).sum(0)
    norm=AsinhNorm(linear_width=scale/9,vmin=-scale if view=='signed' else 0,vmax=scale,clip=True)
    im=ax.imshow(data,cmap='RdBu_r' if view=='signed' else 'magma',norm=norm,interpolation='none')
    ax.set_title(label,fontsize=10);ax.set(xticks=[],yticks=[])
    mass=float(np.abs(raw).sum());norm2=float(np.linalg.norm(raw));signed_sum=float(raw.sum())
    ax.set_xlabel(f'L1 {mass:.3g}   L2 {norm2:.3g}\nSigned sum {signed_sum:.3g}',fontsize=8,labelpad=5)
    cb=ax.figure.colorbar(im,ax=ax,fraction=.035,pad=.025,shrink=.78)
    ticks=[-scale,0,scale] if view=='signed' else [0,scale/2,scale]
    cb.set_ticks(ticks);cb.set_ticklabels(['0' if t==0 else f'{t:.1e}' for t in ticks])
    cb.ax.minorticks_off();cb.ax.tick_params(labelsize=7)
    PANELS.append(dict(group=group,label=label,view=view,raw_array_sha256=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
        raw_shape=list(raw.shape),scale=scale,transform='asinh(value/(scale/9))',rgb_reduction='sum(E)' if view=='signed' else 'sum(abs(E))',
        clipped_pixels=int((np.abs(data)>scale).sum()),absolute_mass=mass,l2_norm=norm2,signed_sum=signed_sum))

def neuron_key(r): return 'm3__'+'__'.join(map(str,(r['task_id'],r['block'],r['token'],r['channel'])))

def main():
    OUT.mkdir(exist_ok=True)
    inv=load(R2/'whole_root_inventory.json')
    for r in inv['records']: bind(R2/r['path'],r['sha256'])
    for root,names in [(M3,['block_summary.csv','cls_patch_summary.csv']),
                       (M4,['attention_block_summary.csv','residual_block_summary.csv']),
                       (M6,['raw_arrays.npz','selection_manifest.json'])]:
        manifest=load(root/'artifact_manifest.json');terminal=load(root/'terminal_receipt.json')
        if 'artifact_manifest_sha256' in terminal: assert sha(root/'artifact_manifest.json')==terminal['artifact_manifest_sha256']
        entries={r['path']:r['sha256'] for r in manifest['files']}
        for name in names:bind(root/name,entries[name])
    r2=np.load(bind(R2/'source_arrays/sloth_bear_display_arrays.npz'))
    arrays=np.load(bind(M6/'raw_arrays.npz'));cases={c['case_id']:c for c in load(M6/'selection_manifest.json')['cases']}
    depth=rows(M3/'block_summary.csv');attention=rows(M4/'attention_block_summary.csv');residual=rows(M4/'residual_block_summary.csv')
    stats=load(OUT/'analysis.json')
    photo=REPO/'assets/data/nns_vit_endpoint_e2_evaluation_20260918/ILSVRC2012_val_00005415.JPEG'
    entry=next(r for r in load(R2/'source_inventory.json')['sources'] if r['path']==str(photo.relative_to(REPO)))
    bind(photo,entry['sha256'])
    img=Image.open(photo).convert('RGB');w,h=img.size
    img=img.resize((248,round(h*248/w)) if w<h else (round(w*248/h),248),Image.Resampling.BICUBIC)
    x=(img.width-224)//2;y=(img.height-224)//2;img=img.crop((x,y,x+224,y+224))
    fig,ax=plt.subplots(1,3,figsize=(10.4,4.8));fig.subplots_adjust(top=.77,bottom=.24,wspace=.35)
    title(fig,1,'From NNS propagation to an Endpoint learning image','One observed image · one registered scalar · the current input Jacobian')
    ax[0].imshow(img);ax[0].axis('off');ax[0].set_title('Input → frozen ViT → class logit')
    s,a=limits([r2['deit_class_gt']])
    map_panel(ax[1],r2['deit_class_gt'],'signed','Signed learning image',s,'F1');map_panel(ax[2],r2['deit_class_gt'],'absolute','Absolute input dependence',a,'F1')
    fig.text(.06,.08,r'$\mathcal{L}(x;x)=\mathrm{diag}(x)\quad\longrightarrow\quad E_s(x)=\mathrm{diag}(x)\nabla_x s(x)$',fontsize=15)
    fig.text(.06,.015,'DeiT-S/16 · ImageNet class 297 (sloth bear) · signed RGB projection and RGB absolute mass',fontsize=9,color=GRAY)
    save(fig,1,'nns_endpoint','NNS identity with signed and absolute class-logit maps.')

    fig,ax=plt.subplots(1,2,figsize=(11,5.3));fig.subplots_adjust(top=.78,bottom=.1,wspace=.22)
    title(fig,2,'Registered targets and complete differential cuts','A scalar target fixes the cotangent; each relevant path crosses the cut once')
    for a0 in ax:a0.axis('off')
    texts=[('Native scalar units','Token × channel; final CLS channel'),('Aggregated responses','Patch sum; activation-weighted energy'),('Structured paths','Heads; Q / K / V; identity / branch'),('Output targets','Ground-truth and frozen clean-top1 logits')]
    for i,(h0,t) in enumerate(texts):
        y=.94-i*.24;ax[0].text(.03,y,h0,weight='bold',fontsize=12);ax[0].text(.03,y-.09,t,fontsize=10,color=GRAY)
    ax[1].text(.02,.93,r'$O=\mathrm{concat}_h(A_hV_h)W_O+b_O$',fontsize=14)
    ax[1].text(.02,.70,r'$\mathrm{d}O_h=(\mathrm{d}A_h)V_h+A_h\mathrm{d}V_h$',fontsize=14)
    ax[1].text(.02,.51,'Q + K = routing; cotangent includes '+r'$W_O$',fontsize=10)
    ax[1].text(.02,.29,r'$y=u+f(u),\quad g_y=\nabla_yF$',fontsize=14)
    ax[1].text(.02,.08,r'$E_y=E_{\mathrm{identity}}+E_{\mathrm{branch}}$',fontsize=14)
    save(fig,2,'registry_and_cuts','Scalar hierarchy, attention output projection and correct residual-output cotangent.')

    fid=load(REPO/'reports/nns_vit_endpoint_m3_r1_20260919/m3a_fidelity.json')
    fig,ax=plt.subplots(1,2,figsize=(11,4.7));fig.subplots_adjust(top=.77,bottom=.2,wspace=.38)
    title(fig,3,'Numerical fidelity and implementation correctness','Independent identities are checked before interpreting finite interventions')
    names=fid['models'];pos=np.arange(len(names))
    for offset,dtype,color in [(-.16,'float64',BLUE),(.16,'float32',ORANGE)]:
        vals=[max(r['relative_l2'] for r in fid['records'] if r['model']==name and r['dtype']==dtype) for name in names]
        ax[0].scatter(pos+offset,vals,label=dtype,color=color,s=45)
    ax[0].set(yscale='log',ylabel='Maximum relative L2 error',xticks=pos,xticklabels=['FCN','CNN','CNN + pool','Residual\nMLP','Residual\nCNN']);ax[0].legend(frameon=False);ax[0].grid(axis='y')
    ax[1].axis('off')
    for i,(n,t) in enumerate([('5,000 / 5,000','NNS–Endpoint comparisons pass'),('76,800 / 76,800','Positive structural closures pass'),('28,800 / 28,800','Omission / duplication controls rejected'),('6,144 / 6,144','Condition-scaled analytic JVP checks pass'),('88 / 88','Operator-direction checks pass')]):
        y=.96-i*.20;ax[1].text(0,y,n,color=BLUE,fontsize=13,weight='bold');ax[1].text(.02,y-.09,t,fontsize=10)
    save(fig,3,'fidelity_and_correctness','Five-network equivalence and independent structural/directional coverage.')

    fig,ax=plt.subplots(1,3,figsize=(11,4.1));fig.subplots_adjust(top=.72,bottom=.19,wspace=.42)
    title(fig,4,'Spatial broadening with depth','Frozen token-channel sample · 100 images per model · image bootstrap 95% intervals')
    for a0,m,l in zip(ax,['entropy','locality','distance'],['Normalized entropy','Own-token locality','Normalized input distance']): curve(a0,depth,m,l)
    save(fig,4,'spatial_broadening','Three complementary spatial trajectories from the frozen stratified sample.')

    fig=plt.figure(figsize=(11,7.7));gs=fig.add_gridspec(2,3,top=.86,bottom=.12,hspace=.5,wspace=.38)
    title(fig,5,'CLS dispersion and signed cancellation','Population trajectories and a preselected high-cancellation scalar unit')
    cls=rows(M3/'cls_patch_summary.csv');a0=fig.add_subplot(gs[0,0:2])
    for model in COLORS:
        rr=sorted([r for r in cls if r['model']==model and r['stratum'].startswith('block_')],key=lambda r:r['stratum'])
        ci=np.array([json.loads(r['ci95']) for r in rr]);xx=np.arange(12)
        a0.plot(xx,[float(r['median_difference']) for r in rr],color=COLORS[model],label=LABELS[model],lw=2);a0.fill_between(xx,ci[:,0],ci[:,1],color=COLORS[model],alpha=.13)
    a0.set(xlabel='Block',ylabel='CLS − patch entropy');a0.legend(frameon=False);a0.grid(axis='y')
    curve(fig.add_subplot(gs[0,2]),depth,'net_ratio','Net / absolute mass')
    unit=cases['m3_high_signed_cancellation']['selected_units'][0];v=arrays[neuron_key(unit)];s,a=limits([v])
    a0=fig.add_subplot(gs[1,0]);a0.imshow(arrays['image__'+unit['task_id']]);a0.axis('off');a0.set_title(f"DeiT-S/16 · B{unit['block']} CLS c{unit['channel']}")
    map_panel(fig.add_subplot(gs[1,1]),v,'signed','Signed projection',s,'F5');map_panel(fig.add_subplot(gs[1,2]),v,'absolute','Absolute mass',a,'F5')
    fig.text(.05,.035,r'$C^{map}=1-|\sum_i E_i|/\sum_i|E_i|$'+f" = {1-abs(v.astype(float).sum())/np.abs(v.astype(float)).sum():.6f}",fontsize=11)
    save(fig,5,'cls_and_cancellation','CLS/patch entropy, surviving signed mass and a signed/absolute cancellation case.')

    fig=plt.figure(figsize=(11,10));gs=fig.add_gridspec(3,3,top=.88,bottom=.09,hspace=.63,wspace=.4)
    title(fig,6,'Deep attention departs from Value-only propagation','Ground-truth logit · 400 images per model · shared scales within the displayed head')
    for j,m,l in [(0,'routing_share','Routing relative L1 scale'),(1,'frozen_attention_endpoint_error','Endpoint relative L2 error'),(2,'frozen_attention_jvp_error','JVP relative L2 discrepancy')]:curve(fig.add_subplot(gs[0,j]),attention,m,l)
    prefix='m4_high_frozen_attention_error';vs=[arrays[prefix+'__'+k] for k in ('routing','value','full')];s,a=limits(vs)
    for j,(key,v) in enumerate(zip(('Routing','Value','Full'),vs)):
        map_panel(fig.add_subplot(gs[1,j]),v,'signed',key+' · signed',s,'F6');map_panel(fig.add_subplot(gs[2,j]),v,'absolute',key+' · absolute',a,'F6')
    head=cases[prefix]['selected_heads'][0];fig.text(.05,.025,f"Preselected: {LABELS[head['model_id']]} · block {head['block']} · head {head['head']} · {head['sample_id']}",fontsize=9)
    save(fig,6,'attention_routing_value','Depth curves with a shared-scale signed/absolute routing, value and full head example.')

    fig=plt.figure(figsize=(11,10));gs=fig.add_gridspec(3,3,top=.88,bottom=.09,hspace=.63,wspace=.4)
    title(fig,7,'Attention and MLP residuals have distinct depth trends','Relative path scales and spatial TV; identity + branch reconstructs the same target')
    curve(fig.add_subplot(gs[0,0]),residual,'identity_share','Attention identity scale',residual_type='attention')
    curve(fig.add_subplot(gs[0,1]),residual,'identity_share','MLP identity scale',residual_type='mlp')
    a0=fig.add_subplot(gs[0,2])
    for model in COLORS:
        for typ,style in [('attention','-'),('mlp','--')]:
            rr=sorted([r for r in residual if r['model_id']==model and r['residual_type']==typ and r['metric']=='output_identity_patch_tv'],key=lambda r:int(r['block']))
            assert len(rr)==12
            a0.plot(range(12),[float(r['median']) for r in rr],style,color=COLORS[model],label=LABELS[model]+' '+typ,lw=2)
    a0.set(xlabel='Block',ylabel='Output–identity patch TV');a0.legend(frameon=False,fontsize=7);a0.grid(axis='y')
    prefix='m4_strong_path_cancellation';vs=[arrays[prefix+'__'+k] for k in ('identity','branch','output')];s,a=limits(vs)
    for j,(key,v) in enumerate(zip(('Identity','Branch','Output'),vs)):
        map_panel(fig.add_subplot(gs[1,j]),v,'signed',key+' · signed',s,'F7');map_panel(fig.add_subplot(gs[2,j]),v,'absolute',key+' · absolute',a,'F7')
    r=cases[prefix]['selected_residuals'][0];fig.text(.05,.025,f"Preselected cancellation example: {LABELS[r['model_id']]} · block {r['block']} · {r['residual_type']} residual",fontsize=9)
    save(fig,7,'residual_identity_branch','Residual depth trends and a common-scale high-cancellation identity/branch example.')

    units=sorted(cases['m3_model_layer_difference']['selected_units'],key=lambda r:(r['model_id'],r['block']))
    fig,ax=plt.subplots(4,3,figsize=(10.5,12));fig.subplots_adjust(top=.86,bottom=.07,hspace=.6,wspace=.32)
    title(fig,8,'One image, two models, early and late scalar targets','Representative depth contrast · independently registered channels are not semantic matches')
    for i,r in enumerate(units):
        same=[arrays[neuron_key(u)] for u in units if u['model_id']==r['model_id']];s,a=limits(same);v=arrays[neuron_key(r)]
        ax[i,0].imshow(arrays['image__'+r['task_id']]);ax[i,0].axis('off');ax[i,0].set_title(f"{LABELS[r['model_id']]} · B{r['block']}\nToken {r['token']} · channel {r['channel']}")
        map_panel(ax[i,1],v,'signed','Signed learning image',s,'F8-'+r['model_id']);map_panel(ax[i,2],v,'absolute',f"Absolute mass · H={r['normalized_spatial_entropy']:.3f}",a,'F8-'+r['model_id'])
    save(fig,8,'representative_atlas','Four scalar targets on one frozen image, with shared early/late scales within each model.')

    families=load(REPO/'reports/nns_vit_endpoint_e6_release_r3_20260918/family_decisions.json')['families']
    fig,ax=plt.subplots(1,2,figsize=(11.5,5.7),gridspec_kw={'width_ratios':[1.6,1]});fig.subplots_adjust(top=.8,bottom=.16,wspace=.5)
    title(fig,9,'Component association and displacement-matched prediction','Point thresholds and direction tests are different conditions')
    ff=[r for r in families if r['claim']=='C4a']
    for i,r in enumerate(ff):
        t=r['test_results'][0];color=COLORS[r['model_id']]
        ax[0].errorbar(t['estimate'],i,xerr=[[t['estimate']-t['ci_lower']],[t['ci_upper']-t['estimate']]],fmt='o' if r['status']=='SUPPORTED' else 's',color=color,ms=5,capsize=2)
    ax[0].set(yticks=range(12),yticklabels=[LABELS[r['model_id']]+' · '+('zero' if r['reference_id']=='normalized_zero' else 'blur')+' · '+r['component_family'].replace('_',' ') for r in ff],xlabel='Median Spearman with finite component effect');ax[0].tick_params(axis='y',labelsize=8);ax[0].axvline(.1,color=GRAY,ls='--',lw=1);ax[0].invert_yaxis();ax[0].grid(axis='x')
    p=bind(REPO/'reports/nns_vit_endpoint_m1_r1_20260918/summary.json.gz');summ=json.load(gzip.open(p,'rt'))
    groups=[(m,r) for m in COLORS for r in ('normalized_zero','blur_rgb')]
    for j,(method,col) in enumerate([('endpoint',BLUE),('taylor',ORANGE)]):
        vals=[next(r['median'] for r in summ if (r['model'],r['reference'],r['family'],r['stratum'],r['predictor'],r['metric'])==(m,ref,'head','within_block_mean',method,'spearman')) for m,ref in groups]
        ax[1].bar(np.arange(4)+(j-.5)*.34,vals,.32,color=col,label=method.title())
    ax[1].set(xticks=range(4),xticklabels=['ViT\nzero','ViT\nblur','DeiT\nzero','DeiT\nblur'],ylabel='Within-block head Spearman',ylim=(0,.8));ax[1].legend(frameon=False);ax[1].grid(axis='y');ax[1].set_axisbelow(True)
    fig.text(.05,.035,'11/12 families meet the original rule; the all-family criterion remains unmet. Right: supplementary frozen comparison.',fontsize=9)
    save(fig,9,'component_calibration','Original C4a family intervals and separately frozen within-block Endpoint/Taylor head comparisons.')

    fig=plt.figure(figsize=(11,8));gs=fig.add_gridspec(2,3,top=.85,bottom=.1,hspace=.6,wspace=.42)
    title(fig,10,'Spatial intervention: effect scale and estimator boundary','Registered block-5 task · 0–25% area · isolated head replacement · 400 images per family')
    a0=fig.add_subplot(gs[0,:2]);groups=[(m,r) for m in COLORS for r in ('normalized_zero','blur_rgb')]
    for j,(method,label,col) in enumerate([('endpoint_component','Endpoint',BLUE),('regional_taylor','Regional Taylor',ORANGE),('random_regions','Random','#8895A5'),('empirical_oracle','Candidate oracle','#45877A')]):
        vals=[next(r['auc']['median'] for r in stats['spatial']['summary'] if (r['model'],r['reference'],r['method'])==(m,ref,method)) for m,ref in groups]
        a0.bar(np.arange(4)+(j-1.5)*.19,vals,.18,color=col,label=label)
    a0.set(xticks=range(4),xticklabels=['ViT-B · zero','ViT-B · blur','DeiT-S · zero','DeiT-S · blur'],ylabel='Absolute-effect AUC (logit units)');a0.legend(frameon=False,fontsize=8,ncols=2);a0.grid(axis='y');a0.set_axisbelow(True)
    a1=fig.add_subplot(gs[0,2]);a1.axis('off')
    for y,t in [(.85,'Identity'),(.46,'Correctness')]:a1.text(.02,y,t,fontsize=13,weight='bold',color=BLUE)
    a1.text(.02,.67,'NNS = local Jacobian seed',fontsize=9);a1.text(.02,.28,'Closed cuts; direction checks',fontsize=9);a1.text(.02,.07,'Utility: measured separately',fontsize=10,weight='bold')
    v=r2['e5_b5_head_0'].squeeze();s,a=limits([v]);map_panel(fig.add_subplot(gs[1,0]),v,'signed','Head 0 · signed Endpoint',s,'F10');map_panel(fig.add_subplot(gs[1,1]),v,'absolute','Head 0 · absolute Endpoint',a,'F10')
    e5=load(REPO/'reports/nns_vit_endpoint_e5_spatial_r1_20260918/formal_results/attempts/deit_s16__014/attempt-01.json')['result']
    score=np.array(e5['references']['normalized_zero']['ranking_scores']['regional_taylor']['head_0']).reshape(4,4);a1=fig.add_subplot(gs[1,2]);im=a1.imshow(score,cmap='magma',interpolation='none');a1.set_title('Regional Taylor magnitude\n16 candidate regions',fontsize=10);a1.set(xticks=[],yticks=[]);fig.colorbar(im,ax=a1,fraction=.035,pad=.025,shrink=.78).ax.tick_params(labelsize=7)
    fig.text(.05,.025,'Same sloth-bear image, DeiT-S/16, GT logit, normalized-zero reference. Taylor colors encode region scores, not pixel Endpoint.',fontsize=8.5)
    save(fig,10,'spatial_intervention','Absolute AUC including evaluated-candidate oracle, signed/absolute Endpoint and regional Taylor magnitude.')

    old=PAPER/'figure_manifest_all_figures_r2.json'
    if not old.exists():old.write_bytes((PAPER/'figure_manifest.json').read_bytes())
    historical=json.loads(old.read_text());supp=[]
    for f in historical['figures']:supp.append(dict(group='archived_r2_main',old_number=f['figure'],outputs=[f[k] for k in ('png','pdf','svg')]))
    supp.extend(historical['supplemental_records'])
    manifest=dict(source_revision='ENDPOINT-PAPER-REVISION-V1',figure_count=10,figures=FIGURES,supplemental_records=supp,
      source_package='reports/nns_vit_endpoint_all_figures_r2_20260919',source_package_preserved=True,main_text_recommended=list(range(1,11)))
    write(PAPER/'figure_manifest.json',manifest);write(OUT/'manifest.json',manifest);write(OUT/'figure_sources.json',SOURCES);write(OUT/'display_receipt.json',dict(panels=PANELS,shared_scale_groups=True,scientific_arrays_unchanged=True))
    cards=''.join(f'<article><h2>Figure {f["figure"]}: {f["slug"].replace("_"," ")}</h2><img src="figures_png/{Path(f["png"]["path"]).name}"><p>{f["description"]}</p><a href="figures_pdf/{Path(f["pdf"]["path"]).name}">PDF</a> · <a href="figures_svg/{Path(f["svg"]["path"]).name}">SVG</a></article>' for f in FIGURES)
    (OUT/'gallery.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Endpoint paper revision</title><style>body{font:16px system-ui;background:#edf2f6;color:#172b42;max-width:1200px;margin:40px auto;padding:20px}article{background:white;padding:28px;margin:30px 0;border-radius:12px}img{width:100%;height:auto}h1{font-size:32px}h2{font-size:20px}a{color:#2372a4}</style><h1>From NNS to Vision Transformers</h1><p>Ten publication figures · signed and absolute maps · preserved R2 sources</p>'+cards+'</html>')
    assert not any(m in sys.modules for m in ('torch','timm'))
    print(json.dumps(dict(figures=len(FIGURES),panels=len(PANELS),sources=len(SOURCES))))

if __name__=='__main__':main()
