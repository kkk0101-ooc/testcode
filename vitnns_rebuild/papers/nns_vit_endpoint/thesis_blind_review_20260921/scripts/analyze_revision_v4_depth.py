#!/usr/bin/env python3
"""CPU-only post-hoc depth reanalysis of sealed Endpoint numeric records."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
V3 = REPO / 'reports/nns_vit_endpoint_revision_v3_step2_r2_20260921'
M4 = REPO / 'reports/nns_vit_endpoint_m4_r1_20260919'
OUT = PACKAGE / 'evidence/revision_v4'
FIG = PACKAGE / 'figures/revision_v4'
NS = 'ENDPOINT_V4_DEPTH_20260921'
B = 10000
SOURCES = {}
MANIFESTS = {}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    return json.loads(p.read_text())

def write(p, a):
    p.write_text(json.dumps(a, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def sealed(root, rel):
    if root not in MANIFESTS:
        terminal = load(root/'terminal_receipt.json')
        assert 'PASS' in terminal['status'], terminal
        assert sha(root/'artifact_manifest.json') == terminal['artifact_manifest_sha256']
        MANIFESTS[root] = {r['path']:r for r in load(root/'artifact_manifest.json')['files']}
        for q in ('terminal_receipt.json','artifact_manifest.json'):
            SOURCES[str((root/q).relative_to(REPO))] = sha(root/q)
    p = root/rel
    h = sha(p)
    assert h == MANIFESTS[root][rel]['sha256'], p
    SOURCES[str(p.relative_to(REPO))] = h
    return p

def read_csv(p):
    with p.open() as f:
        return list(csv.DictReader(f))

def dump(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def valid_median(a):
    a=np.asarray(a,dtype=float); a=a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan

def summarize(values, key):
    a=np.asarray(values,dtype=float); n_total=len(a); a=np.sort(a[np.isfinite(a)]); n=len(a)
    seed=int.from_bytes(hashlib.sha256((NS+'|'+key).encode()).digest()[:8],'big')
    if not n:
        return dict(n_total=n_total,n_valid=0,n_missing=n_total,median=None,ci95_low=None,ci95_high=None,seed=seed)
    rng=np.random.Generator(np.random.PCG64(seed))
    if n%2:
        u=rng.beta((n+1)//2,(n+1)//2,B); draws=a[np.minimum((n*u).astype(int),n-1)]
    else:
        u=rng.beta(n//2,n//2+1,B);v=u+(1-u)*rng.beta(1,n//2,B)
        draws=(a[np.minimum((n*u).astype(int),n-1)]+a[np.minimum((n*v).astype(int),n-1)])/2
    lo,hi=np.quantile(draws,[.025,.975],method='linear')
    return dict(n_total=n_total,n_valid=n,n_missing=n_total-n,median=float(np.median(a)),ci95_low=float(lo),ci95_high=float(hi),seed=seed)

def sign(a):
    a=np.asarray(a,dtype=float); a=a[np.isfinite(a)]
    pos=int((a>1e-12).sum());neg=int((a< -1e-12).sum());n=pos+neg
    p=1. if not n else min(1.,2*math.fsum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n)
    return dict(positive=pos,negative=neg,ties=len(a)-n,p_two_sided=p)

def holm(rows):
    previous=0.
    for i,r in enumerate(sorted(rows,key=lambda r:(r['p_two_sided'],r['block']))):
        previous=max(previous,min(1.,(len(rows)-i)*r['p_two_sided']))
        r['holm_p']=previous;r['holm_reject_005']=previous<=.05

def collect():
    series=defaultdict(dict); units=[]
    def put(model,group,metric,sample,block,value):
        key=(model,group,metric)
        a=series[key].setdefault(sample,np.full(12,np.nan))
        assert not np.isfinite(a[block]), (key,sample,block)
        a[block]=np.nan if value in ('',None) else float(value)
    qmetrics=('absolute_mass','normalized_spatial_entropy','normalized_pairwise_distance','cancellation','relative_scale')
    for r in read_csv(sealed(V3,'qkv_image_values.csv')):
        for metric in qmetrics: put(r['model_id'],'qkv_'+r['path'],metric,r['sample_id'],int(r['block']),r[metric])
    for r in read_csv(sealed(M4,'image_values.csv')):
        keep=(r['domain']=='attention' and r['metric'] in ('routing_share','frozen_attention_endpoint_error','routing_value_path_cancellation','routing_value_cosine')) or (r['domain']=='residual' and r['metric'] in ('identity_share','path_cancellation','output_identity_patch_tv'))
        if keep: put(r['model_id'],r['domain']+'_'+r['group'],r['metric'],r['sample_id'],int(r['block']),r['value'])
    for r in read_csv(sealed(M4,'m4_a1_image_values.csv')):
        if r['metric']=='absolute_mass' and (r['domain']=='residual' or r['group'] in ('routing','value','full')):
            put(r['model_id'],r['domain']+'_'+r['group'],'absolute_mass',r['sample_id'],int(r['block']),r['value'])
    tasks=load(sealed(V3,'task_manifest.json'))['tasks']
    for t in tasks:
        if t['payload']['kind']!='sampling':continue
        rel='results/attempts/'+t['task_id']+'/attempt-01.json'
        r=load(sealed(V3,rel))['result']
        p=REPO/r['sampling_path'];assert sha(p)==r['sampling_sha256']
        # An inherited payload may live in an earlier root; its sealed result binds exact bytes.
        SOURCES[str(p.relative_to(REPO))]=sha(p)
        with np.load(p,allow_pickle=False) as data:
            for gi in range(5):
                for b in range(12):
                    mask=(data['group_index']==gi)&(data['block']==b)
                    for metric in ('normalized_spatial_entropy','own_token_locality_above_uniform','input_dependency_distance'):
                        a=data[metric][mask].astype(float)
                        put(r['model_id'],'sample_'+str(gi+1),metric,r['sample_id'],b,valid_median(a))
                        units.append(dict(model=r['model_id'],group='sample_'+str(gi+1),metric=metric,sample_id=r['sample_id'],block=b,n_units=len(a),n_valid=int(np.isfinite(a).sum()),n_missing=int((~np.isfinite(a)).sum())))
    return series,units

LABELS={'normalized_spatial_entropy':'Spatial entropy','own_token_locality_above_uniform':'Locality above uniform','input_dependency_distance':'Input distance','absolute_mass':'Absolute mass','normalized_pairwise_distance':'Pairwise distance','cancellation':'Map cancellation','relative_scale':'Q/K/V relative scale','routing_share':'Routing relative scale','frozen_attention_endpoint_error':'Value-only relative error','routing_value_path_cancellation':'Route/value cancellation','routing_value_cosine':'Route/value cosine','identity_share':'Identity relative scale','path_cancellation':'Residual cancellation','output_identity_patch_tv':'Output/identity spatial TV'}

def draw_trajectory(ax,key,series,blocks):
    model,group,metric=key; rows=blocks[key]
    ids=sorted(series[key],key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:12]
    for sample in ids:ax.plot(range(12),series[key][sample],color='#9fabb8',alpha=.26,lw=.65)
    med=np.array([r['median'] for r in rows]);lo=[r['ci95_low'] for r in rows];hi=[r['ci95_high'] for r in rows]
    color='#2166ac' if model=='vit_b16' else '#b85c38'
    ax.fill_between(range(12),lo,hi,color=color,alpha=.2)
    ax.plot(range(12),med,'o-',color=color,ms=3,lw=1.8)
    ax.set(title=model+' | '+group,ylabel=LABELS[metric],xticks=range(12),xlabel='Encoder block')
    ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)

def figures(series,blockmap,adjmap):
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'pdf.fonttype':42})
    def sheet(name,keys):
        fig,axes=plt.subplots(len(keys)//2,2,figsize=(11,2.5*(len(keys)//2)),squeeze=False)
        for ax,key in zip(axes.flat,keys):draw_trajectory(ax,key,series,blockmap)
        fig.suptitle('Post-hoc depth profiles: image median, 95% bootstrap CI, 12 fixed image trajectories',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(FIG/(name+'.png'),dpi=180);fig.savefig(FIG/(name+'.pdf'));plt.close(fig)
    models=['vit_b16','deit_s16']
    for g in range(1,6):
        sheet('sample_'+str(g),[(m,'sample_'+str(g),metric) for metric in ('normalized_spatial_entropy','own_token_locality_above_uniform','input_dependency_distance') for m in models])
    for metric in ('absolute_mass','normalized_spatial_entropy','normalized_pairwise_distance','cancellation','relative_scale'):
        sheet('qkv_'+metric,[(m,'qkv_'+p,metric) for p in 'QKV' for m in models])
    sheet('attention',[(m,'attention_heads',metric) for metric in ('routing_share','frozen_attention_endpoint_error','routing_value_path_cancellation','routing_value_cosine') for m in models])
    for group in ('attention','mlp'):
        sheet('residual_'+group,[(m,'residual_'+group,metric) for metric in ('identity_share','path_cancellation','output_identity_patch_tv') for m in models])
    with PdfPages(FIG/'all_profiles_and_adjacent_differences.pdf') as pdf:
        keys=sorted(series)
        for start in range(0,len(keys),3):
            fig,axes=plt.subplots(3,2,figsize=(11.7,10.5))
            for row,key in enumerate(keys[start:start+3]):
                draw_trajectory(axes[row,0],key,series,blockmap)
                a=adjmap[key];ax=axes[row,1]
                med=np.array([r['median'] for r in a]);lo=np.array([r['ci95_low'] for r in a]);hi=np.array([r['ci95_high'] for r in a])
                ax.vlines(range(11),lo,hi,color='#555555');ax.scatter(range(11),med,c=['#2166ac' if r['holm_reject_005'] else '#aaaaaa' for r in a],s=17)
                ax.axhline(0,color='black',lw=.7);ax.set(xticks=range(11),xlabel='b in block (b+1) minus block b',ylabel='Paired image median difference',title='95% pointwise CI; blue: Holm p <= 0.05 across 11 steps');ax.grid(alpha=.15)
            for row in range(len(keys[start:start+3]),3):
                for ax in axes[row]:ax.axis('off')
            fig.suptitle('POST-HOC DESCRIPTIVE REANALYSIS | models and sampling groups kept separate');fig.tight_layout(rect=(0,0,1,.96));pdf.savefig(fig);plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    series,units=collect();print('source series',len(series),flush=True)
    profiles=[];blocks=[];contrasts=[];trends=[];adj=[];coverage=[];trend_images=[]
    blockmap={};adjmap={}
    for key,items in sorted(series.items()):
        model,group,metric=key;base=dict(model=model,group=group,metric=metric);prefix='|'.join(key)
        ids=sorted(items);arr=np.array([items[s] for s in ids]);expected=100 if group.startswith('sample_') else 400
        assert len(ids)==expected,(key,len(ids)); assert arr.shape==(expected,12)
        coverage.append(dict(**base,n_images=expected,n_values=arr.size,n_valid=int(np.isfinite(arr).sum()),n_missing=int((~np.isfinite(arr)).sum())))
        for sample,a in zip(ids,arr):
            profiles.append(dict(**base,sample_id=sample,**{f'block_{b}':None if not np.isfinite(a[b]) else float(a[b]) for b in range(12)}))
        blockmap[key]=[]
        for b in range(12):
            r=dict(**base,block=b,**summarize(arr[:,b],prefix+'|block_'+str(b)));blocks.append(r);blockmap[key].append(r)
        phases={name:np.array([valid_median(a[bs]) if np.isfinite(a[bs]).all() else np.nan for a in arr]) for name,bs in [('early',slice(0,4)),('middle',slice(4,8)),('late',slice(8,12))]}
        for name,a in phases.items():contrasts.append(dict(**base,contrast=name,**summarize(a,prefix+'|'+name)))
        for a,b in [('middle','early'),('late','middle'),('late','early')]:
            name=a+'_minus_'+b;contrasts.append(dict(**base,contrast=name,**summarize(phases[a]-phases[b],prefix+'|'+name)))
        contrasts.append(dict(**base,contrast='block11_minus_block0',**summarize(arr[:,11]-arr[:,0],prefix+'|endpoint')))
        rhos=[]
        for sample,a in zip(ids,arr):
            rho=np.nan if not np.isfinite(a).all() or np.ptp(a)==0 else float(np.corrcoef(np.arange(12),rankdata(a,method='average'))[0,1])
            rhos.append(rho);trend_images.append(dict(**base,sample_id=sample,spearman=None if not np.isfinite(rho) else rho))
        r=dict(**base,**summarize(rhos,prefix+'|spearman'),positive_images=int((np.array(rhos)>0).sum()),negative_images=int((np.array(rhos)<0).sum()),nondecreasing_images=int(np.all(np.diff(arr,axis=1)>=0,axis=1).sum()),nonincreasing_images=int(np.all(np.diff(arr,axis=1)<=0,axis=1).sum()))
        trends.append(r);adjmap[key]=[]
        for b in range(11):
            d=arr[:,b+1]-arr[:,b];r=dict(**base,block=b,**summarize(d,prefix+'|adjacent_'+str(b)),**sign(d));adjmap[key].append(r)
        holm(adjmap[key]);adj.extend(adjmap[key])
    for name,rows in [('image_profiles.csv',profiles),('block_summary.csv',blocks),('phase_endpoint_contrasts.csv',contrasts),('image_trends.csv',trend_images),('trend_summary.csv',trends),('adjacent_differences.csv',adj),('coverage.csv',coverage),('sampling_unit_coverage.csv',units)]:dump(name,rows)
    figures(series,blockmap,adjmap)
    write(OUT/'analysis_receipt.json',dict(status='PASS',analysis_type='post-hoc descriptive reanalysis',new_model_inference=0,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],series=len(series),profile_rows=len(profiles),block_rows=len(blocks),adjacent_rows=len(adj),bootstrap_replicates=B,seed_namespace=NS,bootstrap='exact marginal empirical-bootstrap median law via beta order statistics',quantiles='linear 0.025, 0.975',holm='11 two-sided sign tests per model/group/metric',source_sha256=SOURCES,script_sha256=sha(Path(__file__)),plan_sha256=sha(PACKAGE/'notes/revision_v4_plan.md'),output_sha256={p.name:sha(p) for p in sorted(OUT.glob('*.csv'))}))
    print(json.dumps({'series':len(series),'profiles':len(profiles),'missing':sum(r['n_missing'] for r in coverage),'source_files':len(SOURCES)}),flush=True)

if __name__=='__main__':main()
