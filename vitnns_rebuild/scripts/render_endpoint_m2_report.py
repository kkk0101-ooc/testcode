#!/usr/bin/env python3
"""Data-bound M2 figures and tables; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast
import csv
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_endpoint_m2 import ROOT,REPO,load,sha,write
sys.path.insert(0,str(REPO/'src'))
from vitnns.endpoint_m2_regions import BLOCKS,HEADS,REFERENCES,METHODS,COUNTS

def read_csv(name):
    with (ROOT/name).open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        r['median']=float(r['median']);r['ci95']=ast.literal_eval(r['ci95'])
        if r['block']!='all_blocks_descriptive':r['block']=int(r['block'])
    return rows

def main():
    rows=read_csv('summary.csv');paired=read_csv('paired.csv');heads=read_csv('head_summary.csv')
    tests=load(ROOT/'auc_tests.json')['tests']
    idx={(r['model'],r['block'],r['geometry'],r['reference'],r['outcome'],r['metric'],r['method']):r for r in rows}
    pidx={(r['model'],r['block'],r['geometry'],r['reference'],r['outcome'],r['metric'],r['baseline']):r for r in paired}
    tidx={(r['model'],r['block'],r['geometry'],r['reference'],r['outcome'],r['baseline'],r['direction']):r for r in tests}
    colors={'endpoint_absolute':'#c44536','regional_taylor':'#276a9b','random':'#777777','endpoint_signed':'#4c956c'}
    labels={'endpoint_absolute':'Endpoint absolute','regional_taylor':'Regional Taylor','random':'Random','endpoint_signed':'Endpoint signed aggregate'}
    figures=[]
    for number,outcome in enumerate(('full','isolated'),1):
        fig,axes=plt.subplots(4,3,figsize=(12,12),sharex=True,layout='constrained')
        for row,(model,ref) in enumerate((m,r) for m in HEADS for r in REFERENCES):
            for col,b in enumerate(BLOCKS):
                ax=axes[row,col];x=np.array(COUNTS['patch'])/196*100
                for method in colors:
                    data=[idx[(model,b,'patch',ref,outcome,'effect_'+str(c),method)] for c in COUNTS['patch']]
                    ax.plot(x,[r['median'] for r in data],marker='o',ms=3,label=labels[method],color=colors[method])
                    ax.fill_between(x,[r['ci95'][0] for r in data],[r['ci95'][1] for r in data],alpha=.1,color=colors[method])
                ax.set_title(f'{model} | {ref} | block {b}',fontsize=9)
                ax.grid(alpha=.2);ax.set_ylim(bottom=0)
                if col==0:ax.set_ylabel('Median absolute logit effect')
                if row==3:ax.set_xlabel('Input area replaced (%)')
        handles,labels0=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels0,loc='outside lower center',ncol=4,fontsize=9)
        fig.suptitle('M2: '+('full-input region replacement' if outcome=='full' else 'isolated-head region replacement')+' (400 images/model)',fontsize=13)
        name='m2_'+outcome+'_effect_budget'
        for suffix in ('png','pdf'):fig.savefig(ROOT/(name+'.'+suffix),dpi=180,bbox_inches='tight')
        plt.close(fig)
        figures.append(dict(id=f'M2_F{number}',path=(ROOT/(name+'.png')).relative_to(REPO).as_posix(),
            sha256=sha(ROOT/(name+'.png')),outcome=outcome,source='summary.csv'))
    fig,axes=plt.subplots(2,2,figsize=(12,6),layout='constrained')
    for i,model in enumerate(HEADS):
        for j,ref in enumerate(REFERENCES):
            values=[]
            for b in BLOCKS:
                line=[]
                for h in range(HEADS[model]):
                    a={r['method']:r['median'] for r in heads if r['model']==model and r['block']==b
                        and int(r['head'])==h and r['geometry']=='patch' and r['reference']==ref
                        and r['outcome']=='full' and r['metric']=='auc'}
                    line.append(a['endpoint_absolute']-a['regional_taylor'])
                values.append(line)
            data=np.array(values);limit=max(float(np.abs(data).max()),1e-8)
            im=axes[i,j].imshow(data,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
            axes[i,j].set_yticks(range(3),[str(b) for b in BLOCKS]);axes[i,j].set_xticks(range(HEADS[model]))
            axes[i,j].set_xlabel('Head (0-based)');axes[i,j].set_ylabel('Block (0-based)')
            axes[i,j].set_title(f'{model} | {ref}',fontsize=10);fig.colorbar(im,ax=axes[i,j],label='Difference of median AUCs')
    fig.suptitle('M2 full-input: head-specific Endpoint minus Taylor (descriptive)')
    for suffix in ('png','pdf'):fig.savefig(ROOT/('m2_head_contrasts.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    figures.append(dict(id='M2_F3',path=(ROOT/'m2_head_contrasts.png').relative_to(REPO).as_posix(),
        sha256=sha(ROOT/'m2_head_contrasts.png'),outcome='full',source='head_summary.csv'))
    write(ROOT/'figure_manifest.json',dict(figures=figures))
    paper=REPO/'papers/nns_vit_endpoint'
    write(paper/'figure_manifest_m2.json',dict(stage='M2 supplementary',figures=figures))
    tables={}
    for outcome in ('full','isolated'):
        lines=['| Model | Reference | Block | Endpoint AUC | Taylor AUC | Random AUC | Paired E−T [95% CI] | Holm p(E>T) | Holm p(T>E) |',
            '|---|---|---:|---:|---:|---:|---|---:|---:|']
        for model in HEADS:
            for ref in REFERENCES:
                for b in BLOCKS:
                    key=(model,b,'patch',ref,outcome)
                    vals=[idx[(*key,'auc',method)]['median'] for method in ('endpoint_absolute','regional_taylor','random')]
                    pair=pidx[(*key,'auc','regional_taylor')]
                    p1=tidx[(*key,'regional_taylor','endpoint_greater')]['holm_p']
                    p2=tidx[(*key,'regional_taylor','baseline_greater')]['holm_p']
                    lines.append(f"| {model} | {ref} | {b} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {pair['median']:.4f} [{pair['ci95'][0]:.4f}, {pair['ci95'][1]:.4f}] | {p1:.3g} | {p2:.3g} |")
        tables[outcome]='\n'.join(lines)
    counts={}
    for outcome in ('full','isolated'):
        part=[t for t in tests if t['geometry']=='patch' and t['outcome']==outcome and t['baseline']=='regional_taylor']
        counts[outcome]={direction:sum(t['reject'] for t in part if t['direction']==direction)
            for direction in ('endpoint_greater','baseline_greater')}
    lines=['| Outcome | Baseline | Endpoint greater / 12 | Baseline greater / 12 |','|---|---|---:|---:|']
    for outcome in ('full','isolated'):
        for method in METHODS[1:]:
            part=[t for t in tests if t['geometry']=='patch' and t['outcome']==outcome and t['baseline']==method]
            a=sum(t['reject'] for t in part if t['direction']=='endpoint_greater')
            b=sum(t['reject'] for t in part if t['direction']=='baseline_greater')
            lines.append(f'| {outcome} | {method} | {a} | {b} |')
    tables['baselines']='\n'.join(lines)
    write(ROOT/'report_data.json',dict(scientific_status='SUPPLEMENTARY_LAYER_REFERENCE_OUTCOME_SPECIFIC',
        primary_C4a='NOT_SUPPORTED',primary_C4b='NOT_SUPPORTED',patch_taylor_comparisons=counts,
        tables=tables,task_validation_sha256=sha(ROOT/'task_validation.json'),
        inference_validation_sha256=sha(ROOT/'inference_validation.json')))
    print(json.dumps(dict(figures=3,patch_taylor=counts)),flush=True)

if __name__=='__main__':main()
