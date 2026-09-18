#!/usr/bin/env python3
"""Exhaustive CPU audit of M2 coverage, selections, outcomes and inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import numpy as np
from scipy.stats import binomtest
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from run_endpoint_m2 import ROOT,SPEC,load,sha,write
from vitnns.endpoint_ep1_queue import digest,audit_coverage
from vitnns.endpoint_m2_regions import BLOCKS,HEADS,REFERENCES,METHODS,COUNTS,GRIDS,region_orders
from vitnns.endpoint_m2_statistics import summarize
from vitnns.endpoint_m1_statistics import bootstrap_median

def require(test,code):
    if not test: raise ValueError(code)
def close(a,b,code): require(math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-10),code)
def record(path,value):
    if path.exists():require(load(path)==value,'AUDIT_DRIFT:'+path.name)
    else:write(path,value)

def audit_task(task):
    tid=task['task_id'];payload=task['payload']
    path=ROOT/'results/attempts'/tid/'attempt-01.json';d=load(path)
    expected=d.pop('canonical_digest');require(digest(d)==expected,'CANONICAL')
    r=d['result'];h=HEADS[payload['model_id']]
    for key in ('model_id','sample_id','split_order','class_id','block','image_sha256'):
        require(r[key]==payload[key],'IDENTITY:'+key)
    require(set(r['reference_identity'])==set(REFERENCES),'REFERENCES')
    identity=r['worker_identity']
    require(identity['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and identity['deterministic_algorithms']
        and not identity['cuda_matmul_allow_tf32'] and not identity['cudnn_allow_tf32']
        and not identity['autocast'] and identity['startup_matmul_backward']=='PASS','NUMERICS')
    require(identity['sys_executable_realpath']==str(Path(sys.executable).resolve()),'PYTHON')
    require(r['model_binding']['checkpoint_sha256']==payload['checkpoint_sha256']
        and r['model_binding']['tensor_digest']==payload['tensor_digest'],'MODEL_BINDING')
    require(len(r['checks'])==1+4*h and len(r['e3_parity'])==h,'CONTROL_COVERAGE')
    require(r['closure_error']>=0 and math.isfinite(r['closure_error']),'CLOSURE')
    for c in r['checks']+r['e3_parity']:
        require(c['error']<=1e-4+1e-4*max(abs(c['expected']),abs(c['observed'])),'CONTROL_TOLERANCE')
    for name in ('e3','e4'):
        require(sha(REPO/payload[name+'_path'])==payload[name+'_sha256'],'SOURCE_SHA')
    scores={(s['geometry'],s['reference'],s['head']):s['scores'] for s in r['scores']}
    candidates={(s['geometry'],s['reference'],s['head']):s['candidates'] for s in r['candidates']}
    expected_groups={(g,ref,head) for g in GRIDS for ref in REFERENCES for head in range(h)}
    require(set(scores)==set(candidates)==expected_groups,'SCORE_GROUPS')
    require(len(r['scores'])==len(r['candidates'])==4*h,'GROUP_DUPLICATE')
    curve_index={}
    for curve in r['curves']:
        key=(curve['geometry'],curve['reference'],curve['head'],curve['method'],curve['repeat'])
        require(key not in curve_index,'DUP_CURVE');curve_index[key]=curve
    require(len(curve_index)==h*4*27,'CURVE_COUNT')
    points=0; candidate_count=0
    for (geometry,ref,head),values in scores.items():
        n=GRIDS[geometry];counts=COUNTS[geometry]
        require(set(values)=={'endpoint_absolute','endpoint_signed','signed_raw','regional_taylor',
            'gradient_only','displacement_only','attention'},'SCORE_METHODS')
        for method,v in values.items():
            require(len(v)==n*n and np.isfinite(v).all(),'SCORE_FINITE')
            if method!='signed_raw':require(min(v)>=0,'NEGATIVE_MASS')
        require(values['endpoint_signed']==[abs(v) for v in values['signed_raw']],'SIGNED_SCORE')
        key=f"{r['sample_id']}|{r['model_id']}|{r['block']}|{head}|{ref}"
        orders=region_orders(values,geometry,key)
        selected={(i,) for i in range(n*n)}
        selected|={tuple(sorted(o['order'][:c])) for o in orders for c in counts}
        cache={tuple(c['selected']):c for c in candidates[(geometry,ref,head)]}
        require(set(cache)==selected and len(cache)==len(candidates[(geometry,ref,head)]),'EXACT_CANDIDATES')
        candidate_count+=len(cache)
        for selection,c in cache.items():
            require(tuple(sorted(set(selection)))==selection and min(selection)>=0 and max(selection)<n*n,'MASK_SELECTION')
            require(math.isfinite(c['full_logit']) and math.isfinite(c['isolated_logit']),'LOGIT_FINITE')
        best={c:{out:max(abs(r['clean_score']-v[out+'_logit']) for s,v in cache.items() if len(s)==c)
            for out in ('full','isolated')} for c in counts}
        for o in orders:
            curve=curve_index[(geometry,ref,head,o['method'],o['repeat'])]
            require(curve['seed']==o['seed'] and len(curve['points'])==len(counts),'SEED_POINTS')
            for count,p in zip(counts,curve['points']):
                s=tuple(sorted(o['order'][:count]));require(p['selected']==list(s) and p['count']==count,'RANK_SELECTION')
                close(p['area'],count/(n*n),'ACTUAL_AREA')
                for outcome in ('full','isolated'):
                    e=r['clean_score']-cache[s][outcome+'_logit']
                    close(p['signed'][outcome],e,'FINITE_EFFECT');close(p['absolute'][outcome],abs(e),'ABSOLUTE')
                    close(p['candidate_best'][outcome],best[count][outcome],'CANDIDATE_BEST')
                    close(p['regret'][outcome],best[count][outcome]-abs(e),'REGRET')
                points+=1
            for outcome in ('full','isolated'):
                areas=[0]+[p['area'] for p in curve['points']]
                ys=[0]+[p['absolute'][outcome] for p in curve['points']]
                integral=sum((areas[j]-areas[j-1])*(ys[j]+ys[j-1])/2 for j in range(1,len(areas)))/areas[-1]
                close(curve['auc'][outcome],integral,'INDEPENDENT_AUC')
    return dict(task_id=tid,sha256=sha(path),curves=len(curve_index),points=points,
        candidates=candidate_count,seconds=r['seconds'],peak_gpu_bytes=r['peak_gpu_bytes'])

def tasks_audit():
    manifest=load(ROOT/'task_manifest.json');tasks=manifest['tasks']
    require(len(tasks)==2400 and digest(tasks)==manifest['digest'],'MANIFEST')
    ids=Counter((t['payload']['model_id'],t['payload']['block']) for t in tasks)
    require(ids==Counter({(m,b):400 for m in HEADS for b in BLOCKS}),'MODEL_BLOCK_COHORT')
    samples={}
    for m,b in ids:
        part=[t['payload'] for t in tasks if (t['payload']['model_id'],t['payload']['block'])==(m,b)]
        require({p['split_order'] for p in part}==set(range(400)),'ORDERS')
        samples[m,b]={p['sample_id']:(p['image_sha256'],p['class_id']) for p in part}
        require(len(samples[m,b])==400,'IMAGE_DUP')
    require(all(s==next(iter(samples.values())) for s in samples.values()),'SHARED_COHORT')
    queue=audit_coverage(ROOT/'queue.sqlite3',ROOT/'results')
    require(queue['status']=='PASS','QUEUE_AUDIT')
    with ProcessPoolExecutor(max_workers=8) as pool:
        records=[]
        for i,r in enumerate(pool.map(audit_task,tasks)):
            records.append(r)
            if i%200==0:print('audited',i+1,flush=True)
    require(load(ROOT/'metrics_manifest.json')['records']==[dict(task_id=r['task_id'],sha256=r['sha256']) for r in records],'METRICS_BINDING')
    report=dict(status='PASS',task_count=2400,records=records,
        curves=sum(r['curves'] for r in records),points=sum(r['points'] for r in records),
        candidates=sum(r['candidates'] for r in records),gpu_hours=sum(r['seconds'] for r in records)/3600)
    record(ROOT/'task_validation.json',report)

def inference_audit():
    from run_endpoint_m2_analysis import collect
    groups,heads,records=collect();summaries,pairs,tests=summarize(groups)
    for name,expected in [('summary.csv',summaries),('paired.csv',pairs)]:
        with (ROOT/name).open() as f:observed=list(csv.DictReader(f))
        require(len(observed)==len(expected),'TABLE_COVERAGE')
        for a,b in zip(observed,expected): require(a=={k:str(v) for k,v in b.items()},'TABLE_RECOMPUTE')
    published=load(ROOT/'auc_tests.json');require(published['test_count']==672 and published['tests']==tests,'TEST_RECOMPUTE')
    # Independent scipy sign p-values and monotone Holm correction.
    for t in tests:
        n=t['positive']+t['negative']
        p=1. if n==0 else float(binomtest(t['positive'],n,.5,alternative='greater').pvalue)
        close(t['p'],p,'SCIPY_SIGN')
        require(t['positive']+t['negative']+t['ties']==400,'SIGN_IMAGE_UNIT')
    prev=0
    for rank,t in enumerate(sorted(tests,key=lambda t:(t['p'],t['test_id']))):
        prev=max(prev,min(1.,(672-rank)*t['p']));close(prev,t['holm_p'],'INDEPENDENT_HOLM')
        require(t['reject']==(prev<.05),'REJECT')
    with (ROOT/'head_summary.csv').open() as f:rows=list(csv.DictReader(f))
    require(len(rows)==len(heads),'HEAD_TABLE_COUNT')
    for row,(key,by_id) in zip(rows,sorted(heads.items())):
        model,b,h,geometry,ref,outcome,metric,method=key;vals=[by_id[i] for i in sorted(by_id)]
        require((row['model'],int(row['block']),int(row['head']),row['geometry'],row['reference'],
            row['outcome'],row['metric'],row['method'])==key,'HEAD_TABLE_KEY')
        close(row['median'],np.median(vals),'HEAD_MEDIAN')
        require(json.loads(row['ci95'])==bootstrap_median(vals,'M2|head|'+'|'.join(map(str,key))),'HEAD_CI')
    record(ROOT/'inference_validation.json',dict(status='PASS',groups=len(groups),summaries=len(summaries),
        paired=len(pairs),head_summaries=len(rows),tests=len(tests),image_unit=400))

def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',action='store_true');p.add_argument('--inference',action='store_true');a=p.parse_args()
    for path,s in load(SPEC)['source_bindings'].items():require(sha(REPO/path)==s,'SOURCE:'+path)
    freeze=load(ROOT/'freeze.json')
    for path,s in freeze['sources'].items():require(sha(REPO/path)==s,'FROZEN:'+path)
    require(sha(ROOT/'task_manifest.json')==freeze['task_manifest_sha256'],'FROZEN_MANIFEST')
    startup=load(ROOT/'startup.json');require(startup['producer_commit']==startup['remote_commit'],'PUBLISHED')
    require(subprocess.run(['git','merge-base','--is-ancestor',startup['producer_commit'],'HEAD'],cwd=REPO).returncode==0,'ANCESTRY')
    if a.tasks:tasks_audit()
    if a.inference:inference_audit()
    require(not any(k in sys.modules for k in ('torch','timm','PIL')),'CPU_ONLY')
    print('M2_VALIDATION_PASS',flush=True)

if __name__=='__main__':main()
