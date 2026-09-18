#!/usr/bin/env python3
"""M1 completion audit: source identity, exhaustive coverage and recomputation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
from scipy.stats import rankdata, binomtest
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from run_endpoint_m1_analysis import ROOT,load,sha,merged_task,write
from vitnns.endpoint_ep1_queue import digest
from vitnns.endpoint_m1_statistics import (PREDICTORS,METRICS,FAMILIES,evaluate,bootstrap_median,seed)

def require(condition,code):
    if not condition:
        raise ValueError(code)

def close(a,b,code):
    require((a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)),code)

def record_or_verify(path,value):
    if path.exists():
        require(load(path)==value,'EXISTING_AUDIT_DRIFT:'+path.name)
    else:
        write(path,value)

def independent_metrics(p,y):
    p=np.asarray(p);y=np.asarray(y);order=np.lexsort((np.arange(len(p)),-p));n=len(p)
    top=np.sort(y)[::-1]
    rho=None if n<4 or np.ptp(p)==0 or np.ptp(y)==0 else float(np.corrcoef(rankdata(p),rankdata(y))[0,1])
    pairs=[(i,j) for i in range(n) for j in range(i+1,n) if y[i]!=y[j]]
    accuracy=None if not pairs else sum(0.5 if p[i]==p[j] else float((p[i]>p[j])==(y[i]>y[j])) for i,j in pairs)/len(pairs)
    ideal=sum(v/math.log2(i+2) for i,v in enumerate(top))
    return dict(spearman=rho,capture_at_1=None if top[0]==0 else float(y[order[0]]/top[0]),
                capture_at_3=None if top[:3].sum()==0 else float(y[order[:3]].sum()/top[:3].sum()),
                regret_at_1=float(top[0]-y[order[0]]),
                ndcg=None if ideal==0 else sum(y[j]/math.log2(i+2) for i,j in enumerate(order))/ideal,
                pairwise_accuracy=accuracy,top1_accuracy=float(y[order[0]]==top[0]))

def audit_task(task):
    tid=task['task_id'];p=task['payload']; path=ROOT/'per_image'/(tid+'.json.gz')
    value=load(path); combined=merged_task(task)
    require(value['component_rows']==combined,'MERGED_SOURCE:'+tid)
    require(value['sample_id']==p['sample_id'] and value['class_id']==p['class_id'],'IMAGE_TARGET')
    result=load(ROOT/'results/attempts'/tid/'attempt-01.json')['result']
    identity=result['worker_identity']
    require(identity['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and identity['deterministic_algorithms'] and not identity['cuda_matmul_allow_tf32'] and not identity['cudnn_allow_tf32'] and not identity['autocast'] and identity['startup_matmul_backward']=='PASS','WORKER_NUMERIC_CONTRACT')
    require(identity['sys_executable_realpath']==str(Path(sys.executable).resolve()),'EXECUTABLE')
    require(result['new_condition_count']==(48 if p['split_order']<64 else 96),'ACQUISITION_COUNT')
    require(result['control_count']==10,'CONTROL_COUNT')
    e3path=REPO/p['e3_path'];require(sha(e3path)==p['e3_sha256'],'E3_HASH')
    envelope=load(e3path); unsigned=dict(envelope); expected=unsigned.pop('canonical_digest')
    require(digest(unsigned)==expected,'E3_CANONICAL')
    e3=envelope['result']
    for r in result['rows']:
        source=e3['blocks'][r['block']]['components'][r['e3_component_keys'][0]]
        require(r['m']==source['absolute_mass'] and r['a']==source['signed_sum'],'E3_ENDPOINT_MASS')
    comparisons=0;independent=0;small=0
    for ref,rows in combined.items():
        h=12 if p['model_id']=='vit_b16' else 6
        expected={f'b{b:02d}.head_{i:02d}' for b in range(12) for i in range(h)}
        expected|={f'b{b:02d}.{op}' for b in range(12) for op in ('routing','value','attention_branch','mlp_branch','query','key','identity1','identity2')}
        require({r['component_id'] for r in rows}==expected,'EXACT_COMPONENT_IDS')
        require(all(r['reference_id']==ref for r in rows),'REFERENCE_ID')
        # Exhaustively regenerate every set, all metrics and all random permutations.
        recomputed=evaluate(rows,tid,ref)
        require(recomputed==value['comparisons'][ref],'ALL_METRICS_RANDOM_RECOMPUTE:'+tid)
        by_id={r['component_id']:r for r in rows}
        for c in recomputed:
            comparisons+=1
            if c['stratum']=='within_block_mean':
                continue
            part=[by_id[i] for i in c['component_ids']]; y=[r['absolute_effect'] for r in part]
            # Independent reference implementation on all endpoint and Taylor sets.
            for predictor,pvalues in [('endpoint',[r['m'] for r in part]),('taylor',[abs(r['p_ref']) for r in part])]:
                metrics=independent_metrics(pvalues,y)
                for m in METRICS:
                    close(metrics[m],c['metrics'][predictor][m],'INDEPENDENT_'+m+':'+tid)
                independent+=1
            if len(part)<4:
                require(all(c['metrics'][pr]['spearman'] is None for pr in PREDICTORS),'SMALL_SET_SPEARMAN');small+=1
            require(c['random']['permutations']==1000,'RANDOM_COUNT')
            histogram=c['random']['distribution']['spearman_histogram_20bins']
            require(sum(histogram)==(1000 if c['random']['distribution']['spearman'] is not None else 0),'RANDOM_HISTOGRAM')
    return dict(task_id=tid,sha256=sha(path),comparisons=comparisons,independent_sets=independent,small_sets=small,
                new_conditions=result['new_condition_count'],gpu_seconds=result['seconds'],
                noop_error=result['noop_max_error'],control_error=result['control_max_error'])

def audit_inference():
    from collections import defaultdict
    groups=defaultdict(dict)
    for path in sorted((ROOT/'per_image').glob('*.json.gz')):
        d=load(path)
        for ref,rows in d['comparisons'].items():
            for r in rows:
                groups[(d['model_id'],ref,r['family'],r['stratum'])][d['sample_id']]=r['metrics']
    for key in list(groups):
        if key[0]=='vit_b16':
            a=groups[key];b=groups[('deit_s16',)+key[1:]]
            require(set(a)==set(b) and len(a)==400,'PAIRED_IMAGE_POOL')
            groups[('pooled_descriptive',)+key[1:]]={i:{p:{m:None if a[i][p][m] is None or b[i][p][m] is None else (a[i][p][m]+b[i][p][m])/2 for m in METRICS} for p in PREDICTORS} for i in a}
    summaries=load(ROOT/'summary.json.gz');paired=load(ROOT/'paired_comparisons.json.gz')
    require(len(groups)==504 and len(summaries)==24696 and len(paired)==14112,'SUMMARY_COVERAGE')
    seen=set()
    for r in summaries:
        key=(r['model'],r['reference'],r['family'],r['stratum']);pr=r['predictor'];m=r['metric']
        sid=(*key,pr,m);require(sid not in seen,'DUP_SUMMARY');seen.add(sid)
        values=[v[pr][m] for v in groups[key].values() if v[pr][m] is not None]
        require(r['n_images']==len(values) and r['expected_images']==400,'SUMMARY_IMAGE_UNIT')
        close(r['median'],None if not values else float(np.median(values)),'MEDIAN')
        require(r['ci95']==bootstrap_median(values,'|'.join(key)+'|'+pr+'|'+m),'BOOTSTRAP_RECOMPUTE')
    paired_by_id={}
    for r in paired:
        key=(r['model'],r['reference'],r['family'],r['stratum']);pr=r['baseline'];m=r['metric']
        sign=-1 if m=='regret_at_1' else 1
        values=[sign*(v['endpoint'][m]-v[pr][m]) for v in groups[key].values() if v['endpoint'][m] is not None and v[pr][m] is not None]
        tid='|'.join(key)+'|'+pr+'|'+m
        require(tid not in paired_by_id,'DUP_PAIRED');paired_by_id[tid]=values
        close(r['median'],None if not values else float(np.median(values)),'PAIRED_MEDIAN')
        require(r['ci95']==bootstrap_median(values,'paired|'+tid),'PAIRED_BOOTSTRAP')
    test_total=0
    for filename,count in [('paired_spearman_tests.json',2016),('absolute_spearman_tests.json',672)]:
        report=load(ROOT/filename);tests=report['tests']
        require(report['test_count']==len(tests)==count,'HOLM_SCOPE')
        require(len({r['test_id'] for r in tests})==count,'DUP_TEST')
        for r in tests:
            if filename.startswith('paired'):
                vals=paired_by_id[r['test_id']]
            else:
                key=(r['model'],r['reference'],r['family'],r['stratum'])
                vals=[v['endpoint']['spearman']-r['threshold'] for v in groups[key].values() if v['endpoint']['spearman'] is not None]
            valid=len(vals)==400;require(r['valid']==valid,'VALIDITY')
            if valid:
                pos=sum(v>1e-12 for v in vals);neg=sum(v< -1e-12 for v in vals)
                p=1. if pos+neg==0 else float(binomtest(pos,pos+neg,.5,alternative='greater').pvalue)
                close(r['p'],p,'INDEPENDENT_SIGN_P')
                require(r['positive']==pos and r['negative']==neg and r['ties']==400-pos-neg,'SIGN_COUNTS')
            else:
                require(r['p']==1 and r['decision']=='INVALID_UNDEFINED_METRIC','NA_NOT_PASS')
        prev=0.
        for rank,r in enumerate(sorted(tests,key=lambda r:(r['p'],r['test_id']))):
            prev=max(prev,min(1.,(count-rank)*r['p']))
            close(prev,r['holm_p'],'INDEPENDENT_HOLM')
            require(r['reject']==(prev<.05),'HOLM_REJECTION')
        test_total+=count
    return dict(groups=len(groups),summaries=len(summaries),paired=len(paired),tests=test_total)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tasks',action='store_true');parser.add_argument('--inference',action='store_true');args=parser.parse_args()
    spec=load(REPO/'config/nns_vit_endpoint/endpoint_m1.json')
    for p,s in spec['source_bindings'].items():require(sha(REPO/p)==s,'SOURCE_SHA:'+p)
    for name in ('freeze.json','analysis_freeze.json'):
        for p,s in load(ROOT/name)['sources'].items():require(sha(REPO/p)==s,'FROZEN_SOURCE_SHA:'+p)
    startup=load(ROOT/'startup.json');require(startup['producer_commit']==startup['remote_commit'],'PUBLISHED_PRODUCER')
    require(subprocess.run(['git','merge-base','--is-ancestor',startup['producer_commit'],'HEAD'],cwd=REPO).returncode==0,'PRODUCER_ANCESTRY')
    with sqlite3.connect(ROOT/'queue.sqlite3') as c:
        require(c.execute('SELECT COUNT(*) FROM tasks WHERE state="completed"').fetchone()[0]==800,'QUEUE_COMPLETE')
        require(c.execute('SELECT COUNT(*) FROM attempts WHERE status="COMPLETED"').fetchone()[0]==800,'QUEUE_ATTEMPTS')
    if args.tasks:
        tasks=load(ROOT/'task_manifest.json')['tasks'];require(len(tasks)==800,'TASKS')
        require(load(ROOT/'task_manifest.json')['digest']==digest(tasks),'TASK_DIGEST')
        with ProcessPoolExecutor(max_workers=8) as pool:
            records=[]
            for i,row in enumerate(pool.map(audit_task,tasks)):
                records.append(row)
                if i%50==0:print('audited_tasks',i+1,flush=True)
        manifest=load(ROOT/'metrics_manifest.json')
        require({r['task_id']:r['sha256'] for r in records}=={r['task_id']:r['sha256'] for r in manifest['records']},'METRICS_MANIFEST')
        record_or_verify(ROOT/'task_validation.json',dict(status='PASS',records=records,
            task_count=800,comparison_count=sum(r['comparisons'] for r in records),
            independent_metric_sets=sum(r['independent_sets'] for r in records),
            new_conditions=sum(r['new_conditions'] for r in records),
            attempt_gpu_hours=sum(r['gpu_seconds'] for r in records)/3600))
    if args.inference:
        report=audit_inference();record_or_verify(ROOT/'inference_validation.json',dict(status='PASS',**report))
        print(json.dumps(report),flush=True)
    require(not any(x in sys.modules for x in ('torch','timm','PIL')),'CPU_ONLY_IMPORTS')

if __name__=='__main__':
    main()
