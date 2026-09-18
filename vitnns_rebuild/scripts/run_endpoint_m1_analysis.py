#!/usr/bin/env python3
"""Recompute M1 per-image metrics and paired inference from bound component records."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from vitnns.endpoint_ep1_queue import digest
from vitnns.endpoint_m1_statistics import (evaluate,PREDICTORS,METRICS,bootstrap_median,sign_test,holm)
ROOT=REPO/'reports/nns_vit_endpoint_m1_r1_20260918'

def load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    with path.open('xb') as f:
        f.write(gzip.compress(data,mtime=0) if str(path).endswith('.gz') else data+b'\n')

def merged_task(task):
    tid=task['task_id']; p=task['payload']
    source=REPO/p['e4_path']
    assert sha(source)==p['e4_sha256']
    e4=load(source)['result']
    extra_path=ROOT/'results/attempts'/tid/'attempt-01.json'
    extra=load(extra_path); unsigned=dict(extra); expected=unsigned.pop('canonical_digest')
    assert digest(unsigned)==expected
    extra=extra['result']
    assert extra['model_id']==p['model_id'] and extra['image_sha256']==p['image_sha256']
    result={}
    for ref in ('normalized_zero','blur_rgb'):
        old=[r for r in e4['finite_utility'][ref]['components'] if r['operation']!='joint_routing_value']
        rows=old+[r for r in extra['rows'] if r['reference_id']==ref]
        assert len(rows)==(240 if p['model_id']=='vit_b16' else 168)
        assert len({r['component_id'] for r in rows})==len(rows)
        for r in rows:
            assert abs(r['effect']-(extra['clean_score']-r['replacement_score']))<=1e-4+1e-4*abs(extra['clean_score'])
            assert r['absolute_effect']==abs(r['effect'])
            assert all(np.isfinite(r[k]) for k in ('m','output_norm','gradient_norm','gradient_activation','reference_displacement_norm','p_ref','absolute_effect'))
        result[ref]=rows
    return result

def process_task(task):
    tid=task['task_id']; p=task['payload']; path=ROOT/'per_image'/(tid+'.json.gz')
    if path.exists():
        return tid,sha(path)
    rows=merged_task(task)
    result={ref:evaluate(part,tid,ref) for ref,part in rows.items()}
    write(path,dict(task_id=tid,model_id=p['model_id'],sample_id=p['sample_id'],split_order=p['split_order'],
                    class_id=p['class_id'],component_rows=rows,comparisons=result))
    return tid,sha(path)

def inference():
    groups={}
    paths=sorted((ROOT/'per_image').glob('*.json.gz'))
    assert len(paths)==800
    for path in paths:
        data=load(path); model=data['model_id']; order=data['split_order']
        for ref,rows in data['comparisons'].items():
            for row in rows:
                key=(model,ref,row['family'],row['stratum'])
                arr=groups.setdefault(key,np.full((400,len(PREDICTORS),len(METRICS)),np.nan))
                for pi,p in enumerate(PREDICTORS):
                    for mi,m in enumerate(METRICS):
                        value=row['metrics'][p][m]
                        arr[order,pi,mi]=np.nan if value is None else value
    # Models share images, so pooled observations are paired per image.
    for key in list(groups):
        if key[0]=='vit_b16':
            other=('deit_s16',)+key[1:]
            groups[('pooled_descriptive',)+key[1:]]=(groups[key]+groups[other])/2
    summaries=[]; paired=[]; absolute_tests=[]; paired_tests=[]
    for gi,(key,arr) in enumerate(sorted(groups.items())):
        gid='|'.join(key); identity=dict(model=key[0],reference=key[1],family=key[2],stratum=key[3])
        for pi,p in enumerate(PREDICTORS):
            for mi,m in enumerate(METRICS):
                a=arr[:,pi,mi]; finite=a[np.isfinite(a)]
                summaries.append(dict(**identity,predictor=p,metric=m,n_images=len(finite),expected_images=400,
                    median=None if not len(finite) else float(np.median(finite)),
                    ci95=bootstrap_median(finite,gid+'|'+p+'|'+m),
                    status='DESCRIPTIVE' if len(finite)==400 else 'NA_OR_INCOMPLETE',
                    na_reason=None if len(finite)==400 else 'UNDEFINED_COMPARISON_METRIC'))
        if key[0]=='pooled_descriptive':
            continue
        for pi,p in enumerate(PREDICTORS[1:],1):
            for mi,m in enumerate(METRICS):
                # All differences are oriented so positive favors Endpoint.
                d=(arr[:,0,mi]-arr[:,pi,mi])*(-1 if m=='regret_at_1' else 1)
                finite=d[np.isfinite(d)]; test_id=gid+'|'+p+'|'+m
                record=dict(**identity,baseline=p,metric=m,n_images=len(finite),
                    difference='baseline_minus_endpoint' if m=='regret_at_1' else 'endpoint_minus_baseline',
                    median=None if not len(finite) else float(np.median(finite)),ci95=bootstrap_median(finite,'paired|'+test_id))
                if m=='spearman':
                    test=sign_test(finite) if len(finite)==400 else dict(positive=0,negative=0,ties=len(finite),p=1.0)
                    paired_tests.append(dict(test_id=test_id,**identity,baseline=p,metric=m,valid=len(finite)==400,**test))
                paired.append(record)
        a=arr[:,0,0]; finite=a[np.isfinite(a)]
        for threshold in (0,.1):
            test=sign_test(finite-threshold) if len(finite)==400 else dict(positive=0,negative=0,ties=len(finite),p=1.)
            absolute_tests.append(dict(test_id=gid+'|threshold='+str(threshold),**identity,threshold=threshold,valid=len(finite)==400,**test))
        if gi%50==0:
            print('inference_groups',gi,flush=True)
    holm(paired_tests); holm(absolute_tests)
    for collection in (paired_tests,absolute_tests):
        for row in collection:
            row['decision']='INVALID_UNDEFINED_METRIC' if not row['valid'] else ('SUPPORTED' if row['reject'] else 'NOT_SUPPORTED')
    write(ROOT/'summary.json.gz',summaries)
    write(ROOT/'paired_comparisons.json.gz',paired)
    write(ROOT/'paired_spearman_tests.json',dict(test_count=len(paired_tests),tests=paired_tests))
    write(ROOT/'absolute_spearman_tests.json',dict(test_count=len(absolute_tests),tests=absolute_tests))
    print('INFERENCE_COMPLETE',len(summaries),len(paired),len(paired_tests),len(absolute_tests),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics',action='store_true');p.add_argument('--inference',action='store_true');args=p.parse_args()
    assert load(ROOT/'acquisition_receipt.json')['status']=='PASS'
    if args.metrics:
        if not (ROOT/'analysis_freeze.json').exists():
            write(ROOT/'analysis_freeze.json',dict(time_ns=time.time_ns(),cpu_only=True,CUDA_VISIBLE_DEVICES='',
                sources={s:sha(REPO/s) for s in ('src/vitnns/endpoint_m1_statistics.py','scripts/run_endpoint_m1_analysis.py')},
                bootstrap_algorithm='empirical_CDF_of_exact_joint_uniform_central_order_statistics',
                prior_outcomes='primary_E6_previously_known; supplementary_M1_metrics_not_yet_computed'))
        freeze=load(ROOT/'analysis_freeze.json')
        assert all(sha(REPO/s)==v for s,v in freeze['sources'].items())
        tasks=load(ROOT/'task_manifest.json')['tasks']
        started=time.monotonic(); records=[]
        with ProcessPoolExecutor(max_workers=8) as pool:
            for i,(tid,dig) in enumerate(pool.map(process_task,tasks)):
                records.append(dict(task_id=tid,path='per_image/'+tid+'.json.gz',sha256=dig))
                if i%50==0:
                    print('metrics',i+1,'/800',flush=True)
        write(ROOT/'metrics_manifest.json',dict(records=records,seconds=time.monotonic()-started,task_count=len(records)))
    if args.inference:
        inference()

if __name__=='__main__':
    main()
