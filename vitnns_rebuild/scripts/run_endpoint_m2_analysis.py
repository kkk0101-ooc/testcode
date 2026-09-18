#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from run_endpoint_m2 import ROOT,load,sha,write
from vitnns.endpoint_m2_regions import METHODS,BLOCKS
from vitnns.endpoint_m2_statistics import image_values,summarize
from vitnns.endpoint_m1_statistics import bootstrap_median

def csv_write(path,rows):
    with path.open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def collect():
    groups=defaultdict(dict);head_groups=defaultdict(dict);records=[]
    tasks=load(ROOT/'task_manifest.json')['tasks']
    for i,task in enumerate(tasks):
        p=ROOT/'results/attempts'/task['task_id']/'attempt-01.json';r=load(p)['result']
        heads,block=image_values(r);model=r['model_id'];b=r['block'];sample=r['sample_id']
        for (geometry,ref,method,outcome,metric),v in block.items():
            key=(model,b,geometry,ref,outcome,metric)
            groups[key].setdefault(sample,[None]*len(METHODS))[METHODS.index(method)]=v
        for (geometry,ref,h,method,outcome,metric),v in heads.items():
            head_groups[(model,b,h,geometry,ref,outcome,metric,method)][sample]=v
        records.append(dict(task_id=task['task_id'],sha256=sha(p)))
        if i%200==0:print('collected',i+1,flush=True)
    for key in list(groups):
        model,b,geometry,ref,outcome,metric=key
        if b!=2:continue
        parts=[groups[(model,bb,geometry,ref,outcome,metric)] for bb in BLOCKS]
        ids=set(parts[0])
        assert all(set(p)==ids for p in parts)
        groups[(model,'all_blocks_descriptive',geometry,ref,outcome,metric)]={
            i:np.mean([p[i] for p in parts],axis=0).tolist() for i in sorted(ids)}
    return groups,head_groups,records

def main():
    assert load(ROOT/'acquisition_receipt.json')['status']=='PASS'
    groups,heads,records=collect();summary,pairs,tests=summarize(groups)
    csv_write(ROOT/'summary.csv',summary);csv_write(ROOT/'paired.csv',pairs)
    write(ROOT/'auc_tests.json',dict(test_count=len(tests),holm_scope='all_672_bidirectional_AUC_tests',tests=tests))
    head_rows=[]
    for key,by_id in sorted(heads.items()):
        model,b,h,geometry,ref,outcome,metric,method=key
        assert len(by_id)==400
        vals=[by_id[i] for i in sorted(by_id)]
        head_rows.append(dict(model=model,block=b,head=h,geometry=geometry,reference=ref,
            outcome=outcome,metric=metric,method=method,n_images=400,median=float(np.median(vals)),
            ci95=bootstrap_median(vals,'M2|head|'+'|'.join(map(str,key)))))
    csv_write(ROOT/'head_summary.csv',head_rows)
    with gzip.open(ROOT/'image_values.json.gz','wt') as f:
        json.dump([dict(key=list(k),images=v) for k,v in sorted(groups.items(),key=lambda x:str(x[0]))],f,allow_nan=False)
    write(ROOT/'metrics_manifest.json',dict(records=records,task_count=len(records)))
    assert not any(k in sys.modules for k in ('torch','timm','PIL'))
    print(json.dumps(dict(status='ANALYSIS_COMPLETE',groups=len(groups),summary=len(summary),
        paired=len(pairs),head_summary=len(head_rows),tests=len(tests))),flush=True)

if __name__=='__main__':main()
