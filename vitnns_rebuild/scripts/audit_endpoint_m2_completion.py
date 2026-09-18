#!/usr/bin/env python3
"""Independent M2 acceptance checks beyond the frozen numeric verifier."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re
import sys
import numpy as np
from run_endpoint_m2 import ROOT,REPO,SPEC,load,sha,write
sys.path.insert(0,str(REPO/'src'))
from vitnns.endpoint_m2_regions import HEADS,BLOCKS,GRIDS,REFERENCES,METHODS,COUNTS
from vitnns.endpoint_ep1_queue import digest

def require(condition,code):
    if not condition:raise ValueError(code)
def close(a,b,code):require(np.allclose(a,b,atol=1e-10,rtol=1e-10),code)

def main():
    spec=load(SPEC)
    require(spec['blocks']==list(BLOCKS) and spec['models']==list(HEADS),'CONFIG_SCOPE')
    require(spec['methods']==list(METHODS) and spec['references']==list(REFERENCES),'CONFIG_METHODS')
    require(spec['statistics']['holm_test_count']==672 and spec['statistics']['bootstrap']==10000,'CONFIG_STATS')
    tasks=load(ROOT/'task_manifest.json')['tasks']
    source=load(REPO/'reports/nns_vit_endpoint_m1_r1_20260918/task_manifest.json')['tasks']
    original={r['task_id']:r['payload'] for r in source}
    groups=defaultdict(dict);records=[];control_checks=0;reference_checks=0
    for i,t in enumerate(tasks):
        p=t['payload'];tid=t['task_id'];old=original[tid.rsplit('.b',1)[0]]
        require(p==dict(old,block=p['block'],stage='ENDPOINT-M2'),'EXACT_INHERITED_TASK')
        r=load(ROOT/'results/attempts'/tid/'attempt-01.json')['result']
        e4=load(REPO/p['e4_path']);signed=dict(e4);canonical=signed.pop('canonical_digest')
        require(digest(signed)==canonical,'E4_CANONICAL')
        refs=e4['result']['reference_identity']
        for ref in REFERENCES:
            require(r['reference_identity'][ref]==refs[ref]['tensor_sha256'],'REFERENCE_TENSOR_PARITY')
            reference_checks+=1
        e3=load(REPO/p['e3_path'])['result']['blocks'][p['block']]['components']
        for entry in r['e3_parity']:
            require(entry['expected']==e3['head_'+str(entry['head'])]['absolute_mass'],'E3_PARITY_BINDING')
        score_index={(s['geometry'],s['reference'],s['head']):s['scores'] for s in r['scores']}
        for geometry in GRIDS:
            for h in range(HEADS[p['model_id']]):
                s=score_index[(geometry,'normalized_zero',h)]
                blur=score_index[(geometry,'blur_rgb',h)]
                for method in ('endpoint_absolute','endpoint_signed','signed_raw','gradient_only','attention'):
                    require(s[method]==blur[method],'REFERENCE_FREE_SELECTOR')
                summary=e3['head_'+str(h)]
                for key,values in [('absolute_mass',s['endpoint_absolute']),('signed_sum',s['signed_raw'])]:
                    expected=summary[key];observed=sum(values)
                    require(abs(expected-observed)<=1e-4+1e-4*max(abs(expected),abs(observed)),'E3_REGION_SUMS')
                require(sum(s['attention'])<=1.00001,'ATTENTION_MASS')
        expected={(g,ref,h) for g in GRIDS for ref in REFERENCES for h in range(HEADS[p['model_id']])}
        observed={(c['geometry'],c['reference'],c['head']) for c in r['checks'] if c['kind']=='suffix_native'}
        require(expected==observed,'CONTROL_EXACT_IDENTITIES')
        control_checks+=len(r['checks'])
        # Independently assemble per-head method means, then mean heads per image.
        samples=defaultdict(list)
        for curve in r['curves']:
            prefix=(curve['geometry'],curve['reference'],curve['head'],curve['method'])
            for outcome in ('full','isolated'):
                samples[(*prefix,outcome,'auc')].append(curve['auc'][outcome])
                for point in curve['points']:
                    for metric,field in [('effect','absolute'),('regret','regret')]:
                        samples[(*prefix,outcome,metric+'_'+str(point['count']))].append(point[field][outcome])
        by_method=defaultdict(dict)
        for (g,ref,h,method,outcome,metric),values in samples.items():
            require(len(values)==(20 if method=='random' else 1),'REPEAT_UNIT')
            by_method[(g,ref,outcome,metric,method)][h]=sum(values)/len(values)
        for (g,ref,outcome,metric,method),values in by_method.items():
            require(set(values)==set(range(HEADS[p['model_id']])),'HEAD_UNIT')
            key=(p['model_id'],p['block'],g,ref,outcome,metric)
            groups[key].setdefault(p['sample_id'],[None]*8)[METHODS.index(method)]=sum(values.values())/len(values)
        records.append(dict(task_id=tid,reference_parity=True,independent_aggregation=True))
        if i%200==0:print('completion_extra',i+1,flush=True)
    for key in list(groups):
        if key[1]!=2:continue
        parts=[groups[(key[0],b,*key[2:])] for b in BLOCKS]
        groups[(key[0],'all_blocks_descriptive',*key[2:])]={i:[sum(p[i][j] for p in parts)/3 for j in range(8)] for i in parts[0]}
    with gzip.open(ROOT/'image_values.json.gz','rt') as f:published=json.load(f)
    require(len(published)==len(groups),'IMAGE_VALUE_GROUPS')
    for row in published:
        expected=groups[tuple(row['key'])]
        require(set(row['images'])==set(expected) and len(expected)==400,'IMAGE_VALUE_IDS')
        for i,values in expected.items():close(row['images'][i],values,'INDEPENDENT_IMAGE_VALUES')
    for filename,is_paired in [('summary.csv',False),('paired.csv',True)]:
        with (ROOT/filename).open() as f:rows=list(csv.DictReader(f))
        for row in rows:
            b=row['block'] if row['block']=='all_blocks_descriptive' else int(row['block'])
            key=(row['model'],b,row['geometry'],row['reference'],row['outcome'],row['metric'])
            vals=np.array(list(groups[key].values()))
            if is_paired:
                observed=vals[:,0]-vals[:,METHODS.index(row['baseline'])]
                if row['metric'].startswith('regret'):observed=-observed
            else:observed=vals[:,METHODS.index(row['method'])]
            close(float(row['median']),float(np.median(observed)),'INDEPENDENT_TABLE_MEDIAN')
    evidence=dict(status='PASS',task_count=len(tasks),reference_tensor_checks=reference_checks,
        control_checks=control_checks,independent_image_groups=len(groups),images_per_group=400,
        frozen_production_and_statistics_unchanged=True)
    write(ROOT/'additional_completion_validation.json',evidence)
    print(json.dumps(evidence),flush=True)
    require(not any(x in sys.modules for x in ('torch','timm','PIL')),'CPU_ONLY')

if __name__=='__main__':main()
