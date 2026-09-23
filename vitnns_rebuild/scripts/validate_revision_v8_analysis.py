#!/usr/bin/env python3
"""Independent source-recomputing validator for revision-v8 analyses."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'reports/nns_vit_endpoint_revision_v8_r1_20260923'
REG=REPO/'manifests/nns_vit_endpoint/revision_v8_20260923'
M3=REPO/'reports/nns_vit_endpoint_m3_r1_20260919'
E5=REPO/'reports/nns_vit_revision_v5_r1_20260921/e5'
V7=REPO/'reports/nns_vit_endpoint_revision_v7_r4_20260923'
SRC=REPO/'manifests/nns_vit_endpoint/ep0_r1_20260918'
BLOCKS=(0,3,7,11)


def req(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def ranks(values):
    array=np.asarray(values,float);order=np.argsort(array,kind='mergesort');result=np.empty(len(array));start=0
    while start<len(array):
        stop=start+1
        while stop<len(array) and array[order[stop]]==array[order[start]]:stop+=1
        result[order[start:stop]]=(start+1+stop)/2;start=stop
    return result


def rho(left,right):
    x=ranks(left);y=ranks(right);x-=x.mean();y-=y.mean();denominator=np.linalg.norm(x)*np.linalg.norm(y);req(denominator>0,'CONSTANT_RANK');return float(x@y/denominator)


def target_class(row):
    kind=row['kind'];module=row['module']
    if kind in {'ground_truth_logit','frozen_clean_top1_logit'}:return 'class_logit'
    if kind=='head_channel':return 'attention_head_coordinate'
    if kind=='token_channel' and module=='norm':return 'final_cls_coordinate'
    if kind=='token_channel':return 'post_block_token_coordinate'
    if module=='patch_embed':return 'patch_embedding_aggregate'
    if kind=='patch_channel_energy':return 'post_block_patch_energy'
    if kind=='patch_channel_sum' and module.count('.')==1:return 'post_block_patch_sum'
    if kind=='patch_channel_sum':return 'module_patch_aggregate'
    raise ValueError('TARGET_CLASS')


def main():
    req(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_ONLY')
    protocol=json.loads((ROOT/'protocol.json').read_text());req(protocol['status']=='FROZEN_BEFORE_V8_RESULTS','PROTOCOL')
    for item in json.loads((ROOT/'source_bindings.json').read_text())['files']:
        path=REPO/item['path'];req(path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'SOURCE:'+item['path'])
    target=json.loads((SRC/'target_registry.json').read_text());scalar=json.loads((REG/'scalar_target_registry.json').read_text());decomp=json.loads((REG/'decomposition_registry.json').read_text());inter=json.loads((REG/'intervention_registry.json').read_text())
    req(set(scalar)=={'schema_version','registry_type','status','source','meaning','target_count_by_model','target_class_count_by_model','targets_by_model'},'SCALAR_SCHEMA')
    req(set(decomp)=={'schema_version','registry_type','status','source','object_classes','cuts','cut_blocks','cut_scopes','hook_edges','terminology'},'DECOMP_SCHEMA')
    req(set(inter)=={'schema_version','registry_type','status','source','target','references','operations','component_experiment','regional_experiment','boundary'},'INTERVENTION_SCHEMA')
    for model,data in scalar['targets_by_model'].items():
        req(data==target['targets_by_model'][model] and len(data)==target['target_count_by_model'][model],'TARGET_ROWS')
        req(dict(sorted(Counter(target_class(row) for row in data).items()))==scalar['target_class_count_by_model'][model],'TARGET_CLASS_COUNTS')
    matrix=rows(ROOT/'target_cut_applicability.csv');req(len(matrix)==sum(target['target_count_by_model'].values())*12*8,'PAIR_COVERAGE')
    req(len({(row['model'],row['target_id'],row['cut_block'],row['cut_id']) for row in matrix})==len(matrix),'PAIR_UNIQUE')
    required_status={'VALID_FULL_TARGET_CUT_WITHIN_ATTENTION_BRANCH','VALID_ATTENTION_BRANCH_PARTITION_ONLY','VALID_FULL_TARGET_CUT','NOT_APPLICABLE_TARGET_UPSTREAM_OR_OUTSIDE_BRANCH','NOT_APPLICABLE_DECLARED_RECONSTRUCTION'}
    req(set(row['status'] for row in matrix)==required_status,'PAIR_STATUS')
    entropy=rows(ROOT/'entropy_mass_image_values.csv');summary=rows(ROOT/'entropy_mass_sensitivity.csv');req(len(entropy)==200 and len(summary)==20,'ENTROPY_COVERAGE')
    old={(row['model_id'],row['sample_id']):float(row['late_minus_early_entropy']) for row in rows(M3/'image_values.csv')};max_entropy=0.;computed={}
    for task in json.loads((M3/'task_manifest.json').read_text())['tasks']:
        result=json.loads((M3/'results/attempts'/task['task_id']/'attempt-01.json').read_text())['result']
        with np.load(REPO/result['metrics_path'],allow_pickle=False) as arrays:
            block=arrays['block'];valid=~arrays['zero_gradient']&np.isfinite(arrays['normalized_spatial_entropy']);values=arrays['normalized_spatial_entropy'].astype(float)
            value=float(np.median(values[valid&np.isin(block,range(8,12))])-np.median(values[valid&np.isin(block,range(4))]))
        key=(result['model_id'],result['sample_id']);computed[key]=value;max_entropy=max(max_entropy,abs(value-old[key]))
    observed={(row['model'],row['image']):float(row['trim_p00_late_minus_early']) for row in entropy};req(max(abs(computed[key]-observed[key]) for key in computed)<=1e-12,'ENTROPY_OUTPUT');req(max_entropy<=5e-7,'ENTROPY_ORIGINAL')
    h4=rows(ROOT/'h4_matched_image_values.csv');req(len(h4)==1600,'H4_COVERAGE');old_h4={(row['model'],row['image']):float(row['h4_image_mean']) for row in rows(V7/'h4_recomputed_image_values.csv')};acc=defaultdict(lambda:[[],[]])
    for metadata in sorted(E5.glob('*.json')):
        item=json.loads(metadata.read_text())
        with np.load(REPO/item['output'],allow_pickle=False) as arrays:
            for block in BLOCKS:
                effect=np.abs(arrays[f'b{block:02d}_normalized_zero_head_effect'].astype(float));ag=np.abs(arrays[f'b{block:02d}_head_activation_gradient'].astype(float));l1=arrays[f'b{block:02d}_heads_absolute_mass'].astype(float);signed=np.abs(arrays[f'b{block:02d}_heads_signed_sum'].astype(float))
                acc[item['model_id'],item['image']][0].append(rho(l1,effect)-rho(ag,effect));acc[item['model_id'],item['image']][1].append(rho(signed,effect)-rho(ag,effect))
    index={(row['model'],row['image']):row for row in h4};max_h4=0.
    for key,(original,matched) in acc.items():
        original_value=float(np.mean(original));matched_value=float(np.mean(matched));max_h4=max(max_h4,abs(original_value-old_h4[key]));req(abs(original_value-float(index[key]['original_h4']))<=1e-12 and abs(matched_value-float(index[key]['matched_aggregation_h4']))<=1e-12,'H4_OUTPUT')
    req(max_h4<=1e-12,'H4_ORIGINAL')
    baseline=json.loads((ROOT/'random_region_precision_audit.json').read_text());req(baseline['status']=='ALREADY_COMPLETE_AND_SOURCE_VALIDATED' and baseline['repeat_rows']==32000,'RANDOM_PRECISION')
    receipt=json.loads((ROOT/'terminal_receipt.json').read_text());req(receipt['status']=='REVISION_V8_ANALYSIS_PASS' and receipt['gpu_inference_runs']==0,'TERMINAL')
    report={'schema_version':1,'status':'PASS','checks':['source hashes','three registry schemas','457/385 target rows and class counts','all target-cut pairs','200 entropy image values','1600 matched-H4 image values','random baseline precision evidence'],'entropy_original_max_error':max_entropy,'h4_original_max_error':max_h4,'gpu_inference_runs':0}
    (ROOT/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0


if __name__=='__main__':raise SystemExit(main())
