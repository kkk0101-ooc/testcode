#!/usr/bin/env python3
"""Revision-v8 r2 recovery: type empty mass-quartile strata as retained NA."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve();REPO=HERE.parents[1]
SPEC=importlib.util.spec_from_file_location('revision_v8_r1',REPO/'scripts/run_revision_v8_analysis.py')
base=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(base)
R1=REPO/'reports/nns_vit_endpoint_revision_v8_r1_20260923'
base.ROOT=REPO/'reports/nns_vit_endpoint_revision_v8_r2_20260923'
base.REGISTRY_ROOT=REPO/'manifests/nns_vit_endpoint/revision_v8_r2_20260923'
base.SCRIPT=HERE
base.VALIDATOR=REPO/'scripts/validate_revision_v8_r2_analysis.py'
base.TEST_FILE=REPO/'tests/test_revision_v8_r2.py'
original_sources=base.source_files


def source_files():
    return original_sources()+[R1/name for name in ('protocol.json','stage0_receipt.json','blocked_receipt.json','architecture_graph_audit.json','target_cut_applicability.csv','target_cut_summary.csv')]


def entropy_analysis():
    records=base.m3_records();thresholds={};threshold_rows=[]
    for model in sorted({row['model'] for row in records}):
        for block in base.BLOCKS:
            data=np.concatenate([row['mass'][(row['block']==block)&~row['zero']&np.isfinite(row['mass'])] for row in records if row['model']==model])
            quantiles=np.quantile(data,np.asarray(base.MASS_QUANTILES)/100.0,method='linear')
            thresholds[model,block]={q:float(value) for q,value in zip(base.MASS_QUANTILES,quantiles)}
            threshold_rows.append({'model':model,'block':block,'n_units':len(data),**{f'p{q:02d}':thresholds[model,block][q] for q in base.MASS_QUANTILES}})
    old={(row['model_id'],row['sample_id']):float(row['late_minus_early_entropy']) for row in csv.DictReader((base.M3_ROOT/'image_values.csv').open())}
    image_rows=[];max_error=0.0
    for row in records:
        valid=~row['zero']&np.isfinite(row['mass'])&np.isfinite(row['entropy']);values={};counts={};reasons={}
        for trim in base.TRIMS:
            mask=valid.copy()
            if trim:
                for block in base.BLOCKS:mask[(row['block']==block)&(row['mass']<=thresholds[row['model'],block][trim])]=False
            early=row['entropy'][mask&np.isin(row['block'],range(4))];late=row['entropy'][mask&np.isin(row['block'],range(8,12))]
            base.require(len(early)>0 and len(late)>0,'ENTROPY_TRIM_EMPTY')
            values[f'trim_p{trim:02d}_late_minus_early']=float(np.median(late)-np.median(early));counts[f'trim_p{trim:02d}_retained_units']=int(mask.sum())
        for quartile,(low,high) in enumerate(((None,25),(25,50),(50,75),(75,None)),1):
            mask=valid.copy()
            for block in base.BLOCKS:
                block_mask=row['block']==block
                if low is not None:mask[block_mask&(row['mass']<=thresholds[row['model'],block][low])]=False
                if high is not None:mask[block_mask&(row['mass']>thresholds[row['model'],block][high])]=False
            early=row['entropy'][mask&np.isin(row['block'],range(4))];late=row['entropy'][mask&np.isin(row['block'],range(8,12))]
            metric=f'q{quartile}_late_minus_early';counts[f'q{quartile}_units']=int(mask.sum())
            counts[f'q{quartile}_early_units']=len(early);counts[f'q{quartile}_late_units']=len(late)
            if len(early)==0 or len(late)==0:
                values[metric]='';reasons[f'q{quartile}_na_reason']='NA_EMPTY_EARLY_OR_LATE_MASS_QUARTILE'
            else:
                values[metric]=float(np.median(late)-np.median(early));reasons[f'q{quartile}_na_reason']=''
        correlations=[]
        for block in base.BLOCKS:
            mask=valid&(row['block']==block);value,reason=base.spearman(np.log1p(row['mass'][mask]),row['entropy'][mask]);base.require(value is not None,'ENTROPY_RHO_'+reason);correlations.append(value)
        values['mean_within_block_mass_entropy_spearman']=float(np.mean(correlations))
        values['late_minus_early_log1p_mass']=float(np.median(np.log1p(row['mass'][valid&np.isin(row['block'],range(8,12))]))-np.median(np.log1p(row['mass'][valid&np.isin(row['block'],range(4))])))
        max_error=max(max_error,abs(values['trim_p00_late_minus_early']-old[row['model'],row['image']]))
        image_rows.append({'model':row['model'],'image':row['image'],**values,**counts,**reasons})
    metrics=[f'trim_p{x:02d}_late_minus_early' for x in base.TRIMS]+[f'q{x}_late_minus_early' for x in range(1,5)]+['mean_within_block_mass_entropy_spearman','late_minus_early_log1p_mass'];summary=[]
    for model in sorted({row['model'] for row in image_rows}):
        for metric in metrics:
            total=[row for row in image_rows if row['model']==model];data=[float(row[metric]) for row in total if row[metric]!=''];median,low,high=base.bootstrap_median(data,f'entropy-r2|{model}|{metric}')
            summary.append({'model':model,'metric':metric,'n_total':len(total),'n_valid':len(data),'n_na':len(total)-len(data),'median':median,'ci95_low':low,'ci95_high':high,**base.sign_counts(data),'analysis_identity':'post_hoc_mass_sensitivity'})
    return threshold_rows,image_rows,summary,max_error


def freeze():
    base.source_files=source_files;base.freeze()
    protocol=json.loads((base.ROOT/'protocol.json').read_text());protocol['namespace']='NNS_VIT_REVISION_V8_R2_20260923';protocol['amendment']={'predecessor_status':'BLOCKED_ENTROPY_QUARTILE_EMPTY','predecessor_receipt':base.record(R1/'blocked_receipt.json'),'change':'typed NA for an empty early or late image-quartile cell; thresholds, samples, estimands, and all other code paths unchanged'};base.write_json(base.ROOT/'protocol.json',protocol)
    receipt=json.loads((base.ROOT/'stage0_receipt.json').read_text());receipt['status']='REVISION_V8_R2_STAGE0_PASS_AND_ANALYSIS_FROZEN';receipt['protocol']=base.record(base.ROOT/'protocol.json');base.write_json(base.ROOT/'stage0_receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'source_files':len(json.loads((base.ROOT/'source_bindings.json').read_text())['files'])}))


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--freeze',action='store_true');group.add_argument('--analyze',action='store_true');args=parser.parse_args();base.source_files=source_files;base.entropy_analysis=entropy_analysis;freeze() if args.freeze else base.analyze();return 0


if __name__=='__main__':raise SystemExit(main())
