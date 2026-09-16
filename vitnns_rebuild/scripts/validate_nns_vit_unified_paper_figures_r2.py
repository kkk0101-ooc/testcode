#!/usr/bin/env python3
"""Independent R2 hash, numeric-parity, coverage and visual-review checks."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'reports/nns_vit_unified_paper_figures_r2_20260916'
OLD=REPO/'reports/nns_vit_unified_paper_figures_r1_20260916'
EXCLUSIONS=['whole_root_inventory.json','validation_report.json','terminal_receipt.json']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def signed(value):
    return {**value,'canonical_digest':canonical(value)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(value):
    copy=dict(value); digest=copy.pop('canonical_digest',None)
    return digest==canonical(copy)


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+chr(10),encoding='utf-8')


def csv_rows(path):
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))


def quantile(values,p):
    x=sorted(values); i=(len(x)-1)*p; a=int(i); b=min(a+1,len(x)-1)
    return x[a]+(i-a)*(x[b]-x[a])


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12)


def validate():
    errors=[]; numeric_checks=[]
    def check(condition,message):
        if not condition: errors.append(message)
    manifest=read(ROOT/'render_manifest.json')
    check(verify(manifest),'manifest digest')
    figures=manifest['figures']; check([r['number'] for r in figures]==list(range(1,15)),'figure roster')
    check(manifest['figure_count']==14,'figure count')
    for fig in figures:
        check({r['format'] for r in fig['outputs']}=={'PNG','PDF','SVG'},'format roster')
        for r in fig['outputs']:
            p=REPO/r['path']
            check(p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink(),f'unsafe output {p}')
            if not p.is_file(): errors.append(f'missing output {p}'); continue
            check(sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],f'output hash {p}')
            if r['format']=='PNG':
                with Image.open(p) as im:
                    im.load(); check(im.width>=1200 and im.height>=700,f'PNG dimensions {p}')
            elif r['format']=='PDF': check(p.read_bytes().startswith(b'%PDF'),f'PDF signature {p}')
            else: check(b'<svg' in p.read_bytes()[:4096],f'SVG signature {p}')
    source_records=manifest['source_files']
    check(canonical(source_records)==manifest['source_files_digest'],'source digest')
    check(len({r['path'] for r in source_records})==len(source_records),'source roster duplicates')
    for r in source_records:
        p=next((p for p in [REPO/r['path'],REPO.parent/r['path']] if p.is_file()),None)
        check(p is not None and not p.is_symlink() and sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],'source drift '+r['path'])
    inputs=read(ROOT/'input_inventory.json'); check(verify(inputs) and inputs['source_files']==source_records,'input inventory parity')
    old=read(OLD/'render_manifest.json'); check(verify(old),'R1 manifest digest')
    for f in old['figures']:
        for r in f['outputs']: check(sha(REPO/r['path'])==r['sha256'],'R1 changed '+r['path'])
    for n in [5,8,9,10,12]:
        check([r['sha256'] for r in figures[n-1]['outputs']]==[r['sha256'] for r in old['figures'][n-1]['outputs']],f'unchanged figure {n} parity')
    display=read(ROOT/'numeric_display_data.json'); check(verify(display),'numeric display digest'); d=display['figures']
    d10=REPO/'evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis'
    values=[float(r['delta_fid']) for r in csv_rows(d10/'paper_data/per_sample_delta.csv') if r['status']=='VALID']; s=read(d10/'analysis.json')['summary']
    check(d['4']['values']==values,'Figure 4 full source values')
    check(d['4']['n']==len(values)==500 and d['4']['positive_count']==sum(x>0 for x in values)==500,'Figure 4 counts')
    check(close(d['4']['median'],statistics.median(values)) and d['4']['median']==s['median'],'Figure 4 median')
    check(d['4']['ci']==[s['bootstrap_ci_lower'],s['bootstrap_ci_upper']],'Figure 4 CI')
    numeric_checks.append('Figure 4: all 500 observations, median, positive count, official CI')
    f2=REPO/'evidence/stage_f/f2_r4_recovery_v2_20260907/formal/plot_data'
    panel=read(f2/'f2_b_plot_data.json')['plot_payload']['panels']
    for metric,rows in panel.items():
        check(len(rows)==len(d['6'][metric])==3,'Figure 6 roster')
        for source,shown in zip(rows,d['6'][metric]):
            vals=source['values']
            check(shown['method']==source['method'] and shown['values']==vals and shown['n']==len(vals),'Figure 6 series and counts')
            check(all(close(shown[k],quantile(vals,p)) for k,p in [('q1',.25),('median',.5),('q3',.75)]),'Figure 6 quartiles')
    numeric_checks.append('Figure 6: all six 500-image AUC series and independently computed quartiles')
    groups=read(f2/'f2_c_plot_data.json')['plot_payload']['panels']
    check([g['stratum'] for g in d['7']]==['overall','correct','misclassified'],'Figure 7 strata')
    for shown in d['7']:
        group=next(g for g in groups if g['stratum']==shown['stratum']); source=next(s for s in group['series'] if s['identity']['metric']=='delta_fid')
        check(all(source[k]==shown[k] for k in ['y','lower','upper','source_counts']),'Figure 7 matrix values/intervals/counts')
    check([g['source_counts'][-1]['valid'] for g in d['7']]==[498,417,81],'Figure 7 typed-failure coverage')
    numeric_checks.append('Figure 7: all 12 estimates, 12 intervals, coverage counts and typed failures')
    f3c=REPO/'evidence/stage_f/f3c4_analysis_r1_20260908/formal/paper_data'
    contrasts=csv_rows(f3c/'contrast_summary.csv'); scores=csv_rows(f3c/'per_image_method_scores.csv')
    methods=['conditional_all_blocks_fixed_clean_attention_class_logit_ig','vit_grad_cam_block11_norm1','full_model_class_logit_ig','attention_rollout_mean_head_residual_all_blocks','direct_raw_cls_attention_last_block_head_mean']
    by={(r['sample_id'],r['method_id']):float(r['image_mean_score']) for r in scores}; ids=sorted({r['sample_id'] for r in scores})
    check(len(d['11'])==len(contrasts)==4,'Figure 11 contrast roster')
    for source,shown,method in zip(contrasts,d['11'],methods[1:]):
        estimate=statistics.mean(by[(i,methods[0])]-by[(i,method)] for i in ids)
        check(shown['contrast_id']==source['contrast_id'] and close(shown['mean'],estimate) and shown['mean']==float(source['mean']),'Figure 11 paired means')
        check(shown['ci']==[float(source['bootstrap_ci_95_lower']),float(source['bootstrap_ci_95_upper'])] and shown['n']==len(ids)==50,'Figure 11 CI and n')
    numeric_checks.append('Figure 11: four means recomputed by sample_id, original intervals and n=50')
    tests=read(REPO/'evidence/stage_y/y4_r5_mechanism_analysis_20260914/formal/hypothesis_test_table.json'); p=tests.get('payload',tests); rows=p.get('rows',p.get('records',[]))
    check([r['hypothesis_id'] for r in d['13']]==['B4','B5'],'Figure 13 roster')
    for shown in d['13']:
        source=next(r for r in rows if r['hypothesis_id']==shown['hypothesis_id'])
        check(shown['estimate']==float(source['estimate']) and shown['ci']==[float(source['ci_lower']),float(source['ci_upper'])],'Figure 13 B4/B5')
    numeric_checks.append('Figure 13: exact B4/B5 estimates and interval endpoints')
    endpoints=read(REPO/'evidence/nns_vit_patch_depth/s6_r1_posthoc_exploratory_reporting_20260916/figure_data.json')['payload']['endpoint_rows']
    check([r['endpoint'] for r in d['14']]==['P1','P2','D1','D2'],'Figure 14 roster')
    for source,shown in zip(endpoints,d['14']):
        check(all(source[k]==shown[k] for k in ['endpoint','estimate','n_valid','n_total','n_NA']),'Figure 14 endpoint/counts')
        check(shown['ci']==[source['ordinary_BCa_95']['lower'],source['ordinary_BCa_95']['upper']],'Figure 14 ordinary BCa interval')
    numeric_checks.append('Figure 14: endpoint estimates, ordinary BCa intervals and valid/total/NA')
    for lang in ['zh','en']:
        captions=(ROOT/f'captions_{lang}.md').read_text(); check(all(f'## Figure {n}'+chr(10) in captions for n in range(1,15)),f'{lang} caption coverage')
        check('post-hoc exploratory' in captions.lower(),f'{lang} exploratory caption boundary')
    analysis=(ROOT/'figure_analysis.md').read_text(); check(all(f'## Image #{n}:' in analysis for n in range(1,15)),'analysis coverage')
    qc_path=ROOT/'manual_visual_qc.json'
    if not qc_path.is_file(): errors.append('actual visual review is pending')
    else:
        qc=read(qc_path); check(verify(qc) and qc.get('status')=='PASS','visual review status/digest')
        check([r['number'] for r in qc['records']]==list(range(1,15)) and all(r['status']=='PASS' for r in qc['records']),'visual review coverage')
        for r,fig in zip(qc['records'],figures):
            check(r['png_sha256']==next(o['sha256'] for o in fig['outputs'] if o['format']=='PNG'),'visual review binding')
    audit=read(ROOT/'cpu_access_audit.json'); check(verify(audit) and audit['CUDA_VISIBLE_DEVICES']=='' and not audit['torch_imported'] and not audit['timm_imported'],'CPU-only audit')
    inv_path=ROOT/'whole_root_inventory.json'
    if inv_path.is_file():
        inv=read(inv_path); check(verify(inv) and inv['files']==inventory() and canonical(inv['files'])==inv['files_digest'],'sealed root inventory')
    terminal_path=ROOT/'terminal_receipt.json'
    if terminal_path.is_file():
        term=read(terminal_path); check(verify(term) and term['status']=='PASS_PRESENTATION_RENDER','terminal receipt')
        for key,name in [('render_manifest_sha256','render_manifest.json'),('validation_report_sha256','validation_report.json'),('whole_root_inventory_sha256','whole_root_inventory.json'),('manual_visual_qc_sha256','manual_visual_qc.json')]:
            check(term[key]==sha(ROOT/name),'terminal binding '+key)
        check(term['validator_sha256']==sha(Path(__file__)),'terminal validator binding')
    return errors,numeric_checks,manifest


def inventory():
    return [{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(ROOT.rglob('*')) if p.is_file() and p.relative_to(ROOT).as_posix() not in EXCLUSIONS]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seal',action='store_true'); args=parser.parse_args()
    if args.seal and (ROOT/'terminal_receipt.json').exists(): raise RuntimeError('R2 already sealed; validation only is permitted')
    errors,checks,manifest=validate()
    report=signed({'schema_version':1,'document_type':'nns_vit_unified_paper_r2_validation','status':'FAIL' if errors else 'PASS','errors':errors,'numeric_checks':checks,'figure_count':14,'artifact_count':42,'source_file_count':len(manifest['source_files']),'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'torch_imported':'torch' in sys.modules,'timm_imported':'timm' in sys.modules})
    if not (ROOT/'terminal_receipt.json').exists(): write(ROOT/'validation_report.json',report)
    if args.seal and not errors:
        files=inventory(); write(ROOT/'whole_root_inventory.json',signed({'schema_version':1,'files':files,'files_digest':canonical(files),'exclusions':EXCLUSIONS}))
        write(ROOT/'terminal_receipt.json',signed({'schema_version':1,'status':'PASS_PRESENTATION_RENDER','figure_count':14,'artifact_count':42,'render_manifest_sha256':sha(ROOT/'render_manifest.json'),'validation_report_sha256':sha(ROOT/'validation_report.json'),'whole_root_inventory_sha256':sha(ROOT/'whole_root_inventory.json'),'manual_visual_qc_sha256':sha(ROOT/'manual_visual_qc.json'),'validator_sha256':sha(Path(__file__)),'cpu_only':True,'new_scientific_inference':False,'next_action':'USER_FIGURE_REVIEW'}))
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
    return bool(errors)


if __name__=='__main__':
    raise SystemExit(main())
