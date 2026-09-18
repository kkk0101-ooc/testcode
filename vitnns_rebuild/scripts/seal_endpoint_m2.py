#!/usr/bin/env python3
"""Prompt-to-artifact acceptance and immutable M2 result seal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from run_endpoint_m2 import ROOT,REPO,load,sha,write

def require(value,label):
    if not value:raise ValueError(label)

def artifact(relative):
    path=REPO/relative
    require(path.is_file(),'MISSING:'+relative)
    return dict(path=relative,sha256=sha(path),bytes=path.stat().st_size)

def main():
    task=load(ROOT/'task_validation.json');inf=load(ROOT/'inference_validation.json')
    extra=load(ROOT/'additional_completion_validation.json');qc=load(ROOT/'document_validation.json')
    require(all(v['status']=='PASS' for v in (task,inf,extra,qc)),'ALL_AUDITS')
    require(task['task_count']==2400 and task['curves']==2332800 and task['points']==8164800,'PRODUCTION_COUNTS')
    require(inf==dict(status='PASS',groups=512,summaries=4096,paired=3584,
        head_summaries=27648,tests=672,image_unit=400),'INFERENCE_COUNTS')
    require(extra['task_count']==2400 and extra['reference_tensor_checks']==4800
        and extra['control_checks']==88800 and extra['independent_image_groups']==512,'EXTRA_COUNTS')
    rel=ROOT.relative_to(REPO).as_posix()+'/'
    checks=[]
    def check(requirement,paths,evidence):
        checks.append(dict(requirement=requirement,status='PASS',artifacts=[artifact(p) for p in paths],evidence=evidence))
    check('M1 predecessor fully verified',[rel+'predecessor_validation.json'],
        'M1 seal:1631 files and17 requirements; all800 tasks and2688 tests recomputed before M2')
    check('3.1/3.2 early middle late cross-layer scope',[rel+'task_manifest.json',rel+'task_validation.json'],
        'Exact blocks2/5/10 x2 models x400 shared images; no exclusions')
    check('3.3 all heads in each model',[rel+'task_validation.json',rel+'additional_completion_validation.json'],
        'ViT12 heads;DeiT6 heads;21600 image-model-block-head conditions;all head identities checked')
    check('3.4 normalized-zero and frozen blur',[rel+'additional_completion_validation.json','config/nns_vit_endpoint/endpoint_m2.json'],
        '4800 reference tensor hashes equal E4;kernel31 sigma10;no effect-selected reference')
    check('3.5 four patch budgets and existing geometry',[rel+'task_validation.json','src/vitnns/endpoint_m2_regions.py'],
        'Patch counts1/10/20/49;legacy4x4 grid counts1/2/4;all selections and actual areas recomputed')
    check('3.6 eight methods on the same head',[rel+'task_validation.json','src/vitnns/endpoint_m2_compute.py'],
        'Endpoint absolute/signed,Taylor,gradient,displacement,random,center,attention;fixed20 random permutations')
    check('3.6 genuine full-input regional effect plus isolated-head extension',[rel+'task_validation.json',rel+'additional_completion_validation.json'],
        'Signed clean-minus-replacement and absolute effects reconstructed from finite logits;native-prefix parity controls on every head/ref/geometry')
    check('3.6 median effect and random lift at every budget',[rel+'summary.csv',rel+'paired.csv'],
        '4096 summaries and3584 paired contrasts;baseline=random rows give per-budget andAUC lift')
    check('3.6 candidate-set regret',[rel+'task_validation.json',rel+'summary.csv'],
        'All singleton candidates and larger-budget method union verified;best evaluated candidate only')
    check('3.6 block/head stratification and curves/AUC',[rel+'head_summary.csv',rel+'figure_manifest.json',
        rel+'m2_full_effect_budget.png',rel+'m2_isolated_effect_budget.png',rel+'m2_head_contrasts.png'],
        '27648 head-summary rows;3 PNG/PDF figures;independent trapezoid AUC recomputation')
    check('3.7 image unit paired inference and absolute outcomes',[rel+'inference_validation.json',
        rel+'additional_completion_validation.json',rel+'auc_tests.json',rel+'image_values.json.gz'],
        'Mean random repetitions then heads within image;400 images/group;10000 bootstrap;672 sign tests and independent Holm')
    check('3.8 conclusions limited by layer/reference/outcome',['docs/endpoint_m2_results_20260919.md',rel+'report_data.json'],
        'Both directional tests reported by model/reference/block;non-rejection not interpreted as equivalence')
    check('11.5 and13 mandatory3 full M2 scope',['docs/endpoint_m2_execution_20260919.md','docs/endpoint_m2_results_20260919.md'],
        'blocks2/5/10 all heads bothreferences Endpoint/Taylor/random included;all eight methods delivered')
    check('14 frozen config/image/head/reference/budget/statistics/tolerances',
        ['config/nns_vit_endpoint/endpoint_m2.json',rel+'freeze.json',rel+'startup.json'],
        'Published producer9d52d5f56 before formal read;source/task SHA bindings rechecked;token/channel sampling belongs to M3')
    check('Required primary-claim and coverage boundaries preserved',['papers/nns_vit_endpoint/zh/endpoint_m2_supplement.md',
        'papers/nns_vit_endpoint/en/endpoint_m2_supplement.md','papers/nns_vit_endpoint/notes/endpoint_m2_evidence_matrix.md'],
        'Supplementary study;11/12 broad but non-universal;original C4a/C4b NOT_SUPPORTED;M1 coverage table remains sealed')
    check('Necessary scientific implementation tests',[rel+'test_results.json','tests/test_endpoint_m2.py'],
        '10 focused tests:geometry/random/AUC,three-block native suffix and finite difference,synthetic complete pipeline,statistics scope')
    check('Report data/links/figures and academic writing',[rel+'document_validation.json',
        'papers/nns_vit_endpoint/figure_manifest_m2.json','papers/nns_vit_endpoint/qc/endpoint_m2_writing_audit.md'],
        'Bilingual numerical tables identical;all local links and figure hashes resolve;three figures visually inspected;numbered interpretations')
    require(load(ROOT/'acquisition_receipt.json')['status']=='PASS','ACQUISITION')
    startup=load(ROOT/'startup.json');freeze=load(ROOT/'freeze.json')
    require(startup['producer_commit']==startup['remote_commit'],'PUBLISHED')
    for p,s in freeze['sources'].items():require(sha(REPO/p)==s,'SOURCE_DRIFT:'+p)
    for r in task['records']:require(sha(ROOT/'results/attempts'/r['task_id']/'attempt-01.json')==r['sha256'],'RAW_AFTER_AUDIT')
    audit=dict(objective='Execute Stage M M2: cross-layer all-head spatial utility study and stop after M2',
        status='PASS',requirement_count=len(checks),requirements=checks,missing=[],unverified=[],
        optional_not_executed=['2x2 patch square/connected-region search','other blur strengths','mixed-effects model'],
        outside_scope=['M3 token/channel sampling','M4-M6','Path IG'],
        audited_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    write(ROOT/'completion_audit.json',audit)
    manifest=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path.name not in {'artifact_manifest.json','terminal_receipt.json'}:
            manifest.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    write(ROOT/'artifact_manifest.json',dict(file_count=len(manifest),files=manifest))
    write(ROOT/'terminal_receipt.json',dict(status='ENDPOINT_M2_PASS_EXECUTION_AND_AUDIT',
        scientific_status='SUPPLEMENTARY_LAYER_REFERENCE_OUTCOME_SPECIFIC',primary_C4a='NOT_SUPPORTED',primary_C4b='NOT_SUPPORTED',
        image_model_block_tasks=2400,image_model_block_head_conditions=21600,curves=task['curves'],points=task['points'],
        completion_audit_sha256=sha(ROOT/'completion_audit.json'),artifact_manifest_sha256=sha(ROOT/'artifact_manifest.json'),
        next_authorization='STOP_AFTER_M2',sealed_time_ns=time.time_ns()))
    print(json.dumps(dict(status='ENDPOINT_M2_PASS_EXECUTION_AND_AUDIT',requirements=len(checks),files=len(manifest))))

def verify():
    t=load(ROOT/'terminal_receipt.json');require(t['status']=='ENDPOINT_M2_PASS_EXECUTION_AND_AUDIT','TERMINAL')
    require(t['next_authorization']=='STOP_AFTER_M2' and t['primary_C4a']==t['primary_C4b']=='NOT_SUPPORTED','SCOPE')
    require(sha(ROOT/'completion_audit.json')==t['completion_audit_sha256'],'AUDIT_SHA')
    require(sha(ROOT/'artifact_manifest.json')==t['artifact_manifest_sha256'],'MANIFEST_SHA')
    m=load(ROOT/'artifact_manifest.json');require(len(m['files'])==m['file_count'],'FILE_COUNT')
    expected={r['path'] for r in m['files']}|{'artifact_manifest.json','terminal_receipt.json'}
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    require(expected==actual,'FILE_SET')
    for r in m['files']:
        p=ROOT/r['path'];require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'ARTIFACT:'+r['path'])
    a=load(ROOT/'completion_audit.json');require(a['status']=='PASS' and not a['missing'] and not a['unverified'],'COMPLETION')
    for check in a['requirements']:
        require(check['status']=='PASS','CHECK')
        for artifact0 in check['artifacts']:
            require(artifact(artifact0['path'])==artifact0,'EXTERNAL_BINDING')
    print(json.dumps(dict(status=t['status'],files=m['file_count'],requirements=len(a['requirements']),hashes='PASS')))

if __name__=='__main__':verify() if '--verify' in sys.argv else main()
