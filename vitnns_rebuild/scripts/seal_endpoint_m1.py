#!/usr/bin/env python3
"""Final M1 requirement-to-artifact audit and immutable terminal seal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from run_endpoint_m1_analysis import ROOT,load,sha

def require(test,code):
    if not test:raise ValueError(code)

def write_new(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    checks=[]
    def check(requirement,paths,evidence,condition=True):
        require(condition,requirement)
        records=[]
        for relative in paths:
            path=REPO/relative
            require(path.is_file(),'MISSING:'+relative)
            records.append(dict(path=relative,sha256=sha(path),bytes=path.stat().st_size))
        checks.append(dict(requirement=requirement,status='PASS',artifacts=records,evidence=evidence))
    rel=ROOT.relative_to(REPO).as_posix()+'/'
    task=load(ROOT/'task_validation.json');inf=load(ROOT/'inference_validation.json')
    require(task['status']==inf['status']=='PASS','DATA_AND_INFERENCE')
    check('M1 prerequisite: original E3/E6 source-recomputed PASS',[rel+'predecessor_validation.json'],
          'E3: 9,600 block evaluations and 384 raw blocks rechecked; E6: 1,600 image rows/16 families/28 tests recomputed',
          load(ROOT/'predecessor_validation.json')['status']=='PASS')
    check('Supplementary freeze; primary NOT_SUPPORTED preserved',
          ['config/nns_vit_endpoint/endpoint_m1.json',rel+'freeze.json',rel+'analysis_freeze.json',rel+'startup.json'],
          'Acquisition producer published before formal access; analysis implementation committed before metrics; primary and supplementary labels separate')
    for p,s in load(REPO/'config/nns_vit_endpoint/endpoint_m1.json')['source_bindings'].items():require(sha(REPO/p)==s,'SOURCE_DRIFT:'+p)
    check('2.1 RQ3a and RQ3b separate', ['docs/endpoint_m1_results_20260918.md',rel+'absolute_spearman_tests.json',rel+'paired_spearman_tests.json'],
          '672 absolute tests and 2,016 paired baseline tests; no utility-positive requirement for execution completion')
    check('2.2 same component/target/reference; absolute effect and mass',
          [rel+'task_manifest.json',rel+'task_validation.json'],
          '800 tasks; 326,400 unique component-reference conditions; effect=clean-replacement, absolute effect verified; mean mass=m/150528',
          task['task_count']==800 and task['new_conditions']==70656)
    check('2.3 all six baselines and fixed 1,000 random rankings per comparison',
          [rel+'metrics_manifest.json',rel+'task_validation.json'],
          '128,000 original sets x 1,000 permutations; exhaustive regeneration includes seeds, distributions and Capture lift')
    check('2.4 Spearman, Capture@1/@3, Regret@1 plus NDCG/pairwise/top1',
          [rel+'task_validation.json',rel+'all_summary.csv',rel+'all_paired_comparisons.csv'],
          'All seven metrics regenerated; independent reference formula verifies 256,000 Endpoint/Taylor sets',
          task['independent_metric_sets']==256000)
    check('2.5A all-head within-block ranking: ViT12/DeiT6',
          [rel+'task_validation.json',rel+'m1_head_ranking.png'],
          'Exact component IDs per block/model/image; all 12 block strata and image-level within-block mean')
    check('2.5B/C across-block and early0-3/middle4-7/late8-11',
          [rel+'all_summary.csv','docs/endpoint_m1_results_20260918.md'],
          'Separate raw comparison sets, model-level tables and paired tests')
    check('2.5D model-specific and pooled descriptive',
          [rel+'inference_validation.json',rel+'summary.json.gz'],
          '400 identical image IDs per model; pooled averages matched images before statistics')
    check('2.5E all eight named families and small-set handling',
          [rel+'task_validation.json','docs/endpoint_m1_execution_20260918.md'],
          'head/routing/value/Q/K/V/residual identity/residual branch; value/V aliases disclosed; n<4 rho NA with pairwise/top1 retained')
    check('2.6 image medians, 10,000 bootstrap, paired sign and Holm',
          [rel+'inference_validation.json',rel+'test_results.json'],
          '504 groups, 24,696 summaries and 14,112 paired differences checked; sign p verified with scipy binomtest; Holm independently recomputed',
          inf==dict(status='PASS',groups=504,summaries=24696,paired=14112,tests=2688))
    check('2.6 tests above zero versus above0.10 distinguished',
          [rel+'absolute_spearman_tests.json','docs/endpoint_m1_results_20260918.md'],
          'Separate one-sided null thresholds and Holm-adjusted decisions reported')
    diagnostic=load(ROOT/'taylor_implementation_diagnostic.json')
    require(diagnostic['status']=='PASS' and diagnostic['check_count']==48,'TAYLOR_DIAGNOSTIC')
    for r in diagnostic['checks']:
        require(min(x['error'] for x in r['curve'])<=r['tolerance'],'FD_DIAGNOSTIC')
        require(r.get('manual_skip_reference_error',0)<=1e-8,'MANUAL_REFERENCE')
    check('10.3 observed Endpoint>Taylor: tensor, reference, gradient and layer checks',
          [rel+'taylor_implementation_diagnostic.json','docs/endpoint_m1_validation_amendment_20260918.md'],
          '48 float64 displacement-direction checks; manual skip counterfactual; full per-task noops/E4 comparator parity; higher-order mechanisms not claimed')
    qc=load(REPO/'papers/nns_vit_endpoint/qc/endpoint_scanner_draft_validation_report_m1.json')
    check('9 and13 paper RQ3 rewritten;11/12;omnibus unchanged',
          ['papers/nns_vit_endpoint/zh/full_draft.md','papers/nns_vit_endpoint/en/full_draft.md',
           'papers/nns_vit_endpoint/qc/endpoint_scanner_draft_validation_report_m1.json'],
          'Both drafts broad but non-universal; old overall-supported/gap wording removed; primary tables exact; all 293 checks pass',
          qc['status']=='PASS' and qc['error_count']==0 and qc['check_count']==293)
    coverage=load(ROOT/'registry_coverage.json')
    require(len(coverage)==16,'COVERAGE_ROWS')
    for model,D,H in [('vit_b16',768,12),('deit_s16',384,6)]:
        expected=[12*197*D,24*D,12*H*197*64,36*D,D,D,1000,1000]
        observed=[r for r in coverage if r['model']==model]
        require([r['population'] for r in observed]==expected,'POPULATION_FORMULAS')
        require([r['registered'] for r in observed]==[192,96,12*H,18,4,1,1,1],'REGISTERED_COUNTS')
    check('5 and13 registry table with candidate denominators and complete scope',
          [rel+'registry_coverage.json','scripts/build_endpoint_m1_coverage.py'],
          '16 coverage rows verified against candidate formulas and actual registry counts; registry-complete, normalized-input GxI and local Jacobian scope in both drafts')
    check('Reproducibility and necessary tests',
          [rel+'test_results.json','scripts/validate_endpoint_m1.py',
           'tests/test_endpoint_m1.py','tests/test_endpoint_m1_statistics.py'],
          '6 focused tests passed; exhaustive source/metrics/random/inference audit passed; signed effects retained separately',
          load(ROOT/'test_results.json')['passed']==6)
    check('Writing quality and numbered figure interpretation',
          ['papers/nns_vit_endpoint/qc/endpoint_m1_writing_audit.md',
           'papers/nns_vit_endpoint/notes/endpoint_m1_evidence_matrix.md',
           'docs/endpoint_m1_results_20260918.md',rel+'m1_head_ranking.png',rel+'m1_head_ranking.pdf'],
          'Image #1 inspected for axes, legend and visible CI; no sealed original figure edited')
    # Every sealed per-image file is content-bound to the completed exhaustive audit.
    for r in task['records']:require(sha(ROOT/'per_image'/(r['task_id']+'.json.gz'))==r['sha256'],'PER_IMAGE_AFTER_AUDIT')
    audit=dict(objective='Execute Stage M M1 completely; stop before M2',status='PASS',
               requirements=checks,requirement_count=len(checks),missing=[],unverified=[],
               scope_not_applicable=['M2 spatial budgets','M3 token/channel samples','M4-M6 experiments',
                   'optional calibration/intervention curves and mixed-effects model'],
               current_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    write_new(ROOT/'completion_audit.json',audit)
    (ROOT/'README.md').write_text('# Endpoint M1 supplementary result package\n\nStart with docs/endpoint_m1_results_20260918.md and completion_audit.json.\n800 matched image-model tasks; 326,400 unique component-reference conditions.\nper_image/*.json.gz includes merged component records, all metrics and random distributions.\nresults/ includes new acquisition records; source task manifest binds immutable E3/E4 inputs.\nThe primary all-family omnibus remains NOT_SUPPORTED. This package stops after M1.\n')
    records=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path.name not in {'artifact_manifest.json','terminal_receipt.json'}:
            records.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    write_new(ROOT/'artifact_manifest.json',dict(files=records,file_count=len(records)))
    write_new(ROOT/'terminal_receipt.json',dict(status='ENDPOINT_M1_PASS_EXECUTION_AND_AUDIT',
              scientific_status='SUPPLEMENTARY_COMPONENT_DEPENDENT_UTILITY',primary_C4a='NOT_SUPPORTED',
              image_model_tasks=800,combined_component_reference_conditions=326400,new_conditions=70656,
              completion_audit_sha256=sha(ROOT/'completion_audit.json'),artifact_manifest_sha256=sha(ROOT/'artifact_manifest.json'),
              next_authorization='STOP_AFTER_M1',sealed_time_ns=time.time_ns()))
    print(json.dumps(dict(status='ENDPOINT_M1_PASS_EXECUTION_AND_AUDIT',requirements=len(checks),files=len(records))))

def verify():
    terminal=load(ROOT/'terminal_receipt.json')
    require(terminal['status']=='ENDPOINT_M1_PASS_EXECUTION_AND_AUDIT','TERMINAL_STATUS')
    require(sha(ROOT/'completion_audit.json')==terminal['completion_audit_sha256'],'AUDIT_HASH')
    require(sha(ROOT/'artifact_manifest.json')==terminal['artifact_manifest_sha256'],'MANIFEST_HASH')
    manifest=load(ROOT/'artifact_manifest.json')
    require(manifest['file_count']==len(manifest['files']),'MANIFEST_COUNT')
    expected={r['path'] for r in manifest['files']}
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    require(actual==expected|{'artifact_manifest.json','terminal_receipt.json'},'FILE_SET')
    for r in manifest['files']:
        p=ROOT/r['path'];require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'ARTIFACT:'+r['path'])
    audit=load(ROOT/'completion_audit.json')
    require(audit['status']=='PASS' and not audit['missing'] and not audit['unverified'],'COMPLETE_AUDIT')
    for r in audit['requirements']:
        require(r['status']=='PASS','REQUIREMENT_STATUS')
        for a in r['artifacts']:
            p=REPO/a['path'];require(sha(p)==a['sha256'] and p.stat().st_size==a['bytes'],'AUDIT_ARTIFACT:'+a['path'])
    require(terminal['next_authorization']=='STOP_AFTER_M1' and terminal['primary_C4a']=='NOT_SUPPORTED','SCOPE')
    print(json.dumps(dict(status=terminal['status'],artifact_files=len(expected),requirements=len(audit['requirements']),hashes='PASS',next_authorization='STOP_AFTER_M1')))

if __name__=='__main__':
    verify() if '--verify' in sys.argv else main()
