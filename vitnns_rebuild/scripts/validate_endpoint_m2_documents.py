#!/usr/bin/env python3
"""Bind M2 report text/tables/figure references to audited statistics."""
import json
from pathlib import Path
import re
import csv
from run_endpoint_m2 import ROOT,REPO,load,sha,write

def require(condition,code):
    if not condition:raise ValueError(code)

def main():
    report=load(ROOT/'report_data.json');figures=load(ROOT/'figure_manifest.json')['figures']
    paper=REPO/'papers/nns_vit_endpoint';papers=[paper/lang/'endpoint_m2_supplement.md' for lang in ('zh','en')]
    paths=[REPO/'docs/endpoint_m2_results_20260919.md',*papers,paper/'notes/endpoint_m2_evidence_matrix.md']
    texts=[p.read_text() for p in paths];link_count=0;table_count=0
    for path,text in zip(paths,texts):
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:','#')):continue
            require((path.parent/target.split('#')[0]).resolve().is_file(),'LINK:'+target);link_count+=1
    for path,text in zip(paths[:3],texts[:3]):
        for table in report['tables'].values():
            require(table in text,'REPORT_TABLE:'+path.name);table_count+=1
        for token in ('400','2','5','10','672','11/12','NOT_SUPPORTED','Gradient × Input','Image #1','Image #2','Image #3'):
            require(token in text,'REQUIRED_TEXT:'+token)
        require('10000' in text or '10,000' in text,'BOOTSTRAP_COUNT')
        require('M3' not in text or path.name=='endpoint_m2_results_20260919.md','NO_UNSUPPORTED_M3_CLAIM')
    # Both languages use the same section numbering and complete numeric tables.
    require(re.findall(r'^## (\d+)\.',texts[1],re.M)==re.findall(r'^## (\d+)\.',texts[2],re.M),'SECTION_PARITY')
    require(load(paper/'figure_manifest_m2.json')['figures']==figures,'FIGURE_MANIFEST_PARITY')
    require(len(figures)==3,'FIGURE_COUNT')
    for fig in figures:require(sha(REPO/fig['path'])==fig['sha256'],'FIGURE_SHA')
    require(report['primary_C4a']==report['primary_C4b']=='NOT_SUPPORTED','PRIMARY_STATUS')
    tests=load(ROOT/'auc_tests.json')['tests']
    for geometry,outcome,wins in [('patch','full',11),('patch','isolated',12),
                                  ('legacy_grid','full',9),('legacy_grid','isolated',12)]:
        part=[t for t in tests if t['geometry']==geometry and t['outcome']==outcome and t['baseline']=='regional_taylor']
        require(sum(t['reject'] for t in part if t['direction']=='baseline_greater')==wins,'NARRATIVE_TAYLOR_COUNT')
        require(sum(t['reject'] for t in part if t['direction']=='endpoint_greater')==0,'NARRATIVE_ENDPOINT_COUNT')
    exception=next(t for t in tests if (t['model'],t['reference'],t['block'],t['geometry'],t['outcome'],t['baseline'],t['direction'])
        ==('deit_s16','normalized_zero',5,'patch','full','regional_taylor','baseline_greater'))
    require(format(exception['holm_p'],'.3f')=='0.528','NARRATIVE_EXCEPTION_P')
    with (ROOT/'paired.csv').open() as f:
        pair=next(r for r in csv.DictReader(f) if (r['model'],r['reference'],r['block'],r['geometry'],r['outcome'],r['metric'],r['baseline'])
            ==('deit_s16','normalized_zero','5','patch','full','auc','regional_taylor'))
    require(format(float(pair['median']),'.4f')=='-0.0424','NARRATIVE_EXCEPTION_DELTA')
    require([format(v,'.4f') for v in json.loads(pair['ci95'])]==['-0.0614','-0.0142'],'NARRATIVE_EXCEPTION_CI')
    require(sha(ROOT/'task_validation.json')==report['task_validation_sha256'],'TASK_BINDING')
    require(sha(ROOT/'inference_validation.json')==report['inference_validation_sha256'],'INFERENCE_BINDING')
    require((paper/'qc/endpoint_m2_writing_audit.md').is_file(),'WRITING_AUDIT')
    english=texts[2].lower()
    for phrase in ('supplementary frozen study','does not establish equivalence','global optimum'):
        require(english.count(phrase)==1,'WRITING_PHRASE_COUNT:'+phrase)
    for phrase in ('we emphasize','it should be noted','to ensure fairness','we do not claim'):
        require(phrase not in english,'DEFENSIVE_PHRASE:'+phrase)
    visual=load(ROOT/'visual_inspection.json');require(visual['status']=='PASS' and len(visual['figures'])==3,'VISUAL_INSPECTION')
    write(ROOT/'document_validation.json',dict(status='PASS',documents=len(paths),links=link_count,
        bound_table_copies=table_count,figure_count=3,bilingual_section_parity=True,primary_omnibus='NOT_SUPPORTED',
        sources={p.relative_to(REPO).as_posix():sha(p) for p in paths}))
    print('M2_DOCUMENT_VALIDATION_PASS')

if __name__=='__main__':main()
