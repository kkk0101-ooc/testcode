#!/usr/bin/env python3
"""Revision-v1 audit: sources, actual tables, math, links, figures and requirement coverage."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
import struct
import sys
import xml.etree.ElementTree as ET

PAPER=Path(__file__).resolve().parents[1]
REPO=PAPER.parents[1]
ROOT=REPO/'reports/nns_vit_endpoint_paper_revision_v1_20260920'
R2=REPO/'reports/nns_vit_endpoint_all_figures_r2_20260919'
CHECKS=[]
def check(name,value,detail=''):CHECKS.append(dict(check=name,passed=bool(value),detail=detail))
def load(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def canonical(d):return hashlib.sha256(json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def main():
    texts={lang:(PAPER/lang/'full_draft.md').read_text() for lang in ('zh','en')}
    supp={lang:(PAPER/lang/'supplement_revision_v1.md').read_text() for lang in ('zh','en')}
    manifest=load(PAPER/'figure_manifest.json');analysis=load(ROOT/'analysis.json')
    check('opinion_binding',analysis['opinion_sha256']==sha(REPO/'修改意见/修改意见v1.md'))
    check('figure_manifest_copy',manifest==load(ROOT/'manifest.json'))
    check('main_figure_count',manifest['figure_count']==10 and [r['figure'] for r in manifest['figures']]==list(range(1,11)))
    check('archived_figures',len(manifest['supplemental_records'])==92)
    paths=[]
    for f in manifest['figures']:
        for kind in ('png','pdf','svg'):
            r=f[kind];p=REPO/r['path'];paths.append(r['path'])
            check('figure_sha:'+r['path'],p.is_file() and sha(p)==r['sha256'])
            raw=p.read_bytes()
            if kind=='png':
                width,height=struct.unpack('>II',raw[16:24]);check('png_dimensions:'+str(f['figure']),width>=1400 and height>=650,str((width,height)))
            elif kind=='svg':
                tree=ET.fromstring(raw);check('svg_parse:'+str(f['figure']),tree.tag.endswith('svg'))
            else:check('pdf_header:'+str(f['figure']),raw.startswith(b'%PDF-') and b'%%EOF' in raw[-2048:])
    for r in manifest['supplemental_records']:
        for f in r.get('outputs',[r]):
            p=REPO/f['path'];check('supplement_sha:'+f['path'],p.is_file() and sha(p)==f['sha256'])
    # Current source hashes, not copied green statuses.
    for name in ('analysis_sources.json','figure_sources.json'):
        for path,r in load(ROOT/name).items():
            p=REPO/path;check('source_sha:'+path,p.is_file() and sha(p)==r['sha256'])
    inv=load(R2/'whole_root_inventory.json')
    for r in inv['records']:check('immutable_r2:'+r['path'],sha(R2/r['path'])==r['sha256'])
    for name in ('whole_root_inventory.json','manifest.json','display_receipt.json','source_inventory.json'):
        d=load(R2/name);expected=d.pop('canonical_digest');check('r2_canonical:'+name,canonical(d)==expected)
    for root in ('nns_vit_endpoint_e5_spatial_r1_20260918','nns_vit_endpoint_e6_release_r3_20260918','nns_vit_endpoint_m1_r1_20260918'):
        p=REPO/'reports'/root;m=load(p/'artifact_manifest.json');t=load(p/'terminal_receipt.json')
        if 'canonical_digest' in m:
            expected=m.pop('canonical_digest');check('manifest_digest:'+root,canonical(m['files'])==expected and t['manifest_digest']==expected)
        else:check('manifest_sha:'+root,sha(p/'artifact_manifest.json')==t['artifact_manifest_sha256'])
    e6=load(REPO/'reports/nns_vit_endpoint_e6_release_r3_20260918/terminal_receipt.json')
    check('frozen_omnibus',e6['C4a']==e6['C4b']=='NOT_SUPPORTED')
    families=load(REPO/'reports/nns_vit_endpoint_e6_release_r3_20260918/family_decisions.json')['families']
    check('family_support',sum(r['claim']=='C4a' and r['status']=='SUPPORTED' for r in families)==11 and not any(r['claim']=='C4b' and r['status']=='SUPPORTED' for r in families))
    # All source-generated main table rows must actually appear in both manuscripts.
    from bind_revision_v1 import build_tables
    tables,_=build_tables()
    for lang,text in texts.items():
        check('no_unexpanded_template:'+lang,'{{' not in text and '\x00' not in text and '\u0000' not in text)
        check('no_control_bytes:'+lang,not any(ord(c)<32 and c not in '\n\r\t' for c in text))
        check('balanced_math:'+lang,text.count('$$')%2==0 and text.count('$$')>=16)
        check('ten_embedded_figures:'+lang,re.findall(r'!\[Figure (\d+)\]',text)==[str(i) for i in range(1,11)])
        check('rq_result_order:'+lang,re.findall(r'^### 5\.\d+ (RQ[1-5][abc]?)',text,re.M)==['RQ1','RQ2','RQ3','RQ4a','RQ4b','RQ4c','RQ5a','RQ5b'])
        for n,t in tables.items():check(f'source_table:{lang}:{n}',t in text)
        for n in range(1,6):check(f'supp_table:{lang}:{n}',f'Table S{n}.' in supp[lang])
        check('compressed_main:'+lang,len(text)<.65*len((PAPER/lang/'full_draft_r3_archive.md').read_text()))
        prose=re.sub(r'!\[[^]]*\]\([^)]*\)','',text)
        check('engineering_detail_in_supplement:'+lang,not re.search(r'PASS_FLOAT64|ENDPOINT_M[1-6]_PASS|reports/nns|Review-R2',prose))
        check('claim_decisions_present:'+lang,'C4a' in text and 'C4b' in text and 'NOT_SUPPORTED' in text)
    # Exact display equations and numeric prose tokens must match between languages.
    check('bilingual_display_equations',re.findall(r'\$\$(.*?)\$\$',texts['zh'],re.S)==re.findall(r'\$\$(.*?)\$\$',texts['en'],re.S))
    def structure(t):return re.findall(r'^#{2,3} (\d+(?:\.\d+)?)\b',t,re.M)
    check('bilingual_section_structure',structure(texts['zh'])==structure(texts['en']))
    for n in range(1,8):
        rx=rf'<!-- TABLE_{n}_START -->(.*?)<!-- TABLE_{n}_END -->'
        a=re.search(rx,texts['zh'],re.S).group(1);b=re.search(rx,texts['en'],re.S).group(1)
        check('bilingual_table:'+str(n),a[a.index('|'):]==b[b.index('|'):])
    # Recompute image-level AUC/regret aggregates independently from the released CSV.
    image_rows=list(csv.DictReader((ROOT/'spatial_image_values.csv').open()))
    for r in analysis['spatial']['summary']:
        rr=[x for x in image_rows if (x['model'],x['reference'],x['method'])==(r['model'],r['reference'],r['method'])]
        check('auc_coverage:'+str((r['model'],r['reference'],r['method'])),len(rr)==len({x['sample_id'] for x in rr})==400)
        for key in ('auc','normalized_regret','relative_to_abs_clean_logit','clean_logit'):
            vals=[float(x[key]) for x in rr if x[key]]
            check('aggregate:'+str((r['model'],r['reference'],r['method'],key)),st.median(vals)==r[key]['median'])
        for x in rr:
            au,oracle,cl=float(x['auc']),float(x['oracle_auc']),float(x['clean_logit'])
            if oracle:assert math.isclose(float(x['normalized_regret']),(oracle-au)/oracle,abs_tol=1e-14)
            if cl:assert math.isclose(float(x['relative_to_abs_clean_logit']),au/abs(cl),abs_tol=1e-14)
    check('raw_curve_reconstruction',analysis['spatial']['curve_auc_max_error']==0 and analysis['spatial']['e6_image_score_max_error']<1e-12)
    check('reference_coverage',analysis['reference']['component_pairs']==163200 and analysis['reference']['endpoint_max_difference']==0)
    check('no_added_inference',analysis['new_inferential_tests']==0)
    # Bind the prose's quantitative mechanism claims to sealed summary records.
    evidence_sources={}
    def evidence(root,name):
        p=REPO/'reports'/root;manifest=load(p/'artifact_manifest.json');terminal=load(p/'terminal_receipt.json')
        if 'artifact_manifest_sha256' in terminal:
            check('evidence_manifest:'+root,sha(p/'artifact_manifest.json')==terminal['artifact_manifest_sha256'])
        record=next(r for r in manifest['files'] if r['path']==name)
        check('evidence_file:'+root+'/'+name,sha(p/name)==record['sha256'])
        evidence_sources[str((p/name).relative_to(REPO))]=record['sha256']
        return list(csv.DictReader((p/name).open())) if name.endswith('.csv') else load(p/name)
    def prose_number(value,precision=4):
        token=f'{abs(value):.{precision}f}'
        for lang,text in texts.items():check('prose_number:'+lang+':'+token,token in text)
    e3=load(REPO/'reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json')['independent_validation']
    check('e3_counts',e3['positive_closure_evaluations']==76800 and e3['negative_control_evaluations']==28800)
    e2=load(REPO/'reports/nns_vit_endpoint_e2_atlas_r4_20260918/terminal_receipt.json')
    check('e2_counts',e2['image_model_task_count']==800 and e2['target_evaluation_count']==336800)
    fid=evidence('nns_vit_endpoint_m3_r1_20260919','m3a_fidelity.json')
    check('fidelity_5000',fid['comparison_count']==5000 and fid['failure_count']==0)
    op=evidence('nns_vit_endpoint_m3_r1_20260919','m3a_operator_validation.json')
    check('operator_88',op['record_count']==88 and op['failure_count']==0)
    fail=evidence('nns_vit_endpoint_m3_r1_20260919','initial_numeric_validation_failure.json')
    check('retained_failures',fail['failed_condition_count']==36 and fail['total_target_direction_conditions']==57600)
    for r in evidence('nns_vit_endpoint_m3_r1_20260919','atlas_tests.json')['tests']:
        if r['contrast'] in ('late_minus_early_entropy','late_minus_early_locality','late_minus_early_distance'):
            prose_number(r['median_difference']);check('atlas_direction:'+r['test_id'],max(r['positive'],r['negative'])==100 and r['reject'])
    for r in evidence('nns_vit_endpoint_m4_r1_20260919','attention_block_summary.csv'):
        if r['block'] in ('0','11') and r['metric'] in ('routing_share','frozen_attention_endpoint_error'):prose_number(float(r['median']))
    for r in evidence('nns_vit_endpoint_m4_review_r2_20260919','jvp_block_summary.csv'):
        if r['block']=='11' and r['metric'] in ('relative_error','absolute_error','full_response_denominator'):prose_number(float(r['median']))
    for r in evidence('nns_vit_endpoint_m4_review_r2_20260919','residual_spatial_tests.json')['tests']:
        prose_number(r['median']);check('residual_posthoc:'+r['test_id'],r['reject'])
    for r in evidence('nns_vit_endpoint_m4_review_r2_20260919','saturation_sensitivity_summary.csv'):
        if r['phase']=='all' and r['saturation']=='mean_query_saturation' and r['outcome']=='routing_share' and r['adjustment']=='adjust_output_and_cotangent_norm':prose_number(float(r['median']))
    (ROOT/'manuscript_numeric_sources.json').write_text(json.dumps(evidence_sources,indent=2)+'\n')
    display=load(ROOT/'display_receipt.json')['panels']
    raw_display=load(ROOT/'raw_display_validation.json')
    check('raw_display_reconstruction',raw_display['status']=='PASS' and raw_display['panel_count']==26 and raw_display['manual_qc_matches_current_images'])
    visual=load(ROOT/'manual_visual_qc.json')
    check('visual_review_coverage',visual['reviewed_figures']==list(range(1,11)))
    for f,q in zip(manifest['figures'],visual['outputs']):
        check('visual_review_current:'+str(f['figure']),f['figure']==q['figure'] and f['png']['sha256']==q['png_sha256'])
    for group in ('F1','F5','F6','F7','F8-deit_s16','F8-vit_b16','F10'):
        for view in ('signed','absolute'):
            p=[x for x in display if x['group']==group and x['view']==view]
            check('shared_scale:'+group+':'+view,bool(p) and len({r['scale'] for r in p})==1)
            check('raw_rgb_display:'+group+':'+view,all(r['raw_shape']==[3,224,224] and r['rgb_reduction']==('sum(E)' if view=='signed' else 'sum(abs(E))') for r in p))
    for lang,text in texts.items():
        phrase_sets=(['命题 1','命题 2','每条相关','同一前向状态','cotangent','0.10/0.05','随机初始化','clean-top1','同目标','原点'] if lang=='zh' else ['Proposition 1','Proposition 2','each relevant','same forward state','cotangent','0.10/0.05','Random-initialization','clean-top1','matched-target','origin'])
        for phrase in phrase_sets:check('requirement_phrase:'+lang+':'+phrase,phrase in text)
    required=[PAPER/'README.md',PAPER/'zh/outline.md',PAPER/'notes/revision_v1_protocol.md',PAPER/'notes/revision_v1_evidence_matrix.md',PAPER/'qc/revision_v1_writing_audit.md',PAPER/'qc/revision_v1_completion_audit.md',ROOT/'README.md',ROOT/'figure_analysis.md',ROOT/'manual_visual_qc.json']
    for p in required:check('deliverable:'+str(p.relative_to(REPO)),p.is_file())
    md=[PAPER/lang/name for lang in ('zh','en') for name in ('full_draft.md','supplement_revision_v1.md')]+[p for p in required if p.suffix=='.md' and p.is_file()]
    for p in md:
        for link in re.findall(r'(?<!!)\[[^]]*\]\(([^)]+)\)|!\[[^]]*\]\(([^)]+)\)',p.read_text()):
            target=next(x for x in link if x)
            if '://' in target or target.startswith('#'):continue
            check('link:'+str(p.relative_to(REPO)),(p.parent/target.split('#')[0]).exists(),target)
    for name in ('analyze_revision_v1.py','render_revision_v1.py','bind_revision_v1.py'):
        tree=ast.parse((PAPER/'scripts'/name).read_text());imports=[n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import)]
        check('no_model_import:'+name,not set(imports)&{'torch','timm'})
    errors=[r for r in CHECKS if not r['passed']]
    report=dict(status='PASS' if not errors else 'FAIL',check_count=len(CHECKS),errors=errors,checks=CHECKS,
      figure_count=10,supplemental_records=92,scope='manuscript_and_existing_evidence_presentation_revision',
      deferred_new_experiments=['random_initialization','target_robustness','matched_target_bridge','CNN_visual_page','alternative_checkpoint','functional_method_display'])
    (PAPER/'qc/revision_v1_validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2))
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
