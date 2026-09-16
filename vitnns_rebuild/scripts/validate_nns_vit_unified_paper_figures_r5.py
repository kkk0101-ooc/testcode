#!/usr/bin/env python3
"""Validate measured Figure 1, unchanged Figure 2–14 and R5 package integrity."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
from pathlib import Path
import json
import re
import statistics
import numpy as np
from PIL import Image
import validate_nns_vit_unified_paper_figures_r4 as common

sha,read,write,signed,verify,canonical=common.sha,common.read,common.write,common.signed,common.verify,common.canonical
REPO=common.REPO
ROOT=REPO/'reports/nns_vit_unified_paper_figures_r5_20260916'
OLD=common.ROOT
EXCLUSIONS=common.EXCLUSIONS


def validate():
    errors=[]
    def check(ok,msg):
        if not ok: errors.append(msg)
    def checked(path):
        value=read(path); check(verify(value),'digest '+str(path)); return value
    manifest=checked(ROOT/'render_manifest.json'); old=checked(OLD/'render_manifest.json')
    check(manifest['figure_count']==14 and manifest['artifact_count']==42 and manifest['redesigned_figures']==[1],'counts and scope')
    check([f['number'] for f in manifest['figures']]==list(range(1,15)),'figure roster')
    output_paths=[]
    for f in manifest['figures']:
        n=f['number']; check(len(f['outputs'])==3 and {r['format'] for r in f['outputs']}=={'PNG','PDF','SVG'},'formats '+str(n))
        check(f['redesigned']==(n==1),'redesigned flag '+str(n))
        for o in f['outputs']:
            p=REPO/o['path']; output_paths.append(p)
            check(p.is_relative_to(ROOT) and not p.is_symlink(),'local file '+str(p))
            check(p.is_file() and p.stat().st_size==o['bytes'] and sha(p)==o['sha256'],'output hash '+str(p))
            source=next(r for r in old['figures'][n-1]['outputs'] if r['format']==o['format'])
            check((o['sha256']==source['sha256'])==(n!=1),'copy/redesign '+str(n))
            if o['format']=='PNG':
                with Image.open(p) as im:
                    im.load(); check(im.width>=1200 and im.height>=700,'PNG dimensions')
                    check(all(abs(d-400)<1 for d in im.info.get('dpi',(0,0))),'PNG dpi')
            elif o['format']=='PDF': check(p.read_bytes().startswith(b'%PDF'),'PDF signature')
            else: check(b'<svg' in p.read_bytes()[:4096],'SVG signature')
    actual=[p for folder in ['figures_png','figures_pdf','figures_svg'] for p in (ROOT/folder).iterdir()]
    check(len(actual)==42 and set(actual)==set(output_paths),'exact file roster')
    sources=manifest['source_files']; check(canonical(sources)==manifest['source_files_digest'],'sources digest')
    for r in sources:
        p=REPO/r['path']; check(p.is_file() and sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],'source drift '+r['path'])
    predecessor=checked(ROOT/'predecessor_snapshot.json')
    current=[{**r,'path':OLD.relative_to(REPO).as_posix()+'/'+r['path']} for r in common.inventory(OLD)]
    check(current==predecessor['files'] and canonical(current)==predecessor['files_digest'],'R4 root changed')
    d=checked(ROOT/'numeric_display_data.json'); n=d['figure_1']
    check(d['inherited_R4_numeric']==checked(OLD/'numeric_display_data.json'),'R4 numeric parity')
    f3=checked(REPO/'evidence/stage_f/f3_r7_recovery_v2_20260907/formal/panel_manifest.json')
    case=next(r for r in f3['panels'] if r.get('selection_order')==44 and r.get('sources'))
    source=next(r for r in case['sources'] if r['method']=='conditional_fixed_clean_attention_head_direction_value_path_ig')
    check(n['selection_order']==44 and n['selection_rule']=='median_effect' and case['selection_rules']==['median_effect'],'frozen case selection')
    check(n['sample_id']==case['sample_id'] and n['coordinate']==case['coordinate']==[0,9],'case target')
    check(n['map_path']==source['source_map_path'] and sha(REPO/n['map_path'])==n['map_sha256']==source['source_map_file_sha256'],'map identity')
    check(n['jpeg_path']==case['jpeg']['path'] and sha(REPO/n['jpeg_path'])==n['jpeg_sha256']==case['jpeg']['file_sha256'],'image identity')
    raw=np.load(REPO/n['map_path'],allow_pickle=False)
    check(raw.shape==(56,56) and n['map_shape']==[56,56] and np.isfinite(raw).all(),'measured map shape and values')
    check(float(np.max(np.abs(raw)))==n['max_abs'] and n['kappa']==99 and n['display_transform']=='signed_log1p' and n['interpolation']=='nearest','display transform')
    check(float(raw.min())<0<float(raw.max()),'signed map')
    root=REPO/'evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis'
    with (root/'paper_data/per_sample_delta.csv').open() as stream: rows=list(csv.DictReader(stream))
    values=[float(r['delta_fid']) for r in rows if r['status']=='VALID']; summary=read(root/'analysis.json')['summary']
    check(n['delta_fid_values']==values and n['n']==len(values)==500 and n['positive_count']==sum(v>0 for v in values)==500,'500 observed signs')
    check(n['median']==statistics.median(values)==summary['median'],'median source parity')
    check(n['ci']==[summary['bootstrap_ci_lower'],summary['bootstrap_ci_upper']],'official CI source parity')
    check(n['target']=='fixed_clean_attention_cls_head_direction' and n['endpoint']=='fine_4x4_cell_delta_fid','target and endpoint')
    for language in ['zh','en']:
        sections=common.sections(ROOT/f'captions_{language}.md','## Figure '); prior=common.sections(OLD/f'captions_{language}.md','## Figure ')
        check(set(sections)==set(range(1,15)),'caption coverage '+language)
        for i in range(2,15): check(sections[i]==prior[i],'caption inheritance '+str(i))
        check('500/500' in sections[1] or 'All 500' in sections[1],'Figure 1 cohort caption')
    sections=common.sections(ROOT/'figure_analysis.md','## Image #'); prior=common.sections(OLD/'figure_analysis.md','## Image #')
    check(set(sections)==set(range(1,15)),'analysis roster')
    for i in range(2,15): check(sections[i]==prior[i],'analysis inheritance '+str(i))
    for path in [*ROOT.glob('*.md'),ROOT/'gallery.html']:
        content=path.read_text(); links=re.findall(r'(?:href|src)="([^"]+)"',content) if path.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',content)
        for target in links:
            if target.startswith('#'): check('id="'+target[1:]+'"' in content,'anchor '+target)
            elif '://' not in target: check((path.parent/target.split('#')[0]).exists(),'broken link '+target)
    audit=checked(ROOT/'cpu_access_audit.json')
    check(audit['CUDA_VISIBLE_DEVICES']=='' and not audit['torch_imported'] and not audit['timm_imported'] and not audit['new_inference'],'CPU boundary')
    if not (ROOT/'manual_visual_qc.json').exists(): errors.append('visual review pending')
    else:
        qc=checked(ROOT/'manual_visual_qc.json'); check(qc['status']=='PASS' and [r['number'] for r in qc['records']]==list(range(1,15)),'QC roster')
        for r,f in zip(qc['records'],manifest['figures']):
            check(r['status']=='PASS' and r['png_sha256']==next(o['sha256'] for o in f['outputs'] if o['format']=='PNG'),'QC binding '+str(f['number']))
    if (ROOT/'terminal_receipt.json').exists():
        term=checked(ROOT/'terminal_receipt.json'); inv=checked(ROOT/'whole_root_inventory.json')
        check(term['status']=='PASS_PRESENTATION_RENDER' and term['validator_sha256']==sha(Path(__file__)),'terminal identity')
        check(inv['files']==common.inventory(ROOT,EXCLUSIONS) and canonical(inv['files'])==inv['files_digest'],'sealed inventory')
        for name,digest in term['bindings'].items(): check(sha(ROOT/name)==digest,'terminal binding '+name)
    return errors


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seal',action='store_true'); args=parser.parse_args(); sealed=(ROOT/'terminal_receipt.json').exists()
    if args.seal and sealed: raise RuntimeError('Already sealed')
    errors=validate(); report=signed({'status':'FAIL' if errors else 'PASS','revision':'R5','errors':errors,'figure_count':14,'artifact_count':42,'new_figures':[1],'unchanged_figures':13,'source_values_checked':500,'cpu_only':True})
    if not sealed: write(ROOT/'validation_report.json',report)
    if args.seal and not errors:
        files=common.inventory(ROOT,EXCLUSIONS); write(ROOT/'whole_root_inventory.json',signed({'files':files,'files_digest':canonical(files)}))
        write(ROOT/'terminal_receipt.json',signed({'status':'PASS_PRESENTATION_RENDER','revision':'R5','figure_count':14,'artifact_count':42,'validator_sha256':sha(Path(__file__)),'bindings':{name:sha(ROOT/name) for name in ['render_manifest.json','validation_report.json','manual_visual_qc.json','whole_root_inventory.json']},'new_inference':False}))
    print(json.dumps(report,ensure_ascii=False)); return bool(errors)


if __name__=='__main__': raise SystemExit(main())
