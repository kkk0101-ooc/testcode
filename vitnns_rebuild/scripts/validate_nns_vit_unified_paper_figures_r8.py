#!/usr/bin/env python3
"""Validate R8 exports, signed source, inherited figures and visual review."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
import validate_nns_vit_unified_paper_figures_r4 as common

REPO = common.REPO
ROOT = REPO/'reports/nns_vit_unified_paper_figures_r8_20260916'
OLD = REPO/'reports/nns_vit_unified_paper_figures_r7_20260916'
EXCLUSIONS = {'terminal_receipt.json','validation_report.json','whole_root_inventory.json'}


def validate(require_review=True):
    errors = []
    def check(condition, message):
        if not condition: errors.append(message)
    def checked(path):
        value = common.read(path)
        check(common.verify(value), 'digest: '+str(path))
        return value
    manifest=checked(ROOT/'render_manifest.json'); old=checked(OLD/'render_manifest.json')
    check(manifest['redesigned_figures']==[2] and manifest['figure_count']==14 and manifest['artifact_count']==42,'scope')
    check([row['number'] for row in manifest['figures']]==list(range(1,15)),'roster')
    output_paths=[]
    for figure in manifest['figures']:
        number=figure['number']
        check(figure['redesigned']==(number==2),'redesigned flag')
        check(len(figure['outputs'])==3 and {row['format'] for row in figure['outputs']}=={'PNG','PDF','SVG'},'formats')
        for output in figure['outputs']:
            path=REPO/output['path'];output_paths.append(path)
            check(path.is_relative_to(ROOT) and not path.is_symlink(),'local export')
            check(path.is_file() and common.sha(path)==output['sha256'] and path.stat().st_size==output['bytes'],'export identity')
            previous=next(row for row in old['figures'][number-1]['outputs'] if row['format']==output['format'])
            check((previous['sha256']==output['sha256'])==(number!=2),'inheritance')
            if output['format']=='PNG':
                with Image.open(path) as image:
                    image.load()
                    check(image.width>=1200 and image.height>=700,'PNG dimensions')
                    check(all(abs(value-400)<1 for value in image.info.get('dpi',(0,0))),'PNG dpi')
            elif output['format']=='PDF':check(path.read_bytes().startswith(b'%PDF'),'PDF signature')
            else:check(b'<svg' in path.read_bytes()[:4096],'SVG signature')
    actual=[path for folder in ('figures_png','figures_pdf','figures_svg') for path in (ROOT/folder).iterdir()]
    check(len(actual)==42 and set(actual)==set(output_paths),'export file roster')
    sources=manifest['source_files']
    check(common.canonical(sources)==manifest['source_files_digest'],'sources digest')
    check(checked(ROOT/'input_inventory.json')['source_files']==sources,'input inventory')
    for source in sources:
        path=REPO/source['path']
        if not path.is_file():path=REPO.parent/source['path']
        check(path.is_file() and common.sha(path)==source['sha256'] and path.stat().st_size==source['bytes'],'source drift: '+source['path'])
    previous=checked(ROOT/'predecessor_snapshot.json')
    check(previous['files']==common.inventory(OLD) and previous['files_digest']==common.canonical(previous['files']),'R7 mutation')
    numeric=checked(ROOT/'numeric_display_data.json')
    check(numeric['inherited_R7_numeric']==checked(OLD/'numeric_display_data.json'),'numeric inheritance')
    import render_nns_vit_unified_paper_figures_r8 as renderer
    _,raw,metadata=renderer.load_case()
    check(numeric['figure_2']==metadata,'Figure 2 source')
    check(metadata['coordinate']==[0,10] and metadata['primary_result_coordinate']==[0,9],'illustration coordinate')
    positive,negative=np.maximum(raw,0),np.maximum(-raw,0)
    check(not np.any((positive>0)&(negative>0)) and np.array_equal(positive-negative,raw),'signed reconstruction')
    top,bottom,left,right=metadata['crop_row_col_bounds_exclusive'];crop=raw[top:bottom,left:right]
    check(crop.shape==(16,16) and crop.min()<0<crop.max(),'same signed crop')
    check(metadata['interpolation']=='nearest' and not metadata['spatial_smoothing'] and metadata['kappa']==9,'display transform')
    references=checked(ROOT/'reference_gallery_manifest.json')
    check(references==checked(OLD/'reference_gallery_manifest.json'),'reference inheritance')
    for record in references['records']:
        check(common.sha(ROOT/record['path'])==record['sha256']==common.sha(OLD/record['path']),'reference identity')
    for name,marker in [('captions_zh.md','## Figure '),('captions_en.md','## Figure '),('figure_analysis.md','## Image #')]:
        sections=common.sections(ROOT/name,marker);previous_sections=common.sections(OLD/name,marker)
        check(set(sections)==set(range(1,15)),'text coverage')
        for number in range(1,15):
            if number!=2:check(sections[number]==previous_sections[number],'text inheritance')
        if name=='captions_zh.md':
            check('查看结果后' in sections[2] and 'B0/H9' in sections[2] and 'H10' in sections[2],'selection disclosure')
    for path in [*ROOT.glob('*.md'),ROOT/'gallery.html']:
        content=path.read_text()
        links=re.findall(r'(?:href|src)="([^"]+)"',content) if path.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',content)
        for target in links:
            if target.startswith('#'):check('id="'+target[1:]+'"' in content,'anchor')
            elif '://' not in target:check((path.parent/target.split('#')[0]).exists(),'link: '+target)
    check(checked(ROOT/'layout_audit.json')['records']==[{'number':2,'text_outside_canvas':[]}],'layout')
    audit=checked(ROOT/'cpu_access_audit.json')
    check(audit['CUDA_VISIBLE_DEVICES']=='' and not audit['torch_imported'] and not audit['timm_imported'] and not audit['new_inference'],'CPU boundary')
    if require_review or (ROOT/'manual_visual_qc.json').exists():
        if not (ROOT/'manual_visual_qc.json').exists():errors.append('visual review pending')
        else:
            review=checked(ROOT/'manual_visual_qc.json')
            check(review['status']=='PASS' and [row['number'] for row in review['records']]==list(range(1,15)),'QC roster')
            for row,figure in zip(review['records'],manifest['figures']):
                png=next(output for output in figure['outputs'] if output['format']=='PNG')
                check(row['png_sha256']==png['sha256'] and row['status']=='PASS','QC binding')
                check(row['review_mode']==('direct_visual_review' if figure['number']==2 else 'inherited_sealed_review_and_byte_parity'),'QC mode')
    if (ROOT/'terminal_receipt.json').exists():
        terminal=checked(ROOT/'terminal_receipt.json');inventory=checked(ROOT/'whole_root_inventory.json')
        check(terminal['status']=='PASS_PRESENTATION_RENDER' and terminal['validator_sha256']==common.sha(Path(__file__)),'terminal identity')
        check(inventory['files']==common.inventory(ROOT,EXCLUSIONS) and common.canonical(inventory['files'])==inventory['files_digest'],'sealed inventory')
        for name,digest in terminal['bindings'].items():check(common.sha(ROOT/name)==digest,'terminal binding')
    return errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');parser.add_argument('--draft',action='store_true');args=parser.parse_args()
    sealed=(ROOT/'terminal_receipt.json').exists()
    if args.seal and (sealed or args.draft):parser.error('Sealing requires an unsealed root and visual review')
    errors=validate(not args.draft)
    report=common.signed({'revision':'R8','status':'FAIL' if errors else 'PASS_DRAFT_CHECKS' if args.draft else 'PASS','errors':errors,'figure_count':14,'artifact_count':42,'redesigned_figures':[2],'inherited_figures':13,'reference_pngs':50,'cpu_only':True,'new_inference':False})
    if not sealed:common.write(ROOT/'validation_report.json',report)
    if args.seal and not errors:
        files=common.inventory(ROOT,EXCLUSIONS)
        common.write(ROOT/'whole_root_inventory.json',common.signed({'files':files,'files_digest':common.canonical(files)}))
        common.write(ROOT/'terminal_receipt.json',common.signed({'revision':'R8','status':'PASS_PRESENTATION_RENDER','validator_sha256':common.sha(Path(__file__)),'bindings':{name:common.sha(ROOT/name) for name in ['render_manifest.json','validation_report.json','manual_visual_qc.json','whole_root_inventory.json']},'new_inference':False}))
    print(json.dumps(report,ensure_ascii=False));return bool(errors)


if __name__=='__main__':raise SystemExit(main())
