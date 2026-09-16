#!/usr/bin/env python3
"""Check R4 exports, predecessor parity, explanatory examples and review binding."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'reports/nns_vit_unified_paper_figures_r4_20260916'
R2=REPO/'reports/nns_vit_unified_paper_figures_r2_20260916'
R3=REPO/'reports/nns_vit_unified_paper_figures_r3_20260916'
EXCLUSIONS={'whole_root_inventory.json','validation_report.json','terminal_receipt.json'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def signed(value):
    return {**value,'canonical_digest':canonical(value)}


def verify(value):
    value=dict(value); expected=value.pop('canonical_digest',None)
    return expected==canonical(value)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+chr(10),encoding='utf-8')


def inventory(root,exclude=()):
    return [{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file() and p.relative_to(root).as_posix() not in exclude]


def sections(path,marker):
    return {int(p.split(':',1)[0].split(chr(10),1)[0]):p.split(chr(10),1)[1].strip() for p in path.read_text(encoding='utf-8').split(marker)[1:]}


def validate():
    errors=[]
    def check(ok,msg):
        if not ok: errors.append(msg)
    def checked(path):
        value=read(path); check(verify(value),'canonical digest '+str(path)); return value
    manifest=checked(ROOT/'render_manifest.json'); figures=manifest['figures']
    check([f['number'] for f in figures]==list(range(1,15)),'figure roster')
    check(manifest['figure_count']==14 and manifest['artifact_count']==42,'artifact counts')
    check(manifest['redesigned_figures']==[1,2,3] and manifest['inherited_R3_figures']==[7],'revision scope')
    outputs=[]
    old=checked(R2/'render_manifest.json'); r3=checked(R3/'render_manifest.json')
    for f in figures:
        n=f['number']; check(len(f['outputs'])==3 and {o['format'] for o in f['outputs']}=={'PNG','PDF','SVG'},'formats '+str(n))
        expected='R4' if n<4 else 'R3' if n==7 else 'R2'
        check(f['revision']==expected and f['redesigned']==(n<4),'revision '+str(n))
        for o in f['outputs']:
            p=REPO/o['path']; outputs.append(p)
            check(p.is_relative_to(ROOT) and not p.is_symlink(),'local regular output '+o['path'])
            if not p.is_file(): errors.append('missing '+str(p)); continue
            check(sha(p)==o['sha256'] and p.stat().st_size==o['bytes'],'output hash '+str(p))
            if o['format']=='PNG':
                with Image.open(p) as im:
                    im.load(); check(im.width>=1200 and im.height>=700,'PNG size '+str(n))
                    check(all(abs(d-400)<1 for d in im.info.get('dpi',(0,0))),'400 dpi '+str(n))
            elif o['format']=='PDF': check(p.read_bytes().startswith(b'%PDF'),'PDF signature '+str(n))
            else: check(b'<svg' in p.read_bytes()[:4096],'SVG signature '+str(n))
            origin_outputs=r3['outputs'] if n==7 else old['figures'][n-1]['outputs']
            source=next(r for r in origin_outputs if r['format']==o['format'])
            check((o['sha256']==source['sha256'])==(n>=4),'copy/redesign parity '+str(n)+' '+o['format'])
    actual=[p for folder in ('figures_png','figures_pdf','figures_svg') for p in (ROOT/folder).iterdir()]
    check(set(actual)==set(outputs) and len(actual)==42,'exact output roster')
    sources=manifest['source_files']; check(canonical(sources)==manifest['source_files_digest'],'source inventory digest')
    for r in sources:
        p=REPO/r['path']; check(p.is_file() and sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],'source drift '+r['path'])
    inputs=checked(ROOT/'input_inventory.json'); check(inputs['source_files']==sources,'input inventory parity')
    predecessor=checked(ROOT/'predecessor_snapshot.json')
    current=[{**r,'path':root.relative_to(REPO).as_posix()+'/'+r['path']} for root in (R2,R3) for r in inventory(root)]
    check(current==predecessor['files'] and canonical(current)==predecessor['files_digest'],'predecessor root mutation')
    display=checked(ROOT/'numeric_display_data.json'); previous=checked(R2/'numeric_display_data.json')
    check(display['figures']==previous['figures'],'all inherited numeric values')
    check(display['figures']['7']==checked(R3/'numeric_display_data.json')['series'],'R3 numeric parity')
    schematic=display['schematics']
    check(schematic['status']=='ILLUSTRATIVE_ONLY' and not display['new_inference'],'schematic labeling')
    check(schematic['token_scores']==[16,8] and schematic['uniform_cell_scores']==[1,.5],'different patch scores')
    check(all(c==16*u for c,u in zip(schematic['token_scores'],schematic['uniform_cell_scores'])),'uniform expansion conservation')
    check(not schematic['nns_patch_sums_equal_token_scores_assumed'],'independent attribution coordinates')
    check(sorted(v for row in schematic['local_rank_example'] for v in row)==list(range(1,17)),'local rank permutation')
    for name in ('nns_patch_A','nns_patch_B'):
        values=[v for row in schematic[name] for v in row]
        check(len(values)==16 and min(values)<0<max(values) and len(set(values))>1,'signed nonuniform '+name)
    for language in ('zh','en'):
        captions=sections(ROOT/f'captions_{language}.md','## Figure '); previous_captions=sections(R2/f'captions_{language}.md','## Figure ')
        check(set(captions)==set(range(1,15)),'caption coverage '+language)
        for n in range(4,15): check(captions[n]==((R3/f'captions_{language}.md').read_text().strip() if n==7 else previous_captions[n]),'caption inheritance '+language+' '+str(n))
    analysis=sections(ROOT/'figure_analysis.md','## Image #'); check(set(analysis)==set(range(1,15)),'analysis coverage')
    prior_analysis=sections(R2/'figure_analysis.md','## Image #')
    for n in range(4,15): check(analysis[n]==((R3/'figure_analysis.md').read_text().split(chr(10),1)[1].strip() if n==7 else prior_analysis[n]),'analysis inheritance '+str(n))
    for path in [*ROOT.glob('*.md'),ROOT/'gallery.html']:
        content=path.read_text(); links=re.findall(r'(?:href|src)="([^"]+)"',content) if path.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',content)
        for target in links:
            if target.startswith('#'): check('id="'+target[1:]+'"' in content,'gallery anchor '+target)
            elif '://' not in target: check((path.parent/target.split('#')[0]).exists(),'broken link '+target)
    layout=checked(ROOT/'layout_audit.json')
    check(layout['status']=='PASS' and [r['number'] for r in layout['records']]==[1,2,3] and all(not r['text_outside_canvas'] for r in layout['records']),'canvas bounds')
    audit=checked(ROOT/'cpu_access_audit.json')
    check(audit['CUDA_VISIBLE_DEVICES']=='' and not audit['torch_imported'] and not audit['timm_imported'] and not audit['new_inference'],'CPU boundary')
    qcpath=ROOT/'manual_visual_qc.json'
    if not qcpath.exists(): errors.append('actual visual review pending')
    else:
        qc=checked(qcpath); check(qc['status']=='PASS' and [r['number'] for r in qc['records']]==list(range(1,15)),'visual QC coverage')
        for r,f in zip(qc['records'],figures):
            check(r['status']=='PASS' and r['png_sha256']==next(o['sha256'] for o in f['outputs'] if o['format']=='PNG'),'visual review binding '+str(f['number']))
            check(r['review_mode']==('direct_visual_review' if f['number']<4 else 'inherited_sealed_review_and_byte_parity'),'visual review mode')
    if (ROOT/'terminal_receipt.json').exists():
        terminal=checked(ROOT/'terminal_receipt.json'); inv=checked(ROOT/'whole_root_inventory.json')
        check(terminal['status']=='PASS_PRESENTATION_RENDER','terminal status')
        check(inv['files']==inventory(ROOT,EXCLUSIONS) and canonical(inv['files'])==inv['files_digest'],'whole root inventory')
        for name,digest in terminal['bindings'].items(): check(sha(ROOT/name)==digest,'terminal binding '+name)
        check(terminal['validator_sha256']==sha(Path(__file__)),'validator identity')
    return errors


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seal',action='store_true'); args=parser.parse_args()
    sealed=(ROOT/'terminal_receipt.json').exists()
    if args.seal and sealed: raise RuntimeError('R4 already sealed')
    errors=validate()
    report=signed({'status':'FAIL' if errors else 'PASS','errors':errors,'revision':'R4','figure_count':14,'artifact_count':42,'redesigned_figures':[1,2,3],'inherited_figures_verified':11,'numeric_data_unchanged':not errors,'cpu_only':True})
    if not sealed: write(ROOT/'validation_report.json',report)
    if args.seal and not errors:
        files=inventory(ROOT,EXCLUSIONS); write(ROOT/'whole_root_inventory.json',signed({'files':files,'files_digest':canonical(files),'exclusions':sorted(EXCLUSIONS)}))
        write(ROOT/'terminal_receipt.json',signed({'status':'PASS_PRESENTATION_RENDER','revision':'R4','figure_count':14,'artifact_count':42,'bindings':{name:sha(ROOT/name) for name in ('render_manifest.json','whole_root_inventory.json','manual_visual_qc.json','validation_report.json')},'validator_sha256':sha(Path(__file__)),'new_inference':False}))
    print(json.dumps(report,ensure_ascii=False)); return bool(errors)


if __name__=='__main__': raise SystemExit(main())
