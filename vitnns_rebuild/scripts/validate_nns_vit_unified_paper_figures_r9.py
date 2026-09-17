#!/usr/bin/env python3
"""Validate figure identity, real response maps, predecessor parity and prose links."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
import validate_nns_vit_unified_paper_figures_r4 as common

REPO=common.REPO
ROOT=REPO/'reports/nns_vit_unified_paper_figures_r9_20260917'
EXCLUSIONS={'terminal_receipt.json','validation_report.json','whole_root_inventory.json'}


def validate(draft=False):
    errors=[]
    def check(ok,message):
        if not ok:errors.append(message)
    def signed(path):
        value=common.read(path);check(common.verify(value),'digest '+str(path));return value
    m=signed(ROOT/'render_manifest.json')
    check(m['figure_count']==31 and m['artifact_count']==93,'counts')
    check([f['number'] for f in m['figures']]==list(range(1,32)),'roster')
    expected=[]
    for f in m['figures']:
        check({o['format'] for o in f['outputs']}=={'PNG','PDF','SVG'},'formats')
        for o in f['outputs']:
            p=REPO/o['path'];expected.append(p)
            check(p.is_relative_to(ROOT) and not p.is_symlink(),'local export')
            check(common.sha(p)==o['sha256'] and p.stat().st_size==o['bytes'],'output hash '+str(p))
            if 'inherited_from' in o:check(common.sha(REPO/o['inherited_from'])==o['sha256'],'inherited parity')
            if o['format']=='PNG':
                with Image.open(p) as im:
                    im.load();check(im.width>=1200 and im.height>=700,'image size '+str(p))
            elif o['format']=='PDF':check(p.read_bytes()[:4]==b'%PDF','PDF header')
            else:check(b'<svg' in p.read_bytes()[:4096],'SVG header')
    actual=[p for folder in ['figures_png','figures_pdf','figures_svg'] for p in (ROOT/folder).iterdir()]
    check(set(actual)==set(expected),'export roster')
    check(common.canonical(m['source_files'])==m['source_files_digest'],'source inventory digest')
    for s in m['source_files']:
        p=REPO/s['path'];check(p.is_file() and common.sha(p)==s['sha256'] and p.stat().st_size==s['bytes'],'source drift '+s['path'])
    for root,files in signed(ROOT/'predecessor_snapshot.json')['roots'].items():
        check(common.inventory(REPO/root)==files,'predecessor changed '+root)
    arrays=signed(ROOT/'source_arrays_manifest.json')
    check(arrays['files']==common.inventory(ROOT/'source_arrays'),'source array identities')
    r=signed(ROOT/'module_compute_receipt.json')
    check(r['channels']==[0,255,511,767] and r['block']==0,'fixed selection')
    check(r['runtime']['device']=='cpu' and r['runtime']['CUDA_VISIBLE_DEVICES']=='','CPU compute')
    check(r['full_model_logit_abs_error']<1e-4,'forward parity')
    for s in r['source_files']:check(common.sha(REPO/s['path'])==s['sha256'],'compute source drift')
    data=ROOT/'source_arrays'
    rgb=np.load(data/'module_response_rgb_ig.npy',allow_pickle=False)
    maps=np.load(data/'module_response_maps.npy',allow_pickle=False)
    check(rgb.shape==(3,4,3,224,224) and np.isfinite(rgb).all(),'real RGB map shape')
    check(np.array_equal(maps,rgb.sum(2)),'RGB to spatial map')
    delta=np.array(r['target_deltas']);clean=np.array(r['clean_values']);base=np.array(r['baseline_values'])
    check(np.allclose(delta,clean-base,atol=1e-10,rtol=0),'target deltas')
    check(np.all(np.abs(rgb.sum((2,3,4))-delta)<1e-4+.001*np.abs(delta)),'response completeness')
    check(float(np.max(np.abs(rgb[2]-rgb[0]-rgb[1])))<1e-9,'residual additivity')
    for mi,name in enumerate(['patch_embedding','attention','residual']):
        c=np.load(data/f'{name}_clean_tokens.npy',allow_pickle=False)
        b=np.load(data/f'{name}_baseline_tokens.npy',allow_pickle=False)
        check(c.shape==b.shape==(196,768),'token shape')
        check(np.allclose((c-b)[:,r['channels']].sum(0),delta[mi],atol=1e-10),'token target recompute')
    previous=None
    for a in r['attempts']:
        values=np.load(data/f'module_response_rgb_ig_order_{a["order"]}.npy',allow_pickle=False)
        closure=np.abs(values.sum((2,3,4))-delta)
        check(np.allclose(closure,a['completeness_abs'],rtol=0,atol=1e-12),'reported closure')
        if previous is not None:
            change=np.sum(np.abs(values-previous),axis=(2,3,4))/np.maximum(np.sum(np.abs(values),axis=(2,3,4)),1e-12)
            check(np.allclose(change,a['map_L1_relative_change'],rtol=1e-10,atol=1e-15),'reported convergence')
        previous=values
    check(np.array_equal(previous,rgb),'retained converged map')
    check(np.max(r['attempts'][-1]['map_L1_relative_change'])<.005,'convergence threshold')
    for name,marker in [('captions_zh.md','## Figure '),('captions_en.md','## Figure '),('figure_analysis.md','## Image #')]:
        check(set(common.sections(ROOT/name,marker))==set(range(1,32)),'text coverage '+name)
    document_paths=[*ROOT.glob('*.md'),ROOT/'gallery.html',REPO/'docs/论文大纲_NNS_ViT_R9_20260917.md',REPO/'docs/论文写作前置条件与结论证据图表映射_20260916.md']
    for path in document_paths:
        text=path.read_text(encoding='utf-8')
        check(re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',text) is None,'control character '+str(path))
        links=re.findall(r'(?:href|src)="([^"]+)"',text) if path.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',text)
        for target in links:
            target=target.strip('<>')
            if target.startswith('#'):check('id="'+target[1:]+'"' in text,'anchor '+target)
            elif '://' not in target:check((path.parent/target.split('#')[0]).exists(),'link '+str(path)+' '+target)
    evidence=signed(ROOT/'paper_evidence_audit.json')
    check(evidence['status']=='PASS','paper evidence audit')
    check(evidence['outline_sha256']==common.sha(REPO/'docs/论文大纲_NNS_ViT_R9_20260917.md'),'outline binding')
    for source in evidence['source_records']:
        p=REPO/source['path']
        check(common.sha(p)==source['sha256'] and common.verify(common.read(p)),'paper source '+source['path'])
    layout=signed(ROOT/'layout_audit.json')
    check(layout['status']=='PASS' and {r['number'] for r in layout['records']}=={2,28,29,30,31},'layout coverage')
    if not draft:
        review=signed(ROOT/'manual_visual_qc.json')
        check(review['status']=='PASS' and len(review['records'])==31,'visual review')
        for row,f in zip(review['records'],m['figures']):
            png=next(o for o in f['outputs'] if o['format']=='PNG')
            check(row['number']==f['number'] and row['png_sha256']==png['sha256'] and row['status']=='PASS','QC binding')
    if (ROOT/'terminal_receipt.json').exists():
        t=signed(ROOT/'terminal_receipt.json');i=signed(ROOT/'whole_root_inventory.json')
        check(t['validator_sha256']==common.sha(Path(__file__)),'validator binding')
        check(i['files']==common.inventory(ROOT,EXCLUSIONS),'sealed inventory')
        for name,digest in t['bindings'].items():check(common.sha(ROOT/name)==digest,'receipt '+name)
    return errors


def main():
    p=argparse.ArgumentParser();p.add_argument('--draft',action='store_true');p.add_argument('--seal',action='store_true');a=p.parse_args()
    sealed=(ROOT/'terminal_receipt.json').exists()
    if a.seal and (sealed or a.draft):p.error('Need unsealed reviewed root')
    errors=validate(a.draft)
    report=common.signed({'status':'FAIL' if errors else 'PASS_DRAFT_CHECKS' if a.draft else 'PASS','errors':errors,'figure_count':31,'artifact_count':93,'new_module_maps':'single-input illustration'})
    if not sealed:common.write(ROOT/'validation_report.json',report)
    if a.seal and not errors:
        files=common.inventory(ROOT,EXCLUSIONS)
        common.write(ROOT/'whole_root_inventory.json',common.signed({'files':files,'files_digest':common.canonical(files)}))
        common.write(ROOT/'terminal_receipt.json',common.signed({'status':'PASS_PRESENTATION_RENDER','revision':'R9','validator_sha256':common.sha(Path(__file__)),'bindings':{name:common.sha(ROOT/name) for name in ['render_manifest.json','validation_report.json','whole_root_inventory.json','manual_visual_qc.json','module_compute_receipt.json']}}))
    print(json.dumps(report,ensure_ascii=False));return bool(errors)


if __name__=='__main__':raise SystemExit(main())
