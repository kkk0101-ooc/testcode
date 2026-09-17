#!/usr/bin/env python3
"""Verify two-row response-map diagrams by finite differences and completeness."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from PIL import Image
import validate_nns_vit_unified_paper_figures_r4 as common

REPO=common.REPO
ROOT=REPO/'reports/nns_vit_forward_learning_image_comparison_20260917'
EXCLUSIONS={'terminal_receipt.json','whole_root_inventory.json','validation_report.json'}


def independent_forward(inputs):
    hidden=np.array([(inputs[0]+inputs[1])/2,(inputs[2]+inputs[3])/2])
    scores=hidden[:,None]*hidden[None,:]
    probabilities=np.exp(scores-scores.max(1,keepdims=True));probabilities/=probabilities.sum(1,keepdims=True)
    outputs=probabilities@hidden
    return {'x':inputs,'h':hidden,'q':hidden,'k':hidden,'v':hidden,'scores':scores,'a':probabilities,'o':outputs,'z':hidden+outputs}


def validate(require_review=True):
    errors=[]
    def check(condition,message):
        if not condition:errors.append(message)
    def read(path):
        data=common.read(path);check(common.verify(data),'digest: '+str(path));return data
    data=read(ROOT/'numeric_data.json');manifest=read(ROOT/'render_manifest.json')
    inputs=np.array(data['input']);baseline=np.array(data['reference']);maps={key:np.array(value) for key,value in data['learning_images'].items()}
    expected=independent_forward(inputs);start=independent_forward(baseline)
    for name,value in expected.items():
        check(np.allclose(data['values'][name],value,rtol=0,atol=1e-12),'forward '+name)
        check(np.allclose(data['baseline_values'][name],start[name],rtol=0,atol=1e-12),'reference '+name)
        check(maps[name].shape==value.shape+(2,2),'one input-image map per scalar: '+name)
        check(np.allclose(maps[name].sum((-2,-1)),value-start[name],rtol=0,atol=1e-10),'paired scalar/map completeness: '+name)
    expected_seeds=np.diag(inputs).reshape(4,2,2)
    check(np.allclose(maps['x'],expected_seeds,rtol=0,atol=1e-12),'seed input maps')
    expected_hidden=np.array([[[.5,.5],[0,0]],[[0,0],[1.5,1.5]]])
    check(np.allclose(maps['h'],expected_hidden,rtol=0,atol=1e-12),'patch response maps')
    check(np.allclose(maps['z'],maps['h']+maps['o'],rtol=0,atol=1e-11),'residual elementwise map addition')
    # Recompute each nonlinear unit map from numerical input derivatives and adaptive integration.
    for name in ['scores','a','o','z']:
        for unit in np.ndindex(expected[name].shape):
            for input_index in range(4):
                perturbation=np.zeros(4);perturbation[input_index]=1e-5
                def derivative(fraction):
                    point=baseline+fraction*(inputs-baseline)
                    return (independent_forward(point+perturbation)[name][unit]-independent_forward(point-perturbation)[name][unit])/2e-5
                value,estimated_error=quad(derivative,0,1,epsabs=1e-8,epsrel=1e-8)
                reference_value=(inputs-baseline)[input_index]*value
                observed=maps[name][unit].reshape(-1)[input_index]
                check(abs(reference_value-observed)<2e-7,'independent finite-difference integral '+str((name,unit,input_index)))
    attention_reference=np.array([[[.234536120,.234536120],[1.146260958,1.146260958]],[[.055283274,.055283274],[1.442244102,1.442244102]]])
    check(np.allclose(maps['o'],attention_reference,rtol=0,atol=1e-7),'attention worked example')
    coefficients=np.array(data['attention_learning_image_coefficients'])
    check(np.allclose(np.tensordot(coefficients,maps['h'],axes=(1,0)),maps['o'],rtol=0,atol=1e-12),'attention image input to output coefficients')
    check(not np.allclose(np.tensordot(expected['a'],maps['h'],axes=(1,0)),maps['o']),'full attention not frozen endpoint mixing')
    keys=['01_overview','02_patch_embedding','03_attention','04_residual']
    check([figure['key'] for figure in manifest['figures']]==keys,'figure roster')
    paths=[]
    for figure in manifest['figures']:
        check(len(figure['outputs'])==3 and {output['format'] for output in figure['outputs']}=={'PNG','PDF','SVG'},'formats')
        for output in figure['outputs']:
            path=ROOT/output['path'];paths.append(path)
            check(path.is_file() and not path.is_symlink() and path.is_relative_to(ROOT),'local export')
            check(path.stat().st_size==output['bytes'] and common.sha(path)==output['sha256'],'export identity')
            if output['format']=='PNG':
                with Image.open(path) as image:
                    image.load();check(image.width>2000 and image.height>1000,'PNG dimensions')
                    check(all(abs(dpi-320)<1 for dpi in image.info.get('dpi',(0,0))),'PNG dpi')
            elif output['format']=='PDF':check(path.read_bytes().startswith(b'%PDF'),'PDF signature')
            else:check(b'<svg' in path.read_bytes()[:4096],'SVG signature')
    actual=[path for extension in ['png','pdf','svg'] for path in (ROOT/f'figures_{extension}').iterdir()]
    check(len(actual)==12 and set(actual)==set(paths),'exact exports')
    sources=manifest['source_files'];check(common.canonical(sources)==manifest['source_files_digest'],'source digest')
    for source in sources:
        path=REPO/source['path'];check(common.sha(path)==source['sha256'] and path.stat().st_size==source['bytes'],'source drift '+source['path'])
    predecessor=read(ROOT/'predecessor_snapshot.json')
    check(predecessor['files']==common.inventory(REPO/predecessor['root']) and common.canonical(predecessor['files'])==predecessor['files_digest'],'predecessor unchanged')
    layout=read(ROOT/'layout_audit.json')
    check([row['key'] for row in layout['records']]==keys and all(not row['text_outside_canvas'] for row in layout['records']),'layout')
    for path in [*ROOT.glob('*.md'),REPO/'docs/ViT正常前向与NNS学习图像逐步对照_20260917.md']:
        content=path.read_text()
        for section in content.split('](')[1:]:
            target=section.split(')',1)[0].strip('<>')
            if '://' not in target:check((path.parent/target.split('#')[0]).exists(),'link '+target)
        check(sum(line.strip()=='$$' for line in content.splitlines())%2==0,'math blocks')
    runtime=read(ROOT/'runtime.json')
    check(runtime['CUDA_VISIBLE_DEVICES']=='' and not runtime['torch_imported'] and runtime['new_trained_model_forwards']==runtime['new_trained_model_attributions']==0,'CPU-only analytic scope')
    if require_review:
        if not (ROOT/'manual_visual_qc.json').exists():errors.append('visual review pending')
        else:
            review=read(ROOT/'manual_visual_qc.json');check(review['status']=='PASS' and [row['key'] for row in review['records']]==keys,'review roster')
            for row,figure in zip(review['records'],manifest['figures']):check(row['status']=='PASS' and row['png_sha256']==next(output['sha256'] for output in figure['outputs'] if output['format']=='PNG'),'review binding')
    if (ROOT/'terminal_receipt.json').exists():
        terminal=read(ROOT/'terminal_receipt.json');inventory=read(ROOT/'whole_root_inventory.json')
        check(inventory['files']==common.inventory(ROOT,EXCLUSIONS) and common.canonical(inventory['files'])==inventory['files_digest'],'sealed inventory')
        check(terminal['validator_sha256']==common.sha(Path(__file__)),'validator binding')
        for name,digest_value in terminal['bindings'].items():check(common.sha(ROOT/name)==digest_value,'terminal binding')
    return errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    sealed=(ROOT/'terminal_receipt.json').exists()
    if args.seal and (sealed or args.draft):parser.error('Only reviewed, unsealed roots can be sealed')
    errors=validate(not args.draft)
    report=common.signed({'status':'FAIL' if errors else 'PASS_DRAFT' if args.draft else 'PASS','errors':errors,'figure_count':4,'export_count':12,'paired_forward_response_maps':True,'new_trained_model_compute':False})
    if not sealed:common.write(ROOT/'validation_report.json',report)
    if args.seal and not errors:
        files=common.inventory(ROOT,EXCLUSIONS);common.write(ROOT/'whole_root_inventory.json',common.signed({'files':files,'files_digest':common.canonical(files)}))
        common.write(ROOT/'terminal_receipt.json',common.signed({'status':'PASS_PRESENTATION_RENDER','validator_sha256':common.sha(Path(__file__)),'bindings':{name:common.sha(ROOT/name) for name in ['render_manifest.json','validation_report.json','manual_visual_qc.json','whole_root_inventory.json']}}))
    print(json.dumps(report,ensure_ascii=False));return bool(errors)


if __name__=='__main__':raise SystemExit(main())
