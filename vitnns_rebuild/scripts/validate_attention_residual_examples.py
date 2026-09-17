#!/usr/bin/env python3
"""Independent numerical and source validation of analytic explanatory figures."""
from __future__ import annotations
import argparse
import json
import math
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
from PIL import Image
import validate_nns_vit_unified_paper_figures_r4 as common

REPO=common.REPO
ROOT=REPO/'reports/nns_vit_attention_residual_examples_20260917'
EXCLUDE={'validation_report.json','terminal_receipt.json','whole_root_inventory.json'}


def validate(review=True):
    errors=[]
    def check(condition,message):
        if not condition:errors.append(message)
    def read(path):
        data=common.read(path);check(common.verify(data),'digest '+str(path));return data
    manifest=read(ROOT/'render_manifest.json');numeric=read(ROOT/'numeric_display_data.json')
    example=numeric['examples'];attention=example['attention'];residual=example['residual']
    probability=1/(1+math.exp(-2))
    check(np.allclose(attention['clean_attention'],[1-probability,probability],rtol=0,atol=1e-14),'softmax example')
    check(abs(attention['clean_output']-(1+2*probability))<1e-14,'attention forward')
    full=np.array(attention['full_map']);value=np.array(attention['value_route_map']);weights=np.array(attention['attention_route_map'])
    check(np.allclose(full,value+weights,rtol=0,atol=1e-14),'two-route sum')
    check(abs(full.sum()-attention['clean_output']+attention['baseline_output'])<1e-11,'attention completeness')
    from scipy.integrate import quad
    value_reference=np.array([quad(lambda alpha:1/(1+math.exp(2*alpha*alpha)),0,1,epsabs=1e-12)[0],3*quad(lambda alpha:1/(1+math.exp(-2*alpha*alpha)),0,1,epsabs=1e-12)[0]])
    route_reference=quad(lambda alpha:2*alpha*alpha*math.exp(-2*alpha*alpha)/(1+math.exp(-2*alpha*alpha))**2,0,1,epsabs=1e-12)[0]*np.array([1,3])
    check(np.allclose(value,value_reference,rtol=0,atol=1e-11) and np.allclose(weights,route_reference,rtol=0,atol=1e-11),'independent adaptive quadrature')
    def forward(inputs):
        scores=inputs[0]*inputs;exp=np.exp(scores-scores.max());return float((exp/exp.sum())@inputs)
    gradients=[]
    for coordinate in range(2):
        delta=np.zeros(2);delta[coordinate]=1e-6
        gradients.append((forward(np.array([1.,3.])+delta)-forward(np.array([1.,3.])-delta))/2e-6)
    expected=np.array(attention['clean_value_gradient'])+attention['clean_attention_gradient']
    check(np.allclose(gradients,expected,rtol=0,atol=1e-8),'finite difference attention derivative')
    frozen=np.array([1-probability,3*probability])
    check(np.allclose(attention['frozen_clean_attention_map'],frozen,rtol=0,atol=1e-14),'conditional target formula')
    check(not np.allclose(value,frozen),'frozen target differs from value route of full model')
    check(np.allclose(residual['skip_target_map'],[10,0],rtol=0,atol=1e-11),'analytic skip integral')
    check(np.allclose(residual['branch_target_map'],[20,-5],rtol=0,atol=1e-11),'analytic branch integral')
    check(np.allclose(residual['total_target_map'],[30,-5],rtol=0,atol=1e-11) and residual['final_output']==25,'residual completeness')
    output_paths=[]
    check([item['number'] for item in manifest['figures']]==[22,23,19],'figure roster')
    for figure in manifest['figures']:
        check({row['format'] for row in figure['outputs']}=={'PNG','PDF','SVG'} and len(figure['outputs'])==3,'formats')
        for output in figure['outputs']:
            path=REPO/output['path'];output_paths.append(path)
            check(path.is_relative_to(ROOT) and not path.is_symlink(),'local path')
            check(path.stat().st_size==output['bytes'] and common.sha(path)==output['sha256'],'export hash')
            if output['format']=='PNG':
                with Image.open(path) as image:
                    image.load();check(image.width>1200 and image.height>700,'PNG size')
            elif output['format']=='PDF':check(path.read_bytes().startswith(b'%PDF'),'PDF')
            else:check(b'<svg' in path.read_bytes()[:4096],'SVG')
    actual=[path for folder in ['figures_png','figures_pdf','figures_svg'] for path in (ROOT/folder).iterdir()]
    check(len(actual)==9 and set(actual)==set(output_paths),'exact file roster')
    sources=manifest['source_files'];check(common.canonical(sources)==manifest['source_files_digest'],'sources digest')
    for record in sources:check(common.sha(REPO/record['path'])==record['sha256'],'source identity')
    before=read(ROOT/'predecessor_snapshot.json');check(common.inventory(REPO/before['root'])==before['files'],'old root unchanged')
    check(common.canonical(before['files'])==before['files_digest'],'old inventory digest')
    old_numeric=read(REPO/before['root']/'numeric_display_data.json')
    old_panels={record['panel']:record for record in old_numeric['display']}
    for record in numeric['positive_figure19_panels']:
        block=int(record['panel'].split('_B')[-1])
        name=f'19_B{block}_token_RMS' if 'RMS' in record['panel'] else f'19_B{block}_learning_positive'
        check({key:value for key,value in record.items() if key!='panel'}=={key:value for key,value in old_panels[name].items() if key!='panel'},'positive Figure 19 exact display data')
    check(len(numeric['positive_figure19_panels'])==12,'positive Figure 19 panels')
    for path in ROOT.glob('*.md'):
        for remainder in path.read_text().split('](')[1:]:
            target=remainder.split(')',1)[0].strip('<>')
            if '://' not in target:check((path.parent/target.split('#')[0]).exists(),'link '+target)
    layout=read(ROOT/'layout_audit.json');check(all(not record['text_outside_canvas'] for record in layout['records']),'layout')
    runtime=read(ROOT/'runtime.json');check(runtime['CUDA_VISIBLE_DEVICES']=='' and runtime['new_model_forward_calls']==runtime['new_model_attribution_calls']==0,'CPU no inference')
    if review:
        if not (ROOT/'manual_visual_qc.json').exists():errors.append('visual review pending')
        else:
            qc=read(ROOT/'manual_visual_qc.json');check(qc['status']=='PASS' and [row['number'] for row in qc['records']]==[22,23,19],'review roster')
            for record,figure in zip(qc['records'],manifest['figures']):check(record['png_sha256']==next(row['sha256'] for row in figure['outputs'] if row['format']=='PNG'),'QC identity')
    if (ROOT/'terminal_receipt.json').exists():
        terminal=read(ROOT/'terminal_receipt.json');inventory=read(ROOT/'whole_root_inventory.json')
        check(inventory['files']==common.inventory(ROOT,EXCLUDE) and common.canonical(inventory['files'])==inventory['files_digest'],'terminal inventory')
        check(terminal['validator_sha256']==common.sha(Path(__file__)),'validator binding')
        for name,digest in terminal['bindings'].items():check(common.sha(ROOT/name)==digest,'terminal binding')
    return errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    sealed=(ROOT/'terminal_receipt.json').exists()
    if args.seal and (args.draft or sealed):parser.error('Sealing requires an unsealed reviewed root')
    errors=validate(not args.draft)
    report=common.signed({'status':'FAIL' if errors else 'PASS_DRAFT' if args.draft else 'PASS','errors':errors,'figures':[22,23,19],'exports':9,'independent_quadrature_check':not errors,'new_model_forward_calls':0})
    if not sealed:common.write(ROOT/'validation_report.json',report)
    if args.seal and not errors:
        files=common.inventory(ROOT,EXCLUDE);common.write(ROOT/'whole_root_inventory.json',common.signed({'files':files,'files_digest':common.canonical(files)}))
        common.write(ROOT/'terminal_receipt.json',common.signed({'status':'PASS_PRESENTATION_RENDER','validator_sha256':common.sha(Path(__file__)),'bindings':{name:common.sha(ROOT/name) for name in ['render_manifest.json','validation_report.json','manual_visual_qc.json','whole_root_inventory.json']}}))
    print(json.dumps(report,ensure_ascii=False));return bool(errors)


if __name__=='__main__':raise SystemExit(main())
