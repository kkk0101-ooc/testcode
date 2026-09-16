#!/usr/bin/env python3
"""Verify Figure 7 R3 exports, source values and actual plotted heights/intervals."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from pathlib import Path
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import render_nns_vit_figure_07_r3 as render
from render_nns_vit_unified_paper_figures_r1 import sha256_file, canonical_digest, signed, write_json

ROOT=render.ROOT


def read(path):
    return json.loads(path.read_text())


def verify(value):
    copy=dict(value); claimed=copy.pop('canonical_digest',None)
    return canonical_digest(copy)==claimed


def validate():
    errors=[]
    def check(ok,msg):
        if not ok: errors.append(msg)
    manifest=read(ROOT/'render_manifest.json'); check(verify(manifest),'manifest digest')
    check({r['format'] for r in manifest['outputs']}=={'PNG','PDF','SVG'},'formats')
    for r in manifest['outputs']:
        p=render.REPO/r['path']; check(p.is_file() and sha256_file(p)==r['sha256'] and p.stat().st_size==r['bytes'],'output hash '+r['format'])
        if r['format']=='PNG':
            with Image.open(p) as im: im.load(); check(im.width>2000 and im.height>1000,'PNG size')
        elif r['format']=='PDF': check(p.read_bytes().startswith(b'%PDF'),'PDF signature')
        else: check(b'<svg' in p.read_bytes()[:4096],'SVG signature')
    check(canonical_digest(manifest['sources'])==manifest['sources_digest'],'source digest')
    for r in manifest['sources']:
        p=render.REPO/r['path']; check(sha256_file(p)==r['sha256'],'source drift '+r['path'])
    payload=read(render.SOURCE)['plot_payload']; values=read(ROOT/'numeric_display_data.json')
    check(verify(values),'numeric digest')
    prior=read(render.PREDECESSOR/'numeric_display_data.json')['figures']['7']
    check(values['series']==prior,'R2 numeric parity')
    fig,expected=render.render(payload)
    check(expected==values['series'],'formal source parity')
    # Inspect actual Matplotlib artists, including repeated inset marks.
    for axis,indices in zip(fig.axes,([0,1,2,3],[1,2,3])):
        for bars,series in zip([c for c in axis.containers if hasattr(c,'patches')],expected):
            check(np.allclose([p.get_height() for p in bars.patches],[series['y'][i] for i in indices],rtol=0,atol=1e-12),'plotted bar height')
        for container,series in zip([c for c in axis.containers if hasattr(c,'has_yerr')],expected):
            segments=container.lines[2][0].get_segments()
            check(np.allclose([[s[0,1],s[1,1]] for s in segments],[[series['lower'][i],series['upper'][i]] for i in indices],rtol=0,atol=1e-12),'plotted CI endpoints')
    check(list(fig.axes[0].get_ylim())==values['main_ylim'] and list(fig.axes[1].get_ylim())==values['inset_ylim'],'axis ranges')
    check([s['source_counts'][3]['valid'] for s in expected]==[498,417,81],'typed failure counts')
    plt.close(fig)
    qc_path=ROOT/'manual_visual_qc.json'
    if not qc_path.exists(): errors.append('visual review pending')
    else:
        qc=read(qc_path); check(verify(qc) and qc['status']=='PASS' and qc['png_sha256']==next(r['sha256'] for r in manifest['outputs'] if r['format']=='PNG'),'visual review binding')
    check(manifest['cpu_only'] and not manifest['torch_imported'] and not manifest['timm_imported'] and not manifest['new_inference'],'CPU boundary')
    for language in ('zh','en'): check((ROOT/f'captions_{language}.md').is_file(),'caption '+language)
    terminal=ROOT/'terminal_receipt.json'
    if terminal.exists():
        term=read(terminal); check(verify(term),'terminal digest')
        for r in term['files']: check(sha256_file(ROOT/r['path'])==r['sha256'],'sealed file '+r['path'])
    return errors


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seal',action='store_true'); args=parser.parse_args()
    exists=(ROOT/'terminal_receipt.json').exists()
    if args.seal and exists: raise RuntimeError('R3 already sealed')
    errors=validate()
    report=signed({'status':'FAIL' if errors else 'PASS','errors':errors,'figure':7,'output_count':3,'source_and_R2_numeric_parity':not errors,'plotted_medians_checked':21,'plotted_intervals_checked':21,'cpu_only':True})
    if not exists: write_json(ROOT/'validation_report.json',report)
    if args.seal and not errors:
        files=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha256_file(p)} for p in sorted(ROOT.rglob('*')) if p.is_file()]
        write_json(ROOT/'terminal_receipt.json',signed({'status':'PASS_PRESENTATION_RENDER','figure':7,'revision':'R3','files':files,'new_inference':False}))
    print(json.dumps(report)); return bool(errors)


if __name__=='__main__': raise SystemExit(main())
