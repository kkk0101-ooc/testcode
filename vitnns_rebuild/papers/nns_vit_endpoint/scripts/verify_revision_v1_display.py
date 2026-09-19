#!/usr/bin/env python3
"""Independently reconstruct display statistics and scales from original NPZ arrays."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib
import json
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'reports/nns_vit_endpoint_paper_revision_v1_20260920'

def main():
    arrays={};sources=[]
    for name in ('nns_vit_endpoint_all_figures_r2_20260919/source_arrays/sloth_bear_display_arrays.npz','nns_vit_endpoint_m6_r1_20260919/raw_arrays.npz'):
        p=REPO/'reports'/name;data=np.load(p)
        sources.append(dict(path=str(p.relative_to(REPO)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        for key in data.files:
            value=np.asarray(data[key],dtype=float).squeeze()
            if value.shape==(3,224,224):arrays[hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()]=value
    panels=json.loads((ROOT/'display_receipt.json').read_text())['panels'];checks=[]
    for panel in panels:
        raw=arrays[panel['raw_array_sha256']];signed=raw.sum(0);absolute=np.abs(raw).sum(0)
        group={p['raw_array_sha256'] for p in panels if p['group']==panel['group']}
        grouped=[arrays[k] for k in sorted(group)]
        reduced=[np.abs(a.sum(0)) if panel['view']=='signed' else np.abs(a).sum(0) for a in grouped]
        scale=float(np.quantile(np.concatenate([v.ravel() for v in reduced]),.99))
        values=dict(absolute_mass=float(np.abs(raw).sum()),l2_norm=float(np.linalg.norm(raw)),signed_sum=float(raw.sum()),scale=scale)
        errors={key:abs(panel[key]-value) for key,value in values.items()}
        assert all(np.isclose(panel[key],value,rtol=1e-13,atol=1e-14) for key,value in values.items()),(panel,values)
        display=signed if panel['view']=='signed' else absolute
        assert panel['clipped_pixels']==int((np.abs(display)>scale).sum())
        checks.append(dict(group=panel['group'],view=panel['view'],label=panel['label'],maximum_error=max(errors.values())))
    qc=json.loads((ROOT/'manual_visual_qc.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text())
    for f,q in zip(manifest['figures'],qc['outputs']):
        assert f['figure']==q['figure'] and hashlib.sha256((REPO/f['png']['path']).read_bytes()).hexdigest()==q['png_sha256']
    report=dict(status='PASS',panel_count=len(checks),sources=sources,checks=checks,manual_qc_matches_current_images=True)
    (ROOT/'raw_display_validation.json').write_text(json.dumps(report,indent=2)+chr(10))
    print(json.dumps(dict(status='PASS',panel_count=len(checks))))

if __name__=='__main__':main()
