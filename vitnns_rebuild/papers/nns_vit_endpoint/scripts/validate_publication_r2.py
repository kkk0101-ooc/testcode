#!/usr/bin/env python3
"""Validate the portable paper, figures, real comparison data, and display calculations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast
import hashlib
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET
import numpy as np

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'papers/nns_vit_endpoint/publication_r2_20260920'
checks=[]
def check(name,value,detail=''):checks.append(dict(name=name,passed=bool(value),detail=detail))
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    f=load(ROOT/'figure_manifest.json');check('22_figures',[r['number'] for r in f['figures']]==list(range(1,23)))
    for r in f['figures']:
        for ext,record in r['files'].items():
            p=ROOT/record['path'];check('figure_hash:'+str(r['number'])+':'+ext,p.is_file() and sha(p)==record['sha256'])
            if ext=='png':check('png_size:'+str(r['number']),struct.unpack('>II',p.read_bytes()[16:24])[0]>=1500)
            if ext=='svg':ET.parse(p)
    for path,digest in load(ROOT/'evidence/figure_sources.json').items():check('source:'+path,sha(REPO/path)==digest)
    supplement=load(ROOT/'evidence/supplement_manifest.json');check('92_supplement_pages',len(supplement)==92)
    for r in supplement:check('supplement_hash:'+str(r['number']),sha(ROOT/r['path'])==r['sha256'])
    texts={lang:(ROOT/lang/'manuscript.md').read_text() for lang in ('zh','en')}
    for lang,text in texts.items():
        check('figure_order:'+lang,re.findall(r'!\[Figure (\d+)\]',text)==[str(i) for i in range(1,23)])
        check('7_tables:'+lang,re.findall(r'\*\*Table (\d+)\.',text)==[str(i) for i in range(1,8)])
        check('no_report_states:'+lang,not re.search(r'RQ[1-5]|NOT_SUPPORTED|PASS_|reports/|registry-complete|C4[ab]',text))
        check('balanced_math:'+lang,text.count('$$')%2==0)
        check('no_placeholders:'+lang,'{{' not in text and '\x00' not in text)
    check('same_display_equations',re.findall(r'\$\$(.*?)\$\$',texts['zh'],re.S)==re.findall(r'\$\$(.*?)\$\$',texts['en'],re.S))
    check('same_section_numbers',re.findall(r'^#{2,3} (\d+(?:\.\d+)?)',texts['zh'],re.M)==re.findall(r'^#{2,3} (\d+(?:\.\d+)?)',texts['en'],re.M))
    check('same_table_data',[l for l in texts['zh'].splitlines() if l.startswith('|')]==[l for l in texts['en'].splitlines() if l.startswith('|')])
    paper=ROOT.parent
    active=load(paper/'figure_manifest.json')
    check('active_binding',active['figure_count']==22 and active['source_revision']=='PUBLICATION-R2')
    for source,bound in zip(f['figures'],active['figures']):
        check('active_figure_number:'+str(source['number']),source['number']==bound['figure'])
        for extension in ('png','pdf','svg'):
            check('active_figure_hash:'+str(source['number'])+':'+extension,source['files'][extension]['sha256']==bound[extension]['sha256'])
    for lang,text in texts.items():
        expected=text.replace('](../figures/','](../publication_r2_20260920/figures/').replace('](../references.md)','](../publication_r2_20260920/references.md)').replace('](supplement.md)',f'](../publication_r2_20260920/{lang}/supplement.md)')
        check('current_entry_mirror:'+lang,(paper/lang/'full_draft.md').read_text()==expected)
    for p in [ROOT/lang/name for lang in ('zh','en') for name in ('manuscript.md','supplement.md')]:
        text=p.read_text()
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http','#')):continue
            q=(p.parent/target.split('#')[0]).resolve()
            check('portable_link:'+p.parent.name+':'+target,q.is_file() and ROOT.resolve() in q.parents)
    receipt=load(ROOT/'evidence/compute_receipt.json');check('comparison_tasks',receipt['task_count']>=6 and len(receipt['figure20_orders'])==3)
    selection=load(ROOT/'evidence/figure20_selection.json')
    check('figure20_presealed',selection['status']=='SEALED_BEFORE_COMPARISON_METHOD_COMPUTE' and not selection['comparison_methods_read'])
    check('figure20_exclusions',selection['excluded_by_user']==[4,14,15] and not set(selection['selected_orders'])&{4,14,15})
    check('figure20_binding',receipt['figure20_orders']==selection['selected_orders']==[1,13,7])
    check('figure20_ranking',selection['selected_orders']==[r['order'] for r in selection['ranking'][:3]])
    catalog={}
    for row in receipt['tasks']:
        p=ROOT/'assets'/f"{row['task_id']}.npz";check('computed_hash:'+row['task_id'],sha(p)==row['array_sha256'])
        check('native_parity:'+row['task_id'],row['native_explicit_relative_l2']<1e-7)
        d=np.load(p)
        for k in d.files:
            a=d[k];check('finite:'+row['task_id']+':'+k,np.isfinite(a).all());catalog[hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()]=a
        for k in ('class_endpoint','class_gradient','gradcam','raw_attention','rollout','input_rgb'):check('method_present:'+row['task_id']+':'+k,k in d.files)
        if row['task_id'].endswith('__014'):
            for b in range(12):
                check('layer_pair:'+row['model_id']+':'+str(b),d[f'b{b}.energy.endpoint'].shape==(3,224,224) and d[f'b{b}.energy.feature'].shape==(14,14))
    # E3 arrays are verified against source hashes and indexed for display reconstruction.
    for path,digest in load(ROOT/'evidence/figure_sources.json').items():
        if path.endswith('.npy'):
            a=np.load(REPO/path)
            for v in a:catalog[hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()]=v
    reconstructed=0
    for row in load(ROOT/'evidence/display_receipt.json')['panels']:
        key=row['raw_sha256']
        if key not in catalog:continue # Explicit concept support and derived patch-max displays are separate.
        a=catalog[key];kind=row['kind']
        if kind=='signed':
            v=a.astype(float).squeeze();v=v.sum(0) if v.ndim==3 else v;lim=float(np.quantile(np.abs(v),.99)) or float(abs(v).max()) or 1;shown=np.clip(v/lim,-1,1)
        else:
            if kind=='absolute' and a.ndim==3:v=np.abs(a).sum(0)
            else:
                v=np.asarray(a,dtype=float).squeeze();v=np.maximum(v.sum(0) if v.ndim==3 else v,0)
            nz=v[v>0];lim=float(np.quantile(nz,.99)) if nz.size else 1;shown=np.clip(v/lim,0,1)**row['gamma']
        check('display_limit:'+str(reconstructed),lim==row['limit'])
        check('display_pixels:'+str(reconstructed),hashlib.sha256(np.ascontiguousarray(shown).tobytes()).hexdigest()==row['display_sha256'])
        reconstructed+=1
    check('real_panels_recomputed',reconstructed>=160,str(reconstructed))
    toy=load(ROOT/'evidence/concept_numeric_example.json');x=np.array(toy['x']);W=np.array(toy['W']);v=np.array(toy['v'])
    check('concept_numeric_identity',np.array_equal(x*(v@W),toy['endpoint']) and float(v@W@x)==toy['s'])
    methods=load(ROOT/'qc/method_verification.json')
    check('native_method_recomputation',methods['status']=='PASS' and methods['comparison_tasks']==receipt['task_count'] and methods['layer_pair_checks']==4)
    qc=ROOT/'qc/visual_review.json';check('visual_review_exists',qc.is_file())
    if qc.exists():
        q=load(qc);check('visual_review_all',q['reviewed_figures']==list(range(1,23)))
        check('visual_review_current',q['png_hashes']=={str(r['number']):r['files']['png']['sha256'] for r in f['figures']})
    inventory=load(ROOT/'artifact_inventory.json')
    for record in inventory['files']:
        check('inventory_hash:'+record['path'],sha(ROOT/record['path'])==record['sha256'])
    errors=[r for r in checks if not r['passed']]
    report=dict(status='PASS' if not errors else 'FAIL',check_count=len(checks),errors=errors,real_panels_recomputed=reconstructed,checks=checks)
    (ROOT/'qc/validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+chr(10))
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2))
    return bool(errors)

if __name__=='__main__':raise SystemExit(main())
