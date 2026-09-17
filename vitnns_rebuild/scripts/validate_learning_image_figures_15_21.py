#!/usr/bin/env python3
"""Validate learning-image figures and raw component reconstruction."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import validate_nns_vit_unified_paper_figures_r4 as common

REPO=common.REPO
ROOT=REPO/'reports/nns_vit_learning_image_figures_15_21_20260917'
EXCLUDE={'validation_report.json','terminal_receipt.json','whole_root_inventory.json'}


def digest(array):
    value=np.ascontiguousarray(array);sha=hashlib.sha256()
    for part in [str(value.dtype).encode(),b'\0',str(tuple(value.shape)).encode(),b'\0',value.tobytes()]:sha.update(part)
    return sha.hexdigest()


def validate(review=True):
    errors=[]
    def check(condition,message):
        if not condition:errors.append(message)
    def read(path):
        value=common.read(path);check(common.verify(value),'digest '+str(path));return value
    manifest=read(ROOT/'render_manifest.json');numeric=read(ROOT/'numeric_display_data.json')
    check(manifest['figure_numbers']==list(range(15,22)) and not manifest['base_figures_1_14_modified'],'scope')
    outputs=[]
    for figure in manifest['figures']:
        check(len(figure['outputs'])==3 and {row['format'] for row in figure['outputs']}=={'PNG','PDF','SVG'},'formats')
        for output in figure['outputs']:
            path=REPO/output['path'];outputs.append(path)
            check(path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink(),'local output')
            check(common.sha(path)==output['sha256'] and path.stat().st_size==output['bytes'],'output hash')
            if output['format']=='PNG':
                with Image.open(path) as image:
                    image.load();check(image.width>1200 and image.height>700,'PNG dimensions')
                    check(all(abs(value-400)<1 for value in image.info.get('dpi',(0,0))),'DPI')
            elif output['format']=='PDF':check(path.read_bytes().startswith(b'%PDF'),'PDF')
            else:check(b'<svg' in path.read_bytes()[:4096],'SVG')
    actual=[path for folder in ['figures_png','figures_pdf','figures_svg'] for path in (ROOT/folder).iterdir()]
    check(len(actual)==21 and set(outputs)==set(actual),'export roster')
    source_files=manifest['source_files'];check(common.canonical(source_files)==manifest['source_files_digest'],'source digest')
    for record in source_files:
        path=REPO/record['path'];check(path.is_file() and common.sha(path)==record['sha256'],'source drift '+record['path'])
    before=read(ROOT/'predecessor_snapshot.json')
    check(common.inventory(REPO/before['root'])==before['files'] and common.canonical(before['files'])==before['files_digest'],'base root changed')
    toy=numeric['toy'];delta=np.array(toy['input'])-np.array(toy['baseline']);seeds=np.array(toy['seeds'])
    check(np.array_equal(delta,np.array(toy['delta'])) and np.array_equal(seeds.sum(0),delta),'seed sum')
    check(all(np.count_nonzero(seed)==1 for seed in seeds),'one input coordinate per seed')
    jacobian_a=np.array([[1.,1.],[0.,0.]]);jacobian_b=np.array([[0.,0.],[2.,-1.]])
    check(np.array_equal(np.array(toy['unit_a']),delta*jacobian_a),'unit a map')
    check(np.array_equal(np.array(toy['unit_b']),delta*jacobian_b),'unit b map')
    output=delta*(2*jacobian_a+.5*jacobian_b)
    check(np.array_equal(output,toy['output']) and output.sum()==toy['output_scalar']==7,'analytic generation')
    source=numeric['source'];arrays={}
    terminal=read(REPO/source['source_terminal']['path'])['payload']
    check(source['target_rule']=='clean predicted-class raw logit' and source['class_id']==terminal['target_identity']['class_id'],'same class target')
    for record in source['array_bindings']:
        path=REPO/record['path'];value=np.load(path,allow_pickle=False)
        check(common.sha(path)==record['sha256'] and digest(value)==record['canonical_digest'],'source array identity')
        check(list(value.shape)==record['shape'] and value.dtype.name==record['dtype'] and np.isfinite(value).all(),'array shape and values')
        arrays[path.stem]=value
    total=arrays['input_cell_map'].astype(float);observed=[]
    for block in range(12):
        prefix=f'b{block:02d}_';head_sum=arrays[prefix+'first_heads_cell'].astype(float).sum(0)
        attention=arrays[prefix+'first_attention_cell'].astype(float)
        identity=arrays[prefix+'first_identity_cell'].astype(float)
        residual=arrays[prefix+'second_residual_cell'].astype(float);mlp=arrays[prefix+'second_mlp_cell'].astype(float)
        observed.append({'block':block,'head_sum_to_attention_max_abs':float(abs(head_sum-attention).max()),'first_cut_to_input_max_abs':float(abs(identity+attention-total).max()),'second_cut_to_input_max_abs':float(abs(residual+mlp-total).max())})
    check(observed==numeric['closure']['records'],'raw reconstruction recomputation')
    check(all(value<=numeric['closure']['atol_plus_rtol_threshold'] for record in observed for key,value in record.items() if key!='block'),'closure tolerance')
    archived=read(ROOT/'source_arrays_manifest.json')['files'];check(archived==common.inventory(ROOT/'source_arrays'),'forward arrays inventory')
    logits=np.load(ROOT/'source_arrays/logits.npy',allow_pickle=False);forward=source['cpu_forward']
    check(logits.argmax()==source['class_id'] and abs(float(logits[source['class_id']])-source['recorded_clean_logit'])<=forward['atol'],'forward target parity')
    check(forward['device']=='cpu' and forward['forward_calls']==1 and forward['backward_calls']==0 and forward['new_IG_calls']==0,'CPU-only forward scope')
    for block in range(12):
        value=np.load(ROOT/'source_arrays'/f'b{block:02d}_attention_output.npy',allow_pickle=False)
        check(value.shape==(197,768) and np.isfinite(value).all(),'token response shape')
    archive=source['tar_member']
    with (REPO/archive['archive']).open('rb') as stream:stream.seek(archive['offset_data']);raw=stream.read(archive['bytes'])
    check(hashlib.sha256(raw).hexdigest()==archive['sha256'],'JPEG member identity')
    all_heads=list((ROOT/'all_heads').glob('*.png'));check(len(all_heads)==12,'complete head gallery')
    check(read(ROOT/'all_heads_manifest.json')['files']==common.inventory(ROOT/'all_heads'),'all-head gallery identity')
    panel_records={record['panel']:record for record in numeric['display']}
    for block in [0,2,4,6,8,11]:
        response=np.load(ROOT/'source_arrays'/f'b{block:02d}_attention_output.npy',allow_pickle=False)[1:]
        rms=np.sqrt(np.mean(response.astype(float)**2,axis=1)).reshape(14,14)
        check(digest(rms)==panel_records[f'19_B{block}_token_RMS']['raw_sha256'],'response RMS definition')
        check(float(np.percentile(rms,99))==panel_records[f'19_B{block}_token_RMS']['upper'],'response display scaling')
        for suffix,sign in [('positive',1),('negative',-1)]:
            value=sign*arrays[f'b{block:02d}_first_attention_cell']
            check(digest(value)==panel_records[f'19_B{block}_learning_{suffix}']['raw_sha256'],'learning image display source')
    for path in [*ROOT.glob('*.md')]:
        for remainder in path.read_text().split('](')[1:]:
            target=remainder.split(')',1)[0].strip('<>')
            if '://' not in target and not target.startswith('#'):check((path.parent/target.split('#')[0]).exists(),'broken link '+target)
    for language in ['zh','en']:check(set(common.sections(ROOT/f'captions_{language}.md','## Figure '))==set(range(15,22)),'caption coverage')
    layout=read(ROOT/'layout_audit.json');check([row['number'] for row in layout['records']]==list(range(15,22)) and all(not row['text_outside_canvas'] for row in layout['records']),'layout')
    runtime=read(ROOT/'runtime.json');check(runtime['CUDA_VISIBLE_DEVICES']=='' and runtime['device']=='cpu' and runtime['new_attribution_calls']==0,'runtime')
    journal=read(ROOT/'forward_capture_journal.json')
    check(journal['total_executed_forward_calls']==sum(item['forward_calls'] for item in journal['attempts']),'forward execution journal')
    check(journal['retained_forward_capture_count']==1 and all(item['device']=='cpu' and item['new_IG_calls']==0 for item in journal['attempts']),'forward journal scope')
    if review:
        if not (ROOT/'manual_visual_qc.json').exists():errors.append('visual review pending')
        else:
            qc=read(ROOT/'manual_visual_qc.json');check(qc['status']=='PASS' and [row['number'] for row in qc['records']]==list(range(15,22)),'review roster')
            for row,figure in zip(qc['records'],manifest['figures']):check(row['png_sha256']==next(output['sha256'] for output in figure['outputs'] if output['format']=='PNG'),'review binding')
    if (ROOT/'terminal_receipt.json').exists():
        terminal=read(ROOT/'terminal_receipt.json');inventory=read(ROOT/'whole_root_inventory.json')
        check(inventory['files']==common.inventory(ROOT,EXCLUDE),'sealed inventory')
        check(common.canonical(inventory['files'])==inventory['files_digest'],'sealed digest')
        check(terminal['validator_sha256']==common.sha(Path(__file__)),'validator identity')
        for name,sha in terminal['bindings'].items():check(common.sha(ROOT/name)==sha,'terminal binding')
    return errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    sealed=(ROOT/'terminal_receipt.json').exists()
    if args.seal and (args.draft or sealed):parser.error('Only reviewed, unsealed roots can be sealed')
    errors=validate(not args.draft)
    journal=common.read(ROOT/'forward_capture_journal.json')
    report=common.signed({'status':'FAIL' if errors else 'PASS_DRAFT' if args.draft else 'PASS','errors':errors,'figures':list(range(15,22)),'exports':21,'all_head_pages':12,'new_attribution_computation':False,'retained_CPU_forward_captures':1,'total_CPU_forward_calls_during_development':journal['total_executed_forward_calls']})
    if not sealed:common.write(ROOT/'validation_report.json',report)
    if args.seal and not errors:
        files=common.inventory(ROOT,EXCLUDE);common.write(ROOT/'whole_root_inventory.json',common.signed({'files':files,'files_digest':common.canonical(files)}))
        common.write(ROOT/'terminal_receipt.json',common.signed({'status':'PASS_PRESENTATION_RENDER','validator_sha256':common.sha(Path(__file__)),'bindings':{name:common.sha(ROOT/name) for name in ['render_manifest.json','validation_report.json','manual_visual_qc.json','whole_root_inventory.json']}}))
    print(json.dumps(report,ensure_ascii=False));return bool(errors)


if __name__=='__main__':raise SystemExit(main())
