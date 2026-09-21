#!/usr/bin/env python3
"""Materialize the compact, auditable v5 evidence package from run roots."""
from pathlib import Path
import hashlib, json, shutil

P=Path(__file__).resolve().parents[1]
R=P.parents[2]
RUN=R/'reports/nns_vit_revision_v5_r1_20260921'
SRC=RUN/'analysis'
OUT=P/'evidence/revision_v5'
OUT.mkdir(parents=True,exist_ok=True)
EXISTING=P/'evidence/revision_v5_existing'
E1R2_RUN=R/'reports/nns_vit_revision_v5_e1_r2_r1_20260921'
E1R2_ANALYSIS=E1R2_RUN/'analysis'
E1R2_MANIFEST=R/'manifests/nns_vit_revision_v5_e1_r2'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

analysis_files=[
 'e1_pairs.csv','e1_class_values.csv','e1_summary.csv','e1_activation_selectivity.csv',
 'e1_metadata_image_values.csv','e1_metadata_class_values.csv','e1_metadata_summary.csv','e1_metadata_receipt.json',
 'e2_image_values.csv','e2_summary.csv',
 'e3_single_summary.csv','e3_topk_summary.csv','e3_paired_summary.csv',
 'e4_models.csv','e4_paired.csv','e4_predictions.csv',
 'e4_similar_total_pairs_vit_b16.csv','e4_similar_total_pairs_deit_s16.csv',
 'e4_similar_total_pairs_vit_b16_in1k.csv','e4_similar_total_pairs_vit_s16_augreg.csv',
 'e5_image_values.csv','e5_profiles.csv','e5_summary.csv',
 'e1_receipt.json','e2_receipt.json','e3_receipt.json','e4_receipt.json','e5_receipt.json',
 'e4_fit_deit_s16_shuffled.npz','e4_fit_deit_s16_structure.npz','e4_fit_deit_s16_total.npz',
 'e4_fit_vit_b16_shuffled.npz','e4_fit_vit_b16_structure.npz','e4_fit_vit_b16_total.npz',
 'e4_fit_vit_b16_in1k_shuffled.npz','e4_fit_vit_b16_in1k_structure.npz','e4_fit_vit_b16_in1k_total.npz',
 'e4_fit_vit_s16_augreg_shuffled.npz','e4_fit_vit_s16_augreg_structure.npz','e4_fit_vit_s16_augreg_total.npz']
protocol_files=['protocol.json','cohorts.json','targets.json','image_bindings.json','analysis_estimands.json']
manifest=[]
for name in analysis_files:
    src=SRC/name;assert src.is_file(),src
    dst=OUT/name;shutil.copy2(src,dst);manifest.append({'path':name,'sha256':sha(dst),'bytes':dst.stat().st_size,'source':str(src.relative_to(R))})
for name in protocol_files:
    src=R/'manifests/nns_vit_revision_v5'/name;assert src.is_file(),src
    dst=OUT/name;shutil.copy2(src,dst);manifest.append({'path':name,'sha256':sha(dst),'bytes':dst.stat().st_size,'source':str(src.relative_to(R))})
for name in ['m1_head_summary.csv','m1_head_paired.csv','amplitude_quantile_summary.csv','amplitude_effect_strata.csv','source_receipt.json']:
    src=EXISTING/name;assert src.is_file(),src
    dst=OUT/name;shutil.copy2(src,dst);manifest.append({'path':name,'sha256':sha(dst),'bytes':dst.stat().st_size,'source':str(src.relative_to(R))})
for src,dest_name in [(P/'notes/revision_v5_plan.md','frozen_revision_v5_plan.md'),(P/'notes/revision_v5_reference_checks.json','reference_checks.json'),(P/'notes/revision_v5_e1_r2_recovery.md','e1_r2_recovery.md')]:
    assert src.is_file(),src
    dst=OUT/dest_name;shutil.copy2(src,dst);manifest.append({'path':dest_name,'sha256':sha(dst),'bytes':dst.stat().st_size,'source':str(src.relative_to(R))})
src=P/'notes/reference_sources/dou_equations_10_11.txt';dst=OUT/'dou_equations_10_11.txt';shutil.copy2(src,dst);manifest.append({'path':dst.name,'sha256':sha(dst),'bytes':dst.stat().st_size,'source':str(src.relative_to(R))})
for source_name,dest_name in [('summary.csv','e1_r2_summary.csv'),('class_values.csv','e1_r2_class_values.csv'),('image_values.csv','e1_r2_image_values.csv'),('activation_selectivity.csv','e1_r2_activation_selectivity.csv'),('receipt.json','e1_r2_receipt.json')]:
    src=E1R2_ANALYSIS/source_name;assert src.is_file(),src
    dst=OUT/dest_name;shutil.copy2(src,dst);manifest.append({'path':dest_name,'sha256':sha(dst),'bytes':dst.stat().st_size,'source':str(src.relative_to(R))})
for source_name,dest_name in [('protocol.json','e1_r2_protocol.json'),('cohorts.json','e1_r2_cohorts.json'),('targets.json','e1_r2_targets.json'),('matching.json','e1_r2_matching.json'),('image_bindings.json','e1_r2_image_bindings.json')]:
    src=E1R2_MANIFEST/source_name;assert src.is_file(),src
    dst=OUT/dest_name;shutil.copy2(src,dst);manifest.append({'path':dest_name,'sha256':sha(dst),'bytes':dst.stat().st_size,'source':str(src.relative_to(R))})

counts={};maxima={}
for kind,expected in [('e1',1000),('e2',200),('e3',200),('e5',1600)]:
    receipts=list((RUN/kind).glob('*.json'));assert len(receipts)==expected,(kind,len(receipts))
    checks=[]
    for path in receipts:
        row=json.loads(path.read_text());assert row['status']=='VALID';assert sha(R/row['output'])==row['output_sha256'];checks.append(row['checks'])
    counts[kind]={'expected':expected,'valid':len(checks),'failures':0}
    if kind=='e1':maxima[kind]={'scalar_vjp_parity_max':max(x['scalar_vjp_parity_max'] for x in checks)}
    if kind=='e2':maxima[kind]={'closure_max':max(x['closure_max'] for x in checks),'closure_checks_total':sum(x['closure_checks'] for x in checks)}
    if kind=='e3':maxima[kind]={'clean_suffix_max':max(x['clean_suffix_max'] for x in checks),'suffix_forward_examples':sum(x['suffix_forward_examples'] for x in checks)}
    if kind=='e5':maxima[kind]={'scalar_vjp_parity_max':max(x['scalar_vjp_parity_max'] for x in checks),'closure_max':max(x['closure_max'] for x in checks),'closure_checks_total':sum(x['closure_checks'] for x in checks)}
e1r2_receipts=list((E1R2_RUN/'e1').glob('*.json'));assert len(e1r2_receipts)==1000
e1r2_checks=[]
for path in e1r2_receipts:
    row=json.loads(path.read_text());assert row['status']=='VALID';assert sha(R/row['output'])==row['output_sha256'];e1r2_checks.append(row['checks'])
counts['e1_r2']={'expected':1000,'valid':len(e1r2_checks),'failures':0}
maxima['e1_r2']={'scalar_vjp_parity_max':max(x['scalar_vjp_parity_max'] for x in e1r2_checks)}

receipt={'status':'PASS_COMPACT_EVIDENCE_PACKAGE','run_roots':[str(RUN.relative_to(R)),str(E1R2_RUN.relative_to(R))],'counts':counts,'numeric_checks':maxima,
         'files':manifest,'files_digest':hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
         'large_arrays_included':False,'large_arrays_policy':'approximately 3.9GB of task arrays stay in the two run roots; analysis receipts bind every source JSON and NPZ by SHA-256'}
(OUT/'package_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
index='''# Revision v5 evidence index

This directory is the compact evidence package for the v5 thesis revision.

- protocol, cohort, target, image-binding, and estimand files freeze models, images, neurons, missing-value rules, and tests.
- E1-R2 files contain the prospective WordNet-metadata-matched category-consistency evidence. Earlier E1 files retain the RGB-matched diagnostic and activation selectivity.
- E2 files contain MLP-neuron and attention-head spatial summaries.
- E3 summary files contain neuron-intervention ranking, top-k, activation-matching, and target-selectivity comparisons. Image-level E3 files remain in the run root because they total hundreds of MB.
- E4 files contain held-out ridge fits, predictions, paired errors, and similar-total-map pairs.
- E5 files contain the four frozen independent-image/checkpoint hypotheses.
- The package receipt verifies source counts, numerical controls, file hashes, and the boundary between compact evidence and retained large arrays.

The original C4a/C4b evidence remains under the earlier publication and report roots and is not overwritten by these prospective supplementary experiments.
'''
(OUT/'INDEX.md').write_text(index)
print(json.dumps({'status':receipt['status'],'files':len(manifest),'counts':counts,'numeric_checks':maxima},indent=2))
