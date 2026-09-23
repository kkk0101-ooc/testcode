#!/usr/bin/env python3
"""Create revision-v8 package manifest and terminal review receipt."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'reports/nns_vit_endpoint_revision_v8_r2_20260923'
PUB=REPO/'papers/nns_vit_endpoint/publication_r2_20260920'
QC=PUB/'qc/revision_v8'
REG=REPO/'manifests/nns_vit_endpoint/revision_v8_r2_20260923'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rec(path):return {'path':path.relative_to(REPO).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}


def main():
    analysis=json.loads((ROOT/'terminal_receipt.json').read_text())
    validation=json.loads((ROOT/'validation_report.json').read_text())
    publication=json.loads((ROOT/'publication_validation_report.json').read_text())
    build=json.loads((ROOT/'latex_build_report.json').read_text())
    required=[ROOT/name for name in ('protocol.json','source_bindings.json','stage0_receipt.json','architecture_graph_audit.json','target_cut_applicability.csv','target_cut_summary.csv','entropy_mass_thresholds.csv','entropy_mass_image_values.csv','entropy_mass_sensitivity.csv','h4_matched_block_values.csv','h4_matched_image_values.csv','h4_matched_summary.csv','random_region_precision_audit.json','terminal_receipt.json','validation_report.json','latex_build_report.json','publication_validation_report.json')]
    required += [REG/name for name in ('scalar_target_registry.json','decomposition_registry.json','intervention_registry.json')]
    required += [PUB/language/f'{kind}.{suffix}' for language in ('zh','en') for kind in ('manuscript','supplement') for suffix in ('md','tex','pdf')]
    required += [PUB/language/'supplement_experiments_v1.md' for language in ('zh','en')]
    required += [QC/name for name in ('defensive_writing_audit.md','adversarial_final_review.md','pdf_visual_review.md','revision_disposition.md')]
    required += [REPO/name for name in ('scripts/run_revision_v8_analysis.py','scripts/validate_revision_v8_analysis.py','tests/test_revision_v8.py','scripts/run_revision_v8_r2_analysis.py','scripts/validate_revision_v8_r2_analysis.py','tests/test_revision_v8_r2.py','scripts/build_revision_v8_publication.py','scripts/validate_revision_v8_publication.py','scripts/finalize_revision_v8_package.py','修改意见/修改意见v8_盲审前结构与敏感性补强工作规程.md')]
    if not all(path.is_file() for path in required):raise ValueError('MISSING_PACKAGE_FILE')
    manifest={'schema_version':1,'status':'PASS','files':[rec(path) for path in required],'file_count':len(required),'total_bytes':sum(path.stat().st_size for path in required)}
    (ROOT/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    checks={'analysis':analysis['status']=='REVISION_V8_ANALYSIS_PASS','independent_validation':validation['status']=='PASS','latex':build['status']=='PASS_TWO_CONSECUTIVE_CLEAN_BUILDS','publication':publication['status']=='PASS','defensive_writing':'PASS_DEFENSIVE_WRITING_AUDIT' in (QC/'defensive_writing_audit.md').read_text(),'visual':'PASS_VISUAL_AND_ANONYMITY_REVIEW' in (QC/'pdf_visual_review.md').read_text(),'review':'**可送审**' in (QC/'adversarial_final_review.md').read_text()}
    receipt={'schema_version':1,'status':'REVISION_V8_STAGE3_PASS_AND_READY_TO_COMMIT','checks':checks,'blocking_issues':0,'major_issues':0,'review_decision':'可送审','gpu_inference_runs':0,'pdf_pages':sum(row['pages'] for row in publication['pdfs']),'package_manifest':rec(ROOT/'package_manifest.json'),'next_authorization':'COMMIT_AND_PUSH_REVISION_V8'}
    if not all(checks.values()):receipt['status']='FAIL'
    (ROOT/'final_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))
    return 0 if receipt['status'].startswith('REVISION_V8_STAGE3_PASS') else 1


if __name__=='__main__':raise SystemExit(main())
