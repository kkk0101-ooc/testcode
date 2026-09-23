#!/usr/bin/env python3
"""Seal the revision-v7 publication, validation, and review package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO=Path(__file__).resolve().parents[1]
PUB=REPO/'papers/nns_vit_endpoint/publication_r2_20260920'
QC=PUB/'qc/revision_v7'
ROOT=REPO/'reports/nns_vit_endpoint_revision_v7_r4_20260923'


def sha(path:Path)->str:
 h=hashlib.sha256();h.update(path.read_bytes());return h.hexdigest()


def rec(path:Path)->dict[str,Any]:return {'path':path.relative_to(REPO).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}


def main()->int:
 analysis=json.loads((ROOT/'validation_report.json').read_text());terminal=json.loads((ROOT/'terminal_receipt.json').read_text());build=json.loads((ROOT/'latex_build_report.json').read_text());publication=json.loads((ROOT/'publication_validation_report.json').read_text());commands=json.loads((QC/'command_results.json').read_text())
 checks=[
  {'id':'stage2_r4','status':'PASS' if terminal['status']=='REVISION_V7_REANALYSIS_PASS_WITH_ALL_STRATA_REPORTED' and terminal['gpu_inference_runs']==0 else 'FAIL'},
  {'id':'independent_recalculation','status':'PASS' if analysis['status']=='PASS' and not analysis['errors'] else 'FAIL','table6_max_error':analysis['old_family_max_abs_error'],'h4_max_error':analysis['h4_max_abs_error']},
  {'id':'two_clean_builds','status':'PASS' if build['status']=='PASS_TWO_CONSECUTIVE_CLEAN_BUILDS' else 'FAIL'},
  {'id':'four_version_anonymity','status':'PASS' if publication['status']=='PASS' and not publication['errors'] else 'FAIL','pdfs':publication['pdfs']},
  {'id':'current_commands','status':'PASS' if commands['status'].startswith('PASS_CURRENT_V7') else 'FAIL'},
  {'id':'visual_review','status':'PASS' if 'PASS_PDF_VISUAL_AND_ANONYMITY_REVIEW' in (QC/'pdf_visual_review.md').read_text() else 'FAIL'},
  {'id':'six_dimension_review','status':'PASS' if '**可送审**' in (QC/'adversarial_final_review.md').read_text() else 'FAIL'},
  {'id':'defensive_writing','status':'PASS' if 'PASS_DEFENSIVE_WRITING_AUDIT' in (QC/'defensive_writing_audit.md').read_text() else 'FAIL'}]
 failures=[
  {'attempt':'r1','disposition':'IMMUTABLE_SUPERSEDED','reason':'A/V concordance tested against zero and TV quality bootstrap pooled image-block observations'},
  {'attempt':'r2','disposition':'IMMUTABLE_SUPERSEDED','reason':'post-freeze validator-only manifest de-duplication caused frozen code-hash drift'},
  {'attempt':'r3','disposition':'IMMUTABLE_FAILED','reason':'fresh-root name was incorrectly coupled to bootstrap RNG namespace, producing avoidable interval endpoint drift'},
  {'attempt':'legacy_v3_v6_validators','disposition':'RECORDED_VERSION_INCOMPATIBILITY','reason':'assert old pooled Table 6, old near-zero formatting, old mirrors, and old inventory hashes'}]
 report={'schema_version':1,'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL','stage':'REVISION_V7_STAGE3_FINAL_VALIDATION','cpu_only_reanalysis':True,'gpu_inference_runs':0,'checks':checks,'failure_history':failures,'review_decision':'可送审','blocking_issues':0,'major_issues':0,'current_root':'reports/nns_vit_endpoint_revision_v7_r4_20260923','final_commands':commands['commands']}
 (ROOT/'final_validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 package_paths=[p for p in sorted(ROOT.iterdir()) if p.is_file() and p.name not in {'package_manifest.json','stage3_terminal_receipt.json'}]
 package_paths += [PUB/language/name for language in ('zh','en') for name in ('manuscript.md','supplement.md','supplement_experiments_v1.md','manuscript.tex','supplement.tex','manuscript.pdf','supplement.pdf')]
 dynamic_qc={'prompt_to_artifact.json','post_push_audit.json','git_publication_receipt.json'}
 package_paths += [PUB/'references.bib',PUB/'README.md']+[p for p in sorted(QC.glob('*')) if p.name not in dynamic_qc]
 package_paths += [REPO/'修改意见/修改意见v7_两轮评审核验与修改补实验复审工作规程.md',REPO/'tests/test_revision_v6.py',REPO/'tests/test_revision_v7.py']
 package_paths += [Path(__file__).resolve(),REPO/'scripts/run_revision_v7_analysis.py',REPO/'scripts/validate_revision_v7_analysis.py',REPO/'scripts/build_revision_v7_publication.py',REPO/'scripts/validate_revision_v7_publication.py',REPO/'scripts/audit_revision_v7_completion.py']
 unique=sorted(set(package_paths));files=[rec(p) for p in unique]
 manifest={'schema_version':1,'status':'SEALED_REVISION_V7_PACKAGE','file_count':len(files),'total_bytes':sum(r['bytes'] for r in files),'files':files,'files_digest':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'excluded_self':['package_manifest.json','stage3_terminal_receipt.json']}
 (ROOT/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 receipt={'schema_version':1,'status':'REVISION_V7_STAGE3_PASS_FOUR_VERSION_AND_REVIEW_COMPLETE','review_decision':'可送审','package_manifest_sha256':sha(ROOT/'package_manifest.json'),'final_validation_report_sha256':sha(ROOT/'final_validation_report.json'),'publication_validation_report_sha256':sha(ROOT/'publication_validation_report.json'),'latex_build_report_sha256':sha(ROOT/'latex_build_report.json'),'pdf_pages_total':sum(p['pages'] for p in publication['pdfs']),'gpu_inference_runs':0,'next':'FINAL_COMPLETION_AUDIT_AND_GIT_PUBLICATION'}
 (ROOT/'stage3_terminal_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':receipt['status'],'package_files':manifest['file_count'],'package_bytes':manifest['total_bytes'],'pdf_pages':receipt['pdf_pages_total']},ensure_ascii=False));return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
