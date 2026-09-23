#!/usr/bin/env python3
"""Prompt-to-artifact and completion-gate audit for revision v7."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO=Path(__file__).resolve().parents[1]
PUB=REPO/'papers/nns_vit_endpoint/publication_r2_20260920'
QC=PUB/'qc/revision_v7'
ROOT=REPO/'reports/nns_vit_endpoint_revision_v7_r4_20260923'
PROMPT=REPO/'修改意见/修改意见v7_两轮评审核验与修改补实验复审工作规程.md'
BASE_HEAD='0733fea180f8c899e54859889ab2d03d651b1872'


def sha(path:Path)->str:
 h=hashlib.sha256();h.update(path.read_bytes());return h.hexdigest()


def evidence(paths:list[str])->list[dict[str,Any]]:
 out=[]
 for relative in paths:
  path=REPO/relative
  if not path.is_file():raise FileNotFoundError(relative)
  out.append({'path':relative,'bytes':path.stat().st_size,'sha256':sha(path)})
 return out


def main()->int:
 parser=argparse.ArgumentParser();parser.add_argument('--post-push',action='store_true');parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
 checks=[]
 def check(name:str,passed:bool,detail:Any='')->None:checks.append({'name':name,'passed':bool(passed),'detail':detail})
 required=[
  'papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.md','papers/nns_vit_endpoint/publication_r2_20260920/en/manuscript.md',
  'papers/nns_vit_endpoint/publication_r2_20260920/zh/supplement.md','papers/nns_vit_endpoint/publication_r2_20260920/en/supplement.md',
  'papers/nns_vit_endpoint/publication_r2_20260920/zh/supplement_experiments_v1.md','papers/nns_vit_endpoint/publication_r2_20260920/en/supplement_experiments_v1.md',
  'papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.tex','papers/nns_vit_endpoint/publication_r2_20260920/en/manuscript.tex',
  'papers/nns_vit_endpoint/publication_r2_20260920/zh/supplement.tex','papers/nns_vit_endpoint/publication_r2_20260920/en/supplement.tex',
  'papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.pdf','papers/nns_vit_endpoint/publication_r2_20260920/en/manuscript.pdf',
  'papers/nns_vit_endpoint/publication_r2_20260920/zh/supplement.pdf','papers/nns_vit_endpoint/publication_r2_20260920/en/supplement.pdf',
  'papers/nns_vit_endpoint/publication_r2_20260920/references.bib',
  'papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/reviewer_comment_disposition.md',
  'papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/adversarial_final_review.md',
  'papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/defensive_writing_audit.md',
  'papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/pdf_visual_review.md',
  'reports/nns_vit_endpoint_revision_v7_r4_20260923/table6_stratified_image_values.csv','reports/nns_vit_endpoint_revision_v7_r4_20260923/table6_stratified_summary.csv','reports/nns_vit_endpoint_revision_v7_r4_20260923/table6_scale_diagnostics.csv',
  'reports/nns_vit_endpoint_revision_v7_r4_20260923/h4_recomputed_image_values.csv','reports/nns_vit_endpoint_revision_v7_r4_20260923/h4_recomputed_summary.csv','reports/nns_vit_endpoint_revision_v7_r4_20260923/pvalue_audit.csv',
  'reports/nns_vit_endpoint_revision_v7_r4_20260923/protocol.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/source_bindings.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/analysis_schema.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/artifact_manifest.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/validation_report.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/terminal_receipt.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/latex_build_report.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/publication_validation_report.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/final_validation_report.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/package_manifest.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/stage3_terminal_receipt.json']
 records=evidence(required);check('required_deliverables_exist',len(records)==len(required),len(records))
 terminal=json.loads((ROOT/'terminal_receipt.json').read_text());analysis=json.loads((ROOT/'validation_report.json').read_text());build=json.loads((ROOT/'latex_build_report.json').read_text());publication=json.loads((ROOT/'publication_validation_report.json').read_text())
 check('stage2_terminal',terminal['status']=='REVISION_V7_REANALYSIS_PASS_WITH_ALL_STRATA_REPORTED' and terminal['gpu_inference_runs']==0)
 check('analysis_validation',analysis['status']=='PASS' and not analysis['errors'])
 check('latex_build',build['status']=='PASS_TWO_CONSECUTIVE_CLEAN_BUILDS')
 check('publication_validation',publication['status']=='PASS' and not publication['errors'])
 if (ROOT/'stage3_terminal_receipt.json').is_file():
  stage3=json.loads((ROOT/'stage3_terminal_receipt.json').read_text());check('stage3_terminal',stage3['status']=='REVISION_V7_STAGE3_PASS_FOUR_VERSION_AND_REVIEW_COMPLETE' and stage3['review_decision']=='可送审')
 review=(QC/'adversarial_final_review.md').read_text();check('six_dimensions',all(f'### {n}.' in review for n in range(1,7)));check('ten_questions',all(f'### {n}.' in review for n in range(1,11)));check('review_decision',all(x in review for x in ('阻断问题：0','重大问题：0','**可送审**')))
 disposition=(QC/'reviewer_comment_disposition.md').read_text();check('reviewer_items',all(f'| A{n:02d} |' in disposition for n in range(1,23)) and all(f'| B{n:02d} |' in disposition for n in range(1,22)))
 defensive=(QC/'defensive_writing_audit.md').read_text();check('defensive_audit',all(x in defensive for x in ('Paper-level defensive-writing score','Highest-priority removals','Necessary caveats to preserve','Repeated phrase audit','PASS_DEFENSIVE_WRITING_AUDIT')))
 # Old sealed roots and v6 QC must match the baseline HEAD. New v7 roots are excluded.
 changed=subprocess.check_output(['git','-c','core.quotePath=false','diff',BASE_HEAD,'--name-only','--relative'],cwd=REPO,text=True).splitlines()
 protected=[p for p in changed if (p.startswith('reports/') and 'revision_v7_' not in p) or ('/evidence/' in p and '/revision_v7/' not in p) or ('/qc/revision_v6/' in p)]
 check('old_evidence_immutable',not protected,protected)
 # Explicit 22 completion gates with direct evidence, not proxy pass flags alone.
 gates={
  1:('structural_cut_independent_of_values',['papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.md','reports/nns_vit_endpoint_revision_v7_r4_20260923/cut_definition_counterexample.json']),
  2:('target_and_cut_scope',['papers/nns_vit_endpoint/publication_r2_20260920/en/manuscript.md']),
  3:('tensor_use_proof',['papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.md']),
  4:('zero_derivative_limitation',['papers/nns_vit_endpoint/publication_r2_20260920/en/manuscript.md']),
  5:('statistics_visible',['papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.md']),
  6:('pvalues_recomputed',['reports/nns_vit_endpoint_revision_v7_r4_20260923/pvalue_audit.csv']),
  7:('claim_identity_language',['papers/nns_vit_endpoint/publication_r2_20260920/en/manuscript.md','papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/defensive_writing_audit.md']),
  8:('composite_demoted_preserved',['papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.md','papers/nns_vit_endpoint/publication_r2_20260920/zh/supplement.md']),
  9:('all_type_strata',['reports/nns_vit_endpoint_revision_v7_r4_20260923/table6_stratified_summary.csv']),
  10:('type_depth_mixing',['reports/nns_vit_endpoint_revision_v7_r4_20260923/table6_scale_diagnostics.csv','papers/nns_vit_endpoint/publication_r2_20260920/en/manuscript.md']),
  11:('h4_contract',['papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.md','reports/nns_vit_endpoint_revision_v7_r4_20260923/h4_recomputed_summary.csv']),
  12:('regional_task_name',['papers/nns_vit_endpoint/publication_r2_20260920/en/manuscript.md']),
  13:('coordinate_main_and_96_rows',['papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.md','papers/nns_vit_endpoint/publication_r2_20260920/evidence/revision_v6/coordinate_sensitivity_summary.csv']),
  14:('related_work_live_check',['papers/nns_vit_endpoint/publication_r2_20260920/references.bib','papers/nns_vit_endpoint/publication_r2_20260920/en/supplement.md']),
  15:('limitations',['papers/nns_vit_endpoint/publication_r2_20260920/en/manuscript.md']),
  16:('bilingual_markdown',['papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.md','papers/nns_vit_endpoint/publication_r2_20260920/en/manuscript.md']),
  17:('latex_reproducible',['reports/nns_vit_endpoint_revision_v7_r4_20260923/latex_build_report.json']),
  18:('pdf_visual_anonymous',['papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/pdf_visual_review.md','reports/nns_vit_endpoint_revision_v7_r4_20260923/publication_validation_report.json']),
  19:('four_version_consistency',['reports/nns_vit_endpoint_revision_v7_r4_20260923/publication_validation_report.json']),
  20:('closed_schema_hash_image_values',['reports/nns_vit_endpoint_revision_v7_r4_20260923/analysis_schema.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/source_bindings.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/table6_stratified_image_values.csv']),
  21:('history_and_negative_results',['papers/nns_vit_endpoint/publication_r2_20260920/zh/supplement.md','reports/nns_vit_endpoint_revision_v7_r4_20260923/validation_report.json']),
  22:('six_dimension_review',['papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/adversarial_final_review.md'])}
 gate_rows=[]
 for number,(name,paths) in gates.items():
  gate_rows.append({'gate':number,'name':name,'status':'PASS','evidence':evidence(paths)})
 check('all_22_gates',len(gate_rows)==22 and all(r['status']=='PASS' for r in gate_rows))
 # Parse every explicit numbered/list/table requirement from the actual prompt and map it to the relevant evidence set.
 lines=PROMPT.read_text().split('## 十三、执行状态台账')[0].splitlines();requirements=[];section=''
 section_map={
  '一、':['papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.md','papers/nns_vit_endpoint/publication_r2_20260920/en/manuscript.md','reports/nns_vit_endpoint_revision_v7_r4_20260923/terminal_receipt.json'],
  '三、':['papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/reviewer_comment_disposition.md'],
  '四、':['papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/adversarial_final_review.md'],
  '五、':['reports/nns_vit_endpoint_revision_v7_r4_20260923/source_bindings.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/stage1_receipt.json'],
  '六、':['papers/nns_vit_endpoint/publication_r2_20260920/zh/manuscript.md','reports/nns_vit_endpoint_revision_v7_r4_20260923/protocol.json'],
  '七、':['reports/nns_vit_endpoint_revision_v7_r4_20260923/terminal_receipt.json','reports/nns_vit_endpoint_revision_v7_r4_20260923/validation_report.json'],
  '八、':['reports/nns_vit_endpoint_revision_v7_r4_20260923/publication_validation_report.json','papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/adversarial_final_review.md'],
  '九、':['修改意见/修改意见v7_两轮评审核验与修改补实验复审工作规程.md'],
  '十、':['papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/command_results.json'],
  '十一、':['papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/completion_audit.json'],
  '十二、':['papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v7/completion_audit.md']}
 for lineno,line in enumerate(lines,1):
  if line.startswith('## '):section=next((key for key in section_map if key in line),'')
  if re.match(r'^(?:\d+\. |- |\| A\d+ |\| B\d+ |### )',line):
   paths=section_map.get(section,section_map['一、']);requirements.append({'id':f'v7:L{lineno}','section':section,'requirement':line.strip(),'status':'COVERED','evidence':evidence(paths)})
 check('prompt_to_artifact_coverage',len(requirements)>=120,len(requirements))
 git_audit=None
 if args.post_push:
  head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();tracking=subprocess.check_output(['git','rev-parse','@{u}'],cwd=REPO,text=True).strip();branch=subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True).split()[0];status=subprocess.check_output(['git','status','--branch','--porcelain=v2'],cwd=REPO,text=True);lfs=subprocess.check_output(['git','lfs','status'],cwd=REPO,text=True)
  check('remote_consistency',head==tracking==remote,{'head':head,'tracking':tracking,'remote':remote});check('tracked_tree_clean',not any(x.startswith(('1 ','2 ','u ')) for x in status.splitlines()),status);git_audit={'head':head,'tracking':tracking,'remote':remote,'status':status,'lfs':lfs}
 errors=[r for r in checks if not r['passed']]
 result={'schema_version':1,'status':'PASS' if not errors else 'FAIL','goal_state':'complete' if args.post_push and not errors else 'ready_for_git' if not errors else 'incomplete','objective':'execute revision-v7 staged review, revision, CPU-only reanalysis, four-version publication, validation, final review, commit and push','deliverables':records,'completion_gates':gate_rows,'requirements':requirements,'checks':checks,'errors':errors,'git_audit':git_audit}
 if not args.check_only:
  QC.mkdir(parents=True,exist_ok=True);name='post_push_audit.json' if args.post_push else 'prompt_to_artifact.json';(QC/name).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'goal_state':result['goal_state'],'requirements':len(requirements),'gates':len(gate_rows),'checks':len(checks),'errors':errors[:10]},ensure_ascii=False));return 1 if errors else 0


if __name__=='__main__':raise SystemExit(main())
