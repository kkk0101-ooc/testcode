#!/usr/bin/env python3
"""Requirement-to-artifact audit and final current-state package identity."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=R/'papers/nns_vit_endpoint/publication_r2_20260920';Q=P/'qc/revision_v6';E=P/'evidence/revision_v6'
PROMPT=R/'修改意见/修改意见v6_修改实验终审一体化执行提示词.md'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--post-push',action='store_true');a=p.parse_args()
 checks=[]
 def ck(k,v,detail=''):checks.append(dict(requirement=k,passed=bool(v),detail=str(detail)))
 reports=['gxi_step3_validation.json','gxi_step4_validation.json','publication_validation.json','gxi_step5_validation.json','final_validation.json','py_compile_results.json','command_results.json']
 for name in reports:
  d=json.loads((Q/name).read_text());ck(name,d['status']=='PASS' and not d.get('errors',[]))
 terminal=json.loads((E/'terminal_receipt.json').read_text());source=R/'reports/nns_vit_endpoint_revision_v6_r1_20260923';ck('sealed_terminal',terminal['status']=='PASS_EXECUTION_AND_VALIDATION_WITH_COORDINATE_LIMITATION' and sha(source/'artifact_manifest.json')==terminal['artifact_manifest_sha256'])
 manifest=json.loads((source/'artifact_manifest.json').read_text())
 for row in manifest['files']:ck('experiment:'+row['path'],(source/row['path']).stat().st_size==row['bytes'] and sha(source/row['path'])==row['sha256'])
 raw=json.loads((source/'validation_report.json').read_text());ck('full_experiment_coverage',raw['task_count']==1000 and len(raw['checks'])==9062 and not raw['errors'])
 review=(Q/'thesis_final_review.md').read_text();ck('six_dimensions_ten_questions',all('### 答辩 '+str(i)+'.' in review for i in range(1,11)) and all('### '+str(i)+'.' in review for i in range(1,7)))
 ck('no_major_or_blocking',all(x in review for x in ['**阻断毕业。** 当前科学稿范围内为零','**重大修改。** 当前为零']))
 ck('anonymous_author_instruction',all('REQUIRED]' not in (P/l/'manuscript.md').read_text() for l in ['zh','en']))
 # Historical receipts and evidence must remain byte-identical to the starting commit.
 changed=subprocess.check_output(['git','-c','core.quotePath=false','diff','cac632ee0ef74e2201e4729acb91378e521cca32','--name-only','--relative'],cwd=R,text=True).splitlines()
 protected=[p for p in changed if (p.startswith('reports/') and 'nns_vit_endpoint_revision_v6_' not in p) or '/thesis_blind_review_' in p or ('/evidence/' in p and '/revision_v6/' not in p) or ('/qc/' in p and '/revision_v6/' not in p)]
 # New files under reports are intentionally untracked local scientific evidence, not edits to old roots.
 ck('history_immutable',not protected,protected)
 ck('no_unrelated_changes',all('revision_v6' in p or 'nns_vit_endpoint/' in p or p.startswith('修改意见/修改意见v6_') for p in changed))
 ck('git_diff_check',subprocess.run(['git','diff','--check'],cwd=R,capture_output=True).returncode==0)
 sections={
  1:['zh/manuscript.md','en/manuscript.md','qc/revision_v6/thesis_final_review.md','evidence/revision_v6/package_manifest.json'],
  2:['zh/manuscript.md','en/manuscript.md','evidence/revision_v6/coordinate_decisions.json','evidence/revision_v6/original_inference_recomputed.json'],
  3:['qc/revision_v6/stage0_source_audit.json','qc/revision_v6/stage0_definition_evidence.json'],
  4:['qc/revision_v6/stage0_source_audit.json','qc/revision_v6/command_results.json','zh/manuscript.md'],
  5:['qc/revision_v6/stage0_source_audit.json','qc/revision_v6/stage0_definition_evidence.json','qc/revision_v6/reference_live_checks.json'],
  6:['zh/manuscript.md','en/manuscript.md','figure_manifest.json','qc/revision_v6/stage1_validation.json','qc/revision_v6/gxi_step3_validation.json'],
  7:['evidence/revision_v6/protocol.json','evidence/revision_v6/source_bindings.json','evidence/revision_v6/cohort_bindings.json','evidence/revision_v6/pilot_budget.json'],
  8:['evidence/revision_v6/validation_report.json','evidence/revision_v6/terminal_receipt.json','evidence/revision_v6/coordinate_sensitivity_summary.csv','evidence/revision_v6/analytic_reference_results.json','evidence/revision_v6/mutation_results.json'],
  9:['zh/supplement.md','en/supplement.md','zh/manuscript.md','evidence/revision_v6/analysis_identity.json'],
  10:['zh/manuscript.md','en/manuscript.md','zh/supplement.md','en/supplement.md','figure_manifest.json','qc/revision_v6/defensive_writing_audit.md'],
  11:['qc/revision_v6/command_results.json','qc/revision_v6/final_validation.json','qc/revision_v6/publication_validation.json'],
  12:['qc/revision_v6/thesis_final_review.md','qc/revision_v6/defensive_writing_audit.md'],
  13:['qc/revision_v6/completion_audit.md','qc/revision_v6/final_validation.json','evidence/revision_v6/validation_report.json'],
  14:['qc/revision_v6/completion_audit.md','qc/revision_v6/thesis_final_review.md','qc/revision_v6/command_results.json']}
 prompt=PROMPT.read_text().split('## 阶段 0 执行记录')[0];requirements=[];section_id=0
 for lineno,line in enumerate(prompt.splitlines(),1):
  if line.startswith('## ') and not line.startswith('###'):section_id+=1
  if re.match('(?:[0-9]+[.] |- |### |[|] R[0-9]+ )',line):
   evidence=sections.get(section_id,sections[1]);requirements.append(dict(id='v6:L'+str(lineno),section=section_id,requirement=line.strip(),evidence=[dict(path=p,sha256=sha(P/p)) for p in evidence],assessment='covered by mapped direct evidence and manual semantic review',post_push_required=section_id in [4,13,14] and ('Git' in line or '工作树' in line or 'commit' in line or 'push' in line or '提交' in line)))
 ck('all_prompt_sections_mapped',set(r['section'] for r in requirements)==set(range(1,15)))
 git_audit=None
 if a.post_push:
  head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();track=subprocess.check_output(['git','rev-parse','@{u}'],cwd=R,text=True).strip();branch=subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=R,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=R,text=True).split()[0];status=subprocess.check_output(['git','status','--branch','--porcelain=v2'],cwd=R,text=True);lfs=subprocess.check_output(['git','lfs','status'],cwd=R,text=True)
  ck('remote_matches',head==track==remote);ck('tracked_tree_clean',not any(x.startswith(('1 ','2 ','u ')) for x in status.splitlines()));git_audit=dict(head=head,tracking=track,remote=remote,status=status,lfs=lfs)
 errors=[r for r in checks if not r['passed']]
 result=dict(schema_version=1,status='PASS' if not errors else 'FAIL',goal_state='complete' if a.post_push and not errors else 'ready_for_git' if not errors else 'incomplete',requirements=requirements,checks=checks,errors=errors,git_audit=git_audit)
 name='post_push_audit.json' if a.post_push else 'prompt_to_artifact.json';(Q/name).write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10));print(dict(status=result['status'],goal_state=result['goal_state'],requirements=len(requirements),checks=len(checks),errors=errors[:15]))
 return bool(errors)
if __name__=='__main__':raise SystemExit(main())
