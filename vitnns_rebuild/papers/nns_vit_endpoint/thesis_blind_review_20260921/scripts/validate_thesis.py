#!/usr/bin/env python3
"""Revision-v4 requirement, bilingual, source-number, and artifact validator (stdlib)."""
import csv,hashlib,json,re,statistics
from pathlib import Path
P=Path(__file__).resolve().parents[1];R=P.parents[2];D=P/'evidence/revision_v4';PUB=R/'papers/nns_vit_endpoint/publication_r2_20260920'
CHECKS=[]
def check(name,ok,detail=''):CHECKS.append(dict(name=name,passed=bool(ok),detail=detail))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return p.read_text()
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def gettable(s,id):
 m=re.search(r'^\*\*(?:表|Table) '+re.escape(id)+r'[^\n]*\*\*\n\n((?:\|[^\n]*\n)+)',s,re.M)
 return m.group(1) if m else ''
def nums(s):
 s='\n'.join(s.splitlines()[2:])
 return re.findall(r'(?<![A-Za-z])[-−]?\d+(?:\.\d+)?(?:e[-+]?\d+)?',s.replace('−','-'),re.I)
def section(s,start,end):return s.split(start,1)[1].split(end,1)[0]
def citerange(s):
 out=set()
 for m in re.findall(r'\[([\d, -]+)\]',s):
  for part in m.split(','):
   part=part.strip()
   if re.fullmatch(r'\d+-\d+',part):a,b=map(int,part.split('-'));out.update(range(a,b+1))
   elif part.isdigit():out.add(int(part))
 return out

def main():
 z=read(P/'zh/manuscript.md');e=read(P/'en/manuscript.md');ref=read(P/'references.md');op=R/'修改意见/修改意见v4.md'
 check('v4_source_exists',op.is_file())
 for lang,s in [('zh',z),('en',e)]:
  pat=r'^# 第 ([1-8]) 章 ' if lang=='zh' else r'^# Chapter ([1-8]) '
  check('eight_chapters_'+lang,re.findall(pat,s,re.M)==list('12345678'))
  check('only_three_RQ_'+lang,'RQ4' not in s and set(re.findall(r'RQ([123])[:：]',s))==set('123'))
  check('no_control_characters_'+lang,not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',s))
  check('balanced_math_'+lang,s.count('$$')%2==0)
  before=s.split('# 参考文献' if lang=='zh' else '# References')[0]
  check('citations_1_to_41_'+lang,citerange(before)==set(range(1,42)),str(sorted(citerange(before))))
  check('no_identity_markers_'+lang,not re.search(r'/data01|yeziqi|ByteDance|字节跳动|\[(?:AUTHOR|AFFILIATION).*?REQUIRED\]',s,re.I))
  check('no_unresolved_placeholders_'+lang,not re.search(r'\[(?:EVIDENCE|PARAMETER|REFERENCE) REQUIRED\]',s))
  check('no_unqualified_model_generalization_'+lang,not re.search(r'ViT 普遍|训练导致|全局语义形成|ViTs universally|training causes',before,re.I))
  check('engineering_in_appendix_'+lang,not re.search(r'SHA-256|CUBLAS_WORKSPACE|终端收据|queue|fencing_token',before,re.I))
  check('reference_versions_'+lang,all(x in s for x in ['2404.03214v1','2504.19475v1','2604.14477v1','2603.13652v1','2026-09-21']))
  figures=re.findall(r'!\[[^]]*\]\(([^)]+)\)',s)
  check('all_figure_links_'+lang,all((P/lang/x).resolve().is_file() for x in figures))
  check('nine_main_figures_'+lang,len(figures)==9,str(len(figures)))
 check('parallel_subsection_structure',re.findall(r'^## (\d+(?:\.\d+)*|[AB]\.\d+) ',z,re.M)==re.findall(r'^## (\d+(?:\.\d+)*|[AB]\.\d+) ',e,re.M))
 check('math_exact_bilingual',re.findall(r'\$\$(.*?)\$\$',z,re.S)==re.findall(r'\$\$(.*?)\$\$',e,re.S))
 tableids=['2-1','3-1','3-2','5-1','5-2','5-3','6-1','6-2','6-3','6-4','6-5','6-6','7-1','7-2','7-3','A-1']
 for tid in tableids:
  a,b=gettable(z,tid),gettable(e,tid)
  check('table_present_'+tid,bool(a and b))
  check('table_numeric_parity_'+tid,nums(a)==nums(b),str((nums(a),nums(b))) if nums(a)!=nums(b) else '')
 formulas=[f'3-{i}' for i in range(1,6)]+[f'4-{i}' for i in range(1,17)]+[f'5-{i}' for i in range(1,4)]+['7-1']
 for f in formulas:check('formula_'+f,f'**式（{f}）**' in z and f'**Equation ({f})**' in e)
 # Semantic sentinels tied to named chapter requirements; full prose interpretation is recorded in completion audit.
 requirements={
  'edge_tuple':('c=(u,v,r,z_c).','c=(u,v,r,z_c).'),
  'family_definition':('\\Phi_C(F,x)=\\{E_c^C(x):c\\in C\\}','\\Phi_C(F,x)=\\{E_c^C(x):c\\in C\\}'),
  'five_cut_conditions':('合法切面须满足五项条件','A legal cut satisfies five conditions'),
  'shared_use_ports':('不同端口使用','different operators or ports'),
  'original_operator_map':('原 NNS 操作','Original NNS operation'),
  'no_official_software':('原作者官方实现当前不可获得','official implementation is unavailable'),
  'precision_design':('0.049','0.049'),
  'not_threshold_hypothesis':('不是对总体相关大于 0.10 的直接检验','does not directly test whether population correlation exceeds 0.10'),
  'three_analysis_classes':('后继冻结分析','Successor frozen analysis'),
  'new_analysis_posthoc':('事后描述性重分析','post-hoc descriptive reanalysis'),
  'random_estimator_change':('结果读取后的估计量修正','post-outcome estimator correction'),
  'full_family':('28 个双侧精确符号检验','28 two-sided exact sign tests'),
  'semantic_future':('均为未来工作','These are future designs'),
  'same_image_groups':('共用同一 100 图','same 100 images'),
  'target_scope':('标注类别条件下的结构响应','label-conditioned structural responses'),
  'causal_scope':('规定内部替换效应','prescribed internal-replacement effect'),
  '98_families':('98 个模型','98 model'),
 }
 for name,(a,b) in requirements.items():check(name,a in z and b in e)
 # Joint failure must precede the operational 11/12 description in abstract, result, and conclusion.
 for lang,s,am,ae,rm,re_,cm,ce in [('zh',z,'# 摘要','# 第 1 章','## 7.4','## 7.5','## 8.6','# 参考文献'),('en',e,'# Abstract','# Chapter 1','## 7.4','## 7.5','## 8.6','# References')]:
  for name,a,b in [('abstract',am,ae),('results',rm,re_),('conclusion',cm,ce)]:
   part=section(s,a,b);failure=part.find('未获支持' if lang=='zh' else 'not supported');family=part.find('11' if lang=='zh' else '11')
   if family<0 and lang=='en':family=part.find('Eleven')
   check('joint_failure_first_'+lang+'_'+name,0<=failure<family)
  front=section(s,am,ae);con=section(s,cm,ce)
  check('no_identity_or_check_count_in_front_'+lang,not re.search(r'5[ ,]000|76[ ,]800|28[ ,]800|Gradient × Input|输入乘梯度',front+con))
 # Newly generated tables: every rendered estimate must match a source row, not merely another language.
 for fn,tid in [('trend_summary.csv','6-4'),('phase_endpoint_contrasts.csv','6-5')]:
  content=gettable(z,tid)
  estimates={f"{float(r['median']):.4f} [{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}]" for r in rows(D/fn)}
  found=re.findall(r'-?\d+\.\d{4} \[-?\d+\.\d{4}, -?\d+\.\d{4}\]',content)
  check('source_all_intervals_'+tid,bool(found) and all(v in estimates for v in found))
 target=json.loads((PUB/'evidence/supplement_v1/target_analysis.json').read_text())['summary']
 for r in target:
  if r['subset']=='all' and r['target'] in ('gt','top1') and r['quantity']=='late_minus_early' and r['metric'] in ('routing_scale','value_error','attention_identity_scale','mlp_identity_scale'):
   v=f"{r['mean']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]";check('target_mean_CI_'+str((r['model'],r['target'],r['metric'])),v in gettable(z,'6-6'))
 # Complete frozen amplitude and QKV/sampling table numbers.
 for fn,tid,fields in [('multi_amplitude_summary.csv','7-1',['median']),('multi_amplitude_summary.csv','7-2',['median']),('qkv_depth_contrasts.csv','6-2',['median_difference','ci95_low','ci95_high']),('independent_sampling_summary.csv','6-1',['median','ci95_low','ci95_high'])]:
  rr=rows(PUB/'evidence/revision_v3_step2'/fn);allowed={format(float(r[f]),fmt) for r in rr for f in fields for fmt in ['.4f','.6f']}
  found=re.findall(r'[-−]?\d+\.\d{4,6}(?!\d)',gettable(z,tid));check('source_table_values_'+tid,all(v.replace('−','-') in allowed for v in found),str([v for v in found if v.replace('−','-') not in allowed]))
 fam=json.loads((R/'reports/nns_vit_endpoint_e6_release_r3_20260918/family_decisions.json').read_text())
 check('frozen_joint_status',fam['aggregate_claims']=={'C4a':'NOT_SUPPORTED','C4b':'NOT_SUPPORTED'})
 check('eleven_operational_families',sum(x['status']=='SUPPORTED' for x in fam['families'] if x['claim']=='C4a')==11)
 spatial=rows(PUB/'evidence/spatial_image_values.csv');vals=[]
 for m in ['vit_b16','deit_s16']:
  for refid in ['normalized_zero','blur_rgb']:
   for method in ['endpoint_component','regional_taylor','empirical_oracle']:
    a=[float(r['auc']) for r in spatial if r['model']==m and r['reference']==refid and r['method']==method]
    if a:vals.append(f'{statistics.median(a):.4f}')
 corrected=rows(PUB/'evidence/supplement_v1/baseline_image_values.csv')
 for m in ['vit_b16','deit_s16']:
  for refid in ['normalized_zero','blur_rgb']:
   vals.append(f"{statistics.median(float(r['random_corrected']) for r in corrected if r['model']==m and r['reference']==refid):.4f}")
 check('regional_table_values',all(v in gettable(z,'7-3') for v in vals))
 manifest=json.loads((P/'figure_manifest.json').read_text())
 check('figure_manifest_hashes',all(sha(P/r['path'])==r['sha256'] for r in manifest['figures']))
 data=json.loads((P/'qc/revision_v4_data_validation.json').read_text())
 check('data_audit_pass_current',data['status']=='PASS' and data['families']==98 and data['validator_sha256']==sha(P/'scripts/validate_revision_v4_data.py') and all(sha(D/n)==h for n,h in data['outputs'].items()))
 page=json.loads((P/'qc/page_audit.json').read_text());render=json.loads((P/'qc/render_receipt.json').read_text())
 check('pdf_current_and_audited',page['status']=='PASS' and page['pdf_sha256']==sha(P/'NNS_ViT_学位论文_匿名内容审阅版.pdf') and page['source_sha256']==sha(P/'zh/manuscript.md') and render['fallback_count']==0)
 check('reference_metadata_four_papers',len(json.loads((P/'notes/revision_v4_reference_checks.json').read_text()))>=4)
 for name in ['README.md','notes/claim_evidence_matrix.md','notes/revision_v4_plan.md','qc/completion_audit.md','qc/defensive_writing_audit.md','qc/page_review.md']:
  check('deliverable_'+name,(P/name).is_file() and 'v4' in read(P/name).lower())
 errors=[x for x in CHECKS if not x['passed']]
 out=dict(status='PASS' if not errors else 'FAIL',objective='执行修改意见v4最低送审路线',source_opinion_sha256=sha(op),check_count=len(CHECKS),checks=CHECKS,errors=errors,manuscript_sha256={'zh':sha(P/'zh/manuscript.md'),'en':sha(P/'en/manuscript.md')})
 (P/'qc/validation_report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=out['status'],checks=len(CHECKS),errors=errors),ensure_ascii=False,indent=2));return bool(errors)
if __name__=='__main__':raise SystemExit(main())
