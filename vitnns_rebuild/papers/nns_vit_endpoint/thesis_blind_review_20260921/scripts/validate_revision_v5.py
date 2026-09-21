#!/usr/bin/env python3
"""Fail-closed v5 prompt-to-artifact and manuscript validator."""
import csv, hashlib, json, re
from pathlib import Path

P=Path(__file__).resolve().parents[1]
R=P.parents[2]
E=P/'evidence/revision_v5'
RUN=R/'reports/nns_vit_revision_v5_r1_20260921'
CHECKS=[]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(name,ok,detail=''):CHECKS.append({'name':name,'passed':bool(ok),'detail':str(detail)})
def read(p):return p.read_text()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def table(text,number,english=False):
    prefix='Table' if english else '表'
    m=re.search(r'^\*\*'+prefix+' '+re.escape(number)+r'[^\n]*\*\*\n\n((?:\|[^\n]*\n)+)',text,re.M)
    return m.group(1) if m else ''
def numbers(value):return re.findall(r'(?<![A-Za-z])[-−]?\d+(?:\.\d+)?(?:e[-+]?\d+)?',value.replace('−','-'),re.I)

def main():
    z=read(P/'zh/manuscript.md');e=read(P/'en/manuscript.md');op=R/'修改意见/修改意见v5.md'
    check('v5_source',op.is_file())
    check('title_zh',z.startswith('# 视觉 Transformer 的神经元级与结构路径可解释性研究'))
    check('title_en',e.startswith('# Neuron-Level and Structural-Path Interpretability in Vision Transformers'))
    hz=re.findall(r'^## (\d+(?:\.\d+)*) ',z,re.M);he=re.findall(r'^## (\d+(?:\.\d+)*) ',e,re.M)
    check('bilingual_heading_structure',hz==he,(hz,he) if hz!=he else '')
    check('four_rqs_zh',set(re.findall(r'RQ([1-4])',z))==set('1234'))
    check('four_rqs_en',set(re.findall(r'RQ([1-4])',e))==set('1234'))
    check('math_exact_bilingual',re.findall(r'\$\$(.*?)\$\$',z,re.S)==re.findall(r'\$\$(.*?)\$\$',e,re.S))
    check('table_ids_bilingual',re.findall(r'^\*\*表 (\d+-\d+)',z,re.M)==re.findall(r'^\*\*Table (\d+-\d+)',e,re.M))
    for lang,text in [('zh',z),('en',e)]:
        figures=re.findall(r'!\[[^]]*\]\(([^)]+)\)',text)
        check('figure_count_'+lang,len(figures)==12,len(figures))
        check('figure_links_'+lang,all((P/lang/x).resolve().is_file() for x in figures))
        check('no_placeholders_'+lang,not re.search(r'\[(?:EVIDENCE|PARAMETER|REFERENCE) REQUIRED\]',text))
        check('anonymous_'+lang,not re.search(r'/data01|yeziqi|ByteDance|字节跳动',text,re.I))
        check('balanced_math_'+lang,text.count('$$')%2==0)

    # Every mandatory writing correction in v5 section 8.1.
    sentinels={
      'W01_four_rq':('四个研究问题','Four Research Questions'),
      'W02_nns_equations':('原式（10）','Original Eq. (10)'),
      'W03_fixed_cotangent':('不再对 $g_c$ 求导','without differentiating $g_c$'),
      'W04_raghu':('Raghu 等[42]','Raghu et al. [42]'),
      'W04_method_rows':('| EAP |','| EAP |'),
      'W05_taylor_identity':('计算不需要生成路径图','requires no path map'),
      'W06_activation_gradient':('激活乘梯度不使用参照状态','Activation × gradient uses no reference state'),
      'W07_region_formula':('x^{(R)}=(1-M_R)','x^{(R)}=(1-M_R)'),
      'W08_intervention_scope':('规定的混合内部状态','prescribed hybrid internal states'),
      'W09_vector_invariance':('J_{z\',x}^{\\mathsf T}g_{z\'}','J_{z\',x}^{\\mathsf T}g_{z\'}'),
      'W10_head_aggregation':('先合并各头带符号图','combines signed head maps'),
      'W11_sample_scope':('990 类仍有合格图像','990 classes retain eligible images'),
      'W12_random_mc':('随机均值 MC 标准误','MC SE of random mean'),
      'W13_no_logit_percent':('原目标逻辑值可加共同常数','common logit offset'),
      'W14_preprocess':('31×31 高斯核','31×31 Gaussian kernel'),
      'W15_main_aux_separated':('辅助诊断：参照位移感知泰勒量','Auxiliary Diagnosis: Reference-Displacement Taylor'),
      'W16_signed_cases':('Q/K/V 抵消率为 0.5380','Q/K/V cancellation 0.5380'),
      'X01_e1':('E1-R2 的 16 个模型—层—单元族区间均跨零','All 16 E1-R2 model–block–family intervals cross zero'),
      'X02_e2':('结构性零值，不是缺失数据','structural zero, not missing data'),
      'X03_e3':('路径评分没有稳定优于激活乘梯度','Path score has no stable advantage over activation × gradient'),
      'X04_e4':('结构特征没有提供稳定独立测试增益','Structural features provide no stable held-out gain'),
      'X05_e5':('H1–H3 在四个检查点','H1–H3 retain the frozen positive direction'),
      'X06_target_control':('行为目标选择性未获支持','Behavioral target selectivity is not supported'),
    }
    for name,(a,b) in sentinels.items():check(name,a in z and b in e)
    check('S02_original_fail_preserved',all(x in z and x in e for x in ['C4a','C4b']) and '未获支持' in z and 'not supported' in e)
    check('no_old_three_rq_heading','三个研究问题' not in z and 'Three Research Questions' not in e)
    check('no_logit_percentage_inference','AUC is small relative to the original target-logit scale' not in e)
    refs=json.loads((P/'notes/revision_v5_reference_checks.json').read_text());dou=next(r for r in refs['records'] if r['id']=='dou_nns_published_pdf');excerpt=P/dou['portable_excerpt_path'];excerpt_text=excerpt.read_text() if excerpt.is_file() else '';check('published_nns_equations_checked',dou['source_pdf_sha256']=='33e323f8cb664864b12daad1f06b6ef0d3268bcafaea4e84ac695fd286f3855b' and dou['page']==5 and 'c_pq^(l-1,l) = S(x_q^l) w_pq^l' in excerpt_text and 'C_q^(0,l) = W_q^l' in excerpt_text)

    # Frozen task and compact evidence coverage.
    package=json.loads((E/'package_receipt.json').read_text())
    check('compact_package_status',package['status']=='PASS_COMPACT_EVIDENCE_PACKAGE')
    expected_counts={'e1':{'expected':1000,'valid':1000,'failures':0},'e2':{'expected':200,'valid':200,'failures':0},'e3':{'expected':200,'valid':200,'failures':0},'e5':{'expected':1600,'valid':1600,'failures':0},'e1_r2':{'expected':1000,'valid':1000,'failures':0}}
    check('task_counts',package['counts']==expected_counts,package['counts'])
    check('compact_hashes',all((E/x['path']).is_file() and sha(E/x['path'])==x['sha256'] and (E/x['path']).stat().st_size==x['bytes'] for x in package['files']))
    protocol=json.loads((E/'protocol.json').read_text());cohort=json.loads((E/'cohorts.json').read_text());targets=json.loads((E/'targets.json').read_text())
    check('protocol_bindings',sha(E/'cohorts.json')==protocol['cohorts_sha256'] and sha(E/'targets.json')==protocol['targets_sha256'])
    crows=cohort['rows'];check('cohort_counts',sum(r['cohort']=='e1' for r in crows)==500 and sum(r['cohort']=='e5' for r in crows)==400 and len({r['sample_id'] for r in crows})==900)
    check('four_models',len(protocol['models'])==4 and sum(m['role']=='original' for m in protocol['models'])==2)
    check('target_blocks',all(set(map(int,t.keys()))==set(range(12)) for t in targets.values()))
    check('numeric_controls',package['numeric_checks']['e2']['closure_checks_total']==14400 and package['numeric_checks']['e5']['closure_checks_total']==25600 and package['numeric_checks']['e3']['clean_suffix_max']==0)

    # Source-derived numerical sentinels, not manuscript-to-manuscript copies.
    e1=rows(E/'e1_summary.csv');check('e1_all_primary_opposite',sum(r['metric']=='same_minus_matched' and r['decision']=='OPPOSITE_DIRECTION' for r in e1)==16)
    e1meta=rows(E/'e1_metadata_summary.csv');check('e1_metadata_all_not_supported',len(e1meta)==16 and all(r['decision']=='NOT_SUPPORTED' for r in e1meta))
    e1r2=rows(E/'e1_r2_summary.csv');check('e1_r2_prospective_all_not_supported',len(e1r2)==16 and all(r['decision']=='NOT_SUPPORTED' for r in e1r2))
    e1r2cohort=json.loads((E/'e1_r2_cohorts.json').read_text());base_ids={r['sample_id'] for r in crows};r2_ids={r['sample_id'] for r in e1r2cohort['rows']};check('e1_r2_fresh_cohort',len(r2_ids)==500 and not r2_ids.intersection(base_ids))
    e1r2match=json.loads((E/'e1_r2_matching.json').read_text());check('e1_r2_metadata_matching',len(e1r2match['rows'])==100 and e1r2match['selection_uses_model_outputs'] is False)
    check('e1_r2_values_in_text',all(f"{float(r['median']):.4f}" in table(z,'6-7') for r in e1r2 if int(r['block']) in (0,11)))
    e5=rows(E/'e5_summary.csv');check('e5_decisions',sum(r['hypothesis'] in {'H1','H2','H3'} and r['decision']=='SUPPORTED_DIRECTION' for r in e5)==12 and sum(r['hypothesis']=='H4' and r['decision']=='OPPOSITE_DIRECTION' for r in e5)==4)
    e4=rows(E/'e4_paired.csv');check('e4_total_comparisons',sum(r['baseline']=='total' for r in e4)==4 and sum(r['baseline']=='total' and r['decision']=='OPPOSITE_DIRECTION' for r in e4)==2)
    for tid,name in [('6-7','e1_summary.csv'),('6-8','e2_summary.csv'),('7-6','e3_paired_summary.csv'),('8-1','e4_models.csv'),('8-2','e5_summary.csv')]:
        check('table_present_'+tid,bool(table(z,tid)) and bool(table(e,tid,True)))
    check('e5_values_in_text',all(f"{float(r['median']):.4f}" in table(z,'8-2') for r in e5))
    e4m=rows(E/'e4_models.csv');check('e4_values_in_text',all(f"{float(r['test_mae']):.4f}" in table(z,'8-1') for r in e4m))
    cases=json.loads((P/'figures/revision_v5/case_receipt.json').read_text());check('signed_case_closure',max(x['closure_max'] for x in cases['cases'])<2e-7)
    fig=json.loads((P/'figures/revision_v5/results_figure_receipt.json').read_text());check('result_figure_hashes',all(sha(P/'figures/revision_v5'/n)==h for n,h in fig['outputs'].items()))
    page=json.loads((P/'qc/page_audit.json').read_text());render=json.loads((P/'qc/render_receipt.json').read_text())
    pdf=P/'NNS_ViT_学位论文_匿名内容审阅版.pdf'
    check('pdf_current',page['status']=='PASS' and page['pdf_sha256']==sha(pdf) and page['source_sha256']==sha(P/'zh/manuscript.md') and render['fallback_count']==0)

    errors=[x for x in CHECKS if not x['passed']]
    result={'status':'PASS' if not errors else 'FAIL','objective':'执行修改意见v5并完成论文与补充实验','source_opinion_sha256':sha(op),'check_count':len(CHECKS),'errors':errors,'checks':CHECKS,'manuscript_sha256':{'zh':sha(P/'zh/manuscript.md'),'en':sha(P/'en/manuscript.md')},'pdf_sha256':sha(pdf)}
    (P/'qc/revision_v5_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (P/'qc/validation_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':len(CHECKS),'errors':errors},ensure_ascii=False,indent=2));return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
