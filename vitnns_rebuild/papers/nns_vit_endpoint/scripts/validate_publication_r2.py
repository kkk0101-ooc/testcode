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
    check('11_main_figure_mapping',f.get('main_text_figure_count')==11 and f.get('main_text_figure_map')==[{'main_number':i+1,'package_number':n} for i,n in enumerate((1,2,6,7,8,12,15,16,17,18,21))])
    for r in f['figures']:
        for ext,record in r['files'].items():
            p=ROOT/record['path'];check('figure_hash:'+str(r['number'])+':'+ext,p.is_file() and sha(p)==record['sha256'])
            if ext=='png':check('png_size:'+str(r['number']),struct.unpack('>II',p.read_bytes()[16:24])[0]>=1500)
            if ext=='svg':ET.parse(p)
    for path,digest in load(ROOT/'evidence/figure_sources.json').items():check('source:'+path,sha(REPO/path)==digest)
    supplement=load(ROOT/'evidence/supplement_manifest.json');check('92_supplement_pages',len(supplement)==92)
    for r in supplement:check('supplement_hash:'+str(r['number']),sha(ROOT/r['path'])==r['sha256'])
    texts={lang:(ROOT/lang/'manuscript.md').read_text() for lang in ('zh','en')}
    main_paths={row['main_number']:next(x['files']['png']['path'] for x in f['figures'] if x['number']==row['package_number']) for row in f['main_text_figure_map']}
    for lang,text in texts.items():
        check('11_main_figures:'+lang,re.findall(r'!\[Figure (\d+)\]',text)==[str(i) for i in range(1,12)])
        check('main_figure_paths:'+lang,{int(n):path for n,path in re.findall(r'!\[Figure (\d+)\]\(\.\./([^)]*)\)',text)}==main_paths)
        check('main_figure_caption_order:'+lang,re.findall(r'\*\*Figure (\d+)\.',text)==[str(i) for i in range(1,12)])
        table_numbers=re.findall(r'\*\*Table (\d+)\.',text)
        check('10_numbered_tables:'+lang,table_numbers==[str(i) for i in range(1,11)],table_numbers)
        check('no_historical_report_states:'+lang,not re.search(r'RQ4[a-c]|RQ5|NOT_SUPPORTED|PASS_|reports/|registry-complete|C4[ab]',text))
        check('balanced_math:'+lang,text.count('$$')%2==0)
        check('no_placeholders:'+lang,'{{' not in text and '\x00' not in text)
        check('four_research_questions:'+lang,all(f'RQ{i}' in text for i in range(1,5)) and 'RQ5' not in text)
        check('required_method_terms:'+lang,all(term in text for term in ('Input × Gradient','valid differential cut' if lang=='en' else '合法 differential cut','Q-induced','K-induced','V-induced','identity','branch')))
        check('corrected_random_estimator:'+lang,('Each random repetition takes the absolute effect before integration' if lang=='en' else '随机基线对每次重复先取 absolute effect 并积分') in text)
        check('v3_eight_chapters:'+lang,all(re.search(rf'^## {i}\.',text,re.M) for i in range(1,9)))
        check('v3_result_structure:'+lang,all(f'### 6.{i}' in text for i in range(1,12)) and all(f'### 7.{i}' in text for i in (1,2)) and all(f'### 8.{i}' in text for i in (1,2)))
        check('rq4_prescribed_replacement_scope:'+lang,('规定有限替换任务中的排序能力' if lang=='zh' else 'Ranking in Prescribed Finite-Replacement Tasks') in text)
        check('no_unexecuted_amplitude_claim:'+lang,'0.05,0.1,0.25,0.5,1.0' not in text)
        check('graph_relative_nonuniqueness:'+lang,all(term in text for term in (('计算图依赖与分解非唯一性','一般实现不变性',r'R^{-\top}') if lang=='zh' else ('Computational-graph dependence and non-uniqueness','implementation invariant',r'R^{-\top}'))))
        parameter_state_boundary=('该对照不识别训练过程的因果效应' in text if lang=='zh' else 'this comparison does not identify a causal effect of training' in text.lower())
        check('parameter_state_not_training_cause:'+lang,parameter_state_boundary)
        check('positioning_methods:'+lang,all(term in text for term in ('Gradient × Input','conductance','FullGrad','GlobEnc','DecompX','AttnLRP','LibraGrad','Activation patching','Edge attribution patching')))
        check('no_defensive_stock_phrases:'+lang,not re.search(r'We emphasize that|It should be noted that|We do not claim|To avoid misunderstanding|Taken together|需要强调的是|应当指出|为避免误解|我们不声称',text,re.I))
        check('gxi_step4_results:'+lang,all(term in text for term in ((
            '联合索引必要性','E5 四检查点外部复现','广泛但非普遍','0.1136–0.1862'
        ) if lang=='zh' else (
            'Joint-index necessity','E5 four-checkpoint external replication','broad but non-universal','0.1136–0.1862'
        ))))
    title_zh='# 面向视觉 Transformer 内部目标的双索引局部 Jacobian 扫描与结构审计'
    title_en='# Dual-Indexed Local Jacobian Scanning and Structural Auditing of Internal Targets in Vision Transformers'
    central_zh=r'本文以该已有量为测量原语，对每个注册目标 $s$ 定义 $\mathcal L_{e,i}(s;x)=x_i[J_{z_e,x}(x)^{\mathsf T}\bar z_e^{(s)}]_i$。'
    central_en=r'Using this established quantity as the measurement primitive, we define $\mathcal L_{e,i}(s;x)=x_i[J_{z_e,x}(x)^{\mathsf T}\bar z_e^{(s)}]_i$ for each registered target $s$.'
    check('gxi_frozen_titles',texts['zh'].startswith(title_zh) and texts['en'].startswith(title_en))
    check('gxi_abstract_central_sentences',central_zh in texts['zh'].split('## 1.')[0] and central_en in texts['en'].split('## 1.')[0])
    check('gxi_introduction_positioning','本文进一步保留两个自由索引' in texts['zh'].split('## 2.')[0] and 'We retain two free indices beyond that total map' in texts['en'].split('## 2.')[0])
    contribution_terms={
        'zh':('数学身份与对象定义','目标条件化双索引扫描与结构审计','机制观察与解释边界'),
        'en':('mathematical identity and object definition','target-conditioned dual-indexed scanning and structural auditing','mechanistic observations and an interpretation boundary'),
    }
    for lang,text in texts.items():
        intro=text.split('## 2.')[0]
        check('gxi_three_contributions:'+lang,all(term in intro for term in contribution_terms[lang]))
    check('gxi_discussion_positioning','方法增量是目标条件化的 $[e,i]$ 局部微分图组、合法切面合同和结构审计' in texts['zh'] and 'The technical increment is the target-conditioned $[e,i]$ local-differential map set, its valid-cut contract, and structural audit' in texts['en'])
    check('gxi_conclusion_positioning','两类证据分别界定方法正确性和行为用途' in texts['zh'] and 'these evidence types delimit correctness and behavioral use separately' in texts['en'])
    plan=(REPO/'Endpoint_GxI创新定位与后续五步计划.md').read_text()
    categories=('数学身份','结构方法','实现正确性','经验发现','解释边界')
    check('gxi_claim_registry',all(plan.count('| '+term+' |')>=1 for term in categories) and all(claim in plan for claim in ('MI-1','SM-1','IC-1','EF-1','IB-1')))
    check('gxi_scope_separation',all(term in plan for term in ('PCIA–PEFA 仅作为独立扩展研究或附录材料','CST aquatic-context 研究及其 `NO_GO_NOT_IDENTIFIABLE` 历史结果保持独立','第二步已完成联合能力的新颖性差分审计','第三步已完成方法章节和核心概念图重构','第四步已完成实验结果重排与封存证据绑定')))
    step2=plan.split('## 九、第二步执行记录：联合能力的新颖性差分审计',1)[1]
    source_ids=tuple(f'S{i}' for i in range(1,14))
    check('gxi_step2_source_registry',all(f'| {source_id} |' in step2 for source_id in source_ids))
    required_methods=('Gradient × Input','Integrated Gradients','Conductance','GlobEnc','ALTI','DecompX','LRP 家族','EAP','Activation patching','DAVE','Unpack','NNS–ViT Endpoint Scanner')
    check('gxi_step2_required_methods',all(method in step2 for method in required_methods))
    matrix_header='| 方法族 | 被解释对象 | 内部目标类型 | 真实 consumer edge | 合法切面 | attention routing/value | residual identity/branch | 精确闭合 | mutation 检出 | 经验用途 |'
    check('gxi_step2_matrix_fields',matrix_header in step2)
    novelty_ids=('NOV-ID-1','NOV-SYS-1','NOV-JOINT-1','NOV-ATTN-1','NOV-RES-1','NOV-AUD-1','NOV-EMP-1')
    check('gxi_step2_novelty_registry',all(step2.count(novelty_id)>=2 for novelty_id in novelty_ids))
    gap=step2.split('### 9.5 研究缺口说明（三段）',1)[1].split('### 9.6',1)[0].strip()
    check('gxi_step2_three_paragraph_gap',len([p for p in gap.split('\n\n') if p.strip()])==3)
    audit_table=step2.split('### 9.8 第二步 prompt-to-artifact 验收表',1)[1].split('### 9.9',1)[0]
    check('gxi_step2_prompt_artifact_audit',audit_table.count('| PASS |')==12)
    check('gxi_step2_paper_nearest_neighbors',all(all(term in texts[lang] for term in ('ALTI','HiLRP','GradSkip','Unpack','Vi-CD','CAAP','DAVE')) for lang in ('zh','en')))
    novelty_overclaim=re.compile(r'for the first time|\bfirst (?:method|framework|approach|study|to)\b|\bnovel(?:ty)?\b|首次|首创|不同于既有工作',re.I)
    active_only=texts['zh']+texts['en']
    check('gxi_step2_no_unlinked_priority_claims',not novelty_overclaim.search(active_only))
    check('gxi_no_extension_evidence_in_main',not re.search(r'PCIA|PEFA|aquatic-context|NNS-CST',active_only,re.I))
    prohibited=re.compile(r'Endpoint.{0,24}(新基础归因|新的基础归因|普遍优于)|Endpoint.{0,40}(new base attribution|novel base attribution|universally superior)',re.I)
    check('gxi_no_prohibited_claims',not prohibited.search(active_only))
    step3_required={
        'zh':('## 4. ViT 内部目标的双索引局部微分','Target registry 与双索引对象','producer $p_e$','consumer $c_e$','consumer input slot $q_e$',r'invocation $\nu_e$','use-specific cotangent','在同一观测输入 $x$ 和同一前向状态求值','合法 differential cut','方法消融表：索引必要性与对象退化','审计协议表：验证层、直接证据与封存边界'),
        'en':('## 4. Dual-Indexed Local Differentials for Internal ViT Targets','Target registry and the dual-indexed object','producer','consumer input slot','invocation','use-specific cotangent','same observed input $x$ and forward state','valid differential cut','Method ablation table: index necessity and object degeneration','Audit protocol table: validation layers, direct evidence, and sealed boundaries'),
    }
    for lang,text in texts.items():
        check('gxi_step3_method_contract:'+lang,all(term in text for term in step3_required[lang]))
        check('gxi_step3_claim_links:'+lang,all(text.count(term)>=1 for term in ('NOV-ID-1','NOV-JOINT-1','NOV-ATTN-1','NOV-RES-1','NOV-AUD-1')))
        check('gxi_step3_degenerations:'+lang,all(term in text for term in ('no $i$','no $e$','no registered $s$','no valid-cut contract')))
        check('gxi_step3_audit_boundary:'+lang,all(term in text for term in ('9,600 omit-head','9,600 omit-routing','9,600 duplicate-identity','384 attention-detach','wrong-slot','merged fan-out')))
    expected_definition='\\mathcal L_{e,i}(s;x)\n=x_i\\left[J_{z_e,x}(x)^{\\mathsf T}\\bar z_e^{(s)}\\right]_i.'
    expected_closure='\\sum_{e\\in C}\\mathcal L_{e,i}(s;x)\n=x_i\\frac{\\partial s(x)}{\\partial x_i}.'
    check('gxi_step3_bilingual_definition',all(expected_definition in texts[lang] for lang in ('zh','en')))
    check('gxi_step3_bilingual_closure',all(expected_closure in texts[lang] for lang in ('zh','en')))
    check('gxi_step3_figure_contract',all(all(term in texts[lang] for term in ('consumer-use edge','Figure 1','Figure 2')) for lang in ('zh','en')))
    check('bilingual_main_figure_paths',[path for _,path in re.findall(r'!\[Figure (\d+)\]\(\.\./([^)]*)\)',texts['zh'])]==[path for _,path in re.findall(r'!\[Figure (\d+)\]\(\.\./([^)]*)\)',texts['en'])])
    step4=load(ROOT/'qc/gxi_step4_validation.json')
    check('gxi_step4_evidence_audit',step4['status']=='PASS' and not step4['errors'] and step4['e5_audit']['decision']=='SYNC_FOUR_CHECKPOINT_E5' and step4['new_experiments']==0)
    refs=(ROOT/'references.md').read_text()
    check('bibliography_complete',all(name in refs for name in ('Achtibat, R., Hatefi','Buratti, C., Marchetti','Chen, P.-K., Plaat','Ferrando, J., Gállego','Izadi, A., Banayeeanzade','Mehri, F., Soleymani Baghshah','Nishankar, S., Sanjeewani','Shrikumar, A., Greenside','Srinivas, S., & Fleuret','Modarressi, A., Fayyaz','Syed, A., Rager','Wróbel, A., Gairola','Zhang, F., & Nanda','Żukowska, N., Stammer')) and 'TO VERIFY' not in refs)
    claims=REPO/'papers/nns_vit_endpoint/notes/revision_v3_frozen_claims.md'
    check('v3_frozen_claims',claims.is_file() and all(term in claims.read_text() for term in ('K1','K2','K3','C1 数学身份','C7 区域排序','RQ4 冻结合同')))
    check('same_display_equations',re.findall(r'\$\$(.*?)\$\$',texts['zh'],re.S)==re.findall(r'\$\$(.*?)\$\$',texts['en'],re.S))
    check('same_section_numbers',re.findall(r'^#{2,3} (\d+(?:\.\d+)?)',texts['zh'],re.M)==re.findall(r'^#{2,3} (\d+(?:\.\d+)?)',texts['en'],re.M))
    check('same_table_data',[l for l in texts['zh'].splitlines() if l.startswith('|')]==[l for l in texts['en'].splitlines() if l.startswith('|')])
    paper=ROOT.parent
    active=load(paper/'figure_manifest.json')
    check('active_binding',active['figure_count']==22 and active['source_revision']=='PUBLICATION-R2-STEP3')
    for source,bound in zip(f['figures'],active['figures']):
        check('active_figure_number:'+str(source['number']),source['number']==bound['figure'])
        for extension in ('png','pdf','svg'):
            check('active_figure_hash:'+str(source['number'])+':'+extension,source['files'][extension]['sha256']==bound[extension]['sha256'])
    for lang,text in texts.items():
        expected=text.replace('](../figures/','](../publication_r2_20260920/figures/').replace('](../references.md)','](../publication_r2_20260920/references.md)').replace('](supplement.md)',f'](../publication_r2_20260920/{lang}/supplement.md)').replace('](supplement_experiments_v1.md)',f'](../publication_r2_20260920/{lang}/supplement_experiments_v1.md)')
        check('current_entry_mirror:'+lang,(paper/lang/'full_draft.md').read_text()==expected)
    for p in [ROOT/lang/name for lang in ('zh','en') for name in ('manuscript.md','supplement.md','supplement_experiments_v1.md')]:
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
        check('step3_visual_review',q.get('status')=='PASS_VISUAL_REVIEW_STEP3' and q.get('step3_concept_figures')==[1,2])
    # The original inventory binds publication R2 before the v1 supplemental
    # experiments.  Preserve and verify every historical entry except the
    # documents prospectively superseded by this supplement.
    inventory=load(ROOT/'artifact_inventory.json')
    superseded={'README.md','references.md','figure_manifest.json','en/manuscript.md','en/supplement.md','zh/manuscript.md','zh/supplement.md',
                'figures/png/fig_01_concept_overview.png','figures/pdf/fig_01_concept_overview.pdf','figures/svg/fig_01_concept_overview.svg',
                'figures/png/fig_02_neuron_learning_image.png','figures/pdf/fig_02_neuron_learning_image.pdf','figures/svg/fig_02_neuron_learning_image.svg',
                'gallery.html','figure_analysis.md','evidence/display_receipt.json','qc/visual_review.json','qc/completion_audit.md','qc/writing_audit.md'}
    for record in inventory['files']:
        if record['path'] not in superseded:
            check('inventory_hash:'+record['path'],sha(ROOT/record['path'])==record['sha256'])
    current=load(ROOT/'evidence/supplement_v1/manifest.json')
    check('supplement_v1_manifest_scope',len(current['files'])>=40)
    for record in current['files']:
        check('supplement_v1_hash:'+record['path'],sha(ROOT/record['path'])==record['sha256'])
    errors=[r for r in checks if not r['passed']]
    report=dict(status='PASS' if not errors else 'FAIL',check_count=len(checks),errors=errors,real_panels_recomputed=reconstructed,checks=checks)
    (ROOT/'qc/validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+chr(10))
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2))
    return bool(errors)

if __name__=='__main__':raise SystemExit(main())
