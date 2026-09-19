#!/usr/bin/env python3
"""Bind source-generated tables and ten figure captions to both manuscripts."""
import csv
import gzip
import json
from pathlib import Path
import re

REPO=Path(__file__).resolve().parents[3]
PAPER=REPO/'papers/nns_vit_endpoint'
ROOT=REPO/'reports/nns_vit_endpoint_paper_revision_v1_20260920'
MODEL={'vit_b16':'ViT-B/16','deit_s16':'DeiT-S/16'}
REF={'normalized_zero':'normalized-zero','blur_rgb':'blur'}
def load(p):return json.loads(p.read_text())
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def fmt(v):return 'NA' if v is None else f'{v:.4f}'
def quant(d):return f"{fmt(d['median'])} [{fmt(d['q10'])}, {fmt(d['q90'])}]"

CAPTIONS={
1:('当前输入的 ground-truth class logit 与两种互补学习图像。Signed 图对 RGB 求和，absolute 图先逐通道取绝对值再求和；正负项都保留在分析对象中。输入为冻结展示样本 class 297（懒熊）。',
   'The observed input, its ground-truth class logit, and two complementary learning images. Signed maps sum RGB channels; absolute maps sum channelwise absolute values. Both signs remain in the analysis. The frozen display sample is class 297 (sloth bear).'),
2:('异构标量目录与 attention/residual differential cuts。Head edge 位于 output projection 之前，cotangent 包含该投影；residual identity 使用输出边 cotangent。',
   'Heterogeneous scalar targets and attention/residual differential cuts. Head edges precede output projection, whose derivative is included in the cotangent; residual identity uses the output-edge cotangent.'),
3:('五类兼容网络中的原始 NNS 数值等价，以及结构闭合、负例与独立方向检查。左图每点为对应网络/精度的最大 relative-L2 error；验证项计数不是独立统计样本量。',
   'Original-NNS numerical equivalence across five compatible network families, structural closure, negative controls, and independent direction checks. Each left-panel point is the maximum relative-L2 error for that network/precision; validation counts are not independent statistical sample sizes.'),
4:('冻结分层 token-channel 的 entropy、locality、distance 层深轨迹。蓝色/橙色为 ViT-B/16、DeiT-S/16；阴影为 100 张图的图像 bootstrap 95% CI。',
   'Depth trajectories of entropy, locality, and distance for the frozen stratified token-channel sample. Blue/orange denote ViT-B/16 and DeiT-S/16; bands are image-bootstrap 95% CIs over 100 images.'),
5:('上排为 CLS-minus-patch entropy 与 net/absolute mass 的总体轨迹。下排为冻结指标选出的高 map-cancellation 单元；signed 和 absolute 视图并列，原始 RGB 质量与范数标于图下。',
   'Population trajectories of CLS-minus-patch entropy and net/absolute mass appear above a high-map-cancellation unit selected by frozen metrics. Signed and absolute views are paired, with original RGB masses and norms annotated below.'),
6:('400 图的 routing scale、Value-only Endpoint error 与 JVP discrepancy；阴影为图像 bootstrap 95% CI。下方同一 head 的 routing/value/full maps 在 signed 行及 absolute 行内分别共用尺度，含原始质量、范数和 signed sum。案例只说明分解，不代替总体证据。',
   'Routing scale, Value-only Endpoint error, and JVP discrepancy over 400 images; bands are image-bootstrap 95% CIs. Routing/value/full maps of one head share a scale within the signed row and within the absolute row, with original mass, norm, and signed-sum annotations. The example illustrates decomposition rather than replacing population evidence.'),
7:('Attention/MLP identity scale 与 output–identity patch TV；下方高 path-cancellation 案例的 identity/branch/output 共用组内尺度。TV 曲线为冻结 cohort 描述；late-minus-early 推断来自单独的 post-hoc sensitivity。',
   'Attention/MLP identity scales and output–identity patch TV, above shared-scale identity/branch/output maps from a high-path-cancellation case. TV curves describe the frozen cohort; late-minus-early inference comes from a separate post-hoc sensitivity analysis.'),
8:('按冻结指标选出的同图跨模型深度比较。每模型早晚层共用 signed/absolute 各自尺度；所选 channels 独立注册，不构成跨层或跨模型语义匹配。',
   'A shared-image cross-model depth comparison selected by frozen metrics. Early/late maps share signed and absolute scales within each model. Independently registered channels do not establish semantic matching across layers or models.'),
9:('左图为原 C4a family 点估计及 95% 图像 bootstrap CI，虚线 0.10 是点估计门槛，方形为未通过 family。右图为补充冻结同层 head 排序的 Endpoint/Taylor Spearman；其检验不替代原 C4a。',
   'Original C4a family estimates and image-bootstrap 95% CIs. The dashed 0.10 line is a point-estimate threshold; a square marks the unsupported family. The right panel reports Endpoint/Taylor Spearman for the supplementary frozen within-block head comparison, which does not replace C4a.'),
10:('四个注册组合的 absolute-effect AUC 中位数，包含实际评价候选内的 oracle。下方同一懒熊图像、DeiT-S block 5 head 0 与 normalized-zero reference 的 signed/absolute Endpoint 和 regional Taylor。Taylor 的 16 格表示区域位移预测分数，不能与像素 Endpoint 色值比较。',
    'Median absolute-effect AUC in four registered settings, including an oracle over evaluated candidates. Below are signed/absolute Endpoint and regional Taylor for the same sloth-bear image, DeiT-S block-5 head 0, and normalized-zero reference. Taylor cells represent 16 region-displacement scores and are not comparable to pixel Endpoint color values.')}

def build_tables():
    analysis=load(ROOT/'analysis.json');families=load(REPO/'reports/nns_vit_endpoint_e6_release_r3_20260918/family_decisions.json')['families']
    tables={}
    tables[1]=table(['Model / setting','ViT-B/16','DeiT-S/16'],[
        ['Blocks / heads / hidden dimension','12 / 12 / 768','12 / 6 / 384'],
        ['Registered scalar targets/image','457','385'],['Evaluation images','400','400 (same IDs)'],
        ['Resize / crop','248 / 224','248 / 224'],['Normalization mean','[0.5, 0.5, 0.5]','[0.485, 0.456, 0.406]'],
        ['Normalization std','[0.5, 0.5, 0.5]','[0.229, 0.224, 0.225]']])
    rr=[]
    for r in families:
        if r['claim']!='C4a':continue
        t=r['test_results'][0]
        rr.append([MODEL[r['model_id']],REF[r['reference_id']],r['component_family'].replace('_','/'),fmt(t['estimate']),
         f"[{fmt(t['ci_lower'])}, {fmt(t['ci_upper'])}]",f"{t['holm_adjusted_p_value']:.2e}",r['status'],'yes' if t['ci_lower']>.1 else 'no'])
    tables[2]=table(['Model','Reference','Family','Median rho','95% CI','Holm p','Frozen status','CI lower > 0.10'],rr)
    summ=json.load(gzip.open(REPO/'reports/nns_vit_endpoint_m1_r1_20260918/summary.json.gz','rt'))
    idx={(r['model'],r['reference'],r['family'],r['stratum'],r['predictor'],r['metric']):r['median'] for r in summ}
    rr=[]
    for m in MODEL:
        for ref in REF:
            rr.append([MODEL[m],REF[ref]]+[fmt(idx[m,ref,'head','within_block_mean',p,k]) for k in ('spearman','capture_at_1','capture_at_3','regret_at_1') for p in ('endpoint','taylor')])
    tables[3]=table(['Model','Reference','E rho','T rho','E Capture@1','T Capture@1','E Capture@3','T Capture@3','E Regret@1','T Regret@1'],rr)
    rr=[]
    for r in families:
        if r['claim']!='C4b':continue
        vals={t['baseline']:t['estimate'] for t in r['test_results']}
        rr.append([MODEL[r['model_id']],REF[r['reference_id']]]+[fmt(vals[k]) for k in ('total_gxi','raw_gradient','regional_taylor','random_regions')]+[r['status']])
    tables[4]=table(['Model','Reference','vs total G×I','vs gradient','vs Taylor','vs random','Frozen status'],rr)
    idx={(r['model'],r['reference'],r['method']):r for r in analysis['spatial']['summary']}
    rr=[];regret=[]
    for m in MODEL:
        for ref in REF:
            methods=('endpoint_component','regional_taylor','random_regions','empirical_oracle')
            rr.append([MODEL[m],REF[ref]]+[fmt(idx[m,ref,k]['auc']['median']) for k in methods]+[fmt(idx[m,ref,'endpoint_component']['clean_logit']['median'])])
            regret.append([MODEL[m],REF[ref]]+[fmt(idx[m,ref,k]['normalized_regret']['median']) for k in methods[:3]]+
                          [fmt(idx[m,ref,k]['relative_to_abs_clean_logit']['median']) for k in methods[:2]])
    tables[5]=table(['Model','Reference','Endpoint AUC','Taylor AUC','Random AUC','Candidate oracle AUC','Clean GT logit'],rr)
    tables[6]=table(['Model','Reference','Endpoint regret','Taylor regret','Random regret','Endpoint AUC/abs(logit)','Taylor AUC/abs(logit)'],regret)
    rank=analysis['reference']['rank_summaries'];metrics=('spearman','kendall_tau_b','top1_agreement','top3_overlap','mean_absolute_rank_displacement')
    tables[7]=table(['Model','Quantity','Spearman','Kendall tau-b','Top-1 agreement','Top-3 overlap','Mean abs rank displacement'],
        [[MODEL[r['model']],r['quantity']]+[fmt(r[k]['median']) for k in metrics] for r in rank])
    return tables,analysis

def main():
    manifest=load(PAPER/'figure_manifest.json');tables,analysis=build_tables()
    titles_zh=['模型与输入设置','原成分 family 与阈值敏感性','补充冻结同层 head 排序：E=Endpoint，T=Taylor','原空间干预配对 AUC 差','绝对 AUC 与输出尺度（每项 400 图）','归一化 oracle regret 和逐图 logit 尺度比','跨 reference head 排名一致性（先图内平均 12 层）']
    titles_en=['Models and input settings','Original component families and threshold sensitivity','Supplementary frozen within-block head ranking: E=Endpoint, T=Taylor','Original paired spatial-intervention AUC differences','Absolute AUC and output scale (400 images per setting)','Normalized oracle regret and within-image logit-scale ratios','Cross-reference head-rank concordance (12 blocks averaged within image)']
    coverage=load(REPO/'reports/nns_vit_endpoint_m1_r1_20260918/registry_coverage.json')
    s1=table(['Model','Object','Candidate definition','Candidates','Registered/image','Coverage','Strategy','Scalar neuron'],
        [[r['model'],r['object'],r['population_formula'],f"{r['population']:,}",r['registered'],f"{100*r['coverage']:.5f}%",r['strategy'],r['scalar_neuron']] for r in coverage])
    s2=table(['Resource / validation','Value'],[['Formal scan wall time','1,752.71 s'],['Four-card GPU-hours upper bound','1.9475'],['Throughput','192.16 targets/s'],['Retained data','9.290 GB'],['Peak reserved GPU memory','2.284 GB'],['Peak CPU RSS','2.018 GB'],['M1 random permutations','128,000,000'],['M4 attention / residual / JVP records','86,400 / 19,200 / 691,200']])
    s3=table(['Model','Quantity','Component pairs','Above tolerance','Abs change median [q10,q90]','Relative change median [q10,q90]'],
        [[MODEL[r['model']],r['quantity'],r['component_count'],f"{r['changed_fraction']:.1%}",quant(r['absolute_change']),quant(r['relative_change'])] for r in analysis['reference']['change_summaries']])
    s4=table(['Model','Quantity','Metric','Image median [q10,q90]','n / NA'],
        [[MODEL[r['model']],r['quantity'],k,quant(v),f"{v['n']} / {v['na']}"] for r in analysis['reference']['rank_summaries'] for k,v in r.items() if isinstance(v,dict)])
    s5=table(['Method','Primary object','External reference','Propagation / contract'],[
        ['Endpoint','Current-input local dependence','No; normalization origin matters','True local Jacobian; valid cuts close'],
        ['Activation-space Taylor','Specified replacement effect','Yes','Clean cotangent × displacement'],
        ['Integrated Gradients','Baseline-to-input attribution','Yes','Path-average gradient; baseline-difference completeness'],
        ['Attention rollout','Token flow','No','Accumulated attention / residual flow'],
        ['Grad-CAM','Class-conditioned coarse localization','No','Gradient-weighted feature maps'],
        ['Chefer / AttnLRP','Transformer relevance','Implementation-specific','Method-specific relevance rules'],
        ['Raw attention','Token allocation','No','Forward attention weights']])
    for lang,titles,ci in [('zh',titles_zh,0),('en',titles_en,1)]:
        path=PAPER/lang/'full_draft.md';text=path.read_text()
        for number,t in tables.items():
            block=f'<!-- TABLE_{number}_START -->\n**Table {number}. {titles[number-1]}。**\n\n{t}\n<!-- TABLE_{number}_END -->'
            if lang=='en':block=block.replace('。**','.**')
            token='{{TABLE:'+str(number)+'}}'
            if token in text:text=text.replace(token,block)
            else:text=re.sub(rf'<!-- TABLE_{number}_START -->.*?<!-- TABLE_{number}_END -->',lambda _:block,text,flags=re.S)
        for f in manifest['figures']:
            number=f['figure'];block=f'<!-- FIGURE_{number}_START -->\n![Figure {number}](../../../{f["png"]["path"]})\n\n**Figure {number}.** {CAPTIONS[number][ci]}\n<!-- FIGURE_{number}_END -->'
            token='{{FIGURE:'+str(number)+'}}'
            if token in text:text=text.replace(token,block)
            else:text=re.sub(rf'<!-- FIGURE_{number}_START -->.*?<!-- FIGURE_{number}_END -->',lambda _:block,text,flags=re.S)
        path.write_text(text)
        old=(PAPER/lang/'full_draft_r3_archive.md').read_text()
        # Keep detailed M4 tables, diagnostics and reference table provenance in a clearly historical section.
        detail=old.split('<!-- M4_MECHANISM_START -->')[1].split('<!-- M4_MECHANISM_END -->')[0]
        detail=re.sub(r'Figure (\d+)',r'Archived R2 Figure \1',detail)
        intro=('# 补充材料：Endpoint 修订 v1\n\n[返回正文](full_draft.md)。本文件包含复现参数、完整统计、历史图索引与后续实验范围。主文 RQ 编号按修订版使用。\n\n' if lang=='zh' else '# Supplement: Endpoint revision v1\n\n[Return to manuscript](full_draft.md). This file contains reproducibility parameters, complete statistics, archived figure navigation, and prospective experiment scope. RQ labels in the current manuscript follow the revision.\n\n')
        body=intro+'## S1. Model and registry details\n\n'
        body+='Models: '+chr(96)+'vit_base_patch16_224.augreg2_in21k_ft_in1k'+chr(96)+' and '+chr(96)+'deit_small_patch16_224.fb_in1k'+chr(96)+'. Torch 2.2.2+cu121; timm 0.9.16; torchvision 0.17.2+cu121; NumPy 1.26.4; Pillow 10.2.0.\n\n**Table S1. Original scalar registry coverage.**\n\n'+s1+'\n\n**Table S2. Resources and extended validation coverage.**\n\n'+s2
        body+='\n\n## S2. Frozen decisions and numerical diagnostics\n\nC4a: BROAD_BUT_NON_UNIVERSAL_FAMILY_LEVEL_SUPPORT; original all-family omnibus NOT_SUPPORTED. C4b: NOT_SUPPORTED. The numerical atlas status is PASS_FLOAT64_RECOVERY_WITH_RETAINED_FLOAT32_FAILURE: 36/57,600 float32 failures are retained and 36/36 same-coordinate float64 checks pass. These fields are not interchangeable.\n\n'
        body+='## S3. Reference sensitivity\n\nChange uses atol=1e-12 and rtol=0. Relative change is abs(q0-qblur)/max(abs(q0),abs(qblur)); both-zero denominators are NA. Component quantiles describe the pooled records, not independent inferential samples.\n\n**Table S3. Changes in Taylor magnitude and absolute finite effect.**\n\n'+s3
        body+='\n\nHead rankings use average ranks and Kendall tau-b. Top-K ties are broken by ascending component ID; Top-3 overlap is intersection size / 3. Rank displacement uses original rank positions, so model head counts affect its scale. Each image averages 12 blocks, then summaries use 400 images. Full-order equality counts are Taylor 0/4,800 and effect 0/4,800 for ViT-B, Taylor 46/4,800 and effect 64/4,800 for DeiT-S.\n\n**Table S4. Image-level rank concordance distributions.**\n\n'+s4
        body+='\n\n## S4. Method objects\n\n**Table S5. Functional comparison.**\n\n'+s5
        body+='\n\n## S5. Descriptive absolute AUC and oracle\n\n[Raw recomputation](../../../reports/nns_vit_endpoint_paper_revision_v1_20260920/analysis.json) and [image values](../../../reports/nns_vit_endpoint_paper_revision_v1_20260920/spatial_image_values.csv). The empirical oracle integrates a budgetwise upper envelope of evaluated candidates. Zero oracle or clean-logit denominators are NA; all displayed families have 400 defined image ratios. All five original per-image method scores match the frozen E6 release to less than 1e-12. The random baseline averages signed curves before taking absolute values. Supplementary M2 uses a separate task with blocks 2/5/10 and native patch or legacy grids; its results are not substituted for the original C4b task.\n\n'
        body+='## S6. Detailed attention/residual evidence (historical stage identifiers retained)\n\n'+detail
        body+='\n\n## S7. Archived figure index and display protocol\n\nSigned displays use sum_c E_c; absolute displays use sum_c abs(E_c). Quantitative mass, L2, and signed sums use original normalized-input RGB arrays. The new figures share robust P99 scales within comparison groups and apply asinh with linear width scale/9. Clipped pixel counts and raw limits are in [display_receipt.json](../../../reports/nns_vit_endpoint_paper_revision_v1_20260920/display_receipt.json). Old positive-only and independently normalized displays remain archival spatial-shape views.\n\n'
        for i,r in enumerate(manifest['supplemental_records'],1):
            outputs=r.get('outputs',[r]);links=[]
            for rec in outputs:
                links.append(f'[{Path(rec["path"]).suffix[1:].upper()}](../../../{rec["path"]})')
            body+=f'- S{i:02d} · {r["group"]} · '+Path(outputs[0]['path']).stem+' · '+' / '.join(links)+'\n'
        body+='\n## S8. Prospective experiments and reproducibility index\n\n[Revision protocol and prioritized follow-up experiments](../notes/revision_v1_protocol.md). The suggested random/untrained control, GT/clean-top1 mechanism robustness, matched token-channel bridge, original-NNS CNN display, alternative checkpoint, and optional functional visualization comparison are not executed in this manuscript/presentation revision. Current limitations explicitly retain these gaps.\n\n[Evidence matrix](../notes/revision_v1_evidence_matrix.md), [figure manifest](../figure_manifest.json), [writing audit](../qc/revision_v1_writing_audit.md), and [completion audit](../qc/revision_v1_completion_audit.md).\n'
        (PAPER/lang/'supplement_revision_v1.md').write_text(body)
    (ROOT/'captions_zh.md').write_text('# 修订主图图注\n\n'+'\n\n'.join(f'**Figure {i}.** {v[0]}' for i,v in CAPTIONS.items())+'\n')
    (ROOT/'captions_en.md').write_text('# Revised main-figure captions\n\n'+'\n\n'.join(f'**Figure {i}.** {v[1]}' for i,v in CAPTIONS.items())+'\n')
    print('Bound 7 tables, 10 figures, and 92 archived supplemental records in both languages.')

if __name__=='__main__':main()
