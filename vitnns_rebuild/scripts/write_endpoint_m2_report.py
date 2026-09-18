#!/usr/bin/env python3
"""Write bilingual M2 results from verified tables."""
from run_endpoint_m2 import ROOT,REPO,load

def main():
    data=load(ROOT/'report_data.json');tables=data['tables'];counts=data['patch_taylor_comparisons']
    full=counts['full'];isol=counts['isolated']
    task=load(ROOT/'task_validation.json');inf=load(ROOT/'inference_validation.json')
    figures=load(ROOT/'figure_manifest.json')['figures'];paper=REPO/'papers/nns_vit_endpoint'
    zh=f'''M2在两模型各400张图像上完成blocks 2、5、10的全部heads，比较两种reference、八种方法和四个patch预算。完整输入效应的12个model/reference/block比较中，Endpoint absolute相对regional Taylor优势通过统一Holm的有{full['endpoint_greater']}项，Taylor优势通过的有{full['baseline_greater']}项；单head隔离效应对应为{isol['endpoint_greater']}和{isol['baseline_greater']}项。逐层结果见表，未通过的方向检验不表示两方法等效。

## 1. 协议与定义

本研究为原研究之后冻结的补充实验（supplementary frozen study）。原RQ3为11/12 family广泛但非普遍的支持，原C4a/C4b omnibus保持NOT_SUPPORTED。Endpoint基元为normalized-input Gradient × Input；head图来自局部Jacobian成分传播，结构闭合与有限区域效用分别评价。原registry覆盖表沿用已封存M1；registry-complete指注册集合完整。

采用0-based编号，blocks 2/5/10对应early/middle/late；ViT-B/16每层12 heads，DeiT-S每层6 heads。reference为normalized-zero及raw-RGB blur（kernel31、sigma10）。主要预算是1、10、20、49个16×16 patch，对应面积0.51%、5.10%、10.20%、25%；原4×4网格的1/2/4格结果在完整表中单独报告。

八方法为Endpoint absolute、Endpoint signed aggregate、regional Taylor、gradient-only、activation displacement、random、center和attention。对head h，E_h=x⊙J_hᵀg_h，g_h=∇_{{z_h}}F。absolute predictor为Σ|E|；signed predictor为|ΣE|，未取绝对值的sum保留于原始记录。regional Taylor为|g_hᵀ(z_h(x)−z_h(x_R))|，与指定head的有限激活位移匹配。所有方法从同一候选集合选择区域，按行优先ID打破平局；random固定20次均匀随机排列。

主要outcome是完整输入区域替换的绝对logit变化，|F(x)−F(x_R)|。隔离outcome将替换输入产生的指定head激活放回clean计算图，其余head与bypass保持clean。每个random重复先计算绝对效应与AUC，再平均重复和同层全部heads。统计单位为图像，中位数采用10000次image bootstrap和95% CI；672项双向AUC符号检验统一Holm。逐预算CI、逐head和跨三层均值汇总均为描述性。

## 2. 跨层比较

**Table M2-1. 完整输入区域替换，patch预算归一化AUC。** 配对差值与两方法各自中位数之差不同。

{tables['full']}

**Table M2-2. 单head隔离区域替换，patch预算归一化AUC。**

{tables['isolated']}

## 3. 八方法比较

**Table M2-3. 八方法比较的方向性检验计数。** 每项分母12对应两模型×两reference×三层，计数包含相关条件。

{tables['baselines']}

完整输入替换会同时改变多条模型路径，head隔离固定所选head之外的计算；regional Taylor使用的位移对应后者。跨层结论以每个model/reference/block的paired结果为准。高分辨率成分图的表达能力与有限区域选择效用需要分别评价。

regret以实际评估候选集合最优为参照：单区域预算穷举全部196或16个候选，更大预算使用各方法与random的候选并集。该参照的范围是已评估候选，并非全部区域组合的全局最优。

## 4. 图示
'''
    en=f'''M2 evaluated all heads in blocks 2, 5, and 10 on 400 images per model, using two references, eight selection methods, and four patch budgets. Across the 12 model/reference/block comparisons for full-input effects, {full['endpoint_greater']} supported Endpoint-absolute superiority over regional Taylor and {full['baseline_greater']} supported Taylor superiority after the shared Holm correction. The corresponding counts for isolated-head effects were {isol['endpoint_greater']} and {isol['baseline_greater']}. The tables report each layer separately; a non-rejected directional test does not establish equivalence.

## 1. Protocol and definitions

This supplementary frozen study was specified after the primary study. Original RQ3 showed broad but non-universal family-level support (11/12); original C4a/C4b omnibus decisions remain NOT_SUPPORTED. The base Endpoint is normalized-input Gradient × Input. Head maps propagate local Jacobian components, and structural closure and finite-region utility are evaluated separately. The sealed M1 registry coverage table remains applicable; registry-complete denotes completion of the registered set.

Blocks use zero-based indices: 2/5/10 represent early/middle/late layers. ViT-B/16 has 12 heads per block and DeiT-S has six. References are normalized zero and raw-RGB blur (kernel 31, sigma 10). The primary budgets replace 1, 10, 20, or 49 patches of 16×16 pixels: 0.51%, 5.10%, 10.20%, and 25% of the input. Separate complete tables retain the original 4×4 region grid at 1/2/4 regions.

The eight methods are Endpoint absolute, Endpoint signed aggregate, regional Taylor, gradient-only, activation displacement, random, center, and attention. For head h, E_h=x⊙J_hᵀg_h, where g_h=∇_{{z_h}}F. The absolute predictor is Σ|E|; the signed predictor ranks |ΣE|, with raw signed sums retained. Regional Taylor is |g_hᵀ(z_h(x)−z_h(x_R))|, matching the selected head's finite activation displacement. Methods select from the same candidate pool, resolving ties by row-major region ID; random uses 20 fixed uniform permutations.

The primary outcome is the absolute logit change from full-input regional replacement, |F(x)−F(x_R)|. The isolated outcome inserts the selected head activation induced by that input into the clean computation, retaining other heads and bypasses. Each random repetition yields an absolute effect and AUC before averaging repetitions and heads within an image. Medians use 10,000 image-bootstrap draws and 95% intervals; one Holm family covers 672 bidirectional AUC sign tests. Per-budget intervals, head summaries, and means across three blocks are descriptive.

## 2. Cross-layer comparisons

**Table M2-1. Full-input regional replacement: normalized AUC over patch budgets.** A paired median differs from the difference between two marginal medians.

{tables['full']}

**Table M2-2. Isolated-head regional replacement: normalized AUC over patch budgets.**

{tables['isolated']}

## 3. Eight-method comparison

**Table M2-3. Directional test counts across the eight methods.** Each denominator is 12: two models × two references × three blocks. Counts include correlated conditions.

{tables['baselines']}

Full-input replacement changes multiple model paths, whereas head isolation fixes the remaining computation at its clean values; regional Taylor uses the displacement associated with the latter. Layer conclusions follow the model/reference/block paired contrasts. A component map's spatial resolution and finite-region selection utility are distinct properties.

Regret uses the best evaluated candidate: all 196 or 16 singleton regions are evaluated at the one-region budget, and larger budgets use the union of selections from all methods and random repetitions. This reference covers evaluated candidates rather than a global optimum over all region combinations.

## 4. Figures
'''
    captions={
        'zh':[
            'Image #1：完整输入effect–budget曲线；每panel固定model/reference/block，纵轴为图像级中位绝对logit效应，阴影为95% image-bootstrap CI。',
            'Image #2：相同布局的单head隔离曲线；先在图像内平均heads，再跨图像计算中位数与CI。',
            'Image #3：逐head的Endpoint与Taylor中位AUC之差，描述层级异质性；它与图像级配对中位差不同，色标按panel分别设置。'],
        'en':[
            'Image #1: Full-input effect–budget curves. Each panel fixes model, reference, and block; the ordinate is the image-level median absolute logit effect with 95% image-bootstrap intervals.',
            'Image #2: Isolated-head curves with the same layout. Heads are averaged within each image before medians and intervals are computed across images.',
            'Image #3: Differences between head-specific median Endpoint and Taylor AUCs describe head heterogeneity. These differ from paired-median contrasts; each panel has its own color scale.']}
    for lang,body in [('zh',zh),('en',en)]:
        title='# M2：跨层全head空间效用补充结果' if lang=='zh' else '# M2: Supplementary cross-layer, all-head spatial utility'
        figtext='\n\n'.join(f"![M2 Image #{i+1}](../../../{f['path']})\n\n{captions[lang][i]}" for i,f in enumerate(figures))
        (paper/lang/'endpoint_m2_supplement.md').write_text(title+'\n\n'+body+'\n'+figtext+
            '\n\nSources: [Evidence matrix](../notes/endpoint_m2_evidence_matrix.md); [Figure manifest](../figure_manifest_m2.json).\n')
    figtext='\n\n'.join(f"![M2 Image #{i+1}](../{f['path']})\n\n{captions['zh'][i]}" for i,f in enumerate(figures))
    details=f'''

## 5. 执行、验证与完整交付

完成{task['task_count']}任务，{task['curves']:,}条曲线，{task['points']:,}个有限预算点；所有区域selection、候选logit、signed/absolute effect、candidate best、regret与AUC均从原始记录复算。累计worker时间为{task['gpu_hours']:.3f} GPU-hours。完整汇总包括{inf['summaries']:,}行summary、{inf['paired']:,}行paired和{inf['head_summaries']:,}行逐head结果。

[summary.csv](../reports/nns_vit_endpoint_m2_r1_20260919/summary.csv)包含每预算中位效应和regret；[paired.csv](../reports/nns_vit_endpoint_m2_r1_20260919/paired.csv)中的baseline=random给出随机提升；[head_summary.csv](../reports/nns_vit_endpoint_m2_r1_20260919/head_summary.csv)保留所有heads；[auc_tests.json](../reports/nns_vit_endpoint_m2_r1_20260919/auc_tests.json)包含672项检验。results/attempts/保存原始候选与曲线，artifact_manifest.json绑定其SHA256；原始大体积记录保留于本机结果根，Git发布代码、报告、统计表、图像及审计清单。

实施协议见[冻结协议](endpoint_m2_execution_20260919.md)，额外来源与聚合核验见[验收补充](endpoint_m2_acceptance_addendum_20260919.md)。M1封存主稿保持可重验，M2双语补充稿见[中文](../papers/nns_vit_endpoint/zh/endpoint_m2_supplement.md)与[英文](../papers/nns_vit_endpoint/en/endpoint_m2_supplement.md)。本轮止于M2。
'''
    (REPO/'docs/endpoint_m2_results_20260919.md').write_text('# Endpoint M2 结果与完成审计\n\n'+zh+'\n'+figtext+details)
    evidence='''# M2 evidence matrix

| Claim or output | Evidence | Boundary |
|---|---|---|
| All-head cross-layer execution | task_manifest.json, task_validation.json | 400 images/model, blocks 2/5/10 |
| Endpoint/Taylor direction by layer | auc_tests.json, paired.csv, Tables M2-1/2 | Shared 672-test Holm family; no equivalence inference |
| Eight methods and budget curves | summary.csv, results/attempts/, M2_F1/M2_F2 | Image-unit medians; absolute effects |
| Head heterogeneity | head_summary.csv, M2_F3 | Descriptive difference of medians |
| Candidate regret | raw candidate tables, summary.csv | Singleton exhaustive; larger budgets evaluated union |
| Primary C4a/C4b remain NOT_SUPPORTED | Bound original E6 terminal; report_data.json | Supplementary results remain separate |

Result files are relative to reports/nns_vit_endpoint_m2_r1_20260919/. Figure references resolve through figure_manifest_m2.json. Raw large result records remain in the local result root and are SHA256-bound in artifact_manifest.json; Git includes the code, aggregate tables, figures, and audit chain.
'''
    (paper/'notes/endpoint_m2_evidence_matrix.md').write_text(evidence)
    (ROOT/'README.md').write_text('''# Endpoint M2 supplementary result package

Read docs/endpoint_m2_results_20260919.md and completion_audit.json.
2400 image-model-block tasks; 21600 image-model-block-head conditions.
summary.csv and paired.csv contain all budgets and two geometries/outcomes.
head_summary.csv preserves every head; auc_tests.json contains 672 Holm-adjusted tests.
results/attempts/ contains the raw finite candidate logits, selector scores and curves locally.
artifact_manifest.json binds the raw records by SHA256. Git publishes code, aggregate tables,
figures and audit receipts; the large raw records and queue database remain local.
The original C4a/C4b omnibus stays NOT_SUPPORTED. Stop after M2.
''')
    print('M2_REPORT_WRITTEN')

if __name__=='__main__':main()
