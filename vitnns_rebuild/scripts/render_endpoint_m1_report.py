#!/usr/bin/env python3
"""Render audited M1 tables, figure and synchronized bilingual supplement."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from run_endpoint_m1_analysis import ROOT,load
summaries=load(ROOT/'summary.json.gz'); pairs=load(ROOT/'paired_comparisons.json.gz')
tests=load(ROOT/'paired_spearman_tests.json')['tests']
idx={(r['model'],r['reference'],r['family'],r['stratum'],r['predictor'],r['metric']):r for r in summaries}
pidx={(r['model'],r['reference'],r['family'],r['stratum'],r['baseline'],r['metric']):r for r in pairs}
tidx={(r['model'],r['reference'],r['family'],r['stratum'],r['baseline']):r for r in tests}
models=('vit_b16','deit_s16'); refs=('normalized_zero','blur_rgb')
def get(m,r,f,s,p,k):return idx[m,r,f,s,p,k]
def fmt(x):return 'NA' if x is None else f'{x:.4f}'
for name,data in [('all_summary.csv',summaries),('all_paired_comparisons.csv',pairs)]:
    with (ROOT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
for ax,(m,r) in zip(axes.flat,[(m,r) for m in models for r in refs]):
    for p,c in [('endpoint','#146b8b'),('taylor','#cb5a27'),('random','#888888')]:
        rows=[get(m,r,'head',f'block_{b:02d}',p,'spearman') for b in range(12)]
        y=[v['median'] for v in rows];lo=[v['ci95'][0] for v in rows];hi=[v['ci95'][1] for v in rows]
        ax.plot(range(12),y,'o-',label=p,color=c,ms=3);ax.fill_between(range(12),lo,hi,alpha=.15,color=c)
    ax.set_title(m+' / '+r);ax.axhline(0,color='#999999',lw=.7);ax.grid(alpha=.2);ax.set_xticks(range(12))
axes[0,0].legend(frameon=False);fig.supxlabel('Block (0-based)');fig.supylabel('Median head Spearman (95% image bootstrap CI)')
fig.tight_layout();fig.savefig(ROOT/'m1_head_ranking.png',dpi=180);fig.savefig(ROOT/'m1_head_ranking.pdf');plt.close(fig)
head=['| Model | Reference | Endpoint rho | Taylor rho | Endpoint Capture@1 | Taylor Capture@1 | Endpoint Capture@3 | Taylor Capture@3 | Endpoint Regret@1 | Taylor Regret@1 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for m in models:
 for r in refs:
    vals=[get(m,r,'head','within_block_mean',p,k)['median'] for k in ('spearman','capture_at_1','capture_at_3','regret_at_1') for p in ('endpoint','taylor')]
    head.append('| '+m+' | '+r+' | '+' | '.join(map(fmt,vals))+' |')
head='\n'.join(head)
identity=['| Model | Reference | Endpoint-Taylor rho | 95% CI | Holm p | Capture@1 difference |','|---|---|---:|---|---:|---:|']
for m in models:
 for r in refs:
    d=pidx[m,r,'residual_identity','across_block','taylor','spearman'];t=tidx[m,r,'residual_identity','across_block','taylor'];cap=pidx[m,r,'residual_identity','across_block','taylor','capture_at_1']
    identity.append(f"| {m} | {r} | {d['median']:.4f} | [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] | {t['holm_p']:.3g} | {cap['median']:.4f} |")
family=['| Model | Reference | Family | Stratum | Endpoint rho | Taylor rho | Endpoint-Taylor rho | Holm p |','|---|---|---|---|---:|---:|---:|---:|']
for m in models:
 for r in refs:
  for f in ('head','routing','value','Q','K','V','residual_identity','residual_branch'):
   for s in ('across_block','early','middle','late'):
    a=get(m,r,f,s,'endpoint','spearman');b=get(m,r,f,s,'taylor','spearman');d=pidx[m,r,f,s,'taylor','spearman'];t=tidx[m,r,f,s,'taylor']
    family.append(f"| {m} | {r} | {f} | {s} | {fmt(a['median'])} | {fmt(b['median'])} | {fmt(d['median'])} | {t['holm_p']:.3g} |")
baselines=['| Baseline | Valid tests | Endpoint superiority supported | Not supported | Undefined small-set tests |','|---|---:|---:|---:|---:|']
for p in ('activation_norm','gradient_norm','activation_gradient','displacement_norm','taylor','random'):
    rows=[t for t in tests if t['baseline']==p]
    baselines.append(f"| {p} | {sum(t['valid'] for t in rows)} | {sum(t['decision']=='SUPPORTED' for t in rows)} | {sum(t['decision']=='NOT_SUPPORTED' for t in rows)} | {sum(not t['valid'] for t in rows)} |")
report='''# Endpoint M1 结果与完成审计

M1为原确认性研究之后的**补充冻结实验（supplementary frozen study）**。原C4a all-family omnibus保持NOT_SUPPORTED，11/12 family满足全部family-level条件，结论为广泛但非普遍的成分级关联。

## 1. 执行范围与来源

两模型共享400张图像，0-based blocks 0–11、ground-truth logit、normalized-zero与blur。ViT-B/16每层12 heads，DeiT-S每层6 heads。最终每图每reference分别有240/168个不同成分，合计326,400个成分—reference条件。复用E4已有255,744个条件，补算70,656个条件；两条residual skip edge全400图，Q/K补齐后336图。

每个集合有七种predictor：Endpoint absolute mass、activation norm、gradient norm、absolute activation×gradient、reference displacement norm、absolute activation-space Taylor、random。outcome统一为绝对logit替换效应。Endpoint mean absolute mass为m/150528；固定输入维度下排序相同。signed effect与signed sum保存在逐成分记录，不混入主要绝对效应统计。

128,000个原始比较集合，每个1,000次固定种子随机排序，共128,000,000次permutation；加上图像内block平均得到134,400个图像级集合。保留随机均值、标准差、分位数和20-bin Spearman直方图。

## 2. RQ3a绝对关联与RQ3b基线增量

同层head ranking中，Endpoint具有正向关联；Taylor的中位Spearman、Capture@1/@3均更高，Regret@1更低。下表先在每图内平均12层指标，再取400图中位数。

'''+head+'''

![M1 head ranking](../reports/nns_vit_endpoint_m1_r1_20260918/m1_head_ranking.png)

Image #1：每panel固定一个模型/reference。横轴block，纵轴图像级head Spearman中位数，阴影为95% image-bootstrap CI。同层排序与跨层结果分开报告。

## 3. Residual identity例外及决策指标

跨层residual_identity family包含12层24条skip edges。Endpoint-minus-Taylor Spearman配对中位差在四种组合均为正且通过统一Holm；Capture@1配对中位差均为0。这支持该成分集合上的排序关联增量，不能推广为Top-1决策增益或所有成分的优势。

'''+ '\n'.join(identity)+'''

按§10.3核对：reference沿用E4预处理；Taylor梯度位于clean点，位移为z_clean-z_ref；skip替换保留clean sibling branch。生产中每图检查noop和E4五类已有成分的前向/基线parity。另对两模型固定evaluation order 0、blocks 0/5/11执行float64多epsilon中心差分与手写skip反事实；诊断标为post-hoc implementation diagnostic。跨层尺度及下游非线性可能影响排序，但本轮未将其确认为机制原因。

## 4. 六类基线完整比较

'''+ '\n'.join(baselines)+'''

计数包含相关层级比较和value/V命名别名，不是独立实验成功率。全部24,696条统计汇总见结果根all_summary.csv，14,112条配对差值和95% CI见all_paired_comparisons.csv；相对random的Capture差值即随机提升。Regret使用baseline−Endpoint，使正值均表示Endpoint更好。

## 5. 统计合同

图像为最终单位；中位数、10,000次图像bootstrap、95% percentile CI。中位数bootstrap使用经验CDF与联合中心次序统计量等价采样，并通过普通image-index bootstrap独立测试。pooled先平均同一image ID两模型指标，仅作描述性补充。

2,016项Endpoint-baseline Spearman检验使用单侧exact sign test，完整集合统一Holm。672项绝对检验分别检验median rho>0与>0.10，另做统一Holm。QKV三项、residual两项不定义Spearman，对应INVALID_UNDEFINED_METRIC；其pairwise/top1有效。undefined不表示采集失败，不删除其他有效指标中的图像。所有有效主要比较保留400图；early/middle/late单边family只有四项，Spearman仍离散，需结合pairwise/NDCG解释。CI为逐比较区间，Holm用于注册Spearman检验。

## 6. 分family与层级结果

value与V均指同一V projection edge，不能当作独立证据。routing为softmax attention概率张量，Q/K/V为projection张量；residual_branch合并attention与MLP。每一行的所有基线使用相同成分集合。

'''+ '\n'.join(family)+'''

## 7. 覆盖率与论文

原registry覆盖表从EP0实际条目重算，中英文主稿Table 6同步发布。post-block覆盖率192/(12×197×768)=0.01058%与192/(12×197×384)=0.02115%；complete仅指registry-complete。head scalar是CLS指定channel；module aggregates不是单神经元；不同注册条目可能选中同一logit。M3的6,144目标审计未在本轮执行。

## 8. 复现与验收

配置：config/nns_vit_endpoint/endpoint_m1.json；需求映射：docs/endpoint_m1_execution_20260918.md。task_manifest.json冻结800任务和E3/E4字节哈希；freeze.json与analysis_freeze.json冻结实现；per_image/*.json.gz保留成分数据、指标、随机种子和分布。task_validation.json及inference_validation.json记录源复算，completion_audit.json和terminal_receipt.json记录最终验收。

复算命令（已存在的PASS审计文件会只核对、不覆盖）：

    CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python scripts/validate_endpoint_m1.py --tasks --inference
    CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python papers/nns_vit_endpoint/scripts/validate_endpoint_scanner_draft.py

补算800/800任务，8张H20用时43.26秒，任务GPU时间0.07898 GPU-hours，不包含历史E3/E4成本。最大noop误差1.62125e-5，已有E4前向/基线最大绝对差6.62804e-5，均满足1e-4+1e-4|source|。新增科学结果和execution完成状态分开记录；完成后停在M1。
'''
absolute=load(ROOT/'absolute_spearman_tests.json')['tests']
absolute_lines=['| Model | Reference | Absolute null threshold | Holm p | Decision |',
                '|---|---|---:|---:|---|']
for r in absolute:
    if r['family']=='head' and r['stratum']=='within_block_mean':
        absolute_lines.append(f"| {r['model']} | {r['reference']} | {r['threshold']:.2f} | {r['holm_p']:.3g} | {r['decision']} |")
report=report.replace('## 3. Residual identity例外及决策指标',
    '同层head图像内平均的绝对关联检验如下；大于0与大于0.10是分别计算的假设。\n\n'+
    '\n'.join(absolute_lines)+'\n\n## 3. Residual identity例外及决策指标')
(REPO/'docs/endpoint_m1_results_20260918.md').write_text(report)
for lang in ('zh','en'):
    p=REPO/f'papers/nns_vit_endpoint/{lang}/full_draft.md';s=p.read_text();start='<!-- M1_SUPPLEMENT_START -->';end='<!-- M1_SUPPLEMENT_END -->'
    if start in s:
        a=s.index(start);b=s.index(end,a)+len(end);s=s[:a]+s[b:]
    if lang=='zh':
        section='''### 5.6 M1：强基线与分层排序（补充冻结实验）

M1在原研究之后冻结，采用同一400图、两模型、12层、两种reference和ground-truth logit。
七种predictor共享成分定义：Endpoint mass、activation norm、gradient norm、absolute
activation×gradient、displacement norm、absolute activation-space Taylor和random。
head、routing、value/V、Q、K及两类residual均覆盖400图，value/V为同一edge别名。
每集合1,000次random permutation，共128,000,000次；image bootstrap为10,000次、95% CI。
主Spearman配对检验统一Holm，median rho>0与>0.10分别检验。

**Table 7. M1同层head排序。** 每图先平均12层指标，再对400图取中位数。

'''+head+'''

Taylor在四个组合中均取得更高Spearman和Capture、更低Regret。跨层residual identity的
Endpoint−Taylor Spearman配对中位差为0.0865–0.4209，均通过统一Holm；对应Capture@1
差值中位数均为0。增量价值依赖成分与指标。QKV三项和residual两项使用pairwise/top1，
Spearman记为NA。pooled先合并同图像两模型，仅为描述性补充。完整分层表、CI、随机提升
与复算来源见docs/endpoint_m1_results_20260918.md及M1结果根。原C4a保持NOT_SUPPORTED。
'''
    else:
        section='''### 5.6 M1: Predictor baselines and stratified ranking (supplementary frozen study)

M1 was frozen after the primary study, using the same 400 images, two models, 12 blocks,
two references, and ground-truth logit. Seven predictors share component definitions:
Endpoint mass, activation norm, gradient norm, absolute activation×gradient, displacement
norm, absolute activation-space Taylor, and random. Heads, routing, value/V, Q, K and both
residual families cover 400 images; value/V names the same edge. Each set uses 1,000 random
permutations, totaling 128,000,000. Image bootstrap uses 10,000 replicates and 95% CIs.
Paired Spearman tests receive joint Holm correction; median rho>0 and >0.10 are separate tests.

**Table 7. M1 within-block head ranking.** Metrics are first averaged across 12 blocks per
image and then summarized by the median over 400 images.

'''+head+'''

Taylor had higher Spearman and Capture and lower Regret in all four combinations.
Across-block residual identity showed positive Endpoint-minus-Taylor median Spearman
differences of 0.0865–0.4209, all passing joint Holm; corresponding Capture@1 median
differences were zero. Incremental utility depends on the component and metric. QKV
triplets and residual pairs use pairwise/top1 accuracy, with Spearman recorded as NA.
Pooled results first average the two models per image and are descriptive. Complete
stratified tables, CIs, random-relative gains and source recomputation are in
docs/endpoint_m1_results_20260918.md and the M1 result root. Original C4a remains NOT_SUPPORTED.
'''
    where=s.index('## 6. ');p.write_text(s[:where]+start+'\n'+section+end+'\n\n'+s[where:])
print('M1_REPORT_RENDERED')
