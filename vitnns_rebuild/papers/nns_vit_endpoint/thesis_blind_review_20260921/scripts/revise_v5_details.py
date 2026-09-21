#!/usr/bin/env python3
from pathlib import Path
import csv,json,re,sys
P=Path(__file__).resolve().parents[1];R=P.parents[2];sys.path.insert(0,str(R/'src'))
from vitnns.endpoint_m1_statistics import bootstrap_median
import numpy as np

def table(h,rs):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rs)
def ins(s,n,t):return re.sub(r'(^## '+re.escape(n)+r' [^\n]*\n)',lambda m:m[0]+'\n'+t+'\n\n',s,count=1,flags=re.M)
def fmt(a,key,n=4):
 a=np.asarray(a,float);a=a[np.isfinite(a)];ci=bootstrap_median(a,key);return f'{np.median(a):.{n}f} [{ci[0]:.{n}f}, {ci[1]:.{n}f}]'
pub=R/'papers/nns_vit_endpoint/publication_r2_20260920/evidence/supplement_v1';rr=list(csv.DictReader((pub/'baseline_image_values.csv').open()))
regional=[];differences=[];mc=[]
for model in ['vit_b16','deit_s16']:
 for ref in ['normalized_zero','blur_rgb']:
  part=[r for r in rr if r['model']==model and r['reference']==ref];a=lambda key:np.asarray([float(r[key]) for r in part])
  regional.append([model,ref]+[fmt(a(k),model+ref+k) for k in ['endpoint','taylor','random_corrected','oracle']])
  differences.append([model,ref,fmt(a('endpoint')-a('taylor'),model+ref+'taylor'),fmt(a('endpoint_minus_random'),model+ref+'random',6)])
  mc.append([model,ref]+[f'{np.median(a(k)):.6f}' for k in ['random_mc_se','split_half_difference','loo_max_deviation']])
for lang in ['zh','en']:
 p=P/lang/'manuscript.md';s=p.read_text();zh=lang=='zh'
 if 'V5_DETAILS_APPLIED' in s:continue
 alg=table(['步骤','操作'] if zh else ['Step','Operation'],[[1,'给定固定模型、输入 $x$、标量 $F$ 和合法切面 $C$' if zh else 'Given a fixed model, input $x$, scalar $F$, and legal cut $C$'],[2,'一次前向缓存实际使用端口张量 $z_c$' if zh else 'One forward caches actual-use edge tensors $z_c$'],[3,'反向取得 $g_c=\partial F/\partial z_c$，固定余切种子' if zh else 'Obtain $g_c=\partial F/\partial z_c$ and hold it fixed'],[4,'逐边 $E_c=x\odot\operatorname{VJP}(z_c,x;g_c)$，不再对 $g_c$ 求导' if zh else 'For each edge $E_c=x\odot\operatorname{VJP}(z_c,x;g_c)$ without differentiating $g_c$'],[5,'输出图组、切面重构误差和每个分项统计' if zh else 'Return maps, cut-reconstruction error, and component statistics']])
 alg=('**算法 4-1 同目标结构路径分解**' if zh else '**Algorithm 4-1 Structural decomposition for one target**')+'\n\n'+alg+'\n'+('设前向和反向成本为 $T_f,T_b$，缓存为 $A$，输入维数为 $d$。直接算法约需 $T_f+(1+|C|)T_b$，保存全部图需 $O(A+|C|d)$，流式统计需 $O(A+d)$。批量 VJP 提高并行效率，仍计算不同分项导数。' if zh else 'With forward/backward costs $T_f,T_b$, activation cache $A$, and input dimension $d$, direct computation costs approximately $T_f+(1+|C|)T_b$. Storing all maps needs $O(A+|C|d)$ memory; streaming summaries need $O(A+d)$. Batched VJPs parallelize component derivatives.')
 s=ins(s,'4.8',alg)
 # Explicit source equation numbers without silently resolving the terminal-gate ambiguity.
 if zh:
  s=s.replace('| 输入拆分 |','| 输入拆分，原式（1）–（3）、（12） |').replace('| 线性或卷积传播 |','| 线性或卷积传播，原式（4）、（5）、（11）、（13） |').replace('| ReLU 门 |','| ReLU 门，原式（6）、（9）、（10） |').replace('| 最大池化 |','| 最大池化，原第 3.3 节 |')
  explanation='原式（10）把目标层激活门乘入单层贡献；原式（11）末端写成权重行与之前各层贡献矩阵的乘积。若目标是激活前量，末端不再加门；若目标是激活后量，需包括该目标门。本文将两类标量分开注册，并验证自己的局部雅可比定义。输入坐标、神经元扫描与残差比较均承接原 NNS；增量在 ViT 异构目标、真实使用端口、结构分项和经验评价。'
 else:
  s=s.replace('| Input splitting |','| Input splitting, original Eqs. (1)–(3), (12) |').replace('| Linear or convolutional propagation |','| Linear/convolutional propagation, original Eqs. (4), (5), (11), (13) |').replace('| ReLU gate |','| ReLU gate, original Eqs. (6), (9), (10) |').replace('| Max pooling |','| Max pooling, original Sec. 3.3 |')
  explanation='Original Eq. (10) includes the destination activation gate in a one-layer contribution, while Eq. (11) writes the final weight row times earlier contribution matrices. A pre-activation target has no final gate; a post-activation target includes it. We register those targets separately and validate our local-Jacobian definition. Input coordinates, neuron scanning, and residual comparison build on NNS; the increment concerns ViT target organization, actual-use ports, structural terms, and empirical evaluation.'
 s=ins(s,'3.1',explanation)
 if zh:
  lit='Raghu 等[42]已在 ViT 中观察到有效感受野在中层趋于全局，并分析残差连接及中心图像块依赖。因此，空间范围随深度扩大属于本文在新指标与抽样下复核的现象。本文进一步比较同一类别目标的 Q/K/V 与两类残差的阶段轨迹，将结构异质性与一般感受野扩大分开。'
 else:lit='Raghu et al. [42] observed increasingly global effective receptive fields in intermediate ViT layers and analyzed skip connections and central-patch dependence. Increasing spatial range is therefore a phenomenon replicated here under different metrics and sampling. Our further comparison concerns Q/K/V and the two residual paths under the same class target, separating structural heterogeneity from general receptive-field expansion.'
 s=ins(s,'2.2',lit)
 distinction=r'''$$
\widehat{\Delta F}_c=\langle\nabla_{z_c}F,z_c(x)-z_c(x^{\rm ref})\rangle,\qquad
E_c=\operatorname{diag}(x)J_{z_c,x}^{\mathsf T}\nabla_{z_c}F.
$$'''
 s=ins(s,'2.3',distinction+'\n\n'+('EAP 为候选边赋予位移—梯度近似分数；AtP* 在激活替换近似中处理注意力饱和和抵消问题；Vi-CD 以视觉计算边子图及有限恢复行为为对象，并比较适配到 ViT 的 EAP/EAP-IG。本文不以真实边本身作为文献空缺，研究对象是保留像素坐标的完整切面分解及其空间、尺度和干预统计。' if zh else 'EAP assigns displacement–gradient scores to candidate edges. AtP* addresses saturation and cancellation in activation-patching approximations. Vi-CD studies visual computational-edge subgraphs and finite recovery, including ViT-adapted EAP/EAP-IG comparisons. Actual computational edges are not an unoccupied research topic; the present object is an input-coordinate decomposition of complete cuts and its spatial, scale, and intervention statistics.'))
 oldline=next((l for l in s.splitlines() if l.startswith('| 边归因 / AtP* / Vi-CD') or l.startswith('| Edge attribution / AtP* / Vi-CD')),None)
 if oldline:
  new=('| EAP | 位移与局部梯度的边分数 | 否 | 是 | 是 | 一阶有限替换近似 | 是 | 是 |\n| AtP* | 处理饱和/抵消的替换近似 | 通常否 | 可定位组件/边 | 是 | 指定近似 | 是 | 是 |\n| Vi-CD | 视觉任务边回路及恢复行为 | 通常否 | 是 | 是 | 回路行为 | 是 | 是 |' if zh else '| EAP | Displacement–gradient edge score | No | Yes | Yes | First-order replacement approximation | Yes | Yes |\n| AtP* | Patching approximation addressing saturation/cancellation | Usually no | Components/edges | Yes | Specified approximation | Yes | Yes |\n| Vi-CD | Visual-task circuits and recovery behavior | Usually no | Yes | Yes | Circuit behavior | Yes | Yes |')
  s=s.replace(oldline,new)
 # Regional intervals are recomputed directly from the 1600 image-reference rows.
 h=['模型','参照','路径图','区域泰勒','修正随机','候选上界'] if zh else ['Model','Reference','Path map','Regional Taylor','Corrected random','Candidate bound']
 s=re.sub(r'(\*\*(?:表|Table) 7-3[^\n]*\*\*\n\n)((?:\|[^\n]*\n)+)',lambda m:m[1]+table(h,regional),s)
 more=('**表 7-6 区域 AUC 配对差及 95% 区间**' if zh else '**Table 7-6 Paired regional AUC differences and 95% intervals**')+'\n\n'+table(['模型','参照','路径图−泰勒','路径图−随机'] if zh else ['Model','Reference','Path map−Taylor','Path map−random'],differences)+'\n'+('ViT-B 两种参照区间均偏向随机，DeiT 两种参照区间均跨零。均为事后修正估计量的逐项精度区间。' if zh else 'Both ViT-B intervals favor random selection; both DeiT intervals cross zero. These are pointwise precision intervals for the post-hoc corrected estimand.')
 more+='\n\n'+('**表 7-7 随机 20 次重复的稳定性（图像中位数）**' if zh else '**Table 7-7 Stability of 20 random repeats (image medians)**')+'\n\n'+table(['模型','参照','随机均值 MC 标准误','分半差','留一最大差'] if zh else ['Model','Reference','MC SE of random mean','Split-half difference','Max leave-one difference'],mc)+'\n'+('单图 MC 标准误与跨图配对中位差属于不同层次，不能直接比较后宣布结果无效。原目标逻辑值可加共同常数，因此 AUC/logit 不解释为决策改变百分比。' if zh else 'Per-image MC standard errors and across-image median differences concern different levels and cannot directly invalidate one another. A common logit offset changes AUC/logit without changing the decision, so that ratio has no decision-percentage interpretation.')
 s=ins(s,'7.5',more)
 ref='[42] Raghu M, Unterthiner T, Kornblith S, Zhang C, Dosovitskiy A. Do Vision Transformers See Like Convolutional Neural Networks? NeurIPS, 2021. arXiv:2108.08810v1. https://arxiv.org/pdf/2108.08810v1. Accessed 2026-09-21.'
 s=s.replace('# 附录 A',ref+'\n\n# 附录 A') if zh else s.replace('# Appendix A',ref+'\n\n# Appendix A')
 s+='\n<!-- V5_DETAILS_APPLIED -->\n';p.write_text(s)
refs=P/'references.md'
if '[42]' not in refs.read_text():refs.write_text(refs.read_text()+'\n'+ref+'\n')
print('V5_DETAILS_WRITTEN')
