#!/usr/bin/env python3
"""Integrate frozen E1-E5 evidence into both thesis languages."""
from pathlib import Path
import csv, hashlib, json, re

P = Path(__file__).resolve().parents[1]
R = P.parents[2]
A = R / "reports/nns_vit_revision_v5_r1_20260921/analysis"

def rows(name):
    with (A / name).open() as f:
        return list(csv.DictReader(f))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ci(row, digits=4):
    return f"{float(row['median']):.{digits}f} [{float(row['ci95_low']):.{digits}f}, {float(row['ci95_high']):.{digits}f}]"

def table(head, body):
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in body]
    return "\n".join(lines)

def before(text, heading, content):
    marker = content.splitlines()[0]
    return text if marker in text else text.replace(heading, content + "\n\n" + heading, 1)

def front_zh():
    return r'''# 视觉 Transformer 的神经元级与结构路径可解释性研究

# 摘要

视觉 Transformer 包含标记—通道神经元、多层感知机隐藏单元、查询、键、值、注意力头和残差路径。本文研究两个相互衔接的对象：单神经元在输入坐标中的学习图像，以及按真实计算边组织同一局部输入依赖的结构路径图组。基础量是已有的输入乘梯度 $x\odot\nabla_xs(x)$；本文的技术增量来自明确的 ViT 目标注册、完整切面分解和经验评价，而不是新的基础归因公式。

对神经元或类别标量目标，学习图像在标准化 RGB 坐标中计算。对合法切面 $C$，结构分项为

$$
E_c=\operatorname{diag}(x)J_{z_c,x}^{\mathsf T}g_c,\qquad g_c=\frac{\partial F}{\partial z_c},
$$

其中边 VJP 使用固定的下游余切。方法覆盖 Q/K/V、注意力头、MLP 单元及两类残差子层；闭合、遗漏、重复、批量与标量 VJP、方向导数控制用于检验实现正确性。

经验研究将原两个检查点的证据与前瞻补充实验结合。E1 使用 100 类、每类 5 张共 500 张新图像；E2、E3 使用每类冻结一图；E5 使用互斥的 400 图及两个新增公开检查点。MLP 神经元学习图像随层深扩展，Q/K/V 和两类残差呈现不同轨迹。E1 未支持稳定类别一致性：在冻结的 JS 指标下，同类学习图像相似度低于视觉匹配异类。E3 未发现路径图相对激活乘梯度的稳定神经元干预排序优势。E4 未发现结构特征相对总图特征的稳定独立测试预测增益。独立图像和四个检查点复现了三个结构假设，而路径图的注意力头排序在四个检查点中均低于激活乘梯度。参照位移感知泰勒量仍是辅助替换预测量，计算不需要路径图。

本文的贡献是可核验的“神经元—结构路径”表示及对描述结构与干预排序的经验区分。所有负结果均保留。外推范围为所测试检查点与抽样规则，不扩大到全部视觉 Transformer 或对象语义定位。

**关键词：** 视觉 Transformer；神经元学习图像；NNS；结构路径；计算边；干预

# 第 1 章 绪论

## 1.1 研究背景与问题

视觉 Transformer 通过多头注意力、逐标记 MLP 和残差连接处理图像块[1-3]。单个标记—通道或 MLP 隐藏单元具有输入依赖的局部响应；同一响应又可以按 Q/K/V、注意力头、MLP 单元和残差路径组织。总输入图既不能保留单个神经元身份，也不能说明下游类别目标的依赖经过哪些内部路径。

本文以 NNS 神经元扫描[4]为起点。神经元级对象回答一个标量单元呈现何种输入坐标依赖；结构对象回答下游目标的依赖如何分配到真实使用边。全文证据顺序为神经元表征、结构贡献、行为干预和外部验证。

## 1.2 总体问题与四个研究问题

总体问题是：如何表征 ViT 神经元的局部输入依赖，如何将其组织为注意力、MLP 和残差结构路径，如何检验这些表示与规定干预的关系，以及哪些观察能在独立图像和检查点中保持？

1. **RQ1，神经元级表示。** 如何注册标记—通道、类别通道和 GELU 后 MLP 神经元，其输入坐标学习图像满足什么数学关系？
2. **RQ2，神经元级规律。** 不同神经元族的空间范围、类别一致性、激活选择性、冗余和层深轨迹是什么？
3. **RQ3，结构路径组成。** 同一目标的局部输入依赖如何分配到 Q/K/V、注意力头、MLP 单元和残差路径，哪些切面闭合？
4. **RQ4，行为效用与外部验证。** 神经元和路径评分相对激活、梯度、激活乘梯度、位移、泰勒及随机基线能否预测规定替换，主要观察能否在独立图像和新增检查点中复现？

## 1.3 论证顺序

神经元学习图像是注册标量 $s$ 的 $x\odot\nabla_xs(x)$。结构路径图使用相同输入坐标，并保留边索引 $c$。研究先验证这些数学对象，再测量神经元和路径的分布，随后评价有限干预，最后用冻结假设检验独立图像及新增公开参数状态。

## 1.4 主要贡献

1. 建立 ViT 神经元注册与神经元级学习图像分析，覆盖块后标记—通道单元和 GELU 后 MLP 隐藏单元，并明确定义空间、类别一致性、激活、冗余及缺失规则。
2. 将局部输入依赖组织为完整 Q/K/V、注意力头、MLP 单元和残差切面，明确真实使用端口、固定余切、闭合检验及坐标依赖边界。
3. 区分已有的感受野全局化现象与 Q/K/V、两类残差及 MLP 神经元的异质层深轨迹，补入前瞻的 MLP 单元证据。
4. 以神经元、注意力头和区域替换审计行为效用，与简单及位移感知基线比较，并在互斥图像和新增公开检查点上验证。

## 1.5 论文结构

第 2 章定位 NNS、神经元归因、ViT 感受野、边归因和激活替换；第 3 章定义神经元学习图像；第 4 章构造结构路径图；第 5 章说明样本、干预和统计；第 6 章报告神经元与结构规律；第 7 章评价神经元和组件替换；第 8 章报告独立验证、局限与结论。

## 1.6 本章小结

四个研究问题形成“神经元表征—结构贡献—行为干预—外部验证”的证据链。

'''

def front_en():
    return r'''# Neuron-Level and Structural-Path Interpretability in Vision Transformers

# Abstract

Vision Transformers contain token-channel neurons, MLP hidden units, queries, keys, values, attention heads, and residual paths. This thesis studies two connected objects: input-coordinate learning images for individual neurons, and structural path-map families that organize the same local input dependence by actual computational edges. The base quantity is the established Gradient × Input quantity $x\odot\nabla_xs(x)$; the technical increment lies in explicit ViT target registration, complete-cut decomposition, and empirical evaluation rather than a new foundational attribution rule.

For a scalar neuron or class target, learning images are computed in normalized RGB coordinates. For a legal edge cut $C$, each structural term is

$$
E_c=\operatorname{diag}(x)J_{z_c,x}^{\mathsf T}g_c,\qquad g_c=\frac{\partial F}{\partial z_c},
$$

where the downstream cotangent is held fixed during the edge VJP. The method covers Q/K/V, attention heads, MLP units, and both residual sublayers. Closure, omission, duplication, batched-VJP, scalar-VJP, and directional-derivative controls test implementation correctness.

The empirical study combines the original two-checkpoint evidence with prospective supplementary experiments. E1 uses 500 new images from 100 classes; E2 and E3 use one frozen image per class; E5 uses a disjoint 400-image set and two additional public checkpoints. MLP-neuron learning images become spatially broader across depth, and Q/K/V and the two residual sublayers show distinct profiles. E1 does not support stable category consistency: same-class learning images are less similar than visually matched different-class images under the frozen JS metric. E3 finds no stable path-map advantage over activation × gradient for neuron intervention ranking. E4 finds no stable held-out prediction gain from structural features over total-map features. On independent images and four checkpoints, three frozen structural hypotheses replicate, while path-map head ranking is consistently below activation × gradient. Reference-displacement Taylor remains an auxiliary replacement predictor and does not require path maps.

The resulting contribution is a verifiable neuron-to-path representation and an empirical separation between descriptive structure and intervention ranking. Negative utility results are retained. Generalization concerns the tested checkpoints and sampling rules, not all Vision Transformers or semantic object localization.

**Keywords:** Vision Transformer; neuron learning image; NNS; structural path; computational edge; intervention

# Chapter 1 Introduction

## 1.1 Background and Problem

Vision Transformers process image patches through multi-head attention, token-wise MLPs, and residual connections [1-3]. An individual token-channel or MLP hidden unit has an input-dependent local response. The same response can be organized by Q/K/V, attention-head, MLP-unit, and residual-path identities. A total input map preserves neither the identity of an individual neuron nor the internal routes through which a downstream class target depends on the input.

This thesis starts from NNS neuron scanning [4]. The neuron-level object asks what input-coordinate dependence one scalar unit exhibits. The structural object asks how a downstream target's dependence is allocated among actual-use edges. The evidence is organized as representation, structural contribution, intervention, and external validation.

## 1.2 Overall Question and Four Research Questions

The overall question is how to represent local input dependence of ViT neurons, organize it into attention, MLP, and residual paths, test its relation to prescribed interventions, and determine which observations persist on independent images and checkpoints.

1. **RQ1, neuron-level representation.** How are token-channel, class-channel, and post-GELU MLP neurons registered, and what mathematical relations do their input-coordinate learning images satisfy?
2. **RQ2, neuron-level regularities.** How do spatial extent, category consistency, activation selectivity, redundancy, and depth profiles differ across neuron families?
3. **RQ3, structural-path composition.** How is one target's local dependence partitioned among Q/K/V, heads, MLP units, and residual paths, and which cuts close?
4. **RQ4, behavioral utility and external validation.** How well do neuron and path scores predict prescribed replacement effects relative to activation, gradient, activation × gradient, displacement, Taylor, and random baselines, and which main observations replicate?

## 1.3 Argument and Evidence Order

The neuron learning image is $x\odot\nabla_xs(x)$ for a registered scalar $s$. Structural maps use the same input coordinates but retain the edge index $c$. The study first validates these mathematical objects, then measures neuron and path distributions, evaluates finite intervention, and finally applies frozen tests to independent images and additional public parameter states.

## 1.4 Contributions

1. A ViT neuron registry and neuron-level learning-image analysis covering post-block token-channel units and post-GELU MLP hidden units, with explicit spatial, category-consistency, activation, redundancy, and missing-value definitions.
2. A structural-path organization of local input dependence over complete Q/K/V, head, MLP-unit, and residual cuts, with actual-use ports, fixed cotangent seeds, closure tests, and coordinate-dependence boundaries.
3. A depth analysis separating the previously reported globalizing receptive-field phenomenon from heterogeneous Q/K/V and residual trajectories, including prospective MLP-neuron evidence.
4. A behavioral and external audit comparing path maps with simpler and displacement-aware baselines on neuron, head, and regional replacement, followed by disjoint-image and additional-checkpoint validation.

## 1.5 Thesis Organization

Chapter 2 positions NNS, neuron attribution, ViT receptive-field work, edge attribution, and activation patching. Chapter 3 defines neuron learning images. Chapter 4 constructs structural path maps. Chapter 5 specifies samples, interventions, and inference. Chapter 6 reports neuron and structural regularities. Chapter 7 evaluates neuron and component replacement. Chapter 8 reports held-out external validation, limitations, and conclusions.

## 1.6 Chapter Summary

The four research questions follow a neuron representation → structural contribution → intervention → external-validation chain.

'''

def build_results(en=False):
    e1, e2 = rows("e1_summary.csv"), rows("e2_summary.csv")
    e3, e4 = rows("e3_paired_summary.csv"), rows("e4_models.csv")
    e4p, e5 = rows("e4_paired.csv"), rows("e5_summary.csv")
    b1=[]
    for m in ("vit_b16","deit_s16"):
        for k in ("post","mlp"):
            for b in (0,11):
                r=next(x for x in e1 if x["model"]==m and x["kind"]==k and x["metric"]=="same_minus_matched" and int(x["block"])==b)
                b1.append([m,k,b,ci(r)])
    b2=[]
    for m in ("vit_b16","deit_s16"):
        for k in ("mlp_scalar","heads"):
            values=[]
            for b in (0,5,11):
                r=next(x for x in e2 if x["model"]==m and x["kind"]==k and x["metric"]=="entropy" and int(x["block"])==b)
                values.append(ci(r))
            b2.append([m,k,*values])
    b3=[]
    for m in ("vit_b16","deit_s16"):
        for k in ("post","mlp","family"):
            x=[r for r in e3 if r["model"]==m and r["kind"]==k and r["baseline"]=="activation_gradient" and r["outcome"]=="gt" and r["metric"]=="spearman" and r["alpha"]=="1.0" and r["budget"]=="1"]
            vals=[float(r["median"]) for r in x if r["median"]!="nan"]
            b3.append([m,k,len(x),sum(r["decision"]=="SUPPORTED_DIRECTION" for r in x),sum(r["decision"]=="OPPOSITE_DIRECTION" for r in x),f"{min(vals):.4f} to {max(vals):.4f}"])
    b4=[]
    for m in ("vit_b16","deit_s16","vit_b16_in1k","vit_s16_augreg"):
        x={r["features"]:r for r in e4 if r["model"]==m}; p=next(r for r in e4p if r["model"]==m and r["baseline"]=="total")
        b4.append([m,*[f"{float(x[k]['test_mae']):.4f}" for k in ("total","structure","shuffled")],ci(p)])
    b5=[]
    for m in ("vit_b16","deit_s16","vit_b16_in1k","vit_s16_augreg"):
        b5.append([m,*[ci(next(r for r in e5 if r["model"]==m and r["hypothesis"]==h)) for h in ("H1","H2","H3","H4")]])
    if en:
        return {
        "e1":'''## 6.7 Prospective E1: Category Consistency of Neuron Learning Images

E1 uses 500 new images, five images from each of 100 classes. At blocks 0, 3, 7, and 11, 32 post-block units and 32 post-GELU MLP units are fixed per model and block. Same-class pairs are compared with different-class images matched by an outcome-independent 8×8 RGB descriptor. The primary quantity is same-class minus matched-class Jensen–Shannon similarity, summarized within image and then by class.

**Table 6-7 E1 same-class minus visually matched different-class similarity**

'''+table(["Model","Unit family","Block","Median [95% interval]"],b1)+'''

All 16 model–block–family contrasts are negative after the frozen multiplicity correction. Learning-image distributions are more similar for visually matched different-class images than for arbitrary same-class images under this metric. Activation eta-squared is nonzero (median range 0.20–0.44), but activation selectivity and learning-image category consistency are distinct. E1 does not support stable category semantics for these sampled neurons.
''',
        "e2":'''## 6.8 Prospective E2: MLP Neurons and Structural Paths

E2 evaluates 100 images per original checkpoint across all 12 blocks. A basic MLP neuron is one token × hidden-channel scalar after GELU and before the second linear layer. A channel family contains one hidden channel across 196 patch tokens. Each image and block contains 32 sampled basic neurons and 8 channel families.

**Table 6-8 E2 normalized spatial entropy**

'''+table(["Model","Object","Block 0","Block 5","Block 11"],b2)+'''

MLP-neuron entropy rises sharply while sparsity falls with depth. Pairwise JS similarity among sampled MLP units also rises, showing greater late-layer spatial redundancy. Attention-head entropy instead falls. These dimensionless measures establish different depth profiles without comparing unnormalized head and neuron masses.

Every model–image pair passes 72 MLP, attention, and residual cut checks. The largest raw RGB error is $2.01\times10^{-5}$ under the declared mixed tolerance. At block 11, patch-token MLP channel-family paths to the class logit are exactly zero: after the final token-wise MLP, no later token-mixing operator can carry a patch token to the CLS classification head. This is a structural zero, not missing data.
''',
        "e3":'''## 7.6 Prospective E3: Neuron Replacement and Target Selectivity

E3 evaluates post-block scalars, MLP scalars, and MLP channel families on 100 images per original checkpoint, four blocks, two references, amplitudes 0.1 and 1, and top-k budgets 1, 2, 4, and 8. Path mass is compared with activation, gradient, activation × gradient, displacement, Taylor, random, and activation-matched baselines. High-scoring and hash-frozen random non-target classes control target selectivity.

**Table 7-8 Full-replacement Spearman increment over activation × gradient**

'''+table(["Model","Unit family","Comparisons","Supported","Opposite","Median range"],b3)+'''

Path score has no stable advantage over activation × gradient. Some early MLP-scalar strata show positive rank increments, but no MLP-scalar stratum shows a corrected top-k absolute-effect advantage. Early post-block units show the opposite direction; late effects often collapse to ties or structural zeros. Path-selected sets exceed random and activation-matched sets in many strata, yet this does not become an increment over activation × gradient.

True-class selection does not consistently change the target margin more than high-scoring or frozen-random non-target selection. Behavioral target selectivity is not supported. Neuron maps are therefore treated as descriptive input-coordinate representations rather than uniformly superior intervention rankings.
''',
        "external":'''## 8.1 Prospective E4–E5: Information Gain and External Validation

E4 freezes 200 training, 100 validation, and 100 test images. Ridge models predict mean absolute complete head-replacement effect from the total map, total plus seven correctly labeled structural paths at four blocks, or total plus an equal-dimensional edge-label shuffle. Training-only standardization is used; validation selects the penalty.

**Table 8-1 E4 held-out effect prediction**

'''+table(["Checkpoint","Total MAE","Structural MAE","Shuffled MAE","Total−structural paired error"],b4)+'''

Structural features provide no stable held-out gain. They are significantly worse than total-map features for the original ViT-B/16 and added ViT-S/16; the other intervals cross zero. Correct labels do not consistently beat shuffled labels. Low-total-map-distance pairs retain median structural JS distances of 0.16–0.23, so decomposition preserves information hidden by aggregation, but that information does not improve this predictor.

E5 applies four frozen hypotheses to 400 independent images at two original and two additional public checkpoints. Added checkpoints are different public training recipes, not independent random seeds. H1 is post-neuron entropy depth change; H2 routing-share change; H3 the difference between MLP- and attention-residual identity trajectories; H4 path-map minus activation × gradient head ranking.

**Table 8-2 E5 independent-image and checkpoint contrasts**

'''+table(["Checkpoint","H1","H2","H3","H4"],b5)+'''

H1–H3 retain the frozen positive direction at all four checkpoints. H4 is negative at all four: path-map ranking is below activation × gradient by 0.1136–0.1862 in median within-image correlation. Spatial and structural depth observations replicate across tested public parameter states, whereas the proposed ranking increment is contradicted.

![Figure 8-1 Prospective supplementary results](../figures/revision_v5/revision_v5_results.png)

**Figure 8-1 E1–E5 prospective results.** E1 intervals use classes, E5 uses images, E4 bars are independent-test MAE, and H4 is path map minus activation × gradient.
'''}
    return {
    "e1":'''## 6.7 前瞻实验 E1：神经元学习图像的类别一致性

E1 使用 500 张新图像，覆盖 100 类、每类 5 张。在零起始编号块 0、3、7、11，每模型每块固定 32 个块后单元和 32 个 GELU 后 MLP 单元。同类图像对与由结果无关的 8×8 RGB 描述符匹配的异类图像比较。主要估计量为同类减匹配异类的 Jensen–Shannon 相似度，先在图内汇总，再以类别为统计单位。

**表 6-7 E1 同类减视觉匹配异类的学习图像相似度**

'''+table(["模型","单元族","块","中位数 [95% 区间]"],b1)+'''

16 个模型—层—单元族对比在冻结的多重校正后均为负：该指标下，学习图像对视觉匹配异类的相似度高于对任意同类图像的相似度。激活的类别 eta 平方仍非零，各注册组中位数为 0.20–0.44，但激活选择性与学习图像类别一致性是不同对象。E1 不支持这些抽样神经元具有稳定类别语义。
''',
    "e2":'''## 6.8 前瞻实验 E2：MLP 神经元及其结构路径

E2 在两个原检查点各使用 100 图并覆盖全部 12 块。基本 MLP 神经元定义为 GELU 后、第二线性层前的“标记×隐藏通道”标量；通道族为同一隐藏通道的 196 个 patch 标记集合。每图每块抽取 32 个基本神经元和 8 个通道族。

**表 6-8 E2 归一化空间熵**

'''+table(["模型","对象","块 0","块 5","块 11"],b2)+'''

MLP 神经元熵随深度明显增加，稀疏性下降；抽样 MLP 单元间的 JS 相似度也升高，说明后层空间冗余增加。注意力头熵则下降。这些无量纲指标表明两类子层轨迹不同，没有直接比较未标准化质量。

每个模型—图像执行 72 项 MLP、注意力和残差切面检查，最大原始 RGB 误差为 $2.01\times10^{-5}$，满足声明的混合容差。块 11 的 patch-token MLP 通道族到类别逻辑值的路径严格为零：最后一个逐标记 MLP 后没有新的 token 混合，patch token 无法再影响 CLS 分类头。这是结构性零值，不是缺失数据。
''',
    "e3":'''## 7.6 前瞻实验 E3：神经元替换与目标选择性

E3 对两个原检查点各 100 图的块后标量、MLP 标量和 MLP 通道族执行干预，覆盖四个块、两种参照、0.1 与 1 两种幅度，以及 1、2、4、8 的 top-k 预算。路径质量与激活、梯度、激活乘梯度、位移、泰勒、随机及激活匹配基线比较；高分非目标类和冻结随机非目标类用于目标选择性控制。

**表 7-8 完整替换中路径质量相对激活乘梯度的 Spearman 增量**

'''+table(["模型","单元族","比较数","正向支持","反向","中位数范围"],b3)+'''

路径评分没有稳定优于激活乘梯度。部分早层 MLP 标量族出现秩相关增量，但 MLP 标量的 top-k 绝对效应没有任何校正后优势；早层块后单元出现反向结果，后层还存在大量平局或结构性零效应。路径选择在许多层优于随机和激活匹配集合，说明包含超出激活幅度的信息，但没有形成相对激活乘梯度的稳定增量。

真实类别路径分数所选集合的目标间隔变化未稳定大于高分非目标类或冻结随机非目标类所选集合。行为目标选择性未获支持。神经元图定位为描述性输入坐标表示，不表述为普遍优越的干预排序。
''',
    "external":'''## 8.1 前瞻实验 E4–E5：信息增量与外部验证

E4 冻结 200 张训练、100 张验证和 100 张测试图像。岭模型预测完整注意力头替换绝对效应，输入分别为总图、总图加四块七类正确标注结构路径，或总图加同维度边标签打乱特征。标准化和拟合只使用训练集，验证集选择正则参数。

**表 8-1 E4 独立测试效应预测**

'''+table(["检查点","总图 MAE","结构 MAE","乱序 MAE","总图−结构配对误差"],b4)+'''

结构特征没有提供稳定独立测试增益：在原 ViT-B/16 和新增 ViT-S/16 中显著差于总图，另两个区间跨零；正确边标签也未稳定优于乱序标签。总图 JS 距离最低 10% 的测试配对仍有 0.16–0.23 的结构 JS 距离中位数，说明分项保留了聚合丢失的信息；E4 同时表明这些信息没有改善规定预测器。

E5 在同一批 400 张独立图像上，对两个原检查点及两个新增公开检查点检验四项冻结假设。新增权重来自不同公开训练方案，不称为独立训练种子。H1 为神经元熵层深差；H2 为路由尺度差；H3 为 MLP 与注意力残差恒等路径轨迹之差；H4 为路径图减激活乘梯度的头排序相关。

**表 8-2 E5 独立图像与检查点对比**

'''+table(["检查点","H1","H2","H3","H4"],b5)+'''

H1–H3 在四个检查点中均保持冻结的正方向。H4 在四个检查点中均为负：路径图的图内头排序相关比激活乘梯度低 0.1136–0.1862。空间与结构层深现象在所测试公开参数状态中复现，而路径图排序增量被独立证据否定。

![图 8-1 前瞻补充实验结果](../figures/revision_v5/revision_v5_results.png)

**图 8-1 E1–E5 前瞻补充结果。** E1 区间以类别为单位，E5 以图像为单位；E4 柱为独立测试 MAE；H4 为路径图减激活乘梯度。
'''}

def integrate_body(text, en=False):
    result=build_results(en)
    if en:
        text=text.replace("# Chapter 3 NNS Local Input Representations and Their Computation-Graph Form","# Chapter 3 Neuron-Level NNS Representations in Vision Transformers")
        text=text.replace("# Chapter 6 Depth Differences in NNS–ViT Structural Terms","# Chapter 6 Neuron-Level and Structural-Path Regularities")
        text=text.replace("# Chapter 7 Finite-Replacement Evaluation of NNS–ViT Structural Terms","# Chapter 7 Behavioral Evaluation of Neurons and Structural Paths")
        text=text.replace("# Chapter 8 Discussion and Conclusions","# Chapter 8 External Validation, Discussion, and Conclusions")
        text=text.replace("## 6.7 Discussion and Chapter Summary","## 6.9 Discussion and Chapter Summary")
        text=before(text,"## 6.9 Discussion and Chapter Summary",result["e1"]+"\n\n"+result["e2"])
        text=text.replace("## 7.6 Discussion","## 7.7 Discussion").replace("## 7.7 Chapter Summary","## 7.8 Chapter Summary")
        text=before(text,"## 7.7 Discussion",result["e3"])
        shifts=[("## 8.6 Conclusions","## 8.7 Conclusions"),("## 8.5 Limitations and Future Validation Designs","## 8.6 Limitations"),("## 8.4 Negative Results and Task Scope","## 8.5 Negative Results and Task Scope"),("## 8.3 Structural Differences over the Complete Depth Sequence","## 8.4 Structural Differences over the Complete Depth Sequence"),("## 8.2 Closure, Uniqueness, and Finite Utility","## 8.3 Closure, Uniqueness, and Finite Utility"),("## 8.1 Information Added by Structural Decomposition","## 8.2 Information Added by Structural Decomposition")]
        for old,new in shifts:text=text.replace(old,new)
        text=before(text,"## 8.2 Information Added by Structural Decomposition",result["external"])
        text=text.replace("This chapter answers RQ3:","This chapter addresses the behavioral part of RQ4:").replace("RQ2 is answered by","RQ2 and RQ3 are answered by").replace("RQ3 is answered by","The behavioral part of RQ4 is answered by")
        limitation="## 8.6 Limitations\n\nThe additional checkpoints are public parameter states trained under different recipes, not independent seeds. E1 uses one fixed visual-matching descriptor; its negative result does not test object masks or every semantic notion. E3 evaluates prescribed hybrid internal states. E4 uses a linear ridge predictor and four blocks; a negative result does not prove that no task can use structural indices. No training trajectory, controlled CNN comparison, object-mask localization, or distribution-shift experiment is included. Pixel coordinates are a representation property; primary spatial statistics operate on the 14×14 patch grid and do not establish added value within a patch.\n\n"
        text=re.sub(r"## 8\.6 Limitations\n.*?(?=## 8\.7 Conclusions)",limitation,text,flags=re.S)
        conclusion="## 8.7 Conclusions\n\nRQ1 establishes explicit ViT neuron objects and input-coordinate learning images. RQ2 finds broader and more redundant late MLP-neuron maps, but E1 does not support stable category consistency under the frozen metric. RQ3 organizes class-target dependence into Q/K/V, head, MLP, and residual paths with verified closure and reproducible depth heterogeneity. RQ4 separates descriptive value from intervention ranking: path scores exceed random and activation-matched controls in many neuron strata but do not stably exceed activation × gradient; regional path maps do not beat corrected random; and structural features do not improve the frozen held-out predictor. Reference-displacement Taylor performs best for its matched replacement task.\n\nOn 400 independent images and four public checkpoints, neuron entropy, routing, and residual-trajectory hypotheses retain their directions, while the proposed head-ranking increment is consistently negative. The supported contribution is therefore a neuron-to-path representation, its verified accounting properties, and replicated structural regularities—not a universally stronger intervention score.\n\n"
        text=re.sub(r"## 8\.7 Conclusions\n.*?(?=# References)",conclusion,text,flags=re.S)
    else:
        text=text.replace("# 第 3 章 NNS 局部输入表示及其计算图形式","# 第 3 章 视觉 Transformer 的神经元级 NNS 表示")
        text=text.replace("# 第 6 章 NNS–ViT 结构分项的层深差异","# 第 6 章 神经元级与结构路径规律")
        text=text.replace("# 第 7 章 NNS–ViT 结构分项的有限替换评价","# 第 7 章 神经元与结构路径的行为评价")
        text=text.replace("# 第 8 章 讨论与结论","# 第 8 章 外部验证、讨论与结论")
        text=text.replace("## 6.7 讨论与本章小结","## 6.9 讨论与本章小结")
        text=before(text,"## 6.9 讨论与本章小结",result["e1"]+"\n\n"+result["e2"])
        text=text.replace("## 7.6 讨论","## 7.7 讨论").replace("## 7.7 本章小结","## 7.8 本章小结")
        text=before(text,"## 7.7 讨论",result["e3"])
        shifts=[("## 8.6 结论","## 8.7 结论"),("## 8.5 局限与后续验证设计","## 8.6 局限"),("## 8.4 负结果与适用任务","## 8.5 负结果与适用任务"),("## 8.3 完整层序中的结构差异","## 8.4 完整层序中的结构差异"),("## 8.2 闭合、唯一性与有限效用","## 8.3 闭合、唯一性与有限效用"),("## 8.1 结构分解增加的信息","## 8.2 结构分解增加的信息")]
        for old,new in shifts:text=text.replace(old,new)
        text=before(text,"## 8.2 结构分解增加的信息",result["external"])
        text=text.replace("本章回答 RQ3：","本章回答 RQ4 的行为评价部分：").replace("RQ2 的结果是","RQ2 与 RQ3 的结果是").replace("RQ3 的结果按效用强度依次为：","RQ4 的行为评价结果为：")
        limitation="## 8.6 局限\n\n新增检查点是不同公开训练方案下的参数状态，不是独立训练种子。E1 只使用一个冻结视觉匹配描述符，负结果不等于检验了对象掩码或全部语义概念。E3 操作的是规定的混合内部状态。E4 使用线性岭模型和四个冻结块，负结果不能证明所有任务都无法利用结构索引。本文没有执行训练轨迹、受控 CNN 对比、对象掩码定位或分布变化实验。保留像素坐标是表示性质；主要空间统计仍在 14×14 图像块网格上，不能证明块内像素细节的新增效用。\n\n"
        text=re.sub(r"## 8\.6 局限\n.*?(?=## 8\.7 结论)",limitation,text,flags=re.S)
        conclusion="## 8.7 结论\n\nRQ1 明确定义了 ViT 神经元对象及其输入坐标学习图像。RQ2 发现后层 MLP 神经元图更广、更冗余，但 E1 在冻结指标下不支持稳定类别一致性。RQ3 将类别目标依赖组织为 Q/K/V、注意力头、MLP 和残差路径，验证了切面闭合及可复现的层深异质性。RQ4 区分描述价值和干预排序：路径评分在许多神经元层组中优于随机与激活匹配控制，却未稳定优于激活乘梯度；区域路径图未优于修正随机；结构特征未改善冻结的独立测试预测器。参照位移感知泰勒量在与其信息匹配的替换任务中表现最好。\n\n在 400 张独立图像和四个公开检查点上，神经元熵、路由及残差轨迹假设保持方向，而头排序增量在四个检查点中均为负。因此，本文成立的贡献是“神经元—结构路径”表示、可核验的记账性质及可复现的结构规律，而不是普遍更强的干预评分。\n\n"
        text=re.sub(r"## 8\.7 结论\n.*?(?=# 参考文献)",conclusion,text,flags=re.S)
    return text

def add_cases(text,en=False):
    if "signed_case_1.png" in text:return text
    if en:
        cases='''The main product is also shown as a raw signed reconstruction. Figure 4-3 uses frozen block 5. Figure 4-4 is a descriptive case selected as the largest Q/K/V-cancellation block among the same image's 12 blocks; selection does not enter population inference.

![Figure 4-3 Signed structural reconstruction](../figures/revision_v5/signed_case_1.png)

**Figure 4-3 Signed reconstruction at block 5.** The input, total map, identity and Q/K/V terms, signed reconstruction, and raw error share one registered target.

![Figure 4-4 Descriptive strong-cancellation case](../figures/revision_v5/signed_case_2.png)

**Figure 4-4 Strong-cancellation case.** Block 10 has Q/K/V cancellation 0.5380; raw RGB reconstruction error is $1.37\times10^{-7}$.

'''
    else:
        cases='''主要产物同时以原始带符号重构展示。图 4-3 使用冻结的块 5；图 4-4 在同一图像的 12 块中按 Q/K/V 抵消率最高选取，属于描述性案例，不进入总体推断。

![图 4-3 带符号结构重构](../figures/revision_v5/signed_case_1.png)

**图 4-3 块 5 的带符号重构。** 原图、总图、恒等与 Q/K/V 分项、带符号重构及原始误差来自同一注册目标。

![图 4-4 描述性强抵消案例](../figures/revision_v5/signed_case_2.png)

**图 4-4 强抵消案例。** 零起始编号块 10 的 Q/K/V 抵消率为 0.5380，原始 RGB 重构最大误差为 $1.37\times10^{-7}$。

'''
    return text.replace("## 4.9",cases+"## 4.9",1)

def manifest():
    p=P/"figure_manifest.json"; data=json.loads(p.read_text()); existing={x.get("id") for x in data["figures"]}
    for ident,rel,role in [("revision_v5_signed_case_1","figures/revision_v5/signed_case_1.png","signed reconstruction"),("revision_v5_signed_case_2","figures/revision_v5/signed_case_2.png","descriptive strong-cancellation case"),("revision_v5_results","figures/revision_v5/revision_v5_results.png","prospective E1-E5 summary")]:
        if ident not in existing:data["figures"].append({"id":ident,"path":rel,"sha256":sha(P/rel),"role":role,"analysis_status":"prospective supplementary"})
    data["revision_v5_figure_count"]=len(data["figures"]);p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")

def main():
    for lang,en,front,stop in [("zh",False,front_zh(),"# 第 2 章 相关工作"),("en",True,front_en(),"# Chapter 2 Related Work")]:
        p=P/lang/"manuscript.md";text=p.read_text()
        if "REVISION_V5_RESULTS_INTEGRATED" in text:continue
        text=front+stop+text.split(stop,1)[1];text=integrate_body(text,en);text=add_cases(text,en)
        p.write_text(text+"\n<!-- REVISION_V5_RESULTS_INTEGRATED -->\n")
    manifest();print("REVISION_V5_RESULTS_INTEGRATED")

if __name__=="__main__":main()
