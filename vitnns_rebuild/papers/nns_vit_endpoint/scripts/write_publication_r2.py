#!/usr/bin/env python3
"""Legacy publication-R2 generator.

The current revision-v2 manuscripts contain evidence-bounded hand edits that this
historical template cannot reproduce.  Refuse an accidental overwrite unless the
caller explicitly requests the legacy rebuild for archival comparison.
"""
import json
import re
from pathlib import Path
import shutil
import sys
from bind_revision_v1 import build_tables,table

REPO=Path(__file__).resolve().parents[3]
PAPER=REPO/'papers/nns_vit_endpoint'
ROOT=PAPER/'publication_r2_20260920'

ZH_FRONT=r'''# NNS-ViT：通过学习图像理解视觉 Transformer 的内部响应

- 作者：[AUTHOR REQUIRED]
- 单位：[AFFILIATION REQUIRED]
- 通讯作者：[CORRESPONDING AUTHOR REQUIRED]

## 摘要

视觉 Transformer 通过多层注意力和前馈网络将图像转换为类别预测，但不同内部单元如何依赖输入图像仍难以直接观察。本文提出 NNS-ViT，将 Neural Network Scanner 的学习图像表示推广到视觉 Transformer。该方法为神经元、通道聚合响应、注意力路径和残差分支生成保留输入位置的学习图像，使不同内部对象可以在同一图像坐标中进行分析。我们以局部 Jacobian 形式描述学习图像的生成，并推导注意力 routing/value 和残差 identity/branch 的分解。对 ViT-B/16 和 DeiT-S/16 的研究表明，抽样单元的输入依赖随深度向更广、更远的区域扩展，深层注意力更难由固定权重的 Value-only 近似表示，注意力与 MLP 残差呈现不同的层深变化。逐层可视化进一步展示学习图像与前向特征图的对应关系。有限替换实验揭示，学习图像包含成分级行为信息，而具体区域干预还依赖替换参考与位移。该研究提供了一种观察 ViT 内部响应的统一表示，并刻画其局部结构与行为含义。

关键词：视觉 Transformer；神经网络可视化；学习图像；局部 Jacobian；注意力

## 1. 引言

视觉 Transformer（ViT）将图像划分为 patch tokens，随后通过自注意力和前馈网络逐层建立表示（Dosovitskiy et al., 2021）。这一结构能够处理不同位置间的关系，也使内部计算的解释更加复杂：token-channel 单元描述特定位置的响应，attention head 在多个位置间传递信息，残差连接又将输入与分支输出叠加。要理解这些对象之间的联系，需要一种能够保留输入空间位置、并适用于不同内部对象的表示。

现有可视化通常展示特征图、注意力权重或类别相关区域。这些视图各有用途，但其空间分辨率、目标和传播规则不同。前向特征图说明一个通道在 token 网格上的响应，attention 权重说明 token 间的分配关系；它们与某个内部标量关于原始图像的依赖并不是同一个对象。将内部目标映射回统一输入坐标，可以为跨模块比较提供共同的观察基础。

Neural Network Scanner（NNS）为神经元关联一张保留输入位置的 learning image（Dou et al., 2023）。本文沿用这一思想，将学习图像扩展到 ViT 的归一化、平滑激活、输入依赖注意力和残差结构。Figure 1 概括了方法：正常前向计算产生内部响应和预测，NNS-ViT 则为选定响应生成相应的输入坐标图。因此，研究者可以沿网络逐层查看一个响应依赖哪些图像位置，也可以比较不同路径对同一目标的局部影响。

{{F1}}

本文的贡献包括三个方面。首先，我们给出 NNS 学习图像的局部 Jacobian 表达，并将其推广到 ViT 的实际计算算子。其次，我们建立从标量单元到 attention/residual 路径的统一表示，推导不同路径在合法计算切面上的组合关系。最后，我们结合逐层可视化、全样本空间统计和有限替换实验，分析输入依赖的层深变化、路径差异与行为含义。本文的学习图像在数值上是同目标 Gradient × Input；方法的扩展在于内部对象及其结构关系的统一表达。

下文先介绍相关工作，再以坐标初始化和逐层生成说明学习图像，随后给出 ViT 结构分解。实验部分依次讨论表示的正确性、层间与模块间变化、方法可视化比较及有限干预。

## 2. 相关工作

### 2.1 神经元与特征可视化

神经元可视化通过输入模式或实际样本分析内部响应。NNS 将单个神经元的计算组织为输入坐标 learning image，并用 activation-weighted aggregation 表示卷积通道（Dou et al., 2023）。这种组织方式使具有不同原生空间结构的内部对象可以共享同一图像表示。本文将其用于 token-channel 单元、通道聚合响应和 ViT 内部路径，同时保留它们各自的标量定义。

### 2.2 梯度归因与 Transformer 可视化

Gradient × Input 表示观测点上的输入加权梯度；Integrated Gradients 沿参考到输入的路径平均梯度（Sundararajan et al., 2017），conductance 进一步描述输出归因经过隐藏单元的过程（Dhamdhere et al., 2018）。本文使用当前点的真实 Jacobian。Grad-CAM 通过梯度加权特征图获得类别相关定位（Selvaraju et al., 2017）。Attention rollout 组合跨层 attention flow（Abnar and Zuidema, 2020），Transformer relevance 方法则采用特定传播规则（Chefer et al., 2021；Achtibat et al., 2024）。这些方法的目标不同，实验中将其作为互补视图进行同输入比较。

### 2.3 内部干预与解释评价

Activation replacement 通过替换内部响应测量有限输出变化。Activation-space Taylor 使用 clean gradient 与替换位移的内积近似该变化；AtP* 分析了注意力饱和与抵消带来的近似误差（Kramár et al., 2024）。局部输入依赖与有限替换回答不同问题，因此本文分别考察整成分排序和固定空间预算下的区域选择。
'''

EN_FRONT=r'''# NNS-ViT: Understanding Internal Responses in Vision Transformers through Learning Images

- Authors: [AUTHOR REQUIRED]
- Affiliations: [AFFILIATION REQUIRED]
- Corresponding author: [CORRESPONDING AUTHOR REQUIRED]

## Abstract

Vision Transformers transform images into predictions through layers of attention and feed-forward computation, yet the input dependence of their internal units remains difficult to inspect. We introduce NNS-ViT, extending the learning-image representation of Neural Network Scanner to Vision Transformers. The method associates neurons, channel aggregates, attention paths, and residual branches with images that preserve input positions, allowing heterogeneous internal objects to be examined in a common coordinate system. We formulate learning-image generation through local Jacobians and derive attention routing/value and residual identity/branch decompositions. Studies of ViT-B/16 and DeiT-S/16 show that sampled units depend on broader and more distant input regions with depth, that deep attention departs further from a fixed-weight Value-only approximation, and that attention and MLP residuals exhibit different depth trends. Layerwise visualizations relate learning images to actual forward feature maps. Finite replacement experiments reveal component-level behavioral information while showing that spatial intervention also depends on the replacement reference and displacement. NNS-ViT provides a unified view of internal ViT responses and characterizes its local structural and behavioral meaning.

Keywords: Vision Transformer; neural network visualization; learning image; local Jacobian; attention

## 1. Introduction

Vision Transformers (ViTs) divide images into patch tokens and build representations through self-attention and feed-forward layers (Dosovitskiy et al., 2021). This architecture models relationships across positions while making internal computation difficult to interpret. Token-channel units represent responses at particular positions, attention heads transmit information across positions, and residual connections combine inputs with branch outputs. Understanding their relationships calls for a representation that preserves input locations and applies to different internal objects.

Existing visualizations commonly show feature maps, attention weights, or class-related regions. Each view serves a purpose, but their resolutions, targets, and propagation rules differ. A forward feature map shows a channel's response on the token grid, whereas attention weights describe allocation between tokens. Neither is the same object as an internal scalar's dependence on the original image. Mapping internal targets to common input coordinates provides a shared basis for comparing modules.

Neural Network Scanner (NNS) associates a neuron with a learning image that preserves input positions (Dou et al., 2023). We extend this idea to ViT normalization, smooth activations, input-dependent attention, and residual structure. Figure 1 outlines the approach: ordinary forward computation produces internal responses and predictions, while NNS-ViT generates corresponding input-coordinate maps for selected responses. Researchers can therefore inspect where a response depends on an image across layers and compare the local influence of different paths on the same target.

{{F1}}

This work makes three contributions. First, we express NNS learning images through local Jacobians and extend them to the actual operators used by ViTs. Second, we unify scalar units and attention/residual paths in one representation and derive their combination over valid computational cuts. Third, we combine layerwise visualization, cohort-level spatial statistics, and finite replacement to examine depth trends, path differences, and behavioral meaning. The learning image is numerically Gradient × Input for the same target; the extension concerns the common representation of internal objects and their structural relationships.

We next review related work, explain learning images through coordinate initialization and layerwise generation, and derive their ViT decompositions. The experiments examine representation correctness, changes across layers and modules, visual comparisons with other methods, and finite interventions.

## 2. Related Work

### 2.1 Neuron and feature visualization

Neuron visualization examines internal responses through input patterns or observed samples. NNS organizes a neuron's computation as an input-coordinate learning image and represents convolutional channels through activation-weighted aggregation (Dou et al., 2023). This organization lets internal objects with different native spatial structures share one image representation. We apply it to token-channel units, channel aggregates, and ViT paths while preserving their respective scalar definitions.

### 2.2 Gradient attribution and Transformer visualization

Gradient × Input represents the input-weighted gradient at the observed point. Integrated Gradients averages gradients along a reference-to-input path (Sundararajan et al., 2017), and conductance further describes how output attribution passes through hidden units (Dhamdhere et al., 2018). We use the true current-point Jacobian. Grad-CAM forms class-related localization from gradient-weighted feature maps (Selvaraju et al., 2017). Attention rollout combines attention flow across layers (Abnar and Zuidema, 2020), while Transformer relevance methods use specific propagation rules (Chefer et al., 2021; Achtibat et al., 2024). Their targets differ, so our same-input experiments treat them as complementary views.

### 2.3 Internal intervention and explanation evaluation

Activation replacement measures finite output changes after substituting internal responses. Activation-space Taylor approximates this change by the inner product of the clean gradient and replacement displacement; AtP* analyzes errors associated with attention saturation and cancellation (Kramár et al., 2024). Local input dependence and finite replacement answer different questions. We therefore examine whole-component ranking separately from region selection under a fixed spatial budget.
'''

ZH_RESULTS=r'''## 4. 实验设置

### 4.1 模型与图像

实验采用 ViT-B/16 和 DeiT-S/16 两个 ImageNet 分类模型。两模型共享 400 张 evaluation 图像；另有互不重叠的 64 张 discovery 和 64 张 calibration 图像。输入短边 resize 到 248 后中心裁剪为 224，采用各模型的 normalization 参数，推理时关闭 dropout 和 drop path。Table 1 列出模型设置。

{{T1}}

分层单元研究使用另外 100 张共享图像，每层选择 CLS、15 个空间分层 patch tokens 和 32 个预定 channels。每个 model-image 包含 6,144 个 post-block 标量，共 1,228,800 个记录。原目录覆盖 ViT-B/16 与 DeiT-S/16 post-block token-channel 候选的 0.01058%/0.02115%，分层研究覆盖 0.338%/0.677%。这些比例界定了本文的单元抽样范围。

### 4.2 可视化设置

概念和层间图使用主体可辨的懒熊样本，方法对比另外使用螳螂和小蓝鹭。样本按展示可读性选择，总体结论仍来自完整 cohort。为对应学习图像与前向特征，每层按 patch activation 的空间方差选择通道，并将该通道的 energy 定义为标量目标。图中同时给出通道编号及实际 14×14 前向 feature map；不同层可以选择不同通道。

主学习图像采用深蓝至青、黄、红的正部视图。设 $P=\max(\sum_cE_c,0)$，显示值为 $\sqrt{\min(P/q_{.99},1)}$，其中 $q_{.99}$ 在正值像素上计算。该变换提高弱细节的可见性，并保持像素位置；它不参与任何统计。Figure 22 同时显示 signed 与 absolute 视图。对比图另列 patch-max 加双线性插值的显示变体，明确区分原生细粒度图与平滑区域图。

### 4.3 统计与数值验证

所有总体推断先在图内聚合，以图像作为统计单位，使用 10,000 次 bootstrap。成分相关规则的 0.10 与区域增益规则的 0.05 是点估计门槛；符号检验检验相对零的方向，采用原 28 项 Holm 校正，CI 精度上限分别为 0.30/0.50。单元层深统计按模型使用五项校正；后续 residual spatial-TV 分析单独标为 post-hoc。

表格用于报告离散数值验证，避免把少量误差点绘为趋势。Table 2 汇总五类原始 NNS 兼容网络的数值一致性与结构检查。原始 float32 有限差分失败和 float64 复核保留在补充材料中。

{{V}}

## 5. 结果与分析

### 5.1 学习图像与前向特征的层间变化

Figures 8–11 以同一输入展示内部响应。前向 feature map 给出通道在 token 网格上的值，学习图像给出相应 scalar 对输入像素的局部依赖；两行空间图因此能够相互补充。浅层学习图像保留局部纹理，后续层产生不同的分布和细节。图中也存在背景响应与离散热点，表明高分辨率本身不保证完整物体轮廓。

{{F8}}

{{F9}}

{{F10}}

{{F11}}

单元抽样统计进一步描述这一变化（Figure 16）。ViT-B/16 与 DeiT-S/16 的 late-minus-early entropy 增加 0.2201/0.2250，own-token locality 下降 0.3558/0.3424，normalized distance 增加 0.1312/0.1219。各对比均有 100/100 张图呈相同方向。这里的 early/middle/late 指 blocks 0–3/4–7/8–11，结果表明抽样单元依赖更广、更远的输入区域。

{{F16}}

CLS-minus-patch entropy 从 block 0 的 0.4752/0.4712 缩小到 block 11 的 0.0086/0.0238。Net/absolute mass 同时从 0.00872/0.00712 降至 0.00144/0.00132。空间扩展与较强正负抵消可以同时出现，absolute mass 因而描述局部项的总幅值而非净响应。

### 5.2 注意力头与 routing/value 路径

Figure 12 展示三个深度处的全部 DeiT-S heads；Figure 13 给出三个输入在两模型中的类别图。相同类别目标可通过多个空间上重叠的 head 路径依赖输入。Figure 14 将 attention 进一步分为 Q、K、routing 和 value，使分配权重的变化与内容传递可以分别观察。

{{F12}}

{{F13}}

{{F14}}

在 400 张图的目标条件化分析中，routing scale 从 block 0 的 0.3092/0.2041 增至 block 11 的 0.5154/0.4808；Value-only Endpoint error 从 0.4407/0.2774 增至 0.8022/0.8488（Figure 17）。相应 JVP discrepancy 达到 0.9516/0.9636，完整响应 norm 为 14.9711/1.5155，绝对差异为 13.8064/1.4228。深层差异与完整响应同量级，不能解释为遗漏贡献的百分比。

{{F17}}

控制 activation norm 与下游 cotangent norm 后，mean-query saturation 与 routing scale 的 partial-rank 相关仍为 −0.1442/−0.3257。Sharper attention 不能直接作为更强 routing dependence 的代理；CLS-query 结果较弱，完整敏感性分析见补充材料。

### 5.3 残差路径的不同作用

Figure 15 以同一输入和目标展示 attention 与 MLP residual。两种结构均为 identity 加 branch，但 identity 的输入边不同；其局部依赖也呈现不同的层深趋势。Attention residual 的 late-minus-early identity scale 为 −0.1242/−0.0449，MLP residual 为 +0.1798/+0.1780（Figure 18）。

{{F15}}

{{F18}}

后续全 cohort 空间分析得到，attention output–identity patch TV 的 late-minus-early 变化为 +0.0520/+0.0282，MLP 为 −0.0187/−0.0462，四项通过单独 Holm 校正。这些结果描述两个训练模型中不同 residual 子层的局部传递方式。Token-channel 层间分析与类别路径使用不同目标，因此二者提供互补的模型级观察。

### 5.4 与其他可视化方法的比较

Figures 19–20 比较同一输入上的 Endpoint、Grad-CAM、attention rollout 和 raw attention。Endpoint 与 Grad-CAM 均使用 GT logit；Grad-CAM 在最后一个 block 的 norm1 输出处计算，避免最终 CLS 读出使末端 patch 梯度退化。Attention rollout 使用 head 均值、identity residual、逐行归一化及逐层矩阵相乘，raw attention 使用末层 CLS-to-patch head 均值。

{{F19}}

{{F20}}

Endpoint 保留像素尺度的局部项，Grad-CAM 和 attention 类方法提供较粗的区域分布。Patch-max 插值版本表明，聚合与插值本身就能显著改变学习图像的外观；它不改变基础定义。图中各方法有时突出不同区域，这反映目标与传播规则的差别。该比较用于说明表示粒度与对象，方法排序仍需要对应任务的定量评价。

### 5.5 从成分关联到区域干预

在成分替换实验中，12 个模型–reference–成分族组合有 11 个满足原复合条件。Table 3 直接列出相关估计、区间和阈值关系。ViT-B normalized-zero routing/value 为 0.09956521739130435，DeiT-S 对应估计为 0.1065，两个区间均跨越 0.10。因此 family 条件是否满足与总体效应是否超过 0.10 是不同判断；原 all-family omnibus 未满足。

{{T2}}

补充同层 head 比较中，Taylor 具有更高 Spearman 与 Capture、更低 Regret（Table 4）。跨层 residual identity 的 Endpoint−Taylor Spearman 差为 0.0865–0.4209，但 Capture@1 中位差为零，表明增量价值依赖任务和指标。

{{T3}}

区域实验固定 block 5 的三个 heads，所有方法共用 4×4 候选区域及最高 25% 面积。Figure 21 和 Tables 5–7 给出配对差、绝对 AUC 与归一化 regret。Endpoint−regional Taylor 的配对中位差为 −0.023433、−0.023697、−0.029900、−0.035630；四个 family 都未满足相对全部基线的 +0.05 条件。候选 oracle 仅取实际评价集合的最优效果，不能视为全部可能区域的全局最优。

{{F21}}

{{T4}}

{{T5}}

{{T6}}

跨 reference 分析补充了这一结果的含义。Endpoint 使用观测输入处的 clean map，而 Taylor 显式使用替换位移。对 163,200 个成分配对，Taylor magnitude 与 absolute effect 在原 atol=1e-12、rtol=0 标准下均发生变化；分布及 head 排名一致性见补充表。Reference 会改变干预相关排序，局部输入图与位移估计器因而适用于不同问题。

## 6. 讨论

学习图像使不同内部对象能够在同一输入网格上观察。通过图中明确的目标、层与通道，读者可以比较前向响应和输入依赖，也可以检查注意力与残差如何组合局部路径。本文的理论推导和数值检验保证这些视图对应所定义的 Jacobian 对象；有限替换则进一步说明其行为用途。

蓝底高亮视图便于读取分散纹理，但只显示 signed RGB-sum 的正部。Figure 22 将其与 signed 和 absolute 视图并列。正部表示局部坐标缩放项的符号，不等同于有限删除时的预测支持；较大 absolute mass 也可能伴随较强抵消。

{{F22}}

本研究尚不能区分层深趋势来自架构、初始化还是训练。两模型同时在规模、head 数与训练策略上不同；缺少随机初始化和同架构替代 checkpoint 控制。路径结果以 GT logit 为条件，其在 clean-top1 目标以及正确/错误分类子集中的稳健性仍需研究。抽样 token-channel 与类别路径没有组成逐单元匹配的因果链。空间干预结论也受所选层、区域和 reference 限定。ViT-B Path IG 在既定 512 点范围内未达到共同校准要求，因此本文不报告跨模型 Endpoint–Path 比较。

## 7. 结论

本文将 NNS 的学习图像思想推广到视觉 Transformer，以局部 Jacobian 统一表示神经元、通道聚合响应和结构路径。坐标初始化与逐层生成保留输入位置，attention 和 residual 分解使同一目标的不同路径可以分别观察。两模型实验展示了输入依赖的层深空间扩展、深层 attention 对 routing differential 的依赖，以及 attention/MLP residual 的不同变化。与前向特征和其他方法的可视化比较说明了该表示的粒度与对象；有限替换进一步界定其成分级关联和区域选择能力。后续研究将结合训练控制与匹配目标实验，分析这些局部结构规律的形成来源。

## 参考文献与补充材料

[参考文献](../references.md)；[补充材料](supplement.md)。作者、单位及最终归档信息待补充：[REPOSITORY AND ARCHIVE REQUIRED]。
'''

EN_RESULTS=r'''## 4. Experimental Setup

### 4.1 Models and images

We study ImageNet classification models ViT-B/16 and DeiT-S/16. They share 400 evaluation images, with separate sets of 64 discovery and 64 calibration images. Inputs are resized to short side 248 and center-cropped to 224 using each model's normalization; dropout and drop path are disabled. Table 1 summarizes the models.

{{T1}}

The stratified unit study uses a separate set of 100 shared images. Each block selects CLS, 15 spatially stratified patch tokens, and 32 predetermined channels. This gives 6,144 post-block scalars per model-image and 1,228,800 records. Original post-block token-channel coverage is 0.01058%/0.02115% in ViT-B/16 and DeiT-S/16; the stratified study covers 0.338%/0.677%. These fractions define the scope of unit sampling.

### 4.2 Visualization settings

Concept and layerwise figures use a sloth-bear image with a recognizable subject; method comparisons also include a mantis and a little blue heron. Images are selected for presentation readability, while population conclusions retain the complete cohorts. To relate learning images to forward features, each layer selects the channel with highest spatial variance across patch activations and uses its energy as the scalar target. Figures report the channel index and actual 14×14 forward feature map; selected channels may differ across layers.

Main learning images use a deep-blue, cyan, yellow and red positive view. With $P=\max(\sum_cE_c,0)$, displayed values are $\sqrt{\min(P/q_{.99},1)}$, where $q_{.99}$ is computed over positive pixels. This transform improves visibility of weak detail while preserving pixel positions and is excluded from statistics. Figure 22 pairs it with signed and absolute views. Comparisons separately show a patch-max and bilinear-interpolation display variant to distinguish native fine detail from a smoothed regional view.

### 4.3 Statistics and numerical validation

Population inference aggregates within images and treats images as statistical units, with 10,000 bootstrap replicates. The 0.10 component-correlation and 0.05 regional-gain values are point-estimate thresholds; sign tests examine direction relative to zero, with the original 28-test Holm correction and CI-width limits of 0.30/0.50. Unit-depth statistics use five-test families per model. Subsequent residual spatial-TV analysis is labeled post-hoc and corrected separately.

Discrete numerical checks are reported in tables rather than sparse error plots. Table 2 summarizes numerical agreement in five original-NNS-compatible network families and structural checks. Original float32 finite-difference failures and their float64 reviews remain in the supplement.

{{V}}

## 5. Results and Analysis

### 5.1 Learning images and forward features across layers

Figures 8–11 examine internal responses for one input. A forward feature map gives channel values on the token grid; a learning image describes the corresponding scalar's local dependence on input pixels. The paired views therefore complement each other. Early learning images retain local texture, and subsequent layers produce different distributions and details. Background responses and isolated hotspots also occur, showing that high resolution alone does not ensure a complete object contour.

{{F8}}

{{F9}}

{{F10}}

{{F11}}

The sampled-unit statistics characterize this change further (Figure 16). Late-minus-early entropy increases by 0.2201/0.2250 in ViT-B/16 and DeiT-S/16, own-token locality decreases by 0.3558/0.3424, and normalized distance increases by 0.1312/0.1219. All contrasts have the same direction in 100/100 images. Early/middle/late refer to blocks 0–3/4–7/8–11; sampled units depend on broader and more distant input regions.

{{F16}}

CLS-minus-patch entropy narrows from 0.4752/0.4712 at block 0 to 0.0086/0.0238 at block 11. Net/absolute mass falls from 0.00872/0.00712 to 0.00144/0.00132. Spatial broadening and substantial signed cancellation can coexist, so absolute mass measures the total magnitude of local terms rather than the net response.

### 5.2 Attention heads and routing/value paths

Figure 12 shows all DeiT-S heads at three depths, and Figure 13 compares class maps for three inputs across both models. Multiple spatially overlapping head paths can connect the input to the same class target. Figure 14 separates attention into Q, K, routing and value, allowing changes in allocation weights to be examined separately from content transmission.

{{F12}}

{{F13}}

{{F14}}

In the target-conditioned analysis of 400 images, routing scale increases from 0.3092/0.2041 at block 0 to 0.5154/0.4808 at block 11; Value-only Endpoint error increases from 0.4407/0.2774 to 0.8022/0.8488 (Figure 17). Corresponding JVP discrepancies reach 0.9516/0.9636, with full-response norms 14.9711/1.5155 and absolute discrepancies 13.8064/1.4228. Deep-layer discrepancies are on the scale of the full response and are not omitted-contribution percentages.

{{F17}}

After controlling for activation and downstream-cotangent norms, partial-rank correlations between mean-query saturation and routing scale remain −0.1442/−0.3257. Sharper attention is not a direct proxy for stronger routing dependence. CLS-query results are weaker; complete sensitivity analyses are in the supplement.

### 5.3 Distinct roles of residual paths

Figure 15 shows attention and MLP residuals for the same input and target. Both combine identity and branch paths, but their identity paths originate at different input edges and exhibit different depth trends. Late-minus-early identity scales are −0.1242/−0.0449 for attention residuals and +0.1798/+0.1780 for MLP residuals (Figure 18).

{{F15}}

{{F18}}

Subsequent full-cohort spatial analysis gives late-minus-early output–identity patch-TV changes of +0.0520/+0.0282 for attention and −0.0187/−0.0462 for MLP; all four survive a separate Holm correction. These results describe different local transmission patterns in two trained models. Token-channel depth analysis and class-path analysis use different targets and provide complementary model-level observations.

### 5.4 Comparison with other visual explanations

Figures 19–20 compare same-input Endpoint, Grad-CAM, attention rollout and raw attention. Endpoint and Grad-CAM use the GT logit. Grad-CAM is evaluated at the final block's norm1 output, avoiding degenerate terminal patch gradients caused by CLS readout. Rollout uses head averaging, identity residuals, row normalization and successive matrix multiplication; raw attention uses the final block's mean CLS-to-patch weights.

{{F19}}

{{F20}}

Endpoint retains pixel-scale local terms, whereas Grad-CAM and attention methods provide coarser regional distributions. The patch-max interpolation variant shows that aggregation and interpolation alone can substantially change the appearance of a learning image without changing its base definition. Methods sometimes emphasize different regions, reflecting differences in their targets and propagation rules. This comparison illustrates granularity and explanatory objects; ranking methods requires quantitative evaluation matched to the intended task.

### 5.5 From component association to spatial intervention

In component replacement, 11 of 12 model–reference–component-family combinations satisfy the original composite criteria. Table 3 directly reports estimates, intervals and threshold relationships. ViT-B normalized-zero routing/value has estimate 0.09956521739130435 and the corresponding DeiT-S estimate is 0.1065; both intervals cross 0.10. Satisfying a family criterion and establishing a population effect above 0.10 are therefore different conclusions. The original all-family omnibus criterion is unmet.

{{T2}}

In the supplementary within-block head comparison, Taylor achieves higher Spearman and Capture and lower Regret (Table 4). Across-layer residual identity has Endpoint-minus-Taylor Spearman differences of 0.0865–0.4209 but a zero median Capture@1 difference, indicating task- and metric-dependent incremental value.

{{T3}}

The spatial experiment fixes three heads in block 5, a shared 4×4 region pool and a maximum 25% area. Figure 21 and Tables 5–7 report paired differences, absolute AUC and normalized regret. Paired median Endpoint-minus-regional-Taylor differences are −0.023433, −0.023697, −0.029900 and −0.035630. None of the four families meets the +0.05 criterion over every baseline. The candidate oracle maximizes effects only over evaluated sets and is not a global optimum over all regions.

{{F21}}

{{T4}}

{{T5}}

{{T6}}

Cross-reference analysis clarifies this distinction. Endpoint uses the clean observed-input map, whereas Taylor explicitly uses replacement displacement. Across 163,200 component pairs, Taylor magnitude and absolute effect change under the original atol=1e-12, rtol=0 criterion. Their distributions and head-rank concordance are reported in supplementary tables. Reference choice changes intervention-related ordering, so local input maps and displacement estimators serve different questions.

## 6. Discussion

Learning images place heterogeneous internal objects on the same input grid. Explicit targets, layers and channels let readers compare forward responses with input dependence and inspect how attention and residual paths combine. The derivations and numerical checks establish correspondence to the defined Jacobian objects; finite replacement further characterizes their behavioral use.

Blue high-intensity displays make distributed texture easier to inspect while showing only the positive signed RGB sum. Figure 22 pairs this view with signed and absolute maps. Positive values indicate the sign of local coordinate-scaling terms rather than prediction support under finite deletion, and large absolute mass can coexist with substantial cancellation.

{{F22}}

The study cannot yet separate architectural, initialization and training contributions to depth trends. The two models differ in size, head count and training strategy, and random-initialization and same-architecture alternative-checkpoint controls are absent. Path results condition on GT logits; robustness to clean-top1 targets and correct/incorrect subsets remains to be studied. Sampled token-channel and class paths do not form a unit-matched causal chain. Spatial conclusions also depend on the chosen layer, regions and reference. ViT-B Path IG does not meet the common calibration requirement within the fixed 512-point range, so cross-model Endpoint–Path comparisons are not reported.

## 7. Conclusion

We extend NNS learning images to Vision Transformers through a local-Jacobian representation of neurons, channel aggregates and structural paths. Coordinate initialization and layerwise generation preserve input locations, while attention and residual decompositions expose different paths to the same target. Experiments in two models reveal spatial broadening with depth, stronger deep-attention dependence on routing differentials, and distinct changes in attention and MLP residuals. Comparisons with forward features and other visual explanations clarify the representation's granularity and target; finite replacement further characterizes component association and region-selection performance. Training controls and matched-target studies offer a next step toward explaining the origins of these local structural patterns.

## References and Supplementary Material

[References](../references.md); [supplement](supplement.md). Author, affiliation and final archive information remain to be completed: [REPOSITORY AND ARCHIVE REQUIRED].
'''

def main():
    if '--rebuild-legacy-v1' not in sys.argv:
        raise SystemExit('REFUSED: publication_r2_20260920 contains the review-v2 manuscript; pass --rebuild-legacy-v1 only for an intentional archival rebuild')
    manifest=json.loads((ROOT/'figure_manifest.json').read_text());figs={r['original_number']:r for r in manifest['figures']};tables,_=build_tables()
    # Verification values become a table, replacing the earlier sparse-point figure.
    fid=json.loads((REPO/'reports/nns_vit_endpoint_m3_r1_20260919/m3a_fidelity.json').read_text());vr=[]
    for model in fid['models']:
        vr.append([model]+[f"{max(r['relative_l2'] for r in fid['records'] if r['model']==model and r['dtype']==dtype):.3e}" for dtype in ('float64','float32')]+['1,000'])
    validation=table(['Compatible network','Max relative L2: float64','Max relative L2: float32','Comparisons'],vr)
    validation+='\n\n'+table(['Structural / directional check','Count','Result'],[['Positive closures','76,800','All pass'],['Omission/duplication controls','28,800','All rejected'],['Analytic JVP','6,144','All pass'],['Operator directions','88','All pass'],['Atlas float32 finite differences','57,600','36 outside tolerance'],['Same-coordinate float64 review','36','All pass']])
    for lang,front,results,ci in [('zh',ZH_FRONT,ZH_RESULTS,0),('en',EN_FRONT,EN_RESULTS,1)]:
        original=(PAPER/lang/'full_draft_revision_v1_archive.md').read_text()
        method=original.split('## 3. ')[1].split('## 4. ')[0];method='## 3. '+method
        method=re.sub(r'<!-- FIGURE_\d+_START -->.*?<!-- FIGURE_\d+_END -->','',method,flags=re.S)
        # Retain checked mathematics, introduce it through an explicit learning-image construction.
        if lang=='zh':
            method=method.replace('### 3.1 命题 1：NNS 的局部 Jacobian 身份','### 3.1 学习图像的初始化与生成\n\n一个神经元输出标量值，学习图像则保留该标量所依赖的输入坐标。Figure 2 用四坐标网络说明这种对应关系；Figure 3 将输入拆为坐标图；Figure 4 展示这些图如何按局部导数组合。\n\n{{F2}}\n\n{{F3}}\n\n{{F4}}')
            method=method.replace('### 3.2 标量目标与 filter aggregation','### 3.2 从输入 patch 到内部标量\n\nViT 的 patch embedding 将每个局部图像块映射到 token 向量。Figure 5 表明，线性投影改变响应维度，而学习图像仍位于输入像素网格。\n\n{{F5}}')
            method=method.replace('（Figure 2）','（Figure 8）').replace('注册表区分','本文区分')
            method=method.replace('### 3.3 Attention 与 residual 的真实微分','### 3.3 Attention 与 residual 的路径分解\n\nFigure 6 将内容传递与权重变化分开，Figure 7 则将 residual 的 identity 与 branch 并列。\n\n{{F6}}\n\n{{F7}}')
            method=method.replace('### 3.4 命题 2：合法 edge cut 的 Endpoint closure','### 3.4 完整计算切面上的组合关系')
            method=re.sub(r'Signed 显示为.*?零分母或零范数记为 NA。','统计在完整 signed RGB 张量上计算，零分母或零范数记为 NA。蓝底显示规则在实验设置中说明。',method,flags=re.S)
            method=method.replace('Jacobian 记账','Jacobian 分解')
        else:
            method=method.replace('### 3.1 Proposition 1: the local Jacobian identity of NNS','### 3.1 Learning-image initialization and generation\n\nA neuron produces a scalar value, whereas its learning image retains the input coordinates on which that scalar depends. Figure 2 illustrates this correspondence in a four-coordinate network; Figure 3 splits the input into coordinate images, and Figure 4 combines them using local derivatives.\n\n{{F2}}\n\n{{F3}}\n\n{{F4}}')
            method=method.replace('### 3.2 Scalar targets and filter aggregation','### 3.2 From input patches to internal scalars\n\nPatch embedding maps a local image block to a token vector. Figure 5 shows that linear projection changes response dimensions while the learning image remains on the input-pixel grid.\n\n{{F5}}')
            method=method.replace('(Figure 2)','(Figure 8)').replace('The registry distinguishes','We distinguish')
            method=method.replace('### 3.3 True attention and residual differentials','### 3.3 Attention and residual path decomposition\n\nFigure 6 separates content transmission from weight changes, and Figure 7 relates residual identity and branch paths.\n\n{{F6}}\n\n{{F7}}')
            method=method.replace('### 3.4 Proposition 2: Endpoint closure of a valid edge cut','### 3.4 Combination over a complete computational cut')
            method=re.sub(r'Signed displays use.*?zero denominators or norms yield NA\.','Statistics use complete signed RGB tensors; zero denominators or norms yield NA. The blue display convention is described in the experimental settings.',method,flags=re.S)
            method=method.replace('Jacobian bookkeeping','Jacobian decomposition')
        text=front+'\n'+method+'\n'+results
        for n,f in figs.items():
            cap=f['caption_zh' if lang=='zh' else 'caption_en'];block=f'![Figure {n}](../{f["files"]["png"]["path"]})\n\n**Figure {n}.** {cap}'
            text=text.replace('{{F'+str(n)+'}}',block)
        for n,t in tables.items():
            if n==7:continue
            shown=n if n==1 else n+1
            titles={1:('模型设置','Model settings'),2:('成分关联估计及区间','Component association estimates and intervals'),3:('同层 head 排序：E=Endpoint，T=Taylor','Within-block head ranking: E=Endpoint, T=Taylor'),4:('空间干预配对差','Paired spatial-intervention differences'),5:('绝对 AUC 与输出尺度','Absolute AUC and output scale'),6:('候选 oracle regret 与 logit 尺度比','Candidate-oracle regret and logit-scale ratios')}
            # Keep exact numerical rows; replace machine-state labels with ordinary scientific terms.
            t=t.replace('Frozen status','Composite criterion').replace('NOT_SUPPORTED','Unmet').replace('SUPPORTED','Met')
            text=text.replace('{{T'+str(n)+'}}',f'**Table {shown}. {titles[n][ci]}**\n\n'+t)
        text=text.replace('{{V}}',('**Table 2. 数值一致性与结构检查**' if lang=='zh' else '**Table 2. Numerical agreement and structural checks**')+'\n\n'+validation)
        text=text.replace(chr(92)*2,chr(92))
        numbering={12:13,13:14,14:15,15:17,16:12,17:16}
        text=re.sub(r'Figure (\d+)',lambda m:'Figure '+str(numbering.get(int(m.group(1)),int(m.group(1)))),text)
        text=text.replace('原冻结分析','预先指定的分析').replace('all-family omnibus','全部成分族的联合条件' if lang=='zh' else 'all-family criterion').replace('partial-rank 相关','偏秩相关').replace('完整 cohort','完整样本').replace('原始 arrays','原始数组')
        (ROOT/lang/'manuscript.md').write_text(text)
        # Local supplement includes prior tables and all historical images without external image links.
        supp=(PAPER/lang/'supplement_revision_v1.md').read_text()
        entries=json.loads((ROOT/'evidence/supplement_manifest.json').read_text());lookup={r['source']:'../'+r['path'] for r in entries}
        def localize(m):
            target=m.group(1);relative=target.removeprefix('../../../')
            if relative in lookup:return ']('+lookup[relative]+')'
            return m.group(0)
        supp=re.sub(r'\]\(([^)]+)\)',localize,supp)
        # Replace legacy index with the local complete image index; remove obsolete source-only links.
        supp=supp.split('## S7.')[0]+'## S7. Complete supplementary image index\n\n'
        supp+='\n'.join(f'- [S{r["number"]:02d} — {Path(r["path"]).stem}](../{r["path"]})' for r in entries)
        supp+='\n\n## S8. Current qualitative comparisons\n\nThe six CPU image-model computations use the same source images, checkpoints and preprocessing. Grad-CAM uses the last norm1 patch activations; attention rollout and raw attention are class-agnostic. These presentation-selected examples add no population tests. See [display protocol](../evidence/display_protocol.md) and [compute receipt](../evidence/compute_receipt.json).\n\n'
        supp+='**Table S6. Cross-reference rank concordance.**\n\n'+tables[7]+'\n'
        supp=re.sub(r'\]\(\.\./\.\./\.\./reports/nns_vit_endpoint_paper_revision_v1_20260920/([^)]*)\)',r'](../evidence/\1)',supp)
        supp=supp.replace('(full_draft.md)','(manuscript.md)')
        (ROOT/lang/'supplement.md').write_text(supp)
    refs=(ROOT/'references.md').read_text()
    if 'Selvaraju' not in refs:refs+='\nSelvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. Proceedings of ICCV, 618–626.\n'
    (ROOT/'references.md').write_text(refs)
    root_relative=ROOT.relative_to(REPO).as_posix()
    binding={'source_revision':'PUBLICATION-R2','figure_count':len(manifest['figures']),'package':root_relative,'figures':[]}
    for figure in manifest['figures']:
        row={'figure':figure['number'],'slug':figure['slug']}
        for extension,record in figure['files'].items():
            row[extension]={'path':root_relative+'/'+record['path'],'sha256':record['sha256']}
        binding['figures'].append(row)
    (PAPER/'figure_manifest.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2)+chr(10))
    print('Wrote synchronized formal manuscripts with 22 figures and 7 main tables.')

if __name__=='__main__':main()
