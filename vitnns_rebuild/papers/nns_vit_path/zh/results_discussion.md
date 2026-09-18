# 5. 结果与讨论

## 5.1 Learning image 与真实 ViT 前向计算对应

Figures 24-27 先用解析例把输入、patch projection、attention 和 residual 的正常前向值与 learning image 对齐。Attention 的完整输入导数同时包含 $A,dV$ 与 $(dA)V$；Figure 22 显示，固定 clean attention 后得到的条件 Value-path 图与完整路径图不同。Figure 23 则显示，对一个下游标量目标，residual 与 branch 的成分图在输入坐标逐点相加。

真实 ViT 结果延续了这两项对应关系。Figures 28-31 追踪同一冻结图像的 patch embedding、B0 attention 与第一次 residual。Figures 32-33 增加 B0 MLP 和第二次 residual，使完整 Block 0 闭合。四个固定通道的 MLP 和 block-output learning images 在 128 阶 Gauss-Legendre 积分下满足 completeness；第二次 residual 的最大逐像素 RGB 加法误差为 $3.33\times10^{-16}$。R10 重新计算的第一次 residual 与此前保存结果逐元素相同。

Figure 34 将目标延伸到 class 235 的完整 logit。Clean 与 reference logits 分别为 8.082624 和 -0.384007，目标差值为 8.466630。4096 阶相对 2048 阶的地图 L1 变化为 $8.293\times10^{-4}$，completeness 绝对误差为 0.004874。Figure 35 在同一路径上比较 class 235、runner-up class 174、两者 margin 和全局最低分 class 171。Clean logits 分别为 8.0826、7.6327、0.4499 和 -2.9210；四个目标均通过其冻结 completeness 阈值。

这些图建立的是计算对象与输入坐标的对应。Module-own-response 图回答内部响应来自哪里，class-logit 图回答最终标量变化来自哪里，margin 图则解释两个类别输出之差。Figures 28-35 使用一张冻结图像，因此它们验证方法和数值关系，而不估计通道语义或类别机制的总体分布。

## 5.2 指定内部目标的细粒度干预结果

Stage D 在独立 500 图队列上评价 block 0/head 9 的固定 Value-path 目标。Figure 4 给出每张图像的配对 delta-fidelity，Figure 5 给出 insertion 和 deletion 轨迹。500/500 个差值严格为正，mean 为 2.6307，median 为 2.3082，sample SD 为 1.4807。10,000 次配对图像 bootstrap 得到 median 的 95% CI [2.1111, 2.4452]。该结果通过了预先冻结的 positive-count、effect-size 与 CI 门，Stage D 决策为 `GO_FIDELITY_SCOPE`。

同目标 patch-token 基线保留 token 空间贡献，并在每个原生 patch 内均匀分给 16 个细粒度 cells。两种方法解释相同的内部方向并使用相同干预协议，但其贡献坐标不同。对这一目标，输入归因提供了更有效的 4x4-cell 干预排序。该比较不要求逐 patch 质量相等，且评价对象是区域排序而非每个 signed cell。

较早的 C10 15 图队列给出了相同 fidelity component 的正向证据：15/15 个差值为正，中位数 1.7535，95% CI [1.3318, 3.3813]。C10 的复合判定仍为 `NO_GO_NOVELTY`，因为 translation 与 center-crop/resize spatial gates 失败；其 workflow 为 `BLOCKED`，Holm 状态为 `HOLM_BLOCKED_UNDERSPECIFIED`。Stage D 是独立的 fidelity-only 复制与外推，两个 cohort 没有池化。

## 5.3 主效应依赖空间尺度和内部坐标

Figure 6 比较细粒度、原生 patch 和空间实现。将评价单位改为 14x14 原生 patch 后，500 张图像的 median delta-fidelity 为 0，400 个差值恰为 0，最大绝对差值仅 0.00138。细粒度主效应因此不是一个与评价尺度无关的整体优势；它主要出现在 patch 内的 4x4-cell 排序。

Figure 7 进一步比较四个预定层头坐标。$(0,9)$ 的 valid-only median 为 2.3082，而 $(4,2)$、$(6,10)$ 和 $(11,2)$ 分别为 0.0734、0.0389 和 0.0562。前三个坐标均有 500 个有效结果；$(11,2)$ 有 498 个有效结果和两个保留的 IG-conservation typed failures，发生在 selection orders 117 和 352。该异质性说明方法效果取决于被解释的内部路径；本次比较范围为这四个坐标。

按最终预测分层后，正确分类的 418 张图像 median 为 2.2955，误分类的 82 张图像 median 为 2.3395，两组差值均全部为正。效应出现在两个预测分层中；本研究未为两组幅值设置等价性检验。

Stage E 是 D10 之后的 secondary/descriptive analysis。E2 的非成本条件通过，但实际 cost status 为 `FAIL_COST`，由明确且仅限 E2 的 owner waiver 接受。E3 的两项 typed failure 被保留，最终 E4 package/workflow 通过而 scientific validity 为 `MIXED`。这些状态不改变有效端点的数值，但决定了分母和结论写法。

## 5.4 全 head 图谱描述差异，不建立稳定分工

Figures 8-9 从一个冻结的中位效应样本展示 144 个 head 及其跨深度空间形态。图谱显示不同 block/head 的热点位置、范围和符号结构不同。Figure 20 还固定 blocks 0、5、11 与 heads 0、5、11，在同一类别目标上展示成分图。

这些图以查看前冻结的索引描述空间组织。跨 head 的显示平均不同于类别贡献的代数求和，因为各 head 可对应不同响应方向和独立缩放。在 Stage Y 联合检验中，effective-head-count 对比为 `INCONCLUSIVE`，allocation-ablation association 和 top-versus-bottom ablation 对比均未达到冻结门。当前证据显示 head 图谱具有差异，稳定语义分工仍未确定。

## 5.5 独立类别解释器比较给出互补而非全面优势

Figures 10-11 比较固定 clean attention 的条件类别 NNS、ViT Grad-CAM、full-model IG、attention rollout 和 raw CLS attention。正式 50 图 primary contrast `NNS - Grad-CAM` 的 mean 为 -0.05585，95% CI [-0.07085, -0.04100]，注册的 unadjusted $p=9.9999\times10^{-6}$。NNS 在该 fidelity 指标上低于 Grad-CAM。

Secondary contrast `NNS - rollout` 的 mean 为 0.03943，95% CI [0.02009, 0.05938]，Holm-adjusted $p=0.000780$，显示 NNS 高于 rollout。`NNS - full-model IG` 的 mean 为 0.00558，95% CI [-0.00570, 0.01667]；`NNS - raw CLS attention` 的 mean 为 0.01126，95% CI [-0.00452, 0.02675]。两者 Holm-adjusted $p$ 均为 0.3473，因此本实验未检出差异；这一结果不构成统计等价。

方法比较也强调了目标差异。条件 NNS 固定每层 clean attention，但保留 Value、residual、MLP 和 classifier；full-model IG 允许 attention 随积分输入变化；rollout 与 raw CLS attention 不使用类别目标。Figure 10 的逐面板归一化只显示空间形状，方法高低由 Figure 11 的完整 50 图统计决定。

## 5.6 完整切分把同一类别归因组织为内部成分

Figure 21 展示一个完整 cut 的重构：residual 成分与 12 个 head 成分在着色前相加，恢复同一 total input IG。实现控制中，总 NNS 图与同合同 full-model IG 的逐元素最大误差为 0。135 个完整 cuts、38,107 个 components 通过验证，审计样本上的最大逐像素 cut 重构误差为 $1.5646\times10^{-6}$。

总图与同合同 full-model IG 相同。完整切分增加的是内部成分索引：同一个类别 attribution 可以按 head、attention、MLP、residual 或其他冻结 cut 查看。每个 cut 都是总归因的一种替代分解，应分别解释。

## 5.7 Attention 与 MLP 的贡献份额随深度向相反方向变化

Figure 13 报告 100 张图像上两处 residual cut 的分支绝对贡献份额。第一处 residual 中，attention share 的 deep-minus-shallow estimate 为 0.1391，95% CI [0.1282, 0.1533]，Holm-adjusted $p=0.000140$。第二处 residual 中，MLP share 的对应 estimate 为 -0.2492，95% CI [-0.2572, -0.2349]，adjusted $p=0.000140$。

这两个结果支持冻结类别目标下的深度依赖分支重分配：深层第一处 residual 的 attention 绝对质量份额更高，而深层第二处 residual 的 MLP 份额更低。它们属于两个独立 cut，各自在 identity/attention 或 residual/MLP 内归一化，不能相加为一个三分支百分比。

Stage Y 的其他结果限制了进一步推断。完整类别目标的 patch 内增益和空间支持深浅差异为 `NOT_SUPPORTED`；head organization 为 `INCONCLUSIVE`；patch mixing 与一组扰动端点因有效分母为 0 而 `INVALID`。因此，分支份额变化不能被解释为 Stage D fidelity 优势的已证实原因，也不能扩写成统一的浅层到深层语义规律。

## 5.8 广义稳定性未通过，多尺度只提供探索性线索

Stage X 的 200 条预期 clean routes 中有 199 条有效，但只有 115 个稳定性 cells 中的 50 个通过，且相对 Grad-CAM 的 stability noninferiority 未通过。Broad-claim development 因而保持 `NO_GO`。这项负结果说明，固定设置下的 fidelity 不能自动推广为输入变换下的稳定解释。

Figure 12 汇总后续 post-hoc exploratory 分析。在冻结共同域内，NNS signed Spearman 从 pixel 的 0.1137 上升到 4x4-cell 的 0.2177 和 native-patch 的 0.3378。局部单元内部残差占 SSE 的 0.9255 和 0.9974，与细粒度区域内部重新分配的解释相符。该分析的机制分支为 `MIXED_OR_INCONCLUSIVE`：粗粒度聚合提高 association，却同时隐藏了主干预实验关注的 patch 内差异。

## 5.9 事后深度扩展区分对齐、遮挡和可解码性

Figure 14 汇总 Stage S 的四个 post-hoc exploratory endpoints。NNS 高贡献区域相对位置对照的目标对齐差值 P1 为 0.06260，ordinary BCa 95% CI [0.05739, 0.06789]，235/240 有效。Matched occlusion 差值 P2 为 0.07494，95% CI [0.05708, 0.09273]，224/240 有效。深层减浅层的对齐差值 D1 为 0.03631，95% CI [0.03162, 0.04097]；线性 probe gain 的 D2 为 0.04615，95% CI [0.04409, 0.04824]。

在 233 张共同有效图像上，D1 与 D2 的 Pearson correlation 仅为 0.00161，same-sign fraction 为 0.6180。总体平均都随深度增加，但逐图变化几乎不线性相关，因此对齐增强和可解码性增强是不同现象。P2 的最终 0.07 阈值在 outcome access 后确定，原始规则未通过；本节只生成后续假设，不提供确认性机制结论。

## 5.10 综合讨论

本研究把 learning image 从一种内部可视化对象转化为三类可核对的问题。第一，输入坐标下的图能否重构目标变化以及 residual 相加；第二，图的区域排序能否预测对同一目标的输入干预；第三，完整类别归因如何在内部 cut 上分配。方法展示、Stage D 和 Stage X/Y 分别回答这三个层次。

最清晰的正结果位于固定内部目标和细粒度干预：NNS 输入归因排序优于同目标 token 空间基线的均匀展开。原生 patch 尺度差异消失，其他坐标效应缩小，说明 patch 内空间信息和目标选择共同决定了这一结果。独立 class-output comparison 又显示 NNS 并非所有解释器中的第一名。因而，更合适的定位是一个用于观察指定内部响应和分解类别贡献的补充工具。

完整切分把“图看起来不同”推进到成分记账：同一 total attribution 能够被 head 与 branch 重构，分支份额也可以跨深度比较。当前唯一获得 Stage Y 支持的机制主张是 attention 与 MLP 份额的深度变化；其他未支持、未决或无效端点给出了清楚的研究边界。
