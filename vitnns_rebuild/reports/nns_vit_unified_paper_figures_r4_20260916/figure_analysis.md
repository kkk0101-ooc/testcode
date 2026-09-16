# 统一论文图 R4：逐图分析

Figure 1–3 重画；Figure 7 沿用 R3；其余图沿用 R2。

## Image #1: One ViT, two attribution readouts

先看上半部：同一张图经过同一个已训练 ViT。每个 token 是高维向量，能够编码 patch 内结构；模型对不同 patch 的注意力本来就可以不同。图没有把 patch 描述成“内部像素完全相同”。

再看下半部：NNS 与 token 基线是两种解释读出。左侧 NNS 给输入位置计算不同贡献；右侧基线每 token 只有一个贡献标量。示例中 patch A 总分 16、patch B 总分 8，分别均分为每 cell 1 和 0.5。两块之间不同，各块内部相同。图面直接标明两个 patch，避免将局部 4×4 网格误读为整图 patch 网格。

这张图说明比较对象，不证明 NNS 的细节更准确；准确性需要 Figure 3 的干预评价。NNS 示例与 token 数值是独立的示意，不宣称每个输入 patch 的 NNS 总和等于 c_j。

## Image #2: Same internal target, different contribution coordinates

上方先固定双方共同解释的标量 f：选定 head 的 CLS 聚合结果沿方向 u 的分量。u 来自原图相对灰色参考的 head 输出变化，A(x) 是原图的 CLS attention。固定 A/u 发生在这个解释问题中，正常 ViT 训练和推理仍可产生不同 attention。

左侧从灰色参考逐步走向原图，通过积分梯度计算输入位置贡献，RGB 带符号相加，得到细粒度地图。右侧直接从 token 的 Value 端点变化计算 c_j，然后均匀展开。右侧不是 raw attention；c_j 同时取决于 A、Value 变化和 u。

两个箭头表示独立计算途径：NNS 并不是拿 c_j 做更漂亮的插值。尤其在深层，一个 token 已混合其他位置的信息，因此 c_j 的原图空间映射与 NNS 的输入归因坐标不同。CLS 标量单列，不硬分到某个图像位置。

## Image #3: Evaluate the ranking at a shared spatial scale

左列说明尺度：14×14 是整图的 patch 数量网格；16×16 是每个 patch 的像素尺寸；每个 patch 内有 16 个 4×4-pixel cells。中列比较细分排序：NNS 可区分同 patch 内的小区域，均匀基线在一个 patch 内同分。它仍能区分不同 patch。

右列说明检验：从灰图优先恢复高分区域，观察内部目标恢复的速度；从原图优先遮挡高分区域，观察目标下降的速度。双方使用相同目标和面积预算。展示中的大块只是示意，真实 fine 实验以 4×4 像素为单位。

下方说明汇总：先在每图内平均 5 次同分排序的 AUC，再计算双方的忠实度差值，最后跨图像求 median 和 paired bootstrap CI。五次重复不是五张独立图像。图中局部排序号与 mask 不是真实测量，不能用它们直接判断 NNS 获胜。

## Image #4: Primary effect distribution

删除只有一个点区间的旧 (b)，以直方图与 ECDF 同时呈现分布形状和全部逐图效应。正式 median 与 CI 嵌入图中，CI 不被伪装成逐图分布。500/500 positive 支持冻结目标与 fine endpoint 下的主效应。

## Image #5: Fidelity trajectories

- 图能说明什么：NNS 与 matched 的差异沿 insertion/deletion 轨迹出现的位置。

## Image #6: Scale and operator sensitivity

旧竖向线段只给 median/IQR，无法看见偏态和极值。新版展示全部逐图 AUC 与四分位箱体；箱体不外加默认 whisker，也不隐藏任何 outlier。算子结果是描述性分析，不选择赢家。

## Image #7: Layer/head and prediction strata

三组结果共用四个横轴坐标，直接比较 overall、correct 和 misclassified。完整纵轴保留 block 0/head 9 与其他路径的真实效应差距；右上 inset 让小效应及其区间可读，避免对数轴或断轴。颜色始终表示预测分层，未添加新统计检验。

Figure 7. 四个冻结 layer/head 坐标的 NNS−patch baseline median ΔFID。灰、蓝、橙色分别表示 overall、correct、misclassified，柱高为 median，误差线为原 Stage E 图像级配对 bootstrap 描述性 95% CI。主图所有坐标共用从零起始的纵轴，右上 inset 将后三个坐标按 0–0.10 的线性纵轴放大；两处呈现的是同一组数值，不能重复计为新观测。总体 cohort 为 500，正确/误分类分别为 418/82；block 11/head 2 的有效数为 498/417/81，其余坐标全部有效，两个 typed failures 仍保留在分母中。坐标在 Stage C 冻结，三组包含重叠样本，不是三个独立实验，也未检验组间等效性。区间沿用 Stage E 的描述性 bootstrap namespace，与 Figure 4 的正式主分析区间略有差异。

## Image #8: Frozen case atlas

- 图能说明什么：按冻结规则得到的 median、weak 和 misclassified 案例。
- 使用边界：逐面板显示归一化只用于比较空间形态，不用于比较 attribution 幅值。

## Image #9: Head/depth progression

- 图能说明什么：一个冻结样本在四个 depth bands 上的空间形态变化。
- 使用边界：热图来自一个冻结样本；总体坐标结论以 Stage E cohort statistics 为依据。

## Image #10: Method comparison atlas

- 图能说明什么：formal cohort 前四个位置上 NNS、pooled NNS、Grad-CAM、full IG 与 rollout 的空间差异。
- 使用边界：逐面板 P99 缩放只服务显示；方法间定量关系由 Figure 11 给出。

## Image #11: Method comparison statistics

四个比较使用有正负方向的实心效应条，空心 CI 区域与样本 IQR 分开编码。读者可直接辨认 NNS 低于 Grad-CAM、高于 rollout，而 full IG/raw CLS 区间跨零；不能据此宣称等效。paired mean 通过 sample_id 对齐逐图分数独立复核。

## Image #12: Post-hoc exploratory multiscale mechanism

- 图能说明什么：post-hoc exploratory 的多尺度一致性、ROAD fidelity 和 within/between SSE。
- 使用边界：该图属于 post-hoc exploratory mechanism analysis，不构成广义稳定性结论。

## Image #13: Branch allocation across ViT blocks

完整 block trajectories 说明形状，右侧效应条说明深浅变化的方向和幅度。两条分支用相同语义色贯穿两个面板；CI 原值直接呈现。allocation shares 来自不同 cuts，不相加解释为 100%。

## Image #14: Post-hoc exploratory depth extension

将量纲不同、样本量不同的四个 endpoint 从同一 forest 轴上移到数值表，消除横向位置暗示可直接比较的误导。表中保留 ordinary BCa CI 与 valid/total。其余 progression、matched occlusion、D1/D2 relation 延续原数据；全部仍为 post-hoc exploratory。
