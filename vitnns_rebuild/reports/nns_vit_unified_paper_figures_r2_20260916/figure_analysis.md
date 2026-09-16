# 统一论文图 R2：逐图分析

R2 为展示修订；正式估计、区间和科学状态沿用封存源。

## Image #1: Study overview

用真实冻结原图、token 网格、12-block/12-head 符号与两个贡献网格说明研究对象。读者先看到“同一个 patch 内存在什么信息”，再看到评价。方法图中的矩阵和掩码是示意，未新增实验。

## Image #2: Value-path construction

画面分为目标、积分路径、空间读出三部分。固定 A/u 与可变 Value 的区别直接写入公式；NNS 保留输入空间的 signed 信息，patch baseline 只保留解析 token contribution。CLS 没有被分配到图像位置。

## Image #3: Evaluation protocol

用原生 patch 与放大的 fine cells 消除“14×14 是 patch 像素尺寸”的歧义。恢复/删除序列仅作流程说明，真实实验仍使用原 21 fractions、5 paired repeats 和图像级 bootstrap。

## Image #4: Primary effect distribution

删除只有一个点区间的旧 (b)，以直方图与 ECDF 同时呈现分布形状和全部逐图效应。正式 median 与 CI 嵌入图中，CI 不被伪装成逐图分布。500/500 positive 支持冻结目标与 fine endpoint 下的主效应。

## Image #5: Fidelity trajectories

- 图能说明什么：NNS 与 matched 的差异沿 insertion/deletion 轨迹出现的位置。

## Image #6: Scale and operator sensitivity

旧竖向线段只给 median/IQR，无法看见偏态和极值。新版展示全部逐图 AUC 与四分位箱体；箱体不外加默认 whisker，也不隐藏任何 outlier。算子结果是描述性分析，不选择赢家。

## Image #7: Layer/head and prediction strata

三张走向相同的折线合并成一个 3×4 矩阵：行看预测分层，列看冻结路径。统一线性色阶保留真实效应大小，数值、区间与覆盖度同时可见。两项 typed failures 仍在分母中；三组不是三个独立重复实验，不能将 n 相加。

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
