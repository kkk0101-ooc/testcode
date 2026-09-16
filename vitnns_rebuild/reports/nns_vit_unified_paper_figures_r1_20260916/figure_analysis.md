# 统一论文图 R1：逐图分析

全部图片均从既有封存证据生成；本轮只做展示转换，没有新增模型推理、统计端点或科学决策。

## Image #1: Study overview

- 图能说明什么：研究问题，以及 ViT、Value-path attribution、matched baseline 与 intervention evaluation 的关系。

## Image #2: Value-path construction

- 图能说明什么：fixed-clean-attention head-direction Value-path 的计算对象及匹配 patch-token 比较。

## Image #3: Evaluation protocol

- 图能说明什么：从 attribution 排序、干预曲线到 image-level paired median 的评价协议。

## Image #4: Primary effect distribution

- 图能说明什么：全部 500 个 image-level 效应与 paired-median bootstrap 区间。

## Image #5: Fidelity trajectories

- 图能说明什么：NNS 与 matched 的差异沿 insertion/deletion 轨迹出现的位置。

## Image #6: Scale and operator sensitivity

- 图能说明什么：fidelity 对空间尺度和 upsampling operator 的依赖。

## Image #7: Layer/head and prediction strata

- 图能说明什么：四个冻结 layer/head 坐标及 correct/misclassified strata 的差异。

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

- 图能说明什么：50 图像比较集中的方法分布、四个注册 contrast 和 deletion trajectories。

## Image #12: Post-hoc exploratory multiscale mechanism

- 图能说明什么：post-hoc exploratory 的多尺度一致性、ROAD fidelity 和 within/between SSE。
- 使用边界：该图属于 post-hoc exploratory mechanism analysis，不构成广义稳定性结论。

## Image #13: Branch allocation across ViT blocks

- 图能说明什么：两类残差 branch allocation 的 12-block trajectories 及 deep-minus-shallow contrasts。
- 使用边界：支持的结论是 branch allocation 随深度变化，不是因果形成机制。

## Image #14: Post-hoc exploratory depth extension

- 图能说明什么：post-hoc exploratory endpoint、depth profiles、matched occlusion 和 D1/D2 relation。
- 使用边界：全部面板均属于 post-hoc exploratory 分析。
