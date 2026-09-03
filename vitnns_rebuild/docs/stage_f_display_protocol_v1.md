# 阶段 F 展示选择协议 v1

> 状态：必须在 Stage D 结果访问前冻结。
> 范围：只用于论文定性展示图。
> 科学决策权限：无。

## 1. 目的

阶段 F 规定 Stage D 的 500 图主实验完成、封存并通过 D10 验证后，论文中的定性案例图如何选择和渲染。它的目标是在不改变 Stage D 预注册 primary decision 的前提下，让论文同时包含可审计的固定案例和视觉上有代表性的结果案例。

本协议不授权 Stage D 执行、不授权 Stage D 分析、不重新解释 C10，也不改变 Stage D fidelity-only 的估计目标、样本、指标、阈值、bootstrap、成本门禁或决策规则。C7-C10 recovery replication 与 Stage D D0-D10 仍由各自协议约束。

## 2. 冻结时点

本协议必须在第一次能够揭示 Stage D reserve 结果的访问之前提交并冻结。这里的结果包括图像级 `delta_fid`、预测正确性、failure 状态、定性热图、展示质量或任何可据此挑图的信息。

若在上述访问之后修改本协议，修改后的版本只能作为探索性展示方案，不能写成预注册选择规则。

## 3. 允许输入

阶段 F 只能消费已经封存并验证的输入：

- Stage D D10 已验证分析包；
- D8/D9 封存的每图 L1 metric-ready maps 和 fidelity curves；
- D1 预注册的 5 个 audit IDs 及其保留的 L2 审计证据；
- Stage D 分析封存后，为渲染展示而读取的原始图像。

阶段 F 不得把展示 PNG、叠加图、平滑图、截图或人工标注回流到任何定量结果、统计检验或 primary decision。

## 4. 输出类别

阶段 F 产生两类互相区分的展示集合：固定审计案例和规则选择的说明性案例。两类都可以进入论文图，但必须在图注和机器报告中清楚标明来源。

### A. 固定审计案例

固定使用 D1 已冻结的 5 个审计样本：

- `ILSVRC2012_val/ILSVRC2012_val_00004877.JPEG`
- `ILSVRC2012_val/ILSVRC2012_val_00028787.JPEG`
- `ILSVRC2012_val/ILSVRC2012_val_00024160.JPEG`
- `ILSVRC2012_val/ILSVRC2012_val_00016533.JPEG`
- `ILSVRC2012_val/ILSVRC2012_val_00014223.JPEG`

这些样本适合方法、保留策略、provenance、repeatability 和审计图。它们不是按视觉效果挑选的最佳案例，也不要求最好看。

### B. 规则选择的说明性案例

D10 验证完整 500 图表后，按以下确定性规则从全体 Stage D cohort 中选择展示案例：

1. `best_effect`：primary `delta_fid` 最大的合格图像。
2. `median_effect`：primary `delta_fid` 最接近 500 图 median 的合格图像。
3. `baseline_strong_nns_stronger`：在 matched baseline 有正向 fine-4x4 contribution 的合格图像中，选择正向 `delta_fid` 最大者。
4. `weak_or_failure`：若存在 required-primary scientific failure，选择 `selection_order` 最小的失败图像；否则选择 primary `delta_fid` 最小的合格图像。
5. `misclassified_or_difficult`：在模型误分类的合格图像中，选择正向 `delta_fid` 最大者；若不存在误分类合格图像，则选择 D10 记录的 clean predicted-class margin 最小的合格图像。

同分或同距离时按以下顺序打破平局：

1. 适用时，距离规则目标更小者优先；
2. 适用时，`delta_fid` 更大者优先；
3. D1 `selection_order` 更小者优先；
4. `sample_id` 字典序更小者优先。

同一图可以满足多个规则。不得静默替换重复样本。阶段 F report 必须保留 rule-to-image 映射；渲染面板可以去重，但底层机器可读表必须显示每条规则选中了哪张图。

## 5. 合格性条件

规则选择案例的合格性条件为：

- 图像属于 D1 冻结的 500 图 cohort；
- 对应 D8/D9 source evidence 完整且已由 D10 验证；
- primary NNS 和 matched baseline 都有有效 L1 metric-ready output；
- 构建展示不需要重跑模型推理或 attribution；
- 图像不是根据人工视觉质量选择。

若某条规则及其后备规则都没有合格图像，阶段 F 必须为该规则记录 `NO_ELIGIBLE_EXAMPLE_FOR_RULE`。不得放宽合格性条件，也不得从 500 图以外抽样。

## 6. 渲染策略

每个被选中的图可以渲染：

- 原始 RGB 图；
- primary NNS map；
- matched baseline map；
- NNS-minus-matched 差异图；
- insertion/deletion fidelity curves；
- 简短机器可读图注，包含选择规则、`delta_fid`、预测状态和 D1 `selection_order`。

渲染变换必须确定性记录。平滑、colormap、alpha blending、contour extraction、resize 和 clipping 都是仅展示操作，不能进入任何 metric。

## 7. 图注要求

所有使用规则选择案例的 Stage F 图，都必须包含等价说明：

```text
案例图按 Stage D 结果访问前预注册的规则选择，仅作说明性展示；
所有 primary 结论均基于完整 500 图队列。
```

固定审计案例必须标为固定审计样本，不得标为最佳案例。

## 8. 机器可读报告

阶段 F 必须输出机器可读 report，至少包含：

- 本协议路径和 SHA-256；
- D10 package 路径、SHA-256 和 canonical digest；
- Stage D cohort size 和 D1 registry digest；
- 完整选择规则与 tie-breaks；
- 每条规则一行：selected `sample_id`、`selection_order`、`delta_fid`、预测状态、后备规则状态和 source hashes；
- 每个渲染产物一行：file SHA-256 和渲染参数；
- `primary_decision_modified=false`；
- `stage_d_statistics_modified=false`；
- `display_only=true`。

## 9. 非目标

阶段 F 不得：

- 改变 Stage D primary decision、validity、fidelity、bootstrap、cost 或 acceptance；
- 根据人工视觉质量挑图；
- 隐藏按规则选出的弱效果、失败或误分类案例；
- 创建新的 confirmatory sample；
- 声称选中案例代表超出其规则定义的总体分布；
- 重跑 D8 inference 或 attribution，除非未来另有版本化协议和授权。
