# 步骤 4 恢复修订：来源坐标精确收缩

- 状态：`FROZEN_PROSPECTIVE_ATTEMPT1`
- 日期：2026-09-21（PRC）
- 原失败：`step4_attempt0_receipt.json`，不可改写

## 1. 失败与恢复边界

attempt0 的真实 ViT 结果保留：两个模型的逐坐标一阶端口闭合、PCIA→IG、乘积法则、方向曲率零和与适用负控制通过，但冻结 512 节点不能积分高曲率的 $\mathrm dQ/\mathrm dt$，且四个 patch 象限来源分组的生产投影为 268.38 H20-hours，超过 16 H20-hours 门。attempt0 还因性能修订造成冻结后源码漂移，因此同时记为 `FAIL_SOURCE_BINDING`。

本修订只采用原规划明确允许的“收缩坐标分辨率”。不改模型、样本、目标、参考、路径、端口、切面、精度、容差或预算。

## 2. 冻结对象

来源坐标 $j$ 收缩为覆盖全部 normalized RGB 坐标的单一精确组 $G$。对每个真实端口保留

\[
\bar T_{e,i}=\sum_{j\in G}T_{e,i\leftarrow j}=Q_{e,i}(x)-Q_{e,i}(b)=Q_{e,i}(x),
\]

\[
\bar R_{e,i}=\bar T_{e,i}-A_{e,i}.
\]

$\bar T$ 的端点式来自微积分基本定理，不是不可审计近似；$A$ 仍使用冻结 512 点 Gauss–Legendre。这样保留真实端口 $e$、Endpoint 坐标 $i$、所有合法切面和逐 RGB 坐标闭合，只不再保留来源坐标 $j$ 的分组读出。

完整切面曲率继续用预先固定的两个全输入方向探针验证 $v^T J_D-J_Dv=0$；它检验完整切面的 Hessian 对称性，不声称恢复逐 $i,j$ 曲率张量。

## 3. 验证与成本门

- 两个注册模型各使用 discovery `split_order=0`，ground-truth raw logit。
- block 0 验证 Endpoint 端口闭合、PCIA→IG、PEFA→Endpoint、$\bar T=A+\bar R$、完整切面曲率零和及适用负控制。
- 全 12 block 验证全部 9 类合法切面的逐 RGB 一阶闭合；这证明适配覆盖完整 ViT，不把单 block 冒充完整模型。
- 对 Endpoint、IG、PCIA、source-summed PEFA 测量每图逻辑 forward/reverse 次数、512 积分节点、峰值显存、RSS 与保留存储。
- 生产成本仍按 64 discovery 图、2 模型、12 block 外推；超过 16 H20-hours 即 `NO_GO_COST`。
- attempt1 通过只开放步骤 5 的申请资格，不授权步骤 5。
