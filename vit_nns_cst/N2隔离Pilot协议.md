# NNS-CST N2 隔离 Pilot 协议

日期：2026-09-22  
状态：`FROZEN_BEFORE_MODEL_OUTCOME_ACCESS`  
授权：当前线程用户明确要求执行《执行规划.md》的 N2；N3 未授权

## 1. 目标与边界

本阶段只判断测量是否可行。它不产生 confirmation 证据，也不支持语义机制主张。N2 只读取既有 `pilot` split；所有被解码的 recipient、donor 及其派生版本永久登记为 `pilot_discovery/outcome_exposed`，不得进入 validation、confirmation 或 replication。既有 `confirmatory_holdout` 的图像、mask 和 outcome 均不得读取。

前驱必须是 `evidence/n1_r3_20260922/terminal_receipt.json` 的 `PASS_N1_ANALYTIC_CONTRACT_AND_SKELETON`，并由当前 validator 重跑确认。N2-R1 是 fresh root；其终态后不得原地修改。

## 2. 唯一语义轴和四臂构造

唯一 primary axis 是水生物体的背景上下文：`AQUATIC_CONTEXT -> NON_AQUATIC_CONTEXT`。recipient 前景、外部 ImageNet-S mask、位置、尺寸、target class 和 alpha band在四臂保持一致。每个 source 构造：

- `S0_BASE_COMPOSITE`：水域 donor 背景；
- `S1_SEMANTIC_SWAP`：非水域 donor 背景；
- `N1_WITHIN_STRATUM_SWAP`：另一个水域 donor 背景；
- `N2_RENDER_NULL`：在两个水域 donor 背景之间按冻结剂量混合。

所有 donor 在最终 224 center-crop 坐标中的外部前景 mask 必须不超过 20%，并用同一 deterministic nearest-background fill 移除；四臂使用相同 resize、center crop、mask feather、RGB 和无损 NPZ 存储。候选 donor 的语义标签由本协议的 closed ontology roster 给出，图像盲审只检查背景是否与 roster 一致，不得查看模型结果。该口径由 `N2-R1预冻结测量口径修订.md` 在任何模型 outcome 前澄清。

producer 在不知道模型 outcome 的情况下，枚举有限候选，从低级差异向量

[
L=(L_1, L_2, \Delta RGB, \Delta highpass, \Delta boundary)
]

选择 semantic/null 强度最接近的组合。选择目标、tie-break、donor 最大复用两次和 render-null 剂量网格均在 `config/n2_r1/pilot_contract.json` 冻结。每个 source 的任一 ratio 超过 2.5 或 aggregate log-ratio distance 超过 1.25 即 unmatched；unmatched rate 大于 25% 时终态为 `NO_GO_NOT_IDENTIFIABLE`。不得根据 logit 或 activation 改 donor、caliper 或 source。

## 3. 输入行为门

primary 模型固定为 `vit_base_patch16_224.augreg2_in21k_ft_in1k` 和已登记 checkpoint。行为量固定为 target class raw logit。对 source (i)：

[
D_i^{sem}=F_{c_i}(S0_i)-F_{c_i}(S1_i),\qquad
D_i^{null}=F_{c_i}(S0_i)-F_{c_i}(N1_i),
]

[
B_i=D_i^{sem}-|D_i^{null}|.
]

行为门要求 `median(B_i) >= 0.10 logit` 且至少 6/8 个 `B_i > 0`。render null 只作伪影诊断。门失败立即终止为 `NO_GO_NO_BEHAVIOR`，不运行内部搜索。

## 4. 有限子空间与 readout pilot

只注册 block 2、5、8、11 的 output residual stream；token topology 只保留显式 `CLS` 和 `PATCH_MEAN_14X14`，不做隐式 resize。候选 rank 仅为 1、2、4，ridge 仅为 0 和 1e-6。主估计必须使用 matched-null nuisance residualization；不做跨 source 中心化，跨 source 中心化仅作为不具备选择资格的数值诊断。

选择规则按顺序是：held-out paired sign accuracy、signal/control ratio、leave-two-sources-out projector stability、较浅层优先、较低 rank 优先。readout 只在 pilot source 的确定性 train folds 拟合并序列化；pilot held-out 只用于设计估计。必须同时报告 single-head proxy、uncontrolled PCA、same-rank random subspace 和同容量 linear probe。

数值门：所有数组 finite；basis rotation 后 projector 最大误差不超过 1e-8；NumPy float64 独立重算 projector 误差不超过 1e-7；leave-two-out median projector similarity 至少 0.35；selected subspace 的 signal/control ratio 至少 1.10。合理候选全部失败时终态为 `NO_GO_SUBSPACE_UNSTABLE`；NaN、Inf 或恒等式失败为 `INVALID_NUMERICAL`。

## 5. NNS、传输和干预 pilot

- Endpoint 只在两个 source 上测试 frozen readout 的连通性和成本；
- Path NNS 仅在一个 source 上用 8/16/32 节点测试收敛，reference 固定比较 zero-input 与 blurred-input；32 对 16 的 relative L1 不得超过 0.20，completeness relative error 不得超过 0.05；
- block output residual stream 是本 pilot 的单张量合法 complete cut；semantic/complement 必须逐坐标重建 Gradient × Input，最大绝对误差不超过 1e-6；
- selected block 的 attention、MLP 和 residual 用 JVP 与中心有限差分比较，relative error 不超过 0.05；
- paired injection/erasure 剂量固定为 0.25、0.5、1.0，并比较 same-rank random 和 orthogonal-complement controls；本 pilot 只检查执行、方向和有效性，不把效应当作 confirmation；
- 干预位移除以 discovery paired projected-shift 的 95% 分位数，大于 3.0 即 `OFF_MANIFOLD_INTERVENTION`，该剂量不得科学解释。

逐任务记录 wall time、GPU time、峰值显存、输出字节和失败状态。运行前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，关闭 TF32，执行 matmul 和 backward startup probe。

## 6. 终态和停止边界

终态只能是：`GO_PROTOCOL_FREEZE`、`NO_GO_NO_BEHAVIOR`、`NO_GO_NOT_IDENTIFIABLE`、`NO_GO_SUBSPACE_UNSTABLE`、`BLOCKED_INFRASTRUCTURE` 或 `INVALID_NUMERICAL`。优先级依次为数据/可识别性、行为、数值、子空间、预算。

N2 完成后必须签发独立 validation、prompt-to-artifact completion audit 和 terminal receipt。无论终态为何都停在 N3 显式授权门，不生成 validation/confirmation cohort，不修改 concept、layer、rank、readout、dose 或阈值。
