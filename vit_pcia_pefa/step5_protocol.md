# Endpoint–PCIA–PEFA 步骤 5 pilot 协议

- 状态：FROZEN_PROSPECTIVE
- 运行根：vit_pcia_pefa/step5_pilot_r8_20260921/
- 恢复血缘：r7 按精确 seed 全量运行，14/16 个任务通过；`vit_b16__02` 和 `vit_b16__05` 的 float32 Omega 零场审计确定性失败。固定 seed 诊断证明同一对象在 float64 下以原容差恢复到约 `1e-13`，故 r8 只修订数学审计精度，继承 14 个 canonical-valid 结果并重算 2 个缺失任务。
- 范围：步骤 5 小规模 pilot；不授权步骤 6 或 evaluation split。

## 1. 固定对象

pilot 使用 calibration split_order=0..7 的前缀，共 8 张图、2 个冻结模型、ground-truth raw logit。主参考为 normalized zero，主路径为直线；raw-RGB black reference 与先输入象限 0/3、再输入 1/2 的分段直线路径只作分组敏感性分析。积分固定为 512 点 Gauss–Legendre，Integrated Hessians 的 4×4 区域矩阵固定为递归 IG 的 16×16 节点。forward 为 float32，积分与汇总为 float64。

表征固定在 blocks 0/5/11，主切面为 HEADS_WITH_BYPASS_FULL。报告 D/Q/R/Omega 的 signed mass、absolute mass、抵消率、HHI 路由集中度与最坏闭合样本，并按 residual、attention、Q/K/V、head、projection、MLP 注册类别报告。

r8 每任务 seed 固定为 `uint64_be(SHA256("ENDPOINT_PCIA_PEFA|step5|task_id")[:8])`；随机基线、classifier randomization、RGB 噪声与逐 block Omega 方向均由该 seed 按协议字段确定。每个任务保存 Python、NumPy、Torch CPU/CUDA 的执行前后 RNG state，以及 driver、GPU UUID、环境版本、freeze-time git commit/dirty snapshot、checkpoint 与输入哈希。

逐端口 Omega 科学分数仍由 float32 checkpoint/输入计算并原样报告。数学零场硬门改由独立 float64 审计承担：对代数等价的完整切面 `identity + attention = identity + sum(heads)` 直接求 curl，同时独立求 full-gradient curl，并要求两者各自为零且相互一致。容差仍为 `atol=1e-5, rtol=1e-4, near_zero=1e-6`。该修订不改变科学分数或 claim 阈值，只避免 float32 大项相消伪失败。

## 2. 贡献与干预

输入候选四象限固定比较 Endpoint、IG、conductance/PCIA、4×4 Integrated Hessians、PEFA-T、PEFA-R、确定性随机和输入范数。输入干预把每个象限替换为 normalized zero；端口干预在 block 5 的 attention projection 前逐 head 置零。真实 outcome 分别为 Endpoint 图 L1 变化、目标 logit 绝对变化，以及输入干预后的端口路由 total variation。

所有秩相关在单图内部先计算，再以 image 为统计单位聚合。相关性读出与实际 perturbation/ablation outcome 分字段保存，不把梯度分数本身称为因果效应。

## 3. 稳定性与判定

固定检查 target、reference、path、classifier randomization 和 RGB Gaussian sigma=0.01 小扰动。不同 reference 信息条件分开报告。

步骤 5 的五项 claim 分别判定：数学闭合、负控制检出、成本、相对简单基线增量、预注册干预预测力。只有五项均为 SUPPORTED 才通过 PEFA pilot 升格门。有效执行中任一实证项不获支持，则终态为 NOT_SUPPORTED_EMPIRICAL_VALUE；不会换指标、删样或扩大样本。

## 4. 资源与边界

最多选 4 张满足空闲显存不少于 32768 MiB 且利用率不高于 10% 的 H20；动态全局 SQLite 队列、1800 秒 lease 和 task timeout、最多 2 attempts。上限为 64 H20-hours、72 wall-hours、128 GiB 保留文件、40 GiB/worker GPU memory、32 GiB/worker RSS。记录已有 PID，不终止其他进程。

步骤 6 始终保持 step6_started=false；即使 promotion gate 通过，也需要新的用户明确授权。
