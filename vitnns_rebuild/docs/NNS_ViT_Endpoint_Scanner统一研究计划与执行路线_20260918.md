# NNS-ViT Endpoint Scanner：统一研究计划与执行路线

日期：2026-09-18。本文是 Endpoint Scanner 研究从现在起的**唯一规划与执行入口**。
阅读、授权和执行后续阶段时以本文为准；早期规划、EP0、EP1、EP2-R1 及轨道拆分
文件保留为不可变证据，只在第 11 节查阅，不再要求跨文档拼接当前状态。

## 1. 当前唯一有效状态

```text
ENDPOINT_PRIMARY_TRACK_AUTHORIZED_BY_AMENDMENT
ENDPOINT-E0_SCOPE_AMENDMENT_PASS
ENDPOINT-E1_CONTRACTS_SATISFIED_BY_EP1_REBIND
ENDPOINT-E2_PASS_REGISTERED_ATLAS_AND_RESOURCES
ENDPOINT-E3_PASS_CORRECTNESS_ONLY
ENDPOINT-E4_FAIL_LOCAL_DIRECTION_CONTRACT

PATH_DEPENDENT_TRACK_BLOCKED
EP2_R1_PATH_NUMERICAL_CALIBRATION_FAILED_IMMUTABLE
ORIGINAL_EP3_NOT_AUTHORIZED

FORMAL_EVALUATION_COHORT_BOUND=400_SHARED_IDS
```

当前可以请求的唯一下一阶段是：

```text
NO_NEXT_STAGE_AUTHORIZATION
PROSPECTIVE_E4_RECOVERY_AMENDMENT_REQUIRED
```

ENDPOINT-E2 已由 R4 handoff 独立验证通过；ENDPOINT-E3 已按用户明确授权完成并由
R2 handoff 独立验证通过。ENDPOINT-E4 已按用户明确授权执行，但冻结的局部 finite-eta
合同未通过，阶段 fail closed。原规划中的 EP3、Path IG 恢复和 ENDPOINT-E5 均未授权。

## 2. 论文定位与核心主张

建议标题：

> **NNS-ViT: Endpoint Scanning and Structured Decomposition of Vision Transformers**

中文：

> **NNS-ViT：视觉 Transformer 内部响应的终点扫描与结构化分解**

核心定义是

$$
L_s^{\mathrm{Endpoint}}(x)=x\odot\nabla_xs(x),
$$

其中 $x$ 是模型实际接收的 normalized RGB 张量，$s(x)$ 是注册的内部或输出标量。

必须始终使用以下贡献边界：

1. Endpoint Scanner 的基础输入归因数值上等价于同一内部目标的
   Gradient × Input；不把它声明为新的基础归因公式。
2. 方法贡献是原始 NNS endpoint 原则到 ViT 的形式化适配、统一内部目标扫描、
   完整计算边切面和结构化 endpoint decomposition。
3. 数学身份、实现正确性和行为效用分开报告。闭合和局部微分合同不能替代有限
   干预证据。
4. Endpoint 不具有 IG completeness；当前证据也不支持把它写成 Path IG 的普遍
   替代品。

如果有限干预尚未完成，论文第 4 项贡献只能写成研究目标：检验结构化 Endpoint
分解能否预测匹配干预，不能预先写成已经证实。

## 3. 目标与结构分解

### 3.1 注册目标

两个冻结模型为：

| 模型 | timm 名称 | blocks | heads | 注册目标数 |
|---|---|---:|---:|---:|
| ViT-B/16 | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 12 | 12 | 457 |
| DeiT-S/16 | `deit_small_patch16_224.fb_in1k` | 12 | 6 | 385 |

目标类型固定包括：

- token-channel 单元；
- patch-channel sum；
- patch-channel energy：$\frac12\sum_{t=1}^{196}h_{t,c}^2$，排除 CLS；
- attention head 内部坐标；
- ground-truth raw logit，保留错分图；
- frozen clean top-1 raw logit，作为次级目标；
- final CLS 与冻结的中间模块目标。

Margin 在当前正式协议中不计划。不同模型同编号 channel 不作语义等同解释。

### 3.2 结构分解

对最终标量目标 $F$ 和注册成分 $z_k$：

$$
E^C_{k,i}
=
x_i\left\langle\nabla_{z_k}F(x),
\frac{\partial z_k(x)}{\partial x_i}
\right\rangle.
$$

若计算边切面完整、无遗漏、无重复：

$$
\sum_kE^C_{k,i}=x_i\frac{\partial F(x)}{\partial x_i}.
$$

固定分解包括：

| 切面 | 成分 | 重构对象 |
|---|---|---|
| head branch | 全 heads | $A_F$ |
| head total | identity + 全 heads | $L_F$ |
| attention differential branch | Q + K + V，或 routing + Value | $A_F$ |
| attention differential total | identity + Q + K + V，或 identity + routing + Value | $L_F$ |
| first residual | identity + attention branch | $L_F$ |
| second residual | carry + MLP branch | $L_F$ |

Canonical backend 是 timm 0.9.16 standard Block/Attention 的显式 unfused
实现。Fused kernel 只做 forward parity，不宣称结构 edge 支持。

## 4. 已完成证据

### 4.1 EP0：协议冻结完成

- 数据 split：discovery/calibration/evaluation = 64/64/400，源图像身份互斥；
- 两个模型共享相同 400 张 evaluation ID；
- C4a/C4b 共 16 个确认性 family、28 个 Holm tests；
- 模型、预处理、目标、cut、reference、统计阈值、NA、资源和访问边界均已冻结。

### 4.2 EP1 / ENDPOINT-E1：实现与回归完成

正式状态：`PASS_EP1_IMPLEMENTATION_AND_REGRESSION`。

- CPU 回归 44/44；
- paper-formula CNN 覆盖 Linear/Conv、ReLU 开关、唯一 MaxPool winner、bias
  状态作用、filter energy 和 residual；
- ViT 覆盖 patch embedding、LayerNorm、GELU、Q/K/V、row softmax、multi-head
  concat、output projection、两次 residual、final LayerNorm 和 classifier；
- 两模型全 12 block 覆盖；
- 单卡/4 卡各 8/8 task，逐 task numeric digest 一致；
- fused/unfused logit 最大绝对差 `8.106231689453125e-06`；
- hook forward 最大差 0；
- cut closure 最大绝对差 `1.2665987014770508e-07`，最大 relative L2
  `1.6204431858568569e-06`。

因此 ENDPOINT-E1 无需重跑，由 EP1 R11 直接重绑定满足。

### 4.3 EP2-R1 / P-IG2：Path 数值校准失败

EP2-R1 使用 64 张 calibration 图、两模型、ground-truth 与 frozen top-1 两目标，
完整执行 128/128 tasks，全部 finite。预注册门要求每模型每目标至少 61/64 图通过。

| 模型 | 最大候选 M=512 的 GT | 最大候选 M=512 的 top-1 | 结果 |
|---|---:|---:|---|
| DeiT-S/16 | 61/64 | 62/64 | 选择 M=512 |
| ViT-B/16 | 41/64 | 40/64 | 未通过 |

正式状态保持 `EP2_NUMERICAL_CALIBRATION_FAILED`。这不是 Endpoint 科学失败、
代码失败或基础设施失败，只是 ViT-B 的冻结 Path IG 候选范围没有达到预设数值
覆盖率。R1 不覆盖、不重解释、不追加 M、不放宽阈值。

## 5. 当前双轨结构

### 5.1 Endpoint 主轨

```text
ENDPOINT-E0  范围修订与协议冻结                    PASS
ENDPOINT-E1  数学、算子、闭合与工程合同            PASS，继承 EP1
ENDPOINT-E2  注册目标 atlas 与规模/资源审计         PASS
ENDPOINT-E3  正式 cohort 全 block 结构闭合          PASS
ENDPOINT-E4  局部合同与有限 activation replacement FAIL，局部合同未通过
ENDPOINT-E5  空间定向干预与 mandatory baselines     未授权，受 E4 阻塞
ENDPOINT-E6  统计分析与论文 release                 待前序完成
```

### 5.2 Path IG 轨

```text
P-IG1  既有 20 图边界证据               HISTORICAL_EXPLORATORY_ONLY
P-IG2  EP2-R1 两模型数值校准             FAILED_IMMUTABLE
P-IG3  可选未来恢复                       BLOCKED，需要新前瞻性 amendment
```

Path 轨失败只阻塞：

- 高精度 Endpoint–Path fidelity 比较；
- 高精度 Endpoint–Path stability 比较；
- Endpoint–Path 成本—质量结论；
- 任何跨模型 Path IG 普遍结论。

它不阻塞 Endpoint atlas、Endpoint 结构闭合或 Endpoint-only 干预。

## 6. Endpoint 主轨执行步骤

所有阶段均须满足：前一阶段正式 terminal 可验证、本地 HEAD 与 tracking/remote
一致、代码和协议已在访问前发布、GPU 资格与现有 PID 已记录、动态全局队列、
不可变结果、exact coverage audit、资源 cap 和 fail-closed 出口。每个阶段必须单独
明确授权，不能用“主轨已开放”代替阶段授权。

### ENDPOINT-E2：注册目标 atlas 与规模审计

目标：回答 RQ1——能否稳定、完整地扫描冻结的 ViT 内部目标，并量化实际规模。

输入与权限：

- 原 EP0 冻结的 400 张 evaluation 图像，两个模型共享 ID；
- 只允许 canonical Endpoint $x\odot\nabla_xs(x)$；
- 不运行 Path IG、input fidelity/stability 或有限干预；
- 不输出 C4 utility 结论。

工作：

1. 物化并逐文件哈希 400 个冻结 tar member，不允许换图；
2. 两模型严格加载并绑定 checkpoint、tensor digest 和 preprocessing；
3. 运行 457/385 个注册目标，覆盖所有 block、单元、sum、energy、head 和类别目标；
4. 所有图保存 target summary；原始 signed RGB class maps 全部保留；前 16 个冻结
   evaluation ID 保存完整 atlas；前 8 个按冻结顺序生成 gallery；
5. 原始 `[C,H,W]` 先计算 signed sum 与 absolute mass，再做显示聚合；显示图不进入统计；
6. 报告 targets/sec、images/sec、P50/P95/max latency、peak allocated/reserved memory、
   stored bytes/target、完整 atlas 摊销成本、wall time 和 GPU-hours；
7. 一卡与多卡必须产生相同 task set、seed 和容差内相同数值；遗漏、重复、nonfinite、
   source drift 或超 cap 均保留现场并 fail closed。

验收：

- 400 × 2 图像—模型任务全部有唯一 terminal；
- 注册目标覆盖分别为 457/385，无静默缺失；
- 所有原始数值有限，target identity、模型和输入 hash 完整；
- 固定 gallery/near-zero/failure 索引完整；
- 资源低于 EP0 原 EP3 cap：64 GPU-hours、72 wall-hours、128 GiB 增量 retained；
- 只允许作 `C2_REGISTERED_SCAN_COMPLETION_AND_RESOURCES` 判定。

通过出口：`ENDPOINT_E2_PASS_NEXT_ENDPOINT_E3_REQUIRES_EXPLICIT_REQUEST`。

#### ENDPOINT-E2 完成记录

正式 4 卡动态队列已完成 400 张 evaluation 图像 × 两模型共 800/800 tasks；
ViT-B/16 与 DeiT-S/16 分别完整覆盖 457/385 个 target，共 336,800 个有限结果。
保留 800 份双 class raw signed RGB maps、前 16 evaluation ID × 两模型的 32 份
full raw atlas，以及固定 first-8 gallery。formal wall time 为 1,752.71 秒，按四卡
上界计 1.9475 GPU-hours，retained bytes 约 9.29 GB，均低于冻结 cap。

用户在 formal 800/800 完成后取消了冗余的全量单卡重放；部分 replay 保留并明确
排除在 E2 PASS 和科学证据之外。R1–R3 handoff 的 manifest/provenance 失败均保留为
非权威证据；权威 R4 terminal 位于
`reports/nns_vit_endpoint_e2_atlas_r4_20260918/terminal_receipt.json`，状态为上述通过
出口。E2 没有运行 Path IG、fidelity、stability 或 intervention，也没有作 utility
结论。

### ENDPOINT-E3：正式结构闭合

目标：回答 RQ2——注册的结构切面在正式 cohort 上是否无遗漏闭合。

工作：对两个模型所有 12 blocks 验证 head、Q/K/V、routing/value、attention
residual 和 MLP residual，分别按 `A_F` 与 `L_F` reconstruction ID 比较。故意遗漏
非零 head/routing 或重复 residual 的负例继续必须失败。

验收：使用 EP0 float32 absolute/relative tolerance；所有 block、所有预定切面、
所有必需样本有限且闭合。结果只支持 correctness，不能替代 utility。

#### ENDPOINT-E3 完成记录

正式 8 卡动态队列已完成 400 张 evaluation 图像 × 两模型共 800/800 tasks；全部
12 blocks × 8 cuts 共 76,800 个正向闭合检查通过，12 blocks × 3 个 checker 负例
共 28,800 个均按预期失败。最大 relative L2 为 `1.4144569831842091e-05`，满足
冻结 `rtol=1e-4`。前 16 evaluation ID × 两模型的 384 个 raw block arrays 已由
CPU-only R2 handoff 从原始 component maps 独立重算通过；R2 未重跑 GPU，也未覆盖
R1 terminal。权威 terminal 位于
`reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json`。本结果只支持
correctness，没有运行 Path IG、fidelity/stability、activation replacement 或 utility。

### ENDPOINT-E4：局部合同与有限 activation replacement

目标：回答 RQ3 的第一部分——结构成分能否预测匹配的有限 activation effect。

局部 correctness：

$$
a_k=\sum_iE_{k,i}
\approx
\lim_{\eta\to0}\frac{F(z)-F(z-\eta J_{z_k,x}x)}{\eta}.
$$

固定 $\eta=0.01,0.003,0.001,0.0003$。它只验证微分实现。

有限 utility：

$$
e_k=F(x)-F(x;z_k\leftarrow z_k^{\mathrm{ref}}),
$$

并比较

$$
a_k\leftrightarrow e_k,\qquad
m_k\leftrightarrow |e_k|,\qquad
p_k^{\mathrm{ref}}=\langle\nabla_{z_k}F,z_k(x)-z_k^{\mathrm{ref}}\rangle.
$$

若 $p_k^{\mathrm{ref}}$ 优于 $a_k$，如实报告。这不否定 Endpoint 数学身份。

#### ENDPOINT-E4 完成记录

R1 在发布后的正式根上启动 800 个 image-model task。前 16 个 DeiT-S evaluation
任务均在两次 attempt 中触发 `LOCAL_DIRECTION_CONTRACT_FAILED`，随后停止队列；状态为
16 failed、73 completed、8 claimed、703 pending。独立诊断确认解析 JVP 与 $a_k$ 在
float64 下约 $10^{-15}$ 一致，但固定四个 $\eta$ 的有限扰动曲线仍有成分超过冻结
$10^{-4}+0.01|a_k|$ 门，因此不是可由单纯提高精度修复的实现误差。

权威状态为 `ENDPOINT_E4_FAIL_LOCAL_DIRECTION_CONTRACT`。73 个部分 finite-replacement
结果不作科学解释，不运行 bootstrap、Holm 或 C4a claim decision。E3 correctness 不受
影响；E5 未启动，任何 E4 恢复都须另写前瞻性 amendment 并使用新根。

### ENDPOINT-E5：空间定向干预

目标：检验完整输入分辨 component map 是否比普通标量或简单图提供额外信息。

固定使用 block 5、每模型三个预注册 head、4×4 个 56×56 区域、预算
$q=0,1/16,2/16,4/16$、normalized-zero 与 blur 两个 reference。主排序是 component
map 的 RGB-level absolute mass。Mandatory baselines 为：

- total Endpoint/G×I；
- raw input gradient；
- regional activation-space Taylor；
- 20 个冻结随机区域排列。

不得根据有限效果选择区域。C4b 的“空间信息增益”只有在四个 mandatory baseline
的预注册配对增量均满足阈值和 Holm 检验时才可声明。

### ENDPOINT-E6：统计与论文发布

统计单位始终是图像，不把 head、block 或随机重复当独立样本。保留原 EP0 的
16 个 C4a/C4b families、28 个 Holm tests、`n_valid_min=360`、效应阈值、CI 宽度、
NA 和 deterministic exit rules。每个 family 依次判定：

```text
INVALID → INCONCLUSIVE → SUPPORTED → NOT_SUPPORTED
```

论文最终分别报告 Identity、Correctness、Utility；负结果、失败样本和数值边界不删。

## 7. 全程禁止项

- 覆盖、修改或把 EP2-R1 解释成 PASS；
- 未有新 amendment 就恢复 Path IG；
- 改 Path 阈值、61/64 门、删除难样本或只保留 DeiT-S；
- 启动原 EP3；
- 把 Endpoint 写成不同于 Gradient × Input 的新基础公式；
- 把闭合或局部方向一致写成有限干预有效性；
- 在 ENDPOINT-E2 运行 Path、fidelity、stability 或 intervention；
- 在对应阶段显式授权前读取新的正式 outcome；
- 使用显示归一化图代替原始 signed RGB attribution 做统计。

## 8. 资源与执行通则

- 可用 GPU 按 H20、空闲显存 ≥32768 MiB、利用率 ≤10% 选择；记录已有 PID，
  不终止他人进程；弹性使用 1–8 卡。
- Torch/CUDA import 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；关闭 TF32、
  autocast 和 stochastic operators，记录 Python realpath 与运行时版本。
- 一卡一常驻模型 worker，使用全局动态队列和原子 claim；不使用静态 modulo 分片。
- Lease 180 秒、heartbeat 30 秒、最多 2 attempts；过期 worker 的旧 fencing token
  不得提交。
- 任务超时 1800 秒；单进程 GPU memory cap 24 GiB、CPU RSS 32 GiB；磁盘至少
  保留 50 GiB。
- 触 cap 时停止新 claim、保存完成集和缺失清单，状态为
  `PAUSED_RESOURCE_CAP`，不把有利子集当完整结果。

## 9. 论文问题与结论边界

| 研究问题 | 证据 | 当前状态 |
|---|---|---|
| RQ1：能否忠实、可审计地扫描内部目标？ | 注册覆盖与 atlas/资源 | ENDPOINT-E2 PASS |
| RQ2：结构分解能否闭合？ | 全 block edge-cut closure | ENDPOINT-E3 PASS，correctness only |
| RQ3：成分能否预测匹配干预？ | 局部合同 + finite activation replacement | ENDPOINT-E4 局部门 FAIL；utility 未判定 |
| RQ4：空间图是否有额外价值？ | 同预算空间干预及 mandatory baselines | ENDPOINT-E5 未授权且受阻 |
| Path IG 是否是稳定主对照？ | EP2-R1 数值校准 | ViT-B 未通过，Path 轨 blocked |

建议论文贡献写法：

1. 形式化 NNS endpoint 与 Gradient × Input 的关系并适配 ViT；
2. 建立统一内部目标注册与扫描接口；
3. 构造 head/residual 完整 edge cuts 和 routing/value、Q/K/V 微分分解；
4. 以局部与有限干预评价其行为预测能力——只有实验完成后才能写成已证实结果；
5. 公开方法边界：无 IG completeness，且不声称普遍替代 Path IG。

## 10. 下一步怎么执行

当前不要执行原 EP3，也不要恢复 Path IG。ENDPOINT-E4 已按明确授权执行并在冻结的
局部 finite-eta 合同处失败；不得启动 ENDPOINT-E5。若要恢复 E4，须先创建前瞻性
amendment，冻结新的数值验收方案并使用新结果根。

若用户未来要求恢复 Path 轨，必须先创建新的 `P-IG3` 前瞻性 amendment；不得
复用本 Endpoint 主轨授权。

## 11. 证据索引

以下文件保留作追溯，不再作为当前规划入口：

- [早期论文思路与 EP0–EP6 计划](NNS_ViT_Endpoint_Scanner论文思路与规划与步骤_20260918.md)；
- [EP0 冻结协议](NNS_ViT_Endpoint_Scanner_EP0协议冻结_20260918.md)；
- [EP1 实现与回归](NNS_ViT_Endpoint_Scanner_EP1实现与回归_20260918.md)；
- [EP2-R1 校准结果](NNS_ViT_Endpoint_Scanner_EP2校准结果_20260918.md)；
- [Endpoint/Path 轨道拆分修订](NNS_ViT_Endpoint_Scanner主轨拆分修订_20260918.md)；
- [ENDPOINT-E2 执行合同](NNS_ViT_Endpoint_Scanner_ENDPOINTE2执行合同_20260918.md)；
- [ENDPOINT-E2 取消全量单卡重放修订](NNS_ViT_Endpoint_Scanner_ENDPOINTE2执行修订R1_20260918.md)；
- [ENDPOINT-E2 权威结果](NNS_ViT_Endpoint_Scanner_ENDPOINTE2结果_20260918.md)；
- [ENDPOINT-E3 执行合同](NNS_ViT_Endpoint_Scanner_ENDPOINTE3执行合同_20260918.md)；
- [ENDPOINT-E3 R2 封存恢复](NNS_ViT_Endpoint_Scanner_ENDPOINTE3封存恢复R2_20260918.md)；
- [ENDPOINT-E3 权威结果](NNS_ViT_Endpoint_Scanner_ENDPOINTE3结果_20260918.md)；
- [ENDPOINT-E4 执行合同](NNS_ViT_Endpoint_Scanner_ENDPOINTE4执行合同_20260918.md)；
- [ENDPOINT-E4 权威结果](NNS_ViT_Endpoint_Scanner_ENDPOINTE4结果_20260918.md)；
- [方法数学分析](NNS_ViT_Endpoint_Scanner数学分析与方法对比_20260918.md)；
- [实现与使用说明](endpoint_nns_vit_scanner_20260918.md)。

机器可读权威仍由各自 immutable manifest/receipt 提供；本文统一解释它们当前的
生效关系，不改写任何历史结果。
