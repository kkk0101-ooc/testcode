# 阶段 D：fidelity-only 主实验 step-by-step 执行与跨对话交接文档

> 创建日期：2026-08-25。
> 文档性质：阶段 D 的路线图、任务状态表、数据访问边界和跨对话追加式交接日志。
> 当前状态：阶段 A、B 已完成；Stage C/C0–C10 的执行和证据封存已完成，但 Stage C acceptance 为 `BLOCKED`，C10 v1 primary decision 为 `NO_GO_NOVELTY`。D0 首次尝试因模型/推理与非-reserve JPEG decode 保持 `BLOCKED`；独立 model-free/no-JPEG 重试已 `PASS`，只签发 `APPROVED_TO_PREPARE_STAGE_D`。Stage D 正式执行仍未授权、未启动。
> 当前 next task：`D1`。
> 使用规则：每个新对话必须先完整阅读本文、`docs/阶段C-step-by-step.md` 的 C10 completion record 和上一条 Stage D completion record；一次只执行一个 `D_i` 或一个明确编号的 `D8-Sxx` shard。未更新本文、未追加 completion record、未给出下一任务完整提示词的任务，不算完成。

## 1. 为什么需要这份文档

Stage D 同时包含研究范围修订、500 图样本封存、机器可执行合同、GPU 批量运行、证据封存、统计决策和论文图表。把这些工作放在一个对话中会造成三个风险：

1. 在看到 Stage D 结果后继续修改主指标、阈值、样本或图表选择；
2. 把代码实现、正式执行和统计读取混在一起，难以确认结果由哪个 clean commit 产生；
3. 长时间 GPU 任务中断后，无法区分可恢复的基础设施中断与不可重试的科学失败。

本文把 Stage D 拆成 `D0–D10`。其主链是：

```mermaid
flowchart LR
    D0[范围决议] --> D1[500图样本封存]
    D1 --> D2[协议与统计冻结]
    D2 --> D3[配置/计划/授权实现]
    D3 --> D4[runner与分片实现]
    D4 --> D5[finalize与验证实现]
    D5 --> D6[统计与论文导出实现]
    D6 --> D7[smoke/预算/执行授权]
    D7 --> D80[D8.0 初始化正式root]
    D80 --> D8S[D8-Sxx 逐shard执行]
    D8S --> D9[封存与独立验证]
    D9 --> D10[统计决策与论文包]
```

其中 `D0–D7` 必须在不访问 Stage D outcome 的条件下完成。`D8` 只生产原始 evidence，不做聚合。`D9` 只封存和验证。只有 `D10` 可以生成唯一正式 Stage D 科学决策与论文结果图表。

## 2. 不可改写的上游事实

### 2.1 C10 的历史状态

Stage D 是查看 C10 后新建的、前瞻性的 fidelity-only 研究范围，不能回写或重新解释 C10。以下事实永久保留：

- C10 v1 primary decision：`NO_GO_NOVELTY`；
- C10 v1 Stage C acceptance：`BLOCKED`；
- 唯一 workflow blocker：`HOLM_BLOCKED_UNDERSPECIFIED`；
- C10 spatial：clean joint spatial 和 Gaussian stability 通过，指定的 4-pixel translation 与 center-crop/resize 为 `FAIL_STABILITY`，overall 为 `FAIL_SPATIAL`；
- C10 `downstream_authorized=false`，`stage_d_started=false`；
- Stage D 的新授权不能把上述任一字段改成 `GO`、`PASS` 或 `true`。

与此同时，C10 中预注册的 fidelity component 是有效证据：

- primary：`(block=0, head=9)`；
- confirmatory cohort：15 张固定、历史文件名等间隔抽取且非类别均衡的 holdout；
- comparison：NNS fixed-clean-attention head-direction Value-path IG 对 matched fixed-clean-attention patch-token contribution；
- endpoint：fine-4×4 `delta_fid`；
- `PASS_VALIDITY`：15/15 valid；
- `PASS_EFFECT`：median `1.7534972022995547`，positive `15/15`，95% CI `[1.3317595184568727, 3.3813212483296478]`。

因此不增加 C+ 或另一轮小样本 fidelity 准入试验。Stage D 本身是论文级规模化复制与外推；它若失败，会限制更广泛的论文主张，但不会抹掉 C10 对那 15 张固定 holdout 的 component-level 结果。

### 2.2 C10 正式证据绑定

| 对象 | 冻结值 |
|---|---|
| C10 formal root | `/data00/home/yeziqi.831/testcode/vitnns_c10_formal_92ad813_20260824` |
| C10 aggregation producer | `92ad81342ff22e2710df495a9961a3ab8c015a3a` |
| archive ref | `refs/archive/c10-aggregation-producer` |
| C10 completion commit | `69e1fa887862e00da4441842c6de149fe2424225` |
| Stage C report SHA-256 / canonical | `9a16034d29bc87eaba852aa0e618928ac0e4ecee9bf745d765f42311dd4563b4` / `90232df2010f6a8fc771baeac3aaac44fee93d436597cf0cbdebe3cf8d2142e2` |
| decision SHA-256 / canonical | `07fc5875e46018adf5dc0cd09967e273df3b443c1deeb1653d672d05103e513e` / `60e112d50f22229474a634039ba71c7eed1adba081d8e1097f850a8a4b4d8dc0` |
| provenance SHA-256 / canonical | `4dcca54fdc83ba7401070ef843636b5e67b68a8ce720fa70f729e407fc5717f0` / `4b26452041685cbb6c12d93139736e65fc4330fd44a64bf854848e7cff66c742` |
| manifest SHA-256 / canonical | `507ca3c4ac210882a5918cc602cf7a4656aa398af76ff9f3638a77f3af42a0d8` / `c71ae7c19941eab556a0ed9fdde553cb1a3fae914e149652df9b31cf3538720a` |
| status SHA-256 / canonical | `192d383d5c4788f612018532074c06d48511513edd7565dbb99934706cd96db2` / `d08bcb62a0fa95fef81c75d955b88082a8e6788a76b1a0bfffe1aa4e0181eb87` |
| whole-root inventory digest | `382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4` |

任何 Stage D authorization 必须从 sealed root 重新验证这条链，不能只相信本文抄录值。

### 2.3 现有 500 图候选 reserve

工作区中已有 metadata-only 候选材料 `manifests/stage_c_plus_v2/`，其中：

- `stage_d_v2_reserve.json`：500 rows，file SHA-256 `1432183a0ce9c701e60cc28acc3596fe16bb253dfc013126bf3168b5af05099a`，canonical digest `2e66fa2426fb7c42caefe7907a6ef1ec9a42adb1788528d59af9b5354eb0a99f`；
- ordered sample IDs digest：`b187ff5cbacc1b595f06d12e309602b09965e01e7ee1461fd8a694354d4b8f50`；
- `stage_d_v2_reserve_access_log.json`：file SHA-256 `75a319820db12102dde22308ab2e12196e0e1ca6fcdd870e2943d22a0643a621`，canonical digest `41478eee77f708a1158142d700fdfd65503b120aa672400af01fe14f9eb317fa`，当前 `forbidden_access_detected=false`。

这些文件目前被 `.gitignore` 忽略，只能作为 D1 的候选输入，不是 Stage D 权威 registry。D1 必须在 metadata-only 边界内重新验证并生成 tracked、内容寻址的 Stage D manifest。所谓“未见”只能精确表述为“未出现在已声明且封闭的 prior-access ledger 中”，不能声称绝对没有任何历史人工接触。

## 3. Stage D 要做什么

### 3.1 唯一 primary 科学问题

Stage D 只检验：

> 在冻结模型、预处理、固定 clean-attention head-direction Value-path target、primary `(0,9)` 和相同 intervention schedule 下，NNS 的 fine-4×4-cell fidelity 是否优于解释同一标量的 matched patch-token baseline，并能否在 500 图候选 reserve 正式化后的 cohort 上复制。

这里的 `fine-4×4-cell` 是每个 `16×16` patch 内 16 个非重叠 `4×4` cell，不是原生单像素 attribution。

### 3.2 最小正式 workload 草案

以下是 D2 应冻结的目标设计；D2 可在不访问 outcome 的前提下因实现合同发现必要问题而修正，但必须记录理由。D2 完成后不得根据运行结果改变：

- 500 张正式样本；
- 固定一个 primary `(block=0, head=9)`；
- 仅两种同目标方法：NNS 与 matched patch baseline；
- 仅 fine-4×4 fidelity endpoint；
- insertion/deletion，各 21 个 fraction；
- 5 个配对 tie repeats，只在图内先聚合，不增加统计 `n`；
- `run_stability=false`；
- 不运行 random/center sanity ranking；
- 不运行 patch-scale fidelity endpoint、其他 winner、其他层头、class-logit 方法或 display pipeline；
- 预注册 `audit_image_count=5` 个 audit overlay 样本，固定 `audit_repeatability_runs=2`（含 primary clean run，共两次），仅用于 repeatability/L2，不增加统计 `n`。

若按此设计展开，预期为：

- `500 × 2 = 1,000` immutable execution units；
- 1,000 个 clean maps；
- 5 audit images × 2 methods × (`audit_repeatability_runs - 1`) = 10 个 repeat-clean maps；
- `1,000 × 2 directions × 5 repeats × 21 fractions = 210,000` fine fidelity curve points；
- map work items 为 `1,000 + 10 = 1,010`，总 work items 预计为 `211,010`。

D2 必须由 typed plan 独立重算这些计数。若不一致，应在任何正式图像访问前修正计划和本文；不得在结果产生后以“实现方便”为由改 workload。

每图先分别对 NNS/matched、insertion/deletion 的 5 个 tie-repeat AUC 取算术均值，再计算唯一 `delta_fid`；`positive` 精确定义为 `delta_fid > 0`。Logical work-item count 不是 forward/backward call count，资源预算必须使用 runner 的实际调用 profile。

### 3.3 Validity、fidelity 与 cost

Validity 优先于效果统计：

- 500 个 expected sample coordinates 和 1,000 个 expected unit keys 必须全部进入 reconciliation；
- required-primary 的 missing、failed、non-finite、direction degeneration、invalidating NA、hash/provenance drift 直接导致 `NO_GO_VALIDITY`；
- 不做 complete-case analysis，不补零，不替换样本，不把失败当作负值硬塞进 median；
- 误分类样本保留在 fidelity 分母，并单独作描述性分层。

Stage D 保持 C10 的 endpoint、effect-size threshold、CI 方向和“注册的 failure condition 优先”语义。建议 D2 在 outcome 访问前冻结以下互斥、顺序判定：

1. 若 positive `<400/500` 或 paired image-bootstrap 95% CI upper `<0.02`，先判 `FAIL_FIDELITY`；
2. 否则，若 median `delta_fid >= 0.02`、positive `>=400/500` 且 CI lower `>0`，判 `PASS_FIDELITY`；
3. 其余判 `INCONCLUSIVE_FIDELITY`；CI 恰等于排他的 CI 边界不触发对应条件。

`400/500` 是 Stage D 新协议要注册的门槛，不是把 C10 的绝对 `12/15` 原样搬运。D2 必须记录 500 样本规模与 80% consistency guard 的理由，并冻结 bootstrap 次数、seed namespace、PRNG、paired median statistic、分位数算法、tie 和 invalidity 处理。

决策顺序拟冻结为：

1. validity fail → `NO_GO_VALIDITY`；
2. fidelity fail → `NO_GO_FIDELITY`；
3. fidelity uncertain → `INCONCLUSIVE_FIDELITY`；
4. cost fail → `NO_GO_COST`；
5. 全部通过 → `GO_FIDELITY_SCOPE`。

C10 的 `PASS_COST` 只是旧 8,000-unit workload 的历史可行性证据。D7 必须按新的 1,000-unit typed plan 重新测量或保守估计，并在不放宽既有上限的前提下冻结正式预算。

D2 必须把成本分成两类并冻结行为：`launch_safety_caps` 在 D7/D8 preflight 检查，失败或运行中触发即 workflow `BLOCKED`，不得继续不完整 workload；`completed_campaign_cost_gate` 只在完整 campaign 后由 D9/D10 评价，若实际完整证据超过该科学成本门可输出 `NO_GO_COST`。每个既有 cap 必须唯一归类，并定义计量单位、safety factor、超限动作和是否仍允许完成/封存，避免把安全中止误写成 `NO_GO_COST`。

### 3.4 Spatial、Holm 与禁止主张

Stage D 不运行、不分析新的 spatial/stability endpoint，只在研究设计、主结果和 limitations 中完整引用 C10 的既有通过项与失败项。任何新的 spatial robustness/equivariance 实验属于独立的后续研究，不能在 Stage D 内临时加入。

Stage D 只有一个 primary，不对三个 exploratory winners 做推断时，记录 `holm_status=NOT_APPLICABLE_NO_SECONDARY_INFERENCE`。这不修复或取消 C10 的历史 blocker。D6/D10 若对任何 secondary endpoint 输出 p-value、拒绝结论、winner selection 或正式泛化结论，必须先建立独立的 exact family 与 multiplicity contract；否则只能描述性报告。

Stage D 及论文不得声称：

- C10 已经 `GO` 或 Stage C acceptance 已通过；
- NNS 对 translation/crop 或一般 spatial robustness 已获证明；
- 4×4-cell 是原生单像素 attribution；
- fixed-attention Value-path 是完整 Q/K/V、完整 attention 或 class-logit explanation；
- 无显著差异等于等价；
- 15 图或 500 图结果证明对 ImageNet 普遍优越。

## 4. 哪个任务生成论文数据或图片

| 任务 | 是否产生正式科学数据 | 论文用途 | 不得作为 |
|---|---|---|---|
| D0 | 否 | Methods/Limitations 的范围声明和 C10 历史边界 | 新实验结果 |
| D1 | 只产生样本设计数据 | cohort flow、类别覆盖、互斥性、audit IDs；可作为 Study Design 表/流程图数据 | fidelity 效果数据 |
| D2 | 否 | estimand、workload、统计和 decision-gate 表 | 结果表 |
| D3 | 否 | 可复现性实现说明 | 科学结论 |
| D4 | 否 | runner、sharding、retention、provenance 说明 | 正式结果 |
| D5 | 否 | evidence sealing/QC 方法说明 | 正式结果 |
| D6 | 否；只用 synthetic fixtures | 预先冻结 paper-data schema、表格列和绘图 recipe | Stage D 数值或图片 |
| D7 | 只产生 diagnostic/预算估计 | Supplementary 的预运行估计和 smoke/QC 依据，必须标为 projected/diagnostic | primary fidelity 结果或 actual runtime cost |
| D8 | **是：原始主实验 evidence** | D10 所需的 L0/L1、资源和预注册 audit L2 | 中途汇总、图片或结论 |
| D9 | **是：正式 QC/validity/cost 数据** | Supplementary QC、失败/NA、repeatability 和 actual runtime cost，必须与 D7 projected cost 分列 | fidelity effect decision |
| D10 | **是：最终统计、论文数据和数值图** | 主结果表、C10/Stage D 分 cohort 图、paired-delta 图、insertion/deletion 曲线和补充表 | 事后挑选的最好/最差案例 |

D10 建议固定生成以下 machine-readable paper data 和数值图；只有 validity 通过时才生成效果表/图，否则对应 artifact 记录 `NOT_GENERATED_DUE_TO_VALIDITY`：

1. `paper_data/main_fidelity_table.csv/json`：两方法 AUC、median delta、positive count、CI、decision；
2. `paper_data/cohort_comparison.csv/json`：C10 `n=15` 与 Stage D `n=500` 分开列示，绝不池化为 `n=515`；
3. `paper_data/per_sample_delta.csv`：全部预期坐标、validity/status、误分类标记和 `delta_fid`；
4. `paper_data/resource_and_qc.csv/json`：失败/NA、repeatability、时间、显存和存储；
5. 主图 A：C10 与 Stage D 分 cohort 的 median/95% CI；
6. 主图 B：Stage D paired `delta_fid` 分布或 ECDF；
7. 主图 C：NNS 与 matched insertion/deletion curve 及描述性的 pointwise image-bootstrap band；
8. 定性 overlay 不在 D10 生成；如论文需要，Stage F 另建 display-only recipe，只能使用 D1 预注册 audit IDs，且不得进入数值统计或 decision。

数值图表脚本必须只读 D10 的 machine-readable paper data。不得从 PNG/JPEG 反推数值；不得把 C10 与 Stage D 样本合并后重新统计。

## 5. D0–D10 总路线与状态

任务状态只使用 `NOT_STARTED`、`IN_PROGRESS`、`PASS`、`BLOCKED`。`next_task` 单独记录，不使用 `NEXT` 混入状态。只有代码/文档、测试、产物验证、completion record 和下一提示词都完成后，任务才能标记 `PASS`。

| 任务 | 状态 | 唯一目标 | 允许的数据访问 | 允许产生的结论 | 前置 |
|---|---|---|---|---|---|
| D0 | PASS（首次尝试 BLOCKED；严格重试 PASS） | 冻结 post-C10 fidelity-only scope | sealed C10；reserve metadata existence only | `APPROVED_TO_PREPARE_STAGE_D` | C10 evidence sealed |
| D1 | NOT_STARTED | 正式化 500 图 registry 与 audit overlay | reserve/ledger/官方标签 metadata；禁止图像字节 | 无效果结论 | D0 PASS |
| D2 | NOT_STARTED | 冻结 protocol/config/schema/statistics/exact workload | D0/D1 与 Stage C 合同；禁止图像和 outcome | 无效果结论 | D1 PASS |
| D3 | NOT_STARTED | 实现 strict loaders、typed plan、preflight/authorization | synthetic/metadata only | 无效果结论 | D2 PASS |
| D4 | NOT_STARTED | 实现 image-group runner、shards、artifact/provenance | synthetic/mock only | 无效果结论 | D3 PASS |
| D5 | NOT_STARTED | 实现 finalize 与 source-recomputing validation | synthetic/tampered fixtures only | package/QC 状态 | D4 PASS |
| D6 | NOT_STARTED | 实现 aggregation、statistics、decision、paper export | synthetic fixtures only | synthetic decision only | D5 PASS |
| D7 | NOT_STARTED | 真实已知样本 smoke、预算、冻结 producer 并签发授权 | Stage C 已知 calibration/audit；禁止 reserve 图像 | `AUTHORIZED_STAGE_D_FIDELITY_SCOPE` 或 BLOCKED | D6 PASS |
| D8 | NOT_STARTED | 初始化并逐 shard 运行 500 图/1,000 units | 首次允许 reserve 图像、模型和原始结果 | unit terminal state；禁止科学汇总 | D7 PASS |
| D9 | NOT_STARTED | 封存并独立验证 formal evidence | D8 root；CPU-only；禁止 JPEG/模型 | evidence/QC/validity readiness | D8 `PASS` 且所有 frozen shards terminal |
| D10 | NOT_STARTED | 唯一正式统计决策与 paper artifact package | sealed D9 root、C10 reports；CPU-only | Stage D fidelity-scope decision | D9 authorizes analysis |

### 5.1 每个任务的 completion 总结与下一提示词重点

下面是第 8 节通用模板之外，每个任务必须额外写清的任务特定内容。下一提示词必须把这些已完成事实转换成精确 commit/hash/path，并重述下一任务禁止越界项。

| 当前任务 | completion record 必须重点总结 | 下一提示词必须明确授权 |
|---|---|---|
| D0 | C10 hash 复核、历史状态、scope allowlist/denylist、零 reserve outcome access | D1 只做 metadata-only registry/ledger/audit overlay，不读 JPEG |
| D1 | 500 rows、ordered digests、类别覆盖、C20 互斥、ledger 范围、5 audit IDs | D2 只冻结科学/统计/执行合同，不读图像或 outcome |
| D2 | 所有冻结定义、500/400 理由、bootstrap、retry/cost 分类、exact expected counts | D3 只实现 CPU 控制层，不运行正式或 smoke workload |
| D3 | loader/plan/authorization 命令、plan digest、tamper 测试、零图像访问 | D4 只实现 producer/sharding，不运行正式样本 |
| D4 | group ownership、cache release、retry/recovery、artifact/provenance、no-result-leak | D5 只实现并用 fixtures 测试 finalize/validate |
| D5 | package/validity 分离、reconciliation/inventory、tamper/active-claim 测试 | D6 只用 synthetic fixtures 实现分析与 paper export |
| D6 | per-sample schema、bootstrap golden、truth table、paper-data/figure recipes | D7 仅使用已知 Stage C 图或 synthetic 做 smoke、预算和授权 |
| D7 | 四个子门结果、producer/ref、exact plan/budget、authorization digest、零 reserve access | D8.0 只初始化 formal root 和验证 live assets，不运行 units |
| D8.0 | formal root、ACCESS_STARTED/runnable 状态、live-assets digest、shard ledger | 精确 `D8-S00`，只运行该 shard |
| D8-Sxx | shard receipt、ledger 转换、terminal/failed counts、attempts、无 effect access | 下一个 pending shard、同 shard retry，或 campaign completion；由 ledger 唯一决定 |
| D8 campaign | 所有 receipts、500/1,000 terminal reconciliation、scientific failures 是否存在 | D9 只 finalize/validate，不做效果统计 |
| D9 | seal/inventory、package complete、scientific validity、actual cost、analysis authorization | `analysis_authorized=true` 时 D10；否则精确诊断任务 |
| D10 | 唯一 decision、关键数值/hash、paper artifacts、允许/禁止主张和全部偏差 | 根据结果生成 Stage E 协议准备、Stage F 写作或诊断的完整提示词 |

## 6. 每个任务的具体边界

### D0：冻结 post-C10 fidelity-only 范围

目标：用新版本范围决议授权“准备 Stage D”，但不授权 Stage D 执行。

必须完成：

- 重新验证第 2.2 节 C10 sealed root、Git ref 和关键 hashes；
- 新增 `docs/stage_d_scope_decision.md` 以及 machine-readable scope/schema；
- 原样记录 C10 `NO_GO_NOVELTY`、`FAIL_SPATIAL`、`BLOCKED`、Holm blocker 和旧 `downstream_authorized=false`；
- 把 C10 15 图 `PASS_VALIDITY/PASS_EFFECT` 定义为新 fidelity-only 研究的科学依据；
- 明确 Stage D 是 prospective replication/extension，不是 C10 重判，不需要 C+；
- 冻结 claim allowlist/denylist 和 spatial mandatory disclosure；
- 只允许输出 `APPROVED_TO_PREPARE_STAGE_D`，不得输出 execution authorization。

验收：无 JPEG decode、模型、GPU、Stage D outcome access；C10 root 无任何写入；scope 文档与机器记录双向一致；测试覆盖旧 C10 状态不能被新授权覆盖。

论文贡献：研究范围、历史决策披露和 limitations 文本；无新数值或图片。

完成后：把 D0 标为 `PASS` 或 `BLOCKED`，追加 D0 completion record；PASS 时在记录中写出可直接复制的 D1 完整提示词，BLOCKED 时写诊断/修复提示词。

D0 已完成。权威机器记录为 `config/stage_d_scope_decision.json`，closed schema 为 `config/stage_d_scope_decision.schema.json`，人类可读展开为 `docs/stage_d_scope_decision.md`。D0 只允许准备 D1–D7；`execution_authorization=NOT_ISSUED`。

### D1：正式化 500-image registry 与 audit overlay

目标：将当前 ignored candidate reserve 转换为 tracked、可验证、内容寻址的唯一 Stage D registry。

必须完成：

- 只读验证 `manifests/stage_c_plus_v2/` 的 reserve、access log、ledger、source inventory 和 source spec；
- 生成独立目录（建议 `manifests/stage_d_fidelity_v1/`）及 closed schema、manifest metadata、attestation；
- 精确 500 rows，selection order 连续且唯一，`sample_id`、relative path 和 image SHA-256 唯一；官方 WNID/index 映射逐行有效，类别标签允许重复；
- 与 Stage C 20 IDs 完全互斥；不重抽、不补样、不依据类别、模型预测或结果筛选；
- 用 Stage D 专属 hash namespace 从冻结顺序预注册 5 个 audit overlay IDs；
- 明确 prior-access ledger 的证明范围；生成不可变的 pre-authorization access attestation，保持 image/result access 为 false，不回写候选 sealed access log；
- 对 `.gitignore` 只增加必要的精确 allowlist，不纳入 JPEG 或 runtime artifacts。

验收：500/500 metadata、候选 manifest 中已记录的 image hash 格式/内部 binding、官方标签链和内部 digest 全通过；D1 不读取 live JPEG 因而不声称重算其文件 hash，live-byte equality 延迟到 D8.0；tracked manifest 能从候选 source 确定性重建；audit 仍属于同一 500，不增加统计 `n`。

论文贡献：cohort flow、类别覆盖、互斥性和 audit 样本表的数据源；不产生 fidelity 结果。

完成后：追加 D1 completion record 和 D2 完整提示词。

### D2：冻结正式 protocol、acceptance、config、统计与 exact plan contract

目标：在任何 Stage D outcome 产生前，消除科学和执行自由度。

必须完成：

- 新增 `docs/阶段D_protocol.md`、`docs/stage_d_acceptance.md`、`docs/stage_d_config_contract.md`；
- 新增 closed `config/stage_d.json`、config schema、run spec/schema；
- 固定第 3 节的模型、target、primary、两方法、fine-only、curve、tie、`audit_image_count=5`、`audit_repeatability_runs=2`、retention 和禁止项；
- 冻结 `delta_fid`、validity、500/400、bootstrap、decision priority 和 equality boundaries；
- 给出 500 样本规模与 80% guard 的设计理由或精度/功效说明；
- 定义基础设施重试白名单、scientific failure 不重试、fresh-root 全量重跑规则；
- 定义 actual typed-plan cost formula，保留或收紧既有 caps，不得放宽；
- 写静态合同测试，断言预期 1,000 units、210,000 curve points、211,010 work items，并拒绝任何 spatial/sanity/patch-scale/extra-method work item。

验收：合同无待结果后决定的关键字段；所有 schema closed；无需图像、模型或 GPU；D0/D1/C10 hashes 全绑定。

论文贡献：Methods 的 estimand、工作量、统计和 decision-gate 表；无结果。

完成后：追加 D2 completion record 和 D3 完整提示词。

### D3：实现 strict loaders、typed plan、预算与 authorization

目标：把 D0–D2 合同实现为 fail-closed 的 CPU-only 控制层。

建议模块：

- `stage_d_config.py`：strict config/run-spec loader；
- `stage_d_manifest.py`：registry、ledger、互斥和 metadata validator；
- `stage_d_plan.py`：Stage D 独立 typed plan，不放宽 Stage C planner；
- `stage_d_authorization.py`：C10/source/plan/budget/code binding；
- `stage_d_cli.py`：先提供 validate/plan/estimate/prepare-authorization 控制命令；D4–D7 再依次补齐 runtime、finalize、aggregate 和 execution-authorization 命令。

必须完成：

- exact plan 为 500 image groups、1,000 units，并冻结每个 `image_group_id`、组内 NNS→matched unit 顺序和 pair binding；D7 冻结 shard count 后才生成 group→shard 映射；
- plan、unit、work-item、registry 和 config 都有 canonical digest；
- authorization builder/validator 原样保留旧 C10 状态，只能签发新 scope 的授权；
- CLI 命令分层：`plan/estimate` 验证上游合同；`prepare-authorization` 验证全部冻结准备输入但不要求已有授权；只有后续 `init/run-shard` 必须先验证 D7 execution authorization，且所有检查均发生在模型构造前；
- 用 synthetic fixtures 测试 unknown fields、duplicate keys、tamper、drift、错样本数、错方法、错 scale 和含 spatial work item 均 fail closed。

验收：纯 CPU 测试通过；没有读取 reserve image bytes；没有修改 Stage C/C9/C10 的历史语义或正式产物。

论文贡献：可复现性控制面；无正式数据或图片。

完成后：追加 D3 completion record 和 D4 完整提示词。

### D4：实现 evidence producer、image-group sharding 与恢复语义

目标：复用 Stage C 已验证的数学与 unit primitives，实现最小 Stage D producer；本任务不运行正式样本。

必须完成：

- 复用 `ProductionStageCBackend`、`run_execution_unit`、fidelity、artifact 和 resource primitives，不复制数学公式；Stage D 必须抽出 stage-neutral provenance primitive，或向 backend 注入独立的 Stage D provenance builder/validator，禁止生成仍绑定 Stage C config/schema/selection 的正式证据；
- 增加正式的 per-image cache release/context API，确保一图的 NNS/matched 完成后释放 sample/extraction/target/GPU cache；
- image at frozen selection order `i` 只属于 shard `i % shard_count`；实现原子 shard lease 或 image-group claim，只有 group owner 可以顺序执行同图两个 units，pair 状态发布后才释放 ownership，cache release 必须放在 `finally`；
- 一块 GPU 一个独立进程和 backend，禁止同一 backend 实例并发；
- 实现 init、unit claim、checkpoint、failure table、lifecycle/activity locks、closed layout 和 immutable attempt；
- 默认无 PNG/L3；L0/L1 全覆盖，L2 仅 D1 的 5 audit IDs；
- 中间 CLI/log 只能显示进度、terminal state 和资源，不显示 AUC、delta、排序或汇总；
- required scientific failure 记录后继续其他坐标；tamper、authorization drift 或 unsafe root 则 fail closed；
- 一个 method 的 scientific failure 不阻止同图另一个 method，两个 unit 都必须产生 terminal record；
- shard 必须可按冻结规则幂等重入：验证并跳过 terminal image groups，只运行合法 pending groups；半完成 pair、stale claim、process crash 和 recovery authorization 的处理由 D2 合同唯一规定；
- 重试严格服从 D2 白名单，不得按结果选择性重跑；
- 本任务为独立 CLI 增加 `init` 和 `run-shard`，但不增加正式结果读取命令。

验收：测试覆盖 shard 完整/互斥/顺序、竞争 claim、crash、checkpoint tamper、cache eviction、长序列资源不增长、pair bindings 和 no-result-leak；只用 mock/synthetic 数据。

论文贡献：运行时间、显存、artifact bytes 和 provenance 的生产能力；本任务无正式结果。

完成后：追加 D4 completion record 和 D5 完整提示词。

### D5：实现 finalize 与独立 source-recomputing validation

目标：实现并用 fixtures 测试把 execution root 转换为 sealed evidence root 的完整性边界；本任务不对正式 root 调用 finalize，D9 是唯一正式 seal owner。

必须完成：

- finalize 前拒绝 active claims、未结束 shard、额外文件、symlink 和不安全路径；
- 对 500 registry rows、1,000 unit keys、attempt/status/manifest/artifact/provenance 做 exact left join；
- 验证每图 NNS/matched 共享相同 clean input、模型、target、replacement baseline、fraction grid、seed derivation 和算子版本；不同 ranking 产生的 perturbed tensors 通常不同，要求各自可由冻结规则重算并正确绑定，不要求 tensor hash 相同；
- 验证 endpoint/token parity、IG conservation、finite values、direction validity、audit repeatability 和资源记录；
- 生成 final evidence report、failure/NA reconciliation、payload manifest 和 whole-root ordered inventory；
- 独立 validator 从 source 重新计算关键绑定，不信任自报 summary；
- 区分 `package_complete` 与 `scientific_validity`：完整记录的科学 failure 可封存并交 D10 输出 `NO_GO_VALIDITY`，tamper/missing/unauthorized/incomplete 则 `BLOCKED`。
- 为独立 CLI 增加 `finalize` 和 `validate`；D5 只在 fixtures 上调用，正式调用权属于 D9。

验收：synthetic valid/tampered/missing/failed/active-claim fixtures 全覆盖；finalize/validate 为 CPU-only，不构造模型、不读 JPEG、不做效果聚合。

论文贡献：Supplementary QC、repeatability、failure/NA 和实际成本字段的验证能力；无效果图。

完成后：追加 D5 completion record 和 D6 完整提示词。

### D6：预先实现 aggregation、statistics、decision 与 paper export

目标：在看到 Stage D 结果前冻结分析代码、输出 schema 和图表 recipe。

必须完成：

- CPU-only；科学数值只来自 validated L0，曲线展示可读取 validated L1；完整行覆盖、failure、NA 和 terminal state 必须来自 validator 认证的 sealed reconciliation/status/manifest。不得读取 JPEG、模型、display PNG，L2 不进入任何数值统计或 decision；
- 每图严格配对 NNS/matched insertion/deletion AUC 并导出一个 `delta_fid`；
- 生成完整 500-coordinate table；invalidating evidence 触发 `NO_GO_VALIDITY`，不做 complete-case；
- 实现 D2 冻结的 500-image paired median bootstrap 和 truth table；
- `holm_status=NOT_APPLICABLE_NO_SECONDARY_INFERENCE`，不输出 secondary p-values；
- 冻结 `paper_data` schema、主表列和主图 A/B/C recipe；D10 不生成定性 overlay；
- 图表排序、颜色、轴、缺失显示、cohort separation 和 caption fields 预先固定；
- 主图 C 固定为描述性曲线：每图每方法的 5 个 tie repeats 先取均值；跨图每个 fraction 报告 median，pointwise 95% paired image-bootstrap band 使用与 primary 分离的 Stage-D figure seed namespace、10,000 draws、PCG64 和 linear percentiles；不把 band 当作 simultaneous confidence band，也不参与 primary decision；
- 用 synthetic golden fixtures 覆盖 399/400、median 0.02、CI lower 0、CI upper 0.02、invalidity、misclassification、deterministic bytes 和图表输入一致性。
- 为独立 CLI 增加 `aggregate`/`build-paper-package` 和 `validate-paper-package`；D6 只在 fixtures 上调用，正式调用权属于 D10。

验收：所有输出可由 synthetic sealed source 确定性重建；没有 Stage D image/result access；C10 `n=15` 与 Stage D `n=500` 永不池化。

论文贡献：确定论文表图的格式和生成代码；仍不产生正式论文数值或图片。

完成后：追加 D6 completion record 和 D7 完整提示词。

### D7：已知样本 smoke、预算冻结、producer 封存与执行授权

目标：在不触碰 reserve outcome 的情况下证明完整 Stage D 链可运行，并签发唯一正式执行授权。

D7 内部按四个顺序子门执行：`smoke → budget → freeze producer/ref → authorize`。任一子门失败立即停止，D7 标 `BLOCKED`，不得跳到后续子门或 D8。

必须完成：

- 使用独立 `formal=false` smoke run spec/root，在固定 1 个 Stage C 已访问 audit image 或 synthetic data 上运行两种方法、fine-only 的最小 CPU/GPU integration smoke；不得使用 reserve 图，不得放宽正式 500-row registry validator，smoke 产物不得进入 D8–D10 evidence。若含 2 units、2 clean maps、2×2×5×21 curve points，则为 422 logical work items；是否加 repeat-clean 必须在 smoke spec 中显式说明并由 typed plan 重算；
- 测试同图两 unit、fine-only curves、cache release、artifact、finalize、validate、aggregate 和 paper export 全链；
- 按 D2 typed plan 和 smoke/历史保守实测重新计算 1,000-unit 时间、存储、调用数和显存预算；旧 8,000-unit `PASS_COST` 只作旁证；
- 新预算必须从 typed plan 的 method/work-item profile 求和，禁止调用旧 `STAGE_D_EXECUTION_UNITS=8000` 作为新 preflight；分别报告总 GPU-hours、每 image-group GPU-hours、最大单 unit reserved memory、总 artifact bytes、audit L2 增量以及冻结 safety/overhead factor；
- 运行 Stage D focused tests、受影响 Stage A–C tests、C0 clean/tracked validator 和一次全量 non-GPU suite；
- 冻结一个 clean execution/analysis producer commit，并建立只读 archive ref（建议 `refs/archive/stage-d-fidelity-v1-producer`）；
- 从该 commit 创建 authorization，绑定 C10、D0 scope、D1 registry/ledger、D2 protocol/config/schema/run spec、plan、budget、依赖、模型和代码 hashes；
- authorization 只能输出 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`，并声明它不改变 C10 `downstream_authorized=false`；
- authorization 生成前后均确认 pre-authorization access attestation 无 forbidden access。
- 为独立 CLI 增加 `smoke`、`budget` 和 `authorize-execution`；D8–D10 只能调用 D7 producer 中冻结的命令，不得再改生产或分析代码。

验收：全部 gate 通过、工作树 clean、archive ref 稳定、正式 output root 尚不存在、没有 reserve JPEG decode/model/result access。若任一项失败，D7 `BLOCKED`，不得进入 D8。

论文贡献：Supplementary reproducibility/budget 表的诊断依据；不进入 primary 结果。

完成后：producer commit 与只追加 completion record 的 commit 分开记录；追加 D7 completion record 和 D8.0 初始化提示词。

### D8：正式 500-image execution campaign

目标：严格按 D7 authorization 生产全部 1,000 units 的原始 evidence，不做科学聚合。

#### D8.0：初始化正式 root

- 从 D7 archive ref 的 clean detached worktree 执行；
- 先只用 metadata 重新验证 authorization、plan、registry、access ledger、Git/model/dependency hashes 和 fresh output path；
- 先创建最小 immutable controls、pending/shard table、独立 append-only execution access journal 和 durable `ACCESS_STARTED` lifecycle record，再首次按协议逐图读取、hash、decode 并验证 live reserve JPEG 与官方 label；每次访问进度或最终 live-assets digest 必须持久记录，中断时保留已访问范围；不得修改 D1 的 pre-authorization attestation；
- 全部 live assets 验证通过后才转为 runnable；任何失败保留 blocked root，不得假装尚未访问；
- 不运行 unit，不输出 AUC/delta/summary；
- 完成后追加 `D8.0 completion record` 和第一个 `D8-S00` 提示词。

#### D8-Sxx：逐 shard 执行

D7 必须冻结 `shard_count` 和严格递增的 shard 顺序。为避免多个对话同时修改交接文档，Stage D 采用单 coordinator、串行 shard 交接：一个对话只运行一个 shard，不允许并行 wave。一个 shard 内可以按 D7 冻结的 GPU/process 映射并行处理其 image groups，但下一 shard 必须等当前 shard receipt 已验证、completion record 已提交后才能开始。

每个 shard 必须：

- 只处理 frozen `selection_order % shard_count == shard_index` 的 image groups；
- 同图 NNS/matched 两个 units 不跨 shard；
- 一块 GPU 一个进程，`CUDA_VISIBLE_DEVICES=<physical>` 后统一使用 `cuda:0`；
- 每图完成后释放 backend cache；
- 保留全部 attempt、失败和资源记录；无效果 early stop；
- 只报告 expected/claimed/terminal/succeeded/failed 及资源，不读取或显示汇总 fidelity；
- 不运行 spatial、sanity、patch-scale、其他层头、其他方法或 display；
- 不 finalize、不 aggregate、不生成论文图。

正式计算与交接文档使用两个明确分离的工作树：

- execution worktree：D7 producer archive ref 的 clean detached worktree，只运行正式命令，永不追加或提交交接文档；
- coordination worktree：`main`，只验证 formal-root control/receipt、追加本文记录并产生 completion-only commit，不执行正式 unit。

每个 `D8-Sxx` 必须在 formal root 生成 immutable、shard-scoped receipt，绑定 shard index、authorization、producer、image-group/unit key digests、attempts、terminal counts 和 artifact/status digests。早期 shard 不生成 whole-root digest，因为 root 仍会增长；whole-root inventory 只能由 D9 在所有 worker 退出后生成。Coordinator 验证 receipt 后，在 main 追加 shard completion record。若仍有未运行 shard，next task 必须由权威 shard ledger 确定为精确编号的下一 shard；全部 shard terminal 后，coordinator 追加独立的 `D8 campaign completion record`，把父任务 D8 改为 `PASS` 并生成 D9 提示词。

权威 shard ledger 必须由 D8.0 创建并以 canonical digest 绑定，状态只允许 `PENDING`、`CLAIMED`、`TERMINAL`、`RETRY_AUTHORIZED`、`BLOCKED`。每次转换原子写入事件与新 digest；自由文本 `remaining_shards` 只作摘要，不能决定调度。正常下一 shard 只有在前一 shard 为 `TERMINAL` 且 receipt 校验通过时才获授权。

重试规则由 D2/D7 冻结，最低要求如下：scientific invalidity 是 terminal evidence，不重试；允许白名单内、尚未形成科学结果的基础设施中断时，由 coordinator 校验后签发同一 shard 的 `RETRY_AUTHORIZED`，使用新 attempt ID 并保留旧 attempt；tamper、authorization drift、unsafe root 或无法唯一判定 attempt 时，campaign `BLOCKED`，不得在同 root 继续。若只能新建 formal root，必须经显式诊断/amendment 后全量重跑冻结计划，不能拼接旧 root 或只补有利样本。

验收：全部 500 image groups 和 1,000 unit coordinates 有显式 terminal record；正常、完整记录的 scientific failure 不阻止 D8 获得 execution-completeness `PASS`，但这个 `PASS` 不表示 scientific validity。Tamper、authorization drift、unsafe state 或非终态坐标必须停止并标 `BLOCKED`。

论文贡献：**生成论文主实验的原始 L0/L1 数值证据和预注册 audit L2；不生成图表或科学结论。**

### D9：formal root 封存与独立验证

目标：在所有 GPU worker 退出后，将 D8 root 封存为 D10 唯一可读的 evidence source。

必须完成：

- 使用 D7 producer 的 clean detached worktree，CPU-only finalize；
- 执行 500/1,000 exact reconciliation、pair binding、artifact/provenance/hash、validity、audit repeatability 和 actual resource audit；
- 生成 final evidence report、manifest/status 和 whole-root inventory digest；
- 再从独立 clean worktree 执行 source-recomputing validation；
- 不读取 JPEG、不构造模型、不运行 GPU、不计算 `delta_fid` 汇总或 decision；
- 若 package complete 但存在完整记录的 scientific invalidity，可授权 D10 输出 `NO_GO_VALIDITY`；若 tampered/missing/unauthorized/incomplete，则 D9 `BLOCKED`，不得统计。

验收：formal root closed、immutable、可独立复验；所有 count/hash/identity 与 D7 plan 一致；明确 `analysis_authorized=true/false`。

论文贡献：**生成 Supplementary QC、failure/NA、repeatability 和实际成本数据；不生成主效果结论。**

完成后：追加 D9 completion record；只有 `analysis_authorized=true` 才写 D10 提示词，否则写诊断提示词。

### D10：唯一正式统计、决策与论文 artifact package

目标：CPU-only 消费 D9 sealed root，生成 Stage D 唯一正式 fidelity-scope decision 和可直接进入论文的数据/图片。

必须完成：

- 启动时重跑 D9 validator；科学数值仅读取 sealed L0，曲线可读取 validated L1，完整行覆盖读取 sealed control/reconciliation；不读取 JPEG、模型、L2 或 display PNG；另只读 D9 report 和 sealed C10 reports；
- 使用此前不存在的独立 postprocessing output root；D9 sealed evidence root 始终只读，统计、paper data 和图不得写回 D9 root；
- 生成 per-sample table/schema/manifest、aggregated curves、summary、bootstrap metadata/distribution digests、decision、acceptance、provenance 和完整 inventory；
- validity 失败时输出 `NO_GO_VALIDITY` 且不做伪 effect inference，只生成完整 QC/failure 表；主效果表和主图 A/B/C 必须以结构化 `NOT_GENERATED_DUE_TO_VALIDITY` 占位，不得对有效子集作 complete-case 图；validity 通过后严格执行 D2 truth table；
- C10 `n=15` 与 Stage D `n=500` 分 cohort 统计和作图，不池化；
- C10 始终显示 `NO_GO_NOVELTY / FAIL_SPATIAL / BLOCKED`；Stage D 只输出新的 fidelity-scope decision；
- 在 validity 允许的条件下生成第 4 节列出的 paper data、数值主图和补充表；所有图只从 machine-readable paper data 构建；
- 结果无论正、负或 inconclusive 都生成完整报告，不换 head、阈值、样本或 baseline。

验收：输出可从 D9 source 确定性重建；文件 hash、canonical digest 和 inventory 完整；论文 allowlist/denylist 自动检查；Stage D 未运行任何 spatial endpoint。

论文贡献：**在 validity 允许时生成正式主结果表、主图 A/B/C、全部 supplementary 数据和唯一 Stage D decision；定性 overlay 留给 Stage F 的独立 display-only recipe。**

完成后按结果生成下一提示词：

- `GO_FIDELITY_SCOPE`：先建立并冻结独立的 Stage E descriptive protocol，或直接进入 Stage F 写作；
- `NO_GO_FIDELITY` / `INCONCLUSIVE_FIDELITY`：进入负向结果写作/诊断，不换配置重跑；
- `NO_GO_VALIDITY`：进入 validity 诊断，不能把这 500 张重新声明为未见确认集；
- workflow `BLOCKED`：只生成修复/审计提示词，不推进论文结论。

D10 应先原子生成并完整验证 machine-readable decision package，再从已验证 package 生成 paper data 和数值图。绘图失败不得回写或改变科学 decision；应保留 decision package，将 paper-export 状态标为 `BLOCKED` 并生成只修复导出的提示词。无论哪个结果分支，D10 都必须追加完整 completion record 和对应的可复制下一提示词。

## 7. 跨对话执行与交接协议

### 7.1 每个新对话开始时必须做

1. 完整阅读本文、上一任务 completion record、其内的完整提示词，以及提示词列出的正式 protocol/acceptance/config/schema。
2. 检查 Git root、`HEAD`、branch、worktree/index clean 状态和上一记录的 commit。
3. 复核上一任务列出的关键 file SHA-256、canonical digest、archive ref 和 artifact status。
4. 只执行一个指定 `D_i`；D8 只执行一个明确的 `D8.0` 或 `D8-Sxx`。不得顺手启动下一任务。
5. 若有 dirty 文件，先判断归属；不得覆盖、stash、reset 或删除用户改动。不能建立可信基线时标记 `BLOCKED`。
6. 将第 5 节对应状态改为 `IN_PROGRESS`。D8 shard 另在 execution log 中维护 shard ledger。

### 7.2 每个任务结束时必须做

1. 完成该任务的实现、测试、文档和产物验证，不用后续任务补齐当前验收。
2. 将第 5 节状态改为 `PASS` 或 `BLOCKED`；D8.0 和单个 D8 shard 使用第 8.1 节的子任务状态，父任务 D8 保持 `IN_PROGRESS`，只有 campaign completion record 确认所有 frozen shards terminal 后才能整体 `PASS`。
3. 在第 9 节追加 completion record。失败、偏差、中断、retry 和废弃 root 也必须保留。D8 的上一条权威交接记录是最新 coordinator completion commit 中的 D8.0/shard/campaign record，不是 execution worktree HEAD。
4. completion record 后紧跟下一任务的完整可复制提示词，不能只写“继续 D_i+1”。
5. 最终回复重复：状态、关键文件、实际测试、commit/ref/hash、产物路径、数据访问事实、遗留问题和下一提示词。
6. 原则上一任务一提交并最终 clean。D0–D7/D9/D10 的代码或文档任务遵守普通提交规则；D8 正式 runtime evidence 必须绑定 D7 producer ref，不能归因给 main 上只追加 completion record 的 commit。
7. 只在上一任务 `PASS` 且 `next_task_authorized=true` 时进入正常下一任务；否则 next task 必须是诊断或修复。只有 D7 可以写 `authorization_issued_this_task=true`；D8–D10 只能在重新验证 D7 authorization digest 后写 `execution_authorization_observed_valid=true`。D0 的 `APPROVED_TO_PREPARE_STAGE_D` 只允许 D1–D7 准备工作。

### 7.3 回归测试复用规则

- 若当前 HEAD 精确等于上一 completion commit、工作树 clean、冻结 inputs 未漂移、上一任务已在该 commit 完成全量 non-GPU suite，可复用为“上一任务已验证基线”；不得写成本次重新执行。
- 本任务必须运行自己的 tests 和所有被实际修改的共享模块 tests。修改 Stage C primitives 时必须补跑受影响的 Stage C focused tests；影响面不清时运行全量 non-GPU suite。
- D7 在 producer commit 上必须运行 Stage D focused、受影响 Stage A–C、C0 clean/tracked validator 和全量 non-GPU suite。
- D8 不因文档 completion commit 重跑全部测试，但每个 shard 启动都必须重验 producer ref、authorization 和 immutable controls。

### 7.4 Git、producer 与正式 root

- D0–D6 可以各自一任务一提交。
- D7 必须区分 baseline commit、execution/analysis producer commit、archive ref 和 completion-only commit。
- D8 正式计算、D9 正式封存/验证和 D10 正式分析必须从 archive ref 的 clean detached execution worktree 执行；对应 completion record 由 main coordination worktree 追加。两条 commit 身份分别记录，禁止把 main completion HEAD 写成 producer。
- 正式 root 必须是此前不存在的新绝对路径；禁止覆盖、拼接不同 code commit 的 units 或删除失败记录。
- 每个正式 root 至少有单文件 SHA-256、canonical semantic digest 和 whole-root ordered inventory digest 三层绑定。

## 8. Completion record 模板

除 D8.0/D8-Sxx 子任务外，每个顶层任务都使用以下字段。无内容时写 `none`，不得删除字段：

````markdown
### YYYY-MM-DD HH:MM +0800 — D_i completion record

- status: PASS | BLOCKED
- objective_completed:
- authority_and_scope:
- git_sha_before:
- git_sha_producer: none | <sha>
- git_sha_postprocessor: none | <sha>
- git_sha_completion: SELF
- tree_clean_at_boundaries:
- upstream_commits_and_refs:
- files_added_or_changed:
- frozen_inputs_and_sha256:
- config_plan_and_canonical_digests:
- data_access_statement:
- artifacts_and_absolute_paths:
- status_manifest_inventory_hashes:
- expected_vs_observed_counts:
- tests_reused_from_prior_commit:
- tests_executed_this_task:
- acceptance_results:
- package_complete: true | false | not_applicable
- scientific_validity: PASS | FAIL | not_evaluated
- analysis_authorized: true | false
- budget_results:
- paper_data_or_figures_generated:
- deviations_or_protocol_amendments:
- failures_and_retries_retained:
- known_limitations_and_claim_boundaries:
- next_task_authorized: true | false
- authorization_issued_this_task: true | false
- execution_authorization_observed_valid: true | false
- next_task: D_i+1 | DIAGNOSE_... | STOP

#### 下一对话完整提示词

```text
（写入完整、可直接复制的提示词：唯一任务与禁止越界、工作目录、
前置 commit/ref/hash、必读文件、允许的数据访问、精确 workload、
输出合同、failure/retry 规则、测试、验收、Git 和本文更新要求。）
```
````

### 8.1 D8 shard completion record 补充字段

`D8.0` 和 `D8-Sxx` 是 campaign 子任务，不使用上面的顶层 `status` 字段。它们改写为 `parent_status: IN_PROGRESS` 和 `subtask_status: SUBTASK_PASS | RETRY_PENDING | SUBTASK_BLOCKED`；成功子任务可以用 `shard_continuation_authorized=true` 授权下一 shard，同时父任务 D8 保持 `IN_PROGRESS`。只有最后的 `D8 campaign completion record` 使用第 8 节顶层 `PASS/BLOCKED` 模板。其余通用字段仍须保留。

每个 `D8-Sxx` 除通用字段外，还必须记录：

- `formal_root`；
- `shard_index/shard_count`；
- `expected_image_groups/units/work_items`；
- `claimed/terminal/succeeded/failed`；
- `attempt_ids`；
- `resource_summary_without_effect_metrics`；
- `shard_receipt_sha256/canonical_digest`；
- `shard_ledger_before/after_canonical_digest`；
- `subtask_status: SUBTASK_PASS | RETRY_PENDING | SUBTASK_BLOCKED`；
- `shard_continuation_authorized: true | false`；
- `retry_authorization_digest: none | <digest>`；
- `remaining_shards`（仅摘要，权威调度来自 shard ledger）；
- `effect_summary_accessed=false`。

`D8 campaign completion record` 必须另行记录 `all_shards_terminal`、`all_receipts_verified`、`expected_vs_terminal_image_groups/units` 和 `scientific_failures_present`。D8 `PASS` 只代表计划完整执行和证据可交 D9，不代表 `scientific_validity=PASS`。

## 9. 执行日志

已有记录只允许追加勘误，不允许删除或改写失败史。每个 `D_i` 末尾的“完成后”条款均要求使用第 8 节全部字段，并生成任务特定的完整下一提示词；动态 commit、hash、root 和测试结果必须填实，不能保留占位符。D10 也必须追加 completion record；其下一提示词须根据第 6 节 D10 的四类结果分支完整写出，而不是只写任务名。

### 2026-08-25 10:18 +0800 — D0 初始 completion 草案（已勘误为 BLOCKED）

> 勘误：本段是首次尝试留下的、尚未提交的 PASS 草案，不是有效 completion record。该尝试随后被确认运行了 CPU 模型/推理并解码非-reserve JPEG，违反 D0 操作门，因此其权威状态为 `BLOCKED`。以下原文保留用于审计，不能作为 D1 trust anchor；其 D1 提示词和 `8fff...` digest 均已失效。

- status: PASS
- objective_completed: 冻结 post-C10 fidelity-only 研究范围、C10 历史命名空间、Stage D claim allowlist/denylist、mandatory disclosures 和正交授权状态；新增人类可读决议、单一权威机器记录、closed Draft 2020-12 schema、fail-closed loader、sealed C10 hash-chain verifier 及静态/schema/hash/tamper tests。D0 只签发 `APPROVED_TO_PREPARE_STAGE_D`，未签发 execution/analysis authorization。
- authority_and_scope: `preparation_authorization=APPROVED_TO_PREPARE_STAGE_D`；只授权 D1–D7 准备链，下一步仅为 D1 metadata-only registry formalization；`execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`，不覆盖 C10。
- git_sha_before: `69e1fa887862e00da4441842c6de149fe2424225`
- git_sha_producer: none（D0 不生产正式实验 evidence；scope 合同由本 completion commit 跟踪）
- git_sha_postprocessor: none
- git_sha_completion: SELF（即包含本 record 的单一 D0 commit；精确 SHA 由提交后 `git rev-parse HEAD` 产生并在最终回复报告）
- tree_clean_at_boundaries: 启动时工作树含 4 份用户已授权的 Stage D 路线规划改动：`../README.md`、`../论文1实现审计与后续计划.md`、`README.md`、`docs/阶段D-step-by-step.md`；逐份审阅确认均属于本路线并纳入 D0。未覆盖、stash、reset 或删除其他用户改动；D0 completion commit 后 clean 状态须在最终回复据实报告。
- upstream_commits_and_refs: C10 completion `69e1fa887862e00da4441842c6de149fe2424225`；aggregation producer `92ad81342ff22e2710df495a9961a3ab8c015a3a`；`refs/archive/c10-aggregation-producer` 复算为同一 commit，tree `3551096e358391fe04952e8ced19ea60884fb32f`，`src/vitnns` tree `6874e71aec0e592e110a540a1638ea2cfc697c8d`。
- files_added_or_changed: `../README.md`；`../论文1实现审计与后续计划.md`；`README.md`；`docs/阶段D-step-by-step.md`；`docs/stage_d_scope_decision.md`；`config/stage_d_scope_decision.json`；`config/stage_d_scope_decision.schema.json`；`src/vitnns/stage_d_scope.py`；`tests/test_stage_d_scope_contract.py`。
- frozen_inputs_and_sha256: C10 `stage_c_report.json=9a16034d29bc87eaba852aa0e618928ac0e4ecee9bf745d765f42311dd4563b4`；`decision.json=07fc5875e46018adf5dc0cd09967e273df3b443c1deeb1653d672d05103e513e`；`holm.json=be37ffbfd2ba60b5549ba791076383d9ba53a3b437178273a608fbf527b91781`；`provenance.json=4dcca54fdc83ba7401070ef843636b5e67b68a8ce720fa70f729e407fc5717f0`；`manifest.json=507ca3c4ac210882a5918cc602cf7a4656aa398af76ff9f3638a77f3af42a0d8`；`status.json=192d383d5c4788f612018532074c06d48511513edd7565dbb99934706cd96db2`；model config `cc8d386736b3f0ae299e75184e86772b21b8139cab0a7c0eab17a961d1ad55a3`；state dict binding `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`。
- config_plan_and_canonical_digests: `config/stage_d_scope_decision.json` canonical digest `8fff3955c488a344262e5e0fde5dfc1e9178de6f25dda5b866010c8c0861c3ce`；scope/schema/document/source/test 的最终 file SHA-256 在提交前复算并在最终回复报告。D1 registry、D2 protocol/config/statistics、typed plan 与 budget 尚未冻结。
- data_access_statement: 以下是 task owner 对本次 D0 行为的审计声明；schema/digest 固定声明内容但不独立证明历史进程行为，reserve 的历史边界仍须由 D1 ledger 支持。只读取 sealed C10 顶层 package 文件做完整性验证；没有从 C9 source 重建 C10 科学统计。对 Stage D reserve 只列目录确认候选 metadata 文件存在，未打开 reserve/access-log 内容，未 stat/open/hash/decode 任一 reserve JPEG，未访问 Stage D outcome。`reserve_access_state=METADATA_EXISTENCE_ONLY`、`reserve_image_bytes_accessed=false`、`reserve_result_accessed=false`、`model_or_gpu_used=false`、`c9_evidence_read=false`、`c9_evidence_modified=false`、`c10_evidence_modified=false`。
- artifacts_and_absolute_paths: `/data00/home/yeziqi.831/testcode/vitnns_rebuild/docs/stage_d_scope_decision.md`；`/data00/home/yeziqi.831/testcode/vitnns_rebuild/config/stage_d_scope_decision.json`；`/data00/home/yeziqi.831/testcode/vitnns_rebuild/config/stage_d_scope_decision.schema.json`。无 runtime/formal/paper artifact。
- status_manifest_inventory_hashes: sealed C10 root 精确 12 个普通 single-link files、`524796892` bytes；whole-root ordered inventory digest `382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`；10-payload manifest 和 status→manifest 绑定通过；全部 10 个带自摘要的 sealed JSON canonical digest 复算匹配。
- expected_vs_observed_counts: C10 sealed files `12/12`、manifest payloads `10/10`、关键 file hashes `6/6`、带自摘要 JSON `10/10`、archive commit/tree/source-tree bindings `3/3`；D0 Stage D execution units `0`、reserve image bytes `0`、Stage D outcomes `0`、paper figures `0`。
- tests_reused_from_prior_commit: C10 completion record 在 commit `69e1fa887862e00da4441842c6de149fe2424225` 记录的 `1034 passed, 1 skipped, 1 deselected` 全量 non-GPU suite、producer/ completion validators 和 source-recomputing C10 validation仅作为既有基线证据引用，本任务没有伪称重新执行。
- tests_executed_this_task: D0 focused 与 Stage C static contract 的最终命令/计数、`py_compile`、`git diff --check`、JSON/schema/Markdown 检查及 post-commit复验在最终回复据实报告。开发期 focused 首轮 `17 passed, 2 failed`（路线状态尚未推进、unsigned-tamper 校验次序）；第二轮 `13 passed, 6 failed`（测试预期仍要求 schema-first 错误、路线状态尚未推进）；均为提交前测试收敛，不涉及科学数据，失败历史保留。
- acceptance_results: scope 文档与机器记录由 claim/disclosure IDs、授权枚举和关键事实双向静态核对；C10 sealed evidence hash 链有效且未写入；C10 的 `NO_GO_NOVELTY`、`FAIL_SPATIAL`、`BLOCKED`、`HOLM_BLOCKED_UNDERSPECIFIED`、`downstream_authorized=false`、`stage_d_started=false` 原样保留；C10 15 fixed holdout 的 `(0,9)` `PASS_VALIDITY/PASS_EFFECT` 被限定为 prospective fidelity-only 研究依据；D0 唯一结果为 `APPROVED_TO_PREPARE_STAGE_D`。
- package_complete: true
- scientific_validity: not_evaluated（Stage D 尚无运行结果；此字段不否定已封存的 C10 `PASS_VALIDITY`）
- analysis_authorized: false
- budget_results: not_evaluated；C10 `PASS_COST` 只作旧 8,000-unit workload 的历史可行性证据，D7 必须按新 1,000-unit typed plan 独立验收。
- paper_data_or_figures_generated: none
- deviations_or_protocol_amendments: none；这是显式、post-C10、前瞻性的 scope decision，不是对 C10 v1 的 amendment 或重判。为遵守“不修改或重算 C9/C10 evidence”，D0 的 verifier 只复算 sealed C10 package 的 hash/canonical/manifest/status/inventory/Git ref，不从 C9 重建数值。
- failures_and_retries_retained: 上述两轮开发期测试失败保留在本 record；无模型/GPU/OOM、reserve access、正式 root、科学 failure 或 retry。
- known_limitations_and_claim_boundaries: C10 effect 只覆盖 15 张固定、历史文件名等间隔且非类别均衡 holdout、固定 ViT-B/16、Value-path target、`(0,9)`、matched baseline 和 fine-4×4-cell endpoint；4×4-cell 不是原生单像素；Value-path 不是完整 Q/K/V、attention 或 class-logit attribution；translation/crop failure 必须在设计、结果和 limitations 披露；不主张一般 spatial robustness、等价或普遍 ImageNet 优越；C10/Stage D cohort 不池化。
- next_task_authorized: true（只授权 D1 metadata-only 准备）
- authorization_issued_this_task: false
- execution_authorization_observed_valid: false
- next_task: D1

#### 下一对话完整提示词

```text
你现在只执行 ViT-NNS fidelity-only Stage D 路线的 D1：在严格 metadata-only 边界内，把现有 500-row candidate reserve 正式化为 tracked、内容寻址、可验证的 Stage D registry，并预注册 5 个 audit overlay IDs。不要开始 D2，不要冻结 Stage D 统计/运行合同，不要读取、stat/hash 或 decode 任一 reserve JPEG bytes，不要访问任何 Stage D outcome，不要运行模型/GPU，不要实现 runner，也不要签发 execution authorization。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

前置 D0 边界：
- D0 completion anchor 是 `refs/archive/stage-d-d0-completion`；启动时必须用 `git rev-parse 'refs/archive/stage-d-d0-completion^{commit}'` 得到唯一 D0 completion commit，要求当前 HEAD 与该 commit 完全相等且 index/worktree clean，否则 D1 `BLOCKED`。ref 缺失、类型错误或无法解析也必须 fail closed。
- D0 scope machine record：`config/stage_d_scope_decision.json`，canonical digest `8fff3955c488a344262e5e0fde5dfc1e9178de6f25dda5b866010c8c0861c3ce`。
- D0 schema：`config/stage_d_scope_decision.schema.json`；人类可读决议：`docs/stage_d_scope_decision.md`。启动时从上述 completion ref 对应 tree 读取三个 blob，并现场复算 file SHA-256、机器记录 canonical digest和结构化 Markdown projection；不得依赖聊天或会话外回复作为 trust anchor。
- D0 只签发 `APPROVED_TO_PREPARE_STAGE_D`；`execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`。
- C10 completion commit：`69e1fa887862e00da4441842c6de149fe2424225`。
- C10 producer/ref：`92ad81342ff22e2710df495a9961a3ab8c015a3a` / `refs/archive/c10-aggregation-producer`。
- C10 report SHA-256/canonical：`9a16034d29bc87eaba852aa0e618928ac0e4ecee9bf745d765f42311dd4563b4` / `90232df2010f6a8fc771baeac3aaac44fee93d436597cf0cbdebe3cf8d2142e2`。
- C10 decision SHA-256/canonical：`07fc5875e46018adf5dc0cd09967e273df3b443c1deeb1653d672d05103e513e` / `60e112d50f22229474a634039ba71c7eed1adba081d8e1097f850a8a4b4d8dc0`。
- C10 provenance SHA-256/canonical：`4dcca54fdc83ba7401070ef843636b5e67b68a8ce720fa70f729e407fc5717f0` / `4b26452041685cbb6c12d93139736e65fc4330fd44a64bf854848e7cff66c742`。
- C10 manifest SHA-256/canonical：`507ca3c4ac210882a5918cc602cf7a4656aa398af76ff9f3638a77f3af42a0d8` / `c71ae7c19941eab556a0ed9fdde553cb1a3fae914e149652df9b31cf3538720a`。
- C10 whole-root inventory digest：`382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`。

候选 metadata 输入（均不是正式 Stage D registry；D1 必须先以只读字节流复算 metadata file hash，再解析同一已绑定 bytes）：
- `manifests/stage_c_plus_v2/stage_d_v2_reserve.json`：预期 500 rows；当前已知 file SHA-256 `1432183a0ce9c701e60cc28acc3596fe16bb253dfc013126bf3168b5af05099a`、canonical digest `2e66fa2426fb7c42caefe7907a6ef1ec9a42adb1788528d59af9b5354eb0a99f`、ordered sample IDs digest `b187ff5cbacc1b595f06d12e309602b09965e01e7ee1461fd8a694354d4b8f50`。
- `manifests/stage_c_plus_v2/stage_d_v2_reserve_access_log.json`：当前已知 file SHA-256 `75a319820db12102dde22308ab2e12196e0e1ca6fcdd870e2943d22a0643a621`、canonical digest `41478eee77f708a1158142d700fdfd65503b120aa672400af01fe14f9eb317fa`、预期 `forbidden_access_detected=false`。
- 同目录的 `cplus_registry.json`、`eligible_manifest.json`、`prior_access_ledger.json`、`sample_selection.json`、`source_archive_identity.json`、`source_inventory.json`、`source_spec.json` 只能作为 metadata source；逐一记录 file/canonical hashes 和证明范围。

第一步必须完整阅读：
1. `docs/阶段D-step-by-step.md`，尤其 D0 completion record、D1、§7–§9；
2. `docs/stage_d_scope_decision.md`；
3. `config/stage_d_scope_decision.json` 与 schema；
4. `src/vitnns/stage_d_scope.py` 和 `tests/test_stage_d_scope_contract.py`；
5. `../论文1实现审计与后续计划.md` 的 post-C10、Stage D/E/F；
6. `docs/阶段C_protocol.md` §2、§9–§10，`docs/stage_c_acceptance.md` C0/C8/C9/C10；
7. `manifests/stage_c_sample_registry.csv` 及 metadata，供 20-ID disjointness 检查；
8. 上述 `manifests/stage_c_plus_v2/` metadata 文件。

开始前：
1. 检查 Git root、HEAD、branch、index/worktree clean 和 `refs/archive/stage-d-d0-completion`；要求该 ref 是 commit、HEAD 精确等于它且 index/worktree clean。若有 dirty 文件先判断归属，不得覆盖、stash、reset、删除或擅自提交用户改动；无法建立可信 D0 基线则 `BLOCKED`。
2. 运行 D0 loader/tests，并只读复核 sealed C10 hash/canonical/inventory/ref 链；不得从 C9 重建 C10 科学数值。
3. 结构化解析任一 candidate metadata 前，先以只读字节流计算该 metadata 文件的 size、mtime、SHA-256，再从同一已绑定 bytes 严格解析；保持 reserve JPEG 与 outcome 禁止访问。

D1 必须完成：
1. 只读、严格解析 candidate reserve、access log、prior-access ledger、source inventory/spec/archive identity 和 selection metadata；拒绝 duplicate keys、NaN/Infinity、unknown/missing fields、路径逃逸、symlink 或 digest drift。不得打开 metadata 中引用的 live JPEG。
2. 新建唯一 tracked registry package，建议 `manifests/stage_d_fidelity_v1/`，至少包含 closed Draft 2020-12 schema、500-row registry、registry metadata/manifest、source binding 和不可变 pre-authorization access attestation。机器文件须 canonical、自摘要或由外层 manifest 绑定；避免循环 hash。
3. 精确 500 rows；`selection_order` 连续 `0..499` 且唯一；`sample_id`、relative path、candidate-recorded image SHA-256 唯一；官方 WNID/index/label-source 逐行有效，类别标签允许重复。D1 只验证 candidate metadata 中已记录 image hash 的格式、内部 binding 和来源证明，不对 live JPEG 复算。
4. 与 `manifests/stage_c_sample_registry.csv` 的全部 20 IDs/paths/image hashes 明确互斥；不得重抽、补样、替换或依据类别、模型预测、图像内容、结果或人工偏好筛选。任何冲突或不一致使 D1 `BLOCKED`。
5. 从冻结 registry 顺序使用 Stage D 专属、写入合同的 hash namespace 确定性预注册恰好 5 个 audit overlay IDs；audit 是同一 500 cohort 的子集，不增加统计 n，不读取图像或结果。
6. 精确记录 prior-access ledger 可证明的范围。只允许表述“未出现在已声明且封闭的 prior-access ledger 中”，禁止 `RESERVE_ABSOLUTELY_NEVER_VIEWED`。生成新的 immutable pre-authorization access attestation，保持 `reserve_image_bytes_accessed=false`、`reserve_result_accessed=false`、`model_or_gpu_used=false`；不得回写 candidate access log。
7. 如需修改 `.gitignore`，只增加新 registry package 所需的精确 allowlist；不得跟踪 JPEG、模型、runtime artifact、outcome 或原 ignored candidate 文件副本。
8. 增加 D1 static/schema/hash/tamper/rebuild/disjointness/audit-selection tests；测试只能读取 metadata。保留 D0 claim/authorization/access tests。
9. 更新 `docs/阶段D-step-by-step.md`：D1 标 `PASS` 或 `BLOCKED`，追加完整 D1 completion record；PASS 时写出只执行 D2 的完整提示词，BLOCKED 时写精确诊断提示词。
10. 原则上一任务一提交；最终工作树 clean。completion record 区分本轮实际测试与复用证据，并报告 commit、registry/schema/manifest/attestation file SHA-256 和 canonical digests。

禁止事项：
- 不读取、stat/hash、decode 或预处理任一 reserve JPEG bytes；不得通过 PIL/torchvision/dataset loader 间接访问。
- 不访问或计算 prediction、logit、map、AUC、delta_fid、summary 或任何 Stage D outcome。
- 不运行模型、CPU inference、GPU/CUDA 或 attribution。
- 不开始 D2，不冻结统计阈值/bootstrap/run spec，不实现 planner/runner/finalize/aggregate，不签发 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`。
- 不修改、重算、重命名 C9/C10 evidence；必须保留 C10 `NO_GO_NOVELTY/FAIL_SPATIAL/BLOCKED/HOLM_BLOCKED_UNDERSPECIFIED/downstream_authorized=false/stage_d_started=false`。

D1 验收：
- tracked registry package 可由 candidate metadata 确定性重建，schema closed，hash 链和 source bindings 有效；
- 500/500 rows、顺序、唯一性、官方 label metadata、与 C10 20 图互斥性全部通过；
- 5 audit IDs 由冻结 namespace 唯一导出且不增加 n；
- pre-authorization attestation 明确声明本任务 `reserve_image_bytes_accessed=false`、`reserve_result_accessed=false`、`model_or_gpu_used=false`，并准确限定 prior-access ledger 可证明的范围；schema/digest 只固定声明内容，不冒充独立行为证明；
- D0 scope 仍 valid，C10 sealed root 未写入，D1 没有 outcome 或科学结论；
- tests、文档、completion record 和 D2 提示词完整，一任务一提交，最终工作树 clean。
```

### 2026-08-25 11:57 +0800 — D0 initial attempt blocked record

- attempt_id: `D0_INITIAL_ATTEMPT_20260825`
- status: `BLOCKED`
- authoritative_for_current_authorization: false
- blocking_reasons: `D0_NO_MODEL_OR_INFERENCE_GATE_VIOLATED`；`D0_NO_JPEG_DECODE_GATE_VIOLATED`。
- observed_access: `model_loaded=true`、`model_inference_performed=true`、`model_or_gpu_used=true`、`non_reserve_jpeg_bytes_accessed=true`、`jpeg_decode_performed=true`、`gpu_used=false`。
- reserve_boundary: `reserve_metadata_contents_accessed=false`、`reserve_image_bytes_accessed=false`、`reserve_result_accessed=false`。
- disposition: 不签发 preparation/execution/analysis authorization；不建立 completion ref；该失败永久保留，不能由后续测试改写为“从未发生”。
- superseded_material: 上一节的 PASS 草案、旧 canonical digest `8fff3955c488a344262e5e0fde5dfc1e9178de6f25dda5b866010c8c0861c3ce` 和草案内 D1 提示词均不具权威性。

### 2026-08-25 11:57 +0800 — D0 model-free rerun completion record

- attempt_id: `D0_MODEL_FREE_RERUN_20260825`
- status: `PASS`
- objective_completed: 在不开始 D1 的前提下，修正 D0 产物一致性，冻结 post-C10 fidelity-only scope、claim allowlist/denylist、mandatory disclosures、attempt history 和严格测试 allowlist。
- authority_and_scope: 当前唯一结果为 `preparation_authorization=APPROVED_TO_PREPARE_STAGE_D`；下一任务仅为 D1 metadata-only registry formalization；`execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`，不覆盖 C10。
- git_sha_before: `69e1fa887862e00da4441842c6de149fe2424225`
- git_sha_producer: none（D0 不生产实验 evidence）
- git_sha_postprocessor: none
- git_sha_completion: 由 `refs/archive/stage-d-d0-completion` 唯一解析；本记录所在提交通过该 ref 锚定，避免在提交内容中写自引用 SHA。
- tree_clean_at_boundaries: 启动时有 4 个属于用户已授权 Stage D 路线的 modified 文件和 6 个 D0 untracked 文件；逐份核对后纳入本提交，没有 stash/reset/delete/覆盖无关用户改动。提交后 clean 状态在最终验收据实记录。
- upstream_commits_and_refs: C10 completion `69e1fa887862e00da4441842c6de149fe2424225`；aggregation producer `92ad81342ff22e2710df495a9961a3ab8c015a3a`；`refs/archive/c10-aggregation-producer`、tree `3551096e358391fe04952e8ced19ea60884fb32f`、`src/vitnns` tree `6874e71aec0e592e110a540a1638ea2cfc697c8d`。
- files_added_or_changed: `../README.md`；`../论文1实现审计与后续计划.md`；`README.md`；`docs/阶段D-step-by-step.md`；`docs/stage_d_scope_decision.md`；`config/stage_d_scope_decision.json`；`config/stage_d_scope_decision.schema.json`；`src/vitnns/__init__.py`；`src/vitnns/stage_d_scope.py`；`tests/test_stage_d_scope_contract.py`；`tests/d0_guard/sitecustomize.py`。
- frozen_inputs_and_sha256: C10 `stage_c_report.json=9a16034d29bc87eaba852aa0e618928ac0e4ecee9bf745d765f42311dd4563b4`；`decision.json=07fc5875e46018adf5dc0cd09967e273df3b443c1deeb1653d672d05103e513e`；`holm.json=be37ffbfd2ba60b5549ba791076383d9ba53a3b437178273a608fbf527b91781`；`provenance.json=4dcca54fdc83ba7401070ef843636b5e67b68a8ce720fa70f729e407fc5717f0`；`manifest.json=507ca3c4ac210882a5918cc602cf7a4656aa398af76ff9f3638a77f3af42a0d8`；`status.json=192d383d5c4788f612018532074c06d48511513edd7565dbb99934706cd96db2`。
- config_plan_and_canonical_digests: `config/stage_d_scope_decision.json` canonical digest `7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f`；D1 registry、D2 protocol/config/statistics、typed plan 与 budget 尚未冻结。
- data_access_statement: 仅描述本次重试。只读取 Git/源码/JSON/Markdown 和 sealed C10 package 做 byte-hash/canonical/manifest/status/inventory/ref 验证；没有从 C9 重建科学统计。没有打开候选 reserve/access-log 内容，也未 stat/open/hash/decode 任一 reserve JPEG，未访问 Stage D outcome。`reserve_access_state=METADATA_EXISTENCE_ONLY`、`reserve_metadata_contents_accessed=false`、`reserve_image_bytes_accessed=false`、`reserve_result_accessed=false`、`model_loaded=false`、`model_inference_performed=false`、`model_or_gpu_used=false`、`gpu_used=false`、`non_reserve_jpeg_bytes_accessed=false`、`jpeg_decode_performed=false`。
- verification_policy: `D0_MODEL_FREE_NO_JPEG_V1`；只运行机器记录列出的 exact pytest node IDs，`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`CUDA_VISIBLE_DEVICES=''`；collection 前由 `tests/d0_guard/sitecustomize.py` 拒绝模型/图像栈导入、JPEG/PNG/model path、reserve tree、GPU device 和非 Git 子进程；禁止全量 regression。
- artifacts_and_absolute_paths: `/data00/home/yeziqi.831/testcode/vitnns_rebuild/docs/stage_d_scope_decision.md`；`/data00/home/yeziqi.831/testcode/vitnns_rebuild/config/stage_d_scope_decision.json`；`/data00/home/yeziqi.831/testcode/vitnns_rebuild/config/stage_d_scope_decision.schema.json`。无 runtime/formal/paper artifact。
- status_manifest_inventory_hashes: sealed C10 root 预期/观测均为 12 个普通 single-link files、`524796892` bytes；whole-root digest `382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`；10-payload manifest、status→manifest、10 个自摘要 JSON 和 archive ref 链均匹配。
- expected_vs_observed_counts: C10 sealed files `12/12`、manifest payloads `10/10`、关键 file hashes `6/6`、带自摘要 JSON `10/10`、archive bindings `3/3`；D0 Stage D execution units `0`、reserve image bytes `0`、Stage D outcomes `0`、paper figures `0`。
- tests_reused_from_prior_commit: C10 completion commit 的既有 suite 结果仅作为上游历史证据；首次 D0 尝试运行的广泛回归因越过操作门而不能作为本次 PASS 验收。
- tests_executed_this_task: `D0_MODEL_FREE_NO_JPEG_V1` exact-node allowlist 第一次运行 `23 passed, 1 failed`（README 中文词序断言）；修正后为 `24 passed in 0.22s`；增加 guard 自检后一次 `23 passed, 1 failed`，发现 Python 3.10 `Path.stat()` 的缓存 accessor 可绕过 `os.stat` patch，探针是不存在的 `d0_guard_probe.JPEG`，没有读取图像；补上 `Path.open/stat` pre-collection guard 后同一 allowlist两次均为 `24 passed in 0.19s`（第二次对应最终 machine digest）。guard 下独立 JSON/schema/Markdown projection 检查通过；guard 下 sealed C10 byte-hash/canonical/manifest/status/inventory/Git-ref verifier 通过；`jq empty`、`py_compile`、`git diff --check` 通过。没有运行全量或其他 pytest。
- acceptance_results: machine/schema/Markdown/路线/README 一致；首次 BLOCKED attempt 未被改写；C10 sealed hash 链有效且未写入；C10 历史状态原样保留；当前重试符合 no-model/no-JPEG/no-reserve/no-outcome 门；唯一新授权为 `APPROVED_TO_PREPARE_STAGE_D`。
- package_complete: true
- scientific_validity: not_evaluated（Stage D 未运行）
- analysis_authorized: false
- budget_results: not_evaluated；D7 必须按新 typed plan 独立验收。
- paper_data_or_figures_generated: none
- deviations_or_protocol_amendments: 仅收窄 D0 测试 allowlist 和增加 pre-collection safety guard；没有修改科学 scope、C9/C10 evidence 或 Stage D outcome。
- failures_and_retries_retained: `D0_INITIAL_ATTEMPT_20260825` 的两项 blocker 及真实访问事实已在 machine record、人类决议和本日志保留。
- known_limitations_and_claim_boundaries: C10 effect 只覆盖固定 15 图/ViT-B/16/Value-path/`(0,9)`/matched baseline/fine-4×4-cell；translation/crop failure 必须披露；不主张一般 spatial robustness、原生单像素、完整 attention/class attribution、等价或普遍 ImageNet 优越；C10/Stage D cohort 不池化。
- next_task_authorized: true（只授权 D1 metadata-only 准备）
- authorization_issued_this_task: false
- execution_authorization_observed_valid: false
- next_task: D1

#### 当前权威的下一对话完整提示词

下面提示词替代首次失败尝试草案内的 D1 提示词；动态 trust anchor 是 `refs/archive/stage-d-d0-completion`，machine digest 为 `7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f`。

```text
你现在只执行 ViT-NNS fidelity-only Stage D 路线的 D1：在严格 metadata-only 边界内，把现有 500-row candidate reserve 正式化为 tracked、内容寻址、可验证的 Stage D registry，并预注册 5 个 audit overlay IDs。不要开始 D2，不要读取、stat/hash 或 decode 任一 reserve JPEG bytes，不要访问任何 Stage D outcome，不要运行模型/GPU，不要实现 runner，也不要签发 execution authorization。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

启动边界：
- 用 `git rev-parse 'refs/archive/stage-d-d0-completion^{commit}'` 解析 D0 completion anchor；要求 HEAD 与其相等且 index/worktree clean，否则 D1 BLOCKED。
- 验证 `config/stage_d_scope_decision.json` canonical digest 为 `7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f`，并验证 schema 与 `docs/stage_d_scope_decision.md` structured projection。
- 保留 D0 attempt history：首次尝试 `BLOCKED`，严格 rerun `PASS`；D0 只签发 `APPROVED_TO_PREPARE_STAGE_D`，execution/analysis 仍未授权。
- 只读复核 sealed C10 的既有 hashes/ref/inventory；不得从 C9 重建科学证据。C10 必须保持 `NO_GO_NOVELTY/FAIL_SPATIAL/BLOCKED/HOLM_BLOCKED_UNDERSPECIFIED/downstream_authorized=false/stage_d_started=false`。

第一步完整阅读本文 D1、§7–§9、当前 D0 rerun record，及 `docs/stage_d_scope_decision.md`、machine/schema、`../论文1实现审计与后续计划.md` 的 post-C10 与 Stage D/E/F、Stage C protocol/acceptance 的相关条款。

D1 只允许读取 candidate reserve、access log、prior-access ledger、source inventory/spec/archive identity、selection metadata 和 Stage C 20-row registry 的 metadata 内容。先绑定 metadata bytes 再严格解析；不得打开 metadata 引用的 live JPEG。生成 closed、tracked、内容寻址的 500-row registry package与不可变 pre-authorization attestation；验证连续顺序、ID/path/candidate-recorded hash 唯一性、官方 label metadata、与 Stage C 20 图互斥性，并用冻结 namespace 确定性选恰好 5 个 audit IDs。类别标签允许重复。

不得声称 reserve 绝对从未被查看，只能报告封闭 prior-access ledger 可证明的范围。不得修改 candidate access log，不得读取 outcome，不得运行模型/GPU，不得开始 D2 或实现运行链。完成后更新本文 D1 状态，追加 D1 completion record 和完整 D2 提示词；原则上一任务一提交，最终 clean。
```

## 10. D0 首次启动的完整可复制提示词

```text
你现在只执行 ViT-NNS fidelity-only Stage D 路线的 D0：冻结 post-C10 研究范围和 claim 边界。不要开始 D1，不要正式化或解封 500 图 reserve，不要读取任何 reserve JPEG bytes，不要运行模型/GPU，不要实现 Stage D runner，也不要修改或重算 C9/C10 evidence。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

启动状态说明：
- C10 completion commit 是 69e1fa887862e00da4441842c6de149fe2424225。
- 本轮规划工作可能留下未提交改动：../README.md、../论文1实现审计与后续计划.md、README.md 和 docs/阶段D-step-by-step.md。启动时逐份检查 git status 和 diff，不能仅凭路径推定归属。只有确认属于本路线且仍符合用户授权的改动才可纳入 D0 的规划/范围提交；其余改动保持原样并报告。不得为了追求 clean 而覆盖、stash、reset、删除或擅自提交用户改动。

不可改写的 C10 绑定：
- formal root: /data00/home/yeziqi.831/testcode/vitnns_c10_formal_92ad813_20260824
- aggregation producer/ref: 92ad81342ff22e2710df495a9961a3ab8c015a3a / refs/archive/c10-aggregation-producer
- Stage C report SHA-256/canonical: 9a16034d29bc87eaba852aa0e618928ac0e4ecee9bf745d765f42311dd4563b4 / 90232df2010f6a8fc771baeac3aaac44fee93d436597cf0cbdebe3cf8d2142e2
- decision SHA-256/canonical: 07fc5875e46018adf5dc0cd09967e273df3b443c1deeb1653d672d05103e513e / 60e112d50f22229474a634039ba71c7eed1adba081d8e1097f850a8a4b4d8dc0
- provenance SHA-256/canonical: 4dcca54fdc83ba7401070ef843636b5e67b68a8ce720fa70f729e407fc5717f0 / 4b26452041685cbb6c12d93139736e65fc4330fd44a64bf854848e7cff66c742
- manifest SHA-256/canonical: 507ca3c4ac210882a5918cc602cf7a4656aa398af76ff9f3638a77f3af42a0d8 / c71ae7c19941eab556a0ed9fdde553cb1a3fae914e149652df9b31cf3538720a
- whole-root inventory digest: 382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4
- historical status: decision=NO_GO_NOVELTY, acceptance=BLOCKED, blocker=HOLM_BLOCKED_UNDERSPECIFIED, FAIL_SPATIAL, downstream_authorized=false, stage_d_started=false。

第一步完整阅读：
1. docs/阶段D-step-by-step.md；
2. ../论文1实现审计与后续计划.md，尤其 post-C10 决议和 Stage D/E/F；
3. docs/阶段C-step-by-step.md，尤其 C10 completion record；
4. docs/阶段C_protocol.md §5–§10 与 docs/stage_c_acceptance.md C10；
5. sealed C10 status/report/decision/holm/provenance/manifest。

D0 必须完成：
1. 从 sealed root 只读复算并验证上述 C10 hashes、canonical digests、archive ref 和 whole-root inventory。
2. 新增 docs/stage_d_scope_decision.md 和严格 machine-readable scope/schema；名字可按仓库惯例微调，但必须单一权威。
3. 原样保留 C10 NO_GO_NOVELTY、FAIL_SPATIAL、BLOCKED、Holm blocker 和旧 downstream_authorized=false；不得把 C10 改名为 GO。
4. 记录 C10 15 fixed holdout 上 primary (0,9) 的 PASS_VALIDITY/PASS_EFFECT 是开展新 fidelity-only 研究的科学依据；限定模型、Value-path target、matched baseline、fine-4×4 endpoint 和样本边界。
5. 明确新 Stage D 是 prospective replication/extension，不需要 C+ 或另一轮小样本准入验证；Stage D 自身仍决定能否形成更广论文主张。
6. 冻结 claim allowlist/denylist：Stage D 不运行 spatial；C10 translation/crop failure 必须在研究设计、主结果和 limitations 披露；不声称原生单像素、完整 attention/class attribution、等价或普遍 ImageNet 优越。
7. D0 只能输出 APPROVED_TO_PREPARE_STAGE_D，不得签发 execution authorization，不得访问 reserve 图像或 outcome。
8. 增加静态/schema/hash/tamper tests；运行本任务和受影响测试。
9. 更新 docs/阶段D-step-by-step.md：D0 标 PASS 或 BLOCKED；追加完整 D0 completion record。PASS 时写出执行 D1 的完整提示词，BLOCKED 时写诊断提示词。
10. 原则上一任务一提交，最终工作树 clean；completion record 区分实际运行的测试与复用证据，并报告精确 commit。

D0 验收：scope 文档与机器记录一致；C10 sealed evidence 无写入且 hash 链有效；reserve_result_accessed=false、reserve_image_bytes_accessed=false、model_or_gpu_used=false；测试通过；文档和下一提示词完整。
```
