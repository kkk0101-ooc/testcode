# 阶段 D：fidelity-only 主实验 step-by-step 执行与跨对话交接文档

> 创建日期：2026-08-25。
> 文档性质：阶段 D 的路线图、任务状态表、数据访问边界和跨对话追加式交接日志。
> 当前状态：recovery-v2 已完成 canonical salvage C8、recovery C9、recovery C10，以及严格串行的 D0→D6 pre-reserve rebinding。D0–D2 是 current formal preparation PASS；D3 是 control-plane readiness PASS；D4–D6 是 implementation/readiness PASS。D5 不是正式 500-image evidence validation，D6 不是正式分析或科学结果。Stage D execution/analysis/reserve authorization 均未签发，D7–D10 均未启动。
> Recovery override（2026-09-04）：[`recovery_stage_d_d0_d6_amendment_20260904.md`](recovery_stage_d_d0_d6_amendment_20260904.md) 与 `config/stage_d_recovery_v2_binding_20260904.json` 是当前 pre-reserve identity overlay；历史 D0–D2 PASS 仅为 stale history，历史 D3–D6 PASS 仅为 synthetic/implementation evidence。
> 当前 next task：D7 recovery-v2 gate 正在严格按 `smoke → budget → producer/ref → authorize` 执行。当前 smoke 与 budget 已通过，但 producer/ref 尚未正式冻结，因此 execution authorization 尚未签发；D8 仍未初始化。
> D7 勘误：首个 pre-execution authority `0036dd8d...` 因 strict guard 发现通用 model loader 会对历史 Stage A calibration JPEG 做不必要的 `lstat` 而被 supersede；它未访问 Stage D reserve、未运行 inference、未创建 D8 root，不得用于 launch。当前需由 D7-v2 producer/authority 重新签发。
> 当前权威交接：[`recovery_stage_d_d0_d6_handoff_20260904.md`](recovery_stage_d_d0_d6_handoff_20260904.md)；总 false-PASS audit 为 `reports/stage_d_recovery_v2_20260904_r3/false_pass_audit.json`。以下旧 completion records 和 D7 提示词保留作历史审计，不构成当前授权。
> 使用规则：每个新对话先读文首状态、第 7 节执行协议、第 9 节最新权威记录和当前提示词，再读提示词列出的章节与上游证据；一次只执行一个 `D_i` 或一个明确编号的 `D8-Sxx` shard。未更新本文、未追加 completion record、未给出下一任务完整提示词的任务，不算完成。
> 测试原则：只做与当前任务和实际改动直接相关的必要测试，避免无关、重复或低价值测试。每个测试/验证批次开始前一次性说明测试项目、必要性和预计耗时；只有新增高成本测试或扩大影响面时才补充说明。

## 1. 为什么需要这份文档

Stage D 同时包含研究范围修订、500 图样本封存、机器可执行合同、GPU 批量运行、证据封存、统计决策和论文图表。把这些工作放在一个对话中会造成三个风险：

1. 在看到 Stage D 结果后继续修改主指标、阈值、样本或图表选择；
2. 把代码实现、正式执行和统计读取混在一起，难以确认结果由哪个 clean commit 产生；
3. 长时间 GPU 任务中断后，无法区分可恢复的基础设施中断与不可重试的科学失败。

本文把 Stage D 拆成 `D0–D10`。论文定性展示另设后续 `Stage F`，其选择规则必须在 Stage D 结果访问前冻结；Stage F 不改变 Stage D 主实验、统计或决策。主链是：

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
    D10 --> F[Stage F 定性展示]
```


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

这里的 `0.02` 在 PASS 条件中是 median 点估计门槛；CI lower `>0` 只证明方向为正，不要求 CI lower `>0.02`。FAIL 条件中的 CI upper `<0.02` 是预注册的早期失败门，表示 bootstrap 区间上界也低于最小效果阈值。

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
8. 定性 overlay 不在 D10 生成；如论文需要，Stage F 另建仅展示 protocol/recipe，允许两类展示：D1 预注册审计样本，以及在 D8/D10 结果访问前冻结规则、D10 后从 500 图中按规则选择的说明性案例。两类都不得进入数值统计或 decision。

数值图表脚本必须只读 D10 的 machine-readable paper data。不得从 PNG/JPEG 反推数值；不得把 C10 与 Stage D 样本合并后重新统计。

## 5. D0–D10 总路线与状态

下表保留恢复前的历史工程状态；当前执行门由文首 recovery override
覆盖，不能仅凭表中 D6 `PASS` 或 D7 `NOT_STARTED` 进入 D7。

任务状态只使用 `NOT_STARTED`、`IN_PROGRESS`、`PASS`、`BLOCKED`。`next_task` 单独记录，不使用 `NEXT` 混入状态。只有代码/文档、必要测试或静态验证、产物验证、completion record 和下一提示词都完成后，任务才能标记 `PASS`；若测试不适用，必须写明 `not_applicable` 及理由。

| 任务 | 状态 | 唯一目标 | 允许的数据访问 | 允许产生的结论 | 前置 |
|---|---|---|---|---|---|
| D0 | PASS（首次尝试 BLOCKED；严格重试 PASS） | 冻结 post-C10 fidelity-only scope | sealed C10；reserve metadata existence only | `APPROVED_TO_PREPARE_STAGE_D` | C10 evidence sealed |
| D1 | PASS | 正式化 500 图 registry 与 audit overlay | reserve/ledger/官方标签 metadata；禁止图像字节 | 无效果结论 | D0 PASS |
| D2 | PASS | 冻结 protocol/config/schema/statistics/exact workload | D0/D1 与 Stage C 合同；禁止图像和 outcome | 无效果结论 | D1 PASS |
| D3 | PASS | 实现 strict loaders、typed plan、preflight/authorization | synthetic/metadata only | 无效果结论 | D2 PASS |
| D4 | PASS | 实现 image-group runner、shards、artifact/provenance | synthetic/mock only | 无效果结论 | D3 PASS |
| D5 | PASS | 实现 finalize 与 source-recomputing validation | synthetic/tampered fixtures only | package/QC 状态 | D4 PASS |
| D6 | PASS | 实现 aggregation、statistics、decision、paper export | synthetic fixtures only | synthetic decision only | D5 PASS |
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
| D10 | 唯一 decision、关键数值/hash、paper artifacts、允许/禁止主张和全部偏差 | 根据结果生成 Stage E 协议准备、Stage F 展示/写作或诊断的完整提示词 |

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
- 运行 Stage D focused tests、实际受影响的 Stage A–C tests 和 C0 clean/tracked validator。全量 non-GPU suite 只在 producer commit 缺少可复用全量证据、共享基础代码发生变化且影响面无法可靠界定，或最小受影响集合出现回归信号时运行；不得仅因进入 D7 而机械重复；
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

论文贡献：**在 validity 允许时生成正式主结果表、数值主图、全部 supplementary 数据和唯一 Stage D decision；定性 overlay 留给 Stage F 的独立仅展示协议。**

完成后按结果生成下一提示词：

- `GO_FIDELITY_SCOPE`：先建立并冻结独立的 Stage E descriptive protocol，或进入 Stage F 展示/写作；
- `NO_GO_FIDELITY` / `INCONCLUSIVE_FIDELITY`：进入负向结果写作/诊断，不换配置重跑；
- `NO_GO_VALIDITY`：进入 validity 诊断，不能把这 500 张重新声明为未见确认集；
- workflow `BLOCKED`：只生成修复/审计提示词，不推进论文结论。

D10 应先原子生成并完整验证 machine-readable decision package，再从已验证 package 生成 paper data 和数值图。绘图失败不得回写或改变科学 decision；应保留 decision package，将 paper-export 状态标为 `BLOCKED` 并生成只修复导出的提示词。无论哪个结果分支，D10 都必须追加完整 completion record 和对应的可复制下一提示词。

## 7. 跨对话执行与交接协议

### 7.1 每个新对话开始时必须做

1. 先核对文首 `当前 next task`，再定位第 9 节最新权威 completion record；只把该记录后紧跟的提示词或其追加勘误当作当前指令。完整阅读第 7 节、该记录、当前提示词及提示词列出的正式 protocol/acceptance/config/schema；更早提示词仅作审计历史。
2. 检查 Git root、`HEAD`、branch、worktree/index clean 状态和上一记录的 commit。
3. 复核上一任务列出的关键 file SHA-256、canonical digest、archive ref 和 artifact status。
4. 只执行一个指定 `D_i`；D8 只执行一个明确的 `D8.0` 或 `D8-Sxx`。不得顺手启动下一任务。
5. 若有 dirty 文件，先判断归属；不得覆盖、stash、reset 或删除用户改动。不能建立可信基线时标记 `BLOCKED`。
6. 将第 5 节对应状态改为 `IN_PROGRESS`。D8 shard 另在 execution log 中维护 shard ledger。

### 7.2 每个任务结束时必须做

1. 完成该任务的实现、必要测试或静态验证、文档和产物验证，不用后续任务补齐当前验收；若测试不适用，记录 `not_applicable` 及理由。
2. 将第 5 节状态改为 `PASS` 或 `BLOCKED`；D8.0 和单个 D8 shard 使用第 8.1 节的子任务状态，父任务 D8 保持 `IN_PROGRESS`，只有 campaign completion record 确认所有 frozen shards terminal 后才能整体 `PASS`。
3. 在第 9 节追加 completion record。失败、偏差、中断、retry 和废弃 root 也必须保留。D8 的上一条权威交接记录是最新 coordinator completion commit 中的 D8.0/shard/campaign record，不是 execution worktree HEAD。
4. completion record 后紧跟下一任务的完整可复制提示词，不能只写“继续 D_i+1”。
5. 最终回复重复：状态、关键文件、测试前计划、实际测试、主要未运行测试类别及理由、commit/ref/hash、产物路径、数据访问事实、遗留问题和下一提示词。
6. 原则上一任务一提交并最终 clean。D0–D7/D9/D10 的代码或文档任务遵守普通提交规则；D8 正式 runtime evidence 必须绑定 D7 producer ref，不能归因给 main 上只追加 completion record 的 commit。
7. 只在上一任务 `PASS` 且 `next_task_authorized=true` 时进入正常下一任务；否则 next task 必须是诊断或修复。只有 D7 可以写 `authorization_issued_this_task=true`；D8–D10 只能在重新验证 D7 authorization digest 后写 `execution_authorization_observed_valid=true`。D0 的 `APPROVED_TO_PREPARE_STAGE_D` 只允许 D1–D7 准备工作。

### 7.3 最小充分测试与复用规则

1. 执行本任务测试/验证批次前，一次性向用户说明清单、范围、必要性、覆盖风险和粗略耗时；普通只读启动核验可合并为一项。只有新增高成本检查、扩大影响面或改变测试批次时，才需要补充说明。
2. 默认只运行当前任务新增/修改功能的 focused tests、被实际改动的共享模块测试，以及完成本任务验收不可替代的静态检查。不得运行与本任务无关、重复覆盖同一风险、只为增加数量或价值很低的测试。
3. 若精确 commit、相关文件 digest、冻结输入 digest 与依赖锁/关键运行环境一致，可把既有结果记录为“复用的基线证据”；不得伪称本次重新执行。若只能证明文件未漂移、不能证明运行环境一致，则只复用不依赖运行环境的静态证据。
4. 扩大测试范围前，先写清具体共享模块或依赖链，并先跑最小受影响集合。只有该集合出现回归信号，或仍无法界定影响面时，才运行更大范围或全量 non-GPU suite；“更保险”或“顺便跑一下”不是充分理由。
5. GPU、模型推理、JPEG decode、正式数据访问和长时间 smoke 还必须得到当前任务的明确授权；单元测试不能用昂贵正式 workload 替代。D7 的 producer gate 也按影响面选择测试，只有 exact producer commit 缺少有效全量证据或影响面仍无法界定时才运行全量 non-GPU suite。
6. D8 不因 completion 文档提交重跑测试套件；每个 shard 只执行启动所必需的 producer ref、authorization、immutable controls 和 shard preflight 验证。
7. Completion record 记录测试前计划、实际命令/结果/耗时、复用证据、计划内 skip/deselect，以及主要未运行测试类别及理由；不需要穷举整个 suite。失败后只重跑能验证修复的最小集合。

### 7.4 提示词写作规则

每条新的“下一对话完整提示词”都按以下顺序写，避免把背景、任务和禁令挤在一个长段落里：

1. **本次唯一任务**：用一句话说明要交付什么，并立即写明不得开始的下一任务。
2. **工作目录与开始条件**：列出 commit/ref/hash、clean 状态和 fail-closed 条件。
3. **必读材料**：只列本任务真正需要的文件和章节。
4. **必须完成**：用编号列出可验收的交付物、冻结值和输出合同。
5. **明确禁止**：单列数据访问、模型/GPU、结果读取和越级执行边界。
6. **测试计划**：要求先报告测试/验证批次的项目、必要性和预计耗时，再按第 7.3 节选择最小充分测试。
7. **完成与交接**：写清 PASS/BLOCKED 条件、Git 提交、completion record 和下一提示词要求。

首次出现的英文缩写或内部术语应附简短中文解释；历史事实、当前动作和未来条件必须分开写。动态值必须填实，不把选择留给下一位执行者猜测。

### 7.5 Git、producer 与正式 root

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
- test_plan_announced_before_execution:
- tests_executed_this_task:
- tests_not_run_and_reason:
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
（按第 7.4 节写入完整、可直接复制的提示词：唯一任务、工作目录、
开始条件、必读材料、交付物、禁止事项、failure/retry 规则、
测试前说明与最小充分测试、验收、Git 和本文更新要求。）
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

已有记录只允许追加勘误，不允许删除或改写失败史。历史记录内的提示词反映当时的交接状态，即使仍可复制也不代表当前授权；只有最新权威 completion record 后的提示词或其追加勘误可以执行。每个 `D_i` 末尾的“完成后”条款均要求使用第 8 节全部字段，并按第 7.4 节生成任务特定的完整下一提示词；动态 commit、hash、root 和测试结果必须填实，不能保留占位符。D10 也必须追加 completion record；其下一提示词须根据第 6 节 D10 的四类结果分支完整写出，而不是只写任务名。

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

### 2026-08-25 13:49 +0800 — D1 completion record

- status: PASS
- objective_completed: 在严格 metadata-only 边界内，将既有 500-row candidate reserve 正式化为 tracked、closed-schema、内容寻址、可确定性重建的 `STAGE_D_FIDELITY_V1` registry package，并从冻结 registry 顺序使用 Stage D 专属 namespace 预注册恰好 5 个 audit overlay IDs。
- authority_and_scope: 启动锚点为 `refs/archive/stage-d-d0-completion`，唯一解析到 `4c76d468e1368858c52ff8ef73e640463e15ac49`；D0 machine canonical digest 为 `7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f`。D1 只冻结 cohort metadata 和 audit overlay；没有开始 D2、没有冻结统计/运行合同、没有实现 runner、没有签发 execution authorization。
- git_sha_before: `4c76d468e1368858c52ff8ef73e640463e15ac49`
- git_sha_producer: none（D1 不生产 runtime/scientific evidence；registry package 由本 completion commit 跟踪）
- git_sha_postprocessor: none
- git_sha_completion: 由 `refs/archive/stage-d-d1-completion` 唯一解析；本记录与 D1 package/code/tests 同一提交，避免在提交内容中写自引用 SHA。
- tree_clean_at_boundaries: 新分支 `stage-d-d1-pass` 从 D0 completion ref 的 clean tree 创建；先将先前 fail-closed 诊断提交 `b45ce8858e831cca656d25cd1ce392ef66e6a64f` 保存在 `stage-d-d1-blocked-b45ce`。D1 提交后及最终复验后的 clean 状态在最终回复据实报告。
- upstream_commits_and_refs: D0 completion/ref `4c76d468e1368858c52ff8ef73e640463e15ac49` / `refs/archive/stage-d-d0-completion`；C10 completion `69e1fa887862e00da4441842c6de149fe2424225`；C10 producer/ref `92ad81342ff22e2710df495a9961a3ab8c015a3a` / `refs/archive/c10-aggregation-producer`。
- files_added_or_changed: `.gitignore`（仅精确 allowlist 新 package 的 6 个 JSON）；`src/vitnns/stage_d_registry.py`；`tests/d1_guard/sitecustomize.py`；`tests/test_stage_d_registry_contract.py`；`manifests/stage_d_fidelity_v1/{stage_d_registry_package.schema.json,source_binding.json,registry.json,pre_authorization_access_attestation.json,registry_metadata.json,manifest.json}`；本文。未跟踪 candidate source、JPEG、模型、runtime artifact 或 outcome。
- frozen_inputs_and_sha256: D0 machine/schema/Markdown file SHA-256 为 `217ef1fc00c3dbe971b35ff8d0793fb1eb928a7316db4ac0a5bcd34edfb6c06d` / `425f7fd16a1d004817dd2888f95d9b5b1818f7be6063bef522f05d438e6b5dfb` / `913aaf81f9c56c1bb71d0202384fa29e392af966ef49bd8e27e0ff4df1c8186a`；candidate reserve file/canonical/order digest 为 `1432183a0ce9c701e60cc28acc3596fe16bb253dfc013126bf3168b5af05099a` / `2e66fa2426fb7c42caefe7907a6ef1ec9a42adb1788528d59af9b5354eb0a99f` / `b187ff5cbacc1b595f06d12e309602b09965e01e7ee1461fd8a694354d4b8f50`；candidate access log file/canonical 为 `75a319820db12102dde22308ab2e12196e0e1ca6fcdd870e2943d22a0643a621` / `41478eee77f708a1158142d700fdfd65503b120aa672400af01fe14f9eb317fa`。其余 9 个 metadata source 的 file/canonical hashes 和逐项 proof scope 固定在 `source_binding.json`。
- config_plan_and_canonical_digests: `registry.json` file/canonical `9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b` / `514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028`；schema `2cbeb0536f680103ea5fb3d43e71532832c10fa5b432bed58cda2feaadf9f4fd` / `b5316e2571dac3d2ef82ccc39fbfc9044819d56d0d5bc9c5982972bcb79fba29`；manifest `f38bf9773ad172100c22f50ec3b4fe5251664e5c9f5c7ee78ae46532aec372fe` / `cf3e21b88488dcb928f9a98557c2ee9896a7c5210ddb8dda64427c1792fa299d`；attestation `610794455a1d75ef0243c5252c1e2bc14f6a4ea36f6c2ff2acf380dfd2c27644` / `9d89a6596de5027f72aeb76aca4aa600f03b6bb4cf45c8259840fd01e85638fe`；source binding `f1109304a23674bdadc62306bc927e7ac10b397f68754fae13967971b4f94cac` / `2eafef71035b95035a92c5ccfba7d536c3b924df4081def86fb0a1fa06dba2b8`；registry metadata `0633f0e78501686be0242332db1bc08d4cf069b360a3b2e51d5d85f03ab98c7f` / `e4c35f5935c119c46935cabd69ffbc6f9f168665d6643d4c253fc5ff0e3c6f52`。D2 protocol/config/statistics/run spec 未冻结。
- data_access_statement: 每个允许的 candidate/Stage C metadata 文件先经单一 no-follow、single-link regular-file descriptor 取得 size、mtime_ns、bytes 和 SHA-256，再从同一已绑定 bytes 严格解析；共解析 10 个 JSON 和 1 个 CSV metadata source。没有读取、stat/hash、decode 或 preprocess 任一 reserve JPEG；没有访问 Stage D prediction/logit/map/AUC/`delta_fid`/summary/outcome；没有加载模型、运行 CPU inference 或 GPU/CUDA；没有修改 candidate access log 或 C9/C10 evidence。`reserve_metadata_contents_accessed=true`、`reserve_image_bytes_accessed=false`、`reserve_result_accessed=false`、`model_or_gpu_used=false`、`execution_authorization=NOT_ISSUED`。
- artifacts_and_absolute_paths: `/data00/home/yeziqi.831/testcode/vitnns_rebuild/manifests/stage_d_fidelity_v1/`；无 runtime/formal execution/paper artifact。
- status_manifest_inventory_hashes: package 精确 6 个 tracked JSON；外层 manifest 绑定 schema、source binding、registry、attestation 和 registry metadata 的 bytes/SHA-256/canonical digest，manifest 自摘要另行验证，避免循环 hash。sealed C10 仍为 12 files、`524796892` bytes、whole-root digest `382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`，`scientific_results_recomputed=false`。
- expected_vs_observed_counts: registry `500/500`；selection order `0..499` 连续且唯一；unique sample IDs/paths/candidate-recorded image hashes `500/500/500`；official label mapping frame `1000/1000` classes、50 rows/class，registry unique WNIDs `399`（类别重复允许）；prior-access ledger coverage `50000/50000`，reserve marked `UNSEEN=500/500`、reserve access records `0`；Stage C disjoint overlap by ID/path/hash `0/0/0`；audit overlay `5/5`、statistical n increment `0`。
- tests_reused_from_prior_commit: D0 completion record 的最终 `D0_MODEL_FREE_NO_JPEG_V1` 结果仅作既有基线；C10 completion 的既有 regression 和 scientific evidence 不伪称为本任务重跑。
- tests_executed_this_task: 启动时重新运行 D0 exact allowlist：`24 passed in 0.22s`；D0 ref-tree machine/schema/Markdown file hash、canonical digest、Draft 2020-12 validation和 structured projection 现场通过；sealed C10 byte-hash/canonical/manifest/status/inventory/Git-ref verifier 通过。D1 首轮（排除尚未更新的路线静态断言）为 `8 passed, 1 failed, 1 deselected in 18.83s`，唯一失败是 symlink 测试夹具未创建父目录；修复夹具后为 `9 passed, 1 deselected in 19.08s`。首次完整 D1 suite 为 `9 passed, 1 failed in 18.81s`，唯一失败是路线测试期待的中文短语比 completion record 的等价措辞更窄；将断言改为同时检查 record 的 `5/5` 和 D2 提示词的“恰好 5 个”。移除非内容寻址的绝对 mtime 值并重新生成 package 后，完整 D1 suite 为 `10 passed in 18.87s`，最终提交前重跑为 `10 passed in 18.63s`；D0 保留 allowlist 为 `24 passed in 0.22s`。`py_compile`、Draft 2020-12 schema validation、exact deterministic rebuild 和 `git diff --check` 均通过；post-commit 验证在最终回复据实报告。
- acceptance_results: candidate 70+500 顺序从 49,273 个 ledger-eligible canonical representatives 按声明 hash 算法完整重建；500 行逐行与 50,000-row eligible metadata 的 identity/label/image-hash 记录一致；candidate access log `forbidden_access_detected=false` 且保持只读；closed schema、self-digests、outer manifest、source bindings、三维互斥和 audit derivation 均通过。
- package_complete: true
- scientific_validity: not_evaluated（D1 无 Stage D outcome）
- analysis_authorized: false
- budget_results: not_evaluated（属于 D7）；D1 未冻结统计或运行合同。
- paper_data_or_figures_generated: none
- deviations_or_protocol_amendments: none。一次内部诊断发现 candidate ledger 的 `coverage_rows_canonical_digest` 使用历史兼容摘要域 `canonical_sha256({"rows": coverage_rows})`；按其实际、已由顶层 self-digest 绑定的约定验证后通过，不修改 source。
- failures_and_retries_retained: 保留首次过期 `8fff...` 提示词导致的 fail-closed 诊断提交 `b45ce8858e831cca656d25cd1ce392ef66e6a64f` 于独立分支；本次从 D0 anchor 重新发起。D1 开发测试仅有上述测试夹具失败，无 source/candidate blocker、科学 failure、模型/GPU/OOM 或正式执行 retry。
- known_limitations_and_claim_boundaries: D1 只验证 candidate metadata 中已记录 image SHA-256 的格式、内部绑定和来源链，不复算 live JPEG；live-byte equality 延迟到 D8.0 且只能在 D7 execution authorization 后进行。prior-access 结论只表示 500 个 identity 未出现在声明且封闭的 ledger 范围中；source spec 明确 `historical_directory_byte_audit_performed=false`、`historical_directory_exhaustiveness_claimed=false`，不得声称 `RESERVE_ABSOLUTELY_NEVER_VIEWED`。C10 仍是 `NO_GO_NOVELTY/FAIL_SPATIAL/BLOCKED/HOLM_BLOCKED_UNDERSPECIFIED/downstream_authorized=false/stage_d_started=false`。
- next_task_authorized: true（只授权 D2 metadata-only protocol/statistics/run-contract freeze）
- authorization_issued_this_task: false
- execution_authorization_observed_valid: false
- next_task: D2

#### 下一对话完整提示词

```text
你现在只执行 ViT-NNS fidelity-only Stage D 路线的 D2：在任何 reserve JPEG、模型或 Stage D outcome 访问前，冻结正式 protocol、acceptance、closed config/schema、统计合同与 exact typed run-spec contract。不要开始 D3，不要实现 loader/planner/runner/finalize/aggregate，不要运行 smoke 或正式 workload，也不要签发 execution authorization。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

启动边界：
- D0 completion anchor 仍为 `refs/archive/stage-d-d0-completion` -> `4c76d468e1368858c52ff8ef73e640463e15ac49`，D0 machine canonical digest 为 `7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f`。
- D1 completion anchor 为 `refs/archive/stage-d-d1-completion`；必须用 `git rev-parse 'refs/archive/stage-d-d1-completion^{commit}'` 唯一解析，要求 HEAD 精确等于该 commit、index/worktree clean。重新验证 `manifests/stage_d_fidelity_v1/manifest.json` 的完整 6-file hash chain、closed schema、deterministic rebuild 与 source bindings。
- D1 registry file/canonical digest 为 `9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b` / `514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028`；package manifest file/canonical 为 `f38bf9773ad172100c22f50ec3b4fe5251664e5c9f5c7ee78ae46532aec372fe` / `cf3e21b88488dcb928f9a98557c2ee9896a7c5210ddb8dda64427c1792fa299d`；pre-authorization attestation file/canonical 为 `610794455a1d75ef0243c5252c1e2bc14f6a4ea36f6c2ff2acf380dfd2c27644` / `9d89a6596de5027f72aeb76aca4aa600f03b6bb4cf45c8259840fd01e85638fe`。
- D1 cohort 固定 500 rows，selection order `0..499`；与 Stage C 20 图在 sample ID/path/image hash 上重叠 `0/0/0`；audit overlay 恰好 5 个且 statistical n increment 为 0。D1 只验证 candidate-recorded image hashes，live equality 未评估。
- D1 pre-authorization facts 必须保持 `reserve_image_bytes_accessed=false`、`reserve_result_accessed=false`、`model_or_gpu_used=false`、`execution_authorization=NOT_ISSUED`；candidate access log 不得修改。prior-access claim 只限于声明且封闭 ledger 的范围，不得声称绝对未查看。
- sealed C10 继续保持 `NO_GO_NOVELTY/FAIL_SPATIAL/BLOCKED/HOLM_BLOCKED_UNDERSPECIFIED/downstream_authorized=false/stage_d_started=false`；不得重建 C10 科学结果。

第一步完整阅读 `docs/阶段D-step-by-step.md` 的 D2、§3、§5–§9 与最新 D1 record，D0 scope document/machine/schema，D1 package/schema/attestation/source binding，`../论文1实现审计与后续计划.md` 的 post-C10 与 Stage D/E/F，以及 Stage C protocol §6、§9–§10 和 acceptance C0/C5/C9/C10。只读复核 D0、D1 和 sealed C10 hash/ref/inventory 链。

D2 必须新增 `docs/阶段D_protocol.md`、`docs/stage_d_acceptance.md`、`docs/stage_d_config_contract.md`、closed `config/stage_d.json` 与 schema、run spec 与 schema，消除结果后自由度。固定 500 images、primary `(0,9)`、两种同目标方法、fine-4×4 endpoint、insertion/deletion 各 21 fractions、5 paired tie repeats、`run_stability=false`、无 random/center sanity、无 patch-scale/其他方法/层头；5 个 D1 audit IDs 只用于 repeatability/L2，`audit_repeatability_runs=2` 且不增加统计 n。typed contract 必须独立重算 500 image groups、1,000 units、1,010 map work items、210,000 curve points、总 211,010 logical work items。

冻结 `delta_fid`、全部 validity 条件、误分类保留、500/400 guard 及设计理由、paired image-bootstrap 次数/seed namespace/PCG64/linear percentile/tie/invalidity、互斥 decision priority/equality boundaries、retry/fresh-root 语义、retention、provenance、launch safety caps 与 completed campaign cost gate 的区别。Stage D spatial 仅 mandatory disclosure；`holm_status=NOT_APPLICABLE_NO_SECONDARY_INFERENCE` 只在没有 secondary inference 时成立且不修复 C10 blocker。

禁止读取、stat/hash/decode/preprocess 任一 reserve JPEG；禁止读取 Stage D predictions/logits/maps/AUC/delta_fid/summary/outcome；禁止模型、CPU inference、GPU/CUDA、attribution；禁止开始 D3、实现运行链或签发 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`。D2 tests 只能使用 metadata/静态或 synthetic fixtures，必须保留 D0/D1 access/authorization/hash tests。完成后更新本文 D2 为 PASS/BLOCKED，追加完整 completion record 与 D3/诊断提示词，一任务一提交并最终 clean。
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

### 2026-08-25 16:33 +0800 — D2 completion record

- status: PASS
- objective_completed: 在任何 reserve JPEG、模型或 Stage D outcome 访问前，冻结 `docs/阶段D_protocol.md`、`docs/stage_d_acceptance.md`、`docs/stage_d_config_contract.md`、closed `config/stage_d.json` / schema，以及 exact typed `config/stage_d_run_spec.json` / schema；消除 primary、workload、validity、statistics、decision、retry、retention、provenance 与 cost 分类的结果后自由度。
- authority_and_scope: D2 只冻结 metadata/static 合同；D0 的 `APPROVED_TO_PREPARE_STAGE_D` 仍是唯一准备授权。`execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`、`stage_d_execution_started=false`；下一任务只允许 D3 CPU-only control-plane implementation。
- git_sha_before: `d594eadae5b5aa715d12346f6dc93507f1e093a9`
- git_sha_producer: none（D2 不生产 runtime/scientific evidence）
- git_sha_postprocessor: none
- git_sha_completion: 由 `refs/archive/stage-d-d2-completion` 唯一解析；本记录与 D2 contracts/tests 同一提交，避免在提交内容中写自引用 SHA。
- tree_clean_at_boundaries: 原工作树在启动时有一处未归属的 `docs/阶段D-step-by-step.md` 删除改动，故未覆盖、stash、reset、删除或并入；改在 `/data00/home/yeziqi.831/vitnns_d2_worktree_20260825` 建立隔离分支 `stage-d-d2-pass`，从 clean D1 anchor 精确启动。D2 completion commit 与最终 clean 状态在最终回复据实报告。
- upstream_commits_and_refs: D1 completion/ref `d594eadae5b5aa715d12346f6dc93507f1e093a9` / `refs/archive/stage-d-d1-completion`；D0 completion/ref `4c76d468e1368858c52ff8ef73e640463e15ac49` / `refs/archive/stage-d-d0-completion`；D0 canonical `7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f`；C10 completion `69e1fa887862e00da4441842c6de149fe2424225`；C10 producer/ref `92ad81342ff22e2710df495a9961a3ab8c015a3a` / `refs/archive/c10-aggregation-producer`。
- files_added_or_changed: `docs/{阶段D_protocol.md,stage_d_acceptance.md,stage_d_config_contract.md,阶段D-step-by-step.md}`；`config/{stage_d.json,stage_d.schema.json,stage_d_run_spec.json,stage_d_run_spec.schema.json}`；`tests/d2_guard/sitecustomize.py`；`tests/test_stage_d_protocol_contract.py`；D0/D1 route assertions in `tests/test_stage_d_scope_contract.py` and `tests/test_stage_d_registry_contract.py`。未新增 loader/planner/runner/finalize/aggregate/CLI。
- frozen_inputs_and_sha256: D1 registry file/canonical `9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b` / `514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028`；D1 manifest file/canonical `f38bf9773ad172100c22f50ec3b4fe5251664e5c9f5c7ee78ae46532aec372fe` / `cf3e21b88488dcb928f9a98557c2ee9896a7c5210ddb8dda64427c1792fa299d`；D1 attestation file/canonical `610794455a1d75ef0243c5252c1e2bc14f6a4ea36f6c2ff2acf380dfd2c27644` / `9d89a6596de5027f72aeb76aca4aa600f03b6bb4cf45c8259840fd01e85638fe`；D1 source binding file/canonical `f1109304a23674bdadc62306bc927e7ac10b397f68754fae13967971b4f94cac` / `2eafef71035b95035a92c5ccfba7d536c3b924df4081def86fb0a1fa06dba2b8`；sealed C10 whole-root digest `382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`。
- config_plan_and_canonical_digests: `config/stage_d.json` file/canonical `88fdc9cb2b4a5bd74ff7d8e68c79463cafac6437e29b5564c2115a13f1f1136d` / `21f9dcf1d615f2531f7d7d9a36319b958e06d0606c325890dd31ad89b7c2caae`；`config/stage_d_run_spec.json` file/canonical `43c61f190a9ad5811d878a7516039287f3914548f5444934fc06f41769c97d20` / `56a1d946c96ce5f86699e52a359557c1cdfd6ebbfd5aa43646d5e3a2d6428c85`；config schema file SHA-256 `64fed10aed29a1c97807ebee2d91a29c3dc218993909b4f0c8fab345d8f78c32`；run-spec schema file SHA-256 `ea23b61bb0d9f42fd6a18498c1c6d900b82f159da2b9f87ef51f7c0428d87646`。文档 file SHA-256：protocol `93ff7cb032a51f5fe9c97809f6893a8663718173fbac02a736a1452639d0bd12`、acceptance `f9075786cf46de66337f669849a6522abe7caa35ceb4ddd65b44cc8ccd864fbc`、config contract `051424dc02a7d9e3c6da801771d5d5b05fe589c504fda0867c2ef61d0d036993`。
- data_access_statement: 只读取 Git、Markdown、源码和 JSON/CSV metadata，并在 strict guards 下重验 D0/D1 与 sealed C10 integrity。没有读取、stat/hash/decode/preprocess 任一 reserve JPEG；没有访问 Stage D prediction/logit/map/AUC/`delta_fid`/summary/outcome；没有加载模型、运行 CPU inference 或 GPU/CUDA/attribution；没有修改 candidate access log、D1 package 或 C9/C10 evidence。`reserve_metadata_contents_accessed=true`、`reserve_image_paths_statted=false`、`reserve_image_bytes_accessed=false`、`reserve_result_accessed=false`、`model_or_gpu_used=false`。
- artifacts_and_absolute_paths: `/data00/home/yeziqi.831/vitnns_d2_worktree_20260825/vitnns_rebuild/docs/阶段D_protocol.md`；`.../docs/stage_d_acceptance.md`；`.../docs/stage_d_config_contract.md`；`.../config/stage_d.json`；`.../config/stage_d.schema.json`；`.../config/stage_d_run_spec.json`；`.../config/stage_d_run_spec.schema.json`。无 runtime/formal/paper artifact。
- status_manifest_inventory_hashes: D1 package 仍为 6 files 且完整 hash/canonical/rebuild/source chain `PASS`；D2 不建立 runtime manifest，四个 config/run-spec 文件及三个文档由 self-digest、source bindings、closed schemas、completion commit 和本文 file hashes 共同固定。sealed C10 仍为 12 files、`524796892` bytes、whole-root digest `382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`。
- expected_vs_observed_counts: 500/500 image groups、1,000/1,000 execution units、1,000 primary clean maps、10 audit repeat-clean maps、1,010/1,010 map work items、10,000 fidelity curves、210,000/210,000 curve points、211,010/211,010 logical work items；audit statistical n increment `0`。
- tests_reused_from_prior_commit: D1 completion record 中的最终 `10 passed` 与 D0 `24 passed` 只作既有基线；本任务另行重跑下列门禁。
- tests_executed_this_task: 启动现场在 D1 guard 下通过 D0 projection、D1 6-file hash/schema/rebuild/source bindings 和 sealed C10 ref/inventory verifier；原工作树正确 D1 guard 重跑 `10 passed in 20.09s`；隔离 D2 工作树正确 D0 exact allowlist 重跑 `24 passed in 0.19s`；D2 focused 首轮（路线状态测试 deselected）`12 passed, 1 deselected in 0.13s`，路线更新并补强 mutation matrix 后完整 `23 passed in 0.28s`。曾误用 D2 guard 合并运行 D1+D2，得到 `4 failed, 19 passed`：1 项因 D1 guard 环境变量未设，3 项因隔离 worktree 不含 ignored candidate metadata；这不是 source/合同失败，随后分别在正确 guard/worktree 重跑通过。`py_compile`、Draft 2020-12 schema/instance validation、self-digest/config-binding、exact registry/audit order、`git diff --check` 均通过；post-commit 复验在最终回复据实报告。
- acceptance_results: D2 合同冻结唯一 primary `(0,9)`、同 target 两方法、fine-4×4 only、insertion/deletion 21 fractions、5 paired tie repeats、5 audit overlays 与 2 clean-map runs；冻结全部 validity、misclassification、500/400 precision rationale、10,000-draw PCG64/linear paired bootstrap、ordered mutually-exclusive equality rules、retry/fresh-root、retention/provenance、sharding algorithm 和分离的 safety/campaign cost gates。所有 forbidden workload 显式关闭。
- package_complete: true
- scientific_validity: not_evaluated（D2 不读取或产生 Stage D outcome）
- analysis_authorized: false
- budget_results: not_evaluated；D2 保留/分类 caps，D7 才能从真实 typed plan profile 独立验收预算。
- paper_data_or_figures_generated: none
- deviations_or_protocol_amendments: 为消除已有草案歧义，选择 Stage-D 专属 tie/bootstrap/figure namespaces，明确 tie repeat `ddof=0`、跨图 SD `ddof=1`、偶数 median、one-retry/two-attempt 上限、fresh-root authorization、D7 延迟 shard count 和完成期 aggregate cost 不作中途停止。无结果驱动 amendment。开发中发现一次 canonical digest 计算错误地包含 shell newline；在任何提交前按 frozen no-newline canonical bytes 修正并增加动态 self-digest 测试。
- failures_and_retries_retained: 保留上述错误 guard 组合测试与 canonical newline 诊断；均为 D2 开发验证失败，不是 scientific/source failure，不产生 runtime retry 或 formal root。
- known_limitations_and_claim_boundaries: D2 合同不能证明 live JPEG 等于 candidate-recorded hash，验证延迟到 D8.0 且须先有 D7 authorization；prior-access claim 仍只限封闭 ledger 范围。Stage D spatial 仅 mandatory disclosure，`NOT_APPLICABLE_NO_SECONDARY_INFERENCE` 不修复 C10 Holm blocker；不允许 pooled `n=515`、equivalence、native-pixel、完整 attention/class attribution 或普遍 ImageNet 优越主张。
- next_task_authorized: true（只授权 D3 metadata/synthetic、CPU-only control-plane implementation，不授权 smoke 或正式 workload）
- authorization_issued_this_task: false
- execution_authorization_observed_valid: false
- next_task: D3

#### 下一对话完整提示词

```text
你现在只执行 ViT-NNS fidelity-only Stage D 路线的 D3：把 D0–D2 冻结合同实现为 fail-closed、CPU-only 的 strict loader、typed plan、预算预检与 preparation-only authorization builder/validator/CLI。不要开始 D4，不要实现 evidence producer、image loader、model backend、runner、sharding runtime、finalize、aggregate 或 paper export；不要运行 smoke 或正式 workload，也不要签发 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

启动边界：
- D2 completion anchor 是 `refs/archive/stage-d-d2-completion`；必须用 `git rev-parse 'refs/archive/stage-d-d2-completion^{commit}'` 唯一解析，要求 HEAD 精确等于该 commit且 index/worktree clean，否则 D3 `BLOCKED`。
- 重新验证 D0 ref/record canonical `7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f`；D1 ref/package 6-file chain，registry file/canonical `9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b` / `514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028`；D2 config file/canonical `88fdc9cb2b4a5bd74ff7d8e68c79463cafac6437e29b5564c2115a13f1f1136d` / `21f9dcf1d615f2531f7d7d9a36319b958e06d0606c325890dd31ad89b7c2caae`，run spec file/canonical `43c61f190a9ad5811d878a7516039287f3914548f5444934fc06f41769c97d20` / `56a1d946c96ce5f86699e52a359557c1cdfd6ebbfd5aa43646d5e3a2d6428c85`，及两份 schema 和三份 D2 文档；重验 sealed C10 hash/ref/inventory，不重建科学值。
- 继续保持 `execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`。不得读取、stat/hash/decode/preprocess 任一 reserve JPEG，不得访问 Stage D outcome，不得加载模型、运行 inference/GPU/CUDA/attribution。

第一步完整阅读 `docs/阶段D-step-by-step.md` 的 D3、§3、§5–§9 与最新 D2 record，`docs/阶段D_protocol.md`、`docs/stage_d_acceptance.md`、`docs/stage_d_config_contract.md`，closed `config/stage_d.json` / schema、`config/stage_d_run_spec.json` / schema，D0 scope 与 D1 package/attestation/source binding，以及 Stage C 对应 strict config/plan/provenance primitives。

D3 必须完成：
1. 实现 strict D2 config/run-spec loaders，拒绝 duplicate keys、NaN/Infinity、unknown/missing fields、bool-as-int、路径逃逸、symlink/hardlink、自摘要或 source binding drift，并现场验证 D0/D1/C10 refs/hashes/状态。
2. 实现 Stage D 独立 typed plan，不放宽 Stage C planner：按 D1 registry 顺序构造 500 content-addressed image groups，每组严格 NNS→matched 两 units；显式 typed clean-map、audit-repeat-clean-map 与 fine curve-point coordinates；共享 pair bindings。
3. 独立重算并拒绝任何 drift：500 groups、1,000 units、1,010 map work items、10,000 curves、210,000 curve points、211,010 logical work items；拒绝 spatial/stability、random/center sanity、patch scale、额外方法、额外层头、非 audit repeat 和 audit 增加统计 n。D7 前不冻结 shard count，只验证未来 `selection_order % shard_count` 算法。
4. 实现 metadata-only estimate/preflight，按 typed unit/profile 求和，不把 logical work items 当 calls；保留 D2 launch safety caps 与 completed-campaign gates 的分类，D7 才填实测 profile/最终预算。
5. 实现 preparation-only authorization builder/validator，绑定 D0/D1/D2、plan、budget、code/dependency/model metadata，并原样保留 C10 `NO_GO_NOVELTY/FAIL_SPATIAL/BLOCKED/HOLM_BLOCKED_UNDERSPECIFIED/downstream_authorized=false/stage_d_started=false`。D3 输出只能是准备 artifact，`execution_authorization=NOT_ISSUED`，不得冒充 D7 authorization。
6. 增加独立 Stage D CLI 的 `validate-contracts`、`plan`、`estimate`、`prepare-authorization`（名字可等义但职责必须分层）；所有 metadata checks 在任何未来模型构造前完成。不得增加 `init/run-shard/finalize/aggregate`。
7. 用 metadata/static/synthetic fixtures 覆盖 schema/duplicate/nonfinite/tamper/source drift、错样本数/顺序/方法/scale/head、forbidden work item、audit overlay、pair binding、deterministic digest、预算分类和 preparation-authority 边界；保留 D0/D1/D2 access/hash tests。
8. 更新本文 D3 为 PASS/BLOCKED，追加完整 D3 completion record 和 D4/诊断提示词；原则上一任务一提交并最终 clean。

验收：无需且不得访问 reserve JPEG/model/outcome 即可确定性生成并验证 exact typed plan；所有 500/1,000/1,010/10,000/210,000/211,010 counts 与 source bindings 匹配；所有 forbidden scope fail closed；D3 未开始 D4，未运行 smoke/formal workload，未签发 execution authorization。
```

### 2026-08-25 17:55 +0800 — D2 后 D3 提示词勘误

> 勘误性质：不改变 D2 的 `PASS` 结论、D2 frozen contracts 或任何 hash。上一条 D3 提示词仍是 D2 completion record 的历史内容；为降低误读和过度测试风险，当前执行 D3 时应使用下面的易读版提示词。它只授权 D3，不授权 D4、smoke、formal workload 或 execution authorization。

#### 当前权威 D3 提示词

```text
本次唯一任务：只执行 Stage D 的 D3。

目标是把 D0-D2 已冻结的合同实现成 CPU-only 控制层，包括 strict loaders、typed plan、metadata-only estimate/preflight、preparation-only authorization builder/validator 和对应 CLI。D3 不运行实验，不读取 reserve JPEG，不加载模型，不开始 D4，也不签发 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

开始条件：
1. `refs/archive/stage-d-d2-completion` 必须唯一解析到 `c57438640e882604031775c944d8ff5ce7932713`。
2. 从该 ref 的 clean tree 启动；若当前工作树已有 dirty 文件，先逐项判断归属。不得覆盖、stash、reset、删除或夹带无关改动；若无法建立可信基线，D3 标记 `BLOCKED`。
3. 只读核验 D0/D1/D2/C10 binding，不重建科学结果：
   - D0 canonical：`7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f`；
   - D1 registry file/canonical：`9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b` / `514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028`；
   - D2 config file/canonical：`88fdc9cb2b4a5bd74ff7d8e68c79463cafac6437e29b5564c2115a13f1f1136d` / `21f9dcf1d615f2531f7d7d9a36319b958e06d0606c325890dd31ad89b7c2caae`；
   - D2 run spec file/canonical：`43c61f190a9ad5811d878a7516039287f3914548f5444934fc06f41769c97d20` / `56a1d946c96ce5f86699e52a359557c1cdfd6ebbfd5aa43646d5e3a2d6428c85`；
   - sealed C10 whole-root digest：`382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`。
4. 继续保持 `execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`、`stage_d_execution_started=false`。

必读材料：
1. `docs/阶段D-step-by-step.md`：文首状态、第 3 节、第 5 节状态表、第 7 节、第 8 节模板、第 9 节最新 D2 record 和本勘误提示词。
2. `docs/阶段D_protocol.md`、`docs/stage_d_acceptance.md`、`docs/stage_d_config_contract.md`。
3. `config/stage_d.json`、`config/stage_d.schema.json`、`config/stage_d_run_spec.json`、`config/stage_d_run_spec.schema.json`。
4. D0 scope machine/schema/document、D1 registry package/attestation/source binding，以及 Stage C 中可复用的 strict config/plan/provenance primitives。

D3 必须完成：
1. 实现 strict config/run-spec loaders：拒绝 duplicate keys、NaN/Infinity、unknown/missing fields、bool-as-int、路径逃逸、symlink/hardlink、自摘要漂移和 source binding drift。
2. 实现 Stage D 独立 typed plan。这里的 typed plan 是 CPU-only、metadata/synthetic 的计划对象，不是 runner；它只从 D1 registry 和 D2 run spec 构造 expected coordinates、pair bindings、work-item taxonomy 和 counts。
3. typed plan 必须确定性生成并验证：500 image groups、1,000 units、1,010 map work items、10,000 curves、210,000 curve points、211,010 logical work items。
4. typed plan 必须 fail closed 拒绝：spatial/stability、random/center sanity、patch-scale、额外方法、额外层头、非 audit repeat、audit 增加统计 n、错样本数、错顺序、错 scale 或错 primary。
5. 实现 metadata-only estimate/preflight：按 typed unit/profile 求和，不把 logical work items 当成模型调用数；保留 D2 的 launch safety caps 与 completed-campaign cost gates 分类，D7 才填实测 profile 和最终预算。
6. 实现 preparation-only authorization builder/validator：绑定 D0、D1、D2、plan、budget、code/dependency/model metadata，并原样保留 C10 `NO_GO_NOVELTY/FAIL_SPATIAL/BLOCKED/HOLM_BLOCKED_UNDERSPECIFIED/downstream_authorized=false/stage_d_started=false`。D3 artifact 只能说明准备链完整，不能冒充 D7 execution authorization。
7. 增加 CLI：`validate-contracts`、`plan`、`estimate`、`prepare-authorization`，名称可按仓库风格微调，但职责必须分层。不得新增 `init`、`run-shard`、`finalize`、`aggregate` 或 paper export。
8. 更新本文 D3 状态并追加完整 completion record。PASS 时写出 D4 提示词；BLOCKED 时写诊断/修复提示词。

明确禁止：
- 不读取、stat、hash、decode 或 preprocess 任一 reserve JPEG。
- 不访问 Stage D prediction、logit、map、AUC、`delta_fid`、summary 或其他 outcome。
- 不加载模型，不运行 CPU inference、GPU/CUDA 或 attribution。
- 不实现 evidence producer、image loader、model backend、runner、sharding runtime、finalize、aggregate 或 paper export。
- 不运行 smoke 或正式 workload，不签发 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`。
- 不修改 candidate access log、D1 package、D2 frozen contracts 或 C9/C10 evidence，除非 D3 因发现合同自相矛盾而按 BLOCKED 记录诊断。

测试要求：
1. 测试前先一次性说明本批次测试/验证清单、范围、必要性、覆盖风险和预计耗时。普通只读 ref/digest/schema binding 核验可合并成一项；只有新增高成本测试、扩大影响面或改变批次时才补充说明。
2. 只运行与 D3 代码、CLI、typed plan、authorization 边界和被实际修改共享模块直接相关的必要测试。优先使用 metadata、static 和 synthetic fixtures。
3. 对 D0/D1/D2/C10 只做当前 HEAD 上的 ref/digest/schema binding 静态核验；不要重跑既有 D0/D1/D2/C10 suites。只有 D3 修改了它们共享的 loader/schema/assertion/hash 逻辑时，才运行最小受影响节点。
4. 不运行无关、重复覆盖同一风险或低价值的测试；不运行全量 non-GPU suite、GPU 测试、模型测试、图像测试、smoke 或正式 workload。若确需扩大范围，先写清具体共享模块或依赖链，并先跑最小受影响集合。
5. 建议最小 D3 验证集合包括：new D3 focused tests；CLI help/contract smoke with synthetic or metadata-only inputs；schema meta-validation；typed-count deterministic test；forbidden-work-item negative tests；preparation-authorization boundary tests；只读 D0/D1/D2/C10 binding verifier；`py_compile`；`git diff --check`。根据实际修改范围增减，并记录理由。
6. Completion record 分开记录测试前计划、实际命令/结果/耗时、复用证据、计划内 skip/deselect，以及主要未运行测试类别及理由；不需要穷举整个 suite。

完成与提交：
1. D3 通过时，把第 5 节 D3 状态改为 `PASS`；失败或无法建立可信边界时改为 `BLOCKED`。
2. 在第 9 节追加 D3 completion record，填写全部模板字段、关键 hashes、数据访问事实、测试计划/结果和未运行测试理由。
3. 只提交 D3 相关文件，不夹带无关改动；原则上一任务一提交。提交后确认 worktree clean，并在最终回复报告 commit、关键文件、验证结果和下一任务。
```

### 2026-08-25 21:26 +0800 — D3 completion record

- status: PASS
- objective_completed: 已把 D0–D2 frozen contracts 实现为独立、fail-closed、CPU-only Stage D 控制层：strict config/run-spec loader、同一 descriptor snapshot 的上游 manifest verifier、content-addressed typed plan、metadata-only estimate/preflight、preparation-only authorization builder/validator，以及仅含 `validate-contracts`、`plan`、`estimate`、`prepare-authorization` 的 CLI。没有实现或启动 D4 runtime。
- authority_and_scope: D0 的 `APPROVED_TO_PREPARE_STAGE_D` 仍是唯一准备授权；`execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`、`stage_d_execution_started=false`、scientific result `NOT_EVALUATED`。D3 preparation artifact 不是 D7 execution authorization，且代码不签发 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`。
- git_sha_before: `f3cc4b5ea4688de98828f76dcc63f066a816ef74`；它的直接父提交是冻结 D2 anchor `c57438640e882604031775c944d8ff5ce7932713`，工作树启动时 clean；该单一后继提交只包含当前提示词勘误和路线澄清，属于本任务明确要求读取/沿用的权威 handoff，不是未归属改动。
- git_sha_producer: SELF（D3 control-plane implementation commit；最终 SHA 在提交后以 Git 为准并在最终回复报告）
- git_sha_postprocessor: none
- git_sha_completion: SELF
- tree_clean_at_boundaries: 启动时 clean；实现期间仅出现本 record 列出的 D3 文件；提交后 clean 状态由 post-commit 检查与最终回复报告。未 stash、reset、删除、覆盖或夹带用户改动。
- upstream_commits_and_refs: D2 `refs/archive/stage-d-d2-completion` / `c57438640e882604031775c944d8ff5ce7932713`；D1 `refs/archive/stage-d-d1-completion` / `d594eadae5b5aa715d12346f6dc93507f1e093a9`；D0 `refs/archive/stage-d-d0-completion` / `4c76d468e1368858c52ff8ef73e640463e15ac49`；C10 producer `refs/archive/c10-aggregation-producer` / `92ad81342ff22e2710df495a9961a3ab8c015a3a`，C10 completion `69e1fa887862e00da4441842c6de149fe2424225`。
- files_added_or_changed: 新增 `src/vitnns/{stage_d_config.py,stage_d_manifest.py,stage_d_plan.py,stage_d_authorization.py,stage_d_cli.py}`、`tests/d3_guard/sitecustomize.py`、`tests/test_stage_d_{config,manifest,plan,authorization,cli}.py`；修改 `src/vitnns/stage_d_registry.py` 以允许对同一已捕获 bytes snapshot 验证 D1 package；修改 `pyproject.toml` 注册 `vitnns-stage-d` 并把 loader 运行期需要的 `jsonschema==4.23.0` 从 test extra 提升为基础依赖；最小调整受 D3 路线状态影响的 Stage D tests；更新 `README.md`、本文与仓库顶层 `论文1实现审计与后续计划.md` 的 D3/D4 状态。未修改 D0/D1/D2 frozen JSON/schema/document、candidate access log 或 C9/C10 evidence。
- frozen_inputs_and_sha256: D0 canonical `7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f`；D1 registry file/canonical `9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b` / `514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028`；D1 package manifest file/canonical `f38bf9773ad172100c22f50ec3b4fe5251664e5c9f5c7ee78ae46532aec372fe` / `cf3e21b88488dcb928f9a98557c2ee9896a7c5210ddb8dda64427c1792fa299d`；D1 source binding file/canonical `f1109304a23674bdadc62306bc927e7ac10b397f68754fae13967971b4f94cac` / `2eafef71035b95035a92c5ccfba7d536c3b924df4081def86fb0a1fa06dba2b8`；sealed C10 whole-root digest `382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`。
- config_plan_and_canonical_digests: D2 config file/canonical `88fdc9cb2b4a5bd74ff7d8e68c79463cafac6437e29b5564c2115a13f1f1136d` / `21f9dcf1d615f2531f7d7d9a36319b958e06d0606c325890dd31ad89b7c2caae`；D2 config schema SHA-256 `64fed10aed29a1c97807ebee2d91a29c3dc218993909b4f0c8fab345d8f78c32`；D2 run-spec file/canonical `43c61f190a9ad5811d878a7516039287f3914548f5444934fc06f41769c97d20` / `56a1d946c96ce5f86699e52a359557c1cdfd6ebbfd5aa43646d5e3a2d6428c85`；run-spec schema SHA-256 `ea23b61bb0d9f42fd6a18498c1c6d900b82f159da2b9f87ef51f7c0428d87646`；verified source-manifest digest `146ddf1eb84d4dd8fb5c919e0d531d63532812ba5af1b15e1de4fb04eae1bad5`；typed plan canonical/coordinate digest `5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443` / `cc560e2a2d4e1220ccd812c073119a2b679de213f9770358f7decf51824661fb`；no-profile estimate/preflight digest `30527205d2b349e22c5d4fb9225ce54cd89f2157d5c78313c5483ddeebc92531` / `b117272df621086f4a8b75368d697e7ad099d58908b5195ed1528ebbe30e1fa6`。
- data_access_statement: 只读取 Git、Markdown、Python source、JSON/CSV metadata、`config/model.json` 和 `requirements.lock.txt`；sealed C10 仅做既有 byte/hash/manifest/status/ref/inventory 静态验证，不重算科学结果。没有读取、stat、hash、decode 或 preprocess 任一 reserve JPEG；没有访问 Stage D prediction/logit/map/AUC/`delta_fid`/summary/outcome；没有加载模型或 state-dict，没有运行 CPU inference、GPU/CUDA 或 attribution；没有修改 candidate access log、D1 frozen package、D2 frozen contracts 或 C9/C10 evidence。`reserve_metadata_contents_accessed=true`、`reserve_image_paths_statted=false`、`reserve_image_bytes_accessed=false`、`reserve_result_accessed=false`、`model_or_gpu_used=false`。
- artifacts_and_absolute_paths: 代码位于 `/data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_d_{config,manifest,plan,authorization,cli}.py`；CLI 为 `vitnns-stage-d`。仅为验证生成临时 `/tmp/stage_d_d3_plan.json` 与 `/tmp/stage_d_d3_estimate.json`，不是正式 evidence、未纳入提交，可删除；无 runtime/formal/paper artifact。
- status_manifest_inventory_hashes: D1 package 仍为 6 files，file/canonical chain 由同一 descriptor-backed bytes snapshot 验证；C10 仍为 12 files、`524796892` bytes、whole-root digest `382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`。D3 源文件 SHA-256：config loader `31c4df8e49cb20b87475171ebfe87f17331e1edcb2779c737318fc2543cc5c20`、manifest verifier `3a0075842b15c82a446365bd9b7f772ac1a8e585eb208e8e168d97ce04113223`、plan `ab74e4c61e4fcf73421fdb369f0528acb39075081ae714c9ac56a9f86b60b23d`、authorization `b2f49e7354e26945e65eb6f1b3c917a379f70b1564f25b80cb60e2c66d2c92f9`、CLI `b761bd6677137b0c963bd596aeb083e2400d5b4c9a3ca2ea4cb57f3069e67198`。
- expected_vs_observed_counts: image groups `500/500`；execution units `1,000/1,000`；primary clean maps `1,000/1,000`；audit repeat-clean maps `10/10`；map work items `1,010/1,010`；fidelity curves `10,000/10,000`；curve points `210,000/210,000`；logical work items `211,010/211,010`；statistical n `500/500`，audit increment `0/0`。
- tests_reused_from_prior_commit: D2 completion record 的 focused `23 passed`、D1 `10 passed`、D0 `24 passed` 仅作冻结基线证据；因 D3 未修改 frozen D0/D1/D2/C10 scientific contracts，未重跑它们的全套 suites。
- test_plan_announced_before_execution: 已在测试前一次性声明 D3 focused tests、CLI help/metadata-only smoke、schema meta-validation、typed deterministic counts/negative scope、preparation authorization boundary、只读 binding verifier、`py_compile` 和 `git diff --check`；预计 3–6 分钟；明确排除全量 non-GPU、GPU、模型、图像、smoke/formal workload。后续只因修复失败节点及共享 D1 bytes validator/最终 Git hardening 扩大最小受影响节点，并在执行前说明。
- tests_executed_this_task: 首次命令误用不存在的 `/usr/bin/time`，在 pytest 收集前退出 127；改用 shell `time` 后首轮 `115 passed, 6 failed in 238.93s`，失败为 3 个实现/测试文本边角和 3 个最终路线状态尚未写入；针对前四项修复重跑 `4 passed in 0.19s`。第二轮 focused（尚不含最终路线断言）`118 passed, 2 failed in 249.59s`，两项均为正确 fail-closed 后错误消息 regex 过窄；修复后参数节点 `2 passed in 0.10s`。D1 same-snapshot validator 最小回归 `10 passed in 7.11s`；manifest/auth/CLI hardening 回归 `55 passed in 175.65s`；Git sanitation/ref/snapshot 节点 `5 passed in 0.83s`；完成文档后整批先因 3 个临时 D1 fixture 未隔离新 repo-identity gate 得到 `122 passed, 3 failed in 258.34s`，fixture 修复节点 `3 passed in 0.07s`，最终合并批次 `125 passed in 254.10s`。CLI `--help` PASS；真实 `validate-contracts` PASS（约 3.52s）；真实 `plan` 成功生成（约 29.52s，临时摘要脚本因索引不存在的 `dimensions.sharding` 发生一次 `KeyError`，产品命令输出本身完整有效）；真实 `estimate` PASS（约 29.18s）。新增 Python 文件 `py_compile` PASS；两份 schema meta-validation/current-instance validation PASS；路线状态最小 3 nodes `3 passed in 0.08s`；`git diff --check` PASS；post-commit checks 在最终回复据实补充。
- tests_not_run_and_reason: 计划内没有 skip/deselect。未运行全量 non-GPU suite、D0/D1/D2/C10 suites（仅运行共享 D1 validator 一个节点和四个静态 contract 节点）、GPU/model/image tests、Stage C integration、smoke、正式 workload、runner/finalize/aggregate/paper tests；原因是 D3 只修改 metadata control plane，扩大范围会重复覆盖或违反本阶段访问边界。`prepare-authorization` 的真实 CLI 在提交前必然因 dirty tree fail closed，故由 issued synthetic metadata/clean-Git test double 覆盖，提交后再执行真实命令。
- acceptance_results: strict loaders 拒绝 duplicate/nonfinite/unknown/missing/bool-as-int/path/symlink/hardlink/self-digest/source drift；上游 D0/D1/D2/C10 refs/hashes/status 静态验证通过；typed plan 顺序、pair bindings、work taxonomy、完整坐标 digest 与 exact counts 通过，所有 forbidden scale/sanity/stability/method/head/repeat/stat-n 变体 fail closed；estimate 以 typed profile 相加且 `logical_work_items_used_as_call_count=false`，D2 两类 cost gates 分离，D7 profile/final budget 明确保留；preparation builder 只接受 loader/verifier/capture 私有签发对象，验证 clean Git 与 `config/model.json`/dependency lock metadata 而不接触 state dict，保留全部 C10 历史状态且不能生成执行或分析授权；CLI 仅含四个 D3 命令。
- package_complete: true
- scientific_validity: not_evaluated
- analysis_authorized: false
- budget_results: `D7_PROFILE_REQUIRED` / `NOT_EVALUATED`；D3 只保留并可对 synthetic/pre-D7 profile 投影 launch-safety caps 与 completed-campaign gates，任何该投影仍非 D7 final budget。
- paper_data_or_figures_generated: none
- deviations_or_protocol_amendments: none。没有改动 D2 frozen contract；实现审查中把 `jsonschema` 提升为运行时依赖、增加 D1 same-snapshot bytes validator、强化 ambient Git 隔离与 repo identity、静态 model/dependency metadata binding，这些均为 D3 fail-closed 实现细化，不改变科学 protocol。
- failures_and_retries_retained: 保留 `/usr/bin/time` 未安装导致的 pre-collection exit 127、两轮 focused 开发失败及 plan 后处理摘要脚本 `KeyError`；它们均是测试/显示工具问题，修复后最小节点通过，不是科学或正式 workload failure。没有 runtime attempt/retry/formal root。
- known_limitations_and_claim_boundaries: D3 不证明 live reserve JPEG 等于 D1 candidate-recorded hash；该验证延迟至 D8.0 且须先有 D7 authorization。D3 不提供 image loader、model backend、runner、sharding runtime、finalize、aggregate、paper export，不产生 Stage D validity/effect/cost 结论；shard count 仍延迟到 D7。C10 历史 blocker/spatial failure 不被修复或改写。
- next_task_authorized: true（只授权 D4 使用 synthetic/mock 实现 Stage D evidence producer、image-group runner、sharding/recovery/provenance；不授权 reserve access、smoke、正式 workload或 execution authorization）
- authorization_issued_this_task: false
- execution_authorization_observed_valid: false
- next_task: D4

#### 下一对话完整提示词

```text
本次唯一任务：只执行 Stage D 的 D4。

目标是在 D3 已验证的 CPU-only 控制层之上，实现 Stage D 专属 evidence producer、image-group runner、sharding runtime、immutable attempts/checkpoint/recovery 和 provenance。D4 只使用 synthetic/mock inputs，不运行 smoke 或正式 workload，不读取 reserve JPEG，不加载真实模型，不开始 D5，也不签发 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

开始条件：
1. 读取本文文首状态、第 3 节、第 5 节、第 7–9 节与最新 D3 completion record；核验 D3 completion commit（以 `git log -1 --format=%H` 取得本文 `git_sha_completion: SELF` 的实际 SHA）且工作树/index clean。
2. 唯一解析并复核 D2 `refs/archive/stage-d-d2-completion`=`c57438640e882604031775c944d8ff5ce7932713`，以及 D0/D1/D2/C10 frozen hashes。调用 D3 `validate-contracts` 和 typed-plan validator；不得重建既有科学结果。
3. D3 plan canonical digest 必须为 `5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443`，coordinate digest 必须为 `cc560e2a2d4e1220ccd812c073119a2b679de213f9770358f7decf51824661fb`，counts 必须保持 500/1,000/1,010/10,000/210,000/211,010。
4. 继续保持 `execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`、`stage_d_execution_started=false`；D3 preparation artifact 不是 execution authorization。

必读材料：
1. `docs/阶段D-step-by-step.md` 的 D4、§3、§5–§9 和最新 D3 record；`docs/阶段D_protocol.md`、`docs/stage_d_acceptance.md`、`docs/stage_d_config_contract.md`。
2. D3 modules：`stage_d_config.py`、`stage_d_manifest.py`、`stage_d_plan.py`、`stage_d_authorization.py`、`stage_d_cli.py`，及其 focused tests。
3. Stage C 已验证的 runner/artifact/provenance/resource primitives；只复用数学或生命周期 primitive，不得让 Stage D artifact 绑定 Stage C config/selection/stage identity。

必须完成：
1. 实现 Stage-D-specific evidence producer 和 image-group runner。一次独占完整 image group，严格 NNS→matched 执行；两 units 共享 frozen input/model/target/baseline/fractions/tie/operator bindings。
2. 实现 `init` 与 `run-shard` runtime API/CLI，但 D4 测试只能用 synthetic/mock backend；所有 authorization/metadata checks 必须发生在 backend/model 构造前。
3. 实现 atomic shard lease、image-group claim、immutable attempt IDs/directories、checkpoint、failure table、lifecycle/activity locks 和 closed output layout；一 GPU 一独立进程/backend，禁止共享 backend 并发。
4. 每 unit 显式覆盖 clean map 和 `2 directions × 5 repeats × 21 fractions` coordinates；audit 图每方法仅增加一个 repeat-clean map。禁止 patch scale、spatial/stability、random/center sanity、额外 method/head、secondary inference 和 display。
5. Resume 只跳过完整验证的 terminal unit；half-pair 和 paired scientific failure 按 D2 执行至 terminal。只允许 closed outcome-blind infrastructure failure codes申请一次 retry（最多两 attempts）；scientific invalidity、OOM、contract/authorization drift、tamper/unsafe root、ambiguity、safety-cap breach不可重试。旧 attempts 必须保留。
6. 每个 unit/attempt/artifact 写 Stage-D-specific provenance，绑定 D7 authorization slot、D0/D1/D2、plan/unit/work-item、code/dependency/model、image identity、method/target/operator/seed/fraction、terminal/failure/resource/inventory；D4 不得伪造已存在的 D7 authorization。
7. 按 L0/L1/L2/L3 合同实现 retention：L0/L1 全 1,000 units，L2 仅五个 audit IDs，L3 none；禁止 display PNG 和 per-step/per-pixel/per-dimension file explosion。Runtime CLI/log 只输出 progress/control/terminal/resource，不泄漏 map、AUC、ranking、`delta_fid` 或 summary。
8. 用 synthetic/mock fixtures 覆盖 group ownership/order、pair binding、shard modulo/不拆组、并发 claim、crash/checkpoint/tamper、resume、retry limits、scientific failure terminal、cache/resource release、artifact/provenance 和 no-result-leak；更新本文 D4 状态并追加完整 completion record。

明确禁止：
- 不读取、stat、hash、decode 或 preprocess 任一 reserve JPEG；不访问 Stage D outcome。
- 不加载真实模型/state dict，不运行 CPU inference、GPU/CUDA 或 attribution。
- 不运行 smoke 或正式 workload；不实现 D5 finalize/validation、D6 aggregate/statistics/paper export；不签发 execution/analysis authorization。
- 不修改 D0/D1/D2 frozen contracts、candidate access log 或 C9/C10 evidence。

测试计划：
测试前先一次性说明 focused synthetic/mock 测试、直接受影响共享模块、static checks 和预计耗时。优先 D4 新测试与最小共享节点；不得运行 GPU/model/image tests、smoke/formal workload或全量 non-GPU suite，除非先证明必要依赖链并说明扩展范围。

完成与交接：
D4 只有在 producer/runtime/recovery/provenance 的 synthetic 验收全部通过、未发生 forbidden access、未签发 execution authorization时才标记 PASS。更新第 5 节，追加完整 D4 completion record 与只授权 D5 synthetic/tampered finalize/validator 的下一提示词；只提交 D4 相关文件并确认最终 clean。
```

### 2026-08-26 03:10 +0800 — D4 completion record

- status: PASS
- objective_completed: 已实现 Stage-D-specific evidence producer、一次独占完整 image group 且严格 NNS→matched 的 runner、`init`/`run-shard` runtime、原子 shard lease/group claim、immutable attempts、checkpoint/failure table、lifecycle/activity locks、显式 coordinator recovery、closed L0/L1/L2/L3 retention 与 Stage-D-specific provenance；formal gate 输出在完整 metadata normalization 后被 runtime 私有 issuer 密封，producer 不再接受可伪造 marker。
- authority_and_scope: D0 的 `APPROVED_TO_PREPARE_STAGE_D` 仍是唯一准备授权；`execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`、`stage_d_execution_started=false`、scientific result `NOT_EVALUATED`。D4 没有 D7 issuer，不签发或模拟可用于正式执行的 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`；下一步只授权 D5 对 synthetic/tampered fixtures 实现 finalize/validator。
- git_sha_before: `766e2c7d4e880df96c88a414ea7e63cd15c5d9d9`（D3 completion commit；启动时 worktree/index clean）。
- git_sha_producer: SELF（D4 producer/runtime implementation commit；精确 SHA 在提交后以 Git 为准并在最终回复报告）
- git_sha_postprocessor: none
- git_sha_completion: SELF
- tree_clean_at_boundaries: 启动时 clean；实现期间只有本 record 所列 D4 文件；未 stash/reset/delete/覆盖用户改动。提交后的 clean 状态由 post-commit 检查与最终回复据实报告。
- upstream_commits_and_refs: D2 `refs/archive/stage-d-d2-completion` / `c57438640e882604031775c944d8ff5ce7932713`；D1 `refs/archive/stage-d-d1-completion` / `d594eadae5b5aa715d12346f6dc93507f1e093a9`；D0 `refs/archive/stage-d-d0-completion` / `4c76d468e1368858c52ff8ef73e640463e15ac49`；C10 producer `refs/archive/c10-aggregation-producer` / `92ad81342ff22e2710df495a9961a3ab8c015a3a`，C10 completion `69e1fa887862e00da4441842c6de149fe2424225`。
- files_added_or_changed: 新增 `src/vitnns/{fidelity_seed.py,stage_d_artifacts.py,stage_d_producer.py,stage_d_provenance.py,stage_d_runtime.py}`、`tests/d4_guard/sitecustomize.py`、`tests/test_stage_d_{guard,artifacts,producer,provenance,runtime}.py`；修改 `src/vitnns/{stage_c_fidelity.py,stage_c_runner.py,stage_c_value_path.py,stage_d_cli.py}`、`tests/{test_stage_c_runner.py,test_stage_d_cli.py,test_stage_d_registry_contract.py,test_stage_d_scope_contract.py}`；更新 `README.md`、本文、仓库顶层 `README.md` 和 `论文1实现审计与后续计划.md`。未修改 D0/D1/D2 frozen contracts、candidate access log 或 C9/C10 evidence。
- frozen_inputs_and_sha256: 沿用且未修改 D0 canonical `7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f`；D1 registry file/canonical `9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b` / `514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028`；D1 package manifest file/canonical `f38bf9773ad172100c22f50ec3b4fe5251664e5c9f5c7ee78ae46532aec372fe` / `cf3e21b88488dcb928f9a98557c2ee9896a7c5210ddb8dda64427c1792fa299d`；D1 source binding file/canonical `f1109304a23674bdadc62306bc927e7ac10b397f68754fae13967971b4f94cac` / `2eafef71035b95035a92c5ccfba7d536c3b924df4081def86fb0a1fa06dba2b8`；sealed C10 whole-root digest `382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`。
- config_plan_and_canonical_digests: `validate-contracts` 在当前 tree PASS，config canonical `21f9dcf1d615f2531f7d7d9a36319b958e06d0606c325890dd31ad89b7c2caae`、run-spec canonical `56a1d946c96ce5f86699e52a359557c1cdfd6ebbfd5aa43646d5e3a2d6428c85`、verified source-manifest canonical `146ddf1eb84d4dd8fb5c919e0d531d63532812ba5af1b15e1de4fb04eae1bad5`；typed plan canonical/coordinate digest `5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443` / `cc560e2a2d4e1220ccd812c073119a2b679de213f9770358f7decf51824661fb`。
- data_access_statement: 只读取 Git、文档、Python source 与 Stage D JSON/CSV metadata；没有读取、stat、hash、decode 或 preprocess 任一 reserve JPEG，没有访问 Stage D prediction/logit/map/AUC/`delta_fid`/summary/outcome，没有加载 state dict 或真实模型，没有使用 GPU/CUDA，没有运行 smoke 或正式 workload，没有修改 candidate access log、D0/D1/D2 frozen contracts 或 C9/C10 evidence。strict guard 对本次最终 D4 focused 验证覆盖的路径阻止图片/权重/reserve/outcome、subprocess、模型构造/调用、autograd 与 CUDA API。
- artifacts_and_absolute_paths: D4 产品代码位于 `/data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_d_{artifacts,producer,provenance,runtime}.py`，CLI 位于 `/data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_d_cli.py`；测试仅在 pytest 临时目录创建 synthetic runtime roots，测试结束后未保留或声明任何 formal/runtime evidence root；无 paper artifact。
- status_manifest_inventory_hashes: D4 源文件提交前 SHA-256：`fidelity_seed.py=d76c7124bc93a39bbf392a274912b48f68dbef8146c8d4aece5d67927e4e8a0c`、`stage_c_fidelity.py=258a05650b892861f41ed9bb73ccc28b14b25a09288765aad8d724df5cefd7a0`、`stage_c_runner.py=7ed10ce52c256f2f66deeb7baee1d9e4b6e509a1ac4cb4bd8f490e103a9ca83d`、`stage_c_value_path.py=6551748ab510f9ebae7204978f01198401ec66475e4967cab70ea12d522dcdfe`、`stage_d_artifacts.py=409f85287152ed72caee6d0bbca2448ed723f8c30077ca55a0476122a5767b50`、`stage_d_producer.py=63ddd20e65f27bc038dfb3448f2fd397539998c12a7fe8acf501cd3302b6846e`、`stage_d_provenance.py=4a7b25b7a3d76efce2c19d7fd10803b9af1a74539dc0b4ff55cb136c512b3a43`、`stage_d_runtime.py=4bdfe00d0aee6f103e58c689de6f2523dd9afac8ca7d6cf968c2ff04a8cc1475`、`stage_d_cli.py=4b6a79c41195ff1935409bac50318ca8d5e11d1716aeb7355947811e0c5399a9`。无正式 status/manifest/inventory。
- expected_vs_observed_counts: typed plan 复算 image groups `500/500`、execution units `1,000/1,000`、primary clean maps `1,000/1,000`、audit repeat-clean maps `10/10`、map work items `1,010/1,010`、fidelity curves `10,000/10,000`、curve points `210,000/210,000`、logical work items `211,010/211,010`、statistical n `500/500`、audit increment `0/0`；synthetic shard test 对一个完整 group 验证 NNS/matched 两 units、一次 infrastructure retry 与 terminal resume。
- tests_reused_from_prior_commit: D3 completion `125 passed`、D2 focused `23 passed`、D1 `10 passed`、D0 `24 passed` 仅作已提交基线证据；本任务没有伪称重跑。开发中较早、修复最终 blockers 前的 D4 focused 结果包括 artifact/provenance `40 passed`、CLI `44 passed`、runtime `7 passed`、strict guard+producer `15 passed`；这些仅作开发轨迹，最终验收以后述定向复验为准。
- test_plan_announced_before_execution: 测试前已声明只运行 strict guard、自身及三个已确认 blocker 的 producer/runtime 定向节点、一个真实 synthetic shard lifecycle、一个 coordinator recovery 节点、受影响 Stage C mock regression、D3 metadata-only contract/typed-plan validator、`py_compile` 与 `git diff --check`；预计 2–4 分钟并明确不扩展到 GPU/model/image、smoke、正式 workload 或全量 non-GPU suite。typed-plan fixture 本身较慢，实际多次定向运行超出估时。
- tests_executed_this_task: 最终 blocker 定向 strict-guard 批次 `5 passed in 178.03s`；包含额外 partial-count 节点的前一批 `7 passed in 216.18s`；真实 synthetic shard 的 half-pair/scientific-failure/infrastructure-retry/resume/no-leak 节点 `1 passed in 85.88s`；coordinator crash-recovery durability 节点 `1 passed in 79.45s`；Stage C structured-NA 与 repeatability 两个 mock runner 节点 `2 passed in 1.31s`；Value-path nominal convergence 类型兼容节点 `1 passed in 2.11s`；D3 `validate-contracts` PASS，typed-plan validator 重算 digest/count PASS；D4 Python files `py_compile` 与 `git diff --check` PASS；最终路线文档静态节点 `2 passed in 0.06s`。post-commit checks 见最终回复。
- tests_not_run_and_reason: 未运行全量 non-GPU suite、完整 Stage C suite、GPU/model/image tests、smoke 或正式 workload；前者超出最小受影响依赖链且已由更早基线覆盖，后者由 D4 明文禁止。未运行 D5 finalize/root-seal、D6 aggregate/statistics/paper export 或任何 D7/D8 行为。
- acceptance_results: group ownership/order、pair binding、完整 211-coordinate/unit coverage、shard modulo 不拆组、并发 claim、checkpoint/tamper、crash/recovery、terminal resume、one-retry/two-attempt limit、scientific failure terminal 且 paired unit 继续、cache/resource release、closed artifact/provenance、L0/L1 全覆盖/L2 audit-only/L3 none、runtime source-derived truth、CLI no-result-leak、pre-backend authorization/metadata checks均由 synthetic/mock 验证；三个最终审查项（nominal scientific translation/production partial、manifest-derived count、issuer-backed formal capability）经独立只读复审无 blocker。
- package_complete: true
- scientific_validity: not_evaluated
- analysis_authorized: false
- budget_results: `D7_PROFILE_REQUIRED` / `NOT_EVALUATED`；D4 未做 smoke/profile 或更改 caps。
- paper_data_or_figures_generated: none
- deviations_or_protocol_amendments: none。科学合同、坐标和阈值未修改。开发期一次普通 Stage C toy-model test `test_ig_wrapper_nonconvergence_retains_all_five_attempts` 被误执行；该测试只构造内存 toy model/张量，不读取真实模型/state dict、JPEG、reserve 或 outcome，不使用 GPU，也不产生持久 artifact，但超出 D4 的 no-CPU-inference/no-attribution 测试边界；随后以 strict D4 guard 重跑最终关键验收，guard 会拒绝模型构造/调用、autograd、CUDA 和敏感路径。
- failures_and_retries_retained: 普通沙箱曾因环境缺少 `bubblewrap` 在命令执行前失败，之后使用可用执行环境完成只读/测试；开发期 typed-plan 摘要脚本因误读不存在的 `StageDUnit.counts` 属性失败一次，改为读取 `plan.counts` 后通过；synthetic shard 首次定向测试因旧 fixture 使用已被闭集正确拒绝的 `SYNTHETIC_INVALID` code 而失败，改成冻结 `degenerate_direction` 后通过；一次最终只读 `rg` 模式因双引号内反引号被 shell 展开而附带 `D5: command not found`，随后用安全 quoting 完成检查。上述均非正式 scientific/runtime attempt；没有 formal root 或正式 retry。
- known_limitations_and_claim_boundaries: D4 只证明 synthetic/mock 运行控制、artifact 与 provenance 合同；不证明 live reserve JPEG、真实模型、CUDA kernel、性能预算或 scientific validity。D5 尚未实现 final root sealing/reconciliation；D6 尚未实现统计/论文导出；D7 尚未 smoke/profile/freeze producer 或签发执行授权。C10 历史 blocker/spatial failure不变。
- next_task_authorized: true（只授权 D5 使用 synthetic/tampered fixtures 实现 finalize 与独立 source-recomputing validation；不授权 D6、smoke、正式 workload、reserve JPEG、真实模型/GPU 或 execution/analysis authorization）
- authorization_issued_this_task: false
- execution_authorization_observed_valid: false
- next_task: D5

#### 下一对话完整提示词

```text
本次唯一任务：只执行 Stage D 的 D5。

目标是在 D4 已验证的 synthetic/mock producer、runtime、immutable attempts 和 provenance 之上，实现 execution-root finalize 与独立 source-recomputing validator。D5 只能使用 synthetic valid/tampered/missing/failed/active-claim fixtures，不得对任何正式 root 调用 finalize，不读取 reserve JPEG，不加载模型，不运行 smoke 或正式 workload，不开始 D6，也不签发 execution/analysis authorization。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

开始条件：
1. 读取本文文首状态、第 3 节、第 5 节、第 7–9 节、最新 D4 completion record 和本提示词；核验 D4 completion commit（以 `git log -1 --format=%H` 取得 `git_sha_completion: SELF` 的实际 SHA）且 worktree/index clean。
2. 唯一解析并复核 D0/D1/D2 refs、D3/D4 commits 和 C10 frozen bindings；调用 D3 `validate-contracts` 与 typed-plan validator，不重建既有科学结果。
3. 计划 canonical digest 必须为 `5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443`，coordinate digest 必须为 `cc560e2a2d4e1220ccd812c073119a2b679de213f9770358f7decf51824661fb`，counts 必须保持 500/1,000/1,010/10,000/210,000/211,010。
4. 保持 `execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`、`stage_d_execution_started=false`；D3 preparation artifact 和 D4 synthetic roots 都不是正式授权或结果。

必读材料：
1. `docs/阶段D-step-by-step.md` 的 D5、§3、§5–§9 与最新 D4 record；`docs/阶段D_protocol.md`、`docs/stage_d_acceptance.md`、`docs/stage_d_config_contract.md`。
2. D3/D4 modules：`stage_d_config.py`、`stage_d_manifest.py`、`stage_d_plan.py`、`stage_d_authorization.py`、`stage_d_artifacts.py`、`stage_d_provenance.py`、`stage_d_runtime.py`、`stage_d_cli.py` 及 focused tests。
3. Stage C 已验证的 artifact/provenance validation primitives，只复用 stage-neutral integrity primitive，不得让 Stage D package 绑定 Stage C config/selection/stage identity。

必须完成：
1. 实现 finalize 前 fail-closed 检查：拒绝 active claims、未结束 shard、missing/extra file、symlink/hardlink、不安全 root、未知状态和任何 authorization/plan/provenance drift。
2. 对 500 registry rows、500 groups、1,000 unit keys、全部 immutable attempts/status/manifest/artifact/provenance 做 exact left join；验证 attempt 顺序、terminal 唯一性、retry history、half-pair 与 failure table/checkpoint 一致。
3. 独立从 source bytes 重算文件 SHA-256、canonical digests、status→manifest、artifact inventory 和 whole-root ordered inventory；不得信任 runner summary、自报计数或 artifact 内自报 hash。
4. 验证每图 NNS/matched 共享 frozen input/model/target/baseline/fractions/tie/operator bindings；验证 endpoint/token parity、IG conservation、finite values、direction validity、audit repeatability、资源记录及 L0/L1 全覆盖、L2 仅五个 audit IDs、L3 none。
5. 明确分离 `package_complete` 与 `scientific_validity`：完整记录的 scientific failure 可完成封存但 scientific validity 为 FAIL；tamper/missing/unauthorized/incomplete/active claim 必须 `BLOCKED`。
6. 生成 final evidence report、failure/NA reconciliation、payload manifest 和 whole-root ordered inventory；所有输出 schema closed、原子写入、不可覆盖。
7. 为 CLI 增加 `finalize` 和 `validate`，但 D5 测试只针对 synthetic/tampered roots；不得加入 aggregate、statistics、decision、paper export 或正式 root 路径。
8. 更新本文 D5 状态并追加完整 completion record；PASS 时只授权 D6 synthetic aggregation/statistics/paper-export implementation。

明确禁止：
- 不读取、stat、hash、decode 或 preprocess 任一 reserve JPEG；不访问真实 Stage D outcome。
- 不加载真实模型/state dict，不运行 CPU inference、GPU/CUDA 或 attribution。
- 不对正式 root 调用 finalize/validate，不运行 smoke 或正式 workload；不开始 D6/D7/D8。
- 不聚合 `delta_fid`、不运行 bootstrap、不产生正式 decision/paper data/figure。
- 不签发 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE` 或 analysis authorization，不修改 D0/D1/D2 frozen contracts、candidate access log 或 C9/C10 evidence。

测试计划：
测试前一次性说明 D5 focused synthetic/tampered tests、直接受影响共享节点、static checks 和预计耗时。覆盖 valid/tampered/missing/failed/active-claim root、exact reconciliation、source recomputation、closed schemas、atomic/immutable output 和 CLI no-result-leak；优先最小充分集合，不运行 GPU/model/image tests、smoke/formal workload或全量 non-GPU suite，除非先证明必要依赖链并说明扩展范围。

完成与交接：
D5 只有在 finalize/validator 的 synthetic/tampered 验收全部通过、未发生 forbidden access、未签发 execution/analysis authorization时才标记 PASS。更新第 5 节，追加完整 D5 completion record 与只授权 D6 synthetic aggregation/statistics/paper export 的下一提示词；只提交 D5 相关文件并确认最终 clean。
```

### 2026-08-26 14:22 +0800 — D5 completion record

- status: PASS
- objective_completed: 已实现 Stage D execution-root finalize、独立 source-recomputing validator、model-free scientific evidence validator、final evidence report、failure/NA reconciliation、source/payload manifest、whole-root ordered inventory，以及 CLI `finalize`/`validate`。Finalizer 在 D4 三把 cooperative locks 下保留 root directory descriptor，从 immutable source bytes 重建全部结果，并以 descriptor-bound staging、terminal status last 和 no-replace rename 原子发布；validator 不信任 runner summary、artifact 自报 hash 或 final controls，而是重新遍历并逐字节比较 source-derived documents。
- authority_and_scope: D0 的 `APPROVED_TO_PREPARE_STAGE_D` 仍是唯一准备授权；`execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`、`stage_d_execution_started=false`。D5 未签发、模拟或消费正式 execution/analysis authorization，且无条件拒绝带 formal execution context 的 root；下一步只授权 D6 使用 synthetic sealed fixtures 实现 aggregation/statistics/decision/paper-export。
- git_sha_before: `a330055ebaf3b2ed656eab783ea87d23ad6fedbe`（D4 completion commit；D4 record 的 `git_sha_completion: SELF` 实际 SHA；启动时 index/worktree clean）
- git_sha_producer: none（D5 不生产正式 runtime evidence）
- git_sha_postprocessor: SELF（D5 finalize/source-recomputing validator implementation commit；精确 SHA 在提交后以 Git 为准并在最终回复报告）
- git_sha_completion: SELF
- tree_clean_at_boundaries: 启动时 clean；实现期间只存在本 record 列出的 D5 source、tests、README 与路线状态改动，未覆盖、stash、reset 或删除其他用户改动。completion commit 后由 post-commit 检查确认 index/worktree clean。
- upstream_commits_and_refs: D4 `a330055ebaf3b2ed656eab783ea87d23ad6fedbe`；D3 `766e2c7d4e880df96c88a414ea7e63cd15c5d9d9`；D2 `refs/archive/stage-d-d2-completion` / `c57438640e882604031775c944d8ff5ce7932713`；D1 `refs/archive/stage-d-d1-completion` / `d594eadae5b5aa715d12346f6dc93507f1e093a9`；D0 `refs/archive/stage-d-d0-completion` / `4c76d468e1368858c52ff8ef73e640463e15ac49`；C10 producer `refs/archive/c10-aggregation-producer` / `92ad81342ff22e2710df495a9961a3ab8c015a3a`，C10 completion `69e1fa887862e00da4441842c6de149fe2424225`。以上 refs/commits 已唯一解析并复核。
- files_added_or_changed: 新增 `src/vitnns/stage_d_finalize.py`、`src/vitnns/stage_d_scientific.py`、`tests/d5_guard/sitecustomize.py`、`tests/test_stage_d_d5_guard.py`、`tests/test_stage_d_finalize.py`、`tests/test_stage_d_finalize_schema.py`、`tests/test_stage_d_scientific.py`；修改 `src/vitnns/stage_d_cli.py`、`tests/test_stage_d_cli.py`、`tests/test_stage_d_registry_contract.py`、`tests/test_stage_d_scope_contract.py`、`README.md` 和本文。未修改 D0/D1/D2 frozen contracts、candidate access log 或 C9/C10 evidence。
- frozen_inputs_and_sha256: 沿用且未修改 D0 canonical `7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f`；D1 registry file/canonical `9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b` / `514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028`；D1 package manifest file/canonical `f38bf9773ad172100c22f50ec3b4fe5251664e5c9f5c7ee78ae46532aec372fe` / `cf3e21b88488dcb928f9a98557c2ee9896a7c5210ddb8dda64427c1792fa299d`；D1 source binding file/canonical `f1109304a23674bdadc62306bc927e7ac10b397f68754fae13967971b4f94cac` / `2eafef71035b95035a92c5ccfba7d536c3b924df4081def86fb0a1fa06dba2b8`；sealed C10 whole-root inventory `382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`。
- config_plan_and_canonical_digests: D3 `validate-contracts` 和 typed-plan validator 在 D5 tree 上 PASS；config `21f9dcf1d615f2531f7d7d9a36319b958e06d0606c325890dd31ad89b7c2caae`、run spec `56a1d946c96ce5f86699e52a359557c1cdfd6ebbfd5aa43646d5e3a2d6428c85`、verified source manifest `146ddf1eb84d4dd8fb5c919e0d531d63532812ba5af1b15e1de4fb04eae1bad5`、plan `5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443`、coordinate `cc560e2a2d4e1220ccd812c073119a2b679de213f9770358f7decf51824661fb`。
- data_access_statement: 只读取 Git、文档、Python source、冻结 Stage D JSON/CSV metadata 和 `/dev/shm` 下 D5 synthetic/tampered roots；没有读取、`stat`、hash、decode 或 preprocess 任一 reserve JPEG，没有访问真实 Stage D prediction/logit/map/AUC/`delta_fid`/summary/outcome，没有加载真实模型或 state dict，没有运行 CPU inference、attribution、GPU/CUDA、smoke 或正式 workload，也没有对任何正式 root 调用 finalize/validate。strict D5 guard 覆盖最终 focused tests并阻止 reserve/formal/outcome/image/weight、模型构造/调用、autograd、CUDA 与 subprocess access。
- artifacts_and_absolute_paths: 产品代码为 `/data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_d_finalize.py`、`/data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_d_scientific.py` 和 `/data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_d_cli.py`。验收 roots 仅为 ephemeral synthetic fixtures：`/dev/shm/vitnns-d5-full-valid/d5-valid0/execution-root`、`/dev/shm/vitnns-d5-scientific-failure/d5-failed0/execution-root`、`/dev/shm/vitnns-d5-authorized-retry/d5-retry0/execution-root`；它们不是正式 evidence、authorization 或 scientific result。无 formal root、无 paper artifact。
- status_manifest_inventory_hashes: D5 source SHA-256 为 `stage_d_finalize.py=1ff1e095708960a53027edd8726a888c9e492b67924faadb6cc1c769ddfedd71`、`stage_d_scientific.py=f2d2f01162dc0c6908fe6ac6338400dc164a6955bdaf33d7344c99d2ed586535`、`stage_d_cli.py=ecd93e6969c8a2a09006e26669ed78ca9de8e5b956dc2241f904c2b1d198d579`。Synthetic valid root 的 status file/canonical/sealed-root inventory digest 为 `9a220801361a780608c8b00f4258917d55279f60879228c22156b6a661df2c19` / `798bb77b4ca66143d2c869cfbdd30f3c914c5a344ffcb8bd12250eeecb2c2aa0` / `73697c60efc7b8b835fad7c324a4ce2c9b3c5e7bde067444697581bc4d60995f`；scientific-failure root 为 `a569d6aecac41ce4591202e2e4614aaed210c2115b024954b0869ad121eef5cd` / `137cab7b08d95ae114d915b297f1c055a2ef2c1ebfa46d59d0fd3b0851272601` / `b99ca5909422d9674bad818651496cb449d5e07fbe6cbf5dcae06e48074cbf5f`；authorized-retry root 为 `ab93dddaaa586c33b6f8e6d4b457a4b99a0ebb98a15276fae68efa9ca34b4d15` / `17c6ca42b43d6c420fecd5188b7a66467d8ac5540ff8d0ec93610f7165f9f0d4` / `0351a1ad4f7e7eed05354b1912b4b0818c2a7bf00d604241ebed5ddce7107690`。这些均为 synthetic evidence；无正式 status/manifest/inventory。
- expected_vs_observed_counts: typed plan 固定 registry/images/groups `500/500/500`、units `1,000`、map work items `1,010`、fidelity curves `10,000`、curve points `210,000`、logical work items `211,010`、audit images `5`、L2 units/repeat maps `10`、L3 `0`。Valid synthetic root 精确观察 500 groups、1,000 units、1,000 attempts、1,010 maps、10,000 curves、210,000 points、211,010 logical items、6,014 source files；authorized retry 精确观察 500/1,000/1,001 attempts、其余 workload counts 全匹配、6,016 source files。Scientific-failure root保留 500 groups/1,000 terminal units/1,000 attempts和一条 NA，观测 maps/curves/points/logical items 为 `1,009/9,990/209,790/210,799`，相对注册值的 `1/10/210/211` 缺口由同一 terminal scientific-failure row 精确解释，未丢行或补零。
- tests_reused_from_prior_commit: D4 completion 的最终 synthetic/runtime evidence、D3 `125 passed`、D2 `23 passed`、D1 `10 passed`、D0 `24 passed` 仅作已提交基线；本任务没有伪称重跑。D5 full roots在其 source bytes和当前 code未漂移时复用，避免为每个负例重复生产/复制 1,000 attempts。
- test_plan_announced_before_execution: 测试前已按批次声明 D5 focused synthetic/tampered tests、直接受影响的 CLI/route共享节点、strict guard、static checks及预计耗时；明确只保留一次 500/1,000 正例和具有独立语义的 scientific-failure、authorized-retry 全量分支，其余 missing/tampered/schema/active-claim/atomic/race 走复用或小夹具，避免约 30 个负例重复复制和重验整棵树。
- tests_executed_this_task: 完整 valid 500-group/1,000-unit producer→finalize→independent validate `1 passed in 508.24s`；完整 authorized-retry 1,001-attempt 分支 `1 passed in 609.52s`，当前候选代码随后再次独立 source-recompute validate 同一 sealed root成功；完整 scientific-failure 分支完成 producer/finalize/validate并得到 `package_complete=true, scientific_validity=FAIL`，修复末尾 test-only tuple/list 断言后复用 sealed root重验通过；active-claim 小 root `1 passed in 64.33s`；复用同一 full valid source 的 missing 与 corrupt-NPZ 两个负例均约 52 秒且均 integrity-BLOCKED、未发布 `final/`；retained-root ABA、staging substitution、staging-create cleanup `3 passed in 0.22s`。最终 strict lightweight 批次覆盖 guard、scientific source recomputation、递归 closed schema、CLI/no-result-leak、上述三个 FD/publication 节点及两个路线静态节点，结果 `299 passed, 1 skipped in 2.54s`；skip 是按设计不要求固定出现数据依赖的 terminal-state keys。D3 `validate-contracts`、typed-plan validator、`py_compile` 与 `git diff --check` PASS。
- tests_not_run_and_reason: 未运行全量 non-GPU suite、完整 Stage C suite、GPU/model/image tests、smoke/formal workload或任何正式 root finalize/validate；前两类超出 D5 最小受影响依赖链且相关 frozen baselines可复用，后几类由 D5 明文禁止。未运行 D6 aggregate/statistics/bootstrap/decision/paper export或 D7/D8 行为。未为每个负例重新构造 full root，因为它不会增加边界覆盖，只会重复数千文件的 I/O/hash。
- acceptance_results: valid/tampered/missing/scientific-failed/active-claim fixtures、未知/未终结状态、unsafe root、symlink/hardlink/extra empty directory、authorization/plan/provenance drift均按合同 PASS 或 fail closed；500 registry rows、500 groups、1,000 unit keys、attempt order/terminal/retry/half-pair/failure table/checkpoint/resources/retention完成 exact joins；source file SHA-256、canonical JSON、status→manifest、artifact/provenance inventory与 whole-root inventory均从 bytes重算。成功 evidence 的 shared input/model/target/baseline/fractions/tie/operator bindings、ranking/input-pair/endpoint绑定、endpoint/token parity、IG conservation、finite/direction/audit repeatability通过；L0/L1全覆盖、L2仅五个 audit IDs的十个 units、L3 none。完整记录的 scientific failure可封存为 package complete且 validity FAIL；tamper/missing/unauthorized/incomplete/active claim不产生成功封存。五个 final controls递归 closed、canonical、原子、不可覆盖；CLI只暴露闭合脱敏控制摘要。
- package_complete: true（仅表示 D5 实现与 synthetic acceptance package 完成；不表示正式 Stage D root 已封存）
- scientific_validity: not_evaluated（正式 Stage D scientific evidence 尚不存在；synthetic PASS/FAIL分支只验证状态语义）
- analysis_authorized: false
- budget_results: `D7_PROFILE_REQUIRED` / `NOT_EVALUATED`；D5 未运行 profile/smoke或修改 caps。
- paper_data_or_figures_generated: none
- deviations_or_protocol_amendments: none。未修改冻结科学问题、坐标、阈值、统计或 workload。测试夹具只将反复全 checkpoint 写入改为 test-only 线性构造；attempt/artifact/provenance仍由真实 D4 writer产生，末尾由未 patch 的 D4 validator做全量复核。
- failures_and_retries_retained: scientific-failure 全量测试的首次执行在 finalizer/validator 均成功后因测试断言把 immutable tuple直接与 list比较而失败，已只修正断言并复用已封存 root重验，未重跑 producer；当前候选 full validator成功返回后，外层摘要脚本误取不存在的 `reconciliation["attempts"]` 而报 `KeyError`，不属于 validator失败。D3 `validate-contracts` 首次置于 D5 strict guard下时，其合法只读 `git rev-parse` subprocess被 guard拒绝，随后按 D3 metadata-only边界在不安装 D5 subprocess guard的环境重跑 PASS。以上均非正式 scientific/runtime attempts；唯一 runtime retry 是 synthetic allowlisted infrastructure-retry fixture并完整保留两次 attempts。
- known_limitations_and_claim_boundaries: D5 只证明 synthetic/tampered root sealing和 retained-source validation，不证明 live reserve JPEG、真实模型/CUDA、正式 campaign、性能预算或正式 scientific validity。未保留的 intervention interior tensor bytes只能作为 opaque commitments；validator重算其可由 retained bytes导出的 ranking/input-pair/endpoints并验证跨记录绑定，不虚称重算 interior。并发安全以 D4 cooperative-lock writer合同为边界：retained root/staging FDs及发布前后 inode/source checks会拒绝可观测 ABA/substitution；恶意同 UID进程绕过三把锁并在 `renameat2` 窗口主动替换 namespace不在该合同内，Linux目录 rename没有按已打开 source-directory FD的条件提交原语。D6/D7/D8尚未开始。
- next_task_authorized: true（只授权 D6 使用 synthetic sealed fixtures实现 aggregation、statistics、decision 与 paper-export；不授权 D7、smoke、正式 workload、reserve JPEG、真实模型/GPU、正式 result access或 execution/analysis authorization）
- authorization_issued_this_task: false
- execution_authorization_observed_valid: false
- next_task: D6

#### 下一对话完整提示词

```text
本次唯一任务：只执行 Stage D 的 D6。

目标是在 D5 已验证的 sealed-root source-recomputing boundary 之上，使用 synthetic sealed fixtures 实现 aggregation、冻结统计、decision truth table 与 paper-data/数值图 recipe/export/validation。不得开始 D7，不得运行 smoke 或正式 workload，不得访问任何正式 Stage D result，不得签发 execution/analysis authorization。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

开始条件：
1. 读取本文文首状态、第 3–9 节、最新 D5 completion record 和本提示词；以 `git log -1 --format=%H` 取得 D5 `git_sha_completion: SELF` 的实际 SHA，并核验 index/worktree clean。
2. 唯一解析并复核 D0/D1/D2 refs、D3/D4 commits、D5 completion commit和 C10 frozen bindings；调用 D3 `validate-contracts` 与 typed-plan validator，不重建既有科学结果。
3. 核验 D5 source SHA-256：`stage_d_finalize.py=1ff1e095708960a53027edd8726a888c9e492b67924faadb6cc1c769ddfedd71`、`stage_d_scientific.py=f2d2f01162dc0c6908fe6ac6338400dc164a6955bdaf33d7344c99d2ed586535`、`stage_d_cli.py=ecd93e6969c8a2a09006e26669ed78ca9de8e5b956dc2241f904c2b1d198d579`。任一 binding不符即 fail closed。
4. 计划 canonical digest必须为 `5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443`，coordinate digest必须为 `cc560e2a2d4e1220ccd812c073119a2b679de213f9770358f7decf51824661fb`，counts保持 500/1,000/1,010/10,000/210,000/211,010。
5. 保持 `execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`、`stage_d_execution_started=false`。D5 synthetic roots及其 PASS/FAIL状态只是实现验收夹具，不是正式结果或授权，也不得把 `/dev/shm` 临时路径当作持久依赖。

必读材料：
1. `docs/阶段D-step-by-step.md` 的 D6、§3、§5–§9 与最新 D5 record；`docs/阶段D_protocol.md`、`docs/stage_d_acceptance.md`、`docs/stage_d_config_contract.md`。
2. 冻结 `config/stage_d.json`、`config/stage_d_run_spec.json` 与 schemas；D3 typed-plan/statistics字段；D5 `stage_d_finalize.py`、`stage_d_scientific.py`、`stage_d_cli.py`及 focused tests。
3. 已验证的 Stage C stage-neutral统计/serialization/plotting primitives可复用，但不得让 Stage D package绑定 Stage C config、selection、stage identity或把 C10 `n=15` 与 Stage D `n=500`池化。

必须完成：
1. 建立 D6 closed input boundary：只接受 D5 独立验证通过、`package_complete=true`的 sealed synthetic root；aggregation读取 validated L0，描述性曲线读取 validated L1，完整坐标/status/failure/NA来自 D5 reconciliation。不得信任外部 summary、自报 aggregate或绕过 D5 validator。
2. 对全部 500 ordered image coordinates严格配对 NNS/matched；每图每方法/direction先对完整五个 tie repeats取算术平均，再按冻结公式生成唯一 `delta_fid`，严格 `delta_fid>0`才为 positive。禁止 complete-case、drop、impute、replace或把 invalidity置零。
3. 若任一 required coordinate无效，按冻结优先级输出 `NO_GO_VALIDITY`，不运行 bootstrap、不生成 effect summaries/main figures；对应输出显式 `NOT_GENERATED_DUE_TO_VALIDITY`。package completeness与 scientific validity继续分离。
4. 在 validity PASS时实现全 500 image paired-median bootstrap：恰好 10,000 draws、seed namespace `stage-d-bootstrap|20260821|`、SHA-256前8字节 big-endian unsigned、NumPy PCG64、draw shape `[10000,500]`、linear 2.5%/97.5% percentiles和 frozen even-median规则；tie repeats不得作为独立样本。
5. 实现冻结 truth table和相等边界：positive `<400`或 CI upper `<0.02`先判 `NO_GO_FIDELITY`；否则 median `>=0.02`、positive `>=400`且 CI lower `>0`判 fidelity PASS；其余 `INCONCLUSIVE_FIDELITY`；仅在 fidelity PASS后评价 completed-campaign cost并可能 `NO_GO_COST`，全通过才 `GO_FIDELITY_SCOPE`。workflow `BLOCKED`不是 scientific decision。
6. 固定 `holm_status=NOT_APPLICABLE_NO_SECONDARY_INFERENCE`，不得输出 secondary p-value/winner selection；保留 C10 `NO_GO_NOVELTY`、`BLOCKED`、`FAIL_SPATIAL`、Holm blocker且 C10/Stage D分 cohort展示，绝不池化为 `n=515`。
7. 实现 closed、canonical、deterministic、atomic且不可覆盖的 synthetic analysis package、per-sample table、主 fidelity summary、resource/QC、decision、provenance、payload manifest和 whole-root inventory；冻结 `paper_data`列、表格以及主图 A/B/C 的 machine-readable recipe。图 C 使用独立 `stage-d-figure-bootstrap|20260821|` namespace、每图五 repeats先均值、跨图 pointwise median、10,000 draws、PCG64和 linear percentile；明确不是 simultaneous band且不进入 primary decision。
8. 为 CLI 增加 `aggregate`、`build-paper-package`、`validate-paper-package`，仅输出闭合脱敏控制摘要；D6 测试只针对 synthetic sealed fixtures。不得加入 smoke、budget、execution issuer或正式 root路径。
9. 用 synthetic golden fixtures覆盖 399/400、count=400、median=0.02、CI lower=0、CI upper=0.02、per-image delta=0、validity failure、misclassification保留、deterministic bytes、source tamper、closed schemas、atomic/no-overwrite及图表输入一致性。优先小型数学/schema fixtures；只保留证明 500-coordinate/full-sealed integration不可替代的最少 full fixture，避免重复复制/重验整棵 D5 tree。
10. 更新本文 D6状态并追加完整 completion record；PASS时只授权 D7在已知 Stage C calibration/audit或 synthetic输入上执行 smoke、预算、producer freeze及 execution-authorization gate评估。

明确禁止：
- 不读取、`stat`、hash、decode或 preprocess任一 reserve JPEG；不访问真实 Stage D outcome或任何正式 root。
- 不加载真实模型/state dict，不运行 CPU inference、GPU/CUDA或 attribution；不运行 smoke或正式 workload。
- 不对正式数据聚合 `delta_fid`，不运行正式 bootstrap，不产生正式 decision、paper data或figure；测试产物只能是临时 synthetic golden artifacts。
- 不让 L2/audit/display PNG进入数值统计或 decision，不生成定性 overlay；不把 C10与 Stage D样本池化。
- 不修改 D0/D1/D2 frozen contracts、candidate access log或 C9/C10 evidence；不开始 D7/D8/D9/D10，不签发 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`或 analysis authorization。

测试计划：
测试前一次性说明 D6 focused synthetic golden tests、直接受影响共享节点、static checks及预计耗时。优先数学/closed-schema/CLI小夹具和最少的 sealed-root integration；不得运行 GPU/model/image tests、smoke/formal workload或全量 non-GPU suite，除非先证明必要依赖链并说明扩展范围。

完成与交接：
D6只有在 aggregation/statistics/truth-table/paper-package synthetic验收全部通过、D5 trust boundary未被绕过、未发生 forbidden access且未签发 execution/analysis authorization时才标记 PASS。更新第5节，追加完整 D6 completion record和只授权 D7已知 Stage C/synthetic smoke、预算、producer freeze及授权评估的下一提示词；只提交 D6相关文件并确认最终 clean。
```

### 2026-08-26 17:29 +0800 — D6 completion record

- status: PASS
- objective_completed: 在 D5 sealed-root/source-recomputing boundary 之上实现了 D6 closed input receipt、500-image paired aggregation、冻结 bootstrap/truth table、completed-campaign cost gate、closed synthetic analysis/paper package及其独立 validator。aggregation 只能消费同一 D5 snapshot 中签发的 `StageDValidatedAnalysisInput`；export 只能消费进程内 D6-issued immutable result，拒绝外部 self-reported summary。D6 只产生临时 synthetic golden package，未产生正式 Stage D decision、paper data 或 figures。
- authority_and_scope: `execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`、`stage_d_execution_started=false` 保持不变。没有签发、模拟或消费 execution/analysis authorization；D6 CLI 只返回脱敏 control summary，且 D5 synthetic PASS/FAIL或 D6 synthetic decision均不代表正式结果。
- git_sha_before: `a7aca36bfa1167623f7d5c81af56c59bac8c574f`（D5 completion commit；启动时 index/worktree clean）。
- git_sha_producer: none（D6 不生产正式 runtime evidence）。
- git_sha_postprocessor: SELF（D6 receipt/aggregation/package/CLI implementation commit；精确 SHA 在提交后以 Git 为准）。
- git_sha_completion: SELF
- upstream_commits_and_refs: 已唯一解析并复核 D0 `refs/archive/stage-d-d0-completion` / `4c76d468e1368858c52ff8ef73e640463e15ac49`，D1 `refs/archive/stage-d-d1-completion` / `d594eadae5b5aa715d12346f6dc93507f1e093a9`，D2 `refs/archive/stage-d-d2-completion` / `c57438640e882604031775c944d8ff5ce7932713`，D3 `766e2c7d4e880df96c88a414ea7e63cd15c5d9d9`，D4 `a330055ebaf3b2ed656eab783ea87d23ad6fedbe`，D5 `a7aca36bfa1167623f7d5c81af56c59bac8c574f`，以及 C10 producer `refs/archive/c10-aggregation-producer` / `92ad81342ff22e2710df495a9961a3ab8c015a3a` 和 C10 completion `69e1fa887862e00da4441842c6de149fe2424225`。
- frozen_inputs_and_plan: D3 `validate-contracts` 与 typed plan 已在开始时 PASS；config/run-spec canonical digests 保持 `21f9dcf1d615f2531f7d7d9a36319b958e06d0606c325890dd31ad89b7c2caae` / `56a1d946c96ce5f86699e52a359557c1cdfd6ebbfd5aa43646d5e3a2d6428c85`；plan/coordinate digests 保持 `5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443` / `cc560e2a2d4e1220ccd812c073119a2b679de213f9770358f7decf51824661fb`；counts保持 `500/1,000/1,010/10,000/210,000/211,010`。D5 initial source bindings在启动时匹配；D6 后 finalizer/scientific SHA仍为 `1ff1e095708960a53027edd8726a888c9e492b67924faadb6cc1c769ddfedd71` / `f2d2f01162dc0c6908fe6ac6338400dc164a6955bdaf33d7344c99d2ed586535`，CLI因新增 D6命令而变更。
- files_added_or_changed: 新增 `config/stage_d_analysis_package.schema.json`、`src/vitnns/stage_d_analysis_input.py`、`src/vitnns/stage_d_analysis.py`、`src/vitnns/stage_d_paper.py`、`tests/d6_guard/sitecustomize.py`、`tests/test_stage_d_analysis.py`、`tests/test_stage_d_d6_guard.py`；修改 `src/vitnns/stage_d_cli.py`、`tests/test_stage_d_cli.py` 和本文。未修改 D0/D1/D2 frozen contracts、candidate access log、C9/C10 evidence、D5 finalizer或 scientific verifier。
- D6_contract_results: receipt 在 D5 retained-root locks/same captured source bytes中重建 final controls，拒绝 formal/non-synthetic boundary；L0只用于 AUC/`delta_fid`，L1只用于曲线，L2/display PNG 不进入数值链。每图按 NNS/matched严格 pair，五次 tie repeats先取 mean并保留 ddof=0 SD；image级 `delta_fid` 才进入 all-500 draw `[10000,500]` PCG64 paired-median bootstrap。primary seed使用 `stage-d-bootstrap|20260821|fine_4x4_cell_delta_fid_median`，golden `18136624947342364318`、linear 2.5/97.5 percentile及 little-endian draw/distribution digest。invalidity显式 `NO_GO_VALIDITY`、无 bootstrap/curves/main effect artifacts，绝无 complete-case/drop/impute/zero-fill。truth table按 FAIL→PASS→INCONCLUSIVE顺序，包含 399/400、median=0.02、CI lower=0、CI upper=0.02与 delta=0边界；cost仅在 fidelity PASS后评估。
- package_and_reporting: analysis/paper package为 canonical JSON、deterministic CSV、atomic `renameat2(RENAME_NOREPLACE)`、fresh-root/no-overwrite、payload manifest、whole-root inventory/status和独立 validator。C10 `n=15`与 Stage D `n=500`分 cohort，固定保留 C10 `NO_GO_NOVELTY`、`BLOCKED`、`FAIL_SPATIAL`、`HOLM_BLOCKED_UNDERSPECIFIED`，永不 pool为515；Stage D `holm_status=NOT_APPLICABLE_NO_SECONDARY_INFERENCE`且无 secondary p-value/winner。A/B/C为 machine-readable recipes、不 render PNG；图 C保留84个独立 figure-bootstrap metadata，独立 `stage-d-figure-bootstrap|20260821|` namespace、10,000 PCG64 pointwise median draws、linear band、`simultaneous=false`、`enters_primary_decision=false`。
- tests_executed_this_task: 测试前已一次性说明 D6 focused synthetic golden nodes、D5 receipt/typed-plan/canonical serialization/CLI共享节点、static checks和预计耗时。D6 strict guard 下的非 integration 批次 `76 passed, 1 deselected in 35.24s`，覆盖 guard、primary seed/even median、399/400/count=400/median=0.02/CI equality/delta=0、validity failure与misclassification、tie SD、closed schema、immutable external-summary refusal、deterministic bytes、atomic/no-overwrite、tamper、C10 cohort separation、A/B/C recipe/figure inputs、CLI closed surface及既有 D5 CLI regression；`py_compile`、Draft 2020-12 schema check和 `git diff --check` PASS。最终唯一 full 500-coordinate/1,000-unit D5 synthetic sealed-root integration（same-snapshot receipt、analysis-incomplete L0→no-drop `NO_GO_VALIDITY`、sealed-source tamper refusal）在 D6 guard 下 `1 passed in 589.65s`。未运行全量 non-GPU/Stage C suite、GPU/model/image tests、smoke、正式 workload或任何正式 root，因为它们超出最小依赖链或被 D6禁止。
- data_access_statement: 只读取 Git、文档、D6/D5 Python source、冻结 Stage D JSON/CSV metadata和测试临时 synthetic roots/packages；没有读取、`stat`、hash、decode或 preprocess任一 reserve JPEG，没有访问真实 Stage D outcome/正式 root，没有加载真实模型/state dict，没有 CPU inference、attribution、GPU/CUDA、smoke或正式 workload。D6 strict guard 在 collection前阻止 reserve/formal/outcome/image/weight路径、模型构造/调用、autograd、CUDA和 subprocess；测试输出仅为临时 synthetic artifacts，非持久依赖。
- package_complete: true（仅指 D6 implementation/synthetic acceptance完成，不表示正式 Stage D root或分析包存在）。
- scientific_validity: not_evaluated（正式 Stage D evidence不存在；synthetic PASS/FAIL只验证实现语义）。
- paper_data_or_figures_generated: synthetic temporary only；没有正式 paper-data、PNG或 figure。
- deviations_or_protocol_amendments: none；未改冻结科学问题、coordinates、thresholds、bootstrap、workload或上游 evidence。开发中发现并修复三类测试/implementation issue：重复 detached pytest进程已精确终止后改为单例 full fixture；immutable result的 list/tuple断言改为内容断言；full fixture tamper probe的测试 glob改为实际 `units/<digest>/attempts/attempt-000001` layout。均非正式 runtime/scientific attempts。
- known_limitations_and_claim_boundaries: D6证明的是 synthetic sealed receipt、统计、decision和export contract，不证明 reserve JPEG、真实模型/CUDA、D7 budget、正式 campaign或正式 scientific effect。D6 CLI的聚合入口仍只能在 synthetic D5 root上调用；正式 D10 analysis需D9明确授权。
- next_task_authorized: true（只授权 D7 在已知 Stage C calibration/audit或 synthetic输入上执行 smoke、预算、producer freeze及 execution-authorization gate评估；不授权 reserve image、正式 workload、D8+、正式 aggregation/decision/paper export或 analysis authorization）。
- authorization_issued_this_task: false
- execution_authorization_observed_valid: false
- next_task: D7

#### 下一对话完整提示词

```text
本次唯一任务：只执行 Stage D 的 D7。

目标：在 D6 已验证的 synthetic sealed-root / aggregation / paper-package contract之上，只使用已知 Stage C calibration/audit 或 synthetic inputs，完成受控 smoke、独立预算、producer freeze与 execution-authorization gate评估。不得访问 reserve JPEG、不得运行正式 500图 workload、不得进入 D8/D9/D10，也不得生成正式 aggregation、decision、paper data或 figure。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

开始条件：
1. 阅读本文文首状态、第3–9节、最新 D6 completion record与本提示词；以 `git log -1 --format=%H`取得 D6 `git_sha_completion: SELF`实际 SHA，核验 index/worktree clean。
2. 唯一解析并复核 D0/D1/D2 refs、D3/D4/D5/D6 commits、C10 frozen bindings；重跑D3 `validate-contracts`、typed plan validator和D6 guard/static surface checks，但不得重建C10科学结果。
3. 核验 plan digest `5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443`、coordinate digest `cc560e2a2d4e1220ccd812c073119a2b679de213f9770358f7decf51824661fb`及 counts `500/1,000/1,010/10,000/210,000/211,010`。
4. 初始状态必须仍为 `execution_authorization=NOT_ISSUED`、`analysis_authorization=NOT_EVALUATED`、`stage_d_execution_started=false`；D6 synthetic results/packages都不是正式结果、权限或持久依赖。

必须完成：
1. 只对已知 Stage C calibration/audit IDs或synthetic输入运行最小 smoke；在任何读取前建立严格 guard，拒绝 reserve tree、候选reserve JPEG、正式D8/D9/D10 root、模型权重以外的未授权路径和任何正式 outcome。
2. 冻结 producer source revision/tree、依赖 lock、model config/state-dict binding（只作D7 producer binding/availability验证；不得让 reserve sample执行），以及 Stage D runtime/producer/CLI source inventory；producer变动或dirty tree必须fail closed。
3. 用实际允许的 smoke telemetry建立或验证全部四个 typed resource profiles，计算 projected launch-safety及completed-campaign gates；D7可收紧但不得放宽D2 caps。将 `BLOCKED_NO_LAUNCH`与 `NO_GO_COST`保持分离。
4. 仅当所有D7 preconditions、smoke、profile/budget、source bindings、fresh output-root policy和D6 contract checks均PASS时，构建并独立验证唯一正式 execution authorization；它必须绑定完整500 image/1000 unit plan、D7 frozen producer、shard policy、fresh root和禁止D8 aggregate/decision。任何失败只输出BLOCKED，不降样本、不替换样本、不签发部分授权。
5. 不签发 analysis authorization；它仍为 `NOT_EVALUATED`。D7 execution authorization即使PASS也不执行D8，不读取reserve图像，不初始化正式root。
6. CLI/log只输出闭合脱敏 control summary；加入D7 focused synthetic/known-calibration tests、authorization tamper/fresh-root/budget rejection和strict guard。
7. 更新本文D7状态和完整completion record。仅在D7 PASS且正式 execution authorization已签发时，下一提示词才授权D8.0初始化一个fresh formal root；否则下一提示词只授权诊断/重做D7，不得D8。

明确禁止：
- 不读取、stat、hash、decode或 preprocess任一 reserve JPEG；不访问正式Stage D outcome或formal D8/D9/D10 root。
- 不运行500-image/1000-unit正式workload，不初始化正式root，不执行任何D8 shard。
- 不聚合正式 `delta_fid`，不运行正式bootstrap，不生成正式decision/paper data/figure，不消费或签发analysis authorization。
- 不修改D0/D1/D2冻结contracts、candidate access log、C9/C10 evidence或D6 statistical/reporting contract。

测试计划：
测试前一次性说明D7 focused guard、synthetic/known-calibration smoke、profile/budget、producer binding、authorization/fresh-root/tamper tests、直接共享节点、static checks和预计耗时。禁止全量non-GPU suite、GPU以外模型/图像测试、reserve access和正式 workload，除非有新的明确授权。
```

其中 `D0–D7` 必须在不访问 Stage D outcome 的条件下完成。`D8` 只生产原始 evidence，不做聚合。`D9` 只封存和验证。只有 `D10` 可以生成唯一正式 Stage D 科学决策与论文结果图表。

### 2026-09-04 — recovery-v2 R3 D0-D6 pre-reserve completion record

- status: D0-D2 `FORMAL_PREPARATION_PASS`; D3 `CONTROL_PLANE_READINESS_PASS`; D4 `PRODUCER_IMPLEMENTATION_READINESS_PASS`; D5 `VALIDATION_IMPLEMENTATION_READINESS_PASS`; D6 `ANALYSIS_IMPLEMENTATION_READINESS_PASS`. None is a Stage D execution, analysis or scientific PASS.
- provenance: C8 custody `82825131...` -> C9 producer/authority `0d88fbb...` / `c66e5224...` -> C10 producer/authority/completion `3cf2945...` / `0d0eab0...` / `f1dc44d...` -> R3 producer `753b0f5f...` -> D0 `3ac7bfac...` -> D1 `ad77134e...` -> D2 `87ca4b24...` -> D3 `8440d0f7...` -> D4 `6b208405...` -> D5 `1f4d4ec1...` -> D6 `586634ee...`.
- records: current records are `reports/stage_d_recovery_v2_20260904_r3/d0` through `d6`, each with nonterminal completion/validation material followed by an atomically issued `status.json`.
- false-PASS audit: original D0-D2 are `HISTORICAL_PASS_STALE_BINDING`; original D3-D6 are `SYNTHETIC_OR_IMPLEMENTATION_ONLY_PASS`. The early D0 draft was premature and remains corrected to BLOCKED. Recovery R1 has wrong `$schema` bindings; R2 has incomplete stage source inventories. Both generations are retained as `INVALID_OR_TAMPERED` implementation errors, explicitly not tamper or forgery. R3 supersedes them without overwrite.
- integrity: no forgery, source tamper, cross-root splice, duplicate selected attempt, missing selected attempt or current self-signed PASS without source recomputation was found.
- scope: unchanged fidelity-only `n=500`, primary `(0,9)`, two frozen methods, `fine_4x4_cell_delta_fid`, image statistical unit, 10,000 paired-median bootstrap, `run_stability=false`, no secondary inference and no pooling with C10 `n=15`.
- C10: acceptance remains `BLOCKED`, decision `NO_GO_NOVELTY`, effect `PASS_EFFECT`, translation/crop `FAIL_STABILITY`, overall spatial `FAIL_SPATIAL`, Holm `HOLM_BLOCKED_UNDERSPECIFIED`.
- access and authority: D7/D8/D9/D10 not started; execution/analysis/reserve authorization false; no reserve image path existence check, stat, open, hash, decode, preprocess or inference occurred.
- next task: future D7 is an independent gate and is not authorized by this record.
