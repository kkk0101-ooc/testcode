# 阶段 D：fidelity-only 前瞻性复制与外推协议

> 协议版本：`1.0.0`
>
> 冻结里程碑：`D2`
>
> 当前结果状态：`NOT_EVALUATED`
>
> 当前执行授权：`execution_authorization=NOT_ISSUED`

## 1. 权威性、目的和边界

本协议在访问任何 Stage D reserve 图像或 outcome 前，冻结 Stage D 的科学问题、estimand、运行范围、有效性条件、统计量、决策真值表、恢复规则和成本分类。`config/stage_d.json` 与 `config/stage_d_run_spec.json` 是这些选择的 closed、机器可执行投影；本协议给出公式和跨字段语义。任一文档、配置、schema、run spec 或后续实现不一致时必须 fail closed，不能挑选较宽松的解释。

D2 只冻结合同。它没有实现 loader、planner、runner、finalizer、aggregator 或 paper exporter，没有运行 smoke 或正式 workload，没有产生 Stage D 科学结果，也没有签发 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`。D2 通过后只允许进入 D3 的 CPU-only 控制层实现；最早只能由 D7 在所有前置实现、smoke、独立预算和 producer 封存通过后另行签发执行授权。分析授权只能在 D9 对正式 evidence root 封存并独立验证后另行评价。

D2 的数据边界是 metadata-only：不得读取、`stat`、哈希、解码或预处理任一 reserve JPEG，不得读取 Stage D prediction、logit、map、AUC、`delta_fid`、summary 或其他 outcome，不得加载模型、执行 CPU inference、使用 GPU/CUDA 或 attribution。D1 只验证了候选 metadata 中记录的 image hash；live-byte equality 仍为 `NOT_EVALUATED_D1_DEFERRED_TO_D8_0`。所谓 reserve “未见”只表示其 identity 未出现在已声明且封闭的 prior-access ledger 范围内，不表示绝对无人查看过。

任何科学协议修订都必须版本化，并发生在 D7 执行授权及任何 reserve 图像访问之前。访问后改变 primary、样本、阈值、统计或运行范围会失去本协议下的 original confirmatory status；不能用新版本恢复“未见”状态。

## 2. 不可改写的上游事实

Stage D 是查看 C10 后新建的 prospective replication/extension，不是 C10 的重分析或勘误。以下历史状态永久保持：

- C10 primary decision：`NO_GO_NOVELTY`；
- Stage C acceptance：`BLOCKED`；
- workflow blocker：`HOLM_BLOCKED_UNDERSPECIFIED`；
- C10 spatial：`FAIL_SPATIAL`，其中 clean joint spatial 与 Gaussian stability 通过，指定 4-pixel translation 和 center-crop/resize stability 失败；
- `downstream_authorized=false`、`stage_d_started=false`；
- C10 的 8,000-unit `PASS_COST` 只作历史可行性证据，不授权 Stage D，也不替代 D7 的新预算。

C10 在固定 15 图 holdout 上对同一 Value-path primary `(0,9)`、matched patch baseline 和 fine-4×4-cell endpoint 的 `PASS_VALIDITY` / `PASS_EFFECT` 是开展本研究的有限科学依据。它不能表述为 C10 GO、普遍 ImageNet 优越、原生单像素 attribution、完整 Q/K/V 或完整 attention/class-logit explanation。C10 的 `n=15` 与 Stage D 的 `n=500` 必须分 cohort 报告，禁止池化成 `n=515`。

D1 已冻结 `STAGE_D_FIDELITY_V1` 的 500-row metadata registry，selection order 恰为 `0..499`，与 Stage C 20 图按 sample ID、path 和 candidate-recorded image hash 的重叠为 `0/0/0`。这 500 行全部进入确认性分母；不能换样、补样或按预测正确性筛选。

## 3. 唯一科学问题与精确范围

唯一 primary 问题是：

> 在冻结的 ViT-B/16、预处理、fixed-clean-attention head-direction Value-path target、primary `(block=0, head=9)` 和 intervention schedule 下，NNS 的 fine-4×4-cell fidelity 是否优于解释同一标量的 matched patch-token contribution，并能否在 D1 冻结的 500 图 cohort 上复制和外推 C10 的有限 fidelity component。

冻结范围为：

- model：`vit_base_patch16_224.augreg2_in21k_ft_in1k`；state-dict SHA-256 为 `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`；
- preprocess：RGB，短边 resize 到 248，bicubic，center crop `224×224`，mean/std 都为 `(0.5,0.5,0.5)`；
- primary：唯一 `(block=0, head=9)`，不得重选 layer/head；
- target：`fixed_clean_attention_cls_head_direction`；
- image-group 内方法及固定顺序：
  1. `conditional_fixed_clean_attention_head_direction_value_path_ig`；
  2. `matched_fixed_clean_attention_patch_token_contribution`；
- primary endpoint：`fine_4x4_cell_delta_fid`；统计单位是 image；
- fidelity direction：`insertion`、`deletion`；
- 每条 curve 恰有 21 个 fractions：`0, 0.05, ..., 1.0`；
- 每个 direction 恰有 5 个配对 tie repeats，indices 为 `0..4`；
- audit overlay 恰有 5 图，`audit_repeatability_runs=2`，包含 primary clean run；
- `run_stability=false`，Stage D 不运行或推断新的 spatial/stability endpoint。

禁止范围包括 patch-scale fidelity、random/center sanity、其他方法、class-logit line、其他 block/head、head selection、spatial perturbation、secondary p-value、secondary winner selection、正式 secondary inference 和 display pipeline。top-20% sufficiency、sufficiency gap 和 comprehensiveness 只从已注册 primary curve 的 20% 点派生，不增加 curve point、work item 或决策门。

Stage D 的 `holm_status=NOT_APPLICABLE_NO_SECONDARY_INFERENCE` 只在不产生任何 secondary inference 时成立；它不修复或取消 C10 的 `HOLM_BLOCKED_UNDERSPECIFIED`。未来若要增加 secondary inference，必须在任何相应 outcome 访问前另行冻结完整 hypothesis family、方向、统计量、raw p-value 算法与 multiplicity correction。

## 4. Target、map 和 fine-cell 定义

对 normalized clean input `x`、normalized-zero baseline `x0`、固定 block/head，令 query token 为 CLS（index 0），head dimension 为 64。从 clean 真实 forward 捕获模型实际使用的 attention `A(x)`，并定义：

```text
g(z) = sum(j=0..196) A(x)[0,j] V(z)[j]
d    = g(x) - g(x0)
r    = ||d||_2
u    = d / r
f(z) = u^T g(z)
```

baseline 必须是 normalized tensor 全零，对应 canonical RGB `(0.5,0.5,0.5)`。`g(x0)` 使用 clean fixed `A(x)`，不能改成 `A(x0)`。`A`、`d` 和 `u` 在积分前 detach 并冻结。方向门槛为：

```text
tau_dir = 1e-6 * max(1, ||g(x)||_2, ||g(x0)||_2)
```

任一所需量非有限或 `r <= tau_dir` 即 `degenerate_direction`，是 required-primary scientific invalidity；不得换 head、channel、target、baseline、符号或样本。

NNS 使用 float32 Gauss-Legendre IG，steps schedule 固定为 `[512,1024,2048,4096,8192]`，chunk size 为 8；从小到大保留 attempts，并使用首个满足 IG conservation 的 steps。NNS native map 是 RGB channel 带符号求和后的 float32 `224×224` map。matched contribution 为：

```text
c_j = A(x)[0,j] * u^T(V(x)[j] - V(x0)[j]),  j=0..196
```

`c_0` 作为 CLS scalar 单独保留；`c_1..c_196` reshape 为 signed float32 `14×14` patch-mass map。matched baseline 与 NNS 评价完全相同的 `f(z)`。

fine endpoint 把每个 `16×16` patch 分为 16 个互不重叠的 `4×4` cell，共 3,136 cells。NNS pixel map在每个 cell 内带符号求和；matched patch mass 对对应 16 个 cells 均分，即除以 16。投影在 positive/negative split 之前完成，并以 `atol=rtol=1e-6` 验证质量守恒。这里的 “fine” 只表示 4×4-cell/subpatch fidelity，不是原生单像素 attribution。

## 5. Fidelity curve、tie 和逐图 estimand

对被冻结的 scalar target `t`：

```text
Delta = t(x) - t(x0)
sigma = sign(Delta)
S(z)  = sigma * (t(z) - t(x0)) / |Delta|
```

若 `|Delta| <= 1e-6*max(1,|t(x)|,|t(x0)|)`，则为 `degenerate_target_delta`，使 validity 失败。ranking score 是 target-oriented signed map `sigma*M` 的 positive part。RGB 三通道作为一个空间 unit 同时替换；deletion 从 clean 逐步替换为 baseline，insertion 从 baseline 逐步恢复 clean。不得使用 blur、随机颜色、响应 clipping 或重新估计 baseline。

对 3,136 个 cells，fraction 0 使用 `k=0`；其余使用 `k=ceil(fraction*3136)`。trapezoidal AUC 的横轴是实际 `k/3136`，不是名义 fraction；不得把 `S` 截断到 `[0,1]`。Insertion AUC 越高越好，Deletion AUC 越低越好。

tie-priority 合同为：

- base seed：`20260821`；
- namespace：`stage-d-fidelity|20260821|`；
- seed material：`namespace + sample_id + '|' + block_index + '|' + head_index + '|' + repeat_index`；
- seed：上述 UTF-8 字符串 SHA-256 的前 8 bytes，按 big-endian unsigned 解读；
- golden：首个 registry sample、`(0,9)`、repeat 0 的 uint64 seed 必须为十进制字符串 `6554433287525610283`（字符串编码避免 JSON/JavaScript 对大整数失真）；
- priority：`PCG64_PERMUTATION_THEN_LEXSORT_SCORE_DESC_PRIORITY_ASC`；
- 同一 image/config/repeat 的 priority 在两方法和 insertion/deletion 间配对共享。

对 image `i`、method `m`、direction `d`、repeat `q` 的 AUC 记为 `A(i,m,d,q)`。先在图内对五次 nuisance repeats 取算术均值：

```text
Abar(i,m,d) = (1/5) * sum(q=0..4) A(i,m,d,q)
```

同时保留五个 AUC 和总体标准差 `population SD, ddof=0`。tie repeat 不是独立统计样本。每图唯一 primary value 为：

```text
delta_fid(i) = 0.5 * (
    Abar(i,NNS,insertion) - Abar(i,matched,insertion)
  + Abar(i,matched,deletion) - Abar(i,NNS,deletion)
)
```

`positive(i)` 当且仅当 `delta_fid(i) > 0`；恰等于 0 不计 positive。direction、repeat、patch、cell、curve point 均不能增加统计 `n`。

## 6. Audit repeatability overlay

五个 D1 audit overlay 是 500 图的子集，不是第三种样本角色，也不增加统计分母：

| overlay ID | selection order | sample ID |
|---|---:|---|
| `STAGE_D_FIDELITY_V1_AUDIT_00` | 231 | `ILSVRC2012_val/ILSVRC2012_val_00004877.JPEG` |
| `STAGE_D_FIDELITY_V1_AUDIT_01` | 14 | `ILSVRC2012_val/ILSVRC2012_val_00028787.JPEG` |
| `STAGE_D_FIDELITY_V1_AUDIT_02` | 145 | `ILSVRC2012_val/ILSVRC2012_val_00024160.JPEG` |
| `STAGE_D_FIDELITY_V1_AUDIT_03` | 326 | `ILSVRC2012_val/ILSVRC2012_val_00016533.JPEG` |
| `STAGE_D_FIDELITY_V1_AUDIT_04` | 380 | `ILSVRC2012_val/ILSVRC2012_val_00014223.JPEG` |

每个 audit 图和方法恰运行两次 clean metric-ready map，其中 primary clean run 已在 1,000 个正常 clean maps 内，因此只增加 `5×2×(2-1)=10` 个 repeat-clean map items。两次 map 必须 shape、dtype、coordinate/projection identity 和全部 scientific bindings 相同，并逐元素满足 `allclose(atol=1e-6, rtol=1e-5)`。wall time、CUDA peak allocated/reserved 和 artifact bytes 是排除在 repeatability equality 外的 telemetry；排除 telemetry 不允许忽略任何科学或坐标字段。任一 required comparison 失败即 scientific validity failure。

Audit/L2 不进入 AUC、`delta_fid`、bootstrap、fidelity status 或 primary decision。D2–D10 不生成定性 overlay；若论文需要，Stage F 只能用这些预注册 IDs 建立独立 display-only recipe。

## 7. Exact typed workload

typed plan 必须从 registry、run spec 和公式独立推导下列值，不能只信任存储的 totals：

| 数量 | 公式 | 冻结值 |
|---|---:|---:|
| image groups | registry rows | 500 |
| execution units | `500×2 methods` | 1,000 |
| primary clean maps | one per unit | 1,000 |
| audit repeat-clean maps | `5×2×(2-1)` | 10 |
| map work items | `1,000+10` | 1,010 |
| fidelity curves | `1,000×2 directions×5 repeats` | 10,000 |
| fidelity curve points | `10,000×21 fractions` | 210,000 |
| logical work items | `1,010+210,000` | 211,010 |

AUC、`delta_fid`、reconciliation rows、hashes 和 summary 是派生 records，不是额外执行 work items。logical work item 也不是 forward/backward-call 数；预算必须使用真实 runner profile 和 typed units，不能用 211,010 代替调用数。每个 image group 包含同一图的两个 units，执行顺序固定为 NNS 后 matched，并绑定相同 clean input、model、target、baseline、fractions、tie priorities 和 fine operator。

D2 只冻结 sharding 算法，不冻结 shard count。D7 必须在签发 execution authorization 前冻结正整数 shard count；随后 group assignment 恒为：

```text
shard_id = selection_order modulo frozen_shard_count
```

image group 不得跨 shard；coordinator 采用 serial shard handoff。不能因为 shard count 延后而改变 group、unit 或 scientific work-item 坐标。

## 8. Validity：先于一切效果推断

`package_complete`、`scientific_validity` 和 workflow integrity 是不同状态：完整记录的 scientific failure 是 evidence；缺失/篡改/未授权/无法唯一解释的 package 则是 workflow `BLOCKED`，不能伪装成科学 no-go。只有 D9 验证并授权分析的完整 root 才可进入 D10。

Stage D validity 要求同时满足：

1. 500 个 image coordinates 与 1,000 个 unit coordinates exact left reconciliation；每个 coordinate 恰有一个权威 terminal record。
2. 每个 unit 的 required status、failure、manifest、artifact、五次 insertion/deletion curves、AUC、resource 和 provenance 记录齐全并相互绑定。
3. 所有 required numeric values 有限；shape、dtype、coordinate 和 operator 合法。
4. clean 与 baseline endpoint parity 的 max absolute error 严格 `<1e-6`。
5. NNS IG conservation 满足 `abs_error <= 1e-4 + 0.01*abs(delta)`。
6. matched token parity 满足 `abs_error <= 1e-6 + 1e-5*abs(r)`。
7. 不发生 direction degeneration 或 target-delta degeneration。
8. 两方法共享 clean input、model、target、baseline、fractions、seed、projection 和 operator bindings；误分类样本仍保留。
9. D0/D1/D2、authorization、code、model、dependency、registry、run spec、plan、artifact 或 provenance 不漂移。
10. 五个 audit overlay 的 required repeatability 全部通过。
11. display/L2 不进入 quantitative chain；没有行删除、complete-case、补零、imputation、换样或按结果重跑。

primary invalidating NA 原因的 closed 集合为：`empty_valid_region`、`shape_mismatch`、`non_finite`、`degenerate_direction`、`degenerate_target_delta`、`missing_artifact`。这里，完整 terminal scientific record 中明确记录的算法性 `missing_artifact` 是 scientific invalidity；相反，声称存在但在 manifest/inventory 中缺失、无 terminal record 或不可验证的 artifact 是 package/integrity blocker。

任一 required scientific condition 失败，使最终科学 decision 为 `NO_GO_VALIDITY`。这时不运行 primary bootstrap，不对有效子集推断效果，不生成效果表/图；对应 artifact 必须结构化记录 `NOT_GENERATED_DUE_TO_VALIDITY`，但完整 QC、failure 和 provenance 仍须报告。误分类只可作描述性分层，不能从 fidelity 分母删除。

## 9. 确认性统计合同

只有 validity 对全部 500 图通过后才执行确认性统计。必须报告全部逐图 `delta_fid`、strict-positive count、mean、median、sample standard deviation (`ddof=1`) 和 paired image-bootstrap 95% CI。这里跨图 summary 的 sample SD 与第 5 节五个 nuisance AUC 的 population SD (`ddof=0`) 是两个不同量。

primary bootstrap 冻结为：

- input：按 D1 registry 顺序排列的全部 500 个 image-level `delta_fid`；
- ordered sample IDs SHA-256：`b187ff5cbacc1b595f06d12e309602b09965e01e7ee1461fd8a694354d4b8f50`；
- 10,000 次 image-cluster resamples；每次从 `[0,500)` 有放回抽 500 个 indices；
- statistic：每次 resample 的 paired median（对每图已形成的 paired `delta_fid` 取 median）；
- 500 为偶数，median 精确定义为排序后第 250 与 251 个值（1-based）的算术均值；重采样中的 ties 按 multiplicity 原样保留；
- base seed：`20260821`；namespace：`stage-d-bootstrap|20260821|`；
- metric ID：`fine_4x4_cell_delta_fid_median`；
- seed material：namespace 与 metric ID 直接连接；seed 为 SHA-256 前 8 bytes big-endian unsigned；
- primary uint64 seed golden：十进制字符串 `18136624947342364318`（字符串编码避免大整数失真）；
- PRNG：`numpy.random.Generator(numpy.random.PCG64(seed))`；NumPy contract version `1.26.4`；
- draw shape：`[10000,500]`；indices 使用 `[0,500)`、endpoint false；
- CI：percentile `[0.025,0.975]`，`numpy.quantile(..., method="linear")`；不是 BCa，也不是 simultaneous interval；
- draw indices 序列化为 little-endian int64 C-order，bootstrap distribution 为 little-endian float64 C-order，并记录其 digest。

不得单独 bootstrap tie repeats，也不得把它们、curve points、cells 或 methods 当作 clusters。descriptive figure bands 使用独立 namespace `stage-d-figure-bootstrap|20260821|`，每个 method/direction/fraction 以 `namespace + method_id + '|' + direction + '|' + fraction_index` 派生 seed；同样使用 10,000 次、500-image cluster、PCG64 和 linear 2.5%/97.5% percentile，对图内五次 repeat 均值后的跨图 pointwise median 作 band。它不进入 primary decision，也不得称为 simultaneous confidence band。

## 10. 500 图规模和 400/500 guard 的设计理由

`400/500` 保持 C10 既有方向一致性要求的比例而非机械复制绝对数：`12/15 = 0.8`，因此 Stage D 要求至少 80% 的 image-level `delta_fid` 严格为正。它与 median effect threshold `0.02` 和 bootstrap CI 分别约束方向一致性、效应量和不确定性。

500 图同时提供比 C10 更精确的符号比例估计。二项比例在 `n=500` 的最坏情形标准误为：

```text
sqrt(0.25/500) = 0.022360679774997897
normal 95% half-width ≈ 0.04382693235899588
```

在 `p=0.8` 时：

```text
sqrt(0.8*0.2/500) = 0.01788854381999832
normal 95% half-width ≈ 0.035061545887196705
```

这是 prospective replication/precision rationale，不是 retrospective power analysis。本协议没有指定单独的 alternative distribution、alpha/power test，因此不作 formal power claim。

## 11. Fidelity 真值表和全部 equality boundary

validity 通过后，令 `m=median(delta_fid)`，`n_pos=count(delta_fid>0)`，primary bootstrap CI 为 `[L,U]`。必须按以下顺序且只执行一次：

1. 若 `n_pos < 400` **或** `U < 0.02`，则 `FAIL_FIDELITY`。
2. 否则，若 `m >= 0.02` **且** `n_pos >= 400` **且** `L > 0`，则 `PASS_FIDELITY`。
3. 否则为 `INCONCLUSIVE_FIDELITY`。

冻结的 equality 行为是：

| 边界 | 行为 |
|---|---|
| `n_pos=399` | FAIL |
| `n_pos=400` | 满足 positive-count pass component，不触发 count fail |
| `m=0.02` | 满足 median pass component |
| `L=0` | 不满足 pass；若无更早 failure，则 inconclusive |
| `U=0.02` | 不触发 fail |
| `delta_fid=0` | 不计 positive |

failure 先于 pass 的顺序消除可能的条件重叠。整体 scientific decision priority 为：

1. validity fail → `NO_GO_VALIDITY`；
2. validity pass 且 fidelity fail → `NO_GO_FIDELITY`；
3. validity pass 且 fidelity inconclusive → `INCONCLUSIVE_FIDELITY`；
4. fidelity pass 且 completed-campaign cost fail → `NO_GO_COST`；
5. validity、fidelity、cost 全部 pass → `GO_FIDELITY_SCOPE`。

workflow `BLOCKED` 与科学真值表正交；未完成、篡改、未授权或 unsafe root 不能输出科学 decision。“无显著差异”或 inconclusive 不等于统计等价；本研究不注册 equivalence margin 或 TOST。

## 12. Retry、resume 和 fresh-root 规则

retry 决策只能读取 control status、closed failure code 和 resource telemetry，不能读取 AUC、map、`delta_fid` 或其他科学值。可重试基础设施 failure code 的 closed allowlist 是：

- `SCHEDULER_PREEMPTION`；
- `EXTERNAL_PROCESS_TERMINATION`；
- `HOST_LOSS`；
- `GPU_DEVICE_LOSS_OR_RESET`；
- `TRANSIENT_STORAGE_IO_BEFORE_RESULT_PUBLISH`；
- `WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH`；
- `STALE_CLAIM_OWNER_CONFIRMED_DEAD`。

“尚未形成科学结果”精确定义为该 coordinate 同时没有 valid terminal result manifest，且没有 published L0/L1 scientific payload。retry classification 必须发生在访问 scientific payload 前，由 coordinator 明确授权，并绑定 failure code、prior attempt digest、同一 frozen coordinate 和新的 immutable attempt ID。每 unit 最多授权 **一次** infrastructure retry，因而最多 **两次** attempts；旧 attempt 必须永久保留。retry limit 耗尽即 workflow `BLOCKED`。

下列 closed classes 不可重试：`SCIENTIFIC_INVALIDITY`、`OUT_OF_MEMORY_FOR_FROZEN_WORKLOAD`、`LAUNCH_SAFETY_CAP_BREACH`、`AUTHORIZATION_OR_CONTRACT_DRIFT`、`TAMPER_OR_UNSAFE_ROOT`、`AMBIGUOUS_ATTEMPT_OR_OWNERSHIP`、`MANUAL_CANCELLATION_AFTER_SCIENTIFIC_RESULT_ACCESS`。scientific failure 是 terminal evidence并计入 attempt 上限；不能因它重跑、覆写或删除。一个 method 已科学失败时，paired method 仍须运行至 terminal，以完成 1,000-coordinate reconciliation。半完成 pair 中已验证的 terminal unit 不变，只有 pending 且符合 allowlist 的 unit 可在授权后运行新 attempt。

resume 只能跳过具有 valid complete terminal status、status→manifest binding 和全部 artifact hashes/bytes 的 unit。terminal unit 禁止 overwrite 或 append。tamper、authorization drift、unsafe root 或 attempt/ownership 歧义使同 root `BLOCKED`。

若修复必须使用新 formal root，则必须：

1. 保留并披露旧 root 和全部 attempts；
2. 在任何新访问前完成明确 diagnosis，并取得新的 D7 execution authorization；只有科学合同变化时才另需在访问前完成 versioned amendment；
3. 在 fresh root 重跑全部 500 images / 1,000 units；
4. 禁止跨 root 拼接、只补失败样本或选择有利结果；
5. 只有一个完整、有效、明确授权且带 supersession record 的 root 可替代旧 root，成为唯一分析来源。

## 13. Retention 与 outcome-blind 运行

冻结 retention levels：

- L0：全部 1,000 units 的 terminal status、scalars、五次 repeat AUC、target/prediction、failure、resources 和 provenance；
- L1：全部 1,000 primary metric-ready float32 maps 和 registered curves；
- L2：仅五个 D1 audit IDs 的 raw attribution/attention、endpoints、token contributions、convergence 和 replay evidence；
- L3：`NONE`。Stage D campaign 不保存 display PNG。

禁止 per-quadrature-step、per-pixel-file、per-dimension-file 和 display PNG artifacts。每个 execution attempt immutable；failure table 保留每个 expected unit；campaign complete 要求每个 expected coordinate terminal。运行期 CLI/log 只能暴露 progress、control status、terminal counts 和 resource telemetry，禁止中间 metric、AUC、ranking、`delta_fid` 或 summary 泄漏。

D10 数值只能来自 validated sealed L0/L1；L2 永不进入统计或决策。主统计来自 L0，描述性曲线可从 validated L1 生成。任何图必须从 machine-readable paper data 构建，禁止从 JPEG/PNG 或 display artifact 反推数值。

## 14. Provenance 和 source-recomputing validation

每个 unit、attempt、artifact 和 campaign package 至少绑定：

- producer Git commit、tree 和 clean state；
- D7 execution authorization；
- model config、state dict 和 dependency lock；
- 授权后验证的 live image identity 与 SHA-256；
- D0 scope、D1 registry package 和全部 D2 contracts；
- run spec、plan、unit 和 work-item digests；
- method、target、block/head 和 operator versions；
- projection、sign、normalization、baseline、fraction 和 seed contracts；
- terminal state、failure、resource telemetry 和 artifact inventory。

hash algorithm 固定 SHA-256。canonical JSON 使用 UTF-8、sorted keys、compact separators、`allow_nan=false`；JSON file 有 terminal newline。正式 evidence 只允许 ordinary single-link files，拒绝 symlink/hardlink。验证必须从 source bytes 重新计算 binding、manifest、inventory 和 canonical digest；不能只相信 artifact 自报摘要。

## 15. Cost 分类和超限行为

每个 cap 必须唯一属于以下两类之一。

### 15.1 Launch safety caps

| cap | 冻结值 |
|---|---:|
| L0+L1 bytes per execution unit | `67,108,864` (64 MiB) |
| L2 bytes per audit image | `1,073,741,824` (1 GiB) |
| forward calls per execution unit | `30,000` |
| backward calls per execution unit | `30,000` |
| CUDA peak reserved per execution unit | `96,636,764,160` (90 GiB) |

D7 preflight 必须同时证明 projected completed-campaign cost gate 通过，否则 `BLOCKED_NO_LAUNCH`。运行中触发 safety cap 时停止发放新 claims、保留已有 evidence 并把 workflow 标为 `BLOCKED`；这不是 `NO_GO_COST`。

### 15.2 Completed-campaign scientific cost gates

| gate | 冻结值 |
|---|---:|
| 全部 Stage D retained regular-file bytes | `68,719,476,736` (64 GiB) |
| 全部 Stage D H20-GPU-hours | `2,000.0` |
| H20-GPU-hours per image group | `4.0` |

actual cost 包含全部 attempts 和 audit repeats。该 gate 只在完整 campaign 后由 D9/D10 评价；单纯 aggregate overrun 不停止正在运行的完整 campaign，除非另有 safety cap 触发。只有 validity 与 fidelity 已 pass 而 completed cost fail，才输出 `NO_GO_COST`。

D7 必须独立按 1,000-unit typed plan 和保守 profile 投影：

```text
gpu_hours = 1.25 * sum(method_profile_seconds * planned_method_units
                       + audit_repeat_profile_seconds) / 3600
storage_bytes = 1.25 * (estimated_l0_l1_payload_bytes
                        + estimated_l2_payload_bytes
                        + estimated_control_and_provenance_bytes)
per_image_gpu_hours = 1.25 * max_over_image_groups(
    sum(planned_unit_and_audit_repeat_profile_seconds)) / 3600
```

时间 safety factor 为 1.25；storage/control/provenance overhead factor 为 1.25。D7 可收紧但不得放宽 D2 caps；D2 的 `budget_status=NOT_EVALUATED`。

## 16. 允许和禁止的解释

若未来完整合法证据支持 `GO_FIDELITY_SCOPE`，允许的最宽表述仍只是：在本 500 图前瞻性 cohort、冻结 ViT-B/16、fixed-attention Value-path target、primary `(0,9)`、matched patch baseline 和 fine-4×4-cell intervention endpoint 下，NNS 满足预注册 fidelity、validity 和 cost gates。

无论结果如何，都不得声称 C10 GO、C10 acceptance PASS、spatial robustness、translation/crop robustness、所有 spatial 均失败、native single-pixel attribution、完整 attention/QKV/class-logit explanation、统计等价、普遍 ImageNet 优越，或将 C10 与 Stage D 池化。C10 spatial mixed/failure 事实必须在 Study Design、Results 和 Limitations 披露；Stage D 没有新的 spatial endpoint 也必须在 Methods 明示。
