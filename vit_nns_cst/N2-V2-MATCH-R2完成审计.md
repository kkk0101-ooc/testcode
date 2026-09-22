# N2-V2 low-level matching R2 完成审计

日期：2026-09-22

权威结论：`PASS_INPUT_IDENTIFIABILITY_FROZEN / STOP_BEFORE_BEHAVIOR_GATE_AND_N3`

## 1. 目标与完成标准

1. 新建 fresh matching root，旧失败或无效 root 不原位改写。
2. 输入只包括 I7 accepted registry、其绑定 scene 原图、12 个 recipient 原图与 mask、V2 contract、legacy lineage ledger 和 matching 实现/validator。
3. outcome-blind 冻结 8 个 final pairs 和按冻结规则排序的 4 个 reserve recipients。
4. 8/8 pair 的 `L1`、`L2`、`RGB_MEAN_SHIFT`、`HIGHPASS_L1`、`BOUNDARY_L1` ratio 均不超过 2.5，aggregate log-ratio distance 均不超过 1.25，unmatched rate 为 0。
5. background reuse max 为 1，selected source group 和资产哈希与 legacy lineage ledger 零交集。
6. 不读取 checkpoint，不加载模型，不推理，不读取 behavior outcome，不运行 activation、Endpoint、Path NNS、JVP、intervention 或 GPU。
7. matching PASS 只冻结输入层，不自动授权 behavior gate 或 N3。
8. 同步更新主执行规划。

## 2. prompt-to-artifact checklist

| 明确要求 | 证据 | 审计结果 |
|---|---|---|
| fresh root | `evidence/n2_v2_matching_r2_20260922/`；R1 由 `evidence/n2_v2_matching_r1_posthoc_audit_20260922/invalidation_receipt.json` 保留为 `INVALID_PROTOCOL` | PASS |
| I7 accepted registry | `pair_manifest.json.source_bindings`；validator `SOURCE_PATHS_EXACT`、`SOURCE_HASHES_RECOMPUTED` | PASS |
| I7 绑定 scene 原图 | 43 个 registry hash 全部重算；validator `I7_SCENES_RECOMPUTED` | PASS |
| 12 个 recipient 原图与 mask | validator `TWELVE_RECIPIENTS_AND_MASKS_RECOMPUTED` | PASS |
| V2 contract | manifest 绑定 SHA-256；validator `CONTRACT_METRICS_AND_CALIPERS` | PASS |
| legacy lineage ledger | manifest 绑定 SHA-256；validator 对 source group 和资产哈希独立求交 | PASS |
| matching validator | `validate_n2_v2_matching.py` 的 SHA-256 封存在 `validation.json` | PASS |
| 实际读取白名单 | `production_openat.log`、`validator_openat.log`、`access_audit.json`；两条轨迹均为 70/70 允许输入、0 missing、0 unexpected、0 sensitive | PASS |
| outcome-blind selection | `pair_manifest.json.matching_definition`；完整 87,268 候选，安全保留 374 后精确 MILP，mip gap 0 | PASS |
| final 8 pairs | `pair_manifest.json.records`；validator `GLOBAL_MINIMUM_SELECTION_REPRODUCED`、`EIGHT_OF_EIGHT_AND_ZERO_UNMATCHED` | PASS |
| 4 个 reserve recipients | `pair_manifest.json.reserve_recipients`；validator 按 selection score 后 SHA-256 tie-break 独立重排 | PASS |
| 每项 low-level caliper | 五项 ratio 的全体最大值为 1.0682551880280953，小于 2.5 | PASS |
| aggregate caliper | 全体最大 aggregate distance 为 0.08394202952424384，小于 1.25 | PASS |
| unmatched rate | 8/8 matched，unmatched rate 0 | PASS |
| background reuse max=1 | 24 个 background source group 全部唯一，observed max reuse 1 | PASS |
| lineage 零交集 | validator `SELECTED_LINEAGE_ZERO_INTERSECTION`；source group intersection 和 hash intersection 均为空 | PASS |
| 禁止 checkpoint/model/outcome/GPU 等 | 两份 `openat` trace 无敏感路径；manifest 十项 prohibited counters 全为 0；空 `CUDA_VISIBLE_DEVICES`；无模型框架 import | PASS |
| input-only 停止门 | `terminal_receipt.json.authorization.behavior_gate=false`、`model_checkpoint_access=false`、`n3=false` | PASS |
| 同步执行规划 | `执行规划.md` 状态、N2-V2 记录和下一允许请求均更新到 matching R2 | PASS |

## 3. 数值与选择摘要

- 完整 caliper-valid 候选数：87,268。
- 安全下界剪枝后精确求解候选数：374；HiGHS 状态 `Optimal`，MIP gap 0。
- final pairs：8；matched：8；unmatched rate：0。
- reserve recipients：4，顺序为 `9d6e00fe0105be20`、`489e206add49b3cb`、`ee69ffd74891071a`、`0ebd6922130a5386`。
- 五项 ratio 最大值：L1 1.0400001663172425；L2 1.0377400771654466；RGB mean shift 1.0622834979289602；highpass L1 1.0682551880280953；boundary L1 1.0595266905378764。
- aggregate distance 最大值：0.08394202952424384。
- background source groups：24/24 唯一；legacy source/hash intersection：0/0。
- 独立 validator：24/24 checks PASS；pair arrays 与全部 metric 重算最大误差均为 0。

## 4. 证据覆盖判断

`validation.json` 不只检查 manifest 自报状态，还从 43 张 scene、12 组 recipient/mask 和 frozen contract 重新构造完整候选集、重新求解全局 assignment、重新生成 8 个 pair 数组并重算五项指标。`access_audit.json` 则独立覆盖输入白名单与禁止访问边界。因此没有把单一绿色 receipt、manifest 完整性或已有 PASS 当作代理证据。

当前未发现缺失、弱覆盖或未验证要求。N2-V2 matching R2 的 PASS 仅限输入可识别性；执行在 behavior gate 和 N3 之前停止。
