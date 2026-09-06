# 阶段 E：必要消融与敏感性分析规划

> 创建日期：2026-09-06。
> 当前状态：`NOT_STARTED`；当前唯一下一任务为 `E0`。
> 前置状态：Stage D D10 已完成并封存，正式决策为 `GO_FIDELITY_SCOPE`。
> 文档性质：定义 E0–E4 的职责、边界、依赖、交付物和验收顺序；不是执行授权，也不是已经完成 Stage E 的声明。

## 1. 当前检查点与主流程

D8、D9、D10 均已完成，不是 Stage E 要重跑的任务。Stage E 从现有 D10 completion checkpoint 向前执行：

```text
D10 已完成
→ E0：冻结 Stage E 消融协议
→ E1：主 baseline / patch-scale sensitivity
→ E2：上采样规则 sensitivity
→ E3：层头及正确/误分类描述性分析
→ E4：Stage E 汇总、验证和封存
→ F0–F4：正式成图与论文交付
```

正式论文图片在 E4 完成后的 Stage F 生成。Stage E 只生成消融数值、结构化报告和绘图配方，不生成正式 PNG、PDF、SVG，不人工挑选“好看案例”。

## 2. 不可变上游绑定

Stage E 必须只读绑定以下 Stage D 终态：

| 对象 | 冻结值 |
|---|---|
| D10 completion ref | `refs/archive/stage-d-fidelity-v1-d10-completion` |
| D10 completion commit | `7520689bb031304e4789567f99d093062aa50f72` |
| D9 report | `reports/stage_d_d9_recovery_v2_20260906/d9_formal_validation.json` |
| D9 report SHA-256 / canonical | `bbfb6b5ad879116f6849e6f0a058780fdfd5e54b656e62f0674a3c3184c810d5` / `50e2c1455534e5b10aeacb79fac42d128317e0e943c817addf47671f599ea206` |
| D9 sealed root inventory digest | `76d7d532049a5c90a09c74ba56b019042d7f2f0174ccb439bd8f67713514cc5a` |
| D10 report | `reports/stage_d_d10_recovery_v2_20260906/d10_formal_decision.json` |
| D10 report SHA-256 | `7eac27b11b9df72a91c6e78cd0d46ea30f7d8701f0d41cbc29e9cb76695b5a08` |
| D10 report canonical digest | `db9b235cd8cfdeb3caf99445939443b17007ca5225c710c1164e36e2593503e6` |
| D10 output-root inventory digest | `b72d36430d149598ccd9cfd8dd586297e7231be29e61b44309e9971087e46535` |
| D10 paper root | `evidence/stage_d/d10_recovery_v2_20260906/formal_r2/paper` |
| D10 figure recipes SHA-256 | `aaa1858e5a2b0e820d4ea5cb3072a41e3cb6f4f507daefa319dc9ab66a1f0ad1` |
| Stage D cohort | D1 冻结的 500 图；与 Stage C 20 图互斥 |
| Stage D primary | `(block=0, head=9)` |
| Stage D decision | validity `PASS`；fidelity `PASS_FIDELITY`；cost `PASS_COST`；`GO_FIDELITY_SCOPE` |

D10 主统计是 `n=500`、positive `500/500`、mean `delta_fid=2.630671903027281`、median `2.3081996221714247`、sample SD `1.4806950824072174`、paired image-bootstrap 95% CI `[2.1110806126093067, 2.445235411755993]`。这些值是 Stage E 的只读主结果锚点，不在 E0–E4 重算、替换或重新决策。

C10 历史状态必须继续原样保留：`NO_GO_NOVELTY`、`FAIL_SPATIAL`、`BLOCKED`、`HOLM_BLOCKED_UNDERSPECIFIED`、`downstream_authorized=false`。C10 `n=15` 与 Stage D `n=500` 不得池化为 `n=515`。

## 3. Stage E 的科学角色

Stage E 回答“Stage D 主结果对必要分析选择是否敏感，以及效果在冻结结构与预测分层中如何变化”，不回答新的普遍优越性问题。

必须遵守以下边界：

- Stage D primary、baseline、endpoint、500 图分母和 D10 decision 均不可修改；
- Stage E 是 D10 之后规划的 secondary/descriptive analysis，不能伪装成 Stage D 的事前 confirmatory endpoint；
- E0 必须在读取任何新增 Stage E outcome 前，一次性冻结 E1–E3 的完整范围；
- 默认不输出 p-value、拒绝结论、winner selection 或新的 `GO/NO_GO`；若确需推断，必须在 outcome access 前冻结 exact hypothesis family、方向、统计量、raw p-value 和 multiplicity correction；
- 不因结果不理想而删样本、换 head、换上采样规则、改分母或只报告有利变体；
- 负向、无差异或退化结果是有效的敏感性结论，应进入 E4 和论文 limitations；
- Stage E 结果不能撤销 D10 在其冻结范围内的 `GO_FIDELITY_SCOPE`，但可以限制解释范围。

## 4. 共用数据、计算与保留原则

1. 默认使用与 Stage D 相同的 500 图 cohort、模型、预处理、target、replacement baseline、intervention fractions、tie-repeat 和 seed 规则。
2. E0 先做 reuse inventory：逐项判断 D9/D10 已验证的 L0/L1、curves 和 metadata 能否确定性派生所需结果。可诚实复用的不得无故重新计算。
3. 只有无法从已封存资产得到的 endpoint 才允许新增计算；新增计算必须有独立 run spec、预算、授权、immutable attempts、provenance、finalize 和 validation。
4. D9 sealed root 与 D10 output root 全程只读。Stage E 输出写入独立 `evidence/stage_e/` 和 `reports/stage_e_*` 路径。
5. 完整 500 图是默认分析分母。若某变体在 E0 被明确限定为较小 diagnostic cohort，必须预先冻结样本 ID、选择理由和用途，并禁止把它写成 full-cohort 结论。
6. failure、NA、non-finite、misclassified 和 retry history 全部保留；不得 complete-case 静默删行或零插补。
7. 数值分析只读取验证后的数组、曲线和结构化数据；display PNG、截图、overlay 和人工视觉判断不得进入统计链。
8. E1–E4 不正式渲染论文图片；只允许生成 CSV/JSON、必要数值数组、QC/provenance、汇总表和 machine-readable figure recipes。

## 5. E0：冻结 Stage E 消融协议

### 5.1 唯一目标

在访问新增 Stage E outcome 前，冻结 E1–E3 的科学范围和 E4 的验收合同。D10 已知这一事实必须写入 protocol；E0 不能声称 Stage E 在 Stage D outcome 之前预注册。

### 5.2 必须冻结

- D10/ref/report/paper-data 的路径、SHA-256 和 canonical digest；
- 500 图 registry、样本顺序、预测正确性字段和不丢行规则；
- 每个 analysis family 的方法 ID、变体 ID、尺度、层头集合、统计单位和输出列；
- 可复用资产清单、必须新增计算清单及禁止重算清单；
- intervention fractions、tie 规则、seed namespace、AUC/`delta_fid` 公式和区间算法；
- failure/NA/non-finite/retry、预算、retention、provenance 和 whole-root inventory 规则；
- descriptive-only 状态，或另行完整冻结的 multiplicity contract；
- Stage E figure recipe schema，以及 `renders_png=false`；
- E1、E2、E3、E4 的开始条件和终止条件。

### 5.3 推荐的最小冻结范围

为控制成本并避免结果驱动扩张，本轮必要范围建议固定为：

- cohort：完整 Stage D 500 图；
- primary anchor：`(0,9)` 的 D10 fine-4×4 NNS 与 matched baseline，只读引用；
- patch-scale：把 NNS 聚合到原生 14×14 patch 单位，并与 matched 原生 patch contribution 在同一 patch intervention 口径下比较；
- upsampling：`uniform_mass_divide_by_16`、mass-preserving nearest、bilinear、bicubic；
- layer/head：只用 Stage C 在 holdout access 前冻结的四个 depth-band representatives `(0,9)`、`(4,2)`、`(6,10)`、`(11,2)`；不得根据 D10 或 Stage E 结果新增“更好”head；
- prediction strata：overall、correct、misclassified，全部样本仍保留在 overall；D10 已记录 misclassified `82/500`。

如果 E0 经过计算/可辨识性审计认定某项不能公平实现，必须在读取该项 outcome 前标为 `NOT_RUN_WITH_REASON`，不能用临时近似冒充。

### 5.4 E0 验收

E0 只有在协议、配置/schema、typed plan、预算、授权边界和测试全部冻结，且未读取新增 E1–E3 outcome 时才能 `PASS`。E0 `PASS` 只授权 E1，不代表任何消融结论。

## 6. E1：主 baseline / patch-scale sensitivity

### 6.1 目的

解释 D10 的 fine-4×4 优势是否依赖于细粒度 intervention 单位，并把 matched patch baseline 放回其原生 patch 尺度进行公平比较。

### 6.2 必须报告

- D10 fine-4×4 primary 只读锚点；
- NNS 4×4 cell scores 聚合到 14×14 patch 后的 patch-scale fidelity；
- matched 方法原生 14×14 patch contribution 的 patch-scale fidelity；
- 每图 paired 结果、overall summary、有效/失败/NA 计数和实际成本；
- fine-4×4 与 patch-scale 的差异及其解释边界。

不得把 patch-scale 中表现最好的方法/尺度替换成新的 primary，也不得把 E1 与 D10 合并成一个扩大的确认性检验。

## 7. E2：上采样规则 sensitivity

### 7.1 目的

检验 matched patch contribution 映射到 4×4 cell 网格时，结论对空间细化规则是否敏感。

### 7.2 必须冻结并验证的细节

- 14×14 patch grid 到 56×56 cell grid 的坐标约定；
- `uniform_mass_divide_by_16`、nearest、bilinear、bicubic 的精确定义；
- `align_corners`、边界、signed values、positive-part ranking 和 mass preservation/renormalization；
- 每种规则的 conservation tolerance 与 deterministic golden test；
- 同一图、同一 target、同一 fractions 和 paired seeds。

`uniform_mass_divide_by_16` 与 mass-preserving nearest 可能只相差正常数缩放，从而产生完全相同 ranking。E2 必须先做解析和 golden-test 等价性检查；若等价，应报告为 `EQUIVALENT_RANKING`，不能包装成两项独立支持证据。

E2 输出每规则的逐图结果、汇总、两两差值、等价/退化状态和 figure recipes，但不选择“最佳上采样规则”替换 D10 baseline。

## 8. E3：层头与正确/误分类描述性分析

### 8.1 层/头分析

默认比较 E0 冻结的四个 depth-band representatives：`(0,9)`、`(4,2)`、`(6,10)`、`(11,2)`。`(0,9)` 始终标为 Stage D primary anchor，其余三项只用于描述深度变化。

必须报告所有冻结坐标，不得只展示最强 head，不得根据 Stage E 结果追加候选或重新命名 primary。层/头图的图注必须声明这些坐标来自 Stage C 的预先冻结 selection，而不是在 Stage D 500 图上重新搜索。

### 8.2 正确/误分类分层

按 D10 已记录的 clean prediction 状态分为 correct 与 misclassified，并同时保留 overall 500 图结果。必须报告每层的 `n`、有效/失败/NA、效应分布和区间；misclassified 样本不得从 overall 删除。

该分层是描述性的 effect-modification view，不证明因果关系，也不允许用某一分层替代 full-cohort primary conclusion。若分析 clean margin 等连续难度指标，E0 必须先冻结变量、方向和缺失处理。

## 9. E4：汇总、验证和封存

E4 不新增科学变体。它只负责：

1. source-recompute E1–E3 的 expected/observed coordinates、pair bindings、hashes、失败/NA/retry 和统计汇总；
2. 分开给出 package completeness、workflow integrity、scientific validity 和 actual cost；
3. 生成并验证完整的 `paper_data` CSV/JSON、analysis report、provenance、payload manifest、status 和 figure recipes；
4. 记录所有预注册变体，包括负向、等价、失败和未运行项；
5. 证明 D9/D10 上游未修改、D10 decision 未修改、C10 历史状态未修改；
6. 封存 Stage E root，记录 whole-root inventory digest、Git commit/ref 和下一步 F0 入口。

E4 `PASS` 表示 Stage E 包完整、可追溯且可供 Stage F 消费，不表示每个变体都必须支持 NNS，也不产生新的 Stage D `GO/NO_GO`。

## 10. Stage E 输出合同

E4 至少应交付：

- `paper_data/patch_scale_sensitivity.csv/json`；
- `paper_data/upsampling_sensitivity.csv/json`；
- `paper_data/layer_head_descriptive.csv/json`；
- `paper_data/prediction_strata.csv/json`；
- `paper_data/resource_and_qc.csv/json`；
- `figure_recipes.json`；
- `analysis_report.json`、`provenance.json`、`payload_manifest.json`、`status.json`；
- E0 protocol/config/schema/run-spec 及 E1–E4 completion records。

所有表必须带 variant ID、sample ID/selection order、cohort/stratum、有效性状态、source hashes 和 method/target/scale/layer/head 身份。所有 recipe 必须声明 `numeric_source=paper_data_only` 和 `renders_png=false`。

## 11. Stage F 交接条件

只有 E4 完成封存并通过验证，才进入 Stage F：

- F0：实现并验证 display renderer；
- F1：渲染 D10 主结果统计图；
- F2：渲染 Stage E 消融图；
- F3：按冻结规则选择并渲染 attribution/overlay、weak/failure 和 difficult cases；
- F4：完成 PNG 预览、PDF/SVG 论文版、图注、排版和 machine-readable display report。

Stage F 必须分开两类图：完整 cohort 数据驱动的科学主图，以及按冻结规则选择的定性展示图。后者可以统一美化和排版，但不能改变任何统计结论。定性选择继续受 [阶段 F 展示选择协议 v1](stage_f_display_protocol_v1.md) 约束。

## 12. 当前下一步

当前只执行 `E0`。E0 应先验证 D10 completion ref/report/paper package 与工作区 clean state，再完成 reuse inventory 和整个 E1–E4 合同冻结。不得重跑 D8、D9、D10，不得在 E0 读取新增 Stage E outcome，不得生成正式论文图片。
