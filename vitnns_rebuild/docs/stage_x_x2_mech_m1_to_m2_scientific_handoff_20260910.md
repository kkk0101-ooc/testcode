# X2-MECH M1→M2 科学叙事与阶段交接

> 日期：2026-09-10。
> 状态：`M1-R1 PASS / M2 AUTHORIZED_NOT_STARTED`。
> 文档性质：M1 completion interpretation and M2 scientific handoff。
> 本文不是新协议，不是 M2 结果报告，不是论文正式 Results，也不修改任何上游 scientific decision。
> M2 完成后另建 `docs/stage_x_x2_mech_results_to_paper_handoff_20260910.md`；不得用 M2 结果覆盖本文。

## 1. 文档职责与权威边界

本文在冻结的 [X2-MECH mechanism protocol](stage_x_x2_mechanism_protocol_20260910.md) 与未来 M2 结果之间建立解释边界：记录 M1 完成时已经成立的论文定位、尚待 M2 回答的机制问题，以及结果未知时允许采用的解释分支。

三类材料必须分开：

| 材料 | 回答的问题 | 权威载体 |
|---|---|---|
| prospective protocol | 怎么算、哪些结果允许进入分析 | `stage_x_x2_mechanism_protocol_20260910.md` |
| empirical result | 实际生成和验证了什么 | M1/M2 evidence、completion 和 terminal report |
| paper interpretation | 证据允许论文怎样组织和表述 | 本文；M2 后由 results-to-paper handoff 接替 |

以下对象保持只读：C10、D10、E4、Stage F/F3C、X1-R1、X2、X2-MECH P0-R1、原 M1 BLOCKED root 和 M1-R1 全部 evidence/report/ref。本文不改变 X2 的 immutable `NO_GO_BROAD_CLAIM_DEVELOPMENT`，不授权 X3–X5、F3-Final 或 Stage G。

## 2. 当前科学状态

### 2.1 已经成立的证据线

1. **Stage D：确认性 fine-4×4 Value-path fidelity。** 在冻结 ViT-B/16、fixed-clean-attention head-direction Value-path、primary `(block=0, head=9)`、同目标 matched patch-token baseline 和 fine-4×4-cell intervention endpoint 下，独立 500 图 cohort 为 `500/500` positive，median `delta_fid=2.3081996221714247`，paired image-bootstrap 95% CI `[2.1110806126093067, 2.445235411755993]`。该结论只属于这一冻结范围。
2. **Stage E：post-D10 descriptive applicability。** native 16×16 patch 粒度的 NNS/matched `delta_fid` 几乎相同且接近 0；四个预先冻结坐标 `(0,9)`、`(4,2)`、`(6,10)`、`(11,2)` 的 overall median `delta_fid` 分别为 `2.3081996222`、`0.0733655096`、`0.0389110607`、`0.0561722386`。这些结果表明尺度与 layer/head 依赖性，不构成新的确认性 winner selection。
3. **X1-R1：NNS-v2 method validation。** NNS-v2 total attribution 与相同 target/baseline/path/quadrature 的 full-model IG identity control 一致；135 个 complete cuts、38,107 个 logical components 和注册的 head/residual/MLP accounting 通过。该证据支持 numerical and architectural completeness 的实现验证，不等于普遍解释性能优越。
4. **F3C 与 X2：broad-claim boundary。** F3C 的注册 class-targeted NNS–Grad-CAM primary contrast 为负；X2 development stress test 的 geometric cells 为 `0/60`，总 absolute stability 为 `50/115`，且相对 Grad-CAM 的 stability noninferiority 未通过。因此不能声称普遍 geometric equivariance、普遍优于 Grad-CAM 或普遍优于现有 ViT 解释方法。
5. **M1-R1：机制分析的数据生产完成。** M1 已补齐 transformed/inverse maps、共享 valid masks 和 4×4/16×16 class-logit ROAD fidelity 数据；M2 尚未执行，因此 patch 内外误差、transport ceiling、结构关联和 granularity–fidelity–equivariance 机制结论尚未形成。

### 2.2 尚未成立的机制命题

在 M2 terminal completion 之前，下列说法一律视为 `[M2 EVIDENCE REQUIRED]`：

- geometric instability 主要来自 patch 内细节；
- coarse spatial support 比 pixel-level attribution 更稳定；
- NNS-v2 存在 granularity–fidelity–equivariance trade-off；
- 聚合能够解释或消除 NNS-v2 相对 Grad-CAM 的 stability gap；
- transport/inverse-map geometry 足以解释 observed instability。

## 3. 当前工作论文主线

状态：`WORKING_SCIENTIFIC_STORY_BEFORE_M2`。其中已有证据的句子可以进入 G1 evidence matrix；涉及机制来源的句子必须等待 M2。

> 我们将 NNS 扩展为一个面向 Vision Transformer 的可审计路径与成分级解释框架。在冻结 ViT-B/16 的指定 head-direction Value-path 上，NNS 的 fine-4×4 排序相对同目标 matched patch baseline，在独立 500 图 cohort 中一致获得更高的 intervention fidelity；该收益具有明确的尺度和 layer/head 依赖性。端到端 NNS-v2 进一步提供经过数值与结构验证、直接定义于输入坐标的 class-logit attribution decomposition，但其 total attribution 与相同合同下的 full-model IG 同一。在五模型 development stress test 中，NNS-v2 total attribution 未达到注册的 geometric-equivariance 和相对 Grad-CAM stability-noninferiority 门槛；F3C 的独立 class-targeted primary contrast 也为负。现有证据因而将 NNS 的主要价值定位为揭示 ViT 内部路径和成分结构，而不是取代所有区域显著性方法。

这段主线不包含尚未验证的“几何失败原因”。M2 只允许在其后增加一段 post-hoc exploratory mechanism interpretation，不得回写前述确认性证据或把 exploratory 结果升级为 broad PASS。

## 4. 主张—证据状态表

| 内容 | 当前证据等级 | M2 前是否可写 |
|---|---|---|
| ViT complete-cut path decomposition | X1-R1 method-validation 支持 | 可以，须注明验证范围 |
| 500 图 fine-4×4 Value-path fidelity | Stage D 确认性支持 | 可以作为主结论 |
| scale/layer/head dependency | Stage E post-D10 descriptive 支持 | 可以写“表明”，不能写“证明普遍规律” |
| 不具备普遍 geometric equivariance | X2 development 支持 | 可以作为边界 |
| 不能声称普遍优于 Grad-CAM | F3C/X2 支持 | 可以作为边界 |
| 几何不稳定主要来自 patch 内细节 | 待 M2 | 不可以 |
| 存在 fidelity–equivariance trade-off | 待 M2 | 不可以 |
| coarse spatial support 更稳定 | 待 M2 | 不可以 |

## 5. 四层贡献及其证据身份

1. **方法贡献——X1-R1。** 将 NNS 的 neuron-level 思想重新定义为适用于 ViT 的 complete-cut path decomposition，并验证 input identity 与 head/residual/MLP accounting。
2. **确认性实证贡献——Stage D。** 在冻结的 Value-path/fine-4×4 合同和独立 500 图 cohort 中确认 NNS 相对 matched patch baseline 的 intervention fidelity 优势。
3. **适用范围贡献——Stage E。** 描述性消融表明该优势主要对应 fine-4×4 粒度，并呈现明显的 layer/head dependency；native patch 粒度和其他冻结坐标没有复制 `(0,9)` 的大效应。这里使用“表明”而不是“证明”。
4. **边界与机制贡献——X2 + M2。** X2 已否定当前注册 development contract 下的 broad geometric-stability 与 Grad-CAM-superiority 主张；M2 决定这一边界能否进一步解释为 fine/coarse granularity、within/between-patch error、transport loss 或 attribution structure 的系统性关系。

## 6. 大主张处置

| 原目标 | 现行处理 |
|---|---|
| 普遍优于现有 ViT 方法 | 删除；当前证据反对该措辞 |
| 普遍优于 Grad-CAM | 删除；F3C 和 X2 均为负向 |
| 空间稳定 | 改为明确边界：photometric/noise development 表现较好，geometric equivariance 失败；M2 只解释失败结构 |
| 完整类别解释 | 改称“数值与结构完备的 class-logit attribution decomposition”，不称语义完整解释 |
| 原生像素级 attribution | 可称“直接定义于输入像素坐标、非插值生成”；不能称已证明像素级定位更准确 |
| Object localization | 未验证，保留为 future work |
| 跨模型/跨数据集泛化 | 未经 formal cohort 验证；X2 只能说明 development failure 跨五个模型分布 |
| fine-4×4 fidelity 优势 | 保留为 Stage D 冻结范围内的论文主结论 |
| scale/layer/head specificity | 升级为主要描述性科学发现，同时保留 post-D10 身份 |

## 7. M1 completion 与恢复历史

### 7.1 最终 M1-R1 记录

- completion ref：`refs/archive/stage-x-x2-mech-m1-r1-completion`；
- completion commit：`b6d14965cce3d7a4618b4f5435331c7a9b954d19`；
- completion status：`PASS`；canonical digest `5750d3063cc9d951d7a7a04131c98864a899893f0ddef6e7837cc70b06be6f9f`；
- denominator：200 expected，199 clean-valid，1 immutable `FAILED_ROUTE_LOCAL_NUMERICAL`；
- 4,776 transformed maps、4,776 inverse maps、24 shared masks；
- 796 条 4×4/16×16 class-logit ROAD summaries、71,640 条 detailed ROAD rows；
- actual resource：约 `9.8705` H20-GPU-hours、`2.7364` wall-hours、`255.2783` CPU-hours、`1.8133` GiB retained；全部 frozen caps PASS，无 waiver；
- source-recomputing reconciliation/validation 与 false-pass audit PASS；
- M2、X3–X5、F3-Final 和 Stage G 均未启动；
- 唯一下一授权：`X2_MECH_M2_CPU_ONLY_MULTISCALE_MECHANISM_ANALYSIS_ONLY`。

权威入口：

- `reports/stage_x_x2_mech_m1_r1_20260910/x2_mech_m1_report.md`；
- `evidence/stage_x/x2_mech_m1_r1_20260910/formal/completion/x2_mech_m1_r1.json`；
- `evidence/stage_x/x2_mech_m1_r1_20260910/formal/reconciliation.json`。

### 7.2 两段工程恢复不得解释为科学变更

1. 原 M1 在生成任何 transformed/inverse map、mask 或 ROAD row 前，因 canonical array digest convention mismatch 终止为 `BLOCKED_FROZEN_IMPLEMENTATION_DIGEST_CONVENTION_MISMATCH`。随后 M1-R1 只把 digest convention 对齐到 sealed X2 encoding，并使用 fresh root；所有 398 个 source maps 的 SHA-256 与原 canonical array digest 均已重算匹配。
2. M1-R1 empirical execution完成后，独立 reconciler 曾因用 `1e-15` 比较 float32-derived `actual_pixel_fraction` 而阻塞；verifier amendment 只把 reconciliation tolerance 改为 `atol=rtol=1e-7`，并允许复用尚未转为 terminal 的有效 pending completion。worker code、route terminals、maps、masks、ROAD rows、方法、样本、transform、seed、threshold 和预算均未改变。

两段恢复均属于 provenance/control-plane correction，不得写成方法优化、结果调参或科学重跑。原 BLOCKED root 保持不可变。

## 8. M2 必须回答的问题

M2 只消费 sealed M1-R1 maps、masks、fidelity data 与既有 X2 numeric evidence，严格 CPU-only、model-free、JPEG-free。它必须完成：

- pixel/4×4/16×16 multiscale stability；
- X2 pixel-level reproduction audit；
- 4×4 和 16×16 exact within/between SSE decomposition；
- transport-only control；
- horizontal-flip full-domain control；
- clean-map concentration/high-frequency structure metrics；
- structure–stability/fidelity exploratory associations 与冻结 bootstrap；
- `POST_HOC_EXPLORATORY_ESTIMABLE` 状态、资源、false-pass、inventory 和 terminal report。

## 9. M2 结果未知时冻结的解释分支

M2 不得根据“哪个故事更好”挑指标、mask、scale、method 或样本。结果完成后只允许落入以下预声明分支；若方向不一致，则报告 `MIXED_OR_INCONCLUSIVE`，不得强选。

### 分支 A：coarse stability 恢复，fidelity 未表现出相应损失

适用模式：4×4/16×16 stability 相对 pixel 明显改善，transport control 不能独立解释 observed gap，且 coarse fidelity 没有与 stability 改善同步下降。

允许措辞：

> Post-hoc exploratory analyses indicate that much of the geometric instability is concentrated in fine-scale attribution ordering, while coarser spatial support is more stable.

此分支称为 aggregation/granularity sensitivity，不称为 trade-off，也不恢复 spatial-stability PASS。

### 分支 B：stability 提升，同时 fidelity 下降

适用模式：随 pixel→4×4→16×16 聚合，stability 系统改善而 ROAD fidelity 系统减弱，并由 within/between decomposition 与结构指标提供一致的描述性支持。

允许措辞：

> NNS-v2 exhibits post-hoc exploratory evidence of a granularity–fidelity–equivariance trade-off: finer attribution supports stronger intervention fidelity but weaker geometric equivariance.

该措辞只能用于 exploratory mechanism interpretation，不能推广为确认性定律或因果关系。

### 分支 C：coarse scale 仍然不稳定

适用模式：4×4/16×16 聚合未实质恢复 geometric stability，或 between-block error 仍占主要部分。

允许措辞：

> The geometric instability extends beyond within-patch high-frequency variation and remains observable at coarser spatial scales.

该结果仍是有效边界发现；不得通过改 mask、平滑或选择 transform 寻找更有利结果。

### Mixed/inconclusive

若不同 metric、transform、model 或 scale 给出不一致方向，只报告完整分层结果及替代解释，使用：

> Post-hoc multiscale analyses yielded mixed evidence about whether aggregation accounts for the observed geometric instability.

## 10. M2 后的结果交接

M2 terminal completion 后必须新建、而不是覆盖本文：

`docs/stage_x_x2_mech_results_to_paper_handoff_20260910.md`

该文档必须：

1. 绑定 M2 terminal/ref、completion SHA-256/canonical digest、whole-root inventory 和正式报告；
2. 从第 9 节选择数据支持的分支，或明确 `MIXED_OR_INCONCLUSIVE`；
3. 列出最终允许/禁止主张及证据等级；
4. 决定 X2-MECH 进入正文、Discussion、Supplement 或仅 provenance appendix；
5. 向 F3-Final 交接图表需求，但不得在 M2 结果前预留或承诺新图号；
6. 向 G0/G1 交接 evidence matrix 条目和 working scientific story。

## 11. 后续执行顺序

```text
M1-R1 PASS（已完成）
→ M2 CPU-only mechanism analysis
→ M2 terminal report/completion
→ stage_x_x2_mech_results_to_paper_handoff_20260910.md
→ 更新并执行 F3-Final composition
→ G0 evidence binding
→ G1 scientific story/evidence matrix freeze
→ G2 Results and Discussion
→ G3–G7 manuscript completion
```

在 M2 terminal 与 results-to-paper handoff 完成前，可以讨论本文件中的 working story，但不得冻结新的机制性 Results、Figure 编号、Abstract 或 Conclusion。
