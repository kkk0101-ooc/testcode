# 阶段 G：论文写作规划（原 F4）

> 生效日期：2026-09-08。
> 2026-09-11：原 Stage Y Y1 `BLOCKED_BUDGET` 与
> `next_authorization=NONE` 保持不可变；CPU-only Y1-R1 以一次性 owner
> telemetry waiver 和预算重绑定为依据 PASS，未新增 GPU、图像或科学结果。
> 唯一下一授权为 Y2；F3-Final 继续 `F3_FINAL_PAUSED_PENDING_STAGE_Y`，
> Stage G 不启动。
> 2026-09-10 路由更新：Stage G 继续等待 Stage Y results-to-paper handoff 和据此更新后的 F3-Final terminal PASS。当前不得启动 Stage G；Y0 正式入口见 [Stage Y Y0 protocol](stage_y_route_and_y0_protocol_20260910.md)。
> 当前定义：阶段 G 只负责论文写作、双语一致性、引用核验、编译和稿件封存，不负责生成或修改科学图表。
> 前置阶段：X2 immutable `NO_GO_BROAD_CLAIM_DEVELOPMENT` 不变；post-hoc exploratory X2-MECH M2 已 `PASS`，结果分支为 `MIXED_OR_INCONCLUSIVE`，results-to-paper handoff 已完成。其历史 F3-Final 授权已由 Stage Y Y0 前瞻性暂停；当前唯一下一授权为 `STAGE_Y1_IMPLEMENTATION_VALIDATION_AND_COST_PILOT_ONLY`。Stage Y handoff 与更新后的 F3-Final 图包通过后，Stage G 才能进入正式写作。
> 历史兼容：阶段 E 继续专指 2026-09-06 至 2026-09-07 已执行并封存的消融与敏感性分析。其原规划保留在 [阶段 E 规划](阶段E规划.md)，config、code、evidence、reports、completion 和 archive refs 不改名、不搬移。

## 1. 阶段边界与当前路线

当前顶层路线为：

~~~text
Stage A–D：方法、验证与主实验（已封存）
→ 历史消融阶段 E / E0–E4：post-D10 secondary analysis（已封存；仅保留历史 token）
→ Stage F / F0–F3：论文图表生成
→ Stage G / G0–G7：论文写作
~~~

阶段 F 与阶段 G 的职责必须分开：

| 顶层阶段 | 负责 | 不负责 |
|---|---|---|
| 阶段 F | renderer、统计图、消融图、定性图、补充图、最终图包、caption data 和 figure/source inventory | 论文段落、参考文献叙事、摘要、结论、语言润色 |
| 阶段 G | scientific story、evidence matrix、figure cards、双语稿件、引用、交叉引用、全文 QC 和最终 manuscript package | 新实验、新统计、新图、改样本、改色限、重算结论 |

旧 receipts 中的 `F4_ONLY`、`F4`、`PAUSED_PENDING_F4` 等字段是不可变历史 token。从本决议起，它们只表示“图形工作完成后可进入原 F4 职责”；当前等价入口为 `STAGE_G_WRITING_G0_ONLY`，不再创建新的 F4 evidence、report 或 archive ref。

## 2. 启动条件

阶段 G 写作启动前必须验证：

1. 阶段 F 的 F0、F1、F2、F3 及纳入论文的 F3 扩展工作包均有 terminal completion 或明确的 display-only completion；
2. 最终 Figure 1–6、Supplement S1–S4、caption data、figure inventory 和 source binding 已在阶段 F 的最终图包中冻结；
3. F3 工作包索引与实际 `evidence/`、`reports/`、archive refs 一致；
4. C10、Stage D、历史消融阶段 E、F3-HeadAtlas/F3-ClassCompare 的 scientific boundary 已进入 claim/evidence 约束；
5. local HEAD、tracking ref、remote branch、worktree/index 和 Git LFS 状态可审计。
6. F3-Final 已记录 F3-ClassCompare 的最终交付位置：Supplement S5 或独立随稿 atlas；若纳入，统计图、display-only pages、unclipped controls 和 signed audit 的对应关系已经冻结。

如果阶段 F 只有可验证模块而尚未形成最终图包，允许在 `notes/` 中搭建写作结构，但不得冻结正式图号、图注、Results 数字或 terminal manuscript。

## 3. 写作输入

### 3.1 权威输入

- [论文1当前实现审计、缺口与后续计划](../../论文1实现审计与后续计划.md)：研究问题、旧结果停用规则、允许/禁止主张和论文级验收清单；
- [阶段 G 论文写作指导](阶段G论文写作指导.md)：具体写作与 QC 流程；
- [阶段 F 规划](阶段F规划.md)：正式图表范围、显示边界和历史 completion；
- [X2-MECH M1→M2 科学交接](stage_x_x2_mech_m1_to_m2_scientific_handoff_20260910.md)：working scientific story、M1 状态、证据等级与 M2 预声明解释分支；
- M2 terminal completion 后新增的 `stage_x_x2_mech_results_to_paper_handoff_20260910.md`：M2 的最终 post-hoc exploratory 解释与 F3-Final/G0/G1 交接；该文件在 M2 完成前不得创建为结果文档；
- [F3 工作包和产物索引](阶段F_F3工作包与产物索引.md)及 `manifests/stage_f_f3_work_packages_v1.json`；
- D10、历史消融阶段 E E4、阶段 F 各 terminal package 的 machine-readable paper data、caption data、source manifest 和 completion；
- 核心参考文献：`../相关文章/Understanding neural network through neuron level visualization/`，对应 Dou et al., *Neural Networks* 168 (2023), 484–495, DOI `10.1016/j.neunet.2023.09.030`；
- 其他已核验的真实参考文献 PDF、DOI、BibTeX/EndNote 和文献笔记。后续加入 `../相关文章/` 的文献默认是 related-work 支持材料，不能替代上述 NNS 核心来源文献。

### 3.2 禁止输入

- legacy `49_vit` 中未验证的硬编码数字、错误 Q/K/V、错误 attention scale、错误标签或精选图；
- 从 PNG 反推的定量值；
- AI 凭记忆生成但没有真实来源的引用；
- 尚未通过阶段 F 验证的 draft/contact sheet；
- 为了让叙事更顺而新算的 p-value、合并 cohort、替换失败样本或重画图。
- 把 Dou et al. 2023 的 FCN/CNN 结果写成本项目 ViT 上已经证明的结果。

## 4. 写作目录

阶段 G 新稿件统一写入仓库根目录下的 `papers/<paper_id>/`，不再写入 `evidence/stage_f/f4_*` 或 `reports/stage_f_f4_*`。当前已拆分为两个自包含 paper root：

- `papers/nns_vit_path/`：path-integrated NNS-ViT learning-image 论文；
- `papers/nns_vit_endpoint/`：Endpoint Scanner 论文。

每个 paper root 使用同构目录：

~~~text
papers/<paper_id>/
├── AGENTS.md
├── zh/
│   ├── outline.md
│   ├── abstract.md
│   ├── introduction.md
│   ├── related_work.md
│   ├── methods.md
│   ├── experimental.md
│   ├── results_discussion.md
│   ├── limitations.md
│   └── conclusion.md
├── en/
│   └── 与 zh/ 同构的英文文件
├── notes/
│   ├── scientific_story.md
│   ├── evidence_matrix.md
│   ├── terminology.md
│   ├── claim_reference_matrix.md
│   └── figure_cards/
├── references/
├── tables/
├── figure_manifest.json
└── qc/
~~~

运行报告写入 `reports/stage_g_writing_<milestone>_<date>/`；最终只读封存包写入 `evidence/stage_g_writing/<id>/`。该命名与历史 `evidence/stage_e/` 和 `reports/stage_e_*_20260906` 明确分开。

## 5. 内部里程碑

### G0：稿件初始化与证据绑定

- 验证阶段 F 最终图包及所有上游 completion/ref/hash；
- 建立 `papers/<paper_id>/AGENTS.md`、中英文同构目录和写作状态表；
- 冻结 figure/table 编号、术语、允许/禁止主张和缺证据标记规则；
- 不写 Results 正文，不修改任何科学图或数字。

G0 PASS 只授权 `G1_STORY_AND_EVIDENCE_ONLY`。

若 X2-MECH M2 进入论文，G0 必须验证其 terminal/ref/hash、[results-to-paper handoff](stage_x_x2_mech_results_to_paper_handoff_20260910.md) 和 F3-Final source binding，并保持 `POST_HOC_EXPLORATORY_MECHANISM_ANALYSIS` 标签；M2 不得改写 X2 NO-GO 或 Stage D primary。

### G1：Scientific story、Evidence Matrix 与 Figure Cards

- 从审计文档和 machine-readable 证据重建 central question、主要发现、替代解释和限制；
- 将 Dou et al. 2023 标为 `primary_reference_anchor`：它定义 NNS 源问题和 FCN/CNN 适用范围，本项目贡献是 ViT 迁移与重新验证；
- 每项主张标记 direct/supporting evidence、confidence 和 source path；
- 每张正式图建立 Figure Card，列出 observation、quantitative result、interpretation、alternative explanation 和禁止主张；
- 不使用图像观感替代定量证据。

G1 PASS 只授权 `G2_RESULTS_DISCUSSION_ONLY`。

G1 必须在 evidence matrix 中将 X2 的 broad-claim boundary 与 M2 的 post-hoc mechanism interpretation 分列；只有 M2 terminal 和 results-to-paper handoff 明确允许的机制措辞才能进入 scientific story。

### G2：Results and Discussion

先完成中文，再生成结构、数字、图号和结论一致的英文版本。逐节写作，不一次生成整章。推荐顺序：

1. C10 历史复合结论及 fidelity component；
2. 独立 Stage D 500 图 fidelity 结果；
3. 历史消融阶段 E 的 post-D10 sensitivity；
4. F3 qualitative、HeadAtlas 和 ClassCompare 的限定性观察；
5. failure cases、cost、limitations 和 alternative explanations。

### G3：Methods and Experimental

必须覆盖固定模型和 hash、数据/标签/预处理、真实 forward 与提取器 parity、fixed-clean-attention Value-path 定义、matched baseline、fine-4×4 intervention、fidelity/statistics、资源、provenance、失败/NA 规则。缺失参数标记 `[PARAMETER REQUIRED]`，不得合理补全。

### G4：Introduction、Related Work 与引用

- 在 Results 稳定后再写 Introduction；
- 只引用 `references/` 中已核验来源；
- Introduction 第一层问题应绑定 Dou et al. 2023 的 NNS for FCN/CNN；Related Work 中其他 `相关文章/` 文献按背景、ViT interpretability、attention visualization、gradient attribution、faithfulness evaluation 分类；
- 建立 Claim–Reference Matrix；
- 明确本文不是普遍 ViT 优越性、空间鲁棒性或完整 attention/class attribution 论文。

### G5：Conclusion 与 Abstract

先写 Conclusion，最后写 Abstract。Abstract 中的每个数字必须出现在 Results，每个机制必须在 Discussion 中有证据或明确标为解释性假设。

### G6：双语一致性、模拟审稿与全文 QC

- 中英文结构、图表编号、数字、术语和结论逐项一致；
- 分别进行 novelty/significance、scientific logic/evidence、reproducibility/statistics 三类 reviewer audit；
- 完成 strongest-claims audit、defensive-writing audit、figure/table citation order、引用、缩写、单位、显著位数、重复和缺失参数检查。
- defensive-writing audit 必须采用 [阶段 G 论文写作指导](阶段G论文写作指导.md) 第 24 节的固定表格，区分 `NECESSARY_CAVEAT` 与 `DEFENSIVE`，并优先采用 “claim → evidence → concise boundary”。

### G7：编译、反向审计与封存

- 编译目标期刊稿件和 supplementary；
- 从 manuscript 每个数字、图号、panel、caption 和引用反向定位 source；
- 输出 manuscript inventory、figure binding、reference inventory、bilingual parity、final QC、false-pass audit 和 terminal completion；
- commit、push，并核验 local/tracking/remote/ref/worktree/LFS。

## 6. 科学叙事硬边界

正文必须保持以下层级：

> C10 v1 的综合结论为 `NO_GO_NOVELTY`，因为 fidelity 与 spatial 当时都是预注册硬门；其 fidelity component 在固定 15 图 holdout 上为 `PASS_EFFECT`。独立且不与 C10 池化的 500 图 Stage D cohort 在冻结范围内得到 `GO_FIDELITY_SCOPE`。该结论只支持指定模型、target、primary `(0,9)`、matched baseline 和 fine-4×4 endpoint 下的 fidelity 主张。C10 的 translation 与 center-crop stability 失败仍是强制 limitation；一般空间鲁棒性未被证明。

F3-ClassCompare 中的 class-targeted NNS、Grad-CAM、full-model IG、rollout 和 raw CLS attention 必须按各自 method target 和 class-agnostic 状态描述。面板独立缩放图只支持空间形状观察，不能比较颜色幅值；display improvement 不代表 fidelity 或类别判别能力提高。

### 6.1 Stage F 证据层级与正文用途

Stage G 按下表读取 Stage F，不能以图像观感替代上游数值证据：

| Stage F 输入 | 论文角色 | Stage G 可写结论 |
|---|---|---|
| F1 | 主定量证据 | C10 component result 与独立 Stage D 500 图 fidelity 结果；两个 cohort 分开报告 |
| F2 | post-D10 descriptive sensitivity | fine-4×4 优势的尺度依赖、upsampling sensitivity、layer/head specificity、correct/misclassified 分层及失败/QC |
| F3-Core | 定性审计权威 | 冻结选例、signed/shared-scale 空间结构、failure/NA 和 source provenance |
| F3-Display | 正文定性展示 | 逐面板归一化后的空间支持形状；Figure 5/6 的首选展示层 |
| F3-HeadAtlas | 单样本描述性补充 | 144-head pattern diversity；不足以形成 class-conditioned head weights |
| F3-ClassCompare | 独立 50 图 class-targeted/agnostic 比较 | NNS 对 Grad-CAM 的注册 primary contrast 为负；对 rollout 为正；对 full-model IG/raw CLS attention 未达显著；不产生完整排名或普遍优越性结论 |

F1/F2/F3-ClassCompare 的数字必须从 frozen `plot_data`/`paper_data` 获取；`reports/.../figures_png` 仅用于浏览。F3H-R2 和 F3C4 report-copy-v1 的历史失败回执分别由 F3H-R3 与 report-copy-R1 terminal `PASS` supersede；正文引用必须绑定最终 completion。

### 6.2 图形解释规则

参考论文 Dou et al. 2023 采用“方法示意、定量曲线/表格、蓝底 learning-image 矩阵”的组合。本文沿用这一信息结构：

- Figure 1 解释 ViT Value-path 方法；
- Figure 2–4 承载主结果与消融数字；
- Figure 5–6 使用 F3-Display 蓝色版展示空间形状；
- signed/shared-scale 图进入 supplement 或审计附件，保留符号和同坐标幅值关系；
- F3-ClassCompare 若纳入 S5，先呈现 50 图统计，再呈现 primary、unclipped 和 signed atlas。

蓝色图不是定量强度图。F3-Display 使用逐面板 magnitude normalization；F3C primary pages 使用逐面板 P99 scaling。正文只能据此描述热点位置、支持范围和空间组织，不能比较跨方法、跨层或跨样本颜色强弱。

### 6.3 Results 必须保持的实证层级

1. 主结论：Stage D 在固定 `(0,9)`、同目标 matched baseline 和 fine-4×4 endpoint 上得到 `500/500` positive、median `delta_fid=2.3081996221714247`、95% CI `[2.1110806126093067, 2.445235411755993]`。
2. 历史结果：C10 `n=15` 的 fidelity component 通过，但复合决策仍为 `NO_GO_NOVELTY`，spatial 为 `FAIL_SPATIAL`，acceptance 为 `BLOCKED`。
3. 敏感性：native 14×14 上 NNS/matched 几乎相同；secondary coordinates 的 median `delta_fid` 远小于 `(0,9)`；E3 有两个 typed failures，E4 scientific validity 为 `MIXED`。
4. 定性观察：F3 蓝色图展示 NNS/matched 空间形状差异；完整队列统计而非选例决定 primary conclusion。
5. ClassCompare：50 图 primary `NNS - Grad-CAM` mean `-0.0558543281`、95% CI `[-0.0708505871, -0.0410011760]`、`p=0.0000099999`；NNS 相对 rollout 的 Holm-adjusted contrast 显著为正，相对 full-model IG/raw CLS attention 未达显著，后两项不构成统计等价。

因此正文的总体结论应表述为：

> 在指定 head-direction Value-path 与 fine-4×4 fidelity 设计下，NNS 优于 matched patch baseline；结果具有尺度和 layer/head 依赖性。独立 class-targeted comparison 未显示 NNS 优于 Grad-CAM。

## 7. False-pass audit

阶段 G 至少拒绝以下情况：

1. 阶段 F 最终图包未冻结就锁定图号或 Results；
2. C10 与 Stage D 池化或把 C10 改写为 GO；
3. 隐藏 C10 spatial failure、E2 waiver、E3 typed failures 或 E4 MIXED；
4. 从图像观感、逐面板亮度或精选案例推出方法优越；
5. 引用不存在、未核验或 AI 生成的文献；
6. 中英文数字、图号、术语或结论不一致；
7. 修改阶段 F 图、caption data、sample selection、scale 或 source binding；
8. 新增实验、统计或 endpoint 后仍称为纯写作；
9. Abstract/Conclusion 出现 Results 不支持的数字或机制；
10. 未完成反向 source audit、QC、inventory 和 terminal receipt 就宣称稿件完成。

## 8. 当前状态

截至 2026-09-10，原 F4 和阶段 G 均未启动。Stage F 的既有工作包保持封存；X2-MECH M2 与 results-to-paper handoff 已完成，且不改变 X2 `NO_GO_BROAD_CLAIM_DEVELOPMENT` 或 Stage D primary。当前唯一下一授权为 `F3_FINAL_FIGURE_PACKAGE_ONLY`；该图包完成后，才授权 `STAGE_G_WRITING_G0_ONLY`。

阶段 G 不自动执行。本文只定义写作阶段，不创建稿件正文，不启动新实验，也不修改任何已封存图形或科学证据。
