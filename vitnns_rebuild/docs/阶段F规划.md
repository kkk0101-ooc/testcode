# 阶段 F：正式成图、定性展示与论文交付

> 建立日期：2026-09-07。
> 当前状态：Stage F F2 已完成并封存；当前唯一下一任务为 `F3_ONLY`。
> 文档性质：Stage F 的权威规划、阶段边界和验收合同；不是一次性跳过阶段门禁的总授权。
> 选择规则：定性案例继续受 [阶段 F 展示选择协议 v1](stage_f_display_protocol_v1.md) 约束。

## 1. 目标

Stage F 把已经封存的 D10 主结果和 E4 消融结果转化为可投稿、可复现、可审计的论文图和论文交付包。目标视觉形态参考 [参考文章.pdf](../../论文1/参考文章.pdf)：包含方法流程示意、成组热图、统计曲线、消融对比和多方法定性面板，而不是只输出孤立的默认折线图。

“参考”只指图形组织密度、清晰度和投稿完成度。Stage F 不复制参考文章的图片、数据、文字、网络结构或实验结论；所有本论文图形必须来自本项目已验证的输入，或来自不承载经验结果的方法定义。

Stage F 不产生新的科学结论。它必须同时满足：

1. 科学主图消费完整 D10 或 E4 封存数据；
2. 定性图只使用固定审计样本或冻结规则选择的样本；
3. 显示变换永不回流到数值、统计、排序或 decision；
4. 每个 panel、组合图和最终导出文件都可追溯到输入、代码、参数和 SHA-256；
5. 论文正文继续严格区分 C10、Stage D 和 Stage E 的不同 cohort、角色和有效性状态。

## 2. 启动基线与不可变科学状态

### 2.1 Stage F 启动基线

| 项目 | 冻结值 |
|---|---|
| 当前分支 | `stage-d-d3-pass` |
| E4 completion commit | `7c82eca4e7de488f4537eff2b3fdb59feb3413ab` |
| E4 completion ref | `refs/archive/stage-e-e4-completion` |
| E4 completion | `evidence/stage_e/e4_recovery_v2_20260907/formal/completion/e4.json` |
| E4 completion file SHA-256 | `42ad2b483e40c164b4e26e1d8d65ecfc00703a2c01c2f0589961bcbefa8d58dc` |
| E4 whole-root inventory digest | `75249c74e08cb5038acbc86f71051260f9d9b7b6cc80d23c517e8669f3765f67` |
| D10 completion commit/ref | `7520689bb031304e4789567f99d093062aa50f72` / `refs/archive/stage-d-fidelity-v1-d10-completion` |
| D10 figure recipes SHA-256 | `aaa1858e5a2b0e820d4ea5cb3072a41e3cb6f4f507daefa319dc9ab66a1f0ad1` |
| E4 figure recipes SHA-256 | `57306bc69e399d1ea2353d908ac0d8b22403d17ec29318a3eadf36b3b8d98a54` |
| 展示选择协议 SHA-256 | `de3ee8804f59e5541eb3c170110ab3b6c47f663698638e3fd668ece7f19b35e3` |
| 视觉参考 PDF SHA-256 | `33e323f8cb664864b12daad1f06b6ef0d3268bcafaea4e84ac695fd286f3855b` |
| 当前授权 | `F0_ONLY` |

开始 F0 时必须从源重新核验上述 ref、文件 hash 和 inventory，而不是信任本表。Stage F 的新提交会自然位于 E4 commit 之后，因此验收条件是 E4 ref 为当前 HEAD 的祖先且冻结输入 identity 不变，不要求后续 HEAD 永远等于 E4 commit。

### 2.2 必须原样保留的结论

- Stage D：validity `PASS`、fidelity `PASS_FIDELITY`、cost `PASS_COST`、唯一 decision `GO_FIDELITY_SCOPE`；`n=500`，positive `500/500`，median `delta_fid=2.3081996221714247`，paired image-bootstrap 95% CI `[2.1110806126093067, 2.445235411755993]`。
- C10：acceptance `BLOCKED`、primary decision `NO_GO_NOVELTY`、spatial `FAIL_SPATIAL`、Holm `HOLM_BLOCKED_UNDERSPECIFIED`、`downstream_authorized=false`。C10 `n=15` 与 Stage D `n=500` 必须分 cohort 展示，禁止池化成 `n=515`。
- Stage E E4：package completeness `PASS`、workflow integrity `PASS`、scientific validity `MIXED`、E4 自身 cost `PASS_COST`。
- Stage E E2：科学输出保留，但原始执行曾超过冻结 CPU/GPU budget，只有显式 owner waiver 后才形成 superseding completion；Stage F 不得把它改写为“从未超预算”。
- Stage E E3：3,000 个新计算单元中 2,998 个 valid；selection orders `117` 和 `352` 的失败必须保留、标注且不得补零、删除或隐藏。数值汇总只基于 valid rows，同时报告 expected/valid/failed/NA。
- Spatial robustness 不属于 Stage D fidelity-only 主张；C10 translation/crop failure 必须继续出现在正文 limitation 中。

E4 的 `PASS` 是包完整和工作流可进入 F0，不是把 `MIXED` scientific validity 改写为全量科学 PASS，也不产生新的 Stage D decision。

## 3. 允许输入与禁止输入

### 3.1 数值主图输入

F1 只允许消费：

- `evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis/paper_data/`；
- 同目录的 `figure_recipes.json`；
- D10 completion/report 中与 caption provenance 有关的冻结字段。

F2 只允许消费：

- `evidence/stage_e/e4_recovery_v2_20260907/formal/paper_data/`；
- 同一 E4 formal root 的 `figure_recipes.json`、`analysis_report.json`、`status.json`、`cost.json` 和 completion records。

数值图不得读取 PNG、截图、人工标注或显示平滑后的 map 作为统计输入。CSV/JSON 不一致、source hash 不一致或 recipe 漂移时必须 `BLOCKED`。

### 3.2 定性展示输入

F3 只允许消费：

- D1 冻结 registry：`manifests/stage_d_fidelity_v1/registry.json`；
- D10 `per_sample_delta` 和已验证 paper data；
- D9 封存的 Stage D L1 metric-ready maps 与 fidelity curves：`evidence/stage_d/d8_recovery_v2_20260904/formal_r2/`；
- D1 五个 audit IDs 对应的封存审计证据；
- 被规则选中的原始图像：`assets/data/ILSVRC2012_val/`；
- `docs/stage_f_display_protocol_v1.md`。

F3 可读取被选中样本的 JPEG 以生成显示图，但不得构建或加载模型，不得执行 forward/backward/attribution，不得重跑 D8，也不得创建 retry。当前 Stage F 的 GPU 预算为 `0`。

### 3.3 视觉参考输入

[参考文章.pdf](../../论文1/参考文章.pdf) 只用于 F0 的视觉需求审计。F0 report 必须记录其 SHA-256，并声明：

- `style_reference_only=true`；
- `scientific_input=false`；
- `content_or_graphics_copied=false`；
- `reference_values_entered_results=false`。

参考文章中存在但本项目未验证的 CAM、Grad-CAM、全层 learning image、其他架构或其他数据集结果，不得出现在本轮 Stage F 的经验对比中。若未来新增，必须建立独立的前瞻性实验协议，不能以“补图”为名在 Stage F 计算。

## 4. 图形系统与视觉合同

### 4.1 参考风格的本地化实现

本轮吸收参考文章的以下组织特点：

- 方法流程用矢量节点、箭头和张量/图像缩略图表达；
- 热图按样本、方法或坐标排成规则矩阵；
- 统计曲线使用成组 panel，而不是每个小指标单独占图；
- 原图、方法输出、差异图、overlay 和 fidelity curve 在同一案例行内对应；
- 主文保持少量高信息密度组合图，完整审计案例和 QC 放 supplementary。

不继承会降低可读性或可访问性的做法。默认禁用 rainbow/jet colormap、3D 柱图、双 y 轴、无意义渐变背景和仅靠颜色区分的编码。

### 4.2 尺寸、字体和导出

F0 必须建立且测试至少三个版式 profile：

| profile | 目标宽度 | 用途 |
|---|---:|---|
| `single_column` | 3.5 in | 单栏统计图或补充图 |
| `double_column` | 7.16 in | 主结果和消融组合图 |
| `full_width_qualitative` | 7.16 in | 多列定性案例图 |

统一要求：

- 字体使用可嵌入的 Times/STIX-compatible serif；F0 必须记录实际字体文件和 SHA-256，禁止静默 fallback；
- 最终版正文/刻度不得小于 7 pt，panel label 建议 9–10 pt bold；
- 线宽不得小于 0.75 pt，marker 在灰度打印下仍可区分；
- 矢量图优先导出 PDF 和 SVG；热图可以栅格化，但文字、坐标轴和矢量元素不得整体栅格化；
- 同时生成 600 DPI PNG 论文预览；可另生成低分辨率 contact sheet，但不得作为论文源；
- PDF/SVG 必须嵌入字体、裁掉多余白边并通过透明度/边界检查；
- 所有 panel 使用 `(a)`, `(b)`, ...，顺序与正文和图注一致。

目标期刊模板改变时，只允许增加新的 layout profile；不得因此改变数据、样本、显示选择或科学结论。

### 4.3 颜色、尺度和 overlay

- NNS 与 matched baseline 使用固定、色盲友好的方法颜色；所有统计图保持一致。
- signed attribution map 使用以 0 为中心的 diverging colormap；NNS 与 matched 在同一案例内共享对称色限。
- NNS-minus-matched 使用单独但仍以 0 对称的差异色限。若需要跨案例比较，必须另生成使用全局共享色限的 panel，不能把逐案例归一化图解释为跨案例幅度比较。
- 非负量使用感知均匀的 sequential colormap。
- 原始 map/grid panel 优先保留 cell 边界；平滑或 bicubic resize 只允许出现在标为 `display-only overlay` 的面板。
- overlay 的 resize、clip、alpha、colormap 和归一化参数必须在 F0 冻结并写入每个产物的机器报告。
- 不得用不同色限夸大 NNS 相对 baseline 的视觉优势，不得裁掉失败区域或只展示正值。

现有 `src/vitnns/display.py` 只提供 metric-ready 到 display-only 的显式边界和变换基础；它不等于完整 Stage F renderer。F0 必须在不削弱该边界的前提下实现正式 renderer、CLI、schema、provenance 和导出测试。

### 4.4 确定性与可复现性

同一 committed code、输入 hash、配置、字体和软件环境必须生成字节稳定的 SVG/机器报告，以及像素稳定的 PNG。PDF 若因 metadata 时间戳不能字节稳定，必须先规范化 metadata；不能规范化时必须证明视觉内容和内部对象摘要稳定并记录原因。

每个渲染产物至少记录：

- figure/panel ID、标题和 caption key；
- 所有源文件路径、file SHA-256 和适用的 canonical digest；
- renderer commit、配置 digest、Python/Matplotlib/Pillow/NumPy 版本；
- 字体文件、颜色、尺寸、DPI、线宽、色限、插值、裁剪和 alpha；
- selection rule、sample ID、selection order 和 source map hashes（若适用）；
- PNG/PDF/SVG 的文件 SHA-256；
- `display_only=true`、`primary_decision_modified=false`、`stage_d_statistics_modified=false`。

## 5. 最终论文图目录

F1–F3 先生成可单独验证的模块；F4 只把已验证模块组成主文和 supplementary 图，不重新计算数据。

| 最终图 | 来源阶段 | 内容 | 最低要求 |
|---|---|---|---|
| Figure 1：方法总览 | F3/F4 | 输入图像 → ViT → 固定 block/head 与 clean attention → Value-path IG → fine 4×4 map / matched patch baseline → insertion/deletion fidelity | 矢量图；只表达已实现方法，不承载新经验结果 |
| Figure 2：主结果 | F1/F4 | C10 与 Stage D 分 cohort 的 median/CI、Stage D 500 图 `delta_fid` ECDF、NNS vs matched fidelity curves | 明示 `n=15` 与 `n=500` 不池化；完整 cohort |
| Figure 3：尺度与上采样消融 | F2/F4 | patch-scale sensitivity、uniform/nearest equivalence、bilinear/bicubic sensitivity | 标为 post-D10 descriptive；不得选 winner |
| Figure 4：层头与预测分层 | F2/F4 | 冻结四坐标、overall/correct/misclassified 分层、有效/失败/NA | 明示 scientific validity `MIXED`；保留 orders 117/352 |
| Figure 5：定性对比 | F3/F4 | Original、NNS、matched、difference、NNS overlay、matched overlay、insertion/deletion curves | 样本只按冻结规则选择；共享可比色限 |
| Supplement S1 | F3/F4 | 五个 D1 固定审计案例 | 标为 fixed audit，不得称为 best examples |
| Supplement S2 | F3/F4 | 规则到样本映射、重复选择、weak/failure、misclassified/difficult 案例 | 不得静默替换重复样本 |
| Supplement S3 | F1/F2/F4 | resource/QC、失败/NA、E2 budget waiver 与 E3 mixed-validity disclosure | 与科学图分开但必须随论文包交付 |

Figure 1 的方法示意可以使用确定性声明式布局、TikZ 或等价矢量后端；不得用无法重建的手工截图作为唯一源。Figure 5 当前只比较 NNS 与 frozen matched baseline，不默认添加 CAM/Grad-CAM。

## 6. F0–F4 执行顺序

### F0：冻结并验证统一 renderer

**开始条件**

1. E4 completion/ref/inventory 从源核验通过；
2. local HEAD、tracking ref 和 remote branch 一致，worktree/index clean，Git LFS clean；
3. 当前只持有 `F0_ONLY`，F1–F4 尚未开始。

**必须完成**

1. 建立 Stage F config、schema、run spec、输出 schema、figure catalog 和 caption contract；
2. 实现统一 renderer/CLI，支持统计 panel、heatmap、overlay、curve、method schematic 和组合布局；
3. 冻结第 4 节的版式、字体、颜色、尺度、插值、导出和 provenance；
4. 用 synthetic/golden fixtures 验证 PNG/PDF/SVG、确定性、字体嵌入、色限共享和 display-boundary；
5. 验证所有 D10/E4 recipes 可解析，但不生成正式 F1/F2/F3 图；
6. 输出 source-recomputing validation、false-pass audit、test report、pending→terminal completion 和 `F1_ONLY` 授权。

**建议实现面**

- `config/stage_f_f0.json` 与 schemas；
- `src/vitnns/stage_f_renderer.py`、`src/vitnns/stage_f_cli.py`；
- `tests/test_stage_f_renderer.py`、`tests/test_stage_f_boundary.py`；
- `evidence/stage_f/f0_<id>/` 与 `reports/stage_f_f0_<date>/`。

**验收**

- formal scientific figures rendered = 0；
- model loads、forward、backward、GPU calls = 0；
- golden renders、schema、determinism、font/export、tamper tests 全部通过；
- F0 completion commit/push/archive ref 完成后才可进入 F1。

**预计耗时**：1–1.5 个工作日。

### F1：渲染 D10 主结果模块

**唯一数值来源**：D10 `paper_data` 和 D10 recipes。

**必须生成**

1. `F1-A`：C10 与 Stage D 分 cohort median + interval；
2. `F1-B`：Stage D 500 图 `delta_fid` ECDF/分布；
3. `F1-C`：NNS 与 matched insertion/deletion median curves + pointwise bands；
4. 每个模块的单栏/双栏版本、PNG/PDF/SVG 和 machine-readable report；
5. 主结论数字与 D10 source 的逐字段一致性检查。

**禁止**

- 重跑 bootstrap、改变 CI、过滤 misclassified rows、合并 C10/Stage D；
- 根据画面效果改变 axis range 到隐藏有效数据；
- 在 F1 提前读取 F3 原图或人工选样本。

**验收与授权**：F1 source-recomputing validator 必须证明 plotted values 与 D10 字节/字段绑定一致；completion 只能授权 `F2_ONLY`。

**预计耗时**：0.5–1 个工作日。

### F2：渲染 Stage E 消融模块

**唯一数值来源**：E4 `paper_data` 和 E4 recipes。

**必须生成**

1. `F2-A`：fine 4×4 anchor 与 native 14×14 patch-scale sensitivity；
2. `F2-B`：上采样 sensitivity，并把 uniform/nearest 标为同一等价证据类；
3. `F2-C`：四个冻结 layer/head 坐标及 overall/correct/misclassified 分层；
4. `F2-D`：expected/valid/failed/NA 与 cost/QC disclosure panel 或 supplementary table；
5. 所有 caption 明示 `POST_D10_SECONDARY_DESCRIPTIVE`、无 winner selection、无 confirmatory p-value。

**失败显示合同**

- orders 117 和 352 必须作为 typed failure 计数或标记出现；
- valid-only 数值汇总必须与 expected/valid/failed/NA 同时展示；
- 无合法数值的失败项显示为明确的 missing/failure marker，不得补零或连线穿过；
- E2 owner budget waiver 进入 resource/QC caption 或 supplementary，不得省略。

**验收与授权**：F2 validator 必须重算 plotted row membership 和状态计数，但不得重算新科学 endpoint；completion 只能授权 `F3_ONLY`。

**预计耗时**：0.5–1 个工作日。

### F3：规则选例、方法示意与定性面板

**选择顺序**

1. 固定载入 D1 的 5 个 audit IDs；
2. 按冻结规则计算 `best_effect`、`median_effect`、`baseline_strong_nns_stronger`、`weak_or_failure`、`misclassified_or_difficult`；
3. 完整记录 tie-break、fallback、重复命中和 `NO_ELIGIBLE_EXAMPLE_FOR_RULE`；
4. 选择 manifest 先封存，再读取所选原图和 maps 渲染；人工视觉判断不得改变 manifest。

**必须生成**

1. Figure 1 方法总览的矢量源和导出版本；
2. 每个规则案例的 Original / NNS / matched / difference / overlays / curves 模块；
3. 五个固定审计案例的 supplementary 模块；
4. rule-to-image manifest、caption data、每个源图/map/curve 和 render 的 hash；
5. 可供人工审阅的 contact sheet，但 contact sheet 不是正式论文源。

**显示约束**

- 统一预处理画幅，不得为单个案例手工裁剪目标区域；
- NNS 与 matched 同案例共享色限，difference 独立对称色限；
- 展示平滑、resize 和 alpha 只影响 overlay，并必须标为 display-only；
- 弱效果、误分类、失败或重复选择不得被“更好看”的替代案例替换。

**验收与授权**：validator 必须从 D10 表独立重建选择结果，核对 D9/D1/JPEG/map identities，并证明 model/GPU 未使用；completion 只能授权 `F4_ONLY`。

**预计耗时**：1–2 个工作日。

### F4：组合图、图注、论文写作与最终导出

F4 只能消费已经通过验收的 F1–F3 模块。F4 可以调整 panel 布局、留白、标题位置和文字换行，但不能更改 panel 数据、色限、样本、选择规则或状态。

**必须完成**

1. 按第 5 节形成 Figure 1–5 和 Supplement S1–S3；
2. 完成图注、正文交叉引用、Results、Ablation、Failure Cases、Limitations 和方法图说明；
3. 保持 C10 与 Stage D 分 cohort 叙述，完整披露 spatial failure、E2 waiver 和 E3 mixed validity；
4. 导出 journal-quality PDF/SVG、600 DPI PNG、预览 contact sheet；
5. 生成 manuscript source、编译 PDF（若目标模板已冻结）和 machine-readable display report；
6. 从最终 manuscript 反向核对每个数字、图号、panel、caption 和 source record；
7. 完成 F4 completion、Git/LFS publication 和 Stage F archive ref。

论文骨架、相关工作和方法文字允许在 F0–F3 并行起草，但任何结果数字、图注和最终图号必须在 F4 绑定正式 F1–F3 产物。不存在的文献、实验、p-value、方法比较或统计量不得补写。

**验收**

- manuscript 中出现的每个经验数字都能在 D10/E4 或 Stage F report 中定位；
- 所有正式图片来自已验证模块，draft/contact sheet 未被误引用；
- PDF 无缺字、裁切、低分辨率、字体未嵌入或颜色含义漂移；
- 最终 figure inventory、manuscript inventory 和 Git/LFS 状态 clean；
- Stage F completion 不产生新的科学 decision。

## 7. 输出、封存与 Git 合同

每个阶段使用独立、不可覆盖的输出根：

```text
evidence/stage_f/f0_<id>/
evidence/stage_f/f1_<id>/
evidence/stage_f/f2_<id>/
evidence/stage_f/f3_<id>/
evidence/stage_f/f4_<id>/
reports/stage_f_f0_<date>/
...
reports/stage_f_f4_<date>/
```

F4 的正式论文包必须位于版本化 evidence tree，例如：

```text
evidence/stage_f/f4_<id>/paper/
  manuscript/
  figures/main/
  figures/supplementary/
  captions/
  display_report.json
  figure_inventory.json
  manuscript_inventory.json
```

每阶段均采用 pending→terminal receipt；terminal receipt 必须在所有 payload、manifest、inventory 和 validator 完成后原子发布。不得修改或覆盖上一阶段 root。

建议 archive refs：

- `refs/archive/stage-f-v1-f0-completion`；
- `refs/archive/stage-f-v1-f1-completion`；
- `refs/archive/stage-f-v1-f2-completion`；
- `refs/archive/stage-f-v1-f3-completion`；
- `refs/archive/stage-f-v1-f4-completion`。

每个 terminal 阶段完成后必须 commit、push，并核验 local HEAD、tracking ref、remote branch、archive ref、worktree/index 和 Git LFS 状态。大型二进制使用 Git LFS；若正式 payload 因体积留在本机，至少将代码、配置、manifest、hash、报告和可重建说明纳入版本控制。

## 8. False-pass audit

F0–F4 每阶段都必须生成 false-pass audit，至少主动拒绝以下攻击：

1. E4 ref 存在但 completion/inventory/hash 已漂移；
2. F0 尚未完成就生成或发布正式 F1–F3 图；
3. 从原始 evidence 临时重算出 recipe 未授权的新统计量；
4. C10 与 Stage D 池化，或用 Stage D 结果改写 C10；
5. 删除 misclassified、failed、NA、弱效果或成本超限记录；
6. 把 E4 package/workflow `PASS` 画成 scientific validity 全量 `PASS`；
7. 对 NNS 与 matched 使用误导性的不同色限；
8. 人工查看图片后替换冻结规则选出的案例；
9. 同一图命中多条规则时静默换图；
10. 将 display-smoothed/resized map 回流到 metric 或排序；
11. 重新加载模型、使用 GPU、重跑 attribution 或 fidelity；
12. 添加 CAM/Grad-CAM、其他模型或其他数据集的未验证结果；
13. 从参考文章复制图形内容或把其数值带入本论文；
14. PNG/PDF/SVG 来自不同数据或不同未记录配置；
15. draft/contact sheet 被当作正式 figure；
16. figure caption、正文数字与 machine-readable source 不一致；
17. 在 validator、manifest 或 terminal receipt 之前声明阶段完成；
18. 跳过 commit/push/ref 核验后宣称可复现封存完成。

任一攻击未被可靠拒绝时，对应阶段必须 `BLOCKED`，保留现场并输出诊断；不得自动换数据、换图或降低验收标准。

## 9. 强制图注与写作边界

规则选择的定性图必须包含等价声明：

> 案例图按 Stage D 结果访问前预注册的规则选择，仅作说明性展示；所有 primary 结论均基于完整 500 图队列。

Stage E 图必须包含等价声明：

> Stage E 是在 D10 结果已知后冻结的描述性敏感性分析；数值汇总基于 valid rows，并同时报告 expected、failed 和 NA。它不替换 Stage D primary，也不产生新的 GO/NO-GO。

C10/Stage D 对照图必须包含等价声明：

> C10 的 15 图 holdout 与 Stage D 的独立 500 图 cohort 分开报告且不池化；C10 的历史 `NO_GO_NOVELTY`、`FAIL_SPATIAL` 和 `BLOCKED` 状态保持不变。

固定审计案例必须标为 `fixed audit examples`，不得称为最佳、代表性或随机样本。任何逐案例色限图必须说明色限只支持案例内方法比较，不支持案例间绝对幅度比较。

## 10. F0 completion record（2026-09-07）

F0 最终在不可覆盖的 r4 root 完成：

- 输入绑定：E4 `7c82eca4e7de488f4537eff2b3fdb59feb3413ab`，completion SHA-256 `42ad2b483e40c164b4e26e1d8d65ecfc00703a2c01c2f0589961bcbefa8d58dc`，whole-root canonical digest `75249c74e08cb5038acbc86f71051260f9d9b7b6cc80d23c517e8669f3765f67`；D10 `7520689bb031304e4789567f99d093062aa50f72`。
- 实现：`src/vitnns/stage_f_renderer.py`、`src/vitnns/stage_f_cli.py`、`src/vitnns/stage_f_f0.py`；冻结 config/run spec/catalog/caption 和 9 个 Stage F schema。
- 正式输出：`evidence/stage_f/f0_r4_recovery_v2_20260907/formal/` 与 `reports/stage_f_f0_r4_20260907/`；whole-root canonical digest `28eab8d914cb062ade48e781985cc7055b6cbc703731d2086e323a0b47d5fdc8`。
- 测试：隔离环境 `CUDA_VISIBLE_DEVICES=''` 下 45 passed、0 failed；另有 13 个 Matplotlib/pyparsing deprecation warnings，不影响结果；`git diff --check` PASS。
- golden：3 套 synthetic-only fixtures，覆盖 single-column、double-column 和 full-width qualitative；每套均导出 600 DPI PNG、PDF 和 SVG，并通过重复渲染、字体嵌入、物理尺寸、跨格式内容 identity 与视觉 sanity 检查。正式论文 scientific figures rendered = `0`。
- false-pass audit：12/12 attacks 均 `REJECTED`。formal F0 producer 未加载模型、未导入 Torch/timm、CUDA 隐藏且无 NVIDIA device handle、未读取 JPEG；D10/E4 决策和统计未修改。
- 参考 PDF：仅用于布局密度、方法示意、成组热图、曲线和多面板组织审计；`style_reference_only=true`、`scientific_input=false`、`content_or_graphics_copied=false`、`reference_values_entered_results=false`。
- deviations：开发期误跑全仓测试，触发既有 Torch/CUDA 初始化后被中止；发生在 formal producer 初始化前，无模型/checkpoint/JPEG/scientific compute，已写入 r4 startup gate。r1 因单栏视觉拥挤、r2/r3 因 inventory validator 实现问题均 fail closed 并原样保留为非权威历史尝试；仅 r4 为 terminal PASS。
- Git/LFS：terminal delivery commit 记为 `SELF`；`stage-d-d3-pass` 已按交付合同推送，`refs/archive/stage-f-v1-f0-completion` 指向同一最终 commit，local/tracking/remote 一致且 worktree/index/LFS clean；这些外部状态由提交后的 final verification 核验。

F0 terminal receipt 只授权 `F1_ONLY`；F2、F3、F4 均未开始。

## 11. 当前下一步

当前只授权 `F3_ONLY`。F3 必须按冻结展示选择协议完成规则选例、方法示意与定性面板，并继续提供 report-side PNG review copies 和逐图 `figure_analysis.md`；不得启动 F4 或组合最终论文 Figure 1/5/S1/S2。

## 12. F1 completion record（2026-09-07）

F1 最终在不可覆盖的 r3 root 完成；r1/r2 为非权威失败尝试，均未产生 terminal PASS：

- 启动基线：当前分支 `stage-d-d3-pass`；F0 completion ref `refs/archive/stage-f-v1-f0-completion` 指向 `c4df92ff05054712406caa5524c5d129da7dab42`，且为 F1 renderer commit 的祖先；D10 completion ref `refs/archive/stage-d-fidelity-v1-d10-completion` 指向 `7520689bb031304e4789567f99d093062aa50f72`，且为当前 HEAD 的祖先。
- F0 绑定：completion SHA-256 `83b4ce1be0bf1c0d2ef49fe60a0793975d545944218451d0e4211e5e2261f7ce`；completion canonical digest `7e56bc714033ae9b62edaef04f2bd47b4d2ad9aa7592146fbd8e52aa71a896cc`；whole-root canonical digest `28eab8d914cb062ade48e781985cc7055b6cbc703731d2086e323a0b47d5fdc8`。
- D10 绑定：decision report SHA-256 `7eac27b11b9df72a91c6e78cd0d46ea30f7d8701f0d41cbc29e9cb76695b5a08`；canonical digest `db9b235cd8cfdeb3caf99445939443b17007ca5225c710c1164e36e2593503e6`；output-root inventory digest `b72d36430d149598ccd9cfd8dd586297e7231be29e61b44309e9971087e46535`；figure recipes SHA-256 `aaa1858e5a2b0e820d4ea5cb3072a41e3cb6f4f507daefa319dc9ab66a1f0ad1`。
- D10 paper-data hashes：`cohort_comparison.csv` `1f3974173c772f4bf676211656f742ec5f0440654a46a384520ad00cc4bad6df`，`cohort_comparison.json` `1bc23278d79e267e50c8eb198ba41ea1f22ab2447869524f1761dcde633ea23d`，`per_sample_delta.csv` `72cececb4386eafee1dc69ae98e251b9c3707c8d8325bb7f38cfb1c43db50a86`，`per_sample_delta.json` `8bb2e36dc2b4b3895bc5688b7ab5789719995add0c26cd48608d593486ba0cd2`，`curve_summary.csv` `4607d790c10e4671ae0d1e1339311c6cc9ac3c64ea80b7f2c31eb9f6f2e12904`，`curve_summary.json` `4c71cf24f6b02513a5ccd4441fd2f2c86af58307e0091a80bca8a52b6a4887ee`，`main_fidelity_table.csv` `3522b7e787dd53fa83757c804be9656a37b52a51c3d8e295bbe06268c1ad6a6b`，`main_fidelity_table.json` `03d0df490c6166acc08645d49ef39deaed23d66a67cf1b233f49675118e0748f`，`resource_and_qc.csv` `fd1834676229744da85a358860a07b1ad4d36224ec945909a0092e273c530396`，`resource_and_qc.json` `065deeb0250942f7e153af9d8f330f4df5471b8b2b1680945cc5faee98018504`。
- 实现：`config/stage_f_f1.json`、`config/stage_f_f1_run_spec.json`、F1 schemas、`src/vitnns/stage_f_f1.py`、`src/vitnns/stage_f_cli.py`、`src/vitnns/stage_f_renderer.py`、`tests/test_stage_f_f1.py`、`tests/test_stage_f_f1_boundary.py`。
- 正式输出：`evidence/stage_f/f1_r3_recovery_v2_20260907/formal/` 与 `reports/stage_f_f1_r3_20260907/`；whole-root canonical digest `a32e7998ed13810f1ccd4e2f6d7f6358711add0513cdbcb5438765ce445885f0`；正式 PNG/PDF/SVG 图像和 `reports/stage_f_f1_r3_20260907/figure_analysis.md` 均已纳入 Git 交付。
- 图形模块：F1-A cohort median + interval、F1-B Stage D 500 图 `delta_fid` ECDF、F1-C insertion/deletion median curves with pointwise bands；每个模块均有 `single_column` 和 `double_column`，共 6 个 render instances、18 个 PNG/PDF/SVG 导出文件；final combined Figure 2 rendered = `false`。
- 测试：`CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src /data01/home/yeziqi.831/nxtrec_wksp/testcode/.conda/envs/vitnns/bin/python -m pytest tests/test_stage_f_renderer.py tests/test_stage_f_boundary.py tests/test_stage_f_f0.py tests/test_stage_f_f1.py tests/test_stage_f_f1_boundary.py -q`，41 passed、0 failed；13 个 Matplotlib/pyparsing deprecation warnings。
- false-pass audit：18/18 attacks 均 `REJECTED`，覆盖 F0/D10 漂移、analysis/paper parity、C10/Stage D 池化、C10 状态弱化、重新 bootstrap/CI/new endpoint、ECDF 非 500、删除 misclassified、曲线错配、跨格式不同源、Stage E/JPEG/model/Torch/timm/CUDA 访问、F2/F3/F4/最终 Figure 2 提前生成、未完成 validation/manifest/terminal 先宣称 PASS、错误授权 F3/F4、未完成 Git/archive 核验即宣称完成。
- 科学边界：只读取 D10 `formal_r2/analysis/paper_data/` 与同一 analysis root 的 metadata；`formal_r2/paper/` 仅用于 byte/hash parity；未读取 Stage E paper data、reserve JPEG、legacy 图片、参考论文数据或人工抄录值；未使用模型、Torch、timm、GPU/CUDA；未重算 bootstrap、CI、`delta_fid` 或新 endpoint；C10 `n=15` 与 Stage D `n=500` 保持分 cohort。
- deviations：开发期 r1 因 F1-B caption 字段访问 bug fail closed；r2 因 startup gate 签名顺序 bug fail closed；二者均未发布 terminal PASS，未覆盖或删除，唯一有效 F1 terminal root 为 r3。
- Git/LFS：terminal completion 的 Git 字段为 `SELF`，交付后 `refs/archive/stage-f-v1-f1-completion` 必须指向同一最终提交；最终 local/tracking/remote/archive/LFS 状态由交付核验记录确认。

F1 terminal receipt 只授权 `F2_ONLY`；F2、F3、F4 均未开始。

## 13. F2 completion record（2026-09-07）

F2 最终在不可覆盖的 r4 root 完成；r1 因 pending receipt 缺少 renderer commit、r2/r3 因人工视觉 QC 的文字/边界裁切风险而 fail closed，均保留原始 pending 和 failure event，未产生 terminal PASS：

- 启动基线：分支 `stage-d-d3-pass`；F1 archive ref `refs/archive/stage-f-v1-f1-completion` 精确指向 `7a516571eabf364b7bb5d46ceb8aa74b450b3ae3` 且为 renderer commit 的祖先；E4 archive ref `refs/archive/stage-e-e4-completion` 精确指向 `7c82eca4e7de488f4537eff2b3fdb59feb3413ab`。
- F1 绑定：completion SHA-256 `079a646e7ab613d5eda3c99ed9f24d7b1635bfe31d53d5be422b5b5073cb2df3`；completion canonical digest `cba6d26364aea25a5f8276e96b6459e40742ada69af6beaccd4cdaa37dcaf005`；whole-root canonical digest `a32e7998ed13810f1ccd4e2f6d7f6358711add0513cdbcb5438765ce445885f0`。
- E4 绑定：completion SHA-256 `42ad2b483e40c164b4e26e1d8d65ecfc00703a2c01c2f0589961bcbefa8d58dc`；completion canonical digest `630423730c48c9ea4ae67d8b27b0ec690ee6c96f0ad36a1b3aa1312c3091a3a9`；whole-root canonical digest `75249c74e08cb5038acbc86f71051260f9d9b7b6cc80d23c517e8669f3765f67`；figure recipes SHA-256 `57306bc69e399d1ea2353d908ac0d8b22403d17ec29318a3eadf36b3b8d98a54`。
- E4 paper-data hashes：`patch_scale_sensitivity.csv/json` 为 `4082b21cae46e97666284c9091ad4e5ddaf9fa0900360e892b13fe0cb495515c` / `79deafe6516eeb13e2064f04c28d592d12a9db2a95fbfb14aff9de19ad925f50`；`upsampling_sensitivity.csv/json` 为 `9e24b978acbb1395febaaa38e2a1f62e76099311de501133f28de1307758ff9b` / `fd2b810727ae7e5cedb419fcd5cefbb13ee6a7e3303c624db2d9288ff47b7057`；`layer_head_descriptive.csv/json` 为 `1713b991a1a20706b72b7734794cabc12a3f0a0bd06993e930b24be4c7b7f93c` / `28d247a7f1b7f0450bca0da71713d3864cd257b141473aa60dc762f7bc9c28cc`；`prediction_strata.csv/json` 为 `4fa68255bf226ef4ffa5cdaa335aa9f970bec8151b9656439200f3b86c5037fd` / `86cd86631f676b2f6345d855c85c318a13ae1dbd2f1d8f796e55c40f04476a1b`；`resource_and_qc.csv/json` 为 `48cc2c7705542aa4b73928b47939638a65c487da2e6580f65a663f4486cf320d` / `30c3af9ec4c00edba5b16d8f0783fa5de4333d9a9174a06466f0206840c7205c`；五个 JSON canonical digests 依次为 `8cadb3c4894305de46cee8889de866b8fec2d1fad4f21bad64f6916a7d5a1832`、`38a79ec578b2a46ef2a3b99a4ad99096fa234f68f476a542c5cf1e9109e1f222`、`4ea4decbab836bdfe7eeb2c75e338ac94c39538a12d05bf93b6801681797f84c`、`c17e46153d43ef04ca553c71a7edf8afab1f964db7f6966655e76c5b65f47af7`、`32c5fd33621f55034f5f518d8fb69c5d6fb25439ef08d0fea0849f443bfc583e`。
- 实现：`config/stage_f_f2.json`、run spec 与 7 个 F2 schemas；`src/vitnns/stage_f_f2.py` source adapter/producer/validator；`src/vitnns/stage_f_cli.py`；复用并小幅扩展 `src/vitnns/stage_f_renderer.py`；`tests/test_stage_f_f2.py` 与 `tests/test_stage_f_f2_boundary.py`。
- 正式输出：`evidence/stage_f/f2_r4_recovery_v2_20260907/formal/` 与 `reports/stage_f_f2_r4_20260907/`；terminal completion SHA-256 `f4deb7d7a39714165b31d4978b8179e3418340de7b10ed53e0c091c93cedd31d`，canonical digest `1fa18250f104c7b66dbfd0c2dc1a1fb7bb978cb4940408f09bc4765c038deccc`，whole-root canonical digest `0f16684d2ed9fda74ac0f88406ff4987799c6b76e66112307f76cbe86964a405`。
- 图形模块：F2-A fine 4x4 anchor/native 14x14 patch-scale sensitivity；F2-B upsampling sensitivity 且 uniform/nearest 绑定同一 equivalent evidence class；F2-C 四个冻结 layer/head 坐标与 overall/correct/misclassified；F2-D expected/valid/failed/NA、typed failures 和 E2 waiver。每模块 single/double-column，共 4 modules、8 render instances、24 个 PNG/PDF/SVG；report-side review PNG 8 个；final combined Figure 3/4/S3 rendered = `false`。
- 人工分析：`reports/stage_f_f2_r4_20260907/figure_analysis.md` 按 Image #1..#8 逐图记录“能说明什么、不能说明什么、误读风险”；8 个 review PNG 位于同 report root 的 `figures_png/`。
- 测试：`CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src /data01/home/yeziqi.831/nxtrec_wksp/testcode/.conda/envs/vitnns/bin/python -m pytest tests/test_stage_f_renderer.py tests/test_stage_f_boundary.py tests/test_stage_f_f1.py tests/test_stage_f_f1_boundary.py tests/test_stage_f_f2.py tests/test_stage_f_f2_boundary.py -q`，terminal 前后均为 63 passed、0 failed；13 个 Matplotlib/pyparsing deprecation warnings。`git diff --check` PASS。
- false-pass audit：22/22 attacks 均 `REJECTED`，覆盖 F1/E4/ref/hash/inventory/manifest/paper-data 漂移、CSV/JSON parity、source-to-plotted exactness、uniform/nearest 拆分、stratum/coordinate 混淆、orders 117/352 删除或补零、E2 waiver/E4 MIXED 隐藏、p-value/winner/new decision/new endpoint、跨格式 identity、样式/画布、JPEG/model/Torch/timm/CUDA/NVIDIA、Stage E units/D8-D9 maps、F3/F4/final combined 提前生成及非原子 PASS。
- 科学边界：F2 仅消费 E4 formal `paper_data/`、指定 metadata 和 E0-E4 completion；没有模型、checkpoint、Torch、timm、GPU/CUDA、NVIDIA handle 或 JPEG 访问；没有 forward/backward/IG/attribution/fidelity；没有 p-value、winner selection、新 GO/NO-GO 或新 endpoint。orders 117/352 以完整 sample ID、failure code 和 missing marker 保留；E4 scientific validity 保持 `MIXED`；D10/F0/F1/E4 未修改。
- deviations：r1 在首张图渲染前因 pending 缺少 `renderer_commit` 键 fail closed；r2 完成渲染后发现 F2-D 双栏 failure-code 裁切；r3 完成渲染后发现 waiver 框贴近画布底边。三者均未发布 terminal PASS、未覆盖或删除，只有 r4 为权威 terminal PASS。
- Git/LFS：terminal completion 的 Git 字段为 `SELF`；最终交付 commit、push、`refs/archive/stage-f-v1-f2-completion` 及 local/tracking/remote/archive/worktree/LFS 状态由提交后的 final verification 记录确认。

F2 terminal receipt 只授权 `F3_ONLY`；F3 与 F4 均未开始。
