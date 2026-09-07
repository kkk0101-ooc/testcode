# 阶段 F：正式成图、定性展示与论文交付

> 建立日期：2026-09-07。
> 当前状态：Stage F F0–F3、post-F3 blue-display r4、F3H 和 F3C0 已完成并封存；F4 为 `PAUSED_PENDING_F3C`。当前唯一下一授权是 `F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY`，不得启动 F3C2 pilot、F3C3 formal、F3C4 outcome analysis 或 F4。
> 文档性质：Stage F 的权威规划、阶段边界和验收合同；不是一次性跳过阶段门禁的总授权。
> 选择规则：定性案例继续受 [阶段 F 展示选择协议 v1](stage_f_display_protocol_v1.md) 约束；跨层/头归因矩阵同时受 [Stage F 跨层/头归因展示补充协议](stage_f_cross_layer_display_addendum_20260907.md) 约束。

## 1. 目标

Stage F 把已经封存的 D10 主结果和 E4 消融结果转化为可投稿、可复现、可审计的论文图和论文交付包。目标视觉形态参考 [参考文章.pdf](../../论文1/参考文章.pdf)：包含方法流程示意、成组热图、统计曲线、消融对比和多方法定性面板，而不是只输出孤立的默认折线图。

“参考”只指图形组织密度、清晰度和投稿完成度。Stage F 不复制参考文章的图片、数据、文字、网络结构或实验结论；所有本论文图形必须来自本项目已验证的输入，或来自不承载经验结果的方法定义。

原始 F0–F3 只把已封存的 D10/E4 结果转化为论文模块，不产生新的科学结论。owner 后续前瞻性插入的 F3H 仍为描述性阶段；F3C 则是独立冻结的 class-targeted empirical study，只能产生其协议明确允许的 conditional class-logit fidelity 结果，不得追溯改写 C10、Stage D、Stage E 或 F0–F3。Stage F 必须同时满足：

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
- E3 最终验证的 secondary layer/head typed-terminal map union：`e3_recovery_v2_20260906`、`e3_r1_recovery_v2_20260906`、`e3_r2_recovery_v2_20260906` 和 `e3_r4_recovery_v2_20260907` 四个 formal roots；
- E3-R4 union tables 与 E4 sealed `layer_head_descriptive` copy，用于把 3,000 个逻辑单元唯一解析为 2,998 个 valid maps 和 orders `117`、`352` 两个 typed failures；
- D1 五个 audit IDs 对应的封存审计证据；
- 被规则选中的原始图像：`assets/data/ILSVRC2012_val/`；
- `docs/stage_f_display_protocol_v1.md`；
- `docs/stage_f_cross_layer_display_addendum_20260907.md`。

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
| Figure 6：跨层/头归因矩阵 | F3/F4 | 同一 `median_effect` 样本的 Original，以及 `(0,9)`、`(4,2)`、`(6,10)`、`(11,2)` 上的 NNS/matched attribution 矩阵 | Fig. 11/13 风格；不是 raw attention；坐标内共享色限，跨坐标只比较空间模式 |
| Supplement S1 | F3/F4 | 五个 D1 固定审计案例 | 标为 fixed audit，不得称为 best examples |
| Supplement S2 | F3/F4 | 规则到样本映射、重复选择、weak/failure、misclassified/difficult 案例 | 不得静默替换重复样本 |
| Supplement S3 | F1/F2/F4 | resource/QC、失败/NA、E2 budget waiver 与 E3 mixed-validity disclosure | 与科学图分开但必须随论文包交付 |
| Supplement S4 | F3/F4 | 5 个 fixed audit IDs、5 个既有规则 slots 及 orders 117/352 failure slots 的跨层/头 NNS/matched attribution matrices | 按冻结顺序生成；重复样本可去重显示但 slot mapping 必须保留；failure/NA 原位显示 |

Figure 1 的方法示意可以使用确定性声明式布局、TikZ 或等价矢量后端；不得用无法重建的手工截图作为唯一源。Figure 5 和 Figure 6 当前只比较 NNS 与 frozen matched baseline，不默认添加 CAM/Grad-CAM。Figure 6/S4 展示的是 fixed-clean-attention head-direction Value-path attribution，不是 raw self-attention，也不支持跨坐标绝对幅度比较；完整合同见 [跨层/头展示补充协议](stage_f_cross_layer_display_addendum_20260907.md)。

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
5. Figure 6 主样本固定复用 `median_effect` slot；不得查看 E3 maps 后另选“跨层变化更明显”的样本。
6. Supplement S4 固定使用 D1 五个 audit IDs、五个既有规则 slots，以及 orders `117`、`352` 两个 failure slots；重复命中保留 slot-to-image mapping，渲染可去重但不得换样本。

**必须生成**

1. Figure 1 方法总览的矢量源和导出版本；
2. 每个规则案例的 Original / NNS / matched / difference / overlays / curves 模块；
3. 五个固定审计案例的 supplementary 模块；
4. Figure 6：`median_effect` 样本在 `(0,9)`、`(4,2)`、`(6,10)`、`(11,2)` 上的 NNS/matched attribution progression matrix；
5. Supplement S4：五个 fixed audit IDs、五个既有规则 slots 与 orders `117`、`352` failure slots 的完整跨层/头矩阵；
6. E3 四轮 map union manifest、rule-to-image manifest、caption data、每个源图/map/curve 和 render 的 hash；
7. 可供人工审阅的 contact sheet，但 contact sheet 不是正式论文源。

**显示约束**

- 统一预处理画幅，不得为单个案例手工裁剪目标区域；
- NNS 与 matched 同案例共享色限，difference 独立对称色限；
- 跨层矩阵中同一样本、同一坐标的 NNS/matched 共享色限；不同坐标对应不同 target，颜色只比较空间 pattern，不比较绝对 magnitude；
- 展示平滑、resize 和 alpha 只影响 overlay，并必须标为 display-only；
- 弱效果、误分类、失败或重复选择不得被“更好看”的替代案例替换。
- orders `117`、`352` 或其他 failed/NA cell 必须原位显示状态和 failure code，不得补零、复制相邻坐标或换样本。

**验收与授权**：validator 必须从 D10 表独立重建选择结果，核对 D9/D1/JPEG/map identities，并从 E3/E3-R1/E3-R2/E3-R4 四个 roots 独立重建 3,000 个 typed-terminal union、2,998 个 valid maps 与 2 个原位 failures；同时证明 model/GPU 未使用。completion 只能授权 `F4_ONLY`。

**预计耗时**：1.5–3 个工作日；其中新增跨层/头 source-union 验证与 Figure 6/S4 约增加 0.5–1 个工作日。

### F4：组合图、图注、论文写作与最终导出

F4 只能消费已经通过验收的 F1–F3 模块。F4 可以调整 panel 布局、留白、标题位置和文字换行，但不能更改 panel 数据、色限、样本、选择规则或状态。

**必须完成**

1. 按第 5 节形成 Figure 1–6 和 Supplement S1–S4；
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
19. 跨层图只读取 E3-R4，遗漏 E3/E3-R1/E3-R2 的历史有效 union；
20. 把 Value-path attribution progression 写成 raw attention、完整 attention/QKV 或 class-logit attribution；
21. 根据跨层 map 外观选择样本，或对 failed/NA cell 换样本、补零、插值；
22. 使用不同坐标的颜色强度形成未经支持的层间绝对 magnitude 结论。

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

## 11. F3 后的历史调度记录

F3 terminal receipt 历史上授权 `F4_ONLY`，但 owner 在 2026-09-07 以前瞻性协议暂停 F4，并依次插入 F3H full-head/Grad-CAM 与 F3C class-targeted study。本段只记录当时的调度转换；当前权威状态以第 21–23 节为准：F3H 与 F3C0 已完成，F4 为 `PAUSED_PENDING_F3C`，唯一下一授权为 `F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY`。F4 恢复后只能消费已经通过验收的 F1–F3/F3H/F3C 模块，完成最终组合图、图注、论文写作和反向 source audit；不得改变冻结样本、panel 数据、failure/NA、selection mapping、method target 或统计结果。

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

## 14. F3 completion record（2026-09-07）

F3 最终在不可覆盖的 r7 root 完成；r1 因 E3 两张表误作整对象比较、r2 因 render report 的 NumPy float 序列化、r3–r6 因人工/owner 视觉 QC 分别 fail closed。所有失败尝试均保留 pending receipt、已生成 payload 和结构化 failure event，没有覆盖、删除或伪造 PASS；唯一有效 terminal root 为 r7：

- 启动和选样：F2 archive ref 精确指向 `89dee5ad25b9cf3baa3a2ffdb2261ec8343a2ef6`；F2 completion SHA-256 为 `f4deb7d7a39714165b31d4978b8179e3418340de7b10ed53e0c091c93cedd31d`。selection 和 rule mapping 在任何 JPEG/map access 前原子封存并独立重建，`jpeg_or_map_access_before_seal=false`。
- 规则结果：best order 372、median order 44、baseline-strong order 372、weak order 281、misclassified/difficult order 387。order 372 重复命中被完整保留，5 slots 去重为 4 个样本，没有静默替换。
- Figure 6：唯一主样本为 `ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG`，order 44，严格来自 D10 `median_effect`。
- source union：D8 primary 1,000 maps 全量验证；E3 union 为 3,000 typed terminals、2,998 valid maps、2 failures。有效 origins 为 516/12/636/1,834。orders 117/352 的 `(11,2)` NNS failures 原位保留，matched 和其他有效坐标正常显示。
- 模块：Figure 1；Figure 5 四个去重规则案例；S1 五个 fixed audit examples；S2 完整 mapping 和五个 slots；Figure 6；S4 12 slots / 11 unique samples。共 28 render instances、84 个 PNG/PDF/SVG、28 review PNG；final combined figures rendered = false。
- 显示合同：同例同坐标 NNS/matched 共享联合对称色限，difference 独立，不跨坐标比较 magnitude，无 clipping。r7 对两方法对称叠加各自 signed extrema 50%/80% 的 display-only local contours，使 matched 形态可读；contours 不进入 metric、selection 或跨方法 magnitude 比较。
- 人工检查：`reports/stage_f_f3_r7_20260907/figure_analysis.md` 按 Image #1..#28 逐图记录全部视觉检查项；28/28 PASS。
- 验证：21/21 false-pass checks PASS；79 applicable Stage F tests passed、0 failed、2 deselected。两个 deselected 是 F2 历史生命周期断言（仅在 F2 时要求 F3 root 不存在）；F0/F1/F2 completion validators 另行 PASS；`git diff --check` PASS。
- 科学边界：没有模型/checkpoint、Torch、timm、GPU/CUDA、forward/backward/IG/attribution/fidelity；没有新统计、p-value、winner、endpoint 或 GO/NO-GO；上游 decisions/completions 未修改。
- Grad-CAM/Figure 7：没有生成。F3 冻结协议禁止 CAM/Grad-CAM、模型加载和新 attribution，且没有 sealed Grad-CAM input；若需要该比较，必须另立前瞻性计算协议。
- 正式输出：`evidence/stage_f/f3_r7_recovery_v2_20260907/formal/` 与 `reports/stage_f_f3_r7_20260907/`。最终 commit/push、archive ref 和 Git/LFS 状态由提交后核验。

F3 terminal receipt 只授权 `F4_ONLY`；F4 尚未开始。

## 15. Post-F3 blue-display visual revision completion record（2026-09-07）

Owner review rejected the sealed F3 r7 white-background presentation because the hotspot structure was visually weak, matched attribution was difficult to inspect, and earlier revision attempts had title/row-label layout defects. The scientific F3 r7 package and archive ref remain unchanged; the final display-only replacement is the independently sealed r4 root. Visual r1–r3 remain immutable `BLOCKED` attempts with failure events and no terminal PASS.

- Style contract: valid signed maps are shown as `abs(raw_signed_map)`, normalized by that panel's own maximum, squared (`gamma=2`), and bilinearly resized from 56x56 to 224x224. The fixed palette runs from deep blue through cyan/yellow/orange to red. There is no percentile clipping, smoothing, max pooling, or source-map modification.
- Interpretation: r4 is a shape-only presentation view. Color magnitude cannot be compared between methods, coordinates, or samples. F3 r7 signed maps with shared symmetric scales remain the authoritative magnitude audit.
- Layout correction: case modules use separate title and panel bands plus a compact fidelity legend. Cross-layer matrices use a dedicated header row and a dedicated horizontal row-label column (`NNS`, `Matched patch`); no row label is rotated, clipped, or attached outside a heatmap axis. Typed-failure titles no longer repeat the order.
- Method label: `Matched patch` is the frozen `matched_fixed_clean_attention_patch_token_contribution` baseline for the same fixed-clean-attention head-direction Value-path scalar. It is not Grad-CAM.
- Outputs: `evidence/stage_f/f3_visual_r4_recovery_v2_20260907/formal/` and `reports/stage_f_f3_visual_r4_20260907/`; 27 render instances, 81 PNG/PDF/SVG exports, and 27 report-side PNG review copies. Module counts are Figure 5 = 4, Figure 6 = 1, S1 = 5, S2 = 6, and S4 = 11.
- Figure 6: unchanged frozen median-effect sample `ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG`, order 44. S4 remains 12 slots / 11 unique samples; order 117 and 352 `(11,2)` NNS failures remain in place while matched and all other valid cells are shown.
- Validation: all 27 review copies passed manual visual QC; 9 focused tests passed with 0 failures; all 16 false-pass attacks were rejected. Terminal completion SHA-256 is `0231b38a6886ab8e3b46b56e1d93928490fbe4dc3a858aa0c92781826177547d`; whole-root canonical digest is `26aaa04bef8df84c83d33198ccb2d51f8b4731a711cfbfd45273110d63778e30`.
- Boundaries: no model/checkpoint, Torch, timm, CUDA/GPU, forward/backward, attribution, or fidelity execution; no source, selection, slot, statistic, decision, or F3 archive modification; no Grad-CAM/Figure 7 was fabricated because no sealed Grad-CAM source exists; F4 was not started.
- Delivery: terminal commit is recorded as `SELF`; the independent archive ref is `refs/archive/stage-f-v1-f3-blue-display-revision`. Final local/tracking/remote/ref/worktree/LFS checks are performed after the delivery commit.

截至 blue-display r4 封存时，下一授权原为 `F4_ONLY`；下述后续 owner 授权以前瞻性 F3H 插入阶段暂停该任务。

## 16. F3H prospective insertion（2026-09-07）

Owner 在 F3 与 blue-display r4 封存后要求先建立新的前瞻性补充计算协议，内容为同一 `median_effect` 样本的全 12×12 head atlas、NNS/matched 正负贡献拆分、pair-shared monotonic log 显示和一个 sealed Grad-CAM class-logit 参照；F4 在 F3H terminal completion 前暂停。完整合同见 [F3H full-head/Grad-CAM 协议](stage_f_f3h_full_head_gradcam_protocol_20260907.md)。

F3H 保持 F3 r7 和 blue-display r4 不变：

- 唯一样本仍为 order 44 / `ILSVRC2012_val_00002287.JPEG`，不得根据新 atlas 或 Grad-CAM 外观换样本；
- 12 blocks × 12 heads 全部保留，不选择 best/representative heads；
- 144 coordinates × 2 Value-path methods = 288 terminals；四个已有坐标的 8 个 valid maps 严格继承，其余 280 units 才允许新计算；
- NNS/matched 同一坐标共享一个 `M=max(max_abs_nns,max_abs_matched)` 与 `log1p(999*x/M)/log(1000)` 变换，不允许 method-specific gain、gamma、alpha 或分别自动缩放；
- raw signed map 拆为 positive 与 negative magnitude，display-only 数组不得回流到 metric；
- Grad-CAM 固定为 `blocks[11].norm1`、clean predicted-class raw logit、class index 713，仅一个 terminal unit；它与 Value-path 是不同 target，只作定性参照；
- 不运行 fidelity、stability、sanity、统计、p-value、winner 或新决策；
- 保守预算 33.5 H20-GPU-hours，硬上限 40.0，无自动 waiver。

F3H 原子里程碑为 F3H0 protocol freeze、F3H1 implementation/synthetic preflight、F3H2 formal GPU compute、F3H3 CPU validation/render/seal。F3H0 不加载模型、不解码 JPEG、不导入 Torch/timm、不访问 CUDA/GPU、不运行 attribution。F3H0 PASS 只能授权 `F3H1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY`，不能直接授权正式 GPU 运行，也不能恢复 F4。

## 17. F3H0 protocol-freeze completion record（2026-09-07）

F3H0 已在独立 protocol-only root `evidence/stage_f/f3h0_protocol_v1_20260907/formal/` 完成，状态 `PASS`：

- protocol ID：`STAGE_F_F3H_FULL_HEAD_GRADCAM_V1_20260907`；配置 canonical digest `e5ac9692663797a6a0f217151deedb9d61442902c2128a5a7a53483b1b72936e`；
- 样本固定为 order 44 / `ILSVRC2012_val_00002287.JPEG`，image SHA-256 `c597ef7e061560542edbd77d2cbca6107cc44ff5e86de487a015eb47ae4620a8`，clean input digest `f901fab0e43fbbc725aff3157ef7a91604cb30b4fdd92d2cb2f1aea49f092484`，Grad-CAM class index `713`；
- 模型固定为 recovery-v2 `489bfa2f...` / canonical tensor digest `150234d...`，历史 `2e718...` 禁用；
- workload 固定为 144 coordinates、288 Value-path terminals，其中 8 inherited / 280 new，另有 exactly one Grad-CAM terminal；
- pair-shared log 变换固定 `M=max(max_abs_nns,max_abs_matched)` 与 `kappa=999`，正负贡献分开；禁止 method-specific brightness/gamma/alpha 和 best-head selection；
- 预算保守投影 33.4865 H20-GPU-hours，硬上限 40.0，无自动 waiver；
- 预期 4 个 full atlases、12 个 block paired pages、1 个 Grad-CAM reference，共 17 renders / 17 review copies；
- false-pass audit 20/20 attacks 均 `REJECTED`；F3H0/blue-display 联合回归 27 passed、0 failed；`git diff --check` PASS；
- F3H0 未解码 JPEG、未反序列化或加载模型、未导入 Torch/timm、未访问 CUDA/GPU、未运行 attribution；只流式读取 checkpoint 字节用于 SHA-256 身份核验；
- completion SHA-256 `615e668e223153f73ebe5fde732832b4f5c26024ddefae0bf14fe5cf09d2ac4a`，terminal canonical digest `32c05c18879a926195e99fd61045a86867928c0392b35b278ea2c2dbe0b0de61`。

F4 保持 `PAUSED_PENDING_F3H`。当前唯一下一授权为 `F3H1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY`；正式 GPU 计算尚未授权、尚未开始。

在 F3H0 封存后、任何 F3H pilot/outcome access 前，owner 进一步要求：F3H 完成后先冻结真正的 class-targeted study（F3C），并将所有候选参照纳入。F3H1 前瞻性 implementation addendum 因此补充冻结：

- F3H 增加 per-layer、四个固定 depth bands 和 all-144-head 的 `equal-head-weighted spatial-support consensus`，但它们不是 raw attribution sum、网络级 attribution 或 class-conditioned attribution；
- consensus 使用所有必需 heads 的等权 display-space 平均；任何所需 head FAILED/NA 时对应 consensus 原位标失败，不做 complete-case 平均；
- atlas/consensus 外观、Grad-CAM overlap、亮度或 entropy 均不得定义 class weights；
- 只有存在全部 144 个 signed class-logit coupling coefficients、无遗漏/重复计数的 path accounting identity 和预注册 conservation test，才足以冻结具体数值聚合；F3H 不计算这些 coupling，预期充分性结论为 `INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS`；
- F3C 的冻结候选比较集必须全部包含：class-targeted NNS、ViT Grad-CAM、full-model class-logit IG、attention rollout、block-11 raw CLS attention head mean；后两者继续标为 class-agnostic；
- F3H render/review 数由 17 前瞻性扩展为 20：原 17 个加 per-layer、depth-band、all-head consensus 三个模块。

因此当前 F4 状态进一步收紧为 `PAUSED_PENDING_F3H_AND_F3C`。F3H terminal PASS 只能授权 `F3C0_PROTOCOL_FREEZE_ONLY`，不能直接启动 F4。

## 18. F3H1 implementation completion record（2026-09-07）

F3H1 已在不创建 F3H formal GPU root 的条件下完成：

- 实现 `src/vitnns/stage_f_f3h.py`，复用既有 recovery-v2 model/sample loader、full-head direction resolver、E3 完整 IG schedule、matched token contribution 和 Stage C Grad-CAM；
- 正式 Value-path runner 是 map-only，不调用 fidelity curves、perturbation、stability、sanity、bootstrap、统计或 winner selection；
- 288-unit block-major plan source-recomputable；8 个 sealed inherited terminals 不进入 recompute，280 个新 units 按 coordinate 整对分配到 8 shards；
- 原子 terminal persistence 覆盖 `VALID`/`FAILED`/`NA`，未知实现异常保持 incomplete 并阻断；
- 显示实现覆盖 signed positive/negative 拆分、pair-shared `kappa=999` log transform、四张 12×12 atlas、十二张 block paired pages、Grad-CAM reference，以及新增的 per-layer/depth-band/all-head equal-weight spatial-support consensus，共 20 个 render instances；
- consensus 在任何所需 pair 失败时原位失败，不做 complete-case 平均，不产生 class-conditioned 权重；
- post-run sufficiency 判据预先固定为必须存在 144 个 signed class-logit coupling coefficients、无遗漏/重复计数的 path accounting identity 和 conservation test；atlas 外观不得参与权重定义；
- F3C 未来候选比较集冻结为 class-targeted NNS、ViT Grad-CAM、full-model class-logit IG、attention rollout、block-11 raw CLS attention head mean；
- 44 tests passed、0 failed；implementation false-pass 12/12 rejected；未加载模型、未解码 JPEG、未导入/使用 Torch/timm/CUDA、未运行 attribution；
- F3H1 completion canonical digest `562fc67ef48e5876695ecfec7209e3ec98f933944b5a005585c0dff0f30f3e58`。

F3H1 PASS 只授权 `F3H2_PILOT_ONLY`。F4 继续为 `PAUSED_PENDING_F3H_AND_F3C`。

## 19. F3H v1 启动失败与 R1 前瞻恢复（2026-09-07）

F3H v1 formal root 初始化后，首次四坐标 pilot 在模型反序列化、JPEG 解码和 attribution 开始前 fail closed。通用 recovery-v2 loader 重新核验 `config/recovery_v2_binding.json` 中的原始 amendment SHA，但对应 amendment 后来发生了文档性编辑，因而报 `Recovery amendment SHA-256 drifted`。该失败不是 scientific invalidity；v1 root 保留 pending receipt，Value-path/Grad-CAM terminal 数均为 0，且没有 terminal PASS。

新的 [F3H-R1 启动恢复补充协议](stage_f_f3h_r1_startup_recovery_addendum_20260907.md) 在任何 F3H outcome access 前冻结：

- 不修改旧 binding、amendment、F3H0/F3H1 completion、F3H v1 failure root 或任何 sealed 上游 evidence；
- R1 继续使用同一 recovery-v2 checkpoint 文件、架构、canonical tensor digest、order 44 JPEG、预处理和 clean-tensor digest；
- 只将启动 adapter 改为对这些 F3H 已冻结身份逐项 direct-pin，不再调用失效的 Stage-A/C7 amendment loader；
- R1 写入新的 `f3h_full_head_r1_recovery_v2_20260907` evidence root 与 `stage_f_f3h_full_head_r1_20260907` report root；
- 原 `refs/archive/stage-f-v1-f3h1-implementation` 必须保持指向 `9ea95a6b23979a1d2da1d83658f7353ac0096ad9`；R1 producer 使用独立 ref `refs/archive/stage-f-v1-f3h-r1-producer`；
- R1 仍只先执行冻结的四坐标 pilot；通过资源和完整 typed-terminal 门禁后才允许 bulk 与 sealed Grad-CAM；
- 完整 12×12 atlas 完成后才依据预声明判据评估 head 信息是否足以定义 class-conditioned 聚合。不得凭视觉效果指定权重。

F3C0 只能在 F3H terminal 后冻结，候选集必须全部包含 class-targeted NNS、ViT Grad-CAM、full-model class-logit IG、attention rollout 和 block-11 raw CLS attention head mean。F3C 计算和 F4 均未开始，F4 状态保持 `PAUSED_PENDING_F3H_AND_F3C`。

## 20. F3H-R1 compute completion 与 R2 CPU reconciliation recovery（2026-09-07）

F3H-R1 按冻结顺序完成正式计算：四坐标 pilot 为 8/8 VALID、0 FAILED、0 NA，pilot telemetry 为 0.009995 H20-GPU-hour，按冻结公式投影总成本 0.599830 H20-GPU-hour，低于 40 小时硬上限；随后 8 个 pair-preserving shards 完成其余 272/272 VALID units。连同四个 sealed inherited pairs，完整 atlas 输入为 288/288 valid Value-path terminals。冻结的 class-713 / `blocks[11].norm1` Grad-CAM 也为 1/1 VALID。

首次 CPU reconciliation 在写 `terminal_union.json` 前 fail closed：实现错误地假定所有 inherited result 都有顶层 `target_identity`。D8 两个 terminals 确有该字段；E3 六个 terminals 则使用 `target_response.target_identity`，并在 `source_hashes.target_semantic_digest` 与 `pair_binding.target_semantic_digest` 中双重封存同一 digest。R1 的 280 个新 maps、Grad-CAM、9 份 telemetry 和 pending receipt 全部保留；没有重跑或覆盖。

前瞻性 [F3H-R2 CPU reconciliation recovery](stage_f_f3h_r2_cpu_reconciliation_addendum_20260907.md) 因此冻结：

- R1 compute root 由 1,139-file inventory 整体绑定；R2 只引用其 maps 和 telemetry，不复制、不修改、不重新计算；
- D8 adapter 从顶层 `target_identity` 计算 semantic digest；E3 adapter 必须验证 `source_hashes`、`pair_binding` 和 `target_response.target_identity` 三方 digest 完全一致；
- R2 的 pilot、bulk shard 和 Grad-CAM 入口均 fail closed；模型、Torch/timm、CUDA/GPU 和 attribution 访问为零；
- R1 telemetry 必须恰好 9 份且只计一次；Grad-CAM CLI 曾在参数解析阶段拒绝 `--device`，未执行计算，随后通过同一 sealed Python API 与冻结参数完成唯一 Grad-CAM unit；
- R2 使用新 evidence/report root，终态仍只能授权 `F3C0_PROTOCOL_FREEZE_ONLY`。

F4 继续 `PAUSED_PENDING_F3H_AND_F3C`。

## 21. F3H terminal completion（2026-09-07）

F3H 最终通过 R3 atomic sealing wrapper 完成，状态 `PASS`。R1 完成全部正式计算但在 CPU reconciliation 的跨代 schema adapter 处 fail closed；R2 完成 reconciliation、渲染、验证和人工 QC，但 finalizer 对已签名 validation report 二次签名，因而在 pending→terminal 替换前 fail closed。R1/R2 均作为不可覆盖 blocked attempts 保留；R3 不复制或重渲染 R2 的 1.08 GB payload，只逐文件复核并发布小型 terminal receipts。

- 固定样本：order 44 / `ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG`，即既有 D10 `median_effect`；没有按 atlas 或 Grad-CAM 外观换样本。
- Value-path union：144 coordinates × 2 methods = 288 terminals；8 个 sealed inherited + 280 个 R1 new；288 VALID、0 FAILED、0 NA。
- Grad-CAM：1/1 VALID，class 713、`blocks[11].norm1`、clean predicted-class raw logit；仅为不同 target 的定性参照。
- 显示：576 个 signed components、144 个 pair-shared monotonic-log scale records；无 method-specific gain/gamma/alpha，无 clipping，无 best-head selection。
- 模块：4 个完整 12×12 atlas、12 个 block paired pages、1 个 Grad-CAM reference、3 个 equal-head spatial-support consensus，共 20 renders、60 PNG/PDF/SVG exports、20 report-side review copies。
- 人工 QC：20/20 PASS；没有标题/行标错位或裁切。matched positive 在共同尺度下可见；matched negative 多数接近深蓝来自原始相对幅度，而非人为调暗。
- 资源：R1 9 份 telemetry 只计一次，加 Grad-CAM 后总计 0.188175 H20-GPU-hour，低于 40 小时硬上限，无 waiver。
- 测试：最终 separate-process 回归 64 passed、0 failed（55 F3/F3H/R3 + 9 sealed visual）；R3 专项验证另覆盖 payload/inventory、atomic completion 与 no-model/no-GPU 边界。
- false-pass：F3H 27/27 attacks 和 R3 12/12 sealing attacks 均拒绝。
- 充分性结论：`INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS`。完整 atlas 记录了 head/depth 空间支持多样性，但没有全部 144 个 signed class-logit coupling coefficients、无遗漏/重复计数的 path-accounting identity 或预注册 class-logit conservation test，因此不得从外观、亮度、Grad-CAM overlap 或 entropy 指定 head 权重。
- 科学边界：F3H 没有新增 fidelity、stability、p-value、winner 或 GO/NO-GO；F3 r7、blue-display r4、D10、E4 及其 refs 未修改。R2/R3 recovery 没有模型、Torch/timm、CUDA/GPU 或 attribution 重算。
- 正式 payload：`evidence/stage_f/f3h_full_head_r2_cpu_reconcile_20260907/formal/` 与 `reports/stage_f_f3h_full_head_r2_cpu_reconcile_20260907/`；terminal receipts：`evidence/stage_f/f3h_full_head_r3_terminal_seal_20260907/formal/` 与 `reports/stage_f_f3h_full_head_r3_terminal_seal_20260907/`。

F3H terminal PASS 只授权 `F3C0_PROTOCOL_FREEZE_ONLY`。F3C empirical compute 和 F4 均未开始。

## 22. F3C0 prospective protocol-freeze completion（2026-09-07）

F3C0 已在完整 F3H atlas 和预声明充分性评估之后冻结，状态 `PASS`。本里程碑只冻结方法、样本、指标、预算和执行顺序，没有启动 F3C empirical computation，也没有开始 F4。

- F3H 输入：`refs/archive/stage-f-v1-f3h-completion` 精确指向 `56afedd2e73a760dc5127ab4ee9c38d72623dbbc`；F3H completion 为 PASS、288/288 valid、Grad-CAM VALID，且 `next_authorization=F3C0_PROTOCOL_FREEZE_ONLY`。
- 决策依据：F3H basis assessment 为 `INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS`。因此 F3C 不对 144 张 F3H head-direction maps 猜测、学习或视觉选择权重。
- 新方法：`conditional_all_blocks_fixed_clean_attention_class_logit_ig`，论文标签为 `class-targeted NNS (all-block fixed-clean-attention class-logit IG)`。它在一次 clean forward 中冻结 12 层完整 multi-head attention tensors，并让模型自身的 Value、head concatenation、output projection、residual、MLP、final norm 和 classifier head 将输入连接到 frozen predicted-class raw logit；它不是 F3H maps 的加权和，也不等同于 unmodified full-model IG。
- 五方法集合完整冻结：class-targeted NNS、ViT Grad-CAM、full-model class-logit IG、attention rollout、block-11 raw CLS attention head mean；后两者保持 class-agnostic 标签。
- 样本：从 D1 500 图 registry 通过 `sha256("STAGE_F_F3C_V1_20260907|" + sample_id)` 可复算排序，排除所有已在 F3/S4/F3H 展示或作为 failure slot 的 11 个 orders；冻结 5 张 pilot 和 50 张 formal cohort，二者互斥。pilot orders 为 `416, 350, 438, 411, 337`；order 44 只作已查看 continuity anchor，不进入 pilot、formal 或统计。
- 评价：五方法共享一次冻结的 clean predicted-class raw-logit index；14×14 patch fidelity 为 primary common resolution，formal statistical unit 为 image (`n=50`)。NNS vs Grad-CAM 是单一 primary contrast，其余三项为预声明 secondary Holm family；F3C1 必须在 pilot outcome 前再冻结精确检验算法。
- 显示：每样本五方法使用 target-oriented positive mass-one view 和共同 unclipped 色限；禁止 method-specific gain/gamma/alpha、percentile clipping 和分别自动缩放。signed NNS/full-model IG 另保留正负 audit，非负方法不得伪造 negative map。
- 预算：5-image pilot 后必须重估；provisional total cap 8 H20-GPU-hours、每 image group 0.30、最多 8 devices、每进程 16 GiB reserved、retained artifacts 8 GiB，无自动 waiver。
- 流程：F3C0 → F3C1 implementation/synthetic preflight → F3C2 five-image pilot → F3C3 50-image formal → F3C4 CPU analysis/seal。任何 pilot map/JPEG 外观不得进入 continuation 决策。
- F3C0 tests：12 passed、0 failed；20/20 false-pass attacks rejected。F3C0 未读取 JPEG、未加载模型、未导入 Torch/timm、未访问 CUDA/GPU、未运行 attribution/fidelity。
- 正式协议与合同：`docs/stage_f_f3c_class_targeted_protocol_20260907.md`、`config/stage_f_f3c.json`、`config/stage_f_f3c.schema.json`、`manifests/stage_f_f3c_v1/sample_split.json`。

F3C0 terminal PASS 的唯一下一授权是 `F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY`。F3C pilot/formal computation 尚未开始；F4 保持 `PAUSED_PENDING_F3C`。

## 23. F3C0 后的 head 证据解释与执行总路线（2026-09-07）

### 23.1 现有 head 信息能回答什么、不能回答什么

F3H 的完整 12×12 atlas 不是“没有 head 信息”。它已经证明同一冻结样本内不同 block/head 的 fixed-clean-attention head-direction Value-path attribution 具有可见的空间模式和深度差异，并完整保留 NNS/matched 的正负贡献、pair-shared monotonic-log 显示以及不同 target 的 sealed Grad-CAM 参照。该结果可用于描述 head/depth diversity，也可用于解释为什么单一 head 不能代表整个模型。

但 F3H atlas 不能单独定义 144 个数值化 class-conditioned head 权重，因为它缺少以下三项同时成立的证据：

1. 全部 144 个 head 的 signed class-logit coupling coefficients；
2. 覆盖 head concatenation、output projection、residual、MLP、后续 blocks、final norm 和 classifier，且无遗漏、无重复计数的 path-accounting identity；
3. 在任何 outcome access 前冻结、能够验证聚合 attribution 与目标 class-logit 差值一致的 conservation test。

因此不得从 atlas 外观、亮度、entropy、Grad-CAM overlap、物体重合程度或“最好看 head”推断权重；也不得把 144 张 F3H map 的平均、加权平均或筛选子集称为 class-conditioned attribution。order 44 只是一张已冻结的描述性样本，不足以推出跨样本或跨类别通用权重。该边界不否定 head atlas 的描述价值，只否定未经 class-logit 绑定的数值归因解释。

### 23.2 已冻结的解决路径

F3C 不再尝试学习、调参、排序或人工指定 head 权重，而是直接定义 `conditional_all_blocks_fixed_clean_attention_class_logit_ig`：先以 clean input 冻结 12 层实际使用的完整 multi-head attention tensors，再对 frozen predicted-class raw logit 执行 input IG。Q/K/softmax variation 被切断；Value、全部 head 的原生拼接、attention output projection、residual、MLP、final norm 和 classifier 保持活动，由模型自身的下游计算完成跨 head 信息到 class logit 的传播。

该方法是 all-block、all-head、class-targeted 且 conditional 的 attribution，但不是：

- F3H head-direction maps 的加权和；
- raw attention 或 attention rollout；
- 完整 Q/K/V attribution；
- 未修改模型上的 full-model class-logit IG。

它回答的是“在全部 clean attention 固定的条件模型中，哪些输入区域支持冻结的预测类别 raw logit”，而不是“每个 head 存在一组已由 F3H 证明的独立类别权重”。

### 23.3 后续原子里程碑与门禁

1. **F3C1 implementation and preflight（当前唯一授权）**
   - 实现 all-block clean-attention capture/hash/injection、conditional class-logit IG、五方法统一 runner、immutable typed terminals、资源记录、恢复与 validator；
   - 完成 synthetic tests、tamper tests、false-pass audit，并在任何 pilot outcome access 前冻结精确的 paired permutation/p-value 实现；
   - 只允许使用已经查看且不进入统计的 order 44 做一次无渲染、无人工 map 检视的五方法工程/profile 检查；不得读取 5 个 pilot 或 50 个 formal 样本的 JPEG、map 或 outcome；
   - 必须验证全部 12 层 attention 均来自同一次 clean forward，conditional endpoint 对 clean logits 的 max absolute error `<1e-6`，NNS/full-model IG conservation、固定 class index、normalized-zero baseline、14×14 projection 和 provenance 均符合冻结合同；
   - terminal PASS 只能授权 `F3C2_PILOT_ONLY`。

2. **F3C2 five-image pilot**
   - 只执行冻结 orders `416, 350, 438, 411, 337`，每图五方法，共 25 个 typed-terminal units；
   - continuation decision 前不得查看 JPEG、map 或 display，只能读取 identity、terminal status、endpoint parity、conservation、resource 和 budget telemetry；
   - 只有 25/25 VALID、全部 hash/provenance 完整、NNS endpoint/conservation 和 full-model IG conservation PASS，且按 pilot telemetry 重算后满足总计 8 H20-GPU-hours、每 image group 0.30 H20-GPU-hour、每进程 16 GiB、retained artifacts 8 GiB 上限，才可授权 F3C3；
   - 任一门禁失败即停止。预算超限只能前瞻性修订，不能删方法、减样本、换图或自动 waiver。

3. **F3C3 50-image formal cohort**
   - 仅在 F3C2 terminal PASS 后执行冻结的 50 图 × 5 方法，共 250 个正式 units；
   - 每个 sample×method 必须不可覆盖并保留 `VALID/FAILED/NA`、native map、target、attention hashes、attempts、fidelity curves 和资源 provenance；
   - 不因 failure/NA 换样本、补零、插值或缩小 denominator；misclassified 样本照常保留；pilot、order 44 和 formal `n=50` 不池化。

4. **F3C4 CPU-only analysis and seal**
   - 只消费 sealed F3C2/F3C3 terminals，完成 reconciliation、统计、渲染、人工 QC、false-pass audit、completion、commit/push/archive；
   - primary contrast 固定为 class-targeted NNS versus ViT Grad-CAM；NNS versus full-model IG、attention rollout、raw CLS attention 为同一 secondary Holm family；正式统计单位为 image (`n=50`)；
   - workflow/package PASS 与科学比较结果分离。正结果、无显著差异或负结果均不得改方法、改队列或改显示规则；
   - 没有独立 sealed segmentation/bounding-box ground truth 时，不允许 object-localization superiority claim；
   - 只有 F3C terminal completion 才能授权 `F4_ONLY`。

5. **F4 final composition and manuscript integration**
   - F4 继续暂停，直到 F3C4 terminal PASS；
   - 恢复后只能组合已验收的 F1/F2/F3/F3H/F3C 模块，完成最终 figure numbering、caption、正文、source-backtrace、论文导出和 Stage F sealing；
   - 不得在 F4 改写 F3H/F3C 样本、method target、panel data、color scale、failure/NA、统计队列或 scientific outcome。

### 23.4 当前授权与执行提示词

截至本记录，F3C1、F3C2、F3C3、F3C4 和 F4 均尚未开始。当前唯一允许的新工作是 `F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY`；不得把 F3C1 的 order-44 engineering profile 计入 pilot/formal 统计，也不得据其外观或数值修改冻结方法。下一轮完整执行提示词见 `docs/stage_f_f3c1_execution_prompt_20260907.md`。
