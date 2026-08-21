# 阶段 C：step by step 执行与跨对话交接文档

> 文档性质：阶段 C 的路线图、任务状态表和跨对话交接日志。  
> 创建日期：2026-08-21。  
> 当前状态：阶段 A、阶段 B、阶段 C/C0–C2 已完成；下一任务是 C3。
> 使用规则：每个新对话必须先完整阅读本文，只执行一个指定的 `C_i`；该任务未更新本文并写出下一任务提示词时，不算完成。

## 1. 为什么使用这份文档

阶段 C 同时涉及实验协议、多个解释方法、高成本 GPU 计算、跨样本统计和严格的数据留存，不适合在一个对话中一次完成。本文把它拆成 C0–C10，并承担三个作用：

1. 保存所有对话共同遵守的阶段 C 边界，防止后续对话重新解释研究目标。
2. 记录每个任务实际完成了什么、用了哪个提交和配置、产物在哪里，以及哪些问题尚未解决。
3. 在每个任务完成时生成下一任务的完整提示词，使下一对话可以直接承接，而不依赖聊天记录。

本文是“活文档”，但并非所有部分都可随意修改：

- 第 2–7 节是当前设计提案。C0 负责核验并冻结为正式协议。
- C0 完成后，正式科学协议应另存为 `docs/阶段C_protocol.md`，验收合同应另存为 `docs/stage_c_acceptance.md`。后续不得根据实验结果偷偷修改。
- 第 8 节任务状态和第 10 节执行日志由每个任务持续更新。已有日志只允许追加勘误，不允许删除失败记录。
- 若确需修改已冻结协议，必须增加带理由、时间、旧/新哈希和影响范围的 protocol amendment；已看过的数据不能重新伪装成未见过的留出数据。

## 2. 已有基础与不可越过的边界

### 2.1 可以直接依赖的成果

- 阶段 A 已固定 canonical 模型、state dict 哈希、官方预处理、ImageNet 标签映射和真实 timm forward 提取。
- 阶段 A 已能取得 12 层的 block 输入输出、Q/K/V、实际 attention 和 logits，并完成 forward/parity 验证。
- 阶段 B 已实现 `conditional_fixed_attention_value_path_ig`、守恒检查、重复性、输入扰动 sanity、预算预检、provenance，以及 paper/audit/debug 留存策略。
- 工作区已有绑定 clean commit `30732c9` 的 Stage B `formal=true` paper 产物：`outputs/stage_b_formal_30732c9/`。
- `full_model_ig` 原语已经存在，但尚未接入阶段 C 的批量 runner、统一结果协议和真实模型集成验收。
- 旧 20 个图像 ID 对应的 JPEG 均存在；官方 ImageNet devkit 也已存在。

### 2.2 不能沿用或不能声称的内容

- 旧目录 `49_vit ` 仅作只读历史资产；旧 20 图结果只提供图像 ID 线索，数值结果必须在当前 canonical 模型和预处理下重跑。
- 阶段 B 的 Value-path 不是完整 NNS-ViT、不是类别 logit 解释，也不是完整 Q/K/V 或 attention attribution。
- `class_target` 在当前 Stage B NNS 中只是上下文，不能写成 Value-path 的守恒目标。
- 插值后的 14×14 patch 图不是原生像素级证据。
- colormap、Gaussian 平滑、百分位增强、max-pool 和旧式加权混图不能进入定量指标。
- 阶段 C 的 20 图只能用于 pilot 和大效应筛查，不能单独支持“对所有 ImageNet 图像普遍优于基线”的论文结论。

## 3. 推荐的样本设计

### 3.1 固定 20 个旧 ID

阶段 C 复用以下图像 ID，但重新构造含官方标签、图像 SHA-256、尺寸和选择角色的 registry：

```text
ILSVRC2012_val_00000079
ILSVRC2012_val_00002607
ILSVRC2012_val_00005279
ILSVRC2012_val_00007489
ILSVRC2012_val_00009888
ILSVRC2012_val_00012573
ILSVRC2012_val_00014787
ILSVRC2012_val_00018380
ILSVRC2012_val_00020970
ILSVRC2012_val_00023229
ILSVRC2012_val_00025927
ILSVRC2012_val_00028908
ILSVRC2012_val_00031099
ILSVRC2012_val_00033512
ILSVRC2012_val_00035680
ILSVRC2012_val_00038565
ILSVRC2012_val_00041511
ILSVRC2012_val_00044187
ILSVRC2012_val_00047079
ILSVRC2012_val_00049970
```

### 3.2 推荐的 5/15 划分

在不看新模型结果和热图的前提下，按旧有顺序等间隔选择 5 张校准图：

```text
ILSVRC2012_val_00000079
ILSVRC2012_val_00009888
ILSVRC2012_val_00023229
ILSVRC2012_val_00035680
ILSVRC2012_val_00049970
```

其中建议把首尾两张预注册为完整审计样本：

```text
ILSVRC2012_val_00000079
ILSVRC2012_val_00049970
```

其余 15 张是 pilot 内部留出集。冻结配置后仍在全部 20 张上运行，但必须区分：

- 5 张校准图用于全层/全头校准、候选选择和工程调试，只作描述性报告；
- 15 张留出图用于阶段 C 的 Go/No-Go；
- 全 20 张汇总只能标为 descriptive total；
- 15 张仍不是阶段 D 的独立论文测试集。

C0 可以在发现文件损坏、重复图像或标签资产不合法时修正该划分，但必须根据输入资产事实而不是热图效果修正，并记录理由。

## 4. 两条不能混淆的比较线

### 4.1 Value-path 比较线

当前 Stage B 解释的是单个标量：

```text
O_h[q,c] = sum_j A_h[q,j] * V_h[j,c]
```

它由 `block × head × query token × head channel` 唯一确定，attention 固定为输入端点的真实值。相比之下，direct raw CLS attention 是 `block × head` 的 196 个 patch 权重。二者并非同一目标，因此直接计算相关、JS/KL 或 hotspot overlap 时，只能称作空间结构相似性，不能直接解释为方法忠实性或优越性。

阶段 C 推荐把主 NNS 目标扩展为确定性的 head 级方向标量，而不是任意固定 `head_channel=0`。建议由 C0/C4 精确定义并测试如下候选：

```text
u_h = normalize(O_h(x) - O_h(x0))
f_h(z) = u_h^T [A_h(x) @ V_h(z)]_CLS
```

其中 baseline `x0`、端点 attention `A_h(x)` 和方向 `u_h` 在积分时固定。若方向范数低于预注册阈值，该配置应标为退化，不能临时换 channel。

与它严格匹配的 patch-token contribution 为：

```text
c_j = A_h(x)[CLS,j] * u_h^T (V_h(x)[j] - V_h(x0)[j])
```

需要单独记录 CLS token 的 contribution；只有 patch token `j=1..196` 可形成 14×14 空间图。该 matched patch map 才是判断 NNS 是否比 patch 插值提供新增像素信息的主要对照。

Value-path 比较线至少包含：

- NNS fixed-attention Value-path attribution；
- 同目标 matched patch-token contribution 及其质量守恒式插值；
- 同 block/head 的 raw CLS attention，明确标为不同目标的结构参考。

### 4.2 Class-logit 比较线

Class-logit 解释建议统一使用干净输入的 frozen predicted-class raw logit；真实标签和预测正确性同时记录，但不因预测错误而删除样本。扰动输入也继续解释该 frozen clean target，同时记录扰动是否导致类别翻转。

该比较线包括：

- Grad-CAM；
- `full_model_ig`，在文档和表格中写成“full-model class-logit IG”，不能简称为完整 attention attribution；
- attention rollout，标明它是 class-agnostic attention-flow baseline；
- raw CLS attention，标明它是 class-agnostic attention baseline。

不同目标的方法可以并列表述，但不能用 class-logit fidelity 直接证明内部 Value-path NNS 的目标守恒。

## 5. 统一 map、指标与稳定性提案

C0 必须把本节细化并冻结；C1/C5 再实现。以下是推荐默认口径。

### 5.1 空间与符号

- 跨方法空间比较的共同网格为原生 ViT patch 网格 `14×14`。
- `224×224` attribution 投影到 patch 网格时，对每个 `16×16` patch 求和，保持 attribution mass；不能用图片 resize 代替。
- patch 图还原到 `224×224` 对照图时，应在 patch 内质量守恒地均分，而不是把每个 patch 值复制 256 次后改变总和。
- raw attention 使用 `CLS -> patch tokens`，排除 CLS self-token 后重新归一化。
- signed attribution 的正证据 `max(M,0)` 和负证据 `max(-M,0)` 分开报告；不能默认取绝对值。
- 常量图、零质量图、非有限值和不适用的 signed distribution 必须返回结构化 `NA` 及 reason，不能静默平滑。

### 5.2 至少实现的逐样本指标

- Pearson 和 Spearman correlation；
- Jensen–Shannon divergence 为主要分布距离；
- KL divergence 为辅助量，仅在预注册的非负归一化与 epsilon 规则下计算；
- top-10% 和 top-20% hotspot Jaccard overlap，固定 tie-breaking；
- normalized entropy：`-sum(p log p) / log(HW)`；
- NNS conservation absolute error、relative error、实际判定阈值和收敛 steps；
- endpoint/intervention parity；
- deletion AUC、insertion AUC，以及 C0 最终决定的 sufficiency/comprehensiveness；
- 原图和每种扰动之间的 correlation、JS 和 hotspot overlap；
- 每个 sample × method × configuration 的 wall time、峰值 CUDA allocated/reserved 和 artifact bytes。

### 5.3 Fidelity 的公平性

C0 必须固定 deletion/insertion 的：

- 被解释的标量目标；
- baseline/替换值；
- 删除与插入单位；
- 排序使用正证据还是其他预注册变换；
- steps 和曲线积分方式；
- Value-path 与 class-logit 两条线分别适用哪些方法。

若要检验像素级新增信息，不能只用 16×16 patch 作为删除单位，否则 NNS 的 patch 内排序优势无法被观察；但更细单位会引入分布偏移。因此建议同时保留：

- patch 级 fidelity：公平、稳健，但不支持像素级新增主张；
- 预注册的 patch 内细粒度 fidelity：用于新增信息分析，并配套 random/center 等 sanity baseline。

### 5.4 扰动稳定性

至少包含：

- 小平移；
- 轻微中心裁剪/尺度变化；
- 轻微 Gaussian 噪声。

C0 必须固定像素强度、方向、seed、padding、插值方式和 map 逆变换。几何扰动的解释图必须逆变换到共同坐标，只在共同有效区域比较。每种稳定性同时报告：

- 所有样本、固定 clean target 的结果；
- 预测类别保持不变子集的描述性结果；
- 类别翻转率。

注意：Stage B 已有的 `relative_l1_change >= 0.05` 是“解释不能对输入完全无反应”的 sensitivity sanity，不是阶段 C 所需的稳定性上界，二者必须分别保留。

## 6. 留存、provenance 和资源规则

### 6.1 分层留存

| 级别 | 范围 | 保存内容 | 用途 |
|---|---|---|---|
| L0 | 每个实际运行的 sample × method × config | 状态、逐样本标量、预测、目标、失败原因、时间/显存/磁盘、provenance | 统计与完整性核查 |
| L1 | 所有预注册层/头 | metric-ready 二维数值图；指标用 float32，验证量化后可按策略存 float16 | 重算指标和论文分析 |
| L2 | 1–2 个预注册审计样本 | float32 raw attribution、raw attention、端点、收敛和可复核中间量 | parity、守恒和问题追踪 |
| L3 | 少量预注册展示/失败案例 | 从 L1/L2 生成的 PNG | 展示，不得作为定量输入 |

禁止保存每个 quadrature step、每个像素、每个维度的 PNG 或文件。可由固定代码廉价重算的内容优先保存 provenance，不保存重复副本。

### 6.2 每个结果至少绑定

- Git commit 和 clean/dirty 状态；
- model、image、dependency lock、sample registry、protocol、config、selection 的 SHA-256；
- sample role：calibration、holdout 或 audit；
- 方法精确名称、版本和目标；
- map projection、符号处理和 normalization；
- baseline、IG schedule、seed 和扰动参数；
- 实际状态、失败原因、资源记录及 artifact inventory；
- `status.json` 的完成状态和 manifest hash。

### 6.3 运行和恢复

- 每个 sample × method × config 使用独立输出单元，禁止覆盖或向不完整目录续写。
- 批量恢复只能跳过 `status.complete=true`、manifest hash 和 artifact hash 全部有效的单元。
- 失败单元必须保留在 failure table 中；修复后使用新的空输出单元重跑。
- 运行前按样本、方法、层、头、target、扰动和重复次数估算总调用量与磁盘。
- GPU 时间必须 `cuda synchronize` 后测量；峰值显存测量前重置统计，并区分 allocated/reserved。

## 7. Go / No-Go 提案

C0 必须冻结最终门槛。以下值是推荐默认值，C0 完成前仍标为 proposal。

### 7.1 Validity gate

- endpoint parity 最大绝对误差 `< 1e-6`；
- conservation 满足 `abs_error <= 1e-4 + 0.01 * abs(delta)`；
- primary 配置的 15/15 留出图均运行成功、有限且 provenance 完整；
- 审计样本重复运行满足阶段 B 既有 `atol/rtol`；
- 不静默删除失败、退化或预测错误样本。

任一关键 validity 条件失败时，决策为 `NO_GO_VALIDITY`，不能继续讨论优越性。

### 7.2 Fidelity/novelty gate

建议定义与 matched patch baseline 的逐图综合增益：

```text
delta_fid = 0.5 * [
  (InsertionAUC_NNS - InsertionAUC_patch)
  +
  (DeletionAUC_patch - DeletionAUC_NNS)
]
```

推荐进入阶段 D 的全部条件：

- 15 张留出图的 `median(delta_fid) >= 0.02`；
- 至少 12/15 张方向为正；
- 按图像整簇 bootstrap 的 95% CI 下界大于 0；
- NNS 相对其 patch 投影的 patch 内残差不是纯数值噪声，并在对齐扰动下达到预注册的稳定性门槛；
- 成本低于 C0/C7 冻结的阶段 D 可扩展上限。

低相关或较大 JS 本身不算“新增信息”。只有稳定的 patch 内信息同时带来同目标 fidelity 改善，才能支持 `GO`。

建议使用以下决策枚举：

- `GO`：有效、存在达到最小实质效应的稳定新增信息、成本可扩展；
- `NO_GO_VALIDITY`：守恒、parity、完整性或方法合同失败；
- `NO_GO_NOVELTY`：证据支持达不到预注册的最小有意义增益；
- `NO_GO_COST`：方法有效但无法在阶段 D 预算内扩展；
- `INCONCLUSIVE`：区间过宽或跨越门槛；不得用它宣称有效或等价，也不进入“优于标准方法”的主实验。

“没有显著差异”不等于“二者等价”。若论文要声称 NNS 与 patch 插值等价，需另行预注册等价界值并做 TOST 或相应的区间判定。

### 7.3 统计边界

- 统计单位是图像，Go/No-Go 的 `n=15`；层、头、像素、token、扰动 seed 不能当作额外独立样本。
- 5 张校准图不能进入确认性置信区间。
- bootstrap 必须按图像整簇重采样，建议 10,000 次。
- primary 配置使用配对比较；其他候选只作探索性结果并做 Holm 等多重比较修正。
- 同时报逐样本值、均值、中位数、标准差和 95% CI，不能只保存或报告均值。

## 8. C0–C10 总路线与状态

状态只能使用：`NOT_STARTED`、`IN_PROGRESS`、`PASS`、`BLOCKED`。任何任务只有在代码/文档、测试、产物和本文交接记录都完成后才能标为 `PASS`。

| 任务 | 当前状态 | 目标 | 关键产物 | 前置 |
|---|---|---|---|---|
| C0 | PASS | 冻结协议与样本 registry | protocol、acceptance、sample registry、配置草案 | A/B |
| C1 | PASS | 严格配置、统一 map 与指标合同 | config/map/metrics 模块和测试 | C0 |
| C2 | PASS | direct raw CLS attention 与 rollout | 两种方法实现、方法合同测试 | C1 |
| C3 | NOT_STARTED（NEXT） | Grad-CAM 与 full-model class-logit IG | 两种方法适配、真实模型 smoke test | C1 |
| C4 | NOT_STARTED | head 级 NNS 与 matched patch baseline | 同目标实现、守恒与退化测试 | C1 |
| C5 | NOT_STARTED | fidelity、扰动稳定性、资源仪表 | fidelity/perturbation/stability/resource 模块 | C2–C4 |
| C6 | NOT_STARTED | 单图/批量 runner、留存、provenance、CLI | 可恢复批量流水线及测试 | C1–C5 |
| C7 | NOT_STARTED | 单图 smoke benchmark 与预算冻结 | benchmark、总预算和可运行性结论 | C6 |
| C8 | NOT_STARTED | 5 图全层/全头校准和 selection 冻结 | calibration 产物、`selection.json` | C7 |
| C9 | NOT_STARTED | 全 20 图冻结配置 pilot | 完整逐样本产物和 failure table | C8 commit clean |
| C10 | NOT_STARTED | 汇总统计和 Go/No-Go | per-sample 表、summary、decision、报告 | C9 |

## 9. 每个任务的具体边界

### C0：冻结实验协议与样本 registry

目标：把本文的 proposal 变成在看新结果之前已冻结、可校验的正式协议。

必须完成：

- 使用官方 devkit 重新生成 500 图 manifest，从中精确选取上述 20 ID；
- 建立 tracked sample registry，记录相对路径、图像 SHA-256、尺寸、官方 WNID/index、角色及选择原因；
- 确认 5 calibration、15 holdout、2 audit；
- 写 `docs/阶段C_protocol.md` 和 `docs/stage_c_acceptance.md`；
- 明确两条目标线、head 级目标、matched patch baseline、共同网格、指标、扰动、fidelity、退化/缺失规则、统计和决策门槛；
- 写 Stage C 配置草案和 schema 设计，不实现方法；
- 更新本文状态与 C0 完成记录，写出 C1 的完整提示词。

验收：所有 ID 唯一且文件/hash/官方标签可复核；协议中不存在“待定后看结果决定”的关键分析自由度；不运行昂贵 GPU；测试与文档检查通过；提交后工作树干净。

### C1：严格配置、统一 map 与指标合同

目标：先实现所有方法共同遵守的数据语言，防止每种方法各自归一化。

必须完成：

- 严格 Stage C config schema，未知字段、重复 key、非有限数和非法组合 fail closed；
- 定义 `MethodResult`/等价合同：方法、目标、native tensor、metric-ready 2D map、projection、符号语义、provenance fragment；
- 实现 native 224×224 与 14×14 之间的质量守恒 projection；
- 实现 Pearson、Spearman、JS、KL、top-k overlap、normalized entropy 和结构化 NA；
- 为零图、常量图、signed map、tie、shape mismatch、非有限值写测试；
- 更新本文并生成 C2 提示词。

验收：指标只能读取 metric-ready 数值数组；display 数据无法回流；所有公式和实现与 protocol 一致；全量非 GPU 测试通过。

C1 已完成。简单说，后续方法现在必须先把结果放进同一个、可检查的数据盒子：写清方法和目标，保留原始数值，再生成统一的 14×14 定量图。像素图变 patch 图时用求和，patch 图展开时均分质量；展示图片不能送回指标。配置和数值只要有未知字段、错误组合、非有限值或非法 shape，就会明确失败或返回带原因的 NA，不会偷偷补默认值或把失败当成 0。

### C2：direct raw CLS attention 与 attention rollout

目标：实现两个低成本、可审计的 attention baseline。

必须完成：

- direct 从真实 extractor 的 attention 构造，不手工重演 block；
- 固定 `attention` 与 `attention_used` 的语义，eval 下验证一致性；
- 正确执行 CLS key 排除、patch reshape 和 head 规则；
- rollout 明确 residual identity、row normalization、head fusion、层乘法方向和截止层；
- 单层退化、解析 toy 矩阵、真实 extractor parity、状态恢复测试；
- 更新本文并生成 C3 提示词。

验收：direct 与 extractor 精确一致；rollout 行归一化、数值有限、乘法次序有可手算测试；不保存无关的全 attention 笛卡尔积。

C2 已完成。简单说，direct 现在只读取真实模型 forward 已经使用的 `attention_used`：Value-path 参考严格选同一 block/head，class-logit 参考固定取 block 11 的 12-head 均值；两者都先去掉 CLS key，再把 196 个 patch 归一到总质量 1。rollout 严格使用 `row_norm(I+mean-head attention)`，按 `B_11 ... B_0` 相乘。两者都是 class-agnostic attention baseline，不是类别归因；per-head direct 与 Value-path NNS 也只比较空间结构，不是同目标 fidelity。结果只保留 C1 合同要求的 float32 14×14 数值 map，不保存派生的 raw-attention 全组合。

### C3：Grad-CAM 与 full-model class-logit IG

目标：补齐 class-logit 解释线。

必须完成：

- 实现 ViT reshape transform、目标层和 ReLU/符号规则明确的 Grad-CAM；
- 正式包装已有 `full_model_ig`；
- 两者使用同一个 frozen clean predicted-class raw logit；
- target layer、baseline、IG schedule 和 map projection 全部进入 provenance；
- toy 和真实模型低成本 smoke test；
- 验证 hook 清理、model mode、parameter grad/`requires_grad` 恢复；
- 更新本文并生成 C4 提示词。

验收：输出有限且非空、target-sensitive；两次运行达到配置容差；不能把 full-model IG 写成完整 attention attribution。

### C4：head 级 NNS 与 matched patch baseline

目标：解决当前单 channel NNS 与 direct head 不对齐的问题，并建立最关键的 novelty 对照。

必须完成：

- 按 protocol 实现确定性 head-direction target；
- 明确方向由端点确定后在 IG 中 detach/freeze；
- 构造 matched per-token/patch contribution，单独记录 CLS contribution；
- 支持 12 层×12 头的选择性、分块或批量计算；
- 对零方向、极小 delta、纯 CLS contribution 等退化情形 fail closed 或结构化标记；
- 保持现有 Stage B 单 channel 方法作为兼容/审计路径，不篡改旧产物语义；
- synthetic 精确守恒、真实 block parity 和 wrapper 回归测试；
- 更新本文并生成 C5 提示词。

验收：NNS 与 matched patch baseline 明确解释同一个标量；包含 CLS contribution 后 token contributions 与 target delta 对齐；空间图只声明 patch 部分。

### C5：fidelity、稳定性与资源测量

目标：实现 Go/No-Go 真正依赖的评价层。

必须完成：

- 两条目标线各自正确的 deletion/insertion；
- patch 级和预注册细粒度 fidelity；
- 平移、裁剪、噪声及逆几何对齐；
- clean target 固定、预测翻转记录；
- map 稳定性和 patch 内 residual 稳定性；
- 同步 wall time、CUDA peak allocated/reserved、实际 artifact bytes；
- toy metric、几何 round-trip、CPU/CUDA 条件测试；
- 更新本文并生成 C6 提示词。

验收：没有把 Stage B sensitivity sanity 当成稳定性；所有方法使用完全相同的 perturbation tensors 和 fidelity operator；资源测量范围可解释。

### C6：Stage C runner、留存、provenance 与 CLI

目标：将方法和指标组合成可中断、可核验、不会假成功的批量流水线。

必须完成：

- 单样本 runner 与 manifest-driven batch orchestrator；
- `estimate`、`run-sample`、`run-batch`、`validate`、`aggregate` 或等价 CLI；
- 每个 execution unit 独立的 incomplete/complete 状态和 hash inventory；
- L0/L1/L2/L3 留存策略，raw attention 不再被强制误当成 RGB attribution；
- run-level provenance、failure table、逐样本长表 schema；
- 总预算覆盖样本×方法×配置×扰动×重复；
- crash、resume、tamper、partial failure、hash mismatch 测试；
- 更新本文并生成 C7 提示词。

验收：只有完整且 hash 有效的单元可被跳过；失败不静默；批次 complete 需要所有预期单元有显式终态；不会覆盖旧结果。

### C7：单图 smoke benchmark 与预算冻结

目标：在启动 5×12×12 校准前验证真实 GPU 行为和总成本。

必须完成：

- 在一个 audit 图上运行所有方法；
- NNS 至少覆盖预注册的早/中/晚层和若干头，以观察最坏收敛 steps；
- 记录每方法/配置的时间、峰值显存、调用数和输出字节；
- 用保守上界估计 C8、C9 和可能的阶段 D 成本；
- 只依据资源和数值有效性冻结预算，不依据热图美观选择配置；
- 生成 benchmark report；
- 更新本文并生成 C8 提示词。

验收：无 OOM、无残留 hook/grad、预算估算可复算；若超预算则标记 `BLOCKED` 或提交 protocol amendment，不能静默减少层/头/样本。

### C8：5 图全层/全头校准并冻结 selection

目标：只使用校准图选择阶段 C 的 frozen candidate configurations。

必须完成：

- 5 图×12 层×12 头的规定计算和逐样本标量；
- 仅 2 张 audit 图保留 L2，其余按 L0/L1；
- 先应用 validity/degeneracy filter，再按预注册 score/ranking 选择；
- 建议按 `[0–2]`、`[3–5]`、`[6–8]`、`[9–11]` 四个深度带各冻结一个配置，再冻结其中一个 primary；
- tie 按 `(block_index, head_index)` 升序或 protocol 的固定规则；
- 输出 `selection.json`、selection report 和所有输入哈希；
- 单独提交 selection，确保 C9 前工作树干净；
- 更新本文并生成 C9 提示词。

验收：选择代码不能读取 holdout 结果；selection 文件不可人工改写；选择数量、排序和失败处理与 protocol 完全一致。

### C9：运行冻结配置的同 20 图 pilot

目标：不再调参地生成完整 pilot 结果。

必须完成：

- 验证 clean commit、protocol/config/registry/selection hash；
- 对全部 20 张执行 frozen 方法和配置；
- 所有方法共享 canonical input tensor、模型和目标上下文；
- 扰动生成和配对严格一致；
- 写逐样本标量、metric-ready maps、资源、provenance、failure table；
- 运行结束执行完整性与 hash validation；
- 不在本任务根据结果修改 selection 或门槛；
- 更新本文并生成 C10 提示词。

验收：预期 execution units 全部处于显式终态；primary 的留出样本满足 validity 所需完整性，否则报告而非剔除；无 PNG 被用于定量。

### C10：汇总、统计和 Go/No-Go

目标：根据冻结协议作阶段 C 决策，而不是重新调图或挑配置。

必须完成：

- 输出不丢失行的 `per_sample.csv` 或 Parquet；
- 分别报告 5 calibration、15 holdout 和 descriptive all-20；
- 以图像为 cluster 做配对统计和 bootstrap CI；
- 汇总 validity、fidelity、novelty、stability、cost 和 failure；
- 输出 machine-readable `decision.json`；
- 写阶段 C acceptance/report，明确适用范围和不能声称的结论；
- 若非 GO，明确转向“朴素 NNS 为何失效及如何忠实分解”还是因工程/成本需修复；
- 更新本文最终日志，并生成阶段 D 或失败诊断任务的完整提示词。

验收：决策只使用 15 holdout；不把 calibration 混入确认性 CI；不把层/头像素当独立样本；结论逐项对应预注册门槛。

## 10. 跨对话执行与交接协议

### 10.1 每个新对话开始时必须做

1. 完整阅读本文、正式 Stage C protocol/acceptance（若已存在）和上一任务 completion record。
2. 检查 `git status`、HEAD、上一任务记录的 config/artifact hash 和状态文件。
3. 只执行提示词指定的一个 `C_i`，不要顺手开始 `C_{i+1}` 的正式运行。
4. 若发现已有未提交改动，先判断是否属于用户或上一任务；不得覆盖或回滚。
5. 将本文对应任务状态改为 `IN_PROGRESS`。

### 10.2 每个任务结束时必须做

1. 完成实现、测试、文档和所需产物验证。
2. 将第 8 节对应状态更新为 `PASS` 或 `BLOCKED`。
3. 在第 11 节追加 completion record；即使任务失败也要记录，不得删除失败史。
4. 把下一任务的“完整可复制提示词”写进 completion record；不能只写“继续 C_i+1”。
5. 在最终回复中重复：结果、关键文件、测试、commit/hash、遗留问题和下一提示词。
6. 实现任务原则上一任务一提交；若用户在该对话禁止 commit，则必须记录 dirty diff 和原因，下一任务不得假设 clean。

任务只有满足上述交接要求才能标记 `PASS`。

### 10.3 Completion record 模板

````markdown
### YYYY-MM-DD HH:MM — C_i completion record

- status: PASS | BLOCKED
- objective_completed:
- git_sha_before:
- git_sha_after:
- tree_clean:
- files_added_or_changed:
- inputs_and_sha256:
- configs_and_sha256:
- artifacts_and_absolute_paths:
- artifact_manifest_or_status_hashes:
- tests_and_results:
- acceptance_results:
- deviations_or_protocol_amendments:
- failures_retained:
- known_limitations:
- next_task: C_i+1

#### 下一对话完整提示词

```text
（在此写入完整、可直接复制的新对话提示词，包括工作目录、必读文件、
前置 commit/hash、任务范围、禁止事项、交付物、验收标准和更新本文要求。）
```
````

## 11. 执行日志

### 2026-08-21 16:05 +0800 — C0 completion record

- status: PASS
- objective_completed: 在未查看 Stage C heatmap/候选指标、未运行模型或 GPU attribution 的前提下，使用 Stage A manifest 能力与 hash-gated 官方 devkit 重建 500 图 manifest；冻结 20 图 registry、5/15 统计角色和 2 图 audit overlay；冻结 protocol 1.0.0、fail-closed acceptance、配置草案、registry/config/selection schemas 和 C0 静态验证器。
- git_sha_before: `30732c9f1413c58a36dd6a8d85ba8ea5788e0c8a`
- git_sha_after: `SELF`（即包含本 completion record 的单一 C0 commit；commit SHA 不能无循环地写入自身内容，精确值由提交后 `git rev-parse HEAD` 产生并在最终回复报告）
- tree_clean: `true`（C0 commit 后复验）
- files_added_or_changed: `.gitignore`; `config/stage_c.json`; `config/stage_c.schema.json`; `config/stage_c_registry.schema.json`; `config/stage_c_selection.schema.json`; `docs/阶段C-step-by-step.md`（保留并纳入用户授权基线）; `docs/阶段C_protocol.md`; `docs/stage_c_acceptance.md`; `docs/stage_c_config_contract.md`; `manifests/stage_c_flat_val_500/manifest.csv`; `manifests/stage_c_flat_val_500/manifest_metadata.json`; `manifests/stage_c_sample_registry.csv`; `manifests/stage_c_sample_registry_metadata.json`; `tests/test_stage_c_contract.py`。
- inputs_and_sha256: official devkit `b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953`; Stage A 500 manifest `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`; 500 manifest metadata `9a8df6201fe85c8529e2d307ee09142b3b55960dca059a29882d2e03f933ad5b`; dependency lock `81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984`; legacy read-only list `1f61dcd2f1a76b49d5ea8d6aac1fdc09f3a8f0261719474b80473eea716a9057`; legacy selection script `e2380da105d093d16c92c7274351db275389c0a8e2d7142f3762e362943af113`。
- configs_and_sha256: registry `9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0`; registry metadata `bedb9df9a77807c86da0b898f8d44767c8276f21ab69821965f2b30d7184a4af`; Stage C config `0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669`; config schema `94209b7c8a63a62ef27f3396810c8d5be8f6fbbf0a4df22e663b3661b9ed3f4b`; registry schema `4ce0ac8cd5bff93ae2b323c3e0447a87fc825c3243ad46c54c9b48e05e9b2d6d`; selection schema `867a3cf22519b2a045d81751a20c5cd52570ddeadede26a2da2bb79973f0b59d`; protocol `ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8`; acceptance `f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd`。
- artifacts_and_absolute_paths: tracked 500 manifest `/data00/home/yeziqi.831/testcode/vitnns_rebuild/manifests/stage_c_flat_val_500/manifest.csv`; tracked registry `/data00/home/yeziqi.831/testcode/vitnns_rebuild/manifests/stage_c_sample_registry.csv`; no model/JPEG/devkit/runtime artifact was added to Git。
- artifact_manifest_or_status_hashes: Stage B status SHA-256 `2129134ff256fe9421a599e420ac2f79a19342b005d0f0ce34ecdba670d7a846`; Stage B manifest SHA-256 `57b2ff7e3189c086650d6680252e16373617260e14f404b9ba78223c3963c7eb` and status binding matched; artifact hashes `maps.npz=e58a6335a0d117db2af6df4165c51f19c4cabe5690188e8c1516d3b92a9bdb91`, `metrics.json=4ffc9a5d00f06862722cf3aafcbf67c8c09a8a529cd6906734187df9686b3161`, `provenance.json=279b1dfaa419ccde3efdf778318d3802fcef5e0c1c8e6f008f71fc32cddda497`; provenance recorded commit `30732c9f1413c58a36dd6a8d85ba8ea5788e0c8a`, `formal=true`, `dirty=false`。
- tests_and_results: initial unqualified `python -m vitnns.cli ...` failed immediately with Python 2.7 `No module named vitnns` and wrote nothing; rerun through `conda run -n vitnns` wrote 500 rows. `python tests/test_stage_c_contract.py` PASS; focused `pytest tests/test_stage_c_contract.py tests/test_dataset_labels.py -q` reported `9 passed`; pre-commit full `pytest -m 'not unsafe_pickle' -q` reported `106 passed, 1 deselected`. 提交后必须再运行 validator `--require-tracked --require-clean` 和全量非 GPU suite；其实际结果在本次任务最终回复报告。No GPU or attribution run occurred; the full regression suite may exercise existing Stage A/B CPU forwards.
- acceptance_results: 500/500 official labels; exact 20/20 frozen IDs; unique IDs/paths/hashes; roles 5 calibration + 15 holdout; audits 2 and both calibration; live file hash/bytes/PIL dimensions match; labels independently rebuilt from official devkit; JSON strict parsing/schema shape and cross-field contract pass; Stage B status→manifest→artifact hash chain pass; no unknown/missing required registry/config field in the frozen instance.
- deviations_or_protocol_amendments: no post-result amendment and no role adjustment. C0 resolved proposal ambiguity before results: baseline endpoint uses fixed clean `A(x)`; direction threshold is relative `1e-6`; fine fidelity is 4×4 cells rather than single pixels; perturbation geometry/ties/selection/decision priority and hard budgets are numerical. L1 is frozen float32 rather than optional float16 so C1 needs no unfrozen quantization gate.
- failures_retained: the initial wrong-environment CLI failure above is retained; it created no output. No input asset failed.
- known_limitations: 20 images are a historical filename-spaced, non-class-balanced pilot and not a Stage D independent test; `00012573` is only 90×90 before canonical resize; budget caps are governance limits awaiting C7 measurement; class-logit fidelity may be structured NA when its endpoint delta is degenerate and is secondary to the Value-path Go/No-Go; C0 defines schemas/static checks but C1 must implement the production strict parser/maps/metrics; no equivalence test is registered.
- next_task: C1

#### 下一对话完整提示词

```text
你现在只执行 ViT-NNS 重建工程“阶段 C”的 C1：实现严格配置、统一 map 与指标合同。不要开始 C2，不要实现 direct raw CLS attention、attention rollout、Grad-CAM、head-level NNS 或 full-model class-logit IG 的 Stage C 方法适配，也不要运行昂贵 GPU attribution。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

前置冻结基线：
- C0 commit: `SELF`，即包含本 completion record 的当前 clean HEAD；启动 C1 时先运行 `git rev-parse HEAD`，其精确 SHA 应与 C0 最终回复给出的 commit SHA 一致。
- Stage C registry SHA-256: 9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0
- Stage C config SHA-256: 0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669
- Stage C protocol SHA-256: ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8
- Stage C acceptance SHA-256: f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd
- Stage A 500 manifest SHA-256: 042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89

第一步必须完整阅读：
1. docs/阶段C-step-by-step.md
2. docs/阶段C_protocol.md
3. docs/stage_c_acceptance.md
4. docs/stage_c_config_contract.md
5. config/stage_c.json
6. config/stage_c.schema.json
7. config/stage_c_registry.schema.json
8. docs/阶段A.md 和 docs/stage_a_acceptance.md
9. docs/阶段B.md 和 docs/stage_b_acceptance.md
10. README.md
11. 本文 C0 completion record

开始前必须：
- 检查 `git status` 和 HEAD，要求 HEAD 等于上述 C0 commit 且工作树 clean；若不一致，查明归属，不覆盖用户改动。
- 运行 `conda run -n vitnns python tests/test_stage_c_contract.py --require-tracked --require-clean`，核验 registry/config/protocol hash 与 500 图官方标签链。
- 核验 Stage B formal status→manifest→artifact 哈希链；不要把 Stage B 单-channel Value-path 改称完整 attention attribution或类别解释。

C1 范围与交付物：
1. 实现生产级严格 Stage C config loader/immutable typed config：拒绝 duplicate JSON key、NaN/Infinity、未知/缺失字段、bool 冒充 int、非法 enum/range/order/重复项、未解析引用和所有 protocol 的非法组合；加载后 round-trip 不注入隐式默认。
2. 实现统一 `MethodResult`（或等价）合同，明确 method/version、comparison line、explanation/fidelity target、native tensor、metric-ready 2D map、projection、sign semantics、valid mask、结构化 NA 和 provenance fragment。class-agnostic map 不得冒充 class attribution。
3. 实现并测试质量守恒 projection：signed RGB channel sum；224×224→14×14 的非重叠 16×16 求和；14×14→224×224 每 patch mass 除 256；4×4 cell sum 与每 patch 除 16。必须先投影 signed map 再拆 positive/negative；`atol=rtol=1e-6`。
4. 实现 Pearson、Spearman（average ranks）、base-2 JSD（无 epsilon、0log0=0）、双向 base-2 KL（epsilon=1e-12）、top-10%/20% Jaccard（ceil、value desc + row-major tie）、normalized entropy 和结构化 NA，严格遵守 protocol 的 sign view、mask、constant/zero-mass/nonfinite/shape 规则。
5. 强制指标只接受 metric-ready 数值数组，display PNG/colormap/resize 后展示数据不得回流。
6. 测试必须覆盖正例和 fail-closed 负例：duplicate/unknown/missing/nonfinite、非法 method-target 组合、零图/常量图/signed/unsupported negative/ties/shape mismatch/nonfinite/empty mask、投影质量守恒与 round-trip；并保持 Stage A/B 非 GPU 测试不回归。
7. 更新 docs/阶段C-step-by-step.md：C1 标 PASS 或 BLOCKED；若 PASS 将 C2 标 NEXT；追加完整 completion record 和“新对话执行 C2”的完整可复制提示词。
8. 检查 diff 后提交 C1 tracked 文件；提交后重跑 C0 validator（含 tracked/clean）、C1 测试和全量非 GPU 测试并确认 clean。

禁止事项：
- 不修改冻结的 registry、角色、科学阈值、方法目标或决策 gate；确有矛盾必须停止并走带旧/新 hash 的 protocol amendment，不能静默改。
- 不读取旧/新 display PNG 计算指标，不从 `__pycache__`/`.pyc` 恢复源码，不复用旧数值结果。
- 不实现或运行 C2–C5 的比较方法、fidelity/扰动实验，不运行高成本 GPU。
- 不把 raw attention 与 Value-path NNS 的相关性称为同目标 fidelity，不把 14×14 插值称为原生像素证据。

验收标准：
- `docs/stage_c_acceptance.md` 的 C1 六项全部通过；
- frozen config 严格加载并 round-trip，所有负例 fail closed；
- projection 在正/负/零/随机 map 上守恒，指标公式与冻结协议一致，所有 NA reason 可测试；
- 全量非 GPU pytest 通过，未运行 GPU attribution；
- 一任务一提交，最终工作树 clean，completion record 的 commit/hash/test 信息完整。
```

### C0 启动时使用的历史完整提示词

```text
你现在只执行 ViT-NNS 重建工程“阶段 C”的 C0：冻结实验协议与 20 图样本 registry。不要开始 C1，不要实现比较方法，也不要运行昂贵 GPU attribution。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

如果 `docs/阶段C-step-by-step.md` 此时仍是未提交文件，它是用户已授权创建的阶段 C 基线文档；必须保留、基于它继续，并纳入 C0 的最终提交，不能当作无关脏改动删除或覆盖。若还有其他未提交改动，则先查明归属并避免覆盖。

第一步必须完整阅读：
1. docs/阶段C-step-by-step.md
2. docs/阶段A.md
3. docs/stage_a_acceptance.md
4. docs/阶段B.md
5. docs/stage_b_acceptance.md
6. README.md
7. ../论文1实现审计与后续计划.md 中阶段 C 及选择性留存相关章节

开始前检查并记录：
- git status 和当前 HEAD；
- outputs/stage_b_formal_30732c9/status.json、manifest.json 和 provenance 的完成状态、commit、formal/dirty、artifact hash；
- 旧 20 图清单和所有 JPEG 是否存在；旧目录“49_vit ”只读，不修改，也不复用旧数值结果；
- artifacts/sources/ILSVRC2012_devkit_t12.tar.gz 的实际 SHA-256，并通过现有 Stage A 官方标签路径使用它。

C0 必须完成以下工作：

1. 使用现有 Stage A manifest 能力和官方 devkit，为 flat validation 500 图生成可审计 manifest；从中精确提取 docs/阶段C-step-by-step.md 列出的 20 个图像 ID。
2. 新建受 Git 跟踪的 Stage C sample registry。每行至少包含：sample_id、相对路径、image SHA-256、原始尺寸、官方 WNID、ImageNet-1k index、label source、role、selection rule。不得用模型预测代替真值。
3. 在不查看新 heatmap 或新候选指标的情况下核验并冻结：
   - 5 张 calibration：00000079、00009888、00023229、00035680、00049970；
   - 15 张 holdout：其余图；
   - 2 张 audit：00000079、00049970，且 audit 是 calibration 的子集。
   如因文件/hash/标签异常必须调整，只能基于输入资产事实，记录完整理由，不能基于模型效果调整。
4. 新建 docs/阶段C_protocol.md，把 step-by-step 文档第 2–7 节的 proposal 核验、补全并冻结。必须明确：
   - Value-path 与 class-logit 两条比较线；
   - NNS head-level target 的数学定义、baseline、endpoint attention 和 direction 的冻结方式；
   - matched patch-token contribution、CLS token contribution 和 14×14/224×224 质量守恒投影；
   - direct raw CLS attention、rollout、Grad-CAM、NNS Value-path、full-model class-logit IG 的精确名称与适用比较；
   - signed/positive/negative map、相关、JS/KL、top-k、normalized entropy 的定义和 NA/tie/epsilon 规则；
   - deletion/insertion 的目标、baseline、删除单位、steps、AUC，以及 patch 级和细粒度 fidelity 的边界；
   - 小平移、裁剪/尺度、轻噪声的精确参数、seed、map 逆变换和有效区域；
   - L0/L1/L2/L3 留存、provenance、失败与排除、恢复和预算规则；
   - 统计单位为图像，5 calibration 不进入确认性统计，15 holdout 决定 Go/No-Go；
   - GO、NO_GO_VALIDITY、NO_GO_NOVELTY、NO_GO_COST、INCONCLUSIVE 的数值规则。
5. 新建 docs/stage_c_acceptance.md，把协议转写成逐项、可测试、fail-closed 的验收合同。所有 proposal 数值要么正式冻结，要么在 C0 中基于方法学理由修改并记录；不能留到看 C8/C9 结果后再决定。
6. 新建 Stage C 配置草案及 schema/字段说明，至少覆盖 sample registry、角色、方法、目标、层/头、projection、指标、fidelity、扰动、retention、compute/storage budget、selection 和 decision gates。C0 只定义合同，不实现 C1 的解析器或方法。
7. 为新增 JSON/CSV/文档做适当的静态校验：JSON 可解析、ID 唯一、角色计数正确、文件和 hash 匹配、标签来自官方 devkit、无未知/缺失必需字段。若添加轻量测试，运行它；不要运行高成本 GPU 任务。
8. 更新 docs/阶段C-step-by-step.md：
   - 将 C0 状态改为 PASS 或 BLOCKED；
   - 若 PASS，将 C1 标为 NEXT；
   - 在“执行日志”追加完整 C0 completion record，填写前后 Git SHA、clean 状态、文件、输入/config/artifact hash、测试、验收结果、偏差和已知限制；
   - 在 completion record 中写出“新对话执行 C1”的完整可复制提示词，必须包含必读文件、C0 commit/hash、C1 范围、禁止事项、交付物、测试和验收标准。
9. 检查 git diff，确保没有修改旧归档和无关用户文件。提交 C0 的 tracked 文件，提交信息清楚；提交后重新运行轻量校验并确认工作树干净。不要提交模型、JPEG、devkit 或大型 runtime artifacts。

关键约束：
- 不从遗留 __pycache__ 或 .pyc 恢复或宣称 Stage C 源实现；它们不是可审计源码。
- 不把 Stage B NNS 称为完整 attention attribution 或类别解释。
- 不把 raw attention 与 Value-path NNS 的相关性称为同目标 fidelity。
- 不根据图片内容、预测正确性或热图美观挑 calibration、layer、head。
- 不读取 display PNG 计算任何指标。
- 不静默修正协议中的歧义；应在 protocol 中做明确选择并写理由。
- 若遇到真正无法在 C0 内安全决定的科学问题，将 C0 标为 BLOCKED，完整记录选项和影响；不要带着未冻结关键口径进入 C1。

最终回复必须包含：
- C0 是 PASS 还是 BLOCKED；
- 新增/修改文件的可点击路径；
- 样本角色计数和关键 protocol 决策；
- 执行的校验/测试及结果；
- commit SHA、registry/config/protocol 的 SHA-256、git clean 状态；
- 已知限制；
- 与 step-by-step 文档中完全一致的 C1 完整提示词。
```

### 2026-08-21 18:17 +0800 — C1 completion record

- status: PASS
- objective_completed: 只实现 C1 的共同数据合同。新增生产级严格、深不可变 Stage C config loader；新增统一 MethodResult、11 种结构化 NA、显式 raw/metric-ready/display 边界；新增 signed RGB channel sum、224×224↔14×14 和 4×4 cell 的质量守恒投影；新增 Pearson、average-rank Spearman、base-2 JSD、双向 base-2 KL、top-10%/20% Jaccard 和 normalized entropy。没有实现或运行 C2–C5 方法、fidelity、扰动或 GPU attribution。
- git_sha_before: 05b0290b7caacc772a7c20a724f1491b3057524a
- git_sha_after: SELF（即包含本 completion record 的单一 C1 commit；精确 SHA 在提交后由 git rev-parse HEAD 产生并在最终回复报告）
- tree_clean: true（C1 commit 后复验）
- files_added_or_changed: src/vitnns/stage_c_config.py；src/vitnns/stage_c_maps.py；src/vitnns/stage_c_metrics.py；tests/test_stage_c_config.py；tests/test_stage_c_maps.py；tests/test_stage_c_metrics.py；tests/test_stage_c_contract.py；docs/阶段C-step-by-step.md。
- inputs_and_sha256: C0 commit 05b0290b7caacc772a7c20a724f1491b3057524a；Stage C registry 9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0；Stage A 500 manifest 042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89；Stage B formal status 2129134ff256fe9421a599e420ac2f79a19342b005d0f0ce34ecdba670d7a846；Stage B manifest 57b2ff7e3189c086650d6680252e16373617260e14f404b9ba78223c3963c7eb，status→manifest→artifact hash/bytes、commit 30732c9...、formal=true、dirty=false 全部通过。
- configs_and_sha256: frozen Stage C config 0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669；protocol ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8；acceptance f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd。冻结 config、registry、角色、方法目标、阈值和 decision gates 均未修改。
- artifacts_and_absolute_paths: 无 runtime/GPU artifact。生产源码位于 /data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_c_config.py、stage_c_maps.py、stage_c_metrics.py。
- artifact_manifest_or_status_hashes: 不适用；C1 不生成实验 artifact。
- tests_and_results: 启动时首次裸 conda 因非交互 shell 的 PATH 无 conda 而 exit 127，未执行测试、未写文件；随后用 /data00/home/yeziqi.831/miniforge3/bin/conda run -n vitnns。C0 clean/tracked validator PASS；C1 focused 最终 78 passed；C0/Stage B config/display/retention focused 67 passed；提交前全量非 GPU 184 passed, 1 deselected；python -O production loader smoke PASS。提交后仍须重跑 clean/tracked validator、C1 focused 和全量非 GPU；实际结果在最终回复报告。
- acceptance_results: C1 六项通过。严格 loader 拒绝 duplicate、NaN/Infinity、unknown/missing、bool-as-int、类型/枚举/范围/顺序/重复/引用/跨字段非法组合，且 frozen config exact round-trip 不注入默认；MethodResult 绑定方法版本、两条 comparison line、explanation/fidelity target、native/metric map、注册 projection、sign、mask、NA、provenance，并阻止 class-agnostic class explanation；projection 对正/负/零/随机 map 在 atol=rtol=1e-6 下守恒并做 patch-mass round-trip；指标用 float64 累加且符合 base-2/average-rank/ceil/row-major/epsilon 规则；display/raw/裸数组不能作为指标输入；Stage A/B 非 GPU 回归通过。
- deviations_or_protocol_amendments: 无 protocol amendment。method_version 是每个运行时 MethodResult 的必填字段，不向 frozen config 注入字段；metric-ready/L1 为 float32、指标公式内部累加为 float64；sufficiency_gap_20=1-sufficiency_20 留给 C5 fidelity 实现，C1 未越界实现。
- failures_retained: 保留上述首次 conda PATH 失败；开发中近常量测试初始构造未达到冻结相对阈值而出现 1 个测试失败，修正测试构造后通过。独立 review 曾误报 JSD 使用自然对数；代码和手算测试实际使用 np.log2 且互斥分布 JSD=1，因此未据此改公式。
- known_limitations: C1 只提供共同合同，不生成任何方法 map；patch→pixel/fine 是质量守恒均分，不是原生像素证据；结构化 degenerate_direction 和目标方向 u=(g(x)-g(x0))/||g(x)-g(x0)||₂ 的实际计算留给 C4，阈值仍是冻结相对 1e-6；fidelity/扰动/runner/selection 尚未实现。
- next_task: C2

#### 下一对话完整提示词

```text
你现在只执行 ViT-NNS 重建工程“阶段 C”的 C2：实现 direct raw CLS attention 与 attention rollout。不要开始 C3，不要实现 Grad-CAM、full-model class-logit IG 的 Stage C 适配、head-level NNS、matched patch baseline、fidelity、扰动稳定性、runner 或批量实验，也不要运行昂贵 GPU attribution。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

前置冻结基线：
- C1 commit: SELF，即包含本 completion record 的当前 clean HEAD；启动 C2 时先运行 git rev-parse HEAD，其精确 SHA 应与 C1 最终回复给出的 commit SHA 一致。
- C0 commit: 05b0290b7caacc772a7c20a724f1491b3057524a
- Stage C registry SHA-256: 9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0
- Stage C config SHA-256: 0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669
- Stage C protocol SHA-256: ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8
- Stage C acceptance SHA-256: f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd
- Stage A 500 manifest SHA-256: 042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89

第一步必须完整阅读：
1. docs/阶段C-step-by-step.md，尤其 C0/C1 completion record 与 C2 边界
2. docs/阶段C_protocol.md
3. docs/stage_c_acceptance.md
4. docs/stage_c_config_contract.md
5. config/stage_c.json 与 config/stage_c.schema.json
6. src/vitnns/stage_c_config.py
7. src/vitnns/stage_c_maps.py
8. src/vitnns/stage_c_metrics.py
9. src/vitnns/extractor.py、src/vitnns/modeling.py 和相关 Stage A extractor/parity 测试
10. docs/阶段A.md、docs/stage_a_acceptance.md、docs/阶段B.md、docs/stage_b_acceptance.md
11. README.md

开始前必须：
- 检查 git status 和 HEAD，要求 HEAD 等于 C1 最终回复 SHA 且工作树 clean；若不一致，查明归属，不覆盖用户改动。
- 运行 conda run -n vitnns python tests/test_stage_c_contract.py --require-tracked --require-clean，核验冻结 hash、500 图官方标签链与 Stage B formal status→manifest→artifact 链。
- 运行 C1 focused tests，确认严格配置、map/projection/metrics 合同仍通过。
- 不从 __pycache__/.pyc 恢复任何 Stage C 源码，不读取 display PNG，不复用旧数值结果。

C2 范围与交付物：
1. 实现 direct raw CLS attention，必须直接消费真实 NativeViTExtractor 捕获的 attention_used，不得手工重演 Transformer block、QK 或 softmax。
2. 验证 eval 下 extractor 的 attention 与 attention_used 语义；direct 数值必须与模型实际使用值精确一致。
3. Value-path 线的 per-head direct 必须使用匹配的 block/head；class-logit 线 direct 固定 block 11、先对 12 heads 算术平均，再删除 CLS key，并把 196 patch mass 重新归一到 1。分母非正或非有限必须 fail closed。
4. 实现 attention rollout：每层 B_l=row_norm(I+mean_h A_lh)，顺序严格为 B_11 B_10 ... B_0；取 CLS→patch、排除 CLS、归一到 1。不得偷偷改为其他 residual weight、head fusion 或乘法方向。
5. 两种方法都必须输出 C1 MethodResult，使用冻结 method ID/version、comparison line、null explanation target、正确 fidelity target、nonnegative sign semantics、metric-ready float32 14×14 map、全有效 mask和 provenance。class-agnostic map 不得冒充类别解释；per-head raw attention 只是不同目标的 spatial structure similarity，不是 Value-path fidelity。
6. 测试必须覆盖：手算 toy matrices、CLS key 排除、归一化、head mean 顺序、row normalization、identity/residual、rollout 乘法顺序、单层退化、非法 shape/非有限/零或负 mass、错误 block/head、method contract、hook/model mode 在成功与异常后恢复，以及低成本真实 extractor parity。不得保留 raw attention 的全笛卡尔积。
7. 真实模型测试只做验收所需的低成本 smoke/parity；显式禁用 GPU 或用 CPU，不运行 attribution，不处理 20 图 pilot。
8. 更新 docs/阶段C-step-by-step.md：C2 标 PASS 或 BLOCKED；若 PASS 将 C3 标 NEXT；追加完整 completion record 和“新对话执行 C3”的完整可复制提示词。
9. 检查 diff，只提交 C2 tracked 文件；提交后重跑 C0 validator（含 tracked/clean）、C1+C2 focused tests和全量非 GPU suite，最终确认 clean。

禁止事项：
- 不修改冻结 registry、config、protocol、acceptance、角色、阈值、目标或 decision gates；确有矛盾必须停止并走带旧/新 hash 的 protocol amendment。
- 不实现 Grad-CAM、full-model class-logit IG Stage C wrapper、head-direction NNS、matched patch contribution、fidelity/扰动/runner/selection。
- 不把 raw attention 或 rollout 称为类别归因、完整 attention attribution 或与 Value-path NNS 同目标的 fidelity。
- 不把 14×14 patch map 的质量守恒上采样称为原生像素证据。
- 不运行高成本 GPU attribution或 C8/C9 数据。

验收标准：
- docs/stage_c_acceptance.md 的 C2 四项全部通过；
- direct 从 attention_used 得到并精确一致，CLS 排除/head 规则/归一化正确；
- rollout 行和为 1、数值有限、手算乘法方向正确；
- C1 MethodResult/map/metric/display 边界不被绕过；
- hook、model mode 和捕获状态在成功/异常后恢复；
- 全量非 GPU pytest 通过，没有运行 GPU attribution；
- 一任务一提交，最终工作树 clean，completion record 的 commit/hash/test 信息完整，文档写得简单易懂。
```

### 2026-08-21 21:10 +0800 — C2 completion record

- status: PASS
- objective_completed: 只实现 C2 的 direct raw CLS attention 与 attention rollout。新增三个版本 1.0.0 的 C1 MethodResult 入口：匹配 block/head 的 Value-path 结构参考、block 11 的 12-head 均值 class-logit ranking baseline，以及 blocks 0–11 的 mean-head residual rollout。生产入口只消费 NativeViTExtractor 的 attention_used，不重演 block、QK 或 softmax；同时加固 extractor 的逐次 eval/unfused 检查和注册期/forward 期 hook 清理。没有实现或运行 C3–C10 方法、fidelity、扰动、runner、批量实验或 GPU attribution。
- git_sha_before: bd518afc7a479fb2c5dcf5ede31287d2d514147e
- git_sha_after: SELF（即包含本 completion record 的单一 C2 commit；精确 SHA 在提交后由 git rev-parse HEAD 产生并在最终回复报告）
- tree_clean: true（C2 commit 后复验）
- files_added_or_changed: src/vitnns/stage_c_attention.py；src/vitnns/extractor.py；tests/test_stage_c_attention.py；tests/test_stage_c_contract.py（推进活文档状态断言为 C2 PASS/C3 NEXT，并将 C2 源码/测试加入 tracked 检查）；docs/阶段C-step-by-step.md。
- inputs_and_sha256: C1 commit bd518afc7a479fb2c5dcf5ede31287d2d514147e；C0 commit 05b0290b7caacc772a7c20a724f1491b3057524a；Stage C registry 9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0；Stage A 500 manifest 042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89；启动 validator 再次通过 Stage B formal status→manifest→artifact hash/bytes、commit 30732c9...、formal=true、dirty=false 链。
- configs_and_sha256: frozen Stage C config 0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669；protocol ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8；acceptance f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd。registry/config/protocol/acceptance/schema、角色、阈值、目标和 decision gates 均未修改；五个冻结文件的实际 SHA-256 已逐字节复算一致。
- artifacts_and_absolute_paths: 无 runtime/GPU artifact。生产实现位于 /data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_c_attention.py；只生成内存中的最终 float32 14×14 MethodResult，不写 raw-attention 全组合。
- artifact_manifest_or_status_hashes: 不适用；C2 不生成实验 artifact、manifest 或 status。
- tests_and_results: 启动 C0 clean/tracked validator PASS；启动 C1 focused 78 passed。C2 focused 最终 51 passed；C1+C2+Stage A extractor focused 134 passed；纳入 C0 validator 后最终 focused 139 passed；提交前全量非 GPU（CUDA_VISIBLE_DEVICES 为空，排除 unsafe_pickle）235 passed, 1 deselected。真实模型验收仅用 CPU canonical ViT 单次低成本 extraction，12 层 attention 与 attention_used 均 torch.equal，两个 direct 结果与 attention_used 手算逐元素相等。提交后仍须重跑 clean/tracked validator、focused tests和全量非 GPU suite；实际结果在最终回复报告。
- acceptance_results: C2 四项全部通过。direct 的唯一生产来源是 LayerCapture.attention_used；per-head 精确匹配 block/head，class-line 固定 block 11 且先算术平均 12 heads，两者删除 CLS key 后归一 196 patch，非法 shape/index/dtype/非有限/负值/非正 mass fail closed。rollout 严格计算每层 row_norm(I+mean_h A_lh)，升序遍历时左乘得到 B_11...B_0；toy 覆盖 head fusion、identity、row normalization、不交换乘法顺序、单层退化和后层实际参与，结果有限、非负、行和为 1。三种结果均通过 C1 method/target/projection/sign/map/mask/provenance 合同。成功、forward 异常、hook 注册异常、构造后 train、顶层 eval 但子模块 train 均验证 hook key 与逐模块 mode 不泄漏或被改写；不保留 raw attention 笛卡尔积。
- deviations_or_protocol_amendments: 无 protocol amendment。方法实现版本固定为运行时 1.0.0，不写回冻结 config。纯 rollout toy core 允许较小的 head/token 数以便手算，生产 wrapper 仍严格要求 12 blocks × 12 heads × 197 tokens。attention baseline 的 native/metric map 都是同一 14×14 patch mass；没有把质量守恒上采样称为原生像素证据。
- failures_retained: 普通 shell/conda 首次因运行环境缺少 bubblewrap 而未启动，随后按批准在沙箱外运行；没有测试或文件副作用。C2 focused 首轮为 50 passed, 1 failed：测试把两次 float32 归一化与代数约简后的一次归一化要求 bitwise 相等，最大差 9.31e-10；改为沿冻结 B_l 数值路径手算后 51 passed，未修改算法。独立审查指出“全 12 层”测试原先只有 block 0 非 identity、mixed child mode 未单独覆盖；补入非 identity block 11 和顶层 eval/子模块 train 用例后 focused、联合及全量回归仍全部通过。文档更新后的首次 pre-commit validator 因 tests/test_stage_c_contract.py 仍断言 C2 NOT_STARTED 而失败；按交接协议将该非科学静态断言推进为 C2 PASS/C3 NEXT，并增加 C2 record/C3 prompt 检查后通过，冻结 hash 未变。
- known_limitations: direct 与 rollout 都是 class-agnostic attention baseline，不是类别归因或完整 attention attribution；per-head direct 与 Value-path NNS 只支持 spatial structure similarity，不是同目标 fidelity。14×14 map 是 patch mass，不是原生像素证据。C2 provenance 只是供 C6 扩展的不可变 fragment，不是完整 run provenance；ExtractionResult 是仓库内部受信任类型边界，不是密码学来源证明；extractor 不声明并发/重入线程安全。Grad-CAM、full-model class-logit IG Stage C wrapper、head-direction NNS、matched patch、fidelity、扰动和 runner 尚未实现。
- next_task: C3

#### 下一对话完整提示词

~~~text
你现在只执行 ViT-NNS 重建工程“阶段 C”的 C3：实现 ViT Grad-CAM 与 full-model class-logit IG 的 Stage C 适配。不要开始 C4，不要实现 head-level NNS、matched patch baseline、fidelity、扰动稳定性、runner、selection 或批量实验，也不要运行昂贵 GPU attribution。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

前置冻结基线：
- C2 commit: SELF，即包含本 completion record 的当前 clean HEAD；启动 C3 时先运行 git rev-parse HEAD，其精确 SHA 应与 C2 最终回复给出的 commit SHA 一致。
- C1 commit: bd518afc7a479fb2c5dcf5ede31287d2d514147e
- C0 commit: 05b0290b7caacc772a7c20a724f1491b3057524a
- Stage C registry SHA-256: 9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0
- Stage C config SHA-256: 0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669
- Stage C protocol SHA-256: ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8
- Stage C acceptance SHA-256: f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd
- Stage A 500 manifest SHA-256: 042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89

第一步必须完整阅读：
1. docs/阶段C-step-by-step.md，尤其 C0–C2 completion record 与 C3 边界
2. docs/阶段C_protocol.md
3. docs/stage_c_acceptance.md
4. docs/stage_c_config_contract.md
5. config/stage_c.json 与 config/stage_c.schema.json
6. src/vitnns/stage_c_config.py、stage_c_maps.py、stage_c_metrics.py、stage_c_attention.py
7. src/vitnns/attribution.py、targets.py、extractor.py、modeling.py，以及相关 attribution/hook/mode/grad 测试
8. tests/test_stage_c_attention.py 和 C1 focused tests
9. docs/阶段A.md、docs/stage_a_acceptance.md、docs/阶段B.md、docs/stage_b_acceptance.md
10. README.md

开始前必须：
- 检查 git status 和 HEAD，要求 HEAD 等于 C2 最终回复 SHA 且工作树 clean；若不一致，查明归属，不覆盖用户改动。
- 运行 conda run -n vitnns python tests/test_stage_c_contract.py --require-tracked --require-clean，核验冻结 hash、500 图官方标签链与 Stage B formal status→manifest→artifact 链。
- 运行 C1+C2 focused tests，确认严格配置、MethodResult/map/metric/display、direct/rollout 和 extractor lifecycle 合同仍通过。
- 不从 __pycache__/.pyc 恢复源码，不读取 display PNG，不复用旧数值结果。

C3 范围与交付物：
1. 实现 ViT Grad-CAM，方法 ID 必须为 vit_grad_cam_block11_norm1。目标层固定真实模型 blocks[11].norm1；目标固定为 clean 输入 argmax 的 predicted-class raw logit，并在该次方法调用内不改变。
2. Grad-CAM 必须删除 CLS token，只对 196 patch tokens 处理；channel weight 是目标 logit gradient 在 196 patch tokens 上的算术平均；对加权 activation 做 channel sum后 ReLU；不得 min-max normalization、平滑、percentile enhance 或 display resize。最终是有限、非负 float32 14×14 map。
3. 正式包装已有真实 full_model_ig，精确名称/ID 为 full-model class-logit IG / full_model_class_logit_ig，不得写成完整 attention attribution。使用同一个 frozen clean predicted-class raw logit、normalized-zero baseline、Gauss-Legendre float32 schedule [512,1024,2048,4096,8192]、chunk 8、atol 1e-4、rtol 0.01；不得重演模型或把 conditional fixed-attention IG 冒充 full-model IG。
4. full-model IG 保留 signed RGB float32 native attribution；先 signed RGB channel sum，再按 C1 的 16×16 sum 投影到 signed float32 14×14 metric map。不得先取绝对值或把质量守恒投影称为原生像素证据。
5. 两种方法都输出 C1 MethodResult，方法实现版本显式固定；comparison line 为 class_logit；explanation/fidelity target 都是 clean_predicted_class_raw_logit；class_agnostic=false。Grad-CAM sign 为 nonnegative、projection 为 identity_14x14；full-model IG sign 为 signed、projection 为 pixel_mass_to_patch_sum；使用全有效 mask和不可变、可 JSON 序列化 provenance fragment。
6. 同一 clean input 的 class index 必须由 raw logits argmax 一次确定并冻结；记录 target index、raw clean logit、official truth/correctness（若调用方提供）但不得因预测错误排除。不得在 baseline、IG quadrature 或任何干预中重新选择类别。
7. 测试必须覆盖：手算 toy Grad-CAM、CLS 排除、196-token gradient mean、ReLU、无 min-max、target sensitivity；full-model wrapper 的 frozen target/baseline/schedule/投影/符号；非法 shape/target/非有限/空或全零输出的 fail-closed 行为；MethodResult 合同；重复性；成功和异常后 hook、model/子模块 mode、parameter grad 与 requires_grad 精确恢复。已有调用者 hook/grad 不得被删除或覆盖。
8. 真实模型只做验收所需的低成本 CPU smoke/parity；显式禁用 GPU。不得为了 smoke 运行冻结 512–8192 steps 的昂贵真实 attribution；可用 toy/受控替身验证完整 schedule dispatch，用最小安全真实 forward/backward 检查目标层、shape、target sensitivity和 lifecycle。不得处理 20 图 pilot。
9. 更新 docs/阶段C-step-by-step.md：C3 标 PASS 或 BLOCKED；若 PASS 将 C4 标 NEXT；追加完整 completion record 和“新对话执行 C4”的完整可复制提示词。
10. 检查 diff，只提交 C3 tracked 文件；提交后重跑 C0 validator（含 tracked/clean）、C1–C3 focused tests和全量非 GPU suite，最终确认 clean。

禁止事项：
- 不修改冻结 registry、config、protocol、acceptance、角色、阈值、目标、schedule 或 decision gates；确有矛盾必须停止并走带旧/新 hash 的 protocol amendment。
- 不实现 head-direction NNS、matched patch contribution、fidelity、扰动、资源仪表、runner、CLI、selection、C8/C9 数据或统计决策。
- 不把 Grad-CAM 或 full-model class-logit IG 称为完整 attention attribution；不把 class-logit fidelity 用来证明 Value-path NNS 守恒；不把 patch→pixel 质量均分称为原生像素证据。
- 不读取 display PNG 作为数值输入，不运行高成本 GPU attribution。

验收标准：
- docs/stage_c_acceptance.md 的 C3 四项全部通过；
- 两种方法使用同一个 frozen clean predicted-class raw logit，target layer、Grad-CAM reduction/ReLU、IG baseline/schedule/projection 与冻结协议一致；
- 输出有限、非空、target-sensitive，重复运行达到配置容差；
- C1 MethodResult/map/metric/display 边界不被绕过；
- hook、全部 module mode、parameter grad 和 requires_grad 在成功/异常后精确恢复；
- 全量非 GPU pytest 通过，没有运行昂贵 GPU attribution或 20 图 pilot；
- 一任务一提交，最终工作树 clean，completion record 的 commit/hash/test 信息完整，文档写得简单易懂。

方向仍冻结为 u=(g(x)-g(x0))/||g(x)-g(x0)||₂，积分期间固定，相对退化阈值仍为 1e-6；实际方向计算属于 C4，本次不得提前实现。patch→pixel/fine 仅表示质量守恒均分，不是原生像素证据。
~~~
