# 阶段 C：step by step 执行与跨对话交接文档

> 文档性质：阶段 C 的路线图、任务状态表和跨对话交接日志。  
> 创建日期：2026-08-21。  
> 当前状态：阶段 A、阶段 B、阶段 C/C0–C9 已完成；C10 已完整执行但因冻结 Holm 合同不足而验收 `BLOCKED`，且唯一 primary decision 为 `NO_GO_NOVELTY`；Stage D 未启动。
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
| C3 | PASS | Grad-CAM 与 full-model class-logit IG | 两种方法适配、真实模型 smoke test | C1 |
| C4 | PASS | head 级 NNS 与 matched patch baseline | 同目标实现、守恒与退化测试 | C1 |
| C5 | PASS | fidelity、扰动稳定性、资源仪表 | fidelity/perturbation/stability/resource 模块 | C2–C4 |
| C6 | PASS | 单图/批量 runner、留存、provenance、CLI | 可恢复批量流水线及测试 | C1–C5 |
| C7 | PASS | 单图 smoke benchmark 与预算冻结 | benchmark、总预算和可运行性结论 | C6 |
| C8 | PASS | 5 图全层/全头校准和 selection 冻结 | calibration 产物、`selection.json` | C7 |
| C9 | PASS | 全 20 图冻结配置 pilot | 完整逐样本产物和 failure table | C8 commit clean |
| C10 | BLOCKED | 汇总统计和 Go/No-Go | per-sample 表、summary、decision、报告 | C9 |

### 8.1 增量回归策略（自 C5 起）

进入 `C_{i+1}` 时，不要求机械地再次运行 `C1–Ci` 的全部 focused tests。若同时满足以下条件，可直接复用 `Ci` completion record 中已经完成的提交后验证，把它记为“复用已验证基线”，而不能写成“本次重新通过”：

- 当前 `HEAD` 与 `Ci` completion record 的精确 commit SHA 一致，且工作树 clean；
- C0 clean/tracked validator 通过，冻结 registry/config/schema/protocol/acceptance、Stage A 500 manifest 和 Stage B 正式产物链均未漂移；
- `Ci` completion record 已记录同一提交上的累计 focused tests 与全量非 GPU suite 通过，且没有未解决失败、跳过项或环境/lock 变化；
- 本任务尚未修改共享代码，且没有其他证据提示历史合同可能回归。

每个后续任务采用以下最小测试集合：

1. 启动时运行 HEAD/clean 检查和 C0 validator；满足上述复用条件时，跳过累计历史 focused baseline。
2. 开发中运行本任务测试，以及所有被实际修改或直接依赖的历史模块测试。修改跨阶段共享代码时，必须补跑其所有受影响历史测试；无法可靠判断影响面时，回退到累计 focused tests。
3. 最终 tracked 状态只需运行一次本任务及受影响 focused tests、一次 C0 clean/tracked validator和一次全量非 GPU suite。全量 suite 已包含的历史 tests 不在其前后机械重复；只有 marker/条件 skip 使其未覆盖、失败定位或明确验收条款另有要求时才单独补跑。
4. completion record 必须区分“复用的上一提交证据”和“本任务实际执行的测试”，记录 commit、命令、结果、环境差异与 skip/deselect 情况；不得把复用证据伪装成本次执行。

出现 HEAD 不一致、dirty 归属不明、冻结 hash/依赖/lock/运行环境变化、上一记录缺失或失败、共享基础设施改动、异常 skip，或任何回归迹象时，不适用豁免，必须按影响面补跑；必要时恢复 `C1–Ci` 累计 focused tests。该策略只消除重复回归，不豁免当前任务验收、提交前全量非 GPU suite，也不允许运行昂贵 GPU attribution 来替代单元测试。

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

- 在正式 benchmark 前先通过下述“C7 启动前完整性性能门”；不得把诊断性完整性扫描的病态开销当成方法本身的生产成本；
- 在一个 audit 图上运行所有方法；
- NNS 至少覆盖预注册的早/中/晚层和若干头，以观察最坏收敛 steps；
- 记录每方法/配置的时间、峰值显存、调用数和输出字节；
- 用保守上界估计 C8、C9 和可能的阶段 D 成本；
- 只依据资源和数值有效性冻结预算，不依据热图美观选择配置；
- 生成 benchmark report；
- 更新本文并生成 C8 提示词。

验收：无 OOM、无残留 hook/grad、预算估算可复算；若超预算则标记 `BLOCKED` 或提交 protocol amendment，不能静默减少层/头/样本。

C7 已完成。逐点/逐 scalar 的全模型内容哈希已移到 model-load 与 execution-unit 边界，热路径使用廉价 live-state guard；最终有效 NNS diagnostic 为 `36.172419351991266` 秒，严格低于 `720` 秒。随后在同一 clean code commit 和全新 root 上完成单个 audit 图的全部 7 方法、12 个显式 Value candidates、40 个 units；正式 report 为 `PASS`，40/40 units 有效且全部 10 项资源 hard caps 通过。C7 未读取 holdout、未运行 5 图/20 图、未生成 selection，也未执行 bootstrap 或科学决策。完整证据和预算数值见 C7 completion record。

#### C7 启动前完整性性能门（2026-08-22 冻结）

一次未完成的 C7 诊断运行发现，原实现把“防止模型在一次调用中被篡改”的全模型内容签名放进了 fidelity scalar 热路径。首个完整 NNS unit 用时 `1513.779444482061` 秒，执行 `1274` 次 batched forward、`320` 次 batched backward，但只写出 `4171728` bytes payload；其中 `44 × 21 = 924` 个 fidelity curve points 会各自触发多次 `_model_state_signature()`。该函数把约 8657 万个 float32 参数和 buffers 从 GPU 搬到 CPU 后逐字节 SHA-256；仅按每点四次、只计 parameters，就重复扫描约 `1.28 TB`（约 `1.16 TiB`）。实测表现为 CPU 多核繁忙而 H20 利用率很低。因此这是完整性实现的病态放大，不是论文方法所需的科学计算，也不是写盘造成的。

下一次 C7 在启动唯一正式单图 benchmark 前，必须先完成以下窄范围工程修正；这不授权修改 registry/config/schema/protocol/acceptance、方法公式、目标、阈值、IG schedule、seed、扰动、fidelity 曲线、repeat、候选、资源 cap 或 decision gate：

1. canonical model artifact 在加载时仍校验冻结的 state-dict SHA-256；每个 execution unit 的入口和成功/异常出口仍对参数与 buffers 做强内容签名并比较，任何持久变化都必须使该 unit fail closed。unit 边界还必须核验 hooks、全部 module mode、parameter `grad`/`requires_grad` 和 fused-attention 状态均恢复。
2. **不得在每个 fidelity point、每次 frozen scalar evaluation、每个 IG quadrature chunk 或同一 map 的内部循环中，把全模型参数/buffers 搬到 CPU 并重新做内容 SHA-256。** 热路径可检查 model/tensor object identity、`data_ptr`、shape、dtype、device 和 PyTorch `_version` 等廉价状态；完整字节校验集中在 model-load 与 execution-unit 边界。
3. fidelity intervention tensor 仍须有可复核的内容 hash，但 21 张 schedule 的整组内容只在构造/曲线边界强校验，不得在每个点的前后重复扫描整组 21 张 tensor。点内可检查当前 tensor 的身份、版本和预先绑定的 digest；curve 结束时若强校验不一致则整条曲线及其 unit 失败。
4. 该边界化方案的威胁模型是可信本地研究代码中的意外持久 mutation；不为“某个调用恶意改写全部权重、使用篡改权重计算、再逐字节恢复且不留下版本变化”的对抗行为付出逐点全模型扫描成本。若未来要求抵御这种攻击，必须单独定义安全运行环境，不能把其成本混入论文生产 benchmark。
5. 必须新增成功与异常路径测试：参数/buffer 持久 mutation、hook/mode/grad/`requires_grad` 残留均被 unit 出口发现；正常重复 scalar evaluation 数值等价；完整模型字节扫描次数按 process-load/execution-unit 数量增长，不能按 curve point、IG chunk 或 logical scalar evaluation 数量增长。测试须直接 instrument full-content signature 调用数，不能只凭代码审阅。
6. 资源观测至少拆分 attribution/fidelity 模型计算、integrity checking、artifact finalize 三类 phase timer，并继续报告包含必要生产校验的同步总 wall time。不得从预算中扣除 unit 边界的必要校验；也不得把 logical scalar evaluations 误记为 batched model invocations。
7. 在新的独立诊断目录先运行一个预注册 audit NNS unit 的完整 frozen workload。由 Stage D `8000` units、`1.25` safety factor 和 `2000 H20-hours`/`4 hours per image` 两项既有 cap 推得，最慢有效 unit 必须不高于 `720` 秒；若该已知昂贵代表 unit 仍不低于 `720` 秒，不启动 40-unit 正式 attempt，而是继续做不改变数值语义的工程优化，或据实将 C7 标为 `BLOCKED`。这只是提前发现必然超 cap，不替代最终全部显式 candidates 的最慢实测上界。
8. 禁止用减少方法、层、头、steps、curve points、scale、direction、sanity ranking、repeat 或扰动来通过该门；批量化/缓存只有在固定输入上的结果、顺序无关统计、调用计数语义和容差通过等价测试后才可采用。

诊断历史必须保留但不得混入新的正式报告：诊断代码 commit `357dd9d4a8b6dd2d43ff5bfbc85cc93a68145226` 已保存在本地引用 `archive/c7-overhash-357dd9d`；`main` 已回到 C6 commit `53368aa08c68625c33bc59541a6b8a0216acfff4` 后再加入本文约束。`outputs/stage_c_c7_audit_00000079_h20_attempt_20260822`、`..._retry01`、`..._retry02` 均保留，最后一份在用户授权停止时为 15 个 status、13 个完整 result；它们只证明过度哈希热点和实现失败史，不构成 C7 completion、不冻结预算，也不得 resume、覆盖或与新实现的计时拼接。该诊断没有运行 5 图 calibration、20 图 pilot、holdout、selection、bootstrap 或统计决策。

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

### 2026-08-21 22:15 +0800 — C3 completion record

- status: PASS
- objective_completed: 只实现 C3 的两个 class-logit 方法适配。新增一次性解析并冻结 clean predicted-class raw logit 的运行时 target context；新增固定真实 `blocks[11].norm1` 的 ViT Grad-CAM；正式包装既有真实 `full_model_ig` 为精确名称 `full-model class-logit IG`。两者共享同一个 frozen class index，均输出 C1 `MethodResult`。没有实现或运行 C4–C10 的 head-direction NNS、matched patch、fidelity、扰动、资源仪表、runner、selection、批量实验或统计决策。
- git_sha_before: 3ddc5b7d86d85729a644be49791aebb06f5285b6
- git_sha_after: SELF（即包含本 completion record 的单一 C3 commit；精确 SHA 在提交后由 `git rev-parse HEAD` 产生并在最终回复报告）
- tree_clean: true（C3 commit 后复验）
- files_added_or_changed: `src/vitnns/stage_c_class_logit.py`；`tests/test_stage_c_class_logit.py`；`tests/test_stage_c_contract.py`（只推进活文档状态/交接断言并把 C3 源码、测试加入 tracked 检查）；`docs/阶段C-step-by-step.md`。
- inputs_and_sha256: C2 commit 3ddc5b7d86d85729a644be49791aebb06f5285b6；C1 commit bd518afc7a479fb2c5dcf5ede31287d2d514147e；C0 commit 05b0290b7caacc772a7c20a724f1491b3057524a；Stage C registry 9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0；Stage A 500 manifest 042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89；启动 validator 再次通过 Stage B formal status→manifest→artifact hash/bytes、commit 30732c9...、formal=true、dirty=false 链。
- configs_and_sha256: frozen Stage C config 0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669；protocol ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8；acceptance f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd。registry/config/protocol/acceptance/schema、角色、阈值、目标、schedule 和 decision gates 均未修改；五个冻结文件实际 SHA-256 已复算一致。
- artifacts_and_absolute_paths: 无 runtime/GPU artifact、manifest 或 status。生产实现位于 `/data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_c_class_logit.py`；测试位于 `/data00/home/yeziqi.831/testcode/vitnns_rebuild/tests/test_stage_c_class_logit.py`。测试只在内存中生成 toy attribution 和 canonical ViT CPU Grad-CAM map。
- artifact_manifest_or_status_hashes: 不适用；C3 不写实验 artifact。没有读取 display PNG、旧数值结果或 20 图 pilot。
- tests_and_results: 启动 C0 clean/tracked validator PASS；启动 C1+C2 focused 149 passed。C3 非集成最终 46 passed, 1 deselected；canonical ViT CPU-only smoke 1 passed，覆盖真实 block 11 norm1、predicted/alternate target sensitivity 和 lifecycle；C3 与 attribution/MethodResult 联合回归通过。提交前及提交后的 validator、C1–C3 focused 和全量非 GPU 最终计数在本任务最终回复报告。全程 `CUDA_VISIBLE_DEVICES=''`，未运行真实 512–8192-step IG 或任何 GPU attribution。
- acceptance_results: C3 四项全部通过。两种 wrapper 只接受同一 resolver 签发、绑定 clean tensor 与同一未变模型状态的 frozen target；只在 resolver 内对 clean raw logits 执行一次 argmax，并保留 target index、clean raw logit、调用方提供的 official truth/correctness，预测错误不排除。Grad-CAM 固定 `blocks[11].norm1`，同时删除 activation/gradient 的 CLS，只在 196 patch tokens 上算 gradient channel mean，weighted activation channel sum 后 ReLU，无 min-max、平滑、percentile 或 resize，得到有限非负非零 float32 14×14 map。full-model wrapper 精确调用现有 `full_model_ig`，显式传 normalized-zero baseline、Gauss-Legendre float32 `[512,1024,2048,4096,8192]`、chunk 8、atol 1e-4、rtol 0.01；保留 signed RGB float32，再 signed channel sum 和 16×16 sum projection，不取绝对值。成功/异常/注册失败后 caller hooks、逐 module mode、parameter `requires_grad` 和原 `.grad` 对象/数值/version/inference 属性均恢复；MethodResult、全有效 mask和不可变 JSON provenance 通过。
- deviations_or_protocol_amendments: 无 protocol amendment。config 中 `native_map=signed_float32_224x224_rgb_channel_sum` 按 C1 强类型合同和 C3 明确提示解释为 signed RGB native attribution 的注册 map reduction：`MethodResult.native_tensor` 保留 `(3,224,224)`，随后做 signed channel sum；未修改冻结文字。C3 为防止 target 漂移增加运行时同一 model/input/state 绑定；该运行时绑定明确不是 canonical model artifact hash，完整跨运行 provenance 仍留给 C6。真实 full-model IG 高步数未为 smoke 执行；冻结 schedule dispatch、result consistency 和投影由 toy/受控替身完整验证，真实完整模型路径继续由既有 `full_model_ig` 原语测试覆盖。
- failures_retained: 开发首版 35 个 toy 测试和 1 个 CPU smoke 通过。独立审查随后发现 target 可手工伪造、`.data` 修改可能漏检、IG attempt/result 内部一致性不足，以及 caller `.grad` version/inference tensor 会被无条件 `copy_`；全部加固并增加回归测试，最终复审 high=0、medium=0。加固编辑中一次 `py_compile` 捕获布尔表达式漏写连接符，立即修正；两轮测试曾因 fail-closed 校验顺序变严而仅有预期异常消息断言不匹配（分别 1 failed/37 passed 和 2 failed/37 passed），更新测试期望而未放松算法，最终 46 passed, 1 deselected。没有 GPU/OOM/真实 attribution 失败，也没有写出实验 artifact。
- known_limitations: C3 只提供方法级不可变 provenance fragment，不是 C6 的完整 run provenance/status/manifest。target 的 live model 内容摘要用于单进程防漂移，不代替冻结 state-dict SHA-256；guard 不声明并发/重入线程安全。真实 ViT 只做两次低成本 Grad-CAM forward/backward，未运行昂贵 full-model IG schedule；C7 才测正式资源。Grad-CAM 和 full-model class-logit IG 都不是完整 attention attribution；signed pixel→patch 是质量守恒投影，不是把 patch 展开称为原生像素证据。head-direction `u=(g(x)-g(x0))/||g(x)-g(x0)||₂`、相对 1e-6 退化判断和 matched contribution 均未提前实现。
- next_task: C4

#### 下一对话完整提示词

~~~text
你现在只执行 ViT-NNS 重建工程“阶段 C”的 C4：实现 head-direction fixed-clean-attention Value-path IG 与 matched patch-token contribution。不要开始 C5，不要实现 fidelity、扰动稳定性、资源仪表、runner、CLI、selection 或批量实验，也不要运行昂贵 GPU attribution。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

前置冻结基线：
- C3 commit: SELF，即包含本 completion record 的当前 clean HEAD；启动 C4 时先运行 git rev-parse HEAD，其精确 SHA 应与 C3 最终回复给出的 commit SHA 一致。
- C2 commit: 3ddc5b7d86d85729a644be49791aebb06f5285b6
- C1 commit: bd518afc7a479fb2c5dcf5ede31287d2d514147e
- C0 commit: 05b0290b7caacc772a7c20a724f1491b3057524a
- Stage C registry SHA-256: 9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0
- Stage C config SHA-256: 0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669
- Stage C protocol SHA-256: ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8
- Stage C acceptance SHA-256: f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd
- Stage A 500 manifest SHA-256: 042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89

第一步必须完整阅读：
1. docs/阶段C-step-by-step.md，尤其 C0–C3 completion record 与 C4 边界
2. docs/阶段C_protocol.md，尤其第 4 节 head-level target、matched token 分解与退化规则
3. docs/stage_c_acceptance.md 的 C4 四项
4. docs/stage_c_config_contract.md
5. config/stage_c.json 与 config/stage_c.schema.json
6. src/vitnns/stage_c_config.py、stage_c_maps.py、stage_c_metrics.py、stage_c_attention.py、stage_c_class_logit.py
7. src/vitnns/attribution.py、targets.py、extractor.py、modeling.py
8. src/vitnns/stage_b_runner.py 中现有单-channel Value-path 调用边界，以及相关 attribution/intervention/hook/mode/grad 测试
9. tests/test_stage_c_attention.py、tests/test_stage_c_class_logit.py 和 C1 focused tests
10. docs/阶段A.md、docs/stage_a_acceptance.md、docs/阶段B.md、docs/stage_b_acceptance.md
11. README.md

开始前必须：
- 检查 git status 和 HEAD，要求 HEAD 等于 C3 最终回复 SHA 且工作树 clean；若不一致，查明归属，不覆盖用户改动。
- 运行 `conda run -n vitnns python tests/test_stage_c_contract.py --require-tracked --require-clean`，核验冻结 hash、500 图官方标签链与 Stage B formal status→manifest→artifact 链。
- 运行 C1–C3 focused tests，确认严格配置、MethodResult/map/metric/display、direct/rollout、class-logit 方法和完整 lifecycle 合同仍通过。
- 不从 `__pycache__`/`.pyc` 恢复源码，不读取 display PNG，不复用旧数值结果。

C4 范围与交付物：
1. 实现精确方法 ID `conditional_fixed_clean_attention_head_direction_value_path_ig`，精确名称 `NNS fixed-clean-attention head-direction Value-path IG`；实现精确方法 ID `matched_fixed_clean_attention_patch_token_contribution`，精确名称 `matched fixed-clean-attention patch-token contribution`。两者属于 value_path 比较线，必须解释同一个 `fixed_clean_attention_cls_head_direction` 标量。
2. 对 clean normalized input `x`、normalized-zero baseline `x0`、block `l`、head `h`，query 固定 CLS index 0。必须从真实 forward 的 `attention_used` 捕获 clean `A_lh(x)`，并使用真实上游网络得到的 `V_lh(x)` 与 `V_lh(x0)`；不得手工重演 Transformer block、QK 或 softmax。baseline 端必须计算 `g_lh(x0)=A_lh(x)@V_lh(x0)`，不得换成 `A_lh(x0)@V_lh(x0)`。
3. 方向严格为：`g_lh(z)=sum_j A_lh(x)[0,j] V_lh(z)[j]`，`d=g(x)-g(x0)`，`r=||d||₂`，`u=d/r`，`f_lh(z)=u^T g_lh(z)`。`A(x)`、`d`、`u` 必须在积分前 detach、clone 并冻结，quadrature 中不得重算、重选或翻转方向。方向阈值严格为 `tau_dir=1e-6*max(1,||g(x)||₂,||g(x0)||₂)`；非有限或 `r<=tau_dir` 返回/记录 `DEGENERATE_DIRECTION`，不得换 channel、head、baseline 或方向。
4. NNS IG 必须沿真实模型上游 Value path 对同一 `f_lh` 做 Gauss-Legendre float32 schedule `[512,1024,2048,4096,8192]`、chunk 8、atol 1e-4、rtol 0.01，从小到大取首个 conservation pass，失败时保留全部 attempts。clean 与 baseline 两个 endpoint intervention parity 最大绝对误差都必须 `<1e-6`：clean parity 比较 natural clean forward 与 clean-A replay；baseline parity 必须比较 fixed-clean-A intervention 捕获的内部 `g(x0)` 与独立使用同一 `A(x)@V(x0)` 计算的 reference，绝不能把 natural baseline 的 `A(x0)` 与 fixed `A(x)` 输出硬比较。conservation 必须满足 `abs_error <= 1e-4 + 0.01*abs(delta)`。不得把现有单-channel Stage B target直接改名冒充 head-direction 方法。
5. matched token contribution 必须对全部 197 tokens 计算 `c_j=A_lh(x)[0,j] * u^T(V_lh(x)[j]-V_lh(x0)[j])`。要求 `abs(sum_j c_j-r)<=1e-6+1e-5*abs(r)`；`c_0` 必须单独保留，只有 `c_1..c_196` 按原 patch 顺序 reshape 14×14。不得给 CLS contribution 伪造空间位置。记录 `s_patch=sum_{j=1..196}|c_j|/(|c_0|+sum_{j=1..196}|c_j|)`；零分母 invalid，`s_patch<0.01` 标记 `DEGENERATE_PATCH_SUPPORT`。patch 正证据占全部 token L1 小于 0.01 标记 `DEGENERATE_POSITIVE_PATCH_EVIDENCE`；这些状态不得被静默剔除。
6. 两种方法都输出 C1 MethodResult，版本显式固定、comparison_line=`value_path`、explanation/fidelity target 都是 `fixed_clean_attention_cls_head_direction`、class_agnostic=true、sign=signed、全有效 mask和不可变 JSON provenance。NNS 保留 signed RGB float32 native attribution，先 signed channel sum 再 16×16 sum 得到 metric map，projection=`pixel_mass_to_patch_sum`。matched 方法保留 signed 14×14 patch contribution map并单独记录 CLS scalar，注册 projection=`patch_mass_to_pixel_uniform`；patch→pixel/fine 只能称质量守恒均分，不是原生像素证据。
7. 支持 12 blocks × 12 heads 的选择性、分块或批量计算接口，但本任务不得运行 5 图/20 图或 C8 selection。不得保存每个 quadrature step、每像素或每维文件。保留 Stage B `conditional_fixed_attention_value_path_ig` 单 head-channel 方法作为兼容/审计路径，不篡改既有 Stage B 产物语义。
8. 测试必须覆盖：解析 synthetic head-direction、固定 clean A 的 baseline 端、direction sign/scale、相对 1e-6 退化阈值、非有限/零方向；A/u 在所有 IG steps 冻结；synthetic IG exact conservation；197 token contribution 手算及含 CLS 的 conservation；CLS 分离、196 patch reshape、support/positive-evidence degeneracy；错误 block/head/shape/dtype/非有限/空或全零 fail closed；MethodResult/provenance/重复性；成功和异常后 hooks、全部 module mode、parameter grad 与 requires_grad 精确恢复，caller-owned hook/grad 不得删除或覆盖。
9. 真实模型只做验收所需的低成本 CPU parity/smoke，显式禁用 GPU。不得为了 smoke 在真实 ViT 上运行完整 512–8192-step attribution；可用 toy/受控替身验证完整 schedule dispatch，用最小真实 forward 检查 A/V endpoint、fixed-A `g(x0)`、token identity、shape和 lifecycle。不得处理 pilot 图片集合。
10. 更新 docs/阶段C-step-by-step.md：C4 标 PASS 或 BLOCKED；若 PASS 将 C5 标 NEXT；追加完整 completion record 和“新对话执行 C5”的完整可复制提示词。
11. 检查 diff，只提交 C4 tracked 文件；提交后重跑 C0 validator（含 tracked/clean）、C1–C4 focused tests和全量非 GPU suite，最终确认 clean。

禁止事项：
- 不修改冻结 registry、config、protocol、acceptance、角色、阈值、方向、baseline、schedule、目标或 decision gates；确有矛盾必须停止并走带旧/新 hash 的 protocol amendment。
- 不实现 C5 的 fidelity、扰动稳定性或资源仪表；不实现 C6 runner/CLI/retention orchestration；不实现 selection、C8/C9 数据或 C10 统计决策。
- 不把 Value-path 方法称为类别 logit 解释、完整 Q/K/V 归因或完整 attention attribution；不把 raw attention 与 NNS 的结构相似性称为同目标 fidelity。
- 不把 `class_target` 上下文当作 Value-path conservation target；不把 patch→pixel/fine 质量均分称为原生像素证据。
- 不读取 display PNG 作为数值输入，不运行高成本 GPU attribution或批量 pilot。

验收标准：
- docs/stage_c_acceptance.md 的 C4 四项全部通过；
- clean fixed A、fixed-A baseline `g(x0)`、方向 `u=(g(x)-g(x0))/||g(x)-g(x0)||₂` 和相对 1e-6 退化规则与冻结协议逐项一致；
- endpoint parity、IG conservation、197-token matched conservation和 CLS 分离通过，A/u 在积分中冻结；
- 两种方法严格同目标，输出有限、非空、重复，并遵守 C1 MethodResult/map/display 边界；
- Stage B 单-channel 兼容回归通过，hook、全部 module mode、parameter grad 和 requires_grad 在成功/异常后精确恢复；
- 全量非 GPU pytest 通过，没有运行昂贵 GPU attribution、5 图 calibration 或 20 图 pilot；
- 一任务一提交，最终工作树 clean，completion record 的 commit/hash/test 信息完整，文档简单易懂。

方向继续冻结为 `u=(g(x)-g(x0))/||g(x)-g(x0)||₂`，积分期间固定；退化阈值继续是相对 `1e-6`。patch→pixel/fine 只表示质量守恒均分，不是原生像素证据。
~~~

### 2026-08-22 00:43 +0800 — C4 completion record

- status: PASS
- objective_completed: 只完成 C4。新增精确方法 `conditional_fixed_clean_attention_head_direction_value_path_ig`（`NNS fixed-clean-attention head-direction Value-path IG`）和 `matched_fixed_clean_attention_patch_token_contribution`（`matched fixed-clean-attention patch-token contribution`）。两者共享同一个冻结的 `fixed_clean_attention_cls_head_direction` 标量；前者沿真实上游 Value path 做 fixed-clean-A、fixed-u IG，后者解析分解全部 197 个 token 并把 CLS contribution 单独保留。没有实现或运行 C5–C10 的 fidelity、扰动稳定性、资源仪表、runner、CLI、selection、批量实验或统计决策。
- git_sha_before: `38a9d6493bc8694ab320850d8164699b737c075e`
- git_sha_after: SELF（即包含本 completion record 的单一 C4 commit；精确 SHA 在提交后由 `git rev-parse HEAD` 产生并在最终回复报告）
- tree_clean: true（C4 commit 后复验）
- files_added_or_changed: `src/vitnns/stage_c_value_path.py`；`tests/test_stage_c_value_path.py`；`tests/test_stage_c_contract.py`（只推进活文档状态/交接断言，并把 C4 源码与测试加入 tracked 检查）；`docs/阶段C-step-by-step.md`。
- inputs_and_sha256: C3 commit `38a9d6493bc8694ab320850d8164699b737c075e`；C2 commit `3ddc5b7d86d85729a644be49791aebb06f5285b6`；C1 commit `bd518afc7a479fb2c5dcf5ede31287d2d514147e`；C0 commit `05b0290b7caacc772a7c20a724f1491b3057524a`；Stage C registry `9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0`；Stage A 500 manifest `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`；启动 validator 再次通过 500 图官方标签链和 Stage B formal status→manifest→artifact hash/bytes 链。
- configs_and_sha256: frozen Stage C config `0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669`；config schema `94209b7c8a63a62ef27f3396810c8d5be8f6fbbf0a4df22e663b3661b9ed3f4b`；registry schema `4ce0ac8cd5bff93ae2b323c3e0447a87fc825c3243ad46c54c9b48e05e9b2d6d`；selection schema `867a3cf22519b2a045d81751a20c5cd52570ddeadede26a2da2bb79973f0b59d`；protocol `ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8`；acceptance `f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd`。冻结 registry/config/protocol/acceptance/schema、方法注册、角色、方向、baseline、阈值、schedule、目标和 decision gates 均未修改。
- artifacts_and_absolute_paths: 无 runtime/GPU artifact、manifest、status 或 display PNG。生产实现位于 `/data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_c_value_path.py`；测试位于 `/data00/home/yeziqi.831/testcode/vitnns_rebuild/tests/test_stage_c_value_path.py`。测试只在内存中生成 toy attribution，并对 canonical ViT 做最低成本 CPU endpoint smoke。
- artifact_manifest_or_status_hashes: 不适用；C4 不写实验 artifact，没有读取旧数值结果、display PNG、5 图 calibration 或 20 图 pilot。
- tests_and_results: 启动 C0 clean/tracked validator PASS；启动 C1–C3 focused 216 passed。C4 非集成最终 69 passed, 1 deselected；显式 `CUDA_VISIBLE_DEVICES=''` 的 canonical ViT CPU endpoint smoke 1 passed, 69 deselected；此前 C1–C4 联合 focused 286 passed。提交后的 clean/tracked validator、最终 C1–C4 focused 与全量非 GPU suite 结果在本任务最终回复报告。全程未运行 GPU attribution、真实 ViT 512–8192-step IG、5 图 calibration 或 20 图 pilot。
- acceptance_results: C4 四项全部通过。resolver 从真实 forward 的 `attention_used` 和 fused-QKV Value 捕获 clean/baseline endpoint，baseline 固定使用 `A(x)@V(x0)`；clean natural output、fixed-A replay 和独立 endpoint reference 分别做严格 parity。方向严格为 `d=g(x)-g(x0)`、`r=||d||₂`、`u=d/r`，阈值为 `1e-6*max(1,||g(x)||₂,||g(x0)||₂)`，A/d/u 在积分前成为私有 detached clones，非有限或阈值内方向结构化返回 `DEGENERATE_DIRECTION`。IG 使用 Gauss-Legendre float32 `[512,1024,2048,4096,8192]`、chunk 8、首个 conservation pass 并保留全部 attempts。matched 方法手算全部 197 个 `c_j`，验证含 CLS 的 conservation，只把 196 patch token reshape 为 14×14，并保留 support/positive-evidence 退化状态。两个方法输出严格 C1 `MethodResult`；NNS 保留 signed RGB native tensor后做 signed channel sum与16×16求和，matched 保留 signed patch mass与独立 CLS scalar；patch→pixel/fine 只声明质量守恒均分。选择性和 candidate chunk 接口覆盖全部 12×12 且不改变值；Stage B 单-channel API 和既有语义未修改。成功、异常及 hook-remove 异常路径均恢复 caller hooks、逐 module mode、parameter grad 与 `requires_grad`。
- deviations_or_protocol_amendments: 无 protocol amendment。float32 toy 路径中，独立 endpoint 运算次序可使 callback delta 与解析 r 出现约 `2.38e-6` 舍入差，因此两者一致性使用冻结的 IG conservation 容差验证；协议要求的 clean/baseline intervention replay parity 仍严格使用排他 `<1e-6`，方向、baseline、阈值和 schedule 均未改变。C4 增加 live model/input/state 绑定用于单进程防篡改；它不替代 C6 的跨运行完整 provenance。
- failures_retained: 开发首轮测试捕获缺少 `dataclasses.replace` 导入；resolver 改为同时返回 capture/model output 后捕获一次漏 unpack；新增 baseline decoy 测试最初因零 baseline 的 toy Value 也为零而不能区分错误 attention，随后仅增强测试的 QKV baseline offset。上述均修复并回归通过。独立合同、归因、测试缺口和最终代码审查均无剩余 high/medium issue。没有 GPU、OOM、真实高步数 attribution 或实验 artifact 失败。
- known_limitations: C4 只提供方法级不可变 provenance fragment和选择性/分块计算原语，不提供 C5 fidelity/stability/resource，不提供 C6 runner/retention/CLI。运行时 target 绑定用于当前进程防漂移，不是 canonical state-dict artifact hash，也不声明并发/重入线程安全。真实 canonical ViT 只验证 A/V endpoint、fixed-A baseline、token identity、shape和 lifecycle，没有执行昂贵完整 IG schedule；完整 schedule dispatch/conservation 由 toy/受控替身覆盖。matched patch→pixel/fine 仍只是质量守恒均分，不是原生像素证据；Value-path 方法也不是类别 logit、完整 Q/K/V 或完整 attention attribution。
- next_task: C5

#### 下一对话完整提示词

~~~text
你现在只执行 ViT-NNS 重建工程“阶段 C”的 C5：实现 fidelity、扰动稳定性与资源仪表原语。不要开始 C6，不要实现 runner、CLI、retention orchestration、selection、批量实验或统计决策，也不要运行昂贵 GPU attribution。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

前置冻结基线：
- C4 commit: SELF，即包含本 completion record 的当前 clean HEAD；启动 C5 时先运行 git rev-parse HEAD，其精确 SHA 应与 C4 最终回复给出的 commit SHA 一致。
- C3 commit: 38a9d6493bc8694ab320850d8164699b737c075e
- C2 commit: 3ddc5b7d86d85729a644be49791aebb06f5285b6
- C1 commit: bd518afc7a479fb2c5dcf5ede31287d2d514147e
- C0 commit: 05b0290b7caacc772a7c20a724f1491b3057524a
- Stage C registry SHA-256: 9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0
- Stage C config SHA-256: 0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669
- Stage C config schema SHA-256: 94209b7c8a63a62ef27f3396810c8d5be8f6fbbf0a4df22e663b3661b9ed3f4b
- Stage C protocol SHA-256: ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8
- Stage C acceptance SHA-256: f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd
- Stage A 500 manifest SHA-256: 042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89

第一步必须完整阅读：
1. docs/阶段C-step-by-step.md，尤其 C4 completion record 与 C5 边界
2. docs/阶段C_protocol.md，尤其第 5–7 节 map/fidelity/扰动稳定性与第 9 节资源测量边界
3. docs/stage_c_acceptance.md 的 C5 六项
4. docs/stage_c_config_contract.md
5. config/stage_c.json 与 config/stage_c.schema.json
6. src/vitnns/stage_c_config.py、stage_c_maps.py、stage_c_metrics.py、stage_c_attention.py、stage_c_class_logit.py、stage_c_value_path.py
7. src/vitnns/attribution.py、targets.py、extractor.py、modeling.py
8. src/vitnns/stage_b_runner.py 中已有 fidelity/sensitivity/resource 边界及其测试，只复用合适原语，不把 Stage B sensitivity sanity 改名为 Stage C stability
9. tests/test_stage_c_config.py、test_stage_c_maps.py、test_stage_c_metrics.py、test_stage_c_attention.py、test_stage_c_class_logit.py、test_stage_c_value_path.py，以及相关 intervention/hook/mode/grad 测试
10. docs/阶段A.md、docs/stage_a_acceptance.md、docs/阶段B.md、docs/stage_b_acceptance.md
11. README.md

开始前必须：
- 检查 git status 和 HEAD，要求 HEAD 等于 C4 最终回复 SHA 且工作树 clean；若不一致，查明归属，不覆盖用户改动。
- 运行 `CUDA_VISIBLE_DEVICES='' conda run -n vitnns python tests/test_stage_c_contract.py --require-tracked --require-clean`，核验冻结 hash、500 图官方标签链与 Stage B formal status→manifest→artifact 链。
- 按 8.1 的增量回归策略处理 C1–C4 基线：若精确 C4 clean HEAD、validator 和 C4 completion record 的提交后证据均成立，可复用该证据而不重复累计 focused tests；否则按影响面补跑。本次 C5 启动已实际运行 C1–C4 focused tests。
- 不从 `__pycache__`/`.pyc` 恢复源码，不读取 display PNG，不复用旧数值结果。

C5 范围与交付物：
1. 实现两条目标线各自正确的 fidelity 原语。对冻结标量 t，严格使用 `Delta=t(x)-t(x0)`、`sigma=sign(Delta)`、`S(z)=sigma*(t(z)-t(x0))/abs(Delta)`；不 clip S。Value-path 使用 C4 同一 fixed-clean-A/u/head target，按构造 sigma=+1；class-logit 使用 C3 frozen clean predicted-class raw logit。不得混用目标，也不得在 intervention 中重选 class、A、u、block/head 或 baseline。
2. target delta 阈值严格为 `1e-6*max(1,abs(t(x)),abs(t(x0)))`；退化返回结构化 `degenerate_target_delta`。Value-path 中它与 direction validity failure 对齐；class-logit 中保留为合法 NA，不得换 class/baseline，也不得伪造为 0。
3. ranking score 只能是 target-oriented signed map `sigma*M` 的正部。replacement 固定 normalized-zero baseline，RGB 三通道作为一个空间单位共同替换。deletion 从 clean 替换，insertion 从 baseline 恢复；禁止 blur、随机颜色或动态均值。
4. patch fidelity 固定 196 个 16×16 unit；fine fidelity 固定 3136 个 4×4 cell。NNS pixel map在 cell 内按 signed mass 求和；matched patch mass 每 patch均分到16个 cell，这只叫 4×4-cell/subpatch fidelity，不是单像素 fidelity。所有方法必须共享完全相同的 intervention tensors。
5. 曲线严格使用 21 个 fraction `0,0.05,...,1.0`；`k0=0`，其余 `k=ceil(fraction*N)`，横轴用实际 `k/N`，AUC 用 trapezoid。tie group 使用5个预注册 hash seed permutation；seed 字符串严格为 `stage-c-fidelity|20260821|sample_id|block|head|replicate` 的 SHA-256 前8 bytes big-endian unsigned。五次是配对 nuisance resolution，不是独立样本。另实现五次 global-random mean 和 center-distance-then-row-major sanity ranking。
6. Value-path `delta_fid` 只在同一 C4 target 上比较 NNS 与 matched patch，并严格用 fine-4：`0.5*((InsertionAUC_NNS-InsertionAUC_matched)+(DeletionAUC_matched-DeletionAUC_NNS))`。patch 结果只作支持性分析；class-logit 四方法只在同一 frozen raw logit 内比较，不得跨 line 混算。
7. patch/fine 都报告 top-20% sufficiency、`sufficiency_gap=1-sufficiency` 和 comprehensiveness，使用同一个 corruption operator与 `ceil(0.2*N)`，但不新增 decision gate。
8. 实现并严格验证三种 canonical RGB [0,1]、normalize 前扰动：右下整数平移 `dx=dy=4` + reflection padding；中心裁剪 `[4:220,4:220]` 后 bicubic antialias resize 224；CPU float32 iid Gaussian `std=0.01`、seed `20260821+selection_order` 后 clamp。每个 sample 的同一预生成 tensor/hash必须被所有方法/config共享。
9. 实现协议规定的 inverse geometry、质量 Jacobian与 mask：translation 共同有效 220×220、完整 patch rows/cols 0..12；crop-scale inverse bilinear antialias回216、乘 `(224/216)^2` 并放回 `[4:220]^2`、完整 patch rows/cols 1..12；noise identity/full mask。几何指标只读取完全有效 patch，residual 指标读取共同有效像素。
10. 稳定性使用 C1 指标合同计算 clean 与逆对齐 perturbed map 的 Pearson、Spearman、JS、top-10/20 Jaccard；所有 sample 主报告固定 clean target，natural perturbed prediction、unchanged subset和flip rate只作为区分明确的描述字段。不得把 Stage B normalized-space std=0.05 sensitivity sanity 当作本稳定性结果。
11. 对 NNS native 224×224 map实现 `R(M)=M-U(P(M))` 与 `rho=||R(M)||1/||M||1`。几何边界 patch只用有效像素构造该 patch有效区域的均匀 mass；零分母/常量 residual结构化标记 spatial failure。C5 只实现逐样本/逐扰动原语和结果合同，不实现 C10 的15图 bootstrap、门槛计数或 Go/No-Go。
12. 实现独立、可组合的资源仪表：CUDA 测量边界前后 synchronize、每单元开始前 reset peak，分别记录 peak allocated/reserved；记录 wall time、实际 artifact bytes和可解释的 forward/backward call count。CPU 返回明确的无 CUDA状态；conditional CUDA 测试必须很小，不得运行完整 attribution。不得在 C5 实现 C6 execution-unit目录、manifest/status、resume或 retention orchestration。
13. 输出严格、不可变、JSON 可序列化的逐次 fidelity/perturbation/stability/resource 结果；所有 shape/dtype/有限性/mask/target/method pairing/seed/tensor identity错误 fail closed，合法 NA保留原因，不把 missing/invalid/degenerate静默剔除或填0。
14. 测试必须覆盖：手算 S、delta退化与正负方向；patch/fine corruption identity及端点；ceil fraction/实际横轴/AUC/no-clip；tie与hash seed重复性、paired共享；random/center sanity；top-20%；Value-path matched delta与跨线拒绝；三种扰动精确值/hash共享；inverse geometry/Jacobian/mask round-trip；residual/rho及边界/零/常量；fixed clean targets与prediction flip字段；resource CPU和conditional CUDA；非法shape/dtype/非有限/空mask/错误target fail closed；成功和异常后的 hooks、全部 module mode、parameter grad与requires_grad精确恢复。
15. 真实模型只做验收所需的低成本 CPU target/intervention smoke，显式 `CUDA_VISIBLE_DEVICES=''`。用 toy/受控替身验证21点、5 tie repeats和完整调度；不得在真实 ViT 上运行512–8192-step attribution、整条3136-cell曲线或批量扰动，不得处理5图/20图集合。
16. 更新 docs/阶段C-step-by-step.md：C5 标 PASS 或 BLOCKED；若 PASS 将 C6 标 NEXT；追加完整 completion record 和“新对话执行 C6”的完整可复制提示词。
17. 检查 diff，只提交 C5 tracked 文件；按 8.1 运行 C5 及受影响 focused tests、C0 validator（含 tracked/clean）和一次全量非 GPU suite，避免在全量 suite 前后机械重复其已覆盖的 C1–C4 tests，最终确认 clean。

禁止事项：
- 不修改冻结 registry、config、protocol、acceptance、schema、角色、阈值、扰动、seed、目标、schedule、资源 cap 或 decision gates；确有矛盾必须停止并走带旧/新 hash 的 protocol amendment。
- 不实现 C6 runner/CLI/retention/provenance orchestration；不运行 C7 benchmark；不实现 selection、C8/C9 数据、C10 bootstrap或统计决策。
- 不重新实现或改写 C2–C4 方法语义；不把 class-logit fidelity 用来证明 Value-path 守恒，不把 raw attention 与 NNS 的结构相似性称为同目标 fidelity。
- 不把 patch→pixel 或 patch→fine质量均分称为原生像素证据，不把4×4-cell fidelity称为单像素 fidelity。
- 不读取 display PNG作为数值输入，不运行昂贵 GPU attribution、5图 calibration或20图 pilot。

验收标准：
- docs/stage_c_acceptance.md 的 C5 六项全部通过；
- 两条 line 的 frozen scalar、S归一化、normalized-zero replacement、RGB coupling、21点、ceil/actual fraction、trapezoid/no-clip与协议逐项一致；
- patch/fine operator、五个paired tie seed、sanity ranking、top-20%与Value-path fine-4 delta严格正确，所有方法共享同一扰动tensor；
- translation/crop/noise、inverse/Jacobian/mask、fixed clean targets、residual/rho和prediction flip记录严格正确；
- wall time/CUDA peak allocated+reserved/artifact bytes边界可解释，CPU与conditional CUDA测试通过；
- 错误输入fail closed、合法NA不填0，hook、全部module mode、parameter grad和requires_grad在成功/异常后精确恢复；
- 全量非 GPU pytest通过，没有运行昂贵GPU attribution、5图calibration或20图pilot；
- 一任务一提交，最终工作树clean，completion record的commit/hash/test信息完整，文档简单易懂。

C5只实现评价与测量原语，不实现运行编排或统计决策。Value-path继续冻结clean A/u；patch→pixel/fine只表示质量守恒均分，不是原生像素证据。
~~~

### 2026-08-22 02:45 +0800 — C5 completion record

- status: PASS
- objective_completed: 只完成 C5。新增冻结标量 fidelity、patch/4×4-cell corruption、21 点曲线、五次 paired tie、random/center sanity、top-20%、fine-4 Value-path `delta_fid`；新增三种 canonical RGB 扰动、inverse/Jacobian/mask、C1 sign-view stability、NNS patch-inner residual，以及 wall/CUDA/call/bytes 资源仪表。为 C2–C4 增加 arbitrary-intervention 方法级入口，但未实现 C6 runner、CLI、execution-unit、manifest/status/resume、retention orchestration、selection、批量实验、bootstrap 或统计决策。
- git_sha_before: `a4ada3374ad167fb0b691e29b3d3eb6709efa826`
- git_sha_after: SELF（即包含本 completion record 的单一 C5 commit；精确 SHA 在提交后由 `git rev-parse HEAD` 产生并在最终回复报告）
- tree_clean: true（C5 commit 后复验）
- files_added_or_changed: `src/vitnns/stage_c_fidelity.py`、`stage_c_perturbations.py`、`stage_c_stability.py`、`stage_c_resources.py`；`stage_c_attention.py`、`stage_c_class_logit.py`、`stage_c_value_path.py` 的 frozen-target arbitrary-z 原语；对应 C5/C2–C4 tests；`tests/test_stage_c_contract.py`；`README.md`；本文。未修改冻结 registry/config/schema/protocol/acceptance。
- inputs_and_sha256: C4 commit `a4ada3374ad167fb0b691e29b3d3eb6709efa826`；C3 `38a9d6493bc8694ab320850d8164699b737c075e`；C2 `3ddc5b7d86d85729a644be49791aebb06f5285b6`；C1 `bd518afc7a479fb2c5dcf5ede31287d2d514147e`；C0 `05b0290b7caacc772a7c20a724f1491b3057524a`；registry `9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0`；Stage A 500 manifest `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`。
- configs_and_sha256: Stage C config `0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669`；config schema `94209b7c8a63a62ef27f3396810c8d5be8f6fbbf0a4df22e663b3661b9ed3f4b`；registry schema `4ce0ac8cd5bff93ae2b323c3e0447a87fc825c3243ad46c54c9b48e05e9b2d6d`；selection schema `867a3cf22519b2a045d81751a20c5cd52570ddeadede26a2da2bb79973f0b59d`；protocol `ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8`；acceptance `f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd`；lock `81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984`。
- artifacts_and_absolute_paths: 无实验/GPU artifact、execution-unit、manifest/status、selection 或 display PNG。所有 toy tensor 仅存内存；生产原语位于 `/data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_c_*.py`。
- artifact_manifest_or_status_hashes: 不适用；C5 不写 runtime artifact。
- tests_and_results: 启动 C0 clean/tracked validator PASS；启动 C1–C4 focused `246 passed`。开发末受影响 focused 最近一次 `372 passed, 1 skipped`（conditional CUDA 因本会话显式无 CUDA而跳过）；C3/C4 arbitrary-z lifecycle、完整 21 点与五次 nuisance 调度均由 toy/CPU 覆盖。提交前/后的 validator 和全量非 GPU suite 最终计数在本任务最终回复报告。
- acceptance_results: C5 六项全部通过。`S(z)` 严格 signed endpoint normalization、不 clip；normalized-zero RGB coupled corruption、ceil/actual fraction/trapezoid、paired seed与 fine-4 delta均 fail closed 绑定 method/target/map/score/tensor。扰动生成器签发 canonical 与 normalized tensor/hash；translation/crop/noise、inverse Jacobian和 mask 精确。stability 只从 bound perturbed native map 执行 inverse/project，C1 全 sign views、prediction flip和 residual 绑定 method/target/perturbation。资源边界同步、reset、allocated/reserved、异常、调用和完整文件树 bytes 可解释。
- deviations_or_protocol_amendments: 无 protocol amendment。五个预注册 tie repeats 是完整 nuisance set，结果保留五次原值并报告 population SD（ddof=0），明确不当作独立统计样本；图像 sample SD 仍由冻结统计合同在 C10 使用 ddof=1。crop inverse multiplier使用冻结十进制 `1.075445816186557`。
- failures_retained: 开发中对抗审查依次发现并修复 frozen evaluator adapter缺失、target/map/score/tensor pairing不足、通用 stability/sign views缺失、loose perturbation hash、resource异常遮蔽、arbitrary-z map缺失、canonical/normalized hash域混用、matched-z map缺失及可变 alias。一次裸系统 Python import 因未进入 conda 环境失败、一次代理环境报告缺 timm，均未写文件；主线程 `conda run -n vitnns` 回归正常。没有 GPU/OOM/真实高步数 attribution 失败。
- known_limitations: C5 只提供方法级逐次原语和 JSON-safe结果，不写持久化 provenance/inventory，不提供并发 runner、批量/恢复/留存或预算决策。patch→pixel/fine仍只是质量守恒均分；4×4-cell不是单像素证据。真实 canonical ViT 高步数 IG、3136-cell整曲线、5图/20图集合均未运行。
- next_task: C6

#### 下一对话完整提示词

~~~text
你现在只执行 ViT-NNS 重建工程“阶段 C”的 C6：实现单图/批量 runner、留存、完整 provenance、预算预检、恢复与 CLI。不要开始 C7 benchmark，不要运行5图 calibration或20图 pilot，不要实现 C8 selection、C9 数据、C10 bootstrap/统计决策，也不要修改 C0–C5 冻结科学语义。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

前置冻结基线：
- C5 commit: SELF，即包含本 completion record 的当前 clean HEAD；启动 C6 时先运行 `git rev-parse HEAD`，其精确 SHA 应与 C5 最终回复一致。
- C4 commit: a4ada3374ad167fb0b691e29b3d3eb6709efa826
- C3 commit: 38a9d6493bc8694ab320850d8164699b737c075e
- C2 commit: 3ddc5b7d86d85729a644be49791aebb06f5285b6
- C1 commit: bd518afc7a479fb2c5dcf5ede31287d2d514147e
- C0 commit: 05b0290b7caacc772a7c20a724f1491b3057524a
- Stage C registry/config/schema/protocol/acceptance 与 Stage A manifest SHA-256 沿用 C5 completion record。

第一步必须完整阅读 `docs/阶段C-step-by-step.md`（尤其 C5 record、8.1与C6边界）、冻结 protocol 第9节、acceptance C6、config/schema/contract、C1–C5生产模块与测试、Stage A/B runner/retention/provenance/CLI及其测试、README。检查 HEAD/clean，运行 C0 validator；按8.1仅在基线或影响面异常时重复历史 focused。

C6范围：
1. 从冻结 registry/config/selection 合同实现 estimate、run-sample、run-batch、validate、aggregate或等价接口；C8前 selection 尚不存在时不得伪造选择。
2. preflight精确展开 sample×method×config×target×clean/三扰动×repeatability×21点×五tie及全部普通文件；任何 hard cap在GPU/写盘前拒绝。
3. execution unit写全新独立目录；只有 `complete=true`、manifest hash与全部artifact hash/bytes匹配才resume-skip。失败/incomplete/tampered不得追加或报成功，重跑用新空目录。
4. 实现冻结 L0/L1/L2/L3：L2只限两个audit；L3不能回流指标；禁止逐step/逐pixel/逐dimension文件爆炸。
5. provenance绑定clean/dirty commit、model/image/lock/registry/protocol/config/selection、角色/audit、精确method/target/map/baseline/schedule/seeds/扰动、status/failure、resource和inventory。failure table覆盖每个expected unit；batch complete要求全部显式终态。
6. 组合C5 frozen scalar、fidelity、perturbation/stability/residual和resource原语，不重写公式；canonical与normalized perturbation hash域不得混用；class与Value line不得跨算。
7. CLI保持非交互、可验证、可恢复；crash/resume/hash tamper/partial failure/missing row/non-overwrite/budget测试必须覆盖。
8. 测试以toy/CPU为主；conditional CUDA仅小边界。真实模型只做最低成本CPU或单次小GPU orchestration smoke，不运行C7 benchmark、完整IG、5图或20图。
9. 更新本文：C6 PASS/BLOCKED；PASS时C7 NEXT；追加completion record与完整C7提示词。一任务一提交，提交后按8.1运行validator、受影响focused及一次全量非GPU，确认clean。

禁止修改冻结 registry/config/schema/protocol/acceptance、方法语义、阈值、seed、target、schedule、扰动、caps或decision gates；有矛盾必须走带旧/新hash的protocol amendment。C6只做运行编排与可审计留存，不读取display PNG，不进行selection或统计决策。
~~~

### 2026-08-22 10:30 +0800 — C6 completion record

- status: PASS
- objective_completed: 只完成 C6。新增严格 RunSpec/registry/selection 读取、execution plan 与精确工作量/保守资源预检；新增单图/批量 runner、production C2–C5 组合 backend、L0/L1/L2/L3 immutable attempt writer、完整 provenance、resume/validate/L0 aggregate 与非交互 JSON CLI。C8 前 selection 明确为 `not_applicable_pre_c8`；没有生成 selection、bootstrap、统计决策或 benchmark。
- git_sha_before: `d15021e4a09bfc485ef9277f81cb13b8a7bae12f`
- git_sha_after: SELF（即包含本 completion record 的单一 C6 commit；精确 SHA 在提交后由 `git rev-parse HEAD` 产生并在最终回复报告）
- tree_clean: true（C6 commit 后复验）
- files_added_or_changed: `src/vitnns/stage_c_plan.py`、`stage_c_artifacts.py`、`stage_c_provenance.py`、`stage_c_runner.py`、`stage_c_cli.py`；对应五个 C6 test 文件；`pyproject.toml`；`README.md`；`tests/test_stage_c_contract.py`；本文。未修改冻结 registry/config/schema/protocol/acceptance 或 C0–C5 科学模块。
- inputs_and_sha256: C5 commit `d15021e4a09bfc485ef9277f81cb13b8a7bae12f`；C4 `a4ada3374ad167fb0b691e29b3d3eb6709efa826`；C3 `38a9d6493bc8694ab320850d8164699b737c075e`；C2 `3ddc5b7d86d85729a644be49791aebb06f5285b6`；C1 `bd518afc7a479fb2c5dcf5ede31287d2d514147e`；C0 `05b0290b7caacc772a7c20a724f1491b3057524a`；registry `9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0`；Stage A 500 manifest `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`。
- configs_and_sha256: Stage C config `0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669`；config schema `94209b7c8a63a62ef27f3396810c8d5be8f6fbbf0a4df22e663b3661b9ed3f4b`；registry schema `4ce0ac8cd5bff93ae2b323c3e0447a87fc825c3243ad46c54c9b48e05e9b2d6d`；selection schema `867a3cf22519b2a045d81751a20c5cd52570ddeadede26a2da2bb79973f0b59d`；protocol `ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8`；acceptance `f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd`；lock `81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984`；model `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`。
- exact_plan_counts: C8 五张 calibration、全部 7 方法和 144 个 Value configs 为 2,180 execution units、9,592 map runs、1,349,040 fidelity points、1,358,632 total WorkItems；C9 全 20 图、四个 selection winners 为 320 units、1,312 map runs、221,760 fidelity points、223,072 WorkItems。五次 tie 是 unit 内 nuisance repeats，不是独立统计样本。
- artifacts_and_absolute_paths: C6 只在 pytest 临时目录验证 runtime trees；未生成 C7 benchmark、5 图 calibration、20 图 pilot、selection 或持久实验 artifact。生产代码位于 `/data00/home/yeziqi.831/testcode/vitnns_rebuild/src/vitnns/stage_c_{plan,artifacts,provenance,runner,cli}.py`。
- artifact_manifest_or_status_hashes: 不适用；C6 没有保留实验输出。测试生成的 manifest/status/provenance 均在 pytest 临时目录，生命周期测试验证 exact bytes/SHA-256 chain。
- tests_and_results: 启动 C0 clean/tracked validator PASS，并按 8.1 复用 C5 completion record 在精确 C5 clean HEAD 上的累计 focused/full-suite 证据。提交前 C6 五模块与合同专项最终 `131 passed`；Stage C 受影响累计 focused `479 passed, 1 skipped`（conditional CUDA因本次显式CPU环境跳过）；全量 non-GPU `580 passed, 1 skipped, 1 deselected`（同一 conditional CUDA skip，unsafe-pickle deselected）。真实冻结 ViT CPU 最小 production orchestration smoke 单独 `1 passed`，只跑单图、单 per-head raw-attention clean map，观测1 forward/0 backward；未运行完整IG、C7 benchmark、5图或20图。提交后复验结果由最终回复补充。
- acceptance_results: C6 五项全部通过。preflight 由新鲜 issuer-authenticated estimate 放行，精确展开 WorkItems并计全部普通文件；每 unit 使用新 attempt，只有当前 plan/run-spec/preflight/code bindings与完整 status→manifest→artifact hash/bytes链均有效才 resume-skip。failed/incomplete/tampered 均保留且不追加。L1 固定 float32 14×14，L2 只限两张 audit，L3 仅 display且不进入 aggregate；batch failure table 对 expected units 左连接并区分 orchestration complete 与 successful。production backend 组合 C2–C5，不复制公式；structural raw 三扰动只报告 spatial structure，不伪造 scalar target/fidelity；canonical/normalized hash域、class/Value line和 fidelity pairing 均 fail closed。
- deviations_or_protocol_amendments: 无 protocol amendment。crash 可在原子提交边界留下不可恢复的 incomplete attempt；这是 crash evidence，不被宣称为 terminal/success，下一次运行使用新空 attempt，符合冻结的 incomplete/tampered 恢复语义。CLI exit code 为 0 success、1 input、2 preflight、3 terminal execution failure、4 integrity failure；这是 C6 接口设计，不改变冻结科学合同。
- failures_retained: 开发期间发现并修复 production sample-loader签名、动态 provenance resume绑定、mixed fidelity pairing、structural raw perturbation漏执行、top-20跨 pairing、batch实际全树字节、PNG完整性、ancestor symlink和 structured-NA留存问题。专项中间失败均为测试/实现收敛证据，没有实验数据、GPU/OOM或真实 attribution失败。
- known_limitations: C6 的真实 CPU smoke 只覆盖低成本 direct raw attention clean orchestration；所有方法/扰动/高步数 IG 的真实资源基准属于 C7。C6 aggregate 只输出经验证的 L0 长表，不进行 C8 selection、C10统计或Go/No-Go。进程在 manifest/status 原子切换之间被杀死会留下明确 non-resumable incomplete attempt，必须在新目录重跑。
- next_task: C7

#### 下一对话完整提示词

~~~text
你现在只执行 ViT-NNS 重建工程“阶段 C”的 C7：先修正逐点过度哈希，再执行单图 smoke benchmark 与预算冻结。不要开始 C8 calibration，不要运行 5 图全层/全头选择或 20 图 pilot，不要生成/修改 selection，不要实现 C9 数据生产或 C10 bootstrap/统计决策。除本文“C7 启动前完整性性能门”明确授权的完整性校验分层外，不得修改 C0–C6 冻结科学与运行语义。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

前置冻结基线：
- C7 完整性性能门文档 commit: SELF，即包含本次回退记录和性能门的当前 clean HEAD；启动 C7 时先运行 `git rev-parse HEAD`，其精确 SHA 应与本次最终回复一致。
- C6 code baseline commit: `53368aa08c68625c33bc59541a6b8a0216acfff4`；当前 HEAD 除本文文档约束外，生产代码与该 C6 commit 一致。
- 已放弃的过度哈希 C7 诊断 commit: `357dd9d4a8b6dd2d43ff5bfbc85cc93a68145226`，仅由本地归档引用 `archive/c7-overhash-357dd9d` 保留，不是下一次 C7 的代码基线。
- C5 commit: `d15021e4a09bfc485ef9277f81cb13b8a7bae12f`
- C4 commit: `a4ada3374ad167fb0b691e29b3d3eb6709efa826`
- C3 commit: `38a9d6493bc8694ab320850d8164699b737c075e`
- C2 commit: `3ddc5b7d86d85729a644be49791aebb06f5285b6`
- C1 commit: `bd518afc7a479fb2c5dcf5ede31287d2d514147e`
- C0 commit: `05b0290b7caacc772a7c20a724f1491b3057524a`
- Stage C registry/config/schema/protocol/acceptance、Stage A manifest、dependency lock和model SHA-256沿用 C6 completion record。

第一步必须完整阅读 `docs/阶段C-step-by-step.md`（尤其 C6 completion record、8.1、C7边界和“C7 启动前完整性性能门”）、冻结 protocol第9节、acceptance C7、config/schema/contract、C2–C6生产模块与相关测试、README。检查 HEAD/clean，运行 C0 validator；按8.1仅在基线或影响面异常时重复历史 focused。

C7范围：
1. 正式运行前先实现完整性校验分层。canonical checkpoint 加载仍校验冻结 state-dict SHA-256；每个 execution unit 在任何 target/map/fidelity/stability/IG 工作前建立一次强 model-state snapshot，并在成功或异常 `finally` 出口做一次强内容校验。强边界覆盖 parameter/persistent buffer 的名称、shape、dtype、device、object/storage identity和内容摘要，以及 module topology、mode/hooks、parameter grad/`requires_grad`、fused-attention 等运行状态；任一差异都显式失败并保留 lifecycle evidence。
2. fidelity point、frozen scalar evaluation、IG point/chunk 和普通 method wrapper 的内层只允许使用廉价 live-state guard，例如 object identity、`data_ptr`、`_version`、shape、dtype、device、`requires_grad`；禁止在这些热路径中把全部 parameters/buffers 搬到 CPU 后重算内容 SHA-256。cheap guard 不能冒充内容哈希；因为 `.data` 写入可能不增加 `_version`，unit 末尾强内容校验不可省略。21 张 intervention schedule 只在构造和 curve 边界做整组强内容校验，不得在每个 point 前后重复扫描整组 tensor。
3. 增加直接 instrument 全模型内容签名次数的回归测试：次数按 model-load/execution-unit 边界增长，不随 21 curve points、五次 tie repeats、IG steps/chunks 或 logical scalar evaluation 数线性增长；parameter/buffer 替换、storage rebinding、普通 in-place mutation由 cheap guard拒绝，`_version` 不变的 `.data` 内容篡改由 unit 末尾强哈希拒绝；成功和异常路径均执行结束强校验并恢复hooks/mode/grad/`requires_grad`。固定输入上数值、WorkItem、convergence和实际 batched forward/backward accounting必须保持不变。
4. 至少记录 attribution/fidelity模型计算、integrity checking、artifact finalize 三类 phase timer，并保留包含必要生产校验的同步总wall time。先在新的独立诊断目录运行一个预注册 audit NNS unit的完整冻结 workload；该unit若仍不低于`720`秒，则不启动40-unit正式attempt，继续做不改变数值语义的工程优化或据实判C7 `BLOCKED`。不得把诊断计时当正式budget结果。
5. 仅在一个预注册 audit 图上运行所有 7 个方法；Value-path/NNS至少覆盖预注册早/中/晚层和足以形成保守最坏收敛上界的显式 heads。科学 workload 必须保持同一12个显式Value candidates、clean/三扰动、repeatability、21 points、两个scale/方向、五次paired tie与random repeats、center sanity及完整IG schedule `[512,1024,2048,4096,8192]`/chunk 8；不要运行5图 calibration或20图 pilot。
6. 使用 C6 显式 RunSpec、estimate、runner、retention、provenance与CLI；修正后必须从新的clean commit和全新空output root运行，benchmark输出不得覆盖或resume任何历史attempt。若有失败/OOM/超cap，完整保留failure table和证据。
7. 诊断 commit `357dd9d4a8b6dd2d43ff5bfbc85cc93a68145226` 及其 `attempt_20260822`、`retry01`、`retry02` 和报告只能作为失败/性能诊断证据；正式报告不得引用、拼接或挑选其中任何unit。正式报告只能绑定修正后同一clean commit、同一新output root的全部40个units。
8. 每个方法/config记录同步 wall time、phase timers、实际 batched forward/backward调用、CUDA peak allocated/reserved、L0/L1/L2/L3及全部普通文件bytes、IG实际attempts/selected steps。验证hooks、全部module mode、parameter grad/requires_grad在成功和异常后均无残留。
9. 用最慢有效实测unit × 精确C8/C9 unit数 × 1.25 safety factor投影H20 GPU-hours；用C6精确artifact模型和实测bytes × 1.25 overhead复核 C8+C9与Stage D存储。不得把logical scalar evaluations当batched invocations。
10. 逐项检查冻结hard caps：64 MiB L0+L1/unit、1 GiB L2/audit sample、32 GiB C8+C9、64 GiB Stage D、30,000 forward和backward calls/unit、90 GiB peak CUDA reserved/unit、1,000 H20-hours C8+C9、2,000 H20-hours Stage D且4 hours/image。C7只可基于实测收紧估计，不得放宽cap或静默减少方法、层、头、steps、扰动、repeat。
11. 输出机器可读 benchmark report，绑定clean/dirty commit、model/image/lock/registry/protocol/config、RunSpec/execution plan、每unit manifest/status/provenance hashes和测量环境；不得依据热图美观、prediction correctness或holdout结果调参。
12. C7 PASS条件：热路径无全模型内容哈希且unit边界强校验通过；单图全方法编排完成、无OOM、资源可复算且所有投影在冻结cap内；正式report不引用任何`357dd9d`旧attempt。任一条件无法证明、超cap或无法建立可信最坏上界则C7 `BLOCKED`，或必须在访问holdout前另行提交带旧/新hash的protocol amendment。
13. 测试以完整性分层、数值等价、报告生成、预算数学、失败与恢复为主；实际benchmark只用一个audit图。不得启动C8、读取holdout、运行5图或20图，也不得实现selection/bootstrap/Go-No-Go。
14. 更新本文：C7 PASS/BLOCKED；PASS时C8 NEXT；追加completion record与完整C8提示词。一任务一提交，提交后按8.1运行validator、受影响focused及一次全量非GPU，确认clean。

只授权为完整性校验分层而窄幅修改 C3/C4/C5/C6 运行实现及其测试：把逐point/逐scalar/逐chunk的全模型内容SHA移到model-load与execution-unit边界，并加入cheap live-state guard和phase timer。禁止修改冻结registry/config/schema/protocol/acceptance、方法数值语义、阈值、seed、target、schedule、扰动、caps或decision gates；禁止缓存/复用会改变target、intervention forward、gradient、convergence、调用计数或测量边界的数值结果。任何超cap或合同矛盾必须显式BLOCKED或另走versioned protocol amendment；不得以跳过昂贵方法、减少配置、降低steps或不留证据来制造PASS。
~~~

### 2026-08-23 00:18 +0800 — C7 completion record

- status: PASS
- objective_completed: 只完成 C7。将全模型强内容签名限制在 canonical model-load 和 execution-unit 入口/`finally` 出口，fidelity point、frozen scalar、普通 method wrapper 与 IG chunk 只使用不读取 tensor 内容的 live-state guard；21 张 intervention schedule 只在构造和 curve 边界做整组内容校验。新增四类 phase timer、terminal telemetry/hash 绑定、C7 diagnostic gate、正式 benchmark report、预算投影与 CLI。随后只在一个预注册 audit 图上完成 1-unit NNS diagnostic 和 40-unit 全方法正式 benchmark；没有开始 C8/C9/C10。
- git_sha_before: `9032575d306e178c3f13ab809c0d560cd0175ab5`（C7 启动时 clean HEAD，与本次启动提示要求一致）。
- git_sha_benchmark: `314ae5e7f30b5d41f51d3b0d359992d7f0a95fdd`（正式 diagnostic、gate 和 40-unit report 共同绑定的 clean code commit；由本地只读引用 `archive/c7-benchmark-314ae5e` 保留）。
- git_sha_after: SELF（即包含实现、测试与本 completion record 的单一 C7 main commit；精确 SHA 在提交后由 `git rev-parse HEAD` 产生并在最终回复报告。除本文 completion record 外，生产代码与 `git_sha_benchmark` 一致）。
- tree_clean: 正式 diagnostic 和 benchmark 启动时均为 clean；最终提交后须再次确认为 clean。
- files_added_or_changed: 新增 `src/vitnns/stage_c_integrity.py`、`src/vitnns/stage_c_benchmark.py`、两个 C7 RunSpec 和对应测试；窄幅修改 class-logit、Value-path、fidelity、resources、artifacts、runner、CLI、README 及其受影响测试；本记录更新本文。冻结 registry、`config/stage_c.json`、schemas、protocol、acceptance、方法公式、target、seed、schedule、扰动、caps 与 decision gates 均未修改。
- inputs_and_sha256: C6 baseline `53368aa08c68625c33bc59541a6b8a0216acfff4`；C5 `d15021e4a09bfc485ef9277f81cb13b8a7bae12f`；C4 `a4ada3374ad167fb0b691e29b3d3eb6709efa826`；C3 `38a9d6493bc8694ab320850d8164699b737c075e`；C2 `3ddc5b7d86d85729a644be49791aebb06f5285b6`；C1 `bd518afc7a479fb2c5dcf5ede31287d2d514147e`；C0 `05b0290b7caacc772a7c20a724f1491b3057524a`；model `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`；audit JPEG `cf8f179939645bfab0734fbc75a189e7b3ef9624164cd84f98d6967b16b6a245`；Stage A 500 manifest `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`。
- configs_and_sha256: frozen Stage C config file `0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669`（report 内 canonical config-object digest `a0a9312f607d80e134c32c35b1cf120c65af9ba8d3f48e28026291ed20afab30`）；config schema `94209b7c8a63a62ef27f3396810c8d5be8f6fbbf0a4df22e663b3661b9ed3f4b`；registry `9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0`；registry schema `4ce0ac8cd5bff93ae2b323c3e0447a87fc825c3243ad46c54c9b48e05e9b2d6d`；selection schema `867a3cf22519b2a045d81751a20c5cd52570ddeadede26a2da2bb79973f0b59d`；protocol `ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8`；acceptance `f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd`；lock `81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984`；diagnostic RunSpec file `7966d846ceec1cb372cc1c177e412f87dc67e097e61f5f3f81a1822091b3f7ec`；formal RunSpec file `76f7e1ffd6e15b8e79ed856eba90bc930622399681bf32df8dc564abbe323404`。
- artifacts_and_absolute_paths: 有效 diagnostic root `/data00/home/yeziqi.831/testcode/vitnns_c7_diagnostic_314ae5e_20260822`；PASS gate `/data00/home/yeziqi.831/testcode/vitnns_c7_diagnostic_gate_314ae5e_20260822.json`；正式 root `/data00/home/yeziqi.831/testcode/vitnns_c7_formal_314ae5e_20260822`；PASS report `/data00/home/yeziqi.831/testcode/vitnns_c7_benchmark_report_314ae5e_20260822.json`。报告位于 source batch 外，正式 root 只有每 unit 的 `attempt-000001`。
- artifact_manifest_or_status_hashes: diagnostic gate file SHA-256 `c22416d8dd596c10ff8059d8bdf126b9e5d505720a2e893bf6f18dfb8fdc3abb`、canonical digest `4a25091a76d55bbfbddc001817581b0932f348e12a7681f3116cc97f58a42c0b`；formal report file SHA-256 `d33e4639da522460e8a2e83219b4d3a68bff50cf2adbfda97545a69ed471610c`、canonical digest `ae32465c79f9b0a0d24de4eaaf07d2f7b93e481d1188b62cac49f5aeb75eba7f`。正式 control files：`batch_status.json=6a51c8ef422e7f47781c17a935f2dc45fa884aa545f052fba93d850128189039`、`execution_plan.json=b5ffb843c6554f02a9a9be12dc5e5c049ff90e0cfc84f1df8637c03e7d6d0a08`、`failure_table.json=05af43a01b6eef7d3f312354b08b7dbe469695daecad39a230baec4888c5c708`、`preflight.json=c642c8f7db0ddc63c9c144a18747e66b0ddbe52e92312e15be124d5ac37bf1e8`、`c7_launch_authorization.json=7c5648859527db4b3782884e795e24ef8a801435dbb3159945b6404cfef61476`；plan canonical digest `7c538b932177949a9d30074abf6f542cd85e7f9f04fa2541aed48a385dab2beb`。report 内逐项绑定全部 40 个 unit 的 manifest/status/provenance/result hashes 和 source-tree file inventory。
- tests_and_results: 启动 C0 clean/tracked validator PASS；合并后受影响 focused `363 passed, 1 skipped`（唯一 skip 为显式无 CUDA 的 conditional resource test）；提交前全量 non-GPU `684 passed, 1 skipped, 1 deselected`（同一 conditional CUDA skip，`unsafe_pickle` deselected）。两次真实运行暴露的 provenance `id` 泄漏和 clean-map telemetry repeat index 均补定向回归；最终相关 runner/provenance `62 passed`，runner/benchmark `73 passed`。机器可读 diagnostic/report validator 与独立预算公式断言均 PASS。最终 SELF 提交后的 validator、affected focused、全量 non-GPU 和 clean 状态在最终回复报告。
- acceptance_results: 有效 diagnostic 为一个完整 block 0/head 0 NNS frozen workload，929 WorkItems、44 curves、924 points、5 maps，`1276` forward/`320` backward，H20 同步总 wall `36.172419351991266s < 720s`；integrity signature calls 为 unit 边界 2 次，entry/exit digest 相同，restoration PASS。正式单图范围精确为 7 methods、12 Value candidates、40 units、200 map runs、1,232 curves、25,872 curve points、26,072 WorkItems；40/40 succeeded、40/40 usable、无 OOM、无 retry。最慢有效 unit 是 block 11/head 11 NNS，`76.84819071099628s`；全批最大 `2426` forward、`1472` backward、CUDA allocated `1961380352` bytes、reserved `2833252352` bytes、L0+L1 cap-accounting `463771` bytes；C8 audit L2 保守投影 `488599600` bytes。按最慢 unit × 精确 units × 1.25，C8=`58.1698110242958`、C9=`8.538687856777365`、C8+C9=`66.70849888107315`、Stage D=`213.46719641943412` H20-hours，Stage D=`0.4269343928388682` H20-hours/image。存储投影 C8=`4289463500`、C9=`552286200`、C8+C9=`4841749700`、Stage D=`11497246200` bytes。全部 10 项冻结 hard caps PASS；报告确认 `holdout_accessed=false`、`selection_performed=false`、`bootstrap_performed=false`、`decision_performed=false`，且无 `357dd9d` 或旧 attempt 引用。
- deviations_or_protocol_amendments: 无 protocol amendment、无 cap 放宽、无科学 workload 缩减。phase timers 可嵌套，不要求加和等于 total；预算只使用 status telemetry 的同步总 wall，不使用 diagnostic 或 logical scalar evaluation 数。
- failures_retained: C7 前过度哈希历史 commit `357dd9d4a8b6dd2d43ff5bfbc85cc93a68145226` 及既有 attempts 仍由 `archive/c7-overhash-357dd9d`/原路径保留，未进入正式报告。本任务另保留 `/data00/home/yeziqi.831/testcode/vitnns_c7_diagnostic_da12f4d_20260822`（进入科学 workload 前因 provenance parameters 错含外层 `id` 而 incomplete）和 `/data00/home/yeziqi.831/testcode/vitnns_c7_diagnostic_ec7e357_20260822` 及其 BLOCKED gate（unit 完成但 clean-map `repeat_index` telemetry 与 plan 不一致）；对应代码分别由 `archive/c7-provenance-failure-da12f4d`、`archive/c7-gate-mismatch-ec7e357` 保留。这些证据均未 resume、拼接或用于正式预算。
- known_limitations: C7 是一个预注册 audit 图上的工程/预算边界，不是科学选择或泛化证据；C8 仍必须对精确 5 张 calibration 与全部 144 candidates 完整运行，并在独立 clean commit 上机器生成 selection。C7 的最慢单图 unit 乘法是冻结的保守上界，不能替代 C8 实际资源复核。外部 runtime artifacts 未纳入 Git，依赖上述绝对路径和 SHA-256 保存。
- next_task: C8

#### 下一对话完整提示词

~~~text
你现在只执行 ViT-NNS 重建工程“阶段 C”的 C8：在精确 5 张 calibration 图上运行全部 12×12 Value candidates，并按冻结协议机器生成和冻结 selection。不要开始 C9 的 20 图 pilot，不要读取或运行任何 holdout 图，不要实现 C10 bootstrap/统计决策，也不要依据 prediction correctness、图片内容、display PNG、热图美观或人工偏好选择配置。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

前置冻结基线与证据：
- C7 completion commit: SELF，即包含本 C7 completion record 的当前 clean HEAD；启动 C8 时先运行 `git rev-parse HEAD`，其精确 SHA 应与 C7 最终回复一致。
- C7 正式 benchmark code/evidence commit: `314ae5e7f30b5d41f51d3b0d359992d7f0a95fdd`，由本地只读引用 `archive/c7-benchmark-314ae5e` 保留；C7 最终 HEAD 相对它仅增加 completion 文档及与 C7 PASS/C8 NEXT 同步的静态文档状态断言，生产代码完全一致。
- C7 正式 root: `/data00/home/yeziqi.831/testcode/vitnns_c7_formal_314ae5e_20260822`；机器可读 report: `/data00/home/yeziqi.831/testcode/vitnns_c7_benchmark_report_314ae5e_20260822.json`，file SHA-256 `d33e4639da522460e8a2e83219b4d3a68bff50cf2adbfda97545a69ed471610c`，canonical digest `ae32465c79f9b0a0d24de4eaaf07d2f7b93e481d1188b62cac49f5aeb75eba7f`，status `PASS`。
- C7 diagnostic gate: `/data00/home/yeziqi.831/testcode/vitnns_c7_diagnostic_gate_314ae5e_20260822.json`，file SHA-256 `c22416d8dd596c10ff8059d8bdf126b9e5d505720a2e893bf6f18dfb8fdc3abb`，canonical digest `4a25091a76d55bbfbddc001817581b0932f348e12a7681f3116cc97f58a42c0b`。
- C6 `53368aa08c68625c33bc59541a6b8a0216acfff4`；C5 `d15021e4a09bfc485ef9277f81cb13b8a7bae12f`；C4 `a4ada3374ad167fb0b691e29b3d3eb6709efa826`；C3 `38a9d6493bc8694ab320850d8164699b737c075e`；C2 `3ddc5b7d86d85729a644be49791aebb06f5285b6`；C1 `bd518afc7a479fb2c5dcf5ede31287d2d514147e`；C0 `05b0290b7caacc772a7c20a724f1491b3057524a`。
- 冻结 SHA-256：model `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`；registry `9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0`；Stage C config `0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669`；protocol `ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8`；acceptance `f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd`；lock `81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984`；Stage A manifest `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`。

第一步必须完整阅读 `docs/阶段C-step-by-step.md`（尤其 C7 completion record、8.1 和 C8 边界）、冻结 `docs/阶段C_protocol.md` 第 8/9 节、`docs/stage_c_acceptance.md` C8、config/schema/contract、C6/C7 runner/integrity/benchmark/report 实现与测试、README。检查 HEAD/clean，运行 C0 validator；复核 C7 report/gate 的文件 hash、canonical digest、内部 validator、40/40 usable、10 caps PASS、code/archive binding 和 `holdout_accessed=false`。按 8.1 只运行本任务及实际影响面的必要测试。

C8 范围：
1. 只允许读取 registry 中 role=`calibration` 的精确 5 个 ID：`00000079`、`00009888`、`00023229`、`00035680`、`00049970`；禁止读取、预处理、hash、推理或汇总其余 15 张 holdout。选择流水线必须记录并验证 `holdout_accessed=false`。
2. 为这 5 图建立显式 C8 RunSpec/execution plan：每图运行全部 7 个方法；三种 per-candidate 方法覆盖全部 `12×12=144` block/head candidates，四种 global 方法各一 unit，即精确 `436 units/image`、总计 `2180 units`。不得以 C7 的 12 candidates 替代全 144，不得减少方法、扰动、repeat、21 points、两个 scale/direction、五次 paired tie/random、center sanity 或 IG schedule `[512,1024,2048,4096,8192]`/chunk 8。
3. C8 必须从新的 clean commit 和全新空 output root 启动；使用 C6/C7 显式 RunSpec、estimate、runner、retention、provenance、完整性边界、phase timers 和 CLI。不得覆盖/resume C7 或失败诊断目录；若 C8 自身失败，保留 immutable attempt/failure table，修复后使用新的空 root。
4. 运行前按 C7 冻结的保守估计和 C6 精确 artifact model 做 preflight；运行中逐 unit 执行 frozen hard caps。C7 预算基线为：最慢有效 unit `76.84819071099628s`，C8 投影 `58.1698110242958 H20-hours`，C8+C9 `66.70849888107315 H20-hours`；C8 存储投影 `4289463500` bytes，C8+C9 `4841749700` bytes；C8 audit L2 `488599600` bytes。不得放宽 frozen caps；超 cap 或证据不足即 C8 `BLOCKED`。
5. L0/L1 覆盖所有预期 units；L2 仅允许两个 audit calibration：`ILSVRC2012_val_00000079.JPEG` 和 `ILSVRC2012_val_00049970.JPEG`；L3 仅 display 且永不作为指标输入。每 unit 必须绑定 clean Git、model/image/lock/registry/protocol/config/RunSpec/plan、target/schedule/seed/扰动、manifest/status/provenance hashes、调用数、同步 wall、phase timers、CUDA peaks 和全部普通文件 bytes。
6. 所有 2180 expected units 必须有显式终态并通过 batch/artifact/hash/provenance/lifecycle 校验；只允许完整且 hash 有效的 unit 进入 selection。任何 failed/missing/tampered、OOM、完整性残留或 required numeric NA 都保留并使对应 candidate 不 eligible，不能丢行、补零、重选 target 或暗中重试后挑最好 attempt。
7. 实现或完成 C8 calibration aggregation 和机器生成 selection 的最小必要代码/CLI。候选 eligibility 严格要求 5/5 calibration 的 endpoint parity、nondegenerate direction、IG conservation、token parity、required maps/curves、provenance 和 hard budget 全部有效；空间退化保留在排序中，不作静默剔除。
8. 每个 eligible candidate 仅从 5 calibration 计算：`n_pos=count(delta_fid_fine4>0)`、`m_fid=median(delta_fid_fine4)`、`m_stab=min(三个扰动各自 residual Spearman median)`、`m_res=median(rho)`；NA stability 固定按 `-1` 排序，NA residual ratio 固定按 `0` 排序。科学量先 round 到 6 decimals，再按 `(n_pos, m_fid, m_stab, m_res, -block_index, -head_index)` 降序；不得新增或替换选择指标。
9. 深度带固定为 `[0,1,2]`、`[3,4,5]`、`[6,7,8]`、`[9,10,11]`，每带选 1 个；再按同一规则从四个 winner 选唯一 primary，其余三项标为 exploratory。任一 band 无 eligible candidate，C8 必须 `BLOCKED`，不得放松 validity 或换候选。
10. 输出由程序生成并通过 schema/strict loader 的 `selection/stage_c_selection.json` 和 selection report，记录全部 144 candidates、失败/退化、排序 operands/rounded tuple、四个 band winners、primary，并绑定 code/protocol/acceptance/config/registry/calibration artifact hashes。selection 禁止人工改写；提交前验证修改可检测、primary 属于四个 winners、无重复/遗漏 candidate。
11. C8 只做 calibration 与 selection，不做 holdout 推理，不运行 C9 20 图 pilot，不做 bootstrap/CI/Holm 实际检验或 C10 Go/No-Go，不依据 class correctness、图像内容、C7 结果表现或可视化调参。C7 结果仅用于资源边界，不能作为 C8 candidate score。
12. 测试聚焦：严格 5-ID/144-candidate/2180-unit plan、holdout 不可访问、eligibility fail-closed、6-decimal lexicographic ranking、NA rank、band/primary tie-break、空 band BLOCKED、selection schema/hash/tamper/manual-edit、失败恢复和预算数学。真实 GPU 只运行上述必要 C8 calibration，不额外做 smoke、5 图之外或 20 图实验。
13. 完成后更新本文：C8 标 `PASS` 或 `BLOCKED`；PASS 时 C9 标 `NEXT`；追加 C8 completion record 和完整 C9 提示词。一任务一提交；提交后按 8.1 运行 C0 validator、受影响 focused 和一次全量 non-GPU，确认 clean。

禁止修改冻结 registry/config/schema/protocol/acceptance、科学方法、阈值、seed、target、schedule、扰动、caps 或 decision gates；若发现合同矛盾或预算超限，必须明确 `BLOCKED`，不得通过删样本、删层/头、降 steps、跳过昂贵方法、读取 holdout 或人工修改 selection 制造 PASS。
~~~

### 2026-08-24 06:10 +0800 — C8 completion record

- status: PASS
- objective_completed: 只完成 C8。在精确 5 张 calibration 图上执行全部 `12×12=144` Value candidates 和全部 7 种方法，计划严格为每图 `144×3+4=436` units、合计 `2180` units；随后只从 calibration 证据机器生成四个深度带 winner 和唯一 primary。没有读取或运行 15 张 holdout，没有开始 C9/C10，没有执行 bootstrap、Holm 或科学决策。
- git_sha_before: `6fd4b9c3cd9fd20ea41d0c61a6484eebe1b8b01c`（C7 completion 后的 clean baseline）。
- git_sha_execution_and_finalize: `1fd25392968c9ccfde6c066f6544e9c4557ac4f0`。该 clean commit 是唯一正式 C8 GPU execution root、全部 unit artifact 和 finalize control files 的 producer；selection 不是在该 commit 上 build。
- git_sha_selection_generator_and_recovery_fix: `d7f2ef3bca8c22e1fc1ae38d9333fd2cd3e11579`。该 clean commit 只从 `1fd2539...` 已完成且不可变的 source evidence 重新 build 并 validate selection/report；没有重跑任何 GPU unit。
- git_sha_after: SELF（即包含 C8 实现、机器生成 selection、本 completion record 与状态推进的 C8 completion commit；精确 SHA 在提交后由 `git rev-parse HEAD` 产生，并在最终回复报告）。
- tree_clean: 正式 GPU execution/finalize 启动和结束时绑定 `1fd2539...` clean tree；selection postprocessor build/validate 绑定 `d7f2ef3...` clean tree。最终 SELF 提交后的 clean 状态由最终回复确认。
- files_added_or_changed: C8 增加/收紧 calibration RunSpec、plan/runner/artifact/failure evidence、resource projection、selection aggregation/validation 与 CLI 及其测试；机器生成 `selection/stage_c_selection.json`；本记录只推进本文状态和交接。冻结 registry、config、schemas、protocol、acceptance、科学方法、target、baseline、schedule、seed、扰动、retention、caps 与 decision gates 均未修改。
- inputs_and_sha256: model `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`；registry `9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0`；registry metadata `bedb9df9a77807c86da0b898f8d44767c8276f21ab69821965f2b30d7184a4af`；Stage A 500 manifest `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`；dependency lock `81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984`。
- configs_and_sha256: frozen Stage C config `0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669`；config schema `94209b7c8a63a62ef27f3396810c8d5be8f6fbbf0a4df22e663b3661b9ed3f4b`；registry schema `4ce0ac8cd5bff93ae2b323c3e0447a87fc825c3243ad46c54c9b48e05e9b2d6d`；selection schema `867a3cf22519b2a045d81751a20c5cd52570ddeadede26a2da2bb79973f0b59d`；protocol `ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8`；acceptance `f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd`；C8 RunSpec file SHA-256 `9fae31df8ed2da462f48d1a95a0df2c44cc70a6cfd7db403ef40d69706d3f8c2`、semantic/canonical digest `42f22bb4992dd560382acd48a5e5ec2fe47b0345835e9201636943fb22e7e30b`。
- artifacts_and_absolute_paths: 唯一正式 execution/finalize root 是 `/data00/home/yeziqi.831/testcode/vitnns_c8_formal_1fd2539_20260823`。第一次 postprocessing 生成的 BLOCKED report `/data00/home/yeziqi.831/testcode/vitnns_c8_selection_report_1fd2539_20260823.json` 保留，file SHA-256 `a892f772398402ffa0aa0641bcc27141905f4aeb147a0c63d8c083446231fa41`，不得当作最终 selection。修复后的 canonical PASS report 是 `/data00/home/yeziqi.831/testcode/vitnns_c8_selection_report_final_d7f2ef3_20260824.json`，file SHA-256 `d921d95ebe756e8cec859faea96578f8b3da766e9049e0440f570487d2e40b44`、canonical digest `9eb0df50be93320d7d6279852441c5bc103eaef146b394ff8d128cc64d2c9a6c`。外部 validated recovery selection `/data00/home/yeziqi.831/testcode/vitnns_c8_selection_recovery_d7f2ef3_20260824.json` 与 tracked `selection/stage_c_selection.json` 的精确文件 SHA-256 均为 `5dff9b5aec77cab22b157b9bf2b5d63b2252b7e50ebe728250206d655a142254`。
- artifact_manifest_or_status_hashes: 正式 root control file SHA-256：`execution_plan.json=da24efd1a700c59ae378a87b7b5b575f5d09edcbf40111c515f1321b2db899f9`、`preflight.json=c7d3a9baa1b95c2c6b1fdd9af70b9714c18db9230e34e437a401e94a329899f2`、`c8_launch_authorization.json=86a60e0a3c915b1295a803ace0c9ac6c414ef99a0dac7a879b30abd8663999fe`、`failure_table.json=deacdadd2c7ebe42b253bd1da86a264664e1d4a885c95cb8ed92e2ecfce327d0`、`batch_status.json=7c37bb76ae42d445912b953e582a02b30a16107ba9d5ce43e3fa1f4199f6a708`；calibration artifact manifest digest `705b92c6101b5b4354eb3d2b922aced8d781dcf5ee6d3b7198a6132d9a495788`；execution plan canonical digest `ece22206df300d4099c888d9ce83be83bbf926eaf4d5003059694912881f0892`；preflight canonical digest `175d5c9c8461dd5b082792a57ffb7f627c62f2c540114b8789039ebc334b7a2b`；calibration evidence canonical digest `08daa501d3e18ba927180156537de86a3c62fac6176dde3f3a6a37f0916e1620`。不存在也不引用 shard log 路径。
- tests_and_results: GPU 启动前 focused baseline `413 passed, 1 skipped`；C8 实现后的全量 non-GPU suite `805 passed, 2 skipped`；`d7f2ef3...` recovery 修复的受影响 focused tests `163 passed`；C0 clean/tracked static validator PASS；正式 source evidence build/validate 与 recovery selection/report build/validate 均 PASS。上述为已知已执行结果；SELF completion 提交后的最终 validator/focused/full-suite 计数如需补充，由最终回复据实报告，不在本文预先虚构。
- acceptance_results: 正式 plan 的 `2180/2180` expected units 均有显式 terminal state，`2179` succeeded、`1` evidence-complete failed，且 `2180/2180` trusted。144 candidates 中 `143` eligible、`1` ineligible；唯一 ineligible 是 `(block 11, head 05)`。四个固定深度带 winner 依次为 `(0,9)`、`(4,2)`、`(6,10)`、`(11,2)`，primary 为 `(0,9)`，其余三项为 exploratory。最终 report 为 PASS 且 `integrity_valid=true`、`holdout_accessed=false`、`bootstrap_performed=false`、`holm_performed=false`、`decision_performed=false`。
- budget_results: 全部 10 项 frozen hard caps PASS。实测最大 L0+L1 cap-accounting `466601` bytes/unit；L2 actual maximum `349093216` bytes/audit sample，C8 conservative projected audit L2 `488599600` bytes；最大实际 forward `5558`、backward `4608` batched invocations/unit；最大 CUDA reserved `4412407808` bytes/unit。最慢有效同步 unit `197.04103458393365s`；按精确 unit 数与 1.25 factor，C8=`149.14911645589422` H20-GPU-hours、C9=`21.893448287103737`、C8+C9=`171.04256474299797`、Stage D=`547.3362071775935`，Stage D=`1.094672414355187` H20-GPU-hours/image。存储投影 C8=`3527349125`、C9=`440416200`、C8+C9=`3967765325`、Stage D=`8700496200` bytes。
- deviations_or_protocol_amendments: 无 protocol amendment、无 workload/cap/阈值变更。`d7f2ef3...` 修复两个纯 postprocessing bug：第一，`runtime_model_binding` 是独立 execution unit 的进程内 live-model/tensor guard，跨 unit 比较科学 target 时只归一化掉这一字段，但每条 curve 和每个 unit 仍先校验完整、未删减的 per-unit target identity hash；第二，C7 aggregate storage model 必须先按其冻结的 40-unit/200-map source scale 分解，再投影到 C8/C9，不能拿 C8 的 2180-unit records 错误重分解 C7 aggregate。修复没有修改 frozen protocol/config/schema 或任何 runtime artifact，也没有重跑 GPU。最终证据显式双绑定：artifact producer=`1fd2539...`，selection generator/validator=`d7f2ef3...`。
- failures_retained: 所有旧失败/中断 root 均原样保留且未 resume、拼接或进入最终 selection：`/data00/home/yeziqi.831/testcode/vitnns_c8_formal_057960f_20260823`、`/data00/home/yeziqi.831/testcode/vitnns_c8_formal_b81da13_20260823`、`/data00/home/yeziqi.831/testcode/vitnns_c8_formal_17814e3_20260823`、`/data00/home/yeziqi.831/testcode/vitnns_c8_formal_daa970d_20260823`。正式 `1fd2539...` root 中唯一真实失败也保留：audit calibration `ILSVRC2012_val/ILSVRC2012_val_00049970.JPEG` 的 `conditional_fixed_clean_attention_head_direction_value_path_ig`、`block_11_head_05` 在完整冻结 IG schedule 后未收敛；其 terminal failure/status/hash/resource evidence 完整，该 candidate 因此 fail closed 为 ineligible，未被删除、补零或重跑。原始 BLOCKED selection report 同样保留作为 postprocessing 失败史。
- known_limitations: C8 只用 5 张 calibration 做选择，不是 holdout 或泛化结论；四个 winners 在 C9 之前不可再改。一个非 winner candidate 有真实未收敛失败。runtime artifacts 位于上述绝对路径，不由 Git 保存；C9 必须同时核验 producer `1fd2539...`、generator `d7f2ef3...`、最终 report 和 tracked selection 的完整绑定。
- next_task: C9

#### 下一对话完整提示词

~~~text
你现在只执行 ViT-NNS 重建工程“阶段 C”的 C9：从 C8 已机器冻结的 selection 出发，在 registry 的精确 20 张图上运行 frozen pilot，生成完整、不可变、可 hash 验证的逐 execution-unit 证据。不要开始 C10，不要做 bootstrap/CI、Holm、多重比较、汇总统计或任何 GO/NO-GO/INCONCLUSIVE 决策；不要修改 selection、四个 winners、primary、阈值或科学协议。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

前置 C8 completion 与双代码绑定：
- C8 completion commit: SELF，即包含 C8 实现、tracked selection 与本 completion record 的 clean commit；启动 C9 时先运行 `git rev-parse HEAD` 并与 C8 最终回复给出的精确 SHA 一致。若 HEAD 不一致、工作树 dirty 或 dirty 归属不明，先查明且不得覆盖用户改动；不能建立精确 clean C8 baseline 时，C9 必须 BLOCKED，禁止读取 holdout 或启动 GPU。
- C8 GPU execution/finalize artifact producer: `1fd25392968c9ccfde6c066f6544e9c4557ac4f0`（clean）。不得声称 selection 在这个 commit 上 build。
- C8 selection postprocessor build/validate generator: `d7f2ef3bca8c22e1fc1ae38d9333fd2cd3e11579`（clean）。该修复只重建/验证 selection/report，没有 GPU rerun。
- 正式 C8 source root: `/data00/home/yeziqi.831/testcode/vitnns_c8_formal_1fd2539_20260823`。
- canonical PASS report: `/data00/home/yeziqi.831/testcode/vitnns_c8_selection_report_final_d7f2ef3_20260824.json`，file SHA-256 `d921d95ebe756e8cec859faea96578f8b3da766e9049e0440f570487d2e40b44`，canonical digest `9eb0df50be93320d7d6279852441c5bc103eaef146b394ff8d128cc64d2c9a6c`。
- tracked selection `selection/stage_c_selection.json` 的精确文件 SHA-256 为 `5dff9b5aec77cab22b157b9bf2b5d63b2252b7e50ebe728250206d655a142254`；外部 validated recovery copy `/data00/home/yeziqi.831/testcode/vitnns_c8_selection_recovery_d7f2ef3_20260824.json` 必须同 hash。
- 原始 BLOCKED report `/data00/home/yeziqi.831/testcode/vitnns_c8_selection_report_1fd2539_20260823.json`（SHA-256 `a892f772398402ffa0aa0641bcc27141905f4aeb147a0c63d8c083446231fa41`）只作失败史，不能替代 canonical PASS report。
- C8 calibration artifact manifest digest `705b92c6101b5b4354eb3d2b922aced8d781dcf5ee6d3b7198a6132d9a495788`；execution plan canonical digest `ece22206df300d4099c888d9ce83be83bbf926eaf4d5003059694912881f0892`；RunSpec digest `42f22bb4992dd560382acd48a5e5ec2fe47b0345835e9201636943fb22e7e30b`；preflight canonical digest `175d5c9c8461dd5b082792a57ffb7f627c62f2c540114b8789039ebc334b7a2b`。

启动时复算并要求一致的冻结 SHA-256：
- model `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`
- registry `9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0`
- registry metadata `bedb9df9a77807c86da0b898f8d44767c8276f21ab69821965f2b30d7184a4af`
- Stage C config `0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669`
- config schema `94209b7c8a63a62ef27f3396810c8d5be8f6fbbf0a4df22e663b3661b9ed3f4b`
- registry schema `4ce0ac8cd5bff93ae2b323c3e0447a87fc825c3243ad46c54c9b48e05e9b2d6d`
- selection schema `867a3cf22519b2a045d81751a20c5cd52570ddeadede26a2da2bb79973f0b59d`
- protocol `ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8`
- acceptance `f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd`
- dependency lock `81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984`
- Stage A 500 manifest `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`

第一步必须完整阅读 `docs/阶段C-step-by-step.md`（尤其 C8 completion record、§8.1、C9 边界和交接规则）、`docs/阶段C_protocol.md`（尤其样本/方法/扰动/fidelity/留存；统计章节只作为 C9 禁止提前执行的边界）、`docs/stage_c_acceptance.md` C6/C8/C9/C10、`docs/stage_c_config_contract.md`、config 与三个 schemas、tracked selection、C6–C8 plan/runner/artifacts/provenance/integrity/C8/selection/CLI 实现及测试、README。先检查 HEAD/clean，运行 C0 tracked/clean validator，验证上述所有 hash、C8 report 内部 canonical digest、source evidence、双 commit binding、四 winners/primary 和 selection schema/semantic integrity。

C9 精确范围：
1. 按 registry `selection_order` 运行且只运行以下精确 20 图，不得漏掉、增加、替换或按 prediction correctness 删除：`ILSVRC2012_val_00000079`、`ILSVRC2012_val_00002607`、`ILSVRC2012_val_00005279`、`ILSVRC2012_val_00007489`、`ILSVRC2012_val_00009888`、`ILSVRC2012_val_00012573`、`ILSVRC2012_val_00014787`、`ILSVRC2012_val_00018380`、`ILSVRC2012_val_00020970`、`ILSVRC2012_val_00023229`、`ILSVRC2012_val_00025927`、`ILSVRC2012_val_00028908`、`ILSVRC2012_val_00031099`、`ILSVRC2012_val_00033512`、`ILSVRC2012_val_00035680`、`ILSVRC2012_val_00038565`、`ILSVRC2012_val_00041511`、`ILSVRC2012_val_00044187`、`ILSVRC2012_val_00047079`、`ILSVRC2012_val_00049970`。其中 5 calibration、15 holdout；两个 audit 仍仅为 calibration `00000079` 和 `00049970`。`00012573` 即使原始尺寸为 90×90 也必须保留。
2. C8 selection 的四个 winner 已冻结为 `(block 0, head 9)` primary、`(4,2)`、`(6,10)`、`(11,2)` exploratory。C9 RunSpec 必须使用 `value_config_source=selection_band_winners` 且无 explicit candidates；不得只跑 primary、重排或替换 winner。
3. 每张图在四个 winner 上各运行三种 Value 方法：`direct_raw_cls_attention_per_head`、`conditional_fixed_clean_attention_head_direction_value_path_ig`、`matched_fixed_clean_attention_patch_token_contribution`；每张图各只运行一次四种 global 方法：`direct_raw_cls_attention_last_block_head_mean`、`attention_rollout_mean_head_residual_all_blocks`、`vit_grad_cam_block11_norm1`、`full_model_class_logit_ig`。精确总数是 `20 × (4 winners × 3 Value methods + 4 global methods) = 320 execution units`，绝不能把四种 global 方法乘四个 winners。
4. 精确展开合同为 `240` fidelity-eligible units、`80` structural-reference units、`10560` fidelity curves、`221760` curve points、`1312` map runs、`223072` WorkItems、`80` global class-line units。RunSpec 保持 registered fidelity、sanity、stability、audit repeatability `2` 和 `clean_map_only`；global class fidelity 的显式 pairing context 使用 frozen primary `(0,9)`。
5. 保留完整冻结 workload：同一样本的全部方法/config 共享一个 canonical clean tensor、同一模型/target context 和预生成且 hash 配对的三种 perturbation tensors；两个 fidelity scales/directions、21 fractions、五次 paired target-tie、五次 random sanity、center sanity，以及 IG schedule `[512,1024,2048,4096,8192]`/chunk 8 均不得缩减。class line 始终解释 clean predicted-class raw logit；Value line 固定 clean A/u/block/head/baseline。记录真值、预测正确性、扰动预测与 flip，但不据此删样本。
6. C9 只实现/补全该 frozen pilot 所需的最小 RunSpec、plan、orchestration、CLI 和 validation。正式执行必须绑定一个新的 clean code commit，并在该 commit 存在后创建一个全新空 output root；本提示词不预先虚构 C9 root 名。不得覆盖、resume、拼接或挑选 C7/C8/recovery/diagnostic/失败 root 的 unit。
7. 在读取任何 holdout 图、启动 GPU 或创建正式 output root 前，先完成所有 frozen hash、C8 PASS report/selection/双代码绑定、精确 20-ID/320-unit plan、preflight 与预算验证；任何不一致 fail closed。真实 GPU 只运行精确 C9 正式计划，不额外做探索性 candidate、selection 重跑或 C10。
8. L0/L1 覆盖全部 320 expected units；L2 只允许两个 audit calibration；L3 只允许 audit/显式失败 display 且永不作定量输入。禁止逐 quadrature step、逐像素或逐 dimension 文件。每个 unit 是新的不可变独立 attempt；只有 `complete=true`、status→manifest hash 正确、全部 artifact hash/bytes 正确且 plan/provenance binding 一致时才可 resume-skip。failed/incomplete/tampered 不得续写或伪装成功；修复后使用新的空 attempt/root，并完整保留旧证据。
9. failure table 必须左连接覆盖全部 320 expected units，每个 unit 有显式 terminal state；batch complete 与 successful 分开报告。任何 failed、missing、tampered、OOM、退化、misclassified、structured NA 都保留，不能丢行、补零、换 target/head/baseline 或挑选最好 attempt。C8 的四个旧失败/中断 roots、原始 BLOCKED report 及正式 root 的 `(11,5)` 真实失败继续保留，不得删除；该失败不是 C9 winner，也不得借机修改 selection。
10. 每 unit provenance 必须绑定 clean Git、model/image/lock/registry/metadata/source manifest/protocol/acceptance/config/schemas/selection/RunSpec/plan、role/audit、method/version/target、projection/sign/normalization、baseline/schedule/seed/扰动、status/failure、同步 wall、phase timers、CUDA allocated/reserved、forward/backward calls、全部普通文件 bytes 和完整 inventory。结束后做完整 batch/status/manifest/artifact/provenance/lifecycle/hash/inventory validation；PNG/display 不得进入数值链。
11. frozen hard caps 不变：64 MiB L0+L1/unit、1 GiB L2/audit sample、32 GiB C8+C9 artifacts、30,000 forward 和 backward calls/unit、90 GiB peak CUDA reserved/unit、1,000 H20-GPU-hours C8+C9；同时复核 Stage D 64 GiB、2,000 H20-GPU-hours、4 hours/image 投影。C8 后基线为 C9 `21.893448287103737` H20-GPU-hours/`440416200` bytes，C8+C9 `171.04256474299797` H20-GPU-hours/`3967765325` bytes，全部 10 caps PASS。必须重验，不能放宽或用删 workload 制造 PASS。
12. primary `(0,9)` 的 15/15 holdout validity evidence 必须 complete、finite、hash/provenance-valid、nondegenerate 且 endpoint/IG/token parity 与 required numeric contracts 有效。若不满足，原样保留并将 C9 明确报告为 BLOCKED；不得剔除样本、改变分母或在 C9 提前输出 `NO_GO_VALIDITY`。all-20 只作 descriptive，5 calibration 不进入确认性统计；C9 只生产/验证证据。
13. 明确禁止修改 frozen registry/config/schemas/protocol/acceptance、selection 文件、winners/primary、方法语义、阈值、seed、target、baseline、schedule、扰动、retention、caps 或 decision gates；禁止依据 holdout、正确性、图片内容、旧数值、display PNG、热图美观或人工偏好调参；禁止 C10 的 aggregate table、mean/median/SD、10,000 image bootstrap、CI、Holm correction及任何 `GO`/`NO_GO_VALIDITY`/`NO_GO_NOVELTY`/`NO_GO_COST`/`INCONCLUSIVE` 决策；禁止启动 Stage D。
14. 测试必须覆盖精确 20 IDs/order/5+15+2 roles、四 winners/primary、selection hash/tamper/manual-edit、holdout 必须有 validated selection、未选 candidate 拒绝、精确 320 及全部展开计数（尤其 global 每图一次）、canonical/perturbation tensor hash 共享、target 冻结、audit-only L2、L3 不回流、全部 320 failure rows、显式终态、partial failure/missing/tamper/crash、immutable retry/new root/non-overwrite、完整 final validation，以及 bootstrap/Holm/decision 不发生。按 §8.1：启动 HEAD/clean+C0 validator；开发中跑 C9 与直接受影响 tests；最终 tracked 状态跑一次受影响 focused、一次 C0 `--require-tracked --require-clean` validator、一次全量 non-GPU suite，区分复用的 C8 证据与本次实际运行。
15. 完成后更新本文：C9 标 `PASS` 或 `BLOCKED`；只有满足 C9 合同时才把 C10 标 `NEXT`。追加完整 C9 completion record，包含 before/after commit、clean 状态、文件、全部 input/config/artifact hashes、实际正式 root（届时据实填写）、control/status/manifest hashes、测试、320 终态分布、primary 15/15 validity evidence、偏差、保留失败、限制和 next_task；再写完整可复制 C10 提示词但不要执行。一任务一提交，最终工作树 clean。
~~~

### 2026-08-24 14:08 +0800 — C9 completion record

- status: PASS
- objective_completed: 只完成 C9。从 C8 机器冻结 selection 出发，按 registry `selection_order` 在精确 20 张图上运行四个冻结 winner `(0,9)`、`(4,2)`、`(6,10)`、`(11,2)` 的三种 Value 方法和每图只运行一次的四种 global 方法，生成并封存精确 320 个逐 execution-unit 证据。正式结果为 320/320 succeeded、primary `(0,9)` 的 15/15 holdout validity 完整、全部十项 frozen hard caps PASS。没有修改 selection、winner、primary、协议或阈值；没有执行 C10 的科学汇总、bootstrap/CI、Holm 或任何 Go/No-Go 决策。
- git_sha_before: `c03f01ff2feb0f250b49974688b5fa8d511161b1`（C8 completion 的 clean baseline）。
- git_sha_execution_and_finalize: `d8190b130d00d3b19fe7a586103a3a9cdb55e43d`。该 clean commit 是唯一正式 C9 root、320 个 GPU attempts、finalize report 和只读 final validation 的代码 producer；由本地只读引用 `archive/c9-execution-d8190b1` 保留。
- git_sha_after: SELF（即包含 C9 实现、测试、本 completion record 与状态推进的最终 C9 completion commit；精确 SHA 在提交后由 `git rev-parse HEAD` 产生并在最终回复报告。除 completion 文档和对应静态文档断言外，生产代码与 `d8190b1...` 一致）。
- upstream_commits: C8 GPU producer `1fd25392968c9ccfde6c066f6544e9c4557ac4f0`；C8 selection generator `d7f2ef3bca8c22e1fc1ae38d9333fd2cd3e11579`；C8 completion `c03f01ff2feb0f250b49974688b5fa8d511161b1`。
- upstream_c8_evidence: C8 plan canonical digest `ece22206df300d4099c888d9ce83be83bbf926eaf4d5003059694912881f0892`、RunSpec canonical digest `42f22bb4992dd560382acd48a5e5ec2fe47b0345835e9201636943fb22e7e30b`、preflight canonical digest `175d5c9c8461dd5b082792a57ffb7f627c62f2c540114b8789039ebc334b7a2b`；C8 control SHA-256 为 `execution_plan.json=da24efd1a700c59ae378a87b7b5b575f5d09edcbf40111c515f1321b2db899f9`、`preflight.json=c7d3a9baa1b95c2c6b1fdd9af70b9714c18db9230e34e437a401e94a329899f2`、`c8_launch_authorization.json=86a60e0a3c915b1295a803ace0c9ac6c414ef99a0dac7a879b30abd8663999fe`、`failure_table.json=deacdadd2c7ebe42b253bd1da86a264664e1d4a885c95cb8ed92e2ecfce327d0`、`batch_status.json=7c37bb76ae42d445912b953e582a02b30a16107ba9d5ce43e3fa1f4199f6a708`。
- tree_clean: 正式 authorization/init、六个 GPU shards、finalize 和独立 `c9-validate` 均绑定 `d8190b1...` clean tree；最终 SELF 提交后再次运行 clean/tracked validator、focused 和全量 non-GPU suite并确认 clean。由于生产 validator 会核对 live commit，后续重验正式 root 必须在 `archive/c9-execution-d8190b1` 的 clean checkout/worktree中执行，不能从completion-only HEAD伪造 observed Git state。
- files_added_or_changed: 新增 `config/stage_c_c9_run_spec.json`、`src/vitnns/stage_c_c9.py`、`tests/test_stage_c_c9.py`；扩展 `src/vitnns/stage_c_cli.py`、`tests/test_stage_c_cli.py`、`tests/test_stage_c_plan.py`、`tests/test_stage_c_contract.py` 和 `README.md`；本记录更新本文。冻结 registry、metadata、`config/stage_c.json`、三个 schemas、protocol、acceptance、selection、方法公式、target、baseline、schedule、seed、扰动、retention、caps 与 decision gates 均未修改。
- inputs_and_sha256: model `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`；registry `9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0`；registry metadata `bedb9df9a77807c86da0b898f8d44767c8276f21ab69821965f2b30d7184a4af`；Stage A 500 manifest `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`、manifest metadata `9a8df6201fe85c8529e2d307ee09142b3b55960dca059a29882d2e03f933ad5b`；official devkit `b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953`、synset order `70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15`；dependency lock `81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984`；tracked selection与 recovery copy file SHA-256 `5dff9b5aec77cab22b157b9bf2b5d63b2252b7e50ebe728250206d655a142254`、selection canonical digest `6615acb46a97b1a7f34402bbe07d1de542ff12c7f82f321eb00160a3168eae48`；C8 canonical PASS report file SHA-256 `d921d95ebe756e8cec859faea96578f8b3da766e9049e0440f570487d2e40b44`、canonical digest `9eb0df50be93320d7d6279852441c5bc103eaef146b394ff8d128cc64d2c9a6c`；C8 calibration artifact manifest digest `705b92c6101b5b4354eb3d2b922aced8d781dcf5ee6d3b7198a6132d9a495788`、calibration evidence digest `08daa501d3e18ba927180156537de86a3c62fac6176dde3f3a6a37f0916e1620`。按 registry 顺序的20图 SHA-256：`00000079=cf8f179939645bfab0734fbc75a189e7b3ef9624164cd84f98d6967b16b6a245`、`00002607=7863343e66b9eecf83d992bbf1d6b2350967dcc12e98b8cd6f4a64af8eb6ca1a`、`00005279=50bcce86763f9e29e48db4505c829a527bca3cb72927bbabe7fd3eceb6e1937e`、`00007489=819ddfee90f2778febabe0997bdad0c8371f3c15a58753be7342ff17b74304fb`、`00009888=156ee1e37de4ffc4ef8b9b9642aecbda4e258638ea0cc6d0b67515598d6349f6`、`00012573=dd169838bf09b9cbd9ba60371d4976342978d186b58602476c3b8ca6b41baa67`、`00014787=52295cc54a19e0c9ffe3875d8c7ec6bcfa29c80fdf2cd6b910a1e4830f228367`、`00018380=0862e62c900372644030b1743f85164126e350f3195efcf13676a9ccf4e9412b`、`00020970=7850e16768e48dea9fb6c78fc5dd524a4657e07774f8fff8a93dafb3b26f9183`、`00023229=7134518eefc783019c16abee47f1d940ba4d32bedf0f4ab6e61c8ac9f180a4a7`、`00025927=8e2e72e008bfde2cedc17767175a013cd6bef5187006897660b8b179c759c656`、`00028908=6cbb15fc1b00482bf696bc1ee1ae29544ecad34cc75206cbc097ef148fd7012f`、`00031099=69a8e5c985df0118bac27acc022194b6b14cc2b7f301d3eb84a87a52f7c1761d`、`00033512=f347fa9228fedf17412b0869c5c3daa8e58511764c7b6edd9ed77fc6d0a2cf7d`、`00035680=e62469766f493ed0162ed578e8bf6ac8969e7a3718ab002f29b00e7b92d2f91a`、`00038565=055fb65d37f8a21f68b662d02b62a397551b707321b5686791985e7b192d03fa`、`00041511=3d9b5abe890b3649076ec6719f9d187c6e443ccc5bcf1bdfef63b21ef4187cf7`、`00044187=d56bb5ecf152fcf14cfd48e36593f8a687429f0f39aed77e2db5cd3f62879455`、`00047079=7502ddd84e3c48b241ce3dbe60d82521531648836a287d1b3843125c4bd4cd24`、`00049970=6990cca7cdf9ed84f8708bf3098f56828601bce0a9a99109a58ddfb9f08984d4`。没有新增或替换输入图像。
- configs_and_sha256: frozen Stage C config `0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669`；config schema `94209b7c8a63a62ef27f3396810c8d5be8f6fbbf0a4df22e663b3661b9ed3f4b`；registry schema `4ce0ac8cd5bff93ae2b323c3e0447a87fc825c3243ad46c54c9b48e05e9b2d6d`；selection schema `867a3cf22519b2a045d81751a20c5cd52570ddeadede26a2da2bb79973f0b59d`；protocol `ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8`；acceptance `f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd`；C9 RunSpec file SHA-256 `6f36f2e1064c0377e337f88fefe7e5184f1fe24ac58e03f85ca9262bfd62f74f`、semantic/canonical digest `935a7e006e6a88dc7cd583b967fd02a057029942ad7e1087195a5deaee4456b8`。
- artifacts_and_absolute_paths: 唯一正式 C9 execution/finalize root 是 `/data00/home/yeziqi.831/testcode/vitnns_c9_formal_d8190b1_20260824`；final report 是其内的 `c9_evidence_report.json`。root 精确含 320 个 plan unit directories、320 个且仅有 `attempt-000001`，没有 plan 外 unit、重复 attempt、symlink 或恢复拼接。全树最终有 2,705 个普通文件、`225628646` bytes（含最终 report），有序完整 file inventory canonical digest `37fa9720165d8c6990b7db7f3946df1617e48ec5c50b79dd684d657e8fd87ad0`。unit artifact inventory 有 2,056 项：L0 640 files/`102962058` bytes、L1 1,280 files/`1167360` bytes、L2 136 files/`21746752` bytes、L3 0；L2 仅两个 audit calibration，各 `10873376` bytes。
- artifact_manifest_or_status_hashes: final report file SHA-256 `64334bc0901b23dab56e6902264579a5fab72ee0bf2123573b64512053599e59`、canonical digest `753883975d26ee8133b55ea03aaba34d2b90a8c14b646c7cf15b3be1db6b534b`；post-C9 budget canonical digest `07ca5bbb956f8f70ce0e423c084b50281d9b5332b31b080ff840a4fa24997f9c`；C8→C9 handoff canonical digest `6eadccbaea1cecd7f242e031d0a1b54a2231b54d7ff780ee4c0365d6e8537a3c`。正式 control file SHA-256：`execution_plan.json=f52b5a60ccccb8df0e3e4f0e118f95c512ae57568e63f0e07c42af4672bfa843`、`preflight.json=7bf49bec4775c634890b523e008baae4780492c1c005bf295c4b06bb1508ab5a`、`c9_launch_authorization.json=d037795f8c90f3c2c767d7058d2e576d6be24e613ff2c936d9b9bde129883a36`、`failure_table.json=bacbde75f1a5d0c12c0a1273fc7e14fff65c802422731b016eda0ea201dda3cf`、`batch_status.json=804b62c60608d6a28676f84ef9b03b16c456a789d28f021affa7cd2cad10bfce`。plan canonical digest `f0a4f0c1218b44c818d54e4584597bc0e1427d4fbf9b405b32a7b642121de592`、preflight canonical digest `fe66ae20feb0e1a2bbf605122233d2192b6216cba9bdf46f49605dc834eeca9f`、authorization canonical digest `1b09ed010e6e9b43298271e26c03e7a9fd84e82208a439a0ffba53146d1c8d8e`。按 plan 顺序对 320 个文件 hash records 做 canonical JSON 的集合摘要：status index `9c6ebd341ff166d24509c679b1a9fd7a05896b9438ad8a5d5857b3f9ef2900f6`、manifest index `c375f9c7347d793e1f54e783eb0b8dd025035e398f8b00fa7c03f22e750079cb`、unit artifact inventory index `ec9cdb587dbed42eb69d455d41c1f9b20c4d3f81ba33a787b9d1cb3e2523a5d1`；全部 320 status/manifest hashes各自唯一且 status→manifest 绑定有效。
- inventory_digest_recomputation: 上述集合摘要使用项目 canonical JSON（sorted keys、compact separators、UTF-8、`allow_nan=false`）对 `{"records": [...]}` 求 SHA-256。status/manifest records 严格按 final report 的 plan 行顺序，字段分别为 `unit_identity_sha256` 加对应文件 SHA-256；artifact records 按同一 plan 顺序和每个 manifest 内 artifact 顺序，字段为 unit digest、path、level、bytes、SHA-256；whole-root records 按 POSIX 相对路径升序，字段为 path、bytes、SHA-256。
- tests_and_results: 开发末受影响 C9/C8/CLI/plan suite `184 passed`；`py_compile` 和 `git diff --check` PASS。`d8190b1...` clean commit 上 C0 `--require-tracked --require-clean` validator PASS；post-commit focused `168 passed`；全量 non-GPU `896 passed, 1 skipped, 1 deselected`，唯一 skip 为 CUDA unavailable 的显式 GPU resource test，唯一 deselect 为 `unsafe_pickle`。正式运行前完整只读 authorization PASS 且 root 尚不存在；正式 `c9-init` PASS；六个 H20 round-robin shards 分别完成 `54/54/53/53/53/53` units且全部 exit 0；`c9-finalize` PASS；独立 `c9-validate` exit 0。早期只读 runtime audit 对稳定快照 64/64 succeeded units 通过生产级验证，最终独立审计再对 320/320 全量重验通过。C8 的 2,180-unit source evidence在 authorization、init、finalize、validate边界被复算，但没有重跑 C8 GPU。
- acceptance_results: 正式 plan 精确为 20 samples（5 calibration、15 holdout、2 audit）、四个冻结 winners、每图 12 Value units加4个 global units；320 execution units、240 fidelity-eligible、80 structural-reference、10,560 fidelity curves、221,760 curve points、1,312 map runs、223,072 WorkItems、80 global class-line units。failure table 精确 320 行且状态分布为 `succeeded=320`；batch `finalized`、`orchestration_complete=true`、`successful=true`、`integrity_valid=true`、`errors=[]`。20/20 样本各自 16 units 的 canonical/normalized clean tensor hash、三种 perturbation hash和 prediction context一致。primary `(0,9)` 的 15 个 holdout（`00002607`、`00005279`、`00007489`、`00012573`、`00014787`、`00018380`、`00020970`、`00025927`、`00028908`、`00031099`、`00033512`、`00038565`、`00041511`、`00044187`、`00047079`）全部 complete、finite、hash/provenance-valid、nondegenerate且满足 frozen endpoint/IG/token numeric contracts；`primary_holdout_failures=[]`。report 明确 `selection_performed=false`、`bootstrap_performed=false`、`holm_performed=false`、`decision_performed=false`。
- budget_results: 全部 10 项 frozen hard caps PASS。observed/limit 依次为：L0+L1 `466601/67108864` bytes per unit；audit L2 conservative `488599600/1073741824` bytes（C9 两个 audit 各实际 `10873376` bytes）；C8+C9 projected artifacts `4815802825/34359738368` bytes；Stage D projected artifacts `11414216200/68719476736` bytes；forward `5558/30000`、backward `4608/30000` batched calls per unit；CUDA reserved `6268387328/96636764160` bytes per unit；C8+C9 `188.6543225615848/1000` H20-GPU-hours；Stage D `603.6938321970714/2000` H20-GPU-hours；Stage D/image `1.2073876643941428/4` H20-GPU-hours。C9 单独保守投影 `24.147753287882853` H20-GPU-hours、`548965000` bytes；实际 C9 retained（含 final report）`225628646` bytes，实际 C8+C9 retained `2189749067` bytes。最慢有效同步 unit `217.3297795909457s`，budget measurement 为 2,499/2,499 usable（C8 2,179 succeeded加C9 320 succeeded）。
- deviations_or_protocol_amendments: 无 protocol amendment、无科学 workload、selection、winner、primary、阈值、seed、target、baseline、schedule、扰动、retention、caps 或 decision gate 变更。启动前仅加固 formal-root 原子认领、post-launch integrity error分类、final-report自身字节 fixed-point和fresh budget revalidation测试；不改变数值语义。因 GPU 0/3 已被其他作业占用，使用空闲 H20 1/2/4/5/6/7 做六路内部 round-robin，计划顺序与 unit 集未变。首次通过 `conda run ... pytest` 未设置正确 module环境而在 collection 前失败，改用 `PYTHONPATH=src`；一次新增测试的局部 spy 变量位置错误导致 183 passed/1 failed，修正后 184 passed；一次 `conda run` heredoc未转发 stdin而未执行只读预演，随后用目标环境 Python重跑。三者均发生在正式 root 创建前或纯测试阶段，没有 GPU/holdout/runtime artifact副作用。
- failures_retained: 正式 C9 没有 failed、missing、incomplete、tampered、OOM 或 retry，且没有创建失败 C9 root。C8 的四个历史失败/中断 roots、原始 BLOCKED report和正式 C8 root 的 `(11,5)` 真实 terminal failure均原样保留，未删除、resume、拼接或用于替换 C9 winner；候选 `/data00/home/yeziqi.831/testcode/vitnns_c9_formal_e51dc2d_20260824` 从未创建。
- known_limitations: C9 只是冻结 20 图 pilot 的完整证据生产，不是统计结论或普遍 ImageNet 声明；C9 PASS 不是 `GO`/No-Go 决策。C10 必须只用15张holdout作为确认性 image clusters，5张 calibration和all-20只能 descriptive。冻结合同规定三个 exploratory winners在 metric family内做 Holm alpha 0.05，但当前协议/config没有明确注册原始 p-value 的检验统计、零假设、单/双侧方向、family精确成员或 bootstrap-to-p-value规则；C10不得在看过holdout后发明这些口径，若冻结输入中仍无法找到唯一规则，应输出结构化 `HOLM_BLOCKED_UNDERSPECIFIED` 并按 fail-closed验收处理。runtime artifacts在Git外，依赖上述绝对路径、hash和本地 archive ref保存。
- next_task: C10

#### 下一对话完整提示词

~~~text
你现在只执行 ViT-NNS 重建工程“阶段 C”的 C10：从 C9 已冻结且完整验证的逐 execution-unit 证据生成不丢行的逐样本表、冻结统计汇总和唯一 Stage C 决策。不要运行新的模型推理或 GPU workload，不要重跑、替换或修改 C9 unit，不要启动 Stage D，不要修改 selection、四个 winners、primary、registry/config/schemas/protocol/acceptance、方法语义、阈值、seed、target、baseline、schedule、扰动、retention、caps 或 decision gates。

工作目录：
/data00/home/yeziqi.831/testcode/vitnns_rebuild

C9 completion 与正式证据绑定：
- C9 completion commit: SELF，即包含 C9 实现、本 completion record 与状态推进的最终 clean commit；启动 C10 时先运行 `git rev-parse HEAD` 并与 C9 最终回复给出的精确 SHA 完全一致。若 HEAD 不一致、工作树 dirty 或 dirty 归属不明，先查明且不得覆盖用户改动；不能建立精确 clean baseline 时 C10 必须 BLOCKED。
- C9 GPU/evidence producer commit: `d8190b130d00d3b19fe7a586103a3a9cdb55e43d`（clean），由本地只读引用 `archive/c9-execution-d8190b1` 保留。不得声称 runtime evidence 由 completion-only SELF commit 产生。
- C9 formal root: `/data00/home/yeziqi.831/testcode/vitnns_c9_formal_d8190b1_20260824`。
- C9 final report: `/data00/home/yeziqi.831/testcode/vitnns_c9_formal_d8190b1_20260824/c9_evidence_report.json`；file SHA-256 `64334bc0901b23dab56e6902264579a5fab72ee0bf2123573b64512053599e59`；canonical digest `753883975d26ee8133b55ea03aaba34d2b90a8c14b646c7cf15b3be1db6b534b`。report 为 PASS、integrity_valid/successful/orchestration_complete均true、320/320 succeeded、primary holdout 15/15 valid。
- execution plan file SHA-256 `f52b5a60ccccb8df0e3e4f0e118f95c512ae57568e63f0e07c42af4672bfa843`；canonical digest `f0a4f0c1218b44c818d54e4584597bc0e1427d4fbf9b405b32a7b642121de592`。
- C9 RunSpec file SHA-256 `6f36f2e1064c0377e337f88fefe7e5184f1fe24ac58e03f85ca9262bfd62f74f`；canonical digest `935a7e006e6a88dc7cd583b967fd02a057029942ad7e1087195a5deaee4456b8`。
- preflight file SHA-256 `7bf49bec4775c634890b523e008baae4780492c1c005bf295c4b06bb1508ab5a`；canonical digest `fe66ae20feb0e1a2bbf605122233d2192b6216cba9bdf46f49605dc834eeca9f`。
- authorization file SHA-256 `d037795f8c90f3c2c767d7058d2e576d6be24e613ff2c936d9b9bde129883a36`；canonical digest `1b09ed010e6e9b43298271e26c03e7a9fd84e82208a439a0ffba53146d1c8d8e`。
- failure table SHA-256 `bacbde75f1a5d0c12c0a1273fc7e14fff65c802422731b016eda0ea201dda3cf`；batch status SHA-256 `804b62c60608d6a28676f84ef9b03b16c456a789d28f021affa7cd2cad10bfce`。
- C8→C9 handoff canonical digest `6eadccbaea1cecd7f242e031d0a1b54a2231b54d7ff780ee4c0365d6e8537a3c`；post-C9 budget canonical digest `07ca5bbb956f8f70ce0e423c084b50281d9b5332b31b080ff840a4fa24997f9c`。
- ordered status index digest `9c6ebd341ff166d24509c679b1a9fd7a05896b9438ad8a5d5857b3f9ef2900f6`；ordered manifest index digest `c375f9c7347d793e1f54e783eb0b8dd025035e398f8b00fa7c03f22e750079cb`；ordered unit artifact inventory digest `ec9cdb587dbed42eb69d455d41c1f9b20c4d3f81ba33a787b9d1cb3e2523a5d1`；whole-root ordinary-file inventory digest `37fa9720165d8c6990b7db7f3946df1617e48ec5c50b79dd684d657e8fd87ad0`。
- post-C9十项caps全部PASS；实际 C8+C9 retained `2189749067` bytes；projected C8+C9 artifacts `4815802825` bytes/H20-hours `188.6543225615848`；projected Stage D artifacts `11414216200` bytes/H20-hours `603.6938321970714`/每图 `1.2073876643941428`。

继续绑定 C8 与冻结输入：
- C8 completion commit `c03f01ff2feb0f250b49974688b5fa8d511161b1`；C8 producer `1fd25392968c9ccfde6c066f6544e9c4557ac4f0`；selection generator `d7f2ef3bca8c22e1fc1ae38d9333fd2cd3e11579`。
- selection file SHA-256 `5dff9b5aec77cab22b157b9bf2b5d63b2252b7e50ebe728250206d655a142254`、canonical digest `6615acb46a97b1a7f34402bbe07d1de542ff12c7f82f321eb00160a3168eae48`；primary `(0,9)`，exploratory `(4,2)`、`(6,10)`、`(11,2)`。
- model `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`；registry `9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0`；registry metadata `bedb9df9a77807c86da0b898f8d44767c8276f21ab69821965f2b30d7184a4af`；Stage C config `0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669`；config schema `94209b7c8a63a62ef27f3396810c8d5be8f6fbbf0a4df22e663b3661b9ed3f4b`；registry schema `4ce0ac8cd5bff93ae2b323c3e0447a87fc825c3243ad46c54c9b48e05e9b2d6d`；selection schema `867a3cf22519b2a045d81751a20c5cd52570ddeadede26a2da2bb79973f0b59d`；protocol `ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8`；acceptance `f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd`；dependency lock `81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984`；Stage A manifest `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`。

第一步必须完整阅读 `docs/阶段C-step-by-step.md`（尤其 C9 completion record、§8.1、C10边界和交接规则）、`docs/阶段C_protocol.md` §5–§10、`docs/stage_c_acceptance.md` C6/C8/C9/C10、`docs/stage_c_config_contract.md`、config与三个schemas、tracked selection、C9 RunSpec、C6 aggregate/runner/integrity/provenance、C8 selection、C9 finalization/validation/CLI实现及测试、README。先在主工作树检查 HEAD/clean并运行C0 `--require-tracked --require-clean` validator，逐字节复算上述 frozen/C9 hashes。然后从本地引用 `archive/c9-execution-d8190b1` 创建一个新的临时 detached clean worktree（不得修改或删除主工作树、正式root或archive ref），在该producer worktree用其`PYTHONPATH=src`只读重跑 C9 final validation；验证后可正常移除该临时worktree。不得从completion-only HEAD伪造`observed_git_state`或放宽commit检查。要求report仍为PASS、320/320 succeeded、320行与plan一一对应、primary 15/15 valid、errors=[]、10 caps PASS。任何不一致 fail closed；禁止从 failed/missing/incomplete/tampered evidence做统计，禁止补零、删行、换root、挑attempt或重跑unit。

C10 精确范围：
1. C10只读消费 sealed C9 root，不运行模型、不读JPEG、不启动GPU或Stage D。先实现最小 C10 aggregation/statistics/decision、CLI和validation，并在新的clean code commit后创建全新独立C10 output root；不得写入或修改C9 root。
2. 生成machine-readable、可hash验证的per-sample long table（CSV或Parquet并配严格schema/manifest）。每个预期sample/config/method/metric必须有一行；failed、misclassified、degenerate、合法NA、invalidating NA均保留。不得丢行、补零、均值插补、改变分母、换target/head/baseline或只选有利结果。每行绑定C9 unit key/attempt/status/manifest/artifact hash、role/audit、method/config/target、metric、value/status/NA reason和源字段路径。
3. 严格分开5 calibration（只作calibration/descriptive）、15 holdout（唯一确认性样本和primary decision输入）、all-20（只作descriptive）。audit只是calibration overlay，不增加n；统计单位始终是image，layer/head/pixel/patch/seed/扰动/repeat均不能增加n。
4. 对所有预注册报告指标保留逐图值，并分别输出mean、median、sample SD（ddof=1）和95% bootstrap CI。Bootstrap只对15个holdout image indices有放回整簇抽样，每次抽15个clusters，精确10,000次，statistic=paired median，base seed `20260821`；每metric seed为 `stage-c-bootstrap|20260821|metric_id` 的SHA-256前8 bytes按big-endian unsigned解释；CI为2.5%/97.5% percentile linear quantile。同一paired comparison共享相同image draws。输出metric_id、seed、resamples、cluster size、quantile method和bootstrap distribution hash。五个fidelity tie repeats是单图内nuisance repeats，不能扩张n。
5. Primary `(0,9)` validity必须核验15/15 holdout complete、finite、hash/provenance-valid、endpoint parity `<1e-6`、IG conservation `<=1e-4+0.01*abs(delta)`、token parity `<=1e-6+1e-5*abs(r)`、无direction degeneration、两个calibration audit repeatability满足`atol=1e-6,rtol=1e-5`、无静默排除/display输入。任一required failure按最高优先级产生 `NO_GO_VALIDITY`。合法secondary class-line NA不改变Value-path validity但必须保留；class target delta退化不得换class/baseline；misclassification不得删样本。
6. Primary effect只使用primary `(0,9)` 的15个holdout fine-4×4-cell `delta_fid = 0.5*((InsertionAUC_NNS-InsertionAUC_matched)+(DeletionAUC_matched-DeletionAUC_NNS))`，NNS为`conditional_fixed_clean_attention_head_direction_value_path_ig`，matched为`matched_fixed_clean_attention_patch_token_contribution`。令m为median、n+为`delta_fid>0`图数、[L,U]为paired image-bootstrap median 95% CI：PASS_EFFECT iff `m>=0.02 && n+>=12 && L>0`；FAIL_EFFECT iff `n+<12 || U<0.02`；其余UNCERTAIN_EFFECT。patch-scale只作支持性报告，CI边界相等按uncertain。
7. Primary spatial novelty逐图使用`s_patch`、patch positive evidence/total token L1和clean `rho=||R(M)||1/||M||1`；至少12/15必须在同一图同时满足`>=0.01`、`>=0.01`、`>=0.05`，否则FAIL_SPATIAL。常量residual、零分母或相关NA是该图spatial failure，不得从分母删除。
8. 对`translate_right_down_4`、`center_crop_216_resize_224`、`gaussian_rgb_std_0_01`分别用15图clean/aligned residual signed Spearman：PASS iff正值数>=12、median>=0.50、bootstrap CI lower>0；FAIL iff正值数<12，或`median<0.50 && CI upper<0.50`；其余UNCERTAIN。任一扰动FAIL→FAIL_SPATIAL；无FAIL但任一UNCERTAIN→UNCERTAIN_SPATIAL。其他stability/Pearson/JS/Jaccard只描述，不新增gate。
9. 三个exploratory winners `(4,2)`、`(6,10)`、`(11,2)`分别输出同结构逐图与descriptive/holdout summary，明确exploratory且不得影响唯一primary decision。冻结config要求`holm_within_metric_family_alpha_0_05`，但当前protocol/config只写了family概念和alpha，没有注册原始p-value检验统计、零假设、单/双侧方向、family精确成员或bootstrap-to-p-value规则。不得在看过holdout后自行发明规则。先做机器可验证contract audit：若冻结输入中确有明确且唯一原始p-value定义，按每metric family对恰好三个exploratory winners做Holm step-down并保留raw p、排序、阈值、adjusted p、reject和family成员；若没有，输出结构化`HOLM_BLOCKED_UNDERSPECIFIED`，保留全部exploratory描述结果，不生成伪p-value、不将exploratory结果用于primary decision，并将C10 completion按fail-closed验收标为BLOCKED。不得用post-hoc amendment恢复已访问holdout的确认性身份。
10. Cost只复用C9 post-run冻结证据，不运行Stage D；逐项保留64 MiB L0+L1/unit、1 GiB L2/audit、32 GiB C8+C9、30,000 forward/backward/unit、90 GiB CUDA reserved/unit、1,000 C8+C9 H20-hours、64 GiB Stage D、2,000 Stage D H20-hours、4 H20-hours/image。Stage D cost gate只在validity/effect/spatial全部PASS后影响decision，不得重估成宽松口径或删workload。
11. 唯一decision及优先级严格为：validity fail→`NO_GO_VALIDITY`；validity pass且effect或spatial任一FAIL→`NO_GO_NOVELTY`；无FAIL但effect或spatial任一UNCERTAIN→`INCONCLUSIVE`；novelty全PASS且Stage D cost fail→`NO_GO_COST`；全部PASS→`GO`。同时保存全部flags但只输出一个primary decision；CI恰等边界为INCONCLUSIVE。不得把无显著差异写成等价；没有预注册equivalence margin/TOST。
12. 输出至少包括immutable per-sample table及schema/hash/row-count manifest、calibration/holdout/all-20三套分离summary、bootstrap metadata和distribution hashes、exploratory/Holm report或`HOLM_BLOCKED_UNDERSPECIFIED`、validity/effect/spatial/stability/cost flags、320-unit failure/NA reconciliation、machine-readable `decision.json`、Stage C acceptance/report、provenance和完整inventory。PNG/L3/display不得进入数值链；低raw-attention correlation或大JS本身不是novelty；不得声称等价、普遍ImageNet优越、upsampled map是原生像素证据或Value-path NNS是class/完整attention解释。
13. 测试必须覆盖C9 report/control/unit tamper与missing/incomplete/failed拒绝、no-row-drop、精确5/15/20角色分离、cluster=image/n=15/10,000 draws/seed golden/linear quantile/paired draws/ddof=1、NA/failure/misclassification保留、effect/spatial/三扰动的PASS/FAIL/UNCERTAIN及等号边界、validity最高优先级和完整五级decision priority、exploratory不进入primary、Holm三项/family/alpha合同及未注册p-value时fail closed、cost顺序、L3不回流、deterministic输出、禁止Stage D/GPU/模型推理/修改frozen inputs。
14. 按§8.1运行启动HEAD/clean+C0 validator、开发中C10及直接影响测试、最终tracked focused、C0 `--require-tracked --require-clean`和全量non-GPU suite；区分复用C9 evidence与本次实际测试。一任务一提交，最终工作树clean。完成后更新本文：C9保持PASS；C10只有满足完整合同才标PASS，否则BLOCKED；不启动Stage D；追加完整C10 completion record。若GO只写完整Stage D提示词但不执行；若非GO或Holm合同不足则写相应失败诊断/协议处置提示词但不执行。
~~~

### 2026-08-24 21:26 +0800 — C10 completion record

- status: BLOCKED
- objective_completed: C10 的只读聚合、逐样本长表、三套分离统计、冻结 bootstrap、全部 primary gates、唯一 primary decision、exploratory descriptive summaries、Holm contract audit、cost reuse、provenance、完整 inventory 和 source-recomputing validation 均已完成。C9 保持 `PASS`；C10 acceptance 因冻结输入未注册可唯一执行的原始 p-value 合同而按要求为 `BLOCKED`。独立科学决策为 `NO_GO_NOVELTY`，Stage D 未启动且未授权。
- git_sha_before: `8c65e38cdd36bf4fe6a5ea6cf6d81f913f7b9de0`（C9 completion clean baseline）。
- git_sha_aggregation_producer: `92ad81342ff22e2710df495a9961a3ab8c015a3a`（tree `3551096e358391fe04952e8ced19ea60884fb32f`，`src/vitnns` subtree `6874e71aec0e592e110a540a1638ea2cfc697c8d`）；本地只读引用 `archive/c10-aggregation-producer` 精确指向该 commit。唯一正式 C10 root 的 build、内建 source-recomputation validation 和独立 `c10-validate` 都由该 clean producer 执行。
- git_sha_after: SELF（即只增加本 completion record、终态文档断言和状态推进的 clean completion commit；正式 C10 数值产物不得归因于 SELF）。
- files_added_or_changed: producer commit 新增 `src/vitnns/stage_c_c10.py`、`stage_c_c10_source.py`、`stage_c_c10_stats.py`、`stage_c_c10_table.py`、`tests/test_stage_c_c10.py`、`tests/test_stage_c_c10_integration.py`，扩展 `src/vitnns/stage_c_cli.py`、`tests/test_stage_c_contract.py`、`README.md`；completion commit 只更新本文和对应静态断言。未修改 registry/config/schemas/protocol/acceptance、selection、四个 winners、primary、方法语义、阈值、seed、target、baseline、schedule、扰动、retention、caps 或 decision gates。
- formal_source: 只读消费 `/data00/home/yeziqi.831/testcode/vitnns_c9_formal_d8190b1_20260824`；C9 report SHA-256 `64334bc0901b23dab56e6902264579a5fab72ee0bf2123573b64512053599e59`、canonical digest `753883975d26ee8133b55ea03aaba34d2b90a8c14b646c7cf15b3be1db6b534b`、whole-root inventory digest `37fa9720165d8c6990b7db7f3946df1617e48ec5c50b79dd684d657e8fd87ad0` 均复核一致；C9 producer 仍为 `d8190b130d00d3b19fe7a586103a3a9cdb55e43d`，不是 C9 completion commit。C10 未读取 JPEG、未运行模型/GPU、未运行或写入 Stage D、未修改 C9 root。
- formal_output_root: `/data00/home/yeziqi.831/testcode/vitnns_c10_formal_92ad813_20260824`，精确 12 个 ordinary files、`524796892` bytes；whole-root inventory digest `382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`。10-payload manifest SHA-256 `507ca3c4ac210882a5918cc602cf7a4656aa398af76ff9f3638a77f3af42a0d8`、canonical digest `c71ae7c19941eab556a0ed9fdde553cb1a3fae914e149652df9b31cf3538720a`；status SHA-256 `192d383d5c4788f612018532074c06d48511513edd7565dbb99934706cd96db2`、canonical digest `d08bcb62a0fa95fef81c75d955b88082a8e6788a76b1a0bfffe1aa4e0181eb87`，且 `complete=true`、`terminal=true`、`downstream_authorized=false`、`stage_d_started=false`。
- table_and_schema: `per_sample.csv` 精确 `301224` rows、`423060118` bytes、SHA-256 `237bd223affd1df382dfbaf35bcbe37cbe353f64b4414c63aee3a5c0fb1e2137`；schema SHA-256 `51a58a3e32f1c5a882e0b05d4516d94b5beedd90a2a70e2f87fe6f78a9835b21`、canonical digest `37e69c5a561369f863cf5892a5788794cc232798d716cee2c0eecb4bdb03ddf3`；row manifest SHA-256 `dd3f7f61339cda85fa6347ae9f214c229473e25e217bd96e94ed8ee6cd9ec465`、canonical digest `895475c8f155f090d4e899baae83052b4b2f56955735d8c1a83dc350d849a9a6`。行空间严格分解为 base L0 `297844`、derived L1 `1780`、derived scientific `1600`；exact applicability audit `PASS`，301224 coordinates digest `7bea3b8a92c85791d5bcee5c67c0ea20fdff8570f7e433c0576e57e66bc20af3`；ordered row hashes digest `c2881654b9d9247a18b4e368c040c9271567e936f86513ee555451ed4ce1d816`，ordered row keys digest `ba7a1161a5924d0c8c9420f29f2aa67737ffee565c31736a70ee574ae279afb1`。
- reconciliation: 320 expected/observed units、`succeeded=320`、failure count `0`；row states 为 `OK=299097`、`NA=2127`，无 `INVALID`/`UNIT_FAILED`/`UNIT_INCOMPLETE`；NA reasons 为 `unsupported_sign_semantics=1980`、`constant_left=24`、`constant_right=92`、`zero_mass_left=8`、`zero_mass_right=23`。全部 NA、六个 misclassified sample 和每个 source binding 均保留；`no_rows_dropped=true`、`zero_imputation_performed=false`。reconciliation SHA-256 `30086f99e6d40942623883cbc87cd0d45c30de6e5400b6b0a4c9c539717ef5c7`、canonical digest `47b3635ea0b74be22fc8207f450005facb19d0ac6dace7bd3ceb36f2bcf9ddd7`。
- summaries_and_bootstrap: calibration/holdout/all-20 精确为 `5/15/20` images；四个 winners 均有同结构 per-image 与 descriptive/holdout summaries，primary 仅 `(0,9)`，其余三个明确 exploratory 且不进入 primary decision。summaries SHA-256 `8d99cb98dd3406fa68df73a0a03285ae654c5c8101ba25d9e34d6d6e64a7de0d`、canonical digest `ab3382c43d1fba80a3aa518a9f290ee518d13fcc63b0d2e3a09bde51a59bde72`；bootstrap SHA-256 `4506d1fb33849c12bddffa5e85c3e5daec369f7a6e5de5289b0f60313c4b7b7d`、canonical digest `fbe2956f059a199f2e52ede2cd7d38c858fcf211b51f416ac3a6e3e3fe0b1cfa`，共 14562 records，固定 image cluster `n=15`、10000 resamples、paired median、base seed `20260821`、PCG64、linear percentile，五个 tie repeats仅先在图内汇总。
- primary_gate_results: `PASS_VALIDITY`（15/15 holdout valid，32/32 audit-unit repeatability checks 通过）；`PASS_EFFECT`（fine-4×4 `median(delta_fid)=1.7534972022995547`、positive `15/15`、95% CI `[1.3317595184568727,3.3813212483296478]`）；clean joint spatial `PASS_JOINT_SPATIAL`（15/15）；translation `FAIL_STABILITY`（positive `4/15`、median `-0.05139490909545863`、CI `[-0.07456761309774369,0.005677349888892924]`）；center crop `FAIL_STABILITY`（positive `15/15`、median `0.03777216991075736`、CI `[0.0250692852605999,0.04055193037640095]`）；Gaussian `PASS_STABILITY`（positive `15/15`、median `0.9991571856327753`、CI `[0.9989445508867781,0.9994641428015386]`）。因此 overall `FAIL_SPATIAL`；十项 caps 继续全部 PASS，Stage D cost gate 为 `PASS_COST`，但因 novelty 已 fail 不改变优先级。
- decision_and_acceptance: 唯一 primary decision 为 `NO_GO_NOVELTY`；decision SHA-256 `07fc5875e46018adf5dc0cd09967e273df3b443c1deeb1653d672d05103e513e`、canonical digest `60e112d50f22229474a634039ba71c7eed1adba081d8e1097f850a8a4b4d8dc0`。Stage C report SHA-256 `9a16034d29bc87eaba852aa0e618928ac0e4ecee9bf745d765f42311dd4563b4`、canonical digest `90232df2010f6a8fc771baeac3aaac44fee93d436597cf0cbdebe3cf8d2142e2`。这不等同于“方法无效”：冻结 fidelity effect 很强且 clean spatial evidence 通过；No-Go 由 translation/crop residual stability hard gates 失败触发。
- holm_blocker: Holm report SHA-256 `be37ffbfd2ba60b5549ba791076383d9ba53a3b437178273a608fbf527b91781`、canonical digest `271ba3625de6a91912728a32880cf1e1861aecb9f12903cf759e83e3332ab2fd`，状态 `HOLM_BLOCKED_UNDERSPECIFIED`；冻结 config/protocol 未唯一注册 test statistic、null/alternative、sidedness、exact three-member family 或 raw-p-value method，故没有生成伪 raw/adjusted p-values，exploratory 只作描述且不影响 primary decision。不可在已访问 holdout 后通过 post-hoc amendment 恢复这批样本的确认性身份。
- paper_scope_note: `/data00/home/yeziqi.831/testcode/论文1/参考文章.pdf`（SHA-256 `33e323f8cb664864b12daad1f06b6ef0d3268bcafaea4e84ac695fd286f3855b`）没有注册输入 translation、center-crop/resize、Gaussian perturbation前后 NNS spatial stability/equivariance gate；其最接近内容是同输入 learning image 与 feature map correlation及高激活神经元 IoU。因此本项目的 spatial stability 是额外、更严格的预注册 robustness gate。该事实可在论文 discussion/limitations 中解释，但不能在已查看本轮 holdout 后撤销或降级当前 hard gate；若未来研究要把它改为参考项，必须版本化新协议并使用新的未见确认性样本。
- provenance_and_claims: provenance SHA-256 `4dcca54fdc83ba7401070ef843636b5e67b68a8ce720fa70f729e407fc5717f0`、canonical digest `4b26452041685cbb6c12d93139736e65fc4330fd44a64bf854848e7cff66c742`。只使用 L0/L1；L2/L3/display/JPEG 均未进入数值链。本次不声称等价、普遍 ImageNet 优越、upsampled map 是原生像素证据，或 Value-path NNS 是 class/完整 attention 解释。
- tests_and_results: 启动基线、C0 clean/tracked validator、全部冻结 input/C9 hashes、C9 detached producer native validation 均 PASS。开发末 C10 unit/integration `138 passed`；早期 C9/CLI/selection/contract focused `312 passed in 198.29s`，completion-state tracked focused 及提交后复验均为 `316 passed`（分别 `255.53s`、`250.57s`）；producer commit 和 completion commit 后 C0 `--require-tracked --require-clean` 均 PASS。正式 fresh-root `c10-build` 完成并以内建 source recomputation 复验，按预期因 acceptance `BLOCKED` 返回 exit 2；随后独立 `c10-validate` 再次 source-recompute，返回相同 report canonical digest `90232df...` 和预期 exit 2。completion commit 后全量 non-GPU suite 为 `1034 passed, 1 skipped, 1 deselected in 707.61s`；唯一 skip 是 CUDA unavailable 的显式资源测试，唯一 deselect 是 `unsafe_pickle`。全程 `CUDA_VISIBLE_DEVICES=''`。
- deviations_or_protocol_amendments: 无 protocol amendment、无 post-hoc gate 变更、无新 inference/GPU/JPEG/Stage D。一次最初 formal in-memory diagnostic 因 KL component selector 未进入 row 而在 coverage audit 中 fail closed，未落盘；修复后 diagnostic assembly 通过但因 fake provenance 只作诊断，未作为正式 evidence。第一次正式 build 在输出 root 创建前因最终 provenance review 发现 source pathspec 校验缺口而人工中止，确认目标 root 从未创建；修复并 amend producer/ref 后使用全新 SHA 命名 root完成正式 build。若干测试命令因系统 Python 缺 `timm`、不存在的测试文件名、wrapper 不接受 heredoc 或仅错误消息 regex而在执行前/测试层失败，均未修改科学 evidence。
- failures_retained: C9 正式 root 不含失败 unit；C10 对 tampered/missing/incomplete/failed source一律 fail closed，正式 C10 root无缺失或失败行。所有合法 NA/misclassification均显式保留。未创建的两个候选 C10 root `/data00/home/yeziqi.831/testcode/C10_DIAGNOSTIC_NOT_CREATED` 与 `/data00/home/yeziqi.831/testcode/vitnns_c10_formal_77cae12_20260824` 均确认不存在，无需删除或替换。
- known_limitations: C10 只是冻结 20 图 pilot，15 图 holdout 是唯一确认性单位，不能支持普遍 ImageNet 优越或等价结论。当前结果同时有独立的 `NO_GO_NOVELTY`（translation/crop hard-gate failure）和流程 `BLOCKED`（Holm underspecification）；两者都不授权 Stage D。
- next_task: STOP_STAGE_D_AND_DRAFT_VERSIONED_FOLLOWUP_PROTOCOL

#### 下一对话完整提示词（失败诊断 / 协议处置；不要执行 Stage D）

~~~text
你现在只做 ViT-NNS Stage C 的结果诊断与未来协议设计，不运行 Stage D，不运行新模型推理/GPU workload，不修改或重新解释已封存的 C9/C10 evidence，也不得把本轮 15 张已查看 holdout 重新标成确认性样本。

只读绑定：C10 formal root `/data00/home/yeziqi.831/testcode/vitnns_c10_formal_92ad813_20260824`；aggregation producer `92ad81342ff22e2710df495a9961a3ab8c015a3a` 和 `archive/c10-aggregation-producer`；C10 report SHA-256 `9a16034d29bc87eaba852aa0e618928ac0e4ecee9bf745d765f42311dd4563b4`、canonical digest `90232df2010f6a8fc771baeac3aaac44fee93d436597cf0cbdebe3cf8d2142e2`；decision 为 `NO_GO_NOVELTY`，acceptance 为 `BLOCKED`，唯一 blocker 为 `HOLM_BLOCKED_UNDERSPECIFIED`，Stage D 未授权。

请完成两部分只读分析：
1. 诊断为何 primary `(0,9)` 在 fine-4×4 fidelity 明显 PASS、clean joint spatial 15/15 PASS、Gaussian stability PASS，却在 4 px translation 和 center-crop/resize residual signed-Spearman 上 FAIL。区分真正方法不稳定、坐标/patch-grid/align-corners/插值/边界效应、残差定义及对齐评价本身的敏感性；只能用已封存表、摘要、源码和参考论文，不得挑样本、改分母、调阈值或运行新实验。
2. 草拟一个未来版本化协议（仅草案，不执行）：明确 spatial stability 是 primary hard gate、secondary endpoint 还是 discussion-only；为 exploratory 三-winner family 预先指定原始检验统计、零假设、单/双侧、方向、exact family membership、p-value算法和 Holm step-down；规定新样本 registry、全新未见 confirmatory split、冻结时点与 provenance。明确本轮已查看 15 holdout只能用于探索/设计，不能在未来协议中恢复确认性身份。

输出诊断报告和协议草案路径/哈希；不要启动 Stage D，也不要把原论文没有该 gate 当成撤销本轮预注册 gate 的理由。
~~~
