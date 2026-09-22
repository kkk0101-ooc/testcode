# Endpoint 与 Gradient × Input 等价下的创新定位与后续五步计划

> 执行状态（2026-09-22）：第一步已完成内容冻结，验收记录见第八节；第二步未启动。

## 一、结论

Endpoint 的基础公式为

$$
L_s^{\mathrm{Endpoint}}(x)=x\odot\nabla_x s(x),
$$

因此它与观测输入点上的 Gradient × Input 数学等价。这个等价关系是本文的方法起点，而不是需要回避的问题。本文的创新不建立在新的基础归因算子上，而建立在以下问题上：如何把这一局部测量原语扩展为面向 Vision Transformer 内部目标的结构化扫描器，如何对 attention 和 residual 的真实计算路径进行可闭合记账，以及这种测量能够揭示哪些可复现的 ViT 内部规律。

本文的中心定位为：

> **以 Gradient × Input 为局部测量原语，构建面向 ViT 异构内部目标的结构化、可闭合、可审计扫描框架，并利用该框架研究局部输入依赖如何随层深、attention 路径和 residual 分支重新组织。**

相应的英文中心论点为：

> **We turn Gradient × Input into a structure-aware and auditable scanner for heterogeneous internal targets in Vision Transformers, enabling exact differential accounting and systematic analysis of how local input dependence is reorganized across depth, attention paths, and residual branches.**

这里的技术贡献是完整框架及其经验用途，不是对标准算子重新命名。

## 二、不能作为创新主张的内容

以下内容不再作为论文的新颖性来源：

- $x\odot\nabla_x s(x)$ 不是新的基础归因量；
- 对任意内部标量目标直接计算 Gradient × Input，本身不足以构成方法创新；
- 在 ViT 上生成 Gradient × Input 热图，不足以构成结构解释方法；
- Endpoint 不具有相对 Taylor、IG 或有限干预方法的普遍优势；
- routing/value 和 identity/branch 分量是局部 differential accounting，不是因果贡献百分比；
- registry-complete 表示执行了全部预注册目标，不表示穷举了模型中的全部神经元；
- 单模型或低覆盖率抽样结果只能描述对应检查点和抽样单元，不能写成所有 Transformer 的普遍定律。

这些边界应集中放在方法身份、实验范围和局限性部分各说明一次。摘要、贡献列表和结果段落应优先陈述实际完成的对象、验证和发现。

## 三、真正可以主张的贡献

### 3.1 原始 NNS 的数学统一与对象澄清

在原始 NNS 的线性、卷积和当前输入 ReLU 门控传播条件下，learning-image propagation 等价于相应标量目标的 Gradient × Input。该结果将逐层传播规则统一为局部 Jacobian 表述，并建立以下边界：

- 原始 NNS 的 Endpoint 传播不同于沿参考路径积分的 Path IG；
- Endpoint 描述观测输入点上的局部一阶输入依赖；
- bias、CLS token、position embedding 和归一化 affine 参数属于模型状态或模型常数，不应被均匀铺回像素；
- canonical Endpoint 受输入坐标原点影响；在 normalized RGB 坐标中，该原点对应预处理定义的 raw-RGB 均值；
- Endpoint、Path IG、有限 Taylor 近似和有限替换效应对应不同 estimand。

这项贡献属于数学统一和方法身份澄清。它为后续 ViT 结构扫描提供定义基础，但不被表述为新归因公式。现有依据见 `docs/endpoint_nns_vit_method_identity_20260917.md` 和 `docs/NNS_ViT_Endpoint_Scanner数学分析与方法对比_20260918.md`。

### 3.2 从输出热图扩展到异构内部目标扫描

普通输出级 Gradient × Input 通常回答：某个输出分数对输入坐标具有怎样的局部依赖。本文进一步建立统一的内部目标空间，测量不同 ViT 内部标量目标对输入坐标的局部依赖。注册对象包括：

- token-channel；
- attention head；
- Q、K、V 和 attention output；
- routing 与 value differential；
- residual identity 与 branch；
- MLP branch；
- 类别 logit 和其他预注册标量读出。

这些对象共享统一输入坐标和目标接口，使不同层、模块和分支能够在同一测量语义下比较。创新对象因此是异构内部目标的统一测量空间，而不是单个热图公式。

仅增加目标数量仍然偏向工程实现。该目标空间必须与结构切面、闭合验证和经验问题共同构成论文的方法贡献。

### 3.3 结构化 differential cuts 与可验证记账

本文方法的核心是对真实 ViT 计算图定义结构切面，并对局部 Jacobian 进行可验证的分项记账。对于合法切面 $C$，要求每条有效的输入到目标路径恰好穿过一次，并满足

$$
L_s^{\mathrm{Endpoint}}(x)
=
\sum_{e\in C}L_{s,e}^{\mathrm{Endpoint}}(x).
$$

结构合同包括：

- attention 的 Q、K、V differential 恢复完整 attention differential；
- routing 与 value differential 恢复完整 attention differential；
- residual identity 与 branch 恢复 residual output differential；
- bypass 路径不得遗漏或重复；
- fan-out 后的不同 consumer use 必须分别注册；
- producer、consumer、input slot 和 invocation 共同定义端口身份；
- 所有闭合先在原始 signed input coordinates 上验证，再进行空间聚合和显示。

实现验证不依赖单一总 PASS，而由多类互补证据组成：

- 正向结构闭合；
- 负向 mutation controls；
- JVP/VJP 一致性；
- 中心有限差分；
- 原始 NNS 与 Endpoint 的数值等价复现；
- 独立验证器、覆盖率审计和不可变结果根。

漏边、重复边、错误 input slot、错误 fan-out 合并和错误 detach 必须触发预期失败。这里的新颖性候选是结构化局部 Jacobian 扫描、真实计算边切面和自动审计合同的联合能力。论文需要通过相关工作矩阵核实这一联合能力未被既有方法完整覆盖。

### 3.4 基于统一测量框架的 ViT 机制观察

方法价值最终体现在它能够回答哪些结构问题。现有封存证据支持以下经验结论：

- 在 ViT-B/16 和 DeiT-S/16 的冻结 token-channel 样本中，局部输入依赖随层深发生空间扩展；
- late attention 的局部 differential 更依赖 routing，Value-only 近似的偏差随深度增加；
- attention residual 后期呈现更强的 branch/routing 改写；
- MLP residual 后期呈现更强的 identity preservation；
- MLP path cancellation 在两个模型中方向不同，说明该现象具有模型依赖性；
- Endpoint mass 与有限成分替换效应存在广泛但非普遍的关联；
- 当问题指定 reference 和 replacement displacement 时，reference-matched Taylor 更贴合相应有限替换。

这些结果建立了一条明确的解释边界：

$$
\text{observed-input local dependence}
\neq
\text{reference-specific finite intervention effect}.
$$

Endpoint 适合表示和审计观测输入处的局部依赖。Taylor 或其他显式使用 replacement displacement 的方法更适合预测特定 reference 下的有限替换。两者回答不同问题，不应组织成跨任务的统一方法排行榜。

## 四、建议采用的三项论文贡献

### 贡献一：数学统一与对象定义

证明原始 NNS learning-image propagation 在明确条件下等价于内部标量目标的 Gradient × Input，并区分 Endpoint、Path IG、模型常数和有限替换效应。

### 贡献二：结构化扫描与审计框架

建立 ViT 内部目标 registry、真实计算边切面、attention routing/value 与 Q/K/V 分解、residual identity/branch 分解，并通过闭合、mutation controls、JVP/VJP、有限差分和独立验证器审计。

### 贡献三：机制观察与解释边界

在两个 ViT 上系统测量层深、attention 和 residual 的局部输入依赖规律，并通过有限干预标定 Endpoint 表示与 reference-specific intervention estimator 的区别。

三项贡献各自承担不同作用：第一项定义数学身份，第二项构成方法核心，第三项构成科学核心。

## 五、论文叙事和标题建议

### 5.1 正文顺序

建议正文按照以下顺序组织：

1. 原始 NNS 的数学身份和现有研究缺口；
2. ViT 异构内部目标和真实计算边切面的定义；
3. 可闭合 differential decomposition 与自动审计协议；
4. scanner coverage、数值正确性和 mutation 检出；
5. 层深、attention 和 residual 的机制结果；
6. 有限干预标定及 estimand 边界；
7. 与 Gradient × Input、IG、conductance、GlobEnc、DecompX、ALTI、LRP 和 activation patching 的关系。

Endpoint 与 Taylor 的比较不应成为论文开篇主线。该比较用于界定局部表示和有限替换之间的关系，适合放在结构机制结果之后。

### 5.2 推荐标题

中文首选：

> **从 NNS 到视觉 Transformer：内部目标的可审计 Jacobian 扫描与结构分解**

英文首选：

> **From NNS to Vision Transformers: Auditable Jacobian Scanning and Structural Decomposition of Internal Targets**

备选标题：

> **Structure-Aware Gradient × Input for Auditing Internal Targets in Vision Transformers**

首选标题更能覆盖数学统一、结构方法和经验研究。备选标题更直接地公开基础算子身份，但摘要必须立即说明目标 registry、differential cuts、闭合审计和机制研究，避免被理解为简单地将 Gradient × Input 应用于 ViT。

## 六、接下来怎么做：五步执行计划

### 第一步：冻结唯一论文定位和主张边界

**目标**

把全文从“新归因算法”统一调整为“结构化局部 Jacobian 扫描与机制研究”，停止多条互相竞争的定位。

**具体工作**

1. 以本文档的中心论点为唯一主定位；
2. 建立 claim registry，将所有主张分为 `数学身份`、`结构方法`、`实现正确性`、`经验发现` 和 `解释边界`；
3. 删除或改写“新基础归因量”“普遍优于”“全模型规律”“因果贡献百分比”等表述；
4. 将 PCIA–PEFA 的数学与审计结果保留为扩展或附录，不再作为 Endpoint 主论文的升级路线；
5. 将 CST aquatic-context NO_GO 保留为独立研究记录，不纳入本文主证据。

**交付物**

- 一页论文定位说明；
- 一份逐主张 claim registry；
- 标题、摘要中心句和三项贡献的中英文定稿。

**完成门**

摘要、引言、贡献列表、讨论和结论对 Endpoint 的数学身份使用同一表述；全文不再暗示基础 G×I 公式的新颖性。

### 第二步：完成联合能力的新颖性差分审计

**目标**

证明论文贡献来自结构化扫描与审计能力的组合，而不是通过改名获得。

**具体工作**

1. 系统比较 Gradient × Input、IG、conductance、GlobEnc、DecompX、ALTI、LRP、edge attribution 和 activation patching；
2. 对每项方法统一检查被解释对象、内部目标类型、真实 consumer edge、合法切面、attention routing/value、residual identity/branch、精确闭合、mutation 检出和经验用途；
3. 区分单项已有能力与本文联合能力；
4. 对已经被覆盖的部分直接引用和降级；
5. 只有在正式来源中未发现完整覆盖时，才将相应联合能力写成新颖性主张。

**交付物**

- 一张统一字段的 related-work difference matrix；
- 一段不超过三段的研究缺口说明；
- 每项方法贡献对应的最近邻工作和差异证据。

**完成门**

每个“首次”“提出”或“不同于既有工作”的句子都能回链到比较矩阵；仅有应用场景不同或名称不同不能通过该门。

### 第三步：重构方法章节和核心概念图

**目标**

让读者在方法章节中清楚看到从测量原语到结构化框架的技术增量。

**具体工作**

1. 先给出 Endpoint 与 Gradient × Input 的恒等式；
2. 定义内部标量目标 registry 和统一输入坐标；
3. 定义 computational edge、consumer use 和合法 differential cut；
4. 给出 attention Q/K/V、routing/value 与 residual identity/branch 的闭合式；
5. 将正向闭合、负向 mutation、JVP/VJP、有限差分和独立验证组织为统一审计协议；
6. 重绘核心概念图，使“测量原语—内部目标—结构切面—机制分析”的关系一图可见。

**交付物**

- 重构后的方法章节；
- 一张方法总览图；
- 一张结构切面与闭合关系图；
- 一张审计协议表。

**完成门**

读者无需依赖实验结果即可回答三个问题：基础算子是什么，本文增加了什么结构对象，当前实现如何证明没有漏项或重复记账。

### 第四步：重排实验结果并绑定封存证据

**目标**

让实验直接服务于三项贡献，不再围绕 Endpoint 是否击败其他归因方法组织全文。

**具体工作**

1. 先报告原始 NNS 等价复现、算子验证、结构闭合和 mutation 检出；
2. 再报告 registry coverage 与分层 token-channel atlas；
3. 将层深空间扩展、attention routing/value 和 residual identity/branch 作为主要科学结果；
4. 将 Endpoint–Taylor 和有限替换实验放在行为标定与解释边界部分；
5. 每项结论绑定对应封存 JSON、统计单位、效应量、区间和适用模型；
6. 将 PCIA–PEFA 和 CST 的 NO_GO 作为研究边界或未来设计依据，不混入主方法有效性证据。

**交付物**

- 重排后的 Results 章节；
- 逐主张 evidence index；
- 主文图表与补充材料分配表；
- 所有关键数字的机器重算校验报告。

**完成门**

每个主结果都有直接证据和明确统计单位；失败或局部结果不被总 PASS 覆盖；行为结果统一写为“广泛但非普遍”。

### 第五步：同步重写中英文论文并进行对抗式终审

**目标**

形成一套数学身份诚实、贡献集中、数字一致且能够回答主要审稿质疑的投稿版本。

**具体工作**

1. 同步改写中英文标题、摘要、引言、贡献、相关工作、讨论和结论；
2. 使用“主张—证据—简洁边界”的句式，删除 reviewer-facing 预辩解和重复免责声明；
3. 核对所有样本量、覆盖率、效应量、区间、任务数和模型范围；
4. 从审稿人视角集中检查三项风险：G×I 公式新颖性、与既有 Transformer 分解方法的差异、抽样覆盖对外推范围的限制；
5. 验证正文、附录、图注、表格和 evidence index 的中英文一致性；
6. 只有现有数据无法回答核心研究问题时才新增实验，不为改变既有阴性结论追加搜索。

**交付物**

- 中英文同步终稿；
- claim-to-evidence 对照表；
- 数字与引用自动校验报告；
- 一份模拟审稿意见及逐项修订记录。

**完成门**

全文能够用一句话回答“既然 Endpoint 等于 G×I，本文新增了什么”，且答案稳定落在异构内部目标、结构化 differential cuts、自动审计和机制发现四个方面。

## 七、最终判断

如果论文定位为新的 attribution formula，现有证据不支持其新颖性。如果论文仅展示 ViT 上的 Gradient × Input 热图，技术增量也不足。当前可成立的完整定位是：

> **原始 NNS 数学统一 + ViT 异构内部目标扫描 + exact structural differential cuts + fail-closed 审计 + 两模型机制研究 + 有限干预边界。**

这是一项测量框架与经验机制研究。其方法价值来自结构化局部 Jacobian 记账和验证体系，其科学价值来自对 ViT 局部输入依赖重组规律的系统测量。后续工作的重点是收紧论文叙事、完成相关工作差分审计并建立逐主张证据索引，而不是继续寻找一个新的基础公式。

## 八、第一步执行记录：唯一论文定位与主张边界冻结

冻结日期：2026-09-22。适用入口为 `papers/nns_vit_endpoint/publication_r2_20260920/{zh,en}/manuscript.md` 及其 `papers/nns_vit_endpoint/{zh,en}/full_draft.md` 镜像。后续步骤可以补充证据或收缩主张，不得静默改变本节的基础算子身份、证据层级和外推边界。`notes/revision_v3_frozen_claims.md` 保留为上一轮修订快照，不由本次冻结覆盖。

### 8.1 一页论文定位说明

**研究对象。** 本文研究观测输入点上 ViT 内部标量目标的局部输入依赖。基础测量原语为

$$
L_s(x)=x\odot\nabla_x s(x),
$$

即同一目标的 Gradient × Input。原始 NNS learning-image propagation 在精确当前点局部导数且无归一化、截断或 relevance redistribution 的条件下与该量等价。这个恒等式用于确定方法身份，不构成新基础归因公式。

**方法增量。** 本文把该原语扩展为 ViT 的结构化扫描框架：一是注册 token-channel、attention head、Q/K/V、routing/value、residual identity/branch、MLP branch 和类别 logit 等异构内部目标；二是在真实 producer—consumer edge 上定义合法 differential cut；三是在原始 signed input coordinates 上验证 Q/K/V、multi-head 及 residual 分项对完整局部微分的加和闭合；四是以公式回归、正向闭合、遗漏/重复 mutation controls、JVP/VJP、有限差分和独立验证器组成审计合同。方法主张相对于指定计算图、切面和参数化成立。

**科学问题。** 统一测量框架用于研究局部输入依赖如何随层深、attention 路径和 residual 分支重新组织，并在规定的有限替换任务中区分观测点局部表示与 reference-specific intervention estimator。经验结论限于 ViT-B/16、DeiT-S/16、冻结样本、注册目标和已执行替换算子。

**唯一中心定位。**

> **以 Gradient × Input 为局部测量原语，构建面向 ViT 异构内部目标的结构化、可闭合、可审计扫描框架，并利用该框架研究局部输入依赖如何随层深、attention 路径和 residual 分支重新组织。**

> **We turn Gradient × Input into a structure-aware and auditable scanner for heterogeneous internal targets in Vision Transformers, enabling exact differential accounting and systematic analysis of how local input dependence is reorganized across depth, attention paths, and residual branches.**

**排除范围。** PCIA–PEFA 仅作为独立扩展研究或附录材料，不构成 Endpoint 主论文的升级路线或主有效性证据。CST aquatic-context 研究及其 `NO_GO_NOT_IDENTIFIABLE` 历史结果保持独立，不进入本文的证据集合、贡献列表或结论。

### 8.2 Claim registry

| ID | 类别 | 冻结主张 | 当前依据 | 允许的正文强度 | 禁止扩大为 |
| --- | --- | --- | --- | --- | --- |
| MI-1 | 数学身份 | 在精确当前点局部导数且无额外重分配时，标量 NNS learning image 等于同目标的 $x\odot\nabla_xs(x)$。 | 命题 1；`m3a_fidelity.json` 的 5,000 项公式回归 | 方法身份和定义基础 | 新基础归因量、链式法则创新、一般函数值完整分解 |
| MI-2 | 数学身份 | Endpoint、Path IG、reference-matched Taylor 与有限替换效应是不同 estimand；canonical raw-RGB Endpoint 的坐标原点由 normalization mean 确定。 | 方法推导；M5 reference analysis | 明确测量对象与坐标条件 | 将局部量、路径量和有限效应互换解释 |
| SM-1 | 结构方法 | 统一 target registry 将预注册的 ViT 异构内部标量目标映射到同一输入坐标。 | ViT-B/16 457 个、DeiT-S/16 385 个注册目标；M3 冻结 atlas | 预注册目标空间与跨对象比较接口 | 穷举全部神经元或覆盖全模型 |
| SM-2 | 结构方法 | 合法 differential cut 对真实 consumer use 分项，并支持 Q/K/V、routing/value、multi-head 与 residual identity/branch 记账。 | Endpoint E3/E4 结构合同 | 指定图与切面上的结构化局部 Jacobian 分解 | 首次提出所有 Transformer 路径分解、唯一机制分解 |
| SM-3 | 结构方法 | 分项依赖计算图、切面和参数化；等价重参数化可保持总 Endpoint 而改变路径项。 | 正文 Section 4.4 的 $R/R^{-\top}$ 反例 | graph-relative structural decomposition | 一般实现不变性或仅由端到端函数决定的唯一解释 |
| IC-1 | 实现正确性 | 原始 NNS 兼容图上的传播与同目标 Input × Gradient 数值一致。 | 5,000/5,000 公式回归；float64/float32 误差记录 | 对已测算子与精度的数值身份 | 对任意未注册算子自动成立 |
| IC-2 | 实现正确性 | 冻结 ViT 实现的合法切面在正向检查中闭合，遗漏或重复边的负控被检出。 | 76,800 项正向闭合、28,800 项负控、6,144 项解析 JVP | 指定实现无漏项或重复记账 | 闭合即因果正确、函数值完整或有限干预有效 |
| EF-1 | 经验发现 | 两个固定 checkpoint 的冻结 token-channel 样本在五组独立抽样中均复现随深度空间扩展。 | M3 与 revision-v3 step 2；30/30 model–group–metric 方向一致 | 两模型抽样单元中的稳定层深趋势 | 所有 ViT 或全部内部单元的普遍规律 |
| EF-2 | 经验发现 | 深层 attention 的 routing 相对尺度与 Value-only error 提高；Q/K/V 具有不同层深轨迹。 | M4 与 revision-v3 step 2 全 block 统计 | 两个固定模型、已测目标上的结构趋势 | 将 norm 比例写成因果贡献百分比；给 Q/K/V 强加跨模型统一空间方向 |
| EF-3 | 经验发现 | attention 与 MLP residual 呈不同层深变化，且 MLP cancellation 的部分方向具有模型依赖性。 | M4 封存结果 | 分模型报告的 residual 机制观察 | 合并为所有模型共同规律 |
| EF-4 | 经验发现 | 训练参数状态与三个随机参数状态存在报告的空间扩展幅度和路径趋势差异。 | supplement-v1 `training_analysis.json` | 参数状态差异 | 训练强化、训练导致或训练过程因果效应 |
| EF-5 | 经验发现 | 局部路径图与规定成分替换效应的关联广泛但非普遍；原 all-family omnibus 保持 `NOT_SUPPORTED`。 | E6 `family_decisions.json`；11/12 family | 规定任务中的分层关联 | 普遍有效、普遍优于或局部图等于有限因果效应 |
| EF-6 | 经验发现 | 给定 reference 的规定区域替换中，displacement-aware Taylor 的排序优于 clean-only Endpoint。 | `m2_inherited.json` 与正文 Tables 6–8 | 指定 reference、区域、预算和替换算子下的任务比较 | Taylor 或 Endpoint 的无条件方法排行榜 |
| IB-1 | 解释边界 | local differential closure、path scale 与 finite replacement 分别回答正确性、局部尺度和规定干预效应问题。 | 方法定义与 RQ1–RQ4 分层 | 三类证据分别陈述 | 用任一证据替代另外两类 |
| IB-2 | 解释边界 | registry-complete 只表示全部预注册目标已执行；经验外推限于两个固定 checkpoint、冻结样本与抽样单元。 | registry 与 cohort 合同 | checkpoint- and sample-scoped inference | 模型全集、架构全集或全体神经元结论 |
| IB-3 | 解释边界 | routing/value 与 identity/branch 分项是局部 differential accounting。 | 结构定义、闭合与讨论 | 比较 signed/absolute local differential scale | 前向输出或因果贡献百分比 |

Registry 中每项主张只属于一个主类别。方法边界若同时影响解释，仍以其主要证明对象归类，避免用一个总 PASS 覆盖不同证据层级。

### 8.3 标题、摘要中心句与三项贡献定稿

**中文标题**

> 从 NNS 到视觉 Transformer：内部目标的可审计 Jacobian 扫描与结构分解

**English title**

> From NNS to Vision Transformers: Auditable Jacobian Scanning and Structural Decomposition of Internal Targets

**中文摘要中心句**

> 以 Gradient × Input 为局部测量原语，本文构建面向 ViT 异构内部目标的结构化、可闭合、可审计扫描框架，并用它系统研究局部输入依赖如何随层深、attention 路径和 residual 分支重新组织。

**English abstract central sentence**

> We turn Gradient × Input into a structure-aware and auditable scanner for heterogeneous internal targets in Vision Transformers, enabling exact differential accounting and systematic analysis of how local input dependence is reorganized across depth, attention paths, and residual branches.

**三项贡献**

1. **数学身份与对象定义。** 证明原始 NNS learning-image propagation 在明确条件下等价于同目标的 Gradient × Input，并区分 Endpoint、Path IG、模型常数与 reference-specific finite replacement。该贡献澄清测量对象，不提出新的基础归因公式。
2. **结构化扫描与审计框架。** 建立 ViT 异构内部目标 registry、真实计算边与合法 differential cuts，对 attention 与 residual 路径进行可加记账，并以正向闭合、mutation controls、JVP/VJP、有限差分和独立验证器审计实现。
3. **机制观察与解释边界。** 在两个固定 ViT checkpoint 的冻结样本上测量层深、attention 和 residual 的局部输入依赖规律，并以规定有限替换实验区分观测点局部表示与 reference-specific intervention estimator。

**Three contributions**

1. **Mathematical identity and object definition.** We establish the conditions under which original NNS learning-image propagation equals same-target Gradient × Input and distinguish Endpoint from Path IG, model constants, and reference-specific finite replacement. This contribution identifies the measurement object rather than introducing a new base attribution formula.
2. **Structured scanning and audit framework.** We define a registry of heterogeneous internal ViT targets, actual computational edges, and valid differential cuts; account additively for attention and residual paths; and audit the implementation with positive closure, mutation controls, JVP/VJP checks, finite differences, and independent validators.
3. **Mechanistic observations and interpretation boundary.** Across frozen samples from two fixed ViT checkpoints, we measure depth-, attention-, and residual-dependent reorganization of local input dependence and use prescribed finite-replacement experiments to distinguish observed-input local representations from reference-specific intervention estimators.

### 8.4 禁用表述与扩展隔离

- 禁用 `Endpoint 是新的基础归因算法` 及英文等价表述。
- 禁用 `Endpoint 普遍优于 Taylor/IG/有限干预方法`。
- 禁用将抽样目标写成全模型、所有 ViT 或全部 Transformer 的规律。
- 禁用将 path norm、routing/value、identity/branch 写成因果贡献百分比。
- 禁用将训练参数状态与随机参数状态差异写成训练导致或训练强化。
- PCIA–PEFA 可在独立扩展或附录中按自身合同报告，但不进入 Endpoint 主论文的标题、中心论点或三项主贡献。
- CST aquatic-context 及其终态只留在 `vit_nns_cst/` 的独立研究记录中，不作为 Endpoint 有效性证据或失败证据。

### 8.5 第一步 prompt-to-artifact 验收表

| 显式要求 | 直接产物 | 覆盖方式 | 状态 |
| --- | --- | --- | --- |
| 中心论点成为唯一主定位 | 本文 8.1；双语摘要、引言、讨论与结论 | `gxi_abstract_central_sentences`、`gxi_introduction_positioning`、`gxi_discussion_positioning`、`gxi_conclusion_positioning` | PASS |
| 五类 claim registry | 本文 8.2 | `gxi_claim_registry` 检查五个类别与 MI/SM/IC/EF/IB 入口 | PASS |
| 删除或改写四类越界表述 | 双语主稿与镜像；本文 8.4 | `gxi_no_prohibited_claims` 与 reviewer-facing 套话扫描 | PASS |
| PCIA–PEFA 降为扩展或附录 | 本文 8.1、8.4 | `gxi_scope_separation`、`gxi_no_extension_evidence_in_main` | PASS |
| CST aquatic-context NO_GO 独立保存 | 本文 8.1、8.4；`vit_nns_cst/` 原记录 | 主稿无 CST/aquatic；`vit_nns_cst/N2完成审计.md` 仍保存 `NO_GO_NOT_IDENTIFIABLE` | PASS |
| 一页论文定位说明 | 本文 8.1 | 研究对象、方法增量、科学问题、唯一定位、排除范围齐全 | PASS |
| 标题、摘要中心句、三项贡献中英文定稿 | 本文 8.3；双语主稿与镜像 | `gxi_frozen_titles`、`gxi_abstract_central_sentences`、`gxi_three_contributions:{zh,en}` | PASS |
| 摘要、引言、贡献、讨论、结论身份一致 | 双语主稿与镜像 | 五处均绑定 Gradient × Input 身份与结构化扫描增量；双语公式、章节、数据表一致 | PASS |
| 停在第一步 | Git diff 与执行记录 | 未新增第二步 difference matrix 或实验；只修改定位文档、现役双语稿、README 与校验器 | PASS |

### 8.6 防御性写作审计

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 双语摘要 | 段末以“而非新的基础输入归因量”界定贡献 | D2 / D14 | Major | 是 | 先描述工作再否认一种创新，容易形成预答辩 | 会继续追问公式是否被重新命名 | 先定义测量原语，再正面陈述方法增量 | Gradient × Input 是局部测量原语；本文构建结构化、可闭合、可审计的 ViT 内部目标扫描框架 |
| 双语引言 | “ViT 带来的新问题不是重新定义这一基础量” | D1 / MIXED | Minor | 是 | 以反驳误读开场，技术问题出现较晚 | 能接受边界，但会感到作者在预判质疑 | 将恒等式写成起点，随后直接给出总梯度缺失的结构信息 | 该恒等式确定局部测量原语；框架进一步区分 Q/K/V、heads 与 residual edges |
| 双语贡献列表 | 逐项先列已有工作再说“本文新增” | D14 / MIXED | Major | 部分 | 每项贡献都在防御新颖性，削弱主线 | 贡献像 related-work rebuttal | 用“贡献名—技术对象—直接证据”组织，先例留在相关工作 | 数学身份与对象定义；结构化扫描与审计框架；机制观察与解释边界 |
| 双语讨论 | “既有数学基础上实现”作为首句 | D14 / MIXED | Minor | 否 | 首句先降格，再说明研究对象 | 会认为作者不确定论文中心 | 用唯一定位正面概括 | Gradient × Input 是测量原语；贡献是异构目标的结构化 Jacobian 扫描与可审计记账 |
| 双语结论 | 再次说明身份“并非主要创新” | D2 | Minor | 否 | 摘要和引言已完成边界说明，结论重复否定 | 结尾显得自我削弱 | 正面归纳框架和证据层级 | Gradient × Input 是基础测量原语；结构化扫描、可验证记账和两模型机制研究构成统一框架 |

#### Paper-level defensive-writing score

修订后为 **2/10**。核心段落按“主张—证据—简洁边界”组织，数学身份只在会改变解释的位置保留。

#### Highest-priority removals

1. 删除摘要、引言和结论中重复的“不是新基础归因量”句式，只在方法身份和贡献边界各说明一次。
2. 将贡献列表从“已有工作—本文新增”的预答辩结构改为三个具名贡献及其直接证据。
3. 将讨论首句改为唯一正面定位，不以自我降格开场。
4. 所有有限干预结论保持任务限定，不写跨任务方法排行榜。

#### Necessary caveats to preserve

- NNS 与 Gradient × Input 的等价要求精确当前点局部导数且无额外重分配。
- 路径分项依赖指定计算图、切面和参数化。
- 结构闭合不等于有限干预效用。
- 两个固定 checkpoint 与冻结抽样不支持全模型、训练因果或架构普遍性结论。

#### Repeated phrase audit

现役双语主稿需保持下列 reviewer-facing 套话为 0：`We emphasize that`、`It should be noted that`、`We do not claim`、`To avoid misunderstanding`、`Taken together`、`需要强调的是`、`应当指出`、`为避免误解`、`我们不声称`。

### 8.7 完成门结论与验证记录

完成门通过。摘要、引言、贡献列表、讨论和结论均把 Gradient × Input 写成基础测量原语，把本文增量写成 ViT 异构内部目标的结构化 Jacobian 扫描、合法 differential cuts、可审计记账和机制研究。现役主稿未发现将基础 G×I 公式写成新颖性来源的表述。

验证记录：

- `.conda/envs/vitnns/bin/python3.10 vitnns_rebuild/papers/nns_vit_endpoint/scripts/validate_publication_r2.py`：`PASS`，1,726 项检查，0 errors；其中 10 项为本次定位冻结专项检查，并重算 165 个真实显示面板。
- 双语入口镜像重算：`zh MIRROR_PASS`、`en MIRROR_PASS`。
- `git diff --check`：`PASS`。
- 当前改动范围审计：仅涉及本计划、现役中英文主稿及镜像、便携包 README、现役发布校验器和其生成的验证报告；未运行新实验，未生成第二步 related-work difference matrix。
- 历史 `validate_revision_v3_step2.py` 仍绑定旧 `修改意见v3.md` SHA `5e39a5...`，而仓库当前该文件在 HEAD 即为 `6a9863...`。该历史专项校验会仅在 `opinion_source_sha` 项失败；本轮未修改其源意见、历史合同或科学结果，也不将该旧校验作为第一步完成代理。
