# Endpoint 与 Gradient × Input 等价下的创新定位与后续五步计划

> 执行状态（2026-09-22）：第一步已完成内容冻结，验收记录见第八节；第二步已完成联合能力的新颖性差分审计并通过 1,734 项发布校验，验收记录见第九节；第三步已完成方法章节和核心概念图重构，验收记录见第十节；第四步已完成实验结果重排与封存证据绑定，验收记录见第十一节；第五步已完成中英文同步重写与对抗式终审，验收记录见第十二节。

## 一、结论

Endpoint 的基础公式为

$$
L_s^{\mathrm{Endpoint}}(x)=x\odot\nabla_x s(x),
$$

因此它与观测输入点上的 Gradient × Input 数学等价。这个等价关系确定了本文的测量原语。对固定目标 $s$，本文进一步定义一个同时保留原始输入坐标与具体 consumer-use edge 的目标条件化双索引局部微分对象。设 $e$ 表示由 producer、consumer、input slot 与 invocation 联合确定的实际使用边，$z_e(x)$ 表示进入该 consumer use 的张量，$\bar z_e^{(s)}$ 表示在该 consumer input slot 处、尚未与同一 producer 的其他 fan-out use 合并的 use-specific target cotangent，则

$$
\boxed{
\mathcal L_{e,i}(s;x)
=
x_i
\left[
J_{z_e,x}(x)^{\mathsf T}\bar z_e^{(s)}
\right]_i
}.
$$

其中 $i$ 是原始输入坐标，$e$ 是具体 consumer-use edge，$s$ 是预注册的内部或输出标量目标条件。自由索引为 $(e,i)$；不同 $s$ 通过 target registry 分别实例化。所有 Jacobian 和 cotangent 均在同一个观测输入 $x$ 上求值。若 $C$ 是合法切面，即每条从输入到目标的有效路径恰好穿过一个已登记 use，则

$$
\boxed{
\sum_{e\in C}\mathcal L_{e,i}(s;x)
=
x_i\frac{\partial s(x)}{\partial x_i}
}.
$$

右侧是已有的 Gradient × Input，左侧则把总图展开为 $[\text{edge},\text{input coordinate}]$ 结构图组。本文的方法增量位于这个双索引对象、合法切面合同、结构审计和由此支持的机制问题，而不是右侧算子或链式法则。

本文的中心定位及论文摘要入口为：

> **对每个注册目标，本文将总 Gradient × Input 展开为同时索引原始输入坐标与具体 consumer-use edge 的局部微分图组；合法切面恢复同一总局部微分，并支持 attention 与 residual 路径的空间化层深分析。**

> **For each registered target, we expand total Gradient × Input into local differential maps jointly indexed by original input coordinates and concrete consumer-use edges. Valid cuts reconstruct the same total local differential and enable spatially resolved analyses of attention and residual paths across depth.**

这里的技术贡献是目标条件化双索引表示及其完整框架和经验用途，不是对标准算子重新命名。当前有来源边界的新颖性主张为 `NOV-JOINT-1`：在本轮核查来源中，未发现一个方法同时保留输入坐标与具体 consumer-use edge 身份，对同一当前点真实局部微分执行合法切面上的 attention/residual 分项闭合，并以结构负控和独立数值 oracle 审计后用于层深机制分析。

该创新主张需要三层证据共同成立：

1. **对象成立。** $\mathcal L_{e,i}(s;x)$ 的 edge、input 和 target 索引均有明确计算图语义；
2. **合同成立。** 合法切面、signed closure 和已封存故障检出共同检查结构图组在已测切面与故障类型中没有遗漏或重复记账；
3. **用途成立。** 至少一项主要机制结论必须依赖 $[e,i]$ 联合信息，而不能由总 G×I 图、纯 edge score 或 token-only decomposition 单独恢复。

只有第一层时，贡献是链式法则的工程实现；前两层成立时，贡献是可验证的结构化测量框架；三层共同成立时，论文才具有实质的方法与经验创新。

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

### 3.2 目标条件化双索引局部微分对象与异构内部目标扫描

普通输出级 Gradient × Input 只保留输入索引 $i$，回答某个输出分数对哪些输入坐标具有局部依赖。普通 edge score 只保留边索引 $e$，回答某条边在指定评价量下有多大总体作用。本文保留联合索引 $[e,i]$，使同一输入位置的局部依赖可以按实际计算路径展开，也使同一计算边的作用保留输入空间分布。统一 target registry 再引入目标索引 $s$，测量不同 ViT 内部标量目标的局部输入依赖。注册对象包括：

- token-channel；
- attention head；
- Q、K、V 和 attention output；
- routing 与 value differential；
- residual identity 与 branch；
- MLP branch；
- 类别 logit 和其他预注册标量读出。

这些对象共享统一输入坐标、edge identity 和目标接口，使不同层、模块和分支能够在同一测量语义下比较。创新对象因此是 $\mathcal L_{e,i}(s;x)$ 所定义的目标条件化双索引测量空间，而不是单个热图公式。

仅增加目标数量仍然偏向工程实现。该目标空间必须与结构切面、闭合验证和经验问题共同构成论文的方法贡献。

两个索引缺一不可：

- 删除输入坐标 $i$，只能得到 edge-level scalar，不能研究各路径的空间扩展、位置重分配和 signed spatial cancellation；
- 删除 edge 身份 $e$，只剩总 G×I 图，不能区分 routing/value、Q/K/V、multi-head 或 identity/branch；
- 删除目标 $s$ 的注册合同，跨内部对象的数值将缺少共同语义，无法形成可复现的 target atlas；
- 删除合法切面和 signed closure，结构分项可能漏掉或重复有效路径，绝对值图仍可能显得合理；
- 删除结构负控和数值 oracle，无法排除 hook、edge identity 与局部微分实现错误。

### 3.3 结构化 differential cuts 与可验证记账

本文方法的核心是对真实 ViT 计算图定义结构切面，并对 $\mathcal L_{e,i}(s;x)$ 进行可验证的分项记账。对于合法切面 $C$，要求每条有效的输入到目标路径恰好穿过一次，并满足

$$
\sum_{e\in C}\mathcal L_{e,i}(s;x)
=
x_i\frac{\partial s(x)}{\partial x_i}.
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

结构错误应通过负控或独立 oracle 被检出。当前封存负控覆盖遗漏非零 head、遗漏非零 routing 和重复 identity；attention detach 由独立一致性检查覆盖。错误 input slot 与错误 fan-out 合并尚无封存 mutation 证据，不得写成已验证能力。`NOV-JOINT-1` 的新颖性不来自“闭合”一词，而来自同一对象上同时保留输入坐标、consumer-use edge 与目标身份，并用合法切面和审计合同把结构分项绑定回同一个 current-point local differential。

### 3.4 基于统一测量框架的 ViT 机制观察

方法价值最终体现在双索引对象能够回答哪些总图或纯 edge score 无法回答的结构问题。现有封存证据支持以下经验结论：

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

后续论文必须把这些结果组织为“双索引对象的必要性证据”：总 G×I 只能给出完整 attention 或 residual 图，而 $\mathcal L_{e,i}$ 才能同时保留路径身份与空间位置，进而比较 routing/value 和 identity/branch 的层深重组。若某项结论可由总图、单一激活统计或现有 edge scalar 同等得到，则该结果不能承担 `NOV-JOINT-1` 的经验支撑。

## 四、建议采用的三项论文贡献

### 贡献一：数学统一与对象定义

证明原始 NNS learning-image propagation 在明确条件下等价于内部标量目标的 Gradient × Input，并区分 Endpoint、Path IG、模型常数和有限替换效应。

### 贡献二：结构化扫描与审计框架

定义对固定目标同时保留输入坐标与 consumer-use edge 的 $\mathcal L_{e,i}(s;x)$，建立 ViT 内部目标 registry 和真实计算边切面，将 attention routing/value、Q/K/V 与 residual identity/branch 表示为同一局部微分的可加 input-coordinate 分项，并通过闭合、已封存的遗漏/重复结构负控、JVP/VJP、有限差分和独立验证器审计。

### 贡献三：机制观察与解释边界

使用同一双索引测量族比较层深、attention routing/value 与 residual identity/branch 的空间化局部输入依赖；以现役双语稿已封存的两 checkpoint 结果为最低证据，并在第四步审计后将 E5 的四公开检查点前瞻复现作为优先增强证据。有限替换实验用于标定该表示与 reference-specific intervention estimator 的区别。

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

> **面向视觉 Transformer 内部目标的双索引局部 Jacobian 扫描与结构审计**

英文首选：

> **Dual-Indexed Local Jacobian Scanning and Structural Auditing of Internal Targets in Vision Transformers**

备选标题：

> **从 NNS 到视觉 Transformer：内部目标的可审计 Jacobian 扫描与结构分解**

首选标题直接呈现新增对象的双索引性质，降低被理解为普通 G×I 热图包装的风险。备选标题保留 NNS 的历史连续性。两种标题下，摘要都必须立即定义 $\mathcal L_{e,i}(s;x)$、合法切面、闭合审计和机制用途。

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

把 `NOV-JOINT-1` 写成可检验的方法对象，而不是组件清单。读者应能明确区分已有的 G×I 原语与本文新增的双索引表示，并理解为什么 input coordinate、consumer-use edge、target registry、合法切面和审计合同缺一不可。

**具体工作**

1. 先给出 Endpoint 与 Gradient × Input 的恒等式，明确它是总量测量原语；
2. 正式定义目标条件化的 $\mathcal L_{e,i}(s;x)$，给出自由索引 $i/e$、目标条件 $s$、$z_e$、use-specific consumer cotangent 和 current-point 求值条件；
3. 区分 producer tensor、consumer operator、input slot 与 invocation，说明同一 producer 的不同 use 为什么不能共享 edge identity；
4. 定义合法 differential cut，并给出 $\sum_{e\in C}\mathcal L_{e,i}=x_i\partial_i s$ 的命题、适用条件与证明；
5. 把 attention Q/K/V、routing/value、multi-head 和 residual identity/branch 写成 $\mathcal L_{e,i}$ 的具体切面实例；
6. 增加“去掉一个索引会失去什么”的方法消融表：无 $i$ 退化为 edge scalar，无 $e$ 退化为总 G×I，无 valid cut 失去可核验记账；
7. 将正向闭合、现有三类结构负控、JVP/VJP、有限差分和独立验证组织为统一审计协议；未封存的 wrong-slot、fan-out 和 detach mutation 不写成已验证结果；
8. 重绘核心概念图，按“总 G×I → $[e,i]$ 双索引展开 → 合法切面重构 → 层深机制问题”的顺序呈现；
9. 逐句回链 `NOV-ID-1`、`NOV-JOINT-1`、`NOV-ATTN-1`、`NOV-RES-1` 与 `NOV-AUD-1`，不产生新的单项首次主张。

**交付物**

- 重构后的方法章节；
- $\mathcal L_{e,i}(s;x)$ 的正式定义、合法切面命题与证明；
- 一张双索引方法总览图；
- 一张结构切面与闭合关系图；
- 一张索引必要性/退化对照表；
- 一张审计协议表。
- 扩展后的发布校验器及第三步专项复核记录。

**完成门**

同时满足以下条件才通过：

1. 读者无需依赖实验结果即可回答：基础算子是什么、双索引对象新增了什么、合法切面闭合什么、审计能排除哪些错误；
2. 方法章节明确证明 $\mathcal L_{e,i}$ 在合法切面上恢复总 G×I，但不将链式法则本身写成创新；
3. 图和正文能直观看出无 $i$、无 $e$、无 cut contract 三种退化；
4. 所有 mutation 主张与已封存负控逐项一致；
5. `validate_publication_r2.py` 可执行并通过第二步与第三步新增检查，`git diff --check` 通过。

### 第四步：重排实验结果并绑定封存证据

**目标**

证明双索引对象不是形式上的字段叠加，而是产生不可由总 G×I、纯 edge scalar 或单一激活统计直接恢复的机制证据；同时把跨检查点复现和阴性干预边界纳入同一证据结构。

**具体工作**

1. 先报告原始 NNS 等价复现、算子验证、结构闭合和 mutation 检出；
2. 增加“联合索引必要性”结果：用现有封存数组比较总图、edge scalar 和 $[e,i]$ 结构图组分别能回答什么，至少给出一个 attention 和一个 residual 的预注册或现有冻结汇总；
3. 把层深空间扩展作为背景规律，把 routing/value 与 identity/branch 的空间化层深重组作为 `NOV-JOINT-1` 的核心经验结果；
4. 对每项核心结果执行替代解释审计：总 absolute mass、activation norm、总 G×I 或 edge-only scale 是否已经足以解释观察到的趋势；若足以解释，则降级为描述性结果；
5. 审计 thesis-blind-review 中 E5 的来源、无重叠 cohort、四 checkpoint binding、统计合同和 sealed JSON。通过后将 H1–H3 四检查点方向复现同步到现役中英文稿；未通过则继续按两个固定 checkpoint 报告，不复制数字；
6. 将 H4 在四检查点均为负的结果作为解释边界：结构路径图不提供稳定的头排序增量，不能用外部复现只保留正结果；
7. 将 Endpoint–Taylor 和有限替换实验放在行为标定部分，明确 local representation 与 reference-aware intervention prediction 的 estimand 差异；
8. 每项结论绑定对应封存 JSON、统计单位、效应量、区间、检查点和适用 target；
9. 将 PCIA–PEFA 和 CST 保持为独立研究，不混入 Endpoint 主方法的有效性证据。

**交付物**

- 重排后的 Results 章节；
- 一张“总图/edge scalar/双索引图组”信息差异与结果必要性表；
- attention 与 residual 各一项双索引必要性结果或图；
- E5 四检查点证据审计与同步决定；
- 逐主张 evidence index；
- 主文图表与补充材料分配表；
- 所有关键数字的机器重算校验报告。

**完成门**

同时满足以下条件才通过：

1. 至少一个 attention 结论和一个 residual 结论明确依赖 $[e,i]$ 联合信息，且正文说明总 G×I 或纯 edge scalar 为什么不能直接恢复该结论；
2. 每个主结果都有直接封存证据、明确统计单位、效应量、区间和检查点范围；
3. E5 四检查点结果只有在 source binding、无重叠、统计合同和数字重算全部通过后进入现役稿；
4. 正结果、`NOT_SUPPORTED` 和相反模型趋势均完整保留，行为结果继续写为“广泛但非普遍”；
5. 不新增为了证明 Endpoint 胜出的搜索性实验；现有证据不足以回答必要性时，只允许提出一项前瞻、低成本且有停止门的补充实验。

### 第五步：同步重写中英文论文并进行对抗式终审

**目标**

形成一套以 `NOV-JOINT-1` 为唯一方法新颖性主轴、以双索引必要性和跨检查点机制结果为经验支撑、数字与边界一致的投稿版本。

**具体工作**

1. 同步改写中英文标题、摘要、引言、贡献、相关工作、讨论和结论；
2. 摘要和贡献二直接定义双索引对象的技术增量，避免用“全面、严格、新颖”等形容词替代对象、合同与结果；
3. 使用“主张—证据—简洁边界”的句式，删除 reviewer-facing 预辩解和重复免责声明；
4. 核对所有样本量、覆盖率、效应量、区间、任务数、检查点和模型范围；
5. 从审稿人视角执行四项对抗性检查：是否只是 G×I 加 hooks、是否只是多个已有组件的拼装、结构闭合是否被误写成因果正确、经验结果是否真的需要双索引对象；
6. 对 `NOV-JOINT-1` 做逐字段反例检查，使用 Conductance、DecompX/LRP、EAP/Vi-CD 和 Unpack 作为最强近邻，不使用弱基线制造差异；
7. 验证正文、附录、图注、表格、related-work matrix 和 evidence index 的中英文一致性；
8. 只有现有数据无法回答“联合索引是否必要”时才新增实验，不为改变既有阴性结论追加搜索。

#### 第五步硕士毕业论文评审合同

第五步的终审必须采用以下评审要求。该要求用于约束审查立场、审查内容、写作方式和最终裁定。

**评审立场**

1. 全部评审使用中文。
2. 评审标准采用硕士毕业论文评审专家的标准。首要目标是判断论文是否达到硕士毕业论文的学术要求，不以认可作者设想为目标。
3. 评审对象是一名希望论文能够通过盲审和答辩的研究作者。评审不提供信心确认，也不替作者辩护。
4. 评审语气保持冷静、专业和克制。发现证据不足、设计缺陷或结论越界时，直接指出问题。

**六个固定审查维度**

1. **研究问题与科学价值。** 判断科学问题是否清晰、聚焦。说明研究在当前文献中的真实位置，不以作者预期代替文献证据。检查 `NOV-JOINT-1` 是否对应一个必要且可回答的问题，而不是已有组件的并列组合。
2. **研究设计与因果逻辑。** 检查研究设计是否与研究问题一致。指出未讨论的因果假设、识别条件和不可验证前提。设计不能支持结论时，明确指出证据与结论不匹配的具体位置。重点检查结构闭合、局部路径尺度和有限替换是否被错误地相互替代。
3. **方法学与统计推断。** 检查模型设定、估计量、统计单位、不确定性评估和多重比较是否合理。指出模型依赖、参数敏感性、选择偏倚、测量误差和检查点选择问题。结果表面一致但方法脆弱时，分别评价结果方向与推断可信性。
4. **结果解释与外推边界。** 检查因果语言和机制语言是否越过证据。明确结论适用的模型、检查点、样本、目标、切面和干预条件。若替代解释同样合理且尚未排除，必须提出。检查四检查点复现是否只支持参数状态复现，而非训练因果或全部 ViT 的普遍规律。
5. **学术写作与叙事结构。** 判断论文是否围绕研究问题形成清楚的逻辑推进。检查方法身份、双索引对象、结构审计、机制结果和干预边界是否按证据关系组织。避免把实验、阶段记录和结果简单堆叠。
6. **改进路径。** 给出具体、可执行且有优先级的修改建议。区分能够实质提升论文层级的改动与仅改善表达的修辞性修改。必要时建议缩小问题、重构框架或推翻当前叙事，不限于局部文字调整。

**中文写作约束**

- 尽量避免“引号加 A-B-C”的表达方式；
- 除非语义要求，减少冒号、破折号和分号；
- 避免超长复合句，一个句子尽量只表达一个主要判断；
- 非数据和公式表述中不使用奇怪符号代替自然语言，例如用文字说明关系，不用 `≠`、`≈` 或 `→`；
- 减少使用“链路”“回路”“闭环”“耦合”等复杂或生涩词语；
- 过渡词和连接词使用基础、常用的表达；
- 总结、框架和修改建议可以使用小标题与分列表，但不应把连续论证切碎。

**评审输出要求**

1. 先给出总评和毕业论文标准下的明确判断，包括当前是否达到送盲审条件；
2. 按六个固定维度逐项审查，每项给出问题、证据位置、影响和修改建议；
3. 单列证据与结论不匹配表，至少包含论文位置、当前结论、现有证据、缺口和允许表述；
4. 单列主要风险，并按阻断毕业、重大修改、一般修改和修辞优化分级；
5. 给出有优先级的修改路线。优先处理研究问题、方法有效性、统计推断和外推边界，再处理文字与版式；
6. 结尾给出 `可送审`、`重大修改后送审` 或 `暂不建议送审` 三种结论之一，并列出改变该结论所需的最小条件。

**评审成功标准**

评审的成功标准是使论文更接近一篇能够经受硕士毕业论文盲审和答辩的学术文本。评审不能以让作者感觉良好为完成标准，也不能用安抚性总结替代问题判断。

**交付物**

- 中英文同步终稿；
- 一段可独立阅读的 `NOV-JOINT-1` 方法定位及双索引定义；
- claim-to-evidence 对照表；
- 数字与引用自动校验报告；
- 一份以最强近邻方法为对象的模拟审稿意见及逐项修订记录；
- 一页“创新性成立条件与失效条件”清单。
- 一份按上述六维度生成的硕士毕业论文终审报告；
- 一份证据与结论不匹配表及分级问题清单。

**完成门**

全文必须同时通过以下问题：

1. 能用一句话回答“既然 Endpoint 等于 G×I，本文新增了什么”，答案固定为目标条件化双索引局部微分对象、合法切面合同、结构审计和机制用途；
2. 能用一张图回答“为什么不是 G×I 加 hooks”，并用数据说明至少一个 attention 和一个 residual 结论依赖 $[e,i]$ 联合信息；
3. 能明确说明与 Conductance、DecompX/LRP、EAP/Vi-CD 和 Unpack 的 estimand、索引和闭合对象差异；
4. 中英文所有方法新颖性句子均回链 `NOV-JOINT-1`，不出现任何单项首次主张；
5. 四检查点复现若进入主稿，所有数字与 sealed artifacts 一致；若未通过审计，则全文保持两 checkpoint 口径；
6. 发布校验器、数字重算、引用检查、双语一致性和 Markdown/LaTeX 检查全部通过。
7. 硕士毕业论文终审报告完整覆盖六个固定维度，给出送审结论和改变该结论所需的最小条件。
8. 所有阻断毕业和重大修改问题均已修复，或已通过缩小主张范围得到明确处理；修辞性优化不得替代方法与证据修复。

## 七、最终判断

如果论文定位为新的 attribution formula，现有证据不支持其新颖性。如果论文仅展示 ViT 上的 Gradient × Input 热图，技术增量也不足。当前可成立的完整定位是：

> **原始 NNS 数学统一 + 目标条件化的 $[\text{consumer-use edge},\text{input coordinate}]$ 双索引局部微分对象 + exact structural differential cuts + fail-closed 审计 + 跨检查点机制研究 + 有限干预边界。**

这是一项测量框架与经验机制研究。其方法创新来自 $\mathcal L_{e,i}(s;x)$ 同时保留输入空间和实际使用边身份，并在合法切面上恢复同一 current-point local differential；其可信性来自结构负控与独立数值 oracle；其科学价值来自该联合对象对 routing/value 和 identity/branch 空间化层深重组的测量。后续工作的重点是证明双索引的必要性、审计四检查点复现并建立逐主张证据索引，而不是继续寻找新的基础公式。

创新性是否成立按以下规则判定：

- 若只有 G×I 恒等式，判为 `IDENTITY_CLARIFICATION`；
- 若定义 $\mathcal L_{e,i}$ 并通过切面闭合，判为 `STRUCTURED_MEASUREMENT_FRAMEWORK`；
- 若审计合同还能检出结构错误，判为 `AUDITED_STRUCTURED_MEASUREMENT`；
- 若至少一个 attention 和一个 residual 主结论必须依赖 $[e,i]$ 联合信息，并在独立检查点保持方向，才允许把 `NOV-JOINT-1` 作为论文的主要方法与经验创新；
- 若联合对象未产生超出总图或 edge scalar 的实质结果，论文应降级为 NNS 数学澄清与高质量系统实现，不用“联合创新”补偿经验不足。

## 八、第一步执行记录：唯一论文定位与主张边界冻结

冻结日期：2026-09-22。适用入口为 `papers/nns_vit_endpoint/publication_r2_20260920/{zh,en}/manuscript.md` 及其 `papers/nns_vit_endpoint/{zh,en}/full_draft.md` 镜像。本节保留第一步完成时的历史快照；第二步后的双索引形式化和第三至第五步修订见本文前七节与第九节，不倒写本节当时的标题、中心句和验收记录。`notes/revision_v3_frozen_claims.md` 保留为上一轮修订快照，不由本次冻结覆盖。

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
| EF-6 | 经验发现 | 给定 reference 的规定区域替换中，displacement-aware Taylor 的排序优于 clean-only Endpoint。 | `m2_inherited.json` 与正文 Tables 8–10 | 指定 reference、区域、预算和替换算子下的任务比较 | Taylor 或 Endpoint 的无条件方法排行榜 |
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

## 九、第二步执行记录：联合能力的新颖性差分审计

审计日期与检索截止日均为 **2026-09-22**。本节比较方法论文中明确界定的对象、等式和验证，不根据相似名称推断能力。表中的“未见”表示本轮核查的正式来源未报告该项，不表示对全部文献作绝对不存在断言。预印本按实际核查版本标记；后续版本或未收录工作可能改变结论。

### 9.1 审计问题、字段和判定规则

本轮要检验的候选联合能力为：

> **在同一观测输入点，对预注册的 ViT 异构内部标量目标保留原始输入坐标，将同一真实局部 Jacobian 在指定实际使用边构成的合法切面上分解为 Q/K/V、routing/value、multi-head 与 residual identity/branch 项，并用正向闭合、遗漏/重复负控和独立数值 oracle 审计后开展层深机制分析。**

统一比较字段严格对应第二步要求：`被解释对象`、`内部目标类型`、`真实 consumer edge`、`合法切面`、`attention routing/value`、`residual identity/branch`、`精确闭合`、`mutation 检出`和`经验用途`。其中：

- `真实 consumer edge`要求 cotangent 绑定到具体张量使用边；只有节点、层、component 或抽象 token relation 不计为完整覆盖。本文现有实现已验证多输入算子的输入分别计数、共享张量按 graph edge 而非张量名区分；更一般的 producer–consumer–slot–invocation registry 属于后续扩展，不写成已完成能力。
- `合法切面`要求从输入到同一目标的每条有效路径恰好穿过一次；稀疏回路、候选边集合、逐层矩阵或单个中间层只有在来源明确给出相同切面条件时才计为覆盖。
- `精确闭合`必须标注闭合的数学对象。输出差、函数值 relevance、前向 token decomposition 和当前点 local differential 不是同一个对象。
- `mutation 检出`专指对错误实现进行预期失败测试；消融、参数随机化、输入扰动和与 patching 的相关性评价不计为 mutation 检出。本文已封存的 28,800 项负控只覆盖遗漏非零 head、遗漏非零 routing 和重复 identity；attention detach 另有一致性检查，不计入 mutation suite。错误 input slot 与错误 fan-out 合并尚无封存 mutation 证据。
- `是`表示正式来源明确报告该能力；`部分`表示覆盖相关结构但对象或合同不同；`未见`表示核查来源没有报告；`不适用`表示该字段不属于方法定义。

### 9.2 正式来源登记

| ID | 正式来源与本轮核查范围 | 直接支持的事实 |
| --- | --- | --- |
| S1 | Sundararajan, Taly, and Yan, *Axiomatic Attribution for Deep Networks*, ICML 2017, [PMLR](https://proceedings.mlr.press/v70/sundararajan17a.html)；Ancona et al., *Towards Better Understanding of Gradient-based Attribution Methods*, ICLR 2018, [OpenReview](https://openreview.net/forum?id=Sy21R9JAW) | Gradient × Input 是当前点输入加权梯度；IG 沿 baseline 到输入的路径积分，并以输出差为 completeness 对象。 |
| S2 | Dhamdhere, Sundararajan, and Yan, *How Important Is a Neuron?*, arXiv:1805.12233, [全文](https://arxiv.org/html/1805.12233) | Conductance 以输入坐标分辨 IG 流经隐藏单元；同一完整隐藏层的 conductance 加和为输出差，并用 ablation 与 feature selection 评价。 |
| S3 | Modarressi et al., *GlobEnc*, NAACL 2022, [ACL Anthology](https://aclanthology.org/2022.naacl-main.19/) | 将 attention、residual、normalization 等 encoder 组件纳入逐层 token attribution，并跨层聚合。 |
| S4 | Ferrando, Gállego, and Costa-jussà, *Measuring the Mixing of Contextual Information in the Transformer*, EMNLP 2022, [ACL Anthology](https://aclanthology.org/2022.emnlp-main.595/) | ALTI 分解整个 attention block 的 token-to-token interaction，再用 rollout 聚合；原文明确未纳入 classifier，因而不生成逐类别解释。 |
| S5 | Modarressi et al., *DecompX*, ACL 2023, [ACL Anthology](https://aclanthology.org/2023.acl-long.149/) | 将输入 token 分量连续传播过全部 encoder 组件、非线性 FFN 与分类头，并把 bias 分配进可重组的 representation/logit decomposition。 |
| S6 | Bach et al., *Layer-Wise Relevance Propagation*, PLOS ONE 2015, [DOI](https://doi.org/10.1371/journal.pone.0130140)；Achtibat et al., *AttnLRP*, ICML 2024, [PMLR](https://proceedings.mlr.press/v235/achtibat24a.html) | LRP 的核心对象是逐层 relevance 守恒；AttnLRP 为 softmax、双线性 attention、normalization 等 Transformer 算子定义 relevance rule，并支持输入与 latent attribution。 |
| S7 | Nishankar et al., *HiLRP*, arXiv:2609.01282v1, [全文](https://arxiv.org/html/2609.01282v1)；Buratti et al., *GradSkip*, arXiv:2607.10365v1, [全文](https://arxiv.org/html/2607.10365v1) | HiLRP 将 ViT attention/降采样归约为四类守恒 primitive，并对多种架构验证 conservation；GradSkip 已显式对 attention 与 residual/skip 路径动态分配 relevance。 |
| S8 | Syed, Rager, and Conmy, *Attribution Patching Outperforms Automated Circuit Discovery*, BlackboxNLP 2024, [ACL Anthology](https://aclanthology.org/2024.blackboxnlp-1.25/)；Kramár et al., *AtP\**, arXiv:2403.00745v1, [全文](https://arxiv.org/html/2403.00745v1) | EAP 用 activation displacement 与目标梯度的一阶内积估计具体计算边的 replacement effect；AtP* 分析 attention Q/K 饱和与直接/间接效应抵消，并给出 edge variant。 |
| S9 | Zhang and Nanda, *Towards Best Practices of Activation Patching*, arXiv:2309.16042v2, [全文](https://arxiv.org/html/2309.16042v2)；Izadi et al., *CAAP*, arXiv:2603.13652v1, [全文](https://arxiv.org/html/2603.13652v1) | Activation patching 直接替换内部状态并测量有限输出变化；CAAP 已把内部 activation insertion 用于 ViT patch attribution。 |
| S10 | Żukowska et al., *Seeing Through Circuits*, arXiv:2604.14477v1, [全文](https://arxiv.org/html/2604.14477v1) | Vi-CD 已在 ViT/OpenCLIP 上把 attention heads 与 MLP blocks 组织为 residual-stream edge graph，以 activation patching 发现和验证稀疏视觉回路。 |
| S11 | Wróbel et al., *DAVE*, arXiv:2602.06613v1, [全文](https://arxiv.org/html/2602.06613v1) | 已有 ViT-specific input-gradient decomposition，将 effective input-output mapping 分为 direct transformation 与 local variation，并以分布聚合形成空间归因。 |
| S12 | Chen, Plaat, and van Stein, *Every Component is a Lookup*, arXiv:2605.23393v2, [全文](https://arxiv.org/html/2605.23393v2) | Unpack 已通过非梯度 backward recursion 给出 component interaction、带 K/Q/V composition label 的命名端到端路径和逐 token attribution。 |
| S13 | 本项目冻结协议与结果：`docs/NNS_ViT_Endpoint_Scanner_EP0协议冻结_20260918.md`、`docs/阶段M_endpointnnsvit.md`、`src/vitnns/endpoint_vit.py`、`src/vitnns/endpoint_e3_closure.py`、E3-R2 `validation_report.json` 与现役双语稿 Sections 3–7 | 定义本文 target registry、实际使用边、合法切面和局部微分分项；封存 76,800 项正向闭合、28,800 项遗漏/重复负控与独立 JVP/有限差分验证。 |

上述来源覆盖用户指定的 Gradient × Input、IG、conductance、GlobEnc、DecompX、ALTI、LRP、edge attribution 与 activation patching。S7、S10–S12 是为避免旧文献集合造成虚假新颖性而追加的截至检索日近邻。

### 9.3 统一字段的 related-work difference matrix

| 方法族 | 被解释对象 | 内部目标类型 | 真实 consumer edge | 合法切面 | attention routing/value | residual identity/branch | 精确闭合 | mutation 检出 | 经验用途 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gradient × Input [S1] | 当前点 $x\odot\nabla_xs(x)$ | 可选可微标量；无统一 target registry | 未见 | 未见 | 未见 | 未见 | 一般无函数值 completeness | 未见 | 输入敏感性与显著图 |
| Integrated Gradients [S1] | baseline 到输入的路径归因 | 原始定义面向输出标量；可由接口选择目标 | 未见 | 未见 | 未见 | 未见 | **是：**各输入归因加和为 $F(x)-F(x')$ | 未见 | 输入归因、调试与规则提取 |
| Conductance [S2] | IG attribution 流经隐藏单元的份额，并可保留输入坐标 | neuron、filter 或隐藏层单元集合 | 未见 | **部分：**完整隐藏层是 layer-wise cut，不是任意 consumer-edge cut | 未见 | 未见 | **是：**同层 conductance 加和为 $F(x)-F(x')$ | 未见 | hidden-unit 分析、ablation、feature selection |
| GlobEnc [S3] | 跨 encoder 的 token attribution | token 与 encoder-block components | 未见 | 未见 | 未显式分开 routing/value differential | **部分：**纳入 residual 与 normalization，但不输出本文的两张局部微分图 | 未报告 class-score 或 local-differential 精确闭合 | 未见 | token attribution 与 erasure/correlation 评价 |
| ALTI [S4] | attention block 的 token mixing 及其 rollout | token-to-token interaction；原文未纳入 classifier | 未见 | 未见 | 未显式分开 routing/value differential | **部分：**attention block 包含 residual 与 LayerNorm | **部分：**局部前向向量可分解；最终量不是类别目标的局部微分闭合 | 未见 | contextual mixing、sufficiency/comprehensiveness |
| DecompX [S5] | 输入 token 对 representation 与预测 logit 的前向分量 | 全 encoder token components、FFN、分类头 | 未见具体 consumer-use 端口 | 未见本文的合法 edge-cut 条件 | attention 权重参与前向分解，但未把 Q/K routing differential 与 V differential 分开 | **部分：**传播 residual/FFN 分量，未把 identity/branch 作为本文式局部微分对 | **是：**在其 bias 规则下重组 hidden representation/logit；对象不是 current-point Jacobian | 未见 | prediction-level token attribution 与 token erasure |
| LRP 家族：LRP、AttnLRP、HiLRP、GradSkip [S6–S7] | 输出 relevance 向输入或 latent variables 的反向分配 | input、tokens、latent neurons；覆盖多类 Transformer primitives | **部分：**按算子连接传播 relevance，未见 producer–consumer–slot–invocation 端口注册 | **部分：**逐层 conservation，不等于本文 true-Jacobian valid cut | **部分：**已有 softmax、双线性 attention 及 Q/K/V primitive；未见本文同目标 routing/value 局部微分对 | **是/部分：**已有 skip-aware relevance 与 attention/residual 动态分配 | **是：**relevance conservation；HiLRP 还报告 machine-precision 验证，但闭合对象不是 local differential | 未见遗漏/重复/错 slot/detach 的 mutation suite | input/latent attribution、localization、perturbation 与跨架构评价 |
| Edge attribution：EAP、AtP*、Vi-CD [S8, S10] | reference-specific component/edge replacement effect 或其一阶估计；稀疏任务回路 | heads、MLPs、Q/K/V nodes 与 residual-stream edges | **是：**计算边或 residual-stream connection 已有明确先例 | 未见对全部输入到目标路径恰好一次的 additive cut 合同 | **部分：**AtP* 有 Q/K saturation fix 和 Q/K/V edge types；未保留逐像素 routing/value 局部图 | **部分：**显式使用 residual-stream edges，但目标是 effect/circuit，不是 identity+branch map closure | 未见 local-differential additive closure；以 patching effect 或 circuit fidelity 为对象 | 未见本文式结构故障注入；以 direct patching、diagnostics 或 circuit recovery 验证 | LLM/ViT circuit discovery、行为恢复与 steering |
| Activation patching、causal tracing、CAAP [S9] | 指定 reference 和替换算子下的有限内部效应 | layer、head、MLP、token/patch activation 等被干预变量 | **部分：**vanilla patching 主要定位变量；edge patching 属上一行 | 不适用 | 可分别干预 attention 变量，但不形成 routing/value 加和分解 | 可分别干预分支，但不形成 identity+branch 局部闭合 | 不适用；单项 finite effect 通常不可加 | 未见 | localization、机制定位、事实关联与 ViT patch attribution |
| DAVE [S11] | ViT 类别输出的分布感知空间归因 | effective input-output mapping | 未见 | 未见 | 未见 | 未见 | 未见本文的结构切面闭合 | 未见 | 稳定、局部等变的 ViT 像素归因 |
| Unpack [S12] | component interaction、命名端到端路径与 token credit | attention、MLP、token 与带 K/Q/V label 的路径 | **部分：**保留 component interaction/path，但未见本文 consumer-use 端口合同 | 未见本文的输入到同目标合法切面条件 | **部分：**已提供 K/Q/V composition labels；不是 current-point true-Jacobian routing/value 图 | **部分：**路径递归经过 residual computation，但未见本文 identity/branch 局部图对 | 方法自身给出 credit decomposition；未见原始输入坐标上的 local-differential cut closure | 未见本文式结构故障注入 | LLM component composition、token attribution 与 IOI 机制分析 |
| NNS–ViT Endpoint Scanner [S13] | 同一当前点局部微分的总图、内部目标图和结构边图 | 预注册 token-channel、aggregate、head、Q/K/V、attention output、residual/MLP branch 与 logit | **是：**cotangent 绑定实际使用边；多输入与共享张量使用按 graph edge 区分 | **是：**每条有效输入到目标路径恰好穿过一次 | **是：**Q+K routing、V value 与 Q/K/V 分项均恢复完整 attention differential | **是：**identity+branch 恢复 residual output differential | **是：**原始 signed input coordinates 上恢复同一 Endpoint local differential | **是：**遗漏非零 head、遗漏非零 routing、重复 identity 三类负控全部被拒绝；另有 JVP 与有限差分 oracle | 两模型 target atlas、层深/路径机制统计与分开的有限替换标定 |

### 9.4 已有能力、降级项与联合差异

| 审计项 | 最近邻工作 | 已被覆盖的能力 | 本文允许保留的差异 | 结论标签 |
| --- | --- | --- | --- | --- |
| 基础公式与路径版本 | Gradient × Input、IG [S1] | 当前点输入加权梯度与 baseline-path 完整归因均是既有方法 | 对原 NNS 规则给出条件等价和坐标/estimand 澄清 | `IDENTITY_CLARIFICATION_NOT_NEW_FORMULA` |
| 输入分辨的内部目标 | Conductance [S2]、AttnLRP [S6]、原 NNS | hidden neuron、latent attribution 和输入坐标图均已有 | 统一注册 ViT 异构目标并绑定同一结构合同；registry 本身是系统设计，不单独主张算法首创 | `SYSTEMATIZATION_INCREMENT` |
| Transformer token 与组件分解 | GlobEnc、ALTI、DecompX [S3–S5] | attention block、residual、normalization、FFN、分类头和 token decomposition 已被覆盖 | 研究对象改为同一当前点 true Jacobian 的 input-coordinate structural terms | `DIFFERENT_ESTIMAND_AND_INDEX_SET` |
| relevance 守恒与 ViT 算子覆盖 | AttnLRP、HiLRP、GradSkip [S6–S7] | attention primitives、latent relevance、skip-aware 分配与 machine-precision conservation 均已有 | 不修改 backward rule；闭合的是指定切面上的真实 local differential，并保留每个分项的输入坐标 | `DIFFERENT_CLOSURE_OBJECT` |
| Q/K/V 与命名路径 | AtP*、Unpack、AttnLRP/HiLRP [S6–S8, S12] | Q/K/V 不是未研究对象；已有 Q/K 修正、attention primitive 和带 K/Q/V label 的路径 | 同一输出 differential 同时给出 Q-induced、K-induced、V-induced 与 routing/value input maps，并逐项闭合 | `JOINT_DIFFERENTIAL_ACCOUNTING_CANDIDATE` |
| residual identity/branch | GlobEnc、ALTI、DecompX、AttnLRP、GradSkip [S3–S7] | residual/skip contribution 与动态 relevance 分配已有 | identity 与 branch 是真实 Jacobian 的 input-coordinate 分项，且两项重构 residual output differential | `JOINT_DIFFERENTIAL_ACCOUNTING_CANDIDATE` |
| 真实计算边与 ViT 回路 | EAP、AtP*、Vi-CD [S8, S10] | edge scoring、edge intervention 和 ViT residual-stream circuit discovery 已有 | 不主张首次研究真实边；差异是对指定实际使用边生成 input-coordinate map，并要求合法完整切面 | `EDGE_NOVELTY_DOWNGRADED` |
| 精确闭合 | IG、Conductance、DecompX、LRP/HiLRP [S1–S2, S5–S7] | 输出差、前向 representation/logit 与 relevance conservation 都有精确加和先例 | 闭合对象是未修改模型在观测点的同一 local differential；不得以“精确闭合”三个字单独主张新颖性 | `CLOSURE_OBJECT_SPECIFIC` |
| 自动审计 | HiLRP [S7]、EAP/AtP* [S8] | machine-precision conservation、与直接 patching 对照及 false-negative diagnostics 已有 | 在正向数值 oracle 外，加入 omission-head、omission-routing 与 duplicate-identity 三类结构负控 | `AUDIT_CONTRACT_INCREMENT` |
| 机制与干预用途 | Raghu et al. 的 ViT receptive-field 结果；activation patching/CAAP、Vi-CD [S9–S10] | 深层空间范围扩大、ViT 内部干预和 edge-circuit 行为验证均已有 | 用同一注册测量族比较 routing/value、identity/branch 的层深重组，并把 local map 与 reference-specific finite effect 分开标定 | `EMPIRICAL_SYNTHESIS_NOT_UNIVERSAL_LAW` |

据此，以下表述被直接降级：

- 不写“首次把 Gradient × Input 用于 ViT 或内部目标”；S1、S2、S6 与现有软件接口已覆盖基础量和内部 selector。
- 不写“首次分解 Transformer attention、Q/K/V 或 residual”；S3–S7、S8 与 S12 已分别覆盖这些对象。
- 不写“首次在 ViT 上研究真实计算边或内部干预”；Vi-CD 与 CAAP 已直接覆盖视觉 Transformer 的 edge circuits 和 activation patching [S9–S10]。
- 不把“守恒”“完整性”或“闭合”本身作为新颖性；IG、Conductance、DecompX、LRP 与 HiLRP 均已有各自的精确加和对象。
- 不把空间扩展写成首次发现；本文只报告在冻结样本和自身指标下的复现及结构分项增量。

### 9.5 研究缺口说明（三段）

现有方法已覆盖本研究的多数单项部件。Gradient × Input 和 IG 定义当前点及路径输入归因，Conductance 将路径归因分配到隐藏单元；GlobEnc、ALTI 与 DecompX 解释 Transformer 的 token mixing、encoder components 和预测分量；AttnLRP、HiLRP 与 GradSkip 提供 attention primitive、latent relevance、skip-aware 分配和守恒；EAP、AtP*、Vi-CD 与 activation patching 已覆盖真实边、有限效应与视觉回路。因此，单凭内部目标、Q/K/V、residual、真实边、闭合或 ViT 应用中的任何一项，均不足以构成本文的新颖性主张。

这些工作闭合和保留的索引不同。Conductance 闭合 reference-path attribution，DecompX 重组 forward representation 或 logit，LRP 家族守恒 relevance，EAP/Vi-CD 评分或筛选 reference-specific edges。本文处理的是未修改模型在观测点的同一 true local differential，并同时保留输入坐标和指定实际使用边身份；合法切面要求所有输入到同一目标的有效路径恰好计数一次。该差异是 estimand、索引集合和验证合同的组合，不是术语差异。

截至 2026-09-22，本轮核查的正式来源中未发现一个方法同时提供：ViT 异构内部目标 registry、原始输入坐标上的实际使用边图、Q/K/V 与 routing/value 以及 identity/branch 的同目标 local-differential cut closure、针对遗漏和重复记账的结构负控，以及使用同一已审计测量族开展层深机制分析。本文可将这组**联合能力**作为有来源边界的新颖性主张；表述必须限定为“在所核查来源中未见完整联合覆盖”，不能写成无范围的世界首次。

### 9.6 可回链的新颖性主张注册表

| ID | 可使用的主张 | 最近邻与差异证据 | 允许位置 | 禁止扩大为 |
| --- | --- | --- | --- | --- |
| NOV-ID-1 | 原始 NNS 的指定传播规则可统一为 same-target Gradient × Input。 | G×I 已有 [S1]；差异是对 Dou et al. 规则的条件等价与边界澄清，见 9.4 第一行。 | 方法身份、附录 | 新归因公式或 Gradient × Input 首创 |
| NOV-SYS-1 | 本文把异构 ViT 内部标量目标组织到同一预注册测量空间。 | Conductance/AttnLRP 已有 hidden/latent targets [S2, S6]；差异为 registry 与共同结构合同，见 9.4 第二行。 | 方法、实现 | 首次内部神经元或 latent attribution |
| NOV-JOINT-1 | 对每个固定目标，本文以 $\mathcal L_{e,i}(s;x)$ 联合保留输入坐标与指定实际使用边身份，并在合法切面上闭合相同的当前点 local differential。 | Conductance 有输入坐标和 layer completeness [S2]；EAP/Vi-CD 有 edges [S8, S10]；DecompX/LRP 有其他闭合对象 [S5–S7]；没有一个核查来源覆盖三者组合，见 9.3 全矩阵。 | 摘要中心句、引言、方法贡献、讨论、结论 | 首次 edge attribution、首次完整性或实现不变的唯一分解 |
| NOV-ATTN-1 | 在 `NOV-JOINT-1` 合同内，Q/K/V 与 routing/value 是同一 attention local differential 的可加 input-coordinate 分项。 | AttnLRP/HiLRP、AtP*、Unpack 已覆盖 Q/K/V primitives、修正或 labels [S6–S8, S12]；差异是 true-Jacobian input maps 与 cut closure，见 9.4 第五行。 | 方法、结构结果 | 首次分析 Q/K/V、routing 等同因果贡献 |
| NOV-RES-1 | 在 `NOV-JOINT-1` 合同内，identity 与 branch input-coordinate maps 恢复同一 residual output differential。 | GlobEnc、ALTI、DecompX、AttnLRP、GradSkip 已覆盖 residual/skip [S3–S7]；差异是 local-differential map pair 与闭合，见 9.4 第六行。 | 方法、结构结果 | 首次考虑 residual 或普遍的 branch dominance |
| NOV-AUD-1 | 结构合同除正向闭合外，还由三类遗漏/重复负控和独立数值 oracle 审计。 | HiLRP 已有 machine-precision conservation，EAP/AtP* 已有 patching 对照和 diagnostics [S7–S8]；本文差异是 omission-head、omission-routing、duplicate-identity 负控，见 9.4 第九行。 | 方法验证、实现贡献 | 负控证明解释具有因果正确性、跨实现唯一性，或覆盖尚未执行的错误类型 |
| NOV-EMP-1 | 同一测量族用于比较层深、attention 和 residual 的局部输入依赖，并与规定 finite replacement 分开标定。 | receptive-field、token mixing、circuit/patching 研究已有 [S3–S12]；差异是统一 audited path-map family 下的联合经验设计。 | 结果、讨论 | 空间扩展首见、全 ViT 规律或普遍干预有效 |

现役主稿中的新颖性相关句子回链如下：

| 位置 | 当前句子功能 | 回链 | 门检结论 |
| --- | --- | --- | --- |
| 双语摘要中心句 | 从 G×I 原语到 structure-aware auditable scanner | `NOV-JOINT-1` | 可保留；未声称基础公式首创 |
| 双语引言首段 | 总梯度不能区分 Q/K/V、heads、identity/branch，本文构建结构框架 | `NOV-JOINT-1`、`NOV-ATTN-1`、`NOV-RES-1` | 可保留；相关工作须紧邻承认 Q/K/V、residual 与 edge 先例 |
| 双语贡献一 | 数学身份与对象定义 | `NOV-ID-1` | 降级为 identity clarification |
| 双语贡献二 | registry、actual edges、valid cuts 与 audit | `NOV-SYS-1`、`NOV-JOINT-1`、`NOV-AUD-1` | 联合主张成立；单项不得写“首次” |
| 双语贡献三 | 两模型结构趋势和 intervention boundary | `NOV-EMP-1` | 经验综合；不得写成普遍规律 |
| 双语讨论与结论 | 统一框架总结 | `NOV-JOINT-1`、`NOV-EMP-1` | 可保留；使用限定范围的正面表述 |

### 9.7 防御性写作审计

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 研究缺口 | “我们不是简单地把 G×I 用于 ViT” | D1 / D14 | Major | 否 | 以否认式句子代替可核查技术差异 | 会要求逐字段先例，而非接受自我辩护 | 删除否认句，直接写联合对象 | 本文同时保留输入坐标与实际使用边身份，并在合法切面上闭合同一 local differential |
| edge 主张 | “不同于所有已有 edge attribution” | D13 | Critical | 是 | 无限定全集断言超出检索能力 | 会以 EAP、Vi-CD 或新工作反例推翻 | 限定检索截止日与核查集合 | 截至 2026-09-22，核查来源未见完整覆盖 `NOV-JOINT-1` 的方法 |
| closure 主张 | “首次实现精确闭合” | D14 | Critical | 是 | 忽略 IG、Conductance、DecompX 与 LRP 的不同闭合对象 | 会认为相关工作比较失实 | 明确 estimand 后再比较 | 本文闭合 current-point local differential；既有工作分别闭合输出差、前向分量或 relevance |
| Q/K/V 与 residual | “首次分解 Q/K/V 和残差” | D14 | Critical | 否 | 单项已有直接先例 | 近邻文献会立即否定主张 | 删除首次表述 | Q/K/V 与 residual 分项属于 `NOV-JOINT-1` 的联合合同 |
| 审计 | “严格测试保证解释正确” | D13 / D10 | Major | 是 | mutation 和 oracle 只检验实现合同，不能证明解释语义或因果有效性 | 会质疑从软件正确性跳到解释有效性 | 分开 correctness 与 utility | mutation controls 检验漏项、重复和端口错误；有限替换另行评价 |

#### Paper-level defensive-writing score

第二步修订后为 **2/10**。研究缺口按“已有覆盖—联合差异—检索边界”组织，没有用重复否认或性能形容词补偿新颖性。

#### Highest-priority removals

1. 删除任何“首次 ViT edge attribution”“首次 Q/K/V 或 residual 分解”“首次精确闭合”表述。
2. 不使用“仅仅”“简单应用”“本质不同”等反驳式措辞；直接陈述 estimand、保留索引和验证合同。
3. 不把 mutation controls 写成科学有效性或因果解释的保证。
4. 不遗漏 2026 年的 Vi-CD、CAAP、DAVE、Unpack、GradSkip 和 HiLRP。

#### Necessary caveats to preserve

- 新颖性结论只覆盖截至 2026-09-22 实际核查的来源。
- 各方法的 completeness、conservation、decomposition 与 closure 必须按各自数学对象解释。
- 本文结构分项依赖指定图、切面和参数化。
- 经验贡献限于两个固定 checkpoint、冻结样本和注册目标。

#### Repeated phrase audit

现役中文与英文主稿逐份扫描，下列 reviewer-facing 构式的计数均为 0：`We emphasize that`、`It should be noted that`、`We do not claim`、`To avoid misunderstanding`、`Taken together`、`需要强调的是`、`应当指出`、`为避免误解`、`我们不声称`。

### 9.8 第二步 prompt-to-artifact 验收表

| 显式要求 | 直接产物 | 覆盖证据 | 状态 |
| --- | --- | --- | --- |
| 系统比较 9 类指定方法 | 9.2、9.3 | 指定方法全部有独立行或明确方法族行；另补 6 个近邻 | PASS |
| 统一检查 9 个字段 | 9.1、9.3 | 矩阵表头逐字覆盖被解释对象、内部目标、consumer edge、合法切面、routing/value、identity/branch、闭合、mutation、经验用途 | PASS |
| 区分单项已有与联合能力 | 9.4 | 10 个审计项逐项给出最近邻、已覆盖能力、剩余差异与结论标签 | PASS |
| 已覆盖部分直接引用并降级 | 9.4；现役双语稿 Section 2 | G×I、Q/K/V、residual、edge、closure、空间扩展五类越界表述均有来源和降级规则 | PASS |
| 只在正式来源未见完整覆盖时保留联合主张 | 9.2、9.5 | 正式来源台账加三段研究缺口；使用“核查来源未见完整联合覆盖”的有界表述 | PASS |
| related-work difference matrix | 9.3 | 12 个方法族的统一字段矩阵 | PASS |
| 不超过三段的研究缺口 | 9.5 | 恰为三段 | PASS |
| 每项方法贡献的最近邻与差异证据 | 9.4、9.6 | 7 个可用主张均绑定 source ID、最近邻、差异与禁用扩大项 | PASS |
| 每个“首次/提出/不同于既有工作”句子可回链 | 9.6；双语现役稿扫描 | 现役稿无 `first/首次/novel/新颖/不同于既有工作` 宣称；贡献句逐条回链 `NOV-*` | PASS |
| 应用场景或命名差异不能过门 | 9.1、9.4 | 判定基于 estimand、索引、切面与验证合同；ViT 应用和“Endpoint”命名均未作为新颖性 | PASS |
| 中英文入口同步且验证覆盖本步骤 | 现役双语主稿、镜像与 `validate_publication_r2.py` | related-work 文本与表格同步；新增 `gxi_step2_*` 覆盖检查 | PASS |
| 停在第二步 | Git diff 与执行记录 | 未重构方法章节、未重绘图、未运行新实验 | PASS |

### 9.9 完成门结论与后续写作约束

第二步完成门通过。审计不支持任何单项“首次”主张，也不支持把本文描述为首个 ViT edge、Q/K/V、residual、conservation 或 activation-patching 方法。当前可用的新颖性主张只有 `NOV-JOINT-1`：对每个固定目标，$\mathcal L_{e,i}(s;x)$ 同时保留输入坐标与指定实际使用边身份；在本轮核查的来源中，未发现一个方法对这一目标条件化双索引对象执行合法切面上的 attention/residual 分项闭合，并以遗漏/重复结构负控和独立数值 oracle 审计后用于层深机制分析。

后续第三至第五步必须引用 9.6 的 `NOV-*` ID 收缩或重写相关句子。若新增文献覆盖 `NOV-JOINT-1` 的全部字段，应修改矩阵和主张，而不是通过改名、增加形容词或只强调视觉应用维持原表述。

验证记录：

- 17 个正式来源 URL 均在 2026-09-22 返回 HTTP 200；方法事实由对应论文正文、摘要或出版页核对。
- `.conda/envs/vitnns/bin/python3.10 vitnns_rebuild/papers/nns_vit_endpoint/scripts/validate_publication_r2.py`：`PASS`，1,734 项检查，0 errors；新增 8 项第二步专项检查，并重算 165 个真实显示面板。
- 第二步独立结构检查：12 个方法族、每行 10 列、三段研究缺口、7 个 `NOV-*` 主张均通过；12 项 prompt-to-artifact 验收均为 `PASS`。
- 双语入口镜像重算：`zh MIRROR_PASS`、`en MIRROR_PASS`；中英文数据表逐行一致。
- `python3 -m py_compile .../validate_publication_r2.py` 与 `git diff --check`：`PASS`。
- 封存证据复核：E3-R2 记录 76,800 项正向闭合和 28,800 项负控；源码确认三类负控为 `omit_largest_nonzero_head_from_A_F`、`omit_nonzero_routing_from_A_F`、`duplicate_identity_in_residual1`。本轮据此删除了未被封存证据支持的 wrong-slot、fan-out 和 detach mutation 表述。
- 当前步骤只更新定位计划、现役双语 Related Work 及镜像、引用表、发布校验器和其报告；未启动第三步方法章节重构，未运行新实验。

## 十、第三步执行记录：方法章节与核心概念图重构

执行日期：2026-09-22。现役入口仍为 `papers/nns_vit_endpoint/publication_r2_20260920/{zh,en}/manuscript.md` 及外层双语镜像。本步没有启动第四步的结果重排、E5 四检查点审计或新实验。

### 10.1 正式方法对象

方法章节先保留总量原语 $x\odot\nabla_xs(x)$，再将实际使用边定义为 producer、consumer、input slot 和 invocation 的四元组 $e=(p_e,c_e,q_e,\nu_e)$。对 target registry 中固定的标量目标 $s$，正式定义

$$
\mathcal L_{e,i}(s;x)
=x_i\left[J_{z_e,x}(x)^{\mathsf T}\bar z_e^{(s)}\right]_i.
$$

自由索引为 consumer-use edge $e$ 和输入坐标 $i$；$s$ 是目标条件。$z_e$、Jacobian 与 use-specific cotangent 在同一观测输入和同一前向状态求值。同一 producer 的多个 consumer、slot 或 invocation 不按 tensor name 合并。正文以 `NOV-ID-1` 标识基础身份澄清，以 `NOV-JOINT-1` 标识双索引对象和合法切面合同。

### 10.2 合法切面、实例与证明

重写后的命题 2 将合法 differential cut 定义为每条当前点有效 input-to-target path 恰好穿过一个已登记 use 的边集合，并证明

$$
\sum_{e\in C}\mathcal L_{e,i}(s;x)
=x_i\frac{\partial s(x)}{\partial x_i}.
$$

证明按唯一穿越的 use edge 将 Jacobian path products 分为互不相交且穷尽的组，再将每组因式分解为上游 $J_{z_e,x}$ 与 use-specific downstream cotangent。正文同时给出 $C_{\mathrm{QKV}}$、$C_{\mathrm{RV}}$、$C_{\mathrm{head}}$ 和 $C_{\mathrm{res}}$，并分别回链 `NOV-ATTN-1` 与 `NOV-RES-1`。命题明确来自链式法则，不将证明本身作为新颖性。

### 10.3 索引必要性与审计合同

方法消融表逐项说明四类退化：删除 $i$ 得到 edge scalar，删除 $e$ 得到总 Gradient × Input，删除 target registry 失去固定目标语义，删除 valid-cut contract 得到不可核验的 component list。统一审计协议表将证据分为公式身份、正向闭合、结构 mutation、独立方向 oracle、CPU 独立重算和未封存 mutation 边界。

结构 mutation 只报告已经封存的三类：9,600 项 omit-head、9,600 项 omit-routing 和 9,600 项 duplicate-identity。384 项 attention-detach 是独立一致性检查，不计作 mutation。Wrong-slot、fan-out merge 与 detach mutation 没有封存结果，不列为已验证检出能力。该边界回链 `NOV-AUD-1`。

### 10.4 核心概念图

Figure 1 按“总 Gradient × Input → $[e,i]$ 双索引展开 → 合法切面重构 → 层深机制问题”组织，并在底部直接显示无 $i$、无 $e$、无 cut contract 和无 registry 的退化。Figure 2 将 consumer-use 四元组、attention/head/residual 三类切面、signed closure 和已封存漏项/重复失败模式放在同一图中。两图均由 `render_publication_r2.py` 生成 PNG、PDF 和 SVG；原分辨率检查发现并修正了 Figure 1 closure 公式与 residual 行的重叠，终版两图无文字或公式遮挡。

### 10.5 第三步 prompt-to-artifact 验收表

| 显式要求 | 直接产物 | 覆盖证据 | 状态 |
| --- | --- | --- | --- |
| G×I 作为总量测量原语 | 双语 Section 3.1；Figure 1 | 命题 1 与总图入口 | PASS |
| $\mathcal L_{e,i}(s;x)$ 正式定义 | 双语 Section 4.1 | 自由索引、目标条件、$z_e$、cotangent、current-point 条件齐全 | PASS |
| producer/consumer/slot/invocation | 双语 Section 4.1；Figure 2A | 四元组与 shared-producer 两次 use | PASS |
| 合法切面命题和证明 | 双语 Section 4.3 | 条件、闭合等式、path partition 证明与 bypass 反例齐全 | PASS |
| 四类 ViT 切面实例 | 双语 Section 4.2；Figure 2B | Q/K/V、routing/value、multi-head、identity/branch | PASS |
| 索引必要性/退化表 | 双语方法消融表；Figure 1 底部 | no $i$、no $e$、no registry、no cut contract | PASS |
| 统一审计协议表 | 双语审计协议表 | 正向闭合、三类 mutation、JVP/VJP、中心有限差分、CPU 独立重算与未封存边界 | PASS |
| 双索引方法总览图 | Figure 1 的 PNG/PDF/SVG | 四阶段方法关系及退化同图呈现 | PASS |
| 结构切面与闭合关系图 | Figure 2 的 PNG/PDF/SVG | use identity、切面、闭合与失败模式同图呈现 | PASS |
| 五项 `NOV-*` 逐句回链 | 双语 Sections 4.1–4.3 | `NOV-ID-1`、`NOV-JOINT-1`、`NOV-ATTN-1`、`NOV-RES-1`、`NOV-AUD-1` 全部出现 | PASS |
| mutation 主张与封存证据一致 | 双语审计表；`qc/gxi_step3_writing_audit.md` | 三类负控各 9,600；wrong-slot、fan-out merge、detach mutation 明确未封存 | PASS |
| 扩展发布校验器与专项复核 | `validate_publication_r2.py`、`validate_gxi_step3.py`、`qc/gxi_step3_completion_audit.md` | 覆盖正文、公式、切面、表格、图件、镜像、哈希与边界 | PASS |
| 停在第三步 | Git diff 与执行记录 | 未改 Results 结构，未审计 E5 四检查点，未运行新实验 | PASS |

### 10.6 完成门结论

第三步完成门通过。读者可在不依赖实验结果的情况下回答四个问题：基础算子是同目标 Gradient × Input；新增对象是目标条件化的 $[e,i]$ 局部微分图组；合法切面重构同一 current-point signed local differential；审计排除已测切面中的漏项、重复和对应数值实现错误。图和正文同时显示无 $i$、无 $e$、无 cut contract 的三种核心退化。

验证记录见 `papers/nns_vit_endpoint/publication_r2_20260920/qc/gxi_step3_completion_audit.md`。第四步需要显式后续请求，本轮未启动。

## 十一、第四步执行记录：结果重排与封存证据绑定

执行日期：2026-09-22。现役入口仍为 `papers/nns_vit_endpoint/publication_r2_20260920/{zh,en}/manuscript.md` 及外层双语镜像。本步只重排和审计已有结果，没有运行新实验，也没有把 PCIA–PEFA 或 CST 纳入 Endpoint 主证据。

### 11.1 结果叙事与联合索引必要性

Results 已按“数学身份与实现正确性 → 总图层深背景 → attention/residual 双索引机制结果 → E5 跨 checkpoint 复现 → 有限替换行为标定”重排。新增 Table 4 明确区分三种表示：

- 总 G×I 图保留输入坐标但消去边身份，可支持总体空间扩展；
- edge scalar 保留路径身份但消去输入坐标，可支持 routing/identity 相对尺度；
- $[e,i]$ 图组同时保留路径和位置，可计算 Full–Value 与 output–identity 的归一化空间差异。

Attention 必要性主结果使用每模型 400 图的 Q/K/V normalized spatial entropy。K 与 V 在两模型中均降低，Q 在 ViT-B/16 为 +0.0268 [0.0217, 0.0314]、在 DeiT-S/16 为 −0.0038 [−0.0080, 0.0005]，相反模型趋势完整保留；每模型 16 个事前保留 class-logit map 的 Full–Value patch TV 提供独立描述性支持。Residual 必要性结果使用每模型、每 residual 类型 400 图的 output–identity patch TV late-minus-early 对比：attention 为 +0.0520 [0.0488, 0.0547] / +0.0282 [0.0257, 0.0300]，MLP 为 −0.0187 [−0.0220, −0.0150] / −0.0462 [−0.0516, −0.0412]。两项均在归一化 patch distribution 上比较，不能由总 absolute mass、activation norm、总图或 edge-only scale 直接恢复。

### 11.2 E5 四检查点证据审计与同步决定

E5 审计结论为 `SYNC_FOUR_CHECKPOINT_E5`：

1. `cohorts.json` 由冻结 EP0 cohort registry 派生，选择不读取 outcome 或选中图像字节；E5 的 400 图与原排除集合及同轮 E1 的 500 图均零重叠；
2. 四个 checkpoint 的文件大小和 SHA-256 均与 `protocol.json` 一致；两个新增权重标为不同训练方案下的参数状态，不称独立种子；
3. `analysis_estimands.json` 在 E5 输出前冻结 H1–H4、缺失值、两侧符号检验和 checkpoint 内四项 Holm 合同；
4. 1,600/1,600 个 image–checkpoint 任务有效，25,600 项结构闭合的最大误差为 $1.812\times10^{-5}$；
5. 第四步专项校验逐字节复核 3,200 个源 JSON/NPZ，重新计算 6,400 个图像级 H1–H4 值，和封存 `e5_image_values.csv` 的最大差为 0；median、10,000 次 bootstrap CI、符号检验与 Holm 决策和封存 `e5_summary.csv` 全部一致。专项校验共 68 项，0 errors。

H1–H3 在四个 checkpoint 中均保持正方向。H4 在四个 checkpoint 中均为负，路径图相对 activation × gradient 的图内 head 排序相关差为 −0.1136 至 −0.1862。现役中英文稿同步报告全部四项，不从外部复现中只保留正结果。

### 11.3 第四步交付物

- 重排后的现役中英文 Results：`publication_r2_20260920/{zh,en}/manuscript.md`，并同步至 `{zh,en}/full_draft.md`；
- 信息差异与必要性表：正文 Table 4；
- attention 与 residual 双索引必要性结果：Sections 6.4–6.5；
- E5 审计及同步决定：`evidence/gxi_step4_e5_audit.json`；
- 逐主张 evidence index：`evidence/gxi_step4_evidence_index.md`；
- 主文与补充材料分配表：`evidence/gxi_step4_figure_allocation.md`；
- 机器重算报告：`evidence/gxi_step4_numeric_recalculation.json`；
- 第四步校验器与复核记录：`scripts/validate_gxi_step4.py`、`qc/gxi_step4_validation.json`、`qc/gxi_step4_writing_audit.md` 与 `qc/gxi_step4_completion_audit.md`。

### 11.4 第四步 prompt-to-artifact 验收表

| 显式要求 | 直接产物 | 覆盖证据 | 状态 |
| --- | --- | --- | --- |
| 先报告等价、算子、闭合与 mutation | 双语 Section 6.1；Table 3 | 5,000 项公式回归、88 项算子方向、76,800 项闭合、28,800 项负控 | PASS |
| 总图/edge scalar/双索引信息差异 | 双语 Section 6.3；Table 4 | retained-index 对照和替代解释审计 | PASS |
| Attention 双索引必要性结果 | 双语 Section 6.4 | 400 图/模型 Q/K/V spatial entropy，另有 16 图/模型 Full–Value patch TV 支持 | PASS |
| Residual 双索引必要性结果 | 双语 Section 6.5 | 400 图/模型/类型 output–identity patch TV 与区间 | PASS |
| 背景规律与核心结果分层 | Sections 6.2–6.5 | 总图空间扩展降为背景，空间化路径重组作为核心 | PASS |
| 替代解释审计 | Section 6.3；evidence index | 总 mass、activation norm、总图和 edge scalar 的信息边界逐项说明 | PASS |
| E5 来源、零重叠与 checkpoint binding | `gxi_step4_e5_audit.json` | 400 图零重叠，四权重 byte/hash 校验通过 | PASS |
| E5 统计合同与 sealed JSON | `analysis_estimands.json`、`e5_receipt.json`、`e5_summary.csv` | 冻结 H1–H4、图像级统计、10,000 bootstrap、符号检验和 Holm | PASS |
| E5 数字重算后同步 | 双语 Section 6.6；Table 5 | 3,200 源文件、6,400 图像级值，最大差 0 | PASS |
| H4 阴性结果完整保留 | 摘要、Section 6.6、结论 | 四 checkpoint 均为相反方向 | PASS |
| Endpoint–Taylor estimand 边界 | 双语 Section 7.2 | clean-only representation 与 displacement-aware prediction 分开 | PASS |
| 逐主张证据索引 | `gxi_step4_evidence_index.md` | 统计单位、效应、区间、checkpoint、target 与 SHA 入口 | PASS |
| 主文/补充分配 | `gxi_step4_figure_allocation.md` | Figures/Tables/补充材料与角色逐项映射 | PASS |
| 数字与发布校验 | `gxi_step4_numeric_recalculation.json`、`gxi_step4_validation.json` | 原始数组复算、双语、镜像、链接、图哈希和格式检查 | PASS |
| 不新增胜负导向实验 | 第四步审计元数据 | `new_experiments=0`；仅重算已有封存数据 | PASS |
| PCIA–PEFA/CST 隔离 | 现役稿与专项校验 | 主稿不含 PCIA、PEFA、aquatic-context 或 NNS-CST | PASS |

### 11.5 完成门结论

第四步完成门通过。至少一项 attention 和一项 residual 结论明确依赖 $[e,i]$ 联合信息；核心结果均给出统计单位、效应量、区间、checkpoint 与适用 target。E5 只有在全部绑定和重算通过后才进入现役稿。正结果、未获支持、H4 相反方向和既有跨模型相反趋势均保留，行为结果继续表述为“广泛但非普遍”。第五步需要显式后续请求，本轮不启动。

## 十二、第五步执行记录：中英文同步重写与对抗式终审

执行日期：2026-09-22。现役入口为 `papers/nns_vit_endpoint/publication_r2_20260920/{zh,en}/manuscript.md` 及外层双语镜像。第五步使用第四步已经通过 source、cohort、checkpoint、统计合同和原始数组重算的证据，没有启动新实验。

### 12.1 中英文同步重写

中英文标题统一改为直接命名双索引对象与结构审计。摘要和贡献二不再用“结构化、可闭合、可审计”等属性词代替技术对象，而是直接定义

$$
\mathcal L_{e,i}(s;x)
=x_i\left[J_{z_e,x}(x)^{\mathsf T}\bar z_e^{(s)}\right]_i.
$$

摘要同时说明输入坐标 $i$、实际 consumer-use edge $e$、目标条件 $s$、合法切面和结构审计。引言、相关工作、讨论和结论保持同一身份：基础测量原语是同目标 G×I；方法增量是目标条件化双索引图组、合法切面合同、结构故障检出和由此支持的 attention/residual 空间机制分析。

### 12.2 最强近邻反例与主张失效条件

双语 Related Work 在 Table 1 后增加最强近邻逐字段反例。Conductance 约束隐藏单元路径归因主张，DecompX/LRP 约束组件传播和 conservation 主张，EAP/Vi-CD 约束真实边与有限干预主张，Unpack 约束命名路径与 K/Q/V 标签主张。`NOV-JOINT-1` 只对应 current-point local differential、原始输入坐标、具体 consumer-use edge、合法切面重构、attention/residual 成对图和结构故障检出的联合字段。一个既有方法若覆盖这些字段，方法新颖性即失效。完整成立与失效条件见 `evidence/gxi_step5_novelty_conditions.md`。

### 12.3 对抗式审稿与修订

模拟审稿依次检查四个强制问题：是否只是 G×I 加 hooks、是否只是已有组件拼装、结构闭合是否误写成因果正确、经验结果是否真的需要双索引。修订结果如下：

1. 标题、摘要和贡献二直接给出双索引对象，Figure 1 给出无 $i$、无 $e$ 和无切面合同的退化；
2. Related Work 用四组最强近邻构成可证伪的逐字段反例，不保留任何单项首次主张；
3. 讨论与结论明确把结构闭合的正确性证据和有限替换的行为证据分开；
4. Attention 的 Q/K/V spatial entropy 与 residual 的 output–identity spatial TV 分别承担联合索引必要性，routing/identity share 和总图 broadening 保持支持或背景身份；
5. H1–H3、H4 四项反向、all-family 未获支持和 Q 的模型依赖方向均完整保留。

逐项意见、初审证据和修订后裁定见 `qc/gxi_step5_adversarial_review.md`。

### 12.4 硕士毕业论文终审

终审按六个固定维度执行：研究问题与科学价值、研究设计与因果逻辑、方法学与统计推断、结果解释与外推边界、学术写作与叙事结构、改进路径。每个维度均给出问题、证据位置、影响和修订判断。证据与结论不匹配表覆盖八类高风险位置，风险清单按阻断毕业、重大修改、一般修改和修辞优化分级。

终审结论为 `可送审`。初审发现的三项重大问题已处理：入口未直接定义双索引对象，最强近邻的失效条件不够显式，结论对 closure 与 finite utility 的分层仍可更清楚。当前无未解决的阻断毕业或重大修改项。作者、单位、通讯作者、repository/archive 和学校模板属于发布流程元数据，不改变学术结论。完整报告见 `qc/gxi_step5_thesis_final_review.md` 与 `qc/gxi_step5_evidence_mismatch_and_risk.md`。

### 12.5 第五步交付物

- 中英文同步终稿及外层镜像：`publication_r2_20260920/{zh,en}/manuscript.md`、`{zh,en}/full_draft.md`；
- 独立方法定位与 claim-to-evidence：`evidence/gxi_step5_claim_to_evidence.md`；
- 创新成立与失效条件：`evidence/gxi_step5_novelty_conditions.md`；
- 模拟对抗审稿与逐项修订：`qc/gxi_step5_adversarial_review.md`；
- 防御性写作终审：`qc/gxi_step5_defensive_writing_audit.md`；
- 六维硕士毕业论文终审：`qc/gxi_step5_thesis_final_review.md`；
- 证据错配与分级风险：`qc/gxi_step5_evidence_mismatch_and_risk.md`；
- 第五步专项校验与完成审计：`scripts/validate_gxi_step5.py`、`qc/gxi_step5_validation.json`、`qc/gxi_step5_completion_audit.md`。

### 12.6 第五步 prompt-to-artifact 验收表

| 显式要求 | 直接产物 | 覆盖证据 | 状态 |
| --- | --- | --- | --- |
| 双语标题、摘要、引言、贡献、相关工作、讨论和结论 | 双语现役稿与镜像 | 标题、公式、`NOV-JOINT-1`、章节和镜像专项检查 | PASS |
| 摘要与贡献二直接定义技术增量 | 双语摘要与贡献二 | $\mathcal L_{e,i}$、$e/i/s$、valid cut 与审计直接出现 | PASS |
| 主张—证据—简洁边界 | 主稿、防御性写作审计 | 评分 1/10；固定 reviewer-facing 构式为 0 | PASS |
| 所有关键数字与范围 | 第四步数字重算、第五步校验 | E5 1,600 任务、6,400 个值重算最大差 0；双语表格一致 | PASS |
| 四项对抗性检查 | 模拟对抗审稿 | 四项均含初审证据、修订动作和修订后裁定 | PASS |
| 最强近邻逐字段反例 | Related Work、创新条件清单 | Conductance、DecompX/LRP、EAP/Vi-CD、Unpack 全覆盖 | PASS |
| 正文、附录、图注、表格、matrix 和 evidence index 同步 | 第五步校验 | 主文逐表、公式、图路径及补充材料标题/数字/图路径一致 | PASS |
| 只有必要时新增实验 | 校验元数据 | `new_experiments=0` | PASS |
| claim-to-evidence 对照表 | `gxi_step5_claim_to_evidence.md` | 位置、证据、统计单位、范围与反证边界齐全 | PASS |
| 数字与引用自动校验 | `gxi_step5_validation.json` | 数字门、引用作者、链接、Markdown/LaTeX 全覆盖 | PASS |
| 创新成立与失效条件 | `gxi_step5_novelty_conditions.md` | 六项成立条件、九项失效条件和禁用扩大表述 | PASS |
| 六维硕士终审及送审结论 | `gxi_step5_thesis_final_review.md` | 六维齐全，结论 `可送审`，最小维持条件四项 | PASS |
| 证据错配表与分级问题 | `gxi_step5_evidence_mismatch_and_risk.md` | 八类错配和四级风险齐全 | PASS |
| 阻断毕业和重大修改清零 | 风险清单、终审 | 无阻断项；三项重大问题均已修复 | PASS |

### 12.7 完成门结论

第五步完成门通过。读者可在摘要中直接回答 G×I 等价下的新增对象；Figure 1 与数据分别回答为何不是 G×I 加 hooks；最强近邻的 estimand、索引和闭合对象差异已明确；所有方法贡献句统一锚定 `NOV-JOINT-1`；E5 只在第四步数组级重算通过后进入主稿；双语、引用、公式、链接和 Markdown 校验通过；六维终审给出 `可送审` 结论与最小维持条件。

验证记录：`validate_gxi_step3.py` 30 项通过；`validate_gxi_step4.py` 68 项通过；`validate_publication_r2.py` 1,733 项通过并重算 162 个真实显示面板；`validate_gxi_step5.py` 411 项通过；`tests/test_revision_v5.py` 6 项通过；四个校验器 `py_compile` 与 `git diff --check` 均通过。
