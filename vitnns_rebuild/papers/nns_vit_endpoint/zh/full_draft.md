# 从 NNS 到视觉 Transformer：内部目标的可审计 Jacobian 扫描与结构分解

- 作者：[AUTHOR REQUIRED]
- 单位：[AFFILIATION REQUIRED]
- 通讯作者：[CORRESPONDING AUTHOR REQUIRED]

## 摘要

神经网络扫描器（Neural Network Scanner，NNS）通过逐层传播局部尺度，为网络内部目标生成保留输入位置的学习图像。当各层使用实际前向状态下的精确局部导数，且不加入归一化、截断或相关性重分配时，标量 NNS 学习图像等于 $x\odot\nabla_xs(x)$，即同目标的 Gradient × Input。以 Gradient × Input 为局部测量原语，本文构建面向 ViT 异构内部目标的结构化、可闭合、可审计扫描框架，并用它系统研究局部输入依赖如何随层深、attention 路径和 residual 分支重新组织。该框架注册异构内部目标，在真实计算边上定义合法 differential cuts，将 attention differential 分为 Q-induced、K-induced 与 V-induced 路径，将 residual differential 分为 identity 与 branch 路径，并检验各切面的加和闭合。分项相对于指定计算图、切面和参数化定义。ViT-B/16 与 DeiT-S/16 两个原始固定 checkpoint 上的实验覆盖数值正确性、路径层深统计、训练参数状态与随机参数状态的差异，以及规定有限替换任务中的排序能力。冻结抽样目标的局部输入依赖随深度扩展到更广、更远的 patch；深层 attention 的 routing 相对范数和 Value-only 近似误差提高，attention 与 MLP residual 呈现不同的层深变化。在与旧队列零重叠的 400 张图像和四个公开 checkpoint 上，神经元熵、routing 尺度与 residual 轨迹的冻结方向均复现，而路径图相对 activation × gradient 的 head 排序增量在四个 checkpoint 上均为负。局部路径图与有限成分效应存在广泛但非普遍的关联；给定 reference 的区域替换中，显式使用实际位移的 Taylor 排序优于 clean-only Endpoint。

关键词：视觉 Transformer；Neural Network Scanner；学习图像；局部 Jacobian；结构路径分解

## 1. 引言

视觉 Transformer（ViT）将图像划分为 patch tokens，随后通过自注意力和前馈网络逐层建立表示（Dosovitskiy et al., 2021）。这一结构能够处理不同位置间的关系，也使内部计算的解释更加复杂：token-channel 单元描述特定位置的响应，attention head 在多个位置间传递信息，残差连接又将输入与分支输出叠加。要理解这些对象之间的联系，需要一种能够保留输入空间位置、并适用于不同内部对象的表示。

现有可视化通常展示特征图、注意力权重或类别相关区域。这些视图各有用途，但其空间分辨率、目标和传播规则不同。前向特征图说明一个通道在 token 网格上的响应，attention 权重说明 token 间的分配关系；它们与某个内部标量关于原始图像的依赖并不是同一个对象。将内部目标映射回统一输入坐标，可以为跨模块比较提供共同的观察基础。

Neural Network Scanner（NNS）为神经元关联一张保留输入位置的 learning image（Dou et al., 2023）。原方法的逐层尺度传播在使用实际局部导数时可以写成 Jacobian 链式法则；对标量目标，其终点学习图像就是 Input × Gradient。这个恒等式确定了本文的局部测量原语。普通总梯度不能区分局部响应经过 Q、K、V、不同 heads，还是 residual 的 identity 与 branch。本文据此构建面向 ViT 异构内部目标的结构化、可闭合、可审计扫描框架，将完整局部输入依赖分解到真实计算边，并研究这些依赖如何随层深和结构路径重新组织。

![Figure 1](../publication_r2_20260920/figures/png/fig_01_concept_overview.png)

**Figure 1.** 对每个注册目标，已有的总 Gradient × Input 被展开为同时保留 consumer-use edge 与输入坐标的局部微分图组。合法 attention、head 或 residual 切面在 signed 输入坐标上重构同一总量；路径身份与空间位置共同支持层深机制比较。下方列出删除输入索引、边索引、切面合同或目标 registry 后的退化。

本文的总体问题是：如何在 NNS 框架下，将 ViT 内部目标对输入的局部依赖表示为可计算、可分解和可验证的结构路径图，并评价其层深规律及在规定有限替换任务中的排序能力？具体研究四个问题：

1. **RQ1：NNS 学习图像的数学本质是什么？** 明确逐层尺度传播、Jacobian 链式法则与 Input × Gradient 的关系，以及非齐次项、固定 token 和输入坐标原点带来的限制。
2. **RQ2：NNS 如何推广到 ViT 的结构化计算图？** 定义内部标量目标、真实计算边、合法计算切面及 attention、multi-head 和 residual 的可加局部微分路径。
3. **RQ3：结构路径如何随深度变化？** 在两个固定预训练 checkpoint 的冻结抽样目标上分析空间扩展、routing/value 与 identity/branch 的层深规律，并比较训练参数状态和随机参数状态以及目标与抽样控制。
4. **RQ4：局部路径图能否在规定的有限替换任务中排序成分和空间区域？** 对固定模型、样本、层、head、reference、替换算子和区域预算，比较 clean-only 局部路径图与 reference-aware 位移估计的成分排序和区域选择。

本文作出三项贡献。第一，**数学身份与对象定义**。我们给出原始 NNS learning-image propagation 等于同目标 Gradient × Input 的条件，并区分 Endpoint、Path IG、模型常数和 reference-specific finite replacement；Table 3 的 5,000 项公式回归及 88 个算子方向检查验证了该对象定义。第二，**结构化扫描与审计框架**。我们建立面向 ViT token-channel、通道聚合、attention head、residual edge 与类别目标的 registry，在真实计算边上定义合法 differential cuts，并对 Q/K/V、multi-head 和 identity/branch 局部微分进行可加记账；Table 2 的 457/385 个注册目标以及 76,800 项正向闭合、28,800 项遗漏/重复负控和 6,144 项解析 JVP 构成直接证据。第三，**机制观察与解释边界**。我们用两个原始固定 checkpoint 的冻结样本测量层深、attention 和 residual 的局部输入依赖规律，在独立 400 图和四个公开 checkpoint 上检验冻结方向，并以规定有限替换实验区分观测点局部表示与 reference-specific intervention estimator；Figures 6–11 与 Tables 4–10 给出图像级结果。

下文先界定相关方法与本文定位，再给出 NNS 的一般 Jacobian 表示和 ViT 结构分解。实验按等价性与闭合、层深规律、结构路径和有限干预组织，最后讨论证据范围。

## 2. 相关工作

### 2.1 NNS 与学习图像

神经元可视化通过输入模式或实际样本分析内部响应。NNS 将单个神经元的计算组织为输入坐标 learning image，并用 activation-weighted aggregation 表示卷积通道（Dou et al., 2023）。这种组织方式使不同原生空间结构的内部对象能够映射到同一图像坐标。本文保留“学习图像”这一历史术语；在数学推导中称其为“局部输入依赖图”，以强调它描述当前输入点的一阶响应。

### 2.2 局部梯度、路径积分、FullGrad 与隐藏单元归因

Gradient × Input 是观测点上的输入加权梯度（Shrikumar et al., 2017）；Integrated Gradients 沿 reference 到输入的路径平均梯度并满足相应完整性公理（Sundararajan et al., 2017），conductance 将该路径归因扩展到隐藏单元（Dhamdhere et al., 2018）。FullGrad 将输入梯度项与各层 bias-gradient 项共同组成完整表示，再聚合为显著图（Srinivas and Fleuret, 2019）。本文的 Endpoint 使用当前点的真实 Jacobian，不进行路径积分，也不把 bias 项加入输入图。因此 conductance 的路径完整性、FullGrad 的函数值完整表示和本文合法切面的局部微分闭合是三个不同对象。

DAVE 将 ViT 输入梯度分为直接 effective input-output transformation 与 local-variation 项，再跨输入分布聚合保留项（Wróbel et al., 2026）。它是 ViT 梯度结构分解和空间归因的直接先例，但不保留按内部 consumer use 索引的一组输入坐标图。

### 2.3 Transformer token contribution 与 relevance 传播

GlobEnc 把 encoder block 的 attention、residual 与 normalization 纳入逐层 token attribution，并跨层聚合（Modarressi et al., 2022）。ALTI 分解包含 residual 与 LayerNorm 的整个 attention block 的 token interaction，再用 rollout 聚合；其发表版本未纳入 classifier（Ferrando et al., 2022）。DecompX 传播 token representation 分量，覆盖非线性前馈层和分类头，并按其 bias 分配规则重组 representation 与 logit（Modarressi et al., 2023）。

AttnLRP 为 softmax、双线性 attention、normalization 与 latent attribution 定义 relevance rule（Achtibat et al., 2024）。LibraGrad 在 FullGrad-completeness 目标下剪枝和缩放反向路径，用于 ViT attribution（Mehri et al., 2025）。HiLRP 将多类 ViT attention 和分辨率变化模块归约为四类守恒 primitive，并以机器精度验证 conservation；GradSkip 则显式在 attention 与 residual 路径之间动态分配 relevance（Nishankar et al., 2026; Buratti et al., 2026）。Unpack 不依赖梯度即可生成 component interaction、带 K/Q/V composition label 的命名端到端路径和 token attribution（Chen et al., 2026）。这些工作已经为 Transformer 组件传播、Q/K/V 分析、skip-aware 分配和精确守恒提供直接先例。本文比较的差异因此是保留的索引与闭合对象：不修改模型的真实局部导数、具体 consumer use 的输入坐标分项，以及这些分项在合法切面上对同一局部微分的加和重构。

### 2.4 Activation patching 与 edge attribution

Activation patching 以 clean/corrupted reference 替换真实内部变量并测量有限输出变化；结果依赖 corruption、替换位置和评价指标（Zhang and Nanda, 2023）。Attribution patching 以 activation displacement 与局部梯度的内积近似单个组件的 patching effect；Edge attribution patching 将这一近似用于计算图边（Syed et al., 2024）。AtP* 进一步处理 attention saturation 与抵消造成的近似误差（Kramár et al., 2024）。这些方法直接评价或近似 reference-specified finite effect。本文分别报告 clean-only 路径图、reference-aware Taylor 量和真实替换效应，不用结构闭合代替有限干预效用。

边分析和内部干预也已经进入视觉领域。Vi-CD 通过 activation patching 在 ViT 与 OpenCLIP 中发现并验证稀疏 residual-stream edge circuits（Żukowska et al., 2026）；CAAP 将源图像 activation 插入中性上下文以构造 ViT patch attribution（Izadi et al., 2026）。因此，真实边和 ViT activation patching 都是本文的先例，不是新颖性主张。

### 2.5 视觉归因与本文定位

Grad-CAM 通过梯度加权特征图获得类别相关定位（Selvaraju et al., 2017），attention rollout 组合跨层 attention flow（Abnar and Zuidema, 2020），Chefer et al.（2021）则为 Transformer 设计 relevance 传播。它们与 Gradient × Input、token decomposition、LRP 和有限替换面向不同解释对象。Table 1 用相同的九个字段比较最近邻方法族。“未见”只表示所引来源未报告该能力，不构成对全部文献的绝对不存在断言。本轮文献截止日为 2026 年 9 月 22 日。

**Table 1. 新颖性差分矩阵：精确闭合均指对应单元格标明的方法自身对象**

| Method family | Explained object | Internal targets | Consumer edge | Valid cut | Routing/value | Identity/branch | Exact closure | Mutation detection | Empirical use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gradient × Input | Current-point input-weighted gradient | Any differentiable scalar; no registry | Not found | Not found | Not found | Not found | Generally no function-value completeness | Not found | Input saliency |
| Integrated Gradients | Baseline-to-input path attribution | Output scalar; selectable targets in software | Not found | Not found | Not found | Not found | Output difference | Not found | Input attribution |
| Conductance | Path attribution through hidden units, optionally input-resolved | Neurons or unit sets | Not found | Whole hidden layer only | Not found | Not found | Output difference across a layer | Not found | Unit analysis and ablation |
| GlobEnc | Cross-layer token attribution | Tokens and encoder components | Not found | Not found | Not separated | Residual included, not paired local maps | Not reported for class score/local differential | Not found | Token attribution |
| ALTI | Attention-block token mixing and rollout | Token interactions; classifier excluded | Not found | Not found | Not separated | Residual and LayerNorm included | Local forward decomposition, not target local differential | Not found | Context mixing and token erasure |
| DecompX | Token components of representations and logits | Encoder, FFN, classifier | Not found at consumer-use granularity | Not found | Not separated as local differentials | Residual components, not paired local maps | Representation/logit under its bias rule | Not found | Prediction token attribution |
| LRP family | Conserved relevance to input or latent variables | Inputs, tokens, latent neurons | Operator links, not registered consumer uses | Layerwise conservation, not a true-Jacobian cut | Attention primitives; no paired routing/value local maps | Skip-aware relevance exists | Relevance conservation | No structural mutant suite found | Attribution and localization |
| EAP / AtP* / Vi-CD | Reference-specific edge effects, approximations, or sparse circuits | Heads, MLPs, Q/K/V nodes and residual edges | Yes | No complete additive cut found | Q/K fixes and Q/K/V edge types, not input maps | Residual-stream edges, not paired local maps | Patching effect or circuit fidelity, not local differential | Patching checks/diagnostics, not structural mutants | Circuit discovery and steering |
| Activation patching / CAAP | Finite internal replacement effect | Layers, components, tokens or patch activations | Node-level; edge variants above | Not applicable | Intervenable, not additively decomposed | Intervenable, not additively decomposed | Not additive in general | Not found | Localization and mechanism tests |
| DAVE | Distribution-aware ViT spatial attribution | Effective input-output mapping | Not found | Not found | Not found | Not found | No structural-cut closure found | Not found | Stable ViT pixel attribution |
| Unpack | Component interactions, named paths, token credit | Attention, MLP, tokens, K/Q/V-labelled paths | Component/path identity | No input-to-target valid-cut contract found | K/Q/V labels, not true-Jacobian routing/value maps | Paths traverse residual computation | Method-specific credit decomposition | No structural mutant suite found | LLM composition and IOI analysis |
| NNS–ViT (this work) | Same current-point local differential indexed by input and edge | Registered ViT scalar targets and structural terms | Cotangent at actual use edges; multi-input/shared-tensor uses distinguished | Each live input-target path crosses once | Q+K routing and V value; Q/K/V maps | Identity plus branch maps | Same signed local differential | Omit-head, omit-routing and duplicate-identity controls | Two-model depth/path analysis and separate replacement calibration |

最后一行的每个单项均有先例：内部归因、Transformer 组件分解、Q/K/V 分析、residual 分配、计算边、守恒和 ViT 干预都不是空白。截至检索日，所核查来源中未发现一个方法同时保留输入坐标与具体 consumer-use 身份，在合法切面上重构同一当前点局部微分，输出 attention 与 residual 的成对结构图，并使用结构 mutation 测试。本文的贡献主张是这一有边界的组合及其层深机制分析，不延伸到单项能力或新的基础归因公式。

## 3. NNS 的 Jacobian 解释

### 3.1 NNS 的一般计算图表示

设网络为有向无环计算图 $G=(V,E)$，输入为 $x$，目标为标量 $s(x)$。一个神经元输出标量值，学习图像则保留该标量所依赖的输入坐标。Figure 1 先区分总 Gradient × Input 与本文的双索引展开；坐标初始化和原始 NNS 逐层传播示意保留在补充图件。本文将最终标量图记为 $E_s(x)$，并称为 NNS 局部输入图。

设图像输入为列向量 $x\in\mathbb R^d$，标量梯度 $\nabla_xs\in\mathbb R^d$ 也按列向量记。对 $u(x)\in\mathbb R^m$，learning-image 矩阵为 $\mathcal L(u;x)\in\mathbb R^{m\times d}$。初始化与传播规则为

$$
\mathcal L(x;x)=\operatorname{diag}(x),\qquad
\mathcal L(v;x)=\sum_aJ_{f,u_a}(x)\mathcal L(u_a;x),\quad v=f(u_1,\ldots,u_r).
$$

**命题 1（标量目标等价性）。** 若所有局部 Jacobian 在同一当前前向状态求值，且传播中不引入归一化、截断或相关性重分配，则

$$
\mathcal L(u;x)=J_{u,x}(x)\operatorname{diag}(x),\qquad
\mathcal L(s;x)=\nabla_xs(x)^\top\operatorname{diag}(x),\qquad
E_s(x)=\mathcal L(s;x)^\top=\operatorname{diag}(x)\nabla_xs(x).
$$

**证明。** 输入处 $J_{x,x}=I$。按计算图拓扑顺序递推，将各前驱的表达式代入传播规则，再使用多输入链式法则，即得到每个节点的结论。对标量节点转置得到列向量 Endpoint。在线性、卷积和当前 ReLU gate 下，该规则恢复原始 NNS 系数；MaxPool 使用当前前向 winner。ReLU 零点和 MaxPool 并列遵循框架的固定导数选择，光滑区域内使用通常导数。

Endpoint 表示当前点上目标关于各输入坐标缩放方向的局部一阶依赖：对 $x_i\mapsto(1+\epsilon)x_i$，其一阶响应为 $\epsilon E_{s,i}$。一般有

$$
\sum_iE_{s,i}=x^\top\nabla_xs(x)\ne s(x).
$$

一阶齐次函数在可微点可由 Euler 定理得到等号；含 bias、固定 token、位置编码、LayerNorm、softmax 和 GELU 的一般 ViT 不满足这一条件。Endpoint 重构局部输入 Jacobian 分解，不能作为目标全部来源或函数值的完整分解。CLS token、位置编码和 patch/LN/QKV/MLP/classifier 的偏置没有独立图像坐标项，但会改变当前函数和 Jacobian。

若 $x=(r-\mu)/\sigma$，则

$$
x_i\frac{\partial s}{\partial x_i}=(r_i-\mu_i)\frac{\partial s}{\partial r_i}.
$$

因此 Endpoint 不需要外部选择的 intervention reference，但依赖模型输入坐标及 normalization origin。

## 4. ViT 内部目标的双索引局部微分

### 4.1 Target registry 与双索引对象

Target registry 为每个键固定标量读出 $s$、图中位置和聚合规则。注册对象包括 token-channel、通道聚合、attention head 与 Q/K/V 项、routing/value 项、residual identity/branch、MLP branch 和类别 logit。Registry 规定可比较的目标语义，但不表示穷举全部神经元。对于通道 $c$ 的 patch activations，聚合目标可定义为

$$
s_{\mathrm{energy}}=\frac12\sum_{t=1}^{196}h_{t,c}^2,\qquad
\nabla_xs_{\mathrm{energy}}=\sum_th_{t,c}\nabla_xh_{t,c},\qquad
E_{\mathrm{energy}}=\sum_th_{t,c}E_{h_{t,c}}.
$$

这给出原始 NNS activation-weighted filter aggregation 的 Jacobian 形式；energy 是聚合标量而非单个神经元。目标还包括 patch sum、head 内部坐标、final CLS channel、ground-truth logit 和 frozen clean-top1 logit。跨模型同编号 channel 不视为同语义单元。

令实际使用边

$$
e=(p_e,c_e,q_e,\nu_e),
$$

由 producer $p_e$、consumer $c_e$、consumer input slot $q_e$ 与 invocation $\nu_e$ 联合确定。$z_e(x)$ 是 producer 输出进入该 consumer use 时的张量。$\bar z_e^{(s)}$ 是在该 input slot 处、尚未与同一 producer 的其他 fan-out uses 合并的 target cotangent。本文定义

$$
\boxed{
\mathcal L_{e,i}(s;x)
=x_i\left[J_{z_e,x}(x)^{\mathsf T}\bar z_e^{(s)}\right]_i.
}
$$

$i$ 与 $e$ 是自由索引，$s$ 是 registry 选定的目标条件。Jacobian、cotangent 和 $z_e$ 均在同一观测输入 $x$ 和同一前向状态求值。固定 $e$ 得到一张保留输入位置的图；固定 $i$ 可比较同一输入坐标经不同实际使用边的局部项；对 $e$ 求和则回到总量。该目标条件化的 $[e,i]$ 表示是 `NOV-JOINT-1` 的方法对象。基础 $x\odot\nabla_xs(x)$ 仍按 `NOV-ID-1` 作为已有测量原语和 NNS 身份澄清。

同一 producer tensor 可同时进入多个 consumer，也可进入同一 consumer 的不同 slot，或在重复调用中被多次使用。此时 $J_{z_e,x}$ 可以共享上游数值，但 $\bar z_e^{(s)}$ 对应不同下游 use。按 tensor name 合并这些 cotangent 会提前丢失路径身份，因此 edge identity 必须包含 consumer、slot 与 invocation。Figure 2 将这一合同与 ViT 切面实例并列。

![Figure 2](../publication_r2_20260920/figures/png/fig_02_neuron_learning_image.png)

**Figure 2.** 左栏用 producer、consumer、input slot 与 invocation 区分同一张量的两次实际使用。中栏给出 attention Q/K/V、multi-head 与 residual identity/branch 的合法切面；每条有效输入到目标路径在一个切面上恰好穿过一次。右栏给出双索引定义、signed closure 以及已封存的漏项和重复记账失败模式。

### 4.2 Attention、multi-head 与 residual 切面

Figure 3 将内容传递与权重变化分开，Figure 4 则将 residual 的 identity 与 branch 并列。

![Figure 3](../publication_r2_20260920/figures/png/fig_06_attention_differential.png)

**Figure 3.** Q/K 的变化通过 attention weights 改变 routing；V 的变化通过当前权重传播。两条微分路径相加重构完整 attention 微分，head 下游 cotangent 包含 output projection。

![Figure 4](../publication_r2_20260920/figures/png/fig_07_residual_principle.png)

**Figure 4.** 上方显示 residual 的 identity 与 branch 两条真实计算边；下方用同色虚线连接到同一输入、同一类别目标的路径 learning images。完整 signed maps 相加得到 residual output，蓝底正部用于阅读空间结构。

对每个 head，$Q=XW_Q+b_Q$、$K=XW_K+b_K$、$V=XW_V+b_V$，且

$$
A=\operatorname{softmax}(QK^\top/\sqrt{d_h}),\quad O_h=AV,\quad
\mathrm dO_h=(\mathrm dA)V+A\mathrm dV.
$$

其中 $\mathrm dA=J_{\mathrm{softmax}}(S)\mathrm dS$ 且 $\mathrm dS=(\mathrm dQK^\top+Q\mathrm dK^\top)/\sqrt{d_h}$。第一项因此分为 Q-induced 与 K-induced routing differential，第二项为 V-induced value differential：

$$
E^{\mathrm{attn}}=E^Q+E^K+E^V=E^{\mathrm{routing}}+E^{\mathrm{value}}.
$$

这些量是 $\mathcal L_{e,i}$ 在 attention 切面上的实例，而不是 Q/K/V 激活或前向输出值的组成比例。$C_{\mathrm{QKV}}=\{e_Q,e_K,e_V\}$ 与 $C_{\mathrm{RV}}=\{e_{\mathrm{routing}},e_{\mathrm{value}}\}$ 是同一完整 attention differential 的两种分组。Head 边位于拼接和 output projection 之前：$O=\operatorname{concat}_h(O_h)W_O+b_O$；$C_{\mathrm{head}}=\{e_{h_1},\ldots,e_{h_H}\}$ 中每个 use-specific cotangent 均包含 $W_O$ 及目标之后的全部导数。偏置没有直接输入微分项；Q/K/V 路径共享上游输入，允许在输入空间重叠。路径 norm 比例描述局部尺度，不能解释为互斥因果百分比。这些联合索引切面对应 `NOV-ATTN-1`；Q/K/V 分析本身已有直接先例。

对 residual $y=u+f(u)$，令 $g_y=\nabla_yF$。正确的两项为

$$
E^{\mathrm{id}}=\operatorname{diag}(x)J_{u,x}^\top g_y,\qquad
E^{\mathrm{branch}}=\operatorname{diag}(x)J_{u,x}^\top J_{f,u}^\top g_y.
$$

Identity 使用 residual 输出处的 $g_y$；直接使用已包含 branch 的 $\nabla_uF$ 会重复记账。$C_{\mathrm{res}}=\{e_{\mathrm{id}},e_{\mathrm{branch}}\}$ 相加恢复 residual output differential，但不产生 $F=F_{\mathrm{id}}+F_{\mathrm{branch}}$ 的函数值分解。该 input-coordinate map pair 对应 `NOV-RES-1`；residual 分解本身已有直接先例。



### 4.3 完整计算切面上的组合关系

对固定目标 $s$，若实际使用边集合 $C$ 使每条具有非零当前点局部导数的 input-to-target path 恰好穿过一个已登记 use，则称 $C$ 为合法 differential cut。

**命题 2（合法 differential cut 的闭合）。** 设计算图在 $x$ 处按固定导数约定可微，各成分使用同一前向状态，$\bar z_e^{(s)}$ 位于对应的 consumer input slot，并且 bypass 和 fan-out uses 不漏不重，则对每个输入坐标 $i$，

$$
\boxed{
\sum_{e\in C}\mathcal L_{e,i}(s;x)
=x_i\frac{\partial s(x)}{\partial x_i}.
}
$$

多输入算子的输入边分别计入；同一张量被多次使用时按四元组 $e$ 而不是 tensor name 区分。

**证明。** 将 $\partial s/\partial x_i$ 展开为所有当前点有效输入到目标路径的 Jacobian 乘积。合法切面按路径穿过的唯一 use edge 将这些项分成互不相交且穷尽的组。每组共同的上游因子为 $\partial z_e/\partial x_i$，其余下游乘积之和正是 use-specific cotangent $\bar z_e^{(s)}$，因此该组等于 $[J_{z_e,x}^{\mathsf T}\bar z_e^{(s)}]_i$。对 $e\in C$ 求和并乘 $x_i$ 即得结论。只切 $f(u)$ 会遗漏 $y=u+f(u)$ 的 identity bypass；以完整 $\nabla_uF$ 作为 identity 项后再加入 branch 项则会重复分支。分支内部 cut 只重构该分支，完整 residual 或目标还需纳入其他有效路径。命题来自链式法则；`NOV-JOINT-1` 的增量是将它落实为目标条件化双索引对象和可审计 cut contract，而不是把证明本身写成创新。

**方法消融表：索引必要性与对象退化**

| Retained structure | Observable | Lost when removed | Degeneration |
| --- | --- | --- | --- |
| $[e,i]$ with registered $s$ and valid $C$ | Path-specific spatial maps that reconstruct one total local differential | Nothing in the declared object | Auditable dual-indexed map family |
| no $i$ | Per-edge aggregate magnitude or score | Spatial allocation and signed spatial cancellation | Edge scalar |
| no $e$ | Total input-coordinate dependence | Routing/value, head, and identity/branch identity | Total Gradient × Input |
| no registered $s$ | Maps from ad hoc scalar readouts | Comparable target semantics and reproducible atlas coverage | Unregistered target collection |
| no valid-cut contract | Named component maps | Detection of omitted or duplicated live paths | Unverified component list |

该表给出结构消融而非经验性能消融。删除 $i$ 后，空间扩展和同一路径的空间重分配不可见；删除 $e$ 后，不同路径在总图中相加；删除 $s$ 的 registry 后，跨目标比较失去固定语义；删除合法切面后，图组即使外观合理也不能证明不漏不重。

**审计协议表：验证层、直接证据与封存边界**

| Audit layer | Question | Executed evidence | What it establishes |
| --- | --- | --- | --- |
| Formula identity | Does original compatible NNS equal same-target G×I? | 5,000 regressions; 88 float64 operator directions | Identity under the declared local-derivative rules |
| Positive closure | Do registered valid cuts reconstruct the same signed differential? | 76,800 closure checks | Accounting on the tested cuts and frozen graphs |
| Structural mutations | Can the checker reject prescribed omission or duplication? | 9,600 omit-head; 9,600 omit-routing; 9,600 duplicate-identity controls | Detection of these three error classes |
| Independent direction oracles | Do component sums agree with local directional derivatives? | 6,144 analytic JVP checks; JVP/VJP duality and nine-step central differences over 88 operator directions; 384 attention-detach consistency checks | Local numerical correctness under the tested directions and operators |
| Independent recomputation | Can a separate CPU validator reproduce stored summaries from raw arrays? | 384 raw blocks and 230,400 E3 summary fields | Agreement independent of the producer summaries |
| Unsealed mutations | Can it detect wrong-slot, merged fan-out, or detach mutations? | No sealed mutation result | Not an established capability |

该协议对应 `NOV-AUD-1`。正向闭合检查可加记账；三类已封存 mutation 检查具体漏项或重复；JVP/VJP、中心有限差分和独立 CPU 重算提供不同数值 oracle。Attention detach 的 384 项是独立一致性检查，不计作第四类 mutation。Wrong-slot、fan-out merge 与 detach mutation 尚无封存结果，因此不写成已验证检出能力。

### 4.4 计算图依赖与分解非唯一性

命题 2 的闭合是相对于指定计算图、切面和内部坐标成立的。端到端函数及总 Endpoint 可以在等价重参数化下保持不变，但逐坐标或逐路径项不必保持不变。考虑 $x=(x_1,x_2)^\top$、隐藏变量 $h=x$ 和输出 $s=(1,1)h=x_1+x_2$。按隐藏坐标切分时，两个路径图分别为

$$
E_1=(x_1,0)^\top,\qquad E_2=(0,x_2)^\top.
$$

令可逆变换 $R=\begin{bmatrix}1&1\\0&1\end{bmatrix}$，并以 $h'=Rh$、$w'=R^{-\top}(1,1)^\top=(1,0)^\top$ 重写同一函数 $s=w'^\top h'$。新隐藏坐标下的路径图变为

$$
E'_1=(x_1,x_2)^\top,\qquad E'_2=(0,0)^\top.
$$

两种实现都有 $E_1+E_2=E'_1+E'_2=(x_1,x_2)^\top$，但分项不同。类似地，改变合法切面会改变被命名和汇总的路径组。因此本文的 Q/K/V、head 和 residual 图是对冻结 ViT 实现中指定真实边的分解，不是仅由端到端函数决定的唯一解释，也不具有一般实现不变性。数值闭合证明给定记账无遗漏或重复，不证明该分解跨等价图表示唯一。

### 4.5 质量、抵消与 Value-only 近似

在原始 normalized-input RGB 张量坐标上，以像素 $p$ 的 RGB absolute mass

$$
M(p)=\sum_{c\in\{R,G,B\}}|E_{p,c}|,
$$

作为主空间度量，并保留 $M^+(p)=\sum_c\max(E_{p,c},0)$ 与 $M^-(p)=\sum_c\max(-E_{p,c},0)$ 分析符号。Patch 质量为 $m_j=\sum_{p\in P_j}M(p)$，非零总质量时归一化为 $p_j=m_j/\sum_km_k$；零质量样本传播为 NA，不用 $\varepsilon$ 伪造分布。定义归一化空间熵 $H=-\sum_jp_j\log p_j/\log196$。Patch-token 的 own-token locality 取目标 patch 的 Chebyshev 半径 1 邻域质量并减去同面积均匀基线，CLS 不计算该指标；normalized distance 以目标 patch 到其他 patch 中心的欧氏距离除以 $13\sqrt2$ 后作质量加权平均。

对完整 RGB 张量另定义 $m(E)=\sum_i|E_i|$、$a(E)=\sum_iE_i$。单图抵消和两路径抵消分别为

$$
C^{\mathrm{map}}=1-\frac{|\sum_iE_i|}{\sum_i|E_i|},\qquad
C^{\mathrm{path}}=1-\frac{\|E^{(1)}+E^{(2)}\|_1}{\|E^{(1)}\|_1+\|E^{(2)}\|_1}.
$$

Routing scale 为 $\|E^{\mathrm{routing}}\|_1/(\|E^{\mathrm{routing}}\|_1+\|E^{\mathrm{value}}\|_1)$；identity scale 同理。除比例外，实验同时报告 routing/value/full 与 identity/branch/output 的 absolute mass。Value-only 近似保留 $A\mathrm dV$，遗漏 $(\mathrm dA)V$，其 Endpoint 相对误差为 $\|E^{\mathrm{full}}-E^V\|_2/\|E^{\mathrm{full}}\|_2$；JVP discrepancy 在冻结方向上使用对应的完整方向响应为分母。约 0.95 的误差表示与完整响应同量级的差异。

统计在完整 signed RGB 张量上计算，零分母或零范数记为 NA。蓝底显示规则在实验设置中说明。

### 4.6 有限替换与空间区域选择

成分有限效应为 $e_k=F(x)-F(x;z_k\leftarrow z_k^{\mathrm{ref}})$，主要统计比较每图 $m(E_k)$ 与 $|e_k|$ 的 Spearman 相关。Taylor 量为 $p_k=\langle\nabla_{z_k}F,z_k(x)-z_k^{\mathrm{ref}}\rangle$。干预仅替换模型前向中实际存在的 head output 或 residual edge activation；routing/value 与 Q/K/V differential 是导数路径，不被当作可直接替换的前向激活。Endpoint 和 raw gradient 只使用 clean 状态，属于 clean-only 方法；regional Taylor 显式使用 reference 位移，属于 reference-aware 方法。二者的信息预算不同，比较结果解释为任务适配性，而非无条件方法排名。

空间任务固定 block 5 的三个 heads，将输入分成 $4\times4$ 区域并依次替换 1、2、4 个区域。干预只把目标 head 的边替换为对应扰动输入产生的 activation，其余 heads 与 identity 保持 clean。每条曲线先取 absolute isolated effect，在面积 0–25% 上作梯形积分并除以 0.25，再在图内平均三个 heads。随机基线对每次重复先取 absolute effect 并积分，再平均 20 次 AUC；其估计器与确定性方法使用相同绝对值顺序。Endpoint、total G×I、raw gradient、regional Taylor 和 random 共用候选池与面积预算。


## 5. 实验设置与数值验证

### 5.1 模型与图像

实验采用 ViT-B/16 和 DeiT-S/16 两个 ImageNet 分类模型。两模型共享 400 张 evaluation 图像；另有互不重叠的 64 张 discovery 和 64 张 calibration 图像。输入短边 resize 到 248 后中心裁剪为 224，采用各模型的 normalization 参数，推理时关闭 dropout 和 drop path。Table 2 列出模型设置。

**Table 2. 模型设置**

| Model / setting | ViT-B/16 | DeiT-S/16 |
| --- | --- | --- |
| Blocks / heads / hidden dimension | 12 / 12 / 768 | 12 / 6 / 384 |
| Registered scalar targets/image | 457 | 385 |
| Evaluation images | 400 | 400 (same IDs) |
| Resize / crop | 248 / 224 | 248 / 224 |
| Normalization mean | [0.5, 0.5, 0.5] | [0.485, 0.456, 0.406] |
| Normalization std | [0.5, 0.5, 0.5] | [0.229, 0.224, 0.225] |

分层单元研究使用另外 100 张共享图像，每层选择 CLS、15 个空间分层 patch tokens 和 32 个预定 channels。每个 model-image 包含 6,144 个 post-block 标量，共 1,228,800 个记录。原目录覆盖 ViT-B/16 与 DeiT-S/16 post-block token-channel 候选的 0.01058%/0.02115%，分层研究覆盖 0.338%/0.677%。这些比例界定了本文的单元抽样范围。

E5 外部复现另使用 400 张图像，与上述原始队列和同轮 E1 队列均零重叠。它同时评价原始 ViT-B/16、DeiT-S/16 以及公开的 ViT-B/16 AugReg-IN1K、ViT-S/16 AugReg-IN21K→IN1K 权重。新增 checkpoint 是不同训练方案下的公开参数状态，不作为独立训练种子或纯模型规模对照。四个 checkpoint 共形成 1,600 个 image–checkpoint 任务。

### 5.2 可视化设置

概念和层间图使用主体可辨的懒熊样本，方法对比另外使用螳螂和小蓝鹭。样本按展示可读性选择，总体结论仍来自完整样本。为对应学习图像与前向特征，每层按 patch activation 的空间方差选择通道，并将该通道的 energy 定义为标量目标。图中同时给出通道编号及实际 14×14 前向 feature map；不同层可以选择不同通道。

主学习图像采用深蓝至青、黄、红的正部视图。设 $P=\max(\sum_cE_c,0)$，显示值为 $\sqrt{\min(P/q_{.99},1)}$，其中 $q_{.99}$ 在正值像素上计算。该变换提高弱细节的可见性，并保持像素位置；它不参与任何统计。补充图件同时显示 signed 与 absolute 视图，并另列 patch-max 加双线性插值的显示变体，明确区分原生细粒度图与平滑区域图。

### 5.3 统计与数值验证

所有总体推断先在图内聚合，以图像作为统计单位，使用 10,000 次 bootstrap。成分相关规则的 0.10 与区域增益规则的 0.05 是点估计门槛；符号检验检验相对零的方向，采用原 28 项 Holm 校正，CI 精度上限分别为 0.30/0.50。单元层深统计按模型使用五项校正；后续 residual spatial-TV 分析单独标为 post-hoc。E5 对每个 checkpoint 的 H1–H4 分别计算 400 个图像级值、95% bootstrap 区间和双侧精确符号检验，并在四项假设内作 Holm 校正。

表格用于报告离散数值验证，避免把少量误差点绘为趋势。Table 3 汇总五类原始 NNS 兼容网络的数值一致性与结构检查。原始 float32 有限差分失败和 float64 复核保留在补充材料中。

**Table 3. 数值一致性与结构检查**

| Compatible network | Max relative L2: float64 | Max relative L2: float32 | Comparisons |
| --- | --- | --- | --- |
| fcn | 4.371e-16 | 1.964e-07 | 1,000 |
| cnn | 4.289e-16 | 2.189e-07 | 1,000 |
| cnn_maxpool | 4.769e-16 | 2.427e-07 | 1,000 |
| residual_mlp | 2.686e-16 | 2.155e-07 | 1,000 |
| residual_cnn | 2.923e-16 | 1.692e-07 | 1,000 |

| Structural / directional check | Count | Result |
| --- | --- | --- |
| Positive closures | 76,800 | All pass |
| Omission/duplication controls | 28,800 | All rejected |
| Analytic JVP | 6,144 | All pass |
| Operator directions | 88 | All pass |
| Atlas float32 finite differences | 57,600 | 36 outside tolerance |
| Same-coordinate float64 review | 36 | All pass |

### 5.4 结果顺序与证据层级

结果按“身份与正确性—背景规律—联合索引机制结果—跨 checkpoint 复现—行为标定”排列。第一部分比较原始 NNS 逐层传播与自动微分 Input × Gradient，并检验 Q+K+V、routing+value、各 head 以及 residual identity+branch 的结构闭合。第二部分报告 entropy、own-token locality 和 normalized distance 的总体层深背景。第三部分用路径 absolute mass、相对尺度及归一化空间分布分析 attention 和 residual 的重组，并区分总图、edge scalar 与 $[e,i]$ 图组。第四部分在四个公开 checkpoint 上复现冻结方向并完整保留 H4 阴性结果。第五部分执行有限替换，比较 clean-only Endpoint 与 reference-aware Taylor。正确性、结构机制和有限干预效用是不同证据层级。

## 6. ViT 内部局部依赖的层深规律

### 6.1 RQ1–RQ2：NNS 等价性与结构路径闭合

Table 3 的 5,000 个原始 NNS 兼容网络比较表明，逐层 learning-image propagation 与同目标 Input × Gradient 的最大 relative L2 error 在 float64 下不超过 $4.769\times10^{-16}$，在 float32 下不超过 $2.427\times10^{-7}$。结构验证进一步覆盖 76,800 项正向闭合、28,800 项遗漏/重复负控、6,144 项解析 JVP 和 88 个算子方向。所有正向闭合与解析方向检查通过，所有负控被拒绝；36 个 float32 有限差分超差点在相同坐标的 float64 复核中全部通过。这些结果分别建立命题 1 的数值身份和命题 2 的结构记账；有限干预效用由 Section 7 单独评价。

### 6.2 RQ3：学习图像与前向特征的层间变化

Figure 5 以同一输入展示多类内部目标；全部 12 层和模块级的学习图像—前向 feature 配对移至补充图件。前向 feature map 给出通道在 token 网格上的值，学习图像给出相应 scalar 对输入像素的局部依赖；两类空间图因此能够相互补充。浅层学习图像保留局部纹理，后续层产生不同的分布和细节。图中也存在背景响应与离散热点，表明高分辨率本身不保证完整物体轮廓。

![Figure 5](../publication_r2_20260920/figures/png/fig_08_internal_targets.png)

**Figure 5.** 同一图像的不同内部与类别目标。Energy targets 使用前向空间方差最大的通道；这与单个 token-channel 目标和最终类别目标区分。

单元抽样统计进一步描述这一变化（Figure 6）。ViT-B/16 与 DeiT-S/16 的 late-minus-early entropy 增加 0.2201/0.2250，own-token locality 下降 0.3558/0.3424，normalized distance 增加 0.1312/0.1219。各对比均有 100/100 张图呈相同方向。这里的 early/middle/late 指 blocks 0–3/4–7/8–11，结果表明抽样单元依赖更广、更远的输入区域。

![Figure 6](../publication_r2_20260920/figures/png/fig_12_spatial_broadening.png)

**Figure 6.** 完整样本的图像级层深轨迹，阴影为 95% bootstrap 区间。空间分析每模型 100 图，attention/residual 每模型 400 图；推断沿用预先指定的分析。

CLS-minus-patch entropy 从 block 0 的 0.4752/0.4712 缩小到 block 11 的 0.0086/0.0238。Net/absolute mass 同时从 0.00872/0.00712 降至 0.00144/0.00132。空间扩展与较强正负抵消可以同时出现，absolute mass 因而描述局部项的总幅值而非净响应。

### 6.3 RQ3：联合索引必要性

总 Gradient × Input、edge scalar 和双索引图组保留的信息不同。Table 4 将三者对应到本文的实际结果。层深空间扩展只需要总图，因此作为背景规律；routing share 与 identity share 只需要 edge-level absolute mass，因此作为支持性尺度结果。Attention 的 Full–Value 空间差异和 residual 的 output–identity 空间差异同时要求路径身份 $e$ 与输入位置 $i$，构成 `NOV-JOINT-1` 的核心经验结果。

**Table 4. 总图 / edge scalar / 双索引图组的信息差异与结果必要性**

| Representation | Retained indices | Directly recoverable question | Frozen result | Role after alternative-explanation audit |
| --- | --- | --- | --- | --- |
| Total G×I map | input coordinate $i$ | Is the target's local dependence spatially broader at depth? | Entropy +0.2201/+0.2250; locality −0.3558/−0.3424; distance +0.1312/+0.1219, 100 images/model | Background result; no edge assignment |
| Edge scalar | consumer-use edge $e$ | How does path magnitude change without retaining its spatial distribution? | Routing share +0.0523 [0.0467, 0.0587] / +0.1355 [0.1277, 0.1429], 400 images/model | Supporting scale result; insufficient for spatial reorganization |
| Dual-indexed map set | edge $e$ and input coordinate $i$ | Where do Q/K/V or identity/branch paths redistribute local dependence? | K/V entropy decreases in both models, Q is model-dependent, 400 images/model; residual spatial contrasts in Section 6.5, 400 images/model | Core joint-index result |

替代解释审计采用与正文主张相同的信息边界。总图保留 $i$ 但求和消去 $e$，不能恢复 routing/value 或 identity/branch 的分别空间分布；edge scalar 保留 $e$ 但求和消去 $i$，不能计算归一化 patch distribution、spatial TV 或 signed spatial cancellation。总 absolute mass 和 activation norm 可解释整体尺度，却不能重建这些位置条件差异。Full–Value 与 output–identity TV 均在各自绝对质量归一化后比较空间分布，因此不是总 mass 的重述。由此，空间重组结论依赖 $[e,i]$；routing share、identity share 和总体空间扩展均不单独承担联合索引必要性主张。

### 6.4 RQ3：注意力头与 routing/value differential

Figure 7 将 attention 分为 Q、K、routing 和 value，使分配权重的变化与内容传递可以分别观察。全部 head atlas 与跨模型类别图移至补充图件。

![Figure 7](../publication_r2_20260920/figures/png/fig_15_routing_value_examples.png)

**Figure 7.** 同一懒熊输入在浅层和深层的 Q、K、routing 与 value 路径。蓝底保留正部细节；signed closure 和绝对质量来自完整原始数组。

在 400 张图的目标条件化分析中，routing scale 从 block 0 的 0.3092/0.2041 增至 block 11 的 0.5154/0.4808；预注册的 late-minus-early 图像级中位差为 +0.0523 [0.0467, 0.0587] / +0.1355 [0.1277, 0.1429]。Value-only Endpoint error 从 0.4407/0.2774 增至 0.8022/0.8488，其 late-minus-early 中位差为 +0.1444 [0.1329, 0.1518] / +0.2896 [0.2788, 0.3013]（Figure 8）。ViT-B/16 的 routing/value/full absolute mass 中位数从 block 0 的 2.6596/7.6041/9.8604 变为 block 11 的 40.7842/39.0241/54.6768；DeiT-S/16 对应为 4.5294/18.4261/20.1427 与 35.9257/38.8767/44.5653。相应 JVP discrepancy 达到 0.9516/0.9636，完整响应 norm 为 14.9711/1.5155，绝对差异为 13.8064/1.4228。比例变化与绝对范数共同表明，固定 attention 权重的 $A\mathrm dV$ 在深层更难近似完整局部响应；这些量不是前向输出或因果贡献的百分比。

现有冻结的 16 图原始图组还给出空间化检验：完整 attention 与 Value-only 的归一化 14×14 patch distribution TV 在 ViT-B/16 中从 block 0 的 0.0550 [0.0455, 0.0952] 增至 block 11 的 0.1865 [0.1534, 0.2116]，在 DeiT-S/16 中从 0.0307 [0.0268, 0.0365] 增至 0.1811 [0.1614, 0.2041]。该结果需要同时知道 Value edge 和每个输入位置；总图与 routing/value scalar 均不能直接恢复。

完整 400 图 Q/K/V 图组提供同方向的总体证据。K 的 normalized spatial entropy 在 ViT-B/16 与 DeiT-S/16 中分别变化 −0.0586 [−0.0636, −0.0528] 和 −0.0669 [−0.0736, −0.0590]；V 分别变化 −0.0530 [−0.0591, −0.0471] 和 −0.0669 [−0.0748, −0.0609]。Q 则呈模型依赖：ViT-B/16 为 +0.0268 [0.0217, 0.0314]，DeiT-S/16 为 −0.0038 [−0.0080, 0.0005]。该相反趋势不并入共同规律。Q/K/V 空间轨迹需要同时保留路径标签和 patch distribution，是 attention 的主要联合索引必要性结果；16 图 Full–Value TV 是独立的描述性支持。

![Figure 8](../publication_r2_20260920/figures/png/fig_16_attention_depth.png)

**Figure 8.** 完整样本的图像级层深轨迹，阴影为 95% bootstrap 区间。空间分析每模型 100 图，attention/residual 每模型 400 图；推断沿用预先指定的分析。

控制 activation norm 与下游 cotangent norm 后，mean-query saturation 与 routing scale 的偏秩相关仍为 −0.1442/−0.3257。Sharper attention 不能直接作为更强 routing dependence 的代理；CLS-query 结果较弱，完整敏感性分析见补充材料。

### 6.5 RQ3：Residual identity/branch differential

Figure 9 以同一输入和目标展示 attention 与 MLP residual。两种结构均为 identity 加 branch，但 identity 的输入边不同；其局部依赖也呈现不同的层深趋势。Attention residual 的 late-minus-early identity scale 为 −0.1242/−0.0449，MLP residual 为 +0.1798/+0.1780（Figure 10）。

![Figure 9](../publication_r2_20260920/figures/png/fig_17_residual_examples.png)

**Figure 9.** 同一 block 5 中 attention 与 MLP residual 对同一类别目标的两条路径。两行的 identity 指向不同 residual 输入边。正部图不可直接相加，完整 signed maps 满足闭合。

![Figure 10](../publication_r2_20260920/figures/png/fig_18_residual_depth.png)

**Figure 10.** 完整样本的图像级层深轨迹，阴影为 95% bootstrap 区间。空间分析每模型 100 图，attention/residual 每模型 400 图；推断沿用预先指定的分析。

Attention residual 的 identity/branch absolute mass 在 ViT-B/16 中由 block 0 的 495.2271/63.8344 变为 block 11 的 313.3689/398.0202，在 DeiT-S/16 中由 459.3077/63.9204 变为 372.6598/185.3893。MLP residual 对应从 130.4998/438.0538 变为 827.8407/460.2701，以及从 170.6143/412.5634 变为 614.5721/354.6503。后续全样本空间分析得到，attention output–identity patch TV 的 late-minus-early 图像级中位差为 +0.0520 [0.0488, 0.0547] / +0.0282 [0.0257, 0.0300]，MLP 为 −0.0187 [−0.0220, −0.0150] / −0.0462 [−0.0516, −0.0412]，每模型、每 residual 类型均为 400 图，四项通过单独 Holm 校正。该结论比较同一 residual 输出图与其 identity-edge 图的归一化空间分布：总图不能标识 identity，edge-only mass 不能计算空间 TV，因此 residual 的相反层深轨迹依赖 $[e,i]$ 联合信息。Token-channel 层间分析与类别路径使用不同目标，二者提供互补的模型级观察。

### 6.6 E5 四检查点外部复现

E5 在与原始队列和同轮 E1 队列零重叠的 400 张图像上检验四项事前冻结假设。H1 是 32 个 post-block 神经元的 entropy 在 block 11 相对 block 0 的差；H2 是 routing 相对 Q/K-routing 与 V 总质量的比例差；H3 是 MLP residual identity 轨迹减 attention residual identity 轨迹；H4 是四个代表 block 中，路径图相对 activation × gradient 的 head 排序相关增量。Table 5 报告图像级中位数和 95% bootstrap 区间。

**Table 5. E5 四检查点复现；每格为图像级中位数 [95% bootstrap CI]，$n=400$**

| Checkpoint | H1 | H2 | H3 | H4 |
| --- | --- | --- | --- | --- |
| ViT-B/16 AugReg-IN21K→IN1K | 0.4125 [0.4049, 0.4196] | 0.2386 [0.2206, 0.2594] | 0.8507 [0.8389, 0.8646] | -0.1136 [-0.1285, -0.0909] |
| DeiT-S/16 IN1K | 0.3778 [0.3699, 0.3826] | 0.3223 [0.3097, 0.3293] | 0.5222 [0.4953, 0.5427] | -0.1857 [-0.2286, -0.1571] |
| ViT-B/16 AugReg-IN1K | 0.3358 [0.3300, 0.3396] | 0.2406 [0.2179, 0.2552] | 0.6323 [0.6196, 0.6478] | -0.1862 [-0.2054, -0.1678] |
| ViT-S/16 AugReg-IN21K→IN1K | 0.3900 [0.3837, 0.3986] | 0.2763 [0.2558, 0.2959] | 0.8670 [0.8400, 0.8909] | -0.1357 [-0.1714, -0.1143] |

H1–H3 在四个 checkpoint 中均为正，checkpoint 内 Holm 校正后 $p\leq1.94\times10^{-76}$。H4 在四个 checkpoint 中均为相反方向，路径图的图内 head 排序相关比 activation × gradient 低 0.1136–0.1862，Holm $p\leq8.86\times10^{-20}$。因此空间扩展、routing 与 residual 轨迹在这些公开参数状态中复现；稳定的路径图 head 排序增量未获支持。新增权重不是独立训练种子。

### 6.7 表示对象的可视化比较

补充图件比较同一输入上的 Endpoint、Grad-CAM、attention rollout 和 raw attention。Endpoint 与 Grad-CAM 均使用 GT logit；Grad-CAM 在最后一个 block 的 norm1 输出处计算，避免最终 CLS 读出使末端 patch 梯度退化。Attention rollout 使用 head 均值、identity residual、逐行归一化及逐层矩阵相乘，raw attention 使用末层 CLS-to-patch head 均值。

Endpoint 保留像素尺度的局部项，Grad-CAM 和 attention 类方法提供较粗的区域分布。Patch-max 插值版本表明，聚合与插值本身就能显著改变学习图像的外观；它不改变基础定义。图中各方法有时突出不同区域，这反映目标与传播规则的差别。该比较用于说明表示粒度与对象，方法排序仍需要对应任务的定量评价。

### 6.8 RQ3 控制：训练与初始化

在每模型 32 张共享图像、全部 12 blocks 和 3 个随机初始化状态上，随机模型也表现出空间扩展。训练模型 entropy 的 late−early 为 0.2288/0.2256，随机均值为 0.0940/0.1718；训练−随机差分别为 0.1348 [0.1232, 0.1466] 和 0.0539 [0.0473, 0.0602]。区间同时重采样模型种子和共享图像。Routing scale 的 trained/random 层深变化为 0.0512/−0.0479 与 0.1314/−0.0043；attention identity 在训练模型中减少、随机模型中增加。MLP identity 在两类状态中均增加，ViT-B 的增量差区间跨零。训练参数状态与随机参数状态在空间扩展幅度和部分路径趋势上存在上述差异；该对照不识别训练过程的因果效应。

### 6.9 RQ3 控制：目标稳健性

在每模型 100 张冻结图像的 blocks 0、5、11 上，GT、clean-top1 与 top1 margin 均保留 routing/Value-only error 增加、attention identity 减少和 MLP identity 增加的方向。ViT-B 与 DeiT-S 的正确/错误图数量为 92/8 和 86/14，两个子集分别报告。Margin 改变效应幅度，例如 ViT-B routing 的 B11−B0 从 GT 的 0.2031 变为 0.1439。Native attention-output JVP 不依赖类别目标；另外报告目标 cotangent 收缩的方向 discrepancy。

### 6.10 前向 feature 与 Endpoint 的定量关联

在每模型 32 图、12 blocks、每层 16 个冻结随机 channels 上，12,288 个 patch-energy 目标给出前向平方响应分布与 Endpoint absolute patch mass。正确配对的图像级 Spearman 均值为 0.2663/0.2811，均高于同层错误 channel、跨层同 channel、空间置换和随机 channel 控制；其配对差区间均为正。Cosine、Jensen–Shannon distance 和 top-20 overlap 得到对应的定量比较。前向响应与局部输入依赖具有同通道关联，仍为互补对象。

### 6.11 RQ3 控制：抽样敏感性

复用相同 32 图的原 M3 单元记录，channel/token-stratum/image bootstrap 以及 leave-one-block/stratum-out 均保留 entropy 增大、locality 减小与距离增大的方向。该结果描述既有样本的抽样敏感性。全部实验设置、逐项指标、区间及五张补充图见[补充实验 v1](../publication_r2_20260920/zh/supplement_experiments_v1.md)。

## 7. 规定有限替换任务中的排序能力

### 7.1 RQ4：规定成分替换与区域选择

在成分替换实验中，12 个模型–reference–成分族组合有 11 个满足原复合条件。Table 6 直接列出相关估计、区间和阈值关系。ViT-B normalized-zero routing/value 为 0.0996，DeiT-S 对应估计为 0.1065，两个区间均跨越 0.10。因此 family 条件是否满足与总体效应是否超过 0.10 是不同判断；全部成分族的联合判定未获支持。该结果保持为广泛但非普遍，不从证据结构中删除相反模型趋势。

**Table 6. 成分关联估计及区间**

| Model | Reference | Family | Median rho | 95% CI | Holm p | Composite criterion | CI lower > 0.10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | heads | 0.2909 | [0.2813, 0.2982] | 2.17e-119 | Met | yes |
| ViT-B/16 | normalized-zero | routing/value | 0.0996 | [0.0748, 0.1226] | 3.00e-14 | Unmet | no |
| ViT-B/16 | normalized-zero | residual/branches | 0.1774 | [0.1548, 0.1952] | 2.56e-46 | Met | yes |
| ViT-B/16 | blur | heads | 0.3063 | [0.2957, 0.3181] | 2.17e-119 | Met | yes |
| ViT-B/16 | blur | routing/value | 0.1565 | [0.1322, 0.1774] | 9.96e-29 | Met | yes |
| ViT-B/16 | blur | residual/branches | 0.2091 | [0.1817, 0.2374] | 1.46e-41 | Met | yes |
| DeiT-S/16 | normalized-zero | heads | 0.1910 | [0.1769, 0.2052] | 1.88e-76 | Met | yes |
| DeiT-S/16 | normalized-zero | routing/value | 0.1065 | [0.0817, 0.1291] | 2.11e-10 | Met | no |
| DeiT-S/16 | normalized-zero | residual/branches | 0.1352 | [0.1148, 0.1643] | 4.35e-19 | Met | yes |
| DeiT-S/16 | blur | heads | 0.1817 | [0.1688, 0.1951] | 3.82e-72 | Met | yes |
| DeiT-S/16 | blur | routing/value | 0.1600 | [0.1278, 0.1822] | 9.90e-31 | Met | yes |
| DeiT-S/16 | blur | residual/branches | 0.2287 | [0.2035, 0.2504] | 3.70e-54 | Met | yes |

补充同层 head 比较中，Taylor 具有更高 Spearman 与 Capture、更低 Regret（Table 7）。跨层 residual identity 的 Endpoint−Taylor Spearman 差为 0.0865–0.4209，但 Capture@1 中位差为零，表明增量价值依赖任务和指标。

**Table 7. 同层 head 排序：E=Endpoint，T=Taylor**

| Model | Reference | E rho | T rho | E Capture@1 | T Capture@1 | E Capture@3 | T Capture@3 | E Regret@1 | T Regret@1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | 0.2815 | 0.6742 | 0.5174 | 0.8038 | 0.6149 | 0.8567 | 0.1701 | 0.0571 |
| ViT-B/16 | blur | 0.2727 | 0.6856 | 0.5075 | 0.8203 | 0.6185 | 0.8659 | 0.1163 | 0.0346 |
| DeiT-S/16 | normalized-zero | 0.1952 | 0.6095 | 0.5609 | 0.8028 | 0.7122 | 0.8894 | 0.2381 | 0.0860 |
| DeiT-S/16 | blur | 0.1476 | 0.6238 | 0.5506 | 0.8426 | 0.7105 | 0.8970 | 0.1326 | 0.0443 |

区域实验固定 block 5 的三个 heads，所有方法共用 4×4 候选区域及最高 25% 面积。Figure 11 和 Tables 8–10 给出配对差、绝对 AUC 与归一化 regret。Endpoint−regional Taylor 的配对中位差为 −0.023433、−0.023697、−0.029900、−0.035630；四个 family 都未满足相对全部基线的 +0.05 条件。候选 oracle 仅取实际评价集合的最优效果，不能视为全部可能区域的全局最优。

![Figure 11](../publication_r2_20260920/figures/png/fig_21_intervention_boundary.png)

**Figure 11.** 左侧示例的有限区域替换曲线与右侧完整 400 图 family 的绝对 AUC。候选 oracle 只在已评价区域集合中取最优；数学闭合与区域干预选择属于不同性质。

**Table 8. 空间干预配对差**

| Model | Reference | vs total G×I | vs gradient | vs Taylor | vs random | Composite criterion |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | -0.0009 | -0.0000 | -0.0234 | -0.0023 | Unmet |
| ViT-B/16 | blur | -0.0010 | -0.0007 | -0.0237 | -0.0031 | Unmet |
| DeiT-S/16 | normalized-zero | -0.0018 | -0.0006 | -0.0299 | 0.0009 | Unmet |
| DeiT-S/16 | blur | 0.0002 | -0.0002 | -0.0356 | -0.0010 | Unmet |

**Table 9. 绝对 AUC 与输出尺度**

| Model | Reference | Endpoint AUC | Taylor AUC | Random AUC | Candidate oracle AUC | Clean GT logit |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | 0.0309 | 0.0558 | 0.0339 | 0.0654 | 8.8989 |
| ViT-B/16 | blur | 0.0276 | 0.0518 | 0.0307 | 0.0747 | 8.8989 |
| DeiT-S/16 | normalized-zero | 0.0617 | 0.0961 | 0.0611 | 0.1474 | 8.7974 |
| DeiT-S/16 | blur | 0.0395 | 0.0770 | 0.0411 | 0.1063 | 8.7974 |

**Table 10. 候选 oracle regret 与 logit 尺度比**

| Model | Reference | Endpoint regret | Taylor regret | Random regret | Endpoint AUC/abs(logit) | Taylor AUC/abs(logit) |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | 0.5303 | 0.1442 | 0.5059 | 0.0035 | 0.0062 |
| ViT-B/16 | blur | 0.6366 | 0.2557 | 0.5929 | 0.0031 | 0.0059 |
| DeiT-S/16 | normalized-zero | 0.5646 | 0.3352 | 0.5947 | 0.0072 | 0.0107 |
| DeiT-S/16 | blur | 0.6289 | 0.2705 | 0.6153 | 0.0045 | 0.0089 |

修正随机基线后，Endpoint−random 的配对中位差为 −0.0023、−0.0031、0.0009、−0.0010；DeiT-S 两个区间跨零。Taylor−random 的四个区间均为正。原 block-5 三-head 结果对应确认性协议；补充跨层全-head M2 覆盖 blocks 2/5/10，在完整输入替换的 11/12 family 与单-head 隔离替换的 12/12 family 显著偏向 Taylor，详见补充实验 S14。

### 7.2 行为标定与 estimand 边界

跨 reference 分析补充了这一结果的含义。Endpoint 使用观测输入处的 clean map，而 Taylor 显式使用替换位移。对 163,200 个成分配对，Taylor magnitude 与 absolute effect 在原 atol=1e-12、rtol=0 标准下均发生变化；分布及 head 排名一致性见补充表。Reference 会改变干预相关排序，局部输入图与位移估计器因而适用于不同问题。

## 8. 讨论与结论

### 8.1 讨论

本文的唯一定位是以 Gradient × Input 为局部测量原语，对 ViT 异构内部目标进行结构化 Jacobian 扫描，并在指定真实计算边上实现可闭合、可审计的 differential accounting。命题 1 界定数学身份，命题 2 界定指定图与切面上的闭合条件。核心机制结论由两个原始固定 checkpoint 建立，并在独立 400 图和四个公开 checkpoint 上检验冻结方向。

结构闭合、局部路径尺度、双索引空间重组和有限替换排序对应不同问题。Q/K/V、multi-head 与 identity/branch 的加和闭合说明各项重构指定目标的局部微分；routing 或 identity scale 比较路径图的相对范数；Full–Value 与 output–identity spatial TV 使用 $[e,i]$ 图组定位路径间的空间重组；activation replacement 测量指定 reference 和替换算子下的有限效应。本文的 RQ4 只评价完整替换与固定面积预算下的成分排序和区域选择，不估计 Taylor 误差随干预幅度变化的边界。

学习图像使不同内部对象能够在同一输入网格上观察。明确的目标、层与通道支持比较前向响应和局部输入依赖，以及注意力与残差路径如何组合。正确 feature–Endpoint 配对优于四种负控制，但二者的相关程度远低于等价；高分辨率输入图也受 $16\times16$ patch embedding 与归一化坐标原点约束。

蓝底高亮视图便于读取分散纹理，但只显示 signed RGB-sum 的正部。补充图件将其与 signed 和 absolute 视图并列；正部表示局部坐标缩放项的符号，不等同于有限删除时的预测支持，较大 absolute mass 也可能伴随较强抵消。

训练控制覆盖每架构三个初始化种子；E5 增加两个公开训练方案下的参数状态，但不是独立训练种子。证据范围不包含训练轨迹、独立 channel seeds 或 cascading randomization。错误分类子集含 8/14 图，空间干预覆盖既定区域、references 与 blocks 2/5/10。ViT-B Path IG 在既定 512 点范围内未达到共同校准要求，跨模型 Endpoint–Path 比较位于本文结果范围之外。

### 8.2 结论

本文得到四点结论。第一，使用精确当前点局部导数且无额外重分配时，NNS 标量学习图像等于 Input × Gradient，因而 Gradient × Input 是框架的基础测量原语。第二，本文在指定 ViT 计算图上建立异构内部目标 registry 和真实边 differential cuts；attention 的 Q-induced、K-induced 与 V-induced 路径以及 residual 的 identity 与 branch 路径在合法切面上加和恢复完整局部输入依赖，分项随图、切面和参数化而变。第三，总图给出空间扩展背景，$[e,i]$ 图组进一步揭示 attention Full–Value 与 residual output–identity 的空间重组；H1–H3 在独立 400 图和四个公开 checkpoint 上保持方向。第四，局部路径图在规定成分替换任务中与有限效应具有广泛但非普遍的关联，H4 的 head 排序增量在四个 checkpoint 上均为负，给定 reference 的规定区域替换更适合使用位移感知 Taylor 排序。本文由此形成结构化局部 Jacobian 扫描、可验证记账和跨 checkpoint 机制研究的统一框架。

## 参考文献与补充材料

[参考文献](../publication_r2_20260920/references.md)；[补充材料](../publication_r2_20260920/zh/supplement.md)。逐主张证据索引、主文/补充分配和机器重算记录位于 `evidence/gxi_step4_evidence_index.md`、`evidence/gxi_step4_figure_allocation.md` 与 `evidence/gxi_step4_numeric_recalculation.json`。作者、单位及最终归档信息待补充：[REPOSITORY AND ARCHIVE REQUIRED]。
