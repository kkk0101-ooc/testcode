# 从 NNS 到视觉 Transformer：内部目标的可审计 Jacobian 扫描与结构分解

- 作者：[AUTHOR REQUIRED]
- 单位：[AFFILIATION REQUIRED]
- 通讯作者：[CORRESPONDING AUTHOR REQUIRED]

## 摘要

神经网络扫描器（Neural Network Scanner，NNS）通过逐层传播局部尺度，为网络内部目标生成保留输入位置的学习图像。当各层使用实际前向状态下的精确局部导数，且不加入归一化、截断或相关性重分配时，标量 NNS 学习图像等于 $x\odot\nabla_xs(x)$，即同目标的 Gradient × Input。以 Gradient × Input 为局部测量原语，本文构建面向 ViT 异构内部目标的结构化、可闭合、可审计扫描框架，并用它系统研究局部输入依赖如何随层深、attention 路径和 residual 分支重新组织。该框架注册异构内部目标，在真实计算边上定义合法 differential cuts，将 attention differential 分为 Q-induced、K-induced 与 V-induced 路径，将 residual differential 分为 identity 与 branch 路径，并检验各切面的加和闭合。分项相对于指定计算图、切面和参数化定义。ViT-B/16 与 DeiT-S/16 两个固定 ImageNet 预训练 checkpoint 上的实验覆盖数值正确性、路径层深统计、训练参数状态与随机参数状态的差异，以及规定有限替换任务中的排序能力。冻结抽样目标的局部输入依赖随深度扩展到更广、更远的 patch；深层 attention 的 routing 相对范数和 Value-only 近似误差提高，attention 与 MLP residual 呈现不同的层深变化。局部路径图与有限成分效应存在广泛但非普遍的关联；给定 reference 的区域替换中，显式使用实际位移的 Taylor 排序优于 clean-only Endpoint。

关键词：视觉 Transformer；Neural Network Scanner；学习图像；局部 Jacobian；结构路径分解

## 1. 引言

视觉 Transformer（ViT）将图像划分为 patch tokens，随后通过自注意力和前馈网络逐层建立表示（Dosovitskiy et al., 2021）。这一结构能够处理不同位置间的关系，也使内部计算的解释更加复杂：token-channel 单元描述特定位置的响应，attention head 在多个位置间传递信息，残差连接又将输入与分支输出叠加。要理解这些对象之间的联系，需要一种能够保留输入空间位置、并适用于不同内部对象的表示。

现有可视化通常展示特征图、注意力权重或类别相关区域。这些视图各有用途，但其空间分辨率、目标和传播规则不同。前向特征图说明一个通道在 token 网格上的响应，attention 权重说明 token 间的分配关系；它们与某个内部标量关于原始图像的依赖并不是同一个对象。将内部目标映射回统一输入坐标，可以为跨模块比较提供共同的观察基础。

Neural Network Scanner（NNS）为神经元关联一张保留输入位置的 learning image（Dou et al., 2023）。原方法的逐层尺度传播在使用实际局部导数时可以写成 Jacobian 链式法则；对标量目标，其终点学习图像就是 Input × Gradient。这个恒等式确定了本文的局部测量原语。普通总梯度不能区分局部响应经过 Q、K、V、不同 heads，还是 residual 的 identity 与 branch。本文据此构建面向 ViT 异构内部目标的结构化、可闭合、可审计扫描框架，将完整局部输入依赖分解到真实计算边，并研究这些依赖如何随层深和结构路径重新组织。

![Figure 1](../publication_r2_20260920/figures/png/fig_01_concept_overview.png)

**Figure 1.** 同一输入经过正常前向计算得到预测；NNS-ViT 为内部目标与输出目标提供保留输入位置的学习图像。虚线表示目标与学习图像的对应关系。

本文的总体问题是：如何在 NNS 框架下，将 ViT 内部目标对输入的局部依赖表示为可计算、可分解和可验证的结构路径图，并评价其层深规律及在规定有限替换任务中的排序能力？具体研究四个问题：

1. **RQ1：NNS 学习图像的数学本质是什么？** 明确逐层尺度传播、Jacobian 链式法则与 Input × Gradient 的关系，以及非齐次项、固定 token 和输入坐标原点带来的限制。
2. **RQ2：NNS 如何推广到 ViT 的结构化计算图？** 定义内部标量目标、真实计算边、合法计算切面及 attention、multi-head 和 residual 的可加局部微分路径。
3. **RQ3：结构路径如何随深度变化？** 在两个固定预训练 checkpoint 的冻结抽样目标上分析空间扩展、routing/value 与 identity/branch 的层深规律，并比较训练参数状态和随机参数状态以及目标与抽样控制。
4. **RQ4：局部路径图能否在规定的有限替换任务中排序成分和空间区域？** 对固定模型、样本、层、head、reference、替换算子和区域预算，比较 clean-only 局部路径图与 reference-aware 位移估计的成分排序和区域选择。

本文作出三项贡献。第一，**数学身份与对象定义**。我们给出原始 NNS learning-image propagation 等于同目标 Gradient × Input 的条件，并区分 Endpoint、Path IG、模型常数和 reference-specific finite replacement；Table 3 的 5,000 项公式回归及 88 个算子方向检查验证了该对象定义。第二，**结构化扫描与审计框架**。我们建立面向 ViT token-channel、通道聚合、attention head、residual edge 与类别目标的 registry，在真实计算边上定义合法 differential cuts，并对 Q/K/V、multi-head 和 identity/branch 局部微分进行可加记账；Table 2 的 457/385 个注册目标以及 76,800 项正向闭合、28,800 项遗漏/重复负控和 6,144 项解析 JVP 构成直接证据。第三，**机制观察与解释边界**。我们在两个固定 ViT checkpoint 的冻结样本上测量层深、attention 和 residual 的局部输入依赖规律，并以规定有限替换实验区分观测点局部表示与 reference-specific intervention estimator；Figures 6–11 与 Tables 4–8 给出图像级结果。

下文先界定相关方法与本文定位，再给出 NNS 的一般 Jacobian 表示和 ViT 结构分解。实验按等价性与闭合、层深规律、结构路径和有限干预组织，最后讨论证据范围。

## 2. 相关工作

### 2.1 NNS 与学习图像

神经元可视化通过输入模式或实际样本分析内部响应。NNS 将单个神经元的计算组织为输入坐标 learning image，并用 activation-weighted aggregation 表示卷积通道（Dou et al., 2023）。这种组织方式使不同原生空间结构的内部对象能够映射到同一图像坐标。本文保留“学习图像”这一历史术语；在数学推导中称其为“局部输入依赖图”，以强调它描述当前输入点的一阶响应。

### 2.2 局部梯度、路径积分、FullGrad 与隐藏单元归因

Gradient × Input 是观测点上的输入加权梯度（Shrikumar et al., 2017）；Integrated Gradients 沿 reference 到输入的路径平均梯度并满足相应完整性公理（Sundararajan et al., 2017），conductance 将该路径归因扩展到隐藏单元（Dhamdhere et al., 2018）。FullGrad 将输入梯度项与各层 bias-gradient 项共同组成完整表示，再聚合为显著图（Srinivas and Fleuret, 2019）。本文的 Endpoint 使用当前点的真实 Jacobian，不进行路径积分，也不把 bias 项加入输入图。因此 conductance 的路径完整性、FullGrad 的函数值完整表示和本文合法切面的局部微分闭合是三个不同对象。

### 2.3 Transformer token contribution 与 relevance 传播

GlobEnc 把 encoder block 的 attention、residual 与 normalization 纳入逐层 token attribution，并跨层聚合（Modarressi et al., 2022）。DecompX 传播不混合的 token representation 分量，覆盖非线性前馈层和分类头，将 token decomposition 连接到预测（Modarressi et al., 2023）。AttnLRP 为 attention 与 skip connection 定义 relevance 传播并支持输入和 latent attribution（Achtibat et al., 2024）；LibraGrad 通过剪枝和缩放反向路径校正 ViT 的 gradient-flow imbalance，以满足其 FullGrad-completeness 目标（Mehri et al., 2025）。这些方法已经覆盖 Transformer 内部贡献传播。本文保留模型真实局部导数，研究的是指定计算图切面上各真实边对同一局部微分的可加记账。

### 2.4 Activation patching 与 edge attribution

Activation patching 以 clean/corrupted reference 替换真实内部变量并测量有限输出变化；结果依赖 corruption、替换位置和评价指标（Zhang and Nanda, 2023）。Attribution patching 以 activation displacement 与局部梯度的内积近似单个组件的 patching effect；edge attribution patching 将这一近似用于计算图边（Syed et al., 2023）。AtP* 进一步处理 attention saturation 与抵消造成的近似误差（Kramár et al., 2024）。这些方法直接评价或近似 reference-specified finite effect。本文分别报告 clean-only 路径图、reference-aware Taylor 量和真实替换效应，不用结构闭合代替有限干预效用。

### 2.5 视觉归因与本文定位

Grad-CAM 通过梯度加权特征图获得类别相关定位（Selvaraju et al., 2017），attention rollout 组合跨层 attention flow（Abnar and Zuidema, 2020），Chefer et al.（2021）则为 Transformer 设计 relevance 传播。它们与 Gradient × Input、token decomposition、LRP 和有限替换面向不同解释对象。本文的方法位置汇总于 Table 1。

**Table 1. 方法定位：完整性指各方法自身声明的函数值、归因或局部微分闭合对象**

| Method | Primary object | External reference | Internal-path decomposition | Completeness / closure | Finite-intervention evaluation |
| --- | --- | --- | --- | --- | --- |
| Gradient × Input | Input coordinates at one point | No | No | No, except under homogeneity conditions | Not intrinsic |
| Conductance | Hidden-unit attribution along an input path | Yes | Hidden units, not graph-edge cuts | Yes, for its path-attribution object | Neuron ablation |
| FullGrad | Input and bias-gradient components | No | Layerwise bias components, not graph paths | Yes, for the full-gradient representation | Input perturbation, not internal replacement |
| GlobEnc | Token attribution through encoder blocks | No | Token/component propagation | No formal class-score closure | Input-token evaluation |
| DecompX | Token contributions to a prediction | No | Token decomposition through encoder components | Specified token decomposition, not this work's Jacobian-cut closure | Input-token erasure |
| AttnLRP | Input and latent relevance | No | Attention/skip relevance propagation | Yes, relevance conservation | Input perturbation |
| LibraGrad | ViT input attribution | No | Modified backward paths, not an additive edge cut | Targets FullGrad completeness | Input perturbation |
| Activation patching | Finite effects of internal variables | Yes | Nodes or edges selected for replacement | Not an additive completeness objective | Yes, direct replacement |
| Edge attribution patching | First-order finite-effect estimates on edges | Yes | Yes, computational edges | Not an additive completeness objective | Yes, checked against patching |
| NNS–ViT (this work) | Local input maps for ViT targets and edges | No for maps; yes for replacement evaluation | Yes, on specified Q/K/V, head and residual cuts | Yes, local-differential closure on a valid cut; no function-value completeness | Yes, prescribed replacement tasks |

本文的增量由后三个环节共同构成：把 NNS learning-image 表示实现到 ViT 异构目标与真实计算边；在指定图、切面和参数化下进行不遗漏、不重复的路径记账；以数值闭合、层深统计和有限替换分别评价正确性、结构规律与任务效用。

## 3. NNS 的 Jacobian 解释

### 3.1 NNS 的一般计算图表示

设网络为有向无环计算图 $G=(V,E)$，输入为 $x$，目标为标量 $s(x)$。一个神经元输出标量值，学习图像则保留该标量所依赖的输入坐标。Figure 2 用四坐标网络说明正常计算、输入坐标图与局部导数组合的对应关系；展开的坐标初始化与传播步骤移至补充图件。本文将最终标量图记为 $E_s(x)$，并称为 NNS 局部输入图。

![Figure 2](../publication_r2_20260920/figures/png/fig_02_neuron_learning_image.png)

**Figure 2.** 同一四坐标网络的正常数值计算与 NNS-ViT 学习图像传播。节点与对应学习图像使用同色标记；下方矩阵保留各输入项的位置，并按上方网络的局部系数组合。

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

## 4. ViT 内部目标的结构化 NNS

### 4.1 从输入 patch 到内部标量

ViT 的 patch embedding 将每个局部图像块映射到 token 向量。线性投影改变响应维度，而学习图像仍位于输入像素网格；对应的稀疏投影示意移至补充图件。

本文区分原生 token-channel 单元、聚合响应、结构路径和输出目标（Figure 5）。对通道 $c$ 的 patch activations，定义

$$
s_{\mathrm{energy}}=\frac12\sum_{t=1}^{196}h_{t,c}^2,\qquad
\nabla_xs_{\mathrm{energy}}=\sum_th_{t,c}\nabla_xh_{t,c},\qquad
E_{\mathrm{energy}}=\sum_th_{t,c}E_{h_{t,c}}.
$$

这给出原始 NNS activation-weighted filter aggregation 的 Jacobian 形式；energy 是聚合标量而非单个神经元。目标还包括 patch sum、head 内部坐标、final CLS channel、ground-truth logit 和 frozen clean-top1 logit。跨模型同编号 channel 不视为同语义单元。

### 4.2 Attention、multi-head 与 residual 的路径分解

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

这些量是当前输入和目标下的局部微分路径图，而不是 Q/K/V 激活或前向输出值的组成比例。Head 边位于拼接和 output projection 之前：$O=\operatorname{concat}_h(O_h)W_O+b_O$。每个 head 的下游 cotangent 包含 $W_O$ 及目标之后的全部导数，因而在 output projection 后满足 $E^{\mathrm{MHA}}=\sum_hE_h$。偏置没有直接输入微分项；Q/K/V 路径共享上游输入，允许在输入空间重叠。路径 norm 比例描述局部尺度，不能解释为互斥因果百分比。

对 residual $y=u+f(u)$，令 $g_y=\nabla_yF$。正确的两项为

$$
E^{\mathrm{id}}=\operatorname{diag}(x)J_{u,x}^\top g_y,\qquad
E^{\mathrm{branch}}=\operatorname{diag}(x)J_{u,x}^\top J_{f,u}^\top g_y.
$$

Identity 使用 residual 输出处的 $g_y$；直接使用已包含 branch 的 $\nabla_uF$ 会重复记账。相加恢复 Endpoint，但不产生 $F=F_{\mathrm{id}}+F_{\mathrm{branch}}$ 的函数值分解。



### 4.3 完整计算切面上的组合关系

对目标 $F$ 的切面边 $\{z_k\}$，定义

$$
E^C_{k,i}=x_i\left\langle g_{z_k},\frac{\partial z_k}{\partial x_i}\right\rangle.
$$

**命题 2（合法计算切面的闭合）。** 若每条相关 input-to-target differential path 恰好穿过切面一次，各成分使用同一前向状态，cotangent 位于相应真实使用边，且没有遗漏 bypass 或重复路径，则 $\sum_kE^C_{k,i}=x_i\partial F/\partial x_i$。多输入算子的输入边分别计入；同一张量被多次使用时按计算边而非张量名区分。

**证明。** 展开链式法则为各输入到目标路径的 Jacobian 乘积。合法切面把路径不重不漏地分组；每组可因式分解为到达 $z_k$ 的上游 Jacobian 与该边的下游 cotangent。对所有组求和，再乘输入坐标 $x_i$，即得结论。只切 $f(u)$ 会遗漏 $y=u+f(u)$ 的 identity bypass；以完整 $\nabla_uF$ 作为 identity 项后再加入 branch 项则会重复分支。分支内部的 cut 只重构该分支，完整 residual 或目标还需纳入其余路径。

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

### 5.2 可视化设置

概念和层间图使用主体可辨的懒熊样本，方法对比另外使用螳螂和小蓝鹭。样本按展示可读性选择，总体结论仍来自完整样本。为对应学习图像与前向特征，每层按 patch activation 的空间方差选择通道，并将该通道的 energy 定义为标量目标。图中同时给出通道编号及实际 14×14 前向 feature map；不同层可以选择不同通道。

主学习图像采用深蓝至青、黄、红的正部视图。设 $P=\max(\sum_cE_c,0)$，显示值为 $\sqrt{\min(P/q_{.99},1)}$，其中 $q_{.99}$ 在正值像素上计算。该变换提高弱细节的可见性，并保持像素位置；它不参与任何统计。补充图件同时显示 signed 与 absolute 视图，并另列 patch-max 加双线性插值的显示变体，明确区分原生细粒度图与平滑区域图。

### 5.3 统计与数值验证

所有总体推断先在图内聚合，以图像作为统计单位，使用 10,000 次 bootstrap。成分相关规则的 0.10 与区域增益规则的 0.05 是点估计门槛；符号检验检验相对零的方向，采用原 28 项 Holm 校正，CI 精度上限分别为 0.30/0.50。单元层深统计按模型使用五项校正；后续 residual spatial-TV 分析单独标为 post-hoc。

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

### 5.4 四组实验与证据层级

实验按四个研究问题组织。第一组比较原始 NNS 逐层传播与自动微分 Input × Gradient，检验 RQ1 的数学身份；第二组检验 Q+K+V、routing+value、各 head 以及 residual identity+branch 的结构闭合，对应 RQ2。第三组以 entropy、own-token locality、normalized distance、路径 absolute mass、相对尺度和 Value-only error 分析 RQ3，并用随机初始化、目标类型与抽样敏感性作控制。第四组在真实前向变量上执行成分或区域替换，比较 clean-only Endpoint 与 reference-aware Taylor，对应 RQ4。前两组验证表示身份与结构正确性，第三组描述局部输入依赖规律，第四组检验指定替换操作下的有限效应；四类证据不相互替代。

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

### 6.3 RQ3：注意力头与 routing/value differential

Figure 7 将 attention 分为 Q、K、routing 和 value，使分配权重的变化与内容传递可以分别观察。全部 head atlas 与跨模型类别图移至补充图件。

![Figure 7](../publication_r2_20260920/figures/png/fig_15_routing_value_examples.png)

**Figure 7.** 同一懒熊输入在浅层和深层的 Q、K、routing 与 value 路径。蓝底保留正部细节；signed closure 和绝对质量来自完整原始数组。

在 400 张图的目标条件化分析中，routing scale 从 block 0 的 0.3092/0.2041 增至 block 11 的 0.5154/0.4808；Value-only Endpoint error 从 0.4407/0.2774 增至 0.8022/0.8488（Figure 8）。ViT-B/16 的 routing/value/full absolute mass 中位数从 block 0 的 2.6596/7.6041/9.8604 变为 block 11 的 40.7842/39.0241/54.6768；DeiT-S/16 对应为 4.5294/18.4261/20.1427 与 35.9257/38.8767/44.5653。相应 JVP discrepancy 达到 0.9516/0.9636，完整响应 norm 为 14.9711/1.5155，绝对差异为 13.8064/1.4228。比例变化与绝对范数共同表明，固定 attention 权重的 $A\mathrm dV$ 在深层更难近似完整局部响应；这些量不是前向输出或因果贡献的百分比。

![Figure 8](../publication_r2_20260920/figures/png/fig_16_attention_depth.png)

**Figure 8.** 完整样本的图像级层深轨迹，阴影为 95% bootstrap 区间。空间分析每模型 100 图，attention/residual 每模型 400 图；推断沿用预先指定的分析。

控制 activation norm 与下游 cotangent norm 后，mean-query saturation 与 routing scale 的偏秩相关仍为 −0.1442/−0.3257。Sharper attention 不能直接作为更强 routing dependence 的代理；CLS-query 结果较弱，完整敏感性分析见补充材料。

### 6.4 RQ3：Residual identity/branch differential

Figure 9 以同一输入和目标展示 attention 与 MLP residual。两种结构均为 identity 加 branch，但 identity 的输入边不同；其局部依赖也呈现不同的层深趋势。Attention residual 的 late-minus-early identity scale 为 −0.1242/−0.0449，MLP residual 为 +0.1798/+0.1780（Figure 10）。

![Figure 9](../publication_r2_20260920/figures/png/fig_17_residual_examples.png)

**Figure 9.** 同一 block 5 中 attention 与 MLP residual 对同一类别目标的两条路径。两行的 identity 指向不同 residual 输入边。正部图不可直接相加，完整 signed maps 满足闭合。

![Figure 10](../publication_r2_20260920/figures/png/fig_18_residual_depth.png)

**Figure 10.** 完整样本的图像级层深轨迹，阴影为 95% bootstrap 区间。空间分析每模型 100 图，attention/residual 每模型 400 图；推断沿用预先指定的分析。

Attention residual 的 identity/branch absolute mass 在 ViT-B/16 中由 block 0 的 495.2271/63.8344 变为 block 11 的 313.3689/398.0202，在 DeiT-S/16 中由 459.3077/63.9204 变为 372.6598/185.3893。MLP residual 对应从 130.4998/438.0538 变为 827.8407/460.2701，以及从 170.6143/412.5634 变为 614.5721/354.6503。后续全样本空间分析得到，attention output–identity patch TV 的 late-minus-early 变化为 +0.0520/+0.0282，MLP 为 −0.0187/−0.0462，四项通过单独 Holm 校正。这些结果描述两个训练模型中不同 residual 子层的局部传递方式。Token-channel 层间分析与类别路径使用不同目标，因此二者提供互补的模型级观察。

### 6.5 表示对象的可视化比较

补充图件比较同一输入上的 Endpoint、Grad-CAM、attention rollout 和 raw attention。Endpoint 与 Grad-CAM 均使用 GT logit；Grad-CAM 在最后一个 block 的 norm1 输出处计算，避免最终 CLS 读出使末端 patch 梯度退化。Attention rollout 使用 head 均值、identity residual、逐行归一化及逐层矩阵相乘，raw attention 使用末层 CLS-to-patch head 均值。

Endpoint 保留像素尺度的局部项，Grad-CAM 和 attention 类方法提供较粗的区域分布。Patch-max 插值版本表明，聚合与插值本身就能显著改变学习图像的外观；它不改变基础定义。图中各方法有时突出不同区域，这反映目标与传播规则的差别。该比较用于说明表示粒度与对象，方法排序仍需要对应任务的定量评价。

### 6.6 RQ3 控制：训练与初始化

在每模型 32 张共享图像、全部 12 blocks 和 3 个随机初始化状态上，随机模型也表现出空间扩展。训练模型 entropy 的 late−early 为 0.2288/0.2256，随机均值为 0.0940/0.1718；训练−随机差分别为 0.1348 [0.1232, 0.1466] 和 0.0539 [0.0473, 0.0602]。区间同时重采样模型种子和共享图像。Routing scale 的 trained/random 层深变化为 0.0512/−0.0479 与 0.1314/−0.0043；attention identity 在训练模型中减少、随机模型中增加。MLP identity 在两类状态中均增加，ViT-B 的增量差区间跨零。训练参数状态与随机参数状态在空间扩展幅度和部分路径趋势上存在上述差异；该对照不识别训练过程的因果效应。

### 6.7 RQ3 控制：目标稳健性

在每模型 100 张冻结图像的 blocks 0、5、11 上，GT、clean-top1 与 top1 margin 均保留 routing/Value-only error 增加、attention identity 减少和 MLP identity 增加的方向。ViT-B 与 DeiT-S 的正确/错误图数量为 92/8 和 86/14，两个子集分别报告。Margin 改变效应幅度，例如 ViT-B routing 的 B11−B0 从 GT 的 0.2031 变为 0.1439。Native attention-output JVP 不依赖类别目标；另外报告目标 cotangent 收缩的方向 discrepancy。

### 6.8 前向 feature 与 Endpoint 的定量关联

在每模型 32 图、12 blocks、每层 16 个冻结随机 channels 上，12,288 个 patch-energy 目标给出前向平方响应分布与 Endpoint absolute patch mass。正确配对的图像级 Spearman 均值为 0.2663/0.2811，均高于同层错误 channel、跨层同 channel、空间置换和随机 channel 控制；其配对差区间均为正。Cosine、Jensen–Shannon distance 和 top-20 overlap 得到对应的定量比较。前向响应与局部输入依赖具有同通道关联，仍为互补对象。

### 6.9 RQ3 控制：抽样敏感性

复用相同 32 图的原 M3 单元记录，channel/token-stratum/image bootstrap 以及 leave-one-block/stratum-out 均保留 entropy 增大、locality 减小与距离增大的方向。该结果描述既有样本的抽样敏感性。全部实验设置、逐项指标、区间及五张补充图见[补充实验 v1](../publication_r2_20260920/zh/supplement_experiments_v1.md)。

## 7. 规定有限替换任务中的排序能力

### 7.1 RQ4：规定成分替换与区域选择

在成分替换实验中，12 个模型–reference–成分族组合有 11 个满足原复合条件。Table 4 直接列出相关估计、区间和阈值关系。ViT-B normalized-zero routing/value 为 0.0996，DeiT-S 对应估计为 0.1065，两个区间均跨越 0.10。因此 family 条件是否满足与总体效应是否超过 0.10 是不同判断；全部成分族的联合条件未满足。

**Table 4. 成分关联估计及区间**

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

补充同层 head 比较中，Taylor 具有更高 Spearman 与 Capture、更低 Regret（Table 5）。跨层 residual identity 的 Endpoint−Taylor Spearman 差为 0.0865–0.4209，但 Capture@1 中位差为零，表明增量价值依赖任务和指标。

**Table 5. 同层 head 排序：E=Endpoint，T=Taylor**

| Model | Reference | E rho | T rho | E Capture@1 | T Capture@1 | E Capture@3 | T Capture@3 | E Regret@1 | T Regret@1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | 0.2815 | 0.6742 | 0.5174 | 0.8038 | 0.6149 | 0.8567 | 0.1701 | 0.0571 |
| ViT-B/16 | blur | 0.2727 | 0.6856 | 0.5075 | 0.8203 | 0.6185 | 0.8659 | 0.1163 | 0.0346 |
| DeiT-S/16 | normalized-zero | 0.1952 | 0.6095 | 0.5609 | 0.8028 | 0.7122 | 0.8894 | 0.2381 | 0.0860 |
| DeiT-S/16 | blur | 0.1476 | 0.6238 | 0.5506 | 0.8426 | 0.7105 | 0.8970 | 0.1326 | 0.0443 |

区域实验固定 block 5 的三个 heads，所有方法共用 4×4 候选区域及最高 25% 面积。Figure 11 和 Tables 6–8 给出配对差、绝对 AUC 与归一化 regret。Endpoint−regional Taylor 的配对中位差为 −0.023433、−0.023697、−0.029900、−0.035630；四个 family 都未满足相对全部基线的 +0.05 条件。候选 oracle 仅取实际评价集合的最优效果，不能视为全部可能区域的全局最优。

![Figure 11](../publication_r2_20260920/figures/png/fig_21_intervention_boundary.png)

**Figure 11.** 左侧示例的有限区域替换曲线与右侧完整 400 图 family 的绝对 AUC。候选 oracle 只在已评价区域集合中取最优；数学闭合与区域干预选择属于不同性质。

**Table 6. 空间干预配对差**

| Model | Reference | vs total G×I | vs gradient | vs Taylor | vs random | Composite criterion |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | -0.0009 | -0.0000 | -0.0234 | -0.0023 | Unmet |
| ViT-B/16 | blur | -0.0010 | -0.0007 | -0.0237 | -0.0031 | Unmet |
| DeiT-S/16 | normalized-zero | -0.0018 | -0.0006 | -0.0299 | 0.0009 | Unmet |
| DeiT-S/16 | blur | 0.0002 | -0.0002 | -0.0356 | -0.0010 | Unmet |

**Table 7. 绝对 AUC 与输出尺度**

| Model | Reference | Endpoint AUC | Taylor AUC | Random AUC | Candidate oracle AUC | Clean GT logit |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | 0.0309 | 0.0558 | 0.0339 | 0.0654 | 8.8989 |
| ViT-B/16 | blur | 0.0276 | 0.0518 | 0.0307 | 0.0747 | 8.8989 |
| DeiT-S/16 | normalized-zero | 0.0617 | 0.0961 | 0.0611 | 0.1474 | 8.7974 |
| DeiT-S/16 | blur | 0.0395 | 0.0770 | 0.0411 | 0.1063 | 8.7974 |

**Table 8. 候选 oracle regret 与 logit 尺度比**

| Model | Reference | Endpoint regret | Taylor regret | Random regret | Endpoint AUC/abs(logit) | Taylor AUC/abs(logit) |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | 0.5303 | 0.1442 | 0.5059 | 0.0035 | 0.0062 |
| ViT-B/16 | blur | 0.6366 | 0.2557 | 0.5929 | 0.0031 | 0.0059 |
| DeiT-S/16 | normalized-zero | 0.5646 | 0.3352 | 0.5947 | 0.0072 | 0.0107 |
| DeiT-S/16 | blur | 0.6289 | 0.2705 | 0.6153 | 0.0045 | 0.0089 |

修正随机基线后，Endpoint−random 的配对中位差为 −0.0023、−0.0031、0.0009、−0.0010；DeiT-S 两个区间跨零。Taylor−random 的四个区间均为正。原 block-5 三-head 结果对应确认性协议；补充跨层全-head M2 覆盖 blocks 2/5/10，在完整输入替换的 11/12 family 与单-head 隔离替换的 12/12 family 显著偏向 Taylor，详见补充实验 S14。

跨 reference 分析补充了这一结果的含义。Endpoint 使用观测输入处的 clean map，而 Taylor 显式使用替换位移。对 163,200 个成分配对，Taylor magnitude 与 absolute effect 在原 atol=1e-12、rtol=0 标准下均发生变化；分布及 head 排名一致性见补充表。Reference 会改变干预相关排序，局部输入图与位移估计器因而适用于不同问题。

## 8. 讨论与结论

### 8.1 讨论

本文的唯一定位是以 Gradient × Input 为局部测量原语，对 ViT 异构内部目标进行结构化 Jacobian 扫描，并在指定真实计算边上实现可闭合、可审计的 differential accounting。命题 1 界定数学身份，命题 2 界定指定图与切面上的闭合条件；机制结论来自两个固定 ImageNet 预训练 checkpoint 及其冻结目标样本。

结构闭合、局部路径尺度和有限替换排序对应三个不同问题。Q/K/V、multi-head 与 identity/branch 的加和闭合说明各项重构指定目标的局部微分；routing 或 identity scale 比较路径图的相对范数；activation replacement 测量指定 reference 和替换算子下的有限效应。本文的 RQ4 只评价完整替换与固定面积预算下的成分排序和区域选择，不估计 Taylor 误差随干预幅度变化的边界。

学习图像使不同内部对象能够在同一输入网格上观察。明确的目标、层与通道支持比较前向响应和局部输入依赖，以及注意力与残差路径如何组合。正确 feature–Endpoint 配对优于四种负控制，但二者的相关程度远低于等价；高分辨率输入图也受 $16\times16$ patch embedding 与归一化坐标原点约束。

蓝底高亮视图便于读取分散纹理，但只显示 signed RGB-sum 的正部。补充图件将其与 signed 和 absolute 视图并列；正部表示局部坐标缩放项的符号，不等同于有限删除时的预测支持，较大 absolute mass 也可能伴随较强抵消。

训练控制覆盖每架构三个初始化种子；经验范围尚不包括同架构不同训练 checkpoint、独立 channel seeds、同目标 M3–M4 桥接或 cascading randomization。错误分类子集含 8/14 图，空间干预覆盖既定区域、references 与 blocks 2/5/10。ViT-B Path IG 在既定 512 点范围内未达到共同校准要求，跨模型 Endpoint–Path 比较位于本文结果范围之外。

### 8.2 结论

本文得到四点结论。第一，使用精确当前点局部导数且无额外重分配时，NNS 标量学习图像等于 Input × Gradient，因而 Gradient × Input 是框架的基础测量原语。第二，本文在指定 ViT 计算图上建立异构内部目标 registry 和真实边 differential cuts；attention 的 Q-induced、K-induced 与 V-induced 路径以及 residual 的 identity 与 branch 路径在合法切面上加和恢复完整局部输入依赖，分项随图、切面和参数化而变。第三，在 ViT-B/16 与 DeiT-S/16 两个固定 checkpoint 中，冻结抽样目标总体呈空间扩展，深层 attention 的 routing 相对尺度和 Value-only 误差提高，attention 与 MLP residual 呈不同趋势；训练参数状态与随机参数状态存在报告的差异。第四，局部路径图在规定成分替换任务中与有限效应具有广泛但非普遍的关联；给定 reference 的规定区域替换更适合使用显式位移的 Taylor 排序。本文由此形成结构化局部 Jacobian 扫描、可验证记账和两模型机制研究的统一框架。

## 参考文献与补充材料

[参考文献](../publication_r2_20260920/references.md)；[补充材料](../publication_r2_20260920/zh/supplement.md)。作者、单位及最终归档信息待补充：[REPOSITORY AND ARCHIVE REQUIRED]。
