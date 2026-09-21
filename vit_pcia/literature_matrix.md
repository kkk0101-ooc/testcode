# Endpoint–PCIA–PEFA 碰撞查新矩阵

- 状态：`FROZEN_STEP1_20260921`
- 检索截止：2026-09-21（PRC）
- 用途：决定候选对象是否有资格进入 CPU 参考实现；本文件不是专利法律意见，也不把未检出等同于全球首创。

## 1. 检索协议

本轮在接触任何 PCIA/PEFA 数值结果前完成。检索源包括 arXiv 标题/摘要/HTML/PDF、ACL Anthology、PMLR、出版社页面和官方方法页；优先以原论文公式与方法描述作判断，二级综述只用于扩展关键词。核心检索式按下列簇组合使用：

- `integrated gradients`, `conductance`, `neuron integrated gradients`, `internal influence`；
- `integrated Hessians`, `higher-order attribution`, `meta-attribution`, `attribution of explanations`；
- `DeepLIFT`, `layer-wise relevance propagation`, `Shapley Flow`, `path-wise/path-specific attribution`；
- `ACDC`, `attribution patching`, `EAP`, `EAP-IG`, `AtP*`, `second-order attribution patching`；
- `transformer information flow`, `circuit attribution`, `attribution graph`, `DPA`, `DePass`, `LIG`, `IG-Lens`, `Unpack`；
- 截止日近邻 `DAVE`, `HETA`, `Sparse Inter-Layer Dependencies`, `PA-LRP`, `GradSkip`, `HiLRP`, `ResLRP`, `CA-LIG`, `DAP`, `ICAT`, `Contribution Weights`, `CAAP`, `LibraGrad`, `LeGrad`, `GMAR`；
- 精确短语 `Port-resolved Endpoint Formation Attribution`, `Endpoint Formation Attribution`, `routing curvature attribution`, `antisymmetric Hessian attribution`。

另以 arXiv API 按提交日倒序枚举 `attribution AND transformer`、`vision transformer AND attribution`、`circuit attribution`、`integrated gradients AND transformer`、`Hessian AND attribution AND transformer`、`interlayer attribution`、`information flow AND transformer AND interpretability`、`relevance propagation AND vision transformer` 等查询，逐篇筛到 2026-09-21；[L42] 用作跨架构方法覆盖的独立 landscape cross-check，而非候选算子。最后对项目既有 EP0 `literature_registry.json` 做差集复核，将原 NNS-CNN 与 LibraGrad 回并当前矩阵，并以 LeGrad、GMAR 补齐同一 ViT 梯度/attention 邻域。精确短语未返回同名方法。检索不能证明不存在未索引、非英语、未公开或采用不同术语的工作；后续写作只能使用“在冻结检索范围内未发现等价合同”，不得使用“首次”或“全球首创”。

## 2. 比较字段与判定口径

每项都按同一组字段判断。`真实使用端口` 指 producer tensor 到特定 consumer operator 的特定 input slot 和 invocation，而非仅按层名或隐藏单元名聚合。`合法切面` 指每条有效根到目标路径恰好相交一次。`实现不变性` 分为输出级与内部结构级；任何依赖具体图、端口或替换模型的分解都不声称内部实现不变。`因果边界` 区分梯度/守恒记账与真实 do/patch/ablation 效应。

缩写：Y=明确具备，N=不具备，P=部分或仅在特定构造下具备。

## 3. 逐字段矩阵

| 方法 | 被解释对象 | 索引结构 | 守恒终点 | 真实使用端口 | 合法切面 | 二阶项 | 路径/参考依赖 | 实现不变性 | Q/K/V–head–residual–MLP | 因果边界 | 与本合同的关系 |
|---|---|---|---|---:|---:|---:|---|---|---|---|---|
| Gradient × Input / Endpoint [L43] | 终点局部响应 $E_i^b=(x_i-b_i)\partial_iF(x)$；原 NNS-CNN 还沿 neuron/filter 传播并聚合神经元图 | $i$；NNS-CNN 含 neuron/filter | 一般不守恒到 $F(x)-F(b)$ | N | N | N | 依赖坐标原点 $b$，无路径 | 对同一函数的总量 Y；坐标变换一般 N | 需另做结构分解 | 关联量，非干预效应 | 已知起点；Endpoint 等于 Gradient × Input，且继承 NNS-CNN，不是新基础量 |
| IG [L1] | 输出差 | $i$ | $\sum_i\mathrm{IG}_i=F(x)-F(b)$ | N | N | N | Y | 总量 Y | N | 非因果 | $\sum_e A_{e,i}$ 的终点 |
| Conductance / Neuron IG [L2,L3] | 输入信用经隐藏单元的流 | 隐单元 $y$ × 输入 $i$ | 对完整隐藏层求和恢复 IG/输出差 | 通常 N，按隐藏单元聚合 | 层切面 P | N | Y | 总量可 Y；内部单元 N | 不原生区分 consumer slot、fan-out、完整 ViT 切面 | 非因果 | **PCIA 的最近邻；标量单元情形公式等价** |
| Internal Influence [L4] | 输出对内部表示元素的平均敏感度 | 内部元素 $j$ | 无输出差完整性要求 | N | N | N | 依赖所选分布；路径分布可作为实例 | 内部表示 N | N | 非因果；论文中的 influence 是梯度期望 | 与 $D_e$ 的“内部敏感度”思想接近，不含输入来源或 Endpoint 形成 |
| Integrated Hessians [L5] | IG 解释/输入特征交互 | $i,j$ | $\sum_{ij}\Gamma_{ij}=F(x)-F(b)$ | N | N | Y | Y，嵌套 IG | 总量 Y | N | 非因果 | **$T$ 的最近二阶邻居；目标和权重不同** |
| Higher-Order Feature Attribution [L6] | 递归/高阶归因 | 多特征元组 | 随所选归因算子 | N | N | Y | 通常 Y | 取决于基础算子 | N | 非因果 | 覆盖“高阶归因”类别，阻止把二阶运算本身称新 |
| Metagame directional meta-attribution [L7] | 特征 $j$ 对解释 $\phi_i(f)$ 的方向性影响 | $j\to i$ | 分层分解基础解释 | N | N | 不要求 Hessian；Shapley 元归因 | 依赖 coalition/reference 语义 | 基础解释与博弈定义相关 | N | 非因果 | **覆盖“解释一个解释”和 $j\to i$ 语义；输入级 $T$ 属普通元归因实例** |
| DeepLIFT [L8] | 相对参考激活差 | 输入/节点 | summation-to-delta | 传播节点 P | 网络级规则，不是合法边切面 | N | Y | N，依赖实现与传播规则 | 可适配但非原生完整合同 | 非因果 | 有参考守恒，但不是 Hessian 路径项 |
| LRP / AttnLRP [L9,L10] | 输出 relevance 的逐层重分配 | 输入/层/单元 | 规则内 relevance conservation | 计算节点或模块 P | P，不等于真实使用端口切面 | N | 通常无连续输入路径；参考依规则 | N | AttnLRP 覆盖 attention 与 latent representations，具体规则依架构 | 非因果 | 守恒传播近邻；不等价于原生 Jacobian 的 $T/R/\Omega$ |
| Shapley Flow [L11] | 因果 DAG 的边信用 | 图边 | boundary consistency / efficiency | 图边 Y，但不是自动微分 use-port 身份 | 因果边界 Y，定义不同 | N | 依赖背景/博弈历史 | 对给定因果图 P | 不专指 ViT 算子 | 依赖用户给定因果图，强于纯梯度但不等同神经网络干预 | 覆盖“边信用+边界守恒”，不覆盖 Endpoint/Hessian 合同 |
| Path-wise / path-specific attribution [L12-L14] | 指定因果路径、LSTM influence path 或多段表征映射 | 因果路径/门/latent path | 方法特定 | P | P | 通常 N | Y | N | 架构特定 | 可能含反事实/因果路径；须按论文区分 | 阻止把“沿内部路径归因”本身称新 |
| ACDC [L15] | 对任务指标重要的稀疏电路 | 模型边 | 无加法守恒终点 | Y | 不要求一次相交切面 | N | clean/corrupted 与 patch reference | N | Transformer 组件边 | 以 activation patching/ablation 近似因果重要性 | 目标是发现电路，不是 Endpoint 形成张量 |
| EAP [L16] | 单边 patch 对指标的局部 Taylor 影响 | 边 | 不保证完整性 | Y | N | N | clean/corrupted Y | N | Transformer 边 | 干预效应的一阶近似，不是干预本身 | 与端口索引接近，但无 $i\leftarrow j$ 和 Endpoint 守恒 |
| EAP-IG [L17] | clean/corrupted 边位移对指标的路径梯度近似 | 边 | 不保证跨边完整性 | Y | N | 路径梯度但非本合同二阶 Endpoint 项 | Y | N | Transformer 边 | patch 效应近似 | 与 $A_e$ 都是边/路径梯度，但目标、路径空间和守恒终点不同 |
| AtP* [L18] | 组件 patch 影响的快速近似 | 组件/边 | 无 | Y/P | N | N；含 Q/K 修正和 GradDrop | clean/corrupted Y | N | 显式处理 Transformer attention 失效模式 | patch 近似 | 强调一阶评分会漏项，不覆盖 PEFA 定义 |
| 二阶 attribution-patching 修正 [L19] | 一阶 patch 近似误差 | patch 方向/HVP | 对 patch Taylor 误差给界或修正 | Y/P | N | Y | clean/corrupted patch path | N | 多类 Transformer | 仍是干预近似的二阶修正 | 是 $R$ 的重要近邻，但解释对象和权重不同；不能把 HVP/曲率称新 |
| Circuit Tracing / attribution graphs [L20] | 替换模型中的 feature-to-feature 线性效应 | feature graph edge | 替换图中加法记账 P | Y，CLT feature edge | 图追踪规则，不是原模型合法切面 | 通常 N | 输入实例依赖；非 $b\to x$ 路径 | N，依赖 CLT 替换模型 | residual/MLP feature graph；非原生完整 ViT Jacobian | 需另行 patch 验证 | 覆盖跨层 attribution graph，但不是原图端口 Hessian |
| DePass [L21] | 固定 attention/MLP 激活下的可加表示贡献 | token/组件/子空间 | 修改前向中的可加分解 | 组件 P | P | N | 非输入路径积分 | N，固定内部状态 | attention、MLP、head 等多粒度 | 非干预，需 fidelity 验证 | 覆盖细粒度 flow，但算子与守恒对象不同 |
| LIG / IG-Lens [L22,L23] | 层内 set-to-set flow；跨层概率变化 | 层/模块/集合 | 层内一致性或概率差 telescoping | 模块 P | P | N | Y | 内部级 N | ATT/MLP 或层级 | 非因果 | 覆盖 Transformer 层级 IG，不含 Endpoint 坐标或端口曲率 |
| DPA（Dual Path Attribution）[L24] | SwiGLU Transformer 的 content/control 信息流 | 层/组件/路径 | 其冻结层级传播下的精确分解 | 组件级 P | 非本合同合法切面 | 解析非线性交互，但不是输入 Hessian 张量 | 无 counterfactual；输入实例依赖 | N，架构与传播构造相关 | attention head、MLP neuron；SwiGLU 特定 | 非干预评分，论文另做验证 | 是最新强近邻；不保留 RGB $i,j$、Endpoint 守恒或 $\Omega$ 零和 |
| Unpack [L25] | attention/MLP key-value 组件、命名路径与 token 信用 | 组件对/路径/token | 单次分解内的读出一致性 | 组件连接 P | 非 use-port cut 合同 | 解析组合交互，非 Hessian 路径积分 | 无 $b\to x$ 路径 | N，依赖 key-value 模板 | K/Q/V composition、attention、MLP | 非干预，另以 ablation 检查 | 强近邻，覆盖命名端到端路径，但不等价于 $T/R/\Omega$ |
| GAF / attention flow [L26,L27] | attention 图上的 token flow | token-layer edge | 最大流/rollout 口径 | attention edge P | 网络流切面而非原图 Jacobian cut | N | 通常无输入参考路径 | N | 主要 attention，不完整覆盖 MLP 真实导数 | 非因果 | “flow”语义碰撞，不是本合同微分对象 |
| DAG-SHAP [L28] | 用户 DAG 中逐边干预的特征信用 | causal edge | Shapley 型效率 | 因果图边 Y，非神经图 use port | 因果 DAG 边界 P | N | 背景/coalition 依赖 | 对给定 DAG P | 非 ViT 特定 | 明确 edge intervention | 2026 边级新近邻；不覆盖 Endpoint 形成或端口微分曲率 |
| DAVE [L29] | ViT 输入梯度中局部等变、稳定的有效变换 | RGB 像素/通道 | 无输出差或 Endpoint 完整性 | N，内部算子只用于推导后聚合掉 | N | N；局部扰动期望不是本合同 Hessian 张量 | 依赖局部变换分布与平滑噪声，无 $b\to x$ 路径 | 依赖 ViT 有效变换分解，内部级 N | 解析 attention routing 与算子变化，但不保留端口索引 | 关联量，非干预效应 | 直接覆盖 ViT 梯度分解/稳定输入归因；不覆盖 $e\times i\times j$、Endpoint 守恒或 $\Omega$ |
| HETA [L30] | 自回归目标 token 的输入 token 重要性 | token $i\to T$ | semantic gate 为 simplex；最终混合分数无精确输出差或 Endpoint 守恒 | N | N | Y，embedding Hessian block/HVP sensitivity | 目标条件与 mask reference；非 $b\to x$ 路径积分 | N，依赖 decoder、attention flow 与组合权重 | attention–value flow；不分解完整 Q/K/V–residual–MLP use ports | 含真实 token masking，但最终分数混合梯度与干预信号 | 直接阻止把“Transformer + Hessian + flow”称新；其 Hessian 分数不等价于端口分辨 $R/\Omega$ |
| Sparse Inter-Layer Dependencies [L31] | 上游 neuron/attention component 对目标 FFN pre-activation 的影响 | component $s\to$ neuron $i$ | 目标 pre-activation 的加性表示，relevance 排名不守恒到模型输出 | 组件到目标 neuron P，未定义精确 consumer-slot/invocation ID | N | N | 依赖样本分布方差与 mean-ablation reference | N，依赖 residual-stream 表示与 FFN 基 | attention output、FFN neuron、residual contribution | 用 mean-ablation activation patching 验证功能充分性 | 强 component-to-neuron/circuit 近邻；不含 RGB $i,j$、路径 IG、Endpoint 缺口或反对称零和 |
| HiLRP [L32] | 跨 ViT 架构的 prediction relevance | 输入像素/primitive/stage | primitive 规则逐步 relevance conservation | 运算 primitive P，未冻结真实 use-port identity | 传播边界 P，不是本合同合法 Jacobian 切面 | N | 无连续输入参考路径；规则依稳定化选择 | N，依赖 primitive 分解与传播规则 | 显式覆盖 Q/K/V、双线性 attention、归一化、重索引与多级结构 | 非干预；另与 sampled Shapley 比较 | 最新 conservation 强近邻；守恒对象/算子不是 Endpoint 形成、$R$ 或 $\Omega$ |
| ResLRP [L33] | ViT residual cancellation 下的 prediction relevance | residual branch/输入像素 | residual addition 局部精确守恒；完整流水线取决于后端规则 | residual branch P，未保留 consumer-slot/invocation ID | P，不是本合同合法切面 | N | 无 $b\to x$ 路径；依赖 LRP 传播规则 | N | 显式处理 attention/MLP residual branch | 用 channel intervention/遮挡评估，但 relevance 本身非干预效应 | 最新 residual 强近邻；不含输入来源 $j\to i$、加权 Hessian $R$ 或 $\Omega$ 零和 |
| PA-LRP [L34] | token semantic 与 positional relevance | position–token pair/PE layer | 专门规则恢复 PE relevance conservation | 模块/PE 分量 P，不是运行时 use-port ID | P，不是原图合法 Jacobian 切面 | N | 无连续输入路径；依赖 LRP rule/stabilizer | N，依赖 PE 分解和传播实现 | 显式覆盖输入级/attention 级 PE，并沿用 Transformer LRP | 非干预 | 补齐 positional root 的强近邻；不保留 RGB $i\leftarrow j$ 或 Endpoint 曲率合同 |
| LRP–LOO implementation-invariance audit [L35] | Transformer LRP 与 leave-one-out feature importance 的一致性 | 输入 feature/attention factor | LRP 局部守恒，但不保证 LOO 等价 | N | N | N | LOO reference 与传播规则依赖 | 明确证明 AttnLRP 双线性规则可违反实现不变性 | attention bilinear/softmax，非完整端口族 | LOO 是干预比较；LRP 本身非干预 | 直接支持本合同“不声明内部实现不变”；不覆盖 $T/R/\Omega$ |
| GradSkip [L36] | ViT patch-token relevance | token/head/residual path | 归一化后的跨层近似 relevance conservation | head/residual component P，非 consumer-slot/invocation ID | N | N | 无 $b\to x$ 路径；依赖梯度阈值、Gini 与传播归一化 | N | 显式 head 加权与 attention/skip 分流；不完整分解 MLP/Q/K/V | deletion/insertion 用于评价，归因量本身非干预 | 强 head/residual flow 近邻；无 Endpoint 守恒、来源坐标或反对称项 |
| CA-LIG [L37] | 每层 hidden token relevance 与 attention-gradient context | layer/token/attention pair | 每层基础 IG 有 completeness；融合/rollout 非本合同终点 | 模块 P，非真实 use-port | P，层级边界不是合法原图切面 | N | 各层 activation baseline/path Y | N，依赖层表示与融合规则 | attention heads 与层级 flow；residual/MLP 仅隐式 | 非因果 | 覆盖“Transformer 内部对象做 IG + 跨层融合”，进一步限制 $T$ 的新颖性；不等价于 $R/\Omega$ |
| DAP [L38] | 类别条件的 ViT token attention flow | token-to-token transition | 行归一化传播，不守恒到输出差/Endpoint | attention transition P，非自动微分 use-port | N | N | 无输入路径；依赖 Grad-CAM prior | N | attention + identity residual；不覆盖 MLP 或真实 Q/K/V 导数 | 非因果 | 覆盖 decision-aware attention propagation；不含 Endpoint 形成或端口 Hessian |
| iERF / SRD / ICAT [L39] | 像素证据、PFV 与跨层 concept influence | pixel–PFV / concept $a\to b$ | SRD 局部重分配；ICAT 对 concept-alignment score 归因 | 层/概念 P，非运行时 consumer use | 隔离子网络 P，不是全模型合法切面 | ICAT 可用 IG，非二阶 | concept coefficient baseline/path Y | N，依赖 PFV/SAE/concept basis | ViT 通过 token/PFV 与 AttnLRP；非完整 Q/K/V use-port 合同 | 用 concept insertion/deletion 验证，但 ICAT score 非干预本身 | 强 interlayer concept-IG/meta 近邻；不含 RGB Endpoint 缺口或 $\Omega$ |
| Contribution Weights [L40] | self-attention token/head 对局部输出向量的几何贡献 | source token × head × target token | 投影权重在单个 attention 输出内有加性归一化 | attention contribution P，非下游 consumer identity | 局部 attention 和，不是根到目标切面 | N | 无参考路径 | N，依赖 value basis 与 attention 实现 | 显式 token/value/head；不覆盖 residual/MLP | 另以 token/component removal 验证 | 覆盖注意力内部 signed contribution；不含输入坐标、Endpoint 守恒或曲率运输 |
| CAAP [L41] | 内部 contextualized patch representation 对预测的干预贡献 | image patch × layer range | 无加法守恒要求 | patch activation P，不是 consumer-slot/invocation use-port | N | N | source-to-neutral-target donor 与 layer-range 依赖 | N，依赖所选内部表征与 patch protocol | 跨层 residual patch；不分解 Q/K/V/head/MLP | **真实 activation patching 干预** | 冻结因果边界的强近邻；不等价于任何梯度 $T/R/\Omega$，但其干预可作以后评价参照 |
| LibraGrad / LeGrad / GMAR [L44-L46] | ViT 梯度流平衡、feature formation sensitivity 与梯度加权多头 attention rollout | pixel/token/layer/head | LibraGrad 以 FullGrad-completeness 为目标；LeGrad/GMAR 图不守恒到 Endpoint 或输出差 | 模块/head P，非运行时 consumer-slot/invocation | N | N | 无 $b\to x$ 连续路径；依赖当前样本梯度与所选层/head | N，依赖反向路径修正、层选择或 head 聚合 | 显式覆盖 attention layer/head；不完整覆盖 residual/MLP/Q/K/V use-port | 关联型 saliency，非端口干预效应 | 覆盖 Transformer 梯度流、feature formation 与 head 加权邻域；不含 $e\times i\times j$、Endpoint 缺口或 $\Omega$ 零和 |

## 4. 最近邻公式单列

### 4.1 Conductance 与 PCIA

对隐藏标量单元 $y$，conductance 的输入分辨形式为

\[
\operatorname{Cond}_{y,i}^{\gamma}
=\int_0^1 \dot\gamma_i(t)
\frac{\partial F}{\partial y}(\gamma(t))
\frac{\partial y}{\partial u_i}(\gamma(t))\,dt.
\]

若端口 $e$ 是该标量单元的唯一消费边，令
$D_{e,i}=(\partial F/\partial y)(\partial y/\partial u_i)$，则

\[
A_{e,i}^{\gamma}=\operatorname{Cond}_{y,i}^{\gamma}.
\]

向量端口时对端口坐标求和即可得到同一链式法则形式。PCIA 的差别是严格区分 consumer、input slot、invocation、fan-out，并要求合法切面；这属于图语义收紧和系统化，不是新的基础积分量。故 `PCIA_FOUNDATIONAL_NOVELTY=REJECTED`。

### 4.2 Integrated Hessians 与 PEFA

Integrated Hessians 可写为递归归因

\[
\Gamma_{i,j}^{\mathrm{IH}}(x;b)
=\operatorname{IG}_j\bigl(\operatorname{IG}_i(F;\cdot,b);x,b\bigr).
\]

对直线路径 $u=b+t\Delta$，$i\ne j$ 时其最近邻公式为

\[
\Gamma_{i,j}^{\mathrm{IH}}
=\Delta_i\Delta_j\int_0^1\!\int_0^1
\alpha\beta\,\partial_{ij}^2F(b+\alpha\beta\Delta)\,d\alpha\,d\beta,
\]

且对角项另含一阶导积分；全矩阵求和回到 $F(x)-F(b)$。本合同在完整切面求和后的 PEFA 为

\[
\sum_e T_{e,i\leftarrow j}^{\gamma}
=\operatorname{IG}_j(E_i^b; x,b),
\]

直线路径下

\[
\sum_e T_{e,i\leftarrow j}
=\delta_{ij}\Delta_i\int_0^1\partial_iF(b+t\Delta)dt
+\Delta_i\Delta_j\int_0^1t\,\partial_{ij}^2F(b+t\Delta)dt.
\]

因此两者都属于“解释一个解释”的二阶路径家族，但不相等：IH 解释 IG 并守恒到输出差；本式解释终点 $E_i^b$ 并逐 $i$ 守恒到 Endpoint。此差异支持新的组合合同，不支持宣称发明二阶或元归因。

## 5. 符号等价性审计

### 5.1 $T$

直接令 $G_{e,i}(u)=Q_{e,i}^b(u)$，则

\[
T_{e,i\leftarrow j}^{\gamma}
=\int_0^1\dot\gamma_j(t)\,\partial_jG_{e,i}(\gamma(t))dt,
\]

就是对标量解释对象 $G_{e,i}$ 做一般路径 IG。故：

- 对普通 IG：**操作级完全化约**；
- 对 meta-attribution：**类别级完全覆盖**；
- 对 Integrated Hessians：同属递归路径归因，但基础解释、积分权重和守恒终点不同，**不数值等价**；
- 对 EAP-IG：都含边/路径梯度，但 EAP-IG 积分 clean-corrupted 内部激活并近似 patch metric，**不等价**。
- 对 CA-LIG/ICAT：它们再次覆盖“内部表示或概念分数做 IG”，但路径位于 layer activation/concept coefficient 空间且终点不同，**强化类别覆盖而非本合同数值等价**。

结论：按执行规划的强制规则，$T$ 必须降级为组合框架，不得作为新基础算子。

### 5.2 $R$

乘积法则给出

\[
T_{e,i\leftarrow j}=\delta_{ij}A_{e,i}+R_{e,i\leftarrow j},
\qquad
R_{e,i\leftarrow j}=\int(\gamma_i-b_i)\dot\gamma_j\,\partial_jD_{e,i}\,dt.
\]

对完整切面求和并取直线路径，$R$ 化为加权 Hessian 路径积分；因此“使用二阶导数/HVP”本身已被 IH、高阶归因、HETA 和二阶 attribution-patching 覆盖。DAVE 与 LibraGrad 又覆盖了 ViT 输入梯度的结构分解或反向路径校正，LeGrad/GMAR 覆盖 layer/head 级 feature formation 或梯度加权 attention；PA-LRP、GradSkip、HiLRP、ResLRP 覆盖多类 ViT 内部守恒传播；CA-LIG/ICAT 覆盖内部对象上的 IG；Sparse Inter-Layer Dependencies 与 Contribution Weights 覆盖 component-to-neuron 或 attention 内部贡献；CAAP 覆盖真实内部 patch 干预。冻结检索范围内仍未发现同时满足以下全部条件的已有量：以 Endpoint–IG 的逐坐标差为守恒缺口、保留真实消费端口 $e$ 与有向来源 $j\to i$、并在原始模型合法切面上精确相加。$R$ 的候选增量只允许表述为这组联合约束和恒等式，不允许表述为新 Hessian、梯度分解、Transformer flow 或干预方法。

### 5.3 $\Omega$

\[
\Omega_{e,ij}=\partial_iD_{e,j}-\partial_jD_{e,i}.
\]

这是端口梯度通道的一般反对称 Jacobian/外微分；数学运算本身不是新发明。对 $C^2$ 标量目标和合法完整切面，

\[
\sum_{e\in C}\Omega_{e,ij}
=\partial_i\partial_jF-\partial_j\partial_iF=0.
\]

冻结检索范围内没有发现将该反对称量绑定到原模型真实消费端口、ViT 完整切面并检验“单端口非零、完整切面相消”的等价 attribution 合同。它与 DPA/Unpack/attribution graphs、ICAT、Sparse Inter-Layer Dependencies、Contribution Weights 的内部 flow 语义邻近，也与 PA-LRP/GradSkip/HiLRP/ResLRP 的 conservation 语义邻近，但这些方法未给出端口反对称导数的本零和量。该结论只是可证伪候选，不是优先权证明。

## 6. 冻结查新结论

```text
PCIA_FOUNDATIONAL_NOVELTY=REJECTED_CONDUCTANCE_EQUIVALENT_SPECIALIZATION
T_FOUNDATIONAL_NOVELTY=REJECTED_PATH_IG_OF_ENDPOINT_COMPONENT
R_OPERATOR_NOVELTY=REJECTED_WEIGHTED_HESSIAN_FAMILY
R_PORT_RESOLVED_IDENTITY_COVERAGE=NOT_FOUND_IN_FROZEN_SEARCH
OMEGA_MATHEMATICAL_OPERATOR_NOVELTY=REJECTED_STANDARD_ANTISYMMETRIC_DERIVATIVE
OMEGA_PORT_CUT_ATTRIBUTION_CONTRACT_COVERAGE=NOT_FOUND_IN_FROZEN_SEARCH
STEP1_NOVELTY_GATE=GO_STEP2_COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE
```

这满足“至少一个对象未被等价覆盖”的最低入口，仅因为 $R$ 的联合守恒合同与 $\Omega$ 的端口切面零和在本次检索中未找到完整等价物。项目定位冻结为 **Endpoint 上的端口分辨组合/系统化候选**。若后续发现 $R$ 与 $\Omega$ 的完整等价先例，必须签发 `NO_GO_FOUNDATIONAL_NOVELTY`；即使未发现，也必须经步骤 2–5 的数学、端口与实证门，才可讨论贡献强度。

## 7. 可证伪主张

| ID | 候选主张 | 直接否证条件 |
|---|---|---|
| C1 | 合法切面上 $\sum_jT_{e,i\leftarrow j}=Q_{e,i}^b(x)$，再对端口求和得到 $E_i^b(x)$ | CPU 解析 fixture 或原始 RGB 端口张量在冻结容差外不闭合 |
| C2 | $T=\delta A+R$，且完整切面 $\sum_{e,j}R_{e,i\leftarrow j}=E_i^b-\mathrm{IG}_i$ | 任一未聚合坐标的等式超容差，或只能展示聚合后成立 |
| C3 | 单端口可有 $\Omega_e\ne0$，但合法完整切面 $\sum_e\Omega_e=0$ | 解析反例不产生预期单端口非零，或完整切面零和超容差 |
| C4 | 真实端口 $e\times i\times j$ 比 conductance/IH/EAP-IG 提供可复核的额外结构信息 | 去掉端口身份后所有预注册读出和干预预测不变，或不优于简单基线 |
| C5 | $R/\Omega$ 排名能预测 Endpoint 变化或注册端口干预 | 预注册 pilot 中增量效用为 `NOT_SUPPORTED` 或 `INVALID` |

## 8. 一手来源

- [L1] Sundararajan, Taly, Yan. *Axiomatic Attribution for Deep Networks*. ICML 2017. https://arxiv.org/abs/1703.01365
- [L2] Dhamdhere, Sundararajan, Yan. *How Important Is a Neuron?* 2018. https://arxiv.org/abs/1805.12233
- [L3] Shrikumar et al. *Computationally Efficient Measures of Internal Neuron Importance*. 2018. https://arxiv.org/abs/1807.09946
- [L4] Leino et al. *Influence-Directed Explanations for Deep Convolutional Networks*. 2018. https://arxiv.org/abs/1802.03788
- [L5] Janizek, Sturmfels, Lee. *Explaining Explanations: Axiomatic Feature Interactions for Deep Networks*. 2020/2021. https://arxiv.org/abs/2002.04138
- [L6] Butler, Feng, Djuric. *Higher-Order Feature Attribution*. 2025. https://arxiv.org/abs/2510.06165
- [L7] Baniecki, Biecek, Fumagalli. *Attributions All the Way Down? The Metagame of Interpretability*. 2026. https://arxiv.org/abs/2605.06295
- [L8] Shrikumar, Greenside, Kundaje. *Learning Important Features Through Propagating Activation Differences*. 2017. https://arxiv.org/abs/1704.02685
- [L9] Bach et al. *On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation*. 2015. https://doi.org/10.1371/journal.pone.0130140
- [L10] Achtibat et al. *AttnLRP*. 2024. https://arxiv.org/abs/2402.05602
- [L11] Wang, Wiens, Lundberg. *Shapley Flow*. AISTATS 2021. https://arxiv.org/abs/2010.14592
- [L12] Lu et al. *Influence Paths for Characterizing Subject-Verb Number Agreement in LSTM Language Models*. ACL 2020. https://arxiv.org/abs/2005.01190
- [L13] Klein et al. *Improving Explainability of Disentangled Representations using Multipath-Attribution Mappings*. PMLR 2022. https://proceedings.mlr.press/v172/klein22a.html
- [L14] Ter-Minassian et al. *PWSHAP*. 2023. https://arxiv.org/abs/2306.14672
- [L15] Conmy et al. *Towards Automated Circuit Discovery for Mechanistic Interpretability*. NeurIPS 2023. https://arxiv.org/abs/2304.14997
- [L16] Syed, Rager, Conmy. *Attribution Patching Outperforms Automated Circuit Discovery*. 2023/2024. https://arxiv.org/abs/2310.10348
- [L17] Hanna et al. *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms*. COLM 2024. https://arxiv.org/abs/2403.17806
- [L18] Kramár et al. *AtP*: An efficient and scalable method for localizing LLM behaviour to components. 2024. https://arxiv.org/abs/2403.00745
- [L19] Zhang, Wang. *When Attribution Patching Lies: Diagnosis and a Second-Order Correction*. 2026. https://arxiv.org/abs/2606.09899
- [L20] Ameisen et al. *Circuit Tracing: Revealing Computational Graphs in Language Models*. 2025. https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- [L21] Hong et al. *DePass: Unified Feature Attributing by Simple Decomposed Forward Pass*. NeurIPS 2025. https://arxiv.org/abs/2510.18462
- [L22] Suzuki, Hino, Murata. *LIG: Layer-wise Integrated Gradients for Within-Layer Flow Analysis in Transformers*. 2026. https://arxiv.org/abs/2606.21564
- [L23] Nguyen. *IG-Lens: Exact Additive Probability Attribution Across Transformer Layers via Telescoping Integrated Gradients*. 2026. https://arxiv.org/abs/2606.29693
- [L24] Jantsch, Koh, Lee, Suh. *Dual Path Attribution: Efficient Attribution for SwiGLU-Transformers through Layer-Wise Target Propagation*. 2026. https://arxiv.org/abs/2603.19742
- [L25] Chen, Plaat, van Stein. *Every Component is a Lookup: Token Attribution and Composition from a Single Decomposition*. 2026. https://arxiv.org/abs/2605.23393
- [L26] Abnar, Zuidema. *Quantifying Attention Flow in Transformers*. 2020. https://arxiv.org/abs/2005.00928
- [L27] Azarkhalili, Libbrecht. *Generalized Attention Flow*. ACL 2025. https://aclanthology.org/2025.acl-long.980/
- [L28] Sun et al. *Feature Attribution in Directed Acyclic Graphs Using Edge Intervention*. 2026. https://arxiv.org/abs/2606.15273
- [L29] Wróbel, Gairola, Tabor, Schiele, Zieliński, Rymarczyk. *DAVE: Distribution-aware Attribution via ViT Gradient Decomposition*. 2026. https://arxiv.org/abs/2602.06613
- [L30] Pramanik, Maliha, Bastian, Jha. *Hessian-Enhanced Token Attribution (HETA): Interpreting Autoregressive LLMs*. ICLR 2026. https://arxiv.org/abs/2604.13258
- [L31] Knittel, Pfister. *Sparse Inter-Layer Dependencies of Transformer FFN Neurons*. 2026. https://arxiv.org/abs/2607.11990
- [L32] Nishankar, Sanjeewani, Perera, Thuseethan. *HiLRP: Toward One Trustworthy Explanation for Vision Transformer: Conservation-Valid Attribution via Attention Primitives*. 2026. https://arxiv.org/abs/2609.01282
- [L33] Berend, Achtibat, Schäffer, Binder, Samek, Lapuschkin, Dreyer. *ResLRP: The Role of Residual Cancellation in Attribution Instability in Vision Transformers*. 2026. https://arxiv.org/abs/2609.17152
- [L34] Bakish, Zimerman, Chefer, Wolf. *Revisiting LRP: Positional Attribution as the Missing Ingredient for Transformer Explainability*. 2025. https://arxiv.org/abs/2506.02138
- [L35] You, Zeng, Tsai, Yamada, Zhao. *When LRP Diverges from Leave-One-Out in Transformers*. BlackboxNLP at EMNLP 2025. https://arxiv.org/abs/2510.18810
- [L36] Buratti, Marchetti, Parlapiano, Traini, Ursino, Virgili. *Gradient-Skipping Relevance Propagation for Efficient Explainability of Vision Transformers*. 2026. https://arxiv.org/abs/2607.10365
- [L37] Mersha, Kalita. *Explainable AI: Context-Aware Layer-Wise Integrated Gradients for Explaining Transformer Models*. 2026. https://arxiv.org/abs/2602.16608
- [L38] Jo, Jang, Park. *Decision-Aware Attention Propagation for Vision Transformer Explainability*. 2026. https://arxiv.org/abs/2604.18094
- [L39] Kim, Han, Kwak. *From Local to Global to Mechanistic: An iERF-Centered Unified Framework for Interpreting Vision Models*. TPAMI 2026. https://arxiv.org/abs/2605.00474
- [L40] Cunningham, Cirone. *Contribution Weights: A Geometrical Analysis of Self-Attention Transformers*. 2026. https://arxiv.org/abs/2606.07604
- [L41] Izadi et al. *Causal Attribution via Activation Patching*. 2026. https://arxiv.org/abs/2603.13652
- [L42] Nishankar, Pathirana, Sanjeewani, Perera, Thuseethan. *Does Explainability Transfer? A Controlled Benchmark of Attribution Methods on Vision Transformers and CNNs*. 2026. https://arxiv.org/abs/2608.02396
- [L43] Dou, Shen, Zhao, Mu. *Understanding Neural Network Through Neuron Level Visualization*. Neural Networks 2023. https://doi.org/10.1016/j.neunet.2023.09.030
- [L44] Mehri, Baghshah, Pilehvar. *LibraGrad: Balancing Gradient Flow for Universally Better Vision Transformer Attributions*. CVPR 2025. https://arxiv.org/abs/2411.16760
- [L45] Bousselham, Boggust, Chaybouti, Strobelt, Kuehne. *LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity*. 2024/2025. https://arxiv.org/abs/2404.03214
- [L46] Jo, Jang, Park. *GMAR: Gradient-Driven Multi-Head Attention Rollout for Vision Transformer Interpretability*. 2025. https://arxiv.org/abs/2504.19414
