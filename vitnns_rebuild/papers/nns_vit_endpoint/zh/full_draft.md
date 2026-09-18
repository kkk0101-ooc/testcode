# NNS-ViT Endpoint Scanner：视觉 Transformer 的注册内部目标扫描与结构化分解

- 作者：`[AUTHOR REQUIRED]`
- 单位：`[AFFILIATION REQUIRED]`
- 通讯作者：`[CORRESPONDING AUTHOR REQUIRED]`

> 初稿 R1，2026-09-18。正文图号沿用封存 Endpoint 图包 Figure 1–28；投稿排版可按
> `papers/nns_vit_endpoint/zh/outline.md` 的建议重新编号。

## 摘要

视觉 Transformer（ViT）的内部解释对象横跨 token、channel、attention head、
routing/value 路径、MLP 分支和 residual stream。它们的张量形状与计算角色不同，使得
统一扫描、结构核对和行为评价难以在同一坐标系统中完成。本文提出 NNS-ViT Endpoint
Scanner：把原始 Neural Network Scanner 的输入值初始化与当前状态传播原则适配到 ViT，
对任意注册标量目标生成
$L_s^{\mathrm{Endpoint}}(x)=x\odot\nabla_xs(x)$，并以完整计算边切面把同一最终目标
分解为 head、Q/K/V、routing/value、attention residual 和 MLP residual 成分。该基础量
与 Gradient × Input 相同；本文研究的是 ViT 目标注册、结构化记账以及这些成分在有限
干预中的经验效用。我们在共享的 400 张 ImageNet evaluation 图像上评价 ViT-B/16 和
DeiT-S/16。注册扫描完成 800/800 个图像—模型任务和 336,800 个有限目标值；结构验证
通过 76,800 个正向闭合检查，拒绝 28,800 个遗漏或重复记账负例，并通过 6,144 个
condition-scaled analytic JVP 检查。对于匹配 activation replacement，12 个预注册
family 中有 11 个达到相关、方向、精度和多重检验条件。唯一例外的中位 Spearman
相关为 0.099565，比预设 0.10 阈值低 0.000435；因此 RQ3 的科学综合结论为整体支持且
存在一个注册边界例外，而要求 12/12 的冻结 omnibus 字段仍为 `NOT_SUPPORTED`。在
空间定向干预中，0/4 family 同时优于 total G×I、raw gradient、regional Taylor 和
随机区域；regional Taylor 在四个 family 中均取得更高的中位效果。结果表明，Endpoint
结构成分可以预测成分级有限效应，但这种能力不自动意味着其像素分布是更优的局部区域
选择器。

关键词：Vision Transformer；神经网络解释；Gradient × Input；结构分解；activation
replacement；attention head

## 1. 引言

Vision Transformer 将图像表示组织为 patch tokens，并通过多头自注意力、MLP 和
residual stream 反复变换这些表示（Dosovitskiy et al., 2021）。这套结构产生了丰富的
内部观察对象：一个 token-channel 单元、一条 head 分支、Q/K routing、Value 内容、
MLP 分支或最终类别 logit 都可以成为解释目标。然而，这些对象生活在不同的张量空间。
直接查看 activation 或 attention 权重能够描述内部状态，却不能自动回答某个对象如何依赖
原始输入，也不能保证多个结构成分对同一目标的记账既无遗漏又无重复。

已有 ViT 解释方法从不同角度处理这一问题。Attention rollout 递归聚合 token 间的
attention flow（Abnar and Zuidema, 2020）；Transformer relevance 方法把类别相关信息沿
计算图传播（Chefer et al., 2021；Achtibat et al., 2024）；输入归因方法则把最终输出映射
到像素坐标。它们回答的对象、传播规则和评价标准并不相同。对内部结构进行解释时，还需
区分三个问题：方法的数学身份是什么，代码是否忠实实现这一身份，以及所得数值是否能
预测有限干预。只验证梯度或闭合无法替代干预评价；一张空间细致的图也不一定比一个直接
针对候选区域的局部效应估计更适合区域选择。

Dou et al. (2023) 提出的 Neural Network Scanner（NNS）为 FCN/CNN 内部神经元构造
保留输入位置的 learning image。在线性层、卷积、当前 ReLU 门和唯一 MaxPool winner
采用真实局部传播系数时，该单神经元规则可写为输入值与当前点梯度的乘积。ViT 引入
LayerNorm、GELU、输入依赖的 softmax attention、两处 residual add 以及多种 token/branch
对象，因此需要明确其真实 Jacobian、结构切面和目标身份。本文沿用 NNS 的终点传播思想，
但不把 $x\odot\nabla_xs(x)$ 重新命名为新的基础归因公式。

我们提出 NNS-ViT Endpoint Scanner，并围绕四个研究问题进行评价（Figure 1）：

1. **RQ1：注册扫描。** 能否在两个冻结 ViT 上完整、有限且可审计地扫描统一登记的
   内部标量和输出目标？
2. **RQ2：结构正确性。** Head、Q/K/V、routing/value 和 residual cuts 能否在完整计算
   图上重构其注册对象，并拒绝遗漏或重复记账？
3. **RQ3：成分级效用。** Endpoint 成分的输入空间质量能否预测同一成分被 reference
   activation 替换时的有限 logit effect？
4. **RQ4：空间效用。** 完整输入分辨 component map 能否在固定区域池和预算下，比总
   G×I、raw gradient、regional Taylor 和随机区域更好地定位有限效应？

本文的贡献有五点。第一，我们把 NNS 的终点原则形式化为真实 ViT Jacobian，并明确
normalized-input 坐标原点以及 LayerNorm、GELU、softmax attention、MLP 和 residual 的
传播作用。第二，我们建立两个 ViT 的注册目标目录和可审计扫描流程，在同一 400 图 cohort
上完成 336,800 个目标评价。第三，我们定义完整 edge cuts，把 head、attention differential
paths 和两处 residual 分支映回同一输入坐标并重构注册对象。第四，我们用闭合负例、解析
JVP、匹配 activation replacement 和空间定向干预分别评价 correctness 与 utility。第五，
我们给出两类互补的经验结论：成分级有限效应预测整体成立并保留一个边界例外，而完整
输入分辨空间排序没有超过强局部 Taylor 基线。

## 2. 相关工作

### 2.1 神经元可视化与 NNS

内部可视化方法既可以寻找最大化某个单元的输入，也可以从一个实际样本出发追踪其内部
响应。Feature visualization、network dissection、deconvolutional views 和 class activation
maps 分别刻画偏好输入、语义概念、局部响应或类别相关区域。NNS 为另一类对象：它给
FCN/CNN 的单个神经元或聚合响应关联一张保留输入位置的 learning image（Dou et al.,
2023）。本文继承这一“内部目标—输入坐标图”的组织方式，并把当前状态传播扩展到 ViT
的真实算子。Captum 的 NeuronGradient 和 InputXGradient 已分别提供内部 scalar selector
和输入乘梯度的明确先例，因此本文的增量不在 hook、selector 或基础 G×I 公式，而在注册
范围、结构化切分、闭合验证与有限干预审计。

### 2.2 输入归因、路径积分与 conductance

Integrated Gradients（IG）沿参考输入到当前输入的路径积分梯度，并满足包括 sensitivity
和 implementation invariance 在内的公理（Sundararajan et al., 2017）。Neuron
conductance 进一步衡量输出归因如何流经隐藏单元（Dhamdhere et al., 2018）。这些方法
与 Endpoint 的差别不是 CNN 或 Transformer 的架构标签，而是归因定义：Endpoint 使用
当前点 Jacobian 和输入值 seed；IG 使用参考差值 seed 和路径平均 Jacobian。本文同时保留
这一方法边界。由于 ViT-B/16 在预注册的最大 512 点数值范围内没有达到 Path IG 校准门，
我们不发布跨模型 Endpoint–Path fidelity、stability 或成本—质量结论。

### 2.3 Transformer 解释与结构分析

Raw attention 提供 token 关系，但不包含完整的下游类别导数。Attention rollout 聚合跨层
attention 与 residual 路径（Abnar and Zuidema, 2020）；Chefer et al. (2021) 与 AttnLRP
（Achtibat et al., 2024）采用 relevance 规则处理 Transformer 计算图；LibraGrad 调整
backward 路径以缓解 Transformer 的梯度流不平衡（Mehri et al., 2025）。这些工作说明
attention、skip connection 和梯度路径都影响解释。Endpoint Scanner 保留模型的真实局部
Jacobian，并将 raw attention、relevance 传播和修改后的 backward attribution 视为不同方法。

### 2.4 Activation patching 与解释评价

Activation patching 通过替换内部 activation 直接测量模型输出变化；Attribution Patching
用 activation displacement 与 clean gradient 的内积近似该有限效应，AtP* 进一步分析了
attention saturation 和抵消导致的失效模式（Kramár et al., 2024）。这一区分对本文尤为
重要：局部导数身份属于 correctness，而有限 replacement 才检验行为预测。我们同时考察
整成分排序和固定空间预算下的区域排序，并把 activation-space regional Taylor 作为后者
的 mandatory baseline。

## 3. 方法

### 3.1 Endpoint learning image 的数学身份

设模型实际接收的输入为 $x\in\mathbb R^d$。对计算图节点
$u(x)\in\mathbb R^m$，定义 learning-image 矩阵
$\mathcal L(u;x)\in\mathbb R^{m\times d}$。输入坐标初始化为

$$
\mathcal L(x;x)=\operatorname{diag}(x).
$$

若 $v=f(u_1,\ldots,u_r)$，所有 Jacobian 在真实输入对应的当前前向状态求值，则

$$
\mathcal L(v;x)=\sum_{a=1}^{r}J_{f,u_a}(x)\mathcal L(u_a;x).
$$

对标量目标 $s(x)$ 递归应用链式法则得到

$$
\boxed{L_s^{\mathrm{Endpoint}}(x)=x\odot\nabla_xs(x)}.
$$

Figure 2 区分原始 NNS-CNN、ViT Endpoint 与 Path IG 的 seed 和 Jacobian。Endpoint
不包含 reference forward，也不具备一般的 baseline-difference completeness。输入坐标是
模型实际接收的 normalized RGB。若 $x=(r-\mu)/\sigma$，则

$$
x_i\frac{\partial s}{\partial x_i}
=(r_i-\mu_i)\frac{\partial s}{\partial r_i}.
$$

因此 canonical Endpoint 在 raw-RGB 坐标中以模型 normalization mean 为坐标原点。二维
显示图由原始 signed RGB tensor 做通道求和得到；绝对值、正部或逐图显示缩放不进入统计。

### 3.2 注册内部目标

Scanner 接受任何已登记的标量目标。目录包括 token-channel 单元、排除 CLS 的
patch-channel sum、patch-channel energy、attention head 内部坐标、final CLS channel、
ground-truth raw logit 和 frozen clean-top1 raw logit。Energy 目标定义为

$$
s_{\mathrm{energy}}(x)=\frac12\sum_{t=1}^{196}h_{t,c}(x)^2,
$$

它对应原始 NNS filter aggregation 的激活加权形式。ViT-B/16 每图注册 457 个目标，
DeiT-S/16 每图注册 385 个目标；两者均覆盖全部 12 个 blocks（Figure 7、11）。不同
模型中编号相同的 channel 不解释为语义相同。

### 3.3 ViT 算子与 attention differential paths

Canonical backend 使用 timm 0.9.16 的 standard unfused Block/Attention。Patch embedding、
LayerNorm、GELU、Q/K/V projections、row softmax、multi-head concatenation、output
projection、MLP、两次 residual add、final LayerNorm 和 classifier 均使用当前状态的真实
Jacobian。Fused backend 仅用于 forward parity，因为它不暴露注册结构边。

对于单个 attention head，令

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

$$
A=\operatorname{softmax}(QK^\top/\sqrt{d_h}),\qquad O=AV.
$$

其微分满足

$$
\mathrm dO=(\mathrm dA)V+A(\mathrm dV),
$$

其中 $\mathrm dA$ 继续由 Q 和 K 两条路径构成。由此得到 Q+K+V 与
routing+Value 两种注册分解（Figure 5）。LayerNorm 的均值、方差耦合和 GELU 的平滑
导数均保留，不能替换为冻结统计量或 ReLU 二值门。

### 3.4 Structured endpoint decomposition

对最终标量 $F$ 和切面成分 $z_k$，定义输入坐标成分图

$$
E^C_{k,i}=x_i\left\langle\nabla_{z_k}F(x),
\frac{\partial z_k(x)}{\partial x_i}\right\rangle.
$$

若 cut 包含所有从输入到目标的对应边且没有重复，则

$$
\sum_kE^C_{k,i}=x_i\frac{\partial F(x)}{\partial x_i}.
$$

我们登记六类切面：全 heads 重构 attention branch；identity 加 heads 重构第一个
residual total；Q+K+V 或 routing+Value 重构 attention branch；identity 加相应 differential
paths 重构 residual total；carry 加 MLP branch 重构第二个 residual total（Figure 10、15）。
Residual identity 必须按其真实输入边记账；遗漏一条非零 branch 或重复加入 bypass 都应被
闭合验证拒绝。

### 3.5 局部正确性与有限 activation replacement

对每个成分图，定义 signed sum 和 absolute mass：

$$
a_k=\sum_iE^C_{k,i},\qquad m_k=\sum_i|E^C_{k,i}|.
$$

令 $v_k=J_{z_k,x}x$。解析局部合同为

$$
a_k=\langle\nabla_{z_k}F,v_k\rangle.
$$

它检查 hook、cotangent 和方向构造是否一致。有限效应通过同图 reference activation
replacement 定义：

$$
e_k=F(x)-F(x;z_k\leftarrow z_k^{\mathrm{ref}}).
$$

Reference 分为 normalized-zero 与 RGB blur。RQ3 的主量是每张图像内部
$\operatorname{Spearman}(m_k,|e_k|)$；head、routing/value 和 residual branches 分别形成
family。Activation-displacement Taylor

$$
p_k^{\mathrm{ref}}=\langle\nabla_{z_k}F,z_k(x)-z_k^{\mathrm{ref}}\rangle
$$

作为预注册次级 predictor 报告，但不进入 C4a 的支持门（Figure 19、22）。

### 3.6 空间定向干预

RQ4 固定 block 5 的 head 0、中间 head 和最后一个 head。224×224 输入被划分为 4×4 个
不重叠的 56×56 区域；排序后选择前 1、2 或 4 个区域，对应 6.25%、12.5% 和 25% 面积。
Endpoint 区域分数为成分图的 RGB-level absolute mass。四个 mandatory baselines 是
total Endpoint/G×I、raw input gradient、regional activation-space Taylor 和 20 个冻结
随机区域排列（Figure 23）。

对同一 head 的候选区域并集 $R$，有限效果为

$$
F(x)-F\bigl(z_h(x_R),z_{-h}(x),\mathrm{identity}(x)\bigr),
$$

其中 $x_R$ 使用相应 reference 替换。每图先对三个 head 取平均，再计算 0–25% 面积上的
absolute isolated-effect trapezoidal AUC，并除以 0.25。所有方法共享候选区域池、面积预算和
有限效应定义。

## 4. 实验设计

### 4.1 模型、数据与预处理

实验使用 `vit_base_patch16_224.augreg2_in21k_ft_in1k` 和
`deit_small_patch16_224.fb_in1k`。两者均有 12 个 blocks；ViT-B/16 有 12 heads 和 768
维表示，DeiT-S/16 有 6 heads 和 384 维表示。共同输入来自 ImageNet validation，按冻结
元数据规则构造 64/64/400 张 discovery、calibration 和 evaluation 集合；集合间图像和类别
互斥，两个模型共享全部 400 个 evaluation ID。图像短边 resize 到 248，使用 bicubic
antialias，再 center crop 到 224；两模型使用各自冻结的 mean/std。

生产环境为 Torch 2.2.2+cu121、timm 0.9.16、torchvision 0.17.2+cu121、NumPy
1.26.4 和 Pillow 10.2.0。模型在 eval 模式运行，dropout 和 drop path 关闭；canonical
结构实验禁用 fused attention。模型文件、图像成员、目标表、任务 payload 和结果文件均以
SHA-256 绑定。

**Table 1. 模型、预处理与注册目标。**

| 项目 | ViT-B/16 | DeiT-S/16 |
|---|---:|---:|
| timm 模型 | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | `deit_small_patch16_224.fb_in1k` |
| Blocks / heads / hidden dim | 12 / 12 / 768 | 12 / 6 / 384 |
| 注册目标数/图 | 457 | 385 |
| Evaluation images | 400 | 400（相同 ID） |
| Resize / crop | 短边 248 / center crop 224 | 短边 248 / center crop 224 |
| Normalization mean | [0.5, 0.5, 0.5] | [0.485, 0.456, 0.406] |
| Normalization std | [0.5, 0.5, 0.5] | [0.229, 0.224, 0.225] |

### 4.2 确认性统计

统计计划在读取 evaluation outcomes 前冻结 16 个 family 和 28 个 test IDs。C4a 由
2 模型 × 2 references × 3 成分族形成 12 个 family；主统计量为 400 张图的逐图 Spearman
中位数，实际效应阈值为 0.10。C4b 由 2 模型 × 2 references 形成 4 个 family；每个
family 必须相对四个 mandatory baselines 的图像级配对中位差均至少为 0.05。

每个 family 使用 10,000 次 paired image-cluster percentile bootstrap 构造 95% CI。每个
test 使用双侧精确符号检验，28 个 test IDs 统一进行 Holm 校正，$\alpha=0.05$。每个
family 至少需要 360 张共同有效图，符号检验至少需要 300 个非零图像差；CI 最大宽度分别
为 C4a 的 0.30 和 C4b 的 0.50。统计单位始终是图像，不把 head、block、reference 或
随机排列当作独立样本。

### 4.3 验证与可复现性

各阶段使用不可变结果根、精确覆盖审计和独立验证器。结构闭合同时包含必须通过的正例和
必须失败的遗漏/重复负例。统计验证器不导入 Torch、timm 或 PIL，而是从封存的逐任务
结果重算逐图量、bootstrap、sign tests、Holm 调整和 family decisions。论文图包由 CPU-only
renderer 从封存数字和原始展示数组生成；显示归一化图不用于数值推断。

## 5. 结果

### 5.1 RQ1：注册目标扫描完整且在资源上可执行

两个模型的 800/800 个 image-model tasks 均唯一完成，无缺失、重复、无效结果或正式重试。
ViT-B/16 每图的 457 个目标和 DeiT-S/16 每图的 385 个目标形成 336,800/336,800 个有限
target evaluations。所有图像保留 ground-truth 与 clean-top1 raw signed RGB class maps，
前 16 个 evaluation ID × 两模型保留 32 份 full raw atlas；未出现 near-zero 或失败索引。
固定前 8 张图在两个模型上的结果见 Figure 12，多目标 atlas 示例见 Figure 13。

正式扫描 wall time 为 1,752.71 s，按四卡同时占用的保守上界为 1.9475 GPU-hours。峰值
reserved GPU memory 为 2.284 GB，峰值 CPU RSS 为 2.018 GB，retained data 为 9.290 GB。
正式吞吐为 192.16 targets/s 和 0.4564 images/s。完整 atlas 的 median/P95/max latency 在
DeiT-S/16 上为 7.853/8.015/8.240 s，在 ViT-B/16 上为 9.485/9.751/10.615 s
（Figure 14）。这些结果支持 RQ1：注册目标表可以在固定模型和 cohort 上完整执行并形成
可核对的 atlas。

**Table 2. RQ1 注册扫描覆盖与资源。**

| 指标 | 结果 |
|---|---:|
| Image-model tasks | 800/800 |
| Finite target evaluations | 336,800/336,800 |
| Formal wall time | 1,752.71 s |
| Four-card upper-bound GPU-hours | 1.9475 |
| Formal throughput | 192.16 targets/s |
| Peak reserved GPU memory | 2.284 GB |
| Peak CPU RSS | 2.018 GB |
| Retained data | 9.290 GB |

### 5.2 RQ2：结构切面在正例和负例上均满足合同

正式闭合实验覆盖 400 张图、两个模型、全部 12 个 blocks 和每个 block 的 8 个注册 cuts，
共 76,800 个正向检查。全部检查均满足 float32 合同
`atol=1e-5, rtol=1e-4`；最大 relative L2 为
$1.4144569831842091\times10^{-5}$，最大 absolute error 为
$2.47955322265625\times10^{-5}$。由于 elementwise 合同为
$|\mathrm{error}|\leq\mathrm{atol}+\mathrm{rtol}|\mathrm{reference}|$，后一个数值本身不构成
违约。

每个 block 还执行三个 checker 负例，故意遗漏非零 head/routing 或重复 residual，共
28,800 项；这些负例全部按预期被拒绝（Figure 16）。前 16 个 evaluation ID 的 384 个 raw
block arrays 又由 CPU-only validator 从 component maps 独立重算。RQ2 因此得到支持：在
注册 cut 与两个标准 timm ViT 范围内，结构成分能够重构对应 attention branch 或 residual
total，checker 也能识别典型漏项和重复记账。

### 5.3 局部微分实现通过独立方向检查

解析局部合同进一步检查成分图的 signed sum 是否等于相同 hook 和 cotangent 下的 JVP
方向量。两个模型前 16 个 evaluation order 上的 6,144 个 condition-scaled analytic JVP
checks 全部通过（Figure 18）。最大 absolute error 为
$2.648076753197781\times10^{-5}$；最大
`error/sum_abs_products` 为 $6.948755071487438\times10^{-6}$，低于预先冻结的
$10^{-5}+10^{-5}\sum_i|g_i(Jx)_i|$ 边界。该结果验证局部微分实现，不替代下一节的
finite replacement 评价。

**Table 3. RQ2 结构闭合与局部微分检查。**

| 检查 | 数量 | 结果 | 最大误差 |
|---|---:|---|---:|
| Registered positive closures | 76,800 | 全部通过 | relative L2 = 1.41446e-05 |
| Omission/double-counting controls | 28,800 | 全部按预期拒绝 | 不适用 |
| Condition-scaled analytic JVP | 6,144 | 全部通过 | absolute = 2.64808e-05 |
| JVP condition-scaled ratio | 6,144 | 全部通过 | 6.94876e-06 |

### 5.4 RQ3：结构成分整体上预测匹配的有限 activation effect

RQ3 比较每图的 Endpoint component absolute mass $m_k$ 与同一成分有限 replacement
effect 的绝对值 $|e_k|$。12 个预注册 family 中，11 个满足点估计至少 0.10、正方向、
Holm 显著、覆盖和 CI 精度的全部条件（Figure 20）。各 family 中位 Spearman 相关介于
0.099565 与 0.306320。Head family 在 ViT-B/16 上为 0.290853（normalized-zero）和
0.306320（blur），在 DeiT-S/16 上为 0.190961 和 0.181668。Residual-branch family 的
四个估计为 0.177391、0.209130、0.135217 和 0.228696。

唯一未达到全部条件的是 ViT-B/16 × normalized-zero × routing/value。其点估计为
0.09956521739130435，95% CI 为 [0.07478260869565218, 0.12260869565217392]，Holm
校正 $p=3.0036026037128944\times10^{-14}$。相关方向为正且通过多重检验，但点估计比
预设实际效应阈值 0.10 低 0.000435（Figure 21）。我们据此将 RQ3 解释为
**整体支持，并存在一个注册边界例外**。同时，冻结的 family 状态和要求 12/12 的 omnibus
C4a 字段继续为 `NOT_SUPPORTED`；阈值没有移动，该数值也没有四舍五入成通过。

这一双层报告分别回答两个问题：严格机器字段说明预注册的 all-family 规则未被完全满足；
科学综合说明 11/12 个跨模型、reference 和成分族的结果满足所有条件，剩余 family 也呈
正向且统计可辨的关联。C4a 只检验 Endpoint mass 对有限效应的排序关系，不检验其是否
优于 activation-space Taylor 或其他 predictor。

**Table 4. RQ3 的 12 个 C4a family。** 每项 $n=400$；$p$ 为全 28 项 Holm 校正值。

| 模型 | Reference | 成分族 | 中位 Spearman | 95% CI | Holm $p$ | Family 状态 |
|---|---|---|---:|---:|---:|---|
| ViT-B/16 | normalized-zero | heads | 0.2909 | [0.2813, 0.2982] | 2.17e-119 | SUPPORTED |
| ViT-B/16 | normalized-zero | routing/value | 0.0996 | [0.0748, 0.1226] | 3.00e-14 | NOT_SUPPORTED |
| ViT-B/16 | normalized-zero | residual branches | 0.1774 | [0.1548, 0.1952] | 2.56e-46 | SUPPORTED |
| ViT-B/16 | blur | heads | 0.3063 | [0.2957, 0.3181] | 2.17e-119 | SUPPORTED |
| ViT-B/16 | blur | routing/value | 0.1565 | [0.1322, 0.1774] | 9.96e-29 | SUPPORTED |
| ViT-B/16 | blur | residual branches | 0.2091 | [0.1817, 0.2374] | 1.46e-41 | SUPPORTED |
| DeiT-S/16 | normalized-zero | heads | 0.1910 | [0.1769, 0.2052] | 1.88e-76 | SUPPORTED |
| DeiT-S/16 | normalized-zero | routing/value | 0.1065 | [0.0817, 0.1291] | 2.11e-10 | SUPPORTED |
| DeiT-S/16 | normalized-zero | residual branches | 0.1352 | [0.1148, 0.1643] | 4.35e-19 | SUPPORTED |
| DeiT-S/16 | blur | heads | 0.1817 | [0.1688, 0.1951] | 3.82e-72 | SUPPORTED |
| DeiT-S/16 | blur | routing/value | 0.1600 | [0.1278, 0.1822] | 9.90e-31 | SUPPORTED |
| DeiT-S/16 | blur | residual branches | 0.2287 | [0.2035, 0.2504] | 3.70e-54 | SUPPORTED |

### 5.5 RQ4：完整输入分辨 component map 未提供注册的空间增益

RQ4 要求每个模型—reference family 同时相对 total G×I、raw gradient、regional Taylor
和随机区域取得至少 0.05 的配对中位增益。四个 family 均未满足该联合条件（Figure 26）。
Endpoint 相对随机区域的中位增益均为正，分别为 0.001069、0.004212、0.010689 和
0.008460，但均远低于冻结的 0.05 阈值。相对 total G×I 与 raw gradient 的差值接近零，
方向混合。

Regional Taylor 在全部四个 family 中得到更高的中位效果。Endpoint−regional Taylor
分别为 −0.023433、−0.023697、−0.029900 和 −0.035630，bootstrap intervals 也全部位于
零以下（Figure 27）。真实 block-5 component maps 与 baseline maps 见 Figure 24；这些图
展示空间形态，但统计结论来自 raw arrays 和有限干预曲线。RQ4 因此为
`NOT_SUPPORTED`：在本研究固定的 head、区域池、面积预算、reference 和效果指标下，
完整输入分辨 component map 没有显示超越四个 mandatory baselines 的联合信息增益。

**Table 5. RQ4 的 C4b 图像级配对中位差。** 数值为 Endpoint 减 baseline；每个 family
需四列同时至少为 +0.05 才能支持。

| 模型 | Reference | vs total G×I | vs raw gradient | vs regional Taylor | vs random | Family 状态 |
|---|---|---:|---:|---:|---:|---|
| ViT-B/16 | normalized-zero | −0.0009 | −0.0000 | −0.0234 | 0.0011 | NOT_SUPPORTED |
| ViT-B/16 | blur | −0.0010 | −0.0007 | −0.0237 | 0.0042 | NOT_SUPPORTED |
| DeiT-S/16 | normalized-zero | −0.0018 | −0.0006 | −0.0299 | 0.0107 | NOT_SUPPORTED |
| DeiT-S/16 | blur | 0.0002 | −0.0002 | −0.0356 | 0.0085 | NOT_SUPPORTED |

## 6. 讨论

### 6.1 Identity、Correctness 与 Utility 是三种不同结论

Endpoint 的数学身份由链式法则确定：对同一输入端标量目标，基础图就是
$x\odot\nabla_xs(x)$。结构闭合与解析 JVP 检查说明代码在注册计算图上正确实现了目标、
hook、cotangent 和 edge bookkeeping。有限 activation replacement 与空间干预则回答
这些量是否能预测模型在非无穷小替换下的行为。Figure 28 汇总了这三层证据。把它们分开
可以避免两种常见推断：闭合不等于行为效用，干预中的负结果也不推翻链式法则身份。

### 6.2 RQ3 与 RQ4 的差异来自评价任务

RQ3 和 RQ4 并不矛盾。RQ3 在一张图像内比较多个结构成分，询问“哪个 head、
routing/value 或 residual branch 的整体质量更大，哪个成分被完整替换时的影响也更大”。
RQ4 则固定一个 head，在 16 个候选输入区域中询问“先替换哪些局部区域会产生更大的有限
效应”。前者是 component-level ranking，后者是 within-component spatial ranking。

Endpoint absolute mass 保留结构成分相对输入的完整一阶分布，但它没有显式使用某个候选
区域替换后产生的 activation displacement。Regional Taylor 恰好把这一位移与 clean
cotangent 相乘，因此其估计对象与 RQ4 的区域替换更接近。四个 family 中一致出现的
regional Taylor 优势支持这一任务层面的解释。当前实验不进一步推出 regional Taylor 在
其他层、区域尺度、任务或模型上普遍占优。

### 6.3 如何解释 RQ3 的边界 family

ViT-B/16 normalized-zero routing/value family 的 0.099565 是一个可定位的边界结果。它
通过方向和 Holm 检验，但没有达到冻结的 0.10 实际效应阈值，因此 family 状态必须保持
`NOT_SUPPORTED`。与此同时，差值只有 0.000435，且其余 11 个 family 满足全部条件，
所以把整个 RQ3 简化为“失败”会丢失跨 family 的主要经验模式。本文在结果中同时保留
严格字段和科学综合，不为这个单一例外提出未经检验的架构或 reference 机制。

### 6.4 对 ViT 内部解释评价的启示

本研究显示，一种内部解释可以同时具备以下部分属性而缺少另一些属性：目标覆盖完整、
结构记账闭合、局部方向正确、成分级有限效应可预测、空间区域选择优于强基线。这些属性
需要各自的测试。特别是，可视化分辨率描述图像的坐标粒度，不等于其中包含额外可用的
干预信息。解释方法的评价应与使用目的对应：若任务是筛选结构成分，RQ3 的结果更直接；
若任务是选择局部区域，RQ4 表明 regional activation-space 信息不可忽略。

## 7. 局限

本研究覆盖两个 ImageNet 分类 ViT、一个共享的 400 图 evaluation cohort 和冻结的标准
timm 实现；结论不直接覆盖层级式 Transformer、检测/分割任务、自监督模型或其他数据域。
Canonical Endpoint 依赖输入坐标原点，并且一般不满足相对 baseline 的 completeness。
RQ4 固定在 block 5、每模型三个 heads、4×4 区域池、最高 25% 面积、两种 reference 和
raw-logit isolated-effect AUC；其他空间尺度或干预对象可能形成不同排序。

预注册 Path IG 校准在 ViT-B/16 上未达到 61/64 的数值覆盖门，因此本文没有跨模型
Endpoint–Path fidelity、stability 或成本—质量比较。该数值失败不影响 Endpoint-only 的
身份、闭合和干预结果，但限制了方法间结论。最后，论文中的颜色图按面板进行显示缩放，
只用于比较空间形态；跨图数值结论均来自原始 signed tensors、component summaries 和
有限干预记录。

## 8. 结论

NNS-ViT Endpoint Scanner 把原始 NNS 的终点传播原则适配到 Vision Transformer，为
注册内部目标提供统一的输入坐标图，并通过完整 edge cuts 对 attention 与 residual 结构进行
可核对分解。两个模型和 400 张图的实验支持其扫描完整性、结构闭合与局部微分正确性。
匹配 activation replacement 进一步表明，Endpoint 成分质量在 12 个 family 中有 11 个
满足全部确认性条件，RQ3 整体得到支持并保留一个 0.099565 的注册边界例外。空间定向
实验没有发现相对四个 mandatory baselines 的联合增益，regional Taylor 在四个 family 中
均更强。由此，结构成分级解释和空间区域选择应被视为两个需要分别验证的用途。

## 数据、代码与可复现材料

代码、冻结协议、目标注册表、逐阶段验证报告、统计 release 和 28 张论文图位于本项目
仓库。提交版本的匿名链接、许可证、归档 DOI 与数据访问说明为
`[REPOSITORY AND ARCHIVE REQUIRED]`。

## 参考文献

本初稿使用的参考文献清单见
`papers/nns_vit_endpoint/references/endpoint_scanner_references_r1.md`。投稿前需完成最终 BibTeX、DOI、
页码和 venue metadata 审计。
