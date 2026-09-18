# NNS-ViT Endpoint Scanner：数学分析及与既有 NNS-ViT、NNS-CNN 的比较

日期：2026-09-18

## 1. 结论先行

NNS-ViT Endpoint Scanner 延续的是原始 NNS-CNN 的两个核心原则：

1. 用实际送入模型的输入坐标值初始化 learning images；
2. 在当前样本的单个前向状态上，用局部传播系数把 learning images 传播到内部单元或输出目标。

把“局部传播系数”写成模型在当前点的真实局部 Jacobian 后，任意标量目标 $s(x)$ 的结果为

$$
\boxed{
L_s^{\mathrm{Endpoint}}(x)
=x\odot\nabla_x s(x)
}.
\tag{1}
$$

它与原始 NNS-CNN 的关系是“同一终点原则在不同计算图上的实现”。CNN 与 ViT 改变 Jacobian 的内容；是否使用路径积分决定归因是 Endpoint 还是 Integrated Gradients。

此前项目中的 NNS-ViT learning images 主要使用

$$
\boxed{
L_s^{\mathrm{Path}}(x;x_0)
=(x-x_0)\odot
\int_0^1\nabla_xs\bigl(x_0+\alpha(x-x_0)\bigr)\,\mathrm d\alpha
}.
\tag{2}
$$

式 (2) 就是标量目标 $s$ 的 Integrated Gradients（IG）。它继承了原 NNS 的“内部目标 + 输入坐标图”组织方式，但归因定义已经从终点传播改为路径积分。因此，当前项目应把以下对象分开命名：

| 对象 | 数学定义 | 方法身份 |
|---|---|---|
| 原始 NNS-CNN 单神经元图 | 当前点的权重、ReLU 门和 pooling winner 传播 | 在本文列出的条件下等于 $x\odot\nabla s(x)$ |
| NNS-ViT Endpoint Scanner | $x\odot\nabla s(x)$ | 内部目标的 Gradient × Input scanner |
| 此前的 response learning image | internal-target IG，见式 (2) | 路径积分 learning image |
| Conditional Value-path NNS-ViT | 固定 clean attention 后，对 $A_{\rm clean}V(x)$ 内部标量做 IG | 条件 Value 路径，不是完整 attention |
| NNS-v2 complete-cut | 对最终类别 IG 做完整计算切面分解 | input-resolved path conductance |
| Endpoint structured decomposition | 对最终类别 $x\odot\nabla F(x)$ 做结构化终点分解 | edge-cut 与 attention differential-path decomposition |

因此，Endpoint Scanner 的基础归因公式并不是一个不同于 Gradient × Input 的新量。它的技术内容位于 ViT 内部目标注册、真实算子适配、单元扫描，以及不漏项、不重复记账的 head、Q/K/V 和 residual 终点分解。

## 2. 统一的计算图定义

设模型输入为

$$
x=(x_1,\ldots,x_d)^\top\in\mathbb R^d.
$$

对计算图中的向量节点 $u(x)\in\mathbb R^m$，定义 learning-image 矩阵

$$
\mathcal L(u;x)\in\mathbb R^{m\times d},
$$

其中第 $j$ 行是内部单元 $u_j$ 在全部输入坐标上的 learning image。

### 2.1 输入值初始化

第 $p$ 个输入单元的 learning image 为

$$
\mathcal L(x_p;x)[i]
=x_i\mathbf 1(i=p).
\tag{3}
$$

矩阵形式为

$$
\mathcal L(x;x)=\operatorname{diag}(x).
\tag{4}
$$

这正是原 NNS 论文式 (1)–(3) 的“将输入拆成单坐标图”：

$$
x=\sum_{p=1}^d x_pe_p.
$$

### 2.2 多输入算子的终点传播

设某个节点由多个父节点产生：

$$
v=f(u_1,\ldots,u_r).
$$

所有局部 Jacobian 均在真实输入 $x$ 对应的当前前向状态求值：

$$
\mathcal L(v;x)
=\sum_{a=1}^{r}
J_{f,u_a}(x)\mathcal L(u_a;x),
\tag{5}
$$

其中

$$
J_{f,u_a}(x)
=\left.
\frac{\partial f}{\partial u_a}
\right|_{u_a=u_a(x)}.
$$

对有向无环计算图递归使用式 (5)，链式法则给出

$$
\mathcal L(v;x)
=J_{v,x}(x)\operatorname{diag}(x).
\tag{6}
$$

当 $v=s$ 是标量时，将行向量转回输入张量布局：

$$
\mathcal L(s;x)[i]
=x_i\frac{\partial s(x)}{\partial x_i}.
\tag{7}
$$

式 (7) 说明显式逐层传播和一次完整自动微分 VJP 得到相同的最终输入 learning image。显式传播的作用是暴露中间对象和计算边；它不应改变最终数学量。

### 2.3 路径版本

此前 NNS-ViT 将输入初始化改为相对参考值，并沿路径

$$
x_\alpha=x_0+\alpha(x-x_0)
$$

对 Jacobian 取平均：

$$
\mathcal L^{\mathrm{Path}}(s;x,x_0)[i]
=(x_i-x_{0i})
\int_0^1
\frac{\partial s(x_\alpha)}{\partial x_i}
\,\mathrm d\alpha.
\tag{8}
$$

式 (9) 比较 baseline-relative endpoint extension

$$
L_s^{\mathrm{Endpoint},\Delta}(x;x_0)
=(x-x_0)\odot\nabla_xs(x)
$$

与 Path IG。两者逐坐标之差为

$$
\begin{aligned}
L_s^{\mathrm{Endpoint},\Delta}[i]
-L_s^{\mathrm{Path}}[i]
&=(x_i-x_{0i})
\left[
\frac{\partial s(x)}{\partial x_i}
-\int_0^1\frac{\partial s(x_\alpha)}{\partial x_i}\,\mathrm d\alpha
\right].
\end{aligned}
\tag{9}
$$

二者逐坐标相等的充要条件是式 (9) 的右侧对所有 $i$ 为零。对于 $x_i\ne x_{0i}$ 的坐标，这等价于

$$
\frac{\partial s(x)}{\partial x_i}
=\int_0^1
\frac{\partial s(x_\alpha)}{\partial x_i}\,\mathrm d\alpha.
$$

路径 Jacobian 恒定会保证这个等式，因而是充分条件，不是必要条件。即使 Jacobian 变化，路径平均梯度仍可能因抵消而等于终点梯度；$x_i=x_{0i}$ 的坐标则在两种方法中都归零。

式 (9) 比较的不是主定义式 (1)。Canonical Endpoint 使用 seed $x$，一般 baseline IG 使用 seed $x-x_0$；它们逐坐标相等的条件是

$$
x_i\frac{\partial s(x)}{\partial x_i}
=(x_i-x_{0i})
\int_0^1
\frac{\partial s(x_\alpha)}{\partial x_i}\,\mathrm d\alpha.
$$

即使路径 Jacobian 恒定，当 $x_0\ne0$ 时该等式也未必成立。当 $x_0=0$ 时，canonical Endpoint 与 baseline-relative endpoint 使用相同 seed，比较退化为终点梯度与路径平均梯度的比较。网络架构决定 Jacobian 的内容和路径变化，不决定归因规则。

## 3. 原始 NNS-CNN 的数学身份

本节依据本地论文整理版 [第 3 节：所提方法](<../../相关文章/Understanding neural network through neuron level visualization/zh/03-所提方法.md>) 和 [第 4 节：实验](<../../相关文章/Understanding neural network through neuron level visualization/zh/04-实验.md>)。

### 3.1 Linear/FC

对

$$
z=Wh+b,
$$

原论文式 (4)–(5) 用连接权重作为 learning-image contribution。若 $h$ 已有 learning-image 矩阵，则

$$
\mathcal L(z;x)=W\mathcal L(h;x).
\tag{10}
$$

这正是线性层 Jacobian $J_{z,h}=W$ 的传播。偏置 $b$ 参与前向值，但原公式没有为它定义输入空间 learning image。

### 3.2 ReLU

原论文式 (6)、(9)、(10) 在当前神经元状态上使用

$$
S(z_j)=\mathbf 1(z_j>0).
$$

因此

$$
\mathcal L(\operatorname{ReLU}(z);x)
=D_z\mathcal L(z;x),
\qquad
D_z=\operatorname{diag}\bigl(\mathbf 1(z>0)\bigr).
\tag{11}
$$

在非边界点，这就是 ReLU 的终点 Jacobian。若 $z_j=0$，数学导数不唯一；当前实现采用 PyTorch 的次梯度约定 $\operatorname{ReLU}'(0)=0$。

### 3.3 Convolution

卷积可写为共享参数的稀疏线性映射

$$
z=T_Kh+b,
$$

其中 $T_K$ 是由卷积核 $K$ 生成的 Toeplitz 型矩阵。原论文第 3.3 节把卷积转换成线性连接，因此

$$
\mathcal L(z;x)=T_K\mathcal L(h;x).
\tag{12}
$$

权重共享只改变 $T_K$ 的结构，不改变终点链式法则。

### 3.4 Pooling

原论文描述 pooling 从上一层筛选神经元，被选中神经元直接继承 learning image。对当前 winner 唯一的 MaxPool，局部 Jacobian 是选择矩阵 $P_x$：

$$
\mathcal L(\operatorname{MaxPool}(h);x)
=P_x\mathcal L(h;x).
\tag{13}
$$

这与“当前 winner 路由”一致。出现并列最大值时仍需指定实现的次梯度/索引选择规则。

### 3.5 原始 NNS-CNN 单神经元的条件等价定理

考虑由 affine/conv、ReLU、当前 winner MaxPool 和固定线性读出组成的有向无环网络。输入初始化为 $\operatorname{diag}(x)$，每个非线性算子使用与模型自动微分一致的当前点局部 Jacobian。对当前输入处可导的标量目标 $s(x)$，有

$$
\boxed{
L_s^{\mathrm{NNS\text{-}CNN}}(x)
=x\odot\nabla_xs(x)
}.
\tag{14}
$$

采用第 2 节的行向量约定，同一个结论可写为

$$
\mathcal L(s;x)
=\nabla_xs(x)^\top\operatorname{diag}(x).
$$

证明是把式 (10)–(13) 的局部矩阵沿当前活动计算图相乘，再右乘 $\operatorname{diag}(x)$。因此，等价的直接来源是原论文的输入值 seed 与终点门控规则。

原论文的特殊情形在式 (6) 中明确给当前目标单元应用 ReLU 门，一般情形又在式 (10) 定义 $c_{pq}^{(l-1,l)}=S(x_q^l)w_{pq}^l$；但式 (11) 的末端因子记作 $W_q^l$，没有再次写成 $C_q^{(l-1,l)}$。式 (14) 采用前两处已经明确的规则：凡目标是 ReLU 后单元，其当前门也包含在传播链中。若把式 (11) 的 $W_q^l$ 字面解释为无条件省略目标门，则未激活目标单元也可能保留非零图，此时不再等于该 ReLU 后响应的 Gradient × Input。

该定理描述原论文式 (1)–(14) 在门控一致解释下的单神经元生成规则；它不是关于论文中所有后续聚合、阈值和显示操作的统称。原论文未提供可定位的官方代码仓库，本项目目前复现的是 paper-formula identity。

### 3.6 原论文的滤波器聚合不是简单的“神经元求和”

原论文第 4.2.1 节还定义了滤波器级图。设 feature map $k,l$ 在位置 $p=(r,c)$ 的激活为 $h_p^{k,l}(x)$，该位置神经元的 learning image 为 $L_{h_p}$。论文使用

$$
I^{k,l}
=\sum_p h_p^{k,l}(x)L_{h_p}(x).
\tag{15}
$$

若单神经元图满足式 (14)，则

$$
\begin{aligned}
I^{k,l}
&=\sum_p h_p(x)\left[x\odot\nabla_x h_p(x)\right]\\
&=x\odot\nabla_x
\left[
\frac12\sum_p h_p(x)^2
\right].
\end{aligned}
\tag{16}
$$

所以，该滤波器聚合可以解释为 feature-map energy

$$
s_{\rm energy}(x)=\frac12\lVert h^{k,l}(x)\rVert_2^2
$$

的 endpoint learning image。它不同于当前 Scanner 的默认通道和目标

$$
s_{\rm sum}(x)=\sum_p h_p^{k,l}(x),
\qquad
L_{\rm sum}=x\odot\sum_p\nabla_x h_p(x).
\tag{17}
$$

若要在 ViT 中逐式复现原论文的滤波器级聚合，对应对象应是某个 channel 在全部 patch tokens 上的二次能量目标，而不是未加激活权重的 channel sum。当前 Scanner 已提供单 token-channel 和 channel-sum 目标；数学上还应把 energy target 定义为第三类聚合目标，当前 `EndpointTarget` 注册器尚未实现这个 `kind`。

原论文第 4.2.1 节把 $f_{(x,y)}^{k,l}$ 称为 feature map 中的 activation value，但本地方法与实验文本没有进一步声明该聚合使用 ReLU 前张量还是 ReLU 后张量。因此，式 (16) 中的 $h_p$ 严格表示论文实际用于式 (15) 加权、且与 $L_{h_p}$ 对应的同一个激活标量。选择 pre-activation、post-ReLU 或其他归一化后的张量会定义不同的 energy target；只要权重和 learning-image 目标使用同一个 $h_p$，式 (16) 的代数关系不变。

### 3.7 Bias、常数与完整性

对 affine 标量

$$
s(x)=w^\top x+b,
$$

Endpoint learning image 为

$$
L_s=x\odot w,
\qquad
\sum_iL_s[i]=w^\top x=s(x)-b.
\tag{18}
$$

因此，原 NNS-CNN 的单神经元传播和 Endpoint Scanner 都把 bias 留在模型状态中，却不把它直接分配到图像像素。bias 还会改变 ReLU 门、pooling winner 和后续 Jacobian。

对于零偏置、正一阶齐次的 ReLU 网络，Euler 定理给出

$$
s(x)=x^\top\nabla s(x),
$$

此时 Endpoint 像素和可等于响应，且从零点沿正射线路径的 IG 也与 Endpoint 相等。这个特殊结论依赖正齐次性；带 bias 的 CNN 和包含 LayerNorm、GELU、softmax 的 ViT 一般不满足它。

## 4. 此前项目中的 NNS-ViT

此前“NNS-ViT”在项目中覆盖了几种相关但不同的路径对象。共同点是：选定 ViT 内部标量或最终类别，把输入坐标上的贡献图称为 learning image。它们的数学定义如下。

### 4.1 Internal-target path-integrated learning image

给定内部单元或聚合响应 $h_u(x)$：

$$
L_u^{\mathrm{internal\text{-}IG}}[i]
=(x_i-x_{0i})
\int_0^1
\frac{\partial h_u(x_\alpha)}{\partial x_i}
\,\mathrm d\alpha.
\tag{19}
$$

在数值积分收敛且路径可微时：

$$
\sum_iL_u^{\mathrm{internal\text{-}IG}}[i]
=h_u(x)-h_u(x_0).
\tag{20}
$$

这类图适合解释“某个内部响应相对参考输入发生了什么变化”。它是 IG 在内部目标上的直接应用。

### 4.2 Conditional fixed-attention Value-path IG

Stage B/C 的条件目标先在 clean 输入捕获 attention matrix $A_{\rm clean}$，然后固定它，只让 Value 路径随输入改变。一个典型目标可写为

$$
s_V(x)=u^\top
\left[A_{\rm clean}V(x)\right]_{t,h}.
\tag{21}
$$

对应 learning image 是 $s_V$ 的 IG：

$$
L_V^{\mathrm{conditional}}[i]
=(x_i-x_{0i})
\int_0^1
\frac{\partial s_V(x_\alpha)}{\partial x_i}
\,\mathrm d\alpha.
\tag{22}
$$

因为 $A_{\rm clean}$ 被固定，式 (22) 不含 Query/Key 对 routing 的输入依赖。它回答“在 clean routing 固定时，Value 内容来自哪些输入位置”，而不是“完整 self-attention 如何依赖输入”。

### 4.3 Input-resolved complete-cut path conductance

给定最终类别目标 $F(x)$ 和一个完整计算切面

$$
C=\{z_1,\ldots,z_m\},
$$

此前 NNS-v2 定义

$$
K^C_{j,i}
=(x_i-x_{0i})
\int_0^1
\left\langle
\nabla_{z_j}F(x_\alpha),
\frac{\partial z_j(x_\alpha)}{\partial x_i}
\right\rangle
\,\mathrm d\alpha.
\tag{23}
$$

这里 $z_j$ 可以是标量、向量或张量；内积遍历该成分的全部 token、channel、head 或其他内部坐标。若把每个标量坐标分别注册为成分，式 (23) 退化为标量偏导的乘积形式。若 $F$ 与切面变量沿路径几乎处处可微，并使用固定且一致的次梯度约定，而且 $C$ 在每个路径点都是针对 $F$ 的完整、不重叠计算边切面，使每条输入到目标的输入依赖路径恰好穿过一个注册边，则

$$
\sum_{j=1}^mK^C_{j,i}
=\operatorname{IG}_i(F;x,x_0).
\tag{24}
$$

每个 $K^C_{j,:}$ 是同一最终类别 IG 在该切面内分配给成分 $j$ 的输入坐标图；对输入坐标求和才得到常见的成分级 conductance 标量。不同层的完整切面是同一总 IG 的不同分解视图，不能跨层相加。

### 4.4 此前 NNS-ViT 与原始 NNS-CNN 的关系

此前 NNS-ViT 保留了三项 NNS 思路：

- 为内部对象定义输入坐标图；
- 区分单元、head、branch 和输出目标；
- 在统一输入坐标中比较或组合同一切面内的成分。

它改变了核心归因算子：输入 seed 从 $x$ 变成 $x-x_0$，当前点 Jacobian 变成沿 $x_0\rightarrow x$ 的平均 Jacobian。故它是 NNS 组织方式与 IG/path conductance 的结合，不是原论文终点公式的直接实现。

## 5. Endpoint Scanner 对 ViT 算子的延伸

Endpoint Scanner 不为 ViT 算子设计新的启发式传播规则，而是使用每个算子在当前真实状态上的完整 Jacobian。

### 5.1 Patch embedding

将一个 patch 展平为 $p$，patch embedding 为

$$
e=pW_E+b_E.
$$

因此

$$
\mathcal L(e;x)=W_E^\top\mathcal L(p;x)
\tag{25}
$$

（具体左右乘取决于张量布局）。卷积形式的 patch projection 与原 NNS-CNN 的 convolution 传播完全同构。

### 5.2 CLS token 与位置编码

ViT 初始序列为

$$
H_0=[c_{\rm CLS};E(x)]+P.
\tag{26}
$$

$c_{\rm CLS}$ 与位置编码 $P$ 是模型常数，其直接输入 learning image 为零：

$$
\frac{\partial c_{\rm CLS}}{\partial x}=0,
\qquad
\frac{\partial P}{\partial x}=0.
\tag{27}
$$

它们仍进入后续 LayerNorm、attention 和 MLP 的真实状态，从而改变下游 Jacobian。

### 5.3 LayerNorm

对宽度为 $D$ 的向量 $u$：

$$
\mu=\frac1D\sum_j u_j,
\qquad
\sigma=\sqrt{\frac1D\sum_j(u_j-\mu)^2+\epsilon},
$$

$$
y_i=\gamma_i\frac{u_i-\mu}{\sigma}+\beta_i.
$$

其局部 Jacobian 为

$$
\frac{\partial y_i}{\partial u_j}
=\frac{\gamma_i}{\sigma}
\left[
\delta_{ij}-\frac1D
-\frac{\hat u_i\hat u_j}{D}
\right],
\qquad
\hat u_i=\frac{u_i-\mu}{\sigma}.
\tag{28}
$$

式 (28) 包含均值、方差和通道耦合。只乘 $\gamma/\sigma$ 会漏掉两条归一化路径。

该式对应 PyTorch LayerNorm 的总体方差（biased variance）约定，即方差分母为 $D$；使用其他方差分母时，相应 Jacobian 系数也会改变。Affine offset $\beta$ 对当前输入 $u$ 的局部导数为零，但它参与后续算子的真实状态并影响后续 Jacobian。

### 5.4 GELU

对 $y=\operatorname{GELU}(u)$：

$$
\mathcal L(y;x)
=\operatorname{diag}\bigl(\operatorname{GELU}'(u)\bigr)
\mathcal L(u;x).
\tag{29}
$$

它对应原 NNS 中 ReLU 二值门的连续推广，但使用的是模型实际 GELU 实现的导数。

### 5.5 Self-attention

单个 head 的前向为

$$
Q=UW_Q,
\qquad K=UW_K,
\qquad V=UW_V,
$$

$$
S=\frac{QK^\top}{\sqrt{d_h}}+B,
\qquad A=\operatorname{softmax}(S),
\qquad O=AV.
\tag{30}
$$

完整终点微分为

$$
dO=(dA)V+A(dV),
\tag{31}
$$

$$
dA
=J_{\rm softmax}(S)
\left[
\frac{(dQ)K^\top+Q(dK)^\top}{\sqrt{d_h}}+dB
\right].
\tag{32}
$$

式 (31)–(32)产生 Value、Query routing、Key routing，以及存在输入相关 bias 时的 bias-routing 路径。完整 attention 目标必须包含这些路径；固定 $A$ 的 Value-path 只保留 $A(dV)$。

$J_{\rm softmax}$ 对每个 query row 独立作用。若 $B$ 是固定位置偏置，则 $dB=0$；只有输入相关 bias 才产生输入归因路径。完整 ViT attention 传播还包括多头 concat、output projection $W_O$ 和后续网络；projection bias 的直接输入导数为零，但它仍改变下游状态。当前实现固定 `eval()` 模式，因此 attention dropout 与 drop path 关闭。

### 5.6 Residual

对

$$
r=u+a(u),
$$

目标是 residual 输出自身的单元 $r_j$ 时：

$$
\mathcal L(r_j;x)
=\mathcal L(u_j;x)+\mathcal L(a_j;x).
\tag{33}
$$

若固定的是更下游目标 $F(r)$，令

$$
g_r=\nabla_rF,
$$

则两条输入边的 endpoint 成分为

$$
E_{{\rm id},i}
=x_i\left\langle g_r,\frac{\partial u}{\partial x_i}\right\rangle,
\qquad
E_{{\rm attn},i}
=x_i\left\langle g_r,\frac{\partial a}{\partial x_i}\right\rangle.
\tag{34}
$$

若 $F$ 是 $r$ 的下游函数，且从当前输入到 $F$ 的所有输入依赖路径都经过这个 residual add，则

$$
E_{\rm id}+E_{\rm attn}
=x\odot\nabla_xF(r(x)).
\tag{35}
$$

identity 成分使用加法器输出处的 $g_r$。若直接把总导数 $\nabla_uF$ 当作 identity，再加 attention branch，就会重复计算 $u\rightarrow a(u)\rightarrow r$ 路径。

原 NNS 论文在残差网络上展示了 block 前、中、后的 learning images，但方法节没有单独给多输入加法器的传播式。式 (33)–(35)把这一算子合同写明，并区分“residual 输出单元自身的图”和“某条 residual 输入边对更下游目标的成分图”。

## 6. Structured endpoint decomposition

给定最终目标 $F(x)$ 和一个完整计算边切面 $C=\{z_1,\ldots,z_m\}$，定义

$$
E^C_{j,i}
=x_i
\left\langle
\nabla_{z_j}F(x),
\frac{\partial z_j(x)}{\partial x_i}
\right\rangle.
\tag{36}
$$

若 $F$ 在当前点可微，或所有非光滑算子采用固定且一致的次梯度约定，并且 $C$ 满足以下条件：

1. 每条从输入 $x_i$ 到目标 $F$ 的输入依赖路径恰好穿过一个注册切面边；
2. 不存在绕过切面的输入依赖路径；
3. 同一计算路径没有被两个成分重复覆盖；
4. residual 等共享源节点使用显式加法器输入边，而非已经汇合多条下游路径的原始节点总梯度；
5. 切面包含全部输入相关路径；模型常数无需作为像素成分，因为它们对 $x$ 的直接导数为零；

则

$$
\boxed{
\sum_{j=1}^mE^C_{j,i}
=x_i\frac{\partial F(x)}{\partial x_i}
}.
\tag{37}
$$

### 6.1 Computational edge cuts

Residual 和 multi-head 分解直接对应计算图输入边：

| 切面 | 成分 | 闭合目标 |
|---|---|---|
| 第一次 residual edge cut | identity + attention | 完整类别 endpoint 图 |
| 第二次 residual edge cut | carry + MLP | 完整类别 endpoint 图 |
| Multi-head edge cut | identity + 所有 projection 前 head slices | 完整类别 endpoint 图 |

每个 head 的下游 cotangent 包含原生 concat、output projection 和后续网络。

### 6.2 Attention differential-path decompositions

Q/K/V 和 routing/value 来自乘法与 softmax 的全微分：

| 微分路径分解 | 成分 | 闭合目标 |
|---|---|---|
| Q/K/V | identity + Query + Key + Value | 完整类别 endpoint 图 |
| routing/value | identity + (Query + Key) + Value | 完整类别 endpoint 图 |

`routing = Query + Key` 是一个合并成分；它不能与 Query、Key 再次相加。本文用 **structured endpoint decomposition** 统称 computational edge cuts 与 attention differential-path decompositions。这里的“完整”始终相对于注册目标、指定 block 和精确 hook edge；每一行独立重构同一总图，不同切面、不同 block 或两种 attention 分解不能彼此相加。

式 (36) 与此前的式 (23) 采用对称的向量/张量成分写法，核心差异是 input seed 和 Jacobian 的求值范围：

$$
\begin{array}{lll}
\text{Endpoint cut:}
&x_i
\left\langle\nabla_{z_j}F(x),\partial z_j(x)/\partial x_i\right\rangle,
&\text{单个终点};\\[4pt]
\text{Path cut:}
&(x_i-x_{0i})\int_0^1
\left\langle\nabla_{z_j}F(x_\alpha),\partial z_j(x_\alpha)/\partial x_i\right\rangle
\mathrm d\alpha,
&\text{整条路径}.
\end{array}
\tag{38}
$$

## 7. 输入坐标与“隐式原点”

原始 NNS 的 seed 是送入模型的数值。因此，Endpoint attribution 依赖输入坐标原点。

设原始 RGB 坐标为 $r$，模型归一化输入为

$$
x=\frac{r-\mu}{\sigma}.
\tag{39}
$$

链式法则给出

$$
\frac{\partial s}{\partial r_i}
=\frac1{\sigma_i}
\frac{\partial s}{\partial x_i}.
\tag{40}
$$

于是规范化输入上的 Endpoint 为

$$
x_i\frac{\partial s}{\partial x_i}
=\frac{r_i-\mu_i}{\sigma_i}
\frac{\partial s}{\partial x_i}
=(r_i-\mu_i)
\frac{\partial s}{\partial r_i}.
\tag{41}
$$

所以，虽然 Endpoint Scanner 不执行 baseline forward，也不沿 baseline 路径积分，但归一化坐标的零点对应 RGB 坐标中的 $r=\mu$。当前 canonical Scanner 使用实际模型输入，因此其图在 raw-RGB 坐标下等价于“相对 RGB mean 原点的 endpoint”，而不是 $r\odot\nabla_rs$。

相比之下，IG 在一致的 affine 坐标变换和对应 baseline 下满足

$$
\begin{aligned}
&(x-x_0)\odot
\int_0^1\nabla_xs(x_\alpha)\,\mathrm d\alpha\\
&\qquad=
(r-r_0)\odot
\int_0^1\nabla_rs(r_\alpha)\,\mathrm d\alpha.
\end{aligned}
\tag{42}
$$

这里 $x_0=(r_0-\mu)/\sigma$，且两条路径表示同一组物理输入。这个关系说明相对差值会抵消 affine 坐标缩放，而式 (41) 的 Endpoint 会保留坐标原点的选择。

原 NNS-CNN 论文的热图和“激活像素数量/强度”解释主要建立在其输入表示与实验后处理上。若 ViT 使用含负值的规范化输入和 signed learning images，这些指标不能不经重新定义直接搬用。

原论文方法节对 RGB 输入说明按通道拆分坐标。Endpoint Scanner 同样把 `[C,H,W]` 中每个 RGB 标量作为独立输入坐标；报告中的二维图是对原始 signed RGB tensor 做通道求和后的显示结果。先取绝对值、正部或逐图归一化都会改变比较对象。

## 8. Endpoint 与 Path 的性质比较

下表比较的是归因定义。实际数值比较还必须固定同一个标量目标、输入坐标、预处理、reference（Path 方法）、RGB 聚合和显示前张量；若其中任一项不同，地图差异不能归因于 Endpoint/Path 规则本身。

| 性质 | 原始 NNS-CNN 单神经元 | NNS-ViT Endpoint Scanner | 此前 internal-target IG | NNS-v2 input-resolved path conductance |
|---|---|---|---|---|
| 输入 seed | $x$ | 实际模型输入 $x$ | $x-x_0$ | $x-x_0$ |
| Jacobian 位置 | 当前输入终点 | 当前输入终点 | $x_0\rightarrow x$ 路径 | $x_0\rightarrow x$ 路径 |
| 基础量 | 条件下为 $x\odot\nabla s(x)$ | $x\odot\nabla s(x)$ | $\operatorname{IG}(s;x,x_0)$ | 最终目标 IG 的成分分解 |
| 典型目标 | CNN 单元/获胜输出 | token-channel、聚合响应、head、logit、margin | 相同类型内部标量 | 固定最终类别目标 |
| Attention | 原模型无此算子 | 完整 Q/K/V routing + Value | 沿路径的完整 attention Jacobian，除非明确固定 $A$ | 完整路径切面 |
| Bias/常数 | 影响门，不直接分配到像素 | 影响状态/Jacobian，不直接分配到像素 | 通过响应差和路径状态体现 | 同左，并在完整切面中记直接路径为零 |
| 响应 completeness | 一般无 | 一般无 | 数值收敛时有 | total IG 与每个完整切面均要求闭合 |
| 计算结构 | 显式终点传播；全扫描存储随“单元数 × 输入坐标数”增长 | 一次共享前向 + 每目标一次 VJP | 多个路径点前向/反向 | 多路径点 × 多成分 VJP |
| 与 canonical $x\odot\nabla s(x)$ 的关系 | 门控一致等条件下相等 | 恒等 | 满足第 2.3 节逐坐标条件时相等；Jacobian 恒定只在 $x_0=0$ 时直接给出该结论 | total 等于 IG；与 canonical Endpoint 的关系继承左侧逐坐标条件 |
| 原始科学张量 | 输入坐标 learning image；论文另有热图后处理 | signed `[C,H,W]` | signed `[C,H,W]` | total 与 component signed maps |

### 8.1 一个最小反例

令

$$
s(x)=\operatorname{ReLU}(x-1),
\qquad x_0=0,
\qquad x=2.
$$

Endpoint 使用终点门：

$$
L^{\mathrm{Endpoint}}
=2\,s'(2)=2.
\tag{43}
$$

IG 记录路径上门从关闭到打开：

$$
L^{\mathrm{Path}}
=2\int_0^1\mathbf 1(2\alpha-1>0)\,\mathrm d\alpha
=1.
\tag{44}
$$

其中 $1=s(2)-s(0)$，而 Endpoint 的 2 是把终点局部斜率乘到当前坐标值。差异来自归因定义和 bias 阈值，不来自 CNN/ViT 架构标签。

### 8.2 ViT 中差异通常更普遍

ViT 的 LayerNorm、GELU 和 softmax attention 使

$$
\nabla s(x_\alpha)
$$

沿路径连续变化。Canonical Endpoint 与一般 baseline Path IG 通常满足

$$
x\odot\nabla s(x)
\neq
(x-x_0)\odot\int_0^1\nabla s(x_\alpha)\,\mathrm d\alpha.
\tag{45}
$$

当目标局部接近线性时，两者仍可接近。例如固定 clean attention 后的 Value-path 主要由线性投影和较平滑的上游变换构成，本项目单图对照中 Endpoint/Path Pearson 为 0.9987；完整 classifier logit 的对应 Pearson 为 0.6872。

## 9. 与原 NNS-CNN 的逐项对应

| 原 NNS-CNN 对象 | Endpoint Scanner 对象 | 共同数学结构 | ViT 特有变化 |
|---|---|---|---|
| 输入像素 learning image $x_pe_p$ | 规范化 RGB 坐标 $x_pe_p$ | $\operatorname{diag}(x)$ | 坐标零点由 normalization 决定 |
| 卷积位置神经元 | patch token-channel | 单个内部标量 $s(x)$ | token 含 embedding、全局 mixing 和 residual 历史 |
| convolution 权重传播 | patch projection Jacobian | 共享线性映射 | patch stride 通常不重叠 |
| ReLU 当前门 | GELU 当前导数 | 非线性终点 Jacobian | 连续导数，可为小值或负值 |
| pooling winner | attention routing | 输入相关路由 | softmax 是稠密且依赖 Q/K，不能当固定选择器 |
| filter feature-map aggregate | channel/token aggregate | 多单元目标 | 原论文第 4.2.1 节的未编号聚合式对应 energy；当前默认另有 sum target |
| skip/residual 前后神经元图 | identity/attention、carry/MLP edge cuts | 加法器处 Jacobian 相加 | 必须包含两个 ViT residual add |
| 输出获胜神经元 | class logit | 输出标量的输入 learning image | final norm、CLS 与 classifier 全部参与 |
| 热图显示 | 颜色映射；与 CAM 对比时另做 max-pool 与插值 | 原始 signed RGB 先保存，显示另行降维/缩放 | 数学比较必须回到显示前张量 | 显示不是归因定义的一部分 |

最接近原论文 Figure 7 的 ViT 扫描对象是固定 `(block, token, channel)` 的单元 learning image；最接近原论文 filter aggregate 的对象是固定 `(block, channel)` 的 token-energy learning image；class logit 图对应原论文的输出神经元图。Head、Q/K/V 和完整切面是 ViT 计算图带来的新增结构索引。

## 10. 当前实现与数值证据

核心实现位于 [endpoint_vit.py](../src/vitnns/endpoint_vit.py)，正式定义与复现命令见 [Endpoint Scanner 说明](endpoint_nns_vit_scanner_20260918.md)。它有三个层次：

1. `scan_endpoint_vit`：一次真实前向，注册 token/channel/head/logit/margin 目标；每个目标返回原始 signed $x\odot\nabla s$；
2. `attention_endpoint_components`：对一个内部 head 标量分解 Query、Key、Value；
3. `decompose_endpoint_vit`：固定同一个类别 logit 或 margin，用 residual/head computational edge cuts 以及 Q/K/V、routing/value differential paths 分解总 Endpoint 图。

单图演示使用冻结的 ViT-B/16 和 `ILSVRC2012_val_00007922.JPEG` 的既有 RGB crop，结果如下：

| 验证项 | 结果 |
|---|---:|
| 注册目标 | 40 |
| native logits 与无 hooks 前向最大误差 | 0 |
| 类别 Endpoint 与独立 Gradient × Input 最大误差 | 0 |
| 内部 head Q/K/V 闭合最大误差 | $5.93\times10^{-21}$ |
| 聚合响应 residual 闭合最大误差 | $7.06\times10^{-14}$ |
| 同一类别 structured decomposition 闭合最大误差 | $4.44\times10^{-16}$ |
| Structured decomposition 最大相对 L2 误差 | $3.14\times10^{-15}$ |
| 40 目标扫描，CPU float64、4 threads | 4.32 s |

式 (7)、式 (14) 和式 (37) 的数学身份来自链式法则。上表是实现验证：它检查当前代码的 hooks、VJP cotangents、目标选择和 component bookkeeping 是否满足这些定义。表中的“独立 Gradient × Input”使用另一个调用路径，但仍依赖同一 PyTorch autograd 图，因此属于 parity test，而非对链式法则的独立数学证明。闭合误差也不衡量归因 fidelity 或成分的因果效应。

对应报告为 [scan_report.json](../reports/endpoint_nns_vit_scan_r3_20260918/scan_report.json)，原始数组和 Image #1–#9 的解释见 [逐图说明](endpoint_nns_vit_scanner_figures_20260918.md)。

此前 20 图 Endpoint/Path 探索性比较使用两个参考张量：normalized-zero（RGB 0.5）和 normalized-minus-one（RGB 0）。对 normalized-zero 条件，baseline-relative endpoint 为

$$
(x-0)\odot\nabla s(x)=x\odot\nabla s(x),
$$

因此它在该条件下与 canonical Endpoint 数值相同。下表的 fidelity、flip 和 runtime 全部来自这个 normalized-zero/canonical 条件：

| 指标 | Canonical Endpoint = zero-relative endpoint | Zero-baseline Path IG |
|---|---:|---:|
| 4×4 combined fidelity 中位数 | 0.03639 | 0.10143 |
| 16×16 combined fidelity 中位数 | 0.01161 | 0.02120 |
| flip Spearman 中位数 | 0.34104 | 0.47416 |
| Endpoint/Path 运行时间比中位数 | 0.01492 | 1 |

所谓 reference gate 另行比较了两个 baseline-relative extensions：

$$
L^{\Delta,0}=x\odot\nabla s(x),
\qquad
L^{\Delta,-1}=(x+1)\odot\nabla s(x).
$$

其中第一项恰好是 canonical Endpoint，第二项是 black-RGB reference 下的扩展。记录的 reference Pearson 中位数 $-0.07869$ 衡量这两个 multiplier 定义之间的一致性；它不是 canonical Endpoint 自身具有 reference 参数或 reference sensitivity。

该协议的决策为 `NO_GO_CURRENT_ENDPOINT_PROTOCOL`：normalized-zero/canonical Endpoint 通过了 4×4 positive-fidelity 门和 runtime 门，但没有通过相对 Path IG 的 4×4 非劣启发式门与 flip 门；额外的 two-multiplier comparison 也没有通过预设 reference gate。因此，当前证据支持把 Scanner 用作终点内部分析工具，不支持用它替换 path-integrated learning image 作为论文主归因方法。完整结果见 [20 图决策文档](endpoint_nns_vit_level4_decision_20260917.md)。

## 11. 推荐的术语体系

后续文档和论文应避免单独使用含义不明的“NNS-ViT”。推荐固定以下名称：

| 名称 | 固定含义 |
|---|---|
| `NNS-CNN single-neuron endpoint learning image` | 原论文式 (1)–(14) 的单神经元图 |
| `NNS-ViT Endpoint Scanner` | 实际模型输入上的 $x\odot\nabla s(x)$ 内部扫描器 |
| `baseline-relative endpoint extension` | $(x-x_0)\odot\nabla s(x)$；不是主 Endpoint 定义 |
| `internal-target IG` / `path-integrated learning image` | 式 (19) |
| `conditional fixed-attention Value-path IG` | 式 (22) |
| `input-resolved complete-cut path conductance` | 式 (23)；保留 component-by-input map |
| `structured endpoint decomposition` | 式 (36)；含 computational edge cuts 与 attention differential paths |

### 11.1 可用于方法部分的准确表述

> NNS-ViT Endpoint Scanner 将原始 NNS 的输入值初始化和当前状态传播原则扩展到 ViT。对任意注册的内部标量或输出目标，它在真实输入处传播 patch embedding、LayerNorm、GELU、完整 self-attention、MLP 和 residual 的局部 Jacobian，得到 $x\odot\nabla_xs(x)$。Structured endpoint decomposition 进一步用 residual/head computational edge cuts 和 attention differential paths 分解同一终点图，并分别重构总图。

### 11.2 方法贡献的边界

Endpoint Scanner 的基础 attribution 是 Gradient × Input。方法增量由以下可核对对象构成：

- 与原 NNS 对齐的 ViT 单元目标注册；
- 真实 native-forward 中间张量扫描；
- attention 的 Query、Key、Value 和 routing 微分路径分解；
- 两次 residual 的输入边记账；
- 同一最终类别目标下的多 head 完整计算边切面；
- 原始 signed RGB maps、成分账本和数值闭合验证。

此前 path-integrated NNS-ViT 继续承担 reference-relative attribution、响应差 completeness 和 path conductance 分解。两套方法回答不同问题：Endpoint 描述当前模型状态的局部一阶输入结构；Path 描述从参考输入到当前输入的累计变化。

## 12. 最终关系

对同一个标量目标 $s$，三个核心对象可以写成：

$$
\boxed{
\begin{aligned}
\text{原始 NNS-CNN 单元图}
&\overset{\text{条件}}{=}
x\odot\nabla s_{\rm CNN}(x),\\
\text{NNS-ViT Endpoint Scanner}
&=
x\odot\nabla s_{\rm ViT}(x),\\
\text{此前 path-integrated NNS-ViT}
&=
(x-x_0)\odot
\int_0^1\nabla s_{\rm ViT}(x_\alpha)\,\mathrm d\alpha.
\end{aligned}
}
\tag{46}
$$

前两项共享终点传播原则，差别来自 CNN 与 ViT 的 Jacobian；第三项改变了归因定义，因此等于 IG。NNS-ViT Endpoint Scanner 对原始 NNS 最忠实的部分，是输入值 seed、当前状态 Jacobian、内部单元扫描和输入坐标保留；它对 ViT 新增的部分，是完整 attention、LayerNorm/GELU、双 residual 与结构化分解的显式记账。

## 参考实现与材料

- [原 NNS 论文中文整理：所提方法](<../../相关文章/Understanding neural network through neuron level visualization/zh/03-所提方法.md>)
- [原 NNS 论文中文整理：实验](<../../相关文章/Understanding neural network through neuron level visualization/zh/04-实验.md>)
- [Endpoint 方法身份核验](endpoint_nns_vit_method_identity_20260917.md)
- [Endpoint Scanner 实现说明](endpoint_nns_vit_scanner_20260918.md)
- [此前 NNS-ViT normal-forward / path learning-image 对照](ViT正常前向与NNS学习图像逐步对照_20260917.md)
- [NNS-v2 path complete-cut 协议](stage_x_protocol.md)
- [Endpoint Scanner 代码](../src/vitnns/endpoint_vit.py)
- [IG 代码](../src/vitnns/attribution.py)
- [Path complete-cut 代码](../src/vitnns/stage_x_nns_v2.py)
- [本文的学术措辞审计](NNS_ViT_Endpoint_Scanner数学分析写作审计_20260918.md)
