# 3. 方法

## 3.1 问题定义与 learning image

令 $x\in\mathbb{R}^{3\times224\times224}$ 为规范化后的图像，$x_0$ 为规范化空间中的全零参考，对应 RGB 灰色 0.5。给定模型中的标量目标 $s(x)$，本文把该目标相对参考的 learning image 定义为

$$
L_s[i]=(x_i-x_{0i})\int_0^1
\frac{\partial s(x_0+\alpha(x-x_0))}{\partial x_i}\,d\alpha.
$$

在积分和数值误差允许的范围内，

$$
\sum_i L_s[i]\approx s(x)-s(x_0).
$$

原始 learning image 保留三个 RGB 通道和符号。二维图先对 RGB 通道做带符号求和；正部、负部、signed-log、P99 截断和颜色映射仅用于显示，不参与 completeness、residual additivity 或统计计算。Figures 15–17 说明输入坐标初始化和图像组合，Figure 2 给出真实 attention 响应的 learning image。

## 3.2 ViT 前向结构

冻结模型接收 $224\times224$ RGB 输入，以 $16\times16$ patch 形成 196 个图像 token，并加入 CLS 与位置编码。第 $b$ 个 transformer block 写为

$$
a_b=\operatorname{Attention}(\operatorname{LN}_1(h_b)),\qquad
r_b^{(1)}=h_b+a_b,
$$

$$
m_b=\operatorname{MLP}(\operatorname{LN}_2(r_b^{(1)})),\qquad
h_{b+1}=r_b^{(2)}=r_b^{(1)}+m_b.
$$

12 个 blocks 后，

$$
u=\operatorname{FinalNorm}(h_{12}),\qquad
\ell=W_{\mathrm{cls}}u_{\mathrm{CLS}}+b_{\mathrm{cls}}.
$$

B0 MLP 的维度为 $768\rightarrow3072\rightarrow768$，使用 GELU。分类器是作用于 final-normalized CLS 的 $\operatorname{Linear}(768,1000)$；它与 block 内 MLP 分支是不同模块。Figures 18 和 24–27 以解析例连接正常前向与 learning image，Figures 28–35 使用真实模型输出。

## 3.3 三类目标

### 3.3.1 模块自身响应

对 patch embedding、attention、MLP 或 residual 输出 $q(x)\in\mathbb{R}^{196\times768}$，固定通道 $c$ 的展示目标为

$$
s_{q,c}(x)=\sum_{t=1}^{196}q(x)_{t,c}.
$$

CLS 参与前向 attention，但不进入这一空间 token 求和。$L_{s_{q,c}}$ 回答该模块响应的变化来自哪些输入位置，不乘最终类别的下游导数。Figures 29–33 固定通道 0、255、511 和 767；通道由索引预先确定。

### 3.3.2 固定 clean attention 的条件 Value-path 目标

主干预实验针对 block $b$、head $h$ 的 CLS head-direction。由 clean forward 捕获 attention $A_{\mathrm{clean}}$，并令

$$
g(z)=\sum_{j=0}^{196}A_{\mathrm{clean}}[\mathrm{CLS},j]V(z)_j,
$$

$$
d=g(x)-g(x_0),\qquad u=\frac{d}{\lVert d\rVert_2},\qquad
f(z)=u^\top g(z).
$$

$A_{\mathrm{clean}}$、$d$ 与 $u$ 在积分前冻结。该目标保留 Value 随输入变化的路径，切断 Q、K 和 softmax attention 沿积分路径的变化，因此称为 fixed-clean-attention conditional Value-path attribution。它不是完整 Q/K/V 归因，也不是完整类别解释。

### 3.3.3 完整类别输出与完整切分成分

完整类别 learning image 直接令 $s(x)=\ell_y(x)$，梯度经过全部 blocks、final norm、CLS 与 classifier。完整切分进一步在模型内部选取一组不重叠且覆盖计算的成分 $z_k$，用输入路径方向导数与下游类别导数形成成分图 $K_k$。同一 cut 内有

$$
\sum_k K_k[i]\approx L_{\ell_y}[i].
$$

不同 cuts 是同一总归因的不同记账视图，不能跨 cut 再次相加。Figures 19–21 展示 head/branch 成分和重构；Figure 34 直接展示完整 class logit。

## 3.4 Attention 与 residual 的数值对应

完整 attention learning image 对输入的导数包含 Value 路径和 routing 路径：

$$
d(AV)=A\,dV+(dA)V.
$$

固定 clean attention 的条件目标删除第二项，而真实模块响应和完整类别 IG 保留两项。Figure 22 用两 token 例子展示两者差异。

对共同输入路径和 reference，residual add 的 learning image 按输入坐标相加：

$$
L_{h+a}=L_h+L_a,\qquad
L_{r^{(1)}+m}=L_{r^{(1)}}+L_m.
$$

加法在原始带符号 RGB 数组上检查，之后才执行显示变换。Figure 23 给出下游类别目标的解析例，Figures 31–33 给出真实 Block 0 的两次 residual。

## 3.5 同目标 patch-token 基线

主实验的对照首次写作 target-matched patch-token baseline with uniform within-patch allocation。对第 $j$ 个图像 patch token，

$$
c_j=A_{\mathrm{clean}}[\mathrm{CLS},j]\,
u^\top\left(V(x)_j-V(x_0)_j\right).
$$

196 个 $c_j$ 形成 $14\times14$ token 空间地图；CLS scalar 单独保留。评价时，每个 $16\times16$ patch 被切成 16 个 $4\times4$ cells，基线令

$$
c_{j,k}^{\mathrm{uniform}}=\frac{c_j}{16},\qquad k=1,\ldots,16.
$$

NNS 与该基线共享模型、图像、内部目标、reference、干预 fractions 和 tie seeds。二者的坐标定义不同，且不假定逐 patch 总量相等。这个比较检验输入归因排序与 token 贡献的均匀空间展开在同一内部目标上的干预用途。它不同于 Stage S 中按 patch 数、位置和形状匹配的 matched occlusion control。

## 3.6 干预与忠实度估计量

对目标 $t$，令 $\Delta=t(x)-t(x_0)$、$\sigma=\operatorname{sign}(\Delta)$，归一化响应为

$$
S(z)=\sigma\frac{t(z)-t(x_0)}{|\Delta|}.
$$

排序分数取 $\sigma M$ 的正部。Insertion 从 reference 按排序恢复 RGB 空间单元，deletion 从 clean input 按相同排序替换为空间 reference。两条曲线使用 fractions $0,0.05,\ldots,1$，每个 fraction 的实际横坐标为替换单元数占全部单元数的比例。AUC 不裁剪响应。

对图像 $i$、方法 $m$、方向 $d$ 和 tie repeat $q$，先对五次配对 repeats 求均值 $\bar A(i,m,d)$，再定义

$$
\Delta_{\mathrm{fid}}(i)=\frac{1}{2}\left[
\bar A(i,\mathrm{NNS},\mathrm{ins})-\bar A(i,\mathrm{base},\mathrm{ins})
+\bar A(i,\mathrm{base},\mathrm{del})-\bar A(i,\mathrm{NNS},\mathrm{del})
\right].
$$

正值表示 NNS 排序在配对 insertion/deletion 定义下更高。统计单位是图像；repeat、curve point、cell 和方法都不增加样本量。

## 3.7 显示规则

正式数值来自原始数组。正贡献蓝图使用正部和逐面板 P99（全零时使用 max）缩放；signed 图使用对称色限或 signed-log。需要比较同一坐标下的 NNS 与基线时使用共同色限。逐面板缩放的图只能比较热点位置和空间组织，不能比较跨面板颜色强度。Figure 35 的四个 classifier 标量使用共同 max-absolute signed-log 色标。
