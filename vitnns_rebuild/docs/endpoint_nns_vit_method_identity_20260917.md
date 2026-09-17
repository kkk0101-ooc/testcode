# Endpoint NNS-ViT 方法身份核验与最小验证协议

日期：2026-09-17

状态：方法学纠偏；不改写 R9，不授权 500 图重算。

## 1. 结论

Dou et al. (2023) 的原始 NNS 不是 Integrated Gradients。论文报告的方法将输入拆成单坐标 learning images，再用层权重和当前样本的 ReLU 激活门逐层传播。在论文明确描述的前馈 ReLU 情形下，这一规则等于端点处的 Gradient × Input：

$$
L_s^{\mathrm{NNS}}[i]=x_i\frac{\partial s(x)}{\partial x_i}.
$$

该等价需要逐项声明条件。原论文没有定义参考输入、路径积分或 completeness 合同，也没有说明如何把 bias 分配回图像坐标。本项目此前生成的模块 learning images 使用标准 IG，应命名为 `path-integrated learning images`，不能作为原始 NNS 公式的直接实现。

本轮建立两个独立对象：

$$
L_s^{\mathrm{end}}[i;x_0]=(x_i-x_{0i})\frac{\partial s(x)}{\partial x_i},
$$

$$
L_s^{\mathrm{path}}[i;x_0]=(x_i-x_{0i})\int_0^1\frac{\partial s(x_0+\alpha(x-x_0))}{\partial x_i}\,d\alpha.
$$

前者是 endpoint NNS-ViT；当 $x_0=0$ 时，它与原论文的输入初始化一致。后者是 internal-target IG，也称 path-integrated learning image。

## 2. 原论文逐项核验

原文依据：[The proposed method](<../../相关文章/Understanding neural network through neuron level visualization/en/03-proposed-method.md>)。

| 核验项 | 原论文报告的规则 | 结论 |
|---|---|---|
| 输入初始化 | 将输入 $I$ 拆成 $I_p=x_pe_p$，且 $I=\sum_p I_p$，见公式 (1)–(3) | 是 $L_{x_p}[i]=x_i\mathbf1(i=p)$ |
| Linear/FC | 将连接权重作为 learning-image contribution，见公式 (4)–(5) | 传播权重 |
| Conv | 将卷积写成共享权重的线性连接，见第 3.3 节 | 传播卷积权重 |
| ReLU | $S(x_q^l)=\mathbf1(x_q^l>0)$，$c_{pq}=S(x_q^l)w_{pq}$，见公式 (6)、(9)、(10) | 使用当前样本端点门 |
| Bias | 前向文字提到 neuron value，但公式 (4)–(13) 的 contribution matrix 只含权重和门 | 未报告 bias 的空间归属 |
| 响应重构 | 输入拆分满足求和；论文没有给一般输出神经元的 bias-aware completeness 定理或误差检查 | 未建立一般重构合同 |
| 输入语义 | learning-image seeds 使用传给网络的输入数值 | 输入值；未定义相对参考值 |
| Pooling | 被池化选中的神经元继承上一层 learning image | 当前 winner 路由 |
| Skip connection | 实验比较 residual block 前、中、后 learning images，但方法节没有给多输入加法的独立公式 | 图示/实验存在，算子合同未形式化 |
| 官方代码 | 本地论文、参考文献和公开检索均未发现作者代码仓库链接 | `NOT_AVAILABLE`；本轮只能复现论文公式 |

因此，本轮 Level 1 的名称是 `paper-formula reproduction`，不是 `official-code reproduction`。

## 3. 条件命题

设无偏置前馈网络由线性层、卷积、ReLU 和固定线性聚合组成。对当前输入 $x$，原始 NNS 使用输入 seeds $x_ie_i$，并在每个 ReLU 使用当前输入处的门 $\mathbf1(z>0)$。若目标标量 $s$ 在 $x$ 处可导，则

$$
\boxed{L_s^{\mathrm{NNS}}[i]=x_i\frac{\partial s(x)}{\partial x_i}}.
$$

证明只需把每层的局部 Jacobian 相乘。Linear/Conv 的 Jacobian 是权重；ReLU 的 Jacobian 是当前门；固定聚合的 Jacobian 是固定系数。链式法则给出 $\partial s/\partial x_i$，再乘输入 seed $x_i$。

该命题不覆盖 ReLU 恰好位于零点的情况。实现采用 PyTorch 的零次梯度约定，即 $\operatorname{ReLU}'(0)=0$，并将这一约定写入测试。

## 4. Bias 与模型常数

对 affine 函数

$$
s(x)=w^Tx+b,
$$

端点图的像素和为

$$
\sum_i L_s^{\mathrm{end}}[i]=w^Tx=s(x)-b.
$$

因此端点图只记录图像输入的一阶项，bias 不会自动出现在图像中。当前协议选择：

- 图像地图只解释输入坐标；
- bias、CLS token、position embedding、LayerNorm affine 参数和 classifier bias 记作模型常数；
- 不把常数贡献均匀铺到像素；
- `attribution_sum - response_delta` 作为诊断值报告，不把零误差作为 endpoint 方法公理。

若未来要求完整数值分解，可引入常数坐标 $\tilde x=(x,1)$。常数贡献必须单列，不能伪装为空间 attribution。

## 5. Endpoint 与 path-integrated 版本

Endpoint 版本只在 $x$ 处取一次 Jacobian：

$$
L_s^{\mathrm{end}}=(x-x_0)\odot\nabla_xs(x).
$$

Path-integrated 版本沿整条路径平均 Jacobian：

$$
L_s^{\mathrm{path}}=(x-x_0)\odot\int_0^1\nabla_xs(x_0+\alpha(x-x_0))d\alpha.
$$

当路径上的 Jacobian 恒定时二者相等。GELU、LayerNorm 和输入相关 softmax attention 通常使 Jacobian 随 $\alpha$ 变化，因此二者一般不同。

| 名称 | 数学对象 | 当前代码位置 |
|---|---|---|
| Endpoint NNS-ViT | endpoint Gradient × Input / input-difference | `src/vitnns/endpoint_nns.py` |
| Path-integrated learning image | internal-target IG | `src/vitnns/attribution.py` 与 R9 模块图 |
| Conditional Value-path IG | 固定 clean attention 的条件内部目标 IG | `src/vitnns/stage_c_value_path.py` |
| Component conductance | full-model IG 的完整切分成分 | `src/vitnns/stage_x_nns_v2.py` |

## 6. ViT 算子适配

统一传播式为

$$
L_y[j,i]=\sum_k\left.\frac{\partial y_j}{\partial u_k}\right|_{u=u(x)}L_u[k,i].
$$

递归展开后就是 endpoint Gradient × Input。它提供统一的内部坐标索引，不构成新的基础归因公式。

Attention 必须使用完整局部微分：

$$
Q=UW_Q,\quad K=UW_K,\quad V=UW_V,
$$

$$
A=\operatorname{softmax}(QK^T/\sqrt d+B),\qquad O=AV,
$$

$$
dO=(dA)V+A(dV),
$$

$$
dA=J_{\mathrm{softmax}}\left(\frac{(dQ)K^T+Q(dK)^T}{\sqrt d}+dB\right).
$$

本轮测试分别计算 Q-routing、K-routing 和 Value 三项，并要求三项端点梯度之和等于完整 attention 梯度。这里的分项是一阶微分分解，不解释为独立干预效应。

完整结构范围为：

```text
pixels -> patch embedding -> position/CLS constants
       -> LN1 -> Q/K/V -> softmax attention -> output projection
       -> first residual
       -> LN2 -> fc1 -> GELU -> fc2
       -> second residual
       -> blocks 1..11 -> final norm -> CLS -> classifier
```

## 7. 四级验证与停止条件

### Level 1：论文公式复现

- 无偏置 Linear–ReLU–Linear：论文权重/门传播等于 Gradient × Input；
- 有偏置反例：明确记录响应重构误差；
- Conv、MaxPool、residual：分别验证权重、winner 和 Jacobian 加法；
- 官方代码未找到时，不签发 official-code parity。

### Level 2：ViT 算子

LayerNorm、GELU、softmax、single-head attention、multi-head attention、MLP 和 residual 的 endpoint 梯度与有限差分对照。

### Level 3：单图端到端

沿同一冻结图像输出 patch embedding、attention branch、first residual、MLP branch、second residual、final CLS 和 classifier logit。每个同名目标比较 endpoint 与 path-integrated learning image。另对 B0/H9 conditional Value-path 做同目标 endpoint/path 对照。

### Level 4：20–50 图小样本

只有 Level 1–3 通过后启动。比较 fidelity、尺度、reference、相关性、符号一致性、稳定性和运行时间。该阶段只签发是否值得进行 500 图重算的决策。

### Go/no-go

继续到小样本需要：

1. 论文公式复现通过；
2. endpoint 实现与独立 Gradient × Input 在浮点容差内一致；
3. ViT 算子有限差分测试通过；
4. 单图链目标、常数边界和地图非退化；
5. R9 封存根保持不变。

进入 500 图重算还需要小样本显示结构化 endpoint scanner 回答了直接输出 Gradient × Input 未回答的问题。仅出现数值不同或视觉不同不足以通过该门。

## 8. 允许的论文定位

若验证通过，可以写：

> 原始 NNS 通过当前输入处的局部门控传播神经元 learning images。本文将这一端点传播原则实现到 ViT 的 LayerNorm、GELU、softmax attention、多头结构和两次 residual，并比较端点图与路径积分扩展。

方法贡献限定为 ViT 结构化扫描、内部对象索引、attention 一阶路径分析和实证问题。`Endpoint NNS-ViT` 不应被描述成独立于 Gradient × Input 的新基础归因算法。

## 9. 表述审计

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| 方法身份 | “NNS-ViT 不是简单的 Gradient × Input” | D14 | Critical | 否 | 用否认替代技术差异 | 要求证明基础公式新颖性 | 删除 | “Endpoint NNS-ViT 使用 Gradient × Input，并提供 ViT 内部对象与结构切面。” |
| Bias | “我们不声称地图恢复全部响应” | D2 | Major | 是 | 重复非主张 | 关注重构为何失败 | 正面定义范围 | “图像地图记录输入一阶项；模型常数单列。” |
| Pilot | “这些结果不能证明总体有效” | D12 | Minor | 是 | 结论前先自我削弱 | 追问样本规模 | 写明研究角色 | “单图 pilot 检查实现和目标定义；总体评价由后续小样本决定。” |

防御性写作评分：2/10。需要保留的边界是 bias/常数不进入空间图、单图不支持总体结论、endpoint 分项不是因果分解。
