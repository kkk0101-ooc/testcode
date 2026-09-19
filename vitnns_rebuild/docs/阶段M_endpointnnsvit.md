# NNS-ViT 后续实验与论文重构方案  
## 目标对齐版

> **当前进度：2026-09-19**  
> 原确认性实验、M1 和 M2 已完成。M3–M6 为后续补充计划。  
> 本文档用于替换此前以 Endpoint–Taylor 性能比较为中心的后续方案。

---

# 零、研究目标与总体定位

## 0.1 核心研究目标

本项目的核心目标是：

> **证明 Endpoint 能够忠实地把原始 NNS 的输入坐标 learning-image 思想推广到视觉 Transformer，能够以统一接口扫描异构内部标量目标，能够对 attention 和 residual 等结构执行可验证的局部 Jacobian 分解，并能够利用所得 Endpoint atlas 观察和分析 ViT 的内部机制。**

该目标包含四个依次递进的层次：

1. **忠实性**  
   Endpoint 是否确实继承了原始 NNS 的输入值初始化和沿当前计算图传播的数学思想？

2. **统一性**  
   Endpoint 是否能够用同一输入坐标表示扫描 token-channel、attention head、Q/K/V 路径、residual branch、模块聚合目标和类别输出？

3. **可审计性**  
   Attention 和 residual 的结构成分是否能够精确重构相应目标的 Endpoint map？实现是否通过独立数值验证？

4. **分析能力**  
   Endpoint atlas 是否能够揭示 ViT 内部输入依赖随层深、token 类型、attention routing/value 和 residual identity/branch 变化的规律？

行为实验用于回答：

> Endpoint 图和 Endpoint 统计量与有限干预行为有什么关系，其解释边界在哪里？

行为预测不是整个方法成立的必要条件，也不再以“Endpoint 必须优于 Taylor”为研究目标。

---

## 0.2 论文不再追求的主张

本文不以证明以下命题为主要目标：

- Endpoint 在所有归因任务中优于现有方法；
- Endpoint 比 regional Taylor 更适合具体 reference 下的有限替换；
- Endpoint 图可以直接解释为因果效应图；
- Endpoint absolute mass 是模型“知识量”；
- 当前注册目标代表全部 ViT 神经元；
- 结构闭合自动意味着有限干预预测能力；
- 高分辨率输入图自动意味着更强的区域选择能力。

基础 Endpoint：

\[
E_s(x)=x\odot\nabla_xs(x)
\]

最可靠的解释是：

> 当前输入点上，注册标量目标 \(s\) 关于 normalized-input 坐标缩放方向的局部一阶依赖。

Attention 和 residual 分解最可靠的解释是：

> 当前计算图上的局部 Jacobian 记账。

它们不是一般的函数值分配，也不是唯一的因果分解。

---

## 0.3 M1 和 M2 的重新定位

M1 和 M2 已经完成，不删除、不弱化，也不追加实验试图使 Endpoint“胜出”。

它们在新论文结构中的定位调整如下。

### M1：Endpoint 成分统计量的行为标定

M1 回答：

> Endpoint mass 与有限成分替换效应之间存在多大关联？该关联相对于简单尺度指标和 reference-matched Taylor 处于什么位置？

M1 不再被表述为“证明 Endpoint 优于其他 predictor”。

### M2：Endpoint 图的干预解释边界

M2 回答：

> Endpoint 的高分辨率输入图能否直接被解释为特定 reference 下的有限干预区域图？

现有结果给出的答案是：

> 一般不能；regional Taylor 与具体 replacement displacement 更匹配。

M2 是 Endpoint 语义边界的实验验证，而不是整篇 Endpoint 方法的胜负判决。

---

# 一、证据链与研究问题

论文的整体证据链重构为五个研究问题。

## RQ1：原始 NNS 忠实性

> 原始 NNS 的 learning-image propagation 是否可以形式化为输入值初始化的局部 Jacobian 传播？Endpoint 是否在原始 NNS 适用网络上与逐层 learning-image propagation 数值等价？

需要同时提供：

- 理论命题；
- 按层证明；
- 原始兼容网络上的直接数值复现；
- ViT 新算子的 Jacobian 单元测试。

---

## RQ2：统一内部目标扫描

> Endpoint 是否能够用同一个数学接口扫描 ViT 中异构的内部标量目标？

主要证据包括：

- 冻结 registry；
- registry-complete execution；
- 目标类型覆盖；
- M3 分层 token-channel 神经元审计；
- 明确的覆盖率和非覆盖范围。

---

## RQ3：结构分解与数值可审计性

> Attention 的 Q/K/V、routing/value 以及 residual 的 identity/branch 分解，是否能够重构注册对象的 Endpoint map？

主要证据包括：

- 正向闭合；
- 负向控制；
- JVP/VJP；
- 中心有限差分；
- forward parity；
- 算子级测试；
- 容差合同。

---

## RQ4：Endpoint 揭示了哪些 ViT 内部机制？

> 利用 Endpoint atlas，能够观察到哪些跨层、跨 token 和跨结构路径的输入依赖规律？

重点分析：

1. token-channel 输入依赖是否随层深由局部趋向分散；
2. CLS 和 patch neuron 的输入依赖是否不同；
3. attention routing/value 的局部输入依赖如何随层和 head 变化；
4. frozen-attention approximation 在哪些 heads 中遗漏较多 routing dependence；
5. residual block 是主要保留还是重写已有输入依赖；
6. identity 与 branch 是否发生显著抵消。

这些问题构成后续实验的主要科学发现来源。

---

## RQ5：Endpoint 的行为相关性和解释边界是什么？

该问题由已完成的 M1 和 M2 主要回答：

- Endpoint mass 与有限成分效应存在广泛但非普遍的正相关；
- 原预注册 all-family omnibus 为 `NOT_SUPPORTED`；
- reference-matched Taylor 通常更适合具体 finite replacement；
- Endpoint 图不应直接等同于 reference-specific intervention map。

RQ5 用于标定 Endpoint 的含义，不再作为论文的中心竞争任务。

---

# 二、实验模块总览

| 模块 | 状态 | 主要目的 | 论文角色 |
|---|---|---|---|
| **M1. Endpoint 成分统计量的行为标定** | 已完成 | 测量 Endpoint mass 与有限成分效应的关系 | 行为相关性 |
| **M2. Endpoint 图的干预解释边界** | 已完成 | 检验 Endpoint 图能否直接用于 reference-specific 区域干预 | 解释边界 |
| **M3. NNS 忠实性桥接与分层神经元审计** | 待执行 | 证明与原始 NNS 的等价关系并扩展 token-channel 覆盖 | 忠实性与统一性 |
| **M4. Attention 与 residual 结构机制分析** | 待执行 | 使用结构化 Endpoint 分析 routing/value 和 carry/branch | 核心机制发现 |
| **M5. 方法语义、reference 属性与最小比较** | 待整理，必要时补算 | 界定 Endpoint 与其他方法回答的问题 | 方法定位 |
| **M6. 机制案例与可视化 atlas** | 待执行 | 展示自动选择的代表性机制案例 | 解释与呈现 |

优先级为：

\[
\boxed{M3 > M4 > M6 > M5}
\]

M1、M2 已完成，不再扩展为大规模方法竞赛。

---

# 三、M1：Endpoint 成分统计量的行为标定

## 3.1 研究问题

M1 回答两个问题。

### M1a：绝对行为关联

> Endpoint mass 是否与同成分有限 activation replacement effect 存在稳定正相关？

### M1b：相对位置

> Endpoint mass 相对于 activation norm、gradient norm、activation × gradient、displacement norm、random 和 activation-space Taylor，提供了怎样的成分排序信息？

M1b 的目的不是要求 Endpoint 全面获胜，而是确定：

- Endpoint 是否只是在重复简单尺度；
- Endpoint 在哪些 family 中提供额外信息；
- reference 信息为 Taylor 带来多大优势；
- Endpoint 的使用边界在哪里。

---

## 3.2 数学定义

对注册成分 \(z_k(x)\)，reference replacement effect 定义为：

\[
e_k^{\mathrm{ref}}
=
F(x)-F\bigl(x;z_k\leftarrow z_k^{\mathrm{ref}}\bigr).
\]

主要评价其绝对大小：

\[
y_k^{\mathrm{ref}}=|e_k^{\mathrm{ref}}|.
\]

Endpoint 成分图为：

\[
E_{k,i}
=
x_i
\left\langle
\nabla_{z_k}F,
\frac{\partial z_k}{\partial x_i}
\right\rangle.
\]

主要 Endpoint statistic 为：

\[
m_k^{\mathrm{Endpoint}}
=
\sum_i|E_{k,i}|.
\]

比较方法包括：

\[
m_k^{\mathrm{act}}=\|z_k\|_2,
\]

\[
m_k^{\mathrm{grad}}=\|\nabla_{z_k}F\|_2,
\]

\[
m_k^{\mathrm{actgrad}}
=
|\langle\nabla_{z_k}F,z_k\rangle|,
\]

\[
m_k^{\mathrm{disp,ref}}
=
\|z_k-z_k^{\mathrm{ref}}\|_2,
\]

\[
m_k^{\mathrm{Taylor,ref}}
=
|\langle\nabla_{z_k}F,z_k-z_k^{\mathrm{ref}}\rangle|.
\]

---

## 3.3 已完成范围

M1 已于 2026-09-18 完成。

结果入口：

- `endpoint_m1_results_20260918.md`

封存根：

- `reports/nns_vit_endpoint_m1_r1_20260918/`

状态：

- 执行与审计：`ENDPOINT_M1_PASS_EXECUTION_AND_AUDIT`
- 科学状态：`SUPPLEMENTARY_COMPONENT_DEPENDENT_UTILITY`

执行范围：

- ViT-B/16；
- DeiT-S；
- 共享 400 张 evaluation 图像；
- blocks 0–11；
- normalized-zero 和 blur-rgb；
- frozen ground-truth logit；
- head、routing、value、Q、K、V、residual identity、residual branch；
- 800 个 image-model 任务；
- 326,400 个 component-reference 条件。

---

## 3.4 M1 主要结果

### 3.4.1 同层 head ranking

| Model | Reference | Endpoint \(\rho\) | Taylor \(\rho\) | Endpoint Capture@1 | Taylor Capture@1 | Endpoint Capture@3 | Taylor Capture@3 |
|---|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | normalized-zero | 0.2815 | 0.6742 | 0.5174 | 0.8038 | 0.6149 | 0.8567 |
| ViT-B/16 | blur-rgb | 0.2727 | 0.6856 | 0.5075 | 0.8203 | 0.6185 | 0.8659 |
| DeiT-S | normalized-zero | 0.1952 | 0.6095 | 0.5609 | 0.8028 | 0.7122 | 0.8894 |
| DeiT-S | blur-rgb | 0.1476 | 0.6238 | 0.5506 | 0.8426 | 0.7105 | 0.8970 |

四个 model/reference 组合中的 Endpoint Spearman 均显著大于 0 和 0.10。

因此可以确认：

> Endpoint mass 对同层 head 的有限替换效应包含稳定但有限的排序信息。

但效应量明显低于 reference-matched Taylor。

---

### 3.4.2 基线比较

Endpoint 相对各 predictor 的 Spearman 优势计数如下：

| Baseline | 有效检验数 | Endpoint 优势获支持 | 未获支持 |
|---|---:|---:|---:|
| activation norm | 180 | 71 | 109 |
| gradient norm | 180 | 72 | 108 |
| activation × gradient | 180 | 25 | 155 |
| displacement norm | 180 | 50 | 130 |
| Taylor | 180 | 12 | 168 |
| random | 180 | 122 | 58 |

另有 156 项小集合 Spearman 因离散度不足被标记为 undefined，不属于采集失败。

结论是：

> Endpoint 相对 random 和部分简单尺度基线具有 component-dependent 的增量信息，但没有整体优于 reference-matched Taylor。

---

### 3.4.3 Residual identity 例外

Residual identity family 中，Endpoint-minus-Taylor Spearman 在四个 model/reference 组合中均为正并通过统一 Holm 校正，但 Capture@1 中位差均为 0。

因此只能写：

> Residual identity 是 Endpoint 在排序相关性上的一个稳定正向例外。

不能写：

> Endpoint 在 residual identity 中普遍提供更强的 Top-1 决策能力。

---

## 3.5 M1 正式结论

原 C4a all-family omnibus 保持：

> `NOT_SUPPORTED`

正式表述为：

> 严格的预注册 all-family omnibus 标准未得到支持，因为 12 个 family 中有 1 个未达到预注册的 0.10 点估计门槛。其余 11 个 family 满足 family-level 条件，说明 Endpoint mass 与有限成分效应之间存在广泛但非普遍的正相关。

M1 在新论文中的作用是：

> 为 Endpoint 的行为相关性提供量化标定，而不是证明其为最佳有限效应 predictor。

---

# 四、M2：Endpoint 图的干预解释边界

## 4.1 研究问题

M2 回答：

> Endpoint 的输入空间图能否被直接解释为特定 reference 下的有限区域干预图？

该问题不同于：

> Endpoint 是否是一个正确的局部 Jacobian 表示？

即使 Endpoint 数学和实现均正确，也不必然适合预测特定有限 replacement。

---

## 4.2 已完成范围

M2 已于 2026-09-19 完成。

结果入口：

- `endpoint_m2_results_20260919.md`

封存根：

- `reports/nns_vit_endpoint_m2_r1_20260919/`

状态：

- 执行与审计：`ENDPOINT_M2_PASS_EXECUTION_AND_AUDIT`
- 科学状态：`SUPPLEMENTARY_LAYER_REFERENCE_OUTCOME_SPECIFIC`

执行范围：

- ViT-B/16 和 DeiT-S；
- 共享 400 张 evaluation 图像；
- blocks 2、5、10；
- 全部 heads；
- normalized-zero 和 blur-rgb；
- patch 预算 1、10、20、49；
- 完整输入区域替换；
- 单 head 隔离替换；
- 2,332,800 条曲线；
- 8,164,800 个有限预算点。

---

## 4.3 M2 主要结果

### 完整输入区域替换

- Taylor 对 Endpoint 显著占优：11/12；
- Endpoint 对 Taylor 显著占优：0/12；
- 未通过校正：1/12。

未通过项为：

- DeiT-S；
- normalized-zero；
- block 5。

该项的差值方向仍偏向 Taylor，但未通过 Holm 校正，不能称为等效。

### 单 head 隔离替换

- Taylor 对 Endpoint 显著占优：12/12；
- Endpoint 对 Taylor 显著占优：0/12。

---

## 4.4 M2 的机制解释

Regional Taylor 使用：

\[
T_k^{\mathrm{ref}}
=
\left\langle
\nabla_{z_k}F,
z_k(x)-z_k^{\mathrm{ref}}
\right\rangle,
\]

它直接使用实际 reference replacement 所定义的 activation displacement。

Endpoint 使用：

\[
E_{k,i}
=
x_i
\left\langle
\nabla_{z_k}F,
\frac{\partial z_k}{\partial x_i}
\right\rangle,
\]

其方向来自 normalized-input 坐标缩放，不等于一般的 reference displacement。

通常：

\[
J_{z_k,x}x
\neq
z_k(x)-z_k(x^{\mathrm{ref}}).
\]

因此，M2 结果支持：

> reference displacement matching 对有限替换预测非常重要。

它不否定 Endpoint 的以下价值：

- 统一内部目标输入表示；
- reference-independent registry scan；
- 局部 Jacobian 精确记账；
- attention/residual 结构闭合；
- 内部目标 atlas；
- ViT 机制分析。

---

## 4.5 M2 正式结论

M2 在论文中应写成：

> Endpoint 的高分辨率输入表示不应被未经验证地解释为特定 reference 下的最优有限干预区域图。在本实验设置中，与实际 replacement displacement 匹配的 regional Taylor 更适合该任务。

M2 不再扩展为更多 reference、更多区域形状或更多预算的胜负搜索。除非新的研究问题与 Endpoint 的核心定位直接相关，否则停止继续扩大 Endpoint–Taylor 排行榜。

---

# 五、M3：NNS 忠实性桥接与分层神经元审计

M3 是后续最高优先级模块。

它同时承担两项任务：

1. 正式证明 Endpoint 忠实继承原始 NNS；
2. 扩大真正 token-channel 神经元的扫描覆盖。

---

# 六、M3-A：原始 NNS 与 Endpoint 的理论及数值等价

## 6.1 理论命题

建议在正文或附录中加入：

> **命题 1：原始 NNS learning image 与 Endpoint 的等价性。**  
> 对由仿射层、卷积、当前状态下可微的逐点激活以及具有唯一 winner 的 MaxPool 组成的网络，若原始 NNS 以输入值进行对角初始化，并以当前前向状态的局部传播系数递归传播 learning image，则任意标量目标 \(s(x)\) 的最终 learning image 等于
> \[
> \mathcal L(s;x)
> =
> \nabla_xs(x)^\top\operatorname{diag}(x),
> \]
> 即向量形式下：
> \[
> E_s(x)=x\odot\nabla_xs(x).
> \]

统一定义：

\[
\mathcal L(u;x)
=
J_{u,x}(x)\operatorname{diag}(x).
\]

若 \(v=f(u_1,\ldots,u_r)\)，则：

\[
\mathcal L(v;x)
=
\sum_{a=1}^{r}
J_{f,u_a}(x)\mathcal L(u_a;x).
\]

由链式法则：

\[
\mathcal L(v;x)
=
J_{v,x}(x)\operatorname{diag}(x).
\]

对标量目标 \(s\) 即得 Endpoint。

---

## 6.2 理论条件

命题必须明确以下条件：

1. 输入坐标是模型实际接收的 normalized input；
2. 偏置不作为图像输入坐标；
3. ReLU 在零点处采用框架定义的次梯度；
4. MaxPool 不存在并列 winner，或使用框架实际 winner；
5. 等价的是局部输入依赖图，不是函数值 completeness；
6. CLS token、position embedding 和模型偏置不作为图像输入坐标归因；
7. 对平滑 ViT 算子使用其当前点真实 Jacobian。

---

## 6.3 原始兼容网络数值复现

实现两套彼此独立的计算：

### 方法 A：原始 NNS 式逐层传播

显式传播 learning image：

- Linear；
- ReLU；
- Conv；
- MaxPool；
- residual add。

### 方法 B：自动微分 Endpoint

\[
E_s(x)=x\odot\nabla_xs(x).
\]

测试网络至少包括：

1. 小型 FCN；
2. 小型 CNN；
3. CNN + MaxPool；
4. 小型 residual MLP；
5. 小型 residual CNN。

每个模型使用：

- 至少 100 个随机输入；
- 多个隐藏层标量目标；
- 多个输出目标；
- float64 主验证；
- float32 框架一致性验证。

报告：

\[
\epsilon_{\mathrm{rel}}
=
\frac{
\|E_{\mathrm{NNS}}-E_{\mathrm{AD}}\|_2
}{
\|E_{\mathrm{AD}}\|_2+\epsilon
},
\]

以及容差合同：

\[
\max_i
\frac{
|E_{\mathrm{NNS},i}-E_{\mathrm{AD},i}|
}{
\mathrm{atol}+\mathrm{rtol}|E_{\mathrm{AD},i}|
}.
\]

该实验是“忠实推广 NNS”最直接的证据，优先级高于新增任何 attribution leaderboard。

---

## 6.4 ViT 算子级 Jacobian 测试

分别为以下算子建立小型 float64 单元测试：

- LayerNorm；
- GELU；
- softmax；
- \(QK^\top/\sqrt{d_h}\)；
- \(AV\)；
- head concatenation；
- output projection；
- residual add；
- MLP；
- final LayerNorm；
- classifier。

每项至少比较：

1. 显式或解析 Jacobian；
2. autograd VJP/JVP；
3. 中心有限差分。

中心有限差分：

\[
D_\epsilon(v)
=
\frac{
f(x+\epsilon v)-f(x-\epsilon v)
}{
2\epsilon
}.
\]

比较：

\[
D_\epsilon(v)\approx J_{f,x}v.
\]

对多个 \(\epsilon\) 报告误差曲线，以避免只在单个步长上验证。

---

# 七、M3-B：冻结的分层 token-channel 神经元审计

## 7.1 目标目录

对两个模型的全部 12 个 blocks，每层选择：

- 16 个 tokens；
- 32 个 channels。

每个模型、每张图像共：

\[
12\times16\times32=6,144
\]

个 token-channel 标量目标。

目标为：

\[
s_{b,t,c}(x)=h_{b,t,c}(x).
\]

Endpoint 为：

\[
E_{b,t,c}(x)
=
x\odot\nabla_xh_{b,t,c}(x).
\]

---

## 7.2 图像规模

默认使用：

> 原 400 张图像中按 deterministic hash 冻结抽取的 100 张图像。

两个模型使用相同图像列表。

如果资源允许扩大到 400 张，必须在第一次生产运行前决定，不能根据前 100 张结果调整样本规模。

---

## 7.3 Token 选择

每层选择：

- 1 个 CLS token；
- 15 个 patch tokens。

15 个 patch tokens 使用固定空间分层：

1. 将 14×14 patch grid 划为 15 个近似等面积 strata；
2. 每个 stratum 选择一个固定 patch；
3. 使用距离 stratum 几何中心最近的 patch；
4. 全部图像使用相同位置；
5. 列表和生成脚本公开。

不得根据 Endpoint 图质量挑选 token。

---

## 7.4 Channel 选择

每层独立选择 32 个 channels：

- 固定公开 seed 或 hash；
- 所有图像使用相同列表；
- 两模型分别生成；
- 不根据 activation、class selectivity、M1/M2 或图像质量挑选。

---

## 7.5 覆盖率

ViT-B/16 的全部 post-block token-channel 数量为：

\[
12\times197\times768
=
1,815,552.
\]

抽样覆盖比例：

\[
\frac{6,144}{1,815,552}
\approx0.338\%.
\]

DeiT-S 的全部 post-block token-channel 数量为：

\[
12\times197\times384
=
907,776.
\]

抽样覆盖比例：

\[
\frac{6,144}{907,776}
\approx0.677\%.
\]

因此只能声称：

> 对冻结的、层级与空间分层的 6,144 个 token-channel 单元进行了神经元级 Endpoint 审计。

不能声称扫描了全部 ViT 神经元。

---

## 7.6 流式保存内容

不保存全部 full-resolution maps。对每个目标流式保存：

- signed sum；
- absolute mass；
- L2 norm；
- positive mass；
- negative mass；
- zero-gradient 状态；
- cancellation ratio；
- normalized spatial entropy；
- 196 维 patch absolute mass；
- 196 维 patch signed sum；
- 数值验证状态。

对少量预冻结目标保存完整图。

若：

\[
\sum_i|E_i|=0,
\]

则以下指标记录为 NA：

- cancellation；
- entropy；
- locality；
- distance。

不得通过加入 \(\epsilon\) 将零图解释为低熵或低抵消图。

---

## 7.7 数值验证子集

每层冻结：

- 2 个 tokens；
- 4 个 channels。

共：

\[
12\times2\times4=96
\]

个目标/模型/图像。

验证：

- Endpoint 与 autograd；
- JVP/VJP；
- 中心有限差分；
- 多个随机方向；
- 多个 \(\epsilon\)；
- float64 小批量复核。

M3 的目标是神经元覆盖和机制分析，不再默认执行额外的大规模 neuron replacement。若未来需要，只能作为独立研究问题另行冻结。

---

# 八、M3-C：Neuron atlas 的层级与空间规律

该部分是 Endpoint 用于分析 ViT 内部机制的第一条主线。

## 8.1 空间分布

对 patch-level absolute mass：

\[
a_j
=
\sum_{i\in P_j}|E_i|,
\qquad
p_j
=
\frac{a_j}{\sum_ja_j}.
\]

Normalized entropy：

\[
H_{\mathrm{norm}}(E)
=
-\frac{
\sum_{j=1}^{196}p_j\log p_j
}{
\log196
}.
\]

其含义为：

- 接近 0：输入依赖集中；
- 接近 1：输入依赖分散。

---

## 8.2 Own-token locality

对 patch token \(t\)：

\[
L_t
=
\sum_{j:
\|\operatorname{coord}(j)-\operatorname{coord}(t)\|_\infty\le1
}
p_j.
\]

同时报告相对于均匀分布基线的差值，以修正边缘 token 邻域较小的问题。

CLS token 没有自身 patch 位置，不计算 own-token locality。

---

## 8.3 输入依赖距离

对 patch token \(t\)：

\[
D_t
=
\sum_j
p_j
\frac{
\|\operatorname{coord}(j)-\operatorname{coord}(t)\|_2
}{
13\sqrt2
}.
\]

距离越大，说明输入依赖质量平均分布得越远。

---

## 8.4 Signed cancellation

\[
C(E)
=
1-
\frac{
|\sum_iE_i|
}{
\sum_i|E_i|
}.
\]

该指标衡量同一 Endpoint map 内正负项抵消，不等同于 residual 路径之间的抵消。

---

## 8.5 主要比较

### 层深趋势

比较：

- early：blocks 0–3；
- middle：blocks 4–7；
- late：blocks 8–11。

主要指标：

- entropy；
- own-token locality；
- distance；
- signed cancellation。

只有当以下方向共同出现时，才支持“由局部趋向全局”：

- entropy 上升；
- own-token locality 下降；
- distance 上升。

仅凭 entropy 上升不能单独确认全局化。

### Token 类型

比较：

- CLS neuron；
- patch neuron。

### 空间位置

比较：

- 中心 patch；
- 边缘 patch；
- 全部 patch。

### 模型

分别报告：

- ViT-B/16；
- DeiT-S。

不同模型的 channel ID 不视为对应神经元。

---

## 8.6 神经元图相似性

分别使用：

1. patch absolute distribution cosine；
2. patch signed vector cosine。

比较：

- same token / different channel；
- different token；
- early / middle / late。

每层每组冻结抽取 256 对 unit pairs。

聚类和二维嵌入仅作为描述性可视化，不作为主要统计证据。

---

## 8.7 允许支持的机制结论

如果结果支持，可以写：

> 在抽样 token-channel 单元中，输入依赖随层深变得更分散，并且 patch neuron 对自身邻域的局部依赖减弱。

必须包含限定：

- 抽样单元；
- 所测试模型；
- 所用 Endpoint 定义；
- 局部一阶输入依赖。

不能写成：

> ViT 神经元普遍从局部特征学习到全局语义。

---

# 九、M4：Attention 与 residual 结构机制分析

M4 是后续第二优先级模块，也是论文最可能形成新机制发现的部分。

---

# 十、M4-A：Attention routing/value 与 frozen-attention 诊断

## 10.1 完整 differential

单 head attention：

\[
O=AV,
\qquad
A=\operatorname{softmax}(QK^\top/\sqrt{d_h}).
\]

局部微分：

\[
\mathrm dO
=
(\mathrm dA)V
+
A(\mathrm dV).
\]

进一步分解为：

\[
\mathrm dO
=
\underbrace{
\frac{\partial A}{\partial Q}\mathrm dQ\,V
}_{Q}
+
\underbrace{
\frac{\partial A}{\partial K}\mathrm dK\,V
}_{K}
+
\underbrace{
A\,\mathrm dV
}_{V}.
\]

因此：

\[
E^{\mathrm{attn}}
=
E^Q+E^K+E^V
=
E^{\mathrm{routing}}+E^{\mathrm{value}},
\]

其中：

\[
E^{\mathrm{routing}}=E^Q+E^K.
\]

---

## 10.2 Frozen-attention approximation

Frozen-attention approximation 为：

\[
\mathrm dO\approx A\,\mathrm dV.
\]

在 Endpoint 记账中：

\[
E_h^{\mathrm{FA}}=E_h^{\mathrm{value}},
\]

\[
E_h^{\mathrm{full}}-E_h^{\mathrm{FA}}
=
E_h^{\mathrm{routing}}.
\]

该代数等式用于实现验证，不作为经验发现。

经验问题是：

> routing 项在不同层和不同 head 中实际有多大？

---

## 10.3 执行范围

覆盖：

- 两模型；
- 共享 400 图；
- 12 层；
- 全部 heads；
- frozen ground-truth logit。

已有 E3/M1 记录可以复用；缺失逐 head 数据时只补算必要内容。

---

## 10.4 主要指标

Routing share：

\[
R_h^{\mathrm{routing}}
=
\frac{
\|E_h^{\mathrm{routing}}\|_1
}{
\|E_h^{\mathrm{routing}}\|_1+
\|E_h^{\mathrm{value}}\|_1
}.
\]

Frozen-attention Endpoint error：

\[
\eta_h^E
=
\frac{
\|E_h^{\mathrm{full}}-E_h^{\mathrm{FA}}\|_2
}{
\|E_h^{\mathrm{full}}\|_2
}.
\]

若分母为 0，记录 NA 并单独报告 absolute error。

同时计算未乘输入 \(x\) 的 contracted-gradient error，防止小输入坐标掩盖导数差异。

---

## 10.5 独立验证

使用：

- native autodiff；
- 显式 Q/K/V 分解；
- attention detach；
- forward parity；
- Endpoint closure；
- 随机方向 JVP。

JVP 抽样误差：

\[
\eta_h^{\mathrm{JVP}}
=
\sqrt{
\frac{
\sum_v
\|
(J_{O_h,x}-J_{O_h,x}^{\mathrm{FA}})v
\|_2^2
}{
\sum_v
\|
J_{O_h,x}v
\|_2^2
}
}.
\]

每图固定 8 个单位范数 Rademacher 方向。

---

## 10.6 Attention saturation

定义平均 query-row normalized entropy 的补数：

\[
s_h
=
1-
\frac1{197}
\sum_q
\frac{
-\sum_jA_{h,qj}\log A_{h,qj}
}{
\log197
}.
\]

同时报告 CLS query saturation。

在每图每层 heads 内计算：

- \(s_h\) 与 routing share 的相关；
- \(s_h\) 与 frozen-attention error 的相关。

先做层内比较，再跨图像汇总，避免把共同层深趋势误认为 head 内关系。

不预设相关方向。Softmax 更尖锐并不必然意味着 routing error 更大。

---

## 10.7 允许支持的结论

可以写：

> 某些层或 heads 的局部输入依赖中，routing 项占比较高，因此只保留 value path 的 frozen-attention approximation 会遗漏较多局部输入依赖。

不能写：

> routing share 较高的 head 在因果上更重要。

除非有独立干预证据支持。

---

# 十一、M4-B：Residual identity/branch 记账

## 11.1 数学分解

对：

\[
y=u+f(u),
\]

有：

\[
J_{y,u}=I+J_{f,u}.
\]

固定 residual 输出的 clean cotangent：

\[
g_y=\nabla_yF.
\]

Identity Endpoint：

\[
E^{\mathrm{identity}}
=
x\odot
J_{u,x}^{\mathsf T}g_y.
\]

Branch Endpoint：

\[
E^{\mathrm{branch}}
=
x\odot
J_{u,x}^{\mathsf T}
J_{f,u}^{\mathsf T}g_y.
\]

闭合：

\[
E^{\mathrm{output}}
=
E^{\mathrm{identity}}
+
E^{\mathrm{branch}}.
\]

Identity 必须使用 \(g_y\)，不能错误使用已经包含 branch 作用的 \(\nabla_uF\)。

---

## 11.2 分析对象

分别分析：

1. attention residual：
   - identity；
   - attention branch。

2. MLP residual：
   - carry；
   - MLP branch。

覆盖：

- 两模型；
- 400 图；
- 全部 12 层。

---

## 11.3 主要指标

Identity share：

\[
R^{\mathrm{identity}}
=
\frac{
\|E^{\mathrm{identity}}\|_1
}{
\|E^{\mathrm{identity}}\|_1+
\|E^{\mathrm{branch}}\|_1
}.
\]

Branch share：

\[
R^{\mathrm{branch}}
=
1-R^{\mathrm{identity}}.
\]

跨路径抵消：

\[
C^{\mathrm{path}}
=
1-
\frac{
\|E^{\mathrm{identity}}+E^{\mathrm{branch}}\|_1
}{
\|E^{\mathrm{identity}}\|_1+
\|E^{\mathrm{branch}}\|_1
}.
\]

方向关系：

\[
\cos
\left(
E^{\mathrm{identity}},
E^{\mathrm{branch}}
\right).
\]

空间改变可使用 output 与 identity 的 patch absolute distribution distance。

---

## 11.4 主要问题

M4-B 检验：

1. 深层 Endpoint 是否更多经 identity carry 传播；
2. attention 和 MLP residual 是否具有不同的 identity/branch 比例；
3. branch mass 较大时是否被 identity 抵消；
4. 哪些层主要保留已有空间依赖；
5. 哪些层显著改变 Endpoint 的空间分布；
6. ViT-B/16 与 DeiT-S 是否表现出不同模式。

---

## 11.5 负向控制

故意省略非零 identity 项，确认闭合明显失败。

负向控制用于证明：

> residual identity 不是可忽略的实现细节。

不能把省略 identity 的错误分解作为替代模型的性能实验。

---

# 十二、M4-C：Signed、absolute 与 cancellation

比较：

\[
m_k^{\mathrm{signed}}
=
\left|
\sum_iE_{k,i}
\right|,
\]

\[
m_k^{\mathrm{absolute}}
=
\sum_i|E_{k,i}|.
\]

单图内部 signed cancellation：

\[
C_k
=
1-
\frac{
|\sum_iE_{k,i}|
}{
\sum_i|E_{k,i}|
}.
\]

必须区分：

- **map cancellation**：同一图内部正负项抵消；
- **path cancellation**：identity 与 branch 或 routing 与 value 在相同输入坐标上的抵消。

分析：

- routing 与 value 的 cancellation 差异；
- identity 与 branch 的 cancellation；
- cancellation 随层深的变化；
- absolute mass 较大但 signed aggregate 较小的情况。

行为相关性可以复用 M1，不再建立新一轮大规模 predictor 竞赛。

---

# 十三、M5：方法语义、reference 属性与最小比较

M5 不再执行此前规划的六类 reference、大规模迁移矩阵和扩展空间排行榜，除非 M3/M4 完成后出现一个明确、不可由现有数据回答的新研究问题。

---

## 13.1 M5 的主要任务

M5 主要完成三件事：

1. 明确 Endpoint 与其他方法回答的问题不同；
2. 使用已有 normalized-zero 和 blur 结果说明 reference-specific 行为；
3. 记录成本和信息条件，但不以此预设 Endpoint 的优势。

---

## 13.2 Reference-free 的准确表述

Endpoint 在给定：

- 输入坐标；
- 模型；
- 目标；
- 预处理；

后唯一确定，不需要额外干预 reference。

但 Endpoint 不是绝对“无基线”的。若：

\[
x=\frac{r-\mu}{\sigma},
\]

则：

\[
x_i\frac{\partial F}{\partial x_i}
=
(r_i-\mu_i)\frac{\partial F}{\partial r_i}.
\]

因此 Endpoint 的坐标原点对应 \(r=\mu\)。

推荐表述：

> Endpoint is independent of an externally chosen intervention reference, while remaining tied to the model’s input coordinate system and its normalization origin.

不要写：

> Endpoint 完全无 baseline 或对预处理不变。

---

## 13.3 已有两种 reference 的最小分析

使用 M1 和 M2 已有的：

- normalized-zero；
- blur-rgb。

报告：

1. Endpoint 分数在两个 reference 下不变，这是定义属性；
2. Taylor 分数随 reference 改变；
3. 两种 reference 下实际有限效应排名是否改变；
4. Endpoint 在两种 reference 下的平均行为关联；
5. Taylor 的 reference-specific 优势；
6. 计算和缓存成本。

该分析用于说明信息条件，不用于证明 Endpoint 优越。

---

## 13.4 方法适用性矩阵

| 方法 | 内部标量目标 | 输入空间图 | 需要干预 reference | 局部 Jacobian 结构闭合 | 具体 finite replacement 预测 |
|---|---:|---:|---:|---:|---:|
| Endpoint | 是 | 是 | 否 | 是 | 有限、需实证 |
| Activation-space Taylor | 是 | 可构造区域分数 | 是 | 非主要目标 | 强匹配 |
| Integrated Gradients | 是 | 是 | 是 | 可扩展 | baseline-difference |
| Attention rollout | 通常否 | 是 | 否 | 否 | 非主要目标 |
| Grad-CAM | 部分 | 是 | 否 | 否 | 非主要目标 |
| Chefer/AttnLRP | 部分 | 是 | 通常否 | 依实现 | 非统一 |
| Attention weights | attention 对象 | patch/token | 否 | 否 | 不能单独视为效应 |

---

## 13.5 输出级方法比较的边界

Endpoint logit map：

\[
x\odot\nabla_xF
\]

就是 normalized-input 坐标中的 Input × Gradient。

因此不能把 Endpoint logit 和 Input × Gradient 当作两种独立算法比较。

可以讨论：

\[
x\odot\nabla_xF,
\]

\[
r\odot\nabla_rF,
\]

\[
(r-\mu)\odot\nabla_rF
\]

的坐标原点差异。

Attention rollout、Grad-CAM、Chefer 和 AttnLRP 主要用于功能与语义比较。除非论文声称 Endpoint 在输出定位任务中更优，否则不需要建立大规模统一排行榜。

---

## 13.6 Path IG

Path IG 若未通过预注册校准门，则：

- 不进入确认性跨模型比较；
- 报告 completeness residual；
- 报告积分点数量曲线；
- 报告数值失败原因；
- 放入附录。

不再投入大量资源调整积分方法，除非 IG 是审稿所必需的基线。

---

# 十四、M6：机制案例与可视化 atlas

M6 的目标是将 M3 和 M4 的统计规律可视化，而不是围绕 Endpoint–Taylor 胜负挑选案例。

---

## 14.1 主要案例类别

优先展示：

1. early patch neuron 的局部 Endpoint；
2. late patch neuron 的分散 Endpoint；
3. CLS neuron 与 patch neuron 对比；
4. routing-dominant attention head；
5. value-dominant attention head；
6. frozen-attention error 较高的 head；
7. identity-dominant residual layer；
8. branch-dominant residual layer；
9. identity/branch 强抵消案例；
10. signed cancellation 较高的 neuron；
11. ViT-B/16 与 DeiT-S 的代表性层级差异；
12. M2 中 Endpoint 与 Taylor 语义不同的一个边界案例。

M2 案例只占可视化的一部分，不再主导整套图。

---

## 14.2 自动选择规则

案例必须按照预先冻结的指标选择。

例如：

- routing-dominant：跨图 routing share 位于最高 5%；
- value-dominant：routing share 位于最低 5%；
- high frozen-attention error：\(\eta_h^E\) 位于最高 5%；
- identity-dominant：identity share 位于最高 5%；
- strong path cancellation：\(C^{\mathrm{path}}\) 位于最高 5%；
- local neuron：own-token locality 位于最高 5%；
- dispersed neuron：normalized entropy 位于最高 5%。

在候选集合中使用：

- 最小 image ID；
- 最小 component ID；
- 或固定 hash。

不得根据图是否“好看”再次人工筛选。

---

## 14.3 可视化内容

### Neuron atlas 案例

- 原图；
- signed Endpoint；
- absolute Endpoint；
- patch mass；
- entropy；
- locality；
- distance；
- cancellation；
- block/token/channel。

### Attention 案例

- full Endpoint；
- routing Endpoint；
- value Endpoint；
- Q/K/V；
- attention map；
- routing share；
- frozen-attention error；
- saturation。

### Residual 案例

- output Endpoint；
- identity Endpoint；
- branch Endpoint；
- identity/branch share；
- path cancellation；
- output–identity spatial change。

所有图注必须使用“local input dependence”或“Endpoint dependence”，避免直接称为“神经元学到的概念”或“因果证据”。

---

# 十五、统计分析原则

## 15.1 统计单位

图像是主要统计单位。

不得将同一图像中的：

- heads；
- tokens；
- channels；
- layers；
- references；
- budgets

直接视为独立样本。

先按照各模块规定在图像内聚合，再进行跨图像推断。

---

## 15.2 报告内容

主要报告：

- 中位数；
- image-level 10,000 次 bootstrap 95% CI；
- 图像级配对差值；
- 双侧符号检验；
- Holm 校正；
- 有效图像数；
- NA 和 zero-gradient 数量。

机制分析应优先报告：

- 效应大小；
- 分布；
- 置信区间；
- 跨层趋势。

不能只报告 \(p\) 值。

---

## 15.3 Holm families

建议将主要检验划分为独立 family。

### M3 atlas family

每个模型分别检验：

- late−early entropy；
- late−early locality；
- late−early distance；
- late−early cancellation；
- CLS−patch entropy。

### M4 attention family

每个模型分别检验：

- late−early routing share；
- late−early frozen-attention error；
- saturation–routing relation；
- saturation–error relation。

### M4 residual family

每个模型和 residual 类型分别检验：

- late−early identity share；
- late−early path cancellation。

未定义项不得从校正 family 中事后删除。按照冻结规则，可使用 \(p=1\) 占位并保留状态。

---

## 15.4 方向性措辞

如果未通过显著性检验：

- 写“未获支持”；
- 不写“没有差异”；
- 不写“等效”。

若需要声称等效，必须预先定义等效界值并执行等效性检验。

---

# 十六、注册覆盖率与术语

## 16.1 原始 registry

原 registry 包含：

- token-channel；
- patch sum/energy；
- attention-head coordinates；
- module patch aggregates；
- patch embedding aggregates；
- final CLS coordinate；
- ground-truth logit；
- clean-top1 logit。

论文必须区分：

- registry entries；
- unique scalar functions；
- unique Endpoint maps。

当预测正确时，ground-truth logit 和 clean-top1 logit 可能是同一标量函数。

---

## 16.2 “Complete scan”的定义

只允许写：

> registry-complete scan

其含义是：

> 所有冻结 registry entries 均成功执行，没有目录项缺失。

不能写：

> complete ViT neuron scan

因为当前实验不是对全部 ViT 神经元的穷举扫描。

---

## 16.3 “Neuron”的使用

仅以下对象可以较自然地称为单元或 neuron：

- token-channel scalar；
- final CLS coordinate；
- 明确定义的 attention head output coordinate。

以下对象应称为聚合目标：

- patch sum；
- energy；
- module aggregate；
- logit；
- 多坐标组合。

论文主称呼建议使用：

> registered scalar endpoint

而不是统一称为 neuron。

---

# 十七、论文实验章节重构

## 5.1 From original NNS to ViT Endpoint

内容包括：

- 原始 NNS propagation；
- Jacobian 形式化；
- 等价性命题；
- ReLU/MaxPool 条件；
- normalized-input 坐标；
- bias、CLS 和 position embedding 的非归因范围。

---

## 5.2 Unified registry and scan protocol

内容包括：

- 异构内部目标；
- registry 计数；
- unique target 说明；
- registry-complete 定义；
- 覆盖率；
- confirmatory 与 supplementary 边界。

---

## 5.3 Numerical and structural correctness

内容包括：

- 原始 NNS 数值等价；
- ViT 算子单元测试；
- attention Q/K/V 闭合；
- routing/value 闭合；
- residual identity/branch 闭合；
- JVP/VJP；
- 有限差分；
- 负向控制。

---

## 5.4 Stratified token-channel Endpoint atlas

内容包括：

- M3 的 6,144 个目标；
- token/channel 抽样；
- 成功率；
- entropy；
- locality；
- distance；
- cancellation；
- CLS/patch；
- early/middle/late；
- 模型差异；
- neuron map similarity。

---

## 5.5 Attention mechanism through Endpoint decomposition

内容包括：

- Q/K/V；
- routing/value；
- routing share；
- frozen-attention omission；
- JVP error；
- saturation relation；
- 代表性 heads。

---

## 5.6 Residual preservation and rewriting of input dependence

内容包括：

- attention residual；
- MLP residual；
- identity/branch；
- mass share；
- path cancellation；
- spatial distribution change；
- early/middle/late。

---

## 5.7 Behavioral calibration and interpretation boundary

内容包括：

- M1；
- 11/12 family；
- omnibus `NOT_SUPPORTED`；
- 强基线；
- Taylor 优势；
- residual identity 例外；
- M2；
- 11/12 和 12/12；
- Endpoint 图不等于 reference-specific intervention map。

M1/M2 放在结构机制结果之后，而不是作为整篇论文的主要胜负实验。

---

## 5.8 Relation to existing explanation methods

内容包括：

- 方法适用性矩阵；
- Endpoint 与 Input × Gradient 的数学身份；
- Taylor 的 reference matching；
- IG；
- attention rollout；
- Grad-CAM；
- Chefer/AttnLRP；
- 方法回答问题的差异。

---

## 5.9 Limitations

明确包括：

- 基础公式是 Gradient × Input；
- 只归因图像输入坐标；
- 不包含 bias、CLS、position embedding 的独立贡献；
- 不具备一般函数值 completeness；
- 当前 neuron atlas 是分层抽样；
- 结构分解是局部 differential decomposition；
- finite intervention 结果依赖 reference；
- 只测试两个 ImageNet ViT；
- 机制规律不能直接推广到所有 Transformer。

---

# 十八、论文贡献重写

建议将贡献写为以下四点。

## 贡献一：原始 NNS 的 Jacobian 形式化与 ViT 推广

> 我们将原始 NNS 的输入值初始化和当前状态传播形式化为局部 Jacobian learning-image propagation，并证明其对标量目标等于 normalized-input 坐标中的 Gradient × Input。该形式进一步推广到 LayerNorm、GELU、softmax attention 和 residual connection。

---

## 贡献二：统一的异构内部目标扫描

> 我们构建统一的 ViT registered scalar endpoint 接口，覆盖 token-channel、attention head、Q/K/V、routing/value、residual branch、模块聚合目标和类别输出，并通过分层 token-channel 审计扩大与原始 NNS 神经元扫描范式的联系。

---

## 贡献三：可验证的结构化 Jacobian 记账

> 我们提出 attention 和 residual 的 differential cuts，并通过正向闭合、负向控制、JVP/VJP、有限差分和原始 NNS 数值等价实验验证实现正确性。

---

## 贡献四：ViT 机制观察与行为边界

> 我们利用 Endpoint atlas 分析 token-channel 输入依赖的层级和空间规律、attention routing/value 的分布以及 residual identity/branch 的保留和重写作用。有限替换实验进一步表明，Endpoint statistics 包含广泛但非普遍的行为信息，同时不能被直接等同为 reference-specific finite-intervention estimator。

---

# 十九、摘要建议稿

> 本文将神经网络扫描器（NNS）的输入坐标 learning-image 思想推广到视觉 Transformer。我们首先将原始 NNS 形式化为输入值初始化的局部 Jacobian 传播；对任意注册标量目标，所得 Endpoint 等于 normalized-input 坐标中的 Gradient × Input。本文的主要方法贡献不在该基础归因公式本身，而在于异构 ViT 内部目标的统一注册、attention 的 Q/K/V 与 routing/value 分解、residual identity/branch 记账，以及可独立审计的数值验证协议。  
>   
> 在 ViT-B/16 和 DeiT-S 上，我们执行 registry-complete 内部目标扫描，并通过结构闭合、负向控制、JVP/VJP、中心有限差分和原始 NNS 等价复现验证局部 Jacobian 实现。进一步的分层 token-channel Endpoint atlas 用于分析输入依赖随层深、token 类型和空间位置的变化；attention 和 residual 分解用于研究 routing/value 以及 carry/branch 对输入依赖的相对作用。  
>   
> 行为标定实验显示，Endpoint mass 与有限成分替换效应之间存在广泛但非普遍的正相关；严格的预注册 all-family omnibus 标准未得到支持。在 reference-based 区域干预中，与实际 replacement displacement 匹配的 regional Taylor 表现更强，说明 Endpoint 的高分辨率输入表示不能直接解释为特定 reference 下的最优干预图。  
>   
> 这些结果将 NNS-ViT 定位为一种面向异构 ViT 内部标量目标的统一、输入坐标化且可结构审计的局部表示框架，并明确区分局部输入依赖、有限成分效应和空间干预选择三种不同能力。

其中关于 atlas 和机制发现的句子应在 M3/M4 完成后按实际结果替换，不能提前写入方向性结论。

---

# 二十、推荐标题

首选：

> **NNS-ViT Endpoint Scanner：视觉 Transformer 内部目标的统一输入坐标表示、结构分解与机制分析**

备选：

> **从 NNS 到视觉 Transformer：内部标量目标的 Endpoint 图谱与可审计 Jacobian 分解**

英文可考虑：

> **NNS-ViT: Auditable Endpoint Maps and Structured Jacobian Decomposition for Internal Targets in Vision Transformers**

---

# 二十一、后续执行顺序

| 顺序 | 工作 | 目的 | 必要交付 |
|---|---|---|---|
| 1 | M3-A 原始 NNS 等价证明和数值复现 | 建立“忠实推广”的直接证据 | 命题、证明、FCN/CNN/ResNet 数值表 |
| 2 | ViT 算子级 Jacobian 单元测试 | 提高独立数值可信度 | LayerNorm/GELU/softmax/attention 测试 |
| 3 | M3-B 生成 6,144-unit neuron atlas | 扩大真实神经元覆盖 | 冻结 token/channel 列表、逐目标统计 |
| 4 | M3-C atlas 分布分析 | 形成第一组 ViT 机制结果 | 深度、token、位置、模型比较 |
| 5 | M4-A routing/value 与 frozen-attention | 形成 attention 机制结果 | 层/head 表、误差和 saturation 图 |
| 6 | M4-B residual identity/branch | 形成 residual 机制结果 | share、cancellation、空间改变 |
| 7 | M6 自动选择机制案例 | 支持机制解释 | atlas、attention、residual 主图 |
| 8 | M5 方法定位与最小 reference 分析 | 完成讨论和相关工作 | 适用性矩阵、已有 reference 复用分析 |
| 9 | 重构论文 | 以忠实性—统一性—可审计性—机制发现为主线 | 正文、附录、审计索引 |

---

# 二十二、停止规则与资源控制

以下实验不再作为当前论文必做项：

- 六类 reference 的大规模扩展；
- 更多 Endpoint–Taylor 区域胜负实验；
- 新区域形状搜索；
- 新 blur 强度搜索；
- 追加实验直到 Endpoint 在某条件下胜出；
- 全量百万级 ViT 神经元 atlas；
- 大规模输出定位 benchmark；
- 为改善结果而事后选择 token、channel、head 或图像。

只有在满足以下条件时才新增实验：

1. 直接回答 RQ1–RQ4；
2. 现有数据不能回答；
3. 运行前冻结设置；
4. 不以改变 M1/M2 结论为目的；
5. 具有明确的停止条件。

如果资源有限，最低完成集为：

\[
\boxed{
\text{M3-A 等价桥接}
+
\text{M3 neuron atlas}
+
\text{M4 attention/residual}
+
\text{M6 核心图}
}
\]

---

# 二十三、最终论文结论框架

论文最终结论应按照以下顺序组织。

## 1. 忠实推广

> Endpoint 在原始 NNS 兼容网络上与逐层 learning-image propagation 数值等价，并将相同的局部 Jacobian 思想推广到 ViT 的平滑和输入依赖算子。

## 2. 统一扫描

> Endpoint 为多类 ViT registered scalar targets 提供统一输入坐标表示；当前实验是 registry-complete 和分层 neuron audit，而不是全神经元穷举。

## 3. 可审计结构分解

> Attention 的 Q/K/V、routing/value 和 residual identity/branch 能够在当前计算图的局部 Jacobian 层面闭合，并通过独立数值检查。

## 4. ViT 机制观察

> Endpoint atlas 用于描述 token-channel 输入依赖的层级与空间分布，并量化 attention routing/value 和 residual carry/branch 的相对作用。具体方向必须依据 M3/M4 的实际结果填写。

## 5. 行为相关性与边界

> Endpoint mass 与有限成分效应存在广泛但非普遍的关联；原 all-family omnibus 保持 `NOT_SUPPORTED`。对于特定 reference 下的有限区域干预，regional Taylor 更匹配实际位移，Endpoint 不应被直接解释为最优 intervention map。

---

# 二十四、最终定位

本文最终定位为：

> **NNS-ViT 是对原始 NNS 输入坐标 learning-image 思想的局部 Jacobian 推广。它面向视觉 Transformer 的异构内部标量目标，提供统一 Endpoint 表示、可闭合的 attention/residual 结构记账以及用于分析内部输入依赖规律的 atlas。有限干预实验用于标定该表示的行为相关性和解释边界，而不是将 Endpoint 包装为普遍优于现有方法的干预预测器。**

最简洁的中心论点是：

> **Endpoint is an auditable internal-target representation, not a universal intervention estimator.**

对应中文：

> **Endpoint 的主要价值是统一、可审计地表示 ViT 内部目标的局部输入依赖，而不是成为所有有限干预任务中的通用最优预测器。**