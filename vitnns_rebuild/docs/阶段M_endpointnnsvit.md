# NNS-ViT 补充实验完整方案

> **当前进度（2026-09-19）：M1、M2 已完成；M3–M6 为后续计划。**

[M2 结果](endpoint_m2_results_20260919.md)显示，在 patch 预算下，Taylor 对完整输入区域替换通过 11/12 项方向性优势检验，对单 head 隔离效应通过 12/12 项；Endpoint 的对应优势均为 0/12。完整输入效应的例外为 DeiT-S、normalized-zero、block 5，未通过校正检验不表示两者等效。

后续补充实验围绕 Endpoint 的设计目标展开：reference 未指定时的统一复用与成分筛选、attention/residual 的结构诊断、神经元输入依赖的分布。性能、成本和机制优势均由实验检验。本次更新的是方案，以下新增实验尚未执行。

五类实验直接归入现有模块：reference 稳定性与复用见 §7.5，reference-free 成分筛选见 §7.6，frozen-attention 诊断见 §7.1–7.2，residual 记账见 §7.3–7.4，neuron-atlas 分布见 §4.8。执行顺序与记录方式见第十四节。

**M3 采用冻结的分层 token-channel 抽样：**


- 比仅扫描 CLS channel 的方案 A 更能覆盖空间 patch token；
- 比 MLP 专项方案 C 更贴近当前 Endpoint Scanner 的主要对象；
- 目标数量和计算规模相对可控；
- 可以明确表述为“分层神经元审计”，增强与原始 NNS 的联系；
- 能够与现有 RQ3、RQ4 和注册覆盖率分析自然结合。

但必须明确：这些实验是在原冻结协议之后增加的，应统一标记为：

> **补充性冻结实验（supplementary frozen study）**

或：

> **探索性扩展（exploratory extension）**

这些新增实验不能被事后并入原确认性分析，也不能改变原预注册 omnibus 的 `NOT_SUPPORTED` 结论。

---

# 一、补充实验的整体结构

建议将新增实验组织为六个模块：

| 模块 | 主要目的 | 优先级 |
|---|---|---:|
| **M1. RQ3 强基线与分层分析** | 判断 Endpoint 是否具有额外的成分排序价值 | 必做 |
| **M2. RQ4 跨层、全 head 扩展** | 检验空间负结果是否跨层成立 | 必做 |
| **M3. 分层 token-channel 神经元审计与分布分析** | 说明覆盖范围，研究输入依赖随层深和位置的变化 | 必做 |
| **M4. Transformer 解释方法比较** | 界定不同方法回答的问题 | 推荐 |
| **M5. Reference 复用、成分筛选与结构诊断** | 评价 reference-free 使用场景及 routing/residual 机制 | 本轮新增重点 |
| **M6. 失败案例与机制可视化** | 提升机制解释和论文可读性 | 推荐 |

除以上六个实验模块外，还必须同步完成两项论文层面的修改：

1. 重写 RQ3 的统计结论；
2. 增加注册目标覆盖率表。

---

# 二、M1：RQ3 强基线与分层分析

## 2.1 研究目的

建议将原 RQ3 拆分成两个更清楚的问题。

### RQ3a：绝对关联

> Endpoint mass 是否与同成分有限替换效应之间存在稳定的正向排序关联？

这是现有实验已经主要回答的问题。

### RQ3b：增量价值

> Endpoint mass 是否比简单尺度指标和 reference-matched 一阶近似更适合预测有限替换效应？

这是当前论文缺少的关键比较。

仅仅证明 Endpoint 与有限替换效应存在正相关还不够。还需要判断这种关联是否优于：

- 激活大小；
- 梯度大小；
- reference displacement；
- activation-space Taylor；
- 随机排序。

---

## 2.2 统一成分与有限效应定义

对注册成分 \(z_k(x)\)，定义 reference replacement 后的有限效应：

\[
e_k^{\mathrm{ref}}
=
F(x)
-
F\bigl(x;z_k\leftarrow z_k^{\mathrm{ref}}\bigr).
\]

如果主要研究的是有限效应大小，则定义：

\[
y_k^{\mathrm{ref}}
=
\left|e_k^{\mathrm{ref}}\right|.
\]

如果还需要研究效应方向，可以在补充材料中单独报告 signed effect，但不要将 signed effect 和 absolute effect 混合到同一个主要统计量中。

Endpoint 成分图定义为：

\[
E_{k,i}
=
x_i
\left\langle
\nabla_{z_k}F,
\frac{\partial z_k}{\partial x_i}
\right\rangle.
\]

主要 Endpoint predictor 沿用绝对质量：

\[
m_k^{\mathrm{Endpoint}}
=
\sum_i |E_{k,i}|.
\]

为了减少输入维度或区域数量造成的尺度影响，可以同时报告平均绝对质量：

\[
\bar m_k^{\mathrm{Endpoint}}
=
\frac{1}{D_x}
\sum_i |E_{k,i}|,
\]

其中 \(D_x\) 为输入坐标数量。

---

## 2.3 必须加入的 predictor baselines

所有 predictor 必须针对同一个成分集合、同一目标函数和同一 reference 进行比较。

### 1. Activation norm

\[
m_k^{\mathrm{act}}
=
\|z_k(x)\|_2.
\]

该基线回答：

> 激活更大的成分是否自然具有更大的有限替换效应？

---

### 2. Gradient norm

\[
m_k^{\mathrm{grad}}
=
\|\nabla_{z_k}F(x)\|_2.
\]

该基线回答：

> 仅使用局部敏感度，是否已经足以排序成分？

---

### 3. Activation × Gradient

\[
m_k^{\mathrm{actgrad}}
=
\left|
\left\langle
\nabla_{z_k}F(x),
z_k(x)
\right\rangle
\right|.
\]

这是成分空间中最接近 Gradient × Activation 的基线。

---

### 4. Reference displacement norm

\[
m_k^{\mathrm{disp,ref}}
=
\|z_k(x)-z_k^{\mathrm{ref}}\|_2.
\]

该基线回答：

> 有限效应是否主要由 replacement 位移大小决定？

---

### 5. Attribution patching / activation-space Taylor

\[
m_k^{\mathrm{Taylor,ref}}
=
\left|
\left\langle
\nabla_{z_k}F(x),
z_k(x)-z_k^{\mathrm{ref}}
\right\rangle
\right|.
\]

这是最重要的强基线，因为它与实际 finite replacement 的位移方向直接匹配。

如果它明显优于 Endpoint，论文应如实总结：

> Endpoint 提供 reference-free 的成分级排序信号，但 reference-matched activation-space Taylor 更适合预测具体 reference 下的有限替换效应。

---

### 6. Random baseline

对每张图、每个 component family 生成固定种子的随机排序。

建议：

- 每个 comparison set 至少执行 1,000 次 permutation；
- 报告随机 Spearman 分布；
- 报告 top-\(K\) 指标相对随机排序的提升；
- 固定随机种子并写入配置文件。

---

## 2.4 评价指标

不建议继续只使用 Spearman。需要同时加入更贴近实际筛选任务的决策型指标。

### 2.4.1 主排序指标：图像内 Spearman

对每张图、每个 family 和每种 predictor，计算：

\[
\rho_{i,f,m}
=
\operatorname{Spearman}
\left(
m_{i,f,m,k},
|e_{i,f,k}|
\right),
\]

其中：

- \(i\) 表示图像；
- \(f\) 表示 component family；
- \(m\) 表示 predictor；
- \(k\) 表示该 family 中的成分。

图像应作为最终统计单位。

---

### 2.4.2 Top-\(K\) effect capture

定义：

\[
\mathrm{Capture@K}
=
\frac{
\sum_{k\in\operatorname{TopK}(m)}
|e_k|
}{
\sum_{k\in\operatorname{TopK}(|e|)}
|e_k|
}.
\]

该指标回答：

> 根据 predictor 选出的前 \(K\) 个成分，能够捕获理论最优前 \(K\) 个成分的多少真实有限效应？

---

### 2.4.3 Top-1 regret

定义：

\[
\mathrm{Regret@1}
=
\max_k |e_k|
-
\left|
e_{\arg\max_km_k}
\right|.
\]

数值越低越好。

该指标直接回答：

> 使用某个 predictor 选择一个成分进行干预，与选择真实最有效成分相比损失多少？

---

### 2.4.4 其他推荐指标

如果版面允许，可以继续报告：

- top-1 hit rate；
- top-3 hit rate；
- NDCG；
- pairwise ranking accuracy；
- top-\(K\) intervention curve；
- predictor 分位数与真实效应分位数的 calibration curve。

---

## 2.5 RQ3 分层分析

### A. Block 内 head ranking

在每个 block 内，对该 block 中的所有 heads 进行排序：

- ViT-B/16：每个 block 12 个 heads；
- DeiT-S：每个 block 6 个 heads。

该分析最接近实际使用问题：

> 在同一层中，Endpoint 能否找出有限替换效应最大的 head？

---

### B. 跨 block ranking

将所有 block 中同类成分合并排序。

但跨 block 结果必须与 block 内结果分开报告，因为跨 block 相关性可能主要来源于：

- 层间激活尺度差异；
- 梯度尺度差异；
- residual stream 范数差异；
- 深层和浅层的系统性效应差异。

不能用跨 block 的高相关替代同层内部的成分辨别能力。

---

### C. Early / middle / late 分层

建议固定为：

- early：blocks 0–3；
- middle：blocks 4–7；
- late：blocks 8–11。

全文必须统一使用 0-based 或 1-based 编号。

---

### D. 模型分层

分别报告：

- ViT-B/16；
- DeiT-S；
- pooled descriptive result。

合并结果只能作为描述性补充，不能代替模型级结果。

---

### E. Component family 分层

至少分别报告：

- head；
- routing；
- value；
- Q；
- K；
- V；
- residual identity；
- residual branch。

如果某个 comparison set 中只有两个或三个成分，例如某个 head 内只有 Q/K/V 三项，则 Spearman 的取值过于离散。此时建议使用：

- pairwise ranking accuracy；
- top-1 accuracy；
- 跨 heads 汇总后的 ranking；
- mixed-effects rank model。

不要把仅由三个元素计算的 Spearman 当作高精度连续统计量。

---

## 2.6 RQ3 统计分析

主统计单位必须是图像，而不是将所有成分视为独立样本。

推荐步骤：

1. 每张图分别计算 predictor 与有限效应之间的评价指标；
2. 对图像级指标计算中位数；
3. 使用 image-level bootstrap 给出 95% CI；
4. Endpoint 与 baseline 进行图像级配对比较；
5. 对配对差值使用符号检验或 Wilcoxon signed-rank test；
6. 对 family × baseline 的多重比较使用 Holm 校正。

Endpoint 相对某一 baseline 的配对差值定义为：

\[
d_i
=
\rho_i^{\mathrm{Endpoint}}
-
\rho_i^{\mathrm{baseline}}.
\]

需要分别回答两个统计问题。

### 绝对性能

如果只检验方向：

\[
H_0:
\operatorname{median}(\rho)\le 0.
\]

如果要检验实际效应门槛 0.10：

\[
H_0:
\operatorname{median}(\rho)\le 0.10.
\]

不能用“显著大于 0”代替“显著超过 0.10”。

### 相对性能

\[
H_0:
\operatorname{median}
\left(
\rho^{\mathrm{Endpoint}}
-
\rho^{\mathrm{baseline}}
\right)
\le 0.
\]

如果 Endpoint 不如 attribution patching，应如实报告，而不是将其视为实验失败。

---

# 三、M2：RQ4 跨层、全 head 扩展

## 3.1 研究目的

当前 RQ4 只覆盖 block 5 的少数 heads，结论范围偏窄。

扩展后的研究问题是：

> Endpoint 在空间区域选择上的负结果，是 block 5 的局部现象，还是跨层、跨 head 的稳定现象？

---

## 3.2 推荐 blocks

选择：

- block 2；
- block 5；
- block 10。

分别代表：

- early；
- middle；
- late。

如果使用 1-based 编号，应明确对应为第 3、6、11 个 block。

---

## 3.3 推荐 heads

优先使用三个 block 中的全部 heads：

### ViT-B/16

\[
3\times12=36
\]

个 heads。

### DeiT-S

\[
3\times6=18
\]

个 heads。

如果计算资源不足，可以每层冻结抽取 6 个 heads，但必须：

- 在查看结果之前冻结；
- 使用固定随机种子或 hash；
- 不根据 attention pattern、激活或现有结果挑选。

---

## 3.4 References

保留现有两种 reference：

- normalized-zero；
- blur。

如果 blur 存在多个强度，应：

- 冻结一个主要 blur 强度进入主分析；
- 其他 blur 强度放入敏感性分析；
- 不根据最终效果选择最优强度。

---

## 3.5 空间预算

建议至少使用以下预算：

- 1 patch；
- 5%；
- 10%；
- 25%。

M2 已执行的主要几何为 14×14 个 16×16 patch，预算为 1、10、20、49 个 patch。原 E5 几何是 4×4 个 56×56 区域的网格，并非 4×4 像素区域；该几何以 1、2、4 个区域作为次要比较。

可以比较：

1. 单 patch；
2. 2×2 patch square；
3. 现有固定像素区域；
4. 固定面积的 top-scoring region；
5. 连通区域与非连通 top-\(K\) patch 集合。

---

## 3.6 RQ4 方法比较

对同一个内部成分，至少比较：

1. Endpoint absolute mass；
2. Endpoint signed score；
3. regional Taylor；
4. gradient-only；
5. activation displacement；
6. random region；
7. center region；
8. attention-weight region，仅用于 attention 对象。

主要 outcome 为真实有限区域 replacement effect：

\[
\Delta F(R)
=
F(x)
-
F\bigl(
x;
\text{region }R
\leftarrow
R^{\mathrm{ref}}
\bigr).
\]

建议报告：

- 每个空间预算下的中位有限效应；
- 相对随机区域的提升；
- 相对候选集合中最优区域的 regret；
- block/head 分层结果；
- effect–budget curve；
- effect–budget curve 的 AUC。

---

## 3.7 RQ4 统计分析

以图像为主要统计单位，同时保留 block/head 的层级结构。

可以使用：

- 图像级 bootstrap；
- block-stratified bootstrap；
- 配对符号检验；
- Wilcoxon signed-rank test；
- mixed-effects model 作为补充。

例如，比较 Endpoint 与 regional Taylor：

\[
d_i
=
\Delta F_i(R_{\mathrm{Endpoint}})
-
\Delta F_i(R_{\mathrm{Taylor}}).
\]

如果分析 absolute intervention effect，则必须在所有方法中统一使用绝对值。

---

## 3.8 RQ4 结论边界

如果扩展后 Endpoint 仍然不优于 regional Taylor，可以写：

> Endpoint 的高分辨率输入图不会自动转化为更强的有限区域选择能力；与实际干预位移匹配的 activation-space Taylor 在 early、middle 和 late attention heads 上表现得更稳定。

如果负结果只在 middle 或 late block 中成立，则必须收缩结论，不能推广到所有层。

---

# 四、M3：冻结的分层 token-channel 神经元审计

M3 用于回应两个核心问题：

1. 当前 registry 是否真正覆盖 ViT 神经元？
2. 该工作与原始 NNS 的“神经元扫描”有多强的联系？

---

## 4.1 目标目录

对两个模型的全部 12 个 blocks，每层选择：

- 16 个 tokens；
- 32 个 channels。

每个模型、每张图像的目标总数为：

\[
12\times16\times32=6,144.
\]

目标定义为：

\[
s_{b,t,c}(x)
=
h_{b,t,c}(x).
\]

对应 Endpoint 为：

\[
E_{b,t,c}(x)
=
x\odot\nabla_xh_{b,t,c}(x).
\]

这些目标是真正的 token-channel 标量单元，比现有 registry 中的模块聚合目标更接近传统意义上的神经元。

---

## 4.2 Token 分层方式

每层选择 16 个 tokens：

- 1 个 CLS token；
- 15 个 patch tokens。

15 个 patch tokens 不应人工挑选，而应使用冻结的空间分层规则。

推荐方法：

1. 将 14×14 patch grid 划分为 15 个近似等面积空间 strata；
2. 在每个 stratum 中选择一个固定 patch；
3. 可以选择距离 stratum 中心最近的 patch；
4. 所有图像使用相同 patch 位置；
5. 规则必须在查看 Endpoint 结果前冻结。

这样可以覆盖：

- 中心与边缘；
- 左侧与右侧；
- 上部与下部；
- 多种空间位置。

如果不同模型使用不同输入分辨率，应根据归一化空间坐标匹配，而不是简单复制绝对 token index。

---

## 4.3 Channel 抽样方式

channel 没有自然空间顺序，因此建议：

- 每层独立抽取 32 个 channels；
- 使用固定随机种子或公开 hash；
- 所有图像使用相同 channel 列表；
- 两个模型分别生成冻结的 channel 列表；
- 将列表和生成脚本一起发布。

不得根据以下信息选择 channels：

- activation 大小；
- class selectivity；
- Endpoint 图是否清晰；
- 与 finite effect 的相关性；
- 现有 RQ3 表现。

否则会引入事后选择偏差。

---

## 4.4 图像规模

### 理想版本

在共享的 400 张图像上运行全部 6,144 个目标。

### 预算版本

如果计算成本过高，可以在原 400 张图像中冻结抽取 100 张。

建议使用 image ID hash，例如：

1. 对 image ID 计算固定 hash；
2. 按 hash 排序；
3. 选择前 100 张；
4. 在任何新增实验运行前冻结。

选择过程必须与以下因素无关：

- 模型是否预测正确；
- 类别；
-置信度；
- attribution 结果；
- 干预效果。

论文中可以称为：

> a deterministic 100-image neuron-audit subset of the frozen 400-image corpus.

---

## 4.5 计算与存储策略

不建议为 6,144 个目标、所有图像保存全分辨率 Endpoint map。

建议流式计算并保存以下汇总统计量：

- signed sum；
- absolute mass；
- L1 norm；
- L2 norm；
- positive mass；
- negative mass；
- cancellation ratio；
- spatial entropy；
- patch-level downsample map；
- 数值检查状态；
- 少量预冻结案例的完整 map。

Cancellation ratio 可定义为：

\[
C(E)
=
1-
\frac{
\left|\sum_iE_i\right|
}{
\sum_i|E_i|
}.
\]

其含义是：

- 接近 0：正负贡献抵消较弱；
- 接近 1：局部正负贡献抵消较强。

空间熵可以在 patch-level absolute mass 上计算：

\[
p_j
=
\frac{
\sum_{i\in\text{patch }j}|E_i|
}{
\sum_i|E_i|
},
\]

\[
H(E)
=
-\sum_jp_j\log p_j.
\]

上述比例和熵在总 absolute mass 大于 0 时定义，采用 \(0\log0=0\)。总质量为零时记录 NA 并单独统计发生率；不通过给分母加 epsilon 将零图解释为低熵或低抵消图。RGB 通道保留至取绝对值之后，再汇总 patch mass。

---

## 4.6 M3 报告内容

### A. Endpoint 生成成功率

报告：

- 成功生成目标比例；
- NaN 比例；
- Inf 比例；
- zero-gradient 比例；
- 数值验证通过率；
- 失败目标的层级和类型分布。

---

### B. 分层分布

分别报告：

- CLS vs patch token；
- early/middle/late；
- ViT-B/16 vs DeiT-S；
- 不同空间 strata；
- 不同 channels；
- 正确分类 vs 错误分类，可作为探索性分析。

主要指标包括：

- absolute mass；
- signed sum；
- cancellation ratio；
- spatial entropy；
- positive/negative balance；
- map sparsity；
- 层间尺度变化。

---

### C. 数值验证子集

不需要对全部 6,144 个目标执行昂贵的有限差分。

建议每层冻结：

- 2 个 tokens；
- 4 个 channels。

总计：

\[
12\times2\times4=96
\]

个目标/模型/图像。

对该子集执行：

1. autodiff Endpoint；
2. VJP/JVP 一致性；
3. 中心有限差分方向检查；
4. 多个 \(\epsilon\)；
5. float64 小批量复核。

中心有限差分定义为：

\[
D_\epsilon(v)
=
\frac{
s(x+\epsilon v)-s(x-\epsilon v)
}{
2\epsilon
}.
\]

比较：

\[
D_\epsilon(v)
\approx
\nabla_xs(x)^\top v.
\]

建议使用多个随机方向 \(v\)，并报告误差随 \(\epsilon\) 的变化曲线。

---

### D. 可选的神经元有限干预子集

如果资源允许，可以再冻结一个行为验证子集，每层选择：

- 4 个 tokens；
- 8 个 channels。

总数为：

\[
12\times4\times8=384
\]

个目标/模型/图像。

对这些 token-channel 单元执行 activation replacement，检验：

- neuron-level Endpoint mass；
- activation norm；
- gradient norm；
- activation × gradient；
- attribution patching；

与实际有限效应之间的关系。

如果资源不足，可以不执行这一部分，因为 M1 已承担主要行为验证。M3 的核心任务是建立：

- 更广泛的神经元覆盖；
- 原始 NNS 联系；
- 数值稳定性；
- 层级与空间分布。

---

## 4.7 M3 允许支持的主张

完成 M3 后，可以声称：

> 我们在两个模型中对冻结的、层级与空间分层的 6,144 个 token-channel 单元进行了神经元级 Endpoint 审计。

不能声称：

> 我们扫描了全部 ViT 神经元。

也不能声称：

> 这些目标在统计上代表所有 ViT 内部单元。

更准确的限定是：

> 该审计扩展了 token-channel 神经元在层级、空间位置和 channel 维度上的覆盖，但不包括全部 MLP hidden units、全部 Q/K/V coordinates 或全部 token-channel 组合。

---

## 4.8 Neuron-atlas 分布实验：输入依赖是否由局部变为全局

这一实验直接使用 M3 的 6,144 个 token-channel 目标及同一次生产的统计量。默认采用原 400 图中按固定规则抽取的 100 图，两模型使用相同图像；若采用 400 图，应在运行前确定。样本规模不能根据初步结果调整。

### 4.8.1 分布指标

对每个目标 \(h_{b,t,c}\)，令 \(a_j=\sum_{i\in P_j}|E_{b,t,c,i}|\)、\(p_j=a_j/\sum_ja_j\)。主要记录 normalized spatial entropy 和 signed cancellation：

\[
H_{norm}(E)=-\frac{\sum_{j=1}^{196}p_j\log p_j}{\log196},
\qquad
C(E)=1-\frac{|\sum_iE_i|}{\sum_i|E_i|}.
\]

同时保存 196 维 patch absolute mass 与 patch signed sum；两者分别用于空间分布和带符号图相似性，前者不能由后者取绝对值恢复。零质量和零梯度单列发生率。

### 4.8.2 局部性与依赖距离

熵升高只说明质量更分散。为了检验“局部到全局”，对 patch token \(t\) 增加相对其自身位置的局部质量和平均距离：

\[
L_t=\sum_{j:\|coord(j)-coord(t)\|_\infty\le1}p_j,
\qquad
D_t=\sum_jp_j\frac{\|coord(j)-coord(t)\|_2}{13\sqrt2}.
\]

报告 \(L_t\)、\(D_t\) 及它们减去均匀 patch 分布对应值后的差异，校正边缘邻域较小、几何距离不同的影响。CLS 无自身 patch 位置，只报告熵、抵消和空间图，不计算 own-token locality。

### 4.8.3 分层比较与神经元相似性

主要比较包括：

- early/middle/late 的 entropy、local mass、distance 和 cancellation；
- CLS 与 patch neuron 的差异；
- 中心 token 与边缘 token 的差异；
- ViT-B/16 与 DeiT-S 在相同图像上的差异；
- 同 token 不同 channel、不同 token 的 Endpoint 图相似性随深度的变化。

中心定义为行列都在 4–9 的 patch，边缘定义为距边界不超过 1 格的 patch，其余位置进入全部 patch 汇总。先在每图每层平均抽样 channels，再平均指定 token 集合，最后跨图像推断。不同模型的 channel ID 不作为同一神经元配对。

相似性使用 patch absolute distribution 的 cosine 和 patch signed vector 的 cosine，分别报告。每层从 same-token/different-channel 与 different-token 两组各固定抽取 256 个 unit pairs，规则与 seed 在计算前确定；聚类如需展示，仅作为描述性补充。

entropy 上升、own-token local mass 下降及距离增加共同支持所抽样单元的依赖趋于分散；仅凭几张清晰的图不能确认该层级规律。保留 §4.6 的数值验证与覆盖率说明。

---

# 五、注册覆盖率表

建议在论文中增加两个独立表格。

---

## 5.1 原始 registry 覆盖表

| 对象类型 | 全部候选对象定义 | 模型内候选总数 | 每图注册数 | 覆盖策略 | 是否单神经元 |
|---|---|---:|---:|---|---|
| post-block token-channel | \(B\times T\times D\) | 按模型计算 | 192 | 固定 token × channel 锚点 | 是 |
| patch sum/energy | 按注册聚合定义 | 按模型计算 | 96 | 固定聚合目标 | 否 |
| attention-head coordinate | 按 head 输出标量定义 | 按模型计算 | 144/72 | 固定 head 坐标 | 是或局部聚合 |
| module patch aggregate | 按模块定义 | 按模型计算 | 18 | 固定模块聚合 | 否 |
| patch embedding aggregate | 按注册定义 | 按模型计算 | 4 | 固定聚合 | 否 |
| final CLS coordinate | \(D\) | 768/384 | 1 | 固定 channel | 是 |
| ground-truth logit | 类别数 | 1,000 | 1 | 样本依赖 | 输出 |
| clean-top1 logit | 类别数 | 1,000 | 1 | 样本依赖 | 输出 |

需要在正文中明确：

> “Complete scan”仅表示所有预注册 registry entries 均被执行且没有缺失，不表示所有 ViT 标量单元均被扫描。

如果 ground-truth 与 clean-top1 在正确分类样本上相同，还应区分：

- registry entry 数量；
- unique scalar function 数量；
- unique Endpoint map 数量。

---

## 5.2 补充神经元审计覆盖表

| 模型 | Blocks | 每层 token | 每层 channel | 每图目标数 | 图像数 | 覆盖类型 |
|---|---:|---:|---:|---:|---:|---|
| ViT-B/16 | 12 | 16 | 32 | 6,144 | 100 或 400 | 分层 token-channel |
| DeiT-S | 12 | 16 | 32 | 6,144 | 100 或 400 | 分层 token-channel |

ViT-B/16 的全部 post-block token-channel 数量为：

\[
12\times197\times768
=
1,815,552.
\]

补充审计覆盖比例为：

\[
\frac{6,144}{1,815,552}
\approx0.338\%.
\]

如果 DeiT-S 使用 197 tokens 和 384 hidden dimensions，则全部 post-block token-channel 数量为：

\[
12\times197\times384
=
907,776.
\]

覆盖比例为：

\[
\frac{6,144}{907,776}
\approx0.677\%.
\]

这些比例应直接报告，以避免给读者造成“覆盖了全部神经元”的印象。

---

# 六、M4：与 Transformer 解释方法比较

不同解释方法回答的问题不同，不建议将所有方法强行放入同一个排行榜。

应根据任务类型分组比较。

---

## 6.1 成分级有限效应排序

适合比较：

- Endpoint mass；
- activation norm；
- gradient norm；
- activation × gradient；
- reference displacement norm；
- attribution patching / Taylor；
- random。

评价问题是：

> 哪个 head、routing/value 或 residual component 的有限 replacement effect 更大？

这部分由 M1 完成。

---

## 6.2 输出级空间解释

对 ground-truth logit 或 clean-top1 logit，可以比较：

- Endpoint logit map；
- Input × Gradient；
- Integrated Gradients；
- attention rollout；
- Transformer Grad-CAM；
- Chefer relevance；
- AttnLRP。

评价方式可以包括：

- pixel/patch deletion；
- reference replacement；
- insertion/deletion curve；
- top-region effect；
- pointing game；
- object localization，如果数据具有标注。

必须明确：

\[
\text{Endpoint logit map}
=
x\odot\nabla_xF
\]

在基础公式上就是 normalized-input 坐标中的 Input × Gradient。

因此 Endpoint 与 Input × Gradient 不能被写成两个完全独立的方法。可以比较的是不同坐标原点：

### Normalized-coordinate Endpoint

\[
x\odot\nabla_xF.
\]

### Raw-RGB Input × Gradient

\[
r\odot\nabla_rF.
\]

### Mean-centered raw-RGB 形式

\[
(r-\mu)\odot\nabla_rF.
\]

其中：

\[
x_i\frac{\partial F}{\partial x_i}
=
(r_i-\mu_i)\frac{\partial F}{\partial r_i}.
\]

---

## 6.3 内部成分空间解释

对特定 head、routing/value 或 residual branch，可以比较：

- Endpoint component map；
- regional Taylor；
- gradient-only map；
- attention weights；
- internal-target IG，如果实现可用。

Attention rollout、Chefer 等输出级方法不一定能够自然对应某个内部成分的有限 replacement，因此不应强行放入成分级比较。

---

## 6.4 方法适用性矩阵

建议增加：

| 方法 | 输出空间图 | 内部标量目标 | 成分排序 | Reference-matched | Completeness |
|---|---:|---:|---:|---:|---:|
| Endpoint | 是 | 是 | 是 | 否 | 一般否 |
| Integrated Gradients | 是 | 是 | 可扩展 | 是 | baseline difference |
| Attribution patching | 否或间接 | 是 | 是 | 是 | 局部一阶 |
| Attention rollout | 是 | 否 | 否 | 否 | 否 |
| Chefer/AttnLRP | 是 | 部分 | 部分 | 否 | 依实现而定 |
| Grad-CAM | 是 | 部分 | 否 | 否 | 否 |

该表可以帮助读者理解：

> 不同方法并非都在回答同一个解释问题。

---

# 七、M5：Reference 复用、成分筛选与结构诊断

M5 包括两条共享数据的分析线：§7.1–7.4 使用 attention/residual 分解研究结构，§7.5–7.6 共用“成分 × reference”有限效应表研究复用与筛选。结果可以呈现性能—成本权衡，也可以表明某项增量未获支持。

---

## 7.1 True softmax Jacobian vs frozen attention

完整 attention differential 为：

\[
\mathrm dO
=
(\mathrm dA)V
+
A(\mathrm dV).
\]

Frozen-attention approximation 为：

\[
\mathrm dO
\approx
A(\mathrm dV).
\]

对应的 Jacobian 作用为：

\[
J_{O,x}v=(J_{A,x}v)V+A(J_{V,x}v),
\qquad
J_{O,x}^{FA}v=A(J_{V,x}v).
\]

每次只在被测 block/head 将 clean attention 概率固定，保留相同前向数值，并切断该位置对 Q/K 的反向传播；其余层、heads 和 residual 正常计算。固定完整模型的 clean cotangent \(g_h=\nabla_{O_h}F\)，得到：

\[
E_h^{full}=E_h^{routing}+E_h^{value},
\qquad E_h^{FA}=E_h^{value}.
\]

这衡量一个位置的 frozen-attention 近似，不与全网络同时冻结 attention 混为一项。覆盖两模型、400 图、全部 12 层与全部 heads。

主要报告各 head 的 omitted routing mass、绝对误差、相对误差及完整分解闭合残差：

\[
R_h^{routing}=
\frac{\|E_h^{routing}\|_1}
{\|E_h^{routing}\|_1+\|E_h^{value}\|_1},
\qquad
\eta_h^E=\frac{\|E_h^{full}-E_h^{FA}\|_2}{\|E_h^{full}\|_2}.
\]

分支份额 \(R_h^{routing}\) 在 [0,1]；routing 与 value 抵消时，相对误差可超过 1。分母为零时记录 NA 并报告绝对误差。同时计算未乘输入 \(x\) 的 contracted-gradient 误差，避免小输入坐标掩盖导数差异。

用 native autodiff、显式 Q/K/V 分解和 attention detach 实现交叉检查，验证 forward parity、完整闭合以及 \(E^{full}-E^{FA}=E^{routing}\)。完整 Jacobian 的作用另用每图固定 8 个单位范数 Rademacher 方向做 JVP 检查：

\[
\eta_h^{JVP}=
\sqrt{\frac{\sum_v\|(J_{O_h,x}-J_{O_h,x}^{FA})v\|_2^2}
{\sum_v\|J_{O_h,x}v\|_2^2}}.
\]

该值是抽样方向上的误差，Endpoint map 则是特定 cotangent 下的输入坐标表示。报告逐层、逐 head 和 early/middle/late 差异；按跨图中位误差列出最不适合 frozen attention 的 heads，附 bootstrap 排名频率。

完整分解的闭合残差用于检查实现，省略 routing 后的残差用于测量近似误差。经验发现应回答遗漏在哪些层和 heads 更大，而不把代数等式本身写成机制发现。

---

## 7.2 Routing vs value

完整分解为：

\[
E^{\mathrm{attn}}
=
E^Q+E^K+E^V
=
E^{\mathrm{routing}}
+
E^{\mathrm{value}},
\]

其中：

\[
E^{\mathrm{routing}}
=
E^Q+E^K.
\]

沿用 §7.1 的分解，报告 routing/value mass、Q/K cancellation、early/middle/late 变化及两模型差异。E3 已有整体 routing/value 汇总可复用；逐 head 结果必须使用该 head 的 cotangent，不能将整体比例平均分配给 heads。

Attention saturation 定义为平均 query-row normalized entropy 的补数：

\[
s_h=1-\frac1{197}\sum_q
\frac{-\sum_{j=1}^{197}A_{h,qj}\log A_{h,qj}}{\log197},
\qquad 0\log0=0.
\]

同时保存 CLS query 的 saturation。每图每层在 heads 内比较 \(s_h\) 与 routing share、\(\eta_h^E\) 的关系，再按图像汇总，避免将共同的层深趋势当作 head 内关系。

Saturation 与误差关系采用双向检验。尖锐的 softmax 可缩小 routing Jacobian，其实际误差还受 Q/K/V 和 downstream cotangent 影响，方向由结果确定。若某些 heads 的 routing mass 和独立 JVP 误差持续较大，可报告 frozen attention 在这些位置遗漏较多局部输入依赖；有限替换效用仍使用对应干预结果评价。

---

## 7.3 Residual identity

对于：

\[
y=u+f(u),
\]

局部 Jacobian 为：

\[
J_{y,u}
=
I+J_{f,u}.
\]

对模型输入 \(x\)，固定 residual 输出的 clean cotangent \(g_y=\nabla_yF\)，两条路径为：

\[
E^{identity}=x\odot J_{u,x}^{\mathsf T}g_y,
\qquad
E^{branch}=x\odot J_{u,x}^{\mathsf T}J_{f,u}^{\mathsf T}g_y.
\]

Identity 使用 \(g_y\)，不能使用已经包含 branch 的 \(\nabla_uF\)。对应 Endpoint 闭合为：

\[
E^{\mathrm{output}}
=
E^{\mathrm{identity}}
+
E^{\mathrm{branch}}.
\]

分别对 attention residual 的 identity/attention 和 MLP residual 的 carry/MLP 进行分析。模型函数保持原样，检查完整闭合，并以省略非零 identity 的记账结果作为负向控制。

定义 identity/branch 的 absolute mass 份额与跨路径抵消：

\[
R^{identity}=
\frac{\|E^{identity}\|_1}{\|E^{identity}\|_1+\|E^{branch}\|_1},
\qquad R^{branch}=1-R^{identity},
\]

\[
C^{path}=1-
\frac{\|E^{identity}+E^{branch}\|_1}
{\|E^{identity}\|_1+\|E^{branch}\|_1}.
\]

报告每层质量份额、signed sum、正负质量和 cancellation，比较 early/middle/late、两模型以及 attention/MLP residual。用 identity 与 branch 的 cosine 检查方向关系，用 output 与 identity 的 patch absolute distribution 差异检查空间依赖是否改变。

由此检验：深层局部输入依赖是否更多经 carry 传播；branch 质量大时是否被 identity 抵消；哪些层主要放大既有依赖，哪些层改变其空间分布。各层 cut 都在分解同一个目标的输入依赖，不能将不同层 branch mass 相加视为总贡献，也不能把局部质量份额解释为删层后的性能保持率。

---

## 7.4 Signed vs absolute mass

比较：

### Signed aggregate

\[
m_k^{\mathrm{signed}}
=
\left|
\sum_iE_{k,i}
\right|.
\]

### Absolute mass

\[
m_k^{\mathrm{absolute}}
=
\sum_i|E_{k,i}|.
\]

同时报告 cancellation ratio：

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

总质量为零时记 NA。这里的 signed cancellation 衡量单张 map 内的正负抵消，§7.3 的 path cancellation 衡量同一输入坐标上的跨路径抵消，两者分别报告。

检验以下问题：

- cancellation 高时 absolute mass 是否高估有限效应；
- 未取绝对值的 signed sum 是否更适合预测效应方向；\(|\sum_iE_i|\) 本身不包含方向信息；
- absolute mass 是否更适合预测效应大小；
- routing 和 value 是否具有不同的 cancellation 模式。

---

## 7.5 Reference 稳定性与复用实验

### 7.5.1 问题与共同设置

问题是：在 reference 未指定或需要多次更换时，一个固定排名能提供怎样的效用与成本？Endpoint 的排名不随干预 reference 变化是定义属性，实验需要测量这种复用的实际效果。

主分析沿用共享 400 图、两模型、ground-truth logit，比较全部 12 层内的全部 heads；不将跨层尺度差异混入层内排名。Clean-top1 logit 可另作次要 target 分析。对 head \(k\)，使用同一 projection 输入 slice 的激活 \(z_k\) 和 clean 梯度 \(g_k=\nabla_{z_k}F\)。

Reference-free 指生成排名不需要指定干预 reference。Endpoint 仍依赖坐标原点：若 \(x=(r-\mu)/\sigma\)，则 \(x\odot\nabla_xF=(r-\mu)\odot\nabla_rF\)。这不表示它对预处理变化保持不变。

### 7.5.2 六类 reference

| Reference | 定义 | 需要记录的参数 |
|---|---|---|
| normalized-zero | 归一化输入为 0，原始 RGB 等于预处理均值 \(\mu\) | 模型预处理均值与标准差 |
| blur | 原始 RGB Gaussian blur 后归一化 | 沿用 M2：kernel 31、sigma 10 及同一边界处理 |
| dataset mean | 独立 calibration 图像集的经验 RGB 三通道均值，广播至全图 | 数据来源、图像列表、变换与实际均值 |
| black | 原始 RGB 为 0，归一化后为 \(-\mu/\sigma\) | 与 normalized-zero 分开 |
| noise | 原始 RGB 为 \(\operatorname{clip}(0.5+0.2\epsilon,0,1)\)，\(\epsilon\sim N(0,1)\) | 每图固定 3 次采样，记录 seed 与生成器 |
| another image | 独立 donor 图像经相同评估变换后归一化 | 每图固定 3 个不同 donor，记录配对 |

Calibration 与 donor 共用一个独立的 1,000 图集合，优先从 ImageNet training split 按固定随机种子抽取，排除评估图像；不用类别、预测或效应挑 donor。实际可用来源须在计算前确定。Dataset mean 是该集合经过固定评估变换后的经验均值，不用常见预处理常量代填。若其最终参考张量与 normalized-zero 相同，则作为同一 reference 的别名，在平均中只计一次。

六类 reference 等权；noise 与 donor 的重复先在类内平均。最低表现指这组预定 reference 中的最低表现，不代表对任意 reference 的保证。

### 7.5.3 效应表与比较方法

对 reference \(r\) 的实例 \(j\)，计算同成分替换效应：

\[
e_{k,r,j}=F(x)-F(x;z_k\leftarrow z_k(x_{r,j}^{ref})),
\qquad
y_{k,r}=\frac1{J_r}\sum_j|e_{k,r,j}|.
\]

其余 heads 与 residual bypass 保持 clean。先对每实例取绝对值，再平均，保留 signed effect 供分析。

| 方法 | 分数/策略 | 所需 reference 信息 |
|---|---|---|
| Endpoint | \(\lVert x\odot J_{z_k,x}^{\mathsf T}g_k\rVert_1\) | 无 |
| Activation norm | \(\lVert z_k\rVert_2\) | 无 |
| Gradient norm | \(\lVert g_k\rVert_2\) | 无 |
| Activation × gradient | \(\lvert g_k^{\mathsf T}z_k\rvert\) | 无外部输入 reference |
| Attention weight | \(\sum_{t=1}^{196}A_{k,CLS,t}=1-A_{k,CLS,CLS}\) | 无，限 head 集合 |
| Random | 固定 1,000 个随机排列，逐次评价后平均 | 无，排列跨 reference 复用 |
| Reference-specific Taylor | \(t_{k,r}=J_r^{-1}\sum_j\lvert g_k^{\mathsf T}(z_k-z_k^{ref_{r,j}})\rvert\) | 当前 reference |
| Anchor Taylor | 用 normalized-zero 生成排名，在所有 reference 上复用 | 一个固定 reference |
| Multi-reference Taylor | 每类 Taylor 分数先转 percentile rank，再跨类平均 | 全部 reference |
| Leave-one-reference-out Taylor | 平均其余类的 Taylor percentile rank，在留出类评价，轮换全部类 | 除测试类以外的 reference |

Attention 所有 query 行的总和恒定，不能直接用作 head 分数；上述 CLS-to-patch mass 另报离散度与 ties。所有方法使用同一候选集合，top-K 同分按 component ID，Spearman 使用 average ranks。Taylor 的 percentile rank 定义为 \((rank_{ascending}-1)/(H-1)\)，\(H\) 为该层 head 数，ties 使用 average ranks。

### 7.5.4 性能与排名稳定性

主预算 \(K=3\)，\(K=1\) 为补充。每图每层每 reference 计算 §2.4 的 Capture@K、Spearman 和 Regret@1，设 \(U_{b,r,m}=\operatorname{Capture@3}(m;y_r)\)。主要报告：

\[
U^{ref}_{r,m}=\frac1{12}\sum_bU_{b,r,m},\quad
U^{avg}_m=\frac1{12}\sum_b\frac1R\sum_rU_{b,r,m},\quad
U^{worst}_m=\frac1{12}\sum_b\min_rU_{b,r,m}.
\]

归一化 capture 为主，原始 logit 效应同时报告，避免高效应尺度的 reference 单独主导平均。列出最差 reference 的发生频次。

跨 reference 排名稳定性为：

\[
S_m=\frac{2}{R(R-1)}\sum_{r<s}
\operatorname{Spearman}(m_{\cdot,r},m_{\cdot,s}).
\]

另报 Top-3 overlap \(|\operatorname{Top3}(m_r)\cap\operatorname{Top3}(m_s)|/3\)。Endpoint 与其他固定排名的 overlap 为 1，非恒定分数的 Spearman 为 1；这验证复用一致性，不等于筛选有效。Constant ranks 的 Spearman 记 NA。

增加 Taylor 迁移矩阵：在 reference \(r\) 上生成排名，在 \(s\) 上计算 capture；对角线是 reference-specific 性能。实际有限效应排名 \(y_r,y_s\) 的稳定性也要报告，以区分“方法排名变化”和“任务的重要成分本身变化”。

### 7.5.5 每增加一个 reference 的成本

分别记录第一次生成排名的成本和复用时的增量：

\[
C_m(R)=C_m^{initial}+\sum_{r=1}^R\Delta C_{m,r}.
\]

Endpoint 首次获取各 head VJP 的时间、显存和缓存全部计入。Taylor 复用 clean 激活与梯度，新增 reference 通常只需 reference forward、位移及点积，不能按每次重做 clean backward 计费。固定 reference 的跨图缓存和批处理同样计入。

报告独立运行总成本与共享 clean cache 后的边际成本，包括 forward/VJP 次数、同步计时、GPU-hours、峰值显存及缓存大小。有限效应评价作为所有方法共享的评估成本单列。固定排名的 reference 增量重算次数为 0，读取缓存仍计实际成本。成本曲线实际出现交叉时才报告交叉点。

### 7.5.6 与 M2 对齐的空间复用补充

空间复用沿用 blocks 2/5/10、全部 heads、1/10/20/49 patch 预算及 M2 方法定义，扩展到上述 reference。完整输入效应和单 head 隔离效应分别报告 reference-specific、平均和最差 AUC，附各类原始效应尺度。

区域 Taylor 仍使用区域替换后指定 head 的有限激活位移。在 reference \(r\) 生成的选区迁移至 \(s\) 时，应实际计算该选区在 \(s\) 上的效应。此项为固定设置下的描述性复用分析，不追加区域形状或 blur 强度搜索。

## 7.6 Reference-free 成分筛选

### 7.6.1 两阶段使用场景

第一阶段只使用 clean 输入、模型、目标和候选集合，保存 Endpoint、activation norm、gradient norm、activation×gradient、attention weight、random 的分数与 Top-1/Top-3 列表。第二阶段指定 §7.5 的 reference，使用同一张效应表评价，候选列表不再调整。

Taylor 参加第二阶段比较，并按 reference-specific、anchor、multi-reference、leave-one-reference-out 标注信息条件。第一阶段的筛选代码不读取 reference 激活和有限效应；研究者已经看到 M1/M2 的结果，这属于明确披露的补充实验，不重新声称原研究盲法。

### 7.6.2 跨 reference 效应与覆盖率

定义平均跨 reference 重要性与 reference-robust 重要性：

\[
\bar y_k=\frac1R\sum_ry_{k,r},
\qquad
y_k^{robust}=\min_ry_{k,r}.
\]

这里 robust 表示每类 reference 下都有较大的绝对效应。\(\max_ry_{k,r}\) 则表示至少在一类下效应大，是另一种问题。

主要指标为：

\[
\operatorname{MeanCapture@3}=\operatorname{Capture@3}(m;\bar y),
\qquad
\operatorname{RobustCapture@3}=\operatorname{Capture@3}(m;y^{robust}),
\]

\[
\operatorname{MultiRefCoverage@3}=
\frac1R\sum_r\frac{|\operatorname{Top3}(m)\cap\operatorname{Top3}(y_r)|}{3}.
\]

先在每图每层计算，再平均 12 层。MeanCapture 是“平均效应上的 capture”，不同于 §7.5 的“各 reference capture 的平均”；RobustCapture 是“逐成分最小效应上的 capture”，不同于固定集合的最差表现。

用 \(q_{k,r}=y_{k,r}/\sum_ky_{k,r}\) 替换原始效应再计算平均/最小值，作为效应尺度敏感性分析。所有零分母返回 NA 并报告发生率。

Endpoint 的筛选增量由它相对五种 clean-only 基线的配对差值判断。Taylor 在已知 reference 时更准、固定表示便于统一复用，是可检验的权衡；Endpoint 是否比其他固定表示更有用，必须由这些比较回答。

## 7.7 新增实验的统计与报告

图像是统计单位，不能把同图的 heads、references、tokens 或 channels 当成独立样本。重复实例、heads/层或 token/channel 按各节规则先在图像内聚合，两模型差异按相同 image ID 配对。

使用中位数、10,000 次图像 bootstrap、95% CI；主要配对差值用双侧精确符号检验和 Holm 校正，零差值单列，不计入正负样本数。Reference 平均/最差指标须保留该图全部规定 reference；缺少某类时不能用剩余类替代。

主检验按以下模块分别构成 Holm family，并报告完整比较数；模型分别检验，不另合并一个 omnibus：

| 分析 | 主要比较 | 其余报告 |
|---|---|---|
| Reference 复用 | Endpoint−reference-specific Taylor：各 reference、平均、最差 Capture@3 | Anchor/多 reference/留一法、稳定性、成本、空间 AUC |
| 成分筛选 | Endpoint−五种 clean-only 基线：MeanCapture、RobustCapture、MultiRefCoverage | K=1、尺度敏感性、Taylor 信息条件对照 |
| Routing 诊断 | routing share 的 late−early；head 内 saturation/error 相关系数相对 0 | 逐层/head、JVP 误差与排序频率 |
| Residual 诊断 | 两 residual 类型的 identity share、path cancellation 的 late−early | 模型间差异、cosine 与空间分布 |
| Neuron atlas | patch entropy/local mass/distance/cancellation 的 late−early；CLS−patch entropy | 中心/边缘、模型差异、相似性/聚类 |

六类 reference 均不同的情况下，上述五个 family 分别包含 16、30、4、8、10 项双侧检验，均包含两个模型。若 reference 去重，第一组按 \(2(R+2)\) 重新确定；实际列表在新增结果计算前固定。未执行或无定义的项不从校正范围删除，使用 p=1 占位并保留对应状态。

Routing share 每层先平均 heads，再分别平均 early/late 的四层；saturation/error 的 head 内相关系数先算每图每层，再平均层。Residual 的两类 cut 分开计算；neuron 指标按 §4.8 的 channel/token 权重聚合，CLS−patch entropy 在全部层平均后比较。

有效成对图像少于原 cohort 的 90% 时该比较记为指标不足；缺失任务或数值错误须与合法的零质量分开记录。图像内聚合也要求至少 90% 的规定单元指标有效，并报告分母；reference 平均/最差仍要求全部规定 reference 有定义。未通过校正检验表示对应增量未获支持，不表示等效。CI 是逐比较区间。

复用已见结果的分析标为 post-hoc reanalysis，新增数据的分析规则在计算前确定。原 C4a/C4b 仍为 NOT_SUPPORTED。结构分解的闭合和排名不变属于方法性质，层/head/神经元分布才是经验发现；标准 autodiff 同样可实现这些诊断，论文贡献聚焦统一表示、诊断流程和实际观测。

---

# 八、M6：失败案例与机制可视化

失败案例不能由研究者人工挑选，应使用预先冻结的自动规则。

---

## 8.1 案例类别

建议至少展示以下类型：

1. Endpoint 和 Taylor 都成功；
2. Endpoint 成功、Taylor 失败；
3. Endpoint 失败、Taylor 成功；
4. Endpoint 和 Taylor 都失败；
5. attention saturation 较高；
6. Q/K routing 与 V 强抵消；
7. signed cancellation 高但 absolute mass 大；
8. M2 中 Endpoint 选择错误区域；
9. early 与 late block 对比；
10. CLS neuron 与 patch neuron 对比。

---

## 8.2 冻结选择规则

例如：

### Endpoint 成功案例

Endpoint Spearman 位于全部图像的最高 5%。

### Endpoint 失败案例

Endpoint Spearman 位于最低 5%。

### Taylor 明显反超案例

\[
\rho_{\mathrm{Taylor}}
-
\rho_{\mathrm{Endpoint}}
\]

位于最高 5%。

### 高抵消案例

Cancellation ratio 位于最高 5%。

在满足条件的候选集合中，可以使用：

- 最小 image ID；
- 最小 hash；
- 固定随机种子抽取。

不能根据图片视觉效果再手工选择。

---

## 8.3 每个案例展示内容

建议包括：

- 原始图像；
- Endpoint signed map；
- Endpoint absolute map；
- regional Taylor map；
- attention map；
- predictor 选中的区域；
- 实际最优区域；
- 实际有限效应；
- predictor ranking；
- block/head/target/reference；
- cancellation ratio；
- attention entropy 或 saturation；
- Endpoint 与 Taylor 的排序指标。

---

# 九、原始 RQ3 统计结论必须重写

无论新增实验结果如何，原始确认性分析的结论都必须保持：

> 严格的预注册 all-family omnibus 标准未得到支持，因为 12 个 family 中有 1 个未达到预注册的 0.10 点估计门槛。其余 11 个 family 满足全部 family-level 条件，未通过的 family 仍表现出显著正相关。因此，结果支持广泛但非普遍的成分级关联，而不支持无例外的全 family 结论。

摘要中不建议再写：

> RQ3 整体支持。

建议写：

> RQ3 在 family level 上表现出广泛但非普遍的支持，而预注册的 all-family omnibus 标准未得到支持。

英文可以写成：

> RQ3 showed broad but non-universal support at the family level, while the preregistered all-family omnibus criterion was not supported.

对于点估计 0.099565 与阈值 0.10 的关系，不要过度强调“只差 0.000435”。应写：

> 该点估计接近预注册门槛，但其置信区间同时覆盖门槛上下，因此不能确认该 family 达到预设的实际效应水平。

---

# 十、根据新增实验结果调整结论

## 10.1 如果 Endpoint 优于简单基线，但不如 Taylor

可以写：

> Endpoint mass 在部分 component families 中优于 activation norm、gradient norm 和 displacement-only baselines，表明其包含超出单纯尺度信息的成分级排序信号。然而，reference-matched attribution patching 通常更准确地预测具体有限替换效应。

---

## 10.2 如果 Endpoint 与简单基线相当

可以写：

> Endpoint 的主要价值不在于成为最优的有限效应 predictor，而在于为不同 ViT 内部目标提供统一、输入坐标化且可进行结构闭合审计的表示。

---

## 10.3 如果 Endpoint 明显优于 Taylor

即使出现这种结果，也要进一步检查：

- reference 是否实现正确；
- Taylor 梯度点是否一致；
- replacement 位移是否定义正确；
- component tensor 是否匹配；
- 是否存在 layer normalization 或 downstream nonlinearities 导致高阶效应；
- comparison set 是否受跨层尺度驱动。

不能仅凭结果反常就直接宣称 Endpoint 更适合有限 replacement。

---

## 10.4 M2 完成后的结论

M2 已观察到 Taylor 的优势，具体计数与范围见本文开头及 M2 结果报告。高分辨率 Endpoint 图并未自动转化为更优的有限区域选择。后续研究按 §7.5–7.6 检验统一复用与候选筛选，按 §7.1–7.4 和 §4.8 研究结构与 neuron 分布。

---

# 十一、推荐的论文实验章节结构

## 5.1 Experimental protocol and frozen registry

包括：

- 模型；
- 数据；
- 原始目标目录；
- 原确认性协议；
- 补充冻结协议；
- confirmatory 与 exploratory 的边界。

---

## 5.2 Registry execution and numerical correctness

包括：

- 扫描完成率；
- Endpoint identity；
- attention/residual 闭合；
- JVP/VJP；
- 中心有限差分；
- 负向控制；
- 数值容差合同。

---

## 5.3 Actual registry coverage

包括：

- 原始 registry 覆盖表；
- 每类目标的候选总体；
- 每图注册数量；
- 覆盖率；
- registry-complete 的定义；
- 与全部 ViT 神经元的区别。

---

## 5.4 RQ3: Component-level finite-effect prediction

包括：

- 原始 family 结果；
- omnibus `NOT_SUPPORTED`；
- M1 强基线；
- block 内 ranking；
- 跨 block ranking；
- early/middle/late；
- 模型差异；
- top-\(K\) 指标；
- calibration 和 regret。

---

## 5.5 RQ4: Spatial finite-intervention selection

包括：

- blocks 2/5/10；
- 全部 heads；
- 多空间预算；
- 两种 references；
- Endpoint；
- regional Taylor；
- random；
- gradient；
- effect–budget curve。

---

## 5.6 Stratified token-channel neuron audit

包括：

- M3 的 6,144 个目标；
- token 选择规则；
- channel 选择规则；
- 图像子集规则；
- 数值验证；
- 层级和空间分布；
- entropy、own-token locality、依赖距离与 cancellation 的深度变化；
- CLS/patch、中心/边缘、两模型及 neuron 相似性；
- 可选神经元 replacement。

---

## 5.7 Comparison with Transformer explanation methods

包括：

- 方法适用性矩阵；
- 成分级比较；
- 输出级空间比较；
- 内部对象空间比较；
- Path IG 或 IG 校准情况。

---

## 5.8 Ablations and failure analysis

包括：

- frozen attention；
- routing/value；
- residual identity；
- signed/absolute；
- target/reference；
- 多 reference 的平均/最差效用、排名稳定性与实际复用成本；
- reference 未指定时的成分筛选和高效应成分覆盖率；
- 自动冻结的失败案例。

---

# 十二、Path IG 的补充处理

如果 ViT-B/16 在最多 512 个积分点下仍未通过 Path IG 校准门，应补充报告：

1. completeness residual 随积分点数量的曲线；
2. midpoint 与 trapezoid；
3. Gauss–Legendre；
4. float32 与 float64；
5. adaptive quadrature；
6. 不同 baseline；
7. 失败图像数量；
8. 最大和中位 residual；
9. DeiT-S 是否通过；
10. 哪些模型阶段导致数值变化较大。

如果仍然失败，应明确：

> Path IG 未进入确认性跨模型比较，因为它未满足预注册数值校准标准。

但可以将其作为探索性结果报告，不能完全省略失败细节。

---

# 十三、补充实验的最低可行版本

M1、M2 已完成，保留其结果。后续优先完成 M3、M5 新增实验及对应报告；以下列表用于核对覆盖范围。

## 必做 1：M1 强基线

加入：

- activation norm；
- gradient norm；
- activation × gradient；
- displacement norm；
- attribution patching；
- random。

---

## 必做 2：M1 分层分析

至少报告：

- block 内 head ranking；
- early/middle/late；
- 两个模型分开；
- Spearman；
- Capture@K；
- Regret@1。

---

## 必做 3：M2 跨层扩展

至少执行：

- blocks 2、5、10；
- 每层全部 heads；
- normalized-zero 和 blur；
- Endpoint vs regional Taylor vs random。

---

## 必做 4：M3 神经元审计

资源充足时：

- 400 张图；
- 每图 6,144 个目标。

资源有限时：

- 冻结 100 张图；
- 每图 6,144 个目标；
- 流式保存汇总统计量；
- 对嵌套的 96 个目标执行独立数值验证。

---

## 必做 5：报告和定位修正

必须完成：

- omnibus 保持 `NOT_SUPPORTED`；
- 明确 11/12 family 满足 family-level 条件；
- 使用“广泛但非普遍”；
- 增加 coverage table；
- “complete”仅指 registry-complete；
- 不再声称扫描了全部 ViT 神经元；
- 明确 Endpoint 是 normalized-input Gradient × Input；
- 明确结构分解是局部 Jacobian 分解，而不是函数值因果分解。

## 新增重点：M5 复用、筛选与结构诊断

- Reference-specific、across-reference average、worst-reference performance；
- 固定排名与 Taylor 的迁移稳定性、额外计算成本；
- 平均重要性、reference-robust importance、multi-reference top-coverage；
- Frozen-attention 遗漏的 routing、闭合残差与 saturation 关系；
- Attention/MLP residual 的 identity/branch 份额和 signed/path cancellation。

---

# 十四、后续执行顺序与必要记录

本文件是阶段 M 的统一方案入口。实验参数与实际结果在对应章节更新；实现需要的配置复用已有结构，避免为同一事项重复建立协议、附录和审计文件。历史实验保留原配置与结果，复现时使用对应 Git 提交中的当时版本。

| 顺序 | 工作 | 数据复用与交付 |
|---|---|---|
| 1 | 汇总 E3/M1 已有结构数据，开展 M5 routing/residual 分析 | 复用可重算的记录；缺少逐 head 分解或图形统计时补算所需内容，交付层/head 诊断表 |
| 2 | 按 M3 生成分层 neuron atlas，并完成 §4.8 分布分析 | 同一次生产保存数值验证、patch 图与汇总，不为分布分析重跑 atlas |
| 3 | 确定 reference 来源及参数，先保存 clean-only 筛选结果，再计算新增 reference 效应 | M5 §7.5–7.6 共用效应表，继承 M1 中定义一致的 normalized-zero/blur 数据 |
| 4 | 完成 reference 复用、筛选比较与成本测量 | 输出各 reference/平均/最差效用、三种筛选指标、迁移矩阵和成本曲线 |
| 5 | 按 §7.5.6 完成空间复用补充，再整合 M6 图示与论文解释 | 复用 M2 可用记录，补足新 reference/迁移选区；保留完整输入与隔离效应的区别 |

这是一份后续执行计划，本次文档更新不表示这些实验已经运行。每次实际执行按指定模块推进。

运行前记录影响科学解释的设置即可：

- 图像、token/channel 与 donor 的实际列表，模型、目标与层/head 范围；
- reference 的坐标、数据来源、变换、噪声参数及 seed；
- 方法、预算、指标、聚合顺序、NA 规则、统计比较与数值容差；
- 已有结果的复用范围，以及实际计算的新增部分。

运行后记录每项完成数、异常/NA 数、实际计算成本和主要结果，并保留能重算图表的逐图数据。可视化保存 PNG/PDF，配简短编号说明：reference 迁移与成本、成分覆盖率、routing 误差、residual 记账、neuron 层级分布。来源身份、模型权重和必要数据校验沿用现有机制，不为每条叙述重复生成哈希或单独收据。

---

# 十五、论文结论的组织方式

目前已获得的行为证据是：Endpoint 对有限成分效应具有广泛但非普遍的排序关联；在 M2 的空间区域替换比较中，reference-matched Taylor 表现更好。原 C4a/C4b 的 NOT_SUPPORTED 结论保持。

后续论文围绕以下证据链组织：

1. **表示与数值正确性**：Endpoint 提供注册标量目标的输入坐标表示，attention/residual 的局部导数分解通过独立检查。
2. **Reference 复用与筛选效用**：比较已知 reference 的预测精度、统一排名的平均/最差效用以及实测成本，判断 Endpoint 相对其他 clean-only 方法是否增加信息。
3. **ViT 结构发现**：用 routing omission、residual carry/branch 份额和抵消描述具体层/head 的输入依赖。
4. **Neuron-atlas 分布**：用熵、局部质量、距离和带符号抵消检验抽样神经元的层级与空间规律。

各项待完成实验的结论随结果填写。若统一复用没有带来效用或成本增益，或神经元分布没有呈现预期深度趋势，就报告该项未获支持；结构闭合正确也不能代替这些经验检验。

论文定位为：对 ViT 内部注册标量目标进行统一输入坐标归因、结构化局部 Jacobian 记账与行为评价，并由具体实验界定适用场景。
