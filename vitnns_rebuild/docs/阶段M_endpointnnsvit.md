# NNS-ViT 补充实验完整方案

> **冻结的分层 token-channel 抽样**


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
| **M3. 分层 token-channel 神经元审计** | 加强与原始 NNS 的联系并说明覆盖范围 | 必做 |
| **M4. Transformer 解释方法比较** | 界定不同方法回答的问题 | 推荐 |
| **M5. 关键消融与敏感性分析** | 解释 Endpoint 成功和失败的原因 | 推荐 |
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

如果当前实现主要使用 4×4 像素区域，还应增加 patch-aligned 区域，因为 ViT 的原始空间处理单元是 patch。

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
\sum_i|E_i|+\epsilon
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
\sum_i|E_i|+\epsilon
},
\]

\[
H(E)
=
-\sum_jp_j\log p_j.
\]

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

# 七、M5：关键消融与敏感性分析

不建议无限增加消融。优先完成以下五组。

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

比较内容包括：

- Endpoint map 差异；
- 结构闭合误差；
- M1 成分排序性能；
- M2 区域选择效果；
- early/middle/late 差异；
- routing mass 被忽略的比例。

该消融能够说明为什么必须显式处理 Q/K routing path。

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

建议报告：

- 各层 routing/value absolute mass 比例；
- signed mass 比例；
- routing/value cancellation；
- 哪类成分更能预测有限效应；
- attention saturation 与 routing mass 的关系；
- early/middle/late 层级变化；
- 两个模型之间的差异。

---

## 7.3 Residual identity

对于：

\[
y=x+f(x),
\]

局部 Jacobian 为：

\[
J_{y,x}
=
I+J_{f,x}.
\]

对应 Endpoint 闭合为：

\[
E^{\mathrm{output}}
=
E^{\mathrm{identity}}
+
E^{\mathrm{branch}}.
\]

不建议真正删除模型中的 residual identity，因为这会改变模型函数。

建议执行两类分析：

1. 正确记账下的完整闭合；
2. 故意省略 identity branch 的负向控制。

负向控制应显示闭合显著失败，以证明 identity branch 不能被忽略。

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
\sum_i|E_{k,i}|+\epsilon
}.
\]

检验以下问题：

- cancellation 高时 absolute mass 是否高估有限效应；
- signed score 是否更适合预测效应方向；
- absolute mass 是否更适合预测效应大小；
- routing 和 value 是否具有不同的 cancellation 模式。

---

## 7.5 Target 和 reference 敏感性

建议保留：

### Targets

- ground-truth logit；
- clean-top1 logit。

Probability target 可以作为次要分析，因为 softmax probability 会引入：

- 类别间耦合；
- 概率饱和；
- logit scale 影响。

主结果仍建议使用 logit。

### References

- normalized-zero；
- blur。

可以增加以下探索性 reference：

- dataset mean；
- black image；
- Gaussian noise；
- another-image reference。

但不能在查看结果后选择表现最好的 reference 作为主结果。

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

## 10.4 如果 M2 中 Endpoint 仍不优于 Taylor

可以写：

> 成分级排序能力和空间区域选择能力是两种不同性质。Endpoint 可以对内部成分的重要性提供一定信息，但其高分辨率输入图不自动保证更优的有限区域选择。

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

如果计算资源有限，最低限度建议完成以下五项。

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

---

# 十四、推荐的新增实验冻结文件

为了提高可审计性，建议为新增实验发布一个独立配置文件，例如：

```yaml
supplementary_protocol:
  status: frozen_after_primary_study
  confirmatory_status: supplementary
  models:
    - vit_b_16
    - deit_small_patch16_224

M1:
  predictors:
    - endpoint_absolute_mass
    - activation_norm
    - gradient_norm
    - activation_gradient
    - reference_displacement_norm
    - attribution_patching
    - random
  metrics:
    - spearman
    - capture_at_1
    - capture_at_3
    - regret_at_1
    - ndcg
  strata:
    - within_block
    - across_block
    - early
    - middle
    - late
  references:
    - normalized_zero
    - blur

M2:
  blocks:
    - 2
    - 5
    - 10
  heads: all
  references:
    - normalized_zero
    - blur
  budgets:
    - one_patch
    - 0.05
    - 0.10
    - 0.25
  methods:
    - endpoint_absolute
    - endpoint_signed
    - regional_taylor
    - gradient_only
    - displacement_only
    - random
    - center

M3:
  blocks: all
  tokens_per_block: 16
  cls_tokens_per_block: 1
  patch_tokens_per_block: 15
  channels_per_block: 32
  image_subset:
    method: deterministic_hash
    size: 100
  numerical_validation:
    tokens_per_block: 2
    channels_per_block: 4
    directions_per_target: 5
    precision:
      - float32
      - float64

multiple_testing:
  method: holm

bootstrap:
  unit: image
  confidence_level: 0.95
  replicates: 10000
```

实际字段可以根据代码实现调整，但以下内容必须冻结：

- 图像列表；
- token 列表；
- channel 列表；
- blocks；
- heads；
- references；
- 空间预算；
- predictor；
- 统计指标；
- 多重检验方法；
- 排除规则；
- 数值容差。

---

# 十五、整合后的论文核心结论

完成 M1–M6 后，论文不应以“Endpoint 全面优于其他方法”为目标。

更稳健、也更有科学意义的主结论是：

> NNS-ViT 将原始 NNS 的输入坐标传播思想形式化为适用于 ViT 注册标量目标的局部 Jacobian Endpoint，并提供可闭合的 attention 和 residual 结构记账。分层神经元审计表明，该方法可以稳定应用于跨层、跨空间位置的 token-channel 单元，但当前扫描仍然是冻结抽样，而不是全部神经元穷举。行为实验显示，Endpoint mass 对有限成分效应提供广泛但非普遍的排序信息；其增量价值取决于成分类型、层级和 reference，并且 reference-matched attribution patching 通常更适合预测具体有限替换。进一步地，成分级排序能力不会自动转化为空间区域选择优势。

这套方案将论文的三种能力明确区分开：

1. **数学与实现正确性**
   结构分解能否闭合，Endpoint 是否与局部 Jacobian 一致。

2. **神经元和成分级行为效用**
   Endpoint 是否能够排序有限替换效应，以及是否优于简单基线。

3. **空间级行为效用**
   Endpoint 图是否能够选择真正有效的有限干预区域。

最终建议将论文定位为：

> 一种对 ViT 内部注册标量目标进行统一输入坐标归因、结构化局部 Jacobian 记账和行为审计的 NNS 推广框架，而不是一种声称全面优于现有归因方法的全神经元扫描器。