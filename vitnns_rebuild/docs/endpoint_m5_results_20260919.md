# Endpoint M5 方法定位与最小 reference 复用结果

M5 以 CPU-only、reuse-only 方式完成。分析只读取 M1/M2 的封存数值记录、M4/Review-R2 收据、EP2-R1 校准记录和 EP0 文献 registry；未读取 JPEG，未加载模型或 checkpoint，未导入 Torch/timm，未使用 GPU，也未增加检验 family。

## 1. 方法语义与 reference 属性

对模型实际接收的 normalized input `x` 和注册标量目标 `F`，Endpoint 为

\[
E(x)=x\odot\nabla_xF(x).
\]

它由输入坐标、模型、目标和预处理唯一确定，不需要额外选择一个 intervention reference。若原始 RGB 为 `r`，且

\[
x=(r-\mu)/\sigma,
\]

则

\[
x_i\frac{\partial F}{\partial x_i}
=(r_i-\mu_i)\frac{\partial F}{\partial r_i}.
\]

因此准确表述是：Endpoint 不依赖外部选择的干预 reference，但仍绑定模型的输入坐标系与 normalization origin。对 logit 目标，它就是 normalized-input 坐标中的 Input × Gradient，不是一个可与 Input × Gradient 并列的新输出归因算法。

## 2. 两种既有 reference 的最小分析

复用 M1 的两模型、共享 400 图、12 个 blocks 和全部 attention heads。每图先在每层计算 head 排名 Spearman，再对 12 层取均值；表中为 400 图中位数与固定 10,000 次 image-bootstrap 95% CI。该汇总为描述性复用，不增加 M1 的统计检验。

| Model | Taylor 两 reference 排名一致性 | 有限效应两 reference 排名一致性 | Endpoint rho：zero / blur | Taylor rho：zero / blur | Taylor−Endpoint rho：zero / blur |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 | 0.3782 [0.3666, 0.3896] | 0.3759 [0.3648, 0.3887] | 0.2815 [0.2716, 0.2966] / 0.2727 [0.2587, 0.2864] | 0.6742 [0.6661, 0.6841] / 0.6856 [0.6760, 0.6943] | 0.3878 [0.3727, 0.3998] / 0.4082 [0.3948, 0.4237] |
| DeiT-S/16 | 0.3000 [0.2857, 0.3143] | 0.3071 [0.2905, 0.3214] | 0.1952 [0.1762, 0.2143] / 0.1476 [0.1333, 0.1619] | 0.6095 [0.6000, 0.6286] / 0.6238 [0.6095, 0.6381] | 0.4048 [0.3810, 0.4190] / 0.4738 [0.4500, 0.4905] |

M1 的 163,200 个 component-reference 配对中，Endpoint absolute mass `m` 和 signed sum `a` 在两种 reference 间的最大绝对差均为 0。相同记录中的 `abs(p_ref)` 与实际 absolute finite effect 在 163,200/163,200 项上都发生变化。9,600 个 image-block head 排名中，Taylor 在两 reference 下完全相同的只有 46 个，实际有限效应排名完全相同的只有 64 个。

两种 reference 下的 Taylor 与有限效应排名一致性中位数只有 0.3782/0.3000，实际有限效应自身的跨 reference 排名一致性为 0.3759/0.3071。这说明 reference 不只是改变分数尺度，也会改变“哪些 head 更重要”的顺序。Endpoint 排名固定，但它与 reference-specific 有限效应的关联随 outcome 改变；Taylor 将实际 activation displacement 纳入 predictor，因此在四个 model-reference 组合中均有更高的行为关联。这个结果界定信息条件与任务匹配，不建立 Endpoint 或 Taylor 的普遍优越性。

## 3. M2 空间结果的复用核对

从 M2 封存的 image-level arrays 重算 patch geometry、blocks 2/5/10、full 与 isolated outcome 的 24 个 model/reference/block/outcome 组。所有组中 regional Taylor 的中位 AUC 均高于 Endpoint absolute。原 M2 Holm family 保持不变：完整输入效应的 Taylor 优势为 11/12，隔离 head 效应为 12/12；M5 不重复签发检验。

| Model | Reference | Outcome | Endpoint AUC 范围 | Taylor AUC 范围 | 配对 Endpoint−Taylor 中位数范围 |
|---|---|---|---:|---:|---:|
| ViT-B/16 | normalized-zero | full | 0.1451–0.1640 | 0.1977–0.2377 | −0.0552–−0.0289 |
| ViT-B/16 | blur-rgb | full | 0.3624–0.3783 | 0.4306–0.4850 | −0.1042–−0.0690 |
| ViT-B/16 | normalized-zero | isolated | 0.0241–0.0487 | 0.0470–0.0630 | −0.0229–−0.0140 |
| ViT-B/16 | blur-rgb | isolated | 0.0296–0.0367 | 0.0487–0.0594 | −0.0209–−0.0135 |
| DeiT-S/16 | normalized-zero | full | 0.4608–0.4679 | 0.4998–0.5040 | −0.0424–−0.0374 |
| DeiT-S/16 | blur-rgb | full | 0.2491–0.2651 | 0.3385–0.3782 | −0.0984–−0.0551 |
| DeiT-S/16 | normalized-zero | isolated | 0.0399–0.0770 | 0.0738–0.1156 | −0.0399–−0.0323 |
| DeiT-S/16 | blur-rgb | isolated | 0.0340–0.0568 | 0.0622–0.0977 | −0.0380–−0.0272 |

## 4. 方法适用性矩阵

| 方法 | 主要问题 | 内部标量目标 | 输入空间图 | 外部干预 reference | 局部 Jacobian 结构闭合 | 具体 finite replacement 预测 |
|---|---|---:|---:|---:|---:|---:|
| Endpoint | 当前输入处注册目标的局部依赖 | 是 | 是 | 否 | 是 | 有限、需实证 |
| Activation-space Taylor | 指定 activation replacement 的一阶预测 | 是 | 可构造区域分数 | 是 | 非主要合同 | 强匹配 |
| Integrated Gradients | baseline 到输入的路径归因 | 是 | 是 | 是 | 可扩展 | baseline-difference attribution |
| Attention rollout | 跨层 attention flow | 通常否 | 是 | 否 | 否 | 非主要合同 |
| Grad-CAM | 类别相关的粗粒度 activation 定位 | 部分 | 是 | 否 | 否 | 非主要合同 |
| Chefer/AttnLRP | Transformer relevance propagation | 部分 | 是 | 通常否 | 依实现 | 非统一 |
| Attention weights | token-to-token attention allocation | attention 对象 | patch/token | 否 | 否 | 不能仅由权重给出 |

这个矩阵比较的是 estimand、信息条件和验证合同。Attention rollout、Grad-CAM、Chefer/AttnLRP 与 raw attention 的功能不同于注册内部目标的局部 Jacobian 分解；当前论文没有输出定位优越性主张，因此 M5 不建立跨方法定位排行榜。

## 5. Path IG 与成本边界

EP2-R1 的预注册门要求每模型每 target 至少 61/64 图在最大 512 积分点内同时通过 map 与 completeness 校准。DeiT-S 的 ground-truth/clean-top1 为 61/64 和 62/64；ViT-B 只有 41/64 和 40/64。因此跨模型 Endpoint–Path fidelity、stability 和 cost-quality 结论保持未发布，不追加积分点或调阈值。

| Package | 观测成本与规模 | 可解释范围 |
|---|---|---|
| M1 | 0.079 attempt GPU-hours；43.26 s acquisition；64.49 s CPU analysis；326,400 component-reference conditions | 多 predictor 共享 forward 与已有 E4 结果，只是 package 成本 |
| M2 | 5.982 GPU-hours；2,799.49 s wall；14,803,180 candidates；2,332,800 curves | 八方法、两 outcome 与候选干预的联合 package 成本 |
| EP2 Path IG calibration | ≤0.286 GPU-hours；268.24 s；最高 512 点 | 校准成本；ViT-B 未通过发布门 |

Endpoint 通常需要观测输入 forward 加一次目标 backward/VJP；activation-space Taylor 还需要指定 reference 的 activation displacement；IG 需要沿路径的多个 forward/backward；有限 replacement 为每个候选执行反事实 forward。现有 package 混合共享计算、控制项和多种方法，不能据此分离出公平的逐方法 runtime；M5 因而只报告操作与 package 成本，不签发运行时优势。

## 6. M5 结论与边界

M5 支持的定位是：Endpoint 是绑定输入坐标原点、但不依赖额外 intervention reference 的局部 Jacobian 对象；它适合统一扫描内部标量目标和做结构闭合分析。指定 reference 的有限替换是另一种问题，实际效应排序随 reference 明显变化，而匹配 displacement 的 Taylor 在 M1/M2 的冻结任务上更贴合有限效应。输出级 Endpoint 与 Input × Gradient 是同一公式。Path IG 的跨模型比较仍受 ViT-B 数值校准门限制。

M5 没有执行六类 reference 扩展、Path IG 调参、大规模输出定位 benchmark、新区域搜索或 M6 案例选择。
