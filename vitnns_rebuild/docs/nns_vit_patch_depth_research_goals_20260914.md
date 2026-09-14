# NNS 在 ViT 中的 Patch 与深度语义研究目标

日期：2026-09-14
状态：`PROPOSED_NOT_FROZEN`；研究设计，不是实验结果或执行授权。
配套文档：[接下来执行步骤](nns_vit_patch_depth_execution_plan_20260914.md)。

## 1. 总目标

> 在冻结的 ViT-B/16 中，检验 NNS-style 贡献分解能否识别与目标语义对齐、且具有输出干预相关性的原生 patch，并刻画这种贡献分配如何随网络深度变化。

研究围绕两条主线展开：

1. **Patch 主线**：NNS 高贡献区域是否与目标标注对齐？遮挡这些区域，是否比等面积、位置分布匹配的对照遮挡引起更大的指定类别输出下降？
2. **深度主线**：同一种 NNS component 的目标对齐程度是否存在可重复的深浅层差异？这种差异是否与独立训练的 patch 特征 linear probe 所测得的语义可解码性变化相互印证？

最小研究先完成**目标／前景级别**验证。部件级别是需另行冻结的扩展：只有具备可靠部件标注、类别对应关系和足够样本，才检验“眼睛、耳朵、车轮”等具体部件。

成功标准是获得可复核的科学答案，而不是保证正结论。`SUPPORTED`、`NOT_SUPPORTED`、`INCONCLUSIVE` 和 `INVALID` 都是可能结果。

## 2. 与参考论文、Stage Y4 的关系

### 2.1 借鉴参考论文的问题结构，不直接继承 CNN 结论

Dou et al. 的原始 NNS 研究通过 neuron learning images 比较 FCN/CNN 的层内差异和跨层变化。本研究借鉴“内部单元—输入区域—层间组织”的分析结构，但 ViT 中的 token、残差、LayerNorm 和 attention 需要单独定义和验证。

本项目的 path-integrated complete-cut contribution 是 **NNS-style ViT 迁移定义**，不把它无条件等同于原文的 ReLU/权重传播规则；图像更清晰、热区更大或颜色更亮，不直接等于学到了更多语义。

来源：[原文方法](../../相关文章/Understanding%20neural%20network%20through%20neuron%20level%20visualization/en/03-proposed-method.md)、[原文实验](../../相关文章/Understanding%20neural%20network%20through%20neuron%20level%20visualization/en/04-experiments.md)。

### 2.2 新问题不追溯修改 Y4

[Stage Y4-R5 报告](../reports/stage_y_y4_r5_20260914/report.md)的科学状态保持不变：

| Y4 问题 | 已封存结果 | 与本研究的关系 |
|---|---|---|
| patch 内 4×4-cell 排序增益 | `NOT_SUPPORTED` | 本研究首先比较不同原生 16×16 patches，不重做相同的 patch 内细粒度主张 |
| 有效空间支持深浅层差异 | `NOT_SUPPORTED` | 空间支持大小不等于目标／部件语义对齐 |
| head organization | `INCONCLUSIVE` | 本研究不以挑选少数最佳 heads 为主线 |
| patch mixing | `INVALID` | 先处理读出拓扑与分母定义，不能把缺失矩阵填零后继续检验 |
| 两处 residual branch allocation | `SUPPORTED` | 是研究动机；分支份额变化本身不证明语义变化 |
| 整体 perturbation modulation | `INVALID` | 本研究不重新打开 broad robustness 主张 |

Y4 最终空间支持和 patch-mixing 端点绑定的是 all-head aggregate attention branch，见 [Y4-R3 映射修订](stage_y_y4_r3_attention_cut_recovery_20260914.md)。因此不能把最终 B1 不支持简单归因于“不同层完整 cut 必然重构同一张图”；该恒等式约束仍是新设计需要避免的陷阱，但不是对最终 B1 的替代解释。

旧 cohort 可用于明确标注的探索和工程验证；受旧结果启发的新指标，必须在未用于选指标的新样本上作正式验证。新研究不属于 Y4 recovery，也不默认等于 Y5 或论文写作阶段。

## 3. 必须区分的三种证据

| 证据层 | 研究对象 | 能回答的问题 | 不能单独推出的结论 |
|---|---|---|---|
| 表示 | 原生 patch token 向量、固定训练方案的 linear probe | 目标／部件信息是否可由表示解码 | 分类器实际使用了全部可解码信息 |
| 贡献 | 固定 target 的 token／branch 条件 NNS 分解 | 哪些输入区域沿指定 component 对输出分配了贡献 | token 的完整语义内容、唯一因果责任 |
| 干预 | 冻结遮挡操作下的目标 logit 变化 | 被选区域是否具有该操作下的输出干预相关性 | 自然场景因果效应、指定中间分支的独占作用 |

证据链为：**表示可解码 → 输出贡献与语义区域对齐 → 对应区域干预影响输出**。三种测量可以互补，但共享模型、样本和标注，不是三次独立数据集复现。

## 4. 最小研究范围与数学对象

### 4.1 模型、空间尺度和 target

- 模型沿用 [recovery-v2 绑定](../config/recovery_v2_binding.json)中的 ViT-B/16 checkpoint；执行前重新验证文件和 tensor identity。
- 输入固定为 224×224，原生 patch 为 16×16，共 14×14=196 个 patches；本轮不比较不同 patch size。
- 覆盖 blocks `0..11`；预设 shallow=`0..3`、middle=`4..7`、deep=`8..11`。主要深度对比为 deep-minus-shallow，中层轨迹作为补充。
- **建议 primary target 使用标注目标类别对应的 raw logit**，以保证贡献与语义标注指向同一类别；类别映射必须在推理前冻结。
- 同类多实例使用同类目标 mask 的并集；多类别图像按元数据规则预先指定一个目标类别，不根据预测或热图挑类别。
- 预测错误的样本仍保留。clean predicted-class target 如需研究，另设 secondary 分析，不能与标注类别 target 混合。

这里的 primary target 是相对于 Y4 的一项前瞻性改变。因此 Y4 数组不能无条件直接充当新研究的正式输入。

### 4.2 保留 component 维度，而不是比较完整 cut 总图

沿用 complete-cut 的定义：

\[
K_C[u,j]=(x_u-x_{0,u})\int_0^1
\frac{\partial f_c}{\partial h_{C,j}}
\frac{\partial h_{C,j}}{\partial x_u}\,d\alpha.
\]

完整 cut 的所有分量相加，重构同一个 target 的输入归因：

\[
\sum_j K_C[u,j]=A_c[u].
\]

因此，不能把不同层的完整 cut 总图当成不同层独立学到的特征图。完整重构结果还与相同路径、baseline、target 下的 input integrated gradients 对应，单靠总图不能证明 NNS component 分解的独特价值。

**最小设计预先选定第一处 residual addition 的 aggregate attention branch 为 primary component**，聚合全部 12 heads，不挑最佳 head 或最佳层。其他分支分别报告，不能混成 identity/attention/MLP 三项比例。

对 block \(b\)、输入 patch \(p\)，定义该分支的 signed patch score：

\[
a_{b,p}=\sum_{u\in p}\sum_{j\in C_{b,\mathrm{attn}}}K_b[u,j],
\qquad w_{b,p}=\max(a_{b,p},0).
\]

primary 采用**先在原生 patch 内 signed 求和、再取正部**的顺序；负贡献、absolute mass 和 cancellation 单独报告，不事后互换以寻找正结果。

若使用已验证的 matrix/CLS 表示，必须满足：

\[
a_{b,p}=\sum_{q=1}^{196}T_{b,\mathrm{attn}}[p,q]+C_{b,\mathrm{attn}}[p].
\]

这是投影回**输入 source patch**的贡献，包含经 destination patch tokens 和 CLS 的两部分路径；CLS 本身没有可直接对应目标 mask 的二维位置。

### 4.3 最后一层、零质量与语义坐标

只读 CLS 的分类头可以使最后一个 block 的 destination patch 分量对 class logit 为零，即使 patch 表示本身仍非零。Y4 已出现 patch-matrix mass 为零而 CLS mass 非零的情形，见 [Y4-R4 缺失规则](stage_y_y4_r4_zero_mass_na_recovery_20260914.md)。

本研究据此区分：

- 输入语义对齐：比较 source-patch scores 与同坐标的目标 mask；保留 CLS 路径，不能只归一化 patch-to-patch 子矩阵。
- token 表示分析：可以读取最后层 patch 向量，但不能把可解码信息等同于分类头直接使用的信息。
- component 正质量为零：该正贡献对齐指标为 NA，而不是零对齐；不能加任意 epsilon 将其伪装成有效比例。

如某候选端点在拓扑上退化，必须在独立 pilot 阶段停下或前瞻性改版，不得等正式结果出现后删除 block 11。

## 5. RQ-P：Patch 贡献是否与目标语义及干预响应一致

### 5.1 P1：目标语义对齐

将目标 mask 用与图像一致的 resize/crop 几何变换映射到模型输入。令 \(m_p\in[0,1]\) 为 patch 内目标面积比例，定义：

\[
F_b=\frac{\sum_p w_{b,p}m_p}{\sum_p w_{b,p}}.
\]

只看 \(F_b\) 会受目标面积和居中偏差影响。因此 primary 使用预先定义的位置分层内 score permutation 对照，保持 score 多重集和粗粒度位置分布：

\[
L_b=F_b-\frac1R\sum_{r=1}^R F_b^{\pi_r},
\qquad P1_i=\frac1{12}\sum_{b=0}^{11}L_{i,b}.
\]

排列规则、位置分层和 \(R\) 在正式数据访问前冻结。目标面积基线、原始 \(F_b\)、negative/absolute 对齐均作补充；不能以中心物体的一张漂亮热图替代该检验。

待检验命题：**cohort 中的平均 P1 超过预先指定的最小实质效应 \(\delta_{P1}>0\)**。

### 5.2 P2：匹配遮挡下的输出干预相关性

按 \(w_{b,p}\) 排序选择固定预算的 top-k 原生 patches。零分和并列分数使用冻结 hash 顺序；零分被选中时记录正分覆盖率，不能将其称为正贡献证据。

对 selected set \(S\) 定义：

\[
D_i(S)=f_{c_i}(x_i)-f_{c_i}(\operatorname{mask}(x_i,S)).
\]

匹配对照具有相同 patch 数、冻结的位置分层计数，并按预设规则控制遮挡形状／边界复杂度；不能把“匹配位置”误解为遮挡完全相同坐标。

\[
P2_i=\underset{b,k}{\operatorname{mean}}\left[
D_i(S^{\mathrm{NNS}}_{b,k})-
\underset{r}{\operatorname{mean}}D_i(S^{\mathrm{control},r}_{b,k})
\right].
\]

遮挡预算、填充方式和对照生成器均冻结；target 在遮挡后不改变。primary 保留 raw-logit 差值，不临时选择容易放大效应的归一化分母。

待检验命题：**cohort 中的平均 P2 超过 \(\delta_{P2}>0\)**。输入遮挡只能支持这些区域的干预相关性，不能定位“因果作用全部经过 attention branch”。

### 5.3 Patch 主张的边界

P1、P2 都支持时，允许写：

> 在冻结模型、标注类别 target 和遮挡操作下，NNS attention-component 高贡献 patches 与目标区域对齐，并比匹配对照表现出更大的目标 logit 下降。

P1 单独支持只说明语义对齐；P2 单独支持只说明干预相关性。背景也可能是模型真实使用的证据，因此对齐不支持并不自动表示归因算法错误。

与完整 input attribution 的同 target、同 baseline 对照必须保留：若只有总图表现良好，而 component 无额外解释能力，就如实报告；本研究不据此声称 NNS 普遍优于其他方法。

## 6. RQ-D：语义对齐与表示可解码性是否随深度变化

### 6.1 D1：NNS component 的深度对齐差异

仍使用 P1 的同一个 \(L_{i,b}\)，而不是更换指标：

\[
D1_i=\frac14\sum_{b=8}^{11}L_{i,b}
-\frac14\sum_{b=0}^{3}L_{i,b}.
\]

双侧检验实质性差异 \(|\theta_{D1}|>\delta_{D1}\)，不预设“越深越好”。全部 12 层轨迹、中层变化和其他 residual components 为解释性输出，不作为挑选显著 contrast 的候选池。

### 6.2 D2：独立 linear probe 的深度差异

在固定 block-output patch 表示 \(h_{b,p}\) 上，训练线性 ridge probe 预测同一个目标面积比例 \(m_p\)。多类别数据使用类别条件的线性读出 \(g_{b,c}(h)=v_{b,c}^{\mathsf T}h+\beta_{b,c}\)，每个目标类别都需有独立 train/validation/test 图像；低级对照使用同样的类别条件。这样不会要求一个不知道目标类别的 probe 在多目标图像中猜要分割哪个对象。

模型 backbone 冻结；probe 只用独立的 train/validation split 拟合和选超参数，正式 test 不参与这些选择。若逐类别数据不足，先在 S0 缩小并明确类别范围，不能到正式分析时删除表现差的类别。

令 \(E_{i,b}\) 为图像级 patch mean-squared error；令 \(E_i^0\) 为只使用冻结位置／低级外观特征的对照 probe 误差：

\[
Q_{i,b}=E_i^0-E_{i,b},
\qquad D2_i=\frac14\sum_{b=8}^{11}Q_{i,b}
-\frac14\sum_{b=0}^{3}Q_{i,b}.
\]

较大 \(Q\) 表示相对该对照更容易线性恢复前景面积。记录绝对 \(Q\) 与误差，避免“深层比浅层少坏一点”被误写成强语义表示；由于 \(E_i^0\) 在 depth contrast 中相消，D2 单独支持不等于深层已优于低级对照。

D2 同样作双侧实质效应检验。foreground occupancy probe 只测量这一种语义变量的可解码性，不证明完整的物体概念或所有语义能力。

### 6.3 互证和补充分析

- D1 支持：允许声称指定 NNS component 的目标对齐存在深度差异。
- D2 支持：允许声称前景面积的线性可解码性存在深度差异。
- 两者支持且方向一致：允许声称两种测量显示同方向的深度变化；不能称为独立复现或因果中介证明。
- 两者相反或一项不支持：报告表示与输出贡献的分离，不事后换 probe、branch 或层组使其一致。

补充分析可包括同目标／不同目标 patch similarity、相同输入位置的跨层表示变化、位置与颜色匹配后的邻域关系。每张图先汇总；同一图像的 patches 不能伪装成独立样本。

若要进一步写“浅层纹理—中层部件—深层物体”，必须另加纹理／部件／物体三个任务各自的外部标签、对照与预注册跨层检验；当前两条主线不自动支持这个完整叙事。

## 7. 主端点与决策

| ID | 图像级端点 | 主检验方向 | 主要证据 |
|---|---|---|---|
| P1 | 12 层平均的位置对照校正目标对齐 | 正向，超过 \(\delta_{P1}\) | contribution—annotation 对齐 |
| P2 | 固定层和遮挡预算平均的匹配干预差 | 正向，超过 \(\delta_{P2}\) | input intervention 相关性 |
| D1 | NNS 对齐的 deep-minus-shallow | 双侧，超过 \(\pm\delta_{D1}\) | component 深度组织 |
| D2 | probe 可解码性的 deep-minus-shallow | 双侧，超过 \(\pm\delta_{D2}\) | representation 深度组织 |

建议四个端点共用一个 Holm family，family-wise alpha=0.05；统计单位为 source image，cohort estimand 为图像级端点的均值。配对 bootstrap、阈值感知检验、有效样本下限、效应阈值和类别加权规则必须在正式验证前冻结。

- `SUPPORTED`：有效性、CI 实质效应门槛和 multiplicity-adjusted 检验全部通过。
- `NOT_SUPPORTED`：有效且 CI 排除了预注册的实质效应；不是因为 p≥0.05 就自动判不支持。
- `INCONCLUSIVE`：有效但精度不足，或不满足前两种决策。
- `INVALID`：定义／重构／缺失规则／有效样本门槛未通过。

Patch 联合支持要求 P1、P2 都 `SUPPORTED`；否则报告两项状态，不强行压成一个总 PASS。D1、D2 独立决策，只有两者都支持且方向相同才增加“同方向互证”标签。部件扩展未注册时记为 `NOT_SCHEDULED`，不伪装成失败或有效结果。

## 8. 本轮不包含的主张

| 用户关心的现象 | 本轮定位 |
|---|---|
| 前景与背景的可解码性、贡献对齐 | 最小主研究 |
| 高贡献 patch 是否更有干预影响 | 最小主研究 |
| 同语义 patch 聚类／相似 | 有位置和外观对照的补充分析 |
| 中层物体部件形成 | 部件标注扩展，需单独冻结 |
| 局部到远距离交互 | 另需路径对象和干预定义；attention 可连接不等于存在已验证交互 |
| 深层 token 同质化／过度平滑 | 相似度只能支持同质化描述；过度平滑还需局部任务能力受损证据 |
| patch size 的精度与计算权衡 | 本轮不比较；需要受控模型／训练条件 |
| 普遍空间鲁棒性、跨模型泛化、方法全面胜出 | 不在本轮范围 |

## 9. 预期交付

1. Patch：目标对齐与匹配遮挡效应表，附全部有效／失败／NA 分母。
2. Depth：全 12 层 NNS 对齐和 probe 曲线，附预注册的深浅层对比。
3. 对照：位置／外观控制、完整 input attribution、负贡献与 zero-mass 记录。
4. 展示：按冻结规则抽取的跨层 atlas；显示协议与数值分析分开，保留反例。
5. 结论：逐端点决策和可用措辞；流程完整性与科学支持程度分别报告。

**推荐下一步仅做数据与方法可行性盘点，再做隔离 pilot；不是直接重跑 Y4，也不是立即承诺任何新的 SUPPORTED 结论。**
