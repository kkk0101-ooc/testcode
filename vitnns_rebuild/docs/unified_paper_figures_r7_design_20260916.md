# R7：把 NNS 扫描 ViT 的用途画出来

## 图组安排

R7 保留 Figure 1–14，重画 1、2、8、9、10，其余九张逐字节继承 R6。
这是已有结果的 CPU 制图，不运行模型，不修改实验结果或已封存图包。

| 图 | 读者首先看到什么 | 素材 |
|---|---|---|
| 1 | 一张图进入 ViT，NNS 把不同层、不同 head 的响应对应回输入位置 | order 44 原图，F3H 的 B0/B5/B11、H0/H5/H11 实测图 |
| 2 | 上面是内部计算，下面是与选定响应对应的输入贡献图 | 同一输入、B0/B5/B11 的 H9 正负图，配套目标与输入路径公式 |
| 8 | 中位效应样本的全 144-head 空间汇总 | 用户指定的 F3H Image 19；从原始地图复算同一显示数组 |
| 9 | 四个深度段的空间形态 | F3H Image 18 的全部 36-head/段平均，不挑某个最漂亮的 head |
| 10 | 同一批图像上，不同解释方法画出了什么 | F3C primary page 1 的全部五张样本、全部七个方法/显示列 |

另附 F3H 的 20 张原图，以及 F3C 的 primary、signed、unclipped 共 30 页。
这些 PNG 原样复制，不裁切、不调色。旧 Figure 2 的目标/基线计算图作为补充保留。
原论文 fig-01、fig-02 只作构图参考，不把其 CNN 实验图当成本论文结果。

## 概念与实现

继承 Dou 等提出的 NNS 研究视角：给网络内部单元配一幅保留输入位置的解释图。
本项目的 ViT Value-path 实现通过固定 clean attention 的目标和输入路径积分获得归因；
它不是把 CNN 的 ReLU 门控权重传播公式原样搬到 Transformer。

图 1 的 patch 网格标明输入位置；地图展示各层各 head 的 CLS head-direction 目标。
现有 F3H 不包含每个 patch-query 单独的归因图，因此不把输入区域裁切或 patch 网格称为
“每个 patch token 的实测 NNS”。Figure 2 清楚区分输入 patch/token、所选内部响应和输入贡献图。
若后续需要完整的 layer × query-token × head 归因图库，须单独定义目标并计算。

## 中位效应样本与颜色

Figure 8 保留原先的 order 44、ILSVRC2012_val_00002287.JPEG。
“中位效应”来自 F3 按主实验 B0/H9 的 ΔFID 所冻结的样本选择，不表示全头平均的效应也位于中位。
这次变化是展示单元从一个 head 改成全部 144 个 head 的等权显示平均，不更换样本。
弱效应和误分类案例仍能通过 R6 历史图包查阅。

F3H 每个坐标的 NNS 和 token baseline 共用 max-absolute 色阶；
分别取正部/负部，使用 log1p(999·m/L)/log(1000)，再按原实现线性放大。
对显示数组作等权平均，仅表示空间形态汇总，不是类别归因或原始归因的求和。
渲染器逐项核对全部 68 个已有 consensus 数组摘要，确保沿用真实源图的显示规则。

F3C 沿用原 visual 包的逐面板 P99、正负拆分、4×4 max-pool/bilinear 显示。
热图亮度用于读空间形态，正式方法优劣仍由 Figure 11 的 50 图统计说明。

## 文字检查

使用 academic-defensive-writing-auditor，先讲图中对象和对应关系，再用一句话说明必要范围。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Figure 1 解读 | 不是只好看，它的排序在干预实验里有用 | D10 | Minor | No | 先为价值辩护 | 首图任务不清 | 改为直接说明 | 上面选层和 head，右侧看这个响应对应输入中的哪些位置。 |
| Figure 8 解读 | 这些例子不是人工挑最好看的 | D2 | Minor | Mixed | 围绕选图辩解 | 不清楚实际怎么选 | 写出来源 | 使用先前按主实验效应冻结的 order 44，汇总全部 144 个 head。 |
| Figure 9 解读 | 别用单个样本去推出所有 head 的规律 | D12 | Minor | Yes | 否定式提醒 | 边界盖过读图内容 | 正面限定对象 | 这张图展示同一图像在四个深度段上的空间形态。 |

### Paper-level defensive-writing score

本次改写的图注与读图段落：2/10；不对整篇尚未完成的论文评分。

### Highest-priority removals

1. 删除首图中的 500 方格及统计论证，主结果留在 Figure 4–7。
2. 删除把输入分辨率当作首图贡献的说明。
3. 将反复出现的“不能证明”改为一次明确的目标/显示定义。

### Necessary caveats to preserve

head-direction 不是 class logit；全头显示平均不是完整网络归因；
“中位效应”仍绑定原主实验坐标；F3C 的显示变换不进入 fidelity。

### Repeated phrase audit

只检查本次五张图的替换文字，不统计继承的九张图或整篇论文。
替换段落中“我们不声称”“为避免误解”“这不证明”均为 0 次。
