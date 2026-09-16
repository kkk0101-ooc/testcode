# 统一论文图 R2：视觉设计修订

日期：2026-09-16。用户要求重绘 Figure 1–3，替换 Figure 4(b)、6(b/c)、11(b)、13(b)、14(a) 的稀疏点区间布局，并整合 Figure 7。

本修订只更新 presentation。输入沿用 R1 的封存数值与冻结案例；R1 和上游终态根目录保留。新版在 `reports/nns_vit_unified_paper_figures_r2_20260916/` 交付完整 Figure 1–14，PNG/PDF/SVG 同步。未指定的 Figure 5、8–10、12 继承 R1 图形。

## 参考阅读与设计依据

参考库中的五篇文章和 61 张配图用于理解视觉表达，不引入它们的实证结果：

| 文章 | 可借鉴的配图 | 本轮采用的原则 |
|---|---|---|
| Understanding neural network through neuron level visualization | fig-02–05、08、12–14 | 用张量、网格、路径和并列图像说明计算；共享坐标轴；减少文字框 |
| AttentionViz: A Global View of Transformer Attention | fig-02、04、07、10 | 直接画输入、向量和操作；局部放大；颜色保持语义一致 |
| How Does Attention Work in Vision Transformers? A Visual Analytics Attempt | fig-02–06、08、13 | 显示 token/head 结构、矩阵与路径；总体和局部视图衔接 |
| Towards Evaluating Explanations of Vision Transformers for Medical Imaging | fig-01–03 | 对齐样本、方法和评价；紧凑矩阵 |
| Vision Transformer attention alignment with human visual perception in aesthetic object evaluation | fig-03、05、11、14 | 顺序示意；逐样本散点叠加箱体；用矩阵比较 head 与样本 |

## 图形变更

| 图 | R2 表达 | 保留的信息 |
|---|---|---|
| 1 | 原图、ViT 结构、patch 内贡献放大与评价的图形总览 | 相同内部目标、细粒度与均匀分配的比较 |
| 2 | 固定 A/u 与可变 Value 路径；积分路径和 signed 网格 | g、u、f、IG 和 c_j 定义；CLS 单列 |
| 3 | patch/cell 网格、恢复/删除图像序列、逐图配对与重采样示意 | 16×16-pixel patch、4×4-pixel cell、21 fractions、5 tie repeats、图像为统计单位 |
| 4 | 频数直方图 + ECDF，区间融入分布面板 | 全部 500 个数值、median、bootstrap CI、500/500 positive |
| 6 | 原尺度 ECDF + 三类算子的逐图散点与 IQR 箱体 | 全部 500 图；uniform/nearest 为同一等价类；IQR 不是 CI |
| 7 | 单张带数值、CI 和有效数的 3×4 矩阵 | 三组 × 四冻结坐标；统一线性色阶；最后坐标 498/417/81 valid |
| 11 | 逐图分布、四项有正负方向的效应条及 CI 区域、deletion curves | 原始 paired mean 和 95% CI，跨零结果保留 |
| 13 | 完整 12-block trajectories + deep-minus-shallow 有向效应条 | B4/B5 的符号、大小和原 95% CI |
| 14 | 按端点列出 estimate、ordinary BCa CI、valid/total 的数值表 | 不同量纲不共用效应轴；post-hoc exploratory 与其余三个面板保留 |

方法图中的小型贡献矩阵和网络结构是示意，不能读作真实 attention 或 attribution；图注明确区分这些示意与冻结原图。R2 不从 CI 猜测或拟合概率密度，也不重新运行 bootstrap、模型或干预。

自动校验检查输入输出哈希、原统计量一致性和格式覆盖；目视检查在实际渲染后记录，渲染器不自动签发人工视觉 PASS。完成本轮不表示执行了 Y5、F3-Final 或 Stage G。
