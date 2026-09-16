# Figure 7 R3：分组柱状图与局部放大

用户反馈 R2 的数值矩阵仍不适合作为图形，要求重新画图。R3 用一张分组柱状图叠加局部放大：横轴为四个冻结 block/head 坐标，颜色为 overall、correct、misclassified。完整主轴和 inset 都从零开始；inset 只重复后三个坐标，纵轴明确标为 0–0.10。

柱高、区间和有效样本数完全沿用 F2-C 封存 JSON。overall 不是第三个独立 cohort；三个分层不合并样本数。最后坐标 498/417/81 的覆盖度与两个 typed failures 保留。没有新增推理、bootstrap 或组间检验。

R3 只新增 Figure 7 的 PNG/PDF/SVG、图注和验证。R1/R2 终态根目录保持原样，其他十三张图由 R3 的索引指向 R2。输出为 reports/nns_vit_unified_paper_figures_r3_20260916。
