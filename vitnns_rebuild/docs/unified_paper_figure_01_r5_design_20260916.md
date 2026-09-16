# Figure 1 R5：把内部解释的用途放在首图

用户要求重新审视论文映射文档，并让 Figure 1 简单明了地强调 NNS 的价值；Figure 2–3 的计算与评价定位得到保留。R5 在新目录 reports/nns_vit_unified_paper_figures_r5_20260916 交付完整 Figure 1–14，只重画 Figure 1，其余 13 张图与 R4 逐字节相同。

## 首图的单一问题

NNS 能帮助分析一个已训练 ViT 的哪种内部现象？

图题：**NNS links internal ViT responses to image evidence**。

三个面板只承担三件事：选择一个内部 head；查看它的真实正负输入贡献；用全 cohort 的图像干预结果说明排序的用途。patch-size 教学、均匀展开公式、IG 积分和 bootstrap 流程由 Figure 2–3 承担。

## 数据和选图规则

- 原图和 NNS map 使用 F3 已冻结的 median-effect slot，selection_order=44，coordinate=(0,9)，sample_id=ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG。直接读取 panel_manifest 绑定的 JPEG 和 metric_ready_map.npy，逐文件核对 SHA-256。该 map 是 56×56 signed fine-cell map，非新计算的 pixel map。
- 此案例延用既有代表性规则；不按颜色、轮廓或新的实验效果挑选。图注区分一个展示案例和 500 图总体结果。
- signed map 采用符号保持的显示变换 sign(m)·log(1+99·|m|/L)/log(100)，L 为该 map 最大绝对值。使用 RdBu_r，最近邻显示；不平滑、裁剪、池化或增加实验归因。显示变换不回流指标。
- 验证面板使用 D10 原 per_sample_delta.csv 和 analysis.json：n=500，positive_count=500，median=2.3081996221714247，95% CI=[2.1110806126093067,2.445235411755993]。500 个方格仅编码这 500 个正差值，不编码效应大小或类间分布；详细分布和轨迹见 Figure 4–5。
- 图中的蓝色正差值表示 NNS 的 F 高于同目标 token 空间基线。它不代表模型分类正确，也不验证每一个局部 signed cell 的因果作用。

## 主张边界与文件策略

首图直接说明内部响应追溯和实测排序用途。完整 class-logit 的 NNS-v2 切分作为 Methods/后续结构分析的独立对象；不把 head-direction Value-path 当作 class logit，不将 IG 输入分辨率称为 NNS 独有创新。技术差异与文献新颖性在映射文档单列。

R4 及上游封存根保持不变。R5 同步继承 Figure 2–14 的图注和说明，提供完整本地链接、源清单、数值校验与视觉审阅。Figure 1 在实际 PNG 审阅后才能封存。
