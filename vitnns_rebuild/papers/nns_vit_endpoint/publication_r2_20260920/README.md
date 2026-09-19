# NNS-ViT：论文与完整图件

这个目录包含可独立阅读和复制的论文版本：22 张正文图、7 张正文表、双语补充材料，以及 92 页历史补充图。

- [中文论文](zh/manuscript.md)
- [English manuscript](en/manuscript.md)
- [图片总览](gallery.html)
- [中文补充材料](zh/supplement.md) / [English supplement](en/supplement.md)
- [图件清单](figure_manifest.json)
- [逐图阅读说明](figure_analysis.md)
- [逐项修改验收](qc/completion_audit.md)

figures/png、figures/pdf、figures/svg 保存正文图的三种格式。supplement 保存所有补充图的本地副本；正文和补充材料的图片链接均位于这个目录内。assets 保存本轮真实展示计算的数组，evidence 保存来源、显示规则和计算记录。作者及单位信息仍由作者填写。

本轮参考 Dou et al. (2023) 的概念解释顺序和蓝底图像组织方式。Figures 1–7 讲解学习图像与 ViT 分解；8–11 展示内部目标、逐层学习图像及实际前向特征；12–18 结合总体空间轨迹和路径示例；19–20 比较实际计算的 Endpoint、Grad-CAM、attention rollout 与 raw attention；21–22 说明有限干预与 signed/positive 显示的关系。

对比使用相同输入与冻结模型；Endpoint/Grad-CAM 使用同一 GT logit，attention 方法为 class-agnostic。三个展示样本经过可读性选择，因此图片说明表示对象和空间形态，不用于宣称方法普遍优越。原群体统计与 C4a/C4b 判定保留在补充材料。

复现入口位于上级 scripts/：compute_publication_r2.py、render_publication_r2.py、write_publication_r2.py、verify_publication_r2_methods.py、validate_publication_r2.py。实际阅读和图件复制只需本目录。
