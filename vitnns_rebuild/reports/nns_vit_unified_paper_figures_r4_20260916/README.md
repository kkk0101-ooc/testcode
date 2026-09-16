# NNS ViT unified paper figures R4

完整 Figure 1–14，全部图在当前目录提供 PNG（400 dpi）、PDF、SVG 实体文件，共 42 个导出。

- [浏览画廊](gallery.html)
- [完整图表索引](figure_index.md)
- [中文图注](captions_zh.md) / [English captions](captions_en.md)
- [逐图分析](figure_analysis.md)

Figure 1–3 重新绘制，区分模型表征与归因读出，明确“不同 patch 可以不同，各 patch 内均匀展开”，并分别画出 NNS 和 token 基线的计算路径。数字网格和 mask 都是方法示意。

Figure 7 逐字节继承 R3 的分组柱状图与局部放大，其余 Figure 4–14 逐字节继承 R2。中英文图注、逐图说明均按此继承关系组装；既有实证数值保持原值。

渲染器：scripts/render_nns_vit_unified_paper_figures_r4.py。验证器：scripts/validate_nns_vit_unified_paper_figures_r4.py。仅使用 CPU；封存后的图包不可原地重画。
