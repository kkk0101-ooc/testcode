# NNS ViT unified paper figures R7

[完整画廊](gallery.html) · [Figure 1–14 索引](figure_index.md) · [说人话逐图说明](figure_analysis.md)

重画 Figure 1、2、8、9、10，另九张原样保留。每张均有 PNG/PDF/SVG，共 42 个主图文件。
Figure 1 讲 NNS 扫描 ViT；Figure 2 讲内部响应和输入图的对应关系；Figure 8 换成同一中位效应样本的全 144-head 汇总；Figure 9、10 使用 F3H/F3C 图库的原显示规则。

reference_gallery 收录 F3H 全部 20 张和 F3C 全部 30 页原 PNG。supplement 保留原详细目标/基线计算图的三种格式。历史弱效应、误分类案例仍在 [R6 Figure 8](../nns_vit_unified_paper_figures_r6_20260916/figures_png/figure_08_frozen_case_atlas.png)。

本包仅 CPU 制图，不运行模型；全头显示平均与 class-logit 归因分别标注。中位效应样本身份不变，全部统计图和数值不变。

入口：scripts/render_nns_vit_unified_paper_figures_r7.py；验证：scripts/validate_nns_vit_unified_paper_figures_r7.py。
