# NNS ViT unified paper figures R5

完整 Figure 1–14，PNG（400 dpi）/PDF/SVG 共 42 个实体文件。

- [完整画廊](gallery.html)
- [图表索引](figure_index.md)
- [中文图注](captions_zh.md) / [English captions](captions_en.md)
- [逐图解读](figure_analysis.md)

Figure 1 采用“内部 head → 实测正负输入贡献 → 500 图干预证据”呈现研究用途。案例是已冻结的 median-effect 样本，signed map 为原有 56×56 fine-cell 归因；没有新增模型计算。Figure 2–14 连同图注和说明逐字节或逐节继承 R4。

CPU 渲染入口为 scripts/render_nns_vit_unified_paper_figures_r5.py，验证入口为 scripts/validate_nns_vit_unified_paper_figures_r5.py。最终视觉判断记录在 manual_visual_qc.json；终态根不原地重画。
