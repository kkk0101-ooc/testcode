# Endpoint NNS-ViT 中文汇报 PPT

本目录包含两套依据 `../zh/manuscript.md` 生成的中文汇报：

- `Endpoint_NNS_ViT_20页重点汇报.pptx`：20 页，适合组会或短时汇报；一页一个重点，重图像和直观结论。
- `Endpoint_NNS_ViT_40页详细汇报.pptx`：40 页，适合论文答辩或完整技术报告；补充数学对象、结构合同、统计口径和数据分析。

两套文件都使用 16:9 页面。标题、正文、数据卡和表格可编辑；论文正式图件以图片方式嵌入。每页都包含中文讲者备注。

## 内容口径

- Endpoint 明确写为 `x ⊙ ∇ₓs(x)`，即同目标的 Gradient × Input。
- 方法重点是 ViT 的双索引结构分解、合法切面与结构审计，不把基础公式包装为新算法。
- 正确性、空间规律和有限替换效用分开报告。
- 有限替换部分明确区分当前状态 Endpoint 与 reference-aware Taylor。
- 展示图仅用于空间阅读，数值结论仍来自论文绑定的原始数组和表格。

## 重新生成

在 `testcode` 根目录执行：

```bash
.conda/envs/vitnns/bin/python \
  vitnns_rebuild/papers/nns_vit_endpoint/publication_r2_20260920/presentations/generate_endpoint_presentations.py
```

脚本依赖 `python-pptx` 与 Pillow。生成后会同时更新：

- `slide_outline.md`：两套 PPT 的页级提纲与用图映射。
- `qc/contact_sheet_20.png`、`qc/contact_sheet_40.png`：版式总览。
- `qc/pptx_validation.json`：页数、压缩包完整性、讲者备注和页面边界检查。

Windows PowerPoint 建议使用微软雅黑显示中文；若系统没有该字体，PowerPoint 会自动替换。
