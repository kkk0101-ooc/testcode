# NNS-ViT Endpoint 数学对比图

[图册](gallery.html) · [Image #1–#9](figure_analysis.md) · [数值](numeric_data.json) · [验证](validation_report.json)

9 张图，各有 PNG/PDF/SVG；全部为 CPU 解析示意，不是新增模型实测。继承旧图的输入和布局，补齐 Endpoint 与原始 CNN 对比。

仓库根目录运行（新目录必须不存在）：

```bash
CUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python \
  vitnns_rebuild/scripts/render_endpoint_math_comparison.py \
  --output-dir vitnns_rebuild/reports/nns_vit_endpoint_math_comparison_reproduction
```

复核既有输出的哈希和数值：

```bash
CUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python \
  vitnns_rebuild/scripts/render_endpoint_math_comparison.py \
  --verify vitnns_rebuild/reports/nns_vit_endpoint_math_comparison_r1_20260918
```
