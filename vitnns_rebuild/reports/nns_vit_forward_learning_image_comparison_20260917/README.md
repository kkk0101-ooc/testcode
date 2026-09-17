# ViT forward / NNS learning-image comparison

[画廊](gallery.html) · [逐图说明](figure_analysis.md) · [详细定义](../../docs/ViT正常前向与NNS学习图像逐步对照_20260917.md)

四张新图，每张 PNG/PDF/SVG。统一数学例子，所有显示地图解释正上方单元自身的响应，参考输入为零。展示核心是“正常数值”和“同名 learning image”上下对应。

本例保留 patch 线性映射、完整 softmax attention、residual add；省略 bias/position、LayerNorm、MLP 等以讲解三个算子，不是已训练 ViT 的实测输入输出。
