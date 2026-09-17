# 7. 结论

本文把 NNS 的 learning image 视角扩展到 Vision Transformer，在原始输入坐标下描述指定内部响应和类别输出的变化。方法区分模块自身响应、固定 clean attention 的条件 Value-path 与完整类别贡献切分，并将真实计算从 patch embedding、attention 和第一次 residual 延伸到 MLP、第二次 residual、全部 blocks、final norm、CLS 与 classifier。Completeness 和两处 residual 的逐像素加法检查验证了这些图与目标计算的数值对应。

在预定 block 0/head 9 与 4x4-cell 干预下，NNS 相对同目标 patch-token 基线（patch 内均匀分配）的忠实度差值在 500/500 张图像上为正，中位数为 2.3082。该差异在原生 patch 尺度上降为 0，并在其他三个层头坐标显著缩小。独立类别解释器比较中，NNS 低于 Grad-CAM、高于 rollout，与 full-model IG 和 raw CLS attention 的差异未明确检出。完整切分则支持 attention 和 MLP 绝对贡献份额随深度以相反方向变化。

这些结果把 learning image 定位为连接输入位置、内部响应、干预评价和成分记账的工具。当前证据支持特定目标与尺度下的细粒度用途，并给出可重构的 ViT 内部贡献视图；空间稳定性、跨目标推广和普遍解释器优势仍需独立验证。
