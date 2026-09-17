# 摘要

视觉 Transformer（ViT）把图像表示为 patch token，并通过多头注意力、MLP 与残差流逐层变换这些表示。常规激活图描述内部响应的大小，却不直接说明指定响应相对参考输入的变化来自哪些输入位置。本文将 Neural Network Scanner 的 learning image 视角扩展到 ViT：通过输入路径积分，把内部标量或类别 logit 的变化分配回原始输入坐标；同时区分模块自身响应图、固定 clean attention 的条件 Value-path 归因和完整类别贡献的内部成分图。真实 ViT-B/16 计算从 patch embedding、block-0 attention 和第一次 residual 延伸至 block-0 MLP、第二次 residual、全部 12 个 blocks、final norm、CLS 与 classifier，并以 completeness 和逐像素 residual additivity 检查数值对应。

在冻结的 500 图像队列中，我们针对预定的 block 0/head 9 内部目标，以 4x4 输入单元进行配对 insertion/deletion 干预。NNS 相对同目标 patch-token 基线（patch 内均匀分配）的图像级忠实度差值在 500/500 张图像上为正，中位数为 2.3082，配对中位数 bootstrap 95% 置信区间为 [2.1111, 2.4452]。在原生 14x14 patch 评价下，中位差值降为 0，说明主效应依赖细粒度评价。独立的 50 图像类别输出比较中，NNS 低于 ViT Grad-CAM、高于 attention rollout；与完整模型 IG 和 raw CLS attention 的差异未达到注册检验的显著性。完整切分分析进一步显示，深层相对浅层的 attention 绝对贡献份额增加 0.1391，而 MLP 份额减少 0.2492。

Learning image 能在统一输入坐标下连接 ViT 内部响应、区域排序和成分分配。其经验表现随目标、空间尺度和内部路径而变化；广义空间稳定性门未通过，独立类别输出比较中的 Grad-CAM fidelity 更高。

**关键词：** Vision Transformer；Neural Network Scanner；learning image；内部归因；积分梯度；忠实度；贡献分解
