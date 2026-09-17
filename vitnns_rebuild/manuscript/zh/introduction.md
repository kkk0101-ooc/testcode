# 1. 引言

视觉 Transformer（ViT）将图像切分为 patch，将 patch 投影为 token，并通过多头自注意力和逐 token 的 MLP 反复更新 token 序列（Dosovitskiy et al., 2021）。这种结构为图像分类提供了强大的表征能力，也带来了多层次的内部观察对象：一个响应可以属于某个 token、channel、head、branch、residual stream 或最终类别输出。仅查看激活值或 attention 权重，往往难以回答另一个更直接的问题：指定内部响应相对参考输入的变化，分别来自原图的哪些位置？

神经元和特征可视化为理解深度网络提供了多种视角。Activation maximization 寻找能够激活单元的输入，saliency 与归因方法解释给定输入，attention 可视化则呈现 token 之间的路由结构。Dou et al. (2023) 提出的 Neural Network Scanner（NNS）为 FCN 和 CNN 中的神经元构造 learning image，把单元响应表示为保留输入位置的图像，并允许将多个神经元图组合起来查看模块。将这一视角用于 ViT 不能只把 token 网格放大到图像尺寸：attention 的权重依赖输入，residual 把旁路与分支相加，类别输出又经过多层 token mixing、final normalization 和 CLS classifier。

本文研究三个相互关联的问题。第一，如何为 ViT 内部响应和类别输出定义输入坐标下的 learning image，并让其数值和真实前向计算对应？第二，这些图给出的区域排序是否能通过输入干预进行评价？第三，完整类别归因能否按 head、attention、MLP 和 residual 成分记账，从而描述贡献随深度的分配？Figure 1 概括了从内部目标、learning image 到干预和结构分析的研究路径；Figure 2 展示一个真实 attention 通道响应及其输入位置正贡献；Figure 3 给出两种排序在共同空间单元上的干预比较。

我们沿直线路径使用积分梯度构造 learning image，并明确区分三种对象。模块自身响应图解释 patch embedding、attention、MLP 或 residual 输出本身；固定 clean attention 的条件 Value-path 图解释一个预先指定的内部方向，同时切断 Q/K/softmax 随积分输入变化的路径；完整切分类别成分图保留最终类别导数，并把同一个 full-model IG 总图分配给内部成分。该区分使图像、目标和允许的结论保持一致。

本文的主要实证来自一个冻结 ViT-B/16 和独立的 500 图像队列。在 block 0/head 9 的固定内部目标上，NNS 的 4x4-cell 区域排序相对同目标 patch-token 基线（patch 内均匀分配）在全部 500 张图像上取得正的忠实度差值，中位数为 2.3082。原生 patch 尺度的中位差值为 0，其他三个预定层头坐标的效应也明显较小。在独立的 50 图像类别解释器比较中，NNS 的 fidelity 低于 Grad-CAM、高于 rollout，与 full-model IG 和 raw CLS attention 的差异未明确检出。完整切分的 100 图分析则支持 attention 与 MLP 分支份额随深度以不同方向变化。

本文的贡献包括：

1. 为 ViT 内部标量和类别标量建立输入坐标 learning image，并给出 patch embedding、attention、MLP、两次 residual 与 classifier 路径的对应关系。
2. 在真实冻结模型上验证 completeness 和 residual additivity，将模块展示从 patch embedding 延伸到完整 class logit，并比较正确类、最强竞争类、margin 与低分输出。
3. 用 500 图细粒度干预评价指定内部目标的区域排序，并通过尺度、层头、预测分层和独立解释器比较界定经验范围。
4. 以完整切分将类别归因组织为内部成分，报告可重构性和跨深度分支份额，同时保留稳定性失败、无效端点和事后探索结果。

后文先回顾 NNS、输入归因和 ViT 解释研究，再定义方法与实验协议。结果部分从真实模块计算开始，随后报告干预、敏感性、方法比较、完整切分和深度分析，最后讨论适用范围与局限。
