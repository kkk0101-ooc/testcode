# 1. 引言

人工神经网络已经成为图像识别的重要工具，但模型如何从输入图像中形成内部表征和最终判断，仍然难以直接观察。可解释性研究通常从输入、内部单元或输出三个层次描述模型：输入归因指出哪些区域与预测相关，特征可视化呈现内部单元偏好的模式，attention 可视化展示信息在 token 之间的传递。不同方法产生的结果具有不同的坐标、目标和含义，因此很难把它们连成一条完整的模型工作过程。

视觉 Transformer（ViT）进一步放大了这一问题。ViT 先把图像划分为 patch，再将 patch 编码为 token；每个 transformer block 使用多头注意力交换 token 信息，使用 MLP 变换通道表示，并通过两次 residual addition 保留和叠加信息。经过多层计算后，CLS token 被送入分类器形成类别输出（Dosovitskiy et al., 2021）。在这一过程中，输入图像、patch token、attention head、MLP branch、residual stream 和类别 logit 分属不同表示空间。普通 feature map 或 attention map 可以显示某一层的响应，却难以持续回答“这个响应从原图的哪里学来”以及“不同内部响应怎样组合成最终输出”。

Dou et al. (2023) 提出的 Neural Network Scanner（NNS）从神经元层级观察网络。其核心对象是 learning image：给定一个输入和网络单元，为该单元构造一张保持输入位置的图像，用来表示该单元从输入中学到的特征；多个单元的 learning image 可以进一步组合，以分析层和网络组件的工作过程。原方法在 FCN 和 CNN 上展示了这种统一可视化方式。ViT 的计算结构不同于 FCN 和 CNN：attention 权重随输入变化，多个 head 共同产生输出，MLP 和 residual 在每个 block 内交替作用，最终分类依赖全部 blocks 和 CLS token。因此，将 learning image 用于 ViT，需要重新定义内部观察对象，并保持 attention、branch 和 classifier 之间的计算关系。

本文提出 NNS-ViT，用统一的输入空间图像扫描视觉 Transformer。对于任意选定的内部标量响应，NNS-ViT 沿参考输入到观测输入的路径累计输入贡献，并把这些贡献放回原始位置。这样，patch embedding 中的一个通道、attention 输出中的一个方向、MLP 分支、residual 输出以及最终类别 logit 都可以对应一张 learning image。不同内部成分的图可以在满足完整切分的条件下组合，使同一类别输出能够按 attention head、MLP 和 residual 路径展开。Figure 1 展示 NNS-ViT 的整体扫描视角，Figure 2 展示真实内部响应与 learning image 的对应，Figures 15–27 进一步说明其初始化、组合和 ViT 算子计算。

基于这一表示，我们研究 ViT 工作过程中的四类问题。第一，patch embedding、attention、MLP 和 residual 分别从输入图像的哪些位置形成响应，这些响应如何沿 block 继续组合并到达 classifier。第二，learning image 中的空间结构是否与真实模型响应相关，即按照图中的贡献排序改变输入时，目标响应是否随之变化。第三，不同 layer、head、spatial scale 和类别目标产生的学习图像有何差异，以及一张 attention map 能否代表完整的模型解释。第四，完整类别贡献如何分配给内部 head 和 branch，这种分配是否随网络深度改变。

实验在训练完成的 ViT-B/16 上展开。我们首先用解析示例和真实模型输出验证 learning image 的数值完整性、attention 路径以及 residual 加法关系，并把扫描过程从 patch embedding 延伸到最终类别输出。随后，通过大样本输入干预检验区域排序，结果显示 learning image 能揭示 patch 内部对特定响应有用的空间差异；这种作用取决于评价尺度和内部位置。跨层与全 head 图谱显示 ViT 内部形成多样的空间响应，而独立解释器比较表明不同解释方法各有适用范围。最后，完整切分把最终类别归因分配到 attention 与 MLP 等内部成分，显示这些分支的贡献份额随深度重新组织。

本文的主要贡献如下：

1. 提出面向视觉 Transformer 的 NNS-ViT，在不改变模型结构和训练过程的条件下，为内部响应和类别输出生成保持输入位置的 learning image。
2. 建立从 patch embedding、multi-head attention、MLP 和两次 residual addition 到最终 classifier 的统一扫描方法，使不同层、head、branch 和输出单元能够在同一输入坐标下观察和组合。
3. 将 learning image 从视觉展示连接到可检验的模型行为，通过输入干预、空间尺度、层头变化和解释器比较评价其反映内部响应的能力。
4. 将完整类别归因组织为可重构的内部成分图，并据此研究 attention 与 MLP 等计算分支在网络深度上的贡献分配。

本文其余部分安排如下。第 2 节回顾神经元可视化、输入归因、ViT attention 分析和解释评价。第 3 节介绍 NNS-ViT 的 learning image、ViT 算子和完整切分方法。第 4 节说明模型、数据和评价协议。第 5 节展示 ViT 的扫描结果并分析空间结构、干预效果、解释器差异和深度贡献。第 6 节讨论研究局限，第 7 节总结全文并给出后续研究方向。
