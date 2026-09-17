# 7. 结论与未来工作

本文提出 NNS-ViT，通过 learning image 观察视觉 Transformer 的内部工作过程。给定训练完成的 ViT 和一幅输入图像，该方法无需修改网络结构，即可把指定内部响应或类别输出表示为一张保持原始输入位置的图像。不同 token、channel、attention head 和网络分支的 learning image 可以在统一坐标下展示和组合，使模型内部原本分散在不同表示空间中的计算过程变得可比较。

我们从 patch embedding 开始扫描 ViT，依次研究 attention、MLP、两次 residual addition、跨层贡献以及最终 classifier。真实模型计算表明，learning image 能够对应模块自身响应，并通过完整性和 residual 加法关系进行数值核对。不同通道、head、layer 和类别输出形成不同的空间贡献结构；正确类别、竞争类别和类别间 margin 也对应不同的 learning image。这些结果展示了 ViT 如何从同一输入中组织不同内部响应，并将它们逐步组合成最终输出。

我们进一步从多个角度评价 learning image 的解释能力。输入干预表明，细粒度输入贡献能够提供原生 patch-token 表示未直接展开的空间排序信息；尺度和层头分析说明这种信息随观察对象和评价尺度变化。与其他 ViT 解释方法的比较显示，NNS-ViT 提供的是互补的内部响应视角。完整切分则把最终类别归因分配到 attention head、MLP 和 residual 等内部成分，在保持总归因可重构的同时，揭示不同计算分支随网络深度发生的贡献重组。

NNS-ViT 为视觉 Transformer 建立了一条从输入位置到内部响应、再到最终类别输出的可视化路径。它既可以用于观察单个组件学到了什么，也可以组合多个组件来分析模型如何工作，并通过干预和成分重构检验这些观察。未来工作将把该方法扩展到更多 Vision Transformer 架构、训练目标和视觉任务，研究跨样本稳定的 head 与通道组织，改进输入变化下的空间稳定性，并探索 learning image 在模型诊断、结构比较和模型优化中的应用。
