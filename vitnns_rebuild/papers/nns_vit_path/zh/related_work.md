# 2. 相关工作

## 2.1 神经元与特征可视化

神经网络内部可视化大致分为两类。第一类寻找能够最大化某个单元响应的输入，用于描述模型偏好的刺激；第二类从一个给定输入出发，解释该输入如何形成内部响应或输出。Feature visualization、network dissection、反卷积和类激活图分别从优化输入、语义概念、局部响应和类别判别区域刻画模型。它们提供了有用的观察对象，但不同层的 token、channel 或 feature map 通常不共享统一的输入坐标和守恒关系。

Dou et al. (2023) 提出 Neural Network Scanner（NNS），为 FCN 和 CNN 中的神经元构造 learning image。该方法将输入值连同其位置传播到内部单元，并通过组合神经元图查看卷积、池化和 residual 模块。本文继承“一个内部对象对应一张输入空间图”的研究视角，但采用相对参考输入的路径积分来处理 ViT 中的平滑非线性、输入相关 attention 和类别输出。原工作在 FCN/CNN 上的实验结果不作为本文 ViT 结论的证据。

## 2.2 输入归因与内部路径归因

Integrated Gradients（IG）沿参考输入到观测输入的路径积分梯度，并满足灵敏度和实现不变性等公理；在数值积分充分时，各输入维度归因之和逼近输出差值（Sundararajan et al., 2017）。本文使用 IG 形式构造内部响应或类别标量的 learning image，因此 reference、积分路径和 completeness 都属于方法定义，而不是显示参数。

内部归因进一步追问一个输出通过哪些隐藏单元或路径形成。Neuron conductance 将 IG 的思想扩展到隐藏单元，度量归因经过单元的流量（Dhamdhere et al., 2018）。本文的完整切分同样使用输入方向导数与下游目标导数的链式分解，并把一个 class-logit 归因组织为 ViT 的 head、attention、MLP 和 residual 成分。该数学关系属于 IG/conductance/path-attribution 范畴；本文关注的是 ViT 结构映射、完整切分验证和相关实证，而不把路径积分或链式法则作为新发明。更完整的 path-attribution 技术对照仍需在投稿前补齐 `[REFERENCE REQUIRED]`。

## 2.3 ViT attention 与类别解释

ViT 把 patch token 的关系编码在多层多头 attention 中（Dosovitskiy et al., 2021）。Raw CLS attention 可以直接排成 patch 网格，但它不包含完整下游类别导数。Attention rollout 将层间 attention 与 residual 路径递归相乘，用于估计 token 间的累计 attention flow（Abnar and Zuidema, 2020）。Chefer et al. (2021) 将 relevance 与 attention gradient 结合，在 transformer 图上聚合类别相关解释。Grad-CAM 则以目标梯度对特征通道加权，最初用于卷积网络，后续也被适配到 ViT（Selvaraju et al., 2017）。

另一组研究从可视分析角度研究多头 attention。Li et al. (2023) 比较 head importance、patch 邻域关系和跨层 attention pattern；Yeh et al. (2023) 通过 query-key 联合嵌入提供跨输入的全局 attention 视图。这些方法主要呈现 attention 结构或交互模式。本文以输入坐标下的响应贡献为核心对象，并将 raw attention、rollout、Grad-CAM 和 full-model IG 作为不同目标或不同路径的对照，而不是把它们视为同一种热图。

## 2.4 解释评价

解释图的视觉可读性不能替代其与模型响应的关系。已有工作使用删除、插入、faithfulness correlation、sensitivity、randomization 和复杂度等指标评价解释。Komorowski et al. (2023) 在 ViT 医学影像任务中同时比较 faithfulness、sensitivity 和 complexity，说明解释器的评价需要明确目标与指标。本文的主实验采用配对 insertion/deletion 曲线：按解释图排序逐步恢复或替换输入区域，再比较同一内部目标的响应。独立类别输出实验则在固定 50 图队列上比较五类解释器。

稳定性研究表明，输入的微小变化可能显著改变归因。本文把空间稳定性作为独立问题处理。稳定性门未通过时，细粒度 fidelity 的正结果仍按其原目标报告，但不被扩写为几何鲁棒性。随后进行的多尺度聚合仅提供事后探索性解释。
