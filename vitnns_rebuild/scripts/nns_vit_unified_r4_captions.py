"""R4 explanations: model representation, attribution readout and spatial expansion."""

TITLES = {
    1: 'One ViT, two attribution readouts',
    2: 'Same internal target, different contribution coordinates',
    3: 'Evaluate the ranking at a shared spatial scale',
}

CAPTIONS_ZH = {
    1: '同一个已训练 ViT 的两种归因读出。Patch embedding 将每个 16×16×3 输入区域映射为高维 token，可以编码 patch 内像素位置差异；self-attention 的权重随输入、层和 head 改变。本研究选择一个内部目标，比较 NNS 的输入空间归因与同目标 token 归因基线。基线先计算每个 token 的贡献 c_j，再均匀展开到对应 patch 内的 16 个 4×4-pixel cells。示例 patch A 的 c_A=16、patch B 的 c_B=8，因此各自每 cell 为 1、0.5；不同 patch 的分数仍不同。NNS 的示意网格表示输入位置的 signed 贡献，不由这些 c_j 插值得到。两种方法解释同一目标，不要求每个输入 patch 的 NNS 贡献和等于同索引 token 的 c_j。原图沿用冻结 F3C3 formal position 1；贡献网格与数字均为示意，不是该图像的实测结果。蓝、橙用于方法标识；NNS 网格蓝/红表示负/正贡献。',
    2: '同目标的两条计算路径。对选定 block/head，g(z)=Σ_j A(x)_{0j}V_j(z)，u=[g(x)−g(x₀)]/‖g(x)−g(x₀)‖₂，f(z)=uᵀg(z)。A(x) 是 clean 输入处 CLS query 的 attention，u 是 clean 与参考输入之间 head 输出变化的单位方向；二者对当前图像的归因及干预评价固定，Value 随输入变化。左侧 NNS 对 f 沿 x₀→x 计算输入空间积分梯度，RGB 通道带符号求和后得到像素图，再汇总到 4×4-pixel cells。右侧基线直接计算 c_j=A(x)_{0j}uᵀ[V_j(x)−V_j(x₀)]，保留 196 个 patch-token 标量并将 CLS 标量单列；它结合 attention、Value 变化和目标方向，不是 raw attention。均匀展开给每 cell c_j/16。x₀ 是 normalized-zero / RGB 0.5 灰色参考。这里固定 A 的操作属于解释目标定义，不代表正常 ViT 训练时 attention 固定。深层 token 可包含其他空间位置的信息，token j 的贡献放回输入 patch j 是空间映射约定。图片路径和贡献数字仅作方法示意，NNS patch 总和不被强制设为 c_j。',
    3: '统一空间尺度上的排序与忠实度评价。224×224 图像包含 14×14=196 个原生 patch，每个 16×16-pixel patch 再划为 16 个 4×4-pixel cells，全图 3,136 个 fine cells。NNS 对每 cell 的 signed 输入贡献求和后按目标方向取正部排序；token 基线先把每个 c_j 均分，再按相同规则排序，因此同一 patch 内同分，不同 patch 之间仍可不同。图中的 1–16 为一个 patch 内的示意局部次序；真实实验在全部 cells 上排序，并对所有同分组使用 5 次方法间配对的 tie permutations。Insertion 从同一灰色参考恢复高分区域，deletion 从原图替换高分区域；两者均评价 Figure 2 定义的内部 f，使用相同干预面积预算。图中放大的遮挡块只说明操作顺序，不是实际 4×4-pixel mask 或实验结果。原协议使用 21 fractions、实际干预比例上的 AUC，响应不截断到 [0,1]。每图先平均 5 次 AUC，计算 F=½(AUC_ins−AUC_del) 及 ΔFID=F_NNS−F_token；随后以图像为独立单位报告 median 和 paired bootstrap 95% CI。',
}

CAPTIONS_EN = {
    1: 'Two attribution readouts of the same trained ViT. Patch embedding maps each 16×16×3 input patch to a feature vector and can encode differences between pixel positions within that patch. Self-attention weights depend on the input, block and head. For one selected internal target, NNS computes contributions in input coordinates, while the matched token baseline computes one scalar c_j per token and distributes it uniformly among the 16 fine cells of the corresponding patch. Illustrative scores c_A=16 and c_B=8 yield 1 and 0.5 per cell, respectively: scores still differ between patches. NNS grids illustrate signed input contributions computed independently of c_j, with no enforced equality between an input-patch sum and the contribution of its indexed token. The original image is frozen F3C3 formal position 1; grids and numbers are illustrative, not measured attributions. Blue/orange labels identify methods; blue/red in the NNS grids encode negative/positive contributions.',
    2: 'Two computations for a shared internal scalar. For a selected block/head, g(z)=Σ_j A(x)_{0j}V_j(z), u=[g(x)−g(x₀)]/‖g(x)−g(x₀)‖₂, and f(z)=uᵀg(z). Clean CLS attention A(x) and direction u are held fixed for this image during attribution and intervention evaluation, while Value features vary with the input. NNS integrates input gradients of f along x₀→x, sums RGB channels with sign, and aggregates the resulting pixel contributions into 4×4-pixel cells. The matched baseline computes c_j=A(x)_{0j}uᵀ[V_j(x)−V_j(x₀)], combining attention, Value change and target direction. Its 196 patch-token scalars are spatialized by assigning c_j/16 to each fine cell; the CLS scalar is retained separately. The reference is normalized zero / RGB 0.5. Fixing attention defines this explanation target, not the normal ViT training procedure. At deeper blocks, token j can mix information from other input positions, so spatializing token j at patch j does not imply that its contribution equals the NNS sum in that input patch. Paths and contribution grids are illustrative.',
    3: 'Ranking and fidelity at a shared spatial scale. A 224×224 image has 196 native 16×16-pixel patches, each divided into 16 fine 4×4-pixel cells, giving 3,136 cells. NNS sums signed input contributions within each cell and ranks their target-oriented positive parts. The token baseline uniformly expands c_j before the same ranking rule, giving ties within each patch while preserving differences between patches. The local 1–16 ranks are illustrative; the actual experiment ranks all cells and uses five paired tie permutations for every tie group. Insertion restores high-ranked regions from the gray reference; deletion replaces them in the clean image. Both use the same area budget and evaluate the internal scalar f defined in Figure 2. Enlarged mask blocks illustrate the operation, not actual fine-cell masks or outcomes. The original protocol uses 21 fractions and AUC against actual intervention fractions, with responses unclipped. Five AUC repeats are averaged within each image before computing F=½(AUC_ins−AUC_del) and ΔFID=F_NNS−F_token. Images are the independent units for the median and paired bootstrap 95% CI.',
}

ANALYSIS = {
    1: '''先看上半部：同一张图经过同一个已训练 ViT。每个 token 是高维向量，能够编码 patch 内结构；模型对不同 patch 的注意力本来就可以不同。图没有把 patch 描述成“内部像素完全相同”。

再看下半部：NNS 与 token 基线是两种解释读出。左侧 NNS 给输入位置计算不同贡献；右侧基线每 token 只有一个贡献标量。示例中 patch A 总分 16、patch B 总分 8，分别均分为每 cell 1 和 0.5。两块之间不同，各块内部相同。图面直接标明两个 patch，避免将局部 4×4 网格误读为整图 patch 网格。

这张图说明比较对象，不证明 NNS 的细节更准确；准确性需要 Figure 3 的干预评价。NNS 示例与 token 数值是独立的示意，不宣称每个输入 patch 的 NNS 总和等于 c_j。''',
    2: '''上方先固定双方共同解释的标量 f：选定 head 的 CLS 聚合结果沿方向 u 的分量。u 来自原图相对灰色参考的 head 输出变化，A(x) 是原图的 CLS attention。固定 A/u 发生在这个解释问题中，正常 ViT 训练和推理仍可产生不同 attention。

左侧从灰色参考逐步走向原图，通过积分梯度计算输入位置贡献，RGB 带符号相加，得到细粒度地图。右侧直接从 token 的 Value 端点变化计算 c_j，然后均匀展开。右侧不是 raw attention；c_j 同时取决于 A、Value 变化和 u。

两个箭头表示独立计算途径：NNS 并不是拿 c_j 做更漂亮的插值。尤其在深层，一个 token 已混合其他位置的信息，因此 c_j 的原图空间映射与 NNS 的输入归因坐标不同。CLS 标量单列，不硬分到某个图像位置。''',
    3: '''左列说明尺度：14×14 是整图的 patch 数量网格；16×16 是每个 patch 的像素尺寸；每个 patch 内有 16 个 4×4-pixel cells。中列比较细分排序：NNS 可区分同 patch 内的小区域，均匀基线在一个 patch 内同分。它仍能区分不同 patch。

右列说明检验：从灰图优先恢复高分区域，观察内部目标恢复的速度；从原图优先遮挡高分区域，观察目标下降的速度。双方使用相同目标和面积预算。展示中的大块只是示意，真实 fine 实验以 4×4 像素为单位。

下方说明汇总：先在每图内平均 5 次同分排序的 AUC，再计算双方的忠实度差值，最后跨图像求 median 和 paired bootstrap CI。五次重复不是五张独立图像。图中局部排序号与 mask 不是真实测量，不能用它们直接判断 NNS 获胜。''',
}
