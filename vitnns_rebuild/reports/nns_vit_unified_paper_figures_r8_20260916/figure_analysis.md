# R8：这 14 张图分别在说什么

Figure 2 改为单个响应的红蓝 learning image，其余图沿用 R7。

## Image #1: Scanning a Vision Transformer with NNS

先看左边：同一张图被分成 patch，送进 ViT。再看右边：每一列换一层，每一行换一个 head，NNS 给每个内部响应配了一张能对回原图位置的图。

这就是首图要讲的价值：我们不只看 ViT 最后给出的类别，还能打开它，查看不同层、不同 head 的响应与输入哪些位置有关。亮点是这些响应的正贡献位置，不是 attention 权重。这里只画了九个真实例子，完整 144-head 图在附带画廊。

图里的 patch 网格说明输入位置；已有数据测的是 CLS 的 head 响应，并没有把每个 patch token 都单独算一张解释图。

## Image #2: A learning image locates signed input contributions

这次只解释一件事：一个 head 的响应，怎样对应到输入图像的不同位置。左边选定内部响应，中间把它对应的 learning image 画成红蓝图：红色为正贡献，蓝色为负贡献，白色为零。

右边两张图放大的是同一个框。正图保留红色格子，负图保留蓝色格子；同一个格子不会同时出现在两边。正负可能紧挨着，所以这里保留原始小格，不把它们平滑成相似的轮廓。全图和放大图共用色阶。

例子换成了已有 F3H 的 B0/H10，是看过结果后为讲解选的。主实验仍是 B0/H9；这张图不提供 H10 的新效果结论。这里的“正负”指相对参考输入累计的贡献，也不直接表示单独遮掉这个位置，类别分数就一定升或降。

## Image #3: Evaluate the ranking at a shared spatial scale

这张图说明“怎么判断一张解释图有没有用”。先把图像切成同样的评价单位：每个 ViT patch 里面再分成 16 个 `4×4` 小格。NNS 可以给同一个 patch 里的小格不同分数；token baseline 在同一个 patch 里同分。

然后按分数排序做两件事：从灰图开始优先恢复高分区域，看内部目标恢复得快不快；从原图开始优先遮挡高分区域，看内部目标掉得快不快。谁的排序更贴近模型内部信号，谁的 AUC 就更好。

这图只是流程图，不是结果图。真正结果看 Figure 4 和 Figure 5。图里放大的遮挡块是为了让流程看得懂，真实实验用的是 `4×4` 小格，并且每张图内部会处理同分排序。

## Image #4: Primary effect distribution

这是主结果。每一个点/柱背后是一张图像的 `NNS - token baseline` 忠实度差值。500 张图全部为正，说明在这个固定内部目标和 `4×4` 评价设置下，NNS 的排序稳定高于同目标 token baseline。

中位数和置信区间是总体结论，不是单张图的误差条。它证明的是这个预定设置下的主效应，不是证明 NNS 在所有模型、所有 head、所有解释任务上都更好。

## Image #5: Fidelity trajectories

这张图把 Figure 4 的差值拆开看：差距到底来自恢复曲线，还是遮挡曲线。直觉上，好的解释应该让重要区域恢复得更快、遮掉后响应掉得更快。

看这图时重点看曲线之间的距离和它出现的位置。它不是新的统计结论，而是帮你理解 `ΔFID` 是怎么来的。

## Image #6: Scale and operator sensitivity

这张图回答一个很容易被误解的问题：NNS 的收益是不是只因为我们用更细的评价格子？结果是，细到 `4×4` 时主效应很明显；回到原生 `14×14` patch 尺度，差异接近消失。

所以这张图给论文一个边界：主结果依赖 fine-cell 干预设置。右边的插值/算子比较是敏感性分析，用来确认展示和空间实现没有随便换规则。

## Image #7: Layer/head and prediction strata

这张图说明：NNS 的效果不是 ViT 任意层任意 head 都一样。主 head `block 0/head 9` 很强，另外三个预先指定坐标小很多。右上角只是把小效应放大看清楚，主图和 inset 不是两组数据。

灰、蓝、橙分别是 overall、正确分类、误分类。三组方向类似，说明主结果不是只靠正确分类样本撑起来的。但这不是等效性检验，也不能把三个组当成三个独立实验相加。

## Image #8: A median-effect image viewed through all 144 heads

这张就是你喜欢的全 144-head 汇总视图。它仍然是原来的中位效应样本，只是不再只展示最浅层的一个 head，而是把全部 12 层、每层 12 个 head 的显示图汇总起来。

从左到右看原图、NNS 正向、baseline 正向、NNS 负向、baseline 负向。可以对照按键、显示屏、面板边界附近的空间纹理；“正、负”说的是对各自内部响应的作用方向。

这里的汇总是把显示图平均起来看空间形态，不是说 144 个 head 的类别贡献已经加总完成。“中位效应”仍然指这张图在原主实验里的位置。

## Image #9: Spatial support across transformer depth

这张图把 Figure 8 的全头汇总拆成四段：浅层、两段中间层、深层。每一列都包含三层的全部 head，所以能看到整体形态怎样随深度变化。

先横着看第一行，再看第三行，分别观察正向和负向结构；第二、四行是同目标 token baseline。不要仅凭亮度说哪一层更重要，每个内部坐标的显示尺度不同。

## Image #10: Different explanations of the same images

这张沿用你喜欢的 F3C 图库风格，并完整放入第一页的五张图，没有从中只留 NNS 好看的样本。每一行是一张输入，横着看不同方法给出的解释。

NNS 和 full IG 各有两列：原图尺寸的细节版，以及 4×4 最大池化后的易读版。后者更容易看到成片的结构，但只是换了显示方式，没有提高实验分数。Grad-CAM、rollout、raw CLS 也都保留。

这里问的是类别输出，前面全头图问的是内部 head 响应，两组图的目标不同。到底哪个方法分数更高，接着看 Figure 11。

## Image #11: Method comparison statistics

这张图是方法比较的定量版。结论很直接：在这组 class-targeted 比较里，NNS 低于 Grad-CAM，高于 rollout；和 full IG、raw CLS attention 的差异区间跨过 0。

所以论文不能写“NNS 全面优于现有解释方法”。更准确的说法是：NNS 在主 Value-path 实验中优于同目标 token baseline；在另一个 class-targeted 比较里，它和不同方法的关系各不相同。

## Image #12: Post-hoc exploratory multiscale mechanism

这张图解释一个后续现象：把热图从 pixel 看粗到 `4×4`、`16×16` 后，几何一致性会变高。也就是说，一些不稳定来自小区域内部的重新分配。

这不是正式通过的稳定性结果。它只能作为 post-hoc exploratory 机制线索，不能把之前的稳定性 NO-GO 改写成 PASS。

## Image #13: Branch allocation across ViT blocks

这张图把“输入贡献经过哪些内部分支”说得更具体。左边看 12 个 block 的轨迹，右边看深层减浅层的方向和大小。

可以记住的结论是：attention 分支和 MLP 分支的贡献份额随深度变化，而且方向不同。不要把两条曲线相加当成 100% 概率；它们来自不同的 cut 定义。

## Image #14: Post-hoc exploratory depth extension

这张图是深度方向的探索性扩展。它把几个问题放在一起：高贡献区域是否更贴近目标、遮挡这些区域是否更影响输出、深层表示是否更容易读出目标信息，以及这些变化之间是否同步。

表格里的四个 endpoint 量纲不同，不能横向比谁“大”。整张图都属于 post-hoc exploratory，用来形成后续机制假设，不是主论文的 confirmatory 结论。
