# 统一论文图 R6：说人话逐图说明

这份说明只帮读图，不替代正式 caption。每张图都按“它想让你看懂什么”和“别误读成什么”来写。

## Image #1: NNS links internal ViT responses to image evidence

这张图要先把 NNS 的价值讲清楚：它不是单纯把热图画细，而是把“模型内部某个响应”追到“输入图像里的哪些区域在推它、哪些区域在压它”。

左边表示我们先选一个内部 head；中间是真实样本上的 signed map，红色是正贡献，蓝色是负贡献；右边是最重要的证据：500 张图里，按 NNS 排序做遮挡/恢复，分数都比同目标 token baseline 高。也就是说，这张图想告诉读者：这个内部解释不是只好看，它的排序在干预实验里有用。

别把它读成“红蓝图上每个小格都已经被单独证明有因果作用”。右边的 500/500 验证的是整体排序。公式、baseline 怎么算、怎么遮挡，放到 Figure 2 和 Figure 3 再解释。

## Image #2: Same internal target, different contribution coordinates

这张图解释 Figure 1 中间那张红蓝贡献图是怎么算出来的。先固定一个内部目标 `f`：它来自某个 block/head 的输出方向。这里固定的是“这次解释要看的目标”，不是说 ViT 训练或正常推理时 attention 永远固定。

左边是 NNS 这条路：从灰色参考图走到原图，沿路径看输入像素怎么影响这个内部目标，最后得到输入空间里的 signed map。右边是对照 baseline：它给每个 token 算一个贡献分数 `c_j`，再把这个分数均匀摊到对应 patch 内。

关键点是：NNS 不是把 token 分数做插值，也不是把 attention 权重画细。NNS 和 baseline 是两种坐标下的贡献读出。深层 token 可能已经混入其他位置的信息，所以不能说 token `j` 的贡献就等于原图第 `j` 个 patch 内的 NNS 总和。

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

## Image #8: Frozen case atlas

这张图给读者看具体样本长什么样：一个中位效应样本、一个弱效应样本、一个误分类样本。它让你看到 NNS、token baseline 和差异图在真实图像上是什么空间形态。

这些例子是按冻结规则选出来的，不是人工挑最好看的。它们帮助理解形态，但不能单独支撑总体结论。总体结论仍然看 Figure 4–7。

## Image #9: Head/depth progression

这张图看的是同一张图像在不同深度段上的空间形态。它让你直观看到：随着网络变深，关注形态会变。

这是一张结构展示图，不是 cohort-level 统计图。别用单个样本去推出所有 head 的规律；整体层/head 结论要和 Figure 7 的统计结果一起看。

## Image #10: Method comparison atlas

这张图把 NNS 和常见解释方法放在同一批样本上看。你能直观看到：不同方法给出的空间支持不一样，有的更集中，有的更扩散，有的偏向不同区域。

每个面板为了看得清楚做了显示缩放，所以不能用颜色亮度直接比较方法强弱。谁的 fidelity 更高，要看 Figure 11。

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
