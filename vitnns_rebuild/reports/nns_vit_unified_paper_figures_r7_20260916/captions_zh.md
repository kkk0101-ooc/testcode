# NNS ViT R7 · zh

## Figure 1

NNS 扫描 ViT 的内部响应。左侧将同一张输入图像分成 patch tokens，经过已训练的 ViT；右侧按层（B0、B5、B11）和 head（H0、H5、H11）排列实测 NNS 正贡献图。每格表示输入位置对该坐标 CLS head-direction Value-path 标量的正贡献。层和 head 采用首、中、末位置作展示，不按效果排名。底色到亮色表示该坐标显示尺度下的正贡献。示例为先前冻结的 order 44 中位效应样本。网格标记输入 patch 位置；这些地图的查询目标是 CLS，不是各 patch-query 的独立归因。Figure 2 解释对应关系，Figure 8–9 展示全部 head 的汇总。

## Figure 2

ViT 内部响应与 NNS 输入图的对应关系。上排展示 patch tokens 经过不同 Transformer block，每个 block 内具有多个 head；本图在 B0、B5、B11 都选择 H9 的 CLS 响应。下排每对图与其正上方的目标对应，分别显示正贡献和负贡献幅度。灰色虚线表示响应到解释图的对应关系，下排地图不是沿箭头彼此传播的中间结果。计算使用固定 clean attention 和方向 u 的 Value-path 标量 f，并对参考输入到原图的路径积分；公式中的 i 是输入标量坐标，绘图时按带符号 RGB 和及 4×4 cells 聚合。这里借鉴原 NNS 为内部单元配置输入位置图的构图，ViT 的实际计算为条件路径积分。原来的详细目标/基线对照图另附为补充。

## Figure 3

统一空间尺度上的排序与忠实度评价。224×224 图像包含 14×14=196 个原生 patch，每个 16×16-pixel patch 再划为 16 个 4×4-pixel cells，全图 3,136 个 fine cells。NNS 对每 cell 的 signed 输入贡献求和后按目标方向取正部排序；token 基线先把每个 c_j 均分，再按相同规则排序，因此同一 patch 内同分，不同 patch 之间仍可不同。图中的 1–16 为一个 patch 内的示意局部次序；真实实验在全部 cells 上排序，并对所有同分组使用 5 次方法间配对的 tie permutations。Insertion 从同一灰色参考恢复高分区域，deletion 从原图替换高分区域；两者均评价 Figure 2 定义的内部 f，使用相同干预面积预算。图中放大的遮挡块只说明操作顺序，不是实际 4×4-pixel mask 或实验结果。原协议使用 21 fractions、实际干预比例上的 AUC，响应不截断到 [0,1]。每图先平均 5 次 AUC，计算 F=½(AUC_ins−AUC_del) 及 ΔFID=F_NNS−F_token；随后以图像为独立单位报告 median 和 paired bootstrap 95% CI。

## Figure 4

Fine-scale 主效应。(a) 直方图保留全部 500 个逐图 ΔFID；橙色线为 median，橙色区域为该 median 的正式 paired-image bootstrap 95% CI。(b) 相同数据的 ECDF，圆点定位 median，橙色区域仍表示 median CI，而非 ECDF 的同时置信带。500/500 个差异为正；median=2.3081996222，95% CI=[2.1110806126, 2.4452354118]。

## Figure 5

NNS 与 matched patch 的 median insertion/deletion trajectories。阴影为 pointwise bootstrap interval。

## Figure 6

尺度与上采样敏感性。(a) fine 4×4-cell 与 native-patch 评价的 ΔFID 分布。(b,c) 每个点为一个图像的 insertion/deletion AUC，横向抖动仅用于避让；箱体为第 25–75 百分位，箱内线为 median，每类均为 500 张图像。全部数据点和极值保留；IQR 不是置信区间。Uniform 与 mass-preserving nearest 完全等价，合并为一个 evidence class；bilinear 与 bicubic 为描述性 sensitivity。

## Figure 7

Figure 7. 四个冻结 layer/head 坐标的 NNS−patch baseline median ΔFID。灰、蓝、橙色分别表示 overall、correct、misclassified，柱高为 median，误差线为原 Stage E 图像级配对 bootstrap 描述性 95% CI。主图所有坐标共用从零起始的纵轴，右上 inset 将后三个坐标按 0–0.10 的线性纵轴放大；两处呈现的是同一组数值，不能重复计为新观测。总体 cohort 为 500，正确/误分类分别为 418/82；block 11/head 2 的有效数为 498/417/81，其余坐标全部有效，两个 typed failures 仍保留在分母中。坐标在 Stage C 冻结，三组包含重叠样本，不是三个独立实验，也未检验组间等效性。区间沿用 Stage E 的描述性 bootstrap namespace，与 Figure 4 的正式主分析区间略有差异。

## Figure 8

中位效应图像的全头空间汇总。使用用户指定的 F3H Image 19 所对应的同一数据：order 44、ILSVRC2012_val_00002287.JPEG。左至右为原图、NNS 正贡献汇总、同目标 token baseline 正贡献汇总、NNS 负贡献幅度汇总及 baseline 负贡献幅度汇总。每个 B/H 坐标的两种方法先共用最大绝对值 L，采用 log1p(999·m/L)/log(1000) 并按原 F3H 规则线性放大，再对全部 12×12=144 个 head 等权平均。图像“中位效应”标签来自主实验 B0/H9 的冻结选样，不是对平均图的 fidelity 评价；平均图描述空间支持，不是类别归因。原始 Image 19 也逐字节收录在 reference_gallery。

## Figure 9

同一中位效应样本在四个深度段的空间支持。列为 B0–2、B3–5、B6–8、B9–11；行依次为 NNS 正、token baseline 正、NNS 负贡献幅度及 token baseline 负贡献幅度。每列使用该段全部 3×12=36 个 head，沿用 F3H Image 18 的逐坐标配对色阶和显示数组平均。该图用来观察深度与空间形态的关系，颜色不表示跨层可比的原始幅值，也不把这些阶段预先命名为边缘、部件或语义阶段。

## Figure 10

同一输入的多种解释视图。完整保留 F3C visual 第 1 页的五张样本，按冻结 formal positions 1–5 排列。列为原图、NNS 原生正贡献、NNS 4×4 最大池化后双线性显示、Grad-CAM、full-model IG 正贡献、full IG 的相同池化显示、rollout 和 raw CLS attention。NNS 在这里以条件 class logit 为目标，与 Figure 1、8、9 的单 head 目标不同。每个热图分别按自身 P99 缩放，池化仅用于显示；方法定量比较见 Figure 11，完整 50 图及 signed/unclipped 对照附在画廊。

## Figure 11

50 张冻结图像上的解释方法比较。(a) 每图五次 tie-repeat 平均 fidelity 的散点，箱体表示 IQR，箱内线表示 median。(b) 四个注册的 NNS−comparator paired mean contrasts：实心条从零延伸至估计值，菱形定位 mean，空心矩形左右边界为正式 bootstrap 95% CI；矩形高度只用于展示。NNS−Grad-CAM 为负，NNS−rollout 为正，full IG 与 raw CLS contrasts 的区间跨零。区间不是样本 IQR，也没有由区间拟合概率密度。primary Grad-CAM contrast 使用原始检验，secondary family 沿用 Holm，图面不重新作显著性决策。(c) 原 median deletion trajectories。

## Figure 12

Post-hoc exploratory multiscale mechanism analysis。固定 flip route 的示例 map，以及 pixel、4×4、16×16 下的 geometric agreement、ROAD fidelity 和 NNS-v2 SSE decomposition。

## Figure 13

ViT block-level class-logit complete-cut branch allocation。(a) 完整 12-block attention 与 MLP branch absolute-mass share trajectories，色带为原逐 block 95% CI。(b) B4/B5 的 deep-minus-shallow 估计：实心条表示符号与大小，菱形表示 estimate，空心矩形左右边界表示正式 95% CI。Attention 与 MLP 使用一致的蓝/橙标识。这里比较的是分别定义的 residual cuts 中的 allocation shares，两条曲线不应相加当成一个二分概率分布。受支持的分支结论不替代 Y4 其他 claim 的 INVALID/NOT_SUPPORTED 记录。

## Figure 14

Post-hoc exploratory 深度扩展。(a) 分别列出 P1、P2、D1、D2 的 estimate、ordinary BCa 95% CI 与有效/总图像数，端点含义不同，不共用效应坐标轴。P1、D1 为 235/240，P2 为 224/240，D2 为 3424/3482；对应 NA 为 5、5、16、58，failed 均为 0。ordinary CI 只表示精度，不替代调整后的判断规则。(b) NNS alignment 与 probe gain 的完整 depth profiles，比较各自的深度变化，不将两者幅值视为同一物理量。(c) 固定三种 patch budgets 下的 matched-control-corrected logit drops。(d) 233 个有效配对图像的 D1/D2 关系，Pearson r 约为 0.002。全部 Stage S 结论属于 post-hoc exploratory；端点及 P2 阈值在 outcome access 后形成，不作 confirmatory 或因果机制解释。
