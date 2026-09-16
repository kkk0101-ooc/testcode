# Unified paper figures R2 · zh

## Figure 1

研究总览：从 ViT 的固定 block/head 内部目标出发，比较 NNS 的 patch 内细分与同目标 patch-token 基线的均匀分配，随后以配对图像干预评价。原图复用冻结 F3C3 cohort 的第一个 formal position；网络、贡献网格与恢复序列均为方法示意，不是该图像的实测归因或实验结果。蓝色标识 NNS，橙色标识 patch 基线。

## Figure 2

Head-direction Value-path 构造。(a) 在 clean 输入处确定 attention A 与方向 u，令 f(z)=uᵀg(z)；A 和 u 在整个积分路径上固定，Value features 随输入变化。(b) 输入从 normalized-zero reference x₀ 到 clean x 插值，求 signed input-space integrated gradients，最后对 RGB 通道带符号求和。(c) 同目标 token contribution cⱼ 的 196 个 patch 项形成原生 14×14 map，并均分到每 patch 的 16 个 fine cells；CLS scalar 单独保留。输入使用冻结示例，attention 与 contribution 网格仅为示意。signed 示意网格用蓝/红区分负/正值，不与方法识别色混为一谈。

## Figure 3

评价协议。(a) 224×224 图像对应 14×14 个 16×16-pixel patches；fine endpoint 使用每 patch 的 16 个 4×4-pixel cells，共 3,136 cells。(b) 按 target-oriented positive ranking 执行 insertion/deletion，使用 normalized-zero / RGB 0.5 reference、21 个名义 fractions 和 5 次方法间配对的 tie repeats。为便于阅读，图像序列以放大的示意区域表现恢复/删除，并非实测 mask 或曲线。(c) 先在每图内平均五次 AUC，再求 NNS−patch 的 ΔFID；以图像为独立单位求总体 median 和 paired bootstrap CI。AUC 使用实际干预比例，响应不截断到 [0,1]。

## Figure 4

Fine-scale 主效应。(a) 直方图保留全部 500 个逐图 ΔFID；橙色线为 median，橙色区域为该 median 的正式 paired-image bootstrap 95% CI。(b) 相同数据的 ECDF，圆点定位 median，橙色区域仍表示 median CI，而非 ECDF 的同时置信带。500/500 个差异为正；median=2.3081996222，95% CI=[2.1110806126, 2.4452354118]。

## Figure 5

NNS 与 matched patch 的 median insertion/deletion trajectories。阴影为 pointwise bootstrap interval。

## Figure 6

尺度与上采样敏感性。(a) fine 4×4-cell 与 native-patch 评价的 ΔFID 分布。(b,c) 每个点为一个图像的 insertion/deletion AUC，横向抖动仅用于避让；箱体为第 25–75 百分位，箱内线为 median，每类均为 500 张图像。全部数据点和极值保留；IQR 不是置信区间。Uniform 与 mass-preserving nearest 完全等价，合并为一个 evidence class；bilinear 与 bicubic 为描述性 sensitivity。

## Figure 7

四个冻结 layer/head 坐标与预测分层的统一矩阵。列为 Stage C 预先选定的坐标，行为 overall、correct、misclassified；每格依次显示 median ΔFID、Stage E 描述性 paired-image-bootstrap 95% CI 和 valid/expected 图像数。全矩阵使用同一线性色阶。Block 11/head 2 有 2 个 typed failures：overall 498/500，correct 417/418，misclassified 81/82；其他坐标分别为 500/500、418/418、82/82。该图的描述性 bootstrap namespace 与 Figure 4 正式主分析不同，因此 primary median 相同而 CI 略有差异。三组同向不构成等效性检验，也不能推广为全部 layer/head 的结论。

## Figure 8

冻结案例图。展示 median-effect、weak-effect 和 misclassified 样本的原图、NNS、matched patch 与差异形态；每个 attribution panel 独立缩放。

## Figure 9

一个冻结样本的 head-averaged spatial support。列为四个 depth bands，行为 NNS/matched 的正负 contribution；跨 depth band 不比较绝对幅值。

## Figure 10

解释方法的空间比较。展示 formal cohort 前四个位置的原图、NNS、pooled NNS、Grad-CAM、full-model IG 和 rollout；每个 attribution panel 使用 P99 display scaling。

## Figure 11

50 张冻结图像上的解释方法比较。(a) 每图五次 tie-repeat 平均 fidelity 的散点，箱体表示 IQR，箱内线表示 median。(b) 四个注册的 NNS−comparator paired mean contrasts：实心条从零延伸至估计值，菱形定位 mean，空心矩形左右边界为正式 bootstrap 95% CI；矩形高度只用于展示。NNS−Grad-CAM 为负，NNS−rollout 为正，full IG 与 raw CLS contrasts 的区间跨零。区间不是样本 IQR，也没有由区间拟合概率密度。primary Grad-CAM contrast 使用原始检验，secondary family 沿用 Holm，图面不重新作显著性决策。(c) 原 median deletion trajectories。

## Figure 12

Post-hoc exploratory multiscale mechanism analysis。固定 flip route 的示例 map，以及 pixel、4×4、16×16 下的 geometric agreement、ROAD fidelity 和 NNS-v2 SSE decomposition。

## Figure 13

ViT block-level class-logit complete-cut branch allocation。(a) 完整 12-block attention 与 MLP branch absolute-mass share trajectories，色带为原逐 block 95% CI。(b) B4/B5 的 deep-minus-shallow 估计：实心条表示符号与大小，菱形表示 estimate，空心矩形左右边界表示正式 95% CI。Attention 与 MLP 使用一致的蓝/橙标识。这里比较的是分别定义的 residual cuts 中的 allocation shares，两条曲线不应相加当成一个二分概率分布。受支持的分支结论不替代 Y4 其他 claim 的 INVALID/NOT_SUPPORTED 记录。

## Figure 14

Post-hoc exploratory 深度扩展。(a) 分别列出 P1、P2、D1、D2 的 estimate、ordinary BCa 95% CI 与有效/总图像数，端点含义不同，不共用效应坐标轴。P1、D1 为 235/240，P2 为 224/240，D2 为 3424/3482；对应 NA 为 5、5、16、58，failed 均为 0。ordinary CI 只表示精度，不替代调整后的判断规则。(b) NNS alignment 与 probe gain 的完整 depth profiles，比较各自的深度变化，不将两者幅值视为同一物理量。(c) 固定三种 patch budgets 下的 matched-control-corrected logit drops。(d) 233 个有效配对图像的 D1/D2 关系，Pearson r 约为 0.002。全部 Stage S 结论属于 post-hoc exploratory；端点及 P2 阈值在 outcome access 后形成，不作 confirmatory 或因果机制解释。
