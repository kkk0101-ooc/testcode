# Unified paper figures R5 · zh

## Figure 1

NNS 将选定内部响应与输入区域的贡献联系起来。(a) 对同一个已训练 ViT-B/16 选择一个内部 head 的响应。(b) 展示 F3 预先冻结的 median-effect 案例（selection_order=44，ILSVRC2012_val_00002287.JPEG，block 0/head 9）的真实 NNS signed fine-cell map；红色为对固定 head-direction Value-path 标量的正贡献，蓝色为负贡献。原图与地图一一对应，非新模型计算或示意归因。为同时显示小幅正负结构，颜色采用 sign(m)·log(1+99·|m|/L)/log(100)，L=max|m|；地图为 56×56，每 cell 对应 4×4 输入像素，最近邻显示。该变换仅用于显示。(c) 正式独立 500 图 cohort 中，500/500 个 NNS−同目标 token 空间基线 ΔFID 为正；每个方格表示一张图像的正差值，方格不编码效应大小。median ΔFID=2.3082，正式图像级配对 bootstrap 95% CI=[2.1111,2.4452]。这里的 F 采用对正向贡献排序的 fine-cell insertion/deletion；该总体结果不为案例中每个局部 signed cell 提供单独因果验证。首图说明内部解释用途及其一项实证支持；计算和评价分别见 Figure 2–3，完整分布见 Figure 4–5。

## Figure 2

同目标的两条计算路径。对选定 block/head，g(z)=Σ_j A(x)_{0j}V_j(z)，u=[g(x)−g(x₀)]/‖g(x)−g(x₀)‖₂，f(z)=uᵀg(z)。A(x) 是 clean 输入处 CLS query 的 attention，u 是 clean 与参考输入之间 head 输出变化的单位方向；二者对当前图像的归因及干预评价固定，Value 随输入变化。左侧 NNS 对 f 沿 x₀→x 计算输入空间积分梯度，RGB 通道带符号求和后得到像素图，再汇总到 4×4-pixel cells。右侧基线直接计算 c_j=A(x)_{0j}uᵀ[V_j(x)−V_j(x₀)]，保留 196 个 patch-token 标量并将 CLS 标量单列；它结合 attention、Value 变化和目标方向，不是 raw attention。均匀展开给每 cell c_j/16。x₀ 是 normalized-zero / RGB 0.5 灰色参考。这里固定 A 的操作属于解释目标定义，不代表正常 ViT 训练时 attention 固定。深层 token 可包含其他空间位置的信息，token j 的贡献放回输入 patch j 是空间映射约定。图片路径和贡献数字仅作方法示意，NNS patch 总和不被强制设为 c_j。

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
