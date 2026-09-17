# R10-R4 · 35 张论文图

[画廊](gallery.html) · [图表索引](figure_index.md) · [论文大纲](../../docs/论文大纲_NNS_ViT_R10_20260917.md)

## Figure 1

NNS 扫描 ViT 的内部响应。左侧将同一张输入图像分成 patch tokens，经过已训练的 ViT；右侧按层（B0、B5、B11）和 head（H0、H5、H11）排列实测 NNS 正贡献图。每格表示输入位置对该坐标 CLS head-direction Value-path 标量的正贡献。层和 head 采用首、中、末位置作展示，不按效果排名。底色到亮色表示该坐标显示尺度下的正贡献。示例为先前冻结的 order 44 中位效应样本。网格标记输入 patch 位置；这些地图的查询目标是 CLS，不是各 patch-query 的独立归因。Figure 2 解释对应关系，Figure 8–9 展示全部 head 的汇总。

## Figure 2

真实输入经过 ViT 的 patch embedding 和 B0 attention 后，选定 Figure 30 的 channel 255 在 196 个 patch-token 输出上的总和作为内部响应 s。右侧只显示正贡献 max(Ls,0)，采用与 Figure 30 channel 255 上排完全一致的蓝底 heat palette、逐面板 P99 缩放和 nearest。深蓝表示零或低正贡献，亮色表示较高正贡献。底部 full signed map sum 仍由未裁剪 signed 原始数组计算。当前目标的完整 signed 对照见 supplement/figure_30_signed 的 channel 255；旧 Figure 2 的 channel-0 signed 图仅作为历史备份。输入与 channel 255 均在计算前固定；本图仍是模块自身响应的单样本展示。

## Figure 3

统一空间尺度上的排序与忠实度评价。224×224 图像包含 14×14=196 个原生 patch，每个 16×16-pixel patch 再划为 16 个 4×4-pixel cells，全图 3,136 个 fine cells。NNS 对每 cell 的 signed 输入贡献求和后按目标方向取正部排序；token 基线先把每个 c_j 均分，再按相同规则排序，因此同一 patch 内同分，不同 patch 之间仍可不同。图中的 1–16 为一个 patch 内的示意局部次序；真实实验在全部 cells 上排序，并对所有同分组使用 5 次方法间配对的 tie permutations。Insertion 从同一灰色参考恢复高分区域，deletion 从原图替换高分区域；两者均评价 原协议 B0/H9 的固定 clean-attention head-direction 标量 f，使用相同干预面积预算。图中放大的遮挡块只说明操作顺序，不是实际 4×4-pixel mask 或实验结果。原协议使用 21 fractions、实际干预比例上的 AUC，响应不截断到 [0,1]。每图先平均 5 次 AUC，计算 F=½(AUC_ins−AUC_del) 及 ΔFID=F_NNS−F_token；随后以图像为独立单位报告 median 和 paired bootstrap 95% CI。 本图干预仍针对原协议 B0/H9 的固定 clean-attention Value-path 标量；R9 Figure 2 改用真实模块响应演示通用定义，两者目标分别注明。

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

## Figure 15

把网络里的一个数和一张输入位置图配起来。上面是内部响应的变化量，下面说明这次变化来自输入的哪几个位置。四张图始终用同一套位置坐标。这张采用可手算的线性示例，不是 ViT 实测结果。

## Figure 16

先用原图减去参考图，得到这次需要解释的输入变化。然后拆成四张小图，每张只保留一个位置，其余位置为零；四张相加正好恢复输入差值。真实 RGB 输入对应每个像素的三个通道。实现用自动微分计算这些位置的贡献，不必真的存下十几万张稀疏图。

## Figure 17

沿用上一图的输入差值。单元 a 的图是 [1,2;0,0]，单元 b 的图是 [0,0;6,-4]。输出用 2a+0.5b，因此输出图是 [2,4;3,-2]，总和为 7。你可以看见数值被重新加权，但原来的位置始终没丢。ViT 的系数还随输入变化，所以实际方法要沿参考到原图的路径累计。

## Figure 18

Patch embedding 可以先按线性连接理解；位置编码和偏置仍保留在模型里。到了 attention，输入既会改变注意力权重 A，也会改变 Value，完整方法需要两部分导数。最后 residual 旁路也要算，才能把贡献账算完整。旧的固定 attention 图只研究其中一个条件视角。

## Figure 19

同一类别目标下，六个 block 的 attention 分支正输入贡献与对应 token RMS。上排为既有 56×56 input-cell 成分图，下排为 196 个 patch-token 输出跨通道的 RMS。输入固定为 Y2 队列首图、类别目标 235；采用逐面板 P99 与 nearest。完整 signed/负幅度图保留于本包 supplement。上排解释最终类别的分支成分；Figure 28–31 解释模块自身响应。

## Figure 20

固定同一输入和最终类别，比较不同 head 分到的输入贡献。每个小图是一个 head 的成分 learning image，这比换不同 head 自身的响应目标更适合看共同任务下的差异。图中保留 H0/H5/H11、B0/B5/B11 这些固定位置，完整逐 block 12-head 图另附。颜色逐面板缩放，亮度不用于排名。

## Figure 21

同一个 block 的 residual 旁路图，加上全部 12 个 head 的图，会恢复总输入贡献图。这里先对原始正负数求和，再着色；不是把几张颜色图片叠起来。它展示 learning image 怎样组合成更大组件，同时能检查是否漏掉某条路径。

## Figure 22

两个 token 的解析 attention 例子。固定选第一个 token 作 query，输入 [1,3]、参考 [0,0]，Q/K/V 标量投影权重均为 1，d_k=1。先算 qK、softmax 和输出，再将输入导数分为 A dV 与 (dA)V 两项。沿直线路径 [α,3α] 采用 Gauss–Legendre 数值积分，得到两个输入位置的完整归因。冻结 clean attention 时的图是另一个条件函数的归因，不等于完整路径中 A dV 那一项。本图是解析示意，不是已训练 ViT 的实测结果。

## Figure 23

解析 residual 例子。输入 [2,1]、参考 [0,0]，旁路 r=x₁，分支 b=2x₁−x₂，相加 z=r+b，最终目标 F=z²。沿路径 z=5α，下游导数为 10α。分别经过旁路和分支的输入贡献为 [10,0] 和 [20,−5]，和为 [30,−5]，总和 25 恢复目标差值。分支对 F 的贡献包括下游导数，因此不同于它自身响应的 [4,−1]。两格固定表示输入位置 1 和 2。

## Figure 24

同一小型解析链的总览：输入 [1,1;3,3] 依次经过线性 patch embedding、完整 softmax attention 和 residual。上排是正常响应，下排是同名响应 learning image，虚线逐一对应。 原始四图全部原样纳入 R9。这是算子示意，不是训练后 ViT 实测；真实输出见 Figure 28–31。

## Figure 25

四个单坐标输入图通过 [0.5,0.5] 权重形成两张 token 学习图。图值相加分别得到响应 1 和 3，输入位置保留。 原始四图全部原样纳入 R9。这是算子示意，不是训练后 ViT 实测；真实输出见 Figure 28–31。

## Figure 26

正常值使用 A 组合 V；本例输入 patch 映射为线性，learning image 可用沿路径平均的完整 Jacobian 系数 C 组合。C 含 Q/K/V 路径，不等于 clean attention。 原始四图全部原样纳入 R9。这是算子示意，不是训练后 ViT 实测；真实输出见 Figure 28–31。

## Figure 27

同一参考和路径下，residual 的响应图逐位置相加，L(z)=L(h)+L(o)。相加发生在原始值上，着色在之后。 原始四图全部原样纳入 R9。这是算子示意，不是训练后 ViT 实测；真实输出见 Figure 28–31。

## Figure 28

同一通道 0 连续经过三个模块。固定输入 ILSVRC2012_val_00007922.JPEG，规范化灰色参考为零；patch 输出与 B0 attention、B0 第一次 residual 相加前后的真实响应。每个目标为指定通道在 196 个 patch-token 输出上的总和，CLS 参与前向但不计入这项空间聚合。上排为该响应的输入 IG 正贡献，224×224，RGB 先带符号求和；下排为相同通道的原始 signed token 值，排为 14×14。上排各面板正部 P99（零则 max）缩放，下排各面板 max-abs signed-log(kappa=9)，均用 nearest。完整 signed learning image 与 RGB 原始数组在补充目录。该单样本图不度量总体层间趋势或 feature-map 相关性。

## Figure 29

Patch embedding 的固定通道 0、255、511、767。固定输入 ILSVRC2012_val_00007922.JPEG，规范化灰色参考为零；patch 输出与 B0 attention、B0 第一次 residual 相加前后的真实响应。每个目标为指定通道在 196 个 patch-token 输出上的总和，CLS 参与前向但不计入这项空间聚合。上排为该响应的输入 IG 正贡献，224×224，RGB 先带符号求和；下排为相同通道的原始 signed token 值，排为 14×14。上排各面板正部 P99（零则 max）缩放，下排各面板 max-abs signed-log(kappa=9)，均用 nearest。完整 signed learning image 与 RGB 原始数组在补充目录。该单样本图不度量总体层间趋势或 feature-map 相关性。

## Figure 30

Attention · B0 的固定通道 0、255、511、767。固定输入 ILSVRC2012_val_00007922.JPEG，规范化灰色参考为零；patch 输出与 B0 attention、B0 第一次 residual 相加前后的真实响应。每个目标为指定通道在 196 个 patch-token 输出上的总和，CLS 参与前向但不计入这项空间聚合。上排为该响应的输入 IG 正贡献，224×224，RGB 先带符号求和；下排为相同通道的原始 signed token 值，排为 14×14。上排各面板正部 P99（零则 max）缩放，下排各面板 max-abs signed-log(kappa=9)，均用 nearest。完整 signed learning image 与 RGB 原始数组在补充目录。该单样本图不度量总体层间趋势或 feature-map 相关性。

## Figure 31

Residual add · B0 的固定通道 0、255、511、767。固定输入 ILSVRC2012_val_00007922.JPEG，规范化灰色参考为零；patch 输出与 B0 attention、B0 第一次 residual 相加前后的真实响应。每个目标为指定通道在 196 个 patch-token 输出上的总和，CLS 参与前向但不计入这项空间聚合。上排为该响应的输入 IG 正贡献，224×224，RGB 先带符号求和；下排为相同通道的原始 signed token 值，排为 14×14。上排各面板正部 P99（零则 max）缩放，下排各面板 max-abs signed-log(kappa=9)，均用 nearest。完整 signed learning image 与 RGB 原始数组在补充目录。该单样本图不度量总体层间趋势或 feature-map 相关性。

## Figure 32

B0 MLP branch 的真实响应 learning image。固定通道 0、255、511、767，每个目标为该通道在 196 个 patch-token MLP 输出上的总和。上排为 224×224 输入贡献正部，下排为同一通道的 14×14 signed token 值。真实前向包含 LN2、fc1、GELU、fc2、layer scale 与 drop path；完整 signed 图见 supplement。

## Figure 33

B0 第二次 residual add 后的完整 block output。固定通道和版式与 Figure 32 相同。原始 RGB learning image 逐像素满足 L(second residual)=L(first residual)+L(MLP branch)，最大误差为 3.33×10⁻¹⁶；着色只在加法核对之后进行。

## Figure 34

真实预测类别 235 的完整 logit learning image。梯度经过全部 12 个 blocks、final norm、CLS 选择与 Linear(768,1000)，相对灰色参考积分回 224×224 输入坐标。clean logit 为 8.082624，参考 logit 为 −0.384007。类别标量没有 14×14 token 输出，因此本图不套用模块通道页。

## Figure 35

同一冻结输入的四个 classifier 标量 learning image。类别 235 是官方真值 German shepherd 且为 clean argmax；类别 174 Norwegian elkhound 是最高竞争输出；第三列为逐像素 L235−L174 的判别 margin；类别 171 Italian greyhound 是 1000 个 clean logits 的全局最低分。四图采用共同 max-abs signed-log 色标。低分类别按 logit 排序预先确定，不按归因图观感挑选。
