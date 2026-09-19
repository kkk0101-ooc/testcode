# 修订主图图注

**Figure 1.** 当前输入的 ground-truth class logit 与两种互补学习图像。Signed 图对 RGB 求和，absolute 图先逐通道取绝对值再求和；正负项都保留在分析对象中。输入为冻结展示样本 class 297（懒熊）。

**Figure 2.** 异构标量目录与 attention/residual differential cuts。Head edge 位于 output projection 之前，cotangent 包含该投影；residual identity 使用输出边 cotangent。

**Figure 3.** 五类兼容网络中的原始 NNS 数值等价，以及结构闭合、负例与独立方向检查。左图每点为对应网络/精度的最大 relative-L2 error；验证项计数不是独立统计样本量。

**Figure 4.** 冻结分层 token-channel 的 entropy、locality、distance 层深轨迹。蓝色/橙色为 ViT-B/16、DeiT-S/16；阴影为 100 张图的图像 bootstrap 95% CI。

**Figure 5.** 上排为 CLS-minus-patch entropy 与 net/absolute mass 的总体轨迹。下排为冻结指标选出的高 map-cancellation 单元；signed 和 absolute 视图并列，原始 RGB 质量与范数标于图下。

**Figure 6.** 400 图的 routing scale、Value-only Endpoint error 与 JVP discrepancy；阴影为图像 bootstrap 95% CI。下方同一 head 的 routing/value/full maps 在 signed 行及 absolute 行内分别共用尺度，含原始质量、范数和 signed sum。案例只说明分解，不代替总体证据。

**Figure 7.** Attention/MLP identity scale 与 output–identity patch TV；下方高 path-cancellation 案例的 identity/branch/output 共用组内尺度。TV 曲线为冻结 cohort 描述；late-minus-early 推断来自单独的 post-hoc sensitivity。

**Figure 8.** 按冻结指标选出的同图跨模型深度比较。每模型早晚层共用 signed/absolute 各自尺度；所选 channels 独立注册，不构成跨层或跨模型语义匹配。

**Figure 9.** 左图为原 C4a family 点估计及 95% 图像 bootstrap CI，虚线 0.10 是点估计门槛，方形为未通过 family。右图为补充冻结同层 head 排序的 Endpoint/Taylor Spearman；其检验不替代原 C4a。

**Figure 10.** 四个注册组合的 absolute-effect AUC 中位数，包含实际评价候选内的 oracle。下方同一懒熊图像、DeiT-S block 5 head 0 与 normalized-zero reference 的 signed/absolute Endpoint 和 regional Taylor。Taylor 的 16 格表示区域位移预测分数，不能与像素 Endpoint 色值比较。
