# 统一论文图 R1：中文图注

## Figure 1

研究框架。从输入图像到固定 ViT Value-path attribution，并与匹配 patch-token baseline 在同一细粒度干预协议下比较。

## Figure 2

Head-direction Value-path 构造。NNS 沿固定 clean attention 与指定 head direction 生成 signed input-space map；baseline 保留同图像、同目标与同干预算下的 patch-token contribution。

## Figure 3

评价协议。attribution 在 fine 4×4 cell 或 native 14×14 patch 上排序，随后执行 matched-fraction insertion/deletion，并以图像为统计单位汇总。

## Figure 4

Fine-scale fidelity 主结果。500 个图像的 NNS−matched ΔFID 均为正；paired median 为 2.308，95% bootstrap CI 为 [2.111, 2.445]。

## Figure 5

NNS 与 matched patch 的 median insertion/deletion trajectories。阴影为 pointwise bootstrap interval。

## Figure 6

空间尺度与 upsampling sensitivity。fine 4×4 的 ΔFID 分布与 native 14×14 对照，并给出三类 upsampling evidence classes 的 insertion/deletion AUC 中位数与 IQR。

## Figure 7

四个冻结 layer/head 坐标在 overall、correct 和 misclassified strata 中的 median ΔFID 与 95% image-bootstrap interval。

## Figure 8

冻结案例图。展示 median-effect、weak-effect 和 misclassified 样本的原图、NNS、matched patch 与差异形态；每个 attribution panel 独立缩放。

## Figure 9

一个冻结样本的 head-averaged spatial support。列为四个 depth bands，行为 NNS/matched 的正负 contribution；跨 depth band 不比较绝对幅值。

## Figure 10

解释方法的空间比较。展示 formal cohort 前四个位置的原图、NNS、pooled NNS、Grad-CAM、full-model IG 和 rollout；每个 attribution panel 使用 P99 display scaling。

## Figure 11

解释方法的定量比较。左：50 个 image-level fidelity scores；右上：NNS 相对四个 comparator 的 paired mean difference 与 95% CI；右下：median deletion trajectories。

## Figure 12

Post-hoc exploratory multiscale mechanism analysis。固定 flip route 的示例 map，以及 pixel、4×4、16×16 下的 geometric agreement、ROAD fidelity 和 NNS-v2 SSE decomposition。

## Figure 13

ViT block-level branch allocation。左图为 attention 与 MLP branch 的 12-block absolute-mass allocation shares；右图为 B4/B5 deep-minus-shallow estimates 与 95% CI。

## Figure 14

Post-hoc exploratory depth extension。展示 endpoint estimates、NNS/probe depth progression、matched occlusion responses 和 D1/D2 image-level relation。
