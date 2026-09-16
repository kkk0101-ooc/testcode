# S6-R2 presentation revision 图片分析

这次改图参照 Understanding neural network through neuron level visualization 的多面板叙事方式：用统一的行列结构展示结果，把统计量放在小型标注中，不再让审计门槛占据主视觉。S6-R1 的数值、决策和 terminal 均未修改。

## Image #1 — Evidence atlas

四个端点使用相同的小倍图语法：淡色分布展示图像级异质性，顶部黑线是 ordinary BCa 95% CI，彩色点是 equal-class estimate，虚线是 S5 门槛。它比原图同时呈现总体效应、区间和样本分布，避免大面积空白。

## Image #2 — Depth progression

采用参考文章常见的双面板层级曲线。左图是 NNS corrected alignment，右图是 probe gain；两者保留独立纵轴和不同线型，不暗示数值可直接互换。

## Image #3 — D1/D2 relation

保留逐图散点，因为近零相关本身是关键结论。版式统一为轻量网格和内嵌统计卡。

## Image #4 — P2 decision panels

三项标准拆成三个独立面板，各自拥有正确量纲和轴范围。蓝点为观测值，橙虚线为原规则，绿点线为 S5 exploratory rule。三个面板都直接标示 Original: FAIL 与 S5: PASS，避免原图把不同量纲强放在同一横轴。
