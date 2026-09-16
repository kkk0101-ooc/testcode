# S6 post-hoc exploratory 图片分析

## Image #1 — Endpoint decisions [post-hoc exploratory]

图中展示 P1、P2、D1、D2 的点估计与 ordinary BCa 95% CI。虚线是各端点在 S5 中采用的探索性门槛。四个端点均为 SUPPORTED_POST_HOC_EXPLORATORY。P2 的虚线是 outcome access 后设置的 0.07 门槛，不是原 0.10 规则。

## Image #2 — Depth profiles [post-hoc exploratory]

左图显示 NNS 的位置对照校正前景对齐随 block 的变化；右图显示 feature probe 相对于低层控制的 MSE gain。两个深浅对比均为正，但曲线代表不同构念，纵轴不应互换解释。

## Image #3 — D1/D2 relationship [post-hoc exploratory]

共同有效的 233 张图像形成散点。Pearson r=0.0016，显示两种深度差异没有逐图线性耦合证据。该图不支持 probe 显著等同于 NNS 显著，也不支持因果形成机制。

## Image #4 — P2 rule comparison [post-hoc exploratory]

图中并列标示观测值、原规则门槛和 S5 探索规则门槛。P2 通过 S5 的探索规则，但原规则的三个条件均未通过，因此正式状态保持 ORIGINAL_RULE_NOT_PASSED。
