# R10-R2：classifier 输出单元对比图

R10-R2 继承已封存 R10-R1 Figure 1–34，新增 Figure 35，对同一冻结图像的四个 classifier 标量目标作并列比较。这里的对象是 ImageNet-1k classifier 输出单元，不称为 attention head。

## 冻结选择规则

- 输入：`ILSVRC2012_val_00007922.JPEG`。官方验证标签文件第 7922 行是 `n02106662`；该 WNID 对应模型索引 235，German shepherd。
- 正确且预测最高的输出：class 235，clean logit 8.082624。
- 最强竞争输出：排除 class 235 后 clean logit 最大者，即 class 174，Norwegian elkhound，clean logit 7.632706。
- 判别 margin：`logit[235] - logit[174]`，clean margin 0.449918。
- 低分输出：1000 个 clean logits 的全局 `argmin`，即 class 171，Italian greyhound，clean logit −2.921034。该选择只由数值排序确定，不依据 attribution 画面。

## 计算与验收

- 所有目标使用同一冻结 ViT-B/16、同一规范化零参考、同一直线路径和 CPU float64 自动微分。
- 继承 R10-R1 已验证的 class-235 2048/4096 阶地图；新计算 class 174 和 171 的 2048/4096 阶 Gauss–Legendre IG。margin 地图在每个共同阶数严格定义为 `L_235 - L_174`。
- 三个类别输出各自要求相邻阶地图 L1 相对变化小于 0.005，且 completeness 绝对误差小于 `1e-4 + 0.001 * abs(logit delta)`。
- margin 是两项约 8 的路径变化之差，采用预先声明的 cancellation-aware completeness 阈值 `1e-4 + 0.001 * (abs(delta_235) + abs(delta_174))`；同时要求 margin map 与两类别地图之差逐元素完全一致。
- 若 4096 阶未通过，继续到 8192 阶；不得用后处理缩放修补 completeness。

## Figure 35

图版为原图加四张 signed 224×224 learning image：正确类、最强竞争类、二者 margin、最低分类别。四张图使用一个共同 max-abs signed-log 色标，使空间符号和相对幅度可直接比较；每列同时标注 clean logit、reference logit 和沿路径的变化量。

这仍是单张冻结图像的对照展示。它可以说明不同 classifier 输出单元对应不同输入贡献结构，但不能据此声称模型在总体上如何区分类别。
