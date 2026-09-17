# 6. 局限

第一，主干预结论具有明确的目标范围。Stage D 解释冻结 ViT-B/16 的 block 0/head 9 fixed-clean-attention Value-path，并在 4x4-cell endpoint 上比较同目标 patch-token 基线。它不是完整类别解释、物体定位指标或人类视觉合理性检验。Native-patch 和其他层头结果显示，该效应不能推广为 ViT 全网络的均匀属性。

第二，空间稳定性仍是未解决问题。C10 的 translation 与 center-crop/resize gates 失败，Stage X 的广义稳定性和相对 Grad-CAM 的 noninferiority 也未通过。多尺度聚合提高相关性是一项事后探索观察；它不能替代细粒度稳定性，也不能把 broad stability 决策改为通过。

第三，不同解释器并非完全同目标。50 图比较中的条件 NNS、full-model IG、Grad-CAM、rollout 和 raw CLS attention 使用不同的类别条件或计算路径。正式 contrast 仍能比较冻结 fidelity 指标，但不能据此建立与目标无关的全序排名。未显著的 NNS-full IG 和 NNS-raw attention 对比也不证明等价。

第四，单图模块展示的外推范围有限。Figures 28-35 在一张冻结图像上验证输入坐标、completeness、residual additivity 和 classifier target 差异。固定通道不代表总体语义，四个 classifier 输出也不代表类别间的总体空间机制。

第五，部分后续证据是 descriptive 或 post-hoc exploratory。Stage E 在 D10 之后执行，且 E4 scientific validity 为 `MIXED`；Stage X2-MECH 和 Stage S 不具有确认性地位。Stage S 的 P2 阈值在 outcome access 后确定，原始规则未通过。它们适合提出后续假设，不应提升摘要中的主要因果或确认性结论。

第六，方法新颖性仍需文献级技术审查。Learning image 的源思想来自 Dou et al.，路径积分来自 IG，内部成分计算与 conductance/path attribution 直接相关。本文已经验证 ViT 结构映射和实证用途，但投稿前仍需完成逐项操作对照和最终贡献声明。

最后，当前实验集中在一个冻结 ViT-B/16 和 ImageNet-1k 验证图像上。不同模型规模、训练目标、patch 大小、reference、数据域和视觉任务是否保持相同结论，需要独立研究。
