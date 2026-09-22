# NNS-CST N2-R1C 最终生产器恢复修订

日期：2026-09-22  
状态：`FROZEN_BEFORE_R1C_PAIR_GENERATION_AND_MODEL_OUTCOME_ACCESS`

R1B 的 Telea inpainting 改善了 R1 的放射状填充伪影，但盲化质检仍发现若干大 donor foreground 的可见移除痕迹。R1B 已封存为 `REJECTED_PRE_MODEL_OUTCOME_ARTIFACT_QC`，没有 checkpoint 反序列化和模型 outcome。

R1C 是最后一次 producer-only 恢复。只把 AQUATIC_CONTEXT donor roster 改为盲审中画面以水域为主且最终 foreground 占比较小的八张 pilot 图；Telea 算法、recipient、NON_AQUATIC_CONTEXT donor、语义轴、四臂、matching、行为门、层、rank、readout、NNS、JVP、剂量和阈值均不变。

若 R1C 的八组图仍有会携带 arm identity 的明显 donor-removal artifact，N2 直接终止为 `NO_GO_NOT_IDENTIFIABLE`，不得继续更换 donor、调 producer 或读取模型 outcome。
