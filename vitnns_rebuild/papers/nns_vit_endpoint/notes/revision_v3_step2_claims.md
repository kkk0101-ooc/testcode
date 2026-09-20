# 修改意见 v3 第二步：后继主张清单

冻结日期：2026-09-21。前序 `revision_v3_frozen_claims.md` 保留为第一步快照；本文件只增加第二步三项实验能够直接支持的主张。

| Claim | 支持表述 | 直接证据 | 边界 |
| --- | --- | --- | --- |
| S2-C1 多幅度一阶边界 | 在两个固定 checkpoint、400 图/模型、全部 12 层 attention heads 和两种 reference 下，Taylor 绝对误差随 $\alpha=0.05,0.1,0.25,0.5,1.0$ 单调增大；1600/1600 条图像–reference 误差曲线单调。排序相关随幅度降低。 | `publication_r2_20260920/evidence/revision_v3_step2/multi_amplitude_summary.csv` 与 `multi_amplitude_decision.json` | 支持规定 head-output 插值替换中的局部一阶有效范围，不外推到其他变量、路径或 reference。 |
| S2-C2 Q/K/V 总体差异 | 两个模型的 K、V 路径从 block 0 到 11 均表现为 absolute mass 与 cancellation 增加、entropy 和 normalized pairwise distance 减少；K 相对尺度增加，V 相对尺度减少。Q 的 absolute mass、cancellation 和相对尺度增加，但 entropy 和 distance 的方向不跨模型一致。 | `publication_r2_20260920/evidence/revision_v3_step2/qkv_block_summary.csv` 与 `qkv_depth_contrasts.csv` | 支持 Q/K/V 具有不同的总体层深轨迹；不支持给三条路径赋予统一空间变化方向。 |
| S2-C3 独立抽样复现 | 五组独立 token/channel 抽样在两个模型上均复现 entropy 增加、own-token locality 降低和 normalized distance 增加；30/30 个 model–group–metric 的 late-minus-early 方向一致。 | `publication_r2_20260920/evidence/revision_v3_step2/independent_sampling_summary.csv` 与 `independent_sampling_variation.json` | 结论表述为两个固定 checkpoint 的五组独立抽样目标中稳定复现，不改写为全部内部单元的普遍定律。 |

训练相关表述继续使用第一步冻结边界：训练参数状态与三个随机参数状态存在差异；第二步没有增加训练 checkpoint，因而不引入训练因果效应表述。
