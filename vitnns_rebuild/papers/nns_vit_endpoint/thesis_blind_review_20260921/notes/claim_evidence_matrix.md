# 盲审稿核心结论—证据矩阵

本表将盲审稿中的核心结论绑定到公式、实验、统计量、正文表格和封存源。路径均相对于
vitnns_rebuild/。

| 结论 | 公式或定义 | 实验与统计单位 | 正文表格 | 封存证据 | 结论边界 |
|---|---|---|---|---|---|
| 标量 NNS 学习图像等于同目标输入乘梯度 | 式（3-1）至式（3-2），命题 3-1 | 5 类网络、每类 1 000 次；网络—输入比较 | 表 3-1 | reports/nns_vit_endpoint_m3_r1_20260919/m3a_fidelity.json | 精确当前点局部导数且无额外重分配；不表示函数值完整性 |
| 查询/键/值、多头和残差在合法切面上闭合 | 式（4-2）至式（4-8），命题 4-1 | 400 图/模型；76 800 正向项、28 800 负控制、6 144 解析方向；图像—块—路径 | 表 5-2，正文 4.7 | reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json；reports/nns_vit_endpoint_e4_intervention_r3_20260918/terminal_receipt.json | 指定冻结图、真实边和切面；不证明有限效应 |
| 结构分项不具有一般实现不变性 | 式（4-9）至式（4-11） | 等价线性重参数化构造 | 正文 4.5 | 解析反例；papers/nns_vit_endpoint/notes/revision_v3_frozen_claims.md C3 | 总图不变，内部坐标分项改变 |
| 五组独立抽样复现空间扩展 | 式（4-12）；后期减前期 | 100 图/模型、5 个独立抽样组；图像 | 表 6-1 | papers/nns_vit_endpoint/publication_r2_20260920/evidence/revision_v3_step2/independent_sampling_summary.csv；independent_sampling_variation.json | 两个固定参数状态和冻结目标抽样；不是全部单元普遍性 |
| 查询、键和值具有不同总体层深轨迹 | 式（4-12）至式（4-14）；第 11 块减第 0 块 | 400 图/模型；图像 | 表 6-2 | papers/nns_vit_endpoint/publication_r2_20260920/evidence/revision_v3_step2/qkv_depth_contrasts.csv | 查询空间方向跨模型不一致；不概括为共同扩散或收缩 |
| 深层路由相对尺度和仅值路径误差提高 | 式（4-14）至式（4-15） | 400 图/模型；图像 | 表 6-3 | reports/nns_vit_endpoint_m4_r1_20260919/m4_tests.json；attention_block_summary.csv | 描述局部路径尺度，不是前向贡献百分比 |
| 预训练与随机参数状态存在差异 | 式（4-12）至式（4-15）；预训练减随机层深差 | 32 共享图/模型、3 个随机状态；图像和种子 | 表 5-2，正文 6.6 | papers/nns_vit_endpoint/publication_r2_20260920/evidence/supplement_v1/training_analysis.json | 不识别训练过程的因果效应 |
| 规定头输出插值中的一阶误差随幅度扩大 | 式（5-1）至式（5-2） | 400 图/模型、2 参照状态、5 幅度；图像 | 表 7-1、表 7-2 | papers/nns_vit_endpoint/publication_r2_20260920/evidence/revision_v3_step2/multi_amplitude_summary.csv；multi_amplitude_decision.json | 规定插值方向的经验边界；不推广到任意方向 |
| 局部路径图与完整成分效应广泛但非普遍关联 | 式（5-3）；成分 Spearman、捕获率、遗憾值 | 400 图/模型；图像 | 表 5-2，正文 7.4 | reports/nns_vit_endpoint_e6_release_r3_20260918/family_decisions.json；papers/nns_vit_endpoint/publication_r2_20260920/evidence/supplement_v1/m2_inherited.json | 11/12 成分族满足原条件；联合条件未满足 |
| 区域泰勒在规定区域替换中优于原输入局部图 | 式（5-3）；绝对效应曲线下面积与归一化遗憾值 | 400 图/模型；图像 | 表 7-3 | papers/nns_vit_endpoint/publication_r2_20260920/evidence/analysis.json；spatial_image_values.csv | 两种参照、规定块/头/区域预算；不是无条件方法排名 |

第二步后继关系：第一步曾因多幅度实验尚未执行而冻结较窄 RQ4；第二步封存证据完成后，
本盲审稿只对式（5-1）规定的头输出插值恢复“一阶误差随幅度扩大”结论。原完整区域替换的
未支持结论保持不变。
