# 从 NNS 到视觉 Transformer：内部目标的可审计 Jacobian 扫描与结构分解

这个目录包含可独立阅读和复制的论文版本：11 张正文图、10 张编号正文表、2 张方法合同表、22 张完整渲染图件、双语补充材料，以及 92 页历史补充图。

- [中文论文](zh/manuscript.md)
- [English manuscript](en/manuscript.md)
- [图片总览](gallery.html)
- [中文补充材料](zh/supplement.md) / [English supplement](en/supplement.md)
- [补充实验 v1](zh/supplement_experiments_v1.md) / [Supplementary experiments v1](en/supplement_experiments_v1.md)
- [修改意见 v3 第一步冻结主张](../notes/revision_v3_frozen_claims.md)
- [修改意见 v3 第一步完成审计](qc/revision_v3_step1_completion_audit.md)
- [修改意见 v3 第二步可信度实验](zh/revision_v3_step2_experiments.md) / [Credibility experiments](en/revision_v3_step2_experiments.md)
- [修改意见 v3 第二步后继主张](../notes/revision_v3_step2_claims.md)
- [G×I 五步计划第三步专项审计](qc/gxi_step3_completion_audit.md)
- [G×I 五步计划第四步专项审计](qc/gxi_step4_completion_audit.md)
- [第四步逐主张证据索引](evidence/gxi_step4_evidence_index.md)
- [第四步主文与补充材料分配表](evidence/gxi_step4_figure_allocation.md)
- [Endpoint 与 Gradient × Input 等价下的创新定位及五步计划](../../../Endpoint_GxI创新定位与后续五步计划.md)（第八至十一节为第一至第四步执行记录）
- [图件清单](figure_manifest.json)
- [逐图阅读说明](figure_analysis.md)
- [逐项修改验收](qc/completion_audit.md)
- [修改意见 v2 完成审计](qc/revision_v2_completion_audit.md)
- [补充实验 v1 完成审计](qc/supplement_v1_completion_audit.md)

figures/png、figures/pdf、figures/svg 保存正文图的三种格式。supplement 保存所有补充图的本地副本；正文和补充材料的图片链接均位于这个目录内。assets 保存本轮真实展示计算的数组，evidence 保存来源、显示规则和计算记录。作者及单位信息仍由作者填写。

修改意见 v3 第一步已落实到当前双语稿：核心贡献限定为 NNS 在指定 ViT 计算图上的结构化实现、图相对路径分解和系统验证；相关工作及方法定位表覆盖 Gradient × Input、conductance、FullGrad、GlobEnc、DecompX、AttnLRP、LibraGrad、activation patching 和 edge attribution。Section 4.4 给出等价重参数化反例并明确分解不具有一般实现不变性。RQ4 收窄为规定有限替换任务中的成分与区域排序，训练结果表述为训练参数状态与随机参数状态的差异。冻结主张与证据边界见 `../notes/revision_v3_frozen_claims.md`。

2026-09-22 的定位冻结进一步把全文统一为一个中心论点：以 Gradient × Input 为局部测量原语，构建面向 ViT 异构内部目标的结构化、可闭合、可审计扫描框架，并研究局部输入依赖如何随层深、attention 路径和 residual 分支重新组织。现役双语题目、摘要中心句、三项贡献、讨论和结论已同步；当前五类 claim registry、禁用扩展及 PCIA–PEFA/CST 隔离边界位于五步计划第八节。

完整图件库沿用原 publication-R2 编号：1–7 讲解学习图像与 ViT 分解；8–11 展示内部目标、逐层学习图像及实际前向特征；12–18 结合总体空间轨迹和路径示例；19–20 比较实际计算的 Endpoint、Grad-CAM、attention rollout 与 raw attention；21–22 说明有限干预与 signed/positive 显示的关系。正文通过 figure_manifest.json 的 main_text_figure_map 重新连续编号为 Figures 1–11。

Figures 2–5 和 7 进一步采用参考论文的概念图视觉语法：虚线步骤分区、计算节点与 learning image 同色对应、正常计算在上/输入坐标图在下、关键合成路径以红色虚线强调。图中公式和真实 ViT 路径仍使用本文定义。

第三步已重构现役方法章节和前两张核心概念图。Figure 1 按“总 Gradient × Input → $[e,i]$ 双索引展开 → 合法切面重构 → 层深机制问题”组织；Figure 2 绑定 consumer-use 四元组、attention/head/residual 切面与已封存漏项/重复审计。方法正文另含索引退化表和统一审计协议表。

第四步已按“身份与正确性—背景规律—联合索引机制结果—跨 checkpoint 复现—行为标定”重排 Results。新增 Table 4 区分总图、edge scalar 与双索引图组的可恢复信息；attention Full–Value 和 residual output–identity spatial TV 分别作为两项联合索引必要性结果。E5 在通过来源、零重叠 cohort、四 checkpoint、统计合同和原始数字重算后进入现役双语稿；H1–H3 正向复现与 H4 四 checkpoint 反向结果均完整保留。

对比使用相同输入与冻结模型；Endpoint/Grad-CAM 使用同一 GT logit，attention 方法为 class-agnostic。Figure 20 的狒狒、听诊器和海上钻井平台三例按预先封存的 Endpoint/输入细节可读性规则选定，比较方法未参与排序。图片说明表示对象和空间形态，不用于宣称方法普遍优越。原群体统计与 C4a/C4b 判定保留在补充材料。

修改意见 v3 第二步新增 1000 个 image-model 任务：800 个任务完成五幅度 head-output replacement 和 Q/K/V 全 block 统计，200 个任务各执行 5 组独立 channel/token 抽样。双语结果报告位于 `zh/en/revision_v3_step2_experiments.md`，汇总数据、收据和三张结果图位于 `evidence/revision_v3_step2/`。替代训练 checkpoint 与同目标机制桥接仍不在当前证据范围。

复现入口位于上级 scripts/：compute_publication_r2.py、render_publication_r2.py、verify_publication_r2_methods.py、validate_publication_r2.py；v3 第一步、第二步及 G×I 第三、第四步专项审计分别为 validate_revision_v3_step1.py、validate_revision_v3_step2.py、validate_gxi_step3.py 和 validate_gxi_step4.py。write_publication_r2.py 是旧 v1 生成器，默认拒绝覆盖当前手工审校稿。实际阅读和图件复制只需本目录。
补充实验入口为 `prepare_endpoint_supplement_v1.py`、`run_endpoint_supplement_v1.py`、`analyze_endpoint_supplement_*.py`、`render_endpoint_supplement_v1.py`、`write_endpoint_supplement_v1.py` 与 `validate_endpoint_supplement_v1.py`。
