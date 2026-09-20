# NNS 学习图像的计算图解释与视觉 Transformer 结构路径分析

这个目录包含可独立阅读和复制的论文版本：11 张正文图、7 张正文表、22 张完整渲染图件、双语补充材料，以及 92 页历史补充图。

- [中文论文](zh/manuscript.md)
- [English manuscript](en/manuscript.md)
- [图片总览](gallery.html)
- [中文补充材料](zh/supplement.md) / [English supplement](en/supplement.md)
- [补充实验 v1](zh/supplement_experiments_v1.md) / [Supplementary experiments v1](en/supplement_experiments_v1.md)
- [图件清单](figure_manifest.json)
- [逐图阅读说明](figure_analysis.md)
- [逐项修改验收](qc/completion_audit.md)
- [修改意见 v2 完成审计](qc/revision_v2_completion_audit.md)
- [补充实验 v1 完成审计](qc/supplement_v1_completion_audit.md)

figures/png、figures/pdf、figures/svg 保存正文图的三种格式。supplement 保存所有补充图的本地副本；正文和补充材料的图片链接均位于这个目录内。assets 保存本轮真实展示计算的数组，evidence 保存来源、显示规则和计算记录。作者及单位信息仍由作者填写。

修改意见 v2 已落实到当前双语稿：题名与总论点转向 NNS 的一般计算图解释，明确标量学习图像与 Input × Gradient 的等价条件，以四个 RQ 组织 Jacobian 身份、ViT 结构分解、层深规律和有限干预边界。Attention 统一为 Q/K/V-induced differential，residual 使用输出处共享 cotangent；合法计算切面、空间指标、绝对路径范数与纠正后的随机基线均在正文或补充材料中给出。

完整图件库沿用原 publication-R2 编号：1–7 讲解学习图像与 ViT 分解；8–11 展示内部目标、逐层学习图像及实际前向特征；12–18 结合总体空间轨迹和路径示例；19–20 比较实际计算的 Endpoint、Grad-CAM、attention rollout 与 raw attention；21–22 说明有限干预与 signed/positive 显示的关系。正文通过 figure_manifest.json 的 main_text_figure_map 重新连续编号为 Figures 1–11。

Figures 2–5 和 7 进一步采用参考论文的概念图视觉语法：虚线步骤分区、计算节点与 learning image 同色对应、正常计算在上/输入坐标图在下、关键合成路径以红色虚线强调。图中公式和真实 ViT 路径仍使用本文定义。

对比使用相同输入与冻结模型；Endpoint/Grad-CAM 使用同一 GT logit，attention 方法为 class-agnostic。Figure 20 的狒狒、听诊器和海上钻井平台三例按预先封存的 Endpoint/输入细节可读性规则选定，比较方法未参与排序。图片说明表示对象和空间形态，不用于宣称方法普遍优越。原群体统计与 C4a/C4b 判定保留在补充材料。

补充实验 v1 新增 456 个任务：两模型各三种随机初始化的 32 图 M3/M4 对照、两模型各 100 图的 GT/top1/margin 分解，以及两模型各 32 图的 feature–Endpoint 定量分析。随机区域基线按每次重复先取 absolute effect 再计算 AUC；正文 Tables 5–7 已使用修正后的值。五张新增图和可重算的图像/单元级数据位于 `evidence/supplement_v1/`。

复现入口位于上级 scripts/：compute_publication_r2.py、render_publication_r2.py、verify_publication_r2_methods.py、validate_publication_r2.py。write_publication_r2.py 是旧 v1 生成器，默认拒绝覆盖当前手工审校后的 v2 稿件。实际阅读和图件复制只需本目录。
补充实验入口为 `prepare_endpoint_supplement_v1.py`、`run_endpoint_supplement_v1.py`、`analyze_endpoint_supplement_*.py`、`render_endpoint_supplement_v1.py`、`write_endpoint_supplement_v1.py` 与 `validate_endpoint_supplement_v1.py`。
