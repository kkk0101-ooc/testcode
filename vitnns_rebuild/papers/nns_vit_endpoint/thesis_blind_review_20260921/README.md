# 视觉 Transformer 的神经元级与结构路径可解释性研究（修改意见 v5）

本目录执行修改意见/修改意见v5.md：完成中英文论文改写、已有 M1/区域/多幅度结果复算，以及 E1–E5 前瞻补充实验。原 C4a/C4b 结论和既有封存证据保持不变。

## 阅读入口

- [中文稿](zh/manuscript.md)
- [英文稿](en/manuscript.md)
- [匿名内容审阅 PDF](NNS_ViT_学位论文_匿名内容审阅版.pdf)
- [v5 执行计划与要求清单](notes/revision_v5_plan.md)
- [主张—证据矩阵](notes/claim_evidence_matrix.md)
- [答辩十问](notes/revision_v5_defense_answers.md)
- [紧凑证据索引](evidence/revision_v5/INDEX.md)
- [完成审计](qc/completion_audit.md)

论文采用四个研究问题，证据顺序为神经元表征、结构贡献、行为干预和外部验证。补充实验共使用 1400 张与原队列互斥的新图像，其中正式 E1-R2 与前置 E1 诊断也互不重叠；E5 在两个原检查点和两个新增公开检查点上运行。关键结果如下。

- E1-R2 的 WordNet 元数据匹配未支持神经元类别一致性；前置 RGB 匹配诊断保留为敏感性结果。
- E2 补齐 MLP 基本神经元、通道族及切面闭合。
- E3 未发现路径评分相对激活乘梯度的稳定神经元排序增量。
- E4 未发现结构特征相对总图的稳定独立测试预测增量。
- E5 在四检查点复现三个层深结构假设，并在四检查点否定头排序增量。

负结果直接进入摘要、结果和结论。多幅度低误差明确归属于参照位移感知泰勒量，该量不需要生成路径图。

## 证据与复现

evidence/revision_v5/ 是可随论文交付的紧凑包，包含协议、样本与单元清单、主要图像/类别级统计、E4 拟合参数、分析收据和哈希。逐任务 NPZ 保留在两个 revision_v5 报告根；紧凑包的分析收据逐项绑定这些源文件。

在 vitnns_rebuild 根目录运行：

    CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python3.10 -m pytest -q tests/test_revision_v5.py
    CUDA_VISIBLE_DEVICES='' ../.venv/bin/python papers/nns_vit_endpoint/thesis_blind_review_20260921/scripts/render_review_pdf.py
    CUDA_VISIBLE_DEVICES='' ../.venv/bin/python papers/nns_vit_endpoint/thesis_blind_review_20260921/scripts/audit_review_pdf.py
    CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python3.10 papers/nns_vit_endpoint/thesis_blind_review_20260921/scripts/validate_revision_v5.py

匿名 PDF 为通用 A4 内容审阅版，不替代学校模板、封面和最终匿名规范检查。
