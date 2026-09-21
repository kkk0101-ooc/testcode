# NNS–ViT 结构分解学位论文（修改意见 v4）

本目录执行《修改意见/修改意见v4.md》的最低送审路线，复用已有封存数值完成双语改稿与完整 12 层事后分析，无新增模型推理。原 v3 版本可由 Git 历史恢复，既有科学证据包保持封存。

## 阅读入口

- [中文稿](zh/manuscript.md)
- [同步英文稿](en/manuscript.md)
- [匿名内容审阅 PDF](NNS_ViT_学位论文_匿名内容审阅版.pdf)
- [98 个指标族的完整层序与相邻差图册](figures/revision_v4/all_profiles_and_adjacent_differences.pdf)
- [统一参考文献](references.md)

主要输出为带真实边身份的结构路径图组。正文围绕结构定义与闭合、层深差异、规定有限替换三个研究问题，保留原联合失败和事后随机基线修正。

## 数据与审计

- evidence/revision_v4/：全部图像轨迹、逐块区间、趋势、阶段/端点/相邻差、缺失计数及原检验族。
- figures/revision_v4/：五组目标、全部查询键值指标、注意力与残差曲线，PDF 和 PNG 双格式。
- figure_manifest.json：论文图和补充图的路径、哈希及显示规则。
- notes/revision_v4_plan.md：执行范围和事后分析规则。
- notes/claim_evidence_matrix.md：主张到定义、实验、数字及来源。
- qc/completion_audit.md：v4 每条要求的实际完成证据。

在 vitnns_rebuild 根运行（各命令独立执行）：

    CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python3.10 papers/nns_vit_endpoint/thesis_blind_review_20260921/scripts/analyze_revision_v4_depth.py
    CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python3.10 papers/nns_vit_endpoint/thesis_blind_review_20260921/scripts/validate_revision_v4_data.py
    CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python3.10 papers/nns_vit_endpoint/thesis_blind_review_20260921/scripts/render_revision_v4_regional.py
    python3 papers/nns_vit_endpoint/thesis_blind_review_20260921/scripts/finalize_revision_v4.py
    CUDA_VISIBLE_DEVICES='' ../.venv/bin/python papers/nns_vit_endpoint/thesis_blind_review_20260921/scripts/render_review_pdf.py
    CUDA_VISIBLE_DEVICES='' ../.venv/bin/python papers/nns_vit_endpoint/thesis_blind_review_20260921/scripts/audit_review_pdf.py
    python3 papers/nns_vit_endpoint/thesis_blind_review_20260921/scripts/validate_thesis.py

分析脚本只读取封存数值，不加载模型或图像。匿名 PDF 使用通用 A4 内容审阅格式；学校专用封面与排版需依据实际学校模板套用。可选增强实验的具体设计在第 8.5 节，当前结论不依赖这些未执行实验。
