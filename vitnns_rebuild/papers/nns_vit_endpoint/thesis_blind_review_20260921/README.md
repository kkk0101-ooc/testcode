# NNS–ViT 学位论文盲审版

本目录是 `修改意见/修改意见v3.md` 第三步的独立交付物。它把
`publication_r2_20260920` 及修改意见 v3 第二步的封存证据重组为八章学位论文，
不改写既有证据包。

## 正文

- `zh/manuscript.md`：中文盲审主文。
- `en/manuscript.md`：与中文稿章节、公式、数字、表格和结论同步的英文稿。
- `references.md`：正文实际引用的统一参考文献。

正文已匿名化，不含作者、单位、学号、导师、通讯作者或仓库地址字段。学校专用封面、
授权页、字号、页眉和装订格式需要在取得学校模板后套用；本目录不虚构这些字段。

## 证据与质量控制

- 数值源：`../publication_r2_20260920/evidence/` 及仓库内已封存报告。
- 图源：`../publication_r2_20260920/figures/` 与第二步三张结果图。
- `notes/claim_evidence_matrix.md`：核心结论到公式、实验、统计量、表格和原始证据的映射。
- `scripts/validate_thesis.py`：第三步专用、标准库校验器。
- `qc/defensive_writing_audit.md`：防御性写作审计。
- `qc/page_review.md`：校验后逐节人工检查记录。
- `qc/completion_audit.md`：逐条 prompt-to-artifact 完成审计。

运行：

```bash
python3 papers/nns_vit_endpoint/thesis_blind_review_20260921/scripts/validate_thesis.py
```

通用 A4 内容审阅版由 `scripts/render_review_pdf.py` 生成，并由
`scripts/audit_review_pdf.py` 逐页检查；两者使用仓库现有 Python 3.10 虚拟环境。
