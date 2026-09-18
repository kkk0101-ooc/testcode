# NNS-ViT Endpoint Scanner：ENDPOINT-E6 恢复 R3

日期：2026-09-18。E6-R2 在已发布 commit
`49a89eef47b3ce363a32ad413d5ce7bdd2e2e829` 上完成全部统计并写出原子 terminal，
但首次封存审计发现双语 release、证据矩阵和写作审计中的段落分隔被编码为字面量
反斜杠 n，不满足论文发布格式。因此 R2 不是最终权威 release；其统计文件、terminal 和
错误格式文本保持不可变证据。

R3 只修复 Markdown 换行物化，并使用版本化论文文件和新结果根。冻结 cohort、逐图
统计输入、16 个 family、28 项 Holm family、所有阈值、seed、统计内核和科学判定均不变。
R3 从 E4/E5 原始逐任务 JSON 完整重算，不从 R2 复制统计输出。

```text
reports/nns_vit_endpoint_e6_release_r3_20260918
manuscript/en/endpoint_scanner_results_release_r3.md
manuscript/zh/endpoint_scanner_results_release_r3.md
manuscript/notes/endpoint_scanner_e6_evidence_matrix_r3.md
manuscript/qc/endpoint_scanner_e6_defensive_writing_audit_r3.md
```

本修复、R2 不可变证据和 R3 run spec 必须先提交并推送；远端包含该 commit 后才可
重跑正式统计。R3 validator 必须重算逐图值、bootstrap、sign test、Holm 和 family
decision，并检查新 Markdown 含真实换行而无字面段落转义。
