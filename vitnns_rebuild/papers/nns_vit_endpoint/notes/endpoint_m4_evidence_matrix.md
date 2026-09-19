# Endpoint M4 evidence matrix

| Claim | Evidence | Status |
|---|---|---|
| Full cohort and head coverage | task_manifest.json, coverage_audit.json, analysis_receipt.json | PASS |
| Q/K/V and routing/value closure | per-task closure fields and validation_report.json | PASS |
| Frozen-attention JVP | eight deterministic directions per image-head, explicit softmax differential versus native JVP | PASS |
| Attention detach and forward parity | first 16 frozen evaluation orders per model, all 12 blocks | PASS |
| Residual identity/branch closure | both residual types, all blocks and images | PASS |
| Omitted-identity negative control | one control per residual record | PASS |
| E3 inheritance | 288 scalar summary crosschecks per image-model task | PASS |
| Mechanism statistics | image-first aggregation, 10,000 bootstrap, exact sign and frozen Holm families | COMPLETE |
| Scope | local input dependence, not causal head or branch importance | BOUNDARY |
