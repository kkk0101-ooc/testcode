# Endpoint M2 supplementary result package

Read docs/endpoint_m2_results_20260919.md and completion_audit.json.
2400 image-model-block tasks; 21600 image-model-block-head conditions.
summary.csv and paired.csv contain all budgets and two geometries/outcomes.
head_summary.csv preserves every head; auc_tests.json contains 672 Holm-adjusted tests.
results/attempts/ contains the raw finite candidate logits, selector scores and curves locally.
artifact_manifest.json binds the raw records by SHA256. Git publishes code, aggregate tables,
figures and audit receipts; the large raw records and queue database remain local.
The original C4a/C4b omnibus stays NOT_SUPPORTED. Stop after M2.
