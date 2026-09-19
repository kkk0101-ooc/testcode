# M2 evidence matrix

| Claim or output | Evidence | Boundary |
|---|---|---|
| All-head cross-layer execution | task_manifest.json, task_validation.json | 400 images/model, blocks 2/5/10 |
| Endpoint/Taylor direction by layer | auc_tests.json, paired.csv, Tables M2-1/2 | Shared 672-test Holm family; no equivalence inference |
| Eight methods and budget curves | summary.csv, results/attempts/, M2_F1/M2_F2 | Image-unit medians; absolute effects |
| Head heterogeneity | head_summary.csv, M2_F3 | Descriptive difference of medians |
| Candidate regret | raw candidate tables, summary.csv | Singleton exhaustive; larger budgets evaluated union |
| Primary C4a/C4b remain NOT_SUPPORTED | Bound original E6 terminal; report_data.json | Supplementary results remain separate |

Result files are relative to reports/nns_vit_endpoint_m2_r1_20260919/. Figure references resolve through figure_manifest_m2.json. Raw large result records remain in the local result root and are SHA256-bound in artifact_manifest.json; Git includes the code, aggregate tables, figures, and audit chain.
