# Endpoint M3-SUP-R2 evidence matrix

| Supplement | Primary artifacts | Status |
|---|---|---|
| NNS equivalence distributions | `m3a_error_distribution.csv`, source `m3a_fidelity.json` | PASS |
| Operator-level validation table | `operator_validation_table.csv`, source `m3a_operator_validation.json` | PASS |
| Twelve-block atlas trajectories | `block_image_values.csv`, `block_summary.csv` | SUPPLEMENTARY DESCRIPTIVE |
| Block-first CLS-patch comparison | `cls_patch_image_values.csv`, `cls_patch_summary.csv` | SUPPLEMENTARY DESCRIPTIVE |
| Multi-scale cancellation | `cancellation_map_values.csv`, `cancellation_summary.csv` | SUPPLEMENTARY DESCRIPTIVE |
| Float64 cancellation stability | `cancellation_float64_review.json`, 24 targets | PASS |
| Float32 failure diagnosis | `finite_difference_diagnostics.csv`, all 36 retained failures | PASS FLOAT64 RECOVERY; FLOAT32 FAIL RETAINED |
| Figures | `figure_manifest.json`, five PNG/PDF pairs | PASS |
