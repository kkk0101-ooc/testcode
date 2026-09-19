# Endpoint M6 evidence matrix

| Requirement | Evidence | Status | Boundary |
|---|---|---|---|
| Twelve case categories | selection_manifest.json | PASS | Fixed metrics and ID tie-breaks |
| Selection before images | freeze.json, source_validation.json | PASS | CPU-only numeric pass |
| Neuron atlas fields | m6_neuron_atlas files, raw_arrays.npz | PASS | Local Endpoint dependence |
| Attention full/routing/value/Q/K and attention | m6_attention_cases files | PASS | Relative path norms, not causal percentages |
| Residual output/identity/branch | m6_residual_cases files | PASS | Shared clean output cotangent |
| Cross-model representative case | m6_model_layer_comparison files | PASS | No channel correspondence claim |
| M2 boundary case | m6_m2_semantic_boundary files | PASS | Selection excludes finite outcomes |
| Display and provenance | figure_manifest.json, map_recompute_receipt.json | PASS | Raw arrays authoritative |
