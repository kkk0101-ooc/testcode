# Endpoint M3 evidence matrix

| Claim | Evidence | Status |
|---|---|---|
| Original NNS equivalence | `m3a_fidelity.json`: 5 graph families, 2 dtypes, 100 inputs each, 5,000 target comparisons | PASS |
| ViT operator derivatives | `m3a_operator_validation.json`: 11 operators, 88 directions, analytic JVP/autograd/VJP/FD | PASS |
| Frozen stratified atlas | `target_registry.json`, `cohort.json`, `task_validation.json`: 200 tasks and 1,228,800 targets | PASS |
| Numerical recovery | `initial_numeric_validation_failure.json` plus `m3b_float64_recovery.json`: 36 float32 failures retained and all 10 selected tasks pass float64 review | PASS_WITH_RETAINED_FAILURE |
| M3 atlas decisions | `image_values.csv`, `atlas_summary.csv`, `atlas_tests.json`; image unit and per-model Holm families | PASS |
| Unit-map similarity | `similarity_image_values.csv`, `similarity_phase_image_values.csv`, `similarity_summary.csv`; 256 frozen pairs/layer/group | DESCRIPTIVE |
| Full-resolution examples | `full_map_manifest.json`; 120 pre-frozen raw float32 maps | PASS |
| Scope | 6,144 sampled token-channel units/model/image; no exhaustive-neuron or finite-intervention claim | BOUNDARY |
