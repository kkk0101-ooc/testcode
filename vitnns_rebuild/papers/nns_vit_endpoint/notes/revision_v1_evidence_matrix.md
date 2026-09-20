# Revision v1 evidence matrix

Current manuscripts: [Chinese](../zh/full_draft.md), [English](../en/full_draft.md). The original release matrix remains [archived](endpoint_scanner_paper_evidence_matrix_r1.md). Source SHA-256 records are [analysis_sources.json](../../../reports/nns_vit_endpoint_paper_revision_v1_20260920/analysis_sources.json) and [figure_sources.json](../../../reports/nns_vit_endpoint_paper_revision_v1_20260920/figure_sources.json).

| Claim or result | Current location | Direct source | Interpretation |
|---|---|---|---|
| NNS identity and energy aggregation | 3.1–3.2, Proposition 1 | Chain-rule derivation; reports/nns_vit_endpoint_m3_r1_20260919/m3a_fidelity.json | Known Gradient × Input object under stated propagation conditions |
| Legal-cut closure; residual-output cotangent | 3.3–3.4, Proposition 2, Figure 2 | Chain-rule path partition; E3 registered component arrays and closure results | Covers the specified branch or complete target only when bypass paths are included |
| 5,000 comparisons; float64/float32 maxima | 5.1, Figure 3 | reports/nns_vit_endpoint_m3_r1_20260919/m3a_fidelity.json | Five compatible networks, 100 inputs per network/precision |
| 336,800 target evaluations; 457/385 per image | 5.2 | reports/nns_vit_endpoint_e2_atlas_r4_20260918/terminal_receipt.json | Registry-complete |
| 0.01058%/0.02115% original coverage | 5.2, Table S1 | reports/nns_vit_endpoint_m1_r1_20260918/registry_coverage.json | Post-block token-channel candidates only |
| 6,144 targets per image; 1,228,800 total; 0.338%/0.677% | 4.1, 5.2 | reports/nns_vit_endpoint_m3_r1_20260919/target_registry.json and terminal_receipt.json | Frozen stratified unit sample |
| 76,800 closure positives; 28,800 rejected negatives | 5.3, Figure 3 | reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json and validation_report.json | Elementwise tolerance; numerical identity |
| 6,144 JVP checks and condition-scaled validation | 5.3 | reports/nns_vit_endpoint_e4_intervention_r3_20260918/terminal_receipt.json | Local derivative consistency |
| 88 operator directions; 1.632e-10 best-step FD maximum | 5.3 | reports/nns_vit_endpoint_m3_r1_20260919/m3a_operator_validation.json | 11 operator classes |
| 36 retained float32 failures, float64 recovery | 5.3, Supplement S2 | reports/nns_vit_endpoint_m3_r1_20260919/initial_numeric_validation_failure.json and m3b_float64_recovery.json | Original failed records retained |
| Entropy/locality/distance depth changes, 100/100 directions | 5.4, Figure 4 | reports/nns_vit_endpoint_m3_r1_20260919/atlas_tests.json; m3_sup_r2_20260919/block_summary.csv | Image-level inference on sampled units |
| CLS-patch entropy; net/absolute mass | 5.4, Figure 5 | m3_sup_r2_20260919/cls_patch_summary.csv and block_summary.csv | Absolute mass is not net directional effect |
| Routing scale, Endpoint/JVP errors | 5.5, Figure 6 | reports/nns_vit_endpoint_m4_r1_20260919/attention_block_summary.csv and m4_tests.json | GT-logit-conditioned local path scale |
| JVP full-response denominators and absolute discrepancies | 5.5 | reports/nns_vit_endpoint_m4_review_r2_20260919/jvp_block_summary.csv | Error near one means comparable norm scale |
| Norm-adjusted mean-query saturation relations | 5.5, discussion | m4_review_r2_20260919/saturation_sensitivity_summary.csv and saturation_sensitivity_tests.json | Post-hoc partial-rank association; CLS weaker |
| Attention vs MLP residual identity and TV | 5.6, Figure 7 | m4_r1_20260919/residual_block_summary.csv; m4_review_r2_20260919/residual_spatial_tests.json | Frozen path tests and separate post-hoc spatial sensitivity |
| Case selection and raw signed/absolute maps | Figures 5–8 | reports/nns_vit_endpoint_m6_r1_20260919/selection_manifest.json and raw_arrays.npz | Inherited frozen metric selection; no new visual reselection |
| 11/12 component families; threshold/CI distinction | 5.7, Table 2, Figure 9 | reports/nns_vit_endpoint_e6_release_r3_20260918/family_decisions.json | Original omnibus NOT_SUPPORTED; lower-CI column descriptive |
| Within-block head Endpoint/Taylor results | 5.7, Table 3 | reports/nns_vit_endpoint_m1_r1_20260918/summary.json.gz | Supplementary frozen study, distinct from original C4a |
| Four unsupported spatial families and paired differences | 5.8, Table 4 | e6_release_r3_20260918/family_decisions.json | Original C4b remains NOT_SUPPORTED |
| Absolute AUC, candidate oracle, normalized regret and logit ratios | 5.8, Tables 5–6, Figure 10 | E5 800 task records, independently reconstructed from isolated scores; current analysis.json and spatial_image_values.csv | Descriptive; every original method image score agrees with E6 within 1e-12 |
| Reference differences, explicit tolerance and rank metrics | 5.9, Table 7, Tables S3–S4 | M1 800 checked per-image files; current analysis.json and reference_rank_image_values.csv | 163,200 component pairs; image aggregation for head rankings |
| Figure provenance and archived completeness | Manifest, Supplement S7 | R2 whole_root_inventory.json; all ten PNG/PDF/SVG outputs | Source package unchanged; 40 old main + 52 supplemental records retained |
| Corrected random absolute AUC | Tables 5–7; Supplement v1 S9 | `publication_r2_20260920/evidence/supplement_v1/baseline_analysis.json`; 800 SHA-256-bound E5 records | Absolute effect is taken within each of 20 repeats before AUC averaging; legacy estimator retained only for comparison |
| Cross-layer all-head M2 | Results 5.5; Supplement v1 S14 | `m2_inherited.json`; independently recomputed 672-test Holm family | Blocks 2/5/10; Taylor favored in 11/12 full-input and 12/12 isolated-head families; original C4b remains unchanged |
| Trained versus random initialization | Results 5.6; Supplement v1 S10 | `training_analysis.json`; 32 shared images/model, 3 seeds/model, all 12 blocks | Trained and random parameter states differ in spatial-broadening magnitude and several path trends; the comparison does not identify a causal training effect |
| GT/top1/margin robustness | Results 5.7; Supplement v1 S11 | `target_analysis.json`; 100 images/model, blocks 0/5/11, all heads | Main directions persist; magnitudes depend on target; incorrect subsets contain 8/14 images |
| Feature–Endpoint relation | Results 5.8; Supplement v1 S12 | `feature_analysis.json`; 12,288 frozen-channel energy targets and four negative controls | Same-channel association exceeds tested controls but does not imply equivalence |
| M3 sampling sensitivity | Results 5.9; Supplement v1 S13 | `sampling_analysis.json`; channel/token/image bootstrap and deletion analyses | Core spatial directions persist in the evaluated cohort; not an independent channel-seed replication |
| Remaining matched/CNN/checkpoint/transform controls | Discussion; Supplement v1 S14 | Explicit current scope in supplement v1 | Same-target bridge, cascading randomization, alternative trained checkpoint and transformation/CNN displays remain unexecuted |

The descriptive extension is CPU-only and adds no inferential family. Its analysis script recomputes curve AUC from raw isolated scores, verifies all original E6 image scores, reconstructs image-level oracle/logit ratios, and checks all M1 component-pair hashes. Figure rendering verifies the sealed source inventory before loading arrays. The revision validator checks actual manuscript table rows against sources, both language structures/equations, source and figure hashes, display metadata, and the archived R2 inventory.

Supplement v1 supersedes only the earlier prospective status of random-initialization, target-robustness, and feature–Endpoint controls. The earlier revision protocol and completion audit remain historical records of their pre-experiment scope. Current completion evidence is `publication_r2_20260920/qc/supplement_v1_completion_audit.md`; machine-readable evidence is under `publication_r2_20260920/evidence/supplement_v1/`.
