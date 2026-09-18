# Claim-evidence matrix

| ID | Manuscript claim | Direct evidence | Role | Confidence / decision | Required boundary |
|---|---|---|---|---|---|
| C1 | A learning image maps a scalar response change to input coordinates and satisfies completeness | `reports/nns_vit_unified_paper_figures_r10_r1_20260917/module_compute_receipt.json`; R10-R4 classifier receipt | Method validation | Strong | Numerical identity under the frozen path and quadrature |
| C2 | Both Block-0 residual additions are pixelwise additive before display | R10-R1 module receipt; Figures 31-33 | Method validation | Strong | One frozen image; raw RGB arrays, not colored panels |
| C3 | Stage D fine-cell fidelity is positive on 500/500 images, median 2.3082, 95% CI [2.1111, 2.4452] | `evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis/analysis.json` | Primary result | `GO_FIDELITY_SCOPE` | Frozen ViT-B/16, target, block 0/head 9, baseline, and 4x4-cell endpoint |
| C4 | The advantage nearly disappears at native patch scale | `evidence/stage_e/e4_recovery_v2_20260907/formal/paper_data/patch_scale_sensitivity.json` | Secondary/descriptive | Strong descriptive | Same Stage D cohort; post-D10 analysis |
| C5 | Effect size differs across four prespecified coordinates | `evidence/stage_e/e4_recovery_v2_20260907/formal/paper_data/layer_head_descriptive.json` | Secondary/descriptive | Mixed-validity package | Only four coordinates; block 11/head 2 has 498 valid and 2 typed failures |
| C6 | Positive differences occur in both correct and misclassified strata | `evidence/stage_e/e4_recovery_v2_20260907/formal/paper_data/prediction_strata.json` | Secondary/descriptive | Strong descriptive | Directional consistency is not an equivalence or causal claim |
| C7 | NNS is below Grad-CAM and above rollout in the 50-image class-output comparison | `evidence/stage_f/f3c4_analysis_r1_20260908/formal/paper_data/contrast_summary.json` | Independent comparison | Registered significant contrasts | Conditional class functional; methods do not all share identical targets |
| C8 | Differences from full-model IG and raw CLS attention were not significant | Same contrast table and `scientific_interpretation.json` | Independent comparison | Not significant | Do not claim equivalence |
| C9 | Complete-cut component maps reconstruct the total input attribution | `evidence/stage_x/x1_r1_nns_v2_20260908/formal/identity_control_validation.json`; `complete_cut_validation.json` | Implementation validation | PASS | Total map equals same-contract full-model IG; components are the added index |
| C10 | Attention and MLP branch shares change differently with depth | `evidence/stage_y/y4_r5_mechanism_analysis_20260914/formal/claim_decision_summary.json`; `hypothesis_test_table.json` | Mechanism result | Y-RQ5 SUPPORTED | The two residual cuts are separate denominators |
| C11 | Broad geometric stability was not established | Stage X2 terminal and results-to-paper handoff | Negative result | `NO_GO_BROAD_CLAIM_DEVELOPMENT` | Do not relabel the multiscale post-hoc pattern as a passed stability claim |
| C12 | Coarser aggregation increases signed geometric association | `evidence/stage_x/x2_mech_m2_20260910/formal/completion/x2_mech_m2.json` | Mechanism interpretation | `MIXED_OR_INCONCLUSIVE` | Post-hoc exploratory; associative, not causal |
| C13 | Alignment, matched occlusion, and decoding show depth trends | `evidence/nns_vit_patch_depth/s5_r1_posthoc_exploratory_20260916/endpoint_decisions.json`; `depth_profiles_and_D1_D2_relationship.json` | Extension | Post-hoc exploratory | Original P2 rule did not pass; outcome-access threshold must be disclosed |
| C14 | Correct, competitor, margin, and low-score classifier targets yield distinct maps | R10-R4 `classifier_comparison_receipt.json`; Figure 35 | Method illustration | Strong single-image validation | No population classifier-mechanism inference |

## Historical evidence that must remain visible

- C10: `n=15`, fidelity component `PASS_EFFECT`, overall `NO_GO_NOVELTY`, spatial `FAIL_SPATIAL`, workflow `BLOCKED`, Holm `HOLM_BLOCKED_UNDERSPECIFIED`.
- Stage E: E2 used an explicit stage-specific owner budget waiver; E3 retained two typed failures; E4 workflow/package passed with `MIXED` scientific validity.
- Stage X: implementation identity passed, but broad-claim development stopped after the stability study.
- Stage Y: only Y-RQ5 was supported; Y-RQ1 and Y-RQ2 were not supported, Y-RQ3 inconclusive, and Y-RQ4/Y-RQ6 invalid.
