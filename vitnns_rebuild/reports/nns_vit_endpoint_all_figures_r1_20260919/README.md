# Endpoint-NNSViT 全图片汇总

共 `88` 张去重 PNG，全部平铺在当前目录。

- `fig_01_...` 到 `fig_40_...`：现行论文 Figure 1–40；
- `signed_fig_...`：Figure 34–40 的完整 signed companion；
- `all_heads_...`：DeiT-S/16 与 ViT-B/16 各 12 层的全 head 页；
- `supp_...`：M2、M3、M4、M4 review 与 M6 补充图。

图片来自原权威目录的字节级复制；`manifest.json` 记录原路径和 SHA-256。

验证命令：

```bash
python3 scripts/validate_endpoint_all_figures_collection.py
```

## paper_main_1_28

- [fig_01_endpoint_scanner_study_overview.png](fig_01_endpoint_scanner_study_overview.png) — Endpoint Scanner study overview
- [fig_02_endpoint_cnn_path_method_identity.png](fig_02_endpoint_cnn_path_method_identity.png) — Endpoint / CNN / Path method identity
- [fig_03_same_target_endpoint_versus_path.png](fig_03_same_target_endpoint_versus_path.png) — Same target: Endpoint versus Path
- [fig_04_original_cnn_endpoint_gates.png](fig_04_original_cnn_endpoint_gates.png) — Original CNN endpoint gates
- [fig_05_attention_q_k_v_differential_paths.png](fig_05_attention_q_k_v_differential_paths.png) — Attention Q/K/V differential paths
- [fig_06_residual_edge_accounting.png](fig_06_residual_edge_accounting.png) — Residual edge accounting
- [fig_07_unit_sum_and_energy_targets.png](fig_07_unit_sum_and_energy_targets.png) — Unit, sum, and energy targets
- [fig_08_path_seed_and_input_coordinates.png](fig_08_path_seed_and_input_coordinates.png) — Path, seed, and input coordinates
- [fig_09_layernorm_and_gelu_jacobians.png](fig_09_layernorm_and_gelu_jacobians.png) — LayerNorm and GELU Jacobians
- [fig_10_structured_edge_cuts.png](fig_10_structured_edge_cuts.png) — Structured edge cuts
- [fig_11_registered_target_atlas_design.png](fig_11_registered_target_atlas_design.png) — Registered target atlas design
- [fig_12_frozen_first_eight_endpoint_gallery.png](fig_12_frozen_first_eight_endpoint_gallery.png) — Frozen first-eight endpoint gallery
- [fig_13_one_image_multi_target_atlas.png](fig_13_one_image_multi_target_atlas.png) — One-image multi-target atlas
- [fig_14_scan_scale_latency_and_resources.png](fig_14_scan_scale_latency_and_resources.png) — Scan scale, latency, and resources
- [fig_15_structural_closure_contracts.png](fig_15_structural_closure_contracts.png) — Structural closure contracts
- [fig_16_formal_closure_and_negative_controls.png](fig_16_formal_closure_and_negative_controls.png) — Formal closure and negative controls
- [fig_17_real_structural_component_maps.png](fig_17_real_structural_component_maps.png) — Real structural component maps
- [fig_18_condition_scaled_analytic_jvp_checks.png](fig_18_condition_scaled_analytic_jvp_checks.png) — Condition-scaled analytic JVP checks
- [fig_19_matched_finite_activation_replacement.png](fig_19_matched_finite_activation_replacement.png) — Matched finite activation replacement
- [fig_20_c4a_family_estimates_and_intervals.png](fig_20_c4a_family_estimates_and_intervals.png) — C4a family estimates and intervals
- [fig_21_rq3_threshold_zoom_and_boundary_exception.png](fig_21_rq3_threshold_zoom_and_boundary_exception.png) — RQ3 threshold zoom and boundary exception
- [fig_22_endpoint_and_registered_secondary_predictors.png](fig_22_endpoint_and_registered_secondary_predictors.png) — Endpoint and registered secondary predictors
- [fig_23_spatial_intervention_protocol.png](fig_23_spatial_intervention_protocol.png) — Spatial intervention protocol
- [fig_24_real_block_5_component_and_baseline_maps.png](fig_24_real_block_5_component_and_baseline_maps.png) — Real block-5 component and baseline maps
- [fig_25_one_image_spatial_effect_curves.png](fig_25_one_image_spatial_effect_curves.png) — One-image spatial effect curves
- [fig_26_c4b_mandatory_baseline_contrasts.png](fig_26_c4b_mandatory_baseline_contrasts.png) — C4b mandatory-baseline contrasts
- [fig_27_rq4_family_summary.png](fig_27_rq4_family_summary.png) — RQ4 family summary
- [fig_28_identity_correctness_utility_conclusion.png](fig_28_identity_correctness_utility_conclusion.png) — Identity–Correctness–Utility conclusion

## paper_main_29_33

- [fig_29_m3_sup_block_trajectories.png](fig_29_m3_sup_block_trajectories.png) — Image-level medians and 95% bootstrap intervals over all twelve blocks.
- [fig_30_m3_sup_cls_patch_by_block.png](fig_30_m3_sup_cls_patch_by_block.png) — Paired image-level CLS-minus-patch entropy by block.
- [fig_31_m3_sup_cancellation_scales.png](fig_31_m3_sup_cancellation_scales.png) — Raw-map cancellation at RGB, within-patch, across-patch, and whole-image scales.
- [fig_32_m3_sup_equivalence_errors.png](fig_32_m3_sup_equivalence_errors.png) — 99th-percentile relative L2 by network and precision.
- [fig_33_m3_sup_fd_diagnostics.png](fig_33_m3_sup_fd_diagnostics.png) — Median and 10th-90th percentile absolute FD error across the retained failure set.

## paper_main_34_40

- [fig_34_registered_target_learning_images.png](fig_34_registered_target_learning_images.png) — Endpoint target atlas: real fine-grained learning images
- [fig_35_block5_structural_learning_images.png](fig_35_block5_structural_learning_images.png) — Block 5: real structural Endpoint learning images
- [fig_36_spatial_ranking_learning_images.png](fig_36_spatial_ranking_learning_images.png) — Block 5: real maps used for spatial ranking
- [fig_37_channel0_depth_progression.png](fig_37_channel0_depth_progression.png) — Endpoint-NNSViT across depth
- [fig_38_attention_head_depth_atlas.png](fig_38_attention_head_depth_atlas.png) — Endpoint-NNSViT across heads and depth
- [fig_39_vit_b16_attention_head_depth_atlas.png](fig_39_vit_b16_attention_head_depth_atlas.png) — ViT-B/16 Endpoint across heads and depth
- [fig_40_vit_b16_module_chain.png](fig_40_vit_b16_module_chain.png) — Endpoint-NNSViT through the real module chain

## signed_companions

- [signed_fig_34_registered_target_learning_images_signed.png](signed_fig_34_registered_target_learning_images_signed.png) — Endpoint target atlas: real fine-grained learning images - signed companion
- [signed_fig_35_block5_structural_learning_images_signed.png](signed_fig_35_block5_structural_learning_images_signed.png) — Block 5: real structural Endpoint learning images - signed companion
- [signed_fig_36_spatial_ranking_learning_images_signed.png](signed_fig_36_spatial_ranking_learning_images_signed.png) — Block 5: real maps used for spatial ranking - signed companion
- [signed_fig_37_channel0_depth_progression_signed.png](signed_fig_37_channel0_depth_progression_signed.png) — Endpoint-NNSViT across depth - signed companion
- [signed_fig_38_attention_head_depth_atlas_signed.png](signed_fig_38_attention_head_depth_atlas_signed.png) — Endpoint-NNSViT across heads and depth - signed companion
- [signed_fig_39_vit_b16_attention_head_depth_atlas_signed.png](signed_fig_39_vit_b16_attention_head_depth_atlas_signed.png) — ViT-B/16 Endpoint across heads and depth - signed companion
- [signed_fig_40_vit_b16_module_chain_signed.png](signed_fig_40_vit_b16_module_chain_signed.png) — Endpoint-NNSViT through the real module chain - signed companion

## all_head_pages

- [all_heads_deit_s16_block_00.png](all_heads_deit_s16_block_00.png) — deit_s16 block 00 all heads
- [all_heads_deit_s16_block_01.png](all_heads_deit_s16_block_01.png) — deit_s16 block 01 all heads
- [all_heads_deit_s16_block_02.png](all_heads_deit_s16_block_02.png) — deit_s16 block 02 all heads
- [all_heads_deit_s16_block_03.png](all_heads_deit_s16_block_03.png) — deit_s16 block 03 all heads
- [all_heads_deit_s16_block_04.png](all_heads_deit_s16_block_04.png) — deit_s16 block 04 all heads
- [all_heads_deit_s16_block_05.png](all_heads_deit_s16_block_05.png) — deit_s16 block 05 all heads
- [all_heads_deit_s16_block_06.png](all_heads_deit_s16_block_06.png) — deit_s16 block 06 all heads
- [all_heads_deit_s16_block_07.png](all_heads_deit_s16_block_07.png) — deit_s16 block 07 all heads
- [all_heads_deit_s16_block_08.png](all_heads_deit_s16_block_08.png) — deit_s16 block 08 all heads
- [all_heads_deit_s16_block_09.png](all_heads_deit_s16_block_09.png) — deit_s16 block 09 all heads
- [all_heads_deit_s16_block_10.png](all_heads_deit_s16_block_10.png) — deit_s16 block 10 all heads
- [all_heads_deit_s16_block_11.png](all_heads_deit_s16_block_11.png) — deit_s16 block 11 all heads
- [all_heads_vit_b16_block_00.png](all_heads_vit_b16_block_00.png) — vit_b16 block 00 all heads
- [all_heads_vit_b16_block_01.png](all_heads_vit_b16_block_01.png) — vit_b16 block 01 all heads
- [all_heads_vit_b16_block_02.png](all_heads_vit_b16_block_02.png) — vit_b16 block 02 all heads
- [all_heads_vit_b16_block_03.png](all_heads_vit_b16_block_03.png) — vit_b16 block 03 all heads
- [all_heads_vit_b16_block_04.png](all_heads_vit_b16_block_04.png) — vit_b16 block 04 all heads
- [all_heads_vit_b16_block_05.png](all_heads_vit_b16_block_05.png) — vit_b16 block 05 all heads
- [all_heads_vit_b16_block_06.png](all_heads_vit_b16_block_06.png) — vit_b16 block 06 all heads
- [all_heads_vit_b16_block_07.png](all_heads_vit_b16_block_07.png) — vit_b16 block 07 all heads
- [all_heads_vit_b16_block_08.png](all_heads_vit_b16_block_08.png) — vit_b16 block 08 all heads
- [all_heads_vit_b16_block_09.png](all_heads_vit_b16_block_09.png) — vit_b16 block 09 all heads
- [all_heads_vit_b16_block_10.png](all_heads_vit_b16_block_10.png) — vit_b16 block 10 all heads
- [all_heads_vit_b16_block_11.png](all_heads_vit_b16_block_11.png) — vit_b16 block 11 all heads

## supp_m2

- [supp_m2_01_M2_F1.png](supp_m2_01_M2_F1.png) — M2_F1
- [supp_m2_02_M2_F2.png](supp_m2_02_M2_F2.png) — M2_F2
- [supp_m2_03_M2_F3.png](supp_m2_03_M2_F3.png) — M2_F3

## supp_m3

- [supp_m3_01_m3_atlas_depth.png](supp_m3_01_m3_atlas_depth.png) — m3_atlas_depth
- [supp_m3_02_m3_map_similarity.png](supp_m3_02_m3_map_similarity.png) — m3_map_similarity

## supp_m4

- [supp_m4_01_m4_attention_depth.png](supp_m4_01_m4_attention_depth.png) — m4_attention_depth
- [supp_m4_02_m4_attention_saturation.png](supp_m4_02_m4_attention_saturation.png) — m4_attention_saturation
- [supp_m4_03_m4_residual_depth.png](supp_m4_03_m4_residual_depth.png) — m4_residual_depth

## supp_m4_review

- [supp_m4_review_01_m4_review_jvp_diagnostics.png](supp_m4_review_01_m4_review_jvp_diagnostics.png) — m4_review_jvp_diagnostics
- [supp_m4_review_02_m4_review_saturation_robustness.png](supp_m4_review_02_m4_review_saturation_robustness.png) — m4_review_saturation_robustness
- [supp_m4_review_03_m4_review_residual_spatial.png](supp_m4_review_03_m4_review_residual_spatial.png) — m4_review_residual_spatial
- [supp_m4_review_04_m4_review_m3_m4_bridge.png](supp_m4_review_04_m4_review_m3_m4_bridge.png) — m4_review_m3_m4_bridge

## supp_m6

- [supp_m6_01_m6_neuron_atlas.png](supp_m6_01_m6_neuron_atlas.png) — m6_neuron_atlas
- [supp_m6_02_m6_model_layer_comparison.png](supp_m6_02_m6_model_layer_comparison.png) — m6_model_layer_comparison
- [supp_m6_03_m6_attention_cases.png](supp_m6_03_m6_attention_cases.png) — m6_attention_cases
- [supp_m6_04_m6_residual_cases.png](supp_m6_04_m6_residual_cases.png) — m6_residual_cases
- [supp_m6_05_m6_m2_semantic_boundary.png](supp_m6_05_m6_m2_semantic_boundary.png) — m6_m2_semantic_boundary
