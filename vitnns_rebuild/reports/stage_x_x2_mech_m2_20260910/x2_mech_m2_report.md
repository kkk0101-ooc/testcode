# X2-MECH M2 terminal report

Status: `PASS`. This means package, workflow, and scientific validity completed; it does not issue a favorable mechanism branch or alter X2.

Study type: `POST_HOC_EXPLORATORY_MECHANISM_ANALYSIS`. X2 remains `NO_GO_BROAD_CLAIM_DEVELOPMENT`. M1-R1 remains unchanged at `b6d14965cce3d7a4618b4f5435331c7a9b954d19`.

Mechanism branch: `MIXED_OR_INCONCLUSIVE`. The frozen handoff did not define numeric thresholds that uniquely select A/B/C, so M2 reports the observed directions under the mixed/inconclusive branch.

Core observed directions:

- NNS-v2 median signed Spearman, x2 common mask: pixel `0.11366648574801438`, 4x4 `0.2177360373571473`, 16x16 `0.33779091119911697`.
- ViT Grad-CAM median signed Spearman, x2 common mask: pixel `0.6470336100253422`, 4x4 `0.6532370714163442`, 16x16 `0.7218010653661862`.
- NNS-v2 ROAD Combined: 4x4 `0.05496294885494901`, 16x16 `0.06067247007244031`; Grad-CAM ROAD Combined: 4x4 `0.1499317089177042`, 16x16 `0.15329919519210142`.
- NNS-v2 SSE fractions: 4x4 between `0.07447684921684658`, within `0.9255231507831534`; 16x16 between `0.0026145042858070795`, within `0.997385495714193`.
- Horizontal-flip transport median signed Spearman: `1.0`.

Association intervals, first registered rows:

- `nns_v2` absolute_top10_pixel_mass_share vs median_pixel_signed_spearman: effect `-0.31235607781179026`, 95% exploratory interval `[-0.4529167946038158, -0.13814825434420727]`.
- `nns_v2` absolute_top10_pixel_mass_share vs horizontal_flip_signed_spearman: effect `-0.2866189394687469`, 95% exploratory interval `[-0.46102539323743336, -0.08328791630369906]`.
- `nns_v2` normalized_total_variation vs pixel_to_fine_signed_spearman_change: effect `0.2538214673644712`, 95% exploratory interval `[0.09650361396139384, 0.3920831264942336]`.
- `nns_v2` within_patch_16_residual_energy_fraction vs pixel_to_patch_signed_spearman_change: effect `-0.11566801619433198`, 95% exploratory interval `[-0.2613235637188578, 0.03771592656261815]`.

Counts: expected denominator `200`, valid `199`, failed `1`; pixel reproduction rows `4776`; multiscale stability rows `28656`; M1 maps `9552`, masks `24`, fidelity summaries `796`, ROAD rows `71640` revalidated.

Access and resources: CPU-only; CUDA/GPU false; model/checkpoint/JPEG false; forward/backward calls `0`; no downloads; no PNG/JPEG display output.

Next authorization: `X2_MECH_RESULTS_TO_PAPER_HANDOFF_ONLY`. F3-Final, Stage G, X3, X4, and X5 were not started.
