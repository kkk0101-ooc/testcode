# Stage X X2-MECH P0-R1 protocol freeze

Date: 2026-09-10

Study: `POST_HOC_EXPLORATORY_MECHANISM_ANALYSIS`

Authorization: `X2_MECH_P0_R1_PROTOCOL_AND_IMPLEMENTATION_FREEZE_ONLY`

## 1. Identity and immutable boundary

X2-MECH studies the relationship among NNS-v2 granularity, class-logit fidelity, and geometric equivariance after the X2 outcome was known. It is post-hoc exploratory. P0-R1 freezes the analysis before any new transformed map or multiscale outcome is generated or read; it is not a preregistered confirmation.

X2 remains `NO_GO_BROAD_CLAIM_DEVELOPMENT`. X2-MECH cannot issue a spatial-stability PASS, NNS-v2-versus-Grad-CAM superiority or noninferiority PASS, broad-benchmark GO, localization claim, cross-model claim, or cross-dataset claim. Its strongest permitted wording is `post-hoc exploratory evidence of a granularity-fidelity-equivariance trade-off`. X3-X5 remain unauthorized. C10, D10, E4, Stage F, F3C, and all X2 roots, receipts, commits, and refs are read-only.

The access correction is defined by [stage_x_x2_mech_p0_r1_access_boundary_amendment_20260910.md](stage_x_x2_mech_p0_r1_access_boundary_amendment_20260910.md): `source-code access != empirical-outcome access`. P0-R1 may inspect implementation contracts but must not read real map values or per-route outcome rows.

## 2. Cohort, methods, and transforms

The denominator is the original 200 model-image routes across the five X2 models. The 199 clean-valid routes enter paired analysis. The one immutable `FAILED_ROUTE_LOCAL_NUMERICAL` remains in place and cannot be retried, replaced, imputed, or removed.

Methods are exactly `nns_v2` and `vit_grad_cam`. M1 recomputes exactly these 12 transforms for each valid route:

- translations: x/y at -8, -4, +4, and +8 pixels;
- rotations: -10 and +10 degrees;
- `horizontal_flip`;
- `center_crop_216_resize_224`.

M1 inherits X2's model/checkpoint bindings, preprocessing, fixed clean predicted-class raw-logit target, native-unfused graph, NNS-v2 normalized-zero straight path, per-route selected IG steps, Grad-CAM adapter/layer, and exact forward/inverse geometric definitions. There is no new transform, gamma tuning, attribution change, target change, selected-step change, transform averaging, sample selection, or result-dependent model selection.

## 3. M1 map contract

For 199 routes x 12 transforms x 2 methods, M1 retains 4,776 transformed-coordinate maps and 4,776 inverse-mapped clean-coordinate maps. Each map is raw signed contiguous float32 with shape `[224,224]`; normalization, smoothing, thresholding, and display interpolation are forbidden. M1 also records transform/inverse operator metadata, interpolation, padding/fill, coordinate convention, identity fields, source-clean-map hash/dtype/shape, output hashes and canonical array digests, mask digest, selected steps, target, prediction-change state, X2 terminal binding, and resource telemetry.

Masks are shared by transform rather than copied per instance:

- `x2_frozen_common_domain` exactly reproduces X2;
- `exact_operator_support` depends only on geometric sampling support.

Both horizontal-flip masks cover the full domain. JPEG/PNG outputs, clean-map copies, and component maps are forbidden.

## 4. Frozen multiscale operators and metrics

Both methods use the same non-overlapping signed-sum operator at pixel 1x1, fine-cell 4x4, and native-patch 16x16 scales. `signed_sum` and `absolute_mass_sum` are reported separately. A block enters an analysis only when every pixel in it is valid; partial blocks are excluded and counted. Resize-based pseudo-granularity and method-specific smoothing, normalization, thresholding, or low-pass choices are forbidden.

For every route x transform x method x mask x scale, M2 computes signed Spearman, top-10%-unit signed-value Jaccard, absolute-mass Spearman, top-10%-unit absolute-mass Jaccard, signed SSIM, normalized positive/negative mass displacement, and absolute-mass center-of-mass error in pixel units. Top 10% means `ceil(0.10 * valid units)`, never cumulative 10% mass.

Pixel-scale `x2_frozen_common_domain` outputs must reproduce the sealed X2 stability rows with continuous `atol=rtol=1e-6` and exact Boolean/prediction state. Drift terminates as `BLOCKED_X2_PIXEL_REPRODUCTION_DRIFT`. Horizontal flip is reported separately.

The transport-only control applies the frozen forward operator and inverse operator to each clean attribution without running a model. It estimates loss due only to geometric resampling, crop, and masks. It remains separate from observed transformed attribution and its difference is not a strict causal effect.

## 5. Exact error decomposition

For `D = transformed_inverse - clean`, and separately for every complete valid 4x4 and 16x16 block `g`, M2 uses float64 accumulation:

```text
mu_g        = mean(D within g)
SSE_between = sum_g n_g * mu_g^2
SSE_within  = sum_g sum_i_in_g (D_i - mu_g)^2
SSE_total   = sum_i D_i^2
```

The identity `SSE_total = SSE_between + SSE_within` uses `atol=rtol=1e-10`. Outputs include within/between fractions, block-level signed-rank stability, within-block demeaned residual stability, valid block count, and excluded partial block count. Only SSE is called an exact decomposition; Spearman and Jaccard are not additive decompositions.

## 6. Clean structure and exploratory association

Each clean route/method yields absolute top-10%-pixel mass share, normalized Shannon entropy, effective-support fraction, Gini coefficient, normalized total variation, 4x4 and 16x16 within-block residual-energy fractions, and signed cancellation ratio. These are descriptively associated with median geometric Spearman/Jaccard, horizontal flip, pixel-to-4x4 and pixel-to-16x16 stability changes, and ROAD Combined at each scale.

Associations use Spearman, per-model strata, and model-equal aggregation. The 95% exploratory interval uses 20,000 finite PCG64 draws, namespace `STAGE_X_X2_MECH_ASSOCIATION_V1_20260910`, linear quantiles, and sample-ID cluster resampling synchronized across models. A degenerate draw with undefined Spearman is skipped and replaced while preserving the same deterministic RNG stream; failure to obtain 20,000 finite draws within 2,000,000 attempts is terminal. Confirmatory p-values and significance conclusions are forbidden.

## 7. Multiscale fidelity

Pixel-level NNS-v2/Grad-CAM ROAD Combined is reused only after sealed-X2 source revalidation and is not recomputed. M1 runs ROAD Combined for 4x4 and 16x16 whole-block rankings using the same fixed clean target, normalized response, 10-90 percentiles, five noisy-linear seeds, MoRF/LeRF directions, SHA256-to-PCG64 tie RNG, and trapezoidal aggregation as X2. Actual removed pixel fraction is recorded. Stage D head-direction Value-path fidelity is never combined with this class-logit analysis. No new superiority/noninferiority threshold is defined.

## 8. Estimability, outputs, and resources

Before M1, 4x4/16x16 transformed stability is `NOT_ESTIMABLE_FROM_RETAINED_X2_EVIDENCE`. After M1 it can become only `POST_HOC_EXPLORATORY_ESTIMABLE`, never X2 confirmatory evidence. Future mandatory outputs are enumerated in the frozen config and typed plan, including map/mask manifests, pixel reproduction, multiscale stability/fidelity, SSE decomposition, transport, clean structure, horizontal-flip, associations, estimability, resource ledger, false-pass audit, inventory, completion, and report.

The sealed X2 totals are 74.0271 H20-GPU-hours and 2313.7848 CPU-hours. The conservative P0 projection applies `1.20 * (12/23 + 4/11)`, representing the geometric share plus two scales x two methods out of 11 X2 methods. Frozen projections are 78.7 H20-GPU-hours, 24 wall-hours on no more than four devices, 2459 CPU-hours, 5 GiB peak reserved memory per process, 4 GiB retained output, and zero download. Hard caps are respectively 80, 48, 2500, 16, 8, four, and zero. Any cap excess terminates `BLOCKED_BUDGET`; workload reduction is forbidden. Four deterministic shards use unit digest modulo four, and all transforms for a route stay on one shard.

## 9. P0-R1 boundary and completion

P0-R1 creates contracts, validators, a typed plan, source bindings, resource projection, estimability record, access audit, false-pass audit, inventory, and atomic completion. Its tests use only synthetic NumPy arrays and masks. P0-R1 imports neither Torch nor timm, loads no model/checkpoint/JPEG, accesses no CUDA/GPU, reads no real map value or per-route outcome row, and computes no real mechanism outcome.

P0-R1 PASS authorizes only `X2_MECH_M1_TRANSFORMED_MAP_AND_MULTISCALE_FIDELITY_ONLY`. It does not execute M1 or M2.
