# X2-MECH M2 results-to-paper handoff

Date: 2026-09-10

Status: `PASS`

Next authorization: `F3_FINAL_FIGURE_PACKAGE_ONLY`

## 1. Binding

- M2 completion ref: `refs/archive/stage-x-x2-mech-m2-completion`
- M2 completion commit: `5322ede0fe032838a2211b4de640251a4ece9ad2`
- M2 completion artifact: `evidence/stage_x/x2_mech_m2_20260910/formal/completion/x2_mech_m2.json`
- M2 completion file SHA-256: `a8b8d271f13b414eca4d276993ad39d66fd1098b406cd9f3df64162b45f65e59`
- M2 completion canonical digest: `3e6d7f1f991935ce8633f256bb32785c9ff3ea39826f426a40755733c7272763`
- M2 whole-root inventory: `evidence/stage_x/x2_mech_m2_20260910/formal/whole_root_inventory.json`
- M2 whole-root inventory SHA-256: `25f6a110ba10c201727e683c19a7b17188f6f8902d388395f55feb365a49a313`
- M2 whole-root inventory canonical digest: `9efb2284b0a6f5dbd605e3df5c549f440afd1ed6455002996db0dfc3290143f0`
- M2 inventory files digest: `8ad3ce659fc91b8a52f76c47a692298b87ee13e9c8a419a31d6b69fcc837b322`
- M2 terminal report: `reports/stage_x_x2_mech_m2_20260910/x2_mech_m2_report.md`
- M2 terminal report SHA-256: `406a339ec514ddb4279d7632692c4f3178fcbc15014141342be5ddaa9a626cfe`
- Bound upstream handoff: `docs/stage_x_x2_mech_m1_to_m2_scientific_handoff_20260910.md`

This document supersedes the M1 to M2 handoff only for paper-facing use of M2
results. It does not overwrite that handoff, the mechanism protocol, M1-R1, X2,
Stage D, Stage E, Stage F/F3C, or any archived evidence.

## 2. Observation

M2 completed as a CPU-only, source-recomputing, model-free analysis. It
validated 199 clean-valid routes, retained the one immutable
`FAILED_ROUTE_LOCAL_NUMERICAL` denominator record, reproduced 4,776 sealed X2
pixel stability rows, and analyzed 28,800 multiscale stability rows, 19,200
within/between SSE rows, 28,656 transport-control rows, 2,388 horizontal-flip
control rows, 400 clean-structure rows, and 12 exploratory association rows.

For `x2_frozen_common_domain`, median signed Spearman for NNS-v2 increased from
pixel `0.11366648574801438` to 4x4 `0.2177360373571473` and 16x16
`0.33779091119911697`. For ViT Grad-CAM, the corresponding medians were pixel
`0.6470336100253422`, 4x4 `0.6532370714163442`, and 16x16
`0.7218010653661862`.

M1-R1 multiscale class-logit ROAD Combined medians were NNS-v2 4x4
`0.05496294885494901`, NNS-v2 16x16 `0.06067247007244031`, Grad-CAM 4x4
`0.1499317089177042`, and Grad-CAM 16x16 `0.15329919519210142`.

For NNS-v2 within/between SSE on `x2_frozen_common_domain`, the 4x4 median
between fraction was `0.07447684921684658` and within fraction was
`0.9255231507831534`; the 16x16 median between fraction was
`0.0026145042858070795` and within fraction was `0.997385495714193`. The maximum
SSE identity residual was below `5.5e-12`.

The horizontal-flip transport-control median signed Spearman was `1.0`. This is
a model-free geometric resampling control; observed-minus-transport is not a
causal effect.

Exploratory association intervals reported effects and 95% intervals only. For
NNS-v2, absolute top-10% pixel mass share versus median pixel signed Spearman was
`-0.31235607781179026` with interval `[-0.4529167946038158,
-0.13814825434420727]`; normalized total variation versus pixel-to-4x4 signed
Spearman change was `0.2538214673644712` with interval `[0.09650361396139384,
0.3920831264942336]`. One Grad-CAM association was
`NOT_ESTIMABLE_DEGENERATE`; no p-values were computed.

## 3. Interpretation

M2 selects `MIXED_OR_INCONCLUSIVE`. The observed direction is compatible with
coarser aggregation improving geometric rank stability for NNS-v2, but the
frozen protocol did not define numeric thresholds that can uniquely choose
Branch A, B, or C. The paper may therefore describe these results as post-hoc
exploratory multiscale evidence and may discuss a possible
granularity-fidelity-equivariance trade-off only as a bounded interpretation.

## 4. Claim Boundaries

Allowed claims:

- X2-MECH M2 is a `POST_HOC_EXPLORATORY_MECHANISM_ANALYSIS`.
- M2 provides source-recomputed multiscale stability, within/between SSE,
  transport-control, clean-structure, and exploratory association evidence.
- M2 suggests mixed post-hoc evidence about whether aggregation accounts for the
  observed geometric instability.
- Stage D remains the primary positive paper result in its original scoped
  fidelity contract.

Forbidden claims:

- No spatial-stability PASS.
- No NNS-v2 superiority or noninferiority to Grad-CAM.
- No broad benchmark GO.
- No localization claim.
- No cross-model or cross-dataset generalization claim.
- No confirmatory p-value or significance claim from M2.
- No causal conclusion from observed-minus-transport values.
- No modification of X2 `NO_GO_BROAD_CLAIM_DEVELOPMENT`.

## 5. Paper Placement

M2 should enter the paper as a short Discussion or Supplement mechanism
analysis with provenance appendix links. It should not be promoted to the main
confirmatory Results claim. The main text may cite it to explain why the paper
keeps a scoped fidelity conclusion and does not claim broad geometric
equivariance.

## 6. F3-Final Figure/Table Modules

F3-Final may prepare the following modules without assigning final figure
numbers in advance:

- A compact multiscale stability table for NNS-v2 and Grad-CAM at pixel, 4x4,
  and 16x16 scales.
- A within/between SSE decomposition panel for 4x4 and 16x16.
- A transport and horizontal-flip control table or supplement panel.
- A concise association table with effect and 95% exploratory interval only.
- A provenance appendix card binding M1-R1, M2 completion, source validation,
  access audit, and immutable X2 NO-GO.

## 7. G0/G1 Evidence-Matrix Entries

- `X2_MECH_M2_MULTISCALE_STABILITY`: M2 PASS; use as post-hoc exploratory
  mechanism evidence; branch `MIXED_OR_INCONCLUSIVE`; do not use for broad GO.
- `X2_MECH_M2_WITHIN_BETWEEN_SSE`: M2 PASS; exact SSE decomposition validates
  within/between accounting; do not call rank metrics additive decompositions.
- `X2_MECH_M2_TRANSPORT_CONTROL`: M2 PASS; model-free transport baseline;
  observed-minus-transport is descriptive and non-causal.
- `X2_MECH_M2_ASSOCIATION`: M2 PASS; report Spearman effects and 95%
  exploratory intervals only; no p-values.

## 8. Stop Condition

This handoff authorizes only `F3_FINAL_FIGURE_PACKAGE_ONLY`. F3-Final, Stage G,
X3, X4, and X5 were not started by M2 or by this handoff.
