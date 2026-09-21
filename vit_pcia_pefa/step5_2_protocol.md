# Step 5.2 r2/r3 prospective DeiT-S/16 predict-then-intervene protocol

- Status becomes `FROZEN_PROSPECTIVE` only after `freeze_step5_2.py` succeeds.
- Active recovery root: `vit_pcia_pefa/step5_2_r3_20260922/`.
- Scope is prospectively narrowed to DeiT-S/16 under
  `step5_2_r2_amendment.md`; no cross-architecture replication claim is
  available. The two-model r1 root and its numerical failure remain immutable.
- r2 passed P0 but is immutable after a prediction-worker task-ID plumbing
  failure with zero accepted predictions and no interventions. r3 applies only
  that infrastructure repair and rebinds the validated r2 P0 evidence.
- Scope is Step 5.2 P0 and P1 only. Evaluation, module/block expansion, holdout,
  production and Step 6 remain unauthorized. Step 5 and Step 5.1 terminal
  conclusions remain immutable.

## Frozen object and intervention

For each DeiT-S/16 image, the target is the ground-truth raw logit, the
reference is normalized zero, the path is straight, and the complete cut is
`HEADS_WITH_BYPASS_FULL` at block 5. Only like-for-like attention heads in
that block are compared. For head `e`, the complete normalized-residual to
pre-projection head-output mapping is `h_e`. At its clean input `z*`, use

```text
h_e,lin(z) = h_e(z*) + J_h_e(z*) (z-z*)
h_e,lambda(z) = h_e,lin(z) + lambda (h_e(z)-h_e,lin(z))
```

with `lambda in {1, 0.5, 0}`. Lambda 1 is the unmodified model and is stored
once as the clean prediction phase; interventions run only 0.5 and 0. Each
run changes one head. Clean target logit, prediction, input gradient, Endpoint
and complete-cut identities must retain the numeric tolerance.

## P0 numeric feasibility

A deterministic CPU-float64 two-block ViT fixture and discovery
`split_order=0` in DeiT-S/16 are development-only. The fixture uses 32 versus
64 Gauss--Legendre nodes and the real model uses the prospectively frozen
1024 versus 2048 pair for every image.
Source-summed head `R_e`, full IG and full `G` must each converge under the
frozen global-relative-L2 or absolute-near-zero convergence rule; elementwise
differences remain diagnostic. One head is tested at lambda
0.5 and 0; clean parity and complete-cut closure must pass, lambda 0 must
produce nonzero `Y`, and weakening must be dose ordered (`Y(0.5) <= Y(0)`).
P0 emits no scientific claim. P1 cannot start unless the fixture and the one
real-model P0 task pass.

## P1 prospective calibration pilot

The cohort is the mechanically selected calibration interval
`split_order=16..23`, unused by Steps 5 and 5.1, in `deit_s16` only. Selection
uses only pre-existing split metadata. There are eight image tasks.
`top_k=ceil(0.25*num_heads)`. No evaluation member may be read.

The prediction phase runs the original model only and atomically seals, per
image and head:

- `s_R=||R_e||_1`, `s_A=||A_e||_1`, `s_Q=||Q_e||_1`;
- local cotangent L2 norm, activation L2 norm and SHA-256 random score;
- signed `R_e`, `A_e`, `Q_e` input-coordinate maps, plus full `G`, IG, input
  gradient and Endpoint arrays needed for later checks.

`prediction_freeze_receipt.json` and its artifact hashes must exist before any
intervention process starts. Modified values, deltas, logits and outcomes are
never inputs to a predictor.

For each head and lambda 0.5/0, the primary outcome is

```text
Y_e(lambda) = ||G_e(lambda)-G(1)||_1 / (||G(1)||_1 + 1e-12).
```

Per image, compare each frozen score with `Y_e(0)` using Spearman correlation,
top-1 hit and top-k effect capture. Ties in score ordering use ascending head
index only for deterministic reporting; Spearman uses average ranks. Also
report dose order and cosine/sign agreement between `-R_e` and actual
`delta G_e(0)`. Direction and dose do not substitute for the primary gate.

## Frozen gates and uncertainty

Image is the statistical unit and the decision is limited to DeiT-S/16. A
deterministic image-level bootstrap uses 20,000 resamples and the percentile
95% interval. An undefined Spearman caused by a constant score or outcome is
frozen as zero for gating and is flagged in the per-image record. Undefined
cosine or sign agreement is likewise reported and contributes zero to its
image summary. For DeiT-S/16:

1. `NATURAL_GAP_PREDICTION` requires mean PEFA-R Spearman at least 0.20 and
   its 95% lower bound strictly above zero.
2. The strongest simple baseline is the non-PEFA method with the greatest
   model-level mean Spearman, ties resolved by the frozen method order
   `pcia_A, endpoint_Q, gradient_norm, activation_norm, random`.
3. `PREDICTIVE_INCREMENT_OVER_SIMPLE_BASELINES` requires the paired mean
   PEFA-minus-strongest-baseline Spearman difference to be at least 0.10 and
   its 95% lower bound strictly above zero for DeiT-S/16.
4. `DOSE_ORDER` requires `Y(0.5) <= Y(0)+1e-8` for at least 90% of valid
   image-head pairs. `DIRECTIONAL_ALIGNMENT` reports bootstrap
   intervals for image-level mean cosine and sign agreement without opening
   the primary gate.
5. `NUMERICAL_VALIDITY_AND_COST` requires all tasks, convergence, parity,
   closure, finite values and frozen resource caps to pass.

If PEFA-R fails prediction in DeiT-S/16, the terminal result is
`NOT_SUPPORTED_NATURAL_GAP_PREDICTION`. If it predicts but fails incremental
advantage, it is `NOT_SUPPORTED_PREDICTIVE_INCREMENT`. Numeric
or intervention failure is `INVALID_NUMERICAL_OR_INTERVENTION` and may only be
recovered prospectively in a new root. Even a primary pass is DeiT-S/16-only
and does not establish cross-architecture replication or authorize a holdout.

## Numeric, reproducibility and resources

- Fixture: float64, parity `atol=1e-9`, `rtol=1e-7`, `near_zero=1e-12`;
  convergence `atol=1e-7`, `rtol=1e-4`, `near_zero=1e-12`.
- Real model: float32 forward/autodiff and float64 reductions; parity
  `atol=1e-5`, `rtol=1e-4`, `near_zero=1e-6`; convergence `atol=1e-4`,
  `rtol=1e-2`, `near_zero=1e-6` under the global-relative-L2 or
  absolute-near-zero rule. TF32 and autocast are off.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA import. Executable
  realpath, GPU UUID, driver, hashes, deterministic flags and RNG states are
  recorded. Startup includes CUDA matmul plus backward.
- At most four eligible H20 workers; minimum 32 GiB free memory and at most
  10% utilization at launch. Per worker caps are 48 GiB reserved GPU memory,
  32 GiB RSS, two attempts and four hours per task. Retained output is capped
  at 128 GiB. Step-5.2 is capped at 32 H20-hours and cumulative Step 5 at 64
  H20-hours. Existing processes are recorded and never killed.

Claims are limited to `NATURAL_GAP_PREDICTION`,
`PREDICTIVE_INCREMENT_OVER_SIMPLE_BASELINES`, `DOSE_ORDER`,
`DIRECTIONAL_ALIGNMENT`, and `NUMERICAL_VALIDITY_AND_COST`. No result may be
described as an independent information source beyond joint `(Q_e,A_e)`, a
better saliency method, a unique causal source, or a reversal of prior Step 5
or Step 5.1 conclusions.

## Sealed r3 result

r3 completed as `NOT_SUPPORTED_PREDICTIVE_INCREMENT`. For DeiT-S/16, PEFA-R
mean Spearman was 0.8286 (95% CI 0.7714 to 0.8857), while the strongest simple
baseline PCIA-A was 0.7786. The paired PEFA-minus-PCIA mean was 0.0500 (95% CI
-0.0214 to 0.1714), below the frozen 0.10 minimum and with a nonpositive lower
bound. Natural-gap prediction, dose order, directional alignment and numerical
validity/cost were supported; predictive increment was not supported. The
prompt-to-artifact audit passed. This terminal result is model-specific and
does not authorize any follow-on stage.
