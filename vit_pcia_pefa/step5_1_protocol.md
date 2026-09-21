# Step 5.1 port-localization prospective protocol

- Status: `FROZEN_PROSPECTIVE` after `freeze_step5_1.py` succeeds.
- Root: `vit_pcia_pefa/step5_1_r1_20260921/`.
- Scope: Step 5.1 only. Evaluation remains unread and Step 6 remains unauthorized.
- Prior evidence: Step 5 remains `NOT_SUPPORTED_EMPIRICAL_VALUE`; the extension remains
  `PARTIAL_NOT_SUPPORTED_WITH_E3_INVALID`. Neither decision may be rewritten here.

## A0 numerical feasibility

A deterministic CPU-float64 two-block ViT fixture and discovery `split_order=0` in each
registered model are development-only. The frozen complete cut is
`HEADS_WITH_BYPASS_FULL` at block 5 in real models and block 1 in the fixture. The
intervention is applied to the actual `attn.proj` input consumer use and one head slice.
For clean activation `z*`, selected slice `z`, detached RMS scale `s >= 1e-6`, and
strength `alpha`, the wrapper is

```text
z + alpha * s * tanh(((z-z*)/s)^2).
```

It is bounded, equals `z` at `z*`, and has identity Jacobian there. Strengths are
`0.25, 0.5, 1.0`; exact zero and the clean-point affine tangent are controls. A0 passes
only if clean logit, prediction, Endpoint, terminal port gradients/Endpoint quantities,
and all five complete-cut identities retain the frozen tolerance; zero/tangent controls
have no localization signal; injected PEFA-R is non-zero at `alpha=0.25`; and the true
port PEFA-R delta is nondecreasing across the three strengths. A0 produces no scientific
claim.

## A1 controlled localization pilot

The sample is the mechanically selected, previously unused-in-Step-5 calibration prefix
`split_order=8..15`, for both `vit_b16` and `deit_s16`. Target is the ground-truth raw
logit, reference is normalized zero, path is straight, block is 5, cut is
`HEADS_WITH_BYPASS_FULL`, and Gauss--Legendre quadrature has 512 nodes. For each image,
the primary head is `(split_order-8) mod num_heads` and the predeclared decoy is the next
head modulo `num_heads`. Primary strengths `0.5` and `1.0` are reported separately. A
same-strength decoy-port injection, affine-tangent wrapper, and zero-strength wrapper are
negative controls.

The primary score is per-head `L1(delta R_e)`. Comparators include both change-aware and
level-only alternatives: per-head `L1(delta A_e)` and clean `L1(A_e)` (PCIA),
`L1(delta Q_e)` and clean `L1(Q_e)` (Endpoint), `L1(delta D_e)` and clean `L1(D_e)`,
activation-norm change and clean activation `L2`, plus a SHA-256 deterministic random
ordering. Image is the statistical unit. For every method
and condition report true-port top-1, top-k with `k=ceil(0.25*num_heads)`, conservative
rank, reciprocal rank, percentile rank, paired difference from PEFA, `delta R_e`, and
complete-cut `delta G` closure/direction. Ties receive their worst rank.

`CONTROLLED_CURVATURE_LOCALIZATION` is `SUPPORTED` only when, in each model and at each
primary strength, PEFA-R top-1 is at least 0.75, top-k is at least 0.875, and mean
percentile rank is at least 0.90; the same criteria locate the actually modified decoy
port; and zero/tangent controls emit no score above numerical tolerance.
`INCREMENTAL_OVER_SIMPLE_BASELINES` is `SUPPORTED` only when, in each model and at each
primary strength, PEFA-R exceeds every comparator by at least 0.05 in mean percentile
rank and by at least 0.125 in top-1 rate. Otherwise a numerically valid A1 is
`NOT_SUPPORTED`. Only if both claims and `NUMERICAL_VALIDITY_AND_COST` are `SUPPORTED`
may B1 run.

## B1 natural nonlinearity and C1 attribution audit

B1 is frozen but gated on A1. It uses the same sample, target, reference, path and block.
The module classes are attention softmax, QK matmul, AV matmul, GELU, and LayerNorm;
linear projection and residual add are low-curvature controls. Each nonlinear module is
replaced at its clean activation by `h_lin + lambda*(h-h_lin)`, with
`lambda in {1,0.5,0}`. PEFA scores are frozen before weakened outcomes are read. The
claim requires clean parity, monotone dose direction, positive rank association between
pre-weakening PEFA-R and absolute `delta G`, and at least 0.05 mean-rank-correlation
advantage over every A1 comparator in both models. Failure in either model is
`NOT_SUPPORTED_NONLINEARITY_LOCALIZATION`; no module changes, sample deletion or
resolution escalation are permitted.

C1 is independent after valid A0. Previously used mutations remain development controls.
Its new held-out mutation is `consumer_head_slice_rotation`: the consumer slice IDs in the
machine-readable cut are cyclically reassigned while values and the forward remain
unchanged. Expected failure is `FAIL_PORT_ID_SEMANTIC_MISMATCH`, localized to block 5,
`HEADS_WITH_BYPASS_FULL`, and the head-port category. C1 has its own
`SUPPORTED / NOT_SUPPORTED / INVALID` decision and cannot substitute for B1.

## Numeric, reproducibility and resource contract

- CPU fixture is entirely float64 with `atol=1e-9, rtol=1e-7, near_zero=1e-12`.
- Real models use float32 forward/autodiff and float64 reductions with
  `atol=1e-5, rtol=1e-4, near_zero=1e-6`; TF32/autocast are off.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA import. Deterministic algorithms,
  executable realpath, driver/GPU UUID, hashes and RNG states are recorded.
- Step-5.1 cap is 32 H20-hours; cumulative Step-5 cap remains 64 H20-hours. Maximum four
  eligible H20 workers, 40 GiB reserved GPU memory and 32 GiB RSS per worker, 2 attempts,
  7200 seconds per task, 128 GiB retained total. Existing processes are never killed.
- Any source-binding, clean-parity, closure, nonfinite or cap failure is `INVALID`, not a
  scientific negative. No evaluation image may be read.

Only the five claims listed in the execution plan are sealed. Even a full pass merely
allows a request for a fresh confirmation experiment; it does not authorize Step 6.
