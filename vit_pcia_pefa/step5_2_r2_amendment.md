# Step 5.2 r2 prospective scope and numerical amendment

Status: authorized on 2026-09-22; effective only after the new r2 freeze receipt
is written and before any r2 outcome is read.

## Reason and immutable predecessor

The two-model r1 root remains immutable. Its P0 clean parity, complete-cut
closure, nonzero effect and dose order passed, but the frozen 256-to-512
quadrature convergence gate failed. This is a numerical failure and carries no
scientific decision about prediction quality. No r1 P1 prediction or
intervention was started.

Outcome-blind discovery diagnostics showed materially smaller global relative
L2 differences for DeiT-S/16 than for ViT-B/16. The user therefore authorized
a prospective DeiT-S/16-only study. ViT-B/16 is excluded from r2 because its
numerical convergence remained less favorable, not because of a scientific
result. The r1 failure and all four diagnostic receipts are retained and bound
as predecessor evidence.

## r2 changes

- Model scope is exactly `deit_s16`. P0 has one real-model task and P1 has
  eight image tasks for calibration `split_order=16..23`.
- The real-model quadrature pair is frozen at 1024 and 2048 Gauss--Legendre
  nodes for every P0/P1 image. It cannot be varied by sample or result.
- Convergence of source-summed head R, full IG and full G is gated on global
  relative L2 error at most 0.01 for non-near-zero tensors, or max absolute
  error at most 0.0001 for genuinely near-zero tensors. Elementwise parity is
  retained as a diagnostic but is not the map-level convergence gate because
  signed maps contain many locally near-zero coordinates.
- All prediction thresholds, strongest-baseline comparison, bootstrap plan,
  intervention construction, calibration cohort, resource caps and
  prediction-before-intervention boundary remain unchanged.

## Claim boundary

Every r2 decision is DeiT-S/16-only. Even a full pass may support only a claim
about this frozen DeiT-S/16 instance and block-5 head set. It cannot support
cross-architecture replication, a general ViT claim, automatic holdout,
production, evaluation access, Step 6, other blocks/modules, 14x14, or
pixel-level work. Step 5 and Step 5.1 conclusions remain unchanged.
