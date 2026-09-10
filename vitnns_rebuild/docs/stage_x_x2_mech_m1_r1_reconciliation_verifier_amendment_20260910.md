# X2-MECH M1-R1 Reconciliation Verifier Amendment

Date: 2026-09-10

Authorization: `X2_MECH_M1_TRANSFORMED_MAP_AND_MULTISCALE_FIDELITY_ONLY`

## Context

M1-R1 empirical execution completed its immutable route attempts under the
pre-execution seal for the digest recovery amendment. During independent
source-recomputing reconciliation, the verifier rejected ROAD
`actual_pixel_fraction` values because it compared a float32 mean-derived
record against an exact float64 rational with `1e-15` tolerance.

The maximum observed discrepancy was below `3e-8`, matching float32
representation of `1 - pixel_keep.mean()`. This is a reconciliation verifier
precision issue, not a route-output, ROAD, map, mask, source-binding, or budget
change.

## Prospective verifier correction

This amendment permits only the following reconciliation-verifier changes:

- compare `actual_pixel_fraction` with `abs_tol=rel_tol=1e-7`;
- permit an existing valid pending completion receipt to be reused when a
  reconciliation attempt failed before writing aggregate M1 outputs.

No worker code, route terminal, map file, mask file, source binding, ROAD row,
model output, transform, seed, route, method, scale, metric, threshold, budget,
or claim boundary is changed. M2 remains unauthorized.

The verifier must bind this amendment in the final M1-R1 validation evidence and
must continue to recompute all route terminals and raw files independently.
