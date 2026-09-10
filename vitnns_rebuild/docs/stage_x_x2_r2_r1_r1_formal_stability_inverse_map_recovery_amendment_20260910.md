# Stage X X2-R2-R1-R1 Formal Stability Inverse-Map Recovery Amendment

This prospective amendment authorizes only `X2_R2_R1_R1_FORMAL_STABILITY_INVERSE_MAP_SHAPE_RECOVERY_ONLY`.

X2-R2-R1 remains immutable and terminal as `BLOCKED_INFRASTRUCTURE`. Its roots, terminals, receipts, schemas, run specs, implementation seals, clean/gamma outputs, and `stage_failure.json` must not be modified, overwritten, deleted, or reclassified as PASS.

The only scientific-code repair authorized in this recovery is shape handling inside `stage_x_x2_metrics.inverse_map()` for 2D attribution maps passed through stability inverse transforms. Rotate, resize, and horizontal flip may temporarily add channel/batch dimensions for torchvision compatibility, but must return a restored map of shape `(224, 224)` and a mask of shape `(224, 224)`. The transform set, metrics, thresholds, gamma grid, selected gamma, route roster, checkpoints, NNS-v2 definition, cuts, schedule, tolerances, comparators, margins, futility rule, resource caps, and denominator remain unchanged.

X2-R2-R1-R1 uses fresh roots:

- `evidence/stage_x/x2_r2_r1_r1_nns_v2_development_20260910/formal/`
- `reports/stage_x_x2_r2_r1_r1_nns_v2_development_20260910/`

The new root may source-validate and reuse X2-R2-R1 clean coverage and all 200 gamma terminals. Large scientific arrays remain in their sealed source roots and must not be copied into the new root. Gamma selection is rebound to the source-validated selection: `conv_gamma = 100.0`, `linear_gamma = 0.05`.

The denominator is always reported as 200. Formal paired analysis includes only clean `VALID` routes with valid endpoints for both paired methods. The prior failed clean route remains `FAILED_ROUTE_LOCAL_NUMERICAL` and is excluded from paired-valid effect analysis.

If formal, reconciliation, margins, futility, resource, budget, or false-pass audit does not PASS, X2-R2-R1-R1 must publish an immutable terminal `NO_GO` or `BLOCKED` receipt with `next_authorization = NONE`. X3, X4, X5, X6, F3-Final, and Stage G remain unauthorized.
