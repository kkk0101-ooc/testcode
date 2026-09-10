# X2-MECH M1-R1 Digest Recovery Amendment

Date: 2026-09-10

Authorization: `X2_MECH_M1_TRANSFORMED_MAP_AND_MULTISCALE_FIDELITY_ONLY`

## Prior M1 status

The first M1 execution stopped as
`BLOCKED_FROZEN_IMPLEMENTATION_DIGEST_CONVENTION_MISMATCH` before producing any
transformed map, inverse map, mask, or ROAD row. Four deterministic shards
entered their first valid route and failed during source-clean-map binding
validation. The blocked root is immutable and remains read-only at
`evidence/stage_x/x2_mech_m1_20260910/formal`.

The failure was not source drift. A full source audit recomputed 398/398 source
file SHA-256 values and 398/398 sealed X2 canonical array digests successfully.
The frozen M1 worker instead used a different canonical array digest encoding
and matched 0/398 sealed source map digests.

## Prospective correction

M1-R1 changes only the M1 worker/reconciler digest convention and output root:

- canonical array digest is now the sealed X2 convention:
  `dtype.str + NUL + compact JSON shape + NUL + C-order bytes`;
- M1-R1 writes to
  `evidence/stage_x/x2_mech_m1_r1_20260910/formal`;
- M1-R1 report writes to `reports/stage_x_x2_mech_m1_r1_20260910`;
- completion archive ref is
  `refs/archive/stage-x-x2-mech-m1-r1-completion`.

No route, method, transform, target, selected step, checkpoint, preprocessing,
model graph, mask definition, ROAD scale, percentile, seed, threshold, resource
cap, access boundary, or claim boundary is changed.

## Boundary

The prior blocked evidence is not overwritten, retried in place, or relabeled as
PASS. M1-R1 starts from a new root after this prospective amendment. X2 remains
`NO_GO_BROAD_CLAIM_DEVELOPMENT`. X2-MECH remains post-hoc exploratory and cannot
authorize M2 unless M1-R1 completes PASS. X3-X5, F3-Final, and Stage G remain
unauthorized.
