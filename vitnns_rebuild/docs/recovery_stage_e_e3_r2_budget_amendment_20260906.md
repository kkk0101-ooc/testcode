# Stage E E3-R2 prospective budget-continuation amendment — 2026-09-06

- Amendment ID: `STAGE-E-E3-R2-BUDGET-CONTINUATION-20260906`
- Status: `APPROVED_FOR_E3_R2_COMPLETION_ONLY`
- Scientific contract change: none
- Numerical recovery change: none from E3-R1
- Budget-only change: 128 H20 GPU-hours, 512 CPU-hours, 32 GiB retained
- E4 authorization: none until a complete terminal E3-R2 PASS

This amendment records the owner's explicit decision that the original E3
budget is not a limiting objective and authorizes completion under the enlarged
limits above. It does not convert E3 v1 `FAIL` or E3-R1 `BLOCKED` into PASS,
does not overwrite either receipt, and does not retroactively claim that the
R1 budget gate passed.

E3-R2 binds read-only custody commit
`b3855edc935fe7d2558fd328b2f6b774fa0f8bfc`, tree
`689e7fd52da25ff3b1c02afe5d2d6583ce479f50`, and archive ref
`refs/archive/stage-e-e3-r1-budget-blocked`. It must source-validate the E3 v1
516-attempt partition and the twelve valid E3-R1 pilot attempts. Only the exact
2,472-unit complement without a legal terminal result may execute in the fresh
root `evidence/stage_e/e3_r2_recovery_v2_20260906/formal`.

The frozen R1 numerical schedule remains
`[512,1024,2048,4096,8192,16384,32768]`: each eligible NNS unit must execute
the original five-step prefix first, then 16384 and 32768 only as ordered
fallbacks, stopping at the first pass. Conservation remains
`abs_error <= 1e-4 + 0.01*abs(delta)`. No sample, coordinate, method, target,
model, preprocessing, baseline, seed, fidelity fraction, AUC, delta, bootstrap,
stratum, denominator, missingness, or reporting contract may change.

All E3 v1 and R1 costs, failures, interruptions, and retained artifacts remain
charged and are added to R2 costs. The enlarged limits are hard caps, not a
waiver. A runtime cap breach or any terminal scientific failure stops new work
and authorizes nothing downstream.

E3-R2 PASS requires a pairwise-disjoint union of 516 E3 v1 attempts, 12 R1
pilot attempts, and 2,472 R2 attempts: all 3,000 new-compute units must be
valid, terminal, source-recomputed, and present exactly once. It also requires
the unchanged 1,000 read-only primary method rows, 4,000 total method rows,
2,000 paired rows, twelve coordinate-stratum summary cells, all three enlarged
budget gates, and an explicit false-PASS audit. Only its independent terminal
PASS receipt may authorize E4, and E4 is not started by this amendment.
