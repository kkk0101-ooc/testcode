# Stage E E3-R3 prospective numerical-stability amendment — 2026-09-07

- Amendment ID: `STAGE-E-E3-R3-NUMERICAL-STABILITY-20260907`
- Status: `APPROVED_FOR_E3_R3_GATED_RECOVERY_ONLY`
- Scientific contract change: none
- Threshold change: none (`atol=1e-4`, `rtol=0.01`)
- E4 authorization: none until a complete terminal E3-R3 PASS

This amendment preserves E3 v1 `FAIL`, E3-R1 `BLOCKED`, and E3-R2 `FAIL` as
immutable read-only history. It neither overwrites their receipts or attempts
nor relabels a historical failure as valid.

The formal pilot is exactly selection order 117, coordinate `(11,2)`, NNS
Value-path IG. The model, sample, target, fixed attention, direction, baseline,
preprocessing, seeds, intervention/fidelity curves, statistics, denominators,
and reporting rules remain unchanged. Model forward/backward computation and
stored attribution/map dtypes remain float32. Only quadrature nodes/weights and
gradient integration accumulation change to float64, using SciPy 1.12
`roots_legendre` and Kahan per-chunk accumulation.

The prospective schedule is `[65536,131072,262144]`, attempted in order and
stopped at the first conservation pass. It follows the already preserved R2
failure through 32768 and does not rerun or replace that attempt. Every R3 run
uses `abs_error <= 1e-4 + 0.01*abs(delta)`.

Pilot PASS requires two independent R3 diagnostic executions to converge, use
the same selected step count, have exactly matching target/input identities,
and satisfy map repeatability `atol=1e-6, rtol=1e-5`. Both runs are formally
charged and retained. Neither alone is a replacement attempt. Only after the
repeatability gate passes may one fresh R3 production attempt be created and
the remaining complement execute.

The final union must contain exactly 3,000 valid terminal new-compute units:
1,164 inherited valid attempts from E3 v1/R1/R2 and 1,836 new R3 attempts. The
R2 order-117 failure remains visible in history but is not silently overwritten.
Any R3 scientific failure, repeatability failure, missing/duplicate unit, or
contract drift makes PASS impossible and authorizes neither remaining work nor
E4.
