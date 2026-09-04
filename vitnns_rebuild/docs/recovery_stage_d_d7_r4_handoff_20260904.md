# Stage D recovery-v2 D7-R4 completion and future D8.0 handoff

D7-R4 is the sole current D7 execution-authorization gate record. It is not a
Stage D execution, analysis, scientific-result, validity, effect or cost PASS.
Its producer is `d7c88df66f3e55ec31ad4c6ea34a3002ca6d1a4d`; its authority-only
direct child is `61214b808b75cf804763078479d1b96f7418db32`.

D7-R4 supersedes the uncommitted D7-R3 terminal candidate. R3 remains useful as
synthetic D4-D6 full-chain readiness evidence, but it is `UNSUPPORTED_PASS` as a
current D7 gate because the frozen R3 producer did not implement the D8.0
`ACCESS_STARTED` lifecycle, append-only access journal, live-asset validation,
blocked failure state and final `RUNNABLE` transition required before any future
formal shard can run. The rejected R3 terminal candidate is retained only under
`evidence/stage_d/d7_recovery_v2_20260904/rejected_r3_terminal_candidate/`.

R4 adds and validates that D8.0 initializer in synthetic filesystem fixtures. The
R4 smoke proves durable `ACCESS_STARTED`, per-asset intent before callback,
append-only journal source recomputation, interruption resume without duplicate
validation, fail-closed `BLOCKED`, and `RUNNABLE` only after all 500 asset
contracts validate. No unit attempt directory is created by initialization.

The frozen fidelity-only scope is unchanged: n=500, primary `(0,9)`, the two
pre-registered methods, endpoint `fine_4x4_cell_delta_fid`, image statistical
unit, no stability endpoint, no secondary inference, no C10/Stage D pooling.
C10 remains `BLOCKED` / `NO_GO_NOVELTY`, with translation and crop failures,
overall spatial failure and `HOLM_BLOCKED_UNDERSPECIFIED` preserved.

D8, D9 and D10 have not started. The formal D8 root is absent. D7-R4 did not
check existence, stat, open, hash, decode, preprocess or infer any Stage D
reserve image. A future D8.0 task must revalidate the R4 producer ref, authority
ref, authorization bytes, recovery model, plan, budget, shard mapping, fresh
formal root and pre-authorization attestation before creating the formal root and
running the access initializer. D8.0 still must not execute any unit.
