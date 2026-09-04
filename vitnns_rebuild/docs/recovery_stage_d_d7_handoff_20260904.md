# Stage D recovery-v2 D7 completion and future D8.0 handoff

D7 completed as an execution-authorization gate PASS, not an execution,
analysis, validity, effect, cost-result or scientific PASS. The corrected
producer is `e337f8cae6bc395c08f88f1d4c85b97c6015e994`; its authority-only direct
child is `fe9a7f29b57fe95884407800dddafd0d1f249c17`. The sole current authority is
`config/stage_d_d7_execution_authorization_20260904_v2.json`.

The first producer/authority pair `d077733d...` / `0036dd8d...` is preserved as
pre-execution history and is superseded. A strict D7 guard showed that its
generic recovery-model verifier performed an unnecessary `lstat` on a known
Stage A calibration JPEG. It did not access a Stage D reserve image, run model
inference, initialize D8, or produce a Stage D result. The corrected model-only
verifier reads only the recovery model file, its LFS listing, and tensor bytes.

The successful synthetic smoke is non-formal and contains 1 synthetic sample,
2 method units, 2 clean maps, 20 curves and 420 curve points (422 logical work
items). Complete smoke roots, including two failed attempts and one successful
pre-independent-validation attempt, remain local under
`evidence/stage_d/d7_recovery_v2_20260904`; only the small successful receipt is
tracked. None may enter D8–D10 evidence.

The D7 budget uses sealed recovery C9 H20 telemetry and the actual 1,000-unit
typed plan. With the frozen 1.25 factors it projects 169.29855772962784 H20
GPU-hours, 0.6704893375430806 hours per image group, and 2,104,759,250 retained
bytes. Every launch and completed-campaign preflight cap passes. Eight shards
are frozen by `selection_order modulo 8`, with group counts
`[63,63,63,63,62,62,62,62]`.

The fidelity-only scope is unchanged: independent `n=500`, primary `(0,9)`,
the two frozen methods, endpoint `fine_4x4_cell_delta_fid`, image statistical
unit, no stability endpoint, no secondary inference and no pooling with C10
`n=15`. C10 remains `BLOCKED` / `NO_GO_NOVELTY`, translation and crop remain
`FAIL_STABILITY`, overall spatial remains `FAIL_SPATIAL`, and Holm remains
`HOLM_BLOCKED_UNDERSPECIFIED`.

D8, D9 and D10 have not started. The formal D8 root remains absent. No Stage D
reserve image path was checked for existence, statted, opened, hashed, decoded,
preprocessed or inferred in D7. Analysis authorization remains `NOT_EVALUATED`;
reserve-byte access is deferred to a future, separately executed D8.0.

Future D8.0 must run from the corrected producer archive ref in a clean detached
worktree and independently revalidate producer/authority refs, authorization
bytes, recovery model identity, plan, budget, shard mapping, pre-authorization
attestation and fresh formal root. D8.0 may initialize that root and perform its
durable access-start/live-asset boundary, but it must not run an execution unit.
