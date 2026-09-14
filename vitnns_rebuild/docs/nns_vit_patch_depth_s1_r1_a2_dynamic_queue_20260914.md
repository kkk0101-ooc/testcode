# S1-R1-A2 dynamic unit-claim amendment

Status: `FROZEN_PROSPECTIVELY_AFTER_INTERMEDIATE_BEFORE_ANY_FULL_OR_RECOVERY_LAUNCH`. Owner authorization: 2026-09-14.

At freeze, schedule indices 0–23 already have immutable terminals: 23 `VALID` and one `NA_VOID_MASK`; smoke 0–5 and every completed intermediate unit remain unchanged and must never be recomputed, overwritten, deleted, or replaced. The static K=8 intermediate launch has ended with all eight shard terminals and no pending unit. A2 does not retrospectively rewrite its scheduling history.

For any later full-pilot or recovery launch, one fixed set of K=1..8 qualifying GPUs is locked once. Each GPU owns one long-lived worker. Workers repeatedly claim the smallest not-yet-terminal candidate from a global launch queue. Claim creation uses exclusive atomic file creation plus file and directory fsync, so one index has at most one claimant in a launch. GPU membership never changes within a launch.

Claims are launch-scoped. A crash recovery creates a fresh launch root; stale claims remain immutable evidence but do not block the new launch. Before every claim, all immutable unit terminals across smoke, intermediate, full, and prior dynamic launches are recomputed; completed indices are skipped. Unit seeds and scientific identity remain functions of `schedule_index` and `source_image_id`, never worker, GPU, claim order, or runtime.

Each dynamic unit terminal records claim identity and time, worker ID/PID, physical GPU/UUID, start and end time, and measured unit resources. Final resource accounting stable-sorts and sums unit terminal telemetry with `math.fsum`; it does not infer work from a static shard assignment. This amendment changes execution scheduling only. Data access, sample set, target, component, controls, seeds, tolerances, NA propagation, P1/P2/D1/D2 definitions, scientific boundary, and the A1 aggregate P2 gate of 0.90 are unchanged.
