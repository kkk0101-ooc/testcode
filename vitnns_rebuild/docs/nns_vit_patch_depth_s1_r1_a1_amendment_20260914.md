# S1-R1-A1 prospective execution amendment

Status: `FROZEN_PROSPECTIVELY_BEFORE_INTERMEDIATE_ACCESS`. Scope: S1-R1 intermediate/full launch policy and the aggregate P2 matched-control engineering gate only. Owner authorization: 2026-09-14.

At amendment time, smoke schedule indices 0–5 are complete: 6/6 unit terminals are `VALID`, 3/3 shard terminals are `PASS`, no pending receipt remains, and no S1 writer has the evidence root open. Their SHA-256 and canonical digests are bound in `smoke_baseline_binding.json`; these units must not be deleted, overwritten, recomputed, or replaced. No `probe_train`, `probe_validation`, or `confirmatory_holdout` image, mask, feature, logit, map, or outcome has been accessed.

Owner-authorized changes:

1. Intermediate/full launch is elastic K=1..8. At each launch, scan H20 devices and lock every qualifying card up to 8, with at least 1. Eligibility is: device name `NVIDIA H20`, free memory at least 21 GiB, and instantaneous GPU utilization at most 10%. Existing compute PID count and used memory are recorded per GPU but do not independently disqualify it. K<8 is not failure or BLOCKED; it is recorded in startup and scaling evidence.
2. The aggregate P2 matched-control engineering gate is 0.90 instead of 0.98. The threshold is read from the sealed run spec by producer and validator; it is not hard-coded. Per-coordinate budgets, exact position-stratum matching, boundary-complexity tolerance, seeds, control count, typed NA propagation, and mask operator rules do not change.

Unchanged: pilot-only access; target; component; blocks; geometry; source score and signed-positive order; quadrature and tolerances; P1; D1; D2; schedule and sample identities; outcome-blind escalation; cost-as-reporting-only policy; terminal states; scientific-claim boundary. S1 remains an engineering feasibility stage and must not issue `SUPPORTED` or `NOT_SUPPORTED`. Only new intermediate indices 6–23 may be computed next.
