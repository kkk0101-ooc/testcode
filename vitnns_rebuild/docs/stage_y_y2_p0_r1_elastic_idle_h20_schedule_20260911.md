# Stage Y2-P0-R1 elastic idle-H20 schedule amendment

Token: `STAGE_Y2_P0_R1_ELASTIC_IDLE_H20_SCHEDULE_AND_IMPLEMENTATION_FREEZE_ONLY`

This prospective amendment supersedes only the fixed-eight-device scheduling
part of Y2-P0 commit `66a0dfb889fdb0d45d9e566ed422ba9a40bd30a1`. It was
issued before formal Y2 cohort access, JPEG decode, model/checkpoint load, CUDA
access, scientific arrays, unit terminals, or worker launch. The original P0
evidence remains immutable audit history.

At startup, the gate selects every eligible idle NVIDIA H20 in ascending
physical-index order, capped at eight. Eligibility requires at least 21 GiB free
memory and no active compute PID. At least one eligible card is required; zero
cards yields `BLOCKED_RESOURCE_SCHEDULE`. The selected count `K` and physical
identities are sealed once before shard manifests are built. A card becoming
busy later cannot change `K`, mappings, or accepted outputs.

Y2 uses `shard_id = selection_order_zero_based mod K`. Thus four idle cards
produce `[25,25,25,25]`; all rosters for `K=1..8` are frozen in the schedule.
Y3 uses the same locked-count rule over its canonical 160-unit order, but Y3 is
not authorized here. Rebalancing by runtime, class, prediction, convergence,
attribution, transform difficulty, failure, or outcome remains forbidden.

Aggregate caps remain 64.0 H20-GPU-hours, 218 CPU-hours, and 6 GiB for Y2 and
102.4 H20-GPU-hours, 349 CPU-hours, and 9 GiB for Y3. Per-shard and critical-path
caps are recomputed from the immutable Y1-R1 ledger after `K` is locked.
Aggregate GPU-hours remain summed, never divided by `K`. Fleet concurrent
reserved memory is `21 GiB * K`.

No model, cohort, target, preprocessing, attribution, quadrature, tolerance,
failure, metric, statistic, claim, or historical Y0/Y1/Y1-R1 contract changes.
The only authorization after this implementation freeze passes is
`STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY`.
