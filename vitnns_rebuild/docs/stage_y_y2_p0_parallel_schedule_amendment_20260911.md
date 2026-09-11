# Stage Y2-P0 prospective parallel-schedule amendment

Token: `STAGE_Y2_P0_PARALLEL_SCHEDULE_AMENDMENT_AND_IMPLEMENTATION_FREEZE_ONLY`

This amendment is prospective and occurs before any formal Y2 cohort registry,
selected JPEG decode, checkpoint/model load, CUDA/GPU access, scientific array,
unit terminal, or outcome access. It changes only execution scheduling, worker
isolation, and critical-path wall-time budgeting. Y0 claims, metrics, statistical
families, model/checkpoint/preprocessing, cohort selection and exclusion, target,
blocks, heads, cuts, attribution identity, quadrature, tolerances, failure
semantics, aggregate resource accounting, and the immutable Y1/Y1-R1 record do
not change.

Y2 assigns selection order `i` to `i mod 8`. Shards 00-03 contain 13 clean
images and shards 04-07 contain 12. Y3 is frozen but not launched: its canonical
160-unit order is `(perturbation_subset_order, transform_registry_order)`, and
unit index `i` goes to `i mod 8`, exactly 20 units per shard. Outcome-, runtime-,
class-, convergence-, transform-difficulty-, or attribution-based rebalancing is
forbidden.

Exactly eight homogeneous NVIDIA H20 devices are required. Each process binds a
unique physical GPU with one visible device, uses `cuda:0` internally, records
physical index, `CUDA_VISIBLE_DEVICES`, UUID, PCI bus ID, name, hostname, PID,
and environment digest, and writes only its fixed shard root. A missing eighth
device yields `BLOCKED_RESOURCE_SCHEDULE`; there is no automatic fallback. The
cohort seal is completed before any worker launch and is thereafter read-only.
Each unit transitions atomically from PENDING to one typed terminal while every
attempt remains append-only.

CPU-only reconciliation requires shard roster 00-07, exact 0-99 coverage, the
modulo mapping, one accepted terminal per unit, all retries, common checkpoint,
environment and numeric contracts, complete summed resource telemetry, preserved
failures/NA, at least 90 VALID units, and clean-only execution. It performs no
bootstrap, Holm correction, or claim decision. Y3 remains forbidden until Y2 is
terminal PASS.

This original P0 schedule was superseded prospectively, before formal cohort
access, by `stage_y_y2_p0_r1_elastic_idle_h20_schedule_20260911.md`. Its fixed
eight-device requirement is historical and must not be used to launch Y2.

The source-recomputed caps are frozen in
`config/stage_y/y2_parallel_schedule.json`. Aggregate GPU-hours remain summed
across devices. The only authorization after Y2-P0 PASS is
`STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY`; this amendment does not itself
authorize or start a worker.
