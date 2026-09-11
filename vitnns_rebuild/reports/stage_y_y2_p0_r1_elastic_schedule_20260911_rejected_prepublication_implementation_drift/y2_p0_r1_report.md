# Stage Y2-P0-R1 terminal report

Status: PASS. Completion basis: PROSPECTIVE_PRE_ACCESS_ELASTIC_IDLE_H20_SCHEDULE_REPAIR.

The repair was completed before formal cohort sealing, selected-JPEG decode,
model/checkpoint load, CUDA/GPU access, scientific-result creation,
unit-terminal creation, or formal worker launch. Historical Y2-P0 evidence
remains immutable and digest-bound.

The startup gate locks every eligible idle NVIDIA H20 in ascending physical
index order, with K in 1..8. Four idle cards produce four deterministic
25-unit Y2 shards. Zero eligible cards fails closed as
BLOCKED_RESOURCE_SCHEDULE. Once locked, K and unit assignments cannot change.

Y2 aggregate caps remain 64.0 H20-GPU-hours, 218 CPU-hours, and 6 GiB. Y3
aggregate caps remain 102.4 H20-GPU-hours, 349 CPU-hours, and 9 GiB.
Critical-path and stage-wall caps are source-recomputed for every K; GPU-hours
are summed across workers and never divided by K. Concurrent reserved memory
is 21 GiB * K.

Focused synthetic tests: 26 passed. False-pass audit:
17/17 rejected. The only next authorization is STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY.
