# Stage Y Y1-R1 resource telemetry waiver and budget rebind

Date: 2026-09-11

Token: `STAGE_Y1_R1_RESOURCE_TELEMETRY_WAIVER_AND_BUDGET_REBIND_ONLY`

Status: prospective recovery amendment authorized by the owner.

Y1-R1 is CPU-only, existing-evidence-only, and resource-governance-only. It
does not rewrite the original Y1 `BLOCKED_BUDGET` completion, failure, budget,
inventory, report, or blocked archive ref. It does not decode an image, load a
checkpoint or model, access CUDA, rerun a pilot, create a scientific array,
materialize the Stage Y cohort, or make a scientific claim decision.

The one-time waiver is limited to historical phase-level telemetry that cannot
be recovered honestly from two retained prepublication workloads. The first
workload has no per-phase breakdown. The second has phase entries, but its
`matrix_cls_extraction` timing boundary was placed after extraction and is not
valid. Every unavailable value remains
`NOT_RECOVERABLE_HISTORICAL_PHASE_TELEMETRY`; zero, imputation, estimation, and
back-calculation from a later run are forbidden. Overall calls, attempts, GPU,
CPU, wall, memory, output bytes, retained bytes, and downloads are not waived.

The three retained full workloads are accumulated as a measured lower bound.
Future-cost projection uses, independently for each resource metric and
condition, the maximum observed value across all three workloads, followed by
the frozen Y0 formula and 1.5 safety multiplier. This waiver is specific to
Y1-R1 and does not waive future Y2/Y3 telemetry failures.

If validation passes, Y1-R1 may issue only
`STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY`. Y2 is not started in this
task. F3-Final remains `F3_FINAL_PAUSED_PENDING_STAGE_Y`.
