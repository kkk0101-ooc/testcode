# Stage Y2-R2 resource-reporting recovery amendment

## Authorization and temporal status

This prospective amendment authorizes only
`STAGE_Y2_R2_RESOURCE_REPORTING_RECOVERY_AND_MISSING_UNITS_ONLY`. It is
published after the typed `BLOCKED_BUDGET` terminal of Y2-R1 and therefore is
not outcome-blind. It does not authorize Y3, Y4, bootstrap, Holm correction,
claim decisions, scientific interpretation, or paper writing.

## Immutable predecessors

The original Y2 root/ref remains immutable at
`1a621724b32b208036fb5f2a3993a48697e933c8`. The Y2-R1 root/ref remains
immutable at `7cf9b1f9a25695821b19e98cff4e2edbc369891a`, permanently recording
`BLOCKED_BUDGET`, 84 VALID units, 16 BLOCKED_BUDGET units, and no next
authorization. No predecessor file, receipt, inventory, commit, or ref may be
modified, moved, deleted, or relabelled.

## Scope of the recovery

Y2-R2 changes only resource-failure semantics and evidence rebinding. All
scientific contracts remain byte-for-byte or semantically identical to the
frozen Y0/Y1/Y2/Y2-R1 contracts: cohort and order, source objects and JPEG
hashes, checkpoint and model, preprocessing, clean predicted-class raw-logit
target, native graph, component cuts, grouped JVP/VJP implementation,
quadrature ladder, numerical thresholds, block/head/cell/patch definitions,
array retention, dtype/shape/axes and canonical digest rules, and the ablation
and deletion endpoints.

The Y2-R2 unified view has exactly one source for each selection order:

- 84 `Y2_R1_VALID_REBIND` records, validated from original terminals,
  scientific results, attempts, arrays, source JPEGs, and hashes without GPU
  replay or array copying.
- Three `Y2_R1_COMPLETED_BUDGET_REBIND` records for orders 73, 80, and 86,
  accepted only after independent source and array recomputation proves that
  their completed scientific attempt satisfies the frozen contracts. They are
  not rerun and their original terminals remain `BLOCKED_BUDGET`.
- Thirteen `Y2_R2_COMPUTED` records for orders 77, 81, 84, 85, 88, 89, 90,
  92, 93, 94, 96, 97, and 98. No other unit may enter a GPU shard.

## Resource semantics

CPU-hours, H20-GPU-hours, per-worker wall time, critical-path wall time,
overall wall time, retained bytes/storage, peak memory, and download bytes are
measured and reported separately for Y2, Y2-R1, and Y2-R2 and cumulatively.
Exceeding an estimate or former ceiling is `RESOURCE_OVERRUN_REPORTED`; it
does not stop a healthy unit, relabel a complete unit, or prevent PASS when all
100 units are scientifically valid. Download telemetry is reporting-only and
does not authorize downloads, new data, replacement JPEGs, or checkpoints.
Peak memory is reporting-only; an actual OOM remains
`BLOCKED_INFRASTRUCTURE`.

There are no CPU/GPU/wall/storage/download numerical stop gates in the R2
worker. Unauthorized access is `BLOCKED_ACCESS`; source, hash, schema,
checkpoint, target, cohort, or implementation drift is `BLOCKED_PROVENANCE`;
actual device/process/filesystem/CUDA failure is `BLOCKED_INFRASTRUCTURE`; and
non-finite arrays or frozen closure/identity failure is `FAILED_NUMERICAL` or
`BLOCKED_SCIENTIFIC_VALIDITY`.

## Scheduling and atomicity

At startup, all eligible idle NVIDIA H20 devices up to eight are locked in
ascending physical-index order. Each worker sees only `cuda:0`. The sorted
13-unit missing roster is assigned by its zero-based roster index modulo the
locked worker count. There is no hot-add, migration, or dynamic rebalancing.
Each missing unit has exactly one new scientific attempt; adaptive quadrature
within that attempt is not a retry.

All unit and stage terminals use fsynced PENDING, a fsynced temporary terminal,
`os.replace`, and completion-directory fsync. A PASS requires a source-
recomputed unified 0..99 roster with all 100 units VALID, complete audits and
inventories, and Git/LFS publication. The sole next authorization on PASS is
`STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY`; Y3 is not executed here.
