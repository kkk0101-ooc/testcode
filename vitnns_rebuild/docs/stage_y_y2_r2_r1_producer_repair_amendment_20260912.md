# Stage Y2-R2-R1 prospective producer repair

## Authorization and immutable history

The follow-up authorization to fix the producer and finish the remaining units authorizes this prospective revision of `STAGE_Y2_R2_RESOURCE_REPORTING_RECOVERY_AND_MISSING_UNITS_ONLY`. It is not outcome-blind. Stage Y2-R2 remains `BLOCKED_STARTUP_STATE` at `1fd71d1b3e4d4c10fd3c65374169c734b64b8ff3`; its producer remains frozen at `d0023fa67d494e73422fd0e7df767f768a3c942a`. Neither historical root, report, receipt, configuration, commit, nor archive ref is rewritten. Historical source identities refer to their Git snapshots, not this revision of the shared entry points.

The new stage is `Y2-R2-R1`, with root `evidence/stage_y/y2_r2_r1_clean_formal_20260912/formal/`, reports `reports/stage_y_y2_r2_r1_20260912/`, producer ref `refs/archive/stage-y-y2-r2-r1-resource-reporting-recovery-producer`, and completion ref `refs/archive/stage-y-y2-r2-r1-clean-cohort-evidence-completion`. Startup binds branch `stage-d-d3-pass` at the published blocked commit above. Original Y2 and Y2-R1 remain immutable at `1a621724b32b208036fb5f2a3993a48697e933c8` and `7cf9b1f9a25695821b19e98cff4e2edbc369891a`.

## Producer repair boundary

This revision repairs the attempt identifier producer/consumer mismatch, distinguishes 40-character Git commits from 64-character SHA-256 digests, closes nested artifact schemas, validates every emitted artifact before publication, and tests the worker-output-to-completion path with synthetic arrays before freeze. It also enforces implementation seals, exact shard ownership, source-file inventories, typed failures, and complete resource accounting. These are provenance/reporting repairs, not scientific changes. After publication of the new producer, source, schemas, contracts, and validator are immutable for this attempt.

The numerical engine, model loader, checkpoint, preprocessing, targets, cohort and subsets, quadrature ladder, cut/head/block definitions, grouped JVP/VJP implementation, closure thresholds, identity tolerance, float32 array layout, array digest formula, and ablation/deletion contracts remain those of the frozen Y2-R1 implementation. No empirical result is used to choose a method, threshold, sample, device, or retry.

## Exact recovery partition

The denominator remains 100 with selection orders 0 through 99. Exactly 84 original VALID units are independently source-recomputed as `Y2_R1_VALID_REBIND`. Orders 73, 80, and 86 are independently source-recomputed as `Y2_R1_COMPLETED_BUDGET_REBIND`; the original budget terminals remain unchanged. Rebinding never copies arrays or executes the model. Original attempt journals, raw arrays, closure, input identity, source/checkpoint/environment bindings, and resource records are verified. For the three budget candidates, completed intervention execution is proven by frozen control flow and exact call accounting; discarded ablation/deletion scalar payloads are not claimed to have been recovered or recomputed.

Only orders 77, 81, 84, 85, 88, 89, 90, 92, 93, 94, 96, 97, and 98 may be computed, each with one new scientific attempt. Their source type stays `Y2_R2_COMPUTED`, with producing stage `Y2-R2-R1`. The original cap-exhausted terminals must show no arrays, no scientific result, and zero model calls. Failed rebind candidates may not enter the missing roster. No retry-until-pass, filling, replacement, or early stop at 90 is authorized.

## Environment, execution, and failure semantics

Before any worker Torch import, `CUBLAS_WORKSPACE_CONFIG=:4096:8` is required. Startup probes use the exact designated interpreter, deterministic float32/float64 CUDA matmul and backward, exact repeat equality, and cross-device digest equality. Probe and worker compare the same complete environment identity. Eligible idle H20 devices, at most eight, are locked in ascending physical index once. Sorted missing-roster index modulo the locked count assigns one process per physical GPU; each sees only `cuda:0`. No hot-add or dynamic migration is allowed.

CPU/GPU/wall/storage/download and peak-memory measurements remain reporting-only. Original Y2, Y2-R1, blocked Y2-R2, and this revision are reported separately and cumulatively, including initialization and CPU verification work. Resource excess is `RESOURCE_OVERRUN_REPORTED`, never a reason to interrupt healthy computation or relabel complete science. Downloads are not authorized; only already-bound local assets may be read. Actual OOM, device, process, CUDA, filesystem, or disk failures remain `BLOCKED_INFRASTRUCTURE`; access and provenance drift remain typed blocked failures; numerical non-finiteness or frozen validity failure remains `FAILED_NUMERICAL`. A typed failure stops new work globally and preserves all completed evidence.

## Freeze and acceptance

Before old scientific arrays are opened or empirical workers begin, metadata-only predecessor validation, raw-byte cohort/subset recomputation, prior-stage validators, compilation, closed-schema checks, focused tests, and executable false-pass mutations must pass. Commit and push the producer and its new immutable ref; verify HEAD, tracking, remote branch, and producer ref coincide, then write the implementation seal. Every subsequent command and worker checks that seal.

Each terminal uses fsynced PENDING, fsynced temporary terminal, atomic `os.replace`, and directory fsync. The independent reconciler must recompute source records and scientific arrays for all 100 units and reject any missing, duplicate, mutated, or unvalidated source. PASS requires 84+3+13 VALID, complete inventories and audits, unchanged predecessor roots/refs, and Git/LFS publication. Completion has `formal_outcome_access=false`, meaning no downstream formal statistical or claim decision access; authorized clean-evidence array validation is recorded separately. Only `STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY` is authorized on PASS. No Y3/Y4, bootstrap, Holm, claim decision, scientific interpretation, or paper writing is executed here; otherwise next authorization is `NONE`.
