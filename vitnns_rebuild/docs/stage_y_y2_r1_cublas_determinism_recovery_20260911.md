# Stage Y2-R1 cuBLAS determinism environment recovery

## Authorization

This document authorizes only
`STAGE_Y2_R1_CUBLAS_DETERMINISM_ENVIRONMENT_RECOVERY_ONLY`.  It is a
prospective infrastructure retry of the frozen Y2 clean cohort.  It does not
authorize Y3, bootstrap, Holm correction, or a claim decision.

## Immutable predecessor

The original Y2 root, report, and blocked archive ref remain immutable at
commit `1a621724b32b208036fb5f2a3993a48697e933c8`.  Its terminal remains
`BLOCKED_INFRASTRUCTURE`, with 100 blocked unit terminals, zero valid units,
outcome access recorded, reconciliation-phase failure, and no next
authorization.  Y2-R1 writes only to its distinct formal and report roots.

## Root cause and sole repair

Every original unit failed before target selection or quadrature with the same
deterministic cuBLAS error.  The launcher had not set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch in child processes.
The sole scientific-execution repair is to inject that exact value into every
probe and worker subprocess before process creation and to fail closed inside
the worker before Torch import, JPEG decode, or model computation if it is
missing or different.  Deterministic algorithms remain enabled with
`warn_only=False`; cuDNN benchmarking and both TF32 flags remain disabled.

## Frozen scientific identity

Y2-R1 reuses byte-identical original Y2 cohort, selection proof, audit subset,
perturbation subset, and exclusion union artifacts.  It recomputes the cohort
from the official release tar and metadata during preflight.  The clean-only
condition, predicted-class raw-logit target, model checkpoint, blocks, heads,
cuts, quadrature schedule, convergence thresholds, deletion operator,
ablation contract, retention contract, and deterministic modulo sharding rule
do not change.  Device availability may change worker count but cannot change
cohort order.

## Recovery preflight

Formal workers may start only after a separate preflight verifies the old root
and archive binding, all 100 homogeneous failure terminals, absence of old
scientific arrays and targets, raw-source cohort/subset identity, pre-import
environment injection, a deterministic float32/float64 CUDA matrix multiply
plus backward exact-repeat probe, cross-device digest equality, focused tests,
and the false-pass mutation audit.  The implementation binding is frozen
before any formal retry unit starts.

## Retry and resources

Every new unit terminal binds its original unit and attempt digests, original
failure code, retry index 1, and the fixed retry reason.  Original and recovery
GPU, CPU, and retained-storage usage are added before enforcing the original
Y2 aggregate caps of 64 H20-hours, 218 CPU-hours, 6 GiB retained storage, and
21 GiB peak reserved memory per process.  Only the recovery wall clock resets.

## Terminal rule

PASS requires 100 new terminals, at least 90 VALID clean units, reconciliation
and aggregate budget PASS, source-recomputing validation, false-pass audit, and
Git/LFS publication.  Its sole next authorization is
`STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY`.  Any other terminal is
BLOCKED with `next_authorization=NONE` and requires another fresh root.
