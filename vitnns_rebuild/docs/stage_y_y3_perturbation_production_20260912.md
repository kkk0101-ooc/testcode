# Stage Y3 perturbation mechanism production

Status: prospective and frozen before any Y3 JPEG decode, model load, CUDA
initialization, or outcome access.

Y3 binds the immutable 20-image `perturbation_20` registry in its exact
`subset_rank` order and reuses, without rerunning, the clean arrays that were
validated by Y2-R2-R2.  It computes exactly the eight non-clean conditions in
the frozen transform registry, for 160 new empirical units.  The target for
every condition is the producing clean terminal's predicted-class raw-logit
index. Prediction changes are retained and never change that target.

## Prospective provenance correction

The owner instruction described the clean-source split as 18 Y2-R1 VALID plus
two Y2-R2-R1 computed sources (selection orders 77 and 84). Direct validation
of the immutable producing terminals shows a single descriptive error in that
sentence: selection order 93 is also a Y2-R2-R1 computed source. Its Y2-R1
terminal is `BLOCKED_BUDGET` with no arrays, and its Y2-R2-R1 terminal is
`VALID` with the complete 262-array roster. Therefore the source-derived split
is 17 Y2-R1 VALID plus three Y2-R2-R1 computed sources (77, 84, and 93).

This correction does not change the frozen 20 images, their order, target,
condition roster, scientific contract, evidence bytes, or any historical ref.
It is recorded before Y3 outcome access and is purely a provenance correction.

The first Y3 producer commit `d5e96c0e0c55e685d38f182e63f3227dd8c1cbb9`
was archived before outcome access. Subsequent pre-execution false-pass review
found that its array validator admitted a matching float64 record, its aggregate
stage budget was not reserved across workers, and its final Y2 publication
recheck treated authorized untracked Y3 files as historical dirt. That producer
remains immutable. Y3-R1 prospectively supersedes it with those implementation-
only repairs; scientific scope and all frozen identities remain unchanged.

## Execution boundary

- Formal root: `evidence/stage_y/y3_perturbation_formal_20260910/formal/`.
- Report root: `reports/stage_y_y3_20260912/`.
- `unit_index = subset_rank * 8 + nonclean_condition_index`.
- Elastic startup locks 1--8 idle H20 devices in ascending physical-index
  order. A unit belongs to `unit_index mod locked_worker_count`; no hot-add,
  migration, dynamic balancing, replacement, or retry-until-pass is allowed.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set by the launcher and checked before
  Torch import by each worker. The startup probe executes float32 and float64
  cuBLAS matrix multiplication, backward, and exact repeatability.
- Y3 caps are 102.4 H20-GPU-hours, 349 CPU-hours, 103 wall-hours, 21 GiB peak
  reserved GPU memory per process, 9 GiB retained storage, and zero downloads.
- Clean inputs are source-validated but never executed as Y3 units.
- Every VALID non-clean unit retains the Y2-compatible 262-array complete-cut
  roster, valid mask, patch inverse operator W, predictions, transform
  provenance, all quadrature attempts, interventions, and resource telemetry.
  The one audit-overlap image additionally retains signed/positive/negative
  pixel maps and all-block raw-attention controls. Raw attention is descriptive
  only.
- The independent reconciliation recomputes source identities, arrays, closure,
  target fixedness, masks, inverse operators, seeds, shard ownership, resources,
  and the closed inventories from raw files.

Y3 PASS requires all 160 immutable terminals, at least 18 images with a VALID
clean reference and eight VALID non-clean units, no blocking provenance/access/
infrastructure/schedule/budget state, and successful false-pass and publication
validation. PASS authorizes only
`STAGE_Y4_CPU_ONLY_MECHANISM_ANALYSIS_AND_DECISION`. Y3 performs no bootstrap,
Holm correction, cohort inference, claim decision, scientific interpretation,
Y4, Y5, F3-Final, or paper writing.
