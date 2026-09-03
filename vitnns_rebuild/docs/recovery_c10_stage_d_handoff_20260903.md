# Recovery C10 to future Stage D handoff — 2026-09-03

## Terminal C10 state

- Build complete: `true`; integrity valid: `true`.
- Acceptance: `BLOCKED` by `HOLM_BLOCKED_UNDERSPECIFIED`.
- Primary decision: `NO_GO_NOVELTY`.
- Components: validity `PASS_VALIDITY`; effect `PASS_EFFECT`; clean joint
  spatial `PASS_JOINT_SPATIAL`; translation and center-crop/resize
  `FAIL_STABILITY`; Gaussian `PASS_STABILITY`; overall spatial
  `FAIL_SPATIAL`; cost `PASS_COST`.
- `PASS_EFFECT` is a fidelity component result. It does not override either
  stability hard-gate failure or make C10 an overall PASS/GO.
- Bootstrap ran with 10,000 paired image-cluster draws on holdout `n=15`.
  Holm did not run; no raw or adjusted p-values were generated.

The formal root is `evidence/stage_c/c10_recovery_v2_3cf2945/formal`. Its
report SHA-256 is
`373cdb6761ab0efe5e78bf18ff9257c90cf2c7d36bfed68595cb9cf35c65adff`,
canonical report digest is
`efc4e880793166fdf2e7869396cc0d0df9699e27c53071aa83c83ddce472e849`,
and ordered whole-root inventory digest is
`b63a32ab2110081aae0a0b033d6a79f6798f611b20f3e6b0a3e769b5ff9b5627`.
The source chain is canonical salvage C8 -> recovery C9 -> this C10.

## Permitted future Stage D scientific scope

The existing Stage D scientific scope is internally consistent and remains
unchanged: a prospective fidelity-only replication/extension on a separate
500-image cohort, primary `(0,9)`, NNS versus matched patch baseline, endpoint
`fine_4x4_cell_delta_fid`, and `run_stability=false`. Translation,
center-crop/resize, and Gaussian are not Stage D endpoints or decision gates.
No new Stage D spatial/stability claim may be inferred.

C10 translation and center-crop/resize failures are known limitations and must
remain explicit in Stage D Study Design, Results, and Limitations. C10 `n=15`
and Stage D `n=500` are separate cohorts and must not be pooled into `n=515`
or one statistical conclusion. With no secondary inference, future Stage D
may use `holm_status=NOT_APPLICABLE_NO_SECONDARY_INFERENCE`; that neither
repairs nor cancels C10 `HOLM_BLOCKED_UNDERSPECIFIED`.

## Binding gap and authorization boundary

The current Stage D machine-readable files still bind the historical C10 root,
historical C10 producer, and missing historical `2e718...` model. Before any
reserve-image access, a versioned and outcome-blind renewed D0-D6 binding must
replace those identities with recovery-v2 model `489bfa...` / tensor digest
`150234d8...` and the new salvage C8 -> C9 -> C10 chain. Only after that chain
passes may D7 assess and issue a separate execution authorization.

This handoff is not a Stage D execution or analysis authorization. Stage D was
not started in this task, and no Stage D reserve image was accessed, statted,
hashed, decoded, preprocessed, or used for inference.
