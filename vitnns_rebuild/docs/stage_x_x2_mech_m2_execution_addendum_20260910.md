# X2-MECH M2 execution addendum

Date: 2026-09-10

Authorization: `X2_MECH_M2_CPU_ONLY_MULTISCALE_MECHANISM_ANALYSIS_ONLY`

## Scope

M2 is a CPU-only post-hoc exploratory analysis over sealed M1-R1 maps, masks,
M1 4x4/16x16 class-logit ROAD fidelity rows, and sealed X2 pixel-stability
rows. It does not change M1, X2, Stage D, Stage E, Stage F/F3C, or any archive
ref. It does not authorize F3-Final, Stage G, X3, X4, or X5.

The maximum allowed claim remains:
`post-hoc exploratory evidence of a granularity-fidelity-equivariance trade-off`.
No M2 output may issue a spatial-stability PASS, NNS-v2 superiority or
noninferiority to Grad-CAM, broad benchmark GO, localization claim,
cross-model/cross-dataset generalization claim, confirmatory p-value, or causal
conclusion.

## Producer seal

The producer freezes `config/stage_x_x2_mech_m2.json`,
`config/stage_x_x2_mech_m2.schema.json`,
`config/stage_x_x2_mech_m2_artifact.schema.json`,
`src/vitnns/stage_x_x2_mech_m2.py`, this addendum, and
`tests/test_stage_x_x2_mech_m2.py`. Before the producer ref is created, it may
read only source code, JSON control metadata, and CSV header lines. It must not
read M1 map array values, M1 fidelity rows beyond headers, or X2 per-route
stability rows.

Producer completion writes only `source_binding.json`,
`pre_execution_seal.json`, and `completion/x2_mech_m2_producer.json`. No M2
empirical table is created during producer sealing.

## Empirical gate

After the producer archive ref, M2 first revalidates M1-R1 from source files:
199 route terminals, 9,552 maps, 24 masks, 796 fidelity summaries, 71,640 ROAD
rows, manifest/inventory/resource ledger, completion/ref, denominator, and the
absence of pending files or existing M2 terminal output. Any drift blocks M2.

## CPU-only execution

M2 must run with `CUDA_VISIBLE_DEVICES=""`. It does not import timm, load a
model, load a checkpoint, read JPEG bytes, call forward/backward, download
files, or produce display PNG/JPEG. CPU NumPy/SciPy analysis is allowed. CPU
Torch/torchvision may be used only to reproduce rotate/crop transport geometry
with CUDA invisible, and that import is recorded in `access_audit.json`.

## Branch rule

M2 may choose only the four M1 to M2 handoff branches:

- `BRANCH_A_AGGREGATION_SENSITIVITY`
- `BRANCH_B_GRANULARITY_FIDELITY_EQUIVARIANCE_TRADEOFF`
- `BRANCH_C_COARSE_INSTABILITY_PERSISTS`
- `MIXED_OR_INCONCLUSIVE`

Because the frozen protocol contains no numeric thresholds that uniquely select
A, B, or C, the required terminal branch is `MIXED_OR_INCONCLUSIVE`, with
observed directions reported separately.
