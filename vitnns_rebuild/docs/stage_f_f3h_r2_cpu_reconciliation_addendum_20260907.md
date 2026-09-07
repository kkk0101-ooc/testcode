# Stage F F3H-R2 CPU reconciliation recovery（2026-09-07）

## 1. Trigger and immutable source

F3H-R1 completed the authorized GPU workload: 280/280 newly computed Value-path terminals were published, the eight inherited terminals remained sealed upstream, and the one frozen Grad-CAM terminal was VALID. The first CPU reconciliation then failed before writing `terminal_union.json` because the adapter assumed every inherited result stored a top-level `target_identity`. D8 uses that field, while the three E3 coordinates store the same bound identity through `target_response.target_identity`, `source_hashes.target_semantic_digest`, and `pair_binding.target_semantic_digest`.

R1 remains an immutable `CPU_RECONCILIATION_BLOCKED_AFTER_COMPLETE_GPU_COMPUTE` source. R2 must not load a model, import or use Torch/timm, access CUDA/GPU, or execute attribution. Before exact terminal reconciliation it may only validate and consume R1 plus the eight previously sealed upstream maps. After reconciliation, it may decode only the already frozen order-44 JPEG for display-only original/overlay panels; this access cannot alter sample or head selection.

## 2. Source-adapter rule

For inherited D8 results, R2 computes `target_semantic_digest` from the existing top-level `target_identity`. For inherited E3 results, R2 requires all three of the following to exist and agree exactly:

1. `source_hashes.target_semantic_digest`;
2. `pair_binding.target_semantic_digest`;
3. the recomputed digest of `target_response.target_identity`.

R2 additionally verifies sample ID, selection order, block/head, method, target, image hash, clean-input hash, model hash, signed-map digest, source result SHA-256, and source map SHA-256. Missing or conflicting values block the stage. No target identity may be guessed or reconstructed from visual appearance.

## 3. Recovery roots and no-recompute boundary

R2 outputs are written only to:

- `evidence/stage_f/f3h_full_head_r2_cpu_reconcile_20260907/formal/`;
- `reports/stage_f_f3h_full_head_r2_cpu_reconcile_20260907/`.

R2 binds every R1 compute-source file through an exact inventory. It references R1 maps in place; it does not copy, alter, replace, or recompute them. The new producer ref is `refs/archive/stage-f-v1-f3h-r2-producer`. The final F3H completion ref remains `refs/archive/stage-f-v1-f3h-completion`.

## 4. Permitted outputs

After exact 288-terminal reconciliation, R2 may generate only the already frozen F3H CPU products: 576 signed components, 144 pair-shared log scale records, complete 12×12 atlases, 12 block pages, Grad-CAM reference, equal-head spatial-support consensus, descriptive diversity diagnostics, class-conditioned-basis assessment, review copies, audits, validation, and terminal completion.

The R1 telemetry is counted exactly once. The failed CLI parse attempt performed no computation and is recorded as an operational deviation; the successful Grad-CAM Python API call used the same sealed implementation function and frozen arguments.

## 5. Authorization boundary

R2 terminal PASS can authorize only `F3C0_PROTOCOL_FREEZE_ONLY`. The full atlas may be used only with the predeclared sufficiency rule; visual attractiveness, brightness, Grad-CAM overlap, entropy, or a selected head cannot define numerical class-conditioned weights. F3C empirical computation and F4 remain prohibited.
