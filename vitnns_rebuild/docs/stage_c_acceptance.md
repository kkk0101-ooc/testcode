# Stage C acceptance contract

This is the fail-closed acceptance contract frozen by C0. The scientific definitions and rationale are in `docs/阶段C_protocol.md`; exact machine values are in `config/stage_c.json`. A later task passes only when every gate assigned to that task passes. A structured failure is evidence; an absent row, silent exclusion or changed denominator is not.

## C0: protocol and sample freeze

1. The starting HEAD, dirty files and ownership are recorded. The authorized untracked Stage C step-by-step document is preserved and committed; unrelated user changes are neither overwritten nor staged.
2. The Stage B formal input has `status=complete`, `complete=true`, a status-bound manifest hash, matching artifact hashes/bytes, provenance commit `30732c9f1413c58a36dd6a8d85ba8ea5788e0c8a`, `dirty=false` and `formal=true`.
3. The official devkit actual SHA-256 equals `b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953`. The label path validates 1,000 unique leaf WNIDs, class IDs 1..1000, 50,000 validation entries and 50 entries per class before producing labels.
4. Existing Stage A manifest code rebuilds exactly 500 labeled rows from the flat validation directory. The tracked manifest hash is `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`; every row has a unique ID/path, content hash, byte size, decoded dimensions, official WNID, canonical index and `official_devkit_val_gt`.
5. The Stage C registry has the exact 20 frozen IDs in the exact historic order. It has 5 calibration, 15 holdout and 2 audit rows; audit IDs are exactly `00000079` and `00049970` and are calibration rows.
6. Every registry row is byte-for-byte field-equivalent to its source manifest row for path/hash/bytes/dimensions/WNID/index/source; every live JPEG independently matches these facts. Model predictions are absent from ground-truth fields.
7. The historic directory `49_vit ` remains read-only. Its old numeric results and display files are not copied, parsed as metrics or used for selection.
8. `docs/阶段C_protocol.md` has no unresolved scientific choice that can be decided after C8/C9. `config/stage_c.json` serializes all required numerical choices; the config, registry and selection schema JSON files parse strictly.
9. Draft 2020-12 plus semantic static validation rejects nested unknown/missing fields, duplicate JSON keys, non-finite JSON numbers, bad IDs/counts/roles/hashes/labels, invalid method/target relations, invalid projections, metrics, fidelity, perturbations, retention, budgets, selection or decision gates. The C8 selection contract rejects incomplete hash bindings, untyped/unknown candidate fields, duplicate or incomplete 12×12 candidates, inconsistent band winners and a primary outside the four winners. Optimized Python (`-O`) is rejected rather than silently disabling asset assertions.
10. C0 performs no Stage C method run or GPU attribution and adds no comparison-method implementation, production Stage C parser or selection. The C0 validator itself imports no model and performs no forward; the repository-wide non-GPU regression suite may exercise existing Stage A/B CPU forwards. All C0 tracked files are reviewed and committed; post-commit static validation passes and the worktree is clean.

Any failed item makes C0 `BLOCKED`; it may not hand an unfrozen key decision to C1.

## C1: strict config, map and metric contracts

1. A production loader rejects duplicate keys, NaN/Infinity, unknown/missing keys, wrong types (including bool-as-int), bad enums/ranges/order/uniqueness, unresolved references and every invalid cross-field combination stated in the protocol. It round-trips the frozen config without adding defaults.
2. A `MethodResult` or equivalent records exact method ID/version, comparison line, explanation/fidelity target, native tensor, metric-ready signed/nonnegative 2D map, sign semantics, projection and provenance fragment. Class-agnostic maps cannot claim a class explanation.
3. Pixel→patch sums and patch→pixel division preserve signed mass within `atol=rtol=1e-6`. 4×4 cell operators obey the same contract. Shape, dtype and finite checks fail closed.
4. Pearson/Spearman, JS, bidirectional KL, top-10/20 Jaccard, normalized entropy and structured NA exactly match protocol formulas, including average ranks, base 2, epsilon, ceil counts and deterministic row-major ties.
5. Metric APIs accept only metric-ready numeric arrays plus explicit masks/sign views; image/PNG/display inputs are impossible or rejected. Zero, constant, sparse, signed, unsupported-sign, shape-mismatch, empty-mask and non-finite tests pass.
6. Full non-GPU tests pass. No C2+ method is implemented or run.

## C2: direct raw CLS attention and rollout

1. Direct attention comes from the real extractor's `attention_used`; in eval mode `attention` and `attention_used` semantics are verified and direct values match exactly.
2. Per-head Value-line direct maps use the matched block/head. Class-line direct uses block 11 arithmetic head mean. Both remove the CLS key and renormalize 196 patch entries; invalid mass fails.
3. Rollout uses `B_l=row_norm(I+mean_h A_lh)` and `B_11...B_0`; rows sum to one and values are finite. Identity, fusion, normalization and multiplication order have hand-computable toy tests and a single-layer degeneration test.
4. Hooks/model mode are restored after success or exception. No raw attention Cartesian product is retained.

## C3: class-logit Grad-CAM and full-model IG

1. Both use the same clean predicted-class raw logit, frozen for every intervention, and record official truth/correctness without exclusion.
2. Grad-CAM targets `blocks[11].norm1`, removes CLS, averages gradients over 196 patch tokens, applies ReLU and no min-max scaling. Shape, finiteness, non-empty output and target sensitivity pass.
3. Full-model class-logit IG uses normalized-zero baseline and the frozen Gauss-Legendre schedule. It is always named `full-model class-logit IG`, never complete attention attribution.
4. Hooks, model mode, parameter gradients and `requires_grad` are restored on success/failure. Toy, repeatability and low-cost real-model smoke tests pass.

## C4: head-direction Value-path and matched patch baseline

1. For every candidate, clean `A`, fixed-A `g(x0)`, `d`, `r` and `u` follow section 4 of the protocol. `A/u` are detached and frozen through IG. `r<=tau_dir` produces `DEGENERATE_DIRECTION`; no fallback channel/head is used.
2. Endpoint parity is `<1e-6`; IG conservation is `<=1e-4+0.01|delta|`; all attempts and selected steps are retained.
3. All 197 `c_j` are computed from the same fixed A/u/V endpoints. Their sum passes `1e-6+1e-5|r|`; `c_0` is separate and only 196 patches form a map. Patch support/positive evidence degeneracies use frozen 0.01 thresholds and remain in tables.
4. 12×12 computation is selective/chunked without changing values. Synthetic exact conservation, real block parity, degeneracy and Stage B single-channel regression tests pass.

## C5: fidelity, stability and resources

1. Fidelity uses the correct scalar per line, normalized S response, normalized-zero replacement, RGB-coupled units, positive target-oriented ranking, exact 21 points and actual-fraction trapezoid AUC. No response clipping occurs.
2. Patch (196) and 4×4-cell (3136) operators share identical perturbed tensors across methods. Five hash-derived tie permutations are paired and aggregated; random/center sanity baselines are recorded. Value-path delta_fid only compares NNS to matched patch on fine-4 for the primary gate.
3. Top-20% sufficiency/gap/comprehensiveness use the same corruption operator and are reported without becoming duplicate decision gates.
4. Translation, crop-scale and Gaussian noise exactly match protocol parameters, seeds, inverses, Jacobian and masks. Geometric metrics use only completely valid patches; residual metrics use common valid pixels. Frozen clean class and A/u targets are enforced.
5. All-sample fixed-target results, unchanged-prediction descriptive results and flip rate are distinct. Stage B sensitivity sanity is not relabeled stability.
6. Wall time is synchronized; CUDA peaks are reset and allocated/reserved recorded; bytes are measured. Toy metrics, geometry round-trip, shared-tensor, CPU and conditional CUDA tests pass.

## C6: runner, retention, provenance and CLI

1. Estimate/run-sample/run-batch/validate/aggregate (or equivalent) operate from the frozen registry/config/selection contracts. Preflight expands all sample×method×config×target×perturbation×repeat×curve-point work and rejects any hard cap before GPU/write.
2. Each execution unit is immutable and independently finalized. Resume skips only `complete=true` units with matching manifest and all artifact hashes/bytes. A failed/incomplete/tampered unit is never appended or reported complete; rerun uses a fresh directory.
3. L0/L1/L2/L3 exactly follow protocol. L2 is limited to two audit IDs; L3 cannot be read by metrics; no per-step/per-pixel-file/per-dimension-file explosion occurs.
4. Provenance binds code clean state, model/image/lock/registry/protocol/config/selection, exact method/target/map/baseline/schedule/seeds, status/failure, resources and inventory. Failure table has every expected unit; batch completion requires an explicit terminal state for every unit.
5. Crash, resume, hash tamper, partial failure, missing row, non-overwrite and budget tests pass.

## C7: real one-image smoke and budget check

1. One audit image runs every method; NNS covers preregistered early/middle/late layers and heads sufficient to bound worst convergence without judging visual appeal.
2. Every method/config records synchronized time, CUDA peaks, invocation counts and bytes; hooks/grad/mode have no residue and no OOM occurs.
3. C8/C9 and Stage D projections use exact unit counts, slowest valid measured unit, 1.25 time factor and 25% storage overhead. Existing C0 caps are not loosened silently.
4. If C8/C9 exceeds a cap, C7 is BLOCKED or a pre-holdout amendment is committed. It may not silently reduce samples/layers/heads/steps/perturbations.

## C8: five-image calibration and frozen selection

1. Exactly the five calibration IDs and all 144 block/head candidates are processed. Holdout artifacts are inaccessible and selection records `holdout_accessed=false`. L2 exists only for the two audit IDs.
2. Eligibility requires 5/5 validity/provenance/resource completion. All candidates and failures remain recorded; no correctness/content/heatmap/display criterion is read.
3. Candidate tuple, six-decimal rounding, NA rank values and ascending block/head tie break exactly match protocol. Four bands each produce one winner; one of those is primary. Empty band means BLOCKED.
4. Machine-generated `selection.json` satisfies its schema, binds all inputs/artifacts and is committed from a clean tree before C9. Manual editing is rejected/detectable.

## C9: frozen 20-image pilot

1. C9 starts from the C8 clean commit and exact hashes of registry, protocol, acceptance, config, schemas and selection. No tuning or threshold/selection change occurs.
2. All 20 images run frozen methods/configurations with one canonical clean tensor and paired perturbation tensors. Every expected unit has an explicit terminal status and failure row.
3. Primary holdout validity requires all 15 complete, finite and provenance-valid; failures/degeneracies/misclassifications are reported, never removed.
4. Full hash/inventory validation passes; no PNG/display artifact is a quantitative input.

## C10: aggregation and decision

1. Output has one non-dropped row per expected sample/config/method/metric. Calibration (n=5), holdout (n=15) and descriptive all-20 are separate; only holdout feeds the primary decision.
2. Statistical unit is image. Paired median bootstrap uses exactly 10,000 image-cluster draws and the registered seed derivation/percentile CI. Layer/head/pixel/seed/perturbation never inflate n. Per-sample values, mean, median, sample SD and 95% CI are retained.
3. Validity fails on any required condition and has highest priority: output `NO_GO_VALIDITY`.
4. With validity passing, effect is PASS only if median delta_fid `>=0.02`, positive count `>=12/15` and CI lower `>0`; it is FAIL if count `<12` or CI upper `<0.02`, otherwise UNCERTAIN.
5. Spatial evidence requires 12/15 jointly at `s_patch>=0.01`, positive patch evidence / total token L1 `>=0.01`, and `rho>=0.05`. Each perturbation requires 12/15 Spearman `>0`, median `>=0.50`, CI lower `>0`; registered fail/uncertain boundary is applied without dropping NA rows.
6. Decision priority is exactly: validity fail → `NO_GO_VALIDITY`; any novelty FAIL → `NO_GO_NOVELTY`; no fail but any uncertainty → `INCONCLUSIVE`; novelty pass plus Stage D cost fail → `NO_GO_COST`; all pass → `GO`. Equality at CI boundaries is inconclusive.
7. Low raw-attention correlation/large JS alone is not novelty. No result claims equivalence, universal ImageNet superiority, native pixel evidence for upsampled maps, or that Value-path NNS is a class/complete-attention explanation.

## Required lightweight validation command for C0

```bash
conda run -n vitnns python tests/test_stage_c_contract.py
conda run -n vitnns python -m pytest tests/test_dataset_labels.py -q
```

The validator must perform no torch/model forward. After the C0 commit it must additionally assert every C0 contract file is Git tracked and the worktree is clean.
