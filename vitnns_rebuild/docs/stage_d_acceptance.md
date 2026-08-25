# Stage D fidelity-only acceptance contract

## Purpose and fail-closed rule

This document defines the acceptance gates for the prospective Stage D fidelity-only replication/extension. Exact values are frozen in `config/stage_d.json` and `config/stage_d_run_spec.json`; equations and cross-field semantics are defined in `docs/阶段D_protocol.md`. A later milestone passes only when every gate assigned to it passes against those exact contracts. Unknown, missing, duplicate, non-finite, coerced or contradictory data fail closed.

A complete, structured scientific failure is evidence. An absent coordinate, changed denominator, unbound artifact, ambiguous attempt, unauthorized root or silent exclusion is not scientific evidence and makes the workflow `BLOCKED`. `package_complete`, `scientific_validity`, component status, scientific decision and authorization are separate states and may not be substituted for one another.

The D2 freeze does not authorize execution or analysis. Until a valid D7 authorization exists, `execution_authorization=NOT_ISSUED`; until D9 separately authorizes analysis, no root may be used for the D10 decision. No preparation milestone may rewrite C10's historical `NO_GO_NOVELTY`, `FAIL_SPATIAL`, `BLOCKED`, `HOLM_BLOCKED_UNDERSPECIFIED`, `downstream_authorized=false` or `stage_d_started=false`.

## D2: protocol, statistics and typed run-contract freeze

1. The D0 scope decision and D1 six-file registry package are bound by exact paths, file SHA-256 values and canonical digests. The frozen cohort has exactly 500 rows in unique, contiguous selection order `0..499`; C10 `n=15` remains separate and pooling to `n=515` is prohibited.
2. `docs/阶段D_protocol.md`, this acceptance contract and `docs/stage_d_config_contract.md` agree with the closed config and run spec on every scientific choice, authority state, upstream binding and expected count. No unresolved result-dependent choice is handed to D3.
3. The only primary is `(block=0, head=9)`, with target `fixed_clean_attention_cls_head_direction`, exactly two methods in NNS-then-matched order, and endpoint `fine_4x4_cell_delta_fid`. Fine means 3,136 non-overlapping 4×4 cells, not native pixels.
4. Fidelity has exactly insertion and deletion, fractions `0,0.05,...,1.0`, five paired tie repeats `0..4`, actual-fraction trapezoidal AUC and no response clipping. Tie seeds are Stage-D namespaced, PCG64-derived and paired across methods and directions. The five-repeat AUC SD is population SD (`ddof=0`); repeats do not increase statistical `n`.
5. The five D1 audit IDs are an overlay on the 500. `audit_repeatability_runs=2` includes the primary run and adds exactly ten repeat-clean maps. Required metric-ready maps are compared elementwise with the frozen tolerances after exact shape, dtype, coordinate/projection and scientific-binding checks; wall time, CUDA peaks and artifact bytes are excluded telemetry. Audit/L2 contributes zero observations and never enters the primary statistic or decision.
6. The typed formulas independently yield exactly 500 image groups, 1,000 execution units, 1,000 primary clean maps, 10 audit repeat-clean maps, 1,010 map work items, 10,000 fidelity curves, 210,000 curve points and 211,010 logical work items. Logical work items are not forward/backward-call counts.
7. `run_stability=false`. The run contract cannot express patch-scale fidelity, random/center sanity, another method, class-logit line, another block/head, spatial/stability work, secondary inference, winner selection or display work. Top-20% derived measures add neither work items nor decision gates. `holm_status=NOT_APPLICABLE_NO_SECONDARY_INFERENCE` is valid only under this no-secondary-inference condition and does not repair C10.
8. Validity requires exact 500-image/1,000-unit reconciliation, finite and bound evidence, endpoint parity `<1e-6`, IG conservation `<=1e-4+0.01|delta|`, token parity `<=1e-6+1e-5|r|`, nondegenerate direction/target and all audit repeatability checks. Misclassified samples remain; missing/failed/invalid evidence cannot be dropped, imputed, replaced or converted to zero.
9. Statistics freeze one image-level `delta_fid` per image, strict `delta_fid>0` positivity, all-500 paired median bootstrap, exactly 10,000 draws, Stage-D namespace, PCG64, NumPy `1.26.4`, linear percentile 2.5%/97.5% quantiles and complete serialization/digest semantics. Invalidity suppresses bootstrap and complete-case effect inference.
10. The `400/500` sign guard preserves the pre-existing 80% directional-consistency proportion (`12/15`) at the new sample size. The documented binomial precision calculations are correct and are not presented as a formal power claim.
11. Fidelity is evaluated in order: count `<400` or CI upper `<0.02` is FAIL; otherwise median `>=0.02`, count `>=400` and CI lower `>0` is PASS; otherwise INCONCLUSIVE. Tests cover 399/400, count 400, median 0.02, lower 0, upper 0.02 and per-image delta 0 exactly.
12. Overall priority is validity → fidelity → completed-campaign cost. Allowed scientific decisions are exactly `NO_GO_VALIDITY`, `NO_GO_FIDELITY`, `INCONCLUSIVE_FIDELITY`, `NO_GO_COST` and `GO_FIDELITY_SCOPE`. Workflow `BLOCKED` is not a scientific decision.
13. The infrastructure retry allowlist and nonretryable classes are closed. A coordinator may authorize at most one infrastructure retry per unit, for at most two attempts total, only before a scientific result is formed and without reading scientific values. Scientific invalidity is terminal. All attempts are immutable and retained.
14. Same-root resume, half-pair recovery, exhausted retry, unsafe-state and fresh-root rules are complete. A fresh-root recovery requires a documented diagnosis plus new D7 authorization and reruns all 500 images/1,000 units; a versioned amendment is additionally required only if the scientific contract changes. Cross-root splicing and selective rerun are impossible.
15. Retention is exactly L0 for all units, L1 for all primary maps/curves, L2 only for the five audit IDs and no L3/PNG. Provenance, source recomputation, no-symlink/no-hardlink, immutable attempt and no-result-leak requirements are frozen.
16. Every existing cap belongs to exactly one category. Per-unit bytes/calls/CUDA and per-audit L2 bytes are launch-safety caps; total retained bytes and total/per-image H20-GPU-hours are completed-campaign gates. A safety breach blocks workflow and is not `NO_GO_COST`; actual completed cost includes all attempts and audit repeats.
17. D2 does not inspect, `stat`, hash, decode or preprocess reserve JPEGs; does not access Stage D outcomes; does not load a model or run inference/GPU; does not implement D3+ code; does not run smoke/formal work; and does not issue execution or analysis authorization. Static and synthetic validation only is allowed.
18. Both schemas are Draft 2020-12 and recursively closed. Strict parsing rejects duplicate JSON keys, NaN/Infinity, bool-as-int, implicit defaults, unknown/missing keys, wrong order/cardinality/type/value and unresolved or inconsistent references. Exact frozen documents round-trip without normalization or inserted defaults.
19. Canonical self-digests are recomputed with the frozen self-excluding algorithm, and file SHA-256 covers exact bytes. Schema-valid but self-resigned drift still fails against upstream/source bindings, the D1 package manifest and the D2 completion commit.
20. D2 changes only preparation contracts. No runtime evidence, paper data, result table or figure exists, and D2's budget/result/scientific-validity fields remain `NOT_EVALUATED`.

Any failed item makes D2 `BLOCKED`; it cannot authorize D3 as if the protocol were frozen.

## D3: strict CPU-only control plane

1. Strict loaders implement the closed config and run-spec contracts without defaults or coercion and reproduce their exact canonical forms. Mutation tests reject every D2-forbidden scope and all malformed or contradictory fields.
2. The typed planner expands the frozen 500 registry rows into exactly 500 image groups, 1,000 ordered units, 1,010 map work items and 210,000 curve-point items. It recomputes all totals and rejects stored-total drift, duplicates, omissions or extra coordinates.
3. Each `image_group_id`, unit key, pair binding and work-item key is content addressed. Both methods share the same image/model/target/baseline/fraction/tie/operator context; group order follows registry selection order and method order is NNS then matched.
4. The plan represents audit repeat-clean work only for the five D1 audit IDs and cannot increase `statistical_n`. It cannot create forbidden scale, sanity, stability, method, head or display work.
5. Shard count remains explicitly deferred. The control plane validates that D7 later supplies one positive frozen count and assigns `selection_order modulo shard_count`, without splitting an image group.
6. Authorization construction preserves all C10 and D0/D1/D2 facts, refuses a dirty/drifted producer and cannot sign execution before all D3–D7 gates exist. Planning and estimation never imply execution authorization.
7. Every preflight check that can be performed from metadata occurs before model construction. D3 uses only metadata or synthetic fixtures; it performs no reserve access, model inference, GPU work, smoke or formal execution.
8. Unit, retry, terminal, claim, shard and authorization enums are closed and match the protocol. No code may infer retry eligibility from scientific values.

## D4: image-group runner, immutable attempts and provenance production

1. The runner consumes only a validated authorized plan and owns one complete image group at a time. It runs the two units in frozen order, keeps the paired context identical and releases model/input/cache ownership at the group boundary.
2. Runtime code reuses the validated Stage C mathematical primitives while producing Stage-D-specific provenance; no formal artifact falsely binds Stage C config, selection or stage identity.
3. All 21×5×2 curve coordinates and registered clean-map coordinates are explicit and complete. The runner cannot add patch-scale, stability, sanity, other methods/heads or display work.
4. Attempts and unit directories are immutable. Resume skips only a fully validated terminal unit. A half-complete pair follows the D2 recovery rule, and the paired unit still reaches a terminal record after the other's scientific failure.
5. Only closed, outcome-blind infrastructure codes can request retry. Coordinator authorization binds the prior digest and new attempt; one retry/two attempts is enforced. Scientific failures, OOM for the frozen workload, drift, tamper, ambiguity and cap breaches cannot be reclassified as retryable.
6. Runtime logs reveal only control/progress/terminal/resource information. They never expose maps, AUCs, `delta_fid`, rankings or summaries before authorized analysis.
7. L0/L1/L2 production, artifact limits, source bindings, synchronized resource measurement and complete failure rows meet the retention/provenance contract. No L3 or forbidden file explosion is produced.
8. D4 tests use synthetic/mock inputs only and do not run the reserve campaign or issue execution authorization.

## D5: finalize, reconciliation and source-recomputing validation

1. Finalization performs an exact left join over all 500 images, 1,000 unit coordinates, attempts, terminal states, manifests, artifacts and provenance. Duplicate, missing, extra or ambiguous authority fails closed.
2. Validators recompute all hashes, canonical digests, file sizes, inventories, upstream/authorization bindings and scientific parity/conservation/repeatability checks from source evidence. Self-reported success or self-resigned tampering is insufficient.
3. Package completeness is separate from scientific validity. A complete terminal scientific failure may be sealed for `NO_GO_VALIDITY`; missing/tampered/unauthorized/incomplete evidence makes the workflow `BLOCKED`.
4. Audit repeatability is checked on the exact metric-ready maps and frozen scientific bindings; excluded telemetry cannot hide a map mismatch. L2 is verified but cannot feed a quantitative statistic.
5. Active claim, stale claim, partial write, symlink/hardlink, manifest substitution, cross-root splice and supersession tamper tests fail closed.
6. D5 operates on synthetic/tampered fixtures only and does not access Stage D reserve images or outcomes.

## D6: preimplemented statistics, truth table and paper export

1. CPU-only analysis consumes synthetic sealed fixtures through the same D5 validation boundary that formal evidence will use. It reads no JPEG, model or display artifact.
2. The implementation produces exactly 500 coordinate rows before any effect calculation. Invalidating evidence yields `NO_GO_VALIDITY`, suppresses bootstrap/effect plots and creates `NOT_GENERATED_DUE_TO_VALIDITY` placeholders; no complete-case branch exists.
3. Golden tests reproduce the exact per-image repeat aggregation (`ddof=0` for tie-repeat AUC SD), `delta_fid`, strict-positive count, across-image summaries (`ddof=1`) and deterministic 10,000-draw PCG64/linear bootstrap bytes and digests.
4. Truth-table tests cover every pass/fail/inconclusive combination and every registered equality boundary. Validity, fidelity and completed-cost flags are retained even though only one overall decision is emitted.
5. C10 `n=15` and Stage D `n=500` remain separate in every schema, table and plot recipe. C10 historical statuses remain visible and unchanged.
6. Numeric figures consume only machine-readable paper data. Figure bootstrap uses its independent namespace, is pointwise/descriptive and never changes the primary decision. L2/PNG cannot enter any calculation.
7. Synthetic outputs are byte deterministic under the frozen dependency/serialization contract. D6 produces no formal Stage D result and cannot issue execution or analysis authorization.

## D7: known-input smoke, budget, producer seal and execution authorization

1. Smoke uses only an already accessed Stage C audit/calibration image or synthetic data in an isolated `formal=false` root. It never opens a Stage D reserve image, and smoke artifacts cannot enter D8–D10 evidence.
2. The complete two-method fine-only path passes method, map, curves, finalize and validation checks without residual hook/gradient/model-state contamination or OOM. All smoke work is typed and separately counted.
3. D7 independently projects the 1,000-unit Stage D campaign from conservative validated profiles, all attempts/audit work, exact typed counts, 1.25 time factor and 1.25 storage/control-provenance factor. The C10 cost result is historical evidence only.
4. Every launch-safety cap and every projected completed-campaign gate passes. D7 may tighten but cannot loosen D2 caps. A failed projection is `BLOCKED_NO_LAUNCH`, not a reduced workload.
5. D7 freezes one positive shard count. The complete plan maps every image group by `selection_order modulo frozen_shard_count`; no group splits, no coordinate changes and serial coordinator handoff is enforced.
6. A clean producer commit is sealed only after all affected non-GPU tests and the isolated smoke/budget checks pass. Authorization binds code, dependencies, model, D0/D1/D2, exact plan, budget, shard mapping and all schemas/digests.
7. Only a valid D7 artifact may set execution scope to `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`. It does not authorize analysis, any spatial endpoint, extra method/head/scale or any C10 rewrite.

## D8: formal 500-image execution campaign

1. D8.0 begins at a new, empty, safe formal root and validates authorization, plan, registry, ledger, Git/model/dependency hashes and shard mapping using metadata before model construction or image access.
2. Before the first reserve access it durably creates immutable controls, pending/shard ledger, append-only access journal and `ACCESS_STARTED` lifecycle record. Only then may it read/hash/decode each authorized JPEG and compare live bytes to the D1 recorded identity; progress is durable and D1's historical attestation is never edited.
3. All live assets must validate before the campaign becomes runnable. Any failure preserves the accessed/blocked root and exact access scope; it cannot be described as unaccessed.
4. The coordinator owns serial shard transitions. Only a verified terminal receipt authorizes the next shard. Free-text summaries cannot schedule work; the canonical ledger and its digest are authoritative.
5. All 500 groups and 1,000 units receive explicit terminal records. A complete scientific failure remains terminal evidence and does not prevent execution-completeness PASS; it does prevent later scientific validity PASS.
6. Retry, half-pair and attempt limits exactly follow D2. A tampered, drifted, unsafe or ambiguous root is `BLOCKED`; a required fresh-root recovery reruns the entire campaign under new authorization and never splices evidence.
7. D8 does not aggregate `delta_fid`, inspect scientific rankings, early-stop on effect, choose examples or emit a scientific decision.

## D9: seal, independent validation, QC and analysis authorization

1. D9 is CPU-only and reads no JPEG, constructs no model and uses no GPU. It freezes D8 writes, verifies the authorized root, and source-recomputes its complete status/manifest/inventory/provenance chain.
2. Exact 500/1,000 reconciliation, pair bindings, attempt history, scientific validity conditions, audit repeatability and actual resource/cost records all validate. L0/L1/L2 roles remain enforced.
3. D9 may produce QC, failure/NA, repeatability and actual-cost evidence but does not calculate aggregate `delta_fid`, bootstrap or fidelity decision. Projected D7 cost and actual D9 cost are labeled separately.
4. A package-complete root containing recorded scientific invalidity may authorize D10 solely to emit `NO_GO_VALIDITY`. A missing, tampered, unauthorized, unsafe or incomplete root is `BLOCKED` and receives no analysis authorization.
5. Completed-campaign cost uses all attempts and audit repeats and is evaluated only after complete evidence. Safety-cap history remains a workflow fact and is not relabeled `NO_GO_COST`.

## D10: sole formal statistics, decision and paper package

1. D10 consumes only the D9-sealed, explicitly analysis-authorized root and writes to a separate new postprocessing root. D9 evidence remains read-only.
2. It first atomically builds and validates a machine-readable decision package: complete per-coordinate table, aggregate curves, summaries, bootstrap metadata/distribution digests, component flags, overall decision, provenance and inventory.
3. Validity failure yields only `NO_GO_VALIDITY` plus complete QC/failure tables. Effect/bootstrap/main figures are structurally `NOT_GENERATED_DUE_TO_VALIDITY`; no valid-subset analysis exists.
4. With validity passing, the exact ordered fidelity truth table and equality behavior are applied. Cost is considered only after fidelity PASS. Workflow failures remain `BLOCKED`, not scientific decisions.
5. C10 and Stage D are reported as separate cohorts and C10 retains `NO_GO_NOVELTY / FAIL_SPATIAL / BLOCKED`. Mandatory C10 spatial, Value-path, 4×4-not-pixel, misclassification, failure, no-spatial-Stage-D and cost-boundary disclosures are present.
6. Paper tables and numeric figures are generated only from the validated machine-readable package; a plotting/export failure cannot alter the scientific decision and is reported as an export blocker. Qualitative overlays remain outside D10.
7. Claims remain limited to the frozen model, cohort, primary, target, two methods and fine-4×4-cell endpoint. No statistical-equivalence, universal-superiority, native-pixel, complete-attention/class, spatial-robustness or pooled-`n=515` claim is emitted.
