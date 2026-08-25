# Stage D configuration and typed run-spec contract

## Purpose and authority

`config/stage_d.json` is the frozen D2 scientific, statistical, recovery, retention, provenance and cost configuration. `config/stage_d.schema.json` is its closed Draft 2020-12 schema. `config/stage_d_run_spec.json` is the exact requested workload, and `config/stage_d_run_spec.schema.json` is its closed Draft 2020-12 schema.

`docs/阶段D_protocol.md` is authoritative for equations and cross-field scientific semantics that JSON Schema cannot express. `docs/stage_d_acceptance.md` assigns fail-closed gates to D2–D10. Neither prose document may broaden the machine config or run spec. Any disagreement among these five artifacts is a contract failure.

D2 supplies data contracts only. It deliberately does not provide a production Stage D config loader, typed planner, authorization builder, runner, finalizer, validator, statistics implementation or paper exporter. D3 begins strict loading/planning. The D2 files do not authorize model construction, reserve image access, formal execution or analysis.

## Closed parsing and hashing rules

Both schemas use JSON Schema Draft 2020-12 and recursively close objects with `additionalProperties:false`; every declared instance field is required unless the schema explicitly represents absence with a typed null. Arrays have frozen ordering/cardinality where order is scientific. D3 consumers must additionally reject duplicate JSON keys, `NaN`/`Infinity`, bool-as-int, implicit defaults, coercion, unresolved references and every cross-field contradiction below. Optimized Python must not silently disable these checks.

The frozen JSON instance must round-trip without normalization, inserted defaults or reordered scientific arrays. SHA-256 of a file covers its exact bytes. Canonical JSON uses UTF-8, sorted keys, compact separators and `allow_nan=false`; written JSON has one terminal newline. A document's `canonical_digest` is SHA-256 over the canonical object after removing only that top-level field. A matching self-digest is necessary but insufficient: validation must also bind upstream/source file SHA-256 values, canonical digests, completion refs and manifests, so schema-valid self-resigned tampering fails.

Relative repository paths must be normalized POSIX paths without `..`; formal roots must satisfy the later safe-root policy. Formal evidence admits ordinary single-link files only—no symlink or hardlink.

## `stage_d.json` top-level sections

| Section | Contract role |
|---|---|
| `$schema`, `schema_version`, `document_type`, `stage`, `milestone`, `canonical_digest` | document identity and self-digest |
| `protocol` | version, document/run-spec paths, amendment and result-status boundary |
| `authority` | preparation-only state and the only next task D2 may authorize |
| `upstream` | exact D0, D1 and sealed C10 paths/hashes/status bindings |
| `access_boundary` | D2's metadata-only attestation and prohibited actions |
| `cohort` | 500-image denominator, separation, misclassification and no-replacement rules |
| `model_and_preprocessing` | exact model, lock and preprocessing identity |
| `primary_scope` | fixed `(0,9)`, target, methods, endpoint and estimand |
| `target_and_map` | Value-path equations, thresholds, maps and fine-cell projection |
| `fidelity` | curves, tie seed/pairing, repeat aggregation and `delta_fid` |
| `registered_derived_metrics` | top-20% descriptive values derived from existing curves |
| `audit` | five overlay IDs and exact repeatability comparison |
| `workload` | exact typed counts and count formulas |
| `validity` | required evidence, numerical gates and invalidity behavior |
| `statistics` | image summaries and deterministic paired bootstrap |
| `sample_size_rationale` | 500/400 precision and 80% consistency rationale |
| `decision_gates` | ordered mutually exclusive component and overall decisions |
| `secondary_and_spatial` | no new spatial/secondary inference and conditional Holm status |
| `forbidden_workload` | explicit negative scope |
| `retry_and_recovery` | closed failure classes, attempt limits and fresh-root precedence |
| `retention` | L0–L3, immutable completion and no-result-leak policy |
| `provenance` | required bindings, canonical format and source recomputation |
| `sharding` | deferred count but frozen group-assignment algorithm |
| `cost` | launch-safety versus completed-campaign gates and D7 formulas |

## Identity, authority and upstream bindings

The root identity is exactly `stage=D`, `milestone=D2`, `document_type=stage_d_fidelity_protocol_config`, version `1.0.0`. `protocol.result_status=NOT_EVALUATED`. The amendment policy permits only a versioned change before D7 authorization and before any reserve image access; after access, original confirmatory status cannot be restored by editing a config.

`authority` must retain: preparation `APPROVED_TO_PREPARE_STAGE_D`; execution `NOT_ISSUED`; analysis `NOT_EVALUATED`; `stage_d_execution_started=false`; `does_not_override_c10=true`. A D2 PASS authorizes only `D3_CPU_CONTROL_PLANE_IMPLEMENTATION_ONLY`. No loader may infer execution authority from preparation status or a valid schema.

`upstream.d0` binds the D0 completion ref/commit and scope record file/canonical digests. `upstream.d1` binds the D1 completion ref/commit, six-file package manifest, registry/schema/source/metadata/attestation digests and ordered sample-ID digest. It explicitly says image hashes are candidate-recorded metadata and live equality is not evaluated. `upstream.c10` binds sealed report/decision/provenance/manifest/inventory identities and preserves every old failure/authorization state.

Consumers must verify these bindings from source bytes rather than trust copied values. D2 does not modify the sealed D1 attestation or C9/C10 evidence.

## Access boundary and cohort contract

`access_boundary` is an audit statement scoped to the current D2 metadata-only contract freeze. It must keep image-path stat, live hash, JPEG decode/preprocess, result/outcome access, model/inference/GPU use, runner implementation, smoke/formal run, paper generation and candidate-log/C9/C10 modification false. D2 may read already authorized D0/D1 metadata content. Static tests must guard this boundary and cannot claim to prove all historical access beyond the attested process and closed ledger scope.

`cohort` fixes 500 registry rows and `statistical_n=500`, contiguous unique selection order 0–499 and zero overlap with Stage C on the three D1 dimensions. It prohibits C10/Stage D pooling, sample replacement and complete-case analysis. Misclassified samples remain in the denominator; correctness may only create a descriptive stratum.

## Model, primary, target and map constraints

`model_and_preprocessing` binds the model config, state dict, dependency lock and exact RGB/248/bicubic/224-crop/0.5 normalization pipeline. These are future execution identities, not proof that D2 loaded the assets.

`primary_scope` contains exactly block 0/head 9; the target and two ordered methods must match the protocol. No selection field or alternative head exists. The endpoint is fine 4×4-cell `delta_fid`; image is the only statistical unit and fidelity is the primary hard scientific gate.

`target_and_map` serializes the normalized-zero baseline, clean fixed attention, detached direction, `1e-6` relative direction threshold, Gauss-Legendre schedule/chunk, map/accumulation dtypes, NNS/matched native-map semantics, 3,136-cell projection and separate CLS retention. Required cross-field rules include:

- baseline `g(x0)` uses clean fixed `A(x)`, not baseline attention;
- the NNS and matched method explain and evaluate the same frozen scalar;
- the matched map spatializes only patch contributions and retains CLS separately;
- `fine_cell_count=14*14*16=3136`;
- patch mass is divided equally by 16, while pixel attribution is summed within cells;
- signed mass is projected before positive splitting and must meet frozen tolerances;
- `native_single_pixel_claim_allowed=false`.

## Fidelity and derived-metric contract

`fidelity` fixes two directions, 21 ordered fractions, exact `ceil`/actual-fraction/trapezoid/no-clipping behavior and five tie repeats. `tie_seed_material` is interpreted literally with the runtime values substituted; SHA-256 extraction and PCG64 priority algorithm must match the protocol. The first-sample/repeat-0 golden uint64 seed is serialized as a decimal string to detect delimiter or byte-order drift without JSON precision loss. One priority is paired across both methods and both directions for a given sample/config/repeat.

All five repeat AUCs must exist. `within_image_auc_aggregation=arithmetic_mean_over_complete_five_repeats` feeds `delta_fid`; `within_image_auc_repeat_sd=population_ddof_0_over_complete_five_repeats` is a retained nuisance-repeat summary and does not feed the primary statistic. `positive_definition` is strict greater-than-zero. No incomplete-repeat mean is legal.

`registered_derived_metrics` uses the already present 20% curve coordinate. Its sufficiency/gap/comprehensiveness values are reported but create zero new curve points/work items and no additional decision gate.

## Audit overlay constraints

`audit` binds exactly the five D1 overlay IDs, sample IDs and selection orders. They must resolve bijectively to D1 registry rows marked audit and cannot be accepted as a new role or cohort.

`audit_repeatability_runs=2` includes the primary run, so `additional_repeat_clean_runs_per_method=1`. Repeatability scope is the clean metric-ready map. Before elementwise allclose with `atol=1e-6, rtol=1e-5`, the consumer requires identical shape, dtype, coordinate/projection identity and every applicable scientific binding. Only the enumerated telemetry fields—wall time, CUDA peak allocated/reserved and artifact bytes—are excluded; other provenance differences are not waived. Audit statistical-n increment is zero, L2 is limited to the audit IDs, and neither enters primary statistics/decision.

## Workload and run-spec expansion

The config's `workload` section carries formulas and expected totals. The planner must derive and compare them; it cannot accept them as unchecked claims:

```text
execution_units          = 500 * 2                       = 1,000
audit_repeat_clean_maps = 5 * 2 * (2 - 1)              = 10
map_work_items           = 1,000 + 10                   = 1,010
fidelity_curves          = 1,000 * 2 * 5                = 10,000
fidelity_curve_points    = 10,000 * 21                  = 210,000
logical_work_items       = 1,010 + 210,000              = 211,010
```

The two units within an image group are ordered NNS then matched and share all pair bindings. Clean maps and curve-point coordinates are execution work items; AUC, `delta_fid`, hashes, reconciliation and summaries are derived records. `logical_work_items_are_forward_or_backward_calls=false`, so budget accounting uses runner profiles rather than this total.

### Required run-spec envelope and semantics

The run spec is an exact request for the frozen config, not a second place to choose science. Its identity fields must bind schema version, document type, Stage D/D2, the config canonical digest and D1 registry/package identities. Its scientific fields must request exactly the 500 ordered D1 sample IDs, the one `(0,9)` configuration, the two methods in config order, fine-only insertion/deletion, all 21 fractions and repeats 0–4. Explicit booleans must keep stability, sanity, patch scale, secondary inference and display disabled; absence cannot mean a default.

The run spec must include the five audit IDs and two-run clean-map repeatability request without adding samples or units. Stored expected counts must equal the independently expanded plan. It cannot contain a selected-head slot, arbitrary method/config extensions, an outcome path, a result filter or an execution-authorization assertion.

D3 must define content-addressed typed records at least for image group, execution unit, clean-map item, audit repeat-clean-map item and fidelity curve-point item. Keys must include enough frozen context to prevent aliasing across sample, method, block/head, direction, repeat and fraction. Exact curve completeness means one coordinate for every Cartesian combination authorized by the run spec and no other combination.

## Validity contract

`validity` requires one authoritative terminal record for every 500 image and 1,000 unit coordinate, complete artifact/binding coverage and finite required values. It freezes strict endpoint parity, IG conservation and token parity formulas. Cross-field validation must apply each check to the correct method while requiring the paired scientific context to agree.

The closed `invalidating_na_reasons` list cannot be silently extended by a runner. Scientific failure is terminal evidence. A declared terminal algorithmic invalidity can produce a complete invalid package; missing or unverifiable package evidence remains an integrity blocker. Effect inference is structurally suppressed on invalidity. Row drop, imputation, replacement, display/L2 quantitative input and alternate target/head fallback are impossible.

Audit repeatability is required validity evidence. A consumer must reject a purported repeat that changes scientific identity even if numeric maps happen to compare equal.

## Statistics and sample-size rationale

`statistics` fixes all-500 ordered image-level `delta_fid` input, 10,000 image-cluster paired-median draws, PCG64, NumPy version, draw shape/type/order and linear percentile CI. For even `n=500`, the median is the arithmetic mean of the two middle order statistics; resampling ties retain their multiplicity. The namespace and metric ID produce one exact primary seed, with a serialized golden value to detect encoding drift. Tie repeats are already aggregated and cannot appear as bootstrap clusters. The ordered sample-ID digest prevents permutation drift. Invalid or incomplete input yields `NO_GO_VALIDITY` without bootstrap or complete-case analysis. The nested `descriptive_curve_band` freezes its separate method/direction/fraction-index seed material, 10,000 draws, image-cluster size 500, pointwise median statistic, PCG64 and linear percentile rules; it is descriptive, not simultaneous, and cannot enter the primary decision.

`required_summaries` includes per-sample values, positive count, mean, median, sample SD (`ddof=1`) and CI. This across-image sample SD is distinct from the fidelity section's five-repeat population SD (`ddof=0`). Figure bands have a distinct namespace and are neither primary nor simultaneous. No equivalence test is registered.

`sample_size_rationale` records a precision/replication design, not a formal power calculation. It preserves the 80% consistency fraction as 400/500 and records worst-case and p=0.8 binomial standard-error/normal-half-width values. These numeric facts do not replace the separate median and CI gates.

## Decision-gate constraints

`decision_gates.fidelity_evaluation_order` is normative: FAIL, then PASS, then INCONCLUSIVE. Implementations must not rearrange independent booleans and resolve overlap ad hoc. Exact equality behavior is serialized: 400 counts toward pass; median 0.02 counts toward pass; CI lower 0 does not pass; CI upper 0.02 does not fail; delta 0 is not positive.

The overall priority is validity, fidelity failure, fidelity uncertainty, completed-campaign cost, then all-pass. Allowed decisions are closed. Workflow `BLOCKED` is explicitly not a scientific decision and does not belong in that enum.

## Secondary/spatial and forbidden scope

`secondary_and_spatial` sets `run_stability=false`, new spatial role to mandatory disclosure only and all secondary inference booleans false. `holm_status=NOT_APPLICABLE_NO_SECONDARY_INFERENCE` is conditionally valid only while those booleans remain false; it cannot repair C10. Mandatory C10 spatial disclosure locations are Study Design, Results and Limitations.

`forbidden_workload` is defense in depth. Its false/empty fields are prohibitions, not inactive options that a run spec can turn on. Any request for random/center sanity, patch scale, spatial/stability, another method/head, class-logit line, display or secondary inference is outside the frozen protocol and rejected before model construction.

## Retry, recovery and root authority

`retry_and_recovery.retryable_infrastructure_failure_codes` is the entire allowlist. Only control status, closed failure code and resource telemetry may inform retry classification. The exact “no result formed” predicate requires both no valid terminal result manifest and no published L0/L1 scientific payload. Coordinator authorization must precede retry and bind the prior digest, same coordinate and new attempt ID.

`maximum_authorized_infrastructure_retries_per_unit=1` and `maximum_attempts_per_unit=2` are hard limits. Scientific failure is terminal, nonretryable and consumes an attempt. The paired unit still runs after the other unit's scientific failure. In a half-complete pair, the validated terminal unit remains immutable and only the pending retryable unit can receive a new attempt.

All nonretryable classes are closed, including OOM for the frozen workload, safety-cap breach, drift, tamper/unsafe root, ambiguity and post-result manual cancellation. Retry exhaustion or unsafe same-root state is `BLOCKED`.

A fresh-root rerun requires a documented diagnosis plus a new D7 authorization and covers all 500 images/1,000 units. A versioned amendment is additionally required only when the scientific contract changes, and it must precede any new access. No cross-root splice is legal. Superseded roots remain retained/disclosed; only a complete valid explicitly authorized replacement with a supersession record may become the analysis root.

## Retention and provenance

`retention` assigns all 1,000 units to L0/L1, restricts L2 to five audit IDs and sets L3 to none. It prohibits display PNG and file explosion, makes attempts immutable, defines the only resume-skip predicate, retains every expected failure row and requires every coordinate terminal for campaign completeness. CLI/log metric leakage is forbidden. D10 consumes only validated sealed L0/L1 and excludes L2.

`provenance.required_bindings` is a minimum set, not an invitation to omit coordinate-level identity. Every relevant artifact must bind producer clean state, authorization, model/lock, authorized live image identity, D0/D1/D2 contracts, plan/unit/work-item, method/target/operator, numerical schedules/seeds, terminal/failure/resource state and inventory. Source-recomputing validation and ordinary-file restrictions are mandatory.

## Sharding contract

`sharding.shard_count_status=DEFERRED_TO_D7_BEFORE_EXECUTION_AUTHORIZATION`. Therefore D2 and D3 must not invent a count, placeholder mapping or result-based partition. D2 does freeze the algorithm:

```text
shard_id = selection_order modulo frozen_shard_count
```

D7 must freeze one valid positive count and bind the resulting mapping before execution authorization. An image group cannot split across shards; a single coordinator hands off shards serially. Sharding changes scheduling only, never scientific coordinates or estimands.

## Cost categories and formulas

`cost.old_c10_pass_cost_authorizes_stage_d=false`; D7 must independently budget the Stage D typed plan and may not loosen D2 caps.

`launch_safety_caps` contains only per-unit/per-audit memory, bytes and call limits. A projected breach blocks launch. A runtime breach stops new claims, retains evidence and blocks workflow; `runtime_breach_is_no_go_cost=false`. Projected completed-campaign gates must also pass before launch.

`completed_campaign_cost_gate` contains only total retained bytes and total/per-image-group H20-GPU-hour limits. Actual cost includes every attempt and audit repeat. It is evaluated after a complete campaign; aggregate overrun alone does not truncate a running campaign. If validity and fidelity pass but this gate fails, component status is `FAIL_COST` and overall decision is `NO_GO_COST`.

`d7_projection_formula` is normative. Time uses 1.25 safety factor; storage includes 25% payload/control/provenance overhead. `logical_work_items_used_as_call_count=false`. D2 leaves `budget_status_at_d2=NOT_EVALUATED`.

## D3 semantic validation matrix

In addition to schema validation, D3 must prove at least these relationships fail closed:

- a D0/D1/C10 hash, ref, status or authority drift;
- any reserve-live-equality claim at D2;
- any registry order/count/audit mismatch or C10/Stage D pooling;
- any model/preprocess/target/method/order/block/head/fine-cell drift;
- missing, extra, reordered or duplicate fractions/directions/repeats/methods/images;
- tie seed not paired across methods/directions or repeat SD not `ddof=0`;
- audit repeat on a non-audit image, a third run, a statistical-n increment or telemetry treated as scientific equality;
- stored workload count disagreement with independent expansion;
- a logical work-item count used as a forward/backward call estimate;
- stability, patch scale, sanity, extra method/head, secondary inference or display enabled anywhere;
- a validity threshold/equality/list drift, row-dropping or effect inference after invalidity;
- bootstrap namespace/PRNG/version/shape/quantile/order/dtype or equality behavior drift;
- unknown retry code, a second infrastructure retry, a third attempt, scientific retry, result-informed retry or mutated prior attempt;
- cross-root splice, partial fresh-root rerun or missing supersession/authorization;
- L2 outside five audits, L3/PNG, result-bearing logs or incomplete terminal coverage;
- a cap assigned to both/neither category, safety breach relabeled `NO_GO_COST`, or completed-cost early stop;
- shard count chosen before D7, non-modulo assignment, split image group or scientific-coordinate change;
- any preparation artifact claiming execution/analysis authorization or a Stage D result.

These checks must use metadata/static or synthetic fixtures through D6. No schema, loader or test success grants authority to open a reserve JPEG, construct the model or inspect an outcome.
