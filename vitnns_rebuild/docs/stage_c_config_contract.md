# Stage C configuration and schema contract

## Purpose and authority

`config/stage_c.json` is the frozen C0 scientific configuration. `config/stage_c.schema.json` describes its document envelope and the closed structures whose shape is already stable; `config/stage_c_registry.schema.json` specifies the CSV row contract; `config/stage_c_selection.schema.json` specifies the future C8 selection artifact. The equations, cross-field constraints and enumerated choices in `docs/阶段C_protocol.md` are authoritative when JSON Schema cannot express them.

C0 deliberately supplies data contracts only. It does not provide a production `StageCConfig` parser, map/metric implementation, method dispatch, fidelity/perturbation operator, runner or selection implementation. C1 must implement a strict loader with duplicate-key and non-finite-number rejection and must test it against this frozen instance. C8, not C0, creates `selection/stage_c_selection.json`.

Every object is closed: the C0 main schema is an exact-instance Draft 2020-12 schema whose section-level `const` values reject any nested missing/unknown field, type/value drift or array reordering. The registry and future selection use typed closed schemas plus semantic checks for relationships that JSON Schema cannot express. A consumer must also reject JSON `NaN`, `Infinity`, duplicate keys, silent defaults and coercion from strings to numbers/booleans. Relative paths must be normalized POSIX paths below the repository/workspace roots specified by the field; they must not be absolute or contain `..`.

## Top-level fields

| Field | Type | Meaning | Required constraints |
|---|---|---|---|
| `$schema` | string | Local schema reference | exactly `stage_c.schema.json` |
| `schema_version` | integer | Contract version | exactly `1` |
| `document_type` | string | Document discriminator | `stage_c_protocol_config` |
| `stage` | string | Experiment stage | `C` |
| `protocol` | object | Human-readable authority | path, semantic version, amendment rule |
| `registry` | object | Frozen sample inputs | registry/source paths, hashes, devkit hash and 20/5/15/2 counts |
| `roles` | object | Statistical-use policy | calibration/holdout are exclusive; audit is an overlay |
| `methods` | array | Exact method identities | seven unique IDs and their target/comparison contracts |
| `targets` | object | Scalar targets and IG settings | baseline, Value-path, class-logit and repeatability definitions |
| `layers_heads` | object | Model geometry and candidate/application scope | 12×12, token geometry, rollout and Grad-CAM details |
| `projection` | object | Numeric map transformations | 224↔14 mass preservation, 4×4 cells, signs and dtype |
| `metrics` | object | Required metric semantics | correlation/distribution/hotspot/entropy/NA/resource rules |
| `fidelity` | object | Intervention contract | target orientation, units, fractions, AUC, seeds and eligible pairs |
| `perturbations` | object | Stability interventions | shared tensor domain and three exact perturbations |
| `retention` | object | L0–L3 and recovery contract | audit IDs, forbidden artifacts, completion/hash rules |
| `budget` | object | Hard resource limits | byte, call, memory and H20-GPU-hour caps |
| `selection` | object | C8 frozen selection algorithm | calibration only, filters, ranking, ties and output |
| `statistics` | object | Estimand and resampling | image unit, n=15, 10,000 paired image bootstrap |
| `decision_gates` | object | C10 machine decision | validity/effect/spatial/cost thresholds and priority |

## Registry fields

The registry is CSV because it is directly auditable. Its header is exact and ordered; extra or missing columns fail. Values are parsed strictly according to `config/stage_c_registry.schema.json`, never accepted merely as arbitrary strings.

| Column | Meaning | Constraint |
|---|---|---|
| `schema_version` | row schema | integer 1 |
| `sample_id` | Stage A stable ID | `ILSVRC2012_val/<filename>`; unique |
| `image_id` | extension-free image ID | exact frozen 20-ID order; unique |
| `relative_path` | path below flat validation root | basename only; unique |
| `image_sha256` | exact JPEG bytes | lowercase SHA-256; unique in this registry |
| `size_bytes` | exact file size | positive integer |
| `width`, `height` | original decoded dimensions | positive integers, before preprocessing |
| `official_wnid` | official ground truth | ImageNet leaf WNID from locked devkit |
| `imagenet1k_index` | model output index | canonical 0-based index 0..999 |
| `label_source` | label provenance | exactly `official_devkit_val_gt` |
| `role` | statistical role | `calibration` or `holdout` |
| `audit` | L2 overlay | strict boolean; true only for two calibration IDs |
| `selection_order` | order in frozen 20 | unique integers 0..19 |
| `source_selection_index` | zero-based row in sorted 500 | exact historical linspace indices |
| `selection_rule` | immutable rule ID | exact registered string |

The metadata document binds the registry to the 500-row Stage A manifest, devkit, synset order, legacy read-only evidence, dependency lock, generation Git state and factual asset limitations. It intentionally does not contain its own hash. The config binds its hash, avoiding circular identities.

## Method and target fields

Every method has `id`, `exact_name`, `comparison_line`, `explanation_target_ref`, `fidelity_target_ref`, `class_agnostic`, `fidelity_eligible`, `structural_reference_only`, `native_map` and `projection_ref`. A null explanation target is intentional only for class-agnostic attention maps; a non-null fidelity target says which frozen model response may be used to assess that ranking and does not turn it into a class attribution.

Required cross-field rules are:

- the head-direction NNS and matched patch-token method share `fixed_clean_attention_cls_head_direction` for explanation and fidelity;
- matched per-head raw CLS attention has no fidelity target and is structural-reference-only;
- class-line direct attention and rollout have no explanation target but may rank interventions evaluated against `clean_predicted_class_raw_logit`;
- Grad-CAM and full-model class-logit IG explain and evaluate that raw logit;
- no method may be named or flagged as complete attention attribution;
- all ID and target references must resolve exactly once.

`targets.baseline` fixes normalized zero/canonical RGB 0.5. `targets.ig` fixes quadrature and convergence. `fixed_clean_attention_cls_head_direction` fixes query, head dimension, clean attention at both endpoints, direction threshold, detach/freeze, token parity, spatial support and separate CLS retention. `clean_predicted_class_raw_logit` freezes clean argmax and retains misclassified samples.

## Layers, projection and metrics

`layers_heads` freezes ViT geometry, all 144 C8 candidates, four complete disjoint depth bands, the C8 selection reference, block-11 direct attention, exact rollout algebra and `blocks[11].norm1` Grad-CAM behavior. Lists are unique, ascending and range checked. Global methods are not assigned fake block/head values.

`projection` defines only numeric, mass-preserving operators. `pixel_to_patch_operator` sums 16×16 values; `patch_to_pixel_operator` divides patch mass by 256; fine cells sum 4×4 values and matched patches divide mass by 16. Projection happens before positive/negative splitting. Display artifacts are forbidden inputs.

`metrics.required` is the exhaustive minimum output set. Correlations, distribution branches, base-2 JS/KL, top fractions/ties, entropy normalization and structured NA are fixed by adjacent fields and the protocol. `invalidating_na_reasons` fail a required primary execution unit; other legitimate NA values remain explicit and are never imputed as zero.

## Fidelity and perturbations

`fidelity` fixes replacement baseline, target orientation, RGB-coupled spatial units, 21 curve points, actual-fraction trapezoid AUC, patch and 4×4-cell scales, five deterministic tie replicates, sanity rankings, top-20% sufficiency/comprehensiveness, and valid comparison groups. Random repeats are paired nuisance resolution, not independent samples.

Each item in `perturbations.items` is a discriminated contract. Translation fixes displacement/padding/inverse/mask; crop-scale fixes crop, image/map interpolation, antialias, Jacobian and mask; noise fixes domain/distribution/generator/seed/clamp. Clean class and Value-path targets stay frozen, and methods share exact perturbed tensors.

## Retention, budget, selection and decisions

`retention.levels` specifies L0–L3; the other fields forbid display feedback, overwrites and unsafe resume and require every expected unit to have a retained terminal row. L1 remains float32 in Stage C 1.0.0; quantization requires a pre-holdout amendment.

`budget` values are hard positive caps. Counts include interventions, repeats, curve points, provenance and control artifacts. C7 may measure and tighten but not silently loosen them.

`selection` defines C8 policy but contains no selected head. It requires all five calibration images, prohibits holdout access, applies validity first, ranks a fixed tuple, uses deterministic ties, selects one candidate per band and one primary, and blocks on an empty band. The future selection document must satisfy the closed, typed `stage_c_selection.schema.json`: exact five IDs; enumerated code/model/lock/registry/protocol/acceptance/config/schema/calibration-artifact bindings; 144 typed candidates; four typed winner references; one typed primary reference; structured failures; and `holdout_accessed=false`. Semantic validation additionally requires unique complete 12×12 `(block,head)` coverage, candidate-key/index/band agreement, one referenced winner per band, the primary to be one of those four, exact winner/primary flags, eligibility/ranking consistency and failure references.

`statistics` and `decision_gates` freeze the image-level estimand, paired bootstrap and complete decision truth table. All thresholds and equality behavior are in the config; C8/C9 outcomes cannot select them.

## Hashing and provenance rules

SHA-256 always covers exact file bytes. JSON is UTF-8 with a terminal newline; CSV is UTF-8 with one header. Runtime provenance binds the exact registry, registry metadata, source manifest, protocol, acceptance contract, config, schemas, dependency lock, model, image and (after C8) selection hashes. No tracked file is required to contain its own hash or the hash of the commit that contains itself; the append-only completion record and Git history supply commit binding without an impossible self-reference.
