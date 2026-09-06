# Stage E secondary/descriptive analysis protocol v1

Status: frozen by E0 on 2026-09-06. This protocol is subordinate only to
`docs/阶段E规划.md`. Stage E was planned after the D10 outcome was known. E0 is
therefore not a preregistration completed before Stage D outcomes.

## Scientific boundary

Stage E is descriptive sensitivity analysis. It does not alter the Stage D
primary, its 500-image denominator, or the D10 `GO_FIDELITY_SCOPE` decision. It
does not repair or alter the historical C10 `NO_GO_NOVELTY`, `FAIL_SPATIAL`,
`BLOCKED`, `HOLM_BLOCKED_UNDERSPECIFIED`, `downstream_authorized=false` state.
The C10 15 images and Stage D 500 images are never pooled.

Stage E emits no p-values, rejection decisions, winner selection, or new
GO/NO_GO. Every frozen variant is reported, including negative, null,
equivalent, failed, NA, and `NOT_RUN_WITH_REASON` variants. No Stage E result
may replace the D10 primary `(block=0, head=9)` fine-4x4 comparison.

The D9 sealed root, D10 output root, and D10 paper package are immutable inputs.
Only `evidence/stage_e/`, `reports/stage_e_*`, and the Stage E protocol, config,
schema, implementation, and test paths may be written. Stage E does not render
PNG, PDF, or SVG and does not manually select examples. Numeric figure inputs
come only from sealed `paper_data`.

## Outcome-access firewall

E0 may read sealed D9/D10 reports, validated L0/L1 arrays and curves, metadata,
the D1 registry and sample order, clean-prediction correctness, paths, hashes,
shapes, schemas, provenance, inventories, and outcome-free synthetic fixtures.

E0 may not execute E1-E3 on the real cohort; generate or read a new Stage E
effect, ranking, summary, winner, or figure; or change a variant, coordinate,
denominator, rule, or statistic after a new Stage E outcome is known. E0 PASS is
invalid if such access occurred.

## Common cohort and numerical contract

- Cohort is every row of `manifests/stage_d_fidelity_v1/registry.json`, in exact
  selection order 0 through 499. The ordered sample-ID SHA-256 is frozen in the
  machine-readable config. No replacement or row dropping is allowed.
- Strata are exactly `overall`, `correct`, and `misclassified`. Overall always
  contains all 500 rows; the two prediction strata partition it as 418 and 82.
  Correctness uses the D10 clean-prediction field and is not recomputed.
- The recovery-v2 model, preprocessing, fixed-clean-attention target, normalized
  zero / canonical RGB 0.5 replacement, fractions `0, .05, ..., 1`, five tie
  repeats, and target-oriented positive-part ranking are inherited unchanged.
- At fraction zero, `k=0`; otherwise `k=ceil(fraction*unit_count)`. The abscissa
  is `k/unit_count`. AUC is float64 trapezoidal integration over that abscissa.
  Within an image, each direction's AUC is the arithmetic mean across all five
  paired tie repeats. Repeats are nuisance resolution, never independent units.
- For methods A and B,
  `delta_fid = 0.5*((insertion_auc_A-insertion_auc_B) +
  (deletion_auc_B-deletion_auc_A))`. The statistical unit is image.
- Tie seed is the unsigned big-endian integer represented by the first eight
  bytes of SHA-256 over `namespace + sample_id + '|' + block + '|' + head +
  '|' + repeat`. Ranking is lexicographic score-descending, tie-priority
  ascending, where tie priority is a NumPy PCG64 permutation.
- Descriptive intervals use a paired image bootstrap with exactly 10,000 draws,
  NumPy 1.26.4 PCG64, arithmetic even median, linear 2.5/97.5 percentiles, and
  the namespace frozen in config. Stratum intervals resample images within that
  stratum and are descriptive, not simultaneous or inferential.
- Every expected row is retained. Terminal validity values are `VALID`,
  `FAILED`, and `NA`. Non-finite values are `NA` with reason `non_finite`, never
  JSON NaN/Infinity or zero-imputed. Failures retain class/code and immutable
  attempts. Infrastructure retry is at most one new immutable attempt after a
  coordinator authorization issued before scientific payload access; scientific
  invalidity and contract drift are not retryable.
- SHA-256 covers source bytes. Canonical JSON is UTF-8, sorted keys, compact
  separators, finite-only, with one terminal newline for files. Pair bindings
  include sample, model, image, target, method, block/head, scale, replacement,
  fractions, seeds, source map/curve, attempt, and producer identities. Whole
  roots use lexicographically ordered `{path,bytes,sha256}` records and the
  existing `canonical_sha256({"files": records})` convention.

## E1: baseline and patch-scale sensitivity

The D10 fine-4x4 NNS-versus-matched result is a read-only primary anchor. NNS
patch score `(r,c)` is the float64 sum of the signed 4x4 cell block
`[4r:4r+4, 4c:4c+4]` from its 56x56 L1 map. The matched patch score is the
float64 sum of the corresponding 4x4 block of its frozen uniform-mass 56x56
map, which deterministically recovers the native 14x14 patch contribution.
The separate CLS scalar is retained for conservation QC and never ranked.

Both patch-scale methods rank the same 196 native ViT patches and intervene on
the same 16x16 input-pixel patch per unit, with the common target, replacement,
fractions, and paired tie seeds. Patch-scale responses/AUCs require new forward
evaluation because fine-cell response curves cannot determine patch response
curves. E1 reports paired per-image results, summaries, valid/failed/NA counts,
cost, and differences from the read-only fine anchor. No patch result becomes a
new primary.

## E2: upsampling-rule sensitivity

Source coordinates are patch centers `(r+0.5,c+0.5)` on a 14x14 grid; target
coordinates are cell centers `(u+0.5,v+0.5)/4` on a 56x56 grid. Row is y and
column is x. Signed float64 patch values are transformed before applying the
target-oriented positive part for ranking.

For an interpolation rule, construct its 3136x196 linear operator by applying
PyTorch 2.2.2 `interpolate` to float64 14x14 basis images with scale factor 4.
Nearest uses `mode=nearest`; bilinear and bicubic use `align_corners=false` and
`antialias=false`. Boundary behavior is exactly that operator (edge-value
padding for the non-nearest PyTorch kernels). Normalize each source column by
its float64 column sum before applying it, so each signed patch contribution
retains unit total mass independently; column sums must be finite and nonzero.
No outcome-dependent or global signed-sum renormalization is allowed.

The four frozen rules are:

1. `uniform_mass_divide_by_16`: assign each patch value divided by 16 to its
   aligned 4x4 block.
2. `mass_preserving_nearest`: apply the column-normalized nearest operator.
3. `bilinear`: apply the column-normalized bilinear operator.
4. `bicubic`: apply the column-normalized bicubic operator.

Conservation is checked per synthetic source basis and per transformed map with
`abs(error) <= 1e-12 + 1e-12*abs(source_sum)` in float64. The uniform and
nearest operators are analytically identical because nearest repeats every
patch value in exactly 16 cells and column normalization divides by 16. Golden
tests must prove identical arrays, positive-part rankings, and tie behavior.
They are reported as one evidence class with nearest marked
`EQUIVALENT_RANKING`; they may not be counted as two independent supporting
results. Bilinear and bicubic require new matched-method forward responses. All
rules use the same image, target, fractions, and paired seeds.

## E3: layer/head and prediction-strata descriptions

Coordinates are exactly `(0,9)`, `(4,2)`, `(6,10)`, and `(11,2)`, in that
order. They are the four Stage C depth-band representatives frozen before
holdout access. `(0,9)` remains the Stage D primary anchor; the other three are
descriptive only. No head may be added or omitted, and captions must state that
these were selected in Stage C, not searched on the Stage D cohort.

For every coordinate and each of `overall`, `correct`, and `misclassified`, E3
reports n, valid/failed/NA counts, the frozen effect summaries, and descriptive
intervals. The strata are an effect-modification view, do not establish cause,
and do not replace the full-cohort conclusion. Clean margin or any other
continuous difficulty analysis is `NOT_RUN_WITH_REASON` in v1 because no such
variable, direction, and missingness contract is frozen here.

## E4: source validation, packaging, and sealing

E4 adds no scientific variant. It source-recomputes E1-E3 expected/observed
coordinates, pair bindings, hashes, attempt/failure/NA/retry state, summaries,
and actual cost. It separately reports package completeness, workflow integrity,
scientific validity, and cost. It retains every frozen variant and emits the
exact minimum files listed by `config/stage_e_e0.json`. Recipes declare
`numeric_source=paper_data_only` and `renders_png=false`.

E4 must prove that the D9/D10 roots, D10 decision, and C10 historical state are
unchanged, then record the Stage E ordered whole-root inventory, Git commit/ref,
and F0 entry. E4 PASS means the package is complete and traceable; it is not a
new Stage D decision.

## Gates and authorization

Each stage uses a pending-to-terminal atomic receipt. E0 PASS authorizes only
E1 and contains no ablation result. E1 PASS authorizes only E2; E2 PASS only E3;
E3 PASS only E4; E4 PASS only F0. FAIL or BLOCKED authorizes nothing. Missing
inputs, budget/authorization failures, untyped missing variants, upstream drift,
or outcome-firewall violations fail closed. Stage F cannot begin before E4 PASS.
