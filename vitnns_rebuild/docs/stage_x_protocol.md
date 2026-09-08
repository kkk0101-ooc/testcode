# Stage X NNS-v2 broad-claim prospective protocol v1

Status: frozen by X0 on 2026-09-08, before X1 implementation, new dataset or
checkpoint materialization, or any Stage X empirical outcome. Protocol ID:
`STAGE_X_NNS_V2_BROAD_CLAIM_V1_20260908`. The literature cutoff is
2026-09-08. X0 is CPU-only and metadata-only.

## 1. Scope and immutable history

Stage X is a new prospective study. It does not reinterpret or overwrite C10,
D10, E4, or Stage F. The following facts remain immutable: C10 acceptance
`BLOCKED`, decision `NO_GO_NOVELTY`, spatial `FAIL_SPATIAL`, and Holm status
`HOLM_BLOCKED_UNDERSPECIFIED`; D10 `GO_FIDELITY_SCOPE` for ViT-B/16, `(0,9)`,
the head-direction Value path, matched patch baseline, and fine-4x4 endpoint;
Stage E scientific validity `MIXED`, including the E2 owner waiver and E3
orders 117 and 352; and the F3C primary contrast
`class-targeted NNS - ViT Grad-CAM`, whose sealed 50-image mean is
`-0.05585432814553341`, 95% CI
`[-0.07085058706573369,-0.04100117599983641]`, and sign-flip p-value
`9.99990000099999e-06`. F3C produced no full-method ranking, unique winner,
GO/NO-GO, or fine-4x4 class-targeted intervention terminal.

The strongest permitted Stage X outcome is `GO_BROAD_BENCHMARK_SCOPE`. Its
paper wording is: "Across the preregistered benchmark of models, datasets,
methods, and metrics, NNS-v2 outperformed the evaluated baselines." A logical
or universal superiority claim is outside the registry.

## 2. Registered claims

The machine registry `claim_registry.json` defines eight independently
falsifiable claims: `X-CLM-BROAD`, `X-CLM-SPATIAL`, `X-CLM-COMPLETE`,
`X-CLM-PIXEL`, `X-CLM-GRADCAM`, `X-CLM-LOCALIZE`, `X-CLM-XMODEL`, and
`X-CLM-XDATA`. Each has a population, target, estimand, PASS, NO-GO,
INCONCLUSIVE, invalidity, and permitted wording. A PASS on one evidence line
cannot replace another. Planned scope is never reported as a result.

## 3. Target and unmodified computational graph

For clean normalized input `x`, let `F(x)` be the unmodified evaluation-mode
model and `y*=argmax_k F_k(x)`. The primary scalar is the raw logit
`f(x)=F_y*(x)`. The true class, predicted class, runner-up class, all three raw
logits, and clean margin are retained. Predicted-, true-, and runner-up-target
analyses have separate rows and denominators. The clean `y*` is fixed for all
perturbations of that image.

The primary graph includes patch projection and bias; class token and position
embedding; every Q, K, V projection and bias; scaled dot product; softmax;
attention dropout in frozen evaluation semantics; head concatenation; output
projection and bias; both residual additions; both LayerNorms; both MLP
linears, activation, biases, and dropout; drop-path in frozen evaluation
semantics; final norm; and classifier head and bias. Parameters are not
modified. Any detached attention, surrogate operator, frozen attention, or
replacement forward is a separately named condition and is never the NNS-v2
primary.

## 4. NNS-v2 mathematical definition

### 4.1 Input IG

The path baseline is the normalized all-zero tensor `x0`, corresponding to
canonical RGB `(0.5,0.5,0.5)` under the frozen preprocessing. The straight
path is `x(alpha)=x0+alpha(x-x0)`. For input coordinate `i`,

```text
A_i = (x_i-x0_i) integral_0^1 partial f(x(alpha))/partial x_i d alpha.
```

The primary scientific array is signed float32 `[3,224,224]`; integration and
error accumulation use float64. The signed RGB map is
`M[r,c]=sum_channel A[channel,r,c]`. No resize, interpolation, patch
replication, pooling, smoothing, threshold, clipping, normalization, or
colormap is permitted in a scientific map. Display arrays use a separate type,
schema, API, and output root.

Input completeness is

```text
e_input = sum_i A_i - (f(x)-f(x0)).
```

It passes when `abs(e_input) <= 1e-4 + 0.005*abs(f(x)-f(x0))`.

### 4.2 Complete-cut path-Jacobian decomposition

For a registered computational cut `C`, let `h_C=(h_1,...,h_m)` contain each
cut component exactly once and intercept every differentiable input-to-target
path. Define the pixel-to-component allocation

```text
K^C[i,j] = (x_i-x0_i) integral_0^1
            (partial f/partial h_j)(x(alpha))
            (partial h_j/partial x_i)(x(alpha)) d alpha.
```

The chain rule requires `sum_j K^C[i,j]=A_i`. Component `j` has pixel map
`K^C[:,j]`; its scalar conductance is `sum_i K^C[i,j]`. The cut residuals are

```text
R_pixel(C) = max_i abs(sum_j K^C[i,j]-A_i)
R_total(C) = abs(sum_i,j K^C[i,j]-(f(x)-f(x0))).
```

Each cut passes only when
`R_pixel(C) <= 2e-5 + 0.005*max_i(abs(A_i))` and
`R_total(C) <= 2e-4 + 0.01*abs(f(x)-f(x0))`. X1 must prove these identities
on analytic synthetic networks and native-forward parity fixtures. Failure at
any required cut is `NO_GO_CLASS_LOGIT_COMPLETENESS`; input IG alone may still
be reported as full-model IG but cannot be named NNS-v2.

### 4.3 Registered cuts and architectural accounting

Required cuts are complete edge cuts, not isolated tensors. They are: the
post-embedding token sequence; each block input;
`{residual-identity edge,norm1 output}`;
`{residual-identity edge,all Q,K,V coordinates}`;
`{residual-identity edge,all post-softmax attention coordinates,all V coordinates}`;
`{residual-identity edge,all per-head value-mixed outputs before concat}`;
`{residual-identity edge,attention-projection output}`; the first residual-add
output; `{post-attention residual edge,norm2 output}`;
`{post-attention residual edge,all MLP hidden units}`;
`{post-attention residual edge,MLP output}`; the second residual-add output;
final norm; and classifier input coordinates. Every brace-delimited tuple
intercepts the bypass and transformed paths together; validating Q/K/V or an
attention tensor alone is forbidden because it is not a complete cut.

For attention head `h`, its component is the native head slice before concat
carried through the matching columns of the native output projection. Head maps
must sum to the attention-branch map. No learned or visual head weighting is
allowed. At the first residual add, identity and attention components sum to
the add output; at the second, residual and MLP components sum likewise. These
are separate complete cuts. Layer views are alternative decompositions of the
same total attribution and must never be summed across layers.

Softmax, LayerNorm, GELU, and multiplicative QK/attention-value interactions
remain on the original autograd path in both Jacobian factors. No custom
propagation rule is permitted in v1. Model biases and constant class/position
embeddings remain active and influence all downstream Jacobians; because the
integration varies only the input, their direct path derivative at the point of
introduction is zero. This zero direct term is recorded rather than treating
the bias or embedding as absent.

### 4.4 Quadrature and convergence

Gauss-Legendre schedules are `[128,256,512,1024,2048,4096,8192]`, evaluated
from small to large with alpha chunk 4 and component chunks selected only by a
deterministic memory planner. The first step count passing input completeness,
all required cut residuals, endpoint parity `<1e-6`, and repeatability is
selected; every attempt is retained. Failure at 8192 is typed scientific
invalidity, not an implementation retry. Dropout and drop-path are disabled by
evaluation mode and their exact module states are recorded.

## 5. Comparator families

The systematic search and decisions are frozen in
`manifests/stage_x_literature_registry_v1.json` and
`manifests/stage_x_comparator_registry_v1.json`. The primary class-targeted
family contains NNS-v2 plus ViT Grad-CAM, full-model IG, Transformer
Attribution (Chefer et al.), AttnLRP, RISE, and Occlusion. Every
external adapter is pinned to the recorded official implementation commit and
must pass X1 target, forward, output, and model compatibility tests. Failure
does not silently remove the method; it stops the primary benchmark unless a
prospective pre-outcome amendment changes the registered scope.

Secondary references are attention rollout, raw CLS attention, random map,
center prior, and edge/input-gradient prior. They never enter the primary
class-targeted family. ViT-CX, LeGrad, Transformer Input Sampling, TokenTM, and
LibraGrad are `DEFERRED` because X0 did not verify, respectively, raw-logit
target parity (the official ViT-CX code scores post-softmax probability), a
frozen-timm classifier adapter, an explicit repository license, a same-paper
official implementation, or an explicit repository license. They cannot enter
formal statistics after outcome access.

## 6. Model and dataset matrix

Required model IDs are `vit_small_patch16_224.augreg_in21k_ft_in1k`, the sealed
recovery-v2 `vit_base_patch16_224.augreg2_in21k_ft_in1k`,
`deit_small_patch16_224.fb_in1k`, `deit_base_patch16_224.fb_in1k`, and
`vit_base_patch32_224.augreg_in21k_ft_in1k`. Only the recovery-v2 ViT-B/16 is
materialized and bound in X0. Every other checkpoint is
`PLANNED_NOT_MATERIALIZED`; X0 neither opens nor hashes checkpoint bytes.
Swin-T is deferred unless X1 proves shifted-window and patch-merging cuts.

Datasets are ImageNet-1k, ImageNet-S919, ImageNet-V2 matched-frequency, and
Oxford-IIIT Pet. Oxford Pet is the independent task because its official
release supplies 37 breed labels, head ROI, and pixel trimaps under CC BY-SA
4.0. Its task checkpoints require a separately sealed training protocol and
may not be chosen using explanation outcomes. ImageNet-S is localization data,
not independent classification replication. The full required matrix is five
models by four datasets; every cell is reported even when failed or NA.

Formal planned counts are: ImageNet-1k 1,000 (one per class), ImageNet-S919
919 (one per registered class), ImageNet-V2 matched-frequency 1,000 (one per
class), and Oxford Pet 740 (20 per breed from the official test split).
Insufficient eligible material blocks the cell; sample counts are not reduced.
Localization uses ImageNet-S and Oxford trimaps only.

## 7. Sample isolation

X1 uses synthetic fixtures and only the existing Stage C audit IDs and F3H
order-44 continuity image. X2 uses exactly 40 development images per required
model, selected from the already accessed Stage D registry; they never enter
X3-X5 statistics. All C10 images, the entire Stage D 500-image registry, all
F3/F3H display slots, the F3C pilot, and F3C formal samples are in the
prior-access exclusion ledger.

At authorized materialization, each candidate receives
`sha256(namespace + NUL + dataset_release_id + NUL + class_id + NUL +
source_object_id + NUL + file_sha256)`. Selection is class-stratified by this
digest, then source object ID. Exact hashes are sealed once, before image
decode, prediction, attribution, localization, stability, or method-difference
access. Selection reads only release metadata, class, object ID, bytes, and
file hash. Namespace, exclusions, counts, and materialization gates are in the
sample split plan.

## 8. Six independent evaluation lines

### 8.1 Faithfulness

The primary image statistic is deterministic ROAD Combined:
`0.5*(LeRF-MoRF)`, averaged over removal percentiles
`[10,20,30,40,50,60,70,80,90]` and five registered noisy-linear-imputation
seeds. Targets are normalized clean predicted-class raw-logit responses, and
higher is better. ROAD implementation semantics are pinned before X1; random
noise is generated by per-image PCG64 seed and cannot use global RNG state.
Insertion AUC, deletion AUC, sufficiency, comprehensiveness, and ROAD MoRF/LeRF
are secondary. The fraction grid is `0,.05,...,1`; fractions use exact selected
pixel counts and trapezoidal AUC. Score ties use SHA-256-derived PCG64
permutations shared by methods. Sensitivity corruptions are normalized zero,
channel mean, Gaussian blur (sigma 10 input pixels), and ROAD noisy linear
imputation. The attribution path baseline and perturbation corruption are
separate registered objects.

### 8.2 Spatial stability/equivariance

Geometric transformations are horizontal and vertical translations `+/-4` and
`+/-8` pixels with reflection padding; 216-to-224 center crop/resize; horizontal
flip; and rotations `+/-10` degrees. Photometric transformations are Gaussian
noise sigma `0.01` with five seeds, brightness factors `0.9/1.1`, contrast
factors `0.9/1.1`, and JPEG qualities `70/90`. Geometry is inverted to clean
coordinates and evaluated only on the common valid domain. Signed Spearman,
top-10%-mass Jaccard, signed SSIM, positive/negative attribution-mass
displacement normalized by image diagonal, and center-of-mass pixel error are
reported.

For each registered transformation, absolute PASS requires class-macro median
Spearman `>=0.50`, Jaccard `>=0.30`, SSIM `>=0.50`, normalized mass displacement
`<=0.10`, and center-of-mass error `<=8` pixels. Benchmark spatial PASS also
requires at least 80% required cells to pass and NNS-v2 to be noninferior to
the fixed reference ViT Grad-CAM under the X2-sealed margins. Report all-sample
fixed-target, prediction-unchanged subset, and prediction flip rate. Any
transformation averaging may use only a separately listed training set; all
transformations above remain holdout tests.

### 8.3 Pixel resolution and class specificity

The `[3,224,224]` provenance and signed RGB reduction are required. Synthetic
fixtures must distinguish positions within one patch. Formal intervention
units are 1x1, 2x2, 4x4, 8x8, and 16x16 without map interpolation. Patch/cell
resize masquerading as pixel attribution is invalid. Tests include predicted
versus runner-up, true versus predicted, target swap, positive-support
deletion, negative-contribution deletion, and classifier-to-input progressive
parameter randomization. A label-randomization model is not part of v1.

### 8.4 Object localization

Primary localization is class-macro pixel AUPRC on positive class support.
Secondary metrics are energy inside mask, pointing game, and top-20%-mass IoU.
Bboxes are reported separately and never pooled with masks. The primary
population is registered samples whose target matches the annotation class and
whose prediction is correct; full-manifest results, correct coverage, per-class
coverage, and all-sample secondary results are also retained. Absence or drift
of a real mask/bbox makes localization NA and blocks a localization claim.

### 8.5 Cross-model and cross-dataset estimand

The hierarchy is image, class, model-dataset cell, benchmark. For method `m`
and cell `c`, the image difference is `d_i=ROAD_NNSv2-ROAD_m`. The cell
estimand is the equal-weight mean of within-class image means. The benchmark
estimand is the equal-weight mean of required cell estimands; images are never
pooled across cells. Each cell reports effect, 95% CI, expected/valid/failed/NA,
class and prediction coverage, wall/GPU/CPU time, peak memory, download bytes,
and retained bytes.

## 9. Statistics, margins, and decisions

Cell CIs use 10,000 class-stratified paired bootstrap draws. Benchmark CIs use
20,000 hierarchical draws: cells, then classes within cells, then images within
classes; required cells keep equal weight. RNG is NumPy PCG64, seeds are the
first eight big-endian bytes of SHA-256 over the frozen namespace and analysis
ID, and percentile quantiles use `linear`. Primary family is NNS-v2 against the
seven registered class-targeted comparators. One-sided superiority tests use
Holm at family alpha 0.05; secondary families use separate Holm correction.
No nonsignificant result is equivalence. Required-primary failure/NA blocks the
effect claim; no complete-case substitution, zero imputation, or replacement is
allowed.

Exact noninferiority and futility margins are sealed once at X2 before any X3
outcome. For each endpoint, X2 computes robust scale
`s=MAD(d_i)/0.6744897501960817`. Faithfulness noninferiority margin is
`clip(0.20*s,0.01,0.05)`; the negative futility magnitude is
`clip(0.30*s,0.02,0.075)`. Stability difference margins use the same 20% rule
with endpoint ranges: correlations/Jaccard/SSIM `[0.02,0.10]`, normalized
displacement `[0.01,0.05]`, and COM error `[1,4]` pixels. One sealed margin file
is signed before X3 access; any later change invalidates formal analysis. X2
futility fires when the 95% upper confidence limit for NNS-v2 minus Grad-CAM
ROAD is below the negative futility magnitude.

`GO_BROAD_BENCHMARK_SCOPE` requires all method/completeness/pixel/sanity gates;
for every primary comparator a benchmark 95% CI lower above zero and Holm
rejection; at least 80% of required cells with positive point estimates; every
required cell 95% CI lower above its negative noninferiority margin; localization
versus Grad-CAM benchmark CI lower above zero; spatial absolute PASS and
noninferiority to fixed Grad-CAM; and complete terminal/resource accounting.
This conjunction is intentionally strict because the requested broad wording
joins distinct scientific claims. Failure yields the claim-specific `NO_GO_*`;
valid but underpowered/mixed results yield `INCONCLUSIVE_*`. Scientific
invalidity has priority over effect decisions. Granular claim PASS states remain
reportable even when the broad conjunction fails.

## 10. Resources, retention, and stop rules

X1 hard caps are 16 H20-GPU-hours, 48 wall-hours, 128 CPU-hours, 48 GiB peak
reserved memory per process, zero download bytes, and 32 GiB retained files.
X2 hard caps are 160 H20-GPU-hours, 72 wall-hours, 400 CPU-hours, 48 GiB peak
reserved per process, eight devices, 40 GiB downloads, and 128 GiB retained.
No automatic waiver exists. X2 profiles every method-model-image route and
seals p50/p95/p99 time, memory, forward/backward calls, and bytes.

Before X3, exact formal caps are computed deterministically as
`1.5 * sum(required unit count * X2 p99 unit cost)` plus measured fixed control
overhead, rounded upward to 0.1 GPU-hour, 1 CPU-hour, 1 wall-hour, 1 GiB memory
and storage, and capped by owner ceilings X3=300, X4=900, X5=700 H20-GPU-hours.
If the formula exceeds a ceiling, formal execution is BLOCKED; samples,
comparators, perturbations, targets, or repeats are not reduced. Download and
retention caps use the same p99 formula and frozen dataset release sizes.

Budget exceedance is `BLOCKED_BUDGET`; OOM is typed infrastructure failure with
at most one pre-authorized immutable retry; numerical nonconvergence is
scientific `FAILED`; external outage is `BLOCKED_INFRASTRUCTURE`. None are
converted to NA or silently retried. Completeness, class sensitivity, pixel
provenance, or basic stability failure at X2, or the negative Grad-CAM futility
boundary, publishes immutable `NO_GO_BROAD_CLAIM_DEVELOPMENT`, stops X3-X5,
and returns to scoped F3-Final then Stage G.

Retention levels are L0 scalar terminals and resources for every unit; L1
native scientific maps and primary curves; L2 full component decomposition only
for a preregistered audit subset; L3 display products. L3 never enters metrics.

## 11. Stage gates

X0 PASS authorizes only `X1_NNS_V2_IMPLEMENTATION_ONLY`. X1 uses synthetic and
prior-access audit images. X2 is a development and resource pilot. X3 is
single-model formal; X4 cross-model; X5 cross-dataset; X6 CPU-only statistics
and decisions; X7 reconnects actual passed claims to F3-Final. X0 runs no X1
code, NNS-v2, inference, attribution, download, model load, GPU work, scientific
result, or paper figure. Every stage uses immutable typed terminals,
source-recomputing validation, false-pass audit, and pending-to-terminal atomic
completion.
