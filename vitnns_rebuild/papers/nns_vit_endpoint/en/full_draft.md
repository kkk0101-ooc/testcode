# NNS-ViT Endpoint Scanner: Registered Internal-Target Scanning and Structured Decomposition of Vision Transformers

- Authors: `[AUTHOR REQUIRED]`
- Affiliations: `[AFFILIATION REQUIRED]`
- Corresponding author: `[CORRESPONDING AUTHOR REQUIRED]`

> Draft R1, September 18, 2026. Figure numbers follow the sealed Endpoint package (Figures 1–28).
> They may be renumbered for submission according to
> `papers/nns_vit_endpoint/zh/outline.md` without changing figure content, numerical
> values, or conclusions.

## Abstract

Internal explanations of Vision Transformers (ViTs) span tokens, channels, attention heads,
routing and value paths, MLP branches, and residual streams. Their different tensor shapes and
computational roles make it difficult to scan, structurally reconcile, and behaviorally evaluate
them in a common coordinate system. We present the NNS-ViT Endpoint Scanner, which adapts the
input-value initialization and current-state propagation principle of the Neural Network Scanner
to ViTs. For any registered scalar target, it produces
$L_s^{\mathrm{Endpoint}}(x)=x\odot\nabla_xs(x)$ and decomposes the same final target through
complete computational-edge cuts into head, Q/K/V, routing/value, attention-residual, and
MLP-residual components. The base quantity is Gradient × Input; the subject of this study is the
registration of ViT targets, structural accounting, and the empirical utility of the resulting
components under finite interventions. We evaluated ViT-B/16 and DeiT-S/16 on a shared cohort of
400 ImageNet evaluation images. Registered scanning completed 800/800 image-model tasks and
336,800 finite target evaluations. Structural tests passed 76,800 positive closure checks,
rejected 28,800 omission or double-counting controls, and passed 6,144 condition-scaled analytic
JVP checks. Under matched activation replacement, 11 of 12 preregistered families met the
correlation, direction, precision, and multiplicity criteria. The sole exception had a median
Spearman correlation of 0.099565, which was 0.000435 below the prespecified threshold of 0.10.
We therefore interpret RQ3 as supported overall with one registered boundary exception, while
retaining `NOT_SUPPORTED` for the frozen 12-of-12 omnibus field. In spatially targeted
interventions, none of four families simultaneously outperformed total G×I, raw gradient,
regional Taylor, and random regions; regional Taylor attained a higher median effect in every
family. Endpoint components can thus predict component-level finite effects, but that property
does not automatically make their pixel distributions better local-region selectors.

Keywords: Vision Transformer; neural-network interpretation; Gradient × Input; structural
decomposition; activation replacement; attention head

## 1. Introduction

Vision Transformers organize image representations as patch tokens and repeatedly transform them
through multi-head self-attention, MLPs, and residual streams (Dosovitskiy et al., 2021). This
architecture creates many potential objects of interpretation: a token-channel unit, an attention
head, Q/K routing, Value content, an MLP branch, or a final class logit. These objects occupy
different tensor spaces. Direct inspection of activations or attention weights can describe an
internal state, but it does not by itself show how that state depends on the original input, nor
does it ensure that multiple structural components account for a common target without omission
or double counting.

Existing ViT interpretation methods address different parts of this problem. Attention rollout
recursively aggregates attention flow between tokens (Abnar and Zuidema, 2020). Transformer
relevance methods propagate class-related quantities through a computational graph (Chefer et
al., 2021; Achtibat et al., 2024). Input-attribution methods map a final output to pixel
coordinates. These methods differ in their targets, propagation rules, and evaluation criteria.
Interpretation of internal structure also requires three questions to remain separate: what is the
mathematical identity of a quantity, whether the implementation realizes that identity, and
whether the quantity predicts a finite intervention. Gradient agreement or structural closure
cannot replace intervention tests, and a spatially detailed map need not be a better region
selector than an estimate constructed directly for a candidate intervention.

The Neural Network Scanner (NNS) of Dou et al. (2023) associates a position-preserving learning
image with an internal neuron in an FCN or CNN. When affine layers, convolutions, active ReLU
gates, and unique MaxPool winners use their current local propagation coefficients, the
single-neuron rule can be written as the product of the input and the gradient at the current
point. ViTs introduce LayerNorm, GELU, input-dependent softmax attention, two residual additions,
and multiple token- and branch-level objects. Their treatment requires explicit Jacobians,
structural cuts, and target identities. We retain the endpoint-propagation view of NNS but do not
present $x\odot\nabla_xs(x)$ as a new base attribution formula.

We develop the NNS-ViT Endpoint Scanner and evaluate four research questions (Figure 1):

1. **RQ1: registered scanning.** Can a common registry of internal scalars and output targets be
   scanned completely, finitely, and auditably in two frozen ViTs?
2. **RQ2: structural correctness.** Can head, Q/K/V, routing/value, and residual cuts reconstruct
   their registered objects while rejecting omissions and double counting?
3. **RQ3: component-level utility.** Does the input-space mass of an Endpoint component predict
   the finite logit effect of replacing that component with a reference activation?
4. **RQ4: spatial utility.** Under a fixed region pool and budget, does a full-resolution
   component map localize finite effects better than total G×I, raw gradient, regional Taylor,
   and random regions?

This work makes five contributions. First, it formalizes the NNS endpoint principle through the
actual ViT Jacobian and specifies the roles of the normalized-input origin, LayerNorm, GELU,
softmax attention, MLPs, and residual connections. Second, it constructs registered target
catalogs for two ViTs and completes 336,800 target evaluations on a shared 400-image cohort.
Third, it defines complete edge cuts that map heads, attention differential paths, and both
residual branches into a common input coordinate system and reconstruct their registered objects.
Fourth, it evaluates correctness and utility separately using closure controls, analytic JVPs,
matched activation replacement, and spatially targeted intervention. Fifth, it reports two
complementary empirical outcomes: component-level finite-effect prediction is supported overall
with one boundary exception, whereas full-resolution spatial ranking does not outperform a strong
local Taylor baseline.

## 2. Related Work

### 2.1 Neuron visualization and NNS

Internal visualization methods either search for inputs that maximize a unit or begin with an
observed sample and trace its internal response. Feature visualization, network dissection,
deconvolutional views, and class activation maps characterize preferred inputs, semantic concepts,
local responses, or class-related regions. NNS defines a different object: a position-preserving
learning image associated with a neuron or aggregate response in an FCN or CNN (Dou et al., 2023).
We retain this internal-target-to-input-coordinate organization and extend current-state
propagation to actual ViT operators. Captum NeuronGradient and InputXGradient already provide
clear precedents for internal scalar selectors and input-times-gradient. The technical work here
therefore concerns registry scope, structural cuts, closure validation, and finite-intervention
auditing rather than hooks, selectors, or the base G×I formula.

### 2.2 Input attribution, path integration, and conductance

Integrated Gradients (IG) integrates gradients along a path from a reference to the observed input
and satisfies axioms including sensitivity and implementation invariance (Sundararajan et al.,
2017). Neuron conductance further measures how output attribution flows through a hidden unit
(Dhamdhere et al., 2018). The distinction between these methods and Endpoint is an attribution
definition rather than a CNN-versus-Transformer distinction: Endpoint uses the Jacobian at the
observed input and an input-value seed, whereas IG uses a reference-difference seed and a
path-averaged Jacobian. We preserve that boundary. Because ViT-B/16 did not satisfy the
preregistered Path-IG calibration gate within the maximum 512-point numerical range, we do not
report cross-model Endpoint-versus-Path fidelity, stability, or cost-quality conclusions.

### 2.3 Transformer interpretation and structural analysis

Raw attention describes token relations but lacks a complete downstream class derivative.
Attention rollout aggregates attention and residual routes across layers (Abnar and Zuidema,
2020). Chefer et al. (2021) and AttnLRP (Achtibat et al., 2024) apply relevance rules to
Transformer graphs. LibraGrad modifies backward paths to reduce gradient-flow imbalance in
Transformers (Mehri et al., 2025). These studies establish that attention, skip connections, and
gradient paths all matter for interpretation. Endpoint Scanner retains the model's actual local
Jacobian and treats raw attention, relevance propagation, and modified-backward attribution as
distinct methods.

### 2.4 Activation patching and evaluation of explanations

Activation patching measures an output change by replacing an internal activation. Attribution
Patching approximates a finite effect with the inner product of an activation displacement and a
clean gradient; AtP* further analyzes failures caused by attention saturation and cancellation
(Kramár et al., 2024). This distinction is central here: a local derivative identity establishes
correctness, whereas finite replacement evaluates behavioral prediction. We study both
whole-component ranking and region ranking under a fixed spatial budget, using activation-space
regional Taylor as a mandatory baseline for the latter.

## 3. Method

### 3.1 Mathematical identity of an Endpoint learning image

Let $x\in\mathbb R^d$ be the input tensor actually received by the model. For a computational
node $u(x)\in\mathbb R^m$, define a learning-image matrix
$\mathcal L(u;x)\in\mathbb R^{m\times d}$. Input coordinates are initialized as

$$
\mathcal L(x;x)=\operatorname{diag}(x).
$$

For $v=f(u_1,\ldots,u_r)$, with every Jacobian evaluated at the current forward state,

$$
\mathcal L(v;x)=\sum_{a=1}^{r}J_{f,u_a}(x)\mathcal L(u_a;x).
$$

Recursive application of the chain rule to a scalar target $s(x)$ gives

$$
\boxed{L_s^{\mathrm{Endpoint}}(x)=x\odot\nabla_xs(x)}.
$$

Figure 2 separates the seeds and Jacobians used by the original NNS-CNN rule, ViT Endpoint, and
Path IG. Endpoint has no reference forward and does not generally satisfy baseline-difference
completeness. The coordinate system is the normalized RGB tensor received by the model. If
$x=(r-\mu)/\sigma$, then

$$
x_i\frac{\partial s}{\partial x_i}
=(r_i-\mu_i)\frac{\partial s}{\partial r_i}.
$$

Canonical Endpoint thus uses the model normalization mean as its origin when written in raw-RGB
coordinates. Two-dimensional displays sum the original signed RGB tensor over channels. Absolute
values, positive parts, and per-panel display scaling are not used for statistical inference.

### 3.2 Registered internal targets

The scanner accepts any registered scalar target. The catalog includes token-channel units,
patch-channel sums, patch-channel energies excluding CLS, internal attention-head coordinates, a
final CLS channel, the ground-truth raw logit, and the frozen clean-top1 raw logit. The energy
target is

$$
s_{\mathrm{energy}}(x)=\frac12\sum_{t=1}^{196}h_{t,c}(x)^2,
$$

which corresponds to the activation-weighted filter aggregation in the original NNS. ViT-B/16
has 457 registered targets per image and DeiT-S/16 has 385, both spanning all 12 blocks
(Figures 7 and 11). Equal channel indices across models are not interpreted as semantic matches.

### 3.3 ViT operators and attention differential paths

The canonical backend uses the standard unfused Block/Attention implementation in timm 0.9.16.
Patch embedding, LayerNorm, GELU, Q/K/V projections, row-wise softmax, multi-head concatenation,
output projection, MLPs, both residual additions, final LayerNorm, and the classifier all use
their actual current-state Jacobians. The fused backend is used only for forward parity because it
does not expose the registered structural edges.

For one attention head, let

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

$$
A=\operatorname{softmax}(QK^\top/\sqrt{d_h}),\qquad O=AV.
$$

Its differential is

$$
\mathrm dO=(\mathrm dA)V+A(\mathrm dV),
$$

where $\mathrm dA$ is further composed of Q and K paths. This yields the registered Q+K+V and
routing+Value decompositions (Figure 5). LayerNorm retains its mean-variance coupling, and GELU
retains its smooth derivative; neither is replaced with frozen statistics or a binary ReLU gate.

### 3.4 Structured endpoint decomposition

For a final scalar $F$ and a cut component $z_k$, define the input-coordinate component map

$$
E^C_{k,i}=x_i\left\langle\nabla_{z_k}F(x),
\frac{\partial z_k(x)}{\partial x_i}\right\rangle.
$$

If the cut contains every corresponding path from input to target without duplication, then

$$
\sum_kE^C_{k,i}=x_i\frac{\partial F(x)}{\partial x_i}.
$$

We register six types of cuts: all heads reconstructing the attention branch; identity plus heads
reconstructing the first residual total; Q+K+V or routing+Value reconstructing the attention
branch; identity plus either differential decomposition reconstructing the residual total; and
carry plus the MLP branch reconstructing the second residual total (Figures 10 and 15). A residual
identity follows its actual input edge. Omitting a nonzero branch or adding a bypass twice must be
rejected by closure validation.

### 3.5 Local correctness and finite activation replacement

For each component map, define its signed sum and absolute mass as

$$
a_k=\sum_iE^C_{k,i},\qquad m_k=\sum_i|E^C_{k,i}|.
$$

Let $v_k=J_{z_k,x}x$. The analytic local contract is

$$
a_k=\langle\nabla_{z_k}F,v_k\rangle.
$$

It checks the consistency of the hook, cotangent, and constructed direction. The finite effect of a
same-image reference-activation replacement is

$$
e_k=F(x)-F(x;z_k\leftarrow z_k^{\mathrm{ref}}).
$$

We use normalized-zero and RGB-blur references. The primary RQ3 quantity is the within-image
$\operatorname{Spearman}(m_k,|e_k|)$, with separate families for heads, routing/value, and residual
branches. The activation-displacement Taylor predictor

$$
p_k^{\mathrm{ref}}=\langle\nabla_{z_k}F,z_k(x)-z_k^{\mathrm{ref}}\rangle
$$

is reported as a preregistered secondary predictor but is not part of the C4a support gate
(Figures 19 and 22).

### 3.6 Spatially targeted intervention

RQ4 fixes head 0, the middle head, and the last head in block 5. The 224×224 input is partitioned
into a 4×4 grid of non-overlapping 56×56 regions. Rankings select the top 1, 2, or 4 regions,
corresponding to 6.25%, 12.5%, and 25% of the image. Endpoint scores a region by RGB-level
absolute component mass. The four mandatory baselines are total Endpoint/G×I, raw input
gradient, regional activation-space Taylor, and 20 frozen random-region permutations (Figure 23).

For a candidate-region union $R$ in a single head, the finite effect is

$$
F(x)-F\bigl(z_h(x_R),z_{-h}(x),\mathrm{identity}(x)\bigr),
$$

where $x_R$ applies the corresponding reference replacement. Effects are first averaged over the
three heads within an image. We then compute the trapezoidal area under the absolute isolated-effect
curve from 0% to 25% area and divide by 0.25. All methods share the candidate pool, spatial
budget, and finite-effect definition.

## 4. Experimental Design

### 4.1 Models, data, and preprocessing

We use `vit_base_patch16_224.augreg2_in21k_ft_in1k` and
`deit_small_patch16_224.fb_in1k`. Both have 12 blocks; ViT-B/16 uses 12 heads and a
768-dimensional representation, whereas DeiT-S/16 uses 6 heads and 384 dimensions. Images come
from ImageNet validation. Frozen metadata rules define disjoint discovery, calibration, and
evaluation sets of 64, 64, and 400 images, respectively; both models use all 400 evaluation IDs.
Images are resized to a short side of 248 using bicubic antialiasing and center-cropped to 224.
Each model uses its frozen mean and standard deviation.

Production used Torch 2.2.2+cu121, timm 0.9.16, torchvision 0.17.2+cu121, NumPy 1.26.4, and
Pillow 10.2.0. Models ran in evaluation mode with dropout and drop path disabled, and canonical
structural experiments used unfused attention. SHA-256 digests bind model files, image members,
target registries, task payloads, and result files.

**Table 1. Models, preprocessing, and registered targets.**

| Item | ViT-B/16 | DeiT-S/16 |
|---|---:|---:|
| timm model | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | `deit_small_patch16_224.fb_in1k` |
| Blocks / heads / hidden dim | 12 / 12 / 768 | 12 / 6 / 384 |
| Registered targets/image | 457 | 385 |
| Evaluation images | 400 | 400 (same IDs) |
| Resize / crop | short side 248 / center crop 224 | short side 248 / center crop 224 |
| Normalization mean | [0.5, 0.5, 0.5] | [0.485, 0.456, 0.406] |
| Normalization std | [0.5, 0.5, 0.5] | [0.229, 0.224, 0.225] |

### 4.2 Confirmatory analysis

Before evaluation outcomes were read, the analysis plan froze 16 families and 28 test IDs. C4a
contains 12 families from 2 models × 2 references × 3 component families. Its primary statistic is
the median of 400 image-level Spearman correlations, with a practical-effect threshold of 0.10.
C4b contains four model-reference families. Each must attain a paired image-level median
improvement of at least 0.05 over each of four mandatory baselines.

For each family, 10,000 paired image-cluster percentile bootstrap replicates produce a 95%
interval. Each test uses a two-sided exact sign test, and all 28 test IDs enter one Holm correction
at $\alpha=0.05$. A family requires at least 360 jointly valid images and a sign test requires at
least 300 nonzero image differences. Maximum interval widths are 0.30 for C4a and 0.50 for C4b.
The statistical unit is always the image; heads, blocks, references, and random permutations are
not treated as independent observations.

### 4.3 Validation and reproducibility

Each stage uses immutable result roots, exact-coverage audits, and independent validators.
Structural closure includes positive cases that must pass and omission or duplication controls that
must fail. The statistical validator imports neither Torch, timm, nor PIL; it recomputes per-image
quantities, bootstrap intervals, sign tests, Holm adjustments, and family decisions from sealed
task records. A CPU-only renderer creates the paper figure package from sealed numerical records
and display arrays. Display-normalized images are not inputs to statistical inference.

## 5. Results

### 5.1 RQ1: Registered target scanning was complete and tractable

All 800 image-model tasks finished uniquely, with no missing, duplicate, invalid, or formally
retried task. The 457 targets per ViT-B/16 image and 385 targets per DeiT-S/16 image yielded
336,800/336,800 finite target evaluations. Ground-truth and clean-top1 raw signed RGB class maps
were retained for every image. Full raw atlases were stored for the first 16 evaluation IDs in
both models, giving 32 atlases; neither the near-zero index nor the failure index contained an
entry. Figure 12 shows the frozen first eight images in both models, and Figure 13 shows a
multi-target atlas.

Formal wall time was 1,752.71 s, and the conservative four-card upper bound was 1.9475 GPU-hours.
Peak reserved GPU memory was 2.284 GB, peak CPU resident memory was 2.018 GB, and retained data
occupied 9.290 GB. Formal throughput was 192.16 targets/s and 0.4564 images/s. Median/P95/maximum
full-atlas latency was 7.853/8.015/8.240 s for DeiT-S/16 and 9.485/9.751/10.615 s for ViT-B/16
(Figure 14). These measurements support RQ1: the registered catalog was completed on the fixed
models and cohort and produced an auditable atlas.

**Table 2. RQ1 registered-scan coverage and resources.**

| Metric | Result |
|---|---:|
| Image-model tasks | 800/800 |
| Finite target evaluations | 336,800/336,800 |
| Formal wall time | 1,752.71 s |
| Four-card upper-bound GPU-hours | 1.9475 |
| Formal throughput | 192.16 targets/s |
| Peak reserved GPU memory | 2.284 GB |
| Peak CPU RSS | 2.018 GB |
| Retained data | 9.290 GB |

### 5.2 RQ2: Structural cuts satisfied positive and negative contracts

The formal closure experiment covered 400 images, two models, all 12 blocks, and eight registered
cuts per block, producing 76,800 positive checks. Every check satisfied the float32 contract
`atol=1e-5, rtol=1e-4`. The maximum relative L2 error was
$1.4144569831842091\times10^{-5}$ and the maximum absolute error was
$2.47955322265625\times10^{-5}$. The latter does not violate the contract because the elementwise
criterion is $|\mathrm{error}|\leq\mathrm{atol}+\mathrm{rtol}|\mathrm{reference}|$.

Three checker controls per block deliberately omitted a nonzero head or routing term or duplicated
a residual contribution. All 28,800 controls were rejected as intended (Figure 16). A CPU-only
validator independently reconstructed 384 raw block arrays from the first 16 evaluation IDs.
RQ2 is therefore supported for the registered cuts in these two standard timm ViTs: components
reconstruct their attention-branch or residual-total object, and the checker detects common
omission and double-counting errors.

### 5.3 Local differential implementation passed directional checks

The analytic local contract further tested whether the signed component sum equaled a JVP-derived
directional quantity at the same hook and cotangent. All 6,144 condition-scaled analytic JVP
checks on the first 16 evaluation orders of both models passed (Figure 18). The maximum absolute
error was $2.648076753197781\times10^{-5}$, and the maximum
`error/sum_abs_products` was $6.948755071487438\times10^{-6}$, within the frozen boundary
$10^{-5}+10^{-5}\sum_i|g_i(Jx)_i|$. This result verifies the local differential implementation;
it does not substitute for the finite replacements in the next section.

**Table 3. RQ2 structural-closure and local-differential checks.**

| Check | Count | Result | Maximum error |
|---|---:|---|---:|
| Registered positive closures | 76,800 | all passed | relative L2 = 1.41446e-05 |
| Omission/double-counting controls | 28,800 | all rejected as intended | not applicable |
| Condition-scaled analytic JVP | 6,144 | all passed | absolute = 2.64808e-05 |
| JVP condition-scaled ratio | 6,144 | all passed | 6.94876e-06 |

### 5.4 RQ3: Structural components predicted matched finite effects overall

RQ3 compares Endpoint component mass $m_k$ within an image with the absolute finite replacement
effect $|e_k|$ of the same component. Eleven of the twelve preregistered families satisfied all
conditions on point estimate, positive direction, Holm significance, coverage, and interval
precision (Figure 20). Median family-level Spearman correlations ranged from 0.099565 to 0.306320.
For ViT-B/16, head-family estimates were 0.290853 under normalized zero and 0.306320 under blur;
for DeiT-S/16 they were 0.190961 and 0.181668. The four residual-branch estimates were 0.177391,
0.209130, 0.135217, and 0.228696.

The only family that did not meet all conditions was ViT-B/16 × normalized zero × routing/value.
Its estimate was 0.09956521739130435, with a 95% interval of [0.07478260869565218,
0.12260869565217392] and a Holm-adjusted $p$ value of
$3.0036026037128944\times10^{-14}$. The direction was positive and survived multiplicity
correction, but the point estimate was 0.000435 below the prespecified practical-effect threshold
of 0.10 (Figure 21). We therefore interpret RQ3 as **supported overall with one registered
boundary exception**. The frozen family status and 12-of-12 omnibus C4a field remain
`NOT_SUPPORTED`: the threshold was not moved, and 0.099565 was not rounded into a pass.

This two-level report answers distinct questions. The strict machine field records that the
preregistered all-family rule was not fully met. The scientific synthesis records that 11 of 12
families across models, references, and component types met every condition, while the remaining
family retained a positive, statistically detectable association. C4a evaluates ranking of finite
component effects; it does not test whether Endpoint outperforms activation-space Taylor or other
predictors.

**Table 4. The 12 C4a families for RQ3.** Each row has $n=400$; $p$ is adjusted over all 28
tests by Holm's method.

| Model | Reference | Component family | Median Spearman | 95% CI | Holm $p$ | Family status |
|---|---|---|---:|---:|---:|---|
| ViT-B/16 | normalized zero | heads | 0.2909 | [0.2813, 0.2982] | 2.17e-119 | SUPPORTED |
| ViT-B/16 | normalized zero | routing/value | 0.0996 | [0.0748, 0.1226] | 3.00e-14 | NOT_SUPPORTED |
| ViT-B/16 | normalized zero | residual branches | 0.1774 | [0.1548, 0.1952] | 2.56e-46 | SUPPORTED |
| ViT-B/16 | blur | heads | 0.3063 | [0.2957, 0.3181] | 2.17e-119 | SUPPORTED |
| ViT-B/16 | blur | routing/value | 0.1565 | [0.1322, 0.1774] | 9.96e-29 | SUPPORTED |
| ViT-B/16 | blur | residual branches | 0.2091 | [0.1817, 0.2374] | 1.46e-41 | SUPPORTED |
| DeiT-S/16 | normalized zero | heads | 0.1910 | [0.1769, 0.2052] | 1.88e-76 | SUPPORTED |
| DeiT-S/16 | normalized zero | routing/value | 0.1065 | [0.0817, 0.1291] | 2.11e-10 | SUPPORTED |
| DeiT-S/16 | normalized zero | residual branches | 0.1352 | [0.1148, 0.1643] | 4.35e-19 | SUPPORTED |
| DeiT-S/16 | blur | heads | 0.1817 | [0.1688, 0.1951] | 3.82e-72 | SUPPORTED |
| DeiT-S/16 | blur | routing/value | 0.1600 | [0.1278, 0.1822] | 9.90e-31 | SUPPORTED |
| DeiT-S/16 | blur | residual branches | 0.2287 | [0.2035, 0.2504] | 3.70e-54 | SUPPORTED |

### 5.5 RQ4: Full-resolution component maps did not provide the registered spatial gain

RQ4 required each model-reference family to achieve a paired median improvement of at least 0.05
over total G×I, raw gradient, regional Taylor, and random regions simultaneously. None of the four
families met this joint condition (Figure 26). Endpoint-minus-random median effects were positive
in all four families—0.001069, 0.004212, 0.010689, and 0.008460—but remained well below 0.05.
Differences from total G×I and raw gradient were close to zero and mixed in direction.

Regional Taylor attained a higher median effect in every family. Endpoint-minus-regional-Taylor
contrasts were −0.023433, −0.023697, −0.029900, and −0.035630, with all bootstrap intervals below
zero (Figure 27). Figure 24 shows real block-5 component and baseline maps; the maps visualize
spatial form, whereas the statistical comparisons use raw arrays and finite-intervention curves.
RQ4 is therefore `NOT_SUPPORTED`: for the fixed heads, region pool, spatial budget, references,
and effect metric, full-resolution component maps did not show joint information gain over the
four mandatory baselines.

**Table 5. Image-level paired-median C4b contrasts for RQ4.** Values are Endpoint minus the
baseline. A family required all four contrasts to be at least +0.05.

| Model | Reference | vs total G×I | vs raw gradient | vs regional Taylor | vs random | Family status |
|---|---|---:|---:|---:|---:|---|
| ViT-B/16 | normalized zero | −0.0009 | −0.0000 | −0.0234 | 0.0011 | NOT_SUPPORTED |
| ViT-B/16 | blur | −0.0010 | −0.0007 | −0.0237 | 0.0042 | NOT_SUPPORTED |
| DeiT-S/16 | normalized zero | −0.0018 | −0.0006 | −0.0299 | 0.0107 | NOT_SUPPORTED |
| DeiT-S/16 | blur | 0.0002 | −0.0002 | −0.0356 | 0.0085 | NOT_SUPPORTED |

## 6. Discussion

### 6.1 Identity, correctness, and utility are different conclusions

Endpoint identity follows from the chain rule: for the same input-endpoint scalar target, the base
map is $x\odot\nabla_xs(x)$. Structural closure and analytic JVP checks show that the
implementation realizes the registered target, hook, cotangent, and edge accounting. Finite
activation replacement and spatial intervention ask whether the resulting quantities predict
behavior under non-infinitesimal changes. Figure 28 summarizes these evidence levels. Keeping them
separate prevents two invalid inferences: closure is not behavioral utility, and a negative
intervention result does not overturn the chain-rule identity.

### 6.2 RQ3 and RQ4 differ in their evaluation tasks

The RQ3 and RQ4 outcomes are compatible. RQ3 compares structural components within an image and
asks whether a head, routing/value component, or residual branch with greater total mass also has a
larger whole-component replacement effect. RQ4 fixes a head and asks which of 16 input regions
should be replaced first to produce a larger finite effect. The former is component-level ranking;
the latter is within-component spatial ranking.

Endpoint absolute mass retains a complete first-order distribution of a structural component over
the input, but it does not explicitly incorporate the activation displacement induced by replacing
a candidate region. Regional Taylor multiplies this displacement by the clean cotangent and is
therefore more directly aligned with the regional replacement task. Its consistent advantage in
all four families supports this task-level interpretation. The present experiments do not imply
that regional Taylor is universally superior at other layers, spatial scales, tasks, or models.

### 6.3 Interpreting the RQ3 boundary family

The ViT-B/16 normalized-zero routing/value estimate of 0.099565 is a localized boundary result. It
passed direction and Holm criteria but did not reach the frozen practical-effect threshold of 0.10,
so its family status remains `NOT_SUPPORTED`. The gap was 0.000435, and the other 11 families met
every condition. Reducing the entire RQ3 result to a failure would therefore discard the principal
cross-family empirical pattern. We retain both the strict field and the scientific synthesis and
do not assign an untested architectural or reference-specific mechanism to the single exception.

### 6.4 Implications for evaluating internal ViT explanations

An internal explanation may possess some of the following properties without possessing all of
them: complete target coverage, structurally closed accounting, locally correct derivatives,
prediction of whole-component finite effects, and spatial selection beyond strong baselines. Each
property requires a corresponding test. In particular, display resolution describes coordinate
granularity, not the amount of additional intervention-relevant information. Evaluation should
match the intended use. RQ3 directly informs structural-component screening, whereas RQ4 shows
that regional activation-space information cannot be ignored when selecting local regions.

## 7. Limitations

This study covers two ImageNet classification ViTs, one shared 400-image evaluation cohort, and
frozen standard timm implementations. The results do not directly extend to hierarchical
Transformers, detection or segmentation tasks, self-supervised models, or other data domains.
Canonical Endpoint depends on the input-coordinate origin and does not generally satisfy
baseline-difference completeness. RQ4 is restricted to block 5, three heads per model, a 4×4 region
pool, a maximum area of 25%, two references, and raw-logit isolated-effect AUC; other spatial
scales or intervention objects may yield different rankings.

Preregistered Path-IG calibration did not reach the 61/64 numerical-coverage gate for ViT-B/16.
Consequently, this paper contains no formal cross-model Endpoint-versus-Path fidelity, stability,
or cost-quality comparison. That numerical failure does not change the Endpoint-only identity,
closure, or intervention results, but it limits method-comparison claims. Finally, color images in
the paper use per-panel display scaling and support spatial-form comparisons only. Quantitative
cross-image claims use raw signed tensors, component summaries, and finite-intervention records.

## 8. Conclusion

NNS-ViT Endpoint Scanner adapts the endpoint-propagation principle of NNS to Vision Transformers,
associates registered internal targets with a common input-coordinate map, and uses complete edge
cuts to reconcile attention and residual structure. Experiments on two models and 400 images
support complete scanning, structural closure, and local differential correctness. Under matched
activation replacement, Endpoint component mass met every confirmatory condition in 11 of 12
families; RQ3 is supported overall with one registered boundary exception at 0.099565. Spatially
targeted interventions did not show joint gain over four mandatory baselines, and regional Taylor
was stronger in all four families. Component-level structural explanation and local-region
selection should therefore be treated as distinct uses requiring distinct validation.

## Data, code, and reproducibility materials

Code, frozen protocols, target registries, stage-level validation reports, the statistical release,
and 28 paper figures are maintained in the project repository. The anonymized submission link,
license, archival DOI, and data-access statement remain `[REPOSITORY AND ARCHIVE REQUIRED]`.

## References

The references used by this draft are listed in
`papers/nns_vit_endpoint/references/endpoint_scanner_references_r1.md`. Final BibTeX, DOI, page-range, and venue
metadata require a submission-stage audit.
