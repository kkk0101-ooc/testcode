# NNS-ViT: Auditable Endpoint Maps and Structured Jacobian Decomposition for Internal Targets in Vision Transformers

- Authors: `[AUTHOR REQUIRED]`
- Affiliations: `[AFFILIATION REQUIRED]`
- Corresponding author: `[CORRESPONDING AUTHOR REQUIRED]`

> Draft R3, September 20, 2026. Figure numbers follow the sealed, organized
> `ENDPOINT-ALL-FIGURES-R2` package (Figures 1–40).
> They may be renumbered for submission according to
> `papers/nns_vit_endpoint/zh/outline.md` without changing figure content, numerical
> values, or conclusions.

## Abstract

Internal explanations of Vision Transformers (ViTs) span token-channel units, attention heads,
routing and value paths, MLP branches, residual streams, and class outputs. We present the
NNS-ViT Endpoint Scanner, which formalizes the Neural Network Scanner's input-value initialization
and current-state propagation as local Jacobian propagation. For any registered scalar target
$s$, the resulting input-coordinate map is
$E_s(x)=x\odot\nabla_xs(x)$. The base quantity is Gradient × Input; the technical contribution is
a unified registry for heterogeneous ViT targets, structurally closed attention and residual cuts,
and an independently auditable numerical protocol. Across ViT-B/16 and DeiT-S/16 on 400 shared
ImageNet evaluation images, the registry-complete scan finished 800/800 image-model tasks and
336,800 target evaluations. Structural validation passed 76,800 positive closure checks, rejected
28,800 omission or double-counting controls, and passed 6,144 condition-scaled analytic JVP
checks. A direct fidelity bridge produced 5,000 numerical comparisons between explicit layerwise
NNS propagation and Endpoint across five compatible network families; all met their
precision-specific contracts. A frozen 100-image stratified audit then evaluated 6,144 post-block
token-channel targets per model-image. In both models, late blocks had higher spatial entropy,
lower own-token locality, and greater normalized input distance than early blocks. For the frozen
ground-truth logit, late attention had larger routing-relative path scale and larger discrepancy
from a Value-only fixed-attention approximation. A post-hoc full-cohort sensitivity found that
attention-residual spatial change increased with depth, whereas MLP-residual spatial change
decreased. Behavioral calibration
remained separate: 11 of 12 preregistered component families met the family-level finite-effect
association criteria, while the all-family omnibus remained `NOT_SUPPORTED`; regional Taylor had
higher median spatial-intervention effect in all four registered families. NNS-ViT therefore
provides a unified, auditable representation of target-conditioned local input dependence, while
finite replacement and reference-specific region selection require separate validation.

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

Figure 1 summarizes the method in one line: choose a scalar response in a real ViT, differentiate
it at the observed input, and retain the original input coordinates in the resulting Endpoint map.
The study follows five linked questions:

1. **RQ1: fidelity to original NNS.** Does current-state learning-image propagation reduce to the
   same local Jacobian object, both theoretically and numerically?
2. **RQ2: unified internal-target scanning.** Can one registry scan heterogeneous scalar targets
   completely and auditably across two frozen ViTs?
3. **RQ3: structural and numerical auditability.** Do Q/K/V, routing/value, and residual
   identity/branch cuts reconstruct their registered objects and pass independent differential
   checks?
4. **RQ4: ViT mechanisms.** What depth-, token-, routing/value-, and residual-path patterns are
   visible in the Endpoint atlas?
5. **RQ5: behavioral calibration and interpretation boundary.** How do Endpoint maps and
   statistics relate to finite activation replacement, and where does reference information
   change the appropriate estimator?

![Figure 1](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_01_endpoint_scanner_method_overview.png)

**Figure 1.** Endpoint Scanner in one view. A real sloth-bear image is mapped through a selected
ground-truth class logit and the endpoint rule $E_s(i)=x_i\,\partial s(x)/\partial x_i$ to a
positive learning image in the original input coordinates. The display uses the blue-base
$\max(E,0)$ view with per-panel P99 scaling; the full signed map remains in the supplement.

This work makes four contributions. First, it formalizes original NNS learning-image propagation
as local Jacobian propagation and verifies its numerical equivalence to Endpoint before extending
the same rule to LayerNorm, GELU, softmax attention, MLPs, and residual connections. Second, it
provides one registered-scalar interface for token-channel units, heads, Q/K/V and routing/value
paths, residual branches, aggregates, and class outputs. Third, it introduces complete attention
and residual differential cuts and validates them with positive closure, negative controls,
JVP/VJP checks, finite differences, and operator tests. Fourth, it uses the resulting atlas to
measure depth-related spatial broadening and target-conditioned attention/residual mechanisms,
then calibrates those observations against finite interventions without treating Endpoint as a
universal intervention estimator.

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

This identity recovers the original NNS learning-image rule when input values initialize the
diagonal, biases are excluded from image coordinates, ReLU uses the framework subgradient at
zero, and MaxPool follows the realized framework winner, including its tie rule. It is an identity
of local input dependence rather than a decomposition of the function value. CLS tokens, position
embeddings, and model biases receive no separate image-coordinate allocation. LayerNorm, GELU,
and softmax attention are Jacobian-form generalizations beyond the original NNS operator set.

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

![Figure 2](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_02_endpoint_cnn_and_path_method_identity.png)

**Figure 2.** Three related methods answer different questions. Original NNS-CNN and the ViT
Endpoint Scanner use current-state endpoint propagation, while Path IG uses an input displacement
and a path-averaged Jacobian. The ViT extension changes the computational graph being traced, not
the base $x\odot\nabla_xs(x)$ identity.

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

![Figure 11](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_11_registered_target_atlas_design.png)

**Figure 11.** Composition of the registered target atlas. Horizontal segments show token-channel,
patch-sum, energy, head, and class targets; the difference between 457 and 385 targets is driven
primarily by the models' different attention-head counts.

![Figure 13](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_13_sloth_bear_multi_target_atlas.png)

**Figure 13.** Six preregistered scalar targets for the retained display sample
ILSVRC2012_val_00005415.JPEG, whose sealed ImageNet label is class 297, sloth bear
(`n02134418`). The blue-base panels sum signed RGB channels and display $\max(L,0)$ with
independent P99 scaling. They support within-target spatial reading, not absolute-magnitude
comparison across panels. Complete signed companions and source-array hashes are retained in the
organized R2 package.

![Figure 34](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_34_registered_target_learning_images.png)

**Figure 34.** Expanded fine-grained reading page for six registered targets from the same sealed
sloth-bear sample. The target identities and raw arrays are unchanged; the blue-base positive
rendering is a spatial-shape view with a signed companion in `supplement/signed/`.

![Figure 37](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_37_channel0_depth_progression.png)

**Figure 37.** Following the cross-layer, row-aligned organization of Figure 13 in the original
NNS paper, the figure fixes one DeiT-S/16 input and channel 0 across patch embedding, Blocks 0–11,
final CLS, and the ground-truth class logit. The top row is the positive part of each 224×224
Endpoint learning image. The bottom row is 14×14 absolute Endpoint mass pooled from the same raw
map over native 16×16 input patches; it summarizes input dependence and is **not a forward feature
map**.

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

![Figure 5](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_05_attention_q_k_v_differential_paths.png)

**Figure 5.** Query routing, key routing, and value-content contributions reconstruct the full
attention contribution within each attribution rule. Endpoint evaluates the current state; Path IG
integrates the corresponding routes along its specified path.

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

![Figure 10](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_10_structured_edge_cuts.png)

**Figure 10.** Structured cuts preserve one named downstream target while separating attention
routes or residual branches. A valid cut contains every input-to-target path exactly once.

![Figure 35](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_35_block5_structural_learning_images.png)

**Figure 35.** Fine-grained structural maps for the same image, ground-truth class target, and
block 5. Identity, head, routing, value, attention, MLP, and complete Endpoint panels use the
same structural cuts as Figure 17 but draw on the sealed, closure-validated raw component arrays
for display sample 14. The blue-base positive view is an additional reading layer and does not
replace the signed scientific representation.

![Figure 38](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_38_attention_head_depth_atlas.png)

**Figure 38.** All six attention-head edge contributions to the same ground-truth class logit for
the fixed DeiT-S/16 input at Blocks 0, 1, 5, and 11. Rows encode depth and columns encode head
index. Every panel comes from the sealed E3 raw component arrays; no head was selected by visual appearance.
Per-panel scaling supports spatial-shape comparison, not absolute cross-head or cross-layer
magnitude comparison.

![Figure 39](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_39_vit_b16_attention_head_depth_atlas.png)

**Figure 39.** The symmetric ViT-B/16 view shows all twelve heads at four representative depths
while keeping the same ground-truth class logit as the common downstream target.
Complete pages for all 12 blocks × 12 heads are retained under `supplement/all_heads/vit_b16/`, giving the
paper a compact main figure and an omission-free supplement.

![Figure 40](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_40_vit_b16_module_chain.png)

**Figure 40.** For a fixed ViT-B/16 input and channel 0, the real module chain includes patch
embedding; Norm1, attention projection, Norm2, MLP fc1, GELU, and fc2 at Blocks 0, 5, and 11;
final CLS; and the class logit. This follows the module-reading logic of NNSViT-IG Figures 28–34,
but every panel here is a current-point Endpoint learning image for a registered scalar target.

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

We use normalized-zero and RGB-blur references. The primary finite-effect quantity is the within-image
$\operatorname{Spearman}(m_k,|e_k|)$, with separate families for heads, routing/value, and residual
branches. The activation-displacement Taylor predictor

$$
p_k^{\mathrm{ref}}=\langle\nabla_{z_k}F,z_k(x)-z_k^{\mathrm{ref}}\rangle
$$

is reported as a preregistered secondary predictor but is not part of the C4a support gate
(Figures 19 and 22).

![Figure 22](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_22_endpoint_and_registered_secondary_predictors.png)

**Figure 22.** Median within-image associations for Endpoint mass and the preregistered
descriptive predictors. The outlined Endpoint column alone enters C4a; the other columns help
interpret the task but do not replace the registered test.

### 3.6 Spatially targeted intervention

The spatial-intervention audit fixes head 0, the middle head, and the last head in block 5. The 224×224 input is partitioned
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

### 3.7 Stratified token-channel audit

M3 freezes 100 shared images by a metadata-only hash rule. At every block it samples the CLS
token, 15 fixed spatially stratified patch tokens, and 32 independently hashed channels, yielding
$12\times16\times32=6{,}144$ post-block scalar units per model-image. This covers 0.338% of
ViT-B/16 and 0.677% of DeiT-S post-block token-channel coordinates. Each Endpoint is reduced to
signed and absolute mass, positive and negative mass, L2 norm, 196-patch signed and absolute
vectors, normalized entropy, own-token locality, normalized input distance, and signed
cancellation. Zero-mass maps propagate NA rather than epsilon-regularized ratios.

Units are reduced within each image and block before image-level bootstrap or paired sign tests.
CLS-patch contrasts separately reduce 32 CLS channels and $15\times32$ patch-token channels
within each image and block; phase summaries then take the median of four block-level paired
differences. The image is the inferential unit throughout.

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

**Table 6. Actual coverage of the original registry.** B=12, T=197 (CLS + 196 patches), D=768/384, and head width=64. Candidate populations follow the object definitions below. Module aggregates use the six modules at blocks 0/5/11: four D-wide modules and two 4D-wide modules per selected block.

| Model | Object | Candidate definition | Candidates | Registered/image | Coverage | Strategy | Scalar neuron |
|---|---|---|---:|---:|---:|---|---|
| vit_b16 | post-block token-channel | 12*197*D | 1,815,552 | 192 | 0.01058% | 12 blocks x 4 tokens x 4 channels | yes |
| vit_b16 | post-block patch sum/energy | 2*12*D | 18,432 | 96 | 0.52083% | 12 blocks x 4 channels x 2 aggregates | no |
| vit_b16 | attention-head coordinate | 12*H*197*64 | 1,815,552 | 144 | 0.00793% | all heads; CLS token; within-head channel 0 | yes |
| vit_b16 | module patch aggregate | 3*(4D+2*4D) | 27,648 | 18 | 0.06510% | blocks 0/5/11; six modules; channel 0 | no |
| vit_b16 | patch embedding aggregate | D | 768 | 4 | 0.52083% | four fixed channels; sum over patches | no |
| vit_b16 | final CLS coordinate | D | 768 | 1 | 0.13021% | channel 0 | yes |
| vit_b16 | ground-truth logit | 1000 | 1,000 | 1 | 0.10000% | frozen image label | output |
| vit_b16 | clean-top1 logit | 1000 | 1,000 | 1 | 0.10000% | frozen clean prediction | output |
| deit_s16 | post-block token-channel | 12*197*D | 907,776 | 192 | 0.02115% | 12 blocks x 4 tokens x 4 channels | yes |
| deit_s16 | post-block patch sum/energy | 2*12*D | 9,216 | 96 | 1.04167% | 12 blocks x 4 channels x 2 aggregates | no |
| deit_s16 | attention-head coordinate | 12*H*197*64 | 907,776 | 72 | 0.00793% | all heads; CLS token; within-head channel 0 | yes |
| deit_s16 | module patch aggregate | 3*(4D+2*4D) | 13,824 | 18 | 0.13021% | blocks 0/5/11; six modules; channel 0 | no |
| deit_s16 | patch embedding aggregate | D | 384 | 4 | 1.04167% | four fixed channels; sum over patches | no |
| deit_s16 | final CLS coordinate | D | 384 | 1 | 0.26042% | channel 0 | yes |
| deit_s16 | ground-truth logit | 1000 | 1,000 | 1 | 0.10000% | frozen image label | output |
| deit_s16 | clean-top1 logit | 1000 | 1,000 | 1 | 0.10000% | frozen clean prediction | output |

A complete scan is registry-complete: every preregistered entry was executed. Post-block coordinates cover 0.01058%/0.02115% of their candidate populations. Head coordinates select one CLS head-channel scalar. Ground-truth and clean-top1 remain two registry entries but select the same logit on correctly classified images; registry-entry counts differ from distinct scalar-function counts. Endpoint is normalized-input Gradient × Input. Structural accounting decomposes the local Jacobian, rather than a causal decomposition of function values.

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

M3 also compares explicit layerwise NNS propagation with autograd Endpoint on FCN, CNN,
CNN-MaxPool, residual-MLP, and residual-CNN graphs using 100 inputs per precision and network.
The resulting 5,000 target comparisons are numerical validation cases rather than independent
statistical samples. Eleven ViT operators receive eight float64 directional checks each, using
analytic Jacobian-vector products, autograd JVP/VJP duality, and nine central-difference steps.
The atlas reports image-level medians and 10,000-replicate bootstrap intervals. Its five frozen
tests per model use two-sided exact sign tests and a model-specific Holm family.

## 5. Results

### 5.1 RQ2: Registered target scanning was complete and tractable

All 800 image-model tasks finished uniquely, with no missing, duplicate, invalid, or formally
retried task. The 457 targets per ViT-B/16 image and 385 targets per DeiT-S/16 image yielded
336,800/336,800 finite target evaluations. Ground-truth and clean-top1 raw signed RGB class maps
were retained for every image. Full raw atlases were stored for the first 16 evaluation IDs in
both models, giving 32 atlases; neither the near-zero index nor the failure index contained an
entry. Figures 12 and 13 use retained evaluation order 14—the class-297 sloth-bear sample—to
compare the same ground-truth class-logit Endpoint across both models and to inspect six
preregistered DeiT-S/16 targets. Figure 34 retains the expanded six-target reading page from the
same sealed arrays.

![Figure 12](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_12_sloth_bear_class_logit_maps_across_models.png)

**Figure 12.** The same retained sloth-bear input and ground-truth class-logit target in DeiT-S/16
and ViT-B/16. Each blue-base panel shows $\max(L,0)$ under an independent P99 display scale. The
figure compares spatial shape, not cross-model magnitude; signed companions remain in the
supplement.

Formal wall time was 1,752.71 s, and the conservative four-card upper bound was 1.9475 GPU-hours.
Peak reserved GPU memory was 2.284 GB, peak CPU resident memory was 2.018 GB, and retained data
occupied 9.290 GB. Formal throughput was 192.16 targets/s and 0.4564 images/s. Median/P95/maximum
full-atlas latency was 7.853/8.015/8.240 s for DeiT-S/16 and 9.485/9.751/10.615 s for ViT-B/16
(Figure 14). These measurements support RQ2: the registered catalog was completed on the fixed
models and cohort and produced an auditable atlas.

![Figure 14](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_14_scan_scale_latency_and_resources.png)

**Figure 14.** Registry breadth and measured complete-atlas latency. Filled points mark median
latency and open points mark P95; runtime is an engineering measurement rather than a scientific
success gate.

**Table 2. RQ2 registered-scan coverage and resources.**

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

### 5.2 RQ3: Structural cuts satisfied positive and negative contracts

The formal closure experiment covered 400 images, two models, all 12 blocks, and eight registered
cuts per block, producing 76,800 positive checks. Every check satisfied the float32 contract
`atol=1e-5, rtol=1e-4`. The maximum relative L2 error was
$1.4144569831842091\times10^{-5}$ and the maximum absolute error was
$2.47955322265625\times10^{-5}$. The latter does not violate the contract because the elementwise
criterion is $|\mathrm{error}|\leq\mathrm{atol}+\mathrm{rtol}|\mathrm{reference}|$.

Three checker controls per block deliberately omitted a nonzero head or routing term or duplicated
a residual contribution. All 28,800 controls were rejected as intended (Figure 16). A CPU-only
validator independently reconstructed 384 raw block arrays from the first 16 evaluation IDs.
RQ3 is therefore supported for the registered cuts in these two standard timm ViTs: components
reconstruct their attention-branch or residual-total object, and the checker detects common
omission and double-counting errors.

![Figure 16](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_16_formal_closure_and_negative_controls.png)

**Figure 16.** All 76,800 positive closure checks passed and all 28,800 deliberately broken
controls were rejected. The worst relative-L2 closure error used 14.1% of the frozen tolerance.

![Figure 17](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_17_sloth_bear_structural_component_maps.png)

**Figure 17.** Closure-validated block-5 structural components for the retained sloth-bear input
and the same ground-truth class target. Main panels use the blue-base positive reading view; the
complete signed maps are retained in `supplement/signed/`.

### 5.3 Local differential implementation passed directional checks

The analytic local contract further tested whether the signed component sum equaled a JVP-derived
directional quantity at the same hook and cotangent. All 6,144 condition-scaled analytic JVP
checks on the first 16 evaluation orders of both models passed (Figure 18). The maximum absolute
error was $2.648076753197781\times10^{-5}$, and the maximum
`error/sum_abs_products` was $6.948755071487438\times10^{-6}$, within the frozen boundary
$10^{-5}+10^{-5}\sum_i|g_i(Jx)_i|$. This result verifies the local differential implementation;
it does not substitute for the finite replacements in the next section.

![Figure 18](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_18_condition_scaled_analytic_jvp_checks.png)

**Figure 18.** Coverage, pass count, and worst error-to-threshold ratio for the 6,144 analytic JVP
checks. The maximum ratio was 0.441, inside its condition-scaled failure boundary.

**Table 3. RQ3 structural-closure and local-differential checks.**

| Check | Count | Result | Maximum error |
|---|---:|---|---:|
| Registered positive closures | 76,800 | all passed | relative L2 = 1.41446e-05 |
| Omission/double-counting controls | 28,800 | all rejected as intended | not applicable |
| Condition-scaled analytic JVP | 6,144 | all passed | absolute = 2.64808e-05 |
| JVP condition-scaled ratio | 6,144 | all passed | 6.94876e-06 |

### 5.4 Behavioral calibration: broad but non-universal component-level finite-effect association

The RQ5 behavioral calibration compares Endpoint component mass $m_k$ within an image with the absolute finite replacement
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
correction. The estimate was close to the preregistered 0.10 threshold, but its confidence
interval spanned values below and above that threshold (Figure 21).

**Component-level association showed broad but non-universal support at the family level, while the preregistered
all-family omnibus criterion was not supported.** C4a remains `NOT_SUPPORTED`. Eleven of
twelve families met every condition; the remaining family retained a significant positive
association. Original C4a evaluates absolute ranking association; supplementary M1 separately
evaluates incremental performance against predictor baselines.

![Figure 20](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_20_c4a_family_estimates_and_intervals.png)

**Figure 20.** The twelve C4a family estimates with 95% image-bootstrap intervals. The shaded
region begins at the preregistered 0.10 practical threshold; orange marks the sole family that did
not satisfy every support condition.

![Figure 21](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_21_rq3_threshold_zoom_and_boundary_exception.png)

**Figure 21.** Eleven families met all criteria. The remaining ViT-B/16 normalized-zero
routing/value family had a point estimate of 0.099565; its positive association survived Holm
correction, but its interval crossed the practical threshold.

**Table 4. The 12 C4a families for RQ5 behavioral calibration.** Each row has $n=400$; $p$ is adjusted over all 28
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

### 5.5 Behavioral boundary: full-resolution component maps did not provide the registered spatial gain

The spatial component of RQ5 required each model-reference family to achieve a paired median improvement of at least 0.05
over total G×I, raw gradient, regional Taylor, and random regions simultaneously. None of the four
families met this joint condition (Figure 26). Endpoint-minus-random median effects were positive
in all four families—0.001069, 0.004212, 0.010689, and 0.008460—but remained well below 0.05.
Differences from total G×I and raw gradient were close to zero and mixed in direction.

Regional Taylor attained a higher median effect in every family. Endpoint-minus-regional-Taylor
contrasts were −0.023433, −0.023697, −0.029900, and −0.035630, with all bootstrap intervals below
zero (Figure 27). Figure 24 shows real block-5 component and baseline maps; the maps visualize
spatial form, whereas the statistical comparisons use raw arrays and finite-intervention curves.
The registered spatial-utility criterion is therefore `NOT_SUPPORTED`: for the fixed heads, region pool, spatial budget, references,
and effect metric, full-resolution component maps did not show joint information gain over the
four mandatory baselines.

![Figure 24](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_24_sloth_bear_block_5_maps_for_spatial_ranking.png)

**Figure 24.** Total G×I, raw gradient, and three block-5 head-component maps for the retained
sloth-bear spatial-ranking task. The blue-base main panels show positive spatial shape under independent P99
scales; complete signed maps remain in the supplement.

![Figure 36](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_36_spatial_ranking_learning_images.png)

**Figure 36.** Fine-grained blue-base reading view of the same spatial-ranking map definitions for
the retained sample. It visualizes the inputs to region ranking; behavioral comparisons use the
sealed raw arrays and finite-effect curves.

![Figure 25](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_25_one_image_spatial_effect_curves.png)

**Figure 25.** One-image intervention curves for the same retained sloth-bear task under the
normalized-zero reference. The page explains how the fixed 0–25% regional AUC is formed; the
confirmatory conclusions use image-level aggregation over the complete cohort.

![Figure 26](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_26_c4b_mandatory_baseline_contrasts.png)

**Figure 26.** All sixteen paired contrasts against the four mandatory baselines. The black line
marks equality and the dashed red line marks the registered +0.05 gain threshold; no interval
reaches the shaded support region.

![Figure 27](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_27_rq4_family_summary.png)

**Figure 27.** Compact family-level summary of Endpoint-minus-baseline median AUC differences.
The outlined regional-Taylor column is negative in all four families, while positive differences
against random regions remain far below +0.05.

**Table 5. Image-level paired-median C4b contrasts for RQ5 spatial calibration.** Values are Endpoint minus the
baseline. A family required all four contrasts to be at least +0.05.

| Model | Reference | vs total G×I | vs raw gradient | vs regional Taylor | vs random | Family status |
|---|---|---:|---:|---:|---:|---|
| ViT-B/16 | normalized zero | −0.0009 | −0.0000 | −0.0234 | 0.0011 | NOT_SUPPORTED |
| ViT-B/16 | blur | −0.0010 | −0.0007 | −0.0237 | 0.0042 | NOT_SUPPORTED |
| DeiT-S/16 | normalized zero | −0.0018 | −0.0006 | −0.0299 | 0.0107 | NOT_SUPPORTED |
| DeiT-S/16 | blur | 0.0002 | −0.0002 | −0.0356 | 0.0085 | NOT_SUPPORTED |



<!-- M1_SUPPLEMENT_START -->
### 5.6 M1: Predictor baselines and stratified ranking (supplementary frozen study)

M1 was frozen after the primary study, using the same 400 images, two models, 12 blocks,
two references, and ground-truth logit. Seven predictors share component definitions:
Endpoint mass, activation norm, gradient norm, absolute activation×gradient, displacement
norm, absolute activation-space Taylor, and random. Heads, routing, value/V, Q, K and both
residual families cover 400 images; value/V names the same edge. Each set uses 1,000 random
permutations, totaling 128,000,000. Image bootstrap uses 10,000 replicates and 95% CIs.
Paired Spearman tests receive joint Holm correction; median rho>0 and >0.10 are separate tests.

**Table 7. M1 within-block head ranking.** Metrics are first averaged across 12 blocks per
image and then summarized by the median over 400 images.

| Model | Reference | Endpoint rho | Taylor rho | Endpoint Capture@1 | Taylor Capture@1 | Endpoint Capture@3 | Taylor Capture@3 | Endpoint Regret@1 | Taylor Regret@1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit_b16 | normalized_zero | 0.2815 | 0.6742 | 0.5174 | 0.8038 | 0.6149 | 0.8567 | 0.1701 | 0.0571 |
| vit_b16 | blur_rgb | 0.2727 | 0.6856 | 0.5075 | 0.8203 | 0.6185 | 0.8659 | 0.1163 | 0.0346 |
| deit_s16 | normalized_zero | 0.1952 | 0.6095 | 0.5609 | 0.8028 | 0.7122 | 0.8894 | 0.2381 | 0.0860 |
| deit_s16 | blur_rgb | 0.1476 | 0.6238 | 0.5506 | 0.8426 | 0.7105 | 0.8970 | 0.1326 | 0.0443 |

Taylor had higher Spearman and Capture and lower Regret in all four combinations.
Across-block residual identity showed positive Endpoint-minus-Taylor median Spearman
differences of 0.0865–0.4209, all passing joint Holm; corresponding Capture@1 median
differences were zero. Incremental utility depends on the component and metric. QKV
triplets and residual pairs use pairwise/top1 accuracy, with Spearman recorded as NA.
Pooled results first average the two models per image and are descriptive. Complete
stratified tables, CIs, random-relative gains and source recomputation are in
docs/endpoint_m1_results_20260918.md and the M1 result root. Original C4a remains NOT_SUPPORTED.
<!-- M1_SUPPLEMENT_END -->

### 5.7 RQ1/RQ4: NNS fidelity and depth-related spatial broadening

Explicit layerwise NNS propagation and Endpoint met their precision-specific contracts in all
5,000 target comparisons. The overall relative-L2 median, 99th percentile, and maximum were
$1.088\times10^{-16}$, $1.602\times10^{-7}$, and $2.427\times10^{-7}$. Float64 had a
maximum of $4.769\times10^{-16}$; float32 had a 99th percentile of $1.782\times10^{-7}$.
All 88 operator-direction records passed. Maximum analytic-autograd JVP error was
$3.431\times10^{-16}$, maximum JVP-VJP duality error was $2.344\times10^{-14}$, and the
largest best-step finite-difference error was $1.632\times10^{-10}$ (Figure 32).

![Figure 32](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_32_original_nns_versus_endpoint_numerical_equivalence.png)

**Figure 32.** Direct numerical equivalence of explicit original-NNS propagation and Endpoint
across five compatible network families. Float64 remains near machine precision; float32
differences remain near $10^{-7}$.

The 200 image-model tasks produced 1,228,800 frozen token-channel Endpoint records. At the image
level, every model-specific late-minus-early contrast had the registered direction: entropy
increased by 0.2201/0.2250, locality decreased by 0.3558/0.3424, and normalized distance increased
by 0.1312/0.1219 for ViT-B/16 and DeiT-S. Each contrast had the same sign in 100/100 images and
passed its five-test model-specific Holm family. The complete trajectories show that most entropy
growth occurs through early and middle blocks, whereas locality continues to fall and distance
continues to rise through block 11 (Figure 29). This supports **spatial broadening of local input
dependence** in the frozen unit sample, rather than a uniform whole-image state or global-semantic
claim.

![Figure 29](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_29_m3_block_trajectories.png)

**Figure 29.** Depth trajectories of entropy, locality, normalized input distance, and net-to-
absolute mass. Bands are image-bootstrap 95% intervals over the frozen 100-image sample.

CLS-unit entropy exceeded patch-unit entropy at every block in both models (Figure 30). The paired
block-level gap narrowed from 0.4752/0.4712 at block 0 to 0.0086/0.0238 at block 11. With
block-first aggregation, the early/middle/late image-level gaps were 0.3082/0.0592/0.0283 for
ViT-B/16 and 0.3084/0.0874/0.0406 for DeiT-S. These descriptive values recover the expected
broader CLS input dependence without treating entropy as causal information integration.

![Figure 30](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_30_m3_cls_minus_patch_entropy_by_block.png)

**Figure 30.** Paired CLS-minus-patch entropy differences by block. The positive gap narrows with
depth in both models but remains above zero at block 11.

Signed cancellation was near saturation, so we additionally report
$R_{\mathrm{net}}=|\sum_iE_i|/\sum_i|E_i|$. Across all sampled units, image-level median
$R_{\mathrm{net}}$ fell from 0.00872 to 0.00144 in ViT-B/16 and from 0.00712 to 0.00132 in
DeiT-S between blocks 0 and 11. A 120-map raw subset separates cancellation within RGB channels,
within patches after RGB aggregation, and across signed patch totals (Figure 31). The nested
survival identity held to $6.94\times10^{-18}$. Recomputing the same 24 maps in float64 changed
any cancellation metric by at most $1.84\times10^{-6}$. Absolute mass therefore measures total
local first-order magnitude, not net directional effect.

![Figure 31](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_31_m3_cancellation_scales.png)

**Figure 31.** Cancellation ratios across RGB channels, within patches, across signed patch
totals, and over the whole image. The panels use a common model palette and aligned block anchors.

The frozen float32 directional check retained 36 failures among 57,600 conditions (0.0625%), all
from the first sampled patch token in blocks 7-11. Float32 error increased as the tested step
decreased, whereas the same coordinates in float64 showed an error decrease followed by
round-off increase (Figure 33). All 36 float64 counterparts passed. The post-block targets do not
identify a unique upstream operator, so the audit records a precision-sensitive finite-difference
failure rather than assigning it to LayerNorm, softmax, or another operator.

![Figure 33](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_33_m3_finite_difference_diagnostics.png)

**Figure 33.** Finite-difference error across step sizes for the 36 retained float32 cases and
their float64 counterparts. The precision-separated curves support a numerical, rather than
operator-specific, interpretation of the retained failures.


<!-- M4_MECHANISM_START -->
### 5.8 RQ4: Attention routing/value and residual mechanisms

M4 covered a shared 400-image cohort, both models, all 12 blocks, and every head for the frozen ground-truth logit. The result contains 86,400 image-block-head attention records, 19,200 image-block-residual records, and 691,200 direction-head JVP records. Per-head Q+K+V reconstructed the native attention Endpoint, routing+value reconstructed the same object, and identity/carry plus branch reconstructed the full logit Endpoint at both residual additions.

The independent CPU audit recomputed 230,400 E3 summary fields. Maximum absolute Q/K/V and residual closure errors were 1.574e-05 and 2.480e-05. The maximum relative error between the explicit attention differential and native JVP was 2.502e-05. All 19,200 identity/carry omission controls failed closure as intended, and 384 attention-detach block checks passed for the first 16 frozen evaluation orders.

#### Attention routing and value

| Model | Block | Routing share | Frozen-attention Endpoint error | Frozen-attention JVP error | Mean-query saturation | Routing-value path cancellation |
|---|---:|---:|---:|---:|---:|---:|
| vit_b16 | 0 | 0.3092 | 0.4407 | 0.4508 | 0.4656 | 0.1555 |
| vit_b16 | 5 | 0.5239 | 0.7792 | 0.8547 | 0.3183 | 0.2342 |
| vit_b16 | 11 | 0.5154 | 0.8022 | 0.9516 | 0.3470 | 0.3069 |
| deit_s16 | 0 | 0.2041 | 0.2774 | 0.3099 | 0.3825 | 0.1377 |
| deit_s16 | 5 | 0.4852 | 0.7375 | 0.8548 | 0.3191 | 0.2635 |
| deit_s16 | 11 | 0.4808 | 0.8488 | 0.9636 | 0.3329 | 0.3692 |

Frozen attention retains only A dV. Endpoint and JVP errors therefore quantify omitted routing differentials along the input-scaling direction and eight frozen Rademacher directions, respectively. The tests below aggregate heads and blocks within each image before image-level inference; Holm families were frozen separately by model.

| Model | Domain | Type | Frozen test | Image-level median | Pos/neg/tie | Holm p | Decision |
|---|---|---|---|---:|---:|---:|---|
| vit_b16 | attention | attention-head | late_minus_early_routing_share | 0.0523 | 360/40/0 | 1.715e-65 | SUPPORTED_DIFFERENCE |
| vit_b16 | attention | attention-head | late_minus_early_frozen_attention_endpoint_error | 0.1444 | 383/17/0 | 5.545e-91 | SUPPORTED_DIFFERENCE |
| vit_b16 | attention | attention-head | saturation_routing_relation | -0.2937 | 5/395/0 | 2.611e-109 | SUPPORTED_DIFFERENCE |
| vit_b16 | attention | attention-head | saturation_error_relation | -0.2238 | 12/388/0 | 7.108e-98 | SUPPORTED_DIFFERENCE |
| vit_b16 | residual | attention | late_minus_early_identity_share | -0.1242 | 38/362/0 | 4.068e-67 | SUPPORTED_DIFFERENCE |
| vit_b16 | residual | attention | late_minus_early_path_cancellation | -0.0105 | 173/227/0 | 7.970e-03 | SUPPORTED_DIFFERENCE |
| vit_b16 | residual | mlp | late_minus_early_identity_share | 0.1798 | 396/4/0 | 1.644e-111 | SUPPORTED_DIFFERENCE |
| vit_b16 | residual | mlp | late_minus_early_path_cancellation | 0.0688 | 304/96/0 | 2.840e-26 | SUPPORTED_DIFFERENCE |
| deit_s16 | attention | attention-head | late_minus_early_routing_share | 0.1355 | 399/1/0 | 9.317e-118 | SUPPORTED_DIFFERENCE |
| deit_s16 | attention | attention-head | late_minus_early_frozen_attention_endpoint_error | 0.2896 | 400/0/0 | 3.098e-120 | SUPPORTED_DIFFERENCE |
| deit_s16 | attention | attention-head | saturation_routing_relation | -0.3143 | 15/379/6 | 5.158e-92 | SUPPORTED_DIFFERENCE |
| deit_s16 | attention | attention-head | saturation_error_relation | -0.2286 | 52/338/10 | 1.794e-52 | SUPPORTED_DIFFERENCE |
| deit_s16 | residual | attention | late_minus_early_identity_share | -0.0449 | 48/352/0 | 6.053e-58 | SUPPORTED_DIFFERENCE |
| deit_s16 | residual | attention | late_minus_early_path_cancellation | -0.0074 | 171/229/0 | 4.314e-03 | SUPPORTED_DIFFERENCE |
| deit_s16 | residual | mlp | late_minus_early_identity_share | 0.1780 | 397/3/0 | 1.652e-113 | SUPPORTED_DIFFERENCE |
| deit_s16 | residual | mlp | late_minus_early_path_cancellation | -0.0588 | 103/297/0 | 6.661e-23 | SUPPORTED_DIFFERENCE |

Saturation relations use Spearman correlations across heads within each image and layer before aggregation, separating head-level association from common depth trends. The table reports their observed directions without assuming that sharper softmax rows must yield larger routing omission.

#### Residual identity and branch

| Model | Residual | Block | Identity share | Path cancellation | Identity-branch cosine | Output-identity patch TV |
|---|---|---:|---:|---:|---:|---:|
| vit_b16 | attention | 0 | 0.8847 | 0.1465 | -0.3315 | 0.0150 |
| vit_b16 | attention | 5 | 0.6100 | 0.2381 | 0.0606 | 0.1089 |
| vit_b16 | attention | 11 | 0.4442 | 0.3578 | -0.0506 | 0.1803 |
| vit_b16 | mlp | 0 | 0.2225 | 0.1654 | 0.1849 | 0.1398 |
| vit_b16 | mlp | 5 | 0.5648 | 0.2450 | 0.1321 | 0.1287 |
| vit_b16 | mlp | 11 | 0.6418 | 0.6333 | -0.9071 | 0.1023 |
| deit_s16 | attention | 0 | 0.8729 | 0.1820 | -0.4181 | 0.0284 |
| deit_s16 | attention | 5 | 0.6775 | 0.1983 | 0.1826 | 0.0798 |
| deit_s16 | attention | 11 | 0.6559 | 0.2538 | 0.0757 | 0.1047 |
| deit_s16 | mlp | 0 | 0.3078 | 0.2831 | -0.0186 | 0.1798 |
| deit_s16 | mlp | 5 | 0.6257 | 0.1886 | 0.3342 | 0.0923 |
| deit_s16 | mlp | 11 | 0.6079 | 0.5878 | -0.7599 | 0.1276 |

Identity share partitions absolute Endpoint mass between the skip and branch paths. Path cancellation compares their signed sum with the sum of their absolute masses. Output-identity patch TV measures the spatial change from the identity-only distribution to the residual output. For ViT-B/16, the minimum/maximum spatial-TV blocks were 0/11 for attention residuals and 8/10 for MLP residuals; the DeiT-S pairs were 1/4 and 9/0. These locations are descriptive medians over the frozen cohort.

| Model | Phase | Attention−MLP identity share | Attention−MLP path cancellation | Attention−MLP spatial TV |
|---|---|---:|---:|---:|
| vit_b16 | early | 0.3382 | 0.0004 | -0.0902 |
| vit_b16 | middle | 0.0327 | 0.0165 | -0.0050 |
| vit_b16 | late | 0.0376 | -0.0664 | -0.0173 |
| deit_s16 | early | 0.2697 | -0.0564 | -0.0796 |
| deit_s16 | middle | 0.0280 | 0.0009 | -0.0026 |
| deit_s16 | late | 0.0512 | -0.0041 | -0.0047 |

| Model | Residual | Branch share–path cancellation Spearman across 12 blocks |
|---|---|---:|
| vit_b16 | attention | 0.5140 |
| vit_b16 | mlp | -0.2657 |
| deit_s16 | attention | 0.3182 |
| deit_s16 | mlp | 0.4650 |

The first table directly compares attention and MLP residuals. The second asks whether path cancellation changes with branch mass. Both are frozen A1 descriptive analyses outside the original M4 Holm families.

#### Signed, absolute, and cancellation views

Every path retains signed sum, absolute mass, positive/negative mass, and within-map cancellation. M4 keeps within-map cancellation separate from routing/value or identity/branch path cancellation. Zero denominators, zero norms, and constant-head correlations propagate as NA without epsilon substitution.

| Model | Path | Block | Signed magnitude | Absolute mass | Map cancellation |
|---|---|---:|---:|---:|---:|
| vit_b16 | routing | 0 | 0.0362 | 2.6596 | 0.9867 |
| vit_b16 | routing | 5 | 0.0600 | 33.7856 | 0.9982 |
| vit_b16 | routing | 11 | 0.0608 | 40.7842 | 0.9982 |
| vit_b16 | value | 0 | 0.0306 | 7.6041 | 0.9952 |
| vit_b16 | value | 5 | 0.0380 | 28.0522 | 0.9986 |
| vit_b16 | value | 11 | 0.0654 | 39.0241 | 0.9981 |
| vit_b16 | attention_identity | 0 | 1.3315 | 495.2271 | 0.9972 |
| vit_b16 | attention_identity | 5 | 0.9816 | 387.1208 | 0.9977 |
| vit_b16 | attention_identity | 11 | 0.8778 | 313.3689 | 0.9974 |
| vit_b16 | attention_branch | 0 | 0.3930 | 63.8344 | 0.9934 |
| vit_b16 | attention_branch | 5 | 0.4140 | 250.3045 | 0.9983 |
| vit_b16 | attention_branch | 11 | 0.6133 | 398.0202 | 0.9986 |
| vit_b16 | mlp_identity | 0 | 0.6860 | 130.4998 | 0.9945 |
| vit_b16 | mlp_identity | 5 | 1.0976 | 334.0699 | 0.9971 |
| vit_b16 | mlp_identity | 11 | 1.8214 | 827.8407 | 0.9977 |
| vit_b16 | mlp_branch | 0 | 0.6309 | 438.0538 | 0.9986 |
| vit_b16 | mlp_branch | 5 | 0.4504 | 271.9139 | 0.9983 |
| vit_b16 | mlp_branch | 11 | 0.7961 | 460.2701 | 0.9981 |
| deit_s16 | routing | 0 | 0.0756 | 4.5294 | 0.9829 |
| deit_s16 | routing | 5 | 0.0506 | 38.3865 | 0.9986 |
| deit_s16 | routing | 11 | 0.0504 | 35.9257 | 0.9986 |
| deit_s16 | value | 0 | 0.0818 | 18.4261 | 0.9950 |
| deit_s16 | value | 5 | 0.0776 | 40.1955 | 0.9979 |
| deit_s16 | value | 11 | 0.0586 | 38.8767 | 0.9985 |
| deit_s16 | attention_identity | 0 | 1.6888 | 459.3077 | 0.9960 |
| deit_s16 | attention_identity | 5 | 1.1087 | 370.3635 | 0.9969 |
| deit_s16 | attention_identity | 11 | 1.1539 | 372.6598 | 0.9966 |
| deit_s16 | attention_branch | 0 | 0.4543 | 63.9204 | 0.9922 |
| deit_s16 | attention_branch | 5 | 0.4112 | 169.7186 | 0.9974 |
| deit_s16 | attention_branch | 11 | 0.3639 | 185.3893 | 0.9980 |
| deit_s16 | mlp_identity | 0 | 0.8741 | 170.6143 | 0.9946 |
| deit_s16 | mlp_identity | 5 | 1.1847 | 317.7498 | 0.9962 |
| deit_s16 | mlp_identity | 11 | 1.7620 | 614.5721 | 0.9973 |
| deit_s16 | mlp_branch | 0 | 0.7619 | 412.5634 | 0.9981 |
| deit_s16 | mlp_branch | 5 | 0.3466 | 197.6662 | 0.9982 |
| deit_s16 | mlp_branch | 11 | 0.4532 | 354.6503 | 0.9988 |

#### Review-R2 sensitivity and estimand refinement

Review-R2 tested whether the large late-block Value-only JVP discrepancies were driven by small
denominators. At block 11, the median relative discrepancies were 0.9516 for ViT-B/16 and 0.9636
for DeiT-S/16; the corresponding full-response denominators were 14.9711 and 1.5155, while
absolute discrepancies were 13.8064 and 1.4228. The discrepancies therefore occur at the same
norm scale as the full response rather than arising generally from near-zero denominators.

![M4 Review Figure 1](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/supplement/m4_review/supp_m4_review_01_m4_review_jvp_diagnostics.png)

**M4 Review Figure 1.** Block-11 distributions of normalized JVP discrepancy, absolute
discrepancy, and full-response denominator for both models.

After adjustment for head projection-input activation norm and downstream-cotangent norm,
mean-query saturation remained negatively associated with routing share, Endpoint error, and JVP
error in both models. The fully adjusted correlations were −0.1442/−0.3257,
−0.1129/−0.2185, and −0.1453/−0.2786 for ViT-B/16 and DeiT-S/16, respectively. CLS-query
relations were weaker: the adjusted ViT-B routing-share and Endpoint-error tests had Holm
$p=0.0629$. The supported summary therefore concerns all-query mean saturation, with the
association concentrated in early and middle blocks.

![M4 Review Figure 2](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/supplement/m4_review/supp_m4_review_02_m4_review_saturation_robustness.png)

**M4 Review Figure 2.** Unadjusted and partial-rank saturation relations, including phase-level
sensitivity and the weaker CLS-query result.

The full-cohort spatial analysis measured total variation between output and identity 14×14 patch
absolute distributions. Attention residual late-minus-early differences were +0.0520
([0.0488, 0.0547]) for ViT-B/16 and +0.0282 ([0.0257, 0.0300]) for DeiT-S/16; MLP residual
differences were −0.0187 ([−0.0220, −0.0150]) and −0.0462 ([−0.0516, −0.0412]). All four
post-hoc sensitivity tests survived Holm correction. Branch-relative path scale and spatial TV
were positively associated within images across blocks.

![M4 Review Figure 3](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/supplement/m4_review/supp_m4_review_03_m4_review_residual_spatial.png)

**M4 Review Figure 3.** Full-cohort residual spatial-change sensitivity for attention and MLP
sublayers.

On 16 pre-retained class-logit maps per model, Full–Value patch TV rose from 0.0550/0.0307 at
block 0 to 0.1865/0.1811 at block 11. Its 12-block trajectory was directionally compatible with
M3 entropy, locality, and distance, but M3 used 100-image token-channel targets with only one
overlapping image per model. This is a model-level trajectory bridge, not a unit-level or causal
link between routing and spatial broadening.

![M4 Review Figure 4](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/supplement/m4_review/supp_m4_review_04_m4_review_m3_m4_bridge.png)

**M4 Review Figure 4.** Descriptive bridge between M3 spatial trajectories and full-versus-Value
class-logit map differences.

#### Figures

- Image #1 (m4_attention_depth): routing share, frozen-attention Endpoint error, and eight-direction JVP error across blocks; bands are 10,000-replicate image-bootstrap 95% intervals.
- Image #2 (m4_attention_saturation): descriptive block-level locations of saturation versus routing/error; formal association tests use within-image, within-layer head Spearman correlations.
- Image #3 (m4_residual_depth): identity share, path cancellation, and spatial TV for attention and MLP residuals.

#### Interpretation boundary

M4 quantifies local Jacobian path accounting at the observed input for the frozen ground-truth logit. A larger routing share or branch mass indicates more local input dependence along that path; causal importance requires separate finite-intervention evidence.
<!-- M4_MECHANISM_END -->

## 6. Discussion

### 6.1 Identity, correctness, and utility are different conclusions

Endpoint identity follows from the chain rule: for the same input-endpoint scalar target, the base
map is $x\odot\nabla_xs(x)$. Structural closure and analytic JVP checks show that the
implementation realizes the registered target, hook, cotangent, and edge accounting. Finite
activation replacement and spatial intervention ask whether the resulting quantities predict
behavior under non-infinitesimal changes. Figure 28 summarizes these evidence levels. Keeping them
separate prevents two invalid inferences: closure is not behavioral utility, and a negative
intervention result does not overturn the chain-rule identity.

![Figure 28](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/figures_png/figure_28_identity_correctness_utility_conclusion.png)

**Figure 28.** The paper's evidence hierarchy. Mathematical identity, implementation correctness,
and finite-intervention utility are supported by different tests and are not substituted for one
another.

### 6.2 Component ranking and spatial ranking are different evaluation tasks

The two behavioral outcomes are compatible. Component ranking compares structural components within an image and
asks whether a head, routing/value component, or residual branch with greater total mass also has a
larger whole-component replacement effect. Spatial ranking fixes a head and asks which of 16 input regions
should be replaced first to produce a larger finite effect. The former is component-level ranking;
the latter is within-component spatial ranking.

Endpoint absolute mass retains a complete first-order distribution of a structural component over
the input, but it does not explicitly incorporate the activation displacement induced by replacing
a candidate region. Regional Taylor multiplies this displacement by the clean cotangent and is
therefore more directly aligned with the regional replacement task. Its consistent advantage in
all four families supports this task-level interpretation. The present experiments do not imply
that regional Taylor is universally superior at other layers, spatial scales, tasks, or models.

### 6.3 Interpreting the component-association boundary family

The ViT-B/16 normalized-zero routing/value estimate was 0.099565, with a 95% interval of
[0.0748, 0.1226]. The directional test supports a positive association, while attainment of
the practical-effect threshold of 0.10 remains uncertain. Eleven of twelve families meeting
all conditions supports broad but non-universal association; the preregistered all-family
omnibus remains `NOT_SUPPORTED`.

### 6.4 Implications for evaluating internal ViT explanations

An internal explanation may possess some of the following properties without possessing all of
them: complete registered-target coverage, structurally closed accounting, locally correct derivatives,
prediction of whole-component finite effects, and spatial selection beyond strong baselines. Each
property requires a corresponding test. In particular, display resolution describes coordinate
granularity, not the amount of additional intervention-relevant information. Evaluation should
match the intended use. The C4a analysis directly informs structural-component screening, whereas C4b shows
that regional activation-space information cannot be ignored when selecting local regions.

### 6.5 Spatial broadening and signed cancellation

M3 adds a mechanism-level use distinct from finite-intervention prediction. Entropy, locality,
and distance jointly show that sampled post-block units depend on progressively broader and more
distant input regions. The twelve-block curves locate most of the transition in early and middle
depth rather than implying a binary local-to-global switch. At the same time, small
$R_{\mathrm{net}}$ values show that these broad maps contain opposing signed terms. This helps
explain why absolute Endpoint mass can retain component-ranking information in M1 while remaining
less aligned than reference-matched Taylor with the finite spatial replacements in M2.

### 6.6 Target-conditioned routing and sublayer-specific residual transmission

M3 and M4 describe complementary levels of the local Jacobian. M3 measures how sampled
post-block token-channel maps broaden over input coordinates. M4 fixes the ground-truth logit and
partitions its downstream-cotangent-weighted input Jacobian through attention and residual paths.
Late attention had larger routing-relative path scale and larger Value-only approximation
discrepancy in both models. Attention residuals showed positive late-minus-early spatial change,
whereas MLP residuals showed negative change. The M3 and M4 depth trajectories are directionally
compatible, but their targets and cohorts differ; the evidence supports a model-level mechanism
bridge rather than a unit-level causal chain.

Mean-query saturation retained negative associations with routing scale and both approximation
errors after adjustment for head-output and downstream-cotangent norms. CLS-query associations
were smaller, and two fully adjusted ViT-B tests did not survive the post-hoc sensitivity family.
The finding is therefore specific to all-query mean saturation and to within-image, within-layer
head comparisons.

### 6.7 Method positioning and reference conditions

For a registered scalar target $F$ and normalized input $x$, Endpoint is
$E(x)=x\odot\nabla_xF(x)$. It requires no externally selected intervention reference, but it is
not coordinate-free: if $x=(r-\mu)/\sigma$, then
$x_i\partial F/\partial x_i=(r_i-\mu_i)\partial F/\partial r_i$. The normalization mean is
therefore the raw-RGB origin of canonical Endpoint. For a logit target, it is Input × Gradient in
normalized-input coordinates.

Across all 163,200 paired M1 component records, Endpoint absolute mass and signed sum were
identical under normalized-zero and blur-rgb references. Taylor magnitude and absolute finite
effect changed in every pair. Exact rankings agreed across references in only 46/9,600 head-block
Taylor comparisons and 64/9,600 finite-effect comparisons.

| Model | Taylor cross-reference $\rho$ | Effect cross-reference $\rho$ | Endpoint $\rho$: zero / blur | Taylor $\rho$: zero / blur |
|---|---:|---:|---:|---:|
| ViT-B/16 | 0.3782 | 0.3759 | 0.2815 / 0.2727 | 0.6742 / 0.6856 |
| DeiT-S/16 | 0.3000 | 0.3071 | 0.1952 / 0.1476 | 0.6095 / 0.6238 |

The reference changes component ordering, not merely score scale. Endpoint supplies one
observed-input local-dependence ordering; activation-space Taylor incorporates the selected
replacement displacement and is more closely matched to the finite-replacement outcomes studied
here. The method distinction is therefore an estimand and information-condition distinction.

| Method | Primary object | Internal scalar | Input map | External reference | Local-Jacobian closure | Finite replacement |
|---|---|---:|---:|---:|---:|---:|
| Endpoint | observed-input local dependence | yes | yes | no | yes | empirical |
| Activation-space Taylor | specified replacement prediction | yes | regional score | yes | not primary | displacement matched |
| Integrated Gradients | baseline-to-input path attribution | yes | yes | yes | extensible | baseline difference |
| Attention rollout | across-layer attention flow | usually no | token map | no | no | not primary |
| Grad-CAM | class-related coarse localization | partial | yes | no | no | not primary |
| Chefer/AttnLRP | Transformer relevance propagation | partial | yes | implementation-specific | implementation-specific | not unified |
| Attention weights | token allocation | attention object | token map | no | no | not by weights alone |

Attention rollout, Grad-CAM, Chefer/AttnLRP, and raw attention address different output, relevance,
or token-flow objects and are not ranked by this internal-target study. Path IG is also excluded
from cross-model conclusions because ViT-B/16 passed only 41/64 ground-truth and 40/64 clean-top1
calibration cases at the frozen 512-point maximum, below the 61/64 release gate.

### 6.8 Automatically selected mechanism cases

M6 froze twelve selection rules before image access: eleven cases from complete M3/M4 numerical
records and one M2 estimand-boundary case. Extreme cases came from predefined 5% tails with fixed
ID tie-breaking; the cross-model case was nearest to the cohort median difference. Images and
pixel-level maps were read or recomputed only after sealing the selection. The five pages below
explain aggregate results; population-level claims remain bound to the image-level M3/M4 analyses.

![M6 Figure 1](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/supplement/m6/supp_m6_01_m6_neuron_atlas.png)

**M6 Figure 1.** Automatically selected early-local, late-dispersed, CLS–patch, and high signed-
cancellation token-channel cases, with signed/absolute maps, patch mass, and registered metrics.

![M6 Figure 2](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/supplement/m6/supp_m6_02_m6_model_layer_comparison.png)

**M6 Figure 2.** Predefined representative ViT-B/16 and DeiT-S/16 block-0/block-11 comparison.
Channel IDs are not interpreted as semantic matches across models.

![M6 Figure 3](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/supplement/m6/supp_m6_03_m6_attention_cases.png)

**M6 Figure 3.** Routing-dominant, Value-dominant, and high fixed-attention-error heads. Q+K
reconstructs routing and routing+Value reconstructs full Endpoint for the same target.

![M6 Figure 4](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/supplement/m6/supp_m6_04_m6_residual_cases.png)

**M6 Figure 4.** Identity-dominant, branch-dominant, and high path-cancellation residual cases.
Output, identity, and branch share one residual-output cotangent and satisfy the closure contract.

![M6 Figure 5](../../../reports/nns_vit_endpoint_all_figures_r2_20260919/supplement/m6/supp_m6_05_m6_m2_semantic_boundary.png)

**M6 Figure 5.** Outcome-blind M2 boundary case contrasting observed-input Endpoint ordering with
reference-displacement-matched regional Taylor. The finite-effect curves are displayed only after
selection and illustrate the two estimands on the same task.

## 7. Limitations

This study covers two ImageNet classification ViTs, one shared 400-image evaluation cohort, and
frozen standard timm implementations. The results do not directly extend to hierarchical
Transformers, detection or segmentation tasks, self-supervised models, or other data domains.
The M3 atlas covers 6,144 frozen post-block token-channel coordinates per model-image: 0.338% of
the ViT-B/16 and 0.677% of the DeiT-S candidate populations. It does not enumerate all
token-channel combinations, MLP hidden units, or Q/K/V coordinates. Its depth findings concern
local first-order image-coordinate dependence and do not by themselves establish global semantic
representations or causal importance.
Canonical Endpoint depends on the input-coordinate origin and does not generally satisfy
baseline-difference completeness. The spatial-intervention audit is restricted to block 5, three heads per model, a 4×4 region
pool, a maximum area of 25%, two references, and raw-logit isolated-effect AUC; other spatial
scales or intervention objects may yield different rankings.
M4 path maps are conditioned on the frozen ground-truth-logit downstream cotangent. Routing and
identity shares are relative path-norm scales rather than additive percentages, and normalized JVP
discrepancy is a directional norm ratio. The M3–M4 raw-map bridge uses 16 class-logit maps per
model and is not paired to the 100-image token-channel cohort.

Preregistered Path-IG calibration did not reach the 61/64 numerical-coverage gate for ViT-B/16.
Consequently, this paper contains no formal cross-model Endpoint-versus-Path fidelity, stability,
or cost-quality comparison. That numerical failure does not change the Endpoint-only identity,
closure, atlas, or intervention results, but it limits method-comparison claims. The 36 retained
float32 finite-difference failures passed the frozen float64 recovery, so the numerical state is
`PASS_FLOAT64_RECOVERY_WITH_RETAINED_FLOAT32_FAILURE` rather than unconditional PASS. Finally, color images in
the paper use per-panel display scaling and support spatial-form comparisons only. Quantitative
cross-image claims use raw signed tensors, component summaries, and finite-intervention records.

## 8. Conclusion

NNS-ViT Endpoint Scanner adapts the endpoint-propagation principle of NNS to Vision Transformers,
associates registered internal targets with a common input-coordinate map, and uses complete edge
cuts to reconcile attention and residual structure. Experiments on two models and 400 images
support registry-complete scanning, structural closure, and local differential correctness. A
separate 100-image stratified audit directly links original NNS propagation to Endpoint and shows
depth-related spatial broadening of sampled token-channel input dependence in both models. For the
frozen ground-truth logit, late attention had larger routing-relative path scale and larger
Value-only approximation discrepancy; attention residuals showed increasing spatial change with
depth, whereas MLP residuals showed decreasing spatial change. These are target-conditioned local
path-accounting results. Under finite activation replacement, Endpoint component mass met every
family-level criterion in 11 of 12 families, supporting broad but non-universal association while
the all-family omnibus remains `NOT_SUPPORTED`. Spatial interventions showed no joint gain over
four mandatory baselines, and regional Taylor was stronger in all four families. Endpoint is thus
an auditable internal-target representation, not a universal intervention estimator.

## Data, code, and reproducibility materials

Code, frozen protocols, target registries, stage-level validation reports, the statistical release,
and the organized 40-figure package with signed, M2–M6, and all-head supplements are maintained in
the project repository. The anonymized submission link,
license, archival DOI, and data-access statement remain `[REPOSITORY AND ARCHIVE REQUIRED]`.

## References

The references used by this draft are listed in
`papers/nns_vit_endpoint/references/endpoint_scanner_references_r1.md`. Final BibTeX, DOI, page-range, and venue
metadata require a submission-stage audit.
