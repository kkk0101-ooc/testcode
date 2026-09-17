# Understanding Vision Transformers through Learning Images of Internal Responses and Contributions

## Abstract

Vision Transformers process image patches through multi-head attention, feed-forward networks, and residual connections to form visual representations. Their tokens, channels, attention heads, and branches occupy different representation spaces, which makes it difficult to examine in a unified way what these components learn from an input image and how their responses contribute to a prediction. We present NNS-ViT, an extension of the Neural Network Scanner learning-image method to Vision Transformers. Given an image, NNS-ViT constructs an input-aligned learning image for a specified internal response and combines learning images across neurons, attention heads, and network branches. The method therefore exposes ViT computation in a common image coordinate system without changing or retraining the model.

We study the complete ViT computation from patch embedding through multi-head attention, MLPs, residual connections, cross-layer contribution allocation, and the final classifier. The experiments first verify the numerical correspondence between learning images and real module responses and evaluate their spatial rankings through input interventions. We then compare explanations across spatial scales, layers, and attention heads and against several ViT explanation methods. Finally, we decompose complete class attribution over internal attention heads and branches to examine how visual contribution is organized across network depth.

The results show that NNS-ViT represents the response origins of different ViT components in input space, reveals fine-scale spatial structure that is not directly expressed by native patch representations, and connects this structure to changes in model response. Layers, attention heads, and branches produce different learning images, while class contribution is systematically reallocated across depth. Explanation behavior also varies with the target, spatial scale, and internal path, indicating that the internal operation of a ViT cannot be summarized by a single attention map or network level. NNS-ViT provides a unified input-space view for observing, comparing, and evaluating the internal computation of Vision Transformers.

**Keywords:** Vision Transformer; Neural Network Scanner; learning image; neural network visualization; internal response; contribution decomposition; interpretability

## 1. Introduction

Artificial neural networks have become central to image recognition, yet the process by which they transform an input image into internal representations and a final decision remains difficult to observe directly. Interpretability research approaches this process from several levels. Input attribution identifies regions associated with a prediction, feature visualization displays patterns preferred by internal units, and attention visualization exposes interactions between tokens. Because these methods use different coordinates, targets, and semantics, their outputs are difficult to connect into one account of how a model operates.

Vision Transformers (ViTs) make this problem more pronounced. A ViT divides an image into patches and encodes them as tokens. Each transformer block exchanges token information through multi-head attention, transforms channel representations through an MLP, and preserves and combines information through two residual additions. After multiple blocks, the CLS token enters a classifier to form class outputs (Dosovitskiy et al., 2021). The input image, patch tokens, attention heads, MLP branches, residual streams, and class logits consequently occupy different representation spaces. A feature map or attention map can expose one response, but it does not continuously show where that response originates in the image or how internal responses combine into the final output.

The Neural Network Scanner (NNS) of Dou et al. (2023) observes a network at the neuron level. Its central object is the learning image: given an input and a network unit, it constructs an image that retains input positions while representing the features learned by that unit. Learning images of multiple units can then be combined to inspect layers and network components. The original method demonstrated this unified visualization for FCNs and CNNs. ViT computation differs from these architectures because attention weights depend on the input, multiple heads jointly produce an output, MLP and residual paths alternate within every block, and the final prediction depends on all blocks and the CLS token. Extending learning images to ViTs therefore requires new internal targets that preserve the computational relationships among attention, branches, and the classifier.

We present NNS-ViT, a method for scanning a Vision Transformer through a common input-space representation. For any selected internal scalar response, NNS-ViT accumulates input contribution along a path from a reference to the observed image and places each contribution at its original location. A patch-embedding channel, an attention-output direction, an MLP branch, a residual output, and a final class logit can thus each be associated with a learning image. Under a complete internal cut, maps of different components can be combined so that one class output is expanded over attention heads, MLPs, and residual paths. Figure 1 presents the overall scanning view, Figure 2 links a real internal response to its learning image, and Figures 15–27 explain initialization, composition, and ViT operators.

This representation supports four questions about ViT computation. First, which input locations form responses in patch embedding, attention, MLP, and residual modules, and how do these responses continue through a block to the classifier? Second, does the spatial structure of a learning image correspond to actual model behavior when the input is intervened on in attribution order? Third, how do learning images vary across layers, heads, spatial scales, and class targets, and how much of a complete model explanation can be represented by one attention map? Fourth, how is a complete class contribution allocated to internal heads and branches, and how does this allocation change with depth?

We study these questions on a trained ViT-B/16. Analytic examples and real model outputs first verify learning-image completeness, attention paths, and residual additivity, extending the scan from patch embedding to the final classifier output. Large-cohort input interventions then evaluate spatial rankings and show that learning images expose within-patch spatial differences useful for a specified response, with effects that vary by scale and internal location. Cross-layer and all-head atlases reveal diverse spatial responses, while an independent explainer comparison shows that different methods retain distinct empirical roles. Finally, complete cuts allocate a final class attribution over attention and MLP components and reveal a depth-dependent reorganization of branch contribution.

The main contributions are:

1. We introduce NNS-ViT, which generates input-aligned learning images for internal responses and class outputs without modifying or retraining a Vision Transformer.
2. We establish a unified scan from patch embedding through multi-head attention, MLPs, two residual additions, and the final classifier, allowing layers, heads, branches, and output units to be observed and combined in the same input coordinates.
3. We connect learning images to testable model behavior through input interventions and evaluate their variation across spatial scales, layer/head locations, and explanation methods.
4. We organize complete class attribution into reconstructable internal component maps and use them to examine how attention and MLP contribution is allocated across depth.

Section 2 reviews neuron visualization, input attribution, ViT attention analysis, and explanation evaluation. Section 3 describes learning images, ViT operators, and complete cuts. Section 4 presents the model, data, and evaluation protocols. Section 5 reports the ViT scans and analyzes spatial structure, intervention behavior, explainer differences, and depth-wise contribution. Sections 6 and 7 discuss limitations, conclusions, and future directions.

## 2. Related Work

### 2.1 Neuron and feature visualization

Internal visualization methods broadly serve two purposes. One group searches for inputs that maximize a unit, describing the stimuli preferred by a model. Another starts from an observed input and explains how it produces an internal response or output. Feature visualization, network dissection, deconvolutional views, and class activation maps characterize optimized inputs, semantic concepts, local responses, or class-discriminative regions. These methods provide useful internal objects, but tokens, channels, and feature maps at different layers do not necessarily share a common input coordinate system or an additive accounting rule.

Dou et al. (2023) introduced the Neural Network Scanner (NNS), which constructs a learning image for a neuron in an FCN or CNN. Their method propagates input values together with their positions and combines neuron images to inspect convolution, pooling, and residual modules. We inherit the idea of associating an internal object with an input-space image, while using path integration relative to a reference to handle smooth nonlinearities, input-dependent attention, and class outputs in ViTs. The FCN/CNN experiments of the source paper are background rather than evidence for the present ViT study.

### 2.2 Input attribution and internal-path attribution

Integrated Gradients (IG) integrates gradients along a path from a reference to an observed input and satisfies axioms including sensitivity and implementation invariance. With sufficient numerical integration, the attributions sum to the output difference (Sundararajan et al., 2017). We use this form to construct a learning image for an internal response or class scalar. The reference, path, and completeness relation are therefore part of the method definition rather than display choices.

Internal attribution asks how an output is formed through hidden units or paths. Neuron conductance extends the IG view to hidden units and measures the flow of attribution through a unit (Dhamdhere et al., 2018). Our complete-cut construction likewise uses a chain-rule decomposition of input-direction and downstream-target derivatives to organize one class-logit attribution over ViT heads, attention, MLP, and residual components. This connection places the mathematics within IG, conductance, and path attribution. The contribution studied here is the ViT structural mapping, complete-cut validation, and associated experiments, not path integration or the chain rule itself. A fuller technical comparison with path-attribution literature remains required before submission `[REFERENCE REQUIRED]`.

### 2.3 ViT attention and class explanations

ViTs encode relations between patch tokens through layers of multi-head attention (Dosovitskiy et al., 2021). Raw CLS attention can be arranged on the patch grid but contains no complete downstream class derivative. Attention rollout recursively combines layer attention with residual paths to estimate cumulative token flow (Abnar and Zuidema, 2020). Chefer et al. (2021) combine relevance with attention gradients and aggregate class-related evidence across a transformer graph. Grad-CAM weights feature channels with target gradients; it was introduced for convolutional networks and has subsequently been adapted to ViTs (Selvaraju et al., 2017).

Visual-analytics studies investigate multi-head attention from another angle. Li et al. (2023) compare head importance, patch-neighborhood relations, and attention patterns across layers. Yeh et al. (2023) use joint query-key embeddings to provide a global attention view across inputs. These systems primarily expose attention structure and interaction patterns. Our central object is the input-coordinate contribution to a response. Raw attention, rollout, Grad-CAM, and full-model IG are consequently treated as distinct targets or computational paths, not interchangeable heatmaps.

### 2.4 Evaluation of explanations

Visual readability does not establish an explanation's relationship to the model response. Prior work evaluates explanations with deletion, insertion, faithfulness correlation, sensitivity, randomization, and complexity. Komorowski et al. (2023) jointly assess faithfulness, sensitivity, and complexity for ViT explanations in medical imaging, illustrating the need to specify both the explanation target and evaluation criterion. Our primary experiment uses paired insertion and deletion curves: regions are restored or replaced in attribution order, and the response of the same internal target is measured. A separate class-output experiment compares five explanation methods on a frozen 50-image cohort.

Attribution can also change substantially under small input transformations. We treat spatial stability as a separate question. Failure of a stability gate does not erase a fidelity result for its original target, but it prevents that result from being generalized to geometric robustness. The subsequent multiscale aggregation analysis is reported only as a post-hoc exploratory interpretation.

## 3. Methods

### 3.1 Problem definition and learning image

Let $x\in\mathbb{R}^{3\times224\times224}$ be a normalized image and $x_0$ the all-zero reference in normalized space, corresponding to RGB gray 0.5. For a scalar target $s(x)$ in the model, we define its learning image as

$$
L_s[i]=(x_i-x_{0i})\int_0^1
\frac{\partial s(x_0+\alpha(x-x_0))}{\partial x_i}\,d\alpha.
$$

Up to numerical integration error,

$$
\sum_i L_s[i]\approx s(x)-s(x_0).
$$

The raw learning image retains three RGB channels and their signs. A two-dimensional map is formed by signed summation over RGB. Positive-part views, negative-part views, signed-log transforms, P99 clipping, and color maps are display operations only; they do not enter completeness, residual additivity, or statistical calculations. Figures 15–17 describe coordinate initialization and image composition, while Figure 2 shows a learning image for a real attention response.

### 3.2 ViT forward structure

The frozen model receives a $224\times224$ RGB image, forms 196 image tokens from $16\times16$ patches, and adds a CLS token and position embeddings. Transformer block $b$ is written as

$$
a_b=\operatorname{Attention}(\operatorname{LN}_1(h_b)),\qquad
r_b^{(1)}=h_b+a_b,
$$

$$
m_b=\operatorname{MLP}(\operatorname{LN}_2(r_b^{(1)})),\qquad
h_{b+1}=r_b^{(2)}=r_b^{(1)}+m_b.
$$

After 12 blocks,

$$
u=\operatorname{FinalNorm}(h_{12}),\qquad
\ell=W_{\mathrm{cls}}u_{\mathrm{CLS}}+b_{\mathrm{cls}}.
$$

The block-0 MLP has dimensions $768\rightarrow3072\rightarrow768$ with GELU. The classifier is $\operatorname{Linear}(768,1000)$ applied to the final-normalized CLS token; it is distinct from the MLP branch inside a block. Figures 18 and 24–27 connect the normal forward pass to learning images in analytic examples. Figures 28–35 use real model outputs.

### 3.3 Three target classes

#### 3.3.1 Module-own responses

For a patch-embedding, attention, MLP, or residual output $q(x)\in\mathbb{R}^{196\times768}$, the display target for channel $c$ is

$$
s_{q,c}(x)=\sum_{t=1}^{196}q(x)_{t,c}.
$$

CLS participates in forward attention but is excluded from this spatial-token sum. $L_{s_{q,c}}$ identifies the input locations associated with the module response; it contains no downstream class derivative. Figures 29–33 use fixed channels 0, 255, 511, and 767, selected by index before viewing the maps.

#### 3.3.2 Fixed-clean-attention conditional Value-path target

The primary intervention experiment uses the CLS head-direction for block $b$ and head $h$. We capture attention $A_{\mathrm{clean}}$ on the clean forward pass and define

$$
g(z)=\sum_{j=0}^{196}A_{\mathrm{clean}}[\mathrm{CLS},j]V(z)_j,
$$

$$
d=g(x)-g(x_0),\qquad u=\frac{d}{\lVert d\rVert_2},\qquad
f(z)=u^\top g(z).
$$

$A_{\mathrm{clean}}$, $d$, and $u$ are detached and frozen before integration. The target retains the input-dependent Value path but cuts variation through Q, K, and softmax attention along the integration path. We therefore call it fixed-clean-attention conditional Value-path attribution. It is neither complete Q/K/V attribution nor a complete class explanation.

#### 3.3.3 Complete class outputs and complete-cut components

A complete class learning image sets $s(x)=\ell_y(x)$, with gradients through all blocks, final normalization, CLS selection, and the classifier. A complete cut selects non-overlapping internal components $z_k$ that cover the computation and combines their input-direction derivatives with downstream class derivatives to form component maps $K_k$. Within one cut,

$$
\sum_k K_k[i]\approx L_{\ell_y}[i].
$$

Different cuts are alternative accounts of the same total attribution and are not summed across cuts. Figures 19–21 show head and branch components and their reconstruction. Figure 34 directly explains a complete class logit.

### 3.4 Numerical correspondence for attention and residual paths

The derivative of a complete attention response contains both a Value term and a routing term:

$$
d(AV)=A\,dV+(dA)V.
$$

The fixed-clean-attention conditional target removes the second term, whereas real module responses and full class IG retain both. Figure 22 illustrates the distinction with a two-token example.

For a common input path and reference, a residual addition is additive at every input coordinate:

$$
L_{h+a}=L_h+L_a,\qquad
L_{r^{(1)}+m}=L_{r^{(1)}}+L_m.
$$

We test these identities on raw signed RGB arrays before applying display transforms. Figure 23 gives an analytic downstream-target example, and Figures 31–33 show the two residual additions of a real block 0.

### 3.5 Target-matched patch-token baseline

The primary comparator is the target-matched patch-token baseline with uniform within-patch allocation. For image patch token $j$,

$$
c_j=A_{\mathrm{clean}}[\mathrm{CLS},j]\,
u^\top\left(V(x)_j-V(x_0)_j\right).
$$

The 196 values form a $14\times14$ token-space map; the CLS scalar is retained separately. For evaluation, each $16\times16$ patch is divided into sixteen $4\times4$ cells and the baseline assigns

$$
c_{j,k}^{\mathrm{uniform}}=\frac{c_j}{16},\qquad k=1,\ldots,16.
$$

NNS and the baseline share the model, image, internal target, reference, intervention fractions, and tie seeds. Their coordinate definitions differ, and equality of per-patch totals is not assumed. The comparison asks whether input attribution or a uniform spatial expansion of token contribution provides the more useful intervention ranking for the same internal target. This baseline is distinct from the Stage S matched occlusion control, which matches the number, location, and shape of occluded patches.

### 3.6 Intervention and fidelity estimand

For target $t$, let $\Delta=t(x)-t(x_0)$, $\sigma=\operatorname{sign}(\Delta)$, and

$$
S(z)=\sigma\frac{t(z)-t(x_0)}{|\Delta|}.
$$

The ranking score is the positive part of $\sigma M$. Insertion restores RGB spatial units from the reference in ranking order, whereas deletion replaces units in the clean input with the reference. Both curves use fractions $0,0.05,\ldots,1$, with the actual fraction of replaced units as the horizontal coordinate. Responses are not clipped for AUC calculation.

For image $i$, method $m$, direction $d$, and tie repeat $q$, we first average five paired repeats to obtain $\bar A(i,m,d)$. The image-level contrast is

$$
\Delta_{\mathrm{fid}}(i)=\frac{1}{2}\left[
\bar A(i,\mathrm{NNS},\mathrm{ins})-\bar A(i,\mathrm{base},\mathrm{ins})
+\bar A(i,\mathrm{base},\mathrm{del})-\bar A(i,\mathrm{NNS},\mathrm{del})
\right].
$$

A positive value favors the NNS ranking under the paired insertion/deletion definition. The statistical unit is the image; repeats, curve points, cells, and methods do not increase the sample size.

### 3.7 Display rules

All numerical claims use raw arrays. Positive blue maps use the positive part and per-panel P99 scaling with a maximum fallback. Signed views use symmetric limits or a signed-log transform. NNS and baseline panels share a scale when magnitudes are compared at the same coordinate. Independently scaled panels support comparisons of hotspot location and spatial organization, not color magnitude across panels. The four classifier targets in Figure 35 use one shared maximum-absolute signed-log scale.

## 4. Experimental Setup

### 4.1 Model and preprocessing

The primary experiments use a frozen `vit_base_patch16_224.augreg2_in21k_ft_in1k`. The model contains 12 transformer blocks, 12 attention heads per block, and an embedding width of 768. It accepts 224x224 RGB inputs with 16x16 patches. The frozen Stage D state-dict SHA-256 is `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`. The R10 single-image extension separately records the checkpoint-file SHA-256 `489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d`. These hashes bind different artifact levels and are not used interchangeably.

Canonical preprocessing resizes the shorter RGB edge to 248 with bicubic interpolation, takes a 224x224 center crop, and normalizes with mean and standard deviation $(0.5,0.5,0.5)$. The learning-image reference is the all-zero normalized tensor, corresponding to uniform RGB gray 0.5. The model runs in evaluation mode.

### 4.2 Cohorts and sample roles

The Stage D primary cohort contains 500 ImageNet-1k validation images. Before outcome access, a frozen metadata algorithm ranked samples from 50,000 candidates. The selected samples are disjoint from the earlier Stage C cohort by sample ID, path, and image digest. All 500 images enter the confirmatory denominator without filtering by prediction correctness.

The 15-image C10 holdout is an earlier development and gating cohort. We report it historically and never pool it with Stage D. Stage E examines scale, upsampling, layer/head coordinates, and prediction strata after access to D10 outcomes and is secondary/descriptive. The class-explainer comparison uses an independent formal cohort of 50 images. The complete-cut branch-allocation analysis uses 100 images. Stage X2-MECH and Stage S report their own frozen valid denominators.

Figures 28-35 use the frozen image `ILSVRC2012_val_00007922.JPEG`. Its official label and clean model argmax are both class 235, German shepherd. This sample supports computational correspondence and method illustration, not new population inference.

### 4.3 Numerical integration and computational checks

Stage D computes Value-path learning images with float32 Gauss-Legendre quadrature. The order increases through $[512,1024,2048,4096,8192]$, retaining the first result that meets the conservation threshold. Each image-method pair shares the clean input, reference, target, spatial projection, fractions, and tie priorities. Direction degeneration, target-delta degeneration, non-finite values, shape errors, and missing artifacts are retained as typed invalid outcomes without replacement or zero imputation.

The R10 real-module panels use float64 CPU computation. The block-0 MLP and second residual pass convergence and completeness at order 128. The more nonlinear full-classifier path increases to order 4096. All residual identities are tested coordinatewise on uncolored RGB arrays.

### 4.4 Primary intervention protocol

The primary target is the fixed-clean-attention CLS head-direction Value path at block 0/head 9. NNS and the target-matched patch-token baseline are projected onto 3,136 4x4 cells. Insertion restores cells from the gray reference, and deletion replaces cells in the clean input. Each direction contains 21 fractions and five shared tie repeats. Repeats are averaged within an image before forming the single image-level delta-fidelity value.

The confirmatory summary reports the positive count, mean, median, sample standard deviation, and a 10,000-resample image-level paired-median percentile-bootstrap 95% interval. The frozen PASS rule requires all 500 images to be valid, at least 400 strictly positive differences, a median of at least 0.02, and a confidence-interval lower bound above zero. C10 and Stage D retain separate decisions.

### 4.5 Secondary/descriptive sensitivity analyses

Stage E compares 4x4-cell evaluation with native 14x14-patch evaluation on the same 500 images and describes four prespecified coordinates: $(0,9)$, $(4,2)$, $(6,10)$, and $(11,2)$. Prediction strata are defined by whether the model classifies an image correctly. These analyses do not reselect the primary coordinate or modify the Stage D decision. E2 exceeded its original cost limit and proceeded under an explicit stage-specific owner waiver. E3 retained two typed IG-conservation failures at block 11/head 2; E4 therefore passed at the package/workflow level with `MIXED` scientific validity.

### 4.6 Independent class-explainer comparison

In the 50-image comparison, NNS targets a conditional class logit with attention in all blocks fixed at its clean state. Value tensors, native head concatenation, output projections, residual paths, MLPs, final normalization, and the classifier remain live. The four frozen comparators are ViT Grad-CAM, full-model IG, attention rollout, and raw CLS attention. The first two are class-targeted; rollout and raw CLS attention are class-agnostic.

The primary contrast is NNS minus ViT Grad-CAM under a registered unadjusted test. NNS minus full-model IG, rollout, and raw CLS attention form a Holm-adjusted secondary family. The image is the statistical unit. We report the mean difference, bootstrap 95% interval, and registered p value for each contrast. Independently normalized atlas panels are used only to inspect spatial shape.

### 4.7 Complete cuts and depth analysis

The complete-cut implementation first tests identity between the total NNS map and same-contract full-model IG. It then tests cutwise reconstruction over 135 cuts and 38,107 components. Stage Y separates the complete class contribution at the first residual into identity and attention, and at the second residual into residual-stream and MLP components. Each absolute branch share uses the sum of the two absolute masses in its own cut as the denominator; shares from the two cuts are not added.

Deep-versus-shallow contrasts and other mechanism endpoints use the frozen Holm-adjusted families. Each of six claims receives `SUPPORTED`, `NOT_SUPPORTED`, `INCONCLUSIVE`, or `INVALID`; no omnibus PASS is constructed.

### 4.8 Stability and post-hoc exploratory analyses

Stage X evaluates geometric stability over 200 expected clean routes, of which 199 are valid. The prespecified broad-claim gate does not pass. M2 subsequently analyzes existing signed maps at the pixel, 4x4-cell, and native-patch scales, including SSE decompositions and cross-scale associations, under a post-hoc exploratory label and CPU-only boundary.

Stage S is also post-hoc exploratory. It reports target-region alignment, matched occlusion, a depth contrast in alignment, and linear probe gain. The final P2 threshold was specified after outcome access, and the original stricter P2 rule did not pass. These analyses generate hypotheses without modifying the Stage D, Stage X, or Stage Y decisions.

### 4.9 Reproducibility and evidence binding

Machine-readable manifests, receipts, and inventories record the model, images, sample order, targets, quadrature, typed failures, statistical seeds, and output digests. The R10-R4 package contains 35 figures in PNG, PDF, and SVG formats. Its terminal status is `PASS_PRESENTATION_RENDER`, with `new_inference=false`. Manuscript numbers are read from frozen JSON or CSV records; PNG files are used only for layout and visual inspection.

## 5. Results and Discussion

### 5.1 Learning images correspond to the real ViT forward computation

Figures 24-27 first align normal forward values for the input, patch projection, attention, and a residual addition with their learning images in analytic examples. A complete input derivative through attention contains both $A,dV$ and $(dA)V$. Figure 22 shows that freezing clean attention produces a conditional Value-path map different from the complete-path map. Figure 23 shows that, for a downstream scalar target, residual and branch component maps add at each input coordinate.

Real ViT results preserve both correspondences. Figures 28-31 trace patch embedding, block-0 attention, and the first residual addition for the same frozen image. Figures 32-33 add the block-0 MLP and second residual addition, closing the full block. Learning images for the four fixed MLP and block-output channels satisfy completeness with 128-point Gauss-Legendre integration. The maximum pixelwise RGB additivity error at the second residual is $3.33\times10^{-16}$. The R10 recomputation of the first residual is elementwise identical to the previously stored result.

Figure 34 extends the target to the complete logit of class 235. The clean and reference logits are 8.082624 and -0.384007, yielding a target difference of 8.466630. From integration order 2048 to 4096, the relative map L1 change is $8.293\times10^{-4}$ and the completeness absolute error is 0.004874. Figure 35 uses the same path for class 235, runner-up class 174, their margin, and global minimum class 171. Their clean outputs are 8.0826, 7.6327, 0.4499, and -2.9210, respectively, and all four targets pass their frozen completeness thresholds.

These figures establish correspondence between computational objects and input coordinates. Module-own-response maps locate the origin of an internal response; the class-logit map locates the change in the final scalar; and the margin map explains a difference between two classifier outputs. Because Figures 28-35 use one frozen image, they validate the method and numerical relations but do not estimate population channel semantics or class mechanisms.

### 5.2 Fine-scale intervention for a specified internal target

Stage D evaluates the fixed block-0/head-9 Value-path target on an independent 500-image cohort. Figure 4 shows the image-level paired delta-fidelity distribution, and Figure 5 shows the insertion and deletion trajectories. The difference is strictly positive on 500/500 images. The mean is 2.6307, the median is 2.3082, and the sample standard deviation is 1.4807. A 10,000-resample paired image bootstrap gives a 95% confidence interval of [2.1111, 2.4452] for the median. The result satisfies the frozen positive-count, effect-size, and interval gates, yielding the Stage D decision `GO_FIDELITY_SCOPE`.

The target-matched patch-token baseline retains token-space contributions and distributes each native patch value uniformly over sixteen fine cells. Both methods explain the same internal direction under the same intervention protocol, but their contribution coordinates differ. For this target, input attribution provides the more useful 4x4-cell intervention ranking. Per-patch mass equality was not part of the comparison, and the intervention evaluates the ranking rather than every signed cell separately.

The earlier 15-image C10 cohort provided positive evidence for the same fidelity component: 15/15 differences were positive, with median 1.7535 and 95% CI [1.3318, 3.3813]. Its composite decision remains `NO_GO_NOVELTY` because the translation and center-crop/resize spatial gates failed. The workflow remained `BLOCKED`, with `HOLM_BLOCKED_UNDERSPECIFIED`. Stage D is an independent fidelity-only replication and extension; the two cohorts are not pooled.

### 5.3 The primary effect depends on spatial scale and internal coordinate

Figure 6 compares fine-cell, native-patch, and spatial-operator settings. Under native 14x14-patch evaluation, the median delta-fidelity over 500 images is 0, 400 differences are exactly 0, and the maximum absolute difference is only 0.00138. The primary effect is therefore not scale-invariant; it is concentrated at the within-patch 4x4-cell evaluation.

Figure 7 compares four prespecified layer/head coordinates. The valid-only median is 2.3082 at $(0,9)$, compared with 0.0734, 0.0389, and 0.0562 at $(4,2)$, $(6,10)$, and $(11,2)$. The first three coordinates have 500 valid results. Coordinate $(11,2)$ has 498 valid results and two retained typed IG-conservation failures at selection orders 117 and 352. This heterogeneity links the observed effect to the internal path being explained; the comparison covers these four coordinates.

When stratified by final prediction, the median difference is 2.2955 for 418 correctly classified images and 2.3395 for 82 misclassified images; every difference in both strata is positive. The effect therefore appears in both prediction strata. No equivalence test was specified for their magnitudes.

Stage E is post-D10 secondary/descriptive analysis. E2 passed its non-cost requirements but retained `FAIL_COST` under an explicit owner waiver limited to E2. E3 retained two typed failures, so the final E4 package/workflow passed with `MIXED` scientific validity. These states do not alter valid endpoint values, but they determine denominators and reporting language.

### 5.4 The all-head atlas shows diversity but not a stable division of labor

Figures 8-9 visualize all 144 heads and their progression across depth for one frozen median-effect image. Hotspot locations, spatial extent, and signed structure differ between blocks and heads. Figure 20 also displays component maps for heads 0, 5, and 11 at blocks 0, 5, and 11 under one class target.

These atlases describe spatial organization using indices fixed before viewing. A display average across heads differs from an algebraic class-contribution sum because heads may use different response directions and panel scaling. In the joint Stage Y tests, the effective-head-count contrast is `INCONCLUSIVE`, and neither the allocation-ablation association nor the top-versus-bottom ablation contrast reaches its frozen gate. The evidence supports diverse head maps; stable semantic specialization remains unresolved.

### 5.5 Independent class-explainer comparison shows complementarity rather than general superiority

Figures 10-11 compare fixed-clean-attention conditional class NNS, ViT Grad-CAM, full-model IG, attention rollout, and raw CLS attention. For the 50-image formal cohort, the primary contrast `NNS - Grad-CAM` has mean -0.05585, 95% CI [-0.07085, -0.04100], and registered unadjusted $p=9.9999\times10^{-6}$. NNS has lower fidelity than Grad-CAM under this comparison.

The secondary contrast `NNS - rollout` has mean 0.03943, 95% CI [0.02009, 0.05938], and Holm-adjusted $p=0.000780$, favoring NNS. `NNS - full-model IG` has mean 0.00558 and 95% CI [-0.00570, 0.01667]. `NNS - raw CLS attention` has mean 0.01126 and 95% CI [-0.00452, 0.02675]. Both have Holm-adjusted $p=0.3473$. These tests do not detect a difference and do not establish statistical equivalence.

The comparison also exposes target differences. Conditional NNS freezes attention in every block while retaining Value, residual, MLP, and classifier paths. Full-model IG allows attention to change along the interpolated input. Rollout and raw CLS attention are class-agnostic. Independently normalized panels in Figure 10 show spatial shape, whereas the full-cohort statistics in Figure 11 determine the method contrasts.

### 5.6 Complete cuts organize one class attribution by internal component

Figure 21 illustrates complete-cut reconstruction: the residual component and twelve head components add before coloring to recover the same total input IG. In the implementation control, the total NNS map and same-contract full-model IG have a maximum elementwise error of zero. A total of 135 complete cuts and 38,107 components pass validation, with a maximum pixelwise cut-reconstruction error of $1.5646\times10^{-6}$ on the audit sample.

The total map is identical to same-contract full-model IG. The added analytical object is the internal component index: the same class attribution can be inspected by head, attention, MLP, residual, or another frozen complete cut. Each cut is an alternative decomposition of the total and is interpreted separately.

### 5.7 Attention and MLP allocation change in opposite directions with depth

Figure 13 reports absolute branch shares at two residual cuts over 100 images. At the first residual, the deep-minus-shallow estimate for attention share is 0.1391, with 95% CI [0.1282, 0.1533] and Holm-adjusted $p=0.000140$. At the second residual, the corresponding estimate for MLP share is -0.2492, with 95% CI [-0.2572, -0.2349] and adjusted $p=0.000140$.

The two results support depth-dependent reallocation for the frozen class target: attention accounts for a larger absolute-mass share at the first residual in deeper blocks, while the MLP accounts for a smaller share at the second residual. The shares belong to two separate cuts, normalized within identity/attention and residual/MLP pairs, and cannot be added into one three-branch percentage.

Other Stage Y outcomes constrain further interpretation. Within-patch gain and the deep-versus-shallow spatial-support contrast for the complete class target are `NOT_SUPPORTED`. Head organization is `INCONCLUSIVE`. Patch mixing and one perturbation endpoint are `INVALID` because their valid denominators are zero. Branch reallocation is therefore not an established cause of the Stage D fidelity effect or a universal shallow-to-deep semantic law.

### 5.8 Broad stability does not pass; multiscale aggregation provides an exploratory clue

Stage X has 199 valid routes among 200 expected clean routes, but only 50 of 115 stability cells pass, and the stability noninferiority comparison with Grad-CAM fails. Broad-claim development consequently remains `NO_GO`. Fidelity under a fixed protocol cannot be generalized to stable explanations under input transformations.

Figure 12 summarizes a subsequent post-hoc exploratory analysis. On the frozen common domain, median signed Spearman association for NNS increases from 0.1137 at the pixel level to 0.2177 at 4x4 cells and 0.3378 at native patches. Within-cell residuals account for SSE fractions of 0.9255 and 0.9974, consistent with redistribution inside local units. The mechanism branch remains `MIXED_OR_INCONCLUSIVE`: coarse aggregation raises association while suppressing the within-patch differences targeted by the primary intervention experiment.

### 5.9 Post-hoc depth extension separates alignment, occlusion, and decodability

Figure 14 summarizes four Stage S post-hoc exploratory endpoints. The target-alignment difference between high-contribution NNS regions and a position control, P1, is 0.06260 with ordinary BCa 95% CI [0.05739, 0.06789] and 235/240 valid observations. The matched-occlusion contrast, P2, is 0.07494 with 95% CI [0.05708, 0.09273] and 224/240 valid observations. The deep-minus-shallow alignment contrast D1 is 0.03631 with 95% CI [0.03162, 0.04097]. The linear-probe contrast D2 is 0.04615 with 95% CI [0.04409, 0.04824].

Across 233 jointly valid images, the Pearson correlation between D1 and D2 is only 0.00161, with same-sign fraction 0.6180. Both aggregate contrasts increase with depth, but their image-level changes are nearly uncorrelated. Alignment and decodability are therefore distinct observations. The final P2 threshold of 0.07 was specified after outcome access, and the original rule did not pass. This section generates hypotheses rather than confirmatory mechanism claims.

### 5.10 Synthesis

The study turns the learning image into three testable objects. First, an input-coordinate map can reconstruct a target change and residual additions. Second, its ranking can be tested against interventions on the same target. Third, a complete class attribution can be allocated over internal cuts. The real-module figures, Stage D, and Stages X/Y address these levels separately.

The clearest positive result occurs for a fixed internal target under fine-scale intervention: the NNS input-attribution ranking outperforms a uniform spatial expansion of the target-matched token contribution. The difference disappears at native patch scale and shrinks at other coordinates, linking the result to within-patch spatial information and target choice. The independent class-output comparison further shows that NNS is not the leading method in every comparison. Its empirical role is consequently that of a complementary tool for inspecting specified internal responses and decomposing class contributions.

Complete-cut accounting advances the analysis from visually different maps to component-level bookkeeping. Heads and branches reconstruct one total attribution, and branch shares can be compared across depth. The only supported Stage Y mechanism claim concerns the depth trends of attention and MLP shares; unsupported, inconclusive, and invalid endpoints define the remaining boundary.

## 6. Limitations

First, the primary intervention result has a specific target scope. Stage D explains the fixed-clean-attention Value path at block 0/head 9 of a frozen ViT-B/16 and compares a target-matched patch-token baseline at the 4x4-cell endpoint. It is not a complete class explanation, object-localization metric, or test of agreement with human vision. Native-patch and alternative-coordinate results show that the effect is not a uniform property of the entire ViT.

Second, spatial stability remains unresolved. The C10 translation and center-crop/resize gates fail, as do the broad Stage X stability criterion and noninferiority to Grad-CAM. Increased association after multiscale aggregation is a post-hoc exploratory observation; it neither substitutes for fine-scale stability nor changes the broad stability decision.

Third, the compared explainers do not have identical targets. Conditional NNS, full-model IG, Grad-CAM, rollout, and raw CLS attention differ in class conditioning or computational path. Their frozen fidelity contrasts remain informative, but they do not define a target-independent total order. Non-significant NNS-full IG and NNS-raw attention contrasts do not establish equivalence.

Fourth, the real-module demonstrations are based on one image. Figures 28-35 verify coordinates, completeness, residual additivity, and classifier-target differences for one frozen input. The fixed channels do not establish population semantics, and four classifier outputs do not establish a general spatial mechanism across classes.

Fifth, several extensions are descriptive or post-hoc exploratory. Stage E follows access to D10 outcomes and has `MIXED` scientific validity at E4. Stage X2-MECH and Stage S are not confirmatory. The final Stage S P2 threshold was specified after outcome access, and the original rule did not pass. These results motivate hypotheses rather than primary causal or confirmatory statements.

Sixth, the novelty claim still requires a literature-level technical audit. The learning-image idea originates with Dou et al., path integration with IG, and internal component calculations are directly related to conductance and path attribution. This study validates a ViT structural mapping and its empirical uses, but a line-by-line operational comparison and final contribution statement remain necessary before submission.

Finally, the experiments focus on one frozen ViT-B/16 and ImageNet-1k validation images. Independent studies are needed to assess other model scales, training objectives, patch sizes, references, data domains, and visual tasks.

## 7. Conclusions and Future Work

We presented NNS-ViT, a method for observing the internal operation of a Vision Transformer through learning images. Given a trained ViT and an input image, the method represents a specified internal response or class output as an image that retains the original input positions, without changing the network architecture. Learning images of tokens, channels, attention heads, and network branches can be displayed and combined in one coordinate system, making computations from otherwise different representation spaces directly comparable.

We scanned the ViT from patch embedding through attention, MLPs, two residual additions, cross-layer contribution, and the final classifier. Real-model computations show that learning images correspond to module responses and can be checked through completeness and residual additivity. Channels, heads, layers, and classifier targets form different spatial contribution structures; the correct class, a competing class, and their decision margin also yield distinct learning images. These results show how a ViT organizes multiple internal responses from the same input and progressively combines them into an output.

We further evaluated the explanatory use of learning images from several perspectives. Input interventions show that fine-scale input contribution provides spatial ranking information not directly expanded by native patch-token representations. Scale and layer/head analyses show that this information changes with the observed target and evaluation scale. Comparisons with other ViT explanation methods position NNS-ViT as a complementary view of internal response. Complete cuts allocate a final class attribution to attention heads, MLPs, residual paths, and other internal components, revealing a reorganization of branch contribution across depth while preserving reconstruction of the total attribution.

NNS-ViT establishes a visual path from input location to internal response and final class output. It can show what an individual component learns, combine components to analyze how the model operates, and test these observations through intervention and component reconstruction. Future work will extend the method to additional Vision Transformer architectures, training objectives, and visual tasks; investigate head and channel organization that persists across inputs; improve spatial stability under input transformations; and explore learning images for model diagnosis, architecture comparison, and model optimization.
