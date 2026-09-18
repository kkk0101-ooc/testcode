# 5. Results and Discussion

## 5.1 Learning images correspond to the real ViT forward computation

Figures 24-27 first align normal forward values for the input, patch projection, attention, and a residual addition with their learning images in analytic examples. A complete input derivative through attention contains both $A,dV$ and $(dA)V$. Figure 22 shows that freezing clean attention produces a conditional Value-path map different from the complete-path map. Figure 23 shows that, for a downstream scalar target, residual and branch component maps add at each input coordinate.

Real ViT results preserve both correspondences. Figures 28-31 trace patch embedding, block-0 attention, and the first residual addition for the same frozen image. Figures 32-33 add the block-0 MLP and second residual addition, closing the full block. Learning images for the four fixed MLP and block-output channels satisfy completeness with 128-point Gauss-Legendre integration. The maximum pixelwise RGB additivity error at the second residual is $3.33\times10^{-16}$. The R10 recomputation of the first residual is elementwise identical to the previously stored result.

Figure 34 extends the target to the complete logit of class 235. The clean and reference logits are 8.082624 and -0.384007, yielding a target difference of 8.466630. From integration order 2048 to 4096, the relative map L1 change is $8.293\times10^{-4}$ and the completeness absolute error is 0.004874. Figure 35 uses the same path for class 235, runner-up class 174, their margin, and global minimum class 171. Their clean outputs are 8.0826, 7.6327, 0.4499, and -2.9210, respectively, and all four targets pass their frozen completeness thresholds.

These figures establish correspondence between computational objects and input coordinates. Module-own-response maps locate the origin of an internal response; the class-logit map locates the change in the final scalar; and the margin map explains a difference between two classifier outputs. Because Figures 28-35 use one frozen image, they validate the method and numerical relations but do not estimate population channel semantics or class mechanisms.

## 5.2 Fine-scale intervention for a specified internal target

Stage D evaluates the fixed block-0/head-9 Value-path target on an independent 500-image cohort. Figure 4 shows the image-level paired delta-fidelity distribution, and Figure 5 shows the insertion and deletion trajectories. The difference is strictly positive on 500/500 images. The mean is 2.6307, the median is 2.3082, and the sample standard deviation is 1.4807. A 10,000-resample paired image bootstrap gives a 95% confidence interval of [2.1111, 2.4452] for the median. The result satisfies the frozen positive-count, effect-size, and interval gates, yielding the Stage D decision `GO_FIDELITY_SCOPE`.

The target-matched patch-token baseline retains token-space contributions and distributes each native patch value uniformly over sixteen fine cells. Both methods explain the same internal direction under the same intervention protocol, but their contribution coordinates differ. For this target, input attribution provides the more useful 4x4-cell intervention ranking. Per-patch mass equality was not part of the comparison, and the intervention evaluates the ranking rather than every signed cell separately.

The earlier 15-image C10 cohort provided positive evidence for the same fidelity component: 15/15 differences were positive, with median 1.7535 and 95% CI [1.3318, 3.3813]. Its composite decision remains `NO_GO_NOVELTY` because the translation and center-crop/resize spatial gates failed. The workflow remained `BLOCKED`, with `HOLM_BLOCKED_UNDERSPECIFIED`. Stage D is an independent fidelity-only replication and extension; the two cohorts are not pooled.

## 5.3 The primary effect depends on spatial scale and internal coordinate

Figure 6 compares fine-cell, native-patch, and spatial-operator settings. Under native 14x14-patch evaluation, the median delta-fidelity over 500 images is 0, 400 differences are exactly 0, and the maximum absolute difference is only 0.00138. The primary effect is therefore not scale-invariant; it is concentrated at the within-patch 4x4-cell evaluation.

Figure 7 compares four prespecified layer/head coordinates. The valid-only median is 2.3082 at $(0,9)$, compared with 0.0734, 0.0389, and 0.0562 at $(4,2)$, $(6,10)$, and $(11,2)$. The first three coordinates have 500 valid results. Coordinate $(11,2)$ has 498 valid results and two retained typed IG-conservation failures at selection orders 117 and 352. This heterogeneity links the observed effect to the internal path being explained; the comparison covers these four coordinates.

When stratified by final prediction, the median difference is 2.2955 for 418 correctly classified images and 2.3395 for 82 misclassified images; every difference in both strata is positive. The effect therefore appears in both prediction strata. No equivalence test was specified for their magnitudes.

Stage E is post-D10 secondary/descriptive analysis. E2 passed its non-cost requirements but retained `FAIL_COST` under an explicit owner waiver limited to E2. E3 retained two typed failures, so the final E4 package/workflow passed with `MIXED` scientific validity. These states do not alter valid endpoint values, but they determine denominators and reporting language.

## 5.4 The all-head atlas shows diversity but not a stable division of labor

Figures 8-9 visualize all 144 heads and their progression across depth for one frozen median-effect image. Hotspot locations, spatial extent, and signed structure differ between blocks and heads. Figure 20 also displays component maps for heads 0, 5, and 11 at blocks 0, 5, and 11 under one class target.

These atlases describe spatial organization using indices fixed before viewing. A display average across heads differs from an algebraic class-contribution sum because heads may use different response directions and panel scaling. In the joint Stage Y tests, the effective-head-count contrast is `INCONCLUSIVE`, and neither the allocation-ablation association nor the top-versus-bottom ablation contrast reaches its frozen gate. The evidence supports diverse head maps; stable semantic specialization remains unresolved.

## 5.5 Independent class-explainer comparison shows complementarity rather than general superiority

Figures 10-11 compare fixed-clean-attention conditional class NNS, ViT Grad-CAM, full-model IG, attention rollout, and raw CLS attention. For the 50-image formal cohort, the primary contrast `NNS - Grad-CAM` has mean -0.05585, 95% CI [-0.07085, -0.04100], and registered unadjusted $p=9.9999\times10^{-6}$. NNS has lower fidelity than Grad-CAM under this comparison.

The secondary contrast `NNS - rollout` has mean 0.03943, 95% CI [0.02009, 0.05938], and Holm-adjusted $p=0.000780$, favoring NNS. `NNS - full-model IG` has mean 0.00558 and 95% CI [-0.00570, 0.01667]. `NNS - raw CLS attention` has mean 0.01126 and 95% CI [-0.00452, 0.02675]. Both have Holm-adjusted $p=0.3473$. These tests do not detect a difference and do not establish statistical equivalence.

The comparison also exposes target differences. Conditional NNS freezes attention in every block while retaining Value, residual, MLP, and classifier paths. Full-model IG allows attention to change along the interpolated input. Rollout and raw CLS attention are class-agnostic. Independently normalized panels in Figure 10 show spatial shape, whereas the full-cohort statistics in Figure 11 determine the method contrasts.

## 5.6 Complete cuts organize one class attribution by internal component

Figure 21 illustrates complete-cut reconstruction: the residual component and twelve head components add before coloring to recover the same total input IG. In the implementation control, the total NNS map and same-contract full-model IG have a maximum elementwise error of zero. A total of 135 complete cuts and 38,107 components pass validation, with a maximum pixelwise cut-reconstruction error of $1.5646\times10^{-6}$ on the audit sample.

The total map is identical to same-contract full-model IG. The added analytical object is the internal component index: the same class attribution can be inspected by head, attention, MLP, residual, or another frozen complete cut. Each cut is an alternative decomposition of the total and is interpreted separately.

## 5.7 Attention and MLP allocation change in opposite directions with depth

Figure 13 reports absolute branch shares at two residual cuts over 100 images. At the first residual, the deep-minus-shallow estimate for attention share is 0.1391, with 95% CI [0.1282, 0.1533] and Holm-adjusted $p=0.000140$. At the second residual, the corresponding estimate for MLP share is -0.2492, with 95% CI [-0.2572, -0.2349] and adjusted $p=0.000140$.

The two results support depth-dependent reallocation for the frozen class target: attention accounts for a larger absolute-mass share at the first residual in deeper blocks, while the MLP accounts for a smaller share at the second residual. The shares belong to two separate cuts, normalized within identity/attention and residual/MLP pairs, and cannot be added into one three-branch percentage.

Other Stage Y outcomes constrain further interpretation. Within-patch gain and the deep-versus-shallow spatial-support contrast for the complete class target are `NOT_SUPPORTED`. Head organization is `INCONCLUSIVE`. Patch mixing and one perturbation endpoint are `INVALID` because their valid denominators are zero. Branch reallocation is therefore not an established cause of the Stage D fidelity effect or a universal shallow-to-deep semantic law.

## 5.8 Broad stability does not pass; multiscale aggregation provides an exploratory clue

Stage X has 199 valid routes among 200 expected clean routes, but only 50 of 115 stability cells pass, and the stability noninferiority comparison with Grad-CAM fails. Broad-claim development consequently remains `NO_GO`. Fidelity under a fixed protocol cannot be generalized to stable explanations under input transformations.

Figure 12 summarizes a subsequent post-hoc exploratory analysis. On the frozen common domain, median signed Spearman association for NNS increases from 0.1137 at the pixel level to 0.2177 at 4x4 cells and 0.3378 at native patches. Within-cell residuals account for SSE fractions of 0.9255 and 0.9974, consistent with redistribution inside local units. The mechanism branch remains `MIXED_OR_INCONCLUSIVE`: coarse aggregation raises association while suppressing the within-patch differences targeted by the primary intervention experiment.

## 5.9 Post-hoc depth extension separates alignment, occlusion, and decodability

Figure 14 summarizes four Stage S post-hoc exploratory endpoints. The target-alignment difference between high-contribution NNS regions and a position control, P1, is 0.06260 with ordinary BCa 95% CI [0.05739, 0.06789] and 235/240 valid observations. The matched-occlusion contrast, P2, is 0.07494 with 95% CI [0.05708, 0.09273] and 224/240 valid observations. The deep-minus-shallow alignment contrast D1 is 0.03631 with 95% CI [0.03162, 0.04097]. The linear-probe contrast D2 is 0.04615 with 95% CI [0.04409, 0.04824].

Across 233 jointly valid images, the Pearson correlation between D1 and D2 is only 0.00161, with same-sign fraction 0.6180. Both aggregate contrasts increase with depth, but their image-level changes are nearly uncorrelated. Alignment and decodability are therefore distinct observations. The final P2 threshold of 0.07 was specified after outcome access, and the original rule did not pass. This section generates hypotheses rather than confirmatory mechanism claims.

## 5.10 Synthesis

The study turns the learning image into three testable objects. First, an input-coordinate map can reconstruct a target change and residual additions. Second, its ranking can be tested against interventions on the same target. Third, a complete class attribution can be allocated over internal cuts. The real-module figures, Stage D, and Stages X/Y address these levels separately.

The clearest positive result occurs for a fixed internal target under fine-scale intervention: the NNS input-attribution ranking outperforms a uniform spatial expansion of the target-matched token contribution. The difference disappears at native patch scale and shrinks at other coordinates, linking the result to within-patch spatial information and target choice. The independent class-output comparison further shows that NNS is not the leading method in every comparison. Its empirical role is consequently that of a complementary tool for inspecting specified internal responses and decomposing class contributions.

Complete-cut accounting advances the analysis from visually different maps to component-level bookkeeping. Heads and branches reconstruct one total attribution, and branch shares can be compared across depth. The only supported Stage Y mechanism claim concerns the depth trends of attention and MLP shares; unsupported, inconclusive, and invalid endpoints define the remaining boundary.
