# 4. Experimental Setup

## 4.1 Model and preprocessing

The primary experiments use a frozen `vit_base_patch16_224.augreg2_in21k_ft_in1k`. The model contains 12 transformer blocks, 12 attention heads per block, and an embedding width of 768. It accepts 224x224 RGB inputs with 16x16 patches. The frozen Stage D state-dict SHA-256 is `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`. The R10 single-image extension separately records the checkpoint-file SHA-256 `489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d`. These hashes bind different artifact levels and are not used interchangeably.

Canonical preprocessing resizes the shorter RGB edge to 248 with bicubic interpolation, takes a 224x224 center crop, and normalizes with mean and standard deviation $(0.5,0.5,0.5)$. The learning-image reference is the all-zero normalized tensor, corresponding to uniform RGB gray 0.5. The model runs in evaluation mode.

## 4.2 Cohorts and sample roles

The Stage D primary cohort contains 500 ImageNet-1k validation images. Before outcome access, a frozen metadata algorithm ranked samples from 50,000 candidates. The selected samples are disjoint from the earlier Stage C cohort by sample ID, path, and image digest. All 500 images enter the confirmatory denominator without filtering by prediction correctness.

The 15-image C10 holdout is an earlier development and gating cohort. We report it historically and never pool it with Stage D. Stage E examines scale, upsampling, layer/head coordinates, and prediction strata after access to D10 outcomes and is secondary/descriptive. The class-explainer comparison uses an independent formal cohort of 50 images. The complete-cut branch-allocation analysis uses 100 images. Stage X2-MECH and Stage S report their own frozen valid denominators.

Figures 28-35 use the frozen image `ILSVRC2012_val_00007922.JPEG`. Its official label and clean model argmax are both class 235, German shepherd. This sample supports computational correspondence and method illustration, not new population inference.

## 4.3 Numerical integration and computational checks

Stage D computes Value-path learning images with float32 Gauss-Legendre quadrature. The order increases through $[512,1024,2048,4096,8192]$, retaining the first result that meets the conservation threshold. Each image-method pair shares the clean input, reference, target, spatial projection, fractions, and tie priorities. Direction degeneration, target-delta degeneration, non-finite values, shape errors, and missing artifacts are retained as typed invalid outcomes without replacement or zero imputation.

The R10 real-module panels use float64 CPU computation. The block-0 MLP and second residual pass convergence and completeness at order 128. The more nonlinear full-classifier path increases to order 4096. All residual identities are tested coordinatewise on uncolored RGB arrays.

## 4.4 Primary intervention protocol

The primary target is the fixed-clean-attention CLS head-direction Value path at block 0/head 9. NNS and the target-matched patch-token baseline are projected onto 3,136 4x4 cells. Insertion restores cells from the gray reference, and deletion replaces cells in the clean input. Each direction contains 21 fractions and five shared tie repeats. Repeats are averaged within an image before forming the single image-level delta-fidelity value.

The confirmatory summary reports the positive count, mean, median, sample standard deviation, and a 10,000-resample image-level paired-median percentile-bootstrap 95% interval. The frozen PASS rule requires all 500 images to be valid, at least 400 strictly positive differences, a median of at least 0.02, and a confidence-interval lower bound above zero. C10 and Stage D retain separate decisions.

## 4.5 Secondary/descriptive sensitivity analyses

Stage E compares 4x4-cell evaluation with native 14x14-patch evaluation on the same 500 images and describes four prespecified coordinates: $(0,9)$, $(4,2)$, $(6,10)$, and $(11,2)$. Prediction strata are defined by whether the model classifies an image correctly. These analyses do not reselect the primary coordinate or modify the Stage D decision. E2 exceeded its original cost limit and proceeded under an explicit stage-specific owner waiver. E3 retained two typed IG-conservation failures at block 11/head 2; E4 therefore passed at the package/workflow level with `MIXED` scientific validity.

## 4.6 Independent class-explainer comparison

In the 50-image comparison, NNS targets a conditional class logit with attention in all blocks fixed at its clean state. Value tensors, native head concatenation, output projections, residual paths, MLPs, final normalization, and the classifier remain live. The four frozen comparators are ViT Grad-CAM, full-model IG, attention rollout, and raw CLS attention. The first two are class-targeted; rollout and raw CLS attention are class-agnostic.

The primary contrast is NNS minus ViT Grad-CAM under a registered unadjusted test. NNS minus full-model IG, rollout, and raw CLS attention form a Holm-adjusted secondary family. The image is the statistical unit. We report the mean difference, bootstrap 95% interval, and registered p value for each contrast. Independently normalized atlas panels are used only to inspect spatial shape.

## 4.7 Complete cuts and depth analysis

The complete-cut implementation first tests identity between the total NNS map and same-contract full-model IG. It then tests cutwise reconstruction over 135 cuts and 38,107 components. Stage Y separates the complete class contribution at the first residual into identity and attention, and at the second residual into residual-stream and MLP components. Each absolute branch share uses the sum of the two absolute masses in its own cut as the denominator; shares from the two cuts are not added.

Deep-versus-shallow contrasts and other mechanism endpoints use the frozen Holm-adjusted families. Each of six claims receives `SUPPORTED`, `NOT_SUPPORTED`, `INCONCLUSIVE`, or `INVALID`; no omnibus PASS is constructed.

## 4.8 Stability and post-hoc exploratory analyses

Stage X evaluates geometric stability over 200 expected clean routes, of which 199 are valid. The prespecified broad-claim gate does not pass. M2 subsequently analyzes existing signed maps at the pixel, 4x4-cell, and native-patch scales, including SSE decompositions and cross-scale associations, under a post-hoc exploratory label and CPU-only boundary.

Stage S is also post-hoc exploratory. It reports target-region alignment, matched occlusion, a depth contrast in alignment, and linear probe gain. The final P2 threshold was specified after outcome access, and the original stricter P2 rule did not pass. These analyses generate hypotheses without modifying the Stage D, Stage X, or Stage Y decisions.

## 4.9 Reproducibility and evidence binding

Machine-readable manifests, receipts, and inventories record the model, images, sample order, targets, quadrature, typed failures, statistical seeds, and output digests. The R10-R4 package contains 35 figures in PNG, PDF, and SVG formats. Its terminal status is `PASS_PRESENTATION_RENDER`, with `new_inference=false`. Manuscript numbers are read from frozen JSON or CSV records; PNG files are used only for layout and visual inspection.
