# Unified paper figures R2 · en

## Figure 1

Study overview. A fixed ViT block/head defines the internal target for NNS and the target-matched patch-token baseline with uniform within-patch allocation. The two readouts enter paired image-level intervention evaluation. The photograph is the first frozen F3C3 formal sample, reused from Figure 10. Network glyphs, contribution grids and restoration sequences are schematic; they are not measured attribution or intervention results for that image.

## Figure 2

Head-direction Value-path construction. (a) Clean attention A and direction u define f(z)=uᵀg(z) and remain fixed during integration; Value features vary with the input. (b) Signed input-space integrated gradients are evaluated along x₀+α(x−x₀), then summed over RGB channels with sign. (c) Target-matched patch-token contributions are spatialized on a 14×14 grid and uniformly divided among 16 fine cells per patch; the CLS scalar remains separate. The photograph is a frozen example, while attention and contribution grids are illustrative. In the signed grid, blue/red encode negative/positive contributions.

## Figure 3

Evaluation protocol. (a) Each 224×224 input contains 196 native patches of 16×16 pixels, or 3,136 fine cells of 4×4 pixels. (b) Insertion and deletion use the target-oriented positive ranking, a normalized-zero reference (RGB 0.5), 21 nominal fractions, and five paired tie repeats. Enlarged schematic regions make the intervention sequence visible; these are not measured masks. (c) AUCs are averaged over repeats within each image before forming ΔFID. The image is the independent unit for the median and paired bootstrap interval. AUC uses actual intervened fractions and responses are not clipped to [0,1].

## Figure 4

Primary fine-scale effect. (a) Histogram of all 500 image-level ΔFID values. The orange line marks the median and the orange region is its formal paired-image bootstrap 95% CI. (b) ECDF of the same data, with the median marked by a dot. The shaded region remains a median CI, not a simultaneous ECDF band. All 500 differences are positive; median 2.3081996222, 95% CI [2.1110806126, 2.4452354118].

## Figure 5

Median insertion and deletion trajectories for NNS and the matched patch baseline. Bands show pointwise bootstrap intervals.

## Figure 6

Spatial-scale and upsampling sensitivity. (a) Fine-cell and native-patch ΔFID distributions. (b,c) Every dot is an image-level insertion/deletion AUC, with horizontal jitter for visibility. Boxes show the 25th–75th percentiles and inner lines the medians; all 500 observations per evidence class, including extrema, are retained. IQRs are not confidence intervals. Uniform and mass-preserving nearest allocation form one exactly equivalent evidence class; bilinear and bicubic are descriptive sensitivities.

## Figure 7

One matrix of frozen layer/head coordinates and prediction strata. Each cell reports median ΔFID, the Stage E descriptive paired-image bootstrap 95% CI, and valid/expected images. All cells share a linear color scale. Block 11/head 2 retains two typed failures: 498/500 overall, 417/418 correct and 81/82 misclassified; the other coordinates have complete coverage. Coordinates were selected in Stage C. The descriptive bootstrap namespace differs from the primary analysis in Figure 4, giving the same primary median with a slightly different CI. Similar directions across strata do not establish equivalence or a result for all heads.

## Figure 8

Frozen median-effect, weak-effect, and misclassified examples. Attribution panels are independently scaled and convey spatial shape only.

## Figure 9

Head-averaged positive and negative spatial support across four depth bands for one frozen sample. Absolute magnitude is not compared across bands.

## Figure 10

Spatial comparison of NNS, pooled NNS, Grad-CAM, full-model IG, and rollout on the first four positions of the frozen formal cohort. Attribution panels use P99 display scaling.

## Figure 11

Method comparison in 50 frozen images. (a) Image-level mean-over-repeat fidelity scores, with IQR boxes and median lines. (b) Registered paired mean NNS-minus-comparator contrasts: filled bars run from zero to the estimate, diamonds mark means, and the horizontal bounds of outlined rectangles are the original bootstrap 95% CIs. Rectangle height has no statistical meaning. NNS-minus-Grad-CAM is negative, NNS-minus-rollout is positive, and intervals for full IG and raw CLS cross zero. No density is inferred from an interval. The original primary unadjusted and secondary Holm testing scopes remain unchanged. (c) Median deletion trajectories.

## Figure 12

Post-hoc exploratory multiscale mechanism analysis with a fixed flip-route example, geometric agreement, ROAD fidelity, and NNS-v2 within/between SSE decomposition.

## Figure 13

Class-logit complete-cut branch allocation. (a) Twelve-block attention and MLP absolute-mass allocation trajectories with their original blockwise 95% intervals. (b) B4/B5 deep-minus-shallow contrasts: bars encode signed estimates, diamonds mark estimates, and outlined rectangles encode the original 95% CI endpoints. Colors consistently identify attention and MLP. The two shares refer to distinct residual cuts and do not form a single two-part probability distribution. These branch findings do not imply support for other Y4 claims.

## Figure 14

Post-hoc exploratory depth extension. (a) Endpoint-specific estimates, ordinary BCa 95% CIs and valid/total image counts are tabulated without imposing a common effect axis. P1 and D1 have 235/240 valid images, P2 224/240, and D2 3424/3482; their NA counts are 5, 5, 16 and 58, respectively, with no failed images. Ordinary intervals describe precision and do not replace adjusted decision rules. (b) NNS alignment and probe gain profiles show each measure across depth; their magnitudes are not interchangeable. (c) Matched-control-corrected logit drops at three fixed patch budgets. (d) D1/D2 association in 233 valid paired images, Pearson r approximately 0.002. All S results are post-hoc exploratory, with endpoints and the P2 threshold formed after outcome access, and do not establish confirmatory or causal mechanism claims.
