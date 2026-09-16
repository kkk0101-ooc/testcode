# Unified paper figures R5 · en

## Figure 1

NNS links a selected internal ViT response to input contributions. (a) Select one head response in the same trained ViT-B/16. (b) Measured signed NNS fine-cell map for the previously frozen median-effect example (selection_order 44, ILSVRC2012_val_00002287.JPEG, block 0/head 9). Red and blue encode positive and negative contributions to the fixed head-direction Value-path scalar. The image and attribution belong to the same recorded sample. For display, colors use sign(m)·log(1+99·|m|/L)/log(100), L=max|m|, on the recorded 56×56 map, with nearest-neighbor display; each cell represents 4×4 input pixels. (c) All 500 paired NNS-minus-target-matched-token-baseline ΔFID values in the independent formal cohort are positive. Each square represents one positive image-level difference, not its magnitude. Median ΔFID is 2.3082, with the original paired-image-bootstrap 95% CI [2.1111,2.4452]. F evaluates positive-ranking fine-cell insertion/deletion; the cohort result does not individually validate the causal role of every signed cell. Figure 1 presents an internal-explanation use and its empirical support; Figures 2–3 specify the computation and evaluation, and Figures 4–5 show the full results.

## Figure 2

Two computations for a shared internal scalar. For a selected block/head, g(z)=Σ_j A(x)_{0j}V_j(z), u=[g(x)−g(x₀)]/‖g(x)−g(x₀)‖₂, and f(z)=uᵀg(z). Clean CLS attention A(x) and direction u are held fixed for this image during attribution and intervention evaluation, while Value features vary with the input. NNS integrates input gradients of f along x₀→x, sums RGB channels with sign, and aggregates the resulting pixel contributions into 4×4-pixel cells. The matched baseline computes c_j=A(x)_{0j}uᵀ[V_j(x)−V_j(x₀)], combining attention, Value change and target direction. Its 196 patch-token scalars are spatialized by assigning c_j/16 to each fine cell; the CLS scalar is retained separately. The reference is normalized zero / RGB 0.5. Fixing attention defines this explanation target, not the normal ViT training procedure. At deeper blocks, token j can mix information from other input positions, so spatializing token j at patch j does not imply that its contribution equals the NNS sum in that input patch. Paths and contribution grids are illustrative.

## Figure 3

Ranking and fidelity at a shared spatial scale. A 224×224 image has 196 native 16×16-pixel patches, each divided into 16 fine 4×4-pixel cells, giving 3,136 cells. NNS sums signed input contributions within each cell and ranks their target-oriented positive parts. The token baseline uniformly expands c_j before the same ranking rule, giving ties within each patch while preserving differences between patches. The local 1–16 ranks are illustrative; the actual experiment ranks all cells and uses five paired tie permutations for every tie group. Insertion restores high-ranked regions from the gray reference; deletion replaces them in the clean image. Both use the same area budget and evaluate the internal scalar f defined in Figure 2. Enlarged mask blocks illustrate the operation, not actual fine-cell masks or outcomes. The original protocol uses 21 fractions and AUC against actual intervention fractions, with responses unclipped. Five AUC repeats are averaged within each image before computing F=½(AUC_ins−AUC_del) and ΔFID=F_NNS−F_token. Images are the independent units for the median and paired bootstrap 95% CI.

## Figure 4

Primary fine-scale effect. (a) Histogram of all 500 image-level ΔFID values. The orange line marks the median and the orange region is its formal paired-image bootstrap 95% CI. (b) ECDF of the same data, with the median marked by a dot. The shaded region remains a median CI, not a simultaneous ECDF band. All 500 differences are positive; median 2.3081996222, 95% CI [2.1110806126, 2.4452354118].

## Figure 5

Median insertion and deletion trajectories for NNS and the matched patch baseline. Bands show pointwise bootstrap intervals.

## Figure 6

Spatial-scale and upsampling sensitivity. (a) Fine-cell and native-patch ΔFID distributions. (b,c) Every dot is an image-level insertion/deletion AUC, with horizontal jitter for visibility. Boxes show the 25th–75th percentiles and inner lines the medians; all 500 observations per evidence class, including extrema, are retained. IQRs are not confidence intervals. Uniform and mass-preserving nearest allocation form one exactly equivalent evidence class; bilinear and bicubic are descriptive sensitivities.

## Figure 7

Figure 7. Median NNS-minus-patch-baseline ΔFID at four frozen layer/head coordinates. Gray, blue and orange bars identify the overall, correct and misclassified strata; whiskers are the original Stage E descriptive paired-image-bootstrap 95% CIs. The main axis starts at zero and is shared across all coordinates. The inset repeats the last three coordinates on an explicitly labeled linear 0–0.10 scale, also starting at zero. These repeated marks are not additional observations. Expected image counts are 500/418/82; block 11/head 2 has 498/417/81 valid images, with both typed failures retained in the denominators. Other coordinates have complete coverage. Coordinates were frozen in Stage C. Strata overlap and are not independent experiments; similar directions do not establish equivalence. Stage E descriptive intervals use a different bootstrap namespace from the primary Figure 4 analysis.

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
