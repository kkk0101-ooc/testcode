# R9 · 31 张论文图

[画廊](gallery.html) · [图表索引](figure_index.md) · [论文大纲](../../docs/论文大纲_NNS_ViT_R9_20260917.md)

## Figure 1

NNS scans internal ViT responses. One image is embedded as patch tokens and processed by a trained ViT. Measured positive input-contribution maps are arranged by block (B0, B5, B11) and head (H0, H5, H11). Each panel targets that coordinate's CLS head-direction Value-path scalar. First, middle and last indices provide a display subset, without performance ranking. The input grid marks patch locations; the recorded targets are CLS queries, not separate patch-query attributions. The previously frozen order-44 median-effect image is used throughout. Figures 2 and 8–9 explain the mapping and the all-head summaries.

## Figure 2

A real response learning image. For the fixed input ILSVRC2012_val_00007922.JPEG, s is the sum of channel 0 over the 196 patch-token outputs of block-0 attention, including output projection. The full 224×224 signed map is input IG relative to normalized-zero gray, with all attention derivatives active. RGB contributions are summed; the raw map sum reconstructs the response difference. A single map encodes both signs, using max-absolute signed-log(kappa=9), nearest display and no smoothing. Channel 0 was fixed by index before computation. Figures 3–7 retain the distinct preregistered B0/H9 conditional Value-path target.

## Figure 3

Ranking and fidelity at a shared spatial scale. A 224×224 image has 196 native 16×16-pixel patches, each divided into 16 fine 4×4-pixel cells, giving 3,136 cells. NNS sums signed input contributions within each cell and ranks their target-oriented positive parts. The token baseline uniformly expands c_j before the same ranking rule, giving ties within each patch while preserving differences between patches. The local 1–16 ranks are illustrative; the actual experiment ranks all cells and uses five paired tie permutations for every tie group. Insertion restores high-ranked regions from the gray reference; deletion replaces them in the clean image. Both use the same area budget and evaluate the original B0/H9 CLS head-direction scalar with its clean attention fixed. Enlarged mask blocks illustrate the operation, not actual fine-cell masks or outcomes. The original protocol uses 21 fractions and AUC against actual intervention fractions, with responses unclipped. Five AUC repeats are averaged within each image before computing F=½(AUC_ins−AUC_del) and ΔFID=F_NNS−F_token. Images are the independent units for the median and paired bootstrap 95% CI.

## Figure 4

Primary fine-scale effect. (a) Histogram of all 500 image-level ΔFID values. The orange line marks the median and the orange region is its formal paired-image bootstrap 95% CI. (b) ECDF of the same data, with the median marked by a dot. The shaded region remains a median CI, not a simultaneous ECDF band. All 500 differences are positive; median 2.3081996222, 95% CI [2.1110806126, 2.4452354118].

## Figure 5

Median insertion and deletion trajectories for NNS and the target-matched patch-token baseline with uniform within-patch allocation. Bands show pointwise bootstrap intervals.

## Figure 6

Spatial-scale and upsampling sensitivity. (a) Fine-cell and native-patch ΔFID distributions. (b,c) Every dot is an image-level insertion/deletion AUC, with horizontal jitter for visibility. Boxes show the 25th–75th percentiles and inner lines the medians; all 500 observations per evidence class, including extrema, are retained. IQRs are not confidence intervals. Uniform and mass-preserving nearest allocation form one exactly equivalent evidence class; bilinear and bicubic are descriptive sensitivities.

## Figure 7

Figure 7. Median NNS-minus-patch-baseline ΔFID at four frozen layer/head coordinates. Gray, blue and orange bars identify the overall, correct and misclassified strata; whiskers are the original Stage E descriptive paired-image-bootstrap 95% CIs. The main axis starts at zero and is shared across all coordinates. The inset repeats the last three coordinates on an explicitly labeled linear 0–0.10 scale, also starting at zero. These repeated marks are not additional observations. Expected image counts are 500/418/82; block 11/head 2 has 498/417/81 valid images, with both typed failures retained in the denominators. Other coordinates have complete coverage. Coordinates were frozen in Stage C. Strata overlap and are not independent experiments; similar directions do not establish equivalence. Stage E descriptive intervals use a different bootstrap namespace from the primary Figure 4 analysis.

## Figure 8

All-head spatial support for the frozen median-effect image, order 44 (ILSVRC2012_val_00002287.JPEG). The original image is followed by NNS positive, target-matched token-baseline positive, NNS negative-magnitude and baseline negative-magnitude summaries. At each block/head coordinate, both methods share the maximum absolute value L; each component uses log1p(999m/L)/log(1000), the original F3H linear enlargement, and an equal average over all 144 display maps. These arrays reproduce F3H Image 19. Median-effect selection refers to the original B0/H9 primary endpoint, not to the fidelity of the consensus. The consensus describes spatial support, not class attribution. The original image export is included unchanged in reference_gallery.

## Figure 9

Spatial support across four depth bands for the same frozen image. Columns show B0–2, B3–5, B6–8 and B9–11; rows show NNS positive, token-baseline positive, NNS negative magnitude and baseline negative magnitude. Each band averages all 36 head display maps under the original F3H pair-shared scales, reproducing Image 18. Colors describe within-coordinate display support rather than raw cross-layer magnitude. The bands are indexed by depth without assigning untested semantic stages.

## Figure 10

Explanation views of the same inputs. All five images from F3C visual primary page 1 are retained in frozen formal-position order. Columns show the original, native positive NNS, 4×4 max-pooled/bilinear NNS, Grad-CAM, positive full-model IG, equally pooled full IG, rollout and raw CLS attention. NNS here targets a conditional class logit, unlike the single-head targets in Figures 1, 8 and 9. Each panel uses its own P99 scale; pooling is display-only. Figure 11 reports quantitative comparisons. All 50 inputs and signed/unclipped companion pages are included in the gallery.

## Figure 11

Method comparison in 50 frozen images. (a) Image-level mean-over-repeat fidelity scores, with IQR boxes and median lines. (b) Registered paired mean NNS-minus-comparator contrasts: filled bars run from zero to the estimate, diamonds mark means, and the horizontal bounds of outlined rectangles are the original bootstrap 95% CIs. Rectangle height has no statistical meaning. NNS-minus-Grad-CAM is negative, NNS-minus-rollout is positive, and intervals for full IG and raw CLS cross zero. No density is inferred from an interval. The original primary unadjusted and secondary Holm testing scopes remain unchanged. (c) Median deletion trajectories.

## Figure 12

Post-hoc exploratory multiscale mechanism analysis with a fixed flip-route example, geometric agreement, ROAD fidelity, and NNS-v2 within/between SSE decomposition.

## Figure 13

Class-logit complete-cut branch allocation. (a) Twelve-block attention and MLP absolute-mass allocation trajectories with their original blockwise 95% intervals. (b) B4/B5 deep-minus-shallow contrasts: bars encode signed estimates, diamonds mark estimates, and outlined rectangles encode the original 95% CI endpoints. Colors consistently identify attention and MLP. The two shares refer to distinct residual cuts and do not form a single two-part probability distribution. These branch findings do not imply support for other Y4 claims.

## Figure 14

Post-hoc exploratory depth extension. (a) Endpoint-specific estimates, ordinary BCa 95% CIs and valid/total image counts are tabulated without imposing a common effect axis. P1 and D1 have 235/240 valid images, P2 224/240, and D2 3424/3482; their NA counts are 5, 5, 16 and 58, respectively, with no failed images. Ordinary intervals describe precision and do not replace adjusted decision rules. (b) NNS alignment and probe gain profiles show each measure across depth; their magnitudes are not interchangeable. (c) Matched-control-corrected logit drops at three fixed patch budgets. (d) D1/D2 association in 233 valid paired images, Pearson r approximately 0.002. All S results are post-hoc exploratory, with endpoints and the P2 threshold formed after outcome access, and do not establish confirmatory or causal mechanism claims.

## Figure 15

One internal response is paired with one map in input coordinates. The illustrated linear example uses input difference [1,2;3,4], unit changes 3 and 2, and output change 7. All four maps retain the same input locations. This is an analytic example, not a measured ViT.

## Figure 16

Input-position initialization. The reference is [1,1;1,1] and the input is [2,3;4,5]. Their difference is split into four coordinate images whose sum restores the difference. This is an equivalent mathematical representation for path attribution; the implementation does not forward each sparse seed independently through the nonlinear model.

## Figure 17

Generation in an analytic linear network. For a=x1+x2, b=2x3−x4 and f=2a+0.5b, the learning images of a and b are [1,2;0,0] and [0,0;6,−4]. Their weighted combination is [2,4;3,−2], summing to the output change 7. For nonlinear ViT targets the corresponding coefficients are integrated along the input path.

## Figure 18

ViT operator adaptation. A patch channel with weights [1,0,0,0.5] produces the local contribution map [1,0;0,2] from input differences [1,2;3,4]. Position and bias terms remain in the forward model. Attention requires both dA and dV terms on the full path; the fixed-attention condition suppresses dA. Head and identity components jointly form a complete cut. The drawing is schematic.

## Figure 19

Positive class-component learning images (top) and attention-output token RMS (bottom) for blocks 0,2,4,6,8,11. The first frozen Y2 input and class 235 are held fixed. Top maps retain the original 56×56 input cells; bottom responses are 14×14 patch-token indices. Each nonnegative panel uses P99 scaling with max fallback and nearest display. The signed and negative companion views are included in the supplement. Unlike Figures 28–31, these are downstream class-component maps, not module-own-response maps.

## Figure 20

Class-logit component learning images for heads 0,5,11 at blocks 0,5,11, on the same image and target as Figure 19. Positive signed input-cell contributions are shown with per-panel P99 scaling. These fixed index subsets do not rank heads by appearance or performance; every head and both signs are included in the attached 12 block pages. Head numbers in different blocks identify different parameter sets.

## Figure 21

Component reconstruction at block 5 for the same class target. The raw residual map plus the sum of all twelve raw head maps reconstructs the total input-cell IG map to numerical precision. Addition is performed before coloring. All panels share sign(m) log1p(9 abs(m)/L)/log(10), where L is the maximum absolute raw value across the four panels. This checks complete-cut accounting, not a new fidelity or causal test.

## Figure 22

Analytic two-token attention example with input [1,3], zero reference, scalar Q/K/V weights 1 and head dimension 1. The input derivative includes A dV and (dA)V. Integrated full-path attribution differs from attribution in the model with clean attention fixed. This is an analytic illustration.

## Figure 23

Analytic downstream-target example: r=x1, b=2x1−x2, z=r+b and F=z² at input [2,1] relative to zero. Residual and branch component maps are [10,0] and [20,−5], summing to [30,−5] and output difference 25. These include the downstream derivative of F, unlike branch-own-response maps.

## Figure 24

One analytic chain pairs normal forward values with response learning images: input [1,1;3,3], shared linear patch projection, full softmax attention, and residual addition. These four diagrams are inherited byte-for-byte. The chain omits position, bias, LayerNorm, MLP and classifier for explanation; real trained-ViT outputs appear in Figures 28–31.

## Figure 25

Four coordinate seed images produce two token images with shared weights [0.5,0.5]. Their sums equal token responses 1 and 3 while retaining input coordinates. These four diagrams are inherited byte-for-byte. The chain omits position, bias, LayerNorm, MLP and classifier for explanation; real trained-ViT outputs appear in Figures 28–31.

## Figure 26

Forward values mix V using A. For this linear patch-input example, response images combine using path-averaged full Jacobian coefficients C, including Q/K/V derivatives; C is not clean attention. This endpoint combination is not a general propagation rule through nonlinear upstream layers. These four diagrams are inherited byte-for-byte. The chain omits position, bias, LayerNorm, MLP and classifier for explanation; real trained-ViT outputs appear in Figures 28–31.

## Figure 27

For a common input path and reference, residual response images add at each input coordinate: L(z)=L(h)+L(o). Raw values are added before color mapping. These four diagrams are inherited byte-for-byte. The chain omits position, bias, LayerNorm, MLP and classifier for explanation; real trained-ViT outputs appear in Figures 28–31.

## Figure 28

Channel 0 is followed through patch embedding, B0 attention output and the first residual add before the MLP. The fixed first Y2 input, ILSVRC2012_val_00007922.JPEG, uses normalized-zero gray as reference. Each target sums one channel over the 196 patch-token module outputs; CLS participates in forward attention but is excluded from this spatial output sum. Top: positive parts of RGB-summed response IG in 224×224 input coordinates, with per-panel P99 scaling (max fallback). Bottom: original signed output-channel values at 14×14 token indices, with per-panel max-absolute signed-log(kappa=9). All displays use nearest interpolation. Complete signed companions and raw RGB arrays are retained. This is one-input module visualization, not a population depth or feature-map correlation test.

## Figure 29

Patch embedding is shown at fixed channel indices 0,255,511,767. The fixed first Y2 input, ILSVRC2012_val_00007922.JPEG, uses normalized-zero gray as reference. Each target sums one channel over the 196 patch-token module outputs; CLS participates in forward attention but is excluded from this spatial output sum. Top: positive parts of RGB-summed response IG in 224×224 input coordinates, with per-panel P99 scaling (max fallback). Bottom: original signed output-channel values at 14×14 token indices, with per-panel max-absolute signed-log(kappa=9). All displays use nearest interpolation. Complete signed companions and raw RGB arrays are retained. This is one-input module visualization, not a population depth or feature-map correlation test.

## Figure 30

Attention · B0 is shown at fixed channel indices 0,255,511,767. The fixed first Y2 input, ILSVRC2012_val_00007922.JPEG, uses normalized-zero gray as reference. Each target sums one channel over the 196 patch-token module outputs; CLS participates in forward attention but is excluded from this spatial output sum. Top: positive parts of RGB-summed response IG in 224×224 input coordinates, with per-panel P99 scaling (max fallback). Bottom: original signed output-channel values at 14×14 token indices, with per-panel max-absolute signed-log(kappa=9). All displays use nearest interpolation. Complete signed companions and raw RGB arrays are retained. This is one-input module visualization, not a population depth or feature-map correlation test.

## Figure 31

Residual add · B0 is shown at fixed channel indices 0,255,511,767. The fixed first Y2 input, ILSVRC2012_val_00007922.JPEG, uses normalized-zero gray as reference. Each target sums one channel over the 196 patch-token module outputs; CLS participates in forward attention but is excluded from this spatial output sum. Top: positive parts of RGB-summed response IG in 224×224 input coordinates, with per-panel P99 scaling (max fallback). Bottom: original signed output-channel values at 14×14 token indices, with per-panel max-absolute signed-log(kappa=9). All displays use nearest interpolation. Complete signed companions and raw RGB arrays are retained. This is one-input module visualization, not a population depth or feature-map correlation test.
