# Unified paper figures R1: English captions

## Figure 1

Study overview linking the fixed ViT pathway, Value-path attribution, the matched patch-token baseline, and fine-grained intervention evaluation.

## Figure 2

Head-direction Value-path construction. NNS produces a signed input-space map through fixed clean attention, while the baseline retains patch-token contributions under the same image, target, and intervention budget.

## Figure 3

Evaluation protocol from attribution ranking at fine 4x4-cell or native 14x14-patch scale to matched-fraction insertion/deletion and image-level aggregation.

## Figure 4

Primary fine-scale fidelity result. NNS-minus-matched delta FID was positive for all 500 images; the paired median was 2.308 with a 95% bootstrap interval of [2.111, 2.445].

## Figure 5

Median insertion and deletion trajectories for NNS and the matched patch baseline. Bands show pointwise bootstrap intervals.

## Figure 6

Spatial-scale and upsampling sensitivity, showing fine 4x4 versus native 14x14 delta FID and median/IQR summaries for three upsampling evidence classes.

## Figure 7

Median delta FID and image-bootstrap intervals for four frozen layer/head coordinates in the overall, correct, and misclassified strata.

## Figure 8

Frozen median-effect, weak-effect, and misclassified examples. Attribution panels are independently scaled and convey spatial shape only.

## Figure 9

Head-averaged positive and negative spatial support across four depth bands for one frozen sample. Absolute magnitude is not compared across bands.

## Figure 10

Spatial comparison of NNS, pooled NNS, Grad-CAM, full-model IG, and rollout on the first four positions of the frozen formal cohort. Attribution panels use P99 display scaling.

## Figure 11

Quantitative comparison across 50 images: method-level fidelity distributions, registered NNS-minus-comparator contrasts, and median deletion trajectories.

## Figure 12

Post-hoc exploratory multiscale mechanism analysis with a fixed flip-route example, geometric agreement, ROAD fidelity, and NNS-v2 within/between SSE decomposition.

## Figure 13

Block-level class-logit complete-cut allocation: 12-block attention and MLP branch trajectories and the B4/B5 deep-minus-shallow contrasts.

## Figure 14

Post-hoc exploratory depth extension showing endpoint estimates, depth profiles, matched occlusion responses, and the image-level D1/D2 relation.
