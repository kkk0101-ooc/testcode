# Revised main-figure captions

**Figure 1.** The observed input, its ground-truth class logit, and two complementary learning images. Signed maps sum RGB channels; absolute maps sum channelwise absolute values. Both signs remain in the analysis. The frozen display sample is class 297 (sloth bear).

**Figure 2.** Heterogeneous scalar targets and attention/residual differential cuts. Head edges precede output projection, whose derivative is included in the cotangent; residual identity uses the output-edge cotangent.

**Figure 3.** Original-NNS numerical equivalence across five compatible network families, structural closure, negative controls, and independent direction checks. Each left-panel point is the maximum relative-L2 error for that network/precision; validation counts are not independent statistical sample sizes.

**Figure 4.** Depth trajectories of entropy, locality, and distance for the frozen stratified token-channel sample. Blue/orange denote ViT-B/16 and DeiT-S/16; bands are image-bootstrap 95% CIs over 100 images.

**Figure 5.** Population trajectories of CLS-minus-patch entropy and net/absolute mass appear above a high-map-cancellation unit selected by frozen metrics. Signed and absolute views are paired, with original RGB masses and norms annotated below.

**Figure 6.** Routing scale, Value-only Endpoint error, and JVP discrepancy over 400 images; bands are image-bootstrap 95% CIs. Routing/value/full maps of one head share a scale within the signed row and within the absolute row, with original mass, norm, and signed-sum annotations. The example illustrates decomposition rather than replacing population evidence.

**Figure 7.** Attention/MLP identity scales and output–identity patch TV, above shared-scale identity/branch/output maps from a high-path-cancellation case. TV curves describe the frozen cohort; late-minus-early inference comes from a separate post-hoc sensitivity analysis.

**Figure 8.** A shared-image cross-model depth comparison selected by frozen metrics. Early/late maps share signed and absolute scales within each model. Independently registered channels do not establish semantic matching across layers or models.

**Figure 9.** Original C4a family estimates and image-bootstrap 95% CIs. The dashed 0.10 line is a point-estimate threshold; a square marks the unsupported family. The right panel reports Endpoint/Taylor Spearman for the supplementary frozen within-block head comparison, which does not replace C4a.

**Figure 10.** Median absolute-effect AUC in four registered settings, including an oracle over evaluated candidates. Below are signed/absolute Endpoint and regional Taylor for the same sloth-bear image, DeiT-S block-5 head 0, and normalized-zero reference. Taylor cells represent 16 region-displacement scores and are not comparable to pixel Endpoint color values.
