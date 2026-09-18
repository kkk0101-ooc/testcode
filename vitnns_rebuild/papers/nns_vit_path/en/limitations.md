# 6. Limitations

First, the primary intervention result has a specific target scope. Stage D explains the fixed-clean-attention Value path at block 0/head 9 of a frozen ViT-B/16 and compares a target-matched patch-token baseline at the 4x4-cell endpoint. It is not a complete class explanation, object-localization metric, or test of agreement with human vision. Native-patch and alternative-coordinate results show that the effect is not a uniform property of the entire ViT.

Second, spatial stability remains unresolved. The C10 translation and center-crop/resize gates fail, as do the broad Stage X stability criterion and noninferiority to Grad-CAM. Increased association after multiscale aggregation is a post-hoc exploratory observation; it neither substitutes for fine-scale stability nor changes the broad stability decision.

Third, the compared explainers do not have identical targets. Conditional NNS, full-model IG, Grad-CAM, rollout, and raw CLS attention differ in class conditioning or computational path. Their frozen fidelity contrasts remain informative, but they do not define a target-independent total order. Non-significant NNS-full IG and NNS-raw attention contrasts do not establish equivalence.

Fourth, the real-module demonstrations are based on one image. Figures 28-35 verify coordinates, completeness, residual additivity, and classifier-target differences for one frozen input. The fixed channels do not establish population semantics, and four classifier outputs do not establish a general spatial mechanism across classes.

Fifth, several extensions are descriptive or post-hoc exploratory. Stage E follows access to D10 outcomes and has `MIXED` scientific validity at E4. Stage X2-MECH and Stage S are not confirmatory. The final Stage S P2 threshold was specified after outcome access, and the original rule did not pass. These results motivate hypotheses rather than primary causal or confirmatory statements.

Sixth, the novelty claim still requires a literature-level technical audit. The learning-image idea originates with Dou et al., path integration with IG, and internal component calculations are directly related to conductance and path attribution. This study validates a ViT structural mapping and its empirical uses, but a line-by-line operational comparison and final contribution statement remain necessary before submission.

Finally, the experiments focus on one frozen ViT-B/16 and ImageNet-1k validation images. Independent studies are needed to assess other model scales, training objectives, patch sizes, references, data domains, and visual tasks.
