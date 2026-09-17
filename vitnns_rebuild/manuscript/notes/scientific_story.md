# Scientific story

## Central question

How can a Vision Transformer internal response or class output be represented in the original image coordinates, and what can such a representation support beyond visual inspection?

## Story in one paragraph

NNS introduced the learning-image idea for FCNs and CNNs: associate an internal unit with an image-shaped account of where its response came from. This study adapts that view to ViT computation, where patch tokens, input-dependent attention, residual branches, MLP branches, and the classifier create several distinct attribution targets. A path-integrated learning image maps a specified response change back to the input. Module-own-response maps show where an internal response comes from; a fixed-clean-attention Value-path functional isolates a prescribed internal attention target; complete-cut component maps distribute a class-logit attribution over heads and branches. The main 500-image intervention study finds that the input-space ranking for block 0, head 9 has higher fine-4x4-cell fidelity than a target-matched patch-token baseline with uniform within-patch allocation. That advantage is scale- and coordinate-dependent, disappears at native patch scale, and does not translate into superiority over every class-output explainer. Complete-cut accounting further shows that attention and MLP branch shares change differently with depth.

## Main findings

1. **Numerical correspondence.** Learning images satisfy target completeness within the frozen numerical tolerances, and raw maps obey residual additivity before display transforms.
2. **Primary intervention result.** In the Stage D cohort, all 500 image-level fine-cell differences were positive; median delta fidelity was 2.3082 with paired-median bootstrap 95% CI [2.1111, 2.4452].
3. **Scale and coordinate dependence.** At native 14x14 patch scale the median difference was 0, whereas the three secondary coordinates had valid-only medians 0.0734, 0.0389, and 0.0562.
4. **Comparator boundary.** In an independent 50-image class-output comparison, NNS was below ViT Grad-CAM, above attention rollout, and not significantly different from full-model IG or raw CLS attention under the registered tests.
5. **Component accounting.** Complete cuts reconstruct the same full-model input attribution while indexing it by internal component. In a 100-image analysis, the attention share at the first residual increased with depth and the MLP share at the second residual decreased.
6. **Negative and exploratory results.** The earlier C10 spatial gates failed; the broad Stage X stability claim did not pass; coarse aggregation increased geometric association only in a post-hoc exploratory analysis. Stage S depth findings are also post-hoc exploratory.

## Logical order

1. Define a learning image for a scalar response relative to a reference.
2. Separate the three attribution objects used in the paper.
3. Show their correspondence to patch embedding, attention, residual, MLP, and classifier computation.
4. Validate the main ranking by intervention on the frozen 500-image cohort.
5. Test the scope through scale, coordinates, prediction strata, and independent explainers.
6. Use complete cuts to study component allocation across depth.
7. Report stability failures and exploratory depth extensions as boundaries, not as repairs of the primary result.

## Potential contradictions resolved in the draft

- The Stage D positive result explains a fixed internal Value-path target; the 50-image comparison explains a conditional class output. Their rankings answer different questions and are reported separately.
- Stage D and Stage Y use different targets. Stage Y's failure to support a class-logit within-patch claim does not erase Stage D, and Stage D does not substitute for Stage Y.
- Complete-cut NNS-v2 sums to the same full-model IG under the same contract. The added analytical object is the internal component index, not a different total attribution.
- Figures 28-35 validate computation on one frozen image; they do not estimate population trends.

## Open manuscript decisions

- Target journal, author list, affiliations, corresponding author, acknowledgments, and data/code availability wording.
- Final contribution wording after a complete comparison with conductance and path-attribution literature.
- Final main-text versus supplement allocation for Figures 15-35.
- Formal BibTeX/DOI verification for references that are currently available only as arXiv or local converted records.
