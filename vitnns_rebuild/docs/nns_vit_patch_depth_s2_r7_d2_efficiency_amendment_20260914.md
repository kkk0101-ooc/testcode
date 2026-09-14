# S2-R7 prospective D2 efficiency redesign

S2-R6 is immutable and remains `BLOCKED_SCIENTIFIC_VALIDITY`: two confirmatory images per class provide only 0.6934 conservative D2 power at all 872 classes. This amendment does not lower the 0.90 target, change the D2 threshold or design alternative, remove D2 from the Holm family, or treat repeated fits as independent observations.

The redesign uses every S0-R4-QC-eligible public-validation record belonging to the 872 frozen classes. These records were already metadata-audited and remain outcome-sealed. P1, P2, and D1 keep the original two-image-per-class confirmatory cohort. Only D2 receives the expanded test cohort. The D2 estimand is an equal-weight mean over the 872 fixed classes, with equal image weights inside each class.

The existing six probe-train images remain the only fitting data and the existing two probe-validation images remain the only alpha-selection data. Three deterministic train-only folds test stability and leakage; they do not enlarge the confirmatory denominator. The final ridge probe is fitted on all six train images after alpha selection.

After this amendment is committed, S1 retained pilot arrays may be used to compute two D2 values per complete pilot class by reversing the train/validation direction. A one-way random-effects ICC is computed on the 11 class pairs. A fixed 100,000-replicate class-pair bootstrap gives a one-sided 95% upper bound, clipped to `[0,1]`. Cluster-aware power uses that upper bound and the immutable S2-R4 D2 scale.

No formal image, mask, feature, logit, map, endpoint, model, CUDA operation, probe fit, or occlusion is accessed in S2-R7. PASS can authorize only S3 representation/probe training.
