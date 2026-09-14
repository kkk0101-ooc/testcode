# S4-R2 prospective initialization amendment

This amendment is frozen after S4-R1 initialization failed and before any formal S4 JPEG or mask member access. S4-R1 wrote only pre-access schedule and registry metadata before a Python `NameError` caused by a lowercase JSON boolean in the initializer. No formal member was read, no image was decoded, no model forward was run, and no NNS or occlusion result exists in S4-R1.

S4-R2 changes the initializer's boolean literal from `false` to Python `False` and moves all production paths to fresh roots `evidence/nns_vit_patch_depth/s4_r2_20260915/` and `artifacts/nns_vit_patch_depth/s4_r2_20260915/`. Scientific definitions, cohort, order, target, numerical contract, P1/P2 registries, unit topology, seeds, operators, typed failures, and retention remain unchanged. S4 and S4-R1 roots and their partial metadata are immutable history and are not used as production evidence.
