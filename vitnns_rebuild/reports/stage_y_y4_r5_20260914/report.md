# Stage Y4-R5 scientific decision report

Workflow: **PASS**. CPU-only source-recomputed analysis of the frozen ViT-B/16 Stage Y evidence.

|ID|n|estimate|95% CI|family|Holm adjusted p|reject|outcome|
|---|---:|---:|---:|---|---:|---|---|
|A1|97|-0.000244890111|[-0.000389049402, 0.000269287622]|A|1|False|NOT_SUPPORTED|
|B1|100|0.00600944478|[-0.0150648737, 0.018117191]|B|1|False|NOT_SUPPORTED|
|B2|100|0.591336111|[0.451532199, 0.662505701]|B|0.702392976|False|INCONCLUSIVE|
|B3|0|||B|1|False|INVALID|
|B4|100|0.139108611|[0.128220357, 0.153300736]|B|0.0001399986|True|SUPPORTED|
|B5|100|-0.249226373|[-0.257212462, -0.23485993]|B|0.0001399986|True|SUPPORTED|
|B6|97|-0.0236207239|[-0.0613628743, -0.00526512242]|B|1|False|NOT_SUPPORTED|
|B7|97|0.00627080036|[0.0033181038, 0.00924080834]|B|1|False|NOT_SUPPORTED|
|C1|20|0.0316714491|[0.0266673674, 0.0358656754]|C|1|False|NOT_SUPPORTED|
|C2|20|0.00532703445|[0.00480751254, 0.00635831706]|C|1|False|NOT_SUPPORTED|
|C3|0|||C|1|False|INVALID|
|C4|20|0.0150733363|[0.0144040519, 0.0159806318]|C|1|False|NOT_SUPPORTED|

## Independent claim decisions

- Y-CLM-INTRAPATCH: NOT_SUPPORTED
- Y-CLM-DEPTH-SPATIAL: NOT_SUPPORTED
- Y-CLM-HEAD-ORGANIZATION: INCONCLUSIVE
- Y-CLM-PATCH-MIXING: INVALID
- Y-CLM-BRANCH-ALLOCATION: SUPPORTED
- Y-CLM-PERTURBATION-MODULATION: INVALID

Prediction flips (provenance/descriptive only): 2.

All wording is limited to the frozen ViT-B/16, Stage Y cohort/subset, fixed clean predicted-class raw-logit target, cuts, metrics, thresholds and endpoints.

No Y5, F3-Final, Stage G, paper writing, image decode, model load or GPU execution occurred.

Next authorization: `STAGE_Y5_RESULTS_TO_PAPER_HANDOFF_ONLY`.
