# Revision v8 defensive-writing audit

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Abstract | Internal labels H4, B11−B0, all-family, CPU float64, wrong-slot and fan-out | D9 | Major | No | Execution vocabulary displaced the research question and main findings | The abstract reads like a response log | Replace | State scientific objects, findings and evidence boundary without stage labels |
| Abstract | Several cohort sizes in one result sentence | D3 | Minor | Partly | Dense procedural detail obscured which evidence supported which claim | The contribution is harder to identify quickly | Condense | Describe main analysis and independent robustness checks; retain exact sizes in Methods |
| Introduction question list | RQ1--RQ4 labels | D14 | Minor | No | Engineering-style identifiers add no meaning to the four questions | The paper appears pipeline-driven | Replace | Use Mathematical object, Structural extension, Spatial distribution, and Finite-replacement utility |
| Section 4.1 | One target registry appears to contain targets, differential terms and interventions | D14 | Major | Yes | A single label obscures three different mathematical object classes | Counts and applicability can be misread | Rewrite | Separate scalar-target, local-differential-decomposition and finite-replacement registries |
| Entropy sensitivity | Spatial broadening is robust to low mass | D10 | Major | Yes | Robust would conceal the observed strong mass--entropy dependence | The claim may look stronger than the diagnostic | Replace | Direction persists after trimming, while entropy remains quality-dependent |
| H4 sensitivity | Matched aggregation proves path maps are worse | D13 | Major | Yes | A post-hoc sensitivity cannot establish universal superiority | The claim overgeneralizes beyond the fixed task | Scope | All four tested checkpoints retain a negative median increment under the matched definition |
| Discussion | Repeated statements that structural resolution does not imply utility | D2 | Minor | Yes, once | Repetition can sound like pre-emptive defense | The negative result is already clear | Keep once per abstract/discussion/conclusion | Structural resolution and finite-intervention prediction are distinct outcomes |

### Paper-level defensive-writing score

2/10. The revised abstract leads with the problem, representation, evidence and main result. Necessary boundaries remain concentrated around fixed graphs, normalized-input coordinates and prescribed finite replacements.

### Highest-priority removals

1. Removed H4, B11−B0, all-family, CPU and implementation-fault labels from the abstract.
2. Replaced RQ1--RQ4 labels with descriptive research-question names and section headings.
3. Replaced an unqualified robustness impression with the joint result that direction persists while entropy depends on total mass.
4. Separated scalar targets, local differential components and intervention variables.

### Necessary caveats to preserve

- Endpoint is same-target Gradient × Input under the stated derivative rules, not a new base attribution formula.
- Architecture certification applies to the registry-bound standard unfused pre-LN ViT graph.
- Entropy direction persists under tested mass sensitivities, but entropy values are mass-dependent.
- Matched H4 is post hoc and does not replace the registered negative estimate.
- Finite replacement results are conditional on the stated target, reference and intervention contract.

### Repeated phrase audit

- The abstract contains no H4, B11−B0, all-family, CPU float64, wrong-slot, fan-out, or RQ1--RQ4 labels.
- The main text retains technical labels only where their exact estimands or provenance are needed.
- Result boundaries are stated once near each relevant claim rather than as reviewer-facing prebuttals.

Status: PASS_DEFENSIVE_WRITING_AUDIT
