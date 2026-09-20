# Revision-v2 defensive-writing audit

Applied academic-defensive-writing-auditor to both current manuscripts after the scientific rewrite. The audit preserves evidence boundaries while preferring claim, evidence, and a concise boundary.

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Abstract | NNS-ViT provides a unified view and characterizes meaning | D10/D11 | Major | No | Promotional summary substitutes for the technical result | Requests a sharper novelty statement | State the mathematical identity, graph extension, observations, and intervention result | The extension is a unified representation, closure test, and empirical analysis of actual ViT graph edges, not a new base attribution quantity |
| Introduction | The learning image is numerically Gradient × Input | D14 | Major | Yes, but buried | Novelty boundary appears as an afterthought | May read the earlier contribution bullets as relabeling | Put equivalence before the contribution list and define the actual delta positively | NNS is unified through the Jacobian chain rule; the extension decomposes complete local dependence over actual ViT edges |
| Contribution list | Three broad contributions | D10 | Minor | No | Mathematical clarification, implementation, and empirical scope are conflated | Hard to identify the technical delta | Use four concrete contributions | Equivalence conditions; heterogeneous targets; valid-cut path decomposition; bounded empirical analysis |
| Method | Path shares are not causal percentages | D2 | Minor | Yes | Repeated negation interrupts the construction | Correct but defensive if repeated | Define the positive object first and scope once | Path-norm ratios measure relative local differential scale; finite effects are evaluated separately |
| Results | Original all-family criterion criterion is unmet | D9 | Major | Yes | Machine-oriented and duplicated wording | Looks like an internal report | State the result directly | Eleven of twelve families meet the composite condition; the joint all-family condition is unmet |
| Discussion | Long list of missing controls | D3/D5 | Major | Yes | Caveats were stacked as an anticipated rebuttal | Makes the central result look unstable | Group untested generalization axes in one paragraph | The empirical scope covers two checkpoints, three initialization seeds, registered targets, regions, references, and blocks |
| Conclusion | Training controls and matched-target studies offer a next step | D11 | Minor | No | Generic future-work ending weakens the contribution | Leaves the paper on an unfinished note | End with the demonstrated theoretical and empirical contribution | NNS-ViT extends the theoretical scope and applicable objects of NNS through unified ViT structural-path implementation and validation |

### Paper-level defensive-writing score

2/10 after revision. The manuscripts state the supported object and result first, retain one material boundary per claim, and avoid reviewer-facing prebuttals.

### Highest-priority removals

1. Removed claims that framed Endpoint as a new base input-attribution method.
2. Removed repeated non-claim language around every path definition.
3. Replaced pipeline labels and doubled wording such as criterion criterion with scientific statements.
4. Moved detailed figure-selection and display defenses to the supplement.
5. Consolidated untested controls into one discussion paragraph.

### Necessary caveats to preserve

- Scalar NNS equivalence to Input × Gradient requires exact current-state local derivatives and no extra redistribution.
- Input-weighted local dependence is coordinate-origin dependent and generally lacks function-value completeness.
- Structural closure does not establish finite causal utility.
- Routing/value and identity/branch shares are relative differential-map norms, not forward-value proportions.
- Population claims are limited to two fixed checkpoints and registered target samples.
- The regional experiment does not include a multi-amplitude replacement sweep.
- Incorrect-classification subsets contain 8/14 images; alternative trained checkpoints and independent channel seeds remain untested.

### Repeated phrase audit

The current main texts contain zero instances of: we do not claim, we emphasize, it should be noted, to avoid misunderstanding, taken together, 我们不声称, 需要强调的是, 应当指出, and 为避免误解. The final validator enforces this list.
