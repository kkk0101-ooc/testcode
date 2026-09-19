# Endpoint M4 Review R2 defensive-writing audit

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Opening estimand | M4 estimates target-conditioned local input-Jacobian path accounting for the frozen ground-truth logit. | CLEAN | Minor | Necessary caveat | The target and cotangent scope are stated positively. | The estimand is clear before results. | Keep. | Unchanged. |
| Share interpretation | Routing and identity shares are relative absolute-Endpoint path-norm scales, and JVP error is a normalized RMS directional discrepancy. | CLEAN | Minor | Necessary caveat | This defines the quantities rather than listing prohibited claims. | Prevents percentage language without sounding apologetic. | Keep once. | Unchanged. |
| JVP interpretation | They do not mean that Value-only omitted 95% or that routing contributed 95%. | MIXED D2/D12 | Minor | Necessary | The concrete 0.95 result is easy to misread as a percentage. | One explicit correction is useful; repetition would be defensive. | Keep only adjacent to the JVP table. | The discrepancy is comparable in norm with the full response; it is not an additive contribution fraction. |
| Saturation causality | The association does not establish that saturation causes routing changes. | NECESSARY_CAVEAT D12 | Minor | Necessary | The analysis is correlational. | A single causal boundary is appropriate. | Keep once. | The result is a within-image, within-layer head association. |
| Residual rewrite | The attention-residual rewrite description is supported jointly by relative norm and spatial-distribution evidence. | CLEAN | Minor | Necessary caveat follows | Rewrite is tied to a measured spatial distance. | The stronger term now has direct evidence. | Keep with target-conditioned scope. | Unchanged. |
| M3–M4 bridge | The result is a model-level depth-trajectory bridge, not evidence that Q/K routing causes the M3 broadening. | NECESSARY_CAVEAT D12 | Minor | Necessary | Cohorts and scalar targets differ. | Prevents an invalid unit-level causal merge. | Keep once. | The two model-level depth trajectories are directionally compatible; target and cohort differences preclude unit-level attribution. |

### Paper-level defensive-writing score

2/10. The revision leads with estimands and measurements, then states one concise boundary for each interpretive layer.

### Highest-priority removals

1. Do not repeat the not-95-percent warning outside the JVP subsection and limitations.
2. Avoid reviewer-facing phrases in manuscript prose.
3. Keep the post-hoc label in methods and tables rather than prepending it to every result.
4. Avoid universal claims that attention sharpness is never a routing proxy.

### Necessary caveats to preserve

- M4 components are conditioned on the frozen ground-truth logit downstream cotangent.
- Norm shares and normalized JVP errors are not additive contribution percentages.
- Partial-rank and phase analyses remain correlational post-hoc sensitivities.
- The raw-map bridge uses 16 class-logit maps per model and does not share the M3 target/cohort.
- MLP path-cancellation trends are significantly opposite across the two tested models.

### Repeated phrase audit

Reviewer-facing prebuttals: 0. Repeated non-claim disclaimers in the revised M4 section: 0. Promotional adjectives: 0. AI-style summary constructions: 0.
