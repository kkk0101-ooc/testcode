# Endpoint Scanner full-draft R3 defensive-writing audit

Scope: `en/full_draft.md` and `zh/full_draft.md` after Stage M and
`ENDPOINT-ALL-FIGURES-R2` integration.

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Introduction, NNS identity | “We retain the endpoint-propagation view of NNS but do not present $x\odot\nabla_xs(x)$ as a new base attribution formula.” | MIXED D2/D14 | Minor | Yes | It contains a non-claim, but directly prevents contribution-by-relabeling | The novelty boundary is clear and appears once | Keep once in the introduction | No change |
| Abstract, M4 spatial result | “A post-hoc full-cohort sensitivity found that attention-residual spatial change increased...” | NECESSARY_CAVEAT | Minor | Yes | The post-hoc label changes evidential status | The inferential tier is explicit | Keep the label in the abstract and M4 methods/results | No change |
| M4 Review-R2 JVP interpretation | “The discrepancies therefore occur at the same norm scale as the full response rather than arising generally from near-zero denominators.” | CLEAN | None | Yes | It directly answers the denominator diagnostic | The result is measurable rather than apologetic | Keep | No change |
| M4 Review-R2 bridge | “This is a model-level trajectory bridge, not a unit-level or causal link...” | NECESSARY_CAVEAT D12 | Minor | Yes | Targets and cohorts differ | Prevents an invalid causal merge of M3 and M4 | Keep once | No change |
| M5 method positioning | “The method distinction is therefore an estimand and information-condition distinction.” | CLEAN | None | Yes | Positive formulation defines what differs | The comparison is technically scoped | Keep | No change |
| M6 examples | “The five pages below explain aggregate results; population-level claims remain bound to the image-level M3/M4 analyses.” | NECESSARY_CAVEAT | Minor | Yes | Case studies could otherwise be mistaken for inferential evidence | The role of selected examples is clear | Keep once | No change |
| Limitations | Path-IG, M3 sampling, M4 path-scale, and display-normalization boundaries | CLEAN | None | Yes | Each limitation changes interpretation or scope | Boundaries are consolidated rather than repeated as prebuttals | Keep in Limitations | No change |
| Conclusion | “Endpoint is thus an auditable internal-target representation, not a universal intervention estimator.” | MIXED D2 | Minor | Yes | It uses a contrastive non-claim | It states the paper's central scientific positioning concisely | Keep as the single concluding contrast | No change |

### Paper-level defensive-writing score

2/10. The R3 draft leads with claims and measurements, keeps the major evidence boundaries concise,
and contains no reviewer-facing prebuttal or promotional compensation.

### Highest-priority removals

No mandatory removals remain. During later copy-editing:

1. Do not duplicate the base-formula novelty boundary outside the Introduction and Limitations.
2. Do not repeat the JVP “not a percentage” explanation outside the Review-R2 result and Limitations.
3. Keep the M3--M4 cohort/target mismatch in one bridge paragraph and one limitation only.
4. Keep `post-hoc` on Review-R2 analyses, but do not prepend it to every sentence or caption.

### Necessary caveats to preserve

- The base Endpoint object is Gradient × Input for the same scalar target.
- Registry-complete is not an exhaustive scan of all ViT neurons.
- C4a has broad 11/12 family-level support while the all-family omnibus remains `NOT_SUPPORTED`.
- C4b remains `NOT_SUPPORTED`; regional Taylor is matched to the tested reference displacement.
- Routing and identity shares are relative path-norm scales, not causal contribution percentages.
- M4 Review-R2 is post-hoc sensitivity evidence and does not change the frozen M4 decisions.
- M6 cases explain aggregate findings but do not replace image-level inference.
- Display-normalized maps support spatial-shape reading; numerical claims use raw records.

### Repeated phrase audit

- Reviewer-facing prebuttals: 0
- AI-style automatic summary constructions: 0
- Promotional adjectives used as evidence: 0
- Repeated non-claim disclaimers requiring deletion: 0
- Necessary `not`/`does not` boundaries: concentrated in method identity, result interpretation,
  limitations, and the final positioning sentence
