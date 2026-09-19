# Endpoint M5 defensive-writing audit

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Method identity | For a logit target, Endpoint is Input × Gradient in normalized-input coordinates. | CLEAN D14 | Minor | Yes | It names an established identity directly rather than defending novelty. | The contribution boundary is clear. | Keep once in Methods and M5. | Unchanged. |
| Reference result | The low cross-reference concordance shows that the intervention reference changes component ordering, not merely score scale. | CLEAN | Minor | Yes | It is a direct inference from the measured rank concordance. | The evidence and scope are explicit. | Keep. | Unchanged. |
| Method matrix scope | The comparison concerns estimands and information conditions. | CLEAN | Minor | Yes | It states what the table compares. | The absence of a leaderboard is not argued defensively. | Keep. | Unchanged. |
| Output localization | This study makes no output-localization superiority claim. | NECESSARY_CAVEAT D2/D5 | Minor | Yes | It bounds why methods are compared semantically rather than numerically. | One concise scope sentence is sufficient. | Keep once after the matrix. | The evaluated comparisons concern internal targets and finite replacements; output-localization superiority was outside this protocol. |
| Path IG | Path IG remained outside the cross-model release because ViT-B passed only 41/64 and 40/64 cases at 512 points. | NECESSARY_CAVEAT | Minor | Yes | It reports the failed gate and direct consequence. | The boundary follows a frozen numerical result. | Keep once. | Unchanged. |
| Runtime | The combined packages do not identify comparable per-method runtimes. | NECESSARY_CAVEAT | Minor | Yes | It prevents a package-level measurement from becoming a method claim. | The cost evidence is correctly scoped. | Keep once with the cost table. | Unchanged. |

### Paper-level defensive-writing score

2/10. The section leads with definitions and measurements, states one scope boundary per comparison, and does not excuse negative results.

### Highest-priority removals

1. Do not repeat the output-localization non-claim outside the method matrix and Limitations.
2. Do not describe the missing Path IG release as a strategic omission; report the 61/64 gate and observed counts.
3. Do not convert combined package timings into a runtime advantage.

### Necessary caveats to preserve

- Endpoint is independent of an external intervention reference but tied to the normalized-input origin.
- Endpoint logit and normalized-input Input × Gradient are the same formula.
- Taylor's advantage is task- and reference-specific in the frozen M1/M2 evaluations.
- ViT-B Path IG calibration blocks a cross-model Path comparison.
- M1/M2 package costs do not isolate per-method runtime.

### Repeated phrase audit

Reviewer-facing prebuttals: 0. Repeated non-claim disclaimers: 0. Promotional adjectives: 0. AI-style summary constructions: 0.
