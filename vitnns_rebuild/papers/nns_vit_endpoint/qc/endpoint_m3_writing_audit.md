# Endpoint M3 defensive-writing audit

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Methods scope | This is a stratified neuron audit, not an exhaustive ViT neuron scan. | D2/D12 | Minor | Mixed | The negative non-claim repeats the coverage numbers. | The scope is clear from the numerator and denominator. | State the measured coverage positively. | The stratified audit covers 0.338% of ViT-B/16 and 0.677% of DeiT-S post-block token-channel units. |
| Numerical recovery | The FAIL remains recorded. | D2 | Minor | Necessary caveat | The sentence is procedural but materially affects numerical interpretation. | Retention is useful if paired with the recovery result. | Keep once with the exact counts. | Of 57,600 float32 finite-difference conditions, 36 missed the threshold and remain as the initial FAIL evidence; the published recovery rule selected all 10 affected tasks, whose float64 review passed. |
| Evidence boundary | M3 includes no neuron-replacement study. | D5 | Minor | Mixed | It explains an omission rather than stating the evidence scope. | Better placed as a relation to the completed behavior studies. | Use a positive evidence mapping. | M3 evaluates the stratified atlas; finite-replacement behavior remains calibrated by M1/M2. |
| Mechanism conclusion | This decision describes local first-order input dependence for the frozen token-channel sample. | D12 | Minor | Necessary caveat | It limits the construct and sampling frame, both scientifically material. | The boundary prevents a semantic upgrade to causal or universal claims. | Preserve once adjacent to the conclusion. | Unchanged. |
| Cross-model comparison | Channel IDs are not matched neurons across models. | D2 | Minor | Necessary caveat | It is a negative statement, but prevents invalid unit correspondence. | Necessary for interpreting separate-model tables. | Preserve once. | Unchanged. |

### Paper-level defensive-writing score

2/10 after revision. The result text leads with measurements and keeps only construct, numerical-recovery, and cross-model boundaries that change interpretation.

### Highest-priority removals

1. Replaced the repeated non-exhaustive-scan disclaimer with exact positive coverage.
2. Reframed the omitted neuron-replacement sentence as an evidence mapping to M1/M2.
3. Kept the float32 failure disclosure once with its denominator and prospective float64 recovery.

### Necessary caveats to preserve

- Endpoint is a local first-order input-dependence object over normalized image coordinates.
- The 36/57,600 float32 finite-difference failures remain part of the audit record.
- The atlas samples 6,144 frozen units per model-image; model channel IDs are not corresponding neurons.
- Local-to-global support applies to the tested models and frozen unit sample.

### Repeated phrase audit

Reviewer-facing prebuttals: 0. Repeated non-claim disclaimers: 0. Promotional adjectives: 0. AI-style summary constructions: 0.
