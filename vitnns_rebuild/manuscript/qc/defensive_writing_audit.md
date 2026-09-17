# Defensive-writing audit

Scope: Abstract, Introduction, contribution bullets, Results and Discussion, Limitations, and Conclusion in both language versions. The table records the highest-priority cases found during drafting and the implemented rewrite.

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Abstract ending | The current evidence does not establish broad spatial stability or general superiority | D12 | Major | Mixed | Ends on a generic non-claim | Ask which tests actually failed | Name the observations | The broad stability gate fails, and Grad-CAM has higher fidelity in the independent comparison |
| Introduction, main evidence | The comparison does not show general superiority | D2/D12 | Minor | No | Frames the result around a denied claim | Look for the actual ordering | Report the ordering directly | NNS scores below Grad-CAM, above rollout, and has nonsignificant contrasts with full IG and raw CLS attention |
| Results 5.2 | It does not imply equality of per-patch totals or certify every cell as causal | D2 | Major | Mixed | Stacks two disclaimers after the result | Suspect that the comparator is being oversold | State the evaluated object and omitted test | Per-patch equality was not part of the comparison; the intervention evaluates ranking rather than individual signed cells |
| Results 5.3 | Four coordinates cannot determine a global ordering | D12 | Minor | Yes | Negative framing repeats the scope | The scope is already visible in the design | State the sampled scope | The comparison covers four prespecified coordinates |
| Results 5.3 | Directional agreement does not establish equivalence or causality | D2 | Minor | Mixed | Anticipates two objections at once | Ask whether equivalence was tested | Keep the material statistical fact | The effect appears in both strata; no equivalence test was specified |
| Results 5.4 | The atlases do not select heads by appearance; the evidence is not stable division of labor | D1/D2 | Major | Mixed | Repeated negative framing | Focus shifts from data to rebuttal | Describe frozen selection and decision | Indices were fixed before viewing; stable semantic specialization remains unresolved |
| Results 5.6 | The total map is not a distinct total attribution | D14 | Major | Yes | Defines contribution by denying novelty | Ask what the method actually adds | State identity and added object | The total equals same-contract full IG; the complete cut adds an internal component index |
| Limitations | Multiple negative statements | D8 | Minor | Yes | Legalistic if repeated in Results and Abstract | Perceive an over-defended paper | Keep limitations here and shorten earlier repetitions | Retained as compact, evidence-changing scope statements |

## Paper-level defensive-writing score

- Before cleanup: 4/10 (mild-to-noticeable defensive framing).
- Current draft: 2/10 (necessary scope statements remain, concentrated in Results and Limitations).

## Highest-priority removals

1. Replaced abstract-level generic non-claims with the observed failed stability gate and Grad-CAM contrast.
2. Reframed the Stage D comparator around the measured ranking question.
3. Replaced global disclaimers for the four-coordinate and prediction-strata analyses with explicit design scope.
4. Reframed complete-cut novelty around the full-IG identity and internal component index.
5. Kept one consolidated limitations section instead of repeating every caveat in the conclusion.

## Necessary caveats to preserve

- C10 and Stage D use separate cohorts and decisions.
- The Stage D target is a fixed-clean-attention Value path, not full Q/K/V or a complete class explanation.
- The target-matched patch-token baseline and matched occlusion control are different controls.
- Two Stage E layer/head records are typed failures and the package has mixed scientific validity.
- Nonsignificant method contrasts do not establish equivalence.
- Figures 28–35 are single-image method validations.
- Stage X2-MECH and Stage S are post-hoc exploratory.

## Repeated phrase audit

No reviewer-facing phrases such as “we emphasize,” “it should be noted,” “to avoid misunderstanding,” or “we do not claim” remain in the English manuscript. Residual uses of “does not,” “cannot,” and “not” predominantly encode target definitions, failed gates, or statistical non-equivalence and are retained where they change interpretation.
