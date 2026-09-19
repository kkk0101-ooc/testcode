# Endpoint M3-SUP-R2 defensive-writing audit

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Spatial finding | “local to global” | D13 | Major | Mixed | It compresses spatial broadening into a stronger representational claim. | The measured metrics do not establish uniform whole-image support or global semantics. | Name the measured construct. | “spatial broadening of local input dependence” |
| Fidelity count | “5,000 samples” | D10 | Major | Necessary caveat | The count can sound like inferential sample size. | Readers may overstate statistical independence. | State their role directly. | “5,000 numerical validation cases; inference is not based on treating them as independent samples.” |
| CLS interpretation | “CLS globally integrates information” | D13 | Major | Necessary caveat | Entropy is a local derivative-distribution metric. | Causal integration is not identified. | State the recovered property. | “CLS Endpoint mass has higher spatial entropy than patch-unit mass.” |
| Cancellation | “absolute mass is contribution” | D14 | Major | Necessary caveat | Near-zero net ratios show opposing signed terms. | Absolute and net directional effects would be conflated. | Distinguish the quantities. | “Absolute mass measures total local first-order magnitude; R_net measures the surviving signed aggregate.” |
| Float32 failure | Repeated apology for 36 failures | D4/D8 | Minor | Mixed | Repetition makes a numerical diagnostic sound defensive. | One exact disclosure plus recovery evidence is sufficient. | Report count, curve, recovery, and retained status once. | “36/57,600 float32 conditions failed; all counterparts passed the frozen float64 recovery, and the original failures remain recorded.” |

### Paper-level defensive-writing score

2/10 after revision. Claims lead with measurements and use one material boundary per construct.

### Highest-priority removals

1. Replace “local to global” with the measured spatial-broadening construct.
2. Do not call the 5,000 numerical cases independent samples.
3. Separate CLS entropy from causal information integration.
4. Separate absolute mass from net directional effect.

### Necessary caveats to preserve

- The M3 atlas is a frozen stratified sample of post-block token-channel units.
- The result concerns local first-order image-coordinate dependence.
- The 36 float32 failures remain in the audit trail after float64 recovery.
- Post-block targets do not identify a unique upstream operator as the source of finite-difference error.

### Repeated phrase audit

Reviewer-facing prebuttals: 0. Repeated non-claim disclaimers: 0. Promotional adjectives: 0. AI-style summary constructions: 0.
