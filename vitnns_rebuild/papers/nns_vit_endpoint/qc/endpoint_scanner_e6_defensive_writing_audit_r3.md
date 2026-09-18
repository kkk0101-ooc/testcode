# Endpoint Scanner E6 defensive-writing audit

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| E6 interpretation boundary | Endpoint base attribution is not a new formula distinct from Gradient × Input. | D14 | Minor | Yes | It could read as novelty prebuttal if repeated. | Clear once; distracting if repeated. | Preserve once as the identity result. | The base Endpoint map equals Gradient × Input for the same scalar target. |
| E6 Path boundary | Path IG calibration failed for ViT-B/16 under the frozen candidate range, so the release contains no cross-model Endpoint–Path claim. | CLEAN | None | Yes | It states the direct failed gate and consequence. | The scope follows the registered evidence. | Preserve once. | No change. |

### Paper-level defensive-writing score

2/10. The release leads with measured results and keeps one concise boundary per claim.

### Highest-priority removals

- Do not repeat the Gradient × Input novelty boundary outside Identity and the interpretation boundary.
- Do not excuse any NOT_SUPPORTED family or infer causes without a registered diagnostic.
- Do not convert E3/E4 correctness into utility language.

### Necessary caveats to preserve

- Endpoint base attribution is Gradient × Input for the same target.
- E3 closure and E4-R3 JVP are correctness evidence.
- Path IG cross-model claims remain blocked by the immutable EP2-R1 calibration failure.

### Repeated phrase audit

The release uses no reviewer-facing prebuttal phrase and states each material boundary at most twice across the result and interpretation paragraphs.
