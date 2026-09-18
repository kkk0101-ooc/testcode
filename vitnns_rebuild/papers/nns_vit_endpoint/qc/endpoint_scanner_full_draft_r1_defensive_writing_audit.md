# Endpoint Scanner full draft R1: defensive-writing audit

Audited files:

- `papers/nns_vit_endpoint/zh/full_draft.md`
- `papers/nns_vit_endpoint/en/full_draft.md`

The audit follows `claim -> evidence -> concise boundary`. It does not rescore novelty or recompute
scientific results.

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Abstract, method identity | The base quantity is Gradient × Input; the subject of this study is the registration of ViT targets, structural accounting, and the empirical utility of the resulting components. | NECESSARY_CAVEAT | None | Yes | It prevents contribution-by-relabeling and states the technical delta positively. | Clear method identity. | Preserve once in the abstract and once in Methods. | No change. |
| Introduction, method identity | We retain the endpoint-propagation view of NNS but do not present x-times-gradient as a new base attribution formula. | MIXED/D2 | Minor | Yes | The negative construction is slightly defensive, but the novelty boundary is material. | Likely acceptable if not repeated. | Keep here; avoid repeating in Results and Conclusion. | We study the ViT registration, structural cuts, and intervention behavior of the NNS endpoint rule, whose base map is Gradient × Input. |
| RQ3 result | We therefore interpret RQ3 as supported overall with one registered boundary exception. | CLEAN | None | Yes | It leads with the evidence-weighted conclusion. | Clear scientific synthesis. | Preserve. | No change. |
| RQ3 boundary | The frozen family status and 12-of-12 omnibus C4a field remain NOT_SUPPORTED. | NECESSARY_CAVEAT | None | Yes | It prevents the synthesis from appearing to alter the preregistration. | Transparent distinction between machine field and paper interpretation. | State once in Results and briefly in the Abstract. | No change. |
| RQ3 boundary | The threshold was not moved, and 0.099565 was not rounded into a pass. | MIXED/D1 | Minor | Yes | It anticipates concern about post-result threshold changes. | Reassuring in Results, repetitive elsewhere. | Keep only in the detailed RQ3 Results paragraph. | No change. |
| RQ4 result | RQ4 is therefore NOT_SUPPORTED. | CLEAN | None | Yes | Direct statement of the negative result. | Credible and easy to locate. | Preserve. | No change. |
| Discussion, Taylor | The present experiments do not imply that regional Taylor is universally superior at other layers, spatial scales, tasks, or models. | NECESSARY_CAVEAT/D12 | Minor | Yes | The universal-scope boundary is material but uses a negative construction. | Acceptable once in Discussion. | Preserve once; do not repeat in Limitations. | Regional Taylor was stronger under the registered block-5 intervention setting; other settings were not evaluated. |
| Limitations, Path | Consequently, this paper contains no formal cross-model Endpoint-versus-Path fidelity, stability, or cost-quality comparison. | NECESSARY_CAVEAT | None | Yes | It defines the unavailable comparison caused by a documented calibration failure. | Clear evidence boundary. | Preserve in Limitations only. | No change. |
| Reproducibility | The statistical validator imports neither Torch, timm, nor PIL. | CLEAN | None | Yes | It reports an auditable implementation fact. | Useful reproducibility detail. | Preserve in Experimental Design, not front matter. | No change. |

### Paper-level defensive-writing score

2/10. The draft states the main positive and negative findings directly. Necessary boundaries are
mostly confined to method identity, the RQ3 omnibus distinction, Path calibration, and study scope.

### Highest-priority removals

1. Do not repeat the `C4a = NOT_SUPPORTED` machine field in every Discussion or Conclusion
   paragraph; one abstract mention and one full Results explanation are sufficient.
2. Do not add language such as “despite narrowly missing,” “encouraging,” or “almost passed” around
   the 0.099565 family. Report the exact gap and the 11/12 pattern.
3. Do not excuse RQ4 with unmeasured claims about coarse grids, inadequate sample size, or model
   difficulty.
4. Do not promote source hashes, recovery roots, or execution detours into the scientific narrative;
   keep them in reproducibility material.
5. Replace broad adjectives such as “robust,” “comprehensive,” or “superior” with the registered
   counts and contrasts if they appear during revision.

### Necessary caveats to preserve

- The base Endpoint quantity is Gradient × Input.
- Correctness does not establish finite-intervention utility.
- RQ3 is supported overall, but one family remains below the frozen 0.10 threshold and the strict
  12/12 omnibus field remains `NOT_SUPPORTED`.
- RQ4 remains `NOT_SUPPORTED`, with regional Taylor higher in all four registered families.
- Path-IG calibration failed for ViT-B/16, so cross-model Endpoint–Path conclusions are unavailable.
- Display scaling is not a numerical comparison.

### Repeated phrase audit

| Construction | Chinese draft | English draft | Assessment |
|---|---:|---:|---|
| “不/does not” claim boundaries | 18 | 17 | Mostly technical definitions; monitor during shortening |
| `NOT_SUPPORTED` | 9 | 9 | Includes Table 4/5 status cells and required machine/result labels |
| “整体支持/supported overall” | 2 | 4 | Concentrated in the Abstract, Results, Discussion, and Conclusion |
| “不能/cannot” | 3 | 2 | Concentrated in method distinction and inference boundaries |
| “因此/therefore” | 9 | 8 | Normal argument transitions; no automatic summary chain detected |

No reviewer-facing prebuttal, fairness self-defense, promotional compensation, or omitted-experiment
justification was found.
