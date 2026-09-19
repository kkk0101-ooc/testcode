# Formal-paper writing audit

Applied academic-defensive-writing-auditor, with the local NNS article and How Does Attention Work in Vision Transformers? as style references. Their prose introduces the interpretability problem, explains the representation, then develops observations through figures. This revision follows that progression while retaining Endpoint's narrower mathematical meaning.

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Title | Auditable learning images and structured local-Jacobian analysis | D10 | Minor | No | Leads with an audit property | Reads as an engineering validation report | Lead with interpretive purpose | NNS-ViT: Understanding Internal Responses in Vision Transformers through Learning Images |
| Introduction | RQ1...RQ2...RQ3...RQ4a–c...RQ5a–b | D9 | Major | No | Project question inventory displaces motivation | Reader must learn project labels first | Explain neuron/feature/path objects and why common coordinates help | Introduction develops the problem, NNS bridge, three contributions and article structure |
| Abstract | 5,000 comparisons and registered checks pass | D9 | Major | Counts belong in results | Validation detail comes before the method's purpose | Feels like a completion report | Move counts to a numerical table | Abstract explains learning-image generation, structural decomposition and observations |
| Results | Family SUPPORTED / NOT_SUPPORTED fields in prose | D9 | Major | Scientific decision is necessary | Machine labels dominate the sentence | Administrative wording | Natural-language criterion description; exact fields in supplement | Eleven of twelve combinations meet the original composite criteria |
| Methods | Repeated warnings about non-claims | D2 | Minor | Mixed | Scope repeated before every explanation | Hard to follow the positive construction | Introduce concepts with worked diagrams; retain exact conditions once | Coordinate initialization → local derivatives → patch embedding → attention/residual |
| Visual examples | Frozen selection and receipt terminology | D9 | Minor | Selection must remain disclosed | Pipeline terms interrupt image interpretation | Figure feels like an audit artifact | Use target/layer/channel and presentation selection in captions | Same image; actual feature map; independently selected channel |
| Discussion | Blue positive views might be mistaken for complete signed maps | Evidence condition | None | Yes | Removing it changes interpretation | Necessary scientific context | Retain one comparison figure and concise explanation | Figure 22 pairs positive, signed and absolute views |

Paper-level defensive-writing score: 2/10 (editorial assessment). Main text contains no RQ stage labels, PASS strings, NOT_SUPPORTED fields or result-root paths. Numerical tables retain the established estimates and intervals; exact historical decisions remain in the supplement.

Highest-priority changes: rewrite front matter around interpretation; replace project labels with scientific section names; explain learning images before validation; present sparse numerical comparisons as tables; connect qualitative figures to observed structures; keep one concise limitations discussion.

Necessary caveats: known Gradient × Input identity, coordinate-origin dependence, non-completeness, correct residual cotangent, local rather than finite causal meaning, sample selection, class-agnostic attention comparison, distinct display normalization, original family decisions, untested training and target controls.

Figure 20 revision: the current examples were selected from pre-retained class Endpoint maps by a presealed readability rule before Grad-CAM or attention outputs were computed. The text describes them as presentation examples and does not infer population superiority.

Repeated phrase audit: “we do not claim”, “we emphasize”, “it should be noted”, “to avoid misunderstanding”, and “taken together” are absent from both current main texts. The final validator checks bilingual equations, numerical table rows and section/figure numbering.
