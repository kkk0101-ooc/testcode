# Revision v1 completion audit

Objective: revise the Endpoint paper according to the current 修改意见v1.md and improve the paper figures based on reports/nns_vit_endpoint_all_figures_r2_20260919. Concrete deliverables are synchronized current Chinese/English manuscripts; a concise RQ-ordered narrative; explicit mathematical/statistical conditions; ten main figures; complete supplementary navigation; source-bound descriptive statistics; and reproducible validation.

This file is the historical completion audit for the earlier manuscript-and-figure revision. At that checkpoint, suggested new studies were prospective and no new model inference had been run. The later supplement-v1 execution supersedes that experimental status; see `publication_r2_20260920/qc/supplement_v1_completion_audit.md` for the current audit.

| Opinion / instruction | Artifact and direct evidence | Completion disposition |
|---|---|---|
| 1.1–1.2 NNS formalization/generalization | Both drafts 3.1 Proposition 1; 5.1; Figure 3; original 5,000 comparison records | Implemented; derivative conditions retained |
| 1.3 filter aggregation | Both drafts 3.2 derive energy gradient and weighted Endpoint sum | Implemented |
| 1.4 registry versus all-neuron coverage | Both drafts 5.2; exact percentages; Supplement Table S1 | Implemented; full enumeration not claimed |
| 1.5 local meaning and semantics | 3.1 coordinate-scaling derivative; discussion and limitations | Implemented |
| 2.1 row/column convention | 3.1 column gradient and transposed learning-image row | Implemented; bilingual displayed equations identical |
| 2.2 non-completeness | 3.1 explicit unequal sum and Euler condition | Implemented |
| 2.3 normalization origin | 3.1 normalized-to-RGB identity | Implemented |
| 2.4 bias/CLS/position | 3.1 constants affect Jacobian but have no image-coordinate terms | Implemented |
| 2.5 attention differential | 3.3 full Q/K/V and routing/value; head before output projection; cotangent includes projection | Implemented |
| 2.6 legal edge cut | Proposition 2 states same state, correct edges, every path exactly once and proof | Implemented |
| 2.7 residual | Explicit output cotangent and identity/branch equations | Implemented |
| 2.8 Value-only meaning | 3.5, 5.5 and denominator/absolute-error source tables | Implemented; 0.95 is a norm ratio |
| 2.9 cancellation | Separate map/path formulas; Figures 5/7 and discussion | Implemented |
| 3.1 RQ order | Results 5.1–5.9; matching bilingual section structure | Implemented |
| 3.2 C4a threshold | 4.2, 5.7, Table 2 lower-CI sensitivity, DeiT example | Implemented; original decisions preserved; optional shifted test not added |
| 3.3 C4b threshold | 4.2 and 5.8 explicitly distinguish point criterion from zero-null test | Implemented |
| 3.4 scalar-target wording | 5.2 removes ambiguous finite-target language | Implemented |
| 3.5 absolute AUC, random/oracle/logit/regret | Tables 5–6, Figure 10; CPU raw-curve reconstruction and 400-image groups | Implemented; oracle restricted to evaluated candidates |
| 3.6 tolerance and distributions | 5.9, Table S3, atol=1e-12/rtol=0, model-separated absolute/relative quantiles and changed fractions | Implemented |
| 3.7 rank concordance | Table 7 and S4; Spearman, tau-b, top1, top3, mean rank displacement; 400 image units/model | Implemented |
| 4.1 signed + absolute views | Figures 1/5/6/7/8/10; 26 map panels from source arrays | Implemented |
| 4.2 comparable scales | Shared group scales, explicit colorbars, raw RGB L1/L2/signed sums; display receipt | Implemented |
| 4.3 8–12 main figures | Exactly ten figure records and ten embedded figures per language | Implemented |
| 5.1–5.5 strongest findings | 5.4–5.8 and discussion cover spatial broadening, Value-only departure, residual differences, sharpness relationship, distinct behavioral tasks | Implemented with existing inference labels |
| 6.1–6.6 proposed new experiments | Limitations and revision_v1_protocol.md prospective list | Addressed as absent evidence; no new experiments in this writing scope |
| 7 engineering detail reduction | Supplement S1–S8, current README; 40 old main and 52 old supplemental records indexed | Implemented |
| 8 focused abstract | Current bilingual abstracts cover identity/registry/structure, mechanisms and behavior | Implemented |
| 9 four-part conclusion | Current conclusion: fidelity, structure, observed patterns, behavior | Implemented |
| 10 title | Uses preferred From NNS to Vision Transformers / 从 NNS 到视觉 Transformer title | Implemented |
| 11 priorities | All seven immediate manuscript changes mapped above; proposed experiments listed by priority | Implemented within revision scope |
| User figure beautification | Ten PNG/PDF/SVG figures, gallery, bilingual captions, Image #1–#10 explanation, manual QC | Implemented |
| AGENTS bilingual/source rules | Same displayed equations and table rows, valid links, manifest hashes, unchanged R2 inventory | Covered by revision validator |
| Reproducibility commands | README command sequence; analysis, render, bind and validate executed | Results recorded in validation report |

Validation is not accepted by status alone. The validator compares actual table text to source-generated values; recomputes every E5 curve from isolated scores and cross-checks original image estimates through the analysis; verifies raw source and output hashes; independently aggregates published image CSV values; checks exact bilingual equations and structures; and confirms every R2 inventoried file is unchanged. Manual figure review covers content visibility, colorbar readability and spacing. The current machine result is in [revision_v1_validation_report.json](revision_v1_validation_report.json).

The initial supplementary AUC attempt failed before writing outputs because random repeats were averaged after absolute value. The corrected implementation uses the original signed-curve averaging; all original E6 per-image scores now agree within 1e-12. A validator implementation correction uses each source's actual canonical-digest convention (E5/E6 hash the files list). Neither correction changes a frozen scientific contract.

Remaining submission metadata: author names, affiliations, repository/archive identifier, and provisional bibliography metadata. These pre-existing author inputs are explicit placeholders. New scientific controls remain limitations and prospective work. They are not hidden behind the revision completion status.

Final local validation: 3,073 checks passed with zero errors. The independent raw-display verifier passed all 26 panels and matched visual-QC hashes to the current ten images. Scoped git diff --check for the manuscript passed; the supplied opinion retains its original Markdown hard-break whitespace. No source R2 files changed.
