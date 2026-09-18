# Endpoint Scanner paper evidence matrix R1

This matrix binds the outline and bilingual R1 drafts to sealed sources. It is a writing index, not
a new statistical analysis.

| ID | Manuscript claim | Direct evidence | Figure(s) | Draft conclusion | Boundary |
|---|---|---|---:|---|---|
| P1 | The base Endpoint map for a scalar target is $x\odot\nabla_xs(x)$ | `docs/NNS_ViT_Endpoint_Scanner数学分析与方法对比_20260918.md`; EP1-R11 implementation tests | 2–4, 8–9 | ESTABLISHED_IMPLEMENTATION_IDENTITY | It is Gradient × Input, not a new base formula |
| P2 | ViT-B/16 and DeiT-S/16 have 457 and 385 registered targets per image | EP0 target registry; E2-R4 task records | 7, 11 | SUPPORTED | Equal channel IDs across models are not semantic matches |
| P3 | Registered scanning completed 336,800 finite evaluations | `docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE2结果_20260918.md`; E2-R4 terminal | 12–14 | SUPPORTED | Measures fixed models, cohort, target catalog, and backend |
| P4 | Formal scan wall time was 1,752.71 s and four-card upper bound 1.9475 GPU-hours | E2-R4 resource records and validation | 14 | SUPPORTED | Measured execution, not a universal cost ratio |
| P5 | Structured cuts reconstruct their registered branch/total objects | E3-R2 raw arrays and validation | 10, 15–17 | SUPPORTED_CORRECTNESS | Correctness only; it is not finite utility |
| P6 | 76,800 positive closures passed and 28,800 negative controls were rejected | `docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE3结果_20260918.md` | 16 | SUPPORTED_CORRECTNESS | Applies to registered cuts in two standard timm ViTs |
| P7 | Maximum closure relative L2 was 1.4144569831842091e-05 | E3-R2 validation report | 16 | SUPPORTED_CORRECTNESS | Compared with frozen `rtol=1e-4` and elementwise tolerance |
| P8 | 6,144 condition-scaled analytic JVP checks passed | `docs/NNS_ViT_Endpoint_Scanner_ENDPOINTE4恢复R3结果_20260918.md` | 18 | SUPPORTED_CORRECTNESS | Local differential check, not a finite-effect result |
| P9 | Component absolute mass predicts matched finite replacement effects overall | E6-R3 `family_decisions.json` and R4 interpretation | 19–21 | OVERALL_SUPPORTED_WITH_ONE_REGISTERED_BOUNDARY_EXCEPTION | C4a does not compare Endpoint against Taylor |
| P10 | Eleven of twelve C4a families met every condition | E6-R3 `family_decisions.json` | 20 | SUPPORTED_11_OF_12 | Frozen all-family omnibus remains `NOT_SUPPORTED` |
| P11 | The sole exception was ViT-B × normalized-zero × routing/value at 0.09956521739130435 | E6-R3 family and Holm reports | 21 | REGISTERED_BOUNDARY_EXCEPTION | 0.000435 below 0.10; do not round into a pass |
| P12 | The boundary family was positive and Holm-significant | E6-R3 Holm-adjusted p = 3.0036026037128944e-14 | 21 | SUPPORTED_DIRECTION_AND_MULTIPLICITY_ONLY | Practical-effect threshold still missed |
| P13 | Full-resolution component maps did not produce registered spatial information gain | E6-R3 C4b family decisions | 23–27 | NOT_SUPPORTED | Scope is fixed block, heads, region pool, references, and effect metric |
| P14 | Endpoint-minus-random medians were positive but only 0.001069–0.010689 | E6-R3 bootstrap report | 26–27 | BELOW_PRACTICAL_THRESHOLD | All are below the frozen 0.05 threshold |
| P15 | Regional Taylor was higher in all four C4b families | E6-R3 estimates −0.023433 to −0.035630 for Endpoint minus Taylor | 26–27 | SUPPORTED_COMPARATOR_PATTERN | Does not establish universal Taylor superiority |
| P16 | RQ3 and RQ4 address component-level and within-component spatial rankings | EP0 protocol definitions and E4/E5 run specs | 1, 19, 23 | DESIGN_IDENTITY | Task distinction supports interpretation, not a new causal mechanism |
| P17 | No cross-model Endpoint–Path conclusion is released | EP2-R1 terminal and E6 claim decisions | 2–3, 28 | PATH_COMPARISON_UNAVAILABLE | ViT-B did not pass the frozen numerical calibration gate |
| P18 | Display-normalized figures are spatial-shape views | Endpoint R2 render/validation manifests | 12–13, 17, 24 | PRESENTATION_ONLY | Numerical claims use raw signed arrays and records |

## Required manuscript wording for RQ3

The paper-level conclusion is:

```text
RQ3 was supported overall with one registered boundary exception.
```

The machine-auditable preregistered field is:

```text
C4a = NOT_SUPPORTED
```

Both must be retained. The first synthesizes the 11/12 pattern. The second records that the frozen
omnibus rule required 12/12 and was not changed after outcomes were known.

## Source roots

- RQ1: `reports/nns_vit_endpoint_e2_atlas_r4_20260918/`
- RQ2: `reports/nns_vit_endpoint_e3_closure_r2_20260918/`
- RQ3 inputs: `reports/nns_vit_endpoint_e4_intervention_r3_20260918/`
- RQ4 inputs: `reports/nns_vit_endpoint_e5_spatial_r1_20260918/`
- Confirmatory release: `reports/nns_vit_endpoint_e6_release_r3_20260918/`
- Figures: `reports/nns_vit_endpoint_unified_paper_figures_r2_20260918/`
