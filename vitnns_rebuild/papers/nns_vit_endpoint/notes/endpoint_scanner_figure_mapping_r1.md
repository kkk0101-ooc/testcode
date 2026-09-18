# Endpoint Scanner figure-to-paper mapping R1

Source: `reports/nns_vit_endpoint_unified_paper_figures_r2_20260918`.

## Recommended main-text sequence

| Paper role | Figure | Reason |
|---|---:|---|
| Study overview | 1 | One-page map from registered targets to RQ3/RQ4 |
| Method identity | 2 | Separates Endpoint, original CNN endpoint logic, and Path IG |
| Attention math | 5 | Shows Q/K/V differential paths |
| Structured decomposition | 10 | Defines complete tensor-valued cuts |
| Qualitative registered outputs | 12 | Same frozen first eight images across both models |
| Resource/scale result | 14 | Target counts and measured complete-atlas latency |
| Correctness result | 16 | Closure and negative-control counts plus tolerance |
| RQ3 primary result | 20 | All 12 C4a estimates and intervals |
| RQ3 interpretation | 21 | 11/12 support and the 0.099565 boundary exception |
| Spatial-map example | 24 | Real component and mandatory-baseline maps |
| RQ4 primary result | 26 | All 16 mandatory-baseline contrasts |
| Final evidence map | 28 | Identity, Correctness, RQ3, RQ4, and Path boundary |

## Recommended supplement

- Figures 3–4 and 6–9: analytic distinctions and operator details.
- Figures 11 and 13: registry composition and multi-target atlas.
- Figures 15, 17–19: structural-cut and finite-replacement details.
- Figure 22: registered secondary predictors.
- Figures 23, 25, and 27: spatial protocol, one-image curves, and compact RQ4 matrix.

Main-text and supplement placement is editorial. Figure numbers, source arrays, numerical labels,
and conclusions are frozen by `papers/nns_vit_endpoint/figure_manifest.json`.
