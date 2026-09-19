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

## Fine-grained display addendum (Figures 34--40)

Source: `reports/nns_vit_endpoint_fine_grained_blue_figures_r1_20260919`.

| Paper role | Figure | Reason |
|---|---:|---|
| Registered-target reading view | 34 | Places the fixed input beside six real scalar-target Endpoint maps |
| Structural-component reading view | 35 | Shows identity, head, routing, value, attention, MLP, and complete Endpoint at block 5 |
| Spatial-ranking reading view | 36 | Places total G×I, raw gradient, and three registered head maps on one page |
| Fig.13-style depth progression | 37 | Tracks one fixed channel-0 target from patch embedding through all blocks and final targets |
| DeiT-S head-by-depth atlas | 38 | Compares all six heads at four frozen depths for one registered scalar type |
| ViT-B head-by-depth atlas | 39 | Compares all twelve heads at four frozen depths for the same registered scalar type |
| ViT-B module chain | 40 | Follows channel 0 through patch, Norm, attention, MLP, final CLS, and class logit targets |

These blue-base panels use `max(L,0)` and per-panel P99 scaling for display only. They support
fine-grained spatial reading but not cross-panel magnitude comparisons. Signed companions are
retained in the source package, and all numerical claims remain bound to sealed raw arrays.
