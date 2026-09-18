# M1 supplementary evidence matrix

Root: `reports/nns_vit_endpoint_m1_r1_20260918/`. Primary E3/E4/E6 sources are immutable.

| Claim or table | Evidence | Interpretation |
|---|---|---|
| Same 400 images, 2 models, 12 blocks, 2 references | task_manifest.json; task_validation.json | Exact image/model/component coverage |
| 70,656 new conditions; 326,400 combined conditions | results/; per_image/; task_validation.json | Missing edges acquired, existing conditions inherited |
| Seven predictors and 1,000 random permutations per set | config; per_image/; metrics_manifest.json | Shared finite outcome and component sets |
| Table 6 registry coverage | EP0 target_registry.json; registry_coverage.json | Counts and object-specific candidate populations |
| Table 7 within-block head ranking | summary.json.gz, family=head, stratum=within_block_mean | Taylor higher on rho/Capture and lower on Regret |
| Residual identity paired rho advantage | paired_comparisons.json.gz; paired_spearman_tests.json | Four across-block comparisons positive after Holm; Capture@1 median difference zero |
| Clean Taylor/reference/skip implementation | taylor_implementation_diagnostic.json | 48 float64 checks; diagnostic only |
| Image unit, bootstrap, sign, Holm | inference_validation.json; absolute_spearman_tests.json | Separate absolute >0/>0.10 and relative hypotheses |
| Original RQ3 interpretation | E6-R3 family_decisions.json | 11/12; broad but non-universal; omnibus NOT_SUPPORTED |
| M1 completion | completion_audit.json; terminal_receipt.json | Stop after M1 |

The M1 layer figure is a supplementary artifact at `m1_head_ranking.png`; it is not added to the
sealed 28-figure package. The report gives its numbered Image #1 interpretation.
