# Revision against 修改意见v1.md — 2026-09-20

The deliverables are synchronized Chinese and English current manuscripts, ten readable main figures with signed and absolute views, supplementary material retaining the previous evidence, and a requirement-to-artifact completion audit. The source opinion is the current working-tree version of `修改意见/修改意见v1.md`; its digest is recorded by the revision analysis.

This document records the earlier manuscript-and-presentation revision scope. The later `补充实验/执行协议v1.md` prospectively supersedes the status of trained/random controls, target robustness and feature–Endpoint quantification, all now completed in `publication_r2_20260920/evidence/supplement_v1/`. Matched-target decomposition and alternative trained checkpoints remain prospective.

## Provenance and scope

- Preserve sealed `reports/nns_vit_endpoint_all_figures_r2_20260919/` byte for byte. Build its publication revision at `reports/nns_vit_endpoint_paper_revision_v1_20260920/` and bind it through the paper's `figure_manifest.json`. Archive the old binding as `figure_manifest_all_figures_r2.json`.
- Preserve the previous bilingual manuscript in `zh/full_draft_r3_archive.md` and `en/full_draft_r3_archive.md`. Historical stage drafts retain their original names and claims; the README identifies the current manuscript.
- Recompute supplementary descriptive statistics from SHA-256-checked E5 and M1 records. Frozen C4a/C4b decisions, bootstrap intervals and Holm families remain unchanged. New descriptive tables do not retrospectively replace the original tests.
- Revise the current draft validator prospectively: previous checks assumed 40 main figures, old RQ labels and an old title, contradicting the requested revision. Retain evidence checks but move engineering tables and detail to the supplement, and add checks for the actual revision requirements.

## Supplementary analysis definitions, fixed before execution

1. E5 absolute AUC: reconstruct the trapezoid from each recorded isolated-effect curve over area 0–0.25 and divide by 0.25. For random regions, average signed effects across the 20 repetitions before taking absolute values and integrating, as required by the frozen E5 estimator; then average heads within image and report the median across 400 images per model/reference. Verify individual AUCs and every reconstructed image score against E6.
2. Empirical oracle: at each recorded budget and head, take the largest absolute isolated effect among all actually evaluated method/random curves. Integrate that envelope, then average heads. It is an oracle over evaluated candidates, not the optimum over all possible regions; budgets can use different candidates.
3. Normalized regret per image is `(oracle_auc - method_auc) / oracle_auc`; zero oracle denominators produce NA. Also report per-image AUC divided by absolute clean ground-truth logit, with zero denominators NA. Never take a ratio of cohort medians as a substitute.
4. Reference change uses `abs(q0-qblur) > 1e-12 + 0*max(abs(q0),abs(qblur))`, matching the existing M5 absolute tolerance. Report component-count fraction, absolute and symmetric-relative change median/q10/q90 separately by model and quantity. These component distributions are descriptive, not independent inferential samples. Relative change is divided by `max(abs(q0),abs(qblur))`; both zero is NA.
5. Head rankings: each model-image-block uses average ranks, Spearman, Kendall tau-b, top-1 identity, top-3 overlap divided by 3, and mean absolute rank displacement. Top-K ties use component ID in ascending order. Average 12 blocks per image; report image-level median/q10/q90 and exact full-ranking agreement counts.
6. Main images: signed view is `sum_c E_c`; absolute view is `sum_c abs(E_c)`. Scientific mass/norm/cancellation annotations use the raw RGB tensor. Related maps share signed and absolute robust scales separately; raw scales and clipped counts are recorded. Signed RGB projection may hide inter-channel cancellation, which the absolute view preserves.

## Prompt-to-artifact checklist (completion evidence is added after verification)

| Opinion items | Required disposition | Planned evidence |
|---|---|---|
| 1.1–1.2, 2.1 | NNS identity, conditions, column-gradient convention; formula is established G×I | Method 3.1, Proposition 1, Fig. 1, RQ1 |
| 1.3 | Derive activation-weighted filter aggregation via energy | Method 3.2 |
| 1.4, 3.4 | Registry-complete scope, true-neuron coverage, scalar-target terminology | RQ2, Table 1, supplement coverage table |
| 1.5, 2.2–2.4 | Local scaling meaning, no value completeness, coordinate origin, constants | Method 3.1 and limitations |
| 2.5–2.8 | Q/K/V differential, output projection, legal cut theorem/proof, correct residual cotangent, Value-only error | Methods 3.3–3.4, Fig. 2, RQ3/RQ4b |
| 2.9, 5.1–5.5 | Distinct map/path cancellation and five scientific findings | RQ4a–c, discussion, Figs. 4–8, 10 |
| 3.1, 7 | Reorder RQ1–RQ5 then reference analysis; move engineering detail to supplement | Current bilingual drafts and supplement |
| 3.2–3.3 | 0.10/0.05 point thresholds versus sign tests against zero | Design 4.2, RQ5a/b, CI-lower-bound sensitivity table |
| 3.5 | Absolute Endpoint/Taylor/random/oracle AUC, logit scale and normalized regret | Recomputed analysis JSON/CSV, Tables 6–7, Fig. 10 |
| 3.6–3.7 | Explicit change tolerance, absolute/relative quantiles, fraction changed, rank metrics by model | Tables S3–S4, RQ5/reference analysis, analysis script |
| 4.1 | Signed plus absolute for overview, atlas, routing/value, residual, cancellation, intervention boundary | Figs. 1, 5–8, 10 and display receipt |
| 4.2 | Shared scales or raw mass/norm annotations, no unsupported intensity inference | Display protocol and per-panel metadata |
| 4.3, 11 immediate items 1–7 | 8–12 main figures, supplement all-head/module/diagnostic pages | Ten main figures, supplement index, manifest |
| 6 and 11 suggested experiments | Identify absent controls; prioritize prospective work; do not invent evidence | Limitations and follow-up experiment plan |
| 8–10 | Focused abstract, four-part conclusion, recommended bilingual title | Abstract, conclusion and title |
| User image beautification | Consistent typography/palette/spacing; PNG/PDF/SVG; visual inspection | Gallery, figure_analysis.md, manual QC and exported images |
| AGENTS.md | Bilingual synchronization, sealed evidence binding, Identity/Correctness/Utility, 11/12 support with omnibus and C4b preserved | Updated validator, source hashes, writing audit |

## Prospective experiments, separate from this revision

Analysis correction before publication: the initial local reanalysis failed its E6 equality assertion because it averaged random absolute AUCs. Inspection of E5 `random_aggregates` established that its frozen baseline averages signed curves first. The correction above preserves the existing estimand; no result or scientific gate was changed. The failing attempt wrote no analysis output.

1. Same-architecture random initialization: 32–64 images, all 12 blocks, fixed token/channel sample, entropy/locality/distance and routing/residual metrics; compare trained minus random without repeating large finite intervention sweeps.
2. GT versus frozen clean-top1 logits: separate correctly/incorrectly classified images; repeat routing scale, Value-only error, residual identity share and spatial TV. Existing top1 total maps alone cannot substitute for target-conditioned component decompositions.
3. Matched internal targets: 16–32 shared images, blocks 0/5/11 and frozen token-channel targets; compare full/value entropy, locality and patch TV on the same target.
4. Original-NNS CNN illustration: explicit layerwise and Endpoint pixel maps, energy/filter aggregation and MaxPool-winner inheritance. The existing five-network numerical fidelity test remains the current evidence.
5. If resources permit, same-architecture alternative checkpoints or training regimes.
6. Optional small functional comparison of Endpoint, raw attention, rollout, relevance and regional Taylor; use each method's own object and avoid a leaderboard interpretation.
