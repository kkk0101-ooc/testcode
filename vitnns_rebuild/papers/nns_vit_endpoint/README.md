# NNS-ViT Endpoint Scanner paper

This is the independent bilingual manuscript project for Endpoint scanning and structured
decomposition. It does not share draft, evidence-note, QC, reference, script, or figure-manifest
directories with the path-integrated NNS-ViT paper.

## Reading order

1. `zh/outline.md`
2. `zh/full_draft.md` or `en/full_draft.md`
3. `notes/endpoint_scanner_paper_evidence_matrix_r1.md`
4. `notes/endpoint_scanner_figure_mapping_r1.md`
5. `references/endpoint_scanner_references_r1.md`
6. `qc/endpoint_scanner_full_draft_r1_defensive_writing_audit.md`

The frozen 28-figure binding is `figure_manifest.json`. The source package remains
`reports/nns_vit_endpoint_unified_paper_figures_r2_20260918/`; figures are referenced rather than
copied into this paper root. The independent Figure 34--40 fine-grained display binding is
`figure_manifest_fine_grained_blue_r2.json`, sourced from
`reports/nns_vit_endpoint_fine_grained_blue_figures_r2_20260919/`. Its blue-base positive panels
are display-only spatial reading views; signed companions and sealed raw arrays remain the
scientific reference.

For browsing or packaging, the flat PNG collection is
`reports/nns_vit_endpoint_all_figures_r1_20260919/`. It contains current Figures 1--40, signed
companions, all-layer/all-head pages, and the M2/M3/M4/M6 supplementary figures.

The paper-level RQ3 synthesis is
`BROAD_BUT_NON_UNIVERSAL_FAMILY_LEVEL_SUPPORT`. The frozen 12-of-12 omnibus C4a field
remains `NOT_SUPPORTED`. RQ4/C4b remains `NOT_SUPPORTED`.

Supplementary M1 results are integrated in the current bilingual drafts (Tables 6 and 7).
See `notes/endpoint_m1_evidence_matrix.md` and `docs/endpoint_m1_results_20260918.md`.
Earlier release/interpretation files record their historical wording; the current drafts follow
the Stage M broad-but-non-universal formulation.

Run the independent draft check from the repository root:

```bash
python3 papers/nns_vit_endpoint/scripts/validate_endpoint_scanner_draft.py
```
