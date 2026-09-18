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
copied into this paper root.

The paper-level RQ3 synthesis is
`OVERALL_SUPPORTED_WITH_ONE_REGISTERED_BOUNDARY_EXCEPTION`. The frozen 12-of-12 omnibus C4a field
remains `NOT_SUPPORTED`. RQ4/C4b remains `NOT_SUPPORTED`.

Run the independent draft check from the repository root:

```bash
python3 papers/nns_vit_endpoint/scripts/validate_endpoint_scanner_draft.py
```
