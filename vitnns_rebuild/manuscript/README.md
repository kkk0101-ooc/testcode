# NNS-ViT manuscript

This directory contains the first bilingual manuscript draft based on the R10-R4 outline and frozen 35-figure package. The English and Chinese versions use the same section structure, figure numbers, numerical results, and evidence boundaries.

Ready-to-read deliverables:

- NNS_ViT_论文初稿_中文版.md

## Reading order

1. `notes/scientific_story.md`
2. `notes/evidence_matrix.md`
3. `zh/` or `en/` in section order
4. `references/literature_notes.md`
5. `qc/drafting_status.md`

The current version is a scientific manuscript draft, not a submission-ready package. Author names, affiliations, target journal formatting, final bibliography metadata, and a small number of explicitly marked reference gaps remain to be supplied or verified. No new experiment or sample selection was performed during drafting.

The Endpoint Scanner extension has its own sealed 28-figure package at
`reports/nns_vit_endpoint_unified_paper_figures_r2_20260918`. Use
`endpoint_figure_manifest.json` and `notes/endpoint_scanner_figure_mapping_r1.md` for the
Endpoint main-text/supplement split; do not replace or mutate the earlier R10-R4 manifest.

The independent Endpoint Scanner manuscript R1 is now available as:

- `../docs/论文大纲_NNS_ViT_Endpoint_R1_20260918.md`;
- `zh/endpoint_scanner_full_draft_r1.md`;
- `en/endpoint_scanner_full_draft_r1.md`;
- `notes/endpoint_scanner_paper_evidence_matrix_r1.md`;
- `qc/endpoint_scanner_full_draft_r1_defensive_writing_audit.md`;
- `references/endpoint_scanner_references_r1.md`;
- `scripts/validate_endpoint_scanner_draft.py` and
  `qc/endpoint_scanner_draft_validation_report_r1.json`.

This Endpoint draft preserves two simultaneous RQ3 records: the paper-level synthesis is
`OVERALL_SUPPORTED_WITH_ONE_REGISTERED_BOUNDARY_EXCEPTION`, while the frozen 12-of-12 omnibus
C4a field remains `NOT_SUPPORTED`.
