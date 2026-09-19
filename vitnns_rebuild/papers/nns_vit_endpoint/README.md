# NNS-ViT Endpoint Scanner paper

This is the independent bilingual manuscript project for Endpoint scanning and structured
decomposition. It does not share draft, evidence-note, QC, reference, script, or figure-manifest
directories with the path-integrated NNS-ViT paper.

## Reading order

1. `zh/outline.md`
2. `zh/full_draft.md` or `en/full_draft.md`
3. `notes/endpoint_scanner_stage_m_evidence_matrix_r3.md`
4. `figure_manifest.json`
5. `references/endpoint_scanner_references_r1.md`
6. `qc/endpoint_scanner_full_draft_r3_defensive_writing_audit.md`
7. `qc/endpoint_scanner_r3_completion_audit.md`

The current manuscript binding is `figure_manifest.json`, generated from the sealed organized
`ENDPOINT-ALL-FIGURES-R2` package at
`reports/nns_vit_endpoint_all_figures_r2_20260919/`. It covers Figures 1--40 and all 52
supplemental records used by the paper, including signed companions, M2--M6 figures, and the
complete all-head pages. Figures are referenced rather than copied into this paper root.

The organized package keeps Figures 1--40 under separate
PNG/PDF/SVG directories, moves signed companions and all-layer/all-head pages into
`supplement/`, and provides the requested bear-sample blue-base revisions plus source/display
receipts. The sealed class is ImageNet 297 / `n02134418` (sloth bear), so the paper-facing
labels use that exact class name.
Earlier source packages and the flat R1 package remain archived unchanged.

Within Stage M's RQ5 behavioral calibration, the C4a synthesis is
`BROAD_BUT_NON_UNIVERSAL_FAMILY_LEVEL_SUPPORT`. The frozen 12-of-12 omnibus C4a field
remains `NOT_SUPPORTED`, and the C4b spatial-utility field remains `NOT_SUPPORTED`.

Supplementary M1 results are integrated in the current bilingual drafts (Tables 6 and 7).
See `notes/endpoint_m1_evidence_matrix.md` and `docs/endpoint_m1_results_20260918.md`.
Earlier release/interpretation files record their historical wording; the current drafts follow
the Stage M broad-but-non-universal formulation.

Run the independent draft check from the repository root:

```bash
python3 papers/nns_vit_endpoint/scripts/validate_endpoint_scanner_draft.py
```
