# From NNS to Vision Transformers — Endpoint manuscript

The current revision implements 修改意见/修改意见v1.md using existing sealed Endpoint evidence. Read the [Chinese manuscript](zh/full_draft.md), [English manuscript](en/full_draft.md), or [ten-figure gallery](../../reports/nns_vit_endpoint_paper_revision_v1_20260920/gallery.html).

The manuscript has ten main figures, seven main tables, synchronized RQ1–RQ5 sections, two mathematical propositions, and descriptive absolute-AUC/reference-sensitivity tables. The [Chinese supplement](zh/supplement_revision_v1.md) and [English supplement](en/supplement_revision_v1.md) retain full coverage, detailed mechanism tables, original decisions, and links to all 40 former main figures plus 52 existing supplemental records.

The revised figure package is reports/nns_vit_endpoint_paper_revision_v1_20260920/, with PNG/PDF/SVG exports, bilingual captions, source/display records and a gallery. The user-named reports/nns_vit_endpoint_all_figures_r2_20260919/ remains the immutable source package. The active binding is [figure_manifest.json](figure_manifest.json); [figure_manifest_all_figures_r2.json](figure_manifest_all_figures_r2.json) preserves the original 40-figure binding.

Earlier full_draft_m*, release/interpretation files, and full_draft_r3_archive.md are historical versions, not the current manuscript. Endpoint has a different attribution definition and evidence basis from ../nns_vit_path/.

The original C4a synthesis is broad but non-universal family-level support (11/12). Its all-family omnibus C4a and spatial C4b fields remain NOT_SUPPORTED. Supplementary analyses do not replace those decisions. Suggested random/untrained controls, target robustness and matched-target experiments are documented as missing evidence and prospective work, not claimed as completed.

Review the [revision protocol](notes/revision_v1_protocol.md), [evidence matrix](notes/revision_v1_evidence_matrix.md), [writing audit](qc/revision_v1_writing_audit.md), and [completion audit](qc/revision_v1_completion_audit.md).

From the repository root, run these commands in order:

    CUDA_VISIBLE_DEVICES='' python3 papers/nns_vit_endpoint/scripts/analyze_revision_v1.py
    CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python papers/nns_vit_endpoint/scripts/render_revision_v1.py
    python3 papers/nns_vit_endpoint/scripts/bind_revision_v1.py
    CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python papers/nns_vit_endpoint/scripts/verify_revision_v1_display.py
    CUDA_VISIBLE_DEVICES='' python3 papers/nns_vit_endpoint/scripts/validate_endpoint_scanner_draft.py

The renderer uses NumPy, Matplotlib and Pillow; analysis and validation use the Python standard library. These commands do not run model inference. Author/affiliation/archive fields and provisional bibliography metadata still require author completion before submission.
