# NNS ViT unified paper figures R1

This is a new presentation root. It does not modify or supersede upstream Stage A-F, X, Y, or S evidence.

## Contents

- figures_png/: 14 review-ready raster figures.
- figures_pdf/: 14 publication-layout PDF figures.
- figures_svg/: 14 editable vector figures.
- figure_index.md: Figure 1-14 filename index.
- captions_zh.md and captions_en.md: draft captions.
- figure_analysis.md: numbered interpretation and evidence boundaries.
- render_manifest.json and input_inventory.json: output and source bindings.
- cpu_access_audit.json: CPU-only rendering audit.
- manual_visual_qc.json: visual-review checklist.
- validation_report.json, false_pass_mutation_audit.json, whole_root_inventory.json, and terminal_receipt.json: package checks.

## Regeneration

Set CUDA_VISIBLE_DEVICES to the empty string, run scripts/render_nns_vit_unified_paper_figures_r1.py with --replace-own-root, and then run scripts/validate_nns_vit_unified_paper_figures_r1.py with ../.conda/envs/vitnns/bin/python.

The renderer reads sealed CSV, JSON, JPEG, and NumPy sources. It imports no Torch or timm, loads no model or checkpoint, and performs no new attribution, intervention, or statistical inference.

Figures 8-10 use presentation transforms for spatial-shape inspection. Quantitative claims remain tied to the formal tables used in Figures 4-7 and 11-14.
