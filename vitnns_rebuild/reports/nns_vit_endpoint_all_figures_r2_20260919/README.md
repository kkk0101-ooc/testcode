# Endpoint-NNSViT organized R2 figure package

[Gallery](gallery.html) · [Figure index](figure_index.md) · [逐图分析](figure_analysis.md)

This is the organized successor to `nns_vit_endpoint_all_figures_r1_20260919`. The archived R1 root is unchanged.

- `figures_png/`, `figures_pdf/`, `figures_svg/`: Figure 1–40 only.
- `supplement/signed/`: signed companions for the black-bear blue-base main views.
- `supplement/all_heads/`: full block-by-head pages, separated by model.
- `supplement/m2/`, `m3/`, `m4/`, `m4_review/`, `m6/`: supplemental families.
- `real_outputs/`: individual positive display panels used by redesigned real-map figures.
- `source_arrays/`: exact display arrays extracted from sealed sources plus their receipt.

The fixed presentation sample is `ILSVRC2012_val_00005415.JPEG` (selection order 14, class 297, `n02134418`, black bear). Main real-map panels use the R10 blue-base rule `max(L,0)` with per-panel P99 scaling. Signed companions preserve the complete spatial sign pattern.

No model forward, checkpoint load, bootstrap, hypothesis test, or decision recomputation is performed.

Validation:

```bash
CUDA_VISIBLE_DEVICES='' ../.conda/envs/vitnns/bin/python scripts/validate_endpoint_all_figures_r2.py
```
