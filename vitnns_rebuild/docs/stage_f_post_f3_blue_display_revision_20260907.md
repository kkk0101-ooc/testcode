# Stage F post-F3 blue-display visual revision

Date: 2026-09-07. Scope: display-only replacement modules for the qualitative presentation layer. Scientific authority: none. Upstream authority: sealed F3 r7 at refs/archive/stage-f-v1-f3-completion.

## 1. Motivation

The sealed F3 r7 package is scientifically and provenance-valid, but owner review rejected its white-background signed-map presentation as visually unsuitable and reported that the matched baseline was difficult to read. This revision improves only visual presentation. It does not amend selection, source maps, failure handling, statistics, decisions, or authorization.

The style reference is 论文1/参考文章.pdf, especially Fig. 7, Fig. 11, Fig. 13, and Fig. 14: deep-blue low-response backgrounds, sparse warm highlights, compact image matrices, and minimal in-panel prose. No reference image, text, datum, or conclusion is copied.

## 2. Frozen inputs

- F3 r7 completion and archive ref;
- F3 r7 selection, rule mapping, panel, cross-layer panel, scale, render, and source manifests;
- the exact JPEGs and signed maps already bound by F3 r7;
- the exact Figure 5/S1/S2/S4 samples and Figure 6 median-effect sample;
- the exact orders 117 and 352 typed failures.

The revision must independently re-hash every JPEG/map against the r7 source record. It may not open a model or checkpoint, import Torch/timm, use CUDA/GPU, or execute forward/backward/IG/attribution/fidelity.

## 3. Blue-display transform

Each valid signed 56x56 source map is converted to a presentation-only magnitude image:

1. magnitude = abs(raw_signed_map);
2. shape_normalized = magnitude / max(magnitude) within that one panel;
3. display = shape_normalized squared;
4. resize 56x56 to 224x224 with bilinear interpolation;
5. render with the frozen deep-blue to blue to cyan to yellow to orange to red colormap.

There is no percentile clipping, threshold deletion, smoothing, max pooling, or source-map modification. Zero remains deep blue; the panel maximum maps to red. Raw signed-map hash, raw maximum magnitude, normalized 56x56 hash, resized 224x224 hash, gamma, resize parameters, and output hashes must all be recorded.

Because each panel is independently shape-normalized, color intensity in this revision is not comparable between NNS and matched, between coordinates, or between samples. The purpose is only to reveal spatial support and hotspot organization. The r7 signed shared-scale maps remain the authoritative audit view for within-coordinate magnitude comparison.

## 4. Layout

- Figure 5/S1/S2 cases: compact 2-row layout with Original, NNS, Matched baseline, absolute difference, two overlays, and fidelity curves. Titles are short; filenames and protocol prose stay in captions/reports.
- Figure 6/S4: Original plus a compact 2x4 matrix, NNS first and Matched baseline second, coordinates as column headings. Difference views remain available in r7 and are omitted from the blue matrix.
- Failed/NA cells use the same deep-blue background with explicit white failure text and typed failure code. No value is imputed.
- Panel letters are inside their panels; row/column labels remain outside matrices; no long prose is drawn inside figures.

## 5. Outputs and boundaries

Outputs are evidence/stage_f/f3_visual_r3_recovery_v2_20260907/formal/ and reports/stage_f_f3_visual_r3_20260907/. Visual r1 and r2 are retained as blocked attempts.

The revision produces Figure 5, Figure 6, S1, S2, and S4 replacement modules plus source/transform/render/review inventories, validation, false-pass audit, and completion. It does not modify the F3 r7 archive ref and does not start F4.

Grad-CAM/Figure 7 is outside this revision: no sealed Grad-CAM attribution exists, and generating it requires a new prospective compute protocol with model access. This revision must not fabricate or compute it.
