# User-feedback completion audit — 2026-09-20

| User requirement | Delivered evidence | Verification |
|---|---|---|
| 1. Ten figures are too few | 22 main figures, 66 PNG/PDF/SVG exports, 7 main tables | Manifest and each manuscript's image references |
| 2. Paper and required images in one directory | publication_r2_20260920 contains zh/en text, figures, 92 supplementary pages, reference list and data | Every main/supplement link resolves inside the directory |
| 3. Formal paper style, learn local articles | Both manuscripts rewritten with problem → representation → method → observation → intervention structure | Read NNS abstract/introduction/method/conclusion and ViT visual-analytics abstract; writing audit |
| 4. Ugly text-only concept / sparse points / forest lines | New explanatory diagrams 1–7; former fidelity points moved to Table 2; former association forest moved to Table 3 | Figure content review; numerical table values retained |
| 5. Poor examples, cannot see source/features | Recognizable mantis, sloth bear and little blue heron; originals on every empirical map page; larger paired maps; actual forward feature maps | Visual review; source image hashes; layer channels and targets in computation records |
| 6. Missing overview and learning-image generation figures | Figure 1 overview; Figure 2 neuron/image correspondence; 3 initialization; 4 derivative combination; 5 patch embedding; 6 attention; 7 residual | Worked four-coordinate example independently checked; no copied reference image used as a result |
| 7. Missing Fig-13-style layers | Figures 9–10 show all twelve blocks in both models, learning-image/actual-feature pairs; Figure 11 shows modules at B0/5/11 | Stored native activation arrays and selected channel identities |
| 8. Missing Fig-14-style comparison | Figures 19–20: original, Endpoint, patch-max display, Grad-CAM, rollout, raw attention for three images and two models | Actual six CPU image-model computations; Endpoint/native graph parity; shared input/model target where applicable |
| Repeated 8. Blue high-intensity form | Deep blue/cyan/yellow/red positive learning-image views; square-root display scaling; signed/absolute companion Figure 22 | 172 panel receipts with raw hashes, scales and display transforms |

Scientific safeguards are part of interpretation, not release prose: positive RGB-sum displays are not complete signed maps; patch-max interpolation is labeled as a display variant; Grad-CAM and Endpoint share GT targets while raw attention/rollout are class-agnostic; CNN CAM is not invented for CLS-readout ViTs. Quantitative findings remain tied to prior full cohorts, and display-selected samples are not presented as population evidence.

Numerical display computation: source checkpoint and image hashes verified; original class Endpoint arrays crosschecked with CPU outputs. CPU/GPU differences are retained as precision diagnostics. The implemented attention capture is independently compared with the unmodified native CPU graph at relative L2 < 1e-7. This is a display study, not a new scientific-stage PASS or population test.

Initial generated variants are retained under evidence/initial_display_attempt and evidence/mantis_class_only_attempt. The final displayed channels use spatial variance rather than overall activation energy to avoid selecting nearly constant offsets. No image values or formal scientific outcomes were edited to improve appearance.

Final machine checks are in validation_report.json; visual review binds the exact PNG hashes. The complete current paper is mirrored at the established zh/full_draft.md and en/full_draft.md entry points, with all images linked into this package. The root README identifies this revision as current.

Independent comparison verification: all six tasks reproduce native Endpoint exactly; maximum Grad-CAM absolute difference is 2.80e-9 and rollout difference 1.05e-17. Four early/late layer pairs also reproduce selected channel, forward feature and energy Endpoint. The portable package validator recomputes 165 stored-array display panels and checks every local figure link.

Final local validation: 1,261 checks pass with zero errors; all 22 figures and 92 supplementary copies have verified hashes and local references. Main texts share equations, table data, section numbers and figure order. Six native comparison recomputations and four layer-pair checks pass.

Figure 20 was revised after user feedback. Orders 4/14/15 were excluded by instruction. The remaining 13 retained ViT-B maps were ranked before any Grad-CAM or attention output was read, using effective-patch coverage and three Endpoint/input-detail readability metrics. Orders 1/13/7 were sealed, then comparison methods were computed. Visual review checked rendering only and did not replace cases by comparison-method appearance.
