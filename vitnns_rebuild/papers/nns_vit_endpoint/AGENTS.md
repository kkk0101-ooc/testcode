# Endpoint manuscript working rules

## Language versions

- Keep Chinese files in `zh/` and English files in `en/`.
- Keep section structure, figures, tables, numbers, and conclusions synchronized.
- Do not import content from `../nns_vit_path/` without checking that the attribution definition
  and evidence source apply to Endpoint.

## Evidence discipline

- Bind current numerical claims to `notes/revision_v1_evidence_matrix.md` and sealed sources;
  publication-specific display evidence lives in `publication_r2_20260920/evidence/`.
  `notes/endpoint_scanner_paper_evidence_matrix_r1.md` retains the original release mapping.
- The base Endpoint object is $x\odot\nabla_xs(x)$, numerically Gradient × Input for the same
  scalar target. Do not present the base formula as novel.
- Separate Identity, Correctness, and Utility. Closure and JVP results do not establish finite
  intervention utility.
- In the revised RQ order, state RQ5a as broad but non-universal family-level support (11/12);
  retain the frozen all-family omnibus C4a `NOT_SUPPORTED` field. Report supplementary M1
  comparisons separately from the original confirmatory analysis.
- Preserve RQ5b/C4b `NOT_SUPPORTED` and the stronger registered regional-Taylor result.
- Do not publish cross-model Endpoint–Path claims because ViT-B Path-IG calibration failed.

## Writing and figures

- Use claim -> evidence -> concise boundary.
- Mark missing information with `[EVIDENCE REQUIRED]`, `[PARAMETER REQUIRED]`, or
  `[REFERENCE REQUIRED]`.
- Reference figures through `figure_manifest.json`. The current portable publication is
  `publication_r2_20260920/`, containing both languages and 22 main figures.
- The user's 2026-09-20 instruction authorizes local copies of required supplementary figures
  inside that publication directory. Preserve sealed source packages and record copy hashes.
- Use blue positive learning-image displays with explicit display rules and local signed/absolute
  counterparts. Include the original input in every empirical map page.
- Treat display-normalized maps as spatial-shape views; numerical claims use raw records.
