# Endpoint manuscript working rules

## Language versions

- Keep Chinese files in `zh/` and English files in `en/`.
- Keep section structure, figures, tables, numbers, and conclusions synchronized.
- Do not import content from `../nns_vit_path/` without checking that the attribution definition
  and evidence source apply to Endpoint.

## Evidence discipline

- Bind numerical claims to `notes/endpoint_scanner_paper_evidence_matrix_r1.md` and sealed sources.
- The base Endpoint object is $x\odot\nabla_xs(x)$, numerically Gradient × Input for the same
  scalar target. Do not present the base formula as novel.
- Separate Identity, Correctness, and Utility. Closure and JVP results do not establish finite
  intervention utility.
- State RQ3 as overall supported with one registered boundary exception; retain the frozen omnibus
  C4a `NOT_SUPPORTED` field.
- Preserve RQ4/C4b `NOT_SUPPORTED` and the stronger registered regional-Taylor result.
- Do not publish cross-model Endpoint–Path claims because ViT-B Path-IG calibration failed.

## Writing and figures

- Use claim -> evidence -> concise boundary.
- Mark missing information with `[EVIDENCE REQUIRED]`, `[PARAMETER REQUIRED]`, or
  `[REFERENCE REQUIRED]`.
- Reference figures through `figure_manifest.json`; do not modify or duplicate the sealed package.
- Treat display-normalized maps as spatial-shape views; numerical claims use raw records.
