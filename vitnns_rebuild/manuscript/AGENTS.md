# Manuscript working rules

## Language versions

- Keep Chinese manuscript text in `zh/` and English manuscript text in `en/`.
- Keep section structure, figure numbers, table numbers, numerical values, and scientific conclusions synchronized.
- Do not mix Chinese and English prose in one manuscript section file.

## Evidence discipline

- Do not invent data, parameters, references, or experimental settings.
- Bind every numerical statement to `notes/evidence_matrix.md` and a frozen source record.
- Use `[EVIDENCE REQUIRED]`, `[PARAMETER REQUIRED]`, or `[REFERENCE REQUIRED]` when a required item has not been verified.
- Keep C10 (`n=15`) and Stage D (`n=500`) separate; never pool them.
- Preserve C10 `NO_GO_NOVELTY`, `FAIL_SPATIAL`, `BLOCKED`, and `HOLM_BLOCKED_UNDERSPECIFIED`.
- Treat Stage E as post-D10 secondary/descriptive analysis and preserve its `MIXED` scientific-validity boundary.
- Treat Stage X2-MECH and Stage S findings as post-hoc exploratory.
- Do not use a non-significant contrast as evidence of equivalence.

## Terminology

- Write `target-matched patch-token baseline with uniform within-patch allocation` on first use. Do not shorten it to bare `matched`.
- Reserve `matched occlusion control` for the Stage S intervention control.
- Distinguish module-own-response learning images, fixed-clean-attention conditional Value-path attribution, complete-cut class-component maps, and full-model IG.
- Do not call the block-internal MLP branch a classifier head.
- Do not describe the fixed-clean-attention Value path as full Q/K/V attribution or full attention attribution.

## Writing style

- Use claim -> evidence -> concise boundary.
- Put measured facts before interpretation.
- Prefer direct subject-verb sentences and measurable statements.
- Avoid reviewer-facing prebuttals, repeated non-claim disclaimers, promotional adjectives, and automatic summary sentences.
- Keep one material caveat with a claim; move repeated boundaries to `limitations.md`.

## Figures

- Reference figures through `../figure_manifest.json`; do not duplicate or rerender the frozen figure package inside `manuscript/`.
- Treat display-normalized blue maps as spatial-shape views, not cross-panel magnitude measurements.
- Use raw signed arrays and source records, not PNG pixels, for numerical claims.
