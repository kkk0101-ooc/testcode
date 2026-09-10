# X2-MECH P0-R1 Access-Boundary Amendment

Date: 2026-09-10

## Prior P0 status

The prior P0 session terminated as `BLOCKED_PROCEDURAL_ACCESS_SCOPE_CONTRADICTION`. The task required implementation of validators and tests while incorrectly prohibiting access to the adjacent source code and tests necessary to implement and validate that work. This was a procedural access-scope contradiction, not a scientific failure.

The prior session:

- did not read numeric values from any `.npy` or `.npz` file;
- did not read per-route stability, faithfulness, ROAD, or sanity outcome rows;
- did not read JPEG files or checkpoints, load a model, or access a GPU;
- did not create any P0 evidence or report, commit, or ref.

The X2 conclusion `NO_GO_BROAD_CLAIM_DEVELOPMENT` is unchanged.

## Amended access boundary

`source-code access != empirical-outcome access`

P0-R1 may read the adjacent source code, schemas, configuration contracts, tests, and documentation required to implement and test the protocol-freeze validators. This permission does not authorize reading empirical source maps or empirical outcome values or rows.

The prohibitions on reading `.npy`/`.npz` numeric contents, per-route stability rows, faithfulness rows, ROAD rows, sanity outcome rows, JPEG files, and checkpoints remain in force. Model loading and GPU/CUDA access also remain prohibited. Synthetic fixtures and metadata-only structural checks are allowed only when they cannot disclose empirical outcomes.

This amendment changes only the P0 access boundary. It changes no method, sample, transform, metric, threshold, statistic, or downstream claim. It neither reopens nor modifies any X2 scientific conclusion.
