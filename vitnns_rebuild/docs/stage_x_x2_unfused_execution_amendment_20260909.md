# Stage X X2 native-unfused execution clarification

Date: 2026-09-09. Amendment ID:
`STAGE_X_X2_NATIVE_UNFUSED_EXECUTION_CLARIFICATION_20260909`.

This prospective clarification is sealed before any X2 development image is
decoded and before any X2 prediction, attribution, faithfulness, stability, or
method-difference outcome is computed. It does not modify any X0, X0-R1, X1,
or X1-R1 terminal or evidence byte.

The `forward_hook_parity` requirement in
`manifests/stage_x_model_data_inventory_v1.json` means parity between the
unhooked and hooked/traced forms of the same native-unfused evaluation graph.
That comparison remains strict: every raw logit must have maximum absolute
difference `<1e-6`, and the target class must be unchanged.

Fused scaled-dot-product attention and native-unfused attention are two
floating-point execution kernels for the same model parameters, but they do not
have bitwise-identical reduction order. X1-R1 measured a maximum absolute logit
difference of `1.6927719116210938e-05` with unchanged top-1 class. X2 therefore
records fused-versus-unfused drift as a separate diagnostic. Its prospective
diagnostic acceptance bound is maximum absolute logit difference `<=2e-5` and
identical top-1 class on the X1-R1 continuity image. This diagnostic is not an
attribution identity test and cannot satisfy or relax hooked/unhooked parity,
NNS-v2/full-model-IG identity, endpoint parity, completeness, or target gates.

The only canonical X2 scientific execution path is native-unfused timm 0.9.16
evaluation mode. Every NNS-v2 and comparator adapter, every clean and
transformed input, and every perturbation uses that path. Fused outputs are not
used to select samples, checkpoints, targets, gamma, margins, or effects. A
failure of same-path hooked/unhooked `<1e-6` is
`NO_GO_METHOD_VALIDITY`; a fused diagnostic outside the bound blocks X2 before
development outcome access as `NO_GO_METHOD_VALIDITY`.
