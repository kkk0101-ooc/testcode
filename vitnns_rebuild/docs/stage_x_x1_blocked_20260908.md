# Stage X X1 blocked record: pinned comparator engines unavailable

Date: 2026-09-08. X1 attempt ID: `STAGE_X_X1_V1_20260908`.

X0-R1 passed and authorized `X1_NNS_V2_IMPLEMENTATION_ONLY`. X1 then ran its
dependency preflight before creating an accepted implementation or accessing an
image, checkpoint, model, CUDA device, or empirical outcome. The preflight
found two required competitive-primary engines unavailable locally:

- Transformer Attribution / Chefer et al. official source at commit
  `c3e578f76b954e8528afeaaee26de3f07e3fe559` is not vendored or installed.
- AttnLRP/LXT official source at commit
  `f04344177551bfcc3665e88afe96501f43fe560b` is not vendored and the `lxt`
  package is not installed.

The current environment has the `pytorch_grad_cam` package, and the repository
has tested local kernels for ViT Grad-CAM and full-model IG. That does not cure
the two missing official engines. X0-R1 requires all five competitive methods
and explicitly forbids substituting an invented implementation. X1 also has a
zero-download hard cap. Dropping either method, downloading code in this
attempt, or replacing it with a hand-authored approximation would violate the
frozen comparator identity.

Therefore X1 terminates as `BLOCKED_MISSING_PINNED_COMPARATOR_ENGINE`. This is
an infrastructure/dependency blocker, not an NNS-v2 numerical or scientific
failure. No X1 method validity claim is made. The pre-amendment synthetic draft
described by X0-R1 remains deleted and is not accepted as X1 implementation.
X2, X3, F3-Final, and Stage G remain unstarted.

A future owner-authorized recovery must be prospective and may authorize only
code dependency materialization, with exact repository/commit/license hashes
and a byte cap, before restarting X1 in a fresh root. It must not authorize
dataset or checkpoint download and must not weaken or remove the two comparator
gates.
