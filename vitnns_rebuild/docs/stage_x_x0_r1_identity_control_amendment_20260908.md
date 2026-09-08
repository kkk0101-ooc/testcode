# Stage X X0-R1 identity-control correction amendment

Date: 2026-09-08. Amendment ID:
`STAGE_X_X0_R1_IDENTITY_CONTROL_CORRECTION_20260908`.

## Trigger and status of the original X0

The original X0 completed and was published at commit
`bc4b18580f88fc2e82e1a3c686b29092e9ad7b40` and
`refs/archive/stage-x-nns-v2-x0-completion`. During the required X1
implementation audit, after an uncommitted synthetic-only draft and its tests
had started but before any model/checkpoint/JPEG access or X1 terminal root was
created, a logical contradiction was found:

1. X0 defines the primary NNS-v2 scientific map as unmodified-model input IG.
2. X0 also registers full-model IG with the same target, baseline, path,
   quadrature, convergence schedule, and RGB reduction as a comparator.
3. It then requires NNS-v2 to be strictly superior to every primary comparator.

The NNS-v2 total map and full-model IG control are therefore the same
mathematical quantity. Strict superiority against that identity control is
impossible except through implementation drift, randomness, or numerical error.
The original X0 validator did not reject this false-pass condition. Its
terminal artifacts and archive ref remain immutable, but its authorization is
superseded as `BLOCKED_SUPERSEDED_IDENTITY_COMPARATOR_CONTRADICTION`. No X1
implementation is accepted from the interrupted draft.

## Prospective correction

No method, model, dataset, sample, endpoint, margin algorithm, resource cap, or
historical result is removed. Only comparator roles and the truth table change:

- `full_model_integrated_gradients` becomes
  `INCLUDED_PRIMARY_IDENTITY_CONTROL`.
- The competitive class-targeted superiority family is exactly ViT Grad-CAM,
  Transformer Attribution, AttnLRP, RISE, and Occlusion.
- Holm correction applies to exactly these five competitive comparisons.
- NNS-v2 and the full-model IG identity control must have the same target,
  baseline, path, steps, quadrature nodes/weights, input tensor, signed
  `[3,224,224]` output, and RGB reduction. Every valid sample must satisfy
  elementwise `atol=1e-6`, `rtol=1e-6` and identical selected step count.
- Any identity mismatch is `NO_GO_METHOD_VALIDITY`; it can never be counted as
  a superiority result.
- `GO_BROAD_BENCHMARK_SCOPE` requires superiority against all five competitive
  comparators and identity-control agreement with full-model IG.

The strongest permitted wording becomes:

> Across the preregistered benchmark of models, datasets, methods, and metrics,
> NNS-v2 outperformed the evaluated competitive baselines while matching the
> full-model Integrated Gradients identity control for total input attribution.

The draft was removed in full and is not part of X0-R1. This amendment is
frozen before any accepted X1 implementation or Stage X empirical outcome.
X0-R1 itself is CPU-only and permits no image/model/checkpoint access, download,
Torch/timm import, or CUDA/GPU use. Its access audit records the interrupted
synthetic draft attempt separately. X0-R1 PASS may authorize only a fresh
`X1_NNS_V2_IMPLEMENTATION_ONLY`.
