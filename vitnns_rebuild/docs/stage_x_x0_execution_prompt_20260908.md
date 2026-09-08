# Next execution prompt: X1_NNS_V2_IMPLEMENTATION_ONLY

Work in `/data01/home/yeziqi.831/nxtrec_wksp/testcode/vitnns_rebuild`. Execute
only Stage X X1, token `X1_NNS_V2_IMPLEMENTATION_ONLY`. First verify the X0
completion commit and `refs/archive/stage-x-nns-v2-x0-completion`, all X0
payload hashes/digests, local/tracking/remote equality, clean worktree/index,
and clean LFS. Preserve C10, D10, E4, all Stage F roots and refs read-only.

Implement exactly the NNS-v2 definition in `docs/stage_x_protocol.md` and
`config/stage_x.json`: unmodified evaluation-mode predicted-class raw logit;
straight normalized-zero path; signed `[3,224,224]` input IG; complete-cut
pixel-to-component `K^C[i,j]`; native per-head concat/output-projection
accounting; the two residual branch cuts; MLP/neuron/layer views; original
autograd softmax, LayerNorm, GELU, and multiplicative interactions; frozen
Gauss-Legendre schedule and tolerances. Layer views must not be summed. Bias and
constant embedding direct path terms must be recorded as zero while remaining
active in downstream Jacobians. A detached/frozen-attention/surrogate graph is
not primary NNS-v2. If input completeness and every architectural cut cannot
both pass, publish `NO_GO_CLASS_LOGIT_COMPLETENESS`; do not rename input IG.

Use only analytic/synthetic fixtures and the X0 prior-access ledger's Stage C
audit IDs/order-44 continuity image. Do not open any confirmatory candidate. Do
not download datasets or checkpoints. Do not start X2, X3, F3-Final, or Stage
G. X1 may use GPU only inside the X1 hard caps after a metadata-only preflight.

Build adapters and contract tests for every `INCLUDED_PRIMARY` comparator, but
do not run a benchmark. Each adapter must prove official commit/license
binding, exact clean raw-logit class target, native output semantics, model
compatibility, deterministic defaults/layer choices, and no display-to-metric
path. Deferred methods remain excluded. Class-agnostic references remain
secondary.

Required tests include strict JSON/schema/canonicalization; native forward and
hook parity; analytic linear, residual, multiplicative, softmax, LayerNorm, and
bias fixtures; per-pixel and scalar reconstruction at every cut; head and
branch sum identities; sub-patch position discrimination; target swap; positive
and negative deletion semantics; classifier-to-input randomization; display
type isolation; typed numerical/infrastructure failures; atomic completion;
tampered source/ref/model/method/sample/budget rejection. Record all quadrature
attempts and resource telemetry.

X1 terminal PASS may authorize only `X2_NNS_V2_DEVELOPMENT_AND_FUTILITY_ONLY`.
Commit and push one X1 delivery commit and a new archive ref only after focused
tests, source-recomputing validation, false-pass audit, whole-root inventory,
pending-to-terminal completion, and post-publication Git/LFS verification. Stop
after X1; do not run development images.
