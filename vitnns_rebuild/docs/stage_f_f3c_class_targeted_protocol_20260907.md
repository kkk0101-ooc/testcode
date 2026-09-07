# Stage F F3C class-targeted prospective study protocol（2026-09-07）

## 1. Status, question, and sequencing

This is the prospective F3C0 protocol freeze created only after terminal F3H. F3H observed all 12×12 heads and concluded `INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS`: the atlas contains no complete set of signed class-logit coupling coefficients, no no-omission/no-double-count path-accounting identity, and no class-logit conservation test.

Therefore F3C does **not** learn, tune, rank, or visually choose head weights and does **not** sum the 144 F3H head-direction maps. It asks a new question directly: which image regions support the clean predicted-class raw logit under a precisely defined all-block fixed-clean-attention conditional model, and how does that spatial ranking compare with four frozen reference methods?

Current authorization is `F3C0_PROTOCOL_FREEZE_ONLY`. F3C empirical computation and F4 remain paused. A F3C0 PASS may authorize only `F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY`.

## 2. Exact target and class-targeted NNS definition

For each image, use the verified recovery-v2 ViT-B/16 and canonical preprocessing. Resolve once

```text
y* = argmax_k logit_k(x)
```

and freeze `y*` for all five methods and every perturbation. Official truth and top-1 correctness are recorded; misclassified samples are retained. The baseline is normalized zero, corresponding to canonical RGB `(0.5,0.5,0.5)`, not a black image.

The new exact method ID is:

```text
conditional_all_blocks_fixed_clean_attention_class_logit_ig
```

Paper label: `class-targeted NNS (all-block fixed-clean-attention class-logit IG)`.

In one clean forward, capture the exact post-softmax attention tensor actually consumed by every block, including all 12 heads:

```text
A*_l = A_l(x), l = 0,...,11
```

Detach, clone, hash, and freeze all `A*_l`. Define the conditional model functional

```text
t_NNS(z) = logit_y*(F(z; A_0=A*_0, ..., A_11=A*_11)).
```

Compute input IG from normalized-zero `x0` to `x` for `t_NNS`, using Gauss–Legendre float32 schedule `[512,1024,2048,4096,8192]`, chunk 8, `atol=1e-4`, `rtol=0.01`, stopping at the first conservation pass and retaining every attempted step count. The native attribution is signed RGB `3×224×224`; RGB channels are summed with sign before any spatial projection.

At `x`, the all-block intervention must reproduce the clean raw logits with max absolute error `<1e-6`. IG must conserve

```text
t_NNS(x) - t_NNS(x0)
```

within the frozen tolerance. Q/K/softmax variation is cut by the fixed attentions; Value tensors, all native head concatenation and output projections, residual paths, MLPs, final norm, and classifier head remain live. Thus the class target is propagated through the model's own downstream mixing, not through hand-authored head weights.

This method is class-targeted and all-head, but conditional. It is not raw attention, not a sum of F3H head-direction maps, not full Q/K/V attribution, and not identical to unmodified full-model IG.

## 3. Five mandatory methods

Every pilot/formal sample must produce exactly these five methods:

1. `conditional_all_blocks_fixed_clean_attention_class_logit_ig` — new class-targeted NNS, signed, conditional class-logit functional.
2. `vit_grad_cam_block11_norm1` — ViT Grad-CAM, frozen class raw logit, nonnegative ReLU map, `blocks[11].norm1`, no algorithmic min-max.
3. `full_model_class_logit_ig` — unmodified full-model class-logit IG, signed.
4. `attention_rollout_mean_head_residual_all_blocks` — class-agnostic attention-flow baseline evaluated against the frozen class logit.
5. `direct_raw_cls_attention_last_block_head_mean` — class-agnostic block-11 raw CLS-attention mean baseline evaluated against the frozen class logit.

Methods 2–5 reuse their tested Stage C kernels and exact semantics. All five share image/model/preprocessing/class-index identity. Their explanation-function boundaries remain explicit: only methods 2 and 3 explain the unmodified class-logit functional directly; method 1 explains the conditional fixed-attention class-logit functional; methods 4 and 5 are class-agnostic rankings.

## 4. Prospective sample split

The machine-readable split is `manifests/stage_f_f3c_v1/sample_split.json`, generated before JPEG or F3C outcome access from the sealed D1 500-image registry. Exclude every sample already displayed in F3/S4/F3H, including failure slots, by exact selection orders:

```text
14, 44, 117, 145, 231, 281, 326, 352, 372, 380, 387
```

Sort the remaining rows by `sha256("STAGE_F_F3C_V1_20260907|" + sample_id)`, then by selection order. Freeze the first 5 as pilot and the next 50 as the formal cohort. The two sets are disjoint. No visual replacement is allowed.

Order 44 remains a viewed continuity/display anchor only. It is excluded from pilot, formal statistics, threshold tuning, and method selection.

## 5. Staged execution gates

- `F3C0`: protocol/config/schema/sample split and CPU-only validator; no JPEG/model/Torch/timm/GPU or attribution.
- `F3C1`: implement the new conditional method, all-block intervention, retention, runner, synthetic and real one-image engineering checks. No formal cohort access.
- `F3C2`: run the five frozen pilot samples and all five methods. Before the continuation decision, no JPEG/map/display inspection is allowed; use only identity, terminal status, conservation, parity, resources, and budget.
- `F3C3`: only after pilot PASS, run the 50-image formal cohort with immutable typed terminals.
- `F3C4`: CPU-only reconciliation, statistics, rendering, manual QC, false-pass audit, completion, commit/push/archive.

Unknown implementation errors remain incomplete and block. Scientific invalidities become typed `FAILED/NA` with no zero fill, interpolation, replacement sample, or dropped denominator. Pilot PASS requires 25/25 method units VALID, NNS endpoint parity and conservation PASS, full-model IG conservation PASS, complete hashes, and projected total cost within the cap.

## 6. Common map and display contracts

Quantitative common grid is 14×14 patch mass. Signed pixel methods sum RGB channels with sign and then sum non-overlapping 16×16 pixels. Grad-CAM, rollout, and raw attention retain their existing 14×14 semantics. Display resize is bilinear-only and never enters metrics.

For the main five-method spatial comparison, derive a nonnegative mass-one display distribution per method:

- signed methods: `P=max(sigma*M,0) / sum(max(sigma*M,0))`, where `sigma=sign(t(x)-t(x0))` for that method's functional;
- nonnegative methods: `P=M/sum(M)`.

Zero positive mass is typed NA. Within one sample, all five `P` maps use the shared unclipped color limit `[0, max_m max(P_m)]`; no method-specific gain, gamma, alpha, percentile clipping, or separate autoscaling is allowed. Overlays use one fixed alpha. Raw signed NNS and full-model IG positive/negative audit panels are retained separately; no negative map is fabricated for nonnegative methods.

## 7. Evaluation and statistics

All rankings are evaluated against the unmodified frozen predicted-class raw logit. This common evaluation target does not change method 1's conditional conservation target.

Primary common resolution is 14×14 patches, normalized-zero replacement, 21 fractions `0,0.05,...,1`, actual-fraction trapezoid AUC, and five shared hash-derived tie permutations. Define

```text
F_m = 0.5 * (InsertionAUC_m - DeletionAUC_m)
Delta_m = F_NNS - F_m.
```

The formal statistical unit is image (`n=50`). Report every per-image value, mean, median, sample SD, positive count, and a 10,000-draw image-bootstrap percentile 95% CI. The single primary contrast is class-targeted NNS versus ViT Grad-CAM. The three other NNS contrasts (full-model IG, rollout, raw CLS attention) are secondary and Holm-corrected as one family. Exact p-value/permutation machinery must be frozen in F3C1 before pilot outcomes; F3C0 does not invent a winner threshold. Report effect estimates even when not significant and never choose a display method from these outcomes.

Fine 4×4-cell fidelity for the two signed pixel-native methods is secondary. Spatial similarity, entropy, and hotspot overlap are descriptive only. No object-localization metric or claim is allowed without independently sealed segmentation/bounding-box ground truth. Pilot images and order 44 never enter formal `n=50` statistics.

## 8. Budget and retention

F3C1 must measure a one-image five-method end-to-end profile before pilot. The provisional launch caps are 8 H20-GPU-hours total, 0.30 H20-GPU-hour per image group, 8 devices, 16 GiB peak reserved per process, and 8 GiB retained artifacts. There is no automatic waiver. F3C2 must replace the provisional projection with pilot telemetry; if any cap is exceeded, stop before F3C3 and issue a prospective amendment rather than reducing methods or samples.

Each sample×method unit has immutable status/manifest/artifact hashes and explicit `VALID/FAILED/NA`. Retain native signed maps for methods 1/3, native Grad-CAM, normalized attention maps, class target, fixed-attention hashes, parity/conservation attempts, fidelity curves, resources, and sufficient provenance. Display arrays never feed metrics.

## 9. Claims and terminal authorization

F3C may support a class-logit fidelity statement for the new conditional method and transparent comparisons among the five frozen methods. It cannot retroactively repair C10, replace Stage D's `GO_FIDELITY_SCOPE`, pool C10 `n=15` with Stage D `n=500` or F3C `n=50`, or turn F3H head diversity into numerical class weights.

F3C terminal workflow PASS may coexist with a negative or inconclusive comparative result; scientific outcome and workflow completeness must be separate fields. Only terminal F3C completion may authorize `F4_ONLY`.
