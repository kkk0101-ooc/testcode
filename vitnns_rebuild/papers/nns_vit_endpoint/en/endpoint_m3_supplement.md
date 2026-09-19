# Endpoint M3: NNS fidelity and stratified token-channel atlas

## Methods and execution status

Explicit layerwise NNS learning-image propagation and `x * grad` were compared on five compatible network families using 100 random inputs per float64/float32 condition, yielding 5,000 target-level comparisons. All passed; the maximum relative L2 error was `2.427e-07` and the maximum tolerance ratio was `1.243e-03`. Analytic JVP, autograd JVP/VJP, and central finite differences passed for 88 directions across 11 ViT operators. The maximum analytic-autograd relative error was `3.431e-16`; the worst best-step finite-difference relative error was `1.632e-10`.

The two models shared 100 frozen images. Each model-image audited 12×16×32=6,144 post-block token-channel scalars, for `1,228,800` target records. The stratified audit covers 0.338% of all post-block token-channel units in ViT-B/16 and 0.677% in DeiT-S. The `0` zero-gradient records propagate NA to undefined ratios. Of 57,600 frozen float32 finite-difference conditions, 36 missed the threshold and remain as the initial FAIL evidence. The published recovery rule selected all 10 affected tasks; their float64 multi-step review passed, yielding `PASS_FLOAT64_RECOVERY_WITH_RETAINED_FLOAT32_FAILURE`.

## Equivalence proposition and conditions

Define `L(u;x)=J_(u,x)(x) diag(x)`. For `v=f(u_1,...,u_r)`, the chain rule gives `L(v;x)=sum_a J_(f,u_a)(x)L(u_a;x)=J_(v,x)(x)diag(x)`; for scalar `s`, this is `E_s(x)=x*grad_x s(x)`. The numerical reproduction independently implements this recursion for Linear, Conv, ReLU, MaxPool, and residual addition. Coordinates are the normalized inputs received by the model; biases are not image coordinates; ReLU uses the framework subgradient at zero; MaxPool uses the realized winner; equivalence concerns local input dependence rather than function-value completeness; CLS, position embeddings, and model biases receive no separate image-coordinate allocation; and smooth ViT operators use their true Jacobian at the observed point.

## Atlas results

| Model | Metric | Early | Middle | Late |
|---|---|---:|---:|---:|
| vit_b16 | entropy | 0.6733 | 0.8645 | 0.8908 |
| vit_b16 | locality | 0.4887 | 0.2199 | 0.1300 |
| vit_b16 | distance | 0.1626 | 0.2565 | 0.2978 |
| vit_b16 | cancellation | 0.9962 | 0.9986 | 0.9987 |
| deit_s16 | entropy | 0.6593 | 0.8525 | 0.8783 |
| deit_s16 | locality | 0.5139 | 0.2389 | 0.1761 |
| deit_s16 | distance | 0.1531 | 0.2489 | 0.2748 |
| deit_s16 | cancellation | 0.9966 | 0.9984 | 0.9986 |

| Model | Frozen contrast | Median image-level difference | Pos/neg/tie | Holm p | Decision |
|---|---|---:|---:|---:|---|
| deit_s16 | late_minus_early_entropy | 0.2250 | 100/0/0 | 7.889e-30 | SUPPORTED |
| deit_s16 | late_minus_early_locality | -0.3424 | 0/100/0 | 7.889e-30 | SUPPORTED |
| deit_s16 | late_minus_early_distance | 0.1219 | 100/0/0 | 7.889e-30 | SUPPORTED |
| deit_s16 | late_minus_early_cancellation | 0.0020 | 100/0/0 | 7.889e-30 | SUPPORTED |
| deit_s16 | cls_minus_patch_entropy | 0.1185 | 100/0/0 | 7.889e-30 | SUPPORTED |
| vit_b16 | late_minus_early_entropy | 0.2201 | 100/0/0 | 7.889e-30 | SUPPORTED |
| vit_b16 | late_minus_early_locality | -0.3558 | 0/100/0 | 7.889e-30 | SUPPORTED |
| vit_b16 | late_minus_early_distance | 0.1312 | 100/0/0 | 7.889e-30 | SUPPORTED |
| vit_b16 | late_minus_early_cancellation | 0.0025 | 100/0/0 | 7.889e-30 | SUPPORTED |
| vit_b16 | cls_minus_patch_entropy | 0.0897 | 99/1/0 | 1.593e-28 | SUPPORTED |

The joint local-to-global criterion requires a Holm-supported entropy increase, locality decrease, and distance increase. It is `SUPPORTED` for ViT-B/16 and `SUPPORTED` for DeiT-S. This decision describes local first-order input dependence for the frozen token-channel sample.

## Neuron-map similarity

| Model | Depth | Pair group | Metric | Median | 95% image-bootstrap CI |
|---|---|---|---|---:|---:|
| deit_s16 | early | different_token | patch_absolute_cosine | 0.0299 | [0.0284, 0.0321] |
| deit_s16 | early | different_token | patch_signed_cosine | -0.0000 | [-0.0000, 0.0001] |
| deit_s16 | early | same_token_different_channel | patch_absolute_cosine | 0.9948 | [0.9946, 0.9950] |
| deit_s16 | early | same_token_different_channel | patch_signed_cosine | -0.0062 | [-0.0190, 0.0061] |
| deit_s16 | late | different_token | patch_absolute_cosine | 0.3712 | [0.3530, 0.3893] |
| deit_s16 | late | different_token | patch_signed_cosine | 0.0011 | [0.0006, 0.0017] |
| deit_s16 | late | same_token_different_channel | patch_absolute_cosine | 0.9515 | [0.9465, 0.9553] |
| deit_s16 | late | same_token_different_channel | patch_signed_cosine | -0.0016 | [-0.0085, 0.0016] |
| deit_s16 | middle | different_token | patch_absolute_cosine | 0.2706 | [0.2373, 0.2909] |
| deit_s16 | middle | different_token | patch_signed_cosine | 0.0006 | [0.0002, 0.0008] |
| deit_s16 | middle | same_token_different_channel | patch_absolute_cosine | 0.9645 | [0.9592, 0.9685] |
| deit_s16 | middle | same_token_different_channel | patch_signed_cosine | 0.0049 | [-0.0016, 0.0136] |
| vit_b16 | early | different_token | patch_absolute_cosine | 0.0317 | [0.0306, 0.0346] |
| vit_b16 | early | different_token | patch_signed_cosine | -0.0000 | [-0.0001, 0.0001] |
| vit_b16 | early | same_token_different_channel | patch_absolute_cosine | 0.9953 | [0.9949, 0.9956] |
| vit_b16 | early | same_token_different_channel | patch_signed_cosine | 0.0033 | [-0.0038, 0.0286] |
| vit_b16 | late | different_token | patch_absolute_cosine | 0.4724 | [0.4543, 0.4959] |
| vit_b16 | late | different_token | patch_signed_cosine | 0.0009 | [-0.0003, 0.0016] |
| vit_b16 | late | same_token_different_channel | patch_absolute_cosine | 0.9378 | [0.9342, 0.9423] |
| vit_b16 | late | same_token_different_channel | patch_signed_cosine | 0.0024 | [-0.0056, 0.0079] |
| vit_b16 | middle | different_token | patch_absolute_cosine | 0.2761 | [0.2661, 0.2936] |
| vit_b16 | middle | different_token | patch_signed_cosine | 0.0006 | [0.0003, 0.0010] |
| vit_b16 | middle | same_token_different_channel | patch_absolute_cosine | 0.9595 | [0.9575, 0.9618] |
| vit_b16 | middle | same_token_different_channel | patch_signed_cosine | -0.0009 | [-0.0038, 0.0077] |

Absolute-distribution cosine remains high for same-token/different-channel pairs and increases with depth for different-token pairs. Signed cosine is generally near zero. These are descriptive results for the frozen pairs.

## Evidence boundary

Endpoint denotes a local Jacobian learning image over normalized image coordinates. Channel IDs are not matched neurons across models. M3 evaluates the stratified atlas; finite-replacement behavior remains calibrated by M1/M2.
