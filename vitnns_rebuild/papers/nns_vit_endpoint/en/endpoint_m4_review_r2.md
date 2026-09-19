# Endpoint M4 external-review revision R2

## Conclusion

M4 estimates target-conditioned local input-Jacobian path accounting for the frozen ground-truth logit; every component includes that target's downstream cotangent. Routing and identity shares are relative absolute-Endpoint path-norm scales, and JVP error is a normalized RMS directional discrepancy. They describe path scale and directional mismatch rather than percentages of a function value or causal effect.

Three findings remain after this refinement: late attention has larger routing relative path-norm scale and Value-only approximation error than early attention; mean-query saturation is negatively associated with routing scale/error; and attention versus MLP residuals have opposite identity-path depth trends. The review analysis also rules out generally near-zero block-11 denominators and adds full-cohort spatial inference for the attention-residual rewrite interpretation.

## JVP discrepancy distribution and denominator

| Model | Block-11 metric | q10 | q25 | median [95% image-bootstrap CI] | q75 | q90 |
|---|---|---:|---:|---:|---:|---:|
| vit_b16 | relative_error | 0.8295 | 0.8994 | 0.9516 [0.9450, 0.9634] | 0.9847 | 0.9975 |
| vit_b16 | absolute_error | 4.9251 | 8.1568 | 13.8064 [12.5536, 14.9287] | 22.1253 | 37.3237 |
| vit_b16 | full_response_denominator | 5.5214 | 9.2066 | 14.9711 [13.5656, 16.5269] | 24.1453 | 38.1938 |
| deit_s16 | relative_error | 0.8212 | 0.8966 | 0.9636 [0.9546, 0.9718] | 0.9947 | 1.0191 |
| deit_s16 | absolute_error | 0.4985 | 0.8908 | 1.4228 [1.3004, 1.5479] | 2.3028 | 3.5636 |
| deit_s16 | full_response_denominator | 0.5719 | 0.9790 | 1.5155 [1.4165, 1.6181] | 2.4635 | 3.6999 |

The block-11 normalized RMS discrepancy medians are 0.9516 and 0.9636. The corresponding image-level head-median full-Jacobian response denominators are 14.9711 (q10=5.5214) and 1.5155 (q10=0.5719), while the absolute routing discrepancies are 13.8064 and 1.4228. The high relative errors therefore accompany same-order absolute discrepancies rather than generally near-zero denominators. They indicate directional differences comparable in norm with the full response, not an additive omission or path-contribution fraction.

## Saturation sensitivity

| Model | Saturation | Outcome | Unadjusted rho | After output/cotangent-norm adjustment rho [95% CI] |
|---|---|---|---:|---:|
| vit_b16 | mean_query | routing_share | -0.2937 | -0.1442 [-0.1613, -0.1321] |
| vit_b16 | mean_query | frozen_attention_endpoint_error | -0.2238 | -0.1129 [-0.1293, -0.0968] |
| vit_b16 | mean_query | frozen_attention_jvp_error | -0.3462 | -0.1453 [-0.1555, -0.1342] |
| vit_b16 | cls_query | routing_share | -0.1608 | -0.0296 [-0.0450, -0.0017] |
| vit_b16 | cls_query | frozen_attention_endpoint_error | -0.1469 | -0.0255 [-0.0450, -0.0013] |
| vit_b16 | cls_query | frozen_attention_jvp_error | -0.2133 | -0.0642 [-0.0855, -0.0487] |
| deit_s16 | mean_query | routing_share | -0.3143 | -0.3257 [-0.3521, -0.2894] |
| deit_s16 | mean_query | frozen_attention_endpoint_error | -0.2286 | -0.2185 [-0.2428, -0.1870] |
| deit_s16 | mean_query | frozen_attention_jvp_error | -0.2857 | -0.2786 [-0.3005, -0.2428] |
| deit_s16 | cls_query | routing_share | -0.0857 | -0.1082 [-0.1392, -0.0838] |
| deit_s16 | cls_query | frozen_attention_endpoint_error | -0.0286 | -0.0969 [-0.1372, -0.0596] |
| deit_s16 | cls_query | frozen_attention_jvp_error | -0.0571 | -0.1259 [-0.1549, -0.1023] |

Mean-query saturation remains negatively associated with all three outcomes in both models after adjustment for the head projection-input activation norm and downstream-cotangent norm. CLS-query relations are weaker: the fully adjusted ViT-B routing-share and Endpoint-error tests do not survive the 18-test model-specific sensitivity Holm family (both Holm p=0.0629), whereas its JVP-error relation and all three DeiT-S relations remain supported. The formal summary should therefore use all-query saturation; one CLS row is not an equivalent proxy.

| Model | Phase | Fully adjusted mean-query saturation–Endpoint-error rho | q10 | q90 |
|---|---|---:|---:|---:|
| vit_b16 | early | -0.1704 | -0.3816 | 0.0372 |
| vit_b16 | middle | -0.0992 | -0.3445 | 0.1507 |
| vit_b16 | late | -0.0804 | -0.3566 | 0.2067 |
| deit_s16 | early | -0.5109 | -0.8463 | -0.1098 |
| deit_s16 | middle | -0.1604 | -0.6083 | 0.3660 |
| deit_s16 | late | -0.0266 | -0.4684 | 0.4543 |

The negative association is concentrated in early and middle blocks and approaches zero in the late phase. Thus attention sharpness is not a positive routing proxy under this within-image, within-layer head comparison; the association does not establish that saturation causes routing changes.

## Residual spatial change

The spatial metric is total variation between the output and identity normalized 14×14 patch absolute distributions.

| Model | Residual | Late−early spatial TV [95% CI] | Pos/neg/tie images | Holm p | Branch-share–TV rho [95% CI] |
|---|---|---:|---:|---:|---:|
| vit_b16 | attention | 0.0520 [0.0488, 0.0547] | 378/22/0 | 2.144e-84 | 0.9161 [0.9091, 0.9231] |
| vit_b16 | mlp | -0.0187 [-0.0220, -0.0150] | 111/289/0 | 2.181e-19 | 0.6084 [0.5664, 0.6434] |
| deit_s16 | attention | 0.0282 [0.0257, 0.0300] | 383/17/0 | 1.109e-90 | 0.9021 [0.8951, 0.9091] |
| deit_s16 | mlp | -0.0462 [-0.0516, -0.0412] | 36/364/0 | 4.300e-69 | 0.8182 [0.7972, 0.8322] |

Attention residuals have larger output–identity spatial differences in late than early blocks in both models (+0.0520 for ViT-B/16 and +0.0282 for DeiT-S), whereas MLP residual differences decrease (−0.0187 and −0.0462). All four post-hoc spatial sensitivity tests survive Holm correction. The within-image cross-block association between branch relative path-norm scale and spatial TV is also positive. The attention-residual rewrite description is therefore supported jointly by relative norm and spatial-distribution evidence, while remaining a target-conditioned local-Jacobian result rather than a finite intervention effect.

## Model-dependent MLP path cancellation

| Model | Late−early path cancellation [95% CI] | Pos/neg/tie images | Original M4 Holm p |
|---|---:|---:|---:|
| vit_b16 | 0.0688 [0.0561, 0.0832] | 304/96/0 | 2.840e-26 |
| deit_s16 | -0.0588 [-0.0674, -0.0492] | 103/297/0 | 6.661e-23 |

MLP identity–branch path cancellation increases with depth in ViT-B/16 and decreases in DeiT-S; both original frozen tests survive correction. The paper can state that the models exhibit significant opposite trends, but not a model-invariant MLP cancellation law.

## Descriptive M3–M4 bridge

The 16 pre-retained E3 class-logit raw maps per model permit comparison of complete-attention, routing, and Value patch absolute distributions.

| Model | Block | H(full) | H(value) | H(full)−H(value) | Full–Value patch TV [95% CI] |
|---|---:|---:|---:|---:|---:|
| vit_b16 | 0 | 0.9208 | 0.9257 | -0.0099 | 0.0550 [0.0455, 0.0952] |
| vit_b16 | 5 | 0.8802 | 0.9054 | -0.0237 | 0.1835 [0.1618, 0.2495] |
| vit_b16 | 11 | 0.9033 | 0.8912 | 0.0051 | 0.1865 [0.1534, 0.2116] |
| deit_s16 | 0 | 0.9273 | 0.9282 | 0.0006 | 0.0307 [0.0268, 0.0365] |
| deit_s16 | 5 | 0.8976 | 0.9097 | -0.0224 | 0.1565 [0.1221, 0.2476] |
| deit_s16 | 11 | 0.8826 | 0.8714 | 0.0140 | 0.1811 [0.1403, 0.2226] |

Full–Value patch TV rises from 0.0550/0.0307 at block 0 to 0.1865/0.1811 at block 11, indicating larger spatial-shape differences between late complete attention and its Value-only approximation. Entropy differences are nonmonotonic and cross zero for ViT-B at block 11, so the data do not support an additive claim that routing increases entropy.

| Model | M3 metric | M4 bridge metric | 12-block median trajectory Spearman |
|---|---|---|---:|
| vit_b16 | entropy | full_minus_value_entropy | 0.3916 |
| vit_b16 | entropy | full_value_patch_tv | 0.2727 |
| vit_b16 | locality | full_minus_value_entropy | -0.4406 |
| vit_b16 | locality | full_value_patch_tv | -0.3357 |
| vit_b16 | distance | full_minus_value_entropy | 0.4406 |
| vit_b16 | distance | full_value_patch_tv | 0.3357 |
| deit_s16 | entropy | full_minus_value_entropy | 0.1259 |
| deit_s16 | entropy | full_value_patch_tv | 0.4406 |
| deit_s16 | locality | full_minus_value_entropy | -0.1259 |
| deit_s16 | locality | full_value_patch_tv | -0.5105 |
| deit_s16 | distance | full_minus_value_entropy | 0.1259 |
| deit_s16 | distance | full_value_patch_tv | 0.5105 |

M3 entropy versus Full–Value TV has block-trajectory rho 0.2727/0.4406; locality has −0.3357/−0.5105 and distance 0.3357/0.5105. These directions are compatible with M3 spatial broadening, but M3 uses 100-image token-channel targets whereas the raw bridge uses 16-image class-logit targets, with only one overlapping image per model. The result is a model-level depth-trajectory bridge and does not identify a causal effect of routing on the M3 broadening.

## Cut, cotangent, and numerical locations

The attention cut is each head's AV output before head concatenation and output projection; concatenation, projection, and all later layers enter the downstream cotangent at that AV edge. Q/K are the actual post-q/k-normalization edges and Value is the value-projection edge. The timm models are pre-LN: the first residual is u + attention(norm1(u)) with clean g_r1, and the second is r1 + MLP(norm2(r1)) with clean g_r2. Identity and branch use the same clean residual-output cotangent.

| Numerical diagnostic | Model | Image | Block | Head/Residual | Maximum error |
|---|---|---|---:|---|---:|
| qkv_closure_max_abs | vit_b16 | ILSVRC2012_val_00005415.JPEG | 11 |  | 1.574e-05 |
| frozen_identity_max_abs | vit_b16 | ILSVRC2012_val_00005415.JPEG | 11 |  | 1.621e-05 |
| explicit_jvp_relative_error | vit_b16 | ILSVRC2012_val_00025231.JPEG | 0 | 3 | 2.502e-05 |
| residual_closure_max_abs | vit_b16 | ILSVRC2012_val_00005592.JPEG | 11 | mlp | 2.480e-05 |

The largest explicit-versus-native JVP relative error is 2.502e-5 at ViT-B/16 block 0 head 3. The table locates the other maxima. Algebraic closure, implementation correctness, and behavioral interpretation remain separate evidence claims.

## Revised one-sentence conclusion

For the tested models, observed inputs, and frozen ground-truth logit, late attention's target-conditioned local input Jacobian is less accurately approximated by Value-only propagation with fixed attention weights; attention residuals show increasing branch-relative path norm and spatial change, whereas MLP residuals show increasing identity-relative path norm. These are local Jacobian path and spatial-distribution results, not function-value percentages or finite causal importance.
