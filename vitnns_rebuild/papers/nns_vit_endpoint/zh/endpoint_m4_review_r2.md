# Endpoint M4 外部评价修订结果 R2

## 结论

M4 测量冻结 ground-truth logit 的目标条件化局部输入 Jacobian 路径记账；每个分量都包含该目标的下游 cotangent。Routing share 与 identity share 是两条路径 absolute Endpoint norm 的相对尺度，JVP error 是归一化 RMS 方向差异。它们分别描述路径尺度和方向响应偏差，不是函数值或因果贡献百分比。

三项主发现得到保留和收紧：late attention 的 routing relative path-norm scale 与 Value-only approximation error 高于 early attention；mean-query saturation 与 routing scale/error 呈负相关；attention residual 与 MLP residual 的 identity-path depth trend 相反。补充分析进一步排除了 block-11 JVP relative error 由近零分母普遍放大的解释，并为 attention residual 的空间改变提供了完整 400 图推断。

## JVP discrepancy 的分布和分母

| 模型 | Block-11 指标 | q10 | q25 | median [95% image-bootstrap CI] | q75 | q90 |
|---|---|---:|---:|---:|---:|---:|
| vit_b16 | relative_error | 0.8295 | 0.8994 | 0.9516 [0.9450, 0.9634] | 0.9847 | 0.9975 |
| vit_b16 | absolute_error | 4.9251 | 8.1568 | 13.8064 [12.5536, 14.9287] | 22.1253 | 37.3237 |
| vit_b16 | full_response_denominator | 5.5214 | 9.2066 | 14.9711 [13.5656, 16.5269] | 24.1453 | 38.1938 |
| deit_s16 | relative_error | 0.8212 | 0.8966 | 0.9636 [0.9546, 0.9718] | 0.9947 | 1.0191 |
| deit_s16 | absolute_error | 0.4985 | 0.8908 | 1.4228 [1.3004, 1.5479] | 2.3028 | 3.5636 |
| deit_s16 | full_response_denominator | 0.5719 | 0.9790 | 1.5155 [1.4165, 1.6181] | 2.4635 | 3.6999 |

Block 11 的归一化 RMS discrepancy 中位数为 0.9516 和 0.9636。对应 full-Jacobian response denominator 的图像级 head-median 分别为 14.9711（q10=5.5214）和 1.5155（q10=0.5719），而 absolute routing discrepancy 中位数为 13.8064 和 1.4228。因此高 relative error 伴随同量级 absolute discrepancy，并非由普遍近零分母造成。该比值应解释为方向响应差异与完整响应同量级，而非可加的遗漏或路径贡献分数。

## Saturation 稳健性

| 模型 | Saturation | Outcome | Unadjusted rho | 调整 output/cotangent norm 后 rho [95% CI] |
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

Mean-query saturation 在两个模型和三个 outcome 上，控制 head projection-input activation norm 与 downstream cotangent norm 后仍保持负相关。CLS-query 关系更弱：ViT-B 的 fully adjusted routing-share 与 Endpoint-error 检验在 18 项 model-specific sensitivity Holm family 中未通过（Holm p 均为 0.0629），而 JVP error 及 DeiT-S 的三项关系仍通过。由此，正式结果应以 all-query saturation 为主，不把单一 CLS row 当作等价代理。

| 模型 | Phase | Fully adjusted mean-query saturation–Endpoint-error rho | q10 | q90 |
|---|---|---:|---:|---:|
| vit_b16 | early | -0.1704 | -0.3816 | 0.0372 |
| vit_b16 | middle | -0.0992 | -0.3445 | 0.1507 |
| vit_b16 | late | -0.0804 | -0.3566 | 0.2067 |
| deit_s16 | early | -0.5109 | -0.8463 | -0.1098 |
| deit_s16 | middle | -0.1604 | -0.6083 | 0.3660 |
| deit_s16 | late | -0.0266 | -0.4684 | 0.4543 |

负相关主要集中在 early/middle blocks；late phase 接近零。这进一步限定了“sharpness 不是 routing proxy”：它是同图同层 head-level、且具有深度异质性的相关关系，不是 saturation 导致 routing 变化的因果证据。

## Residual 空间改变

空间指标为 output 与 identity 的归一化 14×14 patch absolute distribution total variation。

| 模型 | Residual | Late−early spatial TV [95% CI] | 正/负/平图像 | Holm p | Branch-share–TV rho [95% CI] |
|---|---|---:|---:|---:|---:|
| vit_b16 | attention | 0.0520 [0.0488, 0.0547] | 378/22/0 | 2.144e-84 | 0.9161 [0.9091, 0.9231] |
| vit_b16 | mlp | -0.0187 [-0.0220, -0.0150] | 111/289/0 | 2.181e-19 | 0.6084 [0.5664, 0.6434] |
| deit_s16 | attention | 0.0282 [0.0257, 0.0300] | 383/17/0 | 1.109e-90 | 0.9021 [0.8951, 0.9091] |
| deit_s16 | mlp | -0.0462 [-0.0516, -0.0412] | 36/364/0 | 4.300e-69 | 0.8182 [0.7972, 0.8322] |

两个模型的 attention residual 在 late blocks 相对 early blocks具有更大的 output–identity 空间差异，ViT-B/16 为 +0.0520，DeiT-S 为 +0.0282；MLP residual 分别为 −0.0187 和 −0.0462。四项 post-hoc spatial sensitivity tests 均通过 Holm。Branch relative path-norm scale 与 spatial TV 的图像内跨层相关也为正。因而“attention residual 更具空间重写特征”现在同时有 share 和空间分布证据；结论仍限定为目标条件化局部 Jacobian，而不是有限干预效应。

## MLP path cancellation 的模型差异

| 模型 | Late−early path cancellation [95% CI] | 正/负/平图像 | 原 M4 Holm p |
|---|---:|---:|---:|
| vit_b16 | 0.0688 [0.0561, 0.0832] | 304/96/0 | 2.840e-26 |
| deit_s16 | -0.0588 [-0.0674, -0.0492] | 103/297/0 | 6.661e-23 |

ViT-B/16 中 MLP identity–branch cancellation 随深度上升，DeiT-S 中下降；两项原冻结检验均通过校正。因此可写“呈相反的显著趋势”，但不能归纳为跨模型统一的 MLP cancellation 规律。

## M3–M4 描述性桥接

E3 事前保留的每模型 16 个 class-logit raw maps 允许比较完整 attention、routing 和 Value 路径的 patch absolute distributions。

| 模型 | Block | H(full) | H(value) | H(full)−H(value) | Full–Value patch TV [95% CI] |
|---|---:|---:|---:|---:|---:|
| vit_b16 | 0 | 0.9208 | 0.9257 | -0.0099 | 0.0550 [0.0455, 0.0952] |
| vit_b16 | 5 | 0.8802 | 0.9054 | -0.0237 | 0.1835 [0.1618, 0.2495] |
| vit_b16 | 11 | 0.9033 | 0.8912 | 0.0051 | 0.1865 [0.1534, 0.2116] |
| deit_s16 | 0 | 0.9273 | 0.9282 | 0.0006 | 0.0307 [0.0268, 0.0365] |
| deit_s16 | 5 | 0.8976 | 0.9097 | -0.0224 | 0.1565 [0.1221, 0.2476] |
| deit_s16 | 11 | 0.8826 | 0.8714 | 0.0140 | 0.1811 [0.1403, 0.2226] |

Full–Value patch TV 从 block 0 的 0.0550/0.0307 增至 block 11 的 0.1865/0.1811，说明深层 Value-only approximation 与完整 attention 的空间形态差异更大。Entropy difference 不是单调且在 ViT-B block 11 跨零，因此不支持“routing 可加地增加 entropy”。

| 模型 | M3 指标 | M4 bridge 指标 | 12-block median trajectory Spearman |
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

M3 entropy 与 Full–Value TV 的 block-trajectory rho 为 0.2727/0.4406，locality 为 −0.3357/−0.5105，distance 为 0.3357/0.5105。方向与 M3 spatial broadening 相容，但 M3 使用 100 图 token-channel targets，而 raw bridge 使用 16 图 class-logit targets；每模型仅 1 张图重叠。因此这些值只构成模型级深度轨迹桥接，不识别 routing 对 M3 空间扩展的因果效应。

## Cut、cotangent 与数值位置

Attention cut 位于每个 head 的 AV 输出、head concatenation 和 output projection 之前；concat、projection 和所有后续层的作用进入该 AV edge 的 downstream cotangent。Q/K 取 q/k normalization 后的实际 edge，Value 取 value projection edge。模型是 pre-LN timm ViT：第一处 residual 为 u + attention(norm1(u))，使用 clean g_r1；第二处为 r1 + MLP(norm2(r1))，使用 clean g_r2。Identity 与 branch 始终共享同一 residual-output cotangent。

| 数值诊断 | 模型 | 图像 | Block | Head/Residual | 最大误差 |
|---|---|---|---:|---|---:|
| qkv_closure_max_abs | vit_b16 | ILSVRC2012_val_00005415.JPEG | 11 |  | 1.574e-05 |
| frozen_identity_max_abs | vit_b16 | ILSVRC2012_val_00005415.JPEG | 11 |  | 1.621e-05 |
| explicit_jvp_relative_error | vit_b16 | ILSVRC2012_val_00025231.JPEG | 0 | 3 | 2.502e-05 |
| residual_closure_max_abs | vit_b16 | ILSVRC2012_val_00005592.JPEG | 11 | mlp | 2.480e-05 |

最大显式-vs-native JVP relative error 为 2.502e-5，位于 ViT-B/16 block 0 head 3；其余最大误差位置如表。闭合、实现正确性和行为含义继续分开报告。

## 修订后的一句话结论

在所测试模型、当前输入和冻结 ground-truth logit 下，late attention 的目标条件化局部输入 Jacobian 更难由固定 attention weights 的 Value-only 路径近似；attention residual 的 branch-relative path norm 与空间改变随深度增强，而 MLP residual 的 identity-relative path norm 增强。以上是局部 Jacobian 路径和空间分布结果，不是函数值贡献比例或有限因果重要性。
