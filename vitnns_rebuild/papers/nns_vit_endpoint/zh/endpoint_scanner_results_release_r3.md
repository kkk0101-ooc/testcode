# Endpoint Scanner E6 结果发布

## Identity、Correctness 与 Utility

对同一输入端标量目标，基础 Endpoint 图为 $x \odot \nabla_x s(x)$，数值上就是 Gradient × Input。注册实现对 ViT-B/16 和 DeiT-S/16 各自扫描同一组 400 张 evaluation 图。结构切面通过 76,800 项闭合检查和 28,800 项遗漏/重复负例，最大 relative L2 为 1.41446e-5；E4-R3 条件尺度合同下的 6,144 项解析 JVP 检查全部通过。

确认性有限干预判定为：**C4a NOT_SUPPORTED**、**C4b NOT_SUPPORTED**。C4a 检验成分 absolute mass 与有限 activation-replacement 幅度的图像级关联；C4b 检验 Endpoint 空间排序相对每个 mandatory baseline 的图像级增量。下表完整报告 16 个注册 family 和统一 Holm 校正的 28 个检验。

| Family | 估计值 | 95% 区间 | Holm p(s) | 判定 |
|---|---:|---:|---:|---|
| C4a.vit_b16.normalized_zero.heads | 0.2909 | [0.2813, 0.2982] | 2.169e-119 | SUPPORTED |
| C4a.vit_b16.normalized_zero.routing_value | 0.0996 | [0.0748, 0.1226] | 3.004e-14 | NOT_SUPPORTED |
| C4a.vit_b16.normalized_zero.residual_branches | 0.1774 | [0.1548, 0.1952] | 2.555e-46 | SUPPORTED |
| C4b.vit_b16.normalized_zero.head_regions | -0.0009, -0.0000, -0.0234, 0.0011 | [-0.0017, -0.0002], [-0.0007, 0.0004], [-0.0257, -0.0217], [0.0001, 0.0019] | 0.05986, 1, 1.12e-106, 0.1493 | NOT_SUPPORTED |
| C4a.vit_b16.blur_rgb.heads | 0.3063 | [0.2957, 0.3181] | 2.169e-119 | SUPPORTED |
| C4a.vit_b16.blur_rgb.routing_value | 0.1565 | [0.1322, 0.1774] | 9.96e-29 | SUPPORTED |
| C4a.vit_b16.blur_rgb.residual_branches | 0.2091 | [0.1817, 0.2374] | 1.456e-41 | SUPPORTED |
| C4b.vit_b16.blur_rgb.head_regions | -0.0010, -0.0007, -0.0237, 0.0042 | [-0.0018, -0.0003], [-0.0021, 0.0000], [-0.0272, -0.0216], [0.0030, 0.0055] | 0.1493, 0.3533, 4.092e-71, 2.95e-09 | NOT_SUPPORTED |
| C4a.deit_s16.normalized_zero.heads | 0.1910 | [0.1769, 0.2052] | 1.88e-76 | SUPPORTED |
| C4a.deit_s16.normalized_zero.routing_value | 0.1065 | [0.0817, 0.1291] | 2.106e-10 | SUPPORTED |
| C4a.deit_s16.normalized_zero.residual_branches | 0.1352 | [0.1148, 0.1643] | 4.353e-19 | SUPPORTED |
| C4b.deit_s16.normalized_zero.head_regions | -0.0018, -0.0006, -0.0299, 0.0107 | [-0.0034, 0.0000], [-0.0022, 0.0009], [-0.0338, -0.0261], [0.0079, 0.0139] | 0.2958, 1, 4.583e-56, 7.373e-18 | NOT_SUPPORTED |
| C4a.deit_s16.blur_rgb.heads | 0.1817 | [0.1688, 0.1951] | 3.824e-72 | SUPPORTED |
| C4a.deit_s16.blur_rgb.routing_value | 0.1600 | [0.1278, 0.1822] | 9.901e-31 | SUPPORTED |
| C4a.deit_s16.blur_rgb.residual_branches | 0.2287 | [0.2035, 0.2504] | 3.699e-54 | SUPPORTED |
| C4b.deit_s16.blur_rgb.head_regions | 0.0002, -0.0002, -0.0356, 0.0085 | [-0.0005, 0.0009], [-0.0011, 0.0004], [-0.0390, -0.0308], [0.0064, 0.0106] | 1, 1, 1.88e-76, 9.991e-22 | NOT_SUPPORTED |

## 解释边界

上述效用结论限定在注册的双模型、双 reference 评价中。基础 Endpoint 归因不是不同于 Gradient × Input 的新公式。结构闭合和局部 JVP 支持 correctness；有限 replacement 和空间干预决定 utility。Path IG 在 ViT-B/16 上未通过冻结候选范围的数值校准，因此本文不发布跨模型 Endpoint–Path 结论。
