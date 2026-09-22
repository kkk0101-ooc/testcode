# 创新性成立条件与失效条件

## 当前允许的创新主张

`NOV-JOINT-1` 是唯一方法新颖性主轴。对每个注册目标 $s$，本文将总 Gradient × Input 展开为 $\mathcal L_{e,i}(s;x)$。该对象同时保留原始输入坐标 $i$ 和具体 consumer-use edge $e$。合法切面上的分项恢复同一 current-point signed local differential。结构负控和独立数值 oracle 检查实现，attention 与 residual 的空间化层深结果检验联合索引的经验用途。

## 成立条件

1. **对象成立。** $e$ 由 producer、consumer、input slot 和 invocation 确定；$i$ 是原始输入坐标；$s$ 来自冻结 target registry。
2. **合同成立。** 合法切面使每条有效输入到目标路径恰好穿过一个登记 use，signed 分项恢复同一总 G×I。
3. **审计成立。** 已报告能力只能包含三类封存 mutation、JVP/VJP、有限差分和独立重算。
4. **必要性成立。** 至少一个 attention 和一个 residual 结果需要路径身份与输入位置的联合信息。现有证据为 Q/K/V 空间分布和 output–identity spatial TV。
5. **文献差分成立。** Conductance、DecompX/LRP、EAP/Vi-CD 与 Unpack 均为直接近邻；来源核查未发现一个方法覆盖全部联合字段。
6. **复现边界成立。** H1–H3 在四个公开 checkpoint 保持方向，同时保留 H4 四项反向和 Q 的模型依赖性。

## 失效条件

- 若只剩 $x\odot\nabla_xs(x)$ 或普通热图，论文降级为 NNS 数学身份澄清。
- 若删除输入坐标 $i$，对象退化为 edge scalar，不能支持空间重组结论。
- 若删除边身份 $e$，对象退化为总 G×I，不能区分 routing/value 或 identity/branch。
- 若切面不能穷尽且不重复地覆盖有效路径，结构记账主张失效。
- 若结构错误不能被已声明审计检出，`NOV-AUD-1` 失效；未封存 wrong-slot、fan-out merge 和 detach mutation 不得补写为已验证。
- 若 attention 和 residual 结果可由总图、activation norm 或 edge-only scale 直接恢复，联合索引的经验必要性失效。
- 若一个既有方法同时覆盖 current-point local differential、输入坐标、具体 consumer-use edge、合法切面重构、attention/residual 成对图和结构故障检出，方法新颖性失效。
- 若将闭合写成因果正确、将 path norm 写成贡献百分比，或将四 checkpoint 写成训练因果和全部 ViT 规律，论文的解释有效性失效。
- 若只保留 H1–H3 而删除 H4，跨 checkpoint 证据构成选择性报告。

## 不受当前证据支持的扩大表述

新基础归因量、普遍优于既有方法、全模型或全架构规律、因果贡献百分比、训练导致、实现不变性、穷举全部神经元，以及局部路径图等于有限替换效应。
