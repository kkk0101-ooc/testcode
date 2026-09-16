# 统一论文图 R4：区分模型表征、token 归因与 patch 内展开

日期：2026-09-16。用户授权依据前一轮概念澄清重画对应图，并在新的 R4 目录交付完整 Figure 1–14。

Figure 1–3 重画；Figure 7 继承已验证的 R3；其余图逐字节继承已验证的 R2。每张图在 R4 内提供 PNG、PDF、SVG 实体文件，中英文图注、逐图说明和画廊均覆盖全部 14 张图。旧报告与实验根目录保持封存。

## 表达修订

- Figure 1 区分同一个已训练 ViT 的高维 token 表征与两种归因读出。用两个不同分数的 patch 展示：patch A 的示意贡献为 16，patch B 为 8；均匀展开后每 fine cell 分别为 1 和 0.5。均匀仅发生在各 patch 内。
- Figure 2 并列给出 input-space IG 和 endpoint token contribution 的独立计算路径。完整保留固定内部标量 f、clean attention A、方向 u、参考输入 x0、RGB 带符号求和和 CLS 单列规则。baseline 的 c_j 不是原始 attention 权重；NNS 不是对 c_j 的插值。深层 token 可以混合其他空间位置的信息，不宣称 NNS 在原图 patch j 内的和必然等于 c_j。
- Figure 3 分别呈现 native patch、fine cell、方法排序、同分处理与干预响应。网格内明确标示一个 16×16-pixel patch 包含 16 个 4×4-pixel cells；NNS 可以在 patch 内排序，均匀基线在 patch 内同分。两者使用相同干预预算、内部目标与配对 tie seeds。

## 科学内容与显示合同

这些图属于展示修订。模型为原已训练模型；所选 head 的 A(x) 与 u 只在本归因目标的计算中固定，不把这个操作描述为正常 ViT 的训练方式。模型可编码 patch 内像素差异；每 token 一个归因标量是基线读出选择，不是模型表达能力限制。

所有贡献网格、数字示例与 mask 序列都在图面和图注标记为 illustrative。冻结原图继续使用 F3C3 formal position 1，仅用于流程示意。示意 NNS 数组与 token 分数分别指定，不赋予每 patch 总和相等的含义，不伪造实测结果。Figure 1–3 不新增 attention、归因或 fidelity 计算。

正式统计沿用既有数值：21 fractions、5 次配对 tie repeats、4×4-pixel intervention、图像级汇总和 paired bootstrap。Figure 4–14 的图形、数值和科学边界由继承源确定。

## 验证与封存

渲染前运行 R2/R3 既有验证器，重新计算源文件 SHA-256 和 canonical digest。R4 检查 14×3 输出覆盖、11 张继承图的逐字节一致性、旧目录快照不变、数值与图注继承、示意展开的质量守恒、图内文本边界和实际 PNG 视觉审阅。视觉审阅记录在查看渲染结果后独立生成；渲染器不自行签发视觉 PASS。封存后仅运行只读验证。
