# 主图逐图说明

本修订将原 R2 的 40 张正文图组织为 10 张主图。旧图及全部 52 项补充记录均由论文补充材料索引。总体结果来自完整 cohort；代表案例沿用已有冻结选择。

1. **Image #1 — NNS 到 Endpoint。** 左侧真实懒熊输入，中间 signed learning image，右侧 RGB absolute mass。公式明确输入值 seed 和当前 Jacobian。Signed/absolute 色标对应不同数值对象，不能把正部单独当作完整学习图像。
2. **Image #2 — Registry 与合法 cut。** 左侧区分标量神经元、聚合响应、结构路径和输出；右侧显示 attention output projection 与 residual-output cotangent。该页服务方法阅读，不夹带工程状态。
3. **Image #3 — 忠实性与正确性。** 五类兼容网络的精度误差与闭合/方向检查并列。Log 轴表达 float64 与 float32 尺度差，计数只说明验证覆盖。
4. **Image #4 — 空间扩展。** Entropy、locality、distance 三条证据并列，统一模型色和 block 刻度；阴影为图像 bootstrap 区间。该图呈现抽样单元随深度的空间变化，不推断语义或训练来源。
5. **Image #5 — CLS 与抵消。** 总体 CLS-minus-patch entropy、net/absolute mass 与预选 high-cancellation 案例对应。图下的 signed sum 和 L1 让接近 1 的 cancellation 有直接数值含义。
6. **Image #6 — Routing/value。** 上排给出 cohort 趋势，下两排对同一 target/head 展示 routing、value、full 的 signed 与 absolute 图。行内共享 robust scale，并标原始质量/范数；未通过逐面板独立增亮制造强度关系。
7. **Image #7 — Residual。** 同时呈现 attention/MLP identity scale 及空间 TV，再以同 cotangent 的 identity、branch、output 展示正负路径抵消。曲线的非单调性完整保留；正文推断使用 late-minus-early 统计。
8. **Image #8 — 代表性 atlas。** 同一图像、两个模型、早晚层，模型内使用共享显示尺度。早层局部图因质量集中而保留小亮区；没有为了观感扩大热点。不同 channels 独立注册，不称为语义匹配。
9. **Image #9 — 成分行为标定。** 左侧 12 个原 family 的点估计/CI 与 0.10 线；右侧单独显示补充冻结 head 排序。方形标明原未通过 family。CI 跨线与是否通过点估计规则在表中分开报告。
10. **Image #10 — 空间干预边界。** 绝对 AUC 同时显示 Endpoint、Taylor、random、候选 oracle。下方同图同 head 的 signed/absolute Endpoint 与区域 Taylor 分数解释估计对象差别；Taylor 16 格保留区域语义，不伪装成像素归因图。

最终视觉检查覆盖十张主图及高密度页面的原尺寸检查。修复了 asinh 自动刻度在零附近拥挤、Figure 8 首排标题与副标题间距不足的问题。共享尺度、RGB reduction、截断像素数、质量和范数记录在 [display_receipt.json](display_receipt.json)。每张图提供可编辑 SVG 和矢量文字 PDF；图像像素来自既有 raw arrays。
