# Stage F F3 blue-display r4 visual review

Review date: 2026-09-07. Scope: all 27 report-side PNG review copies. Result: PASS. The reference PDF was used only for visual grammar: compact matrices, deep-blue low-response background, and sparse cyan/yellow/red hotspots. No reference image, datum, text, or conclusion was copied.

Common interpretation boundary: every blue-display attribution panel is independently shape-normalized from `abs(raw_signed_map)` and therefore supports spatial-pattern inspection only. Color magnitude cannot be compared between NNS and matched patch, between coordinates, or between samples. The sealed F3 r7 signed shared-scale panels remain the magnitude audit. Bilinear resize is display-only and never returns to a metric.

## Image #1: `01_figure5_case_00_order_372.png`

- 图能说明什么：best-effect 规则样本中 NNS 与 matched patch 的空间热点、叠加位置及已封存 fidelity curves。
- 图不能说明什么：不能由独立归一化颜色断言 NNS 绝对幅度更大。最可能的误读是把蓝底图称为 raw attention。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=各自固定 0–1 shape scale，matched 清晰；FAILED/NA=不适用；panel 顺序与图注=一致；原图/map/overlay 对齐=通过；明显插值伪影=无。

## Image #2: `02_figure5_case_01_order_044.png`

- 图能说明什么：median-effect 样本 order 44 的两种方法空间支持和 fidelity 曲线。
- 图不能说明什么：不能比较两图绝对强度或扩展为总体个案结论。最可能的误读是把 matched patch 的稀疏热点理解为更弱。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 清晰；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #3: `03_figure5_case_02_order_281.png`

- 图能说明什么：weak-or-failure 规则样本中两种空间模式及 fidelity 曲线。
- 图不能说明什么：不能把单个弱效果样本当作总体失败。最可能的误读是以颜色饱和度比较方法幅度。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 清晰；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #4: `04_figure5_case_03_order_387.png`

- 图能说明什么：misclassified-or-difficult 规则样本的空间支持差异。
- 图不能说明什么：不能把误分类与 attribution 因果绑定。最可能的误读是把密集 NNS 热点视为类别正确性证据。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 可辨；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #5: `05_s1_fixed_audit_00_order_231.png`

- 图能说明什么：第 1 个 fixed audit example 的冻结空间模式。
- 图不能说明什么：不能称为 best、representative 或 random。最可能的误读是视觉选例。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 清晰；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #6: `06_s1_fixed_audit_01_order_014.png`

- 图能说明什么：第 2 个 fixed audit example 的 NNS/matched patch 空间支持。
- 图不能说明什么：不能作幅度或样本代表性比较。最可能的误读是把边缘热点解释为模型语义。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 清晰；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #7: `07_s1_fixed_audit_02_order_145.png`

- 图能说明什么：第 3 个 fixed audit example 的空间热点与覆盖位置。
- 图不能说明什么：不能由单图证明 NNS 普遍优越。最可能的误读是跨方法比较颜色绝对值。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 清晰；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #8: `08_s1_fixed_audit_03_order_326.png`

- 图能说明什么：第 4 个 fixed audit example 的空间模式。
- 图不能说明什么：不能把热点直接解释为网络学习内容。最可能的误读是 raw self-attention。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 可辨；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #9: `09_s1_fixed_audit_04_order_380.png`

- 图能说明什么：第 5 个 fixed audit example 的两种空间支持。
- 图不能说明什么：不能比较跨图强度。最可能的误读是把局部热块视为单像素证据。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 清晰；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #10: `10_s2_rule_to_image_mapping.png`

- 图能说明什么：完整规则到样本映射、重复命中、order、delta_fid 和预测状态。
- 图不能说明什么：不能视为按外观挑选。最可能的误读是重复样本被补位；表中明确保留重复映射。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=不适用；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=不适用；明显插值伪影=不适用。

## Image #11: `11_s2_slot_00_best_effect_order_372.png`

- 图能说明什么：best_effect slot 的完整定性模块。
- 图不能说明什么：不能替代 500 图总体结论。最可能的误读是将其称为代表性样本。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 清晰；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #12: `12_s2_slot_01_median_effect_order_044.png`

- 图能说明什么：median_effect slot 的完整定性模块。
- 图不能说明什么：不能由显示色强比较方法幅度。最可能的误读是认为该样本由跨层图视觉质量挑选。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 清晰；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #13: `13_s2_slot_02_baseline_strong_nns_stronger_order_372.png`

- 图能说明什么：重复命中的 baseline-strong/NNS-stronger slot，未替换样本。
- 图不能说明什么：不能把重复命中解释为两个独立样本。最可能的误读是扩大样本数。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 清晰；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #14: `14_s2_slot_03_weak_or_failure_order_281.png`

- 图能说明什么：weak_or_failure slot 的完整定性模块。
- 图不能说明什么：不能删去弱效果或据此作总体判断。最可能的误读是把规则名当成 typed failure。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 清晰；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #15: `15_s2_slot_04_misclassified_or_difficult_order_387.png`

- 图能说明什么：misclassified_or_difficult slot 的完整定性模块。
- 图不能说明什么：不能推断误分类原因。最可能的误读是将 attribution 当类别 logit 解释。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；NNS/matched 色限=独立 0–1，matched 可辨；FAILED/NA=不适用；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #16: `16_figure6_median_effect_progression.png`

- 图能说明什么：冻结 median_effect/order 44 在四个 fixed block/head 坐标的 NNS 与 matched patch 空间模式。
- 图不能说明什么：不能跨坐标或跨方法比较绝对 magnitude，也不是 raw self-attention。最可能的误读是把形状演进当成网络逐层学习内容的直接证明。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名与列名位置正确；NNS/matched 色限=各 panel 0–1 shape scale；FAILED/NA=无；panel 顺序与图注=一致；原图/map 对齐=通过；明显插值伪影=无。

## Image #17: `17_s4_matrix_00_order_231.png`

- 图能说明什么：fixed audit 00 的四坐标空间模式。
- 图不能说明什么：不能比较坐标间绝对强度。最可能的误读是把暖色数量当作层间幅度。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名清晰；NNS/matched 色限=独立 0–1；FAILED/NA=无；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #18: `18_s4_matrix_01_order_014.png`

- 图能说明什么：fixed audit 01 的四坐标空间模式。
- 图不能说明什么：不能比较不同坐标 absolute magnitude。最可能的误读是把 matched patch 的方块结构当显示错误。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名清晰；NNS/matched 色限=独立 0–1；FAILED/NA=无；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #19: `19_s4_matrix_02_order_145.png`

- 图能说明什么：fixed audit 02 的四坐标空间模式。
- 图不能说明什么：不能解释为完整 Q/K/V attribution。最可能的误读是比较列间颜色饱和度。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名清晰；NNS/matched 色限=独立 0–1；FAILED/NA=无；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #20: `20_s4_matrix_03_order_326.png`

- 图能说明什么：fixed audit 03 的四坐标空间模式。
- 图不能说明什么：不能断言层间 attribution 变强或变弱。最可能的误读是从颜色推断绝对量。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名清晰；NNS/matched 色限=独立 0–1；FAILED/NA=无；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #21: `21_s4_matrix_04_order_380.png`

- 图能说明什么：fixed audit 04 的四坐标空间模式。
- 图不能说明什么：不能视为 raw attention map。最可能的误读是将局部热点当成类别证据。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名清晰；NNS/matched 色限=独立 0–1；FAILED/NA=无；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #22: `22_s4_matrix_05_order_372.png`

- 图能说明什么：best_effect 与 baseline_strong_nns_stronger 重复命中的同一冻结样本。
- 图不能说明什么：不能将两个 slot 计算为两个独立案例。最可能的误读是重复补位。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名清晰；NNS/matched 色限=独立 0–1；FAILED/NA=无；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #23: `23_s4_matrix_06_order_044.png`

- 图能说明什么：median_effect/order 44 的 S4 映射，与 Figure 6 主样本一致。
- 图不能说明什么：不能声称按 E3 map 外观选样。最可能的误读是将 Figure 6 与该重复显示算作不同样本。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名清晰；NNS/matched 色限=独立 0–1；FAILED/NA=无；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #24: `24_s4_matrix_07_order_281.png`

- 图能说明什么：weak_or_failure slot/order 281 的全部四坐标和两方法。
- 图不能说明什么：不能隐藏弱效果或比较跨列 magnitude。最可能的误读是把稀疏 NNS 视为缺失。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名清晰；NNS/matched 色限=独立 0–1；FAILED/NA=无；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #25: `25_s4_matrix_08_order_387.png`

- 图能说明什么：misclassified_or_difficult slot/order 387 的完整跨层矩阵。
- 图不能说明什么：不能解释误分类因果，也不能作层间幅度比较。最可能的误读是把热点密度当效果大小。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名清晰；NNS/matched 色限=独立 0–1；FAILED/NA=无；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无。

## Image #26: `26_s4_matrix_09_order_117.png`

- 图能说明什么：order 117 的 `(11,2)` NNS typed failure 原位保留；matched patch 与其他有效坐标照常显示。
- 图不能说明什么：不能将 FAILED/NA 当零图或把缺失解释为无 attribution。最可能的误读是认为失败格经过插值。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名清晰；NNS/matched 色限=有效 panel 独立 0–1；FAILED/NA=醒目且 failure code 可读；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无，失败格未补零。

## Image #27: `27_s4_matrix_10_order_352.png`

- 图能说明什么：order 352 的 `(11,2)` NNS typed terminal failure 原位保留；matched patch 与其他有效坐标照常显示。
- 图不能说明什么：不能将 FAILED/NA 当作低响应。最可能的误读是认为失败样本被替换；实际 sample/order 均冻结。
- 视觉检查：文字或 colorbar 裁切=无；字号=可读；水平行名清晰；NNS/matched 色限=有效 panel 独立 0–1；FAILED/NA=醒目且 failure code 可读；panel 顺序与图注=一致；对齐=通过；明显插值伪影=无，失败格未补零。

## Overall decision

All 27 review copies pass manual visual QC. Titles, column headings, horizontal row labels, panel letters, failure text, and the fidelity legend occupy distinct regions. No text or colorbar is clipped; this blue-display revision intentionally has no colorbars because every valid attribution panel uses the same declared 0–1 shape scale, with exact transformations retained in the machine manifest. Matched patch is visible in every valid paired case. No FAILED/NA cell is hidden or imputed.
