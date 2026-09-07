# Stage F F3 图片分析

状态：PASS。审查范围为 figures_png 下 28 张 review copies。所有归因图均是 fixed-clean-attention head-direction Value-path attribution progression，不是 raw self-attention。NNS/matched 颜色严格共享同例同坐标色限；各自 50%/80% signed-extrema contours 只显示方法内部空间形态，不编码可比较幅度。

每张图均检查以下字段：图能说明什么、图不能说明什么、最可能的误读、文字或 colorbar 裁切、字号、NNS/matched 色限、FAILED/NA、panel 顺序和图注、原图/map/overlay 对齐、插值伪影。

## Image #1: Figure 1 method overview

- 图能说明什么：已实现的 Input image → ViT → fixed block/head clean attention → Value-path IG / matched patch-token contribution → fine 4x4 signed map → fidelity 流程。
- 图不能说明什么：不承载经验结果，不包含 CAM、Grad-CAM 或完整 Q/K/V 路径。
- 最可能的误读：matched 是同一固定 target 下的 baseline，不是第二个模型。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：不适用。FAILED/NA：不适用。
- panel 顺序和图注：一致。原图、map、overlay 对齐：不适用。插值伪影：无，全部为矢量元素。

## Image #2: Figure 5 best_effect plus duplicate baseline slot, order 372

- 图能说明什么：完整显示原图、NNS、matched、difference、overlays、curves，并保留两个规则重复命中。
- 图不能说明什么：不是总体代表；contours 不能比较幅度。
- 最可能的误读：matched 淡色被当作无结构；轮廓显示其稀疏形态。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：严格一致，difference 独立。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：raw nearest，overlay bilinear 仅展示，无明显伪影。

## Image #3: Figure 5 median_effect, order 44

- 图能说明什么：展示 D10 median 距离冻结的主例及两方法空间和 fidelity 差异。
- 图不能说明什么：不能替代 500 图 primary，也不能支持跨坐标强度比较。
- 最可能的误读：把 local contours 当跨方法 magnitude 编码。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致且与 Figure 6 sample ID 相同。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #4: Figure 5 weak_or_failure, order 281

- 图能说明什么：最低 valid primary delta_fid 弱效果案例被原样保留。
- 图不能说明什么：不能宣称总体失败。
- 最可能的误读：把弱效果当 scientific failure；该行仍 VALID。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #5: Figure 5 misclassified_or_difficult, order 387

- 图能说明什么：误分类 cohort 冻结案例及两方法/curves。
- 图不能说明什么：不能证明误分类原因或 class-logit attribution。
- 最可能的误读：把横向 pattern 当 class-logit explanation。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致，misclassified 状态明确。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #6: Supplement S1 fixed audit 0, order 231

- 图能说明什么：D1 fixed audit example 的完整方法审计视图。
- 图不能说明什么：不是 best、representative 或 random example。
- 最可能的误读：把 contours 当新增 endpoint。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #7: Supplement S1 fixed audit 1, order 14

- 图能说明什么：第二个 fixed audit example 的完整对比。
- 图不能说明什么：不能由高光区域形成因果解释。
- 最可能的误读：原图反光与 attribution 混淆，原图和 overlay 已分栏。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #8: Supplement S1 fixed audit 2, order 145

- 图能说明什么：第三个 fixed audit example 的两方法空间形态和 curves。
- 图不能说明什么：不能从个例推出普遍定位能力。
- 最可能的误读：matched contours 被解释为与 NNS 同幅度。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #9: Supplement S1 fixed audit 3, order 326

- 图能说明什么：第四个 fixed audit example 的完整审计视图。
- 图不能说明什么：不能把横向 pattern 当网络学习内容的直接证明。
- 最可能的误读：matched 淡色被当成零，contours 显示非零支持。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #10: Supplement S1 fixed audit 4, order 380

- 图能说明什么：第五个 fixed audit example 的完整审计视图。
- 图不能说明什么：不能从鸟轮廓与 attribution 位置作因果推断。
- 最可能的误读：overlay 淡化被当 metric normalization；它只是 display-only alpha blending。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #11: Supplement S2 rule-to-image mapping

- 图能说明什么：完整列出五条规则并保留 order 372 重复命中。
- 图不能说明什么：不是新排名分析。
- 最可能的误读：重复行被当错误；这是协议要求的完整映射。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：不适用。FAILED/NA：fallback 状态由 manifest 绑定。
- panel 顺序和图注：五条冻结规则顺序正确。原图、map、overlay 对齐：不适用。插值伪影：无。

## Image #12: Supplement S2 best_effect, order 372

- 图能说明什么：best_effect slot 的完整定性模块。
- 图不能说明什么：不能称为总体 representative。
- 最可能的误读：同 Image #2，contours 只帮助看形态。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #13: Supplement S2 median_effect, order 44

- 图能说明什么：median_effect slot 的完整定性模块。
- 图不能说明什么：不能依据 E3 map 外观重选样本。
- 最可能的误读：局部 contours 被当幅度编码。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #14: Supplement S2 baseline_strong_nns_stronger, order 372

- 图能说明什么：与 best_effect 重复命中的 slot 被单独保留。
- 图不能说明什么：不能把重复样本当独立证据。
- 最可能的误读：重复渲染被计为第二个 sample；manifest 明确同一 ID。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #15: Supplement S2 weak_or_failure, order 281

- 图能说明什么：弱效果 slot 原样保留。
- 图不能说明什么：不能扩展为总体结论。
- 最可能的误读：较小 delta_fid 被当 scientific failure。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无，属于 valid fallback。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #16: Supplement S2 misclassified_or_difficult, order 387

- 图能说明什么：误分类规则 slot 原样保留。
- 图不能说明什么：不能推出 attribution 导致误分类。
- 最可能的误读：把 status 与 map 作因果绑定。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #17: Figure 6 median_effect matrix, order 44

- 图能说明什么：四个冻结坐标的 NNS、matched、完整 difference progression，matched 稀疏形态清晰可见。
- 图不能说明什么：不是 raw self-attention；跨坐标不能比较绝对 magnitude。
- 最可能的误读：把 local contours 当跨列可比阈值。
- 文字或 colorbar 裁切：无，精确色限在 scale manifest。字号：合格。NNS/matched 色限：每坐标一致、跨坐标不同。FAILED/NA：无。
- panel 顺序和图注：Original 加四列坐标、三行方法正确。原图、map、overlay 对齐：同一 crop。插值伪影：raw nearest，无 clipping/smoothing。

## Image #18: Supplement S4 fixed_audit_00, order 231

- 图能说明什么：fixed audit 0 的四坐标 progression。
- 图不能说明什么：不能比较坐标绝对强度。
- 最可能的误读：contours 被当额外归一化数值。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：每坐标一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #19: Supplement S4 fixed_audit_01, order 14

- 图能说明什么：fixed audit 1 的完整 progression。
- 图不能说明什么：不能比较坐标绝对强度。
- 最可能的误读：反光原图与 pattern 混淆。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：每坐标一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #20: Supplement S4 fixed_audit_02, order 145

- 图能说明什么：fixed audit 2 的完整 progression。
- 图不能说明什么：不能把热点解释为跨层因果演化。
- 最可能的误读：contours 被跨坐标比较。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：每坐标一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #21: Supplement S4 fixed_audit_03, order 326

- 图能说明什么：fixed audit 3 的完整 progression。
- 图不能说明什么：不能把 pattern 称为 raw attention。
- 最可能的误读：difference 被理解为新 metric；仅 display-only subtraction。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：每坐标一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #22: Supplement S4 fixed_audit_04, order 380

- 图能说明什么：fixed audit 4 的完整 progression。
- 图不能说明什么：不能从跨列颜色推断强弱变化。
- 最可能的误读：local contour 阈值被当全局共享阈值。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：每坐标一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #23: Supplement S4 duplicate rule sample, order 372

- 图能说明什么：同一样本对应两个 rule slots 的 progression。
- 图不能说明什么：不是两张独立样本证据。
- 最可能的误读：slot 数等同去重样本数；实际 12 slots / 11 unique。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：每坐标一致。FAILED/NA：无。
- panel 顺序和图注：两个 slot 名均显示。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #24: Supplement S4 median_effect, order 44

- 图能说明什么：与 Figure 6 同一 median_effect 样本的 supplementary progression。
- 图不能说明什么：不能形成第二份独立主文证据。
- 最可能的误读：Figure 6 与 S4 被计为不同样本。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：每坐标一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #25: Supplement S4 weak_or_failure, order 281

- 图能说明什么：弱效果样本的四坐标 progression。
- 图不能说明什么：不能把 weak primary delta 当 secondary failure。
- 最可能的误读：用某个 E3 hotspot 事后解释选样；选样早于 E3 map access。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：每坐标一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #26: Supplement S4 misclassified_or_difficult, order 387

- 图能说明什么：误分类样本的四坐标 progression。
- 图不能说明什么：不能推断误分类因果原因。
- 最可能的误读：跨列颜色被直接比较强度。
- 文字或 colorbar 裁切：无。字号：合格。NNS/matched 色限：每坐标一致。FAILED/NA：无。
- panel 顺序和图注：一致。原图、map、overlay 对齐：一致。插值伪影：无。

## Image #27: Supplement S4 typed failure order 117

- 图能说明什么：11,2 NNS 的 ig_conservation_failed_after_e3_r1_schedule 原位失败，matched 和其他坐标正常显示。
- 图不能说明什么：不能补推、复制或替代失败 NNS map。
- 最可能的误读：difference failure panel 被当另一独立失败；它来自同一缺失 NNS source。
- 文字或 colorbar 裁切：无，长 failure code 框内换行。字号：合格。NNS/matched 色限：有效 cell 每坐标一致。FAILED/NA：清晰原位保留。
- panel 顺序和图注：一致。原图、map、overlay 对齐：有效 cell 一致。插值伪影：无。

## Image #28: Supplement S4 typed failure order 352

- 图能说明什么：11,2 NNS 的 ig_conservation_failed_typed_terminal 原位失败，matched 和其他坐标正常显示。
- 图不能说明什么：不能重建缺失 NNS map。
- 最可能的误读：matched 11,2 被当 NNS 替代；方法行清楚分开。
- 文字或 colorbar 裁切：无，failure code 完整。字号：合格。NNS/matched 色限：有效 cell 每坐标一致。FAILED/NA：清晰原位保留。
- panel 顺序和图注：一致。原图、map、overlay 对齐：有效 cell 一致。插值伪影：无。

## 总体结论

- 28/28 review copies 通过视觉检查；无文字/colorbar 裁切、字号风险或 panel 顺序错误。
- raw map 使用 nearest；无 clipping、smoothing 或失败插补。overlay 的 bilinear resize 和 alpha 均为 display-only。
- matched 在共享主色限下通过明确记录的 local contours 可读；contours 不进入 metric，不支持方法间或坐标间 magnitude 比较。
- orders 117、352 的 FAILED/NA 原位保留，matched 和其他有效坐标照常显示。
- 未生成 Grad-CAM/Figure 7：F3 冻结协议禁止 CAM/Grad-CAM、模型加载和新 attribution，且没有 sealed Grad-CAM input。该比较只能由 F3 之后的前瞻性协议授权。
- 未生成最终组合 Figure 1/5/6/S1/S2/S4；当前产物仅为 F4 消费的独立模块。
