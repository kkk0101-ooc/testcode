# Stage F F2 图片分析

状态：`PASS`

全部图片均为 `POST_D10_SECONDARY_DESCRIPTIVE`，使用 F0 冻结 renderer 和 E4 封存 paper data；未生成最终 Figure 3、Figure 4 或 Supplement S3。

## Image #1: F2-A single-column

- 这个图能说明什么：展示完整 valid rows 的 fine 4x4 anchor、native 14x14 NNS/matched delta_fid 分布及相对 anchor 的关系。
- 这个图不能说明什么：不能替代 D10 primary，也不能据此断言尺度因果或优胜方法。
- 读者可能误读什么：native 曲线靠近零可能被误读为总体无效；它只描述冻结尺度下的敏感性。

## Image #2: F2-A double-column

- 这个图能说明什么：与 Image #1 相同数据，宽版便于比较 anchor 与 native 的分布位置。
- 这个图不能说明什么：不能当作独立复现实验或第二个统计检验。
- 读者可能误读什么：版式更宽不代表证据更强。

## Image #3: F2-B single-column

- 这个图能说明什么：展示 insertion/deletion 对 upsampling operator 的描述性敏感性，并把 uniform/nearest 合并为同一等价证据类。
- 这个图不能说明什么：不能选出最佳 operator，也没有 confirmatory p-value。
- 读者可能误读什么：曲线位置差异不应被解读为预注册 winner ranking。

## Image #4: F2-B double-column

- 这个图能说明什么：与 Image #3 相同 source rows，以宽版并排观察 insertion 和 deletion。
- 这个图不能说明什么：不能把 uniform 与 nearest 当成两份独立证据。
- 读者可能误读什么：重叠等价类不增加样本量。

## Image #5: F2-C single-column

- 这个图能说明什么：展示四个冻结 layer/head 坐标在 overall/correct/misclassified 三个 strata 的 source median 与 interval。
- 这个图不能说明什么：不能把 descriptive interval 当成 confirmatory interval，也不能覆盖 MIXED validity。
- 读者可能误读什么：不同 metric 同轴仅供结构化浏览，不是跨 metric 优劣排序。

## Image #6: F2-C double-column

- 这个图能说明什么：与 Image #5 相同 60 个 source summary rows，宽版便于核对坐标和 strata。
- 这个图不能说明什么：不能依据视觉高低选择 layer/head winner。
- 读者可能误读什么：L11/H2 的 valid-only 数值不包含两个失败值；失败并未补零。

## Image #7: F2-D single-column

- 这个图能说明什么：同时披露 expected/valid/failed/NA、orders 117/352 typed failures，以及 E2 owner budget waiver。
- 这个图不能说明什么：不能把 E4 package PASS 解读为全量 scientific PASS。
- 读者可能误读什么：valid 2998/3000 不等于失败不存在，E2 waiver 也不等于从未超预算。

## Image #8: F2-D double-column

- 这个图能说明什么：与 Image #7 相同 QC 和 failure/cost disclosure，宽版便于人工审阅。
- 这个图不能说明什么：不能产生新 GO/NO-GO 或修复 typed failures。
- 读者可能误读什么：failure marker X 表示缺失科学数值，不是数值零。
