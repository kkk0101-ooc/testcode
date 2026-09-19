# Endpoint M6 自动案例与可视化 atlas 执行协议

本轮只执行《阶段M_endpointnnsvit.md》的 M6。目标是把已封存 M3、M4 的统计规律转为可审计案例图，并加入一个 M2 方法语义边界案例；不新增假设检验，不重跑完整 cohort，不进入后续阶段。

## 1. 前置门与来源

1. M3 主根必须为 `ENDPOINT_M3_PASS_EXECUTION_AND_AUDIT`，M4 主根必须为 `ENDPOINT_M4_PASS_EXECUTION_AND_AUDIT`，M4 Review-R2 必须为 `ENDPOINT_M4_REVIEW_R2_PASS_POST_HOC_SENSITIVITY`，M5 必须为 `ENDPOINT_M5_PASS_EXECUTION_AND_AUDIT`。
2. 配置中列出的 terminal receipt、artifact manifest、task manifest、coverage audit 和阶段文档须重算 SHA-256。
3. 每个实际读取的 M3 metrics、M4 result 和 M2 result 文件还须逐文件匹配其已封存 artifact manifest。
4. 任一状态、文件哈希、任务身份或原图身份不符即停止。

## 2. 自动选择

极端机制案例使用完整封存记录的经验 5% 尾部，并在包含阈值并列项后按 `sample_id -> model_id -> block -> token -> channel -> head -> residual_type` 取最小项：

- early patch locality、late patch entropy、CLS-minus-patch entropy 和 neuron signed cancellation 取最高 5%；
- routing share 和 frozen-attention Endpoint error 取最高 5%；
- value-dominant head 取 routing share 最低 5%；
- identity share 最高/最低 5% 分别定义 identity-/branch-dominant residual；
- residual path cancellation 取最高 5%。

模型层级差异使用两模型共享图像的 `late_minus_early_entropy` 差值，选最接近跨图中位数的图像；随后在 block 0/11、token 30 中选择最接近该图该层 channel 中位 entropy 的单元。M2 案例固定 DeiT-S、block 5、patch geometry 和 normalized-zero，只按 Endpoint absolute 与 regional Taylor patch score 的 `1-Spearman` 选择最接近候选中位数的 image-head；不读取 finite-effect 输赢来选择。所有并列均按固定 ID 顺序处理，禁止看图后复选。

## 3. 选择后提取

选择阶段只读既有数值产物且清空 `CUDA_VISIBLE_DEVICES`。M3/M4 旧产物未保存全部候选的像素级分解图，因此选择封存后只对选中的 model-image-target 重新执行一次 deterministic forward/VJP：

- 导出 M3 选中 token-channel 的原始 signed float32 `[3,224,224]` Endpoint；
- 导出 M4 选中 head 的 full/routing/value/Q/K/V Endpoint 与 CLS-query attention；
- 导出 M4 选中 residual 的 output/identity/branch Endpoint；
- 将重算 summary 与原封存记录交叉核对；
- 使用 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、禁用 TF32/autocast、启用 deterministic algorithms。

这一步只补足呈现数组，不改变选择、统计结论或既有根。

## 4. 图与解释边界

必要图为：neuron atlas、两模型层级对比、attention cases、residual cases 和 M2 semantic boundary。PNG/PDF 同时交付。signed 图按 RGB 求和后做面板内对称显示；absolute/patch mass 按 RGB 绝对值求和；无 smoothing。颜色缩放仅用于显示，数值均来自 raw arrays 与 selection manifest。

图注统一使用 `local input dependence` / `Endpoint dependence`。Routing/identity share 是相对 path-norm scale，不是因果贡献百分比；M2 图展示不同 estimand，不作为新方法胜负检验。

## 5. 验收与停止

M6 必须交付 selection manifest、raw arrays、五张 PNG/PDF、figure manifest、中英文案例说明、证据矩阵、防御性写作审计、独立验证、视觉检查、prompt-to-artifact completion audit、artifact manifest 与 terminal receipt。只有 12 类案例和 14.3 的每个面板要求均被真实文件覆盖时，才签发 `ENDPOINT_M6_PASS_EXECUTION_AND_AUDIT`，并保持 `next_authorization=STOP_AFTER_M6`。
