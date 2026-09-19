# M3-SUP-R1 补充分析协议

本轮只补充 M3 的展示、误差分布与数值诊断。来源为已封存的 `nns_vit_endpoint_m3_r1_20260919`；原根及其 `ENDPOINT_M3_PASS_EXECUTION_AND_AUDIT` receipt 保持不变。补充分析不新增 confirmatory hypothesis，不改变原 10 项 Holm family，也不进入 M4。

## 必做交付

1. 原始 NNS 等价的 relative L2、absolute error 和 tolerance ratio 分布：总体、分网络、分精度及 network×precision，报告中位数、90%、99% 与最大值。5000 个 target comparisons 只作为数值案例，不作为独立统计样本。
2. 11 类 ViT 算子的逐类表：方向数、float64、最大解析–autograd JVP 误差、最大 JVP–VJP 对偶误差、最佳多步长有限差分误差、epsilon 列表和验证方式。
3. 两模型 12 个 block 的 entropy、own-token locality、normalized distance、cancellation 与 `R_net=|sum E|/sum|E|` 完整轨迹；每图先跨冻结 units 取中位数，再跨 100 图报告中位数与 10,000 次 bootstrap CI。
4. CLS–patch entropy 在 overall、early/middle/late 和各 block 的图像级配对差值。每张图、每个 stratum 内分别聚合 CLS 与 15 个 patch tokens，再作差；属于 supplementary descriptive analysis，不加入原 Holm family。
5. cancellation 多尺度诊断：令 `A=sum_(c,h,w)|E|`，`B=sum_(h,w)|sum_c E|`，`P=sum_patch|sum_(c,h,w in patch)E|`，`N=|sum E|`。报告 RGB 通道内抵消 `1-B/A`、RGB 聚合后的 patch 内抵消 `1-P/B`、patch 间 signed 抵消 `1-N/P`、全图抵消 `1-N/A` 和 `R_net=N/A`；生存率满足 `N/A=(B/A)(P/B)(N/P)`。patch-absolute 聚合的总质量恒等于 `A`。对每模型前两图的同 6 个目标用 float64 原始模型重新计算，报告 float32/float64 差异。
6. 36 个 float32 有限差分失败的 epsilon-error 曲线、float32/float64 最佳容差比、目标 block/token/channel 和 analytic magnitude。错误来源只在曲线证据支持时描述为 finite-difference precision/cancellation；不按算子归因，因为目标是 post-block token-channel，不对应某个单独 LayerNorm/softmax 算子。
7. 版本化中英补充稿、数字表、PDF/PNG 图、证据矩阵、写作审计、完成清单及 terminal receipt。

## 结论边界

正式主发现表述为 `spatial broadening of local input dependence` / “层深相关的局部一阶输入依赖空间扩展”。“局部走向全局”仅可作为非严格简称。CLS entropy 是 construct-validity evidence，不是因果信息量。Cancellation 用于区分 absolute mass 与 net first-order directional effect。覆盖范围为每模型每图 6,144 个冻结 post-block token-channel 单元；不覆盖全部 MLP hidden units、Q/K/V coordinates 或全部 token-channel 组合。
