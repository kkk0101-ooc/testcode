# Endpoint M4 外部评价修订协议 R1

本轮接受外部评价中关于 estimand 和措辞的核心判断，并在不修改已封存 M4 根的前提下生成版本化补充根。状态固定为 review-driven post-hoc descriptive/sensitivity；原 M4 的 16 项冻结检验、terminal receipt 和论文交付不重写。

## 评价到交付映射

| 评价要求 | 处理 | 证据 |
|---|---|---|
| 明确 target-conditioned | 所有 M4 分量均表述为 frozen ground-truth logit 的 downstream cotangent 加权局部输入 Jacobian | 修订版方法、结果、摘要和结论 |
| Routing share 不是贡献百分比 | 改称 relative Endpoint path-norm scale | estimand 表与全文禁用语检查 |
| JVP 0.95 不是遗漏 95% | 改称 normalized RMS directional discrepancy | 分子、分母、绝对误差和分位数表 |
| 排除小分母膨胀 | 报告 block 11 及全层 full-response denominator 的 q10/q25/median/q75/q90 | JVP distribution 表和图 |
| Saturation 稳健性 | mean-query/CLS-query；未调整、控制 head-output norm、再控制 downstream cotangent norm；按 phase 报告 | sensitivity 表与 CI |
| Residual share 不是因果比例 | 改称 identity/branch relative path-norm scale | 修订版结果 |
| 重写需要空间证据 | 对 output–identity patch TV 做 late−early image-level CI/sign/Holm；报告 branch-share–TV 相关 | residual spatial 表和图 |
| MLP cancellation 模型差异 | 同时报告校正检验和方向，不合并为统一规律 | 修订版结果 |
| Cut 位置与 cotangent | 明确 pre-LN、attention AV cut、concat/projection 下游 cotangent、两次 residual clean cotangent | 实现附录 |
| 最大误差位置 | 定位模型、图像、block、head/residual type 和误差值 | numeric diagnostics |
| M3–M4 连接 | E3 事前 raw-map 子集计算 full/value entropy 差和 TV；与 M3 仅比较模型级深度轨迹 | bridge 表和图 |

## 硬边界

- CPU-only；CUDA_VISIBLE_DEVICES 为空；不导入 Torch/timm；不读取 JPEG；不加载 checkpoint。
- M4 完整 400 图分析只使用封存 JSON；M1 head norm 控制只使用封存 per-image JSON。
- Full/value 空间分析只能使用 E3 事前保留的每模型 16 个 raw-map tasks。M3 与该子集每模型仅有 1 张图重叠且目标不同，因此不执行逐图因果连接检验。
- H(full)-H(value) 和 patch TV 是 approximation-difference 描述量；entropy 不可加，不能解释为 routing 的可加空间贡献。
- 原 M4 文件不修改。修订版论文使用后缀 m4_review_r1 和 figures/m4_review_r1。
