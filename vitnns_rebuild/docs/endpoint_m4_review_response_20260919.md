# Endpoint M4 外部评价逐项回应

总体判断：评价合理。原 M4 的数值结论保留，但正式论文需要把“目标条件化局部 Jacobian”“相对路径 norm”“归一化方向误差”和“有限因果作用”严格分开。M4-Review-R2 已按此原则完成修订。

| 评价点 | 处理结果 | 新证据或修改 |
|---|---|---|
| 加入目标相关范围 | 接受 | 全文明确 M4 固定 ground-truth logit，所有路径均包含 downstream cotangent |
| Routing share 不是贡献百分比 | 接受 | 改称 relative Endpoint path-norm scale；禁用 percentage-contribution 解释 |
| 0.95 JVP error 不是遗漏 95% | 接受 | 新增 q10/q25/median/q75/q90、absolute discrepancy 和 full-response denominator |
| 排除小分母膨胀 | 已完成 | Block 11 分母中位为 14.9711/1.5155，q10 为 5.5214/0.5719；不是普遍近零 |
| Saturation 潜在混杂 | 已完成 | 控制 head-output norm 和 downstream-cotangent norm；另报 CLS query 与 early/middle/late |
| Residual share 不能直接叫重写 | 接受 | Share 只解释相对路径尺度；“空间重写”另由 output–identity patch TV 支撑 |
| 空间改变需正式统计 | 已完成 | 400 图 late−early TV、bootstrap CI、sign test、四项 Holm；attention 为正、MLP 为负 |
| MLP cancellation 模型差异 | 接受并加强 | 原 M4 两项均通过校正，可报告两模型显著相反趋势，不报告统一规律 |
| Cut/cotangent 细节 | 已补充 | 说明 AV cut、concat/output projection 的 downstream cotangent、pre-LN 两次 residual clean cotangent |
| 最大误差位置 | 已补充 | 四类最大误差定位到模型、图像、block、head/residual type |
| M3–M4 机制链 | 部分支持 | 16 raw class-logit maps/model 显示 Full–Value TV 随深度增大；与 M3 仅做模型级 block trajectory 对照 |
| “Q/K 导致 broadening” | 不支持 | M3 与 bridge 的 target/cohort 不同，每模型仅一张图重叠；禁止逐图因果解释 |

## 修订后的结论层级

1. **直接支持：** late attention 的 routing relative path-norm scale 和 Value-only discrepancy 高于 early attention。
2. **稳健性支持：** mean-query saturation 的负相关在控制 output/cotangent norm 后仍保留；CLS-query 结果更弱且在 ViT-B 两项 fully adjusted tests 中未通过 Holm。
3. **空间证据支持：** attention residual 的 output–identity spatial TV 随深度增加，MLP residual 则下降。
4. **模型依赖：** MLP path cancellation 在 ViT-B/16 上升、DeiT-S 下降。
5. **描述性桥接：** M3 broadening 与 Full–Value spatial discrepancy 的深度轨迹方向相容，但不能推断 routing 导致 M3 broadening。

## 推荐论文一句话

> 在所测试模型、当前输入和冻结 ground-truth logit 下，late attention 的目标条件化局部输入 Jacobian 更难由固定 attention weights 的 Value-only 路径近似；attention residual 的 branch-relative path norm 与空间改变随深度增强，而 MLP residual 的 identity-relative path norm 增强。这些是局部 Jacobian 路径和空间分布结果，不是函数值贡献比例或有限因果重要性。
