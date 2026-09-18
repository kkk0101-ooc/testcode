# Endpoint Scanner 结果解释 R4

## RQ3：成分图能否预测匹配的有限干预

RQ3 整体得到支持。两个模型、两个 reference 和三个成分族构成的 12 个预注册 C4a family 中，有 11 个满足全部确认性条件。唯一例外是 ViT-B 在 normalized-zero reference 下的 routing/value family：其关联方向为正且通过 Holm 校正，中位相关为 0.099565，仅比预设的 0.10 实际效应阈值低 0.000435。

因此，整体证据支持 RQ3，同时保留一个注册边界例外。冻结的全 family omnibus C4a 字段仍为 `NOT_SUPPORTED`，因为其机器规则要求 12/12 family 全部通过；这个字段记录的是最严格的无例外普遍性门，而不是“整体没有预测关联”。

## RQ4：空间信息能否超过 mandatory baselines

RQ4 不成立。四个 C4b family 都没有同时以注册的 0.05 增量超过 total Gradient × Input、raw gradient、regional Taylor 和随机区域排序。Endpoint 相对随机区域在四个 family 中都有小幅正中位增益（0.0011–0.0107），但没有达到实际效应阈值；regional Taylor 在四个 family 中的中位分数都更高。

## 论文级结论

在所评估的 ViT-B/16 与 DeiT-S/16 上，结构化 Endpoint 成分质量总体能够预测匹配的有限 activation effect；完整分辨率空间排序没有证明相对 mandatory baselines 的注册信息增益。
