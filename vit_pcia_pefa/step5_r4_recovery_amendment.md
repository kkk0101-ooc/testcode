# 步骤 5 r4 前瞻性恢复修订

r3 继承的 8 个 DeiT-S 任务保持完成；首批四个 ViT-B attempt-1 均在结果提交前触发冻结的 24 GiB per-worker peak-reserved GPU-memory cap。经明确授权，r3 的自动 attempt-2 在完成前停止，r3 以 `TERMINAL_PAUSED_RESOURCE_CAP` 封存，不作科学结论。

r4 只把每 worker GPU-memory cap 从 24 GiB 提高到 40 GiB。H20 资格门、最多 4 卡、64 H20-hours、72 wall-hours、128 GiB 保留文件、32 GiB RSS、1800 秒 task timeout 和最多 2 attempts 均不变。模型、8 张 calibration 前缀图、目标、参考、路径、端口、切面、512/16 积分节点、精度、干预、稳定性、统计量、阈值和 claim 规则全部不变。

r4 按 canonical digest 继承 r3 中 8 个已验证的 DeiT-S 结果，只调度 8 个缺失的 ViT-B 任务。最终资源账本累计 r1、r2、r3 与 r4；r3 的中断 attempt-2 以日志最后修改时间作为保守结束时刻。最终报告对已经冻结并写入任务结果的 D/Q/R/Omega 注册类别统计做完整展开；协议中的 `input_only_routing_total_variation` 与任务结果中的等义字段统一为 `routing_total_variation`；干预预测按原冻结文本重算为候选相对其他方法的 0.05 增量，而非绝对相关门。这些都是执行前的交付/判定实现一致性修正，不新增或选择 outcome、指标或科学阈值。
