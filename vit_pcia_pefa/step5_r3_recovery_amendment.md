# 步骤 5 r3 前瞻性恢复修订

r2 完成并封存全部 8 个 DeiT-S 任务；四个 ViT-B attempt-1 在结果提交前触发每 worker 24 GiB GPU-memory cap。失败类型为 PAUSED_RESOURCE_CAP，不作科学结论。

r3 继承 r2 的全部科学合同，并按字节哈希复用其 8 个已完成 DeiT-S 结果。唯一实现修订是 Omega 的 VJP 从一次保留全端口二阶图，改为逐端口重建并释放；它计算相同的 (J_e^T-J_e)v，不改端口、方向、范数或阈值。r3 重新建立任务清单和新队列，只调度 8 个缺失的 ViT-B 任务。r1 与 r2 的资源用量均纳入最终累计账本。
