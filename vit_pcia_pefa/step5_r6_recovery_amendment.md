# 步骤 5 r6 前瞻性恢复修订

r5 在新 ViT-B 结果落盘前的隔离 finalizer 演练中发现汇总 schema 错位：D/Q/R 原始结果的类别表位于 `registered_categories`，Omega 的类别表位于 `categories`，r5 finalizer 对二者错误地使用同一字段。r5 因此停止并以 `TERMINAL_INFRASTRUCTURE_FAIL` 封存。

r6 只按量类型读取既有类别字段，不改变任何任务计算、原始结果、统计量或 claim 门。40 GiB per-worker 显存门及模型、样本、目标、参考、路径、端口、切面、512/16 节点、精度、容差、干预、稳定性、阈值全部继承。r6 冻结前必须在隔离临时副本上完成 finalizer 全流程演练；r6 继承 8 个已验证 DeiT-S 结果，仅运行 8 个缺失 ViT-B，最终资源账本累计 r1-r6。
