# 步骤 5 r2 前瞻性恢复修订

r1 在首批四个任务的稳定性阶段因 worker 从协议顶层读取 noise_sigma_rgb 而触发 KeyError。协议实际将该值冻结在 stability.noise_sigma_rgb。四个 attempt-1 均未写入结果文件，r1 保留 queue、日志与 failure_receipt.json，状态为 TERMINAL_INFRASTRUCTURE_FAIL。

r2 只修复配置字段读取为 protocol[stability][noise_sigma_rgb]。模型、16 个任务、8 张 calibration 前缀图、目标、参考、路径、切面、512/16 节点、精度、干预、统计阈值、资源上限和 claim 规则全部继承 r1，禁止根据 r1 中间计算改动。r2 使用新 root、重新冻结源码哈希和新队列；r1 计算时间纳入 r2 的累计资源账本。
