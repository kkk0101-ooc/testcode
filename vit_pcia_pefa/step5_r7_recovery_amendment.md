# 步骤 5 r7 前瞻性恢复修订

r6 outcome 已知，终态为 `NOT_SUPPORTED_EMPIRICAL_VALUE`。随后逐条合同审计发现 r6 的 seed、Omega 完整切面零和与运行身份记录没有覆盖 `method_contract.md` 的既有要求。r7 修订原因仅是补齐先前冻结合同，不按 r6 效应方向改变样本、方法、指标、阈值或 claim 门。

r7 使用精确 seed `int.from_bytes(SHA256("ENDPOINT_PCIA_PEFA|step5|task_id")[:8], "big")`，并记录每任务 Python、NumPy、Torch CPU/CUDA 的执行前后 RNG state、driver、GPU UUID、Python/包/CUDA、freeze-time git commit/dirty snapshot、checkpoint hash 与 input hash。每个 block 的 `HEADS_WITH_BYPASS_FULL` 在逐端口 streamed Omega 后按原始 signed normalized-RGB 坐标求和，并用既有 ViT `atol=1e-5, rtol=1e-4, near_zero=1e-6` 验收零场；失败即 `FAIL_QUADRATURE_CLOSURE:OMEGA_ZERO_SUM`。

seed 修复会影响随机基线、classifier randomization、小噪声与 Omega 方向，因此 r7 不继承 r6 或更早的任务 outcome，重新运行全部 16 个任务。模型、8 张 calibration 前缀图、ground-truth target、参考、路径、端口、切面、512/16 节点、精度、干预、稳定性条件、统计单位、0.05 增量阈值、40 GiB 显存门及其他资源上限全部不变。r1-r6 的 7.078883001111686 H20-hours 与冻结前 discovery Omega 探针的 0.0053738992002777776 H20-hours一并计入 r7，前驱上界为 7.0842569003119635 H20-hours。步骤 6 仍未授权。
