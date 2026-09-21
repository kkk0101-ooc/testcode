# 步骤 5 r8 前瞻性恢复修订

r7 在 40 GiB 门内完成 14/16 个任务；`vit_b16__02` 和 `vit_b16__05` 均在两次允许 attempt 中确定性触发 `FAIL_QUADRATURE_CLOSURE:OMEGA_ZERO_SUM`。r7 已以 `TERMINAL_MATHEMATICAL_CLOSURE_FAIL` 封存，该失败不是资源失败或实证阴性。

固定 seed 的诊断显示，float32 逐端口求和、直接完整切面和 full-gradient curl 均可出现反对称消去残差；同一 checkpoint、图像、类别、seed 和 blocks 0/5/11 在 float64 下恢复到 `2.56e-13` 以下，原 `atol=1e-5, rtol=1e-4, near_zero=1e-6` 不变。完整 head 切面 `identity + sum(heads)` 与已注册的粗粒度切面 `identity + attention` 代数等价；后者的 float64 直接 curl 审计在 block 11 峰值仅 3.05 GB。

r8 因此前瞻冻结双精度职责：逐端口 Omega 科学分数与其他模型计算仍用 float32；数学零场硬门由 float64 的 `identity + attention` 直接完整切面 curl、full-gradient curl 及二者一致性共同承担。r8 对全部 16 个任务补做该独立审计，继承 r7 的 14 个 canonical-valid 科学结果，只重算两个未完成任务。模型、样本、目标、seed、方向、节点、干预、统计量、容差、40 GiB 显存门和五项 claim 规则均不变。

r8 只恢复原步骤 5 pilot。用户新增授权的闭环交换子、非线性条件与分辨率阶梯另建后续 extension root，不得回写 r8 的原 pilot claim。步骤 6 仍未授权。
