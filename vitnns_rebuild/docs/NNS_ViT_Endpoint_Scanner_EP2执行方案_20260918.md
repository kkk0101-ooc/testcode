# NNS-ViT Endpoint Scanner：EP2 校准执行方案

本阶段只执行 EP0 冻结的 `EP2_CALIBRATION_AND_SEAL`。前置状态是已发布的
EP1 PASS commit `ea7b7437f1e58389868d9cb6206d7e9681a81233`。

范围严格限于 EP0 登记的 64 张 calibration 图像和两个模型；只从官方
validation tar 打开这 64 个精确成员，逐成员绑定 SHA-256。Discovery 和
evaluation payload 读取计数必须为零，不允许缺失样本替换。

每个 `model × image` 任务固定 ground-truth logit 与 clean top-1 logit，使用
normalized-zero、直线路径和 Gauss–Legendre 节点计算 M=32/64/128/256/512
及其 2M 检查图。每个模型选择同时满足两个目标的最小 M：64/64 图必须有限，
每目标至少 61/64 图同时满足 map relative L2 ≤ 0.02 与 completeness 条件。
非近零 target delta 使用 residual / max(abs(delta), 1e-6) ≤ 0.01；
abs(delta)<1e-6 时使用 absolute residual ≤ 1e-5。最大候选失败即
`EP2_NUMERICAL_CALIBRATION_FAILED`，不得增加节点或按图调整。

Alpha chunk 按 8/4/2/1 固定顺序选择最大可行值。Endpoint target batch
用每图 16 个已注册目标验证；两个 IG 类别目标保持独立 VJP。执行使用动态全局
SQLite 队列、lease/heartbeat/fencing token、不可变 task result 和 exact coverage
audit。本阶段最多 16 H20 GPU-hours、24 wall-hours、24 GiB retained bytes；触限
时保存现场并 `PAUSED_RESOURCE_CAP`。

EP2 只冻结数值与调度参数，不计算评价指标、干预结果或论文 claim。PASS 后的
EP3 仍需用户另行明确授权。
