# NNS-ViT Endpoint Scanner：ENDPOINT-E3 执行合同

日期：2026-09-18。本阶段由用户明确指令“执行 ENDPOINT-E3”授权，只回答 RQ2：
两个冻结模型在 400 张共享 evaluation 图像、全部 12 blocks 上的注册结构切面是否闭合。

```text
ENDPOINT-E2_AUTHORITY_REVALIDATED
ENDPOINT-E3_FORMAL_CLOSURE_AUTHORIZED
PATH_IG_NOT_AUTHORIZED
ENDPOINT-E4_NOT_AUTHORIZED
```

机器合同是 [run spec](../config/nns_vit_endpoint/endpoint_e3_closure_run_spec.json)。
每个任务固定为一张图像乘一个模型，目标仅为冻结 ground-truth raw logit；对 12 个
block 逐一计算 8 个切面：head branch/total、Q/K/V branch/total、routing/value
branch/total、attention residual 和 MLP residual。`A_F` 与 `L_F` 不混用。

生产使用 float32 的 EP0 容差：`atol=1e-5`、`rtol=1e-4`、near-zero norm
`1e-6`。验收同时要求 elementwise tolerance，以及非零 reference 的 relative-L2
分支或 near-zero reference 的 absolute 分支。全部输入、目标、component、重构和误差
必须有限。每个 block 还运行三个必须失败的 checker 负例：遗漏最大非零 head、遗漏
非零 routing、重复 identity residual。正向科学闭合使用上述正式容差；负例使用零容差，
从而验证任何非零漏边都不会被 checker 静默接受。两种容差在结果中分开记录。

调度沿用动态全局 SQLite 队列、lease/heartbeat/fencing、每卡常驻模型 worker 和
1–8 张合格 H20。正式输入继承 E2 已物化且逐文件哈希的 400 个 payload，不换图；
执行前仍逐任务验证图像、checkpoint、tensor、cohort、target registry 与代码身份。
前 16 个 evaluation order 保存所有 component 与 `A_F/L_F` 原始 float32 RGB map，
其余任务保存完整数值摘要与 digest。

通过只允许写成结构 decomposition correctness；禁止 Path IG、输入 fidelity/stability、
activation replacement、有限干预和 utility 结论。通过出口为：

```text
ENDPOINT_E3_PASS_CORRECTNESS_ONLY_NEXT_ENDPOINT_E4_REQUIRES_EXPLICIT_REQUEST
```

缺失、重复、nonfinite、来源漂移、负例 false-pass 或资源 cap 均 fail closed 并保留
现场；任何恢复必须使用新 root，不覆盖 terminal root。
