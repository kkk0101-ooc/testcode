# NNS-ViT Endpoint Scanner：ENDPOINT-E3 结果

日期：2026-09-18。权威状态为：

```text
ENDPOINT_E3_PASS_CORRECTNESS_ONLY_NEXT_ENDPOINT_E4_REQUIRES_EXPLICIT_REQUEST
```

正式计算使用 EP0 冻结的 400 张 evaluation 图像和两个模型，共 800/800 个唯一
图像—模型任务。每个任务覆盖全部 12 个 block 和 8 个注册 cut，因此完成 9,600 个
block 实例、76,800 个正向闭合检查；每个 block 的 3 个 checker 负例也全部拒绝，
共 28,800 个负例。队列无缺失、无重复、无失败和重试。

所有正向检查均在 EP0 float32 合同 `atol=1e-5, rtol=1e-4, near-zero=1e-6`
下闭合。最大 relative L2 为 `1.4144569831842091e-05`；最大 absolute error 为
`2.47955322265625e-05`。所有 76,800 个 reference 都进入非零 relative 分支。最大
absolute error 大于单独的 `atol` 并不构成失败，因为冻结的 elementwise 判据是
`abs(error) <= atol + rtol * abs(reference)`，同时非零 reference 要求 relative L2
不超过 `rtol`；两项均逐 cut 验证通过。

前 16 个 evaluation order × 两模型保留了 32 个任务、384 个 block 的原始 float32
component/reconstruction maps。R2 独立 CPU 验证从这些 raw arrays 重算 component
求和、`A_F/L_F` reference、max-absolute、relative-L2 和 elementwise 容差，并重新验证
800 个不可变任务结果、400 个输入文件、所有来源/实现 hash、队列 coverage 与资源上限。
R2 没有重跑 GPU，R1 结果和 terminal 未覆盖。

正式 8 卡 wall time 为 712.66 秒，按 8 卡上界计 1.5837 GPU-hours；峰值 reserved
GPU memory 约 2.13 GiB，CPU RSS 约 1.65 GiB，封存前 retained bytes 约 4.56 GB，
均低于冻结 cap。

本结果只支持 RQ2 的结构 decomposition correctness：在这两个冻结 timm 标准
ViT、ground-truth raw logit、400 张正式样本和注册 cut 范围内，head、Q/K/V、
routing/value、attention residual 与 MLP residual 的记账无遗漏闭合。它不支持有限
干预 utility，不运行 Path IG、input fidelity/stability 或 activation replacement。
ENDPOINT-E4 仍须用户另行明确授权。

机器可读权威入口：

- R2 terminal：`reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json`；
- R2 validation：`reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json`；
- R2 verifier-only amendment：`config/nns_vit_endpoint/endpoint_e3_closure_handoff_recovery_r2.json`；
- 不可变 R1 terminal：`reports/nns_vit_endpoint_e3_closure_r1_20260918/terminal_receipt.json`。

阶段完成后，统一路线图按职责更新为 E3 PASS / E4 待授权；独立验证器通过 R1 的
producer commit 校验执行当时冻结的路线图字节，避免把当前导航文档的正常状态迁移
误判为历史输入漂移。
