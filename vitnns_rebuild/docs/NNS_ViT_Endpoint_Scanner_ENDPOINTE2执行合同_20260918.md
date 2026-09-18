# NNS-ViT Endpoint Scanner：ENDPOINT-E2 执行合同

日期：2026-09-18。用户以
`ENDPOINT_E2_ATLAS_REQUIRES_EXPLICIT_USER_REQUEST` 明确授权本阶段。本文只把
[统一研究计划](NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md) 中已经
冻结的 ENDPOINT-E2 条款绑定到可执行代码；不修改 EP2-R1，也不授权 ENDPOINT-E3。

```text
STAGE=ENDPOINT-E2
METHOD=canonical Endpoint x*grad_x(s)
TASKS=400 evaluation images x 2 models
TARGETS=457 ViT-B/16; 385 DeiT-S/16
CLAIM=C2_REGISTERED_SCAN_COMPLETION_AND_RESOURCES
PATH_IG=FORBIDDEN
FIDELITY_STABILITY_INTERVENTION=FORBIDDEN
UTILITY_CLAIM=FORBIDDEN
```

机器可读权威为
[`endpoint_e2_atlas_run_spec.json`](../config/nns_vit_endpoint/endpoint_e2_atlas_run_spec.json)。
它逐哈希绑定 EP0 cohort/model/target/evaluation registry、EP1 terminal、主轨修订
terminal 和统一计划。生产代码、测试与独立验证器必须先提交并推送，且执行前
`HEAD == tracking == remote`。

## 数据、产物和数值约束

只从冻结 ImageNet validation tar 物化 400 个 evaluation member；每个成员记录
SHA-256，不换图。两个模型严格绑定 checkpoint byte hash、state tensor digest、
timm 名称与预处理。每个 image-model task 使用一次共享 forward，并对注册表中每个
scalar 做一次 VJP。原始 map 始终为 normalized input 坐标的 float32 signed RGB
`[3,224,224]`。

- 全 800 个任务保留两个 class target 的原始 signed RGB maps；
- evaluation 顺序前 16 张、两个模型保留全部目标 raw atlas；
- 全 336,800 个 target evaluation 保留 value、signed sum、absolute mass、max abs、
  target identity、model/input hash 和 map digest；
- near-zero 定义为 `max_abs <= 1e-12`，不删除数值行；
- evaluation 顺序前 8 张固定生成 gallery，线性对称 own-max 显示，无 smoothing 或
  clipping；显示产物不进入统计。

正式生产使用 1–8 张合格 H20、SQLite WAL 全局动态队列、180 秒 lease、30 秒
heartbeat、最多两次 attempt 和 fencing token。随后用一张 H20 对完全相同的 800 个
任务重放；逐任务逐目标 numeric digest 必须精确相同。两遍均在 Torch/CUDA import
前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，关闭 TF32/autocast，并启用 deterministic
algorithms。

## Fail-closed 出口

任一 source drift、遗漏、重复、nonfinite、hash/digest 不一致、超时或资源 cap 触发
均保留新 root 并失败关闭；不对有利子集发布 PASS。只有独立 CPU 验证器从原 tar、
冻结 registry、SQLite queue、raw NPY 和 terminal 全量重算通过时，出口才是：

```text
ENDPOINT_E2_PASS_NEXT_ENDPOINT_E3_REQUIRES_EXPLICIT_REQUEST
```

该出口只允许下一次请求 ENDPOINT-E3；本轮执行到 E2 后停止。
