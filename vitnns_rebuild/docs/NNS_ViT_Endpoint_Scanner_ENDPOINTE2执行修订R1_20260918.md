# ENDPOINT-E2 执行修订 R1：取消全量单卡重放

日期：2026-09-18。该修订由用户在 4 卡 formal queue 已完成 800/800、全量单卡
工程重放进行中时明确签发：

> 不要单卡重放了，没意义啊，多卡跑完就行了，你设计规划的有问题。

因此立即停止全量单卡重放。已完成的 86 个 replay task 和中止时的 1 个 claim
保留在原 root，状态统一解释为
`USER_CANCELED_ENGINEERING_REPLAY_NOT_E2_EVIDENCE`；不得补跑、删除、覆盖，也不得
用于 E2 PASS 或任何科学结论。

本修订承认原执行设计把 EP1 已完成的 worker-count 工程回归误扩成了 E2 全 800
任务重算。E2 的科学验收对象是正式多卡生产全集，而不是再次改变 worker 数量重放
同一全集。EP1 R11 的 single/multi 工程合同保持既有证据；E2 改为只验证：

1. formal queue 800/800 唯一 terminal，0 missing、0 duplicate、0 invalid；
2. 两模型分别完整覆盖冻结的 457/385 个 target，合计 336,800 个有限结果；
3. 400 个 tar member、checkpoint bytes/tensor、preprocessing、target identity 和 raw
   artifact hashes 全部重新绑定；
4. 800 个 class-map task、前 16 evaluation ID × 2 模型的 32 个 full raw atlas、
   固定 gallery、near-zero/failure index 完整；
5. wall/GPU-hours、GPU/CPU memory 和 retained bytes 均未越 cap。

该修订不改变 cohort、模型、目标、Endpoint 方法、数值结果、atlas、资源 cap 或
主张边界；仍禁止 Path IG、fidelity/stability、finite intervention 和 utility claim。
它发生在 formal 结果完成后，因此只作为用户明确要求的执行/资源修正，不声称是
outcome-blind 科学协议修改。

机器可读修订为
[`endpoint_e2_atlas_execution_amendment_r1.json`](../config/nns_vit_endpoint/endpoint_e2_atlas_execution_amendment_r1.json)。
通过出口仍唯一为：

```text
ENDPOINT_E2_PASS_NEXT_ENDPOINT_E3_REQUIRES_EXPLICIT_REQUEST
```
