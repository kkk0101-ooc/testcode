# NNS-ViT Endpoint Scanner：ENDPOINT-E4 结果

日期：2026-09-18。权威状态为：

```text
ENDPOINT_E4_FAIL_LOCAL_DIRECTION_CONTRACT
```

## 1. 执行结果

E4 在已发布的 commit `9aa8ec4d19791c41a5dac0df673bee3a3fc09b19` 上启动。启动
门记录了 8 张符合资格的 NVIDIA H20、全部既有 PID、远端一致性以及 E0–E3 的完整
前置验证。正式队列预注册 400 张 evaluation 图像 × 两个模型，共 800 个唯一任务。

按照 fail-closed 顺序，最前面的 16 个 DeiT-S evaluation order 均在局部方向合同上
失败；每个任务完成两次允许的 attempt，32 个失败记录均为：

```text
LOCAL_DIRECTION_CONTRACT_FAILED
```

停止时队列状态为 16 failed、73 completed、8 claimed、703 pending。因主流程被人工
停止，claimed 项不视为完成或失败；完整队列未达到验收覆盖。R1 根保留，不覆盖、
不续写为 PASS。

## 2. 诊断与科学边界

独立诊断确认解析关系

$$
\langle g_k,J_{z_k,x}x\rangle=a_k
$$

在 float64 下达到约 $10^{-15}$ 的一致性，因此没有证据表明 edge hook、cotangent、
JVP 或 E3 component 记账实现错误。失败发生在预注册的有限 $\eta$ 曲线验收：对若干
真实 head、MLP、routing、Q/K 和 joint routing-value 成分，即使把整个局部诊断提升
为 float64，固定最小 $\eta=0.0003$ 的最小误差仍超过
$10^{-4}+0.01|a_k|$。例如首张 DeiT-S 图有 9/156 个成分仍失败。

这说明被冻结的四点 one-sided finite-$\eta$ acceptance 在这些真实成分上过严，不能
通过单纯提高数值精度补救；也不能在看到结果后增加更小 $\eta$ 或放宽阈值并仍称
原 E4 PASS。因此 E4 按原合同失败。该失败是局部 finite-$\eta$ 合同失败，不推翻 E3
结构闭合，也不等同于 Endpoint 有限 replacement utility 为负。

启动后已有 73 个不含局部合同的任务写出 finite replacement 结果，但由于 E4 的前置
局部门失败且全量 coverage 未完成，这些部分结果全部排除于科学解释，不计算
bootstrap、Holm 或 C4a claim decision。E5 未启动。

## 3. 后续状态

```text
NO_NEXT_STAGE_AUTHORIZATION
ENDPOINT_E5_NOT_AUTHORIZED
PROSPECTIVE_E4_RECOVERY_AMENDMENT_REQUIRED
```

任何恢复必须新建根，并在再次读取 outcome 前冻结新的数值合同；可考虑更小的
$\eta$ 候选、中心差分或只把解析 JVP identity 作为 correctness 验证，但不得改写本次
FAIL，也不得继承 73 个部分任务形成有利子集结论。

机器可读入口：

- `reports/nns_vit_endpoint_e4_intervention_r1_20260918/terminal_receipt.json`；
- `reports/nns_vit_endpoint_e4_intervention_r1_20260918/validation_report.json`；
- `reports/nns_vit_endpoint_e4_intervention_r1_20260918/formal_coverage_audit.json`；
- `reports/nns_vit_endpoint_e4_intervention_r1_20260918/formal_queue.sqlite3`。
