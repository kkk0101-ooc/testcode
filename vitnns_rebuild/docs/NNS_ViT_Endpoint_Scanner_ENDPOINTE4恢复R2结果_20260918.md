# NNS-ViT Endpoint Scanner：ENDPOINT-E4 恢复 R2 结果

日期：2026-09-18。权威状态为：

```text
ENDPOINT_E4_R2_FAIL_ANALYTIC_JVP_CONTRACT
```

## 1. 执行结果

R2 在已发布的 commit `248a8254b10f9a6370150888f656771aea594bb8` 上启动，使用
全新结果根和全新 800-task 动态队列，没有继承 R1 的 73 个部分结果。启动门重新验证
了 E0–E3、R1 FAIL、predecessor hashes、远端一致性和 8 张 H20 的资格。

队列最终为 799 completed、1 failed。唯一失败任务 `vit_b16__009` 在允许的两次
attempt 中均触发：

```text
LOCAL_DIRECTION_CONTRACT_FAILED
```

因此 R2 未达到 800/800 coverage，不能形成 E4 PASS。

## 2. 覆盖与边界

独立审计复算了完整 task manifest、queue payload/result digests 和所有已完成输出。R2
完成了 31 个局部任务的 5,916 个解析 JVP component checks，并产生 249,216 个 main、
9,144 个 secondary component-reference evaluations 和 16,728 个 zeroing evaluations。

这些均是失败阶段的部分结果，不做科学解释，不计算 bootstrap、Holm 或最终 C4a
decision。R1 FAIL 及其产物保持不可变；R2 也不得覆盖、续写或筛掉失败样本后变成 PASS。

## 3. 后续状态

```text
NO_NEXT_STAGE_AUTHORIZATION
ENDPOINT_E5_NOT_AUTHORIZED
PROSPECTIVE_E4_RECOVERY_AMENDMENT_REQUIRED
```

任何后续恢复必须先诊断 `vit_b16__009` 的具体 component/error，再在新 outcome 前冻结
新的数值合同并使用新根。若证据显示 $a_k$ 与 JVP 采用了不同数值路径，应修复路径一致性；
若只是接近零量的 float32 reduction 差异，应使用明确的 near-zero 分支，而不能事后删除
该任务或静默放宽 R2。

机器可读入口：

- `reports/nns_vit_endpoint_e4_intervention_r2_20260918/terminal_receipt.json`；
- `reports/nns_vit_endpoint_e4_intervention_r2_20260918/validation_report.json`；
- `reports/nns_vit_endpoint_e4_intervention_r2_20260918/formal_coverage_audit.json`；
- `reports/nns_vit_endpoint_e4_intervention_r2_20260918/formal_queue.sqlite3`。
