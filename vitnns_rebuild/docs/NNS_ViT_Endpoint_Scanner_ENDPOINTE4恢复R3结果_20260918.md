# NNS-ViT Endpoint Scanner：ENDPOINT-E4 恢复 R3 结果

日期：2026-09-18。权威状态为：

```text
ENDPOINT_E4_R3_PASS_EXECUTION_CONDITION_SCALED_JVP_AND_FINITE_REPLACEMENT
```

## 1. 执行与覆盖

R3 在已发布 commit `808d827acdc2f72e6a9dd94bcf4161884eebf6f3` 上启动。它先
独立重验 R2 FAIL 根，再逐文件重算 768 个非局部任务的 payload、canonical digest、
模型/图像/目标 identity、finite component coverage、per-image Spearman、zeroing scope
和 worker determinism；随后在新根上重跑两个模型前 16 个 evaluation order，共 32 个
局部任务。联合索引精确覆盖 800 个唯一 image-model tasks：

- 768 个 `inherited_from_r2_source_revalidated`；
- 32 个 `newly_run_in_r3`；
- 0 missing、0 duplicate、0 invalid。

R1 的 73 个部分结果和 R2 的 31 个局部结果均未继承。R1、R2 的 FAIL receipts 保持
不可变。

## 2. 数值与 finite replacement 结果

R3 新执行的 6,144 个 condition-scaled analytic JVP checks 全部通过；finite-$\eta$
gate count 为 0。最大 absolute error 为 `2.648076753197781e-05`，最大
`error / sum_abs_products` 为 `6.948755071487438e-06`，满足预先冻结的
$10^{-5}+10^{-5}\sum_i|g_i(Jx)_i|$ 门。最大 no-op replacement absolute difference
为 `2.09808349609375e-05`，满足冻结 parity 门。

完整联合结果包含：

- 249,600 个 main component-reference evaluations；
- 9,216 个 secondary component-reference evaluations；
- 16,896 个 zeroing evaluations。

每图描述性 Spearman 汇总已写入 validation report。R3 未运行 bootstrap、Holm 或最终
C4a/C4b claim decision；因此本 PASS 只表示 E4 的执行、correctness、完整覆盖和有限
replacement 数据有效，不表示 RQ3/C4a 已得到 `SUPPORTED`。

## 3. 资源与下一授权

R2 源计算和 R3 新计算合计 `0.6299475847480777` GPU-hours、
`287.32127761713497` wall-seconds；新根封存前 retained bytes 为 `17,168,290`，均低于
冻结上限。

```text
ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST
```

本轮没有启动 E5、E6、Path IG、input fidelity/stability、全局统计或论文 claim decision。

机器可读入口：

- `reports/nns_vit_endpoint_e4_intervention_r3_20260918/terminal_receipt.json`；
- `reports/nns_vit_endpoint_e4_intervention_r3_20260918/validation_report.json`；
- `reports/nns_vit_endpoint_e4_intervention_r3_20260918/artifact_manifest.json`；
- `reports/nns_vit_endpoint_e4_intervention_r3_20260918/combined_result_index.json`；
- `reports/nns_vit_endpoint_e4_intervention_r3_20260918/source_revalidation.json`；
- `reports/nns_vit_endpoint_e4_intervention_r3_20260918/formal_coverage_audit.json`。
