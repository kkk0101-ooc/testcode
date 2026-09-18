# NNS-ViT Endpoint Scanner：ENDPOINT-E4 恢复 R2

日期：2026-09-18。用户已明确要求执行统一路线的 E4 恢复。本修订在读取任何 R2
正式 outcome 前冻结，机器可读合同为
`config/nns_vit_endpoint/endpoint_e4_intervention_recovery_r2.json`。

## 1. 不可变历史与新根

- R1 权威状态保持 `ENDPOINT_E4_FAIL_LOCAL_DIRECTION_CONTRACT`；
- R1 的 73 个已完成结果继续排除于科学解释，R2 不复制、不链接、不继承这些结果；
- R2 从同一冻结的 400 张 evaluation 图像和两个模型创建全新 800-task 动态队列；
- R2 唯一结果根是
  `reports/nns_vit_endpoint_e4_intervention_r2_20260918`，不得写入 R1。

## 2. 恢复理由与局部 correctness 合同

R1 已公开的诊断只用于选择新合同：解析关系

$$
⟨g_k, J_{z_k,x}x⟩ = a_k
$$

在 float64 记账下约为机器精度一致，而冻结的单边 finite-$\eta$ 曲线受真实局部曲率、
截断误差和 float32 forward 分辨率影响。R2 因而删除 finite-$\eta$ 曲线作为 gate，
但不删除、改写或重解释 R1 的失败。

R2 在前 16 个冻结 evaluation order、两个模型、全部 12 blocks 的 main、Q、K 和
joint routing-value 成分上验证解析 JVP identity。两侧都由正式 autodiff 路径产生，再以
float64 dot/reduction 记账；每个成分必须有限并满足

$$
|⟨g_k, J_{z_k,x}x⟩ - a_k| \le 10^{-5}+10^{-4}|a_k|.
$$

这里沿用 E3 的 float32 absolute/relative tolerance，因为解析路径虽以 float64 归约，
输入仍是 float32 autodiff tensors。同一批任务仍须通过 no-op replacement parity。
该门只表示微分实现 correctness，不能
证明有限 replacement utility，也不能消除 R1 的 finite-$\eta$ 失败事实。

## 3. 有限 activation replacement

有限 utility 合同不变：对 400 张共享 evaluation 图像、两个模型、两种 reference，
全量重算 main components；前 64 张保留 Q/K/joint 次级成分和 zeroing。R2 从零计算，
不以 R1 已完成子集作缓存。保留每图 Spearman 及 mandatory descriptive comparators；
不运行 bootstrap、Holm 或最终 C4a/C4b claim decision。

验收计数固定为：

- 800/800 唯一 image-model terminals；
- 6,144 个解析局部 component checks，finite-$\eta$ gate count 为 0；
- 249,600 个 main component-reference evaluations；
- 9,216 个 secondary component-reference evaluations；
- 16,896 个 zeroing evaluations。

## 4. 启动、资源和出口

启动前重新验证 E0–E3、R1 FAIL receipt、全部 predecessor hash，以及本地
HEAD/tracking/remote 一致。沿用 1–8 张合格 H20、全局动态队列、原子 claim、最多两次
attempt、确定性 CUDA、内存/磁盘/时长 cap；任何遗漏、重复、nonfinite、source drift、
局部门或资源门失败均保留新根并 fail closed。

仅当上述全部通过时出口为：

```text
ENDPOINT_E4_R2_PASS_EXECUTION_ANALYTIC_JVP_AND_FINITE_REPLACEMENT
```

该 PASS 仍只是 E4 执行/correctness/coverage 有效，不是最终 C4a 支持结论。下一阶段
仍须用户单独明确授权 `ENDPOINT-E5`。
