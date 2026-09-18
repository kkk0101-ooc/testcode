# NNS-ViT Endpoint Scanner：ENDPOINT-E4 恢复 R3

日期：2026-09-18。R1 和 R2 的 FAIL 均保持不可变。本前瞻性修订在读取任何 R3
outcome 前冻结，机器可读合同为
`config/nns_vit_endpoint/endpoint_e4_intervention_recovery_r3.json`。

## 1. 恢复依据

已发布的 R2 diagnostic 只检查唯一失败任务 `vit_b16__009`，不解释 utility。其唯一
失败成分 `b04.head_00` 满足：

- $a_k=-0.028492351636103308$；
- 解析 JVP dot 为 $-0.02847580080201842$；
- absolute error 为 $1.6550834084888555\times10^{-5}$；
- $\sum_i|g_i(Jx)_i|=11.907848079663662$。

原 R2 门以强消除后的 $|a_k|$ 作为相对误差尺度，不能描述一个含 12,608 个乘积的
条件性归约。诊断中 228 个成分的最大
`absolute_error / sum_abs_products` 为 `6.948755071487438e-06`。

## 2. R3 局部合同

对前 16 个冻结 evaluation order、两个模型、全部 12 blocks 的 main、Q、K 和 joint
routing-value 成分，R3 全部重新执行解析 JVP identity 和 no-op replacement parity。
每个成分必须有限并满足

$$
|⟨g_k,J_{z_k,x}x⟩-a_k|
\le 10^{-5}+10^{-5}\sum_i|g_{k,i}(J_{z_k,x}x)_i|.
$$

该条件尺度阈值在结果前冻结，不因 R3 outcome 调整。它只验证两个 float32 autodiff
路径在由 float64 dot/reduction 记账后的数值一致性，不是 utility claim。

## 3. 计算继承与完整覆盖

- R2 的 31 个已完成局部任务不继承；R3 重跑全部 32 个局部任务，避免只重跑失败样本；
- R2 中 split order 16–399 的 768 个无局部门任务可在逐文件重新验证 payload、result
  canonical digest、模型/图像/目标 identity、finite component coverage、per-image metrics、
  zeroing scope 和 worker determinism 后重绑定；
- R2 中 split order 0–15 的任何结果和 R1 的 73 个结果都不得进入 R3 联合结果；
- R3 必须形成固定 800-task union，来源恰为 `768 inherited + 32 newly run`，无重复、
  无遗漏、无失败。

联合验收计数保持：6,144 个 local checks、249,600 个 main component-reference、9,216 个
secondary component-reference 和 16,896 个 zeroing evaluations。全局 bootstrap、Holm 和
最终 C4a/C4b decision 仍留给 E6。

## 4. 根、资源与出口

R3 唯一新根是 `reports/nns_vit_endpoint_e4_intervention_r3_20260918`。启动前必须验证
R2 FAIL terminal、R2 artifact manifest、诊断文件、E0–E3 和当前远端 commit。新计算沿用
H20 资格、动态队列、最多两次 attempt、确定性 CUDA 与原资源上限；继承计算的原 R2
资源和 R3 新计算资源分别记录并相加。

仅当整个 800-task union 通过独立 source-recomputing validation 时出口为：

```text
ENDPOINT_E4_R3_PASS_EXECUTION_CONDITION_SCALED_JVP_AND_FINITE_REPLACEMENT
```

该 PASS 不是最终 C4a claim decision。ENDPOINT-E5 仍需用户单独明确授权。
