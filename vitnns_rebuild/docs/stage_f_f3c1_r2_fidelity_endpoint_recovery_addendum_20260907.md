# Stage F F3C1-R2 fidelity endpoint recovery addendum（2026-09-07）

F3C1-R1 在 order-44 的五方法与 IG 数值计算后，于 common fidelity evaluator 的端点断言处 fail closed。原因是 evaluator 将 21 个 intervention 分批 forward，而冻结 clean/baseline scalar 以 batch size 1 计算；ViT kernel 的 batch-size-dependent 浮点舍入使本应相同的端点未达到错误实现要求的位级 `== 0/1`。R1 pending receipt 与 failure event 必须永久保留，不得改写为 PASS。

本恢复只允许：

1. 使用新 roots `evidence/stage_f/f3c1_implementation_r2_20260907/formal/` 与 `reports/stage_f_f3c1_implementation_r2_20260907/`；
2. 将每个 fidelity intervention 作为单图调用同一冻结 scalar evaluation path，使 clean/baseline 端点与冻结 target response 完全一致；
3. 原样保留 14×14 grid、21 fractions、五个共享 hash-derived tie permutations、actual-fraction trapezoid AUC、`0.5*(InsertionAUC-DeletionAUC)`、五方法、目标、IG schedule 和所有阈值；
4. 重新执行 order-44 numerical-only engineering/profile check，并记录完整真实资源。

不得删除或覆盖 v1/R1，不得读取 pilot/formal cohort，不得渲染或人工查看 map，不得产生科学比较、winner、GO/NO-GO，也不得启动 F3C2、F3C3、F3C4 或 F4。R2 PASS 的唯一下一授权仍为 `F3C2_PILOT_ONLY`。
