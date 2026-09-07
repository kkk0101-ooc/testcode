# Stage F F3C1-R3 provenance serialization recovery addendum（2026-09-07）

F3C1-R2 完成 order-44 五方法与 fidelity 数值计算后，在 terminal 发布前 fail closed。原因是当前 Torch 的 `torch.__version__` 为 `TorchVersion` 字符串子类，而严格 canonical JSON writer 只接受内建 JSON 类型。R2 pending receipt 与 failure event 必须永久保留，不得改写为 PASS；R2 未发布 terminal。

本恢复只允许：

1. 使用新 roots `evidence/stage_f/f3c1_implementation_r3_20260907/formal/` 与 `reports/stage_f_f3c1_implementation_r3_20260907/`；
2. 将 Torch、timm 和 CUDA runtime version 显式转换为普通字符串后写入资源 provenance；
3. 原样复用已冻结方法、模型、order-44、preprocessing、IG、fidelity、统计与访问边界；
4. 重新执行唯一一次 order-44 numerical-only engineering/profile check。

不得删除或覆盖 v1/R1/R2，不得读取 pilot/formal cohort，不得渲染或人工查看 map，不得产生科学比较、winner、GO/NO-GO，也不得启动 F3C2、F3C3、F3C4 或 F4。R3 PASS 的唯一下一授权仍为 `F3C2_PILOT_ONLY`。
