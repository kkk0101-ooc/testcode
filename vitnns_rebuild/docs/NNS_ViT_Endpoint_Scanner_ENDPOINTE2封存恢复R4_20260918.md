# ENDPOINT-E2 封存恢复 R4

日期：2026-09-18。R3 已通过原 tar、800 个 queue results、raw NPY 和 Git blob
绑定检查，最后因验证器仍引用旧字段 `total_wall_seconds` 而失败；final report 的冻结
正式生产字段是 `formal_wall_seconds`。

R1–R3 handoff root 均保留且不作为权威 terminal。R4 不重跑任何 GPU task，不改变
正式结果，只将资源 cap 验证绑定到 `formal_wall_seconds`，并在运行前静态核对验证器
读取的 report 字段与 finalizer 输出。机器可读合同为
[`endpoint_e2_atlas_handoff_recovery_r4.json`](../config/nns_vit_endpoint/endpoint_e2_atlas_handoff_recovery_r4.json)。
