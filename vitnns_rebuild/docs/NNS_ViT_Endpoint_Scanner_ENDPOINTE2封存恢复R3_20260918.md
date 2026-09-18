# ENDPOINT-E2 封存恢复 R3

日期：2026-09-18。R1 handoff 因把 SQLite 临时 WAL/SHM 写入 manifest 而无效；
R2 虽修复该问题，但在 provenance 中给生产 commit 绑定了当前工作树里的测试文件
hash，而该测试文件已在用户取消 replay 的 amendment 中更新。独立验证器从生产
commit 重算 blob 后以 `IMPLEMENTATION_BINDING` 拒绝 R2。

R1、R2 root 均保留为失败证据且不是权威 terminal。两个问题都属于封存与 provenance
错误，不改变 R1 formal queue 已完成的 800/800 任务或 336,800 个 target results。
R3 不重跑 GPU，只把 implementation binding 改为直接从各自发布 commit 的 Git blob
计算 SHA-256：formal producer 文件绑定 `b28ec78af...`，修订/恢复文件绑定 R3 发布
commit。R1 formal 文件继续逐文件绑定。

机器可读合同为
[`endpoint_e2_atlas_handoff_recovery_r3.json`](../config/nns_vit_endpoint/endpoint_e2_atlas_handoff_recovery_r3.json)。
