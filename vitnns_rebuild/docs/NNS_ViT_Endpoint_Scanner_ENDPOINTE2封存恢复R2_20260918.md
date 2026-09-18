# ENDPOINT-E2 封存恢复 R2

日期：2026-09-18。4 卡 formal queue 的 800/800 结果及用户取消 replay 的执行修订
均不变。第一次 finalize 在生成 manifest 时把 SQLite 连接存在期间的临时
`formal_queue.sqlite3-wal` 与 `formal_queue.sqlite3-shm` 纳入；连接关闭后临时文件
自动消失，独立验证器以 `MANIFEST_PATH` fail closed。

R1 preliminary terminal 因此不是权威 PASS，原 root 完整保留，不覆盖、不删除。
该问题只影响 handoff manifest，不影响 800 个正式结果、raw atlas、任务 coverage 或
数值内容，也不需要重跑 GPU。

R2 新建独立小型 handoff root：

```text
reports/nns_vit_endpoint_e2_atlas_r2_20260918
```

R2 逐文件绑定 R1 的 formal queue、800 个 formal results、raw arrays、formal logs、
materialization receipt、task manifest、startup snapshot 和 coverage audit；显式排除
已取消的 single replay 与 R1 无效 preliminary handoff 文件。R2 自身 manifest 只枚举
稳定文件。独立验证器随后从 frozen tar、registry、SQLite queue、每个 raw NPY 和所有
hash 重新验证。

机器可读恢复合同为
[`endpoint_e2_atlas_handoff_recovery_r2.json`](../config/nns_vit_endpoint/endpoint_e2_atlas_handoff_recovery_r2.json)。
