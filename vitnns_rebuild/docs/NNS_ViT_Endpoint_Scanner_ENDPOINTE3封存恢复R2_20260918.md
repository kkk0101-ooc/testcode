# NNS-ViT Endpoint Scanner：ENDPOINT-E3 封存恢复 R2

日期：2026-09-18。E3 R1 已完成 800/800 个正式 GPU 任务并写入 immutable terminal。
首次独立验证在 `RAW_RELATIVE_L2` 停止：raw array digest、component 求和、reference、
max-absolute 和正式 elementwise closure 均已通过，但 NumPy 与 Torch 对 float32 大数组
的 L2 norm 使用不同归约树，首个 derived scalar 差 `1.4043e-12`。

对全部 3,072 个保留 raw cut 做只读诊断后，最大绝对差为 `3.2593e-11`，最大相对差
为 `1.4328e-5`。因此 R2 仅将“已记录 Torch relative-L2 与独立 NumPy relative-L2
相等”的审计比较注册为 `abs_tol=1e-10, rel_tol=2e-5`。正式科学闭合容差仍为
`atol=1e-5, rtol=1e-4`，并继续由独立 raw component sum 直接检查；不改变任何样本、
cut、结果或科学出口。

机器修订为 [handoff recovery](../config/nns_vit_endpoint/endpoint_e3_closure_handoff_recovery_r2.json)。
R1 结果根和 terminal 不覆盖，不重跑 GPU；R2 只在新 root 中封存独立验证结果。
